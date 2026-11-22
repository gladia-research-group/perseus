//! `pc-check tfhe-digit --case FILE [--reps 5] [--lat-group 5] [--level 44] [--noise]`: the digit decomposition (perseus
//! impl/tfhe_digits.py: a large-precision sign, an exact word) on the device, from a real split: a fresh ciphertext of
//! the case's values (other slots random) runs the StC-first route's input half, its exact phase is read on the host
//! (the reference: the values as the switch sees them), and the case's native program runs on the device's switch
//! (`native_load_shifted`: the windows of the phase) and native ops; the stored rows are decrypted (s_br) against the
//! reference: every sign decided at the program's resolution, every digit of every word. `--noise`: the windows' read
//! errors (`native_extract_shifted`, s_lwe) against the exact windows, per shift. Latency (synced, median of
//! `--reps`): the load (switch + windows' bootstraps), the rounds, the store's lookup, the repack + finish back to CKKS
//! (the sign's outputs decrypted from the finished ciphertext too); and the current switch (one sign LUT, the same
//! values) for comparison.
//!
//! Case (python -m perseus.impl.tfhe_digits --case FILE): {"name", "kind": "sign" | "word", "bits" | "nd", "prog"
//! (Prog JSON), "m" (the loaded values, native units: slot values at the split's cf 2)}.
use crate::check_bts::Tally;
use crate::check_tfhe::{arg, bytemuck_i32, rescale_top, switch_params};
use pc_core::encode::{Complex, Encoder, Payload};
use pc_core::keys::SecretKey;
use pc_core::modmul::Barrett64;
use pc_core::params::{Chain, ChainSpec};
use pc_cuda::Device;
use pc_gpu::bts::staged::{BtsCall, CallOpts, Phase, StagedRef};
use pc_gpu::bts::stc_first::{StcFirstPlan, StcFirstSetup};
use pc_gpu::bts::{BtsCx, BtsSetup, BtsWork};
use pc_gpu::decrypt::{Decryptor, Which};
use pc_gpu::encrypt::Encryptor;
use pc_gpu::keygen::{test_prng, KeyGen};
use pc_gpu::keys::{DevSecret, KeySet, KeySetSpec, KskLayout, RotSpec};
use pc_gpu::ks::KsPlans;
use pc_gpu::ntt::{Parts, Span};
use pc_gpu::ops::{Ops, OpsCfg};
use pc_gpu::poly::CtBuf;
use pc_gpu::{Context, Ct, Executor, MemCfg, MemPlan, PoolClass};
use pc_shadow::{BtsConfig, CtMeta, Quirks, Route, Shadow, ShadowConfig};
use pc_tfhe::lwe::Lwe;
use pc_tfhe::modq::{centered, Ntt};
use pc_tfhe::rlwe::{Rlwe, RnsRlwe};
use pc_tfhe::switch::Switcher;
use serde_json::Value;
use std::collections::BTreeSet;
use std::ffi::c_void;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

type W = u32;
type M = Barrett64;
type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn cmd(args: &[String]) -> ExitCode {
    match run(args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("pc-check tfhe-digit: {e}");
            ExitCode::FAILURE
        }
    }
}

#[derive(Clone, Debug)]
enum St {
    Load { lut: usize, k: usize, dst: Vec<usize>, shifts: Vec<u32> },
    Lin { dst: usize, a: usize, b: usize, off: usize, rows: usize, ca: u64, cb: u64, c: u64 },
    Pbs { src: usize, rows: usize, lut: usize, dst: Vec<usize> },
    Store { src: usize, rows: usize, lut: Option<usize> },
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn run(args: &[String]) -> Res<bool> {
    let path = arg(args, "--case").ok_or("--case FILE")?;
    let case: Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    let reps: usize = arg(args, "--reps").map(|s| s.parse()).transpose()?.unwrap_or(5).max(1);
    let level: u32 = arg(args, "--level").map(|s| s.parse()).transpose()?.unwrap_or(44);
    let name = case["name"].as_str().unwrap_or("digit").to_string();
    let kind = case["kind"].as_str().ok_or("case kind")?.to_string();
    let m_in: Vec<f64> = case["m"].as_array().ok_or("case m")?.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect();
    let b = m_in.len();
    let pr = &case["prog"];
    let cf = 2u32;
    let t0 = Instant::now();

    // ── the CKKS side (as pc-check tfhe) ──
    let chain = Chain::generate(&ChainSpec::n32())?;
    let n = chain.n;
    let slots = n / 2;
    if b == 0 || b > slots {
        return Err(format!("{b} values (1..={slots})").into());
    }
    let mut setup = BtsSetup::n32_shipped();
    setup.raise_drops = vec![0];
    setup.k_max = 1;
    let mut sf = StcFirstSetup::n32_shipped();
    sf.cache_slots = 2;
    let plan = StcFirstPlan::new(&chain, &setup, &sf)?;
    let mut geoms = Vec::new();
    for v in &plan.dense.variants {
        geoms.extend(v.cts.iter().chain(&v.stc).map(|h| h.geom));
    }
    geoms.extend(plan.input.iter().map(|h| h.geom));
    geoms.extend(plan.sources.iter().flatten().map(|s| s.geom));
    let steps_rot: BTreeSet<i64> = crate::bts_stc_first::rot_steps(&geoms, n);
    let spec = KeySetSpec { relin: true, conj: true, rots: steps_rot.iter().map(|&s| RotSpec { step: s, q_top: None }).collect(), encaps: true, lean: None };

    // ── the program: tables, steps, buffers ──
    let mut params = switch_params(args, n)?;
    params.chain = true;
    let (nb, tvl) = (params.n_br, params.tv_len());
    let row = nb + 1;
    let nbufs = 6usize; // 5 + scratch
    let luts = pr["luts"].as_object().ok_or("prog luts")?;
    let mut lut_ix = std::collections::BTreeMap::new();
    let mut lut_k = std::collections::BTreeMap::new();
    let mut lut_warp = std::collections::BTreeMap::new();
    let mut tables: Vec<(Vec<Vec<f64>>, bool)> = Vec::new();
    for (ln, l) in luts {
        let ts: Vec<Vec<f64>> = serde_json::from_value(l["tables"].clone())?;
        lut_ix.insert(ln.clone(), tables.len());
        lut_k.insert(ln.clone(), ts.len());
        lut_warp.insert(ln.clone(), l["warp"].as_bool().unwrap_or(false));
        tables.push((ts, l["warp"].as_bool().unwrap_or(false)));
    }
    let mut prog: Vec<St> = Vec::new();
    let mut idx_words: Vec<u32> = Vec::new();
    let mut rows_max = b;
    let mut load_shifts = 1usize;
    for st in pr["steps"].as_array().ok_or("prog steps")? {
        let u = |k: &str| -> Res<usize> { Ok(st[k].as_u64().ok_or_else(|| format!("step without {k}"))? as usize) };
        let lut = |st: &Value| -> Res<usize> { Ok(lut_ix[st["lut"].as_str().ok_or("lut")?]) };
        prog.push(match st["op"].as_str().unwrap_or("") {
            "load" => {
                let dst: Vec<usize> = match &st["dst"] {
                    Value::Array(a) => a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect(),
                    x => vec![x.as_u64().ok_or("load dst")? as usize],
                };
                let shifts: Vec<u32> = serde_json::from_value(st.get("shifts").cloned().unwrap_or(Value::Array(vec![])))?;
                load_shifts = shifts.len().max(1);
                rows_max = rows_max.max(b * load_shifts);
                St::Load { lut: lut(st)?, k: lut_k[st["lut"].as_str().unwrap()], dst, shifts }
            }
            "lin" => {
                let idx: Vec<i64> = serde_json::from_value(st["idx"].clone())?;
                let r = idx.len() / 2;
                rows_max = rows_max.max(r);
                let off = idx_words.len();
                idx_words.extend(bytemuck_i32(&idx.iter().map(|&i| i as i32).collect::<Vec<_>>()));
                St::Lin { dst: u("dst")?, a: u("a")?, b: u("b")?, off, rows: r, ca: 0, cb: 0, c: 0 }
            }
            "pbs" => {
                let dst: Vec<usize> = st["dst"].as_array().ok_or("pbs dst")?.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect();
                rows_max = rows_max.max(u("rows")?);
                St::Pbs { src: u("src")?, rows: u("rows")?, lut: lut(st)?, dst }
            }
            "store" => St::Store { src: u("src")?, rows: u("rows")?, lut: st["lut"].as_str().map(|l| lut_ix[l]) },
            o => return Err(format!("step {o}: not in a digit program").into()),
        });
    }
    // the lin coefficients mod p0 (set below, once p0 is known)
    let lin_raw: Vec<(i64, i64, f64)> = pr["steps"].as_array().unwrap().iter().filter(|s| s["op"] == "lin").map(|s| (s["ca"].as_i64().unwrap_or(0), s["cb"].as_i64().unwrap_or(0), s["c"].as_f64().unwrap_or(0.0))).collect();
    let Some(St::Store { rows: out_rows, .. }) = prog.last().cloned() else { return Err("a program stores last".into()) };

    let dev = Device::open()?;
    let blt = pc_cuda::blas::Lt::new(&dev)?;
    let cfg = MemCfg { k_max: 1, pool: vec![PoolClass { k: 1, count: 1 }], events: 16, scratch_cts: 2 };
    let mut mp = MemPlan::build::<W, M>(&chain, cfg)?;
    let lay = mp.lay;
    let shapes: Vec<_> = spec.keys(&chain)?.into_iter().map(|(_, s)| s).collect();
    mp.reserve("keys", spec.bytes::<W>(&chain, KskLayout::Dense)?);
    mp.reserve("keygen", KeyGen::<W>::bytes(&lay, &shapes));
    mp.reserve("secret", DevSecret::<W>::bytes(&lay));
    mp.reserve("encrypt", Encryptor::<W>::bytes(&lay, false));
    mp.reserve("decrypt", Decryptor::<W>::bytes(&lay, chain.level_unit, 1));
    mp.reserve("ks.plans", KsPlans::bytes::<W>(&chain, 1)?);
    mp.reserve("ks.regen_a", pc_gpu::ks::KsWork::<W>::regen_bytes(&lay, chain.dnum()));
    let ocfg = OpsCfg::default();
    Ops::<W>::reserve(&mut mp, &ocfg);
    let (req, wshape) = plan.mem::<W>()?;
    req.reserve(&mut mp);
    mp.reserve("bts.work", BtsWork::<W>::bytes(&lay, wshape));
    mp.reserve("check.cts", 3 * pc_cuda::Arena::bytes_for::<W>(CtBuf::<pc_cuda::Compute, W>::words(&lay, 1)));
    let cap = rows_max.min(1 << 15);
    mp.reserve("tfhe", pc_gpu::tfhe::TfheSwitch::bytes(n, &params, cap));
    let ntab: usize = tables.iter().map(|t| t.0.len()).sum::<usize>() + 2;
    mp.reserve("digit", 4 * (nbufs * rows_max * row + idx_words.len() + 2 * rows_max + 2 * ntab * 2 * tvl + rows_max * (params.n_lwe + 1)) + (8 << 20));
    let arena = mp.allocate(&dev)?;
    let mut ex = Executor::<W>::new(&dev, &arena, &mp)?;
    ex.ks.carve_regen(&arena, &lay)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &mp)?;
    let kplans = KsPlans::new(&arena, &ex.compute, &ctx, 1)?;
    let mut ops = Ops::new(&arena, &dev, &ex.compute, &ctx, ocfg)?;
    let mut work = BtsWork::<W>::new(&arena, lay, wshape)?;
    let mut keys = KeySet::<W>::carve(&arena, &chain, &spec, KskLayout::Dense)?;
    let seed: u32 = arg(args, "--seed").map(|s| s.parse()).transpose()?.unwrap_or(0);
    let mut prng = test_prng(0x5d + 0x100 * seed);
    let sk = SecretKey::sample(&mut prng, n)?;
    let mut sec = DevSecret::<W>::carve(&arena, lay)?;
    sec.load(&ctx, &ex.compute, &sk)?;
    let mut kg = KeyGen::<W>::new(&arena, &ex.compute, &chain, &shapes)?;
    kg.generate_set(&ctx, &mut ex, &sec, &mut keys, &mut prng)?;
    let mut encr = Encryptor::<W>::new(&arena, &ex.compute, lay, false, &mut prng)?;
    let mut dec = Decryptor::<W>::new(&arena, &ex.compute, lay, chain.level_unit, 1)?;
    dec.load_secret(&ex.compute, &sec)?;
    let route = plan.build(&arena, &dev, &ex.compute, &ctx, &keys, ex.ks.stage_ptrs())?;
    let mut x = Ct::new(CtMeta::new(0, 1), CtBuf::<pc_cuda::Compute, W>::new(&arena, "check.cts", lay, 1)?);
    let mut z = Ct::new(CtMeta::new(0, 1), CtBuf::<pc_cuda::Compute, W>::new(&arena, "check.cts", lay, 1)?);
    let s: Vec<i64> = sk.s.iter().map(|&v| v as i64).collect();
    let (p0, p1) = (chain.q[0], chain.q[1]);
    let tk = Instant::now();
    let (sw, s_lwe, s_br) = Switcher::generate_keep(params.clone(), &s, p0, p1, &mut prng)?;
    println!("[pc-check] tfhe-digit: setup {:.1} s, switch keys {:.1} s (n_lwe {}, lat_group {}, nu {}); case {name}: {kind}, {b} values, {} windows, {} steps", t0.elapsed().as_secs_f64(), tk.elapsed().as_secs_f64(), params.n_lwe, params.lat_group, params.nu, load_shifts, prog.len());
    let mut ts = pc_gpu::tfhe::TfheSwitch::new(&arena, &dev, &ex.compute, &sw, cap)?;
    ex.compute.sync()?;
    let al = |k: usize| arena.alloc::<pc_cuda::Compute, u32>("digit", k);
    let mut bufs: Vec<pc_cuda::DBuf<'_, pc_cuda::Compute, u32>> = (0..nbufs).map(|_| al(rows_max * row)).collect::<std::result::Result<_, _>>()?;
    let mut d_idx = al(idx_words.len().max(2))?;
    let mut d_pos = al(2 * rows_max)?;
    let mut d_tv = al(ntab * 2 * tvl)?;
    let mut d_ext = al(rows_max * (params.n_lwe + 1))?;
    arena.freeze();
    // the TFHE side on its own stream (the CKKS side's executor is borrowed by the route's calls); synced between
    let ts_s = pc_cuda::Stream::<pc_cuda::Compute>::new(&dev)?;
    let s0 = &ts_s;
    d_idx.upload(s0, &idx_words)?;
    // tables: native (q x / 4, the upper half negated); warp tables a second time in the finish's encoding
    let q = params.q_br;
    let mut tvw: Vec<u32> = Vec::new();
    let mut warp_ix = std::collections::BTreeMap::new();
    let enc_tab = |ts: &Vec<Vec<f64>>, warp: bool| -> Vec<u32> {
        let tvs: Vec<Vec<u64>> = ts
            .iter()
            .map(|t| {
                let h = t.len();
                t.iter()
                    .enumerate()
                    .map(|(j, &v)| {
                        let y = if warp { q as f64 * pc_tfhe::pbs::warp(v, params.out_cf) } else { q as f64 * v / 4.0 };
                        let y = y.round() as i64;
                        (if j < h / 2 { y } else { -y }).rem_euclid(q as i64) as u64
                    })
                    .collect()
            })
            .collect();
        let tv = if tvs.len() == 1 { tvs[0].clone() } else { pc_tfhe::pbs::interleave(&tvs) };
        sw.bsk.rns.residues(&tv).iter().flat_map(|r| r.iter().map(|&x| x as u32)).collect()
    };
    for (i, (ts_, warp)) in tables.iter().enumerate() {
        tvw.extend(enc_tab(ts_, false));
        if *warp {
            warp_ix.insert(i, 0);
        }
    }
    let mut ntv = tables.len();
    for (i, slot) in warp_ix.iter_mut() {
        tvw.extend(enc_tab(&tables[*i].0, true));
        *slot = ntv;
        ntv += 1;
    }
    // the current switch's sign LUT (pbs::sign01: a constant test vector + offset) for the comparison
    let (sg_tv, sg_off) = pc_tfhe::pbs::sign01(tvl, sw.delta_br());
    let sg_ix = ntv;
    tvw.extend(pc_gpu::tfhe::tv_residues(&sw.bsk, &sg_tv));
    d_tv.upload(s0, &tvw)?;
    let tvp = |k: usize| (d_tv.vptr() as *const u32).wrapping_add(k * 2 * tvl) as *const c_void;
    let modp = |x: i64| x.rem_euclid(p0 as i64) as u64;
    let mut li = 0;
    for st in prog.iter_mut() {
        if let St::Lin { ca, cb, c, .. } = st {
            let (a, bb, cc) = lin_raw[li];
            *ca = modp(a);
            *cb = modp(bb);
            *c = modp((cc * p0 as f64 / 4.0).round() as i64);
            li += 1;
        }
    }

    // ── values: the case's at spread slots, the others random (they must not leak into the windows) ──
    let mut vals: Vec<Complex> = (0..slots).map(|_| Complex::new(prng.uniform_f64() * 1.8 - 0.9, 0.0)).collect();
    let chosen: Vec<usize> = (0..b).map(|i| (i * (slots / b) + 3) % slots).collect();
    for (i, &c) in chosen.iter().enumerate() {
        vals[c] = Complex::new(m_in[i], 0.0);
    }
    let bits = (slots as u32).trailing_zeros();
    let coef_of = |i: usize| ((i + slots - 1) % slots).reverse_bits() >> (usize::BITS - bits);
    let pos: Vec<u32> = chosen.iter().map(|&i| coef_of(i) as u32).collect();
    // the store's set: the first out_rows slots of another spread (one value per row)
    let out_slots: Vec<usize> = (0..out_rows).map(|i| (i * (slots / out_rows.max(1)) + 5) % slots).collect();
    let opos: Vec<u32> = out_slots.iter().map(|&i| coef_of(i) as u32).collect();
    d_pos.upload_at(s0, 0, &pos)?;
    d_pos.upload_at(s0, rows_max, &opos)?;
    let posp = d_pos.vptr() as *const c_void;
    let oposp = (d_pos.vptr() as *const u32).wrapping_add(rows_max) as *const c_void;
    let top = (chain.q.len() - 1 - level as usize) as i32;
    let sf_ = ctx.scale.sf(top)?;
    let mut meta = CtMeta::new(level, 1).with_nf(sf_);
    meta.slots = slots as u32;
    let enc = Encoder::new(&chain);
    let sh = Shadow::new(ShadowConfig { chain: ctx.shape, bts: BtsConfig::n32_shipped(), quirks: Quirks::none() }, Some(Arc::new(ctx.scale.clone())));
    let r = Route { cf: Some(cf), stc_first: true, real: false, ..Route::dense() };
    let nq = ctx.shape.n_q;

    // a fresh split of the values into `x` (the load consumes it); evaluates to the finish's call
    macro_rules! fresh {
        () => {{
            let msg = enc.encode_slots(&vals, slots, top as usize, Payload::Complex, sf_)?;
            encr.encrypt_sk(&ctx, &ex.compute, &sec, &[msg], meta, &mut x, &mut prng)?;
            let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
            let mut inp = BtsCall::begin(StagedRef::StcFirst(&route), &cx, std::slice::from_ref(&x), &r, CallOpts { prescaled: false, own_cf: true })?;
            inp.stc_first_input(&mut cx, std::slice::from_mut(&mut x))?.ok_or("input below the StC-first entry")?;
            cx.ex.compute.sync()?;
            inp.finish_call(0)?
        }};
    }
    let ntts = [Ntt::new(p0, n)?, Ntt::new(p1, n)?];
    let q0 = p0 as i128 * p1 as i128;
    // the exact phase of the split in `x` on the host (a copy: coefficient form, rescaled to q0): (ct, phases of `pos`)
    macro_rules! host_phase {
        () => {{
            let stop = (nq - 1 - x.meta.level) as usize;
            pc_program::replay::copy_ct(&ctx, &ex.compute, &mut z, &x)?;
            pc_gpu::ntt::inverse(&ctx, &ex.compute, &mut z.buf, Span::Q { top: stop }, Parts::Both)?;
            let rows: Vec<usize> = (0..=stop).collect();
            let mut parts = Vec::new();
            for p in 0..2 {
                let w = z.buf.download_rows(&ex.compute, 0, p, &rows)?;
                let mut limbs: Vec<Vec<u64>> = w.chunks(n).map(|c| c.iter().map(|&v| v as u64).collect()).collect();
                let mut pr_ = chain.q[..=stop].to_vec();
                while pr_.len() > 2 {
                    rescale_top(&mut limbs, &mut pr_);
                }
                parts.push(limbs);
            }
            let ct = RnsRlwe { limbs: (0..2).map(|l| Rlwe { c0: parts[0][l].clone(), c1: parts[1][l].clone() }).collect() };
            let ph = ct.phase(&s, &ntts);
            let exact: Vec<i128> = pos.iter().map(|&j| ph[j as usize].rem_euclid(q0)).collect();
            (ct, exact)
        }};
    }
    let to_m = |e: i128| 4.0 * (if e >= q0 / 2 { e - q0 } else { e }) as f64 / q0 as f64;
    let mut t = Tally::default();

    // ── --noise: the windows' read errors (s_lwe) per shift, against the exact windows ──
    let shifts_all: Vec<u32> = prog.iter().find_map(|s| if let St::Load { shifts, .. } = s { Some(shifts.clone()) } else { None }).unwrap_or_default();
    if args.iter().any(|a| a == "--noise") && !shifts_all.is_empty() {
        let _ = fresh!();
        let (ct, exact) = host_phase!();
        let cols = params.n_lwe + 1;
        let win = |e: i128, k: u32| -> f64 {
            let w = (e * (1i128 << k)).rem_euclid(q0);
            4.0 * (if w >= q0 / 2 { w - q0 } else { w }) as f64 / q0 as f64
        };
        ts.native_extract_shifted(&ctx, s0, &blt.inner, &mut x, posp, b, &shifts_all, d_ext.vptr_mut(), false)?;
        let raw = d_ext.download_range(s0, 0, shifts_all.len() * b * cols)?;
        // the host's windows of the same split: bit-exact with the device's
        let host: Vec<Lwe> = shifts_all.iter().flat_map(|&k| sw.lwes_shifted(&ct, &pos.iter().map(|&p| p as usize).collect::<Vec<_>>(), k)).collect();
        let mut bad = 0;
        let (mut ss_all, mut cnt_all) = (0f64, 0usize);
        for (wi, &k) in shifts_all.iter().enumerate() {
            let (mut ss, mut worst) = (0f64, 0f64);
            for q_ in 0..b {
                let rr = (wi * b + q_) * cols;
                let l = Lwe { a: raw[rr..rr + params.n_lwe].iter().map(|&v| v as u64).collect(), b: raw[rr + params.n_lwe] as u64 };
                bad += (l != host[wi * b + q_]) as usize;
                let got = 4.0 * centered(l.phase(&s_lwe, p0), p0) as f64 / p0 as f64;
                let e = (got - win(exact[q_], k) + 2.0).rem_euclid(4.0) - 2.0;
                ss += e * e;
                worst = worst.max(e.abs());
            }
            ss_all += ss;
            cnt_all += b;
            println!("[pc-check] tfhe-digit: window shift {k:2}: read error rms {:.3e} max {worst:.3e} (native units; {:.1} sigma of the max)", (ss / b as f64).sqrt(), worst / (ss / b as f64).sqrt().max(1e-12));
        }
        t.add("tfhe-digit: device windows == host windows (pc-tfhe lwes_shifted)", bad, shifts_all.len() * b);
        println!("[pc-check] tfhe-digit: windows' read sigma {:.3e} native over {cnt_all} reads (the model: 5.3e-3)", (ss_all / cnt_all as f64).sqrt());
    }

    // ── the program on the device ──
    let ptr = |bufs: &Vec<pc_cuda::DBuf<'_, pc_cuda::Compute, u32>>, i: usize| bufs[i].vptr() as *mut c_void;
    let scratch = nbufs - 1;
    let idxp = |off: usize| (d_idx.vptr() as *const u32).wrapping_add(off) as *const c_void;
    let decrypt = |w: &[u32], cnt: usize| -> Vec<f64> {
        (0..cnt)
            .map(|r| {
                let l = Lwe { a: w[r * row..r * row + nb].iter().map(|&x| x as u64).collect(), b: w[r * row + nb] as u64 };
                centered(l.phase(&s_br, p0), p0) as f64 * 4.0 / p0 as f64
            })
            .collect()
    };
    let (mut t_load, mut t_body, mut t_store, mut t_fin) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut got: Vec<f64> = Vec::new();
    let mut fin_vals: Vec<f64> = Vec::new();
    let mut m_exact: Vec<f64> = Vec::new();
    for rep in 0..=reps {
        let mut fc = fresh!();
        if rep == 0 {
            let (_, exact) = host_phase!();
            m_exact = exact.iter().map(|&e| to_m(e)).collect();
            let dev_in = m_exact.iter().zip(&m_in).map(|(a, b)| (a - b).abs()).fold(0f64, f64::max);
            println!("[pc-check] tfhe-digit: split at level {}; |phase - value| max {dev_in:.2e} (the CKKS noise the windows read)", x.meta.level);
        }
        let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
        cx.ex.compute.sync()?;
        let tl = Instant::now();
        let mut tb = tl;
        let mut store = None;
        for st in &prog {
            match st {
                St::Load { lut, k, dst, shifts } => {
                    let dsts: Vec<*mut c_void> = dst.iter().map(|&d| ptr(&bufs, d)).collect();
                    if shifts.is_empty() {
                        ts.native_load_k(&ctx, s0, &blt.inner, &mut x, posp, b, tvp(*lut), &vec![0; *k], &dsts, false)?;
                    } else {
                        ts.native_load_shifted(&ctx, s0, &blt.inner, &mut x, posp, b, shifts, tvp(*lut), &vec![0; *k], &dsts, false)?;
                    }
                    s0.sync()?;
                    if rep > 0 { t_load.push(tl.elapsed().as_secs_f64() * 1e3); }
                    tb = Instant::now();
                }
                St::Lin { dst, a, b: bb, off, rows, ca, cb, c } => ts.native_lin_c(&ctx, s0, ptr(&bufs, *dst), ptr(&bufs, *a), ptr(&bufs, *bb), idxp(*off), *rows, *ca, *cb, *c)?,
                St::Pbs { src, rows, lut, dst } => {
                    let dsts: Vec<*mut c_void> = dst.iter().map(|&d| ptr(&bufs, d)).collect();
                    ts.native_pbs(&ctx, s0, &blt.inner, ptr(&bufs, *src), *rows, tvp(*lut), &vec![0; dst.len()], &dsts)?;
                }
                St::Store { src, rows, lut } => store = Some((*src, *rows, *lut)),
            }
        }
        s0.sync()?;
        if rep > 0 { t_body.push(tb.elapsed().as_secs_f64() * 1e3); }
        let (ssrc, srows, slut) = store.unwrap();
        // the store's table read as native values (decrypted below), then in the finish's encoding into the split
        let tst = Instant::now();
        if let Some(l) = slut {
            ts.native_pbs(&ctx, s0, &blt.inner, ptr(&bufs, ssrc), srows, tvp(l), &[0], &[ptr(&bufs, scratch)])?;
        }
        s0.sync()?;
        if rep > 0 { t_store.push(tst.elapsed().as_secs_f64() * 1e3); }
        if rep == 0 {
            let ob = if slut.is_some() { scratch } else { ssrc };
            got = decrypt(&bufs[ob].download_range(s0, 0, srows * row)?, srows);
        }
        if let Some(l) = slut {
            let tf = Instant::now();
            ts.native_pbs(&ctx, s0, &blt.inner, ptr(&bufs, ssrc), srows, tvp(warp_ix.get(&l).copied().unwrap_or(l)), &[0], &[ptr(&bufs, scratch)])?;
            ts.native_store(&ctx, s0, &blt.inner, ptr(&bufs, scratch), oposp, srows, 1, &mut x)?;
            s0.sync()?;
            fc.at_split(&cx, std::slice::from_ref(&x))?;
            fc.run_to(&mut cx, std::slice::from_mut(&mut x), Phase::Done)?;
            cx.ops.realize(&ctx, cx.ex, &sh, std::slice::from_mut(&mut x))?;
            cx.ex.compute.sync()?;
            if rep > 0 { t_fin.push(tf.elapsed().as_secs_f64() * 1e3); }
            if rep == 0 {
                let g = dec.decrypt_slots(&ctx, &cx.ex.compute, &enc, &x, Payload::Complex, Which::Dense)?;
                let gain = (1u64 << (params.out_cf - cf)) as f64;
                fin_vals = out_slots.iter().map(|&i| g[0][i].re * gain).collect();
            }
        }
    }

    // ── the reference ──
    let (mut wrong, mut decided, mut fin_wrong) = (0usize, 0usize, 0usize);
    let mut lines = Vec::new();
    if kind == "sign" {
        let bits_res: f64 = case["resolution"].as_f64().ok_or("sign case: resolution")?;
        let span: f64 = case["span"].as_f64().unwrap_or(1.75);
        let half = case["half"].as_bool().unwrap_or(false);
        let v = if half { 0.5 } else { 1.0 };
        for i in 0..b {
            let m = m_exact[i];
            if m.abs() / span < bits_res {
                continue;
            }
            decided += 1;
            let want = if m >= 0.0 { v } else { -v };
            if (got[i] - want).abs() > 0.1 * v {
                wrong += 1;
                if lines.len() < 8 {
                    lines.push(format!("value {m:+.6e}: got {:.4} want {want}", got[i]));
                }
            }
            if !fin_vals.is_empty() && (fin_vals[i] - want).abs() > 0.05 {
                fin_wrong += 1;
            }
        }
        t.add(&format!("tfhe-digit: {name}: signs exact at |m| >= 2^{:.2} of the bound (native outputs)", bits_res.log2()), wrong, decided.max(1));
        if !fin_vals.is_empty() {
            t.add(&format!("tfhe-digit: {name}: signs through the finish (CKKS slots, err < 0.05)"), fin_wrong, decided.max(1));
        }
    } else {
        let nd = case["nd"].as_u64().ok_or("word case: nd")? as usize;
        let sc = 2f64.powi(2 * nd as i32 - 1);
        for i in 0..b {
            let y = m_exact[i] * sc;
            let ws: Vec<i128> = if (y - y.floor() - 0.5).abs() < 0.1 { vec![y.floor() as i128, y.floor() as i128 + 1] } else { vec![y.round() as i128] };
            let digs: Vec<i64> = (0..nd).map(|d| (got[i * nd + d] * 4.0).round() as i64).collect();
            let ok = ws.iter().any(|&w| {
                let u = w.rem_euclid(1i128 << (2 * nd));
                (0..nd).all(|d| ((u >> (2 * d)) & 3) as i64 == digs[d])
            });
            decided += 1;
            if !ok {
                wrong += 1;
                if lines.len() < 8 {
                    lines.push(format!("value {:+.9e} (x 2^{} = {y:.3}): digits {digs:?}", m_exact[i], 2 * nd - 1));
                }
            }
            if !fin_vals.is_empty() {
                let fd: Vec<i64> = (0..nd).map(|d| (fin_vals[i * nd + d] * 4.0).round() as i64).collect();
                fin_wrong += (fd != digs) as usize;
            }
        }
        t.add(&format!("tfhe-digit: {name}: words exact ({nd} digits, every digit)"), wrong, b);
        if !fin_vals.is_empty() {
            t.add(&format!("tfhe-digit: {name}: digits through the finish == native digits"), fin_wrong, b);
        }
    }
    for l in &lines {
        println!("[pc-check] tfhe-digit:   {l}");
    }

    // ── the current switch: one sign LUT on the same values (load + PBS + repack + finish) ──
    let mut t_sw = Vec::new();
    for _ in 0..reps.max(1) {
        let mut fc = fresh!();
        let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
        cx.ex.compute.sync()?;
        let tw = Instant::now();
        ts.apply_dev(&ctx, s0, &blt.inner, &mut x, posp, b, 1, tvp(sg_ix), sg_off, false)?;
        s0.sync()?;
        fc.at_split(&cx, std::slice::from_ref(&x))?;
        fc.run_to(&mut cx, std::slice::from_mut(&mut x), Phase::Done)?;
        cx.ops.realize(&ctx, cx.ex, &sh, std::slice::from_mut(&mut x))?;
        cx.ex.compute.sync()?;
        t_sw.push(tw.elapsed().as_secs_f64() * 1e3);
    }
    let (ml, mb, ms_, mf, msw) = (median(&mut t_load), median(&mut t_body), median(&mut t_store), if t_fin.is_empty() { 0.0 } else { median(&mut t_fin) }, median(&mut t_sw));
    println!(
        "[pc-check] tfhe-digit: {name}: {kind} of {b} values ({decided} checked, {wrong} wrong): load {ml:.3} ms ({} windows), rounds {mb:.3} ms, store lookup {ms_:.3} ms; with repack + finish {:.3} ms (store table + repack + finish {mf:.3}); current switch (sign LUT + repack + finish) {msw:.3} ms (median of {reps})",
        load_shifts,
        ml + mb + mf
    );
    println!("[pc-check] tfhe-digit: {} gates, {} failing", t.gates, t.bad);
    Ok(t.bad == 0)
}
