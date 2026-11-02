//! CKKS -> TFHE -> CKKS through the device's StC-first route with the host switch (pc-tfhe) in between.
//!
//!   pc-check tfhe [--switch 64] [--level 44] [--cf 2] [--out-cf 8] [--device]
//!
//! A fresh ciphertext of random real values runs the route's input half at its own CF (values at the split's
//! coefficients, phase m q0 / 2^cf), is brought to the host (inverse NTT, exact rescales by p3 and p2: mod q0), checked
//! against its slots (slot i at coefficient brev((i - 1) mod N/2)), switched through the sign LUT on `--switch` spread coefficients, lifted back (x p2 p3, the
//! split's four primes), and finished on the device (ModRaise -> CtS -> EvalMod). Gates: the host decryption equals the
//! device's; the split's phase matches the slots; the finished slots are (1 + sign) / 2 at the switched slots and 0 elsewhere.
use crate::check_bts::Tally;
use pc_core::encode::{Complex, Encoder, Payload};
use pc_core::keys::SecretKey;
use pc_core::modmul::Barrett64;
use pc_core::params::{Chain, ChainSpec};
use pc_cuda::Device;
use pc_gpu::bts::stc_first::{StcFirstPlan, StcFirstSetup};
use pc_gpu::bts::staged::{BtsCall, CallOpts, Phase, StagedRef};
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
use pc_tfhe::modq::{centered, from_signed, invmod, mulmod, submod, Ntt};
use pc_tfhe::params::Params;
use pc_tfhe::pbs::sign01;
use pc_tfhe::rlwe::{Rlwe, RnsRlwe};
use pc_tfhe::switch::Switcher;
use std::collections::BTreeSet;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

type W = u32;
type M = Barrett64;
type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn arg(a: &[String], k: &str) -> Option<String> {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned()
}

/// Drop the top prime of coefficient-form limbs: c -> (c - [c]_top) top^-1 on the others.
fn rescale_top(limbs: &mut Vec<Vec<u64>>, primes: &mut Vec<u64>) {
    let top = limbs.pop().unwrap();
    let pt = primes.pop().unwrap();
    for (l, &p) in limbs.iter_mut().zip(primes.iter()) {
        let inv = invmod(pt % p, p);
        for (x, &t) in l.iter_mut().zip(&top) {
            *x = mulmod(submod(*x, from_signed(centered(t, pt), p), p), inv, p);
        }
    }
}

pub fn cmd(args: &[String]) -> ExitCode {
    match run(args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("pc-check tfhe: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Res<bool> {
    let n_switch: usize = arg(args, "--switch").map(|s| s.parse()).transpose()?.unwrap_or(64);
    let level: u32 = arg(args, "--level").map(|s| s.parse()).transpose()?.unwrap_or(44);
    let cf: u32 = arg(args, "--cf").map(|s| s.parse()).transpose()?.unwrap_or(2);
    // the finish reads the LUT value v as v q0 / 2^out_cf (the LUT is arcsine-warped: pc_tfhe::pbs::warp) and, with its
    // x 2^cf, returns v / 2^(out_cf - cf)
    let out_cf: u32 = arg(args, "--out-cf").map(|s| s.parse()).transpose()?.unwrap_or(3);
    // --device: the same switch on the device too (pc_gpu::tfhe::TfheSwitch), on a second encryption of the values
    let device = args.iter().any(|a| a == "--device");
    // --chunk C: the device switch holds C values per pass (several chunks per call: apply_dev on device arguments)
    let chunk: Option<usize> = arg(args, "--chunk").map(|s| s.parse()).transpose()?;
    let t0 = Instant::now();
    let chain = Chain::generate(&ChainSpec::n32())?;
    let n = chain.n;
    let slots = n / 2;
    let mut setup = BtsSetup::n32_shipped();
    setup.raise_drops = vec![0];
    setup.k_max = 1;
    // --real: the device finish on the real-payload route (one EvalMod chain: the switched values are real)
    let real_fin = args.iter().any(|a| a == "--real");
    setup.real = real_fin;
    let mut sf = StcFirstSetup::n32_shipped();
    sf.cache_slots = 2;
    let plan = StcFirstPlan::new(&chain, &setup, &sf)?;
    let mut geoms = Vec::new();
    for v in &plan.dense.variants {
        geoms.extend(v.cts.iter().chain(&v.stc).map(|h| h.geom));
    }
    geoms.extend(plan.input.iter().map(|h| h.geom));
    geoms.extend(plan.sources.iter().flatten().map(|s| s.geom));
    let steps: BTreeSet<i64> = crate::bts_stc_first::rot_steps(&geoms, n);
    let spec = KeySetSpec { relin: true, conj: true, rots: steps.iter().map(|&s| RotSpec { step: s, q_top: None }).collect(), encaps: true };
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
    mp.reserve("check.cts", 2 * pc_cuda::Arena::bytes_for::<W>(CtBuf::<pc_cuda::Compute, W>::words(&lay, 1)));
    if device {
        mp.reserve("tfhe", pc_gpu::tfhe::TfheSwitch::bytes(n, &switch_params(args, n)?, chunk.unwrap_or(n_switch)));
        mp.reserve("check.tfhe", 4 * (n_switch + 2 * 4096) + 512);
        if args.iter().any(|a| a == "--native-max") {
            mp.reserve("check.native", 4 * (3 * n_switch * (2048 + 1) + 2 * n_switch + 3 * 2 * 4096) + 1024);
        }
    }
    let arena = mp.allocate(&dev)?;
    let mut ex = Executor::<W>::new(&dev, &arena, &mp)?;
    ex.ks.carve_regen(&arena, &lay)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &mp)?;
    let kplans = KsPlans::new(&arena, &ex.compute, &ctx, 1)?;
    let mut ops = Ops::new(&arena, &dev, &ex.compute, &ctx, ocfg)?;
    let mut work = BtsWork::<W>::new(&arena, lay, wshape)?;
    let mut keys = KeySet::<W>::carve(&arena, &chain, &spec, KskLayout::Dense)?;
    // --seed S: another draw of the keys and values (the key switch's error is key-dependent: average over seeds)
    let seed: u32 = arg(args, "--seed").map(|s| s.parse()).transpose()?.unwrap_or(0);
    let mut prng = test_prng(0x7f + 0x100 * seed);
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
    let mut y = Ct::new(CtMeta::new(0, 1), CtBuf::<pc_cuda::Compute, W>::new(&arena, "check.cts", lay, 1)?);
    let mut z = Ct::new(CtMeta::new(0, 1), CtBuf::<pc_cuda::Compute, W>::new(&arena, "check.cts", lay, 1)?);
    println!("[pc-check] tfhe: device setup {:.1} s", t0.elapsed().as_secs_f64());
    // the host switch, keyed by the same CKKS secret (and its device copy)
    let s: Vec<i64> = sk.s.iter().map(|&v| v as i64).collect();
    let (p0, p1) = (chain.q[0], chain.q[1]);
    let mut params = switch_params(args, n)?;
    params.out_cf = out_cf;
    let tk = Instant::now();
    let sw = Switcher::generate(params, &s, p0, p1, &mut prng)?;
    println!("[pc-check] tfhe: host switch keys {:.1} s (p0 {p0}, p1 {p1})", tk.elapsed().as_secs_f64());
    let mut dsw = if device { Some(pc_gpu::tfhe::TfheSwitch::new(&arena, &dev, &ex.compute, &sw, chunk.unwrap_or(n_switch))?) } else { None };
    let mut d_pos = arena.alloc::<pc_cuda::Compute, u32>("check.tfhe", n_switch)?;
    let mut d_tv = arena.alloc::<pc_cuda::Compute, u32>("check.tfhe", 2 * sw.params.tv_len())?;
    let native = args.iter().any(|a| a == "--native-max");
    let nbuf = |k: usize| arena.alloc::<pc_cuda::Compute, u32>("check.native", if native { k } else { 0 });
    let row = sw.params.n_br + 1;
    let mut nat = (nbuf(n_switch * row)?, nbuf(n_switch * row)?, nbuf(n_switch * row)?, nbuf(2 * n_switch)?, nbuf(3 * 2 * sw.params.tv_len())?);
    arena.freeze();
    let sh = Shadow::new(ShadowConfig { chain: ctx.shape, bts: BtsConfig::n32_shipped(), quirks: Quirks::none() }, Some(Arc::new(ctx.scale.clone())));
    let enc = Encoder::new(&chain);
    let nq = ctx.shape.n_q;

    // values: real, |v| < 0.9 (the forward phase m q0 / 4 must not wrap)
    let vals: Vec<Complex> = (0..slots).map(|_| Complex::new(prng.uniform_f64() * 1.8 - 0.9, 0.0)).collect();
    let top = (nq - 1 - level) as usize;
    let sf_ = ctx.scale.sf(top as i32)?;
    let mut meta = CtMeta::new(level, 1).with_nf(sf_);
    meta.slots = slots as u32;
    let msg = enc.encode_slots(&vals, slots, top, Payload::Complex, sf_)?;
    encr.encrypt_sk(&ctx, &ex.compute, &sec, &[msg], meta, &mut x, &mut prng)?;
    let r = Route { cf: Some(cf), stc_first: true, real: false, ..Route::dense() };
    let mut t = Tally::default();
    let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
    let mut inp = BtsCall::begin(StagedRef::StcFirst(&route), &cx, std::slice::from_ref(&x), &r, CallOpts { prescaled: false, own_cf: true })?;
    inp.stc_first_input(&mut cx, std::slice::from_mut(&mut x))?.ok_or("input below the StC-first entry")?;
    let split = x.meta;
    let stop = (nq - 1 - split.level) as usize;
    println!("[pc-check] tfhe: split at level {} deg {} (top limb {stop})", split.level, split.deg);

    // the device's own c0 + c1 s on primes p0, p1 (no rescale): the host decryption below must reproduce it
    let dev_phase = dec.decrypt_rows(&ctx, &cx.ex.compute, &x, Which::Dense)?;
    // to the host: coefficient form, mod q0
    pc_gpu::ntt::inverse(&ctx, &cx.ex.compute, &mut x.buf, Span::Q { top: stop }, Parts::Both)?;
    let rows: Vec<usize> = (0..=stop).collect();
    let mut primes: Vec<u64> = chain.q[..=stop].to_vec();
    let mut parts = Vec::new();
    for p in 0..2 {
        let w = x.buf.download_rows(&cx.ex.compute, 0, p, &rows)?;
        parts.push(w.chunks(n).map(|c| c.iter().map(|&v| v as u64).collect::<Vec<u64>>()).collect::<Vec<_>>());
    }
    for l in 0..2 {
        let host = Rlwe { c0: parts[0][l].clone(), c1: parts[1][l].clone() }.phase(&s, &Ntt::new(chain.q[l], n)?);
        let bad = host.iter().zip(&dev_phase[0][l]).filter(|(a, b)| a != b).count();
        t.add(&format!("tfhe: host c0 + c1 s == device decrypt_rows, prime {l}"), bad, n);
    }
    let mut pr1 = primes.clone();
    for part in parts.iter_mut() {
        let mut pr = primes.clone();
        while pr.len() > 2 {
            rescale_top(part, &mut pr);
        }
        pr1 = pr;
    }
    primes = pr1;
    let ct = RnsRlwe { limbs: (0..2).map(|l| Rlwe { c0: parts[0][l].clone(), c1: parts[1][l].clone() }).collect() };
    let ntts = [Ntt::new(primes[0], n)?, Ntt::new(primes[1], n)?];
    let ph = ct.phase(&s, &ntts);
    let q0 = primes[0] as f64 * primes[1] as f64;
    // slot i sits at coefficient brev((i - 1) mod N/2) (the bootstrap's bit-reversed StC order, one step off), its
    // imaginary part N/2 further; the value at m q0 / 2^cf
    let bits = (slots as u32).trailing_zeros();
    let coef_of = |i: usize| ((i + slots - 1) % slots).reverse_bits() >> (usize::BITS - bits);
    let unit_in = q0 / (1u64 << cf) as f64;
    let mut worst_in = 0f64;
    for i in 0..slots {
        worst_in = worst_in.max((ph[coef_of(i)] as f64 / unit_in - vals[i].re).abs());
        worst_in = worst_in.max((ph[coef_of(i) + slots] as f64 / unit_in - vals[i].im).abs());
    }
    println!("[pc-check] tfhe: split phase vs slots: max err {worst_in:.3e}");
    t.add("tfhe: split phase == slot values at brev((i - 1) mod N/2) (err < 1e-3)", (worst_in > 1e-3) as usize, 1);

    // the switch on spread coefficients
    let chosen: Vec<usize> = (0..n_switch).map(|i| (i * (slots / n_switch) + 7) % slots).collect();
    let valid: Vec<usize> = chosen.iter().map(|&i| coef_of(i)).collect();
    // --identity: the LUT f(m) = m, its finished output against the input measures the switch's input noise
    let identity = args.iter().any(|a| a == "--identity");
    let (tv, off) = if identity { (pc_tfhe::pbs::test_vector(sw.params.tv_len(), sw.params.q_br, |m| sw.lut_value(m)), 0) } else { sign01(sw.params.tv_len(), sw.delta_br()) };
    let ts = Instant::now();
    let out = sw.apply(&ct, &valid, &tv, off);
    println!("[pc-check] tfhe: host switch of {n_switch} coefficients {:.1} s", ts.elapsed().as_secs_f64());

    // back: x p2 p3 into the split's four primes, NTT, the finish
    let lift = |f: &[u64], pi: usize| -> Vec<W> {
        let p = chain.q[pi];
        if pi >= 2 {
            return vec![0; n];
        }
        let m = mulmod(chain.q[2] % p, chain.q[3] % p, p);
        f.iter().map(|&v| mulmod(v, m, p) as W).collect()
    };
    for (p, poly) in [(0usize, 0usize), (1, 1)] {
        let data: Vec<W> = (0..=stop).flat_map(|pi| lift(if poly == 0 { &out.limbs[pi.min(1)].c0 } else { &out.limbs[pi.min(1)].c1 }, pi)).collect();
        y.buf.upload_rows(&cx.ex.compute, 0, p, &rows, &data)?;
    }
    pc_gpu::ntt::forward(&ctx, &cx.ex.compute, &mut y.buf, Span::Q { top: stop }, Parts::Both)?;
    y.meta = split;
    let mut fin = inp.finish_call(0)?;
    fin.at_split(&cx, std::slice::from_ref(&y))?;
    fin.run_to(&mut cx, std::slice::from_mut(&mut y), Phase::Done)?;
    cx.ops.realize(&ctx, cx.ex, &sh, std::slice::from_mut(&mut y))?;
    let got = dec.decrypt_slots(&ctx, &ex.compute, &enc, &y, Payload::Complex, Which::Dense)?;
    let mut bad = 0;
    let mut worst = 0f64;
    let vset: BTreeSet<usize> = chosen.iter().copied().collect();
    for i in 0..slots {
        if !identity && vset.contains(&i) && vals[i].re.abs() < 0.02 {
            continue;
        }
        let want = if vset.contains(&i) { if identity { vals[i].re } else if vals[i].re >= 0.0 { 1.0 } else { 0.0 } } else { 0.0 };
        let gain = (1u64 << (out_cf - cf)) as f64;
        let e = (got[0][i].re * gain - want).abs().max(got[0][i].im.abs() * gain);
        worst = worst.max(e);
        bad += (e > 1e-2) as usize;
    }
    println!("[pc-check] tfhe: finished slots x 2^{} vs (1 + sign) / 2 (switched) and 0 (others): max err {worst:.3e}, output level {} deg {}", out_cf - cf, y.meta.level, y.meta.deg);
    t.add("tfhe: CKKS -> TFHE sign -> CKKS slots (err < 1e-2)", bad, slots);
    if let Some(ts) = dsw.as_mut() {
        // the device switch on a second encryption of the same values
        let msg = enc.encode_slots(&vals, slots, top, Payload::Complex, sf_)?;
        encr.encrypt_sk(&ctx, &ex.compute, &sec, &[msg], meta, &mut y, &mut prng)?;
        let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
        cx.ex.compute.sync()?;
        let t_split = Instant::now();
        let mut inp2 = BtsCall::begin(StagedRef::StcFirst(&route), &cx, std::slice::from_ref(&y), &r, CallOpts { prescaled: false, own_cf: true })?;
        let rr = Route { real: true, ..r };
        let fin_real = if real_fin { Some(BtsCall::begin(StagedRef::StcFirst(&route), &cx, std::slice::from_ref(&y), &rr, CallOpts { prescaled: false, own_cf: true })?) } else { None };
        inp2.stc_first_input(&mut cx, std::slice::from_mut(&mut y))?.ok_or("input below the StC-first entry")?;
        cx.ex.compute.sync()?;
        let t_split = t_split.elapsed().as_secs_f64() * 1e3;
        let pos: Vec<u32> = valid.iter().map(|&j| j as u32).collect();
        let tvr = pc_gpu::tfhe::tv_residues(&sw.bsk, &tv);
        // a warm-up call on scratch (x: any content at the split's level) builds the GEMM plan; then the timed one
        let tm = if native {
            // values / 2 into native LWEs, rounds d = x[2k + 1] - x[2k], x'[k] = x[2k] + relu(d) (compacted,
            // ping-pong), the max warped for the finish (x 2) into the first coefficient
            let q = sw.params.q_br;
            let tl = sw.params.tv_len();
            let tvs: Vec<u32> = [
                pc_tfhe::chain::native_tv(tl, q, |m| m / 2.0),
                pc_tfhe::chain::native_tv(tl, q, |m| m.max(0.0)),
                pc_tfhe::pbs::test_vector(tl, q, |m| sw.lut_value(2.0 * m)),
            ]
            .iter()
            .flat_map(|t| pc_gpu::tfhe::tv_residues(&sw.bsk, t))
            .collect();
            let (a_, b_, d_, ix, tvb) = &mut nat;
            tvb.upload(&cx.ex.compute, &tvs)?;
            d_pos.upload(&cx.ex.compute, &pos)?;
            let tvp = |k: usize| (tvb.vptr() as *const u32).wrapping_add(k * 2 * tl) as *const std::ffi::c_void;
            // warm-up (the GEMM plans), then the timed run
            let mut ms = 0.0;
            for rep in 0..2 {
                cx.ex.compute.sync()?;
                let t0 = Instant::now();
                ts.native_load(&ctx, &cx.ex.compute, &blt.inner, &mut y, d_pos.vptr(), pos.len(), tvp(0), 0, a_.vptr_mut(), false)?;
                let (mut src, mut dst) = (a_.vptr_mut(), b_.vptr_mut());
                let mut cnt = pos.len();
                let p0 = sw.p0;
                while cnt > 1 {
                    let pairs = cnt / 2;
                    let idx: Vec<i32> = (0..pairs).flat_map(|k| [2 * k as i32 + 1, 2 * k as i32]).collect();
                    ix.upload_at(&cx.ex.compute, 0, bytemuck_i32(&idx))?;
                    ts.native_lin(&ctx, &cx.ex.compute, d_.vptr_mut(), src, src, ix.vptr(), pairs, 1, p0 - 1)?;
                    ts.native_pbs(&ctx, &cx.ex.compute, &blt.inner, d_.vptr(), pairs, tvp(1), &[0], &[d_.vptr_mut()])?;
                    let nx = cnt.div_ceil(2);
                    let idx: Vec<i32> = (0..nx).flat_map(|k| [2 * k as i32, if k < pairs { k as i32 } else { -1 }]).collect();
                    ix.upload_at(&cx.ex.compute, 0, bytemuck_i32(&idx))?;
                    ts.native_lin(&ctx, &cx.ex.compute, dst, src, d_.vptr(), ix.vptr(), nx, 1, 1)?;
                    std::mem::swap(&mut src, &mut dst);
                    cnt = nx;
                }
                ts.native_pbs(&ctx, &cx.ex.compute, &blt.inner, src, 1, tvp(2), &[0], &[dst])?;
                if rep == 1 {
                    ts.native_store(&ctx, &cx.ex.compute, &blt.inner, dst, d_pos.vptr(), 1, 1, &mut y)?;
                } else {
                    // the warm-up's input is gone with the load: a fresh split for the timed run
                    cx.ex.compute.sync()?;
                    encr.encrypt_sk(&ctx, &cx.ex.compute, &sec, &[enc.encode_slots(&vals, slots, top, Payload::Complex, sf_)?], meta, &mut y, &mut prng)?;
                    let mut i3 = BtsCall::begin(StagedRef::StcFirst(&route), &cx, std::slice::from_ref(&y), &r, CallOpts { prescaled: false, own_cf: true })?;
                    i3.stc_first_input(&mut cx, std::slice::from_mut(&mut y))?.ok_or("input below the StC-first entry")?;
                    continue;
                }
                cx.ex.compute.sync()?;
                ms = t0.elapsed().as_secs_f64() * 1e3;
            }
            println!("[pc-check] tfhe: native max of {n_switch}: {} rounds, TFHE {ms:.2} ms (load + rounds + warp + repack; no CKKS between)", (n_switch as f64).log2().ceil());
            pc_gpu::tfhe::Timings::default()
        } else if chunk.is_some() {
            d_pos.upload(&cx.ex.compute, &pos)?;
            d_tv.upload(&cx.ex.compute, &tvr)?;
            ts.apply_dev(&ctx, &cx.ex.compute, &blt.inner, &mut x, d_pos.vptr(), pos.len(), 1, d_tv.vptr(), off, true)?;
            ts.apply_dev(&ctx, &cx.ex.compute, &blt.inner, &mut y, d_pos.vptr(), pos.len(), 1, d_tv.vptr(), off, true)?
        } else {
            ts.apply(&ctx, &cx.ex.compute, &blt.inner, &mut x, &pos, &tvr, off)?;
            ts.apply(&ctx, &cx.ex.compute, &blt.inner, &mut y, &pos, &tvr, off)?
        };
        cx.ex.compute.sync()?;
        let t_fin = Instant::now();
        let mut fin2 = match fin_real { Some(c) => c, None => inp2.finish_call(0)? };
        fin2.at_split(&cx, std::slice::from_ref(&y))?;
        // --finish-steps: each finish phase timed (synchronous)
        let mut steps = String::new();
        while fin2.phase() != Phase::Done {
            let t = Instant::now();
            let ph = fin2.step(&mut cx, std::slice::from_mut(&mut y))?;
            if args.iter().any(|a| a == "--finish-steps") {
                cx.ex.compute.sync()?;
                steps += &format!(" {ph:?} {:.2}", t.elapsed().as_secs_f64() * 1e3);
            }
        }
        cx.ops.realize(&ctx, cx.ex, &sh, std::slice::from_mut(&mut y))?;
        cx.ex.compute.sync()?;
        println!("[pc-check] tfhe: CKKS sides: StC-first split {t_split:.2} ms, finish {:.2} ms (first calls){steps}", t_fin.elapsed().as_secs_f64() * 1e3);
        let got = dec.decrypt_slots(&ctx, &ex.compute, &enc, &y, Payload::Complex, Which::Dense)?;
        let (mut bad, mut worst) = (0, 0f64);
        let (mut ss, mut ns) = (0f64, 0usize);
        for i in 0..slots {
            if !identity && vset.contains(&i) && vals[i].re.abs() < 0.02 {
                continue;
            }
            let want = if vset.contains(&i) { if identity { vals[i].re } else if vals[i].re >= 0.0 { 1.0 } else { 0.0 } } else { 0.0 };
            let gain = (1u64 << (out_cf - cf)) as f64;
            let e = (got[0][i].re * gain - want).abs().max(got[0][i].im.abs() * gain);
            worst = worst.max(e);
            bad += (e > 1e-2) as usize;
            if vset.contains(&i) {
                ss += (got[0][i].re * gain - want).powi(2);
                ns += 1;
            }
        }
        if native {
            let gain = (1u64 << (out_cf - cf)) as f64;
            let want = chosen.iter().map(|&i| vals[i].re).fold(f64::MIN, f64::max);
            let rest = (0..slots).filter(|&i| i != chosen[0]).map(|i| got[0][i].re.abs() * gain).fold(0f64, f64::max);
            println!("[pc-check] tfhe: native max: got {:.4} want {want:.4} (err {:.2e}); other slots max |v| {rest:.2e}", got[0][chosen[0]].re * gain, got[0][chosen[0]].re * gain - want);
            t.add("tfhe: native max (err < 3e-2)", ((got[0][chosen[0]].re * gain - want).abs() > 3e-2) as usize, 1);
            println!("[pc-check] tfhe: {} gates, {} failing", t.gates, t.bad);
            return Ok(t.bad == 0);
        }
        println!("[pc-check] tfhe: device switched slots: rms error {:.3e} over {ns} (the LUT's input noise when --identity)", (ss / ns.max(1) as f64).sqrt());
        println!("[pc-check] tfhe: device switch of {n_switch} coefficients: max err {worst:.3e}; ms to LWE {:.2}, key switch {:.2}, bootstrap {:.2}, repack {:.2}", tm.to_lwe, tm.key_switch, tm.bootstrap, tm.repack);
        t.add("tfhe: device CKKS -> TFHE sign -> CKKS slots (err < 1e-2)", bad, slots);
        if args.iter().any(|a| a == "--taylor") {
            // first-order correction of the switch's input error (after Crasson, Mehats, ePrint 2026/1533): LUTs on one
            // switch input, f(m) ~ g(m~) + f'(m~) m with g = f - f' m; --taylor-fn tanh (tanh(2 m) / 2) or phi
            let phi = arg(args, "--taylor-fn").as_deref() == Some("phi");
            let f = move |m: f64| if phi { 0.5 * libm_erf(3.0 * m / std::f64::consts::SQRT_2) } else { (2.0 * m).tanh() / 2.0 };
            let fd = move |m: f64| if phi { 3.0 / (2.0 * std::f64::consts::PI).sqrt() * (-(3.0 * m).powi(2) / 2.0).exp() } else { 1.0 / (2.0 * m).cosh().powi(2) };
            let g = move |m: f64| f(m) - fd(m) * m;
            let luts: [&dyn Fn(f64) -> f64; 4] = [&f, &fd, &|m| m, &g];
            let msg = enc.encode_slots(&vals, slots, top, Payload::Complex, sf_)?;
            encr.encrypt_sk(&ctx, &ex.compute, &sec, &[msg], meta, &mut y, &mut prng)?;
            let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
            let mut inp3 = BtsCall::begin(StagedRef::StcFirst(&route), &cx, std::slice::from_ref(&y), &r, CallOpts { prescaled: false, own_cf: true })?;
            inp3.stc_first_input(&mut cx, std::slice::from_mut(&mut y))?.ok_or("input below the StC-first entry")?;
            let gain = (1u64 << (out_cf - cf)) as f64;
            let mut outs: Vec<Vec<f64>> = Vec::new();
            for lut in luts {
                let tv = pc_tfhe::pbs::test_vector(sw.params.tv_len(), sw.params.q_br, |m| sw.lut_value(lut(m)));
                let tvr = pc_gpu::tfhe::tv_residues(&sw.bsk, &tv);
                pc_program::replay::copy_ct(&ctx, &cx.ex.compute, &mut z, &y)?;
                ts.apply(&ctx, &cx.ex.compute, &blt.inner, &mut z, &pos, &tvr, 0)?;
                let mut fin3 = inp3.finish_call(0)?;
                fin3.at_split(&cx, std::slice::from_ref(&z))?;
                fin3.run_to(&mut cx, std::slice::from_mut(&mut z), Phase::Done)?;
                cx.ops.realize(&ctx, cx.ex, &sh, std::slice::from_mut(&mut z))?;
                let got = dec.decrypt_slots(&ctx, &cx.ex.compute, &enc, &z, Payload::Complex, Which::Dense)?;
                outs.push(chosen.iter().map(|&i| got[0][i].re * gain).collect());
            }
            // --mlut: g and f' interleaved into one switch (one blind rotation, apply_dev_k: two outputs)
            if args.iter().any(|a| a == "--mlut") {
                let half = sw.params.tv_len() / 2;
                let tvs: Vec<Vec<u64>> = [&g as &dyn Fn(f64) -> f64, &fd].iter().map(|lut| pc_tfhe::pbs::test_vector(half, sw.params.q_br, |m| sw.lut_value(lut(m)))).collect();
                let tvr = pc_gpu::tfhe::tv_residues(&sw.bsk, &pc_tfhe::pbs::interleave(&tvs));
                d_pos.upload(&cx.ex.compute, &pos)?;
                d_tv.upload(&cx.ex.compute, &tvr)?;
                pc_program::replay::copy_ct(&ctx, &cx.ex.compute, &mut z, &y)?;
                let tm = ts.apply_dev_k(&ctx, &cx.ex.compute, &blt.inner, &mut z, &mut [&mut x], d_pos.vptr(), pos.len(), 1, d_tv.vptr(), &[0, 0], true)?;
                let mut got2: Vec<Vec<f64>> = Vec::new();
                for c in [&mut z, &mut x] {
                    let mut fin4 = inp3.finish_call(0)?;
                    fin4.at_split(&cx, std::slice::from_ref(&*c))?;
                    fin4.run_to(&mut cx, std::slice::from_mut(c), Phase::Done)?;
                    cx.ops.realize(&ctx, cx.ex, &sh, std::slice::from_mut(c))?;
                    let got = dec.decrypt_slots(&ctx, &cx.ex.compute, &enc, c, Payload::Complex, Which::Dense)?;
                    got2.push(chosen.iter().map(|&i| got[0][i].re * gain).collect());
                }
                let (mut e2, mut w2) = (0f64, 0f64);
                for (k, &i) in chosen.iter().enumerate() {
                    let m = vals[i].re;
                    let d = got2[0][k] + got2[1][k] * m - f(m);
                    e2 += d * d;
                    w2 = w2.max(d.abs());
                }
                println!("[pc-check] tfhe: taylor many-LUT (one rotation) over {}: g(m~) + f'(m~) m rms {:.3e} max {:.3e}; ms bootstrap {:.2}, repack {:.2}", chosen.len(), (e2 / chosen.len() as f64).sqrt(), w2, tm.bootstrap, tm.repack);
            }
            let (mut e, mut w) = ([0f64; 3], [0f64; 3]);
            for (k, &i) in chosen.iter().enumerate() {
                let m = vals[i].re;
                let d = [outs[0][k] - f(m), outs[0][k] + outs[1][k] * (m - outs[2][k]) - f(m), outs[3][k] + outs[1][k] * m - f(m)];
                for j in 0..3 {
                    e[j] += d[j] * d[j];
                    w[j] = w[j].max(d[j].abs());
                }
            }
            let k = chosen.len() as f64;
            println!("[pc-check] tfhe: taylor over {}: f(m~) rms {:.3e} max {:.3e}; f(m~) + f'(m~) (m - m~) rms {:.3e} max {:.3e}; g(m~) + f'(m~) m rms {:.3e} max {:.3e}", chosen.len(), (e[0] / k).sqrt(), w[0], (e[1] / k).sqrt(), w[1], (e[2] / k).sqrt(), w[2]);
        }
    }
    println!("[pc-check] tfhe: {} gates, {} failing", t.gates, t.bad);
    Ok(t.bad == 0)
}

/// `pc-check tfhe-pbs [--count 256]`: the device's programmable bootstrap (PBS_MS -> PBS_INIT -> PBS_ROTATE ->
/// PBS_EXTRACT) on a batch of sign LUTs vs pc-tfhe's `bootstrap` + `mod_switch`, every output word; PBS/s.
pub fn cmd_pbs(args: &[String]) -> ExitCode {
    match run_pbs(args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("pc-check tfhe-pbs: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_pbs(args: &[String]) -> Res<bool> {
    use pc_core::keys::{sample_ternary, DiscreteGaussian};
    use pc_kernels::args::{tfhe_op, TfheArgs};
    use pc_tfhe::gadget::{fixed_ratio, Gadget};
    use pc_tfhe::lwe::Lwe;
    use pc_tfhe::pbs::{BootstrapKey, Rns};
    let count: usize = arg(args, "--count").map(|s| s.parse()).transpose()?.unwrap_or(256);
    // --nu 2: the extended blind rotation (pc-tfhe bootstrap_ext; the batch path)
    let nu: usize = arg(args, "--nu").map(|s| s.parse()).transpose()?.unwrap_or(1);
    // --mlut K: K interleaved sign LUTs (deltas q >> (out_cf + j)) from one blind rotation, every extraction checked
    let kl: usize = arg(args, "--mlut").map(|s| s.parse()).transpose()?.unwrap_or(1);
    // the latency path (PBS_KEYBUNDLE then PBS_ROTATE on the bundles) for small batches; --no-kb forces the batch path
    let lv = pbs_gadget(args)?.map_or(1, |g| g.1);
    let kb_path = count <= if nu == 1 && lv == 1 { pc_gpu::tfhe::KB_MAX } else { pc_gpu::tfhe::KB_MAX_EXT } && !args.iter().any(|a| a == "--no-kb");
    // --no-cluster: one block per LWE; --cluster 2 / 4: that cluster size (default: by batch)
    let cl_mode: i32 = if args.iter().any(|a| a == "--no-cluster") { 1 } else { arg(args, "--cluster").map(|s| s.parse()).transpose()?.unwrap_or(0) };
    // --no-check: time the device only (no host reference); --reps R: the median of R timed runs
    let no_check = args.iter().any(|a| a == "--no-check");
    // --overlap: the latency path's key bundles of window k + 1 on a second stream while window k rotates (two buffers)
    let overlap = args.iter().any(|a| a == "--overlap");
    // --prio (with --overlap): the rotations on a high-priority stream (joined by events)
    let prio = args.iter().any(|a| a == "--prio");
    let reps: usize = arg(args, "--reps").map(|s| s.parse()).transpose()?.unwrap_or(1).max(1);
    let chain = Chain::generate(&ChainSpec::n32())?;
    let p0 = chain.q[0];
    let mut p = Params::new(chain.n)?;
    if let Some(g) = pbs_gadget(args)? {
        p.pbs = g;
    }
    // --n-lwe N --group G: another LWE dimension / multi-bit group (N a multiple of G)
    if let Some(v) = arg(args, "--n-lwe") {
        p.n_lwe = v.parse()?;
    }
    if let Some(v) = arg(args, "--group") {
        p.pbs_group = v.parse()?;
    }
    let (n, nl, q) = (p.n_br, p.n_lwe, p.q_br);
    let groups = p.n_lwe / p.pbs_group;
    let win: usize = arg(args, "--window").map(|s| s.parse()).transpose()?.unwrap_or(pc_gpu::tfhe::KB_WINDOW).min(groups);
    let mut prng = test_prng(0x51);
    let s_lwe: Vec<i64> = (0..nl).map(|_| (prng.next_u32() & 1) as i64).collect();
    let s_br: Vec<i64> = sample_ternary(&mut prng, n).into_iter().map(|x| x as i64).collect();
    let dg = DiscreteGaussian::new(p.bsk_sigma)?;
    let dg_lwe = DiscreteGaussian::new(p.lwe_sigma_rel * p0 as f64)?;
    let t0 = Instant::now();
    let bsk = BootstrapKey::generate(&s_lwe, &s_br, Gadget::new(q, p.pbs, false), Rns::new(p.br_primes.0, p.br_primes.1, n)?, p.pbs_group, &dg, &mut prng);
    println!("[pc-check] tfhe-pbs: host bootstrapping key {:.1} s (n_lwe {nl}, n_br {n}, Q {q} = {} x {}, gadget {:?})", t0.elapsed().as_secs_f64(), p.br_primes.0, p.br_primes.1, p.pbs);
    let sub: Vec<(Vec<u64>, u64)> = (0..kl).map(|j| sign01(nu * n / kl, q >> (p.out_cf + j as u32))).collect();
    let tv = pc_tfhe::pbs::interleave(&sub.iter().map(|x| x.0.clone()).collect::<Vec<_>>());
    let offs: Vec<u64> = sub.iter().map(|x| x.1).collect();
    let lwes: Vec<Lwe> = (0..count)
        .map(|_| {
            let m = prng.uniform_f64() * 1.8 - 0.9;
            Lwe::encrypt(((m * p0 as f64 / 4.0).round() as i64).rem_euclid(p0 as i64) as u64, &s_lwe, p0, &dg_lwe, &mut prng)
        })
        .collect();
    let th = Instant::now();
    let want: Vec<Lwe> = if no_check {
        vec![]
    } else {
        use rayon::prelude::*;
        lwes.par_iter().flat_map(|l| bsk.bootstrap_mlut(&tv, &offs, l, p0, nu).into_iter().map(|o| o.mod_switch(q, p0)).collect::<Vec<_>>()).collect()
    };
    let host_s = th.elapsed().as_secs_f64();

    let img = pc_gpu::tfhe::pbs_image(&bsk);
    let tvr = pc_gpu::tfhe::tv_residues(&bsk, &tv);
    let lin: Vec<u32> = lwes.iter().flat_map(|l| l.a.iter().chain(std::iter::once(&l.b)).map(|&v| v as u32)).collect();
    let dev = Device::open()?;
    // without thread-block clusters (below sm_90) the latency path is one level, nu 1 only
    let kb_path = kb_path && (nu == 1 && lv == 1 || dev.clusters()?);
    let kb_words = if kb_path { count * win * 4 * n * 2 * nu * lv * if overlap { 2 } else { 1 } } else { 0 };
    let words = img.bsk.len() + 2 * img.tw.len() + tvr.len() + count * nu * 4 * n + 2 * lin.len() + kl * count * (n + 1) + kb_words;
    let arena = pc_cuda::Arena::new(&dev, 4 * words + (64 << 20))?;
    let s = pc_cuda::Stream::<pc_cuda::Compute>::new(&dev)?;
    let up = |what: &'static str, v: &[u32]| -> Res<pc_cuda::DBuf<'_, pc_cuda::Compute, u32>> {
        let mut b = arena.alloc::<pc_cuda::Compute, u32>(what, v.len())?;
        b.upload(&s, v)?;
        Ok(b)
    };
    let d_bsk = up("tfhe.bsk", &img.bsk)?;
    let d_tw = up("tfhe.tw", &img.tw)?;
    let d_tws = up("tfhe.tw", &img.tws)?;
    let d_tv = up("tfhe.tv", &tvr)?;
    let d_in = up("tfhe.in", &lin)?;
    let mut d_acc = arena.alloc::<pc_cuda::Compute, u32>("tfhe.acc", count * nu * 4 * n)?;
    let mut d_ms = arena.alloc::<pc_cuda::Compute, u32>("tfhe.ms", lin.len())?;
    let mut d_out = arena.alloc::<pc_cuda::Compute, u32>("tfhe.out", kl * count * (n + 1))?;
    let mut d_kb = arena.alloc::<pc_cuda::Compute, u32>("tfhe.kb", kb_words.max(1))?;
    let table = pc_kernels::table::table(pc_kernels::table::Instance::U32Barrett64)?;
    let go = |a: &TfheArgs| -> Res<()> { Ok(pc_cuda::launch_result("tfhe", unsafe { (table.tfhe)(a, s.raw()) })?) };
    let s2 = pc_cuda::Stream::<pc_cuda::Compute>::new(&dev)?;
    let s3 = pc_cuda::Stream::<pc_cuda::Compute>::high_priority(&dev)?;
    let go3 = |a: &TfheArgs| -> Res<()> { Ok(pc_cuda::launch_result("tfhe", unsafe { (table.tfhe)(a, s3.raw()) })?) };
    let e_out = pc_cuda::Event::new(&dev)?;
    let go2 = |a: &TfheArgs| -> Res<()> { Ok(pc_cuda::launch_result("tfhe", unsafe { (table.tfhe)(a, s2.raw()) })?) };
    let ev = || pc_cuda::Event::new(&dev);
    let (e_in, ekb, erot) = (ev()?, [ev()?, ev()?], [ev()?, ev()?]);
    let kb_half = kb_words / if overlap { 2 } else { 1 };
    let base = TfheArgs { n: n as i32, count: count as i32, i1: nl as i32, q, ..Default::default() };
    let mut cx = [0u64; 8];
    cx[5] = nu as u64;
    cx[6] = img.c[6];
    cx[7] = img.c[7];
    let mut ic = img.c;
    ic[5] = nu as u64;
    let mut run = || -> Res<()> {
        go(&TfheArgs { op: tfhe_op::PBS_MS, x: d_ms.vptr_mut(), y: d_in.vptr(), m: fixed_ratio((2 * nu * n / kl) as u64, p0), c: [p0, 0, 0, 0, kl as u64, nu as u64, 0, 0], ..base })?;
        go(&TfheArgs { op: tfhe_op::PBS_INIT, x: d_acc.vptr_mut(), y: d_tv.vptr(), z: d_ms.vptr(), c: cx, ..base })?;
        let rot = TfheArgs { op: tfhe_op::PBS_ROTATE, i2: p.pbs_group as i32, x: d_acc.vptr_mut(), y: d_bsk.vptr(), z: d_ms.vptr(), tw: d_tw.vptr(), tws: d_tws.vptr(), m: img.m, c: ic, ..base };
        if kb_path && overlap {
            s.record(&e_in)?;
            s2.wait(&e_in)?;
            if prio {
                s3.wait(&e_in)?;
            }
            let (rs, gor): (&pc_cuda::Stream<pc_cuda::Compute>, &dyn Fn(&TfheArgs) -> Res<()>) = if prio { (&s3, &go3) } else { (&s, &go) };
            for (k, g0) in (0..groups).step_by(win).enumerate() {
                let gw = win.min(groups - g0);
                let b = k % 2;
                let kb = (d_kb.vptr_mut() as *mut u32).wrapping_add(b * kb_half) as *mut std::ffi::c_void;
                if k >= 2 {
                    s2.wait(&erot[b])?;
                }
                go2(&TfheArgs { op: tfhe_op::PBS_KEYBUNDLE, i0: g0 as i32, i4: gw as i32, x: kb, ..rot })?;
                s2.record(&ekb[b])?;
                rs.wait(&ekb[b])?;
                gor(&TfheArgs { w: kb as *const std::ffi::c_void, i3: cl_mode, i0: g0 as i32, i4: gw as i32, ..rot })?;
                rs.record(&erot[b])?;
            }
            if prio {
                s3.record(&e_out)?;
                s.wait(&e_out)?;
            }
        } else if kb_path {
            for g0 in (0..groups).step_by(win) {
                let gw = win.min(groups - g0);
                go(&TfheArgs { op: tfhe_op::PBS_KEYBUNDLE, i0: g0 as i32, i4: gw as i32, x: d_kb.vptr_mut(), ..rot })?;
                go(&TfheArgs { w: d_kb.vptr(), i3: cl_mode, i0: g0 as i32, i4: gw as i32, ..rot })?;
            }
        } else {
            go(&rot)?;
        }
        for (j, &off) in offs.iter().enumerate() {
            let x = (d_out.vptr_mut() as *mut u32).wrapping_add(j * count * (n + 1)) as *mut std::ffi::c_void;
            go(&TfheArgs { op: tfhe_op::PBS_EXTRACT, x, y: d_acc.vptr(), m: fixed_ratio(p0, q), c: [off, p0, 0, 0, j as u64, nu as u64, cx[6], cx[7]], ..base })?;
        }
        Ok(())
    };
    run()?;
    s.sync()?;
    let mut ts: Vec<f64> = (0..reps)
        .map(|_| -> Res<f64> {
            let td = Instant::now();
            run()?;
            s.sync()?;
            Ok(td.elapsed().as_secs_f64())
        })
        .collect::<Res<_>>()?;
    ts.sort_by(f64::total_cmp);
    let dev_s = ts[reps / 2];
    let got = d_out.download(&s)?;
    let mut bad = 0;
    for (iw, w) in want.iter().enumerate() {
        // want: [lwe][lut]; got: [lut][lwe]
        let (i, j) = (iw / kl, iw % kl);
        let row = &got[(j * count + i) * (n + 1)..(j * count + i + 1) * (n + 1)];
        bad += w.a.iter().zip(row).filter(|(a, b)| **a != **b as u64).count() + (w.b != row[n] as u64) as usize;
    }
    let mut t = Tally::default();
    if !no_check {
        t.add(&format!("tfhe-pbs: device PBS == pc-tfhe bootstrap + mod_switch, {count} LWEs x {kl} LUTs, every word"), bad, kl * count * (n + 1));
    }
    println!("[pc-check] tfhe-pbs: {} path", if kb_path { format!("latency (key bundles, windows of {win} groups)") } else { "batch".to_string() });
    println!("[pc-check] tfhe-pbs: device {:.2} ms for {count} PBS = {:.0} PBS/s (host {:.1} s on {} threads = {:.1} PBS/s)", dev_s * 1e3, count as f64 / dev_s, host_s, rayon::current_num_threads(), count as f64 / host_s);
    println!("[pc-check] tfhe-pbs: {} gates, {} failing", t.gates, t.bad);
    Ok(t.bad == 0)
}

/// `pc-check tfhe-ks [--count 4096]`: the device LWE key switch (KS_DIGITS -> four int8 GEMMs against the key's
/// balanced base-256 planes -> KS_RECOMBINE) vs pc-tfhe `LweKsk::apply`, every output word; LWEs/s.
pub fn cmd_ks(args: &[String]) -> ExitCode {
    match run_ks(args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("pc-check tfhe-ks: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_ks(args: &[String]) -> Res<bool> {
    use pc_core::keys::{sample_ternary, DiscreteGaussian};
    use pc_core::pcmm_keys::digit_planes;
    use pc_kernels::args::{tfhe_op, TfheArgs};
    use pc_tfhe::gadget::Gadget;
    use pc_tfhe::lwe::{Lwe, LweKsk};
    let count: usize = arg(args, "--count").map(|s| s.parse()).transpose()?.unwrap_or(4096);
    let reps: usize = arg(args, "--reps").map(|s| s.parse()).transpose()?.unwrap_or(1).max(1);
    const PLANES: usize = 4;
    let chain = Chain::generate(&ChainSpec::n32())?;
    let p0 = chain.q[0];
    let p = Params::new(chain.n)?;
    let n_in = chain.n / p.rs_factor;
    let (nl, lv) = (p.n_lwe, p.ks.1);
    let mut prng = test_prng(0x52);
    let s_sub: Vec<i64> = sample_ternary(&mut prng, n_in).into_iter().map(|x| x as i64).collect();
    let s_lwe: Vec<i64> = (0..nl).map(|_| (prng.next_u32() & 1) as i64).collect();
    let dg_lwe = DiscreteGaussian::new(p.lwe_sigma_rel * p0 as f64)?;
    let gad = Gadget::new(p0, p.ks, false);
    let t0 = Instant::now();
    let ksk = LweKsk::generate(&s_sub, &s_lwe, gad.clone(), &dg_lwe, &mut prng);
    println!("[pc-check] tfhe-ks: host key switching key {:.1} s ({n_in} -> {nl}, gadget {:?})", t0.elapsed().as_secs_f64(), p.ks);
    let lwes: Vec<Lwe> = (0..count).map(|_| Lwe { a: (0..n_in).map(|_| prng.uniform_below(p0)).collect(), b: prng.uniform_below(p0) }).collect();
    let th = Instant::now();
    let want: Vec<Lwe> = lwes.iter().map(|l| ksk.apply(l)).collect();
    let host_s = th.elapsed().as_secs_f64();

    // the key as K-contiguous columns: W[o][i levels + t], o < nl the a entries, o = nl the b entry
    let k = n_in * lv;
    let cols = nl + 1;
    let mut w = vec![0i64; cols * k];
    for i in 0..n_in {
        for t in 0..lv {
            let key = &ksk.keys[i][t];
            for o in 0..cols {
                let v = if o < nl { key.a[o] } else { key.b };
                w[o * k + i * lv + t] = pc_tfhe::modq::centered(v, p0);
            }
        }
    }
    let planes = digit_planes(&w, PLANES)?;
    let lin: Vec<u32> = lwes.iter().flat_map(|l| l.a.iter().chain(std::iter::once(&l.b)).map(|&v| v as u32)).collect();
    let dev = Device::open()?;
    let ws_bytes = 64 << 20;
    let bytes = planes.len() + count * k + 4 * (PLANES * cols * count + lin.len() + count * cols) + ws_bytes + (64 << 20);
    let arena = pc_cuda::Arena::new(&dev, bytes)?;
    let s = pc_cuda::Stream::<pc_cuda::Compute>::new(&dev)?;
    let lt = pc_cuda::blas::Lt::new(&dev)?;
    // int8 operands live in byte buffers (the PCMM's weight planes do the same)
    let mut d_planes: Vec<pc_cuda::DBuf<'_, pc_cuda::Compute, u8>> = Vec::new();
    for u in 0..PLANES {
        let mut b = arena.alloc::<pc_cuda::Compute, u8>("tfhe.ksk", cols * k)?;
        let bytes: Vec<u8> = planes[u * cols * k..(u + 1) * cols * k].iter().map(|&x| x as u8).collect();
        b.upload(&s, &bytes)?;
        d_planes.push(b);
    }
    let mut d_in = arena.alloc::<pc_cuda::Compute, u32>("tfhe.in", lin.len())?;
    d_in.upload(&s, &lin)?;
    let mut d_dig = arena.alloc::<pc_cuda::Compute, u8>("tfhe.dig", count * k)?;
    let mut d_g = arena.alloc::<pc_cuda::Compute, i32>("tfhe.g", PLANES * cols * count)?;
    let mut d_out = arena.alloc::<pc_cuda::Compute, u32>("tfhe.out", count * cols)?;
    let mut d_ws = arena.alloc::<pc_cuda::Compute, u8>("tfhe.ws", ws_bytes)?;
    let gemm = pc_gpu::matmul::gemm::I8Gemm::new(&lt.inner, count as u64, cols as u64, k as u64, 8, ws_bytes)?;
    let table = pc_kernels::table::table(pc_kernels::table::Instance::U32Barrett64)?;
    let go = |a: &TfheArgs| -> Res<()> { Ok(pc_cuda::launch_result("tfhe", unsafe { (table.tfhe)(a, s.raw()) })?) };
    let mut run = || -> Res<()> {
        go(&TfheArgs { op: tfhe_op::KS_DIGITS, count: count as i32, i1: n_in as i32, x: d_dig.vptr_mut(), y: d_in.vptr(), q: p0, m: gad.m, c: [p.ks.0 as u64, lv as u64, 0, 0, 0, 0, 0, 0], ..Default::default() })?;
        let gp = d_g.dptr_mut();
        for u in 0..PLANES {
            // SAFETY: the operands are live arena buffers of the plan's lengths, owned by `s`
            unsafe { gemm.run(&s, d_dig.dptr() as *const i8, d_planes[u].dptr() as *const i8, gp.add(u * cols * count), false, &mut d_ws)? };
        }
        go(&TfheArgs { op: tfhe_op::KS_RECOMBINE, count: count as i32, i1: n_in as i32, i2: nl as i32, x: d_out.vptr_mut(), y: d_g.vptr(), z: d_in.vptr(), q: p0, c: [0, 0, PLANES as u64, 0, 0, 0, 0, 0], ..Default::default() })?;
        Ok(())
    };
    run()?;
    s.sync()?;
    let mut ts: Vec<f64> = (0..reps)
        .map(|_| -> Res<f64> {
            let td = Instant::now();
            run()?;
            s.sync()?;
            Ok(td.elapsed().as_secs_f64())
        })
        .collect::<Res<_>>()?;
    ts.sort_by(f64::total_cmp);
    let dev_s = ts[reps / 2];
    let got = d_out.download(&s)?;
    let mut bad = 0;
    for (i, w) in want.iter().enumerate() {
        let row = &got[i * cols..(i + 1) * cols];
        bad += w.a.iter().zip(row).filter(|(a, b)| **a != **b as u64).count() + (w.b != row[nl] as u64) as usize;
    }
    let mut t = Tally::default();
    t.add(&format!("tfhe-ks: device key switch == pc-tfhe LweKsk::apply, {count} LWEs, every word"), bad, count * cols);
    println!("[pc-check] tfhe-ks: device {:.2} ms for {count} = {:.0} LWE/s ({}); host {:.1} s", dev_s * 1e3, count as f64 / dev_s, gemm.label(), host_s);
    println!("[pc-check] tfhe-ks: {} gates, {} failing", t.gates, t.bad);
    Ok(t.bad == 0)
}

/// `pc-check tfhe-bridge [--switch 256]`: the CKKS side of the switch on the device vs pc-tfhe, every
/// word: RESCALE_P1, the ring switch (RS_DIGITS -> NTT -> MAC -> INTT -> + c0), RS_EXTRACT, and the repack
/// (pc-gpu `DevRepack`) from the host bootstrap's outputs.
pub fn cmd_bridge(args: &[String]) -> ExitCode {
    match run_bridge(args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("pc-check tfhe-bridge: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_bridge(args: &[String]) -> Res<bool> {
    use pc_core::keys::{sample_ternary, DiscreteGaussian};
    use pc_kernels::args::{pcmm_op, tfhe_op, PcmmArgs, TfheArgs};
    use pc_tfhe::lwe::Lwe;
    use pc_tfhe::switch::rescale_p1;
    use rayon::prelude::*;
    let n_valid: usize = arg(args, "--switch").map(|s| s.parse()).transpose()?.unwrap_or(256);
    let chain = Chain::generate(&ChainSpec::n32())?;
    let n = chain.n;
    let (p0, p1) = (chain.q[0], chain.q[1]);
    let mut prng = test_prng(0x53);
    let s: Vec<i64> = sample_ternary(&mut prng, n).into_iter().map(|x| x as i64).collect();
    let tk = Instant::now();
    let sw = Switcher::generate(Params::new(n)?, &s, p0, p1, &mut prng)?;
    println!("[pc-check] tfhe-bridge: host switch keys {:.1} s", tk.elapsed().as_secs_f64());
    let (nb, sfac) = (sw.params.n_br, sw.params.rs_factor);
    let n_sub = n / sfac;
    let (rs_lv, rp_lv) = (sw.params.rs.1, sw.params.rp.1);
    let q0 = p0 as i128 * p1 as i128;
    let dg = DiscreteGaussian::new(3.19)?;
    let valid: Vec<usize> = (0..n_valid).map(|i| (i * (n / 2 / n_valid) + 7) % (n / 2)).collect();
    let msg: Vec<i128> = (0..n).map(|_| ((prng.uniform_f64() * 1.8 - 0.9) * q0 as f64 / 4.0).round() as i128).collect();
    let ct = RnsRlwe::encrypt(&msg, &s, &sw.ntt_p, &dg, &mut prng);
    // the host's stages
    let th = Instant::now();
    let low = rescale_p1(&ct, p0, p1);
    let rsw = sw.rs.apply(&low, &sw.ntt_p[0]);
    let ext: Vec<Lwe> = valid.iter().map(|&j| sw.rs.extract(&rsw, j, p0)).collect();
    let ksed: Vec<Lwe> = ext.par_iter().map(|l| sw.ksk.apply(l)).collect();
    let (tv, off) = sign01(nb, sw.delta_br());
    let outs: Vec<Lwe> = sw.bsk.bootstrap_many(&tv, off, &ksed, p0).iter().map(|o| o.mod_switch(sw.params.q_br, p0)).collect();
    let mut slots: Vec<Option<Lwe>> = vec![None; n];
    for (&j, o) in valid.iter().zip(&outs) {
        slots[j] = Some(o.clone());
    }
    let tr = Instant::now();
    let rep = sw.repack.repack(&slots, nb, &sw.ntt_p);
    println!("[pc-check] tfhe-bridge: host pipeline {:.1} s (repack {:.1} s)", th.elapsed().as_secs_f64(), tr.elapsed().as_secs_f64());

    // device
    let mu = |p: u64| -> u64 { ((1u128 << 64) / p as u128) as u64 };
    let (t0f, t0i, ni0) = sw.ntt_p[0].tables();
    let (t1f, t1i, ni1) = sw.ntt_p[1].tables();
    let tw_f: Vec<u32> = [t0f, t1f, t0f].iter().flat_map(|t| t.iter().map(|&x| x as u32)).collect();
    let tw_i: Vec<u32> = [t0i, t1i, t0i].iter().flat_map(|t| t.iter().map(|&x| x as u32)).collect();
    let pa = PcmmArgs { n: n as i32, p: [p0 as u32, p1 as u32, p0 as u32, 0], mu: [mu(p0), mu(p1), mu(p0), 0], c: [ni0 as u32, ni1 as u32, ni0 as u32, 0, 0, 0, 0, 0], ..Default::default() };
    let dev = Device::open()?;
    let bytes = 4 * (6 * n + 2 * 3 * n + rs_lv * n + rs_lv * 2 * n + 2 * n + n_valid * 2 * (n_sub + 1) + n_valid * (nb + 1) + n_valid) + pc_gpu::tfhe::DevRepack::bytes(n, nb, rp_lv) + (64 << 20);
    let arena = pc_cuda::Arena::new(&dev, bytes)?;
    let st = pc_cuda::Stream::<pc_cuda::Compute>::new(&dev)?;
    let table = pc_kernels::table::table(pc_kernels::table::Instance::U32Barrett64)?;
    let go = |a: &TfheArgs| -> Res<()> { Ok(pc_cuda::launch_result("tfhe", unsafe { (table.tfhe)(a, st.raw()) })?) };
    let gp = |a: &PcmmArgs| -> Res<()> { Ok(pc_cuda::launch_result("pcmm", unsafe { (table.pcmm)(a, st.raw()) })?) };
    let up = |what: &'static str, v: &[u32]| -> Res<pc_cuda::DBuf<'_, pc_cuda::Compute, u32>> {
        let mut b = arena.alloc::<pc_cuda::Compute, u32>(what, v.len())?;
        b.upload(&st, v)?;
        Ok(b)
    };
    let mut d_twf = up("tfhe.tw", &tw_f)?;
    let mut d_twi = up("tfhe.tw", &tw_i)?;
    let _ = (&mut d_twf, &mut d_twi);
    let ctv: Vec<u32> = [&ct.limbs[0].c0, &ct.limbs[1].c0, &ct.limbs[0].c1, &ct.limbs[1].c1].iter().flat_map(|v| v.iter().map(|&x| x as u32)).collect();
    let d_ct = up("tfhe.ct", &ctv)?;
    let mut d_low = up("tfhe.low", &vec![0; 2 * n])?;
    let mut d_dig = up("tfhe.dig", &vec![0; rs_lv * n])?;
    let rkey: Vec<u32> = sw.rs.keys().iter().flat_map(|(a, b)| a.iter().chain(b).map(|&x| x as u32)).collect();
    let d_rkey = up("tfhe.rkey", &rkey)?;
    let mut d_rs = up("tfhe.rs", &vec![0; 2 * n])?;
    let idx: Vec<u32> = valid.iter().map(|&j| j as u32).collect();
    let d_idx = up("tfhe.idx", &idx)?;
    let mut d_ext = up("tfhe.ext", &vec![0; n_valid * (n_sub + 1)])?;
    let mut t = Tally::default();
    let tg = Instant::now();
    go(&TfheArgs { op: tfhe_op::RESCALE_P1, n: n as i32, count: 1, x: d_low.vptr_mut(), y: d_ct.vptr(), q: p0, c: [pc_tfhe::modq::invmod(p1 % p0, p0), p1, 0, 0, 0, 0, 0, mu(p0)], ..Default::default() })?;
    st.sync()?;
    let t_rescale = tg.elapsed().as_secs_f64();
    let got_low = d_low.download(&st)?;
    let bad = got_low.iter().zip(low.c0.iter().chain(&low.c1)).filter(|(a, b)| **a as u64 != **b).count();
    t.add("tfhe-bridge: RESCALE_P1 == pc-tfhe rescale_p1", bad, 2 * n);
    let tg = Instant::now();
    let c1 = unsafe { (d_low.vptr() as *const u32).add(n) } as *const std::ffi::c_void;
    go(&TfheArgs { op: tfhe_op::RS_DIGITS, n: n as i32, count: 1, x: d_dig.vptr_mut(), y: c1, q: p0, m: sw.rs.gadget.m, c: [sw.params.rs.0 as u64, rs_lv as u64, 0, 0, 0, 0, 0, 0], ..Default::default() })?;
    gp(&PcmmArgs { op: pcmm_op::NTT, count: rs_lv as i32, i0: rs_lv as i32, x: d_dig.vptr_mut(), tw: d_twf.vptr(), ..pa })?;
    go(&TfheArgs { op: tfhe_op::MAC, n: n as i32, count: rs_lv as i32, i1: 2, x: d_rs.vptr_mut(), y: d_dig.vptr(), z: d_rkey.vptr(), q: p0, c: [0, 0, 0, 0, 0, 0, 0, mu(p0)], ..Default::default() })?;
    gp(&PcmmArgs { op: pcmm_op::INTT, count: 2, i0: 2, x: d_rs.vptr_mut(), tw: d_twi.vptr(), ..pa })?;
    gp(&PcmmArgs { op: pcmm_op::ADD, count: 1, i0: n as i32, x: d_rs.vptr_mut(), y: d_low.vptr(), ..pa })?;
    st.sync()?;
    let t_rs = tg.elapsed().as_secs_f64();
    let got_rs = d_rs.download(&st)?;
    let bad = got_rs.iter().zip(rsw.c0.iter().chain(&rsw.c1)).filter(|(a, b)| **a as u64 != **b).count();
    t.add("tfhe-bridge: ring switch == pc-tfhe RingSwitchKey::apply", bad, 2 * n);
    let tg = Instant::now();
    go(&TfheArgs { op: tfhe_op::RS_EXTRACT, n: n as i32, count: n_valid as i32, i1: n_sub as i32, i2: sfac as i32, x: d_ext.vptr_mut(), y: d_rs.vptr(), z: d_idx.vptr(), q: p0, ..Default::default() })?;
    st.sync()?;
    let t_ext = tg.elapsed().as_secs_f64();
    let got_ext = d_ext.download(&st)?;
    let mut bad = 0;
    for (i, w) in ext.iter().enumerate() {
        let row = &got_ext[i * (n_sub + 1)..(i + 1) * (n_sub + 1)];
        bad += w.a.iter().zip(row).filter(|(a, b)| **a != **b as u64).count() + (w.b != row[n_sub] as u64) as usize;
    }
    t.add("tfhe-bridge: RS_EXTRACT == pc-tfhe extract", bad, n_valid * (n_sub + 1));

    // the repack from the host bootstrap's outputs
    let outv: Vec<u32> = outs.iter().flat_map(|l| l.a.iter().chain(std::iter::once(&l.b)).map(|&v| v as u32)).collect();
    let d_outs = up("tfhe.outs", &outv)?;
    let blt = pc_cuda::blas::Lt::new(&dev)?;
    let mut drep = pc_gpu::tfhe::DevRepack::new(&arena, &st, &sw)?;
    let tg = Instant::now();
    drep.run(table, &st, &blt.inner, d_outs.vptr(), d_idx.vptr(), n_valid, &pa, d_twf.vptr(), d_twi.vptr())?;
    st.sync()?;
    let t_rep = tg.elapsed().as_secs_f64();
    let d_acc = &drep.acc;
    let got_acc = d_acc.download(&st)?;
    let want: Vec<u64> = [&rep.limbs[0].c0, &rep.limbs[1].c0, &rep.limbs[0].c1, &rep.limbs[1].c1].iter().flat_map(|v| v.iter().copied()).collect();
    let bad = got_acc.iter().zip(&want).filter(|(a, b)| **a as u64 != **b).count();
    t.add("tfhe-bridge: repack == pc-tfhe RepackKey::repack", bad, 4 * n);
    println!("[pc-check] tfhe-bridge: device ms: rescale {:.3}, ring switch {:.3}, extract {:.3} ({n_valid}), repack {:.3}", t_rescale * 1e3, t_rs * 1e3, t_ext * 1e3, t_rep * 1e3);
    println!("[pc-check] tfhe-bridge: {} gates, {} failing", t.gates, t.bad);
    Ok(t.bad == 0)
}

/// erf by its series / continued fraction (to ~1e-15)
fn libm_erf(x: f64) -> f64 {
    let t = x.abs();
    let r = if t < 3.0 {
        // Maclaurin series
        let (mut sum, mut term, mut k) = (t, t, 0.0f64);
        while term.abs() > 1e-17 * sum.abs() {
            k += 1.0;
            term *= -t * t / k;
            sum += term / (2.0 * k + 1.0);
        }
        2.0 / std::f64::consts::PI.sqrt() * sum
    } else {
        1.0 - (-t * t).exp() / (t * std::f64::consts::PI.sqrt())
    };
    r.copysign(x)
}

/// The tfhe check's switch parameters: Params' defaults, --nu, --pbs-gadget, --lat-group G, --rs-gadget b,l (the ring
/// switch's), --lwe n,log2 (Params::with_lwe)
fn switch_params(args: &[String], n: usize) -> Res<Params> {
    let mut p = Params::new(n)?;
    p.nu = arg(args, "--nu").map(|s| s.parse()).transpose()?.unwrap_or(1);
    if let Some(g) = pbs_gadget(args)? {
        p.pbs = g;
    }
    p.lat_group = arg(args, "--lat-group").map(|s| s.parse()).transpose()?.unwrap_or(0);
    // --native-max: the tournament max in TFHE (the native chains' key switch)
    p.chain = args.iter().any(|a| a == "--native-max");
    if let Some(v) = arg(args, "--rs-gadget") {
        let (b, l) = v.split_once(',').ok_or("--rs-gadget b,l")?;
        p.rs = (b.parse()?, l.parse()?);
    }
    if let Some(v) = arg(args, "--lwe") {
        let (n_lwe, l) = v.split_once(',').ok_or("--lwe n,log2")?;
        p = p.with_lwe(n_lwe.parse()?, l.parse()?)?;
    }
    Ok(p)
}

/// --pbs-gadget b,l: the bootstrapping key's gadget (base log, levels) instead of Params' (two levels: the batch path)
fn pbs_gadget(args: &[String]) -> Res<Option<(u32, usize)>> {
    let Some(v) = arg(args, "--pbs-gadget") else { return Ok(None) };
    let (b, l) = v.split_once(',').ok_or("--pbs-gadget b,l")?;
    Ok(Some((b.parse()?, l.parse()?)))
}

/// i32 words as the u32 a device index buffer holds.
fn bytemuck_i32(v: &[i32]) -> &[u32] {
    // SAFETY: i32 and u32 share size and alignment
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u32, v.len()) }
}
