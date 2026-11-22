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
    let spec = KeySetSpec { relin: true, conj: true, rots: steps.iter().map(|&s| RotSpec { step: s, q_top: None }).collect(), encaps: true, lean: None };
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
    mp.reserve("check.tfhe", 4 * (n_switch + 2 * 4096) + 512);
    if device {
        mp.reserve("tfhe", pc_gpu::tfhe::TfheSwitch::bytes(n, &switch_params(args, n)?, chunk.unwrap_or(n_switch)));
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
            // one value through 6 rounds, so it swings with the seed: seeds 0..7 at n 840 give 5e-3..3.1e-2 (n 795:
            // 2 of 7 seeds above 3e-2)
            t.add("tfhe: native max (err < 4e-2)", ((got[0][chosen[0]].re * gain - want).abs() > 4e-2) as usize, 1);
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
    // --enc-acc: the test vector as an encrypted accumulator (GLWE under s_br, `BootstrapKey::blind_rotate_acc`;
    // PBS_INIT i0 = 1), one shared by every LWE
    let enc_acc = args.iter().any(|a| a == "--enc-acc");
    let acc_comps: Option<Vec<[pc_tfhe::rlwe::Rlwe; 2]>> = enc_acc.then(|| {
        let tr = bsk.rns.residues(&tv);
        (0..nu)
            .map(|c| {
                let e = dg.sample_vec(&mut prng, n);
                [0, 1].map(|pp| {
                    let ntt = &bsk.rns.ntt[pp];
                    let m: Vec<u64> = (0..n).map(|j| tr[pp][nu * j + c]).collect();
                    let c1: Vec<u64> = (0..n).map(|_| prng.uniform_below(ntt.q)).collect();
                    pc_tfhe::rlwe::Rlwe::encrypt_with(&m, &s_br, c1, &e, ntt)
                })
            })
            .collect()
    });
    let th = Instant::now();
    let want: Vec<Lwe> = if no_check {
        vec![]
    } else {
        use rayon::prelude::*;
        match &acc_comps {
            Some(acc) => lwes.par_iter().flat_map(|l| bsk.bootstrap_acc(acc, &offs, l, p0).into_iter().map(|o| o.mod_switch(q, p0)).collect::<Vec<_>>()).collect(),
            None => lwes.par_iter().flat_map(|l| bsk.bootstrap_mlut(&tv, &offs, l, p0, nu).into_iter().map(|o| o.mod_switch(q, p0)).collect::<Vec<_>>()).collect(),
        }
    };
    let host_s = th.elapsed().as_secs_f64();

    let img = pc_gpu::tfhe::pbs_image(&bsk);
    let tvr: Vec<u32> = match &acc_comps {
        // [part][prime][nu n], the components interleaved
        Some(acc) => (0..2)
            .flat_map(|part| {
                let acc = acc.clone();
                (0..2).flat_map(move |pp| {
                    let acc = acc.clone();
                    (0..nu * n).map(move |x| {
                        let (j, c) = (x / nu, x % nu);
                        (if part == 0 { acc[c][pp].c0[j] } else { acc[c][pp].c1[j] }) as u32
                    })
                })
            })
            .collect(),
        None => pc_gpu::tfhe::tv_residues(&bsk, &tv),
    };
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
    let inst = pc_kernels::table::Instance::U32Barrett64;
    let table = if dev.is_host() { pc_host::table(inst)? } else { pc_kernels::table::table(inst)? };
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
        go(&TfheArgs { op: tfhe_op::PBS_INIT, i0: enc_acc as i32, x: d_acc.vptr_mut(), y: d_tv.vptr(), z: d_ms.vptr(), c: cx, ..base })?;
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
        t.add(&format!("tfhe-pbs: device PBS == pc-tfhe bootstrap + mod_switch, {count} LWEs x {kl} LUTs{}, every word", if enc_acc { ", encrypted accumulator" } else { "" }), bad, kl * count * (n + 1));
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
    let inst = pc_kernels::table::Instance::U32Barrett64;
    let table = if dev.is_host() { pc_host::table(inst)? } else { pc_kernels::table::table(inst)? };
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
    let bytes = 4 * (6 * n + 2 * 3 * n + rs_lv * n + rs_lv * 2 * n + 2 * n + n_valid * 2 * (n_sub + 1) + n_valid * (nb + 1) + n_valid) + pc_gpu::tfhe::DevRepack::bytes(n, nb, rp_lv, n_valid) + (64 << 20);
    let arena = pc_cuda::Arena::new(&dev, bytes)?;
    let st = pc_cuda::Stream::<pc_cuda::Compute>::new(&dev)?;
    let inst = pc_kernels::table::Instance::U32Barrett64;
    let table = if dev.is_host() { pc_host::table(inst)? } else { pc_kernels::table::table(inst)? };
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
    let mut drep = pc_gpu::tfhe::DevRepack::new(&arena, &st, &sw, n_valid)?;
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
    p.lat_group2 = arg(args, "--lat-group2").map(|s| s.parse()).transpose()?.unwrap_or(0);
    // --pbs-group g: the batch bootstrapping key's groups (key bits per blind-rotation step)
    if let Some(g) = arg(args, "--pbs-group") {
        p.pbs_group = g.parse()?;
    }
    // --ks b,l: the LWE key switch's gadget (base log, levels)
    if let Some(v) = arg(args, "--ks") {
        let (b, l) = v.split_once(',').ok_or("--ks b,l")?;
        p.ks = (b.parse()?, l.parse()?);
    }
    // --native-max: the tournament max in TFHE (the native chains' key switch)
    p.chain = args.iter().any(|a| a == "--native-max");
    if let Some(v) = arg(args, "--rs-gadget") {
        let (b, l) = v.split_once(',').ok_or("--rs-gadget b,l")?;
        p.rs = (b.parse()?, l.parse()?);
    }
    if let Some(v) = arg(args, "--lwe") {
        let (n_lwe, l) = v.split_once(',').ok_or("--lwe n,log2")?;
        p = p.with_lwe(n_lwe.parse()?, l.parse()?)?;
        if let Some(w) = p.lwe_warning() {
            eprintln!("[tfhe] warning: --lwe {v}: {w}");
        }
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

/// `pc-check tfhe-noise [--lwe n,log2] [--nu 1|2] [--samples 262144] [--count 16384] [--norm 5] [--seed S] [--out file]`:
/// the phase error at the TFHE bootstrap's input as TFHE-rs defines it for its 2_2 parameter sets (atomic pattern
/// BR -> DP -> KS -> MS): a bootstrap output (device PBS, identity LUT on 16 boxes of the padded half), times the 2-norm
/// `--norm` (TFHE-rs's max_noise_level), key switched s_br -> s_lwe (device KS), mod switched p0 -> 2 nu n_br (the
/// device's mean-compensated switch, on the host). Every stage is decrypted on the host.
///
/// Prints the measured sigma of the bootstrap output, of the key switch's output and of the bootstrap input (95%
/// interval), each next to the analytical model's (`NoiseModel`), and p_fail = P(|e| >= 1/64) (2_2: two message bits,
/// two carry bits and the padding bit) three ways: the Gaussian tail at the measured sigma, at the model's sigma, and a
/// Chernoff bound from the model terms' exact log-MGFs (no Gaussian assumption; terms independent).
///
/// --lwe n,log2: the LWE key (Params::with_lwe; default Params' 840, 2^-18 p0). --nu 2: the extended blind rotation
/// (mod switch to 4 n_br). --samples: total, rounded up to whole batches of --count. --out: the raw records, per
/// sample 5 little-endian f64 [message, e_br, e_ks, e_in, e_in_plain], errors as fractions of the torus (e_in_plain: a
/// plain, uncompensated mod switch, for comparison).
/// Gates: the device KS == pc-tfhe LweKsk::apply on 4 samples; the measured input sigma within 3% of the model's.
pub fn cmd_noise(args: &[String]) -> ExitCode {
    match run_noise(args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("pc-check tfhe-noise: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_noise(args: &[String]) -> Res<bool> {
    use pc_core::keys::{sample_ternary, DiscreteGaussian};
    use pc_core::pcmm_keys::digit_planes;
    use pc_kernels::args::{tfhe_op, TfheArgs};
    use pc_tfhe::chain::ChainKey;
    use pc_tfhe::gadget::{fixed_ratio, ms_body_mean, scale_round, Gadget};
    use pc_tfhe::lwe::Lwe;
    use pc_tfhe::pbs::{BootstrapKey, Rns};
    use rayon::prelude::*;
    use std::io::Write;
    const PLANES: usize = 4;
    const THR: f64 = 1.0 / 64.0;
    let count: usize = arg(args, "--count").map(|s| s.parse()).transpose()?.unwrap_or(16384);
    let samples: usize = arg(args, "--samples").map(|s| s.parse()).transpose()?.unwrap_or(1 << 18);
    let batches = samples.div_ceil(count).max(1);
    let nu: usize = arg(args, "--nu").map(|s| s.parse()).transpose()?.unwrap_or(1);
    if nu != 1 && nu != 2 {
        return Err("--nu 1 or 2".into());
    }
    let norm: u64 = arg(args, "--norm").map(|s| s.parse()).transpose()?.unwrap_or(5);
    let seed: u32 = arg(args, "--seed").map(|s| s.parse()).transpose()?.unwrap_or(0x61);
    let out_path = arg(args, "--out");
    let chain = Chain::generate(&ChainSpec::n32())?;
    let p0 = chain.q[0];
    let mut p = Params::new(chain.n)?;
    p.nu = nu;
    if let Some(v) = arg(args, "--lwe") {
        let (n_lwe, l) = v.split_once(',').ok_or("--lwe n,log2")?;
        p = p.with_lwe(n_lwe.parse()?, l.parse()?)?;
        if let Some(w) = p.lwe_warning() {
            eprintln!("[tfhe] warning: --lwe {v}: {w}");
        }
    }
    let (n, q, nl) = (p.n_br, p.q_br, p.n_lwe);
    let mut prng = test_prng(seed);
    let s_lwe: Vec<i64> = (0..nl).map(|_| (prng.next_u32() & 1) as i64).collect();
    let s_br: Vec<i64> = sample_ternary(&mut prng, n).into_iter().map(|x| x as i64).collect();
    let dg = DiscreteGaussian::new(p.bsk_sigma)?;
    let dg_lwe = DiscreteGaussian::new(p.lwe_sigma_rel * p0 as f64)?;
    let t0 = Instant::now();
    let bsk = BootstrapKey::generate(&s_lwe, &s_br, Gadget::new(q, p.pbs, false), Rns::new(p.br_primes.0, p.br_primes.1, n)?, p.pbs_group, &dg, &mut prng);
    let ck = ChainKey::generate(&s_br, &s_lwe, p0, &p, &mut prng)?;
    println!(
        "[pc-check] tfhe-noise: keys {:.1} s (n_lwe {nl}, sigma 2^{:.2} p0, p0 {p0}, n_br {n}, Q {q}, pbs {:?} group {}, ks {:?}, nu {nu}, norm {norm}; {} samples)",
        t0.elapsed().as_secs_f64(),
        p.lwe_sigma_rel.log2(),
        p.pbs,
        p.pbs_group,
        p.ks,
        batches * count
    );
    // identity LUT on 16 boxes of the padded half: tv[j] = floor(16 j / (nu n)) Q / 32
    let tvl = nu * n;
    let tv: Vec<u64> = (0..tvl).map(|j| ((16 * j / tvl) as f64 * q as f64 / 32.0).round() as u64).collect();
    let img = pc_gpu::tfhe::pbs_image(&bsk);
    let tvr = pc_gpu::tfhe::tv_residues(&bsk, &tv);
    // the chain key as K-contiguous int8 planes (as run_ks)
    let (n_in, lv) = (n, p.ks.1);
    let k = n_in * lv;
    let cols = nl + 1;
    let mut w = vec![0i64; cols * k];
    for i in 0..n_in {
        for t in 0..lv {
            let key = &ck.ksk.keys[i][t];
            for o in 0..cols {
                let v = if o < nl { key.a[o] } else { key.b };
                w[o * k + i * lv + t] = centered(v, p0);
            }
        }
    }
    let planes = digit_planes(&w, PLANES)?;
    drop(w);
    let dev = Device::open()?;
    let lin_words = count * (nl + 1);
    let ks_words = count * (n_in + 1);
    let ws_bytes = 64 << 20;
    let words = img.bsk.len() + 2 * img.tw.len() + tvr.len() + count * nu * 4 * n + 2 * lin_words + count * (n + 1) + ks_words + count * cols + PLANES * cols * count;
    let arena = pc_cuda::Arena::new(&dev, 4 * words + planes.len() + count * k + ws_bytes + (128 << 20))?;
    let s = pc_cuda::Stream::<pc_cuda::Compute>::new(&dev)?;
    let lt = pc_cuda::blas::Lt::new(&dev)?;
    let up = |what: &'static str, v: &[u32]| -> Res<pc_cuda::DBuf<'_, pc_cuda::Compute, u32>> {
        let mut b = arena.alloc::<pc_cuda::Compute, u32>(what, v.len())?;
        b.upload(&s, v)?;
        Ok(b)
    };
    let d_bsk = up("tfhe.bsk", &img.bsk)?;
    let d_tw = up("tfhe.tw", &img.tw)?;
    let d_tws = up("tfhe.tw", &img.tws)?;
    let d_tv = up("tfhe.tv", &tvr)?;
    let mut d_in = arena.alloc::<pc_cuda::Compute, u32>("tfhe.in", lin_words)?;
    let mut d_acc = arena.alloc::<pc_cuda::Compute, u32>("tfhe.acc", count * nu * 4 * n)?;
    let mut d_ms = arena.alloc::<pc_cuda::Compute, u32>("tfhe.ms", lin_words)?;
    let mut d_out = arena.alloc::<pc_cuda::Compute, u32>("tfhe.out", count * (n + 1))?;
    let mut d_planes: Vec<pc_cuda::DBuf<'_, pc_cuda::Compute, u8>> = Vec::new();
    for u in 0..PLANES {
        let mut b = arena.alloc::<pc_cuda::Compute, u8>("tfhe.ksk", cols * k)?;
        let bytes: Vec<u8> = planes[u * cols * k..(u + 1) * cols * k].iter().map(|&x| x as u8).collect();
        b.upload(&s, &bytes)?;
        d_planes.push(b);
    }
    drop(planes);
    let mut d_ksin = arena.alloc::<pc_cuda::Compute, u32>("tfhe.ksin", ks_words)?;
    let mut d_dig = arena.alloc::<pc_cuda::Compute, u8>("tfhe.dig", count * k)?;
    let mut d_g = arena.alloc::<pc_cuda::Compute, i32>("tfhe.g", PLANES * cols * count)?;
    let mut d_ksout = arena.alloc::<pc_cuda::Compute, u32>("tfhe.ksout", count * cols)?;
    let mut d_ws = arena.alloc::<pc_cuda::Compute, u8>("tfhe.ws", ws_bytes)?;
    let gemm = pc_gpu::matmul::gemm::I8Gemm::new(&lt.inner, count as u64, cols as u64, k as u64, 8, ws_bytes)?;
    let inst = pc_kernels::table::Instance::U32Barrett64;
    let table = if dev.is_host() { pc_host::table(inst)? } else { pc_kernels::table::table(inst)? };
    let go = |a: &TfheArgs| -> Res<()> { Ok(pc_cuda::launch_result("tfhe", unsafe { (table.tfhe)(a, s.raw()) })?) };
    let base = TfheArgs { n: n as i32, count: count as i32, i1: nl as i32, q, ..Default::default() };
    let mut cx = [0u64; 8];
    cx[5] = nu as u64;
    cx[6] = img.c[6];
    cx[7] = img.c[7];
    let mut ic = img.c;
    ic[5] = nu as u64;
    let gks = Gadget::new(p0, p.ks, false);
    let two_n = (2 * nu * n) as u64;
    let mm = fixed_ratio(two_n, p0);
    let mut file = match &out_path {
        Some(f) => Some(std::io::BufWriter::new(std::fs::File::create(f)?)),
        None => None,
    };
    let cen = |x: f64, m: f64| -> f64 { (x + m / 2.0).rem_euclid(m) - m / 2.0 };
    // per column (e_br, e_ks, e_in, e_in_plain): sum, sum of squares, max |e|; failures |e_in| >= 1/64
    let mut sums = [[0f64; 3]; 4];
    let mut fails = 0usize;
    let mut tot = 0usize;
    let (mut t_dev, mut t_host) = (0f64, 0f64);
    for bi in 0..batches {
        let th = Instant::now();
        let msgs: Vec<u64> = (0..count).map(|_| (prng.next_u32() & 15) as u64).collect();
        let lwes: Vec<Lwe> = msgs
            .iter()
            .map(|&m| Lwe::encrypt(((m as f64 + 0.5) / 32.0 * p0 as f64).round() as u64 % p0, &s_lwe, p0, &dg_lwe, &mut prng))
            .collect();
        let lin: Vec<u32> = lwes.iter().flat_map(|l| l.a.iter().chain(std::iter::once(&l.b)).map(|&v| v as u32)).collect();
        t_host += th.elapsed().as_secs_f64();
        let td = Instant::now();
        d_in.upload(&s, &lin)?;
        go(&TfheArgs { op: tfhe_op::PBS_MS, x: d_ms.vptr_mut(), y: d_in.vptr(), m: fixed_ratio(two_n, p0), c: [p0, 0, 0, 0, 1, nu as u64, 0, 0], ..base })?;
        go(&TfheArgs { op: tfhe_op::PBS_INIT, x: d_acc.vptr_mut(), y: d_tv.vptr(), z: d_ms.vptr(), c: cx, ..base })?;
        go(&TfheArgs { op: tfhe_op::PBS_ROTATE, i2: p.pbs_group as i32, x: d_acc.vptr_mut(), y: d_bsk.vptr(), z: d_ms.vptr(), tw: d_tw.vptr(), tws: d_tws.vptr(), m: img.m, c: ic, ..base })?;
        go(&TfheArgs { op: tfhe_op::PBS_EXTRACT, x: d_out.vptr_mut(), y: d_acc.vptr(), m: fixed_ratio(p0, q), c: [0, p0, 0, 0, 0, nu as u64, cx[6], cx[7]], ..base })?;
        let outs = d_out.download(&s)?;
        t_dev += td.elapsed().as_secs_f64();
        // x norm (TFHE-rs's max_noise_level scalar): the KS input [count][n + 1] mod p0
        let th = Instant::now();
        let ksin: Vec<u32> = outs.iter().map(|&v| ((v as u64 * norm) % p0) as u32).collect();
        t_host += th.elapsed().as_secs_f64();
        let td = Instant::now();
        d_ksin.upload(&s, &ksin)?;
        go(&TfheArgs { op: tfhe_op::KS_DIGITS, count: count as i32, i1: n_in as i32, x: d_dig.vptr_mut(), y: d_ksin.vptr(), q: p0, m: gks.m, c: [p.ks.0 as u64, lv as u64, 0, 0, 0, 0, 0, 0], ..Default::default() })?;
        let gp = d_g.dptr_mut();
        for u in 0..PLANES {
            // SAFETY: the operands are live arena buffers of the plan's lengths, owned by `s`
            unsafe { gemm.run(&s, d_dig.dptr() as *const i8, d_planes[u].dptr() as *const i8, gp.add(u * cols * count), false, &mut d_ws)? };
        }
        go(&TfheArgs { op: tfhe_op::KS_RECOMBINE, count: count as i32, i1: n_in as i32, i2: nl as i32, x: d_ksout.vptr_mut(), y: d_g.vptr(), z: d_ksin.vptr(), q: p0, c: [0, 0, PLANES as u64, 0, 0, 0, 0, 0], ..Default::default() })?;
        let ks = d_ksout.download(&s)?;
        t_dev += td.elapsed().as_secs_f64();
        let th = Instant::now();
        if bi == 0 {
            // the device KS == the host chain key switch on the first 4 samples
            for i in 0..4 {
                let x = Lwe { a: ksin[i * (n + 1)..i * (n + 1) + n].iter().map(|&v| v as u64).collect(), b: ksin[i * (n + 1) + n] as u64 };
                let want = ck.ksk.apply(&x);
                let row = &ks[i * cols..(i + 1) * cols];
                if want.a.iter().zip(row).any(|(a, b)| *a != *b as u64) || want.b != row[nl] as u64 {
                    println!("[pc-check] tfhe-noise: FAIL device KS != pc-tfhe LweKsk::apply (sample {i})");
                    return Ok(false);
                }
            }
        }
        let p0f = p0 as f64;
        let tnf = two_n as f64;
        let recs: Vec<[f64; 5]> = (0..count)
            .into_par_iter()
            .map(|i| {
                let m = msgs[i];
                // the bootstrap output under s_br
                let o = &outs[i * (n + 1)..(i + 1) * (n + 1)];
                let mut ph: i128 = o[n] as i128;
                for (a, s) in o[..n].iter().zip(&s_br) {
                    ph += *a as i128 * *s as i128;
                }
                let e_br = cen(ph.rem_euclid(p0 as i128) as f64 - m as f64 / 32.0 * p0f, p0f) / p0f;
                // x norm, key switched: under s_lwe
                let t5 = ((norm * m) % 32) as f64 / 32.0;
                let r = &ks[i * cols..(i + 1) * cols];
                let mut ph: i128 = r[nl] as i128;
                for (a, s) in r[..nl].iter().zip(&s_lwe) {
                    ph += *a as i128 * *s as i128;
                }
                let e_ks = cen(ph.rem_euclid(p0 as i128) as f64 - t5 * p0f, p0f) / p0f;
                // mod switched to 2 nu n_br: the next bootstrap's input
                let a: Vec<u64> = r[..nl].iter().map(|&v| v as u64).collect();
                let ms = |mean: bool| -> f64 {
                    let b = if mean { ms_body_mean(&a, r[nl] as u64, mm, two_n) } else { scale_round(r[nl] as u64, mm) % two_n };
                    let mut ph = b as i128;
                    for (x, s) in a.iter().zip(&s_lwe) {
                        ph += (scale_round(*x, mm) % two_n) as i128 * *s as i128;
                    }
                    cen(ph.rem_euclid(two_n as i128) as f64 - t5 * tnf, tnf) / tnf
                };
                [m as f64, e_br, e_ks, ms(true), ms(false)]
            })
            .collect();
        for rcd in &recs {
            for c in 0..4 {
                let e = rcd[c + 1];
                sums[c][0] += e;
                sums[c][1] += e * e;
                sums[c][2] = sums[c][2].max(e.abs());
            }
            fails += (rcd[3].abs() >= THR) as usize;
        }
        tot += count;
        if let Some(f) = file.as_mut() {
            for rcd in &recs {
                for v in rcd {
                    f.write_all(&v.to_le_bytes())?;
                }
            }
        }
        t_host += th.elapsed().as_secs_f64();
        if bi % 8 == 7 && bi + 1 < batches {
            println!("[pc-check] tfhe-noise: {tot} samples, sigma in {:.4e} (dev {t_dev:.1} s, host {t_host:.1} s)", (sums[2][1] / tot as f64).sqrt());
        }
    }
    if let Some(mut f) = file {
        f.flush()?;
    }
    let model = NoiseModel::new(&p, p0, nu, norm as f64);
    let nf = tot as f64;
    // sigma about zero (the mod switch's mean is compensated; the means are printed) and its 95% interval
    let sd = |c: usize| (sums[c][1] / nf).sqrt();
    let ci = |c: usize| (sd(c) * (1.0 - 1.96 / (2.0 * nf).sqrt()), sd(c) * (1.0 + 1.96 / (2.0 * nf).sqrt()));
    let names = ["bootstrap output".to_string(), format!("x {norm}, key switched"), format!("bootstrap input (MS to {two_n})"), "  same, plain MS (comparison)".to_string()];
    let want = [model.var_br_out().sqrt(), model.var_ks().sqrt(), model.var_in().sqrt(), f64::NAN];
    println!("[pc-check] tfhe-noise: {tot} samples, device {t_dev:.1} s, host {t_host:.1} s; errors as fractions of the torus");
    for c in 0..4 {
        let (lo, hi) = ci(c);
        let m = if want[c].is_nan() { String::new() } else { format!(", model {:.4e} ({:+.2}%)", want[c], (sd(c) / want[c] - 1.0) * 1e2) };
        println!("[pc-check] tfhe-noise:   {:34} sigma {:.4e} [{lo:.4e}, {hi:.4e}]{m}; mean {:+.2e}, max |e| {:.2} sigma", names[c], sd(c), sums[c][0] / nf, sums[c][2] / sd(c));
    }
    let (lo, hi) = ci(2);
    let lp = |sig: f64| log2_gauss_tail(THR / sig);
    let (bound, lam) = model.chernoff_log2(THR);
    println!(
        "[pc-check] tfhe-noise: log2 p_fail = log2 P(|e_in| >= 1/64): measured {:.2} [{:.2}, {:.2}], model {:.2}, Chernoff bound <= {bound:.2} (lambda {lam:.0}); observed {fails} of {tot}",
        lp(sd(2)),
        lp(hi),
        lp(lo),
        lp(want[2])
    );
    let dev = (sd(2) / want[2] - 1.0).abs();
    let ok = dev < 0.03;
    println!("[pc-check] tfhe-noise: {} measured input sigma within 3% of the model ({:.2}%)", if ok { "PASS" } else { "FAIL" }, dev * 1e2);
    Ok(ok)
}

/// log2 P(|X| >= z) for X ~ N(0, 1): log2 erfc(z / sqrt 2) (the continued fraction past x = 2, no underflow).
fn log2_gauss_tail(z: f64) -> f64 {
    let x = z / std::f64::consts::SQRT_2;
    let ln = if x < 2.0 {
        (1.0 - libm_erf(x)).ln()
    } else {
        // erfc x = exp(-x^2) / sqrt(pi) / (x + (1/2) / (x + 1 / (x + (3/2) / (x + ...))))
        let mut t = x;
        for k in (1..=200).rev() {
            t = x + k as f64 / 2.0 / t;
        }
        -x * x - (std::f64::consts::PI.sqrt() * t).ln()
    };
    ln / std::f64::consts::LN_2
}

/// log(sinh(x) / x): the log-MGF of U(-1, 1) at x.
fn ln_sinhc(x: f64) -> f64 {
    let x = x.abs();
    if x < 1e-4 {
        x * x / 6.0
    } else {
        x + (-(-2.0 * x).exp()).ln_1p() - (2.0 * x).ln()
    }
}

/// The analytical noise of `pc-check tfhe-noise`'s pattern (variances as fractions of the torus, terms independent):
///   bootstrap output: per multi-bit group, 2^G - 1 GGSWs times (X^e - 1) (two key errors), 2 components x levels x
///   n_br digits uniform in [-B/2, B/2) x sigma_bsk^2 / Q^2; the gadget's rounding (eps_0 + eps_1 s_br, uniform
///   1 / B^levels) on the (1 - 2^-G) of the groups with a non-zero key; the extraction's switch Q -> p0;
///   x norm^2; the key switch: rounding to 2^-(b l) on h = 2 n_br / 3 nonzero s_br coefficients, the key's errors
///   sum_i D_i sigma_lwe^2 with D_i = sum_t d_t^2 the digit energy of a uniform input (exact over Z_p0);
///   the mean-compensated mod switch to 2 nu n_br: n_lwe / 48 + 1 / 12 in units of 1 / (2 nu n_br)^2.
struct NoiseModel {
    n_lwe: f64,
    sigma_lwe: f64,
    two_n: f64,
    norm: f64,
    n_br: f64,
    hw: f64,
    q_br: f64,
    p0: f64,
    bsk_sigma: f64,
    base: f64,
    levels: f64,
    group: usize,
    ks_bits: f64,
    /// (D, P(D)) of the key switch's digit energy per input coefficient
    digit_energy: Vec<(f64, f64)>,
}

impl NoiseModel {
    fn new(p: &Params, p0: u64, nu: usize, norm: f64) -> NoiseModel {
        use rayon::prelude::*;
        let g = pc_tfhe::gadget::Gadget::new(p0, p.ks, false);
        let lv = p.ks.1;
        let hist = (0..p0)
            .into_par_iter()
            .fold(
                || vec![0u64; 64 * lv + 1],
                |mut h, x| {
                    let mut d = [0i64; 64];
                    g.decompose(x, &mut d[..lv]);
                    let e: i64 = d[..lv].iter().map(|v| v * v).sum();
                    h[(e as usize).min(64 * lv)] += 1;
                    h
                },
            )
            .reduce(|| vec![0u64; 64 * lv + 1], |a, b| a.iter().zip(&b).map(|(x, y)| x + y).collect());
        let digit_energy = hist.iter().enumerate().filter(|(_, &c)| c > 0).map(|(e, &c)| (e as f64, c as f64 / p0 as f64)).collect();
        NoiseModel {
            n_lwe: p.n_lwe as f64,
            sigma_lwe: p.lwe_sigma_rel,
            two_n: (2 * nu * p.n_br) as f64,
            norm,
            n_br: p.n_br as f64,
            hw: (p.n_br as f64 * 2.0 / 3.0).round(),
            q_br: p.q_br as f64,
            p0: p0 as f64,
            bsk_sigma: p.bsk_sigma,
            base: (p.pbs.0 as f64).exp2(),
            levels: p.pbs.1 as f64,
            group: p.pbs_group,
            ks_bits: (p.ks.0 as usize * p.ks.1) as f64,
            digit_energy,
        }
    }

    fn groups(&self) -> f64 {
        (self.n_lwe as usize / self.group) as f64
    }

    fn var_br_out(&self) -> f64 {
        let g2 = (1u64 << self.group) as f64;
        let ed2 = self.base * self.base / 12.0 + 1.0 / 6.0;
        let key = self.groups() * (g2 - 1.0) * 4.0 * self.levels * self.n_br * ed2 * self.bsk_sigma.powi(2) / self.q_br.powi(2);
        let rnd = self.groups() * 2.0 * (1.0 - 1.0 / g2) * (1.0 + self.hw) / 12.0 / self.base.powf(2.0 * self.levels);
        key + rnd + (1.0 + self.hw) / 12.0 / self.p0.powi(2)
    }

    fn var_ks(&self) -> f64 {
        let ed = self.digit_energy.iter().map(|(d, p)| d * p).sum::<f64>();
        self.norm.powi(2) * self.var_br_out() + self.hw / 12.0 / (2.0 * self.ks_bits).exp2() + self.n_br * ed * self.sigma_lwe.powi(2)
    }

    fn var_in(&self) -> f64 {
        self.var_ks() + (self.n_lwe / 48.0 + 1.0 / 12.0) / self.two_n.powi(2)
    }

    /// The log-MGF of the input error at `l`, term by term (uniform roundings: ln_sinhc; Gaussian key errors given
    /// their digits: exp(l^2 sigma^2 sum d^2 / 2), averaged over the digits).
    fn log_mgf(&self, l: f64) -> f64 {
        let mut k = self.n_lwe * ln_sinhc(l / 4.0 / self.two_n) + ln_sinhc(l / 2.0 / self.two_n);
        k += self.hw * ln_sinhc(l / 2.0 / self.ks_bits.exp2());
        k += (1.0 + self.hw) * ln_sinhc(l * self.norm / 2.0 / self.p0);
        let s2 = self.sigma_lwe.powi(2);
        let mx = self.digit_energy.iter().map(|(d, _)| *d).fold(0.0, f64::max);
        let e: f64 = self.digit_energy.iter().map(|(d, p)| p * (l * l * s2 * (d - mx) / 2.0).exp()).sum();
        k += self.n_br * (e.ln() + l * l * s2 * mx / 2.0);
        // the bootstrapping key: per digit (uniform in [-B/2, B/2)) a Gaussian of variance norm^2 (2^G - 1) 2 sigma^2 d^2 / Q^2
        let c = l * l * self.norm.powi(2) * ((1u64 << self.group) as f64 - 1.0) * 2.0 * self.bsk_sigma.powi(2) / self.q_br.powi(2) / 2.0;
        let steps = 20000;
        let ed = (0..=steps).map(|i| (c * (self.base * (i as f64 / steps as f64 - 0.5)).powi(2)).exp_m1()).sum::<f64>() / (steps + 1) as f64;
        k += self.groups() * 2.0 * self.levels * self.n_br * ed.ln_1p();
        // its gadget rounding, counted on every group (conservative)
        k += self.groups() * 2.0 * (1.0 + self.hw) * ln_sinhc(l * self.norm / 2.0 / self.base.powf(self.levels));
        k
    }

    /// log2 of the Chernoff bound P(|e| >= thr) <= 2 min_l exp(K(l) - l thr) and the minimizing l (golden section on
    /// ln l; K is convex).
    fn chernoff_log2(&self, thr: f64) -> (f64, f64) {
        let f = |u: f64| self.log_mgf(u.exp()) - u.exp() * thr;
        let (mut a, mut b) = (1e2f64.ln(), 1e6f64.ln());
        let r = (5f64.sqrt() - 1.0) / 2.0;
        for _ in 0..200 {
            let (c, d) = (b - r * (b - a), a + r * (b - a));
            if f(c) < f(d) {
                b = d;
            } else {
                a = c;
            }
        }
        let u = (a + b) / 2.0;
        ((f(u) + std::f64::consts::LN_2) / std::f64::consts::LN_2, u.exp())
    }
}

/// `pc-check tfhe-pack [--reps 9] [--entries 8] [--count 64]`: the device packing key switch (pc_gpu::tfhe::DevPack:
/// PACK_DIGITS -> PACK_MAC -> PACK_FIN) vs pc-tfhe `PackingKey::pack`, every word, for k = 1 / 64 / 2048 LWEs and a
/// weighted case with repeated positions; then an encrypted lookup table on the device: `--entries` t 4-bit entries
/// (one LWE each, span n / t coefficients) packed, `--count` encrypted indices rotate it (PBS_INIT on the packed
/// accumulator), extraction mod p0: every word equals pc-tfhe `bootstrap_acc` + mod_switch and every output decodes
/// to its entry. Timings: packing per k, the rotation, one whole private lookup (pack + one rotation).
pub fn cmd_pack(args: &[String]) -> ExitCode {
    match run_pack(args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("pc-check tfhe-pack: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_pack(args: &[String]) -> Res<bool> {
    use pc_core::keys::{sample_ternary, DiscreteGaussian};
    use pc_kernels::args::{tfhe_op, TfheArgs};
    use pc_tfhe::gadget::{fixed_ratio, Gadget};
    use pc_tfhe::lwe::Lwe;
    use pc_tfhe::pack::{PackingKey, PACK_GADGET};
    use pc_tfhe::pbs::{BootstrapKey, Rns};
    use std::ffi::c_void;
    let reps: usize = arg(args, "--reps").map(|s| s.parse()).transpose()?.unwrap_or(9).max(1);
    let t_ent: usize = arg(args, "--entries").map(|s| s.parse()).transpose()?.unwrap_or(8);
    let count: usize = arg(args, "--count").map(|s| s.parse()).transpose()?.unwrap_or(64);
    let chain = Chain::generate(&ChainSpec::n32())?;
    let p0 = chain.q[0];
    let p = Params::new(chain.n)?;
    let (n, nl, q) = (p.n_br, p.n_lwe, p.q_br);
    if t_ent == 0 || n % t_ent != 0 {
        return Err(format!("--entries {t_ent}: a divisor of {n}").into());
    }
    let mut prng = test_prng(0x9ac);
    let s_lwe: Vec<i64> = (0..nl).map(|_| (prng.next_u32() & 1) as i64).collect();
    let s_br: Vec<i64> = sample_ternary(&mut prng, n).into_iter().map(|x| x as i64).collect();
    let dg = DiscreteGaussian::new(p.bsk_sigma)?;
    let dg_lwe = DiscreteGaussian::new(p.lwe_sigma_rel * p0 as f64)?;
    let rns = Rns::new(p.br_primes.0, p.br_primes.1, n)?;
    let t0 = Instant::now();
    let bsk = BootstrapKey::generate(&s_lwe, &s_br, Gadget::new(q, p.pbs, false), rns.clone(), p.pbs_group, &dg, &mut prng);
    let tk = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let pk = PackingKey::generate(&s_lwe, &s_br, PACK_GADGET, rns.clone(), &dg, &mut prng);
    println!("[pc-check] tfhe-pack: host keys: bootstrapping {tk:.1} s, packing {:.1} s (n_lwe {nl} x {} levels, n {n})", t0.elapsed().as_secs_f64(), PACK_GADGET.1);

    let dev = Device::open()?;
    let img = pc_gpu::tfhe::pbs_image(&bsk);
    let kmax = 2048usize.max(n);
    let groups = nl / p.pbs_group;
    let win = pc_gpu::tfhe::KB_WINDOW.min(groups);
    let kb_words = count.min(pc_gpu::tfhe::KB_MAX) * win * 4 * n * 2;
    let words = img.bsk.len() + 2 * img.tw.len() + kmax * (nl + 1) + 3 * kmax + 4 * n + count * (nl + 1) * 2 + count * 4 * n + count * (n + 1) + kb_words;
    let arena = pc_cuda::Arena::new(&dev, 4 * words + pc_gpu::tfhe::DevPack::bytes(n, nl, PACK_GADGET.1) + (64 << 20))?;
    let s = pc_cuda::Stream::<pc_cuda::Compute>::new(&dev)?;
    let inst = pc_kernels::table::Instance::U32Barrett64;
    let table = if dev.is_host() { pc_host::table(inst)? } else { pc_kernels::table::table(inst)? };
    let mut dp = pc_gpu::tfhe::DevPack::new(&arena, &s, &pk)?;
    let mut d_lwe = arena.alloc::<pc_cuda::Compute, u32>("lwe", kmax * (nl + 1))?;
    let mut d_place = arena.alloc::<pc_cuda::Compute, u32>("place", 2 * kmax)?;
    let mut d_w = arena.alloc::<pc_cuda::Compute, u32>("w", kmax)?;
    let mut d_acc = arena.alloc::<pc_cuda::Compute, u32>("acc", 4 * n)?;
    let mut t = Tally::default();
    let delta = p0 / 32;
    let enc = |m: u64, prng: &mut pc_core::keys::Prng| Lwe::encrypt(m % p0, &s_lwe, p0, &dg_lwe, prng);
    let flat = |c: &[Lwe]| -> Vec<u32> { c.iter().flat_map(|l| l.a.iter().chain(std::iter::once(&l.b)).map(|&v| v as u32)).collect() };
    let cmp = |got: &[u32], want: &[Rlwe; 2]| -> usize {
        let mut bad = 0;
        for part in 0..2 {
            for pr in 0..2 {
                let w = if part == 0 { &want[pr].c0 } else { &want[pr].c1 };
                bad += (0..n).filter(|&o| got[(part * 2 + pr) * n + o] as u64 != w[o]).count();
            }
        }
        bad
    };
    // 1. the packing: k LWEs at spread positions (weight 1); a weighted case with repeated positions
    for (k, weighted) in [(1usize, false), (64, false), (2048, false), (512, true)] {
        let cts: Vec<Lwe> = (0..k).map(|_| enc(prng.uniform_below(16) * delta, &mut prng)).collect();
        let pos: Vec<usize> = (0..k).map(|j| if weighted { j % 128 * 16 } else { (j * n / k) % n }).collect();
        let w: Vec<i64> = (0..k).map(|_| if weighted { prng.uniform_below(7) as i64 - 3 } else { 1 }).collect();
        let th = Instant::now();
        let want = pk.pack(&cts, p0, &pos, weighted.then_some(&w[..]));
        let host_s = th.elapsed().as_secs_f64();
        d_lwe.upload_at(&s, 0, &flat(&cts))?;
        let pl: Vec<i32> = (0..k).flat_map(|j| [j as i32, pos[j] as i32]).collect();
        d_place.upload_at(&s, 0, bytemuck_i32(&pl))?;
        let wi: Vec<i32> = w.iter().map(|&x| x as i32).collect();
        d_w.upload_at(&s, 0, bytemuck_i32(&wi))?;
        let wp = if weighted { d_w.vptr() } else { std::ptr::null() };
        let mut go = || -> Res<f64> {
            let td = Instant::now();
            dp.pack(table, &s, d_lwe.vptr(), d_place.vptr(), wp, k, p0, d_acc.vptr_mut())?;
            s.sync()?;
            Ok(td.elapsed().as_secs_f64())
        };
        go()?;
        let mut ts: Vec<f64> = (0..reps).map(|_| go()).collect::<Res<_>>()?;
        ts.sort_by(f64::total_cmp);
        let got = d_acc.download(&s)?;
        t.add(&format!("tfhe-pack: device packing == pc-tfhe PackingKey::pack, k {k}{}, every word", if weighted { " (weights, repeated positions)" } else { "" }), cmp(&got, &want), 4 * n);
        println!("[pc-check] tfhe-pack: k {k}{}: device {:.3} ms (median of {reps}), host {:.1} ms", if weighted { " weighted" } else { "" }, ts[reps / 2] * 1e3, host_s * 1e3);
    }

    // 2. an encrypted lookup table: entry i (4 bits at p0 / 32) over [i span, (i + 1) span), index i at phase
    // (i span + span / 2) p0 / (2 n): the rotation brings the entry's middle to coefficient 0
    let span = n / t_ent;
    let tab: Vec<u64> = (0..t_ent).map(|_| prng.uniform_below(16)).collect();
    let ent: Vec<Lwe> = tab.iter().map(|&v| enc(v * delta, &mut prng)).collect();
    let pl: Vec<i32> = (0..n).flat_map(|o| [(o / span) as i32, o as i32]).collect();
    let (host_cts, host_pos): (Vec<Lwe>, Vec<usize>) = (0..n).map(|o| (ent[o / span].clone(), o)).unzip();
    let acc_host = pk.pack(&host_cts, p0, &host_pos, None);
    d_lwe.upload_at(&s, 0, &flat(&ent))?;
    d_place.upload_at(&s, 0, bytemuck_i32(&pl))?;
    let idx: Vec<usize> = (0..count).map(|_| prng.uniform_below(t_ent as u64) as usize).collect();
    let ilwe: Vec<Lwe> = idx.iter().map(|&i| enc(((i * span + span / 2) as u64 * p0) / (2 * n as u64), &mut prng)).collect();
    let th = Instant::now();
    let want: Vec<Lwe> = {
        use rayon::prelude::*;
        ilwe.par_iter().map(|l| bsk.bootstrap_acc(&[acc_host.clone()], &[0], l, p0).swap_remove(0).mod_switch(q, p0)).collect()
    };
    let host_s = th.elapsed().as_secs_f64();
    let up = |what: &'static str, v: &[u32]| -> Res<pc_cuda::DBuf<'_, pc_cuda::Compute, u32>> {
        let mut b = arena.alloc::<pc_cuda::Compute, u32>(what, v.len())?;
        b.upload(&s, v)?;
        Ok(b)
    };
    let d_bsk = up("tfhe.bsk", &img.bsk)?;
    let d_tw = up("tfhe.tw", &img.tw)?;
    let d_tws = up("tfhe.tw", &img.tws)?;
    let d_in = up("tfhe.in", &flat(&ilwe))?;
    let mut d_ms = arena.alloc::<pc_cuda::Compute, u32>("tfhe.ms", count * (nl + 1))?;
    let mut d_racc = arena.alloc::<pc_cuda::Compute, u32>("tfhe.racc", count * 4 * n)?;
    let mut d_out = arena.alloc::<pc_cuda::Compute, u32>("tfhe.out", count * (n + 1))?;
    let mut d_kb = arena.alloc::<pc_cuda::Compute, u32>("tfhe.kb", kb_words.max(1))?;
    let go = |a: &TfheArgs| -> Res<()> { Ok(pc_cuda::launch_result("tfhe", unsafe { (table.tfhe)(a, s.raw()) })?) };
    let mut cx = [0u64; 8];
    cx[5] = 1;
    cx[6] = img.c[6];
    cx[7] = img.c[7];
    let mut ic = img.c;
    ic[5] = 1;
    let rotate = |cnt: usize, d_ms: &mut pc_cuda::DBuf<'_, pc_cuda::Compute, u32>, d_racc: &mut pc_cuda::DBuf<'_, pc_cuda::Compute, u32>, d_out: &mut pc_cuda::DBuf<'_, pc_cuda::Compute, u32>, d_kb: &mut pc_cuda::DBuf<'_, pc_cuda::Compute, u32>, acc: *const c_void| -> Res<()> {
        let base = TfheArgs { n: n as i32, count: cnt as i32, i1: nl as i32, q, ..Default::default() };
        go(&TfheArgs { op: tfhe_op::PBS_MS, x: d_ms.vptr_mut(), y: d_in.vptr(), m: fixed_ratio(2 * n as u64, p0), c: [p0, 0, 0, 0, 1, 1, 0, 0], ..base })?;
        go(&TfheArgs { op: tfhe_op::PBS_INIT, i0: 1, x: d_racc.vptr_mut(), y: acc, z: d_ms.vptr(), c: cx, ..base })?;
        let rot = TfheArgs { op: tfhe_op::PBS_ROTATE, i2: p.pbs_group as i32, x: d_racc.vptr_mut(), y: d_bsk.vptr(), z: d_ms.vptr(), tw: d_tw.vptr(), tws: d_tws.vptr(), m: img.m, c: ic, ..base };
        if cnt <= pc_gpu::tfhe::KB_MAX {
            for g0 in (0..groups).step_by(win) {
                let gw = win.min(groups - g0);
                go(&TfheArgs { op: tfhe_op::PBS_KEYBUNDLE, i0: g0 as i32, i4: gw as i32, x: d_kb.vptr_mut(), ..rot })?;
                go(&TfheArgs { w: d_kb.vptr(), i0: g0 as i32, i4: gw as i32, ..rot })?;
            }
        } else {
            go(&rot)?;
        }
        go(&TfheArgs { op: tfhe_op::PBS_EXTRACT, x: d_out.vptr_mut(), y: d_racc.vptr(), m: fixed_ratio(p0, q), c: [0, p0, 0, 0, 0, 1, cx[6], cx[7]], ..base })?;
        Ok(())
    };
    let med = |mut v: Vec<f64>| -> f64 {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    // the whole lookup: pack the t entries (n placements), rotate `cnt` indices
    type B<'x> = pc_cuda::DBuf<'x, pc_cuda::Compute, u32>;
    let mut lookup = |cnt: usize, d_ms: &mut B<'_>, d_racc: &mut B<'_>, d_out: &mut B<'_>, d_kb: &mut B<'_>| -> Res<(f64, f64)> {
        let td = Instant::now();
        dp.pack(table, &s, d_lwe.vptr(), d_place.vptr(), std::ptr::null(), n, p0, d_acc.vptr_mut())?;
        s.sync()?;
        let tp = td.elapsed().as_secs_f64();
        rotate(cnt, d_ms, d_racc, d_out, d_kb, d_acc.vptr())?;
        s.sync()?;
        Ok((tp, td.elapsed().as_secs_f64()))
    };
    lookup(count, &mut d_ms, &mut d_racc, &mut d_out, &mut d_kb)?;
    let full1: Vec<(f64, f64)> = (0..reps).map(|_| lookup(1, &mut d_ms, &mut d_racc, &mut d_out, &mut d_kb)).collect::<Res<_>>()?;
    let fullc: Vec<(f64, f64)> = (0..reps).map(|_| lookup(count, &mut d_ms, &mut d_racc, &mut d_out, &mut d_kb)).collect::<Res<_>>()?;
    drop(lookup);
    let got = d_out.download(&s)?;
    let got_acc = d_acc.download(&s)?;
    t.add(&format!("tfhe-pack: device table accumulator ({t_ent} entries x {span} coefficients) == host"), cmp(&got_acc, &acc_host), 4 * n);
    let (mut bad, mut wrong, mut s2) = (0usize, 0usize, 0.0f64);
    for (r, w) in want.iter().enumerate() {
        let row = &got[r * (n + 1)..(r + 1) * (n + 1)];
        bad += w.a.iter().zip(row).filter(|(a, b)| **a != **b as u64).count() + (w.b != row[n] as u64) as usize;
        let lw = Lwe { a: row[..n].iter().map(|&x| x as u64).collect(), b: row[n] as u64 };
        let ph = centered(lw.phase(&s_br, p0), p0) as f64;
        let v = (ph / delta as f64).round() as i64;
        wrong += (v.rem_euclid(32) != tab[idx[r]] as i64) as usize;
        let e = ph - (tab[idx[r]] * delta) as f64;
        s2 += e * e;
    }
    t.add(&format!("tfhe-pack: device private lookup == pc-tfhe bootstrap_acc + mod_switch, {count} lookups, every word"), bad, count * (n + 1));
    t.add(&format!("tfhe-pack: device private lookup decodes to the entry, {count} lookups"), wrong, count);
    let rms = (s2 / count as f64).sqrt();
    let model = (pk.noise_var(n, 1.0, p0, dg_lwe.sigma()) + pc_tfhe::pack::blind_rotate_var(q, n, nl, p.pbs_group, p.pbs, p.bsk_sigma)).sqrt() * p0 as f64 / q as f64;
    let p_idx = pc_tfhe::pack::index_log2_fail(nl, span, 1, p.lwe_sigma_rel);
    println!("[pc-check] tfhe-pack: output noise rms {rms:.3e} (p0 units; model {model:.3e} + the mod switch), spacing {delta}; index p_fail 2^{p_idx:.1} (span {span})");
    let rot1: Vec<f64> = (0..reps)
        .map(|_| -> Res<f64> {
            let td = Instant::now();
            rotate(1, &mut d_ms, &mut d_racc, &mut d_out, &mut d_kb, d_acc.vptr())?;
            s.sync()?;
            Ok(td.elapsed().as_secs_f64())
        })
        .collect::<Res<_>>()?;
    println!(
        "[pc-check] tfhe-pack: private lookup of {t_ent} entries: pack {:.3} ms; rotation of 1 index {:.3} ms; whole lookup (pack + 1 rotation) {:.3} ms; pack + {count} rotations {:.3} ms (host {:.1} s for the {count} rotations)",
        med(full1.iter().map(|x| x.0).collect()) * 1e3,
        med(rot1) * 1e3,
        med(full1.iter().map(|x| x.1).collect()) * 1e3,
        med(fullc.iter().map(|x| x.1).collect()) * 1e3,
        host_s
    );
    println!("[pc-check] tfhe-pack: {} gates, {} failing", t.gates, t.bad);
    Ok(t.bad == 0)
}

/// `pc-check tfhe-bitwise [--words 1,16,256] [--digits 16] [--reps 7] [--nu 2] [--lat-group 5] [--ops add,cmp,mux,xor]`:
/// native TFHE bitwise ops on radix-4 words, one native LWE (s_br, p0) per digit d at value d / 8, on the device's native
/// ops (`TfheSwitch::native_lin_c` / `native_pbs`: a key switch s_br -> s_lwe and a PBS per lookup). Every lookup
/// table has <= 8 bins over the native [-1, 1) (16 for xor):
/// - add: statuses S_i = sign(s_i - 3) / 4 (kill / propagate / generate, s_i = a_i + b_i), Kogge-Stone prefix rounds
///   S_i = sign(2 S_i + S_(i - d)) (the last one outputs the carry [S == G] / 8), then (s_i + c_i) mod 4: 1 + log2 n +
///   1 PBS rounds;
/// - cmp (a < b): statuses sign(a_i - b_i), a tree of the same combine from the top digit: 1 + log2 n rounds;
/// - mux (sel ? a : b): one PBS per digit, o = f(2 (a - b) / 8 + 2 sel) = -+(a - b) / 16 (the negacyclic half turn
///   flips it by sel), output a + b + 2 o = 2 (sel ? a : b) at value d / 4 (consumers fold the 2 into their
///   coefficients);
/// - xor: one bivariate PBS per digit on a + 4 b (16 bins: p_fail < 2^-128 needs nu 2).
/// Every output decrypted against the exact result; latency (device, synced) per word count; p_fail per lookup by
/// the input noise model (mod switch rounding + key switch).
pub fn cmd_bitwise(args: &[String]) -> ExitCode {
    match run_bitwise(args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("pc-check tfhe-bitwise: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_bitwise(args: &[String]) -> Res<bool> {
    use pc_core::keys::DiscreteGaussian;
    use pc_tfhe::lwe::Lwe;
    use std::ffi::c_void;
    let words: Vec<usize> = arg(args, "--words").unwrap_or_else(|| "1,16,256".into()).split(',').map(|x| x.trim().parse()).collect::<std::result::Result<_, _>>()?;
    let nd: usize = arg(args, "--digits").map(|s| s.parse()).transpose()?.unwrap_or(16);
    let reps: usize = arg(args, "--reps").map(|s| s.parse()).transpose()?.unwrap_or(7).max(1);
    let ops: Vec<String> = arg(args, "--ops").unwrap_or_else(|| "add,cmp,mux,xor".into()).split(',').map(str::to_string).collect();
    if !nd.is_power_of_two() || nd < 2 {
        return Err("--digits: a power of two >= 2".into());
    }
    let wmax = *words.iter().max().ok_or("--words")?;
    let rows = wmax * nd;
    let chain = Chain::generate(&ChainSpec::n32())?;
    let n = chain.n;
    let mut params = switch_params(args, n)?;
    params.chain = true;
    let (nb, nu) = (params.n_br, params.nu);
    let row = nb + 1;
    const NBUF: usize = 8;
    let dev = Device::open()?;
    let blt = pc_cuda::blas::Lt::new(&dev)?;
    let cfg = MemCfg { k_max: 1, pool: vec![PoolClass { k: 1, count: 1 }], events: 16, scratch_cts: 2 };
    let mut mp = MemPlan::build::<W, M>(&chain, cfg)?;
    mp.reserve("tfhe", pc_gpu::tfhe::TfheSwitch::bytes(n, &params, rows));
    mp.reserve("bitwise", 4 * (NBUF * rows * row + 2 * 2 * rows * 16 + 16 * 2 * params.tv_len()) + (1 << 20));
    let arena = mp.allocate(&dev)?;
    let ex = Executor::<W>::new(&dev, &arena, &mp)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &mp)?;
    let s = &ex.compute;
    let mut prng = test_prng(0xa1);
    let sk = SecretKey::sample(&mut prng, n)?;
    let sv: Vec<i64> = sk.s.iter().map(|&v| v as i64).collect();
    let (p0, p1) = (chain.q[0], chain.q[1]);
    let tk = Instant::now();
    let (sw, _s_lwe, s_br) = Switcher::generate_keep(params.clone(), &sv, p0, p1, &mut prng)?;
    println!("[pc-check] tfhe-bitwise: host keys {:.1} s (n_lwe {}, n_br {nb}, nu {nu}, lat_group {}, ks {:?})", tk.elapsed().as_secs_f64(), params.n_lwe, params.lat_group, params.ks);
    let mut ts = pc_gpu::tfhe::TfheSwitch::new(&arena, &dev, s, &sw, rows)?;
    let mut bufs: Vec<pc_cuda::DBuf<'_, pc_cuda::Compute, u32>> = (0..NBUF).map(|_| arena.alloc::<pc_cuda::Compute, u32>("bitwise", rows * row)).collect::<std::result::Result<_, _>>()?;
    let mut d_idx = arena.alloc::<pc_cuda::Compute, u32>("bitwise", 2 * 2 * rows * 16)?;
    let mut d_tv = arena.alloc::<pc_cuda::Compute, u32>("bitwise", 16 * 2 * params.tv_len())?;
    arena.freeze();
    // the native tables (v at phase v q / 4; inputs on [-1, 1))
    let r4 = |m: f64| (4.0 * m).round();
    let sg = |r: f64| if r > 0.0 { 1.0 } else if r < 0.0 { -1.0 } else { 0.0 };
    let tables: Vec<Box<dyn Fn(f64) -> f64>> = vec![
        Box::new(move |m| match r4(m) {                                                    // 0 sgn: sign(round 4 m) / 4
            r if r > 0.0 => 0.25,
            r if r < 0.0 => -0.25,
            _ => 0.0,
        }),
        Box::new(move |m| if r4(m) > 0.0 { 0.125 } else { 0.0 }),                         // 1 carry: [G] / 8
        Box::new(|m: f64| (((m + 1.0) * 4.0).floor().rem_euclid(4.0)) / 8.0),              // 2 out: (s + c) mod 4 / 8
        Box::new(move |m| if r4(m) < 0.0 { 0.125 } else { 0.0 }),                         // 3 lt: [a < b] / 8
        Box::new(move |m| -r4(m) / 16.0),                                                  // 4 mux: -(a - b) / 16
        Box::new(|m: f64| {                                                                // 5 xor on a + 4 b
            let k = ((m + 1.0) * 8.0).floor().clamp(0.0, 15.0) as usize;
            ((k % 4) ^ (k / 4)) as f64 / 8.0
        }),
        // the whole-torus tables of the fan-in 3 / 4 prefix: inputs reach (-2, 2) and the negacyclic half carries the
        // sign, so the only boundaries are the zero band and the wrap (outputs symmetric; lin adds the offsets)
        Box::new(move |m| if r4(m) > 0.0 { 0.0625 } else { -0.0625 }),                   // 6 carry: c / 8 - 1/16
        Box::new(move |m| if r4(m) < 0.0 { 0.0625 } else { -0.0625 }),                   // 7 lt: [a < b] / 8 - 1/16
        Box::new(move |m| sg((8.0 * m).round()) / 8.0),                                  // 8 status on unit 1/8 -> 1/8
        Box::new(move |m| sg((8.0 * m).round()) / 4.0),                                  // 9 status on unit 1/8 -> 1/4
        Box::new(move |m| if (8.0 * m).round() > 0.0 { 0.0625 } else { -0.0625 }),     // 10 carry on unit 1/8
        Box::new(move |m| if (8.0 * m).round() < 0.0 { 0.0625 } else { -0.0625 }),     // 11 lt on unit 1/8
        Box::new(move |m| sg(r4(m)) / 8.0),                                              // 12 status on unit 1/4 -> 1/8
    ];
    let tvl = params.tv_len();
    let mut tvw = Vec::new();
    for f in &tables {
        let tv = pc_tfhe::chain::native_tv(tvl, params.q_br, |m| f(m));
        tvw.extend(sw.bsk.rns.residues(&tv).iter().flat_map(|r| r.iter().map(|&x| x as u32)));
    }
    d_tv.upload(s, &tvw)?;
    let tvp = |k: usize| (d_tv.vptr() as *const u32).wrapping_add(k * 2 * tvl) as *const c_void;
    let enc_v = |v: f64| ((v * p0 as f64 / 4.0).round() as i64).rem_euclid(p0 as i64) as u64;
    let dg = DiscreteGaussian::new(params.lwe_sigma_rel * p0 as f64)?;
    let encrypt = |vals: &[f64], prng: &mut pc_core::keys::Prng| -> Vec<u32> {
        vals.iter().flat_map(|&v| {
            let l = Lwe::encrypt(enc_v(v), &s_br, p0, &dg, prng);
            l.a.into_iter().chain(std::iter::once(l.b)).map(|x| x as u32).collect::<Vec<_>>()
        }).collect()
    };
    let decrypt = |w: &[u32], cnt: usize| -> Vec<f64> {
        (0..cnt).map(|r| {
            let l = Lwe { a: w[r * row..r * row + nb].iter().map(|&x| x as u64).collect(), b: w[r * row + nb] as u64 };
            centered(l.phase(&s_br, p0), p0) as f64 * 4.0 / p0 as f64
        }).collect()
    };
    // the input noise model of a lookup (m units, period 4): the mod switch's rounding to 2 nu n_br
    // ((n_lwe / 2 + 1) / 12) and the key switch s_br -> s_lwe (n_br sum E d^2 sigma^2, NAF digits)
    let ms_sd = ((params.n_lwe as f64 / 2.0 + 1.0) / 12.0).sqrt() * 4.0 / (2 * nu * nb) as f64;
    let ks_sd = (nb as f64 * 5.44).sqrt() * params.lwe_sigma_rel * 4.0;
    let sd = (ms_sd * ms_sd + ks_sd * ks_sd).sqrt();
    let pf = |half: f64| pc_tfhe::pack::log2_tail(half / sd);
    println!("[pc-check] tfhe-bitwise: lookup input noise sd {sd:.2e} (mod switch {ms_sd:.2e}, key switch {ks_sd:.2e}); p_fail per lookup: 8-bin tables 2^{:.1}, 16-bin 2^{:.1}", pf(0.125), pf(0.0625));
    let mut t = Tally::default();
    let lt_ = &blt.inner;
    // a native program step: LWE_LIN (dst, a, b, index pairs, rows, ca, cb, c) / PBS (src, rows, table, dst)
    #[derive(Clone, Copy)]
    enum St {
        Lin(*mut c_void, *const c_void, *const c_void, *const c_void, usize, u64, u64, u64),
        Pbs(*const c_void, usize, usize, *mut c_void),
    }
    let mut idx_off = 0usize;
    let mut upload_pairs = |d: &mut pc_cuda::DBuf<'_, pc_cuda::Compute, u32>, pairs: &[(i64, i64)], off: &mut usize| -> Res<*const c_void> {
        let w: Vec<u32> = pairs.iter().flat_map(|&(a, b)| [a as i32 as u32, b as i32 as u32]).collect();
        d.upload_at(s, *off, &w)?;
        let p = (d.vptr() as *const u32).wrapping_add(*off) as *const c_void;
        *off += w.len();
        Ok(p)
    };
    for op in &ops {
        for &nw in &words {
            let rr = nw * nd;
            let mut rng = test_prng(0x5ee0 + nw as u32);
            let a: Vec<u64> = (0..nw).map(|_| rng.uniform_below(1 << (2 * nd.min(31)))).collect();
            let b: Vec<u64> = (0..nw).map(|i| if i % 4 == 1 { a[i] } else { rng.uniform_below(1 << (2 * nd.min(31))) }).collect();
            let sel: Vec<u64> = (0..nw).map(|_| rng.uniform_below(2)).collect();
            let dig = |x: &[u64]| -> Vec<f64> { x.iter().flat_map(|&v| (0..nd).map(move |i| ((v >> (2 * i)) & 3) as f64 / 8.0)).collect() };
            bufs[0].upload_at(s, 0, &encrypt(&dig(&a), &mut rng))?;
            bufs[1].upload_at(s, 0, &encrypt(&dig(&b), &mut rng))?;
            bufs[2].upload_at(s, 0, &encrypt(&sel.iter().map(|&x| x as f64 / 8.0).collect::<Vec<_>>(), &mut rng))?;
            idx_off = 0;
            let all: Vec<(i64, i64)> = (0..rr as i64).map(|r| (r, r)).collect();
            let p_all = upload_pairs(&mut d_idx, &all, &mut idx_off)?;
            let bp = |b: &pc_cuda::DBuf<'_, pc_cuda::Compute, u32>| b.vptr();
            let bpm = |b: &pc_cuda::DBuf<'_, pc_cuda::Compute, u32>| b.vptr() as *mut c_void;
            let (b0, b1, b2, b3, b4, b5, b6, b7) = (bp(&bufs[0]), bp(&bufs[1]), bp(&bufs[2]), bpm(&bufs[3]), bpm(&bufs[4]), bpm(&bufs[5]), bpm(&bufs[6]), bpm(&bufs[7]));
            // the program of `op` as a list of launches (prepared index pairs), run `reps` times
            let (rounds, out_buf, out_rows): (usize, *mut c_void, usize);
            let mut prog: Vec<St> = Vec::new();
            // the new prefix programs: PBS count, the final lookups' outputs (+-1/16: their noise), and per round the
            // lookup's half bin and its input's weighted noise (sum of squares; negative: weights on PBS outputs)
            let (mut npbs, mut final_buf, mut final_rows) = (0usize, std::ptr::null::<c_void>(), 0usize);
            let mut specs: Vec<(f64, f64)> = Vec::new();
            let sig_in = params.lwe_sigma_rel * 4.0;
            match op.as_str() {
                "add" => {
                    // X = 2 a + 2 b - 3/4, S = sgn(X)
                    let c = enc_v(-0.75);
                    prog.push(St::Lin(b3, b0, b1, p_all, rr, 2, 2, c));
                    prog.push(St::Pbs(b3, rr, 0, b4));
                    let mut d = 1;
                    let (mut cur, mut nxt) = (b4, b5);
                    while d < nd {
                        let pairs: Vec<(i64, i64)> = (0..rr as i64).map(|r| (r, if (r as usize) % nd >= d { r - d as i64 } else { -1 })).collect();
                        let pp = upload_pairs(&mut d_idx, &pairs, &mut idx_off)?;
                        let last = 2 * d >= nd;
                        let (src, dst) = (cur, nxt);
                        prog.push(St::Lin(b6, src, src, pp, rr, 2, 1, 0));
                        prog.push(St::Pbs(b6, rr, if last { 1 } else { 0 }, dst));
                        (cur, nxt) = (nxt, cur);
                        d *= 2;
                    }
                    // Z = X + 2 c_(i - 1) - 1/8, out = (s + c) mod 4
                    let pairs: Vec<(i64, i64)> = (0..rr as i64).map(|r| (r, if (r as usize) % nd > 0 { r - 1 } else { -1 })).collect();
                    let pp = upload_pairs(&mut d_idx, &pairs, &mut idx_off)?;
                    let (cc, c8) = (cur, enc_v(-0.125));
                    prog.push(St::Lin(b6, b3, cc, pp, rr, 1, 2, c8));
                    prog.push(St::Pbs(b6, rr, 2, nxt));
                    (rounds, out_buf, out_rows) = (2 + nd.trailing_zeros() as usize, nxt, rr);
                }
                "cmp" => {
                    prog.push(St::Lin(b3, b0, b1, p_all, rr, 2, p0 - 2, 0));
                    prog.push(St::Pbs(b3, rr, 0, b4));
                    let (mut cur, mut nxt, mut k) = (b4, b5, nd);
                    while k > 1 {
                        let h = k / 2;
                        let pairs: Vec<(i64, i64)> = (0..nw * h).map(|r| ((r / h * k + 2 * (r % h) + 1) as i64, (r / h * k + 2 * (r % h)) as i64)).collect();
                        let pp = upload_pairs(&mut d_idx, &pairs, &mut idx_off)?;
                        let (src, dst, cnt, last) = (cur, nxt, nw * h, h == 1);
                        prog.push(St::Lin(b6, src, src, pp, cnt, 2, 1, 0));
                        prog.push(St::Pbs(b6, cnt, if last { 3 } else { 0 }, dst));
                        (cur, nxt) = (nxt, cur);
                        k = h;
                    }
                    (rounds, out_buf, out_rows) = (1 + nd.trailing_zeros() as usize, cur, nw);
                }
                "mux" => {
                    // T = 2 a - 2 b; Y = T + 16 sel (2 sel: the half turn); o = f(Y); out = a + b + 2 o
                    let ps: Vec<(i64, i64)> = (0..rr as i64).map(|r| (r, r / nd as i64)).collect();
                    let pp = upload_pairs(&mut d_idx, &ps, &mut idx_off)?;
                    prog.push(St::Lin(b3, b0, b1, p_all, rr, 2, p0 - 2, 0));
                    prog.push(St::Lin(b6, b3, b2, pp, rr, 1, 16, 0));
                    prog.push(St::Pbs(b6, rr, 4, b4));
                    prog.push(St::Lin(b5, b0, b1, p_all, rr, 1, 1, 0));
                    prog.push(St::Lin(b6, b5, b4, p_all, rr, 1, 2, 0));
                    (rounds, out_buf, out_rows) = (1, b6, rr);
                }
                "add3" | "addp" | "cmp3" | "cmpp" => {
                    // fan-in f prefix rounds on the whole torus: x = sum_k 2^(f - 1 - k) S_k (S in {0, +-u}), the
                    // status of the first non-propagating input is sign(x) (f 3 at u 1/4 / f 4 at u 1/8: x in
                    // (-2, 2), margin u / 2). add3 / cmp3: digit statuses (u 1/4), fan-in 3 (nu 1); addp / cmpp:
                    // statuses of digit pairs (u 1/8: 4 s_j + s_(j - 1)), fan-in 4 (margin 1/16: nu 2). add: the
                    // output digits are linear in the carries, a_i + b_i + c_(i - 1) - 4 c_i (no output round).
                    let (pair, is_add) = (op.ends_with('p'), op.starts_with("add"));
                    let fmax = if pair { 4 } else { 3 };
                    let ki = |k: i64| -> u64 { k.rem_euclid(p0 as i64) as u64 };
                    let tab = |uin: usize, uout: usize| -> usize {
                        match (uin, uout) {
                            (4, 4) => 0,
                            (4, 8) => 12,
                            (8, 8) => 8,
                            (8, 4) => 9,
                            (4, _) => if is_add { 6 } else { 7 },
                            _ => if is_add { 10 } else { 11 },
                        }
                    };
                    let (s0, k0) = if pair { (2, nd / 2) } else { (1, nd) };
                    let mut fans = Vec::new();
                    if is_add {
                        let mut sp = s0;
                        while sp < nd {
                            let f = fmax.min(nd.div_ceil(sp));
                            fans.push(f);
                            sp *= f;
                        }
                    } else {
                        let mut k = k0;
                        while k > 1 {
                            let f = fmax.min(k);
                            fans.push(f);
                            k = k.div_ceil(f);
                        }
                    }
                    let unit = |f: usize| if f == 4 { 8 } else { 4 };
                    let u1 = if pair { 8 } else { 4 };
                    let id_pairs = |c: usize| -> Vec<(i64, i64)> { (0..c as i64).map(|r| (r, r)).collect() };
                    let rows1 = if pair && !is_add { rr / 2 } else { rr };
                    if pair {
                        // x = (4 s_j + s_(j - 1) - 15) / 8 (add, every j; s_(-1) = 0: no carry in) / (4 d_j + d_(j - 1)) / 8
                        // (cmp, odd j)
                        let (hi, lo): (Vec<i64>, Vec<i64>) = if is_add {
                            ((0..rr as i64).collect(), (0..rr as i64).map(|r| if r as usize % nd > 0 { r - 1 } else { -1 }).collect())
                        } else {
                            ((0..rows1 as i64).map(|q| 2 * q + 1).collect(), (0..rows1 as i64).map(|q| 2 * q).collect())
                        };
                        let ph: Vec<(i64, i64)> = hi.iter().map(|&j| (j, j)).collect();
                        let pl: Vec<(i64, i64)> = lo.iter().map(|&j| (j, j)).collect();
                        let pph = upload_pairs(&mut d_idx, &ph, &mut idx_off)?;
                        let ppl = upload_pairs(&mut d_idx, &pl, &mut idx_off)?;
                        let pid = upload_pairs(&mut d_idx, &id_pairs(rows1), &mut idx_off)?;
                        let sb = if is_add { 1 } else { -1 };
                        prog.push(St::Lin(b6, b0, b1, pph, rows1, 4, ki(4 * sb), 0));
                        prog.push(St::Lin(b7, b0, b1, ppl, rows1, 1, ki(sb), 0));
                        prog.push(St::Lin(b3, b6 as *const c_void, b7 as *const c_void, pid, rows1, 1, 1, if is_add { enc_v(-15.0 / 8.0) } else { 0 }));
                        specs.push((1.0 / 16.0, 34.0 * sig_in * sig_in));
                    } else {
                        let (cb_, c) = if is_add { (2, enc_v(-0.75)) } else { (p0 - 2, 0) };
                        prog.push(St::Lin(b3, b0, b1, p_all, rr, 2, cb_, c));
                        specs.push((0.125, 8.0 * sig_in * sig_in));
                    }
                    let first_out = fans.first().map_or(0, |&f| unit(f));
                    prog.push(St::Pbs(b3, rows1, tab(u1, first_out), b4));
                    npbs += rows1;
                    let (mut cur, mut nxt) = (b4, b5);
                    let (mut uin, mut span, mut k) = (first_out, s0, k0);
                    for (ri, &f) in fans.iter().enumerate() {
                        let uo = fans.get(ri + 1).map_or(0, |&g| unit(g));
                        let w: Vec<u64> = (0..f).map(|x| 1u64 << (f - 1 - x)).collect();
                        // per output row the f source rows, the most significant first (-1: none, a P)
                        let (cnt, src): (usize, Vec<Vec<i64>>) = if is_add {
                            (rr, (0..rr as i64).map(|r| (0..f).map(|x| if r as usize % nd >= x * span { r - (x * span) as i64 } else { -1 }).collect()).collect())
                        } else {
                            let h = k.div_ceil(f);
                            (nw * h, (0..nw * h).map(|q| {
                                let (wd, g) = (q / h, q % h);
                                (0..f).map(|x| { let e = g * f + f - 1 - x; if e < k { (wd * k + e) as i64 } else { -1 } }).collect()
                            }).collect())
                        };
                        let p01: Vec<(i64, i64)> = src.iter().map(|v| (v[0], if f > 1 { v[1] } else { -1 })).collect();
                        let pp = upload_pairs(&mut d_idx, &p01, &mut idx_off)?;
                        let cu = cur as *const c_void;
                        if f <= 2 {
                            prog.push(St::Lin(b3, cu, cu, pp, cnt, w[0], if f > 1 { w[1] } else { 0 }, 0));
                        } else {
                            prog.push(St::Lin(b6, cu, cu, pp, cnt, w[0], w[1], 0));
                            let p23: Vec<(i64, i64)> = src.iter().map(|v| (v[2], if f > 3 { v[3] } else { -1 })).collect();
                            let pp2 = upload_pairs(&mut d_idx, &p23, &mut idx_off)?;
                            prog.push(St::Lin(b7, cu, cu, pp2, cnt, w[2], if f > 3 { w[3] } else { 0 }, 0));
                            let pid = upload_pairs(&mut d_idx, &id_pairs(cnt), &mut idx_off)?;
                            prog.push(St::Lin(b3, b6 as *const c_void, b7 as *const c_void, pid, cnt, 1, 1, 0));
                        }
                        prog.push(St::Pbs(b3, cnt, tab(uin, uo), nxt));
                        npbs += cnt;
                        specs.push((if uin == 8 { 1.0 / 16.0 } else { 0.125 }, w.iter().map(|&x| (x * x) as f64).sum::<f64>() * -1.0));
                        (cur, nxt) = (nxt, cur);
                        uin = uo;
                        span *= f;
                        k = k.div_ceil(f);
                    }
                    final_buf = cur as *const c_void;
                    if is_add {
                        // c = g + 1/16 in {0, 1/8}; out = a + b + c_(i - 1) - 4 c_i
                        let pprev: Vec<(i64, i64)> = (0..rr as i64).map(|r| (r, if r as usize % nd > 0 { r - 1 } else { -1 })).collect();
                        let pv = upload_pairs(&mut d_idx, &pprev, &mut idx_off)?;
                        prog.push(St::Lin(b6, cur as *const c_void, cur as *const c_void, p_all, rr, 1, 0, enc_v(0.0625)));
                        prog.push(St::Lin(b7, b0, b1, p_all, rr, 1, 1, 0));
                        prog.push(St::Lin(b3, b7 as *const c_void, b6 as *const c_void, pv, rr, 1, 1, 0));
                        prog.push(St::Lin(nxt, b3 as *const c_void, b6 as *const c_void, p_all, rr, 1, ki(-4), 0));
                        (rounds, out_buf, out_rows) = (1 + fans.len(), nxt, rr);
                    } else {
                        let pid = upload_pairs(&mut d_idx, &id_pairs(nw), &mut idx_off)?;
                        prog.push(St::Lin(nxt, cur as *const c_void, cur as *const c_void, pid, nw, 1, 0, enc_v(0.0625)));
                        (rounds, out_buf, out_rows) = (1 + fans.len(), nxt, nw);
                    }
                    final_rows = if is_add { rr } else { nw };
                }
                "xor" => {
                    let c = enc_v(-15.0 / 16.0);
                    prog.push(St::Lin(b3, b0, b1, p_all, rr, 1, 4, c));
                    prog.push(St::Pbs(b3, rr, 5, b4));
                    (rounds, out_buf, out_rows) = (1, b4, rr);
                }
                o => return Err(format!("--ops: {o}?").into()),
            }
            let mut go = || -> Res<f64> {
                let td = Instant::now();
                for st in &prog {
                    match *st {
                        St::Lin(dst, a, b, idx, cnt, ca, cb, c) => ts.native_lin_c(&ctx, s, dst, a, b, idx, cnt, ca, cb, c)?,
                        St::Pbs(src, cnt, tv, dst) => ts.native_pbs(&ctx, s, lt_, src, cnt, tvp(tv), &[0], &[dst])?,
                    }
                }
                s.sync()?;
                Ok(td.elapsed().as_secs_f64())
            };
            go()?;
            let mut tm: Vec<f64> = (0..reps).map(|_| go()).collect::<Res<_>>()?;
            tm.sort_by(f64::total_cmp);
            let med = tm[reps / 2];
            // the output buffer's rows (a raw pointer into one of `bufs`)
            let k = bufs.iter().position(|b| b.vptr() as *mut c_void == out_buf).ok_or("output buffer")?;
            let got = decrypt(&bufs[k].download_range(s, 0, out_rows * row)?, out_rows);
            let mask = if 2 * nd >= 64 { u64::MAX } else { (1u64 << (2 * nd)) - 1 };
            let base = if op.starts_with("add") { "add" } else if op.starts_with("cmp") { "cmp" } else { op.as_str() };
            let (want, scale): (Vec<u64>, f64) = match base {
                "add" => (a.iter().zip(&b).map(|(x, y)| (x + y) & mask).collect(), 8.0),
                "cmp" => (a.iter().zip(&b).map(|(x, y)| (x < y) as u64).collect(), 8.0),
                "mux" => ((0..nw).map(|i| if sel[i] == 1 { a[i] } else { b[i] }).collect(), 4.0),
                _ => (a.iter().zip(&b).map(|(x, y)| x ^ y).collect(), 8.0),
            };
            let (mut wrong, mut dev_max) = (0usize, 0f64);
            for w in 0..nw {
                let v: u64 = if base == "cmp" {
                    (got[w] * scale).round() as u64
                } else {
                    (0..nd).map(|i| (((got[w * nd + i] * scale).round() as i64).rem_euclid(4) as u64) << (2 * i)).sum()
                };
                wrong += (v != want[w]) as usize;
                for i in 0..if base == "cmp" { 1 } else { nd } {
                    let x = got[if base == "cmp" { w } else { w * nd + i }] * scale;
                    dev_max = dev_max.max((x - x.round()).abs());
                }
            }
            t.add(&format!("tfhe-bitwise: {op}{} x {nw} words, exact", 2 * nd), wrong, nw);
            let pbs: usize = match op.as_str() {
                _ if npbs > 0 => npbs,
                "add" => rr * (2 + nd.trailing_zeros() as usize),
                "cmp" => rr + nw * (nd - 1),
                _ => rr,
            };
            if !final_buf.is_null() {
                // the PBS output noise (the final lookups' +-1/16), then p_fail per lookup of every round's input
                let k = bufs.iter().position(|b| b.vptr() == final_buf).ok_or("final buffer")?;
                let g = decrypt(&bufs[k].download_range(s, 0, final_rows * row)?, final_rows);
                let sp = (g.iter().map(|&x| { let e = x.abs() - 0.0625; e * e }).sum::<f64>() / g.len() as f64).sqrt();
                let worst = specs.iter().map(|&(half, w)| {
                    let v = if w < 0.0 { -w * sp * sp } else { w };
                    pc_tfhe::pack::log2_tail(half / (sd * sd + v).sqrt())
                }).fold(f64::NEG_INFINITY, f64::max);
                println!("[pc-check] tfhe-bitwise: {op}{}: PBS output noise rms {sp:.2e}; worst p_fail per lookup 2^{worst:.1} (rounds' half bins {:?})", 2 * nd, specs.iter().map(|x| x.0).collect::<Vec<_>>());
            }
            println!("[pc-check] tfhe-bitwise: {op}{}: {nw} words: {:.3} ms (median of {reps}), {rounds} PBS rounds, {pbs} PBS; {:.0} words/s, {:.0} PBS/s; max output dev {dev_max:.2e} (digit units)", 2 * nd, med * 1e3, nw as f64 / med, pbs as f64 / med);
        }
    }
    println!("[pc-check] tfhe-bitwise: {} gates, {} failing", t.gates, t.bad);
    Ok(t.bad == 0)
}

/// `pc-check tfhe-prog --cases FILE [--reps 7] [--lat-group 5] [--nu 1]`: native TFHE programs (perseus impl/tfhe.py
/// `Prog` JSON, as `python -m perseus.impl.tfhe_bitwise --cases FILE` writes them with their test words) on the device's
/// native ops: the loaded values (what the program's load leaves: native LWEs (s_br, p0) at value v) encrypted, every
/// lin (`TfheSwitch::native_lin_c`) and pbs (`native_pbs`: key switch s_br -> s_lwe and the PBS) run as the session
/// runs them, the store's table as one more pbs (a warp table's values read as native ones), the stored rows decrypted
/// against the case's exact rows. Latency (device, synced, median of `--reps`): the program up to the store (the op;
/// in a session the store's lookup rides the finish) and with it; the store outputs' noise (rms off the exact value).
pub fn cmd_prog(args: &[String]) -> ExitCode {
    match run_prog(args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("pc-check tfhe-prog: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_prog(args: &[String]) -> Res<bool> {
    use pc_core::keys::DiscreteGaussian;
    use pc_tfhe::lwe::Lwe;
    use serde_json::Value;
    use std::ffi::c_void;
    let path = arg(args, "--cases").ok_or("--cases FILE")?;
    let reps: usize = arg(args, "--reps").map(|s| s.parse()).transpose()?.unwrap_or(7).max(1);
    let cases: Vec<Value> = std::fs::read_to_string(&path)?.lines().filter(|l| !l.trim().is_empty()).map(serde_json::from_str).collect::<std::result::Result<_, _>>()?;
    if cases.is_empty() {
        return Err("no cases".into());
    }
    let u = |v: &Value, k: &str| -> Res<usize> { Ok(v.get(k).and_then(Value::as_u64).ok_or_else(|| format!("case without {k}"))? as usize) };
    let mut rows = 0usize;
    let (mut nbuf, mut nidx, mut ntab, mut nmacc) = (0usize, 0usize, 0usize, 0usize);
    for c in &cases {
        rows = rows.max(u(c, "rows")?);
        nbuf = nbuf.max(u(c, "bufs")? + 1);
        let pr = &c["prog"];
        let steps = pr["steps"].as_array().ok_or("prog without steps")?;
        nidx = nidx.max(steps.iter().filter_map(|st| st.get("idx").and_then(Value::as_array)).map(|a| a.len()).sum());
        ntab = ntab.max(pr["luts"].as_object().map_or(0, |o| o.values().map(|l| l["tables"].as_array().map_or(0, |t| t.len())).sum()));
        nmacc = nmacc.max(steps.iter().filter(|st| st["op"] == "pbsm").map(|st| st["rows"].as_u64().unwrap_or(0) as usize).sum());
    }
    let chain = Chain::generate(&ChainSpec::n32())?;
    let n = chain.n;
    let mut params = switch_params(args, n)?;
    params.chain = true;
    let (nb, nu) = (params.n_br, params.nu);
    let row = nb + 1;
    let tvl = params.tv_len();
    let dev = Device::open()?;
    let blt = pc_cuda::blas::Lt::new(&dev)?;
    let cfg = MemCfg { k_max: 1, pool: vec![PoolClass { k: 1, count: 1 }], events: 16, scratch_cts: 2 };
    let mut mp = MemPlan::build::<W, M>(&chain, cfg)?;
    // the switch's capacity: one pass of at most 32768 LWEs (a longer pbs runs in passes: the kernels' grid y)
    let cap = rows.min(1 << 15);
    mp.reserve("tfhe", pc_gpu::tfhe::TfheSwitch::bytes(n, &params, cap));
    mp.reserve("prog", 4 * (nbuf * rows * row + nidx + ntab * 2 * tvl + nmacc * 4 * nb) + (8 << 20));
    let arena = mp.allocate(&dev)?;
    let ex = Executor::<W>::new(&dev, &arena, &mp)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &mp)?;
    let s = &ex.compute;
    let mut prng = test_prng(0xb7);
    let sk = SecretKey::sample(&mut prng, n)?;
    let sv: Vec<i64> = sk.s.iter().map(|&v| v as i64).collect();
    let (p0, p1) = (chain.q[0], chain.q[1]);
    let tk = Instant::now();
    let (sw, _s_lwe, s_br) = Switcher::generate_keep(params.clone(), &sv, p0, p1, &mut prng)?;
    println!("[pc-check] tfhe-prog: host keys {:.1} s (n_lwe {}, n_br {nb}, nu {nu}, lat_group {}); {} cases, {nbuf} buffers of {rows} rows", tk.elapsed().as_secs_f64(), params.n_lwe, params.lat_group, cases.len());
    let mut ts = pc_gpu::tfhe::TfheSwitch::new(&arena, &dev, s, &sw, cap)?;
    let mut bufs: Vec<pc_cuda::DBuf<'_, pc_cuda::Compute, u32>> = (0..nbuf).map(|_| arena.alloc::<pc_cuda::Compute, u32>("prog", rows * row)).collect::<std::result::Result<_, _>>()?;
    let mut d_idx = arena.alloc::<pc_cuda::Compute, u32>("prog", nidx.max(2))?;
    let mut d_tv = arena.alloc::<pc_cuda::Compute, u32>("prog", ntab.max(1) * 2 * tvl)?;
    // pbsm: one trivial accumulator per row [row][2 part][2 prime][n_br] (part 0 the row's table, part 1 zero)
    let mut d_macc = arena.alloc::<pc_cuda::Compute, u32>("prog", nmacc.max(1) * 4 * nb)?;
    arena.freeze();
    let modp = |x: i64| x.rem_euclid(p0 as i64) as u64;
    let enc_v = |v: f64| modp((v * p0 as f64 / 4.0).round() as i64);
    let dg = DiscreteGaussian::new(params.lwe_sigma_rel * p0 as f64)?;
    let decrypt = |w: &[u32], cnt: usize| -> Vec<f64> {
        (0..cnt).map(|r| {
            let l = Lwe { a: w[r * row..r * row + nb].iter().map(|&x| x as u64).collect(), b: w[r * row + nb] as u64 };
            centered(l.phase(&s_br, p0), p0) as f64 * 4.0 / p0 as f64
        }).collect()
    };
    #[derive(Clone, Copy)]
    enum St {
        Lin(usize, usize, usize, usize, usize, u64, u64, u64),
        Pbs(usize, usize, usize, [usize; 4], usize),
        PbsM(usize, usize, usize, usize),
    }
    let mut t = Tally::default();
    let lt_ = &blt.inner;
    for c in &cases {
        let name = c["name"].as_str().unwrap_or("?").to_string();
        let pr = &c["prog"];
        // the tables: native values on the grid of tvl points (v at phase v q / 4, the upper half negated)
        let q = params.q_br;
        let mut tab_ix: std::collections::BTreeMap<String, usize> = Default::default();
        let mut tab_k: std::collections::BTreeMap<String, usize> = Default::default();
        let mut tvw: Vec<u32> = Vec::new();
        for (ln, l) in pr["luts"].as_object().ok_or("luts")? {
            let tables = l["tables"].as_array().ok_or("tables")?;
            // k interleaved tables (many-LUT: one rotation, k outputs), each on tvl / k points
            let kt = tables.len();
            if kt == 0 || tvl % kt != 0 {
                return Err(format!("{name}: lut {ln}: {kt} interleaved tables").into());
            }
            let sub = tvl / kt;
            let mut tvs: Vec<Vec<u64>> = Vec::new();
            for t in tables {
                let vals: Vec<f64> = t.as_array().ok_or("table")?.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
                if vals.len() != sub || vals.iter().any(|x| !x.is_finite() || x.abs() > 2.0) {
                    return Err(format!("{name}: lut {ln}: {} points (want {sub}), |v| <= 2", vals.len()).into());
                }
                tvs.push(vals.iter().enumerate().map(|(j, &x)| {
                    let y = (q as f64 * x / 4.0).round() as i64;
                    (if j < sub / 2 { y } else { -y }).rem_euclid(q as i64) as u64
                }).collect());
            }
            let tv = pc_tfhe::pbs::interleave(&tvs);
            tab_k.insert(ln.clone(), kt);
            tab_ix.insert(ln.clone(), tvw.len() / (2 * tvl));
            tvw.extend(sw.bsk.rns.residues(&tv).iter().flat_map(|r| r.iter().map(|&x| x as u32)));
        }
        d_tv.upload(s, &tvw)?;
        let tvp = |k: usize| (d_tv.vptr() as *const u32).wrapping_add(k * 2 * tvl) as *const c_void;
        let steps = pr["steps"].as_array().ok_or("steps")?;
        let mut prog: Vec<St> = Vec::new();
        let (mut load_dst, mut store) = (None, None);
        let mut idx_off = 0usize;
        let (mut npbs, mut launches) = (0usize, 0usize);
        let mut macc: Vec<u32> = Vec::new();
        for st in steps {
            let b = |k: &str| -> Res<usize> { u(st, k) };
            match st["op"].as_str().unwrap_or("") {
                "load" => load_dst = Some(st["dst"].as_u64().ok_or("load: one buffer")? as usize),
                "lin" => {
                    let idx: Vec<i64> = st["idx"].as_array().ok_or("idx")?.iter().map(|x| x.as_i64().unwrap_or(-2)).collect();
                    let r = idx.len() / 2;
                    if idx.iter().any(|&i| i < -1 || i >= rows as i64) {
                        return Err(format!("{name}: lin index outside the buffers").into());
                    }
                    let w: Vec<u32> = idx.iter().map(|&i| i as i32 as u32).collect();
                    d_idx.upload_at(s, idx_off, &w)?;
                    let cf = st["c"].as_f64().unwrap_or(0.0);
                    prog.push(St::Lin(b("dst")?, b("a")?, b("b")?, idx_off, r, modp(st["ca"].as_i64().unwrap_or(0)), modp(st["cb"].as_i64().unwrap_or(0)), enc_v(cf)));
                    idx_off += w.len();
                }
                "pbs" => {
                    let dst = st["dst"].as_array().ok_or("pbs dst")?;
                    let lut = st["lut"].as_str().ok_or("lut")?;
                    if dst.len() != tab_k[lut] || dst.len() > 4 {
                        return Err(format!("{name}: pbs with {} outputs on {} tables", dst.len(), tab_k[lut]).into());
                    }
                    let r = b("rows")?;
                    let mut d4 = [0usize; 4];
                    for (j, d) in dst.iter().enumerate() {
                        d4[j] = d.as_u64().ok_or("dst")? as usize;
                    }
                    prog.push(St::Pbs(b("src")?, r, tab_ix[lut], d4, dst.len()));
                    npbs += r;
                    launches += 1;
                }
                "pbsm" => {
                    if nu != 1 {
                        return Err(format!("{name}: pbsm runs at nu 1").into());
                    }
                    let ls = st["luts"].as_array().ok_or("pbsm luts")?;
                    let off = macc.len() / (4 * nb);
                    for l in ls {
                        let ln = l.as_str().ok_or("pbsm lut")?;
                        if tab_k[ln] != 1 {
                            return Err(format!("{name}: pbsm with a {}-table lut", tab_k[ln]).into());
                        }
                        let k = tab_ix[ln];
                        macc.extend_from_slice(&tvw[k * 2 * tvl..(k + 1) * 2 * tvl]);
                        macc.extend(std::iter::repeat_n(0u32, 2 * nb));
                    }
                    let r = ls.len();
                    prog.push(St::PbsM(b("src")?, r, off, b("dst")?));
                    npbs += r;
                    launches += 1;
                }
                "store" => store = Some((b("src")?, b("rows")?, st["lut"].as_str().map(|l| tab_ix[l]))),
                o => return Err(format!("{name}: step {o}").into()),
            }
        }
        let (Some(ld), Some((ssrc, srows, slut))) = (load_dst, store) else { return Err(format!("{name}: a program loads and stores").into()) };
        if !macc.is_empty() {
            d_macc.upload(s, &macc)?;
        }
        let maccp = |off: usize| (d_macc.vptr() as *const u32).wrapping_add(off * 4 * nb) as *const c_void;
        let m_in: Vec<f64> = c["m_in"].as_array().ok_or("m_in")?.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect();
        let mut rng = test_prng(0x9e0);
        let enc: Vec<u32> = m_in.iter().flat_map(|&v| {
            let l = Lwe::encrypt(enc_v(v), &s_br, p0, &dg, &mut rng);
            l.a.into_iter().chain(std::iter::once(l.b)).map(|x| x as u32).collect::<Vec<_>>()
        }).collect();
        let scratch = nbuf - 1;
        let bp = |b: &pc_cuda::DBuf<'_, pc_cuda::Compute, u32>| b.vptr() as *mut c_void;
        let ptrs: Vec<*mut c_void> = bufs.iter().map(bp).collect();
        let idxp = |off: usize| (d_idx.vptr() as *const u32).wrapping_add(off) as *const c_void;
        let mut go = |bufs: &mut Vec<pc_cuda::DBuf<'_, pc_cuda::Compute, u32>>| -> Res<(f64, f64)> {
            bufs[ld].upload_at(s, 0, &enc)?;
            s.sync()?;
            let td = Instant::now();
            for st in &prog {
                match *st {
                    St::Lin(d, a, b2, off, r, ca, cb, cc) => ts.native_lin_c(&ctx, s, ptrs[d], ptrs[a], ptrs[b2], idxp(off), r, ca, cb, cc)?,
                    St::PbsM(src, r, off, d) => ts.native_plut_each(&ctx, s, lt_, ptrs[src], r, maccp(off), &[0], &[ptrs[d]])?,
                    St::Pbs(src, r, k, d, kl) => {
                        let dsts: Vec<*mut c_void> = d[..kl].iter().map(|&x| ptrs[x]).collect();
                        ts.native_pbs(&ctx, s, lt_, ptrs[src], r, tvp(k), &vec![0u64; kl], &dsts)?
                    }
                }
            }
            s.sync()?;
            let body = td.elapsed().as_secs_f64();
            if let Some(k) = slut {
                ts.native_pbs(&ctx, s, lt_, ptrs[ssrc], srows, tvp(k), &[0], &[ptrs[scratch]])?;
            }
            s.sync()?;
            Ok((body, td.elapsed().as_secs_f64()))
        };
        go(&mut bufs)?;
        let mut tm: Vec<(f64, f64)> = (0..reps).map(|_| go(&mut bufs)).collect::<Res<_>>()?;
        let med = |v: &mut Vec<f64>| -> f64 { v.sort_by(f64::total_cmp); v[v.len() / 2] };
        let body = med(&mut tm.iter().map(|x| x.0).collect());
        let total = med(&mut tm.iter().map(|x| x.1).collect());
        tm.clear();
        let ob = if slut.is_some() { scratch } else { ssrc };
        let got = decrypt(&bufs[ob].download_range(s, 0, srows * row)?, srows);
        let want: Vec<i64> = c["want"].as_array().ok_or("want")?.iter().map(|x| x.as_i64().unwrap_or(-1)).collect();
        let scale = c["scale"].as_f64().unwrap_or(1.0);
        let words = u(c, "words")?;
        let ndig = u(c, "ndig")?;
        let nfl = u(c, "nflag")?;
        let (mut wrong_w, mut wrong_r, mut ss) = (vec![false; words], 0usize, 0f64);
        for (r, &w) in want.iter().enumerate() {
            let x = got[r] * scale;
            let mut v = x.round() as i64;
            if r < words * ndig {
                v = v.rem_euclid(4);
            }
            ss += (x - x.round()) * (x - x.round()) / (scale * scale);
            if v != w {
                wrong_r += 1;
                let word = if r < words * ndig { r / ndig.max(1) } else { (r - words * ndig) % words.max(1) };
                wrong_w[word.min(words - 1)] = true;
            }
        }
        let _ = nfl;
        let wrong = wrong_w.iter().filter(|&&x| x).count();
        t.add(&format!("tfhe-prog: {name}, exact (rows wrong {wrong_r})"), wrong, words);
        let an = &c["analysis"];
        println!(
            "[pc-check] tfhe-prog: {name}: op {:.3} ms, with the store {:.3} ms (median of {reps}); {} rounds, {launches} launches, {npbs} PBS; {:.0} words/s; store output noise rms {:.2e}; p_fail (model) 2^{:.1} / lookup, 2^{:.1} / op",
            body * 1e3,
            total * 1e3,
            an["rounds"].as_u64().unwrap_or(0),
            words as f64 / body.max(1e-9),
            (ss / want.len().max(1) as f64).sqrt(),
            an["log2_pfail_lookup"].as_f64().unwrap_or(f64::NAN),
            an["log2_pfail_op"].as_f64().unwrap_or(f64::NAN)
        );
    }
    println!("[pc-check] tfhe-prog: {} gates, {} failing", t.gates, t.bad);
    Ok(t.bad == 0)
}

/// `pc-check tfhe-tree [--entries 16,64,256,1024,4096] [--lookups 1,16] [--reps 5] [--lat-group 5]`: encrypted
/// lookup tables of t > 8 entries as a selection tree of private lookups (`TfheSwitch::native_pack` /
/// `native_plut_each`): the index as base-8 digits (native LWEs at table_index(d, B)), the table's entries (native
/// LWEs, 4-bit values) packed 8 per leaf accumulator once (the table's preparation); level l rotates every group
/// of the level by digit l in one batch (one accumulator per LWE), the outputs packed 8 per accumulator for the
/// next level: t / 8 + t / 64 + ... rotations and t / 64 + ... packs per lookup, ceil(log8 t) rotation levels.
/// Every output decrypted against its entry; p_fail per lookup = the levels' index decisions (each at most 8 entries
/// per 2048 coefficients: pc_tfhe::pack::index_log2_fail) + the output decision.
pub fn cmd_tree(args: &[String]) -> ExitCode {
    match run_tree(args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("pc-check tfhe-tree: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_tree(args: &[String]) -> Res<bool> {
    use pc_core::keys::DiscreteGaussian;
    use pc_tfhe::lwe::Lwe;
    use std::ffi::c_void;
    let list = |k: &str, d: &str| -> Res<Vec<usize>> { Ok(arg(args, k).unwrap_or_else(|| d.into()).split(',').map(|x| x.trim().parse()).collect::<std::result::Result<_, _>>()?) };
    let ents = list("--entries", "16,64,256,1024,4096")?;
    let looks = list("--lookups", "1,16")?;
    let reps: usize = arg(args, "--reps").map(|s| s.parse()).transpose()?.unwrap_or(5).max(1);
    let bases = |t: usize| -> Res<Vec<usize>> {
        if t < 2 || !t.is_power_of_two() {
            return Err(format!("--entries {t}: a power of two >= 2").into());
        }
        let (mut v, mut r) = (Vec::new(), t);
        while r > 1 {
            v.push(r.min(8));
            r /= r.min(8);
        }
        Ok(v)
    };
    let tmax = *ents.iter().max().ok_or("--entries")?;
    let bmax = *looks.iter().max().ok_or("--lookups")?;
    let lmax = bases(tmax)?.len();
    let rows_cap = (bmax * tmax.div_ceil(8)).max(tmax);
    let chain = Chain::generate(&ChainSpec::n32())?;
    let n = chain.n;
    let mut params = switch_params(args, n)?;
    params.chain = true;
    params.pack = true;
    if params.nu != 1 {
        return Err("tfhe-tree: the private lookup runs at nu 1".into());
    }
    let nb = params.n_br;
    let (row, aw) = (nb + 1, 4 * nb);
    let max = rows_cap.min(1024);
    let dev = Device::open()?;
    let blt = pc_cuda::blas::Lt::new(&dev)?;
    let cfg = MemCfg { k_max: 1, pool: vec![PoolClass { k: 1, count: 1 }], events: 16, scratch_cts: 2 };
    let mut mp = MemPlan::build::<W, M>(&chain, cfg)?;
    mp.reserve("tfhe", pc_gpu::tfhe::TfheSwitch::bytes(n, &params, max));
    let words = (tmax + bmax * lmax + 3 * rows_cap) * row + (tmax / 8 + 1 + rows_cap) * aw + 8 * 3 * nb + 2 * rows_cap + 4096;
    mp.reserve("tree", 4 * words + (1 << 20));
    let arena = mp.allocate(&dev)?;
    let ex = Executor::<W>::new(&dev, &arena, &mp)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &mp)?;
    let s = &ex.compute;
    let mut prng = test_prng(0x7ee);
    let sk = SecretKey::sample(&mut prng, n)?;
    let sv: Vec<i64> = sk.s.iter().map(|&v| v as i64).collect();
    let (p0, p1) = (chain.q[0], chain.q[1]);
    let tk = Instant::now();
    let (sw, _s_lwe, s_br) = Switcher::generate_keep(params.clone(), &sv, p0, p1, &mut prng)?;
    println!("[pc-check] tfhe-tree: host keys {:.1} s (n_lwe {}, n_br {nb}, lat_group {}, packing key)", tk.elapsed().as_secs_f64(), params.n_lwe, params.lat_group);
    let mut ts = pc_gpu::tfhe::TfheSwitch::new(&arena, &dev, s, &sw, max)?;
    let al = |w: usize| arena.alloc::<pc_cuda::Compute, u32>("tree", w);
    let mut d_ent = al(tmax * row)?;
    let mut d_idx = al(bmax * lmax * row)?;
    let d_rep = al(rows_cap * row)?;
    let mut d_out = [al(rows_cap * row)?, al(rows_cap * row)?];
    let mut d_leaf = al((tmax / 8).max(1) * aw)?;
    let mut d_accs = al(rows_cap * aw)?;
    let mut d_pl = al(8 * 3 * nb)?;   // per table size 2..8 (log2 1..3): place [n][2] + w [n]
    let mut d_pairs = al(4 * rows_cap)?;
    // --pack key: the packing by placements (PACK_DIGITS / MAC: a key row NTT each); default: by rows (PACK_RDIG / RMAC)
    let by_rows = arg(args, "--pack").as_deref() != Some("key");
    let mut d_prow = al(3 * 8 * nb)?;                     // P [b][n] int32 for b = 2, 4, 8 (at (lb - 1) 8 n)
    let d_phat = arena.alloc::<pc_cuda::Compute, u64>("tree", 3 * 8 * 2 * nb)?;
    let mut d_rmap = al(rows_cap)?;
    let mut d_chk = al(aw)?;
    arena.freeze();
    let enc_v = |v: f64| ((v * p0 as f64 / 4.0).round() as i64).rem_euclid(p0 as i64) as u64;
    let dg = DiscreteGaussian::new(params.lwe_sigma_rel * p0 as f64)?;
    let encrypt = |vals: &[f64], prng: &mut pc_core::keys::Prng| -> Vec<u32> {
        vals.iter().flat_map(|&v| {
            let l = Lwe::encrypt(enc_v(v), &s_br, p0, &dg, prng);
            l.a.into_iter().chain(std::iter::once(l.b)).map(|x| x as u32).collect::<Vec<_>>()
        }).collect()
    };
    let decrypt = |w: &[u32], cnt: usize| -> Vec<f64> {
        (0..cnt).map(|r| {
            let l = Lwe { a: w[r * row..r * row + nb].iter().map(|&x| x as u64).collect(), b: w[r * row + nb] as u64 };
            centered(l.phase(&s_br, p0), p0) as f64 * 4.0 / p0 as f64
        }).collect()
    };
    // impl/tfhe.py table_place(B): coefficient c serves entry floor((m + 1) B / 2) of m = 2 c / n (c < n / 2) or
    // 2 (c - n) / n, holding -v above n / 2
    for lb in 1..=3usize {
        let bsz = 1usize << lb;
        let mut w: Vec<u32> = Vec::with_capacity(3 * nb);
        let mut ws: Vec<u32> = Vec::with_capacity(nb);
        for c in 0..nb {
            let m = if c < nb / 2 { 2.0 * c as f64 / nb as f64 } else { 2.0 * (c as f64 - nb as f64) / nb as f64 };
            let i = (((m + 1.0) * bsz as f64 / 2.0).floor() as usize).min(bsz - 1);
            w.extend([i as u32, c as u32]);
            ws.push(if c < nb / 2 { 1 } else { (-1i32) as u32 });
        }
        w.extend(ws);
        d_pl.upload_at(s, (lb - 1) * 3 * nb, &w)?;
    }
    for lb in 1..=3usize {
        let bsz = 1usize << lb;
        let mut pr = vec![0u32; bsz * nb];
        for c in 0..nb {
            let m = if c < nb / 2 { 2.0 * c as f64 / nb as f64 } else { 2.0 * (c as f64 - nb as f64) / nb as f64 };
            let i = (((m + 1.0) * bsz as f64 / 2.0).floor() as usize).min(bsz - 1);
            pr[i * nb + c] = if c < nb / 2 { 1 } else { (-1i32) as u32 };
        }
        d_prow.upload_at(s, (lb - 1) * 8 * nb, &pr)?;
        let pp = (d_prow.vptr() as *const u32).wrapping_add((lb - 1) * 8 * nb) as *const c_void;
        let ph = (d_phat.vptr() as *mut u64).wrapping_add((lb - 1) * 8 * 2 * nb) as *mut c_void;
        ts.native_pack_basis(&ctx, s, pp, bsz, ph)?;
    }
    let rowp = |bsz: usize| -> (*const c_void, *const c_void) {
        let lb = bsz.trailing_zeros() as usize;
        ((d_prow.vptr() as *const u32).wrapping_add((lb - 1) * 8 * nb) as *const c_void, (d_phat.vptr() as *const u64).wrapping_add((lb - 1) * 8 * 2 * nb) as *const c_void)
    };
    let rmap: Vec<u32> = (0..rows_cap as u32).collect();
    d_rmap.upload_at(s, 0, &rmap)?;
    let plp = |bsz: usize| -> (*const c_void, *const c_void) {
        let o = (bsz.trailing_zeros() as usize - 1) * 3 * nb;
        let b = d_pl.vptr() as *const u32;
        (b.wrapping_add(o) as *const c_void, b.wrapping_add(o + 2 * nb) as *const c_void)
    };
    let tindex = |d: usize, bsz: usize| -1.0 + (2.0 * d as f64 + 1.0) / bsz as f64;
    let lt_ = &blt.inner;
    let mut t = Tally::default();
    let ms_sd_c = ((params.n_lwe as f64 / 2.0 + 1.0) / 12.0).sqrt();   // the mod switch's sd in 2 n_br units
    for &te in &ents {
        let bs = bases(te)?;
        let lv = bs.len();
        let mut rng = test_prng(0x3e0 + te as u32);
        // the table: 4-bit values v at native (2 v + 1) / 16 - 1
        let vals: Vec<u64> = (0..te).map(|_| rng.uniform_below(16)).collect();
        let enc16 = |v: u64| (2.0 * v as f64 + 1.0) / 16.0 - 1.0;
        d_ent.upload_at(s, 0, &encrypt(&vals.iter().map(|&v| enc16(v)).collect::<Vec<_>>(), &mut rng))?;
        // the table's preparation: the leaf accumulators (B_0 entries each)
        let b0 = bs[0];
        let g0 = te / b0;
        let (pl0, w0) = plp(b0);
        let (pr0, ph0) = rowp(b0);
        let tp = Instant::now();
        if by_rows {
            let per = max / b0;
            for a0 in (0..g0).step_by(per) {
                let ac = per.min(g0 - a0);
                let src = (d_ent.vptr() as *const u32).wrapping_add(a0 * b0 * row) as *const c_void;
                let out = (d_leaf.vptr() as *mut u32).wrapping_add(a0 * aw) as *mut c_void;
                ts.native_pack_rows(&ctx, s, lt_, src, ac * b0, d_rmap.vptr(), ac, b0, pr0, ph0, out)?;
            }
        } else {
            for g in 0..g0 {
                let src = (d_ent.vptr() as *const u32).wrapping_add(g * b0 * row) as *const c_void;
                let out = (d_leaf.vptr() as *mut u32).wrapping_add(g * aw) as *mut c_void;
                ts.native_pack(&ctx, s, lt_, src, b0, pl0, w0, nb, out)?;
            }
        }
        s.sync()?;
        let t_prep = tp.elapsed().as_secs_f64();
        // the packing by rows == the packing by placements, every word (group 0 of the leaves)
        ts.native_pack(&ctx, s, lt_, d_ent.vptr(), b0, pl0, w0, nb, d_chk.vptr() as *mut c_void)?;
        let (x1, x2) = (d_chk.download_range(s, 0, aw)?, d_leaf.download_range(s, 0, aw)?);
        t.add(&format!("tfhe-tree: {te} entries, leaf accumulator 0 == the packing by placements, every word"), x1.iter().zip(&x2).filter(|(a, b)| a != b).count(), aw);
        for &nl in &looks {
            let idx: Vec<usize> = (0..nl).map(|_| rng.uniform_below(te as u64) as usize).collect();
            // digit l of lookup j at row j lv + l
            let mut dv = Vec::with_capacity(nl * lv);
            for &i in &idx {
                let mut r = i;
                for &b in &bs {
                    dv.push(tindex(r % b, b));
                    r /= b;
                }
            }
            d_idx.upload_at(s, 0, &encrypt(&dv, &mut rng))?;
            // per level: digit l of lookup j for every group (rows j g + k), at offsets of the pairs buffer
            let mut poff = Vec::new();
            let (mut off, mut np) = (0usize, te);
            for (l, &b) in bs.iter().enumerate() {
                let g = np / b;
                let pairs: Vec<u32> = (0..nl * g).flat_map(|r| [((r / g) * lv + l) as u32, u32::MAX]).collect();
                d_pairs.upload_at(s, off, &pairs)?;
                poff.push(off);
                off += pairs.len();
                np = g;
            }
            let mut times = Vec::new();
            let (mut t_rot, mut t_pack) = (0f64, 0f64);
            let mut fin = 0usize;
            for rep in 0..=reps {
                s.sync()?;
                let t0 = Instant::now();
                let (mut tr, mut tpk) = (0f64, 0f64);
                let mut cur = 0usize;   // d_out[cur]: the previous level's outputs [j][N]
                let mut nper = te;
                for (l, &b) in bs.iter().enumerate() {
                    let g = nper / b;
                    let cnt = nl * g;
                    let tl = Instant::now();
                    // the accumulators: level 0 the leaves (shared), above the previous outputs packed b per group
                    let accp: *const c_void = if l == 0 {
                        if nl == 1 {
                            d_leaf.vptr()
                        } else {
                            for j in 0..nl {
                                d_accs.copy_from_at(s, j * g * aw, &d_leaf, 0, g * aw)?;
                            }
                            d_accs.vptr()
                        }
                    } else {
                        if by_rows {
                            let (pr, ph) = rowp(b);
                            let per = max / b;
                            for a0 in (0..cnt).step_by(per) {
                                let ac = per.min(cnt - a0);
                                let src = (d_out[cur].vptr() as *const u32).wrapping_add(a0 * b * row) as *const c_void;
                                let out = (d_accs.vptr() as *mut u32).wrapping_add(a0 * aw) as *mut c_void;
                                ts.native_pack_rows(&ctx, s, lt_, src, ac * b, d_rmap.vptr(), ac, b, pr, ph, out)?;
                            }
                        } else {
                            let (pl, w) = plp(b);
                            for r in 0..cnt {
                                let src = (d_out[cur].vptr() as *const u32).wrapping_add(r * b * row) as *const c_void;
                                let out = (d_accs.vptr() as *mut u32).wrapping_add(r * aw) as *mut c_void;
                                ts.native_pack(&ctx, s, lt_, src, b, pl, w, nb, out)?;
                            }
                        }
                        d_accs.vptr()
                    };
                    if rep == reps {
                        s.sync()?;
                    }
                    let tl2 = Instant::now();
                    // digit l of lookup j for every group: rows j g + k
                    let pp = (d_pairs.vptr() as *const u32).wrapping_add(poff[l]) as *const c_void;
                    ts.native_lin_c(&ctx, s, d_rep.vptr() as *mut c_void, d_idx.vptr(), d_idx.vptr(), pp, cnt, 1, 0, 0)?;
                    let nx = 1 - cur;
                    let dst = d_out[nx].vptr() as *mut c_void;
                    ts.native_plut_each(&ctx, s, lt_, d_rep.vptr(), cnt, accp, &[0], &[dst])?;
                    if rep == reps {
                        s.sync()?;
                        tpk += (tl2 - tl).as_secs_f64();
                        tr += tl2.elapsed().as_secs_f64();
                    }
                    cur = nx;
                    nper = g;
                }
                s.sync()?;
                times.push(t0.elapsed().as_secs_f64());
                if rep == reps {
                    (t_rot, t_pack) = (tr, tpk);
                }
                fin = cur;
            }
            let last = times.pop().unwrap_or(0.0);
            times.remove(0);
            times.push(last);
            times.sort_by(f64::total_cmp);
            let med = times[times.len() / 2];
            let got = decrypt(&d_out[fin].download_range(s, 0, nl * row)?, nl);
            let (mut wrong, mut dev) = (0usize, 0f64);
            for (j, &i) in idx.iter().enumerate() {
                let x = (got[j] + 1.0) * 8.0 - 0.5;
                wrong += (x.round() as i64 != vals[i] as i64) as usize;
                dev = dev.max((x - x.round()).abs());
            }
            t.add(&format!("tfhe-tree: {te} entries, {nl} lookups, every output its entry"), wrong, nl);
            let pf: Vec<f64> = bs.iter().map(|&b| pc_tfhe::pack::log2_tail((nb / b) as f64 / 2.0 / ms_sd_c)).collect();
            let pf_all = pf.iter().fold(f64::NEG_INFINITY, |a, &x| { let (h, l) = if a > x { (a, x) } else { (x, a) }; h + (1.0 + (l - h).exp2()).log2() });
            let rot: usize = (0..lv).map(|l| te / bs[..=l].iter().product::<usize>()).sum();
            println!(
                "[pc-check] tfhe-tree: {te} entries (digits {bs:?}), {nl} lookups: {:.3} ms (median of {reps}; last run: rotations {:.3}, packs {:.3}); {rot} rotations + {} packs per lookup, {lv} levels; table prep {:.3} ms ({g0} leaf accumulators, packing by {}); {:.0} lookups/s; p_fail per lookup 2^{pf_all:.1} (index, levels {pf:?}); max output dev {dev:.2e} (of 0.5)",
                med * 1e3, t_rot * 1e3, t_pack * 1e3, rot - te / bs[0], t_prep * 1e3, if by_rows { "rows" } else { "placements" }, nl as f64 / med
            );
        }
    }
    println!("[pc-check] tfhe-tree: {} gates, {} failing", t.gates, t.bad);
    Ok(t.bad == 0)
}

/// `pc-check tfhe-hybrid [--t 16,64,256] [--lookups 64,1024,16384,32768] [--reps 5]`: a batch of L table lookups on
/// native LWEs through CKKS FBS, end to end on the device, against the native PBS on the same L: the indices j in
/// [-t/2, t/2) as native LWEs (value 2 j / t), `native_store` (repack into the StC-first split: the phase lands at
/// the FBS table's input phase j q0 / 2t), the FBS finish (ModRaise, CtS, the EvalMod sine + the table's series:
/// `StcFirstRoute::run_fbs`, real payload), the StC-first input half of its output, `native_extract` (ring switch,
/// extraction, key switch) back to LWEs under s_lwe (a bootstrap's input, as the native lookup's own key switch
/// leaves them), decrypted against the table. The native lookup: `native_pbs` (key switch + PBS) of the same L LWEs.
/// Every hybrid output must decode to its entry.
pub fn cmd_hybrid(args: &[String]) -> ExitCode {
    match run_hybrid(args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("pc-check tfhe-hybrid: {e}");
            ExitCode::FAILURE
        }
    }
}

/// A fixed pseudo-random table on j in [-t/2, t/2): values on a grid of 1/16 in [-1/2, 1/2].
fn hyb_lut(t: u32, j: i64) -> f64 {
    let h = (t / 2) as i64;
    let mut z = ((j + h) as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z ^= z >> 29;
    z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^= z >> 32;
    (z % 17) as f64 / 16.0 - 0.5
}

fn run_hybrid(args: &[String]) -> Res<bool> {
    use pc_core::keys::DiscreteGaussian;
    use pc_tfhe::lwe::Lwe;
    use std::ffi::c_void;
    let list = |k: &str, d: &str| -> Res<Vec<usize>> { Ok(arg(args, k).unwrap_or_else(|| d.into()).split(',').map(|x| x.trim().parse()).collect::<std::result::Result<_, _>>()?) };
    let ts_ = list("--t", "16,64,256")?;
    let looks = list("--lookups", "64,1024,16384,32768")?;
    let reps: usize = arg(args, "--reps").map(|s| s.parse()).transpose()?.unwrap_or(5).max(1);
    let chain = Chain::generate(&ChainSpec::n32())?;
    let n = chain.n;
    let slots = n / 2;
    let lmax = *looks.iter().max().ok_or("--lookups")?;
    if lmax > slots {
        return Err(format!("--lookups: at most {slots} (one real-payload ciphertext)").into());
    }
    let tmax = *ts_.iter().max().ok_or("--t")?;
    let cf: u32 = 2;
    let mut setup = BtsSetup::n32_shipped();
    setup.raise_drops = vec![0];
    setup.k_max = 1;
    setup.real = true;
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
    let spec = KeySetSpec { relin: true, conj: true, rots: steps.iter().map(|&s| RotSpec { step: s, q_top: None }).collect(), encaps: true, lean: None };
    // one FBS per table size: the shipped EvalMod (sine) and the degree-t series of the table (message j at phase
    // j q0 / 2t, CF log2 2t: the series' coefficients x 2^-CF)
    let base = BtsSetup::n32_shipped();
    let specs: Vec<pc_gpu::bts::stc_first::FbsSpec> = ts_
        .iter()
        .map(|&t| {
            let fcf = (2 * t as u32).trailing_zeros();
            let post = fbs_series(t as u32, (fcf as f64).exp2().recip());
            pc_gpu::bts::stc_first::FbsSpec { table: base.evalmod_coeffs.clone(), k: base.boot_k, da_its: base.evalmod_da_its, amp: 1.0, post, table_s: vec![], off: 0.0, post_s: vec![], paired: false, drop: 0, complex: false, real: true }
        })
        .collect();
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
    let (req, mut wshape) = plan.mem::<W>()?;
    req.reserve(&mut mp);
    let hosts: Vec<_> = specs.iter().map(|sp| plan.fbs_host(sp)).collect::<std::result::Result<_, _>>()?;
    for h in &hosts {
        let (b, w) = plan.fbs_mem::<W>(h)?;
        mp.reserve("bts.fbs", b);
        wshape = wshape.max(w);
    }
    mp.reserve("bts.work", BtsWork::<W>::bytes(&lay, wshape));
    mp.reserve("check.cts", 2 * pc_cuda::Arena::bytes_for::<W>(CtBuf::<pc_cuda::Compute, W>::words(&lay, 1)));
    let mut params = switch_params(args, n)?;
    params.chain = true;
    let nb = params.n_br;
    let row = nb + 1;
    mp.reserve("tfhe", pc_gpu::tfhe::TfheSwitch::bytes(n, &params, lmax));
    mp.reserve("check.hyb", 4 * (3 * lmax * row + lmax + 4 * params.tv_len()) + (1 << 20));
    let arena = mp.allocate(&dev)?;
    let mut ex = Executor::<W>::new(&dev, &arena, &mp)?;
    ex.ks.carve_regen(&arena, &lay)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &mp)?;
    let kplans = KsPlans::new(&arena, &ex.compute, &ctx, 1)?;
    let mut ops = Ops::new(&arena, &dev, &ex.compute, &ctx, ocfg)?;
    let mut work = BtsWork::<W>::new(&arena, lay, wshape)?;
    let mut keys = KeySet::<W>::carve(&arena, &chain, &spec, KskLayout::Dense)?;
    let mut prng = test_prng(0x4b1);
    let sk = SecretKey::sample(&mut prng, n)?;
    let mut sec = DevSecret::<W>::carve(&arena, lay)?;
    sec.load(&ctx, &ex.compute, &sk)?;
    let mut kg = KeyGen::<W>::new(&arena, &ex.compute, &chain, &shapes)?;
    kg.generate_set(&ctx, &mut ex, &sec, &mut keys, &mut prng)?;
    let mut encr = Encryptor::<W>::new(&arena, &ex.compute, lay, false, &mut prng)?;
    let route = plan.build(&arena, &dev, &ex.compute, &ctx, &keys, ex.ks.stage_ptrs())?;
    let fbs: Vec<_> = hosts.iter().map(|h| route.build_fbs(&arena, &ex.compute, &ctx, &keys, ex.ks.stage_ptrs(), h)).collect::<std::result::Result<_, _>>()?;
    let mut y = Ct::new(CtMeta::new(0, 1), CtBuf::<pc_cuda::Compute, W>::new(&arena, "check.cts", lay, 1)?);
    let mut z = Ct::new(CtMeta::new(0, 1), CtBuf::<pc_cuda::Compute, W>::new(&arena, "check.cts", lay, 1)?);
    let sv: Vec<i64> = sk.s.iter().map(|&v| v as i64).collect();
    let (p0, p1) = (chain.q[0], chain.q[1]);
    let (sw, s_lwe, s_br) = Switcher::generate_keep(params.clone(), &sv, p0, p1, &mut prng)?;
    let nl = params.n_lwe;
    let mut ts = pc_gpu::tfhe::TfheSwitch::new(&arena, &dev, &ex.compute, &sw, lmax)?;
    let al = |w: usize| arena.alloc::<pc_cuda::Compute, u32>("check.hyb", w);
    let mut d_in = al(lmax * row)?;
    let d_out = al(lmax * row)?;
    let d_nat = al(lmax * row)?;
    let mut d_pos = al(lmax)?;
    let mut d_tv = al(4 * params.tv_len())?;
    arena.freeze();
    let sh = Shadow::new(ShadowConfig { chain: ctx.shape, bts: BtsConfig::n32_shipped(), quirks: Quirks::none() }, Some(Arc::new(ctx.scale.clone())));
    let enc = Encoder::new(&chain);
    let nq = ctx.shape.n_q;
    let tl = sw.params.tv_len();
    // tables: 0 the load's identity (slot value m -> native m), 1 a native LUT for the PBS timing
    let tvs: Vec<u32> = [pc_tfhe::chain::native_tv(tl, sw.params.q_br, |m| m), pc_tfhe::chain::native_tv(tl, sw.params.q_br, |m| (4.0 * m).round() / 8.0)]
        .iter()
        .flat_map(|t| pc_gpu::tfhe::tv_residues(&sw.bsk, t))
        .collect();
    d_tv.upload(&ex.compute, &tvs)?;
    let tvp = |k: usize| (d_tv.vptr() as *const u32).wrapping_add(k * 2 * tl) as *const c_void;
    let enc_v = |v: f64| ((v * p0 as f64 / 4.0).round() as i64).rem_euclid(p0 as i64) as u64;
    let dg = DiscreteGaussian::new(params.lwe_sigma_rel * p0 as f64)?;
    // the hybrid's outputs: LWEs under s_lwe (n_lwe + 1 words), the slot value m at phase m p0 / 4
    let decrypt = |w: &[u32], cnt: usize| -> Vec<f64> {
        (0..cnt).map(|r| {
            let l = Lwe { a: w[r * (nl + 1)..r * (nl + 1) + nl].iter().map(|&x| x as u64).collect(), b: w[r * (nl + 1) + nl] as u64 };
            centered(l.phase(&s_lwe, p0), p0) as f64 * 4.0 / p0 as f64
        }).collect()
    };
    let bits = (slots as u32).trailing_zeros();
    let coef_of = |i: usize| ((i + slots - 1) % slots).reverse_bits() >> (usize::BITS - bits);
    let pos: Vec<u32> = (0..lmax).map(|i| coef_of(i * (slots / lmax)) as u32).collect();
    d_pos.upload(&ex.compute, &pos)?;
    // a split to start from (its content is replaced by the store): a fresh ciphertext through the input half
    let level = 44u32;
    let top = (nq - 1 - level) as usize;
    let sf_ = ctx.scale.sf(top as i32)?;
    let mut meta = CtMeta::new(level, 1).with_nf(sf_);
    meta.slots = slots as u32;
    let zeros = vec![Complex::new(0.0, 0.0); slots];
    let msg = enc.encode_slots(&zeros, slots, top, Payload::Complex, sf_)?;
    let mut t = Tally::default();
    let r_in = Route { cf: Some(cf), stc_first: true, real: false, ..Route::dense() };
    for (ti, &tt) in ts_.iter().enumerate() {
        let fcf = (2 * tt as u32).trailing_zeros();
        let r_fbs = Route { cf: Some(fcf), stc_first: true, real: true, ..Route::dense() };
        for &ll in &looks {
            let h = (tt / 2) as i64;
            let mut rng = test_prng(0x77 + (tt * 131 + ll) as u32);
            let js: Vec<i64> = (0..ll).map(|_| rng.uniform_below(tt as u64) as i64 - h).collect();
            // the indices as native LWEs: value 2 j / t (phase j p0 / 2t) lands at the split's phase j q0 / 2t
            let w: Vec<u32> = js.iter().flat_map(|&j| {
                let l = Lwe::encrypt(enc_v(2.0 * j as f64 / tt as f64), &s_br, p0, &dg, &mut rng);
                l.a.into_iter().chain(std::iter::once(l.b)).map(|x| x as u32).collect::<Vec<_>>()
            }).collect();
            d_in.upload_at(&ex.compute, 0, &w)?;
            let (mut tot, mut st_store, mut st_fbs, mut st_in, mut st_load, mut tnat) = (Vec::new(), 0f64, 0f64, 0f64, 0f64, Vec::new());
            for rep in 0..reps + 1 {
                encr.encrypt_sk(&ctx, &ex.compute, &sec, &[msg.clone()], meta, &mut y, &mut prng)?;
                let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                let mut call = BtsCall::begin(StagedRef::StcFirst(&route), &cx, std::slice::from_ref(&y), &r_fbs, CallOpts { prescaled: false, own_cf: true })?;
                call.stc_first_input(&mut cx, std::slice::from_mut(&mut y))?.ok_or("input below the StC-first entry")?;
                cx.ex.compute.sync()?;
                let mut mark = Vec::new();
                let t0 = Instant::now();
                ts.native_store(&ctx, &cx.ex.compute, &blt.inner, d_in.vptr(), d_pos.vptr(), ll, 1, &mut y)?;
                cx.ex.compute.sync()?;
                mark.push(t0.elapsed().as_secs_f64());
                route.run_fbs(&mut cx, &call.st, std::slice::from_mut(&mut y), &fbs[ti])?;
                cx.ex.compute.sync()?;
                mark.push(t0.elapsed().as_secs_f64());
                let mut back = BtsCall::begin(StagedRef::StcFirst(&route), &cx, std::slice::from_ref(&y), &r_in, CallOpts { prescaled: false, own_cf: true })?;
                back.stc_first_input(&mut cx, std::slice::from_mut(&mut y))?.ok_or("FBS output below the StC-first entry")?;
                cx.ex.compute.sync()?;
                mark.push(t0.elapsed().as_secs_f64());
                ts.native_extract(&ctx, &cx.ex.compute, &blt.inner, &mut y, d_pos.vptr(), ll, d_out.vptr() as *mut c_void, false)?;
                cx.ex.compute.sync()?;
                mark.push(t0.elapsed().as_secs_f64());
                // the native lookup of the same L LWEs
                let t1 = Instant::now();
                ts.native_pbs(&ctx, &cx.ex.compute, &blt.inner, d_in.vptr(), ll, tvp(1), &[0], &[d_nat.vptr() as *mut c_void])?;
                cx.ex.compute.sync()?;
                let tn = t1.elapsed().as_secs_f64();
                if rep > 0 {
                    tot.push(mark[3]);
                    tnat.push(tn);
                    st_store += mark[0] / reps as f64;
                    st_fbs += (mark[1] - mark[0]) / reps as f64;
                    st_in += (mark[2] - mark[1]) / reps as f64;
                    st_load += (mark[3] - mark[2]) / reps as f64;
                }
                let _ = &mut z;
            }
            let got = decrypt(&d_out.download_range(&ex.compute, 0, ll * (nl + 1))?, ll);
            let (mut wrong, mut dev) = (0usize, 0f64);
            for (k, &j) in js.iter().enumerate() {
                let want = hyb_lut(tt as u32, j);
                let e = got[k] - want;
                dev = dev.max(e.abs());
                wrong += (e.abs() >= 1.0 / 32.0) as usize;
            }
            t.add(&format!("tfhe-hybrid: t {tt}, {ll} lookups, every output its entry (|err| < 1/32)"), wrong, ll);
            tot.sort_by(f64::total_cmp);
            tnat.sort_by(f64::total_cmp);
            let (th, tn) = (tot[tot.len() / 2], tnat[tnat.len() / 2]);
            println!(
                "[pc-check] tfhe-hybrid: t {tt}, {ll} lookups: hybrid {:.2} ms (store {:.2}, FBS {:.2}, input half {:.2}, to LWE + key switch {:.2}) vs native key switch + PBS {:.2} ms: {:.2}x; {:.0} vs {:.0} lookups/s; max output err {dev:.2e} (entries 1/16 apart)",
                th * 1e3, st_store * 1e3, st_fbs * 1e3, st_in * 1e3, st_load * 1e3, tn * 1e3, tn / th, ll as f64 / th, ll as f64 / tn
            );
        }
        let _ = tmax;
    }
    println!("[pc-check] tfhe-hybrid: {} gates, {} failing", t.gates, t.bad);
    Ok(t.bad == 0)
}

/// Chebyshev coefficients (c0 not halved) of the degree-t interpolant of `hyb_lut` on s_j = sin(pi j / t), times `gain`
/// (`pc-bench fbs`'s `fbs_post`).
fn fbs_series(t: u32, gain: f64) -> Vec<f64> {
    let n = t as usize;
    let h = (t / 2) as i64;
    let mut c = vec![0f64; n + 1];
    for (deg, ck) in c.iter_mut().enumerate() {
        let mut acc = 0.0;
        for k in 0..=n {
            let w = if k == 0 || k == n { 0.5 } else { 1.0 };
            let j = h - k as i64;
            acc += w * hyb_lut(t, j.min(h - 1)) * (std::f64::consts::PI * (deg * k) as f64 / n as f64).cos();
        }
        *ck = 2.0 / n as f64 * acc * gain;
    }
    c[n] *= 0.5;
    c
}
