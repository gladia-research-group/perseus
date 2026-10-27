//! CKKS -> TFHE -> CKKS through the device's StC-first route with the host switch (pc-tfhe) in between.
//!
//!   pc-check tfhe [--switch 64] [--level 44] [--cf 2] [--out-cf 8]
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
    // the LUT lands at v q0 / 2^out_cf: EvalMod's sine is m only while m << q0, so out_cf > cf; the finish's x 2^cf
    // returns v / 2^(out_cf - cf)
    let out_cf: u32 = arg(args, "--out-cf").map(|s| s.parse()).transpose()?.unwrap_or(8);
    let t0 = Instant::now();
    let chain = Chain::generate(&ChainSpec::n32())?;
    let n = chain.n;
    let slots = n / 2;
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
    let steps: BTreeSet<i64> = crate::bts_stc_first::rot_steps(&geoms, n);
    let spec = KeySetSpec { relin: true, conj: true, rots: steps.iter().map(|&s| RotSpec { step: s, q_top: None }).collect(), encaps: true };
    let dev = Device::open()?;
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
    let arena = mp.allocate(&dev)?;
    let mut ex = Executor::<W>::new(&dev, &arena, &mp)?;
    ex.ks.carve_regen(&arena, &lay)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &mp)?;
    let kplans = KsPlans::new(&arena, &ex.compute, &ctx, 1)?;
    let mut ops = Ops::new(&arena, &dev, &ex.compute, &ctx, ocfg)?;
    let mut work = BtsWork::<W>::new(&arena, lay, wshape)?;
    let mut keys = KeySet::<W>::carve(&arena, &chain, &spec, KskLayout::Dense)?;
    let mut prng = test_prng(0x7f);
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
    arena.freeze();
    let sh = Shadow::new(ShadowConfig { chain: ctx.shape, bts: BtsConfig::n32_shipped(), quirks: Quirks::none() }, Some(Arc::new(ctx.scale.clone())));
    let enc = Encoder::new(&chain);
    let nq = ctx.shape.n_q;
    println!("[pc-check] tfhe: device setup {:.1} s", t0.elapsed().as_secs_f64());

    // the host switch, keyed by the same CKKS secret
    let s: Vec<i64> = sk.s.iter().map(|&v| v as i64).collect();
    let (p0, p1) = (chain.q[0], chain.q[1]);
    let mut params = Params::new(n)?;
    params.out_cf = out_cf;
    let tk = Instant::now();
    let sw = Switcher::generate(params, &s, p0, p1, &mut prng)?;
    println!("[pc-check] tfhe: host switch keys {:.1} s (p0 {p0}, p1 {p1})", tk.elapsed().as_secs_f64());

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
    let (tv, off) = sign01(sw.params.n_br, sw.delta_br());
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
        if vset.contains(&i) && vals[i].re.abs() < 0.02 {
            continue;
        }
        let want = if vset.contains(&i) { if vals[i].re >= 0.0 { 1.0 } else { 0.0 } } else { 0.0 };
        let gain = (1u64 << (out_cf - cf)) as f64;
        let e = (got[0][i].re * gain - want).abs().max(got[0][i].im.abs() * gain);
        worst = worst.max(e);
        bad += (e > 1e-2) as usize;
    }
    println!("[pc-check] tfhe: finished slots x 2^{} vs (1 + sign) / 2 (switched) and 0 (others): max err {worst:.3e}, output level {} deg {}", out_cf - cf, y.meta.level, y.meta.deg);
    t.add("tfhe: CKKS -> TFHE sign -> CKKS slots (err < 1e-2)", bad, slots);
    println!("[pc-check] tfhe: {} gates, {} failing", t.gates, t.bad);
    Ok(t.bad == 0)
}
