//! The user ops at every level of the user region, on one chain: device ms per op (median) and decrypted bits.
//!
//!   pc-bench ops [--reps 20] [--seeds 2]          (chain: PC_BENCH_CHAIN, default n32a4)
//!
//! From the dense landing down to the StC-first entry and the level below it, on a fresh encryption at the level's
//! scale: a rotation (one key, K = 1), the relinearized product x * x and its rescale, a scalar product (element-wise),
//! an add. Bits: -log2 of the worst real-part error of x * x (after the rescale) and of rot(x), relative to 1, over
//! `seeds` inputs of |x| <= 1. What an alternating user region (n32sp) changes against n32a4 is the limbs live at a
//! level and its scale.
use pc_core::encode::{Complex, Encoder, Payload};
use pc_core::keys::SecretKey;
use pc_core::modmul::Barrett64;
use pc_core::params::{Chain, ChainSpec};
use pc_cuda::{Device, Event};
use pc_gpu::decrypt::{Decryptor, Which};
use pc_gpu::encrypt::Encryptor;
use pc_gpu::keygen::{test_prng, KeyGen};
use pc_gpu::keys::{DevSecret, KeySet, KeySetSpec, KskLayout, KskShape, RotSpec};
use pc_gpu::ks::{KeyGroup, KsPlans, KsWork, KskSource};
use pc_gpu::ops::{Ops, OpsCfg};
use pc_gpu::poly::CtBuf;
use pc_gpu::rotate::{rotate, KsCfg};
use pc_gpu::{Context, Ct, Executor, MemCfg, MemPlan, PoolClass};
use pc_shadow::{BtsConfig, CtMeta, Quirks, Shadow, ShadowConfig};
use std::process::ExitCode;
use std::sync::Arc;

type W = u32;
type M = Barrett64;
type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn vals(seed: u64, n: usize) -> Vec<Complex> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            Complex::new((x >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0, 0.0)
        })
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn run(args: &[String]) -> Res<()> {
    let (mut reps, mut seeds) = (20usize, 2usize);
    let mut i = 0;
    while i < args.len() {
        let v = args.get(i + 1).ok_or("missing value")?;
        match args[i].as_str() {
            "--reps" => reps = v.parse()?,
            "--seeds" => seeds = v.parse()?,
            o => return Err(format!("unknown argument {o}").into()),
        }
        i += 2;
    }
    let name = std::env::var("PC_BENCH_CHAIN").unwrap_or_else(|_| "n32a4".into());
    let mut cspec = ChainSpec::preset(&name)?;
    // PC_BENCH_PMARGIN: P's margin over the widest digit (bits; a lean chain's `Lean::p_margin`), the HE-standard check
    // off: for the key-switch noise floor only
    if let (Some(m), Some(l)) = (std::env::var("PC_BENCH_PMARGIN").ok().and_then(|v| v.parse().ok()), cspec.lean.as_mut()) {
        l.p_margin = m;
        cspec.security = pc_core::params::Security::NotSet;
    }
    let chain = Chain::generate(&cspec)?;
    let n = chain.n;
    let spec = KeySetSpec { relin: true, conj: false, rots: vec![RotSpec { step: 1, q_top: None }], encaps: false };
    let dev = Device::open()?;
    let mcfg = MemCfg { k_max: 1, pool: vec![PoolClass { k: 1, count: 1 }], events: 16, scratch_cts: 2 };
    let mut mp = MemPlan::build::<W, M>(&chain, mcfg)?;
    let lay = mp.lay;
    let mut shapes: Vec<KskShape> = spec.keys(&chain)?.into_iter().map(|(_, s)| s).collect();
    shapes.sort();
    shapes.dedup();
    let layout = if std::env::var("PC_BENCH_DENSE_KEYS").is_ok_and(|v| v == "1") { KskLayout::Dense } else { KskLayout::packed_for::<W>(&chain)? };
    mp.reserve("keys", spec.bytes::<W>(&chain, layout)?);
    mp.reserve("keygen", KeyGen::<W>::bytes(&lay, &shapes));
    mp.reserve("secret", DevSecret::<W>::bytes(&lay));
    mp.reserve("encrypt", Encryptor::<W>::bytes(&lay, false));
    mp.reserve("decrypt", Decryptor::<W>::bytes(&lay, chain.level_unit, 1));
    mp.reserve("ks.plans", KsPlans::bytes::<W>(&chain, mp.cfg.k_max)?);
    mp.reserve("ks.regen_a", KsWork::<W>::regen_bytes(&lay, chain.dnum()));
    mp.reserve("ks.groups", 2 * KeyGroup::bytes(1, true));
    let ocfg = OpsCfg::default();
    Ops::<W>::reserve(&mut mp, &ocfg);
    let ctb = pc_cuda::Arena::bytes_for::<W>(CtBuf::<pc_cuda::Compute, W>::words(&lay, 1));
    mp.reserve("bench.cts", 3 * ctb);
    let arena = mp.allocate(&dev)?;
    let mut ex = Executor::<W>::new(&dev, &arena, &mp)?;
    ex.ks.carve_regen(&arena, &lay)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &mp)?;
    let kplans = KsPlans::new(&arena, &ex.compute, &ctx, mp.cfg.k_max)?;
    let mut ops = Ops::new(&arena, &dev, &ex.compute, &ctx, ocfg)?;
    let mut keys = KeySet::<W>::carve(&arena, &chain, &spec, layout)?;
    let mut prng = test_prng(0x5b);
    let sk = SecretKey::sample(&mut prng, n)?;
    let mut sec = DevSecret::<W>::carve(&arena, lay)?;
    sec.load(&ctx, &ex.compute, &sk)?;
    let mut kg = KeyGen::<W>::new(&arena, &ex.compute, &chain, &shapes)?;
    kg.generate_set(&ctx, &mut ex, &sec, &mut keys, &mut prng)?;
    let stage = ex.ks.stage_ptrs();
    let relin = KeyGroup::new(&arena, &ex.compute, n, &[keys.relin()?], stage)?;
    let rot1 = KeyGroup::new(&arena, &ex.compute, n, &[keys.rot(1)?], stage)?;
    let mut encr = Encryptor::<W>::new(&arena, &ex.compute, lay, false, &mut prng)?;
    let mut dec = Decryptor::<W>::new(&arena, &ex.compute, lay, chain.level_unit, 1)?;
    dec.load_secret(&ex.compute, &sec)?;
    let sh = Shadow::new(ShadowConfig { chain: ctx.shape, bts: BtsConfig::n32_shipped(), quirks: Quirks::none() }, Some(Arc::new(ctx.scale.clone())));
    let mut x = Ct::new(CtMeta::new(0, 1), CtBuf::new(&arena, "bench.cts", lay, 1)?);
    let mut y = Ct::new(CtMeta::new(0, 1), CtBuf::new(&arena, "bench.cts", lay, 1)?);
    let mut z = Ct::new(CtMeta::new(0, 1), CtBuf::new(&arena, "bench.cts", lay, 1)?);
    arena.freeze();
    let enc = Encoder::new(&chain);
    let (ev0, ev1) = (Event::timing(&dev)?, Event::timing(&dev)?);
    let kc = KsCfg { plans: &kplans, source: KskSource::Regen };
    let ctw = lay.ct_words();
    // the dense landing (28: CtS and EvalMod above it) down to the level below the StC-first entry
    let shape = ctx.shape;
    let entry = shape.level_at(chain.lone_top().map_or(7, |t| t + 2) as u32).ok_or("no entry level")?;
    println!("[ops] chain {name}: {} Q + {} P primes (log QP {}), entry level {entry};", chain.q.len(), chain.p.len(), chain.log_qp);
    println!("[ops] level, top, live limbs, log2 scale, ms (rotate, relin x*x, rescale, scalar x c, add), bits (x*x, rot)");
    println!("| level | top | live | log2 sf | rotate | x*x relin | rescale | x*c | add | bits x*x | bits rot |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    let mut lv = 28;
    while lv <= entry + shape.unit {
        let top = ctx.top_of(lv)?;
        let sf = ctx.scale.sf(top as i32)?;
        let mut fresh = |x: &mut Ct<'_, W>, ex: &mut Executor<'_, W>, prng: &mut _, seed: u64| -> Res<Vec<Complex>> {
            let v = vals(seed, n / 2);
            let msg = enc.encode_slots(&v, n / 2, top, Payload::Complex, sf)?;
            encr.encrypt_sk(&ctx, &ex.compute, &sec, &[msg], CtMeta::new(lv, 1).with_nf(sf), x, prng)?;
            Ok(v)
        };
        let time = |ex: &mut Executor<'_, W>, f: &mut dyn FnMut(&mut Executor<'_, W>) -> Res<()>| -> Res<f64> {
            let mut t = Vec::new();
            for r in 0..reps + 2 {
                ev0.record(&ex.compute)?;
                f(ex)?;
                ev1.record(&ex.compute)?;
                ev1.sync()?;
                if r >= 2 {
                    t.push(ev0.elapsed_ms(&ev1)? as f64);
                }
            }
            Ok(median(t))
        };
        fresh(&mut x, &mut ex, &mut prng, 7)?;
        let t_rot = time(&mut ex, &mut |ex| Ok(rotate(&ctx, ex, kc, &sh, &rot1, &x, 1, &mut y)?))?;
        let t_mul = time(&mut ex, &mut |ex| Ok(ops.mult(&ctx, ex, kc, &sh, &relin, &mut y, &x, &x)?))?;
        let t_res = time(&mut ex, &mut |ex| {
            z.buf.buf.copy_from(&ex.compute, &y.buf.buf, ctw)?;
            z.meta = y.meta;
            Ok(ops.rescale(&ctx, ex, &sh, std::slice::from_mut(&mut z))?)
        })?;
        let t_sc = time(&mut ex, &mut |ex| {
            z.buf.buf.copy_from(&ex.compute, &x.buf.buf, ctw)?;
            z.meta = x.meta;
            Ok(ops.mult_scalar_inplace(&ctx, ex, &sh, std::slice::from_mut(&mut z), &[Complex::new(0.75, 0.0)])?)
        })?;
        let t_add = time(&mut ex, &mut |ex| Ok(ops.add(&ctx, ex, &sh, std::slice::from_mut(&mut z), &[&x], &[&x])?))?;
        let (mut b_mul, mut b_rot) = (f64::INFINITY, f64::INFINITY);
        for s in 0..seeds {
            let v = fresh(&mut x, &mut ex, &mut prng, 101 + s as u64)?;
            ops.mult(&ctx, &mut ex, kc, &sh, &relin, &mut y, &x, &x)?;
            ops.rescale(&ctx, &mut ex, &sh, std::slice::from_mut(&mut y))?;
            let out = dec.decrypt_slots(&ctx, &ex.compute, &enc, &y, Payload::Complex, Which::Dense)?.remove(0);
            let e = out.iter().zip(&v).map(|(o, a)| (o.re - a.re * a.re).abs()).fold(0.0, f64::max);
            b_mul = b_mul.min(-e.log2());
            rotate(&ctx, &mut ex, kc, &sh, &rot1, &x, 1, &mut z)?;
            let out = dec.decrypt_slots(&ctx, &ex.compute, &enc, &z, Payload::Complex, Which::Dense)?.remove(0);
            let e = (0..n / 2).map(|j| (out[j].re - v[(j + 1) % (n / 2)].re).abs()).fold(0.0, f64::max);
            b_rot = b_rot.min(-e.log2());
        }
        println!("| {lv} | {top} | {} | {:.2} | {t_rot:.3} | {t_mul:.3} | {t_res:.3} | {t_sc:.3} | {t_add:.3} | {b_mul:.2} | {b_rot:.2} |", top + 1, sf.log2());
        lv += shape.unit;
    }
    Ok(())
}

pub fn cmd(args: &[String]) -> ExitCode {
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[ops] error: {e}");
            ExitCode::FAILURE
        }
    }
}
