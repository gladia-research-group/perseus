//! pc-bench: bootstrap, matmul and seam benchmarks on fresh keys.
//!
//!   pc-bench bts|bts-prec --route dense|1|256|512 [--classic --route 8192] [--k ..] [--cf ..] [--real]
//!       device ms per call/ciphertext (CUDA events, median of `reps` after 2 warm-ups) | precision on random real
//!       payloads: bits = -log2(max |out - in| / A) (the bts_accuracy_sweep methodology; `--old` adds the reference row)
//!   pc-bench ccmm|pcmm|stcf|stcf-prec|stcf-pair|stc-high|replay|modmul ...   (src/check_*.rs, bts_*.rs, replay_g0.rs, word_seam.rs)
//!
//! Keys: a fresh secret (dense ternary + sparse h=32), every CtS/StC rotation, conjugation, relinearization and both
//! encapsulation keys generated on the device (dense layout with resident `a`). Full usage: docs/ARCHITECTURE.md.
use pc_core::encode::{Complex, Encoder, Payload};
use pc_core::keys::SecretKey;
use pc_core::modmul::Barrett64;
use pc_core::params::{Chain, ChainSpec};
use pc_cuda::{Compute, Device, Event};
use pc_gpu::bts::dense::DensePlan;
use pc_gpu::bts::{BtsCx, BtsRoute, BtsSetup, BtsWork};
use pc_gpu::decrypt::{Decryptor, Which};
use pc_gpu::encrypt::Encryptor;
use pc_gpu::keygen::{test_prng, KeyGen};
use pc_gpu::keys::{DevSecret, KeySet, KeySetSpec, KskLayout, RotSpec};
use pc_gpu::ks::KsPlans;
use pc_gpu::ops::{Ops, OpsCfg};
use pc_gpu::poly::CtBuf;
use pc_gpu::{Context, Ct, Executor, MemCfg, MemPlan, PoolClass};
use pc_shadow::{CtMeta, Route};
use std::collections::BTreeSet;
use std::process::ExitCode;
use std::time::Instant;

#[path = "../dump.rs"]
#[allow(dead_code)]
mod dump;
#[path = "../dump_g0.rs"]
#[allow(dead_code)]
mod dump_g0;
#[path = "../bts_sparse.rs"]
mod bts_sparse;
#[path = "../bts_classic.rs"]
mod bts_classic;
#[path = "../replay_g0.rs"]
mod replay_g0;
#[path = "../check_bts.rs"]
#[allow(dead_code)]
mod check_bts;
#[path = "../check_ccmm.rs"]
#[allow(dead_code)]
mod check_ccmm;
#[path = "../check_pcmm.rs"]
#[allow(dead_code)]
mod check_pcmm;
#[path = "../bts_stc_first.rs"]
#[allow(dead_code)]
mod bts_stc_first;
#[path = "../bts_all.rs"]
mod bts_all;
#[path = "../word_seam.rs"]
mod word_seam;

type W = u32;
type M = Barrett64;
type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

struct Args {
    cmd: String,
    route: String,
    ks: Vec<usize>,
    reps: usize,
    drop: u32,
    cfs: Vec<u32>,
    real: bool,
    amps: Vec<f64>,
    iters: Vec<u32>,
    out: Option<String>,
    old: Option<String>,
    em_cached: bool,
    lt_group: i32,
    ks_source: Option<pc_gpu::ks::KskSource>,
    lt_baby: Vec<(pc_gpu::ks::KskSource, usize)>,
    graph: bool,
    level: u32,
    mean: f64,
    /// n32 (u32 x Barrett64) or n64 (u64 x Barrett64)
    chain: String,
    /// FIDESLIB_CTS_ALT_BUDGET for the dense route (0 = off)
    cts_alt: u32,
}

fn parse(a: &[String]) -> Res<Args> {
    let mut r = Args {
        cmd: a.first().cloned().ok_or("command")?,
        route: "dense".into(),
        ks: vec![1, 2, 4, 8],
        reps: 10,
        drop: 0,
        cfs: vec![6],
        real: false,
        amps: vec![0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        iters: vec![1],
        out: None,
        old: None,
        em_cached: false,
        lt_group: 0,
        ks_source: None,
        lt_baby: vec![(pc_gpu::ks::KskSource::Regen, pc_gpu::bts::cts::LT_REGEN_DIGITS)],
        graph: false,
        level: 44,
        mean: 0.0,
        // PC_BENCH_CHAIN: the default chain preset (bts-prec --all runs this harness without --chain)
        chain: std::env::var("PC_BENCH_CHAIN").unwrap_or_else(|_| "n32".into()),
        cts_alt: 0,
    };
    let mut i = 1;
    let list = |s: &str| -> Res<Vec<f64>> { s.split(',').map(|x| x.parse::<f64>().map_err(|e| e.into())).collect() };
    while i < a.len() {
        let v = a.get(i + 1).cloned().unwrap_or_default();
        match a[i].as_str() {
            "--route" => r.route = v,
            "--k" => r.ks = list(&v)?.into_iter().map(|x| x as usize).collect(),
            "--reps" => r.reps = v.parse()?,
            "--drop" => r.drop = v.parse()?,
            "--cts-alt" => r.cts_alt = v.parse()?,
            "--cf" => {
                r.cfs = if let Some((lo, hi)) = v.split_once("..") { (lo.parse()?..=hi.parse()?).collect() } else { list(&v)?.into_iter().map(|x| x as u32).collect() }
            }
            "--amps" => r.amps = list(&v)?,
            "--iters" => r.iters = list(&v)?.into_iter().map(|x| x as u32).collect(),
            "--out" => r.out = Some(v),
            "--old" => r.old = Some(v),
            "--lt-group" => r.lt_group = v.parse()?,
            "--ks-source" => {
                r.ks_source = Some(match v.as_str() {
                    "cached" => pc_gpu::ks::KskSource::Cached,
                    "regen" => pc_gpu::ks::KskSource::Regen,
                    _ => return Err(format!("--ks-source {v}: cached | regen").into()),
                })
            }
            "--lt-baby" => {
                r.lt_baby = v
                    .split(',')
                    .map(|x| match x {
                        "cached" => Ok((pc_gpu::ks::KskSource::Cached, usize::MAX)),
                        "regen" => Ok((pc_gpu::ks::KskSource::Regen, pc_gpu::bts::cts::LT_REGEN_DIGITS)),
                        o => o
                            .strip_prefix("regen")
                            .and_then(|d| d.parse().ok())
                            .map(|d| (pc_gpu::ks::KskSource::Regen, d))
                            .ok_or(format!("--lt-baby {o}: cached | regen | regen<digits>")),
                    })
                    .collect::<std::result::Result<_, _>>()?
            }
            "--level" => r.level = v.parse()?,
            "--mean" => r.mean = v.parse()?,
            "--chain" => r.chain = v,
            "--graph" => {
                r.graph = true;
                i += 1;
                continue;
            }
            "--em-cached" => {
                r.em_cached = true;
                i += 1;
                continue;
            }
            "--real" => {
                r.real = true;
                i += 1;
                continue;
            }
            o => return Err(format!("unknown argument {o}").into()),
        }
        i += 2;
    }
    if r.route != "dense" {
        return Err(format!("route {} not built by this card (dense only)", r.route).into());
    }
    Ok(r)
}

fn rand_real(seed: u64, n: usize, amp: f64) -> Vec<Complex> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut v: Vec<f64> = (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * amp
        })
        .collect();
    let m = v.iter().fold(0f64, |a, &b| a.max(b.abs()));
    v.iter_mut().for_each(|y| *y *= amp / m);
    v.into_iter().map(|y| Complex::new(y, 0.0)).collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

/// Old dense aperiodic row (cf, A) of bts_accuracy_\<chain>.json.
fn old_err(doc: &Option<serde_json::Value>, cf: u32, amp: f64) -> Option<f64> {
    let d = doc.as_ref()?;
    d["wall"].as_array()?.iter().find(|r| r["route"] == 0 && r["period"] == 0 && r["cf"] == cf && (r["amp"].as_f64().unwrap_or(-1.0) - amp).abs() < 1e-12).and_then(|r| r["rel_err"].as_f64())
}

fn run(a: Args) -> Res<()> {
    match a.chain.as_str() {
        "n32" => run_t::<W, M>(a, ChainSpec::n32(), BtsSetup::n32_shipped()),
        "n32a4" => run_t::<W, M>(a, ChainSpec::n32a4(), BtsSetup::n32_shipped()),
        "n64" => run_t::<u64, Barrett64>(a, ChainSpec::n64(), BtsSetup::n64()),
        c => Err(format!("--chain {c}: n32 | n32a4 | n64").into()),
    }
}

/// The dense route on `spec` with words `W` and strategy `M` (the generic parameters shadow the n32 aliases).
fn run_t<W: pc_gpu::context::DevWord, M: pc_gpu::context::DevStrategy<W>>(a: Args, spec: ChainSpec, setup0: BtsSetup) -> Res<()> {
    let t0 = Instant::now();
    let chain = Chain::generate(&spec)?;
    let chain_name = a.chain.clone();
    let n = chain.n;
    let kmax = *a.ks.iter().max().unwrap_or(&1).max(&8);
    let mut setup = setup0;
    setup.raise_drops = vec![a.drop];
    setup.cts_alt_budget = a.cts_alt;
    setup.k_max = kmax;
    setup.lt_group = a.lt_group;
    if let Some(src) = a.ks_source {
        setup.ks_source = src;
    }
    println!("[pc-bench] dense route key source {:?}", setup.ks_source);
    if a.em_cached {
        setup.em_relin = pc_gpu::ks::KskSource::Cached;
    }
    let plan = DensePlan::new(&chain, &setup)?;
    let mut steps = BTreeSet::new();
    for v in &plan.variants {
        for h in v.cts.iter().chain(&v.stc) {
            let g = h.geom;
            for i in 1..g.b {
                steps.insert(((i * g.stride) % (n / 2)) as i64);
            }
            if g.g > 1 {
                steps.insert(((g.b * g.stride) % (n / 2)) as i64);
            }
            steps.insert(g.offset.rem_euclid((n / 2) as i64));
        }
    }
    steps.remove(&0);
    {
        let st = pc_core::scale::ScaleTables::precompute(&chain)?;
        let v = &plan.variants[0];
        println!("[pc-bench] scale_dec {:e}; raised {:?}; em in {:?} out {:?}; out {:?}", pc_gpu::bts::stc::scale_dec(&st), v.raised, v.em_in, v.em_out, v.out);
        for (nm, hs) in [("cts", &v.cts), ("stc", &v.stc)] {
            for h in hs.iter() {
                let mx = h.coeffs.iter().flat_map(|c| c.iter()).fold(0i128, |a, &b| a.max(b.abs()));
                println!("[pc-bench]   {nm} top {} nf {:e} width {} geom {:?} end {:?} max|coeff| 2^{:.1}", h.top, h.nf, h.width, h.geom, h.end, (mx as f64).log2());
            }
        }
    }
    let spec = KeySetSpec { relin: true, conj: true, rots: steps.iter().map(|&s| RotSpec { step: s, q_top: None }).collect(), encaps: true };
    println!("[pc-bench] host plan {:.1} s: {} rotation keys, EvalMod buffers {}", t0.elapsed().as_secs_f64(), steps.len(), plan.variants[0].em_slots);
    let dev = Device::open()?;
    let cfg = MemCfg { k_max: kmax, pool: vec![PoolClass { k: 1, count: 1 }], events: 64, scratch_cts: 2 };
    let mut mp = MemPlan::build::<W, M>(&chain, cfg)?;
    let lay = mp.lay;
    let shapes: Vec<_> = spec.keys(&chain)?.into_iter().map(|(_, s)| s).collect();
    mp.reserve("keys", spec.bytes::<W>(&chain, KskLayout::Dense)?);
    mp.reserve("keygen", KeyGen::<W>::bytes(&lay, &shapes));
    mp.reserve("secret", DevSecret::<W>::bytes(&lay));
    mp.reserve("encrypt", Encryptor::<W>::bytes(&lay, false));
    mp.reserve("decrypt", Decryptor::<W>::bytes(&lay, chain.level_unit, kmax));
    mp.reserve("ks.plans", KsPlans::bytes::<W>(&chain, kmax)?);
    mp.reserve("ks.regen_a", pc_gpu::ks::KsWork::<W>::regen_bytes(&lay, chain.dnum()));
    let ocfg = OpsCfg::default();
    Ops::<W>::reserve(&mut mp, &ocfg);
    let (req, wshape) = plan.mem::<W>()?;
    req.reserve(&mut mp);
    mp.reserve("bts.work", BtsWork::<W>::bytes(&lay, wshape));
    mp.reserve("bench.dbg", 256 << 20);
    mp.reserve("bench.cts", pc_cuda::Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(&lay, kmax)));
    println!("[pc-bench] MemPlan {}", mp.report());
    let arena = mp.allocate(&dev)?;
    let mut ex = Executor::<W>::new(&dev, &arena, &mp)?;
    ex.ks.carve_regen(&arena, &lay)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &mp)?;
    let kplans = KsPlans::new(&arena, &ex.compute, &ctx, kmax)?;
    let mut ops = Ops::new(&arena, &dev, &ex.compute, &ctx, ocfg)?;
    let mut work = BtsWork::<W>::new(&arena, lay, wshape)?;
    let mut keys = KeySet::<W>::carve(&arena, &chain, &spec, KskLayout::Dense)?;
    let mut prng = test_prng(0x6a);
    let sk = SecretKey::sample(&mut prng, n)?;
    let mut sec = DevSecret::<W>::carve(&arena, lay)?;
    sec.load(&ctx, &ex.compute, &sk)?;
    let mut kg = KeyGen::<W>::new(&arena, &ex.compute, &chain, &shapes)?;
    let tk = kg.generate_set(&ctx, &mut ex, &sec, &mut keys, &mut prng)?;
    println!("[pc-bench] keygen {} keys in {:.1} s", tk.keys, tk.total);
    let mut encr = Encryptor::<W>::new(&arena, &ex.compute, lay, false, &mut prng)?;
    let mut dec = Decryptor::<W>::new(&arena, &ex.compute, lay, chain.level_unit, kmax)?;
    dec.load_secret(&ex.compute, &sec)?;
    let tb = Instant::now();
    let mut route = plan.build(&arena, &ex.compute, &ctx, &keys, ex.ks.stage_ptrs())?;
    ex.compute.sync()?;
    println!("[pc-bench] dense route tables {:.1} s; arena used {:.1} of {:.1} GB", tb.elapsed().as_secs_f64(), arena.used_bytes() as f64 / 1e9, arena.capacity() as f64 / 1e9);
    let mut x = Ct::new(CtMeta::new(0, 1), CtBuf::<Compute, W>::new(&arena, "bench.cts", lay, kmax)?);
    // debug: identity-diagonal stages (b = 2, one diagonal of ones) with and without an offset rotation of 2
    let mut dbg_stages = Vec::new();
    if a.cmd == "bts-debug" {
        use pc_gpu::bts::cts::{LtEnd, LtGeom, LtPts, LtStage, PtSource};
        let t = 21usize;
        let ones = vec![vec![pc_core::bts_setup::C64::new(1.0, 0.0); n / 2]];
        let sc = ctx.scale.sf(t as i32)?;
        // the fused ModDown-rescale end exists on composite-degree-2 chains only
        let ends: Vec<(i64, LtEnd)> = [(0i64, LtEnd::ModDown), (2, LtEnd::ModDown), (0, LtEnd::FusedRescale)].into_iter().filter(|e| e.1 != LtEnd::FusedRescale || chain.level_unit == 2).collect();
        for (off, end) in ends {
            let pts = LtPts::new(&arena, &ex.compute, &ctx, t, PtSource::Slots { values: &ones, scale: sc })?;
            dbg_stages.push(LtStage::new(&arena, &ex.compute, &ctx, LtGeom { b: 2, g: 1, stride: 1, npts: 1, offset: off }, t, end, sc, pts, &keys, ex.ks.stage_ptrs(), kmax)?);
        }
    }
    arena.freeze();
    let sh = pc_gpu::bts::dense::shadow_of(&ctx.scale);
    let enc = Encoder::new(&chain);
    // input level (primes dropped): 44 = the planner's reactive ceiling (AUTO_BTS_LEVEL), 0 = fresh at the top
    let top = chain.q.len() - 1 - a.level as usize;
    let slots = n / 2;
    // fresh encryptions of k messages at the top level, deg 1
    let mut fresh_at = |x: &mut Ct<'_, W>, ex: &mut Executor<'_, W>, k: usize, vals: &[Vec<Complex>], top: usize| -> Res<()> {
        let sf = ctx.scale.sf(top as i32)?;
        let mut meta = CtMeta::new((chain.q.len() - 1 - top) as u32, 1).with_nf(sf);
        meta.slots = slots as u32;
        x.buf.k = k;
        let msgs: Vec<Vec<Vec<u64>>> = vals.iter().map(|v| enc.encode_slots(v, slots, top, Payload::Complex, sf)).collect::<std::result::Result<_, _>>()?;
        encr.encrypt_sk(&ctx, &ex.compute, &sec, &msgs, meta, x, &mut prng)?;
        Ok(())
    };
    let base_route = Route { real: a.real, raise_drop: a.drop, ..Route::dense() };
    match a.cmd.as_str() {
        "bts" => {
            let ev0 = Event::timing(&dev)?;
            let ev1 = Event::timing(&dev)?;
            let cf = a.cfs[0];
            let r = Route { cf: Some(cf), ..base_route };
            for (k, (src0, nreg)) in a.ks.iter().flat_map(|&k| a.lt_baby.iter().map(move |&s| (k, s))) {
                route.set_lt_baby(src0, nreg)?;
                let src = if src0 == pc_gpu::ks::KskSource::Regen && nreg < 6 { format!("Regen{nreg}") } else { format!("{src0:?}") };
                let vals: Vec<Vec<Complex>> = (0..k).map(|i| rand_real(17 + i as u64, slots, 1.0)).collect();
                let mut dev_ms = Vec::new();
                let mut wall_ms = Vec::new();
                for rep in 0..a.reps + 2 {
                    fresh_at(&mut x, &mut ex, k, &vals, top)?;
                    ex.compute.sync()?;
                    let tw = Instant::now();
                    ev0.record(&ex.compute)?;
                    {
                        let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                        route.run(&mut cx, std::slice::from_mut(&mut x), &r)?;
                        ev1.record(&cx.ex.compute)?;
                    }
                    ev1.sync()?;
                    let w = tw.elapsed().as_secs_f64() * 1e3;
                    if rep >= 2 {
                        dev_ms.push(ev0.elapsed_ms(&ev1)? as f64);
                        wall_ms.push(w);
                    }
                }
                let cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                cx.ops.realize(&ctx, cx.ex, &sh, std::slice::from_mut(&mut x))?;
                let got = dec.decrypt_slots(&ctx, &ex.compute, &enc, &x, Payload::Complex, Which::Dense)?;
                let err = got.iter().zip(&vals).map(|(g, v)| g.iter().zip(v).map(|(a, b)| (a.re - b.re).abs()).fold(0f64, f64::max)).fold(0f64, f64::max);
                // one staged run with an event after every stage (per-stage device ms)
                {
                    fresh_at(&mut x, &mut ex, k, &vals, top)?;
                    let evs: Vec<Event> = (0..12).map(|_| Event::timing(&dev)).collect::<std::result::Result<_, _>>()?;
                    let mut names = Vec::new();
                    let st_api = route.staged().ok_or("staged")?;
                    let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                    evs[0].record(&cx.ex.compute)?;
                    let mut st = st_api.begin(&cx, std::slice::from_ref(&x), &r, false)?;
                    st_api.mod_raise(&mut cx, &mut st, std::slice::from_mut(&mut x))?;
                    evs[1].record(&cx.ex.compute)?;
                    names.push("raise".to_string());
                    for kk in 0..st.n_cts {
                        st_api.cts_stage(&mut cx, &st, std::slice::from_mut(&mut x), kk)?;
                        evs[names.len() + 1].record(&cx.ex.compute)?;
                        names.push(format!("cts{kk}"));
                    }
                    st_api.eval_mod(&mut cx, &st, std::slice::from_mut(&mut x))?;
                    evs[names.len() + 1].record(&cx.ex.compute)?;
                    names.push("evalmod".into());
                    st_api.stc_enter(&mut cx, &st, std::slice::from_mut(&mut x))?;
                    for kk in 0..st.n_stc {
                        st_api.stc_stage(&mut cx, &st, std::slice::from_mut(&mut x), kk)?;
                        evs[names.len() + 1].record(&cx.ex.compute)?;
                        names.push(format!("stc{kk}"));
                    }
                    st_api.finish(&mut cx, &st, std::slice::from_mut(&mut x))?;
                    evs[names.len() + 1].record(&cx.ex.compute)?;
                    names.push("finish".into());
                    evs[names.len()].sync()?;
                    let parts: Vec<String> = names.iter().enumerate().map(|(i, nm)| format!("{nm} {:.2}", evs[i].elapsed_ms(&evs[i + 1]).unwrap_or(0.0))).collect();
                    println!("[pc-bench] stages K={k} lt_baby={src} (ms): {}", parts.join(", "));
                }
                if a.graph {
                    // the whole refresh captured once (fixed input metadata), replayed `reps` times
                    fresh_at(&mut x, &mut ex, k, &vals, top)?;
                    let sp: *const pc_cuda::Stream<Compute> = &ex.compute;
                    let xp: *mut Ct<'_, W> = &mut x;
                    let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                    // SAFETY: the capture only reads the stream handle; x is not otherwise borrowed in the closure
                    let (g, _) = unsafe { &*sp }.capture(|_| route.run(&mut cx, std::slice::from_mut(unsafe { &mut *xp }), &r).map_err(|_| pc_cuda::Error::Unavailable("bootstrap under capture")))?;
                    let mut gm = Vec::new();
                    for rep in 0..a.reps + 2 {
                        fresh_at(&mut x, cx.ex, k, &vals, top)?;
                        ev0.record(&cx.ex.compute)?;
                        g.launch(&cx.ex.compute)?;
                        ev1.record(&cx.ex.compute)?;
                        ev1.sync()?;
                        if rep >= 2 {
                            gm.push(ev0.elapsed_ms(&ev1)? as f64);
                        }
                    }
                    let gd = median(gm);
                    println!("[pc-bench] bts route=dense real={} K={k} graph replay: {:.2} ms/call {:.2} ms/ct", a.real, gd, gd / k as f64);
                }
                let d = median(dev_ms);
                println!(
                    "[pc-bench] bts route=dense real={} lt_baby={src} cf={cf} drop={} input level {} K={k}: device {:.2} ms/call {:.2} ms/ct, wall {:.2} ms/call (median of {}); out level {} deg {}; max err {:.2e} (A=1)",
                    a.real, a.drop, a.level, d, d / k as f64, median(wall_ms), a.reps, x.meta.level, x.meta.deg, err
                );
            }
        }
        "bts-prec" => {
            let old = match &a.old {
                Some(p) => Some(serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(p)?)?),
                None => None,
            };
            let mut rows = Vec::new();
            for &it in &a.iters {
                for &cf in &a.cfs {
                    let r = Route { cf: Some(cf), iterations: it, ..base_route };
                    // one batch of the amplitudes (K = #amps, chunks of k_max)
                    for chunk in a.amps.chunks(kmax.min(if it > 1 { 1 } else { kmax })) {
                        let vals: Vec<Vec<Complex>> = chunk.iter().map(|&amp| rand_real(7777, slots, amp).into_iter().map(|c| Complex::new(c.re + a.mean * amp, 0.0)).collect()).collect();
                        fresh_at(&mut x, &mut ex, chunk.len(), &vals, top)?;
                        {
                            let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                            route.run(&mut cx, std::slice::from_mut(&mut x), &r)?;
                            cx.ops.realize(&ctx, cx.ex, &sh, std::slice::from_mut(&mut x))?;
                        }
                        let got = dec.decrypt_slots(&ctx, &ex.compute, &enc, &x, Payload::Complex, Which::Dense)?;
                        for (j, &amp) in chunk.iter().enumerate() {
                            let e = got[j].iter().zip(&vals[j]).map(|(g, v)| (g.re - v.re).abs()).fold(0f64, f64::max) / amp;
                            let ec = got[j].iter().zip(&vals[j]).map(|(g, v)| (*g - *v).abs()).fold(0f64, f64::max) / amp;
                            let o = old_err(&old, cf, amp);
                            println!(
                                "[pc-bench] bts-prec route=dense real={} level={} iters={it} cf={cf} A={amp}: rel_err {e:.3e} ({:.2} bits), complex |out - in| {ec:.3e} ({:.2} bits){}",
                                a.real,
                                a.level,
                                -e.log2(),
                                -ec.log2(),
                                o.map(|o| format!("; old {o:.3e} ({:.2} bits), new - old {:+.2} bits", -o.log2(), o.log2() - e.log2())).unwrap_or_default()
                            );
                            rows.push(serde_json::json!({"route": "dense", "real": a.real, "input_level": a.level, "iterations": it, "cf": cf, "period": 0, "amp": amp,
                                "rel_err": e, "bits": -e.log2(), "rel_err_complex": ec, "bits_complex": -ec.log2(), "old_rel_err": o, "landing_level": x.meta.level, "landing_deg": x.meta.deg}));
                        }
                    }
                }
            }
            if let Some(p) = &a.out {
                let mut doc: serde_json::Value = std::fs::read_to_string(p).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(serde_json::json!({"meta": {}, "rows": []}));
                doc["meta"]["chain"] = chain_name.clone().into();
                doc["meta"]["methodology"] = "random real payload uniform in [-A, A] (max anchored to A, every slot), encrypted (sk) at the row's input_level (0 = fresh at the top, as the old table), one bootstrap, realized, decrypted; rel_err = max |Re out - in| / A (the old table's real-part read), rel_err_complex = max |out - in| / A (complex); old = perseus/plan/data/bts_accuracy_<chain>.json dense period-0 row (`--old`)".into();
                let arr = doc["rows"].as_array_mut().ok_or("rows")?;
                arr.retain(|r| !(r["route"] == "dense" && r["real"] == a.real && rows.iter().any(|n| n["cf"] == r["cf"] && n["amp"] == r["amp"] && n["iterations"] == r["iterations"] && n["input_level"] == r["input_level"])));
                arr.extend(rows);
                std::fs::write(p, serde_json::to_string_pretty(&doc)?)?;
                println!("[pc-bench] wrote {p}");
            }
        }
        "bts-debug" => {
            for (i, stg) in dbg_stages.iter().enumerate() {
                let vals: Vec<Vec<Complex>> = vec![rand_real(99, slots, 1.0)];
                let t = 21usize;
                let sft = ctx.scale.sf(t as i32)?;
                fresh_at(&mut x, &mut ex, 1, &vals, t)?;
                {
                    let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                    stg.run(&mut cx, &mut x.buf, 1, pc_gpu::ks::KskSource::Cached)?;
                }
                x.meta.deg = 2;
                x.meta.nf = sft * sft;
                if i == 2 {
                    x.meta = sh.rescale(&x.meta)?.out;
                } else {
                    let cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                    cx.ops.rescale(&ctx, cx.ex, &sh, std::slice::from_mut(&mut x))?;
                }
                let got = dec.decrypt_slots(&ctx, &ex.compute, &enc, &x, Payload::Complex, Which::Dense)?;
                let r = if i == 1 { 2 } else { 0 };
                let d = (0..slots).map(|j| (got[0][j] - vals[0][(j + r) % slots]).abs()).fold(0f64, f64::max);
                let d0 = (0..slots).map(|j| (got[0][j] - vals[0][j]).abs()).fold(0f64, f64::max);
                println!("[pc-bench] debug identity stage offset {r}: max |out - rot(in, {r})| {d:.3e}, |out - in| {d0:.3e}, out[0..2] {:?} in[0..3] {:?}", &got[0][..2], &vals[0][..3]);
            }
            // staged run with a decryption after every stage: max |slot| and the best scalar fit out ~ g in
            let vals: Vec<Vec<Complex>> = vec![rand_real(7777, slots, 1.0)];
            fresh_at(&mut x, &mut ex, 1, &vals, top)?;
            let r = Route { cf: Some(a.cfs[0]), ..base_route };
            let st_api = route.staged().ok_or("staged")?;
            let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
            let show = |dec: &mut Decryptor<'_, W>, cx: &mut BtsCx<'_, '_, W, M>, x: &Ct<'_, W>, what: &str| -> Res<()> {
                if x.meta.deg == 2 {
                    println!("[pc-bench] debug {what}: degree 2 (not decryptable from the bottom level without a rescale)");
                    return Ok(());
                }
                let got = dec.decrypt_slots(cx.ctx, &cx.ex.compute, &enc, x, Payload::Complex, Which::Dense)?;
                let mx = got[0].iter().fold(0f64, |a, v| a.max(v.abs()));
                let num: f64 = got[0].iter().zip(&vals[0]).map(|(g, v)| g.re * v.re).sum();
                let den: f64 = vals[0].iter().map(|v| v.re * v.re).sum();
                let gfit = num / den;
                let res = got[0].iter().zip(&vals[0]).map(|(g, v)| (g.re - gfit * v.re).abs()).fold(0f64, f64::max);
                println!("[pc-bench] debug {what}: meta {:?} max|slot| {mx:.3e} fit g {gfit:.4e} max residual {res:.3e}; slots[0..3] {:?}", (x.meta.level, x.meta.deg, x.meta.nf), &got[0][..3]);
                Ok(())
            };
            show(&mut dec, &mut cx, &x, "input")?;
            let mut st = st_api.begin(&cx, std::slice::from_ref(&x), &r, false)?;
            st_api.mod_raise(&mut cx, &mut st, std::slice::from_mut(&mut x))?;
            show(&mut dec, &mut cx, &x, "raise")?;
            for k in 0..st.n_cts {
                st_api.cts_stage(&mut cx, &st, std::slice::from_mut(&mut x), k)?;
                show(&mut dec, &mut cx, &x, &format!("cts{k}"))?;
            }
            st_api.eval_mod(&mut cx, &st, std::slice::from_mut(&mut x))?;
            show(&mut dec, &mut cx, &x, "eval_mod")?;
            st_api.stc_enter(&mut cx, &st, std::slice::from_mut(&mut x))?;
            // host reference of the StC chain (OpenFHE order) on the decrypted EvalMod output
            let f = pc_core::bts_setup::cts_stc_factors(n, (n / 2) as u32, [4, 3], [0, 0], 1.0, pc_gpu::bts::stc::scale_dec(&ctx.scale))?;
            let z0 = dec.decrypt_slots(cx.ctx, &cx.ex.compute, &enc, &x, Payload::Complex, Which::Dense)?;
            let mut z: Vec<pc_core::bts_setup::C64> = z0[0].iter().map(|c| pc_core::bts_setup::C64::new(c.re, c.im)).collect();
            for k in 0..st.n_stc {
                st_api.stc_stage(&mut cx, &st, std::slice::from_mut(&mut x), k)?;
                show(&mut dec, &mut cx, &x, &format!("stc{k}"))?;
                z = f.stc[k].apply(&z);
                let got = dec.decrypt_slots(cx.ctx, &cx.ex.compute, &enc, &x, Payload::Complex, Which::Dense)?;
                let mx = z.iter().fold(0f64, |a, v| a.max(v.abs()));
                let mut best = (f64::MAX, 0usize);
                for r in 0..slots {
                    for &rr in &[r] {
                        let d = (0..64).map(|i| { let g = got[0][(i + rr) % slots]; let w = z[i]; ((g.re - w.re).powi(2) + (g.im - w.im).powi(2)).sqrt() }).fold(0f64, f64::max);
                        if d < best.0 { best = (d, rr); }
                    }
                }
                let full = (0..slots).map(|i| { let g = got[0][(i + best.1) % slots]; let w = z[i]; ((g.re - w.re).powi(2) + (g.im - w.im).powi(2)).sqrt() }).fold(0f64, f64::max);
                println!("[pc-bench] debug   host StC stage {k}: max|ref| {mx:.3e}; best rotation {} max diff (64 slots) {:.3e}, all slots {full:.3e}", best.1, best.0);
            }
            st_api.finish(&mut cx, &st, std::slice::from_mut(&mut x))?;
            cx.ops.rescale(&ctx, cx.ex, &sh, std::slice::from_mut(&mut x))?;
            show(&mut dec, &mut cx, &x, "finish (realized)")?;
        }
        c => return Err(format!("unknown command {c}").into()),
    }
    Ok(())
}

/// `pc-bench modmul`: the three u32 strategies on the same inputs, one table; exit 1 if their outputs differ.
fn modmul(a: &[String]) -> Res<bool> {
    use pc_core::modmul::{Barrett32, Montgomery};
    let mut o = word_seam::RigOpts { check: false, bench: true, reps: 20, ks: vec![1, 8], levels: vec![], seed: 0x3d };
    let mut u64row = false;
    let mut i = 0;
    while i < a.len() {
        match a[i].as_str() {
            "--k" => o.ks = a.get(i + 1).ok_or("--k")?.split(',').map(|x| x.parse()).collect::<Result<_, _>>()?,
            "--reps" => o.reps = a.get(i + 1).ok_or("--reps")?.parse()?,
            "--u64" => {
                u64row = true;
                i += 1;
                continue;
            }
            x => return Err(format!("unknown argument {x}").into()),
        }
        i += 2;
    }
    let n32 = Chain::generate(&ChainSpec::n32())?;
    // interleaved twice (B64, B32, Mont, B64, B32, Mont): the box is noisy, the table keeps the per-op minimum
    let mut runs: Vec<(&str, word_seam::RigOut)> = Vec::new();
    for _round in 0..2 {
        runs.push(("u32xBarrett64", word_seam::run::<u32, Barrett64>(n32.clone(), &o)?));
        runs.push(("u32xBarrett32", word_seam::run::<u32, Barrett32>(n32.clone(), &o)?));
        runs.push(("u32xMontgomery", word_seam::run::<u32, Montgomery>(n32.clone(), &o)?));
    }
    if u64row {
        runs.push(("u64xBarrett64 (n64)", word_seam::run::<u64, Barrett64>(Chain::generate(&ChainSpec::n64())?, &o)?));
    }
    for (name, r) in &runs {
        word_seam::print(&format!("pc-bench modmul {name}"), r);
    }
    let names: Vec<&str> = {
        let mut v: Vec<&str> = runs.iter().map(|r| r.0).collect();
        v.dedup();
        v.sort();
        v.dedup();
        v
    };
    let best = |name: &str, op: &str, k: usize| runs.iter().filter(|r| r.0 == name).filter_map(|r| r.1.bench.iter().find(|b| b.0 == op && b.1 == k).map(|b| b.2)).fold(f64::MAX, f64::min);
    let ops: Vec<(String, usize)> = runs[0].1.bench.iter().map(|b| (b.0.clone(), b.1)).collect();
    println!("[pc-bench modmul] device ms per call, min of 2 interleaved runs x median of {} reps; level 0 (top {}), dense keys", o.reps, n32.q.len() - 1);
    println!("| op | K | {} |", names.join(" | "));
    println!("|---|---|{}", "---|".repeat(names.len()));
    for (op, k) in &ops {
        let cells: Vec<String> = names.iter().map(|nm| format!("{:.4}", best(nm, op, *k))).collect();
        println!("| {op} | {k} | {} |", cells.join(" | "));
    }
    // seam contract: every u32 strategy produces the same words
    let base = &runs[0].1.hashes;
    let mut ok = true;
    for (name, r) in runs.iter().filter(|r| r.0.starts_with("u32")) {
        let diff: Vec<&String> = r.hashes.iter().zip(base).filter(|(x, y)| x.1 != y.1).map(|(x, _)| &x.0).collect();
        println!("[pc-bench modmul] {name}: {} outputs, {} differ from u32xBarrett64 {:?}", r.hashes.len(), diff.len(), diff);
        ok &= diff.is_empty();
    }
    Ok(ok)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // the classic multi-stage sparse route: its own harness
    if args.iter().any(|a| a == "--classic") {
        let mode = match args.first().map(|s| s.as_str()) {
            Some("bts-prec") => bts_classic::Mode::Prec,
            _ => bts_classic::Mode::Bench,
        };
        return bts_classic::cmd(mode, &args[1..]);
    }
    // sparse routes (1 = SPRU, SparseB 256 / 512) run in their own harness
    let route = args.iter().position(|a| a == "--route").and_then(|i| args.get(i + 1)).cloned().unwrap_or_default();
    if route.parse::<u32>().is_ok() {
        let mode = match args.first().map(|s| s.as_str()) {
            Some("bts-prec") => bts_sparse::Mode::Prec,
            _ => bts_sparse::Mode::Bench,
        };
        return bts_sparse::cmd(mode, &args[1..]);
    }
    // every route in one process: route x K x {eager, batched, graph} + MemPlan asserts (src/bts_all.rs)
    if args.first().map(|s| s.as_str()) == Some("bts") && args.iter().any(|a| a == "--all") {
        return bts_all::cmd(&args[1..]);
    }
    if args.first().map(|s| s.as_str()) == Some("bts-prec") && args.iter().any(|a| a == "--all") {
        return bts_all::prec_all(&args[1..]);
    }
    // CCMM per call / SinCBoot / precision (src/check_ccmm.rs)
    if args.first().map(|s| s.as_str()) == Some("ccmm") {
        return check_ccmm::cmd(check_ccmm::Mode::Bench, &args[1..]);
    }
    // PCMM per linear / precision (src/check_pcmm.rs)
    if args.first().map(|s| s.as_str()) == Some("pcmm") {
        return check_pcmm::cmd(check_pcmm::Mode::Bench, &args[1..]);
    }
    // a compiled block program replayed on the g0 oracle (src/replay_g0.rs)
    if args.first().map(|s| s.as_str()) == Some("replay") {
        return replay_g0::cmd(&args[1..]);
    }
    // the modmul seam table (src/word_seam.rs)
    if args.first().map(|s| s.as_str()) == Some("modmul") {
        return match modmul(&args[1..]) {
            Ok(true) => ExitCode::SUCCESS,
            Ok(false) => ExitCode::FAILURE,
            Err(e) => {
                eprintln!("pc-bench modmul: {e}");
                ExitCode::FAILURE
            }
        };
    }
    // the StC-first route, stc_high (src/bts_stc_first.rs)
    if matches!(args.first().map(|s| s.as_str()), Some("stcf" | "stcf-prec" | "stcf-pair" | "stc-high")) {
        return bts_stc_first::cmd(bts_stc_first::Mode::Bench, &args);
    }
    match parse(&args).and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("pc-bench: {e}\nusage: pc-bench bts|bts-prec --route dense [--k 1,2,4,8] [--reps n] [--cf 2..8] [--real] [--amps ..] [--iters 1,2] [--out f] [--old f]");
            ExitCode::FAILURE
        }
    }
}
