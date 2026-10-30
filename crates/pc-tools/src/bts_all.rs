//! Every bootstrap route in one process: the route x K x {eager, batched, graph} table and the MemPlan assert per
//! (route, K).
//!
//!   pc-bench bts --all [--k 1,2,4,8] [--reps 10] [--routes dense,stcf,512,256,1]
//!
//! One setup per K; every route's tables and `BtsWork` are carved between aligned arena offsets and must equal the
//! MemPlan prediction. Per route and K, on K fresh inputs: `ref` (each alone), `eager` (one K-batch `Ct`), `batched`
//! (`BtsBatch`), `graph` (`BtsGraphs`); device ms median of `reps` after 2 warm-ups; every output must equal its input's
//! reference words. Keys fresh, generated on the device.
use pc_core::encode::{Complex, Encoder, Payload};
use pc_core::keys::SecretKey;
use pc_core::modmul::Barrett64;
use pc_core::params::{Chain, ChainSpec};
use pc_cuda::{Arena, Compute, Device, Event};
use pc_gpu::bts::batch::{run_batch, BtsBatch};
use pc_gpu::bts::cts::LtGeom;
use pc_gpu::bts::finish::Pow2;
use pc_gpu::bts::raise::EncapsKeys;
use pc_gpu::bts::sparse::{chain_deg, SparseBRoute, SpruRoute};
use pc_gpu::bts::sparse_b::{SparseBCfg, SparseBPrecomp};
use pc_gpu::bts::spru::{Spru, SpruCfg, SpruKey};
use pc_gpu::bts::stc_first::{StcFirstPlan, StcFirstSetup};
use pc_gpu::bts::{BtsCx, BtsKeyRing, BtsRoute, BtsSetup, BtsWork, BtsWorkShape};
use pc_gpu::decrypt::{Decryptor, Which};
use pc_gpu::encrypt::Encryptor;
use pc_gpu::graph::{BtsGraphs, CaptureLock, Served};
use pc_gpu::keygen::{test_prng, KeyGen};
use pc_gpu::keys::{DevSecret, KeySet, KeySetSpec, KskLayout, KskShape, RotSpec};
use pc_gpu::ks::{KsPlans, KsWork, KskSource};
use pc_gpu::ops::{Ops, OpsCfg};
use pc_gpu::poly::CtBuf;
use pc_gpu::{Context, Ct, Executor, MemCfg, MemPlan, PoolClass};
use pc_shadow::{BtsConfig, CtMeta, Quirks, Route, Shadow, ShadowConfig};
use std::collections::BTreeSet;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

type W = u32;
type M = Barrett64;
type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Reference K = 1 ms of the FIDESlib stack (perseus night: shipped dense = StC-first + real 28.8, SparseB 512 18.6, classic s1 16.1).
fn old_ms(name: &str) -> Option<f64> {
    match name {
        "dense" | "stcf" => Some(28.8),
        "512" => Some(18.6),
        "1" => Some(16.1),
        _ => None,
    }
}

struct Opts {
    ks: Vec<usize>,
    reps: usize,
    routes: Vec<String>,
    memdiag: bool,
    /// key storage (`--keys dense|packed`): sessions run packed keys (no resident `a`: every key switch and every
    /// CtS baby regenerates it); default dense
    packed: bool,
    /// `--grid cfs:amps`: decrypted bits of one refresh per route x CF x amplitude (no timing), over `--seeds` inputs
    grid: Option<(Vec<u32>, Vec<f64>)>,
    seeds: usize,
}

fn parse(a: &[String]) -> Res<Opts> {
    let mut o = Opts { ks: vec![1, 2, 4, 8], reps: 10, routes: ["dense", "stcf", "512", "256", "1"].iter().map(|s| s.to_string()).collect(), memdiag: false, packed: false, grid: None, seeds: 1 };
    let mut i = 0;
    while i < a.len() {
        let v = a.get(i + 1).cloned().unwrap_or_default();
        match a[i].as_str() {
            "--memdiag" => {
                o.memdiag = true;
                i += 1;
                continue;
            }
            "--all" | "bts" => {
                i += 1;
                continue;
            }
            "--k" => o.ks = v.split(',').map(|x| x.parse()).collect::<std::result::Result<_, _>>()?,
            "--reps" => o.reps = v.parse()?,
            "--routes" => o.routes = v.split(',').map(|s| s.to_string()).collect(),
            "--seeds" => o.seeds = v.parse()?,
            "--grid" => {
                let (c, a) = v.split_once(':').ok_or("--grid cfs:amps")?;
                let cfs = c.split(',').map(|x| x.parse()).collect::<std::result::Result<_, _>>()?;
                let amps = a.split(',').map(|x| x.parse()).collect::<std::result::Result<_, _>>()?;
                o.grid = Some((cfs, amps));
            }
            "--keys" => {
                o.packed = match v.as_str() {
                    "packed" => true,
                    "dense" => false,
                    _ => return Err(format!("--keys {v}: dense | packed").into()),
                }
            }
            f => return Err(format!("unknown flag {f}").into()),
        }
        i += 2;
    }
    Ok(o)
}

pub fn cmd(args: &[String]) -> ExitCode {
    match parse(args).and_then(|o| run(&o)) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("[bts-all] error: {e}");
            ExitCode::from(2)
        }
    }
}

fn rot_steps(geoms: &[LtGeom], n: usize) -> BTreeSet<i64> {
    let mut steps = BTreeSet::new();
    for g in geoms {
        for i in 1..g.b {
            steps.insert(((i * g.stride) % (n / 2)) as i64);
        }
        if g.g > 1 {
            steps.insert(((g.b * g.stride) % (n / 2)) as i64);
        }
        steps.insert(g.offset.rem_euclid((n / 2) as i64));
    }
    steps.remove(&0);
    steps
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn rand_vals(seed: u64, n: usize, period: usize) -> Vec<Complex> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let base: Vec<f64> = (0..period)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        })
        .collect();
    (0..n).map(|j| Complex::new(base[j % period], 0.0)).collect()
}

/// One route of the table.
struct RouteCase {
    route: Route,
    /// Entry level (primes dropped) of the fresh inputs, and their slot count / data period.
    level: u32,
    slots: u32,
    period: usize,
}

fn case(name: &str, n: usize, nq: usize, lean: bool) -> Res<RouteCase> {
    let n2 = (n / 2) as u32;
    // input levels: the dense ones two levels above the StC-first entry, SparseB at its deepest (the g0 inputs' top 5
    // on n32; the level above the single prime on a lean chain)
    let (dense_lv, sparse_lv) = if lean { (44, 48) } else { (nq as u32 - 10, nq as u32 - 6) };
    Ok(match name {
        // shipped-order dense, real payload, CF 6 (the dense timing case), input at the reactive ceiling (44 on n32)
        "dense" => RouteCase { route: Route { cf: Some(6), real: true, ..Route::dense() }, level: dense_lv, slots: n2, period: n / 2 },
        // StC-first, real payload, CF 13 (the shipped dense route and timing case)
        "stcf" => RouteCase { route: Route { cf: Some(13), real: true, stc_first: true, ..Route::dense() }, level: dense_lv, slots: n2, period: n / 2 },
        // SparseB, raise drop 1, CF 6, input at top 5 (the g0 bootstrap inputs' top)
        "512" | "256" => {
            let s: u32 = name.parse()?;
            RouteCase { route: Route { slots: s, raise_drop: 1, sparse_b: true, cf: Some(6), ..Route::dense() }, level: sparse_lv, slots: s, period: s as usize }
        }
        // SPRU h = 64, CF 6, fresh level-26 input
        "1" => RouteCase { route: Route { slots: 1, spru_h: 64, cf: Some(6), ..Route::dense() }, level: 26, slots: 1, period: 1 },
        o => return Err(format!("unknown route {o}").into()),
    })
}

/// The chain: a preset (`PC_BENCH_CHAIN`, default n32), or a timing proxy of another shape: `PC_BENCH_LEVELS` (CKKS
/// levels incl. q0) and `PC_BENCH_DNUM` (the HE-standard check off: timing only, never keys anyone uses).
fn bench_spec() -> Res<ChainSpec> {
    let mut spec = ChainSpec::preset(&std::env::var("PC_BENCH_CHAIN").unwrap_or_else(|_| "n32".into()))?;
    let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<u32>().ok());
    if let Some(l) = env("PC_BENCH_LEVELS") {
        spec.num_levels = l;
        spec.security = pc_core::params::Security::NotSet;
    }
    if let Some(d) = env("PC_BENCH_DNUM") {
        spec.dnum = d;
        spec.security = pc_core::params::Security::NotSet;
    }
    Ok(spec)
}

/// One table row.
struct Row {
    route: String,
    k: usize,
    eager: f64,
    batched: f64,
    graph: f64,
    gs: f64,
    cap_ms: f64,
    bad: usize,
    words: usize,
}

fn run(o: &Opts) -> Res<bool> {
    let mut rows = Vec::new();
    let mut mem_rows: Vec<(String, usize, &'static str, usize, usize)> = Vec::new();
    for &kb in &o.ks {
        bench_k(o, kb, &mut rows, &mut mem_rows)?;
    }
    println!("\n[bts-all] MemPlan prediction vs carved (bytes), per (route, K):");
    println!("| route | K | section | predicted | carved | |");
    println!("|---|---|---|---|---|---|");
    let mut mem_bad = 0;
    for (r, k, sec, p, c) in &mem_rows {
        mem_bad += (p != c) as usize;
        println!("| {r} | {k} | {sec} | {p} | {c} | {} |", if p == c { "==" } else { "DIFFER" });
    }
    println!("\n[bts-all] route x K (device ms per ciphertext; median of {} after 2 warm-ups):", o.reps);
    println!("| route | K | eager | batched (gather) | graph replay | gather+scatter ms/call | capture+instantiate host ms | mismatches | old K=1 |");
    println!("|---|---|---|---|---|---|---|---|---|");
    let mut bad = 0;
    for r in &rows {
        bad += r.bad;
        println!(
            "| {} | {} | {:.2} | {:.2} | {:.2} | {:.3} | {:.1} | {} of {} | {} |",
            r.route,
            r.k,
            r.eager,
            r.batched,
            r.graph,
            r.gs,
            r.cap_ms,
            r.bad,
            r.words,
            old_ms(&r.route).map(|x| format!("{x}")).unwrap_or_else(|| "-".into())
        );
    }
    println!("[bts-all] verdict: mismatches {bad}, MemPlan differences {mem_bad}");
    Ok(bad == 0 && mem_bad == 0)
}

#[allow(clippy::too_many_lines)]
fn bench_k(o: &Opts, kb: usize, out_rows: &mut Vec<Row>, mem_rows: &mut Vec<(String, usize, &'static str, usize, usize)>) -> Res<()> {
    let t0 = Instant::now();
    let chain = Chain::generate(&bench_spec()?)?;
    let n = chain.n;
    let nq = chain.q.len();
    let st_tab = pc_core::scale::ScaleTables::precompute(&chain)?;
    let want = |r: &str| o.routes.iter().any(|x| x == r);
    // dense + StC-first: one StcFirstRoute (its shipped order == DenseRoute word for word)
    // PC_EM_FUSE_RESCALE=1: the EvalMod relinearization-rescale fusion on every route (default off, as BtsSetup)
    let em_fuse = std::env::var("PC_EM_FUSE_RESCALE").is_ok_and(|v| v == "1");
    let mut setup = BtsSetup::n32_shipped();
    setup.k_max = kb;
    setup.real = true;
    setup.em_fuse_rescale = em_fuse;
    let mut sf = StcFirstSetup::n32_shipped();
    sf.cache_slots = 4;
    let plan = StcFirstPlan::new(&chain, &setup, &sf)?;
    let mut geoms: Vec<LtGeom> = Vec::new();
    for v in &plan.dense.variants {
        geoms.extend(v.cts.iter().chain(&v.stc).map(|h| h.geom));
    }
    geoms.extend(plan.input.iter().map(|h| h.geom));
    geoms.extend(plan.sources.iter().flatten().map(|s| s.geom));
    let mut steps = rot_steps(&geoms, n);
    let sbs: Vec<u32> = [512u32, 256].into_iter().filter(|s| want(&s.to_string())).collect();
    let sb_cfgs: Vec<SparseBCfg> = sbs
        .iter()
        .map(|&s| {
            let mut c = SparseBCfg::n32(s, 1);
            c.k_max = kb;
            c.em_fuse_rescale = em_fuse;
            c
        })
        .collect();
    for c in &sb_cfgs {
        steps.extend(pc_gpu::bts::sparse_b::rot_steps(n, c)?);
    }
    let mut scfg = SpruCfg::new(64);
    scfg.k_max = kb;
    if want("1") {
        steps.extend(scfg.rot_steps(n));
    }
    steps.remove(&0);
    let spec = KeySetSpec { relin: true, conj: true, rots: steps.iter().map(|&s| RotSpec { step: s, q_top: None }).collect(), encaps: true };

    let dev = Device::open()?;
    let mcfg = MemCfg { k_max: 8.max(kb), pool: vec![PoolClass { k: 1, count: 1 }], events: 64, scratch_cts: 2 };
    let mut mp = MemPlan::build::<W, M>(&chain, mcfg)?;
    let lay = mp.lay;
    let mut shapes: Vec<KskShape> = spec.keys(&chain)?.into_iter().map(|(_, s)| s).collect();
    shapes.push(KskShape::bottom(&chain)?);
    let layout = if o.packed { KskLayout::packed_for::<W>(&chain)? } else { KskLayout::Dense };
    mp.reserve("keys", spec.bytes::<W>(&chain, layout)?);
    mp.reserve("keygen", KeyGen::<W>::bytes(&lay, &shapes));
    mp.reserve("secret", 2 * DevSecret::<W>::bytes(&lay));
    mp.reserve("encrypt", Encryptor::<W>::bytes(&lay, false));
    mp.reserve("decrypt", Decryptor::<W>::bytes(&lay, chain.level_unit, 1));
    mp.reserve("ks.plans", KsPlans::bytes::<W>(&chain, mp.cfg.k_max)?);
    mp.reserve("ks.regen_a", KsWork::<W>::regen_bytes(&lay, chain.dnum()));
    let ocfg = OpsCfg::default();
    Ops::<W>::reserve(&mut mp, &ocfg);
    // predictions: route tables and each route's own work shape
    let (sf_req, sf_shape) = plan.mem::<W>()?;
    let mut pred: Vec<(String, &'static str, usize)> = Vec::new();
    pred.push(("dense+stcf".into(), "tables", sf_req.total()));
    pred.push(("dense+stcf".into(), "bts.work", BtsWork::<W>::bytes(&lay, sf_shape)));
    let mut sb_shapes = Vec::new();
    pred.push(("sparse".into(), "tails", SparseBPrecomp::<W>::tails_bytes(&chain, &st_tab, &lay, &sb_cfgs)?));
    for c in &sb_cfgs {
        let (b, ws) = SparseBPrecomp::<W>::bytes(&chain, &st_tab, &lay, c)?;
        pred.push((c.slots.to_string(), "tables", b + EncapsKeys::bytes() + Pow2::<W>::bytes(lay.rows())));
        pred.push((c.slots.to_string(), "bts.work", BtsWork::<W>::bytes(&lay, ws)));
        sb_shapes.push(ws);
    }
    let spru_shape = BtsWorkShape { k_max: 1, g_max: 1, em_slots: 0, em_kk: 1, spare: 0 };
    if want("1") {
        pred.push(("1".into(), "tables", Spru::<W>::bytes(&lay, &scfg, &chain)? + Spru::<W>::bind_bytes(&scfg, n)));
        pred.push(("1".into(), "bts.work", BtsWork::<W>::bytes(&lay, spru_shape)));
    }
    pred.push(("all".into(), "bts.batch", BtsGraphs::<W>::bytes(&lay, kb)));
    for (_, sec, b) in &pred {
        mp.reserve(sec, *b);
    }
    // inputs: pristine (k = 1 each), separate working cts, one contiguous K-batch
    let ctb = Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(&lay, 1));
    mp.reserve("bench.cts", (2 * kb + 1) * ctb);
    if o.memdiag {
        mp.reserve("bench.memdiag", 2 * sf_req.total());
    }
    println!("[bts-all] K={kb}: {} rotation keys; MemPlan {}", steps.len(), mp.report());
    let arena = mp.allocate(&dev)?;
    let mut ex = Executor::<W>::new(&dev, &arena, &mp)?;
    ex.ks.carve_regen(&arena, &lay)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &mp)?;
    let kplans = KsPlans::new(&arena, &ex.compute, &ctx, mp.cfg.k_max)?;
    let mut ops = Ops::new(&arena, &dev, &ex.compute, &ctx, ocfg)?;
    let mut keys = KeySet::<W>::carve(&arena, &chain, &spec, layout)?;
    let mut prng = test_prng(0x6d);
    let sk = SecretKey::sample(&mut prng, n)?;
    let mut sec = DevSecret::<W>::carve(&arena, lay)?;
    sec.load(&ctx, &ex.compute, &sk)?;
    let mut spru_sec = DevSecret::<W>::carve(&arena, lay)?;
    let mut kg = KeyGen::<W>::new(&arena, &ex.compute, &chain, &shapes)?;
    let tk = kg.generate_set(&ctx, &mut ex, &sec, &mut keys, &mut prng)?;
    println!("[bts-all] keygen {} keys in {:.1} s", tk.keys, tk.total);
    let mut encr = Encryptor::<W>::new(&arena, &ex.compute, lay, false, &mut prng)?;
    let mut dec = Decryptor::<W>::new(&arena, &ex.compute, lay, chain.level_unit, 1)?;
    dec.load_secret(&ex.compute, &sec)?;
    let sh = Shadow::new(ShadowConfig { chain: ctx.shape, bts: BtsConfig::n32_shipped(), quirks: Quirks::none() }, Some(Arc::new(ctx.scale.clone())));
    let align = Arena::bytes_for::<u8>(1);
    let mark = |a: &Arena| a.used_bytes().div_ceil(align) * align;
    let mut carved: Vec<usize> = Vec::new();
    let stage = ex.ks.stage_ptrs();
    // carve in prediction order
    let u = mark(&arena);
    let sfr = plan.build(&arena, &dev, &ex.compute, &ctx, &keys, stage)?;
    carved.push(mark(&arena) - u);
    if o.memdiag {
        // where the dense/StC-first prediction is off: every LT stage carved again on its own, measured vs predicted
        use pc_gpu::bts::cts::{LtPts, LtStage, PtSource};
        let hosts: Vec<_> = plan.dense.variants.iter().flat_map(|v| v.cts.iter().chain(&v.stc).chain(v.stc_real0.iter())).chain(plan.input.iter()).collect();
        for h in hosts {
            let u = mark(&arena);
            let pts = LtPts::new(&arena, &ex.compute, &ctx, h.top, PtSource::Coeffs(&h.coeffs))?;
            let _st = LtStage::new(&arena, &ex.compute, &ctx, h.geom, h.top, h.end, h.nf, pts, &keys, stage, kb)?;
            let c = mark(&arena) - u;
            let p = LtPts::<W>::coeffs_bytes(h.coeffs.len(), h.top + 1 + lay.kp, h.width, n) + LtStage::<W>::plan_bytes(&lay, &chain.primes(), &h.geom, h.top, h.end, kb)?;
            println!("[bts-all]   LT top {} width {} geom {:?}: carved {c} predicted {p} {}", h.top, h.width, h.geom, if c == p { "" } else { "<-" });
        }
        for (v, h) in sfr.variants.iter().zip(&plan.dense.variants) {
            let em_p = pc_gpu::bts::evalmod::EvalModPlan::<W>::table_bytes(&st_tab, &lay, h.em_in, &plan.dense.ps, setup.evalmod_da_its, kb)?;
            println!("[bts-all]   drop {}: EvalMod carved {} / real {:?} vs predicted {em_p}", h.drop, v.em.bytes(), v.em_real.as_ref().map(|e| e.bytes()));
        }
    }
    let u = mark(&arena);
    let mut sf_work = BtsWork::<W>::new(&arena, lay, sf_shape)?;
    carved.push(mark(&arena) - u);
    let mut sb_routes = Vec::new();
    let mut sb_works = Vec::new();
    let u = mark(&arena);
    let sb_tails = SparseBPrecomp::<W>::tails(&arena, &ex.compute, &ctx, &keys, &sb_cfgs)?;
    carved.push(mark(&arena) - u);
    for (c, ws) in sb_cfgs.iter().zip(&sb_shapes) {
        let u = mark(&arena);
        let pre = SparseBPrecomp::new(&arena, &ex.compute, &ctx, c.clone(), &keys, &sb_tails, stage)?;
        let encaps = EncapsKeys::new(&arena, &ex.compute, n, keys.dense_to_sparse()?, keys.sparse_to_dense()?, stage)?;
        let pow2 = Pow2::new(&arena, &ex.compute, &ctx)?;
        carved.push(mark(&arena) - u);
        sb_routes.push(SparseBRoute { slots: c.slots, ring_n: n, variants: vec![pre], encaps, pow2, cf_default: 6, cf_window: (2, 20), deg: chain_deg(&ctx)?, ks_source: if o.packed { KskSource::Regen } else { KskSource::Cached } });
        let u = mark(&arena);
        sb_works.push(BtsWork::<W>::new(&arena, lay, *ws)?);
        carved.push(mark(&arena) - u);
    }
    let mut spru_route = None;
    let mut spru_work = None;
    if want("1") {
        let u = mark(&arena);
        let skey = SpruKey::sample(&mut prng, n, &scfg)?;
        let mut sp = Spru::<W>::new(&arena, &ctx, &mut ex, &mut kg, &mut encr, &sec, &mut spru_sec, &sk, &skey, scfg, &sh, &mut prng)?;
        sp.bind(&arena, &ex.compute, &keys, ex.ks.stage_ptrs())?;
        let c = mark(&arena) - u;
        carved.push(c);
        spru_route = Some(SpruRoute { spru: std::cell::RefCell::new(sp), ring_n: n, cf_default: 6, deg: chain_deg(&ctx)?, carved: c });
        let u = mark(&arena);
        spru_work = Some(BtsWork::<W>::new(&arena, lay, spru_shape)?);
        carved.push(mark(&arena) - u);
    }
    let u = mark(&arena);
    let mut graphs = BtsGraphs::<W>::new(&arena, lay, kb, 64)?;
    carved.push(mark(&arena) - u);
    for ((r, s, p), c) in pred.iter().zip(&carved) {
        mem_rows.push((r.clone(), kb, s, *p, *c));
    }
    let mut src: Vec<Ct<'_, W>> = (0..kb).map(|_| CtBuf::new(&arena, "bench.cts", lay, 1).map(|b| Ct::new(CtMeta::new(0, 1), b))).collect::<pc_gpu::Result<_>>()?;
    let mut sep: Vec<Ct<'_, W>> = (0..kb).map(|_| CtBuf::new(&arena, "bench.cts", lay, 1).map(|b| Ct::new(CtMeta::new(0, 1), b))).collect::<pc_gpu::Result<_>>()?;
    let mut probe = Ct::new(CtMeta::new(0, 1), CtBuf::new(&arena, "bench.cts", lay, 1)?);
    ex.compute.sync()?;
    println!("[bts-all] K={kb} setup {:.1} s; arena used {:.2} of {:.2} GB", t0.elapsed().as_secs_f64(), arena.used_bytes() as f64 / 1e9, arena.capacity() as f64 / 1e9);
    arena.freeze();
    let enc = Encoder::new(&chain);
    let (ev0, ev1) = (Event::timing(&dev)?, Event::timing(&dev)?);
    let ctw = lay.ct_words();
    for name in &o.routes {
        let c = case(name, n, nq, chain.lone_top().is_some())?;
        let (route, work, locks): (&dyn BtsRoute<'_, W, M>, &mut BtsWork<'_, W>, Vec<&dyn CaptureLock>) = match name.as_str() {
            "dense" | "stcf" => (&sfr, &mut sf_work, sfr.last.iter().map(|l| l as &dyn CaptureLock).collect()),
            "512" | "256" => {
                let i = sbs.iter().position(|&s| s.to_string() == *name).ok_or("route not built")?;
                (&sb_routes[i], &mut sb_works[i], vec![])
            }
            "1" => (spru_route.as_ref().ok_or("spru not built")?, spru_work.as_mut().ok_or("spru work")?, vec![]),
            o => return Err(format!("route {o}").into()),
        };
        let top = nq - 1 - c.level as usize;
        let sf_ = ctx.scale.sf(top as i32)?;
        if let Some((cfs, amps)) = &o.grid {
            // precision grid: one refresh per amplitude, CF and seed; bits = -log2 of the worst real-part error
            // relative to the input's max; rms: of the real-part error; min and mean over the seeds
            for &amp in amps {
                for &cf in cfs {
                    let (mut bits, mut bits_r) = (vec![], vec![]);
                    let mut lv = 0;
                    for seed in 0..o.seeds {
                        let mut meta = CtMeta::new(c.level, 1).with_nf(sf_);
                        meta.slots = c.slots;
                        let v: Vec<Complex> = rand_vals(31 + seed as u64, n / 2, c.period).into_iter().map(|z| Complex::new(amp * z.re, 0.0)).collect();
                        let msg = enc.encode_slots(&v, n / 2, top, Payload::Complex, sf_)?;
                        encr.encrypt_sk(&ctx, &ex.compute, &sec, &[msg], meta, &mut src[0], &mut prng)?;
                        let a = v.iter().map(|z| z.re.abs()).fold(0.0, f64::max);
                        let r = Route { cf: Some(cf), ..c.route };
                        sep[0].buf.buf.copy_from(&ex.compute, &src[0].buf.buf, ctw)?;
                        sep[0].meta = src[0].meta;
                        let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut *work };
                        route.run(&mut cx, std::slice::from_mut(&mut sep[0]), &r)?;
                        lv = sep[0].meta.level;
                        probe.buf.buf.copy_from(&ex.compute, &sep[0].buf.buf, ctw)?;
                        probe.meta = CtMeta { slots: 0, ..sep[0].meta };
                        ops.realize(&ctx, &mut ex, &sh, std::slice::from_mut(&mut probe))?;
                        let out = dec.decrypt_slots(&ctx, &ex.compute, &enc, &probe, Payload::Complex, Which::Dense)?.remove(0);
                        let e = out.iter().zip(&v).map(|(x, y)| (x.re - y.re).abs()).fold(0.0, f64::max) / a;
                        let er = (out.iter().zip(&v).map(|(x, y)| (x.re - y.re).powi(2)).sum::<f64>() / v.len() as f64).sqrt() / a;
                        bits.push(-e.log2());
                        bits_r.push(-er.log2());
                    }
                    let stat = |b: &[f64]| (b.iter().copied().fold(f64::INFINITY, f64::min), b.iter().sum::<f64>() / b.len() as f64);
                    let ((mn, me), (_, mer)) = (stat(&bits), stat(&bits_r));
                    println!("[bts-all] grid {name} cf {cf} A {amp}: bits min {mn:.2} mean {me:.2}; rms {mer:.2} ({} seeds), out level {lv}", o.seeds);
                }
            }
            continue;
        }
        // fresh inputs
        for (i, x) in src.iter_mut().enumerate() {
            let mut meta = CtMeta::new(c.level, 1).with_nf(sf_);
            meta.slots = c.slots;
            let v = rand_vals(31 + i as u64, n / 2, c.period);
            let msg = enc.encode_slots(&v, n / 2, top, Payload::Complex, sf_)?;
            encr.encrypt_sk(&ctx, &ex.compute, &sec, &[msg], meta, x, &mut prng)?;
        }
        let restore = |ex: &Executor<'_, W>, dst: &mut Ct<'_, W>, s: &Ct<'_, W>| -> Res<()> {
            dst.buf.buf.copy_from(&ex.compute, &s.buf.buf, ctw)?;
            dst.meta = s.meta;
            Ok(())
        };
        // reference: each input alone
        let mut refs: Vec<Vec<W>> = Vec::new();
        let mut ref_meta = None;
        for i in 0..kb {
            restore(&ex, &mut sep[i], &src[i])?;
            let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut *work };
            route.run(&mut cx, std::slice::from_mut(&mut sep[i]), &c.route)?;
            let t = sep[i].top(ctx.shape)?;
            let rows: Vec<usize> = (0..=t).collect();
            let mut w = sep[i].buf.download_rows(&ex.compute, 0, 0, &rows)?;
            w.extend(sep[i].buf.download_rows(&ex.compute, 0, 1, &rows)?);
            refs.push(w);
            ref_meta = Some(sep[i].meta);
        }
        let ref_meta = ref_meta.ok_or("no reference")?;
        let mut bad = 0;
        let mut words = 0;
        let mut cmp = |ex: &Executor<'_, W>, x: &Ct<'_, W>, c0: usize, i: usize| -> Res<()> {
            let t = x.top(ctx.shape)?;
            let rows: Vec<usize> = (0..=t).collect();
            let mut w = x.buf.download_rows(&ex.compute, c0, 0, &rows)?;
            w.extend(x.buf.download_rows(&ex.compute, c0, 1, &rows)?);
            words += refs[i].len();
            bad += if w.len() != refs[i].len() { refs[i].len() } else { w.iter().zip(&refs[i]).filter(|(a, b)| a != b).count() };
            if x.meta != ref_meta {
                bad += 1;
                println!("[bts-all]   {name} K={kb} member {i}: meta {:?} vs reference {:?}", x.meta, ref_meta);
            }
            Ok(())
        };
        // eager: one contiguous K-batch (the inputs gathered into the batch slots, untimed)
        let mut em = Vec::new();
        for rep in 0..o.reps + 2 {
            let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut *work };
            {
                let b = BtsBatch::new(c.route, src.iter_mut().collect())?;
                graphs.slots.gather(&cx, &b)?;
            }
            ev0.record(&cx.ex.compute)?;
            route.run(&mut cx, std::slice::from_mut(&mut graphs.slots.x), &c.route)?;
            ev1.record(&cx.ex.compute)?;
            ev1.sync()?;
            if rep >= 2 {
                em.push(ev0.elapsed_ms(&ev1)? as f64);
            }
        }
        for i in 0..kb {
            cmp(&ex, &graphs.slots.x, i, i)?;
        }
        // batched: K separate cts, gather -> run -> scatter
        let mut bm = Vec::new();
        for rep in 0..o.reps + 2 {
            for i in 0..kb {
                restore(&ex, &mut sep[i], &src[i])?;
            }
            ev0.record(&ex.compute)?;
            {
                let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut *work };
                let mut b = BtsBatch::new(c.route, sep.iter_mut().collect())?;
                run_batch(route, &mut cx, &mut graphs.slots, &mut b)?;
            }
            ev1.record(&ex.compute)?;
            ev1.sync()?;
            if rep >= 2 {
                bm.push(ev0.elapsed_ms(&ev1)? as f64);
            }
        }
        for i in 0..kb {
            cmp(&ex, &sep[i], 0, i)?;
        }
        // gather + scatter alone (the address-stability cost)
        let mut gsm = Vec::new();
        for _ in 0..o.reps {
            let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut *work };
            let mut b = BtsBatch::new(c.route, sep.iter_mut().collect())?;
            ev0.record(&cx.ex.compute)?;
            graphs.slots.gather(&cx, &b)?;
            graphs.slots.x.meta = ref_meta;
            graphs.slots.scatter(&mut cx, &mut b)?;
            ev1.record(&cx.ex.compute)?;
            ev1.sync()?;
            gsm.push(ev0.elapsed_ms(&ev1)? as f64);
        }
        // graph: first call captures, the rest replay
        let caps0 = graphs.stats;
        let mut gm = Vec::new();
        for rep in 0..o.reps + 3 {
            for i in 0..kb {
                restore(&ex, &mut sep[i], &src[i])?;
            }
            ev0.record(&ex.compute)?;
            let served = {
                let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut *work };
                let mut b = BtsBatch::new(c.route, sep.iter_mut().collect())?;
                graphs.run_route(route, &mut cx, &mut b, &locks)?
            };
            ev1.record(&ex.compute)?;
            ev1.sync()?;
            if rep == 0 && served != Served::Captured || rep > 0 && served != Served::Replayed {
                return Err(format!("{name} K={kb} rep {rep}: served {served:?}").into());
            }
            if rep == 0 || rep >= 3 {
                // the eager first call and the replays are both outputs: check them
                for i in 0..kb {
                    cmp(&ex, &sep[i], 0, i)?;
                }
            }
            if rep >= 3 {
                gm.push(ev0.elapsed_ms(&ev1)? as f64);
            }
        }
        let cap_ms = graphs.stats.capture_ms - caps0.capture_ms;
        let row = Row { route: name.clone(), k: kb, eager: median(em) / kb as f64, batched: median(bm) / kb as f64, graph: median(gm) / kb as f64, gs: median(gsm), cap_ms, bad, words };
        println!(
            "[bts-all] route {name} K={kb}: eager {:.2} / batched {:.2} / graph {:.2} ms/ct; gather+scatter {:.3} ms/call; capture {:.1} ms host; out level {} deg {}; mismatches {} of {}",
            row.eager, row.batched, row.graph, row.gs, row.cap_ms, ref_meta.level, ref_meta.deg, row.bad, row.words
        );
        out_rows.push(row);
    }
    {
        let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut sf_work };
        println!("[bts-all] K={kb}: {} graphs, {} constant slots leased; stats {:?}", graphs.len(), cx.ops.ring.leased(), graphs.stats);
        graphs.clear(&mut cx)?;
    }
    Ok(())
}

// ── bts-prec --all ───────────────────────────────────────────────────────────────────────────────────────────────

const OLD_N32: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../python/perseus/plan/data/bts_accuracy_n32.json");
const OLD_N32L: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../python/perseus/plan/data/bts_accuracy_n32l.json");
const CFS: &str = "2,4,6,8,10,12";
const AMPS: &str = "0.01,0.1,0.5,1,2,5,10";

/// Old table row (route, period, cf, A) -> rel_err.
fn old_rel(doc: &serde_json::Value, route: u64, period: u64, cf: u64, amp: f64) -> Option<f64> {
    doc["wall"].as_array()?.iter().find(|r| r["route"] == route && r["period"] == period && r["cf"] == cf && (r["amp"].as_f64().unwrap_or(-1.0) - amp).abs() < 1e-12).and_then(|r| r["rel_err"].as_f64())
}

/// `pc-bench bts-prec --all [--out docs/acc_n32.json]`: every route x CF through the routes' own precision harnesses
/// (same methodology as their cards), merged into one file, then the route x CF summary.
pub fn prec_all(args: &[String]) -> ExitCode {
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| "docs/acc_n32.json".into());
    match prec_all_run(&out) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("[bts-prec-all] error: {e}");
            ExitCode::from(2)
        }
    }
}

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

fn prec_all_run(out: &str) -> Res<bool> {
    let mut ok = true;
    // dense, shipped order (complex and real payload routes)
    for real in [false, true] {
        let mut a = s(&["bts-prec", "--route", "dense", "--cf", CFS, "--amps", AMPS, "--out", out, "--old", OLD_N32]);
        if real {
            a.push("--real".into());
        }
        crate::run(crate::parse(&a)?)?;
    }
    // StC-first (the shipped dense route): own CF and reference CF 20, complex and real
    let a = s(&["stcf-prec", "--cf", "2,4,6,8,10,12,13", "--amps", "0.1,0.5,1,2", "--real", "--out", out]);
    ok &= crate::bts_stc_first::bench(&a)?;
    // sparse routes: rows written to a side file, merged here with the reference table's row of the same (cf, A)
    let tmp = format!("{out}.sparse.tmp");
    let old = serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(OLD_N32)?)?;
    let oldl = serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(OLD_N32L)?)?;
    for (route, name, period, table, tname) in [(512u32, "sparse512", 512u64, &old, "bts_accuracy_n32.json"), (256, "sparse256", 256, &oldl, "bts_accuracy_n32l.json"), (1, "spru_h64", 1, &old, "bts_accuracy_n32.json")] {
        let o = crate::bts_sparse::parse(crate::bts_sparse::Mode::Prec, &s(&["--route", &route.to_string(), "--cf", CFS, "--amps", AMPS, "--write", &tmp]))?;
        ok &= crate::bts_sparse::run(&o)?;
        let mut rows: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(&tmp)?)?;
        for r in rows.iter_mut() {
            let (cf, amp) = (r["cf"].as_u64().unwrap_or(0), r["amp"].as_f64().unwrap_or(0.0));
            r["old_rel_err"] = old_rel(table, route as u64, period, cf, amp).into();
            r["old_table"] = tname.into();
        }
        let mut doc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(out)?)?;
        let arr = doc["rows"].as_array_mut().ok_or("rows")?;
        arr.retain(|r| r["route"] != name);
        arr.extend(rows);
        std::fs::write(out, serde_json::to_string_pretty(&doc)?)?;
        std::fs::remove_file(&tmp)?;
    }
    summary(out, &old)?;
    Ok(ok)
}

/// The route x CF table: bits at A = 1 and the worst cell over the amplitudes, each against the reference row.
fn summary(out: &str, old: &serde_json::Value) -> Res<()> {
    let mut doc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(out)?)?;
    let rows = doc["rows"].as_array().ok_or("rows")?.clone();
    // (label, filter, old reference (route, period) when the row carries none)
    type F = fn(&serde_json::Value) -> bool;
    let groups: [(&str, F, Option<(u64, u64)>); 7] = [
        ("dense (shipped order, complex)", |r| r["route"] == "dense" && r["real"] == false && r["iterations"] == 1, None),
        ("dense (shipped order, real)", |r| r["route"] == "dense" && r["real"] == true && r["iterations"] == 1, None),
        ("stc_first (complex, ref CF 20)", |r| r["route"] == "stc_first" && r["real"] == false && r["own_cf"] == false && r.get("payload").is_none(), Some((0, 0))),
        ("stc_first (real, ref CF 20)", |r| r["route"] == "stc_first" && r["real"] == true && r["own_cf"] == false && r.get("payload").is_none(), Some((0, 0))),
        ("sparse512 (SparseB, drop 1)", |r| r["route"] == "sparse512", None),
        ("sparse256 (SparseB, drop 1)", |r| r["route"] == "sparse256", None),
        ("spru_h64 (s = 1)", |r| r["route"] == "spru_h64", None),
    ];
    let mut table = Vec::new();
    println!("\n[bts-prec-all] route x CF (bits = -log2 max|Re out - in| / A; old = the old table's row of the same cell):");
    println!("| route | CF | bits A=1 | old A=1 | delta A=1 | min bits over A | median delta over A | min delta over A | cells |");
    println!("|---|---|---|---|---|---|---|---|---|");
    for (label, f, oref) in groups {
        let sel: Vec<&serde_json::Value> = rows.iter().filter(|r| f(r)).collect();
        let mut cfs: Vec<u64> = sel.iter().filter_map(|r| r["cf"].as_u64()).collect();
        cfs.sort();
        cfs.dedup();
        for cf in cfs {
            let cell: Vec<&&serde_json::Value> = sel.iter().filter(|r| r["cf"] == cf).collect();
            let bits = |r: &serde_json::Value| r["bits"].as_f64();
            let oldb = |r: &serde_json::Value| -> Option<f64> {
                let o = r["old_rel_err"].as_f64().or_else(|| oref.and_then(|(rt, p)| old_rel(old, rt, p, cf, r["amp"].as_f64()?)));
                o.map(|x| -x.log2())
            };
            let a1 = cell.iter().find(|r| (r["amp"].as_f64().unwrap_or(0.0) - 1.0).abs() < 1e-12);
            let b1 = a1.and_then(|r| bits(r));
            let o1 = a1.and_then(|r| oldb(r));
            let minb = cell.iter().filter_map(|r| bits(r)).fold(f64::INFINITY, f64::min);
            let mut deltas: Vec<f64> = cell.iter().filter_map(|r| Some(bits(r)? - oldb(r)?)).collect();
            deltas.sort_by(|a, b| a.total_cmp(b));
            let med = deltas.get(deltas.len() / 2).copied();
            let mind = deltas.first().copied();
            let f2 = |x: Option<f64>| x.map(|v| format!("{v:.2}")).unwrap_or_else(|| "-".into());
            println!("| {label} | {cf} | {} | {} | {} | {minb:.2} | {} | {} | {} |", f2(b1), f2(o1), f2(b1.zip(o1).map(|(a, b)| a - b)), f2(med), f2(mind), cell.len());
            table.push(serde_json::json!({"route": label, "cf": cf, "bits_a1": b1, "old_bits_a1": o1, "min_bits": minb, "median_delta": med, "min_delta": mind, "cells": cell.len()}));
        }
    }
    doc["summary"] = table.into();
    doc["meta"]["summary"] = "route x CF; bits at A = 1 and over every amplitude of the cell, deltas against the FIDESlib accuracy table row of the same (route, period, cf, A) (n32 Aug table; 256: n32l; stc_first against the old dense period-0 row, the only old StC-first data being the sylph Gaussian gate)".into();
    std::fs::write(out, serde_json::to_string_pretty(&doc)?)?;
    println!("[bts-prec-all] wrote {out}");
    Ok(())
}
