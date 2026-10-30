//! Sparse bootstrap routes on the device: cross-checks against the g0 dump, timing and precision.
//!
//!   pc-check spru \<g0> [--h 64]                       route 1 (SPRU) on every dumped s = 1 bootstrap input of g0:
//!                                                     decrypted output vs the oracle's dumped output and vs the
//!                                                     exact slot mean of the input
//!   pc-bench bts --route 1 [--k 1,8] [--g0 \<dir>]     device ms per bootstrap (CUDA events, median of 5 x reps)
//!   pc-bench bts-prec --route 1 [--cf 6,8,10,12]      rel_err = max |out - v| / A on A-anchored 1-periodic real data
//!                                                     (FIDESlib `tests/bts_accuracy_sweep.cu` methodology)
//!
//! Keys: generated on the device from the g0 secret (`sk.npy`), so the dumped ciphertexts decrypt and bootstrap
//! under them; the SPRU block key and its key ciphertexts are fresh (the dump has none).
use pc_core::encode::{Complex, Encoder, Payload};
use pc_core::keys::SecretKey;
use pc_core::modmul::Barrett64;
use pc_core::params::{Chain, ChainSpec};
use pc_cuda::{Device, Event};
use pc_gpu::bts::{BtsCx, BtsWork, BtsWorkShape};
use pc_gpu::bts::spru::{Spru, SpruCfg, SpruKey};
use pc_gpu::decrypt::{Decryptor, Which};
use pc_gpu::encrypt::Encryptor;
use pc_gpu::keygen::{test_prng, KeyGen};
use pc_gpu::keys::{DevSecret, KeySet, KeySetSpec, KskLayout, KskShape, RotSpec};
use pc_gpu::ks::{DevKey, KeyA, KsKeyRef, KsPlans, KsWork, KskSource};
use pc_gpu::ops::{Ops, OpsCfg};
use pc_gpu::{Context, Ct, CtPool, Executor, MemCfg, MemPlan, PoolClass};
use crate::dump::HostCt;
use crate::dump_g0::{npy_header, G0};
use pc_shadow::{BtsConfig, CtMeta, Quirks, Shadow, ShadowConfig};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

type W = u32;
type M = Barrett64;
type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub const G0_DEFAULT: &str = "/data02/users/mithrillo/projects/perseus-oracle/fx/dumps/g0";


/// Random complex slots (non-periodic), |re|, |im| <= a.
fn rand_slots(n2: usize, a: f64, seed: u64) -> Vec<Complex> {
    let mut g = rnd(seed);
    let mut u = move || (g() as f64 / u64::MAX as f64) * 2.0 - 1.0;
    (0..n2).map(|_| Complex::new(a * u(), a * u())).collect()
}

/// Print and judge one fold ladder case: max |out - ref| relative to max |ref|.
fn fold_verdict(what: &str, out: &[Complex], want: &[Complex], tol_bits: f64) -> bool {
    let m = cmax(want).max(1e-300);
    let e = cdiff(out, want);
    let bits = -(e / m).log2();
    println!("[fold] {what}: max|ref| {m:.3e}, max|out - ref| {e:.3e} ({bits:.1} bits) {}", if bits >= tol_bits { "OK" } else { "BAD" });
    bits >= tol_bits
}

fn read_sk(p: &Path) -> Res<Vec<i8>> {
    let (d, _shape, off) = npy_header(p)?;
    let b = std::fs::read(p)?;
    Ok(match d.trim_start_matches(['<', '|', '=']) {
        "i1" => b[off..].iter().map(|&x| x as i8).collect(),
        "i8" => b[off..].chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap_or([0; 8])) as i8).collect(),
        o => return Err(format!("{}: dtype {o}", p.display()).into()),
    })
}

fn meta_of(h: &HostCt, nq: usize) -> CtMeta {
    CtMeta { level: (nq - 1 - h.level) as u32, deg: h.noise_level as u8, nf: h.noise_factor, slots: 0, tag: Default::default() }
}

fn upload(ct: &mut Ct<'_, W>, s: &pc_cuda::Stream<pc_cuda::Compute>, h: &HostCt, nq: usize) -> Res<()> {
    for part in 0..2 {
        ct.buf.upload_rows(s, 0, part, &h.ids, &h.c[part])?;
    }
    ct.meta = meta_of(h, nq);
    Ok(())
}

/// Options shared by the subcommands.
#[derive(Clone, Debug)]
pub struct Opts {
    pub mode: Mode,
    pub g0: PathBuf,
    pub route: u32,
    pub h: usize,
    pub ks: Vec<usize>,
    pub cfs: Vec<u32>,
    pub amps: Vec<f64>,
    pub reps: usize,
    pub write: Option<PathBuf>,
    pub raise_drop: Option<u32>,
    /// `--ks-source cached|regen`: rotation / key-switch key source of the route (None = the route's default).
    pub ks_source: Option<KskSource>,
}

#[allow(dead_code)] // each binary dispatches a subset
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Check,
    Bench,
    Prec,
    /// fold_bootstrap ladder test (decrypted vs the host fold) and the bounded route set.
    Fold,
}

pub fn parse(mode: Mode, args: &[String]) -> Res<Opts> {
    let mut o = Opts {
        mode,
        g0: PathBuf::from(G0_DEFAULT),
        route: 1,
        h: 64,
        ks: vec![1],
        cfs: vec![6, 8, 10, 12],
        amps: vec![0.01, 0.1, 1.0, 3.0],
        reps: 10,
        write: None,
        raise_drop: None,
        ks_source: None,
    };
    let mut i = 0;
    let list = |s: &str| -> Res<Vec<f64>> { Ok(s.split(',').map(|x| x.parse::<f64>()).collect::<std::result::Result<_, _>>()?) };
    while i < args.len() {
        let v = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--g0" => o.g0 = PathBuf::from(&v),
            "--route" => o.route = v.parse()?,
            "--h" => o.h = v.parse()?,
            "--k" => o.ks = list(&v)?.into_iter().map(|x| x as usize).collect(),
            "--cf" => o.cfs = list(&v)?.into_iter().map(|x| x as u32).collect(),
            "--amps" => o.amps = list(&v)?,
            "--reps" => o.reps = v.parse()?,
            "--write" => o.write = Some(PathBuf::from(&v)),
            "--raise-drop" => o.raise_drop = Some(v.parse()?),
            "--ks-source" => {
                o.ks_source = Some(match v.as_str() {
                    "cached" => KskSource::Cached,
                    "regen" => KskSource::Regen,
                    _ => return Err(format!("--ks-source {v}: cached | regen").into()),
                })
            }
            d if !d.starts_with("--") => {
                o.g0 = PathBuf::from(d);
                i += 1;
                continue;
            }
            f => return Err(format!("unknown flag {f}").into()),
        }
        i += 2;
    }
    Ok(o)
}

pub fn cmd(mode: Mode, args: &[String]) -> ExitCode {
    match parse(mode, args).and_then(|o| run(&o)) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("[bts-sparse] error: {e}");
            ExitCode::from(2)
        }
    }
}

/// Median over 5 batches of `reps` runs, ms per run (device time on the compute lane).
fn time_it<'a>(dev: &Device, ex: &mut Executor<'a, W>, reps: usize, mut f: impl FnMut(&mut Executor<'a, W>) -> Res<()>) -> Res<(f64, f64)> {
    f(ex)?;
    ex.compute.sync()?;
    let mut v = Vec::new();
    let mut host = Vec::new();
    for _ in 0..5 {
        let (a, b) = (Event::timing(dev)?, Event::timing(dev)?);
        let t = Instant::now();
        a.record(&ex.compute)?;
        for _ in 0..reps {
            f(ex)?;
        }
        b.record(&ex.compute)?;
        let th = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        b.sync()?;
        v.push(a.elapsed_ms(&b)? as f64 / reps as f64);
        host.push(th);
    }
    v.sort_by(|x, y| x.total_cmp(y));
    host.sort_by(|x, y| x.total_cmp(y));
    Ok((v[2], host[2]))
}

pub fn run(o: &Opts) -> Res<bool> {
    if o.route != 1 {
        return run_sb(o);
    }
    let t0 = Instant::now();
    let g0 = G0::open(&o.g0)?;
    let chain = Chain::generate(&ChainSpec::n32())?;
    let qd: Vec<u64> = g0.params["q"].as_array().ok_or("params.q")?.iter().filter_map(|x| x.as_u64()).collect();
    if qd != chain.q {
        return Err("the dump's primes differ from the n32 preset".into());
    }
    let n = chain.n;
    let nq = chain.q.len();
    let sk = SecretKey { s: read_sk(&g0.dir.join("sk.npy"))?, s_sparse: vec![] };
    if sk.s.len() != n {
        return Err(format!("sk.npy has {} coefficients", sk.s.len()).into());
    }
    let cfg = SpruCfg::new(o.h);
    let steps = cfg.rot_steps(n);
    let spec = KeySetSpec { relin: true, conj: true, rots: steps.iter().map(|&step| RotSpec { step, q_top: None }).collect(), encaps: false };
    let layout = KskLayout::Dense;

    let dev = Device::open()?;
    let kmax = 8;
    let mcfg = MemCfg { k_max: kmax, pool: vec![PoolClass { k: 1, count: 8 }], events: 16, scratch_cts: 2 };
    let mut plan = MemPlan::build::<W, M>(&chain, mcfg)?;
    let lay = plan.lay;
    let dnum = chain.dnum();
    plan.reserve("ks.plans", KsPlans::bytes::<W>(&chain, kmax)?);
    plan.reserve("ks.regen_a", KsWork::<W>::regen_bytes(&lay, dnum));
    plan.reserve("keys", spec.bytes::<W>(&chain, layout)?);
    let mut shapes: Vec<KskShape> = spec.keys(&chain)?.into_iter().map(|(_, s)| s).collect();
    shapes.push(KskShape::bottom(&chain)?);
    plan.reserve("keygen", KeyGen::<W>::bytes(&lay, &shapes));
    plan.reserve("keys.secret", 2 * DevSecret::<W>::bytes(&lay));
    plan.reserve("encrypt", Encryptor::<W>::bytes(&lay, false));
    plan.reserve("decrypt", Decryptor::<W>::bytes(&lay, chain.level_unit, 1));
    Ops::<W>::reserve(&mut plan, &OpsCfg::default());
    let spru_bytes = Spru::<W>::bytes(&lay, &cfg, &chain)? + Spru::<W>::bind_bytes(&cfg, n);
    plan.reserve("bts.work", BtsWork::<W>::bytes(&lay, BtsWorkShape { k_max: 1, g_max: 1, em_slots: 0, em_kk: 1, spare: 0 }));
    plan.reserve("bts.spru", spru_bytes);
    println!("[bts-sparse] MemPlan {}", plan.report());
    let arena = plan.allocate(&dev)?;
    let mut ex = Executor::<W>::new(&dev, &arena, &plan)?;
    ex.ks.carve_regen(&arena, &lay)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &plan)?;
    let pool = CtPool::<W>::new(&arena, &plan)?;
    let kplans = KsPlans::new(&arena, &ex.compute, &ctx, kmax)?;
    let mut ops = Ops::<W>::new(&arena, &dev, &ex.compute, &ctx, OpsCfg::default())?;
    let mut prng = test_prng(0x6b);
    let mut secret = DevSecret::<W>::carve(&arena, lay)?;
    secret.load(&ctx, &ex.compute, &sk)?;
    let mut spru_secret = DevSecret::<W>::carve(&arena, lay)?;
    let mut keys = KeySet::<W>::carve(&arena, &chain, &spec, layout)?;
    let mut kg = KeyGen::<W>::new(&arena, &ex.compute, &chain, &shapes)?;
    let tk = Instant::now();
    let kt = kg.generate_set(&ctx, &mut ex, &secret, &mut keys, &mut prng)?;
    println!("[bts-sparse] keygen: {} keys in {:.2} s", kt.keys, tk.elapsed().as_secs_f64());
    let mut enc = Encryptor::<W>::new(&arena, &ex.compute, lay, false, &mut prng)?;
    let mut dec = Decryptor::<W>::new(&arena, &ex.compute, lay, chain.level_unit, 1)?;
    dec.load_secret(&ex.compute, &secret)?;
    let sh = Shadow::new(ShadowConfig { chain: ctx.shape, bts: BtsConfig::n32_shipped(), quirks: Quirks::none() }, Some(Arc::new(ctx.scale.clone())));
    let align = pc_cuda::Arena::bytes_for::<u8>(1);
    let used0 = arena.used_bytes().div_ceil(align) * align;
    let skey = SpruKey::sample(&mut prng, n, &cfg)?;
    let mut spru = Spru::<W>::new(&arena, &ctx, &mut ex, &mut kg, &mut enc, &secret, &mut spru_secret, &sk, &skey, cfg, &sh, &mut prng)?;
    spru.bind(&arena, &ex.compute, &keys, ex.ks.stage_ptrs())?;
    if let Some(src) = o.ks_source {
        spru.rot_source = src;
    }
    println!("[bts-sparse] SPRU rotation key source {:?}", spru.rot_source);
    let spru_used = arena.used_bytes().div_ceil(align) * align - used0;
    let wshape = BtsWorkShape { k_max: 1, g_max: 1, em_slots: 0, em_kk: 1, spare: 0 };
    let mut work = BtsWork::<W>::new(&arena, lay, wshape)?;
    ex.compute.sync()?;
    println!(
        "[bts-sparse] setup {:.1} s; spru material {:.1} MB (MemPlan bts.spru {:.1} MB); arena {} of {} B",
        t0.elapsed().as_secs_f64(),
        spru_used as f64 / 1e6,
        spru_bytes as f64 / 1e6,
        arena.used_bytes(),
        arena.capacity()
    );
    if spru_used != spru_bytes {
        return Err(format!("SPRU carved {spru_used} B != MemPlan section bts.spru {spru_bytes} B").into());
    }
    arena.freeze();
    let encoder = Encoder::new(&chain);
    let deg = 2u32; // log2(q0 / 2^scale_bits) on n32

    // decrypt helper: realize (copy) then bottom decryption, full slots
    let mut dct = pool.acquire(1, CtMeta::new(0, 1))?;
    let mut decrypt = |ex: &mut Executor<'_, W>, ops: &mut Ops<'_, W>, x: &Ct<'_, W>| -> Res<Vec<Complex>> {
        let d = dct.get_mut().ok_or("shared")?;
        let top = x.top(ctx.shape)?;
        let rows: Vec<usize> = (0..=top).collect();
        for part in 0..2 {
            let v = x.buf.download_rows(&ex.compute, 0, part, &rows)?;
            d.buf.upload_rows(&ex.compute, 0, part, &rows, &v)?;
        }
        d.meta = x.meta;
        d.meta.slots = 0;
        ops.realize(&ctx, ex, &sh, std::slice::from_mut(d))?;
        let mut v = dec.decrypt_slots(&ctx, &ex.compute, &encoder, d, Payload::Complex, Which::Dense)?;
        Ok(v.remove(0))
    };
    let mut x = pool.acquire(1, CtMeta::new(0, 1))?;
    let mut ok = true;
    let spru_used = spru_used;
    match o.mode {
        Mode::Check => {
            let bd = g0.bts();
            let mut nck = 0;
            for k in 1..=10 {
                if !bd.exists(&format!("{k}.json")) {
                    continue;
                }
                let j = bd.json(&format!("{k}.json"));
                if j["slots"].as_u64() != Some(1) {
                    continue;
                }
                let corr = j["state"]["correction"].as_u64().ok_or("state.correction")? as u32;
                let hin = bd.ct(&format!("{k}.in"));
                let hout = bd.ct(&format!("{k}.out"));
                let xm = x.get_mut().ok_or("shared")?;
                upload(xm, &ex.compute, &hin, nq)?;
                let vin = decrypt(&mut ex, &mut ops, xm)?;
                let mean = vin.iter().fold(Complex::default(), |a, b| Complex::new(a.re + b.re, a.im + b.im));
                let mean = Complex::new(mean.re / vin.len() as f64, mean.im / vin.len() as f64);
                let t = Instant::now();
                spru.run(&mut BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work }, xm, corr, 0)?;
                ex.compute.sync()?;
                let wall = t.elapsed().as_secs_f64() * 1e3;
                let ours = decrypt(&mut ex, &mut ops, xm)?;
                let ours_meta = xm.meta;
                upload(xm, &ex.compute, &hout, nq)?;
                let old = decrypt(&mut ex, &mut ops, xm)?;
                let amp = mean.abs().max(1e-300);
                let e_ours = ours.iter().map(|z| Complex::new(z.re - mean.re, z.im - mean.im).abs()).fold(0.0, f64::max);
                let e_old = old.iter().map(|z| Complex::new(z.re - mean.re, z.im - mean.im).abs()).fold(0.0, f64::max);
                let e_vs = ours.iter().zip(&old).map(|(a, b)| Complex::new(a.re - b.re, a.im - b.im).abs()).fold(0.0, f64::max);
                let lv_ok = ours_meta.level == meta_of(&hout, nq).level && ours_meta.deg as i64 == hout.noise_level;
                println!(
                    "[spru] bts {k} ({}): in top {} deg {}, cf corr {corr}; mean {:.6e}{:+.6e}i |m| {amp:.3e}; out top {} deg {} (old {} deg {}) {}; \
                     max|ours-mean| {e_ours:.3e} ({:.1} bits rel), max|old-mean| {e_old:.3e} ({:.1} bits rel), max|ours-old| {e_vs:.3e}; {wall:.2} ms wall",
                    j["step"].as_str().unwrap_or(""),
                    hin.level,
                    hin.noise_level,
                    mean.re,
                    mean.im,
                    nq - 1 - ours_meta.level as usize,
                    ours_meta.deg,
                    hout.level,
                    hout.noise_level,
                    if lv_ok { "level OK" } else { "LEVEL MISMATCH" },
                    -(e_ours / amp).log2(),
                    -(e_old / amp).log2(),
                );
                ok &= lv_ok && e_ours <= 2.0 * e_old.max(1e-9 * amp);
                nck += 1;
            }
            println!("[spru] {nck} dumped s=1 bootstraps checked: {}", if ok { "PASS" } else { "FAIL" });
        }
        Mode::Bench => {
            let level = 26u32;
            let top = nq - 1 - level as usize;
            let sf = ctx.scale.sf(top as i32)?;
            let vals: Vec<Complex> = vec![Complex::new(0.25, -0.125); n / 2];
            let rows = encoder.encode_slots(&vals, n / 2, top, Payload::Complex, sf)?;
            let m0 = CtMeta { level, deg: 1, nf: sf, slots: 0, tag: Default::default() };
            let mut saved = pool.acquire(1, m0)?;
            let sv = saved.get_mut().ok_or("shared")?;
            enc.encrypt_sk(&ctx, &ex.compute, &secret, &[rows], m0, sv, &mut prng)?;
            let words = sv.buf.buf.len();
            let xm = x.get_mut().ok_or("shared")?;
            for &k in &o.ks {
                let (ms, host) = time_it(&dev, &mut ex, o.reps, |ex| {
                    for _ in 0..k {
                        // the route consumes its input: restore it on the device first (one D2D copy, ~0.03 ms)
                        xm.buf.buf.copy_from(&ex.compute, &sv.buf.buf, words)?;
                        xm.meta = m0;
                        spru.run(&mut BtsCx { ctx: &ctx, ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work }, xm, 12 - deg, 0)?;
                    }
                    Ok(())
                })?;
                println!("[bench] bts route 1 (SPRU h={}) K={k}: {:.3} ms/call, {:.3} ms/ct (host issue {:.3} ms/call)", o.h, ms, ms / k as f64, host);
            }
            // phase profile (median of 10)
            let evs: Vec<Event> = (0..=pc_gpu::bts::spru::PHASES.len()).map(|_| Event::timing(&dev)).collect::<std::result::Result<_, _>>()?;
            let mut acc = vec![vec![]; pc_gpu::bts::spru::PHASES.len()];
            let mut hacc = vec![];
            for _ in 0..10 {
                xm.buf.buf.copy_from(&ex.compute, &sv.buf.buf, words)?;
                xm.meta = m0;
                ex.compute.sync()?;
                let th = Instant::now();
                spru.run_profiled(&mut BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work }, xm, 12 - deg, 0, Some(&evs))?;
                hacc.push(th.elapsed().as_secs_f64() * 1e3);
                ex.compute.sync()?;
                for (i, a) in acc.iter_mut().enumerate() {
                    a.push(evs[i].elapsed_ms(&evs[i + 1])? as f64);
                }
            }
            let mut line = String::from("[bench] SPRU phases (ms):");
            for (i, a) in acc.iter_mut().enumerate() {
                a.sort_by(|x, y| x.total_cmp(y));
                line += &format!(" {} {:.3}", pc_gpu::bts::spru::PHASES[i], a[5]);
            }
            hacc.sort_by(|x, y| x.total_cmp(y));
            println!("{line}; host issue of one call from an idle device {:.3} ms", hacc[5]);
        }
        Mode::Fold => {
            use pc_gpu::bts::fold::{fold_bootstrap, fold_reference, tag_reduce};
            use pc_gpu::bts::sparse::{chain_deg, RouteSet, SpruRoute};
            let route1 = SpruRoute { spru: std::cell::RefCell::new(spru), ring_n: n, cf_default: 6, deg: chain_deg(&ctx)?, carved: spru_used };
            let mut set = RouteSet::<W, M>::new(&[1]);
            set.add_route(1, &route1)?;
            let undeclared = set.add_route(512, &route1).is_err();
            println!("[fold] route set: declared {:?}, built {:?}; add_route(512) undeclared refused: {undeclared}", set.declared, set.built());
            ok &= undeclared;
            let n2 = n / 2;
            for &(n_live, prescale, a) in &[(n2 as u32, 1.0, 0.5), (1000u32, 0.25, 0.5)] {
                let v = rand_slots(n2, a, 0xf01d + n_live as u64);
                let top = nq - 1 - 20;
                let sf = ctx.scale.sf(top as i32)?;
                let rows = encoder.encode_slots(&v, n2, top, Payload::Complex, sf)?;
                let m0 = CtMeta { level: 20, deg: 1, nf: sf, slots: n2 as u32, tag: Default::default() };
                let xm = x.get_mut().ok_or("shared")?;
                enc.encrypt_sk(&ctx, &ex.compute, &secret, &[rows], m0, xm, &mut prng)?;
                let want_meta = sh.fold_bootstrap(&xm.meta, 1, n_live, prescale)?.out;
                let rt = pc_shadow::Route { slots: 1, spru_h: o.h as u32, cf: Some(8), ..pc_shadow::Route::dense() };
                fold_bootstrap(&mut BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work }, &set, xm, 1, n_live, prescale, &rt)?;
                let out = decrypt(&mut ex, &mut ops, xm)?;
                let want = fold_reference(&v, 1, n_live);
                ok &= fold_verdict(&format!("s=1 (SPRU) n_live {n_live} prescale {prescale}"), &out, &want, 10.0);
                println!("[fold]   output level {} deg {} (Shadow fold_bootstrap recipe: level {} deg {})", xm.meta.level, xm.meta.deg, want_meta.level, want_meta.deg);
                tag_reduce(&sh, xm, 1);
                println!("[fold]   tag_reduce(1): period {}", xm.meta.tag.period);
            }
            set.drop_route(1);
            let dropped = set.get(1).is_err();
            set.readd_route(1)?;
            println!("[fold] drop_route(1) -> get refused: {dropped}; readd -> built {:?}", set.built());
            ok &= dropped;
        }
        Mode::Prec => {
            let mut rows_out = vec![];
            println!("| cf | amp | rel_err | bits (Re) | bits (complex) |");
            println!("|---|---|---|---|---|");
            for &cf in &o.cfs {
                for &a in &o.amps {
                    // 1-periodic real data: one value, anchored at |v| = A (old sweep: uniform in [-A, A], anchored)
                    let v = -a;
                    let vals = vec![Complex::new(v, 0.0); n / 2];
                    let top = nq - 1;
                    let sf = ctx.scale.sf(top as i32)?;
                    let rows = encoder.encode_slots(&vals, n / 2, top, Payload::Real, sf)?;
                    let m0 = CtMeta { level: 0, deg: 1, nf: sf, slots: 0, tag: Default::default() };
                    let xm = x.get_mut().ok_or("shared")?;
                    enc.encrypt_sk(&ctx, &ex.compute, &secret, &[rows], m0, xm, &mut prng)?;
                    spru.run(&mut BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work }, xm, cf.saturating_sub(deg), 0)?;
                    ops.realize(&ctx, &mut ex, &sh, std::slice::from_mut(xm))?;
                    let (lv, dg) = (xm.meta.level, xm.meta.deg);
                    // a constant puts all its energy in coefficient 0: |A| sf > q0 q1 / 2 for A > 2 would wrap the
                    // bottom-limb decryption, so read it 2^-4 smaller (one level below the measured landing)
                    ops.mult_scalar_inplace(&ctx, &mut ex, &sh, std::slice::from_mut(xm), &[Complex::new(1.0 / 16.0, 0.0)])?;
                    let out: Vec<Complex> = decrypt(&mut ex, &mut ops, xm)?.into_iter().map(|z| Complex::new(16.0 * z.re, 16.0 * z.im)).collect();
                    xm.meta.level = lv;
                    xm.meta.deg = dg;
                    let e = out.iter().map(|z| (z.re - v).abs()).fold(0.0, f64::max) / a;
                    let ec = out.iter().map(|z| Complex::new(z.re - v, z.im).abs()).fold(0.0, f64::max) / a;
                    println!("| {cf} | {a} | {e:.3e} | {:.2} | {:.2} |", -e.log2(), -ec.log2());
                    rows_out.push(format!(
                        "{{\"amp\": {a}, \"bits\": {}, \"bits_complex\": {}, \"cf\": {cf}, \"input_level\": 0, \"iterations\": 1, \"landing_deg\": {}, \"landing_level\": {}, \"period\": 1, \"real\": false, \"rel_err\": {e:e}, \"rel_err_complex\": {ec:e}, \"route\": \"spru_h{}\"}}",
                        -e.log2(), -ec.log2(), xm.meta.deg, xm.meta.level, o.h
                    ));
                }
            }
            if let Some(p) = &o.write {
                std::fs::write(p, format!("[\n  {}\n]\n", rows_out.join(",\n  ")))?;
                println!("[prec] {} rows -> {}", rows_out.len(), p.display());
            }
        }
    }
    Ok(ok)
}

// ── routes 512 / 256 (SparseB) ───────────────────────────────────────────────────────────────────────────────────

/// The oracle's keys from the dump (CPU-map `b` + KSKA seed, `a` expanded): every SparseB route key of 512 and
/// 256 is in g0, so the dumped ciphertexts bootstrap under exactly the keys they were made with.
struct OracleRing<'a> {
    rot: BTreeMap<i64, DevKey<'a, W>>,
    conj: DevKey<'a, W>,
    relin: DevKey<'a, W>,
    atob: DevKey<'a, W>,
    btoa: DevKey<'a, W>,
    n: usize,
}

fn miss(what: String) -> pc_gpu::Error {
    pc_gpu::Error::Mem { knob: "oracle keys", value: 0, msg: format!("the dump has no {what} key") }
}

impl pc_gpu::bts::BtsKeyRing for OracleRing<'_> {
    fn rot(&self, step: i64) -> pc_gpu::Result<&dyn KsKeyRef> {
        let k = step.rem_euclid((self.n / 2) as i64);
        self.rot.get(&k).map(|k| k as &dyn KsKeyRef).ok_or_else(|| miss(format!("rotation {k}")))
    }
    fn conj(&self) -> pc_gpu::Result<&dyn KsKeyRef> {
        Ok(&self.conj)
    }
    fn relin(&self) -> pc_gpu::Result<&dyn KsKeyRef> {
        Ok(&self.relin)
    }
    fn dense_to_sparse(&self) -> pc_gpu::Result<&dyn KsKeyRef> {
        Ok(&self.atob)
    }
    fn sparse_to_dense(&self) -> pc_gpu::Result<&dyn KsKeyRef> {
        Ok(&self.btoa)
    }
}

fn load_key<'a>(arena: &'a pc_cuda::Arena, s: &pc_cuda::Stream<pc_cuda::Compute>, ctx: &Context<'_, W, M>, plans: &KsPlans<'_>, g0: &G0, name: &str, g: u64) -> Res<DevKey<'a, W>> {
    let file = if name == "mult" { "keys/mult.b.bin".to_string() } else { format!("keys/auto{name}.b.bin") };
    let b = crate::dump::read_u32(&g0.dir.join(file));
    let v: Vec<u32> = g0.params["keys"][name]["a_seed"].as_array().ok_or(format!("keys.{name}.a_seed"))?.iter().filter_map(|x| x.as_u64().map(|v| v as u32)).collect();
    let seed: [u32; 8] = v.try_into().map_err(|_| "seed must be 8 words")?;
    Ok(DevKey::new(arena, s, ctx, plans, g, ctx.chain.dnum(), &b, Some(seed), KeyA::Expand)?)
}

fn cmax(a: &[Complex]) -> f64 {
    a.iter().map(|z| z.abs()).fold(0.0, f64::max)
}
fn cdiff(a: &[Complex], b: &[Complex]) -> f64 {
    a.iter().zip(b).map(|(x, y)| Complex::new(x.re - y.re, x.im - y.im).abs()).fold(0.0, f64::max)
}

fn run_sb(o: &Opts) -> Res<bool> {
    use pc_gpu::bts::sparse::{chain_deg, SparseBRoute};
    use pc_gpu::bts::sparse_b::{rot_steps, SparseBCfg, SparseBPrecomp};
    use pc_gpu::bts::{raise::EncapsKeys, BtsRoute, Staged};
    let t0 = Instant::now();
    let g0 = G0::open(&o.g0)?;
    let chain = Chain::generate(&ChainSpec::n32())?;
    let qd: Vec<u64> = g0.params["q"].as_array().ok_or("params.q")?.iter().filter_map(|x| x.as_u64()).collect();
    if qd != chain.q {
        return Err("the dump's primes differ from the n32 preset".into());
    }
    let n = chain.n;
    let n2 = 2 * n as u64;
    let nq = chain.q.len();
    let sk = SecretKey { s: read_sk(&g0.dir.join("sk.npy"))?, s_sparse: vec![] };
    let rd = o.raise_drop.unwrap_or_else(|| g0.params["boot_routes"][o.route.to_string()]["raise_drop"].as_u64().unwrap_or(1) as u32);
    let mut scfg = SparseBCfg::n32(o.route, rd);
    scfg.k_max = 8;
    if let Some(src) = o.ks_source {
        scfg.ks_source = src;
    }
    let steps = rot_steps(n, &scfg)?;
    let st = pc_core::scale::ScaleTables::precompute(&chain)?;

    let dev = Device::open()?;
    let kmax = 8;
    let mcfg = MemCfg { k_max: kmax, pool: vec![PoolClass { k: 1, count: 4 }, PoolClass { k: kmax, count: 3 }], events: 16, scratch_cts: 2 };
    let mut plan = MemPlan::build::<W, M>(&chain, mcfg)?;
    let lay = plan.lay;
    let dnum = chain.dnum();
    plan.reserve("ks.plans", KsPlans::bytes::<W>(&chain, kmax)?);
    plan.reserve("ks.regen_a", KsWork::<W>::regen_bytes(&lay, dnum));
    plan.reserve("keys", (steps.len() + 4) * DevKey::<W>::bytes(&lay, dnum, true, true));
    plan.reserve("keys.secret", DevSecret::<W>::bytes(&lay));
    plan.reserve("encrypt", Encryptor::<W>::bytes(&lay, false));
    plan.reserve("decrypt", Decryptor::<W>::bytes(&lay, chain.level_unit, 1));
    Ops::<W>::reserve(&mut plan, &OpsCfg::default());
    let (sb_bytes, wshape) = SparseBPrecomp::<W>::bytes(&chain, &st, &lay, &scfg)?;
    plan.reserve("bts.sparse", sb_bytes + SparseBPrecomp::<W>::tails_bytes(&chain, &st, &lay, std::slice::from_ref(&scfg))?);
    plan.reserve("bts.route", EncapsKeys::bytes() + pc_gpu::bts::finish::Pow2::<W>::bytes(lay.rows()));
    plan.reserve("bts.work", BtsWork::<W>::bytes(&lay, wshape));
    plan.reserve("bts.slack", 64 << 20); // the route assert reports any MemPlan misprediction instead of overflowing
    println!("[bts-sparse] route {} (raise drop {rd}): {} rotation keys; MemPlan {}", o.route, steps.len(), plan.report());
    let arena = plan.allocate(&dev)?;
    let mut ex = Executor::<W>::new(&dev, &arena, &plan)?;
    ex.ks.carve_regen(&arena, &lay)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &plan)?;
    let pool = CtPool::<W>::new(&arena, &plan)?;
    let kplans = KsPlans::new(&arena, &ex.compute, &ctx, kmax)?;
    let mut ops = Ops::<W>::new(&arena, &dev, &ex.compute, &ctx, OpsCfg::default())?;
    let mut prng = test_prng(0x6b2);
    let mut secret = DevSecret::<W>::carve(&arena, lay)?;
    secret.load(&ctx, &ex.compute, &sk)?;
    let mut enc = Encryptor::<W>::new(&arena, &ex.compute, lay, false, &mut prng)?;
    let mut dec = Decryptor::<W>::new(&arena, &ex.compute, lay, chain.level_unit, 1)?;
    dec.load_secret(&ex.compute, &secret)?;
    let tk = Instant::now();
    let s = &ex.compute;
    let mut rot = BTreeMap::new();
    for &stp in &steps {
        let g = pc_core::galois::rotation_galois(stp, n)?;
        rot.insert(stp, load_key(&arena, s, &ctx, &kplans, &g0, &g.to_string(), g)?);
    }
    let ring = OracleRing {
        rot,
        conj: load_key(&arena, s, &ctx, &kplans, &g0, &(n2 - 1).to_string(), n2 - 1)?,
        relin: load_key(&arena, s, &ctx, &kplans, &g0, "mult", 1)?,
        atob: load_key(&arena, s, &ctx, &kplans, &g0, &(n2 - 2).to_string(), 1)?,
        btoa: load_key(&arena, s, &ctx, &kplans, &g0, &(n2 - 4).to_string(), 1)?,
        n,
    };
    println!("[bts-sparse] {} oracle keys in {:.1} s", steps.len() + 4, tk.elapsed().as_secs_f64());
    let stage = ex.ks.stage_ptrs();
    use pc_gpu::bts::BtsKeyRing;
    let encaps = EncapsKeys::new(&arena, s, n, ring.dense_to_sparse()?, ring.sparse_to_dense()?, stage)?;
    let pow2 = pc_gpu::bts::finish::Pow2::new(&arena, s, &ctx)?;
    let tails = SparseBPrecomp::<W>::tails(&arena, s, &ctx, &ring, std::slice::from_ref(&scfg))?;
    let pre = SparseBPrecomp::new(&arena, s, &ctx, scfg.clone(), &ring, &tails, stage)?;
    println!(
        "[bts-sparse] SparseB geometry {:?}; CtS top {} StC top {}; EvalMod {:?} -> {:?}; carved {:.1} MB vs MemPlan bts.sparse {:.1} MB ({})",
        pre.geom,
        pre.cts.top,
        pre.stc.top,
        pre.em.in_meta,
        pre.em.out_meta,
        pre.carved as f64 / 1e6,
        sb_bytes as f64 / 1e6,
        if pre.carved == sb_bytes { "equal" } else { "DIFFER" }
    );
    let mem_ok = pre.carved == sb_bytes;
    if !mem_ok {
        let pred = SparseBPrecomp::<W>::bytes_parts(&chain, &st, &lay, &scfg)?;
        for ((a, x), (_, y)) in pre.parts.iter().zip(&pred) {
            println!("[bts-sparse]   {a:<14} carved {x} predicted {y}{}", if x != y { "  <-" } else { "" });
        }
    }
    let route = SparseBRoute { slots: o.route, ring_n: n, variants: vec![pre], encaps, pow2, cf_default: 6, cf_window: (2, 20), deg: chain_deg(&ctx)?, ks_source: scfg.ks_source };
    println!("[bts-sparse] route {} key source {:?}", o.route, route.ks_source);
    let mut work = BtsWork::<W>::new(&arena, lay, wshape)?;
    ex.compute.sync()?;
    println!("[bts-sparse] setup {:.1} s; arena {} of {} B", t0.elapsed().as_secs_f64(), arena.used_bytes(), arena.capacity());
    arena.freeze();
    let sh = Shadow::new(ShadowConfig { chain: ctx.shape, bts: BtsConfig::n32_shipped(), quirks: Quirks::none() }, Some(Arc::new(ctx.scale.clone())));
    let encoder = Encoder::new(&chain);
    let mut dct = pool.acquire(1, CtMeta::new(0, 1))?;
    let mut decrypt = |ex: &mut Executor<'_, W>, ops: &mut Ops<'_, W>, x: &Ct<'_, W>, c: usize| -> Res<Vec<Complex>> {
        let d = dct.get_mut().ok_or("shared")?;
        let top = x.top(ctx.shape)?;
        let rows: Vec<usize> = (0..=top).collect();
        for part in 0..2 {
            let v = x.buf.download_rows(&ex.compute, c, part, &rows)?;
            d.buf.upload_rows(&ex.compute, 0, part, &rows, &v)?;
        }
        d.meta = x.meta;
        d.meta.slots = 0;
        ops.realize(&ctx, ex, &sh, std::slice::from_mut(d))?;
        let mut v = dec.decrypt_slots(&ctx, &ex.compute, &encoder, d, Payload::Complex, Which::Dense)?;
        Ok(v.remove(0))
    };
    let r0 = pc_shadow::Route { slots: o.route, raise_drop: rd, sparse_b: true, ..pc_shadow::Route::dense() };
    let mut ok = mem_ok;
    match o.mode {
        Mode::Check => {
            let bd = g0.bts();
            let mut x = pool.acquire(1, CtMeta::new(0, 1))?;
            let mut y = pool.acquire(1, CtMeta::new(0, 1))?;
            for k in 1..=10 {
                if !bd.exists(&format!("{k}.json")) {
                    continue;
                }
                let j = bd.json(&format!("{k}.json"));
                if j["slots"].as_u64() != Some(o.route as u64) {
                    continue;
                }
                let corr = j["state"]["correction"].as_u64().ok_or("state.correction")? as u32;
                let rt = pc_shadow::Route { cf: Some(corr + route.deg), ..r0 };
                let hin = bd.ct(&format!("{k}.in"));
                let xm = x.get_mut().ok_or("shared")?;
                upload(xm, &ex.compute, &hin, nq)?;
                xm.meta.slots = o.route;
                let ym = y.get_mut().ok_or("shared")?;
                let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                let mut st = Staged::<W, M>::begin(&route, &cx, std::slice::from_ref(xm), &rt, false)?;
                let stages: Vec<&str> = j["stages"].as_array().map(|a| a.iter().filter_map(|v| v.as_str()).collect()).unwrap_or_default();
                // the dumped stage (if any) after each of ours: bit-exact words and decrypted slots
                let mut cmp = |name: &str, xm: &Ct<'_, W>, ex: &mut Executor<'_, W>, ops: &mut Ops<'_, W>, conj_add: bool| -> Res<()> {
                    if !stages.contains(&name) {
                        return Ok(());
                    }
                    let h = bd.ct(&format!("{k}.{name}"));
                    let mut bad = 0;
                    let mut tot = 0;
                    let same_top = h.level == xm.top(ctx.shape)?;
                    if same_top && !conj_add {
                        for part in 0..2 {
                            let got = xm.buf.download_rows(&ex.compute, 0, part, &h.ids)?;
                            bad += got.iter().zip(&h.c[part]).filter(|(a, b)| a != b).count();
                            tot += got.len();
                        }
                    }
                    upload(ym, &ex.compute, &h, nq)?;
                    let theirs = decrypt(ex, ops, ym, 0)?;
                    let mut ours = decrypt(ex, ops, xm, 0)?;
                    if conj_add {
                        ours = ours.iter().map(|z| Complex::new(2.0 * z.re, 0.0)).collect();
                    }
                    let m = cmax(&theirs).max(1e-300);
                    let e = cdiff(&ours, &theirs);
                    println!(
                        "[sparse] bts {k} {name:<10}: ours top {} deg {} | dump top {} deg {}; words {}; max|ours - dump| {e:.3e} of |dump| {m:.3e} (rel {:.1e})",
                        xm.top(ctx.shape)?,
                        xm.meta.deg,
                        h.level,
                        h.noise_level,
                        if same_top && !conj_add { format!("{bad} mismatches of {tot}") } else { "n/a".into() },
                        e / m
                    );
                    Ok(())
                };
                let hraise = Staged::<W, M>::mod_raise(&route, &mut cx, &mut st, std::slice::from_mut(xm));
                hraise?;
                cmp("mod_raise", xm, cx.ex, cx.ops, false)?;
                Staged::<W, M>::fold(&route, &mut cx, &st, std::slice::from_mut(xm))?;
                cmp("fold", xm, cx.ex, cx.ops, false)?;
                Staged::<W, M>::cts_stage(&route, &mut cx, &st, std::slice::from_mut(xm), 0)?;
                cmp("cts_0", xm, cx.ex, cx.ops, true)?;
                Staged::<W, M>::eval_mod(&route, &mut cx, &st, std::slice::from_mut(xm))?;
                cmp("eval_mod", xm, cx.ex, cx.ops, false)?;
                Staged::<W, M>::stc_stage(&route, &mut cx, &st, std::slice::from_mut(xm), 0)?;
                cmp("stc_0", xm, cx.ex, cx.ops, false)?;
                Staged::<W, M>::finish(&route, &mut cx, &st, std::slice::from_mut(xm))?;
                let lm = BtsRoute::<W, M>::landing(&route, &sh, &meta_of(&hin, nq), &route_r(&route, &st))?;
                // output: words where the stages agree, decrypted vs the dumped output and vs the input's period-s fold
                let hout = bd.ct(&format!("{k}.out"));
                let vin = {
                    upload(ym, &cx.ex.compute, &hin, nq)?;
                    decrypt(cx.ex, cx.ops, ym, 0)?
                };
                let ours = decrypt(cx.ex, cx.ops, xm, 0)?;
                upload(ym, &cx.ex.compute, &hout, nq)?;
                let old = decrypt(cx.ex, cx.ops, ym, 0)?;
                let sl = o.route as usize;
                let copies = (n / 2 / sl) as f64;
                let want: Vec<Complex> = (0..n / 2)
                    .map(|i| {
                        let (mut re, mut im) = (0.0, 0.0);
                        for c in 0..n / 2 / sl {
                            let z = vin[(i % sl) + c * sl];
                            re += z.re;
                            im += z.im;
                        }
                        Complex::new(re / copies, im / copies)
                    })
                    .collect();
                let m = cmax(&want).max(1e-300);
                let (eo, el, ev) = (cdiff(&ours, &want), cdiff(&old, &want), cdiff(&ours, &old));
                let lv_ok = xm.meta.level == meta_of(&hout, nq).level && xm.meta.deg as i64 == hout.noise_level && lm.level == xm.meta.level;
                println!(
                    "[sparse] bts {k} ({}) cf {}: out top {} deg {} (dump top {} deg {}, landing() {}) {}; |want| {m:.3e}; max|ours-want| {eo:.3e} ({:.1} bits), max|old-want| {el:.3e} ({:.1} bits), max|ours-old| {ev:.3e}",
                    j["step"].as_str().unwrap_or(""),
                    st.cf,
                    xm.top(ctx.shape)?,
                    xm.meta.deg,
                    hout.level,
                    hout.noise_level,
                    nq - 1 - lm.level as usize,
                    if lv_ok { "level OK" } else { "LEVEL MISMATCH" },
                    -(eo / m).log2(),
                    -(el / m).log2(),
                );
                ok &= lv_ok && eo <= 2.0 * el.max(1e-12 * m);
            }
            println!("[sparse] route {}: {}", o.route, if ok { "PASS" } else { "FAIL" });
        }
        Mode::Bench => {
            // input: the first dumped bootstrap input of the route (any s-periodic ciphertext times the same)
            let bd = g0.bts();
            let kin = (1..=10).find(|k| bd.exists(&format!("{k}.json")) && bd.json(&format!("{k}.json"))["slots"].as_u64() == Some(o.route as u64));
            let hin = match kin {
                Some(k) => bd.ct(&format!("{k}.in")),
                None => bd.ct("3.in"),
            };
            for &k in &o.ks {
                let mut src = pool.acquire(k, CtMeta::new(0, 1))?;
                let mut x = pool.acquire(k, CtMeta::new(0, 1))?;
                let sv = src.get_mut().ok_or("shared")?;
                for c in 0..k {
                    for part in 0..2 {
                        sv.buf.upload_rows(&ex.compute, c, part, &hin.ids, &hin.c[part])?;
                    }
                }
                let m0 = CtMeta { slots: o.route, ..meta_of(&hin, nq) };
                let words = sv.buf.buf.len();
                let xm = x.get_mut().ok_or("shared")?;
                let route_c = pc_shadow::Route { cf: Some(6), ..r0 };
                let (ms, host) = time_it(&dev, &mut ex, o.reps, |ex| {
                    xm.buf.buf.copy_from(&ex.compute, &sv.buf.buf, words)?;
                    xm.meta = m0;
                    let mut cx = BtsCx { ctx: &ctx, ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                    BtsRoute::<W, M>::run(&route, &mut cx, std::slice::from_mut(xm), &route_c)?;
                    Ok(())
                })?;
                println!("[bench] bts route {} (SparseB, raise drop {rd}) K={k}: {ms:.3} ms/call, {:.3} ms/ct (host issue {host:.3} ms/call)", o.route, ms / k as f64);
                // stage profile (median of 7)
                let names = ["mod_raise", "fold", "cts", "eval_mod", "stc", "finish"];
                let evs: Vec<Event> = (0..=names.len()).map(|_| Event::timing(&dev)).collect::<std::result::Result<_, _>>()?;
                let mut acc = vec![vec![]; names.len()];
                for _ in 0..7 {
                    xm.buf.buf.copy_from(&ex.compute, &sv.buf.buf, words)?;
                    xm.meta = m0;
                    let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                    let xs = std::slice::from_mut(xm);
                    let mut st = Staged::<W, M>::begin(&route, &cx, xs, &route_c, false)?;
                    evs[0].record(&cx.ex.compute)?;
                    Staged::<W, M>::mod_raise(&route, &mut cx, &mut st, xs)?;
                    evs[1].record(&cx.ex.compute)?;
                    Staged::<W, M>::fold(&route, &mut cx, &st, xs)?;
                    evs[2].record(&cx.ex.compute)?;
                    Staged::<W, M>::cts_stage(&route, &mut cx, &st, xs, 0)?;
                    evs[3].record(&cx.ex.compute)?;
                    Staged::<W, M>::eval_mod(&route, &mut cx, &st, xs)?;
                    evs[4].record(&cx.ex.compute)?;
                    Staged::<W, M>::stc_stage(&route, &mut cx, &st, xs, 0)?;
                    evs[5].record(&cx.ex.compute)?;
                    Staged::<W, M>::finish(&route, &mut cx, &st, xs)?;
                    evs[6].record(&cx.ex.compute)?;
                    evs[6].sync()?;
                    for (i, a) in acc.iter_mut().enumerate() {
                        a.push(evs[i].elapsed_ms(&evs[i + 1])? as f64);
                    }
                }
                let mut line = format!("[bench] route {} K={k} stages (ms/call):", o.route);
                for (i, a) in acc.iter_mut().enumerate() {
                    a.sort_by(|x, y| x.total_cmp(y));
                    line += &format!(" {} {:.3}", names[i], a[3]);
                }
                println!("{line}");
            }
        }
        Mode::Fold => {
            use pc_gpu::bts::fold::{fold_bootstrap, fold_reference};
            use pc_gpu::bts::sparse::RouteSet;
            let mut set = RouteSet::<W, M>::new(&[o.route]);
            set.add_route(o.route, &route)?;
            let undeclared = set.add_route(o.route / 2, &route).is_err();
            println!("[fold] route set: declared {:?}, built {:?}; add_route({}) undeclared refused: {undeclared}", set.declared, set.built(), o.route / 2);
            ok &= undeclared;
            let n2 = n / 2;
            let sl = o.route as usize;
            let mut x = pool.acquire(1, CtMeta::new(0, 1))?;
            for &(n_live, prescale, a) in &[((n2 / sl) as u32, 1.0, 0.5), (768u32, 0.5, 0.5)] {
                let v = rand_slots(n2, a, 0xf01d + n_live as u64);
                let top = nq - 1 - 20;
                let sf = ctx.scale.sf(top as i32)?;
                let rows = encoder.encode_slots(&v, n2, top, Payload::Complex, sf)?;
                let m0 = CtMeta { level: 20, deg: 1, nf: sf, slots: n2 as u32, tag: Default::default() };
                let xm = x.get_mut().ok_or("shared")?;
                enc.encrypt_sk(&ctx, &ex.compute, &secret, &[rows], m0, xm, &mut prng)?;
                let want_meta = sh.fold_bootstrap(&xm.meta, o.route, n_live, prescale)?.out;
                let rt = pc_shadow::Route { cf: Some(8), ..r0 };
                fold_bootstrap(&mut BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work }, &set, xm, o.route, n_live, prescale, &rt)?;
                let out = decrypt(&mut ex, &mut ops, xm, 0)?;
                let want = fold_reference(&v, sl, n_live);
                ok &= fold_verdict(&format!("s={sl} (SparseB) n_live {n_live} prescale {prescale}"), &out, &want, 10.0);
                println!("[fold]   output level {} deg {} (Shadow fold_bootstrap recipe: level {} deg {})", xm.meta.level, xm.meta.deg, want_meta.level, want_meta.deg);
            }
            let bad = fold_bootstrap(&mut BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work }, &set, x.get_mut().ok_or("shared")?, 4, 1, 1.0, &r0).is_err();
            println!("[fold] fold_bootstrap at an unbuilt s=4 refused: {bad}");
            ok &= bad;
        }
        Mode::Prec => {
            let mut rows_out = vec![];
            println!("| cf | amp | rel_err | bits (Re) | bits (complex) |");
            println!("|---|---|---|---|---|");
            let sl = o.route as usize;
            let mut x = pool.acquire(1, CtMeta::new(0, 1))?;
            for &cf in &o.cfs {
                for &a in &o.amps {
                    // s-periodic real data, uniform in [-A, A] on the first s slots, anchored at max |v| = A
                    let mut g = rnd(7777);
                    let mut base: Vec<f64> = (0..sl).map(|_| (g() as f64 / u64::MAX as f64) * 2.0 * a - a).collect();
                    let mx = base.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
                    base.iter_mut().for_each(|v| *v *= a / mx);
                    let vals: Vec<Complex> = (0..n / 2).map(|i| Complex::new(base[i % sl], 0.0)).collect();
                    let top = nq - 1;
                    let sf = ctx.scale.sf(top as i32)?;
                    let rows = encoder.encode_slots(&vals, n / 2, top, Payload::Real, sf)?;
                    let m0 = CtMeta { level: 0, deg: 1, nf: sf, slots: o.route, tag: Default::default() };
                    let xm = x.get_mut().ok_or("shared")?;
                    enc.encrypt_sk(&ctx, &ex.compute, &secret, &[rows], m0, xm, &mut prng)?;
                    let route_c = pc_shadow::Route { cf: Some(cf), ..r0 };
                    {
                        let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                        BtsRoute::<W, M>::run(&route, &mut cx, std::slice::from_mut(xm), &route_c)?;
                    }
                    ops.realize(&ctx, &mut ex, &sh, std::slice::from_mut(xm))?;
                    let out = decrypt(&mut ex, &mut ops, xm, 0)?;
                    let e = out.iter().zip(&vals).map(|(z, v)| (z.re - v.re).abs()).fold(0.0, f64::max) / a;
                    let ec = out.iter().zip(&vals).map(|(z, v)| Complex::new(z.re - v.re, z.im).abs()).fold(0.0, f64::max) / a;
                    println!("| {cf} | {a} | {e:.3e} | {:.2} | {:.2} |", -e.log2(), -ec.log2());
                    rows_out.push(format!(
                        "{{\"amp\": {a}, \"bits\": {}, \"bits_complex\": {}, \"cf\": {cf}, \"input_level\": 0, \"iterations\": 1, \"landing_deg\": {}, \"landing_level\": {}, \"period\": {sl}, \"raise_drop\": {rd}, \"real\": false, \"rel_err\": {e:e}, \"rel_err_complex\": {ec:e}, \"route\": \"sparse{sl}\"}}",
                        -e.log2(), -ec.log2(), xm.meta.deg, xm.meta.level
                    ));
                }
            }
            if let Some(p) = &o.write {
                std::fs::write(p, format!("[\n  {}\n]\n", rows_out.join(",\n  ")))?;
                println!("[prec] {} rows -> {}", rows_out.len(), p.display());
            }
        }
    }
    Ok(ok)
}

fn route_r(r: &pc_gpu::bts::sparse::SparseBRoute<'_, W>, st: &pc_gpu::bts::BtsState) -> pc_shadow::Route {
    pc_shadow::Route { slots: r.slots, raise_drop: st.raise_drop, sparse_b: true, cf: Some(st.cf), ..pc_shadow::Route::dense() }
}

fn rnd(seed: u64) -> impl FnMut() -> u64 {
    let mut x = seed | 1;
    move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    }
}
