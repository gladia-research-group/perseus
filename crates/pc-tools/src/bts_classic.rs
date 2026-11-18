//! The classic multi-stage sparse route on the device (`pc_gpu::bts::sparse_classic`).
//!
//!   pc-check sparse-classic [--route 8192] [--old \<dir>]   MemPlan == carve, landing == the compile's table; with
//!       `--old <dir>` (a reference dump: secret `sk.npy`, `cases.json`, input / output limbs per case) the route runs
//!       on the same input ciphertexts and both outputs are decrypted with that secret (gate: bits >= reference - 0.5)
//!   pc-bench bts --classic --route 8192 [--reps 10]        device ms per refresh and the stage split
//!   pc-bench bts-prec --classic --route 8192|512 [--cf ..] [--amps ..] [--write rows.json]
//!       s-periodic uniform real data anchored at A, sk-encrypted at the top, one refresh, realized
//!
//! Keys: generated on the device (packed seeded), from the dump's secret with `--old`, else fresh.
use pc_core::encode::{Complex, Encoder, Payload};
use pc_core::keys::{sample_sparse_ternary, SecretKey};
use pc_core::modmul::Barrett64;
use pc_core::params::{Chain, ChainSpec};
use pc_cuda::{Device, Event};
use pc_gpu::bts::sparse_classic::{ClassicCfg, ClassicPlan};
use pc_gpu::bts::{BtsCx, BtsRoute, BtsWork, Staged};
use pc_gpu::decrypt::{Decryptor, Which};
use pc_gpu::encrypt::Encryptor;
use pc_gpu::keygen::{test_prng, KeyGen};
use pc_gpu::keys::{DevSecret, KeySet, KeySetSpec, KskLayout, KskShape, RotSpec};
use pc_gpu::ks::{KsPlans, KsWork};
use pc_gpu::limbs::{import_limbs, Limbs};
use pc_gpu::ops::{Ops, OpsCfg};
use pc_gpu::{Context, Ct, CtPool, Executor, MemCfg, MemPlan, PoolClass};
use crate::dump_g0::npy_header;
use pc_shadow::{BtsConfig, CtMeta, Quirks, Route, Shadow, ShadowConfig};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

type W = u32;
type M = Barrett64;
type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[allow(dead_code)] // each binary dispatches a subset
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Check,
    Bench,
    Prec,
}

#[derive(Clone, Debug)]
pub struct Opts {
    pub mode: Mode,
    pub route: u32,
    pub old: Option<PathBuf>,
    pub cfs: Vec<u32>,
    pub amps: Vec<f64>,
    pub reps: usize,
    pub write: Option<PathBuf>,
    pub budget: [u32; 2],
    pub raise_drop: u32,
}

pub fn parse(mode: Mode, args: &[String]) -> Res<Opts> {
    let mut o = Opts { mode, route: 8192, old: None, cfs: vec![6, 8, 10, 12], amps: vec![0.01, 0.1, 1.0, 3.0], reps: 10, write: None, budget: [3, 3], raise_drop: 0 };
    let list = |s: &str| -> Res<Vec<f64>> { Ok(s.split(',').map(|x| x.parse::<f64>()).collect::<std::result::Result<_, _>>()?) };
    let mut i = 0;
    while i < args.len() {
        let v = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--classic" => {
                i += 1;
                continue;
            }
            "--route" => o.route = v.parse()?,
            "--old" => o.old = Some(PathBuf::from(&v)),
            "--cf" => o.cfs = list(&v)?.into_iter().map(|x| x as u32).collect(),
            "--amps" => o.amps = list(&v)?,
            "--reps" => o.reps = v.parse()?,
            "--write" => o.write = Some(PathBuf::from(&v)),
            "--raise-drop" => o.raise_drop = v.parse()?,
            "--budget" => {
                let b = list(&v)?;
                if b.len() != 2 {
                    return Err("--budget cts,stc".into());
                }
                o.budget = [b[0] as u32, b[1] as u32];
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
            eprintln!("[sparse-classic] error: {e}");
            ExitCode::from(2)
        }
    }
}

fn read_npy_i8(p: &Path) -> Res<Vec<i8>> {
    let (d, _, off) = npy_header(p)?;
    let b = std::fs::read(p)?;
    match d.trim_start_matches(['<', '|', '=']) {
        "i1" => Ok(b[off..].iter().map(|&x| x as i8).collect()),
        o => Err(format!("{}: dtype {o}", p.display()).into()),
    }
}

/// (rows, cols) u64 array as rows of `W` words.
fn read_npy_rows(p: &Path) -> Res<Vec<Vec<W>>> {
    let (d, shape, off) = npy_header(p)?;
    if d.trim_start_matches(['<', '|', '=']) != "u8" || shape.len() != 2 {
        return Err(format!("{}: want a 2-D uint64 array, got {d} {shape:?}", p.display()).into());
    }
    let b = std::fs::read(p)?;
    let v: Vec<W> = b[off..].chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap_or([0; 8])) as W).collect();
    Ok(v.chunks(shape[1]).map(|r| r.to_vec()).collect())
}

fn bits(err: f64, a: f64) -> f64 {
    -(err / a).log2()
}

pub fn run(o: &Opts) -> Res<bool> {
    let t0 = Instant::now();
    let chain = Chain::generate(&ChainSpec::n32())?;
    let n = chain.n;
    let nq = chain.q.len();
    let mut cfg = ClassicCfg::n32(o.route, &[o.raise_drop]);
    cfg.level_budget = o.budget;
    let plan = ClassicPlan::new(&chain, &cfg)?;
    let steps = plan.rot_steps();
    let layout = KskLayout::packed_for::<W>(&chain)?;
    let spec = KeySetSpec { relin: true, conj: true, rots: steps.iter().map(|&s| RotSpec { step: s, q_top: None }).collect(), encaps: true, lean: None };

    let dev = Device::open()?;
    let mcfg = MemCfg { k_max: 8, pool: vec![PoolClass { k: 1, count: 6 }], events: 16, scratch_cts: 2 };
    let mut mp = MemPlan::build::<W, M>(&chain, mcfg)?;
    let lay = mp.lay;
    let mut shapes: Vec<KskShape> = spec.keys(&chain)?.into_iter().map(|(_, s)| s).collect();
    shapes.push(KskShape::bottom(&chain)?);
    mp.reserve("keys", spec.bytes::<W>(&chain, layout)?);
    mp.reserve("keygen", KeyGen::<W>::bytes(&lay, &shapes));
    mp.reserve("secret", DevSecret::<W>::bytes(&lay));
    mp.reserve("encrypt", Encryptor::<W>::bytes(&lay, false));
    mp.reserve("decrypt", Decryptor::<W>::bytes(&lay, chain.level_unit, 1));
    mp.reserve("ks.plans", KsPlans::bytes::<W>(&chain, mp.cfg.k_max)?);
    mp.reserve("ks.regen_a", KsWork::<W>::regen_bytes(&lay, chain.dnum()));
    let ocfg = OpsCfg::default();
    Ops::<W>::reserve(&mut mp, &ocfg);
    let (req, wshape) = plan.mem::<W>()?;
    req.reserve(&mut mp);
    mp.reserve("bts.work", BtsWork::<W>::bytes(&lay, wshape));
    println!("[sparse-classic] route {} budget {:?} raise drop {}: {} rotation keys; route tables {:.2} GB; MemPlan {}", o.route, o.budget, o.raise_drop, steps.len(), req.total() as f64 / 1e9, mp.report());
    let arena = mp.allocate(&dev)?;
    let mut ex = Executor::<W>::new(&dev, &arena, &mp)?;
    ex.ks.carve_regen(&arena, &lay)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &mp)?;
    let pool = CtPool::<W>::new(&arena, &mp)?;
    let kplans = KsPlans::new(&arena, &ex.compute, &ctx, mp.cfg.k_max)?;
    let mut ops = Ops::<W>::new(&arena, &dev, &ex.compute, &ctx, ocfg)?;
    let mut prng = test_prng(0x6e);
    let sk = match &o.old {
        Some(d) => SecretKey { s: read_npy_i8(&d.join("sk.npy"))?, s_sparse: sample_sparse_ternary(&mut prng, n, SecretKey::SPARSE_H)? },
        None => SecretKey::sample(&mut prng, n)?,
    };
    if sk.s.len() != n || sk.s.iter().any(|&x| !(-1..=1).contains(&x)) {
        return Err("secret: not a ternary vector of length N".into());
    }
    let mut sec = DevSecret::<W>::carve(&arena, lay)?;
    sec.load(&ctx, &ex.compute, &sk)?;
    let mut keys = KeySet::<W>::carve(&arena, &chain, &spec, layout)?;
    let mut kg = KeyGen::<W>::new(&arena, &ex.compute, &chain, &shapes)?;
    let tk = kg.generate_set(&ctx, &mut ex, &sec, &mut keys, &mut prng)?;
    println!("[sparse-classic] keygen {} keys in {:.1} s", tk.keys, tk.total);
    let mut enc = Encryptor::<W>::new(&arena, &ex.compute, lay, false, &mut prng)?;
    let mut dec = Decryptor::<W>::new(&arena, &ex.compute, lay, chain.level_unit, 1)?;
    dec.load_secret(&ex.compute, &sec)?;
    let tb = Instant::now();
    let route = plan.build(&arena, &ex.compute, &ctx, &keys, ex.ks.stage_ptrs())?;
    let mem_ok = route.check_mem().is_ok();
    println!("[sparse-classic] route built in {:.1} s: carved {} B, MemPlan {} B ({})", tb.elapsed().as_secs_f64(), route.carved, req.total(), if mem_ok { "equal" } else { "DIFFER" });
    let mut work = BtsWork::<W>::new(&arena, lay, wshape)?;
    ex.compute.sync()?;
    arena.freeze();
    println!("[sparse-classic] setup {:.1} s; arena {} of {} B", t0.elapsed().as_secs_f64(), arena.used_bytes(), arena.capacity());
    // landing vs the compile's runtime table (realized; raw = one CKKS level above, degree 2)
    let mut bc = BtsConfig::n32_shipped();
    bc.landings.insert(o.route, pc_shadow::route::multi_stage_landing(o.budget, chain.level_unit as u32));
    let sh = Shadow::new(ShadowConfig { chain: ctx.shape, bts: bc, quirks: Quirks::none() }, Some(Arc::new(ctx.scale.clone())));
    let r0 = Route { slots: o.route, raise_drop: o.raise_drop, ..Route::sparse(o.route) };
    let raw = BtsRoute::<W, M>::landing(&route, &sh, &CtMeta::new(10, 1), &r0)?;
    let real = sh.rescale(&raw)?.out;
    let want = sh.bootstrap(&CtMeta::new(10, 1), &r0)?.out;
    let land_ok = (real.level, real.deg) == (want.level, want.deg) && raw.deg == 2;
    println!("[sparse-classic] landing raw ({}, {}) realized ({}, {}) vs the compile table ({}, {}): {}", raw.level, raw.deg, real.level, real.deg, want.level, want.deg, if land_ok { "equal" } else { "DIFFER" });
    let mut ok = mem_ok && land_ok;
    let encoder = Encoder::new(&chain);
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
    let mut xh = pool.acquire(1, CtMeta::new(0, 1))?;
    let mut yh = pool.acquire(1, CtMeta::new(0, 1))?;
    let sl = o.route as usize;
    match o.mode {
        Mode::Check => {
            let Some(dir) = &o.old else {
                println!("[sparse-classic] no --old dump: MemPlan + landing only");
                return Ok(ok);
            };
            let cases: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("cases.json"))?)?;
            println!("| case | CF | A | in (level, deg) | old bits (Re / cplx) | new bits (Re / cplx) | new - old (Re) | max abs new - old / A | out level old / new | old ms | new ms |");
            println!("|---|---|---|---|---|---|---|---|---|---|---|");
            for c in cases.as_array().ok_or("cases.json")? {
                let i = c["case"].as_u64().ok_or("case")?;
                let cf = c["cf"].as_u64().ok_or("cf")? as u32;
                let a = c["amp"].as_f64().ok_or("amp")?;
                let load = |tag: &str| -> Res<Limbs<W>> {
                    let m = &c[tag];
                    Ok(Limbs {
                        c: [read_npy_rows(&dir.join(format!("c{i}.{tag}.c0.npy")))?, read_npy_rows(&dir.join(format!("c{i}.{tag}.c1.npy")))?],
                        moduli: chain.q[..=m["level"].as_u64().ok_or("level")? as usize].to_vec(),
                        level: m["level"].as_u64().ok_or("level")? as usize,
                        noise_level: m["noise_level"].as_u64().ok_or("noise_level")? as u8,
                        noise_factor: m["noise_factor"].as_f64().ok_or("noise_factor")?,
                    })
                };
                let lin = load("in")?;
                let lout = load("out")?;
                let x = xh.get_mut().ok_or("shared")?;
                x.meta = CtMeta { level: (nq - 1 - lin.level) as u32, deg: lin.noise_level, nf: lin.noise_factor, slots: 0, tag: Default::default() };
                import_limbs(&ctx, &mut ex, x, 0, &lin)?;
                let y = yh.get_mut().ok_or("shared")?;
                y.meta = CtMeta { level: (nq - 1 - lout.level) as u32, deg: lout.noise_level, nf: lout.noise_factor, slots: 0, tag: Default::default() };
                import_limbs(&ctx, &mut ex, y, 0, &lout)?;
                let vin = decrypt(&mut ex, &mut ops, x)?;
                let vold = decrypt(&mut ex, &mut ops, y)?;
                let mx = vin.iter().map(|z| z.re.abs()).fold(0.0, f64::max);
                let period = (0..sl).flat_map(|j| (1..n / 2 / sl).map(move |r| (j, r))).map(|(j, r)| (vin[j].re - vin[j + r * sl].re).abs()).fold(0.0, f64::max) / mx;
                let x_meta = x.meta;
                let rc = Route { cf: Some(cf), ..r0 };
                let mut ms = vec![];
                let mut out_meta = x_meta;
                let mut vnew = vec![];
                for rep in 0..(1 + o.reps.min(10)) {
                    let y = yh.get_mut().ok_or("shared")?;
                    // y := the imported input (the reference output was decrypted above)
                    y.meta = x_meta;
                    import_limbs(&ctx, &mut ex, y, 0, &lin)?;
                    let (ea, eb) = (Event::timing(&dev)?, Event::timing(&dev)?);
                    ea.record(&ex.compute)?;
                    {
                        let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                        BtsRoute::<W, M>::run(&route, &mut cx, std::slice::from_mut(y), &rc)?;
                    }
                    eb.record(&ex.compute)?;
                    eb.sync()?;
                    if rep > 0 {
                        ms.push(ea.elapsed_ms(&eb)? as f64);
                    } else {
                        out_meta = y.meta;
                        vnew = decrypt(&mut ex, &mut ops, y)?;
                    }
                }
                ms.sort_by(|p, q| p.total_cmp(q));
                let e_old = vold.iter().zip(&vin).map(|(z, v)| (z.re - v.re).abs()).fold(0.0, f64::max);
                let e_new = vnew.iter().zip(&vin).map(|(z, v)| (z.re - v.re).abs()).fold(0.0, f64::max);
                let c_old = vold.iter().zip(&vin).map(|(z, v)| Complex::new(z.re - v.re, z.im - v.im).abs()).fold(0.0, f64::max);
                let c_new = vnew.iter().zip(&vin).map(|(z, v)| Complex::new(z.re - v.re, z.im - v.im).abs()).fold(0.0, f64::max);
                let d_on = vnew.iter().zip(&vold).map(|(z, v)| Complex::new(z.re - v.re, z.im - v.im).abs()).fold(0.0, f64::max);
                let (b_old, b_new) = (bits(e_old, mx), bits(e_new, mx));
                let pass = b_new >= b_old - 0.5;
                ok &= pass;
                println!(
                    "| {i} | {cf} | {a} | ({}, {}) | {:.2} / {:.2} | {:.2} / {:.2} | {:+.2} {} | {:.2e} | {} / {} | {:.2} | {:.2} |",
                    x_meta.level,
                    x_meta.deg,
                    b_old,
                    bits(c_old, mx),
                    b_new,
                    bits(c_new, mx),
                    b_new - b_old,
                    if pass { "ok" } else { "LOW" },
                    d_on / mx,
                    nq - 1 - lout.level,
                    out_meta.level,
                    c["ms_median"].as_f64().unwrap_or(f64::NAN),
                    ms.get(ms.len() / 2).copied().unwrap_or(f64::NAN)
                );
                println!("[sparse-classic]   case {i}: input max |Re| {mx:.3e}, {sl}-periodicity residue {period:.1e}");
            }
            println!("[sparse-classic] check: {}", if ok { "PASS" } else { "FAIL" });
        }
        Mode::Bench => {
            let top = nq - 1;
            let sf = ctx.scale.sf(top as i32)?;
            let vals: Vec<Complex> = (0..n / 2).map(|i| Complex::new(((i % sl) as f64 / sl as f64) - 0.5, 0.0)).collect();
            let rows = encoder.encode_slots(&vals, n / 2, top, Payload::Real, sf)?;
            let m0 = CtMeta { level: 0, deg: 1, nf: sf, slots: o.route, tag: Default::default() };
            let y = yh.get_mut().ok_or("shared")?;
            let rc = Route { cf: Some(6), ..r0 };
            // whole refresh, then the stage split (an event between stages)
            let mut tot = vec![];
            let mut split: Vec<Vec<(String, f64)>> = vec![];
            for rep in 0..(2 + o.reps) {
                enc.encrypt_sk(&ctx, &ex.compute, &sec, &[rows.clone()], m0, y, &mut prng)?;
                let mut evs = vec![(String::from("start"), Event::timing(&dev)?)];
                evs[0].1.record(&ex.compute)?;
                {
                    let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                    let r: &dyn Staged<'_, W, M> = &route;
                    let ys = std::slice::from_mut(y);
                    let mut st = r.begin(&cx, ys, &rc, false)?;
                    let mut mark = |name: &str, cx: &mut BtsCx<'_, '_, W, M>| -> Res<()> {
                        let e = Event::timing(&dev)?;
                        e.record(&cx.ex.compute)?;
                        evs.push((name.to_string(), e));
                        Ok(())
                    };
                    r.mod_raise(&mut cx, &mut st, ys)?;
                    mark("raise", &mut cx)?;
                    r.fold(&mut cx, &st, ys)?;
                    mark("fold", &mut cx)?;
                    for k in 0..st.n_cts {
                        r.cts_stage(&mut cx, &st, ys, k)?;
                        mark(&format!("cts{k}"), &mut cx)?;
                    }
                    r.eval_mod(&mut cx, &st, ys)?;
                    mark("evalmod", &mut cx)?;
                    r.stc_enter(&mut cx, &st, ys)?;
                    for k in 0..st.n_stc {
                        r.stc_stage(&mut cx, &st, ys, k)?;
                        mark(&format!("stc{k}"), &mut cx)?;
                    }
                    r.finish(&mut cx, &st, ys)?;
                    mark("finish", &mut cx)?;
                }
                evs.last().ok_or("events")?.1.sync()?;
                if rep >= 2 {
                    tot.push(evs[0].1.elapsed_ms(&evs.last().ok_or("events")?.1)? as f64);
                    split.push(evs.windows(2).map(|w| (w[1].0.clone(), w[0].1.elapsed_ms(&w[1].1).map(|x| x as f64).unwrap_or(f64::NAN))).collect());
                }
            }
            tot.sort_by(|p, q| p.total_cmp(q));
            let med = |v: &mut Vec<f64>| {
                v.sort_by(|p, q| p.total_cmp(q));
                v[v.len() / 2]
            };
            println!("[bench] classic route {} K=1: {:.3} ms (median of {}; min {:.3})", o.route, tot[tot.len() / 2], tot.len(), tot[0]);
            let names: Vec<String> = split[0].iter().map(|x| x.0.clone()).collect();
            let mut line = String::from("[bench] stages (ms):");
            for (j, nm) in names.iter().enumerate() {
                let mut v: Vec<f64> = split.iter().map(|s| s[j].1).collect();
                line += &format!(" {nm} {:.2}", med(&mut v));
            }
            println!("{line}");
        }
        Mode::Prec => {
            let mut rows_out = vec![];
            println!("| cf | amp | rel_err | bits (Re) | bits (complex) | landing |");
            println!("|---|---|---|---|---|---|");
            let x = xh.get_mut().ok_or("shared")?;
            for &cf in &o.cfs {
                for &a in &o.amps {
                    let mut g = rnd(7777);
                    let mut base: Vec<f64> = (0..sl).map(|_| (g() as f64 / u64::MAX as f64) * 2.0 * a - a).collect();
                    let mx = base.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
                    base.iter_mut().for_each(|v| *v *= a / mx);
                    let vals: Vec<Complex> = (0..n / 2).map(|i| Complex::new(base[i % sl], 0.0)).collect();
                    let top = nq - 1;
                    let sf = ctx.scale.sf(top as i32)?;
                    let rows = encoder.encode_slots(&vals, n / 2, top, Payload::Real, sf)?;
                    enc.encrypt_sk(&ctx, &ex.compute, &sec, &[rows], CtMeta { level: 0, deg: 1, nf: sf, slots: o.route, tag: Default::default() }, x, &mut prng)?;
                    let rc = Route { cf: Some(cf), ..r0 };
                    {
                        let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                        BtsRoute::<W, M>::run(&route, &mut cx, std::slice::from_mut(x), &rc)?;
                    }
                    ops.realize(&ctx, &mut ex, &sh, std::slice::from_mut(x))?;
                    let out = decrypt(&mut ex, &mut ops, x)?;
                    let e = out.iter().zip(&vals).map(|(z, v)| (z.re - v.re).abs()).fold(0.0, f64::max) / a;
                    let ec = out.iter().zip(&vals).map(|(z, v)| Complex::new(z.re - v.re, z.im).abs()).fold(0.0, f64::max) / a;
                    println!("| {cf} | {a} | {e:.3e} | {:.2} | {:.2} | ({}, {}) |", -e.log2(), -ec.log2(), x.meta.level, x.meta.deg);
                    rows_out.push(format!(
                        "{{\"amp\": {a}, \"bits\": {}, \"bits_complex\": {}, \"cf\": {cf}, \"input_level\": 0, \"iterations\": 1, \"landing_deg\": {}, \"landing_level\": {}, \"period\": {sl}, \"raise_drop\": {}, \"real\": false, \"rel_err\": {e:e}, \"rel_err_complex\": {ec:e}, \"route\": \"classic{sl}\"}}",
                        -e.log2(),
                        -ec.log2(),
                        x.meta.deg,
                        x.meta.level,
                        o.raise_drop
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

fn rnd(seed: u64) -> impl FnMut() -> u64 {
    let mut x = seed | 1;
    move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    }
}
