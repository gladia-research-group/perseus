//! StC-first route, stc_high and the staged call object: oracle checks and benches.
//!
//!   pc-check stcf \<d3> [--k 1,2,4,8] [--graph]   d3 PCMM finish through the staged API (`BtsCall` resumed after the
//!       raise), every stage vs the dump, per K, eager and as one captured CUDA graph; call state vs the dumped BtsState
//!   pc-check stcf \<g0>        g0 bootstrap #2 (dense StC-first, CF 13): dumped split -> output bit-exact per stage;
//!       our stc_first_input and whole refresh vs the dump; landing / entry vs pc-shadow
//!   pc-bench stcf | stcf-prec | stc-high       device ms per call and stage | round-trip precision | stc_high vs host StC
//!   pc-bench stcf-pair [--members cf:amp,...] [--seeds 4] [--dirty 4]   two real refreshes as one complex refresh:
//!       each member vs its K = 1 real refresh, landings, graph == eager, K = 2 pair == two K = 1 pairs, timing
use crate::check_bts::{load_key, ps_plan, OracleKeys, Tally};
use pc_core::bts_setup::{cts_stc_factors, FidesStage, C64};
use pc_core::encode::{Complex, Encoder, Payload};
use pc_core::keys::SecretKey;
use pc_core::modmul::{invmod, Barrett64};
use pc_core::params::{Chain, ChainSpec};
use pc_cuda::{Compute, Device, Event, Stream};
use pc_gpu::bts::cts::{LtEnd, LtGeom, LtPts, LtStage, PtSource};
use pc_gpu::bts::evalmod::{em_work, EmMode, EvalModPlan};
use pc_gpu::bts::finish::Pow2;
use pc_gpu::bts::raise::RaisePlan;
use pc_gpu::bts::stc_first::{nf_key, StcFirstPlan, StcFirstRoute, StcFirstSetup};
use pc_gpu::bts::stc_high::{StcHighSetup, StcHighShape};
use pc_gpu::bts::staged::{BtsCall, BtsLog, CallOpts, Phase, StagedRef};
use pc_gpu::bts::{BtsCx, BtsKeyRing, BtsPrecomp, BtsRoute, BtsSetup, BtsWork, BtsWorkShape, MemReq, StcHigh};
use pc_gpu::decrypt::{Decryptor, Which};
use pc_gpu::encrypt::Encryptor;
use pc_gpu::keygen::{test_prng, KeyGen};
use pc_gpu::keys::{DevSecret, KeySet, KeySetSpec, KskLayout, RotSpec};
use pc_gpu::ks::{KsPlans, KskSource};
use pc_gpu::ops::{Ops, OpsCfg};
use pc_gpu::poly::CtBuf;
use pc_gpu::{Context, Ct, Executor, MemCfg, MemPlan, PoolClass};
use crate::dump::{Dump, HostCt};
use pc_shadow::fides::FMeta;
use pc_shadow::{BtsConfig, CtMeta, Quirks, Route, Shadow, ShadowConfig};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

type W = u32;
type M = Barrett64;
type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub enum Mode {
    Check,
    Bench,
}

fn upload(buf: &mut CtBuf<'_, Compute, W>, s: &Stream<Compute>, c: usize, h: &HostCt) -> Res<()> {
    for part in 0..2 {
        buf.upload_rows(s, c, part, &h.ids, &h.c[part])?;
    }
    Ok(())
}

fn diff(buf: &CtBuf<'_, Compute, W>, s: &Stream<Compute>, c: usize, h: &HostCt) -> Res<(usize, usize)> {
    let mut bad = 0;
    for part in 0..2 {
        let got = buf.download_rows(s, c, part, &h.ids)?;
        bad += got.iter().zip(&h.c[part]).filter(|(a, b)| a != b).count();
    }
    Ok((bad, 2 * h.c[0].len()))
}

fn meta_of(n_q: u32, h: &HostCt) -> CtMeta {
    let mut m = CtMeta::new(n_q - 1 - h.level as u32, h.noise_level as u8).with_nf(h.noise_factor);
    m.slots = h.slots as u32;
    m
}

fn meta_eq(m: &CtMeta, n_q: u32, h: &HostCt) -> bool {
    m.level == n_q - 1 - h.level as u32 && m.deg as i64 == h.noise_level && m.nf.to_bits() == h.noise_factor.to_bits()
}

/// A dumped stage's geometry and plaintexts (`<prefix><k>.pt<j>.{q,p}.bin`).
fn dumped_stage(d: &Dump, st: &Value, prefix: &str, k: usize) -> Res<(LtGeom, Vec<Vec<u32>>, usize, u32, f64, usize)> {
    let b = st["bStep"].as_u64().ok_or("bStep")? as usize;
    let g = st["gStep"].as_u64().ok_or("gStep")? as usize;
    let ri = crate::dump::as_i64v(&st["rotIn"]);
    let ro = crate::dump::as_i64v(&st["rotOut"]);
    let pm = st["pts"].as_array().ok_or("pts")?;
    let geom = LtGeom::from_dump(b, g, &ri, &ro, pm.len());
    let ptw = pm[0]["c0"]["q_sizes"][0].as_u64().ok_or("q_sizes")? as usize;
    let enc = pm[0]["compact_mask"].as_u64().ok_or("compact_mask")? as u32;
    let nqpt = pm[0]["c0"]["ids"].as_array().ok_or("ids")?.len();
    let nf = pm[0]["noise_factor"].as_f64().ok_or("noise_factor")?;
    let rows = (0..pm.len())
        .map(|j| {
            let mut v = d.u32s(&format!("{prefix}{k}.pt{j}.q.bin"));
            v.extend(d.u32s(&format!("{prefix}{k}.pt{j}.p.bin")));
            v
        })
        .collect();
    Ok((geom, rows, ptw, enc, nf, nqpt))
}

pub(crate) fn rot_steps(geoms: &[LtGeom], n: usize) -> BTreeSet<i64> {
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

/// Raw stream access for a capture while the call borrows the executor mutably (the capture only reads the handle).
fn capture<R>(s: *const Stream<Compute>, f: impl FnOnce() -> pc_gpu::Result<R>) -> Res<(pc_cuda::GraphExec, R)> {
    // SAFETY: `s` points at the executor's compute stream, alive for the call; the closure issues work on it
    let s = unsafe { &*s };
    let mut err = None;
    let r = s.capture(|_| {
        f().map_err(|e| {
            err = Some(e.to_string());
            pc_cuda::Error::Unavailable("bootstrap under capture")
        })
    });
    match (r, err) {
        (_, Some(e)) => Err(e.into()),
        (r, None) => Ok(r?),
    }
}

fn read_sk(p: &std::path::Path) -> Res<Vec<i8>> {
    let (dt, _shape, off) = crate::dump_g0::npy_header(p)?;
    let b = std::fs::read(p)?;
    let body = &b[off..];
    Ok(match dt.trim_start_matches(['<', '|', '=']) {
        "i1" => body.iter().map(|&x| x as i8).collect(),
        "i8" => body.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap_or([0; 8])) as i8).collect(),
        o => return Err(format!("{}: dtype {o}", p.display()).into()),
    })
}

/// The integer message polynomial (centred CRT of the bottom two limbs) of every decrypted ciphertext.
fn int_poly(rows: &[Vec<u64>], q: [u64; 2]) -> Vec<i128> {
    let (q0, q1) = (q[0] as i128, q[1] as i128);
    let inv = invmod(q[0] % q[1], q[1]) as i128;
    let m = q0 * q1;
    rows[0]
        .iter()
        .zip(&rows[1])
        .map(|(&a, &b)| {
            let t = ((b as i128 - a as i128).rem_euclid(q1) * inv).rem_euclid(q1);
            let x = a as i128 + q0 * t;
            if x > m / 2 {
                x - m
            } else {
                x
            }
        })
        .collect()
}

fn rel_bits(got: &[i128], want: &[i128]) -> (f64, f64) {
    let mx = want.iter().fold(0f64, |a, &b| a.max((b as f64).abs()));
    let e = got.iter().zip(want).fold(0f64, |a, (&x, &y)| a.max(((x - y) as f64).abs()));
    let r = e / mx.max(1.0);
    (r, -r.log2())
}

// ── d3 ───────────────────────────────────────────────────────────────────────────────────────────────────────────

fn d3(dir: &str, ks_list: &[usize], graph: bool, t: &mut Tally) -> Res<()> {
    let t0 = Instant::now();
    let d = Dump::open(dir);
    let pj = d.json("params.json");
    let chain = Chain::generate(&ChainSpec::n32())?;
    if crate::dump::as_u64v(&pj["q"]) != chain.q {
        return Err("the dump's primes differ from the n32 preset".into());
    }
    let n = chain.n;
    let n2 = 2 * n as u64;
    let nitems = pj["errs"].as_array().ok_or("errs")?.len();
    let boot = &pj["boot"];
    let mut stages_host = Vec::new();
    for (k, st) in boot["CtS"].as_array().ok_or("boot.CtS")?.iter().enumerate() {
        stages_host.push(dumped_stage(&d, st, "cts", k)?);
    }
    let steps = rot_steps(&stages_host.iter().map(|s| s.0).collect::<Vec<_>>(), n);
    let k_max = 8;
    let dev = Device::open()?;
    let cfg = MemCfg { k_max, pool: vec![PoolClass { k: 1, count: 1 }], events: 64, scratch_cts: 2 };
    let mut plan = MemPlan::build::<W, M>(&chain, cfg)?;
    let lay = plan.lay;
    let primes = chain.primes();
    plan.reserve("ks.plans", KsPlans::bytes::<W>(&chain, k_max)?);
    plan.reserve("ks.regen_a", pc_gpu::ks::KsWork::<W>::regen_bytes(&lay, chain.dnum()));
    plan.reserve("keys", (steps.len() + 2) * pc_gpu::ks::DevKey::<W>::bytes(&lay, chain.dnum(), true, true));
    let ocfg = OpsCfg::default();
    Ops::<W>::reserve(&mut plan, &ocfg);
    let mut bts_bytes = RaisePlan::<W>::table_bytes() + Pow2::<W>::bytes(lay.rows()) + (64 << 20);
    let fold0 = d.ct("it0.fold");
    let mut top = fold0.ids.len() - 1;
    for (g, rows, ptw, _, _, _) in &stages_host {
        bts_bytes += LtPts::<W>::bytes(rows.len(), top + 1 + lay.kp, *ptw) + LtStage::<W>::plan_bytes(&lay, &primes, g, top, LtEnd::FusedRescale, k_max)?;
        top -= 2;
    }
    plan.reserve("bts.tables", bts_bytes);
    let in_meta = {
        let m = d.json("it0.cts3.json");
        FMeta { top: m["level"].as_i64().unwrap_or(-1) as i32, nl: m["noise_level"].as_i64().unwrap_or(1) as i32, nf: m["noise_factor"].as_f64().unwrap_or(0.0) }
    };
    let ps = ps_plan(&pj);
    let da = pj["double_angle_its"].as_u64().ok_or("double_angle_its")? as u32;
    let (_, _, nslots) = EvalModPlan::<W>::program(&pc_core::scale::ScaleTables::precompute(&chain)?, in_meta, &ps, da)?;
    let (em_n, em_kk) = em_work(nslots, k_max, EmMode::Complex);
    let wshape = BtsWorkShape { k_max, g_max: 2, em_slots: em_n, em_kk, spare: 1 };
    plan.reserve("bts.work", BtsWork::<W>::bytes(&lay, wshape));
    plan.reserve("check.x", pc_cuda::Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(&lay, k_max)));
    // K separate ciphertexts through BtsBatch + BtsGraphs (fixed gather/scatter slots)
    plan.reserve("check.sep", k_max * pc_cuda::Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(&lay, 1)));
    plan.reserve("bts.batch", pc_gpu::graph::BtsGraphs::<W>::bytes(&lay, k_max));
    let arena = plan.allocate(&dev)?;
    let mut ex = Executor::<W>::new(&dev, &arena, &plan)?;
    ex.ks.carve_regen(&arena, &lay)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &plan)?;
    let kplans = KsPlans::new(&arena, &ex.compute, &ctx, k_max)?;
    let mut ops = Ops::new(&arena, &dev, &ex.compute, &ctx, ocfg)?;
    let mut work = BtsWork::<W>::new(&arena, lay, wshape)?;
    let s = &ex.compute;
    let mut keys = OracleKeys { rot: BTreeMap::new(), conj: None, relin: None, atob: None, btoa: None, n };
    for &st in &steps {
        let g = pc_core::galois::rotation_galois(st, n)?;
        keys.rot.insert(st, load_key(&arena, s, &ctx, &kplans, &d, &pj, &g.to_string(), g)?);
    }
    keys.conj = Some(load_key(&arena, s, &ctx, &kplans, &d, &pj, &(n2 - 1).to_string(), n2 - 1)?);
    keys.relin = Some(load_key(&arena, s, &ctx, &kplans, &d, &pj, "mult", 1)?);
    let stage_ptrs = ex.ks.stage_ptrs();
    let mut stages = Vec::new();
    let mut top = fold0.ids.len() - 1;
    for (g, rows, ptw, enc, nf, nqpt) in stages_host {
        if nqpt != top + 1 {
            return Err(format!("stage plaintexts hold {nqpt} limbs, the ciphertext {}", top + 1).into());
        }
        let pts = LtPts::new(&arena, s, &ctx, top, PtSource::Rows { rows, ptw, enc })?;
        stages.push(LtStage::new(&arena, s, &ctx, g, top, LtEnd::FusedRescale, nf, pts, &keys, stage_ptrs, k_max)?);
        top -= 2;
    }
    let em = EvalModPlan::new(&arena, s, &ctx, EmMode::Complex, in_meta, &ps, da, keys.conj()?, keys.relin()?, stage_ptrs, k_max, KskSource::Regen)?;
    let raise_top = fold0.level;
    let drop = boot["raise_drop"].as_u64().unwrap_or(0) as u32;
    let exact = crate::check_bts::eval_const(&ctx, drop as usize, pc_core::bts_setup::EVALMOD_K);
    let cts0_t = boot["cts0_t"].as_u64().ok_or("cts0_t")? as u32;
    let mut setup = BtsSetup::n32_shipped();
    setup.raise_drops = vec![drop];
    setup.real = false;
    setup.k_max = k_max;
    let mut sf = StcFirstSetup::n32_shipped();
    sf.cf_default = None;
    let pre = BtsPrecomp {
        slots: (n / 2) as u32,
        raise_drop: drop,
        raise: RaisePlan::new(&arena, s, &ctx, raise_top, exact, cts0_t)?,
        cts: stages,
        stc: vec![],
        stc_real0: None,
        em,
        em_real: None,
        cts0_const: exact,
        cts0_t,
        boot_k: pc_core::bts_setup::EVALMOD_K,
    };
    let dg = chain.level_unit;
    let qd = (0..dg).fold(1f64, |a, j| a * chain.q[j] as f64);
    let deg = (qd / (chain.spec.scaling_bits as f64).exp2()).log2().round() as u32;
    let entry_top = boot["stc_first_entry"].as_u64().ok_or("stc_first_entry")? as usize;
    let route = StcFirstRoute::<W, M>::from_parts(setup, sf, vec![pre], vec![], None, None, Pow2::new(&arena, s, &ctx)?, deg, entry_top, vec![], vec![], None, (MemReq::default(), wshape));
    let mut x = Ct::new(CtMeta::new(0, 1), CtBuf::<Compute, W>::new(&arena, "check.x", lay, k_max)?);
    let mut sep: Vec<Ct<'_, W>> = (0..k_max).map(|_| CtBuf::<Compute, W>::new(&arena, "check.sep", lay, 1).map(|b| Ct::new(CtMeta::new(0, 1), b))).collect::<pc_gpu::Result<_>>()?;
    let mut graphs = pc_gpu::graph::BtsGraphs::<W>::new(&arena, lay, k_max, 16)?;
    s.sync()?;
    println!("[pc-check] stcf d3 setup {:.1} s: {} rotation keys, arena {:.1} GB", t0.elapsed().as_secs_f64(), keys.rot.len(), arena.used_bytes() as f64 / 1e9);
    arena.freeze();
    let sh = Shadow::new(ShadowConfig { chain: ctx.shape, bts: BtsConfig::n32_shipped(), quirks: Quirks::none() }, Some(Arc::new(ctx.scale.clone())));
    let nq = ctx.shape.n_q;
    // the call's state from the input's metadata (pre_input: limb 21, before the input StC), as the PCMM takes it
    let pre_in = d.ct("it0.pre_input");
    let cf = boot["correction_factor"].as_u64().ok_or("correction_factor")? as u32;
    let r = Route { cf: Some(cf), stc_first: true, raise_drop: drop, ..Route::dense() };
    let st = route.begin_meta(&ctx, &meta_of(nq, &pre_in), 1, &r, false)?;
    let sfj = &pj["state_fields"];
    // (n_stc: this harness builds no shipped StC stages, the route reports 0; FIDESlib 3)
    let fields = [
        ("stc_first", st.stc_first == sfj["stc_first"].as_bool().unwrap_or(false)),
        ("stc_folded", st.stc_folded == sfj["stc_folded"].as_bool().unwrap_or(false)),
        ("correction", st.correction as u64 == sfj["correction"].as_u64().unwrap_or(0)),
        ("cor_factor", st.cor_factor == sfj["cor_factor"].as_u64().unwrap_or(0)),
        ("n_cts", st.n_cts as u64 == sfj["n_cts"].as_u64().unwrap_or(0)),
        ("eval_const", st.const_eval_mult.to_bits() == sfj["eval_const"].as_f64().unwrap_or(0.0).to_bits()),
        ("exact_const", st.exact_const == sfj["exact_const"].as_bool().unwrap_or(false)),
    ];
    for (f, ok) in &fields {
        if !ok {
            println!("[pc-check]   state field {f} differs: {st:?}");
        }
    }
    t.add("d3 BtsCall state from pre_input (stc_first, folded, correction, cor_factor, n_cts, eval_const, exact)", fields.iter().filter(|f| !f.1).count(), fields.len());
    t.add("d3 entry limb (route) vs boot.stc_first_entry", (route.entry_top != entry_top) as usize, 1);
    let item = |c: usize| c % nitems;
    let fold_meta = meta_of(nq, &fold0);
    let opts = CallOpts { prescaled: false, own_cf: true };
    for &kk in ks_list {
        x.buf.k = kk;
        for c in 0..kk {
            upload(&mut x.buf, &ex.compute, c, &d.ct(&format!("it{}.fold", item(c))))?;
        }
        x.meta = fold_meta.clone();
        let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
        let mut call = BtsCall::resume(StagedRef::StcFirst(&route), st.clone(), Phase::Folded, opts);
        let (mut mbad, mut mtot) = (0, 0);
        while call.phase() != Phase::Done {
            let ph = call.step(&mut cx, std::slice::from_mut(&mut x))?;
            let name = match ph {
                Phase::Cts(k) => Some(format!("cts{}", k - 1)),
                Phase::Done => Some("out".to_string()),
                _ => None,
            };
            if let Some(nm) = name {
                let (mut b, mut o) = (0, 0);
                for c in 0..kk {
                    let h = d.ct(&format!("it{}.{nm}", item(c)));
                    let (x1, o1) = diff(&x.buf, &cx.ex.compute, c, &h)?;
                    b += x1;
                    o += o1;
                    if c == 0 {
                        mtot += 1;
                        if !meta_eq(&x.meta, nq, &h) {
                            mbad += 1;
                            println!("[pc-check]   meta after {nm}: {:?} vs dump top {} deg {} nf {:e}", (x.meta.level, x.meta.deg, x.meta.nf), h.level, h.noise_level, h.noise_factor);
                        }
                    }
                }
                let what = if nm == "out" { "EvalMod + stc_first_output vs it.out".to_string() } else { format!("CtS stage {} vs it.{nm}", &nm[3..]) };
                t.add(&format!("K={kk} staged eager {what}"), b, o);
            }
        }
        t.add(&format!("K={kk} staged metadata after every checked stage (level, deg, nf bits)"), mbad, mtot);
        if graph {
            x.meta = fold_meta.clone();
            let sp: *const Stream<Compute> = &cx.ex.compute;
            let xp: *mut Ct<'_, W> = &mut x;
            let st2 = st.clone();
            let rr = &route;
            let (gexec, _) = capture(sp, || {
                // SAFETY: x is not otherwise borrowed while the closure runs
                let x = unsafe { &mut *xp };
                let mut call = BtsCall::resume(StagedRef::StcFirst(rr), st2, Phase::Folded, opts);
                call.run_to(&mut cx, std::slice::from_mut(x), Phase::Done)
            })?;
            for rep in 0..2 {
                for c in 0..kk {
                    upload(&mut x.buf, &cx.ex.compute, c, &d.ct(&format!("it{}.fold", item(c))))?;
                }
                gexec.launch(&cx.ex.compute)?;
                cx.ex.compute.sync()?;
                let (mut b, mut o) = (0, 0);
                for c in 0..kk {
                    let (x1, o1) = diff(&x.buf, &cx.ex.compute, c, &d.ct(&format!("it{}.out", item(c))))?;
                    b += x1;
                    o += o1;
                }
                t.add(&format!("K={kk} graph replay {rep}: staged finish (CtS x4 + EvalMod + stc_first_output) vs it.out"), b, o);
            }
        }
    }
    if graph {
        // the same finish over K separate ciphertexts as one BtsBatch through BtsGraphs: the first call runs
        // eagerly and captures, the next two replay (gather -> launch -> scatter)
        let locks: Vec<&dyn pc_gpu::graph::CaptureLock> = route.last.iter().map(|l| l as &dyn pc_gpu::graph::CaptureLock).collect();
        for &kk in ks_list {
            for rep in 0..3 {
                for (c, y) in sep.iter_mut().enumerate().take(kk) {
                    upload(&mut y.buf, &ex.compute, 0, &d.ct(&format!("it{}.fold", item(c))))?;
                    y.meta = fold_meta.clone();
                }
                let served = {
                    let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                    let mut b = pc_gpu::bts::batch::BtsBatch::new(r, sep.iter_mut().take(kk).collect())?;
                    let rr = &route;
                    graphs.run(&mut cx, &mut b, 1, &locks, |cx, y| {
                        let mut call = BtsCall::resume(StagedRef::StcFirst(rr), st.clone(), Phase::Folded, opts);
                        call.run_to(cx, std::slice::from_mut(y), Phase::Done)
                    })?
                };
                ex.compute.sync()?;
                let (mut b, mut o, mut mb) = (0, 0, 0);
                for (c, y) in sep.iter().enumerate().take(kk) {
                    let h = d.ct(&format!("it{}.out", item(c)));
                    let (x1, o1) = diff(&y.buf, &ex.compute, 0, &h)?;
                    b += x1;
                    o += o1;
                    mb += !meta_eq(&y.meta, nq, &h) as usize;
                }
                t.add(&format!("K={kk} batched+graph call {rep} ({served:?}): {kk} separate cts, finish vs it.out"), b, o);
                t.add(&format!("K={kk} batched+graph call {rep}: metadata vs it.out"), mb, kk);
            }
        }
        let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
        println!("[pc-check] batched+graph: {} graphs, {:?}, {} constant slots leased", graphs.len(), graphs.stats, cx.ops.ring.leased());
        graphs.clear(&mut cx)?;
    }
    // order checks of the call object (no device work)
    let mut call = BtsCall::<W, M>::resume(StagedRef::StcFirst(&route), st.clone(), Phase::Begun, opts);
    let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
    let refused = [call.fold(&mut cx, std::slice::from_mut(&mut x)).is_err(), call.cts_stage(&mut cx, std::slice::from_mut(&mut x), 1).is_err(), call.stc_enter(&mut cx, std::slice::from_mut(&mut x)).is_err()];
    t.add("BtsCall refuses out-of-order stages (fold / cts_stage(1) / stc_enter at Begun)", refused.iter().filter(|&&r| !r).count(), 3);
    Ok(())
}

// ── g0 ───────────────────────────────────────────────────────────────────────────────────────────────────────────

fn g0(dir: &str, t: &mut Tally) -> Res<()> {
    let t0 = Instant::now();
    let g0 = crate::dump_g0::G0::open(dir)?;
    let pj = g0.params.clone();
    let bd = g0.bts();
    let keysd = Dump::open(&g0.dir);
    let chain = Chain::generate(&ChainSpec::n32())?;
    if crate::dump::as_u64v(&pj["q"]) != chain.q {
        return Err("the dump's primes differ from the n32 preset".into());
    }
    let n = chain.n;
    let n2 = 2 * n as u64;
    let rj = &pj["boot_routes"]["32768"];
    let rd = Dump::open(g0.dir.join("routes").join("32768"));
    let mut cts_host = Vec::new();
    for (k, st) in rj["CtS"].as_array().ok_or("boot_routes.32768.CtS")?.iter().enumerate() {
        cts_host.push(dumped_stage(&rd, st, "cts", k)?);
    }
    let real = pj["run"]["env"]["FIDESLIB_BTS_REAL"].as_str() == Some("1");
    let k_max = 1;
    let mut setup = BtsSetup::n32_shipped();
    setup.raise_drops = vec![rj["raise_drop"].as_u64().unwrap_or(0) as u32];
    setup.real = real;
    setup.k_max = k_max;
    let mut sf = StcFirstSetup::n32_shipped();
    sf.cache_slots = 2;
    let plan = StcFirstPlan::new(&chain, &setup, &sf)?;
    let v0 = &plan.dense.variants[0];
    let mut geoms: Vec<LtGeom> = v0.cts.iter().chain(&v0.stc).chain(&plan.input).map(|h| h.geom).collect();
    geoms.extend(plan.sources.iter().flatten().map(|s| s.geom));
    let steps = rot_steps(&geoms, n);
    let dev = Device::open()?;
    let cfg = MemCfg { k_max, pool: vec![PoolClass { k: 1, count: 1 }], events: 64, scratch_cts: 2 };
    let mut mp = MemPlan::build::<W, M>(&chain, cfg)?;
    let lay = mp.lay;
    let primes = chain.primes();
    mp.reserve("ks.plans", KsPlans::bytes::<W>(&chain, k_max)?);
    mp.reserve("ks.regen_a", pc_gpu::ks::KsWork::<W>::regen_bytes(&lay, chain.dnum()));
    mp.reserve("keys", (steps.len() + 4) * pc_gpu::ks::DevKey::<W>::bytes(&lay, chain.dnum(), true, true));
    let ocfg = OpsCfg::default();
    Ops::<W>::reserve(&mut mp, &ocfg);
    let (req, wshape) = plan.mem::<W>()?;
    req.reserve(&mut mp);
    let mut dump_bytes = 0;
    let mut top = chain.q.len() - 1;
    for (g, rows, ptw, _, _, _) in &cts_host {
        dump_bytes += LtPts::<W>::bytes(rows.len(), top + 1 + lay.kp, *ptw) + LtStage::<W>::plan_bytes(&lay, &primes, g, top, LtEnd::FusedRescale, k_max)?;
        top -= 2;
    }
    mp.reserve("check.dumped_cts", dump_bytes);
    mp.reserve("bts.work", BtsWork::<W>::bytes(&lay, wshape));
    mp.reserve("check.x", 4 * pc_cuda::Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(&lay, 1)));
    mp.reserve("secret", DevSecret::<W>::bytes(&lay));
    mp.reserve("decrypt", Decryptor::<W>::bytes(&lay, chain.level_unit, 1));
    let arena = mp.allocate(&dev)?;
    let mut ex = Executor::<W>::new(&dev, &arena, &mp)?;
    ex.ks.carve_regen(&arena, &lay)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &mp)?;
    let kplans = KsPlans::new(&arena, &ex.compute, &ctx, k_max)?;
    let mut ops = Ops::new(&arena, &dev, &ex.compute, &ctx, ocfg)?;
    let mut work = BtsWork::<W>::new(&arena, lay, wshape)?;
    let s = &ex.compute;
    let mut keys = OracleKeys { rot: BTreeMap::new(), conj: None, relin: None, atob: None, btoa: None, n };
    for &st in &steps {
        let g = pc_core::galois::rotation_galois(st, n)?;
        keys.rot.insert(st, load_key(&arena, s, &ctx, &kplans, &keysd, &pj, &g.to_string(), g)?);
    }
    keys.conj = Some(load_key(&arena, s, &ctx, &kplans, &keysd, &pj, &(n2 - 1).to_string(), n2 - 1)?);
    keys.relin = Some(load_key(&arena, s, &ctx, &kplans, &keysd, &pj, "mult", 1)?);
    keys.atob = Some(load_key(&arena, s, &ctx, &kplans, &keysd, &pj, &(n2 - 2).to_string(), 1)?);
    keys.btoa = Some(load_key(&arena, s, &ctx, &kplans, &keysd, &pj, &(n2 - 4).to_string(), 1)?);
    let stage_ptrs = ex.ks.stage_ptrs();
    let mut route = plan.build(&arena, &dev, s, &ctx, &keys, stage_ptrs)?;
    // the dump's CtS plaintexts (ours differ by encoder rounding)
    let mut dumped = Vec::new();
    let mut top = chain.q.len() - 1;
    for (g, rows, ptw, enc, nf, _) in cts_host {
        let pts = LtPts::new(&arena, s, &ctx, top, PtSource::Rows { rows, ptw, enc })?;
        dumped.push(LtStage::new(&arena, s, &ctx, g, top, LtEnd::FusedRescale, nf, pts, &keys, stage_ptrs, k_max)?);
        top -= 2;
    }
    let own_cts = std::mem::replace(&mut route.variants[0].cts, dumped);
    drop(own_cts);
    let mut sec = DevSecret::<W>::carve(&arena, lay)?;
    let sk = read_sk(&g0.dir.join("sk.npy"))?;
    sec.load(&ctx, s, &SecretKey { s: sk, s_sparse: vec![] })?;
    let mut dec = Decryptor::<W>::new(&arena, s, lay, chain.level_unit, 1)?;
    dec.load_secret(s, &sec)?;
    let mut x = Ct::new(CtMeta::new(0, 1), CtBuf::<Compute, W>::new(&arena, "check.x", lay, 1)?);
    let mut y = Ct::new(CtMeta::new(0, 1), CtBuf::<Compute, W>::new(&arena, "check.x", lay, 1)?);
    let mut z = Ct::new(CtMeta::new(0, 1), CtBuf::<Compute, W>::new(&arena, "check.x", lay, 1)?);
    let mut w = Ct::new(CtMeta::new(0, 1), CtBuf::<Compute, W>::new(&arena, "check.x", lay, 1)?);
    s.sync()?;
    println!("[pc-check] stcf g0 setup {:.1} s: {} rotation keys, arena {:.1} of {:.1} GB; entry limb {}, {} input stages, cache {} entries", t0.elapsed().as_secs_f64(), keys.rot.len(), arena.used_bytes() as f64 / 1e9, arena.capacity() as f64 / 1e9, route.entry_top, route.input.len(), sf.cache_slots);
    arena.freeze();
    let sh = Shadow::new(ShadowConfig { chain: ctx.shape, bts: BtsConfig::n32_shipped(), quirks: Quirks::none() }, Some(Arc::new(ctx.scale.clone())));
    let nq = ctx.shape.n_q;
    let unit = ctx.shape.unit;
    let q2 = [chain.q[0], chain.q[1]];
    // entry and landings vs pc-shadow (FAKECAP_STC_ENTRY 46, StC-first lands FAKECAP_STC_DROP = 6 levels above the shipped order)
    let bcfg = BtsConfig::n32_shipped();
    t.add("entry level vs pc-shadow BtsConfig::stc_first_entry", (route.entry_level(ctx.shape).ok() != bcfg.stc_first_entry) as usize, 1);
    let h_in = bd.ct("2.in");
    let h_out = bd.ct("2.out");
    let r = Route { stc_first: true, real, ..Route::dense() };
    let land = route.landing(&sh, &meta_of(nq, &h_in), &r)?;
    t.add("landing (raw) vs dumped 2.out (level, deg, nf bits)", (!meta_eq(&land, nq, &h_out)) as usize, 1);
    let realized = sh.rescale(&land)?.out;
    let want = bcfg.landing(ctx.shape, &Quirks::none(), &r, h_in_level(nq, &h_in), h_in.noise_level as u8)?;
    let fb = route.landing(&sh, &CtMeta::new(48, 1).with_nf(1.0), &r)?;
    let fb_r = sh.rescale(&fb)?.out;
    let want_fb = bcfg.landing(ctx.shape, &Quirks::none(), &r, 48, 1)?;
    println!("[pc-check]   landing realized {:?} vs shadow {:?}; deep input (level 48) shipped order {:?} vs shadow fallback {:?}", (realized.level, realized.deg), (want.level, want.deg), (fb_r.level, fb_r.deg), (want_fb.level, want_fb.deg));
    t.add("realized StC-first landing vs pc-shadow BtsConfig::landing", ((realized.level, realized.deg) != (want.level, want.deg)) as usize, 1);
    t.add("deep input (level 48 > entry): shipped-order landing vs pc-shadow fallback", ((fb_r.level, fb_r.deg) != (want_fb.level, want_fb.deg)) as usize, 1);
    let cap = route.input_cap(ctx.shape)?;
    t.add("PCMM input cap (entry after a pending rescale) == 44 (old _N32_CAP)", (cap != 44) as usize, 1);
    let st = route.begin_meta(&ctx, &meta_of(nq, &h_in), 1, &r, false)?;
    let sj = &bd.json("2.json")["state"];
    let bad = [
        st.stc_first == sj["stc_first"].as_bool().unwrap_or(false),
        st.stc_folded == sj["stc_folded"].as_bool().unwrap_or(false),
        st.correction as u64 == sj["correction"].as_u64().unwrap_or(0),
        st.cor_factor == sj["cor_factor"].as_u64().unwrap_or(0),
        st.n_cts as u64 == sj["n_cts"].as_u64().unwrap_or(0),
        st.const_eval_mult.to_bits() == sj["eval_const"].as_f64().unwrap_or(0.0).to_bits(),
    ]
    .iter()
    .filter(|&&ok| !ok)
    .count();
    t.add("g0 #2 BtsCall state (stc_first, folded, correction 11, cor_factor 2048, n_cts, eval_const)", bad, 6);
    let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
    // (a) the dumped split through the staged API, every stage bit-exact
    let h_split = bd.ct("2.stc_first_input");
    upload(&mut x.buf, &cx.ex.compute, 0, &h_split)?;
    x.meta = meta_of(nq, &h_split);
    let mut call = BtsCall::resume(StagedRef::StcFirst(&route), st.clone(), Phase::Begun, CallOpts::default());
    call.at_split(&cx, std::slice::from_ref(&x))?;
    while call.phase() != Phase::Done {
        let ph = call.step(&mut cx, std::slice::from_mut(&mut x))?;
        let nm = match ph {
            Phase::Raised => "mod_raise".to_string(),
            Phase::Folded => "fold".to_string(),
            Phase::Cts(k) => format!("cts_{}", k - 1),
            Phase::EvalMod => "eval_mod".to_string(),
            Phase::Done => "stc_first_output".to_string(),
            _ => continue,
        };
        let h = bd.ct(&format!("2.{nm}"));
        let (b, o) = diff(&x.buf, &cx.ex.compute, 0, &h)?;
        t.add(&format!("g0 #2 staged {nm} (from the dumped split)"), b, o);
        if !meta_eq(&x.meta, nq, &h) {
            println!("[pc-check]   meta after {nm}: {:?} vs dump top {} deg {} nf {:e}", (x.meta.level, x.meta.deg, x.meta.nf), h.level, h.noise_level, h.noise_factor);
        }
        t.add(&format!("g0 #2 staged {nm} metadata"), (!meta_eq(&x.meta, nq, &h)) as usize, 1);
    }
    let (b, o) = diff(&x.buf, &cx.ex.compute, 0, &h_out)?;
    t.add("g0 #2 staged output == 2.out", b, o);
    // (b) our input side on the dumped input
    upload(&mut y.buf, &cx.ex.compute, 0, &h_in)?;
    y.meta = meta_of(nq, &h_in);
    let mut call = BtsCall::begin(StagedRef::StcFirst(&route), &cx, std::slice::from_ref(&y), &r, CallOpts::default())?;
    let info = call.stc_first_input(&mut cx, std::slice::from_mut(&mut y))?.ok_or("not StC-first")?;
    println!("[pc-check]   split: nf_in {:e}, cf_stage {} (pre-factor 2^{}), factor {:e}, key {:?}; meta {:?} vs dump top {} deg {} nf {:e}", info.nf_in, info.cf_stage, info.pre_factor_log2, info.factor, info.key, (y.meta.level, y.meta.deg, y.meta.nf), h_split.level, h_split.noise_level, h_split.noise_factor);
    t.add("g0 #2 our split metadata (level, deg, nf bits) == dumped split", (!meta_eq(&y.meta, nq, &h_split)) as usize, 1);
    let (b, o) = diff(&y.buf, &cx.ex.compute, 0, &h_split)?;
    println!("[pc-check]   our StC-first plaintexts (own encoding, not the old OpenFHE rows): {b} of {o} split words differ (informational)");
    // integer-domain comparison: both realized, decrypted with the dump's secret
    let mut cmp = |cx: &mut BtsCx<'_, '_, W, M>, a: &Ct<'_, W>, h: &HostCt, what: &str, t: &mut Tally| -> Res<()> {
        let words = a.buf.buf.len();
        z.buf.buf.copy_from(&cx.ex.compute, &a.buf.buf, words)?;
        z.meta = a.meta.clone();
        upload(&mut w.buf, &cx.ex.compute, 0, h)?;
        w.meta = meta_of(nq, h);
        cx.ops.realize(cx.ctx, cx.ex, cx.sh, std::slice::from_mut(&mut z))?;
        cx.ops.realize(cx.ctx, cx.ex, cx.sh, std::slice::from_mut(&mut w))?;
        let mz = int_poly(&dec.decrypt_rows(cx.ctx, &cx.ex.compute, &z, Which::Dense)?[0], q2);
        let mw = int_poly(&dec.decrypt_rows(cx.ctx, &cx.ex.compute, &w, Which::Dense)?[0], q2);
        let (r, bits) = rel_bits(&mz, &mw);
        println!("[pc-check] {what}: max |ours - dump| / max |dump| = {r:.3e} ({bits:.1} bits) over the decrypted integers (realized, level {})", z.meta.level);
        t.add(&format!("{what}: agreement >= 20 bits"), (bits < 20.0) as usize, 1);
        Ok(())
    };
    cmp(&mut cx, &y, &h_split, "g0 #2 our split vs dumped split", t)?;
    call.run_to(&mut cx, std::slice::from_mut(&mut y), Phase::Done)?;
    t.add("g0 #2 our whole refresh: output metadata == 2.out", (!meta_eq(&y.meta, nq, &h_out)) as usize, 1);
    // precision against the message (the decrypted input): ours vs the oracle refresh of the same input
    let enc = Encoder::new(&chain);
    let mut slots_of = |cx: &mut BtsCx<'_, '_, W, M>, a: Option<&Ct<'_, W>>, h: Option<&HostCt>| -> Res<Vec<Complex>> {
        match (a, h) {
            (Some(a), _) => {
                let words = a.buf.buf.len();
                z.buf.buf.copy_from(&cx.ex.compute, &a.buf.buf, words)?;
                z.meta = a.meta.clone();
            }
            (None, Some(h)) => {
                upload(&mut z.buf, &cx.ex.compute, 0, h)?;
                z.meta = meta_of(nq, h);
            }
            _ => return Err("slots_of".into()),
        }
        cx.ops.realize(cx.ctx, cx.ex, cx.sh, std::slice::from_mut(&mut z))?;
        Ok(dec.decrypt_slots(cx.ctx, &cx.ex.compute, &enc, &z, Payload::Complex, Which::Dense)?.remove(0))
    };
    let msg = slots_of(&mut cx, None, Some(&h_in))?;
    let ours = slots_of(&mut cx, Some(&y), None)?;
    let theirs = slots_of(&mut cx, None, Some(&h_out))?;
    let amp = msg.iter().fold(0f64, |a, v| a.max(v.re.abs()));
    let err = |o: &[Complex]| o.iter().zip(&msg).fold(0f64, |a, (g, v)| a.max((g.re - v.re).abs())) / amp;
    let (eo, et) = (err(&ours), err(&theirs));
    let imx = msg.iter().fold(0f64, |a, v| a.max(v.im.abs()));
    println!("[pc-check] g0 #2 whole refresh vs the decrypted input (Re, max |in| {amp:.3}, max |Im in| {imx:.2e}): ours {eo:.3e} ({:.2} bits), old stack {et:.3e} ({:.2} bits)", -eo.log2(), -et.log2());
    t.add("g0 #2 our refresh precision >= the old stack's - 0.5 bit (same input, CF 13 at reference CF 20)", (-eo.log2() < -et.log2() - 0.5) as usize, 1);
    if let Some(c) = &route.last {
        println!("[pc-check]   last-stage cache: {:?}, keys {:?}", c.stats(), c.keys());
    }
    Ok(())
}

fn h_in_level(nq: u32, h: &HostCt) -> u32 {
    nq - 1 - h.level as u32
}

pub fn cmd(mode: Mode, args: &[String]) -> ExitCode {
    let r = match mode {
        Mode::Check => check(args),
        Mode::Bench => bench(args),
    };
    match r {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("stcf: {e}");
            ExitCode::FAILURE
        }
    }
}

fn check(args: &[String]) -> Res<bool> {
    let mut dir = None;
    let mut ks = vec![1usize, 2, 4, 8];
    let mut graph = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--k" => {
                ks = args.get(i + 1).ok_or("--k")?.split(',').map(|x| x.parse()).collect::<std::result::Result<_, _>>()?;
                i += 1;
            }
            "--graph" => graph = true,
            o => dir = Some(o.to_string()),
        }
        i += 1;
    }
    let dir = dir.ok_or("usage: pc-check stcf <d3|g0> [--k 1,2,4,8] [--graph]")?;
    let mut t = Tally::default();
    if std::path::Path::new(&dir).join("bts").is_dir() {
        g0(&dir, &mut t)?;
    } else {
        d3(&dir, &ks, graph, &mut t)?;
    }
    println!("[pc-check] stcf: {} gates, {} mismatches", t.gates, t.bad);
    Ok(t.bad == 0)
}

struct BArgs {
    cmd: String,
    ks: Vec<usize>,
    reps: usize,
    cfs: Vec<u32>,
    amps: Vec<f64>,
    real: bool,
    level: u32,
    graph: bool,
    out: Option<String>,
    gauss: Option<f64>,
    /// stcf-pair: the members (CF, amplitude), seeds per case, the dirty partner's imaginary lane (amplitude 2^-dirty)
    members: Vec<(u32, f64)>,
    seeds: usize,
    dirty: u32,
    /// fbs: table size t (inputs |j| <= t/2), table (sign | step | relu | rand), non-integer sweep
    fbs_t: u32,
    lut: String,
    frac: bool,
    /// fbs with a smooth series: coefficients file (JSON list, c0 not halved, output units), CF, sine amplitude,
    /// input range |x| <= range, (input, output) f64 pairs dumped to `dump`
    post_file: Option<String>,
    fbs_cf: Option<u32>,
    amp: f64,
    range: f64,
    dump: Option<String>,
    /// fbs: the EvalMod sine table (file of Chebyshev coefficients, evaluator convention), its K and double angles
    /// (a table for K >= 28: the ModRaise overflow |I| > K is a wrong output in a functional bootstrap)
    em_table: Option<String>,
    em_k: f64,
    em_da: u32,
    /// fbs smooth: half the inputs drawn from these samples (raw f32, clipped to the range), half uniform; `pair` > 0:
    /// one x per slot as Re = x - pair, Im = x + pair (two series on disjoint domains of one sine)
    samples: Option<String>,
    pair: f64,
    /// fbs trigonometric form (PC_FBS_ON_ROUTE=1): Q's coefficients file (as --post-file: P), t's offset, the base
    /// pair's degree and double angles (`pc_shadow::evalmod::trig_base`)
    post_s_file: Option<String>,
    trig_off: f64,
    trig_deg: usize,
    trig_da: u32,
}

fn bargs(a: &[String]) -> Res<BArgs> {
    let members = vec![(13, 64.0), (10, 8.0), (6, 1.0), (2, 0.25)];
    let mut r = BArgs { cmd: a.first().cloned().ok_or("command")?, ks: vec![1, 2, 4, 8], reps: 10, cfs: vec![13], amps: vec![0.1, 1.0], real: false, level: 44, graph: false, out: None, gauss: None, members, seeds: 4, dirty: 4, fbs_t: 16, lut: "sign".into(), frac: false, post_file: None, fbs_cf: None, amp: 1.0, range: 1.0, dump: None, em_table: None, em_k: 16.0, em_da: 5, samples: None, pair: 0.0, post_s_file: None, trig_off: 0.0, trig_deg: 27, trig_da: 5 };
    let list = |s: &str| -> Res<Vec<f64>> { s.split(',').map(|x| x.parse::<f64>().map_err(|e| e.into())).collect() };
    let mut i = 1;
    while i < a.len() {
        let v = a.get(i + 1).cloned().unwrap_or_default();
        match a[i].as_str() {
            "--k" => r.ks = list(&v)?.into_iter().map(|x| x as usize).collect(),
            "--reps" => r.reps = v.parse()?,
            "--cf" => r.cfs = list(&v)?.into_iter().map(|x| x as u32).collect(),
            "--amps" => r.amps = list(&v)?,
            "--level" => r.level = v.parse()?,
            "--out" => r.out = Some(v),
            "--gauss" => r.gauss = Some(v.parse()?),
            "--members" => {
                r.members = v
                    .split(',')
                    .map(|m| -> Res<(u32, f64)> {
                        let (c, a) = m.split_once(':').ok_or("--members cf:amp,...")?;
                        Ok((c.parse()?, a.parse()?))
                    })
                    .collect::<Res<_>>()?
            }
            "--seeds" => r.seeds = v.parse()?,
            "--t" => r.fbs_t = v.parse()?,
            "--lut" => r.lut = v.clone(),
            "--post-file" => r.post_file = Some(v.clone()),
            "--fbs-cf" => r.fbs_cf = Some(v.parse()?),
            "--amp" => r.amp = v.parse()?,
            "--range" => r.range = v.parse()?,
            "--dump" => r.dump = Some(v.clone()),
            "--em-table" => r.em_table = Some(v.clone()),
            "--em-k" => r.em_k = v.parse()?,
            "--em-da" => r.em_da = v.parse()?,
            "--samples" => r.samples = Some(v.clone()),
            "--pair" => r.pair = v.parse()?,
            "--post-s-file" => r.post_s_file = Some(v.clone()),
            "--trig-off" => r.trig_off = v.parse()?,
            "--trig-deg" => r.trig_deg = v.parse()?,
            "--trig-da" => r.trig_da = v.parse()?,
            "--frac" => {
                r.frac = true;
                i += 1;
                continue;
            }
            "--dirty" => r.dirty = v.parse()?,
            "--real" => {
                r.real = true;
                i += 1;
                continue;
            }
            "--graph" => {
                r.graph = true;
                i += 1;
                continue;
            }
            o => return Err(format!("unknown argument {o}").into()),
        }
        i += 2;
    }
    Ok(r)
}

fn rand_c(seed: u64, n: usize, amp: f64, real: bool) -> Vec<Complex> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut nx = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        ((x >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * amp
    };
    (0..n).map(|_| Complex::new(nx(), if real { 0.0 } else { nx() })).collect()
}

/// Complex payload with Re, Im ~ N(0, sigma^2) (the sylph PCMM gate's X).
fn rand_gauss(seed: u64, n: usize, sigma: f64) -> Vec<Complex> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut u = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let (a, b) = (u(), u());
            let r = (-2.0 * a.ln()).sqrt() * sigma;
            let th = 2.0 * std::f64::consts::PI * b;
            Complex::new(r * th.cos(), r * th.sin())
        })
        .collect()
}

/// Complex slots of every ciphertext of `x` decoded from the bottom `q.len()` rows (exact centred CRT): the bottom pair
/// alone wraps coefficients beyond q0 / 2 (`Decryptor` built with q.len() rows).
fn decrypt_wide<W: pc_gpu::DevWord, M: pc_gpu::DevStrategy<W>>(dec: &mut Decryptor<'_, W>, ctx: &Context<'_, W, M>, s: &Stream<Compute>, enc: &pc_core::encode::Encoder, x: &Ct<'_, W>, q: &[u64]) -> Res<Vec<Vec<Complex>>> {
    let rows = dec.decrypt_rows(ctx, s, x, Which::Dense)?;
    let big: u128 = q.iter().fold(1u128, |a, &p| a * p as u128);
    let hats: Vec<(u128, u64)> = q.iter().map(|&p| (big / p as u128, pc_core::modmul::invmod(((big / p as u128) % p as u128) as u64, p))).collect();
    let slots = if x.meta.slots == 0 { ctx.lay.n / 2 } else { x.meta.slots as usize };
    let mut out = Vec::new();
    for r in &rows {
        let c: Vec<i128> = (0..r[0].len())
            .map(|j| {
                let mut v: u128 = 0;
                for (i, &(hat, hinv)) in hats.iter().enumerate() {
                    v = (v + pc_core::modmul::mulmod(r[i][j] % q[i], hinv, q[i]) as u128 * hat) % big;
                }
                if v > big / 2 { -((big - v) as i128) } else { v as i128 }
            })
            .collect();
        out.push(enc.decode_centered(&c, slots, Payload::Complex, x.meta.nf)?);
    }
    Ok(out)
}

/// The table of `pc-bench fbs` on j in [-t/2, t/2].
fn fbs_lut(name: &str, t: u32, j: i64) -> f64 {
    let h = (t / 2) as i64;
    match name {
        "sign" => j.signum() as f64,
        "step" => (j >= 0) as u8 as f64,
        "relu" => j.max(0) as f64 / h as f64,
        // a fixed pseudo-random table with values in {-1, ..., 1} on a grid of 1/8 (a "general" LUT)
        "rand" => {
            let mut z = ((j + h) as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            z ^= z >> 29;
            z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 32;
            (z % 17) as f64 / 8.0 - 1.0
        }
        _ => j as f64 / h as f64,
    }
}

/// Chebyshev coefficients (OpenFHE convention: c0 not halved) of the degree-t interpolant of `fbs_lut` on the nodes
/// s_j = sin(pi j / t), j = -t/2..t/2 (the Chebyshev extrema cos(pi k / t), k = t/2 - j), times `gain`.
fn fbs_post(name: &str, t: u32, gain: f64) -> Vec<f64> {
    let n = t as usize;
    let h = (t / 2) as i64;
    let mut c = vec![0f64; n + 1];
    for (deg, ck) in c.iter_mut().enumerate() {
        let mut acc = 0.0;
        for k in 0..=n {
            let w = if k == 0 || k == n { 0.5 } else { 1.0 };
            acc += w * fbs_lut(name, t, h - k as i64) * (std::f64::consts::PI * (deg * k) as f64 / n as f64).cos();
        }
        *ck = 2.0 / n as f64 * acc * gain;
    }
    c[n] *= 0.5;
    c
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

/// One StC stage of FIDESlib's layout on a slot vector (LtStage semantics): sum_i rot_{rot_out\[i\]}(sum_j
/// pts[i b + j] * rot_{rot_in\[j\]}(x)).
fn host_stage(fs: &FidesStage, x: &[C64]) -> Vec<C64> {
    let n = x.len();
    let b = fs.b_step as usize;
    let mut y = vec![C64::ZERO; n];
    for i in 0..fs.g_step as usize {
        let mut acc = vec![C64::ZERO; n];
        for j in 0..b {
            let idx = i * b + j;
            if idx >= fs.num_diags {
                break;
            }
            let xr = pc_core::bts_setup::rotate_left(x, fs.rot_in[j]);
            for k in 0..n {
                acc[k] += fs.pts[idx][k] * xr[k];
            }
        }
        let ar = pc_core::bts_setup::rotate_left(&acc, fs.rot_out[i]);
        for k in 0..n {
            y[k] += ar[k];
        }
    }
    y
}

pub fn bench(args: &[String]) -> Res<bool> {
    let a = bargs(args)?;
    let t0 = Instant::now();
    // PC_BENCH_CHAIN: another chain preset (n32a4: the lean chain's single-prime StC stages)
    let chain = Chain::generate(&ChainSpec::preset(&std::env::var("PC_BENCH_CHAIN").unwrap_or_else(|_| "n32".into()))?)?;
    let n = chain.n;
    let slots = n / 2;
    let kmax = (*a.ks.iter().max().unwrap_or(&1)).max(8);
    let mut setup = BtsSetup::n32_shipped();
    setup.raise_drops = vec![0, 1];
    setup.k_max = kmax;
    // fbs: message j at phase j q0 / 2t (CF log2 2t); EvalMod's output reads x 2^CF, folded into the series
    let fbs_cf = a.fbs_cf.unwrap_or((2 * a.fbs_t).trailing_zeros());
    let mut post_c0 = 0.0;
    if a.cmd == "fbs" {
        if let Some(f) = &a.em_table {
            setup.evalmod_coeffs = std::fs::read_to_string(f)?.split_whitespace().map(|x| x.parse::<f64>()).collect::<std::result::Result<_, _>>()?;
            setup.evalmod_da_its = a.em_da;
            setup.boot_k = a.em_k;
            println!("[pc-bench] fbs EvalMod table {f}: degree {}, K {}, double angles {}", setup.evalmod_coeffs.len() - 1, a.em_k, a.em_da);
        }
        setup.evalmod_amp = a.amp;
        setup.evalmod_post = match &a.post_file {
            Some(f) => {
                // PC_FBS_SPLIT_C0=1: the constant term leaves the series (a plaintext add after the bootstrap,
                // `post_c0`); by default it stays in (exact: the decrypt reads four rows, see `decrypt_wide`)
                let mut v: Vec<f64> = serde_json::from_str(&std::fs::read_to_string(f)?)?;
                if std::env::var("PC_FBS_SPLIT_C0").is_ok_and(|v| v == "1") {
                    post_c0 = v[0] / 2.0;
                    v[0] = 0.0;
                }
                v.iter().map(|c| c * (fbs_cf as f64).exp2().recip()).collect()
            }
            None => fbs_post(&a.lut, a.fbs_t, (fbs_cf as f64).exp2().recip()),
        };
        let c = &setup.evalmod_post;
        println!("[pc-bench] fbs t={} lut={} cf={fbs_cf}: series degree {}, max |c| {:.3}, sum |c| {:.3}", a.fbs_t, a.lut, c.len() - 1, c.iter().fold(0f64, |m, x| m.max(x.abs())) * (fbs_cf as f64).exp2(), c.iter().map(|x| x.abs()).sum::<f64>() * (fbs_cf as f64).exp2());
    }
    // PC_FBS_ON_ROUTE=1: the shipped route plus the series as an FBS of it (`StcFirstPlan::fbs_host`, the planned
    // runtime's path) instead of a route built around the series
    let on_route = a.cmd == "fbs" && std::env::var("PC_FBS_ON_ROUTE").is_ok_and(|v| v == "1");
    if a.post_s_file.is_some() && !on_route {
        return Err("fbs --post-s-file (the trigonometric form) runs on the route: set PC_FBS_ON_ROUTE=1".into());
    }
    let fbs_spec = if on_route {
        let mut sp = pc_gpu::bts::stc_first::FbsSpec { table: setup.evalmod_coeffs.clone(), k: setup.boot_k, da_its: setup.evalmod_da_its, amp: setup.evalmod_amp, post: setup.evalmod_post.clone(), table_s: vec![], off: 0.0, post_s: vec![], paired: std::env::var("PC_FBS_PAIRED").is_ok_and(|v| v == "1"), drop: 0, complex: true, real: true };
        if let Some(f) = &a.post_s_file {
            // the trigonometric form: the K32 base pair of degree --trig-deg, --trig-da paired double angles
            let v: Vec<f64> = serde_json::from_str(&std::fs::read_to_string(f)?)?;
            let (c, s) = pc_shadow::evalmod::trig_base(32.0, a.trig_da, a.trig_deg);
            (sp.table, sp.table_s, sp.k, sp.da_its, sp.off) = (c, s, 32.0, a.trig_da, a.trig_off);
            sp.post_s = v.iter().map(|c| c * (fbs_cf as f64).exp2().recip()).collect();
            println!("[pc-bench] fbs trig: base pair degree {} da {}, Q degree {}, off {}", a.trig_deg, a.trig_da, sp.post_s.len() - 1, a.trig_off);
        }
        let base = BtsSetup::n32_shipped();
        (setup.evalmod_coeffs, setup.boot_k, setup.evalmod_da_its, setup.evalmod_amp, setup.evalmod_post) = (base.evalmod_coeffs, base.boot_k, base.evalmod_da_its, 1.0, vec![]);
        Some(sp)
    } else {
        None
    };
    setup.real = true;
    // PC_EM_FUSE_RESCALE=1: the EvalMod relinearization-rescale fusion (stcf-prec compares the two forms' errors)
    setup.em_fuse_rescale = std::env::var("PC_EM_FUSE_RESCALE").is_ok_and(|v| v == "1");
    let mut sf = StcFirstSetup::n32_shipped();
    sf.cache_slots = 4;
    let high_shapes = vec![StcHighShape { out_level: 50, skip_mask: 1 }, StcHighShape { out_level: 30, skip_mask: 6 }, StcHighShape { out_level: 40, skip_mask: 0 }];
    // PC_BENCH_NO_HIGH=1: no stc_high shapes (their levels are n32 / n32a4 ones; off the grid of an sp chain)
    if !std::env::var("PC_BENCH_NO_HIGH").is_ok_and(|v| v == "1") {
        sf.high = Some(StcHighSetup { shapes: high_shapes.clone(), slots: 1 });
    }
    let plan = StcFirstPlan::new(&chain, &setup, &sf)?;
    let mut geoms: Vec<LtGeom> = Vec::new();
    for v in &plan.dense.variants {
        geoms.extend(v.cts.iter().chain(&v.stc).map(|h| h.geom));
    }
    geoms.extend(plan.input.iter().map(|h| h.geom));
    geoms.extend(plan.sources.iter().flatten().map(|s| s.geom));
    let steps = rot_steps(&geoms, n);
    let spec = KeySetSpec { relin: true, conj: true, rots: steps.iter().map(|&s| RotSpec { step: s, q_top: None }).collect(), encaps: true, lean: None };
    println!("[pc-bench] stcf host plan {:.1} s: entry limb {}, {} input stages, {} rotation keys", t0.elapsed().as_secs_f64(), plan.entry_top, plan.input.len(), steps.len());
    let dev = Device::open()?;
    let cfg = MemCfg { k_max: kmax, pool: vec![PoolClass { k: 1, count: 1 }], events: 64, scratch_cts: 2 };
    let mut mp = MemPlan::build::<W, M>(&chain, cfg)?;
    let lay = mp.lay;
    let shapes: Vec<_> = spec.keys(&chain)?.into_iter().map(|(_, s)| s).collect();
    mp.reserve("keys", spec.bytes::<W>(&chain, KskLayout::Dense)?);
    mp.reserve("keygen", KeyGen::<W>::bytes(&lay, &shapes));
    mp.reserve("secret", DevSecret::<W>::bytes(&lay));
    mp.reserve("encrypt", Encryptor::<W>::bytes(&lay, false));
    // two composite levels of rows: the bottom pair alone (q0 ~ 2^56) wraps a coefficient beyond q0 / 2, i.e. a slot
    // constant beyond ~1.9 at the landing scale (a functional bootstrap's outputs reach it)
    let dec_rows = 2 * chain.level_unit;
    mp.reserve("decrypt", Decryptor::<W>::bytes(&lay, dec_rows, kmax));
    mp.reserve("ks.plans", KsPlans::bytes::<W>(&chain, kmax)?);
    mp.reserve("ks.regen_a", pc_gpu::ks::KsWork::<W>::regen_bytes(&lay, chain.dnum()));
    let ocfg = OpsCfg::default();
    Ops::<W>::reserve(&mut mp, &ocfg);
    let (req, mut wshape) = plan.mem::<W>()?;
    req.reserve(&mut mp);
    let fbs_host = match &fbs_spec {
        Some(sp) => {
            let h = plan.fbs_host(sp)?;
            let (b, w) = plan.fbs_mem::<W>(&h)?;
            mp.reserve("bts.fbs", b);
            wshape = wshape.max(w);
            println!("[pc-bench] fbs on the shipped route: K {} da {} amp {}, series degree {}, EvalMod {:?} -> {:?}", sp.k, sp.da_its, sp.amp, sp.post.len() - 1, h.em_in, h.out);
            Some(h)
        }
        None => None,
    };
    let with_dense = a.cmd == "stcf-prec";
    if with_dense {
        let (dreq, _) = plan.dense.mem::<W>()?;
        mp.reserve("bench.dense", dreq.total());
    }
    mp.reserve("bts.work", BtsWork::<W>::bytes(&lay, wshape));
    mp.reserve("bench.cts", 2 * pc_cuda::Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(&lay, kmax)));
    println!("[pc-bench] MemPlan {}", mp.report());
    let arena = mp.allocate(&dev)?;
    let mut ex = Executor::<W>::new(&dev, &arena, &mp)?;
    ex.ks.carve_regen(&arena, &lay)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &mp)?;
    let kplans = KsPlans::new(&arena, &ex.compute, &ctx, kmax)?;
    let mut ops = Ops::new(&arena, &dev, &ex.compute, &ctx, ocfg)?;
    let mut work = BtsWork::<W>::new(&arena, lay, wshape)?;
    let mut keys = KeySet::<W>::carve(&arena, &chain, &spec, KskLayout::Dense)?;
    let mut prng = test_prng(0x6c);
    let sk = SecretKey::sample(&mut prng, n)?;
    let mut sec = DevSecret::<W>::carve(&arena, lay)?;
    sec.load(&ctx, &ex.compute, &sk)?;
    let mut kg = KeyGen::<W>::new(&arena, &ex.compute, &chain, &shapes)?;
    let tk = kg.generate_set(&ctx, &mut ex, &sec, &mut keys, &mut prng)?;
    println!("[pc-bench] keygen {} keys in {:.1} s", tk.keys, tk.total);
    let mut encr = Encryptor::<W>::new(&arena, &ex.compute, lay, false, &mut prng)?;
    let mut dec = Decryptor::<W>::new(&arena, &ex.compute, lay, dec_rows, kmax)?;
    dec.load_secret(&ex.compute, &sec)?;
    let tb = Instant::now();
    let route = plan.build(&arena, &dev, &ex.compute, &ctx, &keys, ex.ks.stage_ptrs())?;
    let fbs_pre = match &fbs_host {
        Some(h) => Some(route.build_fbs(&arena, &ex.compute, &ctx, &keys, ex.ks.stage_ptrs(), h)?),
        None => None,
    };
    let dense_r = if with_dense { Some(plan.dense.build(&arena, &ex.compute, &ctx, &keys, ex.ks.stage_ptrs())?) } else { None };
    ex.compute.sync()?;
    println!("[pc-bench] stcf route tables {:.1} s; arena used {:.1} of {:.1} GB", tb.elapsed().as_secs_f64(), arena.used_bytes() as f64 / 1e9, arena.capacity() as f64 / 1e9);
    let mut x = Ct::new(CtMeta::new(0, 1), CtBuf::<Compute, W>::new(&arena, "bench.cts", lay, kmax)?);
    let mut y = Ct::new(CtMeta::new(0, 1), CtBuf::<Compute, W>::new(&arena, "bench.cts", lay, kmax)?);
    arena.freeze();
    let sh = Shadow::new(ShadowConfig { chain: ctx.shape, bts: BtsConfig::n32_shipped(), quirks: Quirks::none() }, Some(Arc::new(ctx.scale.clone())));
    let enc = Encoder::new(&chain);
    let nq = ctx.shape.n_q;
    let mut fresh = |x: &mut Ct<'_, W>, ex: &mut Executor<'_, W>, vals: &[Vec<Complex>], level: u32| -> Res<()> {
        let top = pc_shadow::meta::top_limb(ctx.shape, level)? as usize;
        let sf_ = ctx.scale.sf(top as i32)?;
        let mut meta = CtMeta::new(level, 1).with_nf(sf_);
        meta.slots = slots as u32;
        x.buf.k = vals.len();
        let msgs: Vec<Vec<Vec<u64>>> = vals.iter().map(|v| enc.encode_slots(v, slots, top, Payload::Complex, sf_)).collect::<std::result::Result<_, _>>()?;
        encr.encrypt_sk(&ctx, &ex.compute, &sec, &msgs, meta, x, &mut prng)?;
        Ok(())
    };
    let mut t = Tally::default();
    let mut rows = Vec::new();
    match a.cmd.as_str() {
        "stcf" => {
            let ev: Vec<Event> = (0..16).map(|_| Event::timing(&dev)).collect::<std::result::Result<_, _>>()?;
            let cf = a.cfs[0];
            let r = Route { cf: Some(cf), stc_first: true, real: a.real, ..Route::dense() };
            for &k in &a.ks {
                let vals: Vec<Vec<Complex>> = (0..k).map(|i| rand_c(17 + i as u64, slots, 1.0, a.real)).collect();
                let mut dev_ms = Vec::new();
                let mut stage_ms: Vec<(String, f64)> = Vec::new();
                for rep in 0..a.reps + 2 {
                    fresh(&mut x, &mut ex, &vals, a.level)?;
                    let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                    ev[0].record(&cx.ex.compute)?;
                    let mut call = BtsCall::begin(StagedRef::StcFirst(&route), &cx, std::slice::from_ref(&x), &r, CallOpts::default())?;
                    let mut names = Vec::new();
                    while call.phase() != Phase::Done {
                        let ph = call.step(&mut cx, std::slice::from_mut(&mut x))?;
                        names.push(format!("{ph:?}"));
                        ev[names.len()].record(&cx.ex.compute)?;
                    }
                    ev[names.len()].sync()?;
                    if rep >= 2 {
                        dev_ms.push(ev[0].elapsed_ms(&ev[names.len()])? as f64);
                        if rep == a.reps + 1 {
                            stage_ms = names.iter().enumerate().map(|(i, nm)| (nm.clone(), ev[i].elapsed_ms(&ev[i + 1]).unwrap_or(0.0) as f64)).collect();
                        }
                    }
                }
                let d = median(dev_ms);
                println!("[pc-bench] stcf route cf={cf} real={} input level {} K={k}: device {:.2} ms/call {:.2} ms/ct (median of {}); out level {} deg {}", a.real, a.level, d, d / k as f64, a.reps, x.meta.level, x.meta.deg);
                println!("[pc-bench]   stages K={k} (ms): {}", stage_ms.iter().map(|(n, m)| format!("{n} {m:.2}")).collect::<Vec<_>>().join(", "));
                if a.graph {
                    if let Some(c) = &route.last {
                        c.set_locked(true);
                    }
                    fresh(&mut x, &mut ex, &vals, a.level)?;
                    let sp: *const Stream<Compute> = &ex.compute;
                    let xp: *mut Ct<'_, W> = &mut x;
                    let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                    let rr = &route;
                    let (g, _) = capture(sp, || {
                        // SAFETY: x is not otherwise borrowed while the closure runs
                        let x = unsafe { &mut *xp };
                        let mut call = BtsCall::begin(StagedRef::StcFirst(rr), &cx, std::slice::from_ref(x), &r, CallOpts::default())?;
                        call.run_to(&mut cx, std::slice::from_mut(x), Phase::Done)
                    })?;
                    let mut gm = Vec::new();
                    for rep in 0..a.reps + 2 {
                        fresh(&mut x, cx.ex, &vals, a.level)?;
                        ev[0].record(&cx.ex.compute)?;
                        g.launch(&cx.ex.compute)?;
                        ev[1].record(&cx.ex.compute)?;
                        ev[1].sync()?;
                        if rep >= 2 {
                            gm.push(ev[0].elapsed_ms(&ev[1])? as f64);
                        }
                    }
                    if let Some(c) = &route.last {
                        c.set_locked(false);
                    }
                    let gd = median(gm);
                    println!("[pc-bench] stcf K={k} graph replay: {:.2} ms/call {:.2} ms/ct", gd, gd / k as f64);
                }
            }
        }
        "fbs" => {
            let t = a.fbs_t;
            // PC_FBS_OWN_CF=1: the call at the FBS's own CF, as the planned session's FBS service runs it (pc-py
            // FbsExec: own_cf); default: the route's reference CF with the input pre-scaled
            let fbs_opts = CallOpts { own_cf: std::env::var("PC_FBS_OWN_CF").is_ok_and(|v| v == "1"), ..CallOpts::default() };
            let h = (t / 2) as i64;
            // --real: the real-payload route (N/2 real inputs per ciphertext, Im = 0)
            let real = a.real;
            let r = Route { cf: Some(fbs_cf), stc_first: true, real, ..Route::dense() };
            let ev: Vec<Event> = (0..2).map(|_| Event::timing(&dev)).collect::<std::result::Result<_, _>>()?;
            let mut x_rng = 0x5eed_u64 ^ ((t as u64) << 32);
            let mut next = move || {
                x_rng ^= x_rng << 13;
                x_rng ^= x_rng >> 7;
                x_rng ^= x_rng << 17;
                x_rng
            };
            // integer inputs, Re and Im both carry a j (2 N/2 table inputs per ciphertext); --frac: j + u, u uniform
            // in [-1/2, 1/2) (the table's transition between nodes)
            let (smooth, range, pair) = (a.post_file.is_some(), a.range, a.pair);
            let samples: Vec<f64> = match &a.samples {
                Some(f) => std::fs::read(f)?.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64).collect(),
                None => vec![],
            };
            let mut draw = |k: usize, frac: bool| -> Vec<Vec<Complex>> {
                (0..k)
                    .map(|_| {
                        (0..slots)
                            .map(|_| {
                                let mut v = [0f64; 2];
                                if smooth && pair > 0.0 {
                                    let r = next();
                                    let x = if !samples.is_empty() && r & 1 == 0 {
                                        samples[(r >> 1) as usize % samples.len()].clamp(-range, range)
                                    } else {
                                        ((next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * range
                                    };
                                    return Complex::new(x - pair, x + pair);
                                }
                                for z in v.iter_mut() {
                                    if smooth {
                                        let r = next();
                                        *z = if !samples.is_empty() && r & 1 == 0 {
                                            samples[(r >> 1) as usize % samples.len()].clamp(-range, range)
                                        } else {
                                            ((next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * range
                                        };
                                        continue;
                                    }
                                    let j = (next() % (t as u64 + 1)) as i64 - h;
                                    let u = if frac { (next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5 } else { 0.0 };
                                    *z = (j as f64 + u).clamp(-(h as f64), h as f64);
                                }
                                Complex::new(v[0], if real { 0.0 } else { v[1] })
                            })
                            .collect()
                    })
                    .collect()
            };
            let mut landing = (0, 0);
            let mut dumpf = match &a.dump {
                Some(p) => Some(std::io::BufWriter::new(std::fs::File::create(p)?)),
                None => None,
            };
            for &k in &a.ks {
                let vals = draw(k, false);
                let mut ms = Vec::new();
                for rep in 0..a.reps + 2 {
                    fresh(&mut x, &mut ex, &vals, a.level)?;
                    let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                    ev[0].record(&cx.ex.compute)?;
                    let mut call = BtsCall::begin(StagedRef::StcFirst(&route), &cx, std::slice::from_ref(&x), &r, fbs_opts)?;
                    match &fbs_pre {
                    Some(f) => {
                        call.stc_first_input(&mut cx, std::slice::from_mut(&mut x))?;
                        route.run_fbs(&mut cx, &call.st, std::slice::from_mut(&mut x), f)?;
                    }
                    None => call.run_to(&mut cx, std::slice::from_mut(&mut x), Phase::Done)?,
                }
                    ev[1].record(&cx.ex.compute)?;
                    ev[1].sync()?;
                    if rep >= 2 {
                        ms.push(ev[0].elapsed_ms(&ev[1])? as f64);
                    }
                }
                let d = median(ms);
                landing = (x.meta.level, x.meta.deg);
                let outs = if real { slots } else { 2 * slots };
                println!("[pc-bench] fbs t={t} lut={} real={real} K={k}: device {d:.2} ms/call {:.2} ms/ct {:.1} ns/output ({outs} outputs/ct); input level {} -> landing level {} deg {}", a.lut, d / k as f64, d / k as f64 / outs as f64 * 1e6, a.level, x.meta.level, x.meta.deg);
            }
            // accuracy: a.reps batches of kmax ciphertexts
            let (mut n_out, mut wrong, mut sum2, mut emax) = (0usize, 0usize, 0f64, 0f64);
            let gap = match a.lut.as_str() {
                "sign" | "step" => 1.0,
                "rand" => 0.125,
                _ => 1.0 / h as f64,
            };
            let mut hist = vec![0usize; 40];
            let mut tr: BTreeMap<i64, (f64, f64, usize)> = BTreeMap::new();
            for _ in 0..a.reps {
                let vals = draw(kmax, a.frac);
                fresh(&mut x, &mut ex, &vals, a.level)?;
                let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                let mut call = BtsCall::begin(StagedRef::StcFirst(&route), &cx, std::slice::from_ref(&x), &r, fbs_opts)?;
                match &fbs_pre {
                    Some(f) => {
                        call.stc_first_input(&mut cx, std::slice::from_mut(&mut x))?;
                        route.run_fbs(&mut cx, &call.st, std::slice::from_mut(&mut x), f)?;
                    }
                    None => call.run_to(&mut cx, std::slice::from_mut(&mut x), Phase::Done)?,
                }
                cx.ops.realize(&ctx, cx.ex, &sh, std::slice::from_mut(&mut x))?;
                let got = decrypt_wide(&mut dec, &ctx, &ex.compute, &enc, &x, &chain.q[..dec_rows])?;
                for (g, v) in got.iter().zip(&vals) {
                    for (gz, vz) in g.iter().zip(v) {
                        for (gv, xv) in [(gz.re, vz.re), (gz.im, vz.im)].into_iter().take(if real { 1 } else { 2 }) {
                            if let Some(w) = dumpf.as_mut() {
                                use std::io::Write;
                                w.write_all(&xv.to_le_bytes())?;
                                w.write_all(&(gv + post_c0).to_le_bytes())?;
                            }
                            if smooth {
                                continue;
                            }
                            if a.frac {
                                // transition profile: output by input bucket of 1/16 around the jump (|x| < 2)
                                if xv.abs() < 2.0 {
                                    let b = (xv * 16.0).floor() as i64;
                                    let e = tr.entry(b).or_insert((f64::MAX, f64::MIN, 0));
                                    e.0 = e.0.min(gv);
                                    e.1 = e.1.max(gv);
                                    e.2 += 1;
                                }
                                continue;
                            }
                            let want = fbs_lut(&a.lut, t, xv.round() as i64);
                            let e = (gv - want).abs();
                            n_out += 1;
                            sum2 += e * e;
                            emax = emax.max(e);
                            wrong += (e >= gap / 2.0) as usize;
                            let bk = if e > 0.0 { (-e.log2()).floor().clamp(0.0, 39.0) as usize } else { 39 };
                            hist[bk] += 1;
                        }
                    }
                }
            }
            if smooth {
                println!("[pc-bench] fbs smooth series: outputs dumped (landing level {} deg {})", landing.0, landing.1);
            } else if a.frac {
                for (b, (lo, hi, n)) in &tr {
                    println!("[pc-bench] fbs frac x in [{:.4}, {:.4}): out min {lo:.4} max {hi:.4} ({n})", *b as f64 / 16.0, (*b + 1) as f64 / 16.0);
                }
            } else {
                let rms = (sum2 / n_out as f64).sqrt();
                println!(
                    "[pc-bench] fbs t={t} lut={} accuracy: {n_out} outputs, wrong (err >= gap/2 = {}) {wrong} ({:.3e}), max err {emax:.3e} ({:.2} bits), rms {rms:.3e} ({:.2} bits); landing level {} deg {}",
                    a.lut,
                    gap / 2.0,
                    wrong as f64 / n_out as f64,
                    -emax.log2(),
                    -rms.log2(),
                    landing.0,
                    landing.1
                );
                let hs: Vec<String> = hist.iter().enumerate().filter(|(_, &c)| c > 0).map(|(b, c)| format!("2^-{b}:{c}")).collect();
                println!("[pc-bench] fbs err histogram (floor(-log2 err)): {}", hs.join(" "));
            }
        }
        "stcf-prec" => {
            for &cf in &a.cfs {
                for own in [false, true] {
                    for real in if a.real { vec![false, true] } else { vec![false] } {
                        let r = Route { cf: Some(cf), stc_first: true, real, ..Route::dense() };
                        let vals: Vec<Vec<Complex>> = a.amps.iter().map(|&amp| rand_c(7777, slots, amp, real)).collect();
                        for chunk in vals.chunks(kmax).zip(a.amps.chunks(kmax)) {
                            fresh(&mut x, &mut ex, chunk.0, a.level)?;
                            let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                            let land = route.landing(&sh, &x.meta, &r)?;
                            let mut call = BtsCall::begin(StagedRef::StcFirst(&route), &cx, std::slice::from_ref(&x), &r, CallOpts { prescaled: false, own_cf: own })?;
                            let info = call.stc_first_input(&mut cx, std::slice::from_mut(&mut x))?.ok_or("not StC-first")?;
                            call.run_to(&mut cx, std::slice::from_mut(&mut x), Phase::Done)?;
                            t.add(&format!("cf={cf} own_cf={own} real={real}: output metadata == route landing"), (x.meta.level != land.level || x.meta.deg != land.deg || x.meta.nf.to_bits() != land.nf.to_bits()) as usize, 1);
                            cx.ops.realize(&ctx, cx.ex, &sh, std::slice::from_mut(&mut x))?;
                            let want = BtsConfig::n32_shipped().landing(ctx.shape, &Quirks::none(), &r, a.level, 1)?;
                            t.add(&format!("cf={cf} own_cf={own} real={real}: realized landing == pc-shadow"), (x.meta.level != want.level || x.meta.deg != want.deg) as usize, 1);
                            let got = dec.decrypt_slots(&ctx, &ex.compute, &enc, &x, Payload::Complex, Which::Dense)?;
                            for (j, &amp) in chunk.1.iter().enumerate() {
                                let e = got[j].iter().zip(&chunk.0[j]).map(|(g, v)| (g.re - v.re).abs()).fold(0f64, f64::max) / amp;
                                let ec = got[j].iter().zip(&chunk.0[j]).map(|(g, v)| (*g - *v).abs()).fold(0f64, f64::max) / amp;
                                // RMS of the complex error over the slots (the max moves ~5 % between noise draws)
                                let rms = (got[j].iter().zip(&chunk.0[j]).map(|(g, v)| (*g - *v).abs().powi(2)).sum::<f64>() / got[j].len() as f64).sqrt() / amp;
                                println!(
                                    "[pc-bench] stcf-prec cf={cf} own_cf={own} real={real} cf_stage={} input level {} A={amp}: Re {e:.3e} ({:.2} bits), complex {ec:.3e} ({:.2} bits), rms {rms:.4e}; landing {} deg {}",
                                    info.cf_stage,
                                    a.level,
                                    -e.log2(),
                                    -ec.log2(),
                                    x.meta.level,
                                    x.meta.deg
                                );
                                rows.push(serde_json::json!({"route": "stc_first", "real": real, "input_level": a.level, "iterations": 1, "cf": cf, "own_cf": own, "cf_stage": info.cf_stage, "period": 0, "amp": amp,
                                    "rel_err": e, "bits": -e.log2(), "rel_err_complex": ec, "bits_complex": -ec.log2(), "rms": rms, "landing_level": x.meta.level, "landing_deg": x.meta.deg}));
                            }
                        }
                    }
                }
            }
            // the sylph gate's methodology: complex Gaussian payload, bits = log2(max |X| / max(|dRe|, |dIm|))
            if let Some(sig) = a.gauss {
                for &cf in &a.cfs {
                    for own in [false, true] {
                        let r = Route { cf: Some(cf), stc_first: true, ..Route::dense() };
                        let v = vec![rand_gauss(4242, slots, sig)];
                        fresh(&mut x, &mut ex, &v, a.level)?;
                        let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                        let mut call = BtsCall::begin(StagedRef::StcFirst(&route), &cx, std::slice::from_ref(&x), &r, CallOpts { prescaled: false, own_cf: own })?;
                        call.run_to(&mut cx, std::slice::from_mut(&mut x), Phase::Done)?;
                        cx.ops.realize(&ctx, cx.ex, &sh, std::slice::from_mut(&mut x))?;
                        let got = dec.decrypt_slots(&ctx, &ex.compute, &enc, &x, Payload::Complex, Which::Dense)?;
                        let mx = v[0].iter().fold(0f64, |m, z| m.max(z.re.abs()).max(z.im.abs()));
                        let e = got[0].iter().zip(&v[0]).fold(0f64, |m, (g, z)| m.max((g.re - z.re).abs()).max((g.im - z.im).abs()));
                        println!("[pc-bench] stcf-prec gauss sigma={sig} cf={cf} own_cf={own}: max |X| {mx:.3}, max err {e:.3e} ({:.2} bits = log2(max|X| / err); old sylph plain StC-first bootstrap 11.65, PCMM finish 11.38)", (mx / e).log2());
                        rows.push(serde_json::json!({"route": "stc_first", "real": false, "input_level": a.level, "iterations": 1, "cf": cf, "own_cf": own, "payload": "gauss", "sigma": sig, "amp": mx, "max_err": e, "bits_maxx": (mx / e).log2()}));
                    }
                }
            }
            let dense = dense_r.as_ref().ok_or("dense route")?;
            // seam contract (BtsRoute): the shipped order through this route (inputs deeper than the entry, or
            // stc_first off) == DenseRoute on the same input, every word; and the precision of both orders
            {
                let cf = a.cfs[0];
                let v = vec![rand_gauss(31, slots, 0.1)];
                // pc-shadow: the shipped-order dense landing is `dense_shipped` when landings[0] is the StC-first one
                let mut bc = BtsConfig::n32_shipped();
                bc.dense_shipped = bc.stc_first_fallback;
                for (lvl, stcf) in [(a.level, true), (a.level, false), (48u32, true)] {
                    let r = Route { cf: Some(cf), stc_first: stcf, ..Route::dense() };
                    fresh(&mut x, &mut ex, &v, lvl)?;
                    let top_in = (nq - 1 - lvl) as usize;
                    let ids: Vec<usize> = (0..=top_in).collect();
                    let rows_in: Vec<Vec<u32>> = (0..2).map(|p| x.buf.download_rows(&ex.compute, 0, p, &ids)).collect::<std::result::Result<_, _>>()?;
                    let meta_in = x.meta.clone();
                    let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                    let mut call = BtsCall::begin(StagedRef::StcFirst(&route), &cx, std::slice::from_ref(&x), &r, CallOpts::default())?;
                    let order = call.stc_first();
                    call.run_to(&mut cx, std::slice::from_mut(&mut x), Phase::Done)?;
                    if !order {
                        // the same input through the dense route
                        y.buf.k = 1;
                        for (p, rw) in rows_in.iter().enumerate() {
                            y.buf.upload_rows(&cx.ex.compute, 0, p, &ids, rw)?;
                        }
                        y.meta = meta_in.clone();
                        dense.run(&mut cx, std::slice::from_mut(&mut y), &Route { stc_first: false, ..r })?;
                        let ot: Vec<usize> = (0..=(nq - 1 - x.meta.level) as usize).collect();
                        let mut bad = (x.meta.level != y.meta.level || x.meta.deg != y.meta.deg || x.meta.nf.to_bits() != y.meta.nf.to_bits()) as usize;
                        let mut tot = 1;
                        for p in 0..2 {
                            let gx = x.buf.download_rows(&cx.ex.compute, 0, p, &ot)?;
                            let gy = y.buf.download_rows(&cx.ex.compute, 0, p, &ot)?;
                            bad += gx.iter().zip(&gy).filter(|(a, b)| a != b).count();
                            tot += gx.len();
                        }
                        t.add(&format!("contract: input level {lvl} stc_first={stcf}: this route's shipped order == DenseRoute (words + metadata)"), bad, tot);
                    }
                    cx.ops.realize(&ctx, cx.ex, &sh, std::slice::from_mut(&mut x))?;
                    let want = bc.landing(ctx.shape, &Quirks::none(), &r, lvl, 1)?;
                    t.add(&format!("contract: input level {lvl} stc_first={stcf} (ran StC-first: {order}): realized landing == pc-shadow"), (x.meta.level != want.level) as usize, 1);
                    let got = dec.decrypt_slots(&ctx, &ex.compute, &enc, &x, Payload::Complex, Which::Dense)?;
                    let mx = v[0].iter().fold(0f64, |m, z| m.max(z.re.abs()).max(z.im.abs()));
                    let e = got[0].iter().zip(&v[0]).fold(0f64, |m, (g, z)| m.max((g.re - z.re).abs()).max((g.im - z.im).abs()));
                    println!("[pc-bench] contract cf={cf} input level {lvl} stc_first={stcf}: ran StC-first {order}, landing {} deg {}, {:.2} bits (gauss 0.1)", x.meta.level, x.meta.deg, (mx / e).log2());
                }
            }
            // K = 8 batch of distinct ciphertexts (the replay's refresh units) == each refreshed at K = 1, and the PCMM
            // finish pattern under raise drop 1
            let cf = a.cfs[0];
            let r = Route { cf: Some(cf), stc_first: true, ..Route::dense() };
            let v1 = vec![rand_c(99, slots, 1.0, false)];
            let top_in = (nq - 1 - a.level) as usize;
            let ids: Vec<usize> = (0..=top_in).collect();
            let mut want: Vec<Vec<Vec<u32>>> = Vec::with_capacity(kmax);
            for c in 0..kmax {
                let vc = if c == 0 { v1.clone() } else { vec![rand_c(99 + c as u64, slots, 1.0, false)] };
                fresh(&mut y, &mut ex, &vc, a.level)?;
                x.buf.k = kmax;
                for p in 0..2 {
                    let rw = y.buf.download_rows(&ex.compute, 0, p, &ids)?;
                    x.buf.upload_rows(&ex.compute, c, p, &ids, &rw)?;
                }
                x.meta = y.meta.clone();
                let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                let mut call = BtsCall::begin(StagedRef::StcFirst(&route), &cx, std::slice::from_ref(&y), &r, CallOpts::default())?;
                call.run_to(&mut cx, std::slice::from_mut(&mut y), Phase::Done)?;
                let ot: Vec<usize> = (0..=(nq - 1 - y.meta.level) as usize).collect();
                want.push((0..2).map(|p| y.buf.download_rows(&ex.compute, 0, p, &ot)).collect::<std::result::Result<_, _>>()?);
            }
            {
                let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                let mut call = BtsCall::begin(StagedRef::StcFirst(&route), &cx, std::slice::from_ref(&x), &r, CallOpts::default())?;
                call.run_to(&mut cx, std::slice::from_mut(&mut x), Phase::Done)?;
            }
            let ot: Vec<usize> = (0..=(nq - 1 - y.meta.level) as usize).collect();
            let mut bad = (x.meta.level != y.meta.level || x.meta.deg != y.meta.deg) as usize;
            let mut tot = 1;
            for (c, wc) in want.iter().enumerate() {
                for (p, wnt) in wc.iter().enumerate() {
                    let g = x.buf.download_rows(&ex.compute, c, p, &ot)?;
                    bad += g.iter().zip(wnt).filter(|(a, b)| a != b).count();
                    tot += g.len();
                }
            }
            t.add(&format!("K={kmax} StC-first batch of {kmax} distinct ciphertexts == each at K=1, every word"), bad, tot);
            // PCMM pattern: input state at drop 0, own CF, split, finish at drop 1 (the site's raise variant)
            fresh(&mut x, &mut ex, &v1, a.level)?;
            {
                let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                let mut inp = BtsCall::begin(StagedRef::StcFirst(&route), &cx, std::slice::from_ref(&x), &r, CallOpts { prescaled: false, own_cf: true })?;
                inp.stc_first_input(&mut cx, std::slice::from_mut(&mut x))?;
                let mut fin = inp.finish_call(1)?;
                fin.at_split(&cx, std::slice::from_ref(&x))?;
                fin.run_to(&mut cx, std::slice::from_mut(&mut x), Phase::Done)?;
                let mut log = BtsLog::default();
                pc_gpu::bts::staged::record_stc_first_inputs(&mut log, &route, &cx, &[("x", a.level)])?;
                log.refreshes.push(fin.external(&[("x", a.level)], &x.meta, "bench.pcmm", None));
                println!("[pc-bench] PCMM pattern hooks: {:?}", log);
                cx.ops.realize(&ctx, cx.ex, &sh, std::slice::from_mut(&mut x))?;
            }
            let want = BtsConfig::n32_shipped().landing(ctx.shape, &Quirks::none(), &Route { raise_drop: 1, ..r }, a.level, 1)?;
            t.add("PCMM pattern (finish at raise drop 1): realized landing == pc-shadow (drop 1)", (x.meta.level != want.level) as usize, 1);
            let got = dec.decrypt_slots(&ctx, &ex.compute, &enc, &x, Payload::Complex, Which::Dense)?;
            let ec = got[0].iter().zip(&v1[0]).map(|(g, v)| (*g - *v).abs()).fold(0f64, f64::max);
            println!("[pc-bench] PCMM pattern (own CF {cf}, finish at raise drop 1): complex {ec:.3e} ({:.2} bits), landing {} deg {}", -ec.log2(), x.meta.level, x.meta.deg);
            if let Some(c) = &route.last {
                println!("[pc-bench] last-stage cache: {:?}", c.stats());
            }
        }
        "stcf-pair" => {
            // two real refreshes as one complex refresh (`run_pair`): each member against its own K = 1 real refresh
            // (same CF, amplitude and payload), its landing, graph == eager, a K = 2 pair == two K = 1 pairs, timing
            let rr = |cf: u32| Route { cf: Some(cf), stc_first: true, real: true, ..Route::dense() };
            let vec_of = |m: usize, s: usize, role: u64, amp: f64| rand_c(1000 * m as u64 + 10 * s as u64 + role + 1, slots, amp, true);
            let bits = |got: &[Complex], v: &[Complex], amp: f64| -> (f64, f64) {
                let (mut mx, mut ss) = (0f64, 0f64);
                for (g, w) in got.iter().zip(v) {
                    let d = (g.re - w.re).abs();
                    mx = mx.max(d);
                    ss += d * d;
                }
                (-(mx / amp).log2(), -((ss / v.len() as f64).sqrt() / amp).log2())
            };
            let meq = |a: &CtMeta, b: &CtMeta| a.level == b.level && a.deg == b.deg && a.nf.to_bits() == b.nf.to_bits() && a.slots == b.slots;
            let nm = a.members.len();
            let mut single: BTreeMap<(usize, usize, u64), (f64, f64)> = BTreeMap::new();
            for (m, &(cf, amp)) in a.members.iter().enumerate() {
                for s in 0..a.seeds {
                    for role in 0..2u64 {
                        let v = vec![vec_of(m, s, role, amp)];
                        fresh(&mut x, &mut ex, &v, a.level)?;
                        {
                            let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                            route.run(&mut cx, std::slice::from_mut(&mut x), &rr(cf))?;
                            cx.ops.realize(&ctx, cx.ex, &sh, std::slice::from_mut(&mut x))?;
                        }
                        let got = dec.decrypt_slots(&ctx, &ex.compute, &enc, &x, Payload::Complex, Which::Dense)?;
                        single.insert((m, s, role), bits(&got[0], &v[0], amp));
                    }
                }
            }
            // pair - single per member and role (max-error bits, rms bits); the dirty runs against the clean pair
            let mut delta: BTreeMap<(usize, u64), Vec<(f64, f64)>> = BTreeMap::new();
            for mi in 0..nm {
                for mj in 0..nm {
                    let ((cfx, ax), (cfy, ay)) = (a.members[mi], a.members[mj]);
                    let (rx, ry) = (rr(cfx), rr(cfy));
                    let mut clean0 = (0.0, 0.0);
                    for s in 0..=a.seeds {
                        // the last run: seed 0 with the partner's imaginary lane at amplitude A 2^-dirty
                        let dirty = s == a.seeds;
                        let s0 = if dirty { 0 } else { s };
                        let vx = vec![vec_of(mi, s0, 0, ax)];
                        let mut vy = vec![vec_of(mj, s0, 1, ay)];
                        if dirty {
                            let im = rand_c(77, slots, ay * (-(a.dirty as f64)).exp2(), false);
                            for (z, w) in vy[0].iter_mut().zip(&im) {
                                z.im = w.im;
                            }
                        }
                        fresh(&mut x, &mut ex, &vx, a.level)?;
                        fresh(&mut y, &mut ex, &vy, a.level)?;
                        let (inx, iny) = (x.meta, y.meta);
                        {
                            let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                            route.run_pair(&mut cx, &mut x, &mut y, &rx, &ry)?;
                            if s == 0 {
                                let (lx, ly) = (route.landing(&sh, &inx, &rx)?, route.landing(&sh, &iny, &ry)?);
                                t.add(&format!("pair {cfx}:{ax} + {cfy}:{ay}: each output's metadata == its own landing"), (!meq(&x.meta, &lx)) as usize + (!meq(&y.meta, &ly)) as usize, 2);
                            }
                            cx.ops.realize(&ctx, cx.ex, &sh, std::slice::from_mut(&mut x))?;
                            cx.ops.realize(&ctx, cx.ex, &sh, std::slice::from_mut(&mut y))?;
                        }
                        if s == 0 {
                            let want = BtsConfig::n32_shipped().landing(ctx.shape, &Quirks::none(), &rx, a.level, 1)?;
                            t.add(&format!("pair {cfx}:{ax} + {cfy}:{ay}: realized landings == pc-shadow"), (x.meta.level != want.level) as usize + (y.meta.level != want.level) as usize, 2);
                        }
                        let gx = dec.decrypt_slots(&ctx, &ex.compute, &enc, &x, Payload::Complex, Which::Dense)?;
                        let gy = dec.decrypt_slots(&ctx, &ex.compute, &enc, &y, Payload::Complex, Which::Dense)?;
                        let (bx, by) = (bits(&gx[0], &vx[0], ax), bits(&gy[0], &vy[0], ay));
                        let (sx, sy) = (single[&(mi, s0, 0)], single[&(mj, s0, 1)]);
                        if dirty {
                            println!(
                                "[pc-bench] stcf-pair dirty partner (Im lane A 2^-{}): x {cfx}:{ax} {:.2} bits (clean pair {:.2}, single {:.2}), y {cfy}:{ay} {:.2} bits (single {:.2})",
                                a.dirty, bx.0, clean0.0, sx.0, by.0, sy.0
                            );
                        } else {
                            if s == 0 {
                                clean0 = bx;
                            }
                            delta.entry((mi, 0)).or_default().push((bx.0 - sx.0, bx.1 - sx.1));
                            delta.entry((mj, 1)).or_default().push((by.0 - sy.0, by.1 - sy.1));
                            println!(
                                "[pc-bench] stcf-pair seed {s} x {cfx}:{ax} y {cfy}:{ay}: x {:.2} bits (rms {:.2}) vs single {:.2} ({:.2}); y {:.2} ({:.2}) vs single {:.2} ({:.2})",
                                bx.0, bx.1, sx.0, sx.1, by.0, by.1, sy.0, sy.1
                            );
                        }
                        rows.push(serde_json::json!({"route": "stc_first_pair", "input_level": a.level, "seed": s0, "dirty": if dirty { Some(a.dirty) } else { None },
                            "x": {"cf": cfx, "amp": ax, "bits": bx.0, "bits_rms": bx.1, "single_bits": sx.0, "single_bits_rms": sx.1},
                            "y": {"cf": cfy, "amp": ay, "bits": by.0, "bits_rms": by.1, "single_bits": sy.0, "single_bits_rms": sy.1}}));
                    }
                }
            }
            for (&(m, role), d) in &delta {
                let n = d.len() as f64;
                let (dm, dr) = (d.iter().map(|v| v.0).sum::<f64>() / n, d.iter().map(|v| v.1).sum::<f64>() / n);
                let worst = d.iter().map(|v| v.0).fold(f64::INFINITY, f64::min);
                let (cf, amp) = a.members[m];
                println!("[pc-bench] stcf-pair member cf {cf} A {amp} as {}: pair - single over {} runs: max-error bits {dm:+.3} (worst {worst:+.3}), rms bits {dr:+.3}", ["x", "y"][role as usize], d.len());
                t.add(&format!("precision gate cf {cf} A {amp} as {}: mean pair - single >= -0.15 bits (max error) and >= -0.1 (rms)", ["x", "y"][role as usize]), (dm < -0.15 || dr < -0.1) as usize, 1);
            }
            // y's scale off x's by eps inside one cache key (siblings whose scales took different paths): the pair and y's
            // single both go through the entry of that key (the factor of whichever input filled it); y against its single
            for mi in 0..nm {
                for mj in 0..nm {
                    let ((cfx, ax), (cfy, ay)) = (a.members[mi], a.members[mj]);
                    let (rx, ry) = (rr(cfx), rr(cfy));
                    let (vx, vy) = (vec![vec_of(mi, 0, 0, ax)], vec![vec_of(mj, 0, 1, ay)]);
                    fresh(&mut y, &mut ex, &vy, a.level)?;
                    let nf0 = y.meta.nf;
                    let eps = [3e-7, -3e-7].into_iter().find(|e| nf_key(nf0 * (1.0 + e)) == nf_key(nf0)).ok_or("no scale offset inside the key")?;
                    let wy: Vec<Complex> = vy[0].iter().map(|z| *z / (1.0 + eps)).collect();
                    y.meta.nf = nf0 * (1.0 + eps);
                    {
                        let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                        route.run(&mut cx, std::slice::from_mut(&mut y), &ry)?;
                        cx.ops.realize(&ctx, cx.ex, &sh, std::slice::from_mut(&mut y))?;
                    }
                    let gs = dec.decrypt_slots(&ctx, &ex.compute, &enc, &y, Payload::Complex, Which::Dense)?;
                    let sy = bits(&gs[0], &wy, ay);
                    fresh(&mut x, &mut ex, &vx, a.level)?;
                    fresh(&mut y, &mut ex, &vy, a.level)?;
                    y.meta.nf = nf0 * (1.0 + eps);
                    {
                        let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                        route.run_pair(&mut cx, &mut x, &mut y, &rx, &ry)?;
                        cx.ops.realize(&ctx, cx.ex, &sh, std::slice::from_mut(&mut x))?;
                        cx.ops.realize(&ctx, cx.ex, &sh, std::slice::from_mut(&mut y))?;
                    }
                    let gx = dec.decrypt_slots(&ctx, &ex.compute, &enc, &x, Payload::Complex, Which::Dense)?;
                    let gy = dec.decrypt_slots(&ctx, &ex.compute, &enc, &y, Payload::Complex, Which::Dense)?;
                    let (bx, by) = (bits(&gx[0], &vx[0], ax), bits(&gy[0], &wy, ay));
                    println!(
                        "[pc-bench] stcf-pair scales {eps:+.0e} apart: x {cfx}:{ax} {:.2} bits (single {:.2}), y {cfy}:{ay} {:.2} bits (rms {:.2}) vs its single {:.2} ({:.2})",
                        bx.0, single[&(mi, 0, 0)].0, by.0, by.1, sy.0, sy.1
                    );
                    t.add(&format!("pair {cfx}:{ax} + {cfy}:{ay}, scales {eps:+.0e} apart: y within 0.15 bits of its single"), (by.0 < sy.0 - 0.15) as usize, 1);
                    rows.push(serde_json::json!({"route": "stc_first_pair_scales", "eps": eps, "x": {"cf": cfx, "amp": ax, "bits": bx.0},
                        "y": {"cf": cfy, "amp": ay, "bits": by.0, "bits_rms": by.1, "single_bits": sy.0, "single_bits_rms": sy.1}}));
                }
            }
            // degree-2 inputs (a rescale pending, the pair's rescale at the entry): fresh at level - 2, times 1.0 (degree 2,
            // level - 2, effective level `level`); each member against its single refresh of the same ciphertext
            let lv2 = a.level - 2;
            for mi in 0..nm {
                for mj in 0..nm {
                    let ((cfx, ax), (cfy, ay)) = (a.members[mi], a.members[mj]);
                    let (rx, ry) = (rr(cfx), rr(cfy));
                    let (vx, vy) = (vec![vec_of(mi, 1, 0, ax)], vec![vec_of(mj, 1, 1, ay)]);
                    let mut sb = [0f64; 2];
                    for (role, (v, r, amp)) in [(&vx, rx, ax), (&vy, ry, ay)].into_iter().enumerate() {
                        fresh(&mut x, &mut ex, v, lv2)?;
                        let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                        cx.ops.mult_scalar_inplace(&ctx, cx.ex, &sh, std::slice::from_mut(&mut x), &[Complex::new(1.0, 0.0)])?;
                        route.run(&mut cx, std::slice::from_mut(&mut x), &r)?;
                        cx.ops.realize(&ctx, cx.ex, &sh, std::slice::from_mut(&mut x))?;
                        drop(cx);
                        let g = dec.decrypt_slots(&ctx, &ex.compute, &enc, &x, Payload::Complex, Which::Dense)?;
                        sb[role] = bits(&g[0], &v[0], amp).0;
                    }
                    fresh(&mut x, &mut ex, &vx, lv2)?;
                    fresh(&mut y, &mut ex, &vy, lv2)?;
                    {
                        let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                        cx.ops.mult_scalar_inplace(&ctx, cx.ex, &sh, std::slice::from_mut(&mut x), &[Complex::new(1.0, 0.0)])?;
                        cx.ops.mult_scalar_inplace(&ctx, cx.ex, &sh, std::slice::from_mut(&mut y), &[Complex::new(1.0, 0.0)])?;
                        let (inx, iny) = (x.meta, y.meta);
                        route.run_pair(&mut cx, &mut x, &mut y, &rx, &ry)?;
                        let (lx, ly) = (route.landing(&sh, &inx, &rx)?, route.landing(&sh, &iny, &ry)?);
                        t.add(&format!("pair {cfx}:{ax} + {cfy}:{ay}, degree-2 inputs at level {lv2}: each output's metadata == its own landing"), (!meq(&x.meta, &lx)) as usize + (!meq(&y.meta, &ly)) as usize, 2);
                        cx.ops.realize(&ctx, cx.ex, &sh, std::slice::from_mut(&mut x))?;
                        cx.ops.realize(&ctx, cx.ex, &sh, std::slice::from_mut(&mut y))?;
                    }
                    let gx = dec.decrypt_slots(&ctx, &ex.compute, &enc, &x, Payload::Complex, Which::Dense)?;
                    let gy = dec.decrypt_slots(&ctx, &ex.compute, &enc, &y, Payload::Complex, Which::Dense)?;
                    let (bx, by) = (bits(&gx[0], &vx[0], ax).0, bits(&gy[0], &vy[0], ay).0);
                    println!("[pc-bench] stcf-pair degree-2 inputs at level {lv2}: x {cfx}:{ax} {bx:.2} bits (single {:.2}), y {cfy}:{ay} {by:.2} bits (single {:.2})", sb[0], sb[1]);
                    t.add(&format!("pair {cfx}:{ax} + {cfy}:{ay}, degree-2 inputs: both members within 0.15 bits of their singles"), (bx < sb[0] - 0.15) as usize + (by < sb[1] - 0.15) as usize, 2);
                }
            }
            // graph replay == eager and a K = 2 pair == two K = 1 pairs, every word
            let ((cfx, ax), (cfy, ay)) = (a.members[0], a.members[nm - 1]);
            let (rx, ry) = (rr(cfx), rr(cfy));
            let top_in = (nq - 1 - a.level) as usize;
            let ids: Vec<usize> = (0..=top_in).collect();
            let mut ins: Vec<[Vec<Vec<u32>>; 2]> = Vec::new();
            for c in 0..2 {
                fresh(&mut x, &mut ex, &[vec_of(0, c, 0, ax)], a.level)?;
                fresh(&mut y, &mut ex, &[vec_of(nm - 1, c, 1, ay)], a.level)?;
                let rx_: Vec<Vec<u32>> = (0..2).map(|p| x.buf.download_rows(&ex.compute, 0, p, &ids)).collect::<std::result::Result<_, _>>()?;
                let ry_: Vec<Vec<u32>> = (0..2).map(|p| y.buf.download_rows(&ex.compute, 0, p, &ids)).collect::<std::result::Result<_, _>>()?;
                ins.push([rx_, ry_]);
            }
            let min = x.meta;
            let load = |z: &mut Ct<'_, W>, s: &Stream<Compute>, role: usize, cs: &[usize]| -> Res<()> {
                z.buf.k = cs.len();
                for (at, &c) in cs.iter().enumerate() {
                    for p in 0..2 {
                        z.buf.upload_rows(s, at, p, &ids, &ins[c][role][p])?;
                    }
                }
                z.meta = min;
                Ok(())
            };
            let mut want: Vec<[Vec<Vec<u32>>; 2]> = Vec::new();
            let mut out_ids: Vec<usize> = Vec::new();
            for c in 0..2 {
                load(&mut x, &ex.compute, 0, &[c])?;
                load(&mut y, &ex.compute, 1, &[c])?;
                let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                route.run_pair(&mut cx, &mut x, &mut y, &rx, &ry)?;
                out_ids = (0..=(nq - 1 - x.meta.level) as usize).collect();
                let wx: Vec<Vec<u32>> = (0..2).map(|p| x.buf.download_rows(&cx.ex.compute, 0, p, &out_ids)).collect::<std::result::Result<_, _>>()?;
                let wy: Vec<Vec<u32>> = (0..2).map(|p| y.buf.download_rows(&cx.ex.compute, 0, p, &out_ids)).collect::<std::result::Result<_, _>>()?;
                want.push([wx, wy]);
            }
            let cmp = |z: &Ct<'_, W>, s: &Stream<Compute>, at: usize, w: &[Vec<u32>]| -> Res<(usize, usize)> {
                let mut bad = 0;
                let mut tot = 0;
                for (p, wp) in w.iter().enumerate() {
                    let g = z.buf.download_rows(s, at, p, &out_ids)?;
                    bad += g.iter().zip(wp).filter(|(a, b)| a != b).count();
                    tot += g.len();
                }
                Ok((bad, tot))
            };
            {
                load(&mut x, &ex.compute, 0, &[0, 1])?;
                load(&mut y, &ex.compute, 1, &[0, 1])?;
                let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                route.run_pair(&mut cx, &mut x, &mut y, &rx, &ry)?;
                let (mut bad, mut tot) = (0, 0);
                for c in 0..2 {
                    for (z, w) in [(&x, &want[c][0]), (&y, &want[c][1])] {
                        let (b, n) = cmp(z, &cx.ex.compute, c, w)?;
                        bad += b;
                        tot += n;
                    }
                }
                t.add("a K = 2 pair == two K = 1 pairs, every word", bad, tot);
            }
            if let Some(c) = &route.last {
                c.set_locked(true);
            }
            let sp: *const Stream<Compute> = &ex.compute;
            let (xp, yp): (*mut Ct<'_, W>, *mut Ct<'_, W>) = (&mut x, &mut y);
            load(&mut x, &ex.compute, 0, &[0])?;
            load(&mut y, &ex.compute, 1, &[0])?;
            {
                let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                let rt = &route;
                let (g, _) = capture(sp, || {
                    // SAFETY: x and y are not otherwise borrowed while the closure runs
                    let (x, y) = unsafe { (&mut *xp, &mut *yp) };
                    rt.run_pair(&mut cx, x, y, &rx, &ry)
                })?;
                // SAFETY: the capture's borrows ended with the closure
                let (x, y) = unsafe { (&mut *xp, &mut *yp) };
                load(x, &cx.ex.compute, 0, &[0])?;
                load(y, &cx.ex.compute, 1, &[0])?;
                g.launch(&cx.ex.compute)?;
                cx.ex.compute.sync()?;
                let (b1, n1) = cmp(x, &cx.ex.compute, 0, &want[0][0])?;
                let (b2, n2) = cmp(y, &cx.ex.compute, 0, &want[0][1])?;
                t.add("pair graph replay == eager, every word", b1 + b2, n1 + n2);
            }
            if let Some(c) = &route.last {
                c.set_locked(false);
            }
            // device ms: two real singles at K = 1, a real K = 2 batch, a complex single, the pair at K = 1 and 2
            let ev: Vec<Event> = (0..2).map(|_| Event::timing(&dev)).collect::<std::result::Result<_, _>>()?;
            let rc = Route { real: false, ..rr(cfx) };
            for (name, k, pair, r) in [("real single", 1, false, rx), ("real K=2", 2, false, rx), ("complex single", 1, false, rc), ("pair", 1, true, rx), ("pair K=2", 2, true, rx)] {
                let mut ms = Vec::new();
                for rep in 0..a.reps + 2 {
                    let vx: Vec<Vec<Complex>> = (0..k).map(|i| vec_of(0, i, 0, ax)).collect();
                    fresh(&mut x, &mut ex, &vx, a.level)?;
                    if pair {
                        let vy: Vec<Vec<Complex>> = (0..k).map(|i| vec_of(0, i, 1, ax)).collect();
                        fresh(&mut y, &mut ex, &vy, a.level)?;
                    }
                    let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                    ev[0].record(&cx.ex.compute)?;
                    if pair {
                        route.run_pair(&mut cx, &mut x, &mut y, &r, &r)?;
                    } else {
                        route.run(&mut cx, std::slice::from_mut(&mut x), &r)?;
                    }
                    ev[1].record(&cx.ex.compute)?;
                    ev[1].sync()?;
                    if rep >= 2 {
                        ms.push(ev[0].elapsed_ms(&ev[1])? as f64);
                    }
                }
                let d = median(ms);
                let per = if pair { 2 * k } else { k };
                println!("[pc-bench] stcf-pair timing {name} (cf {cfx}, input level {}): {d:.2} ms/call, {:.2} ms per refreshed ciphertext (median of {})", a.level, d / per as f64, a.reps);
                rows.push(serde_json::json!({"route": "stc_first_pair_timing", "case": name, "k": k, "cf": cfx, "input_level": a.level, "ms_call": d, "ms_per_ct": d / per as f64}));
            }
        }
        "stc-high" => {
            let st = pc_core::scale::ScaleTables::precompute(&chain)?;
            let f = cts_stc_factors(n, slots as u32, setup.level_budget, setup.dim1, 1.0, pc_gpu::bts::stc::scale_dec(&st))?;
            let flay = f.fides_layout();
            let cases = [(high_shapes[0], 0.5, false), (high_shapes[1], 1.0, false), (high_shapes[2], 0.125, true), (high_shapes[0], 0.5, false), (high_shapes[0], 0.25, false)];
            for (ci, &(shp, factor, use_ref)) in cases.iter().enumerate() {
                let v = vec![rand_c(500 + ci as u64, slots, 1.0, false)];
                fresh(&mut x, &mut ex, &v, 20)?;
                let zin = dec.decrypt_slots(&ctx, &ex.compute, &enc, &x, Payload::Complex, Which::Dense)?;
                let nf_ref = if use_ref { x.meta.nf * 2.0 } else { 0.0 };
                let nf_in = x.meta.nf;
                let spec = StcHigh { slots: slots as u32, out_level: shp.out_level, factor, skip_mask: shp.skip_mask, nf_ref };
                {
                    let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                    route.staged().ok_or("staged")?.stc_high(&mut cx, std::slice::from_mut(&mut x), &spec)?;
                }
                let got = dec.decrypt_slots(&ctx, &ex.compute, &enc, &x, Payload::Complex, Which::Dense)?;
                let mut z: Vec<C64> = zin[0].iter().map(|c| C64::new(c.re, c.im)).collect();
                for (si, fs) in flay.stc.iter().enumerate() {
                    if shp.skip_mask >> si & 1 == 0 {
                        z = host_stage(fs, &z);
                    }
                }
                let fe = if use_ref { factor * nf_ref / nf_in } else { factor };
                let mx = z.iter().fold(0f64, |a, w| a.max(w.abs())) * fe;
                let e = got[0].iter().zip(&z).map(|(g, w)| ((g.re - fe * w.re).powi(2) + (g.im - fe * w.im).powi(2)).sqrt()).fold(0f64, f64::max) / mx;
                println!("[pc-bench] stc-high {shp:?} factor {factor} nf_ref {}: out level {} deg {}; max |out - host StC| / max |ref| = {e:.3e} (2^{:.1}); max |ref| {mx:.3}", if use_ref { "2 nf" } else { "0" }, x.meta.level, x.meta.deg, e.log2());
                t.add(&format!("stc_high {shp:?} factor {factor}: relative error <= 2^-30"), (!(e <= (-30f64).exp2())) as usize, 1);
                t.add(&format!("stc_high {shp:?}: output level"), (x.meta.level != shp.out_level) as usize, 1);
            }
            if let Some(h) = &route.high {
                for (shp, c) in &h.caches {
                    println!("[pc-bench] stc_high cache {shp:?}: {:?}", c.stats());
                }
            }
        }
        c => return Err(format!("unknown command {c} (stcf | stcf-prec | stcf-pair | stc-high)").into()),
    }
    if let (Some(p), false) = (&a.out, rows.is_empty()) {
        let mut doc: serde_json::Value = std::fs::read_to_string(p).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(serde_json::json!({"meta": {}, "rows": []}));
        let arr = doc["rows"].as_array_mut().ok_or("rows")?;
        arr.retain(|r| !(r["route"] == "stc_first" && rows.iter().any(|n| n["cf"] == r["cf"] && n["amp"] == r["amp"] && n["own_cf"] == r["own_cf"] && n["real"] == r["real"] && n["input_level"] == r["input_level"])));
        arr.extend(rows);
        std::fs::write(p, serde_json::to_string_pretty(&doc)?)?;
        println!("[pc-bench] wrote {p}");
    }
    println!("[pc-bench] {}: {} gates, {} failing", a.cmd, t.gates, t.bad);
    Ok(t.bad == 0)
}
