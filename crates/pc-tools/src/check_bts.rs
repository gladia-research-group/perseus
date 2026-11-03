//! `pc-check bts` through pc-gpu's bootstrap: the d3 finish gates (CtS x4, EvalMod double-angle stash,
//! EvalMod + StC-first output, CUDA-graph replay of the whole finish) and the raise gates (d3: MR-atob -> MR-raised
//! -> X-scaled; g0: the dense StC-first bootstrap's raise / CtS / EvalMod / output and the sparse route's raise).
use pc_core::modmul::Barrett64;
use pc_core::params::{Chain, ChainSpec};
use pc_cuda::{Compute, Device, Stream};
use pc_gpu::bts::cts::{LtEnd, LtGeom, LtPts, LtStage, PtSource};
use pc_gpu::bts::evalmod::{em_work, EmMode, EvalModPlan};
use pc_gpu::bts::finish::{pow2_exp, Pow2};
use pc_gpu::bts::{BtsCx, BtsKeyRing, BtsWork, BtsWorkShape};
use pc_gpu::ks::{DevKey, KeyA, KsKeyRef, KsPlans, KskSource};
use pc_gpu::ops::{Ops, OpsCfg};
use pc_gpu::poly::CtBuf;
use pc_gpu::bts::raise::{self, EncapsKeys, RaisePlan, RaiseStop};
use pc_gpu::{Context, Ct, Executor, MemCfg, MemPlan, PoolClass};
use crate::dump::{Dump, HostCt};
use pc_shadow::evalmod::PsPlan;
use pc_shadow::fides::FMeta;
use pc_shadow::{BtsConfig, CtMeta, Quirks, Shadow, ShadowConfig};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

type W = u32;
type M = Barrett64;
type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Mismatch ledger (one line per gate).
#[derive(Default)]
pub struct Tally {
    pub gates: usize,
    pub bad: usize,
}

impl Tally {
    pub fn add(&mut self, what: &str, bad: usize, of: usize) {
        self.gates += 1;
        self.bad += bad;
        println!("[pc-check] {what}: {bad} mismatches of {of}");
    }
}

/// Oracle keys (dumped `b` + KSKA seed, `a` expanded once), by OpenFHE automorphism index.
pub struct OracleKeys<'a> {
    pub rot: BTreeMap<i64, DevKey<'a, W>>,
    pub conj: Option<DevKey<'a, W>>,
    pub relin: Option<DevKey<'a, W>>,
    pub atob: Option<DevKey<'a, W>>,
    pub btoa: Option<DevKey<'a, W>>,
    pub n: usize,
}

fn miss(what: &str) -> pc_gpu::Error {
    pc_gpu::Error::Mem { knob: "oracle keys", value: 0, msg: format!("the dump has no {what} key") }
}

impl BtsKeyRing for OracleKeys<'_> {
    fn rot(&self, step: i64) -> pc_gpu::Result<&dyn KsKeyRef> {
        let k = step.rem_euclid((self.n / 2) as i64);
        self.rot.get(&k).map(|k| k as &dyn KsKeyRef).ok_or_else(|| miss(&format!("rotation {k}")))
    }
    fn conj(&self) -> pc_gpu::Result<&dyn KsKeyRef> {
        self.conj.as_ref().map(|k| k as &dyn KsKeyRef).ok_or_else(|| miss("conjugation"))
    }
    fn relin(&self) -> pc_gpu::Result<&dyn KsKeyRef> {
        self.relin.as_ref().map(|k| k as &dyn KsKeyRef).ok_or_else(|| miss("relinearization"))
    }
    fn dense_to_sparse(&self) -> pc_gpu::Result<&dyn KsKeyRef> {
        self.atob.as_ref().map(|k| k as &dyn KsKeyRef).ok_or_else(|| miss("dense->sparse"))
    }
    fn sparse_to_dense(&self) -> pc_gpu::Result<&dyn KsKeyRef> {
        self.btoa.as_ref().map(|k| k as &dyn KsKeyRef).ok_or_else(|| miss("sparse->dense"))
    }
}

fn seed_of(pj: &Value, name: &str) -> Res<[u32; 8]> {
    let v: Vec<u32> = pj["keys"][name]["a_seed"].as_array().ok_or(format!("keys.{name}.a_seed"))?.iter().filter_map(|x| x.as_u64().map(|v| v as u32)).collect();
    Ok(v.try_into().map_err(|_| "seed must be 8 words")?)
}

/// Load key `name` (CPU-map file `keys/auto<name>` or `keys/mult`) with galois element `g`.
#[allow(clippy::too_many_arguments)]
pub fn load_key<'a>(arena: &'a pc_cuda::Arena, s: &Stream<Compute>, ctx: &Context<'_, W, M>, plans: &KsPlans<'_>, d: &Dump, pj: &Value, name: &str, g: u64) -> Res<DevKey<'a, W>> {
    let file = if name == "mult" { "keys/mult.b.bin".to_string() } else { format!("keys/auto{name}.b.bin") };
    let b = d.u32s(&file);
    let seed = seed_of(pj, name)?;
    Ok(DevKey::new(arena, s, ctx, plans, g, ctx.chain.dnum(), &b, Some(seed), KeyA::Expand)?)
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

fn fmeta(d: &Dump, name: &str) -> FMeta {
    let m = d.json(&format!("{name}.json"));
    FMeta { top: m["level"].as_i64().unwrap_or(-1) as i32, nl: m["noise_level"].as_i64().unwrap_or(1) as i32, nf: m["noise_factor"].as_f64().unwrap_or(0.0) }
}

pub fn ps_plan(pj: &Value) -> PsPlan {
    let ps = &pj["cheb_ps"];
    let f = |k: &str| crate::dump::as_f64v(&ps[k]);
    PsPlan { k: ps["k"].as_u64().unwrap_or(0) as usize, m: ps["m"].as_u64().unwrap_or(0) as usize, divqr_q: f("divqr_q"), divcs_q: f("divcs_q"), s2: f("s2"), coeffs: vec![], even_shift: None }
}

/// A dumped stage's geometry and plaintexts (`<prefix><k>.pt<j>.{q,p}.bin` with `pts[j]` metadata).
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

/// btsBegin's post-raise constant `preF / (K N) * btsPreScale` (composite chain: preF = sf(L - d drop) / q0q1).
pub fn eval_const(ctx: &Context<'_, W, M>, drop: usize, k: f64) -> f64 {
    let d = ctx.chain.level_unit;
    let mut qd = ctx.chain.q[0] as f64;
    for j in 1..d {
        qd *= ctx.chain.q[j] as f64;
    }
    let pre_f = ctx.scale.sf((ctx.chain.q.len() - 1 - d * drop) as i32).unwrap_or(0.0) / qd;
    pre_f * (1.0 / (k * ctx.chain.n as f64)) * 1.0
}

/// Raw stream access for graph capture while the bootstrap borrows the executor mutably: the capture only reads the
/// stream handle; every launch inside goes to the same stream.
fn capture<R>(s: *const Stream<Compute>, f: impl FnOnce() -> pc_gpu::Result<R>) -> Res<(pc_cuda::GraphExec, R)> {
    // SAFETY: `s` points at the executor's compute stream, alive for the call; the closure issues work on it
    let s = unsafe { &*s };
    let mut err = None;
    let r = s.capture(|_| f().map_err(|e| {
        err = Some(e.to_string());
        pc_cuda::Error::Unavailable("bootstrap under capture")
    }));
    match (r, err) {
        (_, Some(e)) => Err(e.into()),
        (r, None) => Ok(r?),
    }
}

/// d3: CtS x4 from the dumped stage plaintexts, EvalMod, StC-first output, per K, eager and graph replay.
pub fn d3_finish(dir: &str, ks_list: &[usize], graph: bool, lt_baby: KskSource, t: &mut Tally) -> Res<()> {
    let t0 = Instant::now();
    let d = Dump::open(dir);
    let pj = d.json("params.json");
    let chain = Chain::generate(&ChainSpec::n32())?;
    let qd = crate::dump::as_u64v(&pj["q"]);
    if qd != chain.q {
        return Err("the dump's primes differ from the n32 preset".into());
    }
    let n = chain.n;
    let n2 = 2 * n as u64;
    let nitems = pj["errs"].as_array().ok_or("errs")?.len();
    let stash_item = pj.get("stash_em").map(|_| nitems - 1);
    let boot = &pj["boot"];
    let cts_meta = boot["CtS"].as_array().ok_or("boot.CtS")?;
    // stage geometries and the rotations they read
    let mut stages_host = Vec::new();
    for (k, st) in cts_meta.iter().enumerate() {
        stages_host.push(dumped_stage(&d, st, "cts", k)?);
    }
    let mut steps = std::collections::BTreeSet::new();
    for (g, ..) in &stages_host {
        for i in 1..g.b {
            steps.insert(((i * g.stride) % (n / 2)) as i64);
        }
        if g.g > 1 {
            steps.insert(((g.b * g.stride) % (n / 2)) as i64);
        }
        steps.insert(g.offset.rem_euclid((n / 2) as i64));
        steps.remove(&0);
    }
    let dev = Device::open()?;
    let cfg = MemCfg { k_max: 8, pool: vec![PoolClass { k: 1, count: 1 }], events: 64, scratch_cts: 2 };
    let mut plan = MemPlan::build::<W, M>(&chain, cfg)?;
    let lay = plan.lay;
    let primes = chain.primes();
    plan.reserve("ks.plans", KsPlans::bytes::<W>(&chain, 8)?);
    plan.reserve("ks.regen_a", pc_gpu::ks::KsWork::<W>::regen_bytes(&lay, chain.dnum()));
    plan.reserve("keys", (steps.len() + 2) * DevKey::<W>::bytes(&lay, chain.dnum(), true, true));
    let ocfg = OpsCfg::default();
    Ops::<W>::reserve(&mut plan, &ocfg);
    let mut bts_bytes = 0;
    let mut top = d.ct("it0.fold").ids.len() - 1;
    for (g, rows, ptw, _, _, _) in &stages_host {
        bts_bytes += LtPts::<W>::bytes(rows.len(), top + 1 + lay.kp, *ptw) + LtStage::<W>::plan_bytes(&lay, &primes, g, top, LtEnd::FusedRescale, 8)?;
        top -= 2;
    }
    bts_bytes += 64 << 20; // EvalMod constants, monomials, rescale tables, pow2, key groups
    plan.reserve("bts.tables", bts_bytes);
    let in_meta = fmeta(&d, "it0.cts3");
    let ps = ps_plan(&pj);
    let da = pj["double_angle_its"].as_u64().ok_or("double_angle_its")? as u32;
    let (_, _, nslots) = EvalModPlan::<W>::program(&pc_core::scale::ScaleTables::precompute(&chain)?, in_meta, &ps, da)?;
    let (em_n, em_kk) = em_work(nslots, 8, EmMode::Complex);
    let wshape = BtsWorkShape { k_max: 8, g_max: 2, em_slots: em_n, em_kk, spare: 1 };
    plan.reserve("bts.work", BtsWork::<W>::bytes(&lay, wshape));
    plan.reserve("check.x", pc_cuda::Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(&lay, 9)));
    plan.reserve("bts.raise", RaisePlan::<W>::table_bytes());
    let arena = plan.allocate(&dev)?;
    let mut ex = Executor::<W>::new(&dev, &arena, &plan)?;
    ex.ks.carve_regen(&arena, &lay)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &plan)?;
    let kplans = KsPlans::new(&arena, &ex.compute, &ctx, 8)?;
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
    let mut top = d.ct("it0.fold").ids.len() - 1;
    for (g, rows, ptw, enc, nf, nqpt) in stages_host {
        if nqpt != top + 1 {
            return Err(format!("stage plaintexts hold {nqpt} limbs, the ciphertext {}", top + 1).into());
        }
        let pts = LtPts::new(&arena, s, &ctx, top, PtSource::Rows { rows, ptw, enc })?;
        let mut st = LtStage::new(&arena, s, &ctx, g, top, LtEnd::FusedRescale, nf, pts, &keys, stage_ptrs, 8)?;
        st.set_baby(lt_baby)?;
        stages.push(st);
        top -= 2;
    }
    let em = EvalModPlan::new(&arena, s, &ctx, EmMode::Complex, in_meta, &ps, da, keys.conj()?, keys.relin()?, stage_ptrs, 8, KskSource::Regen)?;
    let pow2 = Pow2::new(&arena, s, &ctx)?;
    let mut x = CtBuf::<Compute, W>::new(&arena, "check.x", lay, 8)?;
    let mut r = Ct::new(CtMeta::new(0, 1), CtBuf::<Compute, W>::new(&arena, "check.x", lay, 1)?);
    // the raise of the stash item: its top and the exact constant (computed as btsBegin does, checked against the dump)
    let raise_top = fmeta(&d, &format!("it{}.raise", nitems - 1)).top as usize;
    let drop = boot["raise_drop"].as_u64().unwrap_or(0) as usize;
    let exact = eval_const(&ctx, drop, pc_core::bts_setup::EVALMOD_K);
    let dumped = boot["cts0_const"].as_f64().ok_or("cts0_const")?;
    t.add("post-raise constant pre/(K N) vs dumped cts0_const (bits)", (exact.to_bits() != dumped.to_bits()) as usize, 1);
    let rplan = RaisePlan::new(&arena, s, &ctx, raise_top, exact, boot["cts0_t"].as_u64().ok_or("cts0_t")? as u32, raise::Lift::PerPrime)?;
    s.sync()?;
    println!("[pc-check] key source (LT babies, giant / offset rotations, raise): {lt_baby:?}");
    println!("[pc-check] d3 setup {:.1} s: {} rotation keys, arena {:.1} GB, EvalMod {} ops / {} buffers", t0.elapsed().as_secs_f64(), keys.rot.len(), arena.used_bytes() as f64 / 1e9, em.ops.len(), em.nslots);
    arena.freeze();
    let sh = Shadow::new(ShadowConfig { chain: ctx.shape, bts: BtsConfig::n32_shipped(), quirks: Quirks::none() }, Some(Arc::new(ctx.scale.clone())));
    let cf_all = pj["state_fields"]["cor_factor"].as_u64().ok_or("cor_factor")? << boot["stc_first_deg"].as_u64().ok_or("stc_first_deg")?;
    let cf_e = pow2_exp(cf_all)?;
    let out_top = em.out_meta.top as usize;
    let item = |c: usize| c % nitems;
    if pj.get("stash_raise").is_some() {
        let si = nitems - 1;
        let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
        for (stop, name) in [(RaiseStop::Raised, "MR-raised"), (RaiseStop::Scaled, "X-scaled")] {
            let h = d.ct(&format!("it{si}.stash.MR-atob"));
            upload(&mut r.buf, &cx.ex.compute, 0, &h)?;
            r.meta = CtMeta { level: ctx.shape.n_q - 2, deg: 1, nf: h.noise_factor, ..CtMeta::new(0, 1) };
            raise::raise_from_bottom(&mut cx, &rplan, None, lt_baby, &mut r, stop)?;
            let (b, o) = diff(&r.buf, &cx.ex.compute, 0, &d.ct(&format!("it{si}.stash.{name}")))?;
            t.add(&format!("raise: stash MR-atob -> {name}"), b, o);
        }
    }
    for &kk in ks_list {
        for c in 0..kk {
            upload(&mut x, &ex.compute, c, &d.ct(&format!("it{}.fold", item(c))))?;
        }
        let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
        for (k, st) in stages.iter().enumerate() {
            st.run(&mut cx, &mut x, kk, lt_baby)?;
            let (mut b, mut o) = (0, 0);
            for c in 0..kk {
                let (x1, o1) = diff(&x, &cx.ex.compute, c, &d.ct(&format!("it{}.cts{k}", item(c))))?;
                b += x1;
                o += o1;
            }
            t.add(&format!("K={kk} eager CtS stage {k}"), b, o);
        }
        em.run(&mut cx, &mut x, kk)?;
        if let Some(si) = stash_item {
            let (b, o) = (0..kk).filter(|&c| item(c) == si).map(|c| diff(&cx.work.em[0], &cx.ex.compute, c, &d.ct(&format!("it{si}.stash.EM-DA-Re")))).collect::<Res<Vec<_>>>()?.into_iter().fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
            if o > 0 {
                t.add(&format!("K={kk} eager EvalMod double-angle Re chain vs stash EM-DA-Re"), b, o);
            }
        }
        pow2.mult(&ctx, &cx.ex.compute, pc_gpu::ks::View::of(&x, 0), kk, out_top, cf_e)?;
        let (mut b, mut o) = (0, 0);
        for c in 0..kk {
            let (x1, o1) = diff(&x, &cx.ex.compute, c, &d.ct(&format!("it{}.out", item(c))))?;
            b += x1;
            o += o1;
        }
        t.add(&format!("K={kk} eager EvalMod + StC-first output vs it.out"), b, o);
        if graph {
            let sp: *const Stream<Compute> = &cx.ex.compute;
            let xp: *mut CtBuf<'_, Compute, W> = &mut x;
            let (gexec, _) = capture(sp, || {
                // SAFETY: x is not otherwise borrowed while the closure runs
                let x = unsafe { &mut *xp };
                for st in &stages {
                    st.run(&mut cx, x, kk, lt_baby)?;
                }
                em.run(&mut cx, x, kk)?;
                pow2.mult(&ctx, &cx.ex.compute, pc_gpu::ks::View::of(x, 0), kk, out_top, cf_e)
            })?;
            for rep in 0..2 {
                for c in 0..kk {
                    upload(&mut x, &cx.ex.compute, c, &d.ct(&format!("it{}.fold", item(c))))?;
                }
                gexec.launch(&cx.ex.compute)?;
                cx.ex.compute.sync()?;
                let (mut b, mut o) = (0, 0);
                for c in 0..kk {
                    let (x1, o1) = diff(&x, &cx.ex.compute, c, &d.ct(&format!("it{}.out", item(c))))?;
                    b += x1;
                    o += o1;
                }
                t.add(&format!("K={kk} graph replay {rep}: finish output (CtS x4 + EvalMod + StC-first) vs it.out"), b, o);
            }
        }
    }
    Ok(())
}

/// g0: the dense StC-first bootstrap #2 (stc_first_input -> raise -> CtS x4 -> EvalMod -> StC-first output, every
/// stage against its dump, the route's CtS plaintexts from `routes/32768`) and the sparse route's raise of #3
/// (non-prescaled path with the FLEXIBLEAUTO adjust, raise drop 1), with the dump's encapsulation keys.
pub fn g0_bts(dir: &str, lt_baby: KskSource, t: &mut Tally) -> Res<()> {
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
    let route = &pj["boot_routes"]["32768"];
    let rd = Dump::open(g0.dir.join("routes").join("32768"));
    let cts_meta = route["CtS"].as_array().ok_or("boot_routes.32768.CtS")?;
    let mut stages_host = Vec::new();
    for (k, st) in cts_meta.iter().enumerate() {
        stages_host.push(dumped_stage(&rd, st, "cts", k)?);
    }
    let mut steps = std::collections::BTreeSet::new();
    for (g, ..) in &stages_host {
        for i in 1..g.b {
            steps.insert(((i * g.stride) % (n / 2)) as i64);
        }
        if g.g > 1 {
            steps.insert(((g.b * g.stride) % (n / 2)) as i64);
        }
    }
    steps.remove(&0);
    let dev = Device::open()?;
    let cfg = MemCfg { k_max: 8, pool: vec![PoolClass { k: 1, count: 1 }], events: 64, scratch_cts: 2 };
    let mut plan = MemPlan::build::<W, M>(&chain, cfg)?;
    let lay = plan.lay;
    let primes = chain.primes();
    plan.reserve("ks.plans", KsPlans::bytes::<W>(&chain, 8)?);
    plan.reserve("ks.regen_a", pc_gpu::ks::KsWork::<W>::regen_bytes(&lay, chain.dnum()));
    plan.reserve("keys", (steps.len() + 4) * DevKey::<W>::bytes(&lay, chain.dnum(), true, true));
    let ocfg = OpsCfg::default();
    Ops::<W>::reserve(&mut plan, &ocfg);
    let mut bts_bytes = 2 * RaisePlan::<W>::table_bytes() + (64 << 20);
    let mut top = chain.q.len() - 1;
    for (g, rows, ptw, _, _, _) in &stages_host {
        bts_bytes += LtPts::<W>::bytes(rows.len(), top + 1 + lay.kp, *ptw) + LtStage::<W>::plan_bytes(&lay, &primes, g, top, LtEnd::FusedRescale, 8)?;
        top -= 2;
    }
    plan.reserve("bts.tables", 2 * bts_bytes);
    let in_meta = fmeta(&bd, "2.cts_3");
    let ps = ps_plan(&pj);
    let da = pj["double_angle_its"].as_u64().ok_or("double_angle_its")? as u32;
    let (_, _, nslots) = EvalModPlan::<W>::program(&pc_core::scale::ScaleTables::precompute(&chain)?, in_meta, &ps, da)?;
    let (em_n, em_kk) = em_work(nslots, 8, EmMode::Complex);
    let wshape = BtsWorkShape { k_max: 8, g_max: 2, em_slots: em_n, em_kk, spare: 1 };
    plan.reserve("bts.work", BtsWork::<W>::bytes(&lay, wshape));
    plan.reserve("check.x", pc_cuda::Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(&lay, 1)));
    let arena = plan.allocate(&dev)?;
    let mut ex = Executor::<W>::new(&dev, &arena, &plan)?;
    ex.ks.carve_regen(&arena, &lay)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &plan)?;
    let kplans = KsPlans::new(&arena, &ex.compute, &ctx, 8)?;
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
    let encaps = EncapsKeys::new(&arena, s, n, keys.dense_to_sparse()?, keys.sparse_to_dense()?, stage_ptrs)?;
    let mut stages = Vec::new();
    let mut top = chain.q.len() - 1;
    for (g, rows, ptw, enc, nf, nqpt) in stages_host {
        if nqpt != top + 1 {
            return Err(format!("route plaintexts hold {nqpt} limbs, the stage input {}", top + 1).into());
        }
        let pts = LtPts::new(&arena, s, &ctx, top, PtSource::Rows { rows, ptw, enc })?;
        let mut st = LtStage::new(&arena, s, &ctx, g, top, LtEnd::FusedRescale, nf, pts, &keys, stage_ptrs, 8)?;
        st.set_baby(lt_baby)?;
        stages.push(st);
        top -= 2;
    }
    // FIDESLIB_BTS_REAL=1 in the g0 run: the dense StC-first route runs the real chain
    let real = pj["run"]["env"]["FIDESLIB_BTS_REAL"].as_str() == Some("1");
    let em = EvalModPlan::new(&arena, s, &ctx, if real { EmMode::Real } else { EmMode::Complex }, in_meta, &ps, da, keys.conj()?, keys.relin()?, stage_ptrs, 8, KskSource::Regen)?;
    let pow2 = Pow2::new(&arena, s, &ctx)?;
    let exact0 = eval_const(&ctx, 0, pc_core::bts_setup::EVALMOD_K);
    let r512 = &pj["boot_routes"]["512"];
    let drop512 = r512["raise_drop"].as_u64().unwrap_or(0) as usize;
    let exact512 = eval_const(&ctx, drop512, pc_core::bts_setup::EVALMOD_K);
    t.add("g0 post-raise constant (dense) vs cts0_const (bits)", (exact0.to_bits() != route["cts0_const"].as_f64().unwrap_or(0.0).to_bits()) as usize, 1);
    let rp0 = RaisePlan::new(&arena, s, &ctx, chain.q.len() - 1, exact0, route["cts0_t"].as_u64().ok_or("cts0_t")? as u32, raise::Lift::PerPrime)?;
    let rp512 = RaisePlan::new(&arena, s, &ctx, chain.q.len() - 1 - 2 * drop512, exact512, r512["cts0_t"].as_u64().ok_or("cts0_t")? as u32, raise::Lift::PerPrime)?;
    let mut x = Ct::new(CtMeta::new(0, 1), CtBuf::<Compute, W>::new(&arena, "check.x", lay, 1)?);
    // our own CtS encodings (pc_core factors at pc_shadow levels) vs the route's dumped plaintexts
    let dplan = pc_gpu::bts::dense::DensePlan::new(&chain, &pc_gpu::bts::BtsSetup::n32_shipped())?;
    let mut own = Vec::new();
    for h in &dplan.variants[0].cts {
        own.push((h.width, LtPts::new(&arena, s, &ctx, h.top, PtSource::Coeffs(&h.coeffs))?));
    }
    s.sync()?;
    println!("[pc-check] key source (LT babies, giant / offset rotations, raise): {lt_baby:?}");
    println!("[pc-check] g0 setup {:.1} s, arena {:.1} GB", t0.elapsed().as_secs_f64(), arena.used_bytes() as f64 / 1e9);
    arena.freeze();
    let sh = Shadow::new(ShadowConfig { chain: ctx.shape, bts: BtsConfig::n32_shipped(), quirks: Quirks::none() }, Some(Arc::new(ctx.scale.clone())));
    let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
    let load = |x: &mut Ct<'_, W>, cx: &BtsCx<'_, '_, W, M>, name: &str| -> Res<()> {
        let h = bd.ct(name);
        upload(&mut x.buf, &cx.ex.compute, 0, &h)?;
        x.meta = CtMeta { level: cx.ctx.shape.n_q - 1 - h.level as u32, deg: h.noise_level as u8, nf: h.noise_factor, ..CtMeta::new(0, 1) };
        Ok(())
    };
    let check = |x: &Ct<'_, W>, cx: &BtsCx<'_, '_, W, M>, name: &str, t: &mut Tally| -> Res<()> {
        let h = bd.ct(name);
        let (b, o) = diff(&x.buf, &cx.ex.compute, 0, &h)?;
        t.add(&format!("g0 bts {name}"), b, o);
        let lv = cx.ctx.shape.n_q - 1 - h.level as u32;
        let meta_bad = (x.meta.level != lv || x.meta.deg as i64 != h.noise_level || x.meta.nf.to_bits() != h.noise_factor.to_bits()) as usize;
        if meta_bad != 0 {
            println!("[pc-check]   meta {:?} vs dump level {lv} deg {} nf {}", x.meta, h.noise_level, h.noise_factor);
        }
        t.add(&format!("g0 bts {name} metadata (level, degree, scale bits)"), meta_bad, 1);
        Ok(())
    };
    for (k, st) in stages.iter().enumerate() {
        let (w, p) = &own[k];
        let (mut bad, mut tot) = (0, 0);
        for j in 0..st.pts.npts {
            let a = p.download(&cx.ex.compute, j)?;
            let b = st.pts.download(&cx.ex.compute, j)?;
            if a.len() != b.len() {
                println!("[pc-check]   stage {k}: own width {w} vs dumped {}", st.pts.ptw);
                bad += 1;
                tot += 1;
                break;
            }
            bad += a.iter().zip(&b).filter(|(x, y)| x != y).count();
            tot += a.len();
        }
        println!("[pc-check] g0 own CtS stage {k} plaintexts (width {w}) vs route dump: {bad} words differ of {tot} (encoder rounding allowed); scale own {:e} dump {:e} top {} / {}", dplan.variants[0].cts[k].nf, st.pt_nf, dplan.variants[0].cts[k].top, st.top);
        let a = p.download(&cx.ex.compute, 0)?;
        let b = st.pts.download(&cx.ex.compute, 0)?;
        println!("[pc-check]   pt0 own {:?} dump {:?}", &a[..4], &b[..4]);
        if k == 0 {
            let q0 = chain.q[0];
            let hn = pc_gpu::keys::hostref::HostNtt::new(q0, chain.roots()[0], n);
            let mut row: Vec<u64> = b[..n].iter().map(|&x| x as u64).collect();
            hn.inv(&mut row);
            let mine: Vec<u64> = dplan.variants[0].cts[0].coeffs[0].iter().map(|&c| c.rem_euclid(q0 as i128) as u64).collect();
            let neg: Vec<u64> = mine.iter().map(|&c| (q0 - c) % q0).collect();
            let same = row.iter().zip(&mine).filter(|(a, b)| a == b).count();
            let nsame = row.iter().zip(&neg).filter(|(a, b)| a == b).count();
            let cen = |x: u64| if x > q0 / 2 { x as i64 - q0 as i64 } else { x as i64 };
            println!("[pc-check]   stage 0 pt0 coefficients mod q0: equal {same}, negated {nsame} of {n}; dump {:?} mine {:?}", row[..6].iter().map(|&x| cen(x)).collect::<Vec<_>>(), mine[..6].iter().map(|&x| cen(x)).collect::<Vec<_>>());
            let nz_d: Vec<usize> = (0..n).filter(|&i| row[i] != 0).take(6).collect();
            let nz_m: Vec<usize> = (0..n).filter(|&i| mine[i] != 0).take(6).collect();
            println!("[pc-check]   nonzero idx dump {nz_d:?} mine {nz_m:?}");
            let mut fw = mine.clone();
            hn.fwd(&mut fw);
            let eq_dump = fw.iter().zip(&b[..n]).filter(|(x, y)| **x == **y as u64).count();
            let eq_dev = fw.iter().zip(&a[..n]).filter(|(x, y)| **x == **y as u64).count();
            println!("[pc-check]   host NTT(mine) == dump row: {eq_dump}, == device row: {eq_dev} of {n}; host {:?}", &fw[..4]);
        }
    }
    // #2: dense StC-first (prescaled raise)
    load(&mut x, &cx, "2.stc_first_input")?;
    raise::raise(&mut cx, &rp0, Some(&encaps), lt_baby, std::slice::from_mut(&mut x), None)?;
    check(&x, &cx, "2.mod_raise", t)?;
    for (k, st) in stages.iter().enumerate() {
        st.run(&mut cx, &mut x.buf, 1, lt_baby)?;
        let (ot, od) = st.out_top_deg(2);
        let m = cx.sh.mult_pt(&x.meta, &pc_shadow::PtMeta { level: ctx.shape.n_q - 1 - st.top as u32, deg: 1, nf: st.pt_nf, tag: Default::default() })?.out;
        x.meta = if od == 1 { cx.sh.rescale(&m)?.out } else { m };
        debug_assert_eq!(x.meta.level, ctx.shape.n_q - 1 - ot as u32);
        check(&x, &cx, &format!("2.cts_{k}"), t)?;
    }
    em.run(&mut cx, &mut x.buf, 1)?;
    x.meta = CtMeta { level: ctx.shape.n_q - 1 - em.out_meta.top as u32, deg: em.out_meta.nl as u8, nf: em.out_meta.nf, ..x.meta };
    check(&x, &cx, "2.eval_mod", t)?;
    let st2 = bd.json("2.json");
    let cf_all = st2["state"]["cor_factor"].as_u64().ok_or("cor_factor")? << route["stc_first_deg"].as_u64().ok_or("stc_first_deg")?;
    pow2.mult(&ctx, &cx.ex.compute, pc_gpu::ks::View::of(&x.buf, 0), 1, em.out_meta.top as usize, pow2_exp(cf_all)?)?;
    check(&x, &cx, "2.stc_first_output", t)?;
    check(&x, &cx, "2.out", t)?;
    // #3: sparse route's raise (non-prescaled: FLEXIBLEAUTO adjust with the call's correction)
    let st3 = bd.json("3.json");
    let corr = st3["state"]["correction"].as_u64().ok_or("correction")? as u32;
    load(&mut x, &cx, "3.in")?;
    raise::raise(&mut cx, &rp512, Some(&encaps), lt_baby, std::slice::from_mut(&mut x), Some(corr))?;
    check(&x, &cx, "3.mod_raise", t)?;
    Ok(())
}
