//! `replay`: run a bound block program on the `Compute` lane (one `GraphExec` per program when graphs are on).
//!
//! Every step issues its kernels on `compute`; no level bookkeeping beyond what the op layer's recipes do (the
//! compiler already decided every prelude; `ReplayCfg::check_meta` compares each output's metadata with the
//! compiled one), no device allocation (the arena is frozen), no host synchronization.
//!
//! Graphs (`ReplayCfg::use_graphs`): the first run of a `Bound` executes eagerly (warming the route caches), then
//! the same step sequence is captured into one `GraphExec` held by the `Bound`; later runs launch it. What a
//! captured graph must not see change (slots, plaintext rows, weight frames, ring constants, stage caches) and how
//! each is held: docs/ARCHITECTURE.md "pc-program replay". Inputs / outputs are written and read outside the graph.
use crate::bind::{BoundPt, Bound, CoeffSumStep, FinishStep, Gather, Held, Step, StepOp, Unit, UnitKind, NONE};
use crate::schedule::{self, Sched};
use crate::error::{Error, Result};
use pc_core::encode::Complex;
use pc_cuda::{launch_result, Compute, Event, GraphExec, Stream};
use pc_gpu::bts::{batch, BtsCx, BtsRoute, BtsWork, RouteKind};
use pc_gpu::encode_dev::CoeffRowsDevice;
use pc_gpu::ks::{KeyGroup, KsPlans, KskSource};
use pc_gpu::ops::{Ops, Pt};
use pc_gpu::rotate::{self, KsCfg};
use pc_gpu::{Context, Ct, DevStrategy, DevWord, Executor};
use pc_shadow::{CtMeta, Shadow};
use std::ffi::c_void;
use std::time::Instant;

/// The route objects a runtime holds, by family (one each; `BtsRoute::run` takes the per-call `Route`).
pub struct Routes<'r, 'a, W: DevWord, M: DevStrategy<W>> {
    pub list: Vec<(RouteKind, &'r dyn BtsRoute<'a, W, M>)>,
}

impl<'r, 'a, W: DevWord, M: DevStrategy<W>> Routes<'r, 'a, W, M> {
    pub fn new() -> Self {
        Routes { list: vec![] }
    }
    pub fn add(&mut self, r: &'r dyn BtsRoute<'a, W, M>) {
        self.list.push((r.kind(), r));
    }
    pub fn get(&self, kind: RouteKind) -> Option<&'r dyn BtsRoute<'a, W, M>> {
        self.list.iter().find(|(k, _)| *k == kind).map(|(_, r)| *r)
    }
}

impl<W: DevWord, M: DevStrategy<W>> Default for Routes<'_, '_, W, M> {
    fn default() -> Self {
        Self::new()
    }
}

/// Caches a refresh fills on demand (the StC-first `StageCache`): locked while a capture runs.
pub use pc_gpu::graph::CaptureLock;

/// The runtime service behind `StepOp::Finish` (PCMM / CCMM ops recorded as external refreshes). It runs
/// the whole external op on the compute lane: reads `slots[f.ins[i]]`, writes `slots[f.outs[m]]` at the metadata the
/// op lands at (the replay checks it against `f.out_metas` when `check_meta`). An output slot may be a dying input's:
/// every input is read before any output is written. No host synchronization, no allocation (it may be captured).
pub trait FinishExec<'a, W: DevWord, M: DevStrategy<W>> {
    fn finish(&mut self, cx: &mut BtsCx<'_, 'a, W, M>, f: &FinishStep, slots: &mut [Ct<'a, W>]) -> Result<()>;
}

/// Replay knobs (the `exec` part of the Config: `exec.use_graphs`, debug checks).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplayCfg {
    /// Capture each bound block once and launch the graph afterwards; false = eager every time (debugging).
    pub use_graphs: bool,
    /// Compare every eager step's output metadata with the compiled one (host work per step; debugging).
    pub check_meta: bool,
    /// Captured plaintext-row layouts kept per block (weight ring: a block may cycle through a few slot sets).
    pub graphs_per_block: usize,
}

impl Default for ReplayCfg {
    fn default() -> Self {
        ReplayCfg { use_graphs: true, check_meta: false, graphs_per_block: 2 }
    }
}

/// What one run borrows from the runtime.
pub struct Rt<'r, 'a, W: DevWord, M: DevStrategy<W>> {
    pub ctx: &'r Context<'a, W, M>,
    pub ex: &'r mut Executor<'a, W>,
    pub sh: &'r Shadow,
    pub ks: &'r KsPlans<'a>,
    pub ops: &'r mut Ops<'a, W>,
    pub work: &'r mut BtsWork<'a, W>,
    pub routes: &'r Routes<'r, 'a, W, M>,
    /// Key source of the block's key switches (Regen for seeded / packed keys).
    pub source: KskSource,
    /// Caches to lock while capturing.
    pub locks: &'r [&'r dyn CaptureLock],
    /// The external-op service (`StepOp::Finish`); None = a program with finishes refuses to run.
    pub finish: Option<&'r mut dyn FinishExec<'a, W, M>>,
    /// A second lane for the steps off a block's critical path (`schedule`); None = one lane.
    pub side: Option<&'r mut SideLane<'a, W>>,
    /// The streamed-weight expander (`StepOp::CoeffSum`, compute lane); None = a program with one refuses to run.
    pub coeff: Option<&'r mut CoeffRowsDevice<'a, W>>,
}

/// The side lane: its own executor (stream + key-switching / scratch workspaces) and op layer (constant
/// ring, rescaler), ordering events by step index (`ev`, at least the largest block's step count) and the
/// fork / join events. Built at setup next to the compute executor.
pub struct SideLane<'a, W: DevWord> {
    pub ex: Executor<'a, W>,
    pub ops: Ops<'a, W>,
    pub ev: Vec<Event>,
    pub fork: Event,
    pub join: Event,
}

/// What a replay issues, in step order: a captured static segment or one dynamic step (eager).
enum Item {
    Graph(GraphExec),
    Step(usize),
}

/// A captured block: one graph per static segment (the steps between the dynamic ones) and the dynamic steps in
/// order, its leased constant slots, the output metadata, the plaintext rows it baked.
pub struct Captured {
    items: Vec<Item>,
    dynamic: Vec<u32>,
    lease: Vec<usize>,
    /// The side lane's constant-ring slots leased by the capture.
    side_lease: Vec<usize>,
    out_metas: Vec<CtMeta>,
    pt_rows: Vec<*const c_void>,
    stream_base: *const f32,
    pub capture_ms: f64,
}

/// How a run was served.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Served {
    Eager,
    /// Eager, then captured (first graph run of the block).
    Captured,
    Replayed,
}

/// One run: how it was served, host microseconds spent issuing it (no device wait), metadata mismatches seen
/// (eager with `check_meta`), the first few of them.
#[derive(Clone, Debug)]
pub struct RunInfo {
    pub served: Served,
    pub host_us: f64,
    /// Of `host_us`: the `cudaGraphLaunch` call (graph runs; driver time, proportional to the graph's nodes).
    pub launch_us: f64,
    pub meta_mismatches: usize,
    pub first_mismatches: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ReplayStats {
    pub eager: usize,
    pub captures: usize,
    pub replays: usize,
    /// Graphs dropped because a plaintext address moved.
    pub invalidations: usize,
    pub capture_ms: f64,
}

pub struct Replayer {
    pub cfg: ReplayCfg,
    pub stats: ReplayStats,
    /// Use the runtime's side lane (`Rt::side`) when it has one.
    pub side_lane: bool,
    /// Lane schedules by block label, for the dynamic-step set they were made for.
    scheds: std::collections::HashMap<String, (Vec<u32>, Sched)>,
}

fn rerr(block: &str, msg: impl Into<String>) -> Error {
    Error::Replay { block: block.to_string(), msg: msg.into() }
}

/// `dst = src` (live Q rows, both parts) and its metadata, on `s` (one kernel).
pub fn copy_ct<W: DevWord, M: DevStrategy<W>>(ctx: &Context<'_, W, M>, s: &Stream<Compute>, dst: &mut Ct<'_, W>, src: &Ct<'_, W>) -> Result<()> {
    let top = src.top(ctx.shape)?;
    // compact buffers (registers) hold Q rows 0..nq of their own layout
    if top >= dst.buf.lay.nq.min(src.buf.lay.nq) || src.k() > dst.k() {
        return Err(Error::Replay {
            block: String::new(),
            msg: format!("copy of a ciphertext at level {} (top limb {top}, k {}) into a buffer of {} Q rows (k {})", src.meta.level, src.k(), dst.buf.lay.nq, dst.k()),
        });
    }
    let args = pc_kernels::EwArgs {
        out: dst.buf.w(),
        a: src.buf.r(),
        b: src.buf.r(),
        rs: ctx.lay.rs_q(top),
        sc: std::ptr::null(),
        scs: std::ptr::null(),
        pr: ctx.pr(),
        k: src.k() as i32,
        nparts: 2,
        sub: 0,
        zero_p: 0,
    };
    // SAFETY: both buffers hold k cts with rows 0..=top in both parts (checked above), each read through its own strides
    launch_result("ew_copy", unsafe { (ctx.table.ew_copy)(&args, s.raw()) })?;
    dst.meta = src.meta;
    Ok(())
}

struct Tables<'t, 'a> {
    groups: &'t [KeyGroup<'a>],
    relin: u32,
    lists: &'t [Vec<i64>],
    pts: &'t [BoundPt],
    finishes: &'t [FinishStep],
    coeff_sums: &'t [CoeffSumStep],
    stream_base: *const f32,
}

fn pt<'t>(block: &str, pts: &'t [BoundPt], i: u32) -> Result<Pt<'t>> {
    let bp = pts.get(i as usize).ok_or_else(|| rerr(block, format!("plaintext {i} out of range")))?;
    let meta = bp.meta.ok_or_else(|| rerr(block, format!("plaintext {}@{} not resolved (Bound::resolve_pts)", bp.key.name, bp.key.level)))?;
    // SAFETY: resolved from a resident PtStore slot; the store outlives the run and the caller re-resolves after
    // any staging (address change detected by resolve_pts)
    Ok(unsafe { Pt::from_raw(meta, bp.rows) })
}

fn group<'t, 'a>(block: &str, groups: &'t [KeyGroup<'a>], g: u32) -> Result<&'t KeyGroup<'a>> {
    groups.get(g as usize).ok_or_else(|| rerr(block, format!("key group {g} out of range")))
}

/// Issue one step. `slots` is the bound slot array (raw: an op reads one or two slots and writes another; bind
/// checked that only the declared in-place pairs alias).
fn exec_step<'a, W: DevWord, M: DevStrategy<W>>(block: &str, rt: &mut Rt<'_, 'a, W, M>, st: &Step, slots: *mut Ct<'a, W>, n_slots: usize, tmp: &mut Ct<'a, W>, t: &Tables<'_, 'a>) -> Result<()> {
    let (ctx, sh) = (rt.ctx, rt.sh);
    if let StepOp::Finish { f } = &st.op {
        let fs = t.finishes.get(*f as usize).ok_or_else(|| rerr(block, format!("finish {f} out of range")))?;
        if fs.ins.iter().chain(&fs.outs).any(|&s| s as usize >= n_slots) {
            return Err(rerr(block, format!("finish {f}: slot outside the {n_slots} bound slots")));
        }
        let svc = rt.finish.as_deref_mut().ok_or_else(|| rerr(block, format!("finish {f} ({}): the runtime has no finish service (Rt::finish)", fs.desc)))?;
        // SAFETY: the bound slot array of `n_slots` ciphertexts; no other reference into it is live during the step
        let all = unsafe { std::slice::from_raw_parts_mut(slots, n_slots) };
        let mut cx = BtsCx { ctx, ex: &mut *rt.ex, sh, ks: rt.ks, ops: &mut *rt.ops, work: &mut *rt.work };
        return svc.finish(&mut cx, fs, all);
    }
    if let StepOp::CoeffSum { c } = &st.op {
        let cs = t.coeff_sums.get(*c as usize).ok_or_else(|| rerr(block, format!("coeff_sum {c} out of range")))?;
        if cs.ins.iter().chain(std::iter::once(&st.out)).any(|&s| s as usize >= n_slots) || cs.ins.contains(&st.out) {
            return Err(rerr(block, format!("coeff_sum {c}: slots outside the {n_slots} bound slots, or the output among the terms")));
        }
        if t.stream_base.is_null() {
            return Err(rerr(block, "coeff_sum before the block's weight frame was staged (residency::StreamRing::adopt)"));
        }
        let crd = rt.coeff.as_deref_mut().ok_or_else(|| rerr(block, format!("coeff_sum {c}: the runtime has no CoeffRowsDevice (Rt::coeff)")))?;
        let n = ctx.lay.n;
        // SAFETY: bind checked the slots (< n_slots, output not a term); the output is the only mutable reference
        let o: &mut Ct<'a, W> = unsafe { &mut *slots.add(st.out as usize) };
        let cts: Vec<&Ct<'a, W>> = cs.ins.iter().map(|&s| unsafe { &*slots.add(s as usize) }).collect();
        // SAFETY: the frame holds `Bound::streams.total` rows of N floats (StreamRing sized it); bind checked rows
        let rows: Vec<*const f32> = cs.rows.iter().map(|&r| unsafe { t.stream_base.add(r as usize * n) }).collect();
        crd.sum(ctx, &rt.ex.compute, sh, o, &cts, &rows)?;
        return Ok(());
    }
    // SAFETY: indices checked at bind (< slots.len()); `o` is the only mutable reference, and inputs are taken as
    // shared references only when they are not `o`'s slot
    let o: &mut Ct<'a, W> = unsafe { &mut *slots.add(st.out as usize) };
    let at = |i: u32| -> &Ct<'a, W> { unsafe { &*slots.add(i as usize) } };
    let one = std::slice::from_mut;
    let kc = KsCfg { plans: rt.ks, source: rt.source };
    // the A operand for ops without an in-place kernel: in place -> a copy in `tmp`
    macro_rules! src_a {
        () => {{
            if st.inplace {
                copy_ct(ctx, &rt.ex.compute, tmp, o)?;
                &*tmp
            } else {
                at(st.a)
            }
        }};
    }
    match &st.op {
        StepOp::Add | StepOp::Sub => {
            let sub = st.op == StepOp::Sub;
            if st.inplace {
                let b: &Ct<'a, W> = if st.b == st.out {
                    copy_ct(ctx, &rt.ex.compute, tmp, o)?;
                    &*tmp
                } else {
                    at(st.b)
                };
                if sub {
                    rt.ops.sub_inplace(ctx, rt.ex, sh, one(o), &[b])?
                } else {
                    rt.ops.add_inplace(ctx, rt.ex, sh, one(o), &[b])?
                }
            } else if sub {
                rt.ops.sub(ctx, rt.ex, sh, one(o), &[at(st.a)], &[at(st.b)])?
            } else {
                rt.ops.add(ctx, rt.ex, sh, one(o), &[at(st.a)], &[at(st.b)])?
            }
        }
        StepOp::AddPt(i) | StepOp::MultPt(i) => {
            let p = pt(block, t.pts, *i)?;
            let add = matches!(st.op, StepOp::AddPt(_));
            match (st.inplace, add) {
                (true, true) => rt.ops.add_pt_inplace(ctx, rt.ex, sh, one(o), &[p])?,
                (true, false) => rt.ops.mult_pt_inplace(ctx, rt.ex, sh, one(o), &[p])?,
                (false, true) => rt.ops.add_pt(ctx, rt.ex, sh, one(o), &[at(st.a)], &[p])?,
                (false, false) => rt.ops.mult_pt(ctx, rt.ex, sh, one(o), &[at(st.a)], &[p])?,
            }
        }
        StepOp::AddConst(c) => {
            if st.inplace {
                rt.ops.add_scalar_inplace(ctx, rt.ex, sh, one(o), &[*c])?
            } else {
                rt.ops.add_scalar(ctx, rt.ex, sh, one(o), &[at(st.a)], &[*c])?
            }
        }
        StepOp::MultConst(c) => {
            let z = Complex::new(*c, 0.0);
            if st.inplace {
                rt.ops.mult_scalar_inplace(ctx, rt.ex, sh, one(o), &[z])?
            } else {
                rt.ops.mult_scalar(ctx, rt.ex, sh, one(o), &[at(st.a)], &[z])?
            }
        }
        StepOp::Mult => {
            let a = src_a!();
            let key = group(block, t.groups, t.relin)?;
            rt.ops.mult(ctx, rt.ex, kc, sh, key, o, a, at(st.b))?
        }
        StepOp::Square => {
            let a = src_a!();
            let key = group(block, t.groups, t.relin)?;
            rt.ops.square(ctx, rt.ex, kc, sh, key, o, a)?
        }
        StepOp::Negate => {
            let a = src_a!();
            rt.ops.negate(ctx, rt.ex, sh, one(o), &[a])?
        }
        StepOp::MultI => {
            if st.inplace {
                rt.ops.mult_i_inplace(ctx, rt.ex, sh, one(o))?
            } else {
                rt.ops.mult_i(ctx, rt.ex, sh, one(o), &[at(st.a)])?
            }
        }
        StepOp::Rotate { k, g } => {
            let a = src_a!();
            if *g == NONE {
                copy_ct(ctx, &rt.ex.compute, o, a)?;
            } else {
                rotate::rotate(ctx, rt.ex, kc, sh, group(block, t.groups, *g)?, a, *k, o)?
            }
        }
        StepOp::RotateSum { l, g } => {
            let a = src_a!();
            let steps = t.lists.get(*l as usize).ok_or_else(|| rerr(block, format!("step list {l} out of range")))?;
            rotate::rotate_sum(ctx, rt.ex, kc, sh, group(block, t.groups, *g)?, a, steps, o)?
        }
        StepOp::Conjugate { g } => {
            let a = src_a!();
            rotate::conjugate(ctx, rt.ex, kc, sh, group(block, t.groups, *g)?, a, o)?
        }
        StepOp::Rescale => {
            if st.inplace {
                rt.ops.realize(ctx, rt.ex, sh, one(o))?
            } else {
                rt.ops.realize_to(ctx, rt.ex, sh, o, at(st.a))?
            }
        }
        StepOp::DropTo(l) => {
            if !st.inplace {
                copy_ct(ctx, &rt.ex.compute, o, at(st.a))?;
            }
            rt.ops.drop_to_level(sh, one(o), *l)
        }
        StepOp::Finish { .. } => return Err(rerr(block, "finish step reached the single-output lowering")),
        StepOp::CoeffSum { .. } => return Err(rerr(block, "coeff_sum step reached the single-output lowering")),
        StepOp::Bootstrap { route, kind, .. } => {
            if !st.inplace {
                copy_ct(ctx, &rt.ex.compute, o, at(st.a))?;
            }
            let r = rt.routes.get(*kind).ok_or_else(|| rerr(block, format!("route {kind:?} not held")))?;
            let mut cx = BtsCx { ctx, ex: &mut *rt.ex, sh, ks: rt.ks, ops: &mut *rt.ops, work: &mut *rt.work };
            r.run(&mut cx, one(o), route)?
        }
    }
    Ok(())
}

fn meta_differs(got: &CtMeta, want: &CtMeta) -> bool {
    got.level != want.level || got.deg != want.deg || (want.nf > 0.0 && got.nf > 0.0 && ((got.nf - want.nf) / want.nf).abs() > 1e-9)
}

/// Issue every step of `b` in order (eager, or under capture).
/// A per-step observer for debugging runs (`Replayer::eager_with`): step index, the bound block after it (it may
/// overwrite the step's output, e.g. inject a reference value), the runtime.
pub type StepHook<'h, 'a, W, M> = dyn FnMut(usize, &mut Bound<'a, W>, &mut Rt<'_, 'a, W, M>) -> Result<()> + 'h;

fn issue<'a, W: DevWord, M: DevStrategy<W>>(rt: &mut Rt<'_, 'a, W, M>, b: &mut Bound<'a, W>, check: bool, info: &mut RunInfo, hook: Option<&mut StepHook<'_, 'a, W, M>>, sched: Option<&Sched>) -> Result<()> {
    let n = b.steps.len();
    issue_range(rt, b, 0, n, check, info, hook, sched)
}

/// Swap the compute lane's executor / op layer with the side lane's (a side step issues through `rt` like any
/// other; device buffers do not move, only the owning structs).
fn swap_lane<'a, W: DevWord, M: DevStrategy<W>>(rt: &mut Rt<'_, 'a, W, M>) -> Result<()> {
    let s = rt.side.as_deref_mut().ok_or_else(|| Error::Replay { block: String::new(), msg: "side step without a side lane".into() })?;
    std::mem::swap(&mut *rt.ex, &mut s.ex);
    std::mem::swap(&mut *rt.ops, &mut s.ops);
    Ok(())
}

/// Join the side lane back into compute (end of a region).
fn join_side<'a, W: DevWord, M: DevStrategy<W>>(rt: &mut Rt<'_, 'a, W, M>) -> Result<()> {
    if let Some(s) = rt.side.as_deref_mut() {
        s.ex.compute.record(&s.join)?;
        rt.ex.compute.wait(&s.join)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn issue_range<'a, W: DevWord, M: DevStrategy<W>>(rt: &mut Rt<'_, 'a, W, M>, b: &mut Bound<'a, W>, lo: usize, hi: usize, check: bool, info: &mut RunInfo, mut hook: Option<&mut StepHook<'_, 'a, W, M>>, sched: Option<&Sched>) -> Result<()> {
    if b.slots.len() < b.n_slots {
        return Err(rerr(&b.label, format!("{} slots present, the block needs {} (a shared block outside Bound::with_slab)", b.slots.len(), b.n_slots)));
    }
    let sched = if rt.side.is_some() { sched.filter(|s| s.lane.len() == b.steps.len()) } else { None };
    let mut forked = false;
    // steps below `unit_end` were issued by the unit that holds them
    let mut unit_end = lo;
    // the rotation chain whose gathered inputs and ModUp the compute lane's workspaces (`ks_owner`) and the held
    // buffers (`held_owner`) hold: a later link of another state runs fresh (bind's chains never meet one, a guard)
    let (mut ks_owner, mut held_owner): (Option<u32>, Option<u32>) = (None, None);
    for i in lo..hi {
        let in_unit = i < unit_end;
        let unit = if in_unit {
            None
        } else {
            b.unit_at(i)
                .filter(|u| unit_fits(u, &b.slots))
                .map(|u| (b.unit_of[i], u.hi as usize))
                .filter(|&(_, e)| e <= hi && sched.is_none_or(|s| s.lane[i..e].iter().all(|&l| l == 0)))
        };
        let dynamic = b.dynamic.contains(&(i as u32));
        let lane = sched.map_or(0, |s| s.lane[i]);
        if dynamic && forked {
            join_side(rt)?;
            forked = false;
        }
        if let Some(s) = sched {
            let side = rt.side.as_deref_mut().ok_or_else(|| rerr(&b.label, "schedule without a side lane"))?;
            if lane == 1 && !forked {
                // fork: the side lane starts after everything compute issued so far
                rt.ex.compute.record(&side.fork)?;
                side.ex.compute.wait(&side.fork)?;
                forked = true;
            }
            // a unit waits for what every one of its steps waits for
            let waits = match unit {
                Some((_, e)) => i..e,
                None if in_unit => i..i,
                None => i..i + 1,
            };
            for j in waits {
                let w = s.wait[j];
                if w != NONE {
                    let ev = side.ev.get(w as usize).ok_or_else(|| rerr(&b.label, format!("step {w}: {} side events", side.ev.len())))?;
                    if lane == 1 {
                        side.ex.compute.wait(ev)?;
                    } else {
                        rt.ex.compute.wait(ev)?;
                    }
                }
            }
        }
        if !in_unit {
            if lane == 1 {
                swap_lane(rt)?;
            }
            let pre_metas = if dynamic {
                let st = &b.steps[i];
                let m = |s: u32| if s != NONE { b.slots.get(s as usize).map(|c| c.meta) } else { None };
                Some((m(st.a), m(st.b)))
            } else {
                None
            };
            {
                let Bound { label, slots, tmp, gather, held, steps, groups, relin, step_lists, units, pts, finishes, coeff_sums, stream_base, names, .. } = &mut *b;
                let tmp = tmp.as_mut().ok_or_else(|| rerr(label, "no scratch slot"))?;
                let r = match unit {
                    Some((u, _)) => {
                        let unit = &units[u as usize];
                        let gather = gather.as_mut().ok_or_else(|| rerr(label, "no gather buffer"))?;
                        let mut stale = false;
                        match unit.kind {
                            UnitKind::Grid { held: h, .. } if unit.chain != NONE => {
                                let owner = if h { &mut held_owner } else { &mut ks_owner };
                                stale = *owner != Some(unit.chain);
                                *owner = Some(unit.chain);
                            }
                            _ => ks_owner = None,
                        }
                        exec_unit(label, rt, unit, steps, slots.as_mut_ptr(), slots.len(), gather, held.as_mut(), groups, *relin, step_lists, stale)
                    }
                    None => {
                        if lane == 0 && !quiet(&steps[i].op) {
                            ks_owner = None;
                        }
                        let t = Tables { groups, relin: *relin, lists: step_lists, pts, finishes, coeff_sums, stream_base: *stream_base };
                        exec_step(label, rt, &steps[i], slots.as_mut_ptr(), slots.len(), tmp, &t)
                    }
                };
                if lane == 1 {
                    swap_lane(rt)?;
                }
                if let Err(e) = r {
                    // with the check on, the mismatches before the failing step say where the metadata left the compile's
                    if check {
                        let st = &steps[i];
                        let m = |s: u32| if s != NONE { slots.get(s as usize).map(|c| (c.meta.level, c.meta.deg)) } else { None };
                        return Err(rerr(label, format!(
                            "{e} (step {i} -> {} {}, operands at {:?} / {:?} (compiled {:?} / {:?}), compiled output {:?}; {} metadata mismatches before it, first: {})",
                            names[st.var as usize], format!("{:?}", st.op).chars().take(160).collect::<String>(), m(st.a), m(st.b), st.a_meta.map(|x| (x.level, x.deg)), st.b_meta.map(|x| (x.level, x.deg)),
                            (st.meta.level, st.meta.deg), info.meta_mismatches, info.first_mismatches.join("; "))));
                    }
                    return Err(e);
                }
            }
            if let Some((ma, mb)) = pre_metas {
                let out = b.slots[b.steps[i].out as usize].meta;
                b.dyn_metas.insert(i as u32, (ma, mb, out));
            }
            if let Some((_, e)) = unit {
                unit_end = e;
            }
        }
        if let Some(s) = sched {
            if s.signal[i] {
                let side = rt.side.as_deref_mut().ok_or_else(|| rerr(&b.label, "schedule without a side lane"))?;
                let ev = side.ev.get(i).ok_or_else(|| rerr(&b.label, format!("step {i}: {} side events", side.ev.len())))?;
                if lane == 1 {
                    side.ex.compute.record(ev)?;
                } else {
                    rt.ex.compute.record(ev)?;
                }
            }
        }
        let st = &b.steps[i];
        if check {
            // a finish lands every output; other steps their one
            let outs: Vec<(u32, CtMeta)> = match &st.op {
                StepOp::Finish { f } => b.finishes.get(*f as usize).map(|fs| fs.outs.iter().copied().zip(fs.out_metas.iter().copied()).collect()).unwrap_or_default(),
                _ => vec![(st.out, st.meta)],
            };
            for (slot, want) in outs {
                let got = b.slots[slot as usize].meta;
                if meta_differs(&got, &want) {
                    info.meta_mismatches += 1;
                    if info.first_mismatches.len() < 8 {
                        info.first_mismatches.push(format!(
                            "{} ({:?}): ran to level {} deg {} nf 2^{:.4}, compiled level {} deg {} nf 2^{:.4}",
                            b.names[st.var as usize],
                            st.op,
                            got.level,
                            got.deg,
                            got.nf.log2(),
                            want.level,
                            want.deg,
                            want.nf.log2()
                        ));
                    }
                }
            }
        }
        if let Some(h) = hook.as_mut() {
            h(i, b, rt)?;
        }
    }
    if forked {
        join_side(rt)?;
    }
    Ok(())
}

/// A step that leaves the key-switching workspaces and the gather buffers alone (element-wise ops and rescales).
fn quiet(op: &StepOp) -> bool {
    use StepOp::*;
    matches!(op, Add | Sub | AddPt(_) | MultPt(_) | AddConst(_) | MultConst(_) | Negate | MultI | Rescale | DropTo(_))
}

/// Whether unit `u` can issue as one batch on the operands its slots hold: a K-batched product or refresh runs one
/// recipe for every member, so their operands must share level, degree and scale (else the steps issue one by one);
/// rotations only need one level.
fn unit_fits<W: DevWord>(u: &Unit, slots: &[Ct<'_, W>]) -> bool {
    let meta = |s: &u32| slots.get(*s as usize).map(|c| c.meta);
    let one = |v: &[u32], scale: bool| {
        let m0 = v.first().and_then(meta);
        v.iter().all(|s| meta(s).zip(m0).is_some_and(|(m, f)| m.level == f.level && (!scale || m.deg == f.deg && m.nf.to_bits() == f.nf.to_bits())))
    };
    match &u.kind {
        UnitKind::Grid { .. } | UnitKind::RotateSum { .. } => one(&u.ins, false),
        UnitKind::Mult { b } => one(&u.ins, true) && one(b, true),
        UnitKind::Refresh { .. } => one(&u.ins, true),
    }
}

/// Issue unit `u` (all its steps' outputs at once).
/// `stale`: a chain link whose gathered inputs and ModUp are not in the buffers it reads runs fresh.
#[allow(clippy::too_many_arguments)]
fn exec_unit<'a, W: DevWord, M: DevStrategy<W>>(
    block: &str,
    rt: &mut Rt<'_, 'a, W, M>,
    u: &Unit,
    steps: &[Step],
    slots: *mut Ct<'a, W>,
    n_slots: usize,
    gather: &mut Gather<'a, W>,
    held: Option<&mut Held<'a, W>>,
    groups: &[KeyGroup<'a>],
    relin: u32,
    lists: &[Vec<i64>],
    stale: bool,
) -> Result<()> {
    let span = &steps[u.lo as usize..u.hi as usize];
    if u.ins.iter().chain(span.iter().map(|st| &st.out)).any(|&s| s as usize >= n_slots) {
        return Err(rerr(block, format!("unit at step {}: slots outside the {n_slots} bound slots", u.lo)));
    }
    // SAFETY: bind formed the unit only over outputs in distinct slots none of which is an input's (a grid from
    // `Program::rotation_grids`, a batch from alloc's slots checked there): the inputs are shared references, every
    // output the only mutable one
    let xs: Vec<&Ct<'a, W>> = u.ins.iter().map(|&s| unsafe { &*slots.add(s as usize) }).collect();
    let kc = KsCfg { plans: rt.ks, source: rt.source };
    match &u.kind {
        UnitKind::Grid { steps: st, g, at, fresh, held: h } => {
            let fresh = *fresh || stale;
            if span.len() != at.len() {
                return Err(rerr(block, format!("grid at step {}: {} steps, {} members", u.lo, span.len(), at.len())));
            }
            let mut outs: Vec<(usize, usize, &mut Ct<'a, W>)> = at.iter().zip(span).map(|(&(c, s), x)| (c as usize, s as usize, unsafe { &mut *slots.add(x.out as usize) })).collect();
            let key = group(block, groups, *g)?;
            if *h {
                // a held chain's link: its own gather buffer, its ModUp in the workspace swapped in for the link
                let hd = held.ok_or_else(|| rerr(block, format!("held chain link at step {}: no held buffers", u.lo)))?;
                std::mem::swap(&mut rt.ex.ks.digits, &mut hd.digits);
                let gather = if xs.len() > 1 { Some(&mut hd.x) } else { None };
                let r = rotate::rotate_grid(rt.ctx, rt.ex, kc, rt.sh, key, &xs, gather, fresh, st, &mut outs);
                std::mem::swap(&mut rt.ex.ks.digits, &mut hd.digits);
                r?;
            } else {
                let gather = if xs.len() > 1 { Some(&mut gather.a) } else { None };
                rotate::rotate_grid(rt.ctx, rt.ex, kc, rt.sh, key, &xs, gather, fresh, st, &mut outs)?;
            }
        }
        UnitKind::RotateSum { l, g } => {
            let st = lists.get(*l as usize).ok_or_else(|| rerr(block, format!("step list {l} out of range")))?;
            let mut outs: Vec<&mut Ct<'a, W>> = span.iter().map(|x| unsafe { &mut *slots.add(x.out as usize) }).collect();
            rotate::rotate_sum_batch(rt.ctx, rt.ex, kc, rt.sh, group(block, groups, *g)?, &xs, &mut gather.a, st, &mut outs)?;
        }
        UnitKind::Mult { b } => {
            // the operands gathered into K-ciphertext buffers, one product of K ciphertexts (one relinearization key
            // read for all of them), the products scattered with their own metadata
            if b.iter().any(|&s| s as usize >= n_slots) {
                return Err(rerr(block, format!("product unit at step {}: slots outside the {n_slots} bound slots", u.lo)));
            }
            let ys: Vec<&Ct<'a, W>> = b.iter().map(|&s| unsafe { &*slots.add(s as usize) }).collect();
            let (k, s) = (xs.len(), &rt.ex.compute);
            for (c, x) in xs.iter().enumerate() {
                batch::put(rt.ctx, s, &mut gather.a, c, x)?;
            }
            for (c, y) in ys.iter().enumerate() {
                batch::put(rt.ctx, s, &mut gather.b, c, y)?;
            }
            gather.a.buf.k = k;
            gather.a.meta = xs[0].meta;
            gather.out.buf.k = k;
            let key = group(block, groups, relin)?;
            let Gather { a, b: gb, out } = gather;
            if ys.is_empty() {
                rt.ops.square(rt.ctx, rt.ex, kc, rt.sh, key, out, a)?;
            } else {
                gb.buf.k = k;
                gb.meta = ys[0].meta;
                rt.ops.mult(rt.ctx, rt.ex, kc, rt.sh, key, out, a, gb)?;
            }
            for (c, st) in span.iter().enumerate() {
                let r = if ys.is_empty() { rt.sh.square(&xs[c].meta) } else { rt.sh.mult(&xs[c].meta, &ys[c].meta) };
                let m = r.map_err(|e| rerr(block, e.to_string()))?.out;
                // SAFETY: as the unit's other outputs (distinct slots, none an operand's)
                batch::take(rt.ctx, &rt.ex.compute, unsafe { &mut *slots.add(st.out as usize) }, out, c, m)?;
            }
        }
        UnitKind::Refresh { route, kind } => {
            // the inputs gathered into one K-ciphertext buffer, refreshed in place as one K-batch, scattered back
            let r = rt.routes.get(*kind).ok_or_else(|| rerr(block, format!("route {kind:?} not held")))?;
            for (c, x) in xs.iter().enumerate() {
                batch::put(rt.ctx, &rt.ex.compute, &mut gather.a, c, x)?;
            }
            gather.a.buf.k = xs.len();
            gather.a.meta = xs[0].meta;
            let mut cx = BtsCx { ctx: rt.ctx, ex: &mut *rt.ex, sh: rt.sh, ks: rt.ks, ops: &mut *rt.ops, work: &mut *rt.work };
            r.run(&mut cx, std::slice::from_mut(&mut gather.a), route)?;
            let out = gather.a.meta;
            for (c, st) in span.iter().enumerate() {
                // each member's own landing (its tag), at the batch's level, degree and scale
                let m = r.landing(rt.sh, &xs[c].meta, route).ok().filter(|m| m.level == out.level && m.deg == out.deg).map_or(out, |m| CtMeta { nf: out.nf, ..m });
                batch::take(rt.ctx, &rt.ex.compute, unsafe { &mut *slots.add(st.out as usize) }, &gather.a, c, m)?;
            }
        }
    }
    Ok(())
}

/// A dynamic step between two graph launches: the launches did not move the host-side metadata of the slots they
/// wrote, so its operands get their compiled metadata first.
fn issue_dynamic<'a, W: DevWord, M: DevStrategy<W>>(rt: &mut Rt<'_, 'a, W, M>, b: &mut Bound<'a, W>, i: usize) -> Result<()> {
    let st = b.steps[i].clone();
    // the metadata the step's operands had in the eager run the graphs were captured after
    let (ma, mb, _) = b.dyn_metas.get(&(i as u32)).copied().ok_or_else(|| rerr(&b.label, format!("dynamic step {i} never ran eagerly")))?;
    for (s, m) in [(st.a, ma), (st.b, mb)] {
        if let Some(m) = m {
            if s != NONE && (s as usize) < b.slots.len() {
                b.slots[s as usize].meta = m;
            }
        }
    }
    let mut scratch = RunInfo { served: Served::Eager, host_us: 0.0, launch_us: 0.0, meta_mismatches: 0, first_mismatches: vec![] };
    issue_range(rt, b, i, i + 1, false, &mut scratch, None, None)
}

/// The static segments of `b` (maximal step ranges without a dynamic step) and the dynamic steps, in order.
fn segments<W: DevWord>(b: &Bound<'_, W>) -> Vec<std::result::Result<(usize, usize), usize>> {
    let mut out = vec![];
    let mut lo = 0;
    for &d in &b.dynamic {
        let d = d as usize;
        if d > lo {
            out.push(Ok((lo, d)));
        }
        out.push(Err(d));
        lo = d + 1;
    }
    if lo < b.steps.len() {
        out.push(Ok((lo, b.steps.len())));
    }
    out
}

impl Replayer {
    pub fn new(cfg: ReplayCfg) -> Self {
        Replayer { cfg, stats: ReplayStats::default(), side_lane: true, scheds: Default::default() }
    }

    /// Run `b` once on the inputs the driver wrote into its input slots: eager, or (graphs) the graph captured for
    /// the block's current plaintext rows (those of the last `Bound::resolve_pts`), or, when none matches, eager +
    /// capture (at most `ReplayCfg::graphs_per_block` layouts per block; the oldest is dropped beyond). Asynchronous:
    /// the outputs are ready on `compute` (the caller syncs or records an event).
    /// The lane schedule of `b` (None: no side lane in use).
    pub fn schedule<'a, W: DevWord, M: DevStrategy<W>>(&mut self, b: &Bound<'a, W>, rt: &Rt<'_, 'a, W, M>) -> Option<Sched> {
        if !self.side_lane || rt.side.is_none() {
            return None;
        }
        let dynamic: Vec<u32> = b.dynamic.iter().copied().collect();
        if let Some((d, s)) = self.scheds.get(&b.label) {
            if *d == dynamic {
                return Some(s.clone());
            }
        }
        // the side lane's key switches must read their keys in place (staging goes through compute's buffers)
        let groups = &b.groups;
        let relin = b.relin;
        let side_ok = |st: &Step| -> bool {
            let ok = |g: u32| groups.get(g as usize).is_some_and(|k| k.in_place() || !k.needs_stage());
            match &st.op {
                StepOp::Mult | StepOp::Square => ok(relin),
                StepOp::Rotate { g, .. } if *g == NONE => true,
                StepOp::Rotate { g, .. } | StepOp::RotateSum { g, .. } | StepOp::Conjugate { g } => ok(*g),
                _ => true,
            }
        };
        let s = schedule::plan(b, &side_ok);
        self.scheds.insert(b.label.clone(), (dynamic, s.clone()));
        Some(s)
    }

    pub fn run<'a, W: DevWord, M: DevStrategy<W>>(&mut self, b: &mut Bound<'a, W>, rt: &mut Rt<'_, 'a, W, M>) -> Result<RunInfo> {
        let t0 = Instant::now();
        let mut info = RunInfo { served: Served::Eager, host_us: 0.0, launch_us: 0.0, meta_mismatches: 0, first_mismatches: vec![] };
        if b.pts.iter().any(|p| !p.stream && p.meta.is_none()) {
            return Err(rerr(&b.label, "plaintexts not resolved (Bound::resolve_pts after staging)"));
        }
        let sched = self.schedule(b, rt);
        let sched = sched.as_ref();
        if !self.cfg.use_graphs {
            issue(rt, b, self.cfg.check_meta, &mut info, None, sched)?;
            self.stats.eager += 1;
            info.host_us = t0.elapsed().as_secs_f64() * 1e6;
            return Ok(info);
        }
        let dynamic: Vec<u32> = b.dynamic.iter().copied().collect();
        let hit = b.graphs.iter().position(|g| g.dynamic == dynamic && g.stream_base == b.stream_base && g.pt_rows.len() == b.pts.len() && g.pt_rows.iter().zip(&b.pts).all(|(r, p)| *r == p.rows));
        if let Some(gi) = hit {
            // the dynamic steps need the Bound mutably: take the captured block out while it replays
            let g = b.graphs.remove(gi);
            let mut res = Ok(());
            for it in &g.items {
                res = match it {
                    Item::Graph(exec) => {
                        let tl = Instant::now();
                        let r = exec.launch(&rt.ex.compute).map_err(Error::from);
                        info.launch_us += tl.elapsed().as_secs_f64() * 1e6;
                        r
                    }
                    Item::Step(i) => issue_dynamic(rt, b, *i),
                };
                if res.is_err() {
                    break;
                }
            }
            for (i, (_, s, _)) in b.outputs.iter().enumerate() {
                b.slots[*s as usize].meta = g.out_metas[i];
            }
            b.graphs.insert(gi, g);
            res?;
            self.stats.replays += 1;
            info.served = Served::Replayed;
            info.host_us = t0.elapsed().as_secs_f64() * 1e6;
            return Ok(info);
        }
        if b.graphs.len() >= self.cfg.graphs_per_block.max(1) {
            self.drop_graph(b, rt, 0)?;
            self.stats.invalidations += 1;
        }
        // first run with these rows: eager (this token's result, warms the caches), then capture without executing
        let in_metas: Vec<CtMeta> = b.inputs.iter().map(|(_, s, _)| b.slots[*s as usize].meta).collect();
        issue(rt, b, self.cfg.check_meta, &mut info, None, sched)?;
        let out_metas: Vec<CtMeta> = b.outputs.iter().map(|(_, s, _)| b.slots[*s as usize].meta).collect();
        info.host_us = t0.elapsed().as_secs_f64() * 1e6;
        let tc = Instant::now();
        // the eager outputs must not be overwritten by anything the capture records (nothing executes in a capture,
        // but the lease starts with a lane sync, which also orders the outputs before the caller reads them)
        rt.ops.ring.lease_begin(&rt.ex.compute)?;
        if let Some(s) = rt.side.as_deref_mut() {
            s.ops.ring.lease_begin(&s.ex.compute)?;
        }
        for (i, (_, s, _)) in b.inputs.iter().enumerate() {
            b.slots[*s as usize].meta = in_metas[i];
        }
        rt.locks.iter().for_each(|l| l.set_locked(true));
        let sp: *const Stream<Compute> = &rt.ex.compute;
        // SAFETY: the executor's compute stream outlives the call; the capture only begins / ends on it while the
        // steps issue their work on the same stream through `rt`
        let s = unsafe { &*sp };
        let mut items: Vec<Item> = vec![];
        let mut failed: Option<Error> = None;
        for seg in segments(b) {
            match seg {
                Ok((lo, hi)) => {
                    let mut err: Option<Error> = None;
                    let mut scratch = RunInfo { served: Served::Eager, host_us: 0.0, launch_us: 0.0, meta_mismatches: 0, first_mismatches: vec![] };
                    let r = s.capture(|_| {
                        issue_range(rt, b, lo, hi, false, &mut scratch, None, sched).map_err(|e| {
                            err = Some(e);
                            pc_cuda::Error::Unavailable("replay step under capture")
                        })
                    });
                    match (r, err) {
                        (_, Some(e)) => failed = Some(e),
                        (Err(e), None) => failed = Some(e.into()),
                        (Ok((exec, ())), None) => items.push(Item::Graph(exec)),
                    }
                    if failed.is_some() {
                        break;
                    }
                }
                Err(d) => {
                    // not issued under capture: its output takes the metadata the eager run produced
                    let out = b.steps[d].out as usize;
                    let m = b.dyn_metas.get(&(d as u32)).map(|x| x.2).ok_or_else(|| rerr(&b.label, format!("dynamic step {d} never ran eagerly")));
                    match m {
                        Ok(m) => b.slots[out].meta = m,
                        Err(e) => {
                            failed = Some(e);
                            break;
                        }
                    }
                    items.push(Item::Step(d));
                }
            }
        }
        rt.locks.iter().for_each(|l| l.set_locked(false));
        let lease = rt.ops.ring.lease_end();
        let side_lease = rt.side.as_deref_mut().map(|s| s.ops.ring.lease_end()).unwrap_or_default();
        let release_all = |rt: &mut Rt<'_, 'a, W, M>| {
            rt.ops.ring.release(&lease);
            if let Some(s) = rt.side.as_deref_mut() {
                s.ops.ring.release(&side_lease);
            }
        };
        if let Some(e) = failed {
            release_all(rt);
            return Err(e);
        }
        // the capture left the eager outputs' metadata on the slots again (same steps); keep the recorded ones
        let cap_metas: Vec<CtMeta> = b.outputs.iter().map(|(_, s, _)| b.slots[*s as usize].meta).collect();
        if cap_metas.iter().zip(&out_metas).any(|(a, b)| meta_differs(a, b)) {
            release_all(rt);
            return Err(rerr(&b.label, "captured program lands its outputs at other metadata than its eager run (non-deterministic step)"));
        }
        let capture_ms = tc.elapsed().as_secs_f64() * 1e3;
        self.stats.captures += 1;
        self.stats.capture_ms += capture_ms;
        b.graphs.push(Captured { items, dynamic, lease, side_lease, out_metas, pt_rows: b.pts.iter().map(|p| p.rows).collect(), stream_base: b.stream_base, capture_ms });
        info.served = Served::Captured;
        Ok(info)
    }

    /// One eager run with `hook` called after every step (debugging: per-step decryptions, metadata traces).
    pub fn eager_with<'a, W: DevWord, M: DevStrategy<W>>(&mut self, b: &mut Bound<'a, W>, rt: &mut Rt<'_, 'a, W, M>, hook: &mut StepHook<'_, 'a, W, M>) -> Result<RunInfo> {
        let t0 = Instant::now();
        let mut info = RunInfo { served: Served::Eager, host_us: 0.0, launch_us: 0.0, meta_mismatches: 0, first_mismatches: vec![] };
        issue(rt, b, true, &mut info, Some(hook), None)?;
        self.stats.eager += 1;
        info.host_us = t0.elapsed().as_secs_f64() * 1e6;
        Ok(info)
    }

    /// Drop graph `i` of `b` (after the lane drained: a launch may still read its constant slots) and return its
    /// constant slots to the ring.
    pub fn drop_graph<'a, W: DevWord, M: DevStrategy<W>>(&mut self, b: &mut Bound<'a, W>, rt: &mut Rt<'_, 'a, W, M>, i: usize) -> Result<()> {
        if i < b.graphs.len() {
            let g = b.graphs.remove(i);
            rt.ex.compute.sync()?;
            rt.ops.ring.release(&g.lease);
            if let Some(s) = rt.side.as_deref_mut() {
                s.ops.ring.release(&g.side_lease);
            }
            drop(g.items);
        }
        Ok(())
    }

    /// Drop every graph of `b`.
    pub fn drop_graphs<'a, W: DevWord, M: DevStrategy<W>>(&mut self, b: &mut Bound<'a, W>, rt: &mut Rt<'_, 'a, W, M>) -> Result<()> {
        while !b.graphs.is_empty() {
            self.drop_graph(b, rt, 0)?;
        }
        Ok(())
    }
}
