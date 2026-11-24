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
use crate::bind::{BoundPt, Bound, CoeffSumStep, DynMetas, FinishStep, Gather, Held, Step, StepOp, Unit, UnitKind, NONE};
use crate::schedule::{self, Sched};
use crate::error::{Error, Result};
use pc_core::encode::Complex;
use pc_cuda::{launch_result, Compute, Event, GraphExec, Stream};
use pc_gpu::bts::staged::{BtsCall, Phase, StagedRef};
use pc_gpu::bts::stc_first::{nf_key, StcFirstRoute};
use pc_gpu::matmul::ccmm::{self as ccmm_eng, Ccmm};
use pc_gpu::matmul::pcmm::{GroupOffload, Pcmm, PcmmLinear, PcmmShape};
use pc_gpu::limbs::MonomialTable;
use pc_gpu::bts::{batch, BtsCx, BtsRoute, BtsWork, RouteKind, Staged, StcHigh};
use pc_gpu::encode_dev::CoeffRowsDevice;
use pc_gpu::ks::{KeyGroup, KsPlans, KskSource};
use pc_gpu::ops::{Ops, Pt};
use pc_gpu::rotate::{self, KsCfg};
use pc_gpu::{Context, Ct, DevStrategy, DevWord, Executor};
use pc_shadow::{CtMeta, Shadow, PtMeta};
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
///
/// `remote`: the run's remote refresh device, if any (a finish may hand it part of its output refreshes).
pub trait FinishExec<'a, W: DevWord, M: DevStrategy<W>> {
    fn finish(&mut self, cx: &mut BtsCx<'_, 'a, W, M>, f: &FinishStep, slots: &mut [Ct<'a, W>], remote: Option<&mut dyn RemoteLane<'a, W, M>>) -> Result<()>;
    /// Ownership (PC_OWN): the output slots of `f` this service refreshes on a remote device (with one) and can leave in
    /// the replica's slots there too (`RemoteLane::own_next`); the replay copies over any it did not.
    fn owned(&self, _f: &FinishStep) -> Vec<u32> {
        vec![]
    }
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
    /// A second device for refresh batches (`RemoteLane`); None = one device.
    pub remote: Option<&'r mut dyn RemoteLane<'a, W, M>>,
}

/// A second device that refreshes part of a refresh batch (`UnitKind::Refresh` of K >= 2): `issue` sends the
/// members it is handed there (forked from this device's compute stream: inputs copied over, refreshed with that
/// device's copy of the route, outputs copied back into their slots, metadata set as a local refresh would), `join`
/// makes this device's compute stream wait for all of it. Between the two, this device refreshes its own members, so
/// the halves overlap. Everything is ordered by events, so a block captured into a graph keeps the split inside it.
pub trait RemoteLane<'a, W: DevWord, M: DevStrategy<W>> {
    /// Whether the remote device holds route family `kind`.
    fn has(&self, kind: RouteKind) -> bool;
    /// Issue the remote half: `xs[c]` refreshed into `outs[c]` (this device's memory) on route `route`, ordered after
    /// everything issued on `here` so far.
    fn issue(&mut self, here: &Stream<Compute>, sh: &Shadow, kind: RouteKind, route: &pc_shadow::Route, xs: &[&Ct<'a, W>], outs: &mut [&mut Ct<'a, W>]) -> pc_gpu::Result<()>;
    /// Order `here` after the issued remote half.
    fn join(&mut self, here: &Stream<Compute>) -> pc_gpu::Result<()>;
    /// Whether the remote device holds the StC-first route the PCMM finishes refresh on.
    fn has_finish(&self) -> bool {
        false
    }
    /// Refresh `x` (this device's memory: a finish group at the split of the StC-first call `call`) to `Done` on the
    /// remote device's StC-first route, written back into `x`, ordered after everything issued on `here` so far
    /// (`join` orders `here` after it).
    fn finish_group(&mut self, _here: &Stream<Compute>, _sh: &Shadow, _call: &BtsCall<'_, 'a, W, M>, _x: &mut Ct<'a, W>) -> pc_gpu::Result<()> {
        Err(pc_gpu::Error::Mem { knob: "remote finish", value: 0, msg: "this remote lane runs no finish groups".into() })
    }
    /// The StC-first input stage of `call` (at `Begun`: a PCMM linear's input call) for `xs` (this device's memory) on
    /// the remote device, in batches of at most `GRID_INPUTS`, the results back into `xs` at the split, ordered after
    /// everything issued on `here` so far (`join` orders `here` after it).
    fn stc_inputs(&mut self, _here: &Stream<Compute>, _sh: &Shadow, _call: &BtsCall<'_, 'a, W, M>, _xs: &mut [Ct<'a, W>]) -> pc_gpu::Result<()> {
        Err(pc_gpu::Error::Mem { knob: "remote stc", value: 0, msg: "this remote lane runs no input stages".into() })
    }
    /// Joint linears (`pc_gpu::matmul::pcmm::linear_joint`): whether the remote device holds its planes of `lin`'s far
    /// outputs (its own PCMM engine).
    fn joint_holds(&self, _lin: &str) -> bool {
        false
    }
    /// Joint: the input stage of `call` for inputs `first..` (`xs`) on the remote device, kept in its PCMM inputs and
    /// copied back into `xs` (`join` orders `here` after it).
    fn joint_stc(&mut self, _here: &Stream<Compute>, _sh: &Shadow, _call: &BtsCall<'_, 'a, W, M>, _first: usize, _xs: &mut [Ct<'a, W>]) -> pc_gpu::Result<()> {
        Err(no_joint())
    }
    /// Joint: this device's inputs `0..xs.len()` at the split into the remote device's PCMM inputs.
    fn joint_inputs(&mut self, _here: &Stream<Compute>, _xs: &[Ct<'a, W>]) -> pc_gpu::Result<()> {
        Err(no_joint())
    }
    /// Joint: the far outputs' product and finishes on the remote device, copied back into `outs`.
    #[allow(clippy::too_many_arguments)]
    fn joint_far(&mut self, _here: &Stream<Compute>, _sh: &Shadow, _lin: &PcmmLinear, _sub: &PcmmShape, _call: &BtsCall<'_, 'a, W, M>, _nf_out: f64, _groups: &[Vec<usize>], _drops: &[u32], _outs: &mut [Ct<'a, W>]) -> pc_gpu::Result<()> {
        Err(no_joint())
    }
    /// Row split (`pc_gpu::matmul::pcmm::linear_rows`): whether the remote device holds its planes of `lin`'s last inputs.
    fn rows_holds(&self, _lin: &str) -> bool {
        false
    }
    /// Rows: the input stage of `call` for `xs` (this device's memory) on the remote device, kept in its PCMM inputs
    /// `0..` (copied back into `xs` too when `back`), ordered after everything issued on `here` so far.
    fn rows_stc(&mut self, _here: &Stream<Compute>, _sh: &Shadow, _call: &BtsCall<'_, 'a, W, M>, _xs: &mut [Ct<'a, W>], _back: bool) -> pc_gpu::Result<()> {
        Err(no_joint())
    }
    /// Rows: the remote device's partial product of its staged inputs on its planes of `lin` (sub-shape `sub`), its
    /// buffers returned (peer memory); `rows_wait` orders a stream of this device after it.
    fn rows_product(&mut self, _lin: &PcmmLinear, _sub: &PcmmShape) -> pc_gpu::Result<(*const u32, *const u32)> {
        Err(no_joint())
    }
    fn rows_wait(&mut self, _here: &Stream<Compute>) -> pc_gpu::Result<()> {
        Err(no_joint())
    }
    /// Rows: this device's partial added remotely for the remote outputs, their output stage and finishes there,
    /// copied back into `outs` (`join` orders `here` after it).
    #[allow(clippy::too_many_arguments)]
    fn rows_far(&mut self, _here: &Stream<Compute>, _sh: &Shadow, _sub: &PcmmShape, _call: &BtsCall<'_, 'a, W, M>, _here_partial: (*const u32, *const u32), _far_outs: &[usize], _meta: &CtMeta, _nf_out: f64, _groups: &[Vec<usize>], _drops: &[u32], _outs: &mut [Ct<'a, W>]) -> pc_gpu::Result<()> {
        Err(no_joint())
    }
    /// `stc_high` (a CCMM's operand conversion, spec `spec`) of `xs` (this device's memory, one ciphertext each, all at
    /// one metadata) on the remote device's StC-first route, in batches of at most `GRID_INPUTS`, the results back into
    /// `xs`, ordered after everything issued on `here` so far (`join` orders `here` after it).
    fn stc_high_far(&mut self, _here: &Stream<Compute>, _sh: &Shadow, _xs: &mut [Ct<'a, W>], _spec: &StcHigh) -> pc_gpu::Result<()> {
        Err(pc_gpu::Error::Mem { knob: "remote stc_high", value: 0, msg: "this remote lane runs no operand conversions".into() })
    }
    /// Refresh the single ciphertexts `xs` (this device's memory, at most `GRID_INPUTS`) on the remote device: gathered
    /// into its batch buffer, `f` run there with its StC-first route, scattered back into `xs` at the batch's
    /// metadata, ordered after everything issued on `here` so far (`join` orders `here` after it). The CCMM's output
    /// refreshes.
    fn stcf_singles(&mut self, _here: &Stream<Compute>, _sh: &Shadow, _xs: &mut [&mut Ct<'a, W>], _f: &mut StcfFn<'_, 'a, W, M>) -> pc_gpu::Result<()> {
        Err(pc_gpu::Error::Mem { knob: "remote finish", value: 0, msg: "this remote lane runs no finish refreshes".into() })
    }
    /// Split CCMM products (PC_CCMM_SPLIT): whether the remote device holds a CCMM engine of its own.
    fn ccmm_split_on(&self) -> bool {
        false
    }
    /// A split product's operand conversion: `stc_high` (spec `spec`) of `xs` (this device's memory: ciphertexts
    /// 8 .. 16 of operand `side`, 0 = left, 1 = right) on the remote device, kept there for `ccmm_mul_split`; `back`:
    /// copied back into `xs` too (the word checks). `xs` take the conversion's metadata either way.
    fn ccmm_operands_far(&mut self, _here: &Stream<Compute>, _sh: &Shadow, _side: usize, _xs: &mut [Ct<'a, W>], _spec: &StcHigh, _back: bool) -> pc_gpu::Result<()> {
        Err(no_split())
    }
    /// The product split across both devices (`pc_gpu::matmul::ccmm::mul_split`, in place over `a`) with this
    /// device's engine `eng` on `here` (context `ctx`): this device takes ciphertexts 0 .. 8 of `a` and `b`, the remote
    /// one its operands of `ccmm_operands_far`. Outputs 0 .. 8 land in `a[..8]`, outputs 8 .. 16 stay on the remote
    /// device (`a[8..]` take their metadata) until `ccmm_far_back` / `ccmm_far_singles`.
    fn ccmm_mul_split(&mut self, _here: &Stream<Compute>, _eng: &mut Ccmm<'a, 'static, W>, _ctx: &Context<'a, W, M>, _a: &mut [Ct<'a, W>], _b: &[Ct<'a, W>], _bt: bool) -> pc_gpu::Result<()> {
        Err(no_split())
    }
    /// The remote outputs merged in groups of `g` (o0 + X^1024 o1 + ..: `merge_monomial` with the remote table), the
    /// groups' results moved to the front (as the service does with its own).
    fn ccmm_far_merge(&mut self, _g: usize) -> pc_gpu::Result<()> {
        Err(no_split())
    }
    /// Remote outputs `first .. first + xs.len()` (after `ccmm_far_merge`: the groups' results) into `xs` (this
    /// device's memory) on this device's copy stream, `here` ordered after.
    fn ccmm_far_back(&mut self, _here: &Stream<Compute>, _first: usize, _xs: &mut [Ct<'a, W>]) -> pc_gpu::Result<()> {
        Err(no_split())
    }
    /// `stcf_singles` of the remote outputs `first .. first + xs.len()` where they are (no copy over); `xs` receive the
    /// results as `stcf_singles` gives them back.
    fn ccmm_far_singles(&mut self, _here: &Stream<Compute>, _sh: &Shadow, _first: usize, _xs: &mut [&mut Ct<'a, W>], _f: &mut StcfFn<'_, 'a, W, M>) -> pc_gpu::Result<()> {
        Err(no_split())
    }
    /// PC_CCMM_PHASES: the last split product's phase times per device (ms: ring in, CMT, PPMM, CMT + relinearization,
    /// ring out, total; host-waits for both devices).
    fn ccmm_split_phases(&mut self) -> pc_gpu::Result<Option<[[f64; 6]; 2]>> {
        Ok(None)
    }
    /// The asynchronous lane (`schedule` lanes 2 and 3, PC_REMOTE_ASYNC=1): whole refreshes, each in two parts ordered
    /// by events: `lane_issue` copies the inputs into a ring buffer of the remote device and refreshes them there
    /// (event `lane_event_in` once the inputs are copied), `lane_finish` copies the result back into this device's
    /// slots on this device's copy-back stream (event `lane_event`). Whether this lane offers it.
    fn lane_ok(&self) -> bool {
        false
    }
    /// Fork both parts from `here`: what they issue next comes after everything issued on `here` so far.
    fn lane_fork(&mut self, _here: &Stream<Compute>) -> pc_gpu::Result<()> {
        Err(no_lane())
    }
    /// Order the remote part after `ev` (an event of this device's lanes or of the copy-back stream).
    fn lane_wait(&mut self, _ev: &Event) -> pc_gpu::Result<()> {
        Err(no_lane())
    }
    /// Order the copy-back stream after `ev` (an event of this device's lanes or of an input copy).
    fn lane_out_wait(&mut self, _ev: &Event) -> pc_gpu::Result<()> {
        Err(no_lane())
    }
    /// The remote part of steps `ids` (one refresh, or a refresh pair's two steps): `srcs` (one per step) copied into
    /// the next ring entry and refreshed there on `routes` (one each), each step's input-copy event recorded.
    fn lane_issue(&mut self, _ids: &[usize], _sh: &Shadow, _kind: RouteKind, _routes: &[pc_shadow::Route], _srcs: &[&Ct<'a, W>]) -> pc_gpu::Result<()> {
        Err(no_lane())
    }
    /// The copy back of steps `ids` (after their `lane_issue`): the ring entry's results into `outs` with their
    /// landings' metadata, after the refresh and what `lane_out_wait` ordered, then each step's event.
    fn lane_finish(&mut self, _ids: &[usize], _outs: &mut [&mut Ct<'a, W>]) -> pc_gpu::Result<()> {
        Err(no_lane())
    }
    /// Step `i`'s output copied back (`lane_finish`).
    fn lane_event(&self, _i: usize) -> Option<&Event> {
        None
    }
    /// Step `i`'s inputs copied over (`lane_issue`).
    fn lane_event_in(&self, _i: usize) -> Option<&Event> {
        None
    }
    /// Order `here` after everything issued on both parts.
    fn lane_join(&mut self, _here: &Stream<Compute>) -> pc_gpu::Result<()> {
        Err(no_lane())
    }
    /// Wait on the host for everything issued on both parts (PC_REMOTE_CHECK).
    fn lane_sync(&mut self) -> pc_gpu::Result<()> {
        Err(no_lane())
    }
    /// The replica lane (`replica`, PC_REPLICA): whether the remote device holds this block's second binding.
    fn replica_on(&self) -> bool {
        false
    }
    /// The route families the replica's steps may refresh on (none in v1: its steps are element-wise and key switches).
    fn replica_holds(&self, _kind: RouteKind) -> bool {
        false
    }
    /// Whether the replica holds a copy of plaintext `p` of the block (the plaintext mirror, PC_REPLICA_PT).
    fn replica_pt(&self, _p: u32) -> bool {
        false
    }
    /// Set the metadata of the replica's slot `s` (an owned output: this device's slot's after the step).
    fn replica_set_meta(&mut self, _s: u32, _meta: CtMeta) -> Result<()> {
        Err(no_replica())
    }
    /// Ownership (PC_OWN): the slots the replica's plan expects the step about to run (a finish, a split refresh batch)
    /// to leave in the replica's slots too; `own_take` after the step returns those it did.
    fn own_begin(&mut self, _slots: &[u32]) {}
    /// The slots expected since `own_begin` (empty: no ownership for this step).
    fn own_wanted(&self) -> &[u32] {
        &[]
    }
    /// The next remote service call's outputs, in their order there: the slot each lands in here, and the power of
    /// two (exponent, 0 = none) this device multiplies them by afterwards (the replica's copies take it too).
    fn own_next(&mut self, _slots: Vec<u32>, _shift: u32) {}
    /// The slots left in the replica's slots since `own_begin` (and the expectation cleared).
    fn own_take(&mut self) -> Vec<u32> {
        vec![]
    }
    /// Fork the replica's stream from `here` (a region's start): what it issues next comes after everything issued
    /// on `here` so far.
    fn replica_fork(&mut self, _here: &Stream<Compute>) -> Result<()> {
        Err(no_replica())
    }
    /// Order `here` after everything issued on the replica's stream (a region's end).
    fn replica_join(&mut self, _here: &Stream<Compute>) -> Result<()> {
        Err(no_replica())
    }
    /// Order the replica's stream after `ev` (an event of this device).
    fn replica_wait(&mut self, _ev: &Event) -> Result<()> {
        Err(no_replica())
    }
    /// PC_TL (diagnostic): a timestamp kernel on the replica's stream writing `buf[idx]` (mapped host memory).
    fn replica_stamp(&mut self, _buf: *mut u64, _idx: i32) -> Result<()> {
        Ok(())
    }
    /// PC_REPLICA_DMA: fork both copy streams (this device's, the replica's) from `here` (a region's first early copy:
    /// what they issue next comes after everything issued on `here` so far).
    fn dma_fork(&mut self, _here: &Stream<Compute>) -> Result<()> {
        Err(no_replica())
    }
    /// Order `here` after everything issued on both copy streams (a region's end).
    fn dma_join(&mut self, _here: &Stream<Compute>) -> Result<()> {
        Err(no_replica())
    }
    /// Order device `to`'s copy stream after `ev` (an event of either device).
    fn dma_wait(&mut self, _to: u8, _ev: &Event) -> Result<()> {
        Err(no_replica())
    }
    /// Copy `c` of the replica plan on device `to`'s copy stream (DMA over the bus): `here_slot` (this device's slot `s`)
    /// into the replica's slot `s` (`to` 1), or the replica's slot `s` into it (`to` 0), metadata included; then copy
    /// `c`'s event there.
    fn dma_copy(&mut self, _c: usize, _to: u8, _s: u32, _here_slot: &mut Ct<'a, W>) -> Result<()> {
        Err(no_replica())
    }
    /// Copy `c`'s event (recorded by `dma_copy` on device `to`).
    fn dma_event(&self, _c: usize, _to: u8) -> Option<&Event> {
        None
    }
    /// Copy `src` (this device's slot `s`, metadata included) into the replica's slot `s`, on the replica's stream.
    /// The replica's slot `s` for writing from this device (a peer write on this device's stream).
    fn replica_slot_ptr(&mut self, _s: u32) -> Option<*mut Ct<'a, W>> {
        None
    }
    fn replica_copy_in(&mut self, _s: u32, _src: &Ct<'a, W>) -> Result<()> {
        Err(no_replica())
    }
    /// The replica's slot `s` (a copy back reads it on this device's stream, after its producer's event).
    fn replica_slot(&self, _s: u32) -> Option<&Ct<'a, W>> {
        None
    }
    /// Issue step `i` (or the unit (index, end) it starts) of the replica's binding on the replica's stream, then
    /// record step `signal`'s event there (`i`'s, or a unit issued step by step: its first step's, after its last).
    fn replica_issue(&mut self, _sh: &Shadow, _source: KskSource, _i: usize, _unit: Option<(u32, usize)>, _signal: Option<usize>) -> Result<()> {
        Err(no_replica())
    }
    /// Step `i`'s event on the replica's stream (recorded by `replica_issue` with `signal`).
    fn replica_event(&self, _i: usize) -> Option<&Event> {
        None
    }
    /// The remote op layer's constant ring around a capture, as this device's (`ConstRing::lease_begin` / `lease_end`
    /// / `release`): what the captured remote kernels read stays out of the rotation until the graph is dropped.
    fn ring_lease_begin(&mut self) -> pc_gpu::Result<()> {
        Ok(())
    }
    fn ring_lease_end(&mut self) -> pc_gpu::Result<Vec<usize>> {
        Ok(vec![])
    }
    fn ring_release(&mut self, _pushes: &[usize]) {}
}

fn no_joint() -> pc_gpu::Error {
    pc_gpu::Error::Mem { knob: "remote pcmm", value: 0, msg: "this remote device has no PCMM engine (joint linears)".into() }
}

fn no_replica() -> Error {
    Error::Replay { block: String::new(), msg: "this remote device has no replica lane (PC_REPLICA)".into() }
}

fn no_lane() -> pc_gpu::Error {
    pc_gpu::Error::Mem { knob: "remote lane", value: 0, msg: "this remote device offers no asynchronous lane".into() }
}

/// A refresh run on a remote device's StC-first route (`RemoteLane::stcf_singles`), on its gathered batch.
pub type StcfFn<'f, 'a, W, M> = dyn FnMut(&mut BtsCx<'_, 'a, W, M>, &StcFirstRoute<'a, W, M>, &mut Ct<'a, W>) -> pc_gpu::Result<()> + 'f;

/// Ciphertexts per batch of a remote StC-first stage: the gather buffer's `GRID_INPUTS`, at most the route's
/// `BtsSetup::k_max` (what its stages' work buffers hold).
fn stcf_batch<W: DevWord, M: DevStrategy<W>>(r: &StcFirstRoute<'_, W, M>) -> usize {
    crate::program::GRID_INPUTS.min(r.setup.k_max).max(1)
}

/// A PCMM finish's offload (`pc_gpu::matmul::pcmm::GroupOffload`) over the run's remote lane: the linear's last
/// half of finish groups refreshed on the remote device.
pub struct FarGroups<'x, 'a, W: DevWord, M: DevStrategy<W>> {
    pub lane: &'x mut dyn RemoteLane<'a, W, M>,
    /// How many of the finish's groups (the last ones) go there; None = the last half.
    pub n_far: Option<usize>,
}

impl<'x, 'a, W: DevWord, M: DevStrategy<W>> GroupOffload<'a, W, M> for FarGroups<'x, 'a, W, M> {
    fn issue(&mut self, here: &Stream<Compute>, sh: &Shadow, call: &BtsCall<'_, 'a, W, M>, x: &mut Ct<'a, W>) -> pc_gpu::Result<()> {
        self.lane.finish_group(here, sh, call, x)
    }
    fn join(&mut self, here: &Stream<Compute>) -> pc_gpu::Result<()> {
        self.lane.join(here)
    }
    fn check(&self) -> bool {
        remote_check()
    }
    fn checked(&mut self) {
        REMOTE_FIN_CHECKED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    fn far(&self, total: usize) -> usize {
        self.n_far.unwrap_or(total / 2)
    }
    fn stc_far(&self, n: usize) -> usize {
        if stc_split() && self.lane.has_finish() {
            n / 2
        } else {
            0
        }
    }
    fn stc_issue(&mut self, here: &Stream<Compute>, sh: &Shadow, call: &BtsCall<'_, 'a, W, M>, xs: &mut [Ct<'a, W>]) -> pc_gpu::Result<()> {
        self.lane.stc_inputs(here, sh, call, xs)
    }
    fn joint(&self, lin: &str) -> bool {
        self.lane.joint_holds(lin)
    }
    fn joint_stc(&mut self, here: &Stream<Compute>, sh: &Shadow, call: &BtsCall<'_, 'a, W, M>, first: usize, xs: &mut [Ct<'a, W>]) -> pc_gpu::Result<()> {
        self.lane.joint_stc(here, sh, call, first, xs)
    }
    fn joint_inputs(&mut self, here: &Stream<Compute>, xs: &[Ct<'a, W>]) -> pc_gpu::Result<()> {
        self.lane.joint_inputs(here, xs)
    }
    fn joint_far(&mut self, here: &Stream<Compute>, sh: &Shadow, lin: &PcmmLinear, sub: &PcmmShape, call: &BtsCall<'_, 'a, W, M>, nf_out: f64, groups: &[Vec<usize>], drops: &[u32], outs: &mut [Ct<'a, W>]) -> pc_gpu::Result<()> {
        self.lane.joint_far(here, sh, lin, sub, call, nf_out, groups, drops, outs)
    }

    fn rows(&self, lin: &str) -> bool {
        self.lane.rows_holds(lin)
    }

    fn rows_stc(&mut self, here: &Stream<Compute>, sh: &Shadow, call: &BtsCall<'_, 'a, W, M>, _first: usize, xs: &mut [Ct<'a, W>], back: bool) -> pc_gpu::Result<()> {
        self.lane.rows_stc(here, sh, call, xs, back)
    }

    fn rows_product(&mut self, lin: &PcmmLinear, sub: &PcmmShape) -> pc_gpu::Result<(*const u32, *const u32)> {
        self.lane.rows_product(lin, sub)
    }

    fn rows_wait(&mut self, here: &Stream<Compute>) -> pc_gpu::Result<()> {
        self.lane.rows_wait(here)
    }

    fn rows_far(&mut self, here: &Stream<Compute>, sh: &Shadow, sub: &PcmmShape, call: &BtsCall<'_, 'a, W, M>, here_partial: (*const u32, *const u32), far_outs: &[usize], meta: &CtMeta, nf_out: f64, groups: &[Vec<usize>], drops: &[u32], outs: &mut [Ct<'a, W>]) -> pc_gpu::Result<()> {
        self.lane.rows_far(here, sh, sub, call, here_partial, far_outs, meta, nf_out, groups, drops, outs)
    }
}

/// A remote-device switch that is on unless set to `0` / `off` (unset, empty, `1`, `on` = on; anything else is
/// reported and means off).
pub fn default_on(var: &str) -> bool {
    match std::env::var(var).as_deref().map(str::trim) {
        Err(_) | Ok("") | Ok("1") | Ok("on") => true,
        Ok("0") | Ok("off") => false,
        Ok(v) => {
            eprintln!("[remote] {var}={v}: not 1 / on or 0 / off; off");
            false
        }
    }
}

/// With the finish split: every PCMM linear's last half of inputs takes its StC-first input stage on the remote refresh
/// device, back at the split (small) before the product. `PC_STC_SPLIT`: on by default (faster on every model
/// measured), `0` / `off` = off.
pub fn stc_split() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| default_on("PC_STC_SPLIT"))
}

/// PCMM inputs whose StC-first stage ran on the remote device (process total).
pub static REMOTE_STC_INPUTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// CCMM operands converted (`stc_high`) on the remote device (process total).
pub static REMOTE_CCMM_OPERANDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// With the CCMM finish split: the last half of each CCMM operand side converted on the remote refresh device.
/// `PC_CCMM_OPERAND_SPLIT`: on by default (faster on Llama and Gemma), `0` / `off` = off.
pub fn ccmm_operand_split() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| default_on("PC_CCMM_OPERAND_SPLIT"))
}

fn no_split() -> pc_gpu::Error {
    pc_gpu::Error::Mem { knob: "ccmm split", value: 0, msg: "this remote lane holds no CCMM engine (PC_CCMM_SPLIT)".into() }
}

/// PC_CCMM_SPLIT=1 (with the CCMM finish split and a remote CCMM engine): every CCMM product split across the two
/// devices (`pc_gpu::matmul::ccmm::mul_split`), the remote operand halves kept there and the remote output half
/// refreshed where it is. Default off until it measures faster.
pub fn ccmm_split() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("PC_CCMM_SPLIT").is_ok_and(|v| v.trim() == "1"))
}

/// PC_CCMM_SPLIT_CHECK=1 (diagnostic, graphs off): the CCMM service's word checks of PC_REMOTE_CHECK alone (remote
/// operand conversions, split products, remote merges and output refreshes against the same work on this device).
pub fn ccmm_split_check() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("PC_CCMM_SPLIT_CHECK").is_ok_and(|v| v.trim() == "1"))
}

/// Split CCMM products run (process total).
pub static REMOTE_CCMM_SPLITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The remote device's CCMM engine for split products (PC_CCMM_SPLIT): the engine (`enable_split`), its ciphertexts
/// (0 .. 8: the left operand's 8 .. 16, then the outputs 8 .. 16; 8 .. 16: the right operand's 8 .. 16), its merge
/// table, and per device (`_t` there, `_h` here) the split events, the copy stream and PC_CCMM_PHASES timing events.
/// `back` (here): the copies back of `ccmm_far_back`.
pub struct RemoteCcmm<'r, 'a, W: DevWord> {
    pub engine: &'r mut Ccmm<'a, 'static, W>,
    pub cts: &'r mut [Ct<'a, W>],
    pub mono: &'r MonomialTable<'a, W>,
    pub ev_t: &'r [Event],
    pub cp_t: &'r Stream<Compute>,
    pub ph_t: Option<&'r [Event]>,
    pub ev_h: &'r [Event],
    pub cp_h: &'r Stream<Compute>,
    pub ph_h: Option<&'r [Event]>,
    pub back: &'r Event,
}

/// The remote lane over one device's executor, op layer, keys and routes (`RemoteLane`): `here` / `there` the two
/// devices, `gather` the remote batch buffer, `fork` an event of `here` (recorded on its compute, waited on by the
/// remote one), `join` an event of `there` (the other way).
pub struct Remote<'r, 'a, W: DevWord, M: DevStrategy<W>> {
    pub here: &'r pc_cuda::Device,
    pub there: &'r pc_cuda::Device,
    /// the context on `here` (the outputs' device): what the copy back reads the layout and the chain shape from
    pub here_ctx: &'r Context<'a, W, M>,
    pub ctx: &'r Context<'a, W, M>,
    pub ex: &'r mut Executor<'a, W>,
    pub ks: &'r KsPlans<'a>,
    pub ops: &'r mut Ops<'a, W>,
    pub work: &'r mut BtsWork<'a, W>,
    pub routes: &'r Routes<'r, 'a, W, M>,
    /// the remote device's StC-first route (the PCMM finishes' groups)
    pub stcf: Option<&'r StcFirstRoute<'a, W, M>>,
    pub gather: &'r mut Ct<'a, W>,
    pub fork: &'r Event,
    pub join: &'r Event,
    /// the asynchronous lane (None: not offered): a second one-ciphertext buffer (a pair's second member), one event
    /// per step (recorded on the remote stream), the lane's fork (of `here`) and join (of `there`)
    pub lane: Option<RemoteAsync<'r, 'a, W>>,
    /// joint linears (None: no PCMM engine on the remote device)
    pub pcmm: Option<RemotePcmm<'r, 'a, W>>,
    /// the replica lane (None: off)
    pub replica: Option<RemoteReplica<'r, 'a, W>>,
    /// PC_REMOTE_CB: a stream of the remote device for the far work's copies back and its event (None: they run on
    /// the remote compute stream)
    pub fcb: Option<(&'r Stream<Compute>, &'r Event)>,
    /// ownership (PC_OWN): the current step's expected slots, the next service call's output slots and shift, the
    /// slots left in the replica's slots so far
    pub own: OwnState,
    /// split CCMM products (None: off)
    pub ccmm: Option<RemoteCcmm<'r, 'a, W>>,
}

/// PC_OWN_LAZY_PATHS (diagnostic, with PC_OWN_LAZY): which services skip copying owned outputs back, bits: 1 split
/// refresh batches, 2 joint linears' far groups, 4 CCMM output halves (default 7: all).
fn lazy_path(bit: u32) -> bool {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("PC_OWN_LAZY_PATHS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(7)) & bit != 0
}

/// Ownership's per-step state on a remote lane (`RemoteLane::own_begin` .. `own_take`).
#[derive(Clone, Debug, Default)]
pub struct OwnState {
    pub want: Vec<u32>,
    pub next: Option<(Vec<u32>, u32)>,
    pub done: Vec<u32>,
}

/// Owned outputs left in the replica's slots by the remote services, and those copied over from here instead
/// (process totals).
pub static OWN_KEPT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static OWN_COPIED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Ownership: member `c` of `src` (the remote device's memory) into the replica's slot `slot` at `meta`, on `s` (the
/// remote compute stream, before the join its caller records), when the step's plan expects that slot there.
fn own_copy<'a, W: DevWord, M: DevStrategy<W>>(ctx: &Context<'a, W, M>, s: &Stream<Compute>, replica: &mut Option<RemoteReplica<'_, 'a, W>>, own: &mut OwnState, slot: u32, src: &Ct<'a, W>, c: usize, meta: CtMeta) -> pc_gpu::Result<()> {
    if !own.want.contains(&slot) || own.done.contains(&slot) {
        return Ok(());
    }
    let Some(rp) = replica.as_mut() else { return Ok(()) };
    let Some(dst) = rp.b.slots.get_mut(slot as usize) else { return Ok(()) };
    batch::take(ctx, s, dst, src, c, meta)?;
    own.done.push(slot);
    Ok(())
}

/// PC_REMOTE_CB=1: the remote device's copies back (refreshed far groups, split refresh batches, CCMM output halves)
/// on a stream of their own, so its compute stream moves on. Default off until it measures faster.
pub fn remote_cb() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("PC_REMOTE_CB").is_ok_and(|v| v.trim() == "1"))
}

/// The replica lane of the remote device (`replica`): its own executor and op layer (own stream and workspaces, so it
/// overlaps the joint work on the remote compute executor), the block's second binding over the mirrored slab
/// (swapped in for the run), an event per step on its stream, its fork (this device) and join (remote) events, and
/// its rotation-chain owners.
pub struct RemoteReplica<'r, 'a, W: DevWord> {
    /// handoff events (remote device): the replica's stream to the remote compute stream and back, around a refresh
    /// the replica's plan placed there (PC_REPLICA_BTS)
    pub hand: &'r [Event],
    /// the replica's own refresh workspaces (PC_REPLICA_BTS): its refreshes run on its stream
    pub work: Option<&'r mut BtsWork<'a, W>>,
    pub ex: &'r mut Executor<'a, W>,
    pub ops: &'r mut Ops<'a, W>,
    pub b: &'r mut Bound<'a, W>,
    pub ev: &'r [Event],
    pub fork: &'r Event,
    pub join: &'r Event,
    pub owners: (Option<u32>, Option<u32>),
    /// PC_REPLICA_DMA: the replica device's copy stream, its copy events (one per copy of a block) and its join event;
    /// this device's side uses the asynchronous lane's copy-back stream and events (`RemoteAsync::cb`, `cb_ev`,
    /// `cb_join`, `fork`: a replica block runs no asynchronous lane)
    pub dma: Option<(&'r Stream<Compute>, &'r [Event], &'r Event)>,
}

/// Replica steps and copies run (process totals): steps, copies onto the replica, copies back.
pub static REPLICA_STEPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static REPLICA_COPIES_IN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static REPLICA_COPIES_BACK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Blocks checked word for word against this device alone (`PC_REPLICA_CHECK`).
pub static REPLICA_CHECKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Refreshes (singles, pairs) the replica's plan ran on the remote device (PC_REPLICA_BTS).
pub static REPLICA_BTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// PC_REPLICA_CHECK=1 (diagnostic, graphs off): every block with a replica plan runs twice from the same inputs, on
/// this device alone and with the replica lane, every output word and metadata compared.
pub fn replica_check() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("PC_REPLICA_CHECK").is_ok_and(|v| v == "1" || v == "self" || v == "steps" || v == "units"))
}

/// PC_REPLICA_SYNC=1 (diagnostic, eager): the host waits for both devices around every replica step and copy (a
/// mismatch that survives it is not an ordering race).
fn sync_both(here: &pc_cuda::Device, there: &pc_cuda::Device) -> Result<()> {
    there.make_current()?;
    there.synchronize()?;
    here.make_current()?;
    here.synchronize()?;
    Ok(())
}

fn replica_sync() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("PC_REPLICA_SYNC").is_ok_and(|v| v == "1"))
}
/// The constant-ring pushes of the replica's op layer in a combined lease (`ring_lease_end`): tagged with this bit.
const REPLICA_PUSH: usize = 1 << 48;

/// The remote device's PCMM engine for joint linears: its planes of each joint linear's far outputs, its inputs at the
/// split (`max_in`) and its finish-group outputs (`batch` members each).
pub struct RemotePcmm<'r, 'a, W: DevWord> {
    pub engine: &'r mut Pcmm<'a, 'static, W>,
    pub planes: &'r std::collections::BTreeMap<String, pc_cuda::DBuf<'a, Compute, u8>>,
    pub ins: &'r mut [Ct<'a, W>],
    pub outs: &'r mut [Ct<'a, W>],
}

/// Joint PCMM linears run (process total): far outputs computed and finished on the remote device.
pub static REMOTE_JOINT_LINEARS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static REMOTE_JOINT_OUTPUTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Row-split linears run (process total): the remote half of the inputs' partial product summed there.
pub static REMOTE_ROWS_LINEARS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// PC_REDUCE_SPLIT=1 (with joint linears): a linear with more input than output blocks splits its inputs across the
/// devices instead of its outputs (`pc_gpu::matmul::pcmm::linear_rows`). Default off until it measures faster.
pub fn reduce_split() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("PC_REDUCE_SPLIT").is_ok_and(|v| v.trim() == "1"))
}

/// The asynchronous lane's buffers, streams and events (`Remote::lane`): `ring` one-ciphertext buffers of the remote
/// device (entry e: `ring[2e]`, `ring[2e + 1]` for a pair's second member), refreshed there; `done` per entry (remote:
/// its refresh issued), `ev_in` per step (remote: inputs copied), `cb` this device's copy-back stream and `cb_ev` per
/// step on it (output copied back); `fork` (this device) starts both, `join` (remote) and `cb_join` (this device) end
/// them. `last` (per entry, the step whose copy back must finish before the entry is reused) restarts at every fork.
pub struct RemoteAsync<'r, 'a, W: DevWord> {
    pub ring: &'r mut [Ct<'a, W>],
    pub done: &'r [Event],
    pub ev_in: &'r [Event],
    pub cb: &'r Stream<Compute>,
    pub cb_ev: &'r [Event],
    pub fork: &'r Event,
    pub join: &'r Event,
    pub cb_join: &'r Event,
    pub next: usize,
    pub last: Vec<Option<usize>>,
    /// the ring entry of the step whose copy back is pending (`lane_issue` -> `lane_finish`)
    pub pending: Option<(usize, usize, usize)>,
}

/// Refresh units split across two devices and ciphertexts refreshed remotely (process totals).
pub static REMOTE_UNITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static REMOTE_CTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// PCMM finish groups and their ciphertexts refreshed remotely, and groups checked word for word (process totals).
pub static REMOTE_FIN_GROUPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static REMOTE_FIN_CTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static REMOTE_FIN_CHECKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// CCMM output-refresh halves and their ciphertexts refreshed remotely, and halves checked word for word.
pub static REMOTE_CCMM_RUNS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static REMOTE_CCMM_CTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static REMOTE_CCMM_CHECKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// How the refreshes ran (process totals): single bootstrap steps, complex pairs, local refresh batches and their
/// ciphertexts (`[remote]` report).
pub static BTS_SINGLE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static BTS_PAIRS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static BTS_BATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static BTS_BATCH_CTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Split units checked word for word against a whole local batch (`PC_REMOTE_CHECK`).
pub static REMOTE_CHECKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Refreshes on the asynchronous lane: single steps, pairs, and those checked word for word (`PC_REMOTE_CHECK`).
pub static REMOTE_ASYNC_SINGLES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static REMOTE_ASYNC_PAIRS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static REMOTE_ASYNC_CHECKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// With a remote refresh device, the lane schedule may move single refreshes and refresh pairs whole onto it
/// (`schedule` lanes 2 and 3). `PC_REMOTE_ASYNC`: unset, `1` or `on` = on (default: faster or equal on every model
/// measured), `0` or `off` = off; anything else is reported and means off.
pub fn remote_async() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("PC_REMOTE_ASYNC").as_deref().map(str::trim) {
        Err(_) | Ok("") | Ok("1") | Ok("on") => true,
        Ok("0") | Ok("off") => false,
        Ok(v) => {
            eprintln!("[remote] PC_REMOTE_ASYNC={v}: not 1 / on or 0 / off; the asynchronous lane stays off");
            false
        }
    })
}

/// PC_REMOTE_CHECK=1 (diagnostic, graphs off): after every split refresh unit, the whole batch again on this device
/// and every member's output compared word for word (and its metadata) with what the split produced.
pub fn remote_check() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("PC_REMOTE_CHECK").is_ok_and(|v| v == "1"))
}

impl<'r, 'a, W: DevWord, M: DevStrategy<W>> Remote<'r, 'a, W, M> {
    /// `f` with the remote device current, this one current again afterwards whatever happened.
    fn on_remote<T>(&mut self, f: impl FnOnce(&mut Self) -> pc_gpu::Result<T>) -> pc_gpu::Result<T> {
        self.there.make_current()?;
        let r = f(self);
        let back = self.here.make_current();
        let v = r?;
        back?;
        Ok(v)
    }
}

impl<'r, 'a, W: DevWord, M: DevStrategy<W>> RemoteLane<'a, W, M> for Remote<'r, 'a, W, M> {
    fn has(&self, kind: RouteKind) -> bool {
        self.routes.get(kind).is_some()
    }

    fn lane_ok(&self) -> bool {
        self.lane.as_ref().is_some_and(|l| !l.ev_in.is_empty() && l.ring.len() >= 2 && l.done.len() * 2 == l.ring.len())
    }

    fn lane_fork(&mut self, here: &Stream<Compute>) -> pc_gpu::Result<()> {
        let l = self.lane.as_mut().ok_or_else(no_lane)?;
        here.record(l.fork)?;
        l.cb.wait(l.fork)?;
        l.last.iter_mut().for_each(|x| *x = None);
        l.pending = None;
        self.on_remote(|me| {
            let l = me.lane.as_ref().ok_or_else(no_lane)?;
            me.ex.compute.wait(l.fork)?;
            Ok(())
        })
    }

    fn lane_wait(&mut self, ev: &Event) -> pc_gpu::Result<()> {
        self.on_remote(|me| {
            me.ex.compute.wait(ev)?;
            Ok(())
        })
    }

    fn lane_out_wait(&mut self, ev: &Event) -> pc_gpu::Result<()> {
        let l = self.lane.as_ref().ok_or_else(no_lane)?;
        l.cb.wait(ev)?;
        Ok(())
    }

    fn lane_issue(&mut self, ids: &[usize], sh: &Shadow, kind: RouteKind, routes: &[pc_shadow::Route], srcs: &[&Ct<'a, W>]) -> pc_gpu::Result<()> {
        let r = self.routes.get(kind).ok_or_else(|| pc_gpu::Error::Mem { knob: "remote routes", value: 0, msg: format!("route {kind:?} not held remotely") })?;
        let n = srcs.len();
        let i = ids.first().copied().unwrap_or(0);
        if !(1..=2).contains(&n) || routes.len() != n || ids.len() != n {
            return Err(pc_gpu::Error::Mem { knob: "remote lane", value: n, msg: format!("{n} inputs, {} routes (one refresh or a pair)", routes.len()) });
        }
        let lane = self.lane.as_mut().ok_or_else(no_lane)?;
        if lane.pending.is_some() {
            return Err(pc_gpu::Error::Mem { knob: "remote lane", value: i, msg: "a remote refresh issued before the previous one's copy back".into() });
        }
        if ids.iter().any(|&j| lane.ev_in.get(j).is_none() || lane.cb_ev.get(j).is_none()) {
            return Err(pc_gpu::Error::Mem { knob: "remote lane", value: i, msg: format!("steps {ids:?}: {} lane events", lane.ev_in.len()) });
        }
        let entries = lane.done.len();
        let e = lane.next % entries;
        lane.next += 1;
        // the entry's previous user must be copied back before its buffers are written again
        let reuse = lane.last[e].replace(i);
        lane.pending = Some((i, e, n));
        self.on_remote(|me| {
            let lane = me.lane.as_mut().ok_or_else(no_lane)?;
            if let Some(j) = reuse {
                me.ex.compute.wait(&lane.cb_ev[j])?;
            }
            let (x, rest) = lane.ring[2 * e..2 * e + 2].split_at_mut(1);
            let (x, y) = (&mut x[0], &mut rest[0]);
            {
                let s = &me.ex.compute;
                // SAFETY: the remote stream waited for the events of every write of the inputs (the schedule's waits)
                unsafe { batch::put_peer(me.ctx, s, x, me.there, 0, srcs[0], me.here)? };
                x.meta = srcs[0].meta;
                if n == 2 {
                    // SAFETY: as above
                    unsafe { batch::put_peer(me.ctx, s, y, me.there, 0, srcs[1], me.here)? };
                    y.meta = srcs[1].meta;
                }
                for &j in ids {
                    s.record(&lane.ev_in[j])?;
                }
            }
            let mut cx = BtsCx { ctx: me.ctx, ex: &mut *me.ex, sh, ks: me.ks, ops: &mut *me.ops, work: &mut *me.work };
            if n == 1 {
                r.run(&mut cx, std::slice::from_mut(x), &routes[0])?;
            } else {
                r.run_pair(&mut cx, x, y, &routes[0], &routes[1])?;
            }
            me.ex.join_aux()?;
            me.ex.compute.record(&lane.done[e])?;
            Ok(())
        })
    }

    fn lane_finish(&mut self, ids: &[usize], outs: &mut [&mut Ct<'a, W>]) -> pc_gpu::Result<()> {
        let i = ids.first().copied().unwrap_or(0);
        let lane = self.lane.as_mut().ok_or_else(no_lane)?;
        let (pi, e, n) = lane.pending.take().ok_or_else(|| pc_gpu::Error::Mem { knob: "remote lane", value: i, msg: "a copy back without its refresh".into() })?;
        if pi != i || outs.len() != n {
            return Err(pc_gpu::Error::Mem { knob: "remote lane", value: i, msg: format!("copy back of step {i} ({} outputs) after the refresh of step {pi} ({n})", outs.len()) });
        }
        let cb = lane.cb;
        cb.wait(&lane.done[e])?;
        for (c, o) in outs.iter_mut().enumerate() {
            let src = &lane.ring[2 * e + c];
            let m = src.meta;
            // SAFETY: the copy-back stream waited for the refresh that wrote the ring entry, and for every earlier
            // reader and writer of the output slot (the schedule's waits, `lane_out_wait`)
            unsafe { batch::take_peer(self.here_ctx, cb, o, self.here, src, self.there, 0, m)? };
        }
        for &j in ids {
            cb.record(&lane.cb_ev[j])?;
        }
        Ok(())
    }

    fn lane_event(&self, i: usize) -> Option<&Event> {
        self.lane.as_ref().and_then(|l| l.cb_ev.get(i))
    }

    fn lane_event_in(&self, i: usize) -> Option<&Event> {
        self.lane.as_ref().and_then(|l| l.ev_in.get(i))
    }

    fn lane_join(&mut self, here: &Stream<Compute>) -> pc_gpu::Result<()> {
        self.on_remote(|me| {
            let l = me.lane.as_ref().ok_or_else(no_lane)?;
            me.ex.compute.record(l.join)?;
            Ok(())
        })?;
        let l = self.lane.as_ref().ok_or_else(no_lane)?;
        l.cb.record(l.cb_join)?;
        here.wait(l.join)?;
        here.wait(l.cb_join)?;
        Ok(())
    }

    fn lane_sync(&mut self) -> pc_gpu::Result<()> {
        self.on_remote(|me| {
            me.ex.compute.sync()?;
            Ok(())
        })?;
        let l = self.lane.as_ref().ok_or_else(no_lane)?;
        l.cb.sync()?;
        Ok(())
    }

    fn ring_lease_begin(&mut self) -> pc_gpu::Result<()> {
        self.on_remote(|me| {
            me.ops.ring.lease_begin(&me.ex.compute)?;
            if let Some(rp) = me.replica.as_mut() {
                rp.ops.ring.lease_begin(&rp.ex.compute)?;
            }
            Ok(())
        })
    }

    fn ring_lease_end(&mut self) -> pc_gpu::Result<Vec<usize>> {
        self.on_remote(|me| {
            let mut v = me.ops.ring.lease_end(&me.ex.compute)?;
            if let Some(rp) = me.replica.as_mut() {
                v.extend(rp.ops.ring.lease_end(&rp.ex.compute)?.into_iter().map(|x| x | REPLICA_PUSH));
            }
            Ok(v)
        })
    }

    fn ring_release(&mut self, pushes: &[usize]) {
        let (rep, own): (Vec<usize>, Vec<usize>) = pushes.iter().partition(|&&x| x & REPLICA_PUSH != 0);
        self.ops.ring.release(&own);
        if let Some(rp) = self.replica.as_mut() {
            rp.ops.ring.release(&rep.into_iter().map(|x| x & !REPLICA_PUSH).collect::<Vec<_>>());
        }
    }

    fn replica_on(&self) -> bool {
        self.replica.is_some()
    }

    fn replica_pt(&self, p: u32) -> bool {
        self.replica.as_ref().and_then(|rp| rp.b.pts.get(p as usize)).is_some_and(|bp| !bp.stream && !bp.rows.is_null() && bp.meta.is_some())
    }

    fn replica_holds(&self, kind: RouteKind) -> bool {
        self.replica.is_some() && self.routes.get(kind).is_some()
    }

    fn replica_set_meta(&mut self, s: u32, meta: CtMeta) -> Result<()> {
        let rp = self.replica.as_mut().ok_or_else(no_replica)?;
        let label = rp.b.label.clone();
        rp.b.slots.get_mut(s as usize).ok_or_else(|| rerr(&label, format!("replica slot {s}")))?.meta = meta;
        Ok(())
    }

    fn own_begin(&mut self, slots: &[u32]) {
        self.own = OwnState { want: slots.to_vec(), next: None, done: vec![] };
    }

    fn own_wanted(&self) -> &[u32] {
        &self.own.want
    }

    fn own_next(&mut self, slots: Vec<u32>, shift: u32) {
        if !self.own.want.is_empty() {
            self.own.next = Some((slots, shift));
        }
    }

    fn own_take(&mut self) -> Vec<u32> {
        self.own.want.clear();
        self.own.next = None;
        std::mem::take(&mut self.own.done)
    }

    fn replica_fork(&mut self, here: &Stream<Compute>) -> Result<()> {
        let rp = self.replica.as_mut().ok_or_else(no_replica)?;
        here.record(rp.fork)?;
        rp.owners = (None, None);
        self.there.make_current()?;
        let r = (|| -> Result<()> {
            let rp = self.replica.as_mut().ok_or_else(no_replica)?;
            rp.ex.compute.wait(rp.fork)?;
            Ok(())
        })();
        self.here.make_current()?;
        r
    }

    fn replica_join(&mut self, here: &Stream<Compute>) -> Result<()> {
        self.there.make_current()?;
        let r = (|| -> Result<()> {
            let rp = self.replica.as_mut().ok_or_else(no_replica)?;
            rp.ex.compute.record(rp.join)?;
            Ok(())
        })();
        self.here.make_current()?;
        r?;
        let rp = self.replica.as_ref().ok_or_else(no_replica)?;
        here.wait(rp.join)?;
        Ok(())
    }

    fn replica_wait(&mut self, ev: &Event) -> Result<()> {
        self.there.make_current()?;
        let r = (|| -> Result<()> {
            let rp = self.replica.as_mut().ok_or_else(no_replica)?;
            rp.ex.compute.wait(ev)?;
            Ok(())
        })();
        self.here.make_current()?;
        r
    }

    fn replica_stamp(&mut self, buf: *mut u64, idx: i32) -> Result<()> {
        self.there.make_current()?;
        let r = (|| -> Result<()> {
            let rp = self.replica.as_mut().ok_or_else(no_replica)?;
            // SAFETY: `buf` is mapped portable host memory of at least idx + 1 words (PC_TL's)
            let e = unsafe { pc_kernels::pc_stamp(buf, idx, rp.ex.compute.raw() as *mut std::ffi::c_void) };
            if e != 0 {
                return Err(rerr("PC_TL", format!("stamp launch: CUDA error {e}")));
            }
            Ok(())
        })();
        self.here.make_current()?;
        r
    }

    fn replica_copy_in(&mut self, s: u32, src: &Ct<'a, W>) -> Result<()> {
        let ctx = self.ctx;
        self.there.make_current()?;
        let r = (|| -> Result<()> {
            let rp = self.replica.as_mut().ok_or_else(no_replica)?;
            let label = rp.b.label.clone();
            let dst = rp.b.slots.get_mut(s as usize).ok_or_else(|| rerr(&label, format!("replica slot {s}")))?;
            copy_ct(ctx, &rp.ex.compute, dst, src)?;
            REPLICA_COPIES_IN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        })();
        self.here.make_current()?;
        if replica_sync() {
            sync_both(self.here, self.there)?;
        }
        r
    }

    fn replica_slot(&self, s: u32) -> Option<&Ct<'a, W>> {
        self.replica.as_ref().and_then(|rp| rp.b.slots.get(s as usize))
    }

    fn replica_slot_ptr(&mut self, s: u32) -> Option<*mut Ct<'a, W>> {
        self.replica.as_mut().and_then(|rp| rp.b.slots.get_mut(s as usize)).map(|c| c as *mut Ct<'a, W>)
    }

    fn replica_issue(&mut self, sh: &Shadow, source: KskSource, i: usize, unit: Option<(u32, usize)>, signal: Option<usize>) -> Result<()> {
        if replica_sync() {
            sync_both(self.here, self.there)?;
        }
        self.there.make_current()?;
        let r = (|| -> Result<()> {
            let Remote { ctx, ks, work, routes, replica, ex: rex, ops: rops, .. } = self;
            let rp = replica.as_mut().ok_or_else(no_replica)?;
            // a refresh (single or pair) the plan placed here runs on the remote compute stream, with its refresh
            // workspaces and route caches (the finishes' halves run there too, in issue order): handed over from the
            // replica's stream and back by events
            let boot = {
                let b = &*rp.b;
                match unit {
                    Some((u, _)) => matches!(b.units[u as usize].kind, UnitKind::Pair { .. }),
                    None => matches!(b.steps[i].op, StepOp::Bootstrap { .. }),
                }
            };
            if boot && rp.work.is_some() {
                // its own refresh workspaces: on the replica's stream, in its order (the routes' caches are read only
                // under capture and filled eagerly on a miss)
                let RemoteReplica { ex, ops, b, ev, work: rw, .. } = rp;
                let rw = rw.as_deref_mut().ok_or_else(no_replica)?;
                {
                    let mut rt = Rt { ctx, ex: &mut **ex, sh, ks, ops: &mut **ops, work: rw, routes, source, locks: &[], finish: None, side: None, coeff: None, remote: None };
                    let Bound { label, slots, tmp, gather, held, steps, groups, relin, step_lists, units, pts, finishes, coeff_sums, mult_sums, stream_base, .. } = &mut **b;
                    let tmp = tmp.as_mut().ok_or_else(|| rerr(label, "replica: no scratch slot"))?;
                    let t = Tables { groups, relin: *relin, lists: step_lists, pts, finishes, coeff_sums, mult_sums, stream_base: *stream_base };
                    let mut fused = 0usize;
                    let (mut o0, mut o1) = (None, None);
                    match unit {
                        Some((u, e)) if !unit_fits(&units[u as usize], slots, steps, pts, sh, routes) => {
                            for j in i..e {
                                exec_at(label, &mut rt, &t, units, steps, slots, tmp, gather, held, j, None, (&mut o0, &mut o1), true, &mut fused)?;
                            }
                        }
                        _ => exec_at(label, &mut rt, &t, units, steps, slots, tmp, gather, held, i, unit, (&mut o0, &mut o1), true, &mut fused)?,
                    }
                    rt.ex.join_aux()?;
                    // PC_REPLICA_TRACE=meta: the refresh's inputs and outputs here against the compiled metadata
                    if std::env::var("PC_REPLICA_TRACE").is_ok_and(|v| v.trim() == "meta") {
                        static SEEN_B: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                        let e = unit.map_or(i + 1, |(_, e)| e);
                        for j in i..e {
                            let st = &steps[j];
                            let m = |s: u32| if s != NONE { slots.get(s as usize).map(|c| (c.meta.level, c.meta.deg, c.meta.nf.log2())) } else { None };
                            let out = slots.get(st.out as usize).map(|c| (c.meta.level, c.meta.deg, c.meta.nf.log2()));
                            let bad = out.is_some_and(|o| (o.0, o.1) != (st.meta.level, st.meta.deg) || (st.meta.nf > 0.0 && (o.2 - st.meta.nf.log2()).abs() > 1e-6));
                            if bad && SEEN_B.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                                eprintln!("[replica-trace] {label} refresh step {j} (unit {unit:?}): out slot {} at {out:?}, compiled {:?}; input slot {} at {:?}, compiled {:?}", st.out, (st.meta.level, st.meta.deg, st.meta.nf.log2()), st.a, m(st.a), st.a_meta.map(|x| (x.level, x.deg, x.nf.log2())));
                            }
                        }
                    }
                }
                if let Some(k) = signal {
                    let e = ev.get(k).ok_or_else(|| rerr(&b.label, format!("replica: step {k}, {} events", ev.len())))?;
                    ex.compute.record(e)?;
                }
                REPLICA_STEPS.fetch_add(unit.map_or(1, |(_, e)| (e - i) as u64), std::sync::atomic::Ordering::Relaxed);
                REPLICA_BTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(());
            }
            if boot {
                let RemoteReplica { ex, b, ev, hand, .. } = rp;
                let (h0, h1) = (hand.first().ok_or_else(no_replica)?, hand.get(1).ok_or_else(no_replica)?);
                ex.compute.record(h0)?;
                rex.compute.wait(h0)?;
                {
                    let mut rt = Rt { ctx, ex: &mut **rex, sh, ks, ops: &mut **rops, work: &mut **work, routes, source, locks: &[], finish: None, side: None, coeff: None, remote: None };
                    let Bound { label, slots, tmp, gather, held, steps, groups, relin, step_lists, units, pts, finishes, coeff_sums, mult_sums, stream_base, .. } = &mut **b;
                    let tmp = tmp.as_mut().ok_or_else(|| rerr(label, "replica: no scratch slot"))?;
                    let t = Tables { groups, relin: *relin, lists: step_lists, pts, finishes, coeff_sums, mult_sums, stream_base: *stream_base };
                    let mut fused = 0usize;
                    let (mut o0, mut o1) = (None, None);
                    match unit {
                        Some((u, e)) if !unit_fits(&units[u as usize], slots, steps, pts, sh, routes) => {
                            for j in i..e {
                                exec_at(label, &mut rt, &t, units, steps, slots, tmp, gather, held, j, None, (&mut o0, &mut o1), true, &mut fused)?;
                            }
                        }
                        _ => exec_at(label, &mut rt, &t, units, steps, slots, tmp, gather, held, i, unit, (&mut o0, &mut o1), true, &mut fused)?,
                    }
                    rt.ex.join_aux()?;
                }
                rex.compute.record(h1)?;
                ex.compute.wait(h1)?;
                if let Some(k) = signal {
                    let e = ev.get(k).ok_or_else(|| rerr(&b.label, format!("replica: step {k}, {} events", ev.len())))?;
                    ex.compute.record(e)?;
                }
                REPLICA_STEPS.fetch_add(unit.map_or(1, |(_, e)| (e - i) as u64), std::sync::atomic::Ordering::Relaxed);
                REPLICA_BTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(());
            }
            let RemoteReplica { ex, ops, b, ev, owners, .. } = rp;
            let mut rt = Rt { ctx, ex: &mut **ex, sh, ks, ops: &mut **ops, work: &mut **work, routes, source, locks: &[], finish: None, side: None, coeff: None, remote: None };
            let Bound { label, slots, tmp, gather, held, steps, groups, relin, step_lists, units, pts, finishes, coeff_sums, mult_sums, stream_base, .. } = &mut **b;
            let tmp = tmp.as_mut().ok_or_else(|| rerr(label, "replica: no scratch slot"))?;
            let t = Tables { groups, relin: *relin, lists: step_lists, pts, finishes, coeff_sums, mult_sums, stream_base: *stream_base };
            let mut fused = 0usize;
            match unit {
                // a unit the replica's slots do not fit runs step by step (as `issue_range` does here)
                Some((u, e)) if !unit_fits(&units[u as usize], slots, steps, pts, sh, routes) => {
                    for j in i..e {
                        exec_at(label, &mut rt, &t, units, steps, slots, tmp, gather, held, j, None, (&mut owners.0, &mut owners.1), true, &mut fused)?;
                    }
                }
                _ => {
                    let r = exec_at(label, &mut rt, &t, units, steps, slots, tmp, gather, held, i, unit, (&mut owners.0, &mut owners.1), true, &mut fused);
                    if let Err(e) = r {
                        // the replica's operands (its slots' metadata) against the compiled ones
                        let st = &steps[i];
                        let m = |s: u32| if s != NONE { slots.get(s as usize).map(|c| (c.meta.level, c.meta.deg, c.meta.nf.log2())) } else { None };
                        return Err(rerr(label, format!("{e} (replica step {i}, unit {unit:?}: operands there at {:?} / {:?}, compiled {:?} / {:?})", m(st.a), m(st.b), st.a_meta.map(|x| (x.level, x.deg, x.nf.log2())), st.b_meta.map(|x| (x.level, x.deg, x.nf.log2())))));
                    }
                }
            }
            if let Some(k) = signal {
                let e = ev.get(k).ok_or_else(|| rerr(label, format!("replica: step {k}, {} events", ev.len())))?;
                rt.ex.compute.record(e)?;
            }
            // PC_REPLICA_TRACE=meta (diagnostic): the first replica steps whose output metadata leaves the compiled one
            if std::env::var("PC_REPLICA_TRACE").is_ok_and(|v| v.trim() == "meta") {
                static SEEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                let e = unit.map_or(i + 1, |(_, e)| e);
                let fused = unit.is_some_and(|(u, _)| matches!(units[u as usize].kind, UnitKind::Affine { .. } | UnitKind::MultPtAdd { .. } | UnitKind::MultPtAddChain { .. } | UnitKind::RotAcc { .. } | UnitKind::MultIAdd { .. }));
                for j in i..e {
                    if fused && j + 1 < e {
                        continue;
                    }
                    let st = &steps[j];
                    if let Some(c) = slots.get(st.out as usize) {
                        if (c.meta.level, c.meta.deg) != (st.meta.level, st.meta.deg) && SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 6 {
                            let m = |s: u32| if s != NONE { slots.get(s as usize).map(|c| (c.meta.level, c.meta.deg)) } else { None };
                            eprintln!(
                                "[replica-trace] {label} step {j} ({:?}, unit {unit:?}): out slot {} at {:?}, compiled {:?}; operands there {:?} / {:?}, compiled {:?} / {:?}",
                                format!("{:?}", st.op).chars().take(60).collect::<String>(),
                                st.out,
                                (c.meta.level, c.meta.deg),
                                (st.meta.level, st.meta.deg),
                                m(st.a),
                                m(st.b),
                                st.a_meta.map(|x| (x.level, x.deg)),
                                st.b_meta.map(|x| (x.level, x.deg))
                            );
                        }
                    }
                }
            }
            // PC_REPLICA_TRACE=<slot> (diagnostic): the replica's metadata of that slot after each step writing it
            if let Some(ts) = std::env::var("PC_REPLICA_TRACE").ok().and_then(|v| v.trim().parse::<u32>().ok()) {
                let e = unit.map_or(i + 1, |(_, e)| e);
                if (i..e).any(|j| steps[j].out == ts) {
                    eprintln!("[replica-trace] {label} step {i}..{e} unit {unit:?} wrote slot {ts}: {:?}", slots.get(ts as usize).map(|c| (c.meta.level, c.meta.deg)));
                }
            }
            REPLICA_STEPS.fetch_add(unit.map_or(1, |(_, e)| (e - i) as u64), std::sync::atomic::Ordering::Relaxed);
            Ok(())
        })();
        self.here.make_current()?;
        if replica_sync() {
            sync_both(self.here, self.there)?;
        }
        r
    }

    fn replica_event(&self, i: usize) -> Option<&Event> {
        self.replica.as_ref().and_then(|rp| rp.ev.get(i))
    }

    fn dma_fork(&mut self, here: &Stream<Compute>) -> Result<()> {
        let lane = self.lane.as_ref().ok_or_else(|| rerr("", "PC_REPLICA_DMA without the asynchronous lane's copy-back stream"))?;
        let (rs, _, _) = self.replica.as_ref().and_then(|rp| rp.dma).ok_or_else(|| rerr("", "PC_REPLICA_DMA without the replica's copy stream"))?;
        here.record(lane.fork)?;
        lane.cb.wait(lane.fork)?;
        self.there.make_current()?;
        let r = rs.wait(lane.fork);
        self.here.make_current()?;
        r?;
        Ok(())
    }

    fn dma_join(&mut self, here: &Stream<Compute>) -> Result<()> {
        let lane = self.lane.as_ref().ok_or_else(|| rerr("", "PC_REPLICA_DMA without the asynchronous lane's copy-back stream"))?;
        let (rs, _, rj) = self.replica.as_ref().and_then(|rp| rp.dma).ok_or_else(|| rerr("", "PC_REPLICA_DMA without the replica's copy stream"))?;
        lane.cb.record(lane.cb_join)?;
        here.wait(lane.cb_join)?;
        self.there.make_current()?;
        let r = rs.record(rj);
        self.here.make_current()?;
        r?;
        here.wait(rj)?;
        Ok(())
    }

    fn dma_wait(&mut self, to: u8, ev: &Event) -> Result<()> {
        if to == 0 {
            let lane = self.lane.as_ref().ok_or_else(|| rerr("", "PC_REPLICA_DMA without the asynchronous lane's copy-back stream"))?;
            lane.cb.wait(ev)?;
            return Ok(());
        }
        let (rs, _, _) = self.replica.as_ref().and_then(|rp| rp.dma).ok_or_else(|| rerr("", "PC_REPLICA_DMA without the replica's copy stream"))?;
        self.there.make_current()?;
        let r = rs.wait(ev);
        self.here.make_current()?;
        r?;
        Ok(())
    }

    fn dma_copy(&mut self, c: usize, to: u8, s: u32, here_slot: &mut Ct<'a, W>) -> Result<()> {
        let ctx = self.ctx;
        let (here, there) = (self.here, self.there);
        let rp = self.replica.as_mut().ok_or_else(no_replica)?;
        let (rstream, rev, _) = rp.dma.ok_or_else(|| rerr(&rp.b.label, "PC_REPLICA_DMA without the replica's copy stream"))?;
        let label = rp.b.label.clone();
        let rslot = rp.b.slots.get_mut(s as usize).ok_or_else(|| rerr(&label, format!("replica slot {s}")))?;
        if to == 1 {
            let ev = rev.get(c).ok_or_else(|| rerr(&label, format!("copy {c}: {} copy events on the replica device", rev.len())))?;
            there.make_current()?;
            // SAFETY: the copy stream waits for the version's producer and the replica's accesses of the old content
            // (the plan's waits, issued before); nothing writes either slot before the copy's event
            let r = (|| -> Result<()> {
                unsafe { copy_ct_dma(ctx, rstream, rslot, there, here_slot, here)? };
                rstream.record(ev)?;
                Ok(())
            })();
            here.make_current()?;
            r?;
            REPLICA_COPIES_IN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            let lane = self.lane.as_ref().ok_or_else(|| rerr(&label, "PC_REPLICA_DMA without the asynchronous lane's copy-back stream"))?;
            let ev = lane.cb_ev.get(c).ok_or_else(|| rerr(&label, format!("copy {c}: {} copy events on this device", lane.cb_ev.len())))?;
            // SAFETY: as above, on this device's copy-back stream
            unsafe { copy_ct_dma(ctx, lane.cb, here_slot, here, rslot, there)? };
            lane.cb.record(ev)?;
            REPLICA_COPIES_BACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if replica_sync() {
            sync_both(self.here, self.there)?;
        }
        Ok(())
    }

    fn dma_event(&self, c: usize, to: u8) -> Option<&Event> {
        if to == 0 {
            self.lane.as_ref().and_then(|l| l.cb_ev.get(c))
        } else {
            self.replica.as_ref().and_then(|rp| rp.dma).and_then(|(_, ev, _)| ev.get(c))
        }
    }

    fn issue(&mut self, here: &Stream<Compute>, sh: &Shadow, kind: RouteKind, route: &pc_shadow::Route, xs: &[&Ct<'a, W>], outs: &mut [&mut Ct<'a, W>]) -> pc_gpu::Result<()> {
        let r = self.routes.get(kind).ok_or_else(|| pc_gpu::Error::Mem { knob: "remote routes", value: 0, msg: format!("route {kind:?} not held remotely") })?;
        if xs.is_empty() || xs.len() != outs.len() {
            return Err(pc_gpu::Error::Mem { knob: "remote refresh", value: xs.len(), msg: format!("{} inputs, {} outputs", xs.len(), outs.len()) });
        }
        here.record(self.fork)?;
        self.there.make_current()?;
        let res = (|| -> pc_gpu::Result<()> {
            let s = &self.ex.compute;
            s.wait(self.fork)?;
            for (c, x) in xs.iter().enumerate() {
                // SAFETY: the remote stream waits for `fork`, recorded on `here` after every write of the inputs
                unsafe { batch::put_peer(self.ctx, s, self.gather, self.there, c, x, self.here)? };
            }
            self.gather.buf.k = xs.len();
            self.gather.meta = xs[0].meta;
            let mut cx = BtsCx { ctx: self.ctx, ex: &mut *self.ex, sh, ks: self.ks, ops: &mut *self.ops, work: &mut *self.work };
            r.run(&mut cx, std::slice::from_mut(&mut *self.gather), route)?;
            self.ex.join_aux()?;
            let out = self.gather.meta;
            // ownership: the refreshed members into the replica's slots too (PC_OWN_NOCOPY=1, diagnostic: copied over
            // from here after the step instead)
            let own_next = self.own.next.take().filter(|(v, _)| v.len() == outs.len() && !std::env::var("PC_OWN_NOCOPY").is_ok_and(|x| x == "1"));
            if let Some((slots, _)) = own_next.as_ref() {
                for (c, &slot) in slots.iter().enumerate() {
                    let m = r.landing(sh, &xs[c].meta, route).ok().filter(|m| m.level == out.level && m.deg == out.deg).map_or(out, |m| CtMeta { nf: out.nf, ..m });
                    own_copy(self.ctx, &self.ex.compute, &mut self.replica, &mut self.own, slot, &*self.gather, c, m)?;
                }
            }
            let s: &Stream<Compute> = match self.fcb {
                Some((fs, fe)) => {
                    self.ex.compute.record(fe)?;
                    fs.wait(fe)?;
                    fs
                }
                None => &self.ex.compute,
            };
            // PC_OWN_LAZY: members the ownership above left in the replica's slots are not copied back here
            let lazy = |c: usize| -> bool { crate::replica::own_lazy() && lazy_path(1) && self.fcb.is_none() && own_next.as_ref().is_some_and(|(v, _)| self.own.done.contains(&v[c])) };
            for (c, o) in outs.iter_mut().enumerate() {
                // each member's own landing at the batch's level, degree and scale, as the local batch sets it
                let m = r.landing(sh, &xs[c].meta, route).ok().filter(|m| m.level == out.level && m.deg == out.deg).map_or(out, |m| CtMeta { nf: out.nf, ..m });
                if lazy(c) {
                    o.meta = m;
                    continue;
                }
                // SAFETY: the gather buffer was written by the refresh just issued on this stream; `here` reads the
                // outputs only after `join`
                unsafe { batch::take_peer(self.here_ctx, s, o, self.here, self.gather, self.there, c, m)? };
            }
            tl_remote(s);
            s.record(self.join)?;
            Ok(())
        })();
        // back to this device whatever happened
        let back = self.here.make_current();
        res?;
        back?;
        REMOTE_UNITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        REMOTE_CTS.fetch_add(xs.len() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn join(&mut self, here: &Stream<Compute>) -> pc_gpu::Result<()> {
        here.wait(self.join)?;
        Ok(())
    }

    fn has_finish(&self) -> bool {
        self.stcf.is_some()
    }

    fn finish_group(&mut self, here: &Stream<Compute>, sh: &Shadow, call: &BtsCall<'_, 'a, W, M>, x: &mut Ct<'a, W>) -> pc_gpu::Result<()> {
        let r = self.stcf.ok_or_else(|| pc_gpu::Error::Mem { knob: "remote finish", value: 0, msg: "no StC-first route on the remote device".into() })?;
        here.record(self.fork)?;
        self.there.make_current()?;
        let res = (|| -> pc_gpu::Result<()> {
            let s = &self.ex.compute;
            s.wait(self.fork)?;
            // SAFETY: the remote stream waits for `fork`, recorded on `here` after every write of `x`
            unsafe { batch::put_batch_peer(self.ctx, s, self.gather, self.there, x, self.here)? };
            let mut cx = BtsCx { ctx: self.ctx, ex: &mut *self.ex, sh, ks: self.ks, ops: &mut *self.ops, work: &mut *self.work };
            // the same call resumed at the split on this device's copy of the route
            let mut far = BtsCall::resume(StagedRef::StcFirst(r), call.st.clone(), Phase::Split, call.opts);
            far.split = call.split;
            far.run_to(&mut cx, std::slice::from_mut(&mut *self.gather), Phase::Done)?;
            self.ex.join_aux()?;
            let m = self.gather.meta;
            let s = &self.ex.compute;
            // SAFETY: the gather buffer was written by the refresh just issued on this stream; `here` reads `x` only
            // after `join`
            unsafe { batch::take_batch_peer(self.here_ctx, s, x, self.here, self.gather, self.there, m)? };
            tl_remote(s);
            s.record(self.join)?;
            Ok(())
        })();
        // back to this device whatever happened
        let back = self.here.make_current();
        res?;
        back?;
        REMOTE_FIN_GROUPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        REMOTE_FIN_CTS.fetch_add(x.k() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn joint_holds(&self, lin: &str) -> bool {
        self.stcf.is_some() && self.pcmm.as_ref().is_some_and(|p| p.planes.contains_key(lin))
    }

    fn joint_stc(&mut self, here: &Stream<Compute>, sh: &Shadow, call: &BtsCall<'_, 'a, W, M>, first: usize, xs: &mut [Ct<'a, W>]) -> pc_gpu::Result<()> {
        let r = self.stcf.ok_or_else(no_joint)?;
        if self.pcmm.as_ref().is_none_or(|p| first + xs.len() > p.ins.len()) || xs.iter().any(|x| x.k() != 1) {
            return Err(pc_gpu::Error::Mem { knob: "remote pcmm", value: first + xs.len(), msg: "joint inputs past the remote engine's input ciphertexts".into() });
        }
        here.record(self.fork)?;
        self.there.make_current()?;
        let res = (|| -> pc_gpu::Result<()> {
            self.ex.compute.wait(self.fork)?;
            let mut at = first;
            for chunk in xs.chunks_mut(stcf_batch(r)) {
                {
                    let s = &self.ex.compute;
                    for (c, x) in chunk.iter().enumerate() {
                        // SAFETY: the remote stream waits for `fork`, recorded on `here` after every write of the inputs
                        unsafe { batch::put_peer(self.ctx, s, self.gather, self.there, c, x, self.here)? };
                    }
                    self.gather.buf.k = chunk.len();
                    self.gather.meta = chunk[0].meta;
                }
                let mut cx = BtsCx { ctx: self.ctx, ex: &mut *self.ex, sh, ks: self.ks, ops: &mut *self.ops, work: &mut *self.work };
                let mut far = BtsCall::resume(StagedRef::StcFirst(r), call.st.clone(), Phase::Begun, call.opts);
                far.stc_first_input(&mut cx, std::slice::from_mut(&mut *self.gather))?;
                self.ex.join_aux()?;
                let m = self.gather.meta;
                let s = &self.ex.compute;
                let p = self.pcmm.as_mut().ok_or_else(no_joint)?;
                for (c, o) in chunk.iter_mut().enumerate() {
                    // kept here for this device's product, and sent back
                    batch::take(self.ctx, s, &mut p.ins[at + c], self.gather, c, m)?;
                    // SAFETY: the gather buffer was written by the stage just issued on this stream; `here` reads the
                    // inputs only after `join`
                    unsafe { batch::take_peer(self.here_ctx, s, o, self.here, self.gather, self.there, c, m)? };
                }
                at += chunk.len();
            }
            tl_remote(&self.ex.compute);
            self.ex.compute.record(self.join)?;
            Ok(())
        })();
        let back = self.here.make_current();
        res?;
        back?;
        REMOTE_STC_INPUTS.fetch_add(xs.len() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn joint_inputs(&mut self, here: &Stream<Compute>, xs: &[Ct<'a, W>]) -> pc_gpu::Result<()> {
        if self.pcmm.as_ref().is_none_or(|p| xs.len() > p.ins.len()) {
            return Err(no_joint());
        }
        here.record(self.fork)?;
        self.there.make_current()?;
        let res = (|| -> pc_gpu::Result<()> {
            let s = &self.ex.compute;
            s.wait(self.fork)?;
            let p = self.pcmm.as_mut().ok_or_else(no_joint)?;
            for (c, x) in xs.iter().enumerate() {
                // SAFETY: the remote stream waits for `fork`, recorded on `here` after the inputs' stage
                unsafe { batch::put_peer(self.ctx, s, &mut p.ins[c], self.there, 0, x, self.here)? };
                p.ins[c].buf.k = 1;
                p.ins[c].meta = x.meta;
            }
            Ok(())
        })();
        let back = self.here.make_current();
        res?;
        back?;
        Ok(())
    }

    fn joint_far(&mut self, here: &Stream<Compute>, sh: &Shadow, lin: &PcmmLinear, sub: &PcmmShape, call: &BtsCall<'_, 'a, W, M>, nf_out: f64, groups: &[Vec<usize>], drops: &[u32], outs: &mut [Ct<'a, W>]) -> pc_gpu::Result<()> {
        let r = self.stcf.ok_or_else(no_joint)?;
        if groups.len() != outs.len() || drops.len() != outs.len() || self.pcmm.as_ref().is_none_or(|p| groups.len() > p.outs.len() || sub.n_in > p.ins.len()) {
            return Err(pc_gpu::Error::Mem { knob: "remote pcmm", value: groups.len(), msg: "far groups past the remote engine's outputs".into() });
        }
        self.there.make_current()?;
        let res = (|| -> pc_gpu::Result<()> {
            // ordered after joint_stc / joint_inputs on this stream (the same linear's)
            let p = self.pcmm.as_mut().ok_or_else(no_joint)?;
            let planes = p.planes.get(&lin.name).ok_or_else(|| pc_gpu::Error::Mem { knob: "remote pcmm", value: 0, msg: format!("no far planes of {}", lin.name) })?;
            let mut map = vec![(usize::MAX, 0); sub.n_out];
            for (g, ms) in groups.iter().enumerate() {
                p.outs[g].buf.k = ms.len();
                for (j, &k) in ms.iter().enumerate() {
                    map[k] = (g, j);
                }
            }
            p.engine.product(self.ctx, &self.ex.compute, sub, planes, &p.ins[..sub.n_in], &mut p.outs[..groups.len()], &map, nf_out, None)?;
            let mut cx = BtsCx { ctx: self.ctx, ex: &mut *self.ex, sh, ks: self.ks, ops: &mut *self.ops, work: &mut *self.work };
            for (g, &d) in drops.iter().enumerate() {
                let mut fin = BtsCall::resume(StagedRef::StcFirst(r), r.state_for_drop(&call.st, d)?, Phase::Split, call.opts);
                fin.split = call.split;
                fin.at_split(&cx, &p.outs[g..g + 1])?;
                fin.run_to(&mut cx, &mut p.outs[g..g + 1], Phase::Done)?;
            }
            self.ex.join_aux()?;
            let s: &Stream<Compute> = match self.fcb {
                Some((fs, fe)) => {
                    self.ex.compute.record(fe)?;
                    fs.wait(fe)?;
                    fs
                }
                None => &self.ex.compute,
            };
            // PC_OWN_LAZY: a far group whose every output the ownership below leaves in the replica's slots is not
            // copied back here (its version lives there only; the replica plan copies it over where this device reads it)
            let lazy_slots: Option<Vec<u32>> = self.own.next.as_ref().filter(|(v, _)| crate::replica::own_lazy() && lazy_path(2) && self.fcb.is_none() && self.replica.is_some() && v.len() == groups.iter().map(Vec::len).sum::<usize>()).map(|(v, _)| v.clone());
            let mut at0 = 0usize;
            for (g, o) in outs.iter_mut().enumerate() {
                let m = p.outs[g].meta;
                let k = groups[g].len();
                let skip = lazy_slots.as_ref().is_some_and(|v| v[at0..at0 + k].iter().all(|sl| self.own.want.contains(sl) && !self.own.done.contains(sl)));
                at0 += k;
                if skip {
                    o.meta = m;
                    continue;
                }
                // SAFETY: the outputs were written by the finishes just issued on this stream; `here` reads them only
                // after `join`
                unsafe { batch::take_batch_peer(self.here_ctx, s, o, self.here, &p.outs[g], self.there, m)? };
            }
            // ownership: the far outputs, times 2^shift as this device takes them, into the replica's slots too (after
            // the copies back on this stream; with the copy-back stream they are copied over from here instead)
            if let Some((slots, shift)) = self.own.next.take().filter(|(v, _)| self.fcb.is_none() && v.len() == groups.iter().map(Vec::len).sum::<usize>()) {
                let mut at = 0;
                for (g, ms) in groups.iter().enumerate() {
                    if shift > 0 {
                        p.engine.mult_pow2(self.ctx, &self.ex.compute, &mut p.outs[g], shift)?;
                    }
                    let m = p.outs[g].meta;
                    for j in 0..ms.len() {
                        own_copy(self.ctx, &self.ex.compute, &mut self.replica, &mut self.own, slots[at], &p.outs[g], j, m)?;
                        at += 1;
                    }
                }
            }
            tl_remote(s);
            s.record(self.join)?;
            Ok(())
        })();
        let back = self.here.make_current();
        res?;
        back?;
        REMOTE_JOINT_LINEARS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        REMOTE_JOINT_OUTPUTS.fetch_add(sub.n_out as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn rows_holds(&self, lin: &str) -> bool {
        self.pcmm.as_ref().is_some_and(|p| p.planes.contains_key(lin))
    }

    fn rows_stc(&mut self, here: &Stream<Compute>, sh: &Shadow, call: &BtsCall<'_, 'a, W, M>, xs: &mut [Ct<'a, W>], back: bool) -> pc_gpu::Result<()> {
        let r = self.stcf.ok_or_else(no_joint)?;
        if self.pcmm.as_ref().is_none_or(|p| xs.len() > p.ins.len()) || xs.iter().any(|x| x.k() != 1) {
            return Err(pc_gpu::Error::Mem { knob: "remote pcmm", value: xs.len(), msg: "row-split inputs past the remote engine's input ciphertexts".into() });
        }
        here.record(self.fork)?;
        self.there.make_current()?;
        let res = (|| -> pc_gpu::Result<()> {
            self.ex.compute.wait(self.fork)?;
            let mut at = 0;
            for chunk in xs.chunks_mut(stcf_batch(r)) {
                {
                    let s = &self.ex.compute;
                    for (c, x) in chunk.iter().enumerate() {
                        // SAFETY: the remote stream waits for `fork`, recorded on `here` after every write of the inputs
                        unsafe { batch::put_peer(self.ctx, s, self.gather, self.there, c, x, self.here)? };
                    }
                    self.gather.buf.k = chunk.len();
                    self.gather.meta = chunk[0].meta;
                }
                let mut cx = BtsCx { ctx: self.ctx, ex: &mut *self.ex, sh, ks: self.ks, ops: &mut *self.ops, work: &mut *self.work };
                let mut far = BtsCall::resume(StagedRef::StcFirst(r), call.st.clone(), Phase::Begun, call.opts);
                far.stc_first_input(&mut cx, std::slice::from_mut(&mut *self.gather))?;
                self.ex.join_aux()?;
                let m = self.gather.meta;
                let s = &self.ex.compute;
                let p = self.pcmm.as_mut().ok_or_else(no_joint)?;
                for (c, o) in chunk.iter_mut().enumerate() {
                    batch::take(self.ctx, s, &mut p.ins[at + c], self.gather, c, m)?;
                    if back {
                        // SAFETY: as joint_stc
                        unsafe { batch::take_peer(self.here_ctx, s, o, self.here, self.gather, self.there, c, m)? };
                    }
                }
                at += chunk.len();
            }
            Ok(())
        })();
        let back_cur = self.here.make_current();
        res?;
        back_cur?;
        REMOTE_STC_INPUTS.fetch_add(xs.len() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn rows_product(&mut self, lin: &PcmmLinear, sub: &PcmmShape) -> pc_gpu::Result<(*const u32, *const u32)> {
        self.there.make_current()?;
        let res = (|| -> pc_gpu::Result<(*const u32, *const u32)> {
            // ordered after rows_stc on this stream (the same linear's)
            let p = self.pcmm.as_mut().ok_or_else(no_joint)?;
            let planes = p.planes.get(&lin.name).ok_or_else(|| pc_gpu::Error::Mem { knob: "remote pcmm", value: 0, msg: format!("no row planes of {}", lin.name) })?;
            p.engine.product_rows(self.ctx, &self.ex.compute, sub, planes, &p.ins[..sub.n_in])?;
            tl_remote(&self.ex.compute);
            self.ex.compute.record(self.join)?;
            Ok(p.engine.partial_ptrs())
        })();
        let back = self.here.make_current();
        let v = res?;
        back?;
        Ok(v)
    }

    fn rows_wait(&mut self, here: &Stream<Compute>) -> pc_gpu::Result<()> {
        here.wait(self.join)?;
        Ok(())
    }

    fn rows_far(&mut self, here: &Stream<Compute>, sh: &Shadow, sub: &PcmmShape, call: &BtsCall<'_, 'a, W, M>, here_partial: (*const u32, *const u32), far_outs: &[usize], meta: &CtMeta, nf_out: f64, groups: &[Vec<usize>], drops: &[u32], outs: &mut [Ct<'a, W>]) -> pc_gpu::Result<()> {
        let r = self.stcf.ok_or_else(no_joint)?;
        if groups.len() != outs.len() || drops.len() != outs.len() || self.pcmm.as_ref().is_none_or(|p| groups.len() > p.outs.len()) {
            return Err(pc_gpu::Error::Mem { knob: "remote pcmm", value: groups.len(), msg: "far groups past the remote engine's outputs".into() });
        }
        here.record(self.fork)?;
        self.there.make_current()?;
        let res = (|| -> pc_gpu::Result<()> {
            self.ex.compute.wait(self.fork)?;
            let p = self.pcmm.as_mut().ok_or_else(no_joint)?;
            let mut map = vec![(usize::MAX, 0); far_outs.len()];
            for (g, ms) in groups.iter().enumerate() {
                p.outs[g].buf.k = ms.len();
                for (j, &k) in ms.iter().enumerate() {
                    map[k] = (g, j);
                }
            }
            // this device's partial (peer memory, complete: `fork` follows its product here) for the remote outputs
            p.engine.add_partial(self.ctx, &self.ex.compute, sub, here_partial, far_outs)?;
            p.engine.product_outputs(self.ctx, &self.ex.compute, sub, far_outs, &mut p.outs[..groups.len()], &map, meta, nf_out)?;
            let mut cx = BtsCx { ctx: self.ctx, ex: &mut *self.ex, sh, ks: self.ks, ops: &mut *self.ops, work: &mut *self.work };
            for (g, &d) in drops.iter().enumerate() {
                let mut fin = BtsCall::resume(StagedRef::StcFirst(r), r.state_for_drop(&call.st, d)?, Phase::Split, call.opts);
                fin.split = call.split;
                fin.at_split(&cx, &p.outs[g..g + 1])?;
                fin.run_to(&mut cx, &mut p.outs[g..g + 1], Phase::Done)?;
            }
            self.ex.join_aux()?;
            let s: &Stream<Compute> = match self.fcb {
                Some((fs, fe)) => {
                    self.ex.compute.record(fe)?;
                    fs.wait(fe)?;
                    fs
                }
                None => &self.ex.compute,
            };
            for (g, o) in outs.iter_mut().enumerate() {
                let m = p.outs[g].meta;
                // SAFETY: as joint_far
                unsafe { batch::take_batch_peer(self.here_ctx, s, o, self.here, &p.outs[g], self.there, m)? };
            }
            tl_remote(s);
            s.record(self.join)?;
            Ok(())
        })();
        let back = self.here.make_current();
        res?;
        back?;
        REMOTE_ROWS_LINEARS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn stc_inputs(&mut self, here: &Stream<Compute>, sh: &Shadow, call: &BtsCall<'_, 'a, W, M>, xs: &mut [Ct<'a, W>]) -> pc_gpu::Result<()> {
        let r = self.stcf.ok_or_else(|| pc_gpu::Error::Mem { knob: "remote stc", value: 0, msg: "no StC-first route on the remote device".into() })?;
        if xs.iter().any(|x| x.k() != 1) {
            return Err(pc_gpu::Error::Mem { knob: "remote stc", value: xs.len(), msg: "inputs of one ciphertext each".into() });
        }
        here.record(self.fork)?;
        self.there.make_current()?;
        let res = (|| -> pc_gpu::Result<()> {
            self.ex.compute.wait(self.fork)?;
            for chunk in xs.chunks_mut(stcf_batch(r)) {
                {
                    let s = &self.ex.compute;
                    for (c, x) in chunk.iter().enumerate() {
                        // SAFETY: the remote stream waits for `fork`, recorded on `here` after every write of the inputs
                        unsafe { batch::put_peer(self.ctx, s, self.gather, self.there, c, x, self.here)? };
                    }
                    self.gather.buf.k = chunk.len();
                    self.gather.meta = chunk[0].meta;
                }
                let mut cx = BtsCx { ctx: self.ctx, ex: &mut *self.ex, sh, ks: self.ks, ops: &mut *self.ops, work: &mut *self.work };
                // the same input call on this device's copy of the route
                let mut far = BtsCall::resume(StagedRef::StcFirst(r), call.st.clone(), Phase::Begun, call.opts);
                far.stc_first_input(&mut cx, std::slice::from_mut(&mut *self.gather))?;
                self.ex.join_aux()?;
                let m = self.gather.meta;
                let s = &self.ex.compute;
                for (c, o) in chunk.iter_mut().enumerate() {
                    // SAFETY: the gather buffer was written by the stage just issued on this stream; `here` reads the
                    // inputs only after `join`
                    unsafe { batch::take_peer(self.here_ctx, s, o, self.here, self.gather, self.there, c, m)? };
                }
            }
            tl_remote(&self.ex.compute);
            self.ex.compute.record(self.join)?;
            Ok(())
        })();
        let back = self.here.make_current();
        res?;
        back?;
        REMOTE_STC_INPUTS.fetch_add(xs.len() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn stc_high_far(&mut self, here: &Stream<Compute>, sh: &Shadow, xs: &mut [Ct<'a, W>], spec: &StcHigh) -> pc_gpu::Result<()> {
        let r = self.stcf.ok_or_else(|| pc_gpu::Error::Mem { knob: "remote stc_high", value: 0, msg: "no StC-first route on the remote device".into() })?;
        if xs.iter().any(|x| x.k() != 1 || x.meta != xs[0].meta) {
            return Err(pc_gpu::Error::Mem { knob: "remote stc_high", value: xs.len(), msg: "operands of one ciphertext each at one metadata".into() });
        }
        here.record(self.fork)?;
        self.there.make_current()?;
        let res = (|| -> pc_gpu::Result<()> {
            self.ex.compute.wait(self.fork)?;
            for chunk in xs.chunks_mut(stcf_batch(r)) {
                {
                    let s = &self.ex.compute;
                    for (c, x) in chunk.iter().enumerate() {
                        // SAFETY: the remote stream waits for `fork`, recorded on `here` after every write of the operands
                        unsafe { batch::put_peer(self.ctx, s, self.gather, self.there, c, x, self.here)? };
                    }
                    self.gather.buf.k = chunk.len();
                    self.gather.meta = chunk[0].meta;
                }
                let mut cx = BtsCx { ctx: self.ctx, ex: &mut *self.ex, sh, ks: self.ks, ops: &mut *self.ops, work: &mut *self.work };
                Staged::stc_high(r, &mut cx, std::slice::from_mut(&mut *self.gather), spec)?;
                self.ex.join_aux()?;
                let m = self.gather.meta;
                let s = &self.ex.compute;
                for (c, o) in chunk.iter_mut().enumerate() {
                    // SAFETY: the gather buffer was written by the conversion just issued on this stream; `here` reads the
                    // operands only after `join`
                    unsafe { batch::take_peer(self.here_ctx, s, o, self.here, self.gather, self.there, c, m)? };
                }
            }
            tl_remote(&self.ex.compute);
            self.ex.compute.record(self.join)?;
            Ok(())
        })();
        let back = self.here.make_current();
        res?;
        back?;
        REMOTE_CCMM_OPERANDS.fetch_add(xs.len() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn stcf_singles(&mut self, here: &Stream<Compute>, sh: &Shadow, xs: &mut [&mut Ct<'a, W>], f: &mut StcfFn<'_, 'a, W, M>) -> pc_gpu::Result<()> {
        self.singles_from(here, sh, None, xs, f)
    }

    fn ccmm_split_on(&self) -> bool {
        self.ccmm.is_some() && self.stcf.is_some()
    }

    fn ccmm_operands_far(&mut self, here: &Stream<Compute>, sh: &Shadow, side: usize, xs: &mut [Ct<'a, W>], spec: &StcHigh, back: bool) -> pc_gpu::Result<()> {
        let r = self.stcf.ok_or_else(|| pc_gpu::Error::Mem { knob: "remote stc_high", value: 0, msg: "no StC-first route on the remote device".into() })?;
        let hb = pc_core::ccmm_keys::NBIG / 2;
        if side > 1 || xs.len() != hb || xs.iter().any(|x| x.k() != 1 || x.meta != xs[0].meta) || self.ccmm.is_none() {
            return Err(pc_gpu::Error::Mem { knob: "ccmm split", value: xs.len(), msg: format!("{hb} operands of one ciphertext each at one metadata (side {side})") });
        }
        here.record(self.fork)?;
        self.there.make_current()?;
        let res = (|| -> pc_gpu::Result<()> {
            self.ex.compute.wait(self.fork)?;
            let mut done = 0;
            for chunk in xs.chunks_mut(stcf_batch(r)) {
                {
                    let s = &self.ex.compute;
                    for (c, x) in chunk.iter().enumerate() {
                        // SAFETY: the remote stream waits for `fork`, recorded on `here` after every write of the operands
                        unsafe { batch::put_peer(self.ctx, s, self.gather, self.there, c, x, self.here)? };
                    }
                    self.gather.buf.k = chunk.len();
                    self.gather.meta = chunk[0].meta;
                }
                let mut cx = BtsCx { ctx: self.ctx, ex: &mut *self.ex, sh, ks: self.ks, ops: &mut *self.ops, work: &mut *self.work };
                Staged::stc_high(r, &mut cx, std::slice::from_mut(&mut *self.gather), spec)?;
                self.ex.join_aux()?;
                let m = self.gather.meta;
                let s = &self.ex.compute;
                let rc = self.ccmm.as_mut().ok_or_else(no_split)?;
                for (c, o) in chunk.iter_mut().enumerate() {
                    batch::take(self.ctx, s, &mut rc.cts[side * hb + done + c], self.gather, c, m)?;
                    if back {
                        // SAFETY: the gather buffer was written by the conversion just issued on this stream; `here`
                        // reads the operands only after `join`
                        unsafe { batch::take_peer(self.here_ctx, s, o, self.here, self.gather, self.there, c, m)? };
                    } else {
                        o.meta = m;
                    }
                }
                done += chunk.len();
            }
            tl_remote(&self.ex.compute);
            self.ex.compute.record(self.join)?;
            Ok(())
        })();
        let back_cur = self.here.make_current();
        res?;
        back_cur?;
        REMOTE_CCMM_OPERANDS.fetch_add(xs.len() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn ccmm_mul_split(&mut self, here: &Stream<Compute>, eng: &mut Ccmm<'a, 'static, W>, ctx: &Context<'a, W, M>, a: &mut [Ct<'a, W>], b: &[Ct<'a, W>], bt: bool) -> pc_gpu::Result<()> {
        let nb = pc_core::ccmm_keys::NBIG;
        let hb = nb / 2;
        if a.len() != nb || b.len() != nb {
            return Err(pc_gpu::Error::Mem { knob: "ccmm split", value: a.len(), msg: format!("{} / {} operands, a matrix is {nb}", a.len(), b.len()) });
        }
        let meta = pc_gpu::bts::common_meta(a)?;
        let rc = self.ccmm.as_mut().ok_or_else(no_split)?;
        let (ta, tb) = rc.cts.split_at_mut(hb);
        let oh = ccmm_eng::SplitOps { a: &ccmm_eng::split_members(ctx, "a", &a[..hb], hb)?, b: &ccmm_eng::split_members(ctx, "b", &b[..hb], hb)?, out: &ccmm_eng::split_out_members(&mut a[..hb]) };
        let ot = ccmm_eng::SplitOps { a: &ccmm_eng::split_members(self.ctx, "a", ta, hb)?, b: &ccmm_eng::split_members(self.ctx, "b", &tb[..hb], hb)?, out: &ccmm_eng::split_out_members(ta) };
        let mut h = ccmm_eng::SplitSide { eng, ctx, dev: self.here, s: here, cp: rc.cp_h, ev: rc.ev_h, ph: rc.ph_h };
        let mut t = ccmm_eng::SplitSide { eng: &mut *rc.engine, ctx: self.ctx, dev: self.there, s: &self.ex.compute, cp: rc.cp_t, ev: rc.ev_t, ph: rc.ph_t };
        ccmm_eng::mul_split(&mut h, &mut t, &oh, &ot, bt)?;
        for x in a.iter_mut().chain(ta.iter_mut()) {
            x.meta = meta;
        }
        REMOTE_CCMM_SPLITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn ccmm_far_merge(&mut self, g: usize) -> pc_gpu::Result<()> {
        let hb = pc_core::ccmm_keys::NBIG / 2;
        if g < 2 || hb % g != 0 {
            return Err(pc_gpu::Error::Mem { knob: "ccmm split", value: g, msg: format!("merge {g} of {hb} outputs") });
        }
        self.there.make_current()?;
        let res = (|| -> pc_gpu::Result<()> {
            let rc = self.ccmm.as_mut().ok_or_else(no_split)?;
            for i in 0..hb / g {
                for k in (0..g - 1).rev() {
                    let (lo, hi) = rc.cts.split_at_mut(g * i + k + 1);
                    pc_gpu::limbs::merge_monomial(self.ctx, self.ex, rc.mono, &mut lo[g * i + k], &hi[0])?;
                }
            }
            for i in 1..hb / g {
                rc.cts.swap(i, g * i);
            }
            Ok(())
        })();
        let back = self.here.make_current();
        res?;
        back?;
        Ok(())
    }

    fn ccmm_far_back(&mut self, here: &Stream<Compute>, first: usize, xs: &mut [Ct<'a, W>]) -> pc_gpu::Result<()> {
        let rc = self.ccmm.as_mut().ok_or_else(no_split)?;
        if first + xs.len() > pc_core::ccmm_keys::NBIG / 2 {
            return Err(pc_gpu::Error::Mem { knob: "ccmm split", value: first + xs.len(), msg: "remote outputs past the half".into() });
        }
        self.there.make_current()?;
        let r = self.ex.compute.record(self.join);
        let back = self.here.make_current();
        r?;
        back?;
        // this device's copy stream pulls (after the remote product, merges and everything before them there)
        let cp = rc.cp_h;
        cp.wait(self.join)?;
        for (c, o) in xs.iter_mut().enumerate() {
            let src = &rc.cts[first + c];
            // SAFETY: `cp` waits for the remote stream's event after the outputs' writes; nothing writes them before
            // the remote device's next split product, ordered after this device's (which waits for `back`)
            unsafe { batch::take_peer(self.here_ctx, cp, o, self.here, src, self.there, 0, src.meta)? };
        }
        rc.back.record(cp)?;
        here.wait(rc.back)?;
        Ok(())
    }

    fn ccmm_far_singles(&mut self, here: &Stream<Compute>, sh: &Shadow, first: usize, xs: &mut [&mut Ct<'a, W>], f: &mut StcfFn<'_, 'a, W, M>) -> pc_gpu::Result<()> {
        if self.ccmm.is_none() || first + xs.len() > pc_core::ccmm_keys::NBIG / 2 {
            return Err(no_split());
        }
        self.singles_from(here, sh, Some(first), xs, f)
    }

    fn ccmm_split_phases(&mut self) -> pc_gpu::Result<Option<[[f64; 6]; 2]>> {
        let Some(rc) = self.ccmm.as_ref() else { return Ok(None) };
        let (Some(ph), Some(pt)) = (rc.ph_h, rc.ph_t) else { return Ok(None) };
        let d = |e: &[Event]| -> pc_gpu::Result<[f64; 6]> {
            e[5].sync()?;
            let mut v = [0.0; 6];
            for i in 0..5 {
                v[i] = e[i].elapsed_ms(&e[i + 1])? as f64;
            }
            v[5] = e[0].elapsed_ms(&e[5])? as f64;
            Ok(v)
        };
        let here = d(ph)?;
        self.there.make_current()?;
        let there = d(pt);
        let back = self.here.make_current();
        let there = there?;
        back?;
        Ok(Some([here, there]))
    }
}

impl<'r, 'a, W: DevWord, M: DevStrategy<W>> Remote<'r, 'a, W, M> {
    /// `stcf_singles` with the inputs copied over from `xs` (`local` None) or taken from the remote CCMM outputs
    /// `local ..` where they are.
    fn singles_from(&mut self, here: &Stream<Compute>, sh: &Shadow, local: Option<usize>, xs: &mut [&mut Ct<'a, W>], f: &mut StcfFn<'_, 'a, W, M>) -> pc_gpu::Result<()> {
        let r = self.stcf.ok_or_else(|| pc_gpu::Error::Mem { knob: "remote finish", value: 0, msg: "no StC-first route on the remote device".into() })?;
        if xs.is_empty() || xs.len() > crate::program::GRID_INPUTS {
            return Err(pc_gpu::Error::Mem { knob: "remote finish", value: xs.len(), msg: format!("{} ciphertexts for a {}-ciphertext batch buffer", xs.len(), crate::program::GRID_INPUTS) });
        }
        here.record(self.fork)?;
        self.there.make_current()?;
        let res = (|| -> pc_gpu::Result<()> {
            let s = &self.ex.compute;
            s.wait(self.fork)?;
            for (c, x) in xs.iter().enumerate() {
                match (local, self.ccmm.as_ref()) {
                    (Some(first), Some(rc)) => batch::put(self.ctx, s, self.gather, c, &rc.cts[first + c])?,
                    // SAFETY: the remote stream waits for `fork`, recorded on `here` after every write of the inputs
                    _ => unsafe { batch::put_peer(self.ctx, s, self.gather, self.there, c, x, self.here)? },
                }
            }
            self.gather.buf.k = xs.len();
            self.gather.meta = xs[0].meta;
            let mut cx = BtsCx { ctx: self.ctx, ex: &mut *self.ex, sh, ks: self.ks, ops: &mut *self.ops, work: &mut *self.work };
            f(&mut cx, r, &mut *self.gather)?;
            self.ex.join_aux()?;
            let m = self.gather.meta;
            let s: &Stream<Compute> = match self.fcb {
                Some((fs, fe)) => {
                    self.ex.compute.record(fe)?;
                    fs.wait(fe)?;
                    fs
                }
                None => &self.ex.compute,
            };
            // PC_OWN_LAZY: outputs the ownership below leaves in the replica's slots are not copied back here
            let lazy_slots: Option<Vec<u32>> = self.own.next.as_ref().filter(|(v, sh)| crate::replica::own_lazy() && lazy_path(4) && self.fcb.is_none() && self.replica.is_some() && v.len() == xs.len() && (*sh == 0 || self.pcmm.is_some())).map(|(v, _)| v.clone());
            for (c, o) in xs.iter_mut().enumerate() {
                if lazy_slots.as_ref().is_some_and(|v| self.own.want.contains(&v[c]) && !self.own.done.contains(&v[c])) {
                    o.meta = m;
                    continue;
                }
                // SAFETY: the gather buffer was written by the refresh just issued on this stream; `here` reads the
                // outputs only after `join`
                unsafe { batch::take_peer(self.here_ctx, s, o, self.here, self.gather, self.there, c, m)? };
            }
            // ownership: the refreshed outputs, times 2^shift as this device takes them (the PCMM engine's table),
            // into the replica's slots too
            if let Some((slots, shift)) = self.own.next.take().filter(|(v, _)| self.fcb.is_none() && v.len() == xs.len()) {
                if shift == 0 || self.pcmm.is_some() {
                    if let (true, Some(p)) = (shift > 0, self.pcmm.as_ref()) {
                        p.engine.mult_pow2(self.ctx, &self.ex.compute, &mut *self.gather, shift)?;
                    }
                    for (c, &slot) in slots.iter().enumerate() {
                        own_copy(self.ctx, &self.ex.compute, &mut self.replica, &mut self.own, slot, &*self.gather, c, m)?;
                    }
                }
            }
            tl_remote(s);
            s.record(self.join)?;
            Ok(())
        })();
        let back = self.here.make_current();
        res?;
        back?;
        REMOTE_CCMM_RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        REMOTE_CCMM_CTS.fetch_add(xs.len() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
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
    /// The remote refresh device's constant-ring slots leased by the capture.
    remote_lease: Vec<usize>,
    out_metas: Vec<CtMeta>,
    pt_rows: Vec<*const c_void>,
    /// The slots' buffers it baked (a driver may lend a slot another buffer for a run: pc-py `run_block`).
    slot_rows: Vec<*const c_void>,
    stream_base: *const f32,
    pub capture_ms: f64,
}

fn slot_rows<W: DevWord>(b: &Bound<'_, W>) -> Vec<*const c_void> {
    b.slots.iter().map(|c| c.buf.r().base as *const c_void).collect()
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
    /// Refresh pairs issued as one complex refresh and one by one (`ReplayStats::pairs`).
    pub pairs: [usize; 2],
    /// Plaintext products and adds issued as one pass (`ReplayStats::fused`).
    pub fused: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ReplayStats {
    pub eager: usize,
    pub captures: usize,
    pub replays: usize,
    /// Graphs dropped because a plaintext address moved.
    pub invalidations: usize,
    pub capture_ms: f64,
    /// Refresh pairs the eager runs issued as one complex refresh, and one by one (operand scales in two cache keys).
    pub pairs: usize,
    pub pairs_split: usize,
    /// Plaintext products and adds the eager runs issued as one pass (`UnitKind::MultPtAdd`).
    pub fused: usize,
    /// Blocks whose capture ran out of device memory: they run eager from then on (`Replayer::eager_only`).
    pub graph_oom: usize,
}

pub struct Replayer {
    pub cfg: ReplayCfg,
    pub stats: ReplayStats,
    /// Use the runtime's side lane (`Rt::side`) when it has one.
    pub side_lane: bool,
    /// Lane schedules by block label, for the dynamic-step set they were made for.
    scheds: std::collections::HashMap<String, (Vec<u32>, Sched)>,
    /// Labels of the blocks whose graph did not fit in device memory (instantiation out of memory): with graphs on,
    /// they run eager, so a model whose graphs exceed the free memory replays the blocks that fit.
    eager_only: std::collections::HashSet<String>,
    /// Blocks whose PC_GRAPH_EAGER_FINISH steps were made dynamic (before their first capture).
    eager_marked: std::collections::HashSet<String>,
    /// Per block label: the dynamic-step set of its last run and whether that run put anything on the replica lane.
    replica_last: std::collections::HashMap<String, (Vec<u32>, bool)>,
    /// PC_SEG_TIMING (diagnostic): timing events of this device (set by the driver before a run), one recorded on
    /// compute before each item of a graph replay (a segment or a dynamic step) and one after the last; `seg_kinds` /
    /// `seg_host_us` the items replayed (step index, or usize::MAX for a graph segment) and their host issue times.
    pub seg_ev: Vec<pc_cuda::stream::Event>,
    pub seg_kinds: Vec<usize>,
    pub seg_host_us: Vec<f64>,
}

/// PC_TL=<block label> (diagnostic): timing events recorded after every step of that block on the device it ran on
/// (this device's compute stream, or the replica's stream), events of this device in `a`, of the remote one in `b`
/// (the driver creates them, `program.rs`); `marks` the (step, device, event index) recorded by the last run.
pub struct Tl {
    pub label: String,
    /// mapped portable host memory, `cap` words: every stamp's %globaltimer (ns)
    pub buf: *mut u64,
    pub cap: usize,
    pub marks: Vec<(u32, u8, u32)>,
    pub n: usize,
    pub on: bool,
    /// the step being issued (remote stamps are booked to it)
    pub cur: u32,
}

thread_local! {
    pub static TL: std::cell::RefCell<Option<Tl>> = const { std::cell::RefCell::new(None) };
}

/// PC_TL: a stamp on a remote-device stream (`s`, current device: the remote one) at the end of its part of the
/// current step (device code 4).
fn tl_remote(s: &Stream<Compute>) {
    if !tl_active() {
        return;
    }
    TL.with(|t| {
        let mut g = t.borrow_mut();
        let Some(tl) = g.as_mut().filter(|tl| tl.on && tl.n < tl.cap) else { return };
        // SAFETY: mapped portable host memory of `cap` words
        let e = unsafe { pc_kernels::pc_stamp(tl.buf, tl.n as i32, s.raw() as *mut std::ffi::c_void) };
        if e == 0 {
            tl.marks.push((tl.cur, 4, tl.n as u32));
            tl.n += 1;
        }
    });
}

fn tl_active() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("PC_TL").is_ok_and(|v| !v.is_empty()))
}

/// PC_TL: after step `i` (or the unit it starts) of `label`, on lane `lane` (0 this device, 4 the replica).
fn tl_mark<'a, W: DevWord, M: DevStrategy<W>>(rt: &mut Rt<'_, 'a, W, M>, i: usize, lane: u8, fork: bool) -> Result<()> {
    TL.with(|t| -> Result<()> {
        let mut g = t.borrow_mut();
        let Some(tl) = g.as_mut().filter(|tl| tl.on && tl.n < tl.cap) else { return Ok(()) };
        match lane {
            0 => {
                // SAFETY: mapped portable host memory of `cap` words
                let e = unsafe { pc_kernels::pc_stamp(tl.buf, tl.n as i32, rt.ex.compute.raw() as *mut std::ffi::c_void) };
                if e != 0 {
                    return Err(rerr("PC_TL", format!("stamp launch: CUDA error {e}")));
                }
                tl.marks.push((i as u32, if fork { 2 } else { 0 }, tl.n as u32));
                tl.n += 1;
            }
            4 => {
                if let Some(rm) = rt.remote.as_deref_mut() {
                    rm.replica_stamp(tl.buf, tl.n as i32)?;
                    tl.marks.push((i as u32, if fork { 3 } else { 1 }, tl.n as u32));
                    tl.n += 1;
                }
            }
            _ => {}
        }
        Ok(())
    })
}

/// A capture failed for lack of device memory (cudaErrorMemoryAllocation in the capture or the instantiation).
fn is_oom(e: &pc_cuda::Error) -> bool {
    matches!(e, pc_cuda::Error::Cuda { code: 2, .. })
}

fn rerr(block: &str, msg: impl Into<String>) -> Error {
    Error::Replay { block: block.to_string(), msg: msg.into() }
}

/// `dst = src` (live Q rows, both parts) and its metadata, on `s` (one kernel).
/// `copy_ct` across the devices by DMA (`cudaMemcpyAsync` over the bus, capturable): `src` (memory of `sd`) into `dst`
/// (memory of `dd`) on `s`, the live Q rows of each ciphertext's two parts, metadata included.
///
/// # Safety
/// `s` must be ordered after `src`'s producer and after every access of `dst`'s old content; nothing may write `src`
/// or touch `dst` before an event recorded on `s` after the copy.
pub unsafe fn copy_ct_dma<W: DevWord, M: DevStrategy<W>>(ctx: &Context<'_, W, M>, s: &Stream<Compute>, dst: &mut Ct<'_, W>, dd: &pc_cuda::Device, src: &Ct<'_, W>, sd: &pc_cuda::Device) -> Result<()> {
    let top = src.top(ctx.shape)?;
    if top >= dst.buf.lay.nq.min(src.buf.lay.nq) || src.k() > dst.k() || dst.buf.seeds || dst.buf.packed || src.buf.seeds || src.buf.packed || dst.buf.lay.n != src.buf.lay.n {
        return Err(Error::Replay { block: String::new(), msg: format!("DMA copy of a ciphertext at level {} (top limb {top}, k {}) into a buffer of {} Q rows (k {})", src.meta.level, src.k(), dst.buf.lay.nq, dst.k()) });
    }
    let n = src.buf.lay.n;
    for c in 0..src.k() {
        for part in 0..2 {
            let (od, os) = (dst.buf.offset(c, part, 0), src.buf.offset(c, part, 0));
            dst.buf.buf.copy_from_peer(s, dd, od, &src.buf.buf, sd, os, (top + 1) * n)?;
        }
    }
    dst.meta = src.meta;
    Ok(())
}

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
    mult_sums: &'t [Vec<u32>],
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
        let remote: Option<&mut dyn RemoteLane<'a, W, M>> = match rt.remote.as_deref_mut() {
            Some(r) => Some(r),
            None => None,
        };
        return svc.finish(&mut cx, fs, all, remote);
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
    if let StepOp::MultSum { s } = &st.op {
        let ms = t.mult_sums.get(*s as usize).ok_or_else(|| rerr(block, format!("mult_sum {s} out of range")))?;
        if ms.len() < 4 || ms.len() % 2 != 0 || ms.iter().chain(std::iter::once(&st.out)).any(|&x| x as usize >= n_slots) || ms.contains(&st.out) {
            return Err(rerr(block, format!("mult_sum {s}: {} operand slots, slots outside the {n_slots} bound slots, or the output among them", ms.len())));
        }
        // SAFETY: bind checked the slots (< n_slots, output not an operand); the output and `tmp` are the only
        // mutable references
        let o: &mut Ct<'a, W> = unsafe { &mut *slots.add(st.out as usize) };
        let pairs: Vec<(&Ct<'a, W>, &Ct<'a, W>)> = ms.chunks(2).map(|ab| unsafe { (&*slots.add(ab[0] as usize), &*slots.add(ab[1] as usize)) }).collect();
        let kc = KsCfg { plans: rt.ks, source: rt.source };
        let key = group(block, t.groups, t.relin)?;
        rt.ops.mult_sum(ctx, rt.ex, kc, sh, key, o, tmp, &pairs)?;
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
            if st.inplace && st.b == st.out && rt.ops.add_self_inplace(ctx, rt.ex, sh, o, sub)? {
                // x + x in one pass (no copy of x)
            } else if st.inplace {
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
        StepOp::MultRescale { square, dbl } => {
            let key = group(block, t.groups, t.relin)?;
            let b = if *square { None } else { Some(at(st.b)) };
            if st.inplace && (*square || st.b != st.out) {
                rt.ops.mult_rescale(ctx, rt.ex, kc, sh, key, o, None, b, *dbl)?
            } else {
                let a = src_a!();
                rt.ops.mult_rescale(ctx, rt.ex, kc, sh, key, o, Some(a), b, *dbl)?
            }
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
        StepOp::MultSum { .. } => return Err(rerr(block, "mult_sum step reached the single-output lowering")),
        StepOp::Bootstrap { route, kind, .. } => {
            BTS_SINGLE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
    let tp = Instant::now();
    issue_range(rt, b, 0, n, check, info, hook, sched)?;
    if step_prof() {
        rt.ex.compute.sync()?;
        let mut g = PROF.lock().unwrap();
        let now = Instant::now();
        if let Some((c, t)) = g.0.take() {
            g.1.entry(c).or_insert((0.0, 0)).0 += now.duration_since(t).as_secs_f64() * 1e3;
        }
        let line: Vec<String> = g.1.iter().map(|(k, v)| format!("{k}={}:{:.1}", v.1, v.0)).collect();
        eprintln!("[stepprof] {} total_ms={:.1} | {}", b.label, tp.elapsed().as_secs_f64() * 1e3, line.join(" | "));
        g.1.clear();
    }
    Ok(())
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
static PROF: std::sync::Mutex<(Option<(String, Instant)>, std::collections::BTreeMap<String, (f64, u64)>)> = std::sync::Mutex::new((None, std::collections::BTreeMap::new()));

fn step_prof() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("PC_STEP_PROFILE").is_ok_and(|v| v != "0" && !v.is_empty()))
}

/// PC_STEP_PROFILE_MAP=<json {block label: {var name: graph step}}>: the profile's classes are the graph steps the
/// steps' output vars were recorded in (`<step>`), booked by step instead of by op class.
fn prof_map() -> Option<&'static std::collections::HashMap<String, std::collections::HashMap<String, String>>> {
    static MAP: std::sync::OnceLock<Option<std::collections::HashMap<String, std::collections::HashMap<String, String>>>> = std::sync::OnceLock::new();
    MAP.get_or_init(|| {
        let p = std::env::var("PC_STEP_PROFILE_MAP").ok().filter(|v| !v.is_empty())?;
        let txt = std::fs::read_to_string(&p).map_err(|e| eprintln!("[stepprof] {p}: {e}")).ok()?;
        serde_json::from_str(&txt).map_err(|e| eprintln!("[stepprof] {p}: {e}")).ok()
    })
    .as_ref()
}

/// The graph step of var `name` of block `label` (PC_STEP_PROFILE_MAP; `~pre` / `~raw` states as their var).
fn prof_step(label: &str, name: &str) -> Option<String> {
    let m = prof_map()?.get(label)?;
    let base = name.split('~').next().unwrap_or(name);
    Some(m.get(base).cloned().unwrap_or_else(|| "?".into()))
}

/// (class, count) of a step (a unit of `n` steps of one op) for PC_STEP_PROFILE.
fn prof_class(op: &StepOp, n: usize, outs: impl Fn(u32) -> (usize, String)) -> (String, usize) {
    match op {
        StepOp::Finish { f } => {
            let (k, d) = outs(*f);
            (format!("finish.{}", d.replace(' ', "_").replace('|', "/")), k)
        }
        StepOp::Bootstrap { route, kind, .. } => (format!("bts.{kind:?}.{route:?}").replace(' ', ""), n),
        StepOp::Rotate { .. } | StepOp::RotateSum { .. } | StepOp::Conjugate { .. } => ("rot".into(), n),
        StepOp::Mult | StepOp::Square | StepOp::MultRescale { .. } | StepOp::MultSum { .. } => ("ctct".into(), n),
        StepOp::MultPt(_) | StepOp::AddPt(_) | StepOp::AddConst(_) | StepOp::MultConst(_) => ("pt".into(), n),
        StepOp::Rescale | StepOp::DropTo(_) => ("rescale".into(), n),
        StepOp::CoeffSum { .. } => ("coeffsum".into(), n),
        _ => ("add".into(), n),
    }
}

/// Issue step `i` (or the unit `unit` = (index, end) it starts) of a bound block through `rt`'s compute lane: its
/// tables, units, steps and slot buffers, the lane's rotation-chain owners (`owners`: the key-switching workspaces',
/// the held buffers'). `lane0`: a lone step on this lane releases the key-switching workspaces' chain.
#[allow(clippy::too_many_arguments)]
fn exec_at<'a, W: DevWord, M: DevStrategy<W>>(
    label: &str,
    rt: &mut Rt<'_, 'a, W, M>,
    t: &Tables<'_, 'a>,
    units: &[Unit],
    steps: &[Step],
    slots: &mut [Ct<'a, W>],
    tmp: &mut Ct<'a, W>,
    gather: &mut Option<Gather<'a, W>>,
    held: &mut Option<Held<'a, W>>,
    i: usize,
    unit: Option<(u32, usize)>,
    owners: (&mut Option<u32>, &mut Option<u32>),
    lane0: bool,
    fused: &mut usize,
) -> Result<()> {
    let (ks_owner, held_owner) = owners;
    match unit {
        Some((u, _)) => {
            let unit = &units[u as usize];
            let gather = gather.as_mut().ok_or_else(|| rerr(label, "no gather buffer"))?;
            let mut stale = false;
            match unit.kind {
                UnitKind::Grid { held: h, .. } if unit.chain != NONE => {
                    let owner = if h { &mut *held_owner } else { &mut *ks_owner };
                    stale = *owner != Some(unit.chain);
                    *owner = Some(unit.chain);
                }
                // a fused plaintext product and add leaves the key-switching workspaces alone
                UnitKind::MultPtAdd { .. } => *fused += 1,
                UnitKind::MultPtAddChain { ref pts, .. } => *fused += pts.len(),
                UnitKind::Affine { ref steps } => *fused += steps.len(),
                UnitKind::MultIAdd { .. } => *fused += 1,
                _ => *ks_owner = None,
            }
            exec_unit(label, rt, unit, steps, slots.as_mut_ptr(), slots.len(), gather, held.as_mut(), t.groups, t.relin, t.lists, t.pts, stale)
        }
        None => {
            if lane0 && !quiet(&steps[i].op) {
                *ks_owner = None;
            }
            exec_step(label, rt, &steps[i], slots.as_mut_ptr(), slots.len(), tmp, t)
        }
    }
}

fn issue_range<'a, W: DevWord, M: DevStrategy<W>>(rt: &mut Rt<'_, 'a, W, M>, b: &mut Bound<'a, W>, lo: usize, hi: usize, check: bool, info: &mut RunInfo, mut hook: Option<&mut StepHook<'_, 'a, W, M>>, sched: Option<&Sched>) -> Result<()> {
    if b.slots.len() < b.n_slots {
        return Err(rerr(&b.label, format!("{} slots present, the block needs {} (a shared block outside Bound::with_slab)", b.slots.len(), b.n_slots)));
    }
    let sched = if rt.side.is_some() { sched.filter(|s| s.lane.len() == b.steps.len()) } else { None };
    let mut forked = false;
    // the remote lane (schedule lane 2): forked at the start of every region that has a remote step, joined at its end
    let mut rforked = false;
    // the replica lane (lane 4 of a replica plan): forked at the start of every region that has a replica step
    let rsched = sched.and_then(|s| s.replica.as_ref()).filter(|_| rt.remote.as_ref().is_some_and(|r| r.replica_on()));
    let mut rrforked = false;
    // PC_REPLICA_DMA: both copy streams forked for the current region
    let mut dforked = false;
    let mut region_start = true;
    // steps below `unit_end` were issued by the unit that holds them
    let mut unit_end = lo;
    // the rotation chain whose gathered inputs and ModUp the compute lane's workspaces (`ks_owner`) and the held
    // buffers (`held_owner`) hold: a later link of another state runs fresh (bind's chains never meet one, a guard)
    let (mut ks_owner, mut held_owner): (Option<u32>, Option<u32>) = (None, None);
    // a fused product is never written: the metadata check and the step hooks see the pair step by step
    // PC_REPLICA_CHECK=units: the hooks see the fused units as units
    let stepwise = check || (hook.is_some() && !std::env::var("PC_REPLICA_CHECK").is_ok_and(|v| v == "units"));
    for i in lo..hi {
        let in_unit = i < unit_end;
        // the replica plan's copies (and the region's fork) before anything reads this step's slots: the unit choice
        // below reads their metadata
        if let (false, Some(rs), Some(s)) = (in_unit, rsched, sched) {
            if !b.dynamic.contains(&(i as u32)) {
                if region_start && !rrforked {
                    let end = b.dynamic.range(i as u32..).next().map_or(hi, |&d| (d as usize).min(hi));
                    if s.lane[i..end].contains(&4) {
                        rt.remote.as_deref_mut().ok_or_else(no_replica)?.replica_fork(&rt.ex.compute)?;
                        rrforked = true;
                        if tl_active() {
                            // PC_TL: the region's fork on both devices (device codes 2 / 3: before step i)
                            tl_mark(rt, i, 0, true)?;
                            tl_mark(rt, i, 4, true)?;
                        }
                    }
                }
                if region_start && !dforked {
                    let end = b.dynamic.range(i as u32..).next().map_or(hi, |&d| (d as usize).min(hi));
                    if rs.early.get(i..end).is_some_and(|v| v.iter().any(|c| !c.is_empty())) {
                        rt.remote.as_deref_mut().ok_or_else(no_replica)?.dma_fork(&rt.ex.compute)?;
                        dforked = true;
                    }
                }
            }
        }
        // PC_REPLICA_DMA: the copies issued just before this step (each right after the last of its waits), on the
        // receiver's copy stream
        if let Some(rs) = rsched {
            if rs.early.get(i).is_some_and(|c| !c.is_empty()) {
                if !dforked {
                    rt.remote.as_deref_mut().ok_or_else(no_replica)?.dma_fork(&rt.ex.compute)?;
                    dforked = true;
                }
                for &c in &rs.early[i] {
                    issue_early(rt, b, rs, c as usize, lo)?;
                }
            }
        }
        if let (false, Some(rs), Some(_)) = (in_unit, rsched, sched) {
            if !b.dynamic.contains(&(i as u32)) {
                replica_copies(rt, b, rs, i, lo)?;
            }
        }
        let unit = if in_unit {
            None
        } else {
            let at = b.unit_at(i).filter(|u| !(stepwise && matches!(u.kind, UnitKind::MultPtAdd { .. } | UnitKind::MultPtAddChain { .. } | UnitKind::RotAcc { .. } | UnitKind::Affine { .. } | UnitKind::MultIAdd { .. })));
            // a replica unit is checked against the replica's slots (`replica_issue`), whose versions this device may
            // not hold
            let on_rep = sched.is_some_and(|s| s.lane[i] == 4);
            // a product batch the replica's plan split between the devices issues one by one (PC_REPLICA_SPLITK)
            let split = |i: usize| rsched.is_some_and(|rs| b.unit_of.get(i).is_some_and(|&u| u != NONE && rs.split_unit.get(u as usize).copied().unwrap_or(false)));
            let unit = at
                .filter(|_| !split(i))
                .filter(|u| on_rep || unit_fits(u, &b.slots, &b.steps, &b.pts, rt.sh, &rt.routes))
                .map(|u| (b.unit_of[i], u.hi as usize))
                .filter(|&(_, e)| e <= hi && sched.is_none_or(|s| s.lane[i..e].iter().all(|&l| l == s.lane[i])));
            if at.is_some_and(|u| matches!(u.kind, UnitKind::Pair { .. })) {
                // a pair whose scales fall in two cache keys (or that a segment end cuts) issues one by one
                info.pairs[unit.is_none() as usize] += 1;
            }
            unit
        };
        let dynamic = b.dynamic.contains(&(i as u32));
        let lane = sched.map_or(0, |s| s.lane[i]);
        if dynamic && forked {
            join_side(rt)?;
            forked = false;
        }
        if dynamic && rforked {
            rt.remote.as_deref_mut().ok_or_else(|| rerr(&b.label, "remote lane step without a remote device"))?.lane_join(&rt.ex.compute)?;
            rforked = false;
        }
        if let (true, Some(rs)) = (dynamic, rsched) {
            // the dynamic step's copies back (its slots whose version is on the replica only), inside the region before
            // it, then the replica joins compute (a range the step starts, run between two graphs: the segment before
            // issued them)
            if i > lo {
                replica_copies(rt, b, rs, i, lo)?;
            }
            if rrforked {
                rt.remote.as_deref_mut().ok_or_else(no_replica)?.replica_join(&rt.ex.compute)?;
                rrforked = false;
            }
            if dforked {
                rt.remote.as_deref_mut().ok_or_else(no_replica)?.dma_join(&rt.ex.compute)?;
                dforked = false;
            }
        }
        if let Some(s) = sched {
            if region_start && !dynamic {
                // a region with remote steps: the remote lane starts after everything before the region (its steps
                // then wait only for the events of what they read and overwrite)
                let end = b.dynamic.range(i as u32..).next().map_or(hi, |&d| (d as usize).min(hi));
                if s.lane[i..end].contains(&2) {
                    rt.remote.as_deref_mut().ok_or_else(|| rerr(&b.label, "remote lane step without a remote device"))?.lane_fork(&rt.ex.compute)?;
                    rforked = true;
                }

            }
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
                for (pl, &w) in s.wait[j].iter().enumerate() {
                    // an event of a step before `lo` (an earlier graph segment, joined at its end, or a dynamic step
                    // on compute) is already ordered: a capture must not wait on work outside it
                    if w == NONE || (w as usize) < lo {
                        continue;
                    }
                    let ev: &Event = match pl {
                        0 | 1 => side.ev.get(w as usize).ok_or_else(|| rerr(&b.label, format!("step {w}: {} side events", side.ev.len())))?,
                        _ => {
                            let rm = rt.remote.as_deref().ok_or_else(|| rerr(&b.label, "remote lane step without a remote device"))?;
                            let e = if pl == 2 { rm.lane_event_in(w as usize) } else { rm.lane_event(w as usize) };
                            // SAFETY: the remote lane's events live as long as the run (the borrow ends before the
                            // remote lane is used mutably below)
                            unsafe { &*(e.ok_or_else(|| rerr(&b.label, format!("step {w}: no remote lane event")))? as *const Event) }
                        }
                    };
                    match lane {
                        1 => side.ex.compute.wait(ev)?,
                        2 => rt.remote.as_deref_mut().ok_or_else(|| rerr(&b.label, "remote lane step without a remote device"))?.lane_wait(ev)?,
                        _ => rt.ex.compute.wait(ev)?,
                    }
                }
            }
        }
        region_start = dynamic;
        if !in_unit && step_prof() {
            // PC_STEP_PROFILE=1 (diagnostic): the compute lane drained at every change of op class, the time between
            // two drains booked to the class issued in between ([stepprof] line per eager run)
            let n = unit.map_or(1, |(_, e)| e - i);
            let (cls, cnt) = prof_class(&b.steps[i].op, n, |f| b.finishes.get(f as usize).map_or((0, String::new()), |fs| (fs.outs.len(), fs.desc.to_string())));
            let cls = match b.names.get(b.steps[i].var as usize).and_then(|nm| prof_step(&b.label, nm)) {
                Some(st) => st,
                None => cls,
            };
            let mut g = PROF.lock().unwrap();
            if g.0.as_ref().map(|x| x.0.as_str()) != Some(cls.as_str()) {
                rt.ex.compute.sync()?;
                let now = Instant::now();
                if let Some((c, t)) = g.0.take() {
                    g.1.entry(c).or_insert((0.0, 0)).0 += now.duration_since(t).as_secs_f64() * 1e3;
                }
                g.0 = Some((cls.clone(), now));
            }
            g.1.entry(cls).or_insert((0.0, 0)).1 += cnt as u64;
        }
        if !in_unit {
            if lane == 1 {
                swap_lane(rt)?;
            }
            // a dynamic step: every slot it reads, as it reads them (a finish reads several)
            let pre_metas: Option<Vec<(u32, CtMeta)>> =
                dynamic.then(|| schedule::rw(b, &b.steps[i]).0.into_iter().filter_map(|s| b.slots.get(s as usize).map(|c| (s, c.meta))).collect());
            if let (0, Some(rs)) = (lane, rsched) {
                if rs.own.get(i).is_some_and(|o| !o.is_empty()) {
                    rt.remote.as_deref_mut().ok_or_else(no_replica)?.own_begin(&rs.own[i]);
                }
            }
            {
                let Bound { label, slots, tmp, gather, held, steps, groups, relin, step_lists, units, unit_of, pts, finishes, coeff_sums, mult_sums, stream_base, names, .. } = &mut *b;
                let tmp = tmp.as_mut().ok_or_else(|| rerr(label, "no scratch slot"))?;
                if tl_active() {
                    TL.with(|t| {
                        if let Some(tl) = t.borrow_mut().as_mut() {
                            tl.cur = i as u32;
                        }
                    });
                }
                let r = match unit {
                    _ if lane == 2 => {
                        let s = sched.ok_or_else(|| rerr(label, "remote lane step without a schedule"))?;
                        // a unit's copy back waits for what any of its steps' copy back waits for
                        let e = unit.map_or(i + 1, |(_, e)| e);
                        let outw: Vec<(usize, u32)> = (i..e).flat_map(|j| s.wait_out[j].iter().copied().enumerate()).filter(|&(_, w)| w != NONE && (w as usize) >= lo).collect();
                        remote_refresh(label, rt, unit.map(|(u, _)| &units[u as usize]), steps, i, slots.as_mut_ptr(), slots.len(), &outw)
                    }
                    _ if lane == 4 => {
                        let rs = sched.and_then(|s| s.replica.as_ref()).ok_or_else(|| rerr(label, "replica step without a replica plan"))?;
                        let (sh, source) = (rt.sh, rt.source);
                        let rm = rt.remote.as_deref_mut().ok_or_else(|| rerr(label, "replica step without a remote device"))?;
                        // the plan's event of a unit is keyed by its first step and covers all its steps: a unit
                        // issued step by step (the checks) records it after its last step
                        let planned = unit_of.get(i).filter(|&&u| u != NONE && !rs.split_unit.get(u as usize).copied().unwrap_or(false)).map(|&u| (units[u as usize].lo as usize, units[u as usize].hi as usize));
                        let signal = match (unit, planned) {
                            (None, Some((ulo, uhi))) => (i + 1 == uhi && rs.signal[ulo]).then_some(ulo),
                            _ => rs.signal[i].then_some(i),
                        };
                        rm.replica_issue(sh, source, i, unit, signal)
                    }
                    _ => {
                        let t = Tables { groups, relin: *relin, lists: step_lists, pts, finishes, coeff_sums, mult_sums, stream_base: *stream_base };
                        exec_at(label, rt, &t, units, steps, slots, tmp, gather, held, i, unit, (&mut ks_owner, &mut held_owner), lane == 0, &mut info.fused)
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
                    // where it failed (lane 4: the replica), whatever the check
                    let st = &steps[i];
                    let m = |s: u32| if s != NONE { slots.get(s as usize).map(|c| (c.meta.level, c.meta.deg)) } else { None };
                    return Err(rerr(label, format!(
                        "{e} (step {i} -> {} {} on lane {lane}, unit {:?}, operands here at {:?} / {:?} (compiled {:?} / {:?}))",
                        names[st.var as usize],
                        format!("{:?}", st.op).chars().take(120).collect::<String>(),
                        unit,
                        m(st.a),
                        m(st.b),
                        st.a_meta.map(|x| (x.level, x.deg)),
                        st.b_meta.map(|x| (x.level, x.deg))
                    )));
                }
            }
            if let Some(ins) = pre_metas {
                let outs = schedule::rw(b, &b.steps[i]).1.into_iter().filter_map(|s| b.slots.get(s as usize).map(|c| (s, c.meta))).collect();
                b.dyn_metas.insert(i as u32, DynMetas { ins, outs });
            }
            if let Some((_, e)) = unit {
                unit_end = e;
            }
            if tl_active() {
                tl_mark(rt, i, if lane == 4 { 4 } else { 0 }, false)?;
            }
        }
        if let (Some(rs), 0) = (rsched, lane) {
            // the plan's event of a unit is keyed by its first step and covers all its steps: recorded once the unit
            // issued (at its first step), or, issued step by step (its slots did not fit), after its last step
            let planned = b.unit_of.get(i).filter(|&&u| u != NONE && !rs.split_unit.get(u as usize).copied().unwrap_or(false)).map(|&u| (b.units[u as usize].lo as usize, b.units[u as usize].hi as usize));
            let key = match (unit, planned) {
                (Some(_), _) if !in_unit => Some(i),
                (None, Some((ulo, uhi))) if !in_unit && i + 1 == uhi => Some(ulo),
                (None, None) if !in_unit => Some(i),
                _ => None,
            };
            // PC_OWN_LAZY: the plan has the owned outputs on the remote device only and its readers (there, or this
            // device's copies of them back) wait for this step's event: those the remote services did not leave there
            // are written over now, from here on this device's compute stream, before the event
            let lazy_own = crate::replica::own_lazy();
            if let Some(k) = key.filter(|&k| lazy_own && rs.own.get(k).is_some_and(|o| !o.is_empty())) {
                let rm = rt.remote.as_deref_mut().ok_or_else(no_replica)?;
                let done = rm.own_take();
                for &sl in &rs.own[k] {
                    let src: *const Ct<'a, W> = b.slots.get(sl as usize).ok_or_else(|| rerr(&b.label, format!("slot {sl}")))?;
                    if done.contains(&sl) {
                        OWN_KEPT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    } else {
                        let dst = rm.replica_slot_ptr(sl).ok_or_else(|| rerr(&b.label, format!("replica slot {sl}")))?;
                        // SAFETY: this device's slot (valid: its service copied it back) into the replica's, whose last
                        // reader there the step waited for (the plan's write-after-read wait of an owned write)
                        copy_ct(rt.ctx, &rt.ex.compute, unsafe { &mut *dst }, unsafe { &*src })?;
                        OWN_COPIED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    // SAFETY: as above
                    rm.replica_set_meta(sl, unsafe { (*src).meta })?;
                }
            }
            if let Some(k) = key.filter(|&k| rs.signal[k]) {
                let side = rt.side.as_deref_mut().ok_or_else(|| rerr(&b.label, "replica plan without the side lane's events"))?;
                let ev = side.ev.get(k).ok_or_else(|| rerr(&b.label, format!("step {k}: {} side events", side.ev.len())))?;
                rt.ex.compute.record(ev)?;
            }
            if let Some(k) = key.filter(|&k| !lazy_own && rs.own.get(k).is_some_and(|o| !o.is_empty())) {
                // ownership: the owned outputs the remote services did not leave there copied over (after this step's
                // event), every one's replica metadata this device's
                let ev: *const Event = rt.side.as_deref().and_then(|sd| sd.ev.get(k)).ok_or_else(|| rerr(&b.label, format!("step {k}: no side event")))?;
                let rm = rt.remote.as_deref_mut().ok_or_else(no_replica)?;
                let done = rm.own_take();
                let mut waited = false;
                for &sl in &rs.own[k] {
                    let src: *const Ct<'a, W> = b.slots.get(sl as usize).ok_or_else(|| rerr(&b.label, format!("slot {sl}")))?;
                    if done.contains(&sl) {
                        OWN_KEPT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    } else {
                        if !waited {
                            if !rrforked {
                                // a region without replica steps: forked here, joined at its end like any
                                rm.replica_fork(&rt.ex.compute)?;
                                rrforked = true;
                            }
                            // SAFETY: the side lane's events live as long as the run
                            rm.replica_wait(unsafe { &*ev })?;
                            waited = true;
                        }
                        // SAFETY: this device's slot, read by the copy; nothing writes it during the call
                        rm.replica_copy_in(sl, unsafe { &*src })?;
                        OWN_COPIED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    // SAFETY: as above
                    rm.replica_set_meta(sl, unsafe { (*src).meta })?;
                }
            }
        }
        if let Some(s) = sched {
            if s.signal[i] && lane != 2 {
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
        if check && lane != 4 {
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
    if rforked {
        rt.remote.as_deref_mut().ok_or_else(|| rerr(&b.label, "remote lane step without a remote device"))?.lane_join(&rt.ex.compute)?;
    }
    if let Some(rs) = rsched {
        if hi < b.steps.len() && b.dynamic.contains(&(hi as u32)) {
            // a segment ending at a dynamic step: that step's copies back belong to this segment
            replica_copies(rt, b, rs, hi, lo)?;
        }
        if rrforked {
            rt.remote.as_deref_mut().ok_or_else(no_replica)?.replica_join(&rt.ex.compute)?;
        }
        if dforked {
            rt.remote.as_deref_mut().ok_or_else(no_replica)?.dma_join(&rt.ex.compute)?;
        }
        if hi == b.steps.len() {
            // the block's outputs whose version is on the replica only (joined above: no waits)
            for &(slot, _) in &rs.out_copies {
                replica_copy_back(rt, b, slot)?;
            }
        }
    }
    Ok(())
}

/// The replica plan's waits and copies before step `i` (in a range starting at `lo`): the other device's events it
/// waits for (write after read), then its copies, each after its producer's event (events of steps before `lo` are
/// ordered already: an earlier segment, joined).
fn replica_copies<'a, W: DevWord, M: DevStrategy<W>>(rt: &mut Rt<'_, 'a, W, M>, b: &mut Bound<'a, W>, rs: &crate::replica::ReplicaSched, i: usize, lo: usize) -> Result<()> {
    let on_rep = rs.on_replica(i);
    let n = b.steps.len() as u32;
    // steps before `i` and copies issued at or before it (PC_REPLICA_DMA), from this range on
    let evs: Vec<u32> = rs.wait[i].iter().copied().chain(rs.copies[i].iter().map(|&(_, p)| p)).filter(|&p| p != NONE && rs.pos(p) >= lo && if p < n { (p as usize) < i } else { rs.pos(p) <= i }).collect();
    for p in evs {
        if p >= n {
            // an early copy's event (on its receiver)
            let c = (p - n) as usize;
            let to = rs.xc.get(c).map(|x| x.to).ok_or_else(|| rerr(&b.label, format!("copy {c}")))?;
            let rm = rt.remote.as_deref_mut().ok_or_else(no_replica)?;
            let ev: *const Event = rm.dma_event(c, to).ok_or_else(|| rerr(&b.label, format!("copy {c}: no event")))?;
            // SAFETY: the copy events live as long as the run
            if on_rep {
                rm.replica_wait(unsafe { &*ev })?;
            } else {
                rt.ex.compute.wait(unsafe { &*ev })?;
            }
            continue;
        }
        if on_rep {
            // a step of this device (its event on compute, the side lane's event array)
            let side = rt.side.as_deref().ok_or_else(|| rerr(&b.label, "replica plan without the side lane's events"))?;
            let ev: *const Event = side.ev.get(p as usize).ok_or_else(|| rerr(&b.label, format!("step {p}: {} side events", side.ev.len())))?;
            // SAFETY: the side lane's events live as long as the run (the borrow ends before the remote is used)
            rt.remote.as_deref_mut().ok_or_else(no_replica)?.replica_wait(unsafe { &*ev })?;
        } else {
            let rm = rt.remote.as_deref().ok_or_else(no_replica)?;
            let ev = rm.replica_event(p as usize).ok_or_else(|| rerr(&b.label, format!("replica step {p}: no event")))?;
            rt.ex.compute.wait(ev)?;
        }
    }
    for &(slot, _) in &rs.copies[i] {
        if on_rep {
            let src: *const Ct<'a, W> = b.slots.get(slot as usize).ok_or_else(|| rerr(&b.label, format!("slot {slot}")))?;
            // SAFETY: this device's slot, read by the copy; nothing writes it during the call
            rt.remote.as_deref_mut().ok_or_else(no_replica)?.replica_copy_in(slot, unsafe { &*src })?;
        } else {
            replica_copy_back(rt, b, slot)?;
        }
    }
    Ok(())
}

/// PC_REPLICA_DMA: copy `c` of the plan on its receiver's copy stream, after its waits issued in this range (those
/// before it are ordered already: an earlier segment, joined at its end).
fn issue_early<'a, W: DevWord, M: DevStrategy<W>>(rt: &mut Rt<'_, 'a, W, M>, b: &mut Bound<'a, W>, rs: &crate::replica::ReplicaSched, c: usize, lo: usize) -> Result<()> {
    let x = rs.xc.get(c).ok_or_else(|| rerr(&b.label, format!("copy {c}")))?;
    let n = b.steps.len() as u32;
    for &q in &x.waits {
        if rs.pos(q) < lo {
            continue;
        }
        let ev: *const Event = if q >= n {
            let qc = (q - n) as usize;
            let to = rs.xc.get(qc).map(|y| y.to).ok_or_else(|| rerr(&b.label, format!("copy {qc}")))?;
            rt.remote.as_deref().and_then(|rm| rm.dma_event(qc, to)).ok_or_else(|| rerr(&b.label, format!("copy {qc}: no event")))?
        } else if rs.on_replica(q as usize) {
            rt.remote.as_deref().and_then(|rm| rm.replica_event(q as usize)).ok_or_else(|| rerr(&b.label, format!("replica step {q}: no event")))?
        } else {
            rt.side.as_deref().and_then(|sd| sd.ev.get(q as usize)).ok_or_else(|| rerr(&b.label, format!("step {q}: no side event")))?
        };
        // SAFETY: the events live as long as the run
        rt.remote.as_deref_mut().ok_or_else(no_replica)?.dma_wait(x.to, unsafe { &*ev })?;
    }
    let label = b.label.clone();
    let slot = b.slots.get_mut(x.slot as usize).ok_or_else(|| rerr(&label, format!("slot {}", x.slot)))?;
    rt.remote.as_deref_mut().ok_or_else(no_replica)?.dma_copy(c, x.to, x.slot, slot)
}

/// The replica's slot `slot` copied back into this device's, on compute (ordered by the caller).
fn replica_copy_back<'a, W: DevWord, M: DevStrategy<W>>(rt: &mut Rt<'_, 'a, W, M>, b: &mut Bound<'a, W>, slot: u32) -> Result<()> {
    let rm = rt.remote.as_deref().ok_or_else(no_replica)?;
    let src: *const Ct<'a, W> = rm.replica_slot(slot).ok_or_else(|| rerr(&b.label, format!("replica slot {slot}")))?;
    let label = b.label.clone();
    let dst = b.slots.get_mut(slot as usize).ok_or_else(|| rerr(&label, format!("slot {slot}")))?;
    // SAFETY: the replica's slot outlives the call; its producer was ordered before this copy
    copy_ct(rt.ctx, &rt.ex.compute, dst, unsafe { &*src })?;
    if replica_sync() {
        rt.ex.compute.sync()?;
    }
    REPLICA_COPIES_BACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// A refresh on the remote lane (schedule lane 2): single step `i`, or the refresh pair `unit` (both its steps).
/// PC_REMOTE_CHECK: the same refresh again here on the stashed inputs, every output word and metadata compared.
fn remote_refresh<'a, W: DevWord, M: DevStrategy<W>>(block: &str, rt: &mut Rt<'_, 'a, W, M>, unit: Option<&Unit>, steps: &[Step], i: usize, slots: *mut Ct<'a, W>, n_slots: usize, outw: &[(usize, u32)]) -> Result<()> {
    let at = |s: u32| -> Result<*mut Ct<'a, W>> {
        if (s as usize) < n_slots {
            // SAFETY: in bounds (checked)
            Ok(unsafe { slots.add(s as usize) })
        } else {
            Err(rerr(block, format!("slot {s} of {n_slots}")))
        }
    };
    // (inputs (None: the output holds it), outputs, routes, route family)
    let (ins, outs, routes, kind): (Vec<Option<*mut Ct<'a, W>>>, Vec<*mut Ct<'a, W>>, Vec<pc_shadow::Route>, RouteKind) = match unit {
        Some(u) => {
            let UnitKind::Pair { routes, kind } = &u.kind else { return Err(rerr(block, format!("step {i}: a remote unit other than a refresh pair"))) };
            let lo = u.lo as usize;
            (vec![Some(at(u.ins[0])?), Some(at(u.ins[1])?)], vec![at(steps[lo].out)?, at(steps[lo + 1].out)?], routes.to_vec(), *kind)
        }
        None => {
            let st = &steps[i];
            let StepOp::Bootstrap { route, kind, .. } = &st.op else { return Err(rerr(block, format!("step {i}: a remote step other than a refresh"))) };
            // an input in the output's own slot is read through the output (one reference to the slot)
            (vec![if st.inplace || st.a == st.out { None } else { Some(at(st.a)?) }], vec![at(st.out)?], vec![*route], *kind)
        }
    };
    if outs.len() == 2 && (outs[0] == outs[1] || ins.iter().flatten().any(|x| outs.contains(x))) {
        return Err(rerr(block, format!("step {i}: a remote pair writing an input or one slot twice")));
    }
    let check = remote_check();
    // PC_REMOTE_CHECK (eager): every lane drained, the inputs stashed on the host
    let stash: Vec<(Vec<Vec<W>>, CtMeta)> = if check {
        rt.ex.compute.sync()?;
        if let Some(sd) = rt.side.as_deref() {
            sd.ex.compute.sync()?;
        }
        let mut v = vec![];
        for (c, x) in ins.iter().enumerate() {
            // SAFETY: slots of this block, only read here
            let x = unsafe { &*x.unwrap_or(outs[c]) };
            let top = pc_shadow::meta::top_limb(rt.ctx.shape, x.meta.level).map_err(|e| rerr(block, e.to_string()))? as usize;
            let rows: Vec<usize> = (0..=top).collect();
            v.push(((0..2).map(|p| x.buf.download_rows(&rt.ex.compute, 0, p, &rows)).collect::<std::result::Result<_, _>>()?, x.meta));
        }
        v
    } else {
        vec![]
    };
    let ids: Vec<usize> = (i..i + outs.len()).collect();
    {
        // the remote part: inputs copied over (an input in the output's slot read through it) and refreshed there
        // SAFETY: the block's slots, only read here
        let srcs: Vec<&Ct<'a, W>> = ins.iter().zip(&outs).map(|(x, &o)| unsafe { &*x.unwrap_or(o) }).collect();
        rt.remote.as_deref_mut().ok_or_else(|| rerr(block, "remote lane step without a remote device"))?.lane_issue(&ids, rt.sh, kind, &routes, &srcs)?;
    }
    {
        // the copy back: after every earlier reader and writer of the output slots (the copy-back stream is in order
        // with the earlier copy backs)
        for &(pl, w) in outw {
            let ev: *const Event = match pl {
                0 | 1 => rt.side.as_deref().and_then(|sd| sd.ev.get(w as usize)).ok_or_else(|| rerr(block, format!("step {w}: no side event")))?,
                2 => rt.remote.as_deref().and_then(|rm| rm.lane_event_in(w as usize)).ok_or_else(|| rerr(block, format!("step {w}: no input-copy event")))?,
                _ => continue,
            };
            // SAFETY: the events live as long as the run; the shared borrow above ended
            rt.remote.as_deref_mut().ok_or_else(|| rerr(block, "remote lane step without a remote device"))?.lane_out_wait(unsafe { &*ev })?;
        }
        // SAFETY: distinct output slots (checked for a pair; a single has one), the only references
        let mut outs_r: Vec<&mut Ct<'a, W>> = outs.iter().map(|&p| unsafe { &mut *p }).collect();
        rt.remote.as_deref_mut().ok_or_else(|| rerr(block, "remote lane step without a remote device"))?.lane_finish(&ids, &mut outs_r)?;
    }
    if outs.len() == 2 {
        REMOTE_ASYNC_PAIRS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    } else {
        REMOTE_ASYNC_SINGLES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    if check {
        rt.remote.as_deref_mut().ok_or_else(|| rerr(block, "remote lane step without a remote device"))?.lane_sync()?;
        // the remote outputs, then the same refresh here on the stashed inputs (in the output slots, as the local path
        // runs it), word for word
        let mut far = vec![];
        for &o in &outs {
            // SAFETY: the block's slots, only read here
            let o = unsafe { &*o };
            let top = pc_shadow::meta::top_limb(rt.ctx.shape, o.meta.level).map_err(|e| rerr(block, e.to_string()))? as usize;
            let rows: Vec<usize> = (0..=top).collect();
            let w: Vec<Vec<W>> = (0..2).map(|p| o.buf.download_rows(&rt.ex.compute, 0, p, &rows)).collect::<std::result::Result<_, _>>()?;
            far.push((w, o.meta));
        }
        for (c, &o) in outs.iter().enumerate() {
            // SAFETY: distinct output slots, the only references
            let o = unsafe { &mut *o };
            let (rows_in, m) = &stash[c];
            let top = pc_shadow::meta::top_limb(rt.ctx.shape, m.level).map_err(|e| rerr(block, e.to_string()))? as usize;
            let rows: Vec<usize> = (0..=top).collect();
            for (p, w) in rows_in.iter().enumerate() {
                o.buf.upload_rows(&rt.ex.compute, 0, p, &rows, w)?;
            }
            o.meta = *m;
        }
        let r = rt.routes.get(kind).ok_or_else(|| rerr(block, format!("route {kind:?} not held")))?;
        let mut cx = BtsCx { ctx: rt.ctx, ex: &mut *rt.ex, sh: rt.sh, ks: rt.ks, ops: &mut *rt.ops, work: &mut *rt.work };
        let o0 = outs[0];
        if outs.len() == 2 {
            let o1 = outs[1];
            // SAFETY: distinct output slots
            let (x, y) = unsafe { (&mut *o0, &mut *o1) };
            r.run_pair(&mut cx, x, y, &routes[0], &routes[1])?;
        } else {
            // SAFETY: the output slot, the only reference
            r.run(&mut cx, std::slice::from_mut(unsafe { &mut *o0 }), &routes[0])?;
        }
        rt.ex.compute.sync()?;
        for (c, &o) in outs.iter().enumerate() {
            // SAFETY: the block's slots, only read here
            let o = unsafe { &*o };
            let (w, m) = &far[c];
            let top = pc_shadow::meta::top_limb(rt.ctx.shape, o.meta.level).map_err(|e| rerr(block, e.to_string()))? as usize;
            let rows: Vec<usize> = (0..=top).collect();
            let mut same = o.meta == *m;
            for (p, wp) in w.iter().enumerate() {
                same &= same && o.buf.download_rows(&rt.ex.compute, 0, p, &rows)? == *wp;
            }
            if !same {
                return Err(rerr(block, format!("remote check: the remote lane's refresh at step {i} (output {c} of {}) differs from the same refresh here", outs.len())));
            }
        }
        REMOTE_ASYNC_CHECKED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
/// rotations only need one level; a refresh pair's members share level, degree and the StC-first stage cache's scale
/// key (`nf_key`: the pair's last stage is built for the first member's scale, as a single refresh takes its cached one).
fn unit_fits<'a, W: DevWord, M: DevStrategy<W>>(u: &Unit, slots: &[Ct<'a, W>], steps: &[Step], pts: &[BoundPt], sh: &Shadow, routes: &Routes<'_, 'a, W, M>) -> bool {
    let meta = |s: &u32| slots.get(*s as usize).map(|c| c.meta);
    let one = |v: &[u32], scale: bool| {
        let m0 = v.first().and_then(meta);
        v.iter().all(|s| meta(s).zip(m0).is_some_and(|(m, f)| m.level == f.level && (!scale || m.deg == f.deg && m.nf.to_bits() == f.nf.to_bits())))
    };
    match &u.kind {
        UnitKind::Grid { .. } | UnitKind::RotateSum { .. } => one(&u.ins, false),
        // the adds take no prelude: every operand at one level, degree and scale
        UnitKind::RotAcc { .. } => one(&u.ins, true),
        UnitKind::Mult { b, .. } => one(&u.ins, true) && one(b, true),
        UnitKind::Refresh { .. } => one(&u.ins, true),
        UnitKind::Pair { routes: rs, kind } => match (meta(&u.ins[0]), meta(&u.ins[1])) {
            (Some(x), Some(y)) if x.level == y.level && x.deg == y.deg && rs[0].raise_drop == rs[1].raise_drop => nf_key(x.nf) == nf_key(y.nf),
            // a loose pair (bts.pair_loose): two levels or raise drops, as the route takes them
            (Some(x), Some(y)) => routes.get(*kind).is_some_and(|r| r.pair_ok(sh, &x, &y, &rs[0], &rs[1])),
            _ => false,
        },
        // one pass when neither the add nor the plaintext needs a prelude; a sum landing on the add's other operand
        // also takes none on the product, and one landing on the product's input is not fused
        UnitKind::MultPtAdd { pt, sub, product_first } => {
            let (Some(a), Some(y), Some(p)) = (meta(&u.ins[0]), meta(&u.ins[1]), pts.get(*pt as usize).and_then(|p| p.meta)) else { return false };
            let out = steps[u.hi as usize - 1].out;
            if out == u.ins[1] && u.ins[0] != u.ins[1] {
                Ops::<W>::mult_pt_add_inplace_fusable(sh, &a, &p, &y, *sub, *product_first).unwrap_or(false)
            } else {
                out != u.ins[0] && Ops::<W>::mult_pt_add_fusable(sh, &a, &p, &y, *sub, *product_first).unwrap_or(false)
            }
        }
        // one pass when no product or add takes a prelude
        UnitKind::MultPtAddChain { head, pts: ids, pf } => {
            let pm = |i: u32| pts.get(i as usize).and_then(|p| p.meta);
            let Some(x0) = meta(&u.ins[0]) else { return false };
            let mut terms = Vec::with_capacity(ids.len());
            for (c, (&i, &f)) in ids.iter().zip(pf).enumerate() {
                let (Some(a), Some(p)) = (u.ins.get(1 + c).and_then(meta), pm(i)) else { return false };
                terms.push((a, p, f));
            }
            let tm: Vec<(&CtMeta, &PtMeta, bool)> = terms.iter().map(|(a, p, f)| (a, p, *f)).collect();
            let r = match head {
                Some(h) => pm(*h).map(|p| Ops::<W>::mult_pt_add_chain_meta(sh, Some((&x0, &p)), None, &tm)),
                None => Some(Ops::<W>::mult_pt_add_chain_meta(sh, None, Some(&x0), &tm)),
            };
            matches!(r, Some(Ok(Some(_))))
        }
        // one pass when no step takes a prelude but limb drops
        UnitKind::Affine { steps: st } => {
            let Some(ms) = u.ins.iter().map(meta).collect::<Option<Vec<CtMeta>>>() else { return false };
            matches!(Ops::<W>::affine_meta(sh, &ms, st), Ok(Some(_)))
        }
        UnitKind::MultIAdd { sub, product_first } => match (meta(&u.ins[0]), meta(&u.ins[1])) {
            (Some(a), Some(y)) => Ops::<W>::mult_i_add_fusable(sh, &a, &y, *sub, *product_first).unwrap_or(false),
            _ => false,
        },
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
    pts: &[BoundPt],
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
        UnitKind::RotAcc { gs, gal, steps: st } => {
            let o = span[span.len() - 1].out;
            let keys: Vec<&KeyGroup<'a>> = gs.iter().map(|&g| group(block, groups, g)).collect::<Result<_>>()?;
            // the sum may land on acc_0's slot (then read there, `xs[0]` unused) or on a rotated input's (each is read
            // before the one write at the end)
            let acc = if u.ins[0] == o { None } else { Some(xs[0]) };
            // SAFETY: as the other units' outputs: the only mutable reference into the slots during the unit
            let out = unsafe { &mut *slots.add(o as usize) };
            rotate::rotate_acc(rt.ctx, rt.ex, kc, rt.sh, &keys, acc, &xs[1..], gal, st, out)?;
        }
        UnitKind::RotateSum { l, g } => {
            let st = lists.get(*l as usize).ok_or_else(|| rerr(block, format!("step list {l} out of range")))?;
            let mut outs: Vec<&mut Ct<'a, W>> = span.iter().map(|x| unsafe { &mut *slots.add(x.out as usize) }).collect();
            rotate::rotate_sum_batch(rt.ctx, rt.ex, kc, rt.sh, group(block, groups, *g)?, &xs, &mut gather.a, st, &mut outs)?;
        }
        UnitKind::Mult { b, rs } => {
            // the operands gathered into K-ciphertext buffers, one product of K ciphertexts (one relinearization key
            // read for all of them), the products scattered with their own metadata
            if b.iter().any(|&s| s as usize >= n_slots) {
                return Err(rerr(block, format!("product unit at step {}: slots outside the {n_slots} bound slots", u.lo)));
            }
            let ys: Vec<&Ct<'a, W>> = b.iter().map(|&s| unsafe { &*slots.add(s as usize) }).collect();
            let (k, s, lay) = (xs.len(), &rt.ex.compute, rt.ctx.lay);
            // operands / outputs in consecutive slots are read / written where they lie (no gather copies)
            // SAFETY: the operand views are only read; the output view is the only writer of the outputs (distinct
            // slots, none an operand's) until it is dropped, before their metadata is set
            let va = unsafe { pc_gpu::ct::run_view(lay, &xs) };
            let vb = unsafe { pc_gpu::ct::run_view(lay, &ys) };
            let mut vo = {
                let os: Vec<&Ct<'a, W>> = span.iter().map(|st| unsafe { &*slots.add(st.out as usize) }).collect();
                unsafe { pc_gpu::ct::run_view(lay, &os) }
            };
            if va.is_none() {
                for (c, x) in xs.iter().enumerate() {
                    batch::put(rt.ctx, s, &mut gather.a, c, x)?;
                }
            }
            if vb.is_none() {
                for (c, y) in ys.iter().enumerate() {
                    batch::put(rt.ctx, s, &mut gather.b, c, y)?;
                }
            }
            gather.a.buf.k = k;
            gather.a.meta = xs[0].meta;
            gather.out.buf.k = k;
            let key = group(block, groups, relin)?;
            let Gather { a: ga, b: gb, out: go } = gather;
            let a: &Ct<'a, W> = match &va {
                Some(v) => v,
                None => &*ga,
            };
            let out: &mut Ct<'a, W> = match vo.as_mut() {
                Some(v) => v,
                None => &mut *go,
            };
            if !ys.is_empty() {
                gb.buf.k = k;
                gb.meta = ys[0].meta;
            }
            let b: Option<&Ct<'a, W>> = if ys.is_empty() {
                None
            } else {
                Some(match &vb {
                    Some(v) => v,
                    None => &*gb,
                })
            };
            match (rs, b) {
                (Some(dbl), b) => rt.ops.mult_rescale(rt.ctx, rt.ex, kc, rt.sh, key, out, Some(a), b, *dbl)?,
                (None, None) => rt.ops.square(rt.ctx, rt.ex, kc, rt.sh, key, out, a)?,
                (None, Some(b)) => rt.ops.mult(rt.ctx, rt.ex, kc, rt.sh, key, out, a, b)?,
            }
            let direct = vo.is_some();
            drop(vo);
            for (c, st) in span.iter().enumerate() {
                let r = if ys.is_empty() { rt.sh.square(&xs[c].meta) } else { rt.sh.mult(&xs[c].meta, &ys[c].meta) };
                let mut m = r.map_err(|e| rerr(block, e.to_string()))?.out;
                if let Some(dbl) = rs {
                    if *dbl {
                        m = rt.sh.add(&m, &m).map_err(|e| rerr(block, e.to_string()))?.out;
                    }
                    m = rt.sh.realize(&m).map_err(|e| rerr(block, e.to_string()))?.out;
                }
                // SAFETY: as the unit's other outputs (distinct slots, none an operand's)
                let o = unsafe { &mut *slots.add(st.out as usize) };
                if direct {
                    o.meta = m;
                } else {
                    batch::take(rt.ctx, &rt.ex.compute, o, go, c, m)?;
                }
            }
        }
        UnitKind::MultPtAdd { pt: i, sub, product_first } => {
            let p = pt(block, pts, *i)?;
            if span[1].out == u.ins[1] {
                // SAFETY: the sum lands on the add's other operand, the only mutable reference (`unit_fits`: not on the
                // product's input, read through `xs[0]`)
                let acc = unsafe { &mut *slots.add(u.ins[1] as usize) };
                rt.ops.mult_pt_add_inplace(rt.ctx, rt.ex, rt.sh, acc, xs[0], &p, *sub, *product_first)?;
            } else {
                // SAFETY: `unit_fits` checked the sum's slot is neither operand's: the only mutable reference
                let out = unsafe { &mut *slots.add(span[1].out as usize) };
                rt.ops.mult_pt_add(rt.ctx, rt.ex, rt.sh, out, xs[0], &p, xs[1], *sub, *product_first)?;
            }
        }
        UnitKind::MultPtAddChain { head, pts: ids, pf } => {
            let ps: Vec<Pt<'_>> = ids.iter().map(|&i| pt(block, pts, i)).collect::<Result<_>>()?;
            let hp = head.map(|h| pt(block, pts, h)).transpose()?;
            let out_slot = span[span.len() - 1].out;
            // y (no head) may be the sum's own slot: its rows and metadata are taken before the output is borrowed
            let y = if hp.is_none() { Some((xs[0].buf.r(), xs[0].meta)) } else { None };
            if u.ins[1..].contains(&out_slot) || hp.is_some() && u.ins[0] == out_slot {
                return Err(rerr(block, format!("product chain at step {}: the sum lands on a term's or the head's operand", u.lo)));
            }
            let terms: Vec<(&Ct<'a, W>, &Pt<'_>, bool)> = xs[1..].iter().zip(&ps).zip(pf).map(|((a, p), &f)| (*a, p, f)).collect();
            let head = hp.as_ref().map(|p| (xs[0], p));
            // SAFETY: the sum's slot is no term's or head's operand (checked above): the only mutable reference; y may
            // share it, taken above as rows and metadata (`xs[0]` unused past here in that case)
            let out = unsafe { &mut *slots.add(out_slot as usize) };
            rt.ops.mult_pt_add_chain(rt.ctx, rt.ex, rt.sh, out, head, y, &terms)?;
        }
        UnitKind::Affine { steps: st } => {
            // the sources' rows and metadata are taken before the output (which may share a source's slot) is borrowed
            let srcs: Vec<(pc_kernels::PolyRef, CtMeta, usize)> = xs.iter().map(|x| (x.buf.r(), x.meta, x.k())).collect();
            // SAFETY: the group's last output slot, the only mutable reference; the sources are read through `srcs`
            // (`xs` unused past here)
            let out = unsafe { &mut *slots.add(span[span.len() - 1].out as usize) };
            rt.ops.affine(rt.ctx, rt.ex, rt.sh, out, &srcs, st)?;
        }
        UnitKind::MultIAdd { sub, product_first } => {
            let (a, y) = ((xs[0].buf.r(), xs[0].meta, xs[0].k()), (xs[1].buf.r(), xs[1].meta, xs[1].k()));
            // SAFETY: the add's output slot, the only mutable reference; a and y are read through their rows taken above
            let out = unsafe { &mut *slots.add(span[1].out as usize) };
            rt.ops.mult_i_add(rt.ctx, rt.ex, rt.sh, out, a, y, *sub, *product_first)?;
        }
        UnitKind::Refresh { route, kind } if xs.len() >= 2 && rt.remote.as_ref().is_some_and(|rm| rm.has(*kind)) => {
            // split across two devices: the first ceil(K/2) members gathered and refreshed here, the rest on the
            // remote device at the same time (forked from compute, joined back before the members are read)
            let r = rt.routes.get(*kind).ok_or_else(|| rerr(block, format!("route {kind:?} not held")))?;
            let k = xs.len();
            let h = k - k / 2;
            for (c, x) in xs[..h].iter().enumerate() {
                batch::put(rt.ctx, &rt.ex.compute, &mut gather.a, c, x)?;
            }
            gather.a.buf.k = h;
            gather.a.meta = xs[0].meta;
            {
                // SAFETY: the remote members' output slots, distinct from every input's and from the local members'
                let mut far: Vec<&mut Ct<'a, W>> = span[h..].iter().map(|st| unsafe { &mut *slots.add(st.out as usize) }).collect();
                let rm = rt.remote.as_deref_mut().ok_or_else(|| rerr(block, "no remote lane"))?;
                if !rm.own_wanted().is_empty() {
                    // ownership: the remote members' slots, in their order there
                    rm.own_next(span[h..].iter().map(|st| st.out).collect(), 0);
                }
                rm.issue(&rt.ex.compute, rt.sh, *kind, route, &xs[h..], &mut far)?;
            }
            let mut cx = BtsCx { ctx: rt.ctx, ex: &mut *rt.ex, sh: rt.sh, ks: rt.ks, ops: &mut *rt.ops, work: &mut *rt.work };
            r.run(&mut cx, std::slice::from_mut(&mut gather.a), route)?;
            let out = gather.a.meta;
            for (c, st) in span[..h].iter().enumerate() {
                let m = r.landing(rt.sh, &xs[c].meta, route).ok().filter(|m| m.level == out.level && m.deg == out.deg).map_or(out, |m| CtMeta { nf: out.nf, ..m });
                batch::take(rt.ctx, &rt.ex.compute, unsafe { &mut *slots.add(st.out as usize) }, &gather.a, c, m)?;
            }
            rt.remote.as_deref_mut().ok_or_else(|| rerr(block, "no remote lane"))?.join(&rt.ex.compute)?;
            if remote_check() {
                // the whole batch here, as without the split, against every member's output
                for (c, x) in xs.iter().enumerate() {
                    batch::put(rt.ctx, &rt.ex.compute, &mut gather.a, c, x)?;
                }
                gather.a.buf.k = k;
                gather.a.meta = xs[0].meta;
                let mut cx = BtsCx { ctx: rt.ctx, ex: &mut *rt.ex, sh: rt.sh, ks: rt.ks, ops: &mut *rt.ops, work: &mut *rt.work };
                r.run(&mut cx, std::slice::from_mut(&mut gather.a), route)?;
                let out = gather.a.meta;
                for (c, st) in span.iter().enumerate() {
                    let m = r.landing(rt.sh, &xs[c].meta, route).ok().filter(|m| m.level == out.level && m.deg == out.deg).map_or(out, |m| CtMeta { nf: out.nf, ..m });
                    let o = unsafe { &*slots.add(st.out as usize) };
                    let top = pc_shadow::meta::top_limb(rt.ctx.shape, m.level).map_err(|x| rerr(block, x.to_string()))? as usize;
                    let rows: Vec<usize> = (0..=top).collect();
                    let mut same = o.meta == m;
                    for part in 0..2 {
                        let got = o.buf.download_rows(&rt.ex.compute, 0, part, &rows)?;
                        let want = gather.a.buf.download_rows(&rt.ex.compute, c, part, &rows)?;
                        same &= got == want;
                    }
                    if !same {
                        return Err(rerr(block, format!("remote check: refresh unit at step {} member {c} of {k} ({}) differs from the whole local batch", u.lo, if c < h { "local" } else { "remote" })));
                    }
                }
                REMOTE_CHECKED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        UnitKind::Refresh { route, kind } => {
            BTS_BATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            BTS_BATCH_CTS.fetch_add(xs.len() as u64, std::sync::atomic::Ordering::Relaxed);
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
        UnitKind::Pair { routes, kind } => {
            BTS_PAIRS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // each input copied into its output slot, the two refreshed there as one complex refresh, each landing where
            // its single refresh would
            let r = rt.routes.get(*kind).ok_or_else(|| rerr(block, format!("route {kind:?} not held")))?;
            // SAFETY: as the unit's other outputs (two distinct slots, neither an input's)
            let (x, y) = unsafe { (&mut *slots.add(span[0].out as usize), &mut *slots.add(span[1].out as usize)) };
            copy_ct(rt.ctx, &rt.ex.compute, x, xs[0])?;
            copy_ct(rt.ctx, &rt.ex.compute, y, xs[1])?;
            let mut cx = BtsCx { ctx: rt.ctx, ex: &mut *rt.ex, sh: rt.sh, ks: rt.ks, ops: &mut *rt.ops, work: &mut *rt.work };
            r.run_pair(&mut cx, x, y, &routes[0], &routes[1])?;
        }
    }
    Ok(())
}

/// A dynamic step between two graph launches: the launches did not move the host-side metadata of the slots they
/// wrote, so every operand slot gets the metadata it had in the eager run the graphs were captured after (a finish
/// reads all its inputs' levels and scales).
fn issue_dynamic<'a, W: DevWord, M: DevStrategy<W>>(rt: &mut Rt<'_, 'a, W, M>, b: &mut Bound<'a, W>, i: usize, sched: Option<&Sched>) -> Result<()> {
    let ins = b.dyn_metas.get(&(i as u32)).map(|d| d.ins.clone()).ok_or_else(|| rerr(&b.label, format!("dynamic step {i} never ran eagerly")))?;
    for (s, m) in ins {
        if let Some(c) = b.slots.get_mut(s as usize) {
            c.meta = m;
        }
    }
    let mut scratch = RunInfo { served: Served::Eager, host_us: 0.0, launch_us: 0.0, meta_mismatches: 0, first_mismatches: vec![], pairs: [0; 2], fused: 0 };
    // with the replica's plan: its ownership hooks (a finish leaving outputs in the replica's slots) run here too
    issue_range(rt, b, i, i + 1, false, &mut scratch, None, sched.filter(|s| s.replica.is_some()))
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
        Replayer { cfg, stats: ReplayStats::default(), side_lane: true, scheds: Default::default(), eager_only: Default::default(), eager_marked: Default::default(), replica_last: Default::default(), seg_ev: vec![], seg_kinds: vec![], seg_host_us: vec![] }
    }

    /// Run `b` once on the inputs the driver wrote into its input slots: eager, or (graphs) the graph captured for
    /// the block's current plaintext rows (those of the last `Bound::resolve_pts`), or, when none matches, eager +
    /// capture (at most `ReplayCfg::graphs_per_block` layouts per block; the oldest is dropped beyond). Asynchronous:
    /// the outputs are ready on `compute` (the caller syncs or records an event).
    /// Whether `b`'s last run, with the same dynamic steps, put nothing on the replica lane (no step there, no copy onto
    /// it; or no replica schedule at all), so the replica's plaintext mirror goes unread again. False before its first run.
    pub fn replica_idle<W: DevWord>(&self, b: &Bound<'_, W>) -> bool {
        let dynamic: Vec<u32> = b.dynamic.iter().copied().collect();
        matches!(self.replica_last.get(&b.label), Some((d, false)) if *d == dynamic)
    }

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
        // the share of the chain's primes live at a level (the simulation's cost scale)
        let chain = rt.sh.cfg.chain;
        let top = chain.top_max() as f64 + 1.0;
        let live = |level: u32| chain.primes_dropped(level).map_or(1.0, |d| ((top - d as f64) / top).clamp(0.02, 1.0));
        // PC_REPLICA: the replica lane's plan instead of the side and remote lanes
        if let Some(rr) = rt.remote.as_deref().filter(|r| crate::replica::mode() > 0 && r.replica_on()) {
            let holds = |k: RouteKind| rr.replica_holds(k);
            // one ciphertext across the devices: its level's limbs, both parts, at ~30 GB/s (a kernel reading the peer)
            let n = b.ring_n as f64;
            let (lat, gbs) = crate::compile::two::copy_model();
            let copy_ms = |level: u32| -> f64 {
                let limbs = chain.primes_dropped(level).map_or(top, |d| (top - d as f64).max(1.0));
                lat + 2.0 * limbs * n * std::mem::size_of::<W>() as f64 / (gbs * 1e6)
            };
            let pt_ok = |p: u32| rr.replica_pt(p);
            // ownership (PC_OWN): a finish's owned outputs (its service says which), a split refresh batch's remote
            // members' outputs (the last half, as the split runs them there)
            let fin = rt.finish.as_deref();
            let owned = |i: usize| -> Vec<u32> {
                if !crate::replica::own() {
                    return vec![];
                }
                // PC_OWN_ONLY=fin / units (diagnostic): ownership for the finishes only, or the split refresh batches only
                let only = std::env::var("PC_OWN_ONLY").unwrap_or_default();
                match &b.steps[i].op {
                    StepOp::Finish { .. } if only == "units" => vec![],
                    StepOp::Finish { f } => match (fin, b.finishes.get(*f as usize)) {
                        (Some(fx), Some(fs)) => fx.owned(fs),
                        _ => vec![],
                    },
                    _ if only == "fin" => vec![],
                    _ => match b.unit_of.get(i).filter(|&&u| u != NONE).map(|&u| &b.units[u as usize]) {
                        Some(u) if u.lo as usize == i && matches!(&u.kind, UnitKind::Refresh { kind, .. } if rr.has(*kind)) && u.hi - u.lo >= 2 => {
                            let k = (u.hi - u.lo) as usize;
                            (u.lo as usize + k - k / 2..u.hi as usize).map(|j| b.steps[j].out).collect()
                        }
                        _ => vec![],
                    },
                }
            };
            let rs = crate::replica::plan(b, &holds, &pt_ok, &owned, &copy_ms, crate::replica::mode() == 2);
            // PC_REPLICA_DUMP=<dir> (diagnostic): the plan per step, one JSON line each (<dir>/<block>.jsonl)
            if let Ok(dir) = std::env::var("PC_REPLICA_DUMP") {
                let mut out = String::new();
                for (i, st) in b.steps.iter().enumerate() {
                    let (r, w) = schedule::rw(b, st);
                    let row = serde_json::json!({"i": i, "name": b.names.get(st.var as usize).cloned().unwrap_or_default(), "op": format!("{:?}", st.op).chars().take(40).collect::<String>(),
                        "dev": rs.dev[i], "unit": b.unit_of.get(i).copied().filter(|&u| u != NONE).map(|u| (b.units[u as usize].lo, b.units[u as usize].hi, format!("{:?}", b.units[u as usize].kind).chars().take(20).collect::<String>())),
                        "dyn": b.dynamic.contains(&(i as u32)), "r": r, "w": w, "copies": rs.copies[i], "wait": rs.wait[i], "own": rs.own[i], "signal": rs.signal[i], "lvl": st.meta.level, "deg": st.meta.deg, "early": rs.early.get(i).map_or(vec![], |cs| cs.iter().map(|&c| { let x = &rs.xc[c as usize]; serde_json::json!({"c": c, "slot": x.slot, "to": x.to, "waits": x.waits, "at": x.at}) }).collect::<Vec<_>>()), "hint": b.hint.get(i).copied().unwrap_or(0), "split": b.unit_of.get(i).is_some_and(|&u| u != NONE && rs.split_unit.get(u as usize).copied().unwrap_or(false))});
                    out.push_str(&row.to_string());
                    out.push('\n');
                }
                let _ = std::fs::create_dir_all(&dir);
                let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{}.jsonl", b.label)), out);
            }
            if std::env::var_os("PC_REPLICA_LOG").is_some() {
                eprintln!(
                    "[replica] {}: {} of {} steps on the replica, {} copies; static {:.1} ms -> {:.1} ms; placement sim {:.1} -> {:.1} ms{}{}",
                    b.label,
                    rs.moved,
                    b.steps.len(),
                    rs.n_copies,
                    rs.one_ms,
                    rs.two_ms,
                    rs.base_ms,
                    rs.base_ms - rs.saving_ms,
                    if rs.owner_ms > 0.0 { format!(" (owner-computes {:.1} ms{})", rs.owner_ms, if rs.owner { ", taken" } else { "" }) } else { String::new() },
                    if rs.hint_ms > 0.0 { format!(" (hints {:.1} ms, {})", rs.hint_ms, if rs.hint { "taken" } else { "components taken" }) } else { String::new() }
                );
            }
            let n = b.steps.len();
            if rs.moved == 0 {
                // nothing moves: the block keeps the side and remote lanes
                return self.lane_schedule(b, rt, dynamic, &live);
            }
            // the replica takes the block's lanes (side, asynchronous refreshes): it must save more than the refreshes
            // the asynchronous lane offloads there (~6 ms per refresh step moved, a pair's two steps one refresh)
            // PC_REPLICA_BTS: the replica's plan places the refreshes itself (the lanes' only offload): it stands
            let moves_bts = crate::replica::bts() && (0..n).any(|i| rs.dev[i] == 1 && matches!(b.steps[i].op, StepOp::Bootstrap { .. }));
            // PC_SCHED2: the block was ordered for the two devices; its hints stand
            if crate::replica::mode() == 1 && !moves_bts && !crate::replica::hinted(b) {
                if let Some(ls) = self.lane_schedule(b, rt, dynamic.clone(), &live) {
                    let lane_ms = 6.0 * ls.n_remote as f64;
                    if lane_ms >= rs.saving_ms {
                        if std::env::var_os("PC_REPLICA_LOG").is_some() {
                            eprintln!("[replica] {}: kept on its lanes ({} asynchronous refresh steps ~{lane_ms:.0} ms vs the replica's {:.0} ms)", b.label, ls.n_remote, rs.saving_ms);
                        }
                        return Some(ls);
                    }
                }
            }
            let lane: Vec<u8> = rs.dev.iter().map(|&d| if d == 1 { 4 } else { 0 }).collect();
            let s = Sched { lane, wait: vec![[NONE; 4]; n], wait_out: vec![[NONE; 4]; n], signal: vec![false; n], signal_in: vec![false; n], n_side: 0, n_remote: rs.moved, n_split: 0, total_ms: rs.one_ms, critical_ms: rs.two_ms, replica: Some(rs) };
            self.scheds.insert(b.label.clone(), (dynamic, s.clone()));
            return Some(s);
        }
        self.lane_schedule(b, rt, dynamic, &live)
    }

    /// The side and remote lanes' schedule of `b` (cached for its dynamic-step set).
    fn lane_schedule<'a, W: DevWord, M: DevStrategy<W>>(&mut self, b: &Bound<'a, W>, rt: &Rt<'_, 'a, W, M>, dynamic: Vec<u32>, live: &dyn Fn(u32) -> f64) -> Option<Sched> {
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
        // PC_REMOTE_ASYNC: whole refreshes may move onto the remote device's lane
        let rm = rt.remote.as_deref().filter(|r| remote_async() && r.lane_ok());
        let holds = |k: RouteKind| rm.is_some_and(|r| r.has(k));
        let rp = rm.map(|r| schedule::RemotePlan { holds: &holds, finishes: r.has_finish() });
        let s = schedule::plan(b, &side_ok, live, rp.as_ref());
        self.scheds.insert(b.label.clone(), (dynamic, s.clone()));
        Some(s)
    }

    pub fn run<'a, W: DevWord, M: DevStrategy<W>>(&mut self, b: &mut Bound<'a, W>, rt: &mut Rt<'_, 'a, W, M>) -> Result<RunInfo> {
        let t0 = Instant::now();
        let mut info = RunInfo { served: Served::Eager, host_us: 0.0, launch_us: 0.0, meta_mismatches: 0, first_mismatches: vec![], pairs: [0; 2], fused: 0 };
        if b.pts.iter().any(|p| !p.stream && p.meta.is_none()) {
            return Err(rerr(&b.label, "plaintexts not resolved (Bound::resolve_pts after staging)"));
        }
        let graphs = self.cfg.use_graphs && !self.eager_only.contains(&b.label);
        if graphs && rt.remote.is_some() && remote_check() {
            return Err(rerr(&b.label, "PC_REMOTE_CHECK compares on the host after each remote refresh: run it eager (graphs off)"));
        }
        // PC_GRAPH_EAGER_FINISH (default "ccmm"): finish services run eager between the graph segments (the CCMM's
        // stc_high caches are LRUs whose keys change per call: a capture cannot bake them, ForbidLock refuses it);
        // marked before the lane schedule, whose regions end at the dynamic steps
        if graphs && b.graphs.is_empty() && !self.eager_marked.contains(&b.label) {
            self.eager_marked.insert(b.label.clone());
            let ops = std::env::var("PC_GRAPH_EAGER_FINISH").unwrap_or_else(|_| "ccmm".into());
            let ops: Vec<String> = ops.split(',').filter(|x| !x.is_empty()).map(|x| format!("\"op\": \"{x}\"")).collect();
            let idx: Vec<usize> = (0..b.steps.len())
                .filter(|&i| match &b.steps[i].op {
                    StepOp::Finish { f } => b.finishes.get(*f as usize).is_some_and(|fs| {
                        let d = fs.desc.replace("\":\"", "\": \"");
                        ops.iter().any(|o| d.contains(o.as_str()))
                    }),
                    _ => false,
                })
                .collect();
            for i in idx {
                b.make_dynamic(i)?;
            }
        }
        let sched = self.schedule(b, rt);
        let sched = sched.as_ref();
        let busy = sched.and_then(|s| s.replica.as_ref()).is_some_and(|rs| rs.moved > 0 || rs.dev.contains(&1) || !rs.xc.is_empty() || rs.copies.iter().any(|c| !c.is_empty()));
        self.replica_last.insert(b.label.clone(), (b.dynamic.iter().copied().collect(), busy));
        if graphs && sched.is_some_and(|s| s.replica.is_some()) && replica_check() {
            return Err(rerr(&b.label, "PC_REPLICA_CHECK runs each block twice and compares on the host: run it eager (graphs off)"));
        }
        if !graphs {
            if let (true, Some(s)) = (replica_check(), sched.filter(|s| s.replica.is_some())) {
                // PC_REPLICA_CHECK: the block on this device alone (no replica, no side lane), then again from the same
                // inputs with the replica; every output word and metadata compared
                let ctx = rt.ctx;
                let dl = |rt: &Rt<'_, 'a, W, M>, x: &Ct<'a, W>| -> Result<(Vec<Vec<W>>, CtMeta)> {
                    let top = pc_shadow::meta::top_limb(ctx.shape, x.meta.level).map_err(|e| rerr("replica check", e.to_string()))? as usize;
                    let rows: Vec<usize> = (0..=top).collect();
                    Ok(((0..2).map(|p| x.buf.download_rows(&rt.ex.compute, 0, p, &rows)).collect::<std::result::Result<_, _>>()?, x.meta))
                };
                rt.ex.compute.sync()?;
                let ins: Vec<(u32, (Vec<Vec<W>>, CtMeta))> = b.inputs.iter().map(|(_, sl, _)| Ok((*sl, dl(rt, &b.slots[*sl as usize])?))).collect::<Result<_>>()?;
                // PC_REPLICA_CHECK=steps: both runs step by step, every step's output hashed (where it ran), the first
                // step that differs reported
                let steps_mode = std::env::var("PC_REPLICA_CHECK").is_ok_and(|v| v == "steps" || v == "units");
                // a fused unit's steps but its last are never written: not compared
                let inner = |b: &Bound<'a, W>, i: usize| -> bool {
                    b.unit_of.get(i).filter(|&&u| u != NONE).is_some_and(|&u| {
                        let un = &b.units[u as usize];
                        matches!(un.kind, UnitKind::Affine { .. } | UnitKind::MultPtAdd { .. } | UnitKind::MultPtAddChain { .. } | UnitKind::RotAcc { .. } | UnitKind::MultIAdd { .. }) && i + 1 < un.hi as usize
                    })
                };
                let hash = |rt: &Rt<'_, 'a, W, M>, x: &Ct<'a, W>| -> Result<u64> {
                    let (d, m) = dl(rt, x)?;
                    let mut h: u64 = 0xcbf29ce484222325 ^ (m.level as u64) ^ ((m.deg as u64) << 8) ^ m.nf.to_bits().rotate_left(17) ^ ((m.slots as u64) << 40) ^ format!("{:?}", m.tag).len() as u64;
                    for p in &d {
                        for w in p {
                            h = (h ^ w.to_u64()).wrapping_mul(0x100000001b3);
                        }
                    }
                    Ok(h)
                };
                let mut h1: Vec<u64> = vec![];
                let mut m1: Vec<Vec<CtMeta>> = vec![];
                let mut r1: Vec<Vec<u64>> = vec![];
                if steps_mode {
                    let mut hk = |i: usize, b: &mut Bound<'a, W>, rt: &mut Rt<'_, 'a, W, M>| -> Result<()> {
                        rt.ex.compute.sync()?;
                        let o = b.steps[i].out;
                        h1.push(if (o as usize) < b.slots.len() { hash(rt, &b.slots[o as usize])? } else { 0 });
                        m1.push(vec![b.slots.get(o as usize).map_or(CtMeta::new(0, 0), |c| c.meta)]);
                        let rd = schedule::rw(b, &b.steps[i]).0;
                        r1.push(rd.iter().filter(|&&s| (s as usize) < b.slots.len()).map(|&s| hash(rt, &b.slots[s as usize])).collect::<Result<Vec<_>>>()?);
                        Ok(())
                    };
                    issue(rt, b, false, &mut info, Some(&mut hk), None)?;
                } else {
                    issue(rt, b, false, &mut info, None, None)?;
                }
                rt.ex.compute.sync()?;
                let want: Vec<(Vec<Vec<W>>, CtMeta)> = b.outputs.iter().map(|(_, sl, _)| dl(rt, &b.slots[*sl as usize])).collect::<Result<_>>()?;
                // every slot a step writes (inputs aside) zeroed here, its metadata reset: a version the replica run
                // reads here without copying it back cannot pass for the run alone's
                let mut written: Vec<u32> = b.steps.iter().flat_map(|st| schedule::rw(b, st).1).filter(|&sl| (sl as usize) < b.slots.len() && !ins.iter().any(|(i, _)| *i == sl)).collect();
                written.sort_unstable();
                written.dedup();
                for sl in written {
                    let x = &mut b.slots[sl as usize];
                    x.buf.buf.zero(&rt.ex.compute).map_err(pc_gpu::Error::from)?;
                    x.meta = CtMeta::new(0, 1);
                }
                for (sl, (data, m)) in &ins {
                    let x = &mut b.slots[*sl as usize];
                    let top = pc_shadow::meta::top_limb(ctx.shape, m.level).map_err(|e| rerr("replica check", e.to_string()))? as usize;
                    let rows: Vec<usize> = (0..=top).collect();
                    for (part, d) in data.iter().enumerate() {
                        x.buf.upload_rows(&rt.ex.compute, 0, part, &rows, d)?;
                    }
                    x.meta = *m;
                }
                // PC_REPLICA_CHECK=self (control): the second run on this device alone too
                let self_ctl = std::env::var("PC_REPLICA_CHECK").is_ok_and(|v| v == "self");
                if steps_mode {
                    let rs = s.replica.as_ref();
                    let mut first: Option<String> = None;
                    let mut hk = |i: usize, b: &mut Bound<'a, W>, rt: &mut Rt<'_, 'a, W, M>| -> Result<()> {
                        if first.is_some() {
                            return Ok(());
                        }
                        rt.ex.compute.sync()?;
                        let o = b.steps[i].out;
                        if (o as usize) >= b.slots.len() {
                            return Ok(());
                        }
                        let on_rep = rs.is_some_and(|r| r.on_replica(i));
                        let h = if on_rep {
                            let rm = rt.remote.as_deref().ok_or_else(no_replica)?;
                            let x: *const Ct<'a, W> = rm.replica_slot(o).ok_or_else(|| rerr(&b.label, "replica slot"))?;
                            // SAFETY: the replica's slot, read after its stream drained (PC_REPLICA_SYNC)
                            hash(rt, unsafe { &*x })?
                        } else {
                            hash(rt, &b.slots[o as usize])?
                        };
                        if h1.get(i) != Some(&h) && !inner(b, i) {
                            let st = &b.steps[i];
                            let got_meta = if on_rep { rt.remote.as_deref().and_then(|r| r.replica_slot(o)).map(|c| c.meta) } else { Some(b.slots[o as usize].meta) };
                            let reads: Vec<(u32, CtMeta)> = schedule::rw(b, st).0.into_iter().filter_map(|s| b.slots.get(s as usize).map(|c| (s, c.meta))).collect();
                            let rh: Vec<u64> = schedule::rw(b, st).0.into_iter().filter(|&s| (s as usize) < b.slots.len()).map(|s| hash(rt, &b.slots[s as usize])).collect::<Result<Vec<_>>>()?;
                            eprintln!("[replica-check] step {i}: read-slot hashes with the replica {rh:x?}, alone {:x?}", r1.get(i));
                            // the unit's other members too
                            if let Some(&u) = b.unit_of.get(i).filter(|&&u| u != NONE) {
                                let un = &b.units[u as usize];
                                for j in un.lo as usize..un.hi as usize {
                                    let rj: Vec<u64> = schedule::rw(b, &b.steps[j]).0.into_iter().filter(|&s| (s as usize) < b.slots.len()).map(|s| hash(rt, &b.slots[s as usize])).collect::<Result<Vec<_>>>()?;
                                    let oj = b.steps[j].out;
                                    eprintln!("[replica-check]   unit {u} ({:?}) step {j}: reads {:?} {rj:x?} (alone {:x?}), out {oj} hash {:x} (alone {:x?})", un.kind, schedule::rw(b, &b.steps[j]).0, r1.get(j), hash(rt, &b.slots[oj as usize])?, h1.get(j));
                                }
                            }
                            first = Some(format!("step {i} ({} {:?}, {}, unit {}, reads {:?}, copies {:?}); output meta {:?}, alone {:?}", b.names[st.var as usize], st.op, if on_rep { "replica" } else { "here" }, b.unit_of.get(i).copied().unwrap_or(NONE) as i64, reads, rs.map(|r| r.copies[i].clone()), got_meta, m1.get(i)));
                        }
                        Ok(())
                    };
                    issue(rt, b, false, &mut info, Some(&mut hk), Some(s))?;
                    if let Some(f) = first {
                        return Err(rerr(&b.label, format!("PC_REPLICA_CHECK=steps: first step differing with the replica: {f}")));
                    }
                } else {
                issue(rt, b, self.cfg.check_meta, &mut info, None, if self_ctl { None } else { Some(s) })?;
                }
                rt.ex.compute.sync()?;
                for (k, (_, sl, _)) in b.outputs.iter().enumerate() {
                    let got = dl(rt, &b.slots[*sl as usize])?;
                    if got.1 != want[k].1 || got.0 != want[k].0 {
                        return Err(rerr(&b.label, format!("PC_REPLICA_CHECK: output {k} ({}) differs with the replica (metadata {:?} vs {:?})", b.outputs[k].0, got.1, want[k].1)));
                    }
                }
                REPLICA_CHECKED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            } else {
                issue(rt, b, self.cfg.check_meta, &mut info, None, sched)?;
            }
            self.stats.eager += 1;
            self.stats.pairs += info.pairs[0];
            self.stats.pairs_split += info.pairs[1];
            self.stats.fused += info.fused;
            info.host_us = t0.elapsed().as_secs_f64() * 1e6;
            return Ok(info);
        }
        let dynamic: Vec<u32> = b.dynamic.iter().copied().collect();
        let rows = slot_rows(b);
        let hit = b.graphs.iter().position(|g| g.dynamic == dynamic && g.stream_base == b.stream_base && g.slot_rows == rows && g.pt_rows.len() == b.pts.len() && g.pt_rows.iter().zip(&b.pts).all(|(r, p)| *r == p.rows));
        if let Some(gi) = hit {
            // the dynamic steps need the Bound mutably: take the captured block out while it replays
            let g = b.graphs.remove(gi);
            let mut res = Ok(());
            self.seg_kinds.clear();
            self.seg_host_us.clear();
            let seg = self.seg_ev.len() > g.items.len();
            for (k, it) in g.items.iter().enumerate() {
                let th = Instant::now();
                if seg {
                    self.seg_ev[k].record(&rt.ex.compute)?;
                    self.seg_kinds.push(match it {
                        Item::Graph(_) => usize::MAX,
                        Item::Step(i) => *i,
                    });
                }
                res = match it {
                    Item::Graph(exec) => {
                        let tl = Instant::now();
                        let r = exec.launch(&rt.ex.compute).map_err(Error::from);
                        info.launch_us += tl.elapsed().as_secs_f64() * 1e6;
                        r
                    }
                    Item::Step(i) => issue_dynamic(rt, b, *i, sched),
                };
                if seg {
                    self.seg_host_us.push(th.elapsed().as_secs_f64() * 1e6);
                }
                if res.is_err() {
                    break;
                }
            }
            if seg {
                self.seg_ev[g.items.len()].record(&rt.ex.compute)?;
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
        self.stats.pairs += info.pairs[0];
        self.stats.pairs_split += info.pairs[1];
        self.stats.fused += info.fused;
        let out_metas: Vec<CtMeta> = b.outputs.iter().map(|(_, s, _)| b.slots[*s as usize].meta).collect();
        info.host_us = t0.elapsed().as_secs_f64() * 1e6;
        // keep PC_GRAPH_RESERVE_GB (default 1) of device memory free for what the eager steps and lazily loaded
        // kernels still allocate: below it, the block is not captured and runs eager from then on
        let reserve = std::env::var("PC_GRAPH_RESERVE_GB").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(1.0);
        if pc_cuda::device::mem_info_current().is_ok_and(|(free, _)| (free as f64) < reserve * 1e9) {
            self.eager_only.insert(b.label.clone());
            self.stats.graph_oom += 1;
            self.stats.eager += 1;
            eprintln!("[replay] {}: under {reserve} GB of free device memory, the block runs eager", b.label);
            return Ok(info);
        }
        let tc = Instant::now();
        // the eager outputs must not be overwritten by anything the capture records (nothing executes in a capture,
        // but the lease starts with a lane sync, which also orders the outputs before the caller reads them)
        rt.ops.ring.lease_begin(&rt.ex.compute)?;
        if let Some(s) = rt.side.as_deref_mut() {
            s.ops.ring.lease_begin(&s.ex.compute)?;
        }
        if let Some(r) = rt.remote.as_deref_mut() {
            r.ring_lease_begin()?;
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
        let mut oom = false;
        for seg in segments(b) {
            match seg {
                Ok((lo, hi)) => {
                    let mut err: Option<Error> = None;
                    let mut scratch = RunInfo { served: Served::Eager, host_us: 0.0, launch_us: 0.0, meta_mismatches: 0, first_mismatches: vec![], pairs: [0; 2], fused: 0 };
                    let r = s.capture(|_| {
                        issue_range(rt, b, lo, hi, false, &mut scratch, None, sched).map_err(|e| {
                            err = Some(e);
                            pc_cuda::Error::Unavailable("replay step under capture")
                        })
                    });
                    match (r, err) {
                        (_, Some(e)) => {
                            // the constant ring ran out of free slots, or a locked route cache missed, under capture:
                            // the block cannot be captured as it stands, it runs eager (as out of memory)
                            let m = e.to_string();
                            oom = m.contains("OpsCfg::const_slots") || m.contains("while locked");
                            failed = Some(e)
                        }
                        (Err(e), None) => {
                            oom = is_oom(&e);
                            failed = Some(e.into())
                        }
                        (Ok((exec, ())), None) => items.push(Item::Graph(exec)),
                    }
                    if failed.is_some() {
                        break;
                    }
                }
                Err(d) => {
                    // not issued under capture: its outputs (a finish lands several) take the metadata the eager run
                    // produced
                    match b.dyn_metas.get(&(d as u32)) {
                        Some(m) => {
                            for &(o, mo) in &m.outs {
                                if let Some(c) = b.slots.get_mut(o as usize) {
                                    c.meta = mo;
                                }
                            }
                            // ownership: the replica's copies of the outputs it leaves there (written when the step
                            // runs between the graphs) take them too, for the segments captured after it
                            if let (Some(own), Some(rm)) = (sched.and_then(|s| s.replica.as_ref()).and_then(|rs| rs.own.get(d)), rt.remote.as_deref_mut()) {
                                for &sl in own {
                                    if let Some(&(_, mo)) = m.outs.iter().find(|(o, _)| *o == sl) {
                                        if let Err(e) = rm.replica_set_meta(sl, mo) {
                                            failed = Some(e);
                                        }
                                    }
                                }
                                if failed.is_some() {
                                    break;
                                }
                            }
                        }
                        None => {
                            failed = Some(rerr(&b.label, format!("dynamic step {d} never ran eagerly")));
                            break;
                        }
                    }
                    items.push(Item::Step(d));
                }
            }
        }
        rt.locks.iter().for_each(|l| l.set_locked(false));
        // both lanes end their leases before any error returns (a lease whose upload fails releases its slots)
        let lease = rt.ops.ring.lease_end(&rt.ex.compute);
        let side_lease = rt.side.as_deref_mut().map_or(Ok(vec![]), |s| s.ops.ring.lease_end(&s.ex.compute));
        let remote_lease = rt.remote.as_deref_mut().map_or(Ok(vec![]), |r| r.ring_lease_end());
        let (lease, side_lease, remote_lease) = match (lease, side_lease, remote_lease) {
            (Ok(l), Ok(sl), Ok(rl)) => (l, sl, rl),
            (l, sl, rl) => {
                if let Ok(l) = &l {
                    rt.ops.ring.release(l);
                }
                if let (Ok(sl), Some(s)) = (&sl, rt.side.as_deref_mut()) {
                    s.ops.ring.release(sl);
                }
                if let (Ok(rl), Some(r)) = (&rl, rt.remote.as_deref_mut()) {
                    r.ring_release(rl);
                }
                // the capture's own error first
                let e = failed.or(l.err().map(Error::from)).or(sl.err().map(Error::from)).or(rl.err().map(Error::from));
                return Err(e.unwrap_or_else(|| rerr(&b.label, "constant ring lease")));
            }
        };
        let release_all = |rt: &mut Rt<'_, 'a, W, M>| {
            rt.ops.ring.release(&lease);
            if let Some(s) = rt.side.as_deref_mut() {
                s.ops.ring.release(&side_lease);
            }
            if let Some(r) = rt.remote.as_deref_mut() {
                r.ring_release(&remote_lease);
            }
        };
        if let Some(e) = failed {
            release_all(rt);
            if oom {
                // the graph does not fit: free what this block captured, keep the eager run's outputs, run it eager
                drop(items);
                for (i, (_, s, _)) in b.outputs.iter().enumerate() {
                    b.slots[*s as usize].meta = out_metas[i];
                }
                self.eager_only.insert(b.label.clone());
                self.stats.graph_oom += 1;
                self.stats.eager += 1;
                eprintln!("[replay] {}: graph capture out of device memory, the block runs eager", b.label);
                info.served = Served::Eager;
                return Ok(info);
            }
            return Err(e);
        }
        // the capture left the eager outputs' metadata on the slots again (same steps); keep the recorded ones
        let cap_metas: Vec<CtMeta> = b.outputs.iter().map(|(_, s, _)| b.slots[*s as usize].meta).collect();
        if cap_metas.iter().zip(&out_metas).any(|(a, b)| meta_differs(a, b)) {
            release_all(rt);
            return Err(rerr(&b.label, "captured program lands its outputs at other metadata than its eager run (non-deterministic step)"));
        }
        let capture_ms = tc.elapsed().as_secs_f64() * 1e3;
        if std::env::var_os("PC_GRAPH_LOG").is_some() {
            let free = pc_cuda::device::mem_info_current().map(|m| m.0 as f64 / 1e9).unwrap_or(-1.0);
            let (cu, ct) = rt.ops.ring.leased_units();
            let (su, stt) = rt.side.as_deref().map_or((0, 0), |s| s.ops.ring.leased_units());
            eprintln!("[replay] {}: captured {} graphs, {:.0} ms, constants {} + {} side (leased units {cu} / {ct}, side {su} / {stt}), device free {free:.2} GB",
                b.label, items.iter().filter(|i| matches!(i, Item::Graph(_))).count(), capture_ms, lease.len(), side_lease.len());
        }
        self.stats.captures += 1;
        self.stats.capture_ms += capture_ms;
        b.graphs.push(Captured { items, dynamic, lease, side_lease, remote_lease, out_metas, pt_rows: b.pts.iter().map(|p| p.rows).collect(), slot_rows: rows, stream_base: b.stream_base, capture_ms });
        info.served = Served::Captured;
        Ok(info)
    }

    /// One eager run with `hook` called after every step (debugging: per-step decryptions, metadata traces).
    pub fn eager_with<'a, W: DevWord, M: DevStrategy<W>>(&mut self, b: &mut Bound<'a, W>, rt: &mut Rt<'_, 'a, W, M>, hook: &mut StepHook<'_, 'a, W, M>) -> Result<RunInfo> {
        let t0 = Instant::now();
        let mut info = RunInfo { served: Served::Eager, host_us: 0.0, launch_us: 0.0, meta_mismatches: 0, first_mismatches: vec![], pairs: [0; 2], fused: 0 };
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
            // a launch completes with all its nodes, the remote device's too: the compute sync above covers them
            if let Some(r) = rt.remote.as_deref_mut() {
                r.ring_release(&g.remote_lease);
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
