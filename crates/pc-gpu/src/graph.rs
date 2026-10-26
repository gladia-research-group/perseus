//! CUDA-graph capture and replay of a whole bootstrap batch (replay = one `GraphExec` per program on `compute`).
//!
//! A graph bakes every device address and host-side decision of the captured refresh, so an entry is keyed by
//! everything those depend on: `GraphKey` = (route, entry level, degree, scale, K, entry phase). The batch is gathered
//! into the fixed `BatchSlots` buffer before a launch and scattered back after (two copy launches per member, cheaper
//! than a capture + instantiate per input address). Per-call constants are held by a `ConstRing` lease, on-demand
//! route caches refuse a miss during capture (`CaptureLock`); details: docs/ARCHITECTURE.md "pc-gpu CUDA graphs".
//! The first call of a key computes its result eagerly (a capture records without executing), then captures;
//! later calls replay.
use crate::bts::batch::{BatchKey, BatchSlots, BtsBatch};
use crate::bts::{BtsCx, BtsRoute};
use crate::context::{DevStrategy, DevWord};
use crate::ct::Ct;
use crate::error::{shape, Error, Result};
use crate::poly::Layout;
use pc_cuda::{Arena, Compute, GraphExec, Stream};
use pc_shadow::CtMeta;
use std::time::Instant;

/// A cache a refresh fills on demand: locked while a capture runs, watched for evictions.
pub trait CaptureLock {
    fn set_locked(&self, on: bool);
    /// Monotone eviction counter.
    fn evictions(&self) -> u64;
}

impl<W: DevWord> CaptureLock for crate::bts::stc_first::StageCache<'_, W> {
    fn set_locked(&self, on: bool) {
        crate::bts::stc_first::StageCache::set_locked(self, on)
    }
    fn evictions(&self) -> u64 {
        self.stats().evictions
    }
}

/// What one graph is specialized to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GraphKey {
    pub batch: BatchKey,
    /// Ciphertexts per launch.
    pub k: usize,
    /// Which program of the route (0 = the whole refresh `BtsRoute::run`; callers capturing a staged sub-sequence
    /// pick their own tags, e.g. the PCMM finish from `Phase::Folded`).
    pub entry: u32,
}

/// Counters (setup report / bench).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GraphStats {
    pub captures: usize,
    pub replays: usize,
    pub invalidations: usize,
    /// Host ms spent capturing + instantiating (sum).
    pub capture_ms: f64,
}

struct Entry {
    key: GraphKey,
    exec: GraphExec,
    out: CtMeta,
    lease: Vec<usize>,
}

/// How a call was served.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Served {
    /// First call of the key: eager run, then captured.
    Captured,
    Replayed,
}

/// The graphs of the bootstrap batches of a session, over one fixed gather/scatter buffer.
pub struct BtsGraphs<'a, W: DevWord> {
    pub slots: BatchSlots<'a, W>,
    entries: Vec<Entry>,
    /// Entry bound (`bts.graphs`): graphs are host and device resources; the compiler knows the key set.
    pub cap: usize,
    pub stats: GraphStats,
}

/// Raw stream access for a capture while the refresh borrows the executor mutably: the capture only begins and
/// ends on the handle, the refresh issues its work on the same stream through `cx`.
fn capture<R>(s: *const Stream<Compute>, f: impl FnOnce() -> Result<R>) -> Result<(GraphExec, R)> {
    // SAFETY: `s` points at the executor's compute stream, alive for the call
    let s = unsafe { &*s };
    let mut err = None;
    let r = s.capture(|_| {
        f().map_err(|e| {
            err = Some(e);
            pc_cuda::Error::Unavailable("bootstrap under capture")
        })
    });
    match (r, err) {
        (_, Some(e)) => Err(e),
        (r, None) => Ok(r?),
    }
}

impl<'a, W: DevWord> BtsGraphs<'a, W> {
    /// Arena bytes (MemPlan section "bts.batch").
    pub fn bytes(lay: &Layout, k_max: usize) -> usize {
        BatchSlots::<W>::bytes(lay, k_max)
    }
    pub fn new(arena: &'a Arena, lay: Layout, k_max: usize, cap: usize) -> Result<Self> {
        Ok(BtsGraphs { slots: BatchSlots::new(arena, lay, k_max)?, entries: Vec::new(), cap, stats: GraphStats::default() })
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn keys(&self) -> Vec<GraphKey> {
        self.entries.iter().map(|e| e.key).collect()
    }

    /// Drop every graph (after the lane drained) and return their constant slots.
    pub fn clear<M: DevStrategy<W>>(&mut self, cx: &mut BtsCx<'_, 'a, W, M>) -> Result<()> {
        cx.ex.compute.sync()?;
        for e in self.entries.drain(..) {
            cx.ops.ring.release(&e.lease);
        }
        Ok(())
    }

    /// Refresh batch `b` with the whole route (graph program 0).
    pub fn run_route<M: DevStrategy<W>>(
        &mut self,
        route: &dyn BtsRoute<'a, W, M>,
        cx: &mut BtsCx<'_, 'a, W, M>,
        b: &mut BtsBatch<'_, 'a, W>,
        locks: &[&dyn CaptureLock],
    ) -> Result<Served> {
        let r = b.route;
        self.run(cx, b, 0, locks, |cx, x| route.run(cx, std::slice::from_mut(x), &r))
    }

    /// Run `f` (a refresh program over one K-batch `Ct`, deterministic in its key) on batch `b` through the fixed
    /// slots: replay its graph, or, on the first call of the key, run it eagerly and capture it.
    pub fn run<M: DevStrategy<W>>(
        &mut self,
        cx: &mut BtsCx<'_, 'a, W, M>,
        b: &mut BtsBatch<'_, 'a, W>,
        entry: u32,
        locks: &[&dyn CaptureLock],
        mut f: impl FnMut(&mut BtsCx<'_, 'a, W, M>, &mut Ct<'a, W>) -> Result<()>,
    ) -> Result<Served> {
        self.slots.gather(cx, b)?;
        let key = GraphKey { batch: b.key(), k: b.k(), entry };
        if let Some(e) = self.entries.iter().find(|e| e.key == key) {
            e.exec.launch(&cx.ex.compute)?;
            self.slots.x.meta = e.out;
            self.slots.scatter(cx, b)?;
            self.stats.replays += 1;
            return Ok(Served::Replayed);
        }
        if self.entries.len() >= self.cap {
            return Err(Error::Mem { knob: "bts.graphs", value: self.cap, msg: format!("graph table full; key {key:?} not captured") });
        }
        // first call: eager (warms the route's caches), result scattered
        let in_meta = self.slots.x.meta;
        let ev0: Vec<u64> = locks.iter().map(|l| l.evictions()).collect();
        f(cx, &mut self.slots.x)?;
        let out = self.slots.x.meta;
        self.slots.scatter(cx, b)?;
        if locks.iter().zip(&ev0).any(|(l, &e)| l.evictions() != e) && !self.entries.is_empty() {
            // a cache entry an earlier graph reads may have been rebuilt for another key
            self.clear(cx)?;
            self.stats.invalidations += 1;
        }
        // capture (no execution): fixed slots, leased constants, locked caches
        let t = Instant::now();
        self.slots.x.meta = in_meta;
        cx.ops.ring.lease_begin(&cx.ex.compute)?;
        locks.iter().for_each(|l| l.set_locked(true));
        let sp: *const Stream<Compute> = &cx.ex.compute;
        let x = &mut self.slots.x;
        let r = capture(sp, || f(cx, x));
        locks.iter().for_each(|l| l.set_locked(false));
        let lease = cx.ops.ring.lease_end(&cx.ex.compute);
        let (exec, ()) = match r {
            Ok(v) => v,
            Err(e) => {
                if let Ok(l) = &lease {
                    cx.ops.ring.release(l);
                }
                return Err(e);
            }
        };
        let lease = lease?;
        if self.slots.x.meta != out {
            cx.ops.ring.release(&lease);
            return Err(shape("bts graph", format!("captured refresh lands at {:?}, its eager run at {out:?} (non-deterministic program)", self.slots.x.meta)));
        }
        self.stats.capture_ms += t.elapsed().as_secs_f64() * 1e3;
        self.stats.captures += 1;
        self.entries.push(Entry { key, exec, out, lease });
        Ok(Served::Captured)
    }
}
