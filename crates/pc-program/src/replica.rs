//! `replica`: which steps of a bound block the remote device runs on its replica lane (a second binding of the block
//! over a mirrored slab, Sylph's activation ownership), and the copies and cross-device waits that keeps exact.
//!
//! Placement: whole components move, never single steps (a ciphertext copy costs ~1 ms, a product ~0.09 ms: a
//! step-by-step greedy pays back its moves in copies). The movable steps of a region form weakly connected
//! components through their read-after-write edges (a unit stays whole); the largest are tried on the replica one
//! by one, each kept when a program-order simulation of the block on two in-order devices (this device's compute
//! stream, the replica's stream; every copy occupying the receiving stream for its level's bytes) finishes at least
//! `MIN_GAIN_MS` earlier. Stay on this device: finishes (their services split them
//! themselves), streamed-weight sums, dynamic steps, steps reading plaintexts the replica holds no copy of
//! (`pt_ok`: the plaintext mirror, PC_REPLICA_PT), refresh batches the remote lane already splits, refreshes whose
//! route the remote device does not hold. A finish counts on
//! both devices for half its cost.
//!
//! Exactness: each device has its own slot memory (`Slab`), every slot's current version is valid on a set of
//! devices. A step reading a slot whose version is only on the other device copies it first, on its own stream,
//! after the producer's event (read after write). Overwriting a slot on one device waits for the other device's
//! copies that read that slot since its last write there (write after read): a copy runs on the reader's stream, so
//! its step's event (recorded after the step, which comes after its copies) covers it. Within a device, stream order
//! covers everything. A region (the steps between two dynamic steps) forks the replica from compute at its start and
//! joins it at its end, so a captured segment never waits on an event outside it; a slot's validity survives a region
//! boundary (only the waits restart). The block's outputs (read by the driver) are copied back at its end.
//!
//! Ownership (PC_OWN): the outputs a finish or a split refresh batch refreshes on the remote device stay there too
//! (its services copy them into the replica's slots), so their version is valid on both devices after the step; the
//! replica's readers wait for the step's event, and the step waits for the replica's last access to those slots.
//! PC_REPLICA_PLACE=split: a component left whole is split by owner-computes (each group where its inputs live, a
//! widely read value weighted down): a norm's blocks normalized on their devices, its statistic (summed in halves by
//! PC_SUM_TREE, compile::tree) crossing once each way, the 2-device all-reduce.
use crate::bind::{Bound, StepOp, UnitKind, NONE};
use crate::schedule::rw;
use pc_gpu::bts::RouteKind;
use pc_gpu::DevWord;

/// A component moves only if the simulated block finishes at least this much earlier (ms).
pub const MIN_GAIN_MS: f64 = 0.5;
/// Components tried per block (the largest first), and the smallest worth a try (ms).
const MAX_TRIES: usize = 160;
const MIN_COMP_MS: f64 = 2.0;

/// The replica plan of one block.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ReplicaSched {
    /// 0 = this device (compute), 1 = the replica; a unit's steps all carry its first step's.
    pub dev: Vec<u8>,
    /// Per step (a unit: its first step): slots copied onto the step's device just before it, from the other device,
    /// with the step whose event the copy waits for (the version's producer there; NONE = ordered already).
    pub copies: Vec<Vec<(u32, u32)>>,
    /// Per step (a unit: its first step): steps of the other device whose events it waits for before running (write
    /// after read: the other device's copies of a slot this step overwrites), on top of its copies' waits.
    pub wait: Vec<Vec<u32>>,
    /// Record an event after this step (some step or copy of the other device waits for it).
    pub signal: Vec<bool>,
    /// Per step (a unit: its first step): ownership (PC_OWN), the output slots a finish or a split refresh batch
    /// leaves in the replica's slots too (its remote part's outputs, copied there on the remote device), valid on
    /// both devices after the step.
    pub own: Vec<Vec<u32>>,
    /// Per unit: its members placed on both devices (PC_REPLICA_SPLITK), so they issue one by one, each its own step
    /// for the copies, waits and events.
    pub split_unit: Vec<bool>,
    /// PC_REPLICA_DMA: the copies between the devices as operations of their own (id = the block's steps + index),
    /// issued early on the receiver's copy stream (DMA over the bus while both devices compute); a step's `wait` names
    /// the copies it reads.
    pub xc: Vec<XCopy>,
    /// PC_REPLICA_DMA: per step, the copies (indices into `xc`) issued just before it.
    pub early: Vec<Vec<u32>>,
    /// Slots copied back onto this device at the block's end (outputs whose version is on the replica only), with
    /// their producers.
    pub out_copies: Vec<(u32, u32)>,
    pub moved: usize,
    pub n_copies: usize,
    /// Static-model ms: one device, the two-device estimate; the placement simulation with nothing moved and with the
    /// chosen components (`saving_ms` = their difference).
    pub one_ms: f64,
    pub two_ms: f64,
    pub base_ms: f64,
    pub saving_ms: f64,
    /// PC_REPLICA_PLACE: the owner-computes placement's simulated block (ms, 0 = not tried) and whether it was taken
    pub owner_ms: f64,
    pub owner: bool,
    /// PC_SCHED2: the hinted placement's simulated block (ms, 0 = no hints) and whether it was taken (else the
    /// components' placement on the same order simulated faster)
    pub hint_ms: f64,
    pub hint: bool,
}

impl ReplicaSched {
    pub fn on_replica(&self, i: usize) -> bool {
        self.dev.get(i).is_some_and(|&d| d == 1)
    }
    /// Where id `q` (a step, or a copy: steps + index) is issued: a step at its index, a copy just before step `at`.
    pub fn pos(&self, q: u32) -> usize {
        let n = self.dev.len();
        if (q as usize) < n {
            q as usize
        } else {
            self.xc.get(q as usize - n).map_or(usize::MAX, |c| c.at as usize)
        }
    }
}

/// A copy between the devices issued early (PC_REPLICA_DMA): slot `slot` of device `1 - to` into the same slot of
/// device `to`, on `to`'s copy stream, after the ids `waits` (steps or copies: the version's producer, the receiver's
/// accesses of the slot's old content), just before step `at`; its event orders the steps that read it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct XCopy {
    pub slot: u32,
    pub to: u8,
    pub waits: Vec<u32>,
    pub at: u32,
}

/// PC_OWN_LAZY=1 (with ownership; default off until an A/B wins): an owned output (a finish's or a split refresh
/// batch's remote part) lives on the remote device only: the remote services skip copying it back here, the replica
/// plan copies it over (after the owning step's event) where this device reads it.
pub fn own_lazy() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| own() && matches!(std::env::var("PC_OWN_LAZY").as_deref().map(str::trim), Ok("1") | Ok("on")))
}

/// PC_REPLICA_LOCAL=1 (with the replica lane; default off until an A/B wins): the placement tries, among components of
/// equal cost, those whose inputs are already on the remote device (owned outputs) first.
pub fn local_first() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| matches!(std::env::var("PC_REPLICA_LOCAL").as_deref().map(str::trim), Ok("1") | Ok("on")))
}

/// PC_REPLICA_UNIT_SPLIT=1 (with the replica lane; default off until an A/B wins): the placement simulation counts a
/// split refresh batch's members per device as the remote lane splits it (an odd batch: one more here), so a lone
/// refresh beside it can move there.
pub fn unit_split() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| matches!(std::env::var("PC_REPLICA_UNIT_SPLIT").as_deref().map(str::trim), Ok("1") | Ok("on")))
}

/// PC_REPLICA_DMA (with the replica lane, on by default; `0` / `off` off): copies between the devices on copy
/// streams, issued as soon as their version exists.
pub fn dma() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| !matches!(std::env::var("PC_REPLICA_DMA").as_deref().map(str::trim), Ok("0") | Ok("off")))
}

/// With PC_REPLICA_DMA, the placement simulates the copy engines (copies off the compute streams); PC_DMA_PLACE=0 keeps
/// the copies occupying the receiver (the placement without DMA).
pub fn dma_place() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| !matches!(std::env::var("PC_DMA_PLACE").as_deref().map(str::trim), Ok("0") | Ok("off")))
}

/// Device time of a step (ms, n32): the Llama prefill profile's averages per kind (a finish ~11.7 ms per output
/// refreshed; refreshes ~11.9 ms; products 0.09, key switches 0.06, rescales 0.03, element-wise ~0.01).
fn cost<W: DevWord>(b: &Bound<'_, W>, i: usize) -> f64 {
    match &b.steps[i].op {
        StepOp::Finish { f } => 11.7 * b.finishes.get(*f as usize).map_or(1, |fs| fs.outs.len().max(1)) as f64,
        // a refresh: ~11.9 ms on the dense route (Llama's complex payloads), ~7.5 ms a single on the others, a complex
        // pair ~10 ms (5 per step)
        StepOp::Bootstrap { kind, .. } => match b.unit_of.get(i).filter(|&&u| u != NONE).map(|&u| &b.units[u as usize].kind) {
            Some(UnitKind::Pair { .. }) => 5.0,
            _ if matches!(kind, RouteKind::Dense) => 11.9,
            _ => 7.5,
        },
        StepOp::CoeffSum { .. } => 0.5,
        StepOp::Rotate { g, .. } if *g == NONE => 0.005,
        StepOp::Rotate { .. } | StepOp::Conjugate { .. } => 0.058,
        StepOp::Mult | StepOp::Square | StepOp::MultRescale { .. } => 0.09,
        StepOp::MultSum { .. } => 0.12,
        StepOp::RotateSum { l, .. } => 0.06 * (b.step_lists.get(*l as usize).map_or(1, |v| v.len()) as f64 + 1.0),
        StepOp::Rescale => 0.029,
        StepOp::MultPt(_) | StepOp::AddPt(_) => 0.011,
        _ => 0.008,
    }
}

/// The steps as issue groups: a unit's steps together, others alone ([lo, hi)).
fn groups<W: DevWord>(b: &Bound<'_, W>) -> Vec<(usize, usize)> {
    let n = b.steps.len();
    let mut g = vec![];
    let mut i = 0;
    while i < n {
        let hi = match b.unit_of.get(i).filter(|&&u| u != NONE).map(|&u| &b.units[u as usize]) {
            // PC_REPLICA_SPLITK: a batch of products is no group (its members, often of different heads, would glue
            // the heads' work into one component); members placed on both devices issue one by one
            Some(u) if splitk() && matches!(u.kind, UnitKind::Mult { .. }) => i + 1,
            Some(u) => u.hi as usize,
            None => i + 1,
        }
        .max(i + 1);
        g.push((i, hi));
        i = hi;
    }
    g
}

/// PC_REPLICA_SPLITK (on by default; `0` / `off` off): product batches may be split between the devices by the replica's
/// placement (a head's work on either device: Cerium's head ownership).
pub fn splitk() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| !matches!(std::env::var("PC_REPLICA_SPLITK").as_deref().map(str::trim), Ok("0") | Ok("off")))
}

/// The slots a group reads from outside it and the slots it writes.
fn group_rw<W: DevWord>(b: &Bound<'_, W>, (lo, hi): (usize, usize), ns: usize) -> (Vec<u32>, Vec<u32>) {
    let (mut reads, mut writes): (Vec<u32>, Vec<u32>) = (vec![], vec![]);
    for j in lo..hi {
        let (r, w) = rw(b, &b.steps[j]);
        for s in r {
            if !writes.contains(&s) && !reads.contains(&s) && (s as usize) < ns {
                reads.push(s);
            }
        }
        for s in w {
            if !writes.contains(&s) && (s as usize) < ns {
                writes.push(s);
            }
        }
    }
    (reads, writes)
}

/// The block's time on two in-order devices for group placement `on` (1 = replica): finishes and split refresh
/// batches half on each device after their inputs (here), every copy occupying the receiving stream (with
/// PC_REPLICA_DMA: a copy engine per receiver instead, each copy queued on it from its version's time; only the
/// readers wait).
#[allow(clippy::too_many_arguments)]
fn simulate<W: DevWord>(b: &Bound<'_, W>, gs: &[(usize, usize)], grw: &[(Vec<u32>, Vec<u32>)], gc: &[f64], gown: &[Vec<u32>], on: &[u8], copy_ms: &dyn Fn(u32) -> f64, ns: usize) -> f64 {
    let mut valid: Vec<u8> = vec![1; ns];
    let mut ready: Vec<[f64; 2]> = vec![[0.0; 2]; ns];
    let mut t = [0.0f64; 2];
    let engines = dma() && dma_place();
    let mut ce = [0.0f64; 2];
    for (k, &(lo, _)) in gs.iter().enumerate() {
        if b.dynamic.contains(&(lo as u32)) {
            let m = t[0].max(t[1]);
            t = [m, m];
        }
        let (reads, writes) = &grw[k];
        let cp = copy_ms(b.steps[lo].meta.level);
        let both = matches!(b.steps[lo].op, StepOp::Finish { .. }) || b.unit_of.get(lo).filter(|&&u| u != NONE).is_some_and(|&u| matches!(b.units[u as usize].kind, UnitKind::Refresh { .. }));
        let d = if both { 0 } else { on[k] as usize };
        let mut x = 0.0f64;
        let mut ncp = 0usize;
        for &s in reads {
            let s = s as usize;
            if valid[s] & (1 << d) != 0 {
                x = x.max(ready[s][d]);
            } else if engines {
                ce[d] = ce[d].max(ready[s][1 - d]) + cp;
                x = x.max(ce[d]);
                valid[s] |= 1 << d;
            } else {
                x = x.max(ready[s][1 - d] + cp);
                ncp += 1;
                valid[s] |= 1 << d;
            }
        }
        let start = t[d].max(x);
        for &s in reads {
            let s = s as usize;
            ready[s][d] = ready[s][d].max(start);
        }
        let end = if both {
            // PC_REPLICA_UNIT_SPLIT=1: a split refresh batch of K members as the remote lane splits it (ceil(K/2) here,
            // floor(K/2) there: an odd batch leaves one more refresh here); otherwise (and finishes) half each
            let (h0, h1) = match b.unit_of.get(lo).filter(|&&u| u != NONE).map(|&u| &b.units[u as usize]) {
                Some(u) if unit_split() && matches!(u.kind, UnitKind::Refresh { .. }) && u.hi - u.lo >= 2 => {
                    let n = (u.hi - u.lo) as f64;
                    let r = ((u.hi - u.lo) / 2) as f64;
                    ((n - r) / n, r / n)
                }
                _ => (0.5, 0.5),
            };
            let f0 = start + ncp as f64 * cp + gc[k] * h0;
            let f1 = t[1].max(x) + gc[k] * h1;
            t = [f0, f1];
            f0.max(f1)
        } else {
            t[d] = start + ncp as f64 * cp + gc[k];
            t[d]
        };
        for &s in writes {
            let su = s as usize;
            if gown[k].contains(&s) && own_lazy() {
                // PC_OWN_LAZY: the owned version lives on the remote device only
                valid[su] = 2;
                ready[su][1] = end;
            } else if gown[k].contains(&s) {
                valid[su] = 3;
                ready[su] = [end, end];
            } else {
                valid[su] = 1 << d;
                ready[su][d] = end;
            }
        }
    }
    t[0].max(t[1])
}

/// Group placement: whole components of movable groups, the largest first, each kept when the simulated block
/// finishes `MIN_GAIN_MS` earlier (`force_all`: every movable group).
#[allow(clippy::too_many_arguments)]
fn place<W: DevWord>(b: &Bound<'_, W>, gs: &[(usize, usize)], grw: &[(Vec<u32>, Vec<u32>)], gc: &[f64], gown: &[Vec<u32>], holds: &dyn Fn(RouteKind) -> bool, pt_ok: &dyn Fn(u32) -> bool, copy_ms: &dyn Fn(u32) -> f64, ns: usize, force_all: bool, owner: Option<&[u8]>) -> (Vec<u8>, f64, f64) {
    let mut mv: Vec<bool> = gs.iter().map(|&(lo, _)| movable(b, lo, holds, pt_ok)).collect();
    // PC_REPLICA_BTS_RANGE=lo:hi (diagnostic): only the block's movable refreshes of ordinal lo..hi may move
    if let Some((rl, rh)) = std::env::var("PC_REPLICA_BTS_RANGE").ok().and_then(|v| v.split_once(':').and_then(|(a, b)| Some((a.trim().parse::<usize>().ok()?, b.trim().parse::<usize>().ok()?)))) {
        let mut ord = 0usize;
        for (k, &(lo, _)) in gs.iter().enumerate() {
            if mv[k] && matches!(b.steps[lo].op, StepOp::Bootstrap { .. }) {
                if ord < rl || ord >= rh {
                    mv[k] = false;
                }
                ord += 1;
            }
        }
    }
    if force_all {
        return (mv.iter().map(|&m| m as u8).collect(), 0.0, 0.0);
    }
    let mut par: Vec<usize> = (0..gs.len()).collect();
    fn fd(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    // a refresh (single or pair: PC_REPLICA_BTS) is a candidate of its own, never a link between components (the
    // element-wise work on both sides of it would glue into one)
    let boot: Vec<bool> = gs.iter().map(|&(lo, _)| matches!(b.steps[lo].op, StepOp::Bootstrap { .. })).collect();
    let mut writer: Vec<Option<usize>> = vec![None; ns];
    for (k, &(lo, _)) in gs.iter().enumerate() {
        if b.dynamic.contains(&(lo as u32)) {
            writer.iter_mut().for_each(|w| *w = None);
        }
        let (reads, writes) = &grw[k];
        if mv[k] && !boot[k] {
            for &s in reads {
                if let Some(p) = writer[s as usize].filter(|&p| mv[p] && !boot[p]) {
                    let (a, c) = (fd(&mut par, k), fd(&mut par, p));
                    par[a] = c;
                }
            }
        }
        for &s in writes {
            writer[s as usize] = Some(k);
        }
    }
    let mut comps: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
    for k in 0..gs.len() {
        if mv[k] {
            let r = fd(&mut par, k);
            comps.entry(r).or_default().push(k);
        }
    }
    let mut cl: Vec<(f64, Vec<usize>)> = comps.into_values().map(|ks| (ks.iter().map(|&k| gc[k]).sum(), ks)).filter(|(c, _)| *c >= MIN_COMP_MS).collect();
    if local_first() {
        // PC_REPLICA_LOCAL=1: among components of the same cost (to 0.1 ms: the per-head / per-chain copies of one
        // computation), those whose inputs the replica already holds (owned outputs, or nothing read from outside)
        // first: the greedy then fills the remote device with work that needs no copy before work that does
        let mut writer: Vec<Option<usize>> = vec![None; ns];
        let mut ext_copy: Vec<u32> = vec![0; gs.len()];
        let mut comp_of: Vec<usize> = vec![usize::MAX; gs.len()];
        for (c, (_, ks)) in cl.iter().enumerate() {
            for &k in ks {
                comp_of[k] = c;
            }
        }
        for (k, &(lo, _)) in gs.iter().enumerate() {
            if b.dynamic.contains(&(lo as u32)) {
                writer.iter_mut().for_each(|w| *w = None);
            }
            for &s in &grw[k].0 {
                let w = writer[s as usize];
                let outside = w.is_none_or(|p| comp_of[p] != comp_of[k] || comp_of[k] == usize::MAX);
                let both = w.is_some_and(|p| gown[p].contains(&s));
                if outside && !both {
                    ext_copy[k] += 1;
                }
            }
            for &s in &grw[k].1 {
                writer[s as usize] = Some(k);
            }
        }
        let key = |c: &(f64, Vec<usize>)| -> (i64, u32) { ((c.0 * 10.0).round() as i64, c.1.iter().map(|&k| ext_copy[k]).sum()) };
        cl.sort_by(|a, b| {
            let (ka, kb) = (key(a), key(b));
            kb.0.cmp(&ka.0).then(ka.1.cmp(&kb.1))
        });
    } else {
        cl.sort_by(|a, b| b.0.total_cmp(&a.0));
    }
    cl.truncate(MAX_TRIES);
    let mut on = vec![0u8; gs.len()];
    let base = simulate(b, gs, grw, gc, gown, &on, copy_ms, ns);
    let mut best = base;
    let mut kept = vec![false; cl.len()];
    for (c, (_, ks)) in cl.iter().enumerate() {
        for &k in ks {
            on[k] = 1;
        }
        let v = simulate(b, gs, grw, gc, gown, &on, copy_ms, ns);
        if v + MIN_GAIN_MS <= best {
            best = v;
            kept[c] = true;
        } else {
            for &k in ks {
                on[k] = 0;
            }
        }
    }
    // PC_REPLICA_PLACE=split: a component that stays whole here is split by the owner-computes placement (each part
    // where its inputs live: a norm's blocks on their devices, its statistic summed across), kept when faster
    if let Some(ow) = owner {
        for (c, (_, ks)) in cl.iter().enumerate() {
            if kept[c] || !ks.iter().any(|&k| ow[k] == 1) {
                continue;
            }
            for &k in ks {
                on[k] = ow[k];
            }
            let v = simulate(b, gs, grw, gc, gown, &on, copy_ms, ns);
            if v + MIN_GAIN_MS <= best {
                best = v;
            } else {
                for &k in ks {
                    on[k] = 0;
                }
            }
        }
    }
    (on, base, best)
}

/// Owner-computes placement: in program order, each movable group on the device where its inputs live (a slot's
/// home: where its version was written; an owned output's, the remote device), each input weighted by 1 / its
/// version's readers (a broadcast value counts little against a block's own); ties stay here. Lane-0 groups (finishes,
/// split refresh batches, unmovable steps) run here, their owned outputs homed there.
fn owner_place<W: DevWord>(b: &Bound<'_, W>, gs: &[(usize, usize)], grw: &[(Vec<u32>, Vec<u32>)], gc: &[f64], gown: &[Vec<u32>], mv: &[bool], ns: usize) -> Vec<u8> {
    // readers per slot version (a version = the group that wrote it, or the block's input: usize::MAX)
    let mut writer: Vec<usize> = vec![usize::MAX; ns];
    let mut readers: std::collections::HashMap<(u32, usize), u32> = std::collections::HashMap::new();
    let mut ver: Vec<Vec<usize>> = Vec::with_capacity(gs.len());
    for (k, (reads, writes)) in grw.iter().enumerate() {
        let v: Vec<usize> = reads.iter().map(|&s| writer[s as usize]).collect();
        for (&s, &w) in reads.iter().zip(&v) {
            *readers.entry((s, w)).or_insert(0) += 1;
        }
        ver.push(v);
        for &s in writes {
            writer[s as usize] = k;
        }
    }
    // ciphertext ownership (PC_REPLICA_CTOWN=1, default off): the two members of a refresh pair are two
    // ciphertexts of one activation (a CNN layer packed in two ciphertexts). The second member's output is homed on the
    // remote device (the plan copies it there, 14 MiB at the landing), so that ciphertext's activation follows it; a
    // group whose every input is broadcast (read by >= CTOWN_BCAST groups: a convolution's rotation grid) goes with the
    // pair member it feeds, when it feeds members of one device only (its chains of plaintext products and their
    // rotation accumulation). Same ops, same order: word-exact.
    let ctown = ctown();
    let member1: Vec<Vec<u32>> = gs
        .iter()
        .map(|&(lo, _)| match b.unit_of.get(lo).filter(|&&u| u != NONE).map(|&u| &b.units[u as usize]) {
            Some(u) if ctown && matches!(u.kind, UnitKind::Pair { .. }) && u.lo as usize == lo && u.hi - u.lo == 2 => rw(b, &b.steps[lo + 1]).1,
            _ => vec![],
        })
        .collect();
    // sink[k]: bit d set when group k feeds (before any refresh) a pair member that runs on device d (member c on c);
    // a single refresh, a finish or the block's end feed device 0
    let mut sink: Vec<u8> = vec![0; gs.len()];
    if ctown {
        let mut cons: Vec<Vec<(usize, u32)>> = vec![vec![]; gs.len()];
        let mut wr: Vec<usize> = vec![usize::MAX; ns];
        for (k, (reads, writes)) in grw.iter().enumerate() {
            for &s in reads {
                if let Some(p) = Some(wr[s as usize]).filter(|&p| p != usize::MAX) {
                    cons[p].push((k, s));
                }
            }
            for &s in writes {
                wr[s as usize] = k;
            }
        }
        for k in (0..gs.len()).rev() {
            let mut m = 0u8;
            for &(c, s) in &cons[k] {
                let lo = gs[c].0;
                m |= if !member1[c].is_empty() {
                    // which member of the pair reads slot s
                    if rw(b, &b.steps[lo + 1]).0.contains(&s) { 2 } else { 1 }
                } else if matches!(b.steps[lo].op, StepOp::Bootstrap { .. } | StepOp::Finish { .. }) {
                    1
                } else {
                    sink[c]
                };
            }
            sink[k] = if cons[k].is_empty() { 1 } else { m };
        }
    }
    let mut home: Vec<u8> = vec![0; ns];
    // PC_REPLICA_INPUT_SPLIT=1 (default off until an A/B wins): the block's inputs (slots read before any write, in
    // first-read order) homed half here, half there: a norm on the block's inputs is split like the one on the
    // residual adds (each device normalizes its half, the statistic summed across), the placement copying the second
    // half over where it pays
    if input_split() {
        let mut seen: Vec<bool> = vec![false; ns];
        let mut ins: Vec<u32> = vec![];
        for (reads, writes) in grw {
            for &s in reads {
                if !seen[s as usize] {
                    seen[s as usize] = true;
                    ins.push(s);
                }
            }
            for &s in writes {
                seen[s as usize] = true;
            }
        }
        let n = ins.len();
        for &s in &ins[n - n / 2..] {
            home[s as usize] = 1;
        }
    }
    let mut on = vec![0u8; gs.len()];
    // PC_REPLICA_TIE=load: the owner-computes work placed per device since the region's start (a tie goes to the
    // lighter device)
    let tie_load = tie_load();
    let tie_alt = tie_alt();
    let mut toggle = 0u8;
    // group k: one add of two product-sum chains' outputs (PF_SINC_BI_SPLIT's halves)
    let chain_of = |g: usize| gs.get(g).and_then(|&(lo, _)| b.unit_of.get(lo).filter(|&&u| u != NONE)).is_some_and(|&u| matches!(b.units[u as usize].kind, UnitKind::MultPtAddChain { .. }));
    let halves_sum = |k: usize| {
        let (lo, hi) = gs[k];
        hi - lo == 1 && matches!(b.steps[lo].op, StepOp::Add) && ver[k].len() == 2 && ver[k].iter().all(|&w| w != usize::MAX && chain_of(w))
    };
    let mut load = [0.0f64; 2];
    for (k, &(lo, _)) in gs.iter().enumerate() {
        let (reads, writes) = &grw[k];
        if b.dynamic.contains(&(lo as u32)) {
            load = [0.0; 2];
        }
        let d = if mv[k] && !b.dynamic.contains(&(lo as u32)) && gown[k].is_empty() {
            let mut score = [0.0f64; 2];
            let mut bcast = !reads.is_empty();
            for (&s, &w) in reads.iter().zip(&ver[k]) {
                let r = readers.get(&(s, w)).copied().unwrap_or(1).max(1);
                bcast &= r >= CTOWN_BCAST;
                score[home[s as usize] as usize] += 1.0 / r as f64;
            }
            if score[1] > score[0] {
                1
            } else if tie_load && score[0] > 0.0 && score[1] > 0.0 && (score[1] - score[0]).abs() <= 1e-9 * score[0] {
                // inputs from both devices in equal parts: the lighter device so far
                (load[1] < load[0]) as u8
            } else if tie_alt && score[0] > 0.0 && score[1] > 0.0 && (score[1] - score[0]).abs() <= 1e-9 * score[0] && halves_sum(k) {
                // the sum of two masked half sums, one from each device (a split stage's outputs): the devices in turn
                toggle ^= 1;
                1 - toggle
            } else {
                (ctown && bcast && sink[k] == 2 && !matches!(b.steps[lo].op, StepOp::Bootstrap { .. })) as u8
            }
        } else {
            0
        };
        if mv[k] {
            load[d as usize] += gc[k];
        }
        on[k] = d;
        for &s in writes {
            home[s as usize] = if gown[k].contains(&s) || member1[k].contains(&s) { 1 } else { d };
        }
    }
    on
}

/// PC_REPLICA_TIE=load (with the owner-computes placement; default off until an A/B wins): a group reading as much
/// from each device goes to the device with less owner-computes work so far in the region (else it stays here).
pub fn tie_load() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| matches!(std::env::var("PC_REPLICA_TIE").as_deref().map(str::trim), Ok("load")))
}

/// PC_REPLICA_INPUT_SPLIT=1: owner-computes homes the second half of the block's inputs on the remote device.
pub fn input_split() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| matches!(std::env::var("PC_REPLICA_INPUT_SPLIT").as_deref().map(str::trim), Ok("1") | Ok("on")))
}

/// PC_REPLICA_TIE=alt: such ties go to the two devices in turn (this device first).
pub fn tie_alt() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| matches!(std::env::var("PC_REPLICA_TIE").as_deref().map(str::trim), Ok("alt")))
}

/// PC_REPLICA_CTOWN's broadcast: an input read by at least this many groups counts as shared by both devices.
const CTOWN_BCAST: u32 = 4;

/// PC_REPLICA_CTOWN=1 (with the replica lane; default off until an A/B wins): ciphertext ownership in owner-computes
/// (`owner_place`): a refresh pair's second ciphertext and the work that feeds it or reads it on the remote device.
pub fn ctown() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| matches!(std::env::var("PC_REPLICA_CTOWN").as_deref().map(str::trim), Ok("1") | Ok("on")))
}

/// PC_REPLICA_PLACE: `comp` (the components), `owner` (owner-computes), `best` (owner-computes where it simulates
/// faster than the components), `split` (unset: the components, then each one left whole split by owner-computes where
/// that simulates faster: a norm's all-reduce), `hint` (PC_SCHED2's device hints).
pub fn place_mode() -> u8 {
    static V: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("PC_REPLICA_PLACE").as_deref().map(str::trim) {
        Ok("comp") => 0,
        Ok("owner") => 1,
        Ok("best") => 2,
        Err(_) | Ok("") | Ok("split") => 3,
        Ok("hint") => 4,
        Ok(v) => {
            eprintln!("[replica] PC_REPLICA_PLACE={v}: not comp / owner / best / split / hint; split");
            3
        }
    })
}

/// Whether the block carries PC_SCHED2's device hints and the placement follows them (PC_REPLICA_PLACE unset or
/// `hint`).
pub fn hinted<W: DevWord>(b: &Bound<'_, W>) -> bool {
    let unset = std::env::var("PC_REPLICA_PLACE").map_or(true, |v| v.trim().is_empty());
    (place_mode() == 4 || unset) && b.hint.contains(&1)
}

/// Whether step / unit `i` may run on the replica.
fn movable<W: DevWord>(b: &Bound<'_, W>, i: usize, holds: &dyn Fn(RouteKind) -> bool, pt_ok: &dyn Fn(u32) -> bool) -> bool {
    if b.dynamic.contains(&(i as u32)) {
        return false;
    }
    let unit = b.unit_of.get(i).filter(|&&u| u != NONE).map(|&u| &b.units[u as usize]);
    if let Some(u) = unit {
        match &u.kind {
            UnitKind::Refresh { .. } => return false,
            // a refresh pair whole on the replica (PC_REPLICA_BTS: the remote device's compute stream runs it)
            UnitKind::Pair { kind, .. } => return bts_mode() & 2 != 0 && holds(*kind),
            UnitKind::MultPtAdd { pt, .. } if !pt_ok(*pt) => return false,
            UnitKind::MultPtAddChain { head, pts, .. } if !(head.is_none_or(|h| pt_ok(h)) && pts.iter().all(|&p| pt_ok(p))) => return false,
            _ => {}
        }
        return (u.lo as usize..u.hi as usize).all(|j| step_movable(b, j, holds, pt_ok));
    }
    step_movable(b, i, holds, pt_ok)
}

fn step_movable<W: DevWord>(b: &Bound<'_, W>, i: usize, holds: &dyn Fn(RouteKind) -> bool, pt_ok: &dyn Fn(u32) -> bool) -> bool {
    match &b.steps[i].op {
        StepOp::Finish { .. } | StepOp::CoeffSum { .. } => false,
        StepOp::MultPt(p) | StepOp::AddPt(p) => pt_ok(*p),
        StepOp::Bootstrap { kind, .. } => bts_mode() & 1 != 0 && holds(*kind),
        _ => true,
    }
}

/// PC_REPLICA_BTS (with the replica lane, on by default; `0` / `off` off): single refreshes and refresh pairs may move
/// onto the remote device with the replica's components (run on its compute stream, handed over by events).
pub fn bts() -> bool {
    bts_mode() != 0
}

/// PC_REPLICA_BTS's refreshes: bit 0 single refreshes, bit 1 refresh pairs (`1` / `on` both, `single`, `pair`).
pub fn bts_mode() -> u8 {
    static V: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("PC_REPLICA_BTS").as_deref().map(str::trim) {
        Ok("0") | Ok("off") => 0,
        Ok("single") => 1,
        Ok("pair") => 2,
        _ => 3,
    })
}

/// The replica plan of `b`. `holds`: the route families the replica holds; `pt_ok`: the plaintexts it holds a copy of;
/// `owned`: per step (a finish, a split refresh batch's first step), the output slots its remote part leaves in the
/// replica's slots too (ownership, PC_OWN; empty = none); `copy_ms`: one ciphertext copied across the devices at a
/// level (ms); `force_all` (PC_REPLICA=all, a correctness drill): every movable step on the replica.
#[allow(clippy::too_many_arguments)]
pub fn plan<W: DevWord>(b: &Bound<'_, W>, holds: &dyn Fn(RouteKind) -> bool, pt_ok: &dyn Fn(u32) -> bool, owned: &dyn Fn(usize) -> Vec<u32>, copy_ms: &dyn Fn(u32) -> f64, force_all: bool) -> ReplicaSched {
    let n = b.steps.len();
    let ns = b.n_slots.max(1) + 1;
    let gs = groups(b);
    let grw: Vec<(Vec<u32>, Vec<u32>)> = gs.iter().map(|&g| group_rw(b, g, ns)).collect();
    let gc: Vec<f64> = gs.iter().map(|&(lo, hi)| (lo..hi).map(|j| cost(b, j)).sum()).collect();
    // ownership: the slots a lane-0 group (a finish, a split refresh batch) leaves on both devices
    let gown: Vec<Vec<u32>> = gs
        .iter()
        .enumerate()
        .map(|(k, &(lo, _))| {
            let both = matches!(b.steps[lo].op, StepOp::Finish { .. }) || b.unit_of.get(lo).filter(|&&u| u != NONE).is_some_and(|&u| matches!(b.units[u as usize].kind, UnitKind::Refresh { .. }));
            if both {
                owned(lo).into_iter().filter(|s| grw[k].1.contains(s)).collect()
            } else {
                vec![]
            }
        })
        .collect();
    let pm = place_mode();
    // PC_SCHED2's device hints (the compiler's two-device schedule): each movable group where its ops were scheduled
    let hinted = !force_all && hinted(b);
    let owner_on = (pm > 0 && pm < 4 && !force_all && !hinted).then(|| {
        let mv: Vec<bool> = gs.iter().map(|&(lo, _)| movable(b, lo, holds, pt_ok)).collect();
        owner_place(b, &gs, &grw, &gc, &gown, &mv, ns)
    });
    let (mut hint_ms, mut hint) = (0.0, false);
    let (mut on, base, mut best) = if hinted {
        let on: Vec<u8> = gs.iter().map(|&(lo, hi)| (movable(b, lo, holds, pt_ok) && (lo..hi).any(|j| b.hint.get(j) == Some(&1))) as u8).collect();
        let v = simulate(b, &gs, &grw, &gc, &gown, &on, copy_ms, ns);
        // per block: the components' placement on the same order instead when it simulates faster
        let (on_c, base, v_c) = place(b, &gs, &grw, &gc, &gown, holds, pt_ok, copy_ms, ns, false, None);
        hint_ms = v;
        hint = v <= v_c + MIN_GAIN_MS;
        if hint {
            (on, base, v)
        } else {
            (on_c, base, v_c)
        }
    } else {
        place(b, &gs, &grw, &gc, &gown, holds, pt_ok, copy_ms, ns, force_all, owner_on.as_deref().filter(|_| pm == 3))
    };
    // PC_REPLICA_PLACE=owner / best: owner-computes placement (each movable group where its inputs live, a widely
    // read value counting less: a norm's statistic is broadcast, each device normalizes its own blocks), alone or
    // when it simulates faster than the components
    let (mut owner_ms, mut owner) = (0.0, false);
    if let (Some(on2), true) = (owner_on, pm == 1 || pm == 2) {
        let v2 = simulate(b, &gs, &grw, &gc, &gown, &on2, copy_ms, ns);
        owner_ms = v2;
        if pm == 1 || v2 + MIN_GAIN_MS <= best {
            on = on2;
            best = v2;
            owner = true;
        }
    }
    let own_at: std::collections::HashMap<usize, Vec<u32>> = gs.iter().zip(&gown).filter(|(_, o)| !o.is_empty()).map(|(&(lo, _), o)| (lo, o.clone())).collect();
    let mut place_of = vec![0u8; n];
    for (k, &(lo, hi)) in gs.iter().enumerate() {
        for j in lo..hi {
            place_of[j] = on[k];
        }
    }
    // a product batch whose members landed on both devices issues one by one; one on a single device stays a batch
    let split_unit: Vec<bool> = b.units.iter().map(|u| matches!(u.kind, UnitKind::Mult { .. }) && splitk() && (u.lo as usize..u.hi as usize).any(|j| place_of[j] != place_of[u.lo as usize])).collect();
    let mut rs = ReplicaSched { dev: vec![0; n], copies: vec![vec![]; n], wait: vec![vec![]; n], signal: vec![false; n], own: vec![vec![]; n], split_unit: split_unit.clone(), base_ms: base, saving_ms: base - best, owner_ms, owner, hint_ms, hint, ..Default::default() };
    // per slot: devices holding the current version (bit 0 / bit 1), the version's producer per device (the step that
    // wrote it there, or the step whose copy brought it), the time it is usable per device (static model)
    let mut valid: Vec<u8> = vec![1; ns];
    let mut prod: Vec<[u32; 2]> = vec![[NONE; 2]; ns];
    let mut ready: Vec<[f64; 2]> = vec![[0.0; 2]; ns];
    // per slot and device: the latest step of the OTHER device that copied this device's version of the slot (a
    // write here must wait for it), within the current region
    let mut copied_by: Vec<[u32; 2]> = vec![[NONE; 2]; ns];
    // per slot: the latest replica step that read or wrote the replica's version (its copies included), within the
    // current region (an owned write there by the remote device's services waits for it)
    let mut acc1: Vec<u32> = vec![NONE; ns];
    // per slot: the latest owning step (this device's) whose remote services copied into the replica's buffer of the
    // slot, within the current region: whatever writes that buffer later (a replica step, a copy onto the replica)
    // waits for it, whichever device holds the slot's current version by then (this device may have rewritten it since)
    let mut own_last: Vec<u32> = vec![NONE; ns];
    // PC_REPLICA_DMA: per slot and device, the latest step of that device that read or wrote its buffer of the slot
    // (within the current region): an early copy onto it waits for it (it no longer runs on that device's stream)
    let mut acc: Vec<[u32; 2]> = vec![[NONE; 2]; ns];
    let dma = dma();
    let n_id = n as u32;
    // where each early copy is issued (step index it precedes)
    let mut xat: Vec<usize> = vec![];
    // per group's first step: the group's end (its event is recorded once the whole group issued: a unit issued step by
    // step records it after its last step, so a copy waiting for it is issued after the group)
    let mut gend: Vec<usize> = (1..=n).collect();
    let pos = |q: u32, xat: &[usize]| -> usize {
        if q == NONE {
            usize::MAX
        } else if q < n_id {
            q as usize
        } else {
            xat[(q - n_id) as usize]
        }
    };
    rs.early = vec![vec![]; n];
    let mut t = [0.0f64; 2];
    let mut region_lo = 0usize;
    let mut i = 0usize;
    while i < n {
        if b.dynamic.contains(&(i as u32)) {
            // its copies back are issued in the region before it (a dynamic step runs alone between two graph
            // segments), after their producers' events there; then both devices join, the waits restart and the
            // versions stay where they are
            let (r, _) = rw(b, &b.steps[i]);
            let mut waits = vec![];
            for s in r.into_iter().filter(|&s| (s as usize) < ns) {
                let su = s as usize;
                if valid[su] & 1 == 0 && !rs.copies[i].iter().any(|c| c.0 == s) {
                    let q = copied_by[su][0];
                    if q != NONE && pos(q, &xat) >= region_lo && !waits.contains(&q) {
                        waits.push(q);
                        if q < n_id {
                            rs.signal[q as usize] = true;
                        }
                    }
                    let p = prod[su][1];
                    let p = if p != NONE && pos(p, &xat) >= region_lo { p } else { NONE };
                    if p != NONE && p < n_id {
                        rs.signal[p as usize] = true;
                    }
                    rs.copies[i].push((s, p));
                    rs.n_copies += 1;
                    valid[su] |= 1;
                    prod[su][0] = i as u32;
                    ready[su][0] = ready[su][1] + copy_ms(b.steps[i].meta.level);
                    copied_by[su][1] = i as u32;
                }
            }
            rs.wait[i] = waits;
            let m = t[0].max(t[1]);
            t = [m, m];
            copied_by.iter_mut().for_each(|x| *x = [NONE; 2]);
            acc1.iter_mut().for_each(|x| *x = NONE);
            own_last.iter_mut().for_each(|x| *x = NONE);
            acc.iter_mut().for_each(|x| *x = [NONE; 2]);
            region_lo = i;
        }
        let unit = b.unit_of.get(i).filter(|&&u| u != NONE && !split_unit[u as usize]).map(|&u| &b.units[u as usize]);
        let hi = unit.map_or(i + 1, |u| u.hi as usize).max(i + 1);
        gend[i] = hi;
        let (mut reads, mut writes): (Vec<u32>, Vec<u32>) = (vec![], vec![]);
        let mut c = 0.0;
        for j in i..hi {
            let (r, w) = rw(b, &b.steps[j]);
            for s in r {
                // a read of a slot an earlier step of the unit wrote is internal to the unit
                if !writes.contains(&s) && !reads.contains(&s) {
                    reads.push(s);
                }
            }
            for s in w {
                if !writes.contains(&s) {
                    writes.push(s);
                }
            }
            c += cost(b, j);
        }
        let reads: Vec<u32> = reads.into_iter().filter(|&s| (s as usize) < ns).collect();
        let writes: Vec<u32> = writes.into_iter().filter(|&s| (s as usize) < ns).collect();
        let is_fin = matches!(b.steps[i].op, StepOp::Finish { .. });
        let cp = copy_ms(b.steps[i].meta.level);
        // data arrival on device d (static model)
        let arrive = |d: usize, valid: &[u8], ready: &[[f64; 2]]| -> f64 {
            let mut x = 0.0f64;
            for &s in &reads {
                let s = s as usize;
                if valid[s] & (1 << d) != 0 {
                    x = x.max(ready[s][d]);
                } else {
                    x = x.max(ready[s][1 - d] + cp);
                }
            }
            x
        };
        let d = if is_fin { 0 } else { place_of[i] as usize };
        // copies onto d of the slots it reads that are not valid there
        let start = t[d].max(arrive(d, &valid, &ready));
        let mut waits: Vec<u32> = vec![];
        let war = |q: u32, waits: &mut Vec<u32>, signal: &mut Vec<bool>, xat: &[usize]| {
            if q != NONE && pos(q, xat) >= region_lo && (q as usize) != i {
                if !waits.contains(&q) {
                    waits.push(q);
                }
                if q < n_id {
                    signal[q as usize] = true;
                }
            }
        };
        for &s in &reads {
            let su = s as usize;
            if valid[su] & (1 << d) != 0 {
                // a version the other device's step left here (ownership), or an early copy brought: after its event
                let p = prod[su][d];
                if p != NONE && pos(p, &xat) >= region_lo && (p as usize) != i && (p >= n_id || rs.dev[p as usize] as usize != d) && !waits.contains(&p) {
                    waits.push(p);
                    if p < n_id {
                        rs.signal[p as usize] = true;
                    }
                }
            }
            if valid[su] & (1 << d) == 0 && dma {
                // an early copy: after the version's producer there and every access of d's buffer of the slot since
                // its last write (d's steps, the other device's copies of it, an owned write), issued right after the
                // last of those; this step waits for it
                let mut cw: Vec<u32> = vec![];
                let wp = prod[su][d];
                let wp = if wp != NONE && (wp >= n_id || rs.dev[wp as usize] as usize != d) { wp } else { NONE };
                for q in [prod[su][1 - d], acc[su][d], copied_by[su][d], if d == 1 { own_last[su] } else { NONE }, wp] {
                    if q != NONE && pos(q, &xat) >= region_lo && (q as usize) != i && !cw.contains(&q) {
                        cw.push(q);
                    }
                }
                let at_min = if region_lo > 0 { region_lo + 1 } else { 0 };
                let at = cw.iter().map(|&q| if q < n_id { gend[q as usize] } else { xat[(q - n_id) as usize] }).max().unwrap_or(0).max(at_min).min(i);
                for &q in &cw {
                    if q < n_id {
                        rs.signal[q as usize] = true;
                    }
                }
                let c = rs.xc.len() as u32;
                rs.xc.push(XCopy { slot: s, to: d as u8, waits: cw, at: at as u32 });
                xat.push(at);
                rs.early[at].push(c);
                waits.push(n_id + c);
                rs.n_copies += 1;
                copied_by[su][d] = NONE;
                valid[su] |= 1 << d;
                prod[su][d] = n_id + c;
                ready[su][d] = start;
                copied_by[su][1 - d] = n_id + c;
            } else if valid[su] & (1 << d) == 0 {
                // the copy overwrites d's (stale) version: after the other device's copies of that version, and on the
                // replica after the last owned copy into that buffer
                war(copied_by[su][d], &mut waits, &mut rs.signal, &xat);
                if d == 1 {
                    war(own_last[su], &mut waits, &mut rs.signal, &xat);
                }
                copied_by[su][d] = NONE;
                let p = prod[su][1 - d];
                let p = if p != NONE && pos(p, &xat) >= region_lo { p } else { NONE };
                if p != NONE && p < n_id {
                    rs.signal[p as usize] = true;
                }
                rs.copies[i].push((s, p));
                rs.n_copies += 1;
                valid[su] |= 1 << d;
                prod[su][d] = i as u32;
                ready[su][d] = start;
                // the other device's version was read by this step's copy
                copied_by[su][1 - d] = i as u32;
            }
        }
        // a finish reads its inputs on this device, and the remote device's part of it reads them there (its own
        // copies, inside the service): nothing more to model
        for &s in &writes {
            // write after read: the other device copied this device's version of s since its last write here
            war(copied_by[s as usize][d], &mut waits, &mut rs.signal, &xat);
            // write after write: a version here written from the other device's step (an owned output, written by
            // the remote device's services on another stream) lands before this write
            let p = prod[s as usize][d];
            if p != NONE && (p >= n_id || rs.dev[p as usize] as usize != d) {
                war(p, &mut waits, &mut rs.signal, &xat);
            }
            if d == 1 {
                war(own_last[s as usize], &mut waits, &mut rs.signal, &xat);
            }
        }
        let own = if d == 0 { own_at.get(&i).cloned().unwrap_or_default() } else { vec![] };
        for &s in &own {
            // the remote part writes the replica's s: after every replica step that read or wrote it (this step's
            // fork comes after its waits), and the copies of the replica's version back here are on this stream
            war(acc1[s as usize], &mut waits, &mut rs.signal, &xat);
        }
        rs.wait[i].extend(waits);
        let f = if is_fin {
            // half on each device after its inputs
            let f0 = t[0].max(arrive(0, &valid, &ready)) + c / 2.0;
            let f1 = t[1].max(arrive(0, &valid, &ready)) + c / 2.0;
            t = [f0, f1];
            f0.max(f1)
        } else {
            t[d] = start + c;
            t[d]
        };
        if d == 1 {
            for &s in reads.iter().chain(&writes) {
                acc1[s as usize] = i as u32;
            }
        }
        for &s in reads.iter().chain(&writes) {
            acc[s as usize][d] = i as u32;
        }
        for &s in &writes {
            let su = s as usize;
            if own.contains(&s) {
                // owned: valid on both devices, the replica's readers after this step's event (PC_OWN_LAZY: on the
                // remote device only, this device's readers copy it over after this step's event)
                own_last[su] = i as u32;
                if own_lazy() {
                    valid[su] = 2;
                    prod[su] = [NONE, i as u32];
                    ready[su][1] = f;
                } else {
                    valid[su] = 3;
                    prod[su] = [i as u32; 2];
                    ready[su] = [f; 2];
                }
                copied_by[su] = [NONE; 2];
                acc1[su] = NONE;
                continue;
            }
            valid[su] = 1 << d;
            prod[su] = [NONE; 2];
            prod[su][d] = i as u32;
            ready[su][d] = f;
            // the copies of d's old version are waited for; those of the other device's (stale) version stay pending
            // for the other device's next write of the slot
            copied_by[su][d] = NONE;
        }
        if !own.is_empty() {
            rs.signal[i] = true;
            rs.own[i] = own;
        }
        for j in i..hi {
            rs.dev[j] = d as u8;
        }
        if d == 1 {
            rs.moved += hi - i;
        }
        rs.one_ms += c;
        i = hi;
    }
    // the block's outputs back on this device
    let mut outs: Vec<u32> = b.outputs.iter().map(|o| o.1).collect();
    outs.sort_unstable();
    outs.dedup();
    for s in outs {
        let su = s as usize;
        if su < ns && valid[su] & 1 == 0 {
            let p = prod[su][1];
            let p = if p != NONE && pos(p, &xat) >= region_lo { p } else { NONE };
            if p != NONE && p < n_id {
                rs.signal[p as usize] = true;
            }
            rs.out_copies.push((s, p));
            rs.n_copies += 1;
            t[0] = t[0].max(ready[su][1] + copy_ms(b.steps[su.min(n - 1)].meta.level));
        }
    }
    rs.two_ms = t[0].max(t[1]);
    rs
}

/// PC_REPLICA (with a remote device): unset, `1` or `on` = the replica lane (default: faster on every model measured),
/// `all` = every movable step on it (a correctness drill), `0` / `off` = off.
pub fn mode() -> u8 {
    static V: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("PC_REPLICA").as_deref().map(str::trim) {
        Err(_) | Ok("") | Ok("1") | Ok("on") => 1,
        Ok("all") => 2,
        Ok("0") | Ok("off") => 0,
        Ok(v) => {
            eprintln!("[replica] PC_REPLICA={v}: not 1 / on / all or 0 / off; off");
            0
        }
    })
}

/// PC_REPLICA_PT (with the replica lane): the plaintexts element-wise steps read mirrored on the remote device, so steps
/// reading them may move there. Unset / `1` / `on` = on (faster on Llama, Gemma, BERT; even elsewhere), `0` / `off` = off.
pub fn pt_mirror() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("PC_REPLICA_PT").as_deref().map(str::trim) {
        Ok("0") | Ok("off") => false,
        Err(_) | Ok("") | Ok("1") | Ok("on") => true,
        Ok(v) => {
            eprintln!("[replica] PC_REPLICA_PT={v}: not 1 / on or 0 / off; off");
            false
        }
    })
}

/// PC_OWN (with the replica lane): ownership, the outputs a finish or a split refresh batch refreshes on the remote
/// device also left in the replica's slots there. Unset / `1` / `on` = on (faster on Llama, Gemma, BERT; even elsewhere),
/// `0` / `off` = off.
pub fn own() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("PC_OWN").as_deref().map(str::trim) {
        Ok("0") | Ok("off") => false,
        Err(_) | Ok("") | Ok("1") | Ok("on") => true,
        Ok(v) => {
            eprintln!("[replica] PC_OWN={v}: not 1 / on or 0 / off; off");
            false
        }
    })
}
