//! `schedule`: which lane each step of a bound block runs on (1 compute + side lanes, events at joins). The
//! compiler knows the DAG: steps off the block's critical path (under a static cost model: refreshes and key
//! switches dominate) go to the side lane, everything on it stays on compute. Refreshes, finishes,
//! streamed-weight sums, units (grids, K-batches), dynamic steps and steps using the block's scratch slot always run on compute
//! (their workspaces are compute's); a fused plaintext product and add (no gather buffer) may run on the side lane, and
//! an element-wise group runs as one pass only when its steps share a lane.
//!
//! Correctness does not depend on the heuristic: every slot hazard between the lanes (read after write, write after
//! read, write after write) becomes one event wait per other lane on its latest conflicting step (stream order
//! covers the earlier ones). Hazards are tracked per region (the steps between two dynamic steps): the replay forks
//! the side lane from compute at a region's first side step and joins it back at the region's end, so a captured
//! segment's graph holds the fork / join and never waits on an event of another capture.
//!
//! With a remote refresh device (`RemotePlan`, PC_REMOTE_ASYNC=1) a third lane (2) takes whole refreshes: single
//! refresh steps and refresh pairs, chosen greedily where a three-lane simulation says the block finishes earlier (the
//! remote device is held by every finish and split refresh batch for its whole span, and each remote refresh pays its
//! copies). A remote step runs in two parts: its inputs copied over and refreshed on the remote device (lane 2, after
//! what wrote its inputs), then the result copied back into its slot on this device's copy-back stream (lane 3, after
//! what still reads or writes that slot). Hazards are tracked on the four lanes: a later writer of a remote step's
//! input waits only for its inputs to be copied (lane 2's event), a reader of its output for the copy back (lane 3's).
//! The replay forks lanes 2 and 3 at the start of each region that has a remote step and joins them at its end.
use crate::bind::{Bound, Step, StepOp, UnitKind, NONE};
use pc_gpu::bts::RouteKind;
use pc_gpu::DevWord;
use std::collections::BTreeSet;

/// Lane of every step and the cross-lane waits.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sched {
    /// 0 = compute, 1 = side, 2 = the remote refresh device (its copy back on lane 3, this device's copy-back stream).
    pub lane: Vec<u8>,
    /// Per producer lane (0, 1, 2 = a remote step's inputs copied, 3 = a remote step's output copied back), the step
    /// whose event this step waits for, or NONE. A remote step: what its inputs wait for (before the copy over).
    pub wait: Vec<[u32; 4]>,
    /// A remote step: what its copy back waits for (the slot's earlier readers and writer); NONE elsewhere.
    pub wait_out: Vec<[u32; 4]>,
    /// Record an event after this step (some later step of another lane waits for it; a remote step: after its copy
    /// back, which the replay records anyway).
    pub signal: Vec<bool>,
    /// A remote step whose input copy some later step waits for.
    pub signal_in: Vec<bool>,
    pub n_side: usize,
    /// Steps on the remote lane.
    pub n_remote: usize,
    /// Element-wise groups whose steps sit on two lanes (issued step by step).
    pub n_split: usize,
    /// Static cost estimates (ms): every step, the critical path.
    pub total_ms: f64,
    pub critical_ms: f64,
    /// The replica lane's plan (`replica`; its steps carry lane 4, the others 0, no side or remote lane).
    pub replica: Option<crate::replica::ReplicaSched>,
}

/// Rough device cost of a step (ms, K = 1, n32 decode levels): only the ordering between steps matters.
fn cost(st: &Step, lists: &[Vec<i64>]) -> f64 {
    match &st.op {
        StepOp::Bootstrap { .. } | StepOp::Finish { .. } => 17.0,
        StepOp::CoeffSum { .. } => 1.0,
        StepOp::Mult | StepOp::Square | StepOp::MultRescale { .. } | StepOp::MultSum { .. } => 0.45,
        StepOp::Rotate { g, .. } if *g == NONE => 0.02,
        StepOp::Rotate { .. } | StepOp::Conjugate { .. } => 0.4,
        StepOp::RotateSum { l, .. } => 0.15 + 0.3 * lists.get(*l as usize).map_or(1, |v| v.len()) as f64,
        _ => 0.03,
    }
}

/// Slots a step reads and writes (finishes: their own lists).
pub(crate) fn rw<W: DevWord>(b: &Bound<'_, W>, st: &Step) -> (Vec<u32>, Vec<u32>) {
    if let StepOp::Finish { f } = &st.op {
        if let Some(fs) = b.finishes.get(*f as usize) {
            return (fs.ins.clone(), fs.outs.clone());
        }
    }
    if let StepOp::CoeffSum { c } = &st.op {
        if let Some(cs) = b.coeff_sums.get(*c as usize) {
            return (cs.ins.clone(), vec![st.out]);
        }
    }
    if let StepOp::MultSum { s } = &st.op {
        if let Some(ms) = b.mult_sums.get(*s as usize) {
            return (ms.clone(), vec![st.out]);
        }
    }
    let r = [st.a, st.b].into_iter().filter(|&s| s != NONE).collect();
    (r, vec![st.out])
}

/// Steps that must stay on compute: refreshes / finishes (BtsWork, routes), steps that use the block's scratch slot
/// (an in-place op without an in-place kernel), and what `side_ok` refuses (e.g. key groups that stage through
/// compute's buffers).
fn pinned(st: &Step, side_ok: &dyn Fn(&Step) -> bool) -> bool {
    let tmp = st.inplace && matches!(st.op, StepOp::Mult | StepOp::Square | StepOp::Negate | StepOp::Rotate { .. } | StepOp::RotateSum { .. } | StepOp::Conjugate { .. })
        || (st.inplace && matches!(st.op, StepOp::Add | StepOp::Sub) && st.b == st.out);
    // a streamed-weight sum uses compute's expander scratch and the frame compute joined
    // a fused relinearization-rescale runs on compute (its tables are the compute op layer's); a lazily relinearized
    // product sum copies its operands' preludes into the block's scratch slot
    matches!(st.op, StepOp::Bootstrap { .. } | StepOp::Finish { .. } | StepOp::CoeffSum { .. } | StepOp::MultRescale { .. } | StepOp::MultSum { .. }) || tmp || !side_ok(st)
}

/// The remote refresh device as a lane: the route families it holds and whether finishes use it (the finish split).
pub struct RemotePlan<'f> {
    pub holds: &'f dyn Fn(RouteKind) -> bool,
    pub finishes: bool,
}

/// Host-device copies of one remote refresh (ms per ciphertext, n32 at 38-40 GB/s): in at the input level, back at
/// the landing's.
const REMOTE_IN_MS: f64 = 0.10;
const REMOTE_OUT_MS: f64 = 0.25;
/// Greedy passes over the remote candidates.
const REMOTE_PASSES: usize = 3;

/// The schedule of `b` (its current dynamic steps) for one compute + one side lane (+ the remote refresh device).
pub fn plan<W: DevWord>(b: &Bound<'_, W>, side_ok: &dyn Fn(&Step) -> bool, live: &dyn Fn(u32) -> f64, remote: Option<&RemotePlan<'_>>) -> Sched {
    let n = b.steps.len();
    let dynamic: &BTreeSet<u32> = &b.dynamic;
    let costs: Vec<f64> = b.steps.iter().map(|s| cost(s, &b.step_lists)).collect();
    // data dependencies (read after write through slots, in program order), earliest finish, longest tail
    let mut writer: Vec<Option<usize>> = vec![None; b.n_slots.max(1) + 1];
    let mut preds: Vec<Vec<usize>> = vec![vec![]; n];
    for (i, st) in b.steps.iter().enumerate() {
        let (r, w) = rw(b, st);
        for s in r {
            if let Some(Some(p)) = writer.get(s as usize) {
                preds[i].push(*p);
            }
        }
        for s in w {
            if (s as usize) < writer.len() {
                writer[s as usize] = Some(i);
            }
        }
    }
    let mut fin = vec![0.0f64; n];
    for i in 0..n {
        let start = preds[i].iter().map(|&p| fin[p]).fold(0.0, f64::max);
        fin[i] = start + costs[i];
    }
    let total_cp = fin.iter().copied().fold(0.0, f64::max);
    let mut tail = vec![0.0f64; n];
    for i in (0..n).rev() {
        let t = costs[i] + tail[i];
        for &p in &preds[i] {
            if t > tail[p] {
                tail[p] = t;
            }
        }
    }
    // steps that must run on compute: a unit issues all its steps at once, on compute (its gather buffers are
    // compute's); a product-add pair on one lane, compute when either step is held there; an element-wise group keeps
    // its steps' lanes (`unit_lanes`)
    let forced: Vec<bool> = (0..n)
        .map(|i| {
            let unit = b.unit_of.get(i).filter(|&&u| u != NONE).map(|&u| &b.units[u as usize]);
            let gathers = unit.is_some_and(|u| !matches!(u.kind, UnitKind::MultPtAdd { .. } | UnitKind::MultPtAddChain { .. } | UnitKind::Affine { .. } | UnitKind::MultIAdd { .. }));
            gathers || dynamic.contains(&(i as u32)) || pinned(&b.steps[i], side_ok)
        })
        .collect();
    let mut lane: Vec<u8> = (0..n).map(|i| if forced[i] || fin[i] + tail[i] >= total_cp - 1e-6 { 0 } else { 1 }).collect();
    let spans: Vec<(usize, usize, bool)> = b
        .units
        .iter()
        .filter_map(|u| match u.kind {
            UnitKind::MultPtAdd { .. } | UnitKind::MultPtAddChain { .. } => Some((u.lo as usize, u.hi as usize, true)),
            UnitKind::Affine { .. } | UnitKind::MultIAdd { .. } => Some((u.lo as usize, u.hi as usize, false)),
            _ => None,
        })
        .collect();
    unit_lanes(&mut lane, &spans);
    // the side lane is one in-order stream: a side chain the DAG model gives slack can still finish after compute
    // reaches its consumer (the side lane's backlog); `relieve` moves such chains onto compute where the two-lane
    // simulation says the block finishes earlier (PC_LIN_OFF=sched_relief: off)
    if crate::program::lever_on("sched_relief") {
        let sim: Vec<f64> = b.steps.iter().map(|st| sim_cost(st, &b.step_lists, live(st.meta.level))).collect();
        let dyns: Vec<bool> = (0..n).map(|i| dynamic.contains(&(i as u32))).collect();
        relieve(&sim, &preds, &dyns, &spans, &mut lane, RELIEF_TRIES);
    }
    let n_split = unit_lanes(&mut lane, &spans);
    if let Some(rp) = remote {
        let sim: Vec<f64> = b.steps.iter().map(|st| sim_cost(st, &b.step_lists, live(st.meta.level))).collect();
        let dyns: Vec<bool> = (0..n).map(|i| dynamic.contains(&(i as u32))).collect();
        remote_lanes(b, rp, &sim, &preds, &dyns, &mut lane);
    }
    // hazards per region: last writer and the latest reader per reading lane (0, 1, 2) of every slot since its last
    // write. A step reads on its lane and writes on it, except a remote step: it reads on lane 2 and writes on 3.
    let rlane = |p: usize| lane[p] as usize;
    let wlane = |p: usize| if lane[p] == 2 { 3 } else { lane[p] as usize };
    let mut wait = vec![[NONE; 4]; n];
    let mut wait_out = vec![[NONE; 4]; n];
    let mut signal = vec![false; n];
    let mut signal_in = vec![false; n];
    let ns = b.n_slots.max(1) + 1;
    let mut last_w: Vec<Option<usize>> = vec![None; ns];
    let mut last_r: Vec<[Option<usize>; 3]> = vec![[None; 3]; ns];
    let latest = |d: &mut [Option<usize>; 4], pl: usize, p: usize| {
        if d[pl].map_or(true, |x| p > x) {
            d[pl] = Some(p);
        }
    };
    for i in 0..n {
        if dynamic.contains(&(i as u32)) {
            // a region boundary: the other lanes joined compute before it and fork after it
            last_w.iter_mut().for_each(|x| *x = None);
            last_r.iter_mut().for_each(|x| *x = [None; 3]);
            continue;
        }
        let (lr, lw) = (rlane(i), wlane(i));
        let (r, w) = rw(b, &b.steps[i]);
        // what the reads wait for (read after write) and what the writes wait for (write after write / read)
        let mut dr: [Option<usize>; 4] = [None; 4];
        let mut dw: [Option<usize>; 4] = [None; 4];
        for &s in &r {
            if let Some(Some(p)) = last_w.get(s as usize) {
                if wlane(*p) != lr {
                    latest(&mut dr, wlane(*p), *p);
                }
            }
        }
        for &s in &w {
            if let Some(Some(p)) = last_w.get(s as usize) {
                if wlane(*p) != lw {
                    latest(&mut dw, wlane(*p), *p);
                }
            }
            if let Some(rr) = last_r.get(s as usize) {
                for (ol, x) in rr.iter().enumerate() {
                    if let Some(p) = *x {
                        if ol != lw && p != i {
                            latest(&mut dw, ol, p);
                        }
                    }
                }
            }
        }
        let set = |d: &[Option<usize>; 4], out: &mut [u32; 4], signal: &mut Vec<bool>, signal_in: &mut Vec<bool>| {
            for (pl, x) in d.iter().enumerate() {
                if let Some(p) = *x {
                    if out[pl] == NONE || p as u32 > out[pl] {
                        out[pl] = p as u32;
                    }
                    if pl == 2 {
                        signal_in[p] = true;
                    } else {
                        signal[p] = true;
                    }
                }
            }
        };
        if lane[i] == 2 {
            set(&dr, &mut wait[i], &mut signal, &mut signal_in);
            set(&dw, &mut wait_out[i], &mut signal, &mut signal_in);
        } else {
            set(&dr, &mut wait[i], &mut signal, &mut signal_in);
            set(&dw, &mut wait[i], &mut signal, &mut signal_in);
        }
        for &s in &r {
            if let Some(rr) = last_r.get_mut(s as usize) {
                rr[lr] = Some(i);
            }
        }
        for &s in &w {
            if (s as usize) < ns {
                last_w[s as usize] = Some(i);
                last_r[s as usize] = [None; 3];
            }
        }
    }
    let n_side = lane.iter().filter(|&&x| x == 1).count();
    let n_remote = lane.iter().filter(|&&x| x == 2).count();
    if let Ok(dir) = std::env::var("PC_SCHED_DUMP") {
        dump(&dir, b, &lane, &wait, &wait_out, &costs, &preds, &forced);
    }
    Sched { lane, wait, wait_out, signal, signal_in, n_side, n_remote, n_split, total_ms: costs.iter().sum(), critical_ms: total_cp, replica: None }
}

/// Moves whole refreshes onto the remote lane (2): single refresh steps outside units and refresh pairs whose route
/// family the remote device holds, most slack first, each kept there when `simulate3` says the block finishes earlier
/// (a few passes, until none moves). Finishes (when the remote device runs their split) and refresh batches it splits
/// hold the remote lane for their whole span.
fn remote_lanes<W: DevWord>(b: &Bound<'_, W>, rp: &RemotePlan<'_>, sim: &[f64], preds: &[Vec<usize>], dyns: &[bool], lane: &mut [u8]) {
    let n = lane.len();
    let unit = |i: usize| b.unit_of.get(i).filter(|&&u| u != NONE).map(|&u| &b.units[u as usize]);
    let mut costs = sim.to_vec();
    let mut joint = vec![false; n];
    let mut groups: Vec<Vec<usize>> = vec![];
    for (i, st) in b.steps.iter().enumerate() {
        match (&st.op, unit(i)) {
            (StepOp::Finish { .. }, _) => joint[i] = rp.finishes,
            (StepOp::Bootstrap { kind, .. }, None) if !dyns[i] && (rp.holds)(*kind) => groups.push(vec![i]),
            (StepOp::Bootstrap { .. }, Some(u)) => match &u.kind {
                UnitKind::Refresh { kind, .. } => joint[i] = u.hi - u.lo >= 2 && (rp.holds)(*kind),
                UnitKind::Pair { kind, .. } if i == u.lo as usize && (u.lo..u.hi).all(|j| !dyns[j as usize]) && (rp.holds)(*kind) => {
                    // one complex refresh: a bit more than a single, all of it on the first step
                    costs[i] *= 1.2;
                    costs[i + 1] = 0.0;
                    groups.push((u.lo as usize..u.hi as usize).collect());
                }
                _ => {}
            },
            _ => {}
        }
    }
    if groups.is_empty() {
        return;
    }
    // slack under unbounded lanes: latest minus earliest start (going backwards, `lst[i]` holds the smallest latest
    // start of i's successors, or the critical path, until i's turn subtracts its own cost)
    let mut est = vec![0.0f64; n];
    for i in 0..n {
        est[i] = preds[i].iter().map(|&p| est[p] + costs[p]).fold(0.0, f64::max);
    }
    let cp = (0..n).map(|i| est[i] + costs[i]).fold(0.0, f64::max);
    let mut lst = vec![cp; n];
    for i in (0..n).rev() {
        lst[i] -= costs[i];
        for &p in &preds[i] {
            if lst[i] < lst[p] {
                lst[p] = lst[i];
            }
        }
    }
    groups.sort_by(|a, b| {
        let s = |g: &Vec<usize>| g.iter().map(|&i| lst[i] - est[i]).fold(f64::INFINITY, f64::min);
        s(b).total_cmp(&s(a))
    });
    let mut best = simulate3(&costs, preds, dyns, lane, &joint);
    for _ in 0..REMOTE_PASSES {
        let mut moved = false;
        for g in &groups {
            if lane[g[0]] == 2 {
                continue;
            }
            let old: Vec<u8> = g.iter().map(|&i| lane[i]).collect();
            g.iter().for_each(|&i| lane[i] = 2);
            let t = simulate3(&costs, preds, dyns, lane, &joint);
            if t < best - 1e-6 {
                best = t;
                moved = true;
            } else {
                g.iter().zip(&old).for_each(|(&i, &l)| lane[i] = l);
            }
        }
        if !moved {
            break;
        }
    }
}

/// Three in-order lanes (0 compute, 1 side, 2 remote) issuing the steps in program order: as `simulate`, a remote step
/// paying its copies, a `joint` step (a finish or a split refresh batch) taking the remote lane with its own for its
/// whole span. Returns the finish time.
pub fn simulate3(costs: &[f64], preds: &[Vec<usize>], dynamic: &[bool], lane: &[u8], joint: &[bool]) -> f64 {
    let mut t = [0.0f64; 3];
    let mut fin = vec![0.0f64; costs.len()];
    for i in 0..costs.len() {
        if dynamic.get(i).copied().unwrap_or(false) {
            let m = t[0].max(t[1]).max(t[2]);
            t = [m, m, m];
        }
        let l = lane[i] as usize;
        let mut ready = 0.0f64;
        for &p in &preds[i] {
            let r = fin[p] + if lane[p] as usize != l { SYNC_MS } else { 0.0 };
            if r > ready {
                ready = r;
            }
        }
        fin[i] = if l == 2 {
            t[2].max(ready) + REMOTE_IN_MS + costs[i] + REMOTE_OUT_MS
        } else if joint[i] {
            let f = t[l].max(t[2]).max(ready) + costs[i];
            t[2] = f;
            f
        } else {
            t[l].max(ready) + costs[i]
        };
        t[l] = fin[i];
    }
    t[0].max(t[1]).max(t[2])
}

/// PC_SCHED_DUMP=<dir>: every planned block's steps (op kind, level, degree, lane, slots read and written, the step it
/// waits for per lane, cost, forced onto compute, data predecessors, unit) as `<dir>/<label>.jsonl`, for offline
/// analysis of the lane schedule.
#[allow(clippy::too_many_arguments)]
fn dump<W: DevWord>(dir: &str, b: &Bound<'_, W>, lane: &[u8], wait: &[[u32; 4]], wait_out: &[[u32; 4]], costs: &[f64], preds: &[Vec<usize>], forced: &[bool]) {
    let name: String = b.label.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect();
    let mut out = String::new();
    for (i, st) in b.steps.iter().enumerate() {
        let op = format!("{:?}", st.op);
        let kind: String = op.chars().take_while(|c| c.is_ascii_alphanumeric()).collect();
        let unit = b.unit_of.get(i).copied().filter(|&u| u != NONE).map_or(-1, |u| u as i64);
        let (r, w) = rw(b, st);
        let row = serde_json::json!({"i": i, "op": kind, "name": b.names.get(st.var as usize).cloned().unwrap_or_default(), "lvl": st.meta.level, "deg": st.meta.deg, "lane": lane[i], "r": r, "w": w,
            "wait": wait[i].iter().map(|&w| if w == NONE { -1 } else { w as i64 }).collect::<Vec<_>>(),
            "wait_out": wait_out[i].iter().map(|&w| if w == NONE { -1 } else { w as i64 }).collect::<Vec<_>>(), "cost": costs[i], "forced": forced[i], "preds": preds[i], "unit": unit,
            "dyn": b.dynamic.contains(&(i as u32))});
        out.push_str(&row.to_string());
        out.push('\n');
    }
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(format!("{dir}/{name}.jsonl"), out);
}

/// Device time of a step for the two-lane simulation (ms): pc-bench's K = 1 times at the top of n32a4, the key
/// switches and element-wise ops scaled by `live`, the share of the chain's primes live at the step's level.
fn sim_cost(st: &Step, lists: &[Vec<i64>], live: f64) -> f64 {
    match &st.op {
        StepOp::Bootstrap { .. } | StepOp::Finish { .. } => 10.0,
        StepOp::CoeffSum { .. } => 0.5,
        StepOp::Rotate { g, .. } if *g == NONE => 0.01,
        StepOp::Rotate { .. } | StepOp::Conjugate { .. } => 0.30 * live,
        StepOp::Mult | StepOp::Square | StepOp::MultRescale { .. } => 0.36 * live,
        StepOp::MultSum { .. } => 0.45 * live,
        StepOp::RotateSum { l, .. } => 0.25 * live * (lists.get(*l as usize).map_or(1, |v| v.len()) as f64 + 0.5),
        _ => 0.004 + 0.015 * live,
    }
}

/// A cross-lane wait (event record + wait), ms.
const SYNC_MS: f64 = 0.005;
/// Side producers `relieve` tries to move per block.
const RELIEF_TRIES: usize = 400;

/// Two in-order lanes (0 compute, 1 side) issuing the steps in program order: each step starts when its lane is free
/// and its data predecessors are done (`SYNC_MS` more from the other lane); a dynamic step joins the lanes. Returns
/// the finish time, and per step the time compute waited for a side-lane predecessor and that predecessor (`NONE`).
pub fn simulate(costs: &[f64], preds: &[Vec<usize>], dynamic: &[bool], lane: &[u8]) -> (f64, Vec<(f64, u32)>) {
    let mut t = [0.0f64; 2];
    let mut fin = vec![0.0f64; costs.len()];
    let mut waits = vec![(0.0, NONE); costs.len()];
    for i in 0..costs.len() {
        if dynamic.get(i).copied().unwrap_or(false) {
            let m = t[0].max(t[1]);
            t = [m, m];
        }
        let l = lane[i] as usize;
        let (mut ready, mut who) = (0.0f64, NONE);
        for &p in &preds[i] {
            let r = fin[p] + if lane[p] as usize != l { SYNC_MS } else { 0.0 };
            if r > ready {
                (ready, who) = (r, p as u32);
            }
        }
        if l == 0 && ready > t[0] && who != NONE && lane[who as usize] == 1 {
            waits[i] = (ready - t[0], who);
        }
        fin[i] = t[l].max(ready) + costs[i];
        t[l] = fin[i];
    }
    (t[0].max(t[1]), waits)
}

/// Moves side-lane chains that compute waits for onto compute while the two-lane `simulate` says the block finishes
/// earlier: the producer compute waits for longest, with its side-lane ancestors (and the whole of a unified span it
/// touches), tried once each, at most `tries` producers. Only side -> compute moves. Returns the steps moved.
pub fn relieve(costs: &[f64], preds: &[Vec<usize>], dynamic: &[bool], spans: &[(usize, usize, bool)], lane: &mut [u8], tries: usize) -> usize {
    let n = lane.len();
    let mut span_of = vec![usize::MAX; n];
    for (k, &(lo, hi, unify)) in spans.iter().enumerate() {
        if unify {
            span_of.get_mut(lo..hi.min(n)).into_iter().flatten().for_each(|x| *x = k);
        }
    }
    let mut tried = vec![false; n];
    let mut moved = 0;
    let (mut best, mut waits) = simulate(costs, preds, dynamic, lane);
    for _ in 0..tries {
        let Some((_, p)) = waits.iter().filter(|&&(w, p)| w > 1e-6 && p != NONE && !tried[p as usize]).copied().max_by(|a, b| a.0.total_cmp(&b.0)) else { break };
        tried[p as usize] = true;
        let mut chain = vec![];
        let mut todo = vec![p as usize];
        let mut seen = vec![false; n];
        while let Some(q) = todo.pop() {
            if seen[q] || lane[q] != 1 {
                continue;
            }
            seen[q] = true;
            chain.push(q);
            todo.extend(preds[q].iter().copied());
            if let Some(&(lo, hi, _)) = spans.get(span_of[q]) {
                todo.extend(lo..hi);
            }
        }
        chain.iter().for_each(|&q| lane[q] = 0);
        let (m, w) = simulate(costs, preds, dynamic, lane);
        if m < best - 1e-6 {
            (best, waits) = (m, w);
            moved += chain.len();
        } else {
            chain.iter().for_each(|&q| lane[q] = 1);
        }
    }
    moved
}

/// Lanes of the steps that a unit issues as one launch, over `spans` = (first step, end, unify): a unified unit (a
/// fused plaintext product and add) takes one lane, compute when any of its steps is there; an element-wise group
/// (affine, a product by i and its add) keeps its steps' own lanes, so it issues as one pass only where they agree
/// (the replay issues a group whose steps sit on two lanes step by step). Pulling a side step onto compute would make
/// compute wait there for the side lane's work. Returns the groups left on two lanes.
pub fn unit_lanes(lane: &mut [u8], spans: &[(usize, usize, bool)]) -> usize {
    let mut split = 0;
    for &(lo, hi, unify) in spans {
        let Some(s) = lane.get_mut(lo..hi) else { continue };
        let l = s.iter().copied().min().unwrap_or(0);
        if unify {
            s.iter_mut().for_each(|x| *x = l);
        } else if s.iter().any(|&x| x != l) {
            split += 1;
        }
    }
    split
}
