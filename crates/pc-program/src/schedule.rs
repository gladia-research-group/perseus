//! `schedule`: which lane each step of a bound block runs on (1 compute + side lanes, events at joins). The
//! compiler knows the DAG: steps off the block's critical path (under a static cost model: refreshes and key
//! switches dominate) go to the side lane, everything on it stays on compute. Refreshes, finishes,
//! streamed-weight sums, units (grids, K-batches), dynamic steps and steps using the block's scratch slot always run on compute
//! (their workspaces are compute's); a fused plaintext product and add (no gather buffer) may run on the side lane, and
//! an element-wise group runs as one pass only when its steps share a lane.
//!
//! Correctness does not depend on the heuristic: every slot hazard between the lanes (read after write, write after
//! read, write after write) becomes one event wait on the latest conflicting step of the other lane (stream order
//! covers the earlier ones). Hazards are tracked per region (the steps between two dynamic steps): the replay forks
//! the side lane from compute at a region's first side step and joins it back at the region's end, so a captured
//! segment's graph holds the fork / join and never waits on an event of another capture.
use crate::bind::{Bound, Step, StepOp, UnitKind, NONE};
use pc_gpu::DevWord;
use std::collections::BTreeSet;

/// Lane of every step and the cross-lane waits.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sched {
    /// 0 = compute, 1 = side.
    pub lane: Vec<u8>,
    /// Step whose completion (event) this step waits for on the other lane, or NONE.
    pub wait: Vec<u32>,
    /// Record an event after this step (some later step of the other lane waits for it).
    pub signal: Vec<bool>,
    pub n_side: usize,
    /// Element-wise groups whose steps sit on two lanes (issued step by step).
    pub n_split: usize,
    /// Static cost estimates (ms): every step, the critical path.
    pub total_ms: f64,
    pub critical_ms: f64,
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

/// The schedule of `b` (its current dynamic steps) for one compute + one side lane.
pub fn plan<W: DevWord>(b: &Bound<'_, W>, side_ok: &dyn Fn(&Step) -> bool, live: &dyn Fn(u32) -> f64) -> Sched {
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
    // hazards per region: last writer and the latest reader per lane of every slot since its last write
    let mut wait = vec![NONE; n];
    let mut signal = vec![false; n];
    let ns = b.n_slots.max(1) + 1;
    let mut last_w: Vec<Option<usize>> = vec![None; ns];
    let mut last_r: Vec<[Option<usize>; 2]> = vec![[None, None]; ns];
    for i in 0..n {
        if dynamic.contains(&(i as u32)) {
            // a region boundary: the side lane joined compute before it and forks after it
            last_w.iter_mut().for_each(|x| *x = None);
            last_r.iter_mut().for_each(|x| *x = [None, None]);
            continue;
        }
        let l = lane[i] as usize;
        let (r, w) = rw(b, &b.steps[i]);
        let mut dep: Option<usize> = None;
        let mut need = |p: Option<usize>| {
            if let Some(p) = p {
                if lane[p] as usize != l && dep.map_or(true, |d| p > d) {
                    dep = Some(p);
                }
            }
        };
        for &s in &r {
            if let Some(x) = last_w.get(s as usize) {
                need(*x);
            }
        }
        for &s in &w {
            if let Some(x) = last_w.get(s as usize) {
                need(*x);
            }
            if let Some(rr) = last_r.get(s as usize) {
                need(rr[1 - l]);
            }
        }
        if let Some(d) = dep {
            wait[i] = d as u32;
            signal[d] = true;
        }
        for &s in &r {
            if let Some(rr) = last_r.get_mut(s as usize) {
                rr[l] = Some(i);
            }
        }
        for &s in &w {
            if (s as usize) < ns {
                last_w[s as usize] = Some(i);
                last_r[s as usize] = [None, None];
            }
        }
    }
    let n_side = lane.iter().filter(|&&x| x == 1).count();
    if let Ok(dir) = std::env::var("PC_SCHED_DUMP") {
        dump(&dir, b, &lane, &wait, &costs, &preds, &forced);
    }
    Sched { lane, wait, signal, n_side, n_split, total_ms: costs.iter().sum(), critical_ms: total_cp }
}

/// PC_SCHED_DUMP=<dir>: every planned block's steps (op kind, level, degree, lane, the step it waits for, cost, forced
/// onto compute, data predecessors, unit) as `<dir>/<label>.jsonl`, for offline analysis of the lane schedule.
#[allow(clippy::too_many_arguments)]
fn dump<W: DevWord>(dir: &str, b: &Bound<'_, W>, lane: &[u8], wait: &[u32], costs: &[f64], preds: &[Vec<usize>], forced: &[bool]) {
    let name: String = b.label.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect();
    let mut out = String::new();
    for (i, st) in b.steps.iter().enumerate() {
        let op = format!("{:?}", st.op);
        let kind: String = op.chars().take_while(|c| c.is_ascii_alphanumeric()).collect();
        let unit = b.unit_of.get(i).copied().filter(|&u| u != NONE).map_or(-1, |u| u as i64);
        let row = serde_json::json!({"i": i, "op": kind, "lvl": st.meta.level, "deg": st.meta.deg, "lane": lane[i],
            "wait": if wait[i] == NONE { -1 } else { wait[i] as i64 }, "cost": costs[i], "forced": forced[i], "preds": preds[i], "unit": unit,
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
