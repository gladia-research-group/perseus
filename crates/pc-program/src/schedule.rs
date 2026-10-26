//! `schedule`: which lane each step of a bound block runs on (1 compute + side lanes, events at joins). The
//! compiler knows the DAG: steps off the block's critical path (under a static cost model: refreshes and key
//! switches dominate) go to the side lane, everything on it stays on compute. Refreshes, finishes,
//! streamed-weight sums, units (grids, K-batches), dynamic steps and steps using the block's scratch slot always run on compute
//! (their workspaces are compute's).
//!
//! Correctness does not depend on the heuristic: every slot hazard between the lanes (read after write, write after
//! read, write after write) becomes one event wait on the latest conflicting step of the other lane (stream order
//! covers the earlier ones). Hazards are tracked per region (the steps between two dynamic steps): the replay forks
//! the side lane from compute at a region's first side step and joins it back at the region's end, so a captured
//! segment's graph holds the fork / join and never waits on an event of another capture.
use crate::bind::{Bound, Step, StepOp, NONE};
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
    /// Static cost estimates (ms): every step, the critical path.
    pub total_ms: f64,
    pub critical_ms: f64,
}

/// Rough device cost of a step (ms, K = 1, n32 decode levels): only the ordering between steps matters.
fn cost(st: &Step, lists: &[Vec<i64>]) -> f64 {
    match &st.op {
        StepOp::Bootstrap { .. } | StepOp::Finish { .. } => 17.0,
        StepOp::CoeffSum { .. } => 1.0,
        StepOp::Mult | StepOp::Square => 0.45,
        StepOp::Rotate { g, .. } if *g == NONE => 0.02,
        StepOp::Rotate { .. } | StepOp::Conjugate { .. } => 0.4,
        StepOp::RotateSum { l, .. } => 0.15 + 0.3 * lists.get(*l as usize).map_or(1, |v| v.len()) as f64,
        _ => 0.03,
    }
}

/// Slots a step reads and writes (finishes: their own lists).
fn rw<W: DevWord>(b: &Bound<'_, W>, st: &Step) -> (Vec<u32>, Vec<u32>) {
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
    matches!(st.op, StepOp::Bootstrap { .. } | StepOp::Finish { .. } | StepOp::CoeffSum { .. }) || tmp || !side_ok(st)
}

/// The schedule of `b` (its current dynamic steps) for one compute + one side lane.
pub fn plan<W: DevWord>(b: &Bound<'_, W>, side_ok: &dyn Fn(&Step) -> bool) -> Sched {
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
    let lane: Vec<u8> = (0..n)
        .map(|i| {
            let st = &b.steps[i];
            let on_cp = fin[i] + tail[i] >= total_cp - 1e-6;
            // a unit issues all its steps at once, on compute (its gather buffer is compute's)
            let unit = b.unit_of.get(i).is_some_and(|&u| u != NONE);
            if on_cp || unit || dynamic.contains(&(i as u32)) || pinned(st, side_ok) {
                0
            } else {
                1
            }
        })
        .collect();
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
    Sched { lane, wait, signal, n_side, total_ms: costs.iter().sum(), critical_ms: total_cp }
}
