//! Two-device list schedule (PC_SCHED2=1, with a remote device's replica lane): the block's ops reordered so that
//! two in-order devices overlap, and each op's device chosen with it (`Program::var_dev`, the replica planner's
//! PC_REPLICA_PLACE=hint). Program order makes the replica wait: a ready refresh or head of one device sits behind
//! work the other device issues first, and a value it needs crosses only when its consumer comes up. A list
//! schedule (ready atoms by longest tail first, each on the device that finishes it earliest, copies arriving on
//! a copy engine while both devices compute) interleaves independent chains so both devices stay busy; the slot
//! allocation that follows gives the new order its own buffers (more of them live at once).
//!
//! What moves is an ATOM, not an op: a run of consecutive light ops linked by data (each reads the run's outputs,
//! or the next light op reads both: a plaintext product and the add that consumes it), and the rotations of one
//! input set (a grid or chain shares one ModUp), so bind's fused units (affine groups, product-adds, rotation grids
//! and accumulations) keep their ops adjacent. Refreshes, batches and pinned ops are atoms of their own. Finishes are
//! barriers (nothing crosses them). Streamed-weight sums stay on this device in their order. Same ops, same vars, same
//! metadata: only the order and the device hints change (pure dataflow: every op depends on its inputs only).
use super::batch::{paired, PAIR_K};
use crate::program::{Kind, Op, Program, VarId};
use std::collections::{BinaryHeap, HashMap, HashSet};

/// PC_SCHED2: `1` / `on` = on; unset / `0` / `off` = off (default until an A/B wins).
pub fn on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| matches!(std::env::var("PC_SCHED2").as_deref().map(str::trim), Ok("1") | Ok("on")))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    /// Element-wise, products, rotations: moves, joins its neighbours into atoms.
    Light,
    /// A refresh (single or pair) or a K-batch of products / rotations: moves alone.
    Heavy,
    /// A streamed-weight sum: this device, in order.
    Pinned,
    /// A refresh batch the remote lane splits (K >= 2, not a pair): both devices at once.
    Both,
    /// A finish (PCMM / CCMM): nothing crosses it.
    Barrier,
}

/// Device ms of one op (n32; replica.rs `cost`'s table: the Llama prefill profile's averages per kind).
fn op_ms(p: &Program, op: &Op) -> f64 {
    match &op.kind {
        Kind::Batched(m) => {
            if m.len() == PAIR_K && m.iter().all(|x| paired(p, x, p.batch_report.pair)) {
                10.0
            } else {
                m.iter().map(|x| op_ms(p, x)).sum()
            }
        }
        Kind::Bootstrap(s) => {
            let r = &p.bts_sites[*s as usize].route;
            if r.slots == 0 && !r.stc_first {
                11.9
            } else {
                7.5
            }
        }
        Kind::Finish { .. } => 11.7 * op.outs.len().max(1) as f64,
        Kind::CoeffSum { .. } => 0.5,
        Kind::Rotate(0) => 0.005,
        Kind::Rotate(_) | Kind::Conjugate => 0.058,
        Kind::Mult | Kind::Square | Kind::MultRescale { .. } => 0.09,
        Kind::MultSum { .. } => 0.12,
        Kind::RotateSum(v) => 0.06 * (v.len() as f64 + 1.0),
        Kind::Rescale => 0.029,
        Kind::MultPt(_) | Kind::AddPt(_) => 0.011,
        _ => 0.008,
    }
}

fn class(p: &Program, op: &Op) -> Class {
    match &op.kind {
        Kind::Finish { .. } => Class::Barrier,
        Kind::CoeffSum { .. } => Class::Pinned,
        Kind::Batched(m) if m.iter().all(|x| matches!(x.kind, Kind::Bootstrap(_))) => {
            if m.len() == PAIR_K && m.iter().all(|x| paired(p, x, p.batch_report.pair)) {
                Class::Heavy
            } else {
                Class::Both
            }
        }
        Kind::Batched(_) | Kind::Bootstrap(_) => Class::Heavy,
        _ => Class::Light,
    }
}

/// A single rotation's input (grid / chain key), None for anything else.
fn rot_in(op: &Op) -> Option<VarId> {
    match op.kind {
        Kind::Rotate(k) if k != 0 && op.ins.len() == 1 => Some(op.ins[0]),
        Kind::Conjugate if op.ins.len() == 1 => Some(op.ins[0]),
        _ => None,
    }
}

struct Atom {
    lo: usize,
    hi: usize,
    class: Class,
    ms: f64,
    /// Level of the last op's first output (a copy's size).
    level: u32,
}

fn atoms(p: &Program) -> Vec<Atom> {
    let ops = &p.ops;
    let n = ops.len();
    let cls: Vec<Class> = ops.iter().map(|o| class(p, o)).collect();
    let mut out: Vec<Atom> = vec![];
    let mut wset: HashSet<VarId> = HashSet::new();
    let mut rots: HashSet<VarId> = HashSet::new();
    for i in 0..n {
        let op = &ops[i];
        let ins = op.all_ins();
        let join = match out.last() {
            Some(a) if a.class == Class::Light && cls[i] == Class::Light => {
                ins.iter().any(|v| wset.contains(v))
                    || (i + 1 < n && cls[i + 1] == Class::Light && {
                        let outs = op.all_outs();
                        let nx = ops[i + 1].all_ins();
                        nx.iter().any(|v| outs.contains(v)) && nx.iter().any(|v| wset.contains(v))
                    })
                    || rot_in(op).is_some_and(|v| rots.contains(&v))
            }
            _ => false,
        };
        let level = op.all_outs().first().map_or(0, |&v| p.vars[v as usize].meta.level);
        if join {
            let a = out.last_mut().expect("an atom is open");
            a.hi = i + 1;
            a.ms += op_ms(p, op);
            a.level = level;
        } else {
            wset.clear();
            rots.clear();
            out.push(Atom { lo: i, hi: i + 1, class: cls[i], ms: op_ms(p, op), level });
        }
        wset.extend(op.all_outs());
        if let Some(v) = rot_in(op) {
            rots.insert(v);
        }
    }
    out
}

/// What the pass did: atoms, the static model's one-device ms (every atom in order), program order on two devices
/// with the hints, and the schedule.
#[derive(Default, Debug)]
pub struct Report {
    pub ops: usize,
    pub atoms: usize,
    pub moved: usize,
    pub one_ms: f64,
    pub two_ms: f64,
}

/// Reorder `p.ops` and set `p.var_dev` (every output var of an op the schedule puts on the remote device: 1).
/// `copy_ms(level)`: one ciphertext's copy between the devices.
pub fn schedule(p: &mut Program, copy_ms: &dyn Fn(u32) -> f64) -> Report {
    let at = atoms(p);
    let na = at.len();
    let mut rep = Report { ops: p.ops.len(), atoms: na, ..Default::default() };
    if na == 0 {
        return rep;
    }
    // atom dependencies (RAW through vars; streamed-weight sums chained in order)
    let mut prod: HashMap<VarId, usize> = HashMap::new();
    let mut preds: Vec<Vec<usize>> = vec![vec![]; na];
    let mut last_pin: Option<usize> = None;
    for (k, a) in at.iter().enumerate() {
        let mut d: Vec<usize> = vec![];
        let mut own: HashSet<VarId> = HashSet::new();
        for op in &p.ops[a.lo..a.hi] {
            for v in op.all_ins() {
                if !own.contains(&v) {
                    if let Some(&q) = prod.get(&v) {
                        d.push(q);
                    }
                }
            }
            own.extend(op.all_outs());
        }
        if a.class == Class::Pinned {
            if let Some(q) = last_pin {
                d.push(q);
            }
            last_pin = Some(k);
        }
        d.sort_unstable();
        d.dedup();
        preds[k] = d;
        for op in &p.ops[a.lo..a.hi] {
            for v in op.all_outs() {
                prod.insert(v, k);
            }
        }
    }
    rep.one_ms = at.iter().map(|a| a.ms).sum();
    // regions between barriers; each barrier alone, both devices joined
    let mut order: Vec<usize> = Vec::with_capacity(na);
    let mut dev: Vec<u8> = vec![0; na];
    let mut t_all = 0.0f64;
    let mut lo = 0usize;
    while lo < na {
        if at[lo].class == Class::Barrier {
            order.push(lo);
            t_all += at[lo].ms / 2.0;
            lo += 1;
            continue;
        }
        let mut hi = lo;
        while hi < na && at[hi].class != Class::Barrier {
            hi += 1;
        }
        // best of a few copy penalties (ms charged per copy the choice creates)
        let mut best: Option<(f64, Vec<usize>, Vec<u8>)> = None;
        for beta in [0.0, 0.5, 2.0] {
            let (mk, ord, dv) = region(&at, &preds, lo, hi, beta, copy_ms);
            if best.as_ref().is_none_or(|b| mk < b.0) {
                best = Some((mk, ord, dv));
            }
        }
        let (mk, ord, dv) = best.expect("one schedule at least");
        t_all += mk;
        for (j, &k) in ord.iter().enumerate() {
            order.push(k);
            dev[k] = dv[j];
        }
        lo = hi;
    }
    rep.two_ms = t_all;
    // the new op order and the device hints
    let mut ops: Vec<Option<Op>> = std::mem::take(&mut p.ops).into_iter().map(Some).collect();
    let mut out: Vec<Op> = Vec::with_capacity(ops.len());
    p.var_dev = vec![0; p.vars.len()];
    for &k in &order {
        for slot in ops.iter_mut().take(at[k].hi).skip(at[k].lo) {
            let op = slot.take().expect("each op once");
            if dev[k] == 1 {
                rep.moved += 1;
                for v in op.all_outs() {
                    p.var_dev[v as usize] = 1;
                }
            }
            out.push(op);
        }
    }
    p.ops = out;
    rep
}

/// List schedule of atoms `lo..hi` (no barrier inside) on two in-order devices with one copy engine into each:
/// (makespan from the region's start, atoms in start order, their devices).
fn region(at: &[Atom], preds: &[Vec<usize>], lo: usize, hi: usize, beta: f64, copy_ms: &dyn Fn(u32) -> f64) -> (f64, Vec<usize>, Vec<u8>) {
    let n = hi - lo;
    let mut succ: Vec<Vec<usize>> = vec![vec![]; n];
    let mut npred = vec![0usize; n];
    for k in lo..hi {
        for &q in &preds[k] {
            if q >= lo && q < hi {
                succ[q - lo].push(k - lo);
                npred[k - lo] += 1;
            }
        }
    }
    let cost = |k: usize| if at[lo + k].class == Class::Both { at[lo + k].ms / 2.0 } else { at[lo + k].ms };
    let mut tail = vec![0.0f64; n];
    for k in (0..n).rev() {
        tail[k] = cost(k) + succ[k].iter().map(|&s| tail[s]).fold(0.0, f64::max);
    }
    // ready heap by tail (longest first), ties by program order
    #[derive(PartialEq)]
    struct R(f64, usize);
    impl Eq for R {}
    impl PartialOrd for R {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for R {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            self.0.total_cmp(&o.0).then(o.1.cmp(&self.1))
        }
    }
    let mut heap: BinaryHeap<R> = (0..n).filter(|&k| npred[k] == 0).map(|k| R(tail[k], k)).collect();
    let mut t = [0.0f64; 2];
    let mut ce = [0.0f64; 2];
    let mut fin = vec![0.0f64; n];
    // where each atom's outputs are (bit per device) and when they are usable there
    let mut on = vec![0u8; n];
    let mut avail = vec![[0.0f64; 2]; n];
    let mut start = vec![0.0f64; n];
    let mut dv = vec![0u8; n];
    while let Some(R(_, k)) = heap.pop() {
        let a = &at[lo + k];
        // arrival of the inputs on d (copies queued on d's engine; commit: record them)
        let arr = |d: usize, on: &mut [u8], avail: &mut [[f64; 2]], ce: &mut [f64; 2], commit: bool| -> (f64, usize) {
            let mut x = 0.0f64;
            let mut c = ce[d];
            let mut nc = 0;
            for &q in &preds[lo + k] {
                if q < lo || q >= hi {
                    continue;
                }
                let q = q - lo;
                if on[q] >> d & 1 != 0 {
                    x = x.max(avail[q][d]);
                } else {
                    let s0 = fin[q].max(c);
                    c = s0 + copy_ms(at[lo + q].level);
                    x = x.max(c);
                    nc += 1;
                    if commit {
                        on[q] |= 1 << d;
                        avail[q][d] = c;
                    }
                }
            }
            if commit {
                ce[d] = c;
            }
            (x, nc)
        };
        match a.class {
            Class::Both => {
                let (x0, _) = arr(0, &mut on, &mut avail, &mut ce, true);
                let st = t[0].max(t[1]).max(x0);
                let f = st + cost(k);
                t = [f, f];
                fin[k] = f;
                start[k] = st;
                on[k] = 3;
                avail[k] = [f, f];
                dv[k] = 0;
            }
            _ => {
                let mut opts: Vec<(f64, f64, usize)> = vec![];
                for d in 0..2 {
                    if d == 1 && a.class == Class::Pinned {
                        continue;
                    }
                    let (x, nc) = arr(d, &mut on.clone(), &mut avail.clone(), &mut ce.clone(), false);
                    let f = t[d].max(x) + cost(k);
                    opts.push((f + beta * nc as f64, f, d));
                }
                let &(_, f, d) = opts.iter().min_by(|a, b| a.0.total_cmp(&b.0).then(a.2.cmp(&b.2))).expect("device 0 always");
                arr(d, &mut on, &mut avail, &mut ce, true);
                start[k] = f - cost(k);
                t[d] = f;
                fin[k] = f;
                on[k] = 1 << d;
                avail[k] = [0.0, 0.0];
                avail[k][d] = f;
                dv[k] = d as u8;
            }
        }
        for &s in &succ[k] {
            npred[s] -= 1;
            if npred[s] == 0 {
                heap.push(R(tail[s], s));
            }
        }
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| start[a].total_cmp(&start[b]).then(a.cmp(&b)));
    let mk = t[0].max(t[1]);
    (mk, idx.iter().map(|&k| lo + k).collect(), idx.iter().map(|&k| dv[k]).collect())
}
