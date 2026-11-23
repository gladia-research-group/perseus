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

/// PC_SCHED2_FIN: `barrier` (default until an A/B wins) = a finish is a barrier (both devices joined, nothing
/// crosses it); `async` = a finish runs as the runtime runs it, its two halves each on its device when that device is
/// free (this device joining the remote half at its end), and independent ops may cross it.
fn fin_async() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| matches!(std::env::var("PC_SCHED2_FIN").as_deref().map(str::trim), Ok("async")))
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
        if join {
            let a = out.last_mut().expect("an atom is open");
            a.hi = i + 1;
            a.ms += op_ms(p, op);
        } else {
            wset.clear();
            rots.clear();
            out.push(Atom { lo: i, hi: i + 1, class: cls[i], ms: op_ms(p, op) });
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
    /// Peak live ciphertext bytes (GB) in program order and in the schedule (what the slot allocation's arena follows).
    pub live0_gb: f64,
    pub live_gb: f64,
    /// Over the regions between finishes (barrier mode): their scheduled spans, sum of max(critical path, work / 2),
    /// sum of work / 2, and the regions whose critical path exceeds half their work (count, their spans).
    pub spans: f64,
    pub bound: f64,
    pub half: f64,
    pub chain_regions: usize,
    pub chain_spans: f64,
    /// Device idle in the schedule (ms, both devices): waiting for a copy, for an input made on the same device, for
    /// the remote half of a finish or split refresh batch (this device joining it); and the copies' busy ms (both copy
    /// engines).
    pub idle: [f64; 3],
    pub copies_ms: f64,
    /// The copies' ms by what produced the copied var: block input, light op, heavy atom (refresh, K-batch), finish or
    /// split refresh batch.
    pub copies_by: [f64; 4],
}

/// Work (both devices' ms) and critical path (ms, RAW inside the range, unbounded devices) of atoms `lo..hi`.
fn bounds(at: &[Atom], preds: &[Vec<usize>], lo: usize, hi: usize) -> (f64, f64) {
    let mut fin = vec![0.0f64; hi - lo];
    let mut w = 0.0;
    for k in lo..hi {
        let c = if matches!(at[k].class, Class::Both | Class::Barrier) { at[k].ms / 2.0 } else { at[k].ms };
        w += at[k].ms;
        let st = preds[k].iter().filter(|&&q| q >= lo && q < hi).map(|&q| fin[q - lo]).fold(0.0, f64::max);
        fin[k - lo] = st + c;
    }
    (w, fin.into_iter().fold(0.0, f64::max))
}

/// PC_SCHED2_FILL: `1` / `on` = an atom that would leave its device waiting for its inputs first lets the ready atoms
/// that fit in the wait run there (inputs already on that device, most urgent first); unset / `0` = off (default
/// until an A/B wins).
fn fill() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| matches!(std::env::var("PC_SCHED2_FILL").as_deref().map(str::trim), Ok("1") | Ok("on")))
}

/// PC_SCHED2_MEM=<factor>: the schedule keeps the live ciphertext bytes within factor x the program order's peak
/// (a ready op that would pass it waits while one that fits, or frees the most, goes first); unset = no cap.
fn mem_factor() -> Option<f64> {
    static V: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("PC_SCHED2_MEM").ok().and_then(|v| v.trim().parse::<f64>().ok()).filter(|&f| f >= 1.0))
}

/// Live ciphertext bytes along a schedule: each var from its producer until its last reader (block inputs from the
/// start, block outputs to the end).
#[derive(Clone)]
struct Live {
    /// Readers (atoms) still to come, per var.
    rem: Vec<u32>,
    live: f64,
    peak: f64,
    cap: Option<f64>,
}

impl Live {
    /// Bytes atom `k` adds when it runs: its outputs read later or kept, less its inputs it is the last reader of.
    fn grow(&self, xin: &[VarId], out: &[VarId], keep: &HashSet<VarId>, vb: &dyn Fn(VarId) -> f64) -> f64 {
        out.iter().map(|&v| vb(v)).sum::<f64>() - xin.iter().filter(|&&v| self.rem[v as usize] == 1 && !keep.contains(&v)).map(|&v| vb(v)).sum::<f64>()
    }
    fn take(&mut self, xin: &[VarId], out: &[VarId], keep: &HashSet<VarId>, vb: &dyn Fn(VarId) -> f64) {
        self.live += out.iter().map(|&v| vb(v)).sum::<f64>();
        self.peak = self.peak.max(self.live);
        for &v in xin {
            let r = &mut self.rem[v as usize];
            *r = r.saturating_sub(1);
            if *r == 0 && !keep.contains(&v) {
                self.live -= vb(v);
            }
        }
    }
}

/// Ownership (PC_OWN, the replica lane's): a finish's remote outputs stay valid on the remote device too.
#[cfg(feature = "gpu")]
fn owned() -> bool {
    crate::replica::own()
}
#[cfg(not(feature = "gpu"))]
fn owned() -> bool {
    false
}

/// Where each var is (bit per device) and from when it is usable there, over the whole block: block inputs on this
/// device from the start, every output on the device that computed it, a copy adding the other device.
struct Loc {
    on: Vec<u8>,
    av: Vec<[f64; 2]>,
}

impl Loc {
    /// When the vars `vs` are usable on device `d`: those missing there copied over on `d`'s copy engine (`ce`, one
    /// copy at a time, each from when its var is ready on the other device). `commit` records the copies. Returns the
    /// time and the number of copies.
    fn arrive(&mut self, vs: &[VarId], d: usize, ce: &mut [f64; 2], cp: &dyn Fn(VarId) -> f64, commit: bool) -> (f64, usize) {
        let (mut x, mut c, mut nc) = (0.0f64, ce[d], 0usize);
        for &v in vs {
            let i = v as usize;
            if self.on[i] >> d & 1 != 0 {
                x = x.max(self.av[i][d]);
            } else {
                c = c.max(self.av[i][1 - d]) + cp(v);
                x = x.max(c);
                nc += 1;
                if commit {
                    self.on[i] |= 1 << d;
                    self.av[i][d] = c;
                }
            }
        }
        if commit {
            ce[d] = c;
        }
        (x, nc)
    }
    /// Outputs of a step on device `d` ending at `f`.
    fn put(&mut self, vs: &[VarId], d: usize, f: f64) {
        for &v in vs {
            self.on[v as usize] = 1 << d;
            self.av[v as usize] = [f, f];
        }
    }
    /// Outputs of a finish or a split refresh batch (both devices, inputs here): the first half here, the last half
    /// (the remote part's) on both devices with ownership (PC_OWN), else here.
    fn put_split(&mut self, vs: &[VarId], f: f64) {
        let h = vs.len() - vs.len() / 2;
        for (j, &v) in vs.iter().enumerate() {
            self.on[v as usize] = if j >= h && owned() { 3 } else { 1 };
            self.av[v as usize] = [f, f];
        }
    }
}

/// Reorder `p.ops` and set `p.var_dev` (every output var of an op the schedule puts on the remote device: 1).
/// `copy_ms(level)`: one ciphertext's copy between the devices (degree 1; a degree-2 one costs half again).
/// `var_bytes(level)`: one ciphertext's buffer (the slot allocation's size).
pub fn schedule(p: &mut Program, copy_ms: &dyn Fn(u32) -> f64, var_bytes: &dyn Fn(u32) -> f64) -> Report {
    let at = atoms(p);
    let na = at.len();
    let mut rep = Report { ops: p.ops.len(), atoms: na, ..Default::default() };
    if na == 0 {
        return rep;
    }
    // atom dependencies (RAW through vars; streamed-weight sums chained in order), and each atom's vars read from
    // outside it and written
    let mut prod: HashMap<VarId, usize> = HashMap::new();
    let mut preds: Vec<Vec<usize>> = vec![vec![]; na];
    let mut xin: Vec<Vec<VarId>> = vec![vec![]; na];
    let mut xout: Vec<Vec<VarId>> = vec![vec![]; na];
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
                    if !xin[k].contains(&v) {
                        xin[k].push(v);
                    }
                }
            }
            for v in op.all_outs() {
                own.insert(v);
                xout[k].push(v);
            }
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
    let vars = &p.vars;
    let vb = |v: VarId| var_bytes(vars[v as usize].meta.level);
    // live bytes: the readers of each var, the outputs anything reads later (or the block keeps)
    let keep: HashSet<VarId> = p.outputs.iter().copied().collect();
    let mut nread = vec![0u32; vars.len()];
    for x in &xin {
        for &v in x {
            nread[v as usize] += 1;
        }
    }
    let xlive: Vec<Vec<VarId>> = xout.iter().map(|o| o.iter().copied().filter(|v| nread[*v as usize] > 0 || keep.contains(v)).collect()).collect();
    let live_in: f64 = p.inputs.iter().filter(|v| nread[**v as usize] > 0 || keep.contains(*v)).map(|&v| vb(v)).sum();
    let mut po = Live { rem: nread.clone(), live: live_in, peak: live_in, cap: None };
    for k in 0..na {
        po.take(&xin[k], &xlive[k], &keep, &vb);
    }
    let mut live = Live { rem: nread, live: live_in, peak: live_in, cap: mem_factor().map(|f| f * po.peak) };
    rep.live0_gb = po.peak / 1e9;
    let cp = |v: VarId| {
        let m = &vars[v as usize].meta;
        copy_ms(m.level) * (m.deg.max(1) as f64 + 1.0) / 2.0
    };
    rep.one_ms = at.iter().map(|a| a.ms).sum();
    // regions between barriers; each barrier alone, both devices joined, its inputs here
    let mut loc = Loc { on: vec![1; p.vars.len()], av: vec![[0.0; 2]; p.vars.len()] };
    let mut ce = [0.0f64; 2];
    let mut order: Vec<usize> = Vec::with_capacity(na);
    let mut dev: Vec<u8> = vec![0; na];
    let mut t_all = 0.0f64;
    let mut lo = 0usize;
    let mut vcls = vec![0u8; p.vars.len()];
    for (k, a) in at.iter().enumerate() {
        let c = match a.class {
            Class::Light | Class::Pinned => 1,
            Class::Heavy => 2,
            Class::Both | Class::Barrier => 3,
        };
        for &v in &xout[k] {
            vcls[v as usize] = c;
        }
    }
    while lo < na {
        if at[lo].class == Class::Barrier && !fin_async() {
            let (x, _) = loc.arrive(&xin[lo], 0, &mut ce, &cp, true);
            let f = t_all.max(x) + at[lo].ms / 2.0;
            loc.put_split(&xout[lo], f);
            live.take(&xin[lo], &xlive[lo], &keep, &vb);
            order.push(lo);
            t_all = f;
            lo += 1;
            continue;
        }
        let mut hi = lo;
        while hi < na && (at[hi].class != Class::Barrier || fin_async()) {
            hi += 1;
        }
        // best of a few copy penalties (ms charged per copy the choice creates; larger ones, 5 and 10 ms, scored better
        // in the model but ran slower: ViT 11.76 vs 11.54 s)
        type Run = (f64, Vec<usize>, Vec<u8>, Loc, [f64; 2], Live, [f64; 8], f64);
        let run = |betas: &[f64]| -> Run {
            let mut best: Option<Run> = None;
            for &beta in betas {
                let mut l = Loc { on: loc.on.clone(), av: loc.av.clone() };
                let mut c = ce;
                let mut m = live.clone();
                let mut sts = [0.0f64; 8];
                let g = Graph { at: &at, preds: &preds, xin: &xin, xout: &xout, xlive: &xlive, keep: &keep, vcls: &vcls };
                let (mk, ord, dv) = region(&g, lo, hi, t_all, beta, &mut l, &mut c, &mut m, &cp, &vb, &mut sts);
                if best.as_ref().is_none_or(|b| mk < b.0) {
                    best = Some((mk, ord, dv, l, c, m, sts, beta));
                }
            }
            best.expect("one schedule at least")
        };
        let res = run(&[0.0, 0.5, 2.0]);
        let (mk, ord, dv, l, c, m, sts, _) = res;
        for i in 0..3 {
            rep.idle[i] += sts[i];
        }
        for i in 0..4 {
            rep.copies_by[i] += sts[4 + i];
            rep.copies_ms += sts[4 + i];
        }
        let (w, cp) = bounds(&at, &preds, lo, hi);
        rep.spans += mk - t_all;
        rep.bound += cp.max(w / 2.0);
        rep.half += w / 2.0;
        if cp > w / 2.0 {
            rep.chain_regions += 1;
            rep.chain_spans += mk - t_all;
        }
        t_all = mk;
        loc = l;
        ce = c;
        live = m;
        for (j, &k) in ord.iter().enumerate() {
            order.push(k);
            dev[k] = dv[j];
        }
        lo = hi;
    }
    // the block's outputs back here
    let outs = p.outputs.clone();
    let (x, _) = loc.arrive(&outs, 0, &mut ce, &cp, true);
    rep.two_ms = t_all.max(x);
    rep.live_gb = live.peak / 1e9;
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

/// List schedule of atoms `lo..hi` (no barrier inside) from `t0` (both devices free) on two in-order devices with one
/// copy engine into each, every var read where it is not copied over (`loc`, `ce`: updated): (makespan, atoms in
/// start order, their devices).
/// The block's atoms and their dataflow.
struct Graph<'g> {
    at: &'g [Atom],
    preds: &'g [Vec<usize>],
    /// Vars each atom reads from outside it, writes, and writes that something reads later (or the block keeps).
    xin: &'g [Vec<VarId>],
    xout: &'g [Vec<VarId>],
    xlive: &'g [Vec<VarId>],
    keep: &'g HashSet<VarId>,
    /// Per var: what produced it (0 block input, 1 light, 2 heavy, 3 finish / split refresh batch).
    vcls: &'g [u8],
}

#[allow(clippy::too_many_arguments)]
fn region(g: &Graph<'_>, lo: usize, hi: usize, t0: f64, beta: f64, loc: &mut Loc, ce: &mut [f64; 2], live: &mut Live, cp: &dyn Fn(VarId) -> f64, vb: &dyn Fn(VarId) -> f64, st: &mut [f64; 8]) -> (f64, Vec<usize>, Vec<u8>) {
    let (at, preds, xin, xout) = (g.at, g.preds, g.xin, g.xout);
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
    let cost = |k: usize| if matches!(at[lo + k].class, Class::Both | Class::Barrier) { at[lo + k].ms / 2.0 } else { at[lo + k].ms };
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
    let mut t = [t0; 2];
    let mut start = vec![0.0f64; n];
    let mut dv = vec![0u8; n];
    while !heap.is_empty() {
        // the most urgent ready atom; under a memory cap (PC_SCHED2_MEM) the most urgent that fits, else of all the
        // ready atoms the one that grows the live bytes least (the most urgent of those)
        let k = match live.cap {
            None => heap.pop().expect("not empty").1,
            Some(cap) => {
                let mut tried: Vec<(f64, R)> = vec![];
                let mut pick = None;
                while let Some(r) = heap.pop() {
                    let gr = live.grow(&xin[lo + r.1], &g.xlive[lo + r.1], g.keep, vb);
                    if live.live + gr <= cap {
                        pick = Some(r.1);
                        break;
                    }
                    tried.push((gr, r));
                }
                let k = match pick {
                    Some(k) => k,
                    None => {
                        // `tried` is in urgency order: the first of the least growing
                        let j = (0..tried.len()).min_by(|&a, &b| tried[a].0.total_cmp(&tried[b].0).then(a.cmp(&b))).expect("one tried");
                        tried.swap_remove(j).1 .1
                    }
                };
                for (_, r) in tried {
                    heap.push(r);
                }
                k
            }
        };
        let a = &at[lo + k];
        match a.class {
            Class::Both if !fin_async() => {
                // a split refresh batch: its inputs here, both devices for its span
                let (x, _) = loc.arrive(&xin[lo + k], 0, ce, cp, true);
                let st = t[0].max(t[1]).max(x);
                let f = st + cost(k);
                t = [f, f];
                start[k] = st;
                loc.put_split(&xout[lo + k], f);
                dv[k] = 0;
            }
            Class::Both | Class::Barrier => {
                // a split refresh batch or a finish (PC_SCHED2_FIN=async): its inputs here, each half on its device
                // when that device is free (the remote half reads its inputs over the link itself), this device
                // joining the remote half at the end; issued after both devices' earlier ops
                for &v in &xin[lo + k] {
                    if loc.on[v as usize] & 1 == 0 {
                        st[4 + g.vcls[v as usize] as usize] += cp(v);
                    }
                }
                let (x, _) = loc.arrive(&xin[lo + k], 0, ce, cp, true);
                let (s0, s1) = (t[0].max(x), t[1].max(x));
                let (f0, f1) = (s0 + cost(k), s1 + cost(k));
                let end = f0.max(f1);
                st[1] += (s0 - t[0]) + (s1 - t[1]);
                st[2] += end - f0;
                t = [end, f1];
                start[k] = s0.max(s1);
                loc.put_split(&xout[lo + k], end);
                dv[k] = 0;
            }
            _ => {
                let mut best: Option<(f64, f64, usize)> = None;
                for d in 0..2 {
                    if d == 1 && a.class == Class::Pinned {
                        continue;
                    }
                    let mut c = *ce;
                    let (x, nc) = loc.arrive(&xin[lo + k], d, &mut c, cp, false);
                    let f = t[d].max(x) + cost(k);
                    let score = f + beta * nc as f64;
                    if best.is_none_or(|b| score < b.0) {
                        best = Some((score, f, d));
                    }
                }
                let (_, f, d) = best.expect("device 0 always");
                // PC_SCHED2_FILL: the wait before k on d filled with ready atoms whose inputs are on d already
                let s_k = f - cost(k);
                while fill() && s_k > t[d] {
                    let mut seen: Vec<R> = vec![];
                    let mut found = None;
                    while let Some(r) = heap.pop() {
                        let j = r.1;
                        let cls = at[lo + j].class;
                        if matches!(cls, Class::Light | Class::Heavy) || (cls == Class::Pinned && d == 0) {
                            let xs = &xin[lo + j];
                            if xs.iter().all(|&v| loc.on[v as usize] >> d & 1 != 0) {
                                let st = xs.iter().map(|&v| loc.av[v as usize][d]).fold(t[d], f64::max);
                                if st + cost(j) <= s_k && live.cap.is_none_or(|c| live.live + live.grow(xs, &g.xlive[lo + j], g.keep, vb) <= c) {
                                    found = Some((j, st));
                                    break;
                                }
                            }
                        }
                        seen.push(r);
                        if seen.len() >= 256 {
                            break;
                        }
                    }
                    for r in seen {
                        heap.push(r);
                    }
                    let Some((j, st)) = found else { break };
                    let fj = st + cost(j);
                    start[j] = st;
                    t[d] = fj;
                    loc.put(&xout[lo + j], d, fj);
                    dv[j] = d as u8;
                    live.take(&xin[lo + j], &g.xlive[lo + j], g.keep, vb);
                    for &s2 in &succ[j] {
                        npred[s2] -= 1;
                        if npred[s2] == 0 {
                            heap.push(R(tail[s2], s2));
                        }
                    }
                }
                // idle before k on d: waiting for inputs made on d (dep), then for copies
                let xl = xin[lo + k].iter().filter(|&&v| loc.on[v as usize] >> d & 1 != 0).map(|&v| loc.av[v as usize][d]).fold(t[d], f64::max);
                for &v in &xin[lo + k] {
                    if loc.on[v as usize] >> d & 1 == 0 {
                        st[4 + g.vcls[v as usize] as usize] += cp(v);
                    }
                }
                loc.arrive(&xin[lo + k], d, ce, cp, true);
                start[k] = f - cost(k);
                st[1] += xl.min(start[k]) - t[d];
                st[0] += (start[k] - xl).max(0.0);
                t[d] = f;
                loc.put(&xout[lo + k], d, f);
                dv[k] = d as u8;
            }
        }
        live.take(&xin[lo + k], &g.xlive[lo + k], g.keep, vb);
        for &s in &succ[k] {
            npred[s] -= 1;
            if npred[s] == 0 {
                heap.push(R(tail[s], s));
            }
        }
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| start[a].total_cmp(&start[b]).then(a.cmp(&b)));
    (t[0].max(t[1]), idx.iter().map(|&k| lo + k).collect(), idx.iter().map(|&k| dv[k]).collect())
}
