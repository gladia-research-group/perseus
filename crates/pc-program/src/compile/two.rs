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

/// The copy model (PC_SCHED2_COPY_LAT=<ms per copy>, default 0; PC_SCHED2_COPY_GBS=<GB/s>, default 30): one
/// ciphertext copied between the devices costs lat + its bytes / bandwidth, in the compile schedule and the replica's
/// placement simulation.
pub fn copy_model() -> (f64, f64) {
    static V: std::sync::OnceLock<(f64, f64)> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let f = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.trim().parse::<f64>().ok()).unwrap_or(d);
        (f("PC_SCHED2_COPY_LAT", 0.0), f("PC_SCHED2_COPY_GBS", 30.0).max(1.0))
    })
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
    /// A refresh (single, pair or batch).
    bts: bool,
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
            let bts = matches!(&op.kind, Kind::Bootstrap(_)) || matches!(&op.kind, Kind::Batched(m) if m.iter().all(|x| matches!(x.kind, Kind::Bootstrap(_))));
            out.push(Atom { lo: i, hi: i + 1, class: cls[i], ms: op_ms(p, op), bts });
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
    /// Values copied between the devices in the schedule.
    pub n_copies: usize,
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

/// Ownership (PC_OWN, the replica lane's `replica::own`): a finish's remote outputs stay valid on the remote device
/// too.
fn owned() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| matches!(std::env::var("PC_OWN").as_deref().map(str::trim), Err(_) | Ok("") | Ok("1") | Ok("on")))
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
    let kinds: Vec<String> = if std::env::var_os("PC_SCHED2_DIAG").is_some() {
        at.iter()
            .map(|a| {
                let mut v: Vec<String> = p.ops[a.lo..a.hi].iter().map(|o| { let s = format!("{:?}", o.kind); s.split(['(', ' ', '{']).next().unwrap_or("").to_string() }).collect();
                v.dedup();
                v.join("/")
            })
            .collect()
    } else {
        vec![String::new(); at.len()]
    };
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
        type Run = (f64, Vec<usize>, Vec<u8>, Loc, [f64; 2], Live, [f64; 8], f64, Vec<f64>);
        let g = Graph { at: &at, preds: &preds, xin: &xin, xout: &xout, xlive: &xlive, keep: &keep, vcls: &vcls, kinds: &kinds };
        let run = |betas: &[f64]| -> Run {
            let mut best: Option<Run> = None;
            for &beta in betas {
                let mut l = Loc { on: loc.on.clone(), av: loc.av.clone() };
                let mut c = ce;
                let mut m = live.clone();
                let mut sts = [0.0f64; 8];
                let (mk, ord, dv, sv) = region(&g, lo, hi, t_all, beta, &mut l, &mut c, &mut m, &cp, &vb, &mut sts);
                if best.as_ref().is_none_or(|b| mk < b.0) {
                    best = Some((mk, ord, dv, l, c, m, sts, beta, sv));
                }
            }
            best.expect("one schedule at least")
        };
        // PC_SCHED2_PART: the greedy placement refined into a partition, scheduled with the devices fixed, per round
        // while the model finishes earlier (PC_SCHED2_PART_ALL=1: each copy penalty's greedy schedule refined, the
        // fastest kept)
        let refine = |mut res: Run| -> Run {
            let n = hi - lo;
            let (mut place, mut stt) = (vec![0u8; n], vec![0.0f64; n]);
            for (j, &k) in res.1.iter().enumerate() {
                place[k - lo] = res.2[j];
                stt[k - lo] = res.8[j];
            }
            let (win, tol, abs_tol) = (knob("PC_SCHED2_PART_WIN", 10.0), knob("PC_SCHED2_PART_TOL", 0.1), knob("PC_SCHED2_PART_ABS", 0.5));
            let fm = knob("PC_SCHED2_PART_FM", 1.0) > 0.0;
            let rounds = knob("PC_SCHED2_PART_ROUNDS", 3.0) as usize;
            let slack = knob("PC_SCHED2_PART_SLACK", 0.0);
            let plog = std::env::var_os("PC_SCHED2_PART_LOG").is_some();
            let mk0 = res.0;
            for r in 0..rounds {
                let mut pl = place.clone();
                let cut = if fm { fm_refine(&g, lo, hi, t_all, &loc, &mut pl, &stt, &cp, win, tol, abs_tol) } else { 0.0 };
                let mut l = Loc { on: loc.on.clone(), av: loc.av.clone() };
                let mut c = ce;
                let mut m = live.clone();
                let mut sts = [0.0f64; 8];
                let (mk, ord, dv, sv) = region_fixed(&g, lo, hi, t_all, &pl, &mut l, &mut c, &mut m, &cp, &vb, &mut sts);
                if plog {
                    eprintln!("[sched2-part] {}..{} round {r}: cut -{cut:.1} ms; {:.1} -> {:.1} ms (greedy {mk0:.1}), copies {} ({:.1} ms) vs {} ({:.1} ms), idle copy {:.1} vs {:.1}", lo, hi, res.0 - t_all, mk - t_all, sts[3], sts[4] + sts[5] + sts[6] + sts[7], res.6[3], res.6[4] + res.6[5] + res.6[6] + res.6[7], sts[0], res.6[0]);
                }
                if mk < res.0 * (1.0 + slack) {
                    for (j, &k) in ord.iter().enumerate() {
                        place[k - lo] = dv[j];
                        stt[k - lo] = sv[j];
                    }
                    res = (mk, ord, dv, l, c, m, sts, -1.0, sv);
                } else {
                    break;
                }
            }
            res
        };
        // PC_SCHED2_PART_COMP=1: one more candidate, the replica lane's component placement (weakly connected
        // components of light atoms and K-batches, a refresh a component of its own, the largest first, each moved
        // when the schedule with the devices fixed finishes earlier), then refined
        let fixed = |pl: &[u8]| -> Run {
            let mut l = Loc { on: loc.on.clone(), av: loc.av.clone() };
            let mut c = ce;
            let mut m = live.clone();
            let mut sts = [0.0f64; 8];
            let (mk, ord, dv, sv) = region_fixed(&g, lo, hi, t_all, pl, &mut l, &mut c, &mut m, &cp, &vb, &mut sts);
            (mk, ord, dv, l, c, m, sts, -2.0, sv)
        };
        let comp = || -> Run {
            let n = hi - lo;
            let mv = |k: usize| matches!(at[lo + k].class, Class::Light | Class::Heavy);
            let mut par: Vec<usize> = (0..n).collect();
            fn fd(p: &mut [usize], mut x: usize) -> usize {
                while p[x] != x {
                    p[x] = p[p[x]];
                    x = p[x];
                }
                x
            }
            for k in 0..n {
                if !mv(k) || at[lo + k].bts {
                    continue;
                }
                for &q in &preds[lo + k] {
                    if q >= lo && q < hi && mv(q - lo) && !at[q].bts {
                        let (a, b) = (fd(&mut par, k), fd(&mut par, q - lo));
                        par[a] = b;
                    }
                }
            }
            let mut comps: HashMap<usize, Vec<usize>> = HashMap::new();
            for k in 0..n {
                if mv(k) {
                    let r = fd(&mut par, k);
                    comps.entry(r).or_default().push(k);
                }
            }
            let mut cl: Vec<(f64, Vec<usize>)> = comps.into_values().map(|ks| (ks.iter().map(|&k| at[lo + k].ms).sum(), ks)).filter(|(w, _)| *w >= 2.0).collect();
            cl.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1[0].cmp(&b.1[0])));
            cl.truncate(knob("PC_SCHED2_PART_COMP_N", 160.0) as usize);
            let mut pl = vec![0u8; n];
            let mut best = fixed(&pl);
            for (_, ks) in &cl {
                for &k in ks {
                    pl[k] = 1;
                }
                let r = fixed(&pl);
                if r.0 + 0.5 <= best.0 {
                    best = r;
                } else {
                    for &k in ks {
                        pl[k] = 0;
                    }
                }
            }
            if std::env::var_os("PC_SCHED2_PART_LOG").is_some() {
                eprintln!("[sched2-part] {lo}..{hi}: components {} tried, {:.1} ms", cl.len(), best.0 - t_all);
            }
            best
        };
        let res = if !(part() && hi > lo) {
            run(&[0.0, 0.5, 2.0])
        } else {
            let mut cands: Vec<Run> = vec![];
            if knob("PC_SCHED2_PART_ALL", 0.0) > 0.0 {
                for b in [0.0, 0.5, 2.0] {
                    cands.push(refine(run(&[b])));
                }
            } else {
                cands.push(refine(run(&[0.0, 0.5, 2.0])));
            }
            if knob("PC_SCHED2_PART_COMP", 0.0) > 0.0 {
                cands.push(refine(comp()));
            }
            cands.into_iter().min_by(|a, b| a.0.total_cmp(&b.0)).expect("one schedule at least")
        };
        let (mk, ord, dv, l, c, m, sts, _, _) = res;
        for i in 0..3 {
            rep.idle[i] += sts[i];
        }
        for i in 0..4 {
            rep.copies_by[i] += sts[4 + i];
            rep.copies_ms += sts[4 + i];
        }
        rep.n_copies += sts[3] as usize;
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

/// A ready atom by urgency (its tail, the longest first), ties by program order.
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
    /// Per atom: its ops' kinds (diagnostics).
    kinds: &'g [String],
}

#[allow(clippy::too_many_arguments)]
fn region(g: &Graph<'_>, lo: usize, hi: usize, t0: f64, beta: f64, loc: &mut Loc, ce: &mut [f64; 2], live: &mut Live, cp: &dyn Fn(VarId) -> f64, vb: &dyn Fn(VarId) -> f64, st: &mut [f64; 8]) -> (f64, Vec<usize>, Vec<u8>, Vec<f64>) {
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
                        st[3] += 1.0;
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
                        st[3] += 1.0;
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
    (t[0].max(t[1]), idx.iter().map(|&k| lo + k).collect(), idx.iter().map(|&k| dv[k]).collect(), idx.iter().map(|&k| start[k]).collect())
}

/// PC_SCHED2_PART=1: the greedy schedule's placement refined into a partition (fewer values crossing, each time window
/// of the greedy schedule still split evenly), then scheduled with each atom's device fixed (`region_fixed`); kept per
/// block when the model finishes it earlier. Unset / `0` = off (default until an A/B wins).
fn part() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| matches!(std::env::var("PC_SCHED2_PART").as_deref().map(str::trim), Ok("1") | Ok("on")))
}

/// A float knob from the environment (diagnostic tuning of PC_SCHED2_PART).
fn knob(name: &str, def: f64) -> f64 {
    std::env::var(name).ok().and_then(|v| v.trim().parse::<f64>().ok()).unwrap_or(def)
}

/// The list schedule of atoms `lo..hi` with each movable atom's device fixed (`place`, per atom of the region; pinned
/// atoms here, split atoms on both): the device free first issues its most urgent ready atom, or, when that one would
/// wait for its inputs, the most urgent ready atom of that device whose inputs are there already and that ends within
/// the wait; a split atom (finish, split refresh batch) goes when it is the most urgent ready atom of all. Same state
/// and statistics as `region`.
#[allow(clippy::too_many_arguments)]
fn region_fixed(g: &Graph<'_>, lo: usize, hi: usize, t0: f64, place: &[u8], loc: &mut Loc, ce: &mut [f64; 2], live: &mut Live, cp: &dyn Fn(VarId) -> f64, vb: &dyn Fn(VarId) -> f64, st: &mut [f64; 8]) -> (f64, Vec<usize>, Vec<u8>, Vec<f64>) {
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
    let split = |k: usize| matches!(at[lo + k].class, Class::Both | Class::Barrier);
    let cost = |k: usize| if split(k) { at[lo + k].ms / 2.0 } else { at[lo + k].ms };
    let home = |k: usize| if split(k) { 2 } else if at[lo + k].class == Class::Pinned { 0 } else { place[k] as usize };
    let mut tail = vec![0.0f64; n];
    if knob("PC_SCHED2_PART_CTAIL", 0.0) > 0.0 {
        // the tails with the copies on the way: a value read on the other device adds its copy
        let mut rd: HashMap<VarId, Vec<usize>> = HashMap::new();
        for k in 0..n {
            for &v in &xin[lo + k] {
                rd.entry(v).or_default().push(k);
            }
        }
        for k in (0..n).rev() {
            let mut m = 0.0f64;
            let outs = &xout[lo + k];
            let h = outs.len() - outs.len() / 2;
            for (j, &v) in outs.iter().enumerate() {
                let on: u8 = if split(k) { if j >= h && owned() { 3 } else { 1 } } else { 1 << home(k) };
                for &r in rd.get(&v).map_or(&[][..], |x| &x[..]) {
                    let rdv = if split(r) { 0 } else { home(r) };
                    m = m.max(tail[r] + if on >> rdv & 1 == 0 { cp(v) } else { 0.0 });
                }
            }
            for &s2 in &succ[k] {
                m = m.max(tail[s2]);
            }
            tail[k] = cost(k) + m;
        }
    } else {
        for k in (0..n).rev() {
            tail[k] = cost(k) + succ[k].iter().map(|&s| tail[s]).fold(0.0, f64::max);
        }
    }
    let mut heaps: [BinaryHeap<R>; 3] = [BinaryHeap::new(), BinaryHeap::new(), BinaryHeap::new()];
    for k in 0..n {
        if npred[k] == 0 {
            heaps[home(k)].push(R(tail[k], k));
        }
    }
    let mut t = [t0; 2];
    let mut start = vec![0.0f64; n];
    let mut dv = vec![0u8; n];
    let mut done = 0usize;
    let diag = std::env::var_os("PC_SCHED2_DIAG").is_some();
    let (mut dg, mut dgt) = ([0.0f64; 4], vec![0.0f64; 60]);
    let steal = knob("PC_SCHED2_PART_STEAL", 0.0) > 0.0;
    let steal_m = knob("PC_SCHED2_PART_STEAL_M", 0.0);
    let steal_n = knob("PC_SCHED2_PART_STEAL_N", 16.0) as usize;
    while done < n {
        let top = |h: &BinaryHeap<R>| h.peek().map_or(f64::NEG_INFINITY, |r| r.0);
        let ts = top(&heaps[2]);
        let k;
        if !heaps[2].is_empty() && ts >= top(&heaps[0]) && ts >= top(&heaps[1]) {
            k = heaps[2].pop().expect("not empty").1;
            for &v in &xin[lo + k] {
                if loc.on[v as usize] & 1 == 0 {
                    st[4 + g.vcls[v as usize] as usize] += cp(v);
                    st[3] += 1.0;
                }
            }
            let (x, _) = loc.arrive(&xin[lo + k], 0, ce, cp, true);
            if at[lo + k].class == Class::Both && !fin_async() {
                let s0 = t[0].max(t[1]).max(x);
                let f = s0 + cost(k);
                st[1] += (s0 - t[0]) + (s0 - t[1]);
                t = [f, f];
                start[k] = s0;
                loc.put_split(&xout[lo + k], f);
            } else {
                let (s0, s1) = (t[0].max(x), t[1].max(x));
                let (f0, f1) = (s0 + cost(k), s1 + cost(k));
                let end = f0.max(f1);
                st[1] += (s0 - t[0]) + (s1 - t[1]);
                st[2] += end - f0;
                t = [end, f1];
                start[k] = s0.max(s1);
                loc.put_split(&xout[lo + k], end);
            }
            dv[k] = 0;
        } else {
            let mut d = match (heaps[0].is_empty(), heaps[1].is_empty()) {
                (false, false) => usize::from(t[1] < t[0]),
                (false, true) => 0,
                (true, false) => 1,
                (true, true) => panic!("sched2: no ready atom with {} of {n} left", n - done),
            };
            // PC_SCHED2_PART_STEAL: the device free first with nothing of its own ready takes the other device's
            // ready atom it would start earlier (copies included)
            let dmin = usize::from(t[1] < t[0]);
            if steal && d != dmin {
                let e = d;
                let mut seen: Vec<R> = vec![];
                let mut found = None;
                while let Some(r) = heaps[e].pop() {
                    let j = r.1;
                    let (mut c1, mut c2) = (*ce, *ce);
                    let sd = t[dmin].max(loc.arrive(&xin[lo + j], dmin, &mut c1, cp, false).0);
                    let se = t[e].max(loc.arrive(&xin[lo + j], e, &mut c2, cp, false).0);
                    seen.push(r);
                    if sd + steal_m < se && live.cap.is_none_or(|cp_| live.live + live.grow(&xin[lo + j], &g.xlive[lo + j], g.keep, vb) <= cp_) {
                        found = Some(seen.len() - 1);
                        break;
                    }
                    if seen.len() >= steal_n {
                        break;
                    }
                }
                let got = found.map(|i| seen.swap_remove(i).1);
                for r in seen {
                    heaps[e].push(r);
                }
                if let Some(j) = got {
                    heaps[dmin].push(R(tail[j], j));
                    d = dmin;
                }
            }
            // the most urgent ready atom of d (under a memory cap the most urgent that fits, else the least growing)
            let mut tried: Vec<(f64, R)> = vec![];
            let mut pick = None;
            while let Some(r) = heaps[d].pop() {
                let fits = live.cap.is_none_or(|c| {
                    let gr = live.grow(&xin[lo + r.1], &g.xlive[lo + r.1], g.keep, vb);
                    tried.push((gr, R(r.0, r.1)));
                    live.live + gr <= c
                });
                if fits {
                    if live.cap.is_some() {
                        tried.pop();
                    }
                    pick = Some(r.1);
                    break;
                }
            }
            let mut kk = match pick {
                Some(k) => k,
                None => {
                    let j = (0..tried.len()).min_by(|&a, &b| tried[a].0.total_cmp(&tried[b].0).then(a.cmp(&b))).expect("one tried");
                    tried.swap_remove(j).1 .1
                }
            };
            for (_, r) in tried {
                heaps[d].push(r);
            }
            // its start on d; when it would wait, the most urgent ready atom of d that fits in the wait instead
            let mut c = *ce;
            let (x, _) = loc.arrive(&xin[lo + kk], d, &mut c, cp, false);
            let s_k = t[d].max(x);
            if s_k > t[d] {
                let mut seen: Vec<R> = vec![];
                let mut found = None;
                while let Some(r) = heaps[d].pop() {
                    let j = r.1;
                    let xs = &xin[lo + j];
                    if xs.iter().all(|&v| loc.on[v as usize] >> d & 1 != 0) {
                        let sj = xs.iter().map(|&v| loc.av[v as usize][d]).fold(t[d], f64::max);
                        if sj + cost(j) <= s_k && live.cap.is_none_or(|cp_| live.live + live.grow(xs, &g.xlive[lo + j], g.keep, vb) <= cp_) {
                            found = Some(j);
                            break;
                        }
                    }
                    seen.push(r);
                    if seen.len() >= 256 {
                        break;
                    }
                }
                for r in seen {
                    heaps[d].push(r);
                }
                if let Some(j) = found {
                    heaps[d].push(R(tail[kk], kk));
                    kk = j;
                } else if steal {
                    // nothing fits the wait: the other device's ready atom d would start earlier than it and than kk
                    let e = 1 - d;
                    let mut seen: Vec<R> = vec![];
                    let mut got = None;
                    while let Some(r) = heaps[e].pop() {
                        let j = r.1;
                        let (mut c1, mut c2) = (*ce, *ce);
                        let sd = t[d].max(loc.arrive(&xin[lo + j], d, &mut c1, cp, false).0);
                        let se = t[e].max(loc.arrive(&xin[lo + j], e, &mut c2, cp, false).0);
                        seen.push(r);
                        if sd + steal_m < se && sd + cost(j) <= s_k && live.cap.is_none_or(|cp_| live.live + live.grow(&xin[lo + j], &g.xlive[lo + j], g.keep, vb) <= cp_) {
                            got = Some(seen.len() - 1);
                            break;
                        }
                        if seen.len() >= steal_n {
                            break;
                        }
                    }
                    let got = got.map(|i| seen.swap_remove(i).1);
                    for r in seen {
                        heaps[e].push(r);
                    }
                    if let Some(j) = got {
                        heaps[d].push(R(tail[kk], kk));
                        kk = j;
                    }
                }
            }
            k = kk;
            let xl = xin[lo + k].iter().filter(|&&v| loc.on[v as usize] >> d & 1 != 0).map(|&v| loc.av[v as usize][d]).fold(t[d], f64::max);
            for &v in &xin[lo + k] {
                if loc.on[v as usize] >> d & 1 == 0 {
                    st[4 + g.vcls[v as usize] as usize] += cp(v);
                    st[3] += 1.0;
                }
            }
            let pre: Vec<u8> = if diag { xin[lo + k].iter().map(|&v| loc.on[v as usize]).collect() } else { vec![] };
            let (x, _) = loc.arrive(&xin[lo + k], d, ce, cp, true);
            let s = t[d].max(x);
            st[1] += xl.min(s) - t[d];
            st[0] += (s - xl).max(0.0);
            if diag && s > xl {
                // the copy arriving last: its producer's class
                let (mut bv, mut ba) = (None, f64::NEG_INFINITY);
                for (j, &v) in xin[lo + k].iter().enumerate() {
                    if pre[j] >> d & 1 == 0 && loc.av[v as usize][d] > ba {
                        ba = loc.av[v as usize][d];
                        bv = Some(v);
                    }
                }
                if let Some(v) = bv {
                    dg[g.vcls[v as usize] as usize] += s - xl;
                    let w = (((s - t0) / 50.0) as usize).min(dgt.len() - 1);
                    dgt[w] += s - xl;
                }
            }
            start[k] = s;
            t[d] = s + cost(k);
            loc.put(&xout[lo + k], d, t[d]);
            dv[k] = d as u8;
        }
        live.take(&xin[lo + k], &g.xlive[lo + k], g.keep, vb);
        done += 1;
        for &s2 in &succ[k] {
            npred[s2] -= 1;
            if npred[s2] == 0 {
                heaps[home(s2)].push(R(tail[s2], s2));
            }
        }
    }
    if diag {
        eprintln!("[sched2-diag] idle copy by producer: input {:.1} light {:.1} heavy {:.1} split {:.1}; per 50 ms: {}", dg[0], dg[1], dg[2], dg[3], dgt.iter().map(|x| format!("{x:.0}")).collect::<Vec<_>>().join(" "));
        let mut sp: Vec<(f64, f64, usize, &str)> = (0..n).filter(|&k| split(k)).map(|k| (start[k], cost(k), xout[lo + k].len(), if at[lo + k].class == Class::Barrier { "F" } else { "B" })).collect();
        sp.sort_by(|a, b| a.0.total_cmp(&b.0));
        eprintln!("[sched2-diag] split atoms (start+half ms/outs): {}", sp.iter().map(|x| format!("{}{:.0}+{:.0}/{}", x.3, x.0, x.1, x.2)).collect::<Vec<_>>().join(" "));
        // busy per device per 10 ms
        let mut busy = vec![[0.0f64; 2]; ((t[0].max(t[1]) - t0) / 10.0) as usize + 1];
        for k in 0..n {
            let ds: &[usize] = if split(k) { &[0, 1] } else { if dv[k] == 0 { &[0] } else { &[1] } };
            for &d in ds {
                let (mut a, b) = (start[k] - t0, start[k] - t0 + cost(k));
                while a < b - 1e-9 {
                    let w = (a / 10.0) as usize;
                    let e = ((w + 1) as f64 * 10.0).min(b);
                    if w < busy.len() {
                        busy[w][d] += e - a;
                    }
                    a = e;
                }
            }
        }
        eprintln!("[sched2-diag] busy % per 10 ms (dev0/dev1): {}", busy.iter().map(|x| format!("{:.0}/{:.0}", x[0] * 10.0, x[1] * 10.0)).collect::<Vec<_>>().join(" "));
        if let Some((ra, rb)) = std::env::var("PC_SCHED2_DIAG_RANGE").ok().and_then(|v| v.split_once(':').and_then(|(a, b)| Some((a.parse::<f64>().ok()?, b.parse::<f64>().ok()?)))) {
            let mut ks: Vec<usize> = (0..n).filter(|&k| start[k] - t0 >= ra && start[k] - t0 < rb).collect();
            ks.sort_by(|&a, &b| start[a].total_cmp(&start[b]));
            let pa: HashMap<VarId, usize> = (0..n).flat_map(|k| xout[lo + k].iter().map(move |&v| (v, k))).collect();
            for k in ks {
                let a = &at[lo + k];
                let ins: Vec<String> = xin[lo + k].iter().map(|&v| match pa.get(&v) { Some(&q) => format!("{}@{}{}", q, if split(q) { 9 } else { dv[q] }, if !split(q) && dv[q] != dv[k] && !split(k) { "*" } else { "" }), None => "in".into() }).collect();
                eprintln!("  {:8.2} d{} k{} {:?} {:.3} ms {} ops [{}] ins {}", start[k] - t0, if split(k) { 9 } else { dv[k] }, k, a.class, cost(k), a.hi - a.lo, g.kinds[lo + k], ins.join(","));
            }
        }
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| start[a].total_cmp(&start[b]).then(a.cmp(&b)));
    (t[0].max(t[1]), idx.iter().map(|&k| lo + k).collect(), idx.iter().map(|&k| dv[k]).collect(), idx.iter().map(|&k| start[k]).collect())
}

/// PC_SCHED2_PART's partition: Fiduccia-Mattheyses passes over the movable atoms of `lo..hi` (light, heavy), from
/// `place`, cutting the copy ms of the values that cross (a value copied once per device it is read on and not made
/// on: a split atom's owned outputs on both devices, its other outputs, block inputs and pinned atoms' here, split
/// atoms and the block's outputs reading here), each move keeping the time window it falls in (`start`, the atom's
/// start in a schedule; windows of `win` ms) within `tol` of its work split evenly, or evening it.
#[allow(clippy::too_many_arguments)]
fn fm_refine(g: &Graph<'_>, lo: usize, hi: usize, t0: f64, loc0: &Loc, place: &mut [u8], start: &[f64], cp: &dyn Fn(VarId) -> f64, win: f64, tol: f64, abs_tol: f64) -> f64 {
    let (at, xin, xout) = (g.at, g.xin, g.xout);
    let n = hi - lo;
    let movable = |k: usize| matches!(at[lo + k].class, Class::Light | Class::Heavy);
    let split = |k: usize| matches!(at[lo + k].class, Class::Both | Class::Barrier);
    let nv = loc0.on.len();
    // producer (in region) of each var, its devices when not movable
    let mut prod: Vec<u32> = vec![u32::MAX; nv];
    let mut pfix: Vec<u8> = loc0.on.clone();
    for k in 0..n {
        let outs = &xout[lo + k];
        let h = outs.len() - outs.len() / 2;
        for (j, &v) in outs.iter().enumerate() {
            if movable(k) {
                prod[v as usize] = k as u32;
            } else if split(k) {
                pfix[v as usize] = if j >= h && owned() { 3 } else { 1 };
            } else {
                pfix[v as usize] = 1;
            }
        }
    }
    // readers per var (movable ones by index), fixed readers' devices
    let mut readers: Vec<Vec<u32>> = vec![vec![]; nv];
    let mut cnt: Vec<[u32; 2]> = vec![[0; 2]; nv];
    for k in 0..n {
        for &v in &xin[lo + k] {
            if movable(k) {
                readers[v as usize].push(k as u32);
                cnt[v as usize][place[k] as usize] += 1;
            } else {
                cnt[v as usize][0] += 1;
            }
        }
    }
    for &v in g.keep {
        cnt[v as usize][0] += 1;
    }
    let pdev = |v: usize, place: &[u8]| -> u8 {
        let p = prod[v];
        if p == u32::MAX {
            pfix[v]
        } else {
            1 << place[p as usize]
        }
    };
    let vcost = |c: [u32; 2], pm: u8| -> f64 { (0..2).filter(|&d| c[d] > 0 && pm >> d & 1 == 0).count() as f64 };
    let gain = |k: usize, place: &[u8], cnt: &[[u32; 2]]| -> f64 {
        let d = place[k] as usize;
        let e = 1 - d;
        let mut gsum = 0.0;
        for &v in &xout[lo + k] {
            let vi = v as usize;
            let c = cnt[vi];
            gsum += (vcost(c, 1 << d) - vcost(c, 1 << e)) * cp(v);
        }
        for &v in &xin[lo + k] {
            let vi = v as usize;
            let pm = pdev(vi, place);
            let c = cnt[vi];
            let mut c2 = c;
            c2[d] -= 1;
            c2[e] += 1;
            gsum += (vcost(c, pm) - vcost(c2, pm)) * cp(v);
        }
        gsum
    };
    // windows
    let nw = (((start.iter().fold(t0, |a, &b| a.max(b)) - t0) / win).floor() as usize) + 1;
    let wof: Vec<usize> = (0..n).map(|k| (((start[k] - t0) / win).floor().max(0.0) as usize).min(nw - 1)).collect();
    let mut load = vec![[0.0f64; 2]; nw];
    for k in 0..n {
        if movable(k) {
            load[wof[k]][place[k] as usize] += at[lo + k].ms;
        }
    }
    // PC_SCHED2_PART_LAMBDA > 0: balance as a cost (each ms a window's split is off past its tolerance costs lambda
    // ms of copies) instead of a constraint
    let lam = knob("PC_SCHED2_PART_LAMBDA", 0.0);
    let pen = |a: f64, b: f64| ((a - b).abs() - (tol * (a + b)).max(abs_tol)).max(0.0);
    let bal = |k: usize, place: &[u8], load: &[[f64; 2]]| -> f64 {
        let w = wof[k];
        let d = place[k] as usize;
        let m = at[lo + k].ms;
        pen(load[w][0], load[w][1]) - pen(load[w][d] - m, load[w][1 - d] + m)
    };
    let ok = |k: usize, place: &[u8], load: &[[f64; 2]]| -> bool {
        if lam > 0.0 {
            return true;
        }
        let w = wof[k];
        let d = place[k] as usize;
        let m = at[lo + k].ms;
        let (a, b) = (load[w][d] - m, load[w][1 - d] + m);
        let before = (load[w][0] - load[w][1]).abs();
        let after = (a - b).abs();
        after <= before + 1e-12 || after <= (tol * (a + b)).max(abs_tol)
    };
    let mut inwin: Vec<Vec<usize>> = vec![vec![]; nw];
    if lam > 0.0 {
        for k in 0..n {
            if movable(k) {
                inwin[wof[k]].push(k);
            }
        }
    }
    // the neighbours whose gain a move of k changes
    let neigh = |k: usize, out: &mut Vec<usize>| {
        out.clear();
        for &v in xout[lo + k].iter().chain(&xin[lo + k]) {
            let vi = v as usize;
            if prod[vi] != u32::MAX {
                out.push(prod[vi] as usize);
            }
            for &r in &readers[vi] {
                out.push(r as usize);
            }
        }
        out.sort_unstable();
        out.dedup();
    };
    let passes = knob("PC_SCHED2_PART_PASSES", 8.0) as usize;
    let stall = knob("PC_SCHED2_PART_STALL", 400.0) as usize;
    let mut total = 0.0;
    let mut nb: Vec<usize> = vec![];
    for _ in 0..passes {
        let mut stamp = vec![0u32; n];
        let mut locked = vec![false; n];
        #[derive(PartialEq)]
        struct G(f64, u32, usize);
        impl Eq for G {}
        impl PartialOrd for G {
            fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(o))
            }
        }
        impl Ord for G {
            fn cmp(&self, o: &Self) -> std::cmp::Ordering {
                self.0.total_cmp(&o.0).then(o.2.cmp(&self.2))
            }
        }
        let tg = |k: usize, place: &[u8], cnt: &[[u32; 2]], load: &[[f64; 2]]| gain(k, place, cnt) + if lam > 0.0 { lam * bal(k, place, load) } else { 0.0 };
        let mut heap: BinaryHeap<G> = (0..n).filter(|&k| movable(k)).map(|k| G(tg(k, place, &cnt, &load), 0, k)).collect();
        let mut moves: Vec<usize> = vec![];
        let (mut cum, mut best, mut best_len) = (0.0f64, 0.0f64, 0usize);
        while let Some(G(gk, sk, k)) = heap.pop() {
            if locked[k] || sk != stamp[k] {
                continue;
            }
            locked[k] = true;
            if !ok(k, place, &load) {
                continue;
            }
            // move k
            let d = place[k] as usize;
            let e = 1 - d;
            for &v in &xin[lo + k] {
                cnt[v as usize][d] -= 1;
                cnt[v as usize][e] += 1;
            }
            place[k] = e as u8;
            load[wof[k]][d] -= at[lo + k].ms;
            load[wof[k]][e] += at[lo + k].ms;
            moves.push(k);
            cum += gk;
            if cum > best + 1e-9 {
                best = cum;
                best_len = moves.len();
            }
            if moves.len() - best_len > stall {
                break;
            }
            neigh(k, &mut nb);
            if lam > 0.0 {
                nb.extend_from_slice(&inwin[wof[k]]);
                nb.sort_unstable();
                nb.dedup();
            }
            for &j in &nb {
                if !locked[j] && movable(j) {
                    stamp[j] += 1;
                    heap.push(G(tg(j, place, &cnt, &load), stamp[j], j));
                }
            }
        }
        // back to the best prefix
        for &k in moves[best_len..].iter().rev() {
            let e = place[k] as usize;
            let d = 1 - e;
            for &v in &xin[lo + k] {
                cnt[v as usize][e] -= 1;
                cnt[v as usize][d] += 1;
            }
            place[k] = d as u8;
            load[wof[k]][e] -= at[lo + k].ms;
            load[wof[k]][d] += at[lo + k].ms;
        }
        total += best;
        if best <= 1e-9 {
            break;
        }
    }
    total
}
