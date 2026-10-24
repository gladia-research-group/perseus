//! K-batching: sibling ops (same kind and parameters, same input metas and preludes, same route) whose inputs all
//! exist at the first sibling's position run as one `Kind::Batched` launch of up to `k_max` members. A later sibling
//! is hoisted to the first one's position only when every input it reads is produced before that position (the
//! dependency slack); that also makes the members mutually independent. Passes in order: bootstraps (same route),
//! key-switching rotations (rotate of one step, rotate_sum of one step list, conjugate), relinearized products,
//! rescales. Plaintext and scalar operands may differ between members (the "const shape" is what must agree).
//! Products by a plaintext or a constant stay where the capture put them, next to the sum that reads them.
use crate::program::{BatchReport, Kind, KindCount, Op, Program, VarId};
use std::collections::{BTreeMap, HashMap};

/// The batching key of a single op: None = never batched; `pass` = None accepts every class.
fn key(p: &Program, op: &Op, pass: Option<usize>, cf_free: bool) -> Option<String> {
    let k = &op.kind;
    let class = match k {
        Kind::Bootstrap(_) => 0,
        Kind::Rotate(_) | Kind::RotateSum(_) | Kind::Conjugate => 1,
        Kind::Mult | Kind::Square => 2,
        Kind::Rescale => 3,
        _ => return None,
    };
    if pass.is_some_and(|c| c != class) {
        return None;
    }
    let param = match k {
        Kind::Bootstrap(s) => {
            let site = &p.bts_sites[*s as usize];
            let mut r = site.route;
            if cf_free {
                // the correction factor as a per-member constant of the batched EvalMod
                r.cf = None;
            }
            format!("{r:?}|{:?}|{:?}", site.prescale, site.offset)
        }
        Kind::Rotate(r) => r.to_string(),
        Kind::RotateSum(st) => format!("{st:?}"),
        Kind::MultPt(id) => {
            let pt = &p.pt_refs[*id as usize];
            format!("pt@{}d{}", pt.level, pt.deg)
        }
        _ => String::new(),
    };
    let metas: Vec<(u32, u8)> = op.ins.iter().map(|&v| (p.vars[v as usize].meta.level, p.vars[v as usize].meta.deg)).collect();
    Some(format!("{}|{param}|{metas:?}|{:?}", k.name(), op.pre))
}

fn one_pass(p: &mut Program, pass: usize, k_max: usize, cf_free: bool) {
    let ops = std::mem::take(&mut p.ops);
    // position of each var's producer in the current order
    let mut pos_of: HashMap<VarId, usize> = HashMap::new();
    for (i, op) in ops.iter().enumerate() {
        for v in op.all_outs() {
            pos_of.insert(v, i);
        }
    }
    let keys: Vec<Option<String>> = ops.iter().map(|o| if matches!(o.kind, Kind::Batched(_)) { None } else { key(p, o, Some(pass), cf_free) }).collect();
    let mut bucket: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, k) in keys.iter().enumerate() {
        if let Some(k) = k {
            bucket.entry(k.as_str()).or_default().push(i);
        }
    }
    // ready[j] = the position after which every input of op j exists
    let ready: Vec<usize> = ops.iter().map(|o| o.all_ins().iter().filter_map(|v| pos_of.get(v)).map(|&q| q + 1).max().unwrap_or(0)).collect();
    let mut taken = vec![false; ops.len()];
    let mut groups: Vec<Option<Vec<usize>>> = vec![None; ops.len()];
    for i in 0..ops.len() {
        if taken[i] {
            continue;
        }
        let Some(k) = &keys[i] else { continue };
        let mut g = vec![i];
        taken[i] = true;
        for &j in &bucket[k.as_str()] {
            if g.len() >= k_max {
                break;
            }
            if j <= i || taken[j] {
                continue;
            }
            if ready[j] <= i {
                g.push(j);
                taken[j] = true;
            }
        }
        if g.len() > 1 {
            groups[i] = Some(g);
        } else {
            taken[i] = false;
        }
    }
    let mut ops: Vec<Option<Op>> = ops.into_iter().map(Some).collect();
    let mut out = Vec::with_capacity(ops.len());
    for i in 0..ops.len() {
        if let Some(g) = groups[i].take() {
            let members: Vec<Op> = g.iter().filter_map(|&j| ops[j].take()).collect();
            let origin = members[0].origin.clone();
            out.push(Op { kind: Kind::Batched(members), ins: vec![], outs: vec![], pre: vec![], origin, inplace: false });
        } else if let Some(op) = ops[i].take() {
            out.push(op);
        }
    }
    p.ops = out;
}

/// How ops are ordered before grouping.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Schedule {
    /// The capture's order; later siblings hoisted into the first one's launch when their inputs exist there.
    #[default]
    Program,
    /// List scheduling by (ASAP depth, capture index): independent chains of the same shape advance in lockstep, so
    /// their siblings are ready together (more batching, more live ciphertexts).
    Lockstep,
    /// The capture's order, except that the siblings of a rotate_sum or a dense StC-first refresh (the launches a
    /// K-batch makes cheaper per ciphertext: one read of each key, one pass over the route's tables) are pulled
    /// forward together with the ops they wait for (`PULL_OPS` at most, from the next `PULL_WINDOW` ops): independent
    /// chains of one shape meet at those launches, a few chain states live at once. Then the `Program` passes.
    Siblings,
}

/// Ops a sibling may bring forward with it (its unissued ancestors), and how far ahead siblings are looked for.
const PULL_OPS: usize = 256;
const PULL_WINDOW: usize = 1024;
/// Members of a pulled batch (pc-gpu `ks::KEY_KMAX`: one inner-product launch per key; the StC-first route's work
/// buffers hold as many as the PCMM finish batch).
pub const PULL_K: usize = 4;

pub fn batch(p: &mut Program, k_max: u32, schedule: Schedule, cf_free: bool) {
    let k = k_max.max(1) as usize;
    if k > 1 {
        match schedule {
            Schedule::Program | Schedule::Siblings => {
                if schedule == Schedule::Siblings {
                    pull(p, k.min(PULL_K), cf_free);
                }
                for pass in 0..4 {
                    one_pass(p, pass, k, cf_free);
                }
            }
            Schedule::Lockstep => lockstep(p, k, cf_free),
        }
    }
    p.batch_report = report(p, k_max);
}

/// `Schedule::Siblings`' grouping pass (see there).
fn pull(p: &mut Program, k_max: usize, cf_free: bool) {
    let ops = std::mem::take(&mut p.ops);
    let n = ops.len();
    let mut prod: HashMap<VarId, usize> = HashMap::new();
    for (i, op) in ops.iter().enumerate() {
        for v in op.all_outs() {
            prod.insert(v, i);
        }
    }
    let preds: Vec<Vec<usize>> = ops
        .iter()
        .map(|o| {
            let mut d: Vec<usize> = o.all_ins().iter().filter_map(|v| prod.get(v).copied()).collect();
            d.sort_unstable();
            d.dedup();
            d
        })
        .collect();
    let keys: Vec<Option<String>> = ops
        .iter()
        .map(|o| match &o.kind {
            Kind::RotateSum(_) | Kind::Mult | Kind::Square => key(p, o, None, cf_free),
            Kind::Bootstrap(s) if p.bts_sites[*s as usize].route.slots == 0 && p.bts_sites[*s as usize].route.stc_first => key(p, o, None, cf_free),
            _ => None,
        })
        .collect();
    let mut bucket: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, k) in keys.iter().enumerate() {
        if let Some(k) = k {
            bucket.entry(k.as_str()).or_default().push(i);
        }
    }
    let mut done = vec![false; n];
    let mut ops: Vec<Option<Op>> = ops.into_iter().map(Some).collect();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if done[i] {
            continue;
        }
        done[i] = true;
        let mut members = vec![i];
        if let Some(k) = &keys[i] {
            for &j in bucket[k.as_str()].iter().filter(|&&j| j > i && j <= i + PULL_WINDOW) {
                if members.len() >= k_max {
                    break;
                }
                if done[j] {
                    continue;
                }
                // j's unissued ancestors: none of them a member (j independent of the batch), few enough to pull
                let mut anc: Vec<usize> = vec![];
                let mut stack: Vec<usize> = preds[j].clone();
                let mut ok = true;
                while let Some(a) = stack.pop() {
                    if done[a] && !members.contains(&a) || anc.contains(&a) {
                        continue;
                    }
                    if members.contains(&a) || anc.len() >= PULL_OPS {
                        ok = false;
                        break;
                    }
                    anc.push(a);
                    stack.extend(preds[a].iter().copied());
                }
                if !ok {
                    continue;
                }
                anc.sort_unstable();
                for a in anc {
                    done[a] = true;
                    out.extend(ops[a].take());
                }
                done[j] = true;
                members.push(j);
            }
        }
        let mut m: Vec<Op> = members.iter().filter_map(|&j| ops[j].take()).collect();
        if m.len() == 1 {
            out.append(&mut m);
        } else {
            let origin = m[0].origin.clone();
            out.push(Op { kind: Kind::Batched(m), ins: vec![], outs: vec![], pre: vec![], origin, inplace: false });
        }
    }
    p.ops = out;
}

fn lockstep(p: &mut Program, k_max: usize, cf_free: bool) {
    use std::cmp::Reverse;
    use std::collections::{BTreeSet, BinaryHeap};
    let ops = std::mem::take(&mut p.ops);
    let n = ops.len();
    let mut prod: HashMap<VarId, usize> = HashMap::new();
    for (i, op) in ops.iter().enumerate() {
        for v in op.all_outs() {
            prod.insert(v, i);
        }
    }
    let mut succ: Vec<Vec<usize>> = vec![vec![]; n];
    let mut indeg = vec![0usize; n];
    let mut asap = vec![0usize; n];
    for (i, op) in ops.iter().enumerate() {
        let mut deps: Vec<usize> = op.all_ins().iter().filter_map(|v| prod.get(v).copied()).collect();
        deps.sort_unstable();
        deps.dedup();
        for &d in &deps {
            succ[d].push(i);
            asap[i] = asap[i].max(asap[d] + 1);
        }
        indeg[i] = deps.len();
    }
    let mut ids: HashMap<String, usize> = HashMap::new();
    let keys: Vec<Option<usize>> = ops
        .iter()
        .map(|o| {
            key(p, o, None, cf_free).map(|k| {
                let n = ids.len();
                *ids.entry(k).or_insert(n)
            })
        })
        .collect();
    let mut heap: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new();
    let mut by_key: HashMap<usize, BTreeSet<(usize, usize)>> = HashMap::new();
    let push = |i: usize, heap: &mut BinaryHeap<Reverse<(usize, usize)>>, by_key: &mut HashMap<usize, BTreeSet<(usize, usize)>>| {
        heap.push(Reverse((asap[i], i)));
        if let Some(k) = keys[i] {
            by_key.entry(k).or_default().insert((asap[i], i));
        }
    };
    for i in 0..n {
        if indeg[i] == 0 {
            push(i, &mut heap, &mut by_key);
        }
    }
    let mut done = vec![false; n];
    let mut ops: Vec<Option<Op>> = ops.into_iter().map(Some).collect();
    let mut out = Vec::with_capacity(n);
    while let Some(Reverse((_, i))) = heap.pop() {
        if done[i] {
            continue;
        }
        let mut group = vec![i];
        if let Some(k) = keys[i] {
            let set = by_key.entry(k).or_default();
            set.remove(&(asap[i], i));
            while group.len() < k_max {
                let Some(&e) = set.iter().next() else { break };
                set.remove(&e);
                group.push(e.1);
            }
        }
        for &j in &group {
            done[j] = true;
        }
        let members: Vec<Op> = group.iter().filter_map(|&j| ops[j].take()).collect();
        if members.len() == 1 {
            out.extend(members);
        } else {
            let origin = members[0].origin.clone();
            out.push(Op { kind: Kind::Batched(members), ins: vec![], outs: vec![], pre: vec![], origin, inplace: false });
        }
        for &j in &group {
            for &s in &succ[j] {
                indeg[s] -= 1;
                if indeg[s] == 0 {
                    push(s, &mut heap, &mut by_key);
                }
            }
        }
    }
    p.ops = out;
}

pub fn report(p: &Program, k_max: u32) -> BatchReport {
    let mut r = BatchReport { k_max, ..Default::default() };
    let mut per: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
    for op in &p.ops {
        let m = op.members();
        let name = m[0].kind.name();
        let e = per.entry(name).or_default();
        e.0 += m.len();
        e.1 += 1;
        r.ops += m.len();
        r.launches += 1;
        if matches!(m[0].kind, Kind::Bootstrap(_)) {
            r.bootstraps += m.len();
            r.bootstrap_launches += 1;
            *r.bootstrap_hist.entry(m.len()).or_default() += 1;
        }
    }
    r.per_kind = per.into_iter().map(|(kind, (ops, launches))| KindCount { kind: kind.to_string(), ops, launches }).collect();
    r
}
