//! K-batching: sibling ops (same kind and parameters, same input metas and preludes, same route) whose inputs all
//! exist at the first sibling's position run as one `Kind::Batched` launch of up to `k_max` members. A later sibling
//! is hoisted to the first one's position only when every input it reads is produced before that position (the
//! dependency slack); that also makes the members mutually independent. Passes in order: bootstraps (same route),
//! key-switching rotations (rotate of one step, rotate_sum of one step list, conjugate), relinearized products,
//! rescales. Plaintext and scalar operands may differ between members (the "const shape" is what must agree).
//! Products by a plaintext or a constant stay where the capture put them, next to the sum that reads them.
//! Refresh pairs (`Pairing`, config `bts.pair`): a refresh `Pairing::takes` keys by its route without the CF (and
//! without the site's prescale and offset, which the replay never reads), and two such siblings make a batch: the
//! replay runs them as one complex refresh (pc-gpu `BtsRoute::run_pair`), each member keeping its own CF. `Schedule::Siblings` pulls them together from further
//! apart than other siblings (`pull_pairs`).
use crate::program::{BatchReport, Kind, KindCount, Op, Pairing, Program, VarId};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Members of a refresh pair (x + i y: two real payloads in one complex refresh).
pub const PAIR_K: usize = 2;

/// Whether `op` is a refresh that batches in pairs.
pub fn paired(p: &Program, op: &Op, pair: Option<Pairing>) -> bool {
    match (&op.kind, pair) {
        (Kind::Bootstrap(s), Some(pr)) => pr.takes(&p.bts_sites[*s as usize].route, &p.vars[op.ins[0] as usize].meta),
        _ => false,
    }
}

/// The most members a batch headed by `op` takes.
fn cap(p: &Program, op: &Op, k_max: usize, pair: Option<Pairing>) -> usize {
    if paired(p, op, pair) {
        PAIR_K.min(k_max)
    } else {
        k_max
    }
}

/// The batching key of a single op: None = never batched; `pass` = None accepts every class.
fn key(p: &Program, op: &Op, pass: Option<usize>, cf_free: bool, pair: Option<Pairing>) -> Option<String> {
    let k = &op.kind;
    let class = match k {
        Kind::Bootstrap(_) => 0,
        Kind::Rotate(_) | Kind::RotateSum(_) | Kind::Conjugate => 1,
        Kind::Mult | Kind::Square | Kind::MultRescale { .. } => 2,
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
            if paired(p, op, pair) {
                // a pair member keeps its own CF (its pre- and post-factor); prescale and offset are not replayed
                r.cf = None;
                format!("pair|{r:?}")
            } else {
                if cf_free {
                    // the correction factor as a per-member constant of the batched EvalMod
                    r.cf = None;
                }
                format!("{r:?}|{:?}|{:?}", site.prescale, site.offset)
            }
        }
        Kind::Rotate(r) => r.to_string(),
        Kind::RotateSum(st) => format!("{st:?}"),
        Kind::MultRescale { square, dbl, .. } => format!("{square}{dbl}"),
        Kind::MultPt(id) => {
            let pt = &p.pt_refs[*id as usize];
            format!("pt@{}d{}", pt.level, pt.deg)
        }
        _ => String::new(),
    };
    let metas: Vec<(u32, u8)> = op.ins.iter().map(|&v| (p.vars[v as usize].meta.level, p.vars[v as usize].meta.deg)).collect();
    Some(format!("{}|{param}|{metas:?}|{:?}", k.name(), op.pre))
}

fn one_pass(p: &mut Program, pass: usize, k_max: usize, cf_free: bool, pair: Option<Pairing>) {
    let ops = std::mem::take(&mut p.ops);
    // position of each var's producer in the current order
    let mut pos_of: HashMap<VarId, usize> = HashMap::new();
    for (i, op) in ops.iter().enumerate() {
        for v in op.all_outs() {
            pos_of.insert(v, i);
        }
    }
    let keys: Vec<Option<String>> = ops.iter().map(|o| if matches!(o.kind, Kind::Batched(_)) { None } else { key(p, o, Some(pass), cf_free, pair) }).collect();
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
        let k_i = cap(p, &ops[i], k_max, pair);
        let mut g = vec![i];
        taken[i] = true;
        for &j in &bucket[k.as_str()] {
            if g.len() >= k_i {
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
/// The same for the second member of a refresh pair (launches ahead, ops brought forward per member): a pair saves a
/// refresh, so it may bring more of its chain along (the prefill's same-depth refreshes of two head groups sit 1-4k
/// ops apart). A mate a pulled member brings along (`pull_pairs`) adds up to as many ops again, from up to about twice
/// the window; the live ciphertexts this costs depend on the plan (measure the peak).
const PAIR_OPS: usize = 1024;
const PAIR_WINDOW: usize = 2560;
/// Members of a pulled batch (pc-gpu `ks::KEY_KMAX`: one inner-product launch per key; the StC-first route's work
/// buffers hold as many as the PCMM finish batch).
pub const PULL_K: usize = 4;

/// `pair`: refreshes batch in pairs (see the module doc).
pub fn batch(p: &mut Program, k_max: u32, schedule: Schedule, cf_free: bool, pair: Option<Pairing>) {
    let k = k_max.max(1) as usize;
    if k > 1 {
        match schedule {
            Schedule::Program | Schedule::Siblings => {
                if schedule == Schedule::Siblings {
                    // pairs first, on the capture's order: the pull's batches would tie independent chains together
                    if let Some(pr) = pair {
                        pull_pairs(p, pr);
                    }
                    pull(p, k.min(PULL_K), cf_free, pair);
                }
                for pass in 0..4 {
                    one_pass(p, pass, k, cf_free, pair);
                }
            }
            Schedule::Lockstep => lockstep(p, k, cf_free, pair),
        }
    }
    p.batch_report = report(p, k_max, pair.filter(|_| k > 1));
}

/// `Schedule::Siblings`' grouping pass (see there).
fn pull(p: &mut Program, k_max: usize, cf_free: bool, pair: Option<Pairing>) {
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
            Kind::RotateSum(_) | Kind::Mult | Kind::Square => key(p, o, None, cf_free, pair),
            Kind::Bootstrap(_) if paired(p, o, pair) => None,
            Kind::Bootstrap(s) if p.bts_sites[*s as usize].route.slots == 0 && p.bts_sites[*s as usize].route.stc_first => key(p, o, None, cf_free, pair),
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

/// `Schedule::Siblings`' refresh pairs, before `pull`: the refreshes `pr` takes are matched, then each pair is issued
/// at its first member, with the second member's unissued ancestors brought forward; a matched refresh among those
/// ancestors comes with its own mate (and at most `PAIR_OPS` ops of the mate's ancestors) when it can. Candidates:
/// siblings of one key within `PAIR_WINDOW` launches whose ancestors after the first member are at most `PAIR_OPS` ops
/// and do not include it. The matching takes the refreshes with the fewest candidates first, each with its candidate of
/// fewest candidates (then fewest ops); a pair the issue order cannot keep (its second member issued earlier as another
/// pair's ancestor) is dropped from the candidates and the matching redone, `PAIR_ROUNDS` times at most, keeping the
/// order that forms the most pairs.
fn pull_pairs(p: &mut Program, pr: Pairing) {
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
    let keys: Vec<Option<String>> = ops.iter().map(|o| if paired(p, o, Some(pr)) { key(p, o, None, false, Some(pr)) } else { None }).collect();
    let mut bucket: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, k) in keys.iter().enumerate() {
        if let Some(k) = k {
            bucket.entry(k.as_str()).or_default().push(i);
        }
    }
    let ops: Vec<Option<Op>> = ops.into_iter().map(Some).collect();
    // the candidate graph on the order as it is (everything up to the first member counts as issued)
    let mut adj: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
    for list in bucket.values() {
        for (a, &i) in list.iter().enumerate() {
            let before: Vec<bool> = (0..n).map(|x| x <= i).collect();
            for &j in list[a + 1..].iter().filter(|&&j| j <= i + PAIR_WINDOW) {
                if let Some((_, c)) = unissued_ancestors(&preds, &[i], j, &before, &ops) {
                    adj.entry(i).or_default().push((j, c));
                    adj.entry(j).or_default().push((i, c));
                }
            }
        }
    }
    let mut banned: HashSet<(usize, usize)> = HashSet::new();
    let mut best: Option<(usize, Vec<Op>)> = None;
    for _ in 0..PAIR_ROUNDS {
        let mate = matching(&adj, &banned);
        let (out, formed, lost) = issue_pairs(&preds, &mate, ops.clone());
        if best.as_ref().is_none_or(|b| formed > b.0) {
            best = Some((formed, out));
        }
        if lost.is_empty() {
            break;
        }
        banned.extend(lost);
    }
    p.ops = best.map(|b| b.1).unwrap_or_else(|| ops.into_iter().flatten().collect());
}

/// Rounds of `pull_pairs`' matching.
const PAIR_ROUNDS: usize = 8;

/// j's unissued ancestors (launches) and their ops; None when one of `not` is among them or they exceed `PAIR_OPS` ops.
fn unissued_ancestors(preds: &[Vec<usize>], not: &[usize], j: usize, done: &[bool], ops: &[Option<Op>]) -> Option<(Vec<usize>, usize)> {
    let mut anc: Vec<usize> = vec![];
    let mut seen: HashSet<usize> = HashSet::new();
    let mut stack: Vec<usize> = preds[j].clone();
    let mut n_ops = 0;
    while let Some(a) = stack.pop() {
        if not.contains(&a) {
            return None;
        }
        if done[a] || !seen.insert(a) {
            continue;
        }
        n_ops += ops[a].as_ref().map_or(0, |o| o.members().len());
        if n_ops > PAIR_OPS {
            return None;
        }
        anc.push(a);
        stack.extend(preds[a].iter().copied());
    }
    Some((anc, n_ops))
}

/// A matching of the candidate graph without the `banned` edges (see `pull_pairs`).
fn matching(adj: &BTreeMap<usize, Vec<(usize, usize)>>, banned: &HashSet<(usize, usize)>) -> HashMap<usize, usize> {
    let mut mate: HashMap<usize, usize> = HashMap::new();
    let live = |x: usize, y: usize, mate: &HashMap<usize, usize>| !mate.contains_key(&y) && !banned.contains(&(x.min(y), x.max(y)));
    let degree = |x: usize, mate: &HashMap<usize, usize>| adj[&x].iter().filter(|(y, _)| live(x, *y, mate)).count();
    loop {
        let Some(u) = adj.keys().copied().filter(|&u| !mate.contains_key(&u) && degree(u, &mate) > 0).min_by_key(|&u| (degree(u, &mate), u)) else { break };
        let Some(&(v, _)) = adj[&u].iter().filter(|(y, _)| live(u, *y, &mate)).min_by_key(|&&(y, c)| (degree(y, &mate), c, y)) else { break };
        mate.insert(u, v);
        mate.insert(v, u);
    }
    mate
}

/// The launches in order with the matched pairs issued (see `pull_pairs`): the order, the pairs formed, the matched
/// pairs it could not keep (as (first, second) launch indices).
fn issue_pairs(preds: &[Vec<usize>], mate: &HashMap<usize, usize>, mut ops: Vec<Option<Op>>) -> (Vec<Op>, usize, Vec<(usize, usize)>) {
    let n = ops.len();
    let pair = |a: usize, b: usize, ops: &mut [Option<Op>]| -> Op {
        let m: Vec<Op> = [a, b].iter().filter_map(|&x| ops[x].take()).collect();
        let origin = m[0].origin.clone();
        Op { kind: Kind::Batched(m), ins: vec![], outs: vec![], pre: vec![], origin, inplace: false }
    };
    let (mut done, mut out, mut formed, mut lost) = (vec![false; n], Vec::with_capacity(n), 0, vec![]);
    for i in 0..n {
        if done[i] {
            continue;
        }
        done[i] = true;
        let Some(&j) = mate.get(&i).filter(|&&j| j > i) else {
            out.extend(ops[i].take());
            continue;
        };
        let Some((_, mut anc)) = Some(j).filter(|&j| !done[j]).and_then(|j| unissued_ancestors(preds, &[i], j, &done, &ops)).map(|a| (j, a.0)) else {
            lost.push((i, j));
            out.extend(ops[i].take());
            continue;
        };
        anc.sort_unstable();
        for a in anc {
            if done[a] {
                continue;
            }
            done[a] = true;
            // a matched refresh among them comes with its mate (and the mate's ancestors) when it can
            let nested = mate.get(&a).copied().filter(|&b| b != i && b != j && !done[b]).and_then(|b| unissued_ancestors(preds, &[i, j, a], b, &done, &ops).map(|(x, _)| (b, x)));
            match nested {
                Some((b, mut bx)) => {
                    bx.sort_unstable();
                    for c in bx {
                        done[c] = true;
                        out.extend(ops[c].take());
                    }
                    done[b] = true;
                    out.push(pair(a, b, &mut ops));
                    formed += 1;
                }
                None => out.extend(ops[a].take()),
            }
        }
        done[j] = true;
        out.push(pair(i, j, &mut ops));
        formed += 1;
    }
    (out, formed, lost.into_iter().map(|(a, b)| (a.min(b), a.max(b))).collect())
}

fn lockstep(p: &mut Program, k_max: usize, cf_free: bool, pair: Option<Pairing>) {
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
            key(p, o, None, cf_free, pair).map(|k| {
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
            let k_i = ops[i].as_ref().map_or(k_max, |o| cap(p, o, k_max, pair));
            let set = by_key.entry(k).or_default();
            set.remove(&(asap[i], i));
            while group.len() < k_i {
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

/// `pair`: the refresh pairing the batching ran with.
pub fn report(p: &Program, k_max: u32, pair: Option<Pairing>) -> BatchReport {
    let mut r = BatchReport { k_max, pair, ..Default::default() };
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
            r.pairs += (m.len() == PAIR_K && m.iter().all(|x| paired(p, x, pair))) as usize;
        }
    }
    r.per_kind = per.into_iter().map(|(kind, (ops, launches))| KindCount { kind: kind.to_string(), ops, launches }).collect();
    r
}
