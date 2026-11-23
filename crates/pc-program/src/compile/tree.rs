//! Sums in two halves (PC_SUM_TREE=1): a chain of adds `acc_j = acc_{j-1} + t_j` over K >= 4 terms (every partial
//! sum read only by the next add, terms and sums on one metadata, no preludes, no term a plaintext product: those
//! chains fuse into one pass at bind) rewired so the first ceil(K / 2) terms are summed, the rest are summed apart, and
//! one add joins the two halves. The same ops and vars, each var at its metadata (the plan's pins hold), the same value
//! (sums mod q are exact in any order); only the add that joins moves, after the last term. With a second device
//! holding the last half of a linear's outputs (ownership), each device sums its own blocks and one ciphertext crosses:
//! a norm statistic's all-reduce.
use crate::program::{Kind, Program, VarId};
use std::collections::HashMap;

/// PC_SUM_TREE: `1` / `on` = on; `0` / `off` = off; unset = on with a remote device (PC_REMOTE_DEVICE), off without.
pub fn on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| match std::env::var("PC_SUM_TREE").as_deref().map(str::trim) {
        Ok("1") | Ok("on") => true,
        Ok("0") | Ok("off") => false,
        _ => std::env::var("PC_REMOTE_DEVICE").is_ok_and(|v| !v.trim().is_empty()),
    })
}

struct Chain {
    terms: Vec<VarId>,
    adds: Vec<usize>,
}

/// Rewire in place; returns the chains rewired.
pub fn rewire(p: &mut Program) -> usize {
    let mut uses: HashMap<VarId, usize> = HashMap::new();
    for op in &p.ops {
        for v in op.all_ins() {
            *uses.entry(v).or_default() += 1;
        }
    }
    for &v in &p.outputs {
        *uses.entry(v).or_default() += 2;
    }
    let mut prod: HashMap<VarId, usize> = HashMap::new();
    for (i, op) in p.ops.iter().enumerate() {
        for v in op.all_outs() {
            prod.insert(v, i);
        }
    }
    let same = |a: VarId, b: VarId| {
        let (x, y) = (&p.vars[a as usize].meta, &p.vars[b as usize].meta);
        x.level == y.level && x.deg == y.deg && x.nf.to_bits() == y.nf.to_bits()
    };
    let pt_product = |v: VarId| prod.get(&v).is_some_and(|&i| matches!(p.ops[i].kind, Kind::MultPt(_) | Kind::AddPt(_)));
    let mut chains: Vec<Chain> = vec![];
    let mut head_of: HashMap<VarId, usize> = HashMap::new();
    for (i, op) in p.ops.iter().enumerate() {
        if !matches!(op.kind, Kind::Add) || op.ins.len() != 2 || op.outs.len() != 1 || !op.pre.is_empty() || op.ins[0] == op.ins[1] {
            continue;
        }
        let (x, y, out) = (op.ins[0], op.ins[1], op.outs[0]);
        if !same(out, x) || !same(out, y) {
            continue;
        }
        let ext = |c: VarId, t: VarId| -> Option<(usize, VarId)> {
            let &ci = head_of.get(&c)?;
            (uses.get(&c) == Some(&1) && !pt_product(t)).then_some((ci, t))
        };
        match ext(x, y).or_else(|| ext(y, x)) {
            Some((ci, t)) => {
                head_of.retain(|_, &mut c| c != ci);
                chains[ci].terms.push(t);
                chains[ci].adds.push(i);
                head_of.insert(out, ci);
            }
            None if !pt_product(x) && !pt_product(y) => {
                chains.push(Chain { terms: vec![x, y], adds: vec![i] });
                head_of.insert(out, chains.len() - 1);
            }
            None => {}
        }
    }
    // the joining add (op index) moves right after the chain's last add
    let mut moved: HashMap<usize, usize> = HashMap::new();
    let mut n = 0;
    for c in chains.iter().filter(|c| c.terms.len() >= 4) {
        let k = c.terms.len();
        let h = k - k / 2;
        let a = |j: usize| c.adds[j - 1];
        // acc_j = outs of a(j), j = 1..k-1
        let acc: Vec<VarId> = c.adds.iter().map(|&i| p.ops[i].outs[0]).collect();
        let near = if h == 1 { c.terms[0] } else { acc[h - 2] };
        // the far half's chain at a(h+1) .. a(k-1): acc_h .. acc_{k-2}
        p.ops[a(h + 1)].ins = vec![c.terms[h], c.terms[h + 1]];
        p.ops[a(h + 1)].outs = vec![acc[h - 1]];
        for j in h + 2..k {
            p.ops[a(j)].ins = vec![acc[j - 3], c.terms[j]];
            p.ops[a(j)].outs = vec![acc[j - 2]];
        }
        // the join: a(h), after a(k-1), into acc_{k-1} (what the chain's readers read)
        p.ops[a(h)].ins = vec![near, acc[k - 3]];
        p.ops[a(h)].outs = vec![acc[k - 2]];
        for j in h..k {
            p.ops[a(j)].inplace = false;
        }
        moved.insert(a(h), a(k - 1));
        n += 1;
    }
    if n > 0 {
        let after: HashMap<usize, usize> = moved.iter().map(|(&f, &t)| (t, f)).collect();
        let mut ops: Vec<Option<crate::program::Op>> = std::mem::take(&mut p.ops).into_iter().map(Some).collect();
        let mut out = Vec::with_capacity(ops.len());
        for i in 0..ops.len() {
            if moved.contains_key(&i) {
                continue;
            }
            if let Some(op) = ops[i].take() {
                out.push(op);
            }
            if let Some(&f) = after.get(&i) {
                if let Some(op) = ops[f].take() {
                    out.push(op);
                }
            }
        }
        p.ops = out;
    }
    n
}
