//! Lazy relinearization (config `exec.mult_sum`): a chain of relinearized products added up (`p_1 + p_2 + ...`,
//! each `p_j = Mult(a_j, b_j)` read only by the chain, every product and partial sum on one meta, the adds without
//! preludes) becomes one `Kind::MultSum` at the last add: the tensors accumulate (each pair with its own operand
//! preludes) and one relinearization serves the sum (pc-gpu `Ops::mult_sum`). The same value up to the key-switching
//! error: not bit-identical. The products' and partial sums' vars stay in `fused` with their lowered metadata (the
//! plan may pin those names); alloc places the operands' longer lives (they are read at the last add).
use crate::program::{Kind, Op, Program, VarId};
use std::collections::HashMap;

struct Chain {
    mults: Vec<usize>,
    adds: Vec<usize>,
    head: VarId,
}

/// Fuse in place; returns the chains fused.
pub fn fuse(p: &mut Program) -> usize {
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
    // a product the chain may absorb: an unbatched Mult of two distinct inputs whose one output only this add reads
    let product = |v: VarId| -> Option<usize> {
        let &i = prod.get(&v)?;
        let o = &p.ops[i];
        (matches!(o.kind, Kind::Mult) && o.ins.len() == 2 && o.ins[0] != o.ins[1] && o.outs.len() == 1 && uses.get(&v) == Some(&1)).then_some(i)
    };
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
        let extend = |c: VarId, t: VarId| -> Option<(usize, usize)> {
            let &ci = head_of.get(&c)?;
            if uses.get(&c) != Some(&1) {
                return None;
            }
            Some((ci, product(t)?))
        };
        if let Some((ci, m)) = extend(x, y).or_else(|| extend(y, x)) {
            let c = &mut chains[ci];
            head_of.remove(&c.head);
            c.mults.push(m);
            c.adds.push(i);
            c.head = out;
            head_of.insert(out, ci);
        } else if let (Some(mx), Some(my)) = (product(x), product(y)) {
            chains.push(Chain { mults: vec![mx, my], adds: vec![i], head: out });
            head_of.insert(out, chains.len() - 1);
        }
    }
    if chains.is_empty() {
        return 0;
    }
    let mut ops: Vec<Option<Op>> = std::mem::take(&mut p.ops).into_iter().map(Some).collect();
    for c in &chains {
        let mut ins = vec![];
        let mut fused = vec![];
        for &m in &c.mults {
            let o = ops[m].take().expect("a product in two chains");
            ins.extend_from_slice(&o.ins);
            fused.push(o.outs[0]);
        }
        let last = *c.adds.last().expect("a chain has an add");
        for &a in &c.adds[..c.adds.len() - 1] {
            let o = ops[a].take().expect("an add in two chains");
            fused.push(o.outs[0]);
        }
        let o = ops[last].take().expect("an add in two chains");
        ops[last] = Some(Op { kind: Kind::MultSum { fused }, ins, outs: vec![c.head], pre: vec![], origin: o.origin, inplace: false });
    }
    p.ops = ops.into_iter().flatten().collect();
    chains.len()
}
