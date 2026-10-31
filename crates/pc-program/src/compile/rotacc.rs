//! Rotation accumulations: `acc_j = acc_{j-1} + rot(t_j, s_j)`, j = 1..m (a BSGS linear's giant steps: each baby sum
//! rotated by its giant step and added onto the output). `sink` moves every link but the last next to the last one
//! (each rotation with its add, in chain order): the program computes the same values in another order (the links
//! read nothing produced after the last one's rotation, and nothing outside the chain reads a link's values). The
//! contiguous links then bind as one unit (`Program::rot_acc_chains`, bind `UnitKind::RotAcc`), whose rotations sum in
//! the extended basis with one ModDown onto `acc_0` (pc-gpu `rotate::rotate_acc`).
use crate::program::{Kind, Op, Program, VarId};
use std::collections::HashMap;

/// One link: the rotation's and the add's op indices, the rotated input, the accumulator read, the add's output.
#[derive(Clone, Copy, Debug)]
struct Link {
    rot: usize,
    add: usize,
    acc: VarId,
    out: VarId,
}

fn uses(p: &Program) -> HashMap<VarId, usize> {
    let mut u: HashMap<VarId, usize> = HashMap::new();
    for op in &p.ops {
        for v in op.all_ins() {
            *u.entry(v).or_default() += 1;
        }
    }
    for &v in &p.outputs {
        *u.entry(v).or_default() += 1;
    }
    u
}

fn plain_rot(op: &Op) -> Option<i64> {
    match op.kind {
        Kind::Rotate(k) if k != 0 && op.pre.is_empty() && op.ins.len() == 1 && op.outs.len() == 1 => Some(k),
        _ => None,
    }
}

fn plain_add(op: &Op) -> bool {
    matches!(op.kind, Kind::Add) && op.pre.is_empty() && op.ins.len() == 2 && op.outs.len() == 1 && op.ins[0] != op.ins[1]
}

/// The links of `p` (an add of a single-use rotation onto a single-use accumulator, both at the rotation input's
/// level, degree and scale) keyed by the add's output.
fn links(p: &Program) -> HashMap<VarId, Link> {
    let u = uses(p);
    let mut prod: HashMap<VarId, usize> = HashMap::new();
    for (i, op) in p.ops.iter().enumerate() {
        for v in &op.outs {
            prod.insert(*v, i);
        }
    }
    let same = |a: VarId, b: VarId| {
        let (x, y) = (&p.vars[a as usize].meta, &p.vars[b as usize].meta);
        x.level == y.level && x.deg == y.deg && x.nf.to_bits() == y.nf.to_bits()
    };
    let mut out = HashMap::new();
    for (i, op) in p.ops.iter().enumerate() {
        if !plain_add(op) {
            continue;
        }
        for (r, a) in [(op.ins[1], op.ins[0]), (op.ins[0], op.ins[1])] {
            let Some(&ri) = prod.get(&r) else { continue };
            let rop = &p.ops[ri];
            if ri >= i || plain_rot(rop).is_none() || u.get(&r) != Some(&1) || u.get(&a) != Some(&1) {
                continue;
            }
            let t = rop.ins[0];
            if !same(t, a) || !same(r, a) {
                continue;
            }
            out.insert(op.outs[0], Link { rot: ri, add: i, acc: a, out: op.outs[0] });
            break;
        }
    }
    out
}

/// Sink every chain of two or more links next to its last link; returns the chains moved.
pub fn sink(p: &mut Program) -> usize {
    let ls = links(p);
    // chains: start at a link whose accumulator no link produces, follow the link reading each output
    let by_acc: HashMap<VarId, VarId> = ls.values().map(|l| (l.acc, l.out)).collect();
    let mut chains: Vec<Vec<Link>> = vec![];
    for l in ls.values() {
        if ls.contains_key(&l.acc) {
            continue;
        }
        let mut c = vec![*l];
        let mut cur = l.out;
        while let Some(next) = by_acc.get(&cur) {
            let n = ls[next];
            c.push(n);
            cur = n.out;
        }
        // every earlier link's ops before the last rotation (they only move later, next to it)
        let last = c[c.len() - 1];
        if c.len() >= 2 && c[..c.len() - 1].iter().all(|l| l.rot < last.rot && l.add < last.rot) {
            chains.push(c);
        }
    }
    if chains.is_empty() {
        return 0;
    }
    // new order: drop the moved ops, re-insert them before the last link's rotation, the last add right after it
    let n = p.ops.len();
    let mut moved = vec![false; n];
    let mut before: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut after: HashMap<usize, usize> = HashMap::new();
    for c in &chains {
        let last = c[c.len() - 1];
        let mut seq = vec![];
        for l in &c[..c.len() - 1] {
            seq.push(l.rot);
            seq.push(l.add);
        }
        for &i in &seq {
            moved[i] = true;
        }
        moved[last.add] = true;
        before.insert(last.rot, seq);
        after.insert(last.rot, last.add);
    }
    let mut ops: Vec<Option<Op>> = std::mem::take(&mut p.ops).into_iter().map(Some).collect();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if let Some(seq) = before.get(&i) {
            for &j in seq {
                out.extend(ops[j].take());
            }
            out.extend(ops[i].take());
            if let Some(&a) = after.get(&i) {
                out.extend(ops[a].take());
            }
            continue;
        }
        if moved[i] {
            continue;
        }
        out.extend(ops[i].take());
    }
    debug_assert!(ops.iter().all(|o| o.is_none()));
    p.ops = out;
    chains.len()
}
