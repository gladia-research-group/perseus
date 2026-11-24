//! Rotation cancellation (PC_ROT_CANCEL=1, default off until an A/B wins): a rotation by k whose value reaches only
//! slot-wise ops on its way to rotations by -k (a PCMM's outputs rotated +1, the GLU, the next PCMM's inputs rotated
//! -1) is dropped together with those rotations: the slot-wise ops run on the unrotated values.
//!
//! Slot-wise here = commutes with every rotation exactly as a map on slot vectors: add / sub / negate, products of two
//! ciphertexts, squares, rescales and level drops, scalar adds and products (one value on every slot), the
//! conjugation and the product by i (Galois automorphisms commute; X^{N/2} is fixed by every rotation). A plaintext
//! operand (a mask) is not, nor is a refresh, a finish or anything the block hands out. A component (rotations by k,
//! the slot-wise ops downstream of them, the cancelling rotations) is rewritten only when every slot-wise op in it
//! reads rotated operands alone, every reader of a rotated value is in it, and no rotated value leaves the block.
//!
//! Same values in slot terms; not the same words (two key switches fewer per path, the products' relinearisations
//! and the rescales' roundings on other polynomials): an arithmetic change, gated on accuracy.
use crate::program::{Kind, Program, VarId};
use std::collections::{BTreeMap, HashMap};

/// PC_ROT_CANCEL: `1` / `on` = on; unset / `0` / `off` = off.
pub fn on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| matches!(std::env::var("PC_ROT_CANCEL").as_deref().map(str::trim), Ok("1") | Ok("on")))
}

fn slotwise(k: &Kind) -> bool {
    matches!(
        k,
        Kind::Add | Kind::Sub | Kind::Negate | Kind::Mult | Kind::Square | Kind::MultRescale { .. } | Kind::Rescale | Kind::DropTo(_) | Kind::AddConst(_) | Kind::MultConst(_) | Kind::Conjugate | Kind::MultI
    )
}

fn find(p: &mut [usize], mut x: usize) -> usize {
    while p[x] != x {
        p[x] = p[p[x]];
        x = p[x];
    }
    x
}

/// Returns (rotations removed, components rewritten). `slots`: the ring's slot count (rotation steps mod it).
pub fn cancel(p: &mut Program, slots: i64) -> (usize, usize) {
    let norm = |k: i64| k.rem_euclid(slots);
    let mut removed = 0;
    let mut comps = 0;
    // each step k once (a component of +k rotations ends in -k rotations)
    let mut ks: Vec<i64> = p.ops.iter().filter_map(|o| match o.kind {
        Kind::Rotate(k) if norm(k) != 0 && o.pre.is_empty() && o.ins.len() == 1 && o.outs.len() == 1 => Some(norm(k)),
        _ => None,
    }).collect();
    ks.sort_unstable();
    ks.dedup();
    for k in ks {
        let (r, c) = cancel_k(p, k, slots);
        removed += r;
        comps += c;
    }
    (removed, comps)
}

fn cancel_k(p: &mut Program, k: i64, slots: i64) -> (usize, usize) {
    let norm = |x: i64| x.rem_euclid(slots);
    let n = p.ops.len();
    let nv = p.vars.len();
    let outputs: std::collections::HashSet<VarId> = p.outputs.iter().copied().collect();
    // per var: the component (op index of its union-find root) whose rotated value it holds
    let mut tag: Vec<Option<usize>> = vec![None; nv];
    let mut par: Vec<usize> = (0..n).collect();
    let mut bad: Vec<bool> = vec![false; n];
    let mut role: Vec<u8> = vec![0; n]; // 1 source, 2 slot-wise inside, 3 cancelling rotation
    let simple = |o: &crate::program::Op| o.pre.is_empty() && o.ins.len() == 1 && o.outs.len() == 1;
    for i in 0..n {
        let op = &p.ops[i];
        let tagged: Vec<usize> = op.all_ins().iter().filter_map(|&v| tag[v as usize]).collect();
        let is_src = matches!(op.kind, Kind::Rotate(s) if norm(s) == k) && simple(op);
        if tagged.is_empty() {
            if is_src {
                role[i] = 1;
                tag[op.outs[0] as usize] = Some(i);
            }
            continue;
        }
        // reads a rotated value: joins the components it reads
        let mut r = find(&mut par, tagged[0]);
        for &t in &tagged[1..] {
            let a = find(&mut par, t);
            if a != r {
                par[a] = r;
                bad[r] |= bad[a];
            }
            r = find(&mut par, r);
        }
        let cancels = matches!(op.kind, Kind::Rotate(s) if norm(s + k) == 0) && simple(op);
        let all_tagged = op.ins.iter().all(|&v| tag[v as usize].is_some());
        if cancels {
            role[i] = 3;
            par[i] = r;
            // its output stands for its input after the rewrite: never one the block hands out
            if outputs.contains(&op.outs[0]) {
                bad[r] = true;
            }
        } else if slotwise(&op.kind) && all_tagged && !matches!(op.kind, Kind::Batched(_)) {
            role[i] = 2;
            par[i] = r;
            for &o in &op.outs {
                tag[o as usize] = Some(i);
            }
        } else {
            bad[r] = true;
        }
    }
    // rotated values handed out of the block
    for v in 0..nv {
        if let Some(t) = tag[v] {
            if outputs.contains(&(v as VarId)) {
                let r = find(&mut par, t);
                bad[r] = true;
            }
        }
    }
    // a component needs a source and a cancelling rotation, and must be clean
    let mut has_src: HashMap<usize, bool> = HashMap::new();
    let mut has_end: HashMap<usize, bool> = HashMap::new();
    for i in 0..n {
        if role[i] == 0 {
            continue;
        }
        let r = find(&mut par, i);
        if role[i] == 1 {
            has_src.insert(r, true);
        }
        if role[i] == 3 {
            has_end.insert(r, true);
        }
    }
    let mut ok_root: BTreeMap<usize, bool> = BTreeMap::new();
    for i in 0..n {
        if role[i] == 0 {
            continue;
        }
        let r = find(&mut par, i);
        let ok = !bad[r] && has_src.get(&r).copied().unwrap_or(false) && has_end.get(&r).copied().unwrap_or(false);
        ok_root.insert(r, ok);
    }
    // rewrite: a removed rotation's output reads as its input
    let mut alias: HashMap<VarId, VarId> = HashMap::new();
    let mut drop: Vec<bool> = vec![false; n];
    for i in 0..n {
        if role[i] != 1 && role[i] != 3 {
            continue;
        }
        let r = find(&mut par, i);
        if !ok_root.get(&r).copied().unwrap_or(false) {
            continue;
        }
        let op = &p.ops[i];
        alias.insert(op.outs[0], op.ins[0]);
        drop[i] = true;
    }
    if alias.is_empty() {
        return (0, 0);
    }
    let resolve = |mut v: VarId, a: &HashMap<VarId, VarId>| {
        while let Some(&u) = a.get(&v) {
            v = u;
        }
        v
    };
    let ops = std::mem::take(&mut p.ops);
    let mut kept = Vec::with_capacity(ops.len());
    for (i, mut op) in ops.into_iter().enumerate() {
        if drop[i] {
            continue;
        }
        for v in op.ins.iter_mut() {
            *v = resolve(*v, &alias);
        }
        if let Kind::Batched(ms) = &mut op.kind {
            for m in ms.iter_mut() {
                for v in m.ins.iter_mut() {
                    *v = resolve(*v, &alias);
                }
            }
        }
        kept.push(op);
    }
    p.ops = kept;
    for v in p.names.values_mut() {
        *v = resolve(*v, &alias);
    }
    let comps = ok_root.values().filter(|&&x| x).count();
    (alias.len(), comps)
}
