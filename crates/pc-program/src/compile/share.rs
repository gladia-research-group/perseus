//! Shared rescales: a pending product several ops read, each paying its rescale as a prelude on its own copy (the
//! FLEXIBLEAUTO recipe: `Rescale` of the operand and nothing else on it), is rescaled ONCE by a `Rescale` op before
//! its first such reader, which then read the rescaled value with no prelude on it. The rescale is deterministic, so
//! every reader sees the same residues as before; the ops' recipes and output metas are re-derived (`check::rule`)
//! and must come out unchanged apart from the dropped prelude, else that reader keeps its own rescale.
use super::check::rule;
use crate::program::{Kind, Op, Origin, Program, Var, VarId};
use pc_shadow::{Operand, Prelude, Shadow};
use std::collections::BTreeMap;

/// Returns the rescales removed (readers that now read a shared one, less one per shared rescale).
pub fn share_rescales(p: &mut Program, sh: &Shadow) -> usize {
    let pos_of = |o: Operand| if o == Operand::A { 0usize } else { 1 };
    // readers per var: (op index, operand position) whose only prelude on that operand is its Rescale
    let mut readers: BTreeMap<VarId, Vec<(usize, usize)>> = BTreeMap::new();
    for (i, op) in p.ops.iter().enumerate() {
        if matches!(op.kind, Kind::Batched(_) | Kind::Rescale | Kind::Bootstrap(_) | Kind::Finish { .. } | Kind::CoeffSum { .. }) || op.outs.len() != 1 {
            continue;
        }
        if op.ins.len() == 2 && op.ins[0] == op.ins[1] {
            continue;
        }
        for (k, &v) in op.ins.iter().enumerate().take(2) {
            let on: Vec<&Prelude> = op
                .pre
                .iter()
                .filter(|x| match x {
                    Prelude::Rescale(o) | Prelude::DropTo(o, _) | Prelude::ScalarAdjust(o, _) => pos_of(*o) == k,
                })
                .collect();
            if on.len() == 1 && matches!(on[0], Prelude::Rescale(_)) {
                readers.entry(v).or_default().push((i, k));
            }
        }
    }
    let mut insert: BTreeMap<usize, Vec<Op>> = BTreeMap::new();
    let mut saved = 0;
    for (v, rs) in readers {
        if rs.len() < 2 {
            continue;
        }
        let vm = p.vars[v as usize].meta;
        let Ok((pre, outs)) = rule(sh, &Kind::Rescale, &[vm], &p.pt_refs, &p.bts_sites) else { continue };
        let w = p.vars.len() as VarId;
        let wname = format!("{}~rs", p.vars[v as usize].name);
        let wm = outs[0];
        let mut edits: Vec<(usize, Vec<VarId>, Vec<Prelude>)> = vec![];
        for &(i, k) in &rs {
            let op = &p.ops[i];
            let mut ins = op.ins.clone();
            ins[k] = w;
            let metas: Vec<_> = ins.iter().map(|&x| if x == w { wm } else { p.vars[x as usize].meta }).collect();
            let Ok((npre, nouts)) = rule(sh, &op.kind, &metas, &p.pt_refs, &p.bts_sites) else { continue };
            let want = p.vars[op.outs[0] as usize].meta;
            let o = nouts[0];
            if o.level != want.level || o.deg != want.deg || o.nf.to_bits() != want.nf.to_bits() {
                continue;
            }
            // the recipe minus the shared rescale, the rest unchanged
            let old: Vec<Prelude> = op.pre.iter().copied().filter(|x| !matches!(x, Prelude::Rescale(o) if pos_of(*o) == k)).collect();
            if npre != old {
                continue;
            }
            edits.push((i, ins, npre));
        }
        if edits.len() < 2 {
            continue;
        }
        saved += edits.len() - 1;
        p.vars.push(Var { name: wname.clone(), meta: wm, slot: None });
        let first = edits.iter().map(|x| x.0).min().unwrap_or(0);
        for (i, ins, npre) in edits {
            p.ops[i].ins = ins;
            p.ops[i].pre = npre;
        }
        insert.entry(first).or_default().push(Op { kind: Kind::Rescale, ins: vec![v], outs: vec![w], pre, origin: Origin::Realize(wname), inplace: false });
    }
    if insert.is_empty() {
        return 0;
    }
    let ops = std::mem::take(&mut p.ops);
    let mut out = Vec::with_capacity(ops.len() + insert.len());
    for (i, op) in ops.into_iter().enumerate() {
        if let Some(mut ins) = insert.remove(&i) {
            out.append(&mut ins);
        }
        out.push(op);
    }
    p.ops = out;
    saved
}
