//! Fused relinearization-rescales: a product (`Mult` / `Square`) whose value only flows, through an optional doubling
//! `Add(v, v)` and an optional `AddConst`, into an explicit `Rescale` (a Chebyshev doubling `T_2k = 2 T_k^2 - 1`, a
//! squared shared power) becomes one `MultRescale` (the relinearization's ModDown fused with the rescale, the doubling
//! in the tensor) followed by the `AddConst` on the rescaled value. The same value up to one rounding in place of two
//! (`pc_gpu::relin::RelinRescaler`): not bit-identical.
use super::check::rule;
use crate::program::{Kind, Op, Program, Var, VarId};
use pc_shadow::Shadow;
use std::collections::HashMap;

/// The deepest level a fused product may sit at (the runtime's tables start at top 6).
pub const MAX_LEVEL: u32 = 44;

pub fn fuse(p: &mut Program, sh: &Shadow) -> usize {
    let mut uses: HashMap<VarId, usize> = HashMap::new();
    for op in &p.ops {
        for v in op.all_ins() {
            *uses.entry(v).or_default() += 1;
        }
    }
    for &v in &p.outputs {
        *uses.entry(v).or_default() += 2;
    }
    let mut cons: HashMap<VarId, Vec<usize>> = HashMap::new();
    for (i, op) in p.ops.iter().enumerate() {
        for v in op.all_ins() {
            cons.entry(v).or_default().push(i);
        }
    }
    let only = |v: VarId, cons: &HashMap<VarId, Vec<usize>>| -> Option<usize> {
        let c = cons.get(&v)?;
        (c.iter().all(|&x| x == c[0])).then_some(c[0])
    };
    // (product op, removed ops, fused output var, the AddConst kept: (its op index, its output var))
    let mut plans: Vec<(usize, Vec<usize>, VarId, bool, Option<usize>, Vec<VarId>)> = vec![];
    for (i, op) in p.ops.iter().enumerate() {
        let square = match op.kind {
            Kind::Square if op.ins.len() == 1 => true,
            Kind::Mult if op.ins.len() == 2 && op.ins[0] != op.ins[1] => false,
            _ => continue,
        };
        if op.outs.len() != 1 || p.vars[op.ins[0] as usize].meta.level > MAX_LEVEL {
            continue;
        }
        let v = op.outs[0];
        let Some(mut j) = only(v, &cons) else { continue };
        let mut gone = vec![];
        let mut dbl = false;
        let mut kop = None;
        let c = &p.ops[j];
        if matches!(c.kind, Kind::Add) && c.pre.is_empty() && c.ins.len() == 2 && c.ins[0] == v && c.ins[1] == v && uses.get(&v) == Some(&2) {
            dbl = true;
            gone.push(j);
            let cur = c.outs[0];
            let Some(n) = only(cur, &cons) else { continue };
            if uses.get(&cur) != Some(&1) {
                continue;
            }
            j = n;
        } else if uses.get(&v) != Some(&1) {
            continue;
        }
        let c = &p.ops[j];
        if matches!(c.kind, Kind::AddConst(_)) && c.pre.is_empty() && c.ins.len() == 1 {
            kop = Some(j);
            let cur = c.outs[0];
            let Some(n) = only(cur, &cons) else { continue };
            if uses.get(&cur) != Some(&1) {
                continue;
            }
            j = n;
        }
        let c = &p.ops[j];
        if !matches!(c.kind, Kind::Rescale) || c.ins.len() != 1 || c.outs.len() != 1 {
            continue;
        }
        let r = c.outs[0];
        // the fused op's output: the rescale's output (no constant), else a new var at its meta the AddConst reads
        let ins_m: Vec<_> = op.ins.iter().map(|&x| p.vars[x as usize].meta).collect();
        let Ok((_, outs)) = rule(sh, &Kind::MultRescale { square, dbl, fused: vec![] }, &ins_m, &p.pt_refs, &p.bts_sites) else { continue };
        let want = p.vars[r as usize].meta;
        if outs[0].level != want.level || outs[0].deg != want.deg || outs[0].nf.to_bits() != want.nf.to_bits() || outs[0].level > MAX_LEVEL + sh.unit() {
            continue;
        }
        gone.push(j);
        // the vars never materialized: the product, the doubling, the constant's output (the rescale's input)
        let mut fz = vec![v];
        fz.extend(gone.iter().filter(|&&g| g != j).map(|&g| p.ops[g].outs[0]));
        if let Some(k) = kop {
            fz.push(p.ops[k].outs[0]);
        }
        plans.push((i, gone, r, dbl, kop, fz));
    }
    if plans.is_empty() {
        return 0;
    }
    let n = plans.len();
    let mut remove = vec![false; p.ops.len()];
    let mut replace: HashMap<usize, Vec<Op>> = HashMap::new();
    for (i, gone, r, dbl, kop, fz) in plans {
        let op = p.ops[i].clone();
        let square = matches!(op.kind, Kind::Square);
        let out = match kop {
            Some(_) => {
                let w = p.vars.len() as VarId;
                let name = format!("{}~rr", p.vars[r as usize].name);
                p.vars.push(Var { name, meta: p.vars[r as usize].meta, slot: None });
                w
            }
            None => r,
        };
        let mut seq = vec![Op { kind: Kind::MultRescale { square, dbl, fused: fz }, ins: op.ins.clone(), outs: vec![out], pre: op.pre.clone(), origin: op.origin.clone(), inplace: false }];
        if let Some(k) = kop {
            let mut c = p.ops[k].clone();
            c.ins = vec![out];
            c.outs = vec![r];
            c.pre = vec![];
            seq.push(c);
            remove[k] = true;
        }
        for g in gone {
            remove[g] = true;
        }
        replace.insert(i, seq);
    }
    let ops = std::mem::take(&mut p.ops);
    let mut out = Vec::with_capacity(ops.len());
    for (i, op) in ops.into_iter().enumerate() {
        if let Some(seq) = replace.remove(&i) {
            out.extend(seq);
        } else if !remove[i] {
            out.push(op);
        }
    }
    p.ops = out;
    n
}

/// A product whose one reader pays its rescale as a prelude (`Rescale` its only prelude on that operand) becomes a
/// `MultRescale` the reader reads with no prelude on it (its recipe re-derived and unchanged otherwise, its output
/// meta unchanged). Run after `fuse`.
pub fn fuse_preludes(p: &mut Program, sh: &Shadow) -> usize {
    use pc_shadow::{Operand, Prelude};
    let pos_of = |o: Operand| if o == Operand::A { 0usize } else { 1 };
    let mut uses: HashMap<VarId, usize> = HashMap::new();
    for op in &p.ops {
        for v in op.all_ins() {
            *uses.entry(v).or_default() += 1;
        }
    }
    for &v in &p.outputs {
        *uses.entry(v).or_default() += 2;
    }
    let mut reader: HashMap<VarId, usize> = HashMap::new();
    for (i, op) in p.ops.iter().enumerate() {
        for v in op.all_ins() {
            reader.insert(v, i);
        }
    }
    let mut n = 0;
    for i in 0..p.ops.len() {
        let op = &p.ops[i];
        let square = match op.kind {
            Kind::Square if op.ins.len() == 1 => true,
            Kind::Mult if op.ins.len() == 2 && op.ins[0] != op.ins[1] => false,
            _ => continue,
        };
        if op.outs.len() != 1 {
            continue;
        }
        let v = op.outs[0];
        if uses.get(&v) != Some(&1) {
            continue;
        }
        let Some(&j) = reader.get(&v) else { continue };
        let c = &p.ops[j];
        if matches!(c.kind, Kind::Batched(_) | Kind::Rescale | Kind::Bootstrap(_) | Kind::Finish { .. } | Kind::CoeffSum { .. } | Kind::MultRescale { .. }) || c.outs.len() != 1 {
            continue;
        }
        let Some(k) = c.ins.iter().take(2).position(|&x| x == v) else { continue };
        let on: Vec<&Prelude> = c.pre.iter().filter(|x| match x {
            Prelude::Rescale(o) | Prelude::DropTo(o, _) | Prelude::ScalarAdjust(o, _) => pos_of(*o) == k,
        }).collect();
        if on.len() != 1 || !matches!(on[0], Prelude::Rescale(_)) {
            continue;
        }
        let ins_m: Vec<_> = op.ins.iter().map(|&x| p.vars[x as usize].meta).collect();
        let Ok((_, outs)) = rule(sh, &Kind::MultRescale { square, dbl: false, fused: vec![] }, &ins_m, &p.pt_refs, &p.bts_sites) else { continue };
        let wm = outs[0];
        if wm.level > MAX_LEVEL + sh.unit() {
            continue;
        }
        let mut cins = c.ins.clone();
        let w = p.vars.len() as VarId;
        cins[k] = w;
        let metas: Vec<_> = cins.iter().map(|&x| if x == w { wm } else { p.vars[x as usize].meta }).collect();
        let Ok((npre, nouts)) = rule(sh, &c.kind, &metas, &p.pt_refs, &p.bts_sites) else { continue };
        let want = p.vars[c.outs[0] as usize].meta;
        if nouts[0].level != want.level || nouts[0].deg != want.deg || nouts[0].nf.to_bits() != want.nf.to_bits() {
            continue;
        }
        let old: Vec<Prelude> = c.pre.iter().copied().filter(|x| !matches!(x, Prelude::Rescale(o) if pos_of(*o) == k)).collect();
        if npre != old {
            continue;
        }
        let name = format!("{}~rr", p.vars[v as usize].name);
        p.vars.push(Var { name, meta: wm, slot: None });
        let op = &mut p.ops[i];
        op.kind = Kind::MultRescale { square, dbl: false, fused: vec![v] };
        op.outs = vec![w];
        let c = &mut p.ops[j];
        c.ins = cins;
        c.pre = npre;
        n += 1;
    }
    n
}
