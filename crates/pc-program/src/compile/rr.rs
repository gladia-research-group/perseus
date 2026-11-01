//! Fused relinearization-rescales: a product (`Mult` / `Square`) whose value only flows, through an optional doubling
//! `Add(v, v)` and an optional `AddConst`, into an explicit `Rescale` (a Chebyshev doubling `T_2k = 2 T_k^2 - 1`, a
//! squared shared power) becomes one `MultRescale` (the relinearization's ModDown fused with the rescale, the doubling
//! in the tensor) followed by the `AddConst` on the rescaled value. The same value up to one rounding in place of two
//! (`pc_gpu::relin::RelinRescaler`): not bit-identical.
use super::check::rule;
use crate::program::{Kind, Op, Program, Var, VarId};
use pc_shadow::{ChainShape, Shadow};
use std::collections::HashMap;

/// The deepest level a fused product may sit at on a chain of uniform levels (the runtime's tables start at top 6).
pub const MAX_LEVEL: u32 = 44;

/// Whether the runtime's tables (`pc_gpu::relin::RelinRescaler`) fuse the rescale that lands at `level` on `chain` (the
/// real chain's shape: the compile's shadow runs an alternating user region's levels on its uniform grid): on a chain
/// of uniform levels every one down to `MAX_LEVEL`; on an alternating user region (`ChainShape::singles`) every
/// rescale that drops one prime or two from a top limb of at least 3 (its levels sit on a virtual grid, so the deepest
/// level is the chain's, not `MAX_LEVEL`).
fn served(chain: &ChainShape, level: u32) -> bool {
    if chain.singles == 0 {
        return level <= MAX_LEVEL + chain.unit;
    }
    match (level.checked_sub(chain.unit).and_then(|l| chain.primes_dropped(l)), chain.primes_dropped(level)) {
        (Some(a), Some(b)) => (b == a + 1 || b == a + 2) && chain.top_max().checked_sub(a).is_some_and(|t| t >= 3),
        _ => false,
    }
}

pub fn fuse(p: &mut Program, sh: &Shadow, chain: &ChainShape) -> usize {
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
        if op.outs.len() != 1 || !served(chain, p.vars[op.ins[0] as usize].meta.level + sh.unit()) {
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
        if outs[0].level != want.level || outs[0].deg != want.deg || outs[0].nf.to_bits() != want.nf.to_bits() || !served(chain, outs[0].level) {
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
pub fn fuse_preludes(p: &mut Program, sh: &Shadow, chain: &ChainShape) -> usize {
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
        if !served(chain, wm.level) {
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

/// Eager rescales: a product (`Mult` / `Square`) whose value only flows through rescale-commuting ops (conjugations,
/// rotations, rotate-sums, negations, products by i, constant adds, and adds / subs of two such values) into readers
/// that each pay its rescale as their only prelude on it (a Goldschmidt round's `x y + conj(x y) + c`, a pending
/// square read by several products) becomes a `MultRescale` (fused where the runtime's tables serve the top, else its
/// relinearization and rescale one after the other): those ops then run a level lower on the rescaled values and the
/// readers drop their rescale. Rescaling commutes with them up to rounding (the constants encoded at the lower scale,
/// one rounding of the product in place of one per reader): the same value, not bit-identical. The original vars stay
/// in the fused list at their planned states (the plan's pins); the ops write new `~er` vars. Run after `fuse_preludes`.
pub fn early(p: &mut Program, sh: &Shadow) -> usize {
    use pc_shadow::{Operand, Prelude};
    use std::collections::{BTreeMap, BTreeSet};
    let pos_of = |o: Operand| if o == Operand::A { 0usize } else { 1 };
    let limit = sh.cfg.bts.level_limit;
    let outputs: BTreeSet<VarId> = p.outputs.iter().copied().collect();
    let consumers = |p: &Program| -> HashMap<VarId, Vec<usize>> {
        let mut cons: HashMap<VarId, Vec<usize>> = HashMap::new();
        for (i, op) in p.ops.iter().enumerate() {
            for v in op.all_ins() {
                let c = cons.entry(v).or_default();
                if c.last() != Some(&i) {
                    c.push(i);
                }
            }
        }
        cons
    };
    // a rescale-commuting op with no prelude: unary, or an add / sub of two region values (checked below; an outside
    // operand would need its own rescale, which measured slower than the readers' rescales it saves)
    let linear = |op: &Op| -> bool {
        op.pre.is_empty()
            && op.outs.len() == 1
            && match op.kind {
                Kind::Conjugate | Kind::Rotate(_) | Kind::RotateSum(_) | Kind::Negate | Kind::MultI | Kind::AddConst(_) => op.ins.len() == 1,
                Kind::Add | Kind::Sub => op.ins.len() == 2,
                _ => false,
            }
    };
    let mut cons = consumers(p);
    // region ops already rewritten (a rescaled value's ops are not rescaled again); a sink may later be a product
    let mut done: BTreeSet<usize> = BTreeSet::new();
    let mut n = 0;
    'prod: for i in 0..p.ops.len() {
        let op = &p.ops[i];
        let square = match op.kind {
            Kind::Square if op.ins.len() == 1 => true,
            Kind::Mult if op.ins.len() == 2 && op.ins[0] != op.ins[1] => false,
            _ => continue,
        };
        if op.outs.len() != 1 || done.contains(&i) {
            continue;
        }
        let v = op.outs[0];
        // the region: vars reachable through rescale-commuting ops; sinks: readers whose only prelude on it is a Rescale
        let mut vars = vec![v];
        let mut region: Vec<usize> = vec![];
        let mut sinks: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut q = 0;
        while q < vars.len() {
            let x = vars[q];
            q += 1;
            if outputs.contains(&x) || vars.len() > 24 {
                continue 'prod;
            }
            for &j in cons.get(&x).map(|c| c.as_slice()).unwrap_or(&[]) {
                let c = &p.ops[j];
                if done.contains(&j) {
                    continue 'prod;
                }
                if linear(c) {
                    if !region.contains(&j) {
                        region.push(j);
                        vars.push(c.outs[0]);
                    }
                    continue;
                }
                if matches!(c.kind, Kind::Batched(_) | Kind::Rescale | Kind::Bootstrap(_) | Kind::Finish { .. } | Kind::CoeffSum { .. } | Kind::MultRescale { .. } | Kind::MultSum { .. }) || c.outs.len() != 1 {
                    continue 'prod;
                }
                let ks: Vec<usize> = c.ins.iter().enumerate().filter(|(_, &y)| y == x).map(|(k, _)| k).collect();
                if ks.len() != 1 || ks[0] > 1 || (c.ins.len() == 2 && c.ins[0] == c.ins[1]) {
                    continue 'prod;
                }
                let k = ks[0];
                let on: Vec<&Prelude> = c.pre.iter().filter(|y| match y {
                    Prelude::Rescale(o) | Prelude::DropTo(o, _) | Prelude::ScalarAdjust(o, _) => pos_of(*o) == k,
                }).collect();
                if on.len() != 1 || !matches!(on[0], Prelude::Rescale(_)) {
                    continue 'prod;
                }
                let e = sinks.entry(j).or_default();
                if !e.contains(&k) {
                    e.push(k);
                }
            }
        }
        if sinks.is_empty() || (region.is_empty() && sinks.len() < 2) {
            continue;
        }
        let vset: BTreeSet<VarId> = vars.iter().copied().collect();
        region.sort();
        if region.iter().any(|&j| j < i || p.ops[j].ins.iter().any(|y| !vset.contains(y))) || sinks.keys().any(|&j| region.contains(&j) || j <= i) {
            continue;
        }
        // the product rescaled, then every region op re-derived on the rescaled values (no prelude), every sink without
        // its rescale and with its outputs unchanged
        let ins_m: Vec<_> = op.ins.iter().map(|&x| p.vars[x as usize].meta).collect();
        let Ok((ppre, pouts)) = rule(sh, &Kind::MultRescale { square, dbl: false, fused: vec![] }, &ins_m, &p.pt_refs, &p.bts_sites) else { continue };
        if ppre != op.pre || pouts[0].level >= limit {
            continue;
        }
        let mut map: BTreeMap<VarId, VarId> = BTreeMap::new();
        let mut metas: BTreeMap<VarId, pc_shadow::CtMeta> = BTreeMap::new();
        let mut names: Vec<(VarId, String)> = vec![];
        let mut next = p.vars.len() as VarId;
        let mut fresh = |x: VarId, map: &mut BTreeMap<VarId, VarId>, names: &mut Vec<(VarId, String)>| {
            let w = next;
            next += 1;
            map.insert(x, w);
            names.push((w, format!("{}~er", p.vars[x as usize].name)));
            w
        };
        let w = fresh(v, &mut map, &mut names);
        metas.insert(w, pouts[0]);
        let meta_of = |x: VarId, map: &BTreeMap<VarId, VarId>, metas: &BTreeMap<VarId, pc_shadow::CtMeta>| map.get(&x).and_then(|y| metas.get(y)).copied().unwrap_or(p.vars[x as usize].meta);
        for &j in &region {
            let c = &p.ops[j];
            let m: Vec<_> = c.ins.iter().map(|&x| meta_of(x, &map, &metas)).collect();
            let Ok((pre, outs)) = rule(sh, &c.kind, &m, &p.pt_refs, &p.bts_sites) else { continue 'prod };
            if !pre.is_empty() || outs[0].level >= limit {
                continue 'prod;
            }
            let o = fresh(c.outs[0], &mut map, &mut names);
            metas.insert(o, outs[0]);
        }
        let mut snk = vec![];
        for (&j, ks) in &sinks {
            let c = &p.ops[j];
            let m: Vec<_> = c.ins.iter().map(|&x| meta_of(x, &map, &metas)).collect();
            let Ok((npre, nouts)) = rule(sh, &c.kind, &m, &p.pt_refs, &p.bts_sites) else { continue 'prod };
            let old: Vec<Prelude> = c.pre.iter().copied().filter(|y| !matches!(y, Prelude::Rescale(o) if ks.contains(&pos_of(*o)))).collect();
            let same = nouts.len() >= c.outs.len()
                && c.outs.iter().zip(&nouts).all(|(&o, n)| {
                    let want = p.vars[o as usize].meta;
                    n.level == want.level && n.deg == want.deg && n.nf.to_bits() == want.nf.to_bits()
                });
            if npre != old || !same {
                continue 'prod;
            }
            snk.push((j, npre));
        }
        // apply
        for (w, name) in &names {
            p.vars.push(Var { name: name.clone(), meta: metas[w], slot: None });
        }
        let fused: Vec<VarId> = map.keys().copied().collect();
        let op = &mut p.ops[i];
        op.kind = Kind::MultRescale { square, dbl: false, fused };
        op.outs = vec![map[&v]];
        op.inplace = false;
        for &j in &region {
            let c = &mut p.ops[j];
            c.ins = c.ins.iter().map(|x| *map.get(x).unwrap_or(x)).collect();
            c.outs = c.outs.iter().map(|x| *map.get(x).unwrap_or(x)).collect();
            c.inplace = false;
            done.insert(j);
        }
        for (j, npre) in snk {
            let c = &mut p.ops[j];
            c.ins = c.ins.iter().map(|x| *map.get(x).unwrap_or(x)).collect();
            c.pre = npre;
            c.inplace = false;
        }
        done.insert(i);
        cons = consumers(p);
        n += 1;
    }
    n
}
