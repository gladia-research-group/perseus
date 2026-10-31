//! Buffer assignment: linear scan over the (batched) op order. A ciphertext at level `l` holds `n_q - l` limbs of
//! two polynomials (relinearized: degree 2 is a pending rescale, not a third polynomial). Every var gets a slot
//! when produced; a slot returns to the free list after its var's last read (block outputs are never freed, block
//! inputs are slots the driver writes). A new var takes the smallest free slot that fits (best fit), else a new
//! slot of its exact size. Elementwise single ops whose first input dies there run in place (the output takes that
//! input's slot). `Program.mem.bytes` = the arena the replay needs for ciphertexts.
use crate::program::{Kind, Mem, Program, VarId};
use std::collections::{BTreeMap, BTreeSet};

fn inplace_kind(k: &Kind) -> bool {
    matches!(
        k,
        Kind::Add | Kind::Sub | Kind::AddPt(_) | Kind::MultPt(_) | Kind::AddConst(_) | Kind::MultConst(_) | Kind::Negate | Kind::MultI | Kind::Rescale | Kind::DropTo(_)
    )
}

/// Assign slots with both policies (best fit on exact sizes; every slot at the largest var size, which never
/// fragments) and keep the smaller arena.
pub fn alloc(p: &mut Program, n_q: u32, bytes_per_limb: u64) {
    let mut q = p.clone();
    assign(p, n_q, bytes_per_limb, false);
    assign(&mut q, n_q, bytes_per_limb, true);
    if q.mem.bytes < p.mem.bytes {
        *p = q;
    }
}

fn assign(p: &mut Program, n_q: u32, bytes_per_limb: u64, uniform: bool) {
    let bytes_of = |p: &Program, v: VarId| -> u64 { 2 * (n_q.saturating_sub(p.vars[v as usize].meta.level)) as u64 * bytes_per_limb };
    let keep: BTreeSet<VarId> = p.outputs.iter().copied().collect();
    let mut last_use: BTreeMap<VarId, usize> = BTreeMap::new();
    for (i, op) in p.ops.iter().enumerate() {
        for v in op.all_ins() {
            last_use.insert(v, i);
        }
    }
    let mut slot_bytes: Vec<u64> = vec![];
    // free slots by size: (bytes, slot)
    let mut free: BTreeSet<(u64, u32)> = BTreeSet::new();
    let mut live: u64 = 0;
    let mut live_peak: u64 = 0;
    let max_bytes = (0..p.vars.len()).map(|v| bytes_of(p, v as VarId)).max().unwrap_or(0);
    let take = |p: &mut Program, v: VarId, slot_bytes: &mut Vec<u64>, free: &mut BTreeSet<(u64, u32)>, live: &mut u64| {
        let need = bytes_of(p, v);
        let s = match free.range((need, 0)..).next().copied() {
            Some(e) => {
                free.remove(&e);
                e.1
            }
            None => {
                slot_bytes.push(if uniform { max_bytes } else { need });
                (slot_bytes.len() - 1) as u32
            }
        };
        p.vars[v as usize].slot = Some(s);
        *live += need;
    };
    // a batch's outputs in consecutive slots, member order (the lowest run of free slots that fit, else new slots
    // after the free ones that end the list): a later batch over the same values reads them as one K-ciphertext
    // buffer, and this batch writes them as one (the replay's gathers find nothing to copy)
    let take_run = |p: &mut Program, vs: &[VarId], slot_bytes: &mut Vec<u64>, free: &mut BTreeSet<(u64, u32)>, live: &mut u64| {
        let need = vs.iter().map(|&v| bytes_of(p, v)).max().unwrap_or(0);
        let k = vs.len();
        let fits = |s: usize, free: &BTreeSet<(u64, u32)>| s < slot_bytes.len() && slot_bytes[s] >= need && free.contains(&(slot_bytes[s], s as u32));
        let mut slots: Vec<u32> = free.iter().map(|e| e.1).collect();
        slots.sort_unstable();
        let start = slots.iter().map(|&s| s as usize).find(|&s| (s..s + k).all(|t| fits(t, free))).unwrap_or_else(|| {
            // new slots, after the free slots that fit at the end of the list
            let mut s = slot_bytes.len();
            while s > 0 && slot_bytes.len() - s + 1 < k && fits(s - 1, free) {
                s -= 1;
            }
            s
        });
        for (j, &v) in vs.iter().enumerate() {
            let t = start + j;
            if t < slot_bytes.len() {
                free.remove(&(slot_bytes[t], t as u32));
            } else {
                slot_bytes.push(if uniform { max_bytes } else { need });
            }
            p.vars[v as usize].slot = Some(t as u32);
            *live += bytes_of(p, v);
        }
    };
    let inputs = p.inputs.clone();
    for v in inputs {
        take(p, v, &mut slot_bytes, &mut free, &mut live);
    }
    live_peak = live_peak.max(live);
    for i in 0..p.ops.len() {
        let single = !matches!(p.ops[i].kind, Kind::Batched(_));
        let dying: Vec<VarId> = {
            let mut d: Vec<VarId> = p.ops[i].all_ins().into_iter().filter(|v| last_use.get(v) == Some(&i) && !keep.contains(v)).collect();
            d.sort_unstable();
            d.dedup();
            d
        };
        let outs = p.ops[i].all_outs();
        let mut freed_inplace: Option<VarId> = None;
        if single && inplace_kind(&p.ops[i].kind) && outs.len() == 1 {
            let a = p.ops[i].ins[0];
            if dying.contains(&a) && bytes_of(p, outs[0]) <= bytes_of(p, a) {
                if let Some(s) = p.vars[a as usize].slot {
                    let o = outs[0];
                    p.vars[o as usize].slot = Some(s);
                    live += bytes_of(p, o);
                    live -= bytes_of(p, a);
                    p.ops[i].inplace = true;
                    freed_inplace = Some(a);
                }
            }
        }
        if freed_inplace.is_none() {
            if !single && outs.len() > 1 {
                take_run(p, &outs, &mut slot_bytes, &mut free, &mut live);
            } else {
                for &o in &outs {
                    take(p, o, &mut slot_bytes, &mut free, &mut live);
                }
            }
        }
        live_peak = live_peak.max(live);
        for v in dying {
            if Some(v) == freed_inplace {
                continue;
            }
            if let Some(s) = p.vars[v as usize].slot {
                free.insert((slot_bytes[s as usize], s));
                live -= bytes_of(p, v);
            }
        }
        // a var nothing reads (and not an output) dies at birth
        for &o in &outs {
            if !last_use.contains_key(&o) && !keep.contains(&o) {
                if let Some(s) = p.vars[o as usize].slot {
                    free.insert((slot_bytes[s as usize], s));
                    live -= bytes_of(p, o);
                }
            }
        }
    }
    let mut offsets = Vec::with_capacity(slot_bytes.len());
    let mut acc = 0u64;
    for &b in &slot_bytes {
        offsets.push(acc);
        acc += b;
    }
    p.mem = Mem { slot_bytes, offsets, bytes: acc, live_peak, bytes_per_limb, uniform };
}
