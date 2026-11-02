// Fused CtS / StC stage core (cts.cuh, CtsArgs): hoisted baby rotations of the ModUp'd input and the BSGS plaintext
// dot, the rotations never stored. For row t, baby i >= 1 is computed in source order (key products at slot s) and
// gathered through s = sigma_i^-1(o); baby 0 is P x on Q rows (0 on special rows). The regen form reads `b` through
// the KeyTab and expands `a` (the KSKA lane) for its first nregen digits. Canonical outputs, so the device's partial
// reductions every 8 baby steps leave no trace.
use crate::arith::{automorph_slot, ptidx, Hw};
use crate::expand::{expand_row, seed_at, TAG_KSKA};
use crate::ks::{b_row, slot_of};
use crate::launch;
use crate::raw::{el, par_for};
use pc_kernels::args::*;

type S = CudaStream;

fn rowp<W>(p: *const std::ffi::c_void, off: usize) -> &'static [W] {
    // SAFETY: a row of N words inside the launch's buffers
    unsafe { std::slice::from_raw_parts((p as *const W).add(off), N) }
}

fn run<W: Hw>(r: &CtsRegenArgs, regen: bool) {
    let a = &r.cts;
    par_for(a.nrows as usize, r, |r, y| {
        let (a, kt) = (&r.cts, &r.key);
        let t = el::<u32>(a.rows as _, y) as usize;
        // SAFETY: prime table entry t
        let q = unsafe { &*a.pr.add(t) };
        let p = W::from_u64(q.p);
        let isq = (t as i32) < a.nq;
        let prow = if isq { t } else { a.pt_prow as usize + (t - a.nq as usize) };
        let dt = if isq { el::<i32>(a.digit_of as _, t) as i64 } else { -1 };
        let (pm, pms) = (el::<W>(a.pmod, t), el::<W>(a.pmods, t));
        let (k, g, b) = (a.k as usize, a.g as usize, a.b as usize);
        let rowoff = t * N;
        let x = |c: usize, part: usize| rowp::<W>(a.x, c * a.ct_stride as usize + part * a.part_stride as usize + rowoff);
        let mut acc0 = vec![W::Acc::default(); k * g * N];
        let mut acc1 = vec![W::Acc::default(); k * g * N];
        let (mut v0, mut v1) = (vec![W::default(); k * N], vec![W::default(); k * N]);
        let (mut kb, mut ka) = (vec![W::default(); N], vec![W::default(); N]);
        let po: Vec<usize> = (0..N).map(|o| ptidx(o, a.enc)).collect();
        for i in 0..b {
            if i == 0 {
                for c in 0..k {
                    let (x0, x1) = (x(c, 0), x(c, 1));
                    for o in 0..N {
                        let (r0, r1) = if isq { (W::mul_shoup(x0[o], pm, pms, p), W::mul_shoup(x1[o], pm, pms, p)) } else { (W::default(), W::default()) };
                        v0[c * N + o] = r0;
                        v1[c * N + o] = r1;
                    }
                }
            } else {
                let (mut s0, mut s1) = (vec![W::Acc::default(); k * N], vec![W::Acc::default(); k * N]);
                for d in 0..a.ndig as usize {
                    if regen {
                        let kbp = el::<*const std::ffi::c_void>(kt.kb as _, i - 1);
                        b_row::<W>(kt, kbp, d * kt.srows as usize + slot_of(kt, t, a.nq as usize), &mut kb);
                        if (d as i32) < kt.nregen {
                            expand_row::<W>(&seed_at(el::<*const u32>(kt.seed as _, i - 1)), d as u32, q.p, TAG_KSKA, &mut ka);
                        } else {
                            let kap = el::<*const std::ffi::c_void>(a.ka as _, i - 1);
                            ka.copy_from_slice(rowp::<W>(kap, rowoff + d * a.key_digit_stride as usize));
                        }
                    } else {
                        let (kbp, kap) = (el::<*const std::ffi::c_void>(a.kb as _, i - 1), el::<*const std::ffi::c_void>(a.ka as _, i - 1));
                        kb.copy_from_slice(rowp::<W>(kbp, rowoff + d * a.key_digit_stride as usize));
                        ka.copy_from_slice(rowp::<W>(kap, rowoff + d * a.key_digit_stride as usize));
                    }
                    for c in 0..k {
                        let xv = if d as i64 == dt {
                            x(c, 1)
                        } else {
                            rowp::<W>(a.modup, c * a.mu_ct_stride as usize + d * a.mu_digit_stride as usize + rowoff)
                        };
                        let (o0, o1) = (&mut s0[c * N..(c + 1) * N], &mut s1[c * N..(c + 1) * N]);
                        for s in 0..N {
                            o0[s] += W::wmul(xv[s], kb[s]);
                            o1[s] += W::wmul(xv[s], ka[s]);
                        }
                    }
                }
                let kinv = el::<u32>(a.kinv as _, i - 1);
                for c in 0..k {
                    let x0 = x(c, 0);
                    for o in 0..N {
                        let s = automorph_slot(kinv, o as u32, LOG_N) as usize;
                        let mut r0 = W::reduce(s0[c * N + s], q);
                        if isq {
                            r0 = W::add(r0, W::mul_shoup(x0[s], pm, pms, p), p);
                        }
                        v0[c * N + o] = r0;
                        v1[c * N + o] = W::reduce(s1[c * N + s], q);
                    }
                }
            }
            for j in 0..g {
                let dg = b * j + i;
                if dg >= a.npts as usize {
                    continue;
                }
                // SAFETY: diagonal dg's rows (ptw words each)
                let pt = unsafe { (*(a.pts as *const *const W).add(dg)).add(prow * a.ptw as usize) };
                for c in 0..k {
                    let base = (c * g + j) * N;
                    for o in 0..N {
                        // SAFETY: compact index inside the diagonal's row
                        let w = unsafe { *pt.add(po[o]) };
                        acc0[base + o] += W::wmul(v0[c * N + o], w);
                        acc1[base + o] += W::wmul(v1[c * N + o], w);
                    }
                }
            }
            if i & 7 == 7 {
                // the device's headroom reductions (canonical either way; keeps the host accumulators bounded too)
                for v in acc0.iter_mut().chain(acc1.iter_mut()) {
                    *v = W::Acc::default() + W::wmul(W::reduce(*v, q), W::from_u64(1));
                }
            }
        }
        for c in 0..k {
            for j in 0..g {
                let at = if a.pair_cts > 0 { (j >> 1) * a.pair_cts as usize + (j & 1) } else { c * g + j };
                let res = (a.res as *mut W).wrapping_add(at * a.ct_stride as usize + rowoff);
                let base = (c * g + j) * N;
                for o in 0..N {
                    // SAFETY: result rows of (c, j)
                    unsafe {
                        *res.add(o) = W::reduce(acc0[base + o], q);
                        *res.add(a.part_stride as usize + o) = W::reduce(acc1[base + o], q);
                    }
                }
            }
        }
    })
}

/// The launcher's checks (cts.cuh launch_any): G <= 2, or <= 4 for the K = 1 regen launch with a pair layout.
fn check<W: Hw>(r: &CtsRegenArgs, regen: bool) -> i32 {
    let a = &r.cts;
    let dmax = if W::BITS == 32 { 6 } else { 8 };
    let km = if a.kmax <= 1 { 1 } else if a.kmax <= 2 { 2 } else if a.kmax <= 4 { 4 } else { 8 };
    let g4 = regen && W::BITS == 32 && km == 1 && a.k == 1 && a.pair_cts > 0;
    if a.g > if g4 { 4 } else { 2 } || a.ndig > dmax || (a.pair_cts > 0 && !g4) {
        return -1;
    }
    if regen && W::BITS == 32 && a.b > 64 {
        return -1;
    }
    0
}

pub extern "C" fn cts_fused<W: Hw>(a: *const CtsArgs, s: S) -> i32 {
    // SAFETY: the launcher ABI hands a valid argument struct
    let mut r: CtsRegenArgs = unsafe { std::mem::zeroed() };
    // SAFETY: as above
    r.cts = unsafe { *a };
    launch(&r, s, |r| check::<W>(r, false), |r| run::<W>(r, false))
}

pub extern "C" fn cts_fused_regen<W: Hw>(a: *const CtsRegenArgs, s: S) -> i32 {
    let bad = |r: &CtsRegenArgs| {
        if r.key.bits < 0 || r.key.bits > 32 || (r.key.bits > 0 && W::BITS != 32) || r.cts.ndig > if W::BITS == 32 { 6 } else { 8 } {
            -1
        } else {
            check::<W>(r, true)
        }
    };
    launch(a, s, bad, |r| run::<W>(r, true))
}
