// Composite rescale (rescale.cuh, RescaleArgs): rescale_coef writes the correction rows tmp_i = A_i lift(y_a)
// [+ B_i lift(y')] in coefficient form; rescale_fwd generates the same rows inside the forward NTT (ndrop 2), after
// rescale_prep stored y_a and y' in tmp rows top and top - 1.
use crate::arith::Hw;
use crate::launch;
use crate::ntt::fwd_with;
use crate::raw::{at, buf, el, par_for};
use crate::simd::w as v;
use pc_kernels::args::*;

type S = CudaStream;

#[inline(always)]
fn prime(pr: *const PrimeConst, id: usize) -> &'static PrimeConst {
    // SAFETY: prime table by id (device constant data)
    unsafe { &*pr.add(id) }
}

/// Rows of the two sources of coefficients [lo, lo + n) of (c, part): y_a (scaled by ca) and, at ndrop 2,
/// y' = qinv_ab y_b + C_ab lift(y_a) mod q_b.
fn sources<W: Hw>(a: &RescaleArgs, c: usize, part: usize, lo: usize, ya: &mut [W], yp: &mut [W], acc: &mut [W::Acc]) {
    let n = ya.len();
    let (ta, tb) = (a.top as usize, a.top as usize - 1);
    let pa = prime(a.pr, ta);
    // SAFETY: words [lo, lo + n) of the top rows
    let xa = unsafe { std::slice::from_raw_parts(at::<W>(&a.x, c, part, ta).add(lo), n) };
    if a.ca != 1 {
        v::mul_s::<W>(ya, xa, W::from_u64(a.ca), pa, acc);
    } else {
        ya.copy_from_slice(xa);
    }
    if a.ndrop != 2 {
        return;
    }
    let pb = prime(a.pr, tb);
    let qb = W::from_u64(pb.p);
    // SAFETY: as above
    let xb = unsafe { std::slice::from_raw_parts(at::<W>(&a.x, c, part, tb).add(lo), n) };
    let mut t = buf::<W>(n);
    if a.ca != 1 {
        v::mul_s::<W>(&mut t, xb, W::from_u64(a.cb), pb, acc);
        v::mul_s::<W>(yp, &t, W::from_u64(a.qinv_ab), pb, acc);
    } else {
        v::mul_s::<W>(yp, xb, W::from_u64(a.qinv_ab), pb, acc);
    }
    // lift(y_a) mod q_b, times C_ab
    v::red_other::<W>(&mut t, ya, pb, acc);
    v::csub_gt::<W>(&mut t, ya, W::from_u64(pa.p >> 1), W::from_u64(pa.p % pb.p), qb);
    let mut u = buf::<W>(n);
    v::mul_s::<W>(&mut u, &t, W::from_u64(a.c_ab), pb, acc);
    v::add_rows::<W>(yp, &u, qb);
}

/// The correction row of prime r from the sources: A_r lift(y_a) [+ B_r lift(y')].
fn tmp_row<W: Hw>(a: &RescaleArgs, r: usize, ya: &[W], yp: &[W], out: &mut [W], acc: &mut [W::Acc]) {
    let pq = prime(a.pr, r);
    let q = W::from_u64(pq.p);
    let (ha, hb) = (prime(a.pr, a.top as usize).p >> 1, prime(a.pr, a.top as usize - 1).p >> 1);
    let mut l = buf::<W>(out.len());
    v::red_other::<W>(&mut l, ya, pq, acc);
    v::csub_gt::<W>(&mut l, ya, W::from_u64(ha), el::<W>(a.qam, r), q);
    v::shoup_row::<W>(out, &l, el::<W>(a.a, r), el::<W>(a.a_s, r), q, true);
    if a.ndrop == 2 {
        v::red_other::<W>(&mut l, yp, pq, acc);
        v::csub_gt::<W>(&mut l, yp, W::from_u64(hb), el::<W>(a.qbm, r), q);
        v::shoup_row::<W>(out, &l, el::<W>(a.b, r), el::<W>(a.b_s, r), q, false);
    }
}

pub extern "C" fn rescale_coef<W: Hw>(a: *const RescaleArgs, s: S) -> i32 {
    launch(a, s, |a| if a.ndrop != 1 && a.ndrop != 2 { -1 } else { 0 }, |a| {
        const BLK: usize = 4096;
        let nb = N / BLK;
        par_for(2 * a.k as usize * nb, a, |a, t| {
            let (z, b) = (t / nb, t % nb);
            let (c, part) = (z >> 1, z & 1);
            let (mut ya, mut yp, mut acc) = (buf::<W>(BLK), buf::<W>(BLK), buf::<W::Acc>(BLK));
            sources::<W>(a, c, part, b * BLK, &mut ya, &mut yp, &mut acc);
            for row in 0..a.nrows as usize {
                // SAFETY: the block of the tmp row (the sources were read first, as on the device)
                let o = unsafe { std::slice::from_raw_parts_mut(at::<W>(&a.tmp, c, part, row).add(b * BLK), BLK) };
                tmp_row::<W>(a, row, &ya, &yp, o, &mut acc);
            }
        })
    })
}

pub extern "C" fn rescale_fwd<W: Hw>(a: *const RescaleNttArgs, s: S) -> i32 {
    let bad = |a: &RescaleNttArgs| {
        let t = &a.ntt;
        if a.rs.ndrop != 2 || !t.soff.is_null() || t.epi < 0 || t.epi > 3 || (t.epi != 0 && (t.eout.is_null() || t.ex.is_null() || t.es.is_null() || t.ess.is_null())) {
            -1
        } else {
            0
        }
    };
    launch(a, s, bad, |a| {
        let r = &a.rs;
        // rescale_prep: y_a, y' into tmp rows top, top - 1
        par_for(2 * r.k as usize, r, |r, z| {
            let (c, part) = (z >> 1, z & 1);
            // SAFETY: the two source rows of (c, part)
            let (oa, ob) = unsafe {
                (std::slice::from_raw_parts_mut(at::<W>(&r.tmp, c, part, r.top as usize), N), std::slice::from_raw_parts_mut(at::<W>(&r.tmp, c, part, r.top as usize - 1), N))
            };
            let mut acc = buf::<W::Acc>(N);
            sources::<W>(r, c, part, 0, oa, ob, &mut acc);
        });
        let mut t = a.ntt;
        t.snum = 1;
        let sh = crate::raw::Shared(r);
        let sh = &sh;
        fwd_with::<W>(&t, &|t, i, _p, x| {
            let r = sh.get();
            // SAFETY: rows / prime ids of nrows entries; the sources sit top - rr and top - 1 - rr rows above
            let (roff, rr) = unsafe { (*t.roff.add(i) as usize, *t.rprime.add(i) as usize) };
            let base = t.base as *const W;
            // SAFETY: as above
            let (u, v) = unsafe { (std::slice::from_raw_parts(base.add((roff + r.top as usize - rr) * N), N), std::slice::from_raw_parts(base.add((roff + r.top as usize - 1 - rr) * N), N)) };
            let mut acc = buf::<W::Acc>(N);
            tmp_row::<W>(r, rr, u, v, x, &mut acc);
        });
    })
}
