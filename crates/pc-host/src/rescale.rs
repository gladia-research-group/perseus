// Composite rescale (rescale.cuh, RescaleArgs): rescale_coef writes the correction rows tmp_i = A_i lift(y_a)
// [+ B_i lift(y')] in coefficient form; rescale_fwd generates the same rows inside the forward NTT (ndrop 2), after
// rescale_prep stored y_a and y' in tmp rows top and top - 1.
use crate::arith::Hw;
use crate::launch;
use crate::ntt::fwd_with;
use crate::raw::{at, el, par_for};
use pc_kernels::args::*;

type S = CudaStream;

#[inline(always)]
fn prime(pr: *const PrimeConst, id: usize) -> &'static PrimeConst {
    // SAFETY: prime table by id (device constant data)
    unsafe { &*pr.add(id) }
}

/// y mod q for a residue of another prime (reduce_small: exact either way).
#[inline(always)]
fn red<W: Hw>(y: W, q: u64) -> W {
    let mut v = y.u64();
    if v >= 3 * q {
        return W::from_u64(v % q);
    }
    while v >= q {
        v -= q;
    }
    W::from_u64(v)
}

/// Centred representative of y (mod qo), reduced mod qn.
#[inline(always)]
fn lift<W: Hw>(y: W, qo: W, qn: u64) -> W {
    let yr = W::from_u64(y.u64() % qn);
    if y.u64() > qo.u64() >> 1 {
        let qor = qo.u64() % qn;
        let v = if yr.u64() >= qor { yr.u64() - qor } else { yr.u64() + qn - qor };
        return W::from_u64(v);
    }
    yr
}

/// (y_a, y') of coefficient i of (c, part): the tops (scaled by ca / cb), y' = qinv_ab y_b + C_ab lift(y_a) mod q_b.
#[inline(always)]
fn sources<W: Hw>(a: &RescaleArgs, c: usize, part: usize, i: usize) -> (W, W) {
    let (ta, tb) = (a.top as usize, a.top as usize - 1);
    let pa = prime(a.pr, ta);
    // SAFETY: word i of the top rows
    let mut ya = unsafe { *at::<W>(&a.x, c, part, ta).add(i) };
    if a.ca != 1 {
        ya = W::mul(ya, W::from_u64(a.ca), pa);
    }
    if a.ndrop != 2 {
        return (ya, W::default());
    }
    let pb = prime(a.pr, tb);
    // SAFETY: as above
    let mut yb = unsafe { *at::<W>(&a.x, c, part, tb).add(i) };
    if a.ca != 1 {
        yb = W::mul(yb, W::from_u64(a.cb), pb);
    }
    let qb = W::from_u64(pb.p);
    let yp = W::add(W::mul(W::from_u64(a.qinv_ab), yb, pb), W::mul(W::from_u64(a.c_ab), lift(ya, W::from_u64(pa.p), pb.p), pb), qb);
    (ya, yp)
}

/// tmp word of row r from the sources (u = y_a, v = y').
#[inline(always)]
fn tmp_word<W: Hw>(a: &RescaleArgs, r: usize, u: W, v: W) -> W {
    let q = prime(a.pr, r).p;
    let qw = W::from_u64(q);
    let (ha, hb) = (prime(a.pr, a.top as usize).p >> 1, prime(a.pr, a.top as usize - 1).p >> 1);
    let mut la = red(u, q);
    if u.u64() > ha {
        la = W::sub(la, el::<W>(a.qam, r), qw);
    }
    let mut w = W::mul_shoup(la, el::<W>(a.a, r), el::<W>(a.a_s, r), qw);
    if a.ndrop == 2 {
        let mut lb = red(v, q);
        if v.u64() > hb {
            lb = W::sub(lb, el::<W>(a.qbm, r), qw);
        }
        w = W::add(w, W::mul_shoup(lb, el::<W>(a.b, r), el::<W>(a.b_s, r), qw), qw);
    }
    w
}

pub extern "C" fn rescale_coef<W: Hw>(a: *const RescaleArgs, s: S) -> i32 {
    launch(a, s, |a| if a.ndrop != 1 && a.ndrop != 2 { -1 } else { 0 }, |a| {
        const BLK: usize = 4096;
        let nb = N / BLK;
        par_for(2 * a.k as usize * nb, a, |a, t| {
            let (z, b) = (t / nb, t % nb);
            let (c, part) = (z >> 1, z & 1);
            let src: Vec<(W, W)> = (b * BLK..(b + 1) * BLK).map(|i| sources::<W>(a, c, part, i)).collect();
            for row in 0..a.nrows as usize {
                let o = at::<W>(&a.tmp, c, part, row);
                for (j, &(u, v)) in src.iter().enumerate() {
                    // SAFETY: word of the tmp row
                    unsafe { *o.add(b * BLK + j) = tmp_word(a, row, u, v) };
                }
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
            let (oa, ob) = (at::<W>(&r.tmp, c, part, r.top as usize), at::<W>(&r.tmp, c, part, r.top as usize - 1));
            for i in 0..N {
                let (u, v) = sources::<W>(r, c, part, i);
                // SAFETY: words of the two source rows
                unsafe {
                    *oa.add(i) = u;
                    *ob.add(i) = v;
                }
            }
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
            for (o, xo) in x.iter_mut().enumerate() {
                // SAFETY: as above
                let (u, v) = unsafe { (*base.add((roff + r.top as usize - rr) * N + o), *base.add((roff + r.top as usize - 1 - rr) * N + o)) };
                *xo = tmp_word(r, rr, u, v);
            }
        });
    })
}
