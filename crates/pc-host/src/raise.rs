// Bootstrap raise kernels (bts.cuh): composite ModRaise (each Garner term centred on its own prime, re-reduced),
// smallIntDivRound (Garner, centre against Q_k / 2, optional fixed-point multiply by R >> 192 half up, reduce mod
// every output row), and the one-prime scaling of the n64 chain. Integer arithmetic, word for word the device's.
use crate::arith::Hw;
use crate::launch;
use crate::raw::par_for;
use pc_kernels::args::*;

type S = CudaStream;

#[inline(always)]
fn prime(pr: *const PrimeConst, id: usize) -> &'static PrimeConst {
    // SAFETY: prime table by id (device constant data)
    unsafe { &*pr.add(id) }
}

pub extern "C" fn mod_raise<W: Hw>(a: *const ModRaiseArgs, s: S) -> i32 {
    let bad = |a: &ModRaiseArgs| {
        if a.d < 1 || a.d > 4 || a.r0 < a.d || a.r1 > 64 || a.r1 <= a.r0 || a.nparts < 1 || a.nparts > 2 {
            -1
        } else {
            0
        }
    };
    launch(a, s, bad, |a| {
        let nt = (a.r1 - a.r0) as usize;
        par_for(a.k as usize * a.nparts as usize * nt, a, |a, task| {
            let (z, y) = (task / nt, task % nt);
            let (c, part) = (z / a.nparts as usize, z % a.nparts as usize);
            let t = a.r0 as usize + y;
            let x = (a.x as *mut W).wrapping_add(c * a.ct_stride as usize + part * a.part_stride as usize);
            let qt = prime(a.pr, t);
            let pt = qt.p;
            let mut acc = crate::raw::buf::<W>(N);
            for k in 0..a.d as usize {
                let qk = prime(a.pr, k);
                let pk = qk.p;
                let (hinv, h) = (W::from_u64(a.qhatinv[k]), W::from_u64(a.qhat[k][t]));
                // SAFETY: source row k (rows 0..d are read, rows r0.. written)
                let src = unsafe { std::slice::from_raw_parts(x.add(k * N), N) };
                for (o, &v) in acc.iter_mut().zip(src) {
                    let y = W::mul(v, hinv, qk).u64() as u128;
                    let v = if y as u64 > pk >> 1 { y + pt as u128 - (pk % pt) as u128 } else { y };
                    let r = W::from_u64((v % pt as u128) as u64);
                    *o = W::add(*o, W::mul(r, h, qt), W::from_u64(pt));
                }
            }
            // SAFETY: output row t
            unsafe { std::slice::from_raw_parts_mut(x.add(t * N), N) }.copy_from_slice(&acc);
        })
    })
}

/// smallIntDivRound of one coefficient: the signed scaled integer as (magnitude words, count, negative).
fn small_int_coef(tab: &SmallIntTab, src: &dyn Fn(usize) -> u64) -> ([u32; SI_MAXW + SI_MAXR], usize, bool) {
    let (k, nw) = (tab.k as usize, tab.nw as usize);
    let mut v = [0u32; SI_MAXK];
    for i in 0..k {
        let qi = tab.q[i] as u64;
        let mut t = src(i) % qi;
        for j in 0..i {
            let vj = v[j] as u64 % qi;
            t = (t + qi - vj) % qi;
            t = (t * tab.qinv[i][j] as u64) % qi;
        }
        v[i] = t as u32;
    }
    let mut x = [0u32; SI_MAXW + SI_MAXR];
    let mut nx = 1;
    x[0] = v[k - 1];
    for i in (0..k.saturating_sub(1)).rev() {
        let mut carry = v[i] as u64;
        for w in x.iter_mut().take(nx) {
            let t = *w as u64 * tab.q[i] as u64 + carry;
            *w = t as u32;
            carry = t >> 32;
        }
        if carry != 0 {
            x[nx] = carry as u32;
            nx += 1;
        }
    }
    let mut neg = false;
    let mut gt = false;
    for w in (0..nw).rev() {
        if x[w] != tab.qkh[w] {
            gt = x[w] > tab.qkh[w];
            break;
        }
    }
    if gt {
        neg = true;
        let mut borrow = 0i64;
        for w in 0..nw {
            let mut t = tab.qk[w] as i64 - x[w] as i64 - borrow;
            borrow = (t < 0) as i64;
            if t < 0 {
                t += 1i64 << 32;
            }
            x[w] = t as u32;
        }
    }
    let mut nq = nw;
    if tab.divide != 0 {
        let mut prod = [0u32; SI_MAXW + SI_MAXR];
        for i in 0..nw {
            let mut carry = 0u64;
            for j in 0..tab.nr as usize {
                let t = x[i] as u64 * tab.r[j] as u64 + prod[i + j] as u64 + carry;
                prod[i + j] = t as u32;
                carry = t >> 32;
            }
            let mut p = i + tab.nr as usize;
            while carry != 0 && p < SI_MAXW + SI_MAXR {
                let t = prod[p] as u64 + carry;
                prod[p] = t as u32;
                carry = t >> 32;
                p += 1;
            }
        }
        let mut carry = 1u64 << 31;
        let mut i = 5;
        while i < SI_MAXW + SI_MAXR && carry != 0 {
            let t = prod[i] as u64 + carry;
            prod[i] = t as u32;
            carry = t >> 32;
            i += 1;
        }
        let dw = (SI_SHIFT / 32) as usize;
        nq = (nw + tab.nr as usize).saturating_sub(dw).max(1);
        for i in 0..nq {
            x[i] = prod.get(i + dw).copied().unwrap_or(0);
        }
    }
    (x, nq, neg)
}

pub extern "C" fn small_int<W: Hw>(a: *const SmallIntArgs, s: S) -> i32 {
    let bad = |a: &SmallIntArgs| {
        if W::BITS != 32 {
            -2
        } else if a.nparts < 1 || a.nparts > 2 || a.k < 1 {
            -1
        } else {
            0
        }
    };
    launch(a, s, bad, |a| {
        const BLK: usize = 4096;
        let nb = N / BLK;
        par_for(a.k as usize * a.nparts as usize * nb, a, |a, task| {
            // SAFETY: the launch's table
            let tab = unsafe { &*a.tab };
            let (z, b) = (task / nb, task % nb);
            let (c, part) = (z / a.nparts as usize, z % a.nparts as usize);
            let src = (a.src as *const W).wrapping_add(c * a.src_ct_stride as usize + part * a.src_part_stride as usize);
            let out = (a.out as *mut W).wrapping_add(c * a.ct_stride as usize + part * a.part_stride as usize);
            for idx in b * BLK..(b + 1) * BLK {
                // SAFETY: source rows of the coefficient
                let (x, nq, neg) = small_int_coef(tab, &|i| unsafe { *src.add(i * N + idx) }.u64());
                for o in 0..tab.nout as usize {
                    let row = tab.orow[o] as usize;
                    let q = prime(a.pr, row);
                    let acc: u64 = (0..nq).map(|w| x[w] as u64 * tab.pow32[o][w] as u64).fold(0u64, |s, v| s.wrapping_add(v));
                    let mut r = acc % q.p;
                    if neg && r != 0 {
                        r = q.p - r;
                    }
                    // SAFETY: output row of the coefficient
                    unsafe { *out.add(row * N + idx) = W::from_u64(r) };
                }
            }
        })
    })
}

pub extern "C" fn small_int1<W: Hw>(a: *const SmallInt1Args, s: S) -> i32 {
    let bad = |a: &SmallInt1Args| {
        if a.sh < 1 || a.sh > 127 || a.nrows < 1 || a.nrows > 64 || a.k < 1 || a.nparts < 1 || a.nparts > 2 {
            -1
        } else {
            0
        }
    };
    launch(a, s, bad, |a| {
        par_for(a.k as usize * a.nparts as usize, a, |a, z| {
            let (c, part) = (z / a.nparts as usize, z % a.nparts as usize);
            let x = (a.x as *mut W).wrapping_add(c * a.ct_stride as usize + part * a.part_stride as usize);
            let q0 = prime(a.pr, 0).p;
            for idx in 0..N {
                // SAFETY: row 0 of the polynomial, read before any row is written
                let v = unsafe { *x.add(idx) }.u64();
                let neg = v > q0 >> 1;
                let mag = if neg { q0 - v } else { v };
                let prod = mag as u128 * a.mc as u128 + (1u128 << (a.sh - 1));
                let q = (prod >> a.sh) as u64;
                for t in 0..a.nrows as usize {
                    let p = prime(a.pr, t).p;
                    let r = q % p;
                    // SAFETY: row t of the polynomial
                    unsafe { *x.add(t * N + idx) = W::from_u64(if neg && r != 0 { p - r } else { r }) };
                }
            }
        })
    })
}
