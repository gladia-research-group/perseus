// CCMM (ccmm.cuh, CcmmArgs; semantics per op in args.cuh): the launcher's argument checks and return codes, every op
// word for word. Shapes are the kernels' constants (N = 2^16, NS = 2^13, D = 128, KY = 64, NQ = 4, NP = 5). The
// transforms run on ntt_g, the key-switch MACs and the GEMM planes' recombination on the simd rows; one task per
// column / polynomial / block of words.
use crate::arith::Hw;
use crate::launch;
use crate::mm::{addm, centered, digits4, mulm, ntt_polys, smod, subm};
use crate::raw::{buf, par_for};
use pc_kernels::args::*;

const NB: usize = N;
const NS: usize = 1 << 13;
const D: usize = 128;
const KY: usize = NS / D;
const SW: usize = NB / NS;
const NBIG: usize = D / SW;
const NQ: usize = 4;
const NP: usize = 5;
const COLS: usize = 2 * NQ * D * NS;
const SMEM: usize = 8192;
const CH: usize = 8192;

#[inline(always)]
fn rd<T: Copy>(p: *const std::ffi::c_void, i: usize) -> T {
    // SAFETY: word i of a launch buffer (the launch contract bounds it)
    unsafe { *(p as *const T).add(i) }
}
#[inline(always)]
fn wr<T>(p: *mut std::ffi::c_void, i: usize, v: T) {
    // SAFETY: as above; tasks write disjoint words
    unsafe { *(p as *mut T).add(i) = v }
}
fn rows<'a, T>(p: *const std::ffi::c_void, off: usize, n: usize) -> &'a [T] {
    // SAFETY: n words of a launch buffer
    unsafe { std::slice::from_raw_parts((p as *const T).add(off), n) }
}
fn rows_mut<'a, T>(p: *mut std::ffi::c_void, off: usize, n: usize) -> &'a mut [T] {
    // SAFETY: n words of a launch buffer, disjoint per task
    unsafe { std::slice::from_raw_parts_mut((p as *mut T).add(off), n) }
}

fn check(a: &CcmmArgs) -> i32 {
    let n = a.n as i64;
    match a.op {
        ccmm_op::NTT | ccmm_op::INTT => {
            if n < 64 || n > NB as i64 || n & (n - 1) != 0 || a.count < 1 || a.i0 < 1 || a.i1 < 1 || a.i1 > NP as i32 {
                return -1;
            }
            let l = n.min(SMEM as i64);
            let (s, pp) = (n / l, SMEM as i64 / l);
            if a.count as i64 * n % SMEM as i64 != 0 || a.i0 as i64 % pp != 0 || (s > 1 && a.count > 65535) {
                return -1;
            }
            0
        }
        ccmm_op::KS_DIGITS | ccmm_op::KS_MAC | ccmm_op::MOD_DOWN if a.count < 1 || a.count > 65535 => -1,
        ccmm_op::TWEAK if a.i0 < 1 || a.i0 >= D as i32 => -1,
        ccmm_op::DIGITS | ccmm_op::RECOMBINE if a.i0 < 0 || a.i0 >= NQ as i32 => -1,
        0..=19 => 0,
        _ => -2,
    }
}

pub extern "C" fn ccmm<W: Hw>(a: *const CcmmArgs, s: CudaStream) -> i32 {
    launch(a, s, check, run::<W>)
}

/// out[o] = (a X^e)[o] over the negacyclic ring of NS (mono_at).
#[inline(always)]
fn mono_at(v: &[u32], k: usize, e: i64, p: u32) -> u32 {
    let mut e = e % (2 * NS as i64);
    if e < 0 {
        e += 2 * NS as i64;
    }
    let mut neg = false;
    if e >= NS as i64 {
        e -= NS as i64;
        neg = true;
    }
    let mut src = k as i64 - e;
    if src < 0 {
        src += NS as i64;
        neg = !neg;
    }
    let x = v[src as usize];
    if neg {
        subm(0, x, p)
    } else {
        x
    }
}

const OPS: [&str; 20] = ["ccmm.gather", "ccmm.scatter", "ccmm.ntt", "ccmm.intt", "ccmm.ks_digits", "ccmm.ks_mac", "ccmm.mod_down", "ccmm.ring_in", "ccmm.compose_a", "ccmm.ring_out", "ccmm.mono", "ccmm.bitrev", "ccmm.tweak", "ccmm.scramble", "ccmm.ks_apply", "ccmm.add", "ccmm.gather_u", "ccmm.digits", "ccmm.recombine", "ccmm.scatter_rows"];

fn run<W: Hw>(a: &CcmmArgs) {
    let n = a.n as usize;
    let t0 = crate::mm::op_timer();
    match a.op {
        ccmm_op::GATHER | ccmm_op::SCATTER => par_for(8, a, |a, row| {
            let (part, l) = (row / 4, row % 4);
            let ct = a.i0 as usize * a.ct.ct_stride as usize + part * a.ct.part_stride as usize + l * NB;
            let st = (a.i1 as usize * 8 + row) * NB;
            if a.op == ccmm_op::GATHER {
                rows_mut::<W>(a.x, st, NB).copy_from_slice(rows::<W>(a.ct.base, ct, NB));
            } else {
                rows_mut::<W>(a.ct.base, ct, NB).copy_from_slice(rows::<W>(a.y, st, NB));
            }
        }),
        ccmm_op::NTT | ccmm_op::INTT => {
            let (i0, i1) = (a.i0 as usize, a.i1 as usize);
            let sc = (a.op == ccmm_op::INTT).then_some(&a.c[..i1]);
            ntt_polys(a.x as *mut u32, n, a.count as usize, a.tw as *const u32, &a.p[..i1], sc, move |k| (k / i0) % i1);
        }
        ccmm_op::KS_DIGITS => par_for(a.count as usize, a, |a, q| {
            let polys = a.count as usize;
            for j in 0..NQ {
                let qj = a.p[j];
                let src = j * a.s[0] as usize + q * a.s[1] as usize;
                let d: [&mut [u32]; NP] = std::array::from_fn(|pi| rows_mut::<u32>(a.x, ((j * NP + pi) * polys + q) * n, n));
                for k in 0..n {
                    let v = if a.i4 != 0 { rd::<W>(a.y, src + k).u64() as u32 } else { rd::<u32>(a.y, src + k) };
                    let c = centered(mulm(v, a.c[j], qj, a.mu[j]) as u64, qj);
                    for pi in 0..NP {
                        d[pi][k] = smod(c, a.p[pi]);
                    }
                }
            }
        }),
        ccmm_op::KS_MAC => par_for(a.count as usize * NP, a, |a, t| {
            let (q, pi) = (t / NP, t % NP);
            let polys = a.count as usize;
            let p = a.p[pi];
            let key = q * a.s[0] as usize;
            let (mut ra, mut rb) = (buf::<u32>(n), buf::<u32>(n));
            for j in 0..NQ {
                let d = rows::<u32>(a.y, ((j * NP + pi) * polys + q) * n, n);
                let kb = key + ((j * NP + pi) * 2) * n;
                crate::simd::mac_mod(&mut ra, d, rows::<u32>(a.z, kb, n), p);
                crate::simd::mac_mod(&mut rb, d, rows::<u32>(a.z, kb + n, n), p);
            }
            rows_mut::<u32>(a.x, (pi * polys + q) * n, n).copy_from_slice(&ra);
            rows_mut::<u32>(a.x, ((NP + pi) * polys + q) * n, n).copy_from_slice(&rb);
        }),
        ccmm_op::MOD_DOWN => par_for(a.count as usize * 2, a, |a, t| {
            let (q, ab) = (t / 2, t % 2);
            let polys = a.count as usize;
            let accp = rows::<u32>(a.y, ((ab * NP + NQ) * polys + q) * n, n);
            for i in 0..NQ {
                let qi = a.p[i];
                let v = rows::<u32>(a.y, ((ab * NP + i) * polys + q) * n, n);
                let x = rows_mut::<u32>(a.x, ((ab * NQ + i) * polys + q) * n, n);
                for k in 0..n {
                    x[k] = mulm(subm(v[k], smod(centered(accp[k] as u64, a.p[NQ]), qi), qi), a.c[i], qi, a.mu[i]);
                }
            }
        }),
        ccmm_op::RING_IN => par_for(NQ * D, a, |a, t| {
            let (i, col) = (t / D, t % D);
            let (c, j) = (col / SW, col % SW);
            let x0 = rows_mut::<u32>(a.x, (i * D + col) * NS, NS);
            for (z, o) in x0.iter_mut().enumerate() {
                let idx = j + SW * z;
                let b0 = rd::<W>(a.y, ((c * 2) * 4 + i) * NB + idx).u64() as u32;
                *o = addm(b0, rd(a.z, ((NQ + i) * NBIG + c) * NB + idx), a.p[i]);
            }
            let x1 = rows_mut::<u32>(a.x, ((NQ + i) * D + col) * NS, NS);
            for (z, o) in x1.iter_mut().enumerate() {
                *o = rd(a.z, (i * NBIG + c) * NB + j + SW * z);
            }
        }),
        ccmm_op::COMPOSE_A => par_for(NQ * NBIG, a, |a, t| {
            let (i, c) = (t / NBIG, t % NBIG);
            let x = rows_mut::<u32>(a.x, (i * NBIG + c) * NB, NB);
            for (idx, o) in x.iter_mut().enumerate() {
                *o = rd(a.y, ((NQ + i) * D + c * SW + idx % SW) * NS + idx / SW);
            }
        }),
        ccmm_op::RING_OUT => par_for(NQ * NBIG, a, |a, t| {
            let (i, c) = (t / NBIG, t % NBIG);
            let s0 = rows_mut::<W>(a.x, ((c * 2) * 4 + i) * NB, NB);
            for (idx, o) in s0.iter_mut().enumerate() {
                let xb = rd::<u32>(a.y, (i * D + c * SW + idx % SW) * NS + idx / SW);
                *o = W::from_u64(addm(xb, rd(a.z, ((NQ + i) * NBIG + c) * NB + idx), a.p[i]) as u64);
            }
            let s1 = rows_mut::<W>(a.x, ((c * 2 + 1) * 4 + i) * NB, NB);
            for (idx, o) in s1.iter_mut().enumerate() {
                *o = W::from_u64(rd::<u32>(a.z, (i * NBIG + c) * NB + idx) as u64);
            }
        }),
        ccmm_op::MONO => par_for(2 * NQ * D, a, |a, t| {
            let (pp, col) = (t / D, t % D);
            let base = (pp * D + col) * NS;
            let y = rows::<u32>(a.y, base, NS);
            let p = a.p[pp % NQ];
            let e = a.i0 as i64 * col as i64;
            for (k, o) in rows_mut::<u32>(a.x, base, NS).iter_mut().enumerate() {
                *o = mono_at(y, k, e, p);
            }
        }),
        ccmm_op::BITREV => par_for(2 * NQ * D, a, |a, t| {
            let (pp, col) = (t / D, t % D);
            let r = (col as u32).reverse_bits() as usize >> (32 - 7);
            rows_mut::<u32>(a.x, (pp * D + col) * NS, NS).copy_from_slice(rows::<u32>(a.y, (pp * D + r) * NS, NS));
        }),
        ccmm_op::TWEAK => par_for(2 * NQ * (D / 2), a, |a, t| {
            let (pp, pair) = (t / (D / 2), t % (D / 2));
            let p = a.p[pp % NQ];
            let (h, kk, sgn) = (a.i0 as usize, a.i1 as i64, a.i2 as i64);
            let (g, j) = (pair / h, pair % h);
            let i0 = g * 2 * h + j;
            let i1 = i0 + h;
            let (b0, b1) = ((pp * D + i0) * NS, (pp * D + i1) * NS);
            let (u, w) = (rows::<u32>(a.y, b0, NS), rows::<u32>(a.y, b1, NS));
            let (o0, o1) = (rows_mut::<u32>(a.x, b0, NS), rows_mut::<u32>(a.x, b1, NS));
            let e = 2 * kk * j as i64 * sgn;
            for k in 0..NS {
                let v = mono_at(w, k, e, p);
                let uk = u[k];
                o0[k] = addm(uk, v, p);
                o1[k] = subm(uk, v, p);
            }
        }),
        ccmm_op::SCRAMBLE => par_for(2 * NQ * D, a, |a, t| {
            let (pp, tt) = (t / D, t % D);
            let qi = pp % NQ;
            let p = a.p[qi];
            let h = 2 * KY * tt + 1;
            let ts = rd::<i32>(a.z, tt) as usize;
            let y = rows::<u32>(a.y, (pp * D + ts) * NS, NS);
            let x = rows_mut::<u32>(a.x, (pp * D + tt) * NS, NS);
            for (u, &yv) in y.iter().enumerate() {
                let v = mulm(yv, a.c[qi], p, a.mu[qi]);
                let pos = (u * h) % (2 * NS);
                if pos >= NS {
                    x[pos - NS] = subm(0, v, p);
                } else {
                    x[pos] = v;
                }
            }
        }),
        ccmm_op::KS_APPLY | ccmm_op::ADD => {
            let per = a.s[0] as usize;
            let tot = NQ * per;
            par_for(tot.div_ceil(CH), a, |a, t| {
                let (k0, k1) = (t * CH, ((t + 1) * CH).min(tot));
                for k in k0..k1 {
                    let p = a.p[k / per];
                    if a.op == ccmm_op::ADD {
                        wr(a.x, k, addm(rd(a.x, k), rd(a.y, k), p));
                    } else {
                        wr(a.x, k, addm(rd(a.x, k), rd(a.y, tot + k), p));
                        wr(a.x, tot + k, rd::<u32>(a.y, k));
                    }
                }
            });
        }
        ccmm_op::GATHER_U => par_for(COLS / NS, a, |a, pc| {
            let y = rows::<u32>(a.y, pc * NS, NS);
            let x = rows_mut::<u32>(a.x, pc * NS, NS);
            for i in 0..D {
                for u in 0..KY {
                    x[i * KY + u] = y[i + D * u];
                }
            }
        }),
        ccmm_op::DIGITS => digits(a),
        ccmm_op::RECOMBINE => {
            let gper = KY * 4 * D * D;
            let prime = a.i0 as usize;
            // per block of 2D (r, c) pairs: the 64 u-planes recombined, then u made innermost
            par_for(2 * D, a, |a, r| {
                let p = a.p[prime];
                let mut cn = buf::<u32>(KY * 2 * D);
                for u in 0..KY {
                    let k0 = u * 4 * D * D + r * 2 * D;
                    let g: Vec<&[i32]> = (0..7).map(|e| rows::<i32>(a.y, e * gper + k0, 2 * D)).collect();
                    crate::simd::horner256(&mut cn[u * 2 * D..(u + 1) * 2 * D], &g, p);
                }
                let x = rows_mut::<u32>(a.x, ((prime * 2 * D + r) * 2 * D) * KY, 2 * D * KY);
                for c in 0..2 * D {
                    for u in 0..KY {
                        x[c * KY + u] = cn[u * 2 * D + c];
                    }
                }
            });
        }
        ccmm_op::SCATTER_ROWS => par_for(2 * NQ * D, a, |a, t| {
            let (pp, j) = (t / D, t % D);
            let (part, prime) = (pp / NQ, pp % NQ);
            let x = rows_mut::<u32>(a.x, t * NS, NS);
            for u in 0..KY {
                let src = rows::<u32>(a.y, ((prime * 2 * D + a.i0 as usize + j) * 2 * D + part * D) * KY, D * KY);
                for i in 0..D {
                    x[i + D * u] = src[i * KY + u];
                }
            }
        }),
        _ => {}
    }
    crate::mm::op_prof(OPS.get(a.op as usize).copied().unwrap_or("?"), t0);
}

/// d[dg][u][r][t] = digit dg of centered E[r / D][prime][t][r % D][u] (one task per r: E read along u, the digits
/// staged as [dg][u][t]).
fn digits(a: &CcmmArgs) {
    let per = 2 * D * D * KY;
    let prime = a.i0 as usize;
    par_for(2 * D, a, |a, r| {
        let p = a.p[prime];
        let (part, i) = (r / D, r % D);
        let mut tile = [[[0i8; D]; KY]; 4];
        for t in 0..D {
            let e = rows::<u32>(a.y, (((part * NQ + prime) * D + t) * D + i) * KY, KY);
            for (u, &v) in e.iter().enumerate() {
                let d = digits4(centered(v as u64, p));
                for dg in 0..4 {
                    tile[dg][u][t] = d[dg];
                }
            }
        }
        for (dg, td) in tile.iter().enumerate() {
            for (u, row) in td.iter().enumerate() {
                rows_mut::<i8>(a.x, dg * per + (u * 2 * D + r) * D, D).copy_from_slice(row);
            }
        }
    });
}
