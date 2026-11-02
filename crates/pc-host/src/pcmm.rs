// PCMM at the StC-first split (pcmm.cuh, PcmmArgs; semantics per op in args.cuh): the launcher's argument checks and
// return codes, every op word for word. The transforms run on ntt_g, the key-switch MACs and the GEMM planes'
// recombination on the simd rows; one task per (part, row) or per block of words, as the device grid.
use crate::arith::Hw;
use crate::launch;
use crate::mm::{centered, digits4, mulm, ntt_polys, smod, subm};
use crate::raw::{buf, par_for};
use pc_kernels::args::*;

const N1: usize = N / 16;
/// Words per task of the per-word ops.
const CH: usize = 4096;

fn split_limbs(a: &PcmmArgs) -> usize {
    if a.i4 != 0 {
        a.i4 as usize
    } else {
        4
    }
}

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

fn check(a: &PcmmArgs) -> i32 {
    let limbs_ok = a.i4 == 0 || a.i4 == 3 || a.i4 == 4;
    match a.op {
        pcmm_op::GATHER | pcmm_op::SCATTER | pcmm_op::IN_RESCALE | pcmm_op::LIFT if !limbs_ok => -1,
        pcmm_op::KS_DIGITS if a.count < 1 || a.count > 65535 => -1,
        pcmm_op::KS_MAC if a.i0 < 1 || a.i0 > 65535 => -1,
        pcmm_op::NTT | pcmm_op::INTT if a.n < 4096 || a.n % 4096 != 0 || a.count < 1 || a.count > 65535 => -1,
        pcmm_op::DIGITS if (a.i0 + 31) / 32 > 65535 => -1,
        0..=16 => 0,
        _ => -2,
    }
}

pub extern "C" fn pcmm<W: Hw>(a: *const PcmmArgs, s: CudaStream) -> i32 {
    launch(a, s, check, run::<W>)
}

const OPS: [&str; 17] = ["pcmm.gather", "pcmm.scatter", "pcmm.in_rescale", "pcmm.lift", "pcmm.ks_digits", "pcmm.ks_mac", "pcmm.mod_down", "pcmm.add", "pcmm.ntt", "pcmm.intt", "pcmm.mlwe_a", "pcmm.mlwe_b", "pcmm.digits", "pcmm.recombine", "pcmm.pack_c", "pcmm.add_bpk", "pcmm.compose"];

fn run<W: Hw>(a: &PcmmArgs) {
    let n = a.n as usize;
    let t0 = crate::mm::op_timer();
    match a.op {
        pcmm_op::GATHER | pcmm_op::SCATTER => {
            let ns = split_limbs(a);
            par_for(2 * ns, a, |a, row| {
                let (sp, l) = (row / ns, row % ns);
                let ct = a.i0 as usize * a.ct.ct_stride as usize + (1 - sp) * a.ct.part_stride as usize + l * N;
                let st = (sp * 4 + l) * N;
                if a.op == pcmm_op::GATHER {
                    rows_mut::<W>(a.x, st, N).copy_from_slice(rows::<W>(a.ct.base, ct, N));
                } else {
                    rows_mut::<W>(a.ct.base, ct, N).copy_from_slice(rows::<W>(a.y, st, N));
                }
            });
        }
        pcmm_op::IN_RESCALE => par_for(2 * (N / CH), a, |a, t| {
            let (part, k0) = (t / (N / CH), (t % (N / CH)) * CH);
            let ns = split_limbs(a);
            let (p, c, mu) = (&a.p, &a.c, &a.mu);
            for k in k0..k0 + CH {
                let mut v = [0u32; 4];
                for (i, vi) in v.iter_mut().enumerate().take(ns) {
                    *vi = rd::<W>(a.y, (part * 4 + i) * N + k).u64() as u32;
                }
                if ns == 4 {
                    let c3 = centered(v[3] as u64, p[3]);
                    for i in 0..3 {
                        v[i] = mulm(subm(v[i], smod(c3, p[i]), p[i]), c[i], p[i], mu[i]);
                    }
                }
                let c2 = centered(v[2] as u64, p[2]);
                for i in 0..2 {
                    wr(a.x, (part * 2 + i) * N + k, mulm(subm(v[i], smod(c2, p[i]), p[i]), c[3 + i], p[i], mu[i]));
                }
            }
        }),
        pcmm_op::LIFT => par_for(split_limbs(a), a, |a, i| {
            let x = rows_mut::<W>(a.x, i * N, N);
            if i >= 2 {
                return x.fill(W::default());
            }
            let f = rows::<u32>(a.y, i * N, N);
            for (o, &v) in x.iter_mut().zip(f) {
                *o = W::from_u64(mulm(v, a.c[i], a.p[i], a.mu[i]) as u64);
            }
        }),
        pcmm_op::KS_DIGITS => par_for(a.count as usize, a, |a, poly| {
            let polys = a.count as usize;
            for j in 0..2 {
                let q = a.p[j];
                let x = rows::<u32>(a.y, (j * polys + poly) * n, n);
                let d: [&mut [u32]; 3] = std::array::from_fn(|pi| rows_mut::<u32>(a.x, ((j * 3 + pi) * polys + poly) * n, n));
                for k in 0..n {
                    let c = centered(mulm(x[k], a.c[j], q, a.mu[j]) as u64, q);
                    for pi in 0..3 {
                        d[pi][k] = smod(c, a.p[pi]);
                    }
                }
            }
        }),
        pcmm_op::KS_MAC => {
            let (outs, nl) = (a.i0 as usize, a.i1 as usize);
            par_for(outs * 3, a, |a, t| {
                let (o, pi) = (t / 3, t % 3);
                let p = a.p[pi];
                let polys = outs * nl;
                let (mut ra, mut rb) = (buf::<u32>(n), buf::<u32>(n));
                for j in 0..2 {
                    for l in 0..nl {
                        let d = rows::<u32>(a.y, ((j * 3 + pi) * polys + o * nl + l) * n, n);
                        let kb = (((l * 2 + j) * 3 + pi) * 2) * n;
                        crate::simd::mac_mod(&mut ra, d, rows::<u32>(a.z, kb, n), p);
                        crate::simd::mac_mod(&mut rb, d, rows::<u32>(a.z, kb + n, n), p);
                    }
                }
                rows_mut::<u32>(a.x, (pi * outs + o) * n, n).copy_from_slice(&ra);
                rows_mut::<u32>(a.x, ((3 + pi) * outs + o) * n, n).copy_from_slice(&rb);
            });
        }
        pcmm_op::MOD_DOWN => {
            let outs = a.i0 as usize;
            par_for(outs * 2, a, |a, t| {
                let (o, ab) = (t / 2, t % 2);
                let accp = rows::<u32>(a.y, ((ab * 3 + 2) * outs + o) * n, n);
                for i in 0..2 {
                    let q = a.p[i];
                    let v = rows::<u32>(a.y, ((ab * 3 + i) * outs + o) * n, n);
                    let x = rows_mut::<u32>(a.x, ((ab * 2 + i) * outs + o) * n, n);
                    for k in 0..n {
                        x[k] = mulm(subm(v[k], smod(centered(accp[k] as u64, a.p[2]), q), q), a.c[i], q, a.mu[i]);
                    }
                }
            });
        }
        pcmm_op::ADD => {
            let per = a.i0 as usize;
            let tot = a.count as usize * per;
            par_for(tot.div_ceil(CH), a, |a, t| {
                let (k0, k1) = (t * CH, ((t + 1) * CH).min(tot));
                for k in k0..k1 {
                    let p = a.p[k / per];
                    wr(a.x, k, crate::mm::addm(rd::<u32>(a.x, k), rd::<u32>(a.y, k), p));
                }
            });
        }
        pcmm_op::NTT | pcmm_op::INTT => {
            let i0 = a.i0 as usize;
            let sc = (a.op == pcmm_op::INTT).then_some(&a.c[..3]);
            ntt_polys(a.x as *mut u32, n, a.count as usize, a.tw as *const u32, &a.p[..3], sc, move |k| (k / i0) % 3);
        }
        pcmm_op::MLWE_A => par_for(2 * a.count as usize, a, |a, t| {
            let (pi, row) = (t / a.count as usize, t % a.count as usize);
            let (r, np, total, off) = (a.i2 as usize, a.i3 as usize, a.i0 as usize, a.i1 as usize);
            let p = a.p[pi];
            let (sub, i) = (row / r, row % r);
            let ka = rows::<u32>(a.y, pi * N, N);
            let x = rows_mut::<u32>(a.x, (pi * total + off + row) * N1, N1);
            for (col, o) in x.iter_mut().enumerate() {
                let (l, w) = (col / np, col % np);
                *o = if l <= i {
                    ka[(w * r + (i - l)) * 16 + sub]
                } else {
                    let c = i + r - l;
                    if w == 0 {
                        subm(0, ka[((np - 1) * r + c) * 16 + sub], p)
                    } else {
                        ka[((w - 1) * r + c) * 16 + sub]
                    }
                };
            }
        }),
        pcmm_op::MLWE_B => par_for(2 * a.count as usize, a, |a, t| {
            let (pi, row) = (t / a.count as usize, t % a.count as usize);
            let (r, np, total, off) = (a.i2 as usize, a.i3 as usize, a.i0 as usize, a.i1 as usize);
            let (sub, i) = (row / r, row % r);
            let b = rows::<u32>(a.y, pi * N, N);
            let x = rows_mut::<u32>(a.x, (pi * total + off + row) * np, np);
            for (w, o) in x.iter_mut().enumerate() {
                *o = b[(w * r + i) * 16 + sub];
            }
        }),
        pcmm_op::DIGITS => digits(a),
        pcmm_op::RECOMBINE => {
            let (rc, off, rows_out, cols, e) = (a.i0 as usize, a.i1 as usize, a.i2 as usize, a.i3 as usize, a.i4 as usize);
            let c2 = 2 * cols;
            let count = rc * c2;
            par_for(rc * 2, a, |a, t| {
                let (r, pi) = (t / 2, t % 2);
                let k0 = r * c2 + pi * cols;
                let g: Vec<&[i32]> = (0..e).map(|ee| rows::<i32>(a.y, ee * count + k0, cols)).collect();
                crate::simd::horner256(rows_mut::<u32>(a.x, (pi * rows_out + off + r) * cols, cols), &g, a.p[pi]);
            });
        }
        pcmm_op::PACK_C => par_for(2 * a.count as usize, a, |a, t| {
            let (pi, sl) = (t / a.count as usize, t % a.count as usize);
            let (r, np, total, off, nch) = (a.i2 as usize, a.i3 as usize, a.i0 as usize, a.i1 as usize, a.count as usize);
            let (sub, l) = (sl / r, sl % r);
            let x = rows_mut::<u32>(a.x, (pi * nch + sl) * N1, N1);
            for (idx, o) in x.iter_mut().enumerate() {
                let (w, i) = (idx / r, idx % r);
                *o = rd(a.y, (pi * total + off + sub * r + i) * N1 + l * np + w);
            }
        }),
        pcmm_op::ADD_BPK => par_for(32, a, |a, t| {
            let (pi, sub) = (t / 16, t % 16);
            let (r, np, total, off) = (a.i2 as usize, a.i3 as usize, a.i0 as usize, a.i1 as usize);
            let x = rows_mut::<u32>(a.x, (pi * 16 + sub) * N1, N1);
            for (idx, o) in x.iter_mut().enumerate() {
                let (w, i) = (idx / r, idx % r);
                *o = crate::mm::addm(*o, rd(a.y, (pi * total + off + sub * r + i) * np + w), a.p[pi]);
            }
        }),
        pcmm_op::COMPOSE => par_for(2, a, |a, pi| {
            let x = rows_mut::<u32>(a.x, pi * N, N);
            for (k, o) in x.iter_mut().enumerate() {
                *o = rd(a.y, (pi * 16 + k % 16) * N1 + k / 16);
            }
        }),
        _ => {}
    }
    crate::mm::op_prof(OPS.get(a.op as usize).copied().unwrap_or("?"), t0);
}

/// d[t][pi cols + c][r] = digit t of centered X[pi][r][c] (32 x 32 tiles: read along c, write along r).
fn digits(a: &PcmmArgs) {
    let (nr, nc) = (a.i0 as usize, a.i1 as usize);
    let (tr, tc) = (nr.div_ceil(32), nc.div_ceil(32));
    let count = 2 * nc * nr;
    par_for(2 * tr * tc, a, |a, t| {
        let (pi, rest) = (t / (tr * tc), t % (tr * tc));
        let (r0, c0) = ((rest / tc) * 32, (rest % tc) * 32);
        let p = a.p[pi];
        let mut tile = [[[0i8; 32]; 32]; 4];
        let (re, ce) = ((r0 + 32).min(nr), (c0 + 32).min(nc));
        for r in r0..re {
            let x = rows::<u32>(a.y, (pi * nr + r) * nc + c0, ce - c0);
            for (c, &v) in x.iter().enumerate() {
                let d = digits4(centered(v as u64, p));
                for (tt, &dv) in d.iter().enumerate() {
                    tile[tt][c][r - r0] = dv;
                }
            }
        }
        for c in c0..ce {
            for (tt, tl) in tile.iter().enumerate() {
                rows_mut::<i8>(a.x, tt * count + (pi * nc + c) * nr + r0, re - r0).copy_from_slice(&tl[c - c0][..re - r0]);
            }
        }
    });
}
