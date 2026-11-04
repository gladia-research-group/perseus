// Element-wise kernels (elem.cuh, em.cuh, automorph.cuh, the rescale combines): one task per (ciphertext, part, row)
// as the device grid, each sweeping its N words with the device's per-word formula.
use crate::arith::{automorph_slot, ptidx, Hw};
use crate::launch;
use crate::raw::{at, el, par_for, rs_len, rs_row};
use crate::simd::w as v;
use pc_kernels::args::*;

type S = CudaStream;

/// (ciphertext, part, row index) of task t over K x nparts x rows (rows fastest).
#[inline(always)]
fn split(t: usize, rows: usize, nparts: usize) -> (usize, usize, usize) {
    let (z, y) = (t / rows, t % rows);
    (z / nparts, z % nparts, y)
}

/// One word loop over a row: out[i] = f(i) (raw pointers: out may alias the inputs word for word, as on the device).
#[inline(always)]
fn row_map<W: Copy>(out: *mut W, f: impl Fn(usize) -> W) {
    for i in 0..N {
        let v = f(i);
        // SAFETY: row of N words inside the launch's output
        unsafe { *out.add(i) = v };
    }
}

#[inline(always)]
fn rd<W: Copy>(p: *const W, i: usize) -> W {
    // SAFETY: word i < N of a row inside the launch's inputs
    unsafe { *p.add(i) }
}

fn ew_rows<W: Hw>(a: &EwArgs, f: impl Fn(&EwArgs, usize, usize, usize, W) + Sync + Send) {
    let rows = rs_len(&a.rs);
    par_for((a.k * a.nparts) as usize * rows, a, |a, t| {
        let (c, part, y) = split(t, rows, a.nparts as usize);
        let row = rs_row(&a.rs, y);
        f(a, c, part, y, W::from_u64(a.pr_p(row)));
    });
}

trait PrimeOf {
    fn pr_p(&self, row: usize) -> u64;
}
impl PrimeOf for EwArgs {
    fn pr_p(&self, row: usize) -> u64 {
        // SAFETY: prime table indexed by prime id
        unsafe { (*self.pr.add(row)).p }
    }
}

#[inline(always)]
fn prime(pr: *const PrimeConst, row: usize) -> &'static PrimeConst {
    // SAFETY: the prime table outlives every launch reading it (device constant data)
    unsafe { &*pr.add(row) }
}

pub extern "C" fn ew_add<W: Hw>(a: *const EwArgs, s: S) -> i32 {
    launch(a, s, |_| 0, |a| {
        ew_rows::<W>(a, |a, c, part, y, p| {
            let row = rs_row(&a.rs, y);
            let (x, z, o) = (at::<W>(&a.a, c, part, row), at::<W>(&a.b, c, part, row), at::<W>(&a.out, c, part, row));
            if a.sub != 0 {
                row_map(o, |i| W::sub(rd(x, i), rd(z, i), p));
            } else {
                row_map(o, |i| W::add(rd(x, i), rd(z, i), p));
            }
        })
    })
}

pub extern "C" fn ew_scale<W: Hw>(a: *const EwArgs, s: S) -> i32 {
    launch(a, s, |_| 0, |a| {
        ew_rows::<W>(a, |a, c, part, y, p| {
            let row = rs_row(&a.rs, y);
            let o = at::<W>(&a.out, c, part, row);
            if a.zero_p != 0 && y >= a.rs.nq as usize {
                return row_map(o, |_| W::default());
            }
            let (sc, scs) = (el::<W>(a.sc, row), el::<W>(a.scs, row));
            let x = at::<W>(&a.a, c, part, row);
            row_map(o, |i| W::mul_shoup(rd(x, i), sc, scs, p));
        })
    })
}

pub extern "C" fn ew_sub_mul<W: Hw>(a: *const EwArgs, s: S) -> i32 {
    launch(a, s, |_| 0, |a| {
        ew_rows::<W>(a, |a, c, part, y, p| {
            let row = rs_row(&a.rs, y);
            let (sc, scs) = (el::<W>(a.sc, row), el::<W>(a.scs, row));
            let (x, z, o) = (at::<W>(&a.a, c, part, row), at::<W>(&a.b, c, part, row), at::<W>(&a.out, c, part, row));
            row_map(o, |i| W::mul_shoup(W::sub(rd(x, i), rd(z, i), p), sc, scs, p));
        })
    })
}

pub extern "C" fn ew_copy<W: Hw>(a: *const EwArgs, s: S) -> i32 {
    launch(a, s, |_| 0, |a| {
        ew_rows::<W>(a, |a, c, part, y, _| {
            let row = rs_row(&a.rs, y);
            let (x, o) = (at::<W>(&a.a, c, part, row), at::<W>(&a.out, c, part, row));
            row_map(o, |i| rd(x, i));
        })
    })
}

pub extern "C" fn pt_dot<W: Hw>(a: *const PtDotArgs, s: S) -> i32 {
    launch(a, s, |a| if a.k > 8 { -1 } else { 0 }, |a| {
        par_for(rs_len(&a.rs), a, |a, y| {
            let row = rs_row(&a.rs, y);
            let prow = if y < a.rs.nq as usize { y } else { a.pt_prow as usize + (y - a.rs.nq as usize) };
            let q = prime(a.pr, row);
            let k = a.k as usize;
            let mut s0 = crate::raw::buf::<W::Acc>(k * N);
            let mut s1 = crate::raw::buf::<W::Acc>(k * N);
            for j in 0..a.gstep as usize {
                s0.fill(W::Acc::default());
                s1.fill(W::Acc::default());
                for b in 0..a.bstep as usize {
                    let d = a.bstep as usize * j + b;
                    if d >= a.npts as usize {
                        break;
                    }
                    // SAFETY: diagonal pointer table of npts entries
                    let pt = unsafe { *(a.pts as *const *const W).add(d) };
                    // SAFETY: row prow of ptw words
                    let pt = unsafe { pt.add(prow * a.ptw as usize) };
                    for c in 0..k {
                        // SAFETY: rotation (c, b) row inside the launch's inputs
                        let r0 = unsafe { (a.rot.base as *const W).add(c * a.rot.ct_stride as usize + b * a.rot_stride as usize + row * N) };
                        // SAFETY: as above, part 1
                        let r1 = unsafe { r0.add(a.rot.part_stride as usize) };
                        let (o0, o1) = (&mut s0[c * N..(c + 1) * N], &mut s1[c * N..(c + 1) * N]);
                        for i in 0..N {
                            let w = rd(pt, ptidx(i, a.enc));
                            o0[i] += W::wmul(rd(r0, i), w);
                            o1[i] += W::wmul(rd(r1, i), w);
                        }
                    }
                }
                for c in 0..k {
                    // SAFETY: result (c, j) row
                    let o = unsafe { (a.res.base as *mut W).add(c * a.res.ct_stride as usize + j * a.res_stride as usize + row * N) };
                    // SAFETY: as above, part 1
                    let o1 = unsafe { o.add(a.res.part_stride as usize) };
                    row_map(o, |i| W::reduce(s0[c * N + i], q));
                    row_map(o1, |i| W::reduce(s1[c * N + i], q));
                }
            }
        })
    })
}

/// Words per tensor task: a row splits into N / TENSOR_CH tasks, so a launch over few rows still fills the pool.
const TENSOR_CH: usize = 2048;
const _: () = assert!(N % TENSOR_CH == 0);

pub extern "C" fn tensor<W: Hw>(a: *const TensorArgs, s: S) -> i32 {
    launch(a, s, |_| 0, |a| {
        let n = a.n as usize;
        let nch = N / TENSOR_CH;
        par_for(a.k as usize * n * nch, a, |a, t| {
            let (c, row, lo) = (t / (n * nch), (t / nch) % n, (t % nch) * TENSOR_CH);
            let q = prime(a.pr, row);
            let p = W::from_u64(q.p);
            let np = (a.npairs as usize).min(2);
            let row_of = |r: &PolyRef, part: usize| {
                // SAFETY: TENSOR_CH words of a row of ciphertext c inside the launch's buffers
                unsafe { std::slice::from_raw_parts_mut(at::<W>(r, c, part, row).add(lo), TENSOR_CH) }
            };
            let (mut s0, mut s1, mut s2) = ([W::Acc::default(); TENSOR_CH], [W::Acc::default(); TENSOR_CH], [W::Acc::default(); TENSOR_CH]);
            for j in 0..np {
                let (x, y) = if j == 0 { (&a.a0, &a.b0) } else { (&a.a1, &a.b1) };
                let (x0, x1, y0, y1) = (row_of(x, 0), row_of(x, 1), row_of(y, 0), row_of(y, 1));
                v::mac2::<W>(&mut s0, &mut s1, x0, y0, y1);
                v::mac2::<W>(&mut s1, &mut s2, x1, y0, y1);
            }
            // every input word of the chunk is read: out may alias an input from here on
            let (mut r0, mut r1, mut r2) = ([W::default(); TENSOR_CH], [W::default(); TENSOR_CH], [W::default(); TENSOR_CH]);
            v::reduce::<W>(&s0, q, &mut r0);
            v::reduce::<W>(&s1, q, &mut r1);
            v::reduce::<W>(&s2, q, &mut r2);
            let (o0, o1, d2) = (row_of(&a.out, 0), row_of(&a.out, 1), row_of(&a.d2, 1));
            if a.acc != 0 {
                v::add_rows::<W>(&mut r0, o0, p);
                v::add_rows::<W>(&mut r1, o1, p);
            }
            if a.acc & 2 != 0 {
                v::add_rows::<W>(&mut r2, d2, p);
            }
            if a.dbl != 0 {
                for r in [&mut r0, &mut r1, &mut r2] {
                    r.iter_mut().for_each(|x| *x = W::add(*x, *x, p));
                }
            }
            o0.copy_from_slice(&r0);
            o1.copy_from_slice(&r1);
            d2.copy_from_slice(&r2);
            if !a.coef.is_null() {
                // SAFETY: coefficient row of ciphertext c when the output is requested
                let cf = unsafe { std::slice::from_raw_parts_mut((a.coef as *mut W).add(c * a.coef_ct_stride as usize + row * N + lo), TENSOR_CH) };
                cf.copy_from_slice(&r2);
            }
        })
    })
}

pub extern "C" fn tensor_sum<W: Hw>(a: *const TensorSumArgs, s: S) -> i32 {
    launch(a, s, |a| if a.npairs < 1 || a.npairs as usize > TSUM_MAX { -1 } else { 0 }, |a| {
        let n = a.n as usize;
        let nch = N / TENSOR_CH;
        par_for(a.k as usize * n * nch, a, |a, t| {
            let (c, row, lo) = (t / (n * nch), (t / nch) % n, (t % nch) * TENSOR_CH);
            let q = prime(a.pr, row);
            let p = W::from_u64(q.p);
            let row_of = |r: &PolyRef, part: usize| {
                // SAFETY: TENSOR_CH words of a row of ciphertext c inside the launch's buffers
                unsafe { std::slice::from_raw_parts_mut(at::<W>(r, c, part, row).add(lo), TENSOR_CH) }
            };
            let (mut s0, mut s1, mut s2) = ([W::Acc::default(); TENSOR_CH], [W::Acc::default(); TENSOR_CH], [W::Acc::default(); TENSOR_CH]);
            for j in 0..a.npairs as usize {
                let (x0, x1, y0, y1) = (row_of(&a.a[j], 0), row_of(&a.a[j], 1), row_of(&a.b[j], 0), row_of(&a.b[j], 1));
                v::mac2::<W>(&mut s0, &mut s1, x0, y0, y1);
                v::mac2::<W>(&mut s1, &mut s2, x1, y0, y1);
            }
            let (mut r0, mut r1, mut r2) = ([W::default(); TENSOR_CH], [W::default(); TENSOR_CH], [W::default(); TENSOR_CH]);
            v::reduce::<W>(&s0, q, &mut r0);
            v::reduce::<W>(&s1, q, &mut r1);
            v::reduce::<W>(&s2, q, &mut r2);
            let (o0, o1, d2) = (row_of(&a.out, 0), row_of(&a.out, 1), row_of(&a.d2, 1));
            if a.acc != 0 {
                v::add_rows::<W>(&mut r0, o0, p);
                v::add_rows::<W>(&mut r1, o1, p);
            }
            if a.acc & 2 != 0 {
                v::add_rows::<W>(&mut r2, d2, p);
            }
            o0.copy_from_slice(&r0);
            o1.copy_from_slice(&r1);
            d2.copy_from_slice(&r2);
        })
    })
}

pub extern "C" fn md_epilogue<W: Hw>(a: *const MdEpiArgs, s: S) -> i32 {
    launch(a, s, |_| 0, |a| {
        let n = a.n as usize;
        par_for(2 * a.kk as usize * n, a, |a, t| {
            let (c, part, row) = split(t, n, 2);
            let p = W::from_u64(prime(a.pr, row).p);
            let (bi, bis) = (el::<W>(a.binv, row), el::<W>(a.binvs, row));
            let kk = (!a.k.is_null() && part == 0).then(|| W::red1(el::<W>(a.k, row), p));
            let (d, x, tt) = (at::<W>(&a.dst, c, part, row), at::<W>(&a.x, c, part, row), at::<W>(&a.t, c, part, row));
            row_map(d, |i| {
                let mut r = W::add(rd(d, i), W::mul_shoup(W::sub(rd(x, i), rd(tt, i), p), bi, bis, p), p);
                if a.dbl != 0 {
                    r = W::add(r, r, p);
                }
                if let Some(k) = kk {
                    r = W::add(r, k, p);
                }
                r
            });
        })
    })
}

pub extern "C" fn md_epilogue_to<W: Hw>(a: *const MdEpiToArgs, s: S) -> i32 {
    launch(a, s, |_| 0, |b| {
        let n = b.m.n as usize;
        par_for(2 * b.m.kk as usize * n, b, |b, t| {
            let a = &b.m;
            let (c, part, row) = split(t, n, 2);
            let p = W::from_u64(prime(a.pr, row).p);
            let (bi, bis) = (el::<W>(a.binv, row), el::<W>(a.binvs, row));
            let (x, tt, y, d) = (at::<W>(&a.x, c, part, row), at::<W>(&a.t, c, part, row), at::<W>(&b.a, c, part, row), at::<W>(&a.dst, c, part, row));
            row_map(d, |i| {
                let v = W::mul_shoup(W::sub(rd(x, i), rd(tt, i), p), bi, bis, p);
                if b.sub != 0 {
                    W::sub(rd(y, i), v, p)
                } else {
                    W::add(rd(y, i), v, p)
                }
            });
        })
    })
}

fn wsum_run<W: Hw>(a: &WsumArgs, k: *const std::ffi::c_void) {
    let n = a.n as usize;
    let sh = crate::raw::Shared(&k);
    let sh = &sh;
    par_for(a.k as usize * n, a, |a, t| {
        let (c, row) = (t / n, t % n);
        let p = W::from_u64(prime(a.pr, row).p);
        let kr = (!sh.get().is_null()).then(|| W::red1(el::<W>(*sh.get(), row), p));
        for part in 0..2 {
            let o = at::<W>(&a.out, c, part, row);
            let mut acc = crate::raw::buf::<W>(N);
            for j in 0..a.nsrc as usize {
                let (w, ws) = (el::<W>(a.w, j * a.rowstride as usize + row), el::<W>(a.ws, j * a.rowstride as usize + row));
                let x = at::<W>(&a.src[j], c, part, row);
                for (i, r) in acc.iter_mut().enumerate() {
                    *r = W::add(*r, W::mul_shoup(rd(x, i), w, ws, p), p);
                }
            }
            if part == 0 {
                if let Some(k) = kr {
                    acc.iter_mut().for_each(|r| *r = W::add(*r, k, p));
                }
            }
            row_map(o, |i| acc[i]);
        }
    })
}

pub extern "C" fn wsum<W: Hw>(a: *const WsumArgs, s: S) -> i32 {
    launch(a, s, |a| if a.nsrc > 8 { -1 } else { 0 }, |a| wsum_run::<W>(a, std::ptr::null()))
}

pub extern "C" fn wsum_k<W: Hw>(a: *const WsumKArgs, s: S) -> i32 {
    launch(a, s, |a| if a.w.nsrc > 8 || a.k.is_null() { -1 } else { 0 }, |a| wsum_run::<W>(&a.w, a.k))
}

pub extern "C" fn add_const<W: Hw>(a: *const AddConstArgs, s: S) -> i32 {
    launch(a, s, |_| 0, |a| {
        let n = a.n as usize;
        par_for(a.kk as usize * n, a, |a, t| {
            let (c, row) = (t / n, t % n);
            let p = W::from_u64(prime(a.pr, row).p);
            let k = W::red1(el::<W>(a.k, row), p);
            let d = at::<W>(&a.x, c, 0, row);
            row_map(d, |i| W::add(rd(d, i), k, p));
        })
    })
}

pub extern "C" fn mul_table<W: Hw>(a: *const MulTableArgs, s: S) -> i32 {
    launch(a, s, |_| 0, |a| {
        let n = a.n as usize;
        par_for(2 * a.k as usize * n, a, |a, t| {
            let (c, part, row) = split(t, n, 2);
            let q = prime(a.pr, row);
            // SAFETY: table row
            let tb = unsafe { (a.t as *const W).add(row * N) };
            let (x, o) = (at::<W>(&a.inp, c, part, row), at::<W>(&a.out, c, part, row));
            row_map(o, |i| W::mul(rd(x, i), rd(tb, i), q));
        })
    })
}

pub extern "C" fn mul_add<W: Hw>(a: *const MulAddArgs, s: S) -> i32 {
    launch(a, s, |a| if a.sub < 0 || a.sub > 2 { -1 } else { 0 }, |b| {
        let n = b.m.n as usize;
        par_for(2 * b.m.k as usize * n, b, |b, t| {
            let a = &b.m;
            let (c, part, row) = split(t, n, 2);
            let q = prime(a.pr, row);
            let p = W::from_u64(q.p);
            // SAFETY: table row
            let tb = unsafe { (a.t as *const W).add(row * N) };
            let (x, y, o) = (at::<W>(&a.inp, c, part, row), at::<W>(&b.y, c, part, row), at::<W>(&a.out, c, part, row));
            row_map(o, |i| {
                let m = W::mul(rd(x, i), rd(tb, i), q);
                match b.sub {
                    0 => W::add(rd(y, i), m, p),
                    1 => W::sub(rd(y, i), m, p),
                    _ => W::sub(m, rd(y, i), p),
                }
            });
        })
    })
}

fn combine<W: Hw>(a: &RescaleCombineArgs, out: &PolyRef) {
    let n = a.n as usize;
    let sh = crate::raw::Shared(out);
    let sh = &sh;
    par_for(2 * a.k as usize * n, a, |a, t| {
        let (c, part, row) = split(t, n, 2);
        let p = W::from_u64(prime(a.pr, row).p);
        let (al, als) = (el::<W>(a.alpha, row), el::<W>(a.alphas, row));
        let (x, tt, o) = (at::<W>(&a.x, c, part, row), at::<W>(&a.tmp, c, part, row), at::<W>(sh.get(), c, part, row));
        row_map(o, |i| W::add(W::mul_shoup(rd(x, i), al, als, p), rd(tt, i), p));
    })
}

pub extern "C" fn rescale_combine<W: Hw>(a: *const RescaleCombineArgs, s: S) -> i32 {
    launch(a, s, |_| 0, |a| combine::<W>(a, &a.x))
}

pub extern "C" fn rescale_combine_to<W: Hw>(a: *const RescaleCombineToArgs, s: S) -> i32 {
    launch(a, s, |_| 0, |a| combine::<W>(&a.a, &a.out))
}

pub extern "C" fn automorph<W: Hw>(a: *const AutomorphArgs, s: S) -> i32 {
    launch(a, s, |a| if a.k & 1 == 0 { -1 } else { 0 }, |a| {
        let rows = rs_len(&a.rs);
        let slots: Vec<u32> = (0..N as u32).map(|o| automorph_slot(a.k, o, LOG_N)).collect();
        let sh = crate::raw::Shared(&slots);
        let sh = &sh;
        par_for((a.kk * a.nparts) as usize * rows, a, |a, t| {
            let (c, part, y) = split(t, rows, a.nparts as usize);
            let row = rs_row(&a.rs, y);
            let (x, o) = (at::<W>(&a.inp, c, part, row), at::<W>(&a.out, c, part, row));
            if a.has_add != 0 {
                let p = W::from_u64(prime(a.pr, row).p);
                let ad = at::<W>(&a.add, c, part, row);
                row_map(o, |i| W::add(rd(x, sh.get()[i] as usize), rd(ad, i), p));
            } else {
                row_map(o, |i| rd(x, sh.get()[i] as usize));
            }
        })
    })
}
