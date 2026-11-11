// Batched fast base conversion (bconv.cuh, BconvArgs): per job, y_i = x_i pre_i mod s_i, out_t = sum_i y_i M[i][t]
// mod t, the exact jobs' correction from the FMA double estimate in source order (one fused multiply-add per source,
// as the device's __fma_rn chain), or from the integer one (exact == 2). One task per (job, block of coefficients);
// rows through the vector primitives.
use crate::arith::Hw;
use crate::launch;
use crate::raw::{buf, el, par_for};
use crate::simd::w as v;
use pc_kernels::args::*;

const BLK: usize = 2048;

pub extern "C" fn bconv<W: Hw>(a: *const BconvArgs, s: CudaStream) -> i32 {
    launch(a, s, |_| 0, |a| {
        let nb = N / BLK;
        par_for(a.njobs as usize * nb, a, |a, t| {
            // SAFETY: job table of njobs entries
            let j = unsafe { *a.jobs.add(t / nb) };
            let c0 = (t % nb) * BLK;
            let (ns, nt) = (j.nsrc as usize, j.ntgt as usize);
            let pool = |i: usize| el::<W>(a.pool, j.pool_off as usize + i);
            let pr = |id: usize| -> &PrimeConst {
                // SAFETY: prime table by id
                unsafe { &*a.pr.add(id) }
            };
            let (srow, sprime, pre, trow, tprime, m) = (0, ns, 2 * ns, 3 * ns, 3 * ns + nt, 3 * ns + 2 * nt);
            let mut y = buf::<W>(ns * BLK);
            let mut acc = buf::<W::Acc>(BLK);
            for i in 0..ns {
                let q = pr(pool(sprime + i).u64() as usize);
                // SAFETY: source row inside the launch's input
                let x = unsafe { std::slice::from_raw_parts((a.src as *const W).add((j.src_off as usize + pool(srow + i).u64() as usize) * N + c0), BLK) };
                v::mul_s::<W>(&mut y[i * BLK..(i + 1) * BLK], x, pool(pre + i), q, &mut acc);
            }
            let mut u = buf::<u64>(BLK);
            let mut bmod_off = 0usize;
            if j.exact != 0 {
                // the doubles start at the first 8-byte aligned word past M
                let mut rbw = j.pool_off as usize + m + ns * nt;
                let base = a.pool as usize;
                while (base + rbw * std::mem::size_of::<W>()) % 8 != 0 {
                    rbw += 1;
                }
                let rb = (base + rbw * std::mem::size_of::<W>()) as *const f64;
                let mut vd = buf::<f64>(BLK);
                u.fill(0);
                for i in 0..if j.exact == 2 { ns } else { 0 } {
                    // SAFETY: ns 64-bit reciprocals in the pool
                    let rc = unsafe { *(rb as *const u64).add(i) };
                    let yi = &y[i * BLK..(i + 1) * BLK];
                    if W::BITS < 64 {
                        for (f, &yv) in u.iter_mut().zip(yi) {
                            *f += yv.u64() * rc;
                        }
                    } else {
                        let sh = 68 - pr(pool(sprime + i).u64() as usize).p.leading_zeros();
                        for (f, &yv) in u.iter_mut().zip(yi) {
                            *f += ((yv.u64() as u128 * rc as u128) >> sh) as u64;
                        }
                    }
                }
                for i in 0..if j.exact == 2 { 0 } else { ns } {
                    // SAFETY: ns doubles in the pool
                    let r = unsafe { *rb.add(i) };
                    let yi = &y[i * BLK..(i + 1) * BLK];
                    if W::BITS == 32 {
                        // SAFETY: W is u32
                        crate::simd::fma_f64(&mut vd, unsafe { std::slice::from_raw_parts(yi.as_ptr() as *const u32, BLK) }, r);
                    } else {
                        for (d, &yv) in vd.iter_mut().zip(yi) {
                            *d = (yv.u64() as f64).mul_add(r, *d);
                        }
                    }
                }
                for (uc, &d) in u.iter_mut().zip(vd.iter()) {
                    *uc = if j.exact == 2 { (*uc + (1 << 58)) >> 59 } else { (d + 0.5).floor() as u64 };
                }
                bmod_off = (rbw * std::mem::size_of::<W>() + ns * 8) / std::mem::size_of::<W>();
            }
            let mut r = buf::<W>(BLK);
            if j.exact == 2 {
                for (rc, &uc) in r.iter_mut().zip(u.iter()) {
                    *rc = W::from_u64(uc);
                }
            }
            for tt in 0..nt {
                let q = pr(pool(tprime + tt).u64() as usize);
                let p = W::from_u64(q.p);
                acc.fill(W::Acc::default());
                for i in 0..ns {
                    v::mac_s::<W>(&mut acc, &y[i * BLK..(i + 1) * BLK], pool(m + i * nt + tt));
                }
                // SAFETY: target row inside the launch's output
                let o = unsafe { std::slice::from_raw_parts_mut((a.dst as *mut W).add((j.dst_off as usize + pool(trow + tt).u64() as usize) * N + c0), BLK) };
                if j.exact == 2 {
                    v::mac_s::<W>(&mut acc, &r, el::<W>(a.pool, bmod_off + tt));
                }
                v::reduce::<W>(&acc, q, o);
                if j.exact == 1 {
                    let bm = el::<W>(a.pool, bmod_off + tt);
                    for (c, oc) in o.iter_mut().enumerate() {
                        r[c] = W::reduce(W::wmul(W::from_u64(u[c]), bm), q);
                        *oc = W::sub(*oc, r[c], p);
                    }
                }
            }
        })
    })
}
