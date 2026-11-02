// Batched fast base conversion (bconv.cuh, BconvArgs): per job, y_i = x_i pre_i mod s_i, out_t = sum_i y_i M[i][t]
// mod t, the exact jobs' correction from the FMA double estimate in source order (f64::mul_add is the device's
// __fma_rn). One task per (job, block of coefficients).
use crate::arith::Hw;
use crate::launch;
use crate::raw::{el, par_for};
use pc_kernels::args::*;

const BLK: usize = 1024;

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
            let mut y = vec![W::default(); ns * BLK];
            for i in 0..ns {
                let q = pr(pool(sprime + i).u64() as usize);
                let pi = pool(pre + i);
                // SAFETY: source row inside the launch's input
                let x = unsafe { (a.src as *const W).add((j.src_off as usize + pool(srow + i).u64() as usize) * N + c0) };
                for c in 0..BLK {
                    // SAFETY: as above
                    y[i * BLK + c] = W::mul(unsafe { *x.add(c) }, pi, q);
                }
            }
            let mut u = vec![0u64; BLK];
            let mut bmod_off = 0usize;
            if j.exact != 0 {
                // the doubles start at the first 8-byte aligned word past M
                let mut rbw = j.pool_off as usize + m + ns * nt;
                let base = a.pool as usize;
                while (base + rbw * std::mem::size_of::<W>()) % 8 != 0 {
                    rbw += 1;
                }
                let rb = (base + rbw * std::mem::size_of::<W>()) as *const f64;
                for (c, uc) in u.iter_mut().enumerate() {
                    let mut v = 0.0f64;
                    for i in 0..ns {
                        // SAFETY: ns doubles in the pool
                        v = (y[i * BLK + c].u64() as f64).mul_add(unsafe { *rb.add(i) }, v);
                    }
                    *uc = (v + 0.5).floor() as u64;
                }
                bmod_off = (rbw * std::mem::size_of::<W>() + ns * 8) / std::mem::size_of::<W>();
            }
            let mut acc = vec![W::Acc::default(); BLK];
            for tt in 0..nt {
                let q = pr(pool(tprime + tt).u64() as usize);
                let p = W::from_u64(q.p);
                acc.fill(W::Acc::default());
                for i in 0..ns {
                    let mi = pool(m + i * nt + tt);
                    let yi = &y[i * BLK..(i + 1) * BLK];
                    acc.iter_mut().zip(yi).for_each(|(s, &v)| *s += W::wmul(v, mi));
                }
                let bm = if j.exact != 0 { el::<W>(a.pool, bmod_off + tt) } else { W::default() };
                // SAFETY: target row inside the launch's output
                let o = unsafe { (a.dst as *mut W).add((j.dst_off as usize + pool(trow + tt).u64() as usize) * N + c0) };
                for c in 0..BLK {
                    let mut r = W::reduce(acc[c], q);
                    if j.exact != 0 {
                        r = W::sub(r, W::reduce(W::wmul(W::from_u64(u[c]), bm), q), p);
                    }
                    // SAFETY: as above
                    unsafe { *o.add(c) = r };
                }
            }
        })
    })
}
