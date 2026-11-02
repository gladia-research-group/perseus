// Streamed-weight inner sums (coeffrows.cuh, CoeffArgs): out (+)= sum_r ct_r pt_r over limbs 0..L-1, pt_r the NTT of
// round(rows_r * scale) (half to even, clamped at 2^62 past the int64 range and counted). scratch [term][limb][N] ends
// holding the column pass's state (the first 8 stages), as on the device. One task per limb.
use crate::arith::Hw;
use crate::launch;
use crate::ntt::fwd_exact;
use crate::raw::par_for;
use pc_kernels::args::*;
use std::sync::atomic::{AtomicU64, Ordering};

fn residue<W: Hw>(f: f32, scale: f64, p: u64, ovf: *mut u64) -> W {
    let x = f as f64 * scale;
    let m = if x.abs() < 9.2e18 {
        x.round_ties_even() as i64
    } else {
        if !ovf.is_null() {
            // SAFETY: the launch's overflow counter (device-global u64, shared by the tasks)
            unsafe { AtomicU64::from_ptr(ovf) }.fetch_add(1, Ordering::Relaxed);
        }
        if x < 0.0 {
            -(1i64 << 62)
        } else {
            1i64 << 62
        }
    };
    let v = m.unsigned_abs() % p;
    W::from_u64(if m < 0 && v != 0 { p - v } else { v })
}

pub extern "C" fn coeff_sum<W: Hw>(a: *const CoeffArgs, s: CudaStream) -> i32 {
    let bad = |a: &CoeffArgs| {
        if a.n < 1 || a.n > COEFF_MAXT as i32 || a.l < 1 || a.l > COEFF_MAXL as i32 {
            -1
        } else if a.scratch.is_null() || a.out.is_null() || a.fw.is_null() || a.fws.is_null() || a.pr.is_null() {
            -2
        } else {
            0
        }
    };
    launch(a, s, bad, |a| {
        par_for(a.l as usize, a, |a, l| {
            // SAFETY: prime table entry l
            let q = unsafe { &*a.pr.add(l) };
            let p = W::from_u64(q.p);
            // SAFETY: twiddle rows of prime l
            let (w, ws) = unsafe { (std::slice::from_raw_parts((a.fw as *const W).add(l * N), N), std::slice::from_raw_parts((a.fws as *const W).add(l * N), N)) };
            let ps = a.part_stride as usize;
            let out = a.out as *mut W;
            let (mut acc0, mut acc1) = (vec![W::default(); N], vec![W::default(); N]);
            if a.accumulate != 0 {
                for i in 0..N {
                    // SAFETY: row l of both output parts
                    unsafe {
                        acc0[i] = *out.add(l * N + i);
                        acc1[i] = *out.add(ps + l * N + i);
                    }
                }
            }
            let mut x = vec![W::default(); N];
            for r in 0..a.n as usize {
                // SAFETY: float row r (N coefficients)
                let src = unsafe { std::slice::from_raw_parts(a.rows[r], N) };
                for (xi, &f) in x.iter_mut().zip(src) {
                    *xi = residue::<W>(f, a.scale, q.p, a.overflow);
                }
                // SAFETY: scratch row (r, l)
                let scr = unsafe { std::slice::from_raw_parts_mut((a.scratch as *mut W).add((r * a.l as usize + l) * N), N) };
                fwd_exact(&mut x, w, ws, p, Some(scr));
                let c0 = (a.ct[r] as *const W).wrapping_add(l * N);
                for i in 0..N {
                    // SAFETY: row l of term r's ciphertext parts
                    let (u, v) = unsafe { (*c0.add(i), *c0.add(ps + i)) };
                    acc0[i] = W::reduce(W::wmul(x[i], u) + W::wmul(acc0[i], W::from_u64(1)), q);
                    acc1[i] = W::reduce(W::wmul(x[i], v) + W::wmul(acc1[i], W::from_u64(1)), q);
                }
            }
            for i in 0..N {
                // SAFETY: as above
                unsafe {
                    *out.add(l * N + i) = acc0[i];
                    *out.add(ps + l * N + i) = acc1[i];
                }
            }
        })
    })
}
