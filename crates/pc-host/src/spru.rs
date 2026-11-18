// SPRU (spru.cuh): extraction of the bottom modulus' LWE columns into the bootstrapping plaintext's FFT inputs, the
// batched Stockham passes with table twiddles, and the coefficient scatter. Floating point as the device compiles it:
// sincospi is libdevice's (encode.rs), the pass's complex product contracts as spru_fft_pass's, the scatter's
// c = fma(y.x, z.x, -(y.y z.y)) 2^-16 scale rounds half to even (llrint).
use crate::arith::Hw;
use crate::encode::{sincospi, stockham_pass};
use crate::launch;
use crate::raw::par_for;
use pc_kernels::args::*;

type S = CudaStream;

pub extern "C" fn spru_extract<W: Hw>(a: *const SpruExtractArgs, s: S) -> i32 {
    let bad = |a: &SpruExtractArgs| {
        if a.n <= 0 || a.h <= 0 || a.b * a.h != N as i32 || a.b % (2 * a.n) != 0 || a.k <= 0 {
            -1
        } else {
            0
        }
    };
    launch(a, s, bad, |a| {
        let n = a.n as usize;
        const BLK: usize = 4096;
        let nb = (N * n).div_ceil(BLK);
        par_for(a.k as usize * nb, a, |a, task| {
            let (c, blk) = (task / nb, task % nb);
            let (h, bb) = (a.h as usize, a.b as usize);
            let (p0, p1) = (a.p0, a.p1);
            let q0 = p0 * p1;
            let c0 = (a.c0 as *const W).wrapping_add(c * a.ct_stride as usize);
            let c1 = (a.c1 as *const W).wrapping_add(c * a.ct_stride as usize);
            let crt = |r: *const W, k: usize| -> u64 {
                // SAFETY: rows 0 and 1 of the ciphertext's part
                let (x0, x1) = unsafe { ((*r.add(k)).u64(), (*r.add(N + k)).u64()) };
                let t = ((x1 + p1 - x0 % p1) % p1) as u128 * a.inv as u128 % p1 as u128;
                x0 + p0 * t as u64
            };
            for tid in blk * BLK..((blk + 1) * BLK).min(N * n) {
                let (i, ai) = (tid / n, tid % n);
                let ka = ai * (N / n);
                let mut v = if i <= ka {
                    crt(c1, ka - i)
                } else {
                    let v = crt(c1, N + ka - i);
                    if v != 0 {
                        q0 - v
                    } else {
                        0
                    }
                };
                if i == 0 {
                    v += crt(c0, ka);
                    if v >= q0 {
                        v -= q0;
                    }
                }
                let ang = 2.0 * v as f64 / q0 as f64;
                let (sn, cs) = sincospi(ang);
                let sub = bb / (2 * n);
                let (b, rem) = (i / bb, i % bb);
                let (u, k) = (rem / sub, rem % sub);
                let slot = k * h * n + b * n + ai;
                let m2n = 2 * N as u32 - 1;
                let (mut t, mut b5, mut e) = (1u32, 5u32, slot as u32);
                while e != 0 {
                    if e & 1 != 0 {
                        t = t.wrapping_mul(b5) & m2n;
                    }
                    b5 = b5.wrapping_mul(b5) & m2n;
                    e >>= 1;
                }
                let o = (a.out as *mut [f64; 2]).wrapping_add((c * 2 * n + u) * N);
                // SAFETY: every index of the encoding's N-point input is written exactly once
                unsafe {
                    *o.add((t as usize - 1) / 2) = [a.delta * cs, a.delta * sn];
                    *o.add((2 * N - t as usize - 1) / 2) = [a.delta * cs, -(a.delta * sn)];
                }
            }
        })
    })
}

pub extern "C" fn spru_fft_pass(a: *const SpruFftArgs, s: S) -> i32 {
    launch(a, s, |a| if a.ns <= 0 || a.ns > N as i32 / 2 || a.batch <= 0 { -1 } else { 0 }, |a| {
        let sh = crate::raw::Shared(a);
        let sh = &sh;
        pc_hostrt::par(|| {
            let a = sh.get();
            let ns = a.ns as usize;
            let half = N / 2;
            // SAFETY: twiddle table of N/2 double2
            let tw = unsafe { std::slice::from_raw_parts(a.tw as *const [f64; 2], half) };
            for e in 0..a.batch as usize {
                // SAFETY: transform e of the batch
                let (inp, out) = unsafe {
                    (
                        std::slice::from_raw_parts((a.inp as *const [f64; 2]).add(e * N), N),
                        std::slice::from_raw_parts_mut((a.out as *mut [f64; 2]).add(e * N), N),
                    )
                };
                stockham_pass(inp, out, ns, true, &|k| {
                    let w = tw[k * (half / ns)];
                    (w[1], w[0])
                });
            }
        })
    })
}

pub extern "C" fn spru_scatter<W: Hw>(a: *const SpruScatterArgs, s: S) -> i32 {
    launch(a, s, |a| if a.nl <= 0 || a.batch <= 0 { -1 } else { 0 }, |a| {
        const BLK: usize = 4096;
        let nb = N / BLK;
        par_for(a.batch as usize * nb, a, |a, task| {
            let (e, b) = (task / nb, task % nb);
            let out = (a.out as *mut W).wrapping_add(e * a.out_stride as usize);
            for k in b * BLK..(b + 1) * BLK {
                // SAFETY: FFT output e and the zeta table
                let (y, z) = unsafe { (*(a.y as *const [f64; 2]).add(e * N + k), *(a.zt as *const [f64; 2]).add(k)) };
                let re = y[0].mul_add(z[0], -(y[1] * z[1])) * (1.0 / N as f64);
                let m = (re * a.scale).round_ties_even() as i64;
                let am = m.unsigned_abs();
                for l in 0..a.nl as usize {
                    // SAFETY: prime table entry l
                    let p = unsafe { (*a.pr.add(l)).p };
                    let mut r = am % p;
                    if m < 0 && r != 0 {
                        r = p - r;
                    }
                    // SAFETY: limb l of encoding e
                    unsafe { *out.add(l * N + k) = W::from_u64(r) };
                }
            }
        })
    })
}
