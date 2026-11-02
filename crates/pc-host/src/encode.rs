// Device encode (encode.cuh, EncodeArgs), bit for bit: slot placement, the radix-2 Stockham passes, the coefficient
// extraction and the per-prime scatter. The device's floating point is reproduced exactly: sincospi is libdevice's
// (argument reduction by round-half-even of 2x, the minimax polynomials below, compiled as these FMAs), and the
// complex products contract as the compiled kernels do (t.x = fma(v.x, cs, -(v.y sn)), t.y = fma(v.x, sn, v.y cs);
// c_k = fma(Y.x, cs, -(Y.y sn)) 2^-16 scale).
use crate::arith::Hw;
use crate::launch;
use crate::raw::el;
use pc_kernels::args::*;
use rayon::prelude::*;
use std::sync::OnceLock;

const COS: [u64; 7] = [
    0xBDA8FF8320FD8164,
    0x3E21EEA7C1EF8528,
    0xBE927E4F8E06E6D9,
    0x3EFA01A019DDBCE9,
    0xBF56C16C16C15D47,
    0x3FA5555555555551,
    0xBFE0000000000000,
];
const SIN: [u64; 6] = [0x3DE5DB65F9785EBA, 0xBE5AE5F12CB0D246, 0x3EC71DE369ACE392, 0xBF2A01A019DB62A1, 0x3F81111111110818, 0xBFC5555555555554];
const PI_HI: u64 = 0x400921FB54442D18;
const PI_LO: u64 = 0x3CA1A62633145C07;

#[inline(always)]
fn f(b: u64) -> f64 {
    f64::from_bits(b)
}

/// libdevice sincospi(x) as compiled for the device: (sin(pi x), cos(pi x)).
pub fn sincospi(x: f64) -> (f64, f64) {
    let hi = (x.to_bits() >> 32) as u32;
    let x = if hi << 1 > 0x8680_0000 { x * 0.0 } else { x };
    // 2x by the exponent field, rounded half to even
    let x2 = f64::from_bits(x.to_bits().wrapping_add(1u64 << 52));
    let qf = x2.round_ties_even();
    let q = qf as i64;
    let r = qf.mul_add(-0.5, x);
    let t = r.mul_add(f(PI_HI), r * f(PI_LO));
    let t2 = t * t;
    let mut c = t2.mul_add(f(COS[0]), f(COS[1]));
    for &k in &COS[2..] {
        c = c.mul_add(t2, f(k));
    }
    c = c.mul_add(t2, 1.0);
    let mut s = t2.mul_add(f(SIN[0]), f(SIN[1]));
    for &k in &SIN[2..] {
        s = s.mul_add(t2, f(k));
    }
    s = s.mul_add(t2, 0.0);
    let s = s.mul_add(t, t);
    let (mut sn, mut cs) = if q & 1 != 0 { (c, -s) } else { (s, c) };
    if q & 2 != 0 {
        sn = -sn;
        cs = -cs;
    }
    let cs = cs + 0.0;
    let sn = if x == x.trunc() { x * 0.0 } else { sn };
    (sn, cs)
}

/// Twiddles of the Stockham pass of half-size Ns: entry Ns - 1 + k = sincospi(-k / Ns), k < Ns.
fn pass_twiddles() -> &'static [(f64, f64)] {
    static T: OnceLock<Vec<(f64, f64)>> = OnceLock::new();
    T.get_or_init(|| {
        let mut v = vec![(0.0, 0.0); N];
        let mut ns = 1;
        while ns < N {
            for k in 0..ns {
                v[ns - 1 + k] = sincospi(-(k as f64) / ns as f64);
            }
            ns <<= 1;
        }
        v
    })
}

fn coeff_twiddles() -> &'static [(f64, f64)] {
    static T: OnceLock<Vec<(f64, f64)>> = OnceLock::new();
    T.get_or_init(|| (0..N).map(|k| sincospi(-(k as f64) / N as f64)).collect())
}

/// One forward Stockham pass of half-size ns over N complex values (`tw` the pass's twiddles).
pub fn stockham_pass(inp: &[[f64; 2]], out: &mut [[f64; 2]], ns: usize, tw: &(dyn Fn(usize) -> (f64, f64) + Sync)) {
    let half = inp.len() / 2;
    let outp = out.as_mut_ptr() as usize;
    (0..half).into_par_iter().with_min_len(4096).for_each(|j| {
        let (v0, v1) = (inp[j], inp[j + half]);
        let k = j % ns;
        let (sn, cs) = tw(k);
        let t = [v1[0].mul_add(cs, -(v1[1] * sn)), v1[0].mul_add(sn, v1[1] * cs)];
        let idx = (j / ns) * ns * 2 + k;
        let o = outp as *mut [f64; 2];
        // SAFETY: (idx, idx + ns) are distinct across j (the Stockham output map is a permutation)
        unsafe {
            *o.add(idx) = [v0[0] + t[0], v0[1] + t[1]];
            *o.add(idx + ns) = [v0[0] - t[0], v0[1] - t[1]];
        }
    });
}

fn split_of(m: f64) -> i32 {
    if !(m > 0.0) {
        return 0;
    }
    // frexp: m = f 2^e, f in [0.5, 1)
    let bits = m.to_bits();
    let ex = ((bits >> 52) & 0x7FF) as i32;
    let (f, e) = if ex == 0 {
        let (f, e) = (m * f(0x4350000000000000), -54); // 2^54
        let b = f.to_bits();
        (f64::from_bits((b & !(0x7FFu64 << 52)) | (1022u64 << 52)), e + ((b >> 52) & 0x7FF) as i32 - 1022)
    } else {
        (f64::from_bits((bits & !(0x7FFu64 << 52)) | (1022u64 << 52)), ex - 1022)
    };
    let logc = if f == 0.5 { e - 1 } else { e };
    if logc > 61 {
        logc - 61
    } else {
        0
    }
}

pub extern "C" fn encode<W: Hw>(a: *const EncodeArgs, s: CudaStream) -> i32 {
    launch(a, s, |_| 0, |a| {
        let sh = crate::raw::Shared(a);
        let sh = &sh;
        pc_hostrt::par(|| {
            let a = sh.get();
            // SAFETY: the launch's buffers: v N/2 double2, a / b N double2, c N doubles, out nl rows
            let (v, bufa, bufb, c) = unsafe {
                (
                    std::slice::from_raw_parts(a.v as *const [f64; 2], N / 2),
                    std::slice::from_raw_parts_mut(a.a as *mut [f64; 2], N),
                    std::slice::from_raw_parts_mut(a.b as *mut [f64; 2], N),
                    std::slice::from_raw_parts_mut(a.c as *mut f64, N),
                )
            };
            let m2n = 2 * N as u32 - 1;
            for (j, x) in v.iter().enumerate() {
                let (mut t, mut b, mut e) = (1u32, 5u32, j as u32);
                while e != 0 {
                    if e & 1 != 0 {
                        t = t.wrapping_mul(b) & m2n;
                    }
                    b = b.wrapping_mul(b) & m2n;
                    e >>= 1;
                }
                bufa[(t as usize - 1) / 2] = *x;
                bufa[(2 * N - t as usize - 1) / 2] = [x[0], -x[1]];
            }
            let tw = pass_twiddles();
            let (mut inp, mut out) = (bufa, bufb);
            let mut ns = 1;
            while ns < N {
                stockham_pass(inp, out, ns, &|k| tw[ns - 1 + k]);
                std::mem::swap(&mut inp, &mut out);
                ns <<= 1;
            }
            let ct = coeff_twiddles();
            let scale = a.scale;
            c.par_iter_mut().enumerate().with_min_len(4096).for_each(|(k, ck)| {
                let (sn, cs) = ct[k];
                let y = inp[k];
                *ck = y[0].mul_add(cs, -(y[1] * sn)) * (1.0 / N as f64) * scale;
            });
            let cmax = c.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
            // SAFETY: the launch's max word
            unsafe { *a.cmax = cmax.to_bits() };
            let split = split_of(cmax);
            let nl = a.nl as usize;
            let out = a.out as usize;
            (0..nl).into_par_iter().for_each(|l| {
                let a = sh.get();
                // SAFETY: prime table entry l
                let q = unsafe { &*a.pr.add(l) };
                let p = W::from_u64(q.p);
                let mut fct = el::<W>(a.deg_factor, l);
                for _ in 0..split {
                    fct = W::add(fct, fct, p);
                }
                for (k, &ck) in c.iter().enumerate() {
                    let m = (ck * 2f64.powi(-split)).round() as i64;
                    let am = m.unsigned_abs();
                    let mut r = am % q.p;
                    if m < 0 && r != 0 {
                        r = q.p - r;
                    }
                    // SAFETY: limb l of the output
                    unsafe { *(out as *mut W).add(l * N + k) = W::mul(W::from_u64(r), fct, q) };
                }
            });
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sincospi_is_accurate() {
        for k in 0..4096 {
            let x = -(k as f64) / 4096.0;
            let (s, c) = sincospi(x);
            let (ws, wc) = (std::f64::consts::PI * x).sin_cos();
            assert!((s - ws).abs() < 4e-16 && (c - wc).abs() < 4e-16, "{x}: {s} {ws} {c} {wc}");
        }
        assert_eq!(sincospi(-1.0).0.to_bits(), (-0.0f64).to_bits());
        assert_eq!(sincospi(0.0), (0.0, 1.0));
    }
}
