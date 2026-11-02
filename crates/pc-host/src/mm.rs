// The u32 formulas shared by the PCMM, CCMM and TFHE-bridge launchers (pcmm.cuh / ccmm.cuh / tfhe.cuh): the 64-bit
// Barrett product with the launch's mu, the centred residue, signed residues, the balanced base-256 digits, and the
// op-coded transforms (count polynomials in place, prime of each from the launch).
use crate::ntt_g;
use crate::raw::par_for;

/// PC_HOST_PROF=ops: the op-coded launchers add one profile entry per op (next to the launcher's own).
pub fn op_prof(name: &'static str, t: Option<std::time::Instant>) {
    if let Some(t) = t {
        crate::prof_add(name, t.elapsed().as_secs_f64());
    }
}
pub fn op_timer() -> Option<std::time::Instant> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    ON.get_or_init(|| std::env::var("PC_HOST_PROF").is_ok_and(|v| v == "ops")).then(std::time::Instant::now)
}

#[inline(always)]
pub fn mulm(a: u32, b: u32, p: u32, mu: u64) -> u32 {
    let x = a as u64 * b as u64;
    let q = ((x as u128 * mu as u128) >> 64) as u64;
    let mut r = x.wrapping_sub(q.wrapping_mul(p as u64));
    if r >= p as u64 {
        r -= p as u64;
    }
    if r >= p as u64 {
        r -= p as u64;
    }
    r as u32
}
#[inline(always)]
pub fn addm(a: u32, b: u32, p: u32) -> u32 {
    let s = a.wrapping_add(b);
    if s >= p {
        s.wrapping_sub(p)
    } else {
        s
    }
}
#[inline(always)]
pub fn subm(a: u32, b: u32, p: u32) -> u32 {
    if a >= b {
        a - b
    } else {
        a.wrapping_add(p).wrapping_sub(b)
    }
}
#[inline(always)]
pub fn centered(v: u64, p: u32) -> i64 {
    if v > (p / 2) as u64 {
        v as i64 - p as i64
    } else {
        v as i64
    }
}
/// c mod p for a signed c (C's truncating % then + p); |c| < p without a division.
#[inline(always)]
pub fn smod(c: i64, p: u32) -> u32 {
    let p = p as i64;
    if (0..p).contains(&c) {
        c as u32
    } else if c < 0 && c > -p {
        (c + p) as u32
    } else {
        let r = c % p;
        (if r < 0 { r + p } else { r }) as u32
    }
}
/// Balanced base-256 digits of v (four, the last taking the rest): ((v % 256) + 256) % 256, minus 256 past 127.
#[inline(always)]
pub fn digits4(mut v: i64) -> [i8; 4] {
    let mut d = [0i8; 4];
    for t in d.iter_mut() {
        let q = v as i8;
        *t = q;
        v = (v - q as i64) >> 8;
    }
    d
}

/// The op-coded transforms: `count` polynomials of n words at x, poly k at prime index pidx(k), twiddles
/// tw[prime][n]; the inverse scales by scale[prime] (n^-1). One task per polynomial.
pub fn ntt_polys(x: *mut u32, n: usize, count: usize, tw: *const u32, p: &[u32], scale: Option<&[u32]>, pidx: impl Fn(usize) -> usize + Sync + Send) {
    let tabs: Vec<_> = p.iter().enumerate().map(|(i, &q)| ntt_g::tab(tw.wrapping_add(i * n), n, q)).collect();
    let sc: Vec<(u32, u32)> = scale.map(|s| s.iter().zip(p).map(|(&c, &q)| (c % q, ntt_g::companion(c % q, q))).collect()).unwrap_or_default();
    let xs = x as usize;
    // tasks of about 2^16 words (small rings: many polynomials per task)
    let per = (65536 / n).clamp(1, count);
    par_for(count.div_ceil(per), &(), |_, t| {
        for k in t * per..((t + 1) * per).min(count) {
            let pi = pidx(k);
            let (q, tb) = (p[pi], &tabs[pi]);
            // SAFETY: polynomial k of the launch's buffer (disjoint per task)
            let row = unsafe { std::slice::from_raw_parts_mut((xs as *mut u32).add(k * n), n) };
            if scale.is_none() {
                ntt_g::fwd(row, &tb.w, &tb.ws, q);
            } else {
                ntt_g::inv(row, &tb.w, &tb.ws, q);
                ntt_g::scale(row, sc[pi].0, sc[pi].1, q);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_residues_are_the_devices() {
        for p in [65521u32, 268042241, 2147352577] {
            for c in [0i64, 1, -1, p as i64 - 1, p as i64, -(p as i64), -(p as i64) + 1, 3 * p as i64 + 5, -7 * p as i64 - 3, i32::MIN as i64] {
                let r = c % p as i64;
                assert_eq!(smod(c, p), (if r < 0 { r + p as i64 } else { r }) as u32, "{c} mod {p}");
            }
        }
    }

    #[test]
    fn digits_are_the_devices() {
        for v in [0i64, 1, -1, 127, 128, -128, -129, 255, 1 << 29, -(1 << 29) - 7, 1_073_741_823, -1_073_741_823] {
            let mut w = v;
            let mut want = [0i8; 4];
            for t in &mut want {
                let mut q = ((w % 256) + 256) % 256;
                if q > 127 {
                    q -= 256;
                }
                *t = q as i8;
                w = (w - q) / 256;
            }
            assert_eq!(digits4(v), want, "{v}");
        }
    }
}
