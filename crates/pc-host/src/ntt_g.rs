// Negacyclic u32 NTT rows of any power-of-two size n >= 32 (the pcmm / ccmm / tfhe launchers' transforms): Cooley-Tukey
// forward and Gentleman-Sande inverse (no n^-1) with twiddle i of stage m at w[m + i] and its Shoup companion. Every
// butterfly takes and gives canonical words (p < 2^31), so any schedule returns the device's words. AVX-512: stages
// with t >= 16 on contiguous vectors, t = 8..1 on 32-word blocks split into butterfly halves by permutation; rows
// longer than BLK words run the stages with 2t <= BLK block by block (in L1). `tab` keeps the companions of the
// launchers' twiddle tables (which carry none), checked against the table's words on every use.
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::sync::{Arc, Mutex};

const BLK: usize = 4096;

#[inline(always)]
pub fn shoup(a: u32, w: u32, ws: u32, p: u32) -> u32 {
    let q = ((a as u64 * ws as u64) >> 32) as u32;
    let r = a.wrapping_mul(w).wrapping_sub(q.wrapping_mul(p));
    if r >= p {
        r - p
    } else {
        r
    }
}
#[inline(always)]
fn addm(a: u32, b: u32, p: u32) -> u32 {
    let s = a + b;
    if s >= p {
        s - p
    } else {
        s
    }
}
#[inline(always)]
fn subm(a: u32, b: u32, p: u32) -> u32 {
    if a >= b {
        a - b
    } else {
        a + p - b
    }
}

/// floor(w 2^32 / p) for w < p.
#[inline(always)]
pub fn companion(w: u32, p: u32) -> u32 {
    (((w as u64) << 32) / p as u64) as u32
}

fn stage_s(x: &mut [u32], w: &[u32], ws: &[u32], p: u32, m: usize, t: usize, lo: usize, hi: usize, fwd: bool) {
    for i in lo / (2 * t)..hi / (2 * t) {
        let (tw, tws) = (w[m + i], ws[m + i]);
        let (a, b) = x[2 * i * t..2 * i * t + 2 * t].split_at_mut(t);
        for (u, v) in a.iter_mut().zip(b.iter_mut()) {
            if fwd {
                let s = shoup(*v, tw, tws, p);
                (*u, *v) = (addm(*u, s, p), subm(*u, s, p));
            } else {
                (*u, *v) = (addm(*u, *v, p), shoup(subm(*u, *v, p), tw, tws, p));
            }
        }
    }
}

/// The stage order: the stages crossing BLK blocks over the whole row, the others block by block.
fn schedule(n: usize, fwd: bool, mut stage: impl FnMut(usize, usize, usize, usize)) {
    let blk = BLK.min(n);
    let cut = n / blk;
    if fwd {
        let mut m = 1;
        while m < cut {
            stage(m, n / (2 * m), 0, n);
            m *= 2;
        }
        for lo in (0..n).step_by(blk) {
            let mut m = cut;
            while m < n {
                stage(m, n / (2 * m), lo, lo + blk);
                m *= 2;
            }
        }
    } else {
        for lo in (0..n).step_by(blk) {
            let mut m = n / 2;
            while m >= cut {
                stage(m, n / (2 * m), lo, lo + blk);
                m /= 2;
            }
        }
        let mut m = cut / 2;
        while m >= 1 {
            stage(m, n / (2 * m), 0, n);
            m /= 2;
        }
    }
}

fn wide() -> bool {
    crate::simd::wide()
}

/// Forward transform of x (n = x.len(), canonical words in and out).
pub fn fwd(x: &mut [u32], w: &[u32], ws: &[u32], p: u32) {
    let n = x.len();
    debug_assert!(n.is_power_of_two() && w.len() >= n && ws.len() >= n);
    #[cfg(target_arch = "x86_64")]
    if wide() && n >= 32 {
        // SAFETY: CPU features checked, n >= 32
        return unsafe { run_v(x, w, ws, p, true) };
    }
    schedule(n, true, |m, t, lo, hi| stage_s(x, w, ws, p, m, t, lo, hi, true));
}

/// Inverse transform of x without the n^-1 scaling.
pub fn inv(x: &mut [u32], w: &[u32], ws: &[u32], p: u32) {
    let n = x.len();
    debug_assert!(n.is_power_of_two() && w.len() >= n && ws.len() >= n);
    #[cfg(target_arch = "x86_64")]
    if wide() && n >= 32 {
        // SAFETY: CPU features checked, n >= 32
        return unsafe { run_v(x, w, ws, p, false) };
    }
    schedule(n, false, |m, t, lo, hi| stage_s(x, w, ws, p, m, t, lo, hi, false));
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn ld(p: *const u32) -> __m512i {
    _mm512_loadu_si512(p as *const _)
}
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn st(p: *mut u32, v: __m512i) {
    _mm512_storeu_si512(p as *mut _, v)
}
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mulhi(a: __m512i, b: __m512i) -> __m512i {
    let even = _mm512_srli_epi64::<32>(_mm512_mul_epu32(a, b));
    let odd = _mm512_mul_epu32(_mm512_srli_epi64::<32>(a), _mm512_srli_epi64::<32>(b));
    _mm512_mask_blend_epi32(0xAAAA, even, odd)
}
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn red(a: __m512i, p: __m512i) -> __m512i {
    _mm512_min_epu32(a, _mm512_sub_epi32(a, p))
}
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn shoup_v(a: __m512i, w: __m512i, ws: __m512i, p: __m512i) -> __m512i {
    red(_mm512_sub_epi32(_mm512_mullo_epi32(a, w), _mm512_mullo_epi32(mulhi(a, ws), p)), p)
}
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn bfly(a: __m512i, b: __m512i, w: __m512i, ws: __m512i, p: __m512i, fwd: bool) -> (__m512i, __m512i) {
    if fwd {
        let v = shoup_v(b, w, ws, p);
        (red(_mm512_add_epi32(a, v), p), red(_mm512_add_epi32(_mm512_sub_epi32(a, v), p), p))
    } else {
        let d = red(_mm512_add_epi32(_mm512_sub_epi32(a, b), p), p);
        (red(_mm512_add_epi32(a, b), p), shoup_v(d, w, ws, p))
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn big_v(x: &mut [u32], w: &[u32], ws: &[u32], p: __m512i, m: usize, t: usize, lo: usize, hi: usize, fwd: bool) {
    for i in lo / (2 * t)..hi / (2 * t) {
        let (tw, tws) = (_mm512_set1_epi32(w[m + i] as i32), _mm512_set1_epi32(ws[m + i] as i32));
        let base = x.as_mut_ptr().add(2 * i * t);
        for j in (0..t).step_by(16) {
            let (pa, pb) = (base.add(j), base.add(j + t));
            let (a, b) = bfly(ld(pa), ld(pb), tw, tws, p, fwd);
            st(pa, a);
            st(pb, b);
        }
    }
}

/// A small stage's permutations over a 32-word block [x0 | x1] (t = 1, 2, 4, 8): lanes l of the low / high halves
/// are words (l / t) 2t + l % t (+ t), lane l's group l / t, and the inverse placement of the 32 words.
struct Small {
    lo: [i32; 16],
    hi: [i32; 16],
    grp: [i32; 16],
    back: [i32; 32],
}
const fn small(t: usize) -> Small {
    let mut s = Small { lo: [0; 16], hi: [0; 16], grp: [0; 16], back: [0; 32] };
    let mut l = 0;
    while l < 16 {
        s.lo[l] = ((l / t) * 2 * t + l % t) as i32;
        s.hi[l] = s.lo[l] + t as i32;
        s.grp[l] = (l / t) as i32;
        l += 1;
    }
    let mut j = 0;
    while j < 32 {
        let (g, k) = (j / (2 * t), j % (2 * t));
        s.back[j] = if k < t { (g * t + k) as i32 } else { (16 + g * t + k - t) as i32 };
        j += 1;
    }
    s
}
static SMALL: [Small; 4] = [small(1), small(2), small(4), small(8)];

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn small_v(x: &mut [u32], w: &[u32], ws: &[u32], p: __m512i, m: usize, t: usize, lo: usize, hi: usize, fwd: bool) {
    let sm = &SMALL[t.trailing_zeros() as usize];
    let (il, ih, ig) = (ld(sm.lo.as_ptr() as *const u32), ld(sm.hi.as_ptr() as *const u32), ld(sm.grp.as_ptr() as *const u32));
    let (b0, b1) = (ld(sm.back.as_ptr() as *const u32), ld(sm.back.as_ptr().add(16) as *const u32));
    let groups = 16 / t;
    let gm = ((1u32 << groups) - 1) as u16;
    for blk in (lo..hi).step_by(32) {
        let (x0, x1) = (ld(x.as_ptr().add(blk)), ld(x.as_ptr().add(blk + 16)));
        let a = _mm512_permutex2var_epi32(x0, il, x1);
        let b = _mm512_permutex2var_epi32(x0, ih, x1);
        let i0 = m + blk / (2 * t);
        let wv = _mm512_maskz_loadu_epi32(gm, w.as_ptr().add(i0) as *const _);
        let wsv = _mm512_maskz_loadu_epi32(gm, ws.as_ptr().add(i0) as *const _);
        let (a, b) = bfly(a, b, _mm512_permutexvar_epi32(ig, wv), _mm512_permutexvar_epi32(ig, wsv), p, fwd);
        st(x.as_mut_ptr().add(blk), _mm512_permutex2var_epi32(a, b0, b));
        st(x.as_mut_ptr().add(blk + 16), _mm512_permutex2var_epi32(a, b1, b));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn run_v(x: &mut [u32], w: &[u32], ws: &[u32], p: u32, fwd: bool) {
    let pv = _mm512_set1_epi32(p as i32);
    schedule(x.len(), fwd, |m, t, lo, hi| {
        if t >= 16 {
            big_v(x, w, ws, pv, m, t, lo, hi, fwd)
        } else {
            small_v(x, w, ws, pv, m, t, lo, hi, fwd)
        }
    });
}

/// x[i] = x[i] c mod p (Shoup, c < p, cs its companion; canonical out).
pub fn scale(x: &mut [u32], c: u32, cs: u32, p: u32) {
    #[cfg(target_arch = "x86_64")]
    if wide() {
        // SAFETY: CPU features checked
        return unsafe { scale_v(x, c, cs, p) };
    }
    for v in x.iter_mut() {
        *v = shoup(*v, c, cs, p);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn scale_v(x: &mut [u32], c: u32, cs: u32, p: u32) {
    let (cv, csv, pv) = (_mm512_set1_epi32(c as i32), _mm512_set1_epi32(cs as i32), _mm512_set1_epi32(p as i32));
    let n = x.len() / 16 * 16;
    for i in (0..n).step_by(16) {
        st(x.as_mut_ptr().add(i), shoup_v(ld(x.as_ptr().add(i)), cv, csv, pv));
    }
    for v in x[n..].iter_mut() {
        *v = shoup(*v, c, cs, p);
    }
}

/// A twiddle table reduced mod p with its Shoup companions.
pub struct Tab {
    pub w: Vec<u32>,
    pub ws: Vec<u32>,
}

type Entry = (usize, u32, Vec<u32>, Arc<Tab>);
static CACHE: Mutex<Vec<Entry>> = Mutex::new(Vec::new());

/// The companions of the n words at `t` (a launch's table row for prime p).
pub fn tab(t: *const u32, n: usize, p: u32) -> Arc<Tab> {
    // SAFETY: a table row of n words (the launch contract)
    let raw = unsafe { std::slice::from_raw_parts(t, n) };
    {
        let c = CACHE.lock().unwrap();
        if let Some(e) = c.iter().find(|e| e.0 == t as usize && e.1 == p && e.2 == raw) {
            return e.3.clone();
        }
    }
    let w: Vec<u32> = raw.iter().map(|&v| v % p).collect();
    let ws = w.iter().map(|&v| companion(v, p)).collect();
    let tb = Arc::new(Tab { w, ws });
    let mut c = CACHE.lock().unwrap();
    if c.len() >= 64 {
        c.remove(0);
    }
    c.push((t as usize, p, raw.to_vec(), tb.clone()));
    tb
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: u64) -> impl FnMut(u32) -> u32 {
        let mut st = seed;
        move |m| {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((st >> 33) as u32) % m
        }
    }

    #[test]
    fn vector_rows_are_the_scalar_rows() {
        for (n, p) in [(32usize, 2013265921u32), (64, 2147352577), (2048, 67100673), (8192, 1073479681), (65536, 2147352577)] {
            let mut r = rng(n as u64);
            let w: Vec<u32> = (0..n).map(|_| r(p)).collect();
            let ws: Vec<u32> = w.iter().map(|&v| companion(v, p)).collect();
            let x: Vec<u32> = (0..n).map(|_| r(p)).collect();
            let mut a = x.clone();
            schedule(n, true, |m, t, lo, hi| stage_s(&mut a, &w, &ws, p, m, t, lo, hi, true));
            let mut b = x.clone();
            fwd(&mut b, &w, &ws, p);
            assert_eq!(a, b, "fwd n {n}");
            // the plain stage order (m = 1 .. n/2 over the whole row) is the same transform
            let mut c = x.clone();
            let mut m = 1;
            while m < n {
                stage_s(&mut c, &w, &ws, p, m, n / (2 * m), 0, n, true);
                m *= 2;
            }
            assert_eq!(a, c, "fwd order n {n}");
            schedule(n, false, |m, t, lo, hi| stage_s(&mut a, &w, &ws, p, m, t, lo, hi, false));
            inv(&mut b, &w, &ws, p);
            assert_eq!(a, b, "inv n {n}");
        }
    }

    #[test]
    fn transform_round_trip() {
        // psi a primitive 2n-th root mod p: forward then inverse is n times the identity
        let (n, p) = (256usize, 7681u32);
        let pw = |b: u64, e: u64| (0..e).fold(1u64, |a, _| a * b % p as u64);
        let g = (2..p as u64).find(|&g| pw(g, (p as u64 - 1) / 2) != 1 && pw(g, (p as u64 - 1) / 3) != 1 && pw(g, (p as u64 - 1) / 5) != 1).unwrap();
        let psi = pw(g, (p as u64 - 1) / (2 * n as u64));
        let ipsi = pw(psi, 2 * n as u64 - 1);
        let brev = |i: usize| i.reverse_bits() >> (usize::BITS - n.trailing_zeros());
        let w: Vec<u32> = (0..n).map(|i| pw(psi, brev(i) as u64) as u32).collect();
        let iw: Vec<u32> = (0..n).map(|i| pw(ipsi, brev(i) as u64) as u32).collect();
        let (ws, iws): (Vec<u32>, Vec<u32>) = (w.iter().map(|&v| companion(v, p)).collect(), iw.iter().map(|&v| companion(v, p)).collect());
        let mut r = rng(7);
        let x: Vec<u32> = (0..n).map(|_| r(p)).collect();
        let mut y = x.clone();
        fwd(&mut y, &w, &ws, p);
        inv(&mut y, &iw, &iws, p);
        for (a, b) in x.iter().zip(&y) {
            assert_eq!(*a as u64 * n as u64 % p as u64, *b as u64);
        }
    }
}
