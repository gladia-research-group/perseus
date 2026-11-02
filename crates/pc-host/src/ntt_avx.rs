// The lazy (Harvey) u32 NTT rows of ntt.rs on 16 AVX-512 lanes, butterfly for butterfly the same words: CT takes and
// gives [0, 4p), GS [0, 2p), Shoup products without their last correction. Stages with t >= 16 run on contiguous
// vectors with one twiddle per group; t = 8, 4, 2, 1 de-interleave a 32-word block into its low and high butterfly
// halves (vpermt2d), take the per-group twiddles by permutation, and interleave back. a >= c ? a - c : a is
// min(a, a - c) on unsigned lanes.
#![cfg(target_arch = "x86_64")]
use pc_kernels::args::N;
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn mulhi(a: __m512i, b: __m512i) -> __m512i {
    let even = _mm512_srli_epi64::<32>(_mm512_mul_epu32(a, b));
    let odd = _mm512_mul_epu32(_mm512_srli_epi64::<32>(a), _mm512_srli_epi64::<32>(b));
    _mm512_mask_blend_epi32(0xAAAA, even, odd)
}
#[inline(always)]
unsafe fn shoup_lz(b: __m512i, w: __m512i, ws: __m512i, p: __m512i) -> __m512i {
    _mm512_sub_epi32(_mm512_mullo_epi32(b, w), _mm512_mullo_epi32(mulhi(b, ws), p))
}
#[inline(always)]
unsafe fn red(a: __m512i, c: __m512i) -> __m512i {
    _mm512_min_epu32(a, _mm512_sub_epi32(a, c))
}
#[inline(always)]
unsafe fn ct(a: __m512i, b: __m512i, w: __m512i, ws: __m512i, p: __m512i, p2: __m512i) -> (__m512i, __m512i) {
    let u = red(a, p2);
    let v = shoup_lz(b, w, ws, p);
    (_mm512_add_epi32(u, v), _mm512_add_epi32(_mm512_sub_epi32(u, v), p2))
}
#[inline(always)]
unsafe fn gs(a: __m512i, b: __m512i, w: __m512i, ws: __m512i, p: __m512i, p2: __m512i) -> (__m512i, __m512i) {
    let s = red(_mm512_add_epi32(a, b), p2);
    let d = shoup_lz(_mm512_add_epi32(_mm512_sub_epi32(a, b), p2), w, ws, p);
    (s, d)
}
#[inline(always)]
unsafe fn ld(p: *const u32) -> __m512i {
    _mm512_loadu_si512(p as *const _)
}
#[inline(always)]
unsafe fn st(p: *mut u32, v: __m512i) {
    _mm512_storeu_si512(p as *mut _, v)
}

/// Index vectors of one small-t stage over a 32-word block [x0 | x1]: lo / hi = the words of (idx mod 2t) < t / >= t,
/// in group order; g = the group (within the block) of each lo lane.
fn small_t(t: usize) -> ([i32; 16], [i32; 16], [i32; 16]) {
    let (mut lo, mut hi, mut g) = ([0i32; 16], [0i32; 16], [0i32; 16]);
    for l in 0..16 {
        let (grp, k) = (l / t, l % t);
        lo[l] = (grp * 2 * t + k) as i32;
        hi[l] = (grp * 2 * t + t + k) as i32;
        g[l] = grp as i32;
    }
    (lo, hi, g)
}

#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn small_stage(x: &mut [u32], w: &[u32], ws: &[u32], m: usize, t: usize, p: __m512i, p2: __m512i, fwd: bool, lo_: usize, hi_: usize, base: usize) {
    let (lo, hi, g) = small_t(t);
    let (ilo, ihi, ig) = (ld(lo.as_ptr() as *const u32), ld(hi.as_ptr() as *const u32), ld(g.as_ptr() as *const u32));
    // inverse placement: word j of the block from lo (j mod 2t < t) or hi, at lane (j / 2t) t + j mod t (+16 for hi)
    let mut back = [0i32; 32];
    for j in 0..32 {
        let (grp, k) = (j / (2 * t), j % (2 * t));
        back[j] = if k < t { (grp * t + k) as i32 } else { (16 + grp * t + k - t) as i32 };
    }
    let (ib0, ib1) = (ld(back.as_ptr() as *const u32), ld(back.as_ptr().add(16) as *const u32));
    let groups = 32 / (2 * t);
    for blk in (lo_..hi_).step_by(32) {
        let (x0, x1) = (ld(x.as_ptr().add(blk - base)), ld(x.as_ptr().add(blk - base + 16)));
        let a = _mm512_permutex2var_epi32(x0, ilo, x1);
        let b = _mm512_permutex2var_epi32(x0, ihi, x1);
        let i0 = m + blk / (2 * t);
        // the block's twiddles (groups of them, 16 words loaded; only the first `groups` are used)
        let wv = if groups >= 16 { ld(w.as_ptr().add(i0)) } else { _mm512_maskz_loadu_epi32((1u16 << groups).wrapping_sub(1), w.as_ptr().add(i0) as *const _) };
        let wsv = if groups >= 16 { ld(ws.as_ptr().add(i0)) } else { _mm512_maskz_loadu_epi32((1u16 << groups).wrapping_sub(1), ws.as_ptr().add(i0) as *const _) };
        let (tw, tws) = (_mm512_permutexvar_epi32(ig, wv), _mm512_permutexvar_epi32(ig, wsv));
        let (a, b) = if fwd { ct(a, b, tw, tws, p, p2) } else { gs(a, b, tw, tws, p, p2) };
        st(x.as_mut_ptr().add(blk - base), _mm512_permutex2var_epi32(a, ib0, b));
        st(x.as_mut_ptr().add(blk - base + 16), _mm512_permutex2var_epi32(a, ib1, b));
    }
}

#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn big_stage(x: &mut [u32], w: &[u32], ws: &[u32], m: usize, t: usize, p: __m512i, p2: __m512i, fwd: bool, lo_: usize, hi_: usize, base: usize) {
    for i in lo_ / (2 * t)..hi_ / (2 * t) {
        let (tw, tws) = (_mm512_set1_epi32(w[m + i] as i32), _mm512_set1_epi32(ws[m + i] as i32));
        let base = x.as_mut_ptr().add(2 * i * t - base);
        for j in (0..t).step_by(16) {
            let (pa, pb) = (base.add(j), base.add(j + t));
            let (a, b) = if fwd { ct(ld(pa), ld(pb), tw, tws, p, p2) } else { gs(ld(pa), ld(pb), tw, tws, p, p2) };
            st(pa, a);
            st(pb, b);
        }
    }
}

/// Block size of the second pass: the stages with 2t <= BLK run block by block in L1.
const BLK: usize = 4096;
/// Stages of the first pass (t = N/2 .. BLK): log2(N / BLK).
const S1: usize = 4;

/// The first S1 stages on 16 columns at once: element c + BLK k (k < 16) of column c only meets the same column's
/// elements in these stages, so a strip of 16 columns runs them in registers (one read and one write of the row).
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn cols_pass(x: &mut [u32], w: &[u32], ws: &[u32], p: __m512i, p2: __m512i, fwd: bool, scale: Option<(__m512i, __m512i)>, c0: usize, c1: usize) {
    // twiddles of stage s (m = 2^s): group of k is k >> (S1 - s)... broadcast once per call
    let mut tw = [_mm512_setzero_si512(); 16];
    let mut tws = [_mm512_setzero_si512(); 16];
    for m in [1usize, 2, 4, 8] {
        for i in 0..m {
            tw[m + i] = _mm512_set1_epi32(w[m + i] as i32);
            tws[m + i] = _mm512_set1_epi32(ws[m + i] as i32);
        }
    }
    for c in (c0..c1).step_by(16) {
        let mut v = [_mm512_setzero_si512(); 16];
        for k in 0..16 {
            v[k] = ld(x.as_ptr().add(c + BLK * k));
        }
        let stage = |m: usize, v: &mut [__m512i; 16]| {
            // stage m: t = N / 2m, i.e. 8 / m columns apart in k; group i = k / (16 / m)
            let h = 8 / m;
            for k in 0..16 {
                if k & h == 0 {
                    let i = k / (2 * h);
                    let (a, b) = if fwd { ct(v[k], v[k + h], tw[m + i], tws[m + i], p, p2) } else { gs(v[k], v[k + h], tw[m + i], tws[m + i], p, p2) };
                    v[k] = a;
                    v[k + h] = b;
                }
            }
        };
        if fwd {
            for m in [1usize, 2, 4, 8] {
                stage(m, &mut v);
            }
        } else {
            for m in [8usize, 4, 2, 1] {
                stage(m, &mut v);
            }
        }
        for k in 0..16 {
            let o = match scale {
                Some((n1, n2)) => red(shoup_lz(v[k], n1, n2, p), p),
                None => v[k],
            };
            st(x.as_mut_ptr().add(c + BLK * k), o);
        }
    }
}

/// Stages m0 .. N/2 (forward: m doubling) or N/2 .. m0 (inverse) of one BLK block at `lo`.
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn block_pass(x: &mut [u32], w: &[u32], ws: &[u32], p: __m512i, p2: __m512i, fwd: bool, lo: usize, mut mid: Option<&mut [u32]>) {
    let ms: Vec<usize> = (S1..16).map(|s| 1usize << s).collect();
    let order: Vec<usize> = if fwd { ms } else { ms.into_iter().rev().collect() };
    for (n, &m) in order.iter().enumerate() {
        let t = N / (2 * m);
        if t >= 16 {
            big_stage(x, w, ws, m, t, p, p2, fwd, lo, lo + BLK, 0);
        } else {
            small_stage(x, w, ws, m, t, p, p2, fwd, lo, lo + BLK, 0);
        }
        // the forward state after the stages m = 1..128 (8 stages: S1 of the first pass, then 4 here)
        if fwd && S1 + n + 1 == 8 {
            if let Some(o) = mid.take() {
                o[lo..lo + BLK].copy_from_slice(&x[lo..lo + BLK]);
            }
        }
    }
}

/// Forward lazy NTT of one row (canonical output); `mid` receives the words after the stages m = 1..128.
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
pub unsafe fn fwd(x: &mut [u32], w: &[u32], ws: &[u32], p: u32, mut mid: Option<&mut [u32]>) {
    let (pv, p2) = (_mm512_set1_epi32(p as i32), _mm512_set1_epi32(2 * p as i32));
    cols_pass(x, w, ws, pv, p2, true, None, 0, BLK);
    for lo in (0..N).step_by(BLK) {
        block_pass(x, w, ws, pv, p2, true, lo, mid.as_deref_mut());
        for i in (lo..lo + BLK).step_by(16) {
            let a = ld(x.as_ptr().add(i));
            st(x.as_mut_ptr().add(i), red(red(a, p2), pv));
        }
    }
}

/// Inverse lazy NTT of one row with the N^-1 scaling (canonical output).
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
pub unsafe fn inv(x: &mut [u32], w: &[u32], ws: &[u32], p: u32, ni: u32, nis: u32) {
    let (pv, p2) = (_mm512_set1_epi32(p as i32), _mm512_set1_epi32(2 * p as i32));
    for lo in (0..N).step_by(BLK) {
        block_pass(x, w, ws, pv, p2, false, lo, None);
    }
    cols_pass(x, w, ws, pv, p2, false, Some((_mm512_set1_epi32(ni as i32), _mm512_set1_epi32(nis as i32))), 0, BLK);
}

/// Columns of the first pass handled by one strip task.
pub const STRIP: usize = 256;
/// Words of a second-pass block.
pub const BLOCK: usize = BLK;
/// Second-pass stages before the forward column-pass state is complete (the stages m = 1..128 end there).
pub const MID: usize = 8 - S1;

/// First-pass stages on columns [c0, c0 + STRIP) of the row (forward; or the inverse with the N^-1 scaling).
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
pub unsafe fn strip(x: &mut [u32], w: &[u32], ws: &[u32], p: u32, fwd: bool, ninv: Option<(u32, u32)>, c0: usize) {
    let (pv, p2) = (_mm512_set1_epi32(p as i32), _mm512_set1_epi32(2 * p as i32));
    let sc = ninv.map(|(n1, n2)| (_mm512_set1_epi32(n1 as i32), _mm512_set1_epi32(n2 as i32)));
    cols_pass(x, w, ws, pv, p2, fwd, sc, c0, c0 + STRIP);
}

/// Second-pass stages [from, to) (in transform order) of the block `xb` holding the row's words [lo, lo + BLOCK);
/// `canon`: the forward transform's final canonicalization after them.
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
pub unsafe fn block(xb: &mut [u32], w: &[u32], ws: &[u32], p: u32, fwd: bool, lo: usize, from: usize, to: usize, canon: bool) {
    let (pv, p2) = (_mm512_set1_epi32(p as i32), _mm512_set1_epi32(2 * p as i32));
    let ms: Vec<usize> = (S1..16).map(|s| 1usize << s).collect();
    let order: Vec<usize> = if fwd { ms } else { ms.into_iter().rev().collect() };
    for &m in &order[from..to] {
        let t = N / (2 * m);
        if t >= 16 {
            big_stage(xb, w, ws, m, t, pv, p2, fwd, lo, lo + BLK, lo);
        } else {
            small_stage(xb, w, ws, m, t, pv, p2, fwd, lo, lo + BLK, lo);
        }
    }
    if canon {
        for i in (0..BLK).step_by(16) {
            let a = ld(xb.as_ptr().add(i));
            st(xb.as_mut_ptr().add(i), red(red(a, p2), pv));
        }
    }
}

/// Second-pass stages of the transform.
pub const BLOCK_STAGES: usize = 16 - S1;
