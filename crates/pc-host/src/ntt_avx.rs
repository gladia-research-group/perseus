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
unsafe fn small_stage(x: &mut [u32], w: &[u32], ws: &[u32], m: usize, t: usize, p: __m512i, p2: __m512i, fwd: bool) {
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
    for blk in (0..N).step_by(32) {
        let (x0, x1) = (ld(x.as_ptr().add(blk)), ld(x.as_ptr().add(blk + 16)));
        let a = _mm512_permutex2var_epi32(x0, ilo, x1);
        let b = _mm512_permutex2var_epi32(x0, ihi, x1);
        let i0 = m + blk / (2 * t);
        // the block's twiddles (groups of them, 16 words loaded; only the first `groups` are used)
        let wv = if groups >= 16 { ld(w.as_ptr().add(i0)) } else { _mm512_maskz_loadu_epi32((1u16 << groups).wrapping_sub(1), w.as_ptr().add(i0) as *const _) };
        let wsv = if groups >= 16 { ld(ws.as_ptr().add(i0)) } else { _mm512_maskz_loadu_epi32((1u16 << groups).wrapping_sub(1), ws.as_ptr().add(i0) as *const _) };
        let (tw, tws) = (_mm512_permutexvar_epi32(ig, wv), _mm512_permutexvar_epi32(ig, wsv));
        let (a, b) = if fwd { ct(a, b, tw, tws, p, p2) } else { gs(a, b, tw, tws, p, p2) };
        st(x.as_mut_ptr().add(blk), _mm512_permutex2var_epi32(a, ib0, b));
        st(x.as_mut_ptr().add(blk + 16), _mm512_permutex2var_epi32(a, ib1, b));
    }
}

#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn big_stage(x: &mut [u32], w: &[u32], ws: &[u32], m: usize, t: usize, p: __m512i, p2: __m512i, fwd: bool) {
    for i in 0..m {
        let (tw, tws) = (_mm512_set1_epi32(w[m + i] as i32), _mm512_set1_epi32(ws[m + i] as i32));
        let base = x.as_mut_ptr().add(2 * i * t);
        for j in (0..t).step_by(16) {
            let (pa, pb) = (base.add(j), base.add(j + t));
            let (a, b) = if fwd { ct(ld(pa), ld(pb), tw, tws, p, p2) } else { gs(ld(pa), ld(pb), tw, tws, p, p2) };
            st(pa, a);
            st(pb, b);
        }
    }
}

/// Forward lazy NTT of one row (canonical output); `mid` receives the words after the stages m = 1..128.
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
pub unsafe fn fwd(x: &mut [u32], w: &[u32], ws: &[u32], p: u32, mut mid: Option<&mut [u32]>) {
    let (pv, p2) = (_mm512_set1_epi32(p as i32), _mm512_set1_epi32(2 * p as i32));
    let mut m = 1;
    let mut stage = 0;
    while m < N {
        let t = N / (2 * m);
        if t >= 16 {
            big_stage(x, w, ws, m, t, pv, p2, true);
        } else {
            small_stage(x, w, ws, m, t, pv, p2, true);
        }
        stage += 1;
        if stage == 8 {
            if let Some(o) = mid.take() {
                o.copy_from_slice(x);
            }
        }
        m *= 2;
    }
    for i in (0..N).step_by(16) {
        let a = ld(x.as_ptr().add(i));
        st(x.as_mut_ptr().add(i), red(red(a, p2), pv));
    }
}

/// Inverse lazy NTT of one row with the N^-1 scaling (canonical output).
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
pub unsafe fn inv(x: &mut [u32], w: &[u32], ws: &[u32], p: u32, ni: u32, nis: u32) {
    let (pv, p2) = (_mm512_set1_epi32(p as i32), _mm512_set1_epi32(2 * p as i32));
    let mut m = N / 2;
    while m >= 1 {
        let t = N / (2 * m);
        if t >= 16 {
            big_stage(x, w, ws, m, t, pv, p2, false);
        } else {
            small_stage(x, w, ws, m, t, pv, p2, false);
        }
        m /= 2;
    }
    let (n1, n2) = (_mm512_set1_epi32(ni as i32), _mm512_set1_epi32(nis as i32));
    for i in (0..N).step_by(16) {
        let a = ld(x.as_ptr().add(i));
        st(x.as_mut_ptr().add(i), red(shoup_lz(a, n1, n2, pv), pv));
    }
}
