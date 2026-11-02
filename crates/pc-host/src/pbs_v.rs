// The blind rotation's per-word MACs (tfhe.rs `bundle_s` / `word_s`) on AVX-512. A zmm of key words holds four words'
// uint4 (x | y << 32, z | w << 32 per word): as u64 lanes its low halves are rows 0 and 1 of part 0 and its high halves
// those of part 1, so vpmuludq against the per-word factor (duplicated into both lanes of the word) accumulates both
// rows at once. The product pairs the digits of rows 0 and 1 the same way and sums the lane pairs; Montgomery
// reductions on u64 lanes, results compressed back to words. Exact arithmetic: the scalar forms' canonical words.
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::tfhe::{bundle_s, word_s, Rot, Src};

const TH: usize = 1024;

fn wide() -> bool {
    crate::simd::wide()
}

/// The bundle words of one launch slot of the key bundle [e][nk][thread][4] at word k.
fn bundle_at(r: &Rot, kb: &[u32], k: usize) -> [[u32; 4]; 2] {
    let (e, th) = (k / TH, k % TH);
    let mut g = [[0u32; 4]; 2];
    for (s, gs) in g.iter_mut().enumerate().take(r.nk()) {
        gs.copy_from_slice(&kb[((e * r.nk() + s) * TH + th) * 4..][..4]);
    }
    g
}

/// o = the step's products at every word (2 n words; see `word_s`).
pub fn words(r: &Rot, src: Src, em: &[u32], dig: &[u32], d1y: &[u32], o: &mut [u32]) {
    #[cfg(target_arch = "x86_64")]
    if wide() {
        // SAFETY: CPU features checked; the buffers hold the step's rows
        return unsafe { words_v(r, src, em, dig, d1y, o) };
    }
    for k in 0..2 * r.n {
        let g = match src {
            Src::Key(key) => bundle_s(r, key, em, k),
            Src::Bundle(kb) => bundle_at(r, kb, k),
        };
        word_s(r, &g, k, dig, d1y, o);
    }
}

/// out [e][nk][thread][4] = the key bundle of every word (PBS_KEYBUNDLE).
pub fn bundles(r: &Rot, key: &[u32], em: &[u32], out: &mut [u32]) {
    #[cfg(target_arch = "x86_64")]
    if wide() {
        // SAFETY: CPU features checked
        return unsafe { bundles_v(r, key, em, out) };
    }
    let nk = r.nk();
    for k in 0..2 * r.n {
        let g = bundle_s(r, key, em, k);
        let (e, th) = (k / TH, k % TH);
        for s in 0..nk {
            out[((e * nk + s) * TH + th) * 4..][..4].copy_from_slice(&g[s]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn redc_v(x: __m512i, qv: __m512i, miv: __m512i) -> __m512i {
    let t = _mm512_mul_epu32(_mm512_mul_epu32(x, miv), qv);
    let r = _mm512_srli_epi64::<32>(_mm512_add_epi64(x, t));
    _mm512_min_epu64(r, _mm512_sub_epi64(r, qv))
}
/// u64 lanes (a[0], b[0], a[1], b[1], .., a[3], b[3]).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn pair(a: *const u32, b: *const u32) -> __m512i {
    let (x, y) = (_mm_loadu_si128(a as *const _), _mm_loadu_si128(b as *const _));
    let v = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(_mm_unpacklo_epi32(x, y)), _mm_unpackhi_epi32(x, y));
    _mm512_cvtepu32_epi64(v)
}
/// Lane pairs summed (both lanes of a pair hold the sum).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn psum(v: __m512i) -> __m512i {
    _mm512_add_epi64(v, _mm512_shuffle_epi32::<0x4E>(v))
}

/// Index vectors placing word 4 s + l / 2 of a 16-word vector into both halves of u64 lane l (s < 4).
#[cfg(target_arch = "x86_64")]
fn exp_idx() -> [[i32; 16]; 4] {
    std::array::from_fn(|s| std::array::from_fn(|i| (4 * s + i / 4) as i32))
}

/// The factors of the 16 words at k0 for each mask m: pm1[(E[k] em_m) mod 2 n] (nu = 2: em_m / 2).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512vl,avx512bw")]
unsafe fn factors(r: &Rot, em: &[u32], k0: usize, f: &mut [__m512i; 31]) {
    let (n, p) = (r.n, k0 / r.n);
    let ev = _mm512_loadu_si512(r.ex.as_ptr().add(k0) as *const _);
    let mask = _mm512_set1_epi32((2 * n - 1) as i32);
    let base = r.pm1.as_ptr().add(p * 2 * n);
    for m in 0..r.km {
        let x = if r.nu == 1 { em[m] } else { em[m] >> 1 };
        if em[m] == 0 {
            continue;
        }
        let idx = _mm512_and_si512(_mm512_mullo_epi32(ev, _mm512_set1_epi32(x as i32)), mask);
        f[m] = _mm512_i32gather_epi32::<4>(idx, base as *const _);
    }
}

/// The bundle of the 4 words at k (k = k0 + 4 s) from the key: [slot][part] u64 lanes (row 0, row 1 per word).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn bundle4(r: &Rot, key: &[u32], em: &[u32], f: &[__m512i; 31], xi: __m512i, k: usize, qv: __m512i, miv: __m512i) -> [[__m512i; 2]; 2] {
    let (p, e, th) = (k / r.n, k / TH, k % TH);
    let z = _mm512_setzero_si512();
    let mut acc = [[z; 2]; 2];
    for m in 0..r.km {
        let x = em[m];
        if x == 0 {
            continue;
        }
        let fv = _mm512_permutexvar_epi32(xi, f[m]);
        if r.nu == 1 {
            for (h, ah) in acc.iter_mut().enumerate().take(r.lv) {
                let kz = _mm512_loadu_si512(key.as_ptr().add((((m * 4 + e) * r.lv + h) * TH + th) * 4) as *const _);
                ah[0] = _mm512_add_epi64(ah[0], _mm512_mul_epu32(kz, fv));
                ah[1] = _mm512_add_epi64(ah[1], _mm512_mul_epu32(_mm512_srli_epi64::<32>(kz), fv));
            }
        } else {
            let kz = _mm512_loadu_si512(key.as_ptr().add(((m * 4 + e) * TH + th) * 4) as *const _);
            let kh = _mm512_srli_epi64::<32>(kz);
            let (fa, fb) = if x & 1 != 0 {
                let q32 = _mm512_set1_epi32(r.q[p] as i32);
                let s = _mm512_add_epi32(fv, _mm512_set1_epi32(r.r1[p] as i32));
                (_mm512_set1_epi64((r.q[p] - r.r1[p]) as i64), Some(_mm512_min_epu32(s, _mm512_sub_epi32(s, q32))))
            } else {
                (fv, None)
            };
            acc[0][0] = _mm512_add_epi64(acc[0][0], _mm512_mul_epu32(kz, fa));
            acc[0][1] = _mm512_add_epi64(acc[0][1], _mm512_mul_epu32(kh, fa));
            if let Some(fb) = fb {
                acc[1][0] = _mm512_add_epi64(acc[1][0], _mm512_mul_epu32(kz, fb));
                acc[1][1] = _mm512_add_epi64(acc[1][1], _mm512_mul_epu32(kh, fb));
            }
        }
    }
    for a in acc.iter_mut() {
        for v in a.iter_mut() {
            *v = redc_v(*v, qv, miv);
        }
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512vl,avx512bw")]
unsafe fn words_v(r: &Rot, src: Src, em: &[u32], dig: &[u32], d1y: &[u32], o: &mut [u32]) {
    let n = r.n;
    let xi = exp_idx().map(|v| _mm512_loadu_si512(v.as_ptr() as *const _));
    let lo = _mm512_set1_epi64(0xffff_ffff);
    let rows = 2 * r.lv;
    let mut f = [_mm512_setzero_si512(); 31];
    for k0 in (0..2 * n).step_by(16) {
        let p = k0 / n;
        let (qv, miv) = (_mm512_set1_epi64(r.q[p] as i64), _mm512_set1_epi64(r.mi[p] as i64));
        if let Src::Key(_) = src {
            factors(r, em, k0, &mut f);
        }
        for (s, &x) in xi.iter().enumerate() {
            let k = k0 + 4 * s;
            let (e, th, j) = (k / TH, k % TH, k % n);
            let g = match src {
                Src::Key(key) => bundle4(r, key, em, &f, x, k, qv, miv),
                Src::Bundle(kb) => std::array::from_fn(|s2| {
                    if s2 >= r.nk() {
                        return [_mm512_setzero_si512(); 2];
                    }
                    let kz = _mm512_loadu_si512(kb.as_ptr().add(((e * r.nk() + s2) * TH + th) * 4) as *const _);
                    [_mm512_and_si512(kz, lo), _mm512_srli_epi64::<32>(kz)]
                }),
            };
            let d = |comp: usize, row: usize| dig.as_ptr().add(((comp * rows + row) * 2 + p) * n + j);
            for pt in 0..2 {
                if r.nu == 1 {
                    let mut acc = _mm512_mul_epu32(pair(d(0, 0), d(0, 1)), g[0][pt]);
                    if r.lv == 2 {
                        acc = _mm512_add_epi64(acc, _mm512_mul_epu32(pair(d(0, 2), d(0, 3)), g[1][pt]));
                    }
                    let v = redc_v(psum(acc), qv, miv);
                    _mm512_mask_compressstoreu_epi32(o.as_mut_ptr().add((pt * 2 + p) * n + j) as *mut _, 0x1111, v);
                } else {
                    let (d0, d1) = (pair(d(0, 0), d(0, 1)), pair(d(1, 0), d(1, 1)));
                    let dy = pair(d1y.as_ptr().add(p * n + j), d1y.as_ptr().add((2 + p) * n + j));
                    let s0 = _mm512_add_epi64(_mm512_mul_epu32(d0, g[0][pt]), _mm512_mul_epu32(dy, g[1][pt]));
                    let s1 = _mm512_add_epi64(_mm512_mul_epu32(d1, g[0][pt]), _mm512_mul_epu32(d0, g[1][pt]));
                    _mm512_mask_compressstoreu_epi32(o.as_mut_ptr().add((pt * 2 + p) * n + j) as *mut _, 0x1111, redc_v(psum(s0), qv, miv));
                    _mm512_mask_compressstoreu_epi32(o.as_mut_ptr().add((4 + pt * 2 + p) * n + j) as *mut _, 0x1111, redc_v(psum(s1), qv, miv));
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512vl,avx512bw")]
unsafe fn bundles_v(r: &Rot, key: &[u32], em: &[u32], out: &mut [u32]) {
    let n = r.n;
    let nk = r.nk();
    let xi = exp_idx().map(|v| _mm512_loadu_si512(v.as_ptr() as *const _));
    let mut f = [_mm512_setzero_si512(); 31];
    for k0 in (0..2 * n).step_by(16) {
        let p = k0 / n;
        let (qv, miv) = (_mm512_set1_epi64(r.q[p] as i64), _mm512_set1_epi64(r.mi[p] as i64));
        factors(r, em, k0, &mut f);
        for (s, &x) in xi.iter().enumerate() {
            let k = k0 + 4 * s;
            let (e, th) = (k / TH, k % TH);
            let g = bundle4(r, key, em, &f, x, k, qv, miv);
            for (s2, gs) in g.iter().enumerate().take(nk) {
                let v = _mm512_or_si512(gs[0], _mm512_slli_epi64::<32>(gs[1]));
                _mm512_storeu_si512(out.as_mut_ptr().add(((e * nk + s2) * TH + th) * 4) as *mut _, v);
            }
        }
    }
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

    /// The vector rows equal the scalar rows for every (nu, levels, source) the launcher takes.
    #[test]
    fn vector_words_are_the_scalar_words() {
        let n = 2048;
        let (qa, qb) = (67100673u32, 67096577u32);
        let mut rd = rng(11);
        let pm1: Vec<u32> = (0..4 * n).map(|i| rd(if i < 2 * n { qa } else { qb })).collect();
        let ex: Vec<u32> = (0..2 * n).map(|_| rd(2 * n as u32)).collect();
        for (nu, lv, g) in [(1usize, 1usize, 3usize), (1, 2, 2), (2, 1, 4), (1, 1, 5)] {
            let km = (1 << g) - 1;
            let q = [qa, qb];
            let r1 = q.map(|q| ((1u64 << 32) % q as u64) as u32);
            let yk = (0..2 * n).map(|k| rd(q[k / n])).collect();
            let r = Rot { n, nu, lv, km, q, mi: q.map(|q| crate::tfhe::mont_neg_inv(q)), r1, pm1: &pm1, ex: &ex, yk };
            let key: Vec<u32> = (0..km * 8 * lv * n).map(|i| rd(q[(i / 4 % 4096) / n])).collect();
            let em: Vec<u32> = (0..km).map(|m| if m == 2 { 0 } else { rd((2 * nu * n) as u32) }).collect();
            let dig: Vec<u32> = (0..nu * 4 * lv * n).map(|i| rd(q[(i / n) & 1])).collect();
            let d1y: Vec<u32> = (0..4 * n).map(|i| rd(q[(i / n) & 1])).collect();
            let nk = r.nk();
            let (mut kv, mut ks) = (vec![0u32; 2 * n * nk * 4], vec![0u32; 2 * n * nk * 4]);
            bundles(&r, &key, &em, &mut kv);
            for k in 0..2 * n {
                let gg = bundle_s(&r, &key, &em, k);
                for s in 0..nk {
                    ks[((k / TH * nk + s) * TH + k % TH) * 4..][..4].copy_from_slice(&gg[s]);
                }
            }
            assert_eq!(kv, ks, "bundles nu {nu} lv {lv}");
            for src in [Src::Key(&key), Src::Bundle(&kv)] {
                let (mut ov, mut os) = (vec![0u32; nu * 4 * n], vec![0u32; nu * 4 * n]);
                words(&r, src, &em, &dig, &d1y, &mut ov);
                for k in 0..2 * n {
                    let gg = match src {
                        Src::Key(key) => bundle_s(&r, key, &em, k),
                        Src::Bundle(kb) => bundle_at(&r, kb, k),
                    };
                    word_s(&r, &gg, k, &dig, &d1y, &mut os);
                }
                assert_eq!(ov, os, "words nu {nu} lv {lv}");
            }
        }
    }
}
