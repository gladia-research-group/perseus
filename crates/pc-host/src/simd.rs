// AVX-512 row primitives of the u32 instantiations (scalar fallbacks keep the same results):
//  - mac: acc[i] += x[i] w[i] (u32 x u32 -> u64 lanes, vpmuludq);
//  - reduce: acc[i] mod p for 64-bit accumulators, the quotient from the fp64 mantissa (q = floor(acc * (1/p)) is
//    within 1 of the true quotient for 2^12 <= p < 2^31, any 64-bit acc), the remainder in 64-bit integers, one
//    correction each way.
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn wide() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *W.get_or_init(|| std::env::var_os("PC_HOST_SCALAR").is_none_or(|v| !v.to_string_lossy().contains("SIMD")) && {
            std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512dq") && std::is_x86_feature_detected!("avx512vl")
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// acc[i] += x[i] * w[i].
#[inline]
pub fn mac(acc: &mut [u64], x: &[u32], w: &[u32]) {
    #[cfg(target_arch = "x86_64")]
    if wide() {
        // SAFETY: CPU features checked
        return unsafe { mac_v(acc, x, w) };
    }
    for i in 0..acc.len() {
        acc[i] += x[i] as u64 * w[i] as u64;
    }
}

/// acc0[i] += x[i] * w0[i]; acc1[i] += x[i] * w1[i] (the key-switch pair: one load of x).
#[inline]
pub fn mac2(acc0: &mut [u64], acc1: &mut [u64], x: &[u32], w0: &[u32], w1: &[u32]) {
    #[cfg(target_arch = "x86_64")]
    if wide() {
        // SAFETY: CPU features checked
        return unsafe { mac2_v(acc0, acc1, x, w0, w1) };
    }
    for i in 0..acc0.len() {
        acc0[i] += x[i] as u64 * w0[i] as u64;
        acc1[i] += x[i] as u64 * w1[i] as u64;
    }
}

/// out[i] = acc[i] mod p.
#[inline]
pub fn reduce(acc: &[u64], p: u32, out: &mut [u32]) {
    // the double's quotient is within 1 while acc 2^-53 / p < 1/2: every 64-bit acc once p >= 2^12
    #[cfg(target_arch = "x86_64")]
    if wide() && p >= 1 << 12 {
        // SAFETY: CPU features checked
        return unsafe { reduce_v(acc, p, out) };
    }
    for i in 0..acc.len() {
        out[i] = (acc[i] % p as u64) as u32;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn mac_v(acc: &mut [u64], x: &[u32], w: &[u32]) {
    let n = acc.len();
    let mut i = 0;
    while i + 8 <= n {
        let xv = _mm512_cvtepu32_epi64(_mm256_loadu_si256(x.as_ptr().add(i) as *const _));
        let wv = _mm512_cvtepu32_epi64(_mm256_loadu_si256(w.as_ptr().add(i) as *const _));
        let a = _mm512_loadu_si512(acc.as_ptr().add(i) as *const _);
        _mm512_storeu_si512(acc.as_mut_ptr().add(i) as *mut _, _mm512_add_epi64(a, _mm512_mul_epu32(xv, wv)));
        i += 8;
    }
    for j in i..n {
        acc[j] += x[j] as u64 * w[j] as u64;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn mac2_v(acc0: &mut [u64], acc1: &mut [u64], x: &[u32], w0: &[u32], w1: &[u32]) {
    let n = acc0.len();
    let mut i = 0;
    while i + 8 <= n {
        let xv = _mm512_cvtepu32_epi64(_mm256_loadu_si256(x.as_ptr().add(i) as *const _));
        let b = _mm512_cvtepu32_epi64(_mm256_loadu_si256(w0.as_ptr().add(i) as *const _));
        let a = _mm512_cvtepu32_epi64(_mm256_loadu_si256(w1.as_ptr().add(i) as *const _));
        let s0 = _mm512_loadu_si512(acc0.as_ptr().add(i) as *const _);
        let s1 = _mm512_loadu_si512(acc1.as_ptr().add(i) as *const _);
        _mm512_storeu_si512(acc0.as_mut_ptr().add(i) as *mut _, _mm512_add_epi64(s0, _mm512_mul_epu32(xv, b)));
        _mm512_storeu_si512(acc1.as_mut_ptr().add(i) as *mut _, _mm512_add_epi64(s1, _mm512_mul_epu32(xv, a)));
        i += 8;
    }
    for j in i..n {
        acc0[j] += x[j] as u64 * w0[j] as u64;
        acc1[j] += x[j] as u64 * w1[j] as u64;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn reduce_v(acc: &[u64], p: u32, out: &mut [u32]) {
    let n = acc.len();
    let pinv = _mm512_set1_pd(1.0 / p as f64);
    let pv = _mm512_set1_epi64(p as i64);
    let zero = _mm512_setzero_si512();
    let mut i = 0;
    while i + 8 <= n {
        let a = _mm512_loadu_si512(acc.as_ptr().add(i) as *const _);
        let q = _mm512_cvttpd_epu64(_mm512_mul_pd(_mm512_cvtepu64_pd(a), pinv));
        let mut r = _mm512_sub_epi64(a, _mm512_mullo_epi64(q, pv));
        // r in [-p, 2p): one correction each way
        let neg = _mm512_cmplt_epi64_mask(r, zero);
        r = _mm512_mask_add_epi64(r, neg, r, pv);
        let ge = _mm512_cmpge_epi64_mask(r, pv);
        r = _mm512_mask_sub_epi64(r, ge, r, pv);
        _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut _, _mm512_cvtepi64_epi32(r));
        i += 8;
    }
    for j in i..n {
        out[j] = (acc[j] % p as u64) as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mantissa_reduction_is_exact() {
        for &p in &[268042241u32, 1073479681, 65521, 3] {
            let acc: Vec<u64> = (0..1000u64)
                .map(|i| match i % 4 {
                    0 => i.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 1,
                    1 => (1u64 << 63) - 1 - i,
                    2 => i,
                    _ => (p as u64 - 1) * (p as u64 - 1) * 6,
                })
                .collect();
            let mut out = vec![0u32; acc.len()];
            reduce(&acc, p, &mut out);
            for (a, o) in acc.iter().zip(&out) {
                assert_eq!(*o as u64, a % p as u64, "{a} mod {p}");
            }
        }
    }

    #[test]
    fn macs() {
        let x: Vec<u32> = (0..37).map(|i| i * 77_777_777).collect();
        let w: Vec<u32> = (0..37).map(|i| u32::MAX - i).collect();
        let mut a = vec![5u64; 37];
        let mut b = vec![7u64; 37];
        mac2(&mut a, &mut b, &x, &w, &x);
        for i in 0..37 {
            assert_eq!(a[i], 5 + x[i] as u64 * w[i] as u64);
            assert_eq!(b[i], 7 + x[i] as u64 * x[i] as u64);
        }
    }
}

/// The word-generic forms the kernels call: the AVX-512 rows at u32, the scalar formulas otherwise.
pub mod w {
    use crate::arith::Hw;
    use pc_kernels::PrimeConst;

    #[inline(always)]
    fn u32s<W>(x: &[W]) -> &[u32] {
        // SAFETY: only called when W is u32
        unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u32, x.len()) }
    }
    #[inline(always)]
    fn u32m<W>(x: &mut [W]) -> &mut [u32] {
        // SAFETY: only called when W is u32
        unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut u32, x.len()) }
    }
    #[inline(always)]
    fn accm<A>(x: &mut [A]) -> &mut [u64] {
        // SAFETY: only called when W is u32 (Acc = u64)
        unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut u64, x.len()) }
    }
    #[inline(always)]
    fn accs<A>(x: &[A]) -> &[u64] {
        // SAFETY: as above
        unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u64, x.len()) }
    }
    #[inline(always)]
    fn is32<W: Hw>() -> bool {
        W::BITS == 32
    }

    pub fn mac<W: Hw>(acc: &mut [W::Acc], x: &[W], w: &[W]) {
        if is32::<W>() {
            return super::mac(accm(acc), u32s(x), u32s(w));
        }
        for i in 0..acc.len() {
            acc[i] += W::wmul(x[i], w[i]);
        }
    }
    pub fn mac2<W: Hw>(acc0: &mut [W::Acc], acc1: &mut [W::Acc], x: &[W], w0: &[W], w1: &[W]) {
        if is32::<W>() {
            return super::mac2(accm(acc0), accm(acc1), u32s(x), u32s(w0), u32s(w1));
        }
        for i in 0..acc0.len() {
            acc0[i] += W::wmul(x[i], w0[i]);
            acc1[i] += W::wmul(x[i], w1[i]);
        }
    }
    /// r = r + x w (Shoup), or x w when `set`.
    pub fn shoup_row<W: Hw>(r: &mut [W], x: &[W], w: W, ws: W, p: W, set: bool) {
        if is32::<W>() {
            return super::shoup_row(u32m(r), u32s(x), w.u64() as u32, ws.u64() as u32, p.u64() as u32, set);
        }
        for i in 0..r.len() {
            let v = W::mul_shoup(x[i], w, ws, p);
            r[i] = if set { v } else { W::add(r[i], v, p) };
        }
    }
    pub fn mac_s<W: Hw>(acc: &mut [W::Acc], x: &[W], w: W) {
        if is32::<W>() {
            return super::mac_s(accm(acc), u32s(x), w.u64() as u32);
        }
        for i in 0..acc.len() {
            acc[i] += W::wmul(x[i], w);
        }
    }
    /// out[i] = x[i] w mod p (general product: the row through the accumulator).
    pub fn mul_s<W: Hw>(out: &mut [W], x: &[W], w: W, q: &PrimeConst, scratch: &mut [W::Acc]) {
        scratch.fill(W::Acc::default());
        mac_s::<W>(scratch, x, w);
        reduce::<W>(scratch, q, out);
    }
    /// out[i] = x[i] mod q for residues of another prime (any word value).
    pub fn red_other<W: Hw>(out: &mut [W], x: &[W], q: &PrimeConst, scratch: &mut [W::Acc]) {
        mul_s::<W>(out, x, W::from_u64(1), q, scratch)
    }
    pub fn csub_gt<W: Hw>(x: &mut [W], u: &[W], h: W, c: W, q: W) {
        if is32::<W>() {
            return super::csub_gt(u32m(x), u32s(u), h.u64() as u32, c.u64() as u32, q.u64() as u32);
        }
        for i in 0..x.len() {
            if u[i] > h {
                x[i] = W::sub(x[i], c, q);
            }
        }
    }
    pub fn add_rows<W: Hw>(x: &mut [W], y: &[W], p: W) {
        if is32::<W>() {
            return super::add_rows(u32m(x), u32s(y), p.u64() as u32);
        }
        for i in 0..x.len() {
            x[i] = W::add(x[i], y[i], p);
        }
    }
    pub fn gather<W: Hw>(out: &mut [W], src: &[W], idx: &[u32]) {
        if is32::<W>() {
            return super::gather(u32m(out), u32s(src), idx);
        }
        for i in 0..out.len() {
            out[i] = src[idx[i] as usize];
        }
    }
    pub fn reduce<W: Hw>(acc: &[W::Acc], q: &PrimeConst, out: &mut [W]) {
        if is32::<W>() {
            return super::reduce(accs(acc), q.p as u32, u32m(out));
        }
        for i in 0..acc.len() {
            out[i] = W::reduce(acc[i], q);
        }
    }
}

/// Values of `bits` bits (1..=32) packed little-endian in a u32 stream: out[i] = bits [i bits, (i + 1) bits).
pub fn unpack(words: &[u32], bits: u32, out: &mut [u32]) {
    #[cfg(target_arch = "x86_64")]
    if wide() && vbmi() {
        // SAFETY: CPU features checked
        return unsafe { unpack_v(words, bits, out) };
    }
    unpack_scalar(words, bits, 0, out)
}

fn unpack_scalar(words: &[u32], bits: u32, from: usize, out: &mut [u32]) {
    let mask = if bits == 32 { u32::MAX as u64 } else { (1u64 << bits) - 1 };
    for i in from..out.len() {
        let bit = i as u64 * bits as u64;
        let (w, sh) = ((bit >> 5) as usize, bit & 31);
        let v = (words[w] as u64 | (words.get(w + 1).copied().unwrap_or(0) as u64) << 32) >> sh;
        out[i] = (v & mask) as u32;
    }
}

#[cfg(target_arch = "x86_64")]
fn vbmi() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::is_x86_feature_detected!("avx512vbmi") && std::is_x86_feature_detected!("avx512bw"))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vl")]
unsafe fn unpack_v(words: &[u32], bits: u32, out: &mut [u32]) {
    // 8 values from a byte-aligned base (8 values = `bits` whole bytes): lane k takes bytes rb_k .. rb_k + 8, shifted
    // right by (k bits) mod 8
    let (mut idx, mut sh) = ([0u8; 64], [0u64; 8]);
    for k in 0..8 {
        let bit = k * bits as usize;
        for j in 0..8 {
            idx[8 * k + j] = (bit / 8 + j) as u8;
        }
        sh[k] = (bit % 8) as u64;
    }
    let iv = _mm512_loadu_si512(idx.as_ptr() as *const _);
    let sv = _mm512_loadu_si512(sh.as_ptr() as *const _);
    let mask = _mm512_set1_epi64(if bits == 32 { u32::MAX as i64 } else { (1i64 << bits) - 1 });
    let bytes = words.as_ptr() as *const u8;
    let avail = words.len() * 4;
    let mut i = 0;
    // a group reads 64 bytes from its base; past the buffer, the scalar tail
    while i + 8 <= out.len() && i * bits as usize / 8 + 64 <= avail {
        let d = _mm512_loadu_si512(bytes.add(i * bits as usize / 8) as *const _);
        let v = _mm512_and_si512(_mm512_srlv_epi64(_mm512_permutexvar_epi8(iv, d), sv), mask);
        _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut _, _mm512_cvtepi64_epi32(v));
        i += 8;
    }
    unpack_scalar(words, bits, i, out)
}

#[cfg(test)]
mod unpack_tests {
    use super::*;

    #[test]
    fn unpack_matches_the_bit_stream() {
        for bits in [28u32, 27, 30, 31, 32, 17] {
            let n = 1024;
            let vals: Vec<u32> = (0..n as u64).map(|i| ((i * 0x9E37_79B9) % (1u64 << bits)) as u32).collect();
            let mut words = vec![0u32; n * bits as usize / 32 + 2];
            for (s, &v) in vals.iter().enumerate() {
                let bit = s as u64 * bits as u64;
                let (w, o) = ((bit >> 5) as usize, bit & 31);
                let x = (v as u64) << o;
                words[w] |= x as u32;
                words[w + 1] |= (x >> 32) as u32;
            }
            let mut out = vec![0u32; n];
            unpack(&words, bits, &mut out);
            assert_eq!(out, vals, "bits {bits}");
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mulhi32(a: __m512i, b: __m512i) -> __m512i {
    let even = _mm512_srli_epi64::<32>(_mm512_mul_epu32(a, b));
    let odd = _mm512_mul_epu32(_mm512_srli_epi64::<32>(a), _mm512_srli_epi64::<32>(b));
    _mm512_mask_blend_epi32(0xAAAA, even, odd)
}
/// Shoup product, canonical (any a < 2^32, w < p < 2^31).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn shoup32(a: __m512i, w: __m512i, ws: __m512i, p: __m512i) -> __m512i {
    let r = _mm512_sub_epi32(_mm512_mullo_epi32(a, w), _mm512_mullo_epi32(mulhi32(a, ws), p));
    _mm512_min_epu32(r, _mm512_sub_epi32(r, p))
}
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn addm32(a: __m512i, b: __m512i, p: __m512i) -> __m512i {
    let s = _mm512_add_epi32(a, b);
    _mm512_min_epu32(s, _mm512_sub_epi32(s, p))
}

/// r[i] = r[i] + x[i] w mod p (Shoup), or r[i] = x[i] w when `set`.
pub fn shoup_row(r: &mut [u32], x: &[u32], w: u32, ws: u32, p: u32, set: bool) {
    #[cfg(target_arch = "x86_64")]
    if wide() {
        // SAFETY: CPU features checked
        return unsafe { shoup_row_v(r, x, w, ws, p, set) };
    }
    for i in 0..r.len() {
        let q = ((x[i] as u64 * ws as u64) >> 32) as u32;
        let mut v = x[i].wrapping_mul(w).wrapping_sub(q.wrapping_mul(p));
        if v >= p {
            v -= p;
        }
        r[i] = if set {
            v
        } else {
            let s = r[i] + v;
            if s >= p {
                s - p
            } else {
                s
            }
        };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn shoup_row_v(r: &mut [u32], x: &[u32], w: u32, ws: u32, p: u32, set: bool) {
    let (wv, wsv, pv) = (_mm512_set1_epi32(w as i32), _mm512_set1_epi32(ws as i32), _mm512_set1_epi32(p as i32));
    let n = r.len() / 16 * 16;
    for i in (0..n).step_by(16) {
        let v = shoup32(_mm512_loadu_si512(x.as_ptr().add(i) as *const _), wv, wsv, pv);
        let o = if set { v } else { addm32(_mm512_loadu_si512(r.as_ptr().add(i) as *const _), v, pv) };
        _mm512_storeu_si512(r.as_mut_ptr().add(i) as *mut _, o);
    }
    if n < r.len() {
        shoup_row(&mut r[n..], &x[n..], w, ws, p, set);
    }
}

/// out[i] = src[idx[i]].
pub fn gather(out: &mut [u32], src: &[u32], idx: &[u32]) {
    #[cfg(target_arch = "x86_64")]
    if wide() {
        // SAFETY: CPU features checked; indices inside src (callers' tables)
        return unsafe { gather_v(out, src, idx) };
    }
    for i in 0..out.len() {
        out[i] = src[idx[i] as usize];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn gather_v(out: &mut [u32], src: &[u32], idx: &[u32]) {
    let n = out.len() / 16 * 16;
    for i in (0..n).step_by(16) {
        let iv = _mm512_loadu_si512(idx.as_ptr().add(i) as *const _);
        _mm512_storeu_si512(out.as_mut_ptr().add(i) as *mut _, _mm512_i32gather_epi32::<4>(iv, src.as_ptr() as *const _));
    }
    for i in n..out.len() {
        out[i] = src[idx[i] as usize];
    }
}

/// acc[i] += x[i] * w (one weight for the row).
pub fn mac_s(acc: &mut [u64], x: &[u32], w: u32) {
    #[cfg(target_arch = "x86_64")]
    if wide() {
        // SAFETY: CPU features checked
        return unsafe { mac_s_v(acc, x, w) };
    }
    for i in 0..acc.len() {
        acc[i] += x[i] as u64 * w as u64;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn mac_s_v(acc: &mut [u64], x: &[u32], w: u32) {
    let n = acc.len() / 8 * 8;
    let wv = _mm512_set1_epi64(w as i64);
    for i in (0..n).step_by(8) {
        let xv = _mm512_cvtepu32_epi64(_mm256_loadu_si256(x.as_ptr().add(i) as *const _));
        let a = _mm512_loadu_si512(acc.as_ptr().add(i) as *const _);
        _mm512_storeu_si512(acc.as_mut_ptr().add(i) as *mut _, _mm512_add_epi64(a, _mm512_mul_epu32(xv, wv)));
    }
    for i in n..acc.len() {
        acc[i] += x[i] as u64 * w as u64;
    }
}

/// v[i] = fma(y[i] as f64, r, v[i]) (IEEE fused multiply-add, as the device's __fma_rn).
pub fn fma_f64(v: &mut [f64], y: &[u32], r: f64) {
    #[cfg(target_arch = "x86_64")]
    if wide() {
        // SAFETY: CPU features checked
        return unsafe { fma_f64_v(v, y, r) };
    }
    for i in 0..v.len() {
        v[i] = (y[i] as f64).mul_add(r, v[i]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn fma_f64_v(v: &mut [f64], y: &[u32], r: f64) {
    let n = v.len() / 8 * 8;
    let rv = _mm512_set1_pd(r);
    for i in (0..n).step_by(8) {
        let yv = _mm512_cvtepu32_pd(_mm256_loadu_si256(y.as_ptr().add(i) as *const _));
        let a = _mm512_loadu_pd(v.as_ptr().add(i));
        _mm512_storeu_pd(v.as_mut_ptr().add(i), _mm512_fmadd_pd(yv, rv, a));
    }
    for i in n..v.len() {
        v[i] = (y[i] as f64).mul_add(r, v[i]);
    }
}

/// x[i] = u[i] > h ? x[i] - c mod q : x[i] (c < q): the centred lift's correction.
pub fn csub_gt(x: &mut [u32], u: &[u32], h: u32, c: u32, q: u32) {
    #[cfg(target_arch = "x86_64")]
    if wide() {
        // SAFETY: CPU features checked
        return unsafe { csub_gt_v(x, u, h, c, q) };
    }
    for i in 0..x.len() {
        if u[i] > h {
            x[i] = if x[i] >= c { x[i] - c } else { x[i] + q - c };
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn csub_gt_v(x: &mut [u32], u: &[u32], h: u32, c: u32, q: u32) {
    let n = x.len() / 16 * 16;
    let (hv, cv, qv) = (_mm512_set1_epi32(h as i32), _mm512_set1_epi32(c as i32), _mm512_set1_epi32(q as i32));
    for i in (0..n).step_by(16) {
        let xv = _mm512_loadu_si512(x.as_ptr().add(i) as *const _);
        let m = _mm512_cmpgt_epu32_mask(_mm512_loadu_si512(u.as_ptr().add(i) as *const _), hv);
        // x - c + q, then one conditional q down (min trick)
        let d = _mm512_add_epi32(_mm512_sub_epi32(xv, cv), qv);
        let d = _mm512_min_epu32(d, _mm512_sub_epi32(d, qv));
        _mm512_storeu_si512(x.as_mut_ptr().add(i) as *mut _, _mm512_mask_mov_epi32(xv, m, d));
    }
    if n < x.len() {
        csub_gt(&mut x[n..], &u[n..], h, c, q);
    }
}

/// x[i] = x[i] + y[i] mod p.
pub fn add_rows(x: &mut [u32], y: &[u32], p: u32) {
    #[cfg(target_arch = "x86_64")]
    if wide() {
        // SAFETY: CPU features checked
        return unsafe { add_rows_v(x, y, p) };
    }
    for i in 0..x.len() {
        let s = x[i] + y[i];
        x[i] = if s >= p { s - p } else { s };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
unsafe fn add_rows_v(x: &mut [u32], y: &[u32], p: u32) {
    let n = x.len() / 16 * 16;
    let pv = _mm512_set1_epi32(p as i32);
    for i in (0..n).step_by(16) {
        let s = addm32(_mm512_loadu_si512(x.as_ptr().add(i) as *const _), _mm512_loadu_si512(y.as_ptr().add(i) as *const _), pv);
        _mm512_storeu_si512(x.as_mut_ptr().add(i) as *mut _, s);
    }
    if n < x.len() {
        add_rows(&mut x[n..], &y[n..], p);
    }
}

#[cfg(test)]
mod port_bench {
    #[test]
    #[ignore]
    #[cfg(target_arch = "x86_64")]
    fn multiplier_throughput() {
        use std::arch::x86_64::*;
        #[target_feature(enable = "avx512f,avx512ifma,avx512dq")]
        unsafe fn run(kind: u32, iters: u64) -> f64 {
            let mut a = [_mm512_set1_epi64(3); 8];
            let b = _mm512_set1_epi64(5);
            let mut f = [_mm512_set1_pd(1.0000001); 8];
            let fb = _mm512_set1_pd(0.9999999);
            let t = std::time::Instant::now();
            for _ in 0..iters {
                for k in 0..8 {
                    match kind {
                        0 => a[k] = _mm512_mul_epu32(a[k], b),
                        1 => a[k] = _mm512_madd52lo_epu64(a[k], a[k], b),
                        2 => f[k] = _mm512_fmadd_pd(f[k], fb, fb),
                        _ => a[k] = _mm512_mullo_epi64(a[k], b),
                    }
                }
            }
            let ns = t.elapsed().as_nanos() as f64;
            std::hint::black_box((a, f));
            ns / (iters * 8) as f64
        }
        for (k, name) in [(0, "vpmuludq"), (1, "vpmadd52luq"), (2, "vfmadd231pd"), (3, "vpmullq")] {
            // SAFETY: AVX-512 IFMA machine (bench only)
            let ns = unsafe { run(k, 50_000_000) };
            eprintln!("{name}: {ns:.3} ns per zmm op (8 independent chains)");
        }
    }
}
