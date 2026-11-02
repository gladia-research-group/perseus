// ChaCha12 over 16 consecutive block counters at once (AVX-512: one block per 32-bit lane), and the u32 expander row
// built on it: words accepted below floor((2^32 - 1) / p) p and reduced as the device does (v - umulhi(v,
// floor((2^32 - 1) / p)) p, one guard), rejected words retried block by block (rare: < p / 2^32 of the words).
use pc_core::keys::chacha_block;

const SIGMA: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];

/// 16 blocks with counters ctr0 .. ctr0 + 15: out[j][b] = word j of block ctr0 + b.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn blocks16_v(key: &[u32; 8], ctr0: u32, w13: u32, w14: u32, w15: u32, out: &mut [[u32; 16]; 16]) {
    use std::arch::x86_64::*;
    let s: [__m512i; 16] = [
        _mm512_set1_epi32(SIGMA[0] as i32),
        _mm512_set1_epi32(SIGMA[1] as i32),
        _mm512_set1_epi32(SIGMA[2] as i32),
        _mm512_set1_epi32(SIGMA[3] as i32),
        _mm512_set1_epi32(key[0] as i32),
        _mm512_set1_epi32(key[1] as i32),
        _mm512_set1_epi32(key[2] as i32),
        _mm512_set1_epi32(key[3] as i32),
        _mm512_set1_epi32(key[4] as i32),
        _mm512_set1_epi32(key[5] as i32),
        _mm512_set1_epi32(key[6] as i32),
        _mm512_set1_epi32(key[7] as i32),
        _mm512_add_epi32(_mm512_set1_epi32(ctr0 as i32), _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)),
        _mm512_set1_epi32(w13 as i32),
        _mm512_set1_epi32(w14 as i32),
        _mm512_set1_epi32(w15 as i32),
    ];
    let mut x = s;
    macro_rules! qr {
        ($a:expr, $b:expr, $c:expr, $d:expr) => {
            x[$a] = _mm512_add_epi32(x[$a], x[$b]);
            x[$d] = _mm512_rol_epi32::<16>(_mm512_xor_si512(x[$d], x[$a]));
            x[$c] = _mm512_add_epi32(x[$c], x[$d]);
            x[$b] = _mm512_rol_epi32::<12>(_mm512_xor_si512(x[$b], x[$c]));
            x[$a] = _mm512_add_epi32(x[$a], x[$b]);
            x[$d] = _mm512_rol_epi32::<8>(_mm512_xor_si512(x[$d], x[$a]));
            x[$c] = _mm512_add_epi32(x[$c], x[$d]);
            x[$b] = _mm512_rol_epi32::<7>(_mm512_xor_si512(x[$b], x[$c]));
        };
    }
    for _ in 0..6 {
        qr!(0, 4, 8, 12);
        qr!(1, 5, 9, 13);
        qr!(2, 6, 10, 14);
        qr!(3, 7, 11, 15);
        qr!(0, 5, 10, 15);
        qr!(1, 6, 11, 12);
        qr!(2, 7, 8, 13);
        qr!(3, 4, 9, 14);
    }
    for j in 0..16 {
        _mm512_storeu_si512(out[j].as_mut_ptr() as *mut _, _mm512_add_epi32(x[j], s[j]));
    }
}

fn blocks16_scalar(key: &[u32; 8], ctr0: u32, w13: u32, w14: u32, w15: u32, out: &mut [[u32; 16]; 16]) {
    for b in 0..16u32 {
        let blk = chacha_block(key, ctr0.wrapping_add(b), w13, w14, w15, 12);
        for j in 0..16 {
            out[j][b as usize] = blk[j];
        }
    }
}

pub fn wide() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *W.get_or_init(|| std::env::var_os("PC_HOST_SCALAR").is_none_or(|v| !v.to_string_lossy().contains("CHACHA")) && std::is_x86_feature_detected!("avx512f"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// 16 consecutive blocks, word-major (out[j][b]).
pub fn blocks16(key: &[u32; 8], ctr0: u32, w13: u32, w14: u32, w15: u32, out: &mut [[u32; 16]; 16]) {
    #[cfg(target_arch = "x86_64")]
    if wide() {
        // SAFETY: CPU feature checked
        return unsafe { blocks16_v(key, ctr0, w13, w14, w15, out) };
    }
    blocks16_scalar(key, ctr0, w13, w14, w15, out)
}

/// The u32 expander row (KSKA / ENCA lanes): out[s] for s < out.len(), 16 values per block, retries at counter
/// + t len/16 (t < 16, the 16th kept as is).
pub fn expand_row_u32(seed: &[u32; 8], w13: u32, p: u32, tag: u32, out: &mut [u32]) {
    let n16 = (out.len() / 16) as u32;
    let mu = u32::MAX / p;
    let mp = mu * p;
    let red = |v: u32| {
        let r = v.wrapping_sub(((v as u64 * mu as u64) >> 32) as u32 * p);
        if r >= p {
            r - p
        } else {
            r
        }
    };
    let mut blk = [[0u32; 16]; 16];
    for g in 0..(n16 as usize).div_ceil(16) {
        let b0 = (g * 16) as u32;
        blocks16(seed, b0, w13, p, tag, &mut blk);
        for b in 0..16usize.min(n16 as usize - g * 16) {
            let o = &mut out[(g * 16 + b) * 16..(g * 16 + b + 1) * 16];
            for j in 0..16 {
                let mut v = blk[j][b];
                let mut t = 1u32;
                while v >= mp && t < 16 {
                    v = chacha_block(seed, (b0 + b as u32).wrapping_add(t * n16), w13, p, tag, 12)[j];
                    t += 1;
                }
                o[j] = red(v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_blocks_match_the_scalar_ones() {
        let key = [9u32, 8, 7, 6, 5, 4, 3, 0xffff_fff1];
        let (mut a, mut b) = ([[0u32; 16]; 16], [[0u32; 16]; 16]);
        blocks16(&key, 0xffff_fff8, 3, 268042241, 0x4b53_4b41, &mut a);
        blocks16_scalar(&key, 0xffff_fff8, 3, 268042241, 0x4b53_4b41, &mut b);
        assert_eq!(a, b);
    }
}
