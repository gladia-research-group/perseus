// The seeded expander (chacha.cuh expand_slot): ChaCha12 counter mode, 64 / sizeof(W) values per block, rejection
// retried at block counter + t N / per-block (t < 16), the u64 lane's w13 / tag adjustments. Whole rows, one base
// block per group of slots, as pc-core's KSKA / KSKB twins.
use crate::arith::Hw;
use pc_core::keys::chacha_block;
use pc_kernels::args::N;

pub const TAG_KSKA: u32 = 0x4b53_4b41;
pub const TAG_ENCA: u32 = 0x454e_4341;

#[inline(always)]
fn value_at<W: Hw>(blk: &[u32; 16], j: usize) -> u64 {
    match W::BITS {
        32 => blk[j] as u64,
        64 => blk[2 * j] as u64 | (blk[2 * j + 1] as u64) << 32,
        _ => ((blk[j >> 1] >> (16 * (j & 1))) & 0xFFFF) as u64,
    }
}

/// Row of the expander for (seed, w13, prime p, tag): `out[s]` = the value of slot s, reduced mod p.
pub fn expand_row<W: Hw>(seed: &[u32; 8], w13: u32, p: u64, tag: u32, out: &mut [W]) {
    let per = 64 / (W::BITS as usize / 8);
    let nblk = (N / per) as u32;
    let wmax = if W::BITS == 64 { u64::MAX } else { (1u64 << W::BITS) - 1 };
    let mp = (wmax / p) * p;
    let (w13l, tagl) = if W::BITS == 64 { (w13 ^ (((p >> 32) as u32) << 4), tag + 1) } else { (w13, tag) };
    let w14 = p as u32;
    for (b0, chunk) in out.chunks_exact_mut(per).enumerate() {
        let b0 = b0 as u32;
        let base = chacha_block(seed, b0, w13l, w14, tagl, 12);
        for (j, o) in chunk.iter_mut().enumerate() {
            let mut v = value_at::<W>(&base, j);
            let mut t = 1u32;
            while v >= mp && t < 16 {
                v = value_at::<W>(&chacha_block(seed, b0.wrapping_add(t * nblk), w13l, w14, tagl, 12), j);
                t += 1;
            }
            *o = W::from_u64(v % p);
        }
    }
}

/// The 8-word seed at `p`.
pub fn seed_at(p: *const u32) -> [u32; 8] {
    // SAFETY: an 8-word seed (launch contract)
    unsafe { *(p as *const [u32; 8]) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pc_core::keys::{kska_expand_row, kskb_expand_row, KskSeed};

    #[test]
    fn matches_the_frozen_lanes() {
        let seed = [1u32, 2, 3, 4, 5, 6, 7, 0xdeadbeef];
        let p = 268042241u64;
        let (mut a, mut b) = (vec![0u32; N], vec![0u32; N]);
        expand_row::<u32>(&seed, 3, p, TAG_KSKA, &mut a);
        kska_expand_row(&KskSeed(seed), 3, p as u32, &mut b);
        assert_eq!(a, b);
        let p = (1u64 << 59) - 55;
        let (mut a, mut b) = (vec![0u64; N], vec![0u64; N]);
        expand_row::<u64>(&seed, 2, p, TAG_KSKA, &mut a);
        kskb_expand_row(&KskSeed(seed), 2, p, &mut b);
        assert_eq!(a, b);
    }
}
