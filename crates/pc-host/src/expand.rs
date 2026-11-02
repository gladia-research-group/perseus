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

/// Row of the expander for (seed, w13, prime p, tag): `out[s]` = the value of slot s, reduced mod p. Switching-key
/// rows (KSKA) come from the key-row cache once expanded.
pub fn expand_row<W: Hw>(seed: &[u32; 8], w13: u32, p: u64, tag: u32, out: &mut [W]) {
    if W::BITS == 32 {
        // SAFETY: W is u32
        let o = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u32, out.len()) };
        if tag == TAG_KSKA {
            return cache::row(seed, w13, p as u32, o);
        }
        return crate::chacha::expand_row_u32(seed, w13, p as u32, tag, o);
    }
    expand_row_generic(seed, w13, p, tag, out)
}

fn expand_row_generic<W: Hw>(seed: &[u32; 8], w13: u32, p: u64, tag: u32, out: &mut [W]) {
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

/// Expanded switching-key rows by (seed, digit, prime): the host keeps what the device regenerates, up to
/// PERSEUS_CPU_KEYCACHE_GB (default 16; 0 = off). A full cache keeps its rows (a key set is replayed in the same
/// order every bootstrap, so evicting would only thrash).
pub mod cache {
    use super::TAG_KSKA;
    use std::collections::HashMap;
    use std::sync::{Arc, OnceLock, RwLock};

    type Key = ([u32; 8], u32, u32, usize);
    struct Cache {
        rows: RwLock<HashMap<Key, Arc<[u32]>>>,
        cap: usize,
        used: std::sync::atomic::AtomicUsize,
    }

    fn cache() -> &'static Cache {
        static C: OnceLock<Cache> = OnceLock::new();
        C.get_or_init(|| {
            let gb: f64 = std::env::var("PERSEUS_CPU_KEYCACHE_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(16.0);
            Cache { rows: RwLock::new(HashMap::new()), cap: (gb * 1e9) as usize, used: 0.into() }
        })
    }

    /// The row itself (cached, or expanded now and kept while the budget allows).
    pub fn get(seed: &[u32; 8], w13: u32, p: u32, n: usize) -> Arc<[u32]> {
        let c = cache();
        let key = (*seed, w13, p, n);
        if c.cap > 0 {
            if let Some(r) = c.rows.read().unwrap().get(&key) {
                return r.clone();
            }
        }
        let mut v = vec![0u32; n];
        crate::chacha::expand_row_u32(seed, w13, p, TAG_KSKA, &mut v);
        let r: Arc<[u32]> = v.into();
        use std::sync::atomic::Ordering::Relaxed;
        if c.cap > 0 && c.used.load(Relaxed) + 4 * n <= c.cap {
            c.used.fetch_add(4 * n, Relaxed);
            c.rows.write().unwrap().insert(key, r.clone());
        }
        r
    }

    pub fn row(seed: &[u32; 8], w13: u32, p: u32, out: &mut [u32]) {
        let c = cache();
        let key = (*seed, w13, p, out.len());
        if c.cap > 0 {
            if let Some(r) = c.rows.read().unwrap().get(&key) {
                out.copy_from_slice(r);
                return;
            }
        }
        crate::chacha::expand_row_u32(seed, w13, p, TAG_KSKA, out);
        let bytes = 4 * out.len();
        use std::sync::atomic::Ordering::Relaxed;
        if c.cap > 0 && c.used.load(Relaxed) + bytes <= c.cap {
            c.used.fetch_add(bytes, Relaxed);
            c.rows.write().unwrap().insert(key, out.into());
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
        expand_row::<u32>(&seed, 3, p, TAG_KSKA, &mut a);
        assert_eq!(a, b);
        let mut c = vec![0u32; N];
        expand_row_generic::<u32>(&seed, 7, p, TAG_ENCA, &mut c);
        expand_row::<u32>(&seed, 7, p, TAG_ENCA, &mut a);
        assert_eq!(a, c);
        let p = (1u64 << 59) - 55;
        let (mut a, mut b) = (vec![0u64; N], vec![0u64; N]);
        expand_row::<u64>(&seed, 2, p, TAG_KSKA, &mut a);
        kskb_expand_row(&KskSeed(seed), 2, p, &mut b);
        assert_eq!(a, b);
    }
}
