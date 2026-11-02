// Modular arithmetic over the instantiation's word, the device's formulas (arith.cuh): canonical residues in and
// out. Constant operands use Shoup with the precomputed companion; products of two residues and lazy accumulators
// reduce exactly (the ModMul strategies of the device all return the canonical residue, so one exact reduction stands
// for every strategy).
use pc_kernels::PrimeConst;

pub trait Hw: Copy + Send + Sync + Default + PartialEq + PartialOrd + std::fmt::Debug + 'static {
    const BITS: u32;
    /// Accumulator of lazy sums of products (at least two words wide).
    type Acc: Copy + Send + Sync + Default + std::ops::Add<Output = Self::Acc> + std::ops::AddAssign + 'static;
    fn from_u64(x: u64) -> Self;
    fn u64(self) -> u64;
    fn wmul(a: Self, b: Self) -> Self::Acc;
    fn acc_u128(x: Self::Acc) -> u128;
    /// Exact `x mod p` of an accumulator (the device's double-word Barrett, which is exact on its domain).
    fn reduce(x: Self::Acc, c: &PrimeConst) -> Self;
    /// Shoup product: w' = floor(w 2^BITS / p), any a < 2^BITS, w < p.
    fn mul_shoup(a: Self, w: Self, ws: Self, p: Self) -> Self;
    fn add(a: Self, b: Self, p: Self) -> Self;
    fn sub(a: Self, b: Self, p: Self) -> Self;
    #[inline(always)]
    fn mul(a: Self, b: Self, c: &PrimeConst) -> Self {
        Self::reduce(Self::wmul(a, b), c)
    }
    /// A raw per-row constant reduced once (`k >= p ? k - p : k`).
    #[inline(always)]
    fn red1(k: Self, p: Self) -> Self {
        if k >= p {
            Self::from_u64(k.u64() - p.u64())
        } else {
            k
        }
    }
}

macro_rules! narrow {
    ($w:ty, $acc:ty, $bits:expr) => {
        impl Hw for $w {
            const BITS: u32 = $bits;
            type Acc = $acc;
            #[inline(always)]
            fn from_u64(x: u64) -> Self {
                x as $w
            }
            #[inline(always)]
            fn u64(self) -> u64 {
                self as u64
            }
            #[inline(always)]
            fn wmul(a: Self, b: Self) -> $acc {
                a as $acc * b as $acc
            }
            #[inline(always)]
            fn acc_u128(x: $acc) -> u128 {
                x as u128
            }
            #[inline(always)]
            fn reduce(x: $acc, c: &PrimeConst) -> Self {
                reduce64(x as u64, c) as $w
            }
            #[inline(always)]
            fn mul_shoup(a: Self, w: Self, ws: Self, p: Self) -> Self {
                let q = ((a as u64 * ws as u64) >> $bits) as $w;
                let r = a.wrapping_mul(w).wrapping_sub(q.wrapping_mul(p));
                if r >= p {
                    r - p
                } else {
                    r
                }
            }
            #[inline(always)]
            fn add(a: Self, b: Self, p: Self) -> Self {
                let s = a.wrapping_add(b);
                if s >= p {
                    s.wrapping_sub(p)
                } else {
                    s
                }
            }
            #[inline(always)]
            fn sub(a: Self, b: Self, p: Self) -> Self {
                if a >= b {
                    a - b
                } else {
                    a.wrapping_add(p).wrapping_sub(b)
                }
            }
        }
    };
}
/// Exact `x mod p` for any 64-bit x and p < 2^32 (the device's u32 double-word Barrett with mu = floor(2^64 / p): the
/// quotient estimate is low by at most 2). Built for u16 words, `mu_lo` is floor(2^32 / p): then plain division.
#[inline(always)]
fn reduce64(x: u64, c: &PrimeConst) -> u64 {
    if c.word_bits != 32 {
        return x % c.p;
    }
    let q = ((x as u128 * c.mu_lo as u128) >> 64) as u64;
    let p = c.p;
    let mut r = x.wrapping_sub(q.wrapping_mul(p));
    if r >= p {
        r -= p;
    }
    if r >= p {
        r -= p;
    }
    r
}

narrow!(u16, u64, 16);
narrow!(u32, u64, 32);

impl Hw for u64 {
    const BITS: u32 = 64;
    type Acc = u128;
    #[inline(always)]
    fn from_u64(x: u64) -> Self {
        x
    }
    #[inline(always)]
    fn u64(self) -> u64 {
        self
    }
    #[inline(always)]
    fn wmul(a: Self, b: Self) -> u128 {
        a as u128 * b as u128
    }
    #[inline(always)]
    fn acc_u128(x: u128) -> u128 {
        x
    }
    #[inline(always)]
    fn reduce(x: u128, c: &PrimeConst) -> Self {
        // the device's partial-product Barrett (reduce_dw<uint64_t>): exact for every x < 2^128 at p < 2^62
        let (x0, x1) = (x as u64, (x >> 64) as u64);
        let (m0, m1) = (c.mu_lo, c.mu_hi);
        let mid = x1 as u128 * m0 as u128 + x0 as u128 * m1 as u128 + ((x0 as u128 * m0 as u128) >> 64);
        let q = x1.wrapping_mul(m1).wrapping_add((mid >> 64) as u64);
        let p = c.p;
        let mut r = x0.wrapping_sub(q.wrapping_mul(p));
        while r >= p {
            r -= p;
        }
        r
    }
    #[inline(always)]
    fn mul_shoup(a: Self, w: Self, ws: Self, p: Self) -> Self {
        let q = ((a as u128 * ws as u128) >> 64) as u64;
        let r = a.wrapping_mul(w).wrapping_sub(q.wrapping_mul(p));
        if r >= p {
            r - p
        } else {
            r
        }
    }
    #[inline(always)]
    fn add(a: Self, b: Self, p: Self) -> Self {
        let s = a.wrapping_add(b);
        if s >= p {
            s.wrapping_sub(p)
        } else {
            s
        }
    }
    #[inline(always)]
    fn sub(a: Self, b: Self, p: Self) -> Self {
        if a >= b {
            a - b
        } else {
            a.wrapping_add(p).wrapping_sub(b)
        }
    }
}

/// Bit-reversed evaluation order automorphism: the source slot of sigma_k for output `slot` (log n bits).
#[inline(always)]
pub fn automorph_slot(k: u32, slot: u32, log_n: u32) -> u32 {
    let j = slot.reverse_bits() >> (32 - log_n);
    let r = (((j << 1) + 1).wrapping_mul(k) & ((1u32 << (log_n + 1)) - 1)) >> 1;
    r.reverse_bits() >> (32 - log_n)
}

/// Compact plaintext index (PtDotArgs::enc).
#[inline(always)]
pub fn ptidx(idx: usize, enc: u32) -> usize {
    if enc >> 24 == 0x7F {
        idx >> (enc & 0xFF)
    } else {
        (idx as u32 & enc) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reductions_are_exact() {
        for &p in &[268042241u64, 1073479681, 65521] {
            let c = PrimeConst::new(p, 32);
            for &(a, b) in &[(p - 1, p - 1), (12345, 67890), (0, p - 1)] {
                assert_eq!(<u32 as Hw>::mul(a as u32, b as u32, &c) as u64, (a as u128 * b as u128 % p as u128) as u64);
                let w = b % p;
                let ws = (((w as u128) << 32) / p as u128) as u32;
                assert_eq!(<u32 as Hw>::mul_shoup(a as u32, w as u32, ws, p as u32) as u64, (a as u128 * w as u128 % p as u128) as u64);
            }
        }
        let p = (1u64 << 59) - 55;
        let c = PrimeConst::new(p, 64);
        for &(a, b) in &[(p - 1, p - 1), (1u64 << 58, 3), (7, 9)] {
            assert_eq!(<u64 as Hw>::mul(a, b, &c), (a as u128 * b as u128 % p as u128) as u64);
        }
        assert_eq!(<u64 as Hw>::reduce(u128::MAX, &c), (u128::MAX % p as u128) as u64);
    }
}
