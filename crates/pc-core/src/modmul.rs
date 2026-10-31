//! General modular product strategies (seam "general modular product") and Shoup constants, host implementations.
//!
//! A `ModMul<W>` owns a per-prime constant block (`PrimeConst`) and a residue representation: Barrett keeps residues
//! in standard form, Montgomery in `a*R mod p` form (`R = 2^W::BITS`). Every strategy maps `[0, p) x [0, p) -> [0, p)`
//! in its own representation and must agree with the exact i128 product after `from_repr` (the contract test in
//! `tests/modmul_contract.rs`). The `PrimeConst` structs are `#[repr(C)]` plain-field blocks: the kernels read them
//! verbatim from a device table indexed by prime id, so their layouts are frozen by the size asserts below.
//!
//! Shoup is not a strategy but the fixed way to multiply by a *known* constant `w` (twiddles, CRT constants,
//! plaintext scalars). `shoup(w, p)` takes `w` in standard form and works under every `ModMul`: multiplying a residue
//! in representation `rep(a)` by a standard-form constant gives `rep(a*w)` for both Barrett (`rep = id`) and
//! Montgomery (`rep(a) = aR`, `aR*w = (aw)R`).
use crate::error::{Error, Result};
use crate::word::Word;
use std::fmt::Debug;

// ---------------------------------------------------------------------------------------------------------------------
// Host reference arithmetic (setup-time tables, CRT constants, CPU checks). Exact, slow, u64 moduli.

#[inline]
pub fn mulmod(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128 * b as u128) % p as u128) as u64
}
#[inline]
pub fn addmod(a: u64, b: u64, p: u64) -> u64 {
    let s = a as u128 + b as u128;
    (if s >= p as u128 { s - p as u128 } else { s }) as u64
}
#[inline]
pub fn submod(a: u64, b: u64, p: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        a + (p - b)
    }
}
pub fn powmod(a: u64, mut e: u64, p: u64) -> u64 {
    let mut r = 1u64 % p;
    let mut a = a % p;
    while e > 0 {
        if e & 1 == 1 {
            r = mulmod(r, a, p);
        }
        a = mulmod(a, a, p);
        e >>= 1;
    }
    r
}
/// Modular inverse for ANY modulus `m >= 2` (extended Euclid; equals Fermat's `a^(m-2)` for a prime `m`).
/// Conventions FIDESlib's `modinv` callers rely on: `invmod(0, m) == 0` (the dumped `q_inv` diagonal), and a
/// non-invertible `a` (gcd(a, m) > 1) also returns 0 instead of a wrong value. The CMT modulus 2^14 is not prime
/// (the Fermat form is wrong there), hence the general algorithm.
pub fn invmod(a: u64, m: u64) -> u64 {
    debug_assert!(m >= 2);
    let a = a % m;
    if a == 0 {
        return 0;
    }
    let (mut old_r, mut r) = (a as i128, m as i128);
    let (mut old_s, mut s) = (1i128, 0i128);
    while r != 0 {
        let q = old_r / r;
        (old_r, r) = (r, old_r - q * r);
        (old_s, s) = (s, old_s - q * s);
    }
    if old_r != 1 {
        return 0;
    }
    (old_s.rem_euclid(m as i128)) as u64
}

// ---------------------------------------------------------------------------------------------------------------------
// The seam.

pub trait ModMul<W: Word>: Copy + Clone + Debug + Send + Sync + 'static {
    /// Per-prime constants, `#[repr(C)]`, consumed by device kernels as-is.
    type PrimeConst: Copy + Clone + Debug + PartialEq + Send + Sync + 'static;
    const NAME: &'static str;
    /// Largest prime bit length this strategy is exact for (the contract test sweeps up to it).
    const MAX_PRIME_BITS: u32;

    fn prime_const(p: W) -> Result<Self::PrimeConst>;
    fn modulus(c: &Self::PrimeConst) -> W;
    /// `a, b < p` in this strategy's representation -> `rep(a*b)`, fully reduced.
    fn mul(a: W, b: W, c: &Self::PrimeConst) -> W;
    /// Standard form -> this strategy's representation (`a < p`).
    fn to_repr(a: W, c: &Self::PrimeConst) -> W;
    /// This strategy's representation -> standard form.
    fn from_repr(a: W, c: &Self::PrimeConst) -> W;
}

fn check_prime<W: Word>(p: W, max_bits: u32, who: &str) -> Result<u32> {
    let pv = p.to_u64();
    let bits = 64 - pv.leading_zeros();
    if pv < 3 || pv % 2 == 0 || bits > max_bits {
        return Err(Error::Params(format!(
            "{who}: modulus {pv} ({bits} bits) outside the strategy's range (odd, 3 <= p < 2^{max_bits})"
        )));
    }
    Ok(bits)
}

// ---------------------------------------------------------------------------------------------------------------------
// Barrett64: double-word Barrett. 32-bit words, p < 2^31.

#[derive(Clone, Copy, Debug)]
pub struct Barrett64;

/// Device layout (`FxPrime`). Invariants: `p` odd, `p < 2^31`; `pad == 0`; `mu = floor(2^64 / p)`.
/// Reduction of any `x < 2^64`: `q = mulhi64(x, mu)` underestimates `floor(x/p)` by at most 2, so `x - q*p < 3p`
/// and two conditional subtractions finish it.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Barrett64Const {
    pub p: u32,
    pub pad: u32,
    pub mu: u64,
}
const _: () = assert!(std::mem::size_of::<Barrett64Const>() == 16);

impl Barrett64 {
    /// fx_reduce64: any `x < 2^64` -> `x mod p`.
    #[inline]
    pub fn reduce(x: u64, c: &Barrett64Const) -> u32 {
        let q = ((x as u128 * c.mu as u128) >> 64) as u64;
        let mut r = x.wrapping_sub(q.wrapping_mul(c.p as u64));
        if r >= c.p as u64 {
            r -= c.p as u64;
        }
        if r >= c.p as u64 {
            r -= c.p as u64;
        }
        r as u32
    }
}

impl ModMul<u32> for Barrett64 {
    type PrimeConst = Barrett64Const;
    const NAME: &'static str = "barrett64";
    const MAX_PRIME_BITS: u32 = 31;
    fn prime_const(p: u32) -> Result<Barrett64Const> {
        check_prime(p, Self::MAX_PRIME_BITS, Self::NAME)?;
        Ok(Barrett64Const { p, pad: 0, mu: ((1u128 << 64) / p as u128) as u64 })
    }
    fn modulus(c: &Barrett64Const) -> u32 {
        c.p
    }
    #[inline]
    fn mul(a: u32, b: u32, c: &Barrett64Const) -> u32 {
        Self::reduce(a as u64 * b as u64, c)
    }
    fn to_repr(a: u32, _: &Barrett64Const) -> u32 {
        a
    }
    fn from_repr(a: u32, _: &Barrett64Const) -> u32 {
        a
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Barrett32: FIDESlib `Neal_mult_32` (ModMult.cuh), the NEIL variant of the "better Barrett" with a 32-bit mu.

#[derive(Clone, Copy, Debug)]
pub struct Barrett32;

/// Device layout. Invariants: `qbit = bit_width(p)`, `qbit <= 30` (FIDESlib only routes primes of <= 28 bits through
/// it, the contract test proves exactness up to 30); `mu = floor(2^(2*qbit+1) / p) < 2^(qbit+2)` (FIDESlib `mu_new`
/// with VERSION == NEIL); `pad == 0`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Barrett32Const {
    pub p: u32,
    pub mu: u32,
    pub qbit: u32,
    pub pad: u32,
}
const _: () = assert!(std::mem::size_of::<Barrett32Const>() == 16);

impl ModMul<u32> for Barrett32 {
    type PrimeConst = Barrett32Const;
    const NAME: &'static str = "barrett32";
    const MAX_PRIME_BITS: u32 = 30;
    fn prime_const(p: u32) -> Result<Barrett32Const> {
        let qbit = check_prime(p, Self::MAX_PRIME_BITS, Self::NAME)?;
        let mu = ((1u128 << (2 * qbit + 1)) / p as u128) as u32;
        Ok(Barrett32Const { p, mu, qbit, pad: 0 })
    }
    fn modulus(c: &Barrett32Const) -> u32 {
        c.p
    }
    /// Neal_mult_32 operation for operation (u32 wrapping arithmetic as on the device).
    #[inline]
    fn mul(a: u32, b: u32, c: &Barrett32Const) -> u32 {
        let x = a as u64 * b as u64;
        let rx = (x >> (c.qbit - 2)) as u32;
        let mut rb = ((((rx << (30 - c.qbit)) as u64) * c.mu as u64) >> 32) as u32 >> 1;
        rb = rb.wrapping_mul(c.p);
        let mut lo = (x as u32).wrapping_sub(rb);
        if lo >= c.p {
            lo -= c.p;
        }
        lo
    }
    fn to_repr(a: u32, _: &Barrett32Const) -> u32 {
        a
    }
    fn from_repr(a: u32, _: &Barrett32Const) -> u32 {
        a
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Montgomery over any word, R = 2^W::BITS.

#[derive(Clone, Copy, Debug)]
pub struct Montgomery;

/// Device layout (three words). Invariants: `p` odd, `p < 2^W::MAX_PRIME_BITS`; `p_neg_inv = -p^-1 mod 2^BITS`;
/// `r2 = R^2 mod p`. Residues are stored as `a*R mod p` in `[0, p)`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct MontConst<W: Word> {
    pub p: W,
    pub p_neg_inv: W,
    pub r2: W,
}
const _: () = assert!(std::mem::size_of::<MontConst<u32>>() == 12);
const _: () = assert!(std::mem::size_of::<MontConst<u64>>() == 24);

#[inline]
fn mask<W: Word>() -> u128 {
    if W::BITS == 128 {
        u128::MAX
    } else {
        (1u128 << W::BITS) - 1
    }
}

impl Montgomery {
    /// REDC: `t < p*R` -> `t * R^-1 mod p`.
    #[inline]
    pub fn redc<W: Word>(t: u128, c: &MontConst<W>) -> W {
        let p = c.p.to_u64() as u128;
        let m = ((t & mask::<W>()) * c.p_neg_inv.to_u64() as u128) & mask::<W>();
        // t + m*p < 2pR <= 2^(2*BITS) for p < 2^(BITS-1)
        let u = (t + m * p) >> W::BITS;
        W::from_u128_wrapping(if u >= p { u - p } else { u })
    }
}

impl<W: Word> ModMul<W> for Montgomery {
    type PrimeConst = MontConst<W>;
    const NAME: &'static str = "montgomery";
    const MAX_PRIME_BITS: u32 = W::MAX_PRIME_BITS;
    fn prime_const(p: W) -> Result<MontConst<W>> {
        check_prime(p, W::MAX_PRIME_BITS, "montgomery")?;
        let pv = p.to_u64() as u128;
        // Newton: inv <- inv * (2 - p*inv) doubles the correct low bits each step (p*p == 1 mod 8 seeds 3 bits).
        let mut inv: u128 = pv;
        for _ in 0..7 {
            inv = (inv.wrapping_mul(2u128.wrapping_sub(pv.wrapping_mul(inv)))) & mask::<W>();
        }
        debug_assert_eq!((inv.wrapping_mul(pv)) & mask::<W>(), 1);
        let p_neg_inv = (inv.wrapping_neg()) & mask::<W>();
        let r = ((1u128 << W::BITS) % pv) as u64;
        let r2 = mulmod(r, r, pv as u64);
        Ok(MontConst { p, p_neg_inv: W::from_u128_wrapping(p_neg_inv), r2: W::from_u64(r2) })
    }
    fn modulus(c: &MontConst<W>) -> W {
        c.p
    }
    #[inline]
    fn mul(a: W, b: W, c: &MontConst<W>) -> W {
        Self::redc(a.to_u64() as u128 * b.to_u64() as u128, c)
    }
    fn to_repr(a: W, c: &MontConst<W>) -> W {
        Self::redc(a.to_u64() as u128 * c.r2.to_u64() as u128, c)
    }
    fn from_repr(a: W, c: &MontConst<W>) -> W {
        Self::redc(a.to_u64() as u128, c)
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Shoup constants.

/// Device layout of a Shoup constant. Invariants: `w < p < 2^(BITS-1)`; `w_prime = floor(w * 2^BITS / p)`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct ShoupConst<W: Word> {
    pub w: W,
    pub w_prime: W,
}
const _: () = assert!(std::mem::size_of::<ShoupConst<u32>>() == 8);

/// Shoup precomputation of the standard-form constant `w` (reduced mod `p` first).
pub fn shoup<W: Word>(w: W, p: W) -> ShoupConst<W> {
    let pv = p.to_u64() as u128;
    let w = (w.to_u64() as u128 % pv) as u64;
    ShoupConst { w: W::from_u64(w), w_prime: W::from_u128_wrapping(((w as u128) << W::BITS) / pv) }
}

/// `a * w mod p` for any `a < 2^BITS` (Shoup_mult_32 / Shoup_mult_64): `q = mulhi(a, w')` is `floor(a*w/p)` or one
/// less, so the wrapped difference lands in `[0, 2p)`.
#[inline]
pub fn mul_shoup<W: Word>(a: W, s: &ShoupConst<W>, p: W) -> W {
    let q = (a.to_u64() as u128 * s.w_prime.to_u64() as u128) >> W::BITS;
    let m = mask::<W>();
    let r = ((a.to_u64() as u128 * s.w.to_u64() as u128) & m).wrapping_sub((q * p.to_u64() as u128) & m) & m;
    let pv = p.to_u64() as u128;
    W::from_u128_wrapping(if r >= pv { r - pv } else { r })
}

// ---------------------------------------------------------------------------------------------------------------------
// Device per-prime constants (the kernel ABI, `crates/pc-kernels/kernels/args.cuh` `PrimeConst`).

/// The ONE per-prime constant block every device kernel reads, for every ModMul strategy: one 64-byte entry per
/// prime id per word size, residues in normal form under every strategy (Montgomery kernels
/// convert internally with `mont_r2`). The per-strategy consts above stay the host reference implementations; this
/// block carries what each of them needs plus `mu` for the lazy double-word accumulators (bconv, ks, cts, pt_dot).
///
/// | off | field | value |
/// |---|---|---|
/// | 0 | p | the prime (< 2^(BITS-1); Barrett32 at u32: < 2^30) |
/// | 8 | mu_lo | low 64 bits of floor(2^(2 BITS)/p) (u16: 2^32/p, u32: 2^64/p, u64: floor((2^128-1)/p)) |
/// | 16 | mu_hi | high 64 bits of the same (u64 only, else 0) |
/// | 24 | neal_mu | floor(2^(2 bits+1)/p) (FIDESlib mu_new, NEIL) |
/// | 32 | bits | bit_width(p) |
/// | 36 | word_bits | 16 / 32 / 64 (the word the entry was built for; kernels do not read it, checks do) |
/// | 40 | mont_pinv | -p^-1 mod 2^BITS |
/// | 48 | mont_r2 | 2^(2 BITS) mod p |
/// | 56 | split | u32: `2^32 mod p` (low word) and its Shoup companion `floor((2^32 mod p) 2^32 / p)` (high word), the
/// double-word reduction as `hi 2^32 + lo` in 32-bit products; other words 0 |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DevPrimeConst {
    pub p: u64,
    pub mu_lo: u64,
    pub mu_hi: u64,
    pub neal_mu: u64,
    pub bits: u32,
    pub word_bits: u32,
    pub mont_pinv: u64,
    pub mont_r2: u64,
    pub split: u64,
}

// Layout frozen against args.cuh (`static_assert(sizeof(PrimeConst) == 64)` there; offsets per the table above).
const _: () = {
    use std::mem::{align_of, offset_of, size_of};
    assert!(size_of::<DevPrimeConst>() == 64);
    assert!(align_of::<DevPrimeConst>() == 8);
    assert!(offset_of!(DevPrimeConst, p) == 0);
    assert!(offset_of!(DevPrimeConst, mu_lo) == 8);
    assert!(offset_of!(DevPrimeConst, mu_hi) == 16);
    assert!(offset_of!(DevPrimeConst, neal_mu) == 24);
    assert!(offset_of!(DevPrimeConst, bits) == 32);
    assert!(offset_of!(DevPrimeConst, word_bits) == 36);
    assert!(offset_of!(DevPrimeConst, mont_pinv) == 40);
    assert!(offset_of!(DevPrimeConst, mont_r2) == 48);
    assert!(offset_of!(DevPrimeConst, split) == 56);
};

impl DevPrimeConst {
    /// Constants of `p` for words of `word_bits` (16, 32 or 64). Setup-time builder over primes the chain search
    /// produced: panics on an invalid `(p, word_bits)`; use `try_new` where the input is not already validated.
    pub fn new(p: u64, word_bits: u32) -> Self {
        match Self::try_new(p, word_bits) {
            Ok(c) => c,
            Err(e) => panic!("{e}"),
        }
    }

    /// As `new`, with the range check as an error: `word_bits` in {16, 32, 64}, `p` odd, `2 < p < 2^(word_bits-1)`
    /// (u64: `p < 2^63`).
    pub fn try_new(p: u64, word_bits: u32) -> Result<Self> {
        if !matches!(word_bits, 16 | 32 | 64) {
            return Err(Error::Params(format!("DevPrimeConst: word_bits {word_bits} (expected 16, 32 or 64)")));
        }
        let lim = if word_bits == 64 { 63 } else { word_bits - 1 };
        if p <= 2 || p & 1 == 0 || p >= 1u64 << lim {
            return Err(Error::Params(format!("DevPrimeConst: prime {p} outside (2, 2^{lim}) or even (word {word_bits})")));
        }
        let bits = 64 - p.leading_zeros();
        let (mu_lo, mu_hi) = match word_bits {
            16 => ((1u64 << 32) / p, 0),
            32 => (((1u128 << 64) / p as u128) as u64, 0),
            _ => {
                let m = u128::MAX / p as u128; // == floor(2^128 / p) for odd p > 1
                (m as u64, (m >> 64) as u64)
            }
        };
        let neal_mu = ((1u128 << (2 * bits + 1)) / p as u128) as u64;
        // p^-1 mod 2^64 by Newton (each step doubles the correct low bits), then -p^-1 mod 2^word_bits
        let mut inv = p;
        for _ in 0..6 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(p.wrapping_mul(inv)));
        }
        let mask = if word_bits == 64 { u64::MAX } else { (1u64 << word_bits) - 1 };
        let mont_pinv = inv.wrapping_neg() & mask;
        let mont_r2 = if word_bits == 64 {
            ((u128::MAX % p as u128 + 1) % p as u128) as u64
        } else {
            ((1u128 << (2 * word_bits)) % p as u128) as u64
        };
        let split = if word_bits == 32 {
            let r = (1u64 << 32) % p;
            r | ((r << 32) / p) << 32
        } else {
            0
        };
        Ok(DevPrimeConst { p, mu_lo, mu_hi, neal_mu, bits, word_bits, mont_pinv, mont_r2, split })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn invmod_any_modulus() {
        use super::{invmod, mulmod};
        for &p in &[998244353u64, (1u64 << 31) - 1, 268369921] {
            for a in [1u64, 2, 3, 12345, p - 1] {
                assert_eq!(mulmod(a, invmod(a, p), p), 1, "prime {p}, a {a}");
            }
            assert_eq!(invmod(0, p), 0);
        }
        let m = 1u64 << 14;
        for a in [1u64, 3, 5, 12345, 16383] {
            assert_eq!(a * invmod(a, m) % m, 1, "composite {m}, a {a}");
        }
        assert_eq!(invmod(2, m), 0);
    }

    use super::*;
    #[test]
    fn small_cases() {
        let p = 268042241u32;
        let b = Barrett64::prime_const(p).unwrap();
        assert_eq!(Barrett64::mul(p - 1, p - 1, &b), 1);
        let m = <Montgomery as ModMul<u32>>::prime_const(p).unwrap();
        let x = Montgomery::to_repr(12345u32, &m);
        let y = Montgomery::to_repr(p - 2, &m);
        assert_eq!(Montgomery::from_repr(Montgomery::mul(x, y, &m), &m) as u64, mulmod(12345, (p - 2) as u64, p as u64));
        let s = shoup(5274u32, p);
        assert_eq!(mul_shoup(u32::MAX, &s, p) as u64, mulmod(u32::MAX as u64, 5274, p as u64));
        assert!(Barrett32::prime_const(1u32 << 30 | 1).is_err());
    }

    #[test]
    fn dev_prime_const_fields() {
        let p = 268_042_241u64; // 28-bit NTT prime
        let c = DevPrimeConst::new(p, 32);
        assert_eq!(c.bits, 28);
        assert_eq!(c.word_bits, 32);
        assert_eq!(c.mu_lo, ((1u128 << 64) / p as u128) as u64);
        assert_eq!(c.mu_hi, 0);
        // the first 16 bytes agree with FxPrime / Barrett64Const mu
        assert_eq!(c.mu_lo, Barrett64::prime_const(p as u32).unwrap().mu);
        assert_eq!(c.neal_mu as u32, Barrett32::prime_const(p as u32).unwrap().mu);
        let m = <Montgomery as ModMul<u32>>::prime_const(p as u32).unwrap();
        assert_eq!(c.mont_pinv as u32, m.p_neg_inv);
        assert_eq!(c.mont_r2 as u32, m.r2);
        let q = (1u64 << 60) - 93; // any odd < 2^62
        let d = DevPrimeConst::new(q, 64);
        let mu = ((d.mu_hi as u128) << 64) | d.mu_lo as u128;
        assert_eq!(mu, u128::MAX / q as u128);
        assert_eq!(q.wrapping_mul(d.mont_pinv.wrapping_neg()), 1);
        let u = DevPrimeConst::new(32749, 16);
        assert_eq!(u.mu_lo, (1u64 << 32) / 32749);
        assert!(DevPrimeConst::try_new(1 << 31 | 1, 32).is_err());
        assert!(DevPrimeConst::try_new(268_042_241, 24).is_err());
        assert!(DevPrimeConst::try_new(1 << 20, 32).is_err());
    }
}
