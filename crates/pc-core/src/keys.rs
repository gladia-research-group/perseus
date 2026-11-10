//! Secret / error samplers, key material and the KSKA seed expansion (host twin of the device expander).
//!
//! Two ChaCha users live here and must not be confused:
//! - `Prng`: the sampler DRBG (ChaCha20, keyed from OS entropy; a CSPRNG, no parity with OpenFHE's Blake2 PRNG).
//! - `kska_expand*`: the FROZEN key-switching-key `a` expander (ChaCha12 counter mode, "KSKA" domain tag, rejection
//!   sampling), value-identical to FIDESlib `KskSeedExpand.cuh::expand_coeff` and OpenFHE patch 0007.
//! - `kskb_expand*`: the FROZEN 64-bit-word lane (FIDESlib SPEC v2 `expand_coeff64`, OpenFHE patch kska-seed
//!   `aSeedWide`); valid for digit < 16 and p < 2^60 (`check_kskb`); layout at `kskb_expand`. The device twin is
//!   pc-kernels `chacha.cuh::expand_slot`; `ksk_expand_row` picks the lane by word size.
use crate::error::{Error, Result};
use std::collections::BTreeMap;

const CHACHA_CONST: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];

#[inline(always)]
fn qr(x: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    x[a] = x[a].wrapping_add(x[b]);
    x[d] = (x[d] ^ x[a]).rotate_left(16);
    x[c] = x[c].wrapping_add(x[d]);
    x[b] = (x[b] ^ x[c]).rotate_left(12);
    x[a] = x[a].wrapping_add(x[b]);
    x[d] = (x[d] ^ x[a]).rotate_left(8);
    x[c] = x[c].wrapping_add(x[d]);
    x[b] = (x[b] ^ x[c]).rotate_left(7);
}

/// One ChaCha block with caller-supplied state words 12..15 (`rounds` even: 12 for KSKA, 20 for the DRBG).
/// This is FIDESlib's `chacha_block_tail`: standard constants row, key in words 4..11, feed-forward add.
pub fn chacha_block(key: &[u32; 8], w12: u32, w13: u32, w14: u32, w15: u32, rounds: u32) -> [u32; 16] {
    let s: [u32; 16] = [
        CHACHA_CONST[0], CHACHA_CONST[1], CHACHA_CONST[2], CHACHA_CONST[3], key[0], key[1], key[2], key[3], key[4],
        key[5], key[6], key[7], w12, w13, w14, w15,
    ];
    let mut x = s;
    for _ in 0..rounds / 2 {
        qr(&mut x, 0, 4, 8, 12);
        qr(&mut x, 1, 5, 9, 13);
        qr(&mut x, 2, 6, 10, 14);
        qr(&mut x, 3, 7, 11, 15);
        qr(&mut x, 0, 5, 10, 15);
        qr(&mut x, 1, 6, 11, 12);
        qr(&mut x, 2, 7, 8, 13);
        qr(&mut x, 3, 4, 9, 14);
    }
    for i in 0..16 {
        x[i] = x[i].wrapping_add(s[i]);
    }
    x
}

// ------------------------------------------------------------------------------------------------------------------
// Sampler DRBG
// ------------------------------------------------------------------------------------------------------------------

/// Domain word of the sampler DRBG ("DRBG"), so its keystream never coincides with a KSKA/KSKB stream, even under
/// the same 256-bit key.
const DRBG_TAG: u32 = 0x4742_5244;

/// ChaCha20 DRBG: key from OS entropy (or an explicit seed for reproducible tests / derived per-ciphertext streams),
/// 64-bit block counter in words 12..13, word 14 = stream id, word 15 = "DRBG". Owned by the caller (no global PRNG).
pub struct Prng {
    key: [u32; 8],
    stream: u32,
    ctr: u64,
    buf: [u32; 16],
    pos: usize,
}

impl Prng {
    /// Keyed from the OS CSPRNG (`getrandom`).
    pub fn from_os() -> Result<Self> {
        let mut b = [0u8; 32];
        getrandom::fill(&mut b).map_err(|e| Error::Keys(format!("getrandom failed: {e}")))?;
        let mut key = [0u32; 8];
        for (i, c) in b.chunks_exact(4).enumerate() {
            key[i] = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
        Ok(Self::from_seed(key, 0))
    }

    /// Deterministic stream (tests; derived per-ciphertext / per-key streams).
    pub fn from_seed(key: [u32; 8], stream: u32) -> Self {
        Prng { key, stream, ctr: 0, buf: [0; 16], pos: 16 }
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.pos == 16 {
            self.buf = chacha_block(&self.key, self.ctr as u32, (self.ctr >> 32) as u32, self.stream, DRBG_TAG, 20);
            self.ctr = self.ctr.wrapping_add(1);
            self.pos = 0;
        }
        let v = self.buf[self.pos];
        self.pos += 1;
        v
    }

    pub fn next_u64(&mut self) -> u64 {
        let lo = self.next_u32() as u64;
        lo | (self.next_u32() as u64) << 32
    }

    /// Uniform in [0, q), exact: reject the 2^64 mod q smallest words so the accepted range is a multiple of q.
    pub fn uniform_below(&mut self, q: u64) -> u64 {
        assert!(q > 0, "uniform_below: q = 0");
        let rem = (u64::MAX % q + 1) % q; // 2^64 mod q
        loop {
            let v = self.next_u64();
            if v >= rem {
                return v % q;
            }
        }
    }

    /// Uniform double in [0, 1) with 53 random bits.
    pub fn uniform_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// A fresh 256-bit key-switching-key seed drawn from this stream.
    pub fn ksk_seed(&mut self) -> KskSeed {
        let mut s = [0u32; 8];
        for w in s.iter_mut() {
            *w = self.next_u32();
        }
        KskSeed(s)
    }
}

// ------------------------------------------------------------------------------------------------------------------
// Samplers
// ------------------------------------------------------------------------------------------------------------------

/// Uniform ternary secret: each coefficient uniform in {-1, 0, 1} (OpenFHE UNIFORM_TERNARY).
pub fn sample_ternary(prng: &mut Prng, n: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let mut w = prng.next_u32();
        // 16 two-bit draws per word, value 3 rejected
        for _ in 0..16 {
            let v = w & 3;
            w >>= 2;
            if v < 3 && out.len() < n {
                out.push(v as i8 - 1);
            }
        }
    }
    out
}

/// Sparse ternary secret of Hamming weight exactly `h`: h distinct uniform positions, each +-1 with probability 1/2.
pub fn sample_sparse_ternary(prng: &mut Prng, n: usize, h: usize) -> Result<Vec<i8>> {
    if h > n {
        return Err(Error::Keys(format!("sparse secret: hamming weight h={h} > ring dimension {n}")));
    }
    let mut out = vec![0i8; n];
    let mut placed = 0;
    while placed < h {
        let i = prng.uniform_below(n as u64) as usize;
        if out[i] == 0 {
            out[i] = if prng.next_u32() & 1 == 0 { 1 } else { -1 };
            placed += 1;
        }
    }
    Ok(out)
}

/// Uniform vector mod q.
pub fn sample_uniform_mod(prng: &mut Prng, n: usize, q: u64) -> Vec<u64> {
    (0..n).map(|_| prng.uniform_below(q)).collect()
}

/// Discrete Gaussian over Z (centered, width sigma) by inversion of a cumulative table on |x| <= ceil(14 sigma)
/// (the cut tail has mass < 2^-140; table resolution is the 53-bit uniform). No parity with OpenFHE's sampler.
pub struct DiscreteGaussian {
    sigma: f64,
    /// cdf[k] = P(|X| <= k), k = 0..=bound
    cdf: Vec<f64>,
}

impl DiscreteGaussian {
    /// OpenFHE's default error width.
    pub const SIGMA: f64 = 3.19;

    pub fn new(sigma: f64) -> Result<Self> {
        if !(sigma > 0.0 && sigma < 1e6) {
            return Err(Error::Keys(format!("gaussian sigma={sigma} out of range (0, 1e6)")));
        }
        let bound = (sigma * 14.0).ceil() as usize;
        let w: Vec<f64> = (0..=bound)
            .map(|k| {
                let p = (-((k * k) as f64) / (2.0 * sigma * sigma)).exp();
                if k == 0 {
                    p
                } else {
                    2.0 * p
                }
            })
            .collect();
        let tot: f64 = w.iter().sum();
        let mut acc = 0.0;
        let cdf = w
            .iter()
            .map(|x| {
                acc += x / tot;
                acc
            })
            .collect();
        Ok(DiscreteGaussian { sigma, cdf })
    }

    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    pub fn sample(&self, prng: &mut Prng) -> i64 {
        let u = prng.uniform_f64();
        let k = self.cdf.partition_point(|&c| c <= u).min(self.cdf.len() - 1) as i64;
        if k == 0 {
            0
        } else if prng.next_u32() & 1 == 0 {
            k
        } else {
            -k
        }
    }

    pub fn sample_vec(&self, prng: &mut Prng, n: usize) -> Vec<i64> {
        (0..n).map(|_| self.sample(prng)).collect()
    }
}

// ------------------------------------------------------------------------------------------------------------------
// KSKA / KSKB seed expansion (FROZEN spec)
// ------------------------------------------------------------------------------------------------------------------

/// 256-bit per-key seed of a key-switching key's `a` (little-endian u32 words, as serialized by OpenFHE patch 0007).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KskSeed(pub [u32; 8]);

/// "KSKA": domain word s\[15\] of the 32-bit lane.
pub const KSKA_DOMAIN: u32 = 0x4b53_4b41;
/// "KSKB": domain word s\[15\] of the 64-bit lane.
pub const KSKB_DOMAIN: u32 = 0x4b53_4b42;
/// Attempts per coefficient before the (probability < 2^-64) fallback `v % p`.
pub const KSK_T_MAX: u32 = 16;
/// ChaCha12 (FIDESlib kRounds).
pub const KSK_ROUNDS: u32 = 12;

/// One coefficient of `a` for (seed, hybrid digit, limb modulus VALUE p, eval-form slot) at ring dimension `n`.
/// Attempt t reads word (slot & 15) of block (slot >> 4) + t * (n >> 4); accept v < floor(2^32 / p) * p; value v % p.
/// `a` is sampled directly in evaluation (bit-reversed NTT) form, so `slot` is the eval-form index.
pub fn kska_expand(seed: &KskSeed, digit: u32, modulus: u32, slot: u32, n: usize) -> u32 {
    let n16 = (n as u32) >> 4;
    let b0 = slot >> 4;
    let w = (slot & 15) as usize;
    let mp = (u32::MAX / modulus) * modulus;
    let mut v = 0;
    for t in 0..KSK_T_MAX {
        v = chacha_block(&seed.0, b0.wrapping_add(t.wrapping_mul(n16)), digit, modulus, KSKA_DOMAIN, KSK_ROUNDS)[w];
        if v < mp {
            break;
        }
    }
    v % modulus
}

/// A whole limb row of `a` (`out.len()` = n values): one base block per 16 slots, escalation blocks only for
/// rejected words. Same values as `kska_expand` slot by slot.
pub fn kska_expand_row(seed: &KskSeed, digit: u32, modulus: u32, out: &mut [u32]) {
    let n16 = (out.len() as u32) >> 4;
    let mp = (u32::MAX / modulus) * modulus;
    for (b0, chunk) in out.chunks_exact_mut(16).enumerate() {
        let b0 = b0 as u32;
        let base = chacha_block(&seed.0, b0, digit, modulus, KSKA_DOMAIN, KSK_ROUNDS);
        for (w, o) in chunk.iter_mut().enumerate() {
            let mut v = base[w];
            let mut t = 1;
            while v >= mp && t < KSK_T_MAX {
                v = chacha_block(&seed.0, b0.wrapping_add(t * n16), digit, modulus, KSKA_DOMAIN, KSK_ROUNDS)[w];
                t += 1;
            }
            *o = v % modulus;
        }
    }
}

/// 64-bit lane (primes up to 2^60): s\[13\] = (digit & 0xF) | (p >> 32) << 4, s\[14\] = low word of p, s\[15\] = "KSKB";
/// 8 slots per block (two words per coefficient, little-endian pair), stride n >> 3, accept v < floor((2^64-1)/p) p.
/// FROZEN = FIDESlib `expand_coeff64`; pinned by `tests::kskb_frozen_vector` (no oracle dump of u64 keys exists).
pub fn kskb_expand(seed: &KskSeed, digit: u32, modulus: u64, slot: u32, n: usize) -> u64 {
    let n8 = (n as u32) >> 3;
    let b0 = slot >> 3;
    let w = (slot & 7) as usize;
    let w13 = (digit & 0xF) | (((modulus >> 32) as u32) << 4);
    let mp = (u64::MAX / modulus) * modulus;
    let mut v = 0;
    for t in 0..KSK_T_MAX {
        let o = chacha_block(&seed.0, b0.wrapping_add(t.wrapping_mul(n8)), w13, modulus as u32, KSKB_DOMAIN, KSK_ROUNDS);
        v = o[2 * w] as u64 | (o[2 * w + 1] as u64) << 32;
        if v < mp {
            break;
        }
    }
    v % modulus
}

/// Domain checks of the KSKB lane: the digit index fits s\[13\]'s low nibble and hi32(p) fits its 28 high bits.
pub fn check_kskb(dnum: usize, moduli: &[u64]) -> Result<()> {
    if dnum > 16 {
        return Err(Error::Keys(format!("KSKB lane packs the digit into 4 bits: dnum {dnum} > 16")));
    }
    if let Some(p) = moduli.iter().find(|&&p| p >= 1u64 << 60 || p < 3 || p % 2 == 0) {
        return Err(Error::Keys(format!("KSKB lane needs odd moduli < 2^60, got {p}")));
    }
    Ok(())
}

/// A whole limb row of the KSKB lane (`out.len()` = n): one base block per 8 slots, escalation only for rejected
/// coefficients. Same values as `kskb_expand` slot by slot.
pub fn kskb_expand_row(seed: &KskSeed, digit: u32, modulus: u64, out: &mut [u64]) {
    let n8 = (out.len() as u32) >> 3;
    let w13 = (digit & 0xF) | (((modulus >> 32) as u32) << 4);
    let mp = (u64::MAX / modulus) * modulus;
    let pair = |o: &[u32; 16], w: usize| o[2 * w] as u64 | (o[2 * w + 1] as u64) << 32;
    for (b0, chunk) in out.chunks_exact_mut(8).enumerate() {
        let b0 = b0 as u32;
        let base = chacha_block(&seed.0, b0, w13, modulus as u32, KSKB_DOMAIN, KSK_ROUNDS);
        for (w, o) in chunk.iter_mut().enumerate() {
            let mut v = pair(&base, w);
            let mut t = 1;
            while v >= mp && t < KSK_T_MAX {
                v = pair(&chacha_block(&seed.0, b0.wrapping_add(t * n8), w13, modulus as u32, KSKB_DOMAIN, KSK_ROUNDS), w);
                t += 1;
            }
            *o = v % modulus;
        }
    }
}

/// The `a` row of (seed, digit, p) for a context with `word_bits`-bit words: KSKA at 32, KSKB at 64 (both frozen).
pub fn ksk_expand_row(seed: &KskSeed, digit: u32, modulus: u64, word_bits: u32, out: &mut [u64]) -> Result<()> {
    match word_bits {
        32 => {
            let p = u32::try_from(modulus).map_err(|_| Error::Keys(format!("KSKA lane: modulus {modulus} >= 2^32")))?;
            let mut v = vec![0u32; out.len()];
            kska_expand_row(seed, digit, p, &mut v);
            out.iter_mut().zip(v).for_each(|(o, x)| *o = x as u64);
            Ok(())
        }
        64 => {
            check_kskb(digit as usize + 1, &[modulus])?;
            kskb_expand_row(seed, digit, modulus, out);
            Ok(())
        }
        w => Err(Error::Keys(format!("no frozen KSK expander for {w}-bit words (u32 = KSKA, u64 = KSKB)"))),
    }
}

// ------------------------------------------------------------------------------------------------------------------
// Key material
// ------------------------------------------------------------------------------------------------------------------

/// The secret: dense uniform-ternary `s` (main key) and the sparse encapsulation key (HWT h = 32).
#[derive(Clone)]
pub struct SecretKey {
    pub s: Vec<i8>,
    pub s_sparse: Vec<i8>,
}

impl std::fmt::Debug for SecretKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SecretKey(..)")
    }
}

impl SecretKey {
    /// Hamming weight of the sparse encapsulation secret (FIDESlib ENCAPS route).
    pub const SPARSE_H: usize = 32;

    pub fn sample(prng: &mut Prng, n: usize) -> Result<Self> {
        Ok(SecretKey { s: sample_ternary(prng, n), s_sparse: sample_sparse_ternary(prng, n, Self::SPARSE_H)? })
    }
}

/// A hybrid key-switching key in seeded form: `a` is regenerated from `seed` (KSKA), only `b` is stored.
/// `b_rows` layout: \[digit\]\[limb\]\[n\], limbs in `moduli` order (Q primes by prime id, then P).
/// Relation (OpenFHE KeySwitchHYBRID::KeySwitchGenInternal, eval form, limb i of digit j):
///   b = -a * s_new + e + [i in digit j] * (P mod q_i) * s_old
#[derive(Clone, Debug)]
pub struct KskMaterial {
    pub seed: KskSeed,
    pub dnum: usize,
    pub n: usize,
    pub moduli: Vec<u32>,
    pub b_rows: Vec<u32>,
}

impl KskMaterial {
    pub fn new(seed: KskSeed, dnum: usize, n: usize, moduli: Vec<u32>, b_rows: Vec<u32>) -> Result<Self> {
        if b_rows.len() != dnum * moduli.len() * n {
            return Err(Error::Keys(format!(
                "ksk b_rows has {} words, expected dnum {dnum} x limbs {} x n {n}",
                b_rows.len(),
                moduli.len()
            )));
        }
        if n % 16 != 0 || n == 0 {
            return Err(Error::Keys(format!("ksk ring dimension n={n} is not a positive multiple of 16")));
        }
        Ok(KskMaterial { seed, dnum, n, moduli, b_rows })
    }

    pub fn b_row(&self, digit: usize, limb: usize) -> &[u32] {
        let o = (digit * self.moduli.len() + limb) * self.n;
        &self.b_rows[o..o + self.n]
    }

    /// The regenerated `a` row of (digit, limb).
    pub fn a_row(&self, digit: usize, limb: usize) -> Vec<u32> {
        let mut v = vec![0u32; self.n];
        kska_expand_row(&self.seed, digit as u32, self.moduli[limb], &mut v);
        v
    }
}

/// The bootstrap's key-switching keys, one named field per role (no index-slot conventions).
/// `rots` is keyed by the Galois element g of the automorphism X -> X^g (OpenFHE's key-map index; `galois` module).
/// FIDESlib keeps the two encapsulation keys in the automorphism map under the indices 2N - 2
/// (`sparse_atob`, dense -> sparse) and 2N - 4 (`sparse_btoa`, sparse -> dense), as loaded by FIDESlib
/// `RawCiphertext.cu` (the comment in `BootstrapPrecomputation.cuh` has the two indices swapped).
#[derive(Clone, Debug)]
pub struct BootstrapKeys {
    pub enc_dense_to_sparse: KskMaterial,
    pub enc_sparse_to_dense: KskMaterial,
    /// conjugation (g = 2N - 1)
    pub conj: KskMaterial,
    pub rots: BTreeMap<u64, KskMaterial>,
}
