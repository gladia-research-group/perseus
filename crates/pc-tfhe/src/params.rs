//! The switch's parameters.
//!
//! The TFHE side runs at the CKKS chain's prime p0 (31 bits, NTT-friendly up to degree 2^16): the forward direction
//! is then one exact rescale by p1 and the return path is mod q0 = p0 p1 with the keys carrying the p1 (`switch`).
//! Security, against the HE-standard tables (128-bit classical, ternary secret, sigma 3.19) unless noted:
//! - ring switch key / extracted LWE: secret s' of dimension 2048 (ring N, s'(X^S)), modulus p0 (31 bits): the table
//!   allows 54 bits at 2048;
//! - GLWE / bootstrapping key: degree 2048, k = 1, q_br ~ 2^50: within the 54-bit bound;
//! - repack key: ring 2^16, q0 ~ 2^56, secret s: the CKKS chain's own instance at a smaller modulus;
//! - binary LWE key, n = 920, noise sigma = 2^-17 q: the regime of the TFHE-rs default sets (n 742-918 at relative
//!   noise ~2^-17 for 128 bits). Not run through the lattice estimator here (no Sage on the box): re-check before
//!   publication.
//! Pegasus-style switching is not IND-CPA-D secure (ePrint 2026/285): decryptions must not go back to the server.
use crate::modq::prime_below;
use crate::Result;

#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    /// Binary LWE key dimension (the blind rotation's length) and its noise (absolute, at p0).
    pub n_lwe: usize,
    pub lwe_sigma_rel: f64,
    /// GLWE (k = 1) degree and prime of the blind rotation, and the bootstrapping key's noise.
    pub n_br: usize,
    pub q_br: u64,
    pub key_sigma: f64,
    /// Gadgets as (base log, levels): blind rotation (approximate, over q_br), LWE key switch (approximate, over p0),
    /// ring switch (approximate, over p0), repack (exact, over p0).
    pub pbs: (u32, usize),
    pub ks: (u32, usize),
    pub rs: (u32, usize),
    pub rp: (u32, usize),
    /// Ring-switch factor S: the extracted LWEs have dimension N / S.
    pub rs_factor: usize,
    /// Output scale exponent: a LUT value v lands as v p0 / 2^out_cf (the finish's correction factor).
    pub out_cf: u32,
}

impl Params {
    /// The default set for a CKKS ring of degree `n_ckks` (2^16 in production; tests use 4096 with S = 2).
    pub fn new(n_ckks: usize) -> Result<Params> {
        let n_br = 2048;
        Ok(Params {
            n_lwe: 920,
            lwe_sigma_rel: (-17f64).exp2(),
            n_br,
            q_br: prime_below(50, 2 * n_br as u64)?,
            key_sigma: 3.19,
            pbs: (12, 3),
            ks: (2, 8),
            rs: (8, 3),
            rp: (16, 2),
            rs_factor: n_ckks / 2048,
            out_cf: 6,
        })
    }
}
