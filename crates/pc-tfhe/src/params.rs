//! The switch's parameters.
//!
//! The TFHE side runs at the CKKS chain's prime p0 (28 bits, NTT-friendly up to degree 2^16): the forward direction
//! is then one exact rescale by p1 and the return path is mod q0 = p0 p1 with the keys carrying the p1 (`switch`).
//! Security follows TFHE-rs's rule (its ci/lattice_estimator.sage): lattice-estimator, MATZOV cost model, the minimum
//! of primal uSVP and dual hybrid >= 128 bits (132 to pass without a warning), every noise sigma >= 4. Each key's
//! sigma is set by its LWE instance:
//! - binary LWE key (LWE key switch), n = 921 (a multiple of the PBS group), sigma 2^-17 p0: 154 bits (measured);
//! - ring switch key / extracted LWE: ternary s' of dimension 2048 (ring N, s'(X^S)) mod p0 ~ 2^28, sigma 4: 252 at 3.19;
//! - GLWE / bootstrapping key: degree 2048, k = 1, q_br = q_a q_b ~ 2^52 (two 26-bit NTT primes, the device's u32
//!   RNS), sigma 18.5 = TFHE-rs's q / sigma ~2^47.8 (at sigma 3.19 it measured 131: a warning). PBS output noise by
//!   the model (n/G)(2^G - 1) 2 * 2 l N (B^2 / 12) sigma^2 + (n/G) 2 N (2/3) (q / B^l)^2 / 12 (it matched the measured
//!   rms to 0.1 bit at sigma 3.19): ~2^-14.2 q_br at gadget 2^23;
//! - repack columns: ternary s_x of degree 2048 mod p0 P ~ 2^53, sigma 1024 (the packing's noise is divided by P;
//!   at 3.19 it measured 128.9);
//! - repack ring key: ring 2^16 mod q0 ~ 2^56, secret s, sigma 4: the CKKS chain's own instance at a smaller modulus.
//! Pegasus-style switching is not IND-CPA-D secure (ePrint 2026/285): decryptions must not go back to the server.
use crate::modq::prime_below;
use crate::Result;

#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    /// Binary LWE key dimension (the blind rotation's length, a multiple of `pbs_group`) and its noise (relative to p0).
    pub n_lwe: usize,
    pub lwe_sigma_rel: f64,
    /// GLWE (k = 1) degree, the blind rotation's RNS primes and their product.
    pub n_br: usize,
    pub br_primes: (u64, u64),
    pub q_br: u64,
    /// Key noise: the ring-switch and repack ring keys, the bootstrapping key, the repack columns (module doc).
    pub key_sigma: f64,
    pub bsk_sigma: f64,
    pub rp_sigma: f64,
    /// Gadgets as (base log, levels): blind rotation (approximate, over q_br), LWE key switch (approximate, over p0),
    /// ring switch (approximate, over p0), repack (exact, over p0).
    pub pbs: (u32, usize),
    /// Key bits per blind-rotation step (the multi-bit unrolling: 2^group - 1 GGSWs per group): 3 (n_lwe 921) is 7%
    /// faster than 2 and 18% faster than 4 on the device (NTT(X^e - 1)'s table in shared memory; from L2 its gathers
    /// made 3 slower than 2).
    pub pbs_group: usize,
    pub ks: (u32, usize),
    pub rs: (u32, usize),
    pub rp: (u32, usize),
    /// The repack's auxiliary prime P (p0 P <= 2^54; NTT-friendly at n_br for the host's key generation).
    pub rp_aux: u64,
    /// Ring-switch factor S: the extracted LWEs have dimension N / S.
    pub rs_factor: usize,
    /// Output scale exponent: the finish reads a LUT value v as v / 2^out_cf of q0 (at correction factor cf it returns
    /// v / 2^(out_cf - cf), folded into the consumer). Its EvalMod is a sine, so the LUT stores the arcsine-warped
    /// phase (`pbs::warp`); |v| <= 1 needs out_cf >= 3.
    pub out_cf: u32,
}

impl Params {
    /// The default set for a CKKS ring of degree `n_ckks` (2^16 in production; tests use 4096 with S = 2).
    pub fn new(n_ckks: usize) -> Result<Params> {
        let n_br = 2048;
        let m = 2 * n_br as u64;
        let qa = prime_below(26, m)?;
        let mut qb = qa - m;
        while !crate::modq::is_prime(qb) {
            qb -= m;
        }
        Ok(Params {
            n_lwe: 921,
            lwe_sigma_rel: (-17f64).exp2(),
            n_br,
            br_primes: (qa, qb),
            q_br: qa * qb,
            key_sigma: 4.0,
            bsk_sigma: 18.5,
            rp_sigma: 1024.0,
            pbs: (23, 1),
            pbs_group: 3,
            ks: (2, 8),
            rs: (8, 3),
            rp: (16, 2),
            rp_aux: prime_below(25, m)?,
            rs_factor: n_ckks / 2048,
            out_cf: 3,
        })
    }
}
