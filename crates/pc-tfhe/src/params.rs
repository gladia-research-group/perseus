//! The switch's parameters.
//!
//! The TFHE side runs at the CKKS chain's prime p0 (28 bits, NTT-friendly up to degree 2^16): the forward direction
//! is then one exact rescale by p1 and the return path is mod q0 = p0 p1 with the keys carrying the p1 (`switch`).
//! Security follows TFHE-rs's rule (its ci/lattice_estimator.sage): lattice-estimator, MATZOV cost model, the minimum
//! of primal uSVP and dual hybrid >= 128 bits (132 to pass without a warning), every noise sigma >= 4. Each key's
//! sigma is set by its LWE instance (lattice-estimator run 2026-10-26, bits):
//! - binary LWE key (LWE key switch), n = 795 (a multiple of the PBS group), sigma 2^-17 p0: 133.2 bits. The switch's
//!   input noise is the mod switch to 2 n_br (mean-compensated, `gadget::ms_body_mean`) plus the LWE key switch's
//!   2048 x levels digit x key-noise terms (binary digits: E d^2 = 1/2);
//!   precise set `Params::with_lwe(990, -21)` (n a multiple of 3 and 5);
//! - ring switch gadget (4, 6): its key-noise term grows with sqrt(N) of the CKKS ring;
//! - ring switch key / extracted LWE: ternary s' of dimension 2048 (ring N, s'(X^S)) mod p0 ~ 2^28, sigma 4: 254.6;
//! - GLWE / bootstrapping key: degree 2048, k = 1, q_br = q_a q_b ~ 2^52 (two 26-bit NTT primes, the device's u32
//!   RNS), sigma 18.5 = TFHE-rs's q / sigma ~2^47.8: 138.6 (at sigma 3.19: 131.3, a warning). PBS output noise by
//!   the model (n/G)(2^G - 1) 2 * 2 l N (B^2 / 12) sigma^2 + (n/G) 2 N (2/3) (q / B^l)^2 / 12: ~2^-14.2 q_br at
//!   gadget 2^23;
//! - repack columns: ternary s_x of degree 2048 mod p0 P ~ 2^53, sigma 1024 (the packing's noise is divided by P):
//!   154.2 (at 3.19: 128.9);
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
    /// Key bits per blind-rotation step (the multi-bit unrolling: 2^group - 1 GGSWs per group); 3 needs NTT(X^e - 1)'s
    /// table in shared memory.
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
    /// The extended blind rotation's nu (`pbs::BootstrapKey::blind_rotate_ext`; 1 or 2): the mod switch to 2 nu n_br,
    /// nu times less of its noise (the input noise's largest term) for ~2.2x the bootstrap; the keys are the same.
    pub nu: usize,
    /// A second bootstrapping key grouped by `lat_group` bits (0: none; 5 at n_lwe 795) for small batches: the
    /// blind rotation's sequential steps fall from n_lwe / pbs_group to n_lwe / lat_group (31 keys per bundle,
    /// ~2.7x the key's memory; the latency path computes the bundles off the sequential path).
    pub lat_group: usize,
}

impl Params {
    /// Test-vector entries (`pbs::test_vector(tv_len, ..)`): n_br nu.
    pub fn tv_len(&self) -> usize {
        self.n_br * self.nu
    }

    /// Another binary LWE key: dimension `n_lwe` (a multiple of the PBS groups) and sigma 2^`sigma_log2` p0.
    pub fn with_lwe(mut self, n_lwe: usize, sigma_log2: f64) -> Result<Params> {
        if n_lwe % self.pbs_group != 0 || self.lat_group != 0 && n_lwe % self.lat_group != 0 {
            return Err(crate::Error::Params(format!("n_lwe {n_lwe}: not a multiple of the PBS groups {} / {}", self.pbs_group, self.lat_group)));
        }
        self.n_lwe = n_lwe;
        self.lwe_sigma_rel = sigma_log2.exp2();
        Ok(self)
    }

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
            n_lwe: 795,
            lwe_sigma_rel: (-17f64).exp2(),
            n_br,
            br_primes: (qa, qb),
            q_br: qa * qb,
            key_sigma: 4.0,
            bsk_sigma: 18.5,
            rp_sigma: 1024.0,
            pbs: (23, 1),
            pbs_group: 3,
            ks: (1, 16),
            rs: (4, 6),
            rp: (15, 2),
            rp_aux: prime_below(25, m)?,
            rs_factor: n_ckks / 2048,
            out_cf: 3,
            nu: 1,
            lat_group: 0,
        })
    }
}
