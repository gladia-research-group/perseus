//! CKKS <-> TFHE scheme switching, host reference (the GPU kernels, pc-gpu `tfhe`, are gated against it word for
//! word): lookup tables on CKKS slots, for the decisions polynomials do badly (signs, comparisons, steps).
//!
//! A CKKS ciphertext at the StC-first split (values in the coefficients, mod q0 = p0 p1) goes out as LWEs, through a
//! programmable bootstrap and back as a split ciphertext the bootstrap's finish (ModRaise -> CtS -> EvalMod) turns
//! into a fresh CKKS ciphertext:
//!
//!   rescale by p1 (mod p0) -> ring switch s -> s'(X^S) -> extract LWE(dim N/S) -> key switch to the binary LWE key
//!   -> blind rotation (multi-bit groups, GLWE of degree 2048 over two 26-bit primes) -> sample extract -> mod switch
//!   to p0 -> repack (a GEMM in R_2048 mod p0 P, rescale by P, one ring key switch to s mod q0).
//!
//! Lookup tables: a general LUT reads [-1, 1) (`pbs::test_vector`); a sign reads (-2, 2) (`pbs::sign01`: no padding
//! half). Outputs are arcsine-warped (`pbs::warp`) so the finish's sine returns them exactly at q0 / 2^out_cf.
//! Precision: the input carries the switch's noise (rms ~8e-3 of a general LUT's range, dominated by the mod switch to
//! 2 n_br); the output's is ~1e-4. Signs and steps are exact outside that band; smooth functions keep ~7 bits.
//! Security: `params` (lattice-estimator per TFHE-rs's rule: every instance >= 138 bits, every sigma >= 4).
//!
//! IND-CPA-D: like every Pegasus-style switch (ePrint 2026/285), the switch is not secure against an adversary who sees
//! decryptions of its outputs; decrypted results must not go back to the party holding the ciphertexts.
//!
//! Phase convention everywhere: b + <a, s> (LWE), c0 + c1 s (RLWE).
pub mod gadget;
pub mod lwe;
pub mod modq;
pub mod params;
pub mod pbs;
pub mod rlwe;
pub mod switch;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("parameter error: {0}")]
    Params(String),
    #[error(transparent)]
    Core(#[from] pc_core::error::Error),
}
pub type Result<T> = std::result::Result<T, Error>;
