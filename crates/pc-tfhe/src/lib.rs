//! CKKS <-> TFHE scheme switching, host reference (the GPU kernels are gated against it word for word).
//!
//! A CKKS ciphertext at the StC-first split (values in the coefficients, mod q0 = p0 p1) goes out as LWEs, through a
//! programmable bootstrap (any lookup table on [-1, 1), sign first), and back as a split ciphertext the bootstrap's
//! finish (ModRaise -> CtS -> EvalMod) turns into a fresh CKKS ciphertext:
//!
//!   rescale by p1 (mod p0) -> ring switch s -> s'(X^S) -> extract LWE(dim N/S) -> key switch to the binary LWE key
//!   -> blind rotation at the ~50-bit prime q_br -> sample extract -> mod switch to p0 -> repack into mod q0 under s.
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
