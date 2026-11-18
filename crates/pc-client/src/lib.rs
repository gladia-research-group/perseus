//! pc-client: the CUDA-free client of a perseus server. It holds the secret and does everything that needs
//! it: key generation of what a plan needs (switching keys, SPRU material, PCMM / CCMM keys, a public key) with host
//! NTTs, encoding + encryption of inputs, decryption + decoding of outputs. The server runs on the key bundle and
//! ciphertexts only (`wire`; docs/WIRE.md). Depends on pc-core alone.
pub mod client;
pub mod error;
pub mod ntt;
pub mod secret;
pub mod spec;
pub mod wire;

pub use client::{Client, KeyGenStats, PreEnc};
pub use error::{Error, Result};
pub use secret::ClientSecret;
pub use spec::{KeyReq, KeyShape, KeySpec, Role, SpruReq};
pub use wire::{BundleHeader, BundleReader, CtHeader, CtWire, Entry, WireMeta};
