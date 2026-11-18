//! pc-gpu: the device runtime of perseus.
//!
//! Object graph (all setup-time, then `Arena::freeze`):
//! `Device` (pc-cuda guard) -> `MemPlan::build` (sizes) -> `MemPlan::allocate` (ONE arena) ->
//! `Executor::new` (lanes, events, KS/bconv/scratch workspaces) + `Context::new` (chain, scale tables, device
//! constant tables, kernel table) + `CtPool::new` (eager ciphertext slots) -> `arena.freeze()`.
//! Ops are free functions `(ctx, stream or executor fields, ciphertexts...)`; nothing is global.
pub mod error;    // Error (typed, names the knob)
pub mod context;  // Context<W, M> (Chain + ScaleTables + device constants + KernelTable), DevWord, DevStrategy
pub mod executor; // Executor { compute, copy, side, events, ks_work, bconv_work, scratch }
pub mod mem;      // MemCfg, MemPlan (one arena), CtPool + PoolCt (bounded, size-classed, refcounted)
pub mod poly;     // Layout, RowSet/PolyRef builders, CtBuf
pub mod ct;       // Ct { meta: CtMeta, buf: CtBuf }
pub mod ntt;      // RowTables / RowList, forward / inverse over a CtBuf
pub mod bconv;    // BconvPlan (job tables, word-generic)
pub mod keys;     // device key material (packed b + seed), KskSource
pub mod keygen;   // device key generation
pub mod encrypt;  // public/secret-key encryption on the device
pub mod decrypt;  // decryption (+ host decode hand-off)
pub mod ops;      // leaf ops (add/sub/negate/scalar/pt mult, tensor) executing pc-shadow Recipes
pub mod rescale;  // Rescaler (composite one-pass), DropTo
pub mod pt_store; // plaintext cache + staging worker (PtStorePolicy)
pub mod encode_dev; // device encode (Encoder seam: host FFT / device FFT / coefficient rows)
pub mod ks;       // KeySwitch (ModUp, KS dot, ModDown)
pub mod rotate;   // Rotator (plain / hoisted / rotate-and-sum)
pub mod relin;    // relinearization
pub mod limbs;    // export/import limbs, merge_monomial
pub mod bts;      // bootstrap routes (BtsRoute)
pub mod matmul;   // PCMM / CCMM (cuBLASLt)
pub mod tfhe;     // CKKS <-> TFHE scheme switch (LUTs by programmable bootstrapping at the StC-first split)
pub mod graph;    // CUDA-graph capture / replay of bootstrap batches

pub use context::{Context, DevStrategy, DevWord};
pub use ct::Ct;
pub use error::{Error, Result};
pub use executor::Executor;
pub use mem::{CtPool, MemCfg, MemPlan, PoolClass, PoolCt};
pub use poly::{CtBuf, Layout};
