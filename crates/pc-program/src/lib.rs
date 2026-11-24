//! pc-program: graph + plan -> `Program` (CUDA-free). `compile::compile_dir` / `compile::compile_block` resolve the
//! captured graph's names, apply the plan (refresh sites, hints, realize anchors, finishes), check every var against
//! the plan with the runtime Shadow at compile time, K-batch sibling ops and assign arena slots. The replay
//! executes `Program.ops` in order: no level bookkeeping at run time, every operand prelude is recorded.
//!
//! Feature `gpu`: `bind` (a Program on a device runtime: ciphertext slots, plaintext slots, key groups,
//! routes), `replay` (eager or one CUDA graph per bound block), `residency` (the weight ring), `kv` (KV offload).
pub mod compile;
pub mod error;
pub mod graph;
pub mod plan;
pub mod program;
#[cfg(feature = "gpu")]
pub mod bind;
#[cfg(feature = "gpu")]
pub mod kv;
#[cfg(feature = "gpu")]
pub mod replay;
#[cfg(feature = "gpu")]
pub mod replica;
#[cfg(feature = "gpu")]
pub mod residency;
#[cfg(feature = "gpu")]
pub mod schedule;

pub use compile::{compile_block, compile_dir, CompileOpts};
pub use error::{Error, Result};
pub use program::{BtsSite, Kind, Op, Program, PtRef, Var, VarId};
