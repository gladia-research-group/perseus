//! Typed errors of the CUDA layer. Each names the call (or the knob) that failed.
use crate::ffi;
use std::ffi::CStr;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("CUDA error {code} ({name}) in {call}: {msg}")]
    Cuda { code: i32, name: String, msg: String, call: &'static str },
    #[error("GPU guard: {0}")]
    Guard(String),
    #[error("arena overflow allocating {what}: need {need} B at offset {used} of {cap} B (raise the MemPlan section)")]
    ArenaFull { what: &'static str, need: usize, used: usize, cap: usize },
    #[error("event pool exhausted: all {size} events in flight (knob: MemCfg::events)")]
    EventsExhausted { size: usize },
    #[error("launch {kernel}: argument outside the kernel's range (rc {rc})")]
    LaunchArgs { kernel: &'static str, rc: i32 },
    #[error("{what}: {len} elements requested, buffer holds {cap}")]
    Bounds { what: &'static str, len: usize, cap: usize },
    #[error("{0} is not available in this build")]
    Unavailable(&'static str),
    #[error("host device: {0}")]
    Host(&'static str),
    #[error("cuBLASLt: {0}")]
    Blas(String),
}
pub type Result<T> = std::result::Result<T, Error>;

fn err_strings(code: i32) -> (String, String) {
    // SAFETY: cudaGetErrorName/String return static NUL-terminated strings for every code
    unsafe {
        (
            CStr::from_ptr(ffi::cudaGetErrorName(code)).to_string_lossy().into_owned(),
            CStr::from_ptr(ffi::cudaGetErrorString(code)).to_string_lossy().into_owned(),
        )
    }
}

/// `cudaError_t` -> `Result`.
pub fn check(code: i32, call: &'static str) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    let (name, msg) = err_strings(code);
    Err(Error::Cuda { code, name, msg, call })
}

/// A kernel launcher's return code (pc-kernels ABI): 0 ok, < 0 an argument outside the kernel's range, > 0 the
/// `cudaError_t` of the launch.
pub fn launch_result(kernel: &'static str, rc: i32) -> Result<()> {
    if rc < 0 {
        return Err(Error::LaunchArgs { kernel, rc });
    }
    check(rc, kernel)
}

/// The sticky/last launch error (`cudaGetLastError`), for launch paths that do not return one.
pub fn last_error(call: &'static str) -> Result<()> {
    if crate::device::current_is_host() {
        return Ok(());
    }
    // SAFETY: no arguments
    check(unsafe { ffi::cudaGetLastError() }, call)
}
