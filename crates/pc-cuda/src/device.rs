//! Device selection. By default the process uses the first visible device (`CUDA_VISIBLE_DEVICES` picks it).
//! Shared machines can opt into a guard: with `PERSEUS_GPU_GUARD=<gpu>` set, `Device::open` refuses unless exactly
//! that GPU is visible and the process holds the machine's GPU lock (`PERSEUS_CORE_LOCKED=1`, which `tools/gpu.sh`
//! exports in guarded mode). Every other entry point of this crate needs the `Device` it returns.
use crate::error::{check, Error, Result};
use crate::ffi;
use std::marker::PhantomData;

/// Opt-in guard: the one GPU index this process may use. Unset or empty = no guard.
pub const GUARD_VAR: &str = "PERSEUS_GPU_GUARD";

/// Proof that device selection passed and the device is current on this thread. Not `Send`: CUDA's
/// current device is per host thread, and one host thread issues all work.
pub struct Device {
    runtime: i32,
    driver: i32,
    _not_send: PhantomData<*const ()>,
}

/// The guard rule as a pure function of the three variables (unit-tested without a GPU). Returns whether the guard
/// is active (then exactly one device must be visible).
pub fn guard_check(guard: Option<&str>, locked: Option<&str>, visible: Option<&str>) -> Result<bool> {
    let gpu = match guard.map(str::trim) {
        None | Some("") => return Ok(false),
        Some(g) => g,
    };
    if locked != Some("1") {
        return Err(Error::Guard(format!(
            "{GUARD_VAR}={gpu}, PERSEUS_CORE_LOCKED={locked:?}: not under the GPU lock (run through tools/gpu.sh)"
        )));
    }
    if visible != Some(gpu) {
        return Err(Error::Guard(format!(
            "{GUARD_VAR}={gpu}, CUDA_VISIBLE_DEVICES={visible:?}: refusing to run (only GPU {gpu}; run through tools/gpu.sh)"
        )));
    }
    Ok(true)
}

impl Device {
    /// Check the opt-in guard, select visible device 0 and read the runtime version.
    pub fn open() -> Result<Device> {
        let guard = std::env::var(GUARD_VAR).ok();
        let locked = std::env::var("PERSEUS_CORE_LOCKED").ok();
        let visible = std::env::var("CUDA_VISIBLE_DEVICES").ok();
        let guarded = guard_check(guard.as_deref(), locked.as_deref(), visible.as_deref())?;
        let mut n = 0i32;
        // SAFETY: out-pointers to locals
        unsafe {
            check(ffi::cudaGetDeviceCount(&mut n), "cudaGetDeviceCount")?;
        }
        if n < 1 || (guarded && n != 1) {
            return Err(Error::Guard(format!("{n} devices visible under CUDA_VISIBLE_DEVICES={visible:?}")));
        }
        let (mut runtime, mut driver) = (0i32, 0i32);
        // SAFETY: plain calls / out-pointers to locals
        unsafe {
            check(ffi::cudaSetDevice(0), "cudaSetDevice")?;
            check(ffi::cudaRuntimeGetVersion(&mut runtime), "cudaRuntimeGetVersion")?;
            check(ffi::cudaDriverGetVersion(&mut driver), "cudaDriverGetVersion")?;
        }
        Ok(Device { runtime, driver, _not_send: PhantomData })
    }

    /// cudart version (e.g. 13020 for 13.2).
    pub fn runtime_version(&self) -> i32 {
        self.runtime
    }
    pub fn driver_version(&self) -> i32 {
        self.driver
    }

    /// (free, total) device memory in bytes.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        let (mut f, mut t) = (0usize, 0usize);
        // SAFETY: out-pointers to locals
        check(unsafe { ffi::cudaMemGetInfo(&mut f, &mut t) }, "cudaMemGetInfo")?;
        Ok((f, t))
    }

    /// Whole-device barrier. Setup and teardown only: the hot path syncs the owning stream (host reads) or fences
    /// with events, never the device.
    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: no arguments
        check(unsafe { ffi::cudaDeviceSynchronize() }, "cudaDeviceSynchronize")
    }

    /// Profiler capture range (ncu --profile-from-start off / nsys --capture-range=cudaProfilerApi).
    pub fn profiler(&self, on: bool) -> Result<()> {
        self.synchronize()?;
        // SAFETY: no arguments
        check(unsafe { if on { ffi::cudaProfilerStart() } else { ffi::cudaProfilerStop() } }, "cudaProfiler")
    }
}

#[cfg(test)]
mod tests {
    use super::guard_check;
    #[test]
    fn guard_rule() {
        // no guard: anything goes
        assert_eq!(guard_check(None, None, None).ok(), Some(false));
        assert_eq!(guard_check(Some(""), None, Some("0,1")).ok(), Some(false));
        // guarded on GPU 3
        let g = Some("3");
        assert_eq!(guard_check(g, Some("1"), Some("3")).ok(), Some(true));
        assert!(guard_check(g, None, Some("3")).is_err());
        assert!(guard_check(g, Some("0"), Some("3")).is_err());
        assert!(guard_check(g, Some("1"), None).is_err());
        assert!(guard_check(g, Some("1"), Some("0,1,2,3")).is_err());
        assert!(guard_check(g, Some("1"), Some("2")).is_err());
    }
}

/// (free, total) memory of the calling thread's current device, in bytes (`Device::mem_info` without the handle).
pub fn mem_info_current() -> Result<(usize, usize)> {
    let (mut f, mut t) = (0usize, 0usize);
    // SAFETY: out-pointers to locals
    check(unsafe { ffi::cudaMemGetInfo(&mut f, &mut t) }, "cudaMemGetInfo")?;
    Ok((f, t))
}
