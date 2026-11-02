//! Device selection. By default the process uses the first visible device (`CUDA_VISIBLE_DEVICES` picks it).
//! Shared machines can opt into a guard: with `PERSEUS_GPU_GUARD=<gpu>` set, `Device::open` refuses unless exactly
//! that GPU is visible and the process holds the machine's GPU lock (`PERSEUS_CORE_LOCKED=1`, which `tools/gpu.sh`
//! exports in guarded mode). Every other entry point of this crate needs the `Device` it returns.
//! `PERSEUS_DEVICE=cpu` (or `Device::open_host`) opens the host device instead: the same API over host memory, inline
//! streams and the host kernel tables (pc-hostrt), no GPU touched.
use crate::error::{check, Error, Result};
use crate::ffi;
use std::marker::PhantomData;

/// Opt-in guard: the one GPU index this process may use. Unset or empty = no guard.
pub const GUARD_VAR: &str = "PERSEUS_GPU_GUARD";
/// `cpu` selects the host device in `Device::open`.
pub const DEVICE_VAR: &str = "PERSEUS_DEVICE";

thread_local! {
    /// Whether the device opened last on this thread is the host (CUDA's current device is per thread too).
    static HOST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether this thread's current device is the host one.
pub fn current_is_host() -> bool {
    HOST.with(|h| h.get())
}

/// Whether `PERSEUS_DEVICE` asks for the host device.
pub fn host_requested() -> bool {
    std::env::var(DEVICE_VAR).is_ok_and(|v| matches!(v.trim(), "cpu" | "host"))
}

/// Proof that device selection passed and the device is current on this thread. Not `Send`: CUDA's
/// current device is per host thread, and one host thread issues all work.
pub struct Device {
    runtime: i32,
    driver: i32,
    host: bool,
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
    /// Check the opt-in guard, select visible device 0 and read the runtime version; the host device when
    /// `PERSEUS_DEVICE=cpu`.
    pub fn open() -> Result<Device> {
        if host_requested() {
            return Ok(Device::open_host());
        }
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
        HOST.with(|h| h.set(false));
        Ok(Device { runtime, driver, host: false, _not_send: PhantomData })
    }

    /// The host device (no GPU, no guard): runtime and driver versions read 0.
    pub fn open_host() -> Device {
        HOST.with(|h| h.set(true));
        Device { runtime: 0, driver: 0, host: true, _not_send: PhantomData }
    }

    pub fn is_host(&self) -> bool {
        self.host
    }

    /// cudart version (e.g. 13020 for 13.2).
    pub fn runtime_version(&self) -> i32 {
        self.runtime
    }
    pub fn driver_version(&self) -> i32 {
        self.driver
    }

    /// Streaming multiprocessors of the current device.
    pub fn sm_count(&self) -> Result<usize> {
        if self.host {
            return Ok(pc_hostrt::threads());
        }
        let mut v = 0i32;
        // SAFETY: out-pointer to a local; attribute 16 = cudaDevAttrMultiProcessorCount of device 0 (the visible one)
        check(unsafe { ffi::cudaDeviceGetAttribute(&mut v, 16, 0) }, "cudaDeviceGetAttribute")?;
        Ok(v as usize)
    }

    /// (free, total) device memory in bytes.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        if self.host {
            return Ok(pc_hostrt::mem_info());
        }
        let (mut f, mut t) = (0usize, 0usize);
        // SAFETY: out-pointers to locals
        check(unsafe { ffi::cudaMemGetInfo(&mut f, &mut t) }, "cudaMemGetInfo")?;
        Ok((f, t))
    }

    /// Whole-device barrier. Setup and teardown only: the hot path syncs the owning stream (host reads) or fences
    /// with events, never the device.
    pub fn synchronize(&self) -> Result<()> {
        if self.host {
            return Ok(());
        }
        // SAFETY: no arguments
        check(unsafe { ffi::cudaDeviceSynchronize() }, "cudaDeviceSynchronize")
    }

    /// Profiler capture range (ncu --profile-from-start off / nsys --capture-range=cudaProfilerApi).
    pub fn profiler(&self, on: bool) -> Result<()> {
        if self.host {
            return Ok(());
        }
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
    if current_is_host() {
        return Ok(pc_hostrt::mem_info());
    }
    let (mut f, mut t) = (0usize, 0usize);
    // SAFETY: out-pointers to locals
    check(unsafe { ffi::cudaMemGetInfo(&mut f, &mut t) }, "cudaMemGetInfo")?;
    Ok((f, t))
}
