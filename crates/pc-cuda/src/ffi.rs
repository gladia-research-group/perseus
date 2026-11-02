//! Raw cudart entry points (CUDA 13.2). Allocation entry points are declared in `mem.rs` only.
#![allow(non_camel_case_types)]
use std::ffi::{c_char, c_void};

pub type cudaError_t = i32;
pub type cudaStream_t = *mut c_void;
pub type cudaEvent_t = *mut c_void;
pub type cudaGraph_t = *mut c_void;
pub type cudaGraphExec_t = *mut c_void;

pub const MEMCPY_H2D: i32 = 1;
pub const MEMCPY_D2H: i32 = 2;
pub const MEMCPY_D2D: i32 = 3;
pub const STREAM_NON_BLOCKING: u32 = 1;
pub const EVENT_DISABLE_TIMING: u32 = 2;
pub const CAPTURE_MODE_GLOBAL: i32 = 0;

extern "C" {
    pub fn cudaSetDevice(dev: i32) -> cudaError_t;
    pub fn cudaGetDeviceCount(n: *mut i32) -> cudaError_t;
    pub fn cudaRuntimeGetVersion(v: *mut i32) -> cudaError_t;
    pub fn cudaDriverGetVersion(v: *mut i32) -> cudaError_t;
    pub fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, n: usize, kind: i32, s: cudaStream_t) -> cudaError_t;
    pub fn cudaMemsetAsync(dst: *mut c_void, v: i32, n: usize, s: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamCreateWithFlags(s: *mut cudaStream_t, flags: u32) -> cudaError_t;
    pub fn cudaStreamCreateWithPriority(s: *mut cudaStream_t, flags: u32, priority: i32) -> cudaError_t;
    pub fn cudaDeviceGetStreamPriorityRange(least: *mut i32, greatest: *mut i32) -> cudaError_t;
    pub fn cudaDeviceGetAttribute(v: *mut i32, attr: i32, dev: i32) -> cudaError_t;
    pub fn cudaStreamDestroy(s: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamSynchronize(s: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamWaitEvent(s: cudaStream_t, e: cudaEvent_t, flags: u32) -> cudaError_t;
    pub fn cudaEventCreateWithFlags(e: *mut cudaEvent_t, flags: u32) -> cudaError_t;
    pub fn cudaEventDestroy(e: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventRecord(e: cudaEvent_t, s: cudaStream_t) -> cudaError_t;
    pub fn cudaEventSynchronize(e: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventElapsedTime(ms: *mut f32, a: cudaEvent_t, b: cudaEvent_t) -> cudaError_t;
    pub fn cudaDeviceSynchronize() -> cudaError_t;
    pub fn cudaGetLastError() -> cudaError_t;
    pub fn cudaGetErrorName(e: cudaError_t) -> *const c_char;
    pub fn cudaGetErrorString(e: cudaError_t) -> *const c_char;
    pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t;
    pub fn cudaStreamBeginCapture(s: cudaStream_t, mode: i32) -> cudaError_t;
    pub fn cudaStreamEndCapture(s: cudaStream_t, g: *mut cudaGraph_t) -> cudaError_t;
    pub fn cudaGraphInstantiate(e: *mut cudaGraphExec_t, g: cudaGraph_t, flags: u64) -> cudaError_t;
    pub fn cudaGraphLaunch(e: cudaGraphExec_t, s: cudaStream_t) -> cudaError_t;
    pub fn cudaGraphExecDestroy(e: cudaGraphExec_t) -> cudaError_t;
    pub fn cudaGraphDestroy(g: cudaGraph_t) -> cudaError_t;
    pub fn cudaProfilerStart() -> cudaError_t;
    pub fn cudaProfilerStop() -> cudaError_t;
}
