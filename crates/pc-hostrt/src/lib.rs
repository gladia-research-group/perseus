//! The host stand-in for the CUDA stream model, behind the same raw handles (`*mut c_void`) the kernel launchers and
//! cuBLASLt take. A host stream runs its work inline, in issue order: host issue order is a valid schedule of every
//! correctly evented CUDA program, so waits and syncs have nothing to wait for. While a stream captures, its work is
//! recorded instead (by value, as a CUDA graph keeps its nodes' arguments) and replayed by `HostGraph::launch`.
//! Kernel bodies run on one thread pool sized by `PERSEUS_CPU_THREADS` (default: the visible cores, at most 36).
use std::cell::RefCell;
use std::ffi::c_void;
use std::rc::Rc;
use std::sync::atomic::{fence, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// One unit of recorded work.
pub type Op = Rc<dyn Fn()>;

/// A host stream: inline execution, or recording between `begin_capture` and `end_capture`.
#[derive(Default)]
pub struct HostStream {
    rec: RefCell<Option<Vec<Op>>>,
}

/// A new host stream as a raw handle (freed with `destroy_stream`).
pub fn new_stream() -> *mut c_void {
    Box::into_raw(Box::<HostStream>::default()) as *mut c_void
}

/// # Safety
/// `raw` came from `new_stream` and is not used afterwards.
pub unsafe fn destroy_stream(raw: *mut c_void) {
    drop(Box::from_raw(raw as *mut HostStream));
}

/// # Safety
/// `raw` is a live handle from `new_stream`.
unsafe fn stream<'a>(raw: *mut c_void) -> &'a HostStream {
    &*(raw as *const HostStream)
}

/// Run `op` now, or record it when the stream captures.
///
/// # Safety
/// `raw` is a live handle from `new_stream`; everything `op` dereferences stays valid until it ran (eager: the
/// call; captured: every launch of the graph), as for the CUDA work it stands for.
pub unsafe fn submit(raw: *mut c_void, op: impl Fn() + 'static) {
    let s = stream(raw);
    let mut rec = s.rec.borrow_mut();
    match rec.as_mut() {
        Some(v) => v.push(Rc::new(op)),
        None => {
            drop(rec);
            op();
            fence(Ordering::SeqCst);
        }
    }
}

/// Whether the stream is recording.
///
/// # Safety
/// `raw` is a live handle from `new_stream`.
pub unsafe fn capturing(raw: *mut c_void) -> bool {
    stream(raw).rec.borrow().is_some()
}

/// # Safety
/// `raw` is a live handle from `new_stream`.
pub unsafe fn begin_capture(raw: *mut c_void) -> Result<(), &'static str> {
    let mut rec = stream(raw).rec.borrow_mut();
    if rec.is_some() {
        return Err("capture already active on this stream");
    }
    *rec = Some(Vec::new());
    Ok(())
}

/// # Safety
/// `raw` is a live handle from `new_stream`.
pub unsafe fn end_capture(raw: *mut c_void) -> Result<HostGraph, &'static str> {
    stream(raw).rec.borrow_mut().take().map(|ops| HostGraph { ops: Rc::new(ops) }).ok_or("no capture active on this stream")
}

/// A recorded sequence of work, replayed in order.
pub struct HostGraph {
    ops: Rc<Vec<Op>>,
}

impl HostGraph {
    pub fn len(&self) -> usize {
        self.ops.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
    /// Replay on `raw` (recorded as one unit when that stream captures).
    ///
    /// # Safety
    /// `raw` is a live handle from `new_stream`; the pointers the recorded work holds are still valid.
    pub unsafe fn launch(&self, raw: *mut c_void) {
        let ops = self.ops.clone();
        submit(raw, move || ops.iter().for_each(|op| op()));
    }
}

/// A host event: the time its stream passed the record point (timing), nothing else (ordering is the issue order).
#[derive(Default)]
pub struct HostEvent {
    at: Mutex<Option<Instant>>,
}

pub fn new_event() -> *mut c_void {
    Box::into_raw(Box::<HostEvent>::default()) as *mut c_void
}

/// # Safety
/// `raw` came from `new_event` and is not used afterwards.
pub unsafe fn destroy_event(raw: *mut c_void) {
    drop(Box::from_raw(raw as *mut HostEvent));
}

/// Record `ev` on `stream` (when the stream passes this point).
///
/// # Safety
/// Both handles are live; `ev` outlives every graph that recorded it.
pub unsafe fn record(ev: *mut c_void, stream: *mut c_void) {
    let e = ev as *const HostEvent;
    submit(stream, move || *(*e).at.lock().unwrap() = Some(Instant::now()));
}

/// Milliseconds between two recorded events (`None` while either was never recorded).
///
/// # Safety
/// Both handles are live.
pub unsafe fn elapsed_ms(a: *mut c_void, b: *mut c_void) -> Option<f32> {
    let ta = (*(a as *const HostEvent)).at.lock().unwrap().as_ref().copied()?;
    let tb = (*(b as *const HostEvent)).at.lock().unwrap().as_ref().copied()?;
    Some(if tb >= ta { (tb - ta).as_secs_f64() * 1e3 } else { -((ta - tb).as_secs_f64() * 1e3) } as f32)
}

/// The kernels' thread pool.
pub fn pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new().num_threads(threads()).thread_name(|i| format!("pc-host-{i}")).build().expect("host kernel pool")
    })
}

/// Threads of the kernel pool: `PERSEUS_CPU_THREADS`, else the available cores capped at 36.
pub fn threads() -> usize {
    std::env::var("PERSEUS_CPU_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n: &usize| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(36))
}

/// Run `f` on the kernel pool.
pub fn par<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    pool().install(f)
}

/// (available, total) host memory in bytes (`/proc/meminfo`), the host's `cudaMemGetInfo`.
pub fn mem_info() -> (usize, usize) {
    let txt = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let field = |name: &str| {
        txt.lines()
            .find(|l| l.starts_with(name))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<usize>().ok())
            .map_or(0, |kb| kb * 1024)
    };
    (field("MemAvailable:"), field("MemTotal:"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn inline_then_recorded_then_replayed() {
        let s = new_stream();
        let n = Rc::new(Cell::new(0));
        let m = n.clone();
        unsafe {
            submit(s, move || m.set(m.get() + 1));
            assert_eq!(n.get(), 1);
            begin_capture(s).unwrap();
            assert!(capturing(s));
            let m = n.clone();
            submit(s, move || m.set(m.get() + 10));
            assert_eq!(n.get(), 1);
            let g = end_capture(s).unwrap();
            assert_eq!(g.len(), 1);
            g.launch(s);
            g.launch(s);
            assert_eq!(n.get(), 21);
            destroy_stream(s);
        }
    }

    #[test]
    fn events_time_the_stream() {
        let s = new_stream();
        let (a, b) = (new_event(), new_event());
        unsafe {
            assert!(elapsed_ms(a, b).is_none());
            record(a, s);
            record(b, s);
            assert!(elapsed_ms(a, b).unwrap() >= 0.0);
            destroy_event(a);
            destroy_event(b);
            destroy_stream(s);
        }
    }
}
