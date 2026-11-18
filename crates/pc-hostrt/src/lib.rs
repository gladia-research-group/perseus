//! The host stand-in for the CUDA stream model, behind the same raw handles (`*mut c_void`) the kernel launchers and
//! cuBLASLt take. A host stream runs its work inline, in issue order: host issue order is a valid schedule of every
//! correctly evented CUDA program, so waits and syncs have nothing to wait for. While a stream captures, its work is
//! recorded instead (by value, as a CUDA graph keeps its nodes' arguments) and replayed by `HostGraph::launch`. As
//! in CUDA, a stream that waits on an event recorded inside a capture joins that capture until it ends.
//! Kernel bodies run on one thread pool sized by `PERSEUS_CPU_THREADS` (default: the visible cores, at most 36).
use std::cell::RefCell;
use std::ffi::c_void;
use std::rc::Rc;
use std::sync::atomic::{fence, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// One unit of recorded work.
pub type Op = Rc<dyn Fn()>;

/// One capture: the recorded work of its origin stream and of every stream that joined it.
struct Capture {
    ops: RefCell<Vec<Op>>,
    active: std::cell::Cell<bool>,
}

/// A host stream: inline execution, or recording into a capture (its own, or one it joined).
#[derive(Default)]
pub struct HostStream {
    rec: RefCell<Option<Rc<Capture>>>,
}

impl HostStream {
    /// The capture this stream records into (a joined capture that ended no longer counts).
    fn capture(&self) -> Option<Rc<Capture>> {
        let mut rec = self.rec.borrow_mut();
        if rec.as_ref().is_some_and(|c| !c.active.get()) {
            *rec = None;
        }
        rec.clone()
    }
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
    match stream(raw).capture() {
        Some(c) => c.ops.borrow_mut().push(Rc::new(op)),
        None => {
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
    stream(raw).capture().is_some()
}

/// # Safety
/// `raw` is a live handle from `new_stream`.
pub unsafe fn begin_capture(raw: *mut c_void) -> Result<(), &'static str> {
    let s = stream(raw);
    if s.capture().is_some() {
        return Err("capture already active on this stream");
    }
    *s.rec.borrow_mut() = Some(Rc::new(Capture { ops: RefCell::new(Vec::new()), active: std::cell::Cell::new(true) }));
    Ok(())
}

/// # Safety
/// `raw` is a live handle from `new_stream`.
pub unsafe fn end_capture(raw: *mut c_void) -> Result<HostGraph, &'static str> {
    let c = stream(raw).capture().ok_or("no capture active on this stream")?;
    *stream(raw).rec.borrow_mut() = None;
    c.active.set(false);
    let ops = c.ops.take();
    Ok(HostGraph { ops: Rc::new(ops) })
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

/// A host event: the time its stream passed the record point (timing) and, when it was last recorded inside a
/// capture, that capture (a stream waiting on it joins). Ordering is the issue order.
#[derive(Default)]
pub struct HostEvent {
    at: Mutex<Option<Instant>>,
    cap: RefCell<Option<Rc<Capture>>>,
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
    *(*e).cap.borrow_mut() = self::stream(stream).capture();
    submit(stream, move || *(*e).at.lock().unwrap() = Some(Instant::now()));
}

/// Make `stream` wait for `ev`: nothing to wait for on the host, but an event recorded inside a live capture pulls
/// the stream into that capture (cudaStreamWaitEvent under capture).
///
/// # Safety
/// Both handles are live.
pub unsafe fn wait(stream: *mut c_void, ev: *mut c_void) {
    let c = (*(ev as *const HostEvent)).cap.borrow().clone();
    if let Some(c) = c.filter(|c| c.active.get()) {
        let s = self::stream(stream);
        if s.capture().is_none() {
            *s.rec.borrow_mut() = Some(c);
        }
    }
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
        let cpus = if std::env::var("PERSEUS_CPU_PIN").is_ok_and(|v| v == "0") { Vec::new() } else { core_cpus() };
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads())
            .thread_name(|i| format!("pc-host-{i}"))
            .start_handler(move |i| {
                if !cpus.is_empty() {
                    pin(cpus[i % cpus.len()]);
                }
            })
            .build()
            .expect("host kernel pool")
    })
}

/// One allowed CPU per physical core (the first sibling of each), in CPU order: the AVX-512 kernels are bound by
/// the multiply ports, which hyperthread siblings share.
fn core_cpus() -> Vec<usize> {
    // SAFETY: plain syscall into a zeroed cpu_set_t
    let allowed: Vec<usize> = unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return Vec::new();
        }
        (0..libc::CPU_SETSIZE as usize).filter(|&c| libc::CPU_ISSET(c, &set)).collect()
    };
    let topo = |c: usize, f: &str| std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{c}/topology/{f}")).ok().map(|s| s.trim().to_string());
    let mut seen = std::collections::HashSet::new();
    allowed.into_iter().filter(|&c| seen.insert((topo(c, "physical_package_id"), topo(c, "core_id")))).collect()
}

fn pin(cpu: usize) {
    // SAFETY: plain syscall on the calling thread with a zeroed cpu_set_t
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
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

/// Back `bytes` at `p` with transparent huge pages where the kernel allows (MADV_HUGEPAGE) and fault every page
/// in now, in parallel on the kernel pool, so first launches do not pay page faults one 4 KB page at a time.
/// `PERSEUS_CPU_PREFAULT=0` leaves the memory lazy.
///
/// # Safety
/// `p .. p + bytes` is one live writable allocation whose contents the caller does not need (a byte per page is
/// overwritten with zero).
pub unsafe fn prefault(p: *mut u8, bytes: usize) {
    if std::env::var("PERSEUS_CPU_PREFAULT").is_ok_and(|v| v == "0") {
        return;
    }
    const PAGE: usize = 4096;
    let start = (p as usize).next_multiple_of(PAGE);
    if start < p as usize + bytes {
        libc::madvise(start as *mut c_void, (p as usize + bytes - start) / PAGE * PAGE, libc::MADV_HUGEPAGE);
    }
    let (base, pages) = (p as usize, bytes.div_ceil(PAGE));
    use rayon::prelude::*;
    par(|| {
        (0..pages).into_par_iter().with_min_len(512).for_each(|i| {
            // SAFETY: inside the allocation (caller contract)
            unsafe { std::ptr::write_volatile((base + (i * PAGE).min(bytes - 1)) as *mut u8, 0) };
        })
    });
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
    fn a_waiting_stream_joins_the_capture() {
        let (a, b, e) = (new_stream(), new_stream(), new_event());
        let n = Rc::new(Cell::new(0));
        unsafe {
            begin_capture(a).unwrap();
            record(e, a);
            wait(b, e);
            assert!(capturing(b));
            let m = n.clone();
            submit(b, move || m.set(m.get() + 1));
            let g = end_capture(a).unwrap();
            assert_eq!(n.get(), 0);
            assert!(!capturing(b));
            let m = n.clone();
            submit(b, move || m.set(m.get() + 100));
            assert_eq!(n.get(), 100);
            g.launch(a);
            assert_eq!(n.get(), 101);
            destroy_event(e);
            destroy_stream(a);
            destroy_stream(b);
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
