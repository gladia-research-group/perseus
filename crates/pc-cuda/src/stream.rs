//! Lane-typed streams, events (single and pooled), captured graphs and the `Ready` hand-over token.
use crate::device::Device;
use crate::error::{check, Error, Result};
use crate::ffi;
use crate::lane::Lane;
use crate::RawStream;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::ops::Deref;

/// An owned non-blocking CUDA stream of lane `S`. Not `Clone`, not `Send`: one host thread issues it.
pub struct Stream<S: Lane> {
    raw: ffi::cudaStream_t,
    _s: PhantomData<(S, *const ())>,
}

impl<S: Lane> Stream<S> {
    pub fn new(_dev: &Device) -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: out-pointer to a local
        check(unsafe { ffi::cudaStreamCreateWithFlags(&mut raw, ffi::STREAM_NON_BLOCKING) }, "cudaStreamCreateWithFlags")?;
        Ok(Stream { raw, _s: PhantomData })
    }
    /// A stream of the device's highest priority (its blocks are scheduled before the default streams' pending ones).
    pub fn high_priority(_dev: &Device) -> Result<Self> {
        let (mut least, mut greatest) = (0, 0);
        // SAFETY: out-pointers to locals
        check(unsafe { ffi::cudaDeviceGetStreamPriorityRange(&mut least, &mut greatest) }, "cudaDeviceGetStreamPriorityRange")?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: out-pointer to a local
        check(unsafe { ffi::cudaStreamCreateWithPriority(&mut raw, ffi::STREAM_NON_BLOCKING, greatest) }, "cudaStreamCreateWithPriority")?;
        Ok(Stream { raw, _s: PhantomData })
    }
    /// The `cudaStream_t` for a kernel launcher.
    pub fn raw(&self) -> RawStream {
        self.raw
    }
    /// Block the host until everything issued on this stream finished (host reads of this lane's buffers).
    pub fn sync(&self) -> Result<()> {
        // SAFETY: valid stream handle owned by self
        check(unsafe { ffi::cudaStreamSynchronize(self.raw) }, "cudaStreamSynchronize")
    }
    /// Make this stream wait for `e` (recorded on any stream).
    pub fn wait(&self, e: &Event) -> Result<()> {
        // SAFETY: both handles are live (borrowed)
        check(unsafe { ffi::cudaStreamWaitEvent(self.raw, e.raw, 0) }, "cudaStreamWaitEvent")
    }
    /// Record `e` at the current tail of this stream.
    pub fn record(&self, e: &Event) -> Result<()> {
        e.record(self)
    }
    /// Capture everything `f` issues on this stream into one instantiated graph (replayed with `GraphExec::launch`).
    /// The capture is ended even when `f` fails, so the stream is usable afterwards.
    pub fn capture<R>(&self, f: impl FnOnce(&Self) -> Result<R>) -> Result<(GraphExec, R)> {
        // SAFETY: valid stream; global capture mode
        check(unsafe { ffi::cudaStreamBeginCapture(self.raw, ffi::CAPTURE_MODE_GLOBAL) }, "cudaStreamBeginCapture")?;
        let r = f(self);
        let mut g = std::ptr::null_mut();
        // SAFETY: ends the capture begun above
        let end = check(unsafe { ffi::cudaStreamEndCapture(self.raw, &mut g) }, "cudaStreamEndCapture");
        let r = match (r, end) {
            (Ok(r), Ok(())) => r,
            (Err(e), _) | (Ok(_), Err(e)) => {
                if !g.is_null() {
                    // SAFETY: graph returned by EndCapture, destroyed once
                    unsafe { ffi::cudaGraphDestroy(g) };
                }
                return Err(e);
            }
        };
        let mut e = std::ptr::null_mut();
        // SAFETY: g is a valid captured graph; destroyed right after instantiation (the exec keeps its own copy)
        let inst = check(unsafe { ffi::cudaGraphInstantiate(&mut e, g, 0) }, "cudaGraphInstantiate");
        unsafe { ffi::cudaGraphDestroy(g) };
        inst?;
        Ok((GraphExec { raw: e }, r))
    }
}

impl<S: Lane> Drop for Stream<S> {
    fn drop(&mut self) {
        // SAFETY: owned handle; errors are ignored in Drop
        unsafe {
            ffi::cudaStreamSynchronize(self.raw);
            ffi::cudaStreamDestroy(self.raw);
        }
    }
}

/// A CUDA event. Ordering events are created without timing (cheaper); `timing()` events support `elapsed_ms`.
pub struct Event {
    raw: ffi::cudaEvent_t,
}

impl Event {
    /// Ordering-only event (cudaEventDisableTiming).
    pub fn new(_dev: &Device) -> Result<Self> {
        Self::with_flags(ffi::EVENT_DISABLE_TIMING)
    }
    /// Timing event (for benchmarks).
    pub fn timing(_dev: &Device) -> Result<Self> {
        Self::with_flags(0)
    }
    fn with_flags(flags: u32) -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: out-pointer to a local
        check(unsafe { ffi::cudaEventCreateWithFlags(&mut raw, flags) }, "cudaEventCreateWithFlags")?;
        Ok(Event { raw })
    }
    pub fn record<S: Lane>(&self, s: &Stream<S>) -> Result<()> {
        // SAFETY: both handles are live (borrowed)
        check(unsafe { ffi::cudaEventRecord(self.raw, s.raw()) }, "cudaEventRecord")
    }
    /// Block the host until the recorded work completed.
    pub fn sync(&self) -> Result<()> {
        // SAFETY: live handle
        check(unsafe { ffi::cudaEventSynchronize(self.raw) }, "cudaEventSynchronize")
    }
    /// Milliseconds from `self` to `later` (both timing events, both recorded and completed).
    pub fn elapsed_ms(&self, later: &Event) -> Result<f32> {
        let mut ms = 0f32;
        // SAFETY: live handles, out-pointer to a local
        check(unsafe { ffi::cudaEventElapsedTime(&mut ms, self.raw, later.raw) }, "cudaEventElapsedTime")?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: owned handle; errors ignored in Drop
        unsafe {
            ffi::cudaEventDestroy(self.raw);
        }
    }
}

/// A fixed set of ordering events created at setup (no event creation in the hot path). `get` hands out a
/// `PoolEvent` that returns to the pool when dropped. Re-recording a returned event is safe: a stream wait enqueued
/// earlier captured the event's state at enqueue time, and a host wait already returned.
pub struct EventPool {
    events: Vec<Event>,
    free: RefCell<Vec<usize>>,
}

impl EventPool {
    pub fn new(dev: &Device, size: usize) -> Result<Self> {
        let events = (0..size).map(|_| Event::new(dev)).collect::<Result<Vec<_>>>()?;
        Ok(EventPool { events, free: RefCell::new((0..size).rev().collect()) })
    }
    /// A free event, or `EventsExhausted` (the pool size is the `MemCfg::events` knob).
    pub fn get(&self) -> Result<PoolEvent<'_>> {
        let idx = self.free.borrow_mut().pop().ok_or(Error::EventsExhausted { size: self.events.len() })?;
        Ok(PoolEvent { pool: self, idx })
    }
    pub fn size(&self) -> usize {
        self.events.len()
    }
    pub fn in_flight(&self) -> usize {
        self.events.len() - self.free.borrow().len()
    }
}

/// An event borrowed from an `EventPool`.
pub struct PoolEvent<'p> {
    pool: &'p EventPool,
    idx: usize,
}
impl Deref for PoolEvent<'_> {
    type Target = Event;
    fn deref(&self) -> &Event {
        &self.pool.events[self.idx]
    }
}
impl Drop for PoolEvent<'_> {
    fn drop(&mut self) {
        self.pool.free.borrow_mut().push(self.idx);
    }
}

/// An instantiated CUDA graph (one per (block, variant) in replay).
pub struct GraphExec {
    raw: ffi::cudaGraphExec_t,
}
impl GraphExec {
    pub fn launch<S: Lane>(&self, s: &Stream<S>) -> Result<()> {
        // SAFETY: live exec and stream
        check(unsafe { ffi::cudaGraphLaunch(self.raw, s.raw()) }, "cudaGraphLaunch")
    }
}
impl Drop for GraphExec {
    fn drop(&mut self) {
        // SAFETY: owned handle; errors ignored in Drop
        unsafe {
            ffi::cudaGraphExecDestroy(self.raw);
        }
    }
}

/// A value whose accesses are ordered by lane `Lane` and that can be re-typed to another lane once that lane waits
/// for an event recorded after the last access (`DBuf`, pc-gpu's `CtBuf`).
pub trait OnLane: Sized {
    type Lane: Lane;
    type On<U: Lane>;
    /// Re-type to lane `U`.
    ///
    /// # Safety
    /// Every access issued on `Self::Lane` must be ordered before any access on `U` (an event recorded on the old
    /// lane after the last access, waited for by `U`). `Ready::join` and `DBuf::handoff` are the safe callers.
    unsafe fn relane<U: Lane>(self) -> Self::On<U>;
}

/// A value produced on lane `S` (e.g. an upload on `Copy`) together with the event recorded after the producing
/// work. Consumers `join` it into their own lane (the stream waits on the event, the value changes lane), or the
/// host `wait`s for it. Dropping a `Ready` without joining is allowed (the work is simply never awaited).
pub struct Ready<'e, S: Lane, T> {
    value: T,
    ev: PoolEvent<'e>,
    _s: PhantomData<S>,
}

impl<'e, S: Lane, T> Ready<'e, S, T> {
    /// Record `ev` on `on` after the work producing `value` was issued there.
    pub fn record(value: T, on: &Stream<S>, ev: PoolEvent<'e>) -> Result<Self> {
        ev.record(on)?;
        Ok(Ready { value, ev, _s: PhantomData })
    }
    /// Host wait (event synchronize), then the value, unchanged.
    pub fn wait(self) -> Result<T> {
        self.ev.sync()?;
        Ok(self.value)
    }
    /// Make `to` wait for the producing work and return the value as is (for values that are not lane-typed, e.g.
    /// data later read through a `ConstBuf`).
    pub fn join_as_is<U: Lane>(self, to: &Stream<U>) -> Result<T> {
        to.wait(&self.ev)?;
        Ok(self.value)
    }
}

impl<'e, S: Lane, T: OnLane<Lane = S>> Ready<'e, S, T> {
    /// Make `to` wait for the producing work and move the value to lane `U`.
    pub fn join<U: Lane>(self, to: &Stream<U>) -> Result<T::On<U>> {
        to.wait(&self.ev)?;
        // SAFETY: `to` now waits for the event recorded after the last access on S
        Ok(unsafe { self.value.relane::<U>() })
    }
}
