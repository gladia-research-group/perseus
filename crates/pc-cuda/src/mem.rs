//! Device memory: ONE `cudaMalloc` (the `Arena`), carved into lane-typed buffers at setup and frozen before the hot
//! loop, so nothing allocates or frees while kernels are in flight. This file is the only place in the workspace
//! that names `cudaMalloc*` / `cudaFree*` (grep gate in `tools/ci.sh`).
//!
//! Buffers are typed by their element, `DBuf<'a, S, T: Pod>`, lengths and offsets in elements; `'a` borrows the
//! arena, `S` is the lane whose stream orders every access, and moving a buffer to another lane consumes it
//! (`handoff`, `Ready::join`). Host reads (`download`) synchronize the owning stream only. Design notes:
//! docs/ARCHITECTURE.md "pc-cuda memory".
use crate::device::Device;
use crate::error::{check, Error, Result};
use crate::ffi::{self, MEMCPY_D2D, MEMCPY_D2H, MEMCPY_H2D};
use crate::lane::Lane;
use crate::stream::{Event, OnLane, Stream};
use std::cell::Cell;
use std::ffi::c_void;
use std::marker::PhantomData;

extern "C" {
    fn cudaMalloc(p: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(p: *mut c_void) -> i32;
    fn cudaMallocHost(p: *mut *mut c_void, size: usize) -> i32;
    fn cudaFreeHost(p: *mut c_void) -> i32;
}

/// Alignment of every arena carve (bytes): coalescing and vector loads up to 256-bit never straddle a carve.
pub const ARENA_ALIGN: usize = 256;

/// Element types that may live in device memory and travel by memcpy.
///
/// # Safety
/// `Copy`, no padding bytes, no pointers to host memory, and every bit pattern is a valid value (device memory and
/// zeroed pinned memory are read back as `T`).
pub unsafe trait Pod: Copy + 'static {}
// SAFETY: plain integers / floats / arrays of them; pc-core's DevPrimeConst is 8 x u64-sized fields without padding
// (size-asserted 64 B in pc-core).
unsafe impl Pod for u8 {}
unsafe impl Pod for u16 {}
unsafe impl Pod for u32 {}
unsafe impl Pod for u64 {}
unsafe impl Pod for i32 {}
unsafe impl Pod for i64 {}
unsafe impl Pod for f32 {}
unsafe impl Pod for f64 {}
unsafe impl Pod for pc_core::modmul::DevPrimeConst {}
unsafe impl<T: Pod, const K: usize> Pod for [T; K] {}

enum Backing {
    Device,
    /// Host bytes standing in for device memory: only for the GPU-free carve/freeze unit tests (never dereferenced
    /// as device memory, never passed to a kernel).
    #[cfg(test)]
    Host(#[allow(dead_code)] Vec<u8>),
}

/// The one device allocation of a process, carved by a bump pointer. Not `Sync`: carving happens on the setup
/// thread.
pub struct Arena {
    base: *mut u8,
    cap: usize,
    used: Cell<usize>,
    frozen: Cell<bool>,
    backing: Backing,
}

impl Arena {
    /// One `cudaMalloc` of `bytes`.
    pub fn new(_dev: &Device, bytes: usize) -> Result<Self> {
        let mut p = std::ptr::null_mut();
        // SAFETY: out-pointer to a local; the guard passed (`Device` exists)
        check(unsafe { cudaMalloc(&mut p, bytes.max(ARENA_ALIGN)) }, "cudaMalloc (arena)")?;
        Ok(Arena { base: p as *mut u8, cap: bytes, used: Cell::new(0), frozen: Cell::new(false), backing: Backing::Device })
    }

    #[cfg(test)]
    pub(crate) fn host_for_test(bytes: usize) -> Self {
        let mut v = vec![0u8; bytes + ARENA_ALIGN];
        let off = v.as_mut_ptr().align_offset(ARENA_ALIGN);
        // SAFETY: off < ARENA_ALIGN <= len
        let base = unsafe { v.as_mut_ptr().add(off) };
        Arena { base, cap: bytes, used: Cell::new(0), frozen: Cell::new(false), backing: Backing::Host(v) }
    }

    /// Bytes `n` elements of `T` consume in an arena (one carve, alignment included). `MemPlan` sums these.
    pub const fn bytes_for<T>(n: usize) -> usize {
        (n * std::mem::size_of::<T>()).div_ceil(ARENA_ALIGN) * ARENA_ALIGN
    }

    /// Carve `n` elements of `T` for lane `S` (`ARENA_ALIGN`-aligned). Overflow is an error naming `what`.
    ///
    /// # Panics
    /// After `freeze`: an allocation in the hot loop is a programming error, not a runtime condition.
    pub fn alloc<S: Lane, T: Pod>(&self, what: &'static str, n: usize) -> Result<DBuf<'_, S, T>> {
        assert!(!self.frozen.get(), "arena is frozen: allocation of {what} in the hot loop");
        let off = self.used.get().div_ceil(ARENA_ALIGN) * ARENA_ALIGN;
        let need = n * std::mem::size_of::<T>();
        if off + need > self.cap {
            return Err(Error::ArenaFull { what, need, used: off, cap: self.cap });
        }
        self.used.set(off + need);
        // SAFETY: off + need <= cap, inside the allocation
        let ptr = unsafe { self.base.add(off) } as *mut T;
        Ok(DBuf { ptr, len: n, _l: PhantomData })
    }

    /// No allocation after this (enforced by `alloc`).
    pub fn freeze(&self) {
        self.frozen.set(true);
    }
    pub fn is_frozen(&self) -> bool {
        self.frozen.get()
    }
    pub fn used_bytes(&self) -> usize {
        self.used.get()
    }
    pub fn capacity(&self) -> usize {
        self.cap
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        if matches!(self.backing, Backing::Device) {
            // SAFETY: no buffer outlives the arena (borrow); wait for in-flight work, then the one free. Errors are
            // ignored in Drop.
            unsafe {
                ffi::cudaDeviceSynchronize();
                cudaFree(self.base as *mut c_void);
            }
        }
    }
}

/// A device buffer of `len` elements of `T`, ordered by lane `S`, living in arena `'a`.
pub struct DBuf<'a, S: Lane, T: Pod> {
    ptr: *mut T,
    len: usize,
    _l: PhantomData<(&'a Arena, S, *const ())>,
}

fn bounds(what: &'static str, len: usize, cap: usize) -> Result<()> {
    if len > cap {
        return Err(Error::Bounds { what, len, cap });
    }
    Ok(())
}

impl<'a, S: Lane, T: Pod> DBuf<'a, S, T> {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Read pointer for a launch on this buffer's lane.
    pub fn dptr(&self) -> *const T {
        self.ptr
    }
    /// Write pointer: needs exclusive access.
    pub fn dptr_mut(&mut self) -> *mut T {
        self.ptr
    }
    /// Untyped read pointer (kernel argument structs carry `*const c_void`).
    pub fn vptr(&self) -> *const c_void {
        self.ptr as *const c_void
    }
    pub fn vptr_mut(&mut self) -> *mut c_void {
        self.ptr as *mut c_void
    }

    /// Synchronous upload at element offset `off` (setup / tests): ordered on the lane, returns after the copy
    /// completed (syncs this lane's stream; the source is pageable host memory).
    pub fn upload_at(&mut self, s: &Stream<S>, off: usize, data: &[T]) -> Result<()> {
        bounds("upload", off + data.len(), self.len)?;
        // SAFETY: range checked above; host slice live for the call, and the stream is synced before returning
        unsafe {
            check(
                ffi::cudaMemcpyAsync(
                    self.ptr.add(off) as *mut c_void,
                    data.as_ptr() as *const c_void,
                    std::mem::size_of_val(data),
                    MEMCPY_H2D,
                    s.raw(),
                ),
                "cudaMemcpyAsync H2D",
            )?;
        }
        s.sync()
    }
    pub fn upload(&mut self, s: &Stream<S>, data: &[T]) -> Result<()> {
        self.upload_at(s, 0, data)
    }

    /// Host copy of elements `[off, off + n)`, ordered after everything issued on the lane (syncs the lane).
    pub fn download_range(&self, s: &Stream<S>, off: usize, n: usize) -> Result<Vec<T>> {
        bounds("download", off + n, self.len)?;
        let mut v: Vec<T> = Vec::with_capacity(n);
        // SAFETY: range checked; the copy writes n elements into the reserved capacity, completed by the sync
        // before set_len (T: Pod, any bit pattern valid)
        unsafe {
            check(
                ffi::cudaMemcpyAsync(
                    v.as_mut_ptr() as *mut c_void,
                    self.ptr.add(off) as *const c_void,
                    n * std::mem::size_of::<T>(),
                    MEMCPY_D2H,
                    s.raw(),
                ),
                "cudaMemcpyAsync D2H",
            )?;
            s.sync()?;
            v.set_len(n);
        }
        Ok(v)
    }
    pub fn download(&self, s: &Stream<S>) -> Result<Vec<T>> {
        self.download_range(s, 0, self.len)
    }

    /// Asynchronous H2D from pinned memory into `[off, off + src.len())`.
    ///
    /// # Safety
    /// `src` must stay alive and unmodified until the stream has passed this copy (join/wait the `Ready` that
    /// covers it before reusing the pinned slot).
    pub unsafe fn upload_pinned_async(&mut self, s: &Stream<S>, off: usize, src: &PinnedBuf<T>) -> Result<()> {
        bounds("upload_pinned_async", off + src.len(), self.len)?;
        check(
            ffi::cudaMemcpyAsync(
                self.ptr.add(off) as *mut c_void,
                src.as_ptr() as *const c_void,
                src.len() * std::mem::size_of::<T>(),
                MEMCPY_H2D,
                s.raw(),
            ),
            "cudaMemcpyAsync H2D (pinned)",
        )
    }

    /// Asynchronous H2D of the pinned words `src[src_off..src_off + n]` into `[off, off + n)`.
    ///
    /// # Safety
    /// As `upload_pinned_async`, for that range of `src`.
    pub unsafe fn upload_pinned_range_async(&mut self, s: &Stream<S>, off: usize, src: &PinnedBuf<T>, src_off: usize, n: usize) -> Result<()> {
        bounds("upload_pinned_range_async", off + n, self.len)?;
        bounds("upload_pinned_range_async (source)", src_off + n, src.len())?;
        check(
            ffi::cudaMemcpyAsync(
                self.ptr.add(off) as *mut c_void,
                src.as_ptr().add(src_off) as *const c_void,
                n * std::mem::size_of::<T>(),
                MEMCPY_H2D,
                s.raw(),
            ),
            "cudaMemcpyAsync H2D (pinned range)",
        )
    }

    /// Asynchronous D2H of `[off, off + dst.len())` into pinned memory.
    ///
    /// # Safety
    /// `dst` must not be read or dropped until the stream has passed this copy.
    pub unsafe fn download_pinned_async(&self, s: &Stream<S>, off: usize, dst: &mut PinnedBuf<T>) -> Result<()> {
        bounds("download_pinned_async", off + dst.len(), self.len)?;
        check(
            ffi::cudaMemcpyAsync(
                dst.as_mut_ptr() as *mut c_void,
                self.ptr.add(off) as *const c_void,
                dst.len() * std::mem::size_of::<T>(),
                MEMCPY_D2H,
                s.raw(),
            ),
            "cudaMemcpyAsync D2H (pinned)",
        )
    }

    /// Zero the whole buffer (ordered on the lane).
    pub fn zero(&mut self, s: &Stream<S>) -> Result<()> {
        // SAFETY: whole owned range
        check(
            unsafe { ffi::cudaMemsetAsync(self.ptr as *mut c_void, 0, self.len * std::mem::size_of::<T>(), s.raw()) },
            "cudaMemsetAsync",
        )
    }
    /// Device-to-device copy of `n` elements from another buffer of the same lane.
    pub fn copy_from(&mut self, s: &Stream<S>, src: &DBuf<'_, S, T>, n: usize) -> Result<()> {
        bounds("copy_from (dst)", n, self.len)?;
        bounds("copy_from (src)", n, src.len)?;
        // SAFETY: ranges checked; distinct buffers (&mut self vs &src)
        check(
            unsafe {
                ffi::cudaMemcpyAsync(
                    self.ptr as *mut c_void,
                    src.ptr as *const c_void,
                    n * std::mem::size_of::<T>(),
                    MEMCPY_D2D,
                    s.raw(),
                )
            },
            "cudaMemcpyAsync D2D",
        )
    }
    /// Device-to-device copy of `n` elements of constant data into this buffer.
    pub fn copy_from_const(&mut self, s: &Stream<S>, src: &ConstBuf<'_, T>, n: usize) -> Result<()> {
        bounds("copy_from_const (dst)", n, self.len)?;
        bounds("copy_from_const (src)", n, src.len)?;
        // SAFETY: ranges checked; constant data is never written after setup
        check(
            unsafe {
                ffi::cudaMemcpyAsync(
                    self.ptr as *mut c_void,
                    src.ptr as *const c_void,
                    n * std::mem::size_of::<T>(),
                    MEMCPY_D2D,
                    s.raw(),
                )
            },
            "cudaMemcpyAsync D2D (const)",
        )
    }

    /// Move the buffer to lane `U`: `ev` is recorded after the last access on `S` and `U` waits on it.
    pub fn handoff<U: Lane>(self, from: &Stream<S>, to: &Stream<U>, ev: &Event) -> Result<DBuf<'a, U, T>> {
        ev.record(from)?;
        to.wait(ev)?;
        // SAFETY: `to` waits for the event recorded after the last access on S
        Ok(unsafe { self.relane::<U>() })
    }

    /// Split into two disjoint buffers at element `at` (both stay on lane S; the parent is consumed).
    pub fn split(self, at: usize) -> Result<(DBuf<'a, S, T>, DBuf<'a, S, T>)> {
        bounds("split", at, self.len)?;
        // SAFETY: at <= len
        let hi = unsafe { self.ptr.add(at) };
        Ok((DBuf { ptr: self.ptr, len: at, _l: PhantomData }, DBuf { ptr: hi, len: self.len - at, _l: PhantomData }))
    }

    /// Freeze into shared read-only data: waits for the lane's work, consumes the writable handle.
    pub fn into_const(self, s: &Stream<S>) -> Result<ConstBuf<'a, T>> {
        s.sync()?;
        Ok(ConstBuf { ptr: self.ptr, len: self.len, _a: PhantomData })
    }
}

impl<'a, S: Lane, T: Pod> OnLane for DBuf<'a, S, T> {
    type Lane = S;
    type On<U: Lane> = DBuf<'a, U, T>;
    unsafe fn relane<U: Lane>(self) -> DBuf<'a, U, T> {
        DBuf { ptr: self.ptr, len: self.len, _l: PhantomData }
    }
}

/// Immutable device data written once at setup (followed by a sync of the writing lane): readable from any lane.
pub struct ConstBuf<'a, T: Pod> {
    ptr: *const T,
    len: usize,
    _a: PhantomData<&'a Arena>,
}
impl<T: Pod> Clone for ConstBuf<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: Pod> Copy for ConstBuf<'_, T> {}

impl<'a, T: Pod> ConstBuf<'a, T> {
    /// Carve, upload and sync (setup only).
    pub fn new<S: Lane>(arena: &'a Arena, what: &'static str, s: &Stream<S>, data: &[T]) -> Result<Self> {
        let mut b = arena.alloc::<S, T>(what, data.len())?;
        b.upload(s, data)?;
        b.into_const(s)
    }
    pub fn dptr(&self) -> *const T {
        self.ptr
    }
    pub fn vptr(&self) -> *const c_void {
        self.ptr as *const c_void
    }
    /// Pointer to element `i` (row/table offsets in launch arguments).
    pub fn dptr_at(&self, i: usize) -> Result<*const T> {
        bounds("ConstBuf::dptr_at", i, self.len)?;
        // SAFETY: i <= len
        Ok(unsafe { self.ptr.add(i) })
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Host copy (tests, checks; syncs the given stream).
    pub fn download<S: Lane>(&self, s: &Stream<S>) -> Result<Vec<T>> {
        let view: DBuf<'a, S, T> = DBuf { ptr: self.ptr as *mut T, len: self.len, _l: PhantomData };
        view.download(s)
    }
}

/// Pinned (page-locked) host memory for asynchronous copies. Zero-initialized. Setup-time allocation: the
/// pinned arena (stage ring + KV pages) is carved from one of these.
pub struct PinnedBuf<T: Pod> {
    ptr: *mut T,
    len: usize,
}

impl<T: Pod> PinnedBuf<T> {
    pub fn new(_dev: &Device, len: usize) -> Result<Self> {
        let bytes = (len * std::mem::size_of::<T>()).max(1);
        let mut p = std::ptr::null_mut();
        // SAFETY: out-pointer to a local
        check(unsafe { cudaMallocHost(&mut p, bytes) }, "cudaMallocHost")?;
        // SAFETY: freshly allocated `bytes` of host memory; zero is a valid T (Pod)
        unsafe { std::ptr::write_bytes(p as *mut u8, 0, bytes) };
        Ok(PinnedBuf { ptr: p as *mut T, len })
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: len initialized elements (zeroed at creation, Pod)
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as above, exclusive borrow
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T: Pod> Drop for PinnedBuf<T> {
    fn drop(&mut self) {
        // SAFETY: owned allocation (cudaFreeHost synchronizes); errors ignored in Drop
        unsafe {
            cudaFreeHost(self.ptr as *mut c_void);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lane::Compute;

    #[test]
    fn carve_aligns_and_overflow_is_an_error() {
        let a = Arena::host_for_test(4096);
        let x = a.alloc::<Compute, u32>("x", 3).unwrap();
        let y = a.alloc::<Compute, u64>("y", 5).unwrap();
        assert_eq!(x.dptr() as usize % ARENA_ALIGN, 0);
        assert_eq!(y.dptr() as usize - x.dptr() as usize, ARENA_ALIGN);
        assert_eq!(a.used_bytes(), ARENA_ALIGN + 40);
        assert_eq!(Arena::bytes_for::<u32>(3), ARENA_ALIGN);
        assert_eq!(Arena::bytes_for::<u64>(64), 2 * ARENA_ALIGN);
        match a.alloc::<Compute, u8>("big", 4096) {
            Err(Error::ArenaFull { what: "big", .. }) => {}
            _ => panic!("expected ArenaFull"),
        }
        let (lo, hi) = a.alloc::<Compute, u32>("s", 10).unwrap().split(4).unwrap();
        assert_eq!((lo.len(), hi.len()), (4, 6));
        assert_eq!(hi.dptr() as usize - lo.dptr() as usize, 16);
    }

    #[test]
    #[should_panic(expected = "arena is frozen")]
    fn alloc_after_freeze_panics() {
        let a = Arena::host_for_test(4096);
        let _x = a.alloc::<Compute, u32>("before", 16).unwrap();
        a.freeze();
        let _y = a.alloc::<Compute, u32>("after", 16);
    }
}
