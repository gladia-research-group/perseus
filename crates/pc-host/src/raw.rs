// Raw views of the launch arguments: row pointers of PolyRefs and the parallel loop over a kernel's rows. The
// arguments hold arena pointers (host memory on the host device) that the launch contract keeps valid while the work
// runs; tasks address disjoint rows, as the device grid's blocks do.
use pc_kernels::args::{PolyRef, RowSet, N};
use rayon::prelude::*;

/// Shares a launch's argument struct (raw pointers inside) with the pool's tasks.
pub struct Shared<'a, A: ?Sized>(pub &'a A);
// SAFETY: the tasks of one launch touch disjoint outputs and read inputs nobody writes during the launch (the device
// kernel's own race freedom, which the launch contract guarantees)
unsafe impl<A: ?Sized> Sync for Shared<'_, A> {}
unsafe impl<A: ?Sized> Send for Shared<'_, A> {}
impl<'a, A: ?Sized> Shared<'a, A> {
    /// The shared value (a method, so closures capture the whole `Shared`, not its field).
    #[inline(always)]
    pub fn get(&self) -> &'a A {
        self.0
    }
}

/// `f(a, t)` for every task t < n on the kernel pool.
pub fn par_for<A>(n: usize, a: &A, f: impl Fn(&A, usize) + Sync + Send) {
    let sh = Shared(a);
    let sh = &sh;
    pc_hostrt::par(|| (0..n).into_par_iter().for_each(|t| f(sh.get(), t)));
}

/// Row `row` of part `part` of ciphertext `c`.
#[inline(always)]
pub fn at<W>(r: &PolyRef, c: usize, part: usize, row: usize) -> *mut W {
    // SAFETY: offset arithmetic only (the launch contract bounds it)
    unsafe { (r.base as *mut W).add(c * r.ct_stride as usize + part * r.part_stride as usize + row * N) }
}

/// Row index (== prime id) of the y-th row of a row set.
#[inline(always)]
pub fn rs_row(rs: &RowSet, y: usize) -> usize {
    if y < rs.nq as usize {
        y
    } else {
        rs.qmax as usize + (y - rs.nq as usize)
    }
}

#[inline(always)]
pub fn rs_len(rs: &RowSet) -> usize {
    (rs.nq + rs.np) as usize
}

/// Element `i` of a W table.
#[inline(always)]
pub fn el<W: Copy>(p: *const std::ffi::c_void, i: usize) -> W {
    // SAFETY: table index within the launch contract
    unsafe { *(p as *const W).add(i) }
}

/// A zeroed scratch buffer of n elements from the calling thread's pool, returned to it on drop: kernel tasks take
/// row-sized buffers per launch, which the system allocator would otherwise map and unmap every time (>= 128 KB).
pub struct Buf<T: Copy> {
    v: Vec<u128>,
    n: usize,
    _t: std::marker::PhantomData<T>,
}

thread_local! {
    static POOL: std::cell::RefCell<Vec<Vec<u128>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// n zeroed elements (T: plain integers or floats, all-zero bits = 0).
pub fn buf<T: Copy>(n: usize) -> Buf<T> {
    assert!(std::mem::align_of::<T>() <= 16);
    let words = (n * std::mem::size_of::<T>()).div_ceil(16);
    let mut v = POOL.with(|p| {
        let mut p = p.borrow_mut();
        match p.iter().position(|b| b.capacity() >= words) {
            Some(i) => p.swap_remove(i),
            None => Vec::new(),
        }
    });
    v.clear();
    v.resize(words, 0);
    Buf { v, n, _t: std::marker::PhantomData }
}

impl<T: Copy> Drop for Buf<T> {
    fn drop(&mut self) {
        let v = std::mem::take(&mut self.v);
        POOL.with(|p| {
            let mut p = p.borrow_mut();
            if p.len() < 64 {
                p.push(v);
            }
        });
    }
}
impl<T: Copy> std::ops::Deref for Buf<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: the backing words hold n elements of T (size and alignment checked at creation), zero-initialized
        unsafe { std::slice::from_raw_parts(self.v.as_ptr() as *const T, self.n) }
    }
}
impl<T: Copy> std::ops::DerefMut for Buf<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as above, exclusively borrowed
        unsafe { std::slice::from_raw_parts_mut(self.v.as_mut_ptr() as *mut T, self.n) }
    }
}

/// Runs the closure on drop.
pub struct Defer<F: FnMut()>(pub F);
impl<F: FnMut()> Drop for Defer<F> {
    fn drop(&mut self) {
        (self.0)()
    }
}
