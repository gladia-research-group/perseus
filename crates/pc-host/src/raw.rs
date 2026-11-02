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
