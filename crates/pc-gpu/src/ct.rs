//! A ciphertext batch on the compute lane with its bookkeeping.
use crate::poly::CtBuf;
use pc_cuda::Compute;
use pc_shadow::CtMeta;

/// `buf.k` ciphertexts that share one `CtMeta` (same level, degree, scale, slots, tag): the unit a launch works on
/// (K ciphertexts per launch is how concurrency inside a block is expressed). Eager code uses `k = 1`.
///
/// `meta` is the single source of truth for which rows are live: Q rows `0..=top` with `top = |Q| - 1 - level`
/// (`pc_shadow::meta::top_limb`); ops execute the `pc_shadow::Recipe` for the op and set `meta = recipe.out`.
pub struct Ct<'a, W: pc_cuda::Pod> {
    pub meta: CtMeta,
    pub buf: CtBuf<'a, Compute, W>,
}

impl<'a, W: pc_cuda::Pod> Ct<'a, W> {
    pub fn new(meta: CtMeta, buf: CtBuf<'a, Compute, W>) -> Self {
        Ct { meta, buf }
    }
    pub fn k(&self) -> usize {
        self.buf.k
    }
    /// FIDESlib top limb index of the live Q rows.
    pub fn top(&self, shape: pc_shadow::ChainShape) -> crate::Result<usize> {
        Ok(pc_shadow::meta::top_limb(shape, self.meta.level)? as usize)
    }
}

/// Single ciphertexts of layout `lay` that lie back to back in memory, member c one ciphertext past member c - 1
/// (consecutive slots of one slab), as one `cs.len()`-ciphertext buffer with the first member's metadata: a K-batch
/// reads or writes them in place instead of through a gather buffer. None when they do not.
///
/// # Safety
/// The view aliases the members' buffers: the caller does not use a member while it uses the view.
pub unsafe fn run_view<'a, W: pc_cuda::Pod>(lay: crate::Layout, cs: &[&Ct<'a, W>]) -> Option<Ct<'a, W>> {
    let first = cs.first()?;
    let words = lay.ct_words();
    let base = first.buf.buf.dptr() as *mut W;
    let at = |c: usize| base.wrapping_add(c * words) as *const W;
    if cs.len() < 2 || cs.iter().enumerate().any(|(c, x)| x.k() != 1 || x.buf.lay != lay || x.buf.buf.len() < words || x.buf.buf.dptr() != at(c)) {
        return None;
    }
    let buf = pc_cuda::DBuf::from_raw(base, cs.len() * words);
    Some(Ct::new(first.meta, CtBuf { buf, k: cs.len(), lay }))
}
