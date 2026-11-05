//! Ciphertext batch buffers (word-generic).
//!
//! Layout of a `CtBuf` with `k` ciphertexts: `[c][part][row][N]` words, `c < k`, `part` in {0 (c0), 1 (c1)}, and
//! row index == prime id over the FULL basis (Q ids `0..nq`, special ids `nq..nq+kp`), so every polynomial has room
//! for the extended basis Q u P and a level change never moves data. Kernels address rows through `RowSet`
//! (element-wise) or row tables (`ntt::RowTables`), and ciphertexts through `PolyRef` strides.
use crate::error::{shape, Result};
use pc_cuda::{Arena, DBuf, Lane, OnLane, Stream};
use pc_kernels::{PolyRef, RowSet};

/// Row geometry of every ciphertext buffer of a context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    /// Ring dimension N.
    pub n: usize,
    /// Q prime count of the full chain (`L + 1`).
    pub nq: usize,
    /// Special prime count |P|.
    pub kp: usize,
}

impl Layout {
    pub fn from_chain(chain: &pc_core::params::Chain) -> Self {
        Layout { n: chain.n, nq: chain.q.len(), kp: chain.p.len() }
    }
    /// Rows per polynomial (= prime ids).
    pub fn rows(&self) -> usize {
        self.nq + self.kp
    }
    pub fn poly_words(&self) -> usize {
        self.rows() * self.n
    }
    pub fn ct_words(&self) -> usize {
        2 * self.poly_words()
    }
    /// Q rows `0..=top` (a ciphertext whose top limb is `top`).
    pub fn rs_q(&self, top: usize) -> RowSet {
        RowSet { nq: (top + 1) as i32, np: 0, qmax: self.nq as i32, pad: 0 }
    }
    /// Q rows `0..=top` and every special row (the extended basis during key switching).
    pub fn rs_qp(&self, top: usize) -> RowSet {
        RowSet { nq: (top + 1) as i32, np: self.kp as i32, qmax: self.nq as i32, pad: 0 }
    }
    /// The special rows only.
    pub fn rs_p(&self) -> RowSet {
        RowSet { nq: 0, np: self.kp as i32, qmax: self.nq as i32, pad: 0 }
    }
    /// Word offset of (ciphertext `c`, part, row id) in a buffer of this layout.
    pub fn offset(&self, c: usize, part: usize, row: usize) -> usize {
        c * self.ct_words() + part * self.poly_words() + row * self.n
    }
}

/// `k` ciphertexts (2 polynomials each over the full row layout) in one lane-`S` buffer of words `W`.
pub struct CtBuf<'a, S: Lane, W: pc_cuda::Pod> {
    pub buf: DBuf<'a, S, W>,
    pub k: usize,
    pub lay: Layout,
    /// A seeded ciphertext (`new_seeded`): part 1 holds its rows' 8-word c1 seeds (`pc_core::keys::cts_expand_row`),
    /// not c1; only the kernels that draw c1 from them read it.
    pub seeds: bool,
    /// A seeded ciphertext with c0 packed (`pc_kernels::pack_row`, rows `pack_row_words(n)` apart): only
    /// `mult_sum_many` reads it.
    pub packed: bool,
}

impl<'a, S: Lane, W: pc_cuda::Pod> CtBuf<'a, S, W> {
    /// Carve `k` ciphertexts from the arena (setup only).
    pub fn new(arena: &'a Arena, what: &'static str, lay: Layout, k: usize) -> Result<Self> {
        Ok(CtBuf { buf: arena.alloc::<S, W>(what, k * lay.ct_words())?, k, lay, seeds: false, packed: false })
    }
    /// Carve one seeded ciphertext: c0's rows (packed: `pack_row_words(n)` each), then 8 seed words per row where c1
    /// would start (setup only).
    pub fn new_seeded(arena: &'a Arena, what: &'static str, lay: Layout, packed: bool) -> Result<Self> {
        Ok(CtBuf { buf: arena.alloc::<S, W>(what, Self::words_seeded(&lay, packed))?, k: 1, lay, seeds: true, packed })
    }
    /// Words a `k`-ciphertext buffer takes (MemPlan).
    pub fn words(lay: &Layout, k: usize) -> usize {
        k * lay.ct_words()
    }
    /// Words of one seeded ciphertext (`new_seeded`).
    pub fn words_seeded(lay: &Layout, packed: bool) -> usize {
        Self::c0_words(lay, packed) + 8 * lay.rows()
    }
    /// Words of a seeded ciphertext's c0 (where its seeds start).
    fn c0_words(lay: &Layout, packed: bool) -> usize {
        lay.rows() * if packed { pc_kernels::pack_row_words(lay.n) } else { lay.n }
    }
    /// Part 1 of the views: c1, or a seeded ciphertext's seeds.
    fn part_words(&self) -> usize {
        if self.seeds { Self::c0_words(&self.lay, self.packed) } else { self.lay.poly_words() }
    }
    /// Read view (the kernel only reads through it).
    pub fn r(&self) -> PolyRef {
        PolyRef {
            base: self.buf.dptr() as *mut std::ffi::c_void,
            ct_stride: self.lay.ct_words() as u64,
            part_stride: self.part_words() as u64,
        }
    }
    /// Write view (needs `&mut`: exclusive).
    pub fn w(&mut self) -> PolyRef {
        PolyRef {
            base: self.buf.vptr_mut(),
            ct_stride: self.lay.ct_words() as u64,
            part_stride: self.part_words() as u64,
        }
    }
    pub fn offset(&self, c: usize, part: usize, row: usize) -> usize {
        self.lay.offset(c, part, row)
    }
    fn check_row(&self, c: usize, part: usize, rows: &[usize]) -> Result<()> {
        if c >= self.k || part > 1 || (self.seeds && part == 1) || self.packed || rows.iter().any(|&r| r >= self.lay.rows()) {
            return Err(shape("CtBuf rows", format!("c {c} (k {}), part {part}, rows {rows:?} (layout {:?})", self.k, self.lay)));
        }
        Ok(())
    }
    /// Upload rows (ids `rows`) of one polynomial from a host array laid out `[rows][n]` (synchronous; tests/setup).
    pub fn upload_rows(&mut self, s: &Stream<S>, c: usize, part: usize, rows: &[usize], data: &[W]) -> Result<()> {
        self.check_row(c, part, rows)?;
        let n = self.lay.n;
        if data.len() != rows.len() * n {
            return Err(shape("CtBuf::upload_rows", format!("{} words for {} rows", data.len(), rows.len())));
        }
        for (j, &r) in rows.iter().enumerate() {
            let off = self.offset(c, part, r);
            self.buf.upload_at(s, off, &data[j * n..(j + 1) * n])?;
        }
        Ok(())
    }
    /// Upload a seeded ciphertext's c1 seeds (8 words per row, rows `0..seeds.len() / 8`).
    pub fn upload_seeds(&mut self, s: &Stream<S>, seeds: &[W]) -> Result<()> {
        if !self.seeds || seeds.len() % 8 != 0 || seeds.len() / 8 > self.lay.rows() {
            return Err(shape("CtBuf::upload_seeds", format!("{} words into a {}seeded buffer of {} rows", seeds.len(), if self.seeds { "" } else { "non-" }, self.lay.rows())));
        }
        Ok(self.buf.upload_at(s, self.part_words(), seeds)?)
    }
    /// Upload a packed seeded ciphertext's c0 (`pc_kernels::pack_row` rows, `pack_row_words(n)` words each).
    pub fn upload_packed(&mut self, s: &Stream<S>, rows: &[W]) -> Result<()> {
        if !self.packed || rows.len() % pc_kernels::pack_row_words(self.lay.n) != 0 || rows.len() > self.part_words() {
            return Err(shape("CtBuf::upload_packed", format!("{} words into a {}packed buffer of {} rows", rows.len(), if self.packed { "" } else { "non-" }, self.lay.rows())));
        }
        Ok(self.buf.upload_at(s, 0, rows)?)
    }
    /// Host copy `[rows][n]` of rows `rows` of one polynomial (syncs the lane).
    pub fn download_rows(&self, s: &Stream<S>, c: usize, part: usize, rows: &[usize]) -> Result<Vec<W>> {
        self.check_row(c, part, rows)?;
        let n = self.lay.n;
        let mut v = Vec::with_capacity(rows.len() * n);
        for &r in rows {
            v.extend(self.buf.download_range(s, self.offset(c, part, r), n)?);
        }
        Ok(v)
    }
}

impl<'a, S: Lane, W: pc_cuda::Pod> OnLane for CtBuf<'a, S, W> {
    type Lane = S;
    type On<U: Lane> = CtBuf<'a, U, W>;
    unsafe fn relane<U: Lane>(self) -> CtBuf<'a, U, W> {
        CtBuf { buf: self.buf.relane::<U>(), k: self.k, lay: self.lay, seeds: self.seeds, packed: self.packed }
    }
}
