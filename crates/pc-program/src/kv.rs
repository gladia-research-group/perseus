//! KV cache residency (seam `KvStore`): KV pages in the pinned arena, H2D/D2H on the `Copy` lane with `Ready`
//! tokens, per layer `join(i)`, `prefetch(i + 1)`, run block i, `store(i)`, `finalize(i - 1)` (the perseus
//! `driver.py::_with_kv_residency` choreography without device syncs). Ordering is carried by the two streams and
//! events only (`DBuf::handoff`, `Ready::join`); the host never waits (`wait_stored` is for checkpoints and tests).
//!
//! Implementations: `PinnedKv` (every layer's live rows in pinned host pages, `n_dev` device pages reused round
//! robin: long contexts) and `DeviceKv` (every layer resident on the device: small contexts; the choreography
//! calls are no-ops). The choreography diagram is in docs/ARCHITECTURE.md "pc-program kv".
use crate::error::{Error, Result};
use pc_cuda::{lane, Arena, Compute, Device, Event, EventPool, PinnedBuf, Ready};
use pc_gpu::{Ct, CtBuf, DevWord, Executor, Layout};
use pc_shadow::CtMeta;

fn kerr(msg: impl Into<String>) -> Error {
    Error::Replay { block: "kv".into(), msg: msg.into() }
}

/// Seam `KvStore`: where a layer's cache ciphertext lives between the blocks that read it.
pub trait KvStore<'a, W: DevWord> {
    /// Layers held.
    fn layers(&self) -> usize;
    /// Enqueue layer `i`'s reload into its device page (Copy lane; a no-op for a layer never stored).
    fn prefetch(&mut self, ex: &Executor<'a, W>, i: usize) -> Result<()>;
    /// Make compute wait for layer `i`'s reload; afterwards `page(i)` is readable / writable on compute.
    fn join(&mut self, ex: &Executor<'a, W>, i: usize) -> Result<()>;
    /// Layer `i`'s device ciphertext (joined).
    fn page(&mut self, i: usize) -> Result<&mut Ct<'a, W>>;
    /// Enqueue layer `i`'s copy-out behind the compute issued so far (Copy waits a compute event).
    fn store(&mut self, ex: &Executor<'a, W>, i: usize) -> Result<()>;
    /// Layer `i` is parked on the host from now on (its device page may be refilled by a later prefetch).
    fn finalize(&mut self, ex: &Executor<'a, W>, i: usize) -> Result<()>;
}

/// Every layer resident on the device (small contexts): one ciphertext per layer, the choreography is free.
pub struct DeviceKv<'a, W: DevWord> {
    pages: Vec<Ct<'a, W>>,
}

impl<'a, W: DevWord> DeviceKv<'a, W> {
    pub fn bytes(lay: &Layout, layers: usize) -> usize {
        layers * Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(lay, 1))
    }
    pub fn new(arena: &'a Arena, lay: Layout, layers: usize) -> Result<Self> {
        Ok(DeviceKv { pages: (0..layers).map(|_| Ok(Ct::new(CtMeta::new(0, 1), CtBuf::new(arena, "kv.device", lay, 1)?))).collect::<Result<_>>()? })
    }
}

impl<'a, W: DevWord> KvStore<'a, W> for DeviceKv<'a, W> {
    fn layers(&self) -> usize {
        self.pages.len()
    }
    fn prefetch(&mut self, _ex: &Executor<'a, W>, _i: usize) -> Result<()> {
        Ok(())
    }
    fn join(&mut self, _ex: &Executor<'a, W>, _i: usize) -> Result<()> {
        Ok(())
    }
    fn page(&mut self, i: usize) -> Result<&mut Ct<'a, W>> {
        let n = self.pages.len();
        self.pages.get_mut(i).ok_or_else(|| kerr(format!("layer {i} of {n}")))
    }
    fn store(&mut self, _ex: &Executor<'a, W>, _i: usize) -> Result<()> {
        Ok(())
    }
    fn finalize(&mut self, _ex: &Executor<'a, W>, _i: usize) -> Result<()> {
        Ok(())
    }
}

/// A device page and where it is.
enum Page<'a, 'e, W: DevWord> {
    /// On compute (holding `layer`, or nothing yet).
    Compute(Ct<'a, W>, Option<usize>),
    /// An H2D of `layer` in flight on Copy (joined by `join`).
    Loading(Ready<'e, lane::Copy, CtBuf<'a, lane::Copy, W>>, usize),
    /// On Copy after the D2H of `layer` (a later prefetch on Copy reuses it in stream order).
    Parked(CtBuf<'a, lane::Copy, W>, usize),
    Moving,
}

/// Layer state (host bookkeeping).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerState {
    /// Never stored (no host copy yet).
    Empty,
    /// Reloading or resident in a device page.
    OnDevice,
    /// Copy-out enqueued.
    Storing,
    /// Host copy authoritative; the device page may be reused.
    Parked,
}

/// Pinned host pages per layer (live Q rows `0..rows` of both parts), `n_dev` device pages (layer i uses page
/// `i % n_dev`).
pub struct PinnedKv<'a, 'e, W: DevWord> {
    lay: Layout,
    rows: usize,
    host: Vec<[PinnedBuf<W>; 2]>,
    metas: Vec<CtMeta>,
    state: Vec<LayerState>,
    pages: Vec<Page<'a, 'e, W>>,
    /// D2H completion of each layer (host checkpoints only).
    stored: Vec<Event>,
    /// Whether the layer has a host copy (stored at least once).
    ever_stored: Vec<bool>,
    hand: Event,
    events: &'e EventPool,
}

impl<'a, 'e, W: DevWord> PinnedKv<'a, 'e, W> {
    /// Device bytes (MemPlan section "kv").
    pub fn bytes(lay: &Layout, n_dev: usize) -> usize {
        n_dev * Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(lay, 1))
    }
    /// Pinned host bytes (the pinned arena's KV pages).
    pub fn host_bytes(lay: &Layout, layers: usize, rows: usize) -> usize {
        layers * 2 * rows * lay.n * std::mem::size_of::<W>()
    }
    /// `layers` caches of one ciphertext each, live rows `0..rows` (the cache level's top + 1), `n_dev` device
    /// pages (>= 2: a prefetch overlaps the compute of the layer before it). Setup: allocates and first-touches
    /// the pinned pages (zero-filled by the allocating thread).
    pub fn new(arena: &'a Arena, dev: &Device, lay: Layout, layers: usize, rows: usize, n_dev: usize, events: &'e EventPool) -> Result<Self> {
        if n_dev < 2 || layers == 0 || rows == 0 || rows > lay.nq {
            return Err(Error::Config { knob: "kv", msg: format!("layers {layers}, rows {rows} (<= {}), device pages {n_dev} (>= 2)", lay.nq) });
        }
        let host = (0..layers).map(|_| Ok([PinnedBuf::new(dev, rows * lay.n)?, PinnedBuf::new(dev, rows * lay.n)?])).collect::<std::result::Result<Vec<_>, pc_cuda::Error>>()?;
        let pages = (0..n_dev).map(|_| Ok(Page::Compute(Ct::new(CtMeta::new(0, 1), CtBuf::new(arena, "kv.page", lay, 1)?), None))).collect::<Result<Vec<_>>>()?;
        let stored = (0..layers).map(|_| Event::new(dev)).collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(PinnedKv { lay, rows, host, metas: vec![CtMeta::new(0, 1); layers], state: vec![LayerState::Empty; layers], pages, stored, ever_stored: vec![false; layers], hand: Event::new(dev)?, events })
    }
    fn page_of(&self, i: usize) -> usize {
        i % self.pages.len()
    }
    pub fn state(&self, i: usize) -> LayerState {
        self.state[i]
    }
    /// Host wait for layer `i`'s copy-out (checkpoints / tests; the choreography never calls it).
    pub fn wait_stored(&self, i: usize) -> Result<()> {
        Ok(self.stored[i].sync()?)
    }
    /// Host rows of a stored layer (after `wait_stored`): part 0, part 1, each `[rows][N]`.
    pub fn host_rows(&self, i: usize) -> (&[W], &[W]) {
        (self.host[i][0].as_slice(), self.host[i][1].as_slice())
    }
    /// The page buffer on the Copy lane (from wherever it is), for layer `want`. A layer that only sat in the page
    /// (reloaded, maybe read, never written since its last copy-out) is parked again: its host copy is the state (a
    /// reload nobody joined, e.g. a cache key the next token's stream no longer reads, must not stay `OnDevice`).
    fn to_copy(&mut self, p: usize, ex: &Executor<'a, W>, want: usize) -> Result<CtBuf<'a, lane::Copy, W>> {
        let holder = match &self.pages[p] {
            Page::Compute(_, l) => *l,
            Page::Loading(_, l) | Page::Parked(_, l) => Some(*l),
            Page::Moving => None,
        };
        if let Some(l) = holder.filter(|&l| l != want) {
            if self.state[l] == LayerState::OnDevice {
                self.state[l] = if self.ever_stored[l] { LayerState::Parked } else { LayerState::Empty };
            }
        }
        match std::mem::replace(&mut self.pages[p], Page::Moving) {
            Page::Compute(c, _) => Ok(CtBuf { buf: c.buf.buf.handoff(&ex.compute, &ex.copy, &self.hand)?, k: 1, lay: self.lay, seeds: false, packed: false }),
            Page::Parked(b, _) => Ok(b),
            // a reload nobody joined: Copy is its producer lane, stream order covers it
            Page::Loading(r, _) => Ok(r.join_as_is(&ex.copy)?),
            Page::Moving => Err(kerr("device page in an inconsistent state")),
        }
    }
}

impl<'a, 'e, W: DevWord> KvStore<'a, W> for PinnedKv<'a, 'e, W> {
    fn layers(&self) -> usize {
        self.host.len()
    }
    fn prefetch(&mut self, ex: &Executor<'a, W>, i: usize) -> Result<()> {
        if i >= self.layers() || matches!(self.state[i], LayerState::Empty | LayerState::OnDevice) {
            return Ok(());
        }
        let p = self.page_of(i);
        let mut b = self.to_copy(p, ex, i)?;
        for part in 0..2 {
            let off = self.lay.offset(0, part, 0);
            // SAFETY: the pinned page is rewritten only by a later D2H of this layer, issued on the same Copy stream
            unsafe { b.buf.upload_pinned_async(&ex.copy, off, &self.host[i][part])? };
        }
        self.pages[p] = Page::Loading(Ready::record(b, &ex.copy, self.events.get()?)?, i);
        self.state[i] = LayerState::OnDevice;
        Ok(())
    }
    fn join(&mut self, ex: &Executor<'a, W>, i: usize) -> Result<()> {
        if i >= self.layers() {
            return Err(kerr(format!("layer {i} of {}", self.layers())));
        }
        let p = self.page_of(i);
        let lay = self.lay;
        let page = match std::mem::replace(&mut self.pages[p], Page::Moving) {
            Page::Loading(r, l) if l == i => Page::Compute(Ct::new(self.metas[i], r.join(&ex.compute)?), Some(i)),
            Page::Compute(c, l) if l == Some(i) => Page::Compute(c, l),
            Page::Parked(b, l) if l == i => Page::Compute(Ct::new(self.metas[i], CtBuf { buf: b.buf.handoff(&ex.copy, &ex.compute, &self.hand)?, k: 1, lay, seeds: false, packed: false }), Some(i)),
            other if self.state[i] == LayerState::Empty => {
                // a fresh layer: the compute fills the page (it must not hold a layer still storing)
                self.pages[p] = other;
                let b = self.to_copy(p, ex, i)?;
                self.state[i] = LayerState::OnDevice;
                Page::Compute(Ct::new(CtMeta::new(0, 1), CtBuf { buf: b.buf.handoff(&ex.copy, &ex.compute, &self.hand)?, k: 1, lay, seeds: false, packed: false }), Some(i))
            }
            other => {
                self.pages[p] = other;
                return Err(kerr(format!("layer {i}: its device page holds another layer (prefetch {i} first)")));
            }
        };
        self.pages[p] = page;
        Ok(())
    }
    fn page(&mut self, i: usize) -> Result<&mut Ct<'a, W>> {
        let p = self.page_of(i);
        match &mut self.pages[p] {
            Page::Compute(c, Some(l)) if *l == i => Ok(c),
            _ => Err(kerr(format!("layer {i} is not joined"))),
        }
    }
    fn store(&mut self, ex: &Executor<'a, W>, i: usize) -> Result<()> {
        let p = self.page_of(i);
        let meta = match &self.pages[p] {
            Page::Compute(c, Some(l)) if *l == i => c.meta,
            _ => return Err(kerr(format!("layer {i} is not on compute"))),
        };
        let top = (self.lay.nq - 1).checked_sub(meta.level as usize).ok_or_else(|| kerr(format!("layer {i} level {}", meta.level)))?;
        if top + 1 > self.rows {
            return Err(Error::Config { knob: "kv", msg: format!("layer {i} at level {} has {} rows, pages hold {}", meta.level, top + 1, self.rows) });
        }
        self.metas[i] = meta;
        self.ever_stored[i] = true;
        let b = self.to_copy(p, ex, i)?;
        for part in 0..2 {
            let off = self.lay.offset(0, part, 0);
            // SAFETY: the host reads the pinned page only after `stored[i]` (wait_stored); the next H2D from it is
            // on the same Copy stream after this copy
            unsafe { b.buf.download_pinned_async(&ex.copy, off, &mut self.host[i][part])? };
        }
        self.stored[i].record(&ex.copy)?;
        self.pages[p] = Page::Parked(b, i);
        self.state[i] = LayerState::Storing;
        Ok(())
    }
    fn finalize(&mut self, _ex: &Executor<'a, W>, i: usize) -> Result<()> {
        if i < self.layers() && self.state[i] == LayerState::Storing {
            self.state[i] = LayerState::Parked;
        }
        Ok(())
    }
}
