//! Memory plan: every device byte is sized at setup by `MemPlan::build`, allocated
//! as ONE arena (`MemPlan::allocate`, the only `cudaMalloc` of the runtime, via `pc_cuda::Arena`), carved, then
//! `Arena::freeze()`. Missing memory is a setup error naming the knob, never a runtime allocation.
//!
//! Sections owned by other modules (keys, plaintext cache, bts constants/work, PCMM/CCMM, KV) are added
//! with `MemPlan::reserve(knob, bytes)` before `allocate`; their owners carve them from the arena with the same
//! element counts (use `Arena::bytes_for::<T>(n)` per carve to size a section exactly).
//!
//! Eager mode keeps ciphertexts in a bounded `CtPool`: fixed size classes (`k` ciphertexts per slot), refcounted
//! handles, slots back on the free list when the last handle drops.
use crate::context::{check_chain, ConstCounts, DevStrategy, DevWord};
use crate::ct::Ct;
use crate::error::{Error, Result};
use crate::poly::{CtBuf, Layout};
use pc_core::params::Chain;
use pc_cuda::{Arena, Compute, Device};
use pc_shadow::CtMeta;
use std::cell::{RefCell, UnsafeCell};
use std::ops::Deref;
use std::rc::Rc;

/// One `CtPool` size class: `count` slots of `k` ciphertexts each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolClass {
    pub k: usize,
    pub count: usize,
}

/// The runtime's memory knobs (the `mem`/`exec` part of the typed Config; names are what errors print).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemCfg {
    /// Ciphertexts per launch (K-batching bound, `cfg.bts.k_max`); sizes row tables and workspaces.
    pub k_max: usize,
    /// Eager ciphertext pool classes.
    pub pool: Vec<PoolClass>,
    /// Pre-created events (`EventPool` size).
    pub events: usize,
    /// Executor scratch, in batches of `k_max` ciphertexts.
    pub scratch_cts: usize,
}

impl Default for MemCfg {
    fn default() -> Self {
        MemCfg { k_max: 8, pool: vec![PoolClass { k: 1, count: 16 }, PoolClass { k: 8, count: 2 }], events: 64, scratch_cts: 2 }
    }
}

/// A named section reserved by a later owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Section {
    pub knob: &'static str,
    pub bytes: usize,
}

/// Device bytes of every part of the runtime, in carve units (`Arena::bytes_for`).
#[derive(Clone, Debug)]
pub struct MemPlan {
    pub cfg: MemCfg,
    pub lay: Layout,
    pub word_bits: u32,
    /// Context constant tables (primes, NTT twiddles, row tables, KS scalars).
    pub context: usize,
    /// Key-switching workspace (`ks::KsWork`): the ModUp digits of `k_max` ciphertexts over the full row layout
    /// (`k_max * dnum` polynomials), the extended-basis inner-product outputs and the ModDown conversions
    /// (`k_max` ciphertexts each).
    pub ks_work: usize,
    /// Key-switching digits of the chain (the `ks_work` shape).
    pub ks_dnum: usize,
    /// Base-conversion input workspace: `k_max` coefficient-form polynomials (`k_max * rows * N` words).
    pub bconv_work: usize,
    /// Executor scratch: `scratch_cts * k_max` ciphertexts.
    pub scratch: usize,
    /// Eager ciphertext pool (sum over classes).
    pub ct_pool: usize,
    /// Sections reserved by later cards (`reserve`).
    pub sections: Vec<Section>,
}

impl MemPlan {
    /// Words per element count -> bytes, as the arena carves them.
    fn w<W>(n: usize) -> usize {
        Arena::bytes_for::<W>(n)
    }

    /// Size everything for `chain` at words `W` under `cfg` (the instantiation `M` only has to exist).
    pub fn build<W: DevWord, M: DevStrategy<W>>(chain: &Chain, cfg: MemCfg) -> Result<MemPlan> {
        check_chain::<W>(chain)?;
        if cfg.k_max == 0 {
            return Err(Error::Mem { knob: "MemCfg::k_max", value: 0, msg: "must be >= 1".into() });
        }
        if cfg.events == 0 {
            return Err(Error::Mem { knob: "MemCfg::events", value: 0, msg: "must be >= 1".into() });
        }
        if let Some(c) = cfg.pool.iter().find(|c| c.k == 0) {
            return Err(Error::Mem { knob: "MemCfg::pool", value: c.k, msg: "class with k = 0".into() });
        }
        let lay = Layout::from_chain(chain);
        let rows_n = lay.rows() * lay.n;
        let context = ConstCounts::of(&lay, cfg.k_max).bytes::<W>();
        let ks_work = crate::ks::KsWork::<W>::bytes(&lay, cfg.k_max, chain.dnum());
        let bconv_work = Self::w::<W>(cfg.k_max * rows_n);
        let scratch = Self::w::<W>(CtBuf::<Compute, W>::words(&lay, cfg.scratch_cts * cfg.k_max));
        let ct_pool = cfg.pool.iter().map(|c| c.count * Self::w::<W>(CtBuf::<Compute, W>::words(&lay, c.k))).sum();
        Ok(MemPlan { cfg, lay, word_bits: W::BITS, context, ks_work, ks_dnum: chain.dnum(), bconv_work, scratch, ct_pool, sections: Vec::new() })
    }

    /// Add a section owned by a later card (keys, pt cache, bts, matmul, KV), before `allocate`.
    pub fn reserve(&mut self, knob: &'static str, bytes: usize) {
        self.sections.push(Section { knob, bytes: bytes.div_ceil(pc_cuda::ARENA_ALIGN) * pc_cuda::ARENA_ALIGN });
    }
    pub fn section(&self, knob: &str) -> Option<usize> {
        self.sections.iter().find(|s| s.knob == knob).map(|s| s.bytes)
    }

    /// Every part by name, in bytes.
    pub fn parts(&self) -> Vec<(&'static str, usize)> {
        let mut v = vec![
            ("context", self.context),
            ("ks_work", self.ks_work),
            ("bconv_work", self.bconv_work),
            ("scratch", self.scratch),
            ("ct_pool", self.ct_pool),
        ];
        v.extend(self.sections.iter().map(|s| (s.knob, s.bytes)));
        v
    }
    pub fn total(&self) -> usize {
        self.parts().iter().map(|p| p.1).sum()
    }
    /// One line per part (MB), printed at setup.
    pub fn report(&self) -> String {
        let mb = |b: usize| b as f64 / (1u64 << 20) as f64;
        let mut s: String = self.parts().iter().map(|(k, b)| format!("{k} {:.1} MB, ", mb(*b))).collect();
        s.push_str(&format!("total {:.1} MB", mb(self.total())));
        s
    }

    /// THE device allocation: one arena of `total()` bytes. Refuses (naming the largest part) when the device does
    /// not have that much free memory.
    pub fn allocate(&self, dev: &Device) -> Result<Arena> {
        let total = self.total();
        let (free, _) = dev.mem_info()?;
        if total > free {
            let (k, b) = self.parts().into_iter().max_by_key(|p| p.1).unwrap_or(("context", 0));
            return Err(Error::Mem {
                knob: k,
                value: b,
                msg: format!("plan needs {total} B, device has {free} B free ({})", self.report()),
            });
        }
        Ok(Arena::new(dev, total)?)
    }
}

struct Class<'a, W: DevWord> {
    k: usize,
    /// Q rows per slot: 0 = the full layout; > 0 = a low class (`CtPool::add_low`).
    rows: usize,
    /// A low class of seeded slots (`CtBuf::new_seeded`).
    seeded: bool,
    slots: Vec<UnsafeCell<Ct<'a, W>>>,
    free: RefCell<Vec<usize>>,
}

/// Bounded pool of compute-lane ciphertext slots in fixed size classes.
///
/// Reuse ordering: slots are `CtBuf<Compute>` and the pool never hands a slot's buffer out by value, so every access
/// to a slot is issued on the compute stream; a slot returned when its last handle drops is next written by work
/// issued later on the same stream, which the stream orders after every earlier access (the compute-lane fence is
/// the stream order itself; host reads sync the compute stream).
pub struct CtPool<'a, W: DevWord> {
    classes: Vec<Class<'a, W>>,
}

impl<'a, W: DevWord> CtPool<'a, W> {
    /// Carve every slot of `plan.cfg.pool` (setup only).
    pub fn new(arena: &'a Arena, plan: &MemPlan) -> Result<Self> {
        let mut classes = Vec::new();
        for c in &plan.cfg.pool {
            let slots = (0..c.count)
                .map(|_| Ok(UnsafeCell::new(Ct::new(CtMeta::new(0, 1), CtBuf::new(arena, "ct_pool", plan.lay, c.k)?))))
                .collect::<Result<Vec<_>>>()?;
            classes.push(Class { k: c.k, rows: 0, seeded: false, slots, free: RefCell::new((0..c.count).rev().collect()) });
        }
        Ok(CtPool { classes })
    }

    /// The layout of a low slot: Q rows `0..rows` and no special rows.
    pub fn low_layout(lay: &Layout, rows: usize) -> Layout {
        Layout { n: lay.n, nq: rows, kp: 0 }
    }

    /// Bytes of `count` low slots of `rows` Q rows (the `MemPlan` section "ct_low"; "ct_low_seeded" when `seeded`,
    /// c0 packed when `packed`).
    pub fn low_bytes(lay: &Layout, count: usize, rows: usize, seeded: bool, packed: bool) -> usize {
        let low = Self::low_layout(lay, rows);
        let words = if seeded { CtBuf::<Compute, W>::words_seeded(&low, packed) } else { CtBuf::<Compute, W>::words(&low, 1) };
        // each slot is its own arena allocation, aligned to ARENA_ALIGN
        count * (words * std::mem::size_of::<W>()).div_ceil(pc_cuda::ARENA_ALIGN) * pc_cuda::ARENA_ALIGN
    }

    /// Carve `count` low slots (one ciphertext each, Q rows `0..rows`, no special rows) from the "ct_low" section
    /// (setup only). A low slot holds a ciphertext whose top limb is below `rows`; `acquire` never hands one out, only
    /// `acquire_low` does, and the caller keeps every op that would write rows past `rows - 1` away from it.
    /// `seeded`: seeded slots (c0 and the c1 seeds, `CtBuf::new_seeded`; c0 packed when `packed`) from the
    /// "ct_low_seeded" section instead.
    pub fn add_low(&mut self, arena: &'a Arena, lay: &Layout, count: usize, rows: usize, seeded: bool, packed: bool) -> Result<()> {
        if rows == 0 || rows > lay.nq {
            return Err(Error::Mem { knob: "ct_low rows", value: rows, msg: format!("must be in 1..={}", lay.nq) });
        }
        let low = Self::low_layout(lay, rows);
        let buf = || if seeded { CtBuf::new_seeded(arena, "ct_low_seeded", low, packed) } else { CtBuf::new(arena, "ct_low", low, 1) };
        let slots = (0..count).map(|_| Ok(UnsafeCell::new(Ct::new(CtMeta::new(0, 1), buf()?)))).collect::<Result<Vec<_>>>()?;
        self.classes.push(Class { k: 1, rows, seeded, slots, free: RefCell::new((0..count).rev().collect()) });
        Ok(())
    }

    /// A free low slot (seeded or not) for one ciphertext of top limb `top`, stamped with `meta` (the smallest such low
    /// class that holds it).
    pub fn acquire_low(&self, meta: CtMeta, top: usize, seeded: bool) -> Result<PoolCt<'_, 'a, W>> {
        let class = (0..self.classes.len())
            .filter(|&i| self.classes[i].rows > top && self.classes[i].seeded == seeded && !self.classes[i].free.borrow().is_empty())
            .min_by_key(|&i| self.classes[i].rows)
            .ok_or_else(|| Error::Mem { knob: if seeded { "ct_low_seeded" } else { "ct_low" }, value: top, msg: "no free low slot holds this top limb".into() })?;
        let c = &self.classes[class];
        let idx = c.free.borrow_mut().pop().expect("free slot");
        // SAFETY: idx was just taken off the free list: no other handle refers to this slot
        unsafe { (*c.slots[idx].get()).meta = meta };
        Ok(PoolCt { slot: Rc::new(Slot { pool: self, class, idx }) })
    }

    /// A free slot of the class with exactly `k` ciphertexts, stamped with `meta`. Contents are stale (not zeroed).
    pub fn acquire(&self, k: usize, meta: CtMeta) -> Result<PoolCt<'_, 'a, W>> {
        let class = self.classes.iter().position(|c| c.k == k && c.rows == 0).ok_or_else(|| Error::Mem {
            knob: "MemCfg::pool",
            value: k,
            msg: format!("no pool class of {k} ciphertexts (classes: {:?})", self.classes.iter().map(|c| c.k).collect::<Vec<_>>()),
        })?;
        let c = &self.classes[class];
        let idx = c.free.borrow_mut().pop().ok_or_else(|| Error::Mem {
            knob: "MemCfg::pool",
            value: c.slots.len(),
            msg: format!("all {} slots of class k = {k} in use", c.slots.len()),
        })?;
        let slot = Slot { pool: self, class, idx };
        // SAFETY: idx was just taken off the free list: no other handle refers to this slot
        unsafe { (*c.slots[idx].get()).meta = meta };
        Ok(PoolCt { slot: Rc::new(slot) })
    }

    /// Free slots of class `k` (None if there is no such class).
    pub fn free(&self, k: usize) -> Option<usize> {
        self.classes.iter().find(|c| c.k == k && c.rows == 0).map(|c| c.free.borrow().len())
    }
}

struct Slot<'p, 'a, W: DevWord> {
    pool: &'p CtPool<'a, W>,
    class: usize,
    idx: usize,
}
impl<W: DevWord> Drop for Slot<'_, '_, W> {
    fn drop(&mut self) {
        self.pool.classes[self.class].free.borrow_mut().push(self.idx);
    }
}

/// Refcounted handle to a pool slot. Clones share the slot (read access); `get_mut` gives write access only to the
/// sole handle. The slot returns to the pool when the last handle drops.
pub struct PoolCt<'p, 'a, W: DevWord> {
    slot: Rc<Slot<'p, 'a, W>>,
}

impl<W: DevWord> Clone for PoolCt<'_, '_, W> {
    fn clone(&self) -> Self {
        PoolCt { slot: Rc::clone(&self.slot) }
    }
}

impl<'p, 'a, W: DevWord> PoolCt<'p, 'a, W> {
    fn cell(&self) -> &UnsafeCell<Ct<'a, W>> {
        &self.slot.pool.classes[self.slot.class].slots[self.slot.idx]
    }
    /// Exclusive access when this is the only handle (copy-on-write is the caller's policy).
    pub fn get_mut(&mut self) -> Option<&mut Ct<'a, W>> {
        Rc::get_mut(&mut self.slot)?;
        // SAFETY: this is the only handle to the slot and the slot is off the free list, so no other reference to
        // the cell exists; the &mut borrows self exclusively for its lifetime
        Some(unsafe { &mut *self.cell().get() })
    }
    /// Whether the slot is a low slot (`CtPool::add_low`).
    pub fn is_low(&self) -> bool {
        self.slot.pool.classes[self.slot.class].rows != 0
    }
    /// Handles sharing this slot.
    pub fn refs(&self) -> usize {
        Rc::strong_count(&self.slot)
    }
}

impl<'a, W: DevWord> Deref for PoolCt<'_, 'a, W> {
    type Target = Ct<'a, W>;
    fn deref(&self) -> &Ct<'a, W> {
        // SAFETY: shared access; a &mut exists only through get_mut, which requires this to be the only handle and
        // borrows it exclusively
        unsafe { &*self.cell().get() }
    }
}
