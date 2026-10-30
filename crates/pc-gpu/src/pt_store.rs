//! Plaintext store: the device encode cache keyed by `(name, level, form)` (perseus `inference.h`'s encode
//! cache + `stage_pts` / `adopt_pts` / `erase_pts`), with fixed address-stable device slots sized by the `MemPlan`
//! (no growth; a full store is an error naming `PtStoreCfg::slots` under `ResidentAll`, or evicts the least recently
//! used unpinned slot under `StageAhead`).
//!
//! Staging (`stage`): items are encoded on the host by a rayon pool of `encode_threads` workers straight into a ring
//! of pinned buffers, uploaded on the `Copy` lane, and parked as `Ready` tokens; the first use joins the token into
//! the compute lane (and NTTs the coefficient rows there). A ring buffer is rewritten only after the event recorded
//! behind its upload completed. Eager misses (`mult_cached`, `mult_slot_pt`) encode synchronously on the device.
use crate::context::{Context, DevStrategy, DevWord};
use crate::ct::Ct;
use crate::encode_dev::{host_rows, DevEncoder, EncodeAt, EncoderKind, Payload2, PtForm};
use crate::error::{shape, Error, Result};
use crate::executor::Executor;
use crate::mem::MemPlan;
use crate::ntt::{self, Dir, RowList};
use crate::ops::{Ops, Pt};
use crate::poly::Layout;
use pc_core::encode::{Complex, Encoder, Payload};
use pc_cuda::{lane, Arena, Compute, DBuf, Device, Event, EventPool, PinnedBuf, Ready, Stream};
use pc_shadow::meta::top_limb;
use pc_shadow::{PtMeta, Shadow};
use rayon::prelude::*;
use std::collections::HashMap;

/// Residency policy (seam `PtStorePolicy`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PtStorePolicy {
    /// A worker ring stages the next block's plaintexts while the current one runs; slots are recycled (LRU).
    StageAhead,
    /// Every plaintext stays resident (small models); running out of slots is an error.
    ResidentAll,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PtStoreCfg {
    /// Device plaintext slots (each holds rows 0..|Q| of one plaintext).
    pub slots: usize,
    /// Pinned staging buffers (uploads in flight).
    pub ring: usize,
    /// Host encode workers (`cfg.host.encode_threads`).
    pub encode_threads: usize,
    pub policy: PtStorePolicy,
    /// Encoder for eager misses.
    pub eager: EncoderKind,
    /// Rows per slot (0 = every Q prime): a program whose plaintexts all sit at or above some level needs only
    /// rows `0..=top(level)` (GPT-2 decode: 28 of 54), which halves the store.
    pub rows: usize,
    /// Slot size classes `(rows, count)` for the plaintexts a planned session stages (one class per row count its
    /// levels need); the remaining `slots - sum(count)` slots take `rows`. A key takes the smallest free slot it fits.
    pub classes: Vec<(usize, usize)>,
}

impl Default for PtStoreCfg {
    fn default() -> Self {
        PtStoreCfg { slots: 64, ring: 4, encode_threads: 4, policy: PtStorePolicy::StageAhead, eager: EncoderKind::DeviceFft, rows: 0, classes: Vec::new() }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PtKey {
    pub name: String,
    pub level: u32,
    pub form: PtForm,
}

impl PtKey {
    pub fn new(name: impl Into<String>, level: u32, form: PtForm) -> Self {
        PtKey { name: name.into(), level, form }
    }
}

/// Values to stage.
#[derive(Clone, Debug)]
pub enum StageValues {
    Slots { values: Vec<Complex>, slots: usize, payload: Payload },
    Coeffs(Vec<f64>),
}

impl StageValues {
    fn as_payload(&self) -> Payload2<'_> {
        match self {
            StageValues::Slots { values, slots, payload } => Payload2::Slots { values, slots: *slots, payload: *payload },
            StageValues::Coeffs(c) => Payload2::Coeffs(c),
        }
    }
}

/// One plaintext to stage: key, degree, scale (`None` = the level's scaling factor).
#[derive(Clone, Debug)]
pub struct StageItem {
    pub key: PtKey,
    pub values: StageValues,
    pub deg: u8,
    pub scale: Option<f64>,
}

enum State<'a, 'e, W: DevWord> {
    Free(DBuf<'a, Compute, W>),
    /// uploaded coefficient rows on the copy lane; `ntt` = transform on join (Eval form)
    Pending(Ready<'e, lane::Copy, DBuf<'a, lane::Copy, W>>, bool),
    Resident(DBuf<'a, Compute, W>),
    Moving,
}

struct Slot<'a, 'e, W: DevWord> {
    state: State<'a, 'e, W>,
    /// rows this slot holds (`0..rows`)
    rows: usize,
    key: Option<PtKey>,
    meta: PtMeta,
    last_use: u64,
}

/// A named slot plaintext (`set_slot_pt`): host values, encoded at a level on use.
#[derive(Clone, Debug)]
pub struct NamedPt {
    pub values: Vec<Complex>,
    pub slots: usize,
    pub payload: Payload,
    pub level: u32,
    pub deg: u8,
}

pub struct PtStore<'a, 'e, W: DevWord> {
    pub cfg: PtStoreCfg,
    lay: Layout,
    slots: Vec<Slot<'a, 'e, W>>,
    index: HashMap<PtKey, usize>,
    clock: u64,
    ring: Vec<PinnedBuf<W>>,
    ring_ev: Vec<Event>,
    ring_used: Vec<bool>,
    ring_next: usize,
    hand_ev: Event,
    events: &'e EventPool,
    pool: rayon::ThreadPool,
    host: Encoder,
    primes: Vec<u64>,
    lists: Vec<RowList<'a>>,
    named: HashMap<String, NamedPt>,
    /// device encoder for eager misses and the fused sums
    pub enc: DevEncoder<'a, W>,
    /// one scratch plaintext (mult_pt_sum / mult_coeff_sum rows encoded per term)
    tmp: DBuf<'a, Compute, W>,
}

fn slot_rows(lay: &Layout, cfg: &PtStoreCfg) -> usize {
    if cfg.rows == 0 {
        lay.nq
    } else {
        cfg.rows.min(lay.nq)
    }
}

/// Row capacity of every slot, smallest first (the classes, then the rest at `slot_rows`).
fn slot_caps(lay: &Layout, cfg: &PtStoreCfg) -> Result<Vec<usize>> {
    let mut caps = Vec::with_capacity(cfg.slots);
    for &(r, c) in &cfg.classes {
        if r == 0 || r > lay.nq {
            return Err(Error::Mem { knob: "PtStoreCfg::classes", value: r, msg: format!("a class of {r} rows on a {}-prime chain", lay.nq) });
        }
        caps.extend(std::iter::repeat(r).take(c));
    }
    if caps.len() > cfg.slots {
        return Err(Error::Mem { knob: "PtStoreCfg::slots", value: cfg.slots, msg: format!("fewer slots than the classes hold ({})", caps.len()) });
    }
    caps.resize(cfg.slots, slot_rows(lay, cfg));
    caps.sort_unstable();
    Ok(caps)
}

impl<'a, 'e, W: DevWord> PtStore<'a, 'e, W> {
    /// Reserve the store's device bytes (`MemPlan` sections "pt_store" and "encode").
    pub fn reserve(plan: &mut MemPlan, cfg: &PtStoreCfg) {
        let lay = plan.lay;
        let lists: usize = (0..lay.nq).map(|t| Arena::bytes_for::<u32>(t + 1) + Arena::bytes_for::<i32>(t + 1)).sum();
        // an invalid class list is refused by `new`; reserve what it would carve
        let caps = slot_caps(&lay, cfg).unwrap_or_else(|_| vec![lay.nq; cfg.slots]);
        let slots: usize = caps.iter().map(|&r| Arena::bytes_for::<W>(r * lay.n)).sum();
        plan.reserve("pt_store", slots + Arena::bytes_for::<W>(lay.nq * lay.n) + lists);
        plan.reserve("encode", DevEncoder::<W>::bytes(&lay));
    }

    /// Carve the slots and pinned ring (setup). `events` orders the staged uploads (`Ready` tokens).
    pub fn new<M: DevStrategy<W>>(arena: &'a Arena, dev: &Device, s: &Stream<Compute>, ctx: &Context<'_, W, M>, events: &'e EventPool, cfg: PtStoreCfg) -> Result<Self> {
        if cfg.slots == 0 || cfg.ring == 0 || cfg.encode_threads == 0 {
            return Err(Error::Mem { knob: "PtStoreCfg", value: 0, msg: format!("slots / ring / encode_threads must be >= 1 ({cfg:?})") });
        }
        let lay = ctx.lay;
        let caps = slot_caps(&lay, &cfg)?;
        let w = caps.last().copied().unwrap_or(0) * lay.n;
        let slots = caps
            .iter()
            .map(|&rows| Ok(Slot { state: State::Free(arena.alloc("pt_store.slot", rows * lay.n)?), rows, key: None, meta: PtMeta { level: 0, deg: 1, nf: 0.0, tag: Default::default() }, last_use: 0 }))
            .collect::<Result<Vec<_>>>()?;
        let ring = (0..cfg.ring).map(|_| PinnedBuf::new(dev, w)).collect::<std::result::Result<Vec<_>, _>>()?;
        let ring_ev = (0..cfg.ring).map(|_| Event::new(dev)).collect::<std::result::Result<Vec<_>, _>>()?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cfg.encode_threads)
            .build()
            .map_err(|e| shape("PtStoreCfg::encode_threads", format!("rayon pool: {e}")))?;
        let lists = (0..lay.nq)
            .map(|t| RowList::new(arena, s, &(0..=t).map(|l| (l as u32, l as i32)).collect::<Vec<_>>()))
            .collect::<Result<Vec<_>>>()?;
        let enc = DevEncoder::new(arena, dev, s, ctx)?;
        Ok(PtStore {
            ring_used: vec![false; cfg.ring],
            cfg,
            lay,
            slots,
            index: HashMap::new(),
            clock: 0,
            ring,
            ring_ev,
            ring_next: 0,
            hand_ev: Event::new(dev)?,
            events,
            pool,
            host: Encoder::new(&ctx.chain),
            primes: ctx.chain.primes(),
            lists,
            named: HashMap::new(),
            enc,
            tmp: arena.alloc("pt_store.tmp", lay.nq * lay.n)?,
        })
    }

    pub fn has(&self, key: &PtKey) -> bool {
        self.index.contains_key(key)
    }
    pub fn len(&self) -> usize {
        self.index.len()
    }
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    fn top(&self, sh: &Shadow, level: u32) -> Result<usize> {
        let top = top_limb(sh.cfg.chain, level)? as usize;
        let rows = self.slots.last().map_or(0, |s| s.rows);
        if top >= rows {
            return Err(Error::Mem { knob: "PtStoreCfg::rows", value: rows, msg: format!("a plaintext at level {level} needs {} rows", top + 1) });
        }
        Ok(top)
    }

    /// A slot for `key` (rows `0..=top`): its own, the smallest free one it fits, or (StageAhead) the least recently
    /// used resident one it fits.
    fn slot_for(&mut self, key: &PtKey, top: usize) -> Result<usize> {
        if let Some(&i) = self.index.get(key) {
            return Ok(i);
        }
        // slots are sorted by capacity: the first free one that fits is the smallest
        if let Some(i) = self.slots.iter().position(|s| s.key.is_none() && s.rows > top) {
            return Ok(i);
        }
        if self.cfg.policy == PtStorePolicy::ResidentAll {
            return Err(Error::Mem { knob: "PtStoreCfg::slots", value: self.cfg.slots, msg: format!("store full staging '{}' ({} rows)", key.name, top + 1) });
        }
        let i = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.rows > top && matches!(s.state, State::Resident(_) | State::Free(_)))
            .min_by_key(|(_, s)| s.last_use)
            .map(|(i, _)| i)
            .ok_or_else(|| Error::Mem { knob: "PtStoreCfg::slots", value: self.cfg.slots, msg: "every slot has an upload in flight".into() })?;
        if let Some(k) = self.slots[i].key.take() {
            self.index.remove(&k);
        }
        Ok(i)
    }

    fn take_compute_buf(&mut self, i: usize, compute: &Stream<Compute>) -> Result<DBuf<'a, Compute, W>> {
        match std::mem::replace(&mut self.slots[i].state, State::Moving) {
            State::Free(b) | State::Resident(b) => Ok(b),
            State::Pending(r, _) => Ok(r.join(compute)?),
            State::Moving => Err(shape("pt_store", "slot in an inconsistent state")),
        }
    }

    /// Encode `items` on the host pool into the pinned ring and upload them on the copy lane (asynchronous: the
    /// compute lane joins each on first use).
    pub fn stage(&mut self, ex: &Executor<'_, W>, sh: &Shadow, items: &[StageItem]) -> Result<()> {
        let st = sh.tables.as_ref().ok_or_else(|| shape("stage", "the Shadow has no scale tables"))?;
        for chunk in items.chunks(self.ring.len()) {
            // ring buffers for this chunk
            let idx: Vec<usize> = (0..chunk.len()).map(|j| (self.ring_next + j) % self.ring.len()).collect();
            self.ring_next = (self.ring_next + chunk.len()) % self.ring.len();
            for &r in &idx {
                if self.ring_used[r] {
                    self.ring_ev[r].sync()?;
                }
            }
            let mut ats = Vec::with_capacity(chunk.len());
            for it in chunk {
                let top = self.top(sh, it.key.level)?;
                let scale = match it.scale {
                    Some(s) => s,
                    None => st.sf(top as i32)?,
                };
                ats.push(EncodeAt { top, scale, deg: it.deg, form: it.key.form });
            }
            // host encode, one item per worker, straight into the pinned buffers
            let n = self.lay.n;
            let mut bufs: Vec<(usize, &mut [W])> = Vec::with_capacity(chunk.len());
            {
                let mut by: Vec<Option<&mut PinnedBuf<W>>> = self.ring.iter_mut().map(Some).collect();
                for &r in &idx {
                    let b = by[r].take().ok_or_else(|| shape("stage", "ring index reused in one chunk"))?;
                    bufs.push((r, b.as_mut_slice()));
                }
            }
            let (host, primes) = (&self.host, &self.primes);
            let res: Vec<Result<()>> = self.pool.install(|| {
                chunk
                    .par_iter()
                    .zip(ats.par_iter())
                    .zip(bufs.par_iter_mut())
                    .map(|((it, at), (_, buf))| {
                        let rows = host_rows(host, primes, it.values.as_payload(), at)?;
                        for (l, r) in rows.iter().enumerate() {
                            for (o, &x) in buf[l * n..(l + 1) * n].iter_mut().zip(r) {
                                *o = W::from_u64(x);
                            }
                        }
                        Ok(())
                    })
                    .collect()
            });
            drop(bufs);
            for r in res {
                r?;
            }
            for ((it, at), &r) in chunk.iter().zip(&ats).zip(&idx) {
                let i = self.slot_for(&it.key, at.top)?;
                let buf = self.take_compute_buf(i, &ex.compute)?;
                let mut cb = buf.handoff(&ex.compute, &ex.copy, &self.hand_ev)?;
                // SAFETY: the pinned buffer is rewritten only after ring_ev[r] (recorded right after) completed
                unsafe { cb.upload_pinned_async(&ex.copy, 0, &self.ring[r])? };
                self.ring_ev[r].record(&ex.copy)?;
                self.ring_used[r] = true;
                let ready = Ready::record(cb, &ex.copy, self.events.get()?)?;
                self.clock += 1;
                let slot = &mut self.slots[i];
                slot.state = State::Pending(ready, at.form == PtForm::Eval);
                slot.meta = PtMeta { level: it.key.level, deg: it.deg, nf: at.scale.powi(it.deg as i32), tag: Default::default() };
                slot.key = Some(it.key.clone());
                slot.last_use = self.clock;
                self.index.insert(it.key.clone(), i);
            }
        }
        Ok(())
    }

    /// Join staged items into the compute lane (and NTT the Eval ones there); resident items are untouched.
    pub fn adopt<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &Executor<'_, W>, keys: &[PtKey]) -> Result<()> {
        for k in keys {
            let i = *self.index.get(k).ok_or_else(|| shape("adopt", format!("'{}'@{} is not staged", k.name, k.level)))?;
            self.join(ctx, &ex.compute, i)?;
        }
        Ok(())
    }

    fn join<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, i: usize) -> Result<()> {
        if let State::Pending(..) = self.slots[i].state {
            let st = std::mem::replace(&mut self.slots[i].state, State::Moving);
            if let State::Pending(r, ntt) = st {
                let mut b = r.join(s)?;
                if ntt {
                    let top = top_limb(ctx.shape, self.slots[i].meta.level)? as usize;
                    ntt::transform_rows(ctx, s, Dir::Forward, &mut b, &self.lists[top])?;
                }
                self.slots[i].state = State::Resident(b);
            }
        }
        Ok(())
    }

    /// Drop a key (its slot becomes free; contents are left for the stream order to overwrite).
    pub fn evict(&mut self, ex: &Executor<'_, W>, key: &PtKey) -> Result<bool> {
        let Some(i) = self.index.remove(key) else { return Ok(false) };
        let b = self.take_compute_buf(i, &ex.compute)?;
        self.slots[i].state = State::Free(b);
        self.slots[i].key = None;
        Ok(true)
    }

    /// Insert device-form rows (rows `0..=top(meta.level)`, `[row][N]`, already in `key.form`) synchronously
    /// (imports, tests).
    pub fn insert_rows(&mut self, ex: &Executor<'_, W>, key: PtKey, meta: PtMeta, rows: &[W]) -> Result<()> {
        let n = self.lay.n;
        if rows.is_empty() || rows.len() % n != 0 {
            return Err(shape("insert_rows", format!("{} words is not whole rows of {n}", rows.len())));
        }
        let i = self.slot_for(&key, rows.len() / n - 1)?;
        let mut b = self.take_compute_buf(i, &ex.compute)?;
        let r = b.upload_at(&ex.compute, 0, rows);
        self.clock += 1;
        let slot = &mut self.slots[i];
        slot.state = State::Resident(b);
        r?;
        slot.meta = meta;
        slot.key = Some(key.clone());
        slot.last_use = self.clock;
        self.index.insert(key, i);
        Ok(())
    }

    /// A resident plaintext (joins a staged one first).
    pub fn get<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &Executor<'_, W>, key: &PtKey) -> Result<Pt<'_>> {
        let i = *self.index.get(key).ok_or_else(|| shape("pt_store", format!("miss: '{}'@{} {:?}", key.name, key.level, key.form)))?;
        self.join(ctx, &ex.compute, i)?;
        self.clock += 1;
        self.slots[i].last_use = self.clock;
        self.view_slot(i)
    }

    fn view_slot(&self, i: usize) -> Result<Pt<'_>> {
        match &self.slots[i].state {
            State::Resident(b) => Ok(Pt::of(self.slots[i].meta, b)),
            _ => Err(shape("pt_store", "plaintext not resident (adopt first)")),
        }
    }

    /// A view of a resident key (after `adopt`/`get`).
    pub fn view(&self, key: &PtKey) -> Result<Pt<'_>> {
        let i = *self.index.get(key).ok_or_else(|| shape("pt_store", format!("miss: '{}'@{}", key.name, key.level)))?;
        self.view_slot(i)
    }

    /// Words of one copy of a plaintext at `level` (rows `0..=top(level)`): what `copy_out` writes.
    pub fn words_at(&self, sh: &Shadow, level: u32) -> Result<usize> {
        Ok((self.top(sh, level)? + 1) * self.lay.n)
    }

    /// Copy resident `key`'s rows (`0..=top(level)`) into `dst` (device to device, on the compute lane after every
    /// earlier writer): a snapshot of its encoding, which `copy_in` puts back. Returns its meta.
    pub fn copy_out<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &Executor<'_, W>, sh: &Shadow, key: &PtKey, dst: &mut DBuf<'_, Compute, W>) -> Result<PtMeta> {
        let words = self.words_at(sh, key.level)?;
        let i = *self.index.get(key).ok_or_else(|| shape("copy_out", format!("'{}'@{} is not resident", key.name, key.level)))?;
        self.join(ctx, &ex.compute, i)?;
        match &self.slots[i].state {
            State::Resident(b) => dst.copy_from(&ex.compute, b, words)?,
            _ => return Err(shape("copy_out", "plaintext not resident")),
        }
        Ok(self.slots[i].meta)
    }

    /// Overwrite resident `key`'s slot IN PLACE with rows `copy_out` took (device to device, on the compute lane after
    /// every earlier reader): the slot keeps its address, so a captured graph that reads it stays valid.
    pub fn copy_in<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &Executor<'_, W>, sh: &Shadow, key: &PtKey, src: &DBuf<'_, Compute, W>, meta: PtMeta) -> Result<()> {
        let words = self.words_at(sh, key.level)?;
        if meta.level != key.level {
            return Err(shape("copy_in", format!("a copy at level {} into '{}'@{}", meta.level, key.name, key.level)));
        }
        let i = *self.index.get(key).ok_or_else(|| shape("copy_in", format!("'{}'@{} is not resident", key.name, key.level)))?;
        self.join(ctx, &ex.compute, i)?;
        match &mut self.slots[i].state {
            State::Resident(b) => b.copy_from(&ex.compute, src, words)?,
            _ => return Err(shape("copy_in", "plaintext not resident")),
        }
        self.clock += 1;
        let slot = &mut self.slots[i];
        slot.meta = meta;
        slot.last_use = self.clock;
        Ok(())
    }

    /// Encode `values` on the device (eager) into the slot of `key`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_into<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &Executor<'_, W>, ops: &mut Ops<'_, W>, sh: &Shadow, key: PtKey, values: Payload2<'_>, deg: u8) -> Result<()> {
        let kind = match values {
            Payload2::Coeffs(_) => EncoderKind::CoeffRows,
            _ => self.cfg.eager,
        };
        self.encode_with(ctx, ex, ops, sh, kind, key, values, deg)
    }

    /// `encode_into` with an explicit encoder (seam contract checks).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_with<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &Executor<'_, W>, ops: &mut Ops<'_, W>, sh: &Shadow, kind: EncoderKind, key: PtKey, values: Payload2<'_>, deg: u8) -> Result<()> {
        let st = sh.tables.as_ref().ok_or_else(|| shape("encode", "the Shadow has no scale tables"))?;
        let top = self.top(sh, key.level)?;
        let scale = st.sf(top as i32)?;
        let at = EncodeAt { top, scale, deg, form: key.form };
        let i = self.slot_for(&key, top)?;
        let mut b = self.take_compute_buf(i, &ex.compute)?;
        let r = self.enc.encode(ctx, &ex.compute, &mut ops.ring, kind, values, at, &mut b);
        self.clock += 1;
        let slot = &mut self.slots[i];
        slot.state = State::Resident(b);
        r?;
        slot.meta = PtMeta { level: key.level, deg, nf: scale.powi(deg as i32), tag: Default::default() };
        slot.key = Some(key.clone());
        slot.last_use = self.clock;
        self.index.insert(key, i);
        Ok(())
    }

    // ── named slot plaintexts and cached products ────────────────────────────────────────────────────────────────

    /// `set_slot_pt`: keep `values` as the named plaintext, encoded (at `level`, or the level a product needs) on use.
    pub fn set_slot_pt(&mut self, name: &str, values: Vec<Complex>, slots: usize, payload: Payload, level: u32, deg: u8) {
        self.named.insert(name.to_string(), NamedPt { values, slots, payload, level, deg });
    }
    pub fn has_slot_pt(&self, name: &str) -> bool {
        self.named.contains_key(name)
    }

    /// The level and degree a plaintext needs to meet `ct` without a forced rescale (composite chain: the
    /// effective level at degree 1; the wrapper's `encode_like_params`).
    pub fn encode_like(sh: &Shadow, ct: &Ct<'_, W>) -> (u32, u8) {
        (ct.meta.effective_level(sh.unit()), 1)
    }

    /// Ensure `name` is resident at `level` (eager device encode of the named values on a miss).
    fn ensure<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &Executor<'_, W>, ops: &mut Ops<'_, W>, sh: &Shadow, name: &str, level: u32) -> Result<PtKey> {
        let key = PtKey::new(name, level, PtForm::Eval);
        if self.has(&key) {
            let i = self.index[&key];
            self.join(ctx, &ex.compute, i)?;
            self.clock += 1;
            self.slots[i].last_use = self.clock;
            return Ok(key);
        }
        let n = self.named.get(name).cloned().ok_or_else(|| shape("pt_store", format!("miss: '{name}'@{level} and no slot plaintext of that name (set_slot_pt / stage)")))?;
        self.encode_into(ctx, ex, ops, sh, key.clone(), Payload2::Slots { values: &n.values, slots: n.slots, payload: n.payload }, 1)?;
        Ok(key)
    }

    /// `ct * name` with the cached plaintext at `ct`'s level (lookup, or eager encode on a miss).
    #[allow(clippy::too_many_arguments)]
    pub fn mult_cached<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, ops: &mut Ops<'_, W>, sh: &Shadow, out: &mut Ct<'_, W>, ct: &Ct<'_, W>, name: &str) -> Result<()> {
        let (level, _) = Self::encode_like(sh, ct);
        let key = self.ensure(ctx, ex, ops, sh, name, level)?;
        let pt = self.view(&key)?;
        ops.mult_pt(ctx, ex, sh, std::slice::from_mut(out), &[ct], &[pt])
    }

    /// `ct * the named slot plaintext` (re-levelled to `ct` on a level mismatch, as `weights_at`).
    #[allow(clippy::too_many_arguments)]
    pub fn mult_slot_pt<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, ops: &mut Ops<'_, W>, sh: &Shadow, out: &mut Ct<'_, W>, ct: &Ct<'_, W>, name: &str) -> Result<()> {
        let set = self.named.get(name).map(|n| n.level).ok_or_else(|| shape("mult_slot_pt", format!("slot plaintext '{name}' is not set (set_slot_pt)")))?;
        let (need, _) = Self::encode_like(sh, ct);
        let level = if set == need { set } else { need };
        let key = self.ensure(ctx, ex, ops, sh, name, level)?;
        let pt = self.view(&key)?;
        ops.mult_pt(ctx, ex, sh, std::slice::from_mut(out), &[ct], &[pt])
    }

    /// `out[j] = ct * names[j]`.
    #[allow(clippy::too_many_arguments)]
    pub fn mult_slot_pt_many<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, ops: &mut Ops<'_, W>, sh: &Shadow, out: &mut [Ct<'_, W>], ct: &Ct<'_, W>, names: &[&str]) -> Result<()> {
        if out.len() != names.len() {
            return Err(shape("mult_slot_pt_many", format!("{} outputs for {} names", out.len(), names.len())));
        }
        for (o, n) in out.iter_mut().zip(names) {
            self.mult_slot_pt(ctx, ex, ops, sh, o, ct, n)?;
        }
        Ok(())
    }

    /// `out = sum_j cts[j] * values[j]`, every row encoded on the device at its ciphertext's level and folded into
    /// the accumulator (the loop's bookkeeping: `mult_pt`, then `add_inplace`).
    #[allow(clippy::too_many_arguments)]
    pub fn mult_pt_sum<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, ops: &mut Ops<'_, W>, sh: &Shadow, out: &mut Ct<'_, W>, cts: &[&Ct<'_, W>], values: &[Vec<Complex>], slots: usize, payload: Payload) -> Result<()> {
        let rows: Vec<Payload2<'_>> = values.iter().map(|v| Payload2::Slots { values: v, slots, payload }).collect();
        self.sum_with(ctx, ex, ops, sh, out, cts, &rows, self.cfg.eager)
    }

    /// `out = sum_j cts[j] * pt_j`, `pt_j`'s coefficients = `coeffs[j]` times the level's scaling factor (no FFT).
    #[allow(clippy::too_many_arguments)]
    pub fn mult_coeff_sum<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, ops: &mut Ops<'_, W>, sh: &Shadow, out: &mut Ct<'_, W>, cts: &[&Ct<'_, W>], coeffs: &[Vec<f64>]) -> Result<()> {
        let rows: Vec<Payload2<'_>> = coeffs.iter().map(|c| Payload2::Coeffs(c)).collect();
        self.sum_with(ctx, ex, ops, sh, out, cts, &rows, EncoderKind::CoeffRows)
    }

    #[allow(clippy::too_many_arguments)]
    fn sum_with<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, ops: &mut Ops<'_, W>, sh: &Shadow, out: &mut Ct<'_, W>, cts: &[&Ct<'_, W>], rows: &[Payload2<'_>], kind: EncoderKind) -> Result<()> {
        if cts.is_empty() || cts.len() != rows.len() {
            return Err(shape("mult_pt_sum", format!("{} ciphertexts, {} rows", cts.len(), rows.len())));
        }
        let st = sh.tables.as_ref().ok_or_else(|| shape("mult_pt_sum", "the Shadow has no scale tables"))?.clone();
        for (j, (ct, what)) in cts.iter().zip(rows).enumerate() {
            let (level, deg) = Self::encode_like(sh, ct);
            let top = self.top(sh, level)?;
            let scale = st.sf(top as i32)?;
            let at = EncodeAt { top, scale, deg, form: PtForm::Eval };
            // the scratch plaintext is rewritten per term: stream order puts the rewrite after the previous product
            self.enc.encode(ctx, &ex.compute, &mut ops.ring, kind, *what, at, &mut self.tmp)?;
            let pt = Pt::of(PtMeta { level, deg, nf: scale.powi(deg as i32), tag: Default::default() }, &self.tmp);
            if j == 0 {
                ops.mult_pt(ctx, ex, sh, std::slice::from_mut(out), &[*ct], &[pt])?;
            } else {
                ops.mult_add_many(ctx, ex, sh, out, &[*ct], &[pt])?;
            }
        }
        Ok(())
    }
}
