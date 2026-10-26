//! The dense route in StC-first order (`RouteKind::StcFirst`; FIDESlib FIDESLIB_BTS_STC_FIRST, Lattigo's
//! "DecodeThenModUp"): SlotsToCoeffs runs on the INPUT ciphertext before the raise, at limbs `entry .. 2d-1`, and
//! EvalMod's output is the bootstrap output:
//!   begin -> stc_first_input (StC stages; the "split", where a PCMM/CCMM consumes it) -> mod_raise -> fold ->
//!   CtS x4 -> EvalMod -> stc_first_output (x 2^CF, slot restore)
//! Mode 2 (shipped): the last StC stage carries ModRaise's FLEXIBLEAUTO adjust, so it depends on the call (CF, input
//! scale, raise variant) and is re-levelled on the device per key into a bounded LRU `StageCache` (FIDESlib
//! `relevelPlaintext`). Inputs deeper than the entry take the shipped order. Modes, the reference CF and the cache:
//! docs/ARCHITECTURE.md "pc-gpu StC-first route".
//!
//! On a lean chain (`ChainSpec::lean`) the StC stages run on the single primes above q0, one prime each: a stage's
//! plaintexts sit at that prime's scale times a factor (`THIN_SPLIT`; the last stage's per call), so the product keeps
//! the input's scale but for the factor, and the 2^-CF reduction is spread over the stages.
use super::cts::{diag_coeffs, stage_width, KeyTails, LtEnd, LtGeom, LtPts, LtStage, PtSource};
use super::dense::{DensePlan, LtHost};
use super::evalmod::{EmMode, EvalModPlan};
use super::finish::{pow2_exp, Pow2};
use super::raise::{self, small_int_tab, EncapsKeys, RaisePlan};
use super::stc_high::{StcHighPlan, StcHighSetup};
use super::{common_meta, no_stage, BtsCx, BtsKeyRing, BtsPrecomp, BtsRoute, BtsSetup, BtsState, BtsWorkShape, MemReq, RouteKind, Staged, StcHigh};
use crate::context::{Context, DevStrategy, DevWord};
use crate::ct::Ct;
use crate::error::{shape, Error, Result};
use crate::ks::{self, DotIo, DotRows, KeyGroup, KskSource, View};
use crate::ntt::{self, Dir, RowList};
use crate::poly::{CtBuf, Layout};
use pc_core::bts_setup::{cts_stc_factors, stc_first_mode2_factor, C64};
use pc_core::encode::Complex;
use pc_core::modmul::DevPrimeConst;
use pc_core::params::Chain;
use pc_core::scale::ScaleTables;
use pc_cuda::{launch_result, Arena, Compute, ConstBuf, DBuf, Device, Event, PinnedBuf, Stream};
use pc_kernels::{SmallIntArgs, SmallIntTab};
use pc_shadow::fides::FMeta;
use pc_shadow::{CtMeta, PtMeta, Route, Shadow};
use std::cell::RefCell;
use std::marker::PhantomData;

/// Scale of the re-levelling sources (2^60: any scale works, the division is exact; the integers stay far below the
/// 4-limb product).
pub const SRC_LOG2: i32 = 60;
/// Source limbs a re-levelled plaintext is reconstructed from (FIDESlib `relevelPlaintext`: 4).
pub const SRC_LIMBS: usize = 4;
/// A lean chain's single-prime StC stages: log2 of the fixed factors of the stages before the re-levelled last one.
/// The stages' diagonals have 64, 2048 and 65,536 nonzero coefficients (rounding 2^0.45 / 2^3.7 / 2^6.2 per unit of
/// scale), so the early stages take most of the reduction (measured on the 5090 against [-7, -3.5], [-8, -4], [-5, -3],
/// [-4, -2]: this split loses the least refresh precision against n32's paired stages, 0.2-0.4 bits at CF 8-13).
pub const THIN_SPLIT: [f64; 2] = [-6.0, -3.0];
/// The reference CF of a lean chain's last stage (the factors above are set for it; lower CFs are pre-scaled to it).
pub const THIN_CF_REF: u32 = 13;

/// The top limb of the mode-2 split (degree 2 there, where a PCMM / CCMM consumes it): 2d - 1, or a lean chain's
/// single prime above q0 (`StcFirstRoute::split_top` for the shipped StC budget).
pub fn split_top(chain: &Chain) -> usize {
    chain.lone_top().unwrap_or(2 * chain.level_unit - 1)
}

/// StC-first mode (FIDESLIB_BTS_STC_FIRST).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StcFirstMode {
    /// Every stage fixed; the last carries 2^-deg; the raise adjusts (non-prescaled).
    Plain,
    /// The last stage per call (CF, input scale, variant) carrying the raise adjust; the raise is prescaled.
    Folded,
}

/// The StC-first part of the bootstrap configuration (names of perseus' knobs in the field docs).
#[derive(Clone, Debug, PartialEq)]
pub struct StcFirstSetup {
    /// FIDESLIB_BTS_STC_FIRST (1 / 2).
    pub mode: StcFirstMode,
    /// The reference CF of the cached last stage (FIDESlib kStcFirstCfRef).
    pub cf_ref: u32,
    /// The route's default CF (FIDESLIB_BTS_STC_FIRST_CF; None = `BtsSetup::cf_default`). A `Route::cf` wins.
    pub cf_default: Option<u32>,
    /// Last-stage cache entries (the MemPlan's; FIDESLIB_STC_LAST_CACHE, perseus default 24, ~0.2 GB each).
    pub cache_slots: usize,
    /// `own_cf` of calls through the `Staged` trait (the staged API object passes it per call).
    pub own_cf: bool,
    /// StC outside a bootstrap (`stc_high`): the shapes and cache it serves; None = no stc_high.
    pub high: Option<StcHighSetup>,
}

impl StcFirstSetup {
    /// The shipped n32 StC-first configuration: mode 2, reference CF 20, route CF 13, 4 cache entries.
    pub fn n32_shipped() -> StcFirstSetup {
        StcFirstSetup { mode: StcFirstMode::Folded, cf_ref: 20, cf_default: Some(13), cache_slots: 4, own_cf: false, high: None }
    }
}

// ── re-levelled stage cache ──────────────────────────────────────────────────────────────────────────────────────

/// Words of a `SmallIntTab` (all 4-byte fields).
const TAB_WORDS: usize = std::mem::size_of::<SmallIntTab>() / 4;
const _: () = assert!(std::mem::size_of::<SmallIntTab>() % 4 == 0);

fn tab_words(t: &SmallIntTab) -> &[u32] {
    // SAFETY: SmallIntTab is repr(C) of i32/u32 fields and arrays only (size asserted a multiple of 4)
    unsafe { std::slice::from_raw_parts(t as *const SmallIntTab as *const u32, TAB_WORDS) }
}

/// A stage's diagonals on the host at 2^SRC_LOG2 (the re-levelling source).
#[derive(Clone, Debug)]
pub struct SrcHost {
    pub geom: LtGeom,
    pub coeffs: Vec<Vec<i128>>,
}

impl SrcHost {
    pub fn new(chain: &Chain, geom: LtGeom, values: &[Vec<C64>]) -> Result<SrcHost> {
        Ok(SrcHost { geom, coeffs: diag_coeffs(chain, values, (SRC_LOG2 as f64).exp2())? })
    }
    pub fn bytes<W: DevWord>(&self, n: usize) -> usize {
        Arena::bytes_for::<W>(self.coeffs.len() * SRC_LIMBS * n)
    }
}

/// A re-levelling source on the device: `[diagonal][4 bottom limbs][N]`, coefficient form.
pub struct RelevelSrc<'a, W: DevWord> {
    pub geom: LtGeom,
    rows: ConstBuf<'a, W>,
}

impl<'a, W: DevWord> RelevelSrc<'a, W> {
    pub fn new<M: DevStrategy<W>>(arena: &'a Arena, s: &Stream<Compute>, ctx: &Context<'_, W, M>, h: &SrcHost) -> Result<Self> {
        let n = ctx.lay.n;
        let q: Vec<u64> = (0..SRC_LIMBS).map(|i| ctx.prime(i)).collect();
        let mut w = Vec::with_capacity(h.coeffs.len() * SRC_LIMBS * n);
        for c in &h.coeffs {
            for &p in &q {
                w.extend(c.iter().map(|&x| W::from_u64(x.rem_euclid(p as i128) as u64)));
            }
        }
        Ok(RelevelSrc { geom: h.geom, rows: ConstBuf::new(arena, "bts.stc_first", s, &w)? })
    }
}

/// One re-levelled stage of a cache shape: its source, its input top, how it ends, whether it carries the factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageShape {
    pub src: usize,
    pub top: usize,
    pub end: LtEnd,
    pub scaled: bool,
}

/// What a cache entry was built for (opaque to the cache: the caller's rounding of its continuous parameters).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CacheKey(pub [i64; 3]);

/// The input-scale part of a mode-2 last-stage key: log2(nf) in units of 1e-6, so inputs whose scales differ by less
/// than ~7e-7 (relative) share an entry and the factor of the input that filled it.
pub fn nf_key(nf: f64) -> i64 {
    (nf.log2() * 1e6).round() as i64
}

struct Slot<'a, W: DevWord> {
    key: Option<CacheKey>,
    used: u64,
    stages: Vec<LtStage<'a, W>>,
    tabs: Vec<DBuf<'a, Compute, u32>>,
    pinned: Vec<PinnedBuf<u32>>,
    ntt: Vec<RowList<'a>>,
    /// recorded after the last fill's table copies (the pinned buffers' reuse waits on it)
    ev: Event,
}

/// Cache counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Distinct keys asked for so far (the entries a cache needs to never evict).
    pub keys: u64,
}

struct CacheState<'a, W: DevWord> {
    slots: Vec<Slot<'a, W>>,
    clock: u64,
    locked: bool,
    /// Every access refused (a capture must not bake an LRU entry that later calls evict: `ForbidLock`).
    forbid: bool,
    stats: CacheStats,
    seen: std::collections::HashSet<CacheKey>,
}

/// A bounded LRU cache of re-levelled LT stage sequences of one shape (every entry: the shape's stages at their
/// tops, plaintexts rebuilt in place on a miss). No growth: `slots` entries carved at setup.
pub struct StageCache<'a, W: DevWord> {
    pub what: &'static str,
    pub shape: Vec<StageShape>,
    /// per stage: the prime constants by row position (Q 0..=top, then P), for `small_int`'s output rows
    prs: Vec<ConstBuf<'a, DevPrimeConst>>,
    state: RefCell<CacheState<'a, W>>,
}

fn row_ids(lay: &Layout, top: usize) -> Vec<usize> {
    (0..=top).chain(lay.nq..lay.rows()).collect()
}

impl<'a, W: DevWord> StageCache<'a, W> {
    /// Device bytes of a cache of `slots` entries of `shape` over the sources `srcs`.
    pub fn bytes(lay: &Layout, primes: &[u64], shape: &[StageShape], srcs: &[LtGeom], slots: usize, k_max: usize) -> Result<usize> {
        let mut per = 0;
        let mut fixed = 0;
        for sh in shape {
            let g = srcs.get(sh.src).ok_or_else(|| shape_err(sh.src))?;
            let nrows = sh.top + 1 + lay.kp;
            per += LtPts::<W>::bytes(g.npts, nrows, lay.n) + LtStage::<W>::plan_bytes(lay, primes, g, sh.top, sh.end, k_max)?;
            per += Arena::bytes_for::<u32>(TAB_WORDS);
            per += Arena::bytes_for::<u32>(g.npts * nrows) + Arena::bytes_for::<i32>(g.npts * nrows);
            fixed += Arena::bytes_for::<DevPrimeConst>(nrows);
        }
        Ok(fixed + slots * per)
    }

    /// Carve `slots` entries (zero plaintexts; key tables and plans built once per entry).
    #[allow(clippy::too_many_arguments)]
    pub fn new<M: DevStrategy<W>>(
        what: &'static str,
        arena: &'a Arena,
        dev: &Device,
        s: &Stream<Compute>,
        ctx: &Context<'_, W, M>,
        shape: Vec<StageShape>,
        srcs: &[LtGeom],
        slots: usize,
        keys: &dyn BtsKeyRing,
        stage_ptrs: Option<(*const std::ffi::c_void, *const std::ffi::c_void)>,
        k_max: usize,
        lt_group: i32,
    ) -> Result<Self> {
        let lay = ctx.lay;
        let n = lay.n;
        if slots == 0 {
            return Err(Error::Mem { knob: "StcFirstSetup::cache_slots", value: 0, msg: format!("{what}: a re-levelled stage cache needs at least one entry") });
        }
        let mut prs = Vec::new();
        for sh in &shape {
            let pc: Vec<DevPrimeConst> = row_ids(&lay, sh.top).iter().map(|&i| DevPrimeConst::new(ctx.prime(i), W::BITS)).collect();
            prs.push(ConstBuf::new(arena, "bts.stc_first", s, &pc)?);
        }
        let mut sl = Vec::with_capacity(slots);
        for _ in 0..slots {
            let mut stages = Vec::new();
            let mut tabs = Vec::new();
            let mut pinned = Vec::new();
            let mut nttl = Vec::new();
            for sh in &shape {
                let g = srcs.get(sh.src).ok_or_else(|| shape_err(sh.src))?;
                let ids = row_ids(&lay, sh.top);
                let nrows = ids.len();
                let rows = vec![vec![0u32; nrows * n]; g.npts];
                let pts = LtPts::new(arena, s, ctx, sh.top, PtSource::Rows { rows, ptw: n, enc: 0xFFFF_FFFF })?;
                let pt_nf = ctx.scale.sf(sh.top as i32)?;
                let mut st = LtStage::new(arena, s, ctx, *g, sh.top, sh.end, pt_nf, pts, keys, stage_ptrs, k_max)?;
                st.group = lt_group;
                stages.push(st);
                tabs.push(arena.alloc::<Compute, u32>("bts.stc_first", TAB_WORDS)?);
                pinned.push(PinnedBuf::new(dev, TAB_WORDS)?);
                let list: Vec<(u32, i32)> = (0..g.npts).flat_map(|j| ids.iter().enumerate().map(move |(r, &id)| ((j * nrows + r) as u32, id as i32))).collect();
                nttl.push(RowList::new(arena, s, &list)?);
            }
            sl.push(Slot { key: None, used: 0, stages, tabs, pinned, ntt: nttl, ev: Event::new(dev)? });
        }
        Ok(StageCache { what, shape, prs, state: RefCell::new(CacheState { slots: sl, clock: 0, locked: false, forbid: false, stats: CacheStats::default(), seen: Default::default() }) })
    }

    /// Refuse misses (e.g. while a CUDA graph is captured: a fill inside a capture would replay a later table).
    pub fn set_locked(&self, on: bool) {
        self.state.borrow_mut().locked = on;
    }

    /// Refuse every access (`ForbidLock`, while a capture runs).
    pub fn set_forbidden(&self, on: bool) {
        self.state.borrow_mut().forbid = on;
    }

    pub fn stats(&self) -> CacheStats {
        self.state.borrow().stats
    }

    /// Entries the cache holds.
    pub fn slots(&self) -> usize {
        self.state.borrow().slots.len()
    }

    /// Keys held, most recently used first.
    pub fn keys(&self) -> Vec<CacheKey> {
        let st = self.state.borrow();
        let mut v: Vec<(u64, CacheKey)> = st.slots.iter().filter_map(|s| s.key.map(|k| (s.used, k))).collect();
        v.sort_by(|a, b| b.0.cmp(&a.0));
        v.into_iter().map(|x| x.1).collect()
    }

    /// The entry for `key`, built with `factor` on the scaled stages when missing (LRU eviction). Returns its index.
    pub fn get<M: DevStrategy<W>>(&self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, srcs: &[Option<RelevelSrc<'a, W>>], key: CacheKey, factor: f64) -> Result<usize> {
        let mut st = self.state.borrow_mut();
        if st.forbid {
            return Err(shape(self.what, format!("access to {key:?} while locked: a captured graph would bake an LRU entry (run the caller eager)")));
        }
        st.clock += 1;
        let clock = st.clock;
        if st.seen.insert(key) {
            st.stats.keys += 1;
        }
        if let Some(i) = st.slots.iter().position(|sl| sl.key == Some(key)) {
            st.slots[i].used = clock;
            st.stats.hits += 1;
            return Ok(i);
        }
        if st.locked {
            return Err(shape(self.what, format!("cache miss for {key:?} while locked (warm the key before capturing)")));
        }
        let i = st.slots.iter().enumerate().min_by_key(|(_, sl)| sl.used).map(|(i, _)| i).ok_or_else(|| shape(self.what, "no cache entries"))?;
        if st.slots[i].key.is_some() {
            st.stats.evictions += 1;
        }
        st.stats.misses += 1;
        let slot = &mut st.slots[i];
        slot.key = None;
        self.fill(ctx, s, srcs, slot, factor)?;
        slot.key = Some(key);
        slot.used = clock;
        Ok(i)
    }

    /// Run `f` on entry `i`'s stages.
    pub fn with<R>(&self, i: usize, f: impl FnOnce(&[LtStage<'a, W>]) -> Result<R>) -> Result<R> {
        let st = self.state.borrow();
        let slot = st.slots.get(i).ok_or_else(|| shape(self.what, format!("no entry {i}")))?;
        f(&slot.stages)
    }

    fn fill<M: DevStrategy<W>>(&self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, srcs: &[Option<RelevelSrc<'a, W>>], slot: &mut Slot<'a, W>, factor: f64) -> Result<()> {
        let lay = ctx.lay;
        let n = lay.n;
        let q4: Vec<u64> = (0..SRC_LIMBS).map(|i| ctx.prime(i)).collect();
        // the previous fill's table copies left the pinned buffers
        slot.ev.sync()?;
        for (e, sh) in self.shape.iter().enumerate() {
            let src = srcs.get(sh.src).and_then(|o| o.as_ref()).ok_or_else(|| shape_err(sh.src))?;
            let ids = row_ids(&lay, sh.top);
            let nrows = ids.len();
            let primes: Vec<u64> = ids.iter().map(|&i| ctx.prime(i)).collect();
            let pos: Vec<usize> = (0..nrows).collect();
            let f = if sh.scaled { factor } else { 1.0 };
            // integers x 2^SRC -> x f sf(top): scale by c = f sf(top) / 2^SRC
            let c = f * ctx.scale.sf(sh.top as i32)? * (-(SRC_LOG2 as f64)).exp2();
            let tab = small_int_tab(&q4, &pos, &primes, Some(c))?;
            slot.pinned[e].as_mut_slice().copy_from_slice(tab_words(&tab));
            // SAFETY: the pinned buffer is not written again before `ev` (recorded below) completes
            unsafe { slot.tabs[e].upload_pinned_async(s, 0, &slot.pinned[e])? };
            let npts = src.geom.npts;
            let out = slot.stages[e].pts.rows_mut();
            if out.len() < npts * nrows * n {
                return Err(shape(self.what, format!("entry plaintexts hold {} words, the stage needs {}", out.len(), npts * nrows * n)));
            }
            let a = SmallIntArgs {
                src: src.rows.dptr() as *const std::ffi::c_void,
                out: out.vptr_mut(),
                tab: slot.tabs[e].dptr() as *const SmallIntTab,
                pr: self.prs[e].dptr() as *const pc_kernels::PrimeConst,
                src_ct_stride: (SRC_LIMBS * n) as u64,
                src_part_stride: 0,
                ct_stride: (nrows * n) as u64,
                part_stride: 0,
                k: npts as i32,
                nparts: 1,
            };
            // SAFETY: npts sources of 4 rows and npts outputs of nrows rows inside their buffers; the table (uploaded
            // above on the same lane) maps output row r to the r-th prime of `prs[e]`; sources and outputs are distinct
            launch_result("small_int", unsafe { (ctx.table.small_int)(&a, s.raw()) })?;
            ntt::transform_rows(ctx, s, Dir::Forward, out, &slot.ntt[e])?;
        }
        slot.ev.record(s)?;
        Ok(())
    }
}

fn shape_err(i: usize) -> Error {
    shape("stage cache", format!("no re-levelling source {i}"))
}

/// One LT stage on every batch: pay a pending rescale, check the top, run, metadata through the Shadow.
pub fn run_lt<'a, W: DevWord, M: DevStrategy<W>>(cx: &mut BtsCx<'_, 'a, W, M>, stage: &LtStage<'a, W>, cts: &mut [Ct<'a, W>], source: KskSource) -> Result<()> {
    let ctx = cx.ctx;
    for x in cts.iter_mut() {
        if x.meta.deg == 2 {
            cx.ops.rescale(ctx, cx.ex, cx.sh, std::slice::from_mut(x))?;
        }
        let top = x.top(ctx.shape)?;
        if top != stage.top {
            return Err(shape("LT stage", format!("input top {top}, stage plaintexts at {}", stage.top)));
        }
        let k = x.k();
        stage.run(cx, &mut x.buf, k, source)?;
        let pt = PtMeta { level: ctx.shape.n_q - 1 - stage.top as u32, deg: 1, nf: stage.pt_nf, tag: Default::default() };
        let m = cx.sh.mult_pt(&x.meta, &pt)?.out;
        let (_, od) = stage.out_top_deg(ctx.chain.level_unit);
        x.meta = if od == 1 { cx.sh.rescale(&m)?.out } else { m };
    }
    Ok(())
}

/// One single-prime LT stage of a lean chain's StC-first input on every batch: the plaintexts' scale `pt_nf` is the
/// prime the stage drops times its factor, so the product keeps the input's scale but for that factor. `rescale`: drop
/// the stage's prime now (the last stage leaves it pending at the split, as the composite path leaves its pair).
fn run_lt_thin<'a, W: DevWord, M: DevStrategy<W>>(cx: &mut BtsCx<'_, 'a, W, M>, stage: &LtStage<'a, W>, pt_nf: f64, cts: &mut [Ct<'a, W>], source: KskSource, rescale: bool) -> Result<()> {
    let ctx = cx.ctx;
    for x in cts.iter_mut() {
        let top = x.top(ctx.shape)?;
        if x.meta.deg != 1 || top != stage.top {
            return Err(shape("thin LT stage", format!("input top {top} degree {}, stage plaintexts at {}", x.meta.deg, stage.top)));
        }
        let k = x.k();
        stage.run(cx, &mut x.buf, k, source)?;
        let nf = x.meta.nf * pt_nf;
        x.meta = if rescale {
            cx.ops.rescaler.rescale_one(ctx, &cx.ex.compute, &mut x.buf, top)?;
            CtMeta { level: x.meta.level + 1, deg: 1, nf: nf / ctx.chain.q[top] as f64, ..x.meta.clone() }
        } else {
            CtMeta { deg: 2, nf, ..x.meta.clone() }
        };
    }
    Ok(())
}

// ── host plan ────────────────────────────────────────────────────────────────────────────────────────────────────

/// The StC-first route's host plan: the dense variants (CtS, EvalMod, raise; the shipped StC for deep inputs and
/// stc_high), the fixed input-side stages, and the re-levelling sources.
pub struct StcFirstPlan {
    pub dense: DensePlan,
    pub sf: StcFirstSetup,
    /// Input limb the StC-first input is dropped to (FIDESlib `stc_first_entry`; n32 mode 2: 7 = level 46).
    pub entry_top: usize,
    /// Primes per StC stage: the level unit, or 1 on a lean chain.
    pub step: usize,
    /// Fixed input-side stages in execution order (mode 2: StC stages 0..n-2; mode 1: all, the last x 2^-deg).
    pub input: Vec<LtHost>,
    /// Re-levelling sources by StC stage (Some for the mode-2 last stage and every stage stc_high keeps).
    pub sources: Vec<Option<SrcHost>>,
}

impl StcFirstPlan {
    pub fn new(chain: &Chain, setup: &BtsSetup, sf: &StcFirstSetup) -> Result<StcFirstPlan> {
        let dense = DensePlan::new(chain, setup)?;
        let st = ScaleTables::precompute(chain)?;
        let d = chain.level_unit;
        let n = chain.n;
        let f = cts_stc_factors(n, (n / 2) as u32, setup.level_budget, setup.dim1, 1.0, super::stc::scale_dec(&st))?;
        let lay = f.fides_layout();
        let ns = lay.stc.len();
        if ns == 0 {
            return Err(shape("stc_first plan", "no StC stage"));
        }
        let nfixed = match sf.mode {
            StcFirstMode::Folded => ns - 1,
            StcFirstMode::Plain => ns,
        };
        // FIDESlib loader: entry = (2d - 1) + (mode 2 ? n - 1 : n) d; a lean chain: one prime per stage above q0
        let thin = chain.lone_top().is_some();
        if thin && (sf.mode != StcFirstMode::Folded || THIN_SPLIT.len() < nfixed) {
            return Err(shape("stc_first plan", format!("a lean chain runs mode 2 with at most {} fixed stages", THIN_SPLIT.len())));
        }
        let step = if thin { 1 } else { d };
        let entry_top = if thin { (d - 1) + ns } else { (2 * d - 1) + nfixed * d };
        if entry_top >= chain.q.len() {
            return Err(shape("stc_first plan", format!("entry limb {entry_top} above the chain top")));
        }
        let mut sf = sf.clone();
        if thin {
            sf.cf_ref = sf.cf_ref.min(THIN_CF_REF);
        }
        let mut input = Vec::new();
        for (si, fs) in lay.stc.iter().enumerate().take(nfixed) {
            let top = entry_top - si * step;
            let fac = if si + 1 == ns { (-(dense.deg as f64)).exp2() } else { 1.0 };
            let vals: Vec<Vec<C64>> = fs.pts.iter().map(|v| v.iter().map(|z| z.scale(fac)).collect()).collect();
            // a single-prime stage's plaintexts: the prime it drops times the stage's factor (the scale it keeps)
            let scale = if thin { chain.q[top] as f64 * THIN_SPLIT[si].exp2() } else { st.sf(top as i32)? };
            let coeffs = diag_coeffs(chain, &vals, scale)?;
            input.push(LtHost { geom: LtGeom::from_fides(fs), top, end: LtEnd::ModDown, nf: scale, width: stage_width(n, &coeffs), coeffs });
        }
        let mut need = vec![false; ns];
        if sf.mode == StcFirstMode::Folded {
            need[ns - 1] = true;
        }
        if let Some(h) = &sf.high {
            for shp in &h.shapes {
                for (s, nd) in need.iter_mut().enumerate() {
                    if shp.skip_mask >> s & 1 == 0 {
                        *nd = true;
                    }
                }
            }
        }
        let sources = lay.stc.iter().zip(&need).map(|(fs, &nd)| if nd { SrcHost::new(chain, LtGeom::from_fides(fs), &fs.pts).map(Some) } else { Ok(None) }).collect::<Result<_>>()?;
        Ok(StcFirstPlan { dense, sf, entry_top, step, input, sources })
    }

    fn src_geoms(&self) -> Vec<LtGeom> {
        // every StC stage's geometry, by index (absent sources keep their geometry for shape checks)
        let mut v = Vec::new();
        for (i, s) in self.sources.iter().enumerate() {
            v.push(match s {
                Some(h) => h.geom,
                None => self.dense.variants[0].stc.get(i).map(|h| h.geom).unwrap_or(LtGeom { b: 1, g: 1, stride: 0, npts: 0, offset: 0 }),
            });
        }
        v
    }

    /// The mode-2 last-stage cache shape.
    fn last_shape(&self) -> Option<Vec<StageShape>> {
        match self.sf.mode {
            StcFirstMode::Folded => Some(vec![StageShape { src: self.sources.len() - 1, top: self.split_top(), end: LtEnd::ModDown, scaled: true }]),
            StcFirstMode::Plain => None,
        }
    }

    /// The input top of the last StC stage, where the split leaves the ciphertexts at degree 2 (2d - 1; the single
    /// prime above q0 on a lean chain).
    pub fn split_top(&self) -> usize {
        self.entry_top - self.input.len() * self.step
    }

    /// Device bytes (MemPlan sections "bts.dense" and "bts.stc_first") and the work shape.
    pub fn mem<W: DevWord>(&self) -> Result<(MemReq, BtsWorkShape)> {
        let (mut r, ws) = self.dense.mem::<W>()?;
        let lay = Layout::from_chain(&self.dense.chain);
        let primes = self.dense.chain.primes();
        let k_max = self.dense.setup.k_max;
        let mut b = 0;
        for h in &self.input {
            b += LtPts::<W>::coeffs_bytes(h.coeffs.len(), h.top + 1 + lay.kp, h.width, lay.n) + LtStage::<W>::plan_bytes(&lay, &primes, &h.geom, h.top, h.end, k_max)?;
        }
        for s in self.sources.iter().flatten() {
            b += s.bytes::<W>(lay.n);
        }
        let geoms = self.src_geoms();
        if let Some(shp) = self.last_shape() {
            b += StageCache::<W>::bytes(&lay, &primes, &shp, &geoms, self.sf.cache_slots, k_max)?;
        }
        if let Some(h) = &self.sf.high {
            b += StcHighPlan::<W>::bytes(&lay, &primes, h, &geoms, self.dense.setup.level_budget[1] as usize, self.dense.chain.level_unit, k_max)?;
        }
        r.add("bts.stc_first", b);
        Ok((r, ws))
    }

    /// Device tables (after `MemPlan::allocate`, before `Arena::freeze`). `keys` must hold every CtS and StC
    /// rotation, conjugation, relinearization and both encapsulation keys.
    pub fn build<'a, W: DevWord, M: DevStrategy<W>>(
        &self,
        arena: &'a Arena,
        dev: &Device,
        s: &Stream<Compute>,
        ctx: &Context<'_, W, M>,
        keys: &dyn BtsKeyRing,
        stage_ptrs: Option<(*const std::ffi::c_void, *const std::ffi::c_void)>,
    ) -> Result<StcFirstRoute<'a, W, M>> {
        let setup = &self.dense.setup;
        let n = ctx.lay.n;
        let k_max = setup.k_max;
        // resident `a` tails of the CtS baby keys (sized in `DensePlan::mem`)
        let tails = KeyTails::new(arena, s, ctx, keys, &self.dense.tail_steps(), setup.lt_regen_digits)?;
        let ring = tails.ring(keys, n)?;
        let lt = |h: &LtHost| -> Result<LtStage<'a, W>> {
            let pts = LtPts::new(arena, s, ctx, h.top, PtSource::Coeffs(&h.coeffs))?;
            let mut st = LtStage::new(arena, s, ctx, h.geom, h.top, h.end, h.nf, pts, &ring, stage_ptrs, k_max)?;
            st.group = setup.lt_group;
            Ok(st)
        };
        let mut variants = Vec::new();
        for v in &self.dense.variants {
            let em_k = |mode| {
                let mut em = EvalModPlan::new(arena, s, ctx, mode, v.em_in, &self.dense.ps, setup.evalmod_da_its, keys.conj()?, keys.relin()?, stage_ptrs, k_max, setup.em_relin)?;
                em.fuse_rescale = setup.em_fuse_rescale;
                Ok::<_, crate::error::Error>(em)
            };
            variants.push(BtsPrecomp {
                slots: (n / 2) as u32,
                raise_drop: v.drop,
                raise: RaisePlan::new(arena, s, ctx, v.raise_top, v.exact, setup.cts0_t)?,
                cts: v.cts.iter().map(lt).collect::<Result<_>>()?,
                stc: v.stc.iter().map(lt).collect::<Result<_>>()?,
                stc_real0: v.stc_real0.as_ref().map(lt).transpose()?,
                em: em_k(EmMode::Complex)?,
                em_real: if setup.real { Some(em_k(EmMode::Real)?) } else { None },
                cts0_const: v.exact,
                cts0_t: setup.cts0_t,
                boot_k: setup.boot_k,
            });
        }
        let input = self.input.iter().map(lt).collect::<Result<Vec<_>>>()?;
        let srcs: Vec<Option<RelevelSrc<'a, W>>> = self.sources.iter().map(|o| o.as_ref().map(|h| RelevelSrc::new(arena, s, ctx, h)).transpose()).collect::<Result<_>>()?;
        let geoms = self.src_geoms();
        let last = match self.last_shape() {
            Some(shp) => Some(StageCache::new("stc_first last stage", arena, dev, s, ctx, shp, &geoms, self.sf.cache_slots, keys, stage_ptrs, k_max, setup.lt_group)?),
            None => None,
        };
        let high = match &self.sf.high {
            Some(h) => Some(StcHighPlan::new(arena, dev, s, ctx, h, &geoms, setup.level_budget[1] as usize, keys, stage_ptrs, k_max, setup.lt_group)?),
            None => None,
        };
        let (mem, work_shape) = self.mem::<W>()?;
        let mut route = StcFirstRoute::from_parts(
            setup.clone(),
            self.sf.clone(),
            variants,
            self.dense.variants.iter().map(|v| v.out).collect(),
            Some(EncapsKeys::new(arena, s, n, keys.dense_to_sparse()?, keys.sparse_to_dense()?, stage_ptrs)?),
            Some(KeyGroup::new(arena, s, n, &[keys.conj()?], stage_ptrs)?),
            Pow2::new(arena, s, ctx)?,
            self.dense.deg,
            self.entry_top,
            input,
            srcs,
            last,
            (mem, work_shape),
        );
        route.high = high;
        route.step = self.step;
        route.thin = self.dense.chain.lone_top().is_some();
        Ok(route)
    }
}

// ── the route ────────────────────────────────────────────────────────────────────────────────────────────────────

/// The StC-first dense route's device side.
pub struct StcFirstRoute<'a, W: DevWord, M: DevStrategy<W>> {
    pub setup: BtsSetup,
    pub sf: StcFirstSetup,
    /// Per raise drop: CtS, EvalMod, raise tables; the shipped StC stages (deep inputs, stc_high) when built.
    pub variants: Vec<BtsPrecomp<'a, W>>,
    /// Shipped-order output metadata per variant (deep inputs).
    outs_shipped: Vec<FMeta>,
    encaps: Option<EncapsKeys<'a>>,
    conj: Option<KeyGroup<'a>>,
    pow2: Pow2<'a, W>,
    /// log2(q0 / 2^scale_bits)
    pub deg: u32,
    pub entry_top: usize,
    /// Primes per StC stage (`StcFirstPlan::step`).
    pub step: usize,
    /// The input stages run on a lean chain's single primes (`run_lt_thin`).
    pub thin: bool,
    /// Fixed input-side stages.
    pub input: Vec<LtStage<'a, W>>,
    /// Re-levelling sources by StC stage.
    pub srcs: Vec<Option<RelevelSrc<'a, W>>>,
    /// Mode-2 last stage cache.
    pub last: Option<StageCache<'a, W>>,
    /// StC outside a bootstrap.
    pub high: Option<StcHighPlan<'a, W>>,
    mem: (MemReq, BtsWorkShape),
    _m: PhantomData<M>,
}

/// What `stc_first_input` did (the PCMM / CCMM read the split's scale and the cache key).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitInfo {
    /// The input's scale at the entry (after its pending rescale and the drop).
    pub nf_in: f64,
    /// CF the last stage was built at, and the integer put on the input first (2^(cf_stage - cf)).
    pub cf_stage: u32,
    pub pre_factor_log2: u32,
    /// The mode-2 last stage's factor and cache key.
    pub factor: f64,
    pub key: Option<CacheKey>,
}

impl<'a, W: DevWord, M: DevStrategy<W>> StcFirstRoute<'a, W, M> {
    /// Assemble a route from built parts (checks with dumped plaintexts replace `variants[v].cts`; a route without
    /// encapsulation keys can run every stage after the raise).
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        setup: BtsSetup,
        sf: StcFirstSetup,
        variants: Vec<BtsPrecomp<'a, W>>,
        outs_shipped: Vec<FMeta>,
        encaps: Option<EncapsKeys<'a>>,
        conj: Option<KeyGroup<'a>>,
        pow2: Pow2<'a, W>,
        deg: u32,
        entry_top: usize,
        input: Vec<LtStage<'a, W>>,
        srcs: Vec<Option<RelevelSrc<'a, W>>>,
        last: Option<StageCache<'a, W>>,
        mem: (MemReq, BtsWorkShape),
    ) -> Self {
        // the stage step of the input stages' tops (entry, entry - step, ...); a single stage leaves it unknown
        let step = match input.as_slice() {
            [a, b, ..] => a.top - b.top,
            _ => entry_top.saturating_sub(input.first().map_or(entry_top, |s| s.top)).max(1),
        };
        StcFirstRoute { setup, sf, variants, outs_shipped, encaps, conj, pow2, deg, entry_top, step, thin: false, input, srcs, last, high: None, mem, _m: PhantomData }
    }

    /// Entry level (primes dropped): the deepest EFFECTIVE input level that runs StC-first (pc-shadow
    /// `BtsConfig::stc_first_entry`, FAKECAP_STC_ENTRY).
    pub fn entry_level(&self, n_q: u32) -> u32 {
        n_q - 1 - self.entry_top as u32
    }

    /// The split's top limb (degree 2 there): the last StC stage's input top.
    pub fn split_top(&self) -> usize {
        self.entry_top - self.input.len() * self.step
    }

    /// The input stages run on single primes (a lean chain).
    pub fn thin(&self) -> bool {
        self.thin
    }

    /// The deepest level a degree-2 input may sit at and still reach the entry after its pending rescale: the PCMM's
    /// `level_hint` cap (perseus `Pcmm._measure_cap`: total - entry - unit).
    pub fn input_cap(&self, n_q: u32, unit: u32) -> u32 {
        self.entry_level(n_q) - unit
    }

    pub fn variant(&self, drop: u32) -> Result<usize> {
        self.variants.iter().position(|v| v.raise_drop == drop).ok_or_else(|| Error::Mem {
            knob: "BtsSetup::raise_drops",
            value: drop as usize,
            msg: format!("no StC-first variant for raise drop {drop} (built: {:?})", self.variants.iter().map(|v| v.raise_drop).collect::<Vec<_>>()),
        })
    }

    /// The call's CF: the route's, else the StC-first default, else the setup's.
    pub fn call_cf(&self, route: &Route) -> u32 {
        route.cf.or(self.sf.cf_default).unwrap_or(self.setup.cf_default)
    }

    /// Whether an input with metadata `m` runs StC-first (effective limb at or above the entry; FIDESlib `btsBegin`).
    pub fn runs_stc_first(&self, n_q: u32, unit: u32, m: &CtMeta) -> bool {
        let top = (n_q - 1).saturating_sub(m.level) as usize;
        let eff = if m.deg == 2 { top.saturating_sub(unit as usize) } else { top };
        m.level < n_q && eff >= self.entry_top
    }

    /// `btsBegin` on metadata (the PCMM takes its finish state from the INPUT's metadata, under the finish's raise
    /// variant: `Route::raise_drop`).
    pub fn begin_meta(&self, ctx: &Context<'_, W, M>, in_meta: &CtMeta, k: usize, route: &Route, prescaled: bool) -> Result<BtsState> {
        let n = ctx.lay.n;
        match RouteKind::of(route, n) {
            RouteKind::StcFirst | RouteKind::Dense => {}
            o => return Err(shape("stc_first route", format!("route {route:?} is {o:?}, not the dense route"))),
        }
        if k > self.setup.k_max {
            return Err(shape("stc_first route", format!("batch of {k} cts above BtsSetup::k_max {}", self.setup.k_max)));
        }
        let variant = self.variant(route.raise_drop)?;
        let cf = self.call_cf(route);
        if cf < self.setup.cf_window.0 || cf > self.setup.cf_window.1 || cf < self.deg {
            return Err(shape("bootstrap cf", format!("correction factor {cf} outside [{}, {}] (deg {})", self.setup.cf_window.0, self.setup.cf_window.1, self.deg)));
        }
        let shp = ctx.shape;
        let stc_first = route.stc_first && self.runs_stc_first(shp.n_q, shp.unit, in_meta);
        let pre = &self.variants[variant];
        let mut correction = cf - self.deg;
        let cor_factor = 1u64 << correction;
        let real = route.real && pre.em_real.is_some() && (stc_first || pre.stc_real0.is_some());
        if route.real && !real {
            return Err(shape("bootstrap real", "real-payload variant not built (BtsSetup::real)"));
        }
        // shipped order: the real payload enters as x + conj x (one more raise correction bit); StC-first: the input
        // StC already put (Re x ; Im x) in the coefficients, EvalMod's real chain alone returns Re x
        if real && !stc_first {
            correction += 1;
        }
        Ok(BtsState {
            route: *route,
            slots: (n / 2) as u32,
            old_slots: in_meta.slots,
            prescaled,
            cf,
            correction,
            cor_factor,
            scalar: 1,
            const_eval_mult: pre.cts0_const,
            baked: pre.cts0_const,
            exact_const: true,
            shifted_flow: false,
            sparse_encaps: true,
            sparse_b: false,
            stc_first,
            stc_folded: self.sf.mode == StcFirstMode::Folded,
            real,
            n_cts: pre.cts.len(),
            n_stc: pre.stc.len(),
            variant,
            raise_drop: route.raise_drop,
            in_meta: in_meta.clone(),
        })
    }

    fn set_meta(cx: &BtsCx<'_, 'a, W, M>, x: &mut Ct<'a, W>, f: FMeta) {
        let shp = cx.ctx.shape;
        x.meta = CtMeta { level: shp.n_q - 1 - f.top as u32, deg: f.nl as u8, nf: f.nf, ..x.meta.clone() };
    }

    /// x := x + conj(x) on every batch (shipped order, real payload).
    fn add_conj(&self, cx: &mut BtsCx<'_, 'a, W, M>, cts: &mut [Ct<'a, W>]) -> Result<()> {
        let conj = self.conj.as_ref().ok_or_else(|| shape("stc_first route", "built without the conjugation key"))?;
        let ctx = cx.ctx;
        let lay = ctx.lay;
        for x in cts.iter_mut() {
            let (k, top) = (x.k(), x.top(ctx.shape)?);
            let ex = &mut *cx.ex;
            ks::modup(ctx, &ex.compute, cx.ks, &mut ex.ks, &mut ex.bconv_work, &x.buf, 0, k, 1, top)?;
            let v = View::of(&x.buf, 0);
            let io = DotIo {
                c1: v.part::<W>(1).base,
                c0: v.base,
                ct_stride: v.ct_stride,
                out: ex.ks.ext.buf.vptr_mut(),
                part_stride: lay.poly_words() as u64,
                out_ct_stride: lay.ct_words() as u64,
                out_rot_stride: 0,
                c0_mode: 1,
            };
            let staged = conj.needs_stage() || !conj.has_a();
            if staged {
                let ndig = cx.ks.ndig(top)?;
                ks::stage_key(ctx, &ex.compute, cx.ks, &mut ex.ks, conj, 0, ndig, true)?;
            }
            ks::dot(ctx, &ex.compute, cx.ks, &ex.ks, conj, 0, 1, k, top, DotRows::All, io, staged)?;
            ks::moddown_to(ctx, &ex.compute, cx.ks, &mut ex.ks.ext, &mut ex.ks.conv, v, ks::MdOut::Add, k, top)?;
        }
        Ok(())
    }

    /// `btsStcFirstInput`: drop to the entry, the input-side StC stages; the ciphertexts end at the split (mode 2:
    /// limb 2d-1, degree 2; mode 1: limb 3d-1, degree 2). `own_cf`: build the last stage at the call's CF (the PCMM's
    /// refresh) instead of the reference CF.
    pub fn stc_first_input_cf(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>], own_cf: bool) -> Result<Option<SplitInfo>> {
        if !st.stc_first {
            return Ok(None);
        }
        let ctx = cx.ctx;
        let shp = ctx.shape;
        let entry_level = shp.n_q - 1 - self.entry_top as u32;
        for x in cts.iter_mut() {
            if x.meta.deg == 2 {
                cx.ops.rescale(ctx, cx.ex, cx.sh, std::slice::from_mut(x))?;
            }
            if x.meta.level > entry_level {
                return Err(shape("stc_first", format!("bootstrap input at level {} is below the StC-first entry level {entry_level}", x.meta.level)));
            }
            x.meta.level = entry_level;
            x.meta.slots = st.slots;
        }
        let nf_in = common_meta(cts)?.nf;
        let mut info = SplitInfo { nf_in, cf_stage: st.cf, pre_factor_log2: 0, factor: 1.0, key: None };
        let src = self.setup.ks_source;
        if st.stc_folded {
            let last = self.last.as_ref().ok_or_else(|| no_stage("stc_first", "stc_first_input (mode-2 last-stage cache not built)"))?;
            let cf_stage = if own_cf { st.cf } else { st.cf.max(self.sf.cf_ref) };
            if cf_stage > st.cf {
                let e = cf_stage - st.cf;
                for x in cts.iter_mut() {
                    self.pow2.mult(ctx, &cx.ex.compute, View::of(&x.buf, 0), x.k(), self.entry_top, e)?;
                }
                info.pre_factor_log2 = e;
            }
            let factor = self.mode2_factor(ctx, st, cf_stage, nf_in)?;
            let key = CacheKey([st.variant as i64, cf_stage as i64, nf_key(nf_in)]);
            if self.thin() {
                for s in &self.input {
                    run_lt_thin(cx, s, s.pt_nf, cts, src, true)?;
                }
            } else {
                for s in &self.input {
                    run_lt(cx, s, cts, src)?;
                }
            }
            let i = last.get(ctx, &cx.ex.compute, &self.srcs, key, factor)?;
            last.with(i, |stages| {
                for s in stages {
                    if self.thin() {
                        // the cache builds the plaintexts at factor x sf(top): the stage's scale; the prime stays pending
                        run_lt_thin(cx, s, factor * ctx.scale.sf(s.top as i32)?, cts, src, false)?;
                    } else {
                        run_lt(cx, s, cts, src)?;
                    }
                }
                Ok(())
            })?;
            info.cf_stage = cf_stage;
            info.factor = factor;
            info.key = Some(key);
            let want = (shp.n_q - 1 - self.split_top() as u32, 2u8);
            if let Some(x) = cts.iter().find(|x| (x.meta.level, x.meta.deg) != want) {
                return Err(shape("stc_first", format!("folded last stage left the ciphertext at level {} deg {}", x.meta.level, x.meta.deg)));
            }
        } else {
            for s in &self.input {
                run_lt(cx, s, cts, src)?;
            }
        }
        Ok(Some(info))
    }

    /// Warm the mode-2 last stage for a call (before a CUDA graph capture, or a PCMM's first use): the entry for the
    /// state and an input scale `nf_in` at the entry.
    pub fn warm(&self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, st: &BtsState, nf_in: f64, own_cf: bool) -> Result<CacheKey> {
        let last = self.last.as_ref().ok_or_else(|| no_stage("stc_first", "warm (mode-2 cache not built)"))?;
        let cf_stage = if own_cf { st.cf } else { st.cf.max(self.sf.cf_ref) };
        let factor = self.mode2_factor(ctx, st, cf_stage, nf_in)?;
        let key = CacheKey([st.variant as i64, cf_stage as i64, nf_key(nf_in)]);
        last.get(ctx, s, &self.srcs, key, factor)?;
        Ok(key)
    }

    /// The mode-2 last stage's factor for a call: the raise adjust to the variant's top times its `fdrop` =
    /// sf(chain top) / sf(raise top) (the StC plaintexts' decoding constant is q0 / sf(chain top), as the shipped order's
    /// last StC stage compensates in `DensePlan`): 1 at drops 0 and 1, a 4-8 % output gain error at drops 2-5 without
    /// it. The product no longer depends on the variant, so a PCMM split serves a finish under any drop.
    fn mode2_factor(&self, ctx: &Context<'_, W, M>, st: &BtsState, cf_stage: u32, nf_in: f64) -> Result<f64> {
        let d = ctx.chain.level_unit;
        let target_sf = ctx.scale.sf(self.variants[st.variant].raise.top as i32)?;
        let fdrop = ctx.scale.sf((ctx.lay.nq - 1) as i32)? / target_sf;
        if self.thin() {
            // single-prime stages keep the scale but for their factors: the last stage's plaintexts (scale q_top x F,
            // its prime dropped by the raise's rescale) end the input at 2^-cf_stage target fdrop; the cache scales
            // by factor x sf(top)
            let top = self.split_top();
            let fixed: f64 = THIN_SPLIT[..self.input.len()].iter().sum::<f64>().exp2();
            let f = (-(cf_stage as f64)).exp2() * target_sf * fdrop / (nf_in * fixed);
            return Ok(f * ctx.chain.q[top] as f64 / ctx.scale.sf(top as i32)?);
        }
        Ok(stc_first_mode2_factor(cf_stage as i32, target_sf, ctx.scale.sf((d - 1) as i32)?, ctx.scale.sf(self.entry_top as i32)?, nf_in) * fdrop)
    }

    /// A state for the same input under another raise variant (the PCMM finish: input state at the route's drop, the
    /// finish under the site's).
    pub fn state_for_drop(&self, st: &BtsState, raise_drop: u32) -> Result<BtsState> {
        let v = self.variant(raise_drop)?;
        let pre = &self.variants[v];
        let mut o = st.clone();
        o.variant = v;
        o.raise_drop = raise_drop;
        o.route.raise_drop = raise_drop;
        o.const_eval_mult = pre.cts0_const;
        o.baked = pre.cts0_const;
        o.n_cts = pre.cts.len();
        o.n_stc = pre.stc.len();
        Ok(o)
    }

    /// The landing of a refresh (raw output metadata).
    fn landing_meta(&self, n_q: u32, unit: u32, meta: &CtMeta, route: &Route) -> Result<CtMeta> {
        let v = self.variant(route.raise_drop)?;
        let pre = &self.variants[v];
        let f = if route.stc_first && self.runs_stc_first(n_q, unit, meta) {
            if route.real {
                pre.em_real.as_ref().ok_or_else(|| shape("bootstrap real", "real-payload variant not built"))?.out_meta
            } else {
                pre.em.out_meta
            }
        } else {
            *self.outs_shipped.get(v).ok_or_else(|| no_stage("stc_first", "shipped-order landing"))?
        };
        Ok(CtMeta { level: n_q - 1 - f.top as u32, deg: f.nl as u8, nf: f.nf, ..meta.clone() })
    }

    fn once(&self, cx: &mut BtsCx<'_, 'a, W, M>, cts: &mut [Ct<'a, W>], route: &Route) -> Result<()> {
        super::run_staged(self, cx, cts, route, false).map(|_| ())
    }

    /// Whether refreshes of `mx` on `rx` and of `my` on `ry` can run as one pair (`run_pair`): both real dense
    /// StC-first refreshes of one raise variant, no iterations, mode 2, one level and degree and scales of one cache
    /// key (`nf_key`: the cached last stage carries the factor of whichever refresh of that key filled it, and each
    /// member's coefficients pass through it as in the member's own single, so the pair adds no scale error over the
    /// singles at a given cache state),
    /// inputs at or above the entry (deeper ones take the shipped order), CFs in the window and below cf_ref (each
    /// member is pre-scaled by 2^(cf_ref - 1 - cf)). Err names the first condition that fails.
    pub fn pair_check(&self, n_q: u32, unit: u32, mx: &CtMeta, my: &CtMeta, rx: &Route, ry: &Route) -> Result<()> {
        let n = 2 * self.variants.first().map_or(0, |v| v.slots as usize);
        let bad = |what: String| Err(shape("bts pair", what));
        for r in [rx, ry] {
            if RouteKind::of(r, n) != RouteKind::StcFirst || !r.real || r.iterations > 1 {
                return bad(format!("route {r:?}: a pair takes real dense StC-first refreshes without iterations"));
            }
            let cf = self.call_cf(r);
            if cf < self.setup.cf_window.0 || cf < self.deg || cf >= self.sf.cf_ref {
                return bad(format!("CF {cf} outside [{}, {}) (deg {})", self.setup.cf_window.0.max(self.deg), self.sf.cf_ref, self.deg));
            }
        }
        if rx.raise_drop != ry.raise_drop {
            return bad(format!("raise drops {} and {}", rx.raise_drop, ry.raise_drop));
        }
        self.variant(rx.raise_drop)?;
        if self.sf.mode != StcFirstMode::Folded || self.last.is_none() {
            return bad("the pair runs the mode-2 last stage (FIDESLIB_BTS_STC_FIRST 2)".into());
        }
        if self.sf.own_cf {
            return bad("the pair's last stage is cached at cf_ref, not at each call's CF (StcFirstSetup::own_cf)".into());
        }
        if mx.level != my.level || mx.deg != my.deg || nf_key(mx.nf) != nf_key(my.nf) {
            return bad(format!("inputs at level {} deg {} nf 2^{:.6} and level {} deg {} nf 2^{:.6}", mx.level, mx.deg, mx.nf.log2(), my.level, my.deg, my.nf.log2()));
        }
        if !self.runs_stc_first(n_q, unit, mx) {
            return bad(format!("input at level {} deg {} is past the StC-first entry level {}", mx.level, mx.deg, self.entry_level(n_q)));
        }
        Ok(())
    }

    /// Two real refreshes as one complex refresh (`BtsRoute::run_pair`). Each member goes to the entry as alone
    /// (pending rescale, drop), is multiplied by 2^(cf_ref - 1 - cf) (exact) and becomes x + conj x = 2 Re x (what a
    /// single's real EvalMod chain reads, whatever the imaginary lane holds); z = x + i y runs the input stages and the
    /// last stage cached at cf_ref (the entry a single of that input asks for), the raise, the CtS and the complex
    /// EvalMod, whose Re chain is x's output and Im chain y's (no recombination); each is multiplied by its 2^cf. Per
    /// member: the single's arithmetic, plus the entry conjugation's key-switch noise.
    fn pair(&self, cx: &mut BtsCx<'_, 'a, W, M>, x: &mut Ct<'a, W>, y: &mut Ct<'a, W>, rx: &Route, ry: &Route) -> Result<()> {
        let ctx = cx.ctx;
        let shp = ctx.shape;
        let k = x.k();
        if y.k() != k {
            return Err(shape("bts pair", format!("members of {k} and {} ciphertexts", y.k())));
        }
        self.pair_check(shp.n_q, shp.unit, &x.meta, &y.meta, rx, ry)?;
        let (lx, ly) = (self.landing_meta(shp.n_q, shp.unit, &x.meta, rx)?, self.landing_meta(shp.n_q, shp.unit, &y.meta, ry)?);
        let mut st = self.begin_meta(ctx, &x.meta, k, &Route { cf: Some(self.sf.cf_ref), real: false, ..*rx }, false)?;
        let entry_level = self.entry_level(shp.n_q);
        for (c, r) in [(&mut *x, rx), (&mut *y, ry)] {
            if c.meta.deg == 2 {
                cx.ops.rescale(ctx, cx.ex, cx.sh, std::slice::from_mut(c))?;
            }
            if c.meta.level > entry_level {
                return Err(shape("bts pair", format!("input at level {} is below the StC-first entry level {entry_level}", c.meta.level)));
            }
            c.meta.level = entry_level;
            self.pow2.mult(ctx, &cx.ex.compute, View::of(&c.buf, 0), k, self.entry_top, self.sf.cf_ref - 1 - self.call_cf(r))?;
            self.add_conj(cx, std::slice::from_mut(c))?;
        }
        cx.ops.mult_i_inplace(ctx, cx.ex, cx.sh, std::slice::from_mut(y))?;
        // z = x + i y at x's scale (no scale adjust: the scales share one cache key)
        ks::ew(ctx, &cx.ex.compute, ks::Ew::Add, View::of(&x.buf, 0), View::of(&x.buf, 0), Some(View::of(&y.buf, 0)), ctx.lay.rs_q(self.entry_top), k, 2)?;
        let xs = std::slice::from_mut(x);
        let info = self.stc_first_input_cf(cx, &st, xs, false)?;
        if info.is_none_or(|i| i.pre_factor_log2 != 0) {
            return Err(shape("bts pair", format!("the pair's input stage ran {info:?}, not the last stage at cf_ref {}", self.sf.cf_ref)));
        }
        self.mod_raise(cx, &mut st, xs)?;
        for i in 0..st.n_cts {
            self.cts_stage(cx, &st, xs, i)?;
        }
        let em = &self.variants[st.variant].em;
        let f = FMeta { top: (shp.n_q - 1 - x.meta.level) as i32, nl: x.meta.deg as i32, nf: x.meta.nf };
        if f.top != em.in_meta.top || f.nl != em.in_meta.nl || f.nf.to_bits() != em.in_meta.nf.to_bits() {
            return Err(shape("eval_mod", format!("input {f:?}, the program was built for {:?}", em.in_meta)));
        }
        em.run_split(cx, &mut x.buf, &mut y.buf, k)?;
        let out = em.out_meta;
        for (c, r, l) in [(&mut *x, rx, lx), (&mut *y, ry, ly)] {
            self.pow2.mult(ctx, &cx.ex.compute, View::of(&c.buf, 0), k, out.top as usize, self.call_cf(r))?;
            if l.level != shp.n_q - 1 - out.top as u32 || l.deg as i32 != out.nl || l.nf.to_bits() != out.nf.to_bits() {
                return Err(shape("bts pair", format!("EvalMod left {out:?}, the member's landing is level {} deg {} nf 2^{:.6}", l.level, l.deg, l.nf.log2())));
            }
            c.meta = l;
        }
        Ok(())
    }
}

impl<'a, W: DevWord, M: DevStrategy<W>> Staged<'a, W, M> for StcFirstRoute<'a, W, M> {
    fn begin(&self, cx: &BtsCx<'_, 'a, W, M>, cts: &[Ct<'a, W>], route: &Route, prescaled: bool) -> Result<BtsState> {
        let m = common_meta(cts)?;
        let k = cts.iter().map(|x| x.k()).max().unwrap_or(0);
        self.begin_meta(cx.ctx, &m, k, route, prescaled)
    }
    fn stc_first_input(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>]) -> Result<()> {
        self.stc_first_input_cf(cx, st, cts, self.sf.own_cf).map(|_| ())
    }
    fn mod_raise(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &mut BtsState, cts: &mut [Ct<'a, W>]) -> Result<()> {
        if st.real && !st.stc_first {
            self.add_conj(cx, cts)?;
        }
        let pre = &self.variants[st.variant];
        let prescaled = st.prescaled || (st.stc_first && st.stc_folded);
        let encaps = self.encaps.as_ref().ok_or_else(|| no_stage("stc_first", "mod_raise (built without encapsulation keys)"))?;
        raise::raise(cx, &pre.raise, Some(encaps), self.setup.ks_source, cts, if prescaled { None } else { Some(st.correction) })?;
        st.shifted_flow = true;
        Ok(())
    }
    fn fold(&self, _cx: &mut BtsCx<'_, 'a, W, M>, _st: &BtsState, _cts: &mut [Ct<'a, W>]) -> Result<()> {
        Ok(())
    }
    fn cts_stage(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>], k: usize) -> Result<()> {
        let stage = self.variants[st.variant].cts.get(k).ok_or_else(|| no_stage("stc_first", &format!("cts_stage({k})")))?;
        run_lt(cx, stage, cts, self.setup.ks_source)
    }
    fn eval_mod(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>]) -> Result<()> {
        let pre = &self.variants[st.variant];
        let em = if st.real { pre.em_real.as_ref().ok_or_else(|| no_stage("stc_first", "eval_mod (real)"))? } else { &pre.em };
        let ctx = cx.ctx;
        for x in cts.iter_mut() {
            let f = FMeta { top: (ctx.shape.n_q - 1 - x.meta.level) as i32, nl: x.meta.deg as i32, nf: x.meta.nf };
            if f.top != em.in_meta.top || f.nl != em.in_meta.nl || f.nf.to_bits() != em.in_meta.nf.to_bits() {
                return Err(shape("eval_mod", format!("input {f:?}, the program was built for {:?}", em.in_meta)));
            }
            let k = x.k();
            em.run(cx, &mut x.buf, k)?;
            Self::set_meta(cx, x, em.out_meta);
            if x.meta.deg == 2 && !st.stc_first {
                cx.ops.rescale(ctx, cx.ex, cx.sh, std::slice::from_mut(x))?;
            }
        }
        Ok(())
    }
    fn stc_first_output(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>]) -> Result<()> {
        if !st.stc_first {
            return Err(no_stage("stc_first", "stc_first_output (the call runs the shipped order)"));
        }
        let ctx = cx.ctx;
        let e = pow2_exp(st.cor_factor << self.deg)?;
        for x in cts.iter_mut() {
            let (k, top) = (x.k(), x.top(ctx.shape)?);
            self.pow2.mult(ctx, &cx.ex.compute, View::of(&x.buf, 0), k, top, e)?;
            x.meta.slots = st.old_slots;
        }
        Ok(())
    }
    fn stc_enter(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>]) -> Result<()> {
        let pre = &self.variants[st.variant];
        let stc_top = pre.stc.first().map(|s| s.top).ok_or_else(|| no_stage("stc_first", "stc_enter (shipped StC not built)"))?;
        let ctx = cx.ctx;
        for x in cts.iter_mut() {
            if x.top(ctx.shape)? > stc_top {
                if x.meta.deg == 2 {
                    cx.ops.rescale(ctx, cx.ex, cx.sh, std::slice::from_mut(x))?;
                }
                if x.top(ctx.shape)? > stc_top {
                    x.meta.level = ctx.shape.n_q - 1 - stc_top as u32;
                }
            }
        }
        Ok(())
    }
    fn stc_stage(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>], k: usize) -> Result<()> {
        let pre = &self.variants[st.variant];
        let stage = if k == 0 && st.real { pre.stc_real0.as_ref() } else { pre.stc.get(k) };
        let stage = stage.ok_or_else(|| no_stage("stc_first", &format!("stc_stage({k})")))?;
        run_lt(cx, stage, cts, self.setup.ks_source)
    }
    fn finish(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>]) -> Result<()> {
        if st.real {
            self.add_conj(cx, cts)?;
        }
        let ctx = cx.ctx;
        let e = pow2_exp(st.cor_factor)?;
        for x in cts.iter_mut() {
            let (k, top) = (x.k(), x.top(ctx.shape)?);
            self.pow2.mult(ctx, &cx.ex.compute, View::of(&x.buf, 0), k, top, e)?;
            x.meta.slots = st.old_slots;
        }
        Ok(())
    }
    fn stc_high(&self, cx: &mut BtsCx<'_, 'a, W, M>, cts: &mut [Ct<'a, W>], spec: &StcHigh) -> Result<()> {
        let high = self.high.as_ref().ok_or_else(|| no_stage("stc_first", "stc_high (StcFirstSetup::high not configured)"))?;
        high.run(cx, &self.srcs, cts, spec, self.setup.ks_source)
    }
}

impl<'a, W: DevWord, M: DevStrategy<W>> BtsRoute<'a, W, M> for StcFirstRoute<'a, W, M> {
    fn kind(&self) -> RouteKind {
        RouteKind::StcFirst
    }
    fn landing(&self, sh: &Shadow, meta: &CtMeta, route: &Route) -> Result<CtMeta> {
        self.landing_meta(sh.cfg.chain.n_q, sh.cfg.chain.unit, meta, route)
    }
    fn run_pair(&self, cx: &mut BtsCx<'_, 'a, W, M>, x: &mut Ct<'a, W>, y: &mut Ct<'a, W>, rx: &Route, ry: &Route) -> Result<()> {
        self.pair(cx, x, y, rx, ry)
    }
    fn run(&self, cx: &mut BtsCx<'_, 'a, W, M>, cts: &mut [Ct<'a, W>], route: &Route) -> Result<()> {
        if route.iterations <= 1 {
            return self.once(cx, cts, route);
        }
        // y = bts(x); repeat: y += bts((x - y) 2^p) 2^-p   (perseus eval_bootstrap_iter, BTS_PRECISION p)
        let p = super::dense::ITER_PRECISION as f64;
        let ctx = cx.ctx;
        for x in cts.iter_mut() {
            let k = x.k();
            if cx.work.spare.len() < 2 {
                return Err(shape("bootstrap iterations", "bts.work needs 2 spare buffers (BtsWorkShape::spare)"));
            }
            let mut sa = cx.work.spare.pop().ok_or_else(|| shape("bts", "spare"))?;
            let mut sb = cx.work.spare.pop().ok_or_else(|| shape("bts", "spare"))?;
            let (ka, kb) = (sa.k, sb.k);
            sa.k = k;
            sb.k = k;
            let top_in = x.top(ctx.shape)?;
            let orig = Ct::new(x.meta.clone(), sa);
            ks::ew(ctx, &cx.ex.compute, ks::Ew::Copy, View::of(&orig.buf, 0), View::of(&x.buf, 0), None, ctx.lay.rs_q(top_in), k, 2)?;
            let mut e = Ct::new(x.meta.clone(), sb);
            let r = (|| -> Result<()> {
                self.once(cx, std::slice::from_mut(x), route)?;
                for _ in 1..route.iterations {
                    cx.ops.sub(ctx, cx.ex, cx.sh, std::slice::from_mut(&mut e), &[&orig], &[&*x])?;
                    cx.ops.mult_scalar_inplace(ctx, cx.ex, cx.sh, std::slice::from_mut(&mut e), &[Complex::new(p.exp2(), 0.0)])?;
                    self.once(cx, std::slice::from_mut(&mut e), route)?;
                    cx.ops.mult_scalar_inplace(ctx, cx.ex, cx.sh, std::slice::from_mut(&mut e), &[Complex::new((-p).exp2(), 0.0)])?;
                    cx.ops.add_inplace(ctx, cx.ex, cx.sh, std::slice::from_mut(x), &[&e])?;
                }
                Ok(())
            })();
            let (mut sa, mut sb): (CtBuf<'a, Compute, W>, CtBuf<'a, Compute, W>) = (orig.buf, e.buf);
            sa.k = ka;
            sb.k = kb;
            cx.work.spare.push(sb);
            cx.work.spare.push(sa);
            r?;
        }
        Ok(())
    }
    fn staged(&self) -> Option<&dyn Staged<'a, W, M>> {
        Some(self)
    }
    fn mem(&self) -> (MemReq, BtsWorkShape) {
        self.mem.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pc_core::params::ChainSpec;

    /// The n32 host plan: FIDESlib's loader entry (2d-1) + (n-1) d = 7 in mode 2 (9 in mode 1), the fixed stages at
    /// 7, 5 (mode 1: 9, 7, 5, the last x 2^-deg), the mode-2 last stage re-levelled to limb 3 from its source only.
    #[test]
    fn n32_plan_levels() {
        let chain = Chain::generate(&ChainSpec::n32()).unwrap();
        let mut setup = BtsSetup::n32_shipped();
        setup.real = false;
        let p = StcFirstPlan::new(&chain, &setup, &StcFirstSetup::n32_shipped()).unwrap();
        assert_eq!(p.entry_top, 7);
        assert_eq!(p.input.iter().map(|h| h.top).collect::<Vec<_>>(), vec![7, 5]);
        assert!(p.input.iter().all(|h| h.end == LtEnd::ModDown));
        assert_eq!(p.sources.iter().map(|s| s.is_some()).collect::<Vec<_>>(), vec![false, false, true]);
        assert_eq!(p.last_shape().unwrap(), vec![StageShape { src: 2, top: 3, end: LtEnd::ModDown, scaled: true }]);
        let st = ScaleTables::precompute(&chain).unwrap();
        for h in &p.input {
            assert_eq!(h.nf.to_bits(), st.sf(h.top as i32).unwrap().to_bits());
        }
        let mut sf = StcFirstSetup::n32_shipped();
        sf.mode = StcFirstMode::Plain;
        let p1 = StcFirstPlan::new(&chain, &setup, &sf).unwrap();
        assert_eq!(p1.entry_top, 9);
        assert_eq!(p1.input.iter().map(|h| h.top).collect::<Vec<_>>(), vec![9, 7, 5]);
        assert!(p1.last_shape().is_none() && p1.sources.iter().all(|s| s.is_none()));
        // the entry level is pc-shadow's (FAKECAP_STC_ENTRY)
        assert_eq!(Some(53 - p.entry_top as u32), pc_shadow::BtsConfig::n32_shipped().stc_first_entry);
    }
}
