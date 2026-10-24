//! `Session`: the device runtime behind the eager `Fhe`. One `Session` owns, in one object and with no
//! process-global state: the `Device` guard, the ONE arena, the `Executor`, the `Context`, the eager `CtPool`, the
//! key set, key groups, the op layer (`Ops`), the plaintext store and the encryptor / decryptors.
//! `complete_setup()` freezes the arena; `Session.fhe` is the op surface after that.
//!
//! Lifetimes: the arena is boxed and owned by the same `Dev` as its borrowers (borrows extended to `'static` once,
//! here); `Dev`'s field order is the drop order. Threads: `Session`, `Fhe` and `Ct` are `unsendable`; blocking
//! device calls release the GIL (`nogil`). Details: docs/ARCHITECTURE.md "pc-py Session".
use crate::config::PyConfig;
use crate::shadow::shadow_config;
use crate::LevelError;
use pc_core::keys::{Prng, SecretKey};
use pc_core::modmul::Barrett64;
use pc_core::params::{Chain, ChainSpec};
use pc_cuda::{Arena, Device, Event, EventPool};
use pc_gpu::decrypt::Decryptor;
use pc_gpu::encrypt::Encryptor;
use pc_gpu::keygen::KeyGen;
use pc_gpu::keys::{DevSecret, KeyRole, KeySet, KeySetSpec, KskLayout, RotSpec};
use pc_gpu::ks::{KeyGroup, KsKeyRef, KsPlans, KsWork, KskSource};
use pc_gpu::ops::{Ops, OpsCfg};
use pc_gpu::pt_store::{PtStore, PtStoreCfg, PtStorePolicy};
use pc_gpu::rotate::RotKeys;
use pc_gpu::bts::sparse::{chain_deg, SparseBRoute, SpruRoute};
use pc_gpu::bts::sparse_b::{SparseBCfg, SparseBPrecomp};
use pc_gpu::bts::spru::{Spru, SpruCfg, SpruKey};
use pc_gpu::bts::stc_first::{StcFirstPlan, StcFirstRoute, StcFirstSetup};
use pc_gpu::bts::{BtsKeyRing, BtsSetup, BtsWork, BtsWorkShape, RouteKind};
use pc_gpu::keys::KskShape;
use pc_gpu::{Context, Ct, CtPool, Executor, MemCfg, MemPlan, PoolClass};
use pc_program::bind::{Bound as PBound, Slab};
use pc_program::replay::{CaptureLock, ReplayCfg, Replayer, Routes, SideLane};
use pc_program::{Kind, Program};
use pc_shadow::route::Landing;
use pc_shadow::Shadow;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;
use std::sync::Arc;

pub(crate) type W = u32;
pub(crate) type M = Barrett64;

/// A pc-gpu error as a Python exception: Shadow refusals are `LevelError`, the rest `RuntimeError`.
pub(crate) fn gerr(e: pc_gpu::Error) -> PyErr {
    match e {
        pc_gpu::Error::Shadow(s) => LevelError::new_err(s.to_string()),
        e => PyRuntimeError::new_err(e.to_string()),
    }
}

/// Run `f` with the GIL released (see the module doc for why this is sound with unsendable state).
pub(crate) fn nogil<T, F: FnOnce() -> T>(py: Python<'_>, f: F) -> T {
    struct Unsafe<X>(X);
    // SAFETY: Python::detach runs the closure on this thread; only the GIL is released meanwhile
    unsafe impl<X> Send for Unsafe<X> {}
    impl<X> Unsafe<X> {
        fn into_inner(self) -> X {
            self.0
        }
    }
    let w = Unsafe(f);
    // a method call on `w` captures the whole wrapper (a field access would capture the non-Send field)
    let r = py.detach(move || Unsafe(w.into_inner()()));
    r.into_inner()
}

/// Where the session's keys come from. `generate`: a fresh secret (OS entropy, or `seed` for reproducible tests)
/// and every key generated on the device.
#[pyclass(name = "KeySource", module = "perseus._pc", frozen)]
#[derive(Clone, Debug)]
pub struct PyKeySource {
    pub rot_steps: Vec<i64>,
    pub groups: Vec<Vec<i64>>,
    pub conj: bool,
    pub layout: String,
    pub band: Option<usize>,
    pub seed: Option<u64>,
    /// a client's secret (`ClientSecret.to_bytes()`): the device generates its keys from it (tests)
    pub secret: Option<Vec<u8>>,
}

#[pymethods]
impl PyKeySource {
    /// `rot_steps`: rotation steps (any sign; normalized mod N/2). `groups`: step lists of hoisted / rotate-sum
    /// groups to build exact key tables for (any other list runs on the session-wide table). `layout`: "dense"
    /// (b and a resident: KskSource::Cached) or "packed" (28-bit b + seed, staged per key: KskSource::Regen).
    /// `band`: highest ciphertext top limb the rotation keys serve (None = full rows). `seed`: reproducible
    /// secret and keys (tests); None = OS entropy. `secret`: a client's secret (`ClientSecret.to_bytes()`) to
    /// generate from instead of a fresh one (tests comparing device keys with a client's).
    #[staticmethod]
    #[pyo3(signature = (rot_steps, groups=Vec::new(), conj=true, layout="dense", band=None, seed=None, secret=None))]
    fn generate(rot_steps: Vec<i64>, groups: Vec<Vec<i64>>, conj: bool, layout: &str, band: Option<usize>, seed: Option<u64>, secret: Option<Vec<u8>>) -> PyResult<Self> {
        if layout != "dense" && layout != "packed" {
            return Err(PyValueError::new_err(format!("KeySource.layout = {layout:?}: dense | packed")));
        }
        Ok(PyKeySource { rot_steps, groups, conj, layout: layout.to_string(), band, seed, secret })
    }
    #[getter]
    fn rot_steps(&self) -> Vec<i64> {
        self.rot_steps.clone()
    }
    fn __repr__(&self) -> String {
        format!(
            "KeySource.generate({} rotation steps, {} groups, conj={}, layout={:?}, band={:?})",
            self.rot_steps.len(),
            self.groups.len(),
            self.conj,
            self.layout,
            self.band
        )
    }
}

/// Session sizes (constructor keywords of `Session`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Sizes {
    pub pool: usize,
    pub pt_slots: usize,
    pub encode_threads: usize,
}

/// The model geometry the impl layer reads as `inf.size` (`Dims.from_inf`).
#[derive(Clone, Debug)]
pub(crate) struct Geometry {
    pub hid: usize,
    pub dim: usize,
    pub heads: usize,
    pub heads_real: usize,
    pub exp: usize,
    pub exp_real: usize,
    pub seq_len: usize,
}

/// What a planned session is sized for (`Session(programs=...)`): the compiled block programs (their slots, key
/// groups, refresh routes and rotation steps), the replay rules (the plans' runtime Shadow), the registers that
/// carry ciphertexts across blocks and tokens (block exits, KV states), the plaintext store's size, and the steps
/// a dynamic rotation may take.
#[derive(Clone)]
pub(crate) struct PlanSpec {
    pub programs: Vec<Program>,
    pub shadows: Vec<pc_shadow::ShadowConfig>,
    pub regs: usize,
    /// per register, the richest level it carries (its rows `0..=top`); empty = every register full-layout
    pub reg_levels: Vec<u32>,
    pub pt_rows: usize,
    /// the levels of the plaintexts the setup stages (one per distinct key): the store's slot classes
    pub pt_levels: Vec<u32>,
    pub dyn_rot: Vec<i64>,
    pub stc_cache: usize,
    pub use_graphs: bool,
    /// compare every eager step's output metadata with the compiled one (debug)
    pub check_meta: bool,
    /// streamed linears: name -> rows (`Kind::CoeffSum` weight rows the host streams per block)
    pub streams: BTreeMap<String, usize>,
    /// device weight frames (>= 2)
    pub stream_frames: usize,
    /// pinned KV units: (units, live rows per unit, device pages); None = KV through the registers
    pub kv: Option<(usize, usize, usize)>,
}

/// The streamed linears: row counts (bind's frame layout) and, once `Session.stream_weights` uploaded them, their
/// pinned host rows (the ring's upload source).
pub(crate) struct StreamReg {
    pub rows: BTreeMap<String, usize>,
    pub pinned: HashMap<String, pc_cuda::PinnedBuf<f32>>,
}

impl pc_program::bind::StreamSource for StreamReg {
    fn rows(&self, linear: &str) -> Option<usize> {
        self.rows.get(linear).copied()
    }
}

impl pc_program::residency::StreamHost for StreamReg {
    fn pinned(&self, linear: &str) -> Option<&pc_cuda::PinnedBuf<f32>> {
        self.pinned.get(linear)
    }
}

/// The planned runtime: the refresh routes the programs use, their work buffers, the shared ciphertext slab, the
/// registers, the bound blocks (`Program.bind`) and the replayer. Field order = drop order (graphs before the
/// routes whose tables they read).
pub(crate) struct Planned {
    pub bounds: Vec<PBound<'static, W>>,
    /// the PCMM service behind the programs' `Finish` ops; before the routes it runs on
    pub pcmm: Option<crate::pcmm::PcmmRt>,
    /// the CCMM service, likewise
    pub ccmm: Option<crate::matmul::CcmmRt>,
    /// per bound block: its runtime Shadow (the plan's landings)
    pub shadows: Vec<Shadow>,
    pub replayer: Replayer,
    pub slab: Slab<'static, W>,
    pub regs: Vec<Ct<'static, W>>,
    pub work: BtsWork<'static, W>,
    pub dense: Option<StcFirstRoute<'static, W, M>>,
    /// the dense route in the shipped order (sites without StC-first)
    pub plain: Option<pc_gpu::bts::dense::DenseRoute<'static, W, M>>,
    pub sparse: Vec<SparseBRoute<'static, W>>,
    /// the classic multi-stage sparse routes (slot counts past FIDESLIB_BTS_SPARSE_B_MAX)
    pub classic: Vec<pc_gpu::bts::sparse_classic::SparseClassicRoute<'static, W, M>>,
    pub spru: Option<SpruRoute<'static, W>>,
    #[allow(dead_code)]
    pub spru_secret: Option<DevSecret<'static, W>>,
    pub spec: PlanSpec,
    /// arena bytes reserved for the programs' key groups (bind carves them)
    pub group_bytes: usize,
    pub stats: PlannedStats,
    /// the side lane the replay schedules a block's off-critical-path steps on
    pub side: Option<SideLane<'static, W>>,
    /// streamed weights: the device expander, the weight frames, the pinned rows, the blocks' run order
    pub coeff: Option<pc_gpu::encode_dev::CoeffRowsDevice<'static, W>>,
    pub ring: Option<pc_program::residency::StreamRing<'static, 'static>>,
    pub streams: StreamReg,
    /// per bound block: the id of its weight frame content (equal linears = equal id, whatever the program class)
    pub stream_tag: Vec<Option<usize>>,
    /// frame layouts by id; the ids in run order (`Session.stream_order`)
    pub tag_layouts: Vec<pc_program::bind::StreamLayout>,
    pub stream_order: Vec<usize>,
    /// pinned KV units
    pub kv: Option<pc_program::kv::PinnedKv<'static, 'static, W>>,
    /// events of the ring / KV uploads: after them (they borrow it)
    #[allow(dead_code)]
    pub stream_events: Box<EventPool>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PlannedStats {
    pub runs: u64,
    pub host_us: f64,
    pub launch_us: f64,
    pub cache_evictions_seen: u64,
}

impl Planned {
    pub fn routes(&self) -> Routes<'_, 'static, W, M> {
        let mut r = Routes::new();
        if let Some(d) = &self.dense {
            r.add(d);
        }
        if let Some(d) = &self.plain {
            r.add(d);
        }
        for x in &self.sparse {
            r.add(x);
        }
        for x in &self.classic {
            r.add(x);
        }
        if let Some(x) = &self.spru {
            r.add(x);
        }
        r
    }
    pub fn locks(&self) -> Vec<&dyn CaptureLock> {
        self.dense.iter().flat_map(|d| d.last.iter().map(|l| l as &dyn CaptureLock)).collect()
    }
}

/// Mutable device state (borrowed `&mut` per op).
pub(crate) struct State {
    /// the planned runtime (a `Session(programs=...)`); first: dropped before everything it borrows
    pub planned: Option<Planned>,
    pub ops: Ops<'static, W>,
    pub pts: PtStore<'static, 'static, W>,
    pub ex: Executor<'static, W>,
    pub keys: KeySet<'static, W>,
    pub plans: KsPlans<'static>,
    pub groups: Groups,
    /// secret-key encryption with a secret, public-key encryption (the bundle's key) without
    pub enc: Encryptor<'static, W>,
    /// the device secret and the decryptors; None on a session built from a client's key bundle
    pub secret: Option<SecretSide>,
    /// the host secret, kept until the keys made from it on the host (PCMM / CCMM) exist: `Session.drop_secret()`
    pub sk: Option<SecretKey>,
    pub prng: Prng,
    /// the `PtStore`'s staging events: declared after `pts`, which borrows them (owned here, never read)
    #[allow(dead_code)]
    pub pt_events: Box<EventPool>,
    /// host values of the named slot plaintexts (`set_slot_pt`), encoded per level on use
    pub named: HashMap<String, Named>,
    /// cached plaintexts (`mult_cached` / `stage_pts`): tag -> host values and the levels encoded
    pub cached: HashMap<String, Named>,
    pub enc_hits: u64,
    pub enc_misses: u64,
}

/// What only a session holding the secret has.
pub(crate) struct SecretSide {
    pub secret: DevSecret<'static, W>,
    /// bottom-pair and 4-row decryptors (a value above q0 q1 / 2 at its scale decodes from 4 rows)
    pub dec2: Decryptor<'static, W>,
    pub dec4: Decryptor<'static, W>,
}

#[derive(Clone, Debug)]
pub(crate) struct Named {
    pub values: Vec<pc_core::encode::Complex>,
    /// levels with a device encoding (the store may have dropped one under LRU since; `PtStore::has` decides)
    pub levels: BTreeSet<u32>,
}

/// Setup-built key-switching tables: one session-wide group of every rotation key (+ conj), the exact groups of
/// `KeySource.groups`, and the relinearization key.
pub(crate) struct Groups {
    pub all: Option<KeyGroup<'static>>,
    pub steps: BTreeSet<i64>,
    pub exact: BTreeMap<Vec<i64>, KeyGroup<'static>>,
    pub relin: KeyGroup<'static>,
    pub conj: Option<KeyGroup<'static>>,
    pub half: i64,
}

impl Groups {
    pub fn has_step(&self, s: i64) -> bool {
        let k = s.rem_euclid(self.half);
        k == 0 || self.steps.contains(&k)
    }
}

impl RotKeys for Groups {
    fn group(&self, steps: &[i64]) -> Option<&KeyGroup<'_>> {
        if let Some(g) = self.exact.get(steps) {
            return Some(g);
        }
        if steps.iter().all(|&s| self.has_step(s)) {
            return self.all.as_ref();
        }
        None
    }
}

/// Everything a session owns. Field order = drop order (see the module doc).
pub(crate) struct Dev {
    pub st: RefCell<State>,
    pub pool: CtPool<'static, W>,
    pub ctx: Context<'static, W, M>,
    pub sh: Shadow,
    pub source: KskSource,
    pub radix_bits: u32,
    pub complex: bool,
    pub geometry: Geometry,
    pub debug_refresh: bool,
    pub frozen: std::cell::Cell<bool>,
    pub mem_report: String,
    pub keygen_s: f64,
    pub setup_s: f64,
    arena: Box<Arena>,
    pub device: Device,
}

impl Drop for Dev {
    fn drop(&mut self) {
        // nothing in flight may still read or write the arena when it is freed
        let _ = self.st.get_mut().ex.compute.sync();
    }
}

impl Dev {
    pub fn arena(&self) -> &Arena {
        &self.arena
    }
    /// The arena with the session's lifetime (setup-time carving by `Program.bind`; see `pool`).
    pub fn arena_static(&self) -> &'static Arena {
        // SAFETY: the arena is boxed inside this `Dev`, owned by an `Rc` that every carved object's holder keeps
        // alive (a bound block lives in `State::planned`, dropped before the arena by field order)
        unsafe { &*(self.arena.as_ref() as *const Arena) }
    }
    /// The pool with the session's lifetime (the `Ct` handles carry an `Rc<Dev>` that keeps it alive).
    pub fn pool(&self) -> &'static CtPool<'static, W> {
        // SAFETY: `pool` lives inside an `Rc<Dev>` (stable address); every handle acquired from it is held by a `Ct`
        // that also holds an `Rc<Dev>` declared after the handle, so the pool outlives every reference
        unsafe { &*(&self.pool as *const CtPool<'static, W>) }
    }
    pub fn n(&self) -> usize {
        self.ctx.lay.n
    }
    pub fn slots(&self) -> usize {
        self.ctx.lay.n / 2
    }
    pub fn level_limit(&self) -> u32 {
        self.sh.cfg.bts.level_limit
    }
    /// The dense shipped-order landing (the encode level of fresh token inputs, the C++ `bootstrap_output_level`).
    pub fn bts_out(&self) -> u32 {
        self.landing_dense().level
    }
    pub fn landing_dense(&self) -> Landing {
        let b = &self.sh.cfg.bts;
        b.dense_shipped.or_else(|| b.landings.get(&0).copied()).unwrap_or(Landing { level: b.bootstrap_level, deg: 1 })
    }
    pub fn payload(&self) -> pc_core::encode::Payload {
        if self.complex {
            pc_core::encode::Payload::Complex
        } else {
            pc_core::encode::Payload::Real
        }
    }
    pub fn check_frozen(&self) -> PyResult<()> {
        if !self.frozen.get() {
            return Err(PyRuntimeError::new_err("Session: call complete_setup() before running ops"));
        }
        Ok(())
    }
}

fn knob(cfg: Option<i64>, default: i64, name: &str) -> PyResult<i64> {
    let v = cfg.unwrap_or(default);
    if v < 0 {
        return Err(crate::ConfigError::new_err(format!("{name} = {v} must be >= 0")));
    }
    Ok(v)
}

/// The rotation steps of LT stages (BSGS babies, the giant stride, the offset).
fn lt_steps(geoms: &[pc_gpu::bts::cts::LtGeom], n: usize) -> BTreeSet<i64> {
    let mut steps = BTreeSet::new();
    for g in geoms {
        for i in 1..g.b {
            steps.insert(((i * g.stride) % (n / 2)) as i64);
        }
        if g.g > 1 {
            steps.insert(((g.b * g.stride) % (n / 2)) as i64);
        }
        steps.insert(g.offset.rem_euclid((n / 2) as i64));
    }
    steps.remove(&0);
    steps
}

/// Constant-ring slots of a planned session: 8192, or with CUDA graphs room for every constant the captured
/// programs push at once (a scalar op or a scalar prelude one row table, a refresh or an external finish a few;
/// counted twice over).
fn const_slots(ps: &PlanSpec) -> usize {
    if !ps.use_graphs {
        return 8192;
    }
    let mut n = 0;
    for p in &ps.programs {
        for m in p.ops.iter().flat_map(|o| o.members()) {
            n += match &m.kind {
                Kind::AddConst(_) | Kind::MultConst(_) => 2,
                Kind::Bootstrap(_) => 16,
                Kind::Finish { .. } => 16 * m.outs.len(),
                _ => 0,
            };
            n += m.pre.iter().filter(|x| matches!(x, pc_shadow::Prelude::ScalarAdjust(..))).count();
        }
    }
    (2 * n).max(8192)
}

/// The mode-2 last-stage keys (`stc_first` `CacheKey`: raise variant, the stage's CF, the input scale) the programs'
/// StC-first refreshes can ask for, counted by what sets the input scale: a planned refresh's by its input level and
/// degree (at the reference CF), a PCMM finish's also by the linear's output shift (at its own CF). Inputs past the
/// entry level run the shipped order. A count, not the keys: equal scales reached two ways count twice.
fn stc_first_keys(progs: &[Program], n: usize, entry_level: u32, cf_default: u32, cf_ref: u32) -> usize {
    let mut keys: BTreeSet<(u32, u32, u32, u8, u32)> = BTreeSet::new();
    for p in progs {
        for m in p.ops.iter().flat_map(|o| o.members()) {
            match &m.kind {
                Kind::Bootstrap(s) => {
                    let r = &p.bts_sites[*s as usize].route;
                    let x = p.vars[m.ins[0] as usize].meta;
                    if RouteKind::of(r, n) == RouteKind::StcFirst && x.level <= entry_level {
                        keys.insert((r.raise_drop, r.cf.unwrap_or(cf_default).max(cf_ref), x.level, x.deg, 0));
                    }
                }
                Kind::Finish { desc: Some(d), .. } => {
                    // a PCMM finish refreshes every output under its own drop, from its inputs' scale
                    if let Some(d) = crate::pcmm::Desc::parse(d) {
                        for drop in d.drops() {
                            for &v in &m.ins {
                                let x = p.vars[v as usize].meta;
                                keys.insert((drop, 0, x.level, x.deg, d.shift + 1));
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    keys.len()
}

/// Words of the held rotation chains' ModUp workspace (`ks::KsWork` digits at the session's `k_max`) when some
/// program has a held chain, else 0 (`Slab::new`).
fn held_words(ps: &PlanSpec, lay: &pc_gpu::Layout, dnum: usize, k_max: usize) -> usize {
    if ps.programs.iter().any(|p| p.rotation_chains().iter().any(|c| c.held)) {
        KsWork::<W>::words(lay, k_max, dnum)[0]
    } else {
        0
    }
}

/// The refresh routes a set of programs uses (`replay_g0::run`'s recipe, over every block).
struct RouteNeeds {
    dense_drops: BTreeSet<u32>,
    real: bool,
    n_dense: usize,
    /// dense sites on the shipped order (no StC-first): `DenseRoute`
    plain_drops: BTreeSet<u32>,
    plain_real: bool,
    sparse: BTreeMap<u32, BTreeSet<u32>>,
    /// classic sparse routes (`Route::sparse_b` off): slots -> raise drops
    classic: BTreeMap<u32, BTreeSet<u32>>,
    spru_h: Option<u32>,
    /// raise drops of the SPRU sites (`SpruCfg::drops` bit set)
    spru_drops: u32,
    /// (level, deg, raise drop) of the left operands of the programs' CCMM finishes (SinCBoot's stc_high shapes)
    ccmm_levels: BTreeSet<(u32, u8, u32)>,
}

fn route_needs(progs: &[Program], n: usize) -> RouteNeeds {
    let mut r = RouteNeeds { dense_drops: BTreeSet::new(), real: false, n_dense: 0, plain_drops: BTreeSet::new(), plain_real: false, sparse: BTreeMap::new(), classic: BTreeMap::new(), spru_h: None, spru_drops: 0, ccmm_levels: BTreeSet::new() };
    for p in progs {
        // a PCMM finish refreshes its outputs on the StC-first dense route (its raise drop's variant)
        for op in &p.ops {
            for m in op.members() {
                if let Kind::Finish { desc: Some(d), .. } = &m.kind {
                    if let Some(d) = crate::pcmm::Desc::parse(d) {
                        r.dense_drops.insert(d.raise_drop);
                        r.dense_drops.extend(d.raise_drops.iter().flatten().copied());
                        r.n_dense += 1;
                    } else if let Some(cd) = crate::matmul::CcmmDesc::parse(d) {
                        // a CCMM's SinCBoot: the StC-first route under its raise drop (PF_CCMM_RAISE)
                        r.dense_drops.insert(cd.raise_drop);
                        r.n_dense += 1;
                        // every left operand: each carries its own SinCBoot state (they may sit at different levels)
                        for &a in m.ins.iter().take(pc_core::ccmm_keys::NBIG) {
                            let am = p.vars[a as usize].meta;
                            r.ccmm_levels.insert((am.level, am.deg, cd.raise_drop));
                        }
                    }
                }
            }
        }
        for s in &p.bts_sites {
            match RouteKind::of(&s.route, n) {
                RouteKind::Spru { h, .. } => {
                    r.spru_h = Some(h);
                    r.spru_drops |= 1 << s.route.raise_drop.min(31);
                }
                RouteKind::Sparse(sl) if !s.route.sparse_b => {
                    r.classic.entry(sl).or_default().insert(s.route.raise_drop);
                }
                RouteKind::Sparse(sl) => {
                    r.sparse.entry(sl).or_default().insert(s.route.raise_drop);
                }
                RouteKind::Dense => {
                    r.plain_drops.insert(s.route.raise_drop);
                    r.plain_real |= s.route.real;
                }
                _ => {
                    r.dense_drops.insert(s.route.raise_drop);
                    r.real |= s.route.real;
                    r.n_dense += 1;
                }
            }
        }
    }
    r
}

fn prog_steps(p: &Program, half: i64) -> BTreeSet<i64> {
    let mut s = BTreeSet::new();
    for op in &p.ops {
        for m in op.members() {
            match &m.kind {
                Kind::Rotate(k) => {
                    s.insert(k.rem_euclid(half));
                }
                Kind::RotateSum(v) => s.extend(v.iter().map(|k| k.rem_euclid(half))),
                _ => {}
            }
        }
    }
    s
}

/// (expander scratch rows, weight frame rows) the programs' streamed-weight sums need (0, 0 = none).
fn stream_needs(ps: &PlanSpec) -> PyResult<(usize, usize)> {
    let shape = ps.shadows.first().map(|c| c.chain).ok_or_else(|| PyValueError::new_err("Session(programs=[]): no program"))?;
    let reg = StreamReg { rows: ps.streams.clone(), pinned: HashMap::new() };
    let mut rows = 0;
    let mut frame = 0;
    for p in &ps.programs {
        rows = rows.max(pc_program::bind::coeff_rows_needed(p, shape));
        let l = pc_program::bind::stream_layout(p, &reg).map_err(|m| PyValueError::new_err(format!("{}: streamed linears not declared in Session(streams=...): {}", p.block, m.join(", "))))?;
        frame = frame.max(l.total as usize);
    }
    if rows > 0 && ps.stream_frames < 2 {
        return Err(crate::ConfigError::new_err(format!("Session(stream_frames={}): >= 2 weight frames", ps.stream_frames)));
    }
    Ok((rows, frame))
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)]
/// The keys and refresh routes a session needs (no device): the key set, every key shape, the route plans.
pub(crate) struct Needs {
    layout: KskLayout,
    steps: BTreeSet<i64>,
    spec: KeySetSpec,
    shapes: Vec<KskShape>,
    needs: Option<RouteNeeds>,
    splan: Option<StcFirstPlan>,
    dplan: Option<pc_gpu::bts::dense::DensePlan>,
    sb_cfgs: Vec<SparseBCfg>,
    spru_cfg: Option<SpruCfg>,
    cplans: Vec<pc_gpu::bts::sparse_classic::ClassicPlan>,
}

fn key_needs(cfg: &pc_config::Config, ks: &PyKeySource, chain: &Chain, plan: &Option<PlanSpec>, pcmm: &Option<crate::pcmm::PcmmPlan>, ccmm: &Option<crate::matmul::CcmmPlan>) -> PyResult<Needs> {
    let e = gerr;
    let n = chain.n;
    let layout = match ks.layout.as_str() {
        "packed" => KskLayout::packed_for::<W>(chain).map_err(e)?,
        _ => KskLayout::Dense,
    };
    let half = (n / 2) as i64;
    let mut steps: BTreeSet<i64> = ks.rot_steps.iter().map(|s| s.rem_euclid(half)).collect();
    for g in &ks.groups {
        steps.extend(g.iter().map(|s| s.rem_euclid(half)));
    }
    // planned: the programs' rotations, every route's LT steps, the dynamic rotations (full rows: the routes'
    // LTs read every limb)
    let mut full: BTreeSet<i64> = BTreeSet::new();
    let mut needs: Option<RouteNeeds> = None;
    let mut splan: Option<StcFirstPlan> = None;
    let mut dplan: Option<pc_gpu::bts::dense::DensePlan> = None;
    let mut sb_cfgs: Vec<SparseBCfg> = vec![];
    let mut spru_cfg: Option<SpruCfg> = None;
    let mut cplans: Vec<pc_gpu::bts::sparse_classic::ClassicPlan> = vec![];
    if let Some(ps) = &plan {
        let rn = route_needs(&ps.programs, n);
        // FIDESLIB_CTS_ALT_BUDGET: the dense routes' raise variants of drop >= 1 on a CtS one stage longer
        let cts_alt = knob(cfg.bts.cts_alt_budget, 0, "bts.cts_alt_budget")? as u32;
        // PC_LT_TAILS (default on) / PC_EM_FUSE_RESCALE (default off): pc-gpu BtsSetup::lt_tails / em_fuse_rescale
        let (lt_tails, em_fuse) = (cfg.bts.lt_tails.unwrap_or(true), cfg.bts.em_fuse_rescale.unwrap_or(false));
        for p in &ps.programs {
            steps.extend(prog_steps(p, half));
        }
        steps.extend(ps.dyn_rot.iter().map(|k| k.rem_euclid(half)));
        if rn.n_dense > 0 {
            let mut setup = BtsSetup::n32_shipped();
            setup.raise_drops = rn.dense_drops.iter().copied().collect();
            setup.real = rn.real;
            // batched PCMM finishes run K-member refreshes on this route (its work buffers)
            setup.k_max = pcmm.as_ref().map_or(1, |p| p.batch.max(1));
            setup.cts_alt_budget = cts_alt;
            (setup.lt_tails, setup.em_fuse_rescale) = (lt_tails, em_fuse);
            let mut sfs = StcFirstSetup::n32_shipped();
            // FIDESLIB_BTS_STC_FIRST_CF: the route's default CF (a site's own Route::cf wins); 13 shipped
            if let Some(cf) = cfg.bts.stc_first_cf {
                sfs.cf_default = Some(cf as u32);
            }
            sfs.cache_slots = ps.stc_cache.max(1);
            if ccmm.is_some() {
                // the CCMM operands' stc_high (50, skip 1) and SinCBoot's stage 0 for every left-operand state, plus
                // the calibration probes' (fresh tokens at the bootstrap output level)
                let shc = ps.shadows.first().ok_or_else(|| PyValueError::new_err("Session(ccmm=True): no program"))?;
                let mut levels = rn.ccmm_levels.clone();
                let land = shc.bts.dense_shipped.or_else(|| shc.bts.landings.get(&0).copied()).map(|l| l.level).unwrap_or(shc.bts.bootstrap_level);
                levels.insert((land, 1, 0));
                let shapes = crate::matmul::high_shapes(shc.chain, sfs.cf_default.unwrap_or(13), &levels)?;
                sfs.high = Some(pc_gpu::bts::stc_high::StcHighSetup { shapes, slots: ps.stc_cache.max(4) });
            }
            let mut sp = StcFirstPlan::new(chain, &setup, &sfs).map_err(e)?;
            // a captured graph reads the entries its refreshes used: the cache holds every key the programs ask for
            let keys = stc_first_keys(&ps.programs, n, (chain.q.len() - 1 - sp.entry_top) as u32, sp.sf.cf_default.unwrap_or(13), sp.sf.cf_ref);
            sp.sf.cache_slots = sp.sf.cache_slots.max(keys + 4);
            let mut geoms: Vec<_> = sp.dense.variants.iter().flat_map(|v| v.cts.iter().chain(&v.stc)).chain(&sp.input).map(|h| h.geom).collect();
            geoms.extend(sp.sources.iter().flatten().map(|s| s.geom));
            full.extend(lt_steps(&geoms, n));
            splan = Some(sp);
        }
        if !rn.plain_drops.is_empty() {
            let mut setup = BtsSetup::n32_shipped();
            setup.raise_drops = rn.plain_drops.iter().copied().collect();
            setup.real = rn.plain_real;
            setup.cts_alt_budget = cts_alt;
            (setup.lt_tails, setup.em_fuse_rescale) = (lt_tails, em_fuse);
            setup.k_max = 1;
            let dp = pc_gpu::bts::dense::DensePlan::new(chain, &setup).map_err(e)?;
            let geoms: Vec<_> = dp.variants.iter().flat_map(|v| v.cts.iter().chain(&v.stc).chain(v.stc_real0.iter())).map(|h| h.geom).collect();
            full.extend(lt_steps(&geoms, n));
            dplan = Some(dp);
        }
        for (&sl, drops) in &rn.sparse {
            for &d in drops {
                let mut c = SparseBCfg::n32(sl, d);
                c.k_max = 1;
                (c.lt_tails, c.em_fuse_rescale) = (lt_tails, em_fuse);
                full.extend(pc_gpu::bts::sparse_b::rot_steps(n, &c).map_err(e)?.into_iter().map(|k| k.rem_euclid(half)));
                sb_cfgs.push(c);
            }
        }
        if !rn.classic.is_empty() {
            let budget = pc_program::compile::sparse_level_budget(cfg).map_err(|x| crate::ConfigError::new_err(x.to_string()))?;
            for (&sl, drops) in &rn.classic {
                let mut c = pc_gpu::bts::sparse_classic::ClassicCfg::n32(sl, &drops.iter().copied().collect::<Vec<_>>());
                c.level_budget = budget;
                let cp = pc_gpu::bts::sparse_classic::ClassicPlan::new(chain, &c).map_err(e)?;
                full.extend(cp.rot_steps());
                cplans.push(cp);
            }
        }
        if let Some(h) = rn.spru_h {
            let c = SpruCfg { drops: rn.spru_drops, ..SpruCfg::new(h as usize) };
            full.extend(c.rot_steps(n).into_iter().map(|k| k.rem_euclid(half)));
            spru_cfg = Some(c);
        }
        full.remove(&0);
        needs = Some(rn);
    }
    steps.remove(&0);
    let rots: Vec<RotSpec> = steps.union(&full).map(|&step| RotSpec { step, q_top: if full.contains(&step) { None } else { ks.band } }).collect();
    let steps: BTreeSet<i64> = steps.union(&full).copied().collect();
    // the StC-first route's raise needs the encapsulation keys too (a dense-only plan has no other route that does)
    let spec = KeySetSpec { relin: true, conj: ks.conj || plan.is_some(), rots, encaps: !sb_cfgs.is_empty() || dplan.is_some() || !cplans.is_empty() || splan.is_some() };
    let mut shapes: Vec<_> = spec.keys(chain).map_err(e)?.into_iter().map(|(_, s)| s).collect();
    if spru_cfg.is_some() {
        shapes.push(KskShape::bottom(chain).map_err(e)?);
        shapes.push(KskShape::full(chain));
    }
    Ok(Needs { layout, steps, spec, shapes, needs, splan, dplan, sb_cfgs, spru_cfg, cplans })
}

/// Where a session's keys come from.
pub(crate) enum Origin {
    /// A secret (fresh, or `KeySource.generate(secret=)`) and every key generated on the device.
    Generate,
    /// A client's key bundle: no secret on this side (no decryptor, encryptions with the bundle's public key).
    Bundle(std::path::PathBuf),
    /// A client's key bundle and that client's secret (checks of client keys: decrypt available).
    DebugBundle(std::path::PathBuf, Vec<u8>),
}

/// The key bundle request of a session: every switching key of `Needs` (+ the SPRU route's material), the PCMM
/// shapes, the CCMM keys, a public key; chain identity and the plan contract (hash of the load-bearing Config).
fn manifest(cfg: &pc_config::Config, chain: &Chain, nd: &Needs, pcmm: &Option<crate::pcmm::PcmmPlan>, ccmm: bool) -> PyResult<pc_client::KeySpec> {
    let e = gerr;
    let mut keys: Vec<pc_client::KeyReq> = nd.spec.keys(chain).map_err(e)?.iter().map(|(r, s)| key_req(r, s)).collect();
    let spru = match &nd.spru_cfg {
        Some(c) => {
            keys.push(pc_client::KeyReq { role: pc_client::Role::SpruAtob, shape: shape_req(&KskShape::bottom(chain).map_err(e)?) });
            keys.push(pc_client::KeyReq { role: pc_client::Role::Auto { galois: 2 * chain.n as u64 - 5 }, shape: shape_req(&KskShape::full(chain)) });
            let st = pc_core::scale::ScaleTables::precompute(chain).map_err(|x| e(x.into()))?;
            let nf = st.sf((chain.q.len() - 1) as i32).map_err(|x| e(x.into()))?;
            Some(pc_client::SpruReq { h: c.h, n: c.n, cs_level: 0, cs_nf: nf })
        }
        None => None,
    };
    Ok(pc_client::KeySpec {
        chain: "n32".into(),
        params_hash: pc_client::spec::params_hash(chain),
        contract: contract_hash(cfg)?,
        keys,
        spru,
        pcmm_t: pcmm.as_ref().map(|p| p.ts()).unwrap_or_default(),
        ccmm,
        public_key: true,
    })
}

/// A client error as a Python ValueError (wire format, spec, chain mismatch).
pub(crate) fn cerr(e: pc_client::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// What a bundle carries besides the key set's keys, kept until the routes and services that take it are built.
pub(crate) struct BundleParts {
    /// SPRU: (atob seed, packed b), (conj o rot(1) seed, packed b), (key ciphertexts: rows, nf, words)
    spru_atob: Option<(pc_core::keys::KskSeed, Vec<u32>)>,
    spru_auto: Option<(pc_core::keys::KskSeed, Vec<u32>)>,
    spru_cs: Option<(usize, usize, f64, Vec<u32>)>,
    pcmm: Vec<pc_core::pcmm_keys::PcmmKeys>,
    ccmm: Option<pc_core::ccmm_keys::CcmmKeys>,
    pk: Option<Vec<u32>>,
    bits: u32,
    row_words: usize,
}

impl BundleParts {
    /// The SPRU route on the bundle's material.
    fn spru(&mut self, a: &'static Arena, ctx: &Context<'static, W, M>, s: &pc_cuda::Stream<pc_cuda::Compute>, c: &SpruCfg) -> PyResult<Spru<'static, W>> {
        let e = gerr;
        let missing = || PyValueError::new_err("key bundle: the plan's SPRU route needs the SPRU material (spru_cs, spru_atob, auto 2N-5)");
        let (atob_seed, atob_b) = self.spru_atob.take().ok_or_else(missing)?;
        let (auto_seed, auto_b) = self.spru_auto.take().ok_or_else(missing)?;
        let (k, rows, nf, cs_words) = self.spru_cs.take().ok_or_else(missing)?;
        let (mut atob, mut cs, mut conjrot) = Spru::<W>::carve_material(a, ctx, c).map_err(e)?;
        pc_gpu::keys::import_packed(&mut atob, s, &ctx.chain, atob_seed, self.bits, self.row_words, &atob_b).map_err(e)?;
        pc_gpu::keys::import_packed(&mut conjrot, s, &ctx.chain, auto_seed, self.bits, self.row_words, &auto_b).map_err(e)?;
        let n = ctx.lay.n;
        if k != cs.k() || rows != ctx.lay.nq || (nf / cs.meta.nf - 1.0).abs() > 1e-12 {
            return Err(PyValueError::new_err(format!("key bundle: SPRU key ciphertexts {k} x {rows} rows at scale {nf:e}; the route takes {} x {} at {:e}", cs.k(), ctx.lay.nq, cs.meta.nf)));
        }
        let ids: Vec<usize> = (0..rows).collect();
        for c in 0..k {
            for part in 0..2 {
                let o = (c * 2 + part) * rows * n;
                cs.buf.upload_rows(s, c, part, &ids, &cs_words[o..o + rows * n]).map_err(e)?;
            }
        }
        Spru::<W>::from_material(a, ctx, s, *c, atob, cs, conjrot).map_err(e)
    }
}

/// Import a client's key bundle: header against the request (chain, params, contract, exactly the requested keys),
/// every key-set key into its carved slot (`import_packed`), the rest kept for the routes and services.
fn import_bundle(path: &std::path::Path, req: &pc_client::KeySpec, chain: &Chain, s: &pc_cuda::Stream<pc_cuda::Compute>, keys: &mut KeySet<'static, W>) -> PyResult<BundleParts> {
    use pc_client::{Entry, Role as R};
    let e = gerr;
    let mut rd = pc_client::BundleReader::open(path).map_err(cerr)?;
    let h = rd.header.clone();
    if h.chain != req.chain || h.params_hash != req.params_hash || h.n != chain.n {
        return Err(PyValueError::new_err(format!("key bundle for chain {} ({}, n {}); this session runs {} ({})", h.chain, h.params_hash, h.n, req.chain, req.params_hash)));
    }
    if h.contract != req.contract {
        return Err(PyValueError::new_err(format!("key bundle made for plan contract {}; this session's Config is {} (both sides must run the same load-bearing Config)", h.contract, req.contract)));
    }
    let have: Vec<pc_client::KeyReq> = h.entries.iter().filter_map(|x| if let Entry::Ksk { key, .. } = x { Some(*key) } else { None }).collect();
    if have != req.keys {
        let miss: Vec<_> = req.keys.iter().filter(|k| !have.contains(k)).take(4).collect();
        let extra: Vec<_> = have.iter().filter(|k| !req.keys.contains(k)).take(4).collect();
        return Err(PyValueError::new_err(format!("key bundle keys differ from the session's request ({} vs {}): missing {miss:?}, unexpected {extra:?}", have.len(), req.keys.len())));
    }
    let mut parts = BundleParts { spru_atob: None, spru_auto: None, spru_cs: None, pcmm: vec![], ccmm: None, pk: None, bits: h.bits, row_words: h.row_words };
    let half = (chain.n / 2) as i64;
    for (i, ent) in h.entries.iter().enumerate() {
        let words = rd.read(i).map_err(cerr)?;
        match ent {
            Entry::Ksk { key, seed, .. } => {
                let seed = pc_core::keys::KskSeed(*seed);
                let slot = match key.role {
                    R::Relin => keys.relin.as_mut(),
                    R::Conj => keys.conj.as_mut(),
                    R::Rot { step, .. } => keys.rots.get_mut(&step.rem_euclid(half)),
                    R::EncDenseToSparse => keys.bts.as_mut().map(|b| &mut b.enc_dense_to_sparse),
                    R::EncSparseToDense => keys.bts.as_mut().map(|b| &mut b.enc_sparse_to_dense),
                    R::SpruAtob => {
                        parts.spru_atob = Some((seed, words));
                        continue;
                    }
                    R::Auto { .. } => {
                        parts.spru_auto = Some((seed, words));
                        continue;
                    }
                };
                let k = slot.ok_or_else(|| PyValueError::new_err(format!("key bundle: no carved key for {:?}", key.role)))?;
                if shape_req(&k.shape) != key.shape {
                    return Err(PyValueError::new_err(format!("key bundle: {:?} of shape {:?}, the session carved {:?}", key.role, key.shape, k.shape)));
                }
                pc_gpu::keys::import_packed(k, s, chain, seed, h.bits, h.row_words, &words).map_err(e)?;
            }
            Entry::SpruCs { k, rows, nf, .. } => parts.spru_cs = Some((*k, *rows, *nf, words)),
            Entry::Pcmm { t, primes, split, fwd, back, pack, .. } => {
                let (f, rest) = words.split_at(*fwd);
                let (b, p) = rest.split_at(*back);
                if p.len() != *pack {
                    return Err(PyValueError::new_err(format!("key bundle: pcmm t = {t} sizes")));
                }
                let shape = pc_core::pcmm_keys::Shape::new(*t).map_err(|x| e(x.into()))?;
                parts.pcmm.push(pc_core::pcmm_keys::PcmmKeys { shape, primes: *primes, split: *split, fwd: f.to_vec(), back: b.to_vec(), pack: p.to_vec() });
            }
            Entry::Ccmm { primes, fwd, back, autos, rlk, .. } => {
                let (f, r1) = words.split_at(*fwd);
                let (b, r2) = r1.split_at(*back);
                let (au, rl) = r2.split_at(*autos);
                if rl.len() != *rlk {
                    return Err(PyValueError::new_err("key bundle: ccmm sizes"));
                }
                parts.ccmm = Some(pc_core::ccmm_keys::CcmmKeys::from_arrays(primes, f.to_vec(), b.to_vec(), au.to_vec(), rl.to_vec()).map_err(|x| e(x.into()))?);
            }
            Entry::PublicKey { rows, .. } => {
                if *rows != chain.q.len() {
                    return Err(PyValueError::new_err(format!("key bundle: public key of {rows} rows, the chain has {}", chain.q.len())));
                }
                parts.pk = Some(words);
            }
        }
    }
    let ts: Vec<usize> = parts.pcmm.iter().map(|k| k.shape.t).collect();
    if ts != req.pcmm_t || parts.ccmm.is_some() != req.ccmm {
        return Err(PyValueError::new_err(format!("key bundle: PCMM keys for t = {ts:?} / CCMM {}; the session needs {:?} / {}", parts.ccmm.is_some(), req.pcmm_t, req.ccmm)));
    }
    s.sync().map_err(|x| e(x.into()))?;
    Ok(parts)
}

/// The plan contract both sides check: FNV-1a 64 of the load-bearing Config's JSON.
pub(crate) fn contract_hash(cfg: &pc_config::Config) -> PyResult<String> {
    let j = cfg.load_bearing().to_json().map_err(crate::config_err)?;
    Ok(format!("{:016x}", pc_client::spec::fnv1a64(j.as_bytes())))
}

fn shape_req(s: &KskShape) -> pc_client::KeyShape {
    pc_client::KeyShape { q_top: s.q_top, ndig: s.ndig, nq: s.nq, kp: s.kp }
}

fn key_req(r: &KeyRole, s: &KskShape) -> pc_client::KeyReq {
    use pc_client::Role as R;
    let role = match *r {
        KeyRole::Relin => R::Relin,
        KeyRole::Rot { step, galois } if step == 0 => R::Auto { galois },
        KeyRole::Rot { step, galois } => R::Rot { step, galois },
        KeyRole::Conj => R::Conj,
        KeyRole::EncDenseToSparse => R::EncDenseToSparse,
        KeyRole::EncSparseToDense => R::EncSparseToDense,
    };
    pc_client::KeyReq { role, shape: shape_req(s) }
}

/// `(rows, count)` classes of the Q rows `0..=top` that ciphertexts / plaintexts at `levels` need.
fn level_classes(nq: usize, levels: &[u32], knob: &str) -> PyResult<Vec<(usize, usize)>> {
    let mut h: BTreeMap<usize, usize> = BTreeMap::new();
    for &l in levels {
        let rows = nq.checked_sub(l as usize).filter(|&r| r > 0).ok_or_else(|| PyValueError::new_err(format!("Session({knob}=...): level {l} is past the bottom of a {nq}-prime chain")))?;
        *h.entry(rows).or_default() += 1;
    }
    Ok(h.into_iter().collect())
}

/// Each register's layout: Q rows `0..=top` of the richest level it carries and no special rows (registers are
/// only copied into and out of the slab), else the full layout.
fn reg_layouts(lay: pc_gpu::Layout, ps: &PlanSpec) -> PyResult<Vec<pc_gpu::Layout>> {
    if ps.reg_levels.is_empty() {
        return Ok(vec![lay; ps.regs]);
    }
    if ps.reg_levels.len() != ps.regs {
        return Err(PyValueError::new_err(format!("Session(reg_levels=...): {} levels for {} registers", ps.reg_levels.len(), ps.regs)));
    }
    ps.reg_levels
        .iter()
        .map(|&l| match lay.nq.checked_sub(l as usize) {
            Some(rows) if rows > 0 => Ok(pc_gpu::Layout { n: lay.n, nq: rows, kp: 0 }),
            _ => Err(PyValueError::new_err(format!("Session(reg_levels=...): level {l} is past the bottom of a {}-prime chain", lay.nq))),
        })
        .collect()
}

fn setup(cfg: &pc_config::Config, ks: &PyKeySource, origin: &Origin, sizes: Sizes, geometry: Geometry, debug_refresh: bool, plan: Option<PlanSpec>, pcmm: Option<crate::pcmm::PcmmPlan>, ccmm: Option<crate::matmul::CcmmPlan>) -> PyResult<Dev> {
    let t0 = std::time::Instant::now();
    let e = gerr;
    let sc = shadow_config(cfg, None, "runtime", false)?;
    let complex = cfg.chain.ckks_complex.unwrap_or(false);
    let radix_bits = knob(cfg.model.ladder_radix_bits, 3, "model.ladder_radix_bits")? as u32;
    let device = Device::open().map_err(|x| e(x.into()))?;
    let chain = Chain::generate(&ChainSpec::n32()).map_err(|x| e(x.into()))?;
    let n = chain.n;
    let half = (n / 2) as i64;
    let k_max = 8;
    let mcfg = MemCfg { k_max, pool: vec![PoolClass { k: 1, count: sizes.pool }], events: 64, scratch_cts: 2 };
    let mut mplan = MemPlan::build::<W, M>(&chain, mcfg).map_err(e)?;
    let lay = mplan.lay;
    let nd = key_needs(cfg, ks, &chain, &plan, &pcmm, &ccmm)?;
    let spec_req = match origin {
        Origin::Generate => None,
        _ => Some(manifest(cfg, &chain, &nd, &pcmm, ccmm.is_some())?),
    };
    let Needs { layout, steps, spec, shapes, needs, splan, dplan, sb_cfgs, spru_cfg, cplans } = nd;
    let mp = &mut mplan;
    mp.reserve("ks.plans", KsPlans::bytes::<W>(&chain, k_max).map_err(e)?);
    mp.reserve("keys", spec.bytes::<W>(&chain, layout).map_err(e)?);
    if matches!(origin, Origin::Generate) {
        mp.reserve("keygen", KeyGen::<W>::bytes(&lay, &shapes));
    }
    let nrot = steps.len() + ks.conj as usize;
    let gb: usize = ks.groups.iter().map(|g| KeyGroup::bytes(g.len().max(1), true)).sum::<usize>() + KeyGroup::bytes(nrot.max(1), true) + 2 * KeyGroup::bytes(1, true);
    mp.reserve("ks.groups", gb);
    mp.reserve("ks.regen_a", KsWork::<W>::regen_bytes(&lay, chain.dnum()));
    // planned blocks push every per-call constant through the ring (scalar rows, raise adjusts) and a captured
    // graph keeps the slots it leased: room for every block's
    let ocfg = match &plan {
        Some(ps) => OpsCfg { const_slots: const_slots(ps), ..OpsCfg::default() },
        None => OpsCfg::default(),
    };
    Ops::<W>::reserve(mp, &ocfg);
    let pcfg = PtStoreCfg {
        slots: sizes.pt_slots,
        ring: 4,
        encode_threads: sizes.encode_threads,
        policy: if plan.is_some() { PtStorePolicy::ResidentAll } else { PtStorePolicy::StageAhead },
        eager: pc_gpu::encode_dev::EncoderKind::DeviceFft,
        rows: plan.as_ref().map(|p| p.pt_rows).unwrap_or(0),
        classes: match &plan {
            Some(p) => level_classes(lay.nq, &p.pt_levels, "pt_levels")?,
            None => Vec::new(),
        },
    };
    PtStore::<W>::reserve(mp, &pcfg);
    let with_secret = !matches!(origin, Origin::Bundle(_));
    mp.reserve("encrypt", Encryptor::<W>::bytes(&lay, !with_secret));
    if with_secret {
        mp.reserve("decrypt", Decryptor::<W>::bytes(&lay, 2, 1) + Decryptor::<W>::bytes(&lay, 4, 1));
        mp.reserve("keys.secret", DevSecret::<W>::bytes(&lay));
    }
    // planned: route tables + work, the shared slab, the registers, the programs' key groups
    let mut wshape: Option<BtsWorkShape> = None;
    let mut group_bytes = 0;
    if let Some(ps) = &plan {
        if let Some(sp) = &splan {
            let (req, w) = sp.mem::<W>().map_err(e)?;
            req.reserve(mp);
            wshape = Some(w);
        }
        if let Some(dp) = &dplan {
            let (req, w) = dp.mem::<W>().map_err(e)?;
            req.reserve(mp);
            mp.reserve("bts.route", pc_gpu::bts::raise::EncapsKeys::bytes() + pc_gpu::bts::finish::Pow2::<W>::bytes(lay.rows()) + KeyGroup::bytes(1, true));
            wshape = Some(wshape.map_or(w, |x| x.max(w)));
        }
        let st_tables = pc_core::scale::ScaleTables::precompute(&chain).map_err(|x| e(x.into()))?;
        for c in &sb_cfgs {
            let (b, w) = SparseBPrecomp::<W>::bytes(&chain, &st_tables, &lay, c).map_err(e)?;
            mp.reserve("bts.sparse", b);
            mp.reserve("bts.route", pc_gpu::bts::raise::EncapsKeys::bytes() + pc_gpu::bts::finish::Pow2::<W>::bytes(lay.rows()));
            wshape = Some(wshape.map_or(w, |x| x.max(w)));
        }
        for cp in &cplans {
            let (req, w) = cp.mem::<W>().map_err(e)?;
            req.reserve(mp);
            wshape = Some(wshape.map_or(w, |x| x.max(w)));
        }
        if let Some(c) = &spru_cfg {
            mp.reserve("bts.spru", Spru::<W>::bytes(&lay, c, &chain).map_err(e)? + Spru::<W>::bind_bytes(c, n));
            if matches!(origin, Origin::Generate) {
                mp.reserve("keys.secret", DevSecret::<W>::bytes(&lay));
            }
            let w = BtsWorkShape { k_max: 1, g_max: 1, em_slots: 0, em_kk: 1, spare: 0 };
            wshape = Some(wshape.map_or(w, |x| x.max(w)));
        }
        let w = wshape.unwrap_or(BtsWorkShape { k_max: 1, g_max: 1, em_slots: 0, em_kk: 1, spare: 0 });
        wshape = Some(w);
        mp.reserve("bts.work", BtsWork::<W>::bytes(&lay, w));
        let n_slab = ps.programs.iter().map(|p| p.mem.slot_bytes.len()).max().unwrap_or(0);
        mp.reserve("replay.slab", Slab::<W>::bytes(&lay, n_slab, held_words(ps, &lay, chain.dnum(), k_max)));
        let reg_bytes: usize = reg_layouts(lay, ps)?.iter().map(|l| Arena::bytes_for::<W>(pc_gpu::CtBuf::<pc_cuda::Compute, W>::words(l, 1))).sum();
        mp.reserve("replay.regs", reg_bytes);
        group_bytes = ps.programs.iter().map(|p| PBound::<W>::group_bytes(p, ps.dyn_rot.len())).sum();
        mp.reserve("replay.groups", group_bytes);
        if let Some(pp) = &pcmm {
            pp.reserve(mp, lay.rows())?;
        }
        // streamed weights: the expander's scratch at the deepest sum's limbs, the weight frames
        let (coeff_rows, frame_rows) = stream_needs(ps)?;
        if coeff_rows > 0 {
            mp.reserve("encode.coeff", pc_gpu::encode_dev::CoeffRowsDevice::<W>::bytes(&lay, coeff_rows));
            mp.reserve("stream", pc_program::residency::StreamRing::bytes(n, &pc_program::residency::StreamRingCfg { frames: ps.stream_frames, frame_rows }));
        }
        if let Some((_, _, pages)) = ps.kv {
            mp.reserve("kv", pc_program::kv::PinnedKv::<W>::bytes(&lay, pages));
        }
        // the side lane: a second executor's workspaces and op layer
        let side_bytes = mp.ks_work + mp.bconv_work + mp.scratch + KsWork::<W>::regen_bytes(&lay, chain.dnum());
        mp.reserve("replay.side", side_bytes);
        Ops::<W>::reserve(mp, &ocfg);
        if let Some(cp) = &ccmm {
            cp.reserve(mp, chain.q.len(), n)?;
        }
        mp.reserve("slack", 256 << 20);
    }
    let mem_report = mplan.report();
    let arena = Box::new(mplan.allocate(&device).map_err(e)?);
    // SAFETY: the box is owned by the `Dev` built below together with every borrower, and dropped after them
    let a: &'static Arena = unsafe { &*(arena.as_ref() as *const Arena) };
    let mut ex = Executor::<W>::new(&device, a, &mplan).map_err(e)?;
    ex.ks.carve_regen(a, &lay).map_err(e)?;
    let ctx = Context::<W, M>::new(a, &ex.copy, chain.clone(), &mplan).map_err(e)?;
    let pool = CtPool::<W>::new(a, &mplan).map_err(e)?;
    let plans = KsPlans::new(a, &ex.compute, &ctx, k_max).map_err(e)?;
    let mut keys = KeySet::carve(a, &chain, &spec, layout).map_err(e)?;
    let mut prng = match ks.seed {
        Some(s) => Prng::from_seed([0x5e55_1013, s as u32, (s >> 32) as u32, 4, 5, 6, 7, 8], 0),
        None => Prng::from_os().map_err(|x| e(x.into()))?,
    };
    let host_secret = |b: &[u8]| -> PyResult<SecretKey> { Ok(pc_client::ClientSecret::from_bytes(b).map_err(cerr)?.sk) };
    // the host secret: sampled or given (Generate), the client's (DebugBundle), none (Bundle)
    let sk: Option<SecretKey> = match origin {
        Origin::Generate => Some(match &ks.secret {
            Some(b) => host_secret(b)?,
            None => SecretKey::sample(&mut prng, n).map_err(|x| e(x.into()))?,
        }),
        Origin::DebugBundle(_, b) => Some(host_secret(b)?),
        Origin::Bundle(_) => None,
    };
    let mut secret = match &sk {
        Some(sk) => {
            let mut d = DevSecret::carve(a, lay).map_err(e)?;
            d.load(&ctx, &ex.compute, sk).map_err(e)?;
            Some(d)
        }
        None => None,
    };
    let tk = std::time::Instant::now();
    let (mut kg, mut bundle) = match (origin, &spec_req) {
        (Origin::Generate, _) => {
            let mut kg = KeyGen::new(a, &ex.compute, &chain, &shapes).map_err(e)?;
            let sec = secret.as_ref().ok_or_else(|| PyRuntimeError::new_err("device keygen without a secret"))?;
            kg.generate_set(&ctx, &mut ex, sec, &mut keys, &mut prng).map_err(e)?;
            (Some(kg), None)
        }
        (Origin::Bundle(path) | Origin::DebugBundle(path, _), Some(req)) => (None, Some(import_bundle(path, req, &chain, &ex.compute, &mut keys)?)),
        _ => return Err(PyRuntimeError::new_err("bundle session without its key request")),
    };
    let keygen_s = tk.elapsed().as_secs_f64();
    let stage = ex.ks.stage_ptrs();
    let groups = {
        let mut rot_refs: Vec<&dyn KsKeyRef> = keys.rots.values().map(|k| k as &dyn KsKeyRef).collect();
        if let Some(c) = keys.conj.as_ref() {
            rot_refs.push(c);
        }
        let all = if rot_refs.is_empty() { None } else { Some(KeyGroup::new(a, &ex.compute, n, &rot_refs, stage).map_err(e)?) };
        let mut exact = BTreeMap::new();
        for g in &ks.groups {
            let refs = g
                .iter()
                .filter(|s| s.rem_euclid(half) != 0)
                .map(|&s| keys.rot(s).map(|k| k as &dyn KsKeyRef))
                .collect::<pc_gpu::Result<Vec<_>>>()
                .map_err(e)?;
            if !refs.is_empty() {
                exact.insert(g.clone(), KeyGroup::new(a, &ex.compute, n, &refs, stage).map_err(e)?);
            }
        }
        let relin = KeyGroup::new(a, &ex.compute, n, &[keys.relin().map_err(e)? as &dyn KsKeyRef], stage).map_err(e)?;
        let conj = match keys.conj.as_ref() {
            Some(c) => Some(KeyGroup::new(a, &ex.compute, n, &[c as &dyn KsKeyRef], stage).map_err(e)?),
            None => None,
        };
        Groups { all, steps: steps.clone(), exact, relin, conj, half }
    };
    let ops = Ops::new(a, &device, &ex.compute, &ctx, ocfg).map_err(e)?;
    let pt_events = Box::new(EventPool::new(&device, 64).map_err(|x| e(x.into()))?);
    // SAFETY: `pt_events` is boxed and declared after `pts` in State, so it outlives the store's borrow
    let ev: &'static EventPool = unsafe { &*(pt_events.as_ref() as *const EventPool) };
    let pts = PtStore::new(a, &device, &ex.compute, &ctx, ev, pcfg).map_err(e)?;
    let mut enc = Encryptor::new(a, &ex.compute, lay, secret.is_none(), &mut prng).map_err(e)?;
    if let (Some(pk), Some(b)) = (enc.pk.as_mut(), bundle.as_mut()) {
        let words = b.pk.take().ok_or_else(|| PyValueError::new_err("key bundle: no public key (the server encrypts public values with it)"))?;
        let nq = chain.q.len();
        let rows: Vec<usize> = (0..nq).collect();
        for part in 0..2 {
            pk.buf.upload_rows(&ex.compute, 0, part, &rows, &words[part * nq * n..(part + 1) * nq * n]).map_err(e)?;
        }
    }
    let secret_side = match secret.take() {
        Some(sec) => {
            let mut dec2 = Decryptor::new(a, &ex.compute, lay, 2, 1).map_err(e)?;
            let mut dec4 = Decryptor::new(a, &ex.compute, lay, 4, 1).map_err(e)?;
            dec2.load_secret(&ex.compute, &sec).map_err(e)?;
            dec4.load_secret(&ex.compute, &sec).map_err(e)?;
            Some(SecretSide { secret: sec, dec2, dec4 })
        }
        None => None,
    };
    let tables = Arc::new(ctx.scale.clone());
    // planned: the routes on the generated keys, the slab, the registers (zeroed: a KV input before its first
    // write reads an encryption of 0 at any level)
    let planned = match plan {
        None => None,
        Some(ps) => {
            let shadows: Vec<Shadow> = ps.shadows.iter().map(|c| Shadow::new(c.clone(), Some(tables.clone()))).collect();
            let sh0 = shadows.first().cloned().ok_or_else(|| PyValueError::new_err("Session(programs=[]): no program"))?;
            let dense = match &splan {
                Some(sp) => Some(sp.build(a, &device, &ex.compute, &ctx, &keys, stage).map_err(e)?),
                None => None,
            };
            let plain = match &dplan {
                Some(dp) => Some(dp.build(a, &ex.compute, &ctx, &keys, stage).map_err(e)?),
                None => None,
            };
            let mut sparse = vec![];
            for c in &sb_cfgs {
                let s = &ex.compute;
                let encaps = pc_gpu::bts::raise::EncapsKeys::new(a, s, n, BtsKeyRing::dense_to_sparse(&keys).map_err(e)?, BtsKeyRing::sparse_to_dense(&keys).map_err(e)?, stage).map_err(e)?;
                let pow2 = pc_gpu::bts::finish::Pow2::new(a, s, &ctx).map_err(e)?;
                let pre = SparseBPrecomp::new(a, s, &ctx, c.clone(), &keys, stage).map_err(e)?;
                sparse.push(SparseBRoute { slots: c.slots, ring_n: n, variants: vec![pre], encaps, pow2, cf_default: 6, cf_window: (2, 20), deg: chain_deg(&ctx).map_err(e)?, ks_source: c.ks_source });
            }
            let mut classic = vec![];
            for cp in &cplans {
                let r = cp.build(a, &ex.compute, &ctx, &keys, stage).map_err(e)?;
                r.check_mem().map_err(e)?;
                classic.push(r);
            }
            let mut spru_secret = None;
            let spru = match &spru_cfg {
                Some(c) => {
                    let mut sp = match (&mut kg, bundle.as_mut()) {
                        (Some(kg), _) => {
                            let (sec, sk) = secret_side.as_ref().zip(sk.as_ref()).ok_or_else(|| PyRuntimeError::new_err("SPRU keygen without a secret"))?;
                            let mut ssec = DevSecret::<W>::carve(a, lay).map_err(e)?;
                            let skey = SpruKey::sample(&mut prng, n, c).map_err(e)?;
                            let sp = Spru::<W>::new(a, &ctx, &mut ex, kg, &mut enc, &sec.secret, &mut ssec, sk, &skey, c.clone(), &sh0, &mut prng).map_err(e)?;
                            spru_secret = Some(ssec);
                            sp
                        }
                        (None, Some(b)) => b.spru(a, &ctx, &ex.compute, c)?,
                        (None, None) => return Err(PyRuntimeError::new_err("SPRU route without key material")),
                    };
                    sp.bind(a, &ex.compute, &keys, stage).map_err(e)?;
                    Some(SpruRoute { spru: RefCell::new(sp), ring_n: n, cf_default: 6, deg: chain_deg(&ctx).map_err(e)?, carved: 0 })
                }
                None => None,
            };
            let work = BtsWork::<W>::new(a, lay, wshape.unwrap_or(BtsWorkShape { k_max: 1, g_max: 1, em_slots: 0, em_kk: 1, spare: 0 })).map_err(e)?;
            let n_slab = ps.programs.iter().map(|p| p.mem.slot_bytes.len()).max().unwrap_or(0);
            let slab = Slab::<W>::new(a, lay, n_slab, held_words(&ps, &lay, chain.dnum(), k_max)).map_err(|x| PyRuntimeError::new_err(x.to_string()))?;
            let mut regs = vec![];
            for rl in reg_layouts(lay, &ps)? {
                let mut c = Ct::new(pc_shadow::CtMeta::new(0, 1), pc_gpu::CtBuf::new(a, "replay.regs", rl, 1).map_err(e)?);
                c.buf.buf.zero(&ex.compute).map_err(|x| e(x.into()))?;
                regs.push(c);
            }
            let _ = &needs;
            // the PCMM keys are made here, from the host secret (the driver drops it after setup)
            let pc = match pcmm {
                Some(pp) => {
                    if dense.is_none() {
                        return Err(PyValueError::new_err("Session(pcmm=...): no program has a PCMM finish (no StC-first route)"));
                    }
                    let tp = std::time::Instant::now();
                    let pk = match bundle.as_mut() {
                        Some(b) => std::mem::take(&mut b.pcmm),
                        None => {
                            let sk = sk.as_ref().ok_or_else(|| PyRuntimeError::new_err("PCMM keygen without a secret"))?;
                            pp.ts().iter().map(|&t| pc_core::pcmm_keys::PcmmKeys::generate(&sk.s, &chain.q[..4], t, &mut prng)).collect::<Result<Vec<_>, _>>().map_err(|x| e(x.into()))?
                        }
                    };
                    Some(crate::pcmm::PcmmRt::build(pp, a, &device, &ex.compute, &ctx, pk, tp.elapsed().as_secs_f64())?)
                }
                None => None,
            };
            // the CCMM keys too
            let cc = match ccmm {
                Some(cp) => {
                    let d = dense.as_ref().ok_or_else(|| PyValueError::new_err("Session(ccmm=True): no program has a CCMM finish (no StC-first route)"))?;
                    let cf = d.sf.cf_default.unwrap_or(13);
                    let tc = std::time::Instant::now();
                    let ck = match bundle.as_mut() {
                        Some(b) => b.ccmm.take().ok_or_else(|| PyValueError::new_err("key bundle: no CCMM keys"))?,
                        None => {
                            let sk = sk.as_ref().ok_or_else(|| PyRuntimeError::new_err("CCMM keygen without a secret"))?;
                            pc_core::ccmm_keys::CcmmKeys::generate(&sk.s, &chain.q[..4], &mut prng).map_err(|x| e(x.into()))?.0
                        }
                    };
                    Some(crate::matmul::CcmmRt::build(cp, a, &device, &ex.compute, &ctx, ck, tc.elapsed().as_secs_f64(), cf)?)
                }
                None => None,
            };
            let replayer = Replayer::new(ReplayCfg { use_graphs: ps.use_graphs, check_meta: ps.check_meta, graphs_per_block: 2 });
            let (coeff_rows, frame_rows) = stream_needs(&ps)?;
            let stream_events = Box::new(EventPool::new(&device, ps.stream_frames + ps.kv.map_or(0, |k| k.2) + 16).map_err(|x| e(x.into()))?);
            // SAFETY: boxed and declared after `ring` / `kv` in Planned, so it outlives their borrows
            let sev: &'static EventPool = unsafe { &*(stream_events.as_ref() as *const EventPool) };
            let (coeff, ring) = if coeff_rows > 0 {
                let c = pc_gpu::encode_dev::CoeffRowsDevice::new(a, &ex.compute, lay, coeff_rows).map_err(e)?;
                let r = pc_program::residency::StreamRing::new(a, &device, n, pc_program::residency::StreamRingCfg { frames: ps.stream_frames, frame_rows }, sev).map_err(|x| PyRuntimeError::new_err(x.to_string()))?;
                (Some(c), Some(r))
            } else {
                (None, None)
            };
            let kv = match ps.kv {
                Some((units, rows, pages)) => Some(pc_program::kv::PinnedKv::new(a, &device, lay, units, rows, pages, sev).map_err(|x| PyRuntimeError::new_err(x.to_string()))?),
                None => None,
            };
            let streams = StreamReg { rows: ps.streams.clone(), pinned: HashMap::new() };
            let side = {
                let mut sx = Executor::<W>::new(&device, a, &mplan).map_err(e)?;
                sx.ks.carve_regen(a, &lay).map_err(e)?;
                let sops = Ops::new(a, &device, &sx.compute, &ctx, ocfg).map_err(e)?;
                let n_ev = ps.programs.iter().map(|p| p.ops.iter().map(|o| o.members().len()).sum::<usize>()).max().unwrap_or(0) + 16;
                let ev = (0..n_ev).map(|_| Event::new(&device)).collect::<Result<Vec<_>, _>>().map_err(|x| e(x.into()))?;
                let fork = Event::new(&device).map_err(|x| e(x.into()))?;
                let join = Event::new(&device).map_err(|x| e(x.into()))?;
                sx.compute.sync().map_err(|x| e(x.into()))?;
                Some(SideLane { ex: sx, ops: sops, ev, fork, join })
            };
            Some(Planned { bounds: vec![], pcmm: pc, ccmm: cc, shadows: vec![], replayer, slab, regs, work, dense, plain, sparse, classic, spru, spru_secret, spec: ps, group_bytes, stats: PlannedStats::default(), side, coeff, ring, streams, stream_tag: vec![], tag_layouts: vec![], stream_order: vec![], kv, stream_events })
        }
    };
    ex.compute.sync().map_err(|x| e(x.into()))?;
    let sh = Shadow::new(sc, Some(tables));
    let source = if layout == KskLayout::Dense { KskSource::Cached } else { KskSource::Regen };
    let state = State {
        planned,
        ops,
        pts,
        ex,
        keys,
        plans,
        groups,
        enc,
        secret: secret_side,
        sk,
        prng,
        pt_events,
        named: HashMap::new(),
        cached: HashMap::new(),
        enc_hits: 0,
        enc_misses: 0,
    };
    Ok(Dev {
        st: RefCell::new(state),
        pool,
        ctx,
        sh,
        source,
        radix_bits,
        complex,
        geometry,
        debug_refresh,
        frozen: std::cell::Cell::new(false),
        mem_report,
        keygen_s,
        setup_s: t0.elapsed().as_secs_f64(),
        arena,
        device,
    })
}

/// The constructor arguments of a session (`Session(...)`, `Session.from_bundle`, `Session.key_manifest`).
pub(crate) struct Args {
    cfg: pc_config::Config,
    ks: PyKeySource,
    sizes: Sizes,
    geometry: Geometry,
    plan: Option<PlanSpec>,
    pp: Option<crate::pcmm::PcmmPlan>,
    cp: Option<crate::matmul::CcmmPlan>,
}

#[allow(clippy::too_many_arguments)]
fn args(config: &PyConfig, keys: &PyKeySource, size: Option<HashMap<String, usize>>, pool: usize, pt_slots: usize, encode_threads: usize, programs: Option<Vec<PyRef<'_, crate::program::PyProgram>>>, regs: usize, pt_rows: usize, dyn_rot: Vec<i64>, stc_cache: usize, pcmm: Option<Vec<crate::pcmm::LinSpec>>, pcmm_cf: u32, check_meta: bool, ccmm: bool, streams: Option<BTreeMap<String, usize>>, stream_frames: usize, kv: Option<(usize, usize, usize)>, reg_levels: Vec<u32>, pt_levels: Vec<u32>) -> PyResult<Args> {
    let s = size.unwrap_or_default();
    let g = |k: &str, d: usize| s.get(k).copied().unwrap_or(d);
    let geometry = Geometry {
        hid: g("hidDim", 1024),
        dim: g("dim", 768),
        heads: g("numHeads", 16),
        heads_real: g("numHeadsReal", 12),
        exp: g("expDim", 4096),
        exp_real: g("expanded", 3072),
        seq_len: g("seqLen", 1024),
    };
    let sizes = Sizes { pool, pt_slots, encode_threads };
    let cfg = config.0.clone();
    let ks = keys.clone();
    let plan = match programs {
        None => None,
        Some(ps) => {
            let mut shadows = vec![];
            for p in &ps {
                shadows.push(p.sh_cfg.clone().ok_or_else(|| PyValueError::new_err(format!("{}: no runtime Shadow (compile it with Program.compile)", p.prog.block)))?);
            }
            Some(PlanSpec {
                programs: ps.iter().map(|p| p.prog.clone()).collect(),
                shadows,
                regs,
                reg_levels,
                pt_rows,
                pt_levels,
                dyn_rot,
                stc_cache,
                use_graphs: cfg.exec.use_graphs.unwrap_or(true),
                check_meta,
                streams: streams.unwrap_or_default(),
                stream_frames,
                kv,
            })
        }
    };
    let pp = match pcmm {
        Some(specs) if !specs.is_empty() => {
            if plan.is_none() {
                return Err(PyValueError::new_err("Session(pcmm=...) serves planned programs: pass programs=[...]"));
            }
            // PF_PCMM_BATCH / _DROPS (the tip's defaults 4 / on): finish groups, sized to the programs' finishes
            let batch = knob(cfg.model.prefill.pcmm_batch, 4, "model.prefill.pcmm_batch")? as usize;
            let mut pp = crate::pcmm::PcmmPlan::new(&specs, pcmm_cf, batch, cfg.model.prefill.pcmm_batch_drops.unwrap_or(true))?;
            if let Some(ps) = &plan {
                pp.fit(&ps.programs)?;
            }
            Some(pp)
        }
        _ => None,
    };
    if ccmm && plan.is_none() {
        return Err(PyValueError::new_err("Session(ccmm=True) serves planned programs: pass programs=[...]"));
    }
    let cp = ccmm.then(|| crate::matmul::CcmmPlan { cfg: pc_gpu::matmul::ccmm::CcmmCfg::default() });
    Ok(Args { cfg, ks, sizes, geometry, plan, pp, cp })
}

/// The device session. `Session(config, keys)` sizes and allocates the arena, generates the keys and carves every
/// table; `complete_setup()` freezes the arena (no device allocation after it); `fhe` is the eager op surface.
#[pyclass(name = "Session", module = "perseus._pc", unsendable)]
pub struct PySession {
    pub(crate) dev: Rc<Dev>,
    fhe: Option<Py<crate::fhe::PyFhe>>,
}

#[pymethods]
impl PySession {
    /// `size`: the model geometry `inf.size` reports (`hidDim`, `dim`, `numHeads`, `numHeadsReal`, `expDim`,
    /// `expanded`, `seqLen`; GPT-2 small by default). `pool`: eager ciphertext slots. `pt_slots`: device plaintext
    /// slots (least recently used reused beyond). `debug_refresh`: TEST ONLY — the bootstrap family refreshes by
    /// decrypting and re-encrypting at the route's landing (`Fhe.refresh_via_decrypt`) instead of raising.
    ///
    /// Planned sessions (`programs`: compiled `Program`s, bound later with `Program.bind` before `complete_setup`):
    /// the session also carves the refresh routes the programs' sites use (dense / StC-first with the sites' raise
    /// drops and real-payload chain, SparseB per slot count and drop, SPRU), their work buffers, one ciphertext slab
    /// of the largest program's slot count, `regs` carry registers (block exits, KV states: `run`'s inputs and
    /// outputs) and the programs' key groups; plaintexts stay resident (`pt_slots` slots of `pt_rows` rows, 0 =
    /// every prime). `dyn_rot`: the rotation steps dynamic rotate steps may take (`Program.set_rotation`).
    /// `stc_cache`: StC-first last-stage cache entries (every key a captured block uses must stay cached).
    #[new]
    ///
    /// Streamed weights: `streams` {linear: rows} declares the linears the programs' `CoeffSum`s read
    /// (float32 coefficient rows, uploaded with `stream_weights`); `stream_frames` device weight frames rotate over
    /// the blocks in `stream_order`. `kv` (units, rows, device pages): the K/V states in pinned host pages
    /// (`Program.run(kv_in=, kv_out=)`), `pages` device pages (unit u in page u % pages).
    #[pyo3(signature = (config, keys, size=None, pool=256, pt_slots=256, encode_threads=4, debug_refresh=false, programs=None, regs=0, pt_rows=0, dyn_rot=Vec::new(), stc_cache=8, pcmm=None, pcmm_cf=6, check_meta=false, ccmm=false, streams=None, stream_frames=2, kv=None, reg_levels=Vec::new(), pt_levels=Vec::new()))]
    #[allow(clippy::too_many_arguments)]
    fn new(py: Python<'_>, config: &PyConfig, keys: &PyKeySource, size: Option<HashMap<String, usize>>, pool: usize, pt_slots: usize, encode_threads: usize, debug_refresh: bool, programs: Option<Vec<PyRef<'_, crate::program::PyProgram>>>, regs: usize, pt_rows: usize, dyn_rot: Vec<i64>, stc_cache: usize, pcmm: Option<Vec<crate::pcmm::LinSpec>>, pcmm_cf: u32, check_meta: bool, ccmm: bool, streams: Option<BTreeMap<String, usize>>, stream_frames: usize, kv: Option<(usize, usize, usize)>, reg_levels: Vec<u32>, pt_levels: Vec<u32>) -> PyResult<Self> {
        let x = args(config, keys, size, pool, pt_slots, encode_threads, programs, regs, pt_rows, dyn_rot, stc_cache, pcmm, pcmm_cf, check_meta, ccmm, streams, stream_frames, kv, reg_levels, pt_levels)?;
        let dev = nogil(py, || setup(&x.cfg, &x.ks, &Origin::Generate, x.sizes, x.geometry, debug_refresh, x.plan, x.pp, x.cp))?;
        Ok(PySession { dev: Rc::new(dev), fhe: None })
    }

    /// A session on a client's key bundle (`bundle`: the file the client wrote, docs/WIRE.md): the same arguments
    /// as `Session(...)` (`keys` only for its rotation steps, groups, layout and band; its seed / secret unused), the
    /// keys imported instead of generated. The session holds NO secret: encryptions use the bundle's public key,
    /// decryption raises. The bundle must answer this session's `key_manifest` (same keys, chain, plan contract).
    /// `debug_secret` (tests only): the client's secret, making a debug session that can also decrypt.
    #[staticmethod]
    #[pyo3(signature = (config, keys, bundle, size=None, pool=256, pt_slots=256, encode_threads=4, debug_refresh=false, programs=None, regs=0, pt_rows=0, dyn_rot=Vec::new(), stc_cache=8, pcmm=None, pcmm_cf=6, check_meta=false, ccmm=false, streams=None, stream_frames=2, kv=None, reg_levels=Vec::new(), pt_levels=Vec::new(), debug_secret=None))]
    #[allow(clippy::too_many_arguments)]
    fn from_bundle(py: Python<'_>, config: &PyConfig, keys: &PyKeySource, bundle: std::path::PathBuf, size: Option<HashMap<String, usize>>, pool: usize, pt_slots: usize, encode_threads: usize, debug_refresh: bool, programs: Option<Vec<PyRef<'_, crate::program::PyProgram>>>, regs: usize, pt_rows: usize, dyn_rot: Vec<i64>, stc_cache: usize, pcmm: Option<Vec<crate::pcmm::LinSpec>>, pcmm_cf: u32, check_meta: bool, ccmm: bool, streams: Option<BTreeMap<String, usize>>, stream_frames: usize, kv: Option<(usize, usize, usize)>, reg_levels: Vec<u32>, pt_levels: Vec<u32>, debug_secret: Option<Vec<u8>>) -> PyResult<Self> {
        let x = args(config, keys, size, pool, pt_slots, encode_threads, programs, regs, pt_rows, dyn_rot, stc_cache, pcmm, pcmm_cf, check_meta, ccmm, streams, stream_frames, kv, reg_levels, pt_levels)?;
        let origin = match debug_secret {
            Some(sec) => Origin::DebugBundle(bundle, sec),
            None => Origin::Bundle(bundle),
        };
        let dev = nogil(py, || setup(&x.cfg, &x.ks, &origin, x.sizes, x.geometry, debug_refresh, x.plan, x.pp, x.cp))?;
        if matches!(origin, Origin::Bundle(_)) {
            let st = dev.st.borrow();
            if st.secret.is_some() || st.sk.is_some() {
                return Err(PyRuntimeError::new_err("Session.from_bundle: the session ended up holding a secret"));
            }
        }
        Ok(PySession { dev: Rc::new(dev), fhe: None })
    }

    /// The key bundle a `Session.from_bundle` with these arguments needs (JSON, `pc_client::KeySpec`): every
    /// switching key (role + rows), the SPRU material, the PCMM shapes, CCMM, a public key; the chain's params hash
    /// and the plan contract. No device.
    #[staticmethod]
    #[pyo3(signature = (config, keys, size=None, pool=256, pt_slots=256, encode_threads=4, programs=None, regs=0, pt_rows=0, dyn_rot=Vec::new(), stc_cache=8, pcmm=None, pcmm_cf=6, check_meta=false, ccmm=false, streams=None, stream_frames=2, kv=None, reg_levels=Vec::new(), pt_levels=Vec::new()))]
    #[allow(clippy::too_many_arguments)]
    fn key_manifest(config: &PyConfig, keys: &PyKeySource, size: Option<HashMap<String, usize>>, pool: usize, pt_slots: usize, encode_threads: usize, programs: Option<Vec<PyRef<'_, crate::program::PyProgram>>>, regs: usize, pt_rows: usize, dyn_rot: Vec<i64>, stc_cache: usize, pcmm: Option<Vec<crate::pcmm::LinSpec>>, pcmm_cf: u32, check_meta: bool, ccmm: bool, streams: Option<BTreeMap<String, usize>>, stream_frames: usize, kv: Option<(usize, usize, usize)>, reg_levels: Vec<u32>, pt_levels: Vec<u32>) -> PyResult<String> {
        let x = args(config, keys, size, pool, pt_slots, encode_threads, programs, regs, pt_rows, dyn_rot, stc_cache, pcmm, pcmm_cf, check_meta, ccmm, streams, stream_frames, kv, reg_levels, pt_levels)?;
        let chain = Chain::generate(&ChainSpec::n32()).map_err(|e| gerr(e.into()))?;
        let nd = key_needs(&x.cfg, &x.ks, &chain, &x.plan, &x.pp, &x.cp)?;
        manifest(&x.cfg, &chain, &nd, &x.pp, x.cp.is_some())?.to_json().map_err(cerr)
    }

    /// CCMM service: `nf_ref` / `nf_split` from `token` (a fresh encryption at the bootstrap output level:
    /// ccmm.py `prepare`). Returns (nf_ref, nf_split).
    fn ccmm_prepare(&self, py: Python<'_>, token: PyRef<'_, crate::fhe::PyCt>) -> PyResult<(f64, f64)> {
        let d = &*self.dev;
        let mut st = d.st.borrow_mut();
        let State { planned, ex, ops, plans, .. } = &mut *st;
        let pl = planned.as_mut().ok_or_else(|| PyRuntimeError::new_err("not a planned session"))?;
        let Planned { ccmm, dense, work, shadows, .. } = pl;
        let rt = ccmm.as_mut().ok_or_else(|| PyRuntimeError::new_err("Session has no CCMM service (ccmm=True)"))?;
        let route = dense.as_ref().ok_or_else(|| PyRuntimeError::new_err("no StC-first route"))?;
        let sh = shadows.first().unwrap_or(&d.sh);
        let src = token.ct_ref();
        nogil(py, || -> pc_gpu::Result<(f64, f64)> {
            let mut b = std::mem::take(&mut rt.b);
            let r = (|| {
                pc_program::replay::copy_ct(&d.ctx, &ex.compute, &mut b[0], src).map_err(|x| pc_gpu::Error::Shape { what: "ccmm_prepare", msg: x.to_string() })?;
                let mut cx = pc_gpu::bts::BtsCx { ctx: &d.ctx, ex: &mut *ex, sh, ks: &*plans, ops: &mut *ops, work: &mut *work };
                rt.prepare(&mut cx, route, &mut b[0])?;
                ex.compute.sync().map_err(pc_gpu::Error::from)?;
                Ok((rt.nf_ref, rt.nf_split))
            })();
            rt.b = b;
            r
        })
        .map_err(gerr)
    }

    /// CCMM service: one CCMM of fresh ciphertexts (16 + 16, any SinC slot values) with explicit operand factors
    /// (the calibration probe of ccmm.py `_calibrate`; the outputs are new session ciphertexts, 16 or 8 merged).
    #[pyo3(signature = (a, b, b_transposed, fa, fb, shift=0, merge=false, kappa=1.0))]
    #[allow(clippy::too_many_arguments)]
    fn ccmm_probe(&self, py: Python<'_>, a: Vec<PyRef<'_, crate::fhe::PyCt>>, b: Vec<PyRef<'_, crate::fhe::PyCt>>, b_transposed: bool, fa: f64, fb: f64, shift: u32, merge: bool, kappa: f64) -> PyResult<Vec<crate::fhe::PyCt>> {
        use pc_core::ccmm_keys::NBIG;
        if a.len() != NBIG || b.len() != NBIG {
            return Err(PyValueError::new_err(format!("ccmm_probe: {} / {} operands, a matrix is {NBIG}", a.len(), b.len())));
        }
        let d = &self.dev;
        let n_out = {
            let mut st = d.st.borrow_mut();
            let State { planned, ex, ops, plans, .. } = &mut *st;
            let pl = planned.as_mut().ok_or_else(|| PyRuntimeError::new_err("not a planned session"))?;
            let Planned { ccmm, dense, work, shadows, .. } = pl;
            let rt = ccmm.as_mut().ok_or_else(|| PyRuntimeError::new_err("Session has no CCMM service (ccmm=True)"))?;
            let route = dense.as_ref().ok_or_else(|| PyRuntimeError::new_err("no StC-first route"))?;
            let sh = shadows.first().unwrap_or(&d.sh);
            let srcs: Vec<&Ct<'static, W>> = a.iter().chain(b.iter()).map(|x| x.ct_ref()).collect();
            nogil(py, || -> pc_gpu::Result<usize> {
                let cp = |e: pc_program::Error| pc_gpu::Error::Shape { what: "ccmm_probe", msg: e.to_string() };
                for (i, c) in srcs.iter().enumerate() {
                    let dst = if i < NBIG { &mut rt.a[i] } else { &mut rt.b[i - NBIG] };
                    pc_program::replay::copy_ct(&d.ctx, &ex.compute, dst, c).map_err(cp)?;
                }
                let mut cx = pc_gpu::bts::BtsCx { ctx: &d.ctx, ex: &mut *ex, sh, ks: &*plans, ops: &mut *ops, work: &mut *work };
                rt.run(&mut cx, route, b_transposed, fa, fb, shift, merge, kappa, 0)
            })
            .map_err(gerr)?
        };
        let mut out = vec![];
        for m in 0..n_out {
            let meta = {
                let st = d.st.borrow();
                st.planned.as_ref().and_then(|p| p.ccmm.as_ref()).map(|c| c.a[m].meta).ok_or_else(|| PyRuntimeError::new_err("ccmm"))?
            };
            let mut o = crate::fhe::PyCt::acquire(d, meta)?;
            {
                let mut st = d.st.borrow_mut();
                let State { planned, ex, .. } = &mut *st;
                let c = planned.as_ref().and_then(|p| p.ccmm.as_ref()).ok_or_else(|| PyRuntimeError::new_err("ccmm"))?;
                pc_program::replay::copy_ct(&d.ctx, &ex.compute, o.ct_mut_ref()?, &c.a[m]).map_err(crate::program::perr)?;
            }
            out.push(o);
        }
        Ok(out)
    }

    /// CCMM service: the calibrated output value per unit operand-factor product (ccmm.py `c_unit`) and the
    /// operating-point factor `kappa` of the programs' finishes.
    #[pyo3(signature = (c_unit, kappa_qk=1.0, kappa_pv=1.0))]
    fn ccmm_set(&self, c_unit: f64, kappa_qk: f64, kappa_pv: f64) -> PyResult<()> {
        let kappa = (kappa_qk, kappa_pv);
        if !(c_unit.is_finite() && c_unit > 0.0 && [kappa_qk, kappa_pv].iter().all(|k| k.is_finite() && *k > 0.0)) {
            return Err(PyValueError::new_err(format!("ccmm_set: c_unit {c_unit}, kappa {kappa:?}")));
        }
        let mut st = self.dev.st.borrow_mut();
        let rt = st.planned.as_mut().and_then(|p| p.ccmm.as_mut()).ok_or_else(|| PyRuntimeError::new_err("Session has no CCMM service (ccmm=True)"))?;
        rt.c_unit = Some(c_unit);
        rt.kappa = kappa;
        Ok(())
    }

    /// CCMM service counters: {calls, keygen_s, c_unit, nf_ref, nf_split}.
    fn ccmm_stats(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        let st = self.dev.st.borrow();
        let rt = st.planned.as_ref().and_then(|p| p.ccmm.as_ref()).ok_or_else(|| PyRuntimeError::new_err("Session has no CCMM service (ccmm=True)"))?;
        let dd = pyo3::types::PyDict::new(py);
        dd.set_item("calls", rt.calls)?;
        dd.set_item("keygen_s", rt.keygen_s)?;
        dd.set_item("c_unit", rt.c_unit)?;
        dd.set_item("nf_ref", rt.nf_ref)?;
        dd.set_item("nf_split", rt.nf_split)?;
        dd.set_item("kappa", rt.kappa)?;
        Ok(dd.unbind())
    }

    /// The plaintext store (`PtStore`): stage / evict the programs' plaintexts.
    #[getter]
    fn pt_store(&self) -> crate::store::PyPtStore {
        crate::store::PyPtStore { dev: self.dev.clone() }
    }

    /// Carry registers of a planned session.
    #[getter]
    fn n_regs(&self) -> usize {
        self.dev.st.borrow().planned.as_ref().map(|p| p.regs.len()).unwrap_or(0)
    }



    /// Make the blocks' plaintexts resident before they run (the store is resident in a planned session: joins
    /// any staged upload into the compute lane). Returns the keys joined.
    fn prefetch(&self, programs: Vec<PyRef<'_, crate::program::PyProgram>>) -> PyResult<usize> {
        let mut st = self.dev.st.borrow_mut();
        let State { planned, pts, ex, .. } = &mut *st;
        let pl = planned.as_ref().ok_or_else(|| PyRuntimeError::new_err("not a planned session"))?;
        let mut n = 0;
        for p in programs {
            let i = p.bound_index()?;
            let keys = pl.bounds[i].pt_keys();
            n += keys.len();
            pts.adopt(&self.dev.ctx, ex, &keys).map_err(gerr)?;
        }
        Ok(n)
    }

    /// Upload the int8 digit planes `[nw][rows_out][rows_in]` (bytes) of PCMM linear `name` declared at setup
    /// (`pcmm=`); resident from then on. Synchronous.
    fn pcmm_upload(&self, py: Python<'_>, name: &str, planes: &[u8]) -> PyResult<()> {
        let mut st = self.dev.st.borrow_mut();
        let State { planned, ex, .. } = &mut *st;
        let pc = planned.as_mut().and_then(|p| p.pcmm.as_mut()).ok_or_else(|| PyRuntimeError::new_err("Session has no PCMM service (pcmm=)"))?;
        nogil(py, || pc.upload(&ex.compute, name, planes))
    }

    /// PCMM service counters: {calls, batch (group size, per-drop batching), groups, outputs, keygen_s, linears,
    /// missing (declared, never uploaded)}.
    fn pcmm_stats(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        let st = self.dev.st.borrow();
        let pc = st.planned.as_ref().and_then(|p| p.pcmm.as_ref()).ok_or_else(|| PyRuntimeError::new_err("Session has no PCMM service (pcmm=)"))?;
        let d = pyo3::types::PyDict::new(py);
        d.set_item("calls", pc.calls)?;
        d.set_item("batch", (pc.batch, pc.batch_drops))?;
        d.set_item("groups", pc.groups)?;
        d.set_item("outputs", pc.outputs)?;
        d.set_item("keygen_s", pc.keygen_s)?;
        d.set_item("linears", pc.lins.len())?;
        d.set_item("missing", pc.missing())?;
        Ok(d.unbind())
    }

    /// Upload streamed linear `name`'s float32 coefficient rows (`[rows][N]`, C-contiguous, the rows declared in
    /// `Session(streams=...)`) into pinned host memory: the weight frames are filled from there.
    fn stream_weights(&self, py: Python<'_>, name: &str, rows: Bound<'_, PyAny>) -> PyResult<()> {
        let buf = pyo3::buffer::PyBuffer::<f32>::get(&rows)?;
        let mut st = self.dev.st.borrow_mut();
        let pl = st.planned.as_mut().ok_or_else(|| PyRuntimeError::new_err("not a planned session"))?;
        let want = pl.streams.rows.get(name).copied().ok_or_else(|| PyValueError::new_err(format!("stream_weights: {name} not declared in Session(streams=...)")))?;
        let n = self.dev.n();
        if buf.item_count() != want * n || !buf.is_c_contiguous() {
            return Err(PyValueError::new_err(format!("stream_weights {name}: {} floats (C-contiguous {}), declared {want} rows of {n}", buf.item_count(), buf.is_c_contiguous())));
        }
        let mut pin = pc_cuda::PinnedBuf::<f32>::new(&self.dev.device, want * n).map_err(|x| gerr(x.into()))?;
        buf.copy_to_slice(py, pin.as_mut_slice())?;
        pl.streams.pinned.insert(name.to_string(), pin);
        Ok(())
    }

    /// The run order of the streaming blocks (bound programs; a block of any program class stands for every class's
    /// block with the same linears): block i's rows are staged while block i - 1 runs, cyclically; position i uses
    /// weight frame i % stream_frames.
    fn stream_order(&self, programs: Vec<PyRef<'_, crate::program::PyProgram>>) -> PyResult<()> {
        let idx = programs.iter().map(|p| p.bound_index()).collect::<PyResult<Vec<_>>>()?;
        let mut st = self.dev.st.borrow_mut();
        let pl = st.planned.as_mut().ok_or_else(|| PyRuntimeError::new_err("not a planned session"))?;
        let order = idx.iter().map(|&i| pl.stream_tag.get(i).copied().flatten().ok_or_else(|| PyValueError::new_err(format!("stream_order: bound block {i} streams no weights")))).collect::<PyResult<Vec<_>>>()?;
        let f = pl.spec.stream_frames;
        if order.len() > 1 && (order.len() - 1) % f == 0 {
            return Err(crate::ConfigError::new_err(format!("stream_order of {} blocks with {f} frames: the last and the first block share a frame (pick stream_frames dividing {})", order.len(), order.len())));
        }
        pl.stream_order = order;
        Ok(())
    }

    /// Streamed-weight counters: {stages, reuses, bytes (host -> device), cold, frames, frame_rows, pinned_bytes,
    /// overflows (clamped coefficients: nonzero = a wrong result)}.
    fn stream_stats(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        let st = self.dev.st.borrow();
        let pl = st.planned.as_ref().ok_or_else(|| PyRuntimeError::new_err("not a planned session"))?;
        let d = pyo3::types::PyDict::new(py);
        if let Some(r) = pl.ring.as_ref() {
            d.set_item("stages", r.stats.stages)?;
            d.set_item("reuses", r.stats.reuses)?;
            d.set_item("bytes", r.stats.bytes)?;
            d.set_item("cold", r.stats.cold)?;
            d.set_item("frames", r.cfg.frames)?;
            d.set_item("frame_rows", r.cfg.frame_rows)?;
        }
        d.set_item("pinned_bytes", pl.streams.pinned.values().map(|b| b.len() * 4).sum::<usize>())?;
        if let Some(c) = pl.coeff.as_ref() {
            d.set_item("overflows", nogil(py, || c.overflows(&st.ex.compute)).map_err(gerr)?)?;
        }
        Ok(d.unbind())
    }

    /// Write `ct` into pinned KV unit `u` (setup: a cache state the programs read, e.g. the BOS prefix): device page,
    /// then the copy-out (asynchronous on the lanes).
    fn kv_put(&self, u: usize, ct: PyRef<'_, crate::fhe::PyCt>) -> PyResult<()> {
        use pc_program::kv::KvStore;
        let mut st = self.dev.st.borrow_mut();
        let State { planned, ex, .. } = &mut *st;
        let pl = planned.as_mut().ok_or_else(|| PyRuntimeError::new_err("not a planned session"))?;
        let kv = pl.kv.as_mut().ok_or_else(|| PyRuntimeError::new_err("Session has no pinned KV (kv=)"))?;
        let r = (|| -> pc_program::Result<()> {
            kv.join(ex, u)?;
            let page = kv.page(u)?;
            pc_program::replay::copy_ct(&self.dev.ctx, &ex.compute, page, ct.ct_ref())?;
            kv.store(ex, u)?;
            kv.finalize(ex, u)
        })();
        r.map_err(crate::program::perr)
    }


    /// The host secret is dropped (after the keys made from it on the host, PCMM / CCMM, exist).
    fn drop_secret(&self) {
        self.dev.st.borrow_mut().sk = None;
    }

    /// Whether the session still holds the host secret.
    #[getter]
    fn has_secret(&self) -> bool {
        self.dev.st.borrow().sk.is_some()
    }

    /// Checks: (keys, max |e|) of the switch relation of every key of the set (or the first `limit`), each key
    /// downloaded and checked on the host against the host secret (`keys::hostref`). Needs the host secret.
    #[pyo3(signature = (limit=None))]
    fn debug_key_relation(&self, py: Python<'_>, limit: Option<usize>) -> PyResult<(usize, i64)> {
        use pc_gpu::keys::hostref::HostChain;
        let d = &*self.dev;
        let st = d.st.borrow();
        let sk = st.sk.as_ref().ok_or_else(|| PyRuntimeError::new_err("debug_key_relation: the session holds no host secret"))?;
        let chain = &d.ctx.chain;
        let n = chain.n;
        nogil(py, || -> pc_gpu::Result<(usize, i64)> {
            let hc = HostChain::new(chain);
            let ids: Vec<usize> = (0..chain.primes().len()).collect();
            let s = hc.eval_rows(&sk.s.iter().map(|&x| x as i64).collect::<Vec<_>>(), &ids);
            let sp = hc.eval_rows(&sk.s_sparse.iter().map(|&x| x as i64).collect::<Vec<_>>(), &ids);
            let (mut cnt, mut mx) = (0usize, 0i64);
            for k in st.keys.iter().take(limit.unwrap_or(usize::MAX)) {
                let (b, a) = k.download_dense(&st.ex.compute, chain)?;
                let (s_new, s_old, sq) = match k.role {
                    KeyRole::Relin => (s.clone(), &s, true),
                    KeyRole::Rot { .. } | KeyRole::Conj => {
                        let g = k.role.galois(n).ok_or_else(|| pc_gpu::Error::Shape { what: "debug_key_relation", msg: "no Galois element".into() })?;
                        let gi = pc_core::galois::galois_inverse(g, n)?;
                        (s.iter().map(|r| pc_core::galois::apply_eval(r, gi)).collect::<std::result::Result<Vec<_>, _>>()?, &s, false)
                    }
                    KeyRole::EncDenseToSparse => (sp.clone(), &s, false),
                    KeyRole::EncSparseToDense => (s.clone(), &sp, false),
                };
                let (e, _) = hc.relation_max_err(&k.shape, &b, &a, &s_new, s_old, sq);
                mx = mx.max(e);
                cnt += 1;
            }
            Ok((cnt, mx))
        })
        .map_err(gerr)
    }

    /// Planned-runtime counters: runs, host us issuing them, cudaGraphLaunch us, replayer captures / replays /
    /// eager runs, capture ms, StC-first cache evictions.
    fn planned_stats(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        let st = self.dev.st.borrow();
        let pl = st.planned.as_ref().ok_or_else(|| PyRuntimeError::new_err("not a planned session"))?;
        let d = pyo3::types::PyDict::new(py);
        d.set_item("runs", pl.stats.runs)?;
        d.set_item("host_us", pl.stats.host_us)?;
        d.set_item("launch_us", pl.stats.launch_us)?;
        d.set_item("captures", pl.replayer.stats.captures)?;
        d.set_item("replays", pl.replayer.stats.replays)?;
        d.set_item("eager", pl.replayer.stats.eager)?;
        d.set_item("capture_ms", pl.replayer.stats.capture_ms)?;
        d.set_item("invalidations", pl.replayer.stats.invalidations)?;
        d.set_item("stc_evictions", pl.locks().iter().map(|l| l.evictions()).sum::<u64>())?;
        let stc = pl.dense.as_ref().and_then(|r| r.last.as_ref());
        d.set_item("stc_misses", stc.map_or(0, |c| c.stats().misses))?;
        d.set_item("stc_slots", stc.map_or(0, |c| c.slots()))?;
        d.set_item("stc_keys", stc.map_or(vec![], |c| c.keys().into_iter().map(|k| k.0.to_vec()).collect::<Vec<_>>()))?;
        d.set_item("bound", pl.bounds.len())?;
        d.set_item("group_bytes", pl.group_bytes)?;
        d.set_item("slab", pl.slab.slots.len().max(pl.bounds.iter().map(|b| b.n_slots).max().unwrap_or(0)))?;
        Ok(d.unbind())
    }

    /// Freeze the arena (a later device allocation panics) and enable the op surface. Returns the session.
    fn complete_setup(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.dev.arena().freeze();
        slf.dev.frozen.set(true);
        slf
    }

    /// The eager op surface (`_pc.Fhe`; also the `inf` the impl layer takes).
    #[getter]
    fn fhe(&mut self, py: Python<'_>) -> PyResult<Py<crate::fhe::PyFhe>> {
        self.dev.check_frozen()?;
        if let Some(f) = &self.fhe {
            return Ok(f.clone_ref(py));
        }
        let f = Py::new(py, crate::fhe::PyFhe::new(self.dev.clone()))?;
        self.fhe = Some(f.clone_ref(py));
        Ok(f)
    }

    /// `inf` (the impl layer's name for the session surface) = `fhe`.
    #[getter]
    fn inf(&mut self, py: Python<'_>) -> PyResult<Py<crate::fhe::PyFhe>> {
        self.fhe(py)
    }

    /// MemPlan parts (MB) as sized at setup.
    #[getter]
    fn mem_report(&self) -> String {
        self.dev.mem_report.clone()
    }
    /// (arena bytes used, capacity).
    fn arena_usage(&self) -> (usize, usize) {
        (self.dev.arena().used_bytes(), self.dev.arena().capacity())
    }
    /// Seconds: (whole setup, device keygen).
    fn setup_seconds(&self) -> (f64, f64) {
        (self.dev.setup_s, self.dev.keygen_s)
    }
    /// Number of device keys.
    fn n_keys(&self) -> usize {
        self.dev.st.borrow().keys.len()
    }
    /// Free ciphertext pool slots.
    fn pool_free(&self) -> usize {
        self.dev.pool.free(1).unwrap_or(0)
    }
    /// Wait for the device (the session stays usable).
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let st = self.dev.st.borrow();
        nogil(py, || st.ex.compute.sync()).map_err(|x| gerr(x.into()))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKeySource>()?;
    m.add_class::<PySession>()?;
    Ok(())
}
