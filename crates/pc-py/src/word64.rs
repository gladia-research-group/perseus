//! `_pc.w64`: the eager `Session` / `Fhe` / `Ct` on 64-bit words (`Context<u64, Barrett64>`) over the n64 chain
//! (28 x 53-bit + 60-bit q0, dnum 7): the u64 block-0 eager gate (`python/tests/gpu/test_eager_gpt2_block_u64.py`).
//!
//! GENERATED, do not edit: modules `session` and `fhe` below are `session.rs` / `fhe.rs` verbatim (inner docs dropped)
//! with these substitutions only, each asserted to hit:
//!   `type W = u32` -> `u64`; `ChainSpec::n32()` -> `ChainSpec::n64()`; the Shadow config -> `shadow_config_n64`
//!   (below); `crate::session::` / `crate::fhe::` -> `super::`; the decryptors take 1 and 2 bottom rows instead of 2
//!   and 4 (a two-row centred CRT of 60 + 53 bits fits u128, four rows would not); `KeySource` is the u32 module's
//!   class; pyclass module `perseus._pc.w64`.
//! It exists because Session / Fhe are monomorphic in pc-py: once `Session` is generic over the word (and the
//! chain), this file goes.
use pc_config::Config;
use pc_shadow::route::Landing;
use pc_shadow::{BtsConfig, ChainShape, Quirks, ShadowConfig};
use pyo3::prelude::*;

fn cfg_err(knob: &str, msg: impl std::fmt::Display) -> PyErr {
    crate::ConfigError::new_err(format!("{knob}: {msg}"))
}

/// The runtime Shadow of the n64 chain (`shadow.rs::shadow_config` at mode "runtime", n32 only, is the model): 28
/// primes, one per level. Landings are the n32 recipe's in CKKS levels (n32 prime levels / 2: StC-first dense 14,
/// shipped order 17, 512 -> 13, 256 -> 12, 1 -> 8, entry 23, guard 24); the reactive ceiling is the config's
/// `bts.auto_bts_level` (the old n64 profile: 24); CF floor = q0 bits - scale bits = 7. Only the debug refresh reads
/// the landings until a u64 route set exists.
pub(crate) fn shadow_config_n64(c: &Config) -> PyResult<ShadowConfig> {
    match c.chain.chain.as_deref() {
        Some("n64") => {}
        other => return Err(cfg_err("chain.chain", format!("{other:?}: the w64 session runs the n64 chain (set chain.chain = \"n64\")"))),
    }
    if c.chain.composite_degree.unwrap_or(1) != 1 {
        return Err(cfg_err("chain.composite_degree", "the n64 chain has one prime per level (1)"));
    }
    let l = |level| Landing { level, deg: 1 };
    let limit = c.bts.auto_bts_level.unwrap_or(24);
    if !(1..=26).contains(&limit) {
        return Err(cfg_err("bts.auto_bts_level", format!("{limit} outside 1..=26 on the n64 chain")));
    }
    let bts = BtsConfig {
        bootstrap_level: 14,
        level_limit: limit as u32,
        max_input_level: 24.max(limit as u32),
        landings: [(0, l(14)), (512, l(13)), (256, l(12)), (1, l(8))].into_iter().collect(),
        stc_first_entry: Some(23),
        stc_first_fallback: Some(l(17)),
        dense_shipped: None,
        cf_window: (7, 20),
        raise_drops: vec![1, 2, 3],
        k_max: 8,
        input_room: None,
    };
    Ok(ShadowConfig { chain: ChainShape { n_q: 28, unit: 1, singles: 0 }, bts, quirks: Quirks::none() })
}

/// `_pc.w64` = `Session`, `Fhe`, `Ct` and the module functions at u64 (the `KeySource` class is shared).
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let w = PyModule::new(m.py(), "w64")?;
    session::register(&w)?;
    fhe::register(&w)?;
    w.add_class::<crate::session::PyKeySource>()?;
    m.add_submodule(&w)?;
    Ok(())
}

#[rustfmt::skip]
#[allow(unused_imports)]
pub(crate) mod session {
use crate::config::PyConfig;
use crate::LevelError;
use pc_core::keys::{Prng, SecretKey};
use pc_core::modmul::Barrett64;
use pc_core::params::{Chain, ChainSpec};
use pc_cuda::{Arena, Device, EventPool};
use pc_gpu::decrypt::Decryptor;
use pc_gpu::encrypt::Encryptor;
use pc_gpu::keygen::KeyGen;
use pc_gpu::keys::{DevSecret, KeySet, KeySetSpec, KskLayout, RotSpec};
use pc_gpu::ks::{KeyGroup, KsKeyRef, KsPlans, KsWork, KskSource};
use pc_gpu::ops::{Ops, OpsCfg};
use pc_gpu::pt_store::{PtStore, PtStoreCfg, PtStorePolicy};
use pc_gpu::rotate::RotKeys;
use pc_gpu::{Context, CtPool, Executor, MemCfg, MemPlan, PoolClass};
use pc_shadow::route::Landing;
use pc_shadow::Shadow;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;
use std::sync::Arc;

pub(crate) type W = u64;
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

pub use crate::session::PyKeySource;

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

/// Mutable device state (borrowed `&mut` per op).
pub(crate) struct State {
    pub ops: Ops<'static, W>,
    pub pts: PtStore<'static, 'static, W>,
    pub ex: Executor<'static, W>,
    pub keys: KeySet<'static, W>,
    pub plans: KsPlans<'static>,
    pub groups: Groups,
    pub enc: Encryptor<'static, W>,
    /// bottom-pair and 4-row decryptors (a value above q0 q1 / 2 at its scale decodes from 4 rows)
    pub dec2: Decryptor<'static, W>,
    pub dec4: Decryptor<'static, W>,
    pub secret: DevSecret<'static, W>,
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

fn setup(cfg: &pc_config::Config, ks: &PyKeySource, sizes: Sizes, geometry: Geometry, debug_refresh: bool) -> PyResult<Dev> {
    let t0 = std::time::Instant::now();
    let e = gerr;
    let sc = super::shadow_config_n64(cfg)?;
    let complex = cfg.chain.ckks_complex.unwrap_or(false);
    let radix_bits = knob(cfg.model.ladder_radix_bits, 3, "model.ladder_radix_bits")? as u32;
    let device = Device::open().map_err(|x| e(x.into()))?;
    let chain = Chain::generate(&ChainSpec::n64()).map_err(|x| e(x.into()))?;
    let n = chain.n;
    let k_max = 8;
    let mcfg = MemCfg { k_max, pool: vec![PoolClass { k: 1, count: sizes.pool }], events: 64, scratch_cts: 2 };
    let mut plan = MemPlan::build::<W, M>(&chain, mcfg).map_err(e)?;
    let lay = plan.lay;
    let layout = match ks.layout.as_str() {
        "packed" => KskLayout::packed_for::<W>(&chain).map_err(e)?,
        _ => KskLayout::Dense,
    };
    let half = (n / 2) as i64;
    let mut steps: BTreeSet<i64> = ks.rot_steps.iter().map(|s| s.rem_euclid(half)).collect();
    for g in &ks.groups {
        steps.extend(g.iter().map(|s| s.rem_euclid(half)));
    }
    steps.remove(&0);
    let spec = KeySetSpec { relin: true, conj: ks.conj, rots: steps.iter().map(|&step| RotSpec { step, q_top: ks.band }).collect(), encaps: false };
    let shapes: Vec<_> = spec.keys(&chain).map_err(e)?.into_iter().map(|(_, s)| s).collect();
    plan.reserve("ks.plans", KsPlans::bytes::<W>(&chain, k_max).map_err(e)?);
    plan.reserve("keys", spec.bytes::<W>(&chain, layout).map_err(e)?);
    plan.reserve("keygen", KeyGen::<W>::bytes(&lay, &shapes));
    let nrot = steps.len() + ks.conj as usize;
    let gb: usize = ks.groups.iter().map(|g| KeyGroup::bytes(g.len().max(1), true)).sum::<usize>() + KeyGroup::bytes(nrot.max(1), true) + 2 * KeyGroup::bytes(1, true);
    plan.reserve("ks.groups", gb);
    plan.reserve("ks.regen_a", KsWork::<W>::regen_bytes(&lay, chain.dnum()));
    let ocfg = OpsCfg::default();
    Ops::<W>::reserve(&mut plan, &ocfg);
    let pcfg = PtStoreCfg {
        slots: sizes.pt_slots,
        ring: 4,
        encode_threads: sizes.encode_threads,
        policy: PtStorePolicy::StageAhead,
        eager: pc_gpu::encode_dev::EncoderKind::DeviceFft,
        rows: 0,
        classes: Vec::new(),
    };
    PtStore::<W>::reserve(&mut plan, &pcfg);
    plan.reserve("encrypt", Encryptor::<W>::bytes(&lay, false));
    plan.reserve("decrypt", Decryptor::<W>::bytes(&lay, 1, 1) + Decryptor::<W>::bytes(&lay, 2, 1));
    plan.reserve("keys.secret", DevSecret::<W>::bytes(&lay));
    let mem_report = plan.report();
    let arena = Box::new(plan.allocate(&device).map_err(e)?);
    // SAFETY: the box is owned by the `Dev` built below together with every borrower, and dropped after them
    let a: &'static Arena = unsafe { &*(arena.as_ref() as *const Arena) };
    let mut ex = Executor::<W>::new(&device, a, &plan).map_err(e)?;
    ex.ks.carve_regen(a, &lay).map_err(e)?;
    let ctx = Context::<W, M>::new(a, &ex.copy, chain.clone(), &plan).map_err(e)?;
    let pool = CtPool::<W>::new(a, &plan).map_err(e)?;
    let plans = KsPlans::new(a, &ex.compute, &ctx, k_max).map_err(e)?;
    let mut keys = KeySet::carve(a, &chain, &spec, layout).map_err(e)?;
    let mut prng = match ks.seed {
        Some(s) => Prng::from_seed([0x5e55_1013, s as u32, (s >> 32) as u32, 4, 5, 6, 7, 8], 0),
        None => Prng::from_os().map_err(|x| e(x.into()))?,
    };
    let sk = SecretKey::sample(&mut prng, n).map_err(|x| e(x.into()))?;
    let mut secret = DevSecret::carve(a, lay).map_err(e)?;
    secret.load(&ctx, &ex.compute, &sk).map_err(e)?;
    let tk = std::time::Instant::now();
    let mut kg = KeyGen::new(a, &ex.compute, &chain, &shapes).map_err(e)?;
    kg.generate_set(&ctx, &mut ex, &secret, &mut keys, &mut prng).map_err(e)?;
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
    let enc = Encryptor::new(a, &ex.compute, lay, false, &mut prng).map_err(e)?;
    let mut dec2 = Decryptor::new(a, &ex.compute, lay, 1, 1).map_err(e)?;
    let mut dec4 = Decryptor::new(a, &ex.compute, lay, 2, 1).map_err(e)?;
    dec2.load_secret(&ex.compute, &secret).map_err(e)?;
    dec4.load_secret(&ex.compute, &secret).map_err(e)?;
    ex.compute.sync().map_err(|x| e(x.into()))?;
    let sh = Shadow::new(sc, Some(Arc::new(ctx.scale.clone())));
    let source = if layout == KskLayout::Dense { KskSource::Cached } else { KskSource::Regen };
    let state = State {
        ops,
        pts,
        ex,
        keys,
        plans,
        groups,
        enc,
        dec2,
        dec4,
        secret,
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

/// The device session. `Session(config, keys)` sizes and allocates the arena, generates the keys and carves every
/// table; `complete_setup()` freezes the arena (no device allocation after it); `fhe` is the eager op surface.
#[pyclass(name = "Session", module = "perseus._pc.w64", unsendable)]
pub struct PySession {
    pub(crate) dev: Rc<Dev>,
    fhe: Option<Py<super::fhe::PyFhe>>,
}

#[pymethods]
impl PySession {
    /// `size`: the model geometry `inf.size` reports (`hidDim`, `dim`, `numHeads`, `numHeadsReal`, `expDim`,
    /// `expanded`, `seqLen`; GPT-2 small by default). `pool`: eager ciphertext slots. `pt_slots`: device plaintext
    /// slots (least recently used reused beyond). `debug_refresh`: TEST ONLY — the bootstrap family refreshes by
    /// decrypting and re-encrypting at the route's landing (`Fhe.refresh_via_decrypt`) instead of raising.
    #[new]
    #[pyo3(signature = (config, keys, size=None, pool=256, pt_slots=256, encode_threads=4, debug_refresh=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(py: Python<'_>, config: &PyConfig, keys: &PyKeySource, size: Option<HashMap<String, usize>>, pool: usize, pt_slots: usize, encode_threads: usize, debug_refresh: bool) -> PyResult<Self> {
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
        let dev = nogil(py, || setup(&cfg, &ks, sizes, geometry, debug_refresh))?;
        Ok(PySession { dev: Rc::new(dev), fhe: None })
    }

    /// Freeze the arena (a later device allocation panics) and enable the op surface. Returns the session.
    fn complete_setup(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.dev.arena().freeze();
        slf.dev.frozen.set(true);
        slf
    }

    /// The eager op surface (`_pc.Fhe`; also the `inf` the impl layer takes).
    #[getter]
    fn fhe(&mut self, py: Python<'_>) -> PyResult<Py<super::fhe::PyFhe>> {
        self.dev.check_frozen()?;
        if let Some(f) = &self.fhe {
            return Ok(f.clone_ref(py));
        }
        let f = Py::new(py, super::fhe::PyFhe::new(self.dev.clone()))?;
        self.fhe = Some(f.clone_ref(py));
        Ok(f)
    }

    /// `inf` (the impl layer's name for the session surface) = `fhe`.
    #[getter]
    fn inf(&mut self, py: Python<'_>) -> PyResult<Py<super::fhe::PyFhe>> {
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
    m.add_class::<PySession>()?;
    Ok(())
}
}

#[rustfmt::skip]
pub(crate) mod fhe {
use crate::meta::{PyCtMeta, PyRoute};
use super::session::{gerr, nogil, Dev, Named, State, W};
use crate::tag::PyTag;
use pc_core::encode::{Complex, Payload};
use pc_gpu::encode_dev::{EncoderKind, Payload2, PtForm};
use pc_gpu::pt_store::{PtKey, StageItem, StageValues};
use pc_gpu::rotate::{self, KsCfg, RotKeys};
use pc_gpu::{Ct, PoolCt};
use pc_shadow::{CtMeta, PtMeta, Route};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyKeyError, PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::BTreeSet;
use std::rc::Rc;

type Slot = PoolCt<'static, 'static, W>;

fn rt_err(msg: impl Into<String>) -> PyErr {
    PyRuntimeError::new_err(msg.into())
}

fn cuda(e: pc_cuda::Error) -> PyErr {
    gerr(e.into())
}

/// A device ciphertext of the session (opaque). Field order = drop order: the pool slot before the session.
#[pyclass(name = "Ct", module = "perseus._pc.w64", unsendable, dict)]
pub struct PyCt {
    slot: Slot,
    dev: Rc<Dev>,
}

impl PyCt {
    fn ct(&self) -> &Ct<'static, W> {
        &self.slot
    }
    fn ct_mut(&mut self) -> PyResult<&mut Ct<'static, W>> {
        self.slot.get_mut().ok_or_else(|| rt_err("Ct: shared slot (internal)"))
    }
    fn meta(&self) -> CtMeta {
        self.slot.meta
    }
}

#[pymethods]
impl PyCt {
    /// Levels consumed (primes dropped; the Shadow's convention).
    #[getter]
    fn level(&self) -> u32 {
        self.meta().level
    }
    /// 2 = a rescale is pending.
    #[getter]
    fn deg(&self) -> u8 {
        self.meta().deg
    }
    #[getter]
    fn noise_deg(&self) -> u8 {
        self.meta().deg
    }
    /// Scale (OpenFHE NoiseFactor).
    #[getter]
    fn nf(&self) -> f64 {
        self.meta().nf
    }
    #[getter]
    fn slots(&self) -> u32 {
        self.meta().slots
    }
    /// A copy of the pack tag (assign to change it).
    #[getter]
    fn tag(&self) -> PyTag {
        PyTag(self.meta().tag)
    }
    #[setter]
    fn set_tag(&mut self, t: PyTag) -> PyResult<()> {
        self.ct_mut()?.meta.tag = t.0;
        Ok(())
    }
    #[getter(meta)]
    fn py_meta(&self) -> PyCtMeta {
        PyCtMeta(self.meta())
    }
    fn __repr__(&self) -> String {
        let m = self.meta();
        format!("<Ct L{} d{} slots {}>", m.level, m.deg, m.slots)
    }
}

/// The geometry the impl layer reads as `inf.size`.
#[pyclass(name = "Size", module = "perseus._pc.w64", frozen)]
pub struct PySize {
    #[pyo3(get, name = "hidDim")]
    hid: usize,
    #[pyo3(get)]
    dim: usize,
    #[pyo3(get, name = "numHeads")]
    heads: usize,
    #[pyo3(get, name = "numHeadsReal")]
    heads_real: usize,
    #[pyo3(get, name = "expDim")]
    exp: usize,
    #[pyo3(get)]
    expanded: usize,
    #[pyo3(get, name = "seqLen")]
    seq_len: usize,
}

/// `with inf.step(label):` (graph scoping is the capture's; eager: nothing).
#[pyclass(module = "perseus._pc.w64")]
pub struct NullScope;

#[pymethods]
impl NullScope {
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    #[pyo3(signature = (*_a))]
    fn __exit__(&self, _a: &Bound<'_, PyTuple>) -> bool {
        false
    }
}

/// `with fhe.suppress_auto_bts():` no reactive refresh inside.
#[pyclass(module = "perseus._pc.w64", unsendable)]
pub struct SuppressScope {
    fhe: Py<PyFhe>,
    saved: bool,
}

#[pymethods]
impl SuppressScope {
    fn __enter__<'p>(mut slf: PyRefMut<'p, Self>, py: Python<'p>) -> PyRefMut<'p, Self> {
        let mut f = slf.fhe.borrow_mut(py);
        let saved = f.auto_bts_suppressed;
        f.auto_bts_suppressed = true;
        drop(f);
        slf.saved = saved;
        slf
    }
    #[pyo3(signature = (*_a))]
    fn __exit__(&self, py: Python<'_>, _a: &Bound<'_, PyTuple>) -> bool {
        self.fhe.borrow_mut(py).auto_bts_suppressed = self.saved;
        false
    }
}

/// What `stage_pts` returns: the keys `adopt_pts` joins.
#[pyclass(module = "perseus._pc.w64", unsendable)]
pub struct StageHandle {
    keys: Vec<PtKey>,
}

/// The eager op surface. See the module doc.
#[pyclass(name = "Fhe", module = "perseus._pc.w64", unsendable, dict)]
pub struct PyFhe {
    dev: Rc<Dev>,
    np: Option<Py<PyModule>>,
    #[pyo3(get, set)]
    block_prefix: String,
    #[pyo3(get, set)]
    capture_b: i64,
    #[pyo3(get, set)]
    capture_t: i64,
    #[pyo3(get, set)]
    magnitude_suppressed: bool,
    #[pyo3(get, set)]
    auto_bts_suppressed: bool,
    /// every refresh (deliberate, hint, reactive, fold)
    #[pyo3(get)]
    total_bootstraps: u64,
    #[pyo3(get)]
    n_reactive: u64,
}

impl PyFhe {
    pub(crate) fn new(dev: Rc<Dev>) -> Self {
        PyFhe { dev, np: None, block_prefix: String::new(), capture_b: 0, capture_t: 0, magnitude_suppressed: false, auto_bts_suppressed: false, total_bootstraps: 0, n_reactive: 0 }
    }

    fn numpy<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
        if let Some(m) = &self.np {
            return Ok(m.bind(py).clone());
        }
        let m = py.import("numpy")?;
        self.np = Some(m.clone().unbind());
        Ok(m)
    }

    fn new_ct(&self, meta: CtMeta) -> PyResult<PyCt> {
        self.dev.check_frozen()?;
        let slot = self.dev.pool().acquire(1, meta).map_err(gerr)?;
        Ok(PyCt { slot, dev: self.dev.clone() })
    }

    fn same_session(&self, c: &PyCt) -> PyResult<()> {
        if !Rc::ptr_eq(&self.dev, &c.dev) {
            return Err(PyValueError::new_err("ciphertext belongs to another Session"));
        }
        Ok(())
    }

    // ── host vectors ──────────────────────────────────────────────────────────────────────────────────────────────

    /// A slot vector (numpy / sequence; real or complex) as host complex values.
    fn values(&mut self, py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<Complex>> {
        let np = self.numpy(py)?;
        let arr = np.call_method1("ascontiguousarray", (obj,))?.call_method0("ravel")?;
        let kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;
        if kind == "c" {
            let a = np.call_method1("ascontiguousarray", (arr, "complex128"))?.call_method1("view", ("float64",))?;
            let v = PyBuffer::<f64>::get(&a)?.to_vec(py)?;
            Ok(v.chunks_exact(2).map(|p| Complex::new(p[0], p[1])).collect())
        } else {
            let a = np.call_method1("ascontiguousarray", (arr, "float64"))?;
            let v = PyBuffer::<f64>::get(&a)?.to_vec(py)?;
            Ok(v.into_iter().map(|x| Complex::new(x, 0.0)).collect())
        }
    }

    /// Rows of a 2-D array as float64 vectors.
    fn rows_f64(&mut self, py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
        let np = self.numpy(py)?;
        let arr = np.call_method1("ascontiguousarray", (obj, "float64"))?;
        let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
        if shape.len() != 2 {
            return Err(PyValueError::new_err(format!("expected a 2-D array, got shape {shape:?}")));
        }
        let v = PyBuffer::<f64>::get(&arr)?.to_vec(py)?;
        Ok(v.chunks(shape[1].max(1)).map(|r| r.to_vec()).collect())
    }

    fn np_f64(&mut self, py: Python<'_>, v: &[f64]) -> PyResult<Py<PyAny>> {
        let np = self.numpy(py)?;
        let a = np.call_method1("empty", (v.len(), "float64"))?;
        PyBuffer::<f64>::get(&a)?.copy_from_slice(py, v)?;
        Ok(a.unbind())
    }

    fn np_c128(&mut self, py: Python<'_>, v: &[Complex]) -> PyResult<Py<PyAny>> {
        let np = self.numpy(py)?;
        let a = np.call_method1("empty", (v.len(), "complex128"))?;
        let flat: Vec<f64> = v.iter().flat_map(|c| [c.re, c.im]).collect();
        PyBuffer::<f64>::get(&a.call_method1("view", ("float64",))?)?.copy_from_slice(py, &flat)?;
        Ok(a.unbind())
    }

    fn np_u64_2d(&mut self, py: Python<'_>, rows: &[Vec<W>]) -> PyResult<Py<PyAny>> {
        let np = self.numpy(py)?;
        let n = rows.first().map_or(0, |r| r.len());
        let a = np.call_method1("empty", ((rows.len(), n), "uint64"))?;
        let flat: Vec<u64> = rows.iter().flatten().map(|&x| x as u64).collect();
        PyBuffer::<u64>::get(&a)?.copy_from_slice(py, &flat)?;
        Ok(a.unbind())
    }

    // ── encrypt / decrypt ─────────────────────────────────────────────────────────────────────────────────────────

    /// Fresh secret-key encryption of `values` (zero-padded to `meta.slots`) at `meta` (level, degree, scale).
    fn encrypt_meta(&self, py: Python<'_>, values: &[Complex], meta: CtMeta) -> PyResult<PyCt> {
        let mut o = self.new_ct(meta)?;
        let d = &*self.dev;
        let slots = if meta.slots == 0 { d.slots() } else { meta.slots as usize };
        let payload = d.payload();
        let om = o.ct_mut()?;
        let mut st = d.st.borrow_mut();
        let State { enc, ex, secret, prng, pts, .. } = &mut *st;
        nogil(py, || -> pc_gpu::Result<()> {
            let top = d.ctx.top_of(meta.level)?;
            let rows = pts.enc.host().encode_slots(values, slots, top, payload, meta.nf)?;
            enc.encrypt_sk(&d.ctx, &ex.compute, secret, &[rows], meta, om, prng)
        })
        .map_err(gerr)?;
        Ok(o)
    }

    /// The meta of a fresh encryption at `level` (degree `deg`, the level's scaling factor, every slot).
    fn fresh_meta(&self, level: u32, deg: u8) -> PyResult<CtMeta> {
        let d = &*self.dev;
        let top = d.ctx.top_of(level).map_err(gerr)?;
        let sf = d.ctx.scale.sf(top as i32).map_err(|e| gerr(e.into()))?;
        Ok(CtMeta { level, deg, nf: sf.powi(deg as i32), slots: d.slots() as u32, tag: Default::default() })
    }

    /// Decoded slots of `x` (a pending rescale is paid on a copy first; 4 rows decode when the ciphertext has them).
    fn decrypt_values(&self, py: Python<'_>, x: &PyCt) -> PyResult<Vec<Complex>> {
        self.same_session(x)?;
        let d = &*self.dev;
        let tmp = if x.meta().deg == 2 {
            let mut t = self.new_ct(x.meta())?;
            {
                let tm = t.ct_mut()?;
                let mut st = d.st.borrow_mut();
                let State { ops, ex, .. } = &mut *st;
                let words = tm.buf.buf.len();
                tm.buf.buf.copy_from(&ex.compute, &x.ct().buf.buf, words).map_err(cuda)?;
                ops.realize(&d.ctx, ex, &d.sh, std::slice::from_mut(tm)).map_err(gerr)?;
            }
            Some(t)
        } else {
            None
        };
        let c: &Ct<'static, W> = tmp.as_ref().map_or(x.ct(), |t| t.ct());
        let mut st = d.st.borrow_mut();
        let State { ex, dec2, dec4, pts, .. } = &mut *st;
        let payload = d.payload();
        let n = d.n();
        nogil(py, || -> pc_gpu::Result<Vec<Complex>> {
            let slots = if c.meta.slots == 0 { n / 2 } else { c.meta.slots as usize };
            if !(c.meta.nf > 0.0) {
                return Err(pc_gpu::Error::Shape { what: "decrypt", msg: format!("ciphertext scale (meta.nf) = {}", c.meta.nf) });
            }
            let top = c.top(d.ctx.shape)?;
            let enc = pts.enc.host();
            if top + 1 >= 2 {
                let rows = dec4.decrypt_rows(&d.ctx, &ex.compute, c, pc_gpu::decrypt::Which::Dense)?;
                Ok(decode4(enc, &d.ctx.chain.q[..2], &rows[0], slots, payload, c.meta.nf, n))
            } else {
                let rows = dec2.decrypt_rows(&d.ctx, &ex.compute, c, pc_gpu::decrypt::Which::Dense)?;
                Ok(enc.decode_slots(&rows[0], slots, payload, c.meta.nf)?)
            }
        })
        .map_err(gerr)
    }

    // ── refreshes ─────────────────────────────────────────────────────────────────────────────────────────────────

    fn no_bts(&self, what: &str) -> PyErr {
        PyNotImplementedError::new_err(format!(
            "Fhe.{what}: the eager session has no bootstrap routes (planned runs refresh in Program.run). Tests can \
             build the Session with debug_refresh=True to refresh by decrypt + re-encrypt."
        ))
    }

    /// In place: decrypt `x` and encrypt `values_fn(decrypted)` fresh at `land` (TEST ONLY: `debug_refresh`).
    fn refresh_to(&mut self, py: Python<'_>, x: &mut PyCt, land: CtMeta, what: &str, f: impl FnOnce(&mut Vec<Complex>)) -> PyResult<()> {
        if !self.dev.debug_refresh {
            return Err(self.no_bts(what));
        }
        let mut v = self.decrypt_values(py, x)?;
        f(&mut v);
        let meta = CtMeta { slots: x.meta().slots, ..land };
        let mut fresh = self.encrypt_meta(py, &v, meta)?;
        // x takes the fresh ciphertext's slot; its old slot goes back to the pool with `fresh`
        std::mem::swap(&mut x.slot, &mut fresh.slot);
        drop(fresh);
        self.total_bootstraps += 1;
        Ok(())
    }

    fn refresh_route(&mut self, py: Python<'_>, x: &mut PyCt, route: Route, what: &str) -> PyResult<()> {
        let land = self.dev.sh.bootstrap(&x.meta(), &route).map_err(|e| gerr(e.into()))?.out;
        self.refresh_to(py, x, land, what, |_| {})
    }

    /// The reactive refresh after a product (`maybe_bootstrap`).
    fn reactive(&mut self, py: Python<'_>, x: &mut PyCt) -> PyResult<()> {
        if self.auto_bts_suppressed || x.meta().level < self.dev.level_limit() {
            return Ok(());
        }
        self.n_reactive += 1;
        self.refresh_route(py, x, Route::dense(), "maybe_bootstrap (reactive, after a product)")
    }

    // ── op helpers ────────────────────────────────────────────────────────────────────────────────────────────────

    fn copy_ct(&self, a: &PyCt) -> PyResult<PyCt> {
        let mut o = self.new_ct(a.meta())?;
        let om = o.ct_mut()?;
        let words = om.buf.buf.len();
        om.buf.buf.copy_from(&self.dev.st.borrow().ex.compute, &a.ct().buf.buf, words).map_err(cuda)?;
        om.meta = a.meta();
        Ok(o)
    }

    fn pt_meta(&self, a: &CtMeta) -> PtMeta {
        PtMeta { level: a.effective_level(self.dev.sh.unit()), deg: 1, nf: 0.0, tag: Default::default() }
    }

    /// Ensure `key` holds `values` (encode on the device when missing or `force`).
    fn ensure_pt(st: &mut State, d: &Dev, key: &PtKey, values: &[Complex], force: bool) -> PyResult<bool> {
        if !force && st.pts.has(key) {
            return Ok(false);
        }
        let State { pts, ex, ops, .. } = st;
        let what = Payload2::Slots { values, slots: d.slots(), payload: d.payload() };
        pts.encode_into(&d.ctx, ex, ops, &d.sh, key.clone(), what, 1).map_err(gerr)?;
        Ok(true)
    }

    /// `a * pt` / `a + pt` with the plaintext under `key` (resident after `ensure_pt`).
    fn pt_op(&self, a: &PyCt, key: &PtKey, add: bool) -> PyResult<PyCt> {
        let d = &*self.dev;
        let mut o = self.new_ct(a.meta())?;
        let om = o.ct_mut()?;
        let mut st = d.st.borrow_mut();
        let State { pts, ex, ops, .. } = &mut *st;
        let pt = pts.get(&d.ctx, ex, key).map_err(gerr)?;
        let r = if add {
            ops.add_pt(&d.ctx, ex, &d.sh, std::slice::from_mut(om), &[a.ct()], &[pt])
        } else {
            ops.mult_pt(&d.ctx, ex, &d.sh, std::slice::from_mut(om), &[a.ct()], &[pt])
        };
        r.map_err(gerr)?;
        Ok(o)
    }

    /// Unkeyed plaintext operand: encoded at `a`'s level into the eager scratch key, then used.
    fn pt_values_op(&mut self, py: Python<'_>, a: &PyCt, v: &Bound<'_, PyAny>, add: bool) -> PyResult<PyCt> {
        self.same_session(a)?;
        let values = self.values(py, v)?;
        let key = PtKey::new("__eager", self.pt_meta(&a.meta()).level, PtForm::Eval);
        Self::ensure_pt(&mut self.dev.st.borrow_mut(), &self.dev, &key, &values, true)?;
        let mut o = self.pt_op(a, &key, add)?;
        if !add {
            self.reactive(py, &mut o)?;
        }
        Ok(o)
    }

    /// A named/cached plaintext product at `a`'s level (`cached`: the encode cache's tag map, else the slot names;
    /// `values` registers the tag on its first use).
    fn named_op(&mut self, py: Python<'_>, a: &PyCt, name: &str, values: Option<Vec<Complex>>, cached: bool, add: bool) -> PyResult<PyCt> {
        self.same_session(a)?;
        let d = self.dev.clone();
        let level = self.pt_meta(&a.meta()).level;
        let key = PtKey::new(name, level, PtForm::Eval);
        {
            let mut st = d.st.borrow_mut();
            let State { named, cached: cmap, pts, ex, ops, enc_hits, enc_misses, .. } = &mut *st;
            let map = if cached { cmap } else { named };
            if let Some(v) = values {
                map.entry(name.to_string()).or_insert_with(|| Named { values: v, levels: BTreeSet::new() });
            }
            let ent = map.get_mut(name).ok_or_else(|| PyKeyError::new_err(format!("slot plaintext '{name}' is not set (set_slot_pt)")))?;
            ent.levels.insert(level);
            if pts.has(&key) {
                *enc_hits += cached as u64;
            } else {
                *enc_misses += cached as u64;
                let what = Payload2::Slots { values: &ent.values, slots: d.slots(), payload: d.payload() };
                pts.encode_into(&d.ctx, ex, ops, &d.sh, key.clone(), what, 1).map_err(gerr)?;
            }
        }
        let mut o = self.pt_op(a, &key, add)?;
        if !add {
            self.reactive(py, &mut o)?;
        }
        Ok(o)
    }

    /// `out = a * b` (ct x ct, relinearized), no reactive check.
    fn mult_ct(&self, a: &PyCt, b: &PyCt) -> PyResult<PyCt> {
        let d = &*self.dev;
        let mut o = self.new_ct(a.meta())?;
        let om = o.ct_mut()?;
        let mut st = d.st.borrow_mut();
        let source = d.source;
        let State { ops, ex, plans, groups, .. } = &mut *st;
        let kc = KsCfg { plans, source };
        if std::ptr::eq(a, b) {
            ops.square(&d.ctx, ex, kc, &d.sh, &groups.relin, om, a.ct()).map_err(gerr)?;
        } else {
            ops.mult(&d.ctx, ex, kc, &d.sh, &groups.relin, om, a.ct(), b.ct()).map_err(gerr)?;
        }
        Ok(o)
    }

    fn mult_scalar_ct(&self, a: &PyCt, c: Complex) -> PyResult<PyCt> {
        let d = &*self.dev;
        let mut o = self.new_ct(a.meta())?;
        let om = o.ct_mut()?;
        let mut st = d.st.borrow_mut();
        let State { ops, ex, .. } = &mut *st;
        ops.mult_scalar(&d.ctx, ex, &d.sh, std::slice::from_mut(om), &[a.ct()], &[c]).map_err(gerr)?;
        Ok(o)
    }

    fn add_inplace_ct(&self, a: &mut PyCt, b: &PyCt) -> PyResult<()> {
        let d = &*self.dev;
        let am = a.ct_mut()?;
        let mut st = d.st.borrow_mut();
        let State { ops, ex, .. } = &mut *st;
        ops.add_inplace(&d.ctx, ex, &d.sh, std::slice::from_mut(am), &[b.ct()]).map_err(gerr)
    }

    fn check_steps(&self, steps: &[i64]) -> PyResult<()> {
        let st = self.dev.st.borrow();
        for &s in steps {
            if !st.groups.has_step(s) {
                return Err(PyKeyError::new_err(format!("no rotation key for {s} (KeySource rot_steps)")));
            }
        }
        Ok(())
    }

    fn rotate_ct(&self, a: &PyCt, k: i64) -> PyResult<PyCt> {
        self.same_session(a)?;
        self.check_steps(&[k])?;
        let d = &*self.dev;
        let mut o = self.new_ct(a.meta())?;
        let om = o.ct_mut()?;
        let mut st = d.st.borrow_mut();
        let source = d.source;
        let State { ex, plans, groups, .. } = &mut *st;
        let kc = KsCfg { plans, source };
        let g = groups.group(&[k]).ok_or_else(|| PyKeyError::new_err(format!("no rotation key for {k}")))?;
        rotate::rotate(&d.ctx, ex, kc, &d.sh, g, a.ct(), k, om).map_err(gerr)?;
        Ok(o)
    }

    fn sum_terms(&mut self, py: Python<'_>, cts: &[PyRef<'_, PyCt>], pts: Vec<Payload2Owned>) -> PyResult<PyCt> {
        if cts.is_empty() || cts.len() != pts.len() {
            return Err(PyValueError::new_err(format!("{} ciphertexts, {} plaintexts", cts.len(), pts.len())));
        }
        let d = self.dev.clone();
        let mut out: Option<PyCt> = None;
        for (c, p) in cts.iter().zip(&pts) {
            self.same_session(c)?;
            let pm = self.pt_meta(&c.meta());
            let key = PtKey::new("__sum", pm.level, PtForm::Eval);
            {
                let mut st = d.st.borrow_mut();
                let State { pts: store, ex, ops, .. } = &mut *st;
                let (kind, what) = match p {
                    Payload2Owned::Slots(v) => (EncoderKind::DeviceFft, Payload2::Slots { values: v, slots: d.slots(), payload: d.payload() }),
                    Payload2Owned::Coeffs(v) => (EncoderKind::CoeffRows, Payload2::Coeffs(v)),
                };
                store.encode_with(&d.ctx, ex, ops, &d.sh, kind, key.clone(), what, 1).map_err(gerr)?;
            }
            match out.as_mut() {
                None => out = Some(self.pt_op(c, &key, false)?),
                Some(acc) => {
                    // fused accumulation when the term needs no adjustment, else the loop (Ops::mult_add_many
                    // refuses terms with preludes)
                    let fast = {
                        let r = d.sh.mult_pt(&c.meta(), &PtMeta { level: pm.level, deg: 1, nf: 0.0, tag: Default::default() }).map_err(|e| gerr(e.into()))?;
                        r.pre.is_empty() && d.sh.add(&acc.meta(), &r.out).map_err(|e| gerr(e.into()))?.pre.is_empty()
                    };
                    if fast {
                        let am = acc.ct_mut()?;
                        let mut st = d.st.borrow_mut();
                        let State { pts: store, ex, ops, .. } = &mut *st;
                        let pt = store.get(&d.ctx, ex, &key).map_err(gerr)?;
                        ops.mult_add_many(&d.ctx, ex, &d.sh, am, &[c.ct()], &[pt]).map_err(gerr)?;
                    } else {
                        let t = self.pt_op(c, &key, false)?;
                        self.add_inplace_ct(acc, &t)?;
                    }
                }
            }
        }
        let mut o = out.ok_or_else(|| rt_err("empty sum"))?;
        self.reactive(py, &mut o)?;
        Ok(o)
    }
}

enum Payload2Owned {
    Slots(Vec<Complex>),
    Coeffs(Vec<f64>),
}

/// Centered CRT over 4 rows + the slot decode (`Encoder::decode_slots` reads only the bottom pair).
fn decode4(enc: &pc_core::encode::Encoder, q: &[u64], rows: &[Vec<u64>], slots: usize, payload: Payload, scale: f64, n: usize) -> Vec<Complex> {
    let big: u128 = q.iter().fold(1u128, |a, &x| a * x as u128);
    let hats: Vec<(u128, u64)> = q
        .iter()
        .map(|&qi| {
            let hat = big / qi as u128;
            (hat, pc_core::modmul::invmod((hat % qi as u128) as u64, qi))
        })
        .collect();
    let coef = |j: usize| -> f64 {
        let mut x: u128 = 0;
        for (i, &(hat, hinv)) in hats.iter().enumerate() {
            let t = pc_core::modmul::mulmod(rows[i][j] % q[i], hinv, q[i]) as u128;
            x = (x + t * hat) % big;
        }
        if x > big / 2 {
            -((big - x) as f64)
        } else {
            x as f64
        }
    };
    let gap = n / (2 * slots);
    let mut v: Vec<Complex> = (0..slots).map(|i| Complex::new(coef(gap * i) / scale, coef(gap * i + n / 2) / scale)).collect();
    enc.fft_special(&mut v);
    if payload == Payload::Real {
        v.iter_mut().for_each(|x| x.im = 0.0);
    }
    v
}

fn as_ct<'py>(b: &Bound<'py, PyAny>) -> Option<PyRef<'py, PyCt>> {
    b.extract::<PyRef<'py, PyCt>>().ok()
}

#[pymethods]
impl PyFhe {
    // ── identity / geometry ───────────────────────────────────────────────────────────────────────────────────────

    /// The impl layer reads `inf.fhe`: this object.
    #[getter]
    fn fhe(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    #[getter]
    fn slots(&self) -> usize {
        self.dev.slots()
    }
    #[getter(logN)]
    fn log_n(&self) -> u32 {
        (self.dev.slots() as u64).ilog2() + 1
    }
    #[getter]
    fn size(&self) -> PySize {
        let g = &self.dev.geometry;
        PySize { hid: g.hid, dim: g.dim, heads: g.heads, heads_real: g.heads_real, exp: g.exp, expanded: g.exp_real, seq_len: g.seq_len }
    }
    /// CKKS_COMPLEX: the slots carry complex values.
    #[getter]
    fn complex(&self) -> bool {
        self.dev.complex
    }
    #[getter]
    fn complex_payload(&self) -> bool {
        self.dev.complex
    }
    /// Primes per CKKS level (composite degree).
    #[getter]
    fn unit(&self) -> u32 {
        self.dev.sh.unit()
    }
    #[getter]
    fn has_secret_key(&self) -> bool {
        true
    }
    /// Rotation steps with a key (normalized to [0, slots)).
    #[getter]
    fn loaded_rot_steps(&self) -> Vec<i64> {
        self.dev.st.borrow().groups.steps.iter().copied().collect()
    }
    #[getter]
    fn debug_refresh(&self) -> bool {
        self.dev.debug_refresh
    }
    fn has_rot_keys(&self, steps: Vec<i64>) -> bool {
        let st = self.dev.st.borrow();
        steps.iter().all(|&s| st.groups.has_step(s))
    }
    fn level_limit(&self) -> u32 {
        self.dev.level_limit()
    }
    fn bootstrap_output_level(&self) -> u32 {
        self.dev.bts_out()
    }
    #[pyo3(signature = (_label, _shape=None))]
    fn step(&self, _label: &Bound<'_, PyAny>, _shape: Option<&Bound<'_, PyAny>>) -> NullScope {
        NullScope
    }
    #[pyo3(signature = (*_a))]
    fn name_ct(&self, _a: &Bound<'_, PyTuple>) {}
    #[pyo3(signature = (*_a))]
    fn name_ct_if_absent(&self, _a: &Bound<'_, PyTuple>) {}
    fn var_name(&self, _a: &Bound<'_, PyAny>) -> &'static str {
        "x"
    }
    #[pyo3(signature = (_a, _level))]
    fn level_hint(&self, _a: &Bound<'_, PyAny>, _level: i64) {}
    /// Bookkeeping for a refresh computed outside the recorded ops (PCMM / CCMM finishes).
    #[pyo3(signature = (_a, _inputs, _input_levels, landing=None, desc=None))]
    fn record_external_bootstrap(&mut self, _a: &Bound<'_, PyAny>, _inputs: &Bound<'_, PyAny>, _input_levels: &Bound<'_, PyAny>, landing: Option<&Bound<'_, PyAny>>, desc: Option<&Bound<'_, PyAny>>) {
        let _ = (landing, desc);
        self.total_bootstraps += 1;
    }
    fn suppress_auto_bts(slf: Bound<'_, Self>) -> SuppressScope {
        SuppressScope { fhe: slf.unbind(), saved: false }
    }

    // ── arithmetic ────────────────────────────────────────────────────────────────────────────────────────────────

    /// `a + b` (ciphertext or scalar).
    fn add(&self, a: PyRef<'_, PyCt>, b: &Bound<'_, PyAny>) -> PyResult<PyCt> {
        self.same_session(&a)?;
        let d = &*self.dev;
        let mut o = self.new_ct(a.meta())?;
        let om = o.ct_mut()?;
        let mut st = d.st.borrow_mut();
        let State { ops, ex, .. } = &mut *st;
        match as_ct(b) {
            Some(b) => {
                self.same_session(&b)?;
                ops.add(&d.ctx, ex, &d.sh, std::slice::from_mut(om), &[a.ct()], &[b.ct()]).map_err(gerr)?
            }
            None => ops.add_scalar(&d.ctx, ex, &d.sh, std::slice::from_mut(om), &[a.ct()], &[b.extract::<f64>()?]).map_err(gerr)?,
        }
        Ok(o)
    }

    /// `a - b` (ciphertext or scalar).
    fn sub(&self, a: PyRef<'_, PyCt>, b: &Bound<'_, PyAny>) -> PyResult<PyCt> {
        self.same_session(&a)?;
        let d = &*self.dev;
        let mut o = self.new_ct(a.meta())?;
        let om = o.ct_mut()?;
        let mut st = d.st.borrow_mut();
        let State { ops, ex, .. } = &mut *st;
        match as_ct(b) {
            Some(b) => {
                self.same_session(&b)?;
                ops.sub(&d.ctx, ex, &d.sh, std::slice::from_mut(om), &[a.ct()], &[b.ct()]).map_err(gerr)?
            }
            None => ops.add_scalar(&d.ctx, ex, &d.sh, std::slice::from_mut(om), &[a.ct()], &[-b.extract::<f64>()?]).map_err(gerr)?,
        }
        Ok(o)
    }

    /// `a * b` (ciphertext, relinearized; or scalar), then the reactive refresh check.
    fn mult(&mut self, py: Python<'_>, a: PyRef<'_, PyCt>, b: &Bound<'_, PyAny>) -> PyResult<PyCt> {
        self.same_session(&a)?;
        let mut o = match as_ct(b) {
            Some(b) => {
                self.same_session(&b)?;
                self.mult_ct(&a, &b)?
            }
            None => self.mult_scalar_ct(&a, Complex::new(b.extract::<f64>()?, 0.0))?,
        };
        self.reactive(py, &mut o)?;
        Ok(o)
    }

    fn square(&mut self, py: Python<'_>, a: PyRef<'_, PyCt>) -> PyResult<PyCt> {
        self.same_session(&a)?;
        let mut o = self.mult_ct(&a, &a)?;
        self.reactive(py, &mut o)?;
        Ok(o)
    }

    fn negate(&self, a: PyRef<'_, PyCt>) -> PyResult<PyCt> {
        self.same_session(&a)?;
        let d = &*self.dev;
        let mut o = self.new_ct(a.meta())?;
        let om = o.ct_mut()?;
        let mut st = d.st.borrow_mut();
        let State { ops, ex, .. } = &mut *st;
        ops.negate(&d.ctx, ex, &d.sh, std::slice::from_mut(om), &[a.ct()]).map_err(gerr)?;
        Ok(o)
    }

    /// `i * a` (the monomial X^{N/2}; level-free).
    fn mult_i(&self, a: PyRef<'_, PyCt>) -> PyResult<PyCt> {
        self.same_session(&a)?;
        let d = &*self.dev;
        let mut o = self.new_ct(a.meta())?;
        let om = o.ct_mut()?;
        let mut st = d.st.borrow_mut();
        let State { ops, ex, .. } = &mut *st;
        ops.mult_i(&d.ctx, ex, &d.sh, std::slice::from_mut(om), &[a.ct()]).map_err(gerr)?;
        Ok(o)
    }

    /// In place: `a += b`.
    fn inplace_add(&self, a: &Bound<'_, PyCt>, b: &Bound<'_, PyCt>) -> PyResult<()> {
        let bb = if a.is(b) { Some(self.copy_ct(&b.borrow())?) } else { None };
        let mut am = a.borrow_mut();
        self.same_session(&am)?;
        match &bb {
            Some(c) => self.add_inplace_ct(&mut am, c),
            None => {
                let br = b.borrow();
                self.same_session(&br)?;
                self.add_inplace_ct(&mut am, &br)
            }
        }
    }

    /// `acc += sum_j vs[j] * ss[j]` (ciphertexts or scalars), as the loop of products and adds. Returns True.
    fn mult_add_many(&mut self, py: Python<'_>, acc: &Bound<'_, PyCt>, vs: Vec<Bound<'_, PyAny>>, ss: Vec<Bound<'_, PyAny>>) -> PyResult<bool> {
        if vs.len() != ss.len() {
            return Err(PyValueError::new_err(format!("mult_add_many: {} values, {} factors", vs.len(), ss.len())));
        }
        for (v, s) in vs.iter().zip(&ss) {
            let t = {
                let vr: PyRef<'_, PyCt> = v.extract()?;
                self.mult(py, vr, s)?
            };
            let mut am = acc.borrow_mut();
            self.add_inplace_ct(&mut am, &t)?;
        }
        Ok(true)
    }

    /// `sum_j ts[j] * cs[j]`: one fused weighted sum when every term shares level and degree 1, else the loop.
    fn mult_scalar_sum(&mut self, py: Python<'_>, ts: Vec<PyRef<'_, PyCt>>, cs: Vec<f64>) -> PyResult<PyCt> {
        if ts.is_empty() || ts.len() != cs.len() {
            return Err(PyValueError::new_err(format!("mult_scalar_sum: {} terms, {} scalars", ts.len(), cs.len())));
        }
        for t in &ts {
            self.same_session(t)?;
        }
        let m0 = ts[0].meta();
        let mut o = if ts.iter().all(|t| t.meta().level == m0.level && t.meta().deg == 1) {
            let d = &*self.dev;
            let mut o = self.new_ct(m0)?;
            {
                let om = o.ct_mut()?;
                let mut st = d.st.borrow_mut();
                let State { ops, ex, .. } = &mut *st;
                let refs: Vec<&Ct<'static, W>> = ts.iter().map(|t| t.ct()).collect();
                ops.mult_scalar_sum(&d.ctx, ex, &d.sh, om, &refs, &cs).map_err(gerr)?;
            }
            o
        } else {
            let mut r = self.mult_scalar_ct(&ts[0], Complex::new(cs[0], 0.0))?;
            for (t, &c) in ts.iter().zip(&cs).skip(1) {
                let p = self.mult_scalar_ct(t, Complex::new(c, 0.0))?;
                self.add_inplace_ct(&mut r, &p)?;
            }
            r
        };
        self.reactive(py, &mut o)?;
        Ok(o)
    }

    // ── rotations ─────────────────────────────────────────────────────────────────────────────────────────────────

    /// Rotation by `k` slots (left: `np.roll(x, -k)`).
    fn rotate(&self, a: PyRef<'_, PyCt>, k: i64) -> PyResult<PyCt> {
        self.rotate_ct(&a, k)
    }

    fn conjugate(&self, a: PyRef<'_, PyCt>) -> PyResult<PyCt> {
        self.same_session(&a)?;
        let d = &*self.dev;
        let mut o = self.new_ct(a.meta())?;
        let om = o.ct_mut()?;
        let mut st = d.st.borrow_mut();
        let source = d.source;
        let State { ex, plans, groups, .. } = &mut *st;
        let g = groups.conj.as_ref().ok_or_else(|| PyKeyError::new_err("no conjugation key (KeySource conj=True)"))?;
        rotate::conjugate(&d.ctx, ex, KsCfg { plans, source }, &d.sh, g, a.ct(), om).map_err(gerr)?;
        Ok(o)
    }

    /// Rotations of `a` by each step from one shared decomposition (outputs equal plain rotations).
    fn rotate_hoisted(&self, a: PyRef<'_, PyCt>, steps: Vec<i64>) -> PyResult<Vec<PyCt>> {
        self.same_session(&a)?;
        self.check_steps(&steps)?;
        let d = &*self.dev;
        let mut outs = steps.iter().map(|_| self.new_ct(a.meta())).collect::<PyResult<Vec<_>>>()?;
        {
            let mut refs = outs.iter_mut().map(|o| o.ct_mut()).collect::<PyResult<Vec<_>>>()?;
            let nz: Vec<i64> = steps.iter().copied().filter(|s| s.rem_euclid(d.slots() as i64) != 0).collect();
            let mut st = d.st.borrow_mut();
            let source = d.source;
            let State { ex, plans, groups, .. } = &mut *st;
            let kc = KsCfg { plans, source };
            if nz.is_empty() {
                for (r, &s) in refs.iter_mut().zip(&steps) {
                    let g = groups.relin_or_all();
                    rotate::rotate(&d.ctx, ex, kc, &d.sh, g, a.ct(), s, &mut **r).map_err(gerr)?;
                }
            } else {
                let g = groups.group(&nz).ok_or_else(|| PyKeyError::new_err(format!("no rotation keys for {nz:?}")))?;
                rotate::rotate_hoisted(&d.ctx, ex, kc, &d.sh, g, a.ct(), &steps, &mut refs).map_err(gerr)?;
            }
        }
        Ok(outs)
    }

    /// `a + sum_k rot(a, steps[k])` (one hoisted radix group, one ModDown).
    fn rotate_sum(&self, a: PyRef<'_, PyCt>, steps: Vec<i64>) -> PyResult<PyCt> {
        self.same_session(&a)?;
        self.check_steps(&steps)?;
        let d = &*self.dev;
        let mut o = self.new_ct(a.meta())?;
        let om = o.ct_mut()?;
        let mut st = d.st.borrow_mut();
        let source = d.source;
        let State { ex, plans, groups, .. } = &mut *st;
        let g = groups.group(&steps).ok_or_else(|| PyKeyError::new_err(format!("no rotation keys for {steps:?}")))?;
        rotate::rotate_sum(&d.ctx, ex, KsCfg { plans, source }, &d.sh, g, a.ct(), &steps, om).map_err(gerr)?;
        Ok(o)
    }

    /// The C++ ladder: x += rot(x, g) for g = start, 2 start, ... (|g| < stop), in radix groups of
    /// `model.ladder_radix_bits` (a group with all its keys is one rotate_sum).
    fn rotate_and_sum(&self, a: PyRef<'_, PyCt>, start: i64, stop: i64) -> PyResult<PyCt> {
        self.same_session(&a)?;
        let d = &*self.dev;
        let mut o = self.new_ct(a.meta())?;
        let mut tmp = self.new_ct(a.meta())?;
        let om = o.ct_mut()?;
        let tm = tmp.ct_mut()?;
        let mut st = d.st.borrow_mut();
        let source = d.source;
        let State { ex, plans, groups, .. } = &mut *st;
        let keys: &dyn RotKeys = groups;
        rotate::rotate_and_sum(&d.ctx, ex, KsCfg { plans, source }, &d.sh, keys, a.ct(), start, stop, d.radix_bits, om, tm).map_err(gerr)?;
        Ok(o)
    }

    // ── levels ────────────────────────────────────────────────────────────────────────────────────────────────────

    /// In place: pay a pending rescale (`core.realize_pending_rescale`).
    fn realize(&self, a: &Bound<'_, PyCt>) -> PyResult<()> {
        let mut am = a.borrow_mut();
        self.same_session(&am)?;
        let d = &*self.dev;
        let c = am.ct_mut()?;
        let mut st = d.st.borrow_mut();
        let State { ops, ex, .. } = &mut *st;
        ops.realize(&d.ctx, ex, &d.sh, std::slice::from_mut(c)).map_err(gerr)
    }

    /// In place: drop to `level` if above it (value-preserving modulus reduction; no data moves).
    fn drop_to_level(&self, a: &Bound<'_, PyCt>, level: u32) -> PyResult<()> {
        let mut am = a.borrow_mut();
        let d = &*self.dev;
        let c = am.ct_mut()?;
        let st = d.st.borrow();
        st.ops.drop_to_level(&d.sh, std::slice::from_mut(c), level);
        Ok(())
    }

    /// In place: stamp a reduction output's slot period (pack tag).
    fn tag_reduce(&self, a: &Bound<'_, PyCt>, stride: i64) -> PyResult<()> {
        let mut am = a.borrow_mut();
        let m = self.dev.sh.tag_reduce(&am.meta(), stride).out;
        am.ct_mut()?.meta = m;
        Ok(())
    }

    /// The (level, degree) a plaintext must carry to meet `a` with no forced rescale.
    fn encode_like_level(&self, a: PyRef<'_, PyCt>) -> (u32, u8) {
        (self.pt_meta(&a.meta()).level, 1)
    }

    // ── bootstrap family ──────────────────────────────────────────────────────────────────────────────────────────

    /// In place: refresh `ct` on `route` (dense by default). NotImplemented without `debug_refresh` (module doc).
    #[pyo3(signature = (ct, route=None))]
    fn bootstrap(&mut self, py: Python<'_>, ct: &Bound<'_, PyCt>, route: Option<PyRoute>) -> PyResult<()> {
        let r = route.map(|r| r.0).unwrap_or_else(Route::dense);
        let mut c = ct.borrow_mut();
        self.refresh_route(py, &mut c, r, "bootstrap")
    }

    /// In place: refresh iff the (effective, when `account_pending_rescale`) level is past `level_threshold`.
    #[pyo3(signature = (ct, level_threshold, account_pending_rescale=false))]
    fn bootstrap_hint(&mut self, py: Python<'_>, ct: &Bound<'_, PyCt>, level_threshold: u32, account_pending_rescale: bool) -> PyResult<()> {
        let mut c = ct.borrow_mut();
        let route = Route::dense();
        let h = self.dev.sh.hint(&c.meta(), level_threshold, account_pending_rescale, &route).map_err(|e| gerr(e.into()))?;
        if h.fired {
            self.refresh_to(py, &mut c, h.recipe.out, "bootstrap_hint", |_| {})?;
        }
        Ok(())
    }

    /// In place: the reactive refresh, only if `ct` is at or past `level_limit` (and not suppressed).
    fn maybe_bootstrap(&mut self, py: Python<'_>, ct: &Bound<'_, PyCt>) -> PyResult<()> {
        let mut c = ct.borrow_mut();
        if self.auto_bts_suppressed || c.meta().level < self.dev.level_limit() {
            return Ok(());
        }
        self.refresh_route(py, &mut c, Route::dense(), "maybe_bootstrap")
    }

    /// The sparse slot count a fold for `s_wanted` runs at (the eager session has no sparse routes: the full count).
    fn fold_slots_for(&self, s_wanted: usize) -> usize {
        let _ = s_wanted;
        self.dev.slots()
    }

    /// In place: the sparse refresh finishing a ladder stopped at stride `s` (class sums over the slots/s copies,
    /// divided by `n_live`). NotImplemented without `debug_refresh`.
    #[pyo3(signature = (ct, s, n_live, prescale=1.0))]
    fn fold_bootstrap(&mut self, py: Python<'_>, ct: &Bound<'_, PyCt>, s: usize, n_live: u32, prescale: f64) -> PyResult<()> {
        let mut c = ct.borrow_mut();
        let land = self.dev.sh.fold_bootstrap(&c.meta(), s as u32, n_live, prescale).map_err(|e| gerr(e.into()))?.out;
        let total = self.dev.slots();
        if s == 0 || total % s != 0 {
            return Err(PyValueError::new_err(format!("fold_bootstrap: s = {s} does not divide {total} slots")));
        }
        self.refresh_to(py, &mut c, land, "fold_bootstrap", |v| {
            let mut x: Vec<Complex> = v.iter().map(|&z| z * prescale).collect();
            let mut gap = s;
            while gap < total {
                let y: Vec<Complex> = (0..total).map(|i| x[i] + x[(i + gap) % total]).collect();
                x = y;
                gap *= 2;
            }
            *v = x.into_iter().map(|z| z * (1.0 / n_live as f64 / prescale)).collect();
        })
    }


    // ── plaintext operands ────────────────────────────────────────────────────────────────────────────────────────

    /// `a * values` (encoded at `a`'s level at every call).
    fn mult_pt(&mut self, py: Python<'_>, a: PyRef<'_, PyCt>, values: &Bound<'_, PyAny>) -> PyResult<PyCt> {
        self.pt_values_op(py, &a, values, false)
    }
    fn add_pt(&mut self, py: Python<'_>, a: PyRef<'_, PyCt>, values: &Bound<'_, PyAny>) -> PyResult<PyCt> {
        self.pt_values_op(py, &a, values, true)
    }

    /// `a * (re + i im)`: a plaintext product with a constant vector (costs a level like a mask).
    fn mult_const(&mut self, py: Python<'_>, a: PyRef<'_, PyCt>, re: f64, im: f64) -> PyResult<PyCt> {
        let c = if self.dev.complex { Complex::new(re, im) } else { Complex::new(re, 0.0) };
        let name = format!("__const:{re:e}:{im:e}");
        let v = vec![c; self.dev.slots()];
        self.named_op(py, &a, &name, Some(v), true, false)
    }

    /// Register the named plaintext `name` (host values; encoded per level on first use).
    #[pyo3(signature = (name, values, level=0, deg=1))]
    fn set_slot_pt(&mut self, py: Python<'_>, name: &str, values: &Bound<'_, PyAny>, level: u32, deg: u8) -> PyResult<()> {
        let _ = (level, deg);
        let v = self.values(py, values)?;
        let mut st = self.dev.st.borrow_mut();
        st.named.insert(name.to_string(), Named { values: v, levels: BTreeSet::new() });
        Ok(())
    }
    fn has_slot_pt(&self, name: &str) -> bool {
        self.dev.st.borrow().named.contains_key(name)
    }
    fn mult_slot_pt(&mut self, py: Python<'_>, a: PyRef<'_, PyCt>, name: &str) -> PyResult<PyCt> {
        self.named_op(py, &a, name, None, false, false)
    }
    fn add_slot_pt(&mut self, py: Python<'_>, a: PyRef<'_, PyCt>, name: &str) -> PyResult<PyCt> {
        self.named_op(py, &a, name, None, false, true)
    }
    fn mult_slot_pt_many(&mut self, py: Python<'_>, a: PyRef<'_, PyCt>, names: Vec<String>) -> PyResult<Vec<PyCt>> {
        names.iter().map(|n| self.named_op(py, &a, n, None, false, false)).collect()
    }
    /// Named plaintexts with `prefix` (their encodings are made on use).
    fn load_slot_pts(&self, prefix: &str) -> usize {
        self.dev.st.borrow().named.keys().filter(|k| k.starts_with(prefix)).count()
    }
    /// Drop the device encodings of the named plaintexts with `prefix` (host values stay).
    fn evict_slot_pts(&self, prefix: &str) -> PyResult<usize> {
        let mut st = self.dev.st.borrow_mut();
        let State { named, pts, ex, .. } = &mut *st;
        let mut n = 0;
        for (k, v) in named.iter_mut().filter(|(k, _)| k.starts_with(prefix)) {
            for &l in &v.levels {
                pts.evict(ex, &PtKey::new(k.clone(), l, PtForm::Eval)).map_err(gerr)?;
            }
            v.levels.clear();
            n += 1;
        }
        Ok(n)
    }
    /// Forget the named plaintexts with `prefix` (device encodings and host values).
    fn drop_slot_pts(&self, prefix: &str) -> PyResult<usize> {
        let n = self.evict_slot_pts(prefix)?;
        self.dev.st.borrow_mut().named.retain(|k, _| !k.starts_with(prefix));
        Ok(n)
    }

    /// `a * values` through the encode cache (keyed by `tag` and `a`'s level; encoded on a miss).
    #[pyo3(signature = (a, tag, values, tagged=true))]
    fn mult_cached(&mut self, py: Python<'_>, a: PyRef<'_, PyCt>, tag: &str, values: &Bound<'_, PyAny>, tagged: bool) -> PyResult<PyCt> {
        let _ = tagged;
        let v = self.cached_values(py, tag, values)?;
        self.named_op(py, &a, tag, v, true, false)
    }
    #[pyo3(signature = (a, tag, values, tagged=true))]
    fn add_cached(&mut self, py: Python<'_>, a: PyRef<'_, PyCt>, tag: &str, values: &Bound<'_, PyAny>, tagged: bool) -> PyResult<PyCt> {
        let _ = tagged;
        let v = self.cached_values(py, tag, values)?;
        self.named_op(py, &a, tag, v, true, true)
    }
    fn mult_cached_many(&mut self, py: Python<'_>, a: PyRef<'_, PyCt>, tags: Vec<String>, values: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<PyCt>> {
        let mut out = Vec::with_capacity(tags.len());
        for (t, v) in tags.iter().zip(&values) {
            let vv = self.cached_values(py, t, v)?;
            out.push(self.named_op(py, &a, t, vv, true, false)?);
        }
        Ok(out)
    }
    /// Encode `values` under `tag` at `level` now (synchronously, on the device).
    fn prime_pt(&mut self, py: Python<'_>, tag: &str, values: &Bound<'_, PyAny>, level: u32) -> PyResult<()> {
        let v = self.values(py, values)?;
        let d = self.dev.clone();
        let mut st = d.st.borrow_mut();
        let key = PtKey::new(tag, level, PtForm::Eval);
        st.cached.entry(tag.to_string()).or_insert_with(|| Named { values: v.clone(), levels: BTreeSet::new() }).levels.insert(level);
        Self::ensure_pt(&mut st, &d, &key, &v, false)?;
        Ok(())
    }
    /// Encode `[(tag, level, values)]` on the host workers and upload them on the copy lane; `adopt_pts(handle)`
    /// joins them into the compute lane.
    fn stage_pts(&mut self, py: Python<'_>, items: Vec<(String, u32, Bound<'_, PyAny>)>) -> PyResult<StageHandle> {
        let d = self.dev.clone();
        let mut work = Vec::with_capacity(items.len());
        for (tag, level, v) in &items {
            let values = self.values(py, v)?;
            work.push(StageItem {
                key: PtKey::new(tag.clone(), *level, PtForm::Eval),
                values: StageValues::Slots { values, slots: d.slots(), payload: d.payload() },
                deg: 1,
                scale: None,
            });
        }
        let mut st = d.st.borrow_mut();
        for it in &work {
            if let StageValues::Slots { values, .. } = &it.values {
                st.cached.entry(it.key.name.clone()).or_insert_with(|| Named { values: values.clone(), levels: BTreeSet::new() }).levels.insert(it.key.level);
            }
        }
        let State { pts, ex, .. } = &mut *st;
        nogil(py, || pts.stage(ex, &d.sh, &work)).map_err(gerr)?;
        Ok(StageHandle { keys: work.into_iter().map(|w| w.key).collect() })
    }
    /// Join a `stage_pts` handle; returns the number adopted.
    fn adopt_pts(&self, handle: &StageHandle) -> PyResult<usize> {
        let d = &*self.dev;
        let mut st = d.st.borrow_mut();
        let State { pts, ex, .. } = &mut *st;
        let keys: Vec<PtKey> = handle.keys.iter().filter(|k| pts.has(k)).cloned().collect();
        pts.adopt(&d.ctx, ex, &keys).map_err(gerr)?;
        Ok(keys.len())
    }
    /// Evict every level of each cached `tag`; returns the number of plaintexts freed.
    fn erase_pts(&self, tags: Vec<String>) -> PyResult<usize> {
        let mut st = self.dev.st.borrow_mut();
        let State { cached, pts, ex, .. } = &mut *st;
        let mut n = 0;
        for t in &tags {
            if let Some(e) = cached.remove(t) {
                for l in e.levels {
                    if pts.evict(ex, &PtKey::new(t.clone(), l, PtForm::Eval)).map_err(gerr)? {
                        n += 1;
                    }
                }
            }
        }
        Ok(n)
    }
    fn enc_cache_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let st = self.dev.st.borrow();
        let d = PyDict::new(py);
        d.set_item("size", st.cached.values().map(|e| e.levels.len()).sum::<usize>())?;
        d.set_item("hits", st.enc_hits)?;
        d.set_item("misses", st.enc_misses)?;
        d.set_item("strict_misses", 0)?;
        Ok(d)
    }

    /// `sum_i cts[i] * vecs[i]`, each plaintext encoded on the device at its ciphertext's level (the loop's
    /// bookkeeping: product, then add; fused accumulation where no term needs an adjustment).
    fn mult_pt_sum(&mut self, py: Python<'_>, cts: Vec<PyRef<'_, PyCt>>, vecs: &Bound<'_, PyAny>) -> PyResult<PyCt> {
        let mut terms = Vec::with_capacity(cts.len());
        for v in vecs.try_iter()? {
            terms.push(Payload2Owned::Slots(self.values(py, &v?)?));
        }
        self.sum_terms(py, &cts, terms)
    }
    /// `mult_pt_sum` from the plaintexts' coefficient rows (`layout.coeff_rows`: values / scale per coefficient),
    /// encoded without an FFT. `tags` (pack-tag fields) are not needed eagerly.
    fn mult_coeff_sum(&mut self, py: Python<'_>, cts: Vec<PyRef<'_, PyCt>>, rows: &Bound<'_, PyAny>, tags: &Bound<'_, PyAny>) -> PyResult<PyCt> {
        let _ = tags;
        let r = self.rows_f64(py, rows)?;
        let terms = r.into_iter().take(cts.len()).map(Payload2Owned::Coeffs).collect();
        self.sum_terms(py, &cts, terms)
    }

    // ── data plane ────────────────────────────────────────────────────────────────────────────────────────────────

    /// Fresh encryption of `values` (slot vector) at `level` (degree `deg`). Tests and public-value inputs.
    #[pyo3(signature = (values, level, deg=1))]
    fn encrypt(&mut self, py: Python<'_>, values: &Bound<'_, PyAny>, level: u32, deg: u8) -> PyResult<PyCt> {
        let v = self.values(py, values)?;
        if v.len() > self.dev.slots() {
            return Err(PyValueError::new_err(format!("{} values exceed {} slots", v.len(), self.dev.slots())));
        }
        let meta = self.fresh_meta(level, deg)?;
        self.encrypt_meta(py, &v, meta)
    }

    /// One token's features `x` (<= dim) at slots i*t (t = slots / hidDim), at the bootstrap output level.
    fn encode_token_input(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<PyCt> {
        let v = self.values(py, x)?;
        if v.len() > self.dev.geometry.dim {
            return Err(PyValueError::new_err(format!("encode_token_input: {} features > dim {}", v.len(), self.dev.geometry.dim)));
        }
        let rows = vec![v];
        self.pack_rows(py, &rows, self.dev.bts_out())
    }

    /// Tokens `rows[tok]` (features <= dim each) at slots i*t + tok, at `level`.
    #[pyo3(signature = (rows, level=0))]
    fn pack_tokens(&mut self, py: Python<'_>, rows: &Bound<'_, PyAny>, level: u32) -> PyResult<PyCt> {
        let r: Vec<Vec<Complex>> = rows.try_iter()?.map(|x| self.values(py, &x?)).collect::<PyResult<_>>()?;
        self.pack_rows(py, &r, level)
    }

    /// The prefill input: `pack_tokens` at the bootstrap output level.
    fn encode_prefill_input(&mut self, py: Python<'_>, rows: &Bound<'_, PyAny>) -> PyResult<PyCt> {
        let l = self.dev.bts_out();
        self.pack_tokens(py, rows, l)
    }

    /// Real parts of every slot (float64).
    fn decrypt_slots(&mut self, py: Python<'_>, a: PyRef<'_, PyCt>) -> PyResult<Py<PyAny>> {
        let v = self.decrypt_values(py, &a)?;
        let re: Vec<f64> = v.iter().map(|c| c.re).collect();
        self.np_f64(py, &re)
    }
    fn decrypt_slots_complex(&mut self, py: Python<'_>, a: PyRef<'_, PyCt>) -> PyResult<Py<PyAny>> {
        let v = self.decrypt_values(py, &a)?;
        self.np_c128(py, &v)
    }
    /// One token's features (slots i*t, i < dim).
    fn decode_token_output(&mut self, py: Python<'_>, a: PyRef<'_, PyCt>) -> PyResult<Py<PyAny>> {
        let v = self.decrypt_values(py, &a)?;
        let g = &self.dev.geometry;
        let t = self.dev.slots() / g.hid.max(1);
        let out: Vec<f64> = (0..g.dim).map(|i| v[i * t].re).collect();
        self.np_f64(py, &out)
    }
    /// `layout.decode_linear_output` of the decrypted real slots.
    fn decode_linear_output(&mut self, py: Python<'_>, a: PyRef<'_, PyCt>, d_in: usize, d_out: usize) -> PyResult<Py<PyAny>> {
        let slots = self.decrypt_slots(py, a)?;
        let f = py.import("perseus.impl.layout")?.getattr("decode_linear_output")?;
        Ok(f.call1((slots, self.dev.slots(), d_in, d_out))?.unbind())
    }

    /// Coefficient residues of the live limbs: {"c0", "c1": (limbs, N) uint64, "moduli", "level" (top limb),
    /// "noise_level", "noise_factor"} (perseus `bts.export_limbs`).
    fn export_limbs<'py>(&mut self, py: Python<'py>, a: PyRef<'_, PyCt>) -> PyResult<Bound<'py, PyDict>> {
        self.same_session(&a)?;
        let l = {
            let d = &*self.dev;
            let mut st = d.st.borrow_mut();
            let State { ex, .. } = &mut *st;
            nogil(py, || pc_gpu::limbs::export_limbs(&d.ctx, ex, a.ct(), 0)).map_err(gerr)?
        };
        let out = PyDict::new(py);
        out.set_item("c0", self.np_u64_2d(py, &l.c[0])?)?;
        out.set_item("c1", self.np_u64_2d(py, &l.c[1])?)?;
        out.set_item("moduli", PyList::new(py, &l.moduli)?)?;
        out.set_item("level", l.level)?;
        out.set_item("noise_level", l.noise_level)?;
        out.set_item("noise_factor", l.noise_factor)?;
        Ok(out)
    }
    /// Overwrite `a` (at its current level) from coefficient residues (`export_limbs`' inverse).
    fn import_limbs(&self, py: Python<'_>, a: &Bound<'_, PyCt>, c0: &Bound<'_, PyAny>, c1: &Bound<'_, PyAny>, noise_factor: f64, noise_level: u8) -> PyResult<()> {
        let mut am = a.borrow_mut();
        self.same_session(&am)?;
        let d = &*self.dev;
        let n = d.n();
        let mut parts: [Vec<Vec<W>>; 2] = [Vec::new(), Vec::new()];
        for (p, src) in parts.iter_mut().zip([c0, c1]) {
            let v = PyBuffer::<u64>::get(src)?.to_vec(py)?;
            *p = v.chunks(n).map(|r| r.iter().map(|&x| x as W).collect()).collect();
        }
        let c = am.ct_mut()?;
        let top = c.top(d.ctx.shape).map_err(gerr)?;
        let l = pc_gpu::limbs::Limbs { c: parts, moduli: d.ctx.chain.q[..=top].to_vec(), level: top, noise_level, noise_factor };
        let mut st = d.st.borrow_mut();
        let State { ex, .. } = &mut *st;
        pc_gpu::limbs::import_limbs(&d.ctx, ex, c, 0, &l).map_err(gerr)
    }

    /// Host wait for every op issued so far (tests, timing).
    fn device_sync(&self, py: Python<'_>) -> PyResult<()> {
        let st = self.dev.st.borrow();
        nogil(py, || st.ex.compute.sync()).map_err(cuda)
    }
    /// (free, total) device bytes.
    fn mem_info(&self) -> PyResult<(usize, usize)> {
        self.dev.device.mem_info().map_err(cuda)
    }
    /// Free ciphertext pool slots.
    fn pool_free(&self) -> usize {
        self.dev.pool.free(1).unwrap_or(0)
    }
}

impl PyFhe {
    fn cached_values(&mut self, py: Python<'_>, tag: &str, values: &Bound<'_, PyAny>) -> PyResult<Option<Vec<Complex>>> {
        if self.dev.st.borrow().cached.contains_key(tag) {
            return Ok(None);
        }
        Ok(Some(self.values(py, values)?))
    }

    fn pack_rows(&mut self, py: Python<'_>, rows: &[Vec<Complex>], level: u32) -> PyResult<PyCt> {
        let d = &*self.dev;
        let g = &d.geometry;
        let slots = d.slots();
        let t = slots / g.hid.max(1);
        if rows.len() > t {
            return Err(PyValueError::new_err(format!("{} tokens > {t} token lanes", rows.len())));
        }
        let mut v = vec![Complex::default(); slots];
        for (tok, r) in rows.iter().enumerate() {
            for (i, x) in r.iter().take(g.dim).enumerate() {
                v[i * t + tok] = Complex::new(x.re, 0.0);
            }
        }
        let meta = self.fresh_meta(level, 1)?;
        self.encrypt_meta(py, &v, meta)
    }
}

impl super::session::Groups {
    /// A group for a step-0 rotation (a copy; any group works).
    fn relin_or_all(&self) -> &pc_gpu::ks::KeyGroup<'static> {
        self.all.as_ref().unwrap_or(&self.relin)
    }
}

// ── module-level data plane (perseus `_core.*(inf, ...)` functions) ──────────────────────────────────────────────

#[pyfunction]
fn encode_token_input(inf: &Bound<'_, PyAny>, x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    Ok(inf.call_method1("encode_token_input", (x,))?.unbind())
}
#[pyfunction]
#[pyo3(signature = (inf, rows, target_level=0))]
fn pack_tokens(inf: &Bound<'_, PyAny>, rows: &Bound<'_, PyAny>, target_level: u32) -> PyResult<Py<PyAny>> {
    Ok(inf.call_method1("pack_tokens", (rows, target_level))?.unbind())
}
#[pyfunction]
fn encode_prefill_input(inf: &Bound<'_, PyAny>, rows: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    Ok(inf.call_method1("encode_prefill_input", (rows,))?.unbind())
}
#[pyfunction]
fn decrypt_slots(inf: &Bound<'_, PyAny>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    Ok(inf.call_method1("decrypt_slots", (a,))?.unbind())
}
#[pyfunction]
fn decrypt_slots_complex(inf: &Bound<'_, PyAny>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    Ok(inf.call_method1("decrypt_slots_complex", (a,))?.unbind())
}
#[pyfunction]
fn decode_token_output(inf: &Bound<'_, PyAny>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    Ok(inf.call_method1("decode_token_output", (a,))?.unbind())
}
#[pyfunction]
fn decode_linear_output(inf: &Bound<'_, PyAny>, a: &Bound<'_, PyAny>, d_in: usize, d_out: usize) -> PyResult<Py<PyAny>> {
    Ok(inf.call_method1("decode_linear_output", (a, d_in, d_out))?.unbind())
}
#[pyfunction]
fn realize_pending_rescale(inf: &Bound<'_, PyAny>, a: &Bound<'_, PyAny>) -> PyResult<()> {
    inf.call_method1("realize", (a,))?;
    Ok(())
}
#[pyfunction]
fn export_limbs(inf: &Bound<'_, PyAny>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    Ok(inf.call_method1("export_limbs", (a,))?.unbind())
}
#[pyfunction]
fn import_limbs(inf: &Bound<'_, PyAny>, a: &Bound<'_, PyAny>, c0: &Bound<'_, PyAny>, c1: &Bound<'_, PyAny>, noise_factor: f64, noise_level: u8) -> PyResult<()> {
    inf.call_method1("import_limbs", (a, c0, c1, noise_factor, noise_level))?;
    Ok(())
}
#[pyfunction]
fn block_scope(b: i64) -> String {
    format!("block_{b}.")
}
#[pyfunction]
fn reset_graph_runtime(_inf: &Bound<'_, PyAny>) {}
#[pyfunction]
fn begin_subgraph_capture(_inf: &Bound<'_, PyAny>, _b: i64) -> bool {
    false
}
#[pyfunction]
fn end_subgraph_capture(_inf: &Bound<'_, PyAny>, _b: i64) {}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFhe>()?;
    m.add_class::<PyCt>()?;
    m.add_class::<PySize>()?;
    m.add_class::<NullScope>()?;
    m.add_class::<SuppressScope>()?;
    m.add_class::<StageHandle>()?;
    m.add_function(wrap_pyfunction!(encode_token_input, m)?)?;
    m.add_function(wrap_pyfunction!(pack_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(encode_prefill_input, m)?)?;
    m.add_function(wrap_pyfunction!(decrypt_slots, m)?)?;
    m.add_function(wrap_pyfunction!(decrypt_slots_complex, m)?)?;
    m.add_function(wrap_pyfunction!(decode_token_output, m)?)?;
    m.add_function(wrap_pyfunction!(decode_linear_output, m)?)?;
    m.add_function(wrap_pyfunction!(realize_pending_rescale, m)?)?;
    m.add_function(wrap_pyfunction!(export_limbs, m)?)?;
    m.add_function(wrap_pyfunction!(import_limbs, m)?)?;
    m.add_function(wrap_pyfunction!(block_scope, m)?)?;
    m.add_function(wrap_pyfunction!(reset_graph_runtime, m)?)?;
    m.add_function(wrap_pyfunction!(begin_subgraph_capture, m)?)?;
    m.add_function(wrap_pyfunction!(end_subgraph_capture, m)?)?;
    Ok(())
}
}
