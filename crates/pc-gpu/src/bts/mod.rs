//! Bootstrapping: seam `BtsRoute` (one implementation per route family, dispatched by `RouteKind`: `landing`,
//! `run`, `staged`, `mem`), the staged API `Staged` (FIDESlib's `BootstrapStages.cuh` 1:1, K-batched over
//! `&mut [Ct]`, per-call state in a `BtsState` value), the per-(slots, raise drop) precomputation `BtsPrecomp`, and
//! the sub-seams `Raise` (raise.rs), `LinearTransform` (cts.rs, shared by CtS and StC) and `ModEval` (evalmod.rs).
//!
//! A `Ct` is a batch of `k` ciphertexts sharing one `CtMeta`; every stage runs one set of launches per `Ct` (K
//! ciphertexts per launch, k <= `BtsSetup::k_max`). The slices of the API are independent batches of one call.
//! Stage order and precomputation: docs/ARCHITECTURE.md "pc-gpu bootstrapping".
use crate::context::{Context, DevStrategy, DevWord};
use crate::ct::Ct;
use crate::error::{shape, Error, Result};
use crate::executor::Executor;
use crate::ks::{KsKeyRef, KsPlans, KskSource};
use crate::mem::MemPlan;
use crate::ops::Ops;
use crate::poly::{CtBuf, Layout};
use pc_cuda::{Arena, Compute};
use pc_shadow::{CtMeta, Route, Shadow};

pub mod cts;
pub mod dense;
pub mod evalmod;
pub mod finish;
pub mod raise;
pub mod stc;
pub mod fold; // fold_bootstrap, tag_reduce, sparse auto routing
pub mod sparse; // sparse routes (512/256 SparseB), route set
pub mod sparse_b; // SparseB precomputation
pub mod spru; // s = 1 route by SPRU
pub mod stc_first; // StC-first route, re-levelled stage caches
pub mod stc_high; // StC outside a bootstrap
pub mod staged; // the staged call object, capture hooks
pub mod batch; // K-batching across ciphertexts, gather/scatter slots
pub mod sparse_classic; // classic multi-stage sparse route (slot counts SparseB does not take)

// ── dispatch ─────────────────────────────────────────────────────────────────────────────────────────────────────

/// Route families (seam `BtsRoute`): which implementation runs a `Route`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RouteKind {
    /// Full-slot route, shipped order (raise, CtS, EvalMod, StC, finish).
    Dense,
    /// Sparse slot route (`slots` < N/2; SparseB when `Route::sparse_b`).
    Sparse(u32),
    /// SPRU sparse-secret route (`Route::spru_h` > 0).
    Spru { slots: u32, h: u32 },
    /// Dense route in StC-first order (StC on the input side).
    StcFirst,
}

impl RouteKind {
    /// The family of `route` (`n` = ring dimension; slots 0 or N/2 = dense).
    pub fn of(route: &Route, n: usize) -> RouteKind {
        let dense = route.slots == 0 || route.slots as usize == n / 2;
        if route.spru_h > 0 {
            RouteKind::Spru { slots: route.slots, h: route.spru_h }
        } else if !dense {
            RouteKind::Sparse(route.slots)
        } else if route.stc_first {
            RouteKind::StcFirst
        } else {
            RouteKind::Dense
        }
    }
}

/// Device memory a route asks for, by `MemPlan` section (each carve sized with `Arena::bytes_for`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MemReq {
    pub sections: Vec<(&'static str, usize)>,
}

impl MemReq {
    pub fn add(&mut self, knob: &'static str, bytes: usize) {
        match self.sections.iter_mut().find(|s| s.0 == knob) {
            Some(s) => s.1 += bytes,
            None => self.sections.push((knob, bytes)),
        }
    }
    pub fn merge(&mut self, o: &MemReq) {
        for &(k, b) in &o.sections {
            self.add(k, b);
        }
    }
    pub fn total(&self) -> usize {
        self.sections.iter().map(|s| s.1).sum()
    }
    /// Reserve every section in `plan` (before `MemPlan::allocate`).
    pub fn reserve(&self, plan: &mut MemPlan) {
        for &(k, b) in &self.sections {
            plan.reserve(k, b);
        }
    }
}

// ── configuration ────────────────────────────────────────────────────────────────────────────────────────────────

/// What the bootstrap configuration fixes at setup (the `bts` part of the typed Config; names as perseus' env vars).
#[derive(Clone, Debug, PartialEq)]
pub struct BtsSetup {
    /// LEVEL_BUDGET {CtS, StC} (n32: {4, 3}).
    pub level_budget: [u32; 2],
    /// FIDESLIB_CTS_ALT_BUDGET (0 = off): the raise variants of drop >= 1 take a CtS of this level budget (one stage
    /// more than the shipped) raising one level higher (drop - 1): the same EvalMod input and landing, cheaper
    /// stages (FIDESlib `AddBootstrapRaiseVariant` with `CtS_alt`).
    pub cts_alt_budget: u32,
    /// OpenFHE dim1 {CtS, StC} (0 = OpenFHE's choice).
    pub dim1: [u32; 2],
    /// Raise drops one precomputation variant is built for (the route's base drop + FIDESLIB_BTS_RAISE_DROPS).
    pub raise_drops: Vec<u32>,
    /// CORRECTION_FACTOR default (a `Route` may carry its own) and the accepted window.
    pub cf_default: u32,
    pub cf_window: (u32, u32),
    /// FIDESLIB_BTS_SHIFT_T: the post-raise EvalMod constant applied exactly at the raise times 2^t; the CtS stage-0
    /// plaintexts carry 2^-t (BootstrapPrecomputation::cts0_t).
    pub cts0_t: u32,
    /// EvalMod: Chebyshev coefficients, double-angle iterations, K (`pc_core::bts_setup` tables by default).
    pub evalmod_coeffs: Vec<f64>,
    pub evalmod_da_its: u32,
    pub boot_k: f64,
    /// Build the real-payload variants (FIDESLIB_BTS_REAL: StC stage 0 for `Route::real`).
    pub real: bool,
    /// Ciphertexts per launch the work buffers are sized for (`cfg.bts.k_max`, <= `MemCfg::k_max`).
    pub k_max: usize,
    /// Where key switches inside the bootstrap (raise, giant / offset rotations, conjugations) read the keys' `a`
    /// from: Regen = in place (`ks_dot_key`, faster than Cached at K = 1 and 8 on n32).
    pub ks_source: KskSource,
    /// EvalMod relinearization key source (Regen: `a` regenerated in-kernel, the faster choice).
    pub em_relin: KskSource,
    /// LT kernel ciphertexts per thread group (`cts::LT_GROUP_AUTO` or 1/2/4).
    pub lt_group: i32,
    /// LT baby-key source (`LtStage::set_baby`): Regen = `cts_fused_regen` (seeded `a` in-kernel, half the key
    /// traffic of a stage), Cached = `cts_fused` reading resident `a`.
    pub lt_baby: KskSource,
    /// With `lt_baby` Regen: digits whose `a` is regenerated (the rest read the resident `a` when the keys have it;
    /// usize::MAX = all).
    pub lt_regen_digits: usize,
    /// Resident `a` tails from digit `lt_regen_digits` on for the CtS baby keys of packed key sets (`cts::KeyTails`,
    /// config `bts.lt_tails`); off: those stages regenerate every digit.
    pub lt_tails: bool,
    /// EvalMod: a Regen relinearization and the rescale after it as one exact conversion (`EvalModPlan::fuse_rescale`,
    /// config `bts.em_fuse_rescale`); off: relinearize, then rescale.
    pub em_fuse_rescale: bool,
}

impl BtsSetup {
    /// The shipped n32 configuration of the dense route (LEVEL_BUDGET 4:3, CF 6 in 2..20, SHIFT_T 12, EvalMod deg 14
    /// K 16 with 5 double angles, real payload on, raise drop 0).
    pub fn n32_shipped() -> BtsSetup {
        BtsSetup {
            level_budget: [4, 3],
            cts_alt_budget: 0,
            dim1: [0, 0],
            raise_drops: vec![0],
            cf_default: 6,
            cf_window: (2, 20),
            cts0_t: 12,
            evalmod_coeffs: pc_core::bts_setup::evalmod_coeffs().to_vec(),
            evalmod_da_its: pc_core::bts_setup::EVALMOD_DA_ITS,
            boot_k: pc_core::bts_setup::EVALMOD_K,
            real: true,
            k_max: 8,
            ks_source: KskSource::Regen,
            em_relin: KskSource::Regen,
            lt_group: 0,
            lt_baby: KskSource::Regen,
            lt_regen_digits: cts::LT_REGEN_DIGITS,
            lt_tails: true,
            em_fuse_rescale: false,
        }
    }

    /// The dense route on the n64 chain: the n32 shape with the chain's CF floor, deg = log2(q0 / 2^53)
    /// = 7 (perseus' n64 profile's CORRECTION_FACTOR 7). The u64 words read every key in place (`Regen` everywhere
    /// works at any word: the expander is the frozen KSKB lane).
    pub fn n64() -> BtsSetup {
        BtsSetup { cf_default: 7, cf_window: (7, 20), ..Self::n32_shipped() }
    }
}

// ── keys ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/// The keys a bootstrap reads. Implemented for `KeySet` and for oracle-imported key maps (pc-check).
/// CtS/StC stages read rotation keys through the fused LT kernel, which needs dense `b` rows by prime id and a
/// resident `a` (`KsKeyRef::a_ptr`); the other switches accept any `KsKeyRef` (staged when packed / banded).
pub trait BtsKeyRing {
    /// Rotation by `step` slots (left; any sign, reduced mod N/2).
    fn rot(&self, step: i64) -> Result<&dyn KsKeyRef>;
    fn conj(&self) -> Result<&dyn KsKeyRef>;
    fn relin(&self) -> Result<&dyn KsKeyRef>;
    /// Encapsulation: dense secret -> sparse secret (applied at the composite bottom).
    fn dense_to_sparse(&self) -> Result<&dyn KsKeyRef>;
    /// Encapsulation: sparse secret -> dense secret (applied at the raised level).
    fn sparse_to_dense(&self) -> Result<&dyn KsKeyRef>;
}

impl<W: DevWord> BtsKeyRing for crate::keys::KeySet<'_, W> {
    fn rot(&self, step: i64) -> Result<&dyn KsKeyRef> {
        Ok(crate::keys::KeySet::rot(self, step)?)
    }
    fn conj(&self) -> Result<&dyn KsKeyRef> {
        Ok(crate::keys::KeySet::conj(self)?)
    }
    fn relin(&self) -> Result<&dyn KsKeyRef> {
        Ok(crate::keys::KeySet::relin(self)?)
    }
    fn dense_to_sparse(&self) -> Result<&dyn KsKeyRef> {
        Ok(&self.bts()?.enc_dense_to_sparse)
    }
    fn sparse_to_dense(&self) -> Result<&dyn KsKeyRef> {
        Ok(&self.bts()?.enc_sparse_to_dense)
    }
}

// ── per-call state and context ───────────────────────────────────────────────────────────────────────────────────

/// One bootstrap call's state (FIDESlib `BtsState`, from `btsBegin`): everything the stages read besides the
/// ciphertexts, computed once from the route, the input metadata and the precomputation.
#[derive(Clone, Debug, PartialEq)]
pub struct BtsState {
    pub route: Route,
    /// Route slots (N/2 dense) and the input's slots (restored by finish).
    pub slots: u32,
    pub old_slots: u32,
    /// The input already carries the raise adjust (planned prescale / StC-first mode 2): ModRaise's prescaled path.
    pub prescaled: bool,
    /// Correction factor of the call and the raise correction = CF - deg (deg = log2(q0 / 2^scale_bits)).
    pub cf: u32,
    pub correction: u32,
    /// 2^correction: the integer the output is multiplied by (finish / StC-first output).
    pub cor_factor: u64,
    /// EvalMod's integer post-factor (1 on composite chains).
    pub scalar: u64,
    /// pre / (K N): the post-raise constant; `baked` = the one folded into the CtS stage-0 plaintexts (0 = none).
    pub const_eval_mult: f64,
    pub baked: f64,
    /// The constant is applied exactly at the raise (small-integer scaling) because it matches `baked`.
    pub exact_const: bool,
    /// The CtS stage 0 runs on the shifted (baked) plaintexts (set by mod_raise).
    pub shifted_flow: bool,
    pub sparse_encaps: bool,
    pub sparse_b: bool,
    pub stc_first: bool,
    pub stc_folded: bool,
    /// Real payload on the dense route: one EvalMod chain, StC stage 0 real variant, one more correction bit.
    pub real: bool,
    /// Linear-transform stage counts of this call.
    pub n_cts: usize,
    pub n_stc: usize,
    /// Index of the precomputation variant (raise drop) this call runs on.
    pub variant: usize,
    pub raise_drop: u32,
    /// The input's metadata at begin (iterations and prescale restores read it).
    pub in_meta: CtMeta,
}

/// The objects one bootstrap call borrows (setup-built, owned by the caller; nothing global).
pub struct BtsCx<'c, 'a, W: DevWord, M: DevStrategy<W>> {
    pub ctx: &'c Context<'a, W, M>,
    pub ex: &'c mut Executor<'a, W>,
    pub sh: &'c Shadow,
    pub ks: &'c KsPlans<'a>,
    pub ops: &'c mut Ops<'a, W>,
    pub work: &'c mut BtsWork<'a, W>,
}

/// Bootstrap temporaries for `k_max` ciphertexts per launch (MemPlan section "bts.work"; carved at setup).
/// `lt_*`: the linear-transform stage buffers (`k_max * g_max` giant-step results, extended basis); `em`: one
/// buffer per EvalMod program value (`em_kk` ciphertexts each: K finishes x the Re/Im chains when they share
/// launches); `spare`: whole-ciphertext temporaries (iterations, real-payload conjugates, stage copies).
pub struct BtsWork<'a, W: DevWord> {
    pub shape: BtsWorkShape,
    pub lt_res: CtBuf<'a, Compute, W>,
    pub lt_tmp: CtBuf<'a, Compute, W>,
    pub lt_rot: CtBuf<'a, Compute, W>,
    pub em: Vec<CtBuf<'a, Compute, W>>,
    pub spare: Vec<CtBuf<'a, Compute, W>>,
}

/// Sizes of a `BtsWork`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BtsWorkShape {
    pub k_max: usize,
    /// Largest giant-step count of an LT stage.
    pub g_max: usize,
    /// EvalMod program values and ciphertexts per value buffer.
    pub em_slots: usize,
    pub em_kk: usize,
    /// Spare k_max-ciphertext buffers.
    pub spare: usize,
}

impl BtsWorkShape {
    pub fn max(self, o: BtsWorkShape) -> BtsWorkShape {
        BtsWorkShape {
            k_max: self.k_max.max(o.k_max),
            g_max: self.g_max.max(o.g_max),
            em_slots: self.em_slots.max(o.em_slots),
            em_kk: self.em_kk.max(o.em_kk),
            spare: self.spare.max(o.spare),
        }
    }
}

impl<'a, W: DevWord> BtsWork<'a, W> {
    pub fn bytes(lay: &Layout, s: BtsWorkShape) -> usize {
        let ct = |k: usize| Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(lay, k));
        2 * ct(s.k_max * s.g_max) + ct(s.k_max) + s.em_slots * ct(s.em_kk) + s.spare * ct(s.k_max)
    }
    pub fn new(arena: &'a Arena, lay: Layout, s: BtsWorkShape) -> Result<Self> {
        Ok(BtsWork {
            shape: s,
            lt_res: CtBuf::new(arena, "bts.work", lay, s.k_max * s.g_max)?,
            lt_tmp: CtBuf::new(arena, "bts.work", lay, s.k_max * s.g_max)?,
            lt_rot: CtBuf::new(arena, "bts.work", lay, s.k_max)?,
            em: (0..s.em_slots).map(|_| CtBuf::new(arena, "bts.work", lay, s.em_kk)).collect::<Result<_>>()?,
            spare: (0..s.spare).map(|_| CtBuf::new(arena, "bts.work", lay, s.k_max)).collect::<Result<_>>()?,
        })
    }
}

// ── the precomputation ───────────────────────────────────────────────────────────────────────────────────────────

/// One route's precomputation at one raise drop (FIDESlib `BootstrapPrecomputation`, one entry of its
/// `raise_variants`): device-resident from setup, read-only during calls.
pub struct BtsPrecomp<'a, W: DevWord> {
    pub slots: u32,
    pub raise_drop: u32,
    /// Raise target top limb (`L - d * raise_drop`) and its tables (raise.rs).
    pub raise: raise::RaisePlan<'a, W>,
    /// CtS stages in execution order (FIDESlib CtS; OpenFHE s = budget-1 .. 0).
    pub cts: Vec<cts::LtStage<'a, W>>,
    /// StC stages in execution order (shipped order; empty for StC-first-only variants).
    pub stc: Vec<cts::LtStage<'a, W>>,
    /// StC stage 0 for a real payload (`stcRealA0`), same geometry as `stc[0]`.
    pub stc_real0: Option<cts::LtStage<'a, W>>,
    /// EvalMod (complex: Re/Im chains; real: one chain), built from the CtS output metadata.
    pub em: evalmod::EvalModPlan<'a, W>,
    pub em_real: Option<evalmod::EvalModPlan<'a, W>>,
    /// The post-raise constant folded into CtS stage 0 (`cts0_const`) and its shift t (`cts0_t`).
    pub cts0_const: f64,
    pub cts0_t: u32,
    /// EvalMod K of the route.
    pub boot_k: f64,
}

impl<W: DevWord> BtsPrecomp<'_, W> {
    /// Device bytes of the tables held (setup report).
    pub fn bytes(&self) -> usize {
        self.raise.bytes()
            + self.cts.iter().chain(&self.stc).chain(self.stc_real0.iter()).map(|s| s.bytes()).sum::<usize>()
            + self.em.bytes()
            + self.em_real.as_ref().map_or(0, |e| e.bytes())
    }
}

// ── the seams ────────────────────────────────────────────────────────────────────────────────────────────────────

/// StC outside a bootstrap (`btsStcHigh`): output at `out_level`, stage `s` skipped when bit s of `skip_mask` is set,
/// the last kept stage scaled by `factor` (times `nf_ref / NoiseFactor` when `nf_ref > 0`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StcHigh {
    pub slots: u32,
    pub out_level: u32,
    pub factor: f64,
    pub skip_mask: u32,
    pub nf_ref: f64,
}

/// FIDESlib's staged bootstrap, K-batched. Each stage transforms every `Ct` of `cts` in place (all share the
/// metadata of `begin`'s input). A route without a stage returns `Error::Shape` naming it.
pub trait Staged<'a, W: DevWord, M: DevStrategy<W>> {
    /// `btsBegin`: the call's state (CF from `route.cf` or the setup default; `prescaled` = ModRaise's prescaled path).
    fn begin(&self, cx: &BtsCx<'_, 'a, W, M>, cts: &[Ct<'a, W>], route: &Route, prescaled: bool) -> Result<BtsState>;
    /// StC on the input side (StC-first order; no-op otherwise).
    fn stc_first_input(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>]) -> Result<()>;
    /// Raise to the variant's top (encapsulation switches, exact post-raise constant).
    fn mod_raise(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &mut BtsState, cts: &mut [Ct<'a, W>]) -> Result<()>;
    /// Sparse-slot accumulation after the raise (no-op on the dense route).
    fn fold(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>]) -> Result<()>;
    fn cts_stage(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>], k: usize) -> Result<()>;
    fn eval_mod(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>]) -> Result<()>;
    /// StC-first: the integer post-factor (corFactor << stc_first_deg) and the slot restore.
    fn stc_first_output(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>]) -> Result<()>;
    /// Rescale / drop to the StC stage-0 plaintext level.
    fn stc_enter(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>]) -> Result<()>;
    fn stc_stage(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>], k: usize) -> Result<()>;
    /// Sparse rotate-add, real-payload conjugate-add, the correction factor, the slot restore.
    fn finish(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>]) -> Result<()>;
    /// StC outside a bootstrap.
    fn stc_high(&self, cx: &mut BtsCx<'_, 'a, W, M>, cts: &mut [Ct<'a, W>], spec: &StcHigh) -> Result<()>;
}

/// `BootstrapStaged`: the stage sequence of one refresh (the reference orchestration of the staged API).
pub fn run_staged<'a, W: DevWord, M: DevStrategy<W>>(
    s: &dyn Staged<'a, W, M>,
    cx: &mut BtsCx<'_, 'a, W, M>,
    cts: &mut [Ct<'a, W>],
    route: &Route,
    prescaled: bool,
) -> Result<BtsState> {
    let mut st = s.begin(cx, cts, route, prescaled)?;
    s.stc_first_input(cx, &st, cts)?;
    s.mod_raise(cx, &mut st, cts)?;
    s.fold(cx, &st, cts)?;
    for k in 0..st.n_cts {
        s.cts_stage(cx, &st, cts, k)?;
    }
    s.eval_mod(cx, &st, cts)?;
    if st.stc_first {
        s.stc_first_output(cx, &st, cts)?;
        return Ok(st);
    }
    s.stc_enter(cx, &st, cts)?;
    for k in 0..st.n_stc {
        s.stc_stage(cx, &st, cts, k)?;
    }
    s.finish(cx, &st, cts)?;
    Ok(st)
}

/// Seam `BtsRoute`: a bootstrap route implementation.
pub trait BtsRoute<'a, W: DevWord, M: DevStrategy<W>> {
    fn kind(&self) -> RouteKind;
    /// Metadata of the refresh output (raw, as `run` leaves it: degree 2 before the planner's realize) of an input
    /// with metadata `meta` on `route`.
    fn landing(&self, sh: &Shadow, meta: &CtMeta, route: &Route) -> Result<CtMeta>;
    /// Refresh every batch of `cts` in place (`route.iterations` > 1: the residual iterations of perseus
    /// `eval_bootstrap_iter`).
    fn run(&self, cx: &mut BtsCx<'_, 'a, W, M>, cts: &mut [Ct<'a, W>], route: &Route) -> Result<()>;
    fn staged(&self) -> Option<&dyn Staged<'a, W, M>>;
    /// Device bytes this route holds (precomputation tables) and the work shape it needs.
    fn mem(&self) -> (MemReq, BtsWorkShape);
}

/// Error for a stage a route does not have.
pub fn no_stage(route: &'static str, stage: &str) -> Error {
    shape("bts stage", format!("route {route} has no stage {stage}"))
}

/// All batches of a call share one metadata (the stages run the same recipe on each).
pub fn common_meta<W: DevWord>(cts: &[Ct<'_, W>]) -> Result<CtMeta> {
    let first = cts.first().ok_or_else(|| shape("bootstrap", "no ciphertexts"))?;
    if cts.iter().any(|c| c.meta.level != first.meta.level || c.meta.deg != first.meta.deg || c.meta.nf != first.meta.nf) {
        return Err(shape("bootstrap", "batches of one call must share level, degree and scale"));
    }
    Ok(first.meta)
}
