//! The classic multi-stage sparse route: an `s`-slot bootstrap (OpenFHE sparse packing, collapsed FFT over a 2s-slot
//! view) for the slot counts SparseB does not take (`Route::sparse_b` off), e.g. 8192:
//!
//!   raise (exact post-raise constant pre / (K N)) -> fold Accumulate(4, s, N/2s) -> CtS x budget (fused
//!   ModDown-rescale) -> EvalMod (x + conj x, K = 24 degree-20 series, 5 double angles) -> rescale -> StC x budget
//!   (the last pays the LT-friendly offset, degree 2) -> rotate(s)-add -> x 2^correction.
//!
//! `ClassicPlan::new` is the host side (factors, stage levels, encodings, metadata, `mem`, `rot_steps`);
//! `ClassicPlan::build` carves the device tables. One variant per raise drop. Plaintexts are encoded at their
//! stage level from the slot values (one rounding).
use super::cts::{diag_coeffs, fold_periodic_giants, stage_width, LtEnd, LtGeom, LtPts, LtStage, PtSource};
use super::dense::{shadow_of, LtHost};
use super::evalmod::{em_work, EmMode, EvalModPlan};
use super::finish::Pow2;
use super::raise::{self, EncapsKeys, RaisePlan};
use super::sparse::acc_round;
use super::sparse_b::AccRounds;
use super::{common_meta, no_stage, run_staged, BtsCx, BtsKeyRing, BtsPrecomp, BtsRoute, BtsState, BtsWorkShape, MemReq, RouteKind, Staged, StcHigh};
use crate::context::{Context, DevStrategy, DevWord};
use crate::ct::Ct;
use crate::error::{shape, Error, Result};
use crate::ks::{KskSource, View};
use crate::poly::Layout;
use crate::rotate::KsCfg;
use pc_core::bts_setup::cts_stc_factors;
use pc_core::params::Chain;
use pc_core::scale::ScaleTables;
use pc_cuda::{Arena, Compute, Stream};
use pc_shadow::evalmod::PsPlan;
use pc_shadow::fides::FMeta;
use pc_shadow::route::{cts_stage_levels, stc_stage_levels};
use pc_shadow::{tag, CtMeta, PtMeta, Route, Shadow};
use std::collections::BTreeSet;
use std::ffi::c_void;
use std::marker::PhantomData;

/// Knobs of a classic sparse route (the `bts` part of the Config).
#[derive(Clone, Debug, PartialEq)]
pub struct ClassicCfg {
    pub slots: u32,
    /// SPARSE_LEVEL_BUDGET {CtS, StC} (n32 runtime: {3, 3}).
    pub level_budget: [u32; 2],
    /// BTS_DIM1 {CtS, StC} (0 = OpenFHE's choice).
    pub dim1: [u32; 2],
    /// Raise drops one variant is built for (FIDESLIB_BTS_RAISE_DROP for this s, + the plan's per-site drops).
    pub raise_drops: Vec<u32>,
    /// FIDESLIB_BTS_SHIFT_T: CtS stage 0 carries 2^-t, the raise's exact constant 2^t.
    pub cts0_t: u32,
    /// EvalMod K and series (K = 24, degree 20, 5 double angles; the dense route's K = 16 degree-14 series also
    /// fits).
    pub boot_k: f64,
    pub evalmod_coeffs: Vec<f64>,
    pub evalmod_da_its: u32,
    /// FIDESLIB_ACCUM_BSTEP (rotations per accumulate round + 1).
    pub acc_bstep: usize,
    pub cf_default: u32,
    pub cf_window: (u32, u32),
    pub k_max: usize,
    pub ks_source: KskSource,
    pub em_relin: KskSource,
    pub lt_group: i32,
    pub lt_baby: KskSource,
    pub lt_regen_digits: usize,
}

impl ClassicCfg {
    /// The n32 runtime's classic route at `slots` (SPARSE_LEVEL_BUDGET 3,3, SHIFT_T 12, K = 24, CF 6 in 2..20).
    pub fn n32(slots: u32, raise_drops: &[u32]) -> ClassicCfg {
        ClassicCfg {
            slots,
            level_budget: [3, 3],
            dim1: [0, 0],
            raise_drops: if raise_drops.is_empty() { vec![0] } else { raise_drops.to_vec() },
            cts0_t: 12,
            boot_k: pc_core::bts_setup::SPARSE_B_K,
            evalmod_coeffs: pc_core::bts_setup::SPARSE_B_CHEB_K24.to_vec(),
            evalmod_da_its: pc_core::bts_setup::SPARSE_B_DA_ITS,
            acc_bstep: 4,
            cf_default: 6,
            cf_window: (2, 20),
            k_max: 1,
            ks_source: KskSource::Regen,
            em_relin: KskSource::Regen,
            lt_group: 0,
            lt_baby: KskSource::Regen,
            lt_regen_digits: super::cts::LT_REGEN_DIGITS,
        }
    }
}

/// One raise-drop variant on the host, with the metadata after every stage (FIDESlib coordinates).
pub struct ClassicVariantHost {
    pub drop: u32,
    pub raise_top: usize,
    pub exact: f64,
    pub cts: Vec<LtHost>,
    pub stc: Vec<LtHost>,
    pub em_in: FMeta,
    pub em_out: FMeta,
    pub em_slots: usize,
    /// the bootstrap output (degree 2, before the rotate-add and the integer post-factor, which keep it)
    pub out: FMeta,
}

/// The classic sparse route's host plan.
pub struct ClassicPlan {
    pub cfg: ClassicCfg,
    pub chain: Chain,
    pub variants: Vec<ClassicVariantHost>,
    pub ps: PsPlan,
    /// log2(q0 / 2^scale_bits)
    pub deg: u32,
    /// fold rounds: Accumulate(acc_bstep, s, N/2s)
    pub fold: Vec<Vec<i64>>,
}

fn to_f(shape: pc_shadow::ChainShape, m: &CtMeta) -> FMeta {
    FMeta { top: (shape.n_q - 1 - m.level) as i32, nl: m.deg as i32, nf: m.nf }
}
fn from_f(shape: pc_shadow::ChainShape, f: FMeta) -> CtMeta {
    CtMeta { level: shape.n_q - 1 - f.top as u32, deg: f.nl as u8, nf: f.nf, ..CtMeta::new(0, 1) }
}

impl ClassicPlan {
    pub fn new(chain: &Chain, cfg: &ClassicCfg) -> Result<ClassicPlan> {
        let n = chain.n;
        let s = cfg.slots;
        if s < 2 || !s.is_power_of_two() || s as usize >= n / 2 {
            return Err(shape("classic sparse route", format!("slots {s}: a power of two in [2, N/2) (N = {n})")));
        }
        let st = ScaleTables::precompute(chain)?;
        let sh = shadow_of(&st);
        let shp = sh.cfg.chain;
        let d = chain.level_unit;
        let ps = PsPlan::from_coeffs(&cfg.evalmod_coeffs)?;
        let mut qd = chain.q[0] as f64;
        for j in 1..d {
            qd *= chain.q[j] as f64;
        }
        let deg = (qd / (chain.spec.scaling_bits as f64).exp2()).log2().round() as u32;
        let inner_end = if d == 2 { LtEnd::FusedRescale } else { LtEnd::ModDown };
        // scaleEnc = 1 (ternary secret; the route's K rides the raise constant), LT-friendly layout over 2s slots
        let f = cts_stc_factors(n, s, cfg.level_budget, cfg.dim1, 1.0, super::stc::scale_dec(&st))?;
        let lay = f.fides_layout();
        let mut variants = Vec::new();
        for &drop in &cfg.raise_drops {
            let raise_top = chain.q.len() - 1 - d * drop as usize;
            let fdrop = st.sf((chain.q.len() - 1) as i32)? / st.sf(raise_top as i32)?;
            let exact = (st.sf(raise_top as i32)? / qd) * (1.0 / (cfg.boot_k * n as f64));
            let raised = FMeta { top: raise_top as i32, nl: 1, nf: st.sf(raise_top as i32)? * (cfg.cts0_t as f64).exp2() };
            let mut m = from_f(shp, raised);
            // the fold (rotations) keeps the raised metadata
            let levels = cts_stage_levels(&st, lay.cts.len() as u32, drop, Some(cfg.cts0_t))?;
            let mut cts = Vec::new();
            for (e, fs) in lay.cts.iter().enumerate() {
                let lv = levels[e];
                if lv.top as usize != (shp.n_q - 1 - m.level) as usize || m.deg != 1 {
                    return Err(shape("classic plan", format!("CtS stage {e} plaintexts at top {} but the input is {m:?}", lv.top)));
                }
                let (geom, coeffs) = fold_periodic_giants(n, LtGeom::from_fides(fs), diag_coeffs(chain, &fs.pts, lv.scale)?);
                cts.push(LtHost { geom, top: lv.top as usize, end: inner_end, nf: lv.scale, width: stage_width(n, &coeffs), coeffs });
                m = sh.mult_pt(&m, &PtMeta { level: lv.level, deg: 1, nf: lv.scale, tag: Default::default() })?.out;
                m = sh.rescale(&m)?.out;
            }
            let em_in = to_f(shp, &m);
            let (_, em_out, em_slots) = EvalModPlan::<u32>::program(&st, em_in, &ps, cfg.evalmod_da_its)?;
            let mo = from_f(shp, em_out);
            let slev = stc_stage_levels(&st, mo.level, mo.deg, lay.stc.len() as u32)?;
            let mut m = if mo.deg == 2 { sh.rescale(&mo)?.out } else { mo };
            let mut stc = Vec::new();
            let nst = lay.stc.len();
            for (e, fs) in lay.stc.iter().enumerate() {
                let lv = slev[e];
                let top = shp.n_q - 1 - m.level;
                if lv.top > top {
                    return Err(shape("classic plan", format!("StC stage {e} plaintexts at top {} above the input top {top}", lv.top)));
                }
                m.level = lv.level;
                let end = if e + 1 < nst { inner_end } else { LtEnd::ModDown };
                let geom = LtGeom::from_fides(fs);
                let coeffs = diag_coeffs(chain, &fs.pts, if e + 1 == nst { lv.scale * fdrop } else { lv.scale })?;
                stc.push(LtHost { geom, top: lv.top as usize, end, nf: lv.scale, width: stage_width(n, &coeffs), coeffs });
                m = sh.mult_pt(&m, &PtMeta { level: lv.level, deg: 1, nf: lv.scale, tag: Default::default() })?.out;
                if e + 1 < nst && geom.offset == 0 {
                    m = sh.rescale(&m)?.out;
                }
            }
            variants.push(ClassicVariantHost { drop, raise_top, exact, cts, stc, em_in, em_out, em_slots, out: to_f(shp, &m) });
        }
        let fold = super::spru::accumulate_rounds(cfg.acc_bstep, s as usize, n / 2 / s as usize);
        Ok(ClassicPlan { cfg: cfg.clone(), chain: chain.clone(), variants, ps, deg, fold })
    }

    /// Every rotation step the route reads (keys to generate), reduced to [0, N/2), without 0.
    pub fn rot_steps(&self) -> Vec<i64> {
        let nh = (self.chain.n / 2) as i64;
        let mut v = BTreeSet::new();
        for var in &self.variants {
            for h in var.cts.iter().chain(&var.stc) {
                let g = &h.geom;
                for i in 1..g.b {
                    v.insert(((i * g.stride) as i64) % nh);
                }
                if g.g > 1 {
                    v.insert(((g.b * g.stride) as i64) % nh);
                }
                v.insert(g.offset.rem_euclid(nh));
            }
        }
        v.extend(self.fold.iter().flatten().map(|x| x.rem_euclid(nh)));
        v.insert(self.cfg.slots as i64 % nh);
        v.remove(&0);
        v.into_iter().collect()
    }

    fn lt_bytes<W: DevWord>(&self, lay: &Layout, h: &LtHost) -> Result<usize> {
        Ok(LtPts::<W>::coeffs_bytes(h.coeffs.len(), h.top + 1 + lay.kp, h.width, lay.n) + LtStage::<W>::plan_bytes(lay, &self.chain.primes(), &h.geom, h.top, h.end, self.cfg.k_max)?)
    }

    /// Device bytes of every table `build` carves (MemPlan section "bts.sparse") and the work shape.
    pub fn mem<W: DevWord>(&self) -> Result<(MemReq, BtsWorkShape)> {
        let lay = Layout::from_chain(&self.chain);
        let st = ScaleTables::precompute(&self.chain)?;
        let mut b = Pow2::<W>::bytes(lay.rows()) + EncapsKeys::bytes() + AccRounds::bytes(&self.fold) + AccRounds::bytes(&[vec![self.cfg.slots as i64]]);
        let mut em_slots = 0;
        let mut g_max = 1;
        for v in &self.variants {
            b += RaisePlan::<W>::table_bytes();
            for h in v.cts.iter().chain(&v.stc) {
                b += self.lt_bytes::<W>(&lay, h)?;
                g_max = g_max.max(h.geom.g);
            }
            b += EvalModPlan::<W>::table_bytes(&st, &lay, v.em_in, &self.ps, self.cfg.evalmod_da_its, self.cfg.k_max)?;
            em_slots = em_slots.max(v.em_slots);
        }
        let (n, kk) = em_work(em_slots, self.cfg.k_max, EmMode::Real);
        let mut r = MemReq::default();
        r.add("bts.sparse", b);
        Ok((r, BtsWorkShape { k_max: self.cfg.k_max, g_max: g_max.max(2), em_slots: n, em_kk: kk, spare: 0 }))
    }

    /// Device tables (after `MemPlan::allocate`, before `Arena::freeze`).
    pub fn build<'a, W: DevWord, M: DevStrategy<W>>(
        &self,
        arena: &'a Arena,
        s: &Stream<Compute>,
        ctx: &Context<'_, W, M>,
        keys: &dyn BtsKeyRing,
        stage_ptrs: Option<(*const c_void, *const c_void)>,
    ) -> Result<SparseClassicRoute<'a, W, M>> {
        let align = Arena::bytes_for::<u8>(1);
        let used0 = arena.used_bytes().div_ceil(align) * align;
        let n = ctx.lay.n;
        let k_max = self.cfg.k_max;
        let lt = |h: &LtHost| -> Result<LtStage<'a, W>> {
            let pts = LtPts::new(arena, s, ctx, h.top, PtSource::Coeffs(&h.coeffs))?;
            let mut st = LtStage::new(arena, s, ctx, h.geom, h.top, h.end, h.nf, pts, keys, stage_ptrs, k_max)?;
            st.group = self.cfg.lt_group;
            st.set_baby(self.cfg.lt_baby)?;
            st.set_regen_digits(self.cfg.lt_regen_digits)?;
            Ok(st)
        };
        let mut variants = Vec::new();
        for v in &self.variants {
            variants.push(BtsPrecomp {
                slots: self.cfg.slots,
                raise_drop: v.drop,
                raise: RaisePlan::new(arena, s, ctx, v.raise_top, v.exact, self.cfg.cts0_t)?,
                cts: v.cts.iter().map(lt).collect::<Result<_>>()?,
                stc: v.stc.iter().map(lt).collect::<Result<_>>()?,
                stc_real0: None,
                em: EvalModPlan::new(arena, s, ctx, EmMode::Real, v.em_in, &self.ps, self.cfg.evalmod_da_its, keys.conj()?, keys.relin()?, stage_ptrs, k_max, self.cfg.em_relin)?,
                em_real: None,
                cts0_const: v.exact,
                cts0_t: self.cfg.cts0_t,
                boot_k: self.cfg.boot_k,
            });
        }
        let fold = AccRounds::new(arena, s, n, keys, stage_ptrs, self.fold.clone())?;
        let half = AccRounds::new(arena, s, n, keys, stage_ptrs, vec![vec![self.cfg.slots as i64]])?;
        let encaps = EncapsKeys::new(arena, s, n, keys.dense_to_sparse()?, keys.sparse_to_dense()?, stage_ptrs)?;
        let pow2 = Pow2::new(arena, s, ctx)?;
        s.sync()?;
        let carved = arena.used_bytes().div_ceil(align) * align - used0;
        let (mem, work_shape) = self.mem::<W>()?;
        Ok(SparseClassicRoute {
            cfg: self.cfg.clone(),
            ring_n: n,
            variants,
            outs: self.variants.iter().map(|v| v.out).collect(),
            fold,
            half,
            encaps,
            pow2,
            deg: self.deg,
            mem,
            work_shape,
            carved,
            _m: PhantomData,
        })
    }
}

/// The classic sparse route's device side.
pub struct SparseClassicRoute<'a, W: DevWord, M: DevStrategy<W>> {
    pub cfg: ClassicCfg,
    pub ring_n: usize,
    pub variants: Vec<BtsPrecomp<'a, W>>,
    outs: Vec<FMeta>,
    fold: AccRounds<'a>,
    /// rotate(s)-add closing the StC (one key)
    half: AccRounds<'a>,
    encaps: EncapsKeys<'a>,
    pow2: Pow2<'a, W>,
    pub deg: u32,
    mem: MemReq,
    work_shape: BtsWorkShape,
    /// bytes `build` carved (the MemPlan check compares it with `mem()`)
    pub carved: usize,
    _m: PhantomData<M>,
}

fn total_slots(meta: &CtMeta, ring_n: usize) -> i64 {
    if meta.tag.slots > 0 {
        meta.tag.slots
    } else if meta.slots > 0 {
        meta.slots as i64
    } else {
        (ring_n / 2) as i64
    }
}

impl<'a, W: DevWord, M: DevStrategy<W>> SparseClassicRoute<'a, W, M> {
    fn variant(&self, drop: u32) -> Result<usize> {
        self.variants.iter().position(|v| v.raise_drop == drop).ok_or_else(|| Error::Mem {
            knob: "bts.raise_drops",
            value: drop as usize,
            msg: format!("classic route {} has no raise-drop {drop} variant (built: {:?})", self.cfg.slots, self.variants.iter().map(|v| v.raise_drop).collect::<Vec<_>>()),
        })
    }

    fn acc(&self, cx: &mut BtsCx<'_, 'a, W, M>, rounds: &AccRounds<'_>, x: &mut Ct<'a, W>) -> Result<()> {
        let kc = KsCfg { plans: cx.ks, source: self.cfg.ks_source };
        for (steps, g) in &rounds.rounds {
            acc_round(cx.ctx, cx.ex, kc, cx.sh, g, x, steps)?;
        }
        Ok(())
    }

    fn run_lt(&self, cx: &mut BtsCx<'_, 'a, W, M>, stage: &LtStage<'a, W>, cts: &mut [Ct<'a, W>]) -> Result<()> {
        let ctx = cx.ctx;
        for x in cts.iter_mut() {
            if x.meta.deg == 2 {
                cx.ops.rescale(ctx, cx.ex, cx.sh, std::slice::from_mut(x))?;
            }
            let top = x.top(ctx.shape)?;
            if top != stage.top {
                return Err(shape("classic LT stage", format!("input top {top}, stage plaintexts at {}", stage.top)));
            }
            let k = x.k();
            stage.run(cx, &mut x.buf, k, self.cfg.ks_source)?;
            let pt = PtMeta { level: ctx.shape.n_q - 1 - stage.top as u32, deg: 1, nf: stage.pt_nf, tag: Default::default() };
            let m = cx.sh.mult_pt(&x.meta, &pt)?.out;
            let (_, od) = stage.out_top_deg(ctx.chain.level_unit);
            x.meta = if od == 1 { cx.sh.rescale(&m)?.out } else { m };
        }
        Ok(())
    }

    /// The MemPlan prediction equals the carve.
    pub fn check_mem(&self) -> Result<()> {
        let want = self.mem.total();
        if want != self.carved {
            return Err(Error::Mem { knob: "bts.sparse", value: self.carved, msg: format!("classic route {} carved {} bytes, MemPlan predicted {want}", self.cfg.slots, self.carved) });
        }
        Ok(())
    }
}

impl<'a, W: DevWord, M: DevStrategy<W>> Staged<'a, W, M> for SparseClassicRoute<'a, W, M> {
    fn begin(&self, _cx: &BtsCx<'_, 'a, W, M>, cts: &[Ct<'a, W>], route: &Route, prescaled: bool) -> Result<BtsState> {
        let in_meta = common_meta(cts)?;
        if route.slots != self.cfg.slots || route.sparse_b || route.spru_h > 0 {
            return Err(shape("classic sparse route", format!("route {route:?} on the classic {}-slot route", self.cfg.slots)));
        }
        if let Some(x) = cts.iter().find(|x| x.k() > self.cfg.k_max) {
            return Err(shape("classic sparse route", format!("batch of {} cts above k_max {}", x.k(), self.cfg.k_max)));
        }
        let variant = self.variant(route.raise_drop)?;
        let cf = route.cf.unwrap_or(self.cfg.cf_default);
        if cf < self.cfg.cf_window.0 || cf > self.cfg.cf_window.1 || cf < self.deg {
            return Err(shape("bootstrap cf", format!("correction factor {cf} outside [{}, {}] (deg {})", self.cfg.cf_window.0, self.cfg.cf_window.1, self.deg)));
        }
        let correction = cf - self.deg;
        let pre = &self.variants[variant];
        Ok(BtsState {
            route: *route,
            slots: self.cfg.slots,
            old_slots: in_meta.slots,
            prescaled,
            cf,
            correction,
            cor_factor: 1u64 << correction,
            scalar: 1,
            const_eval_mult: pre.cts0_const,
            baked: pre.cts0_const,
            exact_const: true,
            shifted_flow: false,
            sparse_encaps: true,
            sparse_b: false,
            stc_first: false,
            stc_folded: false,
            // the real-payload variant exists on the dense route only
            real: false,
            n_cts: pre.cts.len(),
            n_stc: pre.stc.len(),
            variant,
            raise_drop: route.raise_drop,
            in_meta,
        })
    }
    fn stc_first_input(&self, _cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, _cts: &mut [Ct<'a, W>]) -> Result<()> {
        if st.stc_first {
            return Err(no_stage("sparse_classic", "stc_first_input"));
        }
        Ok(())
    }
    fn mod_raise(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &mut BtsState, cts: &mut [Ct<'a, W>]) -> Result<()> {
        let pre = &self.variants[st.variant];
        raise::raise(cx, &pre.raise, Some(&self.encaps), self.cfg.ks_source, cts, if st.prescaled { None } else { Some(st.correction) })?;
        st.shifted_flow = true;
        Ok(())
    }
    /// `Accumulate(4, s, N/2s)` after the raise (the exact scaling commutes with it).
    fn fold(&self, cx: &mut BtsCx<'_, 'a, W, M>, _st: &BtsState, cts: &mut [Ct<'a, W>]) -> Result<()> {
        for x in cts.iter_mut() {
            self.acc(cx, &self.fold, x)?;
            if x.meta.deg == 2 {
                cx.ops.rescale(cx.ctx, cx.ex, cx.sh, std::slice::from_mut(x))?;
            }
        }
        Ok(())
    }
    fn cts_stage(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>], k: usize) -> Result<()> {
        let stage = self.variants[st.variant].cts.get(k).ok_or_else(|| no_stage("sparse_classic", &format!("cts_stage({k})")))?;
        self.run_lt(cx, stage, cts)
    }
    /// x + conj x and the route's series (`approxModReductionSparse`), then the rescale.
    fn eval_mod(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>]) -> Result<()> {
        let em = &self.variants[st.variant].em;
        let ctx = cx.ctx;
        for x in cts.iter_mut() {
            if x.meta.deg == 2 && em.in_meta.nl == 1 {
                cx.ops.rescale(ctx, cx.ex, cx.sh, std::slice::from_mut(x))?;
            }
            let f = to_f(ctx.shape, &x.meta);
            if f.top != em.in_meta.top || f.nl != em.in_meta.nl || f.nf.to_bits() != em.in_meta.nf.to_bits() {
                return Err(shape("classic eval_mod", format!("input {f:?}, the program was built for {:?}", em.in_meta)));
            }
            let k = x.k();
            em.run(cx, &mut x.buf, k)?;
            let o = em.out_meta;
            x.meta = CtMeta { level: ctx.shape.n_q - 1 - o.top as u32, deg: o.nl as u8, nf: o.nf, ..x.meta };
            if x.meta.deg == 2 {
                cx.ops.rescale(ctx, cx.ex, cx.sh, std::slice::from_mut(x))?;
            }
        }
        Ok(())
    }
    fn stc_first_output(&self, _cx: &mut BtsCx<'_, 'a, W, M>, _st: &BtsState, _cts: &mut [Ct<'a, W>]) -> Result<()> {
        Err(no_stage("sparse_classic", "stc_first_output"))
    }
    /// A ciphertext above the StC plaintexts' level is dropped to it (exact level reduce).
    fn stc_enter(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>]) -> Result<()> {
        let pre = &self.variants[st.variant];
        let stc_top = pre.stc.first().map(|s| s.top).ok_or_else(|| no_stage("sparse_classic", "stc_enter"))?;
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
        let stage = self.variants[st.variant].stc.get(k).ok_or_else(|| no_stage("sparse_classic", &format!("stc_stage({k})")))?;
        self.run_lt(cx, stage, cts)
    }
    /// rotate(s)-add (the two halves of the 2s-slot view), x 2^correction, the slots restored.
    fn finish(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>]) -> Result<()> {
        let ctx = cx.ctx;
        for x in cts.iter_mut() {
            self.acc(cx, &self.half, x)?;
            let top = x.top(ctx.shape)?;
            if st.correction > 0 {
                self.pow2.mult(ctx, &cx.ex.compute, View::of(&x.buf, 0), x.k(), top, st.correction)?;
            }
            x.meta.tag = tag::t_bootstrap(&st.in_meta.tag, self.cfg.slots as i64, total_slots(&st.in_meta, self.ring_n));
            x.meta.slots = st.old_slots;
        }
        Ok(())
    }
    fn stc_high(&self, _cx: &mut BtsCx<'_, 'a, W, M>, _cts: &mut [Ct<'a, W>], _spec: &StcHigh) -> Result<()> {
        Err(no_stage("sparse_classic", "stc_high"))
    }
}

impl<'a, W: DevWord, M: DevStrategy<W>> BtsRoute<'a, W, M> for SparseClassicRoute<'a, W, M> {
    fn kind(&self) -> RouteKind {
        RouteKind::Sparse(self.cfg.slots)
    }
    /// Raw output: the last StC stage's top, degree 2, scale = EvalMod's x the StC plaintexts' (the rotate-add and the
    /// integer post-factor keep the metadata).
    fn landing(&self, sh: &Shadow, meta: &CtMeta, route: &Route) -> Result<CtMeta> {
        let f = self.outs[self.variant(route.raise_drop)?];
        Ok(CtMeta {
            level: sh.cfg.chain.n_q - 1 - f.top as u32,
            deg: f.nl as u8,
            nf: f.nf,
            slots: meta.slots,
            tag: tag::t_bootstrap(&meta.tag, self.cfg.slots as i64, total_slots(meta, self.ring_n)),
        })
    }
    fn run(&self, cx: &mut BtsCx<'_, 'a, W, M>, cts: &mut [Ct<'a, W>], route: &Route) -> Result<()> {
        for _ in 0..route.iterations.max(1) {
            run_staged(self, cx, cts, route, false)?;
        }
        Ok(())
    }
    fn staged(&self) -> Option<&dyn Staged<'a, W, M>> {
        Some(self)
    }
    fn mem(&self) -> (MemReq, BtsWorkShape) {
        (self.mem.clone(), self.work_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pc_core::bts_setup::{rotate_left, C64};

    /// Canonical embedding of a real polynomial (slot j = t(zeta^(5^j)), zeta = e^(i pi / N)).
    fn embed(t: &[f64]) -> Vec<C64> {
        let n = t.len();
        let m = 2 * n;
        let mut g = 1usize;
        let mut out = Vec::with_capacity(n / 2);
        for _ in 0..n / 2 {
            let mut acc = C64::ZERO;
            for (k, &c) in t.iter().enumerate() {
                if c != 0.0 {
                    acc += C64::cis(std::f64::consts::PI * ((g * k) % m) as f64 / n as f64).scale(c);
                }
            }
            out.push(acc);
            g = g * 5 % m;
        }
        out
    }

    fn tile(v: &[C64], len: usize) -> Vec<C64> {
        (0..len).map(|k| v[k % v.len()]).collect()
    }

    /// FIDESlib's LT on the full N/2-slot vector: plaintexts are the 2s-slot diagonals replicated (sparse encoding),
    /// rotations act on N/2 slots (`LtStage` semantics).
    fn fides_stage(st: &pc_core::bts_setup::FidesStage, x: &[C64]) -> Vec<C64> {
        let len = x.len();
        let mut y = vec![C64::ZERO; len];
        for i in 0..st.g_step as usize {
            let mut inner = vec![C64::ZERO; len];
            for j in 0..st.b_step as usize {
                let idx = st.b_step as usize * i + j;
                if idx >= st.num_diags {
                    continue;
                }
                let p = tile(&st.pts[idx], len);
                let xr = rotate_left(x, st.rot_in[j]);
                for k in 0..len {
                    inner[k] += p[k] * xr[k];
                }
            }
            let o = rotate_left(&inner, st.rot_out[i]);
            for k in 0..len {
                y[k] += o[k];
            }
        }
        y
    }

    /// The route's linear part on a small ring: fold -> CtS (FIDESlib layout, 2s-slot view) -> x + conj x -> StC ->
    /// rotate(s)-add maps the slots of any real polynomial t to C times the slots of t's projection onto the
    /// s-periodic subring (coefficients at multiples of N/2s), one real constant C for every slot count and budget.
    #[test]
    fn linear_part_is_the_subring_projection() {
        let n = 1usize << 10;
        let mut pr = pc_core::keys::Prng::from_seed([7; 8], 0);
        let t: Vec<f64> = (0..n).map(|_| pr.uniform_f64() - 0.5).collect();
        let v = embed(&t);
        let mut consts = vec![];
        for (s, budget) in [(32u32, [3u32, 3u32]), (64, [3, 3]), (16, [2, 2]), (128, [3, 3])] {
            let stride = n / (2 * s as usize);
            let tp: Vec<f64> = t.iter().enumerate().map(|(k, &c)| if k % stride == 0 { c } else { 0.0 }).collect();
            let want = embed(&tp);
            let f = cts_stc_factors(n, s, budget, [0, 0], 1.0, 1.0).unwrap();
            let lay = f.fides_layout();
            // fold
            let mut x = vec![C64::ZERO; n / 2];
            for j in 0..n / 2 / s as usize {
                let r = rotate_left(&v, (j * s as usize) as i64);
                for k in 0..n / 2 {
                    x[k] += r[k];
                }
            }
            for st in &lay.cts {
                x = fides_stage(st, &x);
            }
            let x: Vec<C64> = x.iter().map(|z| *z + z.conj()).collect();
            let mut y = x;
            for st in &lay.stc {
                y = fides_stage(st, &y);
            }
            let r = rotate_left(&y, s as i64);
            let out: Vec<C64> = y.iter().zip(&r).map(|(a, b)| *a + *b).collect();
            // least-squares real constant
            let (mut num, mut den) = (0.0, 0.0);
            for (o, w) in out.iter().zip(&want) {
                num += o.re * w.re + o.im * w.im;
                den += w.re * w.re + w.im * w.im;
            }
            let c = num / den;
            let err = out.iter().zip(&want).map(|(o, w)| (*o - w.scale(c)).abs()).fold(0.0, f64::max);
            let mx = out.iter().map(|o| o.abs()).fold(0.0, f64::max);
            assert!(err / mx < 1e-9, "s {s} budget {budget:?}: out != C proj (C {c}, rel err {:e})", err / mx);
            eprintln!("s {s} budget {budget:?}: C {c} rel err {:e}", err / mx);
            consts.push((s, c));
        }
        // the constant does not depend on s: one raise constant pre / (K N) serves every slot count
        for &(s, c) in &consts {
            assert!((c / consts[0].1 - 1.0).abs() < 1e-9, "C({s}) = {c} vs C({}) = {}", consts[0].0, consts[0].1);
        }
    }

    /// The n32 8192-slot route's plan: its realized landing is the compile table's (`multi_stage_landing({3, 3})` =
    /// 32, the dsw48 plan's PLAN_STEP_LANDING softmax:32), raise drops land `d drop` deeper.
    #[test]
    fn n32_8192_landing_and_size() {
        let chain = Chain::generate(&pc_core::params::ChainSpec::n32()).unwrap();
        let st = ScaleTables::precompute(&chain).unwrap();
        let sh = shadow_of(&st);
        let p = ClassicPlan::new(&chain, &ClassicCfg::n32(8192, &[0, 1])).unwrap();
        let want = pc_shadow::route::multi_stage_landing([3, 3], 2);
        for v in &p.variants {
            let raw = from_f(sh.cfg.chain, v.out);
            let real = sh.rescale(&raw).unwrap().out;
            assert_eq!((raw.deg, real.level, real.deg), (2, want.level + 2 * v.drop, 1), "drop {}", v.drop);
            let nd: Vec<usize> = v.cts.iter().chain(&v.stc).map(|h| h.coeffs.len()).collect();
            let w: Vec<usize> = v.cts.iter().chain(&v.stc).map(|h| h.width).collect();
            eprintln!("drop {}: em in {:?} out {:?}; diagonals {nd:?} widths {w:?}", v.drop, v.em_in, v.em_out);
        }
        let (m, w) = p.mem::<u32>().unwrap();
        eprintln!("rot steps {} {:?}; mem {:.2} GB; work {w:?}", p.rot_steps().len(), p.rot_steps(), m.total() as f64 / 1e9);
    }
}
