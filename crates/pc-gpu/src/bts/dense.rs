//! The dense route (`RouteKind::Dense`): full-slot bootstrap in the shipped order, FIDESlib `bootstrapImpl` /
//! `BootstrapStaged` with `stc_first` off:
//!
//!   begin -> mod_raise (real payload: x + conj x first; encapsulated raise to `L - d drop`, exact post-raise constant)
//!   -> CtS x4 (BSGS, fused ModDown-rescale) -> EvalMod (complex: Re/Im chains; real: one chain) -> rescale
//!   -> StC x3 (the last pays the LT-friendly offset, degree 2) -> finish (real: + conj; x 2^correction).
//!
//! Setup is two steps: `DensePlan::new` (host: factors, levels, encodings, metadata of every stage, sizes ->
//! `mem()`), then `DensePlan::build` (device tables after `MemPlan::allocate`). Every precomputation is encoded from
//! `pc_core::bts_setup` at the levels of `pc_shadow::route`, one variant per raise drop of the setup.
//! `iterations` > 1 runs perseus' `eval_bootstrap_iter` residual loop (y = bts(x); y += bts((x - y) 2^p) 2^-p).
use super::cts::{diag_coeffs, fold_periodic_giants, stage_width, tail_steps, KeyTails, LtEnd, LtGeom, LtPts, LtStage, PtSource};
use super::evalmod::{em_work, EmMode, EvalModPlan};
use super::finish::Pow2;
use super::raise::{self, EncapsKeys, RaisePlan};
use super::{common_meta, no_stage, BtsCx, BtsKeyRing, BtsPrecomp, BtsRoute, BtsSetup, BtsState, BtsWorkShape, MemReq, RouteKind, Staged, StcHigh};
use crate::context::{Context, DevStrategy, DevWord};
use crate::ct::Ct;
use crate::error::{shape, Result};
use crate::ks::{self, DotIo, DotRows, KeyGroup, KskSource, View};
use crate::poly::Layout;
use pc_core::bts_setup::cts_stc_factors;
use pc_core::encode::Complex;
use pc_core::params::Chain;
use pc_core::scale::ScaleTables;
use pc_cuda::{Arena, Compute, Stream};
use pc_shadow::evalmod::PsPlan;
use pc_shadow::fides::FMeta;
use pc_shadow::route::{cts_stage_levels, stc_stage_levels};
use pc_shadow::{BtsConfig, CtMeta, PtMeta, Quirks, Route, Shadow, ShadowConfig};
use std::marker::PhantomData;
use std::sync::Arc;

/// Bootstrap residual iterations' precision (OpenFHE's BTS_PRECISION).
pub const ITER_PRECISION: u32 = 12;

/// One linear-transform stage on the host: geometry, input top, end, plaintext scale, encoded coefficients.
pub struct LtHost {
    pub geom: LtGeom,
    pub top: usize,
    pub end: LtEnd,
    pub nf: f64,
    pub coeffs: Vec<Vec<i128>>,
    pub width: usize,
}

/// One raise-drop variant on the host, with the metadata after every stage (FIDESlib coordinates).
pub struct VariantHost {
    pub drop: u32,
    pub raise_top: usize,
    pub exact: f64,
    pub cts: Vec<LtHost>,
    pub stc: Vec<LtHost>,
    pub stc_real0: Option<LtHost>,
    pub raised: FMeta,
    pub em_in: FMeta,
    pub em_out: FMeta,
    pub em_slots: usize,
    /// the bootstrap output (degree 2)
    pub out: FMeta,
}

/// The dense route's host plan: every variant's tables before any device memory exists.
pub struct DensePlan {
    pub setup: BtsSetup,
    pub chain: Chain,
    pub variants: Vec<VariantHost>,
    pub ps: PsPlan,
    pub deg: u32,
}

fn to_f(shape: pc_shadow::ChainShape, m: &CtMeta) -> Result<FMeta> {
    Ok(FMeta { top: pc_shadow::meta::top_limb(shape, m.level)? as i32, nl: m.deg as i32, nf: m.nf })
}
fn from_f(shape: pc_shadow::ChainShape, f: FMeta) -> Result<CtMeta> {
    Ok(CtMeta { level: pc_shadow::meta::level_from_top_limb(shape, f.top as u32)?, deg: f.nl as u8, nf: f.nf, ..CtMeta::new(0, 1) })
}

/// A metadata-only Shadow over the chain's scale tables (stage bookkeeping).
pub fn shadow_of(st: &ScaleTables) -> Shadow {
    Shadow::new(ShadowConfig { chain: pc_shadow::ChainShape::from_tables(st), bts: BtsConfig::n32_shipped(), quirks: Quirks::none() }, Some(Arc::new(st.clone())))
}

impl DensePlan {
    pub fn new(chain: &Chain, setup: &BtsSetup) -> Result<DensePlan> {
        let st = ScaleTables::precompute(chain)?;
        let sh = shadow_of(&st);
        let shp = sh.cfg.chain;
        let d = chain.level_unit;
        let n = chain.n;
        let ps = PsPlan::from_coeffs(&setup.evalmod_coeffs)?;
        let mut qd = chain.q[0] as f64;
        for j in 1..d {
            qd *= chain.q[j] as f64;
        }
        let deg = (qd / (chain.spec.scaling_bits as f64).exp2()).log2().round() as u32;
        // the fused ModDown-rescale end is a composite-degree-2 kernel; other chains (n64: d = 1) end the inner
        // stages with the plain ModDown and pay the rescale at the next stage's entry (`run_lt`), same levels
        let inner_end = if d == 2 { LtEnd::FusedRescale } else { LtEnd::ModDown };
        let f = cts_stc_factors(n, (n / 2) as u32, setup.level_budget, setup.dim1, 1.0, super::stc::scale_dec(&st))?;
        let lay = f.fides_layout();
        let real0 = if setup.real { Some(super::stc::real_stage0(&f)?) } else { None };
        // FIDESLIB_CTS_ALT_BUDGET: the alternative CtS (budget + 1 stages) for the variants of drop >= 1
        let alt = match setup.cts_alt_budget {
            0 => None,
            b if b == setup.level_budget[0] + 1 => Some(cts_stc_factors(n, (n / 2) as u32, [b, setup.level_budget[1]], setup.dim1, 1.0, super::stc::scale_dec(&st))?.fides_layout()),
            b => return Err(shape("dense plan", format!("cts_alt_budget {b}: must be the CtS level budget + 1 ({})", setup.level_budget[0] + 1))),
        };
        let mut variants = Vec::new();
        for &drop in &setup.raise_drops {
            // an alternative-CtS variant raises one level higher and spends it on the extra CtS stage
            let (cts_lay, rdrop) = match &alt {
                Some(a) if drop >= 1 => (&a.cts, drop - 1),
                _ => (&lay.cts, drop),
            };
            let raise_top = chain.q.len() - 1 - d * rdrop as usize;
            // the shipped StC's decoding constant compensates q0 / sf(chain top); a raise to a lower top carries
            // sf(top) / sf(raise top) on its last StC stage (FIDESlib AddBootstrapRaiseVariant `fdrop`; ~1 at drop 1,
            // a 4-8 % output gain error at drops 2-5 without it)
            let fdrop = st.sf((chain.q.len() - 1) as i32)? / st.sf(raise_top as i32)?;
            let exact = (st.sf(raise_top as i32)? / qd) * (1.0 / (setup.boot_k * n as f64)) * 1.0;
            let raised = FMeta { top: raise_top as i32, nl: 1, nf: st.sf(raise_top as i32)? * (setup.cts0_t as f64).exp2() };
            let mut m = from_f(shp, raised)?;
            let levels = cts_stage_levels(&st, cts_lay.len() as u32, rdrop, Some(setup.cts0_t))?;
            let mut cts = Vec::new();
            for (e, fs) in cts_lay.iter().enumerate() {
                let lv = levels[e];
                if lv.top as usize != pc_shadow::meta::top_limb(shp, m.level)? as usize || m.deg != 1 {
                    return Err(shape("dense plan", format!("CtS stage {e} plaintexts at top {} but the input is {m:?}", lv.top)));
                }
                let (geom, coeffs) = fold_periodic_giants(n, LtGeom::from_fides(fs), diag_coeffs(chain, &fs.pts, lv.scale)?);
                cts.push(LtHost { geom, top: lv.top as usize, end: inner_end, nf: lv.scale, width: stage_width(n, &coeffs), coeffs });
                m = sh.mult_pt(&m, &PtMeta { level: lv.level, deg: 1, nf: lv.scale, tag: Default::default() })?.out;
                m = sh.rescale(&m)?.out;
            }
            let em_in = to_f(shp, &m)?;
            let (_, em_out, em_slots) = EvalModPlan::<u32>::program(&st, em_in, &ps, setup.evalmod_da_its)?;
            let mo = from_f(shp, em_out)?;
            let slev = stc_stage_levels(&st, mo.level, mo.deg, lay.stc.len() as u32)?;
            let mut m = if mo.deg == 2 { sh.rescale(&mo)?.out } else { mo };
            let mut stc = Vec::new();
            let nst = lay.stc.len();
            for (e, fs) in lay.stc.iter().enumerate() {
                let lv = slev[e];
                let top = pc_shadow::meta::top_limb(shp, m.level)?;
                if lv.top > top {
                    return Err(shape("dense plan", format!("StC stage {e} plaintexts at top {} above the input top {top}", lv.top)));
                }
                m.level = lv.level;
                let end = if e + 1 < nst { inner_end } else { LtEnd::ModDown };
                let (geom, coeffs) = fold_periodic_giants(n, LtGeom::from_fides(fs), diag_coeffs(chain, &fs.pts, if e + 1 == nst { lv.scale * fdrop } else { lv.scale })?);
                stc.push(LtHost { geom, top: lv.top as usize, end, nf: lv.scale, width: stage_width(n, &coeffs), coeffs });
                m = sh.mult_pt(&m, &PtMeta { level: lv.level, deg: 1, nf: lv.scale, tag: Default::default() })?.out;
                if e + 1 < nst && geom.offset == 0 {
                    m = sh.rescale(&m)?.out;
                }
            }
            let stc_real0 = match &real0 {
                Some(v) => {
                    let s0 = &stc[0];
                    let (geom, coeffs) = fold_periodic_giants(n, LtGeom::from_fides(&lay.stc[0]), diag_coeffs(chain, v, if nst == 1 { s0.nf * fdrop } else { s0.nf })?);
                    debug_assert_eq!(geom, s0.geom);
                    Some(LtHost { geom, top: s0.top, end: s0.end, nf: s0.nf, width: stage_width(n, &coeffs), coeffs })
                }
                None => None,
            };
            variants.push(VariantHost { drop, raise_top, exact, cts, stc, stc_real0, raised, em_in, em_out, em_slots, out: to_f(shp, &m)? });
        }
        Ok(DensePlan { setup: setup.clone(), chain: chain.clone(), variants, ps, deg })
    }

    /// Bytes of a stage's plaintexts and tables over `kp` special primes, with an identity key (`LtStage::new_with`).
    pub(crate) fn lt_bytes_with<W: DevWord>(&self, lay: &Layout, h: &LtHost, kp: usize, ident: bool) -> Result<usize> {
        Ok(LtPts::<W>::coeffs_bytes(h.coeffs.len(), h.top + 1 + lay.kp, h.width, lay.n) + LtStage::<W>::plan_bytes_with(lay, &self.chain.primes(), &h.geom, h.top, h.end, self.setup.k_max, kp, ident)?)
    }

    /// The CtS baby steps whose keys get a resident `a` tail (`KeyTails`) from digit `lt_regen_digits` on (with
    /// `merge_s2d`: of every stage but the first, whose keys are `tail_steps_s0`'s).
    pub fn tail_steps(&self) -> Vec<i64> {
        if !self.setup.lt_tails {
            return Vec::new();
        }
        let skip = self.setup.merge_s2d as usize;
        let st: Vec<(LtGeom, usize)> = self.variants.iter().flat_map(|v| v.cts.iter().skip(skip).map(|h| (h.geom, h.top))).collect();
        tail_steps(&self.chain, &st, self.setup.lt_regen_digits, super::cts::TAIL_MIN_DIGITS_DENSE)
    }

    /// With `merge_s2d`: the first CtS stage's baby steps whose `SparseRot` keys get tails.
    pub fn tail_steps_s0(&self) -> Vec<i64> {
        if !self.setup.lt_tails || !self.setup.merge_s2d {
            return Vec::new();
        }
        let st: Vec<(LtGeom, usize)> = self.variants.iter().filter_map(|v| v.cts.first().map(|h| (h.geom, h.top))).collect();
        tail_steps(&self.chain, &st, self.setup.lt_regen_digits, super::cts::TAIL_MIN_DIGITS_DENSE)
    }

    /// The first CtS stages' baby steps (the `SparseRot` keys `merge_s2d` needs).
    pub fn s0_steps(&self) -> Vec<i64> {
        let nh = (self.chain.n / 2) as i64;
        let mut v: Vec<i64> = self.variants.iter().filter_map(|v| v.cts.first()).flat_map(|h| (1..h.geom.b).map(move |i| ((i * h.geom.stride) as i64) % nh)).filter(|&r| r != 0).collect();
        v.sort();
        v.dedup();
        v
    }

    /// Device bytes of every table (MemPlan section "bts.dense") and the work shape.
    pub fn mem<W: DevWord>(&self) -> Result<(MemReq, BtsWorkShape)> {
        let lay = Layout::from_chain(&self.chain);
        let mut b = Pow2::<W>::bytes(lay.rows()) + EncapsKeys::bytes() + KeyGroup::bytes(1, true);
        b += KeyTails::<W>::bytes(&lay, self.chain.dnum(), self.setup.lt_regen_digits, self.tail_steps().len());
        b += KeyTails::<W>::bytes(&lay, self.chain.dnum(), self.setup.lt_regen_digits, self.tail_steps_s0().len());
        let mut em_slots = 0;
        let mut g_max = 1;
        for v in &self.variants {
            b += RaisePlan::<W>::table_bytes();
            for (i, h) in v.cts.iter().chain(&v.stc).chain(v.stc_real0.iter()).enumerate() {
                b += self.lt_bytes_with::<W>(&lay, h, lay.kp, self.setup.merge_s2d && i == 0)?;
                g_max = g_max.max(h.geom.g);
            }
            // EvalMod plans (complex + real share the program), exactly as `EvalModPlan::new` carves them
            let st = pc_core::scale::ScaleTables::precompute(&self.chain)?;
            let em = super::evalmod::EvalModPlan::<W>::table_bytes(&st, &lay, v.em_in, &self.ps, self.setup.evalmod_da_its, self.setup.k_max)?;
            b += if self.setup.real { 2 * em } else { em };
            em_slots = em_slots.max(v.em_slots);
        }
        let (n, kk) = em_work(em_slots, self.setup.k_max, EmMode::Complex);
        let mut r = MemReq::default();
        r.add("bts.dense", b);
        Ok((r, BtsWorkShape { k_max: self.setup.k_max, g_max: g_max.max(2), em_slots: n, em_kk: kk, spare: 3 }))
    }

    /// Device tables (after `MemPlan::allocate`, before `Arena::freeze`).
    pub fn build<'a, W: DevWord, M: DevStrategy<W>>(
        &self,
        arena: &'a Arena,
        s: &Stream<Compute>,
        ctx: &Context<'_, W, M>,
        keys: &dyn BtsKeyRing,
        stage_ptrs: Option<(*const std::ffi::c_void, *const std::ffi::c_void)>,
    ) -> Result<DenseRoute<'a, W, M>> {
        if self.setup.merge_s2d || self.setup.stc_ps != 0 {
            return Err(shape("dense route", "merge_s2d / stc_ps are StC-first (StcFirstPlan) and SparseB options"));
        }
        let n = ctx.lay.n;
        let k_max = self.setup.k_max;
        let mut variants = Vec::new();
        let tails = KeyTails::new(arena, s, ctx, keys, &self.tail_steps(), self.setup.lt_regen_digits)?;
        let ring = tails.ring(keys, n)?;
        let lt = |h: &LtHost| -> Result<LtStage<'a, W>> {
            let pts = LtPts::new(arena, s, ctx, h.top, PtSource::Coeffs(&h.coeffs))?;
            let mut st = LtStage::new(arena, s, ctx, h.geom, h.top, h.end, h.nf, pts, &ring, stage_ptrs, k_max)?;
            st.group = self.setup.lt_group;
            st.set_baby(self.setup.lt_baby)?;
            st.set_regen_digits(self.setup.lt_regen_digits)?;
            Ok(st)
        };
        for v in &self.variants {
            let em_k = |mode| {
                let mut em = EvalModPlan::new(arena, s, ctx, mode, v.em_in, &self.ps, self.setup.evalmod_da_its, keys.conj()?, keys.relin()?, stage_ptrs, k_max, self.setup.em_relin)?;
                em.fuse_rescale = self.setup.em_fuse_rescale;
                Ok::<_, crate::error::Error>(em)
            };
            variants.push(BtsPrecomp {
                slots: (n / 2) as u32,
                raise_drop: v.drop,
                raise: RaisePlan::new(arena, s, ctx, v.raise_top, v.exact, self.setup.cts0_t)?,
                cts: v.cts.iter().map(lt).collect::<Result<_>>()?,
                stc: v.stc.iter().map(lt).collect::<Result<_>>()?,
                stc_real0: v.stc_real0.as_ref().map(lt).transpose()?,
                em: em_k(EmMode::Complex)?,
                em_real: if self.setup.real { Some(em_k(EmMode::Real)?) } else { None },
                cts0_const: v.exact,
                cts0_t: self.setup.cts0_t,
                boot_k: self.setup.boot_k,
            });
        }
        let (mem, shape_) = self.mem::<W>()?;
        Ok(DenseRoute {
            setup: self.setup.clone(),
            variants,
            outs: self.variants.iter().map(|v| v.out).collect(),
            encaps: EncapsKeys::new(arena, s, n, keys.dense_to_sparse()?, keys.sparse_to_dense()?, stage_ptrs)?,
            conj: KeyGroup::new(arena, s, n, &[keys.conj()?], stage_ptrs)?,
            pow2: Pow2::new(arena, s, ctx)?,
            deg: self.deg,
            mem,
            work_shape: shape_,
            _m: PhantomData,
        })
    }
}

/// The dense route's device side.
pub struct DenseRoute<'a, W: DevWord, M: DevStrategy<W>> {
    pub setup: BtsSetup,
    pub variants: Vec<BtsPrecomp<'a, W>>,
    /// the output metadata of each variant
    outs: Vec<FMeta>,
    encaps: EncapsKeys<'a>,
    conj: KeyGroup<'a>,
    pow2: Pow2<'a, W>,
    /// log2(q0 / 2^scale_bits)
    pub deg: u32,
    mem: MemReq,
    work_shape: BtsWorkShape,
    _m: PhantomData<M>,
}

impl<'a, W: DevWord, M: DevStrategy<W>> DenseRoute<'a, W, M> {
    /// Switch every LT stage's baby-key source (`BtsSetup::lt_baby`; benches compare both in one process).
    pub fn set_lt_baby(&mut self, src: KskSource, regen_digits: usize) -> Result<()> {
        for v in &mut self.variants {
            for st in v.cts.iter_mut().chain(v.stc.iter_mut()).chain(v.stc_real0.iter_mut()) {
                st.set_baby(src)?;
                st.set_regen_digits(regen_digits)?;
            }
        }
        self.setup.lt_baby = src;
        self.setup.lt_regen_digits = regen_digits;
        Ok(())
    }

    fn variant(&self, drop: u32) -> Result<usize> {
        self.variants.iter().position(|v| v.raise_drop == drop).ok_or_else(|| crate::error::Error::Mem {
            knob: "BtsSetup::raise_drops",
            value: drop as usize,
            msg: format!("no dense variant for raise drop {drop} (built: {:?})", self.variants.iter().map(|v| v.raise_drop).collect::<Vec<_>>()),
        })
    }

    /// x := x + conj(x) on every batch (key switch with the conjugation key, then add).
    fn add_conj(&self, cx: &mut BtsCx<'_, 'a, W, M>, cts: &mut [Ct<'a, W>]) -> Result<()> {
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
            let staged = self.conj.reads_in_place(self.setup.ks_source) || self.conj.needs_stage() || !self.conj.has_a();
            if staged {
                let ndig = cx.ks.ndig(top)?;
                ks::stage_key(ctx, &ex.compute, cx.ks, &mut ex.ks, &self.conj, 0, ndig, true)?;
            }
            ks::dot(ctx, &ex.compute, cx.ks, &ex.ks, &self.conj, 0, 1, k, top, DotRows::All, io, staged)?;
            ks::moddown_to(ctx, &ex.compute, cx.ks, &mut ex.ks.ext, &mut ex.ks.conv, v, ks::MdOut::Add, k, top)?;
        }
        Ok(())
    }

    fn set_meta(cx: &BtsCx<'_, 'a, W, M>, cts: &mut [Ct<'a, W>], f: FMeta) -> Result<()> {
        let shp = cx.ctx.shape;
        for x in cts.iter_mut() {
            x.meta = CtMeta { level: pc_shadow::meta::level_from_top_limb(shp, f.top as u32)?, deg: f.nl as u8, nf: f.nf, ..x.meta };
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
                return Err(shape("LT stage", format!("input top {top}, stage plaintexts at {}", stage.top)));
            }
            let k = x.k();
            stage.run(cx, &mut x.buf, k, self.setup.ks_source)?;
            let pt = PtMeta { level: pc_shadow::meta::level_from_top_limb(ctx.shape, stage.top as u32)?, deg: 1, nf: stage.pt_nf, tag: Default::default() };
            let m = cx.sh.mult_pt(&x.meta, &pt)?.out;
            let (_, od) = stage.out_top_deg(ctx.chain.level_unit);
            x.meta = if od == 1 { cx.sh.rescale(&m)?.out } else { m };
        }
        Ok(())
    }

    /// One refresh (no iterations).
    fn once(&self, cx: &mut BtsCx<'_, 'a, W, M>, cts: &mut [Ct<'a, W>], route: &Route) -> Result<()> {
        super::run_staged(self, cx, cts, route, false).map(|_| ())
    }
}

impl<'a, W: DevWord, M: DevStrategy<W>> Staged<'a, W, M> for DenseRoute<'a, W, M> {
    fn begin(&self, cx: &BtsCx<'_, 'a, W, M>, cts: &[Ct<'a, W>], route: &Route, prescaled: bool) -> Result<BtsState> {
        let in_meta = common_meta(cts)?;
        if RouteKind::of(route, cx.ctx.lay.n) != RouteKind::Dense {
            return Err(shape("dense route", format!("route {route:?} is not the dense shipped-order route")));
        }
        if let Some(x) = cts.iter().find(|x| x.k() > self.setup.k_max) {
            return Err(shape("dense route", format!("batch of {} cts above BtsSetup::k_max {}", x.k(), self.setup.k_max)));
        }
        let variant = self.variant(route.raise_drop)?;
        let cf = route.cf.unwrap_or(self.setup.cf_default);
        if cf < self.setup.cf_window.0 || cf > self.setup.cf_window.1 || cf < self.deg {
            return Err(shape("bootstrap cf", format!("correction factor {cf} outside [{}, {}] (deg {})", self.setup.cf_window.0, self.setup.cf_window.1, self.deg)));
        }
        let mut correction = cf - self.deg;
        let cor_factor = 1u64 << correction;
        let real = route.real && self.variants[variant].em_real.is_some() && self.variants[variant].stc_real0.is_some();
        if route.real && !real {
            return Err(shape("bootstrap real", "real-payload variant not built (BtsSetup::real)"));
        }
        if real {
            correction += 1;
        }
        let pre = &self.variants[variant];
        Ok(BtsState {
            route: *route,
            slots: (cx.ctx.lay.n / 2) as u32,
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
            stc_first: false,
            stc_folded: false,
            real,
            n_cts: pre.cts.len(),
            n_stc: pre.stc.len(),
            variant,
            raise_drop: route.raise_drop,
            in_meta,
        })
    }
    fn stc_first_input(&self, _cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, _cts: &mut [Ct<'a, W>]) -> Result<()> {
        if st.stc_first {
            return Err(no_stage("dense", "stc_first_input"));
        }
        Ok(())
    }
    fn mod_raise(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &mut BtsState, cts: &mut [Ct<'a, W>]) -> Result<()> {
        if st.real {
            // an imaginary residue would leak into the real half's coefficients
            self.add_conj(cx, cts)?;
        }
        let pre = &self.variants[st.variant];
        raise::raise(cx, &pre.raise, Some(&self.encaps), self.setup.ks_source, cts, if st.prescaled { None } else { Some(st.correction) })?;
        st.shifted_flow = true;
        Ok(())
    }
    fn fold(&self, _cx: &mut BtsCx<'_, 'a, W, M>, _st: &BtsState, _cts: &mut [Ct<'a, W>]) -> Result<()> {
        Ok(())
    }
    fn cts_stage(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>], k: usize) -> Result<()> {
        let stage = self.variants[st.variant].cts.get(k).ok_or_else(|| no_stage("dense", &format!("cts_stage({k})")))?;
        self.run_lt(cx, stage, cts)
    }
    fn eval_mod(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>]) -> Result<()> {
        let pre = &self.variants[st.variant];
        let em = if st.real { pre.em_real.as_ref().ok_or_else(|| no_stage("dense", "eval_mod (real)"))? } else { &pre.em };
        let ctx = cx.ctx;
        for x in cts.iter_mut() {
            if x.meta.deg == 2 && em.in_meta.nl == 1 {
                // the last CtS stage ended with the plain ModDown (no fused rescale off composite-degree-2 chains)
                cx.ops.rescale(ctx, cx.ex, cx.sh, std::slice::from_mut(x))?;
            }
            let f = to_f(ctx.shape, &x.meta)?;
            if f.top != em.in_meta.top || f.nl != em.in_meta.nl || f.nf.to_bits() != em.in_meta.nf.to_bits() {
                return Err(shape("eval_mod", format!("input {f:?}, the program was built for {:?}", em.in_meta)));
            }
            let k = x.k();
            em.run(cx, &mut x.buf, k)?;
            Self::set_meta(cx, std::slice::from_mut(x), em.out_meta)?;
            if x.meta.deg == 2 && !st.stc_first {
                cx.ops.rescale(ctx, cx.ex, cx.sh, std::slice::from_mut(x))?;
            }
        }
        Ok(())
    }
    fn stc_first_output(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>]) -> Result<()> {
        let _ = (cx, st, cts);
        Err(no_stage("dense", "stc_first_output (RouteKind::StcFirst)"))
    }
    fn stc_enter(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>]) -> Result<()> {
        let pre = &self.variants[st.variant];
        let stc_top = pre.stc.first().map(|s| s.top).ok_or_else(|| no_stage("dense", "stc_enter"))?;
        let ctx = cx.ctx;
        for x in cts.iter_mut() {
            if x.top(ctx.shape)? > stc_top {
                if x.meta.deg == 2 {
                    cx.ops.rescale(ctx, cx.ex, cx.sh, std::slice::from_mut(x))?;
                }
                if x.top(ctx.shape)? > stc_top {
                    x.meta.level = pc_shadow::meta::level_from_top_limb(ctx.shape, stc_top as u32)?;
                }
            }
        }
        Ok(())
    }
    fn stc_stage(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>], k: usize) -> Result<()> {
        let pre = &self.variants[st.variant];
        let stage = if k == 0 && st.real { pre.stc_real0.as_ref() } else { pre.stc.get(k) };
        let stage = stage.ok_or_else(|| no_stage("dense", &format!("stc_stage({k})")))?;
        self.run_lt(cx, stage, cts)
    }
    fn finish(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>]) -> Result<()> {
        if st.real {
            self.add_conj(cx, cts)?;
        }
        let ctx = cx.ctx;
        let e = super::finish::pow2_exp(st.cor_factor)?;
        for x in cts.iter_mut() {
            let (k, top) = (x.k(), x.top(ctx.shape)?);
            self.pow2.mult(ctx, &cx.ex.compute, View::of(&x.buf, 0), k, top, e)?;
            x.meta.slots = st.old_slots;
        }
        Ok(())
    }
    fn stc_high(&self, _cx: &mut BtsCx<'_, 'a, W, M>, _cts: &mut [Ct<'a, W>], _spec: &StcHigh) -> Result<()> {
        Err(no_stage("dense", "stc_high (RouteKind::StcFirst)"))
    }
}

impl<'a, W: DevWord, M: DevStrategy<W>> BtsRoute<'a, W, M> for DenseRoute<'a, W, M> {
    fn kind(&self) -> RouteKind {
        RouteKind::Dense
    }
    fn landing(&self, sh: &Shadow, meta: &CtMeta, route: &Route) -> Result<CtMeta> {
        let v = self.variant(route.raise_drop)?;
        let f = self.outs[v];
        Ok(CtMeta { level: pc_shadow::meta::level_from_top_limb(sh.cfg.chain, f.top as u32)?, deg: f.nl as u8, nf: f.nf, ..*meta })
    }
    fn run(&self, cx: &mut BtsCx<'_, 'a, W, M>, cts: &mut [Ct<'a, W>], route: &Route) -> Result<()> {
        if route.iterations <= 1 {
            return self.once(cx, cts, route);
        }
        // y = bts(x); repeat: y += bts((x - y) 2^p) 2^-p   (perseus eval_bootstrap_iter, BTS_PRECISION p)
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
            let orig = Ct::new(x.meta, sa);
            ks::ew(ctx, &cx.ex.compute, ks::Ew::Copy, View::of(&orig.buf, 0), View::of(&x.buf, 0), None, ctx.lay.rs_q(top_in), k, 2)?;
            let mut e = Ct::new(x.meta, sb);
            let r = (|| -> Result<()> {
                self.once(cx, std::slice::from_mut(x), route)?;
                for _ in 1..route.iterations {
                    cx.ops.sub(ctx, cx.ex, cx.sh, std::slice::from_mut(&mut e), &[&orig], &[&*x])?;
                    cx.ops.mult_scalar_inplace(ctx, cx.ex, cx.sh, std::slice::from_mut(&mut e), &[Complex::new((ITER_PRECISION as f64).exp2(), 0.0)])?;
                    self.once(cx, std::slice::from_mut(&mut e), route)?;
                    cx.ops.mult_scalar_inplace(ctx, cx.ex, cx.sh, std::slice::from_mut(&mut e), &[Complex::new((-(ITER_PRECISION as f64)).exp2(), 0.0)])?;
                    cx.ops.add_inplace(ctx, cx.ex, cx.sh, std::slice::from_mut(x), &[&e])?;
                }
                Ok(())
            })();
            let (mut sa, mut sb) = (orig.buf, e.buf);
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
        (self.mem.clone(), self.work_shape)
    }
}
