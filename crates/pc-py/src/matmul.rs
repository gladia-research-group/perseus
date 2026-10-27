//! The CCMM service of a planned session and the finish dispatcher of the replay.
//!
//! A captured CCMM (impl/ccmm.py `Ccmm.apply`) is one external refresh per output sharing the operands' 32
//! ciphertexts; the compiler merges them into one `Finish` op whose descriptor is the v2 `args.external` dict
//! (`{op: "ccmm", ...}`). The service runs ccmm.py `_device` with every scale an argument: SinCBoot state ->
//! `stc_high` of both operands -> `Ccmm::mul` -> (merge) -> `sinc_boot` -> x 2^shift. `c_unit` is calibrated on the
//! device at setup from probe matrices (`Session.ccmm_probe`), `nf_ref` / `nf_split` from a token
//! (`Session.ccmm_prepare`); keys from the host secret or the client's
//! bundle (`CcmmKeys::generate`). No host sync, no allocation per call.
//! The pipeline and the operand factors: docs/ARCHITECTURE.md "pc-py CCMM service".
use crate::pcmm::{PcmmExec, PcmmRt};
use crate::session::{gerr, M, W};
use pc_core::ccmm_keys::{CcmmKeys, KY, NB, NBIG};
use pc_gpu::bts::finish::Pow2;
use pc_gpu::bts::stc_first::StcFirstRoute;
use pc_gpu::bts::stc_high::StcHighShape;
use pc_gpu::bts::{BtsCx, BtsState, Staged, StcHigh};
use pc_gpu::limbs::MonomialTable;
use pc_gpu::matmul::ccmm::{self, Ccmm, CcmmCfg};
use pc_gpu::{Ct, CtBuf, MemPlan};
use pc_program::bind::FinishStep;
use pc_program::replay::{copy_ct, FinishExec};
use pc_shadow::{CtMeta, Route};
use pyo3::prelude::*;

/// A CCMM finish descriptor (`Finish::desc`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CcmmDesc {
    pub site: String,
    pub n_out: usize,
    pub b_transposed: bool,
    pub merge: bool,
    pub shift: u32,
    pub amax: f64,
    pub bmax: f64,
    pub out_scale: f64,
    /// The output refreshes' raise drop (PF_CCMM_RAISE, ccmm.py `raise_of(site)`): SinCBoot runs that drop's raise
    /// variant and lands `drop` CKKS levels deeper.
    pub raise_drop: u32,
}

impl CcmmDesc {
    /// `None` for a descriptor that is not a CCMM one.
    pub fn parse(desc: &str) -> Option<CcmmDesc> {
        let v: serde_json::Value = serde_json::from_str(desc).ok()?;
        if v.get("op")?.as_str()? != "ccmm" {
            return None;
        }
        let f = |k: &str| v.get(k).and_then(serde_json::Value::as_f64);
        let b = |k: &str| v.get(k).and_then(serde_json::Value::as_bool).unwrap_or(false);
        Some(CcmmDesc {
            site: v.get("site").and_then(serde_json::Value::as_str).unwrap_or("ccmm").to_string(),
            n_out: v.get("n_out")?.as_u64()? as usize,
            b_transposed: b("b_transposed"),
            merge: b("merge"),
            shift: v.get("shift").and_then(serde_json::Value::as_u64).unwrap_or(0) as u32,
            amax: f("amax").unwrap_or(1.0),
            bmax: f("bmax").unwrap_or(1.0),
            out_scale: f("out_scale").unwrap_or(1.0),
            raise_drop: v.get("raise_drop").and_then(serde_json::Value::as_u64).unwrap_or(0) as u32,
        })
    }

    /// A P V site (`pv<layer>`, the encoders' `mpv<layer>[.i.j]`): the PV operating point, else the QK one.
    pub fn is_pv(&self) -> bool {
        self.site.strip_prefix('m').unwrap_or(&self.site).starts_with("pv")
    }

    /// The operand factors (ccmm.py `_factors`).
    pub fn factors(&self, c_unit: f64) -> (f64, f64) {
        factors(c_unit, self.amax, self.bmax, self.shift, self.out_scale, self.merge)
    }
}

pub(crate) fn factors(c_unit: f64, amax: f64, bmax: f64, shift: u32, out_scale: f64, merge: bool) -> (f64, f64) {
    let p = out_scale * (-(shift as f64)).exp2() / c_unit * if merge { 0.5 } else { 1.0 };
    ((p * bmax / amax).sqrt(), (p * amax / bmax).sqrt())
}

/// The stc_high shapes the service needs on the StC-first route (StcFirstSetup::high): the operands' (50, skip 1)
/// and SinCBoot's stage 0 for every left-operand state (`levels`: (level, deg) of the `a` operands the programs
/// and the probes use, with the raise drop of their finish). The landing is the StC-first route's (`BtsConfig::n32_shipped`:
/// a plan's runtime Shadow may not configure StC-first for its own dense sites).
pub(crate) fn high_shapes(chain: pc_shadow::ChainShape, cf: u32, levels: &std::collections::BTreeSet<(u32, u8, u32)>) -> PyResult<Vec<StcHighShape>> {
    let mut out = vec![StcHighShape { out_level: 50, skip_mask: 1 }];
    let mut bts = pc_shadow::BtsConfig::n32_shipped();
    // the route builds a variant for every drop its finishes use (session `route_needs`)
    bts.raise_drops.extend(levels.iter().map(|x| x.2).filter(|&x| x > 0));
    for &(l, d, drop) in levels {
        let r = Route { stc_first: true, cf: Some(cf), raise_drop: drop, ..Route::dense() };
        let land = bts.landing(chain, &pc_shadow::Quirks::none(), &r, l, d).map_err(crate::level_err)?;
        let s = StcHighShape { out_level: ccmm::stage0_level(&CtMeta::new(land.level, land.deg), chain.unit), skip_mask: 6 };
        if !out.contains(&s) {
            out.push(s);
        }
    }
    Ok(out)
}

/// What the session reserves for the service.
pub(crate) struct CcmmPlan {
    pub cfg: CcmmCfg,
}

impl CcmmPlan {
    pub fn reserve(&self, mp: &mut MemPlan, n_q: usize, n: usize) -> PyResult<()> {
        mp.reserve("ccmm", Ccmm::<W>::bytes(&self.cfg));
        let ct = pc_cuda::Arena::bytes_for::<W>(CtBuf::<pc_cuda::Compute, W>::words(&mp.lay, 1));
        mp.reserve("ccmm.cts", 2 * NBIG * ct);
        mp.reserve("ccmm.mono", MonomialTable::<W>::bytes(n_q, n));
        mp.reserve("ccmm.pow2", Pow2::<W>::bytes(mp.lay.rows()));
        Ok(())
    }
}

/// The service: engine, operand ciphertexts, merge table, 2^shift table, calibration. Field order = drop order:
/// the engine's GEMM plan borrows the cuBLASLt handle declared last.
pub(crate) struct CcmmRt {
    pub ccmm: Ccmm<'static, 'static, W>,
    pub a: Vec<Ct<'static, W>>,
    pub b: Vec<Ct<'static, W>>,
    pub mono: MonomialTable<'static, W>,
    pub pow2: Pow2<'static, W>,
    /// output value per unit operand-factor product (`ccmm_set`, from the probe calibration)
    pub c_unit: Option<f64>,
    /// operand scale reference (a token at the bootstrap output level) and the split's scale
    pub nf_ref: f64,
    pub nf_split: f64,
    /// operating-point factors (QK sites, PV sites): operands x sqrt(kappa) each, SinCBoot's stage-0 transform scaled
    /// by 1/kappa (exact, no level): the SinCBoot input integer |M| moves by kappa at an unchanged output value,
    /// +1 bit of output precision per bit of kappa up to the input-range cliff (measured: `Session.ccmm_probe`)
    pub kappa: (f64, f64),
    /// the SinCBoot route's CF (FIDESLIB_BTS_STC_FIRST_CF)
    pub cf: u32,
    pub calls: u64,
    pub keygen_s: f64,
    #[allow(dead_code)]
    lt: Box<pc_cuda::blas::Lt>,
}

impl CcmmRt {
    /// Setup (before freeze): the engine (PPMM GEMM autotuned) on `keys` (made from the secret by the session or by
    /// the client), the operand cts.
    #[allow(clippy::too_many_arguments)]
    pub fn build(plan: CcmmPlan, arena: &'static pc_cuda::Arena, dev: &pc_cuda::Device, s: &pc_cuda::Stream<pc_cuda::Compute>, ctx: &pc_gpu::Context<'static, W, M>, keys: CcmmKeys, keygen_s: f64, cf: u32) -> PyResult<CcmmRt> {
        let lt = Box::new(pc_cuda::blas::Lt::new(dev).map_err(|x| gerr(x.into()))?);
        // SAFETY: the box is owned by the returned CcmmRt and declared after `ccmm`, so it is dropped after the plan
        // that borrows it; its heap address is stable
        let ltr: &'static pc_cuda::blas::lt::Lt = unsafe { &*(&lt.inner as *const pc_cuda::blas::lt::Lt) };
        let ccmm = Ccmm::<W>::new(arena, dev, s, ctx, ltr, plan.cfg, &keys).map_err(gerr)?;
        let ct = |_| -> PyResult<Ct<'static, W>> { Ok(Ct::new(CtMeta::new(0, 1), CtBuf::new(arena, "ccmm.cts", ctx.lay, 1).map_err(gerr)?)) };
        let a = (0..NBIG).map(ct).collect::<PyResult<Vec<_>>>()?;
        let b = (0..NBIG).map(ct).collect::<PyResult<Vec<_>>>()?;
        let mono = MonomialTable::<W>::new(arena, s, ctx, NB / KY).map_err(gerr)?;
        let pow2 = Pow2::<W>::new(arena, s, ctx).map_err(gerr)?;
        Ok(CcmmRt { ccmm, a, b, mono, pow2, c_unit: None, nf_ref: 0.0, nf_split: 0.0, kappa: (1.0, 1.0), cf, calls: 0, keygen_s, lt })
    }

    fn route(&self, drop: u32) -> Route {
        Route { stc_first: true, cf: Some(self.cf), raise_drop: drop, ..Route::dense() }
    }

    /// The SinCBoot state of `a`'s operands (`bts.begin(a, S, False, True)` before their stc_high) under the raise
    /// variant of `drop` (ccmm.py: `raise_scope(drop)` around the begin).
    pub fn state(&self, cx: &BtsCx<'_, 'static, W, M>, route: &StcFirstRoute<'static, W, M>, a: &CtMeta, drop: u32) -> pc_gpu::Result<BtsState> {
        ccmm::sinc_state(cx.ctx, route, a, &self.route(drop))
    }

    /// `nf_ref` and `nf_split` from a token at the bootstrap output level (ccmm.py `prepare`); `scratch` is overwritten.
    pub fn prepare(&mut self, cx: &mut BtsCx<'_, 'static, W, M>, route: &StcFirstRoute<'static, W, M>, scratch: &mut Ct<'static, W>) -> pc_gpu::Result<()> {
        let st = self.state(cx, route, &scratch.meta, 0)?;
        self.nf_ref = scratch.meta.nf;
        self.nf_split = ccmm::split_nf(cx, route, &st, scratch)?;
        Ok(())
    }

    /// One CCMM on the service's operands `self.a[..16]`, `self.b[..16]` (already copied in): the outputs end in
    /// `self.a[..n_out]` (n_out = 16, or 8 merged).
    /// `kappa`: the operating-point factor of this call (operands x sqrt(kappa) each, SinCBoot stage 0 x 1/kappa);
    /// `drop`: the output refreshes' raise drop.
    #[allow(clippy::too_many_arguments)]
    pub fn run(&mut self, cx: &mut BtsCx<'_, 'static, W, M>, route: &StcFirstRoute<'static, W, M>, bt: bool, fa: f64, fb: f64, shift: u32, merge: bool, kappa: f64, drop: u32) -> pc_gpu::Result<usize> {
        let (fa, fb) = (fa * kappa.sqrt(), fb * kappa.sqrt());
        if !(self.nf_ref > 0.0 && self.nf_split > 0.0) {
            return Err(pc_gpu::Error::Shape { what: "ccmm service", msg: "nf_ref / nf_split unset (Session.ccmm_prepare)".into() });
        }
        // SinCBoot state per left operand, from its metadata before stc_high (ccmm.py: `begin` on each `a`); the
        // operands of a CCMM may come from producers at different levels (the paged encoder's Q copies)
        let a_meta: Vec<CtMeta> = self.a[..NBIG].iter().map(|x| x.meta).collect();
        let slots = (cx.ctx.lay.n / 2) as u32;
        let nf_ref = self.nf_ref;
        let hi = |f: f64| StcHigh { slots, out_level: 50, factor: f, skip_mask: 1, nf_ref };
        // operands whose scales differ (Q chunks of distinct producers) are taken one by one: the factor is
        // normalised per ciphertext against nf_ref
        for (xs, f) in [(&mut self.a, fa), (&mut self.b, fb)] {
            let m0 = xs[0].meta;
            if xs.iter().all(|x| x.meta.nf == m0.nf && x.meta.level == m0.level && x.meta.deg == m0.deg) {
                Staged::stc_high(route, cx, &mut xs[..NBIG], &hi(f)).map_err(|e| ctx_err("operand stc_high", &[m0], e))?;
            } else {
                for x in xs[..NBIG].iter_mut() {
                    let m = x.meta;
                    Staged::stc_high(route, cx, std::slice::from_mut(x), &hi(f)).map_err(|e| ctx_err("operand stc_high", &[m], e))?;
                }
                // the integers now sit at nf_ref's scale on every operand; their scale fields still trace each
                // producer's chain: one stamp, as the CCMM reads integers only (outputs are restamped below)
                let nf = xs[0].meta.nf;
                for x in xs[..NBIG].iter_mut() {
                    x.meta.nf = nf;
                }
            }
        }
        let (ctx, s) = (cx.ctx, &cx.ex.compute);
        self.ccmm.mul_inplace(ctx, s, &mut self.a[..NBIG], &self.b[..NBIG], bt, None).map_err(|e| ctx_err("ccmm mul (a)", &self.a[..NBIG].iter().map(|x| x.meta).collect::<Vec<_>>(), e))?;
        let n_out = if merge {
            for i in 0..NBIG / 2 {
                let (lo, hi) = self.a.split_at_mut(2 * i + 1);
                pc_gpu::limbs::merge_monomial(cx.ctx, cx.ex, &self.mono, &mut lo[2 * i], &hi[0])?;
            }
            // the merged outputs a[2i] -> a[i] (buffer handles swap, no copy)
            for i in 1..NBIG / 2 {
                self.a.swap(i, 2 * i);
            }
            NBIG / 2
        } else {
            NBIG
        };
        // output i comes from left operand i (merged: 2 i); runs of equal operand metadata share one SinCBoot call
        let src = |i: usize| a_meta[if merge { 2 * i } else { i }];
        let mut lo = 0;
        while lo < n_out {
            let m = src(lo);
            let mut hi = lo + 1;
            while hi < n_out && src(hi) == m {
                hi += 1;
            }
            let st = self.state(cx, route, &m, drop)?;
            let outs: Vec<CtMeta> = self.a[lo..hi].iter().map(|x| x.meta).collect();
            sinc_boot_scaled(cx, route, &st, &mut self.a[lo..hi], self.nf_split, 1.0 / kappa)
                .map_err(|e| ctx_err(&format!("SinCBoot outputs {lo}..{hi} (operand {m:?})"), &outs, e))?;
            lo = hi;
        }
        if shift > 0 {
            for x in self.a[..n_out].iter_mut() {
                ccmm::leaf::mult_int(cx.ctx, &cx.ex.compute, &self.pow2, x, 1u64 << shift)?;
            }
        }
        self.calls += 1;
        Ok(n_out)
    }
}

/// `ccmm::sinc_boot` with the StC stage-0 transform scaled by `factor` (its plaintexts: exact, no extra level).
fn sinc_boot_scaled(cx: &mut BtsCx<'_, 'static, W, M>, route: &StcFirstRoute<'static, W, M>, st: &BtsState, cts: &mut [Ct<'static, W>], nf_split: f64, factor: f64) -> pc_gpu::Result<()> {
    use pc_gpu::bts::staged::{BtsCall, CallOpts, Phase, StagedRef};
    for x in cts.iter_mut() {
        ccmm::leaf::set_scale(x, nf_split, 2);
    }
    let mut call = BtsCall::resume(StagedRef::StcFirst(route), st.clone(), Phase::Split, CallOpts::default());
    call.at_split(cx, cts)?;
    call.run_to(cx, cts, Phase::Done)?;
    let m = pc_gpu::bts::common_meta(cts)?;
    let spec = StcHigh { slots: (cx.ctx.lay.n / 2) as u32, out_level: ccmm::stage0_level(&m, cx.ctx.chain.level_unit as u32), factor, skip_mask: 6, nf_ref: 0.0 };
    Staged::stc_high(route, cx, cts, &spec)
}

/// `e` with the step and the ciphertext metadata it ran on.
fn ctx_err(step: &str, metas: &[CtMeta], e: pc_gpu::Error) -> pc_gpu::Error {
    let ms: Vec<String> = metas.iter().map(|m| format!("({}, {}, 2^{:.3})", m.level, m.deg, m.nf.log2())).collect();
    pc_gpu::Error::Shape { what: "ccmm service", msg: format!("{step} on [{}]: {e}", ms.join(" ")) }
}

fn ferr(f: &FinishStep, msg: impl Into<String>) -> pc_program::Error {
    pc_program::Error::Replay { block: format!("finish {}", f.desc), msg: msg.into() }
}

/// The replay's finish service: PCMM, CCMM and TFHE finishes on the session's StC-first route, by descriptor.
pub(crate) struct FinishSvc<'r> {
    pub pcmm: Option<&'r mut PcmmRt>,
    pub ccmm: Option<&'r mut CcmmRt>,
    pub tfhe: Option<&'r mut crate::tfhe::TfheRt>,
    pub route: &'r StcFirstRoute<'static, W, M>,
}

impl<'r> FinishExec<'static, W, M> for FinishSvc<'r> {
    fn finish(&mut self, cx: &mut BtsCx<'_, 'static, W, M>, f: &FinishStep, slots: &mut [Ct<'static, W>]) -> pc_program::Result<()> {
        if crate::tfhe::Desc::parse(&f.desc).is_some() {
            let rt = self.tfhe.as_deref_mut().ok_or_else(|| ferr(f, "this session has no TFHE service (Session(tfhe=...))"))?;
            return crate::tfhe::TfheExec { rt, route: self.route }.finish(cx, f, slots).map_err(|e| match e {
                pc_program::Error::Replay { .. } => e,
                e => ferr(f, e.to_string()),
            });
        }
        if let Some(d) = CcmmDesc::parse(&f.desc) {
            let rt = self.ccmm.as_deref_mut().ok_or_else(|| ferr(f, "this session has no CCMM service (Session(ccmm=True))"))?;
            let c_unit = rt.c_unit.ok_or_else(|| ferr(f, "CCMM not calibrated (Session.ccmm_set)"))?;
            if f.ins.len() != 2 * NBIG || f.outs.len() != d.n_out || d.n_out != if d.merge { NBIG / 2 } else { NBIG } {
                return Err(ferr(f, format!("{} inputs / {} outputs compiled for n_out {} (merge {})", f.ins.len(), f.outs.len(), d.n_out, d.merge)));
            }
            for (i, &sl) in f.ins.iter().enumerate() {
                let dst = if i < NBIG { &mut rt.a[i] } else { &mut rt.b[i - NBIG] };
                copy_ct(cx.ctx, &cx.ex.compute, dst, &slots[sl as usize])?;
            }
            let (fa, fb) = d.factors(c_unit);
            let kappa = if d.is_pv() { rt.kappa.1 } else { rt.kappa.0 };
            let n = rt.run(cx, self.route, d.b_transposed, fa, fb, d.shift, d.merge, kappa, d.raise_drop).map_err(|e| ferr(f, e.to_string()))?;
            for m in 0..n {
                copy_ct(cx.ctx, &cx.ex.compute, &mut slots[f.outs[m] as usize], &rt.a[m])?;
            }
            return Ok(());
        }
        let rt = self.pcmm.as_deref_mut().ok_or_else(|| ferr(f, "this session has no PCMM service (Session(pcmm=...))"))?;
        PcmmExec { rt, route: self.route }.finish(cx, f, slots).map_err(|e| match e {
            pc_program::Error::Replay { .. } => e,
            e => ferr(f, e.to_string()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desc() {
        let d = CcmmDesc::parse(r#"{"op": "ccmm", "site": "b0.qk", "out": 3, "n_out": 8, "b_transposed": true, "merge": true, "prepared_b": false, "shift": 2, "amax": 4.0, "bmax": 2.0, "out_scale": 0.5}"#).unwrap();
        assert_eq!((d.n_out, d.b_transposed, d.merge, d.shift), (8, true, true, 2));
        assert!(CcmmDesc::parse(r#"{"op": "pcmm", "linear": "x"}"#).is_none());
        let site = |s: &str| CcmmDesc::parse(&format!(r#"{{"op": "ccmm", "site": "{s}", "n_out": 8}}"#)).unwrap().is_pv();
        assert!(site("pv3") && site("mpv3") && site("mpv11.1.0"));
        assert!(!site("qk3") && !site("mqk3") && !site("mqk0.0.1") && !site("b0.qk"));
        let (fa, fb) = d.factors(2.0);
        let p: f64 = 0.5 * 0.25 / 2.0 * 0.5;
        assert!((fa * fb - p).abs() < 1e-15 && (fa / fb - 2.0 / 4.0).abs() < 1e-15);
    }
}
