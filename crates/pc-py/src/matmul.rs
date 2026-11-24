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
use pc_program::program::GRID_INPUTS;
use pc_program::replay::{copy_ct, FinishExec, RemoteLane};
use pc_shadow::{CtMeta, Route};
use pyo3::prelude::*;

/// A CCMM finish descriptor (`Finish::desc`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CcmmDesc {
    pub site: String,
    pub n_out: usize,
    pub b_transposed: bool,
    /// Outputs per SinCBoot: 1 (none), 2 (operands duplicated on batch l + 16) or 4 (on l + 8 k): the group's
    /// outputs merged as o0 + X^1024 o1 + X^2048 o2 + X^3072 o3 (Horner on the X^1024 monomial, no key switch).
    pub merge: u32,
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
            merge: match v.get("merge") {
                Some(serde_json::Value::Bool(m)) => if *m { 2 } else { 1 },
                Some(x) => x.as_u64().unwrap_or(1).max(1) as u32,
                None => 1,
            },
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

pub(crate) fn factors(c_unit: f64, amax: f64, bmax: f64, shift: u32, out_scale: f64, merge: u32) -> (f64, f64) {
    let p = out_scale * (-(shift as f64)).exp2() / c_unit / f64::from(merge.max(1));
    ((p * bmax / amax).sqrt(), (p * amax / bmax).sqrt())
}

/// The stc_high shapes the service needs on the StC-first route (StcFirstSetup::high): the operands' (`operand_level`, skip 1)
/// and SinCBoot's stage 0 for every left-operand state (`levels`: (level, deg) of the `a` operands the programs
/// and the probes use, with the raise drop of their finish). The landing is the StC-first route's (`BtsConfig::n32_shipped`:
/// a plan's runtime Shadow may not configure StC-first for its own dense sites).
pub(crate) fn high_shapes(chain: pc_shadow::ChainShape, cf: u32, levels: &std::collections::BTreeSet<(u32, u8, u32)>) -> PyResult<Vec<StcHighShape>> {
    let mut out = vec![StcHighShape { out_level: ccmm::operand_level(chain.n_q), skip_mask: 1 }];
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
    /// `PC_FINISH_PROFILE`: timing events around `run`'s phases, and the last run's phase times (ms: operands' StC
    /// stages, product + merge, output refreshes, shift)
    pub prof: Option<Vec<pc_cuda::Event>>,
    pub last_prof: Option<[f64; 4]>,
    /// PC_CCMM_PHASES=1 (diagnostic): timing events inside the product (`CcmmTimes`: ring in, CMT, PPMM, CMTs +
    /// relinearization, ring out), each call's printed
    pub phases: Option<Vec<pc_cuda::Event>>,
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
        let mut ccmm = Ccmm::<W>::new(arena, dev, s, ctx, ltr, plan.cfg, &keys).map_err(gerr)?;
        // PC_CCMM_SPLIT with a remote device: the half-rows PPMM plan of split products
        if pc_program::replay::ccmm_split() && crate::remote::requested()?.is_some() {
            ccmm.enable_split(ltr, dev, s).map_err(gerr)?;
        }
        let ct = |_| -> PyResult<Ct<'static, W>> { Ok(Ct::new(CtMeta::new(0, 1), CtBuf::new(arena, "ccmm.cts", ctx.lay, 1).map_err(gerr)?)) };
        let a = (0..NBIG).map(ct).collect::<PyResult<Vec<_>>>()?;
        let b = (0..NBIG).map(ct).collect::<PyResult<Vec<_>>>()?;
        let mono = MonomialTable::<W>::new(arena, s, ctx, NB / KY).map_err(gerr)?;
        let pow2 = Pow2::<W>::new(arena, s, ctx).map_err(gerr)?;
        let prof = if crate::pcmm::finish_profile() { Some((0..5).map(|_| pc_cuda::Event::timing(dev)).collect::<Result<Vec<_>, _>>().map_err(|x| gerr(x.into()))?) } else { None };
        let phases = if std::env::var("PC_CCMM_PHASES").is_ok_and(|v| v.trim() == "1") { Some((0..6).map(|_| pc_cuda::Event::timing(dev)).collect::<Result<Vec<_>, _>>().map_err(|x| gerr(x.into()))?) } else { None };
        Ok(CcmmRt { ccmm, a, b, mono, pow2, c_unit: None, nf_ref: 0.0, nf_split: 0.0, kappa: (1.0, 1.0), cf, calls: 0, keygen_s, prof, last_prof: None, phases, lt })
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
    /// `self.a[..n_out]` (n_out = 16 / merge).
    /// `kappa`: the operating-point factor of this call (operands x sqrt(kappa) each, SinCBoot stage 0 x 1/kappa);
    /// `drop`: the output refreshes' raise drop.
    #[allow(clippy::too_many_arguments)]
    ///
    /// `far` (`PC_REMOTE_FINISH`): the last half of every run of output refreshes on the remote refresh device.
    /// `out_slots` (ownership, PC_OWN): output i's slot, for the remote halves left in the replica's slots too.
    pub fn run(&mut self, cx: &mut BtsCx<'_, 'static, W, M>, route: &StcFirstRoute<'static, W, M>, bt: bool, fa: f64, fb: f64, shift: u32, merge: u32, kappa: f64, drop: u32, far: Option<&mut dyn RemoteLane<'static, W, M>>, out_slots: Option<&[u32]>) -> pc_gpu::Result<usize> {
        let mut far = far.filter(|r| crate::pcmm::remote_finish_ccmm() && r.has_finish());
        let check = far.is_some() && (pc_program::replay::remote_check() || pc_program::replay::ccmm_split_check());
        let (fa, fb) = (fa * kappa.sqrt(), fb * kappa.sqrt());
        let g = merge.max(1) as usize;
        if NBIG % g != 0 || !g.is_power_of_two() {
            return Err(pc_gpu::Error::Shape { what: "ccmm service", msg: format!("merge {g} does not divide {NBIG} outputs") });
        }
        if !(self.nf_ref > 0.0 && self.nf_split > 0.0) {
            return Err(pc_gpu::Error::Shape { what: "ccmm service", msg: "nf_ref / nf_split unset (Session.ccmm_prepare)".into() });
        }
        // SinCBoot state per left operand, from its metadata before stc_high (ccmm.py: `begin` on each `a`); the
        // operands of a CCMM may come from producers at different levels (the paged encoder's Q copies)
        let a_meta: Vec<CtMeta> = self.a[..NBIG].iter().map(|x| x.meta).collect();
        let slots = (cx.ctx.lay.n / 2) as u32;
        let nf_ref = self.nf_ref;
        let out_level = ccmm::operand_level(cx.ctx.shape.n_q);
        let hi = |f: f64| StcHigh { slots, out_level, factor: f, skip_mask: 1, nf_ref };
        // operands whose scales differ (Q chunks of distinct producers) are taken one by one: the factor is
        // normalised per ciphertext against nf_ref
        self.mark(cx, 0)?;
        // PC_CCMM_OPERAND_SPLIT: the last half of each operand side converted on the remote device (one metadata per
        // side only), issued first and joined before the product
        let op_split = far.is_some() && pc_program::replay::ccmm_operand_split();
        // PC_CCMM_SPLIT: the product split across both devices, the remote operand halves kept there (both sides at one
        // metadata each, as the operand split takes them)
        let uni = |xs: &[Ct<'static, W>]| xs[..NBIG].iter().all(|x| x.meta.nf == xs[0].meta.nf && x.meta.level == xs[0].meta.level && x.meta.deg == xs[0].meta.deg);
        let split = op_split && pc_program::replay::ccmm_split() && g <= NBIG / 2 && far.as_deref().is_some_and(|r| r.ccmm_split_on()) && uni(&self.a) && uni(&self.b);
        let mut op_far = false;
        let mut op_stash: Vec<(usize, CtMeta, [Vec<W>; 2])> = vec![];
        for (side, (xs, f)) in [(&mut self.a, fa), (&mut self.b, fb)].into_iter().enumerate() {
            let m0 = xs[0].meta;
            if xs[..NBIG].iter().all(|x| x.meta.nf == m0.nf && x.meta.level == m0.level && x.meta.deg == m0.deg) {
                let h = if op_split { NBIG / 2 } else { NBIG };
                if h < NBIG {
                    if check {
                        for (j, x) in xs[h..NBIG].iter().enumerate() {
                            op_stash.push((side * NBIG + h + j, x.meta, download_one(cx, x)?));
                        }
                    }
                    let fr = far.as_deref_mut().ok_or_else(|| pc_gpu::Error::Shape { what: "ccmm service", msg: "no remote lane".into() })?;
                    if split {
                        fr.ccmm_operands_far(&cx.ex.compute, cx.sh, side, &mut xs[h..NBIG], &hi(f), check).map_err(|e| ctx_err("remote operand stc_high (kept)", &[m0], e))?;
                    } else {
                        fr.stc_high_far(&cx.ex.compute, cx.sh, &mut xs[h..NBIG], &hi(f)).map_err(|e| ctx_err("remote operand stc_high", &[m0], e))?;
                    }
                    op_far = true;
                }
                Staged::stc_high(route, cx, &mut xs[..h], &hi(f)).map_err(|e| ctx_err("operand stc_high", &[m0], e))?;
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
        // (split: this device's half of the product needs none of the remote operands)
        if op_far && (!split || check) {
            let fr = far.as_deref_mut().ok_or_else(|| pc_gpu::Error::Shape { what: "ccmm service", msg: "no remote lane".into() })?;
            fr.join(&cx.ex.compute)?;
            // PC_REMOTE_CHECK: the remote operands converted again here from their saved inputs, word for word
            for (idx, m_in, data) in op_stash.drain(..) {
                let (side, i) = (idx / NBIG, idx % NBIG);
                let f = if side == 0 { fa } else { fb };
                let x = if side == 0 { &mut self.a[i] } else { &mut self.b[i] };
                let far_words = (x.meta, download_one(cx, x)?);
                let rows: Vec<usize> = (0..=pc_shadow::meta::top_limb(cx.ctx.shape, m_in.level)? as usize).collect();
                for (part, d) in data.iter().enumerate() {
                    x.buf.upload_rows(&cx.ex.compute, 0, part, &rows, d)?;
                }
                x.meta = m_in;
                Staged::stc_high(route, cx, std::slice::from_mut(x), &hi(f))?;
                let here_words = (x.meta, download_one(cx, x)?);
                if here_words != far_words {
                    return Err(pc_gpu::Error::Shape { what: "ccmm service", msg: format!("remote check: operand {i} of side {side} converted on the remote device differs from the same conversion here") });
                }
                pc_program::replay::REMOTE_CCMM_CHECKED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        self.mark(cx, 1)?;
        let (ctx, s) = (cx.ctx, &cx.ex.compute);
        let ph = self.phases.as_ref().map(|v| [&v[0], &v[1], &v[2], &v[3], &v[4], &v[5]]);
        let hb = NBIG / 2;
        if split {
            let fr = far.as_deref_mut().ok_or_else(|| pc_gpu::Error::Shape { what: "ccmm service", msg: "no remote lane".into() })?;
            // PC_REMOTE_CHECK: every operand word saved here (the remote ones came back with their conversion)
            let stash = if check { Some(self.a[..NBIG].iter().chain(&self.b[..NBIG]).map(|x| Ok((x.meta, download_one(cx, x)?))).collect::<pc_gpu::Result<Vec<_>>>()?) } else { None };
            fr.ccmm_mul_split(s, &mut self.ccmm, ctx, &mut self.a[..NBIG], &self.b[..NBIG], bt).map_err(|e| ctx_err("ccmm mul split (a)", &self.a[..NBIG].iter().map(|x| x.meta).collect::<Vec<_>>(), e))?;
            if let Some([h, t]) = fr.ccmm_split_phases()? {
                eprintln!("[ccmm-phases] split bt={} here ring_in {:.3} cmt {:.3} ppmm {:.3} post {:.3} ring_out {:.3} total {:.3} | there ring_in {:.3} cmt {:.3} ppmm {:.3} post {:.3} ring_out {:.3} total {:.3} ms", bt as u8, h[0], h[1], h[2], h[3], h[4], h[5], t[0], t[1], t[2], t[3], t[4], t[5]);
            }
            if let Some(st) = stash {
                // the remote outputs back here, every output word against the same product on this device alone
                fr.ccmm_far_back(s, 0, &mut self.a[hb..NBIG])?;
                let got = self.a[..NBIG].iter().map(|x| Ok((x.meta, download_one(cx, x)?))).collect::<pc_gpu::Result<Vec<_>>>()?;
                for (x, (m, data)) in self.a[..NBIG].iter_mut().chain(self.b[..NBIG].iter_mut()).zip(&st) {
                    let rows: Vec<usize> = (0..=pc_shadow::meta::top_limb(cx.ctx.shape, m.level)? as usize).collect();
                    for (part, d) in data.iter().enumerate() {
                        x.buf.upload_rows(&cx.ex.compute, 0, part, &rows, d)?;
                    }
                    x.meta = *m;
                }
                self.ccmm.mul_inplace(cx.ctx, &cx.ex.compute, &mut self.a[..NBIG], &self.b[..NBIG], bt, None)?;
                let want = self.a[..NBIG].iter().map(|x| Ok((x.meta, download_one(cx, x)?))).collect::<pc_gpu::Result<Vec<_>>>()?;
                if got != want {
                    let bad: Vec<usize> = (0..NBIG).filter(|&i| got[i] != want[i]).collect();
                    return Err(pc_gpu::Error::Shape { what: "ccmm service", msg: format!("remote check: split product outputs {bad:?} differ from the product on this device alone") });
                }
                pc_program::replay::REMOTE_CCMM_CHECKED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        } else {
            self.ccmm.mul_inplace(ctx, s, &mut self.a[..NBIG], &self.b[..NBIG], bt, ph).map_err(|e| ctx_err("ccmm mul (a)", &self.a[..NBIG].iter().map(|x| x.meta).collect::<Vec<_>>(), e))?;
            if let Some(ev) = ph {
                let t = pc_gpu::matmul::ccmm::CcmmTimes::read(ev)?;
                eprintln!("[ccmm-phases] ring_in {:.3} cmt {:.3} ppmm {:.3} cmt2_relin {:.3} ring_out {:.3} total {:.3} ms", t.ring_in, t.cmt, t.ppmm, t.cmt2_relin, t.ring_out, t.total());
            }
        }
        let n_out = if g > 1 {
            // group i: a[g i] + X^1024 (a[g i + 1] + X^1024 (...)), innermost first (split: the remote groups merge
            // there; the check merges them here too, from the local product)
            let local = if split && !check { hb / g } else { NBIG / g };
            for i in 0..local {
                for k in (0..g - 1).rev() {
                    let (lo, hi) = self.a.split_at_mut(g * i + k + 1);
                    pc_gpu::limbs::merge_monomial(cx.ctx, cx.ex, &self.mono, &mut lo[g * i + k], &hi[0])?;
                }
            }
            if split {
                far.as_deref_mut().ok_or_else(|| pc_gpu::Error::Shape { what: "ccmm service", msg: "no remote lane".into() })?.ccmm_far_merge(g)?;
            }
            // the merged outputs a[g i] -> a[i] (buffer handles swap, no copy)
            for i in 1..NBIG / g {
                self.a.swap(i, g * i);
            }
            NBIG / g
        } else {
            NBIG
        };
        // split: outputs n_out / 2 .. live on the remote device. One run of output refreshes whose remote part lies there
        // refreshes them where they are (`resident`: the first remote output it takes) after the rest come back here;
        // otherwise they all come back first
        let mut resident: Option<usize> = None;
        if split {
            let fr = far.as_deref_mut().ok_or_else(|| pc_gpu::Error::Shape { what: "ccmm service", msg: "no remote lane".into() })?;
            let h2 = n_out / 2;
            if check {
                // the remote merges against this device's (the remote outputs into b's buffers, free now)
                fr.ccmm_far_back(&cx.ex.compute, 0, &mut self.b[..h2])?;
                for j in 0..h2 {
                    if download_one(cx, &self.b[j])? != download_one(cx, &self.a[h2 + j])? || self.b[j].meta != self.a[h2 + j].meta {
                        return Err(pc_gpu::Error::Shape { what: "ccmm service", msg: format!("remote check: remote output {} (merge {g}) differs from this device's", h2 + j) });
                    }
                }
            }
            let one_run = (0..n_out).all(|i| a_meta[g * i] == a_meta[0]);
            let n_far = if n_out >= 2 { (n_out / 2).min(GRID_INPUTS) } else { 0 };
            let mid = n_out - n_far;
            if one_run && n_far > 0 && mid >= h2 {
                if mid > h2 {
                    fr.ccmm_far_back(&cx.ex.compute, 0, &mut self.a[h2..mid])?;
                }
                resident = Some(mid - h2);
            } else {
                fr.ccmm_far_back(&cx.ex.compute, 0, &mut self.a[h2..n_out])?;
            }
        }
        // output i comes from left operand i (merged: g i); runs of equal operand metadata share one SinCBoot call
        self.mark(cx, 2)?;
        let src = |i: usize| a_meta[g * i];
        let mut lo = 0;
        while lo < n_out {
            let m = src(lo);
            let mut hi = lo + 1;
            while hi < n_out && src(hi) == m {
                hi += 1;
            }
            let st = self.state(cx, route, &m, drop)?;
            let outs: Vec<CtMeta> = self.a[lo..hi].iter().map(|x| x.meta).collect();
            let (nf_split, factor) = (self.nf_split, 1.0 / kappa);
            // the run's last half on the remote device, issued first (it starts as soon as the product is done)
            let n_far = if far.is_some() && hi - lo >= 2 { ((hi - lo) / 2).min(GRID_INPUTS) } else { 0 };
            let mid = hi - n_far;
            let mut stash = Vec::new();
            if let (Some(lane), true) = (far.as_deref_mut(), n_far > 0) {
                if check {
                    stash = self.a[mid..hi].iter().map(|x| Ok((x.meta, download_one(cx, x)?))).collect::<pc_gpu::Result<Vec<_>>>()?;
                }
                let st2 = st.clone();
                if let Some(sl) = out_slots.filter(|sl| sl.len() >= hi && !lane.own_wanted().is_empty()) {
                    lane.own_next(sl[mid..hi].to_vec(), shift);
                }
                let mut xs: Vec<&mut Ct<'static, W>> = self.a[mid..hi].iter_mut().collect();
                match resident.filter(|_| lo == 0 && hi == n_out) {
                    Some(first) => lane.ccmm_far_singles(&cx.ex.compute, cx.sh, first, &mut xs, &mut |c1, r1, g| sinc_boot_scaled(c1, r1, &st2, std::slice::from_mut(g), nf_split, factor))?,
                    None => lane.stcf_singles(&cx.ex.compute, cx.sh, &mut xs, &mut |c1, r1, g| sinc_boot_scaled(c1, r1, &st2, std::slice::from_mut(g), nf_split, factor))?,
                }
            }
            sinc_boot_scaled(cx, route, &st, &mut self.a[lo..mid], nf_split, factor)
                .map_err(|e| ctx_err(&format!("SinCBoot outputs {lo}..{mid} (operand {m:?})"), &outs, e))?;
            if let (Some(lane), true) = (far.as_deref_mut(), n_far > 0) {
                lane.join(&cx.ex.compute)?;
                if check {
                    // PC_REMOTE_CHECK: the remote half again here from its saved inputs, against what came back
                    let got = self.a[mid..hi].iter().map(|x| Ok((x.meta, download_one(cx, x)?))).collect::<pc_gpu::Result<Vec<_>>>()?;
                    for (x, (m_in, data)) in self.a[mid..hi].iter_mut().zip(&stash) {
                        let rows: Vec<usize> = (0..=pc_shadow::meta::top_limb(cx.ctx.shape, m_in.level)? as usize).collect();
                        for (part, d) in data.iter().enumerate() {
                            x.buf.upload_rows(&cx.ex.compute, 0, part, &rows, d)?;
                        }
                        x.meta = *m_in;
                    }
                    sinc_boot_scaled(cx, route, &st, &mut self.a[mid..hi], nf_split, factor)?;
                    let want = self.a[mid..hi].iter().map(|x| Ok((x.meta, download_one(cx, x)?))).collect::<pc_gpu::Result<Vec<_>>>()?;
                    if got != want {
                        return Err(pc_gpu::Error::Shape { what: "ccmm service", msg: format!("remote check: SinCBoot outputs {mid}..{hi} refreshed remotely differ from the same outputs refreshed here") });
                    }
                    pc_program::replay::REMOTE_CCMM_CHECKED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
            lo = hi;
        }
        self.mark(cx, 3)?;
        if shift > 0 {
            for x in self.a[..n_out].iter_mut() {
                ccmm::leaf::mult_int(cx.ctx, &cx.ex.compute, &self.pow2, x, 1u64 << shift)?;
            }
        }
        self.mark(cx, 4)?;
        if let Some(ev) = self.prof.as_ref() {
            ev[4].sync()?;
            let ms = |a: usize, b: usize| ev[a].elapsed_ms(&ev[b]).map(f64::from);
            self.last_prof = Some([ms(0, 1)?, ms(1, 2)?, ms(2, 3)?, ms(3, 4)?]);
        }
        self.calls += 1;
        Ok(n_out)
    }
}

/// `ccmm::sinc_boot` with the StC stage-0 transform scaled by `factor` (its plaintexts: exact, no extra level).
impl CcmmRt {
    /// `PC_FINISH_PROFILE`: record phase event `i` on the compute lane.
    fn mark(&self, cx: &BtsCx<'_, 'static, W, M>, i: usize) -> pc_gpu::Result<()> {
        if let Some(ev) = self.prof.as_ref() {
            ev[i].record(&cx.ex.compute)?;
        }
        Ok(())
    }
}

/// Both polynomials' live rows of single ciphertext `x` (host; waits for the compute stream).
fn download_one(cx: &BtsCx<'_, 'static, W, M>, x: &Ct<'static, W>) -> pc_gpu::Result<[Vec<W>; 2]> {
    let rows: Vec<usize> = (0..=x.top(cx.ctx.shape)?).collect();
    Ok([x.buf.download_rows(&cx.ex.compute, 0, 0, &rows)?, x.buf.download_rows(&cx.ex.compute, 0, 1, &rows)?])
}

fn sinc_boot_scaled(cx: &mut BtsCx<'_, 'static, W, M>, route: &StcFirstRoute<'static, W, M>, st: &BtsState, cts: &mut [Ct<'static, W>], nf_split: f64, factor: f64) -> pc_gpu::Result<()> {
    use pc_gpu::bts::staged::{BtsCall, CallOpts, Phase, StagedRef};
    ccmm::enter_split(cx, route, cts, nf_split)?;
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
    pub fbs: Option<&'r mut crate::fbs::FbsRt>,
    pub route: &'r StcFirstRoute<'static, W, M>,
}

impl<'r> FinishExec<'static, W, M> for FinishSvc<'r> {
    fn finish(&mut self, cx: &mut BtsCx<'_, 'static, W, M>, f: &FinishStep, slots: &mut [Ct<'static, W>], remote: Option<&mut dyn pc_program::replay::RemoteLane<'static, W, M>>) -> pc_program::Result<()> {
        if f.desc.contains("\"fbs\"") && crate::fbs::Desc::parse(&f.desc).is_some() {
            let rt = self.fbs.as_deref_mut().ok_or_else(|| ferr(f, "this session has no FBS service (no program named an fbs finish at setup)"))?;
            return crate::fbs::FbsExec { rt, route: self.route }.finish(cx, f, slots, None).map_err(|e| match e {
                pc_program::Error::Replay { .. } => e,
                e => ferr(f, e.to_string()),
            });
        }
        if crate::tfhe::Desc::parse(&f.desc).is_some() {
            let rt = self.tfhe.as_deref_mut().ok_or_else(|| ferr(f, "this session has no TFHE service (Session(tfhe=...))"))?;
            return crate::tfhe::TfheExec { rt, route: self.route }.finish(cx, f, slots, None).map_err(|e| match e {
                pc_program::Error::Replay { .. } => e,
                e => ferr(f, e.to_string()),
            });
        }
        if let Some(d) = CcmmDesc::parse(&f.desc) {
            let rt = self.ccmm.as_deref_mut().ok_or_else(|| ferr(f, "this session has no CCMM service (Session(ccmm=True))"))?;
            let c_unit = rt.c_unit.ok_or_else(|| ferr(f, "CCMM not calibrated (Session.ccmm_set)"))?;
            if f.ins.len() != 2 * NBIG || f.outs.len() != d.n_out || d.n_out != NBIG / d.merge.max(1) as usize {
                return Err(ferr(f, format!("{} inputs / {} outputs compiled for n_out {} (merge {})", f.ins.len(), f.outs.len(), d.n_out, d.merge)));
            }
            for (i, &sl) in f.ins.iter().enumerate() {
                let dst = if i < NBIG { &mut rt.a[i] } else { &mut rt.b[i - NBIG] };
                copy_ct(cx.ctx, &cx.ex.compute, dst, &slots[sl as usize])?;
            }
            let (fa, fb) = d.factors(c_unit);
            let kappa = if d.is_pv() { rt.kappa.1 } else { rt.kappa.0 };
            let n = rt.run(cx, self.route, d.b_transposed, fa, fb, d.shift, d.merge, kappa, d.raise_drop, remote, Some(&f.outs)).map_err(|e| ferr(f, e.to_string()))?;
            if let Some(p) = rt.last_prof.take() {
                eprintln!("[finprof] ccmm {} n_out {} merge {} operand_stc {:.3} product {:.3} refresh {:.3} shift {:.3}", d.site, n, d.merge, p[0], p[1], p[2], p[3]);
            }
            for m in 0..n {
                copy_ct(cx.ctx, &cx.ex.compute, &mut slots[f.outs[m] as usize], &rt.a[m])?;
            }
            return Ok(());
        }
        let rt = self.pcmm.as_deref_mut().ok_or_else(|| ferr(f, "this session has no PCMM service (Session(pcmm=...))"))?;
        PcmmExec { rt, route: self.route }.finish(cx, f, slots, remote).map_err(|e| match e {
            pc_program::Error::Replay { .. } => e,
            e => ferr(f, e.to_string()),
        })
    }

    fn owned(&self, f: &FinishStep) -> Vec<u32> {
        if crate::tfhe::Desc::parse(&f.desc).is_some() {
            return vec![];
        }
        if let Some(d) = CcmmDesc::parse(&f.desc) {
            // the last half of the outputs' run (one run: the operands at one metadata), as `CcmmRt::run` splits it
            let n = d.n_out.min(f.outs.len());
            if !crate::pcmm::remote_finish_ccmm() || n < 2 {
                return vec![];
            }
            return f.outs[n - (n / 2).min(GRID_INPUTS)..n].to_vec();
        }
        self.pcmm.as_deref().map_or_else(Vec::new, |rt| crate::pcmm::owned_far(rt, f))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desc() {
        let d = CcmmDesc::parse(r#"{"op": "ccmm", "site": "b0.qk", "out": 3, "n_out": 8, "b_transposed": true, "merge": true, "prepared_b": false, "shift": 2, "amax": 4.0, "bmax": 2.0, "out_scale": 0.5}"#).unwrap();
        assert_eq!((d.n_out, d.b_transposed, d.merge, d.shift), (8, true, 2, 2));
        let d4 = CcmmDesc::parse(r#"{"op": "ccmm", "site": "b0.pv", "out": 0, "n_out": 4, "merge": 4}"#).unwrap();
        assert_eq!((d4.n_out, d4.merge), (4, 4));
        let d1 = CcmmDesc::parse(r#"{"op": "ccmm", "site": "b0.pv", "out": 0, "n_out": 16, "merge": false}"#).unwrap();
        assert_eq!(d1.merge, 1);
        assert!(CcmmDesc::parse(r#"{"op": "pcmm", "linear": "x"}"#).is_none());
        let site = |s: &str| CcmmDesc::parse(&format!(r#"{{"op": "ccmm", "site": "{s}", "n_out": 8}}"#)).unwrap().is_pv();
        assert!(site("pv3") && site("mpv3") && site("mpv11.1.0"));
        assert!(!site("qk3") && !site("mqk3") && !site("mqk0.0.1") && !site("b0.qk"));
        let (fa, fb) = d.factors(2.0);
        let p: f64 = 0.5 * 0.25 / 2.0 * 0.5;
        assert!((fa * fb - p).abs() < 1e-15 && (fa / fb - 2.0 / 4.0).abs() < 1e-15);
    }
}
