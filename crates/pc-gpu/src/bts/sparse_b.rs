//! SparseB precomputation of one sparse route (FIDESlib `SparseB.cu` + `RawCiphertext.cu::buildSparseB`, eprint
//! 2026/1023 Alg. 4/5): the repetition-aware depth-1 CtS and StC of `slots` = n/2 complex slots repeated r = N / n
//! times, as one BSGS stage each (s = 2n / r diagonals, baby step min(s, 16)), followed by the block partial sums.
//! One precomputation per (slots, raise drop); route: raise -> fold -> CtS + Accumulate -> real EvalMod -> rescale ->
//! StC + Accumulate + rotate-add -> x 2^correction. Plaintexts are encoded at setup straight at the level they are
//! used (one rounding; FIDESlib re-levels from the top, two), so device words differ from FIDESlib's by that
//! rounding. Details: docs/ARCHITECTURE.md "pc-gpu SparseB".
use super::cts::{tail_steps, KeyTails, LtEnd, LtGeom, LtPts, LtStage, PtSource, LT_REGEN_DIGITS};
use super::evalmod::{em_work, EmMode, EvalModPlan};
use super::raise::RaisePlan;
use super::{BtsKeyRing, BtsWorkShape};
use crate::context::{Context, DevStrategy, DevWord};
use crate::error::{shape, Result};
use crate::poly::Layout;
use pc_core::params::Chain;
use pc_core::scale::ScaleTables;
use crate::ks::{KeyGroup, KsKeyRef, KskSource};
use pc_core::bts_setup::{sparse_b_diagonals, sparse_b_geometry, SparseBGeom, SPARSE_B_CHEB_K24, SPARSE_B_DA_ITS};
use pc_cuda::{Arena, Compute, Stream};
use pc_shadow::evalmod::PsPlan;
use pc_shadow::fides::FMeta;
use std::ffi::c_void;

/// Knobs of a SparseB precomputation (the `bts` part of the Config the route reads).
#[derive(Clone, Debug, PartialEq)]
pub struct SparseBCfg {
    pub slots: u32,
    pub raise_drop: u32,
    /// FIDESLIB_BTS_SHIFT_T: the exact post-raise constant times 2^t; CtS carries 2^-t.
    pub cts0_t: u32,
    /// K of the raise's EvalMod normalization (the dense route's, 16): CtS carries K_raise / 24.
    pub raise_k: f64,
    /// FIDESLIB_ACCUM_BSTEP (rotations per accumulate round + 1).
    pub acc_bstep: usize,
    /// FIDESLIB_BTS_SPARSE_B_BSTEP (0 = min(s, 16)).
    pub b_step: u32,
    pub k_max: usize,
    pub ks_source: KskSource,
    /// As `BtsSetup::lt_tails` / `em_fuse_rescale`.
    pub lt_tails: bool,
    pub em_fuse_rescale: bool,
    /// K of the EvalMod series: 24 (FIDESlib's degree-20 table) or 16 (the dense route's degree-14 table); the overflow
    /// integer after the fold has the dense route's distribution.
    pub evalmod_k: f64,
}

impl SparseBCfg {
    pub fn n32(slots: u32, raise_drop: u32) -> Self {
        SparseBCfg { slots, raise_drop, cts0_t: 12, raise_k: pc_core::bts_setup::EVALMOD_K, acc_bstep: 4, b_step: 0, k_max: 8, ks_source: KskSource::Regen, lt_tails: true, em_fuse_rescale: false, evalmod_k: pc_core::bts_setup::SPARSE_B_K }
    }

    /// The EvalMod series' Paterson-Stockmeyer plan for `evalmod_k`.
    pub fn em_series(&self) -> Result<PsPlan> {
        if self.evalmod_k == pc_core::bts_setup::EVALMOD_K {
            Ok(PsPlan::from_coeffs(pc_core::bts_setup::evalmod_coeffs())?)
        } else if self.evalmod_k == pc_core::bts_setup::SPARSE_B_K {
            ps_k24()
        } else {
            Err(shape("sparse_b", format!("no EvalMod series for K = {}", self.evalmod_k)))
        }
    }
}

/// Accumulate rounds as key groups (steps, group).
pub struct AccRounds<'a> {
    pub rounds: Vec<(Vec<i64>, KeyGroup<'a>)>,
}

impl<'a> AccRounds<'a> {
    pub fn new<S: pc_cuda::Lane>(
        arena: &'a Arena,
        s: &Stream<S>,
        n: usize,
        keys: &dyn BtsKeyRing,
        stage: Option<(*const c_void, *const c_void)>,
        rounds: Vec<Vec<i64>>,
    ) -> Result<Self> {
        let mut v = Vec::with_capacity(rounds.len());
        for r in rounds {
            let refs: Vec<&dyn KsKeyRef> = r.iter().map(|&st| keys.rot(st)).collect::<Result<_>>()?;
            let g = KeyGroup::new(arena, s, n, &refs, stage)?;
            v.push((r, g));
        }
        Ok(AccRounds { rounds: v })
    }
    pub fn bytes(rounds: &[Vec<i64>]) -> usize {
        rounds.iter().map(|r| KeyGroup::bytes(r.len(), true)).sum()
    }
    pub fn steps(&self) -> Vec<i64> {
        self.rounds.iter().flat_map(|r| r.0.iter().copied()).collect()
    }
}

/// The device precomputation of one SparseB route variant.
pub struct SparseBPrecomp<'a, W: DevWord> {
    pub cfg: SparseBCfg,
    pub geom: SparseBGeom,
    pub raise: RaisePlan<'a, W>,
    pub fold: AccRounds<'a>,
    pub cts: LtStage<'a, W>,
    pub cts_acc: AccRounds<'a>,
    pub em: EvalModPlan<'a, W>,
    pub stc: LtStage<'a, W>,
    pub stc_acc: AccRounds<'a>,
    /// rotate(n/2)-add closing the StC (one key).
    pub half: AccRounds<'a>,
    pub cts0_const: f64,
    /// Bytes carved by `new` (the MemPlan assert compares it with `bytes`) and the work shape the route needs.
    pub carved: usize,
    pub work: BtsWorkShape,
    /// Carved bytes per component (setup report; `bytes_parts` predicts the same list).
    pub parts: Vec<(&'static str, usize)>,
}

/// The rotation rounds of the route (fold, CtS accumulate, StC accumulate, the half rotation), from the geometry.
pub fn rounds(n: usize, cfg: &SparseBCfg, g: &SparseBGeom) -> [Vec<Vec<i64>>; 4] {
    let acc = super::spru::accumulate_rounds;
    let s = cfg.slots as usize;
    [acc(cfg.acc_bstep, s, n / 2 / s), acc(cfg.acc_bstep, g.s as usize, g.r as usize / 4), acc(cfg.acc_bstep, g.n as usize, g.r as usize / 2), vec![vec![g.n as i64 / 2]]]
}

/// Every rotation step the route reads (keys to generate / import).
pub fn rot_steps(n: usize, cfg: &SparseBCfg) -> Result<Vec<i64>> {
    let g = geometry(n, cfg)?;
    let mut v: Vec<i64> = rounds(n, cfg, &g).into_iter().flatten().flatten().collect();
    v.extend(1..g.b_step as i64);
    if g.g_step > 1 {
        v.push(g.b_step as i64);
    }
    v.sort();
    v.dedup();
    Ok(v)
}

pub fn geometry(n: usize, cfg: &SparseBCfg) -> Result<SparseBGeom> {
    sparse_b_geometry(n as u32, cfg.slots, cfg.b_step).ok_or_else(|| shape("sparse_b", format!("no SparseB geometry for {} slots (n <= r/2)", cfg.slots)))
}

/// OpenFHE / FIDESlib post-raise constant pre / (K N), pre = sf(raise top) / q0 (composite chains).
pub fn eval_const(st: &pc_core::scale::ScaleTables, q: &[u64], d: usize, n: usize, top: usize, k: f64) -> Result<f64> {
    let mut qd = q[0] as f64;
    for &x in q.iter().take(d).skip(1) {
        qd *= x as f64;
    }
    Ok(st.sf(top as i32)? / qd * (1.0 / (k * n as f64)))
}

/// The K = 24 EvalMod series' Paterson-Stockmeyer plan.
pub fn ps_k24() -> Result<PsPlan> {
    Ok(PsPlan::from_coeffs(&SPARSE_B_CHEB_K24)?)
}

impl<'a, W: DevWord> SparseBPrecomp<'a, W> {
    /// Device bytes and work shape of the variant (computed by building the host side; `new` asserts the carve).
    pub fn bytes(chain: &Chain, st: &ScaleTables, lay: &Layout, cfg: &SparseBCfg) -> Result<(usize, BtsWorkShape)> {
        let lay = *lay;
        let n = lay.n;
        let g = geometry(n, cfg)?;
        let tops = Self::tops(chain, st, cfg)?;
        let geo = Self::lt_geom(&g);
        let parts = Self::bytes_parts(chain, st, &lay, cfg)?;
        let b: usize = parts.iter().map(|p| p.1).sum();
        let (_, _, nslots) = EvalModPlan::<W>::program(st, tops.1, &cfg.em_series()?, SPARSE_B_DA_ITS)?;
        let (em_n, em_kk) = em_work(nslots, cfg.k_max, EmMode::Real);
        Ok((b, BtsWorkShape { k_max: cfg.k_max, g_max: geo.g, em_slots: em_n, em_kk, spare: 1 }))
    }

    /// Predicted bytes per component, in carve order (raise, fold, cts, cts_acc, em, stc, stc_acc+half).
    pub fn bytes_parts(chain: &Chain, st: &ScaleTables, lay: &Layout, cfg: &SparseBCfg) -> Result<Vec<(&'static str, usize)>> {
        let lay = *lay;
        let n = lay.n;
        let g = geometry(n, cfg)?;
        let primes = chain.primes();
        let tops = Self::tops(chain, st, cfg)?;
        let rr = rounds(n, cfg, &g);
        let geo = Self::lt_geom(&g);
        let lt = |top: usize| -> Result<usize> {
            Ok(LtPts::<W>::coeffs_bytes(geo.npts, top + 1 + lay.kp, n, n) + LtStage::<W>::plan_bytes(&lay, &primes, &geo, top, LtEnd::ModDown, cfg.k_max)?)
        };
        Ok(vec![
            ("raise", RaisePlan::<W>::table_bytes()),
            ("fold", AccRounds::bytes(&rr[0])),
            ("cts", lt(tops.0)?),
            ("cts_acc", AccRounds::bytes(&rr[1])),
            ("em", Self::em_bytes(st, &lay, tops.1, cfg)?),
            ("stc", lt(tops.2)?),
            ("stc_acc+half", AccRounds::bytes(&rr[2]) + AccRounds::bytes(&rr[3])),
        ])
    }

    /// The CtS baby steps of the variants `cfgs` whose keys get a resident `a` tail (`KeyTails` from digit
    /// `LT_REGEN_DIGITS` on). One `KeyTails` of these steps serves every variant and slot count: a tail belongs to its
    /// key, and the raise variants of a route share their CtS keys.
    pub fn tail_steps(chain: &Chain, st: &ScaleTables, cfgs: &[SparseBCfg]) -> Result<Vec<i64>> {
        let mut steps = Vec::new();
        for cfg in cfgs.iter().filter(|c| c.lt_tails) {
            let g = geometry(chain.n, cfg)?;
            steps.extend(tail_steps(chain, &[(Self::lt_geom(&g), Self::tops(chain, st, cfg)?.0)], LT_REGEN_DIGITS));
        }
        steps.sort();
        steps.dedup();
        Ok(steps)
    }

    /// Device bytes of the shared tails of `cfgs` (`tail_steps`).
    pub fn tails_bytes(chain: &Chain, st: &ScaleTables, lay: &Layout, cfgs: &[SparseBCfg]) -> Result<usize> {
        Ok(KeyTails::<W>::bytes(lay, chain.dnum(), LT_REGEN_DIGITS, Self::tail_steps(chain, st, cfgs)?.len()))
    }

    /// The shared tails of `cfgs`, expanded from the keys' seeds (setup; `tails_bytes`).
    pub fn tails<M: DevStrategy<W>>(arena: &'a Arena, s: &Stream<Compute>, ctx: &Context<'_, W, M>, keys: &dyn BtsKeyRing, cfgs: &[SparseBCfg]) -> Result<KeyTails<'a, W>> {
        KeyTails::new(arena, s, ctx, keys, &Self::tail_steps(&ctx.chain, &ctx.scale, cfgs)?, LT_REGEN_DIGITS)
    }

    /// The EvalMod plan's tables as `EvalModPlan::new` carves them.
    fn em_bytes(st: &ScaleTables, lay: &Layout, input: FMeta, cfg: &SparseBCfg) -> Result<usize> {
        EvalModPlan::<W>::table_bytes(st, lay, input, &cfg.em_series()?, SPARSE_B_DA_ITS, cfg.k_max)
    }

    fn lt_geom(g: &SparseBGeom) -> LtGeom {
        LtGeom { b: g.b_step as usize, g: g.g_step as usize, stride: 1, npts: g.s as usize, offset: 0 }
    }

    /// (CtS top, EvalMod input meta, StC top) of the variant.
    fn tops(chain: &Chain, st: &ScaleTables, cfg: &SparseBCfg) -> Result<(usize, FMeta, usize)> {
        let d = chain.level_unit;
        let top = chain.q.len() - 1 - d * cfg.raise_drop as usize;
        let raised_nf = st.sf(top as i32)? * (cfg.cts0_t as f64).exp2();
        let pt_nf = st.sf(top as i32)? * (-(cfg.cts0_t as f64)).exp2();
        let em_in = FMeta { top: top as i32, nl: 2, nf: raised_nf * pt_nf };
        let (_, out, _) = EvalModPlan::<W>::program(st, em_in, &cfg.em_series()?, SPARSE_B_DA_ITS)?;
        let stc_top = if out.nl == 2 { out.top as usize - d } else { out.top as usize };
        Ok((top, em_in, stc_top))
    }

    /// Build the variant: raise tables, diagonals (encoded at their stage levels), accumulate key groups, EvalMod.
    /// `tails`: the routes' shared CtS key tails (`tail_steps` of every variant built at setup).
    #[allow(clippy::too_many_arguments)]
    pub fn new<M: DevStrategy<W>>(
        arena: &'a Arena,
        s: &Stream<Compute>,
        ctx: &Context<'_, W, M>,
        cfg: SparseBCfg,
        keys: &dyn BtsKeyRing,
        tails: &KeyTails<'_, W>,
        stage: Option<(*const c_void, *const c_void)>,
    ) -> Result<Self> {
        // carves are measured between aligned arena offsets (every allocation starts aligned)
        let align = Arena::bytes_for::<u8>(1);
        let used0 = arena.used_bytes().div_ceil(align) * align;
        let lay = ctx.lay;
        let n = lay.n;
        let d = ctx.chain.level_unit;
        let g = geometry(n, &cfg)?;
        let st = &ctx.scale;
        let (top, em_in, stc_top) = Self::tops(&ctx.chain, &ctx.scale, &cfg)?;
        let cts0_const = eval_const(st, &ctx.chain.q, d, n, top, cfg.raise_k)?;
        let mut parts: Vec<(&'static str, usize)> = vec![];
        let mark = |what: &'static str, parts: &mut Vec<(&'static str, usize)>| {
            let u = arena.used_bytes().div_ceil(align) * align - used0 - parts.iter().map(|p| p.1).sum::<usize>();
            parts.push((what, u));
        };
        let raise = RaisePlan::new(arena, s, ctx, top, cts0_const, cfg.cts0_t)?;
        mark("raise", &mut parts);
        let rr = rounds(n, &cfg, &g);
        let [r_fold, r_cts, r_stc, r_half] = rr;
        let fold = AccRounds::new(arena, s, n, keys, stage, r_fold)?;
        mark("fold", &mut parts);
        // diagonals: gc = K_raise / K * 2^-t (the shifted raise carries 2^t), gd = q0 / sf(L) * fdrop: a raise to a
        // lower top carries sf(L) / sf(raise top) (as the dense routes' last StC stage; 1 at drops 0 and 1, a 4-8 %
        // output gain error at drops 2-5 without it)
        let shift_t = (-(cfg.cts0_t as f64)).exp2();
        let gc = cfg.raise_k / cfg.evalmod_k * shift_t;
        let mut q0 = ctx.chain.q[0] as f64;
        for j in 1..d {
            q0 *= ctx.chain.q[j] as f64;
        }
        let fdrop = st.sf((lay.nq - 1) as i32)? / st.sf(top as i32)?;
        let gd = q0 / st.sf((lay.nq - 1) as i32)? * fdrop;
        let (p, q) = sparse_b_diagonals(n as u32, cfg.slots, &g, gc, gd);
        let geo = Self::lt_geom(&g);
        let sf_c = st.sf(top as i32)?;
        let mut cts_coeffs = super::cts::diag_coeffs(&ctx.chain, &p, sf_c)?;
        super::cts::emu_thin(&mut cts_coeffs, "SBCTS", 0, sf_c * shift_t, 0.0);
        let cts_pts = LtPts::new(arena, s, ctx, top, PtSource::Coeffs(&cts_coeffs))?;
        let steps = if cfg.lt_tails { tail_steps(&ctx.chain, &[(geo, top)], LT_REGEN_DIGITS) } else { Vec::new() };
        let cts = LtStage::new(arena, s, ctx, geo, top, LtEnd::ModDown, sf_c * shift_t, cts_pts, &tails.ring_of(keys, n, &steps)?, stage, cfg.k_max)?;
        mark("cts", &mut parts);
        let cts_acc = AccRounds::new(arena, s, n, keys, stage, r_cts)?;
        mark("cts_acc", &mut parts);
        let mut em = EvalModPlan::new(arena, s, ctx, EmMode::Real, em_in, &cfg.em_series()?, SPARSE_B_DA_ITS, keys.conj()?, keys.relin()?, stage, cfg.k_max, KskSource::Regen)?;
        em.fuse_rescale = cfg.em_fuse_rescale;
        mark("em", &mut parts);
        let sf_s = st.sf(stc_top as i32)?;
        let mut stc_coeffs = super::cts::diag_coeffs(&ctx.chain, &q, sf_s)?;
        super::cts::emu_thin(&mut stc_coeffs, "SBSTC", 0, sf_s, 0.0);
        let stc_pts = LtPts::new(arena, s, ctx, stc_top, PtSource::Coeffs(&stc_coeffs))?;
        let stc = LtStage::new(arena, s, ctx, geo, stc_top, LtEnd::ModDown, sf_s, stc_pts, keys, stage, cfg.k_max)?;
        mark("stc", &mut parts);
        let stc_acc = AccRounds::new(arena, s, n, keys, stage, r_stc)?;
        let half = AccRounds::new(arena, s, n, keys, stage, r_half)?;
        mark("stc_acc+half", &mut parts);
        s.sync()?;
        let carved = arena.used_bytes().div_ceil(align) * align - used0;
        let nslots = em.nslots;
        let (em_n, em_kk) = em_work(nslots, cfg.k_max, EmMode::Real);
        let work = BtsWorkShape { k_max: cfg.k_max, g_max: g.g_step as usize, em_slots: em_n, em_kk, spare: 1 };
        Ok(SparseBPrecomp { cfg, geom: g, raise, fold, cts, cts_acc, em, stc, stc_acc, half, cts0_const, carved, work, parts })
    }
}
