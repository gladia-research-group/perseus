//! Bootstrap routes, their landings, and the per-stage encoding rule of the CtS/StC linear transforms.
use crate::error::{Error, Result};
use crate::meta::{effective_level, ChainShape};
use crate::quirks::{Quirk, Quirks};
use pc_core::scale::ScaleTables;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One bootstrap call's parameters (in place of FIDESlib's global scopes FoldBtsScope, CorrectionScope,
/// RaiseScope, RealScope, ShippedScope).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Route {
    /// Sparse slot count; 0 = dense.
    pub slots: u32,
    /// Correction factor; None = the chain default.
    #[serde(default)]
    pub cf: Option<u32>,
    /// Real-payload route (one EvalMod chain; FIDESLIB_BTS_REAL).
    #[serde(default)]
    pub real: bool,
    /// Raise below the chain top by this many CKKS levels (FIDESLIB_BTS_RAISE_DROP): lands that much deeper.
    #[serde(default)]
    pub raise_drop: u32,
    /// StC-first order (FIDESLIB_BTS_STC_FIRST): StC paid on the input side.
    #[serde(default)]
    pub stc_first: bool,
    /// SparseB depth-1 linear transforms (FIDESLIB_BTS_SPARSE_B).
    #[serde(default)]
    pub sparse_b: bool,
    /// SPRU sparse-secret Hamming weight (FIDESLIB_SPRU); 0 = off.
    #[serde(default)]
    pub spru_h: u32,
    #[serde(default = "one")]
    pub iterations: u32,
}

fn one() -> u32 {
    1
}

impl Route {
    pub fn dense() -> Route {
        Route { slots: 0, cf: None, real: false, raise_drop: 0, stc_first: false, sparse_b: false, spru_h: 0, iterations: 1 }
    }
    pub fn sparse(slots: u32) -> Route {
        Route { slots, ..Route::dense() }
    }
}

/// Where a refresh leaves a ciphertext (absolute level, degree).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Landing {
    pub level: u32,
    pub deg: u8,
}

fn staged_n32() -> Landing {
    Landing { level: 28, deg: 1 }
}

/// A refresh the program issues outside the plan's sites: the outputs of a matrix product (a `Finish` op), refreshed
/// on the StC-first dense route whatever route the plan's own dense sites take (the StC is paid on the staged input
/// side), each output under its own raise drop.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Finish {
    /// A PCMM output (perseus pcmm.py; drops PF_PCMM_RAISE / PF_PCMM_RAISE_HI): the route's EvalMod output,
    /// unrealized (degree 2).
    Pcmm,
    /// A CCMM output (ccmm.py SinCBoot; drops PF_CCMM_RAISE): one StC stage past the route's landing, degree 1
    /// (pc-gpu `ccmm::stage0_level`).
    Ccmm,
}

impl Finish {
    /// The finish a descriptor's `op` names (`args.external.op` of a capture).
    pub fn of_op(op: &str) -> Option<Finish> {
        match op {
            "pcmm" => Some(Finish::Pcmm),
            "ccmm" => Some(Finish::Ccmm),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BtsConfig {
    /// The planner's bootstrap level: the dense landing every consumed depth is measured from (recipe BTS_LEVEL).
    pub bootstrap_level: u32,
    /// The reactive ceiling (AUTO_BTS_LEVEL): a product reaching it refreshes.
    pub level_limit: u32,
    /// The depth guard (BTS_MAX_INPUT_LEVEL): a refresh started deeper returns garbage, refused.
    pub max_input_level: u32,
    /// Landing per route slot count (0 = dense), realized (degree 1) as planned refreshes land.
    pub landings: BTreeMap<u32, Landing>,
    /// StC-first: the deepest EFFECTIVE input level with room for the StC levels; deeper inputs take the shipped
    /// order and land at `stc_first_fallback` (PLAN_STC_FIRST_ENTRY / PLAN_STC_FIRST_FALLBACK).
    #[serde(default)]
    pub stc_first_entry: Option<u32>,
    #[serde(default)]
    pub stc_first_fallback: Option<Landing>,
    /// The shipped-order (StC-last) dense landing, for a dense refresh that does not run StC-first while
    /// `landings[0]` is the StC-first one; None = `landings[0]` serves both orders.
    #[serde(default)]
    pub dense_shipped: Option<Landing>,
    /// Correction-factor window (min = the chain's deg floor FIRST_MOD_BITS - BTP_SCALE_BITS, max = PLAN_CF_MAX).
    pub cf_window: (u32, u32),
    /// Raise drops the runtime builds variants for (FIDESLIB_BTS_RAISE_DROPS); a SPRU route (`Route::spru_h`) takes
    /// any drop.
    #[serde(default)]
    pub raise_drops: Vec<u32>,
    /// Ciphertexts per batched bootstrap launch.
    pub k_max: u32,
    /// The StC-first route's landing for the staged finishes (`finish_landing`), raise drop 0: the measured n32
    /// StC-first landing (28, realized; route table `landings.stcf`), or the raw degree-2 form a fake capture lands
    /// at (26, 2). Kept apart from `landings[0]`, which follows the order the plan's own dense sites run in.
    #[serde(default = "staged_n32")]
    pub staged: Landing,
    /// The deepest effective level a refresh input may take with the raise's rescales still on the chain (the top of
    /// the level above q0's); None = two composite levels above the bottom, `n_q - 2 unit`.
    #[serde(default)]
    pub input_room: Option<u32>,
}

impl BtsConfig {
    /// The shipped n32 recipe (GPT-2 decode PLAN_CMD.txt + runtime line): StC-first dense lands at 28, the shipped
    /// order at 34, sparse 512 at 26 and 1 at 16 (PLAN_SPARSE_BTS_OUT; Llama adds 256 at 24), entry 46, ceiling 44,
    /// guard 48, CF 2..20, raise variants {1,3,5}.
    pub fn n32_shipped() -> BtsConfig {
        let l = |level| Landing { level, deg: 1 };
        BtsConfig {
            bootstrap_level: 28,
            level_limit: 44,
            max_input_level: 48,
            landings: [(0, l(28)), (512, l(26)), (256, l(24)), (1, l(16))].into_iter().collect(),
            stc_first_entry: Some(46),
            stc_first_fallback: Some(l(34)),
            dense_shipped: None,
            cf_window: (2, 20),
            raise_drops: vec![1, 3, 5],
            k_max: 8,
            staged: staged_n32(),
            input_room: None,
        }
    }

    /// Where a finish's output lands under `raise_drop`: the staged route's EvalMod output (`staged`, unrealized),
    /// `raise_drop` CKKS levels deeper (the variant raises that much below the chain top), plus the CCMM's StC
    /// stage. The finishes' inputs are capped at the StC-first entry by the code (pcmm.py / ccmm.py
    /// `bootstrap_hint` + `level_hint`), so no fallback applies. Every drop a finish uses gets its variant
    /// (pc-py `route_needs`), so `raise_drops` does not bound it.
    pub fn finish_landing(&self, chain: ChainShape, finish: Finish, raise_drop: u32) -> Landing {
        let u = chain.unit;
        let s = self.staged;
        let raw = if s.deg == 1 { Landing { level: s.level - u, deg: 2 } } else { s };
        let base = match finish {
            Finish::Pcmm => raw,
            Finish::Ccmm => Landing { level: effective_level(raw.level, raw.deg, u) + u, deg: 1 },
        };
        Landing { level: base.level + raise_drop * u, deg: base.deg }
    }

    /// The landing of a refresh of `input` on `route` (degree as listed; a refresh's own realize is part of it).
    pub fn landing(&self, chain: ChainShape, quirks: &Quirks, route: &Route, in_level: u32, in_deg: u8) -> Result<Landing> {
        if in_level > self.max_input_level {
            return Err(Error::Bts {
                knob: "max_input_level",
                msg: format!("refresh input level {in_level} > {} (BTS_MAX_INPUT_LEVEL): returns garbage", self.max_input_level),
            });
        }
        if let Some(cf) = route.cf {
            if cf < self.cf_window.0 || cf > self.cf_window.1 || cf == 1 {
                return Err(Error::Bts {
                    knob: "cf",
                    msg: format!("correction factor {cf} outside [{}, {}]", self.cf_window.0, self.cf_window.1),
                });
            }
        }
        // SPRU has no plaintext set per drop: the runtime builds a variant for every drop a program uses
        if route.raise_drop > 0 && route.spru_h == 0 && !self.raise_drops.contains(&route.raise_drop) {
            return Err(Error::Bts {
                knob: "raise_drops",
                msg: format!("raise drop {} has no variant (built: {:?})", route.raise_drop, self.raise_drops),
            });
        }
        let mut land = *self.landings.get(&route.slots).ok_or_else(|| Error::Bts {
            knob: "landings",
            msg: format!("no landing for route slots={}", route.slots),
        })?;
        if route.slots == 0 && !route.stc_first {
            if let Some(s) = self.dense_shipped {
                land = s;
            }
        }
        if route.slots == 0 && route.stc_first {
            if let (Some(entry), Some(fb)) = (self.stc_first_entry, self.stc_first_fallback) {
                let lv = if quirks.has(Quirk::CapStcEntryNominal) {
                    in_level
                } else {
                    effective_level(in_level, in_deg, chain.unit)
                };
                if lv > entry {
                    land = fb;
                }
            }
        }
        land.level += route.raise_drop * chain.unit;
        Ok(land)
    }
}

/// CKKS levels EvalMod consumes on the shipped n32 series: K = 16 degree 14 (dense) and K = 24 degree 20 (SparseB, the
/// classic sparse routes), each with 5 double angles (`evalmod::build_evalmod`: input -> output 20 primes on d = 2).
pub const EVALMOD_LEVELS: u32 = 10;

/// The realized landing of a multi-stage route in the shifted flow (FIDESLIB_BTS_SHIFT: the raise keeps the top
/// level, no adjust level): `budget` = {CtS, StC} stages, one CKKS level each, plus EvalMod, raise drop 0 (the caller
/// adds `raise_drop * unit`, as `BtsConfig::landing`). On n32: classic sparse routes {3, 3} at 32, the dense shipped
/// order {4, 3} at 34, SparseB's depth-1 transforms ({1, 1}) at 24.
pub fn multi_stage_landing(budget: [u32; 2], unit: u32) -> Landing {
    Landing { level: unit * (budget[0] + EVALMOD_LEVELS + budget[1]), deg: 1 }
}

/// FIDESLIB_CTS_ALT_BUDGET (`cts_alt`, 0 = off): a raise variant of drop >= 1 runs the alternative CtS of
/// `budget[0] + 1` stages, raised one level higher, spending that level on the extra stage. Returns the variant's
/// level budget and the drop it raises at; its landing is the drop's own (`multi_stage_landing` of the two agree,
/// tested), so the lever changes stage shapes and precision, never a landing.
pub fn cts_alt_variant(budget: [u32; 2], cts_alt: u32, raise_drop: u32) -> Result<([u32; 2], u32)> {
    match cts_alt {
        0 => Ok((budget, raise_drop)),
        b if b == budget[0] + 1 => Ok(if raise_drop >= 1 { ([b, budget[1]], raise_drop - 1) } else { (budget, 0) }),
        b => Err(Error::Bts { knob: "cts_alt_budget", msg: format!("{b}: must be the CtS level budget + 1 ({})", budget[0] + 1) }),
    }
}

/// A linear-transform stage's plaintext encoding: the level it is encoded at and its scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StageEnc {
    pub level: u32,
    pub top: u32,
    pub scale: f64,
}

/// The scale an auxiliary (CtS/StC) plaintext is encoded at, OpenFHE `MakeAuxPlaintext` as patched (0002/0006):
/// `sf[level]`, except on a NON-composite chain at a mixed-size segment boundary, where it is
/// `sf[level+1] * q_drop / sf[level]` (the post-multiplication rescale then lands exactly on the next level's scale).
/// Composite chains store a sentinel 1 off the grid and are uniform by construction: plain `sf[level]` (patch 0006).
pub fn aux_plaintext_scale(st: &ScaleTables, level: u32) -> f64 {
    let sf = &st.sf_real_openfhe;
    let lv = level as usize;
    let mut sc = sf[lv];
    let num_q = st.q.len();
    if st.d == 1 && lv + 1 < num_q {
        let q_drop = st.mrf[num_q - 1 - lv];
        let boundary = sf[lv + 1] * q_drop / sc;
        if (boundary.log2() - sc.log2()).abs() > 0.1 {
            sc = boundary;
        }
    }
    sc
}

/// CtS stages in execution order. Stock OpenFHE: the raise lands at the top (or `raise_drop` levels below it),
/// the FLEXIBLEAUTO adjust spends one level, then stage `e` is encoded at `raise + d (adjust) + d*e`
/// (= `towersToDrop + d*e` of `EvalCoeffsToSlotsPrecompute`). FIDESLIB_BTS_SHIFT folds the adjust constant into
/// stage 0 (`cts0_const`, scale bookkeeping times 2^-`cts0_t`), so the stages start at the raise level.
pub fn cts_stage_levels(st: &ScaleTables, budget_enc: u32, raise_drop: u32, shift: Option<u32>) -> Result<Vec<StageEnc>> {
    let chain = ChainShape::from_tables(st);
    let d = chain.unit;
    let first = d * raise_drop + if shift.is_some() { 0 } else { d };
    (0..budget_enc)
        .map(|e| {
            let level = first + d * e;
            let top = chain.top_max().checked_sub(level).ok_or_else(|| Error::Bts {
                knob: "budget_enc",
                msg: format!("CtS stage {e} at level {level} is past the chain bottom"),
            })?;
            let mut scale = aux_plaintext_scale(st, level);
            if let (0, Some(t)) = (e, shift) {
                scale *= 2f64.powi(-(t as i32));
            }
            Ok(StageEnc { level, top, scale })
        })
        .collect()
}

/// StC stages in execution order, shipped (StC-last) order: stage `e` is encoded at the effective level of the
/// EvalMod output plus `d*e` (`EvalSlotsToCoeffsPrecompute`'s `level0 + d*s`; level0 = the levels the raise
/// adjust, CtS and EvalMod consumed). `multPt` rescales a pending degree first, hence the effective level.
pub fn stc_stage_levels(st: &ScaleTables, em_out_level: u32, em_out_deg: u8, budget_dec: u32) -> Result<Vec<StageEnc>> {
    let chain = ChainShape::from_tables(st);
    let d = chain.unit;
    let first = effective_level(em_out_level, em_out_deg, d);
    (0..budget_dec)
        .map(|e| {
            let level = first + d * e;
            let top = chain.top_max().checked_sub(level).ok_or_else(|| Error::Bts {
                knob: "budget_dec",
                msg: format!("StC stage {e} at level {level} is past the chain bottom"),
            })?;
            Ok(StageEnc { level, top, scale: aux_plaintext_scale(st, level) })
        })
        .collect()
}

/// ModRaise's FLEXIBLEAUTO adjust factor 2^-(CF - deg), deg = log2(q0 / 2^scale_bits) (OpenFHE `EvalBootstrap`:
/// `correction = m_correctionFactor - deg`; refused when CF < deg).
pub fn raise_adjust_factor(cf: u32, deg: u32) -> Result<f64> {
    if cf < deg {
        return Err(Error::Bts { knob: "cf", msg: format!("correction factor {cf} < chain deg {deg}") });
    }
    Ok(2f64.powi(-((cf - deg) as i32)))
}

/// StC-first: the shipped StC plaintexts carry scaleDec ~ 2^deg; run before EvalMod, the bootstrap's integer
/// post-factor becomes 2^(correction + deg) = 2^CF (FIDESlib BootstrapPrecomputation.cuh, StC_first).
pub fn stc_first_post_factor(cf: u32) -> f64 {
    2f64.powi(cf as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The formula reproduces the runtime's known landings: SparseB {1, 1} at 24, the dense shipped order {4, 3} at 34
    /// (`stc_first_fallback` of the n32 recipe), the classic {3, 3} sparse routes at 32.
    #[test]
    fn multi_stage_landings() {
        let b = BtsConfig::n32_shipped();
        assert_eq!(multi_stage_landing([1, 1], 2), Landing { level: 24, deg: 1 });
        assert_eq!(Some(multi_stage_landing([4, 3], 2)), b.stc_first_fallback);
        assert_eq!(multi_stage_landing([3, 3], 2), Landing { level: 32, deg: 1 });
    }

    /// The staged finishes on the n32 measured landing (StC-first 28): PCMM outputs at (26 + 2 d, 2), CCMM outputs at
    /// (30 + 2 d, 1), the levels the runtime pins for them (Llama prefill128: pcmm drops 0..6, ccmm drops 0 and 4);
    /// the fake capture's raw form of the same landing gives the same levels.
    #[test]
    fn finish_landings() {
        // the Llama prefill runtime's variants (FIDESLIB_BTS_RAISE_DROPS=1,2,4,5,6)
        let rt = BtsConfig { raise_drops: vec![1, 2, 4, 5, 6], ..BtsConfig::n32_shipped() };
        let mut cap = rt.clone();
        cap.staged = Landing { level: 26, deg: 2 };
        for b in [&rt, &cap] {
            for d in 0..=6 {
                assert_eq!(b.finish_landing(ChainShape::N32, Finish::Pcmm, d), Landing { level: 26 + 2 * d, deg: 2 });
                assert_eq!(b.finish_landing(ChainShape::N32, Finish::Ccmm, d), Landing { level: 30 + 2 * d, deg: 1 });
            }
        }
        // the same landing as a planned StC-first refresh of an input inside the entry, before its realize
        let r = Route { stc_first: true, raise_drop: 4, ..Route::dense() };
        let planned = rt.landing(ChainShape::N32, &Quirks::none(), &r, 42, 2).unwrap();
        assert_eq!(planned, Landing { level: 36, deg: 1 });
        assert_eq!(rt.finish_landing(ChainShape::N32, Finish::Pcmm, 4).level + 2, planned.level);
    }

    /// FIDESLIB_CTS_ALT_BUDGET = 5 on the dense {4, 3} route: every drop lands where it lands without it.
    #[test]
    fn cts_alt_keeps_landings() {
        let budget = [4, 3];
        for d in 0..=6 {
            let (b2, d2) = cts_alt_variant(budget, 5, d).unwrap();
            let with = multi_stage_landing(b2, 2).level + 2 * d2;
            let without = multi_stage_landing(budget, 2).level + 2 * d;
            assert_eq!(with, without, "drop {d}");
            assert_eq!(b2[0], if d >= 1 { 5 } else { 4 });
        }
        assert_eq!(cts_alt_variant(budget, 0, 3).unwrap(), (budget, 3));
        assert!(cts_alt_variant(budget, 6, 1).is_err());
    }
}
