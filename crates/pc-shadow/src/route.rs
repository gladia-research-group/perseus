//! Bootstrap routes, their landings, and the per-stage encoding rule of the CtS/StC linear transforms.
use crate::error::{Error, Result};
use crate::meta::{effective_level, top_limb, ChainShape};
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
            input_room: None,
        }
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
    let lv = ChainShape::from_tables(st).primes_dropped(level).unwrap_or(level) as usize;
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
            let top = top_limb(chain, level).map_err(|_| Error::Bts {
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
            let top = top_limb(chain, level).map_err(|_| Error::Bts {
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
}
