//! `Shadow` for Python: the pc-shadow rules built from a `Config` (+ the planner's `PlanConfig` for the landing
//! recipe), GPU-free. `mode` picks the quirk preset and the landing source:
//!   * "runtime"  — the C++ rules, `BtsConfig::n32_shipped()` overridden by the Config/PlanConfig, real n32 chain;
//!   * "planner"  — `Quirks::planner()`, bottomless chain (sim.py predicts past q0), landings exactly the
//!     PlanConfig recipe's (bootstrap level, sparse_bts_out, StC-first entry/fallback), no input guard;
//!   * "fake"     — `Quirks::fake()`: every refresh lands at `quirks.fake_bts_out` (34) realized;
//!   * "fakecap"  — `Quirks::fakecap()`: raw (degree 2) landings from `quirks.fakecap_landings`, StC-first entry /
//!     drop from `quirks.fakecap_stc_entry` / `fakecap_stc_drop` (also the finishes' StC-first landing, `staged`).
//! `quirks.shadow` (a list of `Quirk` names) replaces the mode's preset. `strict` keeps the real chain bottom in
//! the fake modes (a level past q0 raises LevelError).
use crate::config::{PyConfig, PyPlanConfig};
use crate::meta::{PyCtMeta, PyPtMeta, PyRecipe, PyRoute};
use crate::{level_err, ConfigError};
use pc_config::{Config, PlanConfig};
use pc_shadow::graph::Node;
use pc_shadow::route::Landing;
use pc_shadow::sim::StepArgs;
use pc_shadow::{BtsConfig, ChainShape, Quirk, Quirks, Route, Shadow, ShadowConfig};
use pyo3::prelude::*;
use std::sync::Arc;

fn cfg_err(knob: &str, msg: impl std::fmt::Display) -> PyErr {
    ConfigError::new_err(format!("shadow config {knob}: {msg}"))
}

fn quirk_named(s: &str) -> Option<Quirk> {
    Quirk::ALL.into_iter().find(|q| format!("{q:?}") == s)
}

/// "s:level[:deg],..." -> landings (deg defaults to `deg`).
fn parse_landings(knob: &str, spec: &str, deg: u8) -> PyResult<Vec<(u32, Landing)>> {
    let mut out = vec![];
    for item in spec.split(',').filter(|x| !x.is_empty()) {
        let p: Vec<&str> = item.split(':').collect();
        let num = |i: usize| -> PyResult<u32> {
            p.get(i).and_then(|x| x.parse().ok()).ok_or_else(|| cfg_err(knob, format!("bad item {item:?} in {spec:?}")))
        };
        let d = if p.len() > 2 { num(2)? as u8 } else { deg };
        out.push((num(0)?, Landing { level: num(1)?, deg: d }));
    }
    Ok(out)
}

fn plan_u32(knob: &str, v: &Option<String>) -> PyResult<Option<u32>> {
    v.as_deref().filter(|s| !s.is_empty()).map(|s| s.parse().map_err(|_| cfg_err(knob, format!("{s:?}")))).transpose()
}

fn u32_of(knob: &str, v: Option<i64>, default: u32) -> PyResult<u32> {
    match v {
        None => Ok(default),
        Some(x) => u32::try_from(x).map_err(|_| cfg_err(knob, x)),
    }
}

/// The ShadowConfig a Config means in `mode` (pc-shadow stays untouched).
pub fn shadow_config(c: &Config, plan: Option<&PlanConfig>, mode: &str, strict: bool) -> PyResult<ShadowConfig> {
    let unit = u32_of("chain.composite_degree", c.chain.composite_degree, 2)?;
    match c.chain.chain.as_deref() {
        None | Some("n32") => {}
        Some(other) => return Err(cfg_err("chain.chain", format!("{other:?}: only the n32 chain shape is known"))),
    }
    let real_chain = ChainShape { n_q: ChainShape::N32.n_q, unit };
    let chain = match mode {
        "runtime" => real_chain,
        "planner" => ChainShape::unbounded(unit),
        "fake" | "fakecap" if strict => real_chain,
        "fake" | "fakecap" => ChainShape::unbounded(unit),
        _ => return Err(cfg_err("mode", format!("{mode:?} (runtime | planner | fake | fakecap)"))),
    };
    let mut quirks = match mode {
        "planner" => Quirks::planner(),
        "fake" => Quirks::fake(),
        "fakecap" => Quirks::fakecap(),
        _ => Quirks::none(),
    };
    if let Some(names) = &c.quirks.shadow {
        let mut q = Quirks::none();
        for n in names {
            q.0.insert(quirk_named(n).ok_or_else(|| cfg_err("quirks.shadow", format!("no quirk {n:?}")))?);
        }
        quirks = q;
    }
    let mut bts = BtsConfig::n32_shipped();
    bts.level_limit = u32_of("bts.auto_bts_level", c.bts.auto_bts_level, 46)?;
    bts.k_max = u32_of("bts.k_max", c.bts.k_max, bts.k_max)?;
    if let Some(d) = &c.bts.raise_drops {
        bts.raise_drops = d.iter().map(|x| x.parse().map_err(|_| cfg_err("bts.raise_drops", x))).collect::<PyResult<_>>()?;
    }
    match mode {
        "fake" => {
            let out = u32_of("quirks.fake_bts_out", c.quirks.fake_bts_out, 34)?;
            bts.landings = [(0, Landing { level: out, deg: 1 })].into_iter().collect();
            bts.stc_first_entry = None;
            bts.stc_first_fallback = None;
            bts.max_input_level = u32::MAX;
        }
        "fakecap" => {
            let spec = c.quirks.fakecap_landings.as_deref().unwrap_or("0:32,1:14,256:22,512:22");
            let ls = parse_landings("quirks.fakecap_landings", spec, 2)?;
            let dense = ls.iter().find(|(s, _)| *s == 0).map(|x| x.1).ok_or_else(|| cfg_err("quirks.fakecap_landings", "no dense (0:) landing"))?;
            let drop = u32_of("quirks.fakecap_stc_drop", c.quirks.fakecap_stc_drop, 6)?;
            bts.landings = ls.into_iter().collect();
            // StC-first dense lands `drop` lower; the shipped order (and a too-deep StC-first input) at the raw landing
            bts.landings.insert(0, Landing { level: dense.level.saturating_sub(drop), deg: dense.deg });
            bts.stc_first_fallback = Some(dense);
            bts.stc_first_entry = Some(u32_of("quirks.fakecap_stc_entry", c.quirks.fakecap_stc_entry, 46)?);
            // the finishes run the StC-first route whatever the capture's own dense refreshes take: its raw landing
            bts.staged = Landing { level: dense.level.saturating_sub(drop), deg: dense.deg };
            bts.max_input_level = u32::MAX;
        }
        "planner" => {
            // The planner's mirror is exactly its recipe: the bootstrap level plus the measured sparse landings (a
            // route with none is the planner's own "assume dense" case, so no shipped defaults leak in), StC-first
            // only where the recipe names its entry / fallback, and no input guard (the refresh envelope is a
            // planner policy, and a plan-free run predicts past the guard to find the over-budget nodes).
            let p = plan.cloned().unwrap_or_default();
            let bl = plan_u32("plan.bootstrap_level", &p.bootstrap_level)?.unwrap_or(bts.bootstrap_level);
            bts.bootstrap_level = bl;
            bts.landings = [(0, Landing { level: bl, deg: 1 })].into_iter().collect();
            if let Some(spec) = p.sparse_bts_out.as_deref() {
                for (s, l) in parse_landings("plan.sparse_bts_out", spec, 1)? {
                    bts.landings.insert(s, l);
                }
            }
            bts.stc_first_entry = plan_u32("plan.stc_first_entry", &p.stc_first_entry)?;
            bts.stc_first_fallback = match p.stc_first_fallback.as_deref().filter(|s| !s.is_empty()) {
                Some(spec) => Some(parse_landings("plan.stc_first_fallback", &format!("0:{spec}"), 1)?[0].1),
                None => None,
            };
            if let Some(cf) = plan_u32("plan.cf_max", &p.cf_max)? {
                bts.cf_window.1 = cf;
            }
            bts.max_input_level = u32::MAX;
        }
        _ => {
            bts.max_input_level = u32_of("bts.bts_max_input_level", c.bts.bts_max_input_level, bts.max_input_level)?;
            if let Some(p) = plan {
                if let Some(bl) = plan_u32("plan.bootstrap_level", &p.bootstrap_level)? {
                    bts.bootstrap_level = bl;
                    bts.landings.insert(0, Landing { level: bl, deg: 1 });
                }
                if let Some(spec) = p.sparse_bts_out.as_deref() {
                    for (s, l) in parse_landings("plan.sparse_bts_out", spec, 1)? {
                        bts.landings.insert(s, l);
                    }
                }
                if let Some(e) = plan_u32("plan.stc_first_entry", &p.stc_first_entry)? {
                    bts.stc_first_entry = Some(e);
                }
                if let Some(spec) = p.stc_first_fallback.as_deref().filter(|s| !s.is_empty()) {
                    let l = parse_landings("plan.stc_first_fallback", &format!("0:{spec}"), 1)?;
                    bts.stc_first_fallback = Some(l[0].1);
                }
                if let Some(cf) = plan_u32("plan.cf_max", &p.cf_max)? {
                    bts.cf_window.1 = cf;
                }
            }
        }
    }
    // the shipped order (a dense refresh not run StC-first) lands where a too-deep StC-first input falls back to
    bts.dense_shipped = bts.stc_first_fallback;
    Ok(ShadowConfig { chain, bts, quirks })
}

#[pyclass(name = "Shadow", module = "perseus._pc", frozen)]
pub struct PyShadow {
    s: Shadow,
    mode: String,
}

impl PyShadow {
    fn build(cfg: ShadowConfig, tables: bool, mode: &str) -> PyResult<PyShadow> {
        let st = if tables {
            let chain = pc_core::params::Chain::generate(&pc_core::params::ChainSpec::n32()).map_err(|e| cfg_err("tables", e))?;
            Some(Arc::new(pc_core::scale::ScaleTables::precompute(&chain).map_err(|e| cfg_err("tables", e))?))
        } else {
            None
        };
        Ok(PyShadow { s: Shadow::new(cfg, st), mode: mode.to_string() })
    }
}

/// One planner graph node (the fields of `plan.placer.ir.Node` the rules read), built once per node and reused
/// across simulations.
#[pyclass(name = "SimNode", module = "perseus._pc", frozen)]
pub struct PySimNode(pub Node);

#[pymethods]
impl PySimNode {
    #[new]
    #[pyo3(signature = (idx, op, inputs, input_levels, output, output_level=None, output_deg=None, step=String::new(),
                        cipher_inputs=Vec::new(), hint_level=None, lvl_cap=None, cost=0, cost_forced=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        idx: usize,
        op: String,
        inputs: Vec<String>,
        input_levels: Vec<Option<f64>>,
        output: String,
        output_level: Option<f64>,
        output_deg: Option<i64>,
        step: String,
        cipher_inputs: Vec<String>,
        hint_level: Option<f64>,
        lvl_cap: Option<f64>,
        cost: i64,
        cost_forced: bool,
    ) -> Self {
        PySimNode(Node {
            idx,
            op,
            inputs,
            input_levels,
            output,
            output_level,
            output_deg,
            step,
            cipher_inputs,
            hint_level,
            lvl_cap,
            cost,
            cost_forced,
        })
    }
    #[getter]
    fn idx(&self) -> usize {
        self.0.idx
    }
    #[getter]
    fn op(&self) -> &str {
        &self.0.op
    }
    fn __repr__(&self) -> String {
        format!("SimNode({} {} -> {})", self.0.idx, self.0.op, self.0.output)
    }
}

fn rec(r: pc_shadow::Result<pc_shadow::Recipe>) -> PyResult<PyRecipe> {
    r.map(PyRecipe::of).map_err(level_err)
}

#[pymethods]
impl PyShadow {
    #[new]
    #[pyo3(signature = (config, mode="runtime", plan=None, tables=false, strict=false))]
    fn new(config: &PyConfig, mode: &str, plan: Option<&PyPlanConfig>, tables: bool, strict: bool) -> PyResult<Self> {
        let cfg = shadow_config(&config.0, plan.map(|p| &p.0), mode, strict)?;
        PyShadow::build(cfg, tables, mode)
    }
    #[getter]
    fn mode(&self) -> &str {
        &self.mode
    }
    #[getter]
    fn unit(&self) -> u32 {
        self.s.unit()
    }
    #[getter]
    fn level_limit(&self) -> u32 {
        self.s.cfg.bts.level_limit
    }
    #[getter]
    fn bootstrap_level(&self) -> u32 {
        self.s.cfg.bts.bootstrap_level
    }
    #[getter]
    fn quirks(&self) -> Vec<String> {
        self.s.cfg.quirks.0.iter().map(|q| format!("{q:?}")).collect()
    }
    fn has(&self, quirk: &str) -> PyResult<bool> {
        let q = quirk_named(quirk).ok_or_else(|| cfg_err("quirk", quirk))?;
        Ok(self.s.has(q))
    }

    fn add(&self, a: &PyCtMeta, b: &PyCtMeta) -> PyResult<PyRecipe> {
        rec(self.s.add(&a.0, &b.0))
    }
    fn sub(&self, a: &PyCtMeta, b: &PyCtMeta) -> PyResult<PyRecipe> {
        rec(self.s.sub(&a.0, &b.0))
    }
    fn add_scalar(&self, a: &PyCtMeta, c: f64) -> PyResult<PyRecipe> {
        rec(self.s.add_scalar(&a.0, c))
    }
    fn add_pt(&self, a: &PyCtMeta, pt: &PyPtMeta) -> PyResult<PyRecipe> {
        rec(self.s.add_pt(&a.0, &pt.0))
    }
    fn mult(&self, a: &PyCtMeta, b: &PyCtMeta) -> PyResult<PyRecipe> {
        rec(self.s.mult(&a.0, &b.0))
    }
    fn square(&self, a: &PyCtMeta) -> PyResult<PyRecipe> {
        rec(self.s.square(&a.0))
    }
    fn mult_scalar(&self, a: &PyCtMeta, c: f64) -> PyResult<PyRecipe> {
        rec(self.s.mult_scalar(&a.0, c))
    }
    fn negate(&self, a: &PyCtMeta) -> PyResult<PyRecipe> {
        rec(self.s.negate(&a.0))
    }
    fn mult_pt(&self, a: &PyCtMeta, pt: &PyPtMeta) -> PyResult<PyRecipe> {
        rec(self.s.mult_pt(&a.0, &pt.0))
    }
    fn conjugate(&self, a: &PyCtMeta) -> PyRecipe {
        PyRecipe::of(self.s.conjugate(&a.0))
    }
    fn mult_i(&self, a: &PyCtMeta) -> PyRecipe {
        PyRecipe::of(self.s.mult_i(&a.0))
    }
    fn rotate(&self, a: &PyCtMeta, k: i64) -> PyRecipe {
        PyRecipe::of(self.s.rotate(&a.0, k))
    }
    fn rotate_sum(&self, a: &PyCtMeta, steps: Vec<i64>) -> PyRecipe {
        PyRecipe::of(self.s.rotate_sum(&a.0, &steps))
    }
    fn rescale(&self, a: &PyCtMeta) -> PyResult<PyRecipe> {
        rec(self.s.rescale(&a.0))
    }
    fn drop_to_level(&self, a: &PyCtMeta, level: u32) -> PyRecipe {
        PyRecipe::of(self.s.drop_to_level(&a.0, level))
    }
    fn realize(&self, a: &PyCtMeta) -> PyResult<PyRecipe> {
        rec(self.s.realize(&a.0))
    }
    fn tag_reduce(&self, a: &PyCtMeta, stride: i64) -> PyRecipe {
        PyRecipe::of(self.s.tag_reduce(&a.0, stride))
    }
    fn hint_level(&self, a: &PyCtMeta) -> PyRecipe {
        PyRecipe::of(self.s.hint_level(&a.0))
    }
    #[pyo3(signature = (a, route=None))]
    fn bootstrap(&self, a: &PyCtMeta, route: Option<&PyRoute>) -> PyResult<PyRecipe> {
        let r = route.map(|r| r.0).unwrap_or_else(Route::dense);
        rec(self.s.bootstrap(&a.0, &r))
    }
    /// `bootstrap_hint(thr, account_pending_rescale)`: `Recipe.fired` says whether it refreshed.
    #[pyo3(signature = (a, threshold, account_pending=false, route=None))]
    fn hint(&self, a: &PyCtMeta, threshold: u32, account_pending: bool, route: Option<&PyRoute>) -> PyResult<PyRecipe> {
        let r = route.map(|r| r.0).unwrap_or_else(Route::dense);
        let h = self.s.hint(&a.0, threshold, account_pending, &r).map_err(level_err)?;
        Ok(PyRecipe { r: h.recipe, fired: Some(h.fired) })
    }
    #[pyo3(signature = (a, s, n_live, prescale=1.0))]
    fn fold_bootstrap(&self, a: &PyCtMeta, s: u32, n_live: u32, prescale: f64) -> PyResult<PyRecipe> {
        rec(self.s.fold_bootstrap(&a.0, s, n_live, prescale))
    }
    /// One captured graph node through the rules (`pc_shadow::Shadow::step`, what `sim.py` runs per node):
    /// `inputs` = the states of `node.cipher_inputs` in order; the keyword args are the plan decisions a hint
    /// reads (`StepArgs`); `dense_fallback` = (StC-first entry, landing level, landing deg). `Recipe.out` is the
    /// node's output, `Recipe.fired` a hint's decision (None for other ops).
    #[pyo3(signature = (node, inputs, hint_force=None, placed=false, landing=None, sparse=false, dense_fallback=None))]
    #[allow(clippy::too_many_arguments)]
    fn step(
        &self,
        node: &PySimNode,
        inputs: Vec<PyCtMeta>,
        hint_force: Option<bool>,
        placed: bool,
        landing: Option<(u32, u8)>,
        sparse: bool,
        dense_fallback: Option<(u32, u32, u8)>,
    ) -> PyResult<PyRecipe> {
        let ins: Vec<pc_shadow::CtMeta> = inputs.iter().map(|m| m.0).collect();
        let args = StepArgs { hint_force, placed, landing, sparse };
        let so = self.s.step(&node.0, &ins, &args, dense_fallback).map_err(level_err)?;
        Ok(PyRecipe { r: pc_shadow::Recipe { pre: vec![], out: so.out, consts: None }, fired: so.hint_fired })
    }
    /// The StC-first entry and fallback landing this engine runs under (None when not configured).
    #[getter]
    fn stc_first(&self) -> (Option<u32>, Option<(u32, u8)>) {
        let b = &self.s.cfg.bts;
        (b.stc_first_entry, b.stc_first_fallback.map(|l| (l.level, l.deg)))
    }
    /// The route slot counts with a landing (0 = dense).
    #[getter]
    fn routes(&self) -> Vec<u32> {
        self.s.cfg.bts.landings.keys().copied().collect()
    }
    /// Where a refresh of `meta` on `route` lands: (level, deg).
    #[pyo3(signature = (route, meta))]
    fn landing(&self, route: &PyRoute, meta: &PyCtMeta) -> PyResult<(u32, u8)> {
        let out = self.s.bootstrap(&meta.0, &route.0).map_err(level_err)?.out;
        Ok((out.level, out.deg))
    }
    /// Where a matrix product's finish output lands (`BtsConfig::finish_landing`): `op` "pcmm" | "ccmm" (the
    /// descriptor's `op`), under its raise drop: (level, deg).
    #[pyo3(signature = (op, raise_drop=0))]
    fn finish_landing(&self, op: &str, raise_drop: u32) -> PyResult<(u32, u8)> {
        let f = pc_shadow::Finish::of_op(op).ok_or_else(|| cfg_err("finish_landing", format!("op {op:?} (pcmm | ccmm | tfhe)")))?;
        let l = self.s.cfg.bts.finish_landing(self.s.cfg.chain, f, raise_drop);
        Ok((l.level, l.deg))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyShadow>()?;
    m.add_class::<PySimNode>()?;
    Ok(())
}
