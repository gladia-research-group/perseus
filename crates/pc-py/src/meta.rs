//! Value types of the Shadow: `CtMeta` (the one stored ciphertext state), `PtMeta`, `Route`, `Prelude`, `Recipe`,
//! and the level-convention conversions (`pc_shadow::meta`).
use crate::level_err;
use crate::tag::PyTag;
use pc_shadow::meta::{self as m, ChainShape};
use pc_shadow::rules::{Consts, Operand, Prelude, PtMeta, Recipe};
use pc_shadow::{CtMeta, Route};
use pyo3::prelude::*;

#[pyclass(name = "CtMeta", module = "perseus._pc")]
#[derive(Clone, Copy)]
pub struct PyCtMeta(pub CtMeta);

#[pymethods]
impl PyCtMeta {
    #[new]
    #[pyo3(signature = (level, deg=1, nf=0.0, slots=0, tag=None))]
    fn new(level: u32, deg: u8, nf: f64, slots: u32, tag: Option<PyTag>) -> Self {
        PyCtMeta(CtMeta { level, deg, nf, slots, tag: tag.map(|t| t.0).unwrap_or_default() })
    }
    #[getter]
    fn level(&self) -> u32 {
        self.0.level
    }
    #[setter]
    fn set_level(&mut self, v: u32) {
        self.0.level = v;
    }
    #[getter]
    fn deg(&self) -> u8 {
        self.0.deg
    }
    #[setter]
    fn set_deg(&mut self, v: u8) {
        self.0.deg = v;
    }
    #[getter]
    fn nf(&self) -> f64 {
        self.0.nf
    }
    #[setter]
    fn set_nf(&mut self, v: f64) {
        self.0.nf = v;
    }
    #[getter]
    fn slots(&self) -> u32 {
        self.0.slots
    }
    #[setter]
    fn set_slots(&mut self, v: u32) {
        self.0.slots = v;
    }
    /// A copy of the tag (setting a field on it does not write back; assign `meta.tag = t`).
    #[getter]
    fn tag(&self) -> PyTag {
        PyTag(self.0.tag)
    }
    #[setter]
    fn set_tag(&mut self, v: PyTag) {
        self.0.tag = v.0;
    }
    fn effective_level(&self, unit: u32) -> u32 {
        self.0.effective_level(unit)
    }
    fn copy(&self) -> Self {
        *self
    }
    /// This state with the given fields replaced.
    #[pyo3(signature = (level=None, deg=None, nf=None, slots=None, tag=None))]
    fn replace(&self, level: Option<u32>, deg: Option<u8>, nf: Option<f64>, slots: Option<u32>, tag: Option<PyTag>) -> Self {
        let mut x = self.0;
        if let Some(v) = level {
            x.level = v;
        }
        if let Some(v) = deg {
            x.deg = v;
        }
        if let Some(v) = nf {
            x.nf = v;
        }
        if let Some(v) = slots {
            x.slots = v;
        }
        if let Some(v) = tag {
            x.tag = v.0;
        }
        PyCtMeta(x)
    }
    fn __eq__(&self, o: &Self) -> bool {
        self.0 == o.0
    }
    fn __repr__(&self) -> String {
        format!("CtMeta(level={}, deg={}, nf={:e}, slots={})", self.0.level, self.0.deg, self.0.nf, self.0.slots)
    }
}

#[pyclass(name = "PtMeta", module = "perseus._pc")]
#[derive(Clone, Copy)]
pub struct PyPtMeta(pub PtMeta);

#[pymethods]
impl PyPtMeta {
    #[new]
    #[pyo3(signature = (level, deg=1, nf=0.0, tag=None))]
    fn new(level: u32, deg: u8, nf: f64, tag: Option<PyTag>) -> Self {
        PyPtMeta(PtMeta { level, deg, nf, tag: tag.map(|t| t.0).unwrap_or_default() })
    }
    #[getter]
    fn level(&self) -> u32 {
        self.0.level
    }
    #[getter]
    fn deg(&self) -> u8 {
        self.0.deg
    }
    #[getter]
    fn nf(&self) -> f64 {
        self.0.nf
    }
    #[getter]
    fn tag(&self) -> PyTag {
        PyTag(self.0.tag)
    }
    fn __repr__(&self) -> String {
        format!("PtMeta(level={}, deg={})", self.0.level, self.0.deg)
    }
}

/// One bootstrap call's parameters (in place of FIDESlib's global scopes).
#[pyclass(name = "Route", module = "perseus._pc", frozen)]
#[derive(Clone, Copy)]
pub struct PyRoute(pub Route);

#[pymethods]
impl PyRoute {
    #[new]
    #[pyo3(signature = (slots=0, cf=None, real=false, raise_drop=0, stc_first=false, sparse_b=false, spru_h=0, iterations=1))]
    #[allow(clippy::too_many_arguments)]
    fn new(slots: u32, cf: Option<u32>, real: bool, raise_drop: u32, stc_first: bool, sparse_b: bool, spru_h: u32, iterations: u32) -> Self {
        PyRoute(Route { slots, cf, real, raise_drop, stc_first, sparse_b, spru_h, iterations })
    }
    #[staticmethod]
    fn dense() -> Self {
        PyRoute(Route::dense())
    }
    #[staticmethod]
    fn sparse(slots: u32) -> Self {
        PyRoute(Route::sparse(slots))
    }
    #[getter]
    fn slots(&self) -> u32 {
        self.0.slots
    }
    #[getter]
    fn cf(&self) -> Option<u32> {
        self.0.cf
    }
    #[getter]
    fn real(&self) -> bool {
        self.0.real
    }
    #[getter]
    fn raise_drop(&self) -> u32 {
        self.0.raise_drop
    }
    #[getter]
    fn stc_first(&self) -> bool {
        self.0.stc_first
    }
    #[getter]
    fn sparse_b(&self) -> bool {
        self.0.sparse_b
    }
    #[getter]
    fn spru_h(&self) -> u32 {
        self.0.spru_h
    }
    #[getter]
    fn iterations(&self) -> u32 {
        self.0.iterations
    }
    fn __eq__(&self, o: &Self) -> bool {
        self.0 == o.0
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// What to do to an operand before the op: `kind` = "rescale" | "drop_to" (`level`) | "scalar" (`scalar`);
/// `operand` = "a" (modified in place) | "b" (a copy of the second operand).
#[pyclass(name = "Prelude", module = "perseus._pc", frozen)]
#[derive(Clone)]
pub struct PyPrelude(pub Prelude);

#[pymethods]
impl PyPrelude {
    #[getter]
    fn kind(&self) -> &'static str {
        match self.0 {
            Prelude::Rescale(_) => "rescale",
            Prelude::DropTo(..) => "drop_to",
            Prelude::ScalarAdjust(..) => "scalar",
        }
    }
    #[getter]
    fn operand(&self) -> &'static str {
        let o = match self.0 {
            Prelude::Rescale(o) | Prelude::DropTo(o, _) | Prelude::ScalarAdjust(o, _) => o,
        };
        match o {
            Operand::A => "a",
            Operand::B => "b",
        }
    }
    #[getter]
    fn level(&self) -> Option<u32> {
        match self.0 {
            Prelude::DropTo(_, l) => Some(l),
            _ => None,
        }
    }
    #[getter]
    fn scalar(&self) -> Option<f64> {
        match self.0 {
            Prelude::ScalarAdjust(_, c) => Some(c),
            _ => None,
        }
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// The Shadow's answer for one op: preludes, the output state, the integer constant row (if scale tables are
/// loaded), and for a hint whether it fired.
#[pyclass(name = "Recipe", module = "perseus._pc", frozen)]
#[derive(Clone)]
pub struct PyRecipe {
    pub r: Recipe,
    pub fired: Option<bool>,
}

impl PyRecipe {
    pub fn of(r: Recipe) -> Self {
        PyRecipe { r, fired: None }
    }
}

#[pymethods]
impl PyRecipe {
    #[getter]
    fn pre(&self) -> Vec<PyPrelude> {
        self.r.pre.iter().map(|p| PyPrelude(*p)).collect()
    }
    #[getter]
    fn out(&self) -> PyCtMeta {
        PyCtMeta(self.r.out)
    }
    /// "mult" | "add" | None.
    #[getter]
    fn consts_kind(&self) -> Option<&'static str> {
        match &self.r.consts {
            Some(Consts::Mult(_)) => Some("mult"),
            Some(Consts::Add(_)) => Some("add"),
            None => None,
        }
    }
    #[getter]
    fn consts(&self) -> Option<Vec<u64>> {
        match &self.r.consts {
            Some(Consts::Mult(v)) | Some(Consts::Add(v)) => Some(v.clone()),
            None => None,
        }
    }
    #[getter]
    fn fired(&self) -> Option<bool> {
        self.fired
    }
    fn __repr__(&self) -> String {
        format!("Recipe(pre={:?}, out={:?}, fired={:?})", self.r.pre, self.r.out, self.fired)
    }
}

fn chain(n_q: Option<u32>, unit: u32) -> ChainShape {
    ChainShape { n_q: n_q.unwrap_or(ChainShape::N32.n_q), unit, singles: 0 }
}

/// FIDESlib's top limb index of a ciphertext at `level` (default chain: n32, 54 primes).
#[pyfunction]
#[pyo3(signature = (level, n_q=None))]
fn top_limb(level: u32, n_q: Option<u32>) -> PyResult<u32> {
    m::top_limb(chain(n_q, 2), level).map_err(level_err)
}
#[pyfunction]
#[pyo3(signature = (top, n_q=None))]
fn level_from_top_limb(top: u32, n_q: Option<u32>) -> PyResult<u32> {
    m::level_from_top_limb(chain(n_q, 2), top).map_err(level_err)
}
/// The planner's consumed depth (`sim.py` `lc`) of a ciphertext at `level`.
#[pyfunction]
fn planner_consumed(level: u32, bootstrap_level: u32) -> i64 {
    m::planner_consumed(level, bootstrap_level)
}
#[pyfunction]
fn level_from_planner(consumed: f64, bootstrap_level: u32) -> PyResult<u32> {
    m::level_from_planner_f64(consumed, bootstrap_level).map_err(level_err)
}

pub fn register(mo: &Bound<'_, PyModule>) -> PyResult<()> {
    mo.add_class::<PyCtMeta>()?;
    mo.add_class::<PyPtMeta>()?;
    mo.add_class::<PyRoute>()?;
    mo.add_class::<PyPrelude>()?;
    mo.add_class::<PyRecipe>()?;
    mo.add_function(wrap_pyfunction!(top_limb, mo)?)?;
    mo.add_function(wrap_pyfunction!(level_from_top_limb, mo)?)?;
    mo.add_function(wrap_pyfunction!(planner_consumed, mo)?)?;
    mo.add_function(wrap_pyfunction!(level_from_planner, mo)?)?;
    Ok(())
}
