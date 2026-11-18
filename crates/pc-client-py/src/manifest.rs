//! `perseus_client.KeySource`, `perseus_client.Session.key_manifest`: the key request of an eager session (no
//! programs, PCMM, CCMM or SPRU: what a client of the eager serve path asks itself for), computed without a device.
//!
//! The same derivation as pc-py's `Session.key_manifest` (session.rs: `key_needs`, `KeySetSpec::keys` in
//! pc-gpu/src/keys.rs, `manifest`, `key_req`) on pc-client's own `KeyShape` (pc-gpu's `KskShape` rules, the same
//! fields): rotation steps of `rot_steps` and `groups` mod N/2 without 0, banded at `low_band` for `low_steps`
//! and at `band` otherwise; relin at the lean top, conj at the lowest rotation band below it; the dense -> sparse
//! key at the bottom shape with `sparse_out` (and sparse -> dense, full, unless lean). A planned session's
//! arguments are refused rather than half served. `tests/test_manifest.py` compares the two where both build.
use pc_client::{KeyReq, KeyShape, KeySpec, Role};
use pc_config::Config;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::BTreeSet;

use crate::cerr;
use crate::config::PyConfig;

/// Where the session's keys come from (pc-py's `KeySource`, the fields a manifest reads and their checks).
#[pyclass(name = "KeySource", module = "perseus_client", frozen)]
#[derive(Clone, Debug)]
pub struct PyKeySource {
    pub rot_steps: Vec<i64>,
    pub groups: Vec<Vec<i64>>,
    pub conj: bool,
    pub layout: String,
    pub band: Option<usize>,
    pub low_steps: Vec<i64>,
    pub low_band: Option<usize>,
    pub sparse_out: bool,
    pub lean: bool,
}

#[pymethods]
impl PyKeySource {
    /// pc-py's signature; `seed` and `secret` (device keygen) are accepted and unused: a client makes its keys with
    /// `Client.keygen`.
    #[staticmethod]
    #[pyo3(signature = (rot_steps, groups=Vec::new(), conj=true, layout="dense", band=None, seed=None, secret=None, sparse_out=false, lean=false, low_steps=Vec::new(), low_band=None))]
    #[allow(clippy::too_many_arguments)]
    fn generate(rot_steps: Vec<i64>, groups: Vec<Vec<i64>>, conj: bool, layout: &str, band: Option<usize>, seed: Option<u64>, secret: Option<Vec<u8>>, sparse_out: bool, lean: bool, low_steps: Vec<i64>, low_band: Option<usize>) -> PyResult<Self> {
        let _ = (seed, secret);
        if layout != "dense" && layout != "packed" {
            return Err(PyValueError::new_err(format!("KeySource.layout = {layout:?}: dense | packed")));
        }
        if lean && band.is_none() {
            return Err(PyValueError::new_err("KeySource(lean=True) needs a band"));
        }
        if low_steps.is_empty() != low_band.is_none() || low_band.zip(band).is_some_and(|(l, b)| l >= b) {
            return Err(PyValueError::new_err("KeySource: low_steps come with a low_band below band"));
        }
        Ok(PyKeySource { rot_steps, groups, conj, layout: layout.to_string(), band, low_steps, low_band, sparse_out, lean })
    }
    #[getter]
    fn rot_steps(&self) -> Vec<i64> {
        self.rot_steps.clone()
    }
    fn __repr__(&self) -> String {
        format!("KeySource.generate({} rotation steps, {} groups, conj={}, layout={:?}, band={:?})", self.rot_steps.len(), self.groups.len(), self.conj, self.layout, self.band)
    }
}

/// pc-py's `contract_hash`: FNV-1a of the load-bearing Config's JSON.
pub fn contract_of(cfg: &Config) -> PyResult<String> {
    let j = cfg.load_bearing().to_json().map_err(crate::config_err)?;
    Ok(format!("{:016x}", pc_client::spec::fnv1a64(j.as_bytes())))
}

/// The eager key request of `cfg` and `ks`.
pub fn eager_spec(cfg: &Config, ks: &PyKeySource) -> PyResult<KeySpec> {
    let preset = cfg.chain.preset.clone().unwrap_or_else(|| "n32".into());
    let chain = pc_client::spec::chain_of(&preset).map_err(|e| crate::ConfigError::new_err(format!("chain.preset: {e}")))?;
    let half = (chain.n / 2) as i64;
    let mut steps: BTreeSet<i64> = ks.rot_steps.iter().map(|s| s.rem_euclid(half)).collect();
    for g in &ks.groups {
        steps.extend(g.iter().map(|s| s.rem_euclid(half)));
    }
    steps.remove(&0);
    let low: BTreeSet<i64> = ks.low_steps.iter().map(|s| s.rem_euclid(half)).filter(|&s| s != 0).collect();
    let rots: Vec<(i64, Option<usize>)> = steps.iter().map(|&s| (s, if low.contains(&s) { ks.low_band } else { ks.band })).collect();
    let lean = if ks.lean { ks.band } else { None };
    let rot_shape = |q_top: Option<usize>| match q_top {
        Some(t) if t + 1 < chain.q.len() => KeyShape::banded(&chain, t),
        _ => Ok(KeyShape::full(&chain)),
    };
    let full = KeyShape::full(&chain);
    let top = match lean {
        Some(t) => rot_shape(Some(t)).map_err(cerr)?,
        None => full,
    };
    let mut keys = vec![KeyReq { role: Role::Relin, shape: top }];
    if ks.conj {
        // a lean set whose rotations sit at two bands conjugates at the lower one
        let below = lean.and_then(|t| rots.iter().filter_map(|r| r.1).min().filter(|&m| m < t));
        keys.push(KeyReq { role: Role::Conj, shape: match below {
            Some(m) => rot_shape(Some(m)).map_err(cerr)?,
            None => top,
        } });
    }
    for &(step, q_top) in &rots {
        keys.push(KeyReq { role: Role::rot(step, chain.n).map_err(cerr)?, shape: rot_shape(q_top).map_err(cerr)? });
    }
    if ks.sparse_out {
        keys.push(KeyReq { role: Role::EncDenseToSparse, shape: KeyShape::bottom(&chain).map_err(cerr)? });
        if lean.is_none() {
            keys.push(KeyReq { role: Role::EncSparseToDense, shape: full });
        }
    }
    Ok(KeySpec {
        chain: preset,
        params_hash: pc_client::spec::params_hash(&chain),
        contract: contract_of(cfg)?,
        keys,
        spru: None,
        pcmm_t: vec![],
        ccmm: false,
        public_key: lean.is_none(),
    })
}

/// Only `Session.key_manifest`: a client never opens a session.
#[pyclass(name = "Session", module = "perseus_client", frozen)]
pub struct PySession;

#[pymethods]
impl PySession {
    /// The key request (JSON for `Client.keygen`) of an eager session over `config` and `keys`. A planned session's
    /// arguments (programs, PCMM, CCMM) are refused: their manifests need pc-py.
    #[staticmethod]
    #[pyo3(signature = (config, keys, programs=None, pcmm=None, ccmm=false))]
    fn key_manifest(config: &PyConfig, keys: &PyKeySource, programs: Option<Bound<'_, PyAny>>, pcmm: Option<Bound<'_, PyAny>>, ccmm: bool) -> PyResult<String> {
        if programs.is_some() || pcmm.is_some() || ccmm {
            return Err(PyValueError::new_err("perseus_client: the eager key manifest only (programs, PCMM and CCMM need perseus._pc)"));
        }
        eager_spec(&config.0, keys)?.to_json().map_err(cerr)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKeySource>()?;
    m.add_class::<PySession>()?;
    Ok(())
}
