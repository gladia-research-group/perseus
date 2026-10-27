//! `perseus._pc`: the Python face of perseus. GPU-free so far: the typed `Config` (pc-config), the
//! `Shadow` rules engine and its value types (pc-shadow: `Tag`, `CtMeta`, `PtMeta`, `Route`, `Recipe`), and a
//! read-only `EmProgram` view; the device `Session` (`session.rs`) and the eager `Fhe` / `Ct` (`fhe.rs`); the
//! client (`client.rs`: keygen, encrypt, decrypt without a device) and the ciphertext wire frames.
use pyo3::prelude::*;

mod client;
mod config;
mod em;
mod fhe;
mod matmul;
mod meta;
mod pcmm;
mod program;
mod session;
mod shadow;
mod store;
mod tag;
mod tfhe;
mod word64;

pyo3::create_exception!(_pc, LevelError, pyo3::exceptions::PyRuntimeError, "A Shadow rule refused an op (level, degree, route or chain range).");
pyo3::create_exception!(_pc, ConfigError, pyo3::exceptions::PyValueError, "A Config value or field the typed config refuses.");

pub(crate) fn level_err(e: pc_shadow::Error) -> PyErr {
    LevelError::new_err(e.to_string())
}
pub(crate) fn config_err(e: pc_config::Error) -> PyErr {
    ConfigError::new_err(e.to_string())
}

#[pymodule]
fn _pc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("LevelError", py.get_type::<LevelError>())?;
    m.add("ConfigError", py.get_type::<ConfigError>())?;
    tag::register(m)?;
    meta::register(m)?;
    shadow::register(m)?;
    config::register(m)?;
    em::register(m)?;
    session::register(m)?;
    fhe::register(m)?;
    program::register(m)?;
    pcmm::register(m)?;
    store::register(m)?;
    client::register(m)?;
    word64::register(m)?; // _pc.w64 (u64 words, n64 chain)
    Ok(())
}
