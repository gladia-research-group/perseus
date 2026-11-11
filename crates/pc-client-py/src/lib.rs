//! `perseus_client`: the client half of `perseus._pc` without CUDA, for user devices (a laptop, an Apple Silicon
//! Mac): `Config` (pc-py's own config.rs), `ClientSecret`, `Client`, `bundle_info`, `contract_hash`, and the eager
//! key request (`KeySource`, `Session.key_manifest`). Same classes, same signatures, same bytes: code written against
//! `perseus._pc` for a client runs on `perseus_client` unchanged.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod client;
#[path = "../../pc-py/src/config.rs"]
mod config;
mod manifest;

pyo3::create_exception!(perseus_client, ConfigError, PyValueError, "A Config value or field the typed config refuses.");

pub(crate) fn config_err(e: pc_config::Error) -> PyErr {
    ConfigError::new_err(e.to_string())
}

/// A client error as a Python ValueError (wire format, spec, chain mismatch).
pub(crate) fn cerr(e: pc_client::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Run `f` with the GIL released (pc-py's `nogil`: the unsendable client stays on this thread meanwhile).
pub(crate) fn nogil<T, F: FnOnce() -> T>(py: Python<'_>, f: F) -> T {
    struct Unsafe<X>(X);
    // SAFETY: Python::detach runs the closure on this thread; only the GIL is released meanwhile
    unsafe impl<X> Send for Unsafe<X> {}
    impl<X> Unsafe<X> {
        fn into_inner(self) -> X {
            self.0
        }
    }
    let w = Unsafe(f);
    // a method call on `w` captures the whole wrapper (a field access would capture the non-Send field)
    let r = py.detach(move || Unsafe(w.into_inner()()));
    r.into_inner()
}

#[pymodule]
fn perseus_client(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ConfigError", m.py().get_type::<ConfigError>())?;
    config::register(m)?;
    client::register(m)?;
    manifest::register(m)?;
    Ok(())
}
