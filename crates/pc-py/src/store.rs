//! `PtStore` for Python: the session's plaintext store as the planned driver fills it. A bound program reads its
//! plaintexts by key: `<block>/pt_N` at the level the compiler resolved (`Program.pt_keys`). `stage` encodes on the
//! device (DeviceFft, the slot values the impl layer computed) into the key's slot: a key that is already resident
//! is re-encoded IN PLACE (same slot, same address), so a per-position mask changes value without invalidating the
//! CUDA graphs that read it; stream order puts the rewrite after every earlier reader.
use crate::session::{gerr, nogil, Dev, State, W};
use pc_core::encode::Complex;
use pc_cuda::{Arena, Compute};
use pc_gpu::encode_dev::{Payload2, PtForm};
use pc_gpu::pt_store::PtKey;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::rc::Rc;

/// A slot vector (numpy / sequence; real or complex) as host complex values.
pub(crate) fn host_values(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<Complex>> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (obj,))?.call_method0("ravel")?;
    let kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;
    if kind == "c" {
        let a = np.call_method1("ascontiguousarray", (arr, "complex128"))?.call_method1("view", ("float64",))?;
        let v = PyBuffer::<f64>::get(&a)?.to_vec(py)?;
        Ok(v.chunks_exact(2).map(|p| Complex::new(p[0], p[1])).collect())
    } else {
        let a = np.call_method1("ascontiguousarray", (arr, "float64"))?;
        let v = PyBuffer::<f64>::get(&a)?.to_vec(py)?;
        Ok(v.into_iter().map(|x| Complex::new(x, 0.0)).collect())
    }
}

/// `stage` / `cache` items [(name, level, values[, deg])] as keys, host values and degrees.
fn stage_items(py: Python<'_>, items: &[Bound<'_, PyAny>], what: &str) -> PyResult<Vec<(PtKey, Vec<Complex>, u8)>> {
    let mut parsed = Vec::with_capacity(items.len());
    for it in items {
        let t: Vec<Bound<'_, PyAny>> = it.extract()?;
        if t.len() < 3 {
            return Err(PyValueError::new_err(format!("PtStore.{what} item = (name, level, values[, deg])")));
        }
        let name: String = t[0].extract()?;
        let level: u32 = t[1].extract()?;
        let deg: u8 = if t.len() > 3 { t[3].extract()? } else { 1 };
        parsed.push((PtKey::new(name, level, PtForm::Eval), host_values(py, &t[2])?, deg));
    }
    Ok(parsed)
}

#[pyclass(name = "PtStore", module = "perseus._pc", unsendable)]
pub struct PyPtStore {
    pub(crate) dev: Rc<Dev>,
}

#[pymethods]
impl PyPtStore {
    /// Encode `items` = [(name, level, values[, deg])] on the device into their slots (in place when resident).
    /// Returns the number encoded. Values: N/2 slot values (or a power-of-two count, periodically extended).
    fn stage(&self, py: Python<'_>, items: Vec<Bound<'_, PyAny>>) -> PyResult<usize> {
        let parsed = stage_items(py, &items, "stage")?;
        let dev = &self.dev;
        let payload = dev.payload();
        let n = parsed.len();
        nogil(py, || -> PyResult<()> {
            let mut st = dev.st.borrow_mut();
            let State { pts, ex, ops, .. } = &mut *st;
            for (key, values, deg) in &parsed {
                let slots = values.len().next_power_of_two().max(1);
                pts.encode_into(&dev.ctx, ex, ops, &dev.sh, key.clone(), Payload2::Slots { values, slots, payload }, *deg).map_err(gerr)?;
            }
            Ok(())
        })?;
        Ok(n)
    }

    /// The position cache. Encode `items` = [(name, level, values[, deg])] into their slots as `stage` does, and keep a
    /// copy of each encoding in the session's own device allocation (sized to the items); returns the copies' ids in
    /// item order, which `restore` takes. Once per session, at setup: a planned decode encodes every position's
    /// plaintexts here instead of in its tokens. The slots are left holding the last item of each key.
    fn cache(&self, py: Python<'_>, items: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<usize>> {
        let parsed = stage_items(py, &items, "cache")?;
        let dev = &self.dev;
        let payload = dev.payload();
        nogil(py, || -> PyResult<Vec<usize>> {
            let mut st = dev.st.borrow_mut();
            let State { pts, ex, ops, pt_cache, pt_cache_arena, .. } = &mut *st;
            if pt_cache_arena.is_some() {
                return Err(PyRuntimeError::new_err("PtStore.cache: the session's position cache is already filled"));
            }
            let words = parsed.iter().map(|(k, _, _)| pts.words_at(&dev.sh, k.level)).collect::<pc_gpu::Result<Vec<_>>>().map_err(gerr)?;
            let bytes: usize = words.iter().map(|&w| Arena::bytes_for::<W>(w)).sum();
            let arena = Box::new(Arena::new(&dev.device, bytes).map_err(|x| gerr(x.into()))?);
            // SAFETY: the arena is boxed and declared after `pt_cache` in State, so it outlives the buffers carved here
            let a: &'static Arena = unsafe { &*(arena.as_ref() as *const Arena) };
            *pt_cache_arena = Some(arena);
            let mut ids = Vec::with_capacity(parsed.len());
            for ((key, values, deg), w) in parsed.iter().zip(words) {
                let slots = values.len().next_power_of_two().max(1);
                pts.encode_into(&dev.ctx, ex, ops, &dev.sh, key.clone(), Payload2::Slots { values, slots, payload }, *deg).map_err(gerr)?;
                let mut buf = a.alloc::<Compute, W>("pt_cache", w).map_err(|x| gerr(x.into()))?;
                let meta = pts.copy_out(&dev.ctx, ex, &dev.sh, key, &mut buf).map_err(gerr)?;
                ids.push(pt_cache.len());
                pt_cache.push((buf, meta));
            }
            ex.compute.sync().map_err(|x| gerr(x.into()))?;
            Ok(ids)
        })
    }

    /// Overwrite the slots of `pairs` = [(name, level, id)] in place with the cached encodings `id` (device copies on
    /// the compute lane, after every earlier reader): a position's plaintexts with no encoding in the clock. Each key
    /// must be resident. Returns the number restored.
    fn restore(&self, py: Python<'_>, pairs: Vec<(String, u32, usize)>) -> PyResult<usize> {
        let dev = &self.dev;
        nogil(py, || -> PyResult<usize> {
            let mut st = dev.st.borrow_mut();
            let State { pts, ex, pt_cache, .. } = &mut *st;
            for (name, level, id) in &pairs {
                let (buf, meta) = pt_cache.get(*id).ok_or_else(|| PyValueError::new_err(format!("PtStore.restore: no cached plaintext {id}")))?;
                pts.copy_in(&dev.ctx, ex, &dev.sh, &PtKey::new(name.clone(), *level, PtForm::Eval), buf, *meta).map_err(gerr)?;
            }
            Ok(pairs.len())
        })
    }

    /// Join staged uploads of `keys` = [(name, level)] into the compute lane.
    fn adopt(&self, keys: Vec<(String, u32)>) -> PyResult<()> {
        let mut st = self.dev.st.borrow_mut();
        let State { pts, ex, .. } = &mut *st;
        let ks: Vec<PtKey> = keys.into_iter().map(|(n, l)| PtKey::new(n, l, PtForm::Eval)).collect();
        pts.adopt(&self.dev.ctx, ex, &ks).map_err(gerr)
    }

    /// Drop `keys` = [(name, level)]; returns how many were resident. A bound program reading one fails its next
    /// run (`resolve_pts`) until it is staged again.
    fn evict(&self, keys: Vec<(String, u32)>) -> PyResult<usize> {
        let mut st = self.dev.st.borrow_mut();
        let State { pts, ex, .. } = &mut *st;
        let mut n = 0;
        for (name, level) in keys {
            n += pts.evict(ex, &PtKey::new(name, level, PtForm::Eval)).map_err(gerr)? as usize;
        }
        Ok(n)
    }

    fn has(&self, name: &str, level: u32) -> bool {
        self.dev.st.borrow().pts.has(&PtKey::new(name, level, PtForm::Eval))
    }

    fn __len__(&self) -> usize {
        self.dev.st.borrow().pts.len()
    }

    fn __repr__(&self) -> PyResult<String> {
        let st = self.dev.st.try_borrow().map_err(|_| PyRuntimeError::new_err("session busy"))?;
        Ok(format!("PtStore({} resident of {} slots)", st.pts.len(), st.pts.cfg.slots))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPtStore>()
}
