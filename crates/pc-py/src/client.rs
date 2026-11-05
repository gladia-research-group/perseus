//! The client side for Python (`_pc.ClientSecret`, `_pc.Client`: pc-client, no device) and the session's ends of
//! the ciphertext wire format (`_pc.export_ct`, `_pc.import_ct`; docs/WIRE.md).
//!
//! A client process only ever builds `ClientSecret` / `Client`: no `Session`, no device. A server session built
//! with `Session.from_bundle` holds no secret; ciphertexts cross between the two as wire frames (bytes).
use crate::config::PyConfig;
use crate::fhe::PyCt;
use crate::session::{cerr, gerr, nogil, PySession, State};
use pc_client::wire::{CT_MAGIC, VERSION};
use pc_client::{Client, ClientSecret, CtHeader, CtWire, WireMeta};
use pc_core::encode::{Complex, Payload};
use pc_core::keys::Prng;
use pc_shadow::CtMeta;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

fn prng_of(seed: Option<u64>) -> PyResult<Prng> {
    match seed {
        Some(s) => Ok(Prng::from_seed([0xc11e_0001, s as u32, (s >> 32) as u32, 4, 5, 6, 7, 8], 0)),
        None => Prng::from_os().map_err(|e| PyRuntimeError::new_err(e.to_string())),
    }
}

/// The client's secret (dense ternary + sparse encapsulation key). `to_bytes` is the client's own storage format;
/// it never goes to a server.
#[pyclass(name = "ClientSecret", module = "perseus._pc", frozen)]
pub struct PyClientSecret(pub Vec<u8>);

#[pymethods]
impl PyClientSecret {
    /// A fresh secret for chain `chain` (OS entropy, or `seed` for reproducible tests).
    #[staticmethod]
    #[pyo3(signature = (chain="n32", seed=None))]
    fn generate(chain: &str, seed: Option<u64>) -> PyResult<Self> {
        let c = pc_client::spec::chain_of(chain).map_err(cerr)?;
        let mut g = prng_of(seed)?;
        Ok(PyClientSecret(ClientSecret::generate(&mut g, c.n).map_err(cerr)?.to_bytes()))
    }
    #[staticmethod]
    fn from_bytes(b: Vec<u8>) -> PyResult<Self> {
        ClientSecret::from_bytes(&b).map_err(cerr)?;
        Ok(PyClientSecret(b))
    }
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.0)
    }
}

/// The client: keygen from a server's key request into a bundle file, encryption of slot vectors, decryption.
/// `complex`: the payload (complex slots, or real: imaginary parts dropped).
#[pyclass(name = "Client", module = "perseus._pc", unsendable)]
pub struct PyClient {
    c: Client,
    prng: Prng,
    /// a master public key (`load_public_key`): `encrypt(..., public=True)`
    pk: Option<(Vec<u32>, usize)>,
    payload: Payload,
    /// precomputed encryption parts by (level, regen) (`precompute`), used first by `encrypt(..., seeded=True)`
    pre: std::collections::HashMap<(u32, bool), Vec<pc_client::PreEnc>>,
}

fn values_of(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<Complex>> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (obj, "complex128"))?.call_method0("ravel")?;
    let flat = arr.call_method1("view", ("float64",))?;
    let v = PyBuffer::<f64>::get(&flat)?.to_vec(py)?;
    Ok(v.chunks_exact(2).map(|p| Complex::new(p[0], p[1])).collect())
}

fn np_c128<'py>(py: Python<'py>, v: &[Complex]) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let a = np.call_method1("empty", (v.len(), "complex128"))?;
    let flat: Vec<f64> = v.iter().flat_map(|c| [c.re, c.im]).collect();
    PyBuffer::<f64>::get(&a.call_method1("view", ("float64",))?)?.copy_from_slice(py, &flat)?;
    Ok(a)
}

fn np_f64<'py>(py: Python<'py>, v: &[f64]) -> PyResult<Bound<'py, PyAny>> {
    let a = py.import("numpy")?.call_method1("empty", (v.len(), "float64"))?;
    PyBuffer::<f64>::get(&a)?.copy_from_slice(py, v)?;
    Ok(a)
}

fn meta_dict<'py>(py: Python<'py>, m: &WireMeta) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("level", m.level)?;
    d.set_item("deg", m.deg)?;
    d.set_item("nf", m.nf)?;
    d.set_item("slots", m.slots)?;
    Ok(d)
}

impl PyClient {
    /// A fresh encryption's metadata at `level`; with `nf` the scale is the caller's, so `level` may be off the
    /// composite grid (a ciphertext of any top limb, e.g. 3 rows on n32, which has no scaling factor of its own).
    fn meta_at(&self, level: u32, deg: u8, nf: Option<f64>) -> PyResult<WireMeta> {
        match nf {
            Some(x) => {
                self.c.top_of(level).map_err(cerr)?;
                Ok(WireMeta { level, deg, nf: x, slots: (self.c.chain.n / 2) as u32, tag: serde_json::Value::Null })
            }
            None => self.c.fresh_meta(level, deg).map_err(cerr),
        }
    }
}

#[pymethods]
impl PyClient {
    #[new]
    #[pyo3(signature = (secret, chain="n32", complex=true, seed=None))]
    fn new(secret: &PyClientSecret, chain: &str, complex: bool, seed: Option<u64>) -> PyResult<Self> {
        let s = ClientSecret::from_bytes(&secret.0).map_err(cerr)?;
        Ok(PyClient { c: Client::new(chain, s).map_err(cerr)?, prng: prng_of(seed)?, payload: if complex { Payload::Complex } else { Payload::Real }, pk: None, pre: Default::default() })
    }

    #[getter]
    fn params_hash(&self) -> String {
        self.c.params_hash.clone()
    }
    #[getter]
    fn slots(&self) -> usize {
        self.c.chain.n / 2
    }
    /// log2 of every prime of the chain by id, Q then P (a key shape's modulus: the product of the primes it stores;
    /// the client's `refuse` bounds it per role).
    #[getter]
    fn prime_log2(&self) -> Vec<f64> {
        self.c.chain.primes().iter().map(|&p| (p as f64).log2()).collect()
    }
    /// (Q primes, P primes, P' = the sparse secrets' special primes) of the chain.
    #[getter]
    fn prime_counts(&self) -> (usize, usize, usize) {
        (self.c.chain.q.len(), self.c.chain.p.len(), self.c.chain.p_sparse)
    }

    /// Generate every key of `spec` (a server's `Session.key_manifest` JSON) into the bundle file `path`, on a
    /// rayon pool of `threads`. Returns the timings and sizes.
    #[pyo3(signature = (spec, path, threads=8))]
    fn keygen<'py>(&mut self, py: Python<'py>, spec: &str, path: std::path::PathBuf, threads: usize) -> PyResult<Bound<'py, PyDict>> {
        let spec = pc_client::KeySpec::from_json(spec).map_err(cerr)?;
        let (c, g) = (&mut self.c, &mut self.prng);
        let st = nogil(py, || -> pc_client::Result<pc_client::KeyGenStats> {
            let f = std::io::BufWriter::with_capacity(1 << 24, std::fs::File::create(&path)?);
            c.keygen(&spec, f, g, threads)
        })
        .map_err(cerr)?;
        let d = PyDict::new(py);
        let j = serde_json::to_value(&st).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        if let serde_json::Value::Object(m) = j {
            for (k, v) in m {
                if let Some(x) = v.as_f64() {
                    d.set_item(k, x)?;
                }
            }
        }
        Ok(d)
    }

    /// Metadata of a fresh encryption at `level` (the level's scaling factor^deg, every slot): a dict.
    #[pyo3(signature = (level, deg=1))]
    fn fresh_meta<'py>(&self, py: Python<'py>, level: u32, deg: u8) -> PyResult<Bound<'py, PyDict>> {
        meta_dict(py, &self.c.fresh_meta(level, deg).map_err(cerr)?)
    }

    /// Wire frame (bytes) of a fresh encryption of `values` (a slot vector, zero-padded) at `level` (scale: the
    /// level's sf^deg unless `nf`; `slots`: live slots, default every slot).
    /// The public key of the key bundle at `bundle` (its `public_key` entry): `encrypt(..., public=True)` encrypts under
    /// it, so a party without that key's secret (a user, under the master key) can send ciphertexts the server computes on.
    fn load_public_key(&mut self, bundle: std::path::PathBuf) -> PyResult<()> {
        let mut rd = pc_client::BundleReader::open(&bundle).map_err(cerr)?;
        let (i, rows) = rd
            .header
            .entries
            .iter()
            .enumerate()
            .find_map(|(i, e)| if let pc_client::Entry::PublicKey { rows, .. } = e { Some((i, *rows)) } else { None })
            .ok_or_else(|| PyValueError::new_err("key bundle without a public key"))?;
        self.pk = Some((rd.read(i).map_err(cerr)?, rows));
        Ok(())
    }

    /// `seeded`: c1 sent as the seeds it is drawn from (half the frame; the server's `import_ct` expands it).
    /// `public`: under the loaded public key (`load_public_key`) instead of this client's secret. `regen` (with `seeded`):
    /// c1 from the random-access expander, so the server can keep the seeds (`import_ct(..., regen=True)`).
    #[pyo3(signature = (values, level, deg=1, nf=None, slots=None, seeded=false, public=false, regen=false))]
    #[allow(clippy::too_many_arguments)]
    fn encrypt<'py>(&mut self, py: Python<'py>, values: &Bound<'_, PyAny>, level: u32, deg: u8, nf: Option<f64>, slots: Option<u32>, seeded: bool, public: bool, regen: bool) -> PyResult<Bound<'py, PyBytes>> {
        let v = values_of(py, values)?;
        let mut meta = self.meta_at(level, deg, nf)?;
        if let Some(x) = slots {
            meta.slots = x;
        }
        if v.len() > meta.slots as usize {
            return Err(PyValueError::new_err(format!("{} values exceed {} slots", v.len(), meta.slots)));
        }
        if seeded && !public {
            if let Some(pre) = self.pre.get_mut(&(level, regen)).and_then(|v| v.pop()) {
                let (c, p) = (&self.c, self.payload);
                let w = nogil(py, || c.encrypt_pre(&v, &meta, p, pre).and_then(|w| w.to_bytes())).map_err(cerr)?;
                return Ok(PyBytes::new(py, &w));
            }
        }
        let (c, g, p) = (&self.c, &mut self.prng, self.payload);
        let w = if public {
            let (pk, rows) = self.pk.as_ref().ok_or_else(|| PyValueError::new_err("encrypt(public=True): load_public_key first"))?;
            nogil(py, || c.encrypt_public(&[v], &meta, p, g, pk, *rows).and_then(|w| w.to_bytes())).map_err(cerr)?
        } else {
            nogil(py, || c.encrypt_as(&[v], &meta, p, g, seeded, regen).and_then(|w| w.to_bytes())).map_err(cerr)?
        };
        Ok(PyBytes::new(py, &w))
    }

    /// Prepare `count` seeded encryptions at `level` ahead of time (the noise and c1 s parts; parallel, each on a stream
    /// seeded from this client's): `encrypt(..., level, seeded=True, regen=regen)` then only encodes and adds. Returns
    /// how many are ready at that level.
    #[pyo3(signature = (level, count, regen=false))]
    fn precompute(&mut self, py: Python<'_>, level: u32, count: usize, regen: bool) -> PyResult<usize> {
        use rayon::prelude::*;
        let keys: Vec<[u32; 8]> = (0..count).map(|_| self.prng.ksk_seed().0).collect();
        let c = &self.c;
        let made = nogil(py, || keys.par_iter().map(|&k| c.precompute(level, &mut pc_core::keys::Prng::from_seed(k, 0), regen)).collect::<pc_client::Result<Vec<_>>>()).map_err(cerr)?;
        let q = self.pre.entry((level, regen)).or_default();
        q.extend(made);
        Ok(q.len())
    }

    /// A seeded frame of the plaintext whose coefficients are `coeffs` (n reals, at the level's scale unless `nf`), no
    /// slot FFT (e.g. selection weights given as coefficients); a precomputed part (`precompute`) when one is ready.
    #[pyo3(signature = (coeffs, level, nf=None, regen=false))]
    fn encrypt_coeffs<'py>(&mut self, py: Python<'py>, coeffs: Vec<f64>, level: u32, nf: Option<f64>, regen: bool) -> PyResult<Bound<'py, PyBytes>> {
        let meta = self.meta_at(level, 1, nf)?;
        let pre = match self.pre.get_mut(&(level, regen)).and_then(|v| v.pop()) {
            Some(p) => p,
            None => self.c.precompute(level, &mut self.prng, regen).map_err(cerr)?,
        };
        let c = &self.c;
        let w = nogil(py, || c.encrypt_pre_coeffs(&coeffs, &meta, pre).and_then(|w| w.to_bytes())).map_err(cerr)?;
        Ok(PyBytes::new(py, &w))
    }

    /// Decoded slots (complex128 arrays, one per ciphertext of the frame) of a wire frame.
    fn decrypt<'py>(&self, py: Python<'py>, frame: &[u8]) -> PyResult<Bound<'py, PyList>> {
        let w = CtWire::from_bytes(frame).map_err(cerr)?;
        let (c, p) = (&self.c, self.payload);
        let vals = nogil(py, || c.decrypt(&w, p)).map_err(cerr)?;
        let out = PyList::empty(py);
        for v in &vals {
            out.append(np_c128(py, v)?)?;
        }
        Ok(out)
    }

    /// The plaintext coefficients over the frame's scale (float64 arrays, one per ciphertext of the frame), no slot
    /// FFT: e.g. integer data carried as coefficients.
    fn decrypt_coeffs<'py>(&self, py: Python<'py>, frame: &[u8]) -> PyResult<Bound<'py, PyList>> {
        let w = CtWire::from_bytes(frame).map_err(cerr)?;
        let c = &self.c;
        let vals = nogil(py, || c.decrypt_coeffs(&w)).map_err(cerr)?;
        let out = PyList::empty(py);
        for v in vals {
            out.append(np_f64(py, &v)?)?;
        }
        Ok(out)
    }

    /// `encrypt` of several slot vectors in parallel (one rayon task each, each on its own stream seeded from this
    /// client's): their frames, in order. `nf`, `seeded`, `regen` as `encrypt`'s.
    #[pyo3(signature = (values, level, nf=None, seeded=false, regen=false))]
    fn encrypt_many<'py>(&mut self, py: Python<'py>, values: Vec<Bound<'_, PyAny>>, level: u32, nf: Option<f64>, seeded: bool, regen: bool) -> PyResult<Bound<'py, PyList>> {
        use rayon::prelude::*;
        let vs = values.iter().map(|v| values_of(py, v)).collect::<PyResult<Vec<_>>>()?;
        let meta = self.meta_at(level, 1, nf)?;
        if let Some(v) = vs.iter().find(|v| v.len() > meta.slots as usize) {
            return Err(PyValueError::new_err(format!("{} values exceed {} slots", v.len(), meta.slots)));
        }
        let keys: Vec<[u32; 8]> = (0..vs.len()).map(|_| self.prng.ksk_seed().0).collect();
        let (c, p) = (&self.c, self.payload);
        let frames = nogil(py, || {
            vs.par_iter()
                .zip(&keys)
                .map(|(v, &k)| {
                    let mut g = pc_core::keys::Prng::from_seed(k, 0);
                    c.encrypt_as(std::slice::from_ref(v), &meta, p, &mut g, seeded, regen).and_then(|w| w.to_bytes())
                })
                .collect::<pc_client::Result<Vec<_>>>()
        })
        .map_err(cerr)?;
        let out = PyList::empty(py);
        for f in frames {
            out.append(PyBytes::new(py, &f))?;
        }
        Ok(out)
    }

    /// `decrypt` of several frames in parallel (one rayon task per frame): their decoded slots, frame by frame.
    fn decrypt_many<'py>(&self, py: Python<'py>, frames: Vec<Vec<u8>>) -> PyResult<Bound<'py, PyList>> {
        use rayon::prelude::*;
        let (c, p) = (&self.c, self.payload);
        let vals = nogil(py, || {
            frames
                .par_iter()
                .map(|f| CtWire::from_bytes(f).and_then(|w| c.decrypt(&w, p)))
                .collect::<pc_client::Result<Vec<_>>>()
        })
        .map_err(cerr)?;
        let out = PyList::empty(py);
        for v in vals {
            let l = PyList::empty(py);
            for x in &v {
                l.append(np_c128(py, x)?)?;
            }
            out.append(l)?;
        }
        Ok(out)
    }

    /// Header of a wire frame: {level, deg, nf, slots, k, rows}.
    #[staticmethod]
    fn frame_meta<'py>(py: Python<'py>, frame: &[u8]) -> PyResult<Bound<'py, PyDict>> {
        let w = CtWire::from_bytes(frame).map_err(cerr)?;
        let d = meta_dict(py, &w.header.meta)?;
        d.set_item("k", w.header.k)?;
        d.set_item("rows", w.header.rows)?;
        Ok(d)
    }

    /// The switch relation of every switching key in the bundle at `path` (max |e| of `b + a s_new - [digit]
    /// (P mod q) s_old`): {keys, max_err, worst}. `limit`: check the first `limit` keys only.
    #[pyo3(signature = (path, limit=None))]
    fn check_bundle<'py>(&self, py: Python<'py>, path: std::path::PathBuf, limit: Option<usize>) -> PyResult<Bound<'py, PyDict>> {
        let c = &self.c;
        let (n, mx, worst) = nogil(py, || -> pc_client::Result<(usize, i64, String)> {
            let mut r = pc_client::BundleReader::open(&path)?;
            let (bits, rw) = (r.header.bits, r.header.row_words);
            let (mut n, mut mx, mut worst) = (0usize, 0i64, String::new());
            for i in 0..r.header.entries.len() {
                if let pc_client::Entry::Ksk { key, seed, .. } = r.header.entries[i].clone() {
                    if matches!(key.role, pc_client::Role::SpruAtob) || limit.is_some_and(|l| n >= l) {
                        continue;
                    }
                    let v = r.read(i)?;
                    let e = c.ksk_relation_max_err(&key, &pc_core::keys::KskSeed(seed), &v, bits, rw)?;
                    if e > mx {
                        mx = e;
                        worst = format!("{:?}", key.role);
                    }
                    n += 1;
                }
            }
            Ok((n, mx, worst))
        })
        .map_err(cerr)?;
        let d = PyDict::new(py);
        d.set_item("keys", n)?;
        d.set_item("max_err", mx)?;
        d.set_item("worst", worst)?;
        Ok(d)
    }
}

/// Bundle file summary: {bytes, entries: {kind: count}, bytes_by_kind: {kind: bytes}, contract, params_hash}.
#[pyfunction]
fn bundle_info<'py>(py: Python<'py>, path: std::path::PathBuf) -> PyResult<Bound<'py, PyDict>> {
    let r = pc_client::BundleReader::open(&path).map_err(cerr)?;
    let d = PyDict::new(py);
    let (cnt, by) = (PyDict::new(py), PyDict::new(py));
    for e in &r.header.entries {
        let k = match e {
            pc_client::Entry::Ksk { .. } => "ksk",
            pc_client::Entry::SpruCs { .. } => "spru_cs",
            pc_client::Entry::Pcmm { .. } => "pcmm",
            pc_client::Entry::Ccmm { .. } => "ccmm",
            pc_client::Entry::PublicKey { .. } => "public_key",
        };
        let c: usize = cnt.get_item(k)?.map_or(Ok(0), |x| x.extract())?;
        cnt.set_item(k, c + 1)?;
        let b: u64 = by.get_item(k)?.map_or(Ok(0), |x| x.extract())?;
        by.set_item(k, b + 4 * e.words() as u64)?;
    }
    d.set_item("bytes", r.bytes())?;
    d.set_item("entries", cnt)?;
    d.set_item("bytes_by_kind", by)?;
    d.set_item("contract", r.header.contract.clone())?;
    d.set_item("params_hash", r.header.params_hash.clone())?;
    Ok(d)
}

/// The plan contract of `config` (what `Session.key_manifest` puts in the request and the bundle must carry).
#[pyfunction]
fn contract_hash(config: &PyConfig) -> PyResult<String> {
    crate::session::contract_hash(&config.0)
}

/// `import_ct(..., low=True, regen=True)`: a seeded regen frame kept as c0 and its seeds in a seeded low slot.
fn import_regen(py: Python<'_>, session: &PySession, frame: &[u8]) -> PyResult<PyCt> {
    let dev = session.dev.clone();
    let w = CtWire::from_bytes(frame).map_err(cerr)?;
    if !(w.header.seeded && w.header.regen) {
        return Err(PyValueError::new_err("import_ct(regen=True): a low import of a seeded regen frame (encrypt(..., seeded=True, regen=True))"));
    }
    let h = &w.header;
    let ph = pc_client::spec::params_hash(&dev.ctx.chain);
    if h.params_hash != ph || h.n != dev.ctx.lay.n || h.k != 1 {
        return Err(PyValueError::new_err(format!("import_ct: frame for params {} n {} k {}; this session runs {ph} n {} (k = 1)", h.params_hash, h.n, h.k, dev.ctx.lay.n)));
    }
    let top = dev.ctx.top_of(h.meta.level).map_err(gerr)?;
    if h.rows != top + 1 {
        return Err(PyValueError::new_err(format!("import_ct: {} rows at level {} (top {top})", h.rows, h.meta.level)));
    }
    let tag = if h.meta.tag.is_null() { Default::default() } else { serde_json::from_value(h.meta.tag.clone()).map_err(|e| PyValueError::new_err(format!("import_ct: tag {e}")))? };
    let meta = CtMeta { level: h.meta.level, deg: h.meta.deg, nf: h.meta.nf, slots: h.meta.slots, tag };
    let mut out = PyCt::acquire_low(&dev, meta, top, true)?;
    let rows: Vec<usize> = (0..h.rows).collect();
    let n = h.n;
    {
        let st = dev.st.borrow();
        let c = out.ct_mut_ref()?;
        nogil(py, || -> pc_gpu::Result<()> {
            // seeded frame: c0's rows, then the rows' 8-word seeds; a packed slot takes c0 packed here
            if c.buf.packed {
                let pw = pc_kernels::pack_row_words(n);
                let mut packed = vec![0u32; h.rows * pw];
                for (src, dst) in w.data[..h.rows * n].chunks_exact(n).zip(packed.chunks_exact_mut(pw)) {
                    if !pc_kernels::pack_row(src, dst) {
                        return Err(pc_gpu::Error::Shape { what: "import_ct", msg: format!("a residue of {} bits or more into a packed slot", pc_kernels::PACK_BITS) });
                    }
                }
                c.buf.upload_packed(&st.ex.compute, &packed)?;
            } else {
                c.buf.upload_rows(&st.ex.compute, 0, 0, &rows, &w.data[..h.rows * n])?;
            }
            c.buf.upload_seeds(&st.ex.compute, &w.data[h.rows * n..h.rows * (n + 8)])
        })
        .map_err(gerr)?;
    }
    Ok(out)
}

/// A frame `import_ct` uploaded with its checksum left to a thread, and the frame's buffer (held until the thread is
/// joined: the thread reads it).
struct Pending {
    ok: std::thread::JoinHandle<bool>,
    _frame: PyBuffer<u8>,
}

/// The imports whose checksum is still being verified (`import_ct`), joined by `export_ct` before it writes a frame.
static PENDING: std::sync::Mutex<Vec<Pending>> = std::sync::Mutex::new(Vec::new());

/// Wait for the checksums of every frame imported so far; an error if any of them did not match.
fn checked_imports() -> PyResult<()> {
    let v = std::mem::take(&mut *PENDING.lock().map_err(|_| PyRuntimeError::new_err("import_ct: checksum threads poisoned"))?);
    let n = v.len();
    let bad = v.into_iter().map(|p| p.ok.join().unwrap_or(false)).filter(|ok| !ok).count();
    if bad > 0 {
        return Err(PyValueError::new_err(format!("import_ct: ciphertext checksum mismatch ({bad} of the {n} frames imported since the last export)")));
    }
    Ok(())
}

/// Wire frame of a session ciphertext (a pending rescale is realized on a copy first: the client decodes degree 1).
/// Every frame imported before it has passed its checksum (`import_ct`) when it returns.
#[pyfunction]
#[pyo3(signature = (session, ct, bits=None))]
fn export_ct<'py>(py: Python<'py>, session: &PySession, ct: PyRef<'_, PyCt>, bits: Option<u32>) -> PyResult<Bound<'py, PyBytes>> {
    checked_imports()?;
    let dev = session.dev.clone();
    if ct.ct_ref().buf.seeds {
        return Err(PyValueError::new_err("export_ct: a seeded ciphertext (import_ct(..., regen=True)) is an input of mult_sum_many only"));
    }
    let meta = ct.ct_ref().meta;
    if meta.deg == 2 {
        ct.check_full()?;                                   // the realize copy reads through the full layout
    }
    let tmp = if meta.deg == 2 {
        let mut t = PyCt::acquire(&dev, meta)?;
        let mut st = dev.st.borrow_mut();
        let State { ops, ex, .. } = &mut *st;
        ops.realize_to(&dev.ctx, ex, &dev.sh, t.ct_mut_ref()?, ct.ct_ref()).map_err(gerr)?;
        Some(t)
    } else {
        None
    };
    let c = tmp.as_ref().map_or(ct.ct_ref(), |t| t.ct_ref());
    let top = dev.ctx.top_of(c.meta.level).map_err(gerr)?;
    let n = dev.ctx.lay.n;
    if let Some(b) = bits {
        // compressed: coefficient form, round(x 2^b / Q) per coefficient (pc_client::wire::pack_coeffs)
        if ct.ct_ref().meta.deg == 2 || c.buf.lay.nq != dev.ctx.lay.nq {
            ct.check_full()?;
        }
        let limbs = {
            let mut st = dev.st.borrow_mut();
            let State { ex, .. } = &mut *st;
            nogil(py, || pc_gpu::limbs::export_limbs(&dev.ctx, ex, c, 0)).map_err(gerr)?
        };
        let mut data = Vec::with_capacity(2 * n);
        for part in 0..2 {
            let rs: Vec<Vec<u64>> = limbs.c[part].iter().map(|r| r.iter().map(|&x| x as u64).collect()).collect();
            let refs: Vec<&[u64]> = rs.iter().map(|r| r.as_slice()).collect();
            data.extend(pc_client::wire::pack_bits(&pc_client::wire::pack_coeffs(&limbs.moduli, &refs, b).map_err(cerr)?, b));
        }
        let m = c.meta;
        let tag = serde_json::to_value(m.tag).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let w = CtWire {
            header: CtHeader { params_hash: pc_client::spec::params_hash(&dev.ctx.chain), n, k: 1, rows: top + 1, meta: WireMeta { level: m.level, deg: m.deg, nf: m.nf, slots: m.slots, tag }, bits: Some(b), seeded: false, regen: false, bits1: None, sparse: false },
            data,
        };
        return Ok(PyBytes::new(py, &w.to_bytes().map_err(cerr)?));
    }
    let st = dev.st.borrow();
    let rows = top + 1;
    let m = c.meta;
    let tag = serde_json::to_value(m.tag).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let header = CtHeader { params_hash: pc_client::spec::params_hash(&dev.ctx.chain), n, k: 1, rows, meta: WireMeta { level: m.level, deg: m.deg, nf: m.nf, slots: m.slots, tag }, bits: None, seeded: false, regen: false, bits1: None, sparse: false };
    let hj = serde_json::to_vec(&header).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    // `CtWire::to_bytes`'s frame, written in place: the header, each part's rows 0..=top (contiguous on the device)
    // downloaded straight into the frame, the checksum over them
    let (d0, part_bytes) = (16 + hj.len(), 4 * rows * n);
    let d1 = d0 + 2 * part_bytes;
    PyBytes::new_with(py, d1 + 8, |out| {
        out[..4].copy_from_slice(CT_MAGIC);
        out[4..8].copy_from_slice(&VERSION.to_le_bytes());
        out[8..16].copy_from_slice(&(hj.len() as u64).to_le_bytes());
        out[16..d0].copy_from_slice(&hj);
        for part in 0..2 {
            let dst = &mut out[d0 + part * part_bytes..d0 + (part + 1) * part_bytes];
            c.buf.buf.download_bytes_into(&st.ex.compute, c.buf.offset(0, part, 0), dst).map_err(|x| gerr(x.into()))?;
        }
        let sum = sum64_le(&out[d0..d1]);
        out[d1..].copy_from_slice(&sum.to_le_bytes());
        Ok(())
    })
}

/// `export_ct(..., bits=b)` of several ciphertexts: the coefficient exports run back to back on the device, the
/// compression and framing in parallel on the host (every frame imported before it has passed its checksum). `bits1`: c1's bits (default `bits`; below 32 bit-packed).
/// `sparse`: each ciphertext first switched to the client's sparse secret (`KeySource(sparse_out=True)`, bottom level):
/// c1's rounding then meets a 32-coefficient secret, so both parts compress further.
#[pyfunction]
/// `user`: switched to that user's key instead (`Session.add_user`): the frames decrypt with the user's secret only.
#[pyo3(signature = (session, cts, bits, bits1=None, sparse=false, user=None))]
#[allow(clippy::too_many_arguments)]
fn export_ct_many<'py>(py: Python<'py>, session: &PySession, cts: Vec<PyRef<'_, PyCt>>, bits: u32, bits1: Option<u32>, sparse: bool, user: Option<String>) -> PyResult<Bound<'py, PyList>> {
    let sparse = sparse || user.is_some();
    use rayon::prelude::*;
    checked_imports()?;
    let dev = session.dev.clone();
    let n = dev.ctx.lay.n;
    let mut limbs = Vec::with_capacity(cts.len());
    for ct in &cts {
        if ct.ct_ref().buf.seeds {
            return Err(PyValueError::new_err("export_ct_many: a seeded ciphertext (import_ct(..., regen=True)) is an input of mult_sum_many only"));
        }
        let meta = ct.ct_ref().meta;
        if meta.deg == 2 || ct.ct_ref().buf.lay.nq != dev.ctx.lay.nq {
            ct.check_full()?;
        }
        let tmp = if meta.deg == 2 || sparse {
            let mut t = PyCt::acquire(&dev, meta)?;
            let mut st = dev.st.borrow_mut();
            let State { ops, ex, plans, groups, users, .. } = &mut *st;
            if meta.deg == 2 {
                ops.realize_to(&dev.ctx, ex, &dev.sh, t.ct_mut_ref()?, ct.ct_ref()).map_err(gerr)?;
            } else {
                pc_program::replay::copy_ct(&dev.ctx, &ex.compute, t.ct_mut_ref()?, ct.ct_ref()).map_err(|x| PyRuntimeError::new_err(x.to_string()))?;
            }
            if sparse {
                let g = match &user {
                    Some(name) => {
                        let u = users.as_ref().ok_or_else(|| PyValueError::new_err("Session without user key slots"))?;
                        let slot = *u.names.get(name).ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(format!("no user {name:?}")))?;
                        &u.groups[slot]
                    }
                    None => groups.d2s.as_ref().ok_or_else(|| PyValueError::new_err("export_ct_many(sparse=True) needs KeySource(sparse_out=True)"))?,
                };
                let tc = t.ct_mut_ref()?;
                let top = tc.top(dev.ctx.shape).map_err(gerr)?;
                if top + 1 != dev.ctx.chain.level_unit {
                    return Err(PyValueError::new_err(format!("sparse export: the ciphertext must sit at the bottom level ({} rows, top {top})", dev.ctx.chain.level_unit)));
                }
                pc_gpu::bts::raise::key_switch_with(&dev.ctx, ex, plans, g, dev.source, &mut tc.buf, 1, top).map_err(gerr)?;
            }
            Some(t)
        } else {
            None
        };
        let c = tmp.as_ref().map_or(ct.ct_ref(), |t| t.ct_ref());
        let mut st = dev.st.borrow_mut();
        let State { ex, .. } = &mut *st;
        let l = nogil(py, || pc_gpu::limbs::export_limbs(&dev.ctx, ex, c, 0)).map_err(gerr)?;
        limbs.push((l, c.meta));
    }
    let hash = pc_client::spec::params_hash(&dev.ctx.chain);
    let frames = nogil(py, || {
        limbs
            .par_iter()
            .map(|(l, m)| -> pc_client::Result<Vec<u8>> {
                let mut data = Vec::with_capacity(2 * n);
                for (part, b) in [(0, bits), (1, bits1.unwrap_or(bits))] {
                    let rs: Vec<Vec<u64>> = l.c[part].iter().map(|r| r.iter().map(|&x| x as u64).collect()).collect();
                    let refs: Vec<&[u64]> = rs.iter().map(|r| r.as_slice()).collect();
                    data.extend(pc_client::wire::pack_bits(&pc_client::wire::pack_coeffs(&l.moduli, &refs, b)?, b));
                }
                let tag = serde_json::to_value(m.tag).map_err(|e| pc_client::Error::Wire(e.to_string()))?;
                let header = CtHeader { params_hash: hash.clone(), n, k: 1, rows: l.moduli.len(), meta: WireMeta { level: m.level, deg: m.deg, nf: m.nf, slots: m.slots, tag }, bits: Some(bits), seeded: false, regen: false, bits1, sparse };
                CtWire { header, data }.to_bytes()
            })
            .collect::<pc_client::Result<Vec<_>>>()
    })
    .map_err(cerr)?;
    let out = PyList::empty(py);
    for f in frames {
        out.append(PyBytes::new(py, &f))?;
    }
    Ok(out)
}

/// A session ciphertext from a wire frame (one ciphertext, this session's chain; bytes or a memoryview, unchanged until
/// the next `export_ct`). The frame's checksum is verified on a thread while the caller goes on; the next `export_ct`
/// fails if it did not match. `low`: into a low slot (the session's `size={"lowPool", "lowRows"}`), an input of
/// `mult_sum` and `export_ct` only. `regen` (with `low`): a `regen` frame kept as c0 and its seeds in a seeded low
/// slot (`size={"lowSeeded"}`), half a low slot (c0 packed on upload when `size={"lowPacked": 1}`); an input of
/// `mult_sum_many` (first operands) only, which draws c1 from the seeds.
#[pyfunction]
#[pyo3(signature = (session, frame, low=false, regen=false))]
fn import_ct(py: Python<'_>, session: &PySession, frame: &Bound<'_, PyAny>, low: bool, regen: bool) -> PyResult<PyCt> {
    if regen {
        if !low {
            return Err(PyValueError::new_err("import_ct(regen=True): a low import of a seeded regen frame (encrypt(..., seeded=True, regen=True))"));
        }
        return import_regen(py, session, &PyBuffer::<u8>::get(frame)?.to_vec(py)?);
    }
    let dev = session.dev.clone();
    // any contiguous bytes-like frame (bytes, or a memoryview into a payload holding several), read where it lies
    let fb = PyBuffer::<u8>::get(frame)?;
    if !fb.is_c_contiguous() {
        return Err(PyValueError::new_err("import_ct: the frame is not a contiguous buffer"));
    }
    // SAFETY: `fb` holds the exporter's buffer (alive and unmoved) until it drops (on an error here, else when the
    // checksum thread it is handed to is joined); C-contiguous u8 items
    let frame: &[u8] = unsafe { std::slice::from_raw_parts(fb.buf_ptr() as *const u8, fb.len_bytes()) };
    let (h, data, sum) = ct_frame(frame)?;
    let h = &h;
    if h.bits.is_some() {
        return Err(PyValueError::new_err("import_ct: a compressed frame (export_ct bits=) is for a client to decrypt"));
    }
    // a seeded frame (c1 from its seed): expanded on the host, its words uploaded from the expansion (its checksum was
    // verified by `from_bytes`)
    let seeded = if h.seeded { Some(CtWire::from_bytes(frame).map_err(cerr)?.expand_seeded(&dev.ctx.chain.q).map_err(cerr)?) } else { None };
    let sbytes: Option<Vec<u8>> = seeded.as_ref().map(|w| w.data.iter().flat_map(|x| x.to_le_bytes()).collect());
    let h = seeded.as_ref().map_or(h, |w| &w.header);
    let data = sbytes.as_deref().unwrap_or(data);
    let ph = pc_client::spec::params_hash(&dev.ctx.chain);
    if h.params_hash != ph || h.n != dev.ctx.lay.n || h.k != 1 {
        return Err(PyValueError::new_err(format!("import_ct: frame for params {} n {} k {}; this session runs {ph} n {} (k = 1)", h.params_hash, h.n, h.k, dev.ctx.lay.n)));
    }
    let top = dev.ctx.top_of(h.meta.level).map_err(gerr)?;
    if h.rows != top + 1 {
        return Err(PyValueError::new_err(format!("import_ct: {} rows at level {} (top {top})", h.rows, h.meta.level)));
    }
    let tag = if h.meta.tag.is_null() { Default::default() } else { serde_json::from_value(h.meta.tag.clone()).map_err(|e| PyValueError::new_err(format!("import_ct: tag {e}")))? };
    let meta = CtMeta { level: h.meta.level, deg: h.meta.deg, nf: h.meta.nf, slots: h.meta.slots, tag };
    let mut out = if low { PyCt::acquire_low(&dev, meta, top, false)? } else { PyCt::acquire(&dev, meta)? };
    let part_bytes = 4 * h.rows * h.n;
    {
        let st = dev.st.borrow();
        let c = out.ct_mut_ref()?;
        if h.rows > c.buf.lay.rows() {
            return Err(PyValueError::new_err(format!("import_ct: {} rows in a {}-row layout", h.rows, c.buf.lay.rows())));
        }
        // the frame's words go up where they lie, rows 0..rows of a part contiguous on both sides: one copy per part
        nogil(py, || -> pc_gpu::Result<()> {
            for part in 0..2 {
                let off = c.buf.offset(0, part, 0);
                c.buf.buf.upload_bytes_at(&st.ex.compute, off, &data[part * part_bytes..(part + 1) * part_bytes])?;
            }
            Ok(())
        })
        .map_err(gerr)?;
    }
    if seeded.is_some() {
        return Ok(out);
    }
    // SAFETY: `fb` (moved into the pending entry) keeps the exporter's buffer alive and in place until the thread is
    // joined; the caller leaves the frame unchanged until then
    let (p, n) = (data.as_ptr() as usize, data.len());
    let check = move || sum64_le(unsafe { std::slice::from_raw_parts(p as *const u8, n) }) == sum;
    match std::thread::Builder::new().name("import_ct sum".into()).spawn(check) {
        Ok(ok) => PENDING.lock().map_err(|_| PyRuntimeError::new_err("import_ct: checksum threads poisoned"))?.push(Pending { ok, _frame: fb }),
        Err(_) if !check() => return Err(PyValueError::new_err("import_ct: ciphertext checksum mismatch")),
        Err(_) => {}
    }
    Ok(out)
}

/// A ciphertext frame (`CtWire::to_bytes`) read in place: its header, its words' little-endian bytes
/// (`[part][row][N]`) and their checksum, with the magic, version and size checked as `CtWire::from_bytes` checks them
/// (the checksum is the caller's to verify).
fn ct_frame(frame: &[u8]) -> PyResult<(CtHeader, &[u8], u64)> {
    let bad = |m: String| PyValueError::new_err(format!("import_ct: {m}"));
    if frame.len() < 16 || &frame[..4] != CT_MAGIC {
        return Err(bad("not a ciphertext frame".into()));
    }
    let ver = u32::from_le_bytes([frame[4], frame[5], frame[6], frame[7]]);
    if ver != VERSION {
        return Err(bad(format!("version {ver}; this build reads version {VERSION}")));
    }
    let hl = u64::from_le_bytes(frame[8..16].try_into().expect("8 bytes"));
    if hl > (frame.len() - 16) as u64 {
        return Err(bad(format!("a header of {hl} bytes in a frame of {}", frame.len())));
    }
    let d0 = 16 + hl as usize;
    let header: CtHeader = serde_json::from_slice(&frame[16..d0]).map_err(|e| bad(format!("header: {e}")))?;
    if header.n == 0 || header.k == 0 || header.rows == 0 || header.n > 1 << 17 || header.k > 64 || header.rows > 128 {
        return Err(bad(format!("ciphertext header out of range: {header:?}")));
    }
    let d1 = d0 + 4 * CtWire::words(&header);
    if frame.len() != d1 + 8 {
        return Err(bad(format!("{} bytes for a frame of {} words", frame.len(), CtWire::words(&header))));
    }
    Ok((header, &frame[d0..d1], u64::from_le_bytes(frame[d1..].try_into().expect("8 bytes"))))
}

/// `pc_client::wire::sum64` of words given as their little-endian bytes (whole words).
fn sum64_le(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut it = b.chunks_exact(8);
    for p in &mut it {
        h ^= u64::from_le_bytes(p.try_into().expect("8 bytes"));
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    if let [x0, x1, x2, x3] = *it.remainder() {
        h ^= u32::from_le_bytes([x0, x1, x2, x3]) as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyClientSecret>()?;
    m.add_class::<PyClient>()?;
    m.add_function(wrap_pyfunction!(bundle_info, m)?)?;
    m.add_function(wrap_pyfunction!(contract_hash, m)?)?;
    m.add_function(wrap_pyfunction!(export_ct, m)?)?;
    m.add_function(wrap_pyfunction!(import_ct, m)?)?;
    m.add_function(wrap_pyfunction!(export_ct_many, m)?)?;
    Ok(())
}
