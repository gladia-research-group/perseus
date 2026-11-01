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
    payload: Payload,
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

fn meta_dict<'py>(py: Python<'py>, m: &WireMeta) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("level", m.level)?;
    d.set_item("deg", m.deg)?;
    d.set_item("nf", m.nf)?;
    d.set_item("slots", m.slots)?;
    Ok(d)
}

#[pymethods]
impl PyClient {
    #[new]
    #[pyo3(signature = (secret, chain="n32", complex=true, seed=None))]
    fn new(secret: &PyClientSecret, chain: &str, complex: bool, seed: Option<u64>) -> PyResult<Self> {
        let s = ClientSecret::from_bytes(&secret.0).map_err(cerr)?;
        Ok(PyClient { c: Client::new(chain, s).map_err(cerr)?, prng: prng_of(seed)?, payload: if complex { Payload::Complex } else { Payload::Real } })
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
    #[pyo3(signature = (values, level, deg=1, nf=None, slots=None))]
    fn encrypt<'py>(&mut self, py: Python<'py>, values: &Bound<'_, PyAny>, level: u32, deg: u8, nf: Option<f64>, slots: Option<u32>) -> PyResult<Bound<'py, PyBytes>> {
        let v = values_of(py, values)?;
        let mut meta = self.c.fresh_meta(level, deg).map_err(cerr)?;
        if let Some(x) = nf {
            meta.nf = x;
        }
        if let Some(x) = slots {
            meta.slots = x;
        }
        if v.len() > meta.slots as usize {
            return Err(PyValueError::new_err(format!("{} values exceed {} slots", v.len(), meta.slots)));
        }
        let (c, g, p) = (&self.c, &mut self.prng, self.payload);
        let w = nogil(py, || c.encrypt(&[v], &meta, p, g).and_then(|w| w.to_bytes())).map_err(cerr)?;
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
fn export_ct<'py>(py: Python<'py>, session: &PySession, ct: PyRef<'_, PyCt>) -> PyResult<Bound<'py, PyBytes>> {
    checked_imports()?;
    let dev = session.dev.clone();
    let meta = ct.ct_ref().meta;
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
    let st = dev.st.borrow();
    let top = dev.ctx.top_of(c.meta.level).map_err(gerr)?;
    let (rows, n) = (top + 1, dev.ctx.lay.n);
    let m = c.meta;
    let tag = serde_json::to_value(m.tag).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let header = CtHeader { params_hash: pc_client::spec::params_hash(&dev.ctx.chain), n, k: 1, rows, meta: WireMeta { level: m.level, deg: m.deg, nf: m.nf, slots: m.slots, tag } };
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

/// A session ciphertext from a wire frame (one ciphertext, this session's chain; bytes or a memoryview, unchanged until
/// the next `export_ct`). The frame's checksum is verified on a thread while the caller goes on; the next `export_ct`
/// fails if it did not match.
#[pyfunction]
fn import_ct(py: Python<'_>, session: &PySession, frame: &Bound<'_, PyAny>) -> PyResult<PyCt> {
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
    let mut out = PyCt::acquire(&dev, meta)?;
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
    Ok(())
}
