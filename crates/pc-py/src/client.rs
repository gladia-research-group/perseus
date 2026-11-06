//! The client side for Python (`_pc.ClientSecret`, `_pc.Client`: pc-client, no device) and the session's ends of
//! the ciphertext wire format (`_pc.export_ct`, `_pc.import_ct`; docs/WIRE.md).
//!
//! A client process only ever builds `ClientSecret` / `Client`: no `Session`, no device. A server session built
//! with `Session.from_bundle` holds no secret; ciphertexts cross between the two as wire frames (bytes).
use crate::config::PyConfig;
use crate::fhe::PyCt;
use crate::session::{cerr, gerr, nogil, PySession, State};
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
    /// `public`: under the loaded public key instead (not seeded), as `encrypt`'s.
    #[pyo3(signature = (coeffs, level, nf=None, regen=false, public=false))]
    fn encrypt_coeffs<'py>(&mut self, py: Python<'py>, coeffs: Vec<f64>, level: u32, nf: Option<f64>, regen: bool, public: bool) -> PyResult<Bound<'py, PyBytes>> {
        let meta = self.meta_at(level, 1, nf)?;
        if public {
            let (pk, rows) = self.pk.as_ref().ok_or_else(|| PyValueError::new_err("encrypt_coeffs(public=True): load_public_key first"))?;
            let (c, g) = (&self.c, &mut self.prng);
            let w = nogil(py, || c.encrypt_public_coeffs(&coeffs, &meta, g, pk, *rows).and_then(|w| w.to_bytes())).map_err(cerr)?;
            return Ok(PyBytes::new(py, &w));
        }
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

/// Wire frame of a session ciphertext (a pending rescale is realized on a copy first: the client decodes degree 1).
#[pyfunction]
#[pyo3(signature = (session, ct, bits=None))]
fn export_ct<'py>(py: Python<'py>, session: &PySession, ct: PyRef<'_, PyCt>, bits: Option<u32>) -> PyResult<Bound<'py, PyBytes>> {
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
    let rows: Vec<usize> = (0..=top).collect();
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
            header: CtHeader { params_hash: pc_client::spec::params_hash(&dev.ctx.chain), n, k: 1, rows: rows.len(), meta: WireMeta { level: m.level, deg: m.deg, nf: m.nf, slots: m.slots, tag }, bits: Some(b), seeded: false, regen: false, bits1: None, sparse: false },
            data,
        };
        return Ok(PyBytes::new(py, &w.to_bytes().map_err(cerr)?));
    }
    let st = dev.st.borrow();
    let data = nogil(py, || -> pc_gpu::Result<Vec<u32>> {
        let mut v = Vec::with_capacity(2 * rows.len() * n);
        for part in 0..2 {
            v.extend(c.buf.download_rows(&st.ex.compute, 0, part, &rows)?);
        }
        Ok(v)
    })
    .map_err(gerr)?;
    let m = c.meta;
    let tag = serde_json::to_value(m.tag).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let w = CtWire {
        header: CtHeader { params_hash: pc_client::spec::params_hash(&dev.ctx.chain), n, k: 1, rows: rows.len(), meta: WireMeta { level: m.level, deg: m.deg, nf: m.nf, slots: m.slots, tag }, bits: None, seeded: false, regen: false, bits1: None, sparse: false },
        data,
    };
    Ok(PyBytes::new(py, &w.to_bytes().map_err(cerr)?))
}

/// The coefficient residues `export_ct_many(..., defer=True)` downloaded, framed by `frames()` (the compression, in
/// parallel on the host, the GIL released): a plain host object, so another thread can frame them while the caller
/// goes on with the session.
#[pyclass(name = "PendingFrames", module = "perseus._pc")]
pub struct PyPendingFrames {
    limbs: Vec<(pc_gpu::limbs::Limbs<crate::session::W>, CtMeta)>,
    /// Residues still in flight (`export_ct_many(..., defer=True)`): pinned rows of each ciphertext, read after `done`.
    inflight: Option<Inflight>,
    hash: String,
    n: usize,
    bits: u32,
    bits1: Option<u32>,
    sparse: bool,
}

/// Pinned row buffers of deferred exports, reused across calls (allocating pinned memory per call would cost more
/// than the copies).
struct Pinned(pc_cuda::PinnedBuf<crate::session::W>);
// SAFETY: plain host memory; a buffer is only read after the event covering its copy has completed, and handed out
// to one owner at a time
unsafe impl Send for Pinned {}
static PINNED: std::sync::Mutex<Vec<Pinned>> = std::sync::Mutex::new(Vec::new());

fn pinned(dev: &pc_cuda::Device, len: usize) -> PyResult<Pinned> {
    let mut pool = PINNED.lock().map_err(|_| PyRuntimeError::new_err("pinned pool poisoned"))?;
    match pool.iter().position(|b| b.0.len() == len) {
        Some(i) => Ok(pool.swap_remove(i)),
        None => Ok(Pinned(pc_cuda::PinnedBuf::new(dev, len).map_err(|e| PyRuntimeError::new_err(e.to_string()))?)),
    }
}

struct Inflight {
    done: pc_cuda::Event,
    /// Per ciphertext: (parts 0 and 1, top limb, meta).
    rows: Vec<([Pinned; 2], usize, CtMeta)>,
    moduli: Vec<u64>,
}
// SAFETY: the event is only synchronized (thread-safe in CUDA); the buffers as `Pinned`
unsafe impl Send for Inflight {}
unsafe impl Sync for Inflight {}

impl Drop for Inflight {
    fn drop(&mut self) {
        // the copies must have landed before the buffers are reused (errors: the buffers are leaked instead)
        if self.done.sync().is_err() {
            std::mem::forget(std::mem::take(&mut self.rows));
            return;
        }
        if let Ok(mut pool) = PINNED.lock() {
            for (b, _, _) in self.rows.drain(..) {
                pool.extend(b);
            }
        }
    }
}

impl PyPendingFrames {
    fn frame_all(&self) -> pc_client::Result<Vec<Vec<u8>>> {
        use rayon::prelude::*;
        let landed: Vec<(pc_gpu::limbs::Limbs<crate::session::W>, CtMeta)> = match &self.inflight {
            None => Vec::new(),
            Some(f) => {
                f.done.sync().map_err(|e| pc_client::Error::Wire(e.to_string()))?;
                f.rows
                    .iter()
                    .map(|(b, top, m)| {
                        let c = [0, 1].map(|p| b[p].0.as_slice().chunks(self.n).map(|r| r.to_vec()).collect::<Vec<_>>());
                        let l = pc_gpu::limbs::Limbs { c, moduli: f.moduli[..=*top].to_vec(), level: *top, noise_level: m.deg, noise_factor: m.nf };
                        (l, *m)
                    })
                    .collect()
            }
        };
        self.limbs
            .par_iter()
            .chain(landed.par_iter())
            .map(|(l, m)| -> pc_client::Result<Vec<u8>> {
                let mut data = Vec::with_capacity(2 * self.n);
                for (part, b) in [(0, self.bits), (1, self.bits1.unwrap_or(self.bits))] {
                    let rs: Vec<Vec<u64>> = l.c[part].iter().map(|r| r.iter().map(|&x| x as u64).collect()).collect();
                    let refs: Vec<&[u64]> = rs.iter().map(|r| r.as_slice()).collect();
                    data.extend(pc_client::wire::pack_bits(&pc_client::wire::pack_coeffs(&l.moduli, &refs, b)?, b));
                }
                let tag = serde_json::to_value(m.tag).map_err(|e| pc_client::Error::Wire(e.to_string()))?;
                let header = CtHeader { params_hash: self.hash.clone(), n: self.n, k: 1, rows: l.moduli.len(), meta: WireMeta { level: m.level, deg: m.deg, nf: m.nf, slots: m.slots, tag }, bits: Some(self.bits), seeded: false, regen: false, bits1: self.bits1, sparse: self.sparse };
                CtWire { header, data }.to_bytes()
            })
            .collect()
    }
}

#[pymethods]
impl PyPendingFrames {
    /// The wire frames, in the order of the exported ciphertexts.
    fn frames<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let frames = nogil(py, || self.frame_all()).map_err(cerr)?;
        let out = PyList::empty(py);
        for f in frames {
            out.append(PyBytes::new(py, &f))?;
        }
        Ok(out)
    }
    fn __len__(&self) -> usize {
        self.limbs.len() + self.inflight.as_ref().map_or(0, |f| f.rows.len())
    }
}

/// `export_ct(..., bits=b)` of several ciphertexts: the coefficient exports run back to back on the device, the
/// compression and framing in parallel on the host. `bits1`: c1's bits (default `bits`; below 32 bit-packed).
/// `sparse`: each ciphertext first switched to the client's sparse secret (`KeySource(sparse_out=True)`, bottom level):
/// c1's rounding then meets a 32-coefficient secret, so both parts compress further.
/// `user`: switched to that user's key instead (`Session.add_user`): the frames decrypt with the user's secret only.
/// `defer`: return the downloaded residues as `PendingFrames` instead, framed by its `frames()` (on any thread).
#[pyfunction]
#[pyo3(signature = (session, cts, bits, bits1=None, sparse=false, user=None, defer=false))]
#[allow(clippy::too_many_arguments)]
fn export_ct_many<'py>(py: Python<'py>, session: &PySession, cts: Vec<PyRef<'_, PyCt>>, bits: u32, bits1: Option<u32>, sparse: bool, user: Option<String>, defer: bool) -> PyResult<Bound<'py, PyAny>> {
    let sparse = sparse || user.is_some();
    let dev = session.dev.clone();
    let n = dev.ctx.lay.n;
    let mut limbs = Vec::with_capacity(cts.len());
    let mut rows = Vec::new();
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
        if defer {
            // downloads queued into pinned rows: the host goes on issuing work while they wait their turn
            let words = (c.top(dev.ctx.shape).map_err(gerr)? + 1) * n;
            let mut b = [pinned(&dev.device, words)?, pinned(&dev.device, words)?];
            let [b0, b1] = &mut b;
            // SAFETY: the buffers move into the Inflight below, read only after its event (recorded after every copy)
            let top = unsafe { pc_gpu::limbs::export_limbs_async(&dev.ctx, ex, c, 0, [&mut b0.0, &mut b1.0]) }.map_err(gerr)?;
            rows.push((b, top, c.meta));
            continue;
        }
        let l = nogil(py, || pc_gpu::limbs::export_limbs(&dev.ctx, ex, c, 0)).map_err(gerr)?;
        limbs.push((l, c.meta));
    }
    let inflight = if defer {
        let done = pc_cuda::Event::new(&dev.device).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        done.record(&dev.st.borrow().ex.compute).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Some(Inflight { done, rows, moduli: dev.ctx.chain.q.clone() })
    } else {
        None
    };
    let pending = PyPendingFrames { limbs, inflight, hash: pc_client::spec::params_hash(&dev.ctx.chain), n, bits, bits1, sparse };
    if defer {
        return Ok(Bound::new(py, pending)?.into_any());
    }
    Ok(pending.frames(py)?.into_any())
}

/// A session ciphertext from a wire frame (one ciphertext, this session's chain). `low`: into a low slot (the
/// session's `size={"lowPool", "lowRows"}`), an input of `mult_sum` and `export_ct` only. `regen` (with `low`): a
/// `regen` frame kept as c0 and its seeds in a seeded low slot (`size={"lowSeeded"}`), half a low slot (c0 packed on
/// upload when `size={"lowPacked": 1}`); an input of `mult_sum_many` (first operands) only, which draws c1 from the seeds.
#[pyfunction]
#[pyo3(signature = (session, frame, low=false, regen=false))]
fn import_ct(py: Python<'_>, session: &PySession, frame: &[u8], low: bool, regen: bool) -> PyResult<PyCt> {
    let dev = session.dev.clone();
    let w = CtWire::from_bytes(frame).map_err(cerr)?;
    if regen && !(low && w.header.seeded && w.header.regen) {
        return Err(PyValueError::new_err("import_ct(regen=True): a low import of a seeded regen frame (encrypt(..., seeded=True, regen=True))"));
    }
    let w = if w.header.seeded && !regen { w.expand_seeded(&dev.ctx.chain.q).map_err(cerr)? } else { w };
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
    let mut out = if low { PyCt::acquire_low(&dev, meta, top, regen)? } else { PyCt::acquire(&dev, meta)? };
    let rows: Vec<usize> = (0..h.rows).collect();
    let n = h.n;
    {
        let st = dev.st.borrow();
        let c = out.ct_mut_ref()?;
        nogil(py, || -> pc_gpu::Result<()> {
            if regen {
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
                return c.buf.upload_seeds(&st.ex.compute, &w.data[h.rows * n..h.rows * (n + 8)]);
            }
            for part in 0..2 {
                c.buf.upload_rows(&st.ex.compute, 0, part, &rows, &w.data[part * h.rows * n..(part + 1) * h.rows * n])?;
            }
            Ok(())
        })
        .map_err(gerr)?;
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyClientSecret>()?;
    m.add_class::<PyClient>()?;
    m.add_class::<PyPendingFrames>()?;
    m.add_function(wrap_pyfunction!(bundle_info, m)?)?;
    m.add_function(wrap_pyfunction!(contract_hash, m)?)?;
    m.add_function(wrap_pyfunction!(export_ct, m)?)?;
    m.add_function(wrap_pyfunction!(import_ct, m)?)?;
    m.add_function(wrap_pyfunction!(export_ct_many, m)?)?;
    Ok(())
}
