//! `perseus_client.ClientSecret`, `perseus_client.Client`, `bundle_info`, `contract_hash`: the client half of
//! `perseus._pc` (pc-py/src/client.rs up to `contract_hash`, verbatim but for this header and `register`), with no
//! device: keygen into a bundle, encryption, decryption. The session's ends of the wire (`export_ct`, `import_ct`,
//! `expand_frame`) stay in pc-py.
use crate::config::PyConfig;
use crate::{cerr, nogil};
use pc_client::{Client, ClientSecret, CtWire, WireMeta};
use pc_core::encode::{Complex, Payload};
use pc_core::keys::Prng;
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
#[pyclass(name = "ClientSecret", module = "perseus_client", frozen)]
pub struct PyClientSecret(pub Vec<u8>);

#[pymethods]
impl PyClientSecret {
    /// A fresh secret for chain `chain` (OS entropy, or `seed` for reproducible tests); `sparse_h`: the Hamming
    /// weight of the sparse (export / encapsulation) secret, 32 by default. `n`: a subring upload client's secret
    /// (`Client(..., sub=True)`) of that many coefficients instead of the chain's. `pack`: with a pack secret
    /// (`with_pack`).
    #[staticmethod]
    #[pyo3(signature = (chain="n32", seed=None, sparse_h=None, n=None, pack=false))]
    fn generate(chain: &str, seed: Option<u64>, sparse_h: Option<usize>, n: Option<usize>, pack: bool) -> PyResult<Self> {
        let c = pc_client::spec::chain_of(chain).map_err(cerr)?;
        let mut g = prng_of(seed)?;
        let h = sparse_h.unwrap_or(pc_core::keys::SecretKey::SPARSE_H);
        let s = ClientSecret::generate_h(&mut g, n.unwrap_or(c.n), h).map_err(cerr)?;
        Ok(PyClientSecret(if pack { s.with_pack(&mut g) } else { s }.to_bytes()))
    }
    /// This secret with a fresh pack secret (dense ternary, independent of the others): what packed frames
    /// (`Client.pack_keygen`, `export_ct_many(..., pack=)`) decrypt under.
    #[pyo3(signature = (seed=None))]
    fn with_pack(&self, seed: Option<u64>) -> PyResult<Self> {
        let mut g = prng_of(seed)?;
        Ok(PyClientSecret(ClientSecret::from_bytes(&self.0).map_err(cerr)?.with_pack(&mut g).to_bytes()))
    }
    #[getter]
    fn has_pack(&self) -> PyResult<bool> {
        Ok(ClientSecret::from_bytes(&self.0).map_err(cerr)?.pack.is_some())
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
/// `complex`: the payload (complex slots, or real: imaginary parts dropped). `sub`: an upload client on the subring
/// of the secret's size n | N (`ClientSecret.generate(n=)`): its frames (`encrypt(.., seeded=True, drop=..)`,
/// `encrypt_coeffs(.., drop=..)` only) are n / N the size; `import_ct` lifts one to a ciphertext under that secret
/// spread over the ring (slots: the frame's n / 2 repeated; coefficient i at i N / n), which a rotation by an entry
/// step (`keygen(entry_steps=, entry_secret=)`) brings, rotated, to the main secret.
#[pyclass(name = "Client", module = "perseus_client", unsendable)]
pub struct PyClient {
    c: Client,
    prng: Prng,
    /// a master public key (`load_public_key`): `encrypt(..., public=True)`
    pk: Option<(Vec<u32>, usize)>,
    payload: Payload,
    /// precomputed encryption parts by (level, regen, coeff) (`precompute`), used first by `encrypt(..., seeded=True)`
    pre: std::collections::HashMap<(u32, bool, bool), Vec<pc_client::PreEnc>>,
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
    /// A precomputed part at `level` in the asked form when one is ready (`precompute`), else made now.
    fn pre_of(&mut self, level: u32, regen: bool, coeff: bool) -> PyResult<pc_client::PreEnc> {
        match self.pre.get_mut(&(level, regen, coeff)).and_then(|v| v.pop()) {
            Some(p) => Ok(p),
            None => self.c.precompute_as(level, &mut self.prng, regen, coeff).map_err(cerr),
        }
    }
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
    #[pyo3(signature = (secret, chain="n32", complex=true, seed=None, sub=false))]
    fn new(secret: &PyClientSecret, chain: &str, complex: bool, seed: Option<u64>, sub: bool) -> PyResult<Self> {
        let s = ClientSecret::from_bytes(&secret.0).map_err(cerr)?;
        let c = if sub { Client::sub(chain, s.sk.s.len(), s) } else { Client::new(chain, s) };
        Ok(PyClient { c: c.map_err(cerr)?, prng: prng_of(seed)?, payload: if complex { Payload::Complex } else { Payload::Real }, pk: None, pre: Default::default() })
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
    /// rayon pool of `threads`. Returns the timings and sizes. `entry_steps`, `entry_secret`: the rotation keys of
    /// those steps made entry keys from a subring upload client's secret (`Client(..., sub=True)`): `rotate(x, step)`
    /// of an imported subring frame is then the rotated plaintext under this client's secret, and those steps rotate
    /// nothing else.
    #[pyo3(signature = (spec, path, threads=8, entry_steps=Vec::new(), entry_secret=None))]
    fn keygen<'py>(&mut self, py: Python<'py>, spec: &str, path: std::path::PathBuf, threads: usize, entry_steps: Vec<i64>, entry_secret: Option<&PyClientSecret>) -> PyResult<Bound<'py, PyDict>> {
        let spec = pc_client::KeySpec::from_json(spec).map_err(cerr)?;
        if entry_steps.is_empty() != entry_secret.is_none() {
            return Err(PyValueError::new_err("keygen: entry_steps come with the entry_secret"));
        }
        let u = match entry_secret {
            Some(u) => ClientSecret::from_bytes(&u.0).map_err(cerr)?.sk.s,
            None => Vec::new(),
        };
        let (c, g) = (&mut self.c, &mut self.prng);
        let st = nogil(py, || -> pc_client::Result<pc_client::KeyGenStats> {
            let f = std::io::BufWriter::with_capacity(1 << 24, std::fs::File::create(&path)?);
            c.keygen_entry(&spec, f, g, threads, &entry_steps, &u)
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
    /// c1 from the random-access expander, so the server can keep the seeds (`import_ct(..., regen=True)`). `drop`
    /// (with `seeded` alone): the short-c0 frame, c0 in coefficient form without its low `drop` bits (noise below
    /// 2^(drop - 1) added; `import_ct` transforms it).
    #[pyo3(signature = (values, level, deg=1, nf=None, slots=None, seeded=false, public=false, regen=false, drop=None))]
    #[allow(clippy::too_many_arguments)]
    fn encrypt<'py>(&mut self, py: Python<'py>, values: &Bound<'_, PyAny>, level: u32, deg: u8, nf: Option<f64>, slots: Option<u32>, seeded: bool, public: bool, regen: bool, drop: Option<u32>) -> PyResult<Bound<'py, PyBytes>> {
        let v = values_of(py, values)?;
        let mut meta = self.meta_at(level, deg, nf)?;
        if let Some(x) = slots {
            meta.slots = x;
        }
        if v.len() > meta.slots as usize {
            return Err(PyValueError::new_err(format!("{} values exceed {} slots", v.len(), meta.slots)));
        }
        if drop.is_some() {
            if !seeded || public || regen || deg != 1 {
                return Err(PyValueError::new_err("encrypt(drop=..): a seeded frame under this client's secret (no regen), deg 1"));
            }
            let pre = self.pre_of(level, false, true)?;
            let (c, p) = (&self.c, self.payload);
            let w = nogil(py, || c.encrypt_pre(&v, &meta, p, pre, drop).and_then(|w| w.to_bytes())).map_err(cerr)?;
            return Ok(PyBytes::new(py, &w));
        }
        if seeded && !public {
            if let Some(pre) = self.pre.get_mut(&(level, regen, false)).and_then(|v| v.pop()) {
                let (c, p) = (&self.c, self.payload);
                let w = nogil(py, || c.encrypt_pre(&v, &meta, p, pre, None).and_then(|w| w.to_bytes())).map_err(cerr)?;
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
    /// how many are ready at that level. `coeff`: parts in coefficient form, the short-c0 frames' (`encrypt(..,
    /// drop=..)`, which then runs no transform).
    #[pyo3(signature = (level, count, regen=false, coeff=false))]
    fn precompute(&mut self, py: Python<'_>, level: u32, count: usize, regen: bool, coeff: bool) -> PyResult<usize> {
        use rayon::prelude::*;
        let keys: Vec<[u32; 8]> = (0..count).map(|_| self.prng.ksk_seed().0).collect();
        let c = &self.c;
        let made = nogil(py, || keys.par_iter().map(|&k| c.precompute_as(level, &mut pc_core::keys::Prng::from_seed(k, 0), regen, coeff)).collect::<pc_client::Result<Vec<_>>>()).map_err(cerr)?;
        let q = self.pre.entry((level, regen, coeff)).or_default();
        q.extend(made);
        Ok(q.len())
    }

    /// A seeded frame of the plaintext whose coefficients are `coeffs` (n reals, at the level's scale unless `nf`), no
    /// slot FFT (e.g. selection weights given as coefficients); a precomputed part (`precompute`) when one is ready.
    /// `public`: under the loaded public key instead (not seeded), as `encrypt`'s. `drop`: the short-c0 frame, as
    /// `encrypt`'s.
    #[pyo3(signature = (coeffs, level, nf=None, regen=false, public=false, drop=None))]
    #[allow(clippy::too_many_arguments)]
    fn encrypt_coeffs<'py>(&mut self, py: Python<'py>, coeffs: Vec<f64>, level: u32, nf: Option<f64>, regen: bool, public: bool, drop: Option<u32>) -> PyResult<Bound<'py, PyBytes>> {
        let meta = self.meta_at(level, 1, nf)?;
        if drop.is_some() && (public || regen) {
            return Err(PyValueError::new_err("encrypt_coeffs(drop=..): under this client's secret, no regen"));
        }
        if public {
            let (pk, rows) = self.pk.as_ref().ok_or_else(|| PyValueError::new_err("encrypt_coeffs(public=True): load_public_key first"))?;
            let (c, g) = (&self.c, &mut self.prng);
            let w = nogil(py, || c.encrypt_public_coeffs(&coeffs, &meta, g, pk, *rows).and_then(|w| w.to_bytes())).map_err(cerr)?;
            return Ok(PyBytes::new(py, &w));
        }
        let pre = self.pre_of(level, regen, drop.is_some())?;
        let c = &self.c;
        let w = nogil(py, || c.encrypt_pre_coeffs(&coeffs, &meta, pre, drop).and_then(|w| w.to_bytes())).map_err(cerr)?;
        Ok(PyBytes::new(py, &w))
    }

    /// The pack key of stride `sub` (this client's secret -> its pack secret, c1 in the subring Z[X^sub]; the secret
    /// needs `ClientSecret.with_pack`) into the bundle file `path`: `b` only, `a` from a seed. Returns {seconds, bytes}.
    #[pyo3(signature = (path, sub=8, threads=8))]
    fn pack_keygen<'py>(&mut self, py: Python<'py>, path: std::path::PathBuf, sub: usize, threads: usize) -> PyResult<Bound<'py, PyDict>> {
        let (c, g) = (&self.c, &mut self.prng);
        let t0 = std::time::Instant::now();
        let bytes = nogil(py, || -> pc_client::Result<u64> {
            let seed: [u32; 8] = std::array::from_fn(|_| g.next_u32());
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads.max(1)).build().map_err(|e| pc_client::Error::Params(e.to_string()))?;
            let words = pool.install(|| c.pack_keygen(sub, &seed, g))?;
            let e = pc_client::Entry::PackKsk { sub, rows: c.pack_ids().len(), seed, words: words.len() };
            write_bundle(c, &path, e, &words)
        })
        .map_err(cerr)?;
        let d = PyDict::new(py);
        d.set_item("seconds", t0.elapsed().as_secs_f64())?;
        d.set_item("bytes", bytes)?;
        Ok(d)
    }

    /// Max |e| of the relation of the pack key in the bundle file `path` (b_j + a'_j t - P' s).
    fn pack_key_error(&self, py: Python<'_>, path: std::path::PathBuf) -> PyResult<i64> {
        let c = &self.c;
        nogil(py, || -> pc_client::Result<i64> {
            let mut rd = pc_client::BundleReader::open(&path)?;
            let (i, sub, seed) = rd
                .header
                .entries
                .iter()
                .enumerate()
                .find_map(|(i, e)| if let pc_client::Entry::PackKsk { sub, seed, .. } = e { Some((i, *sub, *seed)) } else { None })
                .ok_or_else(|| pc_client::Error::Spec("bundle without a pack key".into()))?;
            let words = rd.read(i)?;
            c.pack_relation_max_err(sub, &seed, &words)
        })
        .map_err(cerr)
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

/// A bundle file of one entry (no plan contract: a pack key).
fn write_bundle(c: &Client, path: &std::path::Path, e: pc_client::Entry, words: &[u32]) -> pc_client::Result<u64> {
    let n = c.chain.n;
    let bits = c.chain.prime_bits().into_iter().max().unwrap_or(0);
    let header = pc_client::BundleHeader { chain: c.name.clone(), params_hash: c.params_hash.clone(), contract: String::new(), n, bits, row_words: pc_client::wire::packed_row_words(n, bits), entries: vec![e] };
    let f = std::io::BufWriter::with_capacity(1 << 22, std::fs::File::create(path)?);
    let mut w = pc_client::wire::BundleWriter::new(f, header)?;
    w.put(words)?;
    w.finish()
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
            pc_client::Entry::PackKsk { .. } => "pack_ksk",
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
pub fn contract_hash(config: &PyConfig) -> PyResult<String> {
    crate::manifest::contract_of(&config.0)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyClientSecret>()?;
    m.add_class::<PyClient>()?;
    m.add_function(wrap_pyfunction!(bundle_info, m)?)?;
    m.add_function(wrap_pyfunction!(contract_hash, m)?)?;
    Ok(())
}
