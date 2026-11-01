//! The client: key generation from the secret on the host (every key a `KeySpec` asks for, streamed into a bundle),
//! secret-key encryption of encoded slot vectors, decryption and decoding. Host NTTs and the KSKA expander run on the
//! caller's rayon pool; samplers draw from the caller's `Prng` (per-row / per-digit sub-streams keyed from it).
use crate::error::{Error, Result};
use crate::ntt::Ring;
use crate::secret::ClientSecret;
use crate::spec::{params_hash, KeyReq, KeyShape, KeySpec, Role};
use crate::wire::{self, BundleHeader, BundleWriter, CtHeader, CtWire, Entry, WireMeta};
use pc_core::ccmm_keys::CcmmKeys;
use pc_core::encode::{Complex, Encoder, Payload};
use pc_core::galois;
use pc_core::keys::{kska_expand_row, DiscreteGaussian, KskSeed, Prng};
use pc_core::modmul::{invmod, mulmod};
use pc_core::params::Chain;
use pc_core::pcmm_keys::PcmmKeys;
use pc_core::scale::ScaleTables;
use rayon::prelude::*;
use std::io::Write;
use std::time::Instant;

/// Seconds and sizes of one `Client::keygen`.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct KeyGenStats {
    pub keys: usize,
    pub bytes: u64,
    pub ksk_s: f64,
    pub spru_s: f64,
    pub pcmm_s: f64,
    pub ccmm_s: f64,
    pub pk_s: f64,
    pub total_s: f64,
    /// Bytes per entry kind.
    pub ksk_bytes: u64,
    pub pcmm_bytes: u64,
    pub ccmm_bytes: u64,
    pub other_bytes: u64,
}

pub struct Client {
    pub name: String,
    pub chain: Chain,
    pub ring: Ring,
    pub params_hash: String,
    enc: Encoder,
    scale: ScaleTables,
    secret: ClientSecret,
    /// `s` in eval form at every prime id (Q then P).
    s_eval: Vec<Vec<u32>>,
    gauss: DiscreteGaussian,
    /// The SPRU block key of the last keygen that asked for one (relation checks).
    spru_block: Option<Vec<i8>>,
}

/// mod-p product of two residues below 2^32.
#[inline(always)]
fn mm(a: u32, b: u32, p: u64) -> u64 {
    a as u64 * b as u64 % p
}

impl Client {
    /// A client for chain preset `name` holding `secret` (n32: u32 words, primes below 2^31).
    pub fn new(name: &str, secret: ClientSecret) -> Result<Self> {
        let chain = crate::spec::chain_of(name)?;
        if chain.word_bits != 32 {
            return Err(Error::Params(format!("chain {name}: the client packs 32-bit words only (word {} bits)", chain.word_bits)));
        }
        if secret.sk.s.len() != chain.n || secret.sk.s_sparse.len() != chain.n {
            return Err(Error::Params(format!("secret of {} / {} coefficients for N = {}", secret.sk.s.len(), secret.sk.s_sparse.len(), chain.n)));
        }
        let ring = Ring::new(&chain)?;
        let s_eval = ring.eval_all(&secret.sk.s.iter().map(|&x| x as i64).collect::<Vec<_>>());
        Ok(Client {
            name: name.to_string(),
            params_hash: params_hash(&chain),
            enc: Encoder::new(&chain),
            scale: ScaleTables::precompute(&chain)?,
            ring,
            chain,
            secret,
            s_eval,
            gauss: DiscreteGaussian::new(DiscreteGaussian::SIGMA)?,
            spru_block: None,
        })
    }

    pub fn secret(&self) -> &ClientSecret {
        &self.secret
    }

    /// Top Q prime id of a ciphertext at `level` (primes dropped).
    pub fn top_of(&self, level: u32) -> Result<usize> {
        self.chain.top_at(level).ok_or_else(|| Error::Params(format!("level {level} is not a level of the {}-prime chain", self.chain.q.len())))
    }

    /// The metadata of a fresh encryption at `level`: the level's scaling factor to the power `deg`, every slot.
    pub fn fresh_meta(&self, level: u32, deg: u8) -> Result<WireMeta> {
        let sf = self.scale.sf(self.top_of(level)? as i32)?;
        Ok(WireMeta { level, deg, nf: sf.powi(deg as i32), slots: (self.chain.n / 2) as u32, tag: serde_json::Value::Null })
    }

    // ── key generation ────────────────────────────────────────────────────────────────────────────────────────

    /// `rows[id]` = sigma_g of the eval rows `src[id]` (gather; `src` indexed by prime id) for every id of `ids`.
    fn gathered(src: &[Vec<u32>], g: u64, ids: &[usize], n: usize) -> Result<Vec<Option<Vec<u32>>>> {
        let t = galois::eval_gather_table(g, n)?;
        let mut out: Vec<Option<Vec<u32>>> = vec![None; src.len()];
        let rows: Vec<(usize, Vec<u32>)> = ids.par_iter().map(|&id| (id, t.iter().map(|&i| src[id][i as usize]).collect())).collect();
        for (id, r) in rows {
            out[id] = Some(r);
        }
        Ok(out)
    }

    /// Packed `b` of one switching key (relation in `spec` module doc).
    #[allow(clippy::too_many_arguments)]
    fn ksk_packed(&self, shape: &KeyShape, s_new: &[Option<Vec<u32>>], s_old: &[Vec<u32>], sq: bool, seed: &KskSeed, err_key: [u32; 8], bits: u32, row_words: usize) -> Vec<u32> {
        self.ksk_packed_with(shape, s_new, s_old, sq, seed, err_key, bits, row_words, None)
    }

    /// `ksk_packed` with the digit map and the gadget (P mod q_t by prime id) given (level relinearization keys).
    #[allow(clippy::too_many_arguments)]
    fn ksk_packed_with(&self, shape: &KeyShape, s_new: &[Option<Vec<u32>>], s_old: &[Vec<u32>], sq: bool, seed: &KskSeed, err_key: [u32; 8], bits: u32, row_words: usize, ov: Option<(&[i32], &[u32])>) -> Vec<u32> {
        let n = self.chain.n;
        let ids = shape.row_ids();
        let rows = ids.len();
        let nq = self.ring.nq;
        let errs: Vec<Vec<i64>> = (0..shape.ndig).into_par_iter().map(|d| self.gauss.sample_vec(&mut Prng::from_seed(err_key, d as u32), n)).collect();
        let mut out = vec![0u32; shape.ndig * rows * row_words];
        out.par_chunks_mut(row_words).enumerate().for_each(|(r, o)| {
            let (d, id) = (r / rows, ids[r % rows]);
            let p = self.ring.primes[id];
            let mut b: Vec<u32> = errs[d].iter().map(|&x| x.rem_euclid(p as i64) as u32).collect();
            self.ring.ntt[id].fwd(&mut b);
            let mut a = vec![0u32; n];
            kska_expand_row(seed, d as u32, p as u32, &mut a);
            let sn = s_new[id].as_ref().expect("s_new row of every stored prime id");
            let (inq, pm) = match ov {
                Some((dof, g)) => {
                    let inq = id < nq && dof[id] == d as i32;
                    (inq, if inq { g[id] } else { 0 })
                }
                None => {
                    let inq = id < nq && self.ring.digit_of(id) == Some(d);
                    (inq, if inq { self.ring.gadget(shape.kp)[id] } else { 0 })
                }
            };
            let so = &s_old[id];
            for i in 0..n {
                let mut v = b[i] as u64 + p - mm(a[i], sn[i], p);
                if inq {
                    let x = if sq { mm(so[i], so[i], p) as u32 } else { so[i] };
                    v += mm(pm, x, p);
                }
                b[i] = (v % p) as u32;
            }
            wire::pack_row(&b, bits, o);
        });
        out
    }

    /// A level relinearization key's digit map and gadget (P' mod q_t, Q ids <= top) by prime id.
    fn level_tables(&self, top: usize) -> Result<(Vec<i32>, Vec<u32>)> {
        let chain = &self.chain;
        let (dg, sp, _) = chain.level_relin_layout(top, crate::spec::LEVEL_RELIN_CAP).ok_or_else(|| Error::Spec(format!("no level relinearization key at top {top}")))?;
        let rows = chain.q.len() + chain.p.len();
        let pr = &self.ring.primes;
        let dof: Vec<i32> = (0..rows).map(|t| dg.iter().position(|g| g.contains(&t)).map_or(-1, |d| d as i32)).collect();
        let g: Vec<u32> = (0..rows).map(|t| if t <= top { sp.iter().fold(1u64, |acc, &x| acc * (pr[x] % pr[t]) % pr[t]) as u32 } else { 0 }).collect();
        Ok((dof, g))
    }

    /// Eval rows (every prime id; None outside `ids`) of a signed small polynomial.
    fn rows_of(&self, c: &[i8], ids: &[usize]) -> Vec<Option<Vec<u32>>> {
        let ci: Vec<i64> = c.iter().map(|&x| x as i64).collect();
        let mut out: Vec<Option<Vec<u32>>> = vec![None; self.ring.primes.len()];
        let rows: Vec<(usize, Vec<u32>)> = ids.par_iter().map(|&id| (id, self.ring.eval_row(&ci, id))).collect();
        for (id, r) in rows {
            out[id] = Some(r);
        }
        out
    }

    /// Eval rows of a signed polynomial at every prime id (dense table).
    fn dense_rows(&self, c: &[i8]) -> Vec<Vec<u32>> {
        self.ring.eval_all(&c.iter().map(|&x| x as i64).collect::<Vec<_>>())
    }

    /// Generate every key of `spec` and stream the bundle into `out`. `threads`: the rayon pool size.
    pub fn keygen<Wr: Write + Send>(&mut self, spec: &KeySpec, out: Wr, prng: &mut Prng, threads: usize) -> Result<KeyGenStats> {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads.max(1)).build().map_err(|e| Error::Params(format!("rayon pool of {threads}: {e}")))?;
        pool.install(|| self.keygen_in_pool(spec, out, prng))
    }

    fn keygen_in_pool<Wr: Write>(&mut self, spec: &KeySpec, out: Wr, prng: &mut Prng) -> Result<KeyGenStats> {
        let t0 = Instant::now();
        let chain = &self.chain;
        if spec.chain != self.name || spec.params_hash != self.params_hash {
            return Err(Error::Spec(format!("spec for chain {} ({}), this client runs {} ({})", spec.chain, spec.params_hash, self.name, self.params_hash)));
        }
        spec.check(chain)?;
        let n = chain.n;
        let bits = chain.prime_bits().into_iter().max().unwrap_or(0);
        if bits == 0 || bits > 31 {
            return Err(Error::Params(format!("widest prime has {bits} bits; packing needs 1..=31")));
        }
        let row_words = wire::packed_row_words(n, bits);
        let mut st = KeyGenStats::default();
        // small entries first (generated in memory), then the switching keys streamed one by one
        let mut small: Vec<(Entry, Vec<u32>)> = vec![];
        let tp = Instant::now();
        for &t in &spec.pcmm_t {
            let k = PcmmKeys::generate(&self.secret.sk.s, &chain.q[..4], t, prng)?;
            let words = k.fwd.len() + k.back.len() + k.pack.len();
            let mut v = Vec::with_capacity(words);
            v.extend_from_slice(&k.fwd);
            v.extend_from_slice(&k.back);
            v.extend_from_slice(&k.pack);
            small.push((Entry::Pcmm { t, primes: k.primes, split: k.split, fwd: k.fwd.len(), back: k.back.len(), pack: k.pack.len(), words }, v));
            st.pcmm_bytes += 4 * words as u64;
        }
        st.pcmm_s = tp.elapsed().as_secs_f64();
        let tc = Instant::now();
        if spec.ccmm {
            let (k, _) = CcmmKeys::generate(&self.secret.sk.s, &chain.q[..4], prng)?;
            let words = k.fwd.len() + k.back.len() + k.autos.len() + k.rlk.len();
            let mut v = Vec::with_capacity(words);
            for part in [&k.fwd, &k.back, &k.autos, &k.rlk] {
                v.extend_from_slice(part);
            }
            small.push((Entry::Ccmm { primes: k.primes, fwd: k.fwd.len(), back: k.back.len(), autos: k.autos.len(), rlk: k.rlk.len(), words }, v));
            st.ccmm_bytes += 4 * words as u64;
        }
        st.ccmm_s = tc.elapsed().as_secs_f64();
        let ts = Instant::now();
        self.spru_block = None;
        if let Some(sp) = &spec.spru {
            let block = spru_block_key(prng, n, sp.h);
            let cs = self.spru_cs(&block, sp.h, sp.n, sp.cs_level, sp.cs_nf, prng)?;
            st.other_bytes += 4 * cs.data.len() as u64;
            small.push((Entry::SpruCs { k: cs.header.k, rows: cs.header.rows, level: sp.cs_level, nf: sp.cs_nf, words: cs.data.len() }, cs.data));
            self.spru_block = Some(block);
        }
        st.spru_s = ts.elapsed().as_secs_f64();
        let tk = Instant::now();
        if spec.public_key {
            let pk = self.public_key(prng)?;
            st.other_bytes += 4 * pk.len() as u64;
            small.push((Entry::PublicKey { rows: chain.q.len(), words: pk.len() }, pk));
        }
        st.pk_s = tk.elapsed().as_secs_f64();
        // header: the small entries, then one entry per switching key with its seed drawn now
        let seeds: Vec<(KskSeed, [u32; 8])> = spec.keys.iter().map(|_| (prng.ksk_seed(), prng.ksk_seed().0)).collect();
        let mut entries: Vec<Entry> = small.iter().map(|(e, _)| e.clone()).collect();
        for (k, (seed, _)) in spec.keys.iter().zip(&seeds) {
            entries.push(Entry::Ksk { key: *k, seed: seed.0, words: wire::ksk_words(&k.shape, row_words) });
        }
        let header = BundleHeader { chain: self.name.clone(), params_hash: self.params_hash.clone(), contract: spec.contract.clone(), n, bits, row_words, entries };
        let mut w = BundleWriter::new(out, header)?;
        for (_, v) in &small {
            w.put(v)?;
        }
        drop(small);
        let tks = Instant::now();
        let needs_sparse = spec.keys.iter().any(|k| matches!(k.role, Role::EncDenseToSparse | Role::EncSparseToDense | Role::SparseRot { .. }));
        let s_sparse = if needs_sparse { Some(self.dense_rows(&self.secret.sk.s_sparse)) } else { None };
        let s_all: Vec<Option<Vec<u32>>> = self.s_eval.iter().cloned().map(Some).collect();
        for (k, (seed, ek)) in spec.keys.iter().zip(&seeds) {
            let ids = k.shape.row_ids();
            let packed = match k.role {
                Role::Relin => self.ksk_packed(&k.shape, &s_all, &self.s_eval, true, seed, *ek, bits, row_words),
                Role::RelinLevel { top } => {
                    let (dof, g) = self.level_tables(top)?;
                    self.ksk_packed_with(&k.shape, &s_all, &self.s_eval, true, seed, *ek, bits, row_words, Some((&dof, &g)))
                }
                Role::Rot { .. } | Role::Conj | Role::Auto { .. } => {
                    let g = k.role.galois(n).ok_or_else(|| Error::Spec(format!("{:?} without a Galois element", k.role)))?;
                    let ginv = galois::galois_inverse(g, n)?;
                    let s_new = Self::gathered(&self.s_eval, ginv, &ids, n)?;
                    self.ksk_packed(&k.shape, &s_new, &self.s_eval, false, seed, *ek, bits, row_words)
                }
                Role::EncDenseToSparse => {
                    let sp = s_sparse.as_ref().ok_or_else(|| Error::Spec("sparse secret rows".into()))?;
                    let s_new: Vec<Option<Vec<u32>>> = sp.iter().cloned().map(Some).collect();
                    self.ksk_packed(&k.shape, &s_new, &self.s_eval, false, seed, *ek, bits, row_words)
                }
                Role::EncSparseToDense => {
                    let sp = s_sparse.as_ref().ok_or_else(|| Error::Spec("sparse secret rows".into()))?;
                    self.ksk_packed(&k.shape, &s_all, sp, false, seed, *ek, bits, row_words)
                }
                Role::SparseRot { galois, .. } => {
                    let sp = s_sparse.as_ref().ok_or_else(|| Error::Spec("sparse secret rows".into()))?;
                    let s_new = Self::gathered(&self.s_eval, galois::galois_inverse(galois, n)?, &ids, n)?;
                    self.ksk_packed(&k.shape, &s_new, sp, false, seed, *ek, bits, row_words)
                }
                Role::SpruAtob => {
                    let block = self.spru_block.as_ref().ok_or_else(|| Error::Spec("SpruAtob key without a spru request".into()))?;
                    let s_new = self.rows_of(block, &ids);
                    self.ksk_packed(&k.shape, &s_new, &self.s_eval, false, seed, *ek, bits, row_words)
                }
            };
            w.put(&packed)?;
            st.ksk_bytes += 4 * packed.len() as u64;
            st.keys += 1;
        }
        st.ksk_s = tks.elapsed().as_secs_f64();
        st.bytes = w.finish()?;
        st.total_s = t0.elapsed().as_secs_f64();
        Ok(st)
    }

    /// The SPRU bootstrapping-key ciphertexts Enc(S_u), u < 2n, at `level` (scale `nf`, every slot, real payload).
    fn spru_cs(&self, block: &[i8], h: usize, n_comp: usize, level: u32, nf: f64, prng: &mut Prng) -> Result<CtWire> {
        let n = self.chain.n;
        let meta = WireMeta { level, deg: 1, nf, slots: (n / 2) as u32, tag: serde_json::Value::Null };
        let vals: Vec<Vec<Complex>> = (0..2 * n_comp).map(|u| spru_key_slots(block, h, n_comp, n, u)).collect();
        self.encrypt(&vals, &meta, Payload::Real, prng)
    }

    /// `(pk0, pk1) = (-a s + e, a)`, eval form over every Q id: `[part][row][N]`.
    fn public_key(&self, prng: &mut Prng) -> Result<Vec<u32>> {
        let (n, nq) = (self.chain.n, self.chain.q.len());
        let e = self.gauss.sample_vec(prng, n);
        let keys: Vec<[u32; 8]> = (0..nq).map(|_| prng.ksk_seed().0).collect();
        let mut out = vec![0u32; 2 * nq * n];
        let (p0, p1) = out.split_at_mut(nq * n);
        p0.par_chunks_mut(n).zip(p1.par_chunks_mut(n)).enumerate().for_each(|(r, (c0, c1))| {
            let p = self.ring.primes[r];
            let mut g = Prng::from_seed(keys[r], 0);
            c1.iter_mut().for_each(|x| *x = g.uniform_below(p) as u32);
            c0.iter_mut().zip(&e).for_each(|(x, &v)| *x = v.rem_euclid(p as i64) as u32);
            self.ring.ntt[r].fwd(c0);
            let s = &self.s_eval[r];
            for i in 0..n {
                c0[i] = ((c0[i] as u64 + p - mm(c1[i], s[i], p)) % p) as u32;
            }
        });
        Ok(out)
    }

    // ── encryption / decryption ───────────────────────────────────────────────────────────────────────────────

    /// Secret-key encryption of `values.len()` slot vectors (each zero-padded to `meta.slots`) at `meta`:
    /// `c0 = -a s + e + m`, `c1 = a` (a uniform), eval form over Q ids `0..=top`.
    pub fn encrypt(&self, values: &[Vec<Complex>], meta: &WireMeta, payload: Payload, prng: &mut Prng) -> Result<CtWire> {
        let n = self.chain.n;
        let top = self.top_of(meta.level)?;
        let rows = top + 1;
        let slots = if meta.slots == 0 { n / 2 } else { meta.slots as usize };
        if values.is_empty() {
            return Err(Error::Params("encrypt: no values".into()));
        }
        let k = values.len();
        let mut data = vec![0u32; k * 2 * rows * n];
        for (c, v) in values.iter().enumerate() {
            let m = self.enc.encode_slots(v, slots, top, payload, meta.nf)?;
            let e = self.gauss.sample_vec(prng, n);
            let keys: Vec<[u32; 8]> = (0..rows).map(|_| prng.ksk_seed().0).collect();
            let ct = &mut data[c * 2 * rows * n..(c + 1) * 2 * rows * n];
            let (p0, p1) = ct.split_at_mut(rows * n);
            p0.par_chunks_mut(n).zip(p1.par_chunks_mut(n)).enumerate().for_each(|(r, (c0, c1))| {
                let p = self.ring.primes[r];
                let mut g = Prng::from_seed(keys[r], 0);
                c1.iter_mut().for_each(|x| *x = g.uniform_below(p) as u32);
                for i in 0..n {
                    c0[i] = ((m[r][i] % p + e[i].rem_euclid(p as i64) as u64) % p) as u32;
                }
                self.ring.ntt[r].fwd(c0);
                let s = &self.s_eval[r];
                for i in 0..n {
                    c0[i] = ((c0[i] as u64 + p - mm(c1[i], s[i], p)) % p) as u32;
                }
            });
        }
        Ok(CtWire { header: CtHeader { params_hash: self.params_hash.clone(), n, k, rows, meta: meta.clone() }, data })
    }

    /// Decoded slots of every ciphertext of `ct` (`meta.slots` slots, scale `meta.nf`): `c0 + c1 s` over the bottom
    /// 4 rows (2 when the ciphertext has fewer), exact centered CRT, FFT decode.
    pub fn decrypt(&self, ct: &CtWire, payload: Payload) -> Result<Vec<Vec<Complex>>> {
        let h = &ct.header;
        if h.params_hash != self.params_hash || h.n != self.chain.n {
            return Err(Error::Wire(format!("ciphertext for params {} (n {}), this client runs {} (n {})", h.params_hash, h.n, self.params_hash, self.chain.n)));
        }
        if h.meta.deg != 1 {
            return Err(Error::Wire(format!("ciphertext of scale degree {}: the server realizes (rescales) before export", h.meta.deg)));
        }
        if !(h.meta.nf > 0.0) {
            return Err(Error::Wire(format!("ciphertext scale (meta.nf) = {}", h.meta.nf)));
        }
        if h.rows != self.top_of(h.meta.level)? + 1 {
            return Err(Error::Wire(format!("ciphertext of {} rows at level {} (top {})", h.rows, h.meta.level, self.top_of(h.meta.level)?)));
        }
        let unit = self.chain.level_unit;
        let d = if h.rows >= 4 { 4 } else { unit };
        if h.rows < d {
            return Err(Error::Wire(format!("ciphertext of {} rows is below the bottom level ({d})", h.rows)));
        }
        let n = h.n;
        let slots = if h.meta.slots == 0 { n / 2 } else { h.meta.slots as usize };
        (0..h.k)
            .map(|c| {
                let rows: Vec<Vec<u64>> = (0..d)
                    .into_par_iter()
                    .map(|r| {
                        let p = self.ring.primes[r];
                        let (c0, c1, s) = (ct.row(c, 0, r), ct.row(c, 1, r), &self.s_eval[r]);
                        let mut m: Vec<u32> = (0..n).map(|i| ((c0[i] as u64 + mm(c1[i], s[i], p)) % p) as u32).collect();
                        self.ring.ntt[r].inv(&mut m);
                        m.into_iter().map(|x| x as u64).collect()
                    })
                    .collect();
                if d == 4 {
                    Ok(decode4(&self.enc, &self.chain.q[..4], &rows, slots, payload, h.meta.nf, n))
                } else {
                    Ok(self.enc.decode_slots(&rows, slots, payload, h.meta.nf)?)
                }
            })
            .collect()
    }

    // ── checks ────────────────────────────────────────────────────────────────────────────────────────────────

    /// Max |e| of the switch relation of one bundle key (`b` packed as in the bundle, `a` regenerated from the seed).
    pub fn ksk_relation_max_err(&self, key: &KeyReq, seed: &KskSeed, packed: &[u32], bits: u32, row_words: usize) -> Result<i64> {
        let n = self.chain.n;
        let ids = key.shape.row_ids();
        let rows = ids.len();
        let s_all: Vec<Option<Vec<u32>>> = self.s_eval.iter().cloned().map(Some).collect();
        let (s_new, s_old, sq): (Vec<Option<Vec<u32>>>, Vec<Vec<u32>>, bool) = match key.role {
            Role::Relin | Role::RelinLevel { .. } => (s_all, self.s_eval.clone(), true),
            Role::Rot { .. } | Role::Conj | Role::Auto { .. } => {
                let g = key.role.galois(n).ok_or_else(|| Error::Spec("automorphism key without a Galois element".into()))?;
                (Self::gathered(&self.s_eval, galois::galois_inverse(g, n)?, &ids, n)?, self.s_eval.clone(), false)
            }
            Role::EncDenseToSparse => (self.dense_rows(&self.secret.sk.s_sparse).into_iter().map(Some).collect(), self.s_eval.clone(), false),
            Role::EncSparseToDense => (s_all, self.dense_rows(&self.secret.sk.s_sparse), false),
            Role::SparseRot { galois, .. } => (Self::gathered(&self.s_eval, galois::galois_inverse(galois, n)?, &ids, n)?, self.dense_rows(&self.secret.sk.s_sparse), false),
            Role::SpruAtob => {
                let b = self.spru_block.as_ref().ok_or_else(|| Error::Spec("no SPRU block key in this client".into()))?;
                (self.rows_of(b, &ids), self.s_eval.clone(), false)
            }
        };
        let nq = self.ring.nq;
        let lvl = match key.role {
            Role::RelinLevel { top } => Some(self.level_tables(top)?),
            _ => None,
        };
        let mx = (0..key.shape.ndig * rows)
            .into_par_iter()
            .map(|r| {
                let (d, id) = (r / rows, ids[r % rows]);
                let p = self.ring.primes[id];
                let mut b = vec![0u32; n];
                wire::unpack_row(&packed[r * row_words..(r + 1) * row_words], bits, &mut b);
                let mut a = vec![0u32; n];
                kska_expand_row(seed, d as u32, p as u32, &mut a);
                let sn = s_new[id].as_ref().expect("row");
                let (inq, gm) = match &lvl {
                    Some((dof, g)) => (id < nq && dof[id] == d as i32, g[id]),
                    None => (id < nq && self.ring.digit_of(id) == Some(d), if id < nq { self.ring.gadget(key.shape.kp)[id] } else { 0 }),
                };
                let mut e: Vec<u32> = (0..n)
                    .map(|i| {
                        let mut v = b[i] as u64 + mm(a[i], sn[i], p);
                        if inq {
                            let so = s_old[id][i];
                            let x = if sq { mm(so, so, p) as u32 } else { so };
                            v += p - mm(gm, x, p);
                        }
                        (v % p) as u32
                    })
                    .collect();
                self.ring.ntt[id].inv(&mut e);
                e.iter().map(|&x| if x as u64 > p / 2 { p - x as u64 } else { x as u64 } as i64).max().unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        Ok(mx)
    }
}

/// The SPRU block-binary key: block b holds its single 1 at b B + pos[b], pos[0] = 0 (s'_0 = 1), B = N / h.
pub fn spru_block_key(prng: &mut Prng, n: usize, h: usize) -> Vec<i8> {
    let b = n / h;
    let mut s = vec![0i8; n];
    for i in 0..h {
        let p = if i == 0 { 0 } else { prng.uniform_below(b as u64) as usize };
        s[i * b + p] = 1;
    }
    s
}

/// The slot vector S_u of SPRU bootstrapping-key ciphertext u: slot k h n + b n + a = s'_(b B + u B/(2n) + k).
pub fn spru_key_slots(block: &[i8], h: usize, n_comp: usize, ring_n: usize, u: usize) -> Vec<Complex> {
    let b_sz = ring_n / h;
    let sub = b_sz / (2 * n_comp);
    let mut v = vec![Complex::default(); ring_n / 2];
    for k in 0..sub {
        for b in 0..h {
            for a in 0..n_comp {
                v[k * h * n_comp + b * n_comp + a] = Complex::new(block[b * b_sz + u * sub + k] as f64, 0.0);
            }
        }
    }
    v
}

/// Slots from 4 coefficient rows: exact centered CRT over q0..q3 (u128), then the FFT decode (a value above
/// q0 q1 / 2 at its scale needs the 4 rows).
fn decode4(enc: &Encoder, q: &[u64], rows: &[Vec<u64>], slots: usize, payload: Payload, scale: f64, n: usize) -> Vec<Complex> {
    let big: u128 = q.iter().fold(1u128, |a, &x| a * x as u128);
    let hats: Vec<(u128, u64)> = q
        .iter()
        .map(|&qi| {
            let hat = big / qi as u128;
            (hat, invmod((hat % qi as u128) as u64, qi))
        })
        .collect();
    let coef = |j: usize| -> f64 {
        let mut x: u128 = 0;
        for (i, &(hat, hinv)) in hats.iter().enumerate() {
            let t = mulmod(rows[i][j] % q[i], hinv, q[i]) as u128;
            x = (x + t * hat) % big;
        }
        if x > big / 2 {
            -((big - x) as f64)
        } else {
            x as f64
        }
    };
    let gap = n / (2 * slots);
    let mut v: Vec<Complex> = (0..slots).map(|i| Complex::new(coef(gap * i) / scale, coef(gap * i + n / 2) / scale)).collect();
    enc.fft_special(&mut v);
    if payload == Payload::Real {
        v.iter_mut().for_each(|x| x.im = 0.0);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::KeyShape;

    fn client(seed: u32) -> Client {
        client_on("n32", seed)
    }

    fn client_on(chain: &str, seed: u32) -> Client {
        let mut g = Prng::from_seed([seed, 1, 2, 3, 4, 5, 6, 7], 0);
        let sec = ClientSecret::generate(&mut g, 1 << 16).unwrap();
        Client::new(chain, sec).unwrap()
    }

    /// sigma_{g^-1}(s) gathered in eval form == the coefficient-form automorphism transformed.
    #[test]
    fn gather_is_the_automorphism() {
        let c = client(1);
        let n = c.chain.n;
        let g = galois::rotation_galois(5, n).unwrap();
        let gi = galois::galois_inverse(g, n).unwrap();
        let s: Vec<i64> = c.secret.sk.s.iter().map(|&x| x as i64).collect();
        let coeff = galois::apply_coeff_signed(&s, gi).unwrap();
        let want = c.ring.eval_row(&coeff, 7);
        let got = Client::gathered(&c.s_eval, gi, &[7], n).unwrap();
        assert_eq!(got[7].as_ref().unwrap(), &want);
    }

    /// Every role's key satisfies its switch relation (max |e| within 14 sigma), and the bundle round-trips: on n32 and
    /// on the lean n32a4, whose bottom-shape keys (the sparse secrets') switch over P' only.
    #[test]
    fn keygen_relations_and_bundle() {
        for name in ["n32", "n32a4"] {
            keygen_relations_and_bundle_on(name);
        }
    }

    /// Level relinearization keys (n32a4: tops 26, 40 = 2, 3 digits over P' = P x one idle Q prime): the relation
    /// holds with the level's digits and gadget (small |e|), and a wrong shape is refused.
    #[test]
    fn level_relin_keys() {
        let name = "n32a4";
        let mut c = client_on(name, 3);
        let chain = c.chain.clone();
        let keys: Vec<KeyReq> = [26usize, 40]
            .iter()
            .map(|&top| {
                let (dg, _, q_top) = chain.level_relin_layout(top, crate::spec::LEVEL_RELIN_CAP).unwrap();
                KeyReq { role: Role::RelinLevel { top }, shape: KeyShape { q_top, ndig: dg.len(), nq: chain.q.len(), kp: chain.p.len() } }
            })
            .collect();
        let spec = KeySpec { chain: name.into(), params_hash: c.params_hash.clone(), contract: "test".into(), keys, spru: None, pcmm_t: vec![], ccmm: false, public_key: false };
        spec.check(&chain).unwrap();
        let mut bad = spec.clone();
        bad.keys[0].shape = KeyShape::full(&chain);
        assert!(bad.check(&chain).is_err());
        let dir = std::env::temp_dir().join(format!("pc_client_level_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("b.pckb");
        let f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
        let st = c.keygen(&spec, f, &mut Prng::from_seed([7; 8], 0), 8).unwrap();
        assert_eq!(st.keys, 2);
        let mut r = wire::BundleReader::open(&path).unwrap();
        let (bits, rw) = (r.header.bits, r.header.row_words);
        for i in 0..r.header.entries.len() {
            if let Entry::Ksk { key, seed, .. } = r.header.entries[i].clone() {
                let v = r.read(i).unwrap();
                let mx = c.ksk_relation_max_err(&key, &KskSeed(seed), &v, bits, rw).unwrap();
                assert!(mx > 0 && mx <= 45, "{:?}: max |e| {mx}", key.role);
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    fn keygen_relations_and_bundle_on(name: &str) {
        let mut c = client_on(name, 2);
        let chain = c.chain.clone();
        let full = KeyShape::full(&chain);
        let keys = vec![
            KeyReq { role: Role::Relin, shape: full },
            KeyReq { role: Role::rot(1, chain.n).unwrap(), shape: full },
            KeyReq { role: Role::rot(-3, chain.n).unwrap(), shape: KeyShape::banded(&chain, 21).unwrap() },
            KeyReq { role: Role::Conj, shape: full },
            KeyReq { role: Role::EncDenseToSparse, shape: KeyShape::bottom(&chain).unwrap() },
            KeyReq { role: Role::EncSparseToDense, shape: full },
            KeyReq { role: Role::SpruAtob, shape: KeyShape::bottom(&chain).unwrap() },
            KeyReq { role: Role::Auto { galois: 2 * chain.n as u64 - 5 }, shape: full },
            KeyReq { role: Role::SparseRot { step: 2048, galois: galois::rotation_galois(2048, chain.n).unwrap() }, shape: full },
        ];
        let mut keys = keys;
        if chain.p_sparse < chain.p.len() {
            // a lean chain's StC-first input stages: a rotation key over P' at the single primes
            keys.push(KeyReq { role: Role::rot(1, chain.n).unwrap(), shape: KeyShape::stc_ps(&chain, chain.level_unit + 2).unwrap() });
            // and over a shorter prefix of P (6 special primes still exceed the digit by 16 bits; 5 do not)
            keys.push(KeyReq { role: Role::rot(2, chain.n).unwrap(), shape: KeyShape::stc_ps_kp(&chain, chain.level_unit + 2, 6).unwrap() });
            assert!(KeyShape::stc_ps_kp(&chain, chain.level_unit + 2, 5).is_err());
        }
        let nkeys = keys.len();
        let spec = KeySpec {
            chain: name.into(),
            params_hash: c.params_hash.clone(),
            contract: "test".into(),
            keys,
            spru: Some(crate::spec::SpruReq { h: 64, n: 2, cs_level: 0, cs_nf: c.fresh_meta(0, 1).unwrap().nf }),
            pcmm_t: vec![],
            ccmm: false,
            public_key: true,
        };
        if chain.p_sparse < chain.p.len() {
            // a key to a sparse secret over the whole of P is refused
            let mut wide = spec.clone();
            wide.keys[4].shape = KeyShape::banded(&chain, 1).unwrap();
            assert!(wide.check(&chain).is_err());
            // P' above the bottom only for rotation keys
            let mut ps_relin = spec.clone();
            ps_relin.keys[0].shape = KeyShape::stc_ps(&chain, chain.level_unit + 2).unwrap();
            assert!(ps_relin.check(&chain).is_err());
        }
        let dir = std::env::temp_dir().join(format!("pc_client_test_{name}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("b.pckb");
        let f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
        let mut g = Prng::from_seed([9; 8], 0);
        let st = c.keygen(&spec, f, &mut g, 8).unwrap();
        assert_eq!(st.keys, nkeys);
        let mut r = wire::BundleReader::open(&path).unwrap();
        assert_eq!(r.bytes(), st.bytes);
        let (bits, rw) = (r.header.bits, r.header.row_words);
        assert_eq!(bits, 28);
        for i in 0..r.header.entries.len() {
            let e = r.header.entries[i].clone();
            let v = r.read(i).unwrap();
            match e {
                Entry::Ksk { key, seed, .. } => {
                    let mx = c.ksk_relation_max_err(&key, &KskSeed(seed), &v, bits, rw).unwrap();
                    assert!(mx > 0 && mx <= 45, "{:?}: max |e| {mx}", key.role);
                }
                Entry::SpruCs { k, rows, level, nf, .. } => {
                    let w = CtWire { header: CtHeader { params_hash: c.params_hash.clone(), n: chain.n, k, rows, meta: WireMeta { level, deg: 1, nf, slots: (chain.n / 2) as u32, tag: serde_json::Value::Null } }, data: v };
                    let d = c.decrypt(&w, Payload::Real).unwrap();
                    let block = c.spru_block.as_ref().unwrap();
                    for (u, du) in d.iter().enumerate() {
                        let want = spru_key_slots(block, 64, 2, chain.n, u);
                        let err = du.iter().zip(&want).map(|(a, b)| (a.re - b.re).abs()).fold(0.0, f64::max);
                        assert!(err < 1e-9, "cs {u}: {err}");
                    }
                }
                Entry::PublicKey { rows, .. } => {
                    // pk0 + pk1 s = e: small after the inverse transform
                    let n = chain.n;
                    for r in [0usize, rows - 1] {
                        let p = c.ring.primes[r];
                        let mut x: Vec<u32> = (0..n).map(|i| ((v[r * n + i] as u64 + mm(v[(rows + r) * n + i], c.s_eval[r][i], p)) % p) as u32).collect();
                        c.ring.ntt[r].inv(&mut x);
                        let mx = x.iter().map(|&y| if y as u64 > p / 2 { p - y as u64 } else { y as u64 }).max().unwrap();
                        assert!(mx <= 45, "pk row {r}: {mx}");
                    }
                }
                _ => {}
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// encrypt -> decrypt, 4-row and 2-row decodes, complex payload.
    #[test]
    fn encrypt_decrypt() {
        let c = client(3);
        let mut g = Prng::from_seed([4; 8], 0);
        for level in [0u32, 20, 50] {
            let meta = c.fresh_meta(level, 1).unwrap();
            let v: Vec<Complex> = (0..c.chain.n / 2).map(|i| Complex::new(((i * 37) % 101) as f64 / 50.0 - 1.0, ((i * 11) % 7) as f64 / 7.0)).collect();
            let ct = c.encrypt(&[v.clone()], &meta, Payload::Complex, &mut g).unwrap();
            let back = CtWire::from_bytes(&ct.to_bytes().unwrap()).unwrap();
            let d = c.decrypt(&back, Payload::Complex).unwrap();
            let err = d[0].iter().zip(&v).map(|(a, b)| (a.re - b.re).abs().max((a.im - b.im).abs())).fold(0.0, f64::max);
            assert!(err < 2f64.powi(-36), "level {level}: {err:e}");
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;

    /// Phase times of one full key (cargo test --release -- --ignored --nocapture keygen_phases).
    #[test]
    #[ignore]
    fn keygen_phases() {
        let mut g = Prng::from_seed([1; 8], 0);
        let c = Client::new("n32", ClientSecret::generate(&mut g, 1 << 16).unwrap()).unwrap();
        let n = c.chain.n;
        let p = c.ring.primes[5];
        let t = Instant::now();
        let mut a = vec![0u32; n];
        for _ in 0..100 {
            kska_expand_row(&KskSeed([3; 8]), 0, p as u32, &mut a);
        }
        println!("kska row {:.3} ms", t.elapsed().as_secs_f64() * 10.0);
        let t = Instant::now();
        for _ in 0..100 {
            c.ring.ntt[5].fwd(&mut a);
        }
        println!("ntt row {:.3} ms", t.elapsed().as_secs_f64() * 10.0);
        let t = Instant::now();
        let e = c.gauss.sample_vec(&mut g, n);
        println!("gauss row {:.3} ms", t.elapsed().as_secs_f64() * 1e3);
        let s = &c.s_eval[5];
        let t = Instant::now();
        let mut acc = 0u64;
        for _ in 0..100 {
            for i in 0..n {
                acc = acc.wrapping_add(mm(a[i], s[i], p) + e[i].rem_euclid(p as i64) as u64);
            }
        }
        println!("mulmod row {:.3} ms ({acc})", t.elapsed().as_secs_f64() * 10.0);
        let full = KeyShape::full(&c.chain);
        let s_all: Vec<Option<Vec<u32>>> = c.s_eval.iter().cloned().map(Some).collect();
        let t = Instant::now();
        let rw = wire::packed_row_words(n, 28);
        let v = c.ksk_packed(&full, &s_all, &c.s_eval, false, &KskSeed([4; 8]), [5; 8], 28, rw);
        println!("full key {:.3} s ({} words)", t.elapsed().as_secs_f64(), v.len());
    }
}
