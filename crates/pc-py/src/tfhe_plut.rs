//! Encrypted lookup tables on the TFHE host reference (pc-tfhe `pack`): the public functional packing key switch
//! LWE (s_lwe, q_in) -> GLWE (s_br, Q) and the blind rotation of an encrypted accumulator, as a self-contained key set
//! (`TfhePrivateLut`): LWEs and GLWEs cross as plain integer lists. Host only (the device path is gated by
//! `pc-check tfhe-pbs --enc-acc`).
use pc_core::keys::{sample_ternary, DiscreteGaussian, Prng};
use pc_tfhe::gadget::Gadget;
use pc_tfhe::lwe::Lwe;
use pc_tfhe::pack::{PackingKey, PACK_GADGET};
use pc_tfhe::params::Params;
use pc_tfhe::pbs::{BootstrapKey, Rns};
use pc_tfhe::rlwe::Rlwe;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Glwe = (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>);

fn ve<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A key set for encrypted tables: the binary LWE key (`n_lwe`, mod `q_in`), the blind rotation's GLWE key and
/// bootstrapping key (Perseus's TFHE parameters), the packing key (gadget base 2^8 x 3 over Q).
#[pyclass]
pub(crate) struct TfhePrivateLut {
    p: Params,
    q_in: u64,
    s_lwe: Vec<i64>,
    s_br: Vec<i64>,
    rns: Rns,
    bsk: BootstrapKey,
    pk: PackingKey,
    dg_lwe: DiscreteGaussian,
    prng: Prng,
}

fn glwe_of(c: &[Rlwe; 2]) -> Glwe {
    (c[0].c0.clone(), c[0].c1.clone(), c[1].c0.clone(), c[1].c1.clone())
}

fn rlwe_of(g: &Glwe) -> [Rlwe; 2] {
    [Rlwe { c0: g.0.clone(), c1: g.1.clone() }, Rlwe { c0: g.2.clone(), c1: g.3.clone() }]
}

#[pymethods]
impl TfhePrivateLut {
    #[new]
    #[pyo3(signature = (seed=0, q_in_bits=28))]
    fn new(seed: u64, q_in_bits: u32) -> PyResult<Self> {
        let p = Params::new(1 << 16).map_err(ve)?;
        let q_in = pc_tfhe::modq::prime_below(q_in_bits, 1 << 17).map_err(ve)?;
        let mut prng = Prng::from_seed([seed as u32, (seed >> 32) as u32, 0x7f1e, 0x9a7e, 0, 0, 0, 0], 11);
        let s_lwe: Vec<i64> = (0..p.n_lwe).map(|_| (prng.next_u32() & 1) as i64).collect();
        let s_br: Vec<i64> = sample_ternary(&mut prng, p.n_br).into_iter().map(|x| x as i64).collect();
        let rns = Rns::new(p.br_primes.0, p.br_primes.1, p.n_br).map_err(ve)?;
        let dg = DiscreteGaussian::new(p.bsk_sigma).map_err(ve)?;
        let bsk = BootstrapKey::generate(&s_lwe, &s_br, Gadget::new(rns.q, p.pbs, false), rns.clone(), p.pbs_group, &dg, &mut prng);
        let pk = PackingKey::generate(&s_lwe, &s_br, PACK_GADGET, rns.clone(), &dg, &mut prng);
        let dg_lwe = DiscreteGaussian::new(p.lwe_sigma_rel * q_in as f64).map_err(ve)?;
        Ok(TfhePrivateLut { p, q_in, s_lwe, s_br, rns, bsk, pk, dg_lwe, prng })
    }

    /// (q_in, Q, n_lwe, n_br).
    fn moduli(&self) -> (u64, u64, usize, usize) {
        (self.q_in, self.rns.q, self.p.n_lwe, self.p.n_br)
    }

    /// An LWE of the integer phase `m` (mod q_in, already scaled) under s_lwe: (a, b).
    fn encrypt(&mut self, m: u64) -> (Vec<u64>, u64) {
        let c = Lwe::encrypt(m % self.q_in, &self.s_lwe, self.q_in, &self.dg_lwe, &mut self.prng);
        (c.a, c.b)
    }

    /// The packing key switch: LWEs (a, b) mod q_in at coefficients `pos`, times `w` (None: 1) -> GLWE under s_br mod
    /// Q as residues (c0_a, c1_a, c0_b, c1_b).
    #[pyo3(signature = (lwes, pos, w=None))]
    fn pack(&self, py: Python<'_>, lwes: Vec<(Vec<u64>, u64)>, pos: Vec<usize>, w: Option<Vec<i64>>) -> PyResult<Glwe> {
        if lwes.len() != pos.len() || pos.iter().any(|&x| x >= self.p.n_br) || lwes.iter().any(|l| l.0.len() != self.p.n_lwe) {
            return Err(ve("pack: one position < n_br per LWE of dimension n_lwe"));
        }
        let cts: Vec<Lwe> = lwes.into_iter().map(|(a, b)| Lwe { a, b }).collect();
        Ok(py.allow_threads(|| glwe_of(&self.pk.pack(&cts, self.q_in, &pos, w.as_deref()))))
    }

    /// The blind rotation of the accumulator `acc` (a GLWE from `pack`) by the LWE `idx` (mod q_in), nu = 1:
    /// the rotated GLWE.
    #[pyo3(signature = (acc, idx, k=1))]
    fn blind_rotate(&self, py: Python<'_>, acc: Glwe, idx: (Vec<u64>, u64), k: usize) -> Glwe {
        let ct = Lwe { a: idx.0, b: idx.1 };
        py.allow_threads(|| glwe_of(&self.bsk.blind_rotate_acc(&[rlwe_of(&acc)], &ct, self.q_in, k)[0]))
    }

    /// The sample extraction of coefficient `j` of a GLWE: an LWE (dimension n_br, mod Q) under s_br.
    #[pyo3(signature = (acc, j=0))]
    fn extract(&self, acc: Glwe, j: usize) -> (Vec<u64>, u64) {
        let c = rlwe_of(&acc);
        let ex = [0, 1].map(|p| c[p].sample_extract(j, self.rns.ntt[p].q));
        ((0..self.p.n_br).map(|k| self.rns.crt(ex[0].a[k], ex[1].a[k])).collect(), self.rns.crt(ex[0].b, ex[1].b))
    }

    /// Decryption (test use): the centered phase of an extracted LWE mod Q under s_br.
    fn phase_out(&self, lwe: (Vec<u64>, u64)) -> i128 {
        let c = Lwe { a: lwe.0, b: lwe.1 };
        pc_tfhe::modq::centered(c.phase(&self.s_br, self.rns.q), self.rns.q) as i128
    }

    /// Decryption (test use): a GLWE's centered phase coefficients mod Q.
    fn phase_glwe(&self, acc: Glwe) -> Vec<i128> {
        pc_tfhe::pack::glwe_phase(&rlwe_of(&acc), &self.s_br, &self.rns)
    }

    /// The noise model: std (Q units) of a packed coefficient (`k` inputs, weight `w`) and of the blind rotation;
    /// log2 p_fail of an index lookup with `span` coefficients per entry (nu = 1) and of an output decision at
    /// `spacing` (Q units).
    #[pyo3(signature = (k, span, spacing, w=1.0))]
    fn noise(&self, k: usize, span: usize, spacing: f64, w: f64) -> (f64, f64, f64, f64) {
        let acc = self.pk.noise_var(k, w, self.q_in, self.dg_lwe.sigma());
        let br = pc_tfhe::pack::blind_rotate_var(self.rns.q, self.p.n_br, self.p.n_lwe, self.p.pbs_group, self.p.pbs, self.p.bsk_sigma);
        let idx = pc_tfhe::pack::index_log2_fail(self.p.n_lwe, span, 1, self.p.lwe_sigma_rel);
        let out = pc_tfhe::pack::log2_tail(spacing / 2.0 / (acc + br).sqrt());
        (acc.sqrt(), br.sqrt(), idx, out)
    }
}
