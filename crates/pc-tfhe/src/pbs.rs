//! Programmable bootstrapping: GGSW bootstrapping keys over the GLWE (k = 1) of degree n_br at q_br, the external
//! product, CMux blind rotation, sample extraction. A LUT f on [-1, 1) reads an input of phase m q / 4 (the forward
//! switch's scale): phi = round(2 n_br phase / q) lies in [-n_br/2, n_br/2), and the negacyclic test vector holds
//! f(2 phi / n_br) for phi >= 0 at index phi and -f(2 (j - n_br) / n_br) at index j in [n_br/2, n_br).
use pc_core::keys::{DiscreteGaussian, Prng};
use rayon::prelude::*;

use crate::gadget::Gadget;
use crate::lwe::Lwe;
use crate::modq::{addmod, mulmod, Ntt};
use crate::rlwe::{add_into, sub_into, Rlwe};

/// GGSW of a small integer mu: rows (Z + mu g_t (1, 0)) and (Z + mu g_t (0, 1)), stored in NTT form.
pub struct Ggsw {
    /// [component c][level t] -> (c0, c1) in NTT form
    rows: Vec<Vec<(Vec<u64>, Vec<u64>)>>,
}

impl Ggsw {
    pub fn encrypt(mu: i64, s: &[i64], gadget: &Gadget, ntt: &Ntt, dg: &DiscreteGaussian, prng: &mut Prng) -> Ggsw {
        let (n, q) = (ntt.n, ntt.q);
        let rows = (0..2)
            .map(|c| {
                (0..gadget.levels)
                    .map(|t| {
                        let mut z = Rlwe::encrypt(&vec![0; n], s, ntt, dg, prng);
                        let v = mulmod(crate::modq::from_signed(mu, q), gadget.g(t), q);
                        let target = if c == 0 { &mut z.c0 } else { &mut z.c1 };
                        target[0] = addmod(target[0], v, q);
                        ntt.fwd(&mut z.c0);
                        ntt.fwd(&mut z.c1);
                        (z.c0, z.c1)
                    })
                    .collect()
            })
            .collect();
        Ggsw { rows }
    }

    /// GGSW(mu) [x] ct: phase mu (c0 + c1 s) + noise.
    pub fn external_product(&self, ct: &Rlwe, gadget: &Gadget, ntt: &Ntt) -> Rlwe {
        let (n, q, l) = (ntt.n, ntt.q, gadget.levels);
        let mut out0 = vec![0u64; n];
        let mut out1 = vec![0u64; n];
        let mut d = vec![0i64; l];
        for (c, poly) in [&ct.c0, &ct.c1].into_iter().enumerate() {
            let mut digits = vec![vec![0u64; n]; l];
            for (i, &x) in poly.iter().enumerate() {
                gadget.decompose(x, &mut d);
                for t in 0..l {
                    digits[t][i] = crate::modq::from_signed(d[t], q);
                }
            }
            for (t, mut dt) in digits.into_iter().enumerate() {
                ntt.fwd(&mut dt);
                let (r0, r1) = &self.rows[c][t];
                for i in 0..n {
                    out0[i] = addmod(out0[i], mulmod(dt[i], r0[i], q), q);
                    out1[i] = addmod(out1[i], mulmod(dt[i], r1[i], q), q);
                }
            }
        }
        ntt.inv(&mut out0);
        ntt.inv(&mut out1);
        Rlwe { c0: out0, c1: out1 }
    }
}

/// The bootstrapping key: GGSW(s_i) for the binary LWE key, under the GLWE key s_br.
pub struct BootstrapKey {
    pub gadget: Gadget,
    pub ntt: Ntt,
    pub ggsw: Vec<Ggsw>,
}

impl BootstrapKey {
    pub fn generate(s_lwe: &[i64], s_br: &[i64], gadget: Gadget, ntt: Ntt, dg: &DiscreteGaussian, prng: &mut Prng) -> BootstrapKey {
        let ggsw = s_lwe.iter().map(|&si| Ggsw::encrypt(si, s_br, &gadget, &ntt, dg, prng)).collect();
        BootstrapKey { gadget, ntt, ggsw }
    }

    /// acc = (X^-b~ tv, 0), then acc <- acc + GGSW(s_i) [x] (acc X^-a~_i - acc): the phase rotates by X^-phi.
    pub fn blind_rotate(&self, tv: &[u64], ct: &Lwe, q_in: u64) -> Rlwe {
        let n = self.ntt.n;
        let q = self.ntt.q;
        let two_n = 2 * n as u64;
        let sw = |x: u64| -> usize { ((x as u128 * two_n as u128 + q_in as u128 / 2) / q_in as u128 % two_n as u128) as usize };
        let neg = |k: usize| (2 * n - k) % (2 * n);
        let mut acc = Rlwe { c0: crate::rlwe::monomial_mul(tv, neg(sw(ct.b)), q), c1: vec![0; n] };
        for (ai, key) in ct.a.iter().zip(&self.ggsw) {
            let k = sw(*ai);
            if k == 0 {
                continue;
            }
            let mut diff = acc.monomial_mul(neg(k), q);
            sub_into(&mut diff.c0, &acc.c0, q);
            sub_into(&mut diff.c1, &acc.c1, q);
            let p = key.external_product(&diff, &self.gadget, &self.ntt);
            add_into(&mut acc.c0, &p.c0, q);
            add_into(&mut acc.c1, &p.c1, q);
        }
        acc
    }

    /// The LUT's output LWE (dimension n_br, mod q_br, phase f(m) + offset).
    pub fn bootstrap(&self, tv: &[u64], offset: u64, ct: &Lwe, q_in: u64) -> Lwe {
        let mut out = self.blind_rotate(tv, ct, q_in).sample_extract(0, self.ntt.q);
        out.b = addmod(out.b, offset, self.ntt.q);
        out
    }

    pub fn bootstrap_many(&self, tv: &[u64], offset: u64, cts: &[Lwe], q_in: u64) -> Vec<Lwe> {
        cts.par_iter().map(|c| self.bootstrap(tv, offset, c, q_in)).collect()
    }
}

/// The test vector of f (values already scaled mod q) for inputs of phase m q / 4, m in [-1, 1).
pub fn test_vector(n: usize, q: u64, f: impl Fn(f64) -> u64) -> Vec<u64> {
    (0..n)
        .map(|j| {
            if j < n / 2 {
                f(2.0 * j as f64 / n as f64)
            } else {
                let v = f(2.0 * (j as f64 - n as f64) / n as f64);
                if v == 0 {
                    0
                } else {
                    q - v
                }
            }
        })
        .collect()
}

/// (1 + sign(m)) / 2 at scale `delta`: a constant test vector delta / 2 (sign by negacyclicity) plus the offset
/// delta / 2. Returns (tv, offset).
pub fn sign01(n: usize, delta: u64) -> (Vec<u64>, u64) {
    (vec![delta / 2; n], delta / 2)
}
