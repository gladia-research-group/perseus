//! Programmable bootstrapping: GGSW bootstrapping keys over the GLWE (k = 1) of degree n_br at Q = q_a q_b (two
//! ~25-bit NTT primes in RNS: the device's u32 butterflies), the external product, CMux blind rotation, sample
//! extraction. Values crossing the interface (test vector, output LWE) are mod Q; inside, every polynomial is a pair
//! of residue polynomials and a gadget digit is taken from the CRT of the pair (`crt`). A LUT f on [-1, 1) reads an
//! input of phase m q / 4 (the forward switch's scale): phi = round(2 n_br phase / q) lies in [-n_br/2, n_br/2), and the
//! negacyclic test vector holds f(2 phi / n_br) for phi >= 0 at index phi and -f(2 (j - n_br) / n_br) at index j in
//! [n_br/2, n_br).
use pc_core::keys::{DiscreteGaussian, Prng};
use rayon::prelude::*;

use crate::gadget::Gadget;
use crate::lwe::Lwe;
use crate::modq::{addmod, from_signed, invmod, mulmod, submod, Ntt};
use crate::rlwe::{add_into, monomial_mul, sub_into, Rlwe};

/// The RNS ring of the blind rotation: Q = q_a q_b.
#[derive(Clone, Debug)]
pub struct Rns {
    pub ntt: [Ntt; 2],
    pub q: u64,
    /// q_a^-1 mod q_b
    pub inv: u64,
}

impl Rns {
    pub fn new(qa: u64, qb: u64, n: usize) -> crate::Result<Rns> {
        Ok(Rns { ntt: [Ntt::new(qa, n)?, Ntt::new(qb, n)?], q: qa * qb, inv: invmod(qa % qb, qb) })
    }

    pub fn n(&self) -> usize {
        self.ntt[0].n
    }

    /// x mod Q from its residues: r_a + q_a ((r_b - r_a) q_a^-1 mod q_b).
    #[inline]
    pub fn crt(&self, ra: u64, rb: u64) -> u64 {
        let (qa, qb) = (self.ntt[0].q, self.ntt[1].q);
        ra + qa * mulmod(submod(rb, ra % qb, qb), self.inv, qb)
    }

    pub fn residues(&self, v: &[u64]) -> [Vec<u64>; 2] {
        [0, 1].map(|l| v.iter().map(|&x| x % self.ntt[l].q).collect())
    }
}

/// GGSW of a small integer mu: rows (Z + mu g_t (1, 0)) and (Z + mu g_t (0, 1)), per prime, stored in NTT form.
pub struct Ggsw {
    /// [component c][level t][prime] -> (c0, c1) in NTT form
    rows: Vec<Vec<[(Vec<u64>, Vec<u64>); 2]>>,
}

impl Ggsw {
    pub fn encrypt(mu: i64, s: &[i64], gadget: &Gadget, rns: &Rns, dg: &DiscreteGaussian, prng: &mut Prng) -> Ggsw {
        let n = rns.n();
        let rows = (0..2)
            .map(|c| {
                (0..gadget.levels)
                    .map(|t| {
                        // an encryption of zero mod Q: c1 uniform per prime, one integer error
                        let e = dg.sample_vec(prng, n);
                        let v = mulmod(from_signed(mu, rns.q), gadget.g(t), rns.q);
                        [0, 1].map(|l| {
                            let ntt = &rns.ntt[l];
                            let q = ntt.q;
                            let c1: Vec<u64> = (0..n).map(|_| prng.uniform_below(q)).collect();
                            let mut z = Rlwe::encrypt_with(&vec![0; n], s, c1, &e, ntt);
                            let target = if c == 0 { &mut z.c0 } else { &mut z.c1 };
                            target[0] = addmod(target[0], v % q, q);
                            ntt.fwd(&mut z.c0);
                            ntt.fwd(&mut z.c1);
                            (z.c0, z.c1)
                        })
                    })
                    .collect()
            })
            .collect();
        Ggsw { rows }
    }

    /// [component c][level t][prime] -> (c0, c1), NTT form: the device layout's source.
    pub fn rows(&self) -> &[Vec<[(Vec<u64>, Vec<u64>); 2]>] {
        &self.rows
    }

    /// GGSW(mu) [x] ct (residues per prime): phase mu (c0 + c1 s) + noise.
    pub fn external_product(&self, ct: &[Rlwe; 2], gadget: &Gadget, rns: &Rns) -> [Rlwe; 2] {
        let (n, l) = (rns.n(), gadget.levels);
        let mut out = [0, 1].map(|_| (vec![0u64; n], vec![0u64; n]));
        let mut d = vec![0i64; l];
        for c in 0..2 {
            let (pa, pb) = if c == 0 { (&ct[0].c0, &ct[1].c0) } else { (&ct[0].c1, &ct[1].c1) };
            let mut digits = vec![[vec![0u64; n], vec![0u64; n]]; l];
            for i in 0..n {
                gadget.decompose(rns.crt(pa[i], pb[i]), &mut d);
                for t in 0..l {
                    for p in 0..2 {
                        digits[t][p][i] = from_signed(d[t], rns.ntt[p].q);
                    }
                }
            }
            for (t, dt) in digits.into_iter().enumerate() {
                for (p, mut dp) in dt.into_iter().enumerate() {
                    let ntt = &rns.ntt[p];
                    let q = ntt.q;
                    ntt.fwd(&mut dp);
                    let (r0, r1) = &self.rows[c][t][p];
                    for i in 0..n {
                        out[p].0[i] = addmod(out[p].0[i], mulmod(dp[i], r0[i], q), q);
                        out[p].1[i] = addmod(out[p].1[i], mulmod(dp[i], r1[i], q), q);
                    }
                }
            }
        }
        [0, 1].map(|p| {
            let (mut o0, mut o1) = std::mem::take(&mut out[p]);
            rns.ntt[p].inv(&mut o0);
            rns.ntt[p].inv(&mut o1);
            Rlwe { c0: o0, c1: o1 }
        })
    }
}

/// The bootstrapping key: GGSW(s_i) for the binary LWE key, under the GLWE key s_br.
pub struct BootstrapKey {
    pub gadget: Gadget,
    pub rns: Rns,
    pub ggsw: Vec<Ggsw>,
}

impl BootstrapKey {
    pub fn generate(s_lwe: &[i64], s_br: &[i64], gadget: Gadget, rns: Rns, dg: &DiscreteGaussian, prng: &mut Prng) -> BootstrapKey {
        let ggsw = s_lwe.iter().map(|&si| Ggsw::encrypt(si, s_br, &gadget, &rns, dg, prng)).collect();
        BootstrapKey { gadget, rns, ggsw }
    }

    /// acc = (X^-b~ tv, 0), then acc <- acc + GGSW(s_i) [x] (acc X^-a~_i - acc): the phase rotates by X^-phi.
    pub fn blind_rotate(&self, tv: &[u64], ct: &Lwe, q_in: u64) -> [Rlwe; 2] {
        let n = self.rns.n();
        let two_n = 2 * n as u64;
        let m = crate::gadget::fixed_ratio(two_n, q_in);
        let sw = |x: u64| -> usize { (crate::gadget::scale_round(x, m) % two_n) as usize };
        let neg = |k: usize| (2 * n - k) % (2 * n);
        let tvr = self.rns.residues(tv);
        let mut acc = [0, 1].map(|p| Rlwe { c0: monomial_mul(&tvr[p], neg(sw(ct.b)), self.rns.ntt[p].q), c1: vec![0; n] });
        for (ai, key) in ct.a.iter().zip(&self.ggsw) {
            let k = sw(*ai);
            if k == 0 {
                continue;
            }
            let diff = [0, 1].map(|p| {
                let q = self.rns.ntt[p].q;
                let mut d = acc[p].monomial_mul(neg(k), q);
                sub_into(&mut d.c0, &acc[p].c0, q);
                sub_into(&mut d.c1, &acc[p].c1, q);
                d
            });
            let pr = key.external_product(&diff, &self.gadget, &self.rns);
            for p in 0..2 {
                let q = self.rns.ntt[p].q;
                add_into(&mut acc[p].c0, &pr[p].c0, q);
                add_into(&mut acc[p].c1, &pr[p].c1, q);
            }
        }
        acc
    }

    /// The LUT's output LWE (dimension n_br, mod Q, phase f(m) + offset).
    pub fn bootstrap(&self, tv: &[u64], offset: u64, ct: &Lwe, q_in: u64) -> Lwe {
        let acc = self.blind_rotate(tv, ct, q_in);
        let ex = [0, 1].map(|p| acc[p].sample_extract(0, self.rns.ntt[p].q));
        let q = self.rns.q;
        let a = (0..self.rns.n()).map(|k| self.rns.crt(ex[0].a[k], ex[1].a[k])).collect();
        Lwe { a, b: addmod(self.rns.crt(ex[0].b, ex[1].b), offset, q) }
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
