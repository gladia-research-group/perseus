//! Programmable bootstrapping: GGSW bootstrapping keys over the GLWE (k = 1) of degree n_br at Q = q_a q_b (two
//! ~25-bit NTT primes in RNS: the device's u32 butterflies), the external product, CMux blind rotation, sample
//! extraction. Values crossing the interface (test vector, output LWE) are mod Q; inside, every polynomial is a pair
//! of residue polynomials and a gadget digit is taken from the CRT of the pair (`crt`). A LUT f on [-1, 1) reads an
//! input of phase m q / 4 (the forward switch's scale): phi = round(2 n_br phase / q) lies in [-n_br/2, n_br/2), and the
//! negacyclic test vector holds f(2 phi / n_br) for phi >= 0 at index phi and -f(2 (j - n_br) / n_br) at index j in
//! [n_br/2, n_br).
//! The blind rotation is unrolled by two key bits (Bourse, Minelli, Minihold, Paillier 2018; Zhou et al. 2018): per pair
//! (i, j) the key holds GGSW(s_i s_j), GGSW(s_i (1 - s_j)), GGSW((1 - s_i) s_j) and
//! acc <- acc + sum_m (X^-e_m - 1) (K_m [x] acc), e = (a_i + a_j, a_i, a_j): one decomposition and one set of NTTs per
//! pair (half the transforms of the per-bit CMux) for three key MACs, at twice the key noise per term.
use pc_core::keys::{DiscreteGaussian, Prng};
use rayon::prelude::*;

use crate::gadget::Gadget;
use crate::lwe::Lwe;
use crate::modq::{addmod, from_signed, invmod, mulmod, submod, Ntt};
use crate::rlwe::{add_into, monomial_mul, Rlwe};

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

    /// NTT(X^e - 1) at prime p, e in [0, 2n).
    pub fn monomial_m1(&self, p: usize, e: usize) -> Vec<u64> {
        let (n, q) = (self.n(), self.ntt[p].q);
        let mut f = vec![0u64; n];
        f[e % n] = if e < n { 1 } else { q - 1 };
        f[0] = submod(f[0], 1, q);
        self.ntt[p].fwd(&mut f);
        f
    }

    /// The device's tables for NTT(X^e - 1) at prime p: (g^x - 1 for x in [0, 2n), E [n]) with NTT(X)[j] = g^E[j]
    /// (g = NTT(X)[0], a primitive 2n-th root): NTT(X^e - 1)[j] = (g^(E[j] e mod 2n)) - 1.
    pub fn monomial_tables(&self, p: usize) -> (Vec<u64>, Vec<u32>) {
        let (n, q) = (self.n(), self.ntt[p].q);
        let x = {
            let mut f = vec![0u64; n];
            f[1] = 1;
            self.ntt[p].fwd(&mut f);
            f
        };
        let mut pw = Vec::with_capacity(2 * n);
        let mut log = std::collections::HashMap::new();
        let mut v = 1u64;
        for k in 0..2 * n {
            pw.push(v);
            log.insert(v, k as u32);
            v = mulmod(v, x[0], q);
        }
        let e = x.iter().map(|y| log[y]).collect();
        (pw.into_iter().map(|v| submod(v, 1, q)).collect(), e)
    }
}

/// The gadget digits of both components of ct (residues per prime) in NTT form: [c][t][prime].
fn ntt_digits(ct: &[Rlwe; 2], gadget: &Gadget, rns: &Rns) -> Vec<Vec<[Vec<u64>; 2]>> {
    let (n, l) = (rns.n(), gadget.levels);
    let mut d = vec![0i64; l];
    (0..2)
        .map(|c| {
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
            for dt in digits.iter_mut() {
                for (p, dp) in dt.iter_mut().enumerate() {
                    rns.ntt[p].fwd(dp);
                }
            }
            digits
        })
        .collect()
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

    /// sum_{c, t} digits[c][t] * rows[c][t]: per prime (part 0, part 1), NTT form.
    fn mac(&self, digits: &[Vec<[Vec<u64>; 2]>], rns: &Rns) -> [(Vec<u64>, Vec<u64>); 2] {
        let n = rns.n();
        let mut out = [0, 1].map(|_| (vec![0u64; n], vec![0u64; n]));
        for (c, dc) in digits.iter().enumerate() {
            for (t, dt) in dc.iter().enumerate() {
                for (p, dp) in dt.iter().enumerate() {
                    let q = rns.ntt[p].q;
                    let (r0, r1) = &self.rows[c][t][p];
                    for i in 0..n {
                        out[p].0[i] = addmod(out[p].0[i], mulmod(dp[i], r0[i], q), q);
                        out[p].1[i] = addmod(out[p].1[i], mulmod(dp[i], r1[i], q), q);
                    }
                }
            }
        }
        out
    }

    /// GGSW(mu) [x] ct (residues per prime): phase mu (c0 + c1 s) + noise.
    pub fn external_product(&self, ct: &[Rlwe; 2], gadget: &Gadget, rns: &Rns) -> [Rlwe; 2] {
        let mut out = self.mac(&ntt_digits(ct, gadget, rns), rns);
        [0, 1].map(|p| {
            let (mut o0, mut o1) = std::mem::take(&mut out[p]);
            rns.ntt[p].inv(&mut o0);
            rns.ntt[p].inv(&mut o1);
            Rlwe { c0: o0, c1: o1 }
        })
    }
}

/// The bootstrapping key, unrolled by pairs of the binary LWE key (even dimension): per pair (s_i, s_j)
/// GGSW(s_i s_j), GGSW(s_i (1 - s_j)), GGSW((1 - s_i) s_j), under the GLWE key s_br.
pub struct BootstrapKey {
    pub gadget: Gadget,
    pub rns: Rns,
    pub ggsw: Vec<[Ggsw; 3]>,
}

impl BootstrapKey {
    pub fn generate(s_lwe: &[i64], s_br: &[i64], gadget: Gadget, rns: Rns, dg: &DiscreteGaussian, prng: &mut Prng) -> BootstrapKey {
        assert!(s_lwe.len() % 2 == 0, "the unrolled bootstrapping key pairs the LWE key");
        let ggsw = s_lwe
            .chunks(2)
            .map(|p| [p[0] * p[1], p[0] * (1 - p[1]), (1 - p[0]) * p[1]].map(|mu| Ggsw::encrypt(mu, s_br, &gadget, &rns, dg, prng)))
            .collect();
        BootstrapKey { gadget, rns, ggsw }
    }

    /// acc = (X^-b~ tv, 0), then per key pair acc <- acc + sum_m (X^-e_m - 1) (K_m [x] acc) (module doc): the phase
    /// rotates by X^-phi.
    pub fn blind_rotate(&self, tv: &[u64], ct: &Lwe, q_in: u64) -> [Rlwe; 2] {
        let n = self.rns.n();
        let two_n = 2 * n as u64;
        let m = crate::gadget::fixed_ratio(two_n, q_in);
        let sw = |x: u64| -> usize { (crate::gadget::scale_round(x, m) % two_n) as usize };
        let neg = |k: usize| (2 * n - k) % (2 * n);
        let tvr = self.rns.residues(tv);
        let mut acc = [0, 1].map(|p| Rlwe { c0: monomial_mul(&tvr[p], neg(sw(ct.b)), self.rns.ntt[p].q), c1: vec![0; n] });
        for (a, keys) in ct.a.chunks(2).zip(&self.ggsw) {
            let (ki, kj) = (sw(a[0]), sw(a[1]));
            if ki == 0 && kj == 0 {
                continue;
            }
            let digits = ntt_digits(&acc, &self.gadget, &self.rns);
            let mut out = [0, 1].map(|_| (vec![0u64; n], vec![0u64; n]));
            for (key, e) in keys.iter().zip([neg((ki + kj) % (2 * n)), neg(ki), neg(kj)]) {
                if e == 0 {
                    continue;
                }
                let pr = key.mac(&digits, &self.rns);
                for p in 0..2 {
                    let q = self.rns.ntt[p].q;
                    let f = self.rns.monomial_m1(p, e);
                    for i in 0..n {
                        out[p].0[i] = addmod(out[p].0[i], mulmod(f[i], pr[p].0[i], q), q);
                        out[p].1[i] = addmod(out[p].1[i], mulmod(f[i], pr[p].1[i], q), q);
                    }
                }
            }
            for p in 0..2 {
                let ntt = &self.rns.ntt[p];
                let (o0, o1) = &mut out[p];
                ntt.inv(o0);
                ntt.inv(o1);
                add_into(&mut acc[p].c0, o0, ntt.q);
                add_into(&mut acc[p].c1, o1, ntt.q);
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
