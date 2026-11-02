//! Programmable bootstrapping: GGSW bootstrapping keys over the GLWE (k = 1) of degree n_br at Q = q_a q_b (two
//! ~26-bit NTT primes in RNS: the device's u32 butterflies), the external product, CMux blind rotation, sample
//! extraction. Values crossing the interface (test vector, output LWE) are mod Q; inside, every polynomial is a pair
//! of residue polynomials and a gadget digit is taken from the CRT of the pair (`crt`). A LUT f on [-1, 1) reads an
//! input of phase m q / 4 (the forward switch's scale): phi = round(2 n_br phase / q) lies in [-n_br/2, n_br/2), and the
//! negacyclic test vector holds f(2 phi / n_br) for phi >= 0 at index phi and -f(2 (j - n_br) / n_br) at index j in
//! [n_br/2, n_br).
//! The blind rotation is unrolled by groups of G key bits (Bourse, Minelli, Minihold, Paillier 2018; Zhou et al. 2018;
//! the multi-bit PBS of TFHE-rs): per group the key holds, for every non-empty subset S (bit mask m = 1 .. 2^G - 1),
//! K_m = GGSW(prod_{x in S} s_x prod_{x not in S} (1 - s_x)) (exactly one is GGSW(1) unless the group's bits are all 0)
//! and acc <- acc + sum_m (X^-e_m - 1) (K_m [x] acc), e_m = sum_{x in S} a_x: one decomposition and one set of NTTs per
//! group (1 / G of the transforms of the per-bit CMux) for 2^G - 1 key MACs, at twice the key noise per term.
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

    /// NTT(X^e) at prime p, e in [0, 2n).
    pub fn monomial(&self, p: usize, e: usize) -> Vec<u64> {
        let (n, q) = (self.n(), self.ntt[p].q);
        let mut f = vec![0u64; n];
        f[e % n] = if e < n { 1 } else { q - 1 };
        self.ntt[p].fwd(&mut f);
        f
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

/// The bootstrapping key, unrolled by groups of `group` bits of the binary LWE key (its dimension a multiple): per
/// group [K_m for m = 1 .. 2^group - 1] (module doc), under the GLWE key s_br.
pub struct BootstrapKey {
    pub gadget: Gadget,
    pub rns: Rns,
    pub group: usize,
    pub ggsw: Vec<Vec<Ggsw>>,
}

impl BootstrapKey {
    pub fn generate(s_lwe: &[i64], s_br: &[i64], gadget: Gadget, rns: Rns, group: usize, dg: &DiscreteGaussian, prng: &mut Prng) -> BootstrapKey {
        assert!(group >= 1 && s_lwe.len() % group == 0, "the unrolled bootstrapping key groups the LWE key");
        let ggsw = s_lwe
            .chunks(group)
            .map(|g| {
                (1..1usize << group)
                    .map(|m| {
                        let mu = g.iter().enumerate().map(|(x, &s)| if m >> x & 1 == 1 { s } else { 1 - s }).product();
                        Ggsw::encrypt(mu, s_br, &gadget, &rns, dg, prng)
                    })
                    .collect()
            })
            .collect();
        BootstrapKey { gadget, rns, group, ggsw }
    }

    /// acc = (X^-b~ tv, 0), then per key group acc <- acc + sum_m (X^-e_m - 1) (K_m [x] acc) (module doc): the phase
    /// rotates by X^-phi.
    pub fn blind_rotate(&self, tv: &[u64], ct: &Lwe, q_in: u64) -> [Rlwe; 2] {
        let [acc] = <[_; 1]>::try_from(self.blind_rotate_ext(tv, ct, q_in, 1)).ok().unwrap();
        acc
    }

    /// The extended blind rotation: acc in R_(nu n) = R_n[X] / (X^nu - Y) as nu GLWE components of R_n under the same
    /// key, the input switched to 2 nu n, tv of nu n entries. Returns the components, [component][prime].
    pub fn blind_rotate_ext(&self, tv: &[u64], ct: &Lwe, q_in: u64, nu: usize) -> Vec<[Rlwe; 2]> {
        self.blind_rotate_mlut(tv, ct, q_in, nu, 1)
    }

    /// `blind_rotate_ext` for `k` interleaved LUTs (many-LUT, Carpov, Izabachene, Mollimard 2019): the input switched
    /// to 2 nu n / k and scaled by k, so coefficient j of acc reads LUT j (tv: `interleave`).
    pub fn blind_rotate_mlut(&self, tv: &[u64], ct: &Lwe, q_in: u64, nu: usize, kl: usize) -> Vec<[Rlwe; 2]> {
        let n = self.rns.n();
        let big = nu * n;
        assert_eq!(tv.len(), big, "the extended test vector has nu n entries");
        assert!(kl >= 1 && (2 * big) % kl == 0, "many-LUT: k divides 2 nu n");
        let two_n = 2 * big as u64;
        let m = crate::gadget::fixed_ratio(two_n / kl as u64, q_in);
        let sw = |x: u64| -> usize { kl * (crate::gadget::scale_round(x, m) % (two_n / kl as u64)) as usize };
        let neg = |k: usize| (2 * big - k) % (2 * big);
        let tvr = self.rns.residues(tv);
        let bt = kl * crate::gadget::ms_body_mean(&ct.a, ct.b, m, two_n / kl as u64) as usize;
        let tvx = [0, 1].map(|p| monomial_mul(&tvr[p], neg(bt), self.rns.ntt[p].q));
        let mut acc: Vec<[Rlwe; 2]> =
            (0..nu).map(|i| [0, 1].map(|p| Rlwe { c0: (0..n).map(|j| tvx[p][nu * j + i]).collect(), c1: vec![0; n] })).collect();
        for (a, keys) in ct.a.chunks(self.group).zip(&self.ggsw) {
            let k: Vec<usize> = a.iter().map(|&x| sw(x)).collect();
            if k.iter().all(|&x| x == 0) {
                continue;
            }
            let digits: Vec<_> = acc.iter().map(|c| ntt_digits(c, &self.gadget, &self.rns)).collect();
            let mut out: Vec<[(Vec<u64>, Vec<u64>); 2]> = (0..nu).map(|_| [0, 1].map(|_| (vec![0u64; n], vec![0u64; n]))).collect();
            let exps = (1..1usize << self.group).map(|m| neg(k.iter().enumerate().filter(|(x, _)| m >> x & 1 == 1).map(|(_, &v)| v).sum::<usize>() % (2 * big)));
            for (key, e) in keys.iter().zip(exps) {
                if e == 0 {
                    continue;
                }
                for (i, di) in digits.iter().enumerate() {
                    let pr = key.mac(di, &self.rns);
                    let ei = (e + i) % (2 * big);
                    let (t, w) = (ei % nu, ei / nu);   // X^e X^i = X^t Y^w, w < 2 n
                    for p in 0..2 {
                        let q = self.rns.ntt[p].q;
                        let f = self.rns.monomial(p, w);
                        for x in 0..n {
                            out[t][p].0[x] = addmod(out[t][p].0[x], mulmod(f[x], pr[p].0[x], q), q);
                            out[t][p].1[x] = addmod(out[t][p].1[x], mulmod(f[x], pr[p].1[x], q), q);
                            out[i][p].0[x] = submod(out[i][p].0[x], pr[p].0[x], q);
                            out[i][p].1[x] = submod(out[i][p].1[x], pr[p].1[x], q);
                        }
                    }
                }
            }
            for (ai, oi) in acc.iter_mut().zip(out.iter_mut()) {
                for p in 0..2 {
                    let ntt = &self.rns.ntt[p];
                    let (o0, o1) = &mut oi[p];
                    ntt.inv(o0);
                    ntt.inv(o1);
                    add_into(&mut ai[p].c0, o0, ntt.q);
                    add_into(&mut ai[p].c1, o1, ntt.q);
                }
            }
        }
        acc
    }

    /// The LUT's output LWE (dimension n_br, mod Q, phase f(m) + offset).
    pub fn bootstrap(&self, tv: &[u64], offset: u64, ct: &Lwe, q_in: u64) -> Lwe {
        self.bootstrap_ext(tv, offset, ct, q_in, 1)
    }

    /// `bootstrap` on the extended blind rotation (`blind_rotate_ext`; tv of nu n entries).
    pub fn bootstrap_ext(&self, tv: &[u64], offset: u64, ct: &Lwe, q_in: u64, nu: usize) -> Lwe {
        self.bootstrap_mlut(tv, &[offset], ct, q_in, nu).swap_remove(0)
    }

    /// The outputs of `offsets.len()` interleaved LUTs from one blind rotation (`blind_rotate_mlut`): LUT j's from
    /// coefficient j, + offsets[j].
    pub fn bootstrap_mlut(&self, tv: &[u64], offsets: &[u64], ct: &Lwe, q_in: u64, nu: usize) -> Vec<Lwe> {
        let acc = self.blind_rotate_mlut(tv, ct, q_in, nu, offsets.len()).swap_remove(0);
        let q = self.rns.q;
        offsets
            .iter()
            .enumerate()
            .map(|(j, &off)| {
                let ex = [0, 1].map(|p| acc[p].sample_extract(j, self.rns.ntt[p].q));
                let a = (0..self.rns.n()).map(|k| self.rns.crt(ex[0].a[k], ex[1].a[k])).collect();
                Lwe { a, b: addmod(self.rns.crt(ex[0].b, ex[1].b), off, q) }
            })
            .collect()
    }

    pub fn bootstrap_many(&self, tv: &[u64], offset: u64, cts: &[Lwe], q_in: u64) -> Vec<Lwe> {
        cts.par_iter().map(|c| self.bootstrap(tv, offset, c, q_in)).collect()
    }
}

/// The output phase x (a fraction of the modulus) of a LUT value v that the CKKS finish reads as v / 2^out_cf. The
/// finish's EvalMod is a sine, sin(2 pi x) / (2 pi), so the LUT stores x = asin(2 pi v / 2^out_cf) / (2 pi) (needs
/// |v| 2 pi <= 2^out_cf): the output may then sit at q / 8 instead of q / 2^8 (5 bits of the bootstrap's noise budget),
/// and the sine's slope there (cos <= 1) only shrinks the noise.
pub fn warp(v: f64, out_cf: u32) -> f64 {
    let t = std::f64::consts::TAU;
    (t * v / (1u64 << out_cf) as f64).asin() / t
}

/// v from the output phase x (`warp`'s inverse: what the finish returns, times 2^out_cf).
pub fn unwarp(x: f64, out_cf: u32) -> f64 {
    let t = std::f64::consts::TAU;
    (t * x).sin() / t * (1u64 << out_cf) as f64
}

/// k test vectors of n / k entries each interleaved into one of n (many-LUT: `BootstrapKey::blind_rotate_mlut`).
pub fn interleave(tvs: &[Vec<u64>]) -> Vec<u64> {
    let k = tvs.len();
    (0..k * tvs[0].len()).map(|x| tvs[x % k][x / k]).collect()
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
