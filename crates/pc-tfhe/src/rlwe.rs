//! RLWE ciphertexts mod a prime (phase c0 + c1 s mod X^N + 1), sample extraction, negacyclic monomials, and an RNS
//! wrapper (one RLWE per prime, the same integer error and message) for the mod q0 = p0 p1 return path.
use pc_core::keys::{DiscreteGaussian, Prng};

use crate::lwe::Lwe;
use crate::modq::{addmod, from_signed, submod, Ntt};

#[derive(Clone, Debug, PartialEq)]
pub struct Rlwe {
    pub c0: Vec<u64>,
    pub c1: Vec<u64>,
}

pub fn signed_poly(s: &[i64], q: u64) -> Vec<u64> {
    s.iter().map(|&x| from_signed(x, q)).collect()
}

pub fn add_into(a: &mut [u64], b: &[u64], q: u64) {
    for (x, &y) in a.iter_mut().zip(b) {
        *x = addmod(*x, y, q);
    }
}

pub fn sub_into(a: &mut [u64], b: &[u64], q: u64) {
    for (x, &y) in a.iter_mut().zip(b) {
        *x = submod(*x, y, q);
    }
}

/// a X^k mod X^N + 1, k in [0, 2N).
pub fn monomial_mul(a: &[u64], k: usize, q: u64) -> Vec<u64> {
    let n = a.len();
    let k = k % (2 * n);
    let mut out = vec![0u64; n];
    for (i, &v) in a.iter().enumerate() {
        let j = i + k;
        let (idx, neg) = if j < n {
            (j, false)
        } else if j < 2 * n {
            (j - n, true)
        } else {
            (j - 2 * n, false)
        };
        out[idx] = if neg && v != 0 { q - v } else { v };
    }
    out
}

impl Rlwe {
    pub fn zero(n: usize) -> Rlwe {
        Rlwe { c0: vec![0; n], c1: vec![0; n] }
    }

    /// Encryption of the polynomial m (coefficient form, already scaled, mod q) under the small secret s.
    pub fn encrypt(m: &[u64], s: &[i64], ntt: &Ntt, dg: &DiscreteGaussian, prng: &mut Prng) -> Rlwe {
        let (n, q) = (ntt.n, ntt.q);
        let c1: Vec<u64> = (0..n).map(|_| prng.uniform_below(q)).collect();
        Rlwe::encrypt_with(m, s, c1, &dg.sample_vec(prng, n), ntt)
    }

    /// c0 = m + e - c1 s for a given c1 and integer error e.
    pub fn encrypt_with(m: &[u64], s: &[i64], c1: Vec<u64>, e: &[i64], ntt: &Ntt) -> Rlwe {
        let q = ntt.q;
        let c1s = ntt.mul(&c1, &signed_poly(s, q));
        let c0 = (0..ntt.n).map(|i| submod(addmod(m[i], from_signed(e[i], q), q), c1s[i], q)).collect();
        Rlwe { c0, c1 }
    }

    pub fn phase(&self, s: &[i64], ntt: &Ntt) -> Vec<u64> {
        let mut p = ntt.mul(&self.c1, &signed_poly(s, ntt.q));
        add_into(&mut p, &self.c0, ntt.q);
        p
    }

    /// The LWE of coefficient j under the secret's coefficient vector: a_k = c1[j-k] (k <= j), -c1[N+j-k] (k > j).
    pub fn sample_extract(&self, j: usize, q: u64) -> Lwe {
        let n = self.c0.len();
        let a = (0..n)
            .map(|k| {
                if k <= j {
                    self.c1[j - k]
                } else {
                    let v = self.c1[n + j - k];
                    if v == 0 {
                        0
                    } else {
                        q - v
                    }
                }
            })
            .collect();
        Lwe { a, b: self.c0[j] }
    }

    pub fn monomial_mul(&self, k: usize, q: u64) -> Rlwe {
        Rlwe { c0: monomial_mul(&self.c0, k, q), c1: monomial_mul(&self.c1, k, q) }
    }
}

/// An RLWE over q = prod p_l as one `Rlwe` per prime (the CKKS limb layout).
#[derive(Clone, Debug, PartialEq)]
pub struct RnsRlwe {
    pub limbs: Vec<Rlwe>,
}

impl RnsRlwe {
    /// Encryption of the integer polynomial m (centered) under s: c1 uniform per limb, one integer error.
    pub fn encrypt(m: &[i128], s: &[i64], ntts: &[Ntt], dg: &DiscreteGaussian, prng: &mut Prng) -> RnsRlwe {
        let n = ntts[0].n;
        let e = dg.sample_vec(prng, n);
        let limbs = ntts
            .iter()
            .map(|t| {
                let c1: Vec<u64> = (0..n).map(|_| prng.uniform_below(t.q)).collect();
                let ml: Vec<u64> = m.iter().map(|&v| v.rem_euclid(t.q as i128) as u64).collect();
                Rlwe::encrypt_with(&ml, s, c1, &e, t)
            })
            .collect();
        RnsRlwe { limbs }
    }

    /// The phase as centered integers mod prod p_l (CRT of the limb phases; two limbs).
    pub fn phase(&self, s: &[i64], ntts: &[Ntt]) -> Vec<i128> {
        assert_eq!(ntts.len(), 2, "RnsRlwe::phase: two limbs");
        let (p0, p1) = (ntts[0].q as i128, ntts[1].q as i128);
        let ph: Vec<Vec<u64>> = self.limbs.iter().zip(ntts).map(|(c, t)| c.phase(s, t)).collect();
        let inv = crate::modq::invmod(ntts[0].q % ntts[1].q, ntts[1].q) as i128;
        let q = p0 * p1;
        (0..ntts[0].n)
            .map(|i| {
                let (x0, x1) = (ph[0][i] as i128, ph[1][i] as i128);
                let k = ((x1 - x0).rem_euclid(p1) * inv).rem_euclid(p1);
                let v = x0 + p0 * k;
                if v > q / 2 {
                    v - q
                } else {
                    v
                }
            })
            .collect()
    }
}
