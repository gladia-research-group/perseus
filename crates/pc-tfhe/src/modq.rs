//! Arithmetic mod a prime q < 2^62 (u128 products) and the negacyclic NTT of any power-of-two degree over it: the host
//! reference's one ring engine (pc-core's `NttTab` is u32-only; the blind rotation runs at a ~50-bit prime).
use crate::{Error, Result};

#[inline]
pub fn mulmod(a: u64, b: u64, q: u64) -> u64 {
    ((a as u128 * b as u128) % q as u128) as u64
}

#[inline]
pub fn addmod(a: u64, b: u64, q: u64) -> u64 {
    let s = a + b;
    if s >= q {
        s - q
    } else {
        s
    }
}

#[inline]
pub fn submod(a: u64, b: u64, q: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        a + q - b
    }
}

pub fn powmod(mut b: u64, mut e: u64, q: u64) -> u64 {
    let mut r = 1 % q;
    b %= q;
    while e > 0 {
        if e & 1 == 1 {
            r = mulmod(r, b, q);
        }
        b = mulmod(b, b, q);
        e >>= 1;
    }
    r
}

pub fn invmod(a: u64, q: u64) -> u64 {
    powmod(a, q - 2, q)
}

/// The signed representative of `a` in (-q/2, q/2].
#[inline]
pub fn centered(a: u64, q: u64) -> i64 {
    if a > q / 2 {
        a as i64 - q as i64
    } else {
        a as i64
    }
}

#[inline]
pub fn from_signed(v: i64, q: u64) -> u64 {
    v.rem_euclid(q as i64) as u64
}

/// Deterministic Miller-Rabin for n < 2^64.
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n % p == 0 {
            return n == p;
        }
    }
    let (mut d, mut s) = (n - 1, 0);
    while d % 2 == 0 {
        d /= 2;
        s += 1;
    }
    'w: for a in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        let mut x = powmod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = mulmod(x, x, n);
            if x == n - 1 {
                continue 'w;
            }
        }
        return false;
    }
    true
}

/// The largest prime below 2^bits congruent to 1 mod m.
pub fn prime_below(bits: u32, m: u64) -> Result<u64> {
    let mut p = ((1u64 << bits) - 1) / m * m + 1;
    while p > 1u64 << (bits - 1) {
        if is_prime(p) {
            return Ok(p);
        }
        p -= m;
    }
    Err(Error::Params(format!("no {bits}-bit prime = 1 mod {m}")))
}

fn primitive_root(q: u64) -> u64 {
    let mut fs = vec![];
    let mut m = q - 1;
    let mut f = 2;
    while f * f <= m {
        if m % f == 0 {
            fs.push(f);
            while m % f == 0 {
                m /= f;
            }
        }
        f += 1;
    }
    if m > 1 {
        fs.push(m);
    }
    (2..).find(|&g| fs.iter().all(|&f| powmod(g, (q - 1) / f, q) != 1)).unwrap()
}

fn brev(x: usize, bits: u32) -> usize {
    if bits == 0 {
        0
    } else {
        x.reverse_bits() >> (usize::BITS - bits)
    }
}

/// Negacyclic NTT tables of one prime at one degree (`psi` powers in bit-reversed order, the pc-core convention).
#[derive(Clone, Debug)]
pub struct Ntt {
    pub q: u64,
    pub n: usize,
    psi: Vec<u64>,
    ipsi: Vec<u64>,
    ninv: u64,
}

impl Ntt {
    pub fn new(q: u64, n: usize) -> Result<Ntt> {
        if !n.is_power_of_two() || (q - 1) % (2 * n as u64) != 0 || !is_prime(q) || q >= 1 << 62 {
            return Err(Error::Params(format!("ntt: {q} is not a prime < 2^62 = 1 mod 2*{n}")));
        }
        let psi = powmod(primitive_root(q), (q - 1) / (2 * n as u64), q);
        let ipsi = invmod(psi, q);
        let bits = n.trailing_zeros();
        let (mut pw, mut ipw) = (vec![0u64; n], vec![0u64; n]);
        let (mut a, mut b) = (1u64, 1u64);
        for k in 0..n {
            let r = brev(k, bits);
            pw[r] = a;
            ipw[r] = b;
            a = mulmod(a, psi, q);
            b = mulmod(b, ipsi, q);
        }
        Ok(Ntt { q, n, psi: pw, ipsi: ipw, ninv: invmod(n as u64 % q, q) })
    }

    /// Forward, in place (natural -> bit-reversed).
    pub fn fwd(&self, a: &mut [u64]) {
        let (n, q) = (self.n, self.q);
        let (mut m, mut t) = (1, n / 2);
        while m < n {
            for i in 0..m {
                let w = self.psi[m + i];
                let j0 = 2 * i * t;
                for j in j0..j0 + t {
                    let u = a[j];
                    let v = mulmod(a[j + t], w, q);
                    a[j] = addmod(u, v, q);
                    a[j + t] = submod(u, v, q);
                }
            }
            m *= 2;
            t /= 2;
        }
    }

    /// Inverse, in place (bit-reversed -> natural, times n^-1).
    pub fn inv(&self, a: &mut [u64]) {
        let (n, q) = (self.n, self.q);
        let (mut m, mut t) = (n / 2, 1);
        while m >= 1 {
            for i in 0..m {
                let w = self.ipsi[m + i];
                let j0 = 2 * i * t;
                for j in j0..j0 + t {
                    let (u, v) = (a[j], a[j + t]);
                    a[j] = addmod(u, v, q);
                    a[j + t] = mulmod(submod(u, v, q), w, q);
                }
            }
            m /= 2;
            t *= 2;
        }
        for x in a.iter_mut() {
            *x = mulmod(*x, self.ninv, q);
        }
    }

    /// a * b mod (X^n + 1, q), coefficient form in and out.
    pub fn mul(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        let (mut fa, mut fb) = (a.to_vec(), b.to_vec());
        self.fwd(&mut fa);
        self.fwd(&mut fb);
        for (x, y) in fa.iter_mut().zip(&fb) {
            *x = mulmod(*x, *y, self.q);
        }
        self.inv(&mut fa);
        fa
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ntt_matches_schoolbook() {
        let q = prime_below(50, 2 * 64).unwrap();
        let t = Ntt::new(q, 64).unwrap();
        let a: Vec<u64> = (0..64).map(|i| (i * 7919 + 3) % q).collect();
        let b: Vec<u64> = (0..64).map(|i| (i * i * 104729 + 11) % q).collect();
        let mut want = vec![0u64; 64];
        for i in 0..64 {
            for j in 0..64 {
                let p = mulmod(a[i], b[j], q);
                let k = i + j;
                if k < 64 {
                    want[k] = addmod(want[k], p, q);
                } else {
                    want[k - 64] = submod(want[k - 64], p, q);
                }
            }
        }
        assert_eq!(t.mul(&a, &b), want);
    }
}
