//! Host negacyclic NTT per prime in the device's order (OpenFHE `ForwardTransformToBitReverse`: natural order in,
//! bit-reversed out, twiddle `psi^brev(i)`), with Shoup constants for primes below 2^30, and the chain-wide `Ring`
//! that turns signed small polynomials into eval rows by prime id.
use crate::error::{Error, Result};
use pc_core::modmul::{invmod, mulmod};
use pc_core::params::Chain;
use rayon::prelude::*;

/// `x * w mod p` with `ws = floor(w 2^32 / p)` (Shoup), for `p < 2^31` and `x < 2^32`.
#[inline(always)]
pub fn mul_shoup(x: u32, w: u32, ws: u32, p: u32) -> u32 {
    let q = ((x as u64 * ws as u64) >> 32) as u32;
    let r = x.wrapping_mul(w).wrapping_sub(q.wrapping_mul(p));
    if r >= p {
        r - p
    } else {
        r
    }
}

#[inline(always)]
pub fn shoup(w: u32, p: u32) -> u32 {
    (((w as u64) << 32) / p as u64) as u32
}

/// One prime's transform tables.
pub struct Ntt {
    pub p: u32,
    fw: Vec<(u32, u32)>,
    iw: Vec<(u32, u32)>,
    ninv: (u32, u32),
}

impl Ntt {
    pub fn new(p: u64, psi: u64, n: usize) -> Result<Self> {
        if p >= 1 << 30 || !n.is_power_of_two() {
            return Err(Error::Params(format!("host NTT needs a prime below 2^30 (lazy butterflies keep 4p in 32 bits) and a power-of-two ring (p {p}, n {n})")));
        }
        let logn = n.trailing_zeros();
        let ipsi = invmod(psi, p);
        let (mut pw, mut ipw) = (vec![1u64; n], vec![1u64; n]);
        for i in 1..n {
            pw[i] = mulmod(pw[i - 1], psi, p);
            ipw[i] = mulmod(ipw[i - 1], ipsi, p);
        }
        let br = |i: usize| if logn == 0 { 0 } else { (i as u32).reverse_bits() as usize >> (32 - logn) };
        let p32 = p as u32;
        let tab = |v: &[u64]| (0..n).map(|i| (v[br(i)] as u32, shoup(v[br(i)] as u32, p32))).collect();
        let ni = invmod(n as u64 % p, p) as u32;
        Ok(Ntt { p: p32, fw: tab(&pw), iw: tab(&ipw), ninv: (ni, shoup(ni, p32)) })
    }

    /// Forward transform (Harvey's lazy butterflies: values kept below 4p, reduced once at the end).
    pub fn fwd(&self, a: &mut [u32]) {
        let (n, p) = (a.len(), self.p);
        let p2 = 2 * p;
        let (mut t, mut m) = (n, 1);
        while m < n {
            t /= 2;
            for i in 0..m {
                let (w, ws) = self.fw[m + i];
                let (lo, hi) = a[2 * i * t..2 * i * t + 2 * t].split_at_mut(t);
                for (u, v) in lo.iter_mut().zip(hi.iter_mut()) {
                    let x = (*u).min(u.wrapping_sub(p2));
                    let q = ((*v as u64 * ws as u64) >> 32) as u32;
                    let y = v.wrapping_mul(w).wrapping_sub(q.wrapping_mul(p));
                    *u = x + y;
                    *v = x + p2 - y;
                }
            }
            m *= 2;
        }
        a.iter_mut().for_each(|x| {
            let y = (*x).min(x.wrapping_sub(p2));
            *x = y.min(y.wrapping_sub(p));
        });
    }

    /// Inverse transform, N^-1 included (lazy butterflies, values below 2p between stages).
    pub fn inv(&self, a: &mut [u32]) {
        let (n, p) = (a.len(), self.p);
        let p2 = 2 * p;
        let (mut t, mut m) = (1, n);
        while m > 1 {
            let h = m / 2;
            for i in 0..h {
                let (w, ws) = self.iw[h + i];
                let (lo, hi) = a[2 * i * t..2 * i * t + 2 * t].split_at_mut(t);
                for (u, v) in lo.iter_mut().zip(hi.iter_mut()) {
                    let (x, y) = (*u, *v);
                    let s = x + y;
                    *u = s.min(s.wrapping_sub(p2));
                    let d = x + p2 - y;
                    let q = ((d as u64 * ws as u64) >> 32) as u32;
                    *v = d.wrapping_mul(w).wrapping_sub(q.wrapping_mul(p));
                }
            }
            t *= 2;
            m = h;
        }
        let (w, ws) = self.ninv;
        a.iter_mut().for_each(|x| *x = mul_shoup((*x).min(x.wrapping_sub(p)), w, ws, p));
    }
}

/// The chain on the host: one `Ntt` per prime id (Q then P) and the hybrid constants.
pub struct Ring {
    pub n: usize,
    pub nq: usize,
    pub primes: Vec<u64>,
    pub digits: Vec<Vec<usize>>,
    pub ntt: Vec<Ntt>,
    /// P mod q_i for every Q prime id.
    pub p_mod_q: Vec<u32>,
    /// P' mod q_i (P' = the first `Chain::p_sparse` special primes: the bottom-shape keys' gadget).
    pub ps_mod_q: Vec<u32>,
    /// Special primes in P (a key over fewer uses P').
    pub kp: usize,
}

impl Ring {
    pub fn new(chain: &Chain) -> Result<Self> {
        let primes = chain.primes();
        let roots = chain.roots();
        let ntt = primes.par_iter().zip(roots.par_iter()).map(|(&p, &r)| Ntt::new(p, r, chain.n)).collect::<Result<Vec<_>>>()?;
        let prod = |ps: &[u64]| chain.q.iter().map(|&q| ps.iter().fold(1u64, |a, &pj| mulmod(a, pj % q, q)) as u32).collect();
        let (p_mod_q, ps_mod_q) = (prod(&chain.p), prod(&chain.p[..chain.p_sparse]));
        Ok(Ring { n: chain.n, nq: chain.q.len(), primes, digits: chain.digits.clone(), ntt, p_mod_q, ps_mod_q, kp: chain.p.len() })
    }

    /// The gadget factor (P or P' mod q) of a key over `kp` special primes.
    pub fn gadget(&self, kp: usize) -> &[u32] {
        if kp < self.kp {
            &self.ps_mod_q
        } else {
            &self.p_mod_q
        }
    }

    /// Eval row of a signed small polynomial at prime id `id`.
    pub fn eval_row(&self, c: &[i64], id: usize) -> Vec<u32> {
        let p = self.primes[id] as i64;
        let mut v: Vec<u32> = c.iter().map(|&x| x.rem_euclid(p) as u32).collect();
        self.ntt[id].fwd(&mut v);
        v
    }

    /// Eval rows of a signed small polynomial at every prime id (Q then P).
    pub fn eval_all(&self, c: &[i64]) -> Vec<Vec<u32>> {
        (0..self.primes.len()).into_par_iter().map(|id| self.eval_row(c, id)).collect()
    }

    pub fn digit_of(&self, id: usize) -> Option<usize> {
        self.digits.iter().position(|d| d.contains(&id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pc_core::params::ChainSpec;

    /// Forward then inverse is the identity; the forward transform is the evaluation at the odd powers of psi in
    /// bit-reversed order (slot i holds a(psi^(2 brev(i) + 1))), the convention every eval-form table assumes.
    #[test]
    fn ntt_convention() {
        let chain = Chain::generate(&ChainSpec::n32()).unwrap();
        let n = 64;
        let p = chain.q[3];
        // a primitive 2n-th root from the chain's 2N-th root
        let psi = pc_core::modmul::powmod(chain.q_roots[3], (chain.n / n) as u64, p);
        let t = Ntt::new(p, psi, n).unwrap();
        let a: Vec<u32> = (0..n as u64).map(|i| ((i * 7919 + 3) % p) as u32).collect();
        let mut f = a.clone();
        t.fwd(&mut f);
        for i in 0..n {
            let br = (i as u32).reverse_bits() >> (32 - n.trailing_zeros());
            let x = pc_core::modmul::powmod(psi, 2 * br as u64 + 1, p);
            let mut acc = 0u64;
            let mut xp = 1u64;
            for &c in &a {
                acc = (acc + mulmod(c as u64, xp, p)) % p;
                xp = mulmod(xp, x, p);
            }
            assert_eq!(f[i] as u64, acc, "slot {i}");
        }
        t.inv(&mut f);
        assert_eq!(f, a);
    }
}
