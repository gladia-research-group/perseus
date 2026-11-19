//! Host reference for checks (tests, `pc-check keys`): OpenFHE-order NTT per prime, the switch relation of a key
//! (`b + a s_new - [digit] (P mod q) s_old` = small error), and a plain hybrid key switch (approximate ModUp per
//! digit, inner product, approximate ModDown, OpenFHE `KeySwitchHYBRID` structure). Exact integer arithmetic, rayon,
//! no device: an independent twin of the device key switch for the key gates.
use crate::error::{shape, Result};
use crate::keys::KskShape;
use pc_core::modmul::{addmod, invmod, mulmod, submod};
use pc_core::params::Chain;
use rayon::prelude::*;

/// Negacyclic NTT of one prime in OpenFHE order (bit-reversed output, `fw[i] = psi^brev(i)`), as the device's.
pub struct HostNtt {
    pub p: u64,
    fw: Vec<u64>,
    iw: Vec<u64>,
    ninv: u64,
}

impl HostNtt {
    pub fn new(p: u64, psi: u64, n: usize) -> Self {
        let logn = n.trailing_zeros();
        let ipsi = invmod(psi, p);
        let (mut pw, mut ipw) = (vec![1u64; n], vec![1u64; n]);
        for i in 1..n {
            pw[i] = mulmod(pw[i - 1], psi, p);
            ipw[i] = mulmod(ipw[i - 1], ipsi, p);
        }
        let br = |i: usize| (i as u32).reverse_bits() as usize >> (32 - logn);
        HostNtt { p, fw: (0..n).map(|i| pw[br(i)]).collect(), iw: (0..n).map(|i| ipw[br(i)]).collect(), ninv: invmod(n as u64 % p, p) }
    }
    /// OpenFHE ForwardTransformToBitReverse.
    pub fn fwd(&self, a: &mut [u64]) {
        let (n, p) = (a.len(), self.p);
        let (mut t, mut m) = (n, 1);
        while m < n {
            t /= 2;
            for i in 0..m {
                let s = self.fw[m + i];
                for j in 2 * i * t..2 * i * t + t {
                    let u = a[j];
                    let v = mulmod(a[j + t], s, p);
                    a[j] = addmod(u, v, p);
                    a[j + t] = submod(u, v, p);
                }
            }
            m *= 2;
        }
    }
    /// OpenFHE InverseTransformFromBitReverse (N^-1 included).
    pub fn inv(&self, a: &mut [u64]) {
        let (n, p) = (a.len(), self.p);
        let (mut t, mut m) = (1, n);
        while m > 1 {
            let h = m / 2;
            let mut j1 = 0;
            for i in 0..h {
                let s = self.iw[h + i];
                for j in j1..j1 + t {
                    let u = a[j];
                    let v = a[j + t];
                    a[j] = addmod(u, v, p);
                    a[j + t] = mulmod(submod(u, v, p), s, p);
                }
                j1 += 2 * t;
            }
            t *= 2;
            m = h;
        }
        a.iter_mut().for_each(|x| *x = mulmod(*x, self.ninv, p));
    }
}

/// Host twin of a chain: per-prime NTTs and the hybrid constants.
pub struct HostChain {
    pub n: usize,
    pub nq: usize,
    pub kp: usize,
    pub primes: Vec<u64>,
    pub digits: Vec<Vec<usize>>,
    pub ntt: Vec<HostNtt>,
    /// P mod q_i (Q ids)
    pub p_mod_q: Vec<u64>,
}

impl HostChain {
    pub fn new(chain: &Chain) -> Self {
        let primes = chain.primes();
        let roots = chain.roots();
        let n = chain.n;
        let ntt = primes.par_iter().zip(roots.par_iter()).map(|(&p, &r)| HostNtt::new(p, r, n)).collect();
        let p_mod_q = chain.q.iter().map(|&q| chain.p.iter().fold(1u64, |a, &pj| mulmod(a, pj % q, q))).collect();
        HostChain { n, nq: chain.q.len(), kp: chain.p.len(), primes, digits: chain.digits.clone(), ntt, p_mod_q }
    }
    /// Signed coefficients -> eval rows of the listed prime ids.
    pub fn eval_rows(&self, c: &[i64], ids: &[usize]) -> Vec<Vec<u64>> {
        ids.par_iter()
            .map(|&id| {
                let p = self.primes[id];
                let mut v: Vec<u64> = c.iter().map(|&x| x.rem_euclid(p as i64) as u64).collect();
                self.ntt[id].fwd(&mut v);
                v
            })
            .collect()
    }
    fn digit_of(&self, id: usize) -> Option<usize> {
        self.digits.iter().position(|d| d.contains(&id))
    }

    /// Max |e| over every stored row of a key: e = INTT(b + a s_new - [id in digit d] (P mod q_id) s_old), where
    /// `s_new` / `s_old` are eval rows by prime id (`sq`: s_old squared) and `b`, `a` are `[digit][slot][N]`.
    /// Returns (max |e|, rows checked).
    pub fn relation_max_err(&self, sh: &KskShape, b: &[u64], a: &[u64], s_new: &[Vec<u64>], s_old: &[Vec<u64>], sq: bool) -> (i64, usize) {
        self.relation_max_err_shift(sh, b, a, s_new, s_old, sq, 0)
    }

    /// `relation_max_err` of a key with error shift `shift` (`keygen::err_shift`): > 0 gives max |e| / 2^shift over
    /// the CRT of the (one digit's) rows.
    #[allow(clippy::too_many_arguments)]
    pub fn relation_max_err_shift(&self, sh: &KskShape, b: &[u64], a: &[u64], s_new: &[Vec<u64>], s_old: &[Vec<u64>], sq: bool, shift: u32) -> (i64, usize) {
        let (n, rows) = (self.n, sh.rows());
        let ids = sh.row_ids();
        let erows: Vec<Vec<u64>> = (0..sh.ndig * rows)
            .into_par_iter()
            .map(|r| {
                let (d, id) = (r / rows, ids[r % rows]);
                let p = self.primes[id];
                let inq = id < self.nq && self.digit_of(id) == Some(d);
                let o = r * n;
                let mut e: Vec<u64> = (0..n)
                    .map(|i| {
                        let mut v = addmod(b[o + i], mulmod(a[o + i], s_new[id][i], p), p);
                        if inq {
                            let so = if sq { mulmod(s_old[id][i], s_old[id][i], p) } else { s_old[id][i] };
                            v = submod(v, mulmod(self.p_mod_q[id], so, p), p);
                        }
                        v
                    })
                    .collect();
                self.ntt[id].inv(&mut e);
                e
            })
            .collect();
        let mx = if shift > 0 {
            let pr: Vec<u64> = ids.iter().map(|&id| self.primes[id]).collect();
            pc_core::keys::wide_err_max(&pr, &erows, shift).ceil() as i64
        } else {
            erows
                .par_iter()
                .enumerate()
                .map(|(r, e)| {
                    let p = self.primes[ids[r % rows]];
                    e.iter().map(|&x| if x > p / 2 { p - x } else { x } as i64).max().unwrap_or(0)
                })
                .max()
                .unwrap_or(0)
        };
        (mx, sh.ndig * rows)
    }

    /// Approximate fast base conversion of coefficient rows `src` (ids `src_ids`) to prime `t` (centered residues).
    fn conv(&self, src: &[&[u64]], src_ids: &[usize], t: usize) -> Vec<u64> {
        let pt = self.primes[t];
        let bp: Vec<u64> = src_ids.iter().map(|&i| self.primes[i]).collect();
        // y_i = x_i * (B/b_i)^-1 mod b_i; out = sum y_i (B/b_i) mod t
        let pre: Vec<u64> = (0..bp.len()).map(|i| invmod((0..bp.len()).filter(|&k| k != i).fold(1, |h, k| mulmod(h, bp[k] % bp[i], bp[i])), bp[i])).collect();
        let hat: Vec<u64> = (0..bp.len()).map(|i| (0..bp.len()).filter(|&k| k != i).fold(1u64, |h, k| mulmod(h, bp[k] % pt, pt))).collect();
        (0..self.n)
            .map(|j| {
                let mut acc = 0u64;
                for i in 0..bp.len() {
                    // centered y: the conversion overshoot u (B multiples) is then centered on 0, so the ModDown
                    // rounding term u s has no mean (an unsigned u ~ kp/2 on average biases the slots near zeta = 1)
                    let y = mulmod(src[i][j], pre[i], bp[i]);
                    let ym = if y > bp[i] / 2 { submod(0, (bp[i] - y) % pt, pt) } else { y % pt };
                    acc = addmod(acc, mulmod(ym, hat[i], pt), pt);
                }
                acc
            })
            .collect()
    }

    /// Hybrid key switch of the eval rows `x` (Q ids `0..=top`) with key `(b, a)` (`[digit][slot][N]` of shape `sh`):
    /// returns eval rows `(u0, u1)` over Q ids `0..=top` with `u0 + u1 s_new = x s_old + small` (OpenFHE structure:
    /// approximate ModUp per digit into Q_top u P, inner product, approximate ModDown by P).
    pub fn key_switch(&self, x: &[Vec<u64>], top: usize, sh: &KskShape, b: &[u64], a: &[u64]) -> Result<(Vec<Vec<u64>>, Vec<Vec<u64>>)> {
        if !sh.covers(top) || x.len() != top + 1 {
            return Err(shape("hostref::key_switch", format!("top {top}, {} rows, key shape {sh:?}", x.len())));
        }
        let n = self.n;
        let pids: Vec<usize> = (self.nq..self.nq + self.kp).collect();
        let ext: Vec<usize> = (0..=top).chain(pids.iter().copied()).collect();
        let xc: Vec<Vec<u64>> = x
            .par_iter()
            .enumerate()
            .map(|(id, r)| {
                let mut v = r.clone();
                self.ntt[id].inv(&mut v);
                v
            })
            .collect();
        let rows = sh.rows();
        let mut s0: Vec<Vec<u64>> = vec![vec![0; n]; ext.len()];
        let mut s1: Vec<Vec<u64>> = vec![vec![0; n]; ext.len()];
        for (d, dg) in self.digits.iter().enumerate() {
            let src_ids: Vec<usize> = dg.iter().copied().filter(|&i| i <= top).collect();
            if src_ids.is_empty() {
                continue;
            }
            if d >= sh.ndig {
                return Err(shape("hostref::key_switch", format!("digit {d} needed at top {top}, key stores {}", sh.ndig)));
            }
            let src: Vec<&[u64]> = src_ids.iter().map(|&i| xc[i].as_slice()).collect();
            // D_d on every extended row (eval form): the digit's own rows are x itself
            let dd: Vec<Vec<u64>> = ext
                .par_iter()
                .map(|&t| {
                    if src_ids.contains(&t) {
                        x[t].clone()
                    } else {
                        let mut v = self.conv(&src, &src_ids, t);
                        self.ntt[t].fwd(&mut v);
                        v
                    }
                })
                .collect();
            s0.par_iter_mut().zip(s1.par_iter_mut()).zip(ext.par_iter()).zip(dd.par_iter()).for_each(|(((o0, o1), &t), dv)| {
                let p = self.primes[t];
                let slot = sh.slot_of(t).expect("covered row");
                let off = (d * rows + slot) * n;
                for i in 0..n {
                    o0[i] = addmod(o0[i], mulmod(dv[i], b[off + i], p), p);
                    o1[i] = addmod(o1[i], mulmod(dv[i], a[off + i], p), p);
                }
            });
        }
        let nq_ext = top + 1;
        let down = |s: &Vec<Vec<u64>>| -> Vec<Vec<u64>> {
            let pc: Vec<Vec<u64>> = (0..self.kp)
                .into_par_iter()
                .map(|j| {
                    let mut v = s[nq_ext + j].clone();
                    self.ntt[pids[j]].inv(&mut v);
                    v
                })
                .collect();
            let src: Vec<&[u64]> = pc.iter().map(|v| v.as_slice()).collect();
            (0..=top)
                .into_par_iter()
                .map(|t| {
                    let q = self.primes[t];
                    let mut c = self.conv(&src, &pids, t);
                    self.ntt[t].fwd(&mut c);
                    let pinv = invmod(self.p_mod_q[t], q);
                    (0..n).map(|i| mulmod(submod(s[t][i], c[i], q), pinv, q)).collect()
                })
                .collect()
        };
        Ok((down(&s0), down(&s1)))
    }
}

/// Centered max |x| over coefficient rows reconstructed from the bottom two primes (exact CRT, i128).
pub fn centered_crt2(r0: &[u64], r1: &[u64], q0: u64, q1: u64) -> Vec<i128> {
    let m = q0 as u128 * q1 as u128;
    let i0 = invmod(q1 % q0, q0) as u128;
    let i1 = invmod(q0 % q1, q1) as u128;
    r0.iter()
        .zip(r1)
        .map(|(&a, &b)| {
            let x = ((a as u128 * i0 % q0 as u128) * q1 as u128 + (b as u128 * i1 % q1 as u128) * q0 as u128) % m;
            if x > m / 2 {
                x as i128 - m as i128
            } else {
                x as i128
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ntt_roundtrip() {
        // p = 1 mod 2N, N = 16
        let n = 16usize;
        let p = 97u64; // 97 = 1 mod 32
        let psi = (2..p).find(|&g| pc_core::modmul::powmod(g, n as u64, p) == p - 1).unwrap();
        let t = HostNtt::new(p, psi, n);
        let x: Vec<u64> = (0..n as u64).map(|i| (i * 7 + 3) % p).collect();
        let mut y = x.clone();
        t.fwd(&mut y);
        // value check: y[brev(i)] = x(psi^(2i+1))
        for i in 0..n {
            let w = pc_core::modmul::powmod(psi, 2 * i as u64 + 1, p);
            let v = x.iter().rev().fold(0u64, |acc, &c| addmod(mulmod(acc, w, p), c, p));
            let bi = (i as u32).reverse_bits() as usize >> (32 - 4);
            assert_eq!(y[bi], v);
        }
        t.inv(&mut y);
        assert_eq!(x, y);
    }
}
