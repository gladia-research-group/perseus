//! The CKKS side of the switch and the keys tying the two schemes together.
//!
//! Forward: a split ciphertext (RLWE of degree N mod q0 = p0 p1 under the CKKS secret s, phase m q0 / 4 at the
//! coefficients that carry values; at the device's StC-first split slot i sits at coefficient brev((i - 1) mod N/2),
//! its imaginary part N/2 further: `pc-check tfhe`) is rescaled by p1 (mod p0, phase m p0 / 4), key-switched from s to s''(X) =
//! s'(X^S) (s' ternary of dimension N / S), and its coefficient j read as an LWE of dimension N / S under s':
//! b = c0[j], a_k = +-c1[j - S k mod N]. Then the LWE key switch to the binary key and the bootstrap (pbs.rs).
//!
//! Return: the bootstrap's outputs (dimension n_br under s_br, mod q_br) are mod-switched to p0, b_j + <a_j, s_br> =
//! v_j + e_j + p0 I_j, and repacked (`RepackKey`: a GEMM in R_{n_br} mod p0 P, a rescale by P, X^j P_j(X^S) into R_N,
//! one ring key switch to s mod q0) into the split format again (phase v q0 / 2^out_cf), what the finish takes.
use pc_core::keys::{sample_ternary, DiscreteGaussian, Prng};
use rayon::prelude::*;

use crate::gadget::Gadget;
use crate::lwe::{Lwe, LweKsk};
use crate::modq::{centered, from_signed, invmod, mulmod, submod, Ntt};
use crate::params::Params;
use crate::pbs::BootstrapKey;
use crate::rlwe::{add_into, Rlwe, RnsRlwe};
use crate::Result;

fn to_i64(v: Vec<i8>) -> Vec<i64> {
    v.into_iter().map(|x| x as i64).collect()
}

/// Exact rescale of an RLWE mod p0 p1 by p1: c -> (c_p0 - [c_p1]) p1^-1 mod p0 (the CKKS rescale).
pub fn rescale_p1(ct: &RnsRlwe, p0: u64, p1: u64) -> Rlwe {
    let inv = invmod(p1 % p0, p0);
    let f = |x0: &[u64], x1: &[u64]| -> Vec<u64> {
        x0.iter().zip(x1).map(|(&a, &b)| mulmod(submod(a, from_signed(centered(b, p1), p0), p0), inv, p0)).collect()
    };
    Rlwe { c0: f(&ct.limbs[0].c0, &ct.limbs[1].c0), c1: f(&ct.limbs[0].c1, &ct.limbs[1].c1) }
}

/// RLWE_{s''}(g_t s), s'' = s'(X^S), ring N mod p0.
pub struct RingSwitchKey {
    pub gadget: Gadget,
    pub factor: usize,
    /// [t] in NTT form
    keys: Vec<(Vec<u64>, Vec<u64>)>,
}

impl RingSwitchKey {
    pub fn generate(s: &[i64], s_sub: &[i64], factor: usize, gadget: Gadget, ntt: &Ntt, dg: &DiscreteGaussian, prng: &mut Prng) -> RingSwitchKey {
        let q = ntt.q;
        let mut emb = vec![0i64; ntt.n];
        for (k, &v) in s_sub.iter().enumerate() {
            emb[factor * k] = v;
        }
        let keys = (0..gadget.levels)
            .map(|t| {
                let m: Vec<u64> = s.iter().map(|&x| mulmod(from_signed(x, q), gadget.g(t), q)).collect();
                let mut k = Rlwe::encrypt(&m, &emb, ntt, dg, prng);
                ntt.fwd(&mut k.c0);
                ntt.fwd(&mut k.c1);
                (k.c0, k.c1)
            })
            .collect();
        RingSwitchKey { gadget, factor, keys }
    }

    /// [t] -> (c0, c1) in NTT form: the device layout's source.
    pub fn keys(&self) -> &[(Vec<u64>, Vec<u64>)] {
        &self.keys
    }

    /// ct under s -> ct under s'(X^S): (c0 + sum_t D_t(c1) k0_t, sum_t D_t(c1) k1_t).
    pub fn apply(&self, ct: &Rlwe, ntt: &Ntt) -> Rlwe {
        let (n, q, l) = (ntt.n, ntt.q, self.gadget.levels);
        let mut digits = vec![vec![0u64; n]; l];
        let mut d = vec![0i64; l];
        for (i, &x) in ct.c1.iter().enumerate() {
            self.gadget.decompose(x, &mut d);
            for t in 0..l {
                digits[t][i] = from_signed(d[t], q);
            }
        }
        let (mut o0, mut o1) = (vec![0u64; n], vec![0u64; n]);
        for (t, mut dt) in digits.into_iter().enumerate() {
            ntt.fwd(&mut dt);
            for i in 0..n {
                o0[i] = (o0[i] + mulmod(dt[i], self.keys[t].0[i], q)) % q;
                o1[i] = (o1[i] + mulmod(dt[i], self.keys[t].1[i], q)) % q;
            }
        }
        ntt.inv(&mut o0);
        ntt.inv(&mut o1);
        add_into(&mut o0, &ct.c0, q);
        Rlwe { c0: o0, c1: o1 }
    }

    /// Coefficient j of a ct under s'(X^S) as an LWE of dimension N / S under s'.
    pub fn extract(&self, ct: &Rlwe, j: usize, q: u64) -> Lwe {
        let n = ct.c0.len();
        let a = (0..n / self.factor)
            .map(|k| {
                let sk = self.factor * k;
                if sk <= j {
                    ct.c1[j - sk]
                } else {
                    let v = ct.c1[n + j - sk];
                    if v == 0 {
                        0
                    } else {
                        q - v
                    }
                }
            })
            .collect();
        Lwe { a, b: ct.c0[j] }
    }
}

/// The return path's keys, around an auxiliary prime P (p0 P <= 2^54: the HE-standard bound at degree n_br):
/// - columns K_kt = RLWE_{s_x}(P g_t s_br[k]) mod p0 P in R_{n_br} under a fresh ternary s_x, coefficient form;
/// - the ring key R_t = RLWE_s(p1 g_t s_x(X^S)) mod q0 = p0 p1 in R_N, NTT form per limb.
/// An LWE (a, b) mod p0 under s_br packs into P_j = (P b, 0) + sum_kt d_t(a_k) K_kt (exact gadget over p0): phase
/// P (b + <a, s_br>) + small = P (v + e) mod p0 P at coefficient 0 and only noise elsewhere (no junk: every column
/// encrypts a scalar), the p0 wraparound cancels against P. A rescale by P leaves v + e mod p0 under s_x; X^j P_j(X^S)
/// puts it at coefficient j (S = N / n_br), the values sum into one RLWE mod p0 under s_x(X^S), and the ring key
/// switches that to s mod q0: p1 c0 + sum_t d_t(c1) R_t, phase p1 (v + e) (the wraparound cancels again, the key
/// noise is not multiplied by p1). The packing is a GEMM of the digits [value][k t] by the columns [k t][part][limb][i]
/// (no NTT), the ring switch one key switch at N.
pub struct RepackKey {
    pub gadget: Gadget,
    pub aux: u64,
    /// [k][t][limb p0 | P] -> (c0, c1), coefficient form, degree n_br
    cols: Vec<Vec<Vec<(Vec<u64>, Vec<u64>)>>>,
    /// [t][limb p0 | p1] -> (c0, c1), NTT form, degree N
    ring: Vec<Vec<(Vec<u64>, Vec<u64>)>>,
}

impl RepackKey {
    /// `dg_cols`: the columns' noise (divided by P in the rescale: it can be large), `dg`: the ring key's.
    pub fn generate(s: &[i64], s_br: &[i64], gadget: Gadget, aux: u64, ntts: &[Ntt], dg_cols: &DiscreteGaussian, dg: &DiscreteGaussian, prng: &mut Prng) -> Result<RepackKey> {
        let (n, nb) = (ntts[0].n, s_br.len());
        let p0 = gadget.q;
        if (p0 as u128) * (aux as u128) > 1 << 54 {
            return Err(crate::Error::Params(format!("repack: p0 P = {p0} x {aux} exceeds 2^54 (degree {nb} security)")));
        }
        let small = [Ntt::new(p0, nb)?, Ntt::new(aux, nb)?];
        let s_x = to_i64(sample_ternary(prng, nb));
        let g = |t: usize| centered(gadget.g(t), p0) as i128;
        let coef = |ct: RnsRlwe| ct.limbs.into_iter().map(|r| (r.c0, r.c1)).collect::<Vec<_>>();
        let cols = s_br
            .iter()
            .map(|&sk| {
                (0..gadget.levels)
                    .map(|t| {
                        let mut m = vec![0i128; nb];
                        m[0] = aux as i128 * g(t) * sk as i128;
                        coef(RnsRlwe::encrypt(&m, &s_x, &small, dg_cols, prng))
                    })
                    .collect()
            })
            .collect();
        let p1 = ntts[1].q as i128;
        let ring = (0..gadget.levels)
            .map(|t| {
                let mut m = vec![0i128; n];
                for (k, &v) in s_x.iter().enumerate() {
                    m[k * (n / nb)] = p1 * g(t) * v as i128;
                }
                let ct = RnsRlwe::encrypt(&m, s, ntts, dg, prng);
                ct.limbs
                    .into_iter()
                    .zip(ntts)
                    .map(|(mut r, t)| {
                        t.fwd(&mut r.c0);
                        t.fwd(&mut r.c1);
                        (r.c0, r.c1)
                    })
                    .collect()
            })
            .collect();
        Ok(RepackKey { gadget, aux, cols, ring })
    }

    /// [k][t][limb p0 | P] -> (c0, c1), coefficient form: the device layout's source.
    pub fn cols(&self) -> &[Vec<Vec<(Vec<u64>, Vec<u64>)>>] {
        &self.cols
    }

    /// [t][limb p0 | p1] -> (c0, c1), NTT form.
    pub fn ring(&self) -> &[Vec<(Vec<u64>, Vec<u64>)>] {
        &self.ring
    }

    /// One LWE (mod p0, dimension n_br) packed and rescaled: (c0, c1) mod p0 in R_{n_br} under s_x.
    pub fn pack(&self, c: &Lwe) -> (Vec<u64>, Vec<u64>) {
        let (p0, aux, l) = (self.gadget.q, self.aux, self.gadget.levels);
        let nb = c.a.len();
        let mut digits = vec![0i64; nb * l];
        for (k, &a) in c.a.iter().enumerate() {
            self.gadget.decompose(a, &mut digits[k * l..(k + 1) * l]);
        }
        // [limb][part] sums < 4096 2^15 2^28 = 2^55: exact in i64
        let mut acc = vec![vec![vec![0i64; nb]; 2]; 2];
        for k in 0..nb {
            for t in 0..l {
                let d = digits[k * l + t];
                if d == 0 {
                    continue;
                }
                for (li, (k0, k1)) in self.cols[k][t].iter().enumerate() {
                    for i in 0..nb {
                        acc[li][0][i] += d * k0[i] as i64;
                        acc[li][1][i] += d * k1[i] as i64;
                    }
                }
            }
        }
        let b = centered(c.b, p0) as i128;
        acc[0][0][0] += (aux as i128 * b).rem_euclid(p0 as i128) as i64;
        let inv = invmod(aux % p0, p0);
        let resc = |x0: i64, x1: i64| -> u64 {
            let r0 = x0.rem_euclid(p0 as i64) as u64;
            let r1 = centered(x1.rem_euclid(aux as i64) as u64, aux);
            mulmod(submod(r0, from_signed(r1, p0), p0), inv, p0)
        };
        let part = |pt: usize| (0..nb).map(|i| resc(acc[0][pt][i], acc[1][pt][i])).collect();
        (part(0), part(1))
    }

    /// `lwes[j]` (mod p0, dimension n_br; None = an exact zero) into one RLWE of degree N mod p0 p1 under s.
    pub fn repack(&self, lwes: &[Option<Lwe>], n_br: usize, ntts: &[Ntt]) -> RnsRlwe {
        let (n, p0, l) = (ntts[0].n, self.gadget.q, self.gadget.levels);
        let sf = n / n_br;
        let packed: Vec<(usize, (Vec<u64>, Vec<u64>))> =
            lwes.par_iter().enumerate().filter_map(|(j, c)| c.as_ref().map(|c| (j, self.pack(c)))).collect();
        // X^j P_j(X^S), summed: under s_x(X^S), mod p0
        let (mut c0, mut c1) = (vec![0u64; n], vec![0u64; n]);
        for (j, (x0, x1)) in &packed {
            for i in 0..n_br {
                let y = j + sf * i;
                if y < n {
                    c0[y] = crate::modq::addmod(c0[y], x0[i], p0);
                    c1[y] = crate::modq::addmod(c1[y], x1[i], p0);
                } else {
                    c0[y - n] = submod(c0[y - n], x0[i], p0);
                    c1[y - n] = submod(c1[y - n], x1[i], p0);
                }
            }
        }
        // ring switch s_x(X^S) -> s mod q0: p1 c0 + sum_t d_t(c1) R_t
        let mut d = vec![0i64; l];
        let mut digits = vec![vec![0i64; n]; l];
        for (i, &x) in c1.iter().enumerate() {
            self.gadget.decompose(x, &mut d);
            for t in 0..l {
                digits[t][i] = d[t];
            }
        }
        let p1 = ntts[1].q;
        let limbs = ntts
            .iter()
            .enumerate()
            .map(|(li, t)| {
                let q = t.q;
                let (mut o0, mut o1) = (vec![0u64; n], vec![0u64; n]);
                for (tt, dt) in digits.iter().enumerate() {
                    let mut dq: Vec<u64> = dt.iter().map(|&v| from_signed(v, q)).collect();
                    t.fwd(&mut dq);
                    let (k0, k1) = &self.ring[tt][li];
                    for i in 0..n {
                        o0[i] = (o0[i] + mulmod(dq[i], k0[i], q)) % q;
                        o1[i] = (o1[i] + mulmod(dq[i], k1[i], q)) % q;
                    }
                }
                t.inv(&mut o0);
                t.inv(&mut o1);
                let p1q = p1 % q;
                let pb: Vec<u64> = c0.iter().map(|&b| mulmod(from_signed(centered(b, p0), q), p1q, q)).collect();
                add_into(&mut o0, &pb, q);
                Rlwe { c0: o0, c1: o1 }
            })
            .collect();
        RnsRlwe { limbs }
    }
}

/// Every key of the switch, from the CKKS secret s (ternary, degree N) and the chain's (p0, p1).
pub struct Switcher {
    pub params: Params,
    pub n: usize,
    pub p0: u64,
    pub p1: u64,
    pub ntt_p: Vec<Ntt>,
    pub rs: RingSwitchKey,
    pub ksk: LweKsk,
    pub bsk: BootstrapKey,
    pub repack: RepackKey,
}

impl Switcher {
    pub fn generate(params: Params, s: &[i64], p0: u64, p1: u64, prng: &mut Prng) -> Result<Switcher> {
        let n = s.len();
        let dg = DiscreteGaussian::new(params.key_sigma)?;
        let (dg_bsk, dg_rp) = (DiscreteGaussian::new(params.bsk_sigma)?, DiscreteGaussian::new(params.rp_sigma)?);
        let dg_lwe = DiscreteGaussian::new(params.lwe_sigma_rel * p0 as f64)?;
        let ntt_p = vec![Ntt::new(p0, n)?, Ntt::new(p1, n)?];
        let s_sub = to_i64(sample_ternary(prng, n / params.rs_factor));
        let s_lwe: Vec<i64> = (0..params.n_lwe).map(|_| (prng.next_u32() & 1) as i64).collect();
        let s_br = to_i64(sample_ternary(prng, params.n_br));
        let rs = RingSwitchKey::generate(s, &s_sub, params.rs_factor, Gadget::new(p0, params.rs, false), &ntt_p[0], &dg, prng);
        let ksk = LweKsk::generate(&s_sub, &s_lwe, Gadget::new(p0, params.ks, false), &dg_lwe, prng);
        let rns = crate::pbs::Rns::new(params.br_primes.0, params.br_primes.1, params.n_br)?;
        let bsk = BootstrapKey::generate(&s_lwe, &s_br, Gadget::new(params.q_br, params.pbs, false), rns, params.pbs_group, &dg_bsk, prng);
        let repack = RepackKey::generate(s, &s_br, Gadget::new(p0, params.rp, true), params.rp_aux, &ntt_p, &dg_rp, &dg, prng)?;
        Ok(Switcher { params, n, p0, p1, ntt_p, rs, ksk, bsk, repack })
    }

    /// The LUT `tv` (+ `offset`, mod q_br) on the coefficients `valid` of a split ciphertext (phase m q0 / 4 there);
    /// every other coefficient comes back as an exact zero. Returns the split ciphertext of the outputs (mod p0 p1).
    pub fn apply(&self, ct: &RnsRlwe, valid: &[usize], tv: &[u64], offset: u64) -> RnsRlwe {
        let p0 = self.p0;
        let low = rescale_p1(ct, p0, self.p1);
        let sw = self.rs.apply(&low, &self.ntt_p[0]);
        let lwes: Vec<Lwe> = valid.par_iter().map(|&j| self.ksk.apply(&self.rs.extract(&sw, j, p0))).collect();
        let outs = self.bsk.bootstrap_many(tv, offset, &lwes, p0);
        let mut slots: Vec<Option<Lwe>> = vec![None; self.n];
        for (&j, o) in valid.iter().zip(outs) {
            slots[j] = Some(o.mod_switch(self.params.q_br, p0));
        }
        self.repack.repack(&slots, self.params.n_br, &self.ntt_p)
    }

    /// The bootstrap's output phase for the LUT value 1 (`pbs::warp`: the finish reads it as 2^-out_cf); `sign01`'s delta.
    pub fn delta_br(&self) -> u64 {
        self.lut_value(1.0)
    }

    /// The test-vector entry (mod q_br) of the LUT value v: q_br warp(v).
    pub fn lut_value(&self, v: f64) -> u64 {
        let x = (self.params.q_br as f64 * crate::pbs::warp(v, self.params.out_cf)).round() as i64;
        x.rem_euclid(self.params.q_br as i64) as u64
    }
}
