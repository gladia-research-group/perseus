//! The CKKS side of the switch and the keys tying the two schemes together.
//!
//! Forward: a split ciphertext (RLWE of degree N mod q0 = p0 p1 under the CKKS secret s, phase m q0 / 4 at the
//! coefficients that carry values; at the device's StC-first split slot i sits at coefficient brev((i - 1) mod N/2),
//! its imaginary part N/2 further: `pc-check tfhe`) is rescaled by p1 (mod p0, phase m p0 / 4), key-switched from s to s''(X) =
//! s'(X^S) (s' ternary of dimension N / S), and its coefficient j read as an LWE of dimension N / S under s':
//! b = c0[j], a_k = +-c1[j - S k mod N]. Then the LWE key switch to the binary key and the bootstrap (pbs.rs).
//!
//! Return: the bootstrap's outputs (dimension n_br under s_br, mod q_br) are mod-switched to p0, b_j + <a_j, s_br> =
//! v_j + e_j + p0 I_j. With A_k(X) = sum_j a_jk X^j and B(X) = sum_j b_j X^j, the repack key K_kt = RLWE_s(p1 g_t
//! s_br[k]) mod q0 (exact gadget over p0) gives p1 B + sum_kt D_kt(A_k) K_kt with phase p1 (B + A s_br) = p1 (v + e) +
//! q0 I = p1 (v + e) mod q0: the p0 wraparound cancels and the key noise is not multiplied by p1. The result is the
//! split format again (phase v q0 / 2^out_cf), what the finish takes.
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

/// K_kt = RLWE_s(p1 g_t s_br[k]) mod p0 p1 (exact gadget over p0), NTT form per limb.
pub struct RepackKey {
    pub gadget: Gadget,
    /// [k][t][limb] -> (c0, c1)
    keys: Vec<Vec<Vec<(Vec<u64>, Vec<u64>)>>>,
}

impl RepackKey {
    pub fn generate(s: &[i64], s_br: &[i64], gadget: Gadget, ntts: &[Ntt], dg: &DiscreteGaussian, prng: &mut Prng) -> RepackKey {
        let n = ntts[0].n;
        let p1 = ntts[1].q as i128;
        let keys = s_br
            .iter()
            .map(|&sk| {
                (0..gadget.levels)
                    .map(|t| {
                        let mut m = vec![0i128; n];
                        m[0] = p1 * centered(gadget.g(t), gadget.q) as i128 * sk as i128;
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
                    .collect()
            })
            .collect();
        RepackKey { gadget, keys }
    }

    /// [k][t][limb] -> (c0, c1) in NTT form: the device layout's source.
    pub fn keys(&self) -> &[Vec<Vec<(Vec<u64>, Vec<u64>)>>] {
        &self.keys
    }

    /// `lwes[j]` (mod p0, dimension n_br; None = an exact zero) into one RLWE of degree N mod p0 p1 under s.
    pub fn repack(&self, lwes: &[Option<Lwe>], n_br: usize, ntts: &[Ntt]) -> RnsRlwe {
        let n = ntts[0].n;
        let p0 = self.gadget.q;
        let p1 = ntts[1].q;
        let l = self.gadget.levels;
        // B(X) and the digit polynomials D_kt(X) of A_k(X) = sum_j a_jk X^j
        let bpoly: Vec<i64> = (0..n).map(|j| lwes[j].as_ref().map_or(0, |c| centered(c.b, p0))).collect();
        let limbs: Vec<Rlwe> = ntts
            .iter()
            .enumerate()
            .map(|(li, t)| {
                let q = t.q;
                let parts: Vec<(Vec<u64>, Vec<u64>)> = (0..n_br)
                    .into_par_iter()
                    .map(|k| {
                        let mut d = vec![0i64; l];
                        let mut digits = vec![vec![0u64; n]; l];
                        for j in 0..n {
                            if let Some(c) = &lwes[j] {
                                self.gadget.decompose(c.a[k], &mut d);
                                for tt in 0..l {
                                    digits[tt][j] = from_signed(d[tt], q);
                                }
                            }
                        }
                        let (mut o0, mut o1) = (vec![0u64; n], vec![0u64; n]);
                        for (tt, mut dt) in digits.into_iter().enumerate() {
                            t.fwd(&mut dt);
                            let (k0, k1) = &self.keys[k][tt][li];
                            for i in 0..n {
                                o0[i] = (o0[i] + mulmod(dt[i], k0[i], q)) % q;
                                o1[i] = (o1[i] + mulmod(dt[i], k1[i], q)) % q;
                            }
                        }
                        (o0, o1)
                    })
                    .collect();
                let (mut o0, mut o1) = (vec![0u64; n], vec![0u64; n]);
                for (a, b) in parts {
                    add_into(&mut o0, &a, q);
                    add_into(&mut o1, &b, q);
                }
                t.inv(&mut o0);
                t.inv(&mut o1);
                let p1q = p1 % q;
                let pb: Vec<u64> = bpoly.iter().map(|&b| mulmod(from_signed(b, q), p1q, q)).collect();
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
        let dg_lwe = DiscreteGaussian::new(params.lwe_sigma_rel * p0 as f64)?;
        let ntt_p = vec![Ntt::new(p0, n)?, Ntt::new(p1, n)?];
        let s_sub = to_i64(sample_ternary(prng, n / params.rs_factor));
        let s_lwe: Vec<i64> = (0..params.n_lwe).map(|_| (prng.next_u32() & 1) as i64).collect();
        let s_br = to_i64(sample_ternary(prng, params.n_br));
        let rs = RingSwitchKey::generate(s, &s_sub, params.rs_factor, Gadget::new(p0, params.rs, false), &ntt_p[0], &dg, prng);
        let ksk = LweKsk::generate(&s_sub, &s_lwe, Gadget::new(p0, params.ks, false), &dg_lwe, prng);
        let ntt_br = Ntt::new(params.q_br, params.n_br)?;
        let bsk = BootstrapKey::generate(&s_lwe, &s_br, Gadget::new(params.q_br, params.pbs, false), ntt_br, &dg, prng);
        let repack = RepackKey::generate(s, &s_br, Gadget::new(p0, params.rp, true), &ntt_p, &dg, prng);
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

    /// The bootstrap's output scale for LUT values v: v q_br / 2^out_cf (it lands as v p0 / 2^out_cf).
    pub fn delta_br(&self) -> u64 {
        self.params.q_br >> self.params.out_cf
    }
}
