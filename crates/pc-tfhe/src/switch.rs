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
use std::sync::OnceLock;

use pc_core::keys::{mask_expand_row64, sample_ternary, tfhe_tag, DiscreteGaussian, MaskSeed, Prng};
use rayon::prelude::*;

use crate::gadget::Gadget;
use crate::lwe::{Lwe, LweKsk};
use crate::modq::{addmod, centered, from_signed, invmod, mulmod, submod, Ntt};
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

/// RLWE_{s''}(g_t s), s'' = s'(X^S), ring N mod p0; c1 stored or seeded (level t = the mask row (seed, t) of length
/// N in NTT form).
pub struct RingSwitchKey {
    pub gadget: Gadget,
    pub factor: usize,
    /// [t] c0, NTT form
    c0: Vec<Vec<u64>>,
    c1: Option<Vec<Vec<u64>>>,
    pub seed: Option<MaskSeed>,
    cache: OnceLock<Vec<(Vec<u64>, Vec<u64>)>>,
}

impl RingSwitchKey {
    /// `seeded`: c1 = the mask row (tag "TRSK") and c0 = NTT(m + e) - c1 NTT(s''); else c1 uniform from `prng`.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(s: &[i64], s_sub: &[i64], factor: usize, gadget: Gadget, ntt: &Ntt, dg: &DiscreteGaussian, seeded: bool, prng: &mut Prng) -> RingSwitchKey {
        let q = ntt.q;
        let mut emb = vec![0i64; ntt.n];
        for (k, &v) in s_sub.iter().enumerate() {
            emb[factor * k] = v;
        }
        let msg = |t: usize| -> Vec<u64> { s.iter().map(|&x| mulmod(from_signed(x, q), gadget.g(t), q)).collect() };
        if seeded {
            let seed = MaskSeed { seed: prng.ksk_seed(), tag: tfhe_tag::RING_SWITCH };
            let mut e_ntt = crate::rlwe::signed_poly(&emb, q);
            ntt.fwd(&mut e_ntt);
            let c0 = (0..gadget.levels)
                .map(|t| {
                    let e = dg.sample_vec(prng, ntt.n);
                    let mut x: Vec<u64> = msg(t).iter().zip(&e).map(|(&m, &v)| addmod(m, from_signed(v, q), q)).collect();
                    ntt.fwd(&mut x);
                    let mut m = vec![0u64; ntt.n];
                    mask_expand_row64(&seed, t as u32, q, &mut m);
                    x.iter().zip(&m).zip(&e_ntt).map(|((&a, &c), &z)| submod(a, mulmod(c, z, q), q)).collect()
                })
                .collect();
            return RingSwitchKey { gadget, factor, c0, c1: None, seed: Some(seed), cache: OnceLock::new() };
        }
        let (mut c0, mut c1) = (Vec::new(), Vec::new());
        for t in 0..gadget.levels {
            let mut k = Rlwe::encrypt(&msg(t), &emb, ntt, dg, prng);
            ntt.fwd(&mut k.c0);
            ntt.fwd(&mut k.c1);
            c0.push(k.c0);
            c1.push(k.c1);
        }
        RingSwitchKey { gadget, factor, c0, c1: Some(c1), seed: None, cache: OnceLock::new() }
    }

    /// [t] -> (c0, c1) in NTT form: the device layout's source (materialized once).
    pub fn keys(&self) -> &[(Vec<u64>, Vec<u64>)] {
        self.cache.get_or_init(|| {
            let q = self.gadget.q;
            (0..self.c0.len())
                .map(|t| {
                    let c1 = match (&self.c1, &self.seed) {
                        (Some(c1), _) => c1[t].clone(),
                        (None, Some(seed)) => {
                            let mut m = vec![0u64; self.c0[t].len()];
                            mask_expand_row64(seed, t as u32, q, &mut m);
                            m
                        }
                        (None, None) => unreachable!("a ring-switch key holds its masks or their seed"),
                    };
                    (self.c0[t].clone(), c1)
                })
                .collect()
        })
    }

    /// Bytes as stored / shipped.
    pub fn stored_bytes(&self) -> usize {
        let w: usize = self.c0.iter().map(|r| r.len()).sum();
        4 * if self.c1.is_some() { 2 * w } else { w } + if self.seed.is_some() { 36 } else { 0 }
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
                o0[i] = (o0[i] + mulmod(dt[i], self.keys()[t].0[i], q)) % q;
                o1[i] = (o1[i] + mulmod(dt[i], self.keys()[t].1[i], q)) % q;
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
    /// [k][t][limb p0 | P] c0, coefficient form, degree n_br
    cols_c0: Vec<Vec<[Vec<u64>; 2]>>,
    /// [t][limb p0 | p1] c0, NTT form, degree N
    ring_c0: Vec<[Vec<u64>; 2]>,
    /// the stored c1s (cols, ring), or the seeds of their masks (cols: lane (k levels + t) 2 + limb, length n_br,
    /// coefficient form; ring: lane 2 t + limb, length N, NTT form)
    c1: Option<(Vec<Vec<[Vec<u64>; 2]>>, Vec<[Vec<u64>; 2]>)>,
    pub seeds: Option<(MaskSeed, MaskSeed)>,
    /// the limb moduli: cols (p0, P), ring (p0, p1)
    mods: ([u64; 2], [u64; 2]),
    cols_cache: OnceLock<Vec<Vec<Vec<(Vec<u64>, Vec<u64>)>>>>,
    ring_cache: OnceLock<Vec<Vec<(Vec<u64>, Vec<u64>)>>>,
}

impl RepackKey {
    /// `dg_cols`: the columns' noise (divided by P in the rescale: it can be large), `dg`: the ring key's. `seeded`:
    /// the c1s are mask rows (tags "TRPC", "TRPR"; c0 = m + e - c1 s), else uniform from `prng`.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(s: &[i64], s_br: &[i64], gadget: Gadget, aux: u64, ntts: &[Ntt], dg_cols: &DiscreteGaussian, dg: &DiscreteGaussian, seeded: bool, prng: &mut Prng) -> Result<RepackKey> {
        let (n, nb) = (ntts[0].n, s_br.len());
        let p0 = gadget.q;
        if (p0 as u128) * (aux as u128) > 1 << 54 {
            return Err(crate::Error::Params(format!("repack: p0 P = {p0} x {aux} exceeds 2^54 (degree {nb} security)")));
        }
        let small = [Ntt::new(p0, nb)?, Ntt::new(aux, nb)?];
        let s_x = to_i64(sample_ternary(prng, nb));
        let g = |t: usize| centered(gadget.g(t), p0) as i128;
        let lv = gadget.levels;
        let p1 = ntts[1].q as i128;
        let mods = ([p0, aux], [ntts[0].q, ntts[1].q]);
        let col_msg = |sk: i64, t: usize| {
            let mut m = vec![0i128; nb];
            m[0] = aux as i128 * g(t) * sk as i128;
            m
        };
        let ring_msg = |t: usize| {
            let mut m = vec![0i128; n];
            for (k, &v) in s_x.iter().enumerate() {
                m[k * (n / nb)] = p1 * g(t) * v as i128;
            }
            m
        };
        if seeded {
            let seeds = (MaskSeed { seed: prng.ksk_seed(), tag: tfhe_tag::REPACK_COLS }, MaskSeed { seed: prng.ksk_seed(), tag: tfhe_tag::REPACK_RING });
            let errs: Vec<Vec<i64>> = (0..nb * lv).map(|_| dg_cols.sample_vec(prng, nb)).collect();
            let sx = [0, 1].map(|li| crate::rlwe::signed_poly(&s_x, small[li].q));
            let cols_c0 = s_br
                .par_iter()
                .enumerate()
                .map(|(k, &sk)| {
                    (0..lv)
                        .map(|t| {
                            let m = col_msg(sk, t);
                            [0, 1].map(|li| {
                                let q = small[li].q;
                                let mut c1 = vec![0u64; nb];
                                mask_expand_row64(&seeds.0, ((k * lv + t) * 2 + li) as u32, q, &mut c1);
                                let c1s = small[li].mul(&c1, &sx[li]);
                                (0..nb).map(|i| submod(addmod(m[i].rem_euclid(q as i128) as u64, from_signed(errs[k * lv + t][i], q), q), c1s[i], q)).collect()
                            })
                        })
                        .collect()
                })
                .collect();
            let ring_c0 = (0..lv)
                .map(|t| {
                    let m = ring_msg(t);
                    let e = dg.sample_vec(prng, n);
                    [0, 1].map(|li| {
                        let nt = &ntts[li];
                        let q = nt.q;
                        let mut sn = crate::rlwe::signed_poly(s, q);
                        nt.fwd(&mut sn);
                        let mut x: Vec<u64> = (0..n).map(|i| addmod(m[i].rem_euclid(q as i128) as u64, from_signed(e[i], q), q)).collect();
                        nt.fwd(&mut x);
                        let mut c1 = vec![0u64; n];
                        mask_expand_row64(&seeds.1, (2 * t + li) as u32, q, &mut c1);
                        (0..n).map(|i| submod(x[i], mulmod(c1[i], sn[i], q), q)).collect()
                    })
                })
                .collect();
            return Ok(RepackKey { gadget, aux, cols_c0, ring_c0, c1: None, seeds: Some(seeds), mods, cols_cache: OnceLock::new(), ring_cache: OnceLock::new() });
        }
        let pair = |ct: RnsRlwe| {
            let mut it = ct.limbs.into_iter();
            let (a, b) = (it.next().unwrap(), it.next().unwrap());
            ([a.c0, b.c0], [a.c1, b.c1])
        };
        let (mut cols_c0, mut cols_c1) = (Vec::new(), Vec::new());
        for &sk in s_br {
            let (mut r0, mut r1) = (Vec::new(), Vec::new());
            for t in 0..lv {
                let (a, b) = pair(RnsRlwe::encrypt(&col_msg(sk, t), &s_x, &small, dg_cols, prng));
                r0.push(a);
                r1.push(b);
            }
            cols_c0.push(r0);
            cols_c1.push(r1);
        }
        let (mut ring_c0, mut ring_c1) = (Vec::new(), Vec::new());
        for t in 0..lv {
            let mut ct = RnsRlwe::encrypt(&ring_msg(t), s, ntts, dg, prng);
            for (r, nt) in ct.limbs.iter_mut().zip(ntts) {
                nt.fwd(&mut r.c0);
                nt.fwd(&mut r.c1);
            }
            let (a, b) = pair(ct);
            ring_c0.push(a);
            ring_c1.push(b);
        }
        Ok(RepackKey { gadget, aux, cols_c0, ring_c0, c1: Some((cols_c1, ring_c1)), seeds: None, mods, cols_cache: OnceLock::new(), ring_cache: OnceLock::new() })
    }

    /// [k][t][limb p0 | P] -> (c0, c1), coefficient form: the device layout's source (materialized once).
    pub fn cols(&self) -> &[Vec<Vec<(Vec<u64>, Vec<u64>)>>] {
        self.cols_cache.get_or_init(|| {
            let lv = self.gadget.levels;
            self.cols_c0
                .par_iter()
                .enumerate()
                .map(|(k, kt)| {
                    kt.iter()
                        .enumerate()
                        .map(|(t, limbs)| {
                            (0..2)
                                .map(|li| {
                                    let c1 = match (&self.c1, &self.seeds) {
                                        (Some((c, _)), _) => c[k][t][li].clone(),
                                        (None, Some((sd, _))) => {
                                            let mut m = vec![0u64; limbs[li].len()];
                                            mask_expand_row64(sd, ((k * lv + t) * 2 + li) as u32, self.mods.0[li], &mut m);
                                            m
                                        }
                                        (None, None) => unreachable!("a repack key holds its masks or their seeds"),
                                    };
                                    (limbs[li].clone(), c1)
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect()
        })
    }

    /// [t][limb p0 | p1] -> (c0, c1), NTT form (materialized once).
    pub fn ring(&self) -> &[Vec<(Vec<u64>, Vec<u64>)>] {
        self.ring_cache.get_or_init(|| {
            self.ring_c0
                .iter()
                .enumerate()
                .map(|(t, limbs)| {
                    (0..2)
                        .map(|li| {
                            let c1 = match (&self.c1, &self.seeds) {
                                (Some((_, r)), _) => r[t][li].clone(),
                                (None, Some((_, sd))) => {
                                    let mut m = vec![0u64; limbs[li].len()];
                                    mask_expand_row64(sd, (2 * t + li) as u32, self.mods.1[li], &mut m);
                                    m
                                }
                                (None, None) => unreachable!("a repack key holds its masks or their seeds"),
                            };
                            (limbs[li].clone(), c1)
                        })
                        .collect()
                })
                .collect()
        })
    }

    /// Bytes as stored / shipped (cols, ring).
    pub fn stored_bytes(&self) -> (usize, usize) {
        let wc: usize = self.cols_c0.iter().flatten().flatten().map(|r| r.len()).sum();
        let wr: usize = self.ring_c0.iter().flatten().map(|r| r.len()).sum();
        match self.seeds {
            Some(_) => (4 * wc + 36, 4 * wr + 36),
            None => (8 * wc, 8 * wr),
        }
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
                for (li, (k0, k1)) in self.cols()[k][t].iter().enumerate() {
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
                    let (k0, k1) = &self.ring()[tt][li];
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
    /// the latency key (Params::lat_group), same secrets and gadget
    pub bsk_lat: Option<BootstrapKey>,
    /// the mid-batch latency key (Params::lat_group2)
    pub bsk_lat2: Option<BootstrapKey>,
    pub repack: RepackKey,
    /// Params::chain: the native chains' key switch
    pub chain: Option<crate::chain::ChainKey>,
    /// Params::pack: the packing key (encrypted lookup tables)
    pub pack: Option<crate::pack::PackingKey>,
}

impl Switcher {
    pub fn generate(params: Params, s: &[i64], p0: u64, p1: u64, prng: &mut Prng) -> Result<Switcher> {
        Ok(Self::generate_keep(params, s, p0, p1, prng)?.0)
    }

    /// `generate`, also returning the TFHE secrets (s_lwe, s_br) it sampled: for tests that encrypt and decrypt
    /// native LWEs (pc-check tfhe-bitwise).
    pub fn generate_keep(params: Params, s: &[i64], p0: u64, p1: u64, prng: &mut Prng) -> Result<(Switcher, Vec<i64>, Vec<i64>)> {
        let n = s.len();
        let dg = DiscreteGaussian::new(params.key_sigma)?;
        let (dg_bsk, dg_rp) = (DiscreteGaussian::new(params.bsk_sigma)?, DiscreteGaussian::new(params.rp_sigma)?);
        let dg_lwe = DiscreteGaussian::new(params.lwe_sigma_rel * p0 as f64)?;
        let ntt_p = vec![Ntt::new(p0, n)?, Ntt::new(p1, n)?];
        let s_sub = to_i64(sample_ternary(prng, n / params.rs_factor));
        let s_lwe: Vec<i64> = (0..params.n_lwe).map(|_| (prng.next_u32() & 1) as i64).collect();
        let s_br = to_i64(sample_ternary(prng, params.n_br));
        let rs = RingSwitchKey::generate(s, &s_sub, params.rs_factor, Gadget::new(p0, params.rs, false), &ntt_p[0], &dg, params.seeded, prng);
        let gk = Gadget::new(p0, params.ks, false);
        let ksk = if params.seeded { LweKsk::generate_seeded(&s_sub, &s_lwe, gk, pc_core::keys::tfhe_tag::LWE_KSK, &dg_lwe, prng) } else { LweKsk::generate(&s_sub, &s_lwe, gk, &dg_lwe, prng) };
        let rns = crate::pbs::Rns::new(params.br_primes.0, params.br_primes.1, params.n_br)?;
        let gen_bsk = |group: usize, prng: &mut Prng| {
            let g = Gadget::new(params.q_br, params.pbs, false);
            if params.seeded {
                BootstrapKey::generate_seeded(&s_lwe, &s_br, g, rns.clone(), group, pc_core::keys::tfhe_tag::BSK, &dg_bsk, prng)
            } else {
                BootstrapKey::generate(&s_lwe, &s_br, g, rns.clone(), group, &dg_bsk, prng)
            }
        };
        let bsk = gen_bsk(params.pbs_group, prng);
        let bsk_lat = (params.lat_group > 0).then(|| gen_bsk(params.lat_group, prng));
        let repack = RepackKey::generate(s, &s_br, Gadget::new(p0, params.rp, true), params.rp_aux, &ntt_p, &dg_rp, &dg, params.seeded, prng)?;
        let chain = if params.chain { Some(crate::chain::ChainKey::generate(&s_br, &s_lwe, p0, &params, prng)?) } else { None };
        // the packing key: GLWE(s_br, q_br) at the bootstrapping key's sigma (the same instance)
        let pack = params.pack.then(|| {
            if params.seeded {
                crate::pack::PackingKey::generate_seeded(&s_lwe, &s_br, crate::pack::PACK_GADGET, rns.clone(), pc_core::keys::tfhe_tag::PACK, &dg_bsk, prng)
            } else {
                crate::pack::PackingKey::generate(&s_lwe, &s_br, crate::pack::PACK_GADGET, rns.clone(), &dg_bsk, prng)
            }
        });
        let bsk_lat2 = (params.lat_group2 > 0).then(|| gen_bsk(params.lat_group2, prng));
        Ok((Switcher { params, n, p0, p1, ntt_p, rs, ksk, bsk, bsk_lat, bsk_lat2, repack, chain, pack }, s_lwe, s_br))
    }

    /// The LUT `tv` (+ `offset`, mod q_br) on the coefficients `valid` of a split ciphertext (phase m q0 / 4 there);
    /// every other coefficient comes back as an exact zero. Returns the split ciphertext of the outputs (mod p0 p1).
    pub fn apply(&self, ct: &RnsRlwe, valid: &[usize], tv: &[u64], offset: u64) -> RnsRlwe {
        let p0 = self.p0;
        let low = rescale_p1(ct, p0, self.p1);
        let sw = self.rs.apply(&low, &self.ntt_p[0]);
        let lwes: Vec<Lwe> = valid.par_iter().map(|&j| self.ksk.apply(&self.rs.extract(&sw, j, p0))).collect();
        let outs: Vec<Lwe> = lwes.par_iter().map(|l| self.bsk.bootstrap_ext(tv, offset, l, p0, self.params.nu)).collect();
        let mut slots: Vec<Option<Lwe>> = vec![None; self.n];
        for (&j, o) in valid.iter().zip(outs) {
            slots[j] = Some(o.mod_switch(self.params.q_br, p0));
        }
        self.repack.repack(&slots, self.params.n_br, &self.ntt_p)
    }

    /// The keys' bytes as stored / shipped (u32 words; seeded masks: the seeds), per key kind.
    pub fn stored_sizes(&self) -> Vec<(String, usize)> {
        let mut v = vec![("ring switch".to_string(), self.rs.stored_bytes()), ("LWE key switch".to_string(), self.ksk.stored_bytes())];
        if let Some(c) = &self.chain {
            v.push(("chain key switch".to_string(), c.ksk.stored_bytes()));
        }
        v.push((format!("bootstrapping (groups of {})", self.bsk.group), self.bsk.stored_bytes()));
        for k in [&self.bsk_lat, &self.bsk_lat2].into_iter().flatten() {
            v.push((format!("bootstrapping (groups of {})", k.group), k.stored_bytes()));
        }
        let (c, r) = self.repack.stored_bytes();
        v.push(("repack columns".to_string(), c));
        v.push(("repack ring".to_string(), r));
        if let Some(p) = &self.pack {
            v.push(("packing".to_string(), p.stored_bytes()));
        }
        v
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
