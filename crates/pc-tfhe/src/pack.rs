//! Encrypted (private) lookup tables: the public functional packing key switch LWE -> GLWE (Chillotti, Gama,
//! Georgieva, Izabachene, TFHE, J. Cryptol. 2020, Alg. 2 "PublicKS" with a linear map), landing under the blind
//! rotation's GLWE key s_br at its modulus Q = q_a q_b, and the blind rotation of a caller-supplied (encrypted)
//! accumulator (`pbs::BootstrapKey::blind_rotate_acc`).
//!
//! Packing: k LWEs c_j = (a_j, b_j) mod q_in under the binary LWE key the PBS reads (s_lwe, dimension n_lwe) go to one
//! GLWE whose phase is sum_j w_j X^(p_j) phase(c_j) (q_in -> Q): every word is lifted round(x Q / q_in), and with
//! K[i][t] = GLWE_s_br(g_t s_i) (a constant polynomial; the gadget approximate over Q)
//!   out = sum_j w_j X^(p_j) (b_j, 0) + sum_(i, t) K[i][t] * D_it,   D_it = sum_j w_j d_t(a_ji) X^(p_j),
//! i.e. n_lwe x levels polynomial products whatever k (one NTT of D_it per prime each).
//! Noise per output coefficient (Q units; `PackingKey::noise_var`): the key term k_c n_lwe l E[d^2] sigma_K^2 w^2
//! (k_c the inputs landing on that coefficient's convolution: the packed count), the decomposition's truncation
//! (n_lwe / 2) (Q / B^l)^2 / 12 w^2, the lift's rounding (n_lwe / 2 + 1) / 12 w^2, and the inputs' own noise
//! (Q / q_in)^2 sigma_in^2 w^2.
//! Security: the packing key is GLWE at (s_br, Q) with the bootstrapping key's sigma (`Params::bsk_sigma`): the same
//! instance as the bootstrapping key (security/README.md: 138.6 bits).
use std::sync::OnceLock;

use pc_core::keys::{mask_expand_row64, DiscreteGaussian, MaskSeed, Prng};
use rayon::prelude::*;

use crate::gadget::Gadget;
use crate::lwe::Lwe;
use crate::modq::{addmod, from_signed, mulmod, submod};
use crate::pbs::Rns;
use crate::rlwe::Rlwe;

/// The packing key s_lwe -> GLWE(s_br, Q): K[i][t] per prime (c0, c1) in NTT form; c1 stored or seeded (row
/// (i, t, prime) = the mask row (seed, (i levels + t) 2 + prime) of length n, `pc_core::keys::mask_expand_row`).
pub struct PackingKey {
    pub gadget: Gadget,
    pub rns: Rns,
    pub sigma: f64,
    /// [i][t][prime] -> c0, NTT form
    c0: Vec<Vec<[Vec<u64>; 2]>>,
    /// stored c1 [i][t][prime] (None: seeded)
    c1: Option<Vec<Vec<[Vec<u64>; 2]>>>,
    pub seed: Option<MaskSeed>,
    cache: OnceLock<Vec<Vec<[(Vec<u64>, Vec<u64>); 2]>>>,
}

/// The default packing gadget over Q: base 2^8, 3 levels (truncation Q / 2^24: below a q_in ~ 2^28 input's own noise).
pub const PACK_GADGET: (u32, usize) = (8, 3);

impl PackingKey {
    /// The stored form (c1 uniform per prime from `prng`, coefficient form).
    pub fn generate(s_lwe: &[i64], s_br: &[i64], gadget: (u32, usize), rns: Rns, dg: &DiscreteGaussian, prng: &mut Prng) -> PackingKey {
        let gadget = Gadget::new(rns.q, gadget, false);
        let n = rns.n();
        let (mut c0, mut c1) = (Vec::new(), Vec::new());
        for &si in s_lwe {
            let (mut r0, mut r1) = (Vec::new(), Vec::new());
            for t in 0..gadget.levels {
                let v = mulmod(from_signed(si, rns.q), gadget.g(t), rns.q);
                let e = dg.sample_vec(prng, n);
                let [(a0, a1), (b0, b1)] = [0, 1].map(|p| {
                    let ntt = &rns.ntt[p];
                    let q = ntt.q;
                    let c1: Vec<u64> = (0..n).map(|_| prng.uniform_below(q)).collect();
                    let mut m = vec![0u64; n];
                    m[0] = v % q;
                    let mut z = Rlwe::encrypt_with(&m, s_br, c1, &e, ntt);
                    ntt.fwd(&mut z.c0);
                    ntt.fwd(&mut z.c1);
                    (z.c0, z.c1)
                });
                r0.push([a0, b0]);
                r1.push([a1, b1]);
            }
            c0.push(r0);
            c1.push(r1);
        }
        PackingKey { gadget, rns, sigma: dg.sigma(), c0, c1: Some(c1), seed: None, cache: OnceLock::new() }
    }

    /// The seeded form (one seed from `prng`, domain tag `tag`): c1 = the mask row in NTT form, c0 = NTT(e) + v -
    /// c1 NTT(s_br) (v = g_t s_i, a constant: v in every NTT slot). The errors come from `prng` in row order.
    pub fn generate_seeded(s_lwe: &[i64], s_br: &[i64], gadget: (u32, usize), rns: Rns, tag: u32, dg: &DiscreteGaussian, prng: &mut Prng) -> PackingKey {
        let gadget = Gadget::new(rns.q, gadget, false);
        let (n, lv) = (rns.n(), gadget.levels);
        let seed = MaskSeed { seed: prng.ksk_seed(), tag };
        let s_ntt = [0, 1].map(|p| {
            let mut v = crate::rlwe::signed_poly(s_br, rns.ntt[p].q);
            rns.ntt[p].fwd(&mut v);
            v
        });
        let errs: Vec<Vec<i64>> = (0..s_lwe.len() * lv).map(|_| dg.sample_vec(prng, n)).collect();
        let c0 = s_lwe
            .par_iter()
            .enumerate()
            .map(|(i, &si)| {
                (0..lv)
                    .map(|t| {
                        let v = mulmod(from_signed(si, rns.q), gadget.g(t), rns.q);
                        [0, 1].map(|p| {
                            let ntt = &rns.ntt[p];
                            let q = ntt.q;
                            let mut m = vec![0u64; n];
                            mask_expand_row64(&seed, ((i * lv + t) * 2 + p) as u32, q, &mut m);
                            let mut x: Vec<u64> = errs[i * lv + t].iter().map(|&v| from_signed(v, q)).collect();
                            ntt.fwd(&mut x);
                            for j in 0..n {
                                x[j] = addmod(submod(x[j], mulmod(m[j], s_ntt[p][j], q), q), v % q, q);
                            }
                            x
                        })
                    })
                    .collect()
            })
            .collect();
        PackingKey { gadget, rns, sigma: dg.sigma(), c0, c1: None, seed: Some(seed), cache: OnceLock::new() }
    }

    /// Key rows (n_lwe).
    pub fn n_lwe(&self) -> usize {
        self.c0.len()
    }

    /// Row (i, t)'s (c0, c1) at a prime, NTT form (a seeded c1 expanded here).
    pub fn row(&self, i: usize, t: usize, p: usize) -> (&[u64], std::borrow::Cow<'_, [u64]>) {
        let c1 = match (&self.c1, &self.seed) {
            (Some(c1), _) => std::borrow::Cow::Borrowed(&c1[i][t][p][..]),
            (None, Some(seed)) => {
                let mut m = vec![0u64; self.rns.n()];
                mask_expand_row64(seed, ((i * self.gadget.levels + t) * 2 + p) as u32, self.rns.ntt[p].q, &mut m);
                std::borrow::Cow::Owned(m)
            }
            (None, None) => unreachable!("a packing key holds its masks or their seed"),
        };
        (&self.c0[i][t][p], c1)
    }

    /// [i][t][prime] -> (c0, c1), NTT form (materialized once).
    pub fn keys(&self) -> &[Vec<[(Vec<u64>, Vec<u64>); 2]>] {
        self.cache.get_or_init(|| {
            (0..self.c0.len()).map(|i| (0..self.gadget.levels).map(|t| [0, 1].map(|p| {
                let (a, b) = self.row(i, t, p);
                (a.to_vec(), b.into_owned())
            })).collect()).collect()
        })
    }

    /// The stored form of this key (the same rows).
    pub fn to_stored(&self) -> PackingKey {
        let c1 = (0..self.c0.len()).map(|i| (0..self.gadget.levels).map(|t| [0, 1].map(|p| self.row(i, t, p).1.into_owned())).collect()).collect();
        PackingKey { gadget: self.gadget.clone(), rns: self.rns.clone(), sigma: self.sigma, c0: self.c0.clone(), c1: Some(c1), seed: None, cache: OnceLock::new() }
    }

    /// Bytes of the key as stored / shipped (u32 words; a seeded key's seed and tag).
    pub fn stored_bytes(&self) -> usize {
        let w: usize = self.c0.iter().flatten().flatten().map(|r| r.len()).sum();
        4 * if self.c1.is_some() { 2 * w } else { w } + if self.seed.is_some() { 36 } else { 0 }
    }

    /// x mod q_in -> round(x Q / q_in) mod Q.
    pub fn lift(&self, x: u64, q_in: u64) -> u64 {
        let q = self.rns.q as u128;
        (((x as u128 * q) + q_in as u128 / 2) / q_in as u128 % q) as u64
    }

    /// Packs `cts` (mod `q_in`, under s_lwe) into one GLWE under s_br mod Q (residues per prime, coefficient form):
    /// input j at coefficient `pos[j]` (< n; a repeated position adds) times the integer `w[j]` (None: 1).
    pub fn pack(&self, cts: &[Lwe], q_in: u64, pos: &[usize], w: Option<&[i64]>) -> [Rlwe; 2] {
        let n = self.rns.n();
        assert_eq!(cts.len(), pos.len(), "pack: one position per LWE");
        assert!(pos.iter().all(|&p| p < n), "pack: positions below n");
        let l = self.gadget.levels;
        let nl = self.c0.len();
        let wv = |j: usize| w.map_or(1, |w| w[j]);
        // digits [j][i][t] of the lifted a's
        let mut d = vec![0i64; l];
        let digits: Vec<Vec<i64>> = cts
            .iter()
            .map(|c| {
                assert_eq!(c.a.len(), nl, "pack: LWE dimension != the key's");
                let mut out = vec![0i64; nl * l];
                for (i, &a) in c.a.iter().enumerate() {
                    self.gadget.decompose(self.lift(a, q_in), &mut d);
                    out[i * l..(i + 1) * l].copy_from_slice(&d);
                }
                out
            })
            .collect();
        let mut acc = [0, 1].map(|_| (vec![0u64; n], vec![0u64; n]));
        let mut dp = [vec![0u64; n], vec![0u64; n]];
        for i in 0..nl {
            for t in 0..l {
                let mut any = false;
                for x in dp.iter_mut() {
                    x.iter_mut().for_each(|v| *v = 0);
                }
                for (j, dj) in digits.iter().enumerate() {
                    let v = dj[i * l + t] * wv(j);
                    if v == 0 {
                        continue;
                    }
                    any = true;
                    for (p, x) in dp.iter_mut().enumerate() {
                        let q = self.rns.ntt[p].q;
                        x[pos[j]] = addmod(x[pos[j]], from_signed(v, q), q);
                    }
                }
                if !any {
                    continue;
                }
                for p in 0..2 {
                    let ntt = &self.rns.ntt[p];
                    let q = ntt.q;
                    ntt.fwd(&mut dp[p]);
                    let (k0, k1) = &self.keys()[i][t][p];
                    for x in 0..n {
                        acc[p].0[x] = addmod(acc[p].0[x], mulmod(dp[p][x], k0[x], q), q);
                        acc[p].1[x] = addmod(acc[p].1[x], mulmod(dp[p][x], k1[x], q), q);
                    }
                }
            }
        }
        [0, 1].map(|p| {
            let ntt = &self.rns.ntt[p];
            let q = ntt.q;
            let (mut c0, mut c1) = std::mem::take(&mut acc[p]);
            ntt.inv(&mut c0);
            ntt.inv(&mut c1);
            for (j, c) in cts.iter().enumerate() {
                let b = mulmod(self.lift(c.b, q_in) % q, from_signed(wv(j), q), q);
                c0[pos[j]] = addmod(c0[pos[j]], b, q);
            }
            Rlwe { c0, c1 }
        })
    }

    /// The model's variance per output coefficient (Q units) of `k` packed inputs of weight |w| <= `w`, input noise
    /// std `sigma_in` (q_in units).
    pub fn noise_var(&self, k: usize, w: f64, q_in: u64, sigma_in: f64) -> f64 {
        let nl = self.c0.len() as f64;
        let (b, l) = ((self.gadget.base_log as f64).exp2(), self.gadget.levels as f64);
        let ed2 = b * b / 12.0 + 1.0 / 6.0;
        let q = self.rns.q as f64;
        let trunc = q / (b.powf(l));
        let key = k as f64 * nl * l * ed2 * self.sigma * self.sigma;
        let tr = nl / 2.0 * trunc * trunc / 12.0;
        let lift = (nl / 2.0 + 1.0) / 12.0;
        let inp = (q / q_in as f64).powi(2) * sigma_in * sigma_in;
        w * w * (key + tr + lift + inp)
    }
}

/// The GLWE's phase mod Q (centered) under s_br, from its residues.
pub fn glwe_phase(ct: &[Rlwe; 2], s_br: &[i64], rns: &Rns) -> Vec<i128> {
    let ph = [0, 1].map(|p| ct[p].phase(s_br, &rns.ntt[p]));
    let q = rns.q as i128;
    (0..rns.n())
        .map(|i| {
            let v = rns.crt(ph[0][i], ph[1][i]) as i128;
            if v > q / 2 {
                v - q
            } else {
                v
            }
        })
        .collect()
}

/// The blind rotation's noise per extracted coefficient (Q units, `pbs::BootstrapKey`'s gadget `pbs` over Q, groups
/// of `group` key bits, ternary s_br of degree n, key sigma `sigma_bsk`): every group adds (2^G - 1) external products
/// (digits of both parts: 2 l n E[d^2] sigma^2 each, x2 for the multi-bit sum's key noise) and the decomposition's
/// truncation carried by the selector (n E[s^2] + 1) (Q / B^l)^2 / 12. The accumulator's own noise adds unchanged
/// (monomials keep it, the CMux selects it).
pub fn blind_rotate_var(q: u64, n: usize, n_lwe: usize, group: usize, pbs: (u32, usize), sigma_bsk: f64) -> f64 {
    let (b, l) = ((pbs.0 as f64).exp2(), pbs.1 as f64);
    let ed2 = b * b / 12.0;
    let steps = (n_lwe / group) as f64;
    let terms = ((1usize << group) - 1) as f64;
    let ext = 2.0 * l * n as f64 * ed2 * sigma_bsk * sigma_bsk * 2.0;
    let trunc = (n as f64 * 2.0 / 3.0 + 1.0) * (q as f64 / b.powf(l)).powi(2) / 12.0;
    steps * (terms * ext + trunc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modq::{centered, prime_below};
    use crate::params::Params;
    use crate::pbs::BootstrapKey;
    use pc_core::keys::sample_ternary;

    fn prng(seed: u64) -> Prng {
        Prng::from_seed([seed as u32, (seed >> 32) as u32, 0x5eed, 0, 0, 0, 0, 0], 7)
    }

    struct Fix {
        p: Params,
        q_in: u64,
        s_lwe: Vec<i64>,
        s_br: Vec<i64>,
        rns: Rns,
        dg_lwe: DiscreteGaussian,
    }

    fn fix(seed: u64) -> Fix {
        let p = Params::new(1 << 16).unwrap();
        let q_in = prime_below(28, 1 << 17).unwrap();
        let mut pr = prng(seed);
        let s_lwe: Vec<i64> = (0..p.n_lwe).map(|_| (pr.next_u32() & 1) as i64).collect();
        let s_br: Vec<i64> = sample_ternary(&mut pr, p.n_br).into_iter().map(|x| x as i64).collect();
        let rns = Rns::new(p.br_primes.0, p.br_primes.1, p.n_br).unwrap();
        let dg_lwe = DiscreteGaussian::new(p.lwe_sigma_rel * q_in as f64).unwrap();
        Fix { p, q_in, s_lwe, s_br, rns, dg_lwe }
    }

    /// Packing k LWEs of random messages: every coefficient's phase is the message lifted to Q, within the model's
    /// 6 sigma; untouched coefficients stay at the key noise; measured std <= model std.
    #[test]
    fn pack_decrypts() {
        let f = fix(1);
        let mut pr = prng(2);
        let dg = DiscreteGaussian::new(f.p.bsk_sigma).unwrap();
        let pk = PackingKey::generate(&f.s_lwe, &f.s_br, PACK_GADGET, f.rns.clone(), &dg, &mut pr);
        for k in [1usize, 64, 2048] {
            let msgs: Vec<u64> = (0..k).map(|_| pr.uniform_below(16)).collect();
            let delta = f.q_in / 32;
            let cts: Vec<Lwe> = msgs.iter().map(|&m| Lwe::encrypt(m * delta, &f.s_lwe, f.q_in, &f.dg_lwe, &mut pr)).collect();
            let pos: Vec<usize> = (0..k).map(|j| (j * 2048 / k) % 2048).collect();
            let out = pk.pack(&cts, f.q_in, &pos, None);
            let ph = glwe_phase(&out, &f.s_br, &f.rns);
            let q = f.rns.q as f64;
            let sd = pk.noise_var(k, 1.0, f.q_in, f.dg_lwe.sigma()).sqrt();
            let mut s2 = 0.0;
            for (j, &m) in msgs.iter().enumerate() {
                let want = (m * delta) as f64 * q / f.q_in as f64;
                let e = ph[pos[j]] as f64 - want;
                let e = e - q * (e / q).round();
                s2 += e * e;
                assert!(e.abs() < 6.0 * sd, "k {k}: coefficient {} off by {e:.3e}, model sigma {sd:.3e}", pos[j]);
            }
            let meas = (s2 / k as f64).sqrt();
            println!("pack k {k}: measured std {meas:.3e} (2^{:.1} of Q), model {sd:.3e}", (meas / q).log2());
            // the sample variance of k outputs: within 4 of its standard errors (sqrt(2 / k)) of the model's bound
            assert!(meas * meas <= sd * sd * (1.0 + 4.0 * (2.0 / k as f64).sqrt()), "k {k}: {meas:.3e} vs {sd:.3e}");
        }
    }

    /// Private lookup: a table of 16 entries packed as an encrypted test vector (entry v at the coefficients its
    /// index selects), an encrypted index i drives the blind rotation of that accumulator, the extracted LWE decrypts
    /// to table[i], over many trials, 0 errors; the variance model bounds the measured one.
    #[test]
    fn private_lut_selects() {
        let f = fix(3);
        let mut pr = prng(4);
        let dg = DiscreteGaussian::new(f.p.bsk_sigma).unwrap();
        let bsk = BootstrapKey::generate(&f.s_lwe, &f.s_br, Gadget::new(f.rns.q, f.p.pbs, false), f.rns.clone(), f.p.pbs_group, &dg, &mut pr);
        let pk = PackingKey::generate(&f.s_lwe, &f.s_br, PACK_GADGET, f.rns.clone(), &dg, &mut pr);
        let (n, t) = (f.p.n_br, 8usize);
        let span = n / t;                       // coefficients per entry: index i rotates by i span
        // table entries v_i (4 bits) at output scale Q / 32 (padding: |v| < 16 of 32); entry i fills [i span, (i + 1) span)
        let tab: Vec<u64> = (0..t).map(|_| pr.uniform_below(16)).collect();
        let delta_in = f.q_in / 32;
        let enc: Vec<Lwe> = tab.iter().map(|&v| Lwe::encrypt(v * delta_in, &f.s_lwe, f.q_in, &f.dg_lwe, &mut pr)).collect();
        let mut cts = Vec::new();
        let mut pos = Vec::new();
        for (i, c) in enc.iter().enumerate() {
            for o in 0..span {
                cts.push(c.clone());
                pos.push(i * span + o);
            }
        }
        // the rotation by X^-phi brings coefficient phi to 0: index i at phase (i span + span / 2) q_in / (2 n)
        let acc = pk.pack(&cts, f.q_in, &pos, None);
        let acc_var = pk.noise_var(1, 1.0, f.q_in, f.dg_lwe.sigma());
        let br_var = blind_rotate_var(f.rns.q, n, f.p.n_lwe, f.p.pbs_group, f.p.pbs, f.p.bsk_sigma);
        let sd = (acc_var + br_var).sqrt();
        let q = f.rns.q as f64;
        let spacing = q / 32.0;
        // per lookup: the output's decision and the index's (the mod switch to 2 n: its rounding std, the input's
        // noise x 2 n, against half the span)
        let p_fail_log2 = index_log2_fail(f.p.n_lwe, span, 1, f.p.lwe_sigma_rel).max(log2_tail(spacing / 2.0 / sd));
        assert!(p_fail_log2 < -128.0, "p_fail 2^{p_fail_log2}");
        let (mut wrong, mut s2, mut cnt) = (0, 0.0, 0);
        for trial in 0..64 {
            let i = (trial * 7 + 3) % t;
            let phase = ((i * span + span / 2) as u64 * f.q_in) / (2 * n as u64);
            let idx = Lwe::encrypt(phase, &f.s_lwe, f.q_in, &f.dg_lwe, &mut pr);
            let out = bsk.blind_rotate_acc(&[acc.clone()], &idx, f.q_in, 1).swap_remove(0);
            let ex = [0, 1].map(|p| out[p].sample_extract(0, f.rns.ntt[p].q));
            let lwe = Lwe { a: (0..n).map(|k| f.rns.crt(ex[0].a[k], ex[1].a[k])).collect(), b: f.rns.crt(ex[0].b, ex[1].b) };
            let ph = centered(lwe.phase(&f.s_br, f.rns.q), f.rns.q) as f64;
            let got = (ph / spacing).round() as i64;
            let e = ph - tab[i] as f64 * spacing;
            s2 += e * e;
            cnt += 1;
            wrong += (got != tab[i] as i64) as usize;
        }
        let meas = (s2 / cnt as f64).sqrt();
        println!("private LUT: 0..{t} x {cnt} trials, wrong {wrong}; measured std {meas:.3e}, model {sd:.3e} (acc {:.3e}, rotation {:.3e}); p_fail per lookup 2^{p_fail_log2:.1}", acc_var.sqrt(), br_var.sqrt());
        assert_eq!(wrong, 0);
        assert!(meas <= sd);
    }

    /// The encrypted-accumulator rotation on a trivial (noiseless, c1 = 0) accumulator is the plaintext one, bit for
    /// bit, for nu 1 / 2 and many-LUT k 2.
    #[test]
    fn trivial_acc_is_the_plaintext_rotation() {
        let f = fix(5);
        let mut pr = prng(6);
        let dg = DiscreteGaussian::new(f.p.bsk_sigma).unwrap();
        let bsk = BootstrapKey::generate(&f.s_lwe[..96], &f.s_br, Gadget::new(f.rns.q, f.p.pbs, false), f.rns.clone(), f.p.pbs_group, &dg, &mut pr);
        let n = f.p.n_br;
        for (nu, kl) in [(1usize, 1usize), (2, 1), (1, 2)] {
            let tv: Vec<u64> = (0..nu * n).map(|_| pr.uniform_below(f.rns.q)).collect();
            let idx = Lwe::encrypt(pr.uniform_below(f.q_in), &f.s_lwe[..96], f.q_in, &f.dg_lwe, &mut pr);
            let a = bsk.blind_rotate_mlut(&tv, &idx, f.q_in, nu, kl);
            let tvr = f.rns.residues(&tv);
            let comps: Vec<[Rlwe; 2]> = (0..nu).map(|c| [0, 1].map(|p| Rlwe { c0: (0..n).map(|j| tvr[p][nu * j + c]).collect(), c1: vec![0; n] })).collect();
            let b = bsk.blind_rotate_acc(&comps, &idx, f.q_in, kl);
            assert_eq!(a, b, "nu {nu} k {kl}");
        }
    }
}

/// log2 of the probability that an index LWE (binary key of dimension `n_lwe`, noise std `sigma_rel` of its
/// modulus) lands more than half a `span` of coefficients off after the mod switch to 2 nu n (`ms_body_mean`'s
/// rounding: (n_lwe / 2 + 1) / 12 in switched units).
pub fn index_log2_fail(n_lwe: usize, span: usize, nu: usize, sigma_rel: f64) -> f64 {
    let _ = nu;
    let v = (n_lwe as f64 / 2.0 + 1.0) / 12.0 + (sigma_rel * 2.0 * (nu * 2048) as f64).powi(2);
    log2_tail(span as f64 / 2.0 / v.sqrt())
}

/// log2 P(|N(0, 1)| > z).
pub fn log2_tail(z: f64) -> f64 {
    if z < 8.0 {
        return (libm_erfc(z / std::f64::consts::SQRT_2)).log2();
    }
    let ln = -z * z / 2.0 - z.ln() - 0.5 * (std::f64::consts::PI / 2.0).ln() + (1.0 - 1.0 / (z * z)).ln();
    ln / std::f64::consts::LN_2
}

fn libm_erfc(x: f64) -> f64 {
    // Abramowitz-Stegun 7.1.26 is too coarse in the tail; a continued fraction for x >= 0
    let t = 1.0 / (1.0 + 0.5 * x);
    let y = t * (-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))).exp();
    y
}
