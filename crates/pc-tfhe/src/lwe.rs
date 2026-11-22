//! LWE ciphertexts (phase b + <a, s> mod q), the modulus switch and the LWE key switch.
use std::sync::OnceLock;

use pc_core::keys::{mask_expand_row, DiscreteGaussian, MaskSeed, Prng};
use rayon::prelude::*;

use crate::gadget::{fixed_ratio, scale_round, Gadget};
use crate::modq::{addmod, from_signed, mulmod};

#[derive(Clone, Debug, PartialEq)]
pub struct Lwe {
    pub a: Vec<u64>,
    pub b: u64,
}

impl Lwe {
    pub fn trivial(n: usize, b: u64) -> Lwe {
        Lwe { a: vec![0; n], b }
    }

    pub fn phase(&self, s: &[i64], q: u64) -> u64 {
        self.a.iter().zip(s).fold(self.b, |acc, (&a, &si)| addmod(acc, mulmod(a, from_signed(si, q), q), q))
    }

    /// Encryption of m (already scaled) under s, noise from `dg`.
    pub fn encrypt(m: u64, s: &[i64], q: u64, dg: &DiscreteGaussian, prng: &mut Prng) -> Lwe {
        let a: Vec<u64> = (0..s.len()).map(|_| prng.uniform_below(q)).collect();
        let e = from_signed(dg.sample(prng), q);
        let mut ct = Lwe { a, b: 0 };
        let p = ct.phase(s, q);
        ct.b = addmod(addmod(m, e, q), q - p, q) % q;
        ct
    }

    /// self + o (mod q), the same key.
    pub fn add(&self, o: &Lwe, q: u64) -> Lwe {
        Lwe { a: self.a.iter().zip(&o.a).map(|(&x, &y)| addmod(x, y, q)).collect(), b: addmod(self.b, o.b, q) }
    }

    /// self - o (mod q), the same key.
    pub fn sub(&self, o: &Lwe, q: u64) -> Lwe {
        Lwe { a: self.a.iter().zip(&o.a).map(|(&x, &y)| addmod(x, q - y, q)).collect(), b: addmod(self.b, q - o.b, q) }
    }

    /// Every word x -> round(x to / from) mod `to` (to < from), the device's fixed-point form (`gadget::scale_round`).
    pub fn mod_switch(&self, from: u64, to: u64) -> Lwe {
        let m = fixed_ratio(to, from);
        let sw = |x: u64| -> u64 { scale_round(x, m) % to };
        Lwe { a: self.a.iter().map(|&x| sw(x)).collect(), b: sw(self.b) }
    }
}

/// The masks of a key-switching key: seeded (regenerated from a seed, `pc_core::keys::mask_expand`: the uniform
/// a's are ~n_out / (n_out + 1) of the key) or stored. Both are exact uniform mod q; the bodies are computed the same way.
#[derive(Clone, Debug)]
pub enum KsMask {
    /// column o (mask coordinate o < n_out) of the key rows r = i levels + t: the mask row (seed, lane o) of length
    /// n_in levels (`mask_expand(seed, o, q, r, n_in levels)`)
    Seeded(MaskSeed),
    /// [o][r] (K-contiguous columns)
    Stored(Vec<u32>),
}

/// Key-switching key s_in -> s_out: for each input coordinate i and gadget level t, key row r = i levels + t is
/// LWE_{s_out}(g_t s_in[i]) = (a_r, b[r]).
pub struct LweKsk {
    pub gadget: Gadget,
    pub n_in: usize,
    pub n_out: usize,
    /// the bodies b[r]
    pub b: Vec<u64>,
    pub mask: KsMask,
    /// a seeded key's masks [o][r], expanded on first use (`masks`)
    cache: OnceLock<Vec<u32>>,
}

impl LweKsk {
    /// The stored form: every row's mask drawn from `prng` (row by row, then its error).
    pub fn generate(s_in: &[i64], s_out: &[i64], gadget: Gadget, dg: &DiscreteGaussian, prng: &mut Prng) -> LweKsk {
        let q = gadget.q;
        let (lv, n_out) = (gadget.levels, s_out.len());
        let k = s_in.len() * lv;
        let mut masks = vec![0u32; n_out * k];
        let mut b = Vec::with_capacity(k);
        for (i, &si) in s_in.iter().enumerate() {
            for t in 0..lv {
                let ct = Lwe::encrypt(mulmod(gadget.g(t), from_signed(si, q), q), s_out, q, dg, prng);
                for (o, &a) in ct.a.iter().enumerate() {
                    masks[o * k + i * lv + t] = a as u32;
                }
                b.push(ct.b);
            }
        }
        LweKsk { gadget, n_in: s_in.len(), n_out, b, mask: KsMask::Stored(masks), cache: OnceLock::new() }
    }

    /// The seeded form (domain `tag`, a fresh seed from `prng`): the masks are `mask_expand` rows, the bodies
    /// b[r] = g_t s_in[i] + e_r - <a_r, s_out> (e_r drawn in row order). n_in levels must be a multiple of 16 and q < 2^32.
    pub fn generate_seeded(s_in: &[i64], s_out: &[i64], gadget: Gadget, tag: u32, dg: &DiscreteGaussian, prng: &mut Prng) -> LweKsk {
        let q = gadget.q;
        let (lv, n_out) = (gadget.levels, s_out.len());
        let k = s_in.len() * lv;
        assert!(k % 16 == 0 && q < 1 << 32, "seeded key-switching key: {k} rows (a multiple of 16), q {q} < 2^32");
        let seed = MaskSeed { seed: prng.ksk_seed(), tag };
        // <a_r, s_out> for every row, the masks column by column (|sum| < n_out q)
        let dot: Vec<i64> = (0..n_out)
            .into_par_iter()
            .fold(
                || vec![0i64; k],
                |mut acc, o| {
                    if s_out[o] != 0 {
                        let mut w = vec![0u32; k];
                        mask_expand_row(&seed, o as u32, q as u32, &mut w);
                        for (x, &v) in acc.iter_mut().zip(&w) {
                            *x += s_out[o] * v as i64;
                        }
                    }
                    acc
                },
            )
            .reduce(
                || vec![0i64; k],
                |mut a, b| {
                    a.iter_mut().zip(&b).for_each(|(x, &y)| *x += y);
                    a
                },
            );
        let mut b = Vec::with_capacity(k);
        for (i, &si) in s_in.iter().enumerate() {
            for t in 0..lv {
                let m = mulmod(gadget.g(t), from_signed(si, q), q);
                let e = from_signed(dg.sample(prng), q);
                let p = (dot[i * lv + t].rem_euclid(q as i64)) as u64;
                b.push(addmod(addmod(m, e, q), q - p, q) % q);
            }
        }
        LweKsk { gadget, n_in: s_in.len(), n_out, b, mask: KsMask::Seeded(seed), cache: OnceLock::new() }
    }

    /// Key rows (n_in levels).
    pub fn rows(&self) -> usize {
        self.n_in * self.gadget.levels
    }

    /// The masks [o][r] (expanded once for a seeded key).
    pub fn masks(&self) -> &[u32] {
        match &self.mask {
            KsMask::Stored(m) => m,
            KsMask::Seeded(seed) => self.cache.get_or_init(|| {
                let k = self.rows();
                let q = self.gadget.q as u32;
                let cols: Vec<Vec<u32>> = (0..self.n_out)
                    .into_par_iter()
                    .map(|o| {
                        let mut w = vec![0u32; k];
                        mask_expand_row(seed, o as u32, q, &mut w);
                        w
                    })
                    .collect();
                cols.concat()
            }),
        }
    }

    /// The stored form of this key (the same masks and bodies).
    pub fn to_stored(&self) -> LweKsk {
        LweKsk { gadget: self.gadget.clone(), n_in: self.n_in, n_out: self.n_out, b: self.b.clone(), mask: KsMask::Stored(self.masks().to_vec()), cache: OnceLock::new() }
    }

    pub fn seed(&self) -> Option<&MaskSeed> {
        match &self.mask {
            KsMask::Seeded(s) => Some(s),
            KsMask::Stored(_) => None,
        }
    }

    /// Key row (i, t) as an LWE.
    pub fn key(&self, i: usize, t: usize) -> Lwe {
        let (k, r) = (self.rows(), i * self.gadget.levels + t);
        let m = self.masks();
        Lwe { a: (0..self.n_out).map(|o| m[o * k + r] as u64).collect(), b: self.b[r] }
    }

    /// Bytes of the key as stored / shipped: the bodies (u32) and the seed, or every word.
    pub fn stored_bytes(&self) -> usize {
        let k = self.rows();
        match &self.mask {
            KsMask::Seeded(_) => 4 * k + 36,
            KsMask::Stored(_) => 4 * k * (self.n_out + 1),
        }
    }

    pub fn apply(&self, ct: &Lwe) -> Lwe {
        let q = self.gadget.q;
        let (l, k) = (self.gadget.levels, self.rows());
        assert!((k as u128) << (self.gadget.base_log.max(1) - 1) < (1u128 << 63) / q as u128, "key switch sums exceed i64");
        let mut d = vec![0i64; k];
        for (i, &ai) in ct.a.iter().enumerate() {
            self.gadget.decompose(ai, &mut d[i * l..(i + 1) * l]);
        }
        let m = self.masks();
        let r = |v: i64| v.rem_euclid(q as i64) as u64;
        let a: Vec<u64> = (0..self.n_out)
            .into_par_iter()
            .map(|o| r(m[o * k..(o + 1) * k].iter().zip(&d).map(|(&w, &x)| x * w as i64).sum::<i64>()))
            .collect();
        let b = d.iter().zip(&self.b).map(|(&x, &w)| x * w as i64).sum::<i64>();
        Lwe { a, b: addmod(r(b), ct.b, q) }
    }
}
