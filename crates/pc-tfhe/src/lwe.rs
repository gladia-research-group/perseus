//! LWE ciphertexts (phase b + <a, s> mod q), the modulus switch and the LWE key switch.
use pc_core::keys::{DiscreteGaussian, Prng};
use rayon::prelude::*;

use crate::gadget::Gadget;
use crate::modq::{addmod, centered, from_signed, mulmod};

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

    /// Every word times round(to / from) as a real ratio: x -> round(x to / from) (centered), mod `to`.
    pub fn mod_switch(&self, from: u64, to: u64) -> Lwe {
        let sw = |x: u64| -> u64 {
            let c = centered(x, from) as i128;
            let v = (c * to as i128 * 2 + from as i128).div_euclid(2 * from as i128);
            from_signed(v as i64, to)
        };
        Lwe { a: self.a.iter().map(|&x| sw(x)).collect(), b: sw(self.b) }
    }
}

/// Key-switching key s_in -> s_out: for each input coordinate i and gadget level t, LWE_{s_out}(g_t s_in[i]).
pub struct LweKsk {
    pub gadget: Gadget,
    pub n_out: usize,
    /// [i][t]
    pub keys: Vec<Vec<Lwe>>,
}

impl LweKsk {
    pub fn generate(s_in: &[i64], s_out: &[i64], gadget: Gadget, dg: &DiscreteGaussian, prng: &mut Prng) -> LweKsk {
        let q = gadget.q;
        let keys = s_in
            .iter()
            .map(|&si| (0..gadget.levels).map(|t| Lwe::encrypt(mulmod(gadget.g(t), from_signed(si, q), q), s_out, q, dg, prng)).collect())
            .collect();
        LweKsk { n_out: s_out.len(), gadget, keys }
    }

    pub fn apply(&self, ct: &Lwe) -> Lwe {
        let q = self.gadget.q;
        let l = self.gadget.levels;
        let parts: Vec<(Vec<i128>, i128)> = ct
            .a
            .par_chunks(64)
            .enumerate()
            .map(|(c, chunk)| {
                let mut acc = vec![0i128; self.n_out];
                let mut accb = 0i128;
                let mut d = vec![0i64; l];
                for (o, &ai) in chunk.iter().enumerate() {
                    self.gadget.decompose(ai, &mut d);
                    for (t, &dt) in d.iter().enumerate() {
                        if dt == 0 {
                            continue;
                        }
                        let k = &self.keys[c * 64 + o][t];
                        for (x, &y) in acc.iter_mut().zip(&k.a) {
                            *x += dt as i128 * y as i128;
                        }
                        accb += dt as i128 * k.b as i128;
                    }
                }
                (acc, accb)
            })
            .collect();
        let mut a = vec![0i128; self.n_out];
        let mut b = ct.b as i128;
        for (pa, pb) in parts {
            for (x, y) in a.iter_mut().zip(pa) {
                *x += y;
            }
            b += pb;
        }
        let r = |v: i128| v.rem_euclid(q as i128) as u64;
        Lwe { a: a.into_iter().map(r).collect(), b: r(b) }
    }
}
