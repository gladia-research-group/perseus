//! TFHE-native chains: a bootstrap's output back to a bootstrap input (mod switch to p0, key switch s_br -> s_lwe),
//! skipping the CKKS round trip. Native encoding: v at phase v q / 4 (`native_tv`).
use pc_core::keys::{DiscreteGaussian, Prng};

use crate::gadget::Gadget;
use crate::lwe::{Lwe, LweKsk};
use crate::params::Params;
use crate::pbs::test_vector;
use crate::Result;

pub struct ChainKey {
    pub ksk: LweKsk,
    pub p0: u64,
    pub q_br: u64,
}

impl ChainKey {
    pub fn generate(s_br: &[i64], s_lwe: &[i64], p0: u64, params: &Params, prng: &mut Prng) -> Result<ChainKey> {
        let dg = DiscreteGaussian::new(params.lwe_sigma_rel * p0 as f64)?;
        Ok(ChainKey { ksk: LweKsk::generate(s_br, s_lwe, Gadget::new(p0, params.ks, false), &dg, prng), p0, q_br: params.q_br })
    }

    /// A bootstrap output (or a sum of them) as the next bootstrap's input: LWE(dim n_lwe) mod p0.
    pub fn input(&self, out: &Lwe) -> Lwe {
        self.ksk.apply(&out.mod_switch(self.q_br, self.p0))
    }
}

/// The test vector of f in the native encoding: input m q / 4, output f(m) q / 4 (|f| < 2).
pub fn native_tv(n: usize, q: u64, f: impl Fn(f64) -> f64) -> Vec<u64> {
    test_vector(n, q, |m| ((q as f64 * f(m) / 4.0).round() as i64).rem_euclid(q as i64) as u64)
}
