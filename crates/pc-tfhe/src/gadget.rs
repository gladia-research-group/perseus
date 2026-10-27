//! Signed base-2^b gadget decomposition over a prime q.
//!
//! Approximate: x is rounded to `levels * b` bits of precision (x~ = round(x 2^(l b) / q)) and x~ is split into balanced
//! digits; the gadget values are round(q B^t / 2^(l b)), so sum d_t g_t = x up to q / 2^(l b + 1). Exact: l b >=
//! log2 q, x~ is x itself (centered) and g_t = B^t, so the recomposition is x as an integer in (-q/2, q/2].
use crate::modq::{centered, from_signed};

#[derive(Clone, Debug)]
pub struct Gadget {
    pub q: u64,
    pub base_log: u32,
    pub levels: usize,
    pub exact: bool,
    g: Vec<u64>,
}

impl Gadget {
    pub fn new(q: u64, (base_log, levels): (u32, usize), exact: bool) -> Gadget {
        let bits = base_log as usize * levels;
        assert!(bits < 62, "gadget: {bits} bits");
        if exact {
            assert!(1u128 << bits >= q as u128, "exact gadget: {levels} x {base_log} bits do not cover q = {q}");
        }
        let g = (0..levels)
            .map(|t| {
                if exact {
                    from_signed(1i64 << (base_log as usize * t), q)
                } else {
                    let v = (q as f64) * ((base_log as usize * t) as f64 - bits as f64).exp2();
                    (v.round() as u64) % q
                }
            })
            .collect();
        Gadget { q, base_log, levels, exact, g }
    }

    /// g_t mod q.
    pub fn g(&self, t: usize) -> u64 {
        self.g[t]
    }

    /// Balanced digits d_t in [-B/2, B/2) of x (low digit first).
    pub fn decompose(&self, x: u64, out: &mut [i64]) {
        let bits = self.base_log as usize * self.levels;
        let xc = centered(x, self.q) as i128;
        let mut v: i128 = if self.exact { xc } else { (xc * (1i128 << bits) + (self.q as i128) / 2).div_euclid(self.q as i128) };
        let b = 1i128 << self.base_log;
        for t in 0..self.levels {
            let mut r = v.rem_euclid(b);
            if r >= b / 2 {
                r -= b;
            }
            out[t] = r as i64;
            v = (v - r) >> self.base_log;
        }
        // the edge value 2^(l b - 1) leaves a carry: fold it into the top digit
        out[self.levels - 1] += (v as i64) << self.base_log;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modq::prime_below;

    #[test]
    fn recompose() {
        let q = prime_below(31, 1 << 17).unwrap();
        for (exact, gl) in [(true, (16u32, 2usize)), (false, (8, 3)), (false, (3, 5))] {
            let g = Gadget::new(q, gl, exact);
            let mut d = vec![0i64; gl.1];
            for x in [0u64, 1, q / 2, q / 2 + 1, q - 1, 123456789 % q, 2_000_000_001 % q] {
                g.decompose(x, &mut d);
                let mut acc: i128 = 0;
                for t in 0..gl.1 {
                    acc += d[t] as i128 * centered(g.g(t), q) as i128;
                    assert!(d[t].abs() <= 1 << gl.0, "digit {} out of range", d[t]);
                }
                let err = (acc - centered(x, q) as i128).rem_euclid(q as i128);
                let err = err.min(q as i128 - err);
                let tol = if exact { 0 } else { (q >> (gl.0 as usize * gl.1)) as i128 + (1 << gl.0) };
                assert!(err <= tol, "x {x} exact {exact} err {err} tol {tol}");
            }
        }
    }
}
