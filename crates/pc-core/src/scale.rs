//! Scaling-factor and CRT tables of a chain, and the scalar encoders.
//!
//! `ScaleTables::precompute` is the patched OpenFHE `CryptoParametersCKKSRNS::PrecomputeCRTTables` +
//! `CryptoParametersRNS::PrecomputeCRTTables` (HYBRID tables; patches 0002/0006 for the mixed-segment and composite
//! scaling factors), done with exact modular products instead of BigInteger division (each identity is stated where
//! it replaces a division). The FIDESlib views (`sfr`, `sfr_big`, `mrf`, indexed by limb top) are what the GPU side and
//! the scalar encoders read. `elem_for_eval_mult` / `elem_for_eval_add` are FIDESlib's `ContextData::ElemForEvalMult`
//! / `ElemForEvalAddOrSub` (Context.cu), ported operation for operation so the integer constants are bit-identical
//! (checked against the reference run's memo in `tests/scale_tables.rs`).
use crate::error::{Error, Result};
use crate::modmul::{invmod, mulmod};
use crate::params::{Chain, Scaling};

#[derive(Clone, Debug)]
pub struct ScaleTables {
    /// Top limb index `L` and primes per level.
    pub l: usize,
    pub d: usize,
    /// Primes a rescale from limb top `t` drops (`Chain::drop_at`), 0 off the level grid.
    pub drop: Vec<usize>,
    pub q: Vec<u64>,
    pub p: Vec<u64>,

    // OpenFHE-indexed (index = number of dropped primes).
    /// m_scalingFactorsReal (sizeQ entries; off-grid composite entries are OpenFHE's sentinel 1.0).
    pub sf_real_openfhe: Vec<f64>,
    /// m_scalingFactorsRealBig (sizeQ - 1 entries).
    pub sf_real_big_openfhe: Vec<f64>,

    // FIDESlib-indexed by limb top (index = top live prime id), as ContextData reads them.
    /// ScalingFactorReal[L - i] = GetScalingFactorReal(i).
    pub sfr: Vec<f64>,
    /// ScalingFactorRealBig[L - i] = GetScalingFactorRealBig(i) (0.0 where OpenFHE falls back to m_approxSF = 0).
    pub sfr_big: Vec<f64>,
    /// ModReduceFactor\[i\] = q_i as double (m_dmoduliQ).
    pub mrf: Vec<f64>,

    /// m_QlQlInvModqlDivqlModq by dropped prime: `qlql[l][i]` (i < l) = floor((Q_{l-1}^-1 mod q_l) * Q_{l-1} / q_l)
    /// mod q_i, with Q_{l-1} = q_0...q_{l-1}. That quotient is `(X - 1) / q_l` for `X = Q_{l-1}^-1 Q_{l-1} == 1 mod q_l`,
    /// and `X == 0 mod q_i`, so it equals `-q_l^-1 mod q_i`. (OpenFHE stores it as `[sizeQ-1-l][i]`.)
    pub qlql: Vec<Vec<u64>>,
    /// m_qlInvModq by dropped prime: `ql_inv_mod_q[l][i] = q_l^-1 mod q_i` (i < l).
    pub ql_inv_mod_q: Vec<Vec<u64>>,
    /// FIDESlib Globals::q_inv over every prime id (Q then P): `q_inv[i][j] = q_i^-1 mod q_j` (0 on the diagonal).
    pub q_inv: Vec<Vec<u64>>,

    /// m_PModq\[i\] = P mod q_i (= FIDESlib Constants::P).
    pub p_mod_q: Vec<u64>,
    /// m_PInvModq\[i\] = P^-1 mod q_i.
    pub p_inv_mod_q: Vec<u64>,
    /// m_PHatInvModp\[j\] = (P/p_j)^-1 mod p_j (FIDESlib ModDown_pre_scale at the special ids).
    pub p_hat_inv_mod_p: Vec<u64>,
    /// m_PHatModq\[j\]\[i\] = (P/p_j) mod q_i.
    pub p_hat_mod_q: Vec<Vec<u64>>,
    /// m_QlHatInvModq\[l\]\[i\] = (Q_l/q_i)^-1 mod q_i, Q_l = q_0...q_l, i <= l.
    pub ql_hat_inv_mod_q: Vec<Vec<u64>>,
    /// m_PartQlHatInvModq\[k\]\[t\]\[i\]: digit k truncated to its first t+1 primes, (Q_kt/q_ki)^-1 mod q_ki.
    pub part_ql_hat_inv_mod_q: Vec<Vec<Vec<u64>>>,
    /// m_paramsComplPartQ as prime ids: `compl_part_ids[l][k]` = ModUp target of digit k at limb top l (the other live
    /// Q primes in id order, then the special primes).
    pub compl_part_ids: Vec<Vec<Vec<usize>>>,
    /// m_PartQlHatModp\[l\]\[k\]\[i\]\[j\] = (Q_k^(l) / q_ki) mod (compl_part_ids\[l\]\[k\]\[j\]).
    pub part_ql_hat_mod_p: Vec<Vec<Vec<Vec<u64>>>>,
}

fn prod_mod(xs: impl Iterator<Item = u64>, m: u64) -> u64 {
    xs.fold(1u64 % m, |a, x| mulmod(a, x % m, m))
}

impl ScaleTables {
    pub fn precompute(chain: &Chain) -> Result<ScaleTables> {
        let q = chain.q.clone();
        let p = chain.p.clone();
        let size_q = q.len();
        let d = chain.level_unit;
        let composite = chain.scaling == Scaling::CompositeManual;
        let pt_mod = chain.spec.scaling_bits as f64;

        // --- scaling factors (CryptoParametersCKKSRNS::PrecomputeCRTTables, FLEXIBLE* / COMPOSITESCALING*)
        let mut sfr_o = vec![0.0f64; size_q];
        if size_q == 1 && !composite {
            sfr_o[0] = 2f64.powf(pt_mod);
        } else {
            sfr_o[0] = q[size_q - 1] as f64;
            if composite {
                for j in 1..d {
                    sfr_o[0] *= q[size_q - j - 1] as f64;
                }
            }
            let mut segment_base = sfr_o[0];
            for k in 1..size_q {
                if composite {
                    if k % d == 0 {
                        let prev = sfr_o[k - d];
                        sfr_o[k] = prev * prev;
                        for j in 0..d {
                            sfr_o[k] /= q[size_q - k + j] as f64;
                        }
                    } else {
                        sfr_o[k] = 1.0;
                    }
                    continue;
                }
                let prev = sfr_o[k - 1];
                let q_drop = q[size_q - k] as f64;
                let cand = prev * prev / q_drop;
                if (cand.log2() - q_drop.log2()).abs() > 1.5 {
                    if q_drop > cand {
                        // ascending boundary: the boundary-stage plaintext lands the ciphertext on the new prime
                        sfr_o[k] = q_drop;
                    } else {
                        // descending run: geometric descent to the size class after the run
                        let mut m = 1usize;
                        let run = q_drop.log2();
                        while size_q as i64 - k as i64 - m as i64 >= 1
                            && ((q[size_q - k - m] as f64).log2() - run).abs() <= 1.5
                        {
                            m += 1;
                        }
                        let target = if size_q as i64 - k as i64 - m as i64 >= 1 {
                            (q[size_q - k - m] as f64).log2()
                        } else {
                            pt_mod
                        };
                        let step = (prev.log2() - target) / m as f64;
                        sfr_o[k] = prev * 2f64.powf(-step);
                    }
                    segment_base = sfr_o[k];
                    continue;
                }
                sfr_o[k] = cand;
                let ratio = sfr_o[k] / segment_base;
                if ratio <= 0.5 || ratio >= 2.0 {
                    return Err(Error::Params(format!(
                        "FLEXIBLEAUTO scaling failed at level {k} with scaling factor ratio {ratio}"
                    )));
                }
            }
        }
        if chain.spec.lean.is_some() {
            // a lean chain's single-prime levels (the composite rule above leaves its sentinel there): walk the levels
            // down from the top, sf' = sf^2 / (the prime dropped); the composite levels keep the rule's values
            let mut k = 0;
            for &g in chain.groups.iter().skip(1).rev() {
                if g < d {
                    let prev = sfr_o[k];
                    sfr_o[k + g] = prev * prev / q[size_q - 1 - k] as f64;
                }
                k += g;
            }
        }
        let mut big_o = vec![0.0f64; size_q - 1];
        if !big_o.is_empty() {
            big_o[0] = sfr_o[0] * sfr_o[0];
            for k in 1..size_q - 1 {
                big_o[k] = sfr_o[k] * sfr_o[k];
            }
        }
        // FIDESlib's RawParams flip; out-of-range getters return m_approxSF, which is 0 on these techniques.
        let l = size_q - 1;
        let mut sfr = vec![0.0; size_q];
        let mut sfr_big = vec![0.0; size_q];
        for i in 0..size_q {
            sfr[l - i] = sfr_o[i];
            sfr_big[l - i] = big_o.get(i).copied().unwrap_or(0.0);
        }
        let mrf: Vec<f64> = q.iter().map(|&x| x as f64).collect();

        // --- rescale tables
        let mut qlql = vec![Vec::new(); size_q];
        let mut ql_inv_mod_q = vec![Vec::new(); size_q];
        for lq in 1..size_q {
            for i in 0..lq {
                let inv = invmod(q[lq] % q[i], q[i]);
                ql_inv_mod_q[lq].push(inv);
                qlql[lq].push(if inv == 0 { 0 } else { q[i] - inv });
            }
        }
        let all = chain.primes();
        let q_inv: Vec<Vec<u64>> = all.iter().map(|&a| all.iter().map(|&b| invmod(a % b, b)).collect()).collect();

        // --- HYBRID key-switching tables (CryptoParametersRNS::PrecomputeCRTTables)
        let p_mod_q: Vec<u64> = q.iter().map(|&qi| prod_mod(p.iter().copied(), qi)).collect();
        let p_inv_mod_q: Vec<u64> = p_mod_q.iter().zip(&q).map(|(&x, &qi)| invmod(x, qi)).collect();
        let p_hat = |j: usize, m: u64| prod_mod(p.iter().enumerate().filter(|(k, _)| *k != j).map(|(_, &x)| x), m);
        let p_hat_inv_mod_p: Vec<u64> = (0..p.len()).map(|j| invmod(p_hat(j, p[j]), p[j])).collect();
        let p_hat_mod_q: Vec<Vec<u64>> = (0..p.len()).map(|j| q.iter().map(|&qi| p_hat(j, qi)).collect()).collect();
        let ql_hat_inv_mod_q: Vec<Vec<u64>> = (0..size_q)
            .map(|lt| {
                (0..=lt)
                    .map(|i| invmod(prod_mod((0..=lt).filter(|&k| k != i).map(|k| q[k]), q[i]), q[i]))
                    .collect()
            })
            .collect();

        let digits = &chain.digits;
        let alpha = digits[0].len();
        let part_ql_hat_inv_mod_q: Vec<Vec<Vec<u64>>> = digits
            .iter()
            .map(|dg| {
                (0..dg.len())
                    .map(|t| {
                        (0..=t)
                            .map(|i| {
                                let qi = q[dg[i]];
                                invmod(prod_mod((0..=t).filter(|&k| k != i).map(|k| q[dg[k]]), qi), qi)
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let mut compl_part_ids = Vec::with_capacity(size_q);
        let mut part_ql_hat_mod_p = Vec::with_capacity(size_q);
        for lt in 0..size_q {
            let beta = (lt + 1).div_ceil(alpha);
            let mut ids_l = Vec::with_capacity(beta);
            let mut hat_l = Vec::with_capacity(beta);
            for k in 0..beta {
                let live: Vec<usize> = digits[k].iter().copied().filter(|&i| i <= lt).collect();
                let mut ids: Vec<usize> = (0..=lt).filter(|i| !digits[k].contains(i)).collect();
                ids.extend(chain.special_ids());
                let hats: Vec<Vec<u64>> = live
                    .iter()
                    .map(|&qi_id| {
                        ids.iter()
                            .map(|&t| prod_mod(live.iter().filter(|&&x| x != qi_id).map(|&x| q[x]), all[t]))
                            .collect()
                    })
                    .collect();
                ids_l.push(ids);
                hat_l.push(hats);
            }
            compl_part_ids.push(ids_l);
            part_ql_hat_mod_p.push(hat_l);
        }

        let drop: Vec<usize> = (0..size_q).map(|t| chain.drop_at(t).unwrap_or(0)).collect();
        Ok(ScaleTables {
            l,
            d,
            drop,
            q,
            p,
            sf_real_openfhe: sfr_o,
            sf_real_big_openfhe: big_o,
            sfr,
            sfr_big,
            mrf,
            qlql,
            ql_inv_mod_q,
            q_inv,
            p_mod_q,
            p_inv_mod_q,
            p_hat_inv_mod_p,
            p_hat_mod_q,
            ql_hat_inv_mod_q,
            part_ql_hat_inv_mod_q,
            compl_part_ids,
            part_ql_hat_mod_p,
        })
    }

    /// ContextData::sfAtLimb: the scaling factor of a ciphertext whose top limb is `limb_top` (must be on the grid).
    pub fn sf(&self, limb_top: i32) -> Result<f64> {
        if limb_top < 0 || limb_top as usize > self.l || self.drop[limb_top as usize] == 0 {
            return Err(Error::Params(format!(
                "sfAtLimb({limb_top}) is off the composite level grid (L={}, d={})",
                self.l, self.d
            )));
        }
        Ok(self.sfr[limb_top as usize])
    }

    /// ContextData::modReduceProduct: the product of the factors dropped by a rescale from `limb_top` (`d` of them
    /// but at a lean chain's single-prime level).
    pub fn mod_reduce_product(&self, limb_top: i32) -> f64 {
        let mut f = 1.0f64;
        let g = self.drop.get(limb_top as usize).copied().filter(|&g| g > 0).unwrap_or(self.d);
        for j in 0..g as i32 {
            f *= self.mrf[(limb_top - j) as usize];
        }
        f
    }

    /// ContextData::ElemForEvalMult(level, operand, level_in): the scalar `operand` as a CRT vector over q_0..q_level,
    /// scaled so that a multiply-then-rescale lands a `level_in` ciphertext on the scale of `level - d`.
    pub fn elem_for_eval_mult(&self, level: i32, operand: f64, level_in: i32) -> Result<Vec<u64>> {
        let num = (level + 1) as usize;
        let cd = self.drop.get(level as usize).copied().filter(|&g| g > 0).unwrap_or(self.d) as i32;
        let sc: f64 = if level_in == -1 || level_in == level {
            self.sf(level)?
        } else {
            if level < cd {
                return Err(Error::Params(format!("ElemForEvalMult: level {level} < composite degree {cd}")));
            }
            let sf_in = self.sf(level_in)?;
            let sf_out = self.sf(level - cd)?;
            let resc = self.mod_reduce_product(level);
            sf_out * resc / sf_in
        };
        let mut log_approx: i32 = 0;
        let res = (operand * sc).abs();
        if res > 0.0 {
            let log_sf = res.log2().ceil() as i32;
            let log_valid = if log_sf <= 125 { log_sf } else { 125 };
            log_approx = log_sf - log_valid;
        }
        let approx = 2f64.powf(log_approx as f64);
        let large: i128 = (operand / approx * sc + 0.5) as i128;
        let large_abs = large.abs();
        let bound: i128 = 1i128 << 63;
        let mut f: Vec<u64> = vec![0; num];
        if large_abs > bound {
            for i in 0..num {
                let r = large % self.q[i] as i128;
                f[i] = if r < 0 { (r + self.q[i] as i128) as u64 } else { r as u64 };
            }
        } else {
            let c = large as i64;
            for i in 0..num {
                let r = c % self.q[i] as i64;
                f[i] = if r < 0 { (r + self.q[i] as i64) as u64 } else { r as u64 };
            }
        }
        if log_approx > 0 {
            self.times_pow2(&mut f, log_approx);
        }
        Ok(f.iter().zip(&self.q).map(|(&x, &q)| x % q).collect())
    }

    /// ContextData::ElemForEvalAddOrSub(level, operand >= 0, noise_deg) (FLEXIBLEAUTO, level < L or not EXT).
    pub fn elem_for_eval_add(&self, level: i32, operand: f64, noise_deg: i32) -> Result<Vec<u64>> {
        let num = (level + 1) as usize;
        let sc = self.sf(level)?;
        let mut log_approx: i32 = 0;
        let res = (operand * sc).abs();
        if res > 0.0 {
            let log_sf = res.log2().ceil() as i32;
            let log_valid = if log_sf <= 61 { log_sf } else { 61 };
            log_approx = log_sf - log_valid;
        }
        let approx = 2f64.powf(log_approx as f64);
        let scc: u64 = (operand * sc / approx + 0.5) as u64;
        let mut c: Vec<u64> = (0..num).map(|i| scc % self.q[i]).collect();
        if log_approx > 0 {
            self.times_pow2(&mut c, log_approx);
        }
        let isf: u64 = (sc + 0.5) as u64;
        for _ in 1..noise_deg {
            for i in 0..num {
                c[i] = mulmod(c[i], isf % self.q[i], self.q[i]);
            }
        }
        Ok(c.iter().zip(&self.q).map(|(&x, &q)| x % q).collect())
    }

    /// Multiply a CRT vector by 2^la in steps of at most 2^60 (the reference's chunking).
    fn times_pow2(&self, f: &mut [u64], mut la: i32) {
        let num = f.len();
        let step = la.min(60);
        let mut crt: Vec<u64> = (0..num).map(|i| (1u64 << step) % self.q[i]).collect();
        la -= step;
        while la > 0 {
            let st = la.min(60);
            for i in 0..num {
                crt[i] = mulmod(crt[i], (1u64 << st) % self.q[i], self.q[i]);
            }
            la -= st;
        }
        for i in 0..num {
            f[i] = mulmod(f[i], crt[i], self.q[i]);
        }
    }
}
