//! Noise model of a functional bootstrap (FBS, `stc_first::FbsSpec`): a variance bound on the lookup's phase at
//! EvalMod's input and on the output, from which the caller derives Gaussian-tail failure probabilities
//! (python/perseus/impl/fbs.py `p_fail`).
//!
//! Per lookup (one coefficient of the split = one real slot component of EvalMod), as standard deviations:
//! - the StC-first split: every StC stage's rescale rounding (a ring element with i.i.d. coefficients of variance
//!   (1 + h) / 12, h = E|s|^2 = 2N/3 for the uniform ternary secret) and key-switch noise, carried to the lookup by the
//!   later stages' operator norm (power iteration on the stages' own diagonals, 2 % margin);
//!   the rescale to q0, the encapsulation key switch to the sparse secret and the switch back after ModRaise (phase
//!   units: / q0);
//! - the CtS stages (slot domain: per real component N (1 + h) / 24 per rescale, the same operator-norm carry) and the
//!   conjugation of the complex EvalMod;
//! - EvalMod and the series: `pc_shadow::evalmod::noise` on the device's own program, linearized (the phase noise by
//!   the program's derivative; every rescale's rounding and relinearization by its adjoint, independent), at every
//!   ModRaise integer |I| <= K; plus the other lookup of the complex slot: its chain's fresh noise has an imaginary
//!   part, which the recombination re + i im lands on this lookup (bounded by the largest over the inputs given).
//! Validated against pc-bench fbs on n32 (K32 table, 2^23 outputs per case): the predicted std is >= 3.3x the measured
//! RMS at every table node (sign t = 16, 64) and >= 1.7x per input bin of the identity series (phase noise).
//! Key-switch noise after ModDown: sum over digits of N D^2 sigma^2 / 12 per coefficient divided by P^2 (D the largest
//! digit modulus, sigma `DiscreteGaussian::SIGMA`), plus ModDown's rounding.
use super::stc_first::{fbs_chain_check, FbsHost, StcFirstPlan};
use crate::error::{shape, Result};
use pc_core::bts_setup::{cts_stc_factors, FftStage, C64};
use pc_core::scale::ScaleTables;
use pc_shadow::evalmod::noise::{self, Fresh};
use pc_shadow::fides::FMeta;

/// The bootstrap-side noise terms of a plan (everything but the caller's input and the EvalMod program).
#[derive(Clone, Debug)]
pub struct FbsNoiseModel {
    st: ScaleTables,
    /// EvalMod's input on the drop-0 variant (where every FBS reads).
    pub em_in: FMeta,
    /// StC split stages: std per lookup in input units (x), the input's slot values.
    pub split_x: f64,
    /// Rescale to q0, encapsulation and the switch back after ModRaise: std in phase units (u = coefficient / q0).
    pub q0_u: f64,
    /// CtS stages and EvalMod's conjugation: std at EvalMod's input (y = u / K); the conjugation's part alone.
    pub cts_y: f64,
    pub conj_y: f64,
    /// Fresh variances of EvalMod's rescales and relinearizations (integer units, per real slot component).
    pub fresh: Fresh,
}

/// One input point of `FbsNoiseModel::eval`: the output over every ModRaise integer |I| <= K (min, max of the
/// noise-free program value; the largest std bound).
#[derive(Clone, Copy, Debug)]
pub struct FbsPoint {
    pub lo: f64,
    pub hi: f64,
    pub sigma: f64,
}

/// (value, std bound) at one ModRaise integer (`FbsNoise::by_i`); NaN where the EvalMod input leaves [-1, 1].
pub type FbsAt = (f64, f64);

/// `FbsNoiseModel::eval`'s result.
#[derive(Clone, Debug)]
pub struct FbsNoise {
    /// Phase-noise std at EvalMod's input, the bootstrap's own (u units), and with the caller's input noise.
    pub phase_own: f64,
    pub phase: f64,
    pub points: Vec<FbsPoint>,
    /// Per input point, at I = -K..=K: the noise-free value and the std bound.
    pub by_i: Vec<Vec<FbsAt>>,
}

/// The adjoint of `FftStage::apply`.
fn apply_adj(s: &FftStage, y: &[C64]) -> Vec<C64> {
    let len = y.len() as i64;
    let c = s.center();
    let mut x = vec![C64::ZERO; y.len()];
    for k in 0..y.len() {
        for (d, dv) in s.diags.iter().enumerate() {
            let src = (k as i64 + (d as i64 - c) * s.stride as i64).rem_euclid(len) as usize;
            let a = dv[k].conj();
            x[src] = C64::new(x[src].re + a.re * y[k].re - a.im * y[k].im, x[src].im + a.re * y[k].im + a.im * y[k].re);
        }
    }
    x
}

fn norm2(z: &[C64]) -> f64 {
    z.iter().map(|c| c.re * c.re + c.im * c.im).sum::<f64>().sqrt()
}

/// The operator norm of the product of `stages` (applied in order) by power iteration on A^H A (40 steps, 2 %
/// margin for the convergence from below); 1 for no stage.
fn op_norm(stages: &[&FftStage]) -> f64 {
    let Some(first) = stages.first() else { return 1.0 };
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut z: Vec<C64> = (0..first.len())
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            C64::new((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5, (seed % 1021) as f64 / 1021.0 - 0.5)
        })
        .collect();
    let mut lam = 0.0;
    for _ in 0..40 {
        let n0 = norm2(&z);
        let mut y: Vec<C64> = z.iter().map(|c| c.scale(1.0 / n0)).collect();
        for s in stages {
            y = s.apply(&y);
        }
        for s in stages.iter().rev() {
            y = apply_adj(s, &y);
        }
        lam = norm2(&y);
        z = y;
    }
    1.02 * lam.sqrt()
}

fn max_diag(s: &FftStage) -> f64 {
    s.diags.iter().flat_map(|d| d.iter()).fold(0f64, |m, z| m.max(z.abs()))
}

impl FbsNoiseModel {
    /// The terms of `plan` (its drop-0 variant), the encapsulation secret of Hamming weight `h_sparse`.
    pub fn new(plan: &StcFirstPlan, h_sparse: usize) -> Result<FbsNoiseModel> {
        let chain = &plan.dense.chain;
        fbs_chain_check(chain)?;
        let st = ScaleTables::precompute(chain)?;
        let v = plan.dense.variants.iter().find(|v| v.drop == 0).ok_or_else(|| shape("fbs noise", "no drop-0 variant"))?;
        let setup = &plan.dense.setup;
        let n = chain.n as f64;
        let h = 2.0 * n / 3.0;
        let sigma = pc_core::keys::DiscreteGaussian::SIGMA;
        let p: f64 = chain.p.iter().map(|&x| x as f64).product();
        let p_sparse: f64 = chain.p[..chain.p_sparse.max(1)].iter().map(|&x| x as f64).product();
        let digit = chain.digits.iter().map(|d| d.iter().map(|&i| chain.q[i] as f64).product::<f64>()).fold(0f64, f64::max);
        let beta = chain.digits.len() as f64;
        // per coefficient: rescale / ModDown rounding, key-switch noise after ModDown (all digits active: a bound)
        let r_coef = (1.0 + h) / 12.0;
        let ks_coef = beta * n * digit * digit * sigma * sigma / 12.0 / (p * p) + r_coef;
        // per real slot component: N / 2 times the coefficient variance
        let (r_slot, ks_slot) = (n / 2.0 * r_coef, n / 2.0 * ks_coef);
        let f = cts_stc_factors(chain.n, (chain.n / 2) as u32, setup.level_budget, setup.dim1, 1.0, super::stc::scale_dec(&st))?;

        // the split: StC stages s = 0.. in order; the stages a call runs at q0 are the q0 term
        let ns = f.stc.len();
        let fixed = plan.input.len().min(ns);
        // the lookup's coefficient value per input value: the whole StC on random slots, sqrt(N/2) x (coefficient
        // view); the smallest of a few draws (a lower ratio = a larger noise in x units)
        let mut rho = f64::MAX;
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..3 {
            let mut z: Vec<C64> = (0..f.stc[0].len())
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let a = (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    C64::new(a, (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
                })
                .collect();
            let n0: f64 = z.iter().map(|c| c.abs().powi(2)).sum::<f64>().sqrt();
            for s in &f.stc {
                z = s.apply(&z);
            }
            let n1: f64 = z.iter().map(|c| c.abs().powi(2)).sum::<f64>().sqrt();
            rho = rho.min(n1 / (n / 2.0).sqrt() / n0);
        }
        let thin = chain.lone_top().is_some();
        let entry_scale = st.sf(plan.entry_top as i32)?;
        let mut split_w = 0.0;
        let mut acc_split = 0.0;
        for (s, lt) in plan.input.iter().enumerate().take(fixed) {
            // the stage's output scale: a lean chain's stages keep the input's scale but for their factors
            let scale = if thin {
                acc_split += super::stc_first::THIN_SPLIT.get(s).copied().unwrap_or(0.0);
                entry_scale * acc_split.exp2()
            } else {
                st.sf(lt.top as i32 - plan.step as i32)?
            };
            let later = op_norm(&f.stc[s + 1..].iter().collect::<Vec<_>>());
            let fresh = r_coef + f.stc[s].num_diags() as f64 * max_diag(&f.stc[s]).powi(2) * ks_coef;
            split_w += later * fresh.sqrt() / scale;
        }
        let split_x = split_w / rho;
        // q0: the rescale to q0 and the switch to the sparse secret (one digit, q0, over the bottom key's special
        // primes; ModDown rounding under the sparse secret), the switch back after ModRaise (top modulus)
        let q0: f64 = (0..chain.level_unit).map(|i| chain.q[i] as f64).product();
        let encaps = n * q0 * q0 * sigma * sigma / 12.0 / (p_sparse * p_sparse) + (1.0 + h_sparse as f64) / 12.0;
        let q0_u = (r_coef.sqrt() + encaps.sqrt() + ks_coef.sqrt()) / q0;

        // CtS: evaluation order s = budget - 1 .. 0 (`BtsFactors::cts`), output scales at least the smallest sf
        // between EvalMod's input and the raise
        let order: Vec<&FftStage> = f.cts.iter().rev().collect();
        let mut dmin = f64::MAX;
        for t in v.em_in.top..=v.raise_top as i32 {
            if let Ok(x) = st.sf(t) {
                dmin = dmin.min(x);
            }
        }
        let mut cts_y = 0.0;
        for (s, stg) in order.iter().enumerate() {
            let later = op_norm(&order[s + 1..]);
            let fresh = r_slot + stg.num_diags() as f64 * max_diag(stg).powi(2) * ks_slot;
            cts_y += later * fresh.sqrt() / dmin;
        }
        // EvalMod's conjugation (the real / imaginary split) at its input scale
        let conj_y = (ks_slot + r_slot).sqrt() / v.em_in.nf;
        cts_y += conj_y;
        Ok(FbsNoiseModel { st, em_in: v.em_in, split_x, q0_u, cts_y, conj_y, fresh: Fresh { rescale: r_slot, relin: ks_slot } })
    }

    /// The FBS `h` (its EvalMod program from `EvalModPlan::program`) at input values `xs` (input units: phase
    /// x / 2^cf), the caller's input noise std `sigma_x` (input units); outputs in output units (x 2^cf, as the
    /// StC-first output reads them).
    pub fn eval(&self, h: &FbsHost, cf: u32, xs: &[f64], sigma_x: f64) -> Result<FbsNoise> {
        if h.em_in.top != self.em_in.top || h.em_in.nl != self.em_in.nl || h.em_in.nf.to_bits() != self.em_in.nf.to_bits() {
            return Err(shape("fbs noise", format!("the FBS reads {:?}, the model was built for {:?}", h.em_in, self.em_in)));
        }
        self.eval_ps(&h.ps, h.spec.da_its, h.spec.k, cf, xs, sigma_x)
    }

    /// `eval` for an EvalMod plan `ps` (`PsPlan::fbs`, `da_its` double angles, table K `k`) on the model's variant.
    pub fn eval_ps(&self, ps: &pc_shadow::evalmod::PsPlan, da_its: u32, k: f64, cf: u32, xs: &[f64], sigma_x: f64) -> Result<FbsNoise> {
        let h_em_in = self.em_in;
        let (prog, out, _) = super::evalmod::EvalModPlan::<u32>::program(&self.st, h_em_in, ps, da_its)?;
        let g = (cf as f64).exp2();
        let phase_own = self.split_x / g + self.q0_u + self.cts_y * k;
        let phase = phase_own + sigma_x / g;
        let sy = phase / k;
        let nf = h_em_in.nf;
        let pre = noise::prepare(&self.st, &prog.ops, &prog.consts);
        // per (input, I): value, the variance the own phase noise carries, the fresh variance of the chain (EvalMod's
        // rescales and relinearizations, the conjugation's noise)
        let kk = k as i64;
        let mut raw: Vec<Vec<Option<(f64, f64, f64)>>> = Vec::with_capacity(xs.len());
        let mut partner: f64 = 0.0;
        for &x in xs {
            let u = (x / g).rem_euclid(1.0);
            let u = if u >= 0.5 { u - 1.0 } else { u };
            let mut at = Vec::with_capacity(2 * kk as usize + 1);
            for i in -kk..=kk {
                let y = (u + i as f64) / k;
                if y.abs() > 1.0 {
                    at.push(None);
                    continue;
                }
                let l = noise::linearize(&self.st, &prog.ops, &pre, y * nf, 0, self.fresh);
                // d out / d y = dy nf (the input buffer holds y nf)
                let (val, dy, var) = noise::decode(&self.st, l, out.top, out.nl, out.nf, self.fresh);
                let d = dy * nf;
                let fresh = var + (d * self.conj_y).powi(2);
                partner = partner.max(fresh);
                at.push(Some((val * g, (d * sy * g).powi(2), fresh * g * g)));
            }
            raw.push(at);
        }
        // a complex slot carries two lookups whose chains run apart and recombine as re + i im: the imaginary part of
        // the partner chain's fresh noise lands on this lookup (any partner: the largest over the inputs given)
        let partner = partner * g * g;
        let mut points = Vec::with_capacity(xs.len());
        let mut by_i = Vec::with_capacity(xs.len());
        for at in raw {
            let (mut lo, mut hi, mut sg) = (f64::MAX, f64::MIN, 0f64);
            let mut row = Vec::with_capacity(at.len());
            for a in at {
                match a {
                    None => row.push((f64::NAN, f64::NAN)),
                    Some((v, vin, vf)) => {
                        let e = (vin + vf + partner).sqrt();
                        lo = lo.min(v);
                        hi = hi.max(v);
                        sg = sg.max(e);
                        row.push((v, e));
                    }
                }
            }
            points.push(FbsPoint { lo, hi, sigma: sg });
            by_i.push(row);
        }
        Ok(FbsNoise { phase_own, phase, points, by_i })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bts::stc_first::StcFirstSetup;
    use crate::bts::BtsSetup;
    use pc_core::params::{Chain, ChainSpec};

    #[test]
    fn chains_refused() {
        for p in ["n32", "n32a4", "n32sp", "n32sph"] {
            let c = Chain::generate(&ChainSpec::preset(p).unwrap()).unwrap();
            assert!(fbs_chain_check(&c).is_ok(), "{p}");
        }
        for p in ["n64", "n32/lp4b3"] {
            let c = Chain::generate(&ChainSpec::preset(p).unwrap()).unwrap();
            let e = fbs_chain_check(&c).expect_err(p).to_string();
            assert!(e.contains("validated on n32 / n32a4 / n32sph"), "{p}: {e}");
        }
    }

    /// The model on n32 for the degree-16 sign table (phase j / 32): phase noise far below the half-step margin
    /// (1/64), the noise-free outputs on the table at every ModRaise integer, bounded stds.
    #[test]
    fn n32_sign_table() {
        let chain = Chain::generate(&ChainSpec::n32()).unwrap();
        let mut setup = BtsSetup::n32_shipped();
        setup.raise_drops = vec![0];
        let plan = StcFirstPlan::new(&chain, &setup, &StcFirstSetup::n32_shipped()).unwrap();
        let m = FbsNoiseModel::new(&plan, pc_core::keys::SecretKey::SPARSE_H).unwrap();
        // the degree-16 interpolant of sign(j) on s_j = sin(pi j / 16), times 2^-5 (the StC-first output's 2^cf)
        let t = 16usize;
        let mut post = vec![0f64; t + 1];
        for (d, c) in post.iter_mut().enumerate() {
            let mut acc = 0.0;
            for k in 0..=t {
                let w = if k == 0 || k == t { 0.5 } else { 1.0 };
                acc += w * ((t / 2) as f64 - k as f64).signum() * (std::f64::consts::PI * (d * k) as f64 / t as f64).cos();
            }
            *c = 2.0 / t as f64 * acc / 32.0;
        }
        post[t] *= 0.5;
        let ps = pc_shadow::evalmod::PsPlan::fbs(&pc_core::bts_setup::EVALMOD_K32_CHEB, pc_core::bts_setup::EVALMOD_K32_DA_ITS, 1.0, &post).unwrap();
        let xs: Vec<f64> = (-8..=8).map(|j| j as f64).collect();
        let r = m.eval_ps(&ps, pc_core::bts_setup::EVALMOD_K32_DA_ITS, pc_core::bts_setup::EVALMOD_K32, 5, &xs, 0.0).unwrap();
        assert!(r.phase_own > 0.0 && r.phase_own < 1e-7, "phase {:e}", r.phase_own);
        for (x, p) in xs.iter().zip(&r.points) {
            let want = x.signum();
            assert!((p.lo - want).abs() < 1e-5 && (p.hi - want).abs() < 1e-5, "x {x}: [{}, {}]", p.lo, p.hi);
            assert!(p.sigma > 0.0 && p.sigma < 1e-5, "x {x}: sigma {:e}", p.sigma);
        }
    }
}
