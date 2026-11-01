//! The forward switch's noise per stage (ignored: `cargo test --release -p pc-tfhe --test noise_budget -- --ignored
//! --nocapture`): ring switch + extract, LWE key switch, mean-compensated mod switch to 2 n_br, in units of the LUT's
//! input range (phase m q / 4: one unit = q / 4).
use pc_core::keys::{sample_ternary, DiscreteGaussian, Prng};
use pc_tfhe::gadget::{fixed_ratio, ms_body_mean, scale_round, Gadget};
use pc_tfhe::lwe::{Lwe, LweKsk};
use pc_tfhe::modq::{centered, prime_below, Ntt};
use pc_tfhe::params::Params;
use pc_tfhe::rlwe::RnsRlwe;
use pc_tfhe::switch::{rescale_p1, RingSwitchKey};

const N: usize = 4096;

fn rms(v: &[f64]) -> f64 {
    (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
}

#[test]
#[ignore]
fn forward_noise_per_stage() {
    // NOISE_SEEDS key sets (the key switch's error has a key-dependent bias under biased digits: average over keys)
    let seeds: u32 = std::env::var("NOISE_SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    for seed in 0..seeds {
        one_key_set(seed);
    }
}

fn one_key_set(seed: u32) {
    let mut prng = Prng::from_seed([5 + seed, 1, 2, 3, 4, 5, 6, 7], 0);
    let p0 = prime_below(28, 1 << 17).unwrap();
    let p1 = prime_below(28, 1 << 18).unwrap();
    let q0 = p0 as i128 * p1 as i128;
    // overrides for parameter studies: NOISE_NLWE, NOISE_SIGMA_LOG2 (the LWE key's sigma / p0), NOISE_NU (the mod
    // switch to 2 nu n_br: an extended blind rotation in R_(nu n_br))
    let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok());
    let mut params = Params::new(N).unwrap();
    if let Some(v) = env("NOISE_NLWE") {
        params.n_lwe = v as usize;
    }
    if let Some(v) = env("NOISE_SIGMA_LOG2") {
        params.lwe_sigma_rel = v.exp2();
    }
    let nu = env("NOISE_NU").unwrap_or(1.0) as u64;
    let s: Vec<i64> = sample_ternary(&mut prng, N).into_iter().map(|x| x as i64).collect();
    let dg = DiscreteGaussian::new(params.key_sigma).unwrap();
    let dg_lwe = DiscreteGaussian::new(params.lwe_sigma_rel * p0 as f64).unwrap();
    let ntt_p = vec![Ntt::new(p0, N).unwrap(), Ntt::new(p1, N).unwrap()];
    let s_sub: Vec<i64> = sample_ternary(&mut prng, N / params.rs_factor).into_iter().map(|x| x as i64).collect();
    let s_lwe: Vec<i64> = (0..params.n_lwe).map(|_| (prng.next_u32() & 1) as i64).collect();
    let rs = RingSwitchKey::generate(&s, &s_sub, params.rs_factor, Gadget::new(p0, params.rs, false), &ntt_p[0], &dg, &mut prng);
    let ksk = LweKsk::generate(&s_sub, &s_lwe, Gadget::new(p0, params.ks, false), &dg_lwe, &mut prng);
    // the key switching key's own errors e[i][t] = phase - g_t s_i
    let kerr: Vec<Vec<i64>> = ksk.keys.iter().zip(&s_sub).map(|(ks, &si)| ks.iter().enumerate().map(|(t, k)| {
        let want = (centered(ksk.gadget.g(t), p0) as i128 * si as i128).rem_euclid(p0 as i128) as u64;
        centered((k.phase(&s_lwe, p0) + p0 - want) % p0, p0)
    }).collect()).collect();
    let ke: Vec<f64> = kerr.iter().flatten().map(|&e| e as f64).collect();
    eprintln!("key errors: rms {:.1} (sigma_rel p0 = {:.1}), mean {:.2}", rms(&ke), params.lwe_sigma_rel * p0 as f64, ke.iter().sum::<f64>() / ke.len() as f64);
    let (mut e_key, mut dsq) = (vec![], 0f64);
    let dgc = DiscreteGaussian::new(3.19).unwrap();
    let m: Vec<f64> = (0..N).map(|_| prng.uniform_f64() * 1.8 - 0.9).collect();
    let msg: Vec<i128> = m.iter().map(|&v| (v * q0 as f64 / 4.0).round() as i128).collect();
    let ct = RnsRlwe::encrypt(&msg, &s, &ntt_p, &dgc, &mut prng);
    let sw = rs.apply(&rescale_p1(&ct, p0, p1), &ntt_p[0]);
    let two_n = 2 * nu * params.n_br as u64;
    let msm = fixed_ratio(two_n, p0);
    let (mut e_ext, mut e_ks, mut e_ms, mut e_ms_plain, mut e_rnd) = (vec![], vec![], vec![], vec![], vec![]);
    for j in (0..N).step_by(4) {
        let want = m[j] / 4.0;
        let ext = rs.extract(&sw, j, p0);
        let ph = |l: &Lwe, s: &[i64]| centered(l.phase(s, p0), p0) as f64 / p0 as f64;
        let x0 = ph(&ext, &s_sub);
        let ks = ksk.apply(&ext);
        let x1 = ph(&ks, &s_lwe);
        // the key switch's rounding part: sum_i s_i (sum_t d_t g_t - a_i)
        let g = &ksk.gadget;
        let mut d = vec![0i64; g.levels];
        let mut rnd: i128 = 0;
        for (&ai, &si) in ext.a.iter().zip(&s_sub) {
            g.decompose(ai, &mut d);
            let rec: i128 = d.iter().enumerate().map(|(t, &dt)| dt as i128 * centered(g.g(t), p0) as i128).sum();
            let diff = (rec - centered(ai, p0) as i128).rem_euclid(p0 as i128);
            let diff = if diff > p0 as i128 / 2 { diff - p0 as i128 } else { diff };
            rnd += si as i128 * diff;
        }
        let mut kk: i128 = 0;
        for (i, &ai) in ext.a.iter().enumerate() {
            g.decompose(ai, &mut d);
            for (t, &dt) in d.iter().enumerate() {
                kk += dt as i128 * kerr[i][t] as i128;
                dsq += (dt * dt) as f64;
            }
        }
        e_key.push(4.0 * kk as f64 / p0 as f64);
        e_rnd.push(4.0 * rnd as f64 / p0 as f64);
        // the blind rotation's input: a~ = round(a 2 n_br / p0), b~ mean-compensated
        let at: Vec<u64> = ks.a.iter().map(|&x| scale_round(x, msm) % two_n).collect();
        let phase2 = |b: u64| -> f64 {
            let v = at.iter().zip(&s_lwe).fold(b as i64, |acc, (&a, &si)| acc + a as i64 * si).rem_euclid(two_n as i64);
            let c = if v >= two_n as i64 / 2 { v - two_n as i64 } else { v };
            c as f64 / two_n as f64
        };
        let x2 = phase2(ms_body_mean(&ks.a, ks.b, msm, two_n));
        let x2p = phase2(scale_round(ks.b, msm) % two_n);
        e_ext.push(4.0 * (x0 - want));
        e_ks.push(4.0 * (x1 - x0));
        e_ms.push(4.0 * (x2 - x1));
        e_ms_plain.push(4.0 * (x2p - x1));
    }
    eprintln!("key part sum d e: rms {:.3e}, mean sum d^2 per LWE {:.0}", rms(&e_key), dsq / e_key.len() as f64);
    let total: Vec<f64> = e_ext.iter().zip(&e_ks).zip(&e_ms).map(|((a, b), c)| a + b + c).collect();
    eprintln!("n_lwe {} n_br {} nu {nu} ks {:?} sigma_rel 2^{:.1}", params.n_lwe, params.n_br, params.ks, params.lwe_sigma_rel.log2());
    eprintln!("rms (units of the LUT input range): ring switch + extract {:.3e}, LWE key switch {:.3e} (rounding {:.3e}), mod switch {:.3e} (plain {:.3e}), total {:.3e} over {}", rms(&e_ext), rms(&e_ks), rms(&e_rnd), rms(&e_ms), rms(&e_ms_plain), rms(&total), total.len());
}
