//! The whole switch on a degree-4096 CKKS ring (S = 2: the extracted LWEs have the production dimension 2048): values
//! at phase m q0 / 4 go through the sign LUT and come back as (1 + sign m) / 2 at q0 / 2^out_cf, zeros elsewhere.
use pc_core::keys::{sample_ternary, DiscreteGaussian, Prng};
use pc_tfhe::lwe::Lwe;
use pc_tfhe::modq::{prime_below, Ntt};
use pc_tfhe::params::Params;
use pc_tfhe::pbs::{sign01, test_vector};
use pc_tfhe::rlwe::RnsRlwe;
use pc_tfhe::switch::Switcher;

const N: usize = 4096;

fn setup(seed: u32) -> (Switcher, Vec<i64>, Prng) {
    let mut prng = Prng::from_seed([seed, 1, 2, 3, 4, 5, 6, 7], 0);
    let p0 = prime_below(31, 1 << 17).unwrap();
    let p1 = prime_below(28, 1 << 17).unwrap();
    let s: Vec<i64> = sample_ternary(&mut prng, N).into_iter().map(|x| x as i64).collect();
    let sw = Switcher::generate(Params::new(N).unwrap(), &s, p0, p1, &mut prng).unwrap();
    (sw, s, prng)
}

#[test]
fn pbs_sign_on_fresh_lwes() {
    use pc_tfhe::gadget::Gadget;
    use pc_tfhe::pbs::BootstrapKey;
    let mut prng = Prng::from_seed([11, 0, 0, 0, 0, 0, 0, 0], 0);
    let p = Params::new(N).unwrap();
    let p0 = prime_below(31, 1 << 17).unwrap();
    let s_lwe: Vec<i64> = (0..p.n_lwe).map(|_| (prng.next_u32() & 1) as i64).collect();
    let s_br: Vec<i64> = sample_ternary(&mut prng, p.n_br).into_iter().map(|x| x as i64).collect();
    let dg = DiscreteGaussian::new(p.key_sigma).unwrap();
    let dg_lwe = DiscreteGaussian::new(p.lwe_sigma_rel * p0 as f64).unwrap();
    let bsk = BootstrapKey::generate(&s_lwe, &s_br, Gadget::new(p.q_br, p.pbs, false), Ntt::new(p.q_br, p.n_br).unwrap(), &dg, &mut prng);
    let delta = p.q_br >> p.out_cf;
    let (tv, off) = sign01(p.n_br, delta);
    let ms: Vec<f64> = (0..24).map(|i| -0.95 + 1.9 * i as f64 / 23.0).collect();
    let lwes: Vec<Lwe> = ms
        .iter()
        .map(|&m| Lwe::encrypt(((m * p0 as f64 / 4.0).round() as i64).rem_euclid(p0 as i64) as u64, &s_lwe, p0, &dg_lwe, &mut prng))
        .collect();
    let outs = bsk.bootstrap_many(&tv, off, &lwes, p0);
    let mut worst = 0f64;
    for (m, o) in ms.iter().zip(outs) {
        let ph = pc_tfhe::modq::centered(o.phase(&s_br, p.q_br), p.q_br) as f64 / delta as f64;
        let want = if *m >= 0.0 { 1.0 } else { 0.0 };
        worst = worst.max((ph - want).abs());
        assert!((ph - want).abs() < 1e-3, "m {m}: want {want} got {ph}");
    }
    eprintln!("pbs_sign_on_fresh_lwes: max output error {worst:.3e} (units of delta)");
}

#[test]
fn switch_sign_end_to_end() {
    let (sw, s, mut prng) = setup(7);
    let (p0, p1) = (sw.p0, sw.p1);
    let q0 = p0 as i128 * p1 as i128;
    let dg = DiscreteGaussian::new(3.19).unwrap();
    // values everywhere (the switch must ignore the invalid ones); valid: 48 spread coefficients
    let m: Vec<f64> = (0..N).map(|_| prng.uniform_f64() * 1.8 - 0.9).collect();
    let valid: Vec<usize> = (0..48).map(|i| i * 83 + 5).collect();
    let msg: Vec<i128> = m.iter().map(|&v| (v * q0 as f64 / 4.0).round() as i128).collect();
    let ct = RnsRlwe::encrypt(&msg, &s, &sw.ntt_p, &dg, &mut prng);
    let (tv, off) = sign01(sw.params.n_br, sw.delta_br());
    let out = sw.apply(&ct, &valid, &tv, off);
    let ph = out.phase(&s, &sw.ntt_p);
    let unit = q0 as f64 / (1u64 << sw.params.out_cf) as f64;
    let mut worst = 0f64;
    for j in 0..N {
        // the input reaches the blind rotation with ~1% of its range in noise: no sign is promised inside that band
        if valid.contains(&j) && m[j].abs() < 0.02 {
            continue;
        }
        let want = if valid.contains(&j) { if m[j] >= 0.0 { 1.0 } else { 0.0 } } else { 0.0 };
        let got = ph[j] as f64 / unit;
        worst = worst.max((got - want).abs());
        assert!((got - want).abs() < 1e-3, "coefficient {j}: m {} want {want} got {got}", m[j]);
    }
    eprintln!("switch_sign_end_to_end: max |out - (1 + sign) / 2| = {worst:.3e} (units of q0 / 2^{})", sw.params.out_cf);
}

#[test]
fn lut_general() {
    let (sw, s, mut prng) = setup(3);
    let (p0, p1) = (sw.p0, sw.p1);
    let q0 = p0 as i128 * p1 as i128;
    let dg = DiscreteGaussian::new(3.19).unwrap();
    let m: Vec<f64> = (0..N).map(|_| prng.uniform_f64() * 1.8 - 0.9).collect();
    let valid: Vec<usize> = (0..16).map(|i| i * 251 + 1).collect();
    let msg: Vec<i128> = m.iter().map(|&v| (v * q0 as f64 / 4.0).round() as i128).collect();
    let ct = RnsRlwe::encrypt(&msg, &s, &sw.ntt_p, &dg, &mut prng);
    // f(m) = max(m, 0)^2 / 2 at the output scale (any function on [-1, 1))
    let d = sw.delta_br() as f64;
    let tv = test_vector(sw.params.n_br, sw.params.q_br, |x| ((x.max(0.0).powi(2) / 2.0) * d).round() as u64);
    let out = sw.apply(&ct, &valid, &tv, 0);
    let ph = out.phase(&s, &sw.ntt_p);
    let unit = q0 as f64 / (1u64 << sw.params.out_cf) as f64;
    let mut worst = 0f64;
    for &j in &valid {
        let want = m[j].max(0.0).powi(2) / 2.0;
        let got = ph[j] as f64 / unit;
        // f' <= 1 on [-1, 1): the output error is the input's (key switch + mod switch noise, ~1% of the range)
        worst = worst.max((got - want).abs());
        assert!((got - want).abs() < 0.03, "j {j} m {} want {want} got {got}", m[j]);
    }
    eprintln!("lut_general: max |out - f(m)| = {worst:.3e}");
}
