//! A TFHE-native tournament max: x_i (bootstrap outputs, native encoding) -> per round d = x[i + h] - x[i], one
//! many-LUT bootstrap (relu, [d >= 0]) on the chained input, x[i] += relu(d); no CKKS between rounds.
use pc_core::keys::{sample_ternary, DiscreteGaussian, Prng};
use pc_tfhe::chain::{native_tv, ChainKey};
use pc_tfhe::gadget::Gadget;
use pc_tfhe::lwe::Lwe;
use pc_tfhe::modq::{centered, prime_below};
use pc_tfhe::params::Params;
use pc_tfhe::pbs::{interleave, BootstrapKey, Rns};

#[test]
fn native_tournament_max() {
    let mut prng = Prng::from_seed([21, 0, 0, 0, 0, 0, 0, 0], 0);
    let p = Params::new(4096).unwrap();
    let p0 = prime_below(28, 1 << 17).unwrap();
    let q = p.q_br;
    let s_lwe: Vec<i64> = (0..p.n_lwe).map(|_| (prng.next_u32() & 1) as i64).collect();
    let s_br: Vec<i64> = sample_ternary(&mut prng, p.n_br).into_iter().map(|x| x as i64).collect();
    let dg = DiscreteGaussian::new(p.bsk_sigma).unwrap();
    let dg_lwe = DiscreteGaussian::new(p.lwe_sigma_rel * p0 as f64).unwrap();
    let rns = Rns::new(p.br_primes.0, p.br_primes.1, p.n_br).unwrap();
    let bsk = BootstrapKey::generate(&s_lwe, &s_br, Gadget::new(q, p.pbs, false), rns, p.pbs_group, &dg, &mut prng);
    let ck = ChainKey::generate(&s_br, &s_lwe, p0, &p, &mut prng).unwrap();
    let dec = |o: &Lwe| centered(o.phase(&s_br, q), q) as f64 / (q as f64 / 4.0);
    // 32 values in [-0.45, 0.45], fresh at p0 under s_lwe, then one identity LUT each (into the native s_br domain)
    let n = 32;
    let xs: Vec<f64> = (0..n).map(|_| prng.uniform_f64() * 0.9 - 0.45).collect();
    let id = native_tv(p.n_br, q, |m| m);
    let mut x: Vec<Lwe> = xs
        .iter()
        .map(|&v| Lwe::encrypt(((v * p0 as f64 / 4.0).round() as i64).rem_euclid(p0 as i64) as u64, &s_lwe, p0, &dg_lwe, &mut prng))
        .map(|l| bsk.bootstrap(&id, 0, &l, p0))
        .collect();
    let id_err: f64 = x.iter().zip(&xs).map(|(o, &v)| (dec(o) - v).powi(2)).sum::<f64>() / n as f64;
    // relu and [d >= 0] (at 1/2) from one rotation
    let tv = interleave(&[native_tv(p.n_br / 2, q, |m| m.max(0.0)), native_tv(p.n_br / 2, q, |m| if m >= 0.0 { 0.25 } else { -0.25 })]);
    let offs = [0, q / 16];
    let mut h = 1;
    let mut truth = xs.clone();
    let mut wrong = 0;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            let d = x[i + h].sub(&x[i], q);
            let o = bsk.bootstrap_mlut(&tv, &offs, &ck.input(&d), p0, 1);
            x[i] = x[i].add(&o[0], q);
            let s = dec(&o[1]) > 0.25;
            wrong += (s != (truth[i + h] >= truth[i]) && (truth[i + h] - truth[i]).abs() > 0.035) as usize;
            truth[i] = truth[i].max(truth[i + h]);
        }
        h *= 2;
    }
    let got = dec(&x[0]);
    eprintln!("native_tournament_max: identity rms {:.2e}, max {:.4} want {:.4} (err {:.2e}), decisions wrong past 0.035 (~4 sigma of the chained input noise): {wrong}", id_err.sqrt(), got, truth[0], got - truth[0]);
    assert!((got - truth[0]).abs() < 0.03, "max {got} want {}", truth[0]);
    assert_eq!(wrong, 0);
}
