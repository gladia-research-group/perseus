//! Windows of the split's phase (the digit decomposition's loads, `Switcher::lwes_shifted` / `load_shifted`) on a
//! degree-4096 CKKS ring (S = 2: the extracted LWEs have the production dimension 2048): the phase times 2^k before the
//! switch's rescale reads 2^k m mod 4 at the switch's own precision, for every k; a sign from the windows' statuses
//! (perseus impl/tfhe_digits.py) is exact far below the switch's ~7.5 bits.
use pc_core::keys::{sample_ternary, DiscreteGaussian, Prng};
use pc_tfhe::chain::native_tv;
use pc_tfhe::modq::{centered, prime_below};
use pc_tfhe::params::Params;
use pc_tfhe::rlwe::RnsRlwe;
use pc_tfhe::switch::Switcher;

const N: usize = 4096;

/// (switch, s_lwe, s_br, the encrypted split, the exact phases of `valid` as m = phase / (q0 / 4) in i128 units)
fn setup(seed: u32, ms: &[f64]) -> (Switcher, Vec<i64>, Vec<i64>, RnsRlwe, Vec<usize>, Vec<i128>) {
    let mut prng = Prng::from_seed([seed, 9, 2, 3, 4, 5, 6, 7], 0);
    let p0 = prime_below(28, 1 << 17).unwrap();
    let p1 = prime_below(28, 1 << 18).unwrap();
    let s: Vec<i64> = sample_ternary(&mut prng, N).into_iter().map(|x| x as i64).collect();
    let (sw, s_lwe, s_br) = Switcher::generate_keep(Params::new(N).unwrap(), &s, p0, p1, &mut prng).unwrap();
    let q0 = p0 as i128 * p1 as i128;
    let dg = DiscreteGaussian::new(3.19).unwrap();
    // the values at spread coefficients, noise elsewhere (other slots' values must not leak into the windows)
    let valid: Vec<usize> = (0..ms.len()).map(|i| (i * 61 + 3) % N).collect();
    let mut msg = vec![0i128; N];
    for (j, &v) in valid.iter().zip(ms) {
        msg[*j] = (v * q0 as f64 / 4.0).round() as i128;
    }
    for (j, x) in msg.iter_mut().enumerate() {
        if !valid.contains(&j) {
            *x = ((prng.uniform_f64() * 1.8 - 0.9) * q0 as f64 / 4.0).round() as i128;
        }
    }
    let ct = RnsRlwe::encrypt(&msg, &s, &sw.ntt_p, &dg, &mut prng);
    let ph = ct.phase(&s, &sw.ntt_p);
    let exact: Vec<i128> = valid.iter().map(|&j| ph[j].rem_euclid(q0)).collect();
    (sw, s_lwe, s_br, ct, valid, exact)
}

/// 2^k phase mod q0 as a native value in [-2, 2) (q0 = 4 native units).
fn window(phase: i128, k: u32, q0: i128) -> f64 {
    let w = (phase * (1i128 << k)).rem_euclid(q0);
    let c = if w >= q0 / 2 { w - q0 } else { w };
    4.0 * c as f64 / q0 as f64
}

fn wrap4(x: f64) -> f64 {
    (x + 2.0).rem_euclid(4.0) - 2.0
}

#[test]
fn windows_read_the_shifted_phase() {
    let ms: Vec<f64> = (0..48).map(|i| -0.95 + 1.9 * i as f64 / 47.0 + 1e-3 * (i as f64).sin()).collect();
    let (sw, s_lwe, _s_br, ct, valid, exact) = setup(5, &ms);
    let (p0, q0) = (sw.p0, sw.p0 as i128 * sw.p1 as i128);
    for k in [0u32, 3, 6, 9, 14, 18, 22, 30] {
        let lwes = sw.lwes_shifted(&ct, &valid, k);
        let (mut ss, mut worst) = (0f64, 0f64);
        for (l, &ph) in lwes.iter().zip(&exact) {
            let got = 4.0 * centered(l.phase(&s_lwe, p0), p0) as f64 / p0 as f64;
            let e = wrap4(got - window(ph, k, q0));
            ss += e * e;
            worst = worst.max(e.abs());
        }
        let rms = (ss / lwes.len() as f64).sqrt();
        eprintln!("windows: shift {k:2}: read error rms {rms:.3e} max {worst:.3e} (native units)");
        // the switch's input noise (~5e-3 rms at N 2^16; less on this ring): 10 sigma
        assert!(worst < 0.05, "shift {k}: max read error {worst}");
    }
}

#[test]
fn sign_from_window_statuses() {
    // statuses sign(round(4 2^(3 j) m)) of 6 windows (16 bits of the bound 1.75): the topmost nonzero is sign(m)
    let span = 1.75f64;
    let mut ms: Vec<f64> = vec![0.0, 1e-9, -1e-9, span * 2f64.powi(-17), -span * 2f64.powi(-17), 1.7, -1.7];
    for i in 0..6 {
        for j in [-3.0f64, -1.0, 0.0, 2.0] {
            ms.push((j + 0.5) / (4.0 * 8f64.powi(i)));
        }
    }
    let mut prng = Prng::from_seed([77, 0, 0, 0, 0, 0, 0, 0], 0);
    while ms.len() < 64 {
        let e = (prng.next_u32() % 18) as i32;
        ms.push((prng.uniform_f64() * 2.0 - 1.0) * span * 2f64.powi(-e));
    }
    let (sw, _s_lwe, s_br, ct, valid, exact) = setup(6, &ms);
    let (p0, q0) = (sw.p0, sw.p0 as i128 * sw.p1 as i128);
    let shifts: Vec<u32> = (0..6).map(|j| 3 * j).collect();
    // sign(round(4 m)) / 4 (f64::signum(0) is 1: the zero read apart)
    let tv = native_tv(sw.params.tv_len(), sw.params.q_br, |m| {
        let r = (4.0 * m).round();
        if r == 0.0 { 0.0 } else { r.signum() / 4.0 }
    });
    let outs = sw.load_shifted(&ct, &valid, &shifts, &tv, 0);
    let b = valid.len();
    let res = 0.84 / (4.0 * 8f64.powi(5));
    let mut decided = 0;
    for q in 0..b {
        let m = {
            let c = if exact[q] >= q0 / 2 { exact[q] - q0 } else { exact[q] };
            4.0 * c as f64 / q0 as f64
        };
        let st: Vec<f64> = (0..6).map(|j| 4.0 * centered(outs[j * b + q].phase(&s_br, p0), p0) as f64 / p0 as f64).collect();
        for &v in &st {
            assert!((v - 0.25 * (4.0 * v).round()).abs() < 1e-2 && v.abs() < 0.3, "a status off its grid: {v}");
        }
        let top = st.iter().map(|v| (4.0 * v).round() as i64).find(|&v| v != 0).unwrap_or(0);
        let got = if top >= 0 { 1.0 } else { -1.0 };
        if m.abs() >= res {
            decided += 1;
            assert_eq!(got, m.signum(), "value {m:e} ({q}): statuses {st:?}");
        }
    }
    eprintln!("sign_from_window_statuses: {decided} of {b} values at |m| >= {res:.2e} decided, all exact");
}
