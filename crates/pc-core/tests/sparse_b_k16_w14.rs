//! `SPARSE_B_K16_W14` against the frozen K = 16 table: the noise-free EvalMod (the Chebyshev series on t / 16, c0
//! halved, then 5 double angles) vs sin(2 pi t) / 2 pi, t within 1/32 of the overflow integer I.
use pc_core::bts_setup::{EVALMOD_CHEB, EVALMOD_DA_ITS, EVALMOD_K, SPARSE_B_K16_W14};

/// The series (evaluator convention: the table's c0 is twice the constant) by Clenshaw, then the double angles.
fn evalmod(coef: &[f64], t: f64) -> f64 {
    let y = t / EVALMOD_K;
    let (mut b1, mut b2) = (0.0, 0.0);
    for &c in coef.iter().skip(1).rev() {
        (b1, b2) = (2.0 * y * b1 - b2 + c, b1);
    }
    let mut v = y * b1 - b2 + 0.5 * coef[0];
    let r = (EVALMOD_DA_ITS as f64).exp2();
    let mut s = (2.0 * std::f64::consts::PI).powf(-1.0 / r);
    for _ in 0..EVALMOD_DA_ITS {
        v = 2.0 * v * v - s * s;
        s *= s;
    }
    v
}

/// log2 of the largest error over t in [I - 1/32, I + 1/32].
fn err_bits(coef: &[f64], i: i32) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    (0..=400)
        .map(|j| {
            let t = i as f64 + (j as f64 / 200.0 - 1.0) / 32.0;
            (evalmod(coef, t) - (two_pi * t).sin() / two_pi).abs()
        })
        .fold(0.0, f64::max)
        .log2()
}

#[test]
fn w14_beats_the_frozen_table_where_the_overflow_integer_lives() {
    for i in -6..=6 {
        let (w, f) = (err_bits(&SPARSE_B_K16_W14, i), err_bits(&EVALMOD_CHEB, i));
        assert!(w <= -35.0, "I = {i}: w14 2^{w:.2}");
        assert!(w <= f + 0.5, "I = {i}: w14 2^{w:.2} vs frozen 2^{f:.2}");
    }
    for i in [-7, 7] {
        assert!(err_bits(&SPARSE_B_K16_W14, i) <= -33.5, "I = {i}");
    }
    for i in [-8, 8] {
        assert!(err_bits(&SPARSE_B_K16_W14, i) <= err_bits(&EVALMOD_CHEB, i), "I = {i}");
    }
    // the frozen table's worst bulk integer (I = 1, P ~ 0.2) is where the gain is
    assert!(err_bits(&SPARSE_B_K16_W14, 1) <= err_bits(&EVALMOD_CHEB, 1) - 8.0);
}

#[test]
fn w14_has_the_frozen_tables_shape() {
    // same degree and every coefficient within 1e-9: the same products and the same noise amplification
    assert_eq!(SPARSE_B_K16_W14.len(), EVALMOD_CHEB.len());
    for (a, b) in SPARSE_B_K16_W14.iter().zip(EVALMOD_CHEB.iter()) {
        assert!((a - b).abs() <= 1e-9, "{a} vs {b}");
    }
}
