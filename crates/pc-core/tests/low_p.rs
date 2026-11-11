//! `LowP` (preset suffix `/lp<kp>b<top>`): key switching at the bottom limb tops over the first special primes only.
use pc_core::params::{Chain, ChainSpec, LowP};

#[test]
fn low_p_preset() {
    let base = Chain::generate(&ChainSpec::preset("n32").unwrap()).unwrap();
    let spec = ChainSpec::preset("n32/lp4b3").unwrap();
    assert_eq!(spec.low_p, Some(LowP { kp: 4, top: 3, under: None }));
    let c = Chain::generate(&spec).unwrap();
    // the same primes, digits and log QP (the HE-standard bound is unchanged); P' = the first 4 special primes
    assert_eq!((&c.q, &c.p, &c.digits, c.log_qp), (&base.q, &base.p, &base.digits, base.log_qp));
    assert_eq!((c.p_sparse, &c.low), (4, &vec![(3, 4)]));
    assert!(base.low.is_empty());
    for top in 0..c.q.len() {
        assert_eq!(c.specials_at(top), if top <= 3 { 4 } else { c.p.len() }, "top {top}");
        assert_eq!(base.specials_at(top), base.p.len());
    }
    // P' must cover the live digit at the low tops (3 primes of 28 bits do not cover q0 q1 q2 q3)
    assert!(Chain::generate(&ChainSpec::preset("n32/lp3b3").unwrap()).is_err());
    assert!(Chain::generate(&ChainSpec::preset("n32/lp4b9").unwrap()).is_err());
    assert!(ChainSpec::preset("n32/lp4").is_err() && ChainSpec::preset("n32/x").is_err());
}

#[test]
fn low_p_two_tiers() {
    let base = Chain::generate(&ChainSpec::preset("n32").unwrap()).unwrap();
    let spec = ChainSpec::preset("n32/lp4b3/lp2b1").unwrap();
    assert_eq!(spec.low_p, Some(LowP { kp: 4, top: 3, under: Some((1, 2)) }));
    let c = Chain::generate(&spec).unwrap();
    assert_eq!((&c.q, &c.p, &c.digits, c.log_qp), (&base.q, &base.p, &base.digits, base.log_qp));
    // tops 0..=1 over 2 special primes (P' of the bottom keys), 2..=3 over 4, the rest over P
    assert_eq!((c.p_sparse, &c.low), (2, &vec![(1, 2), (3, 4)]));
    for top in 0..c.q.len() {
        assert_eq!(c.specials_at(top), if top <= 1 { 2 } else if top <= 3 { 4 } else { c.p.len() }, "top {top}");
        assert_eq!(c.band_specials(top), if top <= 1 { Some(2) } else if top <= 3 { Some(4) } else { None });
    }
    // the second tier sits below the first, and covers its live digit
    assert!(Chain::generate(&ChainSpec::preset("n32/lp4b3/lp4b1").unwrap()).is_err());
    assert!(Chain::generate(&ChainSpec::preset("n32/lp4b3/lp2b3").unwrap()).is_err());
    assert!(Chain::generate(&ChainSpec::preset("n32/lp4b3/lp1b1").unwrap()).is_err());
    assert!(ChainSpec::preset("n32/lp4b3/lp2").is_err());
}
