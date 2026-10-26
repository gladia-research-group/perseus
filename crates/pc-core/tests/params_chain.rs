//! Prime search on both presets: the n32 chain equals the d3 dump's, the n64 chain is a valid chain.
use pc_core::params::{max_log_qp, nbtheory, Chain, ChainSpec};

const D3: &str = "/data02/users/mithrillo/projects/perseus-oracle/fx/dumps/d3/params.json";

fn check_valid(c: &Chain) {
    let m = 2 * c.n as u64;
    let all = c.primes();
    let mut s = all.clone();
    s.sort();
    s.dedup();
    assert_eq!(s.len(), all.len(), "duplicate primes");
    for (&p, &r) in all.iter().zip(c.roots().iter()) {
        assert!(nbtheory::is_prime(p));
        assert_eq!(p % m, 1, "prime {p} not NTT-friendly");
        assert_eq!(pc_core::modmul::powmod(r, c.n as u64, p), p - 1, "root of {p} not primitive 2N-th");
    }
    assert!(c.log_qp <= max_log_qp(c.spec.security, c.n as u32).unwrap());
    assert_eq!(c.groups.iter().sum::<usize>(), c.q.len());
    assert_eq!(c.digits.iter().map(|d| d.len()).sum::<usize>(), c.q.len());
}

#[test]
fn n32_chain() {
    let c = Chain::generate(&ChainSpec::n32()).unwrap();
    check_valid(&c);
    assert_eq!(c.q.len(), 54);
    assert_eq!(c.p.len(), 9);
    assert_eq!((c.groups.len(), c.p_sparse, c.lone_top()), (27, 9, None));
    assert!(c.q.iter().chain(&c.p).all(|&p| p < 1 << 28));
    println!("n32: logQP = {}", c.log_qp);
    match std::fs::read_to_string(D3) {
        Ok(s) => {
            let j: serde_json::Value = serde_json::from_str(&s).unwrap();
            let v = |k: &str| j[k].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect::<Vec<u64>>();
            assert_eq!(c.q, v("q"));
            assert_eq!(c.p, v("p"));
            assert_eq!(c.roots(), v("const_root"));
            assert_eq!(c.inv_roots(), v("const_inv_root"));
        }
        Err(_) => println!("SKIP d3 comparison: {D3} not readable"),
    }
}

/// n32a4: n32 minus its primes 5..=7, every level above the lean bottom n32's own (primes, scales), the single level
/// above q0, P = n32's P first (the sparse keys' modulus q0 x P' unchanged) and covering the widest dnum-4 digit.
#[test]
fn n32a4_chain() {
    let b = Chain::generate(&ChainSpec::n32()).unwrap();
    let c = Chain::generate(&ChainSpec::n32a4()).unwrap();
    check_valid(&c);
    assert_eq!(c.q.len(), 51);
    assert_eq!(c.q[..5], b.q[..5]);
    assert_eq!(c.q[5..], b.q[8..]);
    assert_eq!(c.dnum(), 4);
    assert_eq!(c.p[..9], b.p[..]);
    assert_eq!(c.p_sparse, 9);
    assert!(c.q.iter().chain(&c.p).all(|&p| p < 1 << 28));
    let lg = |xs: &[u64]| xs.iter().map(|&x| (x as f64).log2()).sum::<f64>();
    let widest = c.digits.iter().map(|d| lg(&d.iter().map(|&i| c.q[i]).collect::<Vec<_>>())).fold(0.0, f64::max);
    assert!(lg(&c.p) >= widest);
    assert_eq!(c.groups[..3], [2, 1, 2]);
    assert_eq!(c.groups.len(), 26);
    assert_eq!((c.lone_top(), c.drop_at(2), c.drop_at(1), c.drop_at(4), c.drop_at(3)), (Some(2), Some(1), Some(2), Some(2), None));
    let (sb, sc) = (pc_core::scale::ScaleTables::precompute(&b).unwrap(), pc_core::scale::ScaleTables::precompute(&c).unwrap());
    for t in (4..=50).step_by(2) {
        assert_eq!(sc.sf(t).unwrap().to_bits(), sb.sf(t + 3).unwrap().to_bits(), "scale at top {t}");
    }
    assert!(sc.sf(2).is_ok() && sc.sf(3).is_err() && sc.sf(0).is_err());
    assert_eq!(sc.mod_reduce_product(2), c.q[2] as f64);
    println!("n32a4: {} Q + {} P primes (P' {}), logQP = {}", c.q.len(), c.p.len(), c.p_sparse, c.log_qp);
}

#[test]
fn n64_chain() {
    let c = Chain::generate(&ChainSpec::n64()).unwrap();
    check_valid(&c);
    assert_eq!(nbtheory::bit_length(c.q[0]), 60);
    for &q in &c.q[1..] {
        assert!((q as f64).log2() > 52.0 && (q as f64).log2() < 54.0);
    }
    assert!(c.p.iter().all(|&p| nbtheory::bit_length(p) == 60));
    println!("n64: {} Q + {} P primes, logQP = {}", c.q.len(), c.p.len(), c.log_qp);
}
