//! The moduli (log2) of every key in the security tables, from Perseus's own chain and key code.
//!
//! - main (dense) key: its key-switching keys live at Q P, the whole chain (every Q prime and every special prime);
//! - sparse bootstrapping (EncDenseToSparse) and SPRU (SpruAtob) keys: one digit over q0 (the bottom level) and the
//!   first `p_sparse` special primes P'; error widened by 2^t (`wide_err_shift_of`), sd 2^t sqrt(sigma^2 + 1/12);
//! - PCMM / CCMM keys: the chain's two / four bottom primes and their 31-bit special prime (`find_prime`,
//!   `CcmmKeys::special_prime`);
//! - TFHE: q_br (two 26-bit primes, `Params::new`), the repack's p0 P, the ring switch / LWE keys' p0 (the chain's
//!   first prime).
use pc_core::ccmm_keys::CcmmKeys;
use pc_core::keys::{wide_err_shift_of, DiscreteGaussian};
use pc_core::params::{Chain, ChainSpec};
use pc_core::pcmm_keys::{find_prime, N as PCMM_N, P_BITS as PCMM_P_BITS};

fn lg(xs: &[u64]) -> f64 {
    xs.iter().map(|&x| (x as f64).log2()).sum()
}

fn main() {
    let sigma = DiscreteGaussian::SIGMA;
    println!("chain   logQP(main key)   q0      P'       q0P'     t    log2 sd(wide)  q0 x first 3 specials");
    for name in ["n32", "n32a4", "n32sp", "n32sph", "n64"] {
        let c = Chain::generate(&ChainSpec::preset(name).unwrap()).unwrap();
        let q0 = &c.q[..c.groups[0]];
        let ps = &c.p[..c.p_sparse];
        let t = wide_err_shift_of(&c, c.groups[0] - 1, c.p_sparse);
        let sd = t as f64 + (sigma * sigma + 1.0 / 12.0).sqrt().log2();
        println!(
            "{name:7} {:15.3}   {:6.3}  {:7.3}  {:7.3}  {t:3}  {sd:12.3}  {:.3}",
            lg(&c.q) + lg(&c.p),
            lg(q0),
            lg(ps),
            lg(q0) + lg(ps),
            lg(q0) + lg(&c.p[..3.min(c.p.len())])
        );
    }

    let c = Chain::generate(&ChainSpec::n32()).unwrap();
    let q4 = [c.q[0], c.q[1], c.q[2], c.q[3]];
    let pp = find_prime(PCMM_P_BITS, 2 * PCMM_N as u64, &q4).unwrap();
    let pc = CcmmKeys::special_prime(&c.q).unwrap();
    println!("\nn32 bottom primes q0..q3 = {:?}", q4);
    println!("PCMM (ring 2^12): q0 q1 P = {} x {} x {pp}: {:.5}", q4[0], q4[1], lg(&[q4[0], q4[1], pp]));
    println!("CCMM (ring 2^13): q0..q3 P, P = {pc}: {:.5}", lg(&q4) + (pc as f64).log2());

    let t = pc_tfhe::params::Params::new(1 << 16).unwrap();
    let p0 = c.q[0];
    println!("\nTFHE p0 = {p0} ({:.6})", (p0 as f64).log2());
    println!("TFHE bootstrapping key (GLWE 2^11): q_br = {} x {} = {:.5}", t.br_primes.0, t.br_primes.1, (t.q_br as f64).log2());
    println!("TFHE repack columns (2^11): p0 P = {p0} x {} = {:.5}", t.rp_aux, lg(&[p0, t.rp_aux]));
    println!("TFHE ring switch key (2^11) and LWE key: p0 = {:.5}", (p0 as f64).log2());
    for (n, s) in [(795usize, -17f64), (840, -18.0), (990, -21.0)] {
        println!("TFHE LWE key n {n}, sigma 2^{s} p0 = {:.6}", s.exp2() * p0 as f64);
    }
}
