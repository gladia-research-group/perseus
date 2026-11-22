//! Seeded key masks: the keys' masks are `pc_core::keys::mask_expand` rows, the keys decrypt as the stored form's,
//! and every operation on a seeded key is bit-identical to the same key's stored form (the same expanded masks).
use pc_core::keys::{mask_expand, sample_ternary, tfhe_tag, DiscreteGaussian, Prng};
use pc_tfhe::gadget::Gadget;
use pc_tfhe::lwe::{Lwe, LweKsk};
use pc_tfhe::modq::{centered, from_signed, mulmod, prime_below};
use pc_tfhe::pack::{glwe_phase, PackingKey, PACK_GADGET};
use pc_tfhe::params::Params;
use pc_tfhe::pbs::{BootstrapKey, Rns};

fn ternary(prng: &mut Prng, n: usize) -> Vec<i64> {
    sample_ternary(prng, n).into_iter().map(|x| x as i64).collect()
}

#[test]
fn seeded_lwe_ksk() {
    let mut prng = Prng::from_seed([31, 0, 0, 0, 0, 0, 0, 0], 0);
    let p0 = prime_below(28, 1 << 17).unwrap();
    let (n_in, n_out) = (128, 96);
    let s_in = ternary(&mut prng, n_in);
    let s_out: Vec<i64> = (0..n_out).map(|_| (prng.next_u32() & 1) as i64).collect();
    let dg = DiscreteGaussian::new(4.0).unwrap();
    let g = Gadget::new(p0, (1, 16), false);
    let ksk = LweKsk::generate_seeded(&s_in, &s_out, g.clone(), tfhe_tag::LWE_KSK, &dg, &mut prng);
    let seed = *ksk.seed().unwrap();
    let k = ksk.rows();
    // the masks are the seed's rows (column o, index r), the rows decrypt to g_t s_i + e
    for (o, r) in [(0, 0), (5, 17), (95, k - 1), (40, 1000)] {
        assert_eq!(ksk.key(r / 16, r % 16).a[o], mask_expand(&seed, o as u32, p0 as u32, r as u32, k) as u64);
    }
    for i in 0..n_in {
        for t in 0..16 {
            let want = mulmod(g.g(t), from_signed(s_in[i], p0), p0);
            let e = centered((ksk.key(i, t).phase(&s_out, p0) + p0 - want) % p0, p0);
            assert!(e.abs() <= 30, "row ({i}, {t}): error {e}");
        }
    }
    assert!(ksk.stored_bytes() < 4 * k + 64);
    // the key switch: identical to the stored form's
    let st = ksk.to_stored();
    for _ in 0..4 {
        let ct = Lwe { a: (0..n_in).map(|_| prng.uniform_below(p0)).collect(), b: prng.uniform_below(p0) };
        assert_eq!(ksk.apply(&ct), st.apply(&ct));
    }
}

#[test]
fn seeded_bootstrap_and_packing_keys() {
    let mut prng = Prng::from_seed([32, 0, 0, 0, 0, 0, 0, 0], 0);
    let p = Params::new(4096).unwrap();
    let p0 = prime_below(28, 1 << 17).unwrap();
    let q = p.q_br;
    let n_lwe = 24;
    let s_lwe: Vec<i64> = (0..n_lwe).map(|_| (prng.next_u32() & 1) as i64).collect();
    let s_br = ternary(&mut prng, p.n_br);
    let dg = DiscreteGaussian::new(p.bsk_sigma).unwrap();
    let rns = Rns::new(p.br_primes.0, p.br_primes.1, p.n_br).unwrap();
    let bsk = BootstrapKey::generate_seeded(&s_lwe, &s_br, Gadget::new(q, p.pbs, false), rns.clone(), 3, tfhe_tag::BSK, &dg, &mut prng);
    let st = bsk.to_stored();
    assert_eq!(2 * bsk.stored_bytes(), st.stored_bytes() + 72);
    // the identity in the native encoding: the outputs decrypt to the inputs, the stored form's bit for bit
    let id = pc_tfhe::chain::native_tv(p.n_br, q, |m| m);
    let dg_in = DiscreteGaussian::new(3.19).unwrap();
    for v in [-0.4, -0.1, 0.0, 0.2, 0.45] {
        let ct = Lwe::encrypt(((v * p0 as f64 / 4.0).round() as i64).rem_euclid(p0 as i64) as u64, &s_lwe, p0, &dg_in, &mut prng);
        let o = bsk.bootstrap(&id, 0, &ct, p0);
        assert_eq!(o, st.bootstrap(&id, 0, &ct, p0));
        let got = centered(o.phase(&s_br, q), q) as f64 / (q as f64 / 4.0);
        assert!((got - v).abs() < 0.01, "identity({v}) = {got}");
    }
    // the packing key: two LWEs at positions 0, 5 decrypt to their values there; the stored form's GLWE bit for bit
    let pk = PackingKey::generate_seeded(&s_lwe, &s_br, PACK_GADGET, rns.clone(), tfhe_tag::PACK, &dg, &mut prng);
    let cts: Vec<Lwe> = [0.3, -0.2].iter().map(|&v: &f64| Lwe::encrypt(((v * p0 as f64 / 4.0).round() as i64).rem_euclid(p0 as i64) as u64, &s_lwe, p0, &dg_in, &mut prng)).collect();
    let g = pk.pack(&cts, p0, &[0, 5], None);
    assert_eq!(g, pk.to_stored().pack(&cts, p0, &[0, 5], None));
    let ph = glwe_phase(&g, &s_br, &rns);
    for (pos, v) in [(0usize, 0.3), (5, -0.2)] {
        let got = ph[pos] as f64 / (q as f64 / 4.0);
        assert!((got - v).abs() < 0.01, "packed {pos}: {got} want {v}");
    }
}
