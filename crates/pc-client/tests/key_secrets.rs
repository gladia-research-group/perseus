//! The secret behind every new switching-key role, checked from the keys themselves: for each digit d and stored prime
//! t of a generated key, e = b + a s_new - [t in digit d] G_t s_old (mod t) must be the key's small error (discrete
//! Gaussian, sigma 3.19), with s_new, s_old and the gadget G computed here from the secret's coefficients, independently
//! of the keygen's helpers. So the key's samples are RLWE samples under s_new, clean (no s_old term) mod every prime
//! outside digit d, at the modulus of the primes the key stores.
use pc_client::ntt::Ring;
use pc_client::spec::{chain_of, params_hash, KeyReq, KeyShape, KeySpec, Role, LEVEL_RELIN_CAP};
use pc_client::{wire, Client, ClientSecret, Entry};
use pc_core::keys::{kska_expand_row, KskSeed, Prng};
use pc_core::modmul::mulmod;

/// a(X) -> a(X^k) mod X^N + 1 on signed coefficients.
fn automorph(c: &[i8], k: u64) -> Vec<i64> {
    let n = c.len() as u64;
    let mut out = vec![0i64; c.len()];
    for (i, &x) in c.iter().enumerate() {
        let j = (i as u64 * k) % (2 * n);
        if j < n {
            out[j as usize] += x as i64;
        } else {
            out[(j - n) as usize] -= x as i64;
        }
    }
    out
}

/// k^-1 mod m (extended Euclid).
fn inv_mod(k: u64, m: u64) -> u64 {
    let (mut a, mut b, mut x0, mut x1) = (k as i128, m as i128, 1i128, 0i128);
    while b != 0 {
        let q = a / b;
        (a, b) = (b, a - q * b);
        (x0, x1) = (x1, x0 - q * x1);
    }
    assert_eq!(a, 1, "{k} is not invertible mod {m}");
    x0.rem_euclid(m as i128) as u64
}

fn bits(ps: &[u64]) -> f64 {
    ps.iter().map(|&p| (p as f64).log2()).sum()
}

/// What a role's samples are claimed to be: s_new, s_old (coefficients), s_old squared, the gadget's special primes
/// (prime ids) and the digit partition (Q prime ids).
struct Claim {
    s_new: Vec<i64>,
    s_old: Vec<i64>,
    sq: bool,
    specials: Vec<usize>,
    digits: Vec<Vec<usize>>,
    sparse_new: bool,
}

#[test]
fn new_roles_are_samples_under_the_declared_secret() {
    let chain = chain_of("n32a4").unwrap();
    let n = chain.n;
    let nq = chain.q.len();
    let primes = chain.primes();
    let secret = ClientSecret::generate(&mut Prng::from_seed([7; 8], 0), n).unwrap();
    let (s, sp) = (secret.sk.s.clone(), secret.sk.s_sparse.clone());
    // the secrets' distributions: s uniform ternary (dense), s_sparse ternary of Hamming weight 32
    assert!(s.iter().all(|&x| (-1..=1).contains(&x)) && s.iter().filter(|&&x| x != 0).count() > n / 2);
    assert!(sp.iter().all(|&x| (-1..=1).contains(&x)) && sp.iter().filter(|&&x| x != 0).count() == 32);
    let mut c = Client::new("n32a4", secret).unwrap();
    let full = KeyShape::full(&chain);
    let rot = |st: i64| Role::rot(st, n).unwrap();
    let mut keys = vec![
        KeyReq { role: Role::EncDenseToSparse, shape: KeyShape::bottom(&chain).unwrap() },
        KeyReq { role: Role::SparseRot { step: 2048, galois: pc_core::galois::rotation_galois(2048, n).unwrap() }, shape: full },
        KeyReq { role: Role::SparseRot { step: 512, galois: pc_core::galois::rotation_galois(512, n).unwrap() }, shape: full },
        KeyReq { role: rot(1), shape: KeyShape::stc_ps_kp(&chain, 4, 6).unwrap() },
        KeyReq { role: rot(1024), shape: KeyShape::stc_ps_kp(&chain, 4, 6).unwrap() },
    ];
    let level_tops: Vec<usize> = (16..=46).step_by(2).filter(|&t| chain.level_relin_layout(t, LEVEL_RELIN_CAP).is_some()).collect();
    assert!(!level_tops.is_empty());
    for &t in &level_tops {
        let (dg, _, q_top) = chain.level_relin_layout(t, LEVEL_RELIN_CAP).unwrap();
        keys.push(KeyReq { role: Role::RelinLevel { top: t }, shape: KeyShape { q_top, ndig: dg.len(), nq, kp: chain.p.len() } });
    }
    let spec = KeySpec { chain: "n32a4".into(), params_hash: params_hash(&chain), contract: "test".into(), keys, spru: None, pcmm_t: vec![], ccmm: false, public_key: false };
    let dir = std::env::temp_dir().join(format!("pc_client_key_secrets_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("b.pckb");
    c.keygen(&spec, std::io::BufWriter::new(std::fs::File::create(&path).unwrap()), &mut Prng::from_seed([9; 8], 0), 8).unwrap();
    let ring = Ring::new(&chain).unwrap();
    let mut r = wire::BundleReader::open(&path).unwrap();
    let (wbits, rw) = (r.header.bits, r.header.row_words);
    let m2 = 2 * n as u64;
    let mut checked = 0;
    for i in 0..r.header.entries.len() {
        let Entry::Ksk { key, seed, .. } = r.header.entries[i].clone() else { continue };
        let packed = r.read(i).unwrap();
        let sh = key.shape;
        let pfx = |k: usize| (nq..nq + k).collect::<Vec<usize>>();
        let claim = match key.role {
            Role::EncDenseToSparse => Claim { s_new: sp.iter().map(|&x| x as i64).collect(), s_old: s.iter().map(|&x| x as i64).collect(), sq: false, specials: pfx(sh.kp), digits: chain.digits.clone(), sparse_new: true },
            Role::SparseRot { galois, .. } => Claim { s_new: automorph(&s, inv_mod(galois, m2)), s_old: sp.iter().map(|&x| x as i64).collect(), sq: false, specials: pfx(sh.kp), digits: chain.digits.clone(), sparse_new: false },
            Role::Rot { galois, .. } => Claim { s_new: automorph(&s, inv_mod(galois, m2)), s_old: s.iter().map(|&x| x as i64).collect(), sq: false, specials: pfx(sh.kp), digits: chain.digits.clone(), sparse_new: false },
            Role::RelinLevel { top } => {
                let (dg, spc, _) = chain.level_relin_layout(top, LEVEL_RELIN_CAP).unwrap();
                Claim { s_new: s.iter().map(|&x| x as i64).collect(), s_old: s.iter().map(|&x| x as i64).collect(), sq: true, specials: spc, digits: dg, sparse_new: false }
            }
            other => panic!("unexpected role {other:?}"),
        };
        // the modulus of the stored primes; a sparse s_new only at the bottom shape (q0 q1 x P', the record's estimate)
        let ids = sh.row_ids();
        let lq = bits(&ids.iter().map(|&t| primes[t]).collect::<Vec<_>>());
        if claim.sparse_new {
            assert!(sh.q_top <= 1 && sh.kp <= chain.p_sparse, "{:?}: a sparse secret above the bottom shape", key.role);
        } else {
            assert!(lq <= 1734.5, "{:?}: dense samples at log modulus {lq:.1}", key.role);
        }
        let rows = ids.len();
        for d in 0..sh.ndig {
            for (ri, &t) in ids.iter().enumerate() {
                let p = primes[t];
                let mut b = vec![0u32; n];
                wire::unpack_row(&packed[(d * rows + ri) * rw..(d * rows + ri + 1) * rw], wbits, &mut b);
                let mut a = vec![0u32; n];
                kska_expand_row(&KskSeed(seed), d as u32, p as u32, &mut a);
                let sn = ring.eval_row(&claim.s_new, t);
                let so = ring.eval_row(&claim.s_old, t);
                let in_digit = t < nq && claim.digits.get(d).is_some_and(|g| g.contains(&t));
                let g = claim.specials.iter().fold(1u64 % p, |acc, &x| mulmod(acc, primes[x] % p, p));
                let mut e: Vec<u32> = (0..n)
                    .map(|j| {
                        let mut v = (b[j] as u64 + mulmod(a[j] as u64, sn[j] as u64, p)) % p;
                        if in_digit {
                            let x = if claim.sq { mulmod(so[j] as u64, so[j] as u64, p) } else { so[j] as u64 };
                            v = (v + p - mulmod(g, x, p)) % p;
                        }
                        v as u32
                    })
                    .collect();
                ring.ntt[t].inv(&mut e);
                let mx = e.iter().map(|&x| if x as u64 > p / 2 { p - x as u64 } else { x as u64 }).max().unwrap();
                assert!(mx > 0 && mx <= 45, "{:?} digit {d} prime id {t}: max |e| {mx}", key.role);
            }
        }
        checked += 1;
    }
    assert_eq!(checked, spec.keys.len());
    std::fs::remove_dir_all(&dir).ok();
}
