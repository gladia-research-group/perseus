//! The TFHE keys' seeded-mask expander (`mask_expand`, one domain tag per key kind) is frozen: every seeded TFHE key
//! regenerates its masks from it (on the host and in the key switch's kernel, pc-kernels tfhe.cuh KS_REGEN). Pins the
//! keystream at the TFHE moduli; the row form must equal the per-index form and stay in [0, p).
use pc_core::keys::{mask_expand, mask_expand_row, tfhe_tag, KskSeed, MaskSeed};

const SEED: KskSeed = KskSeed([0x0123_4567, 0x89ab_cdef, 0xdead_beef, 0x0bad_f00d, 1, 2, 3, 0xffff_ffff]);

fn fnv(v: &[u32]) -> u64 {
    v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x0000_0100_0000_01b3))
}

/// (tag, lane, p, len): the n32 chain's p0 (LWE key switches, length n_in levels 32768), the bootstrapping primes
/// (length n_br 2048), a 25-bit modulus (the repack columns' P), the ring-switch length 2^16.
const CASES: [(u32, u32, u32, usize); 7] = [
    (tfhe_tag::LWE_KSK, 0, 268_042_241, 32768),
    (tfhe_tag::CHAIN_KSK, 839, 268_042_241, 32768),
    (tfhe_tag::BSK, 5, 67_104_769, 2048),
    (tfhe_tag::BSK, 20831, 67_084_289, 2048),
    (tfhe_tag::PACK, 7, 67_104_769, 2048),
    (tfhe_tag::REPACK_COLS, 3, 33_538_049, 2048),
    (tfhe_tag::RING_SWITCH, 1, 268_042_241, 65536),
];

#[test]
fn tfhe_mask_row_equals_index_form_and_is_pinned() {
    let mut got = Vec::new();
    for &(tag, lane, p, len) in &CASES {
        let seed = MaskSeed { seed: SEED, tag };
        let mut row = vec![0u32; len];
        mask_expand_row(&seed, lane, p, &mut row);
        assert!(row.iter().all(|&x| x < p), "tag {tag:x} lane {lane}");
        for i in (0..len).step_by(97).chain(len - 16..len) {
            assert_eq!(row[i], mask_expand(&seed, lane, p, i as u32, len), "tag {tag:x} lane {lane} index {i}");
        }
        got.push((row[0], row[1], row[len - 1], fnv(&row)));
    }
    assert_eq!(got, PINNED.to_vec());
}

#[test]
fn tfhe_mask_tags_are_distinct() {
    let tags = [tfhe_tag::LWE_KSK, tfhe_tag::CHAIN_KSK, tfhe_tag::BSK, tfhe_tag::PACK, tfhe_tag::RING_SWITCH, tfhe_tag::REPACK_COLS, tfhe_tag::REPACK_RING];
    let ckks = [pc_core::keys::KSKA_DOMAIN, pc_core::keys::KSKB_DOMAIN, pc_core::keys::CTS_DOMAIN];
    for (i, a) in tags.iter().enumerate() {
        assert!(!ckks.contains(a));
        assert!(tags[i + 1..].iter().all(|b| b != a));
    }
    // a different tag, lane or modulus is a different stream
    let (s1, s2) = (MaskSeed { seed: SEED, tag: tfhe_tag::LWE_KSK }, MaskSeed { seed: SEED, tag: tfhe_tag::CHAIN_KSK });
    let (mut a, mut b, mut c) = (vec![0u32; 64], vec![0u32; 64], vec![0u32; 64]);
    mask_expand_row(&s1, 3, 268_042_241, &mut a);
    mask_expand_row(&s2, 3, 268_042_241, &mut b);
    mask_expand_row(&s1, 4, 268_042_241, &mut c);
    assert_ne!(a, b);
    assert_ne!(a, c);
}

/// (row[0], row[1], row[len - 1], FNV-1a of the row) per case, from this implementation at its freeze (2026-11-22).
const PINNED: [(u32, u32, u32, u64); 7] = [
    (256406397, 35585794, 103215388, 10495060758099797212),
    (185633497, 152877236, 223899361, 18401007052261890452),
    (62765577, 22768456, 49429844, 7647921995758153554),
    (13718255, 3582460, 24146624, 2805493244166676458),
    (65041084, 25212742, 19952697, 895160167185477043),
    (29565916, 2934517, 1738645, 14069534940155916975),
    (77721940, 174035979, 50864917, 16764929928790643408),
];
