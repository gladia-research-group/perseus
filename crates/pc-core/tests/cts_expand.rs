//! The regenerable-ciphertext c1 expander: rows equal slot-by-slot draws, values below p, distinct per seed and prime.
use pc_core::keys::{cts_expand, cts_expand_row, kska_expand_row, KskSeed};

const N: usize = 1 << 12;
const P: [u32; 3] = [134_176_769, 268_369_921, 132_120_577];

fn seed(i: u32) -> [u32; 8] {
    std::array::from_fn(|w| (i + 1).wrapping_mul(0x9e37_79b9) ^ (w as u32).wrapping_mul(0x85eb_ca6b))
}

#[test]
fn row_matches_slots() {
    for (i, &p) in P.iter().enumerate() {
        let s = seed(i as u32);
        let mut row = vec![0u32; N];
        cts_expand_row(&s, p, &mut row);
        for slot in 0..N as u32 {
            assert_eq!(row[slot as usize], cts_expand(&s, p, slot, N), "p {p} slot {slot}");
        }
        assert!(row.iter().all(|&x| x < p));
    }
}

#[test]
fn rows_are_distinct() {
    let (mut a, mut b, mut c, mut k) = (vec![0u32; N], vec![0u32; N], vec![0u32; N], vec![0u32; N]);
    cts_expand_row(&seed(0), P[0], &mut a);
    cts_expand_row(&seed(1), P[0], &mut b);
    cts_expand_row(&seed(0), P[2], &mut c);
    kska_expand_row(&KskSeed(seed(0)), 0, P[0], &mut k);
    let same = |x: &[u32], y: &[u32]| x.iter().zip(y).filter(|(u, v)| u == v).count();
    assert!(same(&a, &b) < 4 && same(&a, &c) < 4, "seed and prime separate the streams");
    assert!(same(&a, &k) < 4, "the domain word separates ciphertext rows from KSKA rows");
}

#[test]
fn roughly_uniform() {
    let p = P[1];
    let mut row = vec![0u32; N];
    cts_expand_row(&seed(7), p, &mut row);
    let mean = row.iter().map(|&x| x as f64).sum::<f64>() / N as f64;
    let want = (p as f64 - 1.0) / 2.0;
    assert!((mean - want).abs() < 4.0 * p as f64 / (12f64.sqrt() * (N as f64).sqrt()), "mean {mean} vs {want}");
}
