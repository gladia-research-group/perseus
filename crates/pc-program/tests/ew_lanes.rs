//! `schedule::unit_lanes`: a fused plaintext product and add takes one lane (compute when either step is there); an
//! element-wise group (affine, a product by i and its add) keeps its steps' own lanes, so a group whose steps sit on two
//! lanes issues step by step and never pulls a side step onto compute, where compute would wait for the side lane.
#![cfg(feature = "gpu")]
use pc_program::schedule::unit_lanes;

#[test]
fn product_add_pairs_take_one_lane() {
    let mut lane = vec![1, 0, 1, 1];
    assert_eq!(unit_lanes(&mut lane, &[(0, 2, true), (2, 4, true)]), 0);
    assert_eq!(lane, [0, 0, 1, 1]);
}

#[test]
fn element_wise_groups_keep_their_steps_lanes() {
    let mut lane = vec![0, 1, 1, 1, 1, 0, 0];
    // a mixed group (steps 0-2) stays split, a side group (3-4) and a compute group (5-6) are whole
    assert_eq!(unit_lanes(&mut lane, &[(0, 3, false), (3, 5, false), (5, 7, false)]), 1);
    assert_eq!(lane, [0, 1, 1, 1, 1, 0, 0]);
}

#[test]
fn spans_past_the_steps_are_ignored() {
    let mut lane = vec![1, 0];
    assert_eq!(unit_lanes(&mut lane, &[(1, 3, true), (0, 9, false)]), 0);
    assert_eq!(lane, [1, 0]);
}
