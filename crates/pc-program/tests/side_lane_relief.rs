//! `schedule::relieve`: the side lane is one in-order stream, so a chain the DAG model gives slack can finish after
//! compute reaches its consumer when earlier side work is queued ahead of it. `relieve` moves such a chain onto compute
//! only where the two-lane `simulate` finishes the block earlier; a chain with real slack stays on the side lane.
#![cfg(feature = "gpu")]
use pc_program::schedule::{relieve, simulate};

const NONE: u32 = u32::MAX;

#[test]
fn a_chain_queued_behind_side_work_moves_to_compute() {
    // side: a long op (0) with nothing waiting for it soon, then a short one (1) compute needs at once (2)
    let costs = [3.0, 0.2, 0.1];
    let preds = vec![vec![], vec![], vec![1]];
    let mut lane = vec![1, 1, 0];
    let (before, waits) = simulate(&costs, &preds, &[false; 3], &lane);
    assert!(waits[2].0 > 3.0 && waits[2].1 == 1, "{waits:?}");
    assert_eq!(relieve(&costs, &preds, &[false; 3], &[], &mut lane, 10), 1);
    assert_eq!(lane, [1, 0, 0]);
    let (after, waits) = simulate(&costs, &preds, &[false; 3], &lane);
    assert!(after < before && waits.iter().all(|w| w.1 == NONE), "{after} vs {before}");
}

#[test]
fn a_chain_with_slack_stays_on_the_side_lane() {
    // compute is busy (0) long enough for the side op (1) its consumer (2) reads
    let costs = [3.0, 0.2, 0.1];
    let preds = vec![vec![], vec![], vec![1]];
    let mut lane = vec![0, 1, 0];
    assert_eq!(relieve(&costs, &preds, &[false; 3], &[], &mut lane, 10), 0);
    assert_eq!(lane, [0, 1, 0]);
}

#[test]
fn moving_a_chain_that_does_not_help_is_undone() {
    // compute waits at 4 for the side chain (2, 3), but running that chain on compute after 0 and 1 would end later
    let costs = [1.0, 1.0, 1.5, 1.0, 0.1];
    let preds = vec![vec![], vec![0], vec![], vec![2], vec![1, 3]];
    let mut lane = vec![0, 0, 1, 1, 0];
    assert_eq!(relieve(&costs, &preds, &[false; 5], &[], &mut lane, 10), 0);
    assert_eq!(lane, [0, 0, 1, 1, 0]);
}

#[test]
fn a_unified_span_and_the_side_ancestors_move_together() {
    // side: a long op 0, a side op 1 feeding a two-step unified span (2, 3); compute's 4 reads the span's end
    let costs = [3.0, 0.1, 0.1, 0.1, 0.1];
    let preds = vec![vec![], vec![], vec![1], vec![2], vec![3]];
    let mut lane = vec![1, 1, 1, 1, 0];
    assert_eq!(relieve(&costs, &preds, &[false; 5], &[(2, 4, true)], &mut lane, 10), 3);
    assert_eq!(lane, [1, 0, 0, 0, 0]);
}

#[test]
fn a_dynamic_step_joins_the_lanes() {
    // the dynamic step (1) starts after both lanes drain, so the side op (0) delays it
    let costs = [2.0, 0.1, 0.1];
    let preds = vec![vec![], vec![], vec![]];
    let (m, _) = simulate(&costs, &preds, &[false, true, false], &[1, 0, 0]);
    assert!((m - 2.2).abs() < 1e-9, "{m}");
}
