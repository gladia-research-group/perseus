//! `compile::user_region_sites` on raise drops: a drop moves a refresh's landing down with the rest of it, and the guard
//! takes it where EvalMod stays above the alternating region, every route landing at most one level below the region's
//! top. n32sp's region top is the dense route's landing (28), so one dense drop at most; n32sph keeps three uniform pairs
//! above its region (top 34), so a dense drop of up to 4 levels.
use pc_core::params::{Chain, ChainSpec};
use pc_program::compile::user_region_sites;
use pc_program::program::{BtsSite, SiteKind};
use pc_shadow::{ChainShape, CtMeta, Landing, Route};

fn shape(preset: &str) -> ChainShape {
    ChainShape::of(&Chain::generate(&ChainSpec::preset(preset).unwrap()).unwrap())
}

/// A refresh on route `slots` at `raise_drop`, landing realized (degree 1) at `level`.
fn site(slots: u32, raise_drop: u32, level: u32) -> BtsSite {
    BtsSite {
        var: format!("v_{slots}_{raise_drop}"),
        kind: SiteKind::Placement,
        route: Route { slots, raise_drop, ..Route::dense() },
        input: CtMeta::new(40, 1),
        landing: Landing { level, deg: 1 },
        realize: true,
        prescale: None,
        offset: None,
    }
}

#[test]
fn n32sp_takes_one_dense_drop() {
    let c = shape("n32sp");
    assert!(user_region_sites(&c, &[site(0, 0, 28), site(0, 1, 30)]).is_empty());
    for d in 2..=3 {
        assert_eq!(user_region_sites(&c, &[site(0, d, 28 + 2 * d)]).len(), 1, "dense drop {d}");
    }
}

#[test]
fn n32sph_takes_dense_drops_that_land_above_its_region() {
    let c = shape("n32sph");
    for d in 0..=4 {
        assert!(user_region_sites(&c, &[site(0, d, 28 + 2 * d)]).is_empty(), "dense drop {d}");
    }
    let e = user_region_sites(&c, &[site(0, 5, 38)]);
    assert!(e.len() == 1 && e[0].contains("runs EvalMod on the user region"), "{e:?}");
    // SparseB and SPRU one level below the region top
    assert!(user_region_sites(&c, &[site(512, 6, 36), site(1, 10, 36)]).is_empty());
    assert_eq!(user_region_sites(&c, &[site(512, 7, 38), site(1, 11, 38)]).len(), 2);
}

#[test]
fn uniform_chains_have_no_region() {
    assert!(user_region_sites(&shape("n32a4"), &[site(0, 8, 44)]).is_empty());
}
