//! MemPlan sizing without a GPU: formulas, refusals naming the knob, reserved sections.
use pc_core::modmul::{Barrett64, Montgomery};
use pc_core::params::{Chain, ChainSpec};
use pc_gpu::ntt::RowTables;
use pc_gpu::{Error, Layout, MemCfg, MemPlan, PoolClass};

#[test]
fn n32_plan() {
    let chain = Chain::generate(&ChainSpec::n32()).unwrap();
    let cfg = MemCfg { k_max: 8, pool: vec![PoolClass { k: 8, count: 1 }, PoolClass { k: 1, count: 2 }], events: 16, scratch_cts: 1 };
    let mut plan = MemPlan::build::<u32, Barrett64>(&chain, cfg.clone()).unwrap();
    let lay = Layout::from_chain(&chain);
    assert_eq!((lay.nq, lay.kp, lay.n), (54, 9, 1 << 16));
    let ct = lay.ct_words() * 4;
    assert_eq!(plan.scratch, 8 * ct);
    assert_eq!(plan.ct_pool, 8 * ct + 2 * ct);
    // ModUp digits (k_max x dnum polys) + extended dot outputs (a 7-key sum of 4 cts: 28 cts) + ModDown conversions
    // (k_max cts)
    assert_eq!(plan.ks_work, 8 * 6 * lay.poly_words() * 4 + (28 + 8) * ct);
    assert_eq!(plan.bconv_work, 8 * lay.poly_words() * 4);
    // twiddles dominate the context: 4 tables x 63 primes x N words
    assert!(plan.context > 4 * 63 * (1 << 16) * 4 && plan.context < 4 * 63 * (1 << 16) * 4 + (1 << 20));
    assert_eq!(RowTables::entries(&lay, 8), 4 * 8 * (54 * 55 / 2 + (54 * 55 / 2 + 54 * 9) + 9));
    let t0 = plan.total();
    plan.reserve("keys", 1000);
    assert_eq!(plan.section("keys"), Some(1024));
    assert_eq!(plan.total(), t0 + 1024);
    // the same chain sized for another u32 instantiation is identical
    assert_eq!(MemPlan::build::<u32, Montgomery>(&chain, cfg).unwrap().total(), t0);
}

#[test]
fn refusals_name_the_knob() {
    let chain = Chain::generate(&ChainSpec::n32()).unwrap();
    let bad = MemCfg { k_max: 0, ..MemCfg::default() };
    match MemPlan::build::<u32, Barrett64>(&chain, bad) {
        Err(Error::Mem { knob: "MemCfg::k_max", .. }) => {}
        r => panic!("expected k_max refusal, got {:?}", r.map(|p| p.total())),
    }
    // n32 primes are 32-bit-word primes: a u64 context over the same chain is refused (chain.spec.word)
    match MemPlan::build::<u64, Barrett64>(&chain, MemCfg::default()) {
        Err(Error::Shape { what: "chain.spec.word", .. }) => {}
        r => panic!("expected word refusal, got {:?}", r.map(|p| p.total())),
    }
}
