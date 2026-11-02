//! Small synthetic graphs through the whole compiler: lowering of each plan decision, v2 args, the check's error
//! kinds, batching and alloc invariants, serde round trip.
use pc_config::{Config, PlanConfig};
use pc_program::compile::{compile_block, CompileOpts};
use pc_program::graph::{BlockGraph, GraphFile};
use pc_program::plan::{BlockPlan, PlacementFile};
use pc_program::{Kind, Program};
use pc_shadow::graph::LoadOpts;
use serde_json::json;
use std::collections::BTreeMap;

fn cfg() -> Config {
    let env = [
        ("CHAIN", "n32"),
        ("LOGN", "16"),
        ("AUTO_BTS_LEVEL", "44"),
        ("BTS_MAX_INPUT_LEVEL", "48"),
        ("FIDESLIB_BTS_RAISE_DROPS", "1,3,5"),
        ("FIDESLIB_BTS_STC_FIRST", "2"),
        ("FIDESLIB_BTS_STC_FIRST_ROUTES", "dense"),
        ("FIDESLIB_BTS_SPARSE_B", "1"),
        ("FIDESLIB_BTS_REAL", "1"),
    ];
    Config::from_legacy_env(env.iter().map(|(k, v)| (*k, Some(*v)))).unwrap().0
}

fn node(op: &str, ins: &[&str], lv: &[i64], out: &str, ol: i64, od: i64) -> serde_json::Value {
    json!({"op_type": op, "inputs": ins, "input_levels": lv, "output": out, "step": "blk0.t", "output_level": ol, "output_noise_level": od})
}

fn compile(nodes: Vec<serde_json::Value>, plan: serde_json::Value, opts: &CompileOpts) -> Program {
    let gf: GraphFile = serde_json::from_value(json!({"version": 2, "nodes": nodes})).unwrap();
    let bg = BlockGraph::from_file(gf, "t", &LoadOpts { level_unit: 2, delib_landing_remap: BTreeMap::new() }).unwrap();
    let pf: PlacementFile = serde_json::from_value(plan).unwrap();
    let bp = BlockPlan::from_file(pf, "t.json").unwrap();
    compile_block("t", &bg, &bp, &cfg(), &PlanConfig::default(), &BTreeMap::new(), None, opts).unwrap()
}

fn keep() -> CompileOpts {
    CompileOpts { keep_going: true, ..Default::default() }
}

fn rules() -> serde_json::Value {
    json!({"bootstrap_level": 28, "max_level": 50, "source_level": 36, "cache_read_level": 36})
}

/// x(26,1) -> square -> sum -> placement (dense, realized) -> * pt -> hint (fired, sparse 1) -> exit.
fn chain_graph() -> Vec<serde_json::Value> {
    vec![
        node("square", &["anon_1"], &[26], "v_1", 28, 2),
        json!({"op_type": "rotate_sum", "inputs": ["v_1", "rot(1,2,3)"], "input_levels": [28, -1], "output": "v_2", "step": "blk0.t",
               "output_level": 28, "output_noise_level": 2, "args": {"steps": [1, 2, 3]}}),
        json!({"op_type": "mult", "inputs": ["v_2", "const(0.333333)"], "input_levels": [28, -1], "output": "v_3", "step": "blk0.t",
               "output_level": 30, "output_noise_level": 2, "args": {"scalar": 0.3333333333333333}}),
        node("mult", &["v_3", "pt_1"], &[30, 32], "v_4", 32, 2),
        node("hint", &["v_4", "hint_lev(30)"], &[32, -1], "v_5", 16, 1),
        node("add", &["v_5", "const(1)"], &[16, -1], "v_6", 16, 1),
    ]
}

fn chain_plan(v3_level: u32) -> serde_json::Value {
    json!({
        "version": 2, "rules": rules(),
        "placements": [{"type": "bootstrap_after_node", "node_op": "mult", "target_var": "v_3"}],
        "hint_fire": ["v_5"], "rescale_after": ["v_3", "v_5"],
        "sparse_slots": {"v_5": 1}, "correction_factor": {"v_3": 9, "v_5": 12}, "real_route": [],
        "summary": {
            "final_named_levels": {"v_1": 28, "v_2": 28, "v_3": v3_level, "v_4": 30, "v_5": 16, "v_6": 16},
            "final_named_degs": {"v_1": 2, "v_2": 2, "v_3": 1, "v_4": 2, "v_5": 1, "v_6": 1},
            "exit_var": "v_6", "exit_level": 16, "exit_deg": 1
        }
    })
}

#[test]
fn lowering_and_check() {
    let p = compile(chain_graph(), chain_plan(28), &keep());
    assert!(p.check.errors.is_empty(), "{:?}", p.check.errors);
    let kinds: Vec<&str> = p.ops.iter().map(|o| o.kind.name()).collect();
    assert_eq!(
        kinds,
        ["square", "rotate_sum", "mult_const", "bootstrap", "rescale", "mult_pt", "bootstrap", "rescale", "add_const"]
    );
    // v2 args: exact scalar, steps
    assert!(matches!(p.ops[2].kind, Kind::MultConst(s) if s.exact && s.value == 1.0 / 3.0));
    assert!(matches!(&p.ops[1].kind, Kind::RotateSum(s) if s == &vec![1, 2, 3]));
    // the placement: dense StC-first route with the plan's CF, raw (26, 2) then realized (28, 1)
    let s0 = &p.bts_sites[0];
    assert_eq!((s0.route.slots, s0.route.cf, s0.route.stc_first, s0.landing.level, s0.landing.deg), (0, Some(9), true, 26, 2));
    let v3 = p.var("v_3").unwrap();
    assert_eq!((p.vars[v3 as usize].meta.level, p.vars[v3 as usize].meta.deg), (28, 1));
    assert_eq!(p.vars[p.ops[3].ins[0] as usize].name, "v_3~pre");
    // the plaintext is encoded where it is used (effective level of v_3: 28)
    assert_eq!((p.pt_refs[0].level, p.pt_refs[0].captured_level), (28, Some(32)));
    // the fired hint: sparse s=1 route with SPRU off (no FIDESLIB_SPRU), lands (14, 2) realized (16, 1)
    let s1 = &p.bts_sites[1];
    assert_eq!((s1.route.slots, s1.route.cf, s1.landing.level), (1, Some(12), 14));
    assert_eq!(p.accounting.hint_bootstraps, 1);
    assert!(p.accounting.balanced());
    assert_eq!(p.outputs[0], p.var("v_6").unwrap());
    assert_eq!(p.check.level_exact + p.check.level_richer, 6);
    assert_eq!(p.check.inexact_scalars, 1);
}

#[test]
fn level_and_degree_errors() {
    // the plan pins v_3 one level shallower than the runtime lands it: [plan_level_error] on v_3
    let p = compile(chain_graph(), chain_plan(26), &keep());
    assert!(p.check.errors.iter().any(|e| e.starts_with("[plan_level_error] v_3")), "{:?}", p.check.errors);
    // without the realize anchor the placement lands degree 2: [plan_deg_error] on v_3
    let mut plan = chain_plan(28);
    plan["rescale_after"] = json!(["v_5"]);
    let p = compile(chain_graph(), plan, &keep());
    assert!(p.check.errors.iter().any(|e| e.starts_with("[plan_deg_error] v_3")), "{:?}", p.check.errors);
    // and the strict compile refuses
    let gf: GraphFile = serde_json::from_value(json!({"version": 2, "nodes": chain_graph()})).unwrap();
    let bg = BlockGraph::from_file(gf, "t", &LoadOpts { level_unit: 2, delib_landing_remap: BTreeMap::new() }).unwrap();
    let bp = BlockPlan::from_file(serde_json::from_value(chain_plan(26)).unwrap(), "t.json").unwrap();
    let r = compile_block("t", &bg, &bp, &cfg(), &PlanConfig::default(), &BTreeMap::new(), None, &CompileOpts::default());
    assert!(matches!(r, Err(pc_program::Error::Check { .. })));
}

#[test]
fn unknown_op_is_an_error() {
    let gf: GraphFile = serde_json::from_value(json!({"version": 1, "nodes": [node("frobnicate", &["anon_1"], &[26], "v_1", 26, 1)]})).unwrap();
    let bg = BlockGraph::from_file(gf, "t", &LoadOpts { level_unit: 2, delib_landing_remap: BTreeMap::new() }).unwrap();
    let bp = BlockPlan::from_file(serde_json::from_value(json!({"version": 2, "rules": rules()})).unwrap(), "t.json").unwrap();
    let r = compile_block("t", &bg, &bp, &cfg(), &PlanConfig::default(), &BTreeMap::new(), None, &keep());
    match r {
        Err(pc_program::Error::Graph { msg, .. }) => assert!(msg.contains("unknown op kind \"frobnicate\""), "{msg}"),
        other => panic!("{:?}", other.map(|p| p.ops.len())),
    }
}

#[test]
fn unfired_hint_aliases_and_finish_siblings_merge() {
    let nodes = vec![
        node("hint", &["anon_1", "hint_lev(40)"], &[30, -1], "v_1", 30, 1),
        node("deliberate_bootstrap", &["v_1"], &[30], "v_2", 26, 2),
        node("deliberate_bootstrap", &["v_1"], &[30], "v_3", 26, 2),
        node("add", &["v_2", "v_3"], &[26, 26], "v_4", 26, 2),
    ];
    let plan = json!({"version": 2, "rules": rules(), "hint_fire": [], "rescale_after": [],
        "summary": {"final_named_levels": {"v_1": 30, "v_2": 26, "v_3": 26, "v_4": 26}, "final_named_degs": {"v_1": 2, "v_2": 2, "v_3": 2, "v_4": 2}}});
    let p = compile(nodes, plan, &keep());
    assert!(p.check.errors.is_empty(), "{:?}", p.check.errors);
    assert_eq!(p.var("v_1"), p.var("anon_1"));
    assert_eq!(p.ops.len(), 2);
    assert!(matches!(&p.ops[0].kind, Kind::Finish { landing, .. } if landing.level == 26 && landing.deg == 2));
    assert_eq!(p.ops[0].outs.len(), 2);
    assert_eq!((p.accounting.elided, p.accounting.merged), (1, 1));
    assert!(p.accounting.balanced());
}

#[test]
fn siblings_batch_and_slots_never_overlap() {
    // four independent squares of one input level, then their sum
    let mut nodes = vec![];
    for i in 0..4 {
        nodes.push(node("square", &[&format!("anon_{}", i + 1)], &[26], &format!("v_{}", i + 1), 28, 2));
    }
    nodes.push(node("add", &["v_1", "v_2"], &[28, 28], "v_5", 28, 2));
    nodes.push(node("add", &["v_3", "v_4"], &[28, 28], "v_6", 28, 2));
    nodes.push(node("add", &["v_5", "v_6"], &[28, 28], "v_7", 28, 2));
    let plan = json!({"version": 2, "rules": rules(), "summary": {"exit_var": "v_7"}});
    let p = compile(nodes.clone(), plan.clone(), &CompileOpts { keep_going: true, ..Default::default() });
    // all four inputs exist at the first square: one launch
    assert_eq!(p.batch_report.per_kind.iter().find(|k| k.kind == "square").map(|k| (k.ops, k.launches)), Some((4, 1)));
    assert!(p.check.errors.is_empty(), "{:?}", p.check.errors);
    // live vars never share a slot
    let mut live: std::collections::BTreeMap<u32, u32> = Default::default();
    for &v in &p.inputs {
        assert!(live.insert(p.vars[v as usize].slot.unwrap(), v).is_none());
    }
    let mut last = std::collections::BTreeMap::new();
    for (i, op) in p.ops.iter().enumerate() {
        for v in op.all_ins() {
            last.insert(v, i);
        }
    }
    for (i, op) in p.ops.iter().enumerate() {
        for o in op.all_outs() {
            let s = p.vars[o as usize].slot.unwrap();
            if let Some(&prev) = live.get(&s) {
                // only an in-place op may reuse a slot whose var is still mentioned by this op
                assert!(last.get(&prev).map_or(true, |&l| l <= i), "slot {s} of {} reused while live", p.vars[prev as usize].name);
            }
            live.insert(s, o);
        }
    }
    assert!(p.mem.bytes >= p.mem.live_peak && p.mem.live_peak > 0);
    let unbatched = compile(nodes, plan, &CompileOpts { keep_going: true, no_batch: true, ..Default::default() });
    assert_eq!(unbatched.batch_report.launches, 7);
}

#[test]
fn serde_round_trip() {
    let p = compile(chain_graph(), chain_plan(28), &keep());
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("pc_program_rt_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("block_0.program.json");
    p.save(&f).unwrap();
    let q = Program::load(&f).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(p, q);
}

/// A refresh whose input leaves no room for the raise's two rescales (pending product + scale adjust) above the
/// composite bottom is a compile error: the device route would run off the chain.
#[test]
fn bootstrap_input_needs_raise_room() {
    use pc_program::compile::check::rule;
    use pc_program::program::{BtsSite, SiteKind};
    use pc_shadow::route::{Landing, Route};
    use pc_shadow::CtMeta;
    let mut c = cfg();
    c.set_env("BTS_MAX_INPUT_LEVEL", Some("50")).unwrap();
    let sh = pc_program::compile::runtime_shadow(&c, &PlanConfig::default(), Some(34)).unwrap();
    let site = BtsSite {
        var: "x".into(),
        kind: SiteKind::Placement,
        route: Route::dense(),
        input: CtMeta::new(50, 2),
        landing: Landing { level: 34, deg: 1 },
        realize: false,
        prescale: None,
        offset: None,
    };
    let sites = vec![site];
    // (50, 2): nominal 50 passes BTS_MAX_INPUT_LEVEL, effective 52 has no room for the adjust rescale
    let err = rule(&sh, &Kind::Bootstrap(0), &[CtMeta::new(50, 2)], &[], &sites).unwrap_err();
    assert!(err.starts_with("[bts_input_error]"), "{err}");
    // (50, 1) and (48, 2): effective 50, the raise reaches the bottom
    assert!(rule(&sh, &Kind::Bootstrap(0), &[CtMeta::new(50, 1)], &[], &sites).is_ok());
    assert!(rule(&sh, &Kind::Bootstrap(0), &[CtMeta::new(48, 2)], &[], &sites).is_ok());
}

/// On a lean chain (CHAIN_PRESET=n32a4: 51 primes, a single s1 above the q0 pair) the runtime shadow takes the real
/// limb count, and a refresh input may reach effective level 48 (s1's top), not n32's 50.
#[test]
fn runtime_shadow_on_the_lean_chain() {
    use pc_program::compile::check::rule;
    use pc_program::program::{BtsSite, SiteKind};
    use pc_shadow::route::{Landing, Route};
    use pc_shadow::CtMeta;
    let mut c = cfg();
    c.set_env("BTS_MAX_INPUT_LEVEL", Some("50")).unwrap();
    let n32 = pc_program::compile::runtime_shadow(&c, &PlanConfig::default(), Some(34)).unwrap();
    assert_eq!((n32.cfg.chain.n_q, n32.cfg.bts.input_room), (54, Some(50)));
    c.set_env("CHAIN_PRESET", Some("n32a4")).unwrap();
    let sh = pc_program::compile::runtime_shadow(&c, &PlanConfig::default(), Some(34)).unwrap();
    assert_eq!((sh.cfg.chain.n_q, sh.cfg.bts.input_room), (51, Some(48)));
    let site = BtsSite {
        var: "x".into(),
        kind: SiteKind::Placement,
        route: Route::dense(),
        input: CtMeta::new(48, 1),
        landing: Landing { level: 34, deg: 1 },
        realize: false,
        prescale: None,
        offset: None,
    };
    let sites = vec![site];
    assert!(rule(&sh, &Kind::Bootstrap(0), &[CtMeta::new(48, 1)], &[], &sites).is_ok());
    assert!(rule(&sh, &Kind::Bootstrap(0), &[CtMeta::new(46, 2)], &[], &sites).is_ok());
    let err = rule(&sh, &Kind::Bootstrap(0), &[CtMeta::new(48, 2)], &[], &sites).unwrap_err();
    assert!(err.starts_with("[bts_input_error]"), "{err}");
}
