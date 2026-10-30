//! Planner equivalence: the OLD `sim.py::simulate`, run by `tests/tools/sim_dump.py` on the three shipped plans
//! (its inputs reconstructed from the placement files and checked against their `final_named_levels/degs`), and
//! `pc_shadow::sim::simulate` on the same graph.json must agree on every node: the rule output (node_out,
//! node_deg), the var's final state (consumed, deg), hint decisions and the budget check. Two runs per block: the
//! shipped plan, and a plan-free run (no refreshes, hints by threshold) that drives the deep levels and the budget.
use pc_shadow::graph::{Graph, LoadOpts};
use pc_shadow::meta::{level_from_planner_f64, ChainShape};
use pc_shadow::route::BtsConfig;
use pc_shadow::sim::{simulate, SimInputs};
use pc_shadow::{Quirks, Shadow, ShadowConfig};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

fn lvl(c: &Value, bl: u32) -> u32 {
    level_from_planner_f64(c.as_f64().unwrap(), bl).unwrap()
}

struct Tally {
    nodes: usize,
    diffs: usize,
    hints: usize,
    over: usize,
}

#[allow(clippy::too_many_arguments)]
fn compare(name: &str, run: &str, blk: &str, exp: &Value, g: &Graph, res: &pc_shadow::sim::SimResult, bl: u32, t: &mut Tally, verbose: bool) {
    let node_out = exp["node_out"].as_array().unwrap();
    let node_deg = exp["node_deg"].as_array().unwrap();
    let final_c = exp["final_c"].as_array().unwrap();
    let final_d = exp["final_d"].as_array().unwrap();
    let mut bad = 0;
    for n in &g.nodes {
        let i = n.idx;
        let mut ok = true;
        let got_c = res.node_out.get(&i).copied();
        if got_c != node_out[i].as_f64() {
            ok = false;
        }
        if res.node_deg.get(&i).map(|&d| d as u64) != node_deg[i].as_u64() {
            ok = false;
        }
        if !n.output.is_empty() {
            let m = res.state.get(&n.output);
            if m.map(|m| m.level as f64 - bl as f64) != final_c[i].as_f64() || m.map(|m| m.deg as u64) != final_d[i].as_u64() {
                ok = false;
            }
        }
        if !ok {
            bad += 1;
            if verbose && bad <= 5 {
                println!(
                    "  {name} {blk} {run} node {i} {} {:?} -> {:?}: python ({}, {}) final ({}, {}) | rust ({:?}, {:?}) final {:?}",
                    n.op,
                    n.inputs,
                    n.output,
                    node_out[i],
                    node_deg[i],
                    final_c[i],
                    final_d[i],
                    got_c,
                    res.node_deg.get(&i),
                    res.state.get(&n.output).map(|m| (m.level as i64 - bl as i64, m.deg))
                );
            }
        }
    }
    let hf: HashMap<usize, bool> =
        exp["hint_fired"].as_object().unwrap().iter().map(|(k, v)| (k.parse().unwrap(), v.as_bool().unwrap())).collect();
    let got_hf: HashMap<usize, bool> = res.hint_fired.iter().map(|(&k, &v)| (k, v)).collect();
    if hf != got_hf {
        if verbose {
            println!("  {name} {blk} {run}: hint decisions differ: python {hf:?} rust {got_hf:?}");
        }
        bad += 1;
    }
    let over: Vec<(usize, f64)> = exp["over_budget"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| (p[0].as_u64().unwrap() as usize, p[1].as_f64().unwrap()))
        .collect();
    if over != res.over_budget {
        if verbose {
            println!("  {name} {blk} {run}: over_budget differs: python {} entries, rust {}", over.len(), res.over_budget.len());
        }
        bad += 1;
    }
    t.nodes += g.nodes.len();
    t.diffs += bad;
    t.hints += hf.values().filter(|&&f| f).count();
    t.over += over.len();
}

fn run_dump(file: &str) -> Option<(usize, usize, usize, usize, usize)> {
    run_dump_with(file, &Quirks::planner(), true)
}

/// Replay a dump with `quirks`; errors from the rules count as one difference per block.
fn run_dump_with(file: &str, quirks: &Quirks, verbose: bool) -> Option<(usize, usize, usize, usize, usize)> {
    let path = format!("{}/tests/data/{file}", env!("CARGO_MANIFEST_DIR"));
    let doc: Value = serde_json::from_slice(&std::fs::read(&path).expect("sim dump")).unwrap();
    let unit = doc["level_unit"].as_i64().unwrap();
    let env = &doc["env"];
    let opts = LoadOpts {
        level_unit: unit,
        delib_landing_remap: env
            .get("PLAN_DELIB_LANDING_REMAP")
            .and_then(Value::as_str)
            .map(|s| LoadOpts::parse_remap(s).unwrap())
            .unwrap_or_default(),
    };
    let delib_deg_captured = env.get("PLAN_DELIB_DEG").and_then(Value::as_str) == Some("captured");
    let kv = env.get("PLAN_STC_FIRST_KV_LANDING").and_then(Value::as_str).map(|s| {
        let (l, d) = s.split_once(':').unwrap_or((s, "2"));
        (l.parse::<f64>().unwrap() as u32, d.parse::<u8>().unwrap_or(2))
    });
    let mut planned = Tally { nodes: 0, diffs: 0, hints: 0, over: 0 };
    let mut bare = Tally { nodes: 0, diffs: 0, hints: 0, over: 0 };
    let name = file.trim_end_matches(".json");
    for b in doc["blocks"].as_array().unwrap() {
        // relative to the repository root (sim_dump.py runs there)
        let gf = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(b["graph_file"].as_str().unwrap());
        let gf = gf.to_str().unwrap();
        if !std::path::Path::new(gf).is_file() {
            println!("SKIP: {gf} not readable");
            return None;
        }
        let g0 = Graph::load(gf, &opts).unwrap();
        let (g, _) = g0.erase_reactive_bootstraps(&opts).unwrap();
        let outs: Vec<&str> = b["outputs"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        let mine: Vec<&str> = g.nodes.iter().map(|n| n.output.as_str()).collect();
        assert_eq!(outs, mine, "{name} {}: topological order differs", b["block"]);
        let bl = b["bootstrap_level"].as_f64().unwrap() as u32;
        let mut bts = BtsConfig::n32_shipped();
        bts.bootstrap_level = bl;
        let shadow = Shadow::new(
            ShadowConfig { chain: ChainShape::unbounded(unit as u32), bts, quirks: quirks.clone() },
            None,
        );
        let seeds: HashMap<String, (u32, u8)> = b["seeds"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), (lvl(&v[0], bl), v[1].as_u64().unwrap() as u8)))
            .collect();
        let refreshed: HashMap<String, (u32, u8)> = b["refreshed"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), (lvl(&v[0], bl), v[1].as_u64().unwrap() as u8)))
            .collect();
        let set = |k: &str| -> HashSet<String> { b[k].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect() };
        let dense_fallback = b["dense_fallback"].as_array().map(|a| {
            (
                (a[0].as_f64().unwrap()) as u32,
                lvl(&a[1], bl),
                a[2].as_u64().unwrap() as u8,
            )
        });
        let base = SimInputs {
            budget_l: b["budget_L"].as_f64().unwrap(),
            guard_margin: true,
            seeds,
            deliberate_clamp0: b["deliberate_clamp0"].as_bool().unwrap(),
            delib_deg_captured,
            stc_first_kv_landing: kv,
            dense_fallback,
            ..Default::default()
        };
        let inp = SimInputs {
            refreshed,
            sparse_refreshed: set("sparse_refreshed"),
            placed: set("placed"),
            realized: set("realized"),
            hint_force: b["hint_force"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_bool().unwrap())).collect(),
            ..base.clone()
        };
        let blk = b["block"].as_str().unwrap();
        match simulate(&shadow, &g, &inp) {
            Ok(res) => compare(name, "planned", blk, &b["planned"], &g, &res, bl, &mut planned, verbose),
            Err(e) => {
                planned.diffs += 1;
                if verbose {
                    println!("  {name} {blk} planned: {e}");
                }
            }
        }
        match simulate(&shadow, &g, &base) {
            Ok(res) => compare(name, "bare", blk, &b["bare"], &g, &res, bl, &mut bare, verbose),
            Err(e) => {
                bare.diffs += 1;
                if verbose {
                    println!("  {name} {blk} bare: {e}");
                }
            }
        }
    }
    if verbose {
        println!(
            "[planner-equiv] {name}: {} blocks, {} nodes; planned: {} differences ({} hints fired); bare: {} differences ({} hints fired, {} over budget)",
            doc["blocks"].as_array().unwrap().len(),
            planned.nodes,
            planned.diffs,
            planned.hints,
            bare.diffs,
            bare.hints,
            bare.over
        );
    }
    Some((planned.nodes, planned.diffs, bare.diffs, bare.over, planned.hints))
}

/// Plan-free runs of the dumped sim.py land a fired hint without a landing spec at (bootstrap level, deg 2); the
/// rules realize it (deg 1, one level deeper). Nodes downstream of such a hint differ, per dump (planned runs pin
/// every hint's landing: 0 there).
const BARE_HINT_LANDINGS: (usize, usize, usize, usize) = (48, 56, 33, 32);

#[test]
fn gpt2_decode() {
    if let Some((n, d1, d2, _, _)) = run_dump("sim_gpt2_decode.json") {
        assert!(n > 0);
        assert_eq!((d1, d2), (0, BARE_HINT_LANDINGS.0));
    }
}

#[test]
fn gpt2_prefill() {
    if let Some((n, d1, d2, _, _)) = run_dump("sim_gpt2_prefill.json") {
        assert!(n > 0);
        assert_eq!((d1, d2), (0, BARE_HINT_LANDINGS.1));
    }
}

#[test]
fn llama_decode() {
    if let Some((n, d1, d2, _, _)) = run_dump("sim_llama_decode.json") {
        assert!(n > 0);
        assert_eq!((d1, d2), (0, BARE_HINT_LANDINGS.2));
    }
}

#[test]
fn llama_prefill() {
    if let Some((n, d1, d2, _, _)) = run_dump("sim_llama_prefill.json") {
        assert!(n > 0);
        assert_eq!((d1, d2), (0, BARE_HINT_LANDINGS.3));
    }
}

/// Which planner quirks the shipped plans depend on: drop one at a time and count the node differences (planned
/// run / plan-free run). Informational (the table goes to docs/QUIRKS.md); it asserts only that the
/// full planner set is what reproduces sim.py.
#[test]
fn quirk_dependence() {
    let files = ["sim_gpt2_decode.json", "sim_gpt2_prefill.json", "sim_llama_decode.json", "sim_llama_prefill.json"];
    for q in Quirks::planner().0.iter() {
        let mut qs = Quirks::planner();
        qs.0.remove(q);
        let mut row = format!("[quirk-dependence] without {q:?}:");
        for f in files {
            match run_dump_with(f, &qs, false) {
                Some((_, d1, d2, _, _)) => row += &format!(" {}={}/{}", f.trim_start_matches("sim_").trim_end_matches(".json"), d1, d2),
                None => return,
            }
        }
        println!("{row}");
    }
}
