//! Compile every block of the shipped plans against their graphs (CPU only).
//!   * 0 level/degree check errors (the runtime Shadow over each compiled program vs the plan's pins);
//!   * op accounting: graph - erased autos + placements == ops (+ elided no-ops, merged finish siblings, hint-
//!     consumed placements, - realize rescales);
//!   * batch_report of GPT-2 decode block 0; serde round-trip of every program.
//! Graph dirs: the plan's PLAN_CMD.txt `graph=`, under the repository root. A plan or graph that is not on disk
//! (the Llama captures ship as release artifacts) is skipped with a SKIP line.
use pc_program::compile::{compile_dir, CompileOpts};
use pc_program::Program;
use std::path::{Path, PathBuf};

fn assets() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn find_graph(name: &str) -> Option<PathBuf> {
    Some(assets().join("graphs").join(name)).filter(|p| p.join("block_0/graph.json").is_file())
}

fn run(plan: &str, graph: &str) -> Option<Vec<(String, Program)>> {
    let pd = assets().join("bootstrap_placements").join(plan);
    let Some(gd) = find_graph(graph) else {
        println!("SKIP {plan}: graph {graph} not on disk");
        return None;
    };
    if !pd.join("block_0_placement.json").is_file() {
        println!("SKIP {plan}: plan not on disk");
        return None;
    }
    let t = std::time::Instant::now();
    let dc = compile_dir(&pd, &gd, &CompileOpts { keep_going: true, no_batch: false, schedule: Default::default(), batch_cf_free: false, ..Default::default() }).expect("plan dir");
    let mut out = vec![];
    let (mut ops, mut bts, mut launches, mut blaunch, mut errs, mut peak, mut arena) = (0, 0, 0, 0, 0, 0u64, 0u64);
    println!("[m7a] {plan} (graph {graph}): {} programs", dc.programs.len());
    println!("  block | nodes autos place | ops bts(pl/hint/delib) finish rescale elided | launches bts_launches | pins exact richer | pt_relevel | arena MB live MB | errors");
    for (label, r) in dc.programs {
        let p = match r {
            Ok(p) => p,
            Err(e) => panic!("{plan} {label}: {e}"),
        };
        let a = &p.accounting;
        let b = &p.batch_report;
        let c = &p.check;
        println!(
            "  {label} | {} {} {} | {} {}({}/{}/{}) {} {} {} | {} {} | {} {} {} | {} | {:.1} {:.1} | {}",
            a.graph_nodes,
            a.erased_autos,
            a.placements,
            a.ops,
            b.bootstraps,
            a.bootstraps,
            a.hint_bootstraps,
            a.deliberate_bootstraps,
            a.finishes,
            a.rescales,
            a.elided,
            b.launches,
            b.bootstrap_launches,
            c.pinned,
            c.level_exact,
            c.level_richer,
            c.pt_relevels,
            p.mem.bytes as f64 / 1e6,
            p.mem.live_peak as f64 / 1e6,
            c.errors.len()
        );
        for e in c.errors.iter().take(6) {
            println!("      {e}");
        }
        assert!(a.balanced(), "{plan} {label}: accounting {a:?}");
        let plan_file = pc_program::plan::BlockPlan::load(assets().join("bootstrap_placements").join(plan).join(format!("{label}_placement.json"))).unwrap();
        let st = plan_file.file.summary.total_bootstraps.unwrap_or(0) as usize;
        // the planner counts every refresh node: placements, fired hints, deliberate and finish nodes
        assert_eq!(st, b.bootstraps + a.finishes + a.merged, "{plan} {label}: summary.total_bootstraps");
        ops += a.ops;
        bts += b.bootstraps;
        launches += b.launches;
        blaunch += b.bootstrap_launches;
        errs += c.errors.len();
        peak = peak.max(p.mem.bytes);
        arena += p.mem.bytes;
        out.push((label, p));
    }
    println!(
        "[m7a] {plan}: ops {ops}, bootstraps {bts}, launches {launches} (bootstrap launches {blaunch}), max block arena {:.1} MB, check errors {errs}, {:.1} s",
        peak as f64 / 1e6,
        t.elapsed().as_secs_f64()
    );
    let _ = arena;
    let (mut pins, mut exact, mut rich, mut inexact, mut relevel, mut json, mut fin, mut res) = (0, 0, 0, 0, 0, 0, 0, 0);
    for (_, p) in &out {
        pins += p.check.pinned;
        exact += p.check.level_exact;
        rich += p.check.level_richer;
        inexact += p.check.inexact_scalars;
        relevel += p.check.pt_relevels;
        json = p.to_json().unwrap().len().max(json);
        fin += p.accounting.finishes;
        res += p.accounting.rescales;
    }
    println!(
        "[m7a] {plan}: pins {pins} (exact {exact}, richer {rich}), finishes {fin}, rescales {res}, pt relevels {relevel}, inexact v1 scalars {inexact}, max program json {:.2} MB",
        json as f64 / 1e6
    );
    Some(out)
}

fn roundtrip(progs: &[(String, Program)]) {
    for (label, p) in progs {
        let s = p.to_json().unwrap();
        let q = Program::from_json(&s).unwrap();
        assert!(q == *p, "{label}: serde round trip differs");
    }
}

fn gate(plan: &str, graph: &str) -> Option<Vec<(String, Program)>> {
    let progs = run(plan, graph)?;
    roundtrip(&progs);
    let errs: usize = progs.iter().map(|(_, p)| p.check.errors.len()).sum();
    assert_eq!(errs, 0, "{plan}: {errs} check errors");
    Some(progs)
}

#[test]
fn gpt2_decode() {
    let Some(progs) = gate("gpt2_decode", "gpt2_decode") else { return };
    let p = &progs.iter().find(|(l, _)| l == "block_0").unwrap().1;
    println!("[m7a] gpt2 decode block_0 batch_report: {}", serde_json::to_string(&p.batch_report).unwrap());
}

#[test]
fn gpt2_prefill() {
    gate("gpt2_prefill128", "gpt2_prefill128");
}

#[test]
fn gpt2_decode_ctx1024() {
    gate("gpt2_decode_ctx1024", "gpt2_decode_ctx1024");
}

#[test]
fn bert() {
    gate("bert_sst2_128", "bert_sst2_128");
}

#[test]
fn vit() {
    gate("vit_b16_224", "vit_b16_224");
}

#[test]
fn llama_decode() {
    gate("llama31_8b_decode", "llama31_8b_decode");
}

#[test]
fn llama_prefill() {
    gate("llama31_8b_prefill128", "llama31_8b_prefill128");
}

#[test]
fn llama_handoff_decode() {
    gate("llama31_8b_handoff_decode", "llama31_8b_handoff_decode");
}

/// Informational: the two schedules side by side (launches, bootstrap launches, arena), every block compiled
/// and checked under each.
#[test]
fn schedules() {
    use pc_program::compile::batch::Schedule;
    for (plan, graph) in [
        ("gpt2_decode", "gpt2_decode"),
        ("gpt2_prefill128", "gpt2_prefill128"),
        ("llama31_8b_decode", "llama31_8b_decode"),
    ] {
        let pd = assets().join("bootstrap_placements").join(plan);
        let Some(gd) = find_graph(graph) else { continue };
        if !pd.is_dir() {
            continue;
        }
        for (sched, cf_free) in [(Schedule::Program, false), (Schedule::Lockstep, false), (Schedule::Program, true), (Schedule::Lockstep, true)] {
            let dc = compile_dir(&pd, &gd, &CompileOpts { keep_going: true, no_batch: false, schedule: sched, batch_cf_free: cf_free, ..Default::default() }).unwrap();
            let (mut ops, mut l, mut bts, mut bl, mut errs, mut arena) = (0, 0, 0, 0, 0, 0u64);
            let mut b0 = String::new();
            for (label, r) in &dc.programs {
                let p = r.as_ref().unwrap();
                let b = &p.batch_report;
                ops += b.ops;
                l += b.launches;
                bts += b.bootstraps;
                bl += b.bootstrap_launches;
                errs += p.check.errors.len();
                arena = arena.max(p.mem.bytes);
                if label == "block_0" {
                    b0 = format!("block_0: launches {}/{} ops, bootstrap launches {}/{} hist {:?}, arena {:.1} MB", b.launches, b.ops, b.bootstrap_launches, b.bootstraps, b.bootstrap_hist, p.mem.bytes as f64 / 1e6);
                }
            }
            println!(
                "[m7a-sched] {plan} {sched:?} cf_free={cf_free}: launches {l}/{ops} ops, bootstrap launches {bl}/{bts}, max arena {:.1} MB, errors {errs}; {b0}",
                arena as f64 / 1e6
            );
            assert_eq!(errs, 0);
        }
    }
}
