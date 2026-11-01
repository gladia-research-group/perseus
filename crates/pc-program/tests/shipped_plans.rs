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
    let Some(progs) = gate("gpt2_decode_python_n32", "gpt2_decode_python_n32_gcheb") else { return };
    let p = &progs.iter().find(|(l, _)| l == "block_0").unwrap().1;
    println!("[m7a] gpt2 decode block_0 batch_report: {}", serde_json::to_string(&p.batch_report).unwrap());
}

/// The refreshes of a decode plan whose input sits past the StC-first entry, so the route runs the shipped order
/// (pc-gpu `StcFirstRoute::runs_stc_first`) and lands at its fallback: exactly `sites` (block, var, CF), the ones
/// PLAN_FALLBACK_CF prices for that order. An input realized under the entry would run StC-first at that CF, where the
/// slots overflow its band with no error raised; one moved past it would run the shipped order at a CF priced for
/// the slots.
fn shipped_order_sites(plan: &str, progs: &[(String, Program)], sites: &[(&str, &str, u32)]) {
    let bts = pc_shadow::BtsConfig::n32_shipped();
    let (entry, fallback) = (bts.stc_first_entry.unwrap(), bts.stc_first_fallback.unwrap());
    let unit = pc_shadow::ChainShape::N32.unit;
    let eff = |l: pc_shadow::Landing| pc_shadow::meta::effective_level(l.level, l.deg, unit);
    let mut past = vec![];
    for (block, p) in progs {
        for s in p.bts_sites.iter().filter(|s| s.route.slots == 0 && s.route.stc_first) {
            if s.input.effective_level(unit) > entry {
                assert_eq!(eff(s.landing), eff(fallback), "{plan} {block} {}: landing {:?}", s.var, s.landing);
                past.push((block.as_str(), s.var.as_str(), s.route.cf.unwrap_or(0)));
            }
        }
    }
    let mut want = sites.to_vec();
    past.sort();
    want.sort();
    assert_eq!(past, want, "{plan}: the refreshes past the StC-first entry (block, var, CF)");
}

/// FusedUnits' decode plan (m16i with the shipped order's CF where that order runs): the five down-projection hints at
/// 2; the argmax's v_2 keeps its 5.
#[test]
fn gpt2_decode_speedrun_fcf() {
    let Some(progs) = gate("gpt2_decode_python_n32_m16i_fcf", "gpt2_decode_python_n32_gcheb") else { return };
    let sites = [("block_1", "v_1017", 2), ("block_2", "v_1483", 2), ("block_4", "v_1113", 2), ("block_10", "v_1075", 2),
                 ("block_11", "v_1394", 2), ("block_13", "v_2", 5)];
    shipped_order_sites("gpt2_decode_python_n32_m16i_fcf", &progs, &sites);
}

/// m16i_fcf's blocks with CutMax's scaled normalisers in block 13 (its entry, three vector refreshes and the last one
/// past the StC-first entry).
#[test]
fn gpt2_decode_fcf_cm() {
    let Some(progs) = gate("gpt2_decode_python_n32_m16i_fcf_cm", "gpt2_decode_python_n32_cm") else { return };
    let sites = [("block_1", "v_1017", 2), ("block_2", "v_1483", 2), ("block_4", "v_1113", 2), ("block_10", "v_1075", 2),
                 ("block_11", "v_1394", 2), ("block_13", "v_16", 2), ("block_13", "v_135", 5), ("block_13", "v_212", 5),
                 ("block_13", "v_289", 5), ("block_13", "v_366", 2)];
    shipped_order_sites("gpt2_decode_python_n32_m16i_fcf_cm", &progs, &sites);
}

/// Decode-8's plan without CutMax's scaled normalisers: the ILP placer, one optimum on every machine
/// (PLAN_ILP_TIEBREAK), on the merged capture of the softmax and GELU fitted to its 16 positions (gpt2_base_n32_kc16):
/// two down-projection hints past the entry, the argmax's v_2 at its 7.
#[test]
fn gpt2_decode_kc16() {
    let Some(progs) = gate("gpt2_decode_python_n32_kc16", "gpt2_decode_python_n32_kc16m") else { return };
    let sites = [("block_2", "v_1136", 2), ("block_11", "v_1182", 2), ("block_13", "v_2", 7)];
    shipped_order_sites("gpt2_decode_python_n32_kc16", &progs, &sites);
}

/// The plan speedrun/server.py serves Decode-8: kc16's blocks, CutMax's scaled normalisers in block 13 (planned at its
/// block-12 exit): kc16's two hints and block 13's five past the entry.
#[test]
fn gpt2_decode_kc16_cm() {
    let Some(progs) = gate("gpt2_decode_python_n32_kc16_cm", "gpt2_decode_python_n32_kc16_cm") else { return };
    let sites = [("block_2", "v_1136", 2), ("block_11", "v_1182", 2), ("block_13", "v_16", 2), ("block_13", "v_135", 5),
                 ("block_13", "v_212", 5), ("block_13", "v_289", 5), ("block_13", "v_366", 2)];
    shipped_order_sites("gpt2_decode_python_n32_kc16_cm", &progs, &sites);
}

/// The plan speedrun/server.py serves Decode-8: gpt2_decode_python_n32_kc16r's recipe and captures (softmax centred on
/// per-head reference keys) on the lean chain n32a4 (its runtime line's CHAIN_PRESET; make_n32a4_decode_plan.sh),
/// CutMax's scaled normalisers in block 13 on the ILP: three down-projection hints and block 13's five past the entry.
#[test]
fn gpt2_decode_n32a4_kc16r_cm() {
    let Some(progs) = gate("gpt2_decode_python_n32a4_kc16r_cm", "gpt2_decode_python_n32_kc16r_cm") else { return };
    let sites = [("block_2", "v_1120", 2), ("block_9", "v_921", 2), ("block_11", "v_1182", 2), ("block_13", "v_16", 2),
                 ("block_13", "v_135", 5), ("block_13", "v_212", 5), ("block_13", "v_289", 5), ("block_13", "v_366", 2)];
    shipped_order_sites("gpt2_decode_python_n32a4_kc16r_cm", &progs, &sites);
}

/// Decode-8's served plan, CutMax's `_cmfl` (ln_f folded into the LM head): blocks 0-11 on the kc16r capture, stages 12
/// and 13 on the `_cmfl` capture (its PLAN_CMD.txt `graph=` line), joined in one graph dir.
#[test]
fn gpt2_decode_n32a4_kc16r_cmfl() {
    let g = assets().join("graphs");
    let (base, tail) = (g.join("gpt2_decode_python_n32_kc16r_cm"), g.join("gpt2_decode_python_n32_kc16r_cmfl"));
    if !base.join("block_0/graph.json").is_file() || !tail.join("block_12/graph.json").is_file() {
        println!("SKIP gpt2_decode_python_n32a4_kc16r_cmfl: graphs not on disk");
        return;
    }
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("graph_kc16r_cmfl");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for b in 0..14 {
        let src = if b < 12 { &base } else { &tail };
        std::os::unix::fs::symlink(src.join(format!("block_{b}")), dir.join(format!("block_{b}"))).unwrap();
    }
    for f in ["capture_env.json", "capture_config.json"] {
        std::fs::copy(tail.join(f), dir.join(f)).unwrap();
    }
    let Some(progs) = gate("gpt2_decode_python_n32a4_kc16r_cmfl", dir.to_str().unwrap()) else { return };
    let sites = [("block_2", "v_1120", 2), ("block_9", "v_921", 2), ("block_11", "v_1182", 2)];
    shipped_order_sites("gpt2_decode_python_n32a4_kc16r_cmfl", &progs, &sites);
}

/// Decode-8's served plan on the alternating chain n32sp (speedrun/make_n32sp_decode_plan.sh): stages 0-12 on the merged
/// capture, CutMax's stage 13 on its `_cm` capture, joined in one graph dir. Its refreshes all land where EvalMod stays
/// above the user region (the compile's `[user_region]` check, part of the 0 errors).
#[test]
fn gpt2_decode_n32sp_kc16r_lh() {
    let g = assets().join("graphs");
    let (base, tail) = (g.join("gpt2_decode_python_n32sp_kc16r_lhm"), g.join("gpt2_decode_python_n32sp_kc16r_lh_cm"));
    if !base.join("block_0/graph.json").is_file() || !tail.join("block_13/graph.json").is_file() {
        println!("SKIP gpt2_decode_python_n32sp_kc16r_lh: graphs not on disk");
        return;
    }
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("graph_n32sp_lh");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for b in 0..14 {
        let src = if b < 13 { &base } else { &tail };
        std::os::unix::fs::symlink(src.join(format!("block_{b}")), dir.join(format!("block_{b}"))).unwrap();
    }
    for f in ["capture_env.json", "capture_config.json"] {
        std::fs::copy(base.join(f), dir.join(f)).unwrap();
    }
    gate("gpt2_decode_python_n32sp_kc16r_lh", dir.to_str().unwrap());
}

#[test]
fn gpt2_prefill() {
    gate("gpt2_prefill_python_n32_pcmmstcf", "gpt2_prefill_python_n32_pcmmstcf");
}

/// PackedPrefill's Prefill-128 plan, blocks and tail.
#[test]
fn gpt2_prefill_speedrun_pf() {
    gate("gpt2_prefill128_pf", "gpt2_prefill128_pf");
}

/// gpt2_prefill128_pf with CutMax's scaled normalisers in stage 13.
#[test]
fn gpt2_prefill_speedrun_pf_cm() {
    gate("gpt2_prefill128_pf_cm", "gpt2_prefill128_pf_cm");
}

/// BabyStepPrefill's Prefill-128 plan: speedrun/make_prefill_plan.sh's, CutMax's scaled normalisers in stage 13.
#[test]
fn gpt2_prefill_speedrun_pfbhrd_cm() {
    gate("gpt2_prefill128_pfbhrd_cm", "gpt2_prefill128_pfbh_cm");
}

/// The Prefill-128 plan the speedrun serves (speedrun/server.py): speedrun/make_prefill_plan.sh PRESET=n32a4's on the
/// same captures, on the lean chain n32a4 (its runtime line's CHAIN_PRESET), CutMax's stage 13 on the ILP.
#[test]
fn gpt2_prefill_speedrun() {
    gate("gpt2_prefill128_n32a4_pfbhrd_cm", "gpt2_prefill128_pfbh_cm");
}

#[test]
fn llama_prefill() {
    gate("llama31_8b_prefill_python_n32_pcmmstcf", "llama31_8b_prefill_python_n32_pcmmstcf");
}

/// Informational: the two schedules side by side (launches, bootstrap launches, arena), every block compiled
/// and checked under each.
#[test]
fn schedules() {
    use pc_program::compile::batch::Schedule;
    for (plan, graph) in [
        ("gpt2_decode_python_n32", "gpt2_decode_python_n32_gcheb"),
        ("gpt2_prefill_python_n32_pcmmstcf", "gpt2_prefill_python_n32_pcmmstcf"),
        ("llama31_8b_prefill_python_n32_pcmmstcf", "llama31_8b_prefill_python_n32_pcmmstcf"),
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
