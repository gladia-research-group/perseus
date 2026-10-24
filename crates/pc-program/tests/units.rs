//! The structure the replay's units rely on, on the shipped GPT-2 plans (CPU only; a plan or graph that is not on
//! disk is skipped):
//!   * `Schedule::Siblings` only reorders: the same member ops as `Schedule::Program`, every op after its inputs'
//!     producers, a batch's members mutually independent, no check error;
//!   * rotation grids: rotation ops only, at most `GRID_INPUTS` inputs of one level, none produced inside the run,
//!     every output in a slot of its own that is no input's;
//!   * rotation chains: every link rotates the chain's inputs; only element-wise ops and rescales between the links of
//!     a chain kept in place; held chains do not overlap; no op between a chain's first and last link writes an
//!     input's slot.
use pc_program::compile::batch::Schedule;
use pc_program::compile::{compile_dir, CompileOpts};
use pc_program::program::{Kind, Op, Program, VarId, GRID_INPUTS};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

fn assets() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn compile(plan: &str, graph: &str, schedule: Schedule) -> Option<Vec<(String, Program)>> {
    let pd = assets().join("bootstrap_placements").join(plan);
    let gd = assets().join("graphs").join(graph);
    if !pd.join("block_0_placement.json").is_file() || !gd.join("block_0/graph.json").is_file() {
        println!("SKIP {plan}: plan or graph not on disk");
        return None;
    }
    let dc = compile_dir(&pd, &gd, &CompileOpts { keep_going: true, schedule, ..Default::default() }).expect("plan dir");
    // blocks whose graph is not shipped (the prefill's tail stages) do not compile here
    Some(dc.programs.into_iter().filter_map(|(l, r)| r.ok().map(|p| (l, p))).collect())
}

fn member_set(p: &Program) -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    for x in p.ops.iter().flat_map(|o| o.members()) {
        *m.entry(format!("{:?}|{:?}|{:?}|{:?}", x.kind, x.ins, x.outs, x.pre)).or_default() += 1;
    }
    m
}

fn siblings(plan: &str, graph: &str) {
    let (Some(a), Some(b)) = (compile(plan, graph, Schedule::Program), compile(plan, graph, Schedule::Siblings)) else { return };
    assert_eq!(a.len(), b.len());
    for ((la, pa), (lb, pb)) in a.iter().zip(&b) {
        assert_eq!(la, lb);
        assert!(pb.check.errors.is_empty(), "{lb}: {:?}", pb.check.errors.first());
        assert_eq!(member_set(pa), member_set(pb), "{lb}: the schedules run different ops");
        let mut made: BTreeSet<VarId> = pb.inputs.iter().copied().collect();
        for op in &pb.ops {
            let outs: BTreeSet<VarId> = op.all_outs().into_iter().collect();
            for v in op.all_ins() {
                assert!(made.contains(&v), "{lb}: {} reads var {v} before it is produced", op.members()[0].kind.name());
                assert!(!outs.contains(&v), "{lb}: a batch reads what one of its members writes");
            }
            made.extend(outs);
        }
    }
}

fn rot(op: &Op) -> bool {
    op.members().iter().all(|m| matches!(m.kind, Kind::Rotate(k) if k != 0))
}

fn quiet(op: &Op) -> bool {
    op.members().iter().all(|m| matches!(m.kind, Kind::Add | Kind::Sub | Kind::AddPt(_) | Kind::MultPt(_) | Kind::AddConst(_) | Kind::MultConst(_) | Kind::Negate | Kind::MultI | Kind::Rescale | Kind::DropTo(_)))
}

fn grids_and_chains(plan: &str, graph: &str, schedule: Schedule) {
    let Some(progs) = compile(plan, graph, schedule) else { return };
    let (mut n_grids, mut n_chains) = (0, 0);
    for (label, p) in &progs {
        let slot = |v: VarId| p.vars[v as usize].slot;
        for g in p.rotation_grids() {
            n_grids += 1;
            let ops = &p.ops[g.ops.0..g.ops.1];
            assert!(ops.iter().all(rot), "{label}: a grid over a non-rotation op");
            assert!(!g.ins.is_empty() && g.ins.len() <= GRID_INPUTS);
            let level = p.vars[g.ins[0] as usize].meta.level;
            assert!(g.ins.iter().all(|&x| p.vars[x as usize].meta.level == level), "{label}: grid inputs at several levels");
            let made: BTreeSet<VarId> = ops.iter().flat_map(|o| o.all_outs()).collect();
            assert!(g.ins.iter().all(|x| !made.contains(x)), "{label}: a grid input produced inside the grid");
            let in_slots: BTreeSet<Option<u32>> = g.ins.iter().map(|&x| slot(x)).collect();
            let outs: Vec<Option<u32>> = ops.iter().flat_map(|o| o.all_outs()).map(slot).collect();
            let distinct: BTreeSet<Option<u32>> = outs.iter().copied().collect();
            assert_eq!(distinct.len(), outs.len(), "{label}: two grid outputs share a slot");
            assert!(outs.iter().all(|s| s.is_some() && !in_slots.contains(s)), "{label}: a grid output in an input's slot");
        }
        let chains = p.rotation_chains();
        let mut held_spans: Vec<(usize, usize)> = vec![];
        for c in &chains {
            n_chains += 1;
            assert!(c.ops.len() >= 2 && c.ins.len() <= GRID_INPUTS);
            let set: BTreeSet<VarId> = c.ins.iter().copied().collect();
            let in_slots: BTreeSet<Option<u32>> = c.ins.iter().map(|&x| slot(x)).collect();
            for &i in &c.ops {
                let op = &p.ops[i];
                assert!(rot(op), "{label}: a chain link that is not a rotation");
                let ins: BTreeSet<VarId> = op.all_ins().into_iter().collect();
                assert_eq!(ins, set, "{label}: a chain link rotating other inputs");
                assert!(op.all_outs().iter().all(|&o| !in_slots.contains(&slot(o))), "{label}: a link writes an input's slot");
            }
            let (first, last) = (c.ops[0], *c.ops.last().expect("links"));
            for (i, op) in p.ops.iter().enumerate().take(last).skip(first + 1) {
                assert!(op.all_outs().iter().all(|&o| !in_slots.contains(&slot(o))), "{label}: op {i} writes a chain input's slot");
                if !c.held && !c.ops.contains(&i) {
                    assert!(quiet(op), "{label}: a key-switching op between the links of an in-place chain");
                }
            }
            if c.held {
                held_spans.push((first, last));
            }
        }
        held_spans.sort_unstable();
        assert!(held_spans.windows(2).all(|w| w[0].1 < w[1].0), "{label}: held chains overlap");
    }
    println!("{plan} {schedule:?}: {n_grids} grids, {n_chains} chains checked");
}

#[test]
fn siblings_only_reorders() {
    siblings("gpt2_prefill128_m16s0_tail", "gpt2_prefill128_stcf_m16");
    siblings("gpt2_decode_python_n32_m16i", "gpt2_decode_python_n32_gcheb");
}

#[test]
fn grid_and_chain_invariants() {
    grids_and_chains("gpt2_prefill128_m16s0_tail", "gpt2_prefill128_stcf_m16", Schedule::Siblings);
    grids_and_chains("gpt2_prefill128_m16s0_tail", "gpt2_prefill128_stcf_m16", Schedule::Program);
    grids_and_chains("gpt2_decode_python_n32_m16i", "gpt2_decode_python_n32_gcheb", Schedule::Program);
}
