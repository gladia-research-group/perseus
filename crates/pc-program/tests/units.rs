//! The structure the replay's units rely on, on the shipped GPT-2 plans (CPU only; a plan or graph that is not on
//! disk is skipped):
//!   * `Schedule::Siblings` only reorders: the same member ops as `Schedule::Program`, every op after its inputs'
//!     producers, a batch's members mutually independent, no check error;
//!   * rotation grids: rotation ops only, at most `GRID_INPUTS` inputs of one level, none produced inside the run,
//!     every output in a slot of its own that is no input's;
//!   * rotation chains: every link rotates the chain's inputs; only element-wise ops and rescales between the links of
//!     a chain kept in place; held chains do not overlap; no op between a chain's first and last link writes an
//!     input's slot;
//!   * refresh pairs (`Pairing`) only reorder; a pair is two independent `Pairing::takes` refreshes of one level,
//!     degree and drop; a fused product (`Program::mult_pt_adds`) feeds only the next add and is no output.
use pc_program::compile::batch::{self, Schedule, PAIR_K};
use pc_program::compile::{compile_dir, CompileOpts};
use pc_program::program::{Kind, Op, Origin, Pairing, Program, VarId, GRID_INPUTS};
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

/// Every op after its inputs' producers; a batch's members independent.
fn ordered(label: &str, p: &Program) {
    let mut made: BTreeSet<VarId> = p.inputs.iter().copied().collect();
    for op in &p.ops {
        let outs: BTreeSet<VarId> = op.all_outs().into_iter().collect();
        for v in op.all_ins() {
            assert!(made.contains(&v), "{label}: {} reads var {v} before it is produced", op.members()[0].kind.name());
            assert!(!outs.contains(&v), "{label}: a batch reads what one of its members writes");
        }
        made.extend(outs);
    }
}

fn siblings(plan: &str, graph: &str) {
    let (Some(a), Some(b)) = (compile(plan, graph, Schedule::Program), compile(plan, graph, Schedule::Siblings)) else { return };
    assert_eq!(a.len(), b.len());
    for ((la, pa), (lb, pb)) in a.iter().zip(&b) {
        assert_eq!(la, lb);
        assert!(pb.check.errors.is_empty(), "{lb}: {:?}", pb.check.errors.first());
        assert_eq!(member_set(pa), member_set(pb), "{lb}: the schedules run different ops");
        ordered(lb, pb);
    }
}

/// The unbatched programs (capture order) batched again with refresh pairs on (n32: StC-first entry 46, unit 2).
fn pairs(plan: &str, graph: &str) {
    let pd = assets().join("bootstrap_placements").join(plan);
    let gd = assets().join("graphs").join(graph);
    if !pd.join("block_0_placement.json").is_file() || !gd.join("block_0/graph.json").is_file() {
        println!("SKIP {plan}: plan or graph not on disk");
        return;
    }
    let dc = compile_dir(&pd, &gd, &CompileOpts { keep_going: true, no_batch: true, ..Default::default() }).expect("plan dir");
    let pr = Pairing { entry: pc_shadow::BtsConfig::n32_shipped().stc_first_entry.expect("n32 StC-first entry"), unit: 2 };
    let mut n_pairs = 0;
    for (label, p) in dc.programs.iter().filter_map(|(l, r)| r.as_ref().ok().map(|p| (l, p))) {
        let mut q = p.clone();
        batch::batch(&mut q, 8, Schedule::Siblings, false, Some(pr));
        assert_eq!(member_set(p), member_set(&q), "{label}: pairing runs different ops");
        ordered(label, &q);
        let mut formed = 0;
        for op in &q.ops {
            let m = op.members();
            let takes = |x: &Op| batch::paired(&q, x, Some(pr));
            if m.len() > 1 && m.iter().any(takes) {
                assert!(m.len() == PAIR_K && m.iter().all(takes), "{label}: a pairable refresh in a batch that is not a pair");
                let site = |x: &Op| match x.kind {
                    Kind::Bootstrap(s) => (q.vars[x.ins[0] as usize].meta.level, q.vars[x.ins[0] as usize].meta.deg, q.bts_sites[s as usize].route.raise_drop),
                    _ => unreachable!(),
                };
                assert_eq!(site(&m[0]), site(&m[1]), "{label}: a pair of two levels, degrees or raise drops");
                formed += 1;
            }
        }
        assert_eq!(formed, q.batch_report.pairs, "{label}: the batch report's pair count");
        n_pairs += formed;
    }
    println!("{plan}: {n_pairs} refresh pairs checked");
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
fn refresh_pairs_only_reorder() {
    pairs("gpt2_prefill128_pf", "gpt2_prefill128_pf");
}

#[test]
fn grid_and_chain_invariants() {
    grids_and_chains("gpt2_prefill128_m16s0_tail", "gpt2_prefill128_stcf_m16", Schedule::Siblings);
    grids_and_chains("gpt2_prefill128_m16s0_tail", "gpt2_prefill128_stcf_m16", Schedule::Program);
    grids_and_chains("gpt2_decode_python_n32_m16i", "gpt2_decode_python_n32_gcheb", Schedule::Program);
}

fn fused_pairs(plan: &str, graph: &str) {
    let Some(progs) = compile(plan, graph, Schedule::Program) else { return };
    let mut per: Vec<String> = vec![];
    for (label, p) in &progs {
        let mut uses: BTreeMap<VarId, usize> = BTreeMap::new();
        for v in p.ops.iter().flat_map(|o| o.all_ins()) {
            *uses.entry(v).or_default() += 1;
        }
        let pairs = p.mult_pt_adds();
        for &i in &pairs {
            let (m, a) = (&p.ops[i], &p.ops[i + 1]);
            assert!(matches!(m.kind, Kind::MultPt(_)) && matches!(a.kind, Kind::Add | Kind::Sub), "{label}: op {i} is not a product and its add");
            let v = m.outs[0];
            assert_eq!(uses.get(&v), Some(&1), "{label}: a fused product read twice");
            assert!(a.ins.contains(&v) && !p.outputs.contains(&v), "{label}: a fused product the add does not read, or an output");
        }
        per.push(format!("{label} {}", pairs.len()));
    }
    println!("{plan}: fused pairs {}", per.join(", "));
}

fn op(kind: Kind, ins: Vec<VarId>, out: VarId) -> Op {
    Op { kind, ins, outs: vec![out], pre: vec![], origin: Origin::Node(out), inplace: false }
}

#[test]
fn mult_pt_add_pairs() {
    fused_pairs("gpt2_decode_python_n32_m16i_fcf", "gpt2_decode_python_n32_gcheb");
    fused_pairs("gpt2_prefill128_pf", "gpt2_prefill128_pf");
    // x = var 0, y = var 1: y + x pt and x pt - y fuse (the product first or second, add or sub); a product read
    // twice, a product that is an output, a product the next op does not read, a batched product do not
    let prog = |ops: Vec<Op>, outputs: Vec<VarId>| Program { ops, outputs, ..Default::default() };
    let p = prog(vec![op(Kind::MultPt(0), vec![0], 2), op(Kind::Add, vec![1, 2], 3), op(Kind::MultPt(1), vec![0], 4), op(Kind::Sub, vec![4, 1], 5)], vec![3, 5]);
    assert_eq!(p.mult_pt_adds(), vec![0, 2]);
    let twice = prog(vec![op(Kind::MultPt(0), vec![0], 2), op(Kind::Add, vec![1, 2], 3), op(Kind::Add, vec![2, 3], 4)], vec![4]);
    assert!(twice.mult_pt_adds().is_empty());
    let output = prog(vec![op(Kind::MultPt(0), vec![0], 2), op(Kind::Add, vec![1, 2], 3)], vec![3, 2]);
    assert!(output.mult_pt_adds().is_empty());
    let other = prog(vec![op(Kind::MultPt(0), vec![0], 2), op(Kind::Add, vec![1, 1], 3), op(Kind::Add, vec![2, 3], 4)], vec![4]);
    assert!(other.mult_pt_adds().is_empty());
    let batched = Op { kind: Kind::Batched(vec![op(Kind::MultPt(0), vec![0], 2)]), ins: vec![], outs: vec![], pre: vec![], origin: Origin::Node(2), inplace: false };
    let batch = prog(vec![batched, op(Kind::Add, vec![1, 2], 3)], vec![3]);
    assert!(batch.mult_pt_adds().is_empty());
}
