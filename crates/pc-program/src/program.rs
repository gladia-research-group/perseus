//! `Program`: one block compiled against its plan. SSA over `VarId`s, every op carrying the operand preludes the
//! Shadow decided (so the executor does no level bookkeeping), every var its `CtMeta` and its arena slot.
//!
//! On-disk cache: `Program::save` writes JSON (`block_<b>[_<variant>].program.json` next to the plan); JSON because
//! the workspace has serde_json and no binary codec, and a block program is ~1-2 MB. `Program::load` reads it back
//! and `cfg_stamp` lets a caller refuse a cache compiled under another Config.
use crate::error::{json, Error, Result};
use pc_shadow::route::Landing;
use pc_shadow::{CtMeta, Prelude, Route};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

pub type VarId = u32;
pub type PtId = u32;
pub type SiteId = u32;

/// Where an op came from (op accounting and diagnostics).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Origin {
    /// Graph node (index in the planner's order, auto_bootstraps erased).
    Node(u32),
    /// Graph nodes merged into one op (an external finish with several outputs).
    Nodes(Vec<u32>),
    /// A placement (`bootstrap_after_node`) on this var.
    Placement(String),
    /// A plan `rescale_after` anchor on this var.
    Realize(String),
}

/// A scalar operand. v1 graphs print `const(c)` with 6 significant digits: `exact` is false there, and the replay
/// must bind the value from the driver (v2 `args.scalar` is exact).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scalar {
    pub value: f64,
    pub exact: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Kind {
    Add,
    Sub,
    /// ct + pt (`pt_refs[id]`).
    AddPt(PtId),
    /// ct * pt.
    MultPt(PtId),
    /// ct + c (sub of a constant is `AddConst(-c)`).
    AddConst(Scalar),
    MultConst(Scalar),
    /// ct * ct, relinearized.
    Mult,
    Square,
    Negate,
    Rotate(i64),
    /// a + sum rot(a, k) over the steps (one hoisted radix group).
    RotateSum(Vec<i64>),
    Conjugate,
    MultI,
    /// Pay the pending rescale (plan anchor or landing realize).
    Rescale,
    /// Drop limbs to this level (`level_reduce`).
    DropTo(u32),
    /// A refresh: `bts_sites[site]`.
    Bootstrap(SiteId),
    /// A refresh made outside the recorded ops (PCMM/CCMM finish): `ins` -> `outs`, each landing at `landing`.
    /// `desc` = the PCMM site (`b<block>.<site>`, from pcmm_calib.json / v2 `args.desc`); None = not known.
    /// `outs`: every output's own landing when they differ (empty = all at `landing`).
    Finish {
        landing: Landing,
        desc: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        outs: Vec<Landing>,
    },
    /// K sibling ops with identical (kind, metas, route, const shape), mutually independent: one launch.
    Batched(Vec<Op>),
    /// `sum_i ins[i] * pt_refs[pts[i]]` over streamed weight rows (`PtForm::CoeffStream`): a streamed linear's
    /// baby-step inner sum, the loop of `MultPt` + `Add` the graph records fused into one op (perseus
    /// `mult_coeff_sum`). `fused`: the loop's intermediate vars (products, partial sums), never materialized; the
    /// check gives them their lowered metadata so the plan's pins on those names still bind.
    CoeffSum {
        pts: Vec<PtId>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        fused: Vec<VarId>,
    },
}

impl Kind {
    pub fn name(&self) -> &'static str {
        match self {
            Kind::Add => "add",
            Kind::Sub => "sub",
            Kind::AddPt(_) => "add_pt",
            Kind::MultPt(_) => "mult_pt",
            Kind::AddConst(_) => "add_const",
            Kind::MultConst(_) => "mult_const",
            Kind::Mult => "mult",
            Kind::Square => "square",
            Kind::Negate => "negate",
            Kind::Rotate(_) => "rotate",
            Kind::RotateSum(_) => "rotate_sum",
            Kind::Conjugate => "conjugate",
            Kind::MultI => "mult_i",
            Kind::Rescale => "rescale",
            Kind::DropTo(_) => "drop_to",
            Kind::Bootstrap(_) => "bootstrap",
            Kind::Finish { .. } => "finish",
            Kind::Batched(_) => "batched",
            Kind::CoeffSum { .. } => "coeff_sum",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Op {
    pub kind: Kind,
    /// Cipher operands (empty for `Batched`: the members carry theirs).
    pub ins: Vec<VarId>,
    pub outs: Vec<VarId>,
    /// What to do to the operands first (the Shadow recipe: `Operand::A` in place on `ins[0]`, `Operand::B` on a
    /// copy of `ins[1]` / the plaintext).
    pub pre: Vec<Prelude>,
    pub origin: Origin,
    /// The output may take `ins[0]`'s slot (the input dies here; alloc decided).
    #[serde(default)]
    pub inplace: bool,
}

impl Op {
    /// The ops this launch runs (itself, or a batch's members).
    pub fn members(&self) -> &[Op] {
        match &self.kind {
            Kind::Batched(m) => m,
            _ => std::slice::from_ref(self),
        }
    }
    pub fn all_ins(&self) -> Vec<VarId> {
        self.members().iter().flat_map(|m| m.ins.iter().copied()).collect()
    }
    pub fn all_outs(&self) -> Vec<VarId> {
        self.members().iter().flat_map(|m| m.outs.iter().copied()).collect()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Var {
    /// The graph name (`v_N`, `anon_k`, `sc_...`), or `<name>~pre` / `~raw` for the states before a planned
    /// refresh / realize (the graph name always holds the var's final, plan-checked state).
    pub name: String,
    pub meta: CtMeta,
    /// Arena slot (alloc); None until assigned.
    pub slot: Option<u32>,
}

/// How a plaintext operand is produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PtForm {
    /// Encoded at its use, like the ciphertext it meets (the Python impl's `encode_like`: effective level, deg 1);
    /// the driver supplies the values by name.
    AtUse,
    /// A named slot plaintext encoded once at a fixed level (`impl.<key>@<lvl>d<deg>`, v2 `pt.kind = slot`).
    Slot,
    /// The runtime's encode cache (masks, v2 `pt.kind = cached`).
    Cached,
    /// A streamed weight row (v2 `pt.kind = stream`, name `<linear>#<row>`): float32 coefficient rows on the host,
    /// expanded on the device inside a `Kind::CoeffSum`, at the ciphertext's level (like `AtUse`).
    CoeffStream,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PtRef {
    pub name: String,
    /// Encoding level and degree (resolved by the check for `AtUse`).
    pub level: u32,
    pub deg: u8,
    pub form: PtForm,
    /// The level the capture encoded it at (v1 `input_levels`), for diagnostics.
    pub captured_level: Option<u32>,
}

/// Who decided a refresh.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SiteKind {
    /// `bootstrap_after_node` placement.
    Placement,
    /// A plan-bound (or threshold) hint that fires.
    Hint,
    /// A captured `deliberate_bootstrap` the code itself asked for (route inferred from its captured landing).
    Deliberate,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BtsSite {
    /// The var the plan keys the site by (placement target / hint output / deliberate output).
    pub var: String,
    pub kind: SiteKind,
    pub route: Route,
    /// Input state, raw landing (before any realize), and whether the plan realizes it.
    pub input: CtMeta,
    pub landing: Landing,
    pub realize: bool,
    pub prescale: Option<f64>,
    pub offset: Option<f64>,
}

/// Arena layout: slot sizes in bytes (slot i at `offsets[i]`), total = `bytes`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Mem {
    pub slot_bytes: Vec<u64>,
    pub offsets: Vec<u64>,
    /// Arena size (sum of the slots).
    pub bytes: u64,
    /// Largest simultaneously-live ciphertext bytes (the lower bound any assignment needs).
    pub live_peak: u64,
    pub bytes_per_limb: u64,
    /// Every slot at the largest var size (chosen when that arena is smaller than best fit's).
    #[serde(default)]
    pub uniform: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct KindCount {
    pub kind: String,
    pub ops: usize,
    pub launches: usize,
}

/// What K-batching achieved.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BatchReport {
    pub k_max: u32,
    pub ops: usize,
    pub launches: usize,
    pub bootstraps: usize,
    pub bootstrap_launches: usize,
    /// Batch size -> number of batches (bootstraps).
    pub bootstrap_hist: std::collections::BTreeMap<usize, usize>,
    pub per_kind: Vec<KindCount>,
}

/// Op accounting against the graph (the identity `tests/shipped_plans.rs` checks).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Accounting {
    pub graph_nodes: usize,
    pub erased_autos: usize,
    pub placements: usize,
    /// Nodes lowered to no op: unfired hints, level_hints, no-op level_reduces (their output aliases the input).
    pub elided: usize,
    /// Extra nodes folded into a multi-output finish.
    pub merged: usize,
    /// Ops folded into a `CoeffSum` (all but one per sum).
    #[serde(default)]
    pub fused: usize,
    /// Placements consumed by a fired hint on the same var (the C++ refreshes once).
    pub placements_on_hints: usize,
    pub rescales: usize,
    /// Ops before batching.
    pub ops: usize,
    pub bootstraps: usize,
    pub hint_bootstraps: usize,
    pub deliberate_bootstraps: usize,
    pub finishes: usize,
}

impl Accounting {
    /// graph - autos + placements == ops + elided + merged + fused + placements_on_hints - rescales.
    pub fn balanced(&self) -> bool {
        let lhs = self.graph_nodes as i64 - self.erased_autos as i64 + self.placements as i64;
        let rhs = (self.ops + self.elided + self.merged + self.fused + self.placements_on_hints) as i64 - self.rescales as i64;
        lhs == rhs
    }
}

/// Check outcome (all errors are also a compile error unless the caller asked to keep going).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CheckReport {
    pub pinned: usize,
    pub level_exact: usize,
    pub level_richer: usize,
    pub errors: Vec<String>,
    /// AtUse plaintexts whose resolved level differs from the captured one.
    pub pt_relevels: usize,
    pub inexact_scalars: usize,
    /// `rescale_after` anchors on deliberate outputs: inert in the C++ and in the planner's sim.
    #[serde(default)]
    pub ignored_anchors: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub block: String,
    pub ops: Vec<Op>,
    pub vars: Vec<Var>,
    /// Block inputs in the driver's order (`anon_*` by first use).
    pub inputs: Vec<VarId>,
    /// The plan's exit var first, then every other var nothing in the block consumes.
    pub outputs: Vec<VarId>,
    /// The final binding of every graph name (aliases included: an unfired hint's output is its input).
    pub names: BTreeMap<String, VarId>,
    /// The plan's `final_named_levels/degs` for the names of this block.
    pub pins: BTreeMap<String, (u32, u8)>,
    pub pt_refs: Vec<PtRef>,
    pub bts_sites: Vec<BtsSite>,
    pub mem: Mem,
    /// `Config::load_bearing()` + the PlanConfig, JSON.
    pub cfg_stamp: String,
    pub batch_report: BatchReport,
    pub accounting: Accounting,
    pub check: CheckReport,
}

/// Ciphertexts a rotation grid reads (pc-gpu `ks::KEY_KMAX`: one inner-product launch serves them all).
pub const GRID_INPUTS: usize = 4;

/// A run of consecutive rotation ops the replay executes as one grid (pc-gpu `rotate::rotate_grid`): ops
/// `ops.0..ops.1`, the ciphertexts they rotate and their distinct steps (both in first-use order).
#[derive(Clone, Debug, PartialEq)]
pub struct GridSpec {
    pub ops: (usize, usize),
    pub ins: Vec<VarId>,
    pub steps: Vec<i64>,
}

/// Rotation ops of the same inputs that share one ModUp (`Program::rotation_chains`): their op indices, the inputs
/// and the distinct steps (first-use order); `held`: the ModUp lives in buffers of the chain's own.
#[derive(Clone, Debug, PartialEq)]
pub struct ChainSpec {
    pub ops: Vec<usize>,
    pub ins: Vec<VarId>,
    pub steps: Vec<i64>,
    pub held: bool,
}

impl Program {
    /// The rotation grids: maximal runs of consecutive rotation ops (single or K-batched) reading at most
    /// `GRID_INPUTS` ciphertexts of one level, none produced in the run, every output in a slot of its own that is no
    /// input's (alloc may hand a slot freed inside the run to a later output: the run stops there). A run is a grid
    /// when it rotates some input by two or more steps (a ModUp to share): the hoisted baby steps of a linear, or the
    /// compiler's K-batches of one step over the same few inputs, step after step.
    pub fn rotation_grids(&self) -> Vec<GridSpec> {
        let rot = |op: &Op| op.members().iter().all(|m| matches!(m.kind, Kind::Rotate(k) if k != 0) && m.pre.is_empty() && m.ins.len() == 1 && m.outs.len() == 1);
        let slot = |v: VarId| self.vars[v as usize].slot;
        let mut out = vec![];
        let mut i = 0;
        while i < self.ops.len() {
            if !rot(&self.ops[i]) {
                i += 1;
                continue;
            }
            let (mut ins, mut outs, mut made): (Vec<VarId>, Vec<Option<u32>>, Vec<VarId>) = (vec![], vec![], vec![]);
            let mut j = i;
            while j < self.ops.len() && rot(&self.ops[j]) {
                let (mut ins2, mut outs2) = (ins.clone(), outs.clone());
                let mut ok = true;
                for m in self.ops[j].members() {
                    let x = m.ins[0];
                    ok &= !made.contains(&x) && ins2.first().is_none_or(|&f| self.vars[f as usize].meta.level == self.vars[x as usize].meta.level);
                    if !ins2.contains(&x) {
                        ins2.push(x);
                    }
                    outs2.push(slot(m.outs[0]));
                }
                let in_slots: Vec<Option<u32>> = ins2.iter().map(|&x| slot(x)).collect();
                let mut seen = std::collections::BTreeSet::new();
                ok &= ins2.len() <= GRID_INPUTS && outs2.iter().all(|s| s.is_some() && !in_slots.contains(s) && seen.insert(*s));
                if !ok {
                    break;
                }
                made.extend(self.ops[j].all_outs());
                (ins, outs) = (ins2, outs2);
                j += 1;
            }
            if j == i {
                i += 1;
                continue;
            }
            let mut pairs: Vec<(VarId, i64)> = vec![];
            let mut steps: Vec<i64> = vec![];
            for m in self.ops[i..j].iter().flat_map(|o| o.members()) {
                if let Kind::Rotate(k) = m.kind {
                    if !steps.contains(&k) {
                        steps.push(k);
                    }
                    if !pairs.contains(&(m.ins[0], k)) {
                        pairs.push((m.ins[0], k));
                    }
                }
            }
            if ins.iter().any(|&x| pairs.iter().filter(|q| q.0 == x).count() >= 2) {
                out.push(GridSpec { ops: (i, j), ins, steps });
            }
            i = j;
        }
        out
    }

    /// The plaintext products consumed once, by the add or sub right after them (a BSGS baby sum's terms): op
    /// indices of the products. The pair can run as one pass (`Ops::mult_pt_add`), the product never written.
    pub fn mult_pt_adds(&self) -> Vec<usize> {
        let mut uses: BTreeMap<VarId, usize> = BTreeMap::new();
        for op in &self.ops {
            for v in op.all_ins() {
                *uses.entry(v).or_default() += 1;
            }
        }
        let mut out = vec![];
        for i in 0..self.ops.len().saturating_sub(1) {
            let (m, a) = (&self.ops[i], &self.ops[i + 1]);
            if !matches!(m.kind, Kind::MultPt(_)) || !matches!(a.kind, Kind::Add | Kind::Sub) || m.outs.len() != 1 || a.ins.len() != 2 {
                continue;
            }
            let v = m.outs[0];
            if uses.get(&v) == Some(&1) && a.ins.contains(&v) && !self.outputs.contains(&v) {
                out.push(i);
            }
        }
        out
    }

    /// The rotation chains: rotation ops outside the grids (single or K-batched, at most `GRID_INPUTS` ciphertexts of
    /// one level, outputs in slots of their own that are no input's) that rotate the same inputs. The first link
    /// gathers its inputs and runs their ModUp, the later links reuse both. A `held` chain keeps them in buffers of
    /// its own (one set: held chains do not overlap; the first ones in op order are taken), so anything may run
    /// between its links; any other chain keeps them in the key-switching workspaces and the gather buffer, so only
    /// element-wise ops and rescales (which leave both alone) separate its links. `ChainSpec::ops` lists the links
    /// (two or more); `ins` and `steps` are in first-use order, like a grid's.
    pub fn rotation_chains(&self) -> Vec<ChainSpec> {
        let grids = self.rotation_grids();
        let in_grid = |i: usize| grids.iter().any(|g| i >= g.ops.0 && i < g.ops.1);
        let slot = |v: VarId| self.vars[v as usize].slot;
        let quiet = |op: &Op| {
            op.members().iter().all(|m| {
                matches!(m.kind, Kind::Add | Kind::Sub | Kind::AddPt(_) | Kind::MultPt(_) | Kind::AddConst(_) | Kind::MultConst(_) | Kind::Negate | Kind::MultI | Kind::Rescale | Kind::DropTo(_))
            })
        };
        // a rotation op's inputs (first-use order), when it can be a chain link
        let link = |op: &Op| -> Option<Vec<VarId>> {
            let m = op.members();
            if !m.iter().all(|x| matches!(x.kind, Kind::Rotate(k) if k != 0) && x.pre.is_empty() && x.ins.len() == 1 && x.outs.len() == 1) {
                return None;
            }
            let mut ins: Vec<VarId> = vec![];
            for x in m {
                if !ins.contains(&x.ins[0]) {
                    ins.push(x.ins[0]);
                }
            }
            let level = self.vars[ins[0] as usize].meta.level;
            let in_slots: Vec<Option<u32>> = ins.iter().map(|&x| slot(x)).collect();
            let mut seen = std::collections::BTreeSet::new();
            let ok = ins.len() <= GRID_INPUTS
                && ins.iter().all(|&x| self.vars[x as usize].meta.level == level)
                && m.iter().all(|x| slot(x.outs[0]).is_some_and(|s| !in_slots.contains(&Some(s)) && seen.insert(s)));
            ok.then_some(ins)
        };
        let same = |a: &[VarId], b: &[VarId]| a.len() == b.len() && a.iter().all(|x| b.contains(x));
        let add = |c: &mut ChainSpec, i: usize| {
            c.ops.push(i);
            for m in self.ops[i].members() {
                if let Kind::Rotate(k) = m.kind {
                    if !c.steps.contains(&k) {
                        c.steps.push(k);
                    }
                }
            }
        };
        let links: Vec<Option<Vec<VarId>>> = self.ops.iter().enumerate().map(|(i, op)| if in_grid(i) { None } else { link(op) }).collect();
        // held chains: every link of one input set, spans [first, last] taken in order when they do not overlap
        let mut by_set: Vec<ChainSpec> = vec![];
        for (i, l) in links.iter().enumerate() {
            if let Some(ins) = l {
                match by_set.iter_mut().find(|c| same(&c.ins, ins)) {
                    Some(c) => add(c, i),
                    None => {
                        let mut c = ChainSpec { ops: vec![], ins: ins.clone(), steps: vec![], held: true };
                        add(&mut c, i);
                        by_set.push(c);
                    }
                }
            }
        }
        let mut out: Vec<ChainSpec> = vec![];
        let mut end = None;
        for c in by_set.into_iter().filter(|c| c.ops.len() >= 2) {
            if end.is_none_or(|e| c.ops[0] > e) {
                end = c.ops.last().copied();
                out.push(c);
            }
        }
        let taken: std::collections::BTreeSet<usize> = out.iter().flat_map(|c| c.ops.iter().copied()).collect();
        // the other links: chains of consecutive links of one input set, only quiet ops between them
        let mut cur: Option<ChainSpec> = None;
        let close = |cur: &mut Option<ChainSpec>, out: &mut Vec<ChainSpec>| {
            if let Some(c) = cur.take().filter(|c| c.ops.len() >= 2) {
                out.push(c);
            }
        };
        for (i, op) in self.ops.iter().enumerate() {
            match links[i].as_ref().filter(|_| !taken.contains(&i)) {
                Some(ins) => {
                    if !cur.as_ref().is_some_and(|c| same(&c.ins, ins)) {
                        close(&mut cur, &mut out);
                        cur = Some(ChainSpec { ops: vec![], ins: ins.clone(), steps: vec![], held: false });
                    }
                    add(cur.as_mut().expect("a chain is open"), i);
                }
                None if taken.contains(&i) || quiet(op) => {}
                None => close(&mut cur, &mut out),
            }
        }
        close(&mut cur, &mut out);
        out
    }

    /// The var a graph name is bound to at the end of the block.
    pub fn var(&self, name: &str) -> Option<VarId> {
        self.names.get(name).copied()
    }
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let p = path.as_ref();
        let s = serde_json::to_vec(self).map_err(|err| Error::Json { path: p.display().to_string(), err })?;
        std::fs::write(p, s).map_err(|err| Error::Io { path: p.display().to_string(), err })
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Program> {
        json(path.as_ref())
    }
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|err| Error::Json { path: self.block.clone(), err })
    }
    pub fn from_json(s: &str) -> Result<Program> {
        serde_json::from_str(s).map_err(|err| Error::Json { path: "<str>".into(), err })
    }
}
