//! `bind`: a compiled `Program` on a device runtime, once, at setup (before `Arena::freeze`).
//!
//! * every `VarId` -> its compiler-assigned slot, one full-layout ciphertext buffer each (`CtBuf::words(lay, 1)`),
//!   plus one scratch ciphertext for the in-place ops without an in-place kernel;
//! * every `PtRef` -> a `PtStore` key `(name, level, Eval)` (`resolve_pts` -> device row addresses);
//! * every rotation step list / conjugation / relinearization -> a setup-built `KeyGroup`;
//! * every refresh site -> its `Route`; every scalar -> an exact `f64`;
//! * every streamed weight row (`PtForm::CoeffStream`) -> its row in the block's weight frame (`StreamLayout`);
//! * the per-op argument table (`Step`): kind, slot indices, the compiled output metadata;
//! * the steps issued together (`Unit`): rotation grids and chains (`Program::rotation_grids` / `rotation_chains`),
//!   K-batches, plaintext products with their add (`mult_pt_adds`). Other batches: member by member.
//! Details in docs/ARCHITECTURE.md "pc-program bind".
use crate::error::{Error, Result};
use crate::program::{Kind, Op, Origin, Program, Scalar, GRID_INPUTS};
use crate::replay::Routes;
use pc_cuda::{Arena, Compute, Stream};
use pc_gpu::bts::{BtsKeyRing, RouteKind};
use pc_gpu::encode_dev::PtForm as DevForm;
use pc_gpu::ks::{KeyGroup, KsKeyRef};
use pc_gpu::pt_store::{PtKey, PtStore};
use pc_gpu::{Ct, CtBuf, DevStrategy, DevWord, Layout};
use pc_shadow::{CtMeta, PtMeta, Route};
use std::collections::BTreeMap;
use std::ffi::c_void;

/// No operand.
pub const NONE: u32 = u32::MAX;

/// What one step runs (indices into the `Bound` tables).
#[derive(Clone, Debug, PartialEq)]
pub enum StepOp {
    Add,
    Sub,
    AddPt(u32),
    MultPt(u32),
    AddConst(f64),
    MultConst(f64),
    Mult,
    Square,
    Negate,
    /// Rotation by `k` through key group `g`.
    Rotate { k: i64, g: u32 },
    /// `a + sum rot(a, s)` over step list `l` (`Bound::step_lists`) through key group `g`.
    RotateSum { l: u32, g: u32 },
    Conjugate { g: u32 },
    MultI,
    Rescale,
    DropTo(u32),
    /// Refresh site `site` of the program on `route` (family `kind`).
    Bootstrap { site: u32, route: Route, kind: RouteKind },
    /// External finish `f` (`Bound::finishes`): a PCMM / CCMM op with several inputs and outputs, run by the
    /// runtime's `replay::FinishExec`.
    Finish { f: u32 },
    /// Streamed-weight sum `c` (`Bound::coeff_sums`), run by the runtime's `CoeffRowsDevice` on the rows of the
    /// block's weight frame (`Bound::stream_base`).
    CoeffSum { c: u32 },
}

/// A `Kind::CoeffSum` as bound: the term ciphertexts' slots and their rows' offsets in the weight frame (rows of N
/// floats).
#[derive(Clone, Debug, PartialEq)]
pub struct CoeffSumStep {
    pub ins: Vec<u32>,
    pub rows: Vec<u32>,
}

/// The weight frame of a bound block: every linear it streams (first-use order) at a row offset, whole (`rows` =
/// the linear's row count, whether or not the block reads every row); `total` rows.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StreamLayout {
    pub linears: Vec<(String, u32, u32)>,
    pub total: u32,
}

/// Row counts of the streamed linears the runtime holds (`<linear>` of the `<linear>#<row>` plaintext names).
pub trait StreamSource {
    fn rows(&self, linear: &str) -> Option<usize>;
}

/// `<linear>#<row>` -> (linear, row).
pub fn split_stream(name: &str) -> Option<(&str, u32)> {
    name.rsplit_once('#').and_then(|(l, k)| k.parse::<u32>().ok().map(|k| (l, k)))
}

/// The weight frame of `p`: its streamed linears in first-use order, back to back, whole. Err = the linears `src`
/// does not hold.
pub fn stream_layout(p: &Program, src: &dyn StreamSource) -> std::result::Result<StreamLayout, Vec<String>> {
    let mut out = StreamLayout::default();
    let mut missing: Vec<String> = vec![];
    for op in &p.ops {
        for m in op.members() {
            let Kind::CoeffSum { pts, .. } = &m.kind else { continue };
            for &id in pts {
                let Some((lin, _)) = p.pt_refs.get(id as usize).and_then(|r| split_stream(&r.name)) else { continue };
                if out.linears.iter().any(|x| x.0 == lin) || missing.iter().any(|x| x == lin) {
                    continue;
                }
                match src.rows(lin) {
                    Some(n) => {
                        out.linears.push((lin.to_string(), out.total, n as u32));
                        out.total += n as u32;
                    }
                    None => missing.push(lin.to_string()),
                }
            }
        }
    }
    if missing.is_empty() {
        Ok(out)
    } else {
        Err(missing)
    }
}

/// Limbs (top + 1) of the deepest-reaching streamed-weight sum of `p` (the expander's scratch rows), 0 = none.
pub fn coeff_rows_needed(p: &Program, shape: pc_shadow::ChainShape) -> usize {
    let mut r = 0;
    for op in &p.ops {
        for m in op.members() {
            if let Kind::CoeffSum { .. } = m.kind {
                if let Some(&v) = m.ins.first() {
                    if let Ok(t) = pc_shadow::meta::top_limb(shape, p.vars[v as usize].meta.level) {
                        r = r.max(t as usize + 1);
                    }
                }
            }
        }
    }
    r
}

/// What a unit runs.
#[derive(Clone, Debug, PartialEq)]
pub enum UnitKind {
    /// A rotation grid (`rotate::rotate_grid`): its distinct steps and their key group `g`, per step of the unit its
    /// (input, step) indices into `Unit::ins` / `steps`. `fresh` false: a later link of a rotation chain, which reuses
    /// the gather and ModUp of the chain's first; `held`: a held chain's link (`Held` buffers).
    Grid { steps: Vec<i64>, g: u32, at: Vec<(u8, u16)>, fresh: bool, held: bool },
    /// A K-batch of rotate_sums of step list `l` through key group `g` (`rotate::rotate_sum_batch`): step `lo + c`
    /// reads `Unit::ins[c]`.
    RotateSum { l: u32, g: u32 },
    /// A K-batch of relinearized products (`Ops::mult`, `b` empty: `Ops::square`): step `lo + c` multiplies
    /// `Unit::ins[c]` by `b[c]`.
    Mult { b: Vec<u32> },
    /// A K-batch of refreshes on one route (`BtsRoute::run` of K ciphertexts): step `lo + c` refreshes `Unit::ins[c]`.
    Refresh { route: Route, kind: RouteKind },
    /// A plaintext product and the add consuming it as one pass (`Ops::mult_pt_add`): step `lo` multiplies
    /// `Unit::ins[0]` by plaintext `pt`, step `lo + 1` adds (`sub`: subtracts) it and `Unit::ins[1]`, the product
    /// first when `product_first`.
    MultPtAdd { pt: u32, sub: bool, product_first: bool },
}

/// Steps `lo..hi` issued together (their outputs at once), reading the slots `ins`. `chain`: the rotation chain the
/// unit is a link of (NONE: none). `off`: a step of the unit (or of its chain) was made dynamic, so it issues step by
/// step.
#[derive(Clone, Debug, PartialEq)]
pub struct Unit {
    pub lo: u32,
    pub hi: u32,
    pub ins: Vec<u32>,
    pub kind: UnitKind,
    pub chain: u32,
    pub off: bool,
}

/// A `Kind::Finish` as bound: input and output slots (op order: output m = the descriptor's output m), the
/// compiled landing of every output, the descriptor (v2 `args.external` JSON, or a calibration site name).
#[derive(Clone, Debug, PartialEq)]
pub struct FinishStep {
    pub ins: Vec<u32>,
    pub outs: Vec<u32>,
    pub out_metas: Vec<CtMeta>,
    pub desc: String,
}

/// One launch member: op, slots, the compiled output metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub op: StepOp,
    pub out: u32,
    pub a: u32,
    pub b: u32,
    /// The output takes `a`'s slot.
    pub inplace: bool,
    /// Output var (diagnostics) and its compiled metadata.
    pub var: u32,
    pub meta: CtMeta,
    /// Compiled metadata of the operands in `a` / `b` (a dynamic step issued between two graph launches reads
    /// them: a launch does not move the host-side metadata of the slots it wrote).
    pub a_meta: Option<CtMeta>,
    pub b_meta: Option<CtMeta>,
}

/// Ciphertext slots shared by every bound block of a runtime (`Bound::new_shared`): blocks run one after the other
/// on one lane, so one slab of the largest block's slot count serves them all (`Bound::with_slab`); what crosses a
/// block boundary (the exit, KV states) is copied out by the driver.
pub struct Slab<'a, W: DevWord> {
    pub slots: Vec<Ct<'a, W>>,
    pub tmp: Option<Ct<'a, W>>,
    /// The units' gather buffers.
    pub gather: Option<Gather<'a, W>>,
    /// The held rotation chains' buffers (`Program::rotation_chains`), when some program has one.
    pub held: Option<Held<'a, W>>,
}

/// Where a unit's members meet (`GRID_INPUTS` ciphertexts each): their first operands, their second operands, their
/// outputs (a K-batched product's).
pub struct Gather<'a, W: DevWord> {
    pub a: Ct<'a, W>,
    pub b: Ct<'a, W>,
    pub out: Ct<'a, W>,
}

impl<'a, W: DevWord> Gather<'a, W> {
    pub fn new(arena: &'a Arena, lay: Layout) -> Result<Self> {
        let ct = || -> Result<Ct<'a, W>> { Ok(Ct::new(CtMeta::new(0, 1), CtBuf::new(arena, "replay.gather", lay, GRID_INPUTS)?)) };
        Ok(Gather { a: ct()?, b: ct()?, out: ct()? })
    }
    pub fn bytes(lay: &Layout) -> usize {
        3 * Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(lay, GRID_INPUTS))
    }
}

/// What a held rotation chain keeps between its links: its gathered inputs (`GRID_INPUTS` ciphertexts) and their
/// ModUp (a key-switching ModUp workspace, `ks::KsWork::digits`, swapped in for the chain's links).
pub struct Held<'a, W: DevWord> {
    pub x: Ct<'a, W>,
    pub digits: pc_cuda::DBuf<'a, Compute, W>,
}

impl<'a, W: DevWord> Slab<'a, W> {
    /// Bytes of a slab of `n` slots; `held_words`: the ModUp workspace's words (`ks::KsWork::words`) when a program
    /// has a held rotation chain, else 0.
    pub fn bytes(lay: &Layout, n: usize, held_words: usize) -> usize {
        let ct = Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(lay, 1));
        let held = if held_words > 0 { Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(lay, GRID_INPUTS)) + Arena::bytes_for::<W>(held_words) } else { 0 };
        (n + 1) * ct + Gather::<W>::bytes(lay) + held
    }
    pub fn new(arena: &'a Arena, lay: Layout, n: usize, held_words: usize) -> Result<Self> {
        let slots = (0..n).map(|_| Ok(Ct::new(CtMeta::new(0, 1), CtBuf::new(arena, "replay.slab", lay, 1)?))).collect::<Result<Vec<_>>>()?;
        let tmp = Some(Ct::new(CtMeta::new(0, 1), CtBuf::new(arena, "replay.slab", lay, 1)?));
        let gather = Some(Gather::new(arena, lay)?);
        let held = if held_words > 0 {
            Some(Held { x: Ct::new(CtMeta::new(0, 1), CtBuf::new(arena, "replay.slab", lay, GRID_INPUTS)?), digits: arena.alloc("replay.slab", held_words)? })
        } else {
            None
        };
        Ok(Slab { slots, tmp, gather, held })
    }
}

/// A bound plaintext operand: store key and, once resolved, its device rows.
#[derive(Clone, Debug)]
pub struct BoundPt {
    pub key: PtKey,
    pub meta: Option<PtMeta>,
    pub rows: *const c_void,
    /// A streamed weight row (read through the weight frame by a `CoeffSum`, never from the PtStore).
    pub stream: bool,
}

/// Exact scalars for the program's inexact (v1, 6-digit) literals, by graph node index (`Origin::Node`).
pub type ScalarTable = BTreeMap<u32, f64>;

/// What `bind` borrows from the runtime (setup phase: the arena is not frozen yet).
pub struct BindEnv<'e, 'a> {
    pub arena: &'a Arena,
    pub stream: &'e Stream<Compute>,
    pub lay: Layout,
    pub keys: &'e dyn BtsKeyRing,
    /// `Executor::ks.stage_ptrs()` (staging buffers for keys that need them).
    pub stage: Option<(*const c_void, *const c_void)>,
    pub scalars: &'e ScalarTable,
    /// The streamed linears' row counts (a program with a `CoeffSum` needs it).
    pub streams: Option<&'e dyn StreamSource>,
}

/// A program bound to one runtime: ciphertext slots, plaintext keys, key groups, routes, the step table, and its
/// captured graph (replay.rs). One `Bound` per (block, variant).
pub struct Bound<'a, W: DevWord> {
    pub label: String,
    /// Own slots (`Bound::new`), or empty for a block bound on a shared `Slab` (swapped in by `with_slab`).
    pub slots: Vec<Ct<'a, W>>,
    /// Scratch for in-place ops without an in-place kernel (copy, then out of place); the slab's while swapped in.
    pub tmp: Option<Ct<'a, W>>,
    /// The units' gather buffers; the slab's while swapped in.
    pub gather: Option<Gather<'a, W>>,
    /// The held rotation chains' buffers; the slab's while swapped in (None: those chains issue step by step).
    pub held: Option<Held<'a, W>>,
    /// Slots the program needs (`Program.mem`).
    pub n_slots: usize,
    /// Steps whose arguments the driver changes between runs (`set_scalar` / `set_rotation`): issued eagerly, the
    /// static segments between them replayed as graphs.
    pub dynamic: std::collections::BTreeSet<u32>,
    /// Key group of the rotations a dynamic rotate step may take (`Bound::new_shared(.., dyn_rot)`), and its steps.
    pub dyn_group: u32,
    pub dyn_steps: Vec<i64>,
    /// Per dynamic step: the operand and output metadata of its last eager issue (the compile leaves scales
    /// unknown; a capture and the replays between graph launches take these).
    pub dyn_metas: BTreeMap<u32, (Option<CtMeta>, Option<CtMeta>, CtMeta)>,
    pub steps: Vec<Step>,
    pub groups: Vec<KeyGroup<'a>>,
    /// Index of the relinearization group in `groups` (NONE when the block has no ct x ct product).
    pub relin: u32,
    pub step_lists: Vec<Vec<i64>>,
    /// Steps issued together, and per step the unit it belongs to (NONE: none).
    pub units: Vec<Unit>,
    pub unit_of: Vec<u32>,
    pub pts: Vec<BoundPt>,
    /// External finishes (`StepOp::Finish`).
    pub finishes: Vec<FinishStep>,
    /// Streamed-weight sums (`StepOp::CoeffSum`) and the weight frame they read.
    pub coeff_sums: Vec<CoeffSumStep>,
    pub streams: StreamLayout,
    /// Device address of the frame holding this block's rows (`residency::StreamRing::adopt`), null until staged.
    pub stream_base: *const f32,
    /// (graph name, slot, metadata) of the block inputs, in `Program.inputs` order (the driver writes them).
    pub inputs: Vec<(String, u32, CtMeta)>,
    /// (graph name, slot, metadata) of `Program.outputs` (exit var first).
    pub outputs: Vec<(String, u32, CtMeta)>,
    /// Var names by id (diagnostics).
    pub names: Vec<String>,
    /// Ring dimension N (route families).
    pub ring_n: usize,
    /// Captured graphs of this block, one per plaintext-row layout it ran with (most recent last; a weight ring
    /// may place a block's rows in a few different slot sets over a token cycle).
    pub(crate) graphs: Vec<crate::replay::Captured>,
}

fn berr(block: &str, msg: impl Into<String>) -> Error {
    Error::Bind { block: block.to_string(), msg: msg.into() }
}

fn list<T: std::fmt::Display>(v: &[T]) -> String {
    let head: Vec<String> = v.iter().take(12).map(|x| x.to_string()).collect();
    if v.len() > 12 {
        format!("{} (+{} more)", head.join(", "), v.len() - 12)
    } else {
        head.join(", ")
    }
}

/// What a program's key groups are: relinearization, conjugation, rotation step lists (dedup, first use order).
fn key_needs(p: &Program) -> (bool, bool, Vec<Vec<i64>>) {
    let (mut relin, mut conj) = (false, false);
    let mut lists: Vec<Vec<i64>> = vec![];
    for op in &p.ops {
        for m in op.members() {
            match &m.kind {
                Kind::Mult | Kind::Square => relin = true,
                Kind::Conjugate => conj = true,
                Kind::Rotate(k) => {
                    if !lists.contains(&vec![*k]) {
                        lists.push(vec![*k]);
                    }
                }
                Kind::RotateSum(s) => {
                    if !lists.contains(s) {
                        lists.push(s.clone());
                    }
                }
                _ => {}
            }
        }
    }
    (relin, conj, lists)
}

impl<'a, W: DevWord> Bound<'a, W> {
    /// Arena bytes `bind` carves for `p` (MemPlan section "replay"): its slots + scratch, its key groups.
    pub fn bytes(p: &Program, lay: &Layout) -> usize {
        let ct = Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(lay, 1));
        (p.mem.slot_bytes.len() + 1) * ct + Gather::<W>::bytes(lay) + Self::group_bytes(p, 0)
    }

    /// Arena bytes of `p`'s key groups (+ a dynamic-rotation group of `dyn_rot` steps): what `new_shared` carves.
    pub fn group_bytes(p: &Program, dyn_rot: usize) -> usize {
        let (relin, conj, lists) = key_needs(p);
        let grids: usize = p.rotation_grids().iter().map(|g| KeyGroup::bytes(g.steps.len(), true)).sum::<usize>() + p.rotation_chains().iter().map(|c| KeyGroup::bytes(c.steps.len(), true)).sum::<usize>();
        lists.iter().map(|l| KeyGroup::bytes(l.len().max(1), true)).sum::<usize>() + (relin as usize + conj as usize) * KeyGroup::bytes(1, true) + if dyn_rot > 0 { KeyGroup::bytes(dyn_rot, true) } else { 0 } + grids
    }

    /// `new` without slots: the block runs on a shared `Slab` (`with_slab`). `dyn_rot`: rotation steps a dynamic
    /// rotate step may be set to (`set_rotation`; one extra key group).
    pub fn new_shared<M: DevStrategy<W>>(p: &Program, env: &BindEnv<'_, 'a>, routes: &Routes<'_, 'a, W, M>, dyn_rot: &[i64]) -> Result<Self> {
        let mut b = Self::build(p, env, routes, false)?;
        let half = (env.lay.n / 2) as i64;
        let steps: Vec<i64> = dyn_rot.iter().copied().filter(|s| s.rem_euclid(half) != 0).collect();
        if !steps.is_empty() {
            let mut refs: Vec<&dyn KsKeyRef> = vec![];
            let mut missing = vec![];
            for &s in &steps {
                match env.keys.rot(s) {
                    Ok(k) => refs.push(k),
                    Err(_) => missing.push(s),
                }
            }
            if !missing.is_empty() {
                return Err(berr(&b.label, format!("dynamic rotation key(s) missing from the key ring: {}", list(&missing))));
            }
            b.dyn_group = b.groups.len() as u32;
            b.groups.push(KeyGroup::new(env.arena, env.stream, env.lay.n, &refs, env.stage)?);
            b.dyn_steps = steps;
        }
        Ok(b)
    }

    /// Bind `p` (see the module doc). `routes` must hold every route family the program's refresh sites use;
    /// plaintexts are resolved later (`resolve_pts`), so a weight ring can stage them after setup.
    pub fn new<M: DevStrategy<W>>(p: &Program, env: &BindEnv<'_, 'a>, routes: &Routes<'_, 'a, W, M>) -> Result<Self> {
        Self::build(p, env, routes, true)
    }

    fn build<M: DevStrategy<W>>(p: &Program, env: &BindEnv<'_, 'a>, routes: &Routes<'_, 'a, W, M>, carve: bool) -> Result<Self> {
        let block = p.block.as_str();
        let lay = env.lay;
        let n_slots = p.mem.slot_bytes.len();
        let slot_of = |v: u32| -> Result<u32> {
            p.vars.get(v as usize).and_then(|x| x.slot).ok_or_else(|| berr(block, format!("var {v} has no slot (alloc not run?)")))
        };
        // key groups
        let (need_relin, need_conj, lists) = key_needs(p);
        let mut groups: Vec<KeyGroup<'a>> = vec![];
        let mut missing: Vec<String> = vec![];
        let mut relin = NONE;
        if need_relin {
            match env.keys.relin() {
                Ok(k) => {
                    groups.push(KeyGroup::new(env.arena, env.stream, lay.n, &[k], env.stage)?);
                    relin = 0;
                }
                Err(_) => missing.push("relin".into()),
            }
        }
        let mut conj = NONE;
        if need_conj {
            match env.keys.conj() {
                Ok(k) => {
                    conj = groups.len() as u32;
                    groups.push(KeyGroup::new(env.arena, env.stream, lay.n, &[k], env.stage)?);
                }
                Err(_) => missing.push("conj".into()),
            }
        }
        let half = (lay.n / 2) as i64;
        let mut list_group: Vec<u32> = vec![];
        for l in &lists {
            let mut refs: Vec<&dyn KsKeyRef> = vec![];
            for &s in l.iter().filter(|s| s.rem_euclid(half) != 0) {
                match env.keys.rot(s) {
                    Ok(k) => refs.push(k),
                    Err(_) => missing.push(format!("rot {s}")),
                }
            }
            if refs.is_empty() {
                // every step is 0 mod N/2: the identity (rotate copies, rotate_sum refuses at run time)
                list_group.push(NONE);
            } else {
                list_group.push(groups.len() as u32);
                groups.push(KeyGroup::new(env.arena, env.stream, lay.n, &refs, env.stage)?);
            }
        }
        missing.dedup();
        if !missing.is_empty() {
            return Err(berr(block, format!("{} key(s) missing from the key ring: {}", missing.len(), list(&missing))));
        }
        let list_idx = |l: &Vec<i64>| -> u32 { lists.iter().position(|x| x == l).unwrap_or(0) as u32 };
        // plaintext keys (resolved later)
        // a block's own plaintexts (`pt_N`, encoded at use) are per-block names: namespaced, so a weight ring can hold
        // several blocks; masks and named slot plaintexts keep their shared names
        let pts: Vec<BoundPt> = p
            .pt_refs
            .iter()
            .map(|r| {
                // a graph's `pt_N` is numbered per block whatever its form (v2 names cached masks `pt_N` too)
                let stream = r.form == crate::program::PtForm::CoeffStream;
                let name = if !stream && (r.form == crate::program::PtForm::AtUse || r.name.starts_with("pt_")) { format!("{block}/{}", r.name) } else { r.name.clone() };
                BoundPt { key: PtKey::new(name, r.level, DevForm::Eval), meta: None, rows: std::ptr::null(), stream }
            })
            .collect();
        // steps
        let mut steps = vec![];
        let mut no_scalar: Vec<String> = vec![];
        let mut no_route: Vec<String> = vec![];
        let mut alias: Vec<String> = vec![];
        let mut finishes: Vec<FinishStep> = vec![];
        let mut coeff_sums: Vec<CoeffSumStep> = vec![];
        let streams = match env.streams {
            Some(src) => stream_layout(p, src).map_err(|missing| berr(block, format!("{} streamed linear(s) the runtime does not hold (StreamSource): {}", missing.len(), list(&missing))))?,
            None if p.ops.iter().any(|o| o.members().iter().any(|m| matches!(m.kind, Kind::CoeffSum { .. }))) => {
                return Err(berr(block, "the program streams weights (coeff_sum) and the runtime has no StreamSource"));
            }
            None => StreamLayout::default(),
        };
        let scalar = |m: &Op, c: &Scalar, no_scalar: &mut Vec<String>| -> f64 {
            if c.exact {
                return c.value;
            }
            if let Origin::Node(i) = m.origin {
                if let Some(&v) = env.scalars.get(&i) {
                    // the table holds the graph's scalar; a `sub` of a constant is lowered to add(-c)
                    return if (v - c.value).abs() <= (v + c.value).abs() { v } else { -v };
                }
            }
            no_scalar.push(format!("{:?}: {}", m.origin, c.value));
            c.value
        };
        let mut first_step: Vec<usize> = Vec::with_capacity(p.ops.len() + 1);
        for op in &p.ops {
            first_step.push(steps.len());
            for m in op.members() {
                let a = m.ins.first().map(|&v| slot_of(v)).transpose()?.unwrap_or(NONE);
                let b = m.ins.get(1).map(|&v| slot_of(v)).transpose()?.unwrap_or(NONE);
                if let Kind::Finish { desc, .. } = &m.kind {
                    let desc = desc.clone().ok_or_else(|| berr(block, format!("finish {:?} has no descriptor (v2 args.external)", m.origin)))?;
                    let ins = m.ins.iter().map(|&v| slot_of(v)).collect::<Result<Vec<_>>>()?;
                    let outs = m.outs.iter().map(|&v| slot_of(v)).collect::<Result<Vec<_>>>()?;
                    if outs.is_empty() {
                        return Err(berr(block, format!("finish {:?} without outputs", m.origin)));
                    }
                    let out_metas = m.outs.iter().map(|&v| p.vars[v as usize].meta).collect();
                    let var = m.outs[0];
                    finishes.push(FinishStep { ins: ins.clone(), outs: outs.clone(), out_metas, desc });
                    let vm = |k: usize| m.ins.get(k).map(|&v| p.vars[v as usize].meta);
                    // outputs may take dying inputs' slots: the service reads every input before writing an output
                    steps.push(Step { op: StepOp::Finish { f: finishes.len() as u32 - 1 }, out: outs[0], a: ins.first().copied().unwrap_or(NONE), b: NONE, inplace: false, var, meta: p.vars[var as usize].meta, a_meta: vm(0), b_meta: None });
                    continue;
                }
                if let Kind::CoeffSum { pts: ids, .. } = &m.kind {
                    let ins = m.ins.iter().map(|&v| slot_of(v)).collect::<Result<Vec<_>>>()?;
                    let var = *m.outs.first().ok_or_else(|| berr(block, format!("coeff_sum {:?} without an output", m.origin)))?;
                    let out = slot_of(var)?;
                    if ins.contains(&out) {
                        alias.push(format!("{:?}: coeff_sum output slot {out} is one of its terms", m.origin));
                    }
                    let mut rows = Vec::with_capacity(ids.len());
                    for &id in ids {
                        let r = p.pt_refs.get(id as usize).ok_or_else(|| berr(block, format!("plaintext {id} out of range")))?;
                        let (lin, row) = split_stream(&r.name).ok_or_else(|| berr(block, format!("stream plaintext {:?} is not <linear>#<row>", r.name)))?;
                        let (_, off, n) = streams.linears.iter().find(|x| x.0 == lin).ok_or_else(|| berr(block, format!("{lin} missing from the weight frame")))?;
                        if row >= *n {
                            return Err(berr(block, format!("{}: row {row} of a {n}-row streamed linear", r.name)));
                        }
                        rows.push(off + row);
                    }
                    coeff_sums.push(CoeffSumStep { ins: ins.clone(), rows });
                    let vm = |k: usize| m.ins.get(k).map(|&v| p.vars[v as usize].meta);
                    steps.push(Step { op: StepOp::CoeffSum { c: coeff_sums.len() as u32 - 1 }, out, a: ins[0], b: NONE, inplace: false, var, meta: p.vars[var as usize].meta, a_meta: vm(0), b_meta: None });
                    continue;
                }
                if m.outs.len() != 1 {
                    return Err(berr(block, format!("{} op with {} outputs ({:?}): PCMM/CCMM finishes replay through the matmul services only", m.kind.name(), m.outs.len(), m.origin)));
                }
                let var = m.outs[0];
                let out = slot_of(var)?;
                let sop = match &m.kind {
                    Kind::Add => StepOp::Add,
                    Kind::Sub => StepOp::Sub,
                    Kind::AddPt(i) => StepOp::AddPt(*i),
                    Kind::MultPt(i) => StepOp::MultPt(*i),
                    Kind::AddConst(c) => StepOp::AddConst(scalar(m, c, &mut no_scalar)),
                    Kind::MultConst(c) => StepOp::MultConst(scalar(m, c, &mut no_scalar)),
                    Kind::Mult => StepOp::Mult,
                    Kind::Square => StepOp::Square,
                    Kind::Negate => StepOp::Negate,
                    Kind::Rotate(k) => StepOp::Rotate { k: *k, g: list_group[list_idx(&vec![*k]) as usize] },
                    Kind::RotateSum(s) => StepOp::RotateSum { l: list_idx(s), g: list_group[list_idx(s) as usize] },
                    Kind::Conjugate => StepOp::Conjugate { g: conj },
                    Kind::MultI => StepOp::MultI,
                    Kind::Rescale => StepOp::Rescale,
                    Kind::DropTo(l) => StepOp::DropTo(*l),
                    Kind::Bootstrap(site) => {
                        let s = p.bts_sites.get(*site as usize).ok_or_else(|| berr(block, format!("bootstrap site {site} out of range")))?;
                        let kind = RouteKind::of(&s.route, lay.n);
                        if routes.get(kind).is_none() {
                            no_route.push(format!("{kind:?} (site {} on {})", site, s.var));
                        }
                        StepOp::Bootstrap { site: *site, route: s.route, kind }
                    }
                    Kind::Finish { .. } => return Err(berr(block, "finish reached the single-output lowering")),
                    Kind::CoeffSum { .. } => return Err(berr(block, "coeff_sum reached the single-output lowering")),
                    Kind::Batched(_) => return Err(berr(block, "nested batch")),
                };
                let inplace = m.inplace;
                if inplace && out != a {
                    alias.push(format!("{:?}: in place but out slot {out} != a slot {a}", m.origin));
                }
                if !inplace && a != NONE && out == a {
                    alias.push(format!("{:?}: out of place but out slot {out} == a slot", m.origin));
                }
                if b != NONE && out == b && !(inplace && a == b) {
                    alias.push(format!("{:?}: out slot {out} == b slot", m.origin));
                }
                let vm = |k: usize| m.ins.get(k).map(|&v| p.vars[v as usize].meta);
                steps.push(Step { op: sop, out, a, b, inplace, var, meta: p.vars[var as usize].meta, a_meta: vm(0), b_meta: vm(1) });
            }
        }
        first_step.push(steps.len());
        // rotation grids: their steps, input slots, a key group of their steps (a grid with a step that is the
        // identity mod N/2 issues step by step)
        let mut units: Vec<Unit> = vec![];
        let mut unit_of = vec![NONE; steps.len()];
        for gs in p.rotation_grids() {
            if gs.steps.iter().any(|s| s.rem_euclid(half) == 0) {
                continue;
            }
            let mut refs: Vec<&dyn KsKeyRef> = vec![];
            for &s in &gs.steps {
                match env.keys.rot(s) {
                    Ok(k) => refs.push(k),
                    Err(_) => missing.push(format!("rot {s}")),
                }
            }
            if refs.len() != gs.steps.len() {
                continue;
            }
            let (lo, hi) = (first_step[gs.ops.0], first_step[gs.ops.1]);
            let mut at = Vec::with_capacity(hi - lo);
            for m in p.ops[gs.ops.0..gs.ops.1].iter().flat_map(|o| o.members()) {
                let Kind::Rotate(k) = m.kind else { return Err(berr(block, "rotation grid over a non-rotation op")) };
                let c = gs.ins.iter().position(|&x| x == m.ins[0]).ok_or_else(|| berr(block, "rotation grid input missing"))?;
                let s = gs.steps.iter().position(|&x| x == k).ok_or_else(|| berr(block, "rotation grid step missing"))?;
                at.push((c as u8, s as u16));
            }
            unit_of[lo..hi].iter_mut().for_each(|x| *x = units.len() as u32);
            let ins = gs.ins.iter().map(|&x| slot_of(x)).collect::<Result<Vec<_>>>()?;
            units.push(Unit { lo: lo as u32, hi: hi as u32, ins, kind: UnitKind::Grid { steps: gs.steps.clone(), g: groups.len() as u32, at, fresh: true, held: false }, chain: NONE, off: false });
            groups.push(KeyGroup::new(env.arena, env.stream, lay.n, &refs, env.stage)?);
        }
        // rotation chains: one key group of the chain's steps, one unit per link, the first one fresh
        for (ci, cs) in p.rotation_chains().iter().enumerate() {
            if cs.steps.iter().any(|s| s.rem_euclid(half) == 0) {
                continue;
            }
            let refs: Vec<&dyn KsKeyRef> = cs.steps.iter().filter_map(|&s| env.keys.rot(s).ok().map(|k| k as &dyn KsKeyRef)).collect();
            if refs.len() != cs.steps.len() {
                continue;
            }
            let ins = cs.ins.iter().map(|&x| slot_of(x)).collect::<Result<Vec<_>>>()?;
            let g = groups.len() as u32;
            for (j, &oi) in cs.ops.iter().enumerate() {
                let (lo, hi) = (first_step[oi], first_step[oi + 1]);
                let mut at = Vec::with_capacity(hi - lo);
                for m in p.ops[oi].members() {
                    let Kind::Rotate(k) = m.kind else { return Err(berr(block, "rotation chain over a non-rotation op")) };
                    let c = cs.ins.iter().position(|&x| x == m.ins[0]).ok_or_else(|| berr(block, "rotation chain input missing"))?;
                    let s = cs.steps.iter().position(|&x| x == k).ok_or_else(|| berr(block, "rotation chain step missing"))?;
                    at.push((c as u8, s as u16));
                }
                unit_of[lo..hi].iter_mut().for_each(|x| *x = units.len() as u32);
                units.push(Unit { lo: lo as u32, hi: hi as u32, ins: ins.clone(), kind: UnitKind::Grid { steps: cs.steps.clone(), g, at, fresh: j == 0, held: cs.held }, chain: ci as u32, off: false });
            }
            groups.push(KeyGroup::new(env.arena, env.stream, lay.n, &refs, env.stage)?);
        }
        // K-batches of rotate_sums of one step list (inputs of one level, outputs in slots of their own)
        for (oi, op) in p.ops.iter().enumerate() {
            let m = op.members();
            let Kind::RotateSum(list) = &m[0].kind else { continue };
            let (lo, hi) = (first_step[oi], first_step[oi + 1]);
            let level = |v: u32| p.vars[v as usize].meta.level;
            if m.len() < 2 || m.len() > GRID_INPUTS || !m.iter().all(|x| matches!(&x.kind, Kind::RotateSum(l) if l == list) && x.pre.is_empty() && level(x.ins[0]) == level(m[0].ins[0])) {
                continue;
            }
            let StepOp::RotateSum { l, g } = steps[lo].op else { continue };
            let ins: Vec<u32> = steps[lo..hi].iter().map(|st| st.a).collect();
            let outs: Vec<u32> = steps[lo..hi].iter().map(|st| st.out).collect();
            if g == NONE || outs.iter().enumerate().any(|(c, o)| ins.contains(o) || outs[..c].contains(o)) {
                continue;
            }
            unit_of[lo..hi].iter_mut().for_each(|x| *x = units.len() as u32);
            units.push(Unit { lo: lo as u32, hi: hi as u32, ins, kind: UnitKind::RotateSum { l, g }, chain: NONE, off: false });
        }
        // plaintext products consumed by the next add (the product's input and the add's other operand stay live)
        for oi in p.mult_pt_adds() {
            let lo = first_step[oi];
            let (StepOp::MultPt(pt), add) = (&steps[lo].op, &steps[lo + 1]) else { continue };
            if !matches!(add.op, StepOp::Add | StepOp::Sub) || unit_of[lo] != NONE || unit_of[lo + 1] != NONE {
                continue;
            }
            let sub = add.op == StepOp::Sub;
            let product_first = p.ops[oi + 1].ins[0] == p.ops[oi].outs[0];
            let y = if product_first { add.b } else { add.a };
            if y == NONE {
                continue;
            }
            unit_of[lo..lo + 2].iter_mut().for_each(|x| *x = units.len() as u32);
            units.push(Unit { lo: lo as u32, hi: lo as u32 + 2, ins: vec![steps[lo].a, y], kind: UnitKind::MultPtAdd { pt: *pt, sub, product_first }, chain: NONE, off: false });
        }
        // K-batches of products and of refreshes (siblings of one recipe / route; outputs in slots of their own, the
        // refresh route built for K ciphertexts)
        for (oi, op) in p.ops.iter().enumerate() {
            let m = op.members();
            let (lo, hi) = (first_step[oi], first_step[oi + 1]);
            if m.len() < 2 || m.len() > GRID_INPUTS || hi - lo != m.len() || steps[lo..hi].iter().any(|st| st.inplace) {
                continue;
            }
            let ins: Vec<u32> = steps[lo..hi].iter().map(|st| st.a).collect();
            let bs: Vec<u32> = steps[lo..hi].iter().map(|st| st.b).collect();
            let outs: Vec<u32> = steps[lo..hi].iter().map(|st| st.out).collect();
            if outs.iter().enumerate().any(|(c, o)| ins.contains(o) || bs.contains(o) || outs[..c].contains(o)) {
                continue;
            }
            let kind = match (&m[0].kind, &steps[lo].op) {
                (Kind::Mult, StepOp::Mult) if steps[lo..hi].iter().all(|st| st.op == StepOp::Mult) && relin != NONE => UnitKind::Mult { b: bs },
                (Kind::Square, StepOp::Square) if steps[lo..hi].iter().all(|st| st.op == StepOp::Square) && relin != NONE => UnitKind::Mult { b: vec![] },
                (Kind::Bootstrap(_), StepOp::Bootstrap { route, kind, .. }) => {
                    let same = steps[lo..hi].iter().all(|st| matches!(&st.op, StepOp::Bootstrap { route: r, .. } if r == route));
                    if !same || routes.get(*kind).is_none_or(|r| r.mem().1.k_max < m.len()) {
                        continue;
                    }
                    UnitKind::Refresh { route: *route, kind: *kind }
                }
                _ => continue,
            };
            unit_of[lo..hi].iter_mut().for_each(|x| *x = units.len() as u32);
            units.push(Unit { lo: lo as u32, hi: hi as u32, ins, kind, chain: NONE, off: false });
        }
        if !missing.is_empty() {
            missing.dedup();
            return Err(berr(block, format!("{} key(s) missing from the key ring: {}", missing.len(), list(&missing))));
        }
        if !no_scalar.is_empty() {
            return Err(berr(block, format!("{} inexact v1 scalar(s) with no exact value in the driver's table: {}", no_scalar.len(), list(&no_scalar))));
        }
        if !no_route.is_empty() {
            return Err(berr(block, format!("route(s) not held by the runtime: {}", list(&no_route))));
        }
        if !alias.is_empty() {
            return Err(berr(block, format!("slot aliasing the replay cannot run: {}", list(&alias))));
        }
        let (slots, tmp, gather) = if carve {
            let slots = (0..n_slots).map(|_| Ok(Ct::new(CtMeta::new(0, 1), CtBuf::new(env.arena, "replay.slot", lay, 1)?))).collect::<Result<Vec<_>>>()?;
            let tmp = Some(Ct::new(CtMeta::new(0, 1), CtBuf::new(env.arena, "replay.slot", lay, 1)?));
            (slots, tmp, Some(Gather::new(env.arena, lay)?))
        } else {
            (vec![], None, None)
        };
        // a block on its own slots carves no held-chain buffers: those chains issue step by step
        let io = |vs: &[u32]| -> Result<Vec<(String, u32, CtMeta)>> { vs.iter().map(|&v| Ok((p.vars[v as usize].name.clone(), slot_of(v)?, p.vars[v as usize].meta))).collect() };
        Ok(Bound {
            label: block.to_string(),
            slots,
            tmp,
            gather,
            held: None,
            n_slots,
            dynamic: Default::default(),
            dyn_group: NONE,
            dyn_steps: vec![],
            dyn_metas: BTreeMap::new(),
            steps,
            groups,
            relin,
            step_lists: lists,
            units,
            unit_of,
            pts,
            finishes,
            coeff_sums,
            streams,
            stream_base: std::ptr::null(),
            inputs: io(&p.inputs)?,
            outputs: io(&p.outputs)?,
            names: p.vars.iter().map(|v| v.name.clone()).collect(),
            ring_n: lay.n,
            graphs: vec![],
        })
    }

    /// Resolve every plaintext key to its resident device rows (missing = one error listing them). Returns whether
    /// any address changed since the last call (a captured graph baked the old ones: replay re-captures).
    pub fn resolve_pts(&mut self, store: &PtStore<'_, '_, W>) -> Result<bool> {
        let mut missing: Vec<String> = vec![];
        let mut moved = false;
        for bp in self.pts.iter_mut().filter(|p| !p.stream) {
            match store.view(&bp.key) {
                Ok(pt) => {
                    moved |= pt.rows != bp.rows;
                    bp.rows = pt.rows;
                    bp.meta = Some(pt.meta);
                }
                Err(_) => missing.push(format!("{}@{}", bp.key.name, bp.key.level)),
            }
        }
        if !missing.is_empty() {
            missing.sort();
            missing.dedup();
            return Err(berr(&self.label, format!("{} plaintext(s) not resident in the PtStore: {}", missing.len(), list(&missing))));
        }
        Ok(moved)
    }

    /// The distinct plaintext keys of the block (staging list for the weight ring).
    pub fn pt_keys(&self) -> Vec<PtKey> {
        let mut v: Vec<PtKey> = vec![];
        for p in self.pts.iter().filter(|p| !p.stream) {
            if !v.contains(&p.key) {
                v.push(p.key.clone());
            }
        }
        v
    }

    /// The ciphertext a block input is written into (the driver encrypts / uploads here before `run`); its
    /// metadata is set to the compiled input metadata.
    pub fn input_mut(&mut self, i: usize) -> Result<&mut Ct<'a, W>> {
        let (_, s, m) = self.inputs.get(i).cloned().ok_or_else(|| berr(&self.label, format!("no input {i}")))?;
        let label = self.label.clone();
        let c = self.slots.get_mut(s as usize).ok_or_else(|| berr(&label, "slots not present (a shared block outside with_slab)"))?;
        c.meta = m;
        Ok(c)
    }

    /// Output `i` (0 = the exit var) after a run.
    pub fn output(&self, i: usize) -> Result<&Ct<'a, W>> {
        let (_, s, _) = self.outputs.get(i).ok_or_else(|| berr(&self.label, format!("no output {i}")))?;
        self.slots.get(*s as usize).ok_or_else(|| berr(&self.label, "slots not present (a shared block outside with_slab)"))
    }

    /// Re-route refresh site `site` (a driver decision the graph does not carry, e.g. a deliberate refresh of a
    /// real payload: perseus `bootstrap_real`). Same route family only: the runtime's route table was checked at
    /// bind. Drops nothing: call before the first run (a captured graph keeps the old route).
    pub fn set_site_route(&mut self, site: u32, route: Route) -> Result<usize> {
        let mut n = 0;
        for st in self.steps.iter_mut() {
            if let StepOp::Bootstrap { site: s, route: r, kind } = &mut st.op {
                if *s == site {
                    let k = RouteKind::of(&route, self.ring_n);
                    if k != *kind {
                        return Err(berr(&self.label, format!("site {site}: {route:?} is route family {k:?}, bound as {kind:?}")));
                    }
                    *r = route;
                    n += 1;
                }
            }
        }
        if !self.graphs.is_empty() {
            return Err(berr(&self.label, "set_site_route after a capture"));
        }
        Ok(n)
    }

    /// Captured graphs held.
    pub fn n_graphs(&self) -> usize {
        self.graphs.len()
    }

    /// Run `f` with `slab`'s slots as this block's slots (a block bound by `new_shared`); the slab gets them back
    /// whatever `f` returns.
    pub fn with_slab<T>(&mut self, slab: &mut Slab<'a, W>, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        if !self.slots.is_empty() || self.tmp.is_some() || self.gather.is_some() {
            return Err(berr(&self.label, "with_slab on a block with its own slots"));
        }
        if slab.slots.len() < self.n_slots || slab.tmp.is_none() || slab.gather.is_none() {
            return Err(berr(&self.label, format!("slab of {} slots, the block needs {}", slab.slots.len(), self.n_slots)));
        }
        std::mem::swap(&mut self.slots, &mut slab.slots);
        std::mem::swap(&mut self.tmp, &mut slab.tmp);
        std::mem::swap(&mut self.gather, &mut slab.gather);
        std::mem::swap(&mut self.held, &mut slab.held);
        let r = f(self);
        std::mem::swap(&mut self.slots, &mut slab.slots);
        std::mem::swap(&mut self.tmp, &mut slab.tmp);
        std::mem::swap(&mut self.gather, &mut slab.gather);
        std::mem::swap(&mut self.held, &mut slab.held);
        r
    }

    /// The steps producing graph name `name` (a var `name` or its `~pre` / `~raw` states).
    fn steps_of(&self, name: &str) -> Vec<usize> {
        (0..self.steps.len()).filter(|&i| self.names[self.steps[i].var as usize].split('~').next() == Some(name)).collect()
    }

    fn make_dynamic(&mut self, i: usize) -> Result<()> {
        if !self.dynamic.contains(&(i as u32)) {
            if !self.graphs.is_empty() {
                return Err(berr(&self.label, format!("step {i} made dynamic after a graph capture (declare dynamic steps before the first run)")));
            }
            self.dynamic.insert(i as u32);
            // a unit with a dynamic step issues step by step, and so does the rest of its rotation chain
            if let Some(&u) = self.unit_of.get(i).filter(|&&u| u != NONE) {
                let chain = self.units[u as usize].chain;
                for (j, x) in self.units.iter_mut().enumerate() {
                    if j == u as usize || chain != NONE && x.chain == chain {
                        x.off = true;
                    }
                }
            }
        }
        Ok(())
    }

    /// The unit step `i` heads (an enabled unit starting at step `i`; a held chain's link only with the held
    /// buffers present).
    pub fn unit_at(&self, i: usize) -> Option<&Unit> {
        let u = self.units.get(*self.unit_of.get(i)? as usize)?;
        let held = matches!(u.kind, UnitKind::Grid { held: true, .. });
        (u.lo as usize == i && !u.off && (!held || self.held.is_some())).then_some(u)
    }

    /// Set the scalar of the constant op producing `name` (a per-position constant); the step becomes dynamic.
    pub fn set_scalar(&mut self, name: &str, value: f64) -> Result<()> {
        let idx: Vec<usize> = self.steps_of(name).into_iter().filter(|&i| matches!(self.steps[i].op, StepOp::AddConst(_) | StepOp::MultConst(_))).collect();
        if idx.len() != 1 {
            return Err(berr(&self.label, format!("set_scalar {name}: {} constant steps produce it", idx.len())));
        }
        let i = idx[0];
        self.make_dynamic(i)?;
        match &mut self.steps[i].op {
            StepOp::AddConst(c) | StepOp::MultConst(c) => *c = value,
            _ => {}
        }
        Ok(())
    }

    /// Set the rotation of the rotate op producing `name` to `k` (0 mod N/2 = a copy; else one of the steps the
    /// block was bound with in `dyn_rot`); the step becomes dynamic.
    pub fn set_rotation(&mut self, name: &str, k: i64) -> Result<()> {
        let idx: Vec<usize> = self.steps_of(name).into_iter().filter(|&i| matches!(self.steps[i].op, StepOp::Rotate { .. })).collect();
        if idx.len() != 1 {
            return Err(berr(&self.label, format!("set_rotation {name}: {} rotate steps produce it", idx.len())));
        }
        let half = (self.ring_n / 2) as i64;
        let g = if k.rem_euclid(half) == 0 {
            NONE
        } else if self.dyn_steps.contains(&k) {
            self.dyn_group
        } else {
            return Err(berr(&self.label, format!("set_rotation {name}: step {k} is not among the dynamic rotation keys {:?}", self.dyn_steps)));
        };
        let i = idx[0];
        self.make_dynamic(i)?;
        self.steps[i].op = StepOp::Rotate { k, g };
        Ok(())
    }

    /// Declare the step producing `name` dynamic without changing it (its arguments will change later).
    pub fn mark_dynamic(&mut self, name: &str) -> Result<()> {
        let idx = self.steps_of(name);
        if idx.is_empty() {
            return Err(berr(&self.label, format!("mark_dynamic {name}: no step produces it")));
        }
        for i in idx {
            self.make_dynamic(i)?;
        }
        Ok(())
    }
}

impl Program {
    /// `Bound::new`.
    pub fn bind<'a, W: DevWord, M: DevStrategy<W>>(&self, env: &BindEnv<'_, 'a>, routes: &Routes<'_, 'a, W, M>, store: &PtStore<'_, '_, W>) -> Result<Bound<'a, W>> {
        let mut b = Bound::new(self, env, routes)?;
        b.resolve_pts(store)?;
        Ok(b)
    }
}
