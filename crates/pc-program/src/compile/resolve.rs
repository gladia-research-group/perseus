//! Name resolution: graph nodes -> `RNode`s with typed op kinds. Cipher operands stay graph names here (the
//! lowering maps them to program vars, which plan application splits into pre/raw/final states); literals become
//! op fields; plaintext operands become `PtDraft`s; captured deliberate refreshes are classified into code-asked
//! refreshes and external (PCMM/CCMM) finishes, sibling finishes of one input merged into one op.
use crate::error::{Error, Result};
use crate::graph::BlockGraph;
use crate::program::{PtForm, Scalar};
use pc_shadow::graph::{is_literal_input, is_plaintext_name, Node};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Debug, PartialEq)]
pub enum RKind {
    Add,
    Sub,
    AddPt(usize),
    MultPt(usize),
    AddConst(Scalar),
    MultConst(Scalar),
    Mult,
    Square,
    Negate,
    Rotate(i64),
    RotateSum(Vec<i64>),
    Conjugate,
    MultI,
    /// A recorded copy: the output aliases the input.
    Clone,
    /// `bootstrap_hint(thr, acct)`.
    Hint { thr: u32, acct: bool },
    /// `level_hint(lvl_cap)`: a recorded no-op the planner reads as a cap.
    LevelHint { cap: u32 },
    /// `drop_to_level(t)`; v1 records no target: the captured output level is used.
    LevelReduce { target: u32 },
    /// The code's own refresh (`bootstrap()`): lands at the captured level; `real` = `bootstrap_real()` (the
    /// real-payload route when the runtime has it; v2 `args.route.real`).
    Deliberate { level: u32, deg: u8, real: bool },
    /// External finish (PCMM/CCMM): `outs` (several sibling nodes merged) land at the captured level; `lands` = every
    /// output's own landing (siblings may differ: per-output raise drops of a PCMM, e.g. gate_up's up half).
    Finish { level: u32, deg: u8, desc: Option<String>, lands: Vec<(u32, u8)>, descs: Vec<Option<String>> },
    FoldBootstrap { s: u32, n_live: u32, prescale: f64 },
}

#[derive(Clone, Debug)]
pub struct RNode {
    /// Planner-order index (the first one for a merged finish).
    pub idx: u32,
    pub merged: Vec<u32>,
    pub kind: RKind,
    pub ins: Vec<String>,
    pub outs: Vec<String>,
    pub step: String,
}

#[derive(Clone, Debug)]
pub struct PtDraft {
    pub name: String,
    pub form: PtForm,
    /// Fixed level/deg (Slot form); AtUse/Cached are resolved against the ciphertext at lowering.
    pub fixed: Option<(u32, u8)>,
    pub captured_level: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct Resolved {
    /// Pack tags of the graph's outputs (v2 `tag`): a planned refresh routes by its value's period.
    pub tags: HashMap<String, pc_shadow::tag::Tag>,
    pub nodes: Vec<RNode>,
    pub pts: Vec<PtDraft>,
    /// Block inputs in first-use order.
    pub inputs: Vec<String>,
    pub elided_by_merge: usize,
    pub inexact_scalars: usize,
}

fn gerr(block: &str, n: &Node, msg: impl std::fmt::Display) -> Error {
    Error::Graph { block: block.into(), msg: format!("node {} {} {:?} -> {}: {msg}", n.idx, n.op, n.inputs, n.output) }
}

fn lit<'a>(n: &'a Node, prefix: &str) -> Option<&'a str> {
    n.inputs.iter().find_map(|x| x.strip_prefix(prefix).and_then(|r| r.strip_suffix(')')))
}

fn level_u32(x: f64) -> Option<u32> {
    (x >= 0.0 && x.fract() == 0.0 && x < u32::MAX as f64).then_some(x as u32)
}

/// The PCMM calibration key of a linear step: `blk<b>.<site>` -> `b<b>.<site'>` (out_proj -> out).
fn calib_key(step: &str) -> Option<String> {
    let (blk, site) = step.split_once('.')?;
    let b = blk.strip_prefix("blk")?;
    let site = match site {
        "out_proj" => "out",
        s => s,
    };
    Some(format!("b{b}.{site}"))
}

fn pt_form(name: &str, args: Option<&Value>) -> (PtForm, Option<(u32, u8)>) {
    if let Some(pt) = args.and_then(|a| a.get("pt")) {
        let kind = pt.get("kind").and_then(Value::as_str).unwrap_or("");
        return match kind {
            // a v2 slot plaintext is registered at one level but the runtime re-encodes it like the ciphertext
            // it meets (`FheOps._encode_like`): the v1 graphs' AtUse, at the consumer's level
            "slot" => (PtForm::AtUse, None),
            "cached" => (PtForm::Cached, None),
            // a streamed weight row: `pt.name` = `<linear>#<row>`
            "stream" => (PtForm::CoeffStream, None),
            _ => (PtForm::AtUse, None),
        };
    }
    // impl.<key>@<lvl>d<deg>: a slot plaintext encoded once at that level
    if let Some(rest) = name.strip_prefix("impl.") {
        if let Some((_, ld)) = rest.rsplit_once('@') {
            let (l, d) = ld.split_once('d').unwrap_or((ld, "1"));
            if let (Ok(l), Ok(d)) = (l.parse::<u32>(), d.parse::<u8>()) {
                return (PtForm::Slot, Some((l, d)));
            }
        }
    }
    if name.starts_with("mask.") {
        return (PtForm::Cached, None);
    }
    (PtForm::AtUse, None)
}

/// A v2 graph tag (`{slots, period, kind, offset, stride, count, width, field}`).
fn tag_of_json(v: &Value) -> Option<pc_shadow::tag::Tag> {
    let g = |k: &str| v.get(k).and_then(Value::as_i64);
    let sp = pc_shadow::tag::Support::new(g("kind")? as i32, g("offset")?, g("stride")?, g("count")?, g("width")?);
    let mut t = pc_shadow::tag::Tag::new(g("slots")?, g("period")?, sp);
    t.field = g("field").unwrap_or(0) as i32;
    Some(t)
}

pub fn resolve(bg: &BlockGraph, pcmm_calib: &BTreeMap<String, f64>) -> Result<Resolved> {
    let block = bg.name.as_str();
    let g = &bg.graph;
    let mut pts: Vec<PtDraft> = vec![];
    let mut nodes: Vec<RNode> = vec![];
    let mut inexact = 0usize;
    // deliberate nodes sharing an input set within one step are one external op's outputs (the PCMM records one
    // deliberate_bootstrap per output, all of the same input)
    // finishes merge into one call by (cipher inputs, step, the external descriptor's linear): two linears may read the
    // same inputs in one step (sana.fold's main part and a one-pair group's folded part)
    let ext_linear = |out: &str| -> String {
        bg.args(out).and_then(|a| a.get("external")).and_then(|e| e.get("linear")).and_then(Value::as_str).unwrap_or("").to_string()
    };
    let mut delib_group: HashMap<(Vec<String>, String, String), usize> = HashMap::new();
    let mut delib_count: HashMap<(Vec<String>, String, String), usize> = HashMap::new();
    for n in &g.nodes {
        if n.is_deliberate_bts() {
            *delib_count.entry((n.cipher_inputs.clone(), n.step.clone(), ext_linear(&n.output))).or_default() += 1;
        }
    }
    let mut elided_by_merge = 0;
    for n in &g.nodes {
        let args = bg.args(&n.output);
        let cins = n.cipher_inputs.clone();
        let mut pt_idx = None;
        for (x, lv) in n.inputs.iter().zip(&n.input_levels) {
            if is_plaintext_name(x) {
                let (form, fixed) = pt_form(x, args);
                let name = match form {
                    PtForm::CoeffStream => args
                        .and_then(|a| a.get("pt"))
                        .and_then(|p| p.get("name"))
                        .and_then(Value::as_str)
                        .filter(|s| s.contains('#'))
                        .ok_or_else(|| gerr(block, n, "a stream plaintext without args.pt.name = <linear>#<row>"))?
                        .to_string(),
                    _ => x.clone(),
                };
                pts.push(PtDraft { name, form, fixed, captured_level: lv.and_then(level_u32) });
                pt_idx = Some(pts.len() - 1);
            }
        }
        let scalar = || -> Option<Scalar> {
            if let Some(v) = args.and_then(|a| a.get("scalar")).and_then(Value::as_f64) {
                return Some(Scalar { value: v, exact: true });
            }
            lit(n, "const(").and_then(|s| s.trim().parse::<f64>().ok()).map(|value| Scalar { value, exact: false })
        };
        let need = |k: usize| -> Result<()> {
            if cins.len() != k {
                return Err(gerr(block, n, format!("{} cipher input(s), the op takes {k}", cins.len())));
            }
            Ok(())
        };
        let op = n.op.as_str();
        let kind = match op {
            "add" | "add_inplace" | "sub" | "sub_inplace" | "sub_ct" | "sub_inplace_ct" => {
                let sub = op.starts_with("sub");
                if cins.len() == 2 {
                    if sub {
                        RKind::Sub
                    } else {
                        RKind::Add
                    }
                } else if let Some(p) = pt_idx {
                    need(1)?;
                    if sub {
                        return Err(gerr(block, n, "ct - pt is not a recorded primitive"));
                    }
                    RKind::AddPt(p)
                } else {
                    need(1)?;
                    let mut c = scalar().ok_or_else(|| gerr(block, n, "no operand besides the ciphertext"))?;
                    if sub {
                        c.value = -c.value;
                    }
                    RKind::AddConst(c)
                }
            }
            "mult" | "mult_inplace" => {
                if cins.len() == 2 {
                    RKind::Mult
                } else if let Some(p) = pt_idx {
                    need(1)?;
                    RKind::MultPt(p)
                } else {
                    need(1)?;
                    RKind::MultConst(scalar().ok_or_else(|| gerr(block, n, "no operand besides the ciphertext"))?)
                }
            }
            "square" | "square_inplace" => {
                need(1)?;
                RKind::Square
            }
            "negate" | "negate_inplace" => {
                need(1)?;
                RKind::Negate
            }
            "rotate" => {
                need(1)?;
                let k = args
                    .and_then(|a| a.get("steps").or_else(|| a.get("k")))
                    .and_then(|v| v.as_i64().or_else(|| v.as_array().and_then(|a| a.first()).and_then(Value::as_i64)))
                    .or_else(|| lit(n, "rot(").and_then(|s| s.trim().parse().ok()))
                    .ok_or_else(|| gerr(block, n, "no rot(k)"))?;
                RKind::Rotate(k)
            }
            "rotate_sum" => {
                need(1)?;
                let steps: Vec<i64> = match args.and_then(|a| a.get("steps")).and_then(Value::as_array) {
                    Some(a) => a.iter().filter_map(Value::as_i64).collect(),
                    None => lit(n, "rot(")
                        .ok_or_else(|| gerr(block, n, "no rot(steps)"))?
                        .split(',')
                        .map(|s| s.trim().parse::<i64>().map_err(|_| gerr(block, n, format!("bad step {s:?}"))))
                        .collect::<Result<_>>()?,
                };
                if steps.is_empty() {
                    return Err(gerr(block, n, "rotate_sum with no steps"));
                }
                RKind::RotateSum(steps)
            }
            "conjugate" => {
                need(1)?;
                RKind::Conjugate
            }
            "mult_i" => {
                need(1)?;
                RKind::MultI
            }
            "clone" => {
                need(1)?;
                RKind::Clone
            }
            "hint" => {
                need(1)?;
                let h = args.and_then(|a| a.get("hint"));
                let thr = h
                    .and_then(|h| h.get("thr"))
                    .and_then(Value::as_f64)
                    .or(n.hint_level)
                    .and_then(level_u32)
                    .ok_or_else(|| gerr(block, n, "no hint_lev(t)"))?;
                let acct = h.and_then(|h| h.get("acct")).and_then(Value::as_bool).unwrap_or(false);
                RKind::Hint { thr, acct }
            }
            "level_hint" => {
                need(1)?;
                let cap = args
                    .and_then(|a| a.get("cap"))
                    .and_then(Value::as_f64)
                    .or(n.lvl_cap)
                    .and_then(level_u32)
                    .ok_or_else(|| gerr(block, n, "no lvl_cap(l)"))?;
                RKind::LevelHint { cap }
            }
            "level_reduce" => {
                need(1)?;
                let t = args
                    .and_then(|a| a.get("target"))
                    .and_then(Value::as_f64)
                    .or(n.output_level)
                    .and_then(level_u32)
                    .ok_or_else(|| gerr(block, n, "no drop target"))?;
                RKind::LevelReduce { target: t }
            }
            "fold_bootstrap" => {
                need(1)?;
                let a = args.ok_or_else(|| gerr(block, n, "fold_bootstrap needs v2 args {s, n_live, prescale}"))?;
                let g = |k: &str| a.get(k).and_then(Value::as_f64);
                RKind::FoldBootstrap {
                    s: g("s").ok_or_else(|| gerr(block, n, "args.s"))? as u32,
                    n_live: g("n_live").ok_or_else(|| gerr(block, n, "args.n_live"))? as u32,
                    prescale: g("prescale").unwrap_or(1.0),
                }
            }
            "deliberate_bootstrap" => {
                if cins.is_empty() {
                    return Err(gerr(block, n, "deliberate_bootstrap without a cipher input"));
                }
                let level = n.output_level.and_then(level_u32).ok_or_else(|| gerr(block, n, "no captured landing"))?;
                let deg = n.output_deg.unwrap_or(2) as u8;
                let ext = args.and_then(|a| a.get("external"));
                // v2 `args.external` dict (pcmm.py / ccmm.py descriptor: linear, out, n_in, n_out, t, shift, raise_drop)
                // as JSON: what the replay's finish service rebuilds the call from; else the calibration site name
                let desc = ext
                    .filter(|e| e.is_object())
                    .map(|e| e.to_string())
                    .or_else(|| args.and_then(|a| a.get("desc")).and_then(Value::as_str).map(str::to_string))
                    .or_else(|| calib_key(&n.step).filter(|k| pcmm_calib.contains_key(k)));
                let key = (cins.clone(), n.step.clone(), ext_linear(&n.output));
                let siblings = delib_count.get(&key).copied().unwrap_or(1);
                let finish = ext.is_some() || desc.is_some() || cins.len() > 1 || siblings > 1;
                if finish {
                    if let Some(&gi) = delib_group.get(&key) {
                        let rn: &mut RNode = &mut nodes[gi];
                        if let RKind::Finish { lands, descs, .. } = &mut rn.kind {
                            lands.push((level, deg));
                            descs.push(desc.clone());
                        }
                        rn.outs.push(n.output.clone());
                        rn.merged.push(n.idx as u32);
                        elided_by_merge += 1;
                        continue;
                    }
                    delib_group.insert(key, nodes.len());
                    RKind::Finish { level, deg, desc: desc.clone(), lands: vec![(level, deg)], descs: vec![desc] }
                } else {
                    let real = args.and_then(|a| a.get("route")).and_then(|r| r.get("real")).and_then(Value::as_bool).unwrap_or(false);
                    RKind::Deliberate { level, deg, real }
                }
            }
            "auto_bootstrap" => return Err(gerr(block, n, "auto_bootstrap survived erase_reactive_bootstraps")),
            other => return Err(gerr(block, n, format!("unknown op kind {other:?}"))),
        };
        if matches!(kind, RKind::AddConst(Scalar { exact: false, .. }) | RKind::MultConst(Scalar { exact: false, .. })) {
            inexact += 1;
        }
        if n.output.is_empty() {
            return Err(gerr(block, n, "node has no output var"));
        }
        // every non-literal, non-plaintext input is a cipher operand; a literal the op does not read is refused
        for x in &n.inputs {
            if is_literal_input(x) && !(x.starts_with("rot(") || x.starts_with("const(") || x.starts_with("hint_lev(") || x.starts_with("lvl_cap(")) {
                return Err(gerr(block, n, format!("literal {x:?} (ct_null / pt_null) has no lowering")));
            }
        }
        nodes.push(RNode { idx: n.idx as u32, merged: vec![], kind, ins: cins, outs: vec![n.output.clone()], step: n.step.clone() });
    }
    // block inputs: consumed and never produced, in first-use order (the driver's input order)
    let produced: std::collections::HashSet<&str> = nodes.iter().flat_map(|n| n.outs.iter().map(String::as_str)).collect();
    let mut inputs: Vec<String> = vec![];
    for n in &nodes {
        for v in &n.ins {
            if !produced.contains(v.as_str()) && !inputs.contains(v) {
                inputs.push(v.clone());
            }
        }
    }
    let mut tags = HashMap::new();
    for (name, ex) in &bg.extra {
        if let Some(t) = ex.tag.as_ref().and_then(tag_of_json) {
            tags.insert(name.clone(), t);
        }
    }
    Ok(Resolved { tags, nodes, pts, inputs, elided_by_merge, inexact_scalars: inexact })
}
