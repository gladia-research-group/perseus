//! graph + plan -> `Program`. Passes, in order: `resolve` (names, literals, op kinds), `apply_plan` (the lowering:
//! plan decisions as ops, the Shadow alongside), `batch` (K sibling ops per launch), `alloc` (arena slots),
//! `check::verify` (the Shadow re-run over the finished program: pins, ceilings, recorded preludes).
//!
//! The runtime truth the check runs on is `runtime_shadow`: the pc-shadow C++ rules (no quirks) on the real chain
//! (`chain.preset`, n32 by default), with FIDESlib's realized route landings (dense StC-first 28 / shipped order 34 past the StC-first entry,
//! 1024, 512 and 256 at 24, 1 at 16) and the per-route raise drops of the runtime line (`FIDESLIB_BTS_RAISE_DROP`): the
//! planner's own landing table (`PLAN_SPARSE_BTS_OUT`, PLAN_STEP_LANDING floors) only sets the pins, which the check
//! reads one-sided.
pub mod alloc;
pub mod apply_plan;
pub mod batch;
pub mod check;
pub mod resolve;
pub mod stream;

use crate::error::{Error, Result};
use crate::graph::BlockGraph;
use crate::plan::{BlockPlan, PlanDir};
use crate::program::{Pairing, Program, VarId};
use pc_config::{Config, PlanConfig};
use pc_shadow::graph::{parse_lvl_suffix, LoadOpts};
use pc_shadow::route::Landing;
use pc_shadow::{BtsConfig, ChainShape, CtMeta, Quirk, Quirks, Route, Shadow, ShadowConfig};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

fn cfg_err(knob: &'static str, msg: impl std::fmt::Display) -> Error {
    Error::Config { knob, msg: msg.to_string() }
}

fn u32_of(knob: &'static str, v: Option<i64>, default: u32) -> Result<u32> {
    match v {
        None => Ok(default),
        Some(x) => u32::try_from(x).map_err(|_| cfg_err(knob, x)),
    }
}

fn plan_u32(knob: &'static str, v: &Option<String>) -> Result<Option<u32>> {
    v.as_deref().filter(|s| !s.is_empty()).map(|s| s.parse().map_err(|_| cfg_err(knob, format!("{s:?}")))).transpose()
}

/// "s:level[:deg],..." (deg defaults to `deg`).
fn parse_landings(knob: &'static str, spec: &str, deg: u8) -> Result<Vec<(u32, Landing)>> {
    let mut out = vec![];
    for item in spec.split(',').filter(|x| !x.is_empty()) {
        let p: Vec<&str> = item.split(':').collect();
        let num = |i: usize| -> Result<u32> { p.get(i).and_then(|x| x.parse().ok()).ok_or_else(|| cfg_err(knob, format!("bad item {item:?}"))) };
        let d = if p.len() > 2 { num(2)? as u8 } else { deg };
        out.push((num(0)?, Landing { level: num(1)?, deg: d }));
    }
    Ok(out)
}

/// The per-call route parameters the runtime line fixes (everything but the plan's per-site choices).
#[derive(Clone, Debug, PartialEq)]
pub struct RouteEnv {
    pub ring_dim: u32,
    pub stc_first: bool,
    /// FIDESLIB_BTS_STC_FIRST_ROUTES: "all" | "dense" | "sparse" | slot counts.
    pub stc_first_routes: Vec<String>,
    pub real_enabled: bool,
    /// FIDESLIB_BTS_RAISE_DROP per-route defaults (slots -> CKKS levels).
    pub raise_drop_default: BTreeMap<u32, u32>,
    pub sparse_b: bool,
    pub sparse_b_max: Option<u32>,
    pub spru: u32,
    pub iterations: u32,
    /// Sparse routes the runtime holds for this compile (`CompileOpts::routes_built`): a planned refresh whose value
    /// outgrew its planned slot count runs on the smallest of them its period allows (perseus `planned_route_for`,
    /// SM_PERIODIC=2). Empty = the plan's slot count, as planned.
    pub built: Vec<u32>,
    /// A refresh whose value outgrew every built sparse route runs dense (the perseus runtime refreshes it at the planned
    /// slot count, which folds a non-periodic value); false = as the perseus runtime.
    pub dense_fallback: bool,
}

impl RouteEnv {
    pub fn from_config(cfg: &Config) -> Result<RouteEnv> {
        let logn = u32_of("chain.logn", cfg.chain.logn, 16)?;
        let mut raise_drop_default = BTreeMap::new();
        if let Some(spec) = cfg.bts.raise_drop.as_deref() {
            for item in spec.split(',').filter(|x| !x.is_empty()) {
                let (s, k) = item.split_once(':').ok_or_else(|| cfg_err("bts.raise_drop", format!("{item:?}")))?;
                let s: u32 = s.parse().map_err(|_| cfg_err("bts.raise_drop", format!("{item:?}")))?;
                let k: u32 = k.parse().map_err(|_| cfg_err("bts.raise_drop", format!("{item:?}")))?;
                raise_drop_default.insert(s, k);
            }
        }
        let spru = match cfg.bts.spru {
            None | Some(0) => 0,
            Some(1) => 64,
            Some(h) => u32::try_from(h).map_err(|_| cfg_err("bts.spru", h))?,
        };
        Ok(RouteEnv {
            ring_dim: 1 << logn,
            stc_first: cfg.bts.stc_first.unwrap_or(0) > 0,
            stc_first_routes: cfg.bts.stc_first_routes.clone().unwrap_or_else(|| vec!["all".into()]),
            real_enabled: cfg.bts.real.unwrap_or(false),
            raise_drop_default,
            sparse_b: cfg.bts.sparse_b.unwrap_or(false),
            sparse_b_max: cfg.bts.sparse_b_max.map(|m| m.max(0) as u32).filter(|&m| m > 0),
            spru,
            iterations: u32_of("bts.bts_iterations", cfg.bts.bts_iterations, 1)?.max(1),
            built: vec![],
            dense_fallback: false,
        })
    }

    /// FIDESlib `stcFirstRoute`.
    fn stc_first_route(&self, slots: u32) -> bool {
        if !self.stc_first {
            return false;
        }
        let r = &self.stc_first_routes;
        if r.is_empty() || r.iter().any(|x| x == "all") {
            return true;
        }
        r.iter().any(|x| (x == "dense" && slots == 0) || (x == "sparse" && slots > 0) || x.parse::<u32>().ok() == Some(slots))
    }

    /// FIDESlib `sparseBGeometry`: SparseB applies to a sparse route with n = 2s > r/2, r = N/n.
    fn sparse_b_route(&self, slots: u32) -> bool {
        if !self.sparse_b || slots == 0 || slots >= self.ring_dim / 2 || self.sparse_b_max.is_some_and(|m| slots > m) {
            return false;
        }
        let n = 2 * slots as u64;
        let r = self.ring_dim as u64 / n;
        n > r / 2
    }

    /// The route of a slot count with the runtime's defaults.
    pub fn base(&self, slots: u32) -> Route {
        Route {
            slots,
            cf: None,
            real: false,
            raise_drop: self.raise_drop_default.get(&slots).copied().unwrap_or(0),
            stc_first: self.stc_first_route(slots),
            sparse_b: self.sparse_b_route(slots),
            spru_h: if slots == 1 { self.spru } else { 0 },
            iterations: self.iterations,
        }
    }

    /// The route the plan gives the site keyed by `var` (`inner_bootstrap_planned`).
    pub fn route(&self, var: &str, plan: &BlockPlan) -> Route {
        self.route_tagged(var, plan, None)
    }

    /// `route` with the value's tag: `planned_route_for` (the planned s, or the smallest built route the value's
    /// period allows when it outgrew s); the plan's correction factor, raise drop and real chain stay.
    pub fn route_tagged(&self, var: &str, plan: &BlockPlan, tag: Option<&pc_shadow::tag::Tag>) -> Route {
        let p = &plan.file;
        let mut slots = p.sparse_slots.get(var).copied().filter(|&s| s != self.ring_dim / 2).unwrap_or(0);
        if let Some(t) = tag {
            let s2 = planned_route_for(t, slots, &self.built, self.ring_dim / 2);
            // nothing built fits the value's period: dense (any value) when allowed
            let outgrown = slots > 0 && t.known() && !t.periodic_at(slots as i64) && !t.fold_collision_free_at(slots as i64);
            slots = if s2 == slots && outgrown && self.dense_fallback { 0 } else { s2 };
        }
        let mut r = self.base(slots);
        r.cf = p.correction_factor.get(var).copied();
        r.real = plan.real_route.contains(var);
        if let Some(&d) = p.raise_drop.get(var) {
            r.raise_drop = d;
        }
        if plan.shipped_order.contains(var) {
            r.stc_first = false;
        }
        r
    }

    /// The default route that lands a refresh of `input` at `cap` (raw degree-2 landing, or realized at degree 1).
    pub fn route_for_landing(&self, sh: &Shadow, input: &CtMeta, cap: Landing) -> Option<Route> {
        for &s in sh.cfg.bts.landings.keys() {
            let r = self.base(s);
            // a classic sparse route serves a deliberate refresh only when the runtime holds it (the table lists every
            // s SparseB does not take; their landings coincide)
            if s > 1 && s < self.ring_dim / 2 && !r.sparse_b && !self.built.contains(&s) {
                continue;
            }
            // the dense route: StC-first by default (its own shipped-order fallback past the entry); a capture at the
            // shipped order's landing that route cannot give (IMPL_STC_FIRST_KV=0: the K/V push in the shipped order)
            // takes the plain shipped-order route
            let alts = if s == 0 && r.stc_first { vec![r, Route { stc_first: false, ..r }] } else { vec![r] };
            for r in alts {
                let Ok(rec) = sh.bootstrap(input, &r) else { continue };
                let got = if cap.deg == 2 { (rec.out.level.checked_sub(sh.unit())?, 2) } else { (rec.out.level, rec.out.deg) };
                if got == (cap.level, cap.deg) {
                    return Some(r);
                }
            }
        }
        None
    }
}

/// The perseus wrapper's `fold_slots_for`: `s_wanted` when built, else the smallest built s' > s_wanted, else `total`.
fn fold_slots_for(built: &[u32], s_wanted: u32, total: u32) -> u32 {
    if s_wanted >= total || built.contains(&s_wanted) {
        return s_wanted.min(total);
    }
    let mut b = built.to_vec();
    b.sort_unstable();
    b.into_iter().find(|&s| s > s_wanted).unwrap_or(total)
}

/// The perseus wrapper's `planned_route_for` (pc-gpu `bts::fold::planned_route_for`): a planned sparse refresh of a value
/// that is no longer periodic at the planned s runs on the smallest built route its period allows.
fn planned_route_for(tag: &pc_shadow::tag::Tag, s: u32, built: &[u32], total: u32) -> u32 {
    if s == 0 || built.is_empty() || !tag.known() || tag.periodic_at(s as i64) || tag.fold_collision_free_at(s as i64) {
        return s;
    }
    let mut want = s;
    while (want as i64) < tag.period {
        want <<= 1;
    }
    let s2 = fold_slots_for(built, want, total);
    if s2 > s && built.contains(&s2) && tag.periodic_at(s2 as i64) {
        s2
    } else {
        s
    }
}

/// The runtime Shadow a Config (+ the plan's PlanConfig and rules) means: C++ rules, real chain, FIDESlib landings.
pub fn runtime_shadow(cfg: &Config, plan_cfg: &PlanConfig, bootstrap_level: Option<u32>) -> Result<Shadow> {
    let unit = u32_of("chain.composite_degree", cfg.chain.composite_degree, 2)?;
    match cfg.chain.chain.as_deref() {
        None | Some("n32") => {}
        Some(other) => return Err(cfg_err("chain.chain", format!("{other:?}: only the n32 chain shape is known"))),
    }
    // the limbs of the chain the Config names (`chain.preset`, default n32: 54; the lean n32a4: 51)
    let preset = cfg.chain.preset.as_deref().unwrap_or("n32");
    let spec = pc_core::params::ChainSpec::preset(preset).map_err(|e| cfg_err("chain.preset", e))?;
    let real = pc_core::params::Chain::generate(&spec).map_err(|e| cfg_err("chain.preset", e))?;
    let n_q = real.q.len() as u32;
    let chain = ChainShape { n_q, unit };
    let mut quirks = Quirks::none();
    if let Some(names) = &cfg.quirks.shadow {
        for n in names {
            let q = Quirk::ALL.into_iter().find(|q| format!("{q:?}") == *n).ok_or_else(|| cfg_err("quirks.shadow", format!("no quirk {n:?}")))?;
            quirks.0.insert(q);
        }
    }
    let mut bts = BtsConfig::n32_shipped();
    let l = |level| Landing { level, deg: 1 };
    // FIDESlib's realized landings at raise drop 0 (the shipped decode's 512 -> 26 is 24 + its FIDESLIB_BTS_RAISE_DROP=512:1)
    // (1024: the SparseB route SM_PERIODIC=2 builds past two score groups, Llama decode; same depth as 512)
    bts.landings = [(0, l(28)), (1024, l(24)), (512, l(24)), (256, l(24)), (1, l(16))].into_iter().collect();
    // the classic multi-stage sparse routes (pc-gpu `sparse_classic`): every power-of-two s SparseB does not take
    // (s > FIDESLIB_BTS_SPARSE_B_MAX, or SparseB off), on the SPARSE_LEVEL_BUDGET precomputation (default {4, 3})
    let budget = sparse_level_budget(cfg)?;
    let sb_max = if cfg.bts.sparse_b.unwrap_or(false) { cfg.bts.sparse_b_max.filter(|&m| m > 0).map_or(u32::MAX, |m| m as u32) } else { 0 };
    let half = 1u32 << (u32_of("chain.logn", cfg.chain.logn, 16)? - 1);
    let mut s = 2;
    while s < half {
        if s > sb_max {
            bts.landings.insert(s, pc_shadow::route::multi_stage_landing(budget, unit));
        }
        s <<= 1;
    }
    if let Some(bl) = bootstrap_level {
        bts.bootstrap_level = bl;
    }
    bts.level_limit = u32_of("bts.auto_bts_level", cfg.bts.auto_bts_level, 44)?;
    bts.max_input_level = u32_of("bts.bts_max_input_level", cfg.bts.bts_max_input_level, bts.max_input_level)?;
    bts.k_max = u32_of("bts.k_max", cfg.bts.k_max, bts.k_max)?;
    if let Some(d) = &cfg.bts.raise_drops {
        bts.raise_drops = d.iter().map(|x| x.parse().map_err(|_| cfg_err("bts.raise_drops", x))).collect::<Result<_>>()?;
    }
    if let Some(e) = plan_u32("plan.stc_first_entry", &plan_cfg.stc_first_entry)? {
        bts.stc_first_entry = Some(e);
    }
    if let Some(spec) = plan_cfg.stc_first_fallback.as_deref().filter(|s| !s.is_empty()) {
        bts.stc_first_fallback = Some(parse_landings("plan.stc_first_fallback", &format!("0:{spec}"), 1)?[0].1);
    }
    if let Some(cf) = plan_u32("plan.cf_max", &plan_cfg.cf_max)? {
        bts.cf_window.1 = cf;
    }
    if cfg.bts.stc_first.unwrap_or(0) == 0 {
        // no StC-first: the dense route is the shipped order everywhere
        if let Some(fb) = bts.stc_first_fallback {
            bts.landings.insert(0, fb);
        }
        bts.stc_first_entry = None;
        bts.stc_first_fallback = None;
    }
    // the shipped order (a dense refresh not run StC-first) lands where a too-deep StC-first input falls back to
    bts.dense_shipped = bts.stc_first_fallback;
    // a refresh input's deepest effective level: the top of the level above q0's (n32: 50; n32a4's single s1: 48)
    bts.input_room = Some(n_q - real.groups.iter().take(2).sum::<usize>() as u32);
    Ok(Shadow::new(ShadowConfig { chain, bts, quirks }, None))
}

/// SPARSE_LEVEL_BUDGET {CtS, StC} of the classic sparse routes ("3,3" / "3:3"; unset = the dense {4, 3}).
pub fn sparse_level_budget(cfg: &Config) -> Result<[u32; 2]> {
    match &cfg.bts.sparse_level_budget {
        None => Ok([4, 3]),
        Some(v) => {
            let xs: Vec<u32> = v.iter().flat_map(|x| x.split([',', ':'])).filter(|x| !x.is_empty()).map(|x| x.trim().parse().map_err(|_| cfg_err("bts.sparse_level_budget", format!("{v:?}")))).collect::<Result<_>>()?;
            match xs.as_slice() {
                [a, b] if *a > 0 && *b > 0 => Ok([*a, *b]),
                _ => Err(cfg_err("bts.sparse_level_budget", format!("{v:?}: two entries (CtS, StC)"))),
            }
        }
    }
}

/// `plan_block`'s seeds (perseus plan `plan_block` / `sim_dump.py`), in absolute levels: the entry var takes the
/// block entry (the previous block's exit), KV-cache reads their read level, other inputs their first consumer.
pub fn seeds(bg: &BlockGraph, rules: &crate::plan::Rules, entry: Option<(u32, u8)>, inputs: &[String]) -> Result<HashMap<String, (u32, u8)>> {
    let g = &bg.graph;
    let bl = rules.bootstrap_level.ok_or_else(|| cfg_err("rules.bootstrap_level", "missing"))? as f64;
    let source = rules.source_level.unwrap_or(36) as f64;
    let cache_read = rules.cache_read_level.unwrap_or(36) as f64;
    let ins: BTreeSet<&str> = g.inputs.iter().map(String::as_str).collect();
    let mut input_level_of: HashMap<&str, f64> = HashMap::new();
    let mut first_consumer: HashMap<&str, f64> = HashMap::new();
    let mut entry_var: Option<&str> = None;
    for n in &g.nodes {
        for (i, v) in n.inputs.iter().enumerate() {
            if !ins.contains(v.as_str()) {
                continue;
            }
            if entry_var.is_none() {
                entry_var = Some(v.as_str());
            }
            if let Some(Some(l)) = n.input_levels.get(i) {
                input_level_of.entry(v.as_str()).or_insert(*l);
            }
            if let Some(ol) = n.output_level {
                first_consumer.entry(v.as_str()).or_insert(ol - n.cost as f64);
            }
        }
    }
    let cache_side = |v: &str| entry.is_some() && Some(v) != entry_var && v.starts_with("anon_") && input_level_of.contains_key(v);
    let level = |v: &str| -> f64 {
        let exp = parse_lvl_suffix(v);
        if v.contains("cache.") {
            return exp.unwrap_or(cache_read);
        }
        if cache_side(v) {
            return input_level_of[v];
        }
        if let Some((l, _)) = entry {
            return l as f64;
        }
        if let Some(&f) = first_consumer.get(v) {
            return f;
        }
        if let Some(e) = exp {
            return e;
        }
        input_level_of.get(v).copied().unwrap_or(source)
    };
    let mut out = HashMap::new();
    for v in inputs {
        let l = level(v);
        let d = if v.contains("cache.") || cache_side(v) {
            2
        } else if let Some((_, d)) = entry {
            d
        } else {
            match (parse_lvl_suffix(v), first_consumer.get(v.as_str())) {
                (Some(e), Some(&f)) => {
                    if e > f {
                        2
                    } else {
                        1
                    }
                }
                _ => {
                    if l > bl {
                        2
                    } else {
                        1
                    }
                }
            }
        };
        if l < 0.0 || l.fract() != 0.0 {
            return Err(Error::Graph { block: bg.name.clone(), msg: format!("input {v} seeds at non-level {l}") });
        }
        out.insert(v.clone(), (l as u32, d));
    }
    Ok(out)
}

/// Compile options.
#[derive(Clone, Debug, Default)]
pub struct CompileOpts {
    /// Return the program even when the check failed (the report lists the errors).
    pub keep_going: bool,
    /// Skip K-batching (`k_max` = 1).
    pub no_batch: bool,
    pub schedule: batch::Schedule,
    /// Bootstrap siblings may differ in correction factor (a per-member EvalMod constant). Off: a batched
    /// bootstrap takes one route.
    pub batch_cf_free: bool,
    /// Per block label: graph names whose final state must survive the block (outputs after the exit var) although
    /// something inside the block reads them, e.g. the KV cache states the next token's run takes as inputs.
    pub live_out: BTreeMap<String, Vec<String>>,
    /// Per block label: (level, deg) of block inputs whose entry state the driver knows (a K/V state written by
    /// the previous token), replacing `seeds`' guess for those names.
    pub input_metas: BTreeMap<String, BTreeMap<String, (u32, u8)>>,
    /// The sparse routes the runtime holds while these blocks run (SM_PERIODIC=2: the base routes + the class's
    /// Gp x tH route): planned refreshes route by their value's period (`RouteEnv::built`). Empty = as planned.
    pub routes_built: Vec<u32>,
    /// `RouteEnv::dense_fallback`.
    pub dense_fallback: bool,
}

/// One block: graph + its plan -> Program (`entry` = the block's entry state, None = plan_block's own seeds).
#[allow(clippy::too_many_arguments)]
pub fn compile_block(
    label: &str,
    bg: &BlockGraph,
    plan: &BlockPlan,
    cfg: &Config,
    plan_cfg: &PlanConfig,
    pcmm_calib: &BTreeMap<String, f64>,
    entry: Option<(u32, u8)>,
    opts: &CompileOpts,
) -> Result<Program> {
    let sh = runtime_shadow(cfg, plan_cfg, plan.file.rules.bootstrap_level)?;
    let mut env = RouteEnv::from_config(cfg)?;
    env.built = opts.routes_built.clone();
    env.dense_fallback = opts.dense_fallback;
    let res = resolve::resolve(bg, pcmm_calib)?;
    let mut seeds = seeds(bg, &plan.file.rules, entry, &res.inputs)?;
    if let Some(m) = opts.input_metas.get(label) {
        for (k, &v) in m {
            if let Some(s) = seeds.get_mut(k) {
                *s = v;
            }
        }
    }
    let mut lw = apply_plan::lower(&sh, &env, plan, &res, &seeds);
    let keep: Vec<VarId> = opts.live_out.get(label).map(|ns| ns.iter().filter_map(|n| lw.cur.get(n).copied()).collect()).unwrap_or_default();
    stream::fuse(&mut lw, &sh, &keep);
    let mut acct = lw.acct.clone();
    acct.graph_nodes = bg.n_nodes;
    acct.erased_autos = bg.n_auto;
    acct.merged = res.elided_by_merge;
    let mut lower_errors = lw.check.errors.clone();
    if !acct.balanced() {
        lower_errors.push(format!("op accounting does not balance: {acct:?}"));
    }
    let s = &plan.file.summary;
    let mut pins = BTreeMap::new();
    for (name, &l) in &s.final_named_levels {
        if lw.cur.contains_key(name) {
            pins.insert(name.clone(), (l, s.final_named_degs.get(name).copied().unwrap_or(1)));
        }
    }
    let names: BTreeMap<String, VarId> = lw.cur.iter().map(|(k, &v)| (k.clone(), v)).collect();
    // outputs: the plan's exit var, then every final state nothing reads
    let mut consumed: BTreeSet<VarId> = lw.ops.iter().flat_map(|o| o.ins.iter().copied()).collect();
    // a fused sum's intermediates are never materialized: not outputs
    for o in &lw.ops {
        if let crate::program::Kind::CoeffSum { fused, .. } = &o.kind {
            consumed.extend(fused.iter().copied());
        }
    }
    let input_set: BTreeSet<VarId> = lw.inputs.iter().copied().collect();
    let mut outputs: Vec<VarId> = vec![];
    if let Some(&v) = s.exit_var.as_ref().and_then(|e| lw.cur.get(e)) {
        outputs.push(v);
    }
    for (i, v) in lw.vars.iter().enumerate() {
        let i = i as VarId;
        if !v.name.contains('~') && !consumed.contains(&i) && !input_set.contains(&i) && !outputs.contains(&i) {
            outputs.push(i);
        }
    }
    if let Some(names) = opts.live_out.get(label) {
        let mut unknown = vec![];
        for n in names {
            match lw.cur.get(n) {
                Some(&v) if !outputs.contains(&v) => outputs.push(v),
                Some(_) => {}
                None => unknown.push(n.clone()),
            }
        }
        if !unknown.is_empty() {
            return Err(Error::Graph { block: label.into(), msg: format!("live_out names the graph does not produce: {}", unknown.join(", ")) });
        }
    }
    let stamp = serde_json::json!({ "config": cfg.load_bearing(), "plan": plan_cfg }).to_string();
    let mut p = Program {
        block: label.to_string(),
        ops: lw.ops,
        vars: lw.vars,
        inputs: lw.inputs,
        outputs,
        names,
        pins,
        pt_refs: lw.pt_refs,
        bts_sites: lw.sites,
        cfg_stamp: stamp,
        accounting: acct,
        ..Default::default()
    };
    p.check.pt_relevels = lw.check.pt_relevels;
    p.check.ignored_anchors = lw.check.ignored_anchors.clone();
    p.check.inexact_scalars = res.inexact_scalars;
    let k_max = if opts.no_batch { 1 } else { sh.cfg.bts.k_max.max(1) };
    // PC_BTS_PAIR: sibling real refreshes in pairs, where the inputs enter StC-first
    let pair = sh.cfg.bts.stc_first_entry.filter(|_| cfg.bts.pair.unwrap_or(false)).map(|entry| Pairing { entry, unit: sh.cfg.chain.unit });
    batch::batch(&mut p, k_max, opts.schedule, opts.batch_cf_free, pair);
    let word_bytes = 4u64;
    alloc::alloc(&mut p, sh.cfg.chain.n_q, env.ring_dim as u64 * word_bytes);
    let mut rep = check::verify(&sh, &p);
    let mut errors = lower_errors;
    errors.append(&mut rep.errors);
    rep.errors = errors;
    p.check = rep;
    if !p.check.errors.is_empty() && !opts.keep_going {
        return Err(Error::Check { block: label.into(), n: p.check.errors.len(), first: p.check.errors[0].clone() });
    }
    Ok(p)
}

/// Every block (and variant) of a plan dir, entries chained from the previous block's compiled exit.
pub struct DirCompile {
    pub programs: Vec<(String, Result<Program>)>,
}

pub fn load_opts(plan_cfg: &PlanConfig) -> Result<LoadOpts> {
    let unit = plan_u32("plan.level_unit", &plan_cfg.level_unit)?.unwrap_or(2) as i64;
    let remap = match &plan_cfg.delib_landing_remap {
        Some(v) if !v.is_empty() => LoadOpts::parse_remap(&v.join(","))?,
        _ => BTreeMap::new(),
    };
    Ok(LoadOpts { level_unit: unit, delib_landing_remap: remap })
}

/// Compile a plan directory against its graph directory.
pub fn compile_dir(plan_dir: &Path, graph_dir: &Path, opts: &CompileOpts) -> Result<DirCompile> {
    let pd = PlanDir::open(plan_dir)?;
    let plan_cfg = pd.plan_config()?;
    let lo = load_opts(&plan_cfg)?;
    let first_entry = match (plan_u32("plan.first_entry_level", &plan_cfg.first_entry_level)?, plan_u32("plan.first_entry_deg", &plan_cfg.first_entry_deg)?) {
        (Some(l), d) => Some((l, d.unwrap_or(1) as u8)),
        _ => None,
    };
    let mut out = vec![];
    let mut prev_exit: Option<(u32, u8)> = first_entry;
    for (b, variant, path) in &pd.blocks {
        let label = match variant {
            Some(v) => format!("block_{b}_{v}"),
            None => format!("block_{b}"),
        };
        let gpath: PathBuf = graph_dir.join(format!("block_{b}")).join("graph.json");
        let r = (|| -> Result<Program> {
            let plan = BlockPlan::load(path)?;
            let cfg = pd.config(&plan)?;
            let bg = BlockGraph::load(&gpath, &label, &lo)?;
            let entry = match variant.as_deref() {
                Some("feedback") => bg.fed_back_entry,
                Some(_) => None,
                None => prev_exit,
            };
            compile_block(&label, &bg, &plan, &cfg, &plan_cfg, &pd.pcmm_calib, entry, opts)
        })();
        if variant.is_none() {
            // the next block enters at this block's compiled exit (what the runtime hands it)
            prev_exit = match &r {
                Ok(p) => p.outputs.first().map(|&v| (p.vars[v as usize].meta.level, p.vars[v as usize].meta.deg)),
                // a block that did not compile: the next one enters where its plan says it exits
                Err(_) => BlockPlan::load(path)
                    .ok()
                    .and_then(|bp| bp.file.summary.exit_level.map(|l| (l, bp.file.summary.exit_deg.unwrap_or(1)))),
            };
        }
        out.push((label, r));
    }
    Ok(DirCompile { programs: out })
}
