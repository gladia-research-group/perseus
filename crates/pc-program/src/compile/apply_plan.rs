//! Plan application = the lowering pass: resolved nodes in planner order -> ops on program vars, with the plan's
//! decisions applied as the perseus C++ wrapper applies them at run time (`maybe_apply_planned_bootstrap_after`,
//! `inner_bootstrap_planned`, `bootstrap_hint`, `maybe_apply_planned_rescale_after`): placements refresh an op's
//! output (fire-once, realized iff a `rescale_after` anchor), hints fire per `hint_fire` or their threshold,
//! unfired hints / level_hints / clones alias, `level_reduce` drops, deliberate refreshes keep their captured
//! landing, PCMM/CCMM finishes become `Finish` ops (rules in docs/ARCHITECTURE.md "pc-program apply_plan").
//!
//! The Shadow runs alongside (`check::rule`) because the C++ decisions it mirrors read the live state (a realize
//! only on degree 2, the StC-first fallback on the input's effective level, a plaintext encoded at its use).
use super::check::rule;
use super::resolve::{RKind, Resolved};
use super::RouteEnv;
use crate::plan::BlockPlan;
use crate::program::{Accounting, BtsSite, CheckReport, Kind, Op, Origin, PtForm, PtRef, SiteKind, Var, VarId};
use pc_shadow::route::Landing;
use pc_shadow::{CtMeta, Shadow};
use std::collections::{BTreeSet, HashMap};

pub struct Lowered {
    pub ops: Vec<Op>,
    pub vars: Vec<Var>,
    pub inputs: Vec<VarId>,
    pub pt_refs: Vec<PtRef>,
    pub sites: Vec<BtsSite>,
    pub cur: HashMap<String, VarId>,
    pub acct: Accounting,
    pub check: CheckReport,
}

struct Lower<'a> {
    sh: &'a Shadow,
    tags: &'a HashMap<String, pc_shadow::tag::Tag>,
    env: &'a RouteEnv,
    plan: &'a BlockPlan,
    ops: Vec<Op>,
    vars: Vec<Var>,
    cur: HashMap<String, VarId>,
    pt_refs: Vec<PtRef>,
    sites: Vec<BtsSite>,
    placed_left: BTreeSet<String>,
    rescale_left: BTreeSet<String>,
    acct: Accounting,
    check: CheckReport,
    deliberate_outs: BTreeSet<String>,
}

impl<'a> Lower<'a> {
    fn new_var(&mut self, name: &str, meta: CtMeta) -> VarId {
        self.vars.push(Var { name: name.to_string(), meta, slot: None });
        (self.vars.len() - 1) as VarId
    }

    fn meta(&self, v: VarId) -> CtMeta {
        self.vars[v as usize].meta
    }

    /// Give `name`'s current var a scratch suffix if it carries the graph name (the graph name is reserved for
    /// the final state, which is what the plan pins).
    fn demote(&mut self, name: &str, suffix: &str) {
        if let Some(&v) = self.cur.get(name) {
            if self.vars[v as usize].name == name {
                self.vars[v as usize].name = format!("{name}~{suffix}");
            }
        }
    }

    fn emit(&mut self, kind: Kind, ins: Vec<VarId>, out_names: &[String], origin: Origin) -> Result<Vec<VarId>, String> {
        let metas: Vec<CtMeta> = ins.iter().map(|&v| self.meta(v)).collect();
        let (pre, outs) = rule(self.sh, &kind, &metas, &self.pt_refs, &self.sites)?;
        let mut ids = vec![];
        for (k, n) in out_names.iter().enumerate() {
            let m = outs.get(k).or(outs.first()).copied().ok_or("rule returned no output")?;
            let id = self.new_var(n, m);
            self.cur.insert(n.clone(), id);
            ids.push(id);
        }
        self.ops.push(Op { kind, ins, outs: ids.clone(), pre, origin, inplace: false });
        Ok(ids)
    }

    /// The bootstrap of `name`'s current var on the route the plan gives `site_var`.
    fn refresh(&mut self, name: &str, site_var: &str, kind: SiteKind, origin: Origin, captured: Option<Landing>) -> Result<(), String> {
        let input = *self.cur.get(name).ok_or_else(|| format!("refresh of unbound {name}"))?;
        let in_meta = self.meta(input);
        let p = &self.plan.file;
        let mut route = self.env.route_tagged(site_var, self.plan, self.tags.get(name));
        if captured.is_none() {
            route = self.keep_planned_landing(name, site_var, &in_meta, route);
        }
        let mut captured = captured;
        if let Some(cap) = captured {
            // a deliberate refresh: the route whose landing the capture recorded (the code's scope chose it)
            route = match self.env.route_for_landing(self.sh, &in_meta, cap) {
                Some(r) => r,
                // a dense refresh captured on StC-first whose input now sits past the entry: the shipped order, at
                // that order's default CF (the StC-first route's own default is tuned for slot values), landing at
                // the fallback the planner prices
                None if self.past_stc_entry(&in_meta) => {
                    captured = None;
                    pc_shadow::route::Route { stc_first: false, ..self.env.base(0) }
                }
                None => return Err(format!("deliberate {name}: no route lands at ({}, {}) from ({}, {})", cap.level, cap.deg, in_meta.level, in_meta.deg)),
            };
        }
        let realize = match kind {
            SiteKind::Deliberate => false,
            _ if self.plan.rescale_bound => self.rescale_left.remove(site_var),
            _ => true,
        };
        let raw_landing = match captured {
            Some(c) => c,
            None => {
                let r = self.sh.bootstrap(&in_meta, &route).map_err(|e| format!("refresh {name}: {e}"))?;
                Landing { level: r.out.level.checked_sub(self.sh.unit()).ok_or("landing above the chain top")?, deg: 2 }
            }
        };
        let site = BtsSite {
            var: site_var.to_string(),
            kind,
            route,
            input: in_meta,
            landing: raw_landing,
            realize,
            prescale: p.prescale.get(site_var).copied(),
            offset: p.offset.get(site_var).copied(),
        };
        self.sites.push(site);
        let sid = (self.sites.len() - 1) as u32;
        match kind {
            SiteKind::Placement => self.acct.bootstraps += 1,
            SiteKind::Hint => self.acct.hint_bootstraps += 1,
            SiteKind::Deliberate => self.acct.deliberate_bootstraps += 1,
        }
        self.demote(name, "pre");
        let raw_name = if realize { format!("{name}~raw") } else { name.to_string() };
        let ids = self.emit(Kind::Bootstrap(sid), vec![input], &[raw_name], origin)?;
        if realize && self.meta(ids[0]).deg == 2 {
            self.acct.rescales += 1;
            self.emit(Kind::Rescale, ids, &[name.to_string()], Origin::Realize(site_var.to_string()))?;
        } else if realize {
            // nothing to pay: the raw state is the final one
            let v = ids[0];
            self.vars[v as usize].name = name.to_string();
        }
        Ok(())
    }

    /// SM_PERIODIC=2: a planned refresh whose value outgrew its planned slot count runs on a bigger built route
    /// (`RouteEnv::route_tagged`), which may land deeper (a classic multi-stage route at 32, SparseB at 24). The plan's
    /// raise drop is absolute and was sized on the planned route: kept, it lands that much deeper than the plan pinned
    /// the refresh (a step landing floor such as PLAN_STEP_LANDING softmax:32 absorbs part of it on the planned route
    /// only). Returns the route with the largest built drop (FIDESLIB_BTS_RAISE_DROPS, 0, the route's default) whose
    /// realized landing is no deeper than the plan's pin (else than the planned route's landing); the route unchanged
    /// when none is (the check reports it).
    fn keep_planned_landing(&self, name: &str, site_var: &str, input: &CtMeta, route: pc_shadow::Route) -> pc_shadow::Route {
        let planned = self.env.route(site_var, self.plan);
        if route.slots == planned.slots || route.raise_drop == 0 {
            return route;
        }
        let level = |r: &pc_shadow::Route| self.sh.bootstrap(input, r).ok().map(|x| x.out.level);
        let s = &self.plan.file.summary;
        // a pinned raw (unrealized, degree-2) landing sits one unit above the realized level the Shadow reports
        let want = match s.final_named_levels.get(name) {
            Some(&l) => Some(l + if s.final_named_degs.get(name).copied().unwrap_or(1) == 2 { self.sh.unit() } else { 0 }),
            None => level(&planned),
        };
        let (Some(want), Some(got)) = (want, level(&route)) else { return route };
        if got <= want {
            return route;
        }
        let mut drops: Vec<u32> = self.sh.cfg.bts.raise_drops.iter().copied().chain([0, self.env.base(route.slots).raise_drop]).filter(|&d| d < route.raise_drop).collect();
        drops.sort_unstable_by(|a, b| b.cmp(a));
        drops.dedup();
        for d in drops {
            let r = pc_shadow::Route { raise_drop: d, ..route };
            if level(&r).is_some_and(|l| l <= want) {
                return r;
            }
        }
        route
    }

    /// The dense default route is StC-first and `input` is past its entry (the route lands at the shipped order's
    /// fallback).
    fn past_stc_entry(&self, input: &CtMeta) -> bool {
        let b = &self.sh.cfg.bts;
        let r = self.env.base(0);
        r.stc_first
            && b.stc_first_fallback.is_some()
            && b.stc_first_entry.is_some_and(|e| input.effective_level(self.sh.unit()) > e)
            && self.sh.bootstrap(input, &r).is_ok_and(|x| b.stc_first_fallback.is_some_and(|f| (x.out.level, x.out.deg) == (f.level, f.deg)))
    }

    /// The wrapper's op-exit hook on a graph-named output.
    fn exit_hook(&mut self, name: &str) -> Result<(), String> {
        if self.placed_left.remove(name) {
            return self.refresh(name, name, SiteKind::Placement, Origin::Placement(name.to_string()), None);
        }
        if self.plan.rescale_bound && self.rescale_left.remove(name) {
            let v = *self.cur.get(name).ok_or("unbound")?;
            if self.meta(v).deg == 2 {
                self.demote(name, "raw");
                self.acct.rescales += 1;
                self.emit(Kind::Rescale, vec![v], &[name.to_string()], Origin::Realize(name.to_string()))?;
            }
        }
        Ok(())
    }

}

pub fn lower(
    sh: &Shadow,
    env: &RouteEnv,
    plan: &BlockPlan,
    r: &Resolved,
    seeds: &HashMap<String, (u32, u8)>,
) -> Lowered {
    let mut lw = Lower {
        sh,
        tags: &r.tags,
        env,
        plan,
        ops: vec![],
        vars: vec![],
        cur: HashMap::new(),
        pt_refs: vec![],
        sites: vec![],
        placed_left: plan.placed.clone(),
        rescale_left: plan.rescale_after.clone(),
        acct: Accounting::default(),
        check: CheckReport::default(),
        deliberate_outs: BTreeSet::new(),
    };
    let mut inputs = vec![];
    for name in &r.inputs {
        let (l, d) = seeds.get(name).copied().unwrap_or((sh.cfg.bts.bootstrap_level, 1));
        let id = lw.new_var(name, CtMeta::new(l, d));
        lw.cur.insert(name.clone(), id);
        inputs.push(id);
    }
    let unit = sh.unit();
    for n in &r.nodes {
        let res: Result<(), String> = (|| {
            let ins: Vec<VarId> = n
                .ins
                .iter()
                .map(|x| lw.cur.get(x).copied().ok_or_else(|| format!("reads {x}, which nothing produced")))
                .collect::<Result<_, _>>()?;
            let out = n.outs[0].clone();
            let origin = Origin::Node(n.idx);
            let pt_ref = |lw: &mut Lower, p: usize| -> u32 {
                let d = &r.pts[p];
                let a = lw.meta(ins[0]);
                let (level, deg) = match (d.form, d.fixed) {
                    (PtForm::Slot, Some(f)) => f,
                    // the Python impl's encode_like: the ciphertext's effective level, degree 1
                    _ => (a.effective_level(unit), 1),
                };
                if d.captured_level.is_some_and(|c| c != level) {
                    lw.check.pt_relevels += 1;
                }
                lw.pt_refs.push(PtRef { name: d.name.clone(), level, deg, form: d.form, captured_level: d.captured_level });
                (lw.pt_refs.len() - 1) as u32
            };
            let leaf = |k: Kind| Some(k);
            let kind = match &n.kind {
                RKind::Add => leaf(Kind::Add),
                RKind::Sub => leaf(Kind::Sub),
                RKind::AddPt(p) => {
                    let id = pt_ref(&mut lw, *p);
                    leaf(Kind::AddPt(id))
                }
                RKind::MultPt(p) => {
                    let id = pt_ref(&mut lw, *p);
                    leaf(Kind::MultPt(id))
                }
                RKind::AddConst(c) => leaf(Kind::AddConst(*c)),
                RKind::MultConst(c) => leaf(Kind::MultConst(*c)),
                RKind::Mult => leaf(Kind::Mult),
                RKind::Square => leaf(Kind::Square),
                RKind::Negate => leaf(Kind::Negate),
                RKind::Rotate(k) => leaf(Kind::Rotate(*k)),
                RKind::RotateSum(s) => leaf(Kind::RotateSum(s.clone())),
                RKind::Conjugate => leaf(Kind::Conjugate),
                RKind::MultI => leaf(Kind::MultI),
                RKind::Clone | RKind::LevelHint { .. } => {
                    lw.acct.elided += 1;
                    lw.cur.insert(out.clone(), ins[0]);
                    None
                }
                RKind::LevelReduce { target } => {
                    if *target > lw.meta(ins[0]).level {
                        leaf(Kind::DropTo(*target))
                    } else {
                        lw.acct.elided += 1;
                        lw.cur.insert(out.clone(), ins[0]);
                        None
                    }
                }
                RKind::Hint { thr, acct } => {
                    let a = lw.meta(ins[0]);
                    let fired = if plan.hints_bound {
                        plan.hint_fire.contains(&out)
                    } else {
                        let lv = if *acct { a.effective_level(unit) } else { a.level };
                        lv > *thr
                    };
                    lw.cur.insert(out.clone(), ins[0]);
                    if fired {
                        if lw.placed_left.remove(&out) {
                            lw.acct.placements_on_hints += 1;
                        }
                        lw.refresh(&out, &out, SiteKind::Hint, origin.clone(), None)?;
                    } else {
                        lw.acct.elided += 1;
                    }
                    None
                }
                RKind::Deliberate { level, deg, real } => {
                    lw.deliberate_outs.insert(out.clone());
                    lw.cur.insert(out.clone(), ins[0]);
                    lw.refresh(&out, &out, SiteKind::Deliberate, origin.clone(), Some(Landing { level: *level, deg: *deg }))?;
                    if *real && lw.env.real_enabled {
                        // `bootstrap_real` (v2 args.route.real): the real-payload chain on the same route family
                        // (the perseus wrapper's set_real_payload scope; the landing is the same)
                        if let Some(s) = lw.sites.last_mut() {
                            s.route.real = true;
                        }
                    }
                    // the code's own bootstrap() records without the op-exit hook
                    return Ok(());
                }
                RKind::Finish { level, deg, desc, lands, descs } => {
                    lw.acct.finishes += 1;
                    let outs: Vec<Landing> = if lands.iter().all(|&l| l == (*level, *deg)) { vec![] } else { lands.iter().map(|&(level, deg)| Landing { level, deg }).collect() };
                    let kind = Kind::Finish { landing: Landing { level: *level, deg: *deg }, desc: merged_desc(desc, descs), outs };
                    let origin = if n.merged.is_empty() {
                        origin.clone()
                    } else {
                        Origin::Nodes(std::iter::once(n.idx).chain(n.merged.iter().copied()).collect())
                    };
                    lw.emit(kind, ins.clone(), &n.outs, origin)?;
                    for o in &n.outs {
                        lw.exit_hook(o)?;
                    }
                    return Ok(());
                }
                RKind::FoldBootstrap { .. } => {
                    return Err("fold_bootstrap is not lowered (no shipped graph records one)".into());
                }
            };
            if let Some(k) = kind {
                lw.emit(k, ins, &[out.clone()], origin)?;
            }
            lw.exit_hook(&out)?;
            Ok(())
        })();
        if let Err(msg) = res {
            lw.check.errors.push(format!("node {} {:?} -> {:?} ({}): {msg}", n.idx, n.kind, n.outs, n.step));
            // keep going: bind the outputs to their inputs so later nodes still resolve
            if let Some(&a) = n.ins.first().and_then(|x| lw.cur.get(x)) {
                for o in &n.outs {
                    lw.cur.entry(o.clone()).or_insert(a);
                }
            }
        }
    }
    for v in &lw.placed_left {
        lw.check.errors.push(format!("placement on {v}: no op produced it"));
    }
    if plan.rescale_bound {
        for v in &lw.rescale_left {
            if lw.deliberate_outs.contains(v) {
                // the code's own bootstrap() never reaches the op-exit hook, and the planner's sim never realizes a
                // deliberate landing either: the anchor is inert on both sides
                lw.check.ignored_anchors.push(v.clone());
            } else {
                lw.check.errors.push(format!("rescale_after anchor {v} never applied"));
            }
        }
    }
    lw.acct.placements = plan.placed.len();
    lw.acct.ops = lw.ops.len();
    Lowered {
        ops: lw.ops,
        vars: lw.vars,
        inputs,
        pt_refs: lw.pt_refs,
        sites: lw.sites,
        cur: lw.cur,
        acct: lw.acct,
        check: lw.check,
    }
}

/// One descriptor for a merged finish: the first sibling's, plus `raise_drops` (one per output, op order) when the
/// siblings' `raise_drop` differ (a PCMM with per-output raise drops: gate_up's up half).
fn merged_desc(first: &Option<String>, descs: &[Option<String>]) -> Option<String> {
    let f = first.as_ref()?;
    let parse = |d: &Option<String>| d.as_ref().and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok());
    let mut v = match parse(first) {
        Some(v) if v.is_object() => v,
        _ => return Some(f.clone()),
    };
    let drops: Vec<serde_json::Value> = descs.iter().map(|d| parse(d).and_then(|x| x.get("raise_drop").cloned()).unwrap_or(serde_json::Value::Null)).collect();
    if drops.iter().all(|d| *d == drops[0]) {
        return Some(f.clone());
    }
    v["raise_drops"] = serde_json::Value::Array(drops);
    Some(v.to_string())
}
