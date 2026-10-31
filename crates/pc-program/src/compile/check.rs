//! The compile-time level check. `rule` is the one place an op's recipe comes from (lowering and verification
//! both call it): the pc-shadow rules in runtime mode (the C++ FLEXIBLEAUTO behaviour, real chain), so the
//! `CtMeta` every var carries is what the executor will see. `verify` re-runs the Shadow over a finished program
//! (batched or not) from its inputs and asserts (a) every op's recorded preludes and output metas, (b) the plan's
//! pins: `final_named_levels` one-sided (a var may sit richer than planned, never deeper: the perseus runtime's
//! `[plan_level_error]`) and `final_named_degs` exactly (`[plan_deg_error]`), (c) no var reaches the reactive
//! ceiling the plan did not predict (`[plan_bts_error]`, an unplanned safety-net bootstrap).
use crate::program::{BtsSite, CheckReport, Kind, Program, PtRef, SiteKind, VarId};
use pc_shadow::rules::PtMeta;
use pc_shadow::{CtMeta, Prelude, Shadow};

/// The recipe of one (non-batched) op: operand preludes and the output metas.
pub fn rule(sh: &Shadow, kind: &Kind, ins: &[CtMeta], pts: &[PtRef], sites: &[BtsSite]) -> Result<(Vec<Prelude>, Vec<CtMeta>), String> {
    let e = |x: pc_shadow::Error| x.to_string();
    let one = |k: usize| -> Result<(), String> {
        if ins.len() == k {
            Ok(())
        } else {
            Err(format!("{} takes {k} cipher input(s), got {}", kind.name(), ins.len()))
        }
    };
    let pt = |id: u32| -> Result<PtMeta, String> {
        let p = pts.get(id as usize).ok_or_else(|| format!("no plaintext {id}"))?;
        Ok(PtMeta { level: p.level, deg: p.deg, nf: 0.0, tag: Default::default() })
    };
    let r = match kind {
        Kind::Add | Kind::Sub => {
            one(2)?;
            if matches!(kind, Kind::Add) {
                sh.add(&ins[0], &ins[1]).map_err(e)?
            } else {
                sh.sub(&ins[0], &ins[1]).map_err(e)?
            }
        }
        Kind::AddPt(p) => {
            one(1)?;
            sh.add_pt(&ins[0], &pt(*p)?).map_err(e)?
        }
        Kind::MultPt(p) => {
            one(1)?;
            sh.mult_pt(&ins[0], &pt(*p)?).map_err(e)?
        }
        Kind::AddConst(c) => {
            one(1)?;
            sh.add_scalar(&ins[0], c.value).map_err(e)?
        }
        Kind::MultConst(c) => {
            one(1)?;
            sh.mult_scalar(&ins[0], c.value).map_err(e)?
        }
        Kind::Mult => {
            one(2)?;
            sh.mult(&ins[0], &ins[1]).map_err(e)?
        }
        Kind::Square => {
            one(1)?;
            sh.square(&ins[0]).map_err(e)?
        }
        Kind::MultRescale { square, dbl, .. } => {
            let r = if *square {
                one(1)?;
                sh.square(&ins[0]).map_err(e)?
            } else {
                one(2)?;
                sh.mult(&ins[0], &ins[1]).map_err(e)?
            };
            let mut m = r.out;
            if *dbl {
                m = sh.add(&m, &m).map_err(e)?.out;
            }
            let out = sh.realize(&m).map_err(e)?.out;
            return Ok((r.pre, vec![out]));
        }
        Kind::Negate => {
            one(1)?;
            sh.negate(&ins[0]).map_err(e)?
        }
        Kind::Rotate(k) => {
            one(1)?;
            sh.rotate(&ins[0], *k)
        }
        Kind::RotateSum(steps) => {
            one(1)?;
            sh.rotate_sum(&ins[0], steps)
        }
        Kind::Conjugate => {
            one(1)?;
            sh.conjugate(&ins[0])
        }
        Kind::MultI => {
            one(1)?;
            sh.mult_i(&ins[0])
        }
        Kind::Rescale => {
            one(1)?;
            if ins[0].deg != 2 {
                return Err(format!("rescale of a degree-{} ciphertext", ins[0].deg));
            }
            sh.realize(&ins[0]).map_err(e)?
        }
        Kind::DropTo(t) => {
            one(1)?;
            sh.drop_to_level(&ins[0], *t)
        }
        Kind::Bootstrap(s) => {
            one(1)?;
            let site = sites.get(*s as usize).ok_or_else(|| format!("no bootstrap site {s}"))?;
            // the raise rescales a pending product, then rescales once more after its scale adjust (ModRaise's
            // non-prescaled path) before it drops to the composite bottom: both must stay on the chain
            let unit = sh.unit();
            let room = sh.cfg.bts.input_room.unwrap_or(sh.cfg.chain.n_q.saturating_sub(2 * unit));
            let eff = ins[0].effective_level(unit);
            if eff > room {
                return Err(format!(
                    "[bts_input_error] refresh of {} enters at ({}, {}): effective level {eff} > {room}, the raise's rescales run off the {}-prime chain",
                    site.var, ins[0].level, ins[0].deg, sh.cfg.chain.n_q
                ));
            }
            let r = sh.bootstrap(&ins[0], &site.route).map_err(e)?;
            // the route table lists realized landings; FIDESlib exits degree 2 one CKKS level above it
            let lv = r.out.level.checked_sub(sh.unit()).ok_or_else(|| format!("landing {} has no raw state", r.out.level))?;
            let raw = CtMeta { level: lv, deg: 2, ..r.out };
            let out = match site.kind {
                SiteKind::Deliberate => {
                    let via = if site.landing.deg == 2 { raw } else { r.out };
                    if (via.level, via.deg) != (site.landing.level, site.landing.deg) {
                        return Err(format!(
                            "deliberate {} captured at ({}, {}) but route {:?} from ({}, {}) lands at ({}, {})",
                            site.var, site.landing.level, site.landing.deg, site.route, ins[0].level, ins[0].deg, via.level, via.deg
                        ));
                    }
                    via
                }
                _ => raw,
            };
            return Ok((vec![], vec![out]));
        }
        Kind::Finish { landing, outs, .. } => {
            if ins.is_empty() {
                return Err("finish without inputs".into());
            }
            // outs count is the caller's: one landing each (all at `landing` unless listed per output)
            if !outs.is_empty() {
                return Ok((vec![], outs.iter().map(|l| CtMeta::new(l.level, l.deg)).collect()));
            }
            return Ok((vec![], vec![CtMeta::new(landing.level, landing.deg)]));
        }
        Kind::Batched(_) => return Err("rule() of a batch: run it per member".into()),
        Kind::CoeffSum { pts: ids, .. } => {
            // the loop it stands for: mult_pt of each term, add into the first; no prelude anywhere (the fusion's
            // condition)
            if ins.is_empty() || ins.len() != ids.len() {
                return Err(format!("coeff_sum of {} cipher inputs and {} plaintexts", ins.len(), ids.len()));
            }
            let mut acc: Option<CtMeta> = None;
            for (k, (m, id)) in ins.iter().zip(ids).enumerate() {
                let r = sh.mult_pt(m, &pt(*id)?).map_err(e)?;
                if !r.pre.is_empty() {
                    return Err(format!("coeff_sum term {k}: the product needs preludes {:?}", r.pre));
                }
                acc = Some(match acc {
                    None => r.out,
                    Some(a) => {
                        let s = sh.add(&a, &r.out).map_err(e)?;
                        if !s.pre.is_empty() {
                            return Err(format!("coeff_sum term {k}: the sum needs preludes {:?}", s.pre));
                        }
                        s.out
                    }
                });
            }
            return Ok((vec![], acc.into_iter().collect()));
        }
    };
    Ok((r.pre, vec![r.out]))
}

/// Level and degree are what the program binds (nf is 0 without scale tables, the tag is advisory).
fn same(a: &CtMeta, b: &CtMeta) -> bool {
    a.level == b.level && a.deg == b.deg
}

/// The plan pin of one name: level one-sided (richer is fine), degree exact.
pub fn pin_errors(name: &str, m: &CtMeta, (el, ed): (u32, u8), rep: &mut CheckReport) {
    rep.pinned += 1;
    if m.level > el {
        rep.errors.push(format!("[plan_level_error] {name} expected {el} actual {}", m.level));
    } else if m.level == el {
        rep.level_exact += 1;
    } else {
        rep.level_richer += 1;
    }
    if m.deg != ed {
        rep.errors.push(format!("[plan_deg_error] {name} expected deg {ed} actual {}", m.deg));
    }
}

/// Re-run the Shadow over `p` from its inputs. Returns the report (errors listed, none = pass).
pub fn verify(sh: &Shadow, p: &Program) -> CheckReport {
    let mut rep = CheckReport {
        pt_relevels: p.check.pt_relevels,
        inexact_scalars: p.check.inexact_scalars,
        ignored_anchors: p.check.ignored_anchors.clone(),
        ..Default::default()
    };
    let mut state: Vec<Option<CtMeta>> = vec![None; p.vars.len()];
    for &v in &p.inputs {
        state[v as usize] = Some(p.vars[v as usize].meta);
    }
    let limit = sh.cfg.bts.level_limit;
    for (oi, op) in p.ops.iter().enumerate() {
        for m in op.members() {
            let mut ins = vec![];
            for &v in &m.ins {
                match state.get(v as usize).copied().flatten() {
                    Some(x) => ins.push(x),
                    None => {
                        rep.errors.push(format!("op {oi} {} reads {} before it is produced", m.kind.name(), p.vars[v as usize].name));
                    }
                }
            }
            if ins.len() != m.ins.len() {
                continue;
            }
            match rule(sh, &m.kind, &ins, &p.pt_refs, &p.bts_sites) {
                Err(msg) => rep.errors.push(format!("op {oi} {} -> {:?}: {msg}", m.kind.name(), m.outs.iter().map(|&o| &p.vars[o as usize].name).collect::<Vec<_>>())),
                Ok((pre, outs)) => {
                    if pre != m.pre {
                        rep.errors.push(format!("op {oi} {}: recorded preludes {:?} != {:?}", m.kind.name(), m.pre, pre));
                    }
                    for (k, &o) in m.outs.iter().enumerate() {
                        let got = outs.get(k).or(outs.first()).copied();
                        let var = &p.vars[o as usize];
                        match got {
                            Some(g) if same(&g, &var.meta) => {}
                            Some(g) => rep.errors.push(format!(
                                "op {oi} {}: {} recorded ({}, {}) recomputed ({}, {})",
                                m.kind.name(),
                                var.name,
                                var.meta.level,
                                var.meta.deg,
                                g.level,
                                g.deg
                            )),
                            None => rep.errors.push(format!("op {oi} {}: no output meta", m.kind.name())),
                        }
                        if state[o as usize].is_some() {
                            rep.errors.push(format!("op {oi}: {} produced twice", var.name));
                        }
                        state[o as usize] = Some(var.meta);
                        // the reactive ceiling: states before a planned refresh / realize are not checked (the C++
                        // checks after its op-exit hook)
                        let m = var.meta;
                        if !var.name.contains('~') && m.level >= limit && !p.pins.get(&var.name).is_some_and(|&(el, _)| m.level <= el) {
                            rep.errors.push(format!(
                                "[plan_bts_error] {} at level {} reaches the reactive ceiling {limit} unplanned",
                                var.name, m.level
                            ));
                        }
                    }
                    // a fused sum's intermediates keep their lowered states (pins and the ceiling read them)
                    let fz: &[VarId] = match &m.kind {
                        Kind::CoeffSum { fused, .. } | Kind::MultRescale { fused, .. } => fused,
                        _ => &[],
                    };
                    {
                        for &f in fz {
                            let var = &p.vars[f as usize];
                            if state[f as usize].is_some() {
                                rep.errors.push(format!("op {oi}: {} produced twice", var.name));
                            }
                            state[f as usize] = Some(var.meta);
                            let m = var.meta;
                            // a fused product's `~raw` state is before its realize, unchecked as above
                            if !var.name.contains('~') && m.level >= limit && !p.pins.get(&var.name).is_some_and(|&(el, _)| m.level <= el) {
                                rep.errors.push(format!("[plan_bts_error] {} at level {} reaches the reactive ceiling {limit} unplanned", var.name, m.level));
                            }
                        }
                    }
                }
            }
        }
    }
    for (name, &(el, ed)) in &p.pins {
        match p.names.get(name) {
            Some(&v) => match state[v as usize] {
                Some(m) => {
                    pin_errors(name, &m, (el, ed), &mut rep);
                }
                None => rep.errors.push(format!("pinned {name} is never produced")),
            },
            None => rep.errors.push(format!("pinned {name} is not a name of this block")),
        }
    }
    rep
}
