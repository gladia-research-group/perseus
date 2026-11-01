//! `Shadow`: one rule per ciphertext op, returning a `Recipe` (what to do to the operands first, what comes out,
//! which constants the op needs). The rules are FIDESlib's FLEXIBLEAUTO behaviour (`fides`), as executed through the
//! perseus wrapper (`fideslib_wrapper.h`) and FIDESlib's API layer (`api/CryptoContext.cpp`); `ShadowConfig::quirks`
//! switches individual rules to the planner's or a fake's deviation (see `quirks`).
//!
//! Operand convention of a binary recipe: preludes on `Operand::A` modify the first operand (the op's output is
//! built from it); preludes on `Operand::B` apply to a COPY of the second operand (FIDESlib copies `b` before it
//! adjusts it). Preludes run in order.
use crate::error::{rule, Result};
use crate::fides::{FMeta, Fx, Step};
use crate::meta::{effective_level, level_from_top_limb, top_limb, ChainShape, CtMeta};
use crate::quirks::{Quirk, Quirks};
use crate::route::{BtsConfig, Landing, Route};
use crate::tag::{self, Tag};
use pc_core::scale::ScaleTables;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operand {
    A,
    B,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Prelude {
    /// Pay the pending rescale (drops `unit` primes).
    Rescale(Operand),
    /// Drop limbs down to this level (primes dropped; no arithmetic).
    DropTo(Operand, u32),
    /// Multiply by this scalar encoded at the operand's current level (`multScalarNoPrecheck`: degree + 1).
    ScalarAdjust(Operand, f64),
}

/// Integer constant rows an op needs (residues over q_0..q_top), from `pc_core::scale`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Consts {
    /// `ElemForEvalMult(top, c)`.
    Mult(Vec<u64>),
    /// `ElemForEvalAddOrSub(top, |c|, deg)`, negated mod q_i for c < 0 (FIDESlib keeps `q - 0 = q` unreduced).
    Add(Vec<u64>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Recipe {
    pub pre: Vec<Prelude>,
    pub out: CtMeta,
    pub consts: Option<Consts>,
}

/// A plaintext operand: its encoding level, degree and scale.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PtMeta {
    pub level: u32,
    pub deg: u8,
    pub nf: f64,
    #[serde(default)]
    pub tag: Tag,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShadowConfig {
    pub chain: ChainShape,
    pub bts: BtsConfig,
    #[serde(default)]
    pub quirks: Quirks,
}

/// The rules. `tables` = None runs levels and degrees only (`nf` stays 0, no constants).
#[derive(Clone, Debug)]
pub struct Shadow {
    pub cfg: ShadowConfig,
    pub tables: Option<Arc<ScaleTables>>,
}

/// Outcome of a `bootstrap_hint`.
#[derive(Clone, Debug, PartialEq)]
pub struct HintRecipe {
    pub fired: bool,
    pub recipe: Recipe,
}

impl Shadow {
    pub fn new(mut cfg: ShadowConfig, tables: Option<Arc<ScaleTables>>) -> Shadow {
        // an alternating user region: with scale tables its levels map to its limbs, without them the rules run on
        // the virtual grid as a uniform chain
        match &tables {
            Some(st) if st.singles != 0 => cfg.chain = ChainShape::from_tables(st),
            None if cfg.chain.singles != 0 => cfg.chain = cfg.chain.grid(),
            _ => {}
        }
        Shadow { cfg, tables }
    }
    pub fn unit(&self) -> u32 {
        self.cfg.chain.unit
    }
    pub fn has(&self, q: Quirk) -> bool {
        self.cfg.quirks.has(q)
    }
    pub(crate) fn fx(&self) -> Fx<'_> {
        Fx { d: self.cfg.chain.unit as i32, st: self.tables.as_deref() }
    }
    fn to_f(&self, m: &CtMeta) -> Result<FMeta> {
        Ok(FMeta { top: top_limb(self.cfg.chain, m.level)? as i32, nl: m.deg as i32, nf: m.nf })
    }
    fn from_f(&self, f: &FMeta, like: &CtMeta, tag: Tag) -> Result<CtMeta> {
        if f.nl < 1 || f.nl > 2 {
            return Err(rule("from_f", format!("degree {} out of range", f.nl)));
        }
        Ok(CtMeta { level: level_from_top_limb(self.cfg.chain, f.top as u32)?, deg: f.nl as u8, nf: f.nf, slots: like.slots, tag })
    }
    fn pre(&self, who: Operand, steps: &[Step]) -> Result<Vec<Prelude>> {
        steps
            .iter()
            .map(|s| match *s {
                Step::Rescale => Ok(Prelude::Rescale(who)),
                Step::DropTo(t) => Ok(Prelude::DropTo(who, level_from_top_limb(self.cfg.chain, t as u32)?)),
                Step::Scalar(c) => Ok(Prelude::ScalarAdjust(who, c)),
            })
            .collect()
    }
    fn mult_consts(&self, top: i32, c: f64) -> Result<Option<Consts>> {
        match &self.tables {
            Some(st) => Ok(Some(Consts::Mult(st.elem_for_eval_mult(top, c, -1)?))),
            None => Ok(None),
        }
    }

    // ── additive ──────────────────────────────────────────────────────────────────────────────────────────────────

    /// ct + ct (`Ciphertext::add`: adjust a to b, else a copy of b to a, then add; the sum keeps a's scale).
    pub fn add(&self, a: &CtMeta, b: &CtMeta) -> Result<Recipe> {
        self.add_like(a, b, false)
    }
    /// ct - ct.
    pub fn sub(&self, a: &CtMeta, b: &CtMeta) -> Result<Recipe> {
        self.add_like(a, b, true)
    }
    fn add_like(&self, a: &CtMeta, b: &CtMeta, sub: bool) -> Result<Recipe> {
        let tag = tag::t_add(&a.tag, &b.tag);
        if self.has(Quirk::FakeAddMaxDeg) {
            let out = CtMeta { level: a.level.max(b.level), deg: a.deg.max(b.deg), nf: a.nf, slots: a.slots, tag };
            return Ok(Recipe { pre: vec![], out, consts: None });
        }
        let fx = self.fx();
        let (mut fa, mut fb) = (self.to_f(a)?, self.to_f(b)?);
        let mut pre = Vec::new();
        // Ciphertext::add recursion: at most one swap is ever needed (the copy of b always adjusts to a)
        for _ in 0..3 {
            let mut sa = Vec::new();
            if fx.adjust_add(&mut fa, &fb, &mut sa)? {
                pre.extend(self.pre(Operand::A, &sa)?);
                if fa.top != fb.top || fa.nl != fb.nl {
                    return Err(rule(if sub { "sub" } else { "add" }, format!("adjust left {fa:?} vs {fb:?}")));
                }
                let out = self.from_f(&fa, a, tag)?;
                return Ok(Recipe { pre, out, consts: None });
            }
            let mut sb = Vec::new();
            if !fx.adjust_add(&mut fb, &fa, &mut sb)? {
                return Err(rule("add", format!("adjustForAddOrSub failed both ways: {fa:?} {fb:?}")));
            }
            pre.extend(self.pre(Operand::B, &sb)?);
        }
        Err(rule("add", "adjustForAddOrSub did not converge"))
    }

    /// ct + scalar (`addScalar`): no level change; the constant is encoded at the ciphertext's top and degree.
    pub fn add_scalar(&self, a: &CtMeta, c: f64) -> Result<Recipe> {
        let out = CtMeta { tag: tag::t_add_scalar(&a.tag, c), ..*a };
        let consts = match (&self.tables, c != 0.0) {
            (Some(st), true) => {
                let top = top_limb(self.cfg.chain, a.level)? as i32;
                let mut e = st.elem_for_eval_add(top, c.abs(), a.deg as i32)?;
                if c < 0.0 {
                    for (i, v) in e.iter_mut().enumerate() {
                        *v = st.q[i] - *v;
                    }
                }
                Some(Consts::Add(e))
            }
            _ => None,
        };
        Ok(Recipe { pre: vec![], out, consts })
    }

    /// ct + pt (`Ciphertext::addPt`).
    pub fn add_pt(&self, a: &CtMeta, pt: &PtMeta) -> Result<Recipe> {
        let tag = tag::t_add(&a.tag, &pt.tag);
        let u = self.unit();
        if self.has(Quirk::FakeAddPtFree) {
            return Ok(Recipe { pre: vec![], out: CtMeta { tag, ..*a }, consts: None });
        }
        if self.has(Quirk::AddPtRealizes) {
            // sim.py / fakecap: a pending rescale is always realized against a degree-1 plaintext
            let (level, deg, pre) = if a.deg == 2 && pt.deg == 1 {
                (a.effective_level(u).max(pt.level), 1, vec![Prelude::Rescale(Operand::A)])
            } else {
                (a.level.max(pt.level), a.deg, vec![])
            };
            return Ok(Recipe { pre, out: CtMeta { level, deg, tag, ..*a }, consts: None });
        }
        let fx = self.fx();
        let mut fa = self.to_f(a)?;
        let mut pre = Vec::new();
        for _ in 0..3 {
            let mut fp = FMeta { top: top_limb(self.cfg.chain, pt.level)? as i32, nl: pt.deg as i32, nf: pt.nf };
            let mut sa = Vec::new();
            if fp.nl == 1 && fa.nl == 2 && fp.top == fx.down(fa.top) {
                fx.rescale(&mut fa, &mut sa)?;
            }
            if fp.top != fa.top || fp.nl != fa.nl {
                let mut sp = Vec::new();
                if !fx.adjust_pt(&mut fp, &fa, &mut sp)? {
                    // the plaintext is deeper: drop the ciphertext to it and retry
                    fx.drop_to(&mut fa, fp.top, &mut sa);
                    pre.extend(self.pre(Operand::A, &sa)?);
                    continue;
                }
                pre.extend(self.pre(Operand::A, &sa)?);
                pre.extend(self.pre(Operand::B, &sp)?);
            } else {
                pre.extend(self.pre(Operand::A, &sa)?);
            }
            let out = self.from_f(&fa, a, tag)?;
            return Ok(Recipe { pre, out, consts: None });
        }
        Err(rule("add_pt", "addPt did not converge"))
    }

    // ── multiplicative ───────────────────────────────────────────────────────────────────────────────────────────

    /// ct * ct (`Ciphertext::mult`, relinearized, rescale pending): degree 2 at the deeper effective level.
    pub fn mult(&self, a: &CtMeta, b: &CtMeta) -> Result<Recipe> {
        let tag = tag::t_mult(&a.tag, &b.tag);
        if self.has(Quirk::FakeProductRescaled) {
            return Ok(self.fake_product(a.level.max(b.level), a, tag));
        }
        let fx = self.fx();
        let (mut fa, mut fb) = (self.to_f(a)?, self.to_f(b)?);
        let mut pre = Vec::new();
        for _ in 0..3 {
            let mut sa = Vec::new();
            let ok = fx.adjust_mult(&mut fa, &fb, &mut sa)?;
            pre.extend(self.pre(Operand::A, &sa)?);
            if ok {
                if fa.nl != 1 || fa.top != fb.top || fb.nl != 1 {
                    return Err(rule("mult", format!("adjustForMult left {fa:?} vs {fb:?}")));
                }
                let f = FMeta { top: fa.top, nl: fa.nl + fb.nl, nf: fa.nf * fb.nf };
                return Ok(Recipe { pre, out: self.from_f(&f, a, tag)?, consts: None });
            }
            let mut sb = Vec::new();
            if !fx.adjust_mult(&mut fb, &fa, &mut sb)? {
                return Err(rule("mult", format!("adjustForMult failed both ways: {fa:?} {fb:?}")));
            }
            pre.extend(self.pre(Operand::B, &sb)?);
        }
        Err(rule("mult", "adjustForMult did not converge"))
    }

    /// ct^2 (`Ciphertext::square`: rescale first, then the product with itself).
    pub fn square(&self, a: &CtMeta) -> Result<Recipe> {
        let tag = tag::t_square(&a.tag);
        if self.has(Quirk::FakeProductRescaled) {
            return Ok(self.fake_product(a.level, a, tag));
        }
        let fx = self.fx();
        let mut fa = self.to_f(a)?;
        let mut sa = Vec::new();
        if fa.nl == 2 {
            fx.rescale(&mut fa, &mut sa)?;
        }
        let f = FMeta { top: fa.top, nl: 2, nf: fa.nf * fa.nf };
        Ok(Recipe { pre: self.pre(Operand::A, &sa)?, out: self.from_f(&f, a, tag)?, consts: None })
    }

    /// ct * scalar (`multScalar`): degree 2 at the effective level; the constant is `ElemForEvalMult(top, c)`.
    pub fn mult_scalar(&self, a: &CtMeta, c: f64) -> Result<Recipe> {
        let tag = tag::t_mult_scalar(&a.tag, c);
        if self.has(Quirk::FakeProductRescaled) {
            return Ok(self.fake_product(a.level, a, tag));
        }
        self.scalar_rule(a, c, tag)
    }
    fn scalar_rule(&self, a: &CtMeta, c: f64, tag: Tag) -> Result<Recipe> {
        let fx = self.fx();
        let mut fa = self.to_f(a)?;
        let mut sa = Vec::new();
        if fa.nl == 2 {
            fx.rescale(&mut fa, &mut sa)?;
        }
        let consts = self.mult_consts(fa.top, c)?;
        let mut ignore = Vec::new();
        fx.scalar_noprecheck(&mut fa, c, &mut ignore)?;
        Ok(Recipe { pre: self.pre(Operand::A, &sa)?, out: self.from_f(&fa, a, tag)?, consts })
    }

    /// -ct. The GPU path of FIDESlib's API (`CryptoContextImpl::EvalNegate`) is `multScalar(-1.0)`: it costs the
    /// pending rescale and leaves degree 2 (the captured graphs record exactly that).
    pub fn negate(&self, a: &CtMeta) -> Result<Recipe> {
        let tag = tag::t_mult_scalar(&a.tag, -1.0);
        if self.has(Quirk::FakeNegateFree) {
            return Ok(Recipe { pre: vec![], out: CtMeta { tag, ..*a }, consts: None });
        }
        self.scalar_rule(a, -1.0, tag)
    }

    /// ct * pt (`Ciphertext::multPt`): rescale first, the plaintext adjusted to the ciphertext, degree 2.
    pub fn mult_pt(&self, a: &CtMeta, pt: &PtMeta) -> Result<Recipe> {
        let tag = tag::t_mult(&a.tag, &pt.tag);
        let u = self.unit();
        if self.has(Quirk::FakeProductRescaled) {
            return Ok(self.fake_product(a.level, a, tag));
        }
        if self.has(Quirk::CapMultPtMax) || self.has(Quirk::PtLevelIgnored) {
            let mut level = a.effective_level(u);
            if self.has(Quirk::CapMultPtMax) {
                level = level.max(pt.level);
            }
            let pre = if a.deg == 2 { vec![Prelude::Rescale(Operand::A)] } else { vec![] };
            let out = CtMeta { level, deg: 2, nf: 0.0, slots: a.slots, tag };
            return Ok(Recipe { pre, out, consts: None });
        }
        let fx = self.fx();
        let mut fa = self.to_f(a)?;
        let mut sa = Vec::new();
        if fa.nl == 2 {
            fx.rescale(&mut fa, &mut sa)?;
        }
        let mut fp = FMeta { top: top_limb(self.cfg.chain, pt.level)? as i32, nl: pt.deg as i32, nf: pt.nf };
        let mut sp = Vec::new();
        if fp.top != fa.top || fp.nl == 2 {
            if !fx.adjust_pt(&mut fp, &fa, &mut sp)? {
                return Err(rule(
                    "mult_pt",
                    format!("plaintext at level {} is deeper than its ciphertext (level {})", pt.level, a.effective_level(u)),
                ));
            }
            if fp.nl == 2 {
                fx.rescale(&mut fp, &mut sp)?;
            }
        }
        let mut pre = self.pre(Operand::A, &sa)?;
        pre.extend(self.pre(Operand::B, &sp)?);
        let f = FMeta { top: fa.top, nl: fa.nl + fp.nl, nf: fa.nf * fp.nf };
        Ok(Recipe { pre, out: self.from_f(&f, a, tag)?, consts: None })
    }

    fn fake_product(&self, level: u32, a: &CtMeta, tag: Tag) -> Recipe {
        // fake.py `_product`: the product lands rescaled one unit deeper
        let out = CtMeta { level: level + self.unit(), deg: 1, nf: 0.0, slots: a.slots, tag };
        Recipe { pre: vec![], out, consts: None }
    }

    // ── level / degree only ──────────────────────────────────────────────────────────────────────────────────────

    pub fn conjugate(&self, a: &CtMeta) -> Recipe {
        Recipe { pre: vec![], out: CtMeta { tag: tag::t_conjugate(&a.tag), ..*a }, consts: None }
    }
    /// ct * i via the monomial X^{N/2}: level and degree preserved.
    pub fn mult_i(&self, a: &CtMeta) -> Recipe {
        Recipe { pre: vec![], out: CtMeta { tag: tag::t_mult_i(&a.tag), ..*a }, consts: None }
    }
    pub fn rotate(&self, a: &CtMeta, k: i64) -> Recipe {
        Recipe { pre: vec![], out: CtMeta { tag: tag::t_rotate(&a.tag, k), ..*a }, consts: None }
    }
    /// a + sum_k rot(a, k) over one hoisted radix group (the tag as fakecap's `_c_rotate_sum` builds it).
    pub fn rotate_sum(&self, a: &CtMeta, steps: &[i64]) -> Recipe {
        let mut t = a.tag;
        let mut i = 0u32;
        while ((1usize << i) - 1) < steps.len() {
            t = tag::t_add(&t, &tag::t_rotate(&t, steps[(1usize << i) - 1]));
            i += 1;
        }
        Recipe { pre: vec![], out: CtMeta { tag: t, ..*a }, consts: None }
    }
    pub fn rescale(&self, a: &CtMeta) -> Result<Recipe> {
        let fx = self.fx();
        let mut fa = self.to_f(a)?;
        let mut s = Vec::new();
        fx.rescale(&mut fa, &mut s)?;
        Ok(Recipe { pre: vec![], out: self.from_f(&fa, a, a.tag)?, consts: None })
    }
    /// The wrapper's `drop_to_level(t)`: `max(cur, t)`, degree and scale kept.
    pub fn drop_to_level(&self, a: &CtMeta, level: u32) -> Recipe {
        Recipe { pre: vec![], out: CtMeta { level: a.level.max(level), ..*a }, consts: None }
    }
    /// Pay a pending rescale now (the wrapper's realize / plan `rescale_after` anchor).
    pub fn realize(&self, a: &CtMeta) -> Result<Recipe> {
        if a.deg == 2 {
            let mut r = self.rescale(a)?;
            r.pre = vec![Prelude::Rescale(Operand::A)];
            return Ok(r);
        }
        Ok(Recipe { pre: vec![], out: *a, consts: None })
    }
    /// `tag_reduce(stride)`: metadata-only assertion that the value is `stride`-periodic.
    pub fn tag_reduce(&self, a: &CtMeta, stride: i64) -> Recipe {
        let slots = if a.tag.slots > 0 { a.tag.slots } else { a.slots as i64 };
        Recipe { pre: vec![], out: CtMeta { tag: tag::t_reduce(&a.tag, slots, stride), ..*a }, consts: None }
    }
    /// The wrapper's `level_hint`: a recorded no-op carrying `lvl_cap(l)` for the planner.
    pub fn hint_level(&self, a: &CtMeta) -> Recipe {
        Recipe { pre: vec![], out: *a, consts: None }
    }

    // ── refreshes ────────────────────────────────────────────────────────────────────────────────────────────────

    /// A refresh on `route`. The landing is the route's (BtsConfig), deeper by the raise drop; the tag becomes the
    /// route's period (`t_bootstrap`).
    pub fn bootstrap(&self, a: &CtMeta, route: &Route) -> Result<Recipe> {
        let slots = if a.tag.slots > 0 { a.tag.slots } else { a.slots as i64 };
        let tag = tag::t_bootstrap(&a.tag, route.slots as i64, slots);
        let land = if self.has(Quirk::FakeBootstrapRealized) {
            *self.cfg.bts.landings.get(&0).ok_or_else(|| rule("bootstrap", "no dense landing configured"))?
        } else {
            self.cfg.bts.landing(self.cfg.chain, &self.cfg.quirks, route, a.level, a.deg)?
        };
        Ok(Recipe { pre: vec![], out: self.landed(a, land, tag)?, consts: None })
    }
    fn landed(&self, a: &CtMeta, land: Landing, tag: Tag) -> Result<CtMeta> {
        let nf = match &self.tables {
            Some(st) => st.sf(top_limb(self.cfg.chain, land.level)? as i32)?.powi(land.deg as i32),
            None => 0.0,
        };
        Ok(CtMeta { level: land.level, deg: land.deg, nf, slots: a.slots, tag })
    }

    /// `bootstrap_hint(thr, account_pending_rescale)`: refresh iff the (effective, when accounted) level is past the
    /// threshold. A fired hint lands like a planned refresh of `route` (realized).
    pub fn hint(&self, a: &CtMeta, threshold: u32, account_pending: bool, route: &Route) -> Result<HintRecipe> {
        let lv = if account_pending { a.effective_level(self.unit()) } else { a.level };
        if lv <= threshold {
            return Ok(HintRecipe { fired: false, recipe: Recipe { pre: vec![], out: *a, consts: None } });
        }
        Ok(HintRecipe { fired: true, recipe: self.bootstrap(a, route)? })
    }

    /// `fold_bootstrap(s, n_live, prescale)`: the s-slot route, then the recovery constant copies/n_live/prescale
    /// as one scalar product on the landing (one level, degree 2), the prescale (if any) one level on the input.
    pub fn fold_bootstrap(&self, a: &CtMeta, s: u32, n_live: u32, prescale: f64) -> Result<Recipe> {
        if n_live == 0 || !(prescale > 0.0) {
            return Err(rule("fold_bootstrap", format!("n_live {n_live} / prescale {prescale} must be > 0")));
        }
        let mut pre = vec![];
        let mut x = *a;
        if prescale != 1.0 {
            let r = self.mult_scalar(&x, prescale)?;
            pre.extend(r.pre);
            pre.push(Prelude::ScalarAdjust(Operand::A, prescale));
            x = r.out;
        }
        let route = Route::sparse(s);
        let landed = self.bootstrap(&x, &route)?.out;
        let total = if a.slots > 0 { a.slots as i64 } else { a.tag.slots };
        if s == 0 || total <= 0 || total % s as i64 != 0 {
            return Err(rule("fold_bootstrap", format!("fold to {s} slots of a {total}-slot ciphertext")));
        }
        let full = (total / s as i64) as f64;
        let recovery = full / n_live as f64 / prescale;
        let out = if recovery != 1.0 { self.scalar_rule(&landed, recovery, landed.tag)?.out } else { landed };
        Ok(Recipe { pre, out, consts: None })
    }

    /// The effective level a refresh of `a` starts at (what the depth envelope is measured in).
    pub fn refresh_input_level(&self, a: &CtMeta) -> u32 {
        effective_level(a.level, a.deg, self.unit())
    }
}
