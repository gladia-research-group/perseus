//! Sparse-slot bootstrap routes (s < N/2) behind the `BtsRoute` seam, and the bounded route set.
//!
//! * `SparseBRoute`: s in {512, 256} (any s with a SparseB geometry): raise -> fold -> SparseB CtS -> EvalMod
//!   (K = 24) -> SparseB StC -> x 2^correction (`sparse_b.rs` holds the precomputation), staged 1:1 with FIDESlib
//!   (`btsBegin .. btsFinish` with `sparseB` set), K-batched.
//! * `SpruRoute`: s = 1 by SPRU (`spru.rs`); not staged (no raise, no CtS / StC).
//! * `RouteSet`: the routes a session holds. Every route is declared in the Config up front (`cfg.bts.sparse_routes`)
//!   and sized into the MemPlan before `allocate`; `add_route` of an undeclared slot count is an error naming the
//!   knob; `drop_route` takes a route out of dispatch (the arena is frozen: nothing is freed or grown).
use super::raise::{self, EncapsKeys};
use super::sparse_b::{AccRounds, SparseBPrecomp};
use super::spru::Spru;
use super::{common_meta, no_stage, run_staged, BtsCx, BtsRoute, BtsState, BtsWorkShape, MemReq, RouteKind, Staged, StcHigh};
use crate::context::{DevStrategy, DevWord};
use crate::ct::Ct;
use crate::error::{shape, Error, Result};
use crate::ks::{KskSource, View};
use crate::rotate::{self, KsCfg};
use pc_shadow::{tag, CtMeta, PtMeta, Route, Shadow};
use std::cell::RefCell;
use std::collections::BTreeMap;

/// deg = round(log2(q0 / sf(L))) of the chain (FIDESlib `btsBegin`; 2 on n32).
pub fn chain_deg<W: DevWord, M: DevStrategy<W>>(ctx: &crate::Context<'_, W, M>) -> Result<u32> {
    let d = ctx.chain.level_unit;
    let mut q = 0f64;
    for j in 0..d {
        q += (ctx.chain.q[j] as f64).log2();
    }
    Ok((q - ctx.scale.sf((ctx.lay.nq - 1) as i32)?.log2()).round().max(0.0) as u32)
}

fn total_slots(meta: &CtMeta, ring_n: usize) -> i64 {
    if meta.tag.slots > 0 {
        meta.tag.slots
    } else if meta.slots > 0 {
        meta.slots as i64
    } else {
        (ring_n / 2) as i64
    }
}

/// One SparseB route: precomputation variants by raise drop, the encapsulation keys, the integer post-factor table.
pub struct SparseBRoute<'a, W: DevWord> {
    pub slots: u32,
    pub ring_n: usize,
    pub variants: Vec<SparseBPrecomp<'a, W>>,
    pub encaps: EncapsKeys<'a>,
    pub pow2: super::finish::Pow2<'a, W>,
    pub cf_default: u32,
    pub cf_window: (u32, u32),
    pub deg: u32,
    pub ks_source: KskSource,
}

impl<'a, W: DevWord> SparseBRoute<'a, W> {
    fn variant(&self, drop: u32) -> Result<usize> {
        self.variants.iter().position(|v| v.cfg.raise_drop == drop).ok_or_else(|| Error::Mem {
            knob: "bts.raise_drops",
            value: drop as usize,
            msg: format!("route {} has no raise-drop {drop} variant (built: {:?})", self.slots, self.variants.iter().map(|v| v.cfg.raise_drop).collect::<Vec<_>>()),
        })
    }

    fn acc<M: DevStrategy<W>>(&self, cx: &mut BtsCx<'_, 'a, W, M>, rounds: &AccRounds<'_>, x: &mut Ct<'a, W>) -> Result<()> {
        let kc = KsCfg { plans: cx.ks, source: self.ks_source };
        for (steps, g) in &rounds.rounds {
            acc_round(cx.ctx, cx.ex, kc, cx.sh, g, x, steps)?;
        }
        Ok(())
    }

    fn realize<M: DevStrategy<W>>(cx: &mut BtsCx<'_, 'a, W, M>, x: &mut Ct<'a, W>) -> Result<()> {
        if x.meta.deg == 2 {
            cx.ops.rescale(cx.ctx, cx.ex, cx.sh, std::slice::from_mut(x))?;
        }
        Ok(())
    }

    fn lt<M: DevStrategy<W>>(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &super::cts::LtStage<'_, W>, x: &mut Ct<'a, W>) -> Result<()> {
        let k = x.k();
        if x.top(cx.ctx.shape)? != st.top {
            return Err(shape("sparse lt", format!("input top {} vs the stage's {}", x.top(cx.ctx.shape)?, st.top)));
        }
        st.run(cx, &mut x.buf, k, self.ks_source)?;
        let pm = PtMeta { level: cx.ctx.shape.n_q - 1 - st.top as u32, deg: 1, nf: st.pt_nf, tag: Default::default() };
        x.meta = cx.sh.mult_pt(&x.meta, &pm)?.out;
        Ok(())
    }
}

impl<'a, W: DevWord, M: DevStrategy<W>> Staged<'a, W, M> for SparseBRoute<'a, W> {
    fn begin(&self, _cx: &BtsCx<'_, 'a, W, M>, cts: &[Ct<'a, W>], route: &Route, prescaled: bool) -> Result<BtsState> {
        let in_meta = common_meta(cts)?;
        if route.slots != self.slots {
            return Err(shape("sparse route", format!("route {route:?} on the {}-slot route", self.slots)));
        }
        let variant = self.variant(route.raise_drop)?;
        let pre = &self.variants[variant];
        if let Some(x) = cts.iter().find(|x| x.k() > pre.cfg.k_max) {
            return Err(shape("sparse route", format!("batch of {} cts above k_max {}", x.k(), pre.cfg.k_max)));
        }
        let cf = route.cf.unwrap_or(self.cf_default);
        if cf < self.cf_window.0 || cf > self.cf_window.1 || cf < self.deg {
            return Err(shape("bootstrap cf", format!("correction factor {cf} outside [{}, {}] (deg {})", self.cf_window.0, self.cf_window.1, self.deg)));
        }
        let correction = cf - self.deg;
        Ok(BtsState {
            route: *route,
            slots: self.slots,
            old_slots: in_meta.slots,
            prescaled,
            cf,
            correction,
            cor_factor: 1u64 << correction,
            scalar: 1,
            const_eval_mult: pre.cts0_const,
            baked: pre.cts0_const,
            exact_const: true,
            shifted_flow: false,
            sparse_encaps: true,
            sparse_b: true,
            stc_first: false,
            stc_folded: false,
            real: false,
            n_cts: 1,
            n_stc: 1,
            variant,
            raise_drop: route.raise_drop,
            in_meta,
        })
    }
    fn stc_first_input(&self, _cx: &mut BtsCx<'_, 'a, W, M>, _st: &BtsState, _cts: &mut [Ct<'a, W>]) -> Result<()> {
        Ok(())
    }
    fn mod_raise(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &mut BtsState, cts: &mut [Ct<'a, W>]) -> Result<()> {
        let pre = &self.variants[st.variant];
        raise::raise(cx, &pre.raise, Some(&self.encaps), self.ks_source, cts, if st.prescaled { None } else { Some(st.correction) })?;
        st.shifted_flow = true;
        Ok(())
    }
    /// `btsFold`: Accumulate(4, s, N/2 / s), the class sums over the N/2s copies.
    fn fold(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>]) -> Result<()> {
        let pre = &self.variants[st.variant];
        for x in cts.iter_mut() {
            self.acc(cx, &pre.fold, x)?;
            Self::realize(cx, x)?;
        }
        Ok(())
    }
    /// `SparseBCoeffsToSlots` (its closing conj-add is the real EvalMod's input step).
    fn cts_stage(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>], k: usize) -> Result<()> {
        if k != 0 {
            return Err(no_stage("sparse_b", &format!("cts_stage({k})")));
        }
        let pre = &self.variants[st.variant];
        for x in cts.iter_mut() {
            Self::realize(cx, x)?;
            self.lt(cx, &pre.cts, x)?;
            self.acc(cx, &pre.cts_acc, x)?;
        }
        Ok(())
    }
    /// x + conj x, the K = 24 series and the double angles (`approxModReductionSparse`), then the rescale.
    fn eval_mod(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>]) -> Result<()> {
        let em = &self.variants[st.variant].em;
        for x in cts.iter_mut() {
            let top = x.top(cx.ctx.shape)? as i32;
            if top != em.in_meta.top || x.meta.deg as i32 != em.in_meta.nl {
                return Err(shape("sparse eval_mod", format!("input top {top} deg {} vs the plan's {:?}", x.meta.deg, em.in_meta)));
            }
            let k = x.k();
            em.run(cx, &mut x.buf, k)?;
            x.meta = CtMeta { level: cx.ctx.shape.n_q - 1 - em.out_meta.top as u32, deg: em.out_meta.nl as u8, nf: em.out_meta.nf, ..x.meta };
            Self::realize(cx, x)?;
        }
        Ok(())
    }
    fn stc_first_output(&self, _cx: &mut BtsCx<'_, 'a, W, M>, _st: &BtsState, _cts: &mut [Ct<'a, W>]) -> Result<()> {
        Err(no_stage("sparse_b", "stc_first_output"))
    }
    fn stc_enter(&self, _cx: &mut BtsCx<'_, 'a, W, M>, _st: &BtsState, _cts: &mut [Ct<'a, W>]) -> Result<()> {
        Ok(())
    }
    /// `SparseBSlotsToCoeffs`: BSGS over Q, Accumulate(4, n, r/2), + rotate(n/2).
    fn stc_stage(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>], k: usize) -> Result<()> {
        if k != 0 {
            return Err(no_stage("sparse_b", &format!("stc_stage({k})")));
        }
        let pre = &self.variants[st.variant];
        for x in cts.iter_mut() {
            Self::realize(cx, x)?;
            self.lt(cx, &pre.stc, x)?;
            self.acc(cx, &pre.stc_acc, x)?;
            self.acc(cx, &pre.half, x)?;
        }
        Ok(())
    }
    /// The integer correction 2^correction (SparseB's StC already closed with its rotate-add), slots restored.
    fn finish(&self, cx: &mut BtsCx<'_, 'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>]) -> Result<()> {
        for x in cts.iter_mut() {
            let top = x.top(cx.ctx.shape)?;
            if st.correction > 0 {
                self.pow2.mult(cx.ctx, &cx.ex.compute, View::of(&x.buf, 0), x.k(), top, st.correction)?;
            }
            x.meta.tag = tag::t_bootstrap(&st.in_meta.tag, self.slots as i64, total_slots(&st.in_meta, self.ring_n));
            x.meta.slots = st.old_slots;
        }
        Ok(())
    }
    fn stc_high(&self, _cx: &mut BtsCx<'_, 'a, W, M>, _cts: &mut [Ct<'a, W>], _spec: &StcHigh) -> Result<()> {
        Err(no_stage("sparse_b", "stc_high"))
    }
}

impl<'a, W: DevWord, M: DevStrategy<W>> BtsRoute<'a, W, M> for SparseBRoute<'a, W> {
    fn kind(&self) -> RouteKind {
        RouteKind::Sparse(self.slots)
    }
    /// Raw output: the StC stage's top (the EvalMod output rescaled), degree 2, scale = EvalMod's x StC's plaintexts.
    fn landing(&self, sh: &Shadow, meta: &CtMeta, route: &Route) -> Result<CtMeta> {
        let pre = &self.variants[self.variant(route.raise_drop)?];
        let nq = sh.cfg.chain.n_q;
        let em = pre.em.out_meta;
        let mut m = CtMeta { level: nq - 1 - em.top as u32, deg: em.nl as u8, nf: em.nf, slots: meta.slots, tag: meta.tag };
        if m.deg == 2 {
            m = sh.rescale(&m)?.out;
        }
        let pm = PtMeta { level: nq - 1 - pre.stc.top as u32, deg: 1, nf: pre.stc.pt_nf, tag: Default::default() };
        let mut out = sh.mult_pt(&m, &pm)?.out;
        out.slots = meta.slots;
        out.tag = tag::t_bootstrap(&meta.tag, self.slots as i64, total_slots(meta, self.ring_n));
        Ok(out)
    }
    fn run(&self, cx: &mut BtsCx<'_, 'a, W, M>, cts: &mut [Ct<'a, W>], route: &Route) -> Result<()> {
        for _ in 0..route.iterations.max(1) {
            run_staged(self, cx, cts, route, false)?;
        }
        Ok(())
    }
    fn staged(&self) -> Option<&dyn Staged<'a, W, M>> {
        Some(self)
    }
    fn mem(&self) -> (MemReq, BtsWorkShape) {
        let mut r = MemReq::default();
        for v in &self.variants {
            r.add("bts.sparse", v.carved);
        }
        let shape = self.variants.iter().map(|v| v.work).fold(BtsWorkShape { k_max: 1, g_max: 1, em_slots: 0, em_kk: 1, spare: 0 }, |a, b| a.max(b));
        (r, shape)
    }
}

/// SPRU behind the seam (its work buffers are route-owned, hence the cell: one call at a time per route).
pub struct SpruRoute<'a, W: DevWord> {
    pub spru: RefCell<Spru<'a, W>>,
    pub ring_n: usize,
    pub cf_default: u32,
    pub deg: u32,
    pub carved: usize,
}

impl<'a, W: DevWord, M: DevStrategy<W>> BtsRoute<'a, W, M> for SpruRoute<'a, W> {
    fn kind(&self) -> RouteKind {
        RouteKind::Spru { slots: 1, h: self.spru.borrow().cfg.h as u32 }
    }
    /// Raw output: depth_primes below the variant's top, degree 2 (the recombination's two plaintext products, no
    /// rescale).
    fn landing(&self, sh: &Shadow, meta: &CtMeta, route: &Route) -> Result<CtMeta> {
        let level = self.spru.borrow().out_level(sh.cfg.chain.unit, route.raise_drop);
        let nq = sh.cfg.chain.n_q;
        let tables = sh.tables.as_deref().ok_or_else(|| shape("spru landing", "Shadow without scale tables"))?;
        let nf = tables.sf((nq - 1 - level) as i32)?.powi(2);
        Ok(CtMeta { level, deg: 2, nf, slots: meta.slots, tag: tag::t_bootstrap(&meta.tag, 1, total_slots(meta, self.ring_n)) })
    }
    fn run(&self, cx: &mut BtsCx<'_, 'a, W, M>, cts: &mut [Ct<'a, W>], route: &Route) -> Result<()> {
        let cf = route.cf.unwrap_or(self.cf_default);
        if cf < self.deg {
            return Err(shape("bootstrap cf", format!("correction factor {cf} below the chain deg {}", self.deg)));
        }
        let mut s = self.spru.borrow_mut();
        for _ in 0..route.iterations.max(1) {
            for x in cts.iter_mut() {
                s.run(cx, x, cf - self.deg, route.raise_drop)?;
            }
        }
        Ok(())
    }
    fn staged(&self) -> Option<&dyn Staged<'a, W, M>> {
        None
    }
    fn mem(&self) -> (MemReq, BtsWorkShape) {
        let mut r = MemReq::default();
        r.add("bts.spru", self.carved);
        (r, BtsWorkShape { k_max: 1, g_max: 1, em_slots: 0, em_kk: 1, spare: 0 })
    }
}

/// The sparse routes of a session, by slot count: declared in the Config (`bts.sparse_routes`), built before
/// `Arena::freeze`; `active` is what dispatch sees.
pub struct RouteSet<'r, 'a, W: DevWord, M: DevStrategy<W>> {
    pub declared: Vec<u32>,
    routes: BTreeMap<u32, &'r dyn BtsRoute<'a, W, M>>,
    active: BTreeMap<u32, bool>,
}

impl<'r, 'a, W: DevWord, M: DevStrategy<W>> RouteSet<'r, 'a, W, M> {
    pub fn new(declared: &[u32]) -> Self {
        RouteSet { declared: declared.to_vec(), routes: BTreeMap::new(), active: BTreeMap::new() }
    }
    /// Register a built route for `slots` (it must be declared: routes never grow lazily).
    pub fn add_route(&mut self, slots: u32, route: &'r dyn BtsRoute<'a, W, M>) -> Result<()> {
        if !self.declared.contains(&slots) {
            return Err(Error::Mem {
                knob: "bts.sparse_routes",
                value: slots as usize,
                msg: format!("route {slots} is not declared (declared {:?}): add it to the Config so the MemPlan sizes it", self.declared),
            });
        }
        self.routes.insert(slots, route);
        self.active.insert(slots, true);
        Ok(())
    }
    /// Take a route out of dispatch (its memory stays carved); returns whether it was active.
    pub fn drop_route(&mut self, slots: u32) -> bool {
        self.active.insert(slots, false).unwrap_or(false)
    }
    /// Re-activate a dropped route (declared and built).
    pub fn readd_route(&mut self, slots: u32) -> Result<()> {
        if !self.routes.contains_key(&slots) {
            return Err(Error::Mem { knob: "bts.sparse_routes", value: slots as usize, msg: format!("route {slots} was never built") });
        }
        self.active.insert(slots, true);
        Ok(())
    }
    /// Active slot counts, ascending (what `fold_slots_for` and the auto routing choose from).
    pub fn built(&self) -> Vec<u32> {
        self.active.iter().filter(|(_, &a)| a).map(|(&s, _)| s).collect()
    }
    pub fn get(&self, slots: u32) -> Result<&'r dyn BtsRoute<'a, W, M>> {
        match (self.routes.get(&slots), self.active.get(&slots)) {
            (Some(r), Some(true)) => Ok(*r),
            _ => Err(Error::Mem { knob: "bts.sparse_routes", value: slots as usize, msg: format!("no active route for {slots} slots (active {:?})", self.built()) }),
        }
    }
    /// The MemPlan request of every registered route.
    pub fn mem(&self) -> (MemReq, BtsWorkShape) {
        let mut r = MemReq::default();
        let mut w = BtsWorkShape { k_max: 1, g_max: 1, em_slots: 0, em_kk: 1, spare: 0 };
        for route in self.routes.values() {
            let (m, s) = route.mem();
            r.merge(&m);
            w = w.max(s);
        }
        (r, w)
    }
}

/// One Accumulate round, `x += sum_j rot(x, steps[j])` (FIDESlib `Accumulate`: hoisted rotations summed in the
/// extended basis, one ModDown fused with the add; = `rotate::rotate_sum_inplace`'s arithmetic, bit-identical). When
/// every rotation of the batch fits the extended workspace the sums happen in place in the dot's output slots (no
/// copy into the conversion buffer); otherwise `rotate_sum_inplace`.
pub fn acc_round<W: DevWord, M: DevStrategy<W>>(
    ctx: &crate::Context<'_, W, M>,
    ex: &mut crate::Executor<'_, W>,
    kc: KsCfg<'_, '_>,
    sh: &Shadow,
    g: &crate::ks::KeyGroup<'_>,
    x: &mut Ct<'_, W>,
    steps: &[i64],
) -> Result<()> {
    use crate::ks::{self, DotIo, DotRows};
    let (k, m) = (x.k(), steps.len());
    let in_place = g.reads_in_place(kc.source);
    let staged = !in_place && (kc.source != KskSource::Cached || g.needs_stage() || !g.has_a());
    if m == 0 || staged || k * m > ex.ks.ext.k || g.len() != m {
        return rotate::rotate_sum_inplace(ctx, ex, kc, sh, g, x, steps);
    }
    let lay = ctx.lay;
    let top = x.top(ctx.shape)?;
    let cw = lay.ct_words() as u64;
    ks::modup(ctx, &ex.compute, kc.plans, &mut ex.ks, &mut ex.bconv_work, &x.buf, 0, k, 1, top)?;
    let va = View::of(&x.buf, 0);
    let ext0 = View::of(&ex.ks.ext, 0);
    let io = DotIo { c1: va.part::<W>(1).base, c0: va.base, ct_stride: va.ct_stride, out: ext0.base, part_stride: lay.poly_words() as u64, out_ct_stride: cw, out_rot_stride: k as u64 * cw, c0_mode: 1 };
    ks::dot(ctx, &ex.compute, kc.plans, &ex.ks, g, 0, m, k, top, DotRows::All, io, in_place)?;
    for j in 1..m {
        ks::ew(ctx, &ex.compute, ks::Ew::Add, ext0, ext0, Some(View::of(&ex.ks.ext, j * k)), lay.rs_qp(top), k, 2)?;
    }
    ks::moddown_conv(ctx, &ex.compute, kc.plans, &mut ex.ks.ext, &mut ex.ks.conv, k, top)?;
    ks::md_epilogue(ctx, &ex.compute, View::of(&x.buf, 0), View::of(&ex.ks.ext, 0), View::of(&ex.ks.conv, 0), k, top)?;
    x.meta = sh.rotate_sum(&x.meta, steps).out;
    Ok(())
}
