//! Leaf ops: add / sub / negate / scalar and plaintext products / tensor (ct x ct) / mult_i / fused sums, each
//! executing the `pc_shadow::Recipe` of the op: the preludes first (`Rescale`, `DropTo`, `ScalarAdjust`; operand A in
//! place, operand B on a copy, as FIDESlib copies `b`), then the op's kernel, then `meta = recipe.out`. Kernels are
//! The element-wise kernel set on `Executor::compute`.
//!
//! K-batching: a `Ct` is a batch of `k` ciphertexts sharing one `CtMeta`, and every op runs ONE launch per batch
//! (the kernels take `K` and a ciphertext stride). The slice arguments (`&[&Ct]`, `&mut [Ct]`) are independent
//! batches, one launch each. A plaintext operand is broadcast over the `k` ciphertexts of its batch (stride 0).
//!
//! Per-call constants (`ElemForEvalMult` / `ElemForEvalAddOrSub` rows, wsum weights, folded alphas) go through the
//! `ConstRing`: pinned host slots copied to device slots on the compute lane, in stream order with the kernels that
//! read them (no host sync except when a pinned slot is reused before its copy ran).
/// ct x ct = tensor + relinearization; the Recipe preludes run in front of it.
pub use crate::relin::mult;

use crate::context::{Context, DevStrategy, DevWord};
use crate::error::{shape, Error, Result};
use crate::executor::Executor;
use crate::mem::MemPlan;
use crate::poly::{CtBuf, Layout};
use crate::rescale::{RescaleKind, Rescaler};
use crate::ct::Ct;
use pc_core::encode::Complex;
use pc_core::modmul::{addmod, mulmod, shoup, submod};
use pc_cuda::{launch_result, Arena, Compute, ConstBuf, DBuf, Device, Event, Lane, PinnedBuf, Stream};
use pc_kernels::{AddConstArgs, EwArgs, MulAddArgs, MulTableArgs, PolyRef, RowSet, TensorArgs, WsumArgs};
use pc_shadow::meta::top_limb;
use pc_shadow::{tag, CtMeta, Operand, Prelude, PtMeta, Recipe, Shadow};
use rayon::prelude::*;
use std::ffi::c_void;

/// Knobs of the op layer (the `exec` part of the typed Config).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpsCfg {
    /// Constant-ring slots (in flight per-call constant vectors before a pinned slot is reused).
    pub const_slots: usize,
    pub rescale: RescaleKind,
    /// Fold a `ScalarAdjust` into the following `Rescale` (composite one-pass only; exact).
    pub fold_scalar: bool,
    /// Out-of-place ops read their A operand where it is (no copy into the output first) and an A `Rescale` prelude
    /// rescales into the output (exact: the same words).
    pub out_of_place: bool,
}

impl Default for OpsCfg {
    fn default() -> Self {
        OpsCfg { const_slots: 256, rescale: RescaleKind::CompositeOnePass, fold_scalar: true, out_of_place: true }
    }
}

/// wsum takes at most this many sources per launch (`WsumArgs::src`).
pub const WSUM_MAX: usize = 8;

/// Per-call constant vectors: `slots` device slots of `slot_words` words, each with its own pinned host slot and an
/// event recorded after its upload. A push writes the pinned slot (after syncing that event when the slot is reused),
/// enqueues the H2D copy on the compute lane and returns the device pointer; the kernels reading it are issued later
/// on the same lane, and the next copy into the same device slot is ordered after them by the stream.
///
/// Graph capture (`lease_begin` .. `lease_end`): a captured graph replays the pinned -> device copy, so the pinned
/// slot must keep the captured value for the graph's lifetime. Pushes while leasing take slots out of the rotation
/// (`leased`) until `release`; their copies record no event (the slot is never rewritten while leased).
pub struct ConstRing<'a, W: DevWord> {
    dev: DBuf<'a, Compute, W>,
    host: Vec<PinnedBuf<W>>,
    ev: Vec<Event>,
    used: Vec<bool>,
    leased: Vec<bool>,
    /// Slots taken by the open lease (None: not leasing).
    lease: Option<Vec<usize>>,
    next: usize,
    slot_words: usize,
}

impl<'a, W: DevWord> ConstRing<'a, W> {
    pub fn slot_words(lay: &Layout) -> usize {
        2 * WSUM_MAX * lay.rows()
    }
    pub fn bytes(lay: &Layout, slots: usize) -> usize {
        Arena::bytes_for::<W>(slots * Self::slot_words(lay))
    }
    pub fn new(arena: &'a Arena, dev: &Device, lay: &Layout, slots: usize) -> Result<Self> {
        if slots == 0 {
            return Err(Error::Mem { knob: "OpsCfg::const_slots", value: 0, msg: "must be >= 1".into() });
        }
        let sw = Self::slot_words(lay);
        let host = (0..slots).map(|_| PinnedBuf::new(dev, sw)).collect::<std::result::Result<Vec<_>, _>>()?;
        let ev = (0..slots).map(|_| Event::new(dev)).collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(ConstRing { dev: arena.alloc("ops.const_ring", slots * sw)?, host, ev, used: vec![false; slots], leased: vec![false; slots], lease: None, next: 0, slot_words: sw })
    }

    /// Upload `fill`'s words (it writes at most `slot_words`) and return the slot's device base pointer.
    pub fn push(&mut self, s: &Stream<Compute>, fill: impl FnOnce(&mut [W])) -> Result<*const W> {
        let ns = self.host.len();
        let i = (0..ns).map(|d| (self.next + d) % ns).find(|&j| !self.leased[j]).ok_or_else(|| Error::Mem {
            knob: "OpsCfg::const_slots",
            value: ns,
            msg: "every constant slot is leased by a captured graph (release graphs or raise the knob)".into(),
        })?;
        self.next = (i + 1) % ns;
        if self.used[i] {
            // the previous copy out of this pinned slot must have run before the host rewrites it (a lease begins
            // with a stream sync, so this never waits inside a capture)
            self.ev[i].sync()?;
            self.used[i] = false;
        }
        fill(self.host[i].as_mut_slice());
        let off = i * self.slot_words;
        // SAFETY: the pinned slot is not rewritten before `ev[i]` (recorded after this copy) completes, or, leased,
        // before `release` (after the graphs replaying the copy are gone)
        unsafe { self.dev.upload_pinned_async(s, off, &self.host[i])? };
        if let Some(l) = self.lease.as_mut() {
            self.leased[i] = true;
            l.push(i);
        } else {
            self.ev[i].record(s)?;
            self.used[i] = true;
        }
        // SAFETY: off < slots * slot_words
        Ok(unsafe { self.dev.dptr().add(off) })
    }

    /// Start a lease (before a capture): waits for the lane, so no slot of the rotation has a copy in flight.
    pub fn lease_begin(&mut self, s: &Stream<Compute>) -> Result<()> {
        if self.lease.is_some() {
            return Err(shape("ConstRing::lease_begin", "a lease is already open"));
        }
        s.sync()?;
        self.used.iter_mut().for_each(|u| *u = false);
        self.lease = Some(Vec::new());
        Ok(())
    }
    /// End the open lease: the slots pushed since `lease_begin` (they stay out of the rotation until `release`).
    pub fn lease_end(&mut self) -> Vec<usize> {
        self.lease.take().unwrap_or_default()
    }
    /// Return leased slots to the rotation (every graph replaying their copies has been dropped and its last
    /// launch has completed).
    pub fn release(&mut self, slots: &[usize]) {
        for &i in slots {
            if let Some(l) = self.leased.get_mut(i) {
                *l = false;
            }
        }
    }
    /// Slots currently leased.
    pub fn leased(&self) -> usize {
        self.leased.iter().filter(|&&l| l).count()
    }

    /// `nt` tables of per-prime values (`vals[j]`, by prime id, rows `0..len`), laid out as `[j][rows]` values then
    /// `[j][rows]` Shoup companions: the `sc/scs` (one table) or `w/ws` (wsum, `rowstride = rows`) form.
    pub fn push_rows(&mut self, s: &Stream<Compute>, primes: &[u64], vals: &[&[u64]]) -> Result<(*const c_void, *const c_void)> {
        let rows = primes.len();
        let nt = vals.len();
        if 2 * nt * rows > self.slot_words {
            return Err(shape("ConstRing::push_rows", format!("{nt} tables of {rows} rows exceed a slot ({} words)", self.slot_words)));
        }
        let base = self.push(s, |h| {
            for (j, v) in vals.iter().enumerate() {
                for (i, &p) in primes.iter().enumerate() {
                    let c = shoup(W::from_u64(v.get(i).copied().unwrap_or(0) % p), W::from_u64(p));
                    h[j * rows + i] = c.w;
                    h[(nt + j) * rows + i] = c.w_prime;
                }
            }
        })?;
        // SAFETY: inside the slot (checked above)
        Ok((base as *const c_void, unsafe { base.add(nt * rows) } as *const c_void))
    }

    /// Raw per-prime words (no Shoup), e.g. add_const's `k` (values may equal `p`: the kernel reduces them).
    pub fn push_plain(&mut self, s: &Stream<Compute>, vals: &[u64]) -> Result<*const c_void> {
        if vals.len() > self.slot_words {
            return Err(shape("ConstRing::push_plain", format!("{} words exceed a slot", vals.len())));
        }
        Ok(self.push(s, |h| {
            for (o, &v) in h.iter_mut().zip(vals) {
                *o = W::from_u64(v);
            }
        })? as *const c_void)
    }
}

/// A device plaintext operand: rows by prime id `0..=top` of `meta.level`, `[row][N]` words in the NTT domain.
#[derive(Clone, Copy, Debug)]
pub struct Pt<'r> {
    pub meta: PtMeta,
    /// Device address of row 0 (rows at a stride of N words).
    pub rows: *const c_void,
    _r: std::marker::PhantomData<&'r ()>,
}

impl<'r> Pt<'r> {
    /// A view of rows owned by a buffer that outlives `'r` (PtStore slot, import buffer).
    pub fn of<W: DevWord, S: Lane>(meta: PtMeta, buf: &'r DBuf<'_, S, W>) -> Pt<'r> {
        Pt { meta, rows: buf.vptr(), _r: std::marker::PhantomData }
    }
    /// # Safety
    /// `rows` must address live device rows `0..=top(meta.level)` for as long as `'r`.
    pub unsafe fn from_raw(meta: PtMeta, rows: *const c_void) -> Pt<'r> {
        Pt { meta, rows, _r: std::marker::PhantomData }
    }
    fn poly(&self) -> PolyRef {
        PolyRef { base: self.rows as *mut c_void, ct_stride: 0, part_stride: 0 }
    }
}

/// The degree-2 result of a tensor product before relinearization: `(c0, c1)` in `c` (its meta is the recipe's
/// output: the relinearized product has the same bookkeeping) and `d2` in part 1 of `d2` (part 0 unused; the
/// `tensor` kernel's layout). `ops::mult = tensor + relin`.
pub struct Triple<'t, 'a, W: DevWord> {
    pub c: &'t mut Ct<'a, W>,
    pub d2: &'t mut CtBuf<'a, Compute, W>,
}

/// The op layer's device state (setup-carved): constant ring, NTT(X^{N/2}) rows, the rescaler.
pub struct Ops<'a, W: DevWord> {
    pub cfg: OpsCfg,
    pub ring: ConstRing<'a, W>,
    /// `mono[id][i]` = NTT of the monomial X^{N/2} mod q_id (Q ids): `mult_i` multiplies by it.
    mono: ConstBuf<'a, W>,
    pub rescaler: Rescaler<'a, W>,
    primes: Vec<u64>,
    lay: Layout,
}

/// OpenFHE ForwardTransformToBitReverse (`w[i] = psi^brev(i)`), for the setup-time monomial table.
fn host_ntt(a: &mut [u64], w: &[u64], p: u64) {
    let n = a.len();
    let (mut t, mut m) = (n, 1);
    while m < n {
        t /= 2;
        for i in 0..m {
            let s = w[m + i];
            for j in 2 * i * t..2 * i * t + t {
                let u = a[j];
                let v = mulmod(a[j + t], s, p);
                a[j] = addmod(u, v, p);
                a[j + t] = submod(u, v, p);
            }
        }
        m *= 2;
    }
}

fn mono_row(p: u64, psi: u64, n: usize, power: usize) -> Vec<u64> {
    let logn = n.trailing_zeros();
    let mut pw = vec![1u64; n];
    for i in 1..n {
        pw[i] = mulmod(pw[i - 1], psi, p);
    }
    let w: Vec<u64> = (0..n).map(|i| pw[(i as u32).reverse_bits() as usize >> (32 - logn)]).collect();
    let mut a = vec![0u64; n];
    a[power] = 1;
    host_ntt(&mut a, &w, p);
    a
}

fn rows_q(lay: &Layout, top: usize) -> RowSet {
    lay.rs_q(top)
}

/// Bookkeeping view of an operand while its preludes run.
struct Track {
    meta: CtMeta,
}

impl<'a, W: DevWord> Ops<'a, W> {
    /// Device bytes of the op layer (`MemPlan` sections "ops" and "rescale").
    pub fn reserve(plan: &mut MemPlan, cfg: &OpsCfg) {
        let lay = plan.lay;
        plan.reserve("ops", ConstRing::<W>::bytes(&lay, cfg.const_slots) + Arena::bytes_for::<W>(lay.nq * lay.n));
        let r = Rescaler::<W>::bytes(plan);
        plan.reserve("rescale", r);
    }

    pub fn new<S: Lane, M: DevStrategy<W>>(arena: &'a Arena, dev: &Device, s: &Stream<S>, ctx: &Context<'_, W, M>, cfg: OpsCfg) -> Result<Self> {
        let lay = ctx.lay;
        let ring = ConstRing::new(arena, dev, &lay, cfg.const_slots)?;
        let roots = ctx.chain.roots();
        let n = lay.n;
        let mono_v: Vec<W> = ctx.chain.q
            .par_iter()
            .zip(roots.par_iter())
            .flat_map_iter(|(&p, &psi)| mono_row(p, psi, n, n / 2).into_iter().map(W::from_u64))
            .collect();
        let mono = ConstBuf::new(arena, "ops.mono_i", s, &mono_v)?;
        let rescaler = Rescaler::new(arena, s, ctx, cfg.rescale)?;
        Ok(Ops { cfg, ring, mono, rescaler, primes: ctx.chain.primes(), lay })
    }

    // ── launch helpers (raw PolyRefs: views into Ct batches, scratch, plaintext rows) ─────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn ew<M: DevStrategy<W>>(
        ctx: &Context<'_, W, M>,
        s: &Stream<Compute>,
        name: &'static str,
        out: PolyRef,
        a: PolyRef,
        b: PolyRef,
        rs: RowSet,
        k: usize,
        nparts: usize,
        sub: bool,
        sc: (*const c_void, *const c_void),
    ) -> Result<()> {
        let args = EwArgs { out, a, b, rs, sc: sc.0, scs: sc.1, pr: ctx.pr(), k: k as i32, nparts: nparts as i32, sub: sub as i32, zero_p: 0 };
        let f = match name {
            "ew_add" => ctx.table.ew_add,
            "ew_scale" => ctx.table.ew_scale,
            "ew_copy" => ctx.table.ew_copy,
            _ => ctx.table.ew_sub_mul,
        };
        // SAFETY: the callers build the PolyRefs from buffers of the context layout holding >= k cts (or a
        // plaintext broadcast at stride 0) over rows of rs; outputs are exclusively borrowed
        launch_result(name, unsafe { f(&args, s.raw()) })?;
        Ok(())
    }

    fn mul_table<M: DevStrategy<W>>(ctx: &Context<'_, W, M>, s: &Stream<Compute>, out: PolyRef, inp: PolyRef, t: *const c_void, rows: usize, k: usize) -> Result<()> {
        let args = MulTableArgs { out, inp, t, pr: ctx.pr(), n: rows as i32, k: k as i32 };
        // SAFETY: as `ew`; t holds `rows` rows of N words
        launch_result("mul_table", unsafe { (ctx.table.mul_table)(&args, s.raw()) })?;
        Ok(())
    }

    fn top(&self, ctx_shape: pc_shadow::ChainShape, m: &CtMeta) -> Result<usize> {
        Ok(top_limb(ctx_shape, m.level)? as usize)
    }

    fn check_k(what: &'static str, a: usize, b: usize) -> Result<()> {
        if a != b {
            return Err(shape(what, format!("batch sizes differ: {a} vs {b}")));
        }
        Ok(())
    }

    /// Copy the live rows of `src` (k cts) into `dst`.
    fn copy_ct<M: DevStrategy<W>>(ctx: &Context<'_, W, M>, s: &Stream<Compute>, dst: PolyRef, src: PolyRef, k: usize, top: usize) -> Result<()> {
        Self::ew(ctx, s, "ew_copy", dst, src, src, ctx.lay.rs_q(top), k, 2, false, (std::ptr::null(), std::ptr::null()))
    }

    /// Run the preludes of `who` on `x` (k cts, in place), tracking `t.meta`. A `ScalarAdjust` directly followed by
    /// a `Rescale` rides the rescale when the rescaler folds (exact).
    fn run_pre<M: DevStrategy<W>>(
        &mut self,
        ctx: &Context<'_, W, M>,
        s: &Stream<Compute>,
        x: &mut CtBuf<'_, Compute, W>,
        t: &mut Track,
        pre: &[Prelude],
        who: Operand,
    ) -> Result<()> {
        self.run_pre_from(ctx, s, x, None, t, pre, who).map(|_| ())
    }

    /// `run_pre` when `x` does not hold the operand yet (`src` = Some: it is still at `src`): `DropTo`s
    /// move no data; a leading `Rescale` (or a folded `ScalarAdjust; Rescale`) rescales from `src` into `x` (the
    /// rescaler's out-of-place form: no full copy); any other data prelude copies `src` into `x` first. Returns the
    /// operand's view after the preludes: `src` itself when none of them touched data (no copy at all), else `x`.
    #[allow(clippy::too_many_arguments)]
    fn run_pre_from<M: DevStrategy<W>>(
        &mut self,
        ctx: &Context<'_, W, M>,
        s: &Stream<Compute>,
        x: &mut CtBuf<'_, Compute, W>,
        mut src: Option<PolyRef>,
        t: &mut Track,
        pre: &[Prelude],
        who: Operand,
    ) -> Result<PolyRef> {
        let mine: Vec<Prelude> = pre
            .iter()
            .copied()
            .filter(|p| matches!(p, Prelude::Rescale(o) | Prelude::DropTo(o, _) | Prelude::ScalarAdjust(o, _) if *o == who))
            .collect();
        let shape_ = ctx.shape;
        let unit = shape_.unit;
        let out_of_place = self.cfg.out_of_place && self.rescaler.can_fold();
        let mut i = 0;
        while i < mine.len() {
            let top = self.top(shape_, &t.meta)?;
            if let (Some(sv), false) = (src, matches!(mine[i], Prelude::DropTo(..))) {
                // the first data prelude: it reads `src` (rescale) or `src` is copied into x now
                if !(out_of_place && matches!(mine[i], Prelude::Rescale(_))) && !(out_of_place && matches!(mine[i], Prelude::ScalarAdjust(..)) && matches!(mine.get(i + 1), Some(Prelude::Rescale(_)))) {
                    Self::copy_ct(ctx, s, x.w(), sv, x.k, top)?;
                    src = None;
                }
            }
            match mine[i] {
                Prelude::Rescale(_) => {
                    match src.take() {
                        Some(sv) => self.rescaler.rescale_to(ctx, s, sv, x, top, None)?,
                        None => self.rescaler.rescale(ctx, s, x, top, None)?,
                    }
                    t.meta.level += unit;
                    t.meta.deg = t.meta.deg.saturating_sub(1);
                }
                Prelude::DropTo(_, l) => {
                    if l < t.meta.level {
                        return Err(shape("DropTo", format!("target level {l} is above the operand's {}", t.meta.level)));
                    }
                    t.meta.level = l;
                }
                Prelude::ScalarAdjust(_, c) => {
                    let fold = self.cfg.fold_scalar && self.rescaler.can_fold() && matches!(mine.get(i + 1), Some(Prelude::Rescale(_)));
                    let rows = ctx.scale.elem_for_eval_mult(top as i32, c, -1)?;
                    let q = &ctx.chain.q;
                    let foldable = fold && !(rows[top] % q[top] == 1 && rows[top - 1] % q[top - 1] != 1);
                    if foldable {
                        let al = self.rescaler.folded_alpha(top, &rows, q)?;
                        let (av, als) = self.ring.push_rows(s, &self.primes, &[&al])?;
                        match src.take() {
                            Some(sv) => self.rescaler.rescale_to(ctx, s, sv, x, top, Some((&rows, av, als)))?,
                            None => self.rescaler.rescale(ctx, s, x, top, Some((&rows, av, als)))?,
                        }
                        t.meta.level += unit;
                        i += 2;
                        continue;
                    }
                    if let Some(sv) = src.take() {
                        // not foldable after all: copy, then scale in place
                        Self::copy_ct(ctx, s, x.w(), sv, x.k, top)?;
                    }
                    let sc = self.ring.push_rows(s, &self.primes, &[&rows])?;
                    let r = x.w();
                    Self::ew(ctx, s, "ew_scale", r, r, r, ctx.lay.rs_q(top), x.k, 2, false, sc)?;
                    t.meta.deg += 1;
                }
            }
            i += 1;
        }
        Ok(src.unwrap_or_else(|| x.r()))
    }

    /// The B operand's copy: `b`'s live rows in `ex.scratch` (first `k` cts), its preludes run there.
    fn prep_b<M: DevStrategy<W>>(
        &mut self,
        ctx: &Context<'_, W, M>,
        ex: &mut Executor<'_, W>,
        b_ref: PolyRef,
        b_meta: CtMeta,
        k: usize,
        pre: &[Prelude],
    ) -> Result<Option<(PolyRef, CtMeta)>> {
        if !pre.iter().any(|p| matches!(p, Prelude::Rescale(Operand::B) | Prelude::DropTo(Operand::B, _) | Prelude::ScalarAdjust(Operand::B, _))) {
            return Ok(None);
        }
        if k > ex.scratch.k {
            return Err(Error::Mem { knob: "MemCfg::scratch_cts", value: ex.scratch.k, msg: format!("operand copy of {k} cts") });
        }
        let kfull = ex.scratch.k;
        ex.scratch.k = k;
        let mut t = Track { meta: b_meta };
        let r = if self.cfg.out_of_place {
            // the preludes read `b` where it is (an out-of-place rescale into the scratch copy)
            self.run_pre_from(ctx, &ex.compute, &mut ex.scratch, Some(b_ref), &mut t, pre, Operand::B)
        } else {
            let top = self.top(ctx.shape, &b_meta)?;
            Self::copy_ct(ctx, &ex.compute, ex.scratch.w(), b_ref, k, top).and_then(|_| self.run_pre(ctx, &ex.compute, &mut ex.scratch, &mut t, pre, Operand::B).map(|_| ex.scratch.r()))
        };
        ex.scratch.k = kfull;
        Ok(Some((r?, t.meta)))
    }

    /// Operand A into `out`: copy (when the op is out of place) and run A's preludes there.
    fn prep_a<M: DevStrategy<W>>(
        &mut self,
        ctx: &Context<'_, W, M>,
        s: &Stream<Compute>,
        out: &mut Ct<'_, W>,
        a: Option<&Ct<'_, W>>,
        pre: &[Prelude],
    ) -> Result<Track> {
        let meta = match a {
            Some(a) => {
                Self::check_k("op", out.k(), a.k())?;
                let top = self.top(ctx.shape, &a.meta)?;
                Self::copy_ct(ctx, s, out.buf.w(), a.buf.r(), a.k(), top)?;
                a.meta
            }
            None => out.meta,
        };
        let mut t = Track { meta };
        self.run_pre(ctx, s, &mut out.buf, &mut t, pre, Operand::A)?;
        Ok(t)
    }

    /// Operand A of an op whose kernel can read its input from another buffer than `out`: no copy when
    /// the op is out of place and A has no data prelude (the kernel reads `a`), an out-of-place rescale into `out`
    /// for a leading Rescale prelude, copy + preludes otherwise. Returns the view the kernel reads A from.
    fn prep_a_view<M: DevStrategy<W>>(
        &mut self,
        ctx: &Context<'_, W, M>,
        s: &Stream<Compute>,
        out: &mut Ct<'_, W>,
        a: Option<&Ct<'_, W>>,
        pre: &[Prelude],
    ) -> Result<(Track, PolyRef)> {
        let Some(a) = a else {
            let mut t = Track { meta: out.meta };
            self.run_pre(ctx, s, &mut out.buf, &mut t, pre, Operand::A)?;
            return Ok((t, out.buf.r()));
        };
        Self::check_k("op", out.k(), a.k())?;
        if !self.cfg.out_of_place {
            let t = self.prep_a(ctx, s, out, Some(a), pre)?;
            return Ok((t, out.buf.r()));
        }
        let mut t = Track { meta: a.meta };
        let v = self.run_pre_from(ctx, s, &mut out.buf, Some(a.buf.r()), &mut t, pre, Operand::A)?;
        Ok((t, v))
    }

    fn finish(out: &mut Ct<'_, W>, r: &Recipe, t: &Track) -> Result<()> {
        if t.meta.level != r.out.level {
            return Err(shape("recipe", format!("preludes left level {} but the recipe's output is at {}", t.meta.level, r.out.level)));
        }
        out.meta = r.out;
        Ok(())
    }

    // ── additive ─────────────────────────────────────────────────────────────────────────────────────────────────

    fn add_like<M: DevStrategy<W>>(
        &mut self,
        ctx: &Context<'_, W, M>,
        ex: &mut Executor<'_, W>,
        sh: &Shadow,
        out: &mut Ct<'_, W>,
        a: Option<&Ct<'_, W>>,
        b: &Ct<'_, W>,
        sub: bool,
    ) -> Result<()> {
        let am = a.map_or(out.meta, |a| a.meta);
        let r = if sub { sh.sub(&am, &b.meta)? } else { sh.add(&am, &b.meta)? };
        Self::check_k("add", out.k(), b.k())?;
        let bp = self.prep_b(ctx, ex, b.buf.r(), b.meta, b.k(), &r.pre)?;
        let s = &ex.compute;
        let (t, av) = self.prep_a_view(ctx, s, out, a, &r.pre)?;
        let top = self.top(ctx.shape, &t.meta)?;
        let bref = bp.map_or(b.buf.r(), |x| x.0);
        let o = out.buf.w();
        Self::ew(ctx, s, "ew_add", o, av, bref, rows_q(&ctx.lay, top), out.k(), 2, sub, (std::ptr::null(), std::ptr::null()))?;
        Self::finish(out, &r, &t)
    }

    /// `out[j] = a[j] + b[j]` (`Ciphertext::add`).
    pub fn add<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, out: &mut [Ct<'_, W>], a: &[&Ct<'_, W>], b: &[&Ct<'_, W>]) -> Result<()> {
        lens("add", out.len(), &[a.len(), b.len()])?;
        for ((o, a), b) in out.iter_mut().zip(a).zip(b) {
            self.add_like(ctx, ex, sh, o, Some(a), b, false)?;
        }
        Ok(())
    }
    /// `out[j] = a[j] - b[j]`.
    pub fn sub<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, out: &mut [Ct<'_, W>], a: &[&Ct<'_, W>], b: &[&Ct<'_, W>]) -> Result<()> {
        lens("sub", out.len(), &[a.len(), b.len()])?;
        for ((o, a), b) in out.iter_mut().zip(a).zip(b) {
            self.add_like(ctx, ex, sh, o, Some(a), b, true)?;
        }
        Ok(())
    }
    /// `a[j] += b[j]`.
    pub fn add_inplace<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, a: &mut [Ct<'_, W>], b: &[&Ct<'_, W>]) -> Result<()> {
        lens("add_inplace", a.len(), &[b.len()])?;
        for (o, b) in a.iter_mut().zip(b) {
            self.add_like(ctx, ex, sh, o, None, b, false)?;
        }
        Ok(())
    }
    /// `a[j] -= b[j]`.
    pub fn sub_inplace<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, a: &mut [Ct<'_, W>], b: &[&Ct<'_, W>]) -> Result<()> {
        lens("sub_inplace", a.len(), &[b.len()])?;
        for (o, b) in a.iter_mut().zip(b) {
            self.add_like(ctx, ex, sh, o, None, b, true)?;
        }
        Ok(())
    }

    /// `out[j] = a[j] + c[j]` (`addScalar`: the constant at the ciphertext's top and degree, no level change).
    pub fn add_scalar<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, out: &mut [Ct<'_, W>], a: &[&Ct<'_, W>], c: &[f64]) -> Result<()> {
        lens("add_scalar", out.len(), &[a.len(), c.len()])?;
        for ((o, a), &c) in out.iter_mut().zip(a).zip(c) {
            self.add_scalar1(ctx, &ex.compute, sh, o, Some(a), c)?;
        }
        Ok(())
    }
    pub fn add_scalar_inplace<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, a: &mut [Ct<'_, W>], c: &[f64]) -> Result<()> {
        lens("add_scalar_inplace", a.len(), &[c.len()])?;
        for (o, &c) in a.iter_mut().zip(c) {
            self.add_scalar1(ctx, &ex.compute, sh, o, None, c)?;
        }
        Ok(())
    }
    fn add_scalar1<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, sh: &Shadow, out: &mut Ct<'_, W>, a: Option<&Ct<'_, W>>, c: f64) -> Result<()> {
        let am = a.map_or(out.meta, |a| a.meta);
        let r = sh.add_scalar(&am, c)?;
        let t = self.prep_a(ctx, s, out, a, &r.pre)?;
        let top = self.top(ctx.shape, &t.meta)?;
        if let Some(pc_shadow::Consts::Add(rows)) = &r.consts {
            let k = self.ring.push_plain(s, rows)?;
            let args = AddConstArgs { x: out.buf.w(), k, pr: ctx.pr(), n: (top + 1) as i32, kk: out.k() as i32 };
            // SAFETY: k cts of the context layout, rows 0..=top; k is a ring slot written earlier on this stream
            launch_result("add_const", unsafe { (ctx.table.add_const)(&args, s.raw()) })?;
        } else if c != 0.0 {
            return Err(shape("add_scalar", "the Shadow has no scale tables (no constant rows)"));
        }
        Self::finish(out, &r, &t)
    }

    /// `out[j] = a[j] + pt[j]` (`Ciphertext::addPt`; the plaintext broadcast over the batch).
    pub fn add_pt<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, out: &mut [Ct<'_, W>], a: &[&Ct<'_, W>], pt: &[Pt<'_>]) -> Result<()> {
        lens("add_pt", out.len(), &[a.len(), pt.len()])?;
        for ((o, a), p) in out.iter_mut().zip(a).zip(pt) {
            self.add_pt1(ctx, ex, sh, o, Some(a), p)?;
        }
        Ok(())
    }
    pub fn add_pt_inplace<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, a: &mut [Ct<'_, W>], pt: &[Pt<'_>]) -> Result<()> {
        lens("add_pt_inplace", a.len(), &[pt.len()])?;
        for (o, p) in a.iter_mut().zip(pt) {
            self.add_pt1(ctx, ex, sh, o, None, p)?;
        }
        Ok(())
    }
    fn add_pt1<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, out: &mut Ct<'_, W>, a: Option<&Ct<'_, W>>, pt: &Pt<'_>) -> Result<()> {
        let am = a.map_or(out.meta, |a| a.meta);
        let r = sh.add_pt(&am, &pt.meta)?;
        let pr = self.pt_operand(ctx, ex, pt, &r.pre)?;
        let s = &ex.compute;
        let (t, av) = self.prep_a_view(ctx, s, out, a, &r.pre)?;
        let top = self.top(ctx.shape, &t.meta)?;
        let o = out.buf.w();
        if av.base != o.base {
            // only part 0 gets the plaintext: part 1 is copied
            let c1 = |r: PolyRef| PolyRef { base: (r.base as *mut W).wrapping_add(r.part_stride as usize) as *mut c_void, ..r };
            Self::ew(ctx, s, "ew_copy", c1(o), c1(av), c1(av), rows_q(&ctx.lay, top), out.k(), 1, false, (std::ptr::null(), std::ptr::null()))?;
        }
        Self::ew(ctx, s, "ew_add", o, av, pr, rows_q(&ctx.lay, top), out.k(), 1, false, (std::ptr::null(), std::ptr::null()))?;
        Self::finish(out, &r, &t)
    }

    /// A plaintext operand after its B preludes (on a scratch copy, broadcast at stride 0), or the rows as they are.
    fn pt_operand<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, pt: &Pt<'_>, pre: &[Prelude]) -> Result<PolyRef> {
        let m = CtMeta { level: pt.meta.level, deg: pt.meta.deg, nf: pt.meta.nf, slots: 0, tag: pt.meta.tag };
        // both parts of the copy read the plaintext, so the rescale of the copy works on defined words
        Ok(match self.prep_b(ctx, ex, pt.poly(), m, 1, pre)? {
            Some((r, _)) => PolyRef { ct_stride: 0, ..r },
            None => pt.poly(),
        })
    }

    // ── multiplicative ───────────────────────────────────────────────────────────────────────────────────────────

    /// `out[j] = a[j] * c[j]` (`multScalar`; a complex scalar `re + i im` is `re * a + im * (X^{N/2} a)`, both
    /// scalars as ElemForEvalMult rows at the operand's top, one wsum launch).
    pub fn mult_scalar<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, out: &mut [Ct<'_, W>], a: &[&Ct<'_, W>], c: &[Complex]) -> Result<()> {
        lens("mult_scalar", out.len(), &[a.len(), c.len()])?;
        for ((o, a), &c) in out.iter_mut().zip(a).zip(c) {
            self.mult_scalar1(ctx, ex, sh, o, Some(a), c)?;
        }
        Ok(())
    }
    pub fn mult_scalar_inplace<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, a: &mut [Ct<'_, W>], c: &[Complex]) -> Result<()> {
        lens("mult_scalar_inplace", a.len(), &[c.len()])?;
        for (o, &c) in a.iter_mut().zip(c) {
            self.mult_scalar1(ctx, ex, sh, o, None, c)?;
        }
        Ok(())
    }
    fn mult_scalar1<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, out: &mut Ct<'_, W>, a: Option<&Ct<'_, W>>, c: Complex) -> Result<()> {
        let am = a.map_or(out.meta, |a| a.meta);
        let mut r = sh.mult_scalar(&am, c.re)?;
        if c.im != 0.0 {
            // a complex scalar mixes the real and imaginary lanes
            if r.out.tag.field != tag::F_UNKNOWN {
                r.out.tag = tag::with_field(&r.out.tag, tag::F_COMPLEX);
            }
        }
        let (t, av) = self.prep_a_view(ctx, &ex.compute, out, a, &r.pre)?;
        let top = self.top(ctx.shape, &t.meta)?;
        let s = &ex.compute;
        let o = out.buf.w();
        if c.im == 0.0 {
            let rows = match &r.consts {
                Some(pc_shadow::Consts::Mult(rows)) => rows.clone(),
                _ => ctx.scale.elem_for_eval_mult(top as i32, c.re, -1)?,
            };
            let sc = self.ring.push_rows(s, &self.primes, &[&rows])?;
            Self::ew(ctx, s, "ew_scale", o, av, av, rows_q(&ctx.lay, top), out.k(), 2, false, sc)?;
        } else {
            let k = out.k();
            if k > ex.scratch.k {
                return Err(Error::Mem { knob: "MemCfg::scratch_cts", value: ex.scratch.k, msg: format!("complex scalar of {k} cts") });
            }
            let re = ctx.scale.elem_for_eval_mult(top as i32, c.re, -1)?;
            let im = ctx.scale.elem_for_eval_mult(top as i32, c.im, -1)?;
            let tmp = ex.scratch.w();
            Self::mul_table(ctx, s, tmp, av, self.mono.vptr(), top + 1, k)?;
            let (w, ws) = self.ring.push_rows(s, &self.primes, &[&re, &im])?;
            let mut src = [PolyRef::default(); WSUM_MAX];
            src[0] = av;
            src[1] = tmp;
            let args = WsumArgs { out: o, src, w, ws, pr: ctx.pr(), nsrc: 2, rowstride: self.primes.len() as i32, n: (top + 1) as i32, k: k as i32 };
            // SAFETY: element-wise per (ct, row, i): out aliases src[0] but each thread reads its sources before
            // writing the same index
            launch_result("wsum", unsafe { (ctx.table.wsum)(&args, s.raw()) })?;
        }
        Self::finish(out, &r, &t)
    }

    /// `out[j] = -a[j]` (`EvalNegate` on the GPU = `multScalar(-1.0)`: costs the pending rescale, degree 2).
    pub fn negate<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, out: &mut [Ct<'_, W>], a: &[&Ct<'_, W>]) -> Result<()> {
        lens("negate", out.len(), &[a.len()])?;
        for (o, a) in out.iter_mut().zip(a) {
            let r = sh.negate(&a.meta)?;
            let s = &ex.compute;
            let (t, av) = self.prep_a_view(ctx, s, o, Some(a), &r.pre)?;
            let top = self.top(ctx.shape, &t.meta)?;
            if let Some(pc_shadow::Consts::Mult(rows)) = &r.consts {
                let sc = self.ring.push_rows(s, &self.primes, &[rows])?;
                let ob = o.buf.w();
                Self::ew(ctx, s, "ew_scale", ob, av, av, rows_q(&ctx.lay, top), o.k(), 2, false, sc)?;
            } else {
                // levels-only Shadow (FakeNegateFree or no tables): -x = (p - 1) x
                let m1: Vec<u64> = self.primes.iter().map(|p| p - 1).collect();
                let sc = self.ring.push_rows(s, &self.primes, &[&m1])?;
                let ob = o.buf.w();
                Self::ew(ctx, s, "ew_scale", ob, av, av, rows_q(&ctx.lay, top), o.k(), 2, false, sc)?;
            }
            Self::finish(o, &r, &t)?;
        }
        Ok(())
    }

    /// `out[j] = a[j] * pt[j]` (`Ciphertext::multPt`: rescale first, the plaintext adjusted on a copy).
    pub fn mult_pt<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, out: &mut [Ct<'_, W>], a: &[&Ct<'_, W>], pt: &[Pt<'_>]) -> Result<()> {
        lens("mult_pt", out.len(), &[a.len(), pt.len()])?;
        for ((o, a), p) in out.iter_mut().zip(a).zip(pt) {
            self.mult_pt1(ctx, ex, sh, o, Some(a), p)?;
        }
        Ok(())
    }
    pub fn mult_pt_inplace<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, a: &mut [Ct<'_, W>], pt: &[Pt<'_>]) -> Result<()> {
        lens("mult_pt_inplace", a.len(), &[pt.len()])?;
        for (o, p) in a.iter_mut().zip(pt) {
            self.mult_pt1(ctx, ex, sh, o, None, p)?;
        }
        Ok(())
    }
    fn mult_pt1<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, out: &mut Ct<'_, W>, a: Option<&Ct<'_, W>>, pt: &Pt<'_>) -> Result<()> {
        let am = a.map_or(out.meta, |a| a.meta);
        let r = sh.mult_pt(&am, &pt.meta)?;
        let pr = self.pt_operand(ctx, ex, pt, &r.pre)?;
        let s = &ex.compute;
        let (t, av) = self.prep_a_view(ctx, s, out, a, &r.pre)?;
        let top = self.top(ctx.shape, &t.meta)?;
        let o = out.buf.w();
        Self::mul_table(ctx, s, o, av, pr.base, top + 1, out.k())?;
        Self::finish(out, &r, &t)
    }

    /// The add / sub recipe of `mult_pt_add` (the product as operand A when `product_first`).
    fn mac_recipes(sh: &Shadow, a: &CtMeta, pt: &PtMeta, y: &CtMeta, sub: bool, product_first: bool) -> Result<(Recipe, Recipe)> {
        let r1 = sh.mult_pt(a, pt)?;
        let (x, z) = if product_first { (r1.out, *y) } else { (*y, r1.out) };
        let r2 = if sub { sh.sub(&x, &z)? } else { sh.add(&x, &z)? };
        Ok((r1, r2))
    }

    /// Whether `mult_pt_add` runs `y +- a * pt` as one pass (metadata only, a bind-time check): the add's recipe has
    /// no prelude and the product's has none on the plaintext. With preludes on `a`, `out` must not alias `y`.
    pub fn mult_pt_add_fusable(sh: &Shadow, a: &CtMeta, pt: &PtMeta, y: &CtMeta, sub: bool, product_first: bool) -> Result<bool> {
        let (r1, r2) = Self::mac_recipes(sh, a, pt, y, sub, product_first)?;
        let on_pt = r1.pre.iter().any(|p| matches!(p, Prelude::Rescale(Operand::B) | Prelude::DropTo(Operand::B, _) | Prelude::ScalarAdjust(Operand::B, _)));
        Ok(r2.pre.is_empty() && !on_pt)
    }

    /// `out = y + a * pt` (`sub`: `y - a * pt`, or `a * pt - y` when `product_first`): exactly `mult_pt` into a
    /// temporary, then `add` / `sub` of it with `y` (the product as the add's operand A when `product_first`), as one
    /// pass (the plaintext products of BSGS baby sums, consumed once by an add). Needs `mult_pt_add_fusable`; `a`'s
    /// preludes run into `out` as in `mult_pt`.
    #[allow(clippy::too_many_arguments)]
    pub fn mult_pt_add<M: DevStrategy<W>>(
        &mut self,
        ctx: &Context<'_, W, M>,
        ex: &mut Executor<'_, W>,
        sh: &Shadow,
        out: &mut Ct<'_, W>,
        a: &Ct<'_, W>,
        pt: &Pt<'_>,
        y: &Ct<'_, W>,
        sub: bool,
        product_first: bool,
    ) -> Result<()> {
        if !Self::mult_pt_add_fusable(sh, &a.meta, &pt.meta, &y.meta, sub, product_first)? {
            return Err(shape("mult_pt_add", "the add has preludes or the plaintext needs one: run mult_pt then add"));
        }
        let (r1, r2) = Self::mac_recipes(sh, &a.meta, &pt.meta, &y.meta, sub, product_first)?;
        Self::check_k("mult_pt_add", out.k(), y.k())?;
        if !r1.pre.is_empty() && out.buf.r().base == y.buf.r().base {
            return Err(shape("mult_pt_add", "the product's preludes run in `out`, which must not alias y"));
        }
        let s = &ex.compute;
        let (t, av) = self.prep_a_view(ctx, s, out, Some(a), &r1.pre)?;
        if t.meta.level != r1.out.level {
            return Err(shape("mult_pt_add", format!("preludes left level {} but the product is at {}", t.meta.level, r1.out.level)));
        }
        let top = self.top(ctx.shape, &t.meta)?;
        let m = MulTableArgs { out: out.buf.w(), inp: av, t: pt.poly().base, pr: ctx.pr(), n: (top + 1) as i32, k: out.k() as i32 };
        let args = MulAddArgs { m, y: y.buf.r(), sub: if !sub { 0 } else if product_first { 2 } else { 1 }, pad: 0 };
        // SAFETY: k cts of the context layout over rows 0..=top (the plaintext broadcast); element-wise, so out may
        // alias y or a's view (each thread reads its operands before writing the same index)
        launch_result("mul_add", unsafe { (ctx.table.mul_add)(&args, s.raw()) })?;
        out.meta = r2.out;
        Ok(())
    }

    /// Whether `mult_pt_add_inplace` runs: `mult_pt_add_fusable` and no prelude on `a` either (it would run in `acc`).
    pub fn mult_pt_add_inplace_fusable(sh: &Shadow, a: &CtMeta, pt: &PtMeta, acc: &CtMeta, sub: bool, product_first: bool) -> Result<bool> {
        let (r1, _) = Self::mac_recipes(sh, a, pt, acc, sub, product_first)?;
        Ok(r1.pre.is_empty() && Self::mult_pt_add_fusable(sh, a, pt, acc, sub, product_first)?)
    }

    /// In place: `acc = acc + a * pt` (`sub`: `acc - a * pt`, or `a * pt - acc` when `product_first`), `mult_pt_add`
    /// with out = y = acc (an add landing in its other operand's slot); the same words and output meta. Needs
    /// `mult_pt_add_inplace_fusable`.
    #[allow(clippy::too_many_arguments)]
    pub fn mult_pt_add_inplace<M: DevStrategy<W>>(
        &mut self,
        ctx: &Context<'_, W, M>,
        ex: &mut Executor<'_, W>,
        sh: &Shadow,
        acc: &mut Ct<'_, W>,
        a: &Ct<'_, W>,
        pt: &Pt<'_>,
        sub: bool,
        product_first: bool,
    ) -> Result<()> {
        if !Self::mult_pt_add_inplace_fusable(sh, &a.meta, &pt.meta, &acc.meta, sub, product_first)? {
            return Err(shape("mult_pt_add_inplace", "a prelude on a, the plaintext or the add: run mult_pt then add"));
        }
        let (r1, r2) = Self::mac_recipes(sh, &a.meta, &pt.meta, &acc.meta, sub, product_first)?;
        Self::check_k("mult_pt_add_inplace", acc.k(), a.k())?;
        if a.meta.level != r1.out.level {
            return Err(shape("mult_pt_add_inplace", format!("a at level {} but the product is at {}", a.meta.level, r1.out.level)));
        }
        let top = self.top(ctx.shape, &a.meta)?;
        let y = acc.buf.r();
        let m = MulTableArgs { out: acc.buf.w(), inp: a.buf.r(), t: pt.poly().base, pr: ctx.pr(), n: (top + 1) as i32, k: acc.k() as i32 };
        let args = MulAddArgs { m, y, sub: if !sub { 0 } else if product_first { 2 } else { 1 }, pad: 0 };
        // SAFETY: as in mult_pt_add; out == y is the element-wise in-place form (each thread reads before it writes)
        launch_result("mul_add", unsafe { (ctx.table.mul_add)(&args, ex.compute.raw()) })?;
        acc.meta = r2.out;
        Ok(())
    }

    /// `out[j] = i * a[j]`: the monomial X^{N/2} (level and degree kept; exact).
    pub fn mult_i<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, out: &mut [Ct<'_, W>], a: &[&Ct<'_, W>]) -> Result<()> {
        lens("mult_i", out.len(), &[a.len()])?;
        for (o, a) in out.iter_mut().zip(a) {
            Self::check_k("mult_i", o.k(), a.k())?;
            let r = sh.mult_i(&a.meta);
            let top = self.top(ctx.shape, &a.meta)?;
            Self::mul_table(ctx, &ex.compute, o.buf.w(), a.buf.r(), self.mono.vptr(), top + 1, a.k())?;
            o.meta = r.out;
        }
        Ok(())
    }
    pub fn mult_i_inplace<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, a: &mut [Ct<'_, W>]) -> Result<()> {
        for o in a.iter_mut() {
            let r = sh.mult_i(&o.meta);
            let top = self.top(ctx.shape, &o.meta)?;
            let b = o.buf.w();
            Self::mul_table(ctx, &ex.compute, b, b, self.mono.vptr(), top + 1, o.k())?;
            o.meta = r.out;
        }
        Ok(())
    }

    /// The tensor half of `a * b` (`Ciphertext::mult` before relinearization): the recipe's preludes, then
    /// `(a0 b0, a0 b1 + a1 b0, a1 b1)`; `out.c.meta` = the product's bookkeeping. Relinearization is `relin`'s.
    pub fn tensor<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, out: &mut Triple<'_, '_, W>, a: &Ct<'_, W>, b: &Ct<'_, W>) -> Result<()> {
        let r = sh.mult(&a.meta, &b.meta)?;
        Self::check_k("tensor", a.k(), b.k())?;
        let bp = self.prep_b(ctx, ex, b.buf.r(), b.meta, b.k(), &r.pre)?;
        let s = &ex.compute;
        let t = self.prep_a(ctx, s, out.c, Some(a), &r.pre)?;
        let top = self.top(ctx.shape, &t.meta)?;
        let bref = bp.map_or(b.buf.r(), |x| x.0);
        let x = out.c.buf.r();
        self.tensor_launch(ctx, s, out, &[(x, bref)], false, top)?;
        Self::finish(out.c, &r, &t)
    }

    /// The tensor half of `a^2` (`Ciphertext::square`: rescale first).
    pub fn square_tensor<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, out: &mut Triple<'_, '_, W>, a: &Ct<'_, W>) -> Result<()> {
        let r = sh.square(&a.meta)?;
        let s = &ex.compute;
        let t = self.prep_a(ctx, s, out.c, Some(a), &r.pre)?;
        let top = self.top(ctx.shape, &t.meta)?;
        let x = out.c.buf.r();
        self.tensor_launch(ctx, s, out, &[(x, x)], false, top)?;
        Self::finish(out.c, &r, &t)
    }

    /// Sum of tensors `sum_j a_j * b_j` (the tensor half of `mult_add_many` over ciphertext pairs, one relin later):
    /// every pair must share the bookkeeping (no preludes), `out.c.meta` = the product's.
    pub fn tensor_sum<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, out: &mut Triple<'_, '_, W>, a: &[&Ct<'_, W>], b: &[&Ct<'_, W>]) -> Result<()> {
        lens("tensor_sum", a.len(), &[b.len()])?;
        if a.is_empty() {
            return Err(shape("tensor_sum", "no pairs"));
        }
        let r = sh.mult(&a[0].meta, &b[0].meta)?;
        if !r.pre.is_empty() || a.iter().zip(b).any(|(x, y)| x.meta.level != a[0].meta.level || y.meta.level != a[0].meta.level || x.meta.deg != 1 || y.meta.deg != 1) {
            return Err(shape("tensor_sum", "pairs must share level and degree 1 (no preludes); use tensor + add"));
        }
        let top = self.top(ctx.shape, &a[0].meta)?;
        let pairs: Vec<(PolyRef, PolyRef)> = a.iter().zip(b).map(|(x, y)| (x.buf.r(), y.buf.r())).collect();
        for (i, ch) in pairs.chunks(2).enumerate() {
            self.tensor_launch(ctx, &ex.compute, out, ch, i > 0, top)?;
        }
        out.c.meta = r.out;
        Ok(())
    }

    fn tensor_launch<M: DevStrategy<W>>(&self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, out: &mut Triple<'_, '_, W>, pairs: &[(PolyRef, PolyRef)], acc: bool, top: usize) -> Result<()> {
        let k = out.c.k();
        if out.d2.k < k {
            return Err(shape("tensor d2", format!("d2 buffer holds {} cts, the batch {k}", out.d2.k)));
        }
        let (a0, b0) = pairs[0];
        let (a1, b1) = pairs.get(1).copied().unwrap_or(pairs[0]);
        let args = TensorArgs {
            out: out.c.buf.w(),
            d2: out.d2.w(),
            a0,
            b0,
            a1,
            b1,
            coef: std::ptr::null_mut(),
            coef_ct_stride: 0,
            pr: ctx.pr(),
            npairs: pairs.len() as i32,
            acc: acc as i32,
            n: (top + 1) as i32,
            k: k as i32,
        };
        // SAFETY: per (ct, row, i) the kernel reads all inputs before writing out[c] parts 0/1 and d2[c] part 1, so
        // out aliasing an input is element-wise safe; rows 0..=top of k cts
        launch_result("tensor", unsafe { (ctx.table.tensor)(&args, s.raw()) })?;
        Ok(())
    }

    /// The tensor half of `a * b` (`b = None`: `a^2`) staged for relinearization: the recipe's preludes (A copied
    /// into `out` and adjusted there, B on a scratch copy), then `(c0, c1)` into `out` and d2 into `ex.ks.ext` part 1
    /// (where `relin::relin_staged` expects it). Returns the live top and the recipe; `out.meta` is left to the caller
    /// (the relinearized product takes `recipe.out`).
    pub fn tensor_for_relin<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, out: &mut Ct<'_, W>, a: &Ct<'_, W>, b: Option<&Ct<'_, W>>) -> Result<(usize, Recipe)> {
        let r = match b {
            Some(b) => sh.mult(&a.meta, &b.meta)?,
            None => sh.square(&a.meta)?,
        };
        let k = a.k();
        if k > ex.ks.k_max {
            return Err(shape("mult", format!("batch of {k} > the key-switching k_max {}", ex.ks.k_max)));
        }
        let bp = match b {
            Some(b) => {
                Self::check_k("mult", k, b.k())?;
                self.prep_b(ctx, ex, b.buf.r(), b.meta, k, &r.pre)?
            }
            None => None,
        };
        let (t, x) = self.prep_a_view(ctx, &ex.compute, out, Some(a), &r.pre)?;
        if t.meta.level != r.out.level {
            return Err(shape("mult", format!("preludes left level {} but the recipe's output is at {}", t.meta.level, r.out.level)));
        }
        if let Some((_, bm)) = bp {
            if bm.level != t.meta.level {
                return Err(shape("mult", format!("operand preludes left levels {} / {}", t.meta.level, bm.level)));
            }
        }
        let top = self.top(ctx.shape, &t.meta)?;
        let y = match (b, bp) {
            (_, Some((p, _))) => p,
            (Some(b), None) => b.buf.r(),
            (None, None) => x,
        };
        let lay = ctx.lay;
        let args = TensorArgs {
            out: out.buf.w(),
            d2: crate::ks::View::of(&ex.ks.ext, 0).r(),
            a0: x,
            b0: y,
            a1: x,
            b1: y,
            coef: std::ptr::null_mut(),
            coef_ct_stride: lay.poly_words() as u64,
            pr: ctx.pr(),
            npairs: 1,
            acc: 0,
            n: (top + 1) as i32,
            k: k as i32,
        };
        // SAFETY: per (ct, row, i) the kernel reads all inputs before writing out[c] parts 0/1, so out aliasing the A
        // operand (it holds A's adjusted copy) is element-wise safe; ext is distinct from out and the scratch copy of
        // B; rows 0..=top of k cts
        launch_result("tensor", unsafe { (ctx.table.tensor)(&args, ex.compute.raw()) })?;
        Ok((top, r))
    }

    /// `a * b` relinearized (FIDESlib `Ciphertext::mult`): the Recipe preludes (Rescale / DropTo / ScalarAdjust; A
    /// on its copy in `out`, B on a scratch copy — the operands themselves are untouched, unlike perseus' wrapper,
    /// which rescaled `a` in place), the tensor, then `relin::relin_staged`. Degree 2 out (rescale pending).
    #[allow(clippy::too_many_arguments)]
    pub fn mult<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, kc: crate::rotate::KsCfg<'_, '_>, sh: &Shadow, key: &crate::ks::KeyGroup<'_>, out: &mut Ct<'_, W>, a: &Ct<'_, W>, b: &Ct<'_, W>) -> Result<()> {
        let (top, r) = self.tensor_for_relin(ctx, ex, sh, out, a, Some(b))?;
        crate::relin::relin_staged(ctx, ex, kc, key, out, top)?;
        out.meta = r.out;
        Ok(())
    }

    /// `a^2` relinearized (FIDESlib `square`: rescale first), preludes as `mult`.
    #[allow(clippy::too_many_arguments)]
    pub fn square<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, kc: crate::rotate::KsCfg<'_, '_>, sh: &Shadow, key: &crate::ks::KeyGroup<'_>, out: &mut Ct<'_, W>, a: &Ct<'_, W>) -> Result<()> {
        let (top, r) = self.tensor_for_relin(ctx, ex, sh, out, a, None)?;
        crate::relin::relin_staged(ctx, ex, kc, key, out, top)?;
        out.meta = r.out;
        Ok(())
    }

    // ── fused sums ───────────────────────────────────────────────────────────────────────────────────────────────

    /// `acc += sum_j a[j] * pt[j]` with the loop's bookkeeping (`mult_pt` then `add_inplace` per term). When every
    /// term needs no prelude and lands on `acc`'s meta, the products accumulate on the device without per-term
    /// metadata work; otherwise the loop runs as written.
    pub fn mult_add_many<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, acc: &mut Ct<'_, W>, a: &[&Ct<'_, W>], pt: &[Pt<'_>]) -> Result<()> {
        lens("mult_add_many", a.len(), &[pt.len()])?;
        let k = acc.k();
        if k > ex.scratch.k {
            return Err(Error::Mem { knob: "MemCfg::scratch_cts", value: ex.scratch.k, msg: format!("mult_add_many of {k} cts") });
        }
        for (x, p) in a.iter().zip(pt) {
            Self::check_k("mult_add_many", x.k(), k)?;
            let r = sh.mult_pt(&x.meta, &p.meta)?;
            let fast = r.pre.is_empty() && sh.add(&acc.meta, &r.out)?.pre.is_empty();
            if !fast {
                // the general loop: product into scratch with its recipe, then the add recipe
                return self.mult_add_many_loop(ctx, ex, sh, acc, a, pt);
            }
        }
        let s = &ex.compute;
        for (x, p) in a.iter().zip(pt) {
            let r = sh.mult_pt(&x.meta, &p.meta)?;
            let top = self.top(ctx.shape, &r.out)?;
            let tmp = ex.scratch.w();
            Self::mul_table(ctx, s, tmp, x.buf.r(), p.rows, top + 1, k)?;
            let o = acc.buf.w();
            Self::ew(ctx, s, "ew_add", o, o, tmp, rows_q(&ctx.lay, top), k, 2, false, (std::ptr::null(), std::ptr::null()))?;
            acc.meta = sh.add(&acc.meta, &r.out)?.out;
        }
        Ok(())
    }

    fn mult_add_many_loop<M: DevStrategy<W>>(&mut self, _ctx: &Context<'_, W, M>, _ex: &mut Executor<'_, W>, _sh: &Shadow, _acc: &mut Ct<'_, W>, _a: &[&Ct<'_, W>], _pt: &[Pt<'_>]) -> Result<()> {
        // the product needs its own ciphertext slot (scratch holds only the operand copies); the eager session
        // (pc-py) runs the loop with pool slots
        Err(shape("mult_add_many", "terms need preludes: run mult_pt + add_inplace with pool slots (fused path needs equal metas)"))
    }

    /// `out = sum_j a[j] * c[j]` (`mult_scalar` + `add_inplace` per term): when every term shares level and degree
    /// 1 the products are one wsum launch per 8 terms (`evalLinearWSum` form, ElemForEvalMult at the common top).
    pub fn mult_scalar_sum<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, out: &mut Ct<'_, W>, a: &[&Ct<'_, W>], c: &[f64]) -> Result<()> {
        lens("mult_scalar_sum", a.len(), &[c.len()])?;
        if a.is_empty() {
            return Err(shape("mult_scalar_sum", "no terms"));
        }
        let m0 = a[0].meta;
        if a.iter().any(|x| x.meta.level != m0.level || x.meta.deg != 1) {
            return Err(shape("mult_scalar_sum", "terms must share level at degree 1 (realize first)"));
        }
        let mut r = sh.mult_scalar(&m0, c[0])?;
        for (x, &ci) in a.iter().zip(c).skip(1) {
            let ri = sh.mult_scalar(&x.meta, ci)?;
            r.out = sh.add(&r.out, &ri.out)?.out;
        }
        let top = self.top(ctx.shape, &m0)?;
        let rows: Vec<Vec<u64>> = c.iter().map(|&ci| ctx.scale.elem_for_eval_mult(top as i32, ci, -1)).collect::<std::result::Result<_, _>>()?;
        self.wsum_launches(ctx, &ex.compute, out, a, &rows, top)?;
        out.meta = r.out;
        Ok(())
    }

    /// FIDESlib `evalLinearWSumMutable`: `out = sum_j w[j] * a[j]`, every input at degree 1; weights are
    /// `ElemForEvalMult(top, w_j, top_j)` at the first input's top (richer inputs are scaled onto it); out at degree
    /// 2 with NoiseFactor `sf(top)^2`.
    pub fn eval_linear_wsum<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, out: &mut Ct<'_, W>, a: &[&Ct<'_, W>], w: &[f64]) -> Result<()> {
        lens("eval_linear_wsum", a.len(), &[w.len()])?;
        if a.is_empty() {
            return Err(shape("eval_linear_wsum", "no inputs"));
        }
        let top = self.top(ctx.shape, &a[0].meta)?;
        let mut rows = Vec::with_capacity(a.len());
        let mut tg = tag::t_mult_scalar(&a[0].meta.tag, w[0]);
        for (j, (x, &wj)) in a.iter().zip(w).enumerate() {
            let tj = self.top(ctx.shape, &x.meta)?;
            if x.meta.deg != 1 || tj < top {
                return Err(shape("eval_linear_wsum", format!("input {j}: degree {} top {tj} (need degree 1, top >= {top})", x.meta.deg)));
            }
            rows.push(ctx.scale.elem_for_eval_mult(top as i32, wj, tj as i32)?);
            if j > 0 {
                tg = tag::t_add(&tg, &tag::t_mult_scalar(&x.meta.tag, wj));
            }
        }
        self.wsum_launches(ctx, &ex.compute, out, a, &rows, top)?;
        let sf = ctx.scale.sf(top as i32)?;
        let slots = a.iter().map(|x| x.meta.slots).max().unwrap_or(0);
        out.meta = CtMeta { level: a[0].meta.level, deg: 2, nf: sf * sf, slots, tag: tg };
        Ok(())
    }

    /// `out = sum_j rows[j] * a[j]` in launches of up to 8 sources (later launches carry `out` with weight 1).
    fn wsum_launches<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, out: &mut Ct<'_, W>, a: &[&Ct<'_, W>], rows: &[Vec<u64>], top: usize) -> Result<()> {
        let k = out.k();
        for x in a {
            Self::check_k("wsum", x.k(), k)?;
        }
        let ones: Vec<u64> = vec![1; self.primes.len()];
        let o = out.buf.w();
        let mut j = 0;
        let mut first = true;
        while j < a.len() {
            let room = if first { WSUM_MAX } else { WSUM_MAX - 1 };
            let end = (j + room).min(a.len());
            let mut src = [PolyRef::default(); WSUM_MAX];
            let mut w: Vec<&[u64]> = Vec::with_capacity(WSUM_MAX);
            let mut n = 0;
            if !first {
                src[0] = o;
                w.push(&ones);
                n = 1;
            }
            for i in j..end {
                src[n] = a[i].buf.r();
                w.push(&rows[i]);
                n += 1;
            }
            let (wv, ws) = self.ring.push_rows(s, &self.primes, &w)?;
            let args = WsumArgs { out: o, src, w: wv, ws, pr: ctx.pr(), nsrc: n as i32, rowstride: self.primes.len() as i32, n: (top + 1) as i32, k: k as i32 };
            // SAFETY: element-wise; when `out` is a source each thread reads it before writing the same index
            launch_result("wsum", unsafe { (ctx.table.wsum)(&args, s.raw()) })?;
            first = false;
            j = end;
        }
        Ok(())
    }

    // ── level ops ────────────────────────────────────────────────────────────────────────────────────────────────

    /// Explicit rescale (`Ciphertext::rescale`).
    pub fn rescale<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, x: &mut [Ct<'_, W>]) -> Result<()> {
        for c in x.iter_mut() {
            let r = sh.rescale(&c.meta)?;
            let top = self.top(ctx.shape, &c.meta)?;
            self.rescaler.rescale(ctx, &ex.compute, &mut c.buf, top, None)?;
            c.meta = r.out;
        }
        Ok(())
    }
    /// Pay a pending rescale now (the plan's `rescale_after` anchor / the wrapper's realize).
    pub fn realize<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, x: &mut [Ct<'_, W>]) -> Result<()> {
        for c in x.iter_mut() {
            let r = sh.realize(&c.meta)?;
            let mut t = Track { meta: c.meta };
            self.run_pre(ctx, &ex.compute, &mut c.buf, &mut t, &r.pre, Operand::A)?;
            Self::finish(c, &r, &t)?;
        }
        Ok(())
    }
    /// `out = realize(a)` out of place: the realize rescales from `a` into `out` (no copy first), or
    /// copies when the realize is a no-op on the data.
    pub fn realize_to<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, sh: &Shadow, out: &mut Ct<'_, W>, a: &Ct<'_, W>) -> Result<()> {
        Self::check_k("realize", out.k(), a.k())?;
        let r = sh.realize(&a.meta)?;
        let mut t = Track { meta: a.meta };
        let s = &ex.compute;
        let v = if self.cfg.out_of_place {
            self.run_pre_from(ctx, s, &mut out.buf, Some(a.buf.r()), &mut t, &r.pre, Operand::A)?
        } else {
            let top = self.top(ctx.shape, &a.meta)?;
            Self::copy_ct(ctx, s, out.buf.w(), a.buf.r(), a.k(), top)?;
            self.run_pre(ctx, s, &mut out.buf, &mut t, &r.pre, Operand::A)?;
            out.buf.r()
        };
        if v.base != out.buf.r().base {
            let top = self.top(ctx.shape, &t.meta)?;
            Self::copy_ct(ctx, s, out.buf.w(), v, a.k(), top)?;
        }
        Self::finish(out, &r, &t)
    }
    /// `drop_to_level(t)`: `max(cur, t)`, degree and scale kept; rows above the new top become dead (no data moves).
    pub fn drop_to_level(&self, sh: &Shadow, x: &mut [Ct<'_, W>], level: u32) {
        for c in x.iter_mut() {
            c.meta = sh.drop_to_level(&c.meta, level).out;
        }
    }

    /// The op layer's context layout (tests).
    pub fn layout(&self) -> Layout {
        self.lay
    }
}

fn lens(what: &'static str, n: usize, others: &[usize]) -> Result<()> {
    if others.iter().any(|&m| m != n) {
        return Err(shape(what, format!("argument slices differ in length: {n} vs {others:?}")));
    }
    Ok(())
}
