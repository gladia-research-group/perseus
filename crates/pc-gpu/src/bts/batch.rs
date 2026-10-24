//! K-batching of bootstraps across separate ciphertexts (K ciphertexts per launch).
//!
//! The routes already run a `Ct` of `k` ciphertexts with one set of launches (K-batched CtS, whose
//! `cts_fused` shares key and plaintext reads across K, the 2K EvalMod chains, the raise, SparseB, SPRU). What the
//! compiler hands over are sibling refreshes living in separate buffers; `BtsBatch` groups them by identical
//! (route, entry level, degree, scale) and `BatchSlots` gathers a group into one fixed k_max-ciphertext buffer, runs
//! the route once, and scatters the outputs back. The fixed buffer is also what makes a captured graph
//! address-stable (graph.rs).
//!
//! Mixed-level (or mixed-degree / mixed-scale) batches are refused with `Error::Shape`: the compiler never forms them
//! (pc-program's sibling rule), and the route recipes are per metadata.
use super::{BtsCx, BtsRoute};
use crate::context::{DevStrategy, DevWord};
use crate::ct::Ct;
use crate::error::{shape, Error, Result};
use crate::ks::{self, View};
use crate::poly::{CtBuf, Layout};
use pc_cuda::{Arena, Compute};
use pc_shadow::{CtMeta, Route};

/// What a batch shares: the route and the input metadata the stage recipes depend on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BatchKey {
    pub route: Route,
    pub level: u32,
    pub deg: u8,
    /// Scale (bit pattern of `CtMeta::nf`): the raise adjust constant is a function of it.
    pub nf_bits: u64,
}

impl BatchKey {
    pub fn of(route: &Route, m: &CtMeta) -> BatchKey {
        BatchKey { route: *route, level: m.level, deg: m.deg, nf_bits: m.nf.to_bits() }
    }
}

/// Sibling refreshes of one route at one entry metadata: run as one K-batch (`k()` ciphertexts per launch).
pub struct BtsBatch<'b, 'a, W: pc_cuda::Pod> {
    pub route: Route,
    pub cts: Vec<&'b mut Ct<'a, W>>,
}

impl<'b, 'a, W: pc_cuda::Pod> BtsBatch<'b, 'a, W> {
    /// A batch of `cts` on `route`; every input must share level, degree and scale.
    pub fn new(route: Route, cts: Vec<&'b mut Ct<'a, W>>) -> Result<Self> {
        let first = cts.first().ok_or_else(|| shape("BtsBatch", "empty batch"))?;
        let key = BatchKey::of(&route, &first.meta);
        if let Some(c) = cts.iter().find(|c| BatchKey::of(&route, &c.meta) != key) {
            return Err(shape(
                "BtsBatch",
                format!(
                    "mixed batch: level {} deg {} nf {:e} next to level {} deg {} nf {:e} (siblings must share entry metadata)",
                    c.meta.level, c.meta.deg, c.meta.nf, first.meta.level, first.meta.deg, first.meta.nf
                ),
            ));
        }
        Ok(BtsBatch { route, cts })
    }
    /// Ciphertexts of the batch (sum of the members' k).
    pub fn k(&self) -> usize {
        self.cts.iter().map(|c| c.k()).sum()
    }
    pub fn key(&self) -> BatchKey {
        BatchKey::of(&self.route, &self.cts[0].meta)
    }
    pub fn meta(&self) -> CtMeta {
        self.cts[0].meta
    }
}

/// Group `(route, ct)` refreshes into batches of at most `k_max` ciphertexts, by identical `BatchKey`, in order
/// of first appearance (a member is never split across batches).
pub fn group<'b, 'a, W: pc_cuda::Pod>(items: Vec<(Route, &'b mut Ct<'a, W>)>, k_max: usize) -> Result<Vec<BtsBatch<'b, 'a, W>>> {
    let mut open: Vec<(BatchKey, Vec<BtsBatch<'b, 'a, W>>)> = Vec::new();
    for (route, ct) in items {
        if ct.k() > k_max {
            return Err(Error::Mem { knob: "bts.k_max", value: k_max, msg: format!("a member of {} ciphertexts exceeds the batch bound", ct.k()) });
        }
        let key = BatchKey::of(&route, &ct.meta);
        let slot = match open.iter().position(|(k, _)| *k == key) {
            Some(i) => i,
            None => {
                open.push((key, Vec::new()));
                open.len() - 1
            }
        };
        let list = &mut open[slot].1;
        match list.last_mut() {
            Some(b) if b.k() + ct.k() <= k_max => b.cts.push(ct),
            _ => list.push(BtsBatch { route, cts: vec![ct] }),
        }
    }
    Ok(open.into_iter().flat_map(|(_, l)| l).collect())
}

/// The fixed gather/scatter buffer of `k_max` ciphertexts (MemPlan section "bts.batch"; setup-carved).
pub struct BatchSlots<'a, W: DevWord> {
    pub x: Ct<'a, W>,
    pub k_max: usize,
}

impl<'a, W: DevWord> BatchSlots<'a, W> {
    pub fn bytes(lay: &Layout, k_max: usize) -> usize {
        Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(lay, k_max))
    }
    pub fn new(arena: &'a Arena, lay: Layout, k_max: usize) -> Result<Self> {
        Ok(BatchSlots { x: Ct::new(CtMeta::new(0, 1), CtBuf::new(arena, "bts.batch", lay, k_max)?), k_max })
    }

    /// Copy the live rows (Q `0..=top`, both parts) of every member into consecutive slots; the slot `Ct` takes the
    /// batch's metadata and k.
    pub fn gather<M: DevStrategy<W>>(&mut self, cx: &BtsCx<'_, 'a, W, M>, b: &BtsBatch<'_, 'a, W>) -> Result<()> {
        let k = b.k();
        if k > self.k_max {
            return Err(Error::Mem { knob: "bts.k_max", value: self.k_max, msg: format!("batch of {k} ciphertexts") });
        }
        let top = b.cts[0].top(cx.ctx.shape)?;
        let mut at = 0;
        for c in &b.cts {
            ks::ew(cx.ctx, &cx.ex.compute, ks::Ew::Copy, View::of(&self.x.buf, at), View::of(&c.buf, 0), None, cx.ctx.lay.rs_q(top), c.k(), 2)?;
            at += c.k();
        }
        self.x.buf.k = k;
        self.x.meta = b.meta();
        Ok(())
    }

    /// Copy the slot outputs (live rows of the slot metadata) back into the members; they take its metadata.
    pub fn scatter<M: DevStrategy<W>>(&self, cx: &BtsCx<'_, 'a, W, M>, b: &mut BtsBatch<'_, 'a, W>) -> Result<()> {
        let top = self.x.top(cx.ctx.shape)?;
        let mut at = 0;
        for c in b.cts.iter_mut() {
            ks::ew(cx.ctx, &cx.ex.compute, ks::Ew::Copy, View::of(&c.buf, 0), View::of(&self.x.buf, at), None, cx.ctx.lay.rs_q(top), c.k(), 2)?;
            at += c.k();
            c.meta = self.x.meta;
        }
        Ok(())
    }
}

/// Eager K-batched refresh of `b` on `route`: one member runs in place; several are gathered into `slots`, run as
/// one K-batch and scattered back.
pub fn run_batch<'a, W: DevWord, M: DevStrategy<W>>(
    route: &dyn BtsRoute<'a, W, M>,
    cx: &mut BtsCx<'_, 'a, W, M>,
    slots: &mut BatchSlots<'a, W>,
    b: &mut BtsBatch<'_, 'a, W>,
) -> Result<()> {
    let r = b.route;
    if b.cts.len() == 1 {
        return route.run(cx, std::slice::from_mut(&mut *b.cts[0]), &r);
    }
    slots.gather(cx, b)?;
    route.run(cx, std::slice::from_mut(&mut slots.x), &r)?;
    slots.scatter(cx, b)
}

/// Copy the live rows (Q `0..=top` at `src`'s metadata, both parts) of single ciphertext `src` into ciphertext `at` of
/// the K-ciphertext buffer `dst` (a gather), on the compute lane.
pub fn put<W: DevWord, M: DevStrategy<W>>(ctx: &crate::Context<'_, W, M>, s: &pc_cuda::Stream<Compute>, dst: &mut Ct<'_, W>, at: usize, src: &Ct<'_, W>) -> Result<()> {
    let top = src.top(ctx.shape)?;
    if src.k() != 1 || at >= dst.buf.buf.len() / ctx.lay.ct_words() || src.buf.lay != ctx.lay || dst.buf.lay != ctx.lay {
        return Err(shape("batch put", format!("ciphertext {at} of a {}-ciphertext buffer", dst.buf.buf.len() / ctx.lay.ct_words())));
    }
    ks::ew(ctx, s, ks::Ew::Copy, View::of(&dst.buf, at), View::of(&src.buf, 0), None, ctx.lay.rs_q(top), 1, 2)
}

/// Copy ciphertext `at` of the K-ciphertext buffer `src` (its live rows at `meta`) into single ciphertext `dst`, which
/// takes `meta` (a scatter), on the compute lane.
pub fn take<W: DevWord, M: DevStrategy<W>>(ctx: &crate::Context<'_, W, M>, s: &pc_cuda::Stream<Compute>, dst: &mut Ct<'_, W>, src: &Ct<'_, W>, at: usize, meta: CtMeta) -> Result<()> {
    let top = pc_shadow::meta::top_limb(ctx.shape, meta.level)? as usize;
    if dst.k() != 1 || at >= src.k() || src.buf.lay != ctx.lay || dst.buf.lay != ctx.lay {
        return Err(shape("batch take", format!("ciphertext {at} of a {}-ciphertext batch", src.k())));
    }
    ks::ew(ctx, s, ks::Ew::Copy, View::of(&dst.buf, 0), View::of(&src.buf, at), None, ctx.lay.rs_q(top), 1, 2)?;
    dst.meta = meta;
    Ok(())
}
