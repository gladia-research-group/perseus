//! Rotations: seam `Rotator` and the rotation ops, bit-exact with FIDESlib's single-GPU path.
//!
//! - `Plain`: ModUp(c1) -> `ks_dot` with `P * c0` added, output at the automorphism's slot -> ModDown. Key for step r
//!   = OpenFHE's index g = 5^r mod 2N; the kernel scatters by g^-1 = 5^(2N - r) (`pc_core::galois`).
//! - `Hoisted`: one ModUp shared by the steps, each output ModDowned separately (bit-identical to plain rotations).
//! - `rotate_sum`: rotations summed in the extended basis, ONE ModDown (FIDESlib `rotateSumHoisted`).
//! - `LadderFused`: `rotate_and_sum(x, start, stop)` = the perseus C++ ladder (`src/bindings/ops.cu`), same op stream.
//!
//! Key tables: every op takes a `KeyGroup` built at setup; `RotKeys` looks groups up. Details: docs/ARCHITECTURE.md
//! "pc-gpu rotations".
use crate::context::{Context, DevStrategy, DevWord};
use crate::ct::Ct;
use crate::error::{shape, Result};
use crate::executor::Executor;
use crate::ks::{self, ew, DotIo, DotRows, Ew, KeyGroup, KsPlans, KskSource, View};
use pc_shadow::Shadow;

/// Seam `Rotator`: how a set of rotations of one ciphertext is executed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rotator {
    /// One key switch per rotation.
    Plain,
    /// One shared decomposition for all the steps of an op.
    Hoisted,
    /// Rotate-and-sum ladders as hoisted radix groups summed before one ModDown.
    LadderFused,
}

/// Key-switching configuration an op runs under.
#[derive(Clone, Copy)]
pub struct KsCfg<'p, 'a> {
    pub plans: &'p KsPlans<'a>,
    pub source: KskSource,
}

/// Lookup of setup-built key groups by their exact step list.
pub trait RotKeys {
    fn group(&self, steps: &[i64]) -> Option<&KeyGroup<'_>>;
}

fn top_of<W: DevWord, M: DevStrategy<W>>(ctx: &Context<'_, W, M>, x: &Ct<'_, W>) -> Result<usize> {
    x.top(ctx.shape)
}

fn same_shape<W: DevWord>(x: &Ct<'_, W>, out: &Ct<'_, W>, what: &'static str) -> Result<()> {
    if x.k() != out.k() || x.buf.lay != out.buf.lay {
        return Err(shape(what, format!("input has {} cts, output {}", x.k(), out.k())));
    }
    Ok(())
}

pub(crate) fn key_index(group: &KeyGroup<'_>, g: u64) -> Result<usize> {
    group.galois.iter().position(|&x| x == g).ok_or_else(|| shape("rotation key", format!("galois element {g} not in the key group {:?}", group.galois)))
}

/// Inner products of the k ciphertexts' ModUp (in `ex.ks.digits`) against keys `[r0, r0 + r)` of `group`, honouring
/// the key source (Regen rotations and packed / banded keys: in place through `ks_dot_key`).
#[allow(clippy::too_many_arguments)]
fn dots<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    ex: &mut Executor<'_, W>,
    kc: KsCfg<'_, '_>,
    group: &KeyGroup<'_>,
    r0: usize,
    r: usize,
    k: usize,
    top: usize,
    io: DotIo,
) -> Result<()> {
    if kc.source == KskSource::Cached && !group.needs_stage() && group.has_a() {
        return ks::dot(ctx, &ex.compute, kc.plans, &ex.ks, group, r0, r, k, top, DotRows::All, io, false);
    }
    // Regen, or keys whose `b` is packed / banded or whose `a` is not resident: read in place (`ks_dot_key`, every key
    // of the range in one launch per form) when seeded, else one key at a time through staging
    if group.in_place() {
        return ks::dot(ctx, &ex.compute, kc.plans, &ex.ks, group, r0, r, k, top, DotRows::All, io, true);
    }
    let ndig = kc.plans.ndig(top)?;
    for j in 0..r {
        ks::stage_key(ctx, &ex.compute, kc.plans, &mut ex.ks, group, r0 + j, ndig, true)?;
        let out = (io.out as *mut W).wrapping_add(j * io.out_rot_stride as usize) as *mut std::ffi::c_void;
        ks::dot(ctx, &ex.compute, kc.plans, &ex.ks, group, r0 + j, 1, k, top, DotRows::All, DotIo { out, ..io }, true)?;
    }
    Ok(())
}

/// The dot I/O of `k` ciphertexts of `x` written at `out` (extended layout, (r, c) at r * rot_stride + c * ct).
fn io_of<W: DevWord>(x: &Ct<'_, W>, out: View, rot_stride_cts: usize) -> DotIo {
    let lay = x.buf.lay;
    let v = View::of(&x.buf, 0);
    DotIo {
        c1: v.part::<W>(1).base,
        c0: v.base,
        ct_stride: v.ct_stride,
        out: out.base,
        part_stride: lay.poly_words() as u64,
        out_ct_stride: lay.ct_words() as u64,
        out_rot_stride: (rot_stride_cts * lay.ct_words()) as u64,
        c0_mode: 1,
    }
}

fn copy_q<W: DevWord, M: DevStrategy<W>>(ctx: &Context<'_, W, M>, ex: &Executor<'_, W>, x: &Ct<'_, W>, out: &mut Ct<'_, W>, top: usize) -> Result<()> {
    let k = x.k();
    ew(ctx, &ex.compute, Ew::Copy, View::of(&out.buf, 0), View::of(&x.buf, 0), None, ctx.lay.rs_q(top), k, 2)
}

/// Rotation by `step` slots (left) through the key of `group` with element 5^step: `out = rot(x, step)`.
#[allow(clippy::too_many_arguments)]
pub fn rotate<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    ex: &mut Executor<'_, W>,
    kc: KsCfg<'_, '_>,
    sh: &Shadow,
    group: &KeyGroup<'_>,
    x: &Ct<'_, W>,
    step: i64,
    out: &mut Ct<'_, W>,
) -> Result<()> {
    let g = pc_core::galois::rotation_galois(step, ctx.lay.n)?;
    switch_auto(ctx, ex, kc, group, x, g, out)?;
    out.meta = sh.rotate(&x.meta, step).out;
    Ok(())
}

/// Complex conjugation (key 2N - 1).
pub fn conjugate<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    ex: &mut Executor<'_, W>,
    kc: KsCfg<'_, '_>,
    sh: &Shadow,
    group: &KeyGroup<'_>,
    x: &Ct<'_, W>,
    out: &mut Ct<'_, W>,
) -> Result<()> {
    switch_auto(ctx, ex, kc, group, x, pc_core::galois::conj_galois(ctx.lay.n)?, out)?;
    out.meta = sh.conjugate(&x.meta).out;
    Ok(())
}

/// out = sigma_g(x) key-switched back (g = 1: a copy).
fn switch_auto<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    ex: &mut Executor<'_, W>,
    kc: KsCfg<'_, '_>,
    group: &KeyGroup<'_>,
    x: &Ct<'_, W>,
    g: u64,
    out: &mut Ct<'_, W>,
) -> Result<()> {
    same_shape(x, out, "rotate")?;
    let (k, top) = (x.k(), top_of(ctx, x)?);
    if g == 1 {
        return copy_q(ctx, ex, x, out, top);
    }
    let r = key_index(group, g)?;
    ks::modup(ctx, &ex.compute, kc.plans, &mut ex.ks, &mut ex.bconv_work, &x.buf, 0, k, 1, top)?;
    let io = io_of(x, View::of(&out.buf, 0), 0);
    dots(ctx, ex, kc, group, r, 1, k, top, io)?;
    ks::moddown_inplace(ctx, &ex.compute, kc.plans, &mut out.buf, &mut ex.ks.conv, k, top)
}

/// Rotations of many single ciphertexts of one level by one step, `outs[i] = rot(xs[i], step)`: up to `batch` at a
/// time (bounded by the key-switching batch and half the executor scratch) copied into the scratch, one ModUp, one dot
/// launch reading the key once, one ModDown over the batch; bit-identical to `rotate` one by one.
#[allow(clippy::too_many_arguments)]
pub fn rotate_many<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    ex: &mut Executor<'_, W>,
    kc: KsCfg<'_, '_>,
    sh: &Shadow,
    group: &KeyGroup<'_>,
    xs: &[&Ct<'_, W>],
    step: i64,
    outs: &mut [&mut Ct<'_, W>],
    batch: usize,
) -> Result<()> {
    let count = xs.len();
    if outs.len() != count {
        return Err(shape("rotate_many", format!("{count} inputs, {} outputs", outs.len())));
    }
    if count == 0 {
        return Ok(());
    }
    if xs.iter().any(|x| x.k() != 1 || x.meta != xs[0].meta || x.buf.lay != ctx.lay) || outs.iter().any(|o| o.k() != 1 || o.buf.lay != ctx.lay) {
        return Err(shape("rotate_many", "single full-layout ciphertexts of one level"));
    }
    let lay = ctx.lay;
    let top = top_of(ctx, xs[0])?;
    let meta = sh.rotate(&xs[0].meta, step).out;
    let g = pc_core::galois::rotation_galois(step, lay.n)?;
    if g == 1 {
        for (x, o) in xs.iter().zip(outs.iter_mut()) {
            copy_q(ctx, ex, x, o, top)?;
            o.meta = meta;
        }
        return Ok(());
    }
    let r = key_index(group, g)?;
    let (cw, pw) = (lay.ct_words(), lay.poly_words());
    let km = batch.min(ex.ks.k_max).min(kc.plans.k_max).min(ex.scratch.k / 2).max(1);
    let rs = lay.rs_q(top);
    let at = |v: View, c: usize| View { base: (v.base as *mut W).wrapping_add(c * cw) as *mut std::ffi::c_void, ..v };
    let mut u0 = 0;
    while u0 < count {
        let k = km.min(count - u0);
        // inputs into scratch[km, km + k); the dot's extended outputs into scratch[0, k), ModDowned in place
        let s0 = View::of(&ex.scratch, 0);
        for u in 0..k {
            ks::ew(ctx, &ex.compute, Ew::Copy, at(s0, km + u), View::of(&xs[u0 + u].buf, 0), None, rs, 1, 2)?;
        }
        ks::modup(ctx, &ex.compute, kc.plans, &mut ex.ks, &mut ex.bconv_work, &ex.scratch, km, k, 1, top)?;
        let inp = at(s0, km);
        let io = DotIo {
            c1: inp.part::<W>(1).base,
            c0: inp.base,
            ct_stride: cw as u64,
            out: s0.base,
            part_stride: pw as u64,
            out_ct_stride: cw as u64,
            out_rot_stride: 0,
            c0_mode: 1,
        };
        dots(ctx, ex, kc, group, r, 1, k, top, io)?;
        ks::moddown_inplace(ctx, &ex.compute, kc.plans, &mut ex.scratch, &mut ex.ks.conv, k, top)?;
        for u in 0..k {
            let o = &mut outs[u0 + u];
            ks::ew(ctx, &ex.compute, Ew::Copy, View::of(&o.buf, 0), at(s0, u), None, rs, 1, 2)?;
            o.meta = meta;
        }
        u0 += k;
    }
    Ok(())
}

/// A tree of rotations of `x` in one pass: `outs[0] = x`, then round by round the list so far followed by its rotation
/// by each of the round's nonzero steps (`rounds[t][j]` = the step's key group and the step), so `outs` holds
/// prod_t (1 + rounds[t].len()) ciphertexts. Up to `batch` inputs of a round at a time are gathered into the scratch
/// for ONE ModUp shared by the round's steps; each step is one dot launch and one ModDown over the batch, scattered
/// to the outputs (any slots that hold `x`'s rows). Every output is bit-identical to the plain rotations.
#[allow(clippy::too_many_arguments)]
pub fn rotate_tree<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    ex: &mut Executor<'_, W>,
    kc: KsCfg<'_, '_>,
    sh: &Shadow,
    rounds: &[Vec<(&KeyGroup<'_>, i64)>],
    x: &Ct<'_, W>,
    outs: &mut [&mut Ct<'_, W>],
    batch: usize,
) -> Result<()> {
    use std::ffi::c_void;
    let lay = ctx.lay;
    let top = top_of(ctx, x)?;
    let count = rounds.iter().map(|r| r.len() + 1).product::<usize>();
    if x.k() != 1 || x.buf.lay.nq <= top || outs.len() != count || outs.iter().any(|o| o.k() != 1 || o.buf.lay.nq <= top) {
        return Err(shape("rotate_tree", format!("{} outputs for a tree of {count}, single ciphertexts holding {} rows", outs.len(), top + 1)));
    }
    let mut keys = Vec::with_capacity(rounds.len());
    for round in rounds {
        let mut ks = Vec::with_capacity(round.len());
        for &(group, st) in round {
            let g = pc_core::galois::rotation_galois(st, lay.n)?;
            if g == 1 {
                return Err(shape("rotate_tree", "a round's steps are nonzero"));
            }
            ks.push((group, key_index(group, g)?, st));
        }
        keys.push(ks);
    }
    let (cw, pw) = (lay.ct_words(), lay.poly_words());
    let km = batch.min(ex.ks.k_max).min(kc.plans.k_max).min(ex.scratch.k / 2).max(1);
    let nul = std::ptr::null::<c_void>();
    let s0 = View::of(&ex.scratch, 0).base;
    let off = |words: usize| (s0 as *mut W).wrapping_add(words) as *mut c_void;
    let part = |c: &Ct<'_, W>, p: usize| View::of(&c.buf, 0).part::<W>(p).base as *const c_void;
    ks::ew_many(ctx, &ex.compute, &[[part(outs[0], 0), part(x, 0), nul, nul], [part(outs[0], 1), part(x, 1), nul, nul]], top + 1)?;
    outs[0].meta = x.meta;
    let mut have = 1;
    for round in &keys {
        let mut q0 = 0;
        while q0 < have {
            let k = km.min(have - q0);
            // the inputs into scratch[km, km + k); a step's extended dot into scratch[0, k), ModDowned in place
            let polys: Vec<_> = (0..2 * k).map(|i| [off((km + i / 2) * cw + (i % 2) * pw) as *const c_void, part(outs[q0 + i / 2], i % 2), nul, nul]).collect();
            ks::ew_many(ctx, &ex.compute, &polys, top + 1)?;
            ks::modup(ctx, &ex.compute, kc.plans, &mut ex.ks, &mut ex.bconv_work, &ex.scratch, km, k, 1, top)?;
            for (j, &(group, r, st)) in round.iter().enumerate() {
                let io = DotIo {
                    c1: off(km * cw + pw),
                    c0: off(km * cw),
                    ct_stride: cw as u64,
                    out: s0,
                    part_stride: pw as u64,
                    out_ct_stride: cw as u64,
                    out_rot_stride: 0,
                    c0_mode: 1,
                };
                dots(ctx, ex, kc, group, r, 1, k, top, io)?;
                ks::moddown_inplace(ctx, &ex.compute, kc.plans, &mut ex.scratch, &mut ex.ks.conv, k, top)?;
                let o0 = (j + 1) * have + q0;
                let polys: Vec<_> = (0..2 * k).map(|i| [part(outs[o0 + i / 2], i % 2), off((i / 2) * cw + (i % 2) * pw) as *const c_void, nul, nul]).collect();
                ks::ew_many(ctx, &ex.compute, &polys, top + 1)?;
                for u in 0..k {
                    outs[o0 + u].meta = sh.rotate(&outs[q0 + u].meta, st).out;
                }
            }
            q0 += k;
        }
        have *= round.len() + 1;
    }
    Ok(())
}

/// One round of an expansion by rotations (a ciphertext split into its rotation-sum and -difference halves, the
/// differences shifted by a monomial) on the ciphertexts `xs` of one level: with
/// `y_i = rot(x_i, step)` where `rot[i]` and `y_i = x_i` elsewhere, `sums[i] = x_i + y_i` and, where `diffs[i]` is
/// given, `diffs[i] = (x_i - y_i) * t` (`t` = an evaluation-form table over rows 0..=top, the round's monomial). Up to
/// `batch` rotated inputs are gathered into the scratch for ONE ModUp, dot launch and ModDown; the sums and the
/// differences read the rotations there, one launch each per batch (any slots that hold the rows). Every output is
/// bit-identical to `rotate`, `add`, `sub` and `mult_pt_exact` one by one.
#[allow(clippy::too_many_arguments)]
pub fn expand_round<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    ex: &mut Executor<'_, W>,
    kc: KsCfg<'_, '_>,
    sh: &Shadow,
    group: &KeyGroup<'_>,
    step: i64,
    t: *const std::ffi::c_void,
    xs: &[&Ct<'_, W>],
    rot: &[bool],
    sums: &mut [&mut Ct<'_, W>],
    diffs: &mut [Option<&mut Ct<'_, W>>],
    batch: usize,
) -> Result<()> {
    use std::ffi::c_void;
    let (lay, n) = (ctx.lay, xs.len());
    if n == 0 {
        return Ok(());
    }
    if rot.len() != n || sums.len() != n || diffs.len() != n {
        return Err(shape("expand_round", format!("{n} inputs, {} flags, {} sums, {} differences", rot.len(), sums.len(), diffs.len())));
    }
    let top = top_of(ctx, xs[0])?;
    let holds = |c: &Ct<'_, W>| c.k() == 1 && c.buf.lay.nq > top;
    for i in 0..n {
        if top_of(ctx, xs[i])? != top || !holds(xs[i]) || !holds(sums[i]) || !diffs[i].as_deref().is_none_or(holds) {
            return Err(shape("expand_round", format!("single ciphertexts of one level holding {} rows", top + 1)));
        }
        if diffs[i].is_some() && !rot[i] {
            return Err(shape("expand_round", "a difference needs its input rotated"));
        }
    }
    let g = pc_core::galois::rotation_galois(step, lay.n)?;
    if g == 1 {
        return Err(shape("expand_round", "the step is nonzero"));
    }
    let r = key_index(group, g)?;
    let mut metas = Vec::with_capacity(n);
    for i in 0..n {
        let ym = if rot[i] { sh.rotate(&xs[i].meta, step).out } else { xs[i].meta };
        let (a, d) = (sh.add(&xs[i].meta, &ym)?, sh.sub(&xs[i].meta, &ym)?);
        if !a.pre.is_empty() || !d.pre.is_empty() {
            return Err(shape("expand_round", "the sum needs an adjustment"));
        }
        metas.push((a.out, d.out));
    }
    let (cw, pw) = (lay.ct_words(), lay.poly_words());
    let km = batch.min(ex.ks.k_max).min(kc.plans.k_max).min(ex.scratch.k / 2).max(1);
    let nul = std::ptr::null::<c_void>();
    let s0 = View::of(&ex.scratch, 0).base;
    let off = |words: usize| (s0 as *mut W).wrapping_add(words) as *mut c_void;
    let part = |c: &Ct<'_, W>, p: usize| View::of(&c.buf, 0).part::<W>(p).base as *const c_void;
    // not rotated: x + x
    let mut polys = Vec::new();
    for i in (0..n).filter(|&i| !rot[i]) {
        for p in 0..2 {
            polys.push([part(sums[i], p), part(xs[i], p), part(xs[i], p), nul]);
        }
    }
    ks::ew_many(ctx, &ex.compute, &polys, top + 1)?;
    let live: Vec<usize> = (0..n).filter(|&i| rot[i]).collect();
    for ch in live.chunks(km) {
        let k = ch.len();
        // the inputs into scratch[km, km + k); their rotations into scratch[0, k), ModDowned in place
        let polys: Vec<_> = (0..2 * k).map(|i| [off((km + i / 2) * cw + (i % 2) * pw) as *const c_void, part(xs[ch[i / 2]], i % 2), nul, nul]).collect();
        ks::ew_many(ctx, &ex.compute, &polys, top + 1)?;
        ks::modup(ctx, &ex.compute, kc.plans, &mut ex.ks, &mut ex.bconv_work, &ex.scratch, km, k, 1, top)?;
        let io = DotIo { c1: off(km * cw + pw), c0: off(km * cw), ct_stride: cw as u64, out: s0, part_stride: pw as u64, out_ct_stride: cw as u64, out_rot_stride: 0, c0_mode: 1 };
        dots(ctx, ex, kc, group, r, 1, k, top, io)?;
        ks::moddown_inplace(ctx, &ex.compute, kc.plans, &mut ex.scratch, &mut ex.ks.conv, k, top)?;
        let (mut add, mut sub) = (Vec::with_capacity(2 * k), Vec::new());
        for (u, &i) in ch.iter().enumerate() {
            for p in 0..2 {
                let y = off(u * cw + p * pw) as *const c_void;
                add.push([part(sums[i], p), part(xs[i], p), y, nul]);
                if let Some(d) = diffs[i].as_deref() {
                    sub.push([part(d, p), part(xs[i], p), y]);
                }
            }
        }
        ks::ew_many(ctx, &ex.compute, &add, top + 1)?;
        ks::submul_many(ctx, &ex.compute, &sub, t, top + 1)?;
    }
    for i in 0..n {
        sums[i].meta = metas[i].0;
        if let Some(d) = diffs[i].as_deref_mut() {
            d.meta = metas[i].1;
        }
    }
    Ok(())
}

/// Hoisted rotations: `outs[i] = rot(x, steps[i])` from one ModUp. `group` holds the nonzero steps' keys in order.
#[allow(clippy::too_many_arguments)]
pub fn rotate_hoisted<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    ex: &mut Executor<'_, W>,
    kc: KsCfg<'_, '_>,
    sh: &Shadow,
    group: &KeyGroup<'_>,
    x: &Ct<'_, W>,
    steps: &[i64],
    outs: &mut [&mut Ct<'_, W>],
) -> Result<()> {
    if steps.len() != outs.len() {
        return Err(shape("rotate_hoisted", format!("{} steps, {} outputs", steps.len(), outs.len())));
    }
    let (k, top) = (x.k(), top_of(ctx, x)?);
    let n = ctx.lay.n;
    let mut live = Vec::new(); // (output index, key index)
    for (i, &st) in steps.iter().enumerate() {
        same_shape(x, outs[i], "rotate_hoisted")?;
        let g = pc_core::galois::rotation_galois(st, n)?;
        if g == 1 {
            copy_q(ctx, ex, x, outs[i], top)?;
        } else {
            live.push((i, key_index(group, g)?));
        }
        outs[i].meta = sh.rotate(&x.meta, st).out;
    }
    if live.is_empty() {
        return Ok(());
    }
    ks::modup(ctx, &ex.compute, kc.plans, &mut ex.ks, &mut ex.bconv_work, &x.buf, 0, k, 1, top)?;
    let per = (kc.plans.k_max / k).max(1);
    let mut j = 0;
    while j < live.len() {
        // a chunk of consecutive keys (one dot launch over a contiguous key range)
        let mut m = 1;
        while j + m < live.len() && m < per && live[j + m].1 == live[j].1 + m {
            m += 1;
        }
        let io = io_of(x, View::of(&ex.ks.ext, 0), k);
        dots(ctx, ex, kc, group, live[j].1, m, k, top, io)?;
        ks::moddown_conv(ctx, &ex.compute, kc.plans, &mut ex.ks.ext, &mut ex.ks.conv, m * k, top)?;
        for t in 0..m {
            let o = &mut *outs[live[j + t].0];
            ew(ctx, &ex.compute, Ew::SubMulPinv(top), View::of(&o.buf, 0), View::of(&ex.ks.ext, t * k), Some(View::of(&ex.ks.conv, t * k)), ctx.lay.rs_q(top), k, 2)?;
        }
        j += m;
    }
    Ok(())
}

/// How `rotate_sum_inplace_as` sums the extended rotations before the ModDown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SumForm {
    /// Dot outputs chunk by chunk, added into `ks.conv` element-wise (any k, any number of keys).
    Chunked,
    /// Every dot output in `ks.ext` at once, added inside the ModDown (`ks::moddown_sum_to`) when the keys are
    /// consecutive in the group and fit the extended workspace; Chunked otherwise. The same residues.
    InModDown,
}

/// A grid of rotations of up to `KEY_KMAX` ciphertexts: `outs[j].2 = rot(xs[outs[j].0], steps[outs[j].1])`. One ModUp
/// per input (hoisted), each key's inner product over every input in one launch (K-batched: one read or regeneration
/// of the key for all of them), the ModDown of every input at every step an output takes; every output is
/// bit-identical to a plain rotation.
/// Several inputs are first copied into `gather` (the inner product reads its inputs at one stride); outputs sharing
/// an (input, step) pair are computed once. `group` holds the steps' keys in order; the outputs are distinct buffers,
/// none of them an input. `fresh` false: `gather` and the ModUp workspace still hold these inputs' copies and ModUp
/// from an earlier grid of the same inputs (nothing switched keys on this executor since), so both are reused.
#[allow(clippy::too_many_arguments)]
pub fn rotate_grid<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    ex: &mut Executor<'_, W>,
    kc: KsCfg<'_, '_>,
    sh: &Shadow,
    group: &KeyGroup<'_>,
    xs: &[&Ct<'_, W>],
    gather: Option<&mut Ct<'_, W>>,
    fresh: bool,
    steps: &[i64],
    outs: &mut [(usize, usize, &mut Ct<'_, W>)],
) -> Result<()> {
    let k = xs.len();
    let lay = ctx.lay;
    let fits = |g: &Ct<'_, W>| g.buf.lay == lay && g.buf.buf.len() >= k * lay.ct_words();
    if k == 0 || k > ks::KEY_KMAX || (k > 1 && !gather.as_deref().is_some_and(fits)) {
        return Err(shape("rotate_grid", format!("{k} inputs (1..={}; several need a gather buffer of as many)", ks::KEY_KMAX)));
    }
    let top = top_of(ctx, xs[0])?;
    for x in xs {
        if x.k() != 1 || x.buf.lay != lay || top_of(ctx, x)? != top {
            return Err(shape("rotate_grid", "inputs must be single ciphertexts of the context layout at one level"));
        }
    }
    let mut key = Vec::with_capacity(steps.len());
    for &st in steps {
        let g = pc_core::galois::rotation_galois(st, lay.n)?;
        key.push(if g == 1 { None } else { Some(key_index(group, g)?) });
    }
    for (c, s, o) in outs.iter_mut() {
        if *c >= k || *s >= steps.len() || o.k() != 1 || o.buf.lay != lay {
            return Err(shape("rotate_grid", format!("output of input {c}, step {s}: {k} inputs, {} steps, one ciphertext each", steps.len())));
        }
        o.meta = sh.rotate(&xs[*c].meta, steps[*s]).out;
        if key[*s].is_none() {
            ew(ctx, &ex.compute, Ew::Copy, View::of(&o.buf, 0), View::of(&xs[*c].buf, 0), None, lay.rs_q(top), 1, 2)?;
        }
    }
    // (key index, step index) of the steps some output takes, in key order: a run of consecutive keys is one launch
    // per stage
    let mut live: Vec<(usize, usize)> = key.iter().enumerate().filter(|(s, _)| outs.iter().any(|o| o.1 == *s)).filter_map(|(s, r)| r.map(|r| (r, s))).collect();
    live.sort_unstable();
    live.dedup_by_key(|x| x.0);
    if live.is_empty() {
        return Ok(());
    }
    let x: &Ct<'_, W> = match gather {
        Some(g) if k > 1 => {
            if fresh {
                g.buf.k = k;
                g.meta = xs[0].meta;
                for (c, x) in xs.iter().enumerate() {
                    ew(ctx, &ex.compute, Ew::Copy, View::of(&g.buf, c), View::of(&x.buf, 0), None, lay.rs_q(top), 1, 2)?;
                }
            } else if g.buf.k != k {
                return Err(shape("rotate_grid", format!("reused gather buffer holds {} inputs, the grid reads {k}", g.buf.k)));
            }
            &*g
        }
        _ => xs[0],
    };
    if fresh {
        ks::modup(ctx, &ex.compute, kc.plans, &mut ex.ks, &mut ex.bconv_work, &x.buf, 0, k, 1, top)?;
    }
    let per = (kc.plans.k_max / k).max(1);
    let mut j = 0;
    while j < live.len() {
        let mut m = 1;
        while j + m < live.len() && m < per && live[j + m].0 == live[j].0 + m {
            m += 1;
        }
        let io = io_of(x, View::of(&ex.ks.ext, 0), k);
        dots(ctx, ex, kc, group, live[j].0, m, k, top, io)?;
        ks::moddown_conv(ctx, &ex.compute, kc.plans, &mut ex.ks.ext, &mut ex.ks.conv, m * k, top)?;
        for (t, &(r, _)) in live[j..j + m].iter().enumerate() {
            for (c, s, o) in outs.iter_mut() {
                if key[*s] == Some(r) {
                    let e = t * k + *c;
                    ew(ctx, &ex.compute, Ew::SubMulPinv(top), View::of(&o.buf, 0), View::of(&ex.ks.ext, e), Some(View::of(&ex.ks.conv, e)), lay.rs_q(top), 1, 2)?;
                }
            }
        }
        j += m;
    }
    Ok(())
}

/// In place: `acc = acc + sum_k rot(acc, steps[k])`, the rotations summed in the extended basis, one ModDown.
#[allow(clippy::too_many_arguments)]
pub fn rotate_sum_inplace<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    ex: &mut Executor<'_, W>,
    kc: KsCfg<'_, '_>,
    sh: &Shadow,
    group: &KeyGroup<'_>,
    acc: &mut Ct<'_, W>,
    steps: &[i64],
) -> Result<()> {
    rotate_sum_inplace_as(ctx, ex, kc, sh, group, acc, steps, SumForm::InModDown)
}

/// `rotate_sum_inplace` with the summation form chosen (`SumForm`; benches and checks compare the two).
#[allow(clippy::too_many_arguments)]
pub fn rotate_sum_inplace_as<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    ex: &mut Executor<'_, W>,
    kc: KsCfg<'_, '_>,
    sh: &Shadow,
    group: &KeyGroup<'_>,
    acc: &mut Ct<'_, W>,
    steps: &[i64],
    form: SumForm,
) -> Result<()> {
    let out = View::of(&acc.buf, 0);
    rotate_sum_core(ctx, ex, kc, group, acc, out, ks::MdOut::Add, steps, form)?;
    acc.meta = sh.rotate_sum(&acc.meta, steps).out;
    Ok(())
}

/// The sum of the rotations of `x` by `steps`, ModDowned into `out` as `mode` says (`Add` with out = x's view: in
/// place; `Onto(x)`: out = x + the sum). The caller sets the output's meta.
#[allow(clippy::too_many_arguments)]
fn rotate_sum_core<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    ex: &mut Executor<'_, W>,
    kc: KsCfg<'_, '_>,
    group: &KeyGroup<'_>,
    x: &Ct<'_, W>,
    out: View,
    mode: ks::MdOut,
    steps: &[i64],
    form: SumForm,
) -> Result<()> {
    let (k, top) = (x.k(), top_of(ctx, x)?);
    let n = ctx.lay.n;
    let mut keys = Vec::with_capacity(steps.len());
    for &st in steps {
        let g = pc_core::galois::rotation_galois(st, n)?;
        if g == 1 {
            return Err(shape("rotate_sum", "step 0 in a rotate_sum group"));
        }
        keys.push(key_index(group, g)?);
    }
    if keys.is_empty() {
        return Err(shape("rotate_sum", "no steps"));
    }
    let lay = ctx.lay;
    let rs = lay.rs_qp(top);
    ks::modup(ctx, &ex.compute, kc.plans, &mut ex.ks, &mut ex.bconv_work, &x.buf, 0, k, 1, top)?;
    let per = (ex.ks.ext.k / k).max(1);
    let run = keys.iter().enumerate().all(|(j, &r)| r == keys[0] + j);
    if form == SumForm::InModDown && run && keys.len() <= per {
        // every extended rotation in ks.ext at once, summed inside the ModDown
        let io = io_of(x, View::of(&ex.ks.ext, 0), k);
        dots(ctx, ex, kc, group, keys[0], keys.len(), k, top, io)?;
        return ks::moddown_sum_to(ctx, &ex.compute, kc.plans, &mut ex.ks.ext, keys.len(), k, &mut ex.ks.conv, out, mode, k, top);
    }
    // extended rotations, chunk by chunk, summed into ks.conv (Q rows 0..=top and P)
    let mut j = 0;
    let mut first = true;
    while j < keys.len() {
        let mut m = 1;
        while j + m < keys.len() && m < per && keys[j + m] == keys[j] + m {
            m += 1;
        }
        {
            let io = io_of(x, View::of(&ex.ks.ext, 0), k);
            dots(ctx, ex, kc, group, keys[j], m, k, top, io)?;
        }
        let cv = View::of(&ex.ks.conv, 0);
        for t in 0..m {
            let e = View::of(&ex.ks.ext, t * k);
            if first {
                ew(ctx, &ex.compute, Ew::Copy, cv, e, None, rs, k, 2)?;
                first = false;
            } else {
                ew(ctx, &ex.compute, Ew::Add, cv, cv, Some(e), rs, k, 2)?;
            }
        }
        j += m;
    }
    // ModDown of the sum (conversion into ks.ext), fused with the accumulate: out = y + (sum - conv) P^-1
    ks::moddown_to(ctx, &ex.compute, kc.plans, &mut ex.ks.conv, &mut ex.ks.ext, out, mode, k, top)
}

/// `out = x + sum_k rot(x, steps[k])` (see `rotate_sum_inplace`; x is the ModDown's accumulate source, no copy).
#[allow(clippy::too_many_arguments)]
pub fn rotate_sum<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    ex: &mut Executor<'_, W>,
    kc: KsCfg<'_, '_>,
    sh: &Shadow,
    group: &KeyGroup<'_>,
    x: &Ct<'_, W>,
    steps: &[i64],
    out: &mut Ct<'_, W>,
) -> Result<()> {
    same_shape(x, out, "rotate_sum")?;
    let o = View::of(&out.buf, 0);
    rotate_sum_core(ctx, ex, kc, group, x, o, ks::MdOut::Onto(View::of(&x.buf, 0)), steps, SumForm::InModDown)?;
    out.meta = sh.rotate_sum(&x.meta, steps).out;
    Ok(())
}

/// K sibling rotate_sums of one step list: `outs[c] = xs[c] + sum_k rot(xs[c], steps[k])`, the inputs copied into
/// `gather` (the launches read K ciphertexts at one stride) and summed as one K-batch, so each key is read or
/// regenerated once for all of them; the sums are copied out. Bit-identical to `rotate_sum` of each input.
#[allow(clippy::too_many_arguments)]
pub fn rotate_sum_batch<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    ex: &mut Executor<'_, W>,
    kc: KsCfg<'_, '_>,
    sh: &Shadow,
    group: &KeyGroup<'_>,
    xs: &[&Ct<'_, W>],
    gather: &mut Ct<'_, W>,
    steps: &[i64],
    outs: &mut [&mut Ct<'_, W>],
) -> Result<()> {
    let k = xs.len();
    let lay = ctx.lay;
    if k == 0 || k > ks::KEY_KMAX || outs.len() != k || gather.buf.lay != lay || gather.buf.buf.len() < k * lay.ct_words() {
        return Err(shape("rotate_sum_batch", format!("{k} inputs, {} outputs (1..={} each, a gather buffer of as many)", outs.len(), ks::KEY_KMAX)));
    }
    let top = top_of(ctx, xs[0])?;
    for (x, o) in xs.iter().zip(outs.iter()) {
        if x.k() != 1 || o.k() != 1 || x.buf.lay != lay || o.buf.lay != lay || top_of(ctx, x)? != top {
            return Err(shape("rotate_sum_batch", "inputs and outputs must be single ciphertexts of the context layout, inputs at one level"));
        }
    }
    gather.buf.k = k;
    gather.meta = xs[0].meta;
    for (c, x) in xs.iter().enumerate() {
        ew(ctx, &ex.compute, Ew::Copy, View::of(&gather.buf, c), View::of(&x.buf, 0), None, lay.rs_q(top), 1, 2)?;
    }
    rotate_sum_inplace(ctx, ex, kc, sh, group, gather, steps)?;
    for (c, o) in outs.iter_mut().enumerate() {
        ew(ctx, &ex.compute, Ew::Copy, View::of(&o.buf, 0), View::of(&gather.buf, c), None, lay.rs_q(top), 1, 2)?;
        o.meta = sh.rotate_sum(&xs[c].meta, steps).out;
    }
    Ok(())
}

fn add_inplace<W: DevWord, M: DevStrategy<W>>(ctx: &Context<'_, W, M>, ex: &Executor<'_, W>, acc: &mut Ct<'_, W>, y: &Ct<'_, W>, top: usize) -> Result<()> {
    let a = View::of(&acc.buf, 0);
    ew(ctx, &ex.compute, Ew::Add, a, a, Some(View::of(&y.buf, 0)), ctx.lay.rs_q(top), acc.k(), 2)
}

/// The rotate-and-sum ladder (C++ `rotate_and_sum` semantics and op stream, see the module doc): `out` = the sum of
/// x over the rotations by every multiple of `start` below `stop`; `tmp` is a scratch ciphertext of x's shape.
/// `radix_bits` < 2: rung by rung.
#[allow(clippy::too_many_arguments)]
pub fn rotate_and_sum<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    ex: &mut Executor<'_, W>,
    kc: KsCfg<'_, '_>,
    sh: &Shadow,
    keys: &dyn RotKeys,
    x: &Ct<'_, W>,
    start: i64,
    stop: i64,
    radix_bits: u32,
    out: &mut Ct<'_, W>,
    tmp: &mut Ct<'_, W>,
) -> Result<()> {
    if start == 0 || stop <= 0 {
        return Err(shape("rotate_and_sum", format!("start {start} must be non-zero, stop {stop} positive")));
    }
    same_shape(x, out, "rotate_and_sum")?;
    same_shape(x, tmp, "rotate_and_sum")?;
    let top = top_of(ctx, x)?;
    let sign = start.signum();
    let gap = start.abs();
    if gap >= stop {
        copy_q(ctx, ex, x, out, top)?;
        out.meta = x.meta;
        return Ok(());
    }
    let mut gaps = Vec::new();
    let mut g = gap;
    while g < stop {
        gaps.push(g);
        g *= 2;
    }
    let rb = if radix_bits >= 2 { radix_bits as usize } else { 1 };
    let mut have = false;
    let single = |st: i64| keys.group(&[st]).ok_or_else(|| shape("rotate_and_sum", format!("no rotation key group for step {st}")));
    let mut i0 = 0;
    while i0 < gaps.len() {
        let n = rb.min(gaps.len() - i0);
        let steps: Vec<i64> = (1..(1i64 << n)).map(|j| sign * j * gaps[i0]).collect();
        if radix_bits >= 2 && n > 1 {
            if let Some(grp) = keys.group(&steps) {
                if !have {
                    copy_q(ctx, ex, x, out, top)?;
                    out.meta = x.meta;
                    have = true;
                }
                rotate_sum_inplace(ctx, ex, kc, sh, grp, out, &steps)?;
                i0 += n;
                continue;
            }
        }
        for r in 0..n {
            let st = sign * gaps[i0 + r];
            if !have {
                rotate(ctx, ex, kc, sh, single(st)?, x, st, tmp)?;
                copy_q(ctx, ex, x, out, top)?;
                out.meta = x.meta;
                add_inplace(ctx, ex, out, tmp, top)?;
                have = true;
            } else {
                rotate(ctx, ex, kc, sh, single(st)?, out, st, tmp)?;
                add_inplace(ctx, ex, out, tmp, top)?;
            }
            out.meta = sh.add(&out.meta, &tmp.meta)?.out;
        }
        i0 += n;
    }
    Ok(())
}
