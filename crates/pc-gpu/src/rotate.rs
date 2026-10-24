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

fn key_index(group: &KeyGroup<'_>, g: u64) -> Result<usize> {
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
            ew(ctx, &ex.compute, Ew::SubMulPinv, View::of(&o.buf, 0), View::of(&ex.ks.ext, t * k), Some(View::of(&ex.ks.conv, t * k)), ctx.lay.rs_q(top), k, 2)?;
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
/// of the key for all of them), one ModDown per (input, step) pair; every output is bit-identical to a plain rotation.
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
                    ew(ctx, &ex.compute, Ew::SubMulPinv, View::of(&o.buf, 0), View::of(&ex.ks.ext, e), Some(View::of(&ex.ks.conv, e)), lay.rs_q(top), 1, 2)?;
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
    let (k, top) = (acc.k(), top_of(ctx, acc)?);
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
    ks::modup(ctx, &ex.compute, kc.plans, &mut ex.ks, &mut ex.bconv_work, &acc.buf, 0, k, 1, top)?;
    let per = (ex.ks.ext.k / k).max(1);
    let run = keys.iter().enumerate().all(|(j, &r)| r == keys[0] + j);
    if form == SumForm::InModDown && run && keys.len() <= per {
        // every extended rotation in ks.ext at once, summed inside the ModDown
        let io = io_of(acc, View::of(&ex.ks.ext, 0), k);
        dots(ctx, ex, kc, group, keys[0], keys.len(), k, top, io)?;
        ks::moddown_sum_to(ctx, &ex.compute, kc.plans, &mut ex.ks.ext, keys.len(), k, &mut ex.ks.conv, View::of(&acc.buf, 0), k, top)?;
        acc.meta = sh.rotate_sum(&acc.meta, steps).out;
        return Ok(());
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
            let io = io_of(acc, View::of(&ex.ks.ext, 0), k);
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
    // ModDown of the sum (conversion into ks.ext), fused with the accumulate: acc += (sum - conv) P^-1
    ks::moddown_to(ctx, &ex.compute, kc.plans, &mut ex.ks.conv, &mut ex.ks.ext, View::of(&acc.buf, 0), true, k, top)?;
    acc.meta = sh.rotate_sum(&acc.meta, steps).out;
    Ok(())
}

/// `out = x + sum_k rot(x, steps[k])` (see `rotate_sum_inplace`).
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
    let top = top_of(ctx, x)?;
    copy_q(ctx, ex, x, out, top)?;
    out.meta = x.meta;
    rotate_sum_inplace(ctx, ex, kc, sh, group, out, steps)
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
