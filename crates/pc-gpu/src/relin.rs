//! Relinearization and `mult = tensor + relin`, bit-exact with FIDESlib's `Ciphertext::mult` (FLEXIBLEAUTO,
//! moddown = true, no fused rescale): `binomialMult` gives (c0, c1, d2) = (a0 b0, a0 b1 + a1 b0, a1 b1); d2 is key
//! switched with the relinearization key (`MGPUkeySwitchCore`: ModUp, dot, ModDown) and added: c0 += ModDown(ks0),
//! c1 += ModDown(ks1). The degree stays 2 (rescale pending) until the next op pays it.
//!
//! The K-batched pipeline keeps d2 out of ciphertext slots: the tensor kernel writes d2 (evaluation form) into part 1
//! of `KsWork::ext`, where the ModUp's INTT reads it (out of place). The ModDown is fused with the accumulation
//! (`dst += (dot - conv) P^-1`):
//! - `KskSource::Regen` (fused relin): `ks_dot_regen` over the special rows only, the conversion,
//!   then `ks_dot_regen` over the Q rows straight into the epilogue (the Q-row inner product is never stored);
//! - `KskSource::Cached`: one `ks_dot` over all rows into `ext`, the conversion, `md_epilogue`.
use crate::context::{Context, DevStrategy, DevWord};
use crate::ct::Ct;
use crate::error::{shape, Result};
use crate::executor::Executor;
use crate::ks::{self, md_epilogue, DotIo, DotRows, KeyGroup, KskSource, View};
use crate::rotate::KsCfg;
use pc_cuda::launch_result;
use pc_kernels::{EwArgs, PolyRef, TensorArgs};
use pc_shadow::Shadow;

/// (c0, c1) of `out` = the tensor's first two terms, d2 into `ex.ks.ext` part 1; Q rows 0..=top of k ciphertexts. `a`
/// and `b` must already sit at the same top with degree 1 (the Recipe preludes ran).
pub fn tensor<W: DevWord, M: DevStrategy<W>>(ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, a: &Ct<'_, W>, b: &Ct<'_, W>, out: &mut Ct<'_, W>, top: usize) -> Result<()> {
    let k = a.k();
    if b.k() != k || out.k() != k || k > ex.ks.k_max {
        return Err(shape("tensor", format!("k {} x {} -> {} (k_max {})", k, b.k(), out.k(), ex.ks.k_max)));
    }
    let lay = ctx.lay;
    let va = View::of(&a.buf, 0).r();
    let vb = View::of(&b.buf, 0).r();
    let args = TensorArgs {
        out: View::of(&out.buf, 0).r(),
        d2: View::of(&ex.ks.ext, 0).r(),
        a0: va,
        b0: vb,
        a1: va,
        b1: vb,
        coef: std::ptr::null_mut(),
        coef_ct_stride: lay.poly_words() as u64,
        pr: ctx.pr(),
        npairs: 1,
        acc: 0,
        n: (top + 1) as i32,
        k: k as i32,
        dbl: 0,
        pad: 0,
    };
    // SAFETY: k-ct views of live buffers of the context layout; out, ext and the workspace are distinct from a / b
    launch_result("tensor", unsafe { (ctx.table.tensor)(&args, ex.compute.raw()) })?;
    Ok(())
}

/// Relinearize in place: `ct += KeySwitch(d2)` with d2 staged by `tensor` (k = ct.k() ciphertexts at `top`).
pub fn relin_staged<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    ex: &mut Executor<'_, W>,
    kc: KsCfg<'_, '_>,
    key: &KeyGroup<'_>,
    ct: &mut Ct<'_, W>,
    top: usize,
) -> Result<()> {
    let d2 = View::of(&ex.ks.ext, 0).part::<W>(1);
    relin_staged_from(ctx, ex, kc, key, ct, top, d2)
}

/// `relin_staged` with d2 read from `d2` (evaluation rows 0..=top of k polynomials, ciphertext-layout strides) instead
/// of part 1 of the key-switching workspace.
pub fn relin_staged_from<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    ex: &mut Executor<'_, W>,
    kc: KsCfg<'_, '_>,
    key: &KeyGroup<'_>,
    ct: &mut Ct<'_, W>,
    top: usize,
    d2: View,
) -> Result<()> {
    let k = ct.k();
    relin_staged_view(ctx, ex, kc, key, View::of(&ct.buf, 0), k, top, d2)
}

/// `relin_staged_from` into a view of k ciphertexts of the context layout (`dst`) instead of a `Ct`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn relin_staged_view<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    ex: &mut Executor<'_, W>,
    kc: KsCfg<'_, '_>,
    key: &KeyGroup<'_>,
    dst: View,
    k: usize,
    top: usize,
    d2: View,
) -> Result<()> {
    if key.galois.first() != Some(&1) {
        return Err(shape("relin", "key group 0 is not the relinearization key (galois 1)"));
    }
    let lay = ctx.lay;
    let ext = View::of(&ex.ks.ext, 0);
    ks::modup_from(ctx, &ex.compute, kc.plans, &mut ex.ks, &mut ex.bconv_work, d2, k, top)?;
    let io = DotIo {
        c1: d2.base,
        c0: std::ptr::null(),
        ct_stride: d2.ct_stride,
        out: ext.base,
        part_stride: lay.poly_words() as u64,
        out_ct_stride: lay.ct_words() as u64,
        out_rot_stride: 0,
        c0_mode: 0,
    };
    match kc.source {
        KskSource::Regen => {
            let ndig = kc.plans.ndig(top)?;
            ks::stage_key(ctx, &ex.compute, kc.plans, &mut ex.ks, key, 0, ndig, false)?;
            ks::dot_regen(ctx, &ex.compute, kc.plans, &ex.ks, key, k, top, DotRows::P, io, None)?;
            ks::moddown_conv(ctx, &ex.compute, kc.plans, &mut ex.ks.ext, &mut ex.ks.conv, k, top)?;
            let epi = (dst, View::of(&ex.ks.conv, 0));
            ks::dot_regen(ctx, &ex.compute, kc.plans, &ex.ks, key, k, top, DotRows::Q, io, Some(epi))
        }
        KskSource::Cached => {
            let staged = key.needs_stage() || !key.has_a();
            if staged {
                let ndig = kc.plans.ndig(top)?;
                ks::stage_key(ctx, &ex.compute, kc.plans, &mut ex.ks, key, 0, ndig, true)?;
            }
            ks::dot(ctx, &ex.compute, kc.plans, &ex.ks, key, 0, 1, k, top, DotRows::All, io, staged)?;
            ks::moddown_conv(ctx, &ex.compute, kc.plans, &mut ex.ks.ext, &mut ex.ks.conv, k, top)?;
            md_epilogue(ctx, &ex.compute, dst, View::of(&ex.ks.ext, 0), View::of(&ex.ks.conv, 0), k, top)
        }
    }
}

/// `relin_staged_from` of single ciphertexts at `top`, the d2 of `outs[u]` at `d2` + u `d2.ct_stride` words (rows
/// 0..=top, `top + 1` rows apart). With cached keys the key switch runs `k_max` ciphertexts per launch (each batch's
/// d2 copied into the coefficient workspace, `[c][row][N]` polys of the layout, for the ModUp's INTT), the epilogue
/// one ciphertext at a time; with regenerated keys, one ciphertext at a time.
pub fn relin_many<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    ex: &mut Executor<'_, W>,
    kc: KsCfg<'_, '_>,
    key: &KeyGroup<'_>,
    outs: &mut [&mut Ct<'_, W>],
    top: usize,
    d2: View,
) -> Result<()> {
    let lay = ctx.lay;
    let rows = (top + 1) * lay.n;
    if d2.ct_stride != rows as u64 || outs.iter().any(|o| o.k() != 1) {
        return Err(shape("relin_many", "single ciphertexts, d2 rows 0..=top packed"));
    }
    if key.galois.first() != Some(&1) {
        return Err(shape("relin", "key group 0 is not the relinearization key (galois 1)"));
    }
    let at = |u: usize| View { base: (d2.base as *mut W).wrapping_add(u * rows) as *mut std::ffi::c_void, ..d2 };
    if kc.source == KskSource::Regen {
        for (u, o) in outs.iter_mut().enumerate() {
            relin_staged_from(ctx, ex, kc, key, o, top, at(u))?;
        }
        return Ok(());
    }
    let km = ex.ks.k_max;
    for (bi, batch) in outs.chunks_mut(km).enumerate() {
        let k = batch.len();
        let src = at(bi * km);
        let coef = PolyRef { base: ex.bconv_work.vptr_mut(), ct_stride: lay.poly_words() as u64, part_stride: 0 };
        let cp = EwArgs {
            out: coef,
            a: src.r(),
            b: src.r(),
            rs: lay.rs_q(top),
            sc: std::ptr::null(),
            scs: std::ptr::null(),
            pr: ctx.pr(),
            k: k as i32,
            nparts: 1,
            sub: 0,
            zero_p: 0,
        };
        if ex.bconv_work.len() < k * lay.poly_words() {
            return Err(shape("relin_many", "coefficient workspace too small"));
        }
        // SAFETY: k polys of rows 0..=top from the d2 rows ((top + 1) N words apart) into the coefficient workspace
        // (poly_words apart, checked above; distinct buffers)
        launch_result("ew_copy", unsafe { (ctx.table.ew_copy)(&cp, ex.compute.raw()) })?;
        ks::modup_coef_ready(ctx, &ex.compute, kc.plans, &mut ex.ks, &mut ex.bconv_work, k, top)?;
        let ext = View::of(&ex.ks.ext, 0);
        let io = DotIo {
            c1: src.base,
            c0: std::ptr::null(),
            ct_stride: src.ct_stride,
            out: ext.base,
            part_stride: lay.poly_words() as u64,
            out_ct_stride: lay.ct_words() as u64,
            out_rot_stride: 0,
            c0_mode: 0,
        };
        let staged = key.needs_stage() || !key.has_a();
        if staged {
            let ndig = kc.plans.ndig(top)?;
            ks::stage_key(ctx, &ex.compute, kc.plans, &mut ex.ks, key, 0, ndig, true)?;
        }
        ks::dot(ctx, &ex.compute, kc.plans, &ex.ks, key, 0, 1, k, top, DotRows::All, io, staged)?;
        ks::moddown_conv(ctx, &ex.compute, kc.plans, &mut ex.ks.ext, &mut ex.ks.conv, k, top)?;
        for (u, o) in batch.iter_mut().enumerate() {
            md_epilogue(ctx, &ex.compute, View::of(&o.buf, 0), View::of(&ex.ks.ext, u), View::of(&ex.ks.conv, u), 1, top)?;
        }
    }
    Ok(())
}

/// `out = a * b` relinearized (degree 2, rescale pending). The operands must need no Recipe prelude (same level,
/// degree 1): `ops::mult` runs the preludes (Rescale / DropTo / ScalarAdjust) and then this.
#[allow(clippy::too_many_arguments)]
pub fn mult<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    ex: &mut Executor<'_, W>,
    kc: KsCfg<'_, '_>,
    sh: &Shadow,
    key: &KeyGroup<'_>,
    a: &Ct<'_, W>,
    b: &Ct<'_, W>,
    out: &mut Ct<'_, W>,
) -> Result<()> {
    let r = sh.mult(&a.meta, &b.meta)?;
    if !r.pre.is_empty() {
        return Err(shape("relin::mult", format!("operands need preludes {:?} (run them first: ops::mult)", r.pre)));
    }
    let top = a.top(ctx.shape)?;
    tensor(ctx, ex, a, b, out, top)?;
    relin_staged(ctx, ex, kc, key, out, top)?;
    out.meta = r.out;
    Ok(())
}

/// `out = a^2` relinearized (FIDESlib `square`; the same residues as `mult(a, a)`).
pub fn square<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    ex: &mut Executor<'_, W>,
    kc: KsCfg<'_, '_>,
    sh: &Shadow,
    key: &KeyGroup<'_>,
    a: &Ct<'_, W>,
    out: &mut Ct<'_, W>,
) -> Result<()> {
    let r = sh.square(&a.meta)?;
    if !r.pre.is_empty() {
        return Err(shape("relin::square", format!("operand needs preludes {:?} (run them first)", r.pre)));
    }
    let top = a.top(ctx.shape)?;
    tensor(ctx, ex, a, a, out, top)?;
    relin_staged(ctx, ex, kc, key, out, top)?;
    out.meta = r.out;
    Ok(())
}
