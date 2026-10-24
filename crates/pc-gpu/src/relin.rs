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
use pc_kernels::TensorArgs;
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
    let k = ct.k();
    if key.galois.first() != Some(&1) {
        return Err(shape("relin", "key group 0 is not the relinearization key (galois 1)"));
    }
    let lay = ctx.lay;
    let ext = View::of(&ex.ks.ext, 0);
    ks::modup_from(ctx, &ex.compute, kc.plans, &mut ex.ks, &mut ex.bconv_work, ext.part::<W>(1), k, top)?;
    let io = DotIo {
        c1: ext.part::<W>(1).base,
        c0: std::ptr::null(),
        ct_stride: lay.ct_words() as u64,
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
            let epi = (View::of(&ct.buf, 0), View::of(&ex.ks.conv, 0));
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
            md_epilogue(ctx, &ex.compute, View::of(&ct.buf, 0), View::of(&ex.ks.ext, 0), View::of(&ex.ks.conv, 0), k, top)
        }
    }
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
