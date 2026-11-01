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
        dbl: 0,
        acc_d2: 0,
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

// ── relinearization fused with the rescale after it ─────────────────────────────────────────────────────────────────

/// A per-prime scalar and its Shoup companion (the epilogue's alpha / beta, the top weights).
struct Scalars<'a, W: DevWord> {
    v: pc_cuda::ConstBuf<'a, W>,
    s: pc_cuda::ConstBuf<'a, W>,
}

impl<'a, W: DevWord> Scalars<'a, W> {
    fn new<S: pc_cuda::Lane>(arena: &'a pc_cuda::Arena, st: &pc_cuda::Stream<S>, primes: &[u64], vals: &[u64]) -> Result<Self> {
        let (v, s): (Vec<W>, Vec<W>) = vals
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let p = primes[i % primes.len()];
                let c = pc_core::modmul::shoup(W::from_u64(x % p), W::from_u64(p));
                (c.w, c.w_prime)
            })
            .unzip();
        Ok(Scalars { v: pc_cuda::ConstBuf::new(arena, "relin.rr", st, &v)?, s: pc_cuda::ConstBuf::new(arena, "relin.rr", st, &s)? })
    }
    fn bytes(n: usize) -> usize {
        2 * pc_cuda::Arena::bytes_for::<W>(n)
    }
}

/// The relinearization ModDown fused with the rescale after it at one top (EvalMod's `RelinRescale`, for the
/// program's `Mult [x2]; Rescale`): the extended product is `V = P c + dot` (c the tensor, doubled on x2; dot the key
/// inner product), the result round(V / (P q_D)) = `c q_D^-1 + (dot - conv) (P q_D)^-1` on rows 0..=top-g, q_D the
/// product of the g dropped tops (two on a composite level, one on an alternating user region's single-prime level),
/// conv the exact centred conversion of V from P and the dropped tops: one rounding, one NTT of the remaining rows
/// instead of the ModDown's and the rescale's.
struct RrTop<'a, W: DevWord> {
    /// primes dropped (1 or 2)
    g: usize,
    /// dot rows: the dropped tops (top - g + 1..=top), then the special rows
    rows: pc_cuda::ConstBuf<'a, u32>,
    /// rows 0..=top-g
    qrows: pc_cuda::ConstBuf<'a, u32>,
    md: crate::bts::cts::MdPlan<'a>,
    alpha: Scalars<'a, W>,
    beta: Scalars<'a, W>,
    /// `P c + dot` on the dropped tops: P mod q_t of each, then 1 each
    wtop: Scalars<'a, W>,
}

/// `RrTop` tables for every composite top the full special set serves, and on an alternating user region's chain
/// every single-prime top (built once at setup).
pub struct RelinRescaler<'a, W: DevWord> {
    tops: std::collections::BTreeMap<usize, RrTop<'a, W>>,
    k_max: usize,
}

impl<'a, W: DevWord> RelinRescaler<'a, W> {
    /// The served tops and the primes their rescale drops.
    fn tops(chain: &pc_core::params::Chain) -> Vec<(usize, usize)> {
        let nq = chain.q.len();
        let singles = chain.spec.sp.is_some();
        (3..nq)
            .filter_map(|t| match chain.drop_at(t) {
                Some(2) if chain.drop_at(t - 2).is_some() => Some((t, 2)),
                Some(1) if singles && chain.drop_at(t - 1).is_some() => Some((t, 1)),
                _ => None,
            })
            .collect()
    }

    /// Device bytes of the tables (`MemPlan` section "relin.rr").
    pub fn bytes(chain: &pc_core::params::Chain, lay: &crate::poly::Layout, k_max: usize) -> Result<usize> {
        let primes = chain.primes();
        let nq = lay.nq;
        let mut b = 0;
        for (top, g) in Self::tops(chain) {
            b += pc_cuda::Arena::bytes_for::<u32>(g + lay.kp) + pc_cuda::Arena::bytes_for::<u32>(top + 1 - g);
            b += crate::bts::cts::md_plan_bytes::<W>(&primes, k_max, 2, lay.kp + g, top + 1 - g, true)?;
            b += 2 * Scalars::<W>::bytes(nq) + Scalars::<W>::bytes(2 * g);
        }
        Ok(b)
    }

    pub fn new<S: pc_cuda::Lane, M: DevStrategy<W>>(arena: &'a pc_cuda::Arena, s: &pc_cuda::Stream<S>, ctx: &Context<'_, W, M>, k_max: usize) -> Result<Self> {
        use pc_core::modmul::{invmod, mulmod};
        let lay = ctx.lay;
        let nq = lay.nq;
        let primes = ctx.chain.primes();
        let pmod: Vec<u64> = (0..nq).map(|i| ctx.chain.p.iter().fold(1u64, |a, &x| mulmod(a, x % primes[i], primes[i]))).collect();
        let specials: Vec<usize> = (nq..lay.rows()).collect();
        let mut tops = std::collections::BTreeMap::new();
        for (top, g) in Self::tops(&ctx.chain) {
            let dropped: Vec<usize> = (top + 1 - g..=top).collect();
            let mut src = specials.clone();
            src.extend(&dropped);
            let r: Vec<u32> = dropped.iter().chain(&specials).map(|&t| t as u32).collect();
            let qr: Vec<u32> = (0..=(top - g) as u32).collect();
            let qd = |i: usize| dropped.iter().fold(1u64, |a, &t| mulmod(a, primes[t] % primes[i], primes[i]));
            let al: Vec<u64> = (0..nq).map(|i| if i + g <= top { invmod(qd(i), primes[i]) } else { 0 }).collect();
            let be: Vec<u64> = (0..nq).map(|i| if i + g <= top { invmod(mulmod(qd(i), pmod[i], primes[i]), primes[i]) } else { 0 }).collect();
            let wp: Vec<u64> = dropped.iter().map(|&t| primes[t]).collect();
            let wv: Vec<u64> = dropped.iter().map(|&t| pmod[t]).chain(std::iter::repeat_n(1, g)).collect();
            tops.insert(
                top,
                RrTop {
                    g,
                    rows: pc_cuda::ConstBuf::new(arena, "relin.rr", s, &r)?,
                    qrows: pc_cuda::ConstBuf::new(arena, "relin.rr", s, &qr)?,
                    md: crate::bts::cts::md_plan::<_, W>(arena, s, &lay, &primes, &(0..k_max).collect::<Vec<_>>(), &[0, 1], &src, top + 1 - g, true)?,
                    alpha: Scalars::new(arena, s, &primes[..nq], &al)?,
                    beta: Scalars::new(arena, s, &primes[..nq], &be)?,
                    wtop: Scalars::new(arena, s, &wp, &wv)?,
                },
            );
        }
        Ok(RelinRescaler { tops, k_max })
    }

    /// Whether a product at `top` (k ciphertexts) can take the fused path.
    pub fn serves(&self, top: usize, k: usize) -> bool {
        self.tops.contains_key(&top) && k <= self.k_max
    }

    /// `ct = rescale(c + KeySwitch(d2))` with the tensor staged by `tensor` (c in `ct` rows 0..=top, d2 in `ex.ks.ext`
    /// part 1; doubled there on x2): rows 0..=top-g of `ct` (the caller sets the rescaled metadata). Regen key.
    pub fn relin_rescale<M: DevStrategy<W>>(&self, ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, kc: KsCfg<'_, '_>, key: &KeyGroup<'_>, ct: &mut Ct<'_, W>, top: usize) -> Result<()> {
        let k = ct.k();
        let rr = self.tops.get(&top).ok_or_else(|| shape("relin_rescale", format!("no tables for top {top}")))?;
        if k > self.k_max || kc.source != KskSource::Regen {
            return Err(shape("relin_rescale", "a Regen key and at most k_max ciphertexts"));
        }
        let lay = ctx.lay;
        let (pw, cw, nn) = (lay.poly_words() as u64, lay.ct_words() as u64, lay.n);
        let ext = View::of(&ex.ks.ext, 0);
        ks::modup_from(ctx, &ex.compute, kc.plans, &mut ex.ks, &mut ex.bconv_work, ext.part::<W>(1), k, top)?;
        let io = DotIo { c1: ext.part::<W>(1).base, c0: std::ptr::null(), ct_stride: cw, out: ext.base, part_stride: pw, out_ct_stride: cw, out_rot_stride: 0, c0_mode: 0 };
        let g = rr.g;
        // 1. the dot over the dropped tops and the special rows, raw into ks.ext
        ks::dot_regen_ex(ctx, &ex.compute, kc.plans, &ex.ks, key, k, top, (rr.rows.dptr(), g + lay.kp), io, None)?;
        // 2. the tops of the extended product: P c + dot
        let dst = View::of(&ct.buf, 0);
        let row = |v: View, r: usize| pc_kernels::PolyRef { base: (v.base as *mut W).wrapping_add(r * nn) as *mut std::ffi::c_void, ct_stride: v.ct_stride, part_stride: v.part_stride };
        let lo = top + 1 - g;
        let mut src = [pc_kernels::PolyRef::default(); 8];
        src[0] = row(dst, lo);
        src[1] = row(ext, lo);
        let w = pc_kernels::WsumArgs { out: row(ext, lo), src, w: rr.wtop.v.vptr(), ws: rr.wtop.s.vptr(), pr: ctx.pr().wrapping_add(lo), nsrc: 2, rowstride: g as i32, n: g as i32, k: k as i32 };
        // SAFETY: rows lo..=top of k cts of ct and of ks.ext (the prime table offset to those rows); element-wise,
        // out aliasing src[1] index by index
        launch_result("wsum", unsafe { (ctx.table.wsum)(&w, ex.compute.raw()) })?;
        // 3. coefficient form of the source rows, the exact conversion to rows 0..=top-g, their NTT
        rr.md.run(ctx, &ex.compute, &mut ex.ks.ext.buf, &mut ex.ks.conv.buf, k)?;
        // 4. the Q-row dot into ct = alpha c + beta (dot - conv)
        let conv = View::of(&ex.ks.conv, 0);
        let e = pc_kernels::EpiArgs { dst: dst.base, conv: conv.base, binv: rr.beta.v.vptr(), binvs: rr.beta.s.vptr(), k: std::ptr::null(), part_stride: pw, dbl: 0, pad: 0, alpha: rr.alpha.v.vptr(), alphas: rr.alpha.s.vptr() };
        ks::dot_regen_ex(ctx, &ex.compute, kc.plans, &ex.ks, key, k, top, (rr.qrows.dptr(), lo), io, Some((e, dst.ct_stride, conv.ct_stride)))
    }
}
