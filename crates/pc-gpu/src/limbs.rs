//! Limb I/O and monomial merges.
//!
//! `export_limbs` / `import_limbs`: perseus' `bts.export_limbs` / `bts.import_limbs` format, coefficient-form residues
//! of the live Q limbs per polynomial (row i = limb i, prime id i) plus the metadata (moduli, FIDESlib top limb
//! index as `level`, noise level, noise factor), for cross-checks and the PCMM/CCMM key path.
//!
//! `merge_monomial(ct, other, power)`: `ct += X^power * other` (perseus `bts.merge_monomial`: FIDESlib
//! `multMonomial` on a copy, then `add`): no level change, no key switch. The monomial's evaluation rows are built
//! on the host at setup (`MonomialTable`, one per power): X^p at eval index i is psi^((2 brev(i) + 1) p), the value
//! FIDESlib's NTT of the coefficient vector e_p gives (psi = the chain's 2N-th root, OpenFHE order).
use crate::context::{Context, DevStrategy, DevWord};
use crate::ct::Ct;
use crate::error::{shape, Result};
use crate::executor::Executor;
use crate::ks::{ew, Ew, View};
use crate::ntt::{self, Dir, Parts, Span};
use pc_core::modmul::mulmod;
use pc_cuda::{launch_result, Arena, Compute, ConstBuf, Stream};
use pc_kernels::MulTableArgs;
use rayon::prelude::*;

/// Coefficient-form residues of one ciphertext (`c[part][limb][N]`) and its metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct Limbs<W> {
    pub c: [Vec<Vec<W>>; 2],
    pub moduli: Vec<u64>,
    /// FIDESlib top limb index (limbs = level + 1), as perseus' export.
    pub level: usize,
    pub noise_level: u8,
    pub noise_factor: f64,
}

impl<W: DevWord> Limbs<W> {
    /// The metadata in perseus' JSON format (`moduli`, `level`, `noise_level`, `noise_factor`).
    pub fn meta_json(&self) -> String {
        let m: Vec<String> = self.moduli.iter().map(|q| q.to_string()).collect();
        format!(
            "{{\"moduli\": [{}], \"level\": {}, \"noise_level\": {}, \"noise_factor\": {:?}}}",
            m.join(", "),
            self.level,
            self.noise_level,
            self.noise_factor
        )
    }
}

/// Coefficient residues of ciphertext `c` of `x` (copied through `ex.scratch`, x untouched; syncs compute).
pub fn export_limbs<W: DevWord, M: DevStrategy<W>>(ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, x: &Ct<'_, W>, c: usize) -> Result<Limbs<W>> {
    if c >= x.k() {
        return Err(shape("export_limbs", format!("ciphertext {c} of {}", x.k())));
    }
    let top = x.top(ctx.shape)?;
    let s = &ex.compute;
    ew(ctx, s, Ew::Copy, View::of(&ex.scratch, 0), View::of(&x.buf, c), None, ctx.lay.rs_q(top), 1, 2)?;
    ntt::transform_k(ctx, s, Dir::Inverse, &mut ex.scratch, Span::Q { top }, Parts::Both, 1)?;
    let n = ctx.lay.n;
    let ids: Vec<usize> = (0..=top).collect();
    let mut out: [Vec<Vec<W>>; 2] = [Vec::new(), Vec::new()];
    for (part, o) in out.iter_mut().enumerate() {
        let rows = ex.scratch.download_rows(s, 0, part, &ids)?;
        *o = rows.chunks(n).map(|r| r.to_vec()).collect();
    }
    Ok(Limbs { c: out, moduli: ctx.chain.q[..=top].to_vec(), level: top, noise_level: x.meta.deg, noise_factor: x.meta.nf })
}

/// `export_limbs` without the wait: rows 0..=top of both parts of ciphertext `c` of `x` (coefficient form, through
/// `ex.scratch`) queued for download into `dst[part]` (pinned, `(top + 1) N` words each). Returns `x`'s top limb.
///
/// # Safety
/// `dst` must not be read, reused or dropped until the compute lane has passed the copies (sync an event recorded
/// after this call).
pub unsafe fn export_limbs_async<W: DevWord, M: DevStrategy<W>>(ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, x: &Ct<'_, W>, c: usize, dst: [&mut pc_cuda::PinnedBuf<W>; 2]) -> Result<usize> {
    let top = x.top(ctx.shape)?;
    if c >= x.k() || dst.iter().any(|d| d.len() != (top + 1) * ctx.lay.n) {
        return Err(shape("export_limbs_async", format!("ciphertext {c} of {} into buffers of {} words", x.k(), dst[0].len())));
    }
    let s = &ex.compute;
    ew(ctx, s, Ew::Copy, View::of(&ex.scratch, 0), View::of(&x.buf, c), None, ctx.lay.rs_q(top), 1, 2)?;
    ntt::transform_k(ctx, s, Dir::Inverse, &mut ex.scratch, Span::Q { top }, Parts::Both, 1)?;
    for (part, d) in dst.into_iter().enumerate() {
        // SAFETY: the caller keeps d untouched until the lane has passed the copy; rows 0..=top of part `part` of
        // scratch ciphertext 0 are contiguous
        unsafe { ex.scratch.buf.download_pinned_async(s, part * ctx.lay.poly_words(), d)? };
    }
    Ok(top)
}

/// Overwrite ciphertext `c` of `x` (at its current top) from coefficient residues (`export_limbs`' inverse; sets
/// the degree and scale of the whole batch's meta from `l`).
pub fn import_limbs<W: DevWord, M: DevStrategy<W>>(ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, x: &mut Ct<'_, W>, c: usize, l: &Limbs<W>) -> Result<()> {
    let top = x.top(ctx.shape)?;
    if c >= x.k() || l.level != top || l.c.iter().any(|p| p.len() != top + 1 || p.iter().any(|r| r.len() != ctx.lay.n)) {
        return Err(shape("import_limbs", format!("limbs of top {} into ciphertext {c} of {} at top {top}", l.level, x.k())));
    }
    if l.moduli.as_slice() != &ctx.chain.q[..=top] {
        return Err(shape("import_limbs", "moduli differ from the chain's"));
    }
    let s = &ex.compute;
    let ids: Vec<usize> = (0..=top).collect();
    for part in 0..2 {
        let flat: Vec<W> = l.c[part].iter().flatten().copied().collect();
        ex.scratch.upload_rows(s, 0, part, &ids, &flat)?;
    }
    ntt::transform_k(ctx, s, Dir::Forward, &mut ex.scratch, Span::Q { top }, Parts::Both, 1)?;
    ew(ctx, s, Ew::Copy, View::of(&x.buf, c), View::of(&ex.scratch, 0), None, ctx.lay.rs_q(top), 1, 2)?;
    x.meta.deg = l.noise_level;
    x.meta.nf = l.noise_factor;
    Ok(())
}

/// Evaluation rows of X^power for every Q prime (setup; one table per power).
pub struct MonomialTable<'a, W: DevWord> {
    pub power: usize,
    t: ConstBuf<'a, W>,
}

impl<'a, W: DevWord> MonomialTable<'a, W> {
    pub fn bytes(ctx_nq: usize, n: usize) -> usize {
        Arena::bytes_for::<W>(ctx_nq * n)
    }
    /// `power` in [0, 2N) (X^(N + j) = -X^j).
    pub fn new<M: DevStrategy<W>>(arena: &'a Arena, s: &Stream<Compute>, ctx: &Context<'_, W, M>, power: usize) -> Result<Self> {
        let n = ctx.lay.n;
        if power >= 2 * n {
            return Err(shape("MonomialTable", format!("power {power} >= 2N")));
        }
        let logn = n.trailing_zeros();
        let roots = ctx.chain.roots();
        let rows: Vec<Vec<W>> = ctx
            .chain
            .q
            .par_iter()
            .zip(roots.par_iter())
            .map(|(&p, &psi)| {
                // psi^e for e in [0, 2N): table of psi^power steps
                let step = pc_core::modmul::powmod(psi, power as u64, p);
                // value at eval index i: (psi^power)^(2 brev(i) + 1)
                let mut pw = vec![0u64; n];
                let s2 = mulmod(step, step, p);
                let mut v = step;
                for j in 0..n {
                    pw[j] = v; // (psi^power)^(2j + 1)
                    v = mulmod(v, s2, p);
                }
                (0..n).map(|i| W::from_u64(pw[(i as u32).reverse_bits() as usize >> (32 - logn)])).collect()
            })
            .collect();
        let flat: Vec<W> = rows.into_iter().flatten().collect();
        Ok(MonomialTable { power, t: ConstBuf::new(arena, "limbs.monomial", s, &flat)? })
    }
}

/// `ct += X^power * other` over the live Q rows (both at the same top and degree; `other` unchanged). Uses
/// `ex.scratch` for the product.
pub fn merge_monomial<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    ex: &mut Executor<'_, W>,
    mono: &MonomialTable<'_, W>,
    ct: &mut Ct<'_, W>,
    other: &Ct<'_, W>,
) -> Result<()> {
    let k = ct.k();
    let top = ct.top(ctx.shape)?;
    if other.k() != k || other.top(ctx.shape)? != top || other.meta.deg != ct.meta.deg || k > ex.scratch.k {
        return Err(shape("merge_monomial", format!("operands differ (k {} / {}, top {top} / {:?})", k, other.k(), other.top(ctx.shape))));
    }
    let s = &ex.compute;
    let a = MulTableArgs {
        out: View::of(&ex.scratch, 0).r(),
        inp: View::of(&other.buf, 0).r(),
        t: mono.t.vptr(),
        pr: ctx.pr(),
        n: (top + 1) as i32,
        k: k as i32,
    };
    // SAFETY: k-ct views of live buffers; the table covers every Q row
    launch_result("mul_table", unsafe { (ctx.table.mul_table)(&a, s.raw()) })?;
    let v = View::of(&ct.buf, 0);
    ew(ctx, s, Ew::Add, v, v, Some(View::of(&ex.scratch, 0)), ctx.lay.rs_q(top), k, 2)
}
