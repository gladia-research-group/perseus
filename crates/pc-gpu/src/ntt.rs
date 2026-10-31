//! Batched NTT over ciphertext rows (seam `Ntt<W, M>`: the 4-step register-pass kernel, one launch per row
//! chunk of 128 rows inside the launcher). Rows are addressed by tables `(roff, rprime)` in device memory (limb
//! offset in units of N from the buffer base, prime id), built at setup: `RowTables` for the spans every op needs,
//! `RowList` for anything else (key-switching digit sets). Optional: an out-of-place first pass (`Src`) and a forward
//! epilogue stored in place of the transform (`Epi`: the ModDown and rescale last steps).
use crate::context::{Context, DevStrategy, DevWord};
use crate::error::{shape, Result};
use crate::poly::{CtBuf, Layout};
use pc_cuda::{launch_result, Arena, Compute, ConstBuf, DBuf, Lane, Stream};
use pc_kernels::{ntt_epi, NttArgs};
use std::ffi::c_void;

/// Rows per launch chunk (both passes of a chunk run back to back, so the intermediate stays in L2).
pub const NTT_CHUNK: i32 = 128;

/// The live rows of every ciphertext in a span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Span {
    /// Q rows `0..=top`.
    Q { top: usize },
    /// Q rows `0..=top` and every special row.
    QP { top: usize },
    /// Special rows only.
    P,
}

/// Which polynomials of each ciphertext.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Parts {
    Both,
    C0,
    C1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    Forward,
    Inverse,
}

/// Row tables for every `Span` of up to `k_max` ciphertexts of one layout. Per span two sections: `[part][c <
/// k_max][live row]` (the rows of one part of the first `k` ciphertexts are a contiguous prefix of that part's half)
/// and `[c < k_max][part][live row]` (both parts of the first `k` ciphertexts are a prefix: one launch for any k).
pub struct RowTables<'a> {
    roff: ConstBuf<'a, u32>,
    rprime: ConstBuf<'a, i32>,
    /// (start entry, live rows per ciphertext) by span key (`Q{top}` -> top, `QP{top}` -> nq + top, `P` -> 2 nq)
    index: Vec<(usize, usize)>,
    lay: Layout,
    k_max: usize,
}

impl<'a> RowTables<'a> {
    fn spans(lay: &Layout) -> impl Iterator<Item = Span> + '_ {
        (0..lay.nq).map(|top| Span::Q { top }).chain((0..lay.nq).map(|top| Span::QP { top })).chain([Span::P])
    }
    fn ids(lay: &Layout, span: Span) -> Vec<usize> {
        match span {
            Span::Q { top } => (0..=top).collect(),
            Span::QP { top } => (0..=top).chain(lay.nq..lay.rows()).collect(),
            Span::P => (lay.nq..lay.rows()).collect(),
        }
    }
    fn key(&self, span: Span) -> Result<usize> {
        let nq = self.lay.nq;
        match span {
            Span::Q { top } | Span::QP { top } if top >= nq => Err(shape("ntt span", format!("top {top} >= |Q| = {nq}"))),
            Span::Q { top } => Ok(top),
            Span::QP { top } => Ok(nq + top),
            Span::P => Ok(2 * nq),
        }
    }
    /// Table entries for `lay` and `k_max` (MemPlan).
    pub fn entries(lay: &Layout, k_max: usize) -> usize {
        Self::spans(lay).map(|s| 4 * k_max * Self::ids(lay, s).len()).sum()
    }
    pub fn new<S: Lane>(arena: &'a Arena, s: &Stream<S>, lay: Layout, k_max: usize) -> Result<Self> {
        let (mut roff, mut rprime, mut index) = (Vec::new(), Vec::new(), Vec::new());
        let rows = lay.rows();
        for span in Self::spans(&lay) {
            let ids = Self::ids(&lay, span);
            index.push((roff.len(), ids.len()));
            for part in 0..2 {
                for c in 0..k_max {
                    for &id in &ids {
                        roff.push(((c * 2 + part) * rows + id) as u32);
                        rprime.push(id as i32);
                    }
                }
            }
            for c in 0..k_max {
                for part in 0..2 {
                    for &id in &ids {
                        roff.push(((c * 2 + part) * rows + id) as u32);
                        rprime.push(id as i32);
                    }
                }
            }
        }
        Ok(RowTables {
            roff: ConstBuf::new(arena, "ctx.rows.roff", s, &roff)?,
            rprime: ConstBuf::new(arena, "ctx.rows.rprime", s, &rprime)?,
            index,
            lay,
            k_max,
        })
    }
    /// The launches `(first entry, rows)` covering `parts` of the first `k` ciphertexts over `span`.
    pub fn launches(&self, span: Span, parts: Parts, k: usize) -> Result<Vec<(usize, usize)>> {
        if k == 0 || k > self.k_max {
            return Err(shape("ntt k", format!("{k} ciphertexts, row tables cover 1..={} (MemCfg::k_max)", self.k_max)));
        }
        let (start, live) = self.index[self.key(span)?];
        let sec = self.k_max * live;
        Ok(match parts {
            Parts::Both if k == self.k_max => vec![(start, 2 * sec)],
            // one launch (the by-ciphertext section) instead of one per part
            Parts::Both => vec![(start + 2 * sec, 2 * k * live)],
            Parts::C0 => vec![(start, k * live)],
            Parts::C1 => vec![(start + sec, k * live)],
        })
    }
}

/// An arbitrary row list `(limb offset in units of N, prime id)` for buffers other than the canonical ciphertext
/// layout (built at setup by its owner).
pub struct RowList<'a> {
    pub roff: ConstBuf<'a, u32>,
    pub rprime: ConstBuf<'a, i32>,
    /// Largest limb offset + 1 (bounds check against the buffer).
    pub span_limbs: usize,
}

impl<'a> RowList<'a> {
    pub fn new<S: Lane>(arena: &'a Arena, s: &Stream<S>, rows: &[(u32, i32)]) -> Result<Self> {
        let roff: Vec<u32> = rows.iter().map(|r| r.0).collect();
        let rprime: Vec<i32> = rows.iter().map(|r| r.1).collect();
        let span_limbs = roff.iter().map(|&o| o as usize + 1).max().unwrap_or(0);
        Ok(RowList {
            roff: ConstBuf::new(arena, "rowlist.roff", s, &roff)?,
            rprime: ConstBuf::new(arena, "rowlist.rprime", s, &rprime)?,
            span_limbs,
        })
    }
    pub fn len(&self) -> usize {
        self.roff.len()
    }
    pub fn is_empty(&self) -> bool {
        self.roff.is_empty()
    }
}

/// The first pass reads row i at `base + soff[i] N` of another buffer (the transform lands in the row list's buffer):
/// the sum mod p of `n` such rows `stride` words apart.
#[derive(Clone, Copy, Debug)]
pub struct Src {
    pub base: *const c_void,
    pub soff: *const u32,
    pub n: usize,
    pub stride: usize,
}

/// What the forward transform stores instead of v (rows at the row list's offsets from `out` / `x`; scalars by prime
/// id with Shoup companions).
#[derive(Clone, Copy, Debug)]
pub enum Epi {
    /// out = (x - v) s
    SubMul { out: *mut c_void, x: *const c_void, s: *const c_void, ss: *const c_void },
    /// out = d + (x - v) s (d at out's offsets; null: out itself)
    SubMulAcc { out: *mut c_void, x: *const c_void, s: *const c_void, ss: *const c_void, d: *const c_void },
    /// out = x s + v
    MulAdd { out: *mut c_void, x: *const c_void, s: *const c_void, ss: *const c_void },
    /// out = d + (x_0 + .. + x_(n-1) - v) s, the x_j `stride` words apart (d as in SubMulAcc)
    SubMulAccSum { out: *mut c_void, x: *const c_void, n: usize, stride: usize, s: *const c_void, ss: *const c_void, d: *const c_void },
}

impl Epi {
    /// (mode, out, x, x rows summed, their stride, s, ss, d)
    #[allow(clippy::type_complexity)]
    fn args(self) -> (i32, *mut c_void, *const c_void, usize, usize, *const c_void, *const c_void, *const c_void) {
        let nul = std::ptr::null();
        match self {
            Epi::SubMul { out, x, s, ss } => (ntt_epi::SUB_MUL, out, x, 1, 0, s, ss, nul),
            Epi::SubMulAcc { out, x, s, ss, d } => (ntt_epi::SUB_MUL_ACC, out, x, 1, 0, s, ss, d),
            Epi::MulAdd { out, x, s, ss } => (ntt_epi::MUL_ADD, out, x, 1, 0, s, ss, nul),
            Epi::SubMulAccSum { out, x, n, stride, s, ss, d } => (ntt_epi::SUB_MUL_ACC, out, x, n, stride, s, ss, d),
        }
    }
}

fn launch<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    dir: Dir,
    base: *mut c_void,
    roff: *const u32,
    rprime: *const i32,
    nrows: usize,
) -> Result<()> {
    launch_ex(ctx, s, dir, base, roff, rprime, nrows, None, None)
}

#[allow(clippy::too_many_arguments)]
fn launch_ex<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    dir: Dir,
    base: *mut c_void,
    roff: *const u32,
    rprime: *const i32,
    nrows: usize,
    src: Option<Src>,
    epi: Option<Epi>,
) -> Result<()> {
    if nrows == 0 {
        return Ok(());
    }
    if epi.is_some() && dir == Dir::Inverse {
        return Err(shape("ntt epilogue", "the epilogues are forward-only"));
    }
    let a = args_ex(ctx, base, roff, rprime, nrows, src, epi);
    let (f, name) = match dir {
        Dir::Forward => (ctx.table.ntt_fwd, "ntt_fwd"),
        Dir::Inverse => (ctx.table.ntt_inv, "ntt_inv"),
    };
    // SAFETY: the tables are context constants of this chain; the row tables address rows inside `base`'s buffer
    // (checked by the callers); `base` is exclusively borrowed by the caller for the launch's duration on `s`.
    launch_result(name, unsafe { f(&a, s.raw()) })?;
    Ok(())
}

/// The launch arguments of a transform of `nrows` rows (both directions; the epilogue is forward-only).
#[allow(clippy::too_many_arguments)]
fn args_ex<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    base: *mut c_void,
    roff: *const u32,
    rprime: *const i32,
    nrows: usize,
    src: Option<Src>,
    epi: Option<Epi>,
) -> NttArgs {
    let (mode, eout, ex, xn, xs, es, ess, ed) =
        epi.map(Epi::args).unwrap_or((0, std::ptr::null_mut(), std::ptr::null(), 1, 0, std::ptr::null(), std::ptr::null(), std::ptr::null()));
    NttArgs {
        base,
        roff,
        rprime,
        fw: ctx.ntt.fw.vptr(),
        fws: ctx.ntt.fws.vptr(),
        iw: ctx.ntt.iw.vptr(),
        iws: ctx.ntt.iws.vptr(),
        ninv: ctx.ntt.ninv.vptr(),
        ninvs: ctx.ntt.ninvs.vptr(),
        pr: ctx.pr(),
        nrows: nrows as i32,
        chunk: NTT_CHUNK,
        src: src.map(|x| x.base).unwrap_or(std::ptr::null()),
        soff: src.map(|x| x.soff).unwrap_or(std::ptr::null()),
        eout,
        ex,
        es,
        ess,
        epi: mode,
        snum: src.map(|x| x.n.max(1)).unwrap_or(1) as i32,
        sstride: src.map(|x| x.stride).unwrap_or(0) as u64,
        xnum: xn.max(1) as i32,
        lazy: ctx.ntt.lazy,
        xstride: xs as u64,
        ed,
    }
}

/// The forward arguments of entries `[0, nrows)` of `list` over `buf` with `epi`, for a pipeline that runs the
/// transform itself (`rescale_fwd`).
pub fn forward_args<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    buf: &mut DBuf<'_, Compute, W>,
    list: &RowList<'_>,
    nrows: usize,
    epi: Option<Epi>,
) -> Result<NttArgs> {
    if nrows > list.len() || list.span_limbs * ctx.lay.n > buf.len() {
        return Err(shape("ntt rows", format!("{nrows} of {} rows; list spans {} limbs, buffer holds {}", list.len(), list.span_limbs, buf.len() / ctx.lay.n)));
    }
    Ok(args_ex(ctx, buf.vptr_mut(), list.roff.dptr(), list.rprime.dptr(), nrows, None, epi))
}

/// In-place NTT of `parts` over `span` of every ciphertext in `ct` (one or two launches).
pub fn transform<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    dir: Dir,
    ct: &mut CtBuf<'_, Compute, W>,
    span: Span,
    parts: Parts,
) -> Result<()> {
    let k = ct.k;
    transform_k(ctx, s, dir, ct, span, parts, k)
}

/// `transform` over the first `k` ciphertexts of `ct` only.
pub fn transform_k<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    dir: Dir,
    ct: &mut CtBuf<'_, Compute, W>,
    span: Span,
    parts: Parts,
    k: usize,
) -> Result<()> {
    if ct.lay != ctx.lay || k > ct.k {
        return Err(shape("ntt", format!("{k} of {} cts, buffer layout {:?}, context {:?}", ct.k, ct.lay, ctx.lay)));
    }
    let base = ct.buf.vptr_mut();
    for (start, n) in ctx.rows.launches(span, parts, k)? {
        launch(ctx, s, dir, base, ctx.rows.roff.dptr_at(start)?, ctx.rows.rprime.dptr_at(start)?, n)?;
    }
    Ok(())
}

/// Forward `transform_k` storing `epi` of the transform instead (`ct`'s rows hold the intermediate only); the
/// epilogue's buffers share `ct`'s layout (row offsets).
pub fn forward_k_epi<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    ct: &mut CtBuf<'_, Compute, W>,
    span: Span,
    parts: Parts,
    k: usize,
    epi: Epi,
) -> Result<()> {
    if ct.lay != ctx.lay || k > ct.k {
        return Err(shape("ntt", format!("{k} of {} cts, buffer layout {:?}, context {:?}", ct.k, ct.lay, ctx.lay)));
    }
    let base = ct.buf.vptr_mut();
    for (start, n) in ctx.rows.launches(span, parts, k)? {
        launch_ex(ctx, s, Dir::Forward, base, ctx.rows.roff.dptr_at(start)?, ctx.rows.rprime.dptr_at(start)?, n, None, Some(epi))?;
    }
    Ok(())
}

/// `transform_k` with an out-of-place (summing) first pass from `src` at the row tables' offsets (`Src::soff` unused:
/// the context's own row offsets are the source offsets) and an optional forward epilogue.
#[allow(clippy::too_many_arguments)]
pub fn transform_k_ex<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    dir: Dir,
    ct: &mut CtBuf<'_, Compute, W>,
    span: Span,
    parts: Parts,
    k: usize,
    src: Option<Src>,
    epi: Option<Epi>,
) -> Result<()> {
    if ct.lay != ctx.lay || k > ct.k {
        return Err(shape("ntt", format!("{k} of {} cts, buffer layout {:?}, context {:?}", ct.k, ct.lay, ctx.lay)));
    }
    let base = ct.buf.vptr_mut();
    for (start, n) in ctx.rows.launches(span, parts, k)? {
        let roff = ctx.rows.roff.dptr_at(start)?;
        let src = src.map(|x| Src { soff: roff, ..x });
        launch_ex(ctx, s, dir, base, roff, ctx.rows.rprime.dptr_at(start)?, n, src, epi)?;
    }
    Ok(())
}

/// Inverse `transform_k_ex` with row i's last-pass scale `scale.0[i]` (Shoup `scale.1[i]`) in place of N^-1: the
/// tables hold one entry per row of the span's launches, periodic in the span's live rows (any launch start).
#[allow(clippy::too_many_arguments)]
pub fn inverse_k_scaled<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    ct: &mut CtBuf<'_, Compute, W>,
    span: Span,
    parts: Parts,
    k: usize,
    src: Option<Src>,
    scale: (*const c_void, *const c_void),
) -> Result<()> {
    if ct.lay != ctx.lay || k > ct.k {
        return Err(shape("ntt", format!("{k} of {} cts, buffer layout {:?}, context {:?}", ct.k, ct.lay, ctx.lay)));
    }
    let base = ct.buf.vptr_mut();
    for (start, n) in ctx.rows.launches(span, parts, k)? {
        let roff = ctx.rows.roff.dptr_at(start)?;
        let mut a = args_ex(ctx, base, roff, ctx.rows.rprime.dptr_at(start)?, n, src.map(|x| Src { soff: roff, ..x }), None);
        (a.es, a.ess) = scale;
        // SAFETY: as `launch_ex`; the scale tables hold an entry for every row of the launch
        launch_result("ntt_inv", unsafe { (ctx.table.ntt_inv)(&a, s.raw()) })?;
    }
    Ok(())
}

pub fn forward<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    ct: &mut CtBuf<'_, Compute, W>,
    span: Span,
    parts: Parts,
) -> Result<()> {
    transform(ctx, s, Dir::Forward, ct, span, parts)
}

pub fn inverse<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    ct: &mut CtBuf<'_, Compute, W>,
    span: Span,
    parts: Parts,
) -> Result<()> {
    transform(ctx, s, Dir::Inverse, ct, span, parts)
}

/// In-place NTT of the rows of `list` in an arbitrary compute buffer (limb offsets relative to `buf`).
pub fn transform_rows<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    dir: Dir,
    buf: &mut DBuf<'_, Compute, W>,
    list: &RowList<'_>,
) -> Result<()> {
    if list.span_limbs * ctx.lay.n > buf.len() {
        return Err(shape("ntt rows", format!("row list spans {} limbs, buffer holds {}", list.span_limbs, buf.len() / ctx.lay.n)));
    }
    launch(ctx, s, dir, buf.vptr_mut(), list.roff.dptr(), list.rprime.dptr(), list.len())
}

/// `transform_rows` over the first `nrows` entries of `list` only (lists ordered by ciphertext serve every k).
pub fn transform_rows_prefix<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    dir: Dir,
    buf: &mut DBuf<'_, Compute, W>,
    list: &RowList<'_>,
    nrows: usize,
) -> Result<()> {
    if nrows > list.len() || list.span_limbs * ctx.lay.n > buf.len() {
        return Err(shape("ntt rows", format!("{nrows} of {} rows; list spans {} limbs, buffer holds {}", list.len(), list.span_limbs, buf.len() / ctx.lay.n)));
    }
    launch(ctx, s, dir, buf.vptr_mut(), list.roff.dptr(), list.rprime.dptr(), nrows)
}

/// Inverse `transform_rows_prefix_ex` with row i's last-pass scale `scale.0[i]` (Shoup `scale.1[i]`, W words of one
/// entry per row of `list`) in place of N^-1.
pub fn inverse_rows_prefix_scaled<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    buf: &mut DBuf<'_, Compute, W>,
    list: &RowList<'_>,
    nrows: usize,
    src: Option<Src>,
    scale: (*const c_void, *const c_void),
) -> Result<()> {
    if nrows > list.len() || list.span_limbs * ctx.lay.n > buf.len() {
        return Err(shape("ntt rows", format!("{nrows} of {} rows; list spans {} limbs, buffer holds {}", list.len(), list.span_limbs, buf.len() / ctx.lay.n)));
    }
    if nrows == 0 {
        return Ok(());
    }
    let mut a = args_ex(ctx, buf.vptr_mut(), list.roff.dptr(), list.rprime.dptr(), nrows, src, None);
    (a.es, a.ess) = scale;
    // SAFETY: as `launch_ex`; the scale tables hold one W word per row of `list` (>= nrows)
    launch_result("ntt_inv", unsafe { (ctx.table.ntt_inv)(&a, s.raw()) })?;
    Ok(())
}

/// `transform_rows_prefix` with an out-of-place first pass (`src`: one offset per entry of `list`, read before any
/// row of `buf` is written) and an optional forward epilogue.
#[allow(clippy::too_many_arguments)]
pub fn transform_rows_prefix_ex<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    dir: Dir,
    buf: &mut DBuf<'_, Compute, W>,
    list: &RowList<'_>,
    nrows: usize,
    src: Option<Src>,
    epi: Option<Epi>,
) -> Result<()> {
    if nrows > list.len() || list.span_limbs * ctx.lay.n > buf.len() {
        return Err(shape("ntt rows", format!("{nrows} of {} rows; list spans {} limbs, buffer holds {}", list.len(), list.span_limbs, buf.len() / ctx.lay.n)));
    }
    launch_ex(ctx, s, dir, buf.vptr_mut(), list.roff.dptr(), list.rprime.dptr(), nrows, src, epi)
}
