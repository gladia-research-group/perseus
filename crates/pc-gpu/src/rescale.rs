//! Rescaling (seam `Rescaler`): drop the top `level_unit` primes of a batch of ciphertexts and divide by their
//! product, in the NTT domain, with the centered lift of FIDESlib `Ciphertext::rescale`.
//!
//! Two implementations, bit-identical to each other and to FIDESlib (`pc-check leaf --ops rescale` runs both):
//! * `CompositeOnePass` (EvalMod path): INTT the dropped tops, one `rescale_coef` pass computes the
//!   combined correction for every target row in coefficient form, one NTT of the target rows, one
//!   `rescale_combine` (`x_i = alpha_i x_i + tmp_i`). Optional scalar fold: a `ScalarAdjust` row vector `c` rides
//!   the rescale (`ca`, `cb` scale the tops, `alpha_i c_i` the targets); exact, so it equals scale-then-rescale.
//! * `Sequential`: OpenFHE order, one prime at a time (`d` passes of the same kernels with `ndrop = 1`).
//!
//! Level drops (`drop_to_level`) move no data: rows above the live top are dead in the full-row layout, so a drop
//! is a metadata change and exact by construction.
use crate::context::{Context, DevStrategy, DevWord};
use crate::error::{shape, Result};
use crate::mem::MemPlan;
use crate::ntt::{self, Dir, RowList};
use crate::poly::{CtBuf, Layout};
use pc_core::modmul::{mulmod, shoup};
use pc_cuda::{launch_result, Arena, Compute, ConstBuf, Lane, Stream};
use pc_kernels::{EwArgs, PolyRef, RescaleArgs, RescaleCombineArgs, RescaleCombineToArgs, RowSet};
use std::ffi::c_void;

/// Which rescale algorithm (a config choice; both produce identical words).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RescaleKind {
    CompositeOnePass,
    Sequential,
}

/// Per-top constant table `[top][prime id]` with Shoup companions.
struct TopTable<'a, W: DevWord> {
    v: ConstBuf<'a, W>,
    s: ConstBuf<'a, W>,
    rows: usize,
}

impl<'a, W: DevWord> TopTable<'a, W> {
    fn new<S: Lane>(arena: &'a Arena, what: &'static str, s: &Stream<S>, primes: &[u64], vals: &[Vec<u64>]) -> Result<Self> {
        let rows = primes.len();
        let (mut v, mut sh) = (Vec::with_capacity(vals.len() * rows), Vec::with_capacity(vals.len() * rows));
        for t in vals {
            for (i, &p) in primes.iter().enumerate() {
                let c = shoup(W::from_u64(t.get(i).copied().unwrap_or(0) % p), W::from_u64(p));
                v.push(c.w);
                sh.push(c.w_prime);
            }
        }
        Ok(TopTable { v: ConstBuf::new(arena, what, s, &v)?, s: ConstBuf::new(arena, what, s, &sh)?, rows })
    }
    fn at(&self, top: usize) -> Result<(*const c_void, *const c_void)> {
        Ok((self.v.dptr_at(top * self.rows)? as *const c_void, self.s.dptr_at(top * self.rows)? as *const c_void))
    }
}

/// Host copy of a composite drop's scalars (the fold multiplies `alpha` on the host).
#[derive(Clone, Debug)]
struct DropConsts {
    /// `alpha_i` by prime id, rows `0..top - 1`.
    alpha: Vec<u64>,
    qinv_ab: u64,
    c_ab: u64,
}

struct CompTables<'a, W: DevWord> {
    a: TopTable<'a, W>,
    b: TopTable<'a, W>,
    alpha: TopTable<'a, W>,
    qam: ConstBuf<'a, W>,
    qbm: ConstBuf<'a, W>,
}

/// Rescale tables and workspace, built at setup (`Rescaler::bytes` sizes the `MemPlan` section "rescale").
pub struct Rescaler<'a, W: DevWord> {
    pub kind: RescaleKind,
    lay: Layout,
    d: usize,
    k_max: usize,
    comp: Option<CompTables<'a, W>>,
    s_a: TopTable<'a, W>,
    s_alpha: TopTable<'a, W>,
    s_qam: ConstBuf<'a, W>,
    /// INTT row lists of the dropped tops for the first k ciphertexts: `intt2[top][k-1]` (rows top-1, top) and
    /// `intt1[top][k-1]` (row top).
    intt2: Vec<Vec<RowList<'a>>>,
    intt1: Vec<Vec<RowList<'a>>>,
    /// NTT row lists of the correction rows `0..=t` of the first k ciphertexts of `tmp`, both parts in one list
    /// (one launch instead of the context tables' one per part when k < k_max): `fwd_tmp[t][k-1]`.
    fwd_tmp: Vec<Vec<RowList<'a>>>,
    host: Vec<Option<DropConsts>>,
    /// `k_max` ciphertexts of the context layout: the correction rows (coefficient form, then NTT'd).
    tmp: CtBuf<'a, Compute, W>,
}

fn rowlists_for<'a, S: Lane>(arena: &'a Arena, s: &Stream<S>, lay: &Layout, k_max: usize, rows: &[usize]) -> Result<Vec<RowList<'a>>> {
    (1..=k_max)
        .map(|k| {
            let mut e = Vec::with_capacity(k * 2 * rows.len());
            for c in 0..k {
                for part in 0..2 {
                    for &r in rows {
                        e.push((((c * 2 + part) * lay.rows() + r) as u32, r as i32));
                    }
                }
            }
            RowList::new(arena, s, &e)
        })
        .collect()
}

impl<'a, W: DevWord> Rescaler<'a, W> {
    /// Device bytes of the tables, row lists and workspace (the `MemPlan` section "rescale").
    pub fn bytes(plan: &MemPlan) -> usize {
        let lay = plan.lay;
        let (nq, np, k) = (lay.nq, lay.rows(), plan.cfg.k_max);
        let w = |n: usize| Arena::bytes_for::<W>(n);
        // 5 top tables (value + Shoup) and 3 plain [top][prime] tables
        let tables = 5 * 2 * w(nq * np) + 3 * w(nq * np);
        let mut lists = 0;
        for top in 0..nq {
            for kk in 1..=k {
                for rows in [if top >= 1 { 2usize } else { 1 }, 1, top + 1] {
                    lists += Arena::bytes_for::<u32>(kk * 2 * rows) + Arena::bytes_for::<i32>(kk * 2 * rows);
                }
            }
        }
        tables + lists + w(CtBuf::<Compute, W>::words(&lay, k))
    }

    pub fn new<S: Lane, M: DevStrategy<W>>(arena: &'a Arena, s: &Stream<S>, ctx: &Context<'_, W, M>, kind: RescaleKind) -> Result<Self> {
        let lay = ctx.lay;
        let (nq, k_max) = (lay.nq, ctx.k_max);
        let d = ctx.chain.level_unit;
        let st = &ctx.scale;
        let primes = ctx.chain.primes();
        let np = primes.len();
        let q = &ctx.chain.q;
        if d > 2 && kind == RescaleKind::CompositeOnePass {
            return Err(shape("rescale.kind", format!("CompositeOnePass drops at most 2 primes, the chain's level unit is {d}")));
        }
        let cv = |v: &[u64]| v.iter().map(|&x| W::from_u64(x)).collect::<Vec<W>>();
        // single-prime drop from top t: x_i' = qinv_t(i) x_i + qlql_t(i) lift(x_t)
        let (mut a1, mut al1, mut qam1) = (vec![vec![0u64; np]; nq], vec![vec![0u64; np]; nq], vec![0u64; nq * np]);
        for t in 0..nq {
            for i in 0..t {
                a1[t][i] = st.qlql[t][i];
                al1[t][i] = st.q_inv[t][i];
            }
            for (i, &p) in primes.iter().enumerate() {
                qam1[t * np + i] = q[t] % p;
            }
        }
        let s_a = TopTable::new(arena, "rescale.s_a", s, &primes, &a1)?;
        let s_alpha = TopTable::new(arena, "rescale.s_alpha", s, &primes, &al1)?;
        let s_qam = ConstBuf::new(arena, "rescale.s_qam", s, &cv(&qam1))?;

        // composite drop of a = t, b = t - 1 (args.cuh RescaleArgs)
        let mut host = vec![None; nq];
        let comp = if d == 2 {
            let (mut av, mut bv, mut alv) = (vec![vec![0u64; np]; nq], vec![vec![0u64; np]; nq], vec![vec![0u64; np]; nq]);
            let (mut qam, mut qbm) = (vec![0u64; nq * np], vec![0u64; nq * np]);
            for t in 2..nq {
                let (ta, tb) = (t, t - 1);
                for i in 0..tb {
                    let qi = q[i];
                    av[t][i] = mulmod(st.q_inv[tb][i], st.qlql[ta][i], qi);
                    bv[t][i] = st.qlql[tb][i];
                    alv[t][i] = mulmod(st.q_inv[tb][i], st.q_inv[ta][i], qi);
                }
                for (i, &p) in primes.iter().enumerate() {
                    qam[t * np + i] = q[ta] % p;
                    qbm[t * np + i] = q[tb] % p;
                }
                host[t] = Some(DropConsts { alpha: alv[t][..tb].to_vec(), qinv_ab: st.q_inv[ta][tb], c_ab: st.qlql[ta][tb] });
            }
            Some(CompTables {
                a: TopTable::new(arena, "rescale.a", s, &primes, &av)?,
                b: TopTable::new(arena, "rescale.b", s, &primes, &bv)?,
                alpha: TopTable::new(arena, "rescale.alpha", s, &primes, &alv)?,
                qam: ConstBuf::new(arena, "rescale.qam", s, &cv(&qam))?,
                qbm: ConstBuf::new(arena, "rescale.qbm", s, &cv(&qbm))?,
            })
        } else {
            None
        };
        let mut intt2 = Vec::with_capacity(nq);
        let mut intt1 = Vec::with_capacity(nq);
        for t in 0..nq {
            let two: Vec<usize> = if t >= 1 { vec![t - 1, t] } else { vec![t] };
            intt2.push(rowlists_for(arena, s, &lay, k_max, &two)?);
            intt1.push(rowlists_for(arena, s, &lay, k_max, &[t])?);
        }
        let fwd_tmp = (0..nq).map(|t| rowlists_for(arena, s, &lay, k_max, &(0..=t).collect::<Vec<_>>())).collect::<Result<Vec<_>>>()?;
        let tmp = CtBuf::new(arena, "rescale.tmp", lay, k_max)?;
        s.sync()?;
        Ok(Rescaler { kind, lay, d, k_max, comp, s_a, s_alpha, s_qam, intt2, intt1, fwd_tmp, host, tmp })
    }

    /// Whether `rescale` accepts a scalar fold (composite one-pass on a 2-prime level unit).
    pub fn can_fold(&self) -> bool {
        self.kind == RescaleKind::CompositeOnePass && self.d == 2
    }

    /// The fold's target scalars `alpha_i c_i` (host side, rows `0..top - 1`; the caller uploads them with Shoup
    /// companions and passes the device pointers to `rescale`).
    pub fn folded_alpha(&self, top: usize, c: &[u64], q: &[u64]) -> Result<Vec<u64>> {
        let h = self.host.get(top).and_then(|h| h.as_ref()).ok_or_else(|| shape("rescale fold", format!("no composite drop from top {top}")))?;
        Ok(h.alpha.iter().zip(c).zip(q).map(|((&a, &ci), &qi)| mulmod(a, ci % qi, qi)).collect())
    }

    /// Rescale the `x.k` ciphertexts of `x` (live top limb `top`) by the chain's level unit. `fold` = (`c` rows by
    /// prime id over `0..=top`, device `alpha c` values, their Shoup companions): rescale `c * x` in the same pass
    /// (composite only; the kernel scales the tops only when `c[top] != 1`, so callers fold only then).
    pub fn rescale<M: DevStrategy<W>>(
        &mut self,
        ctx: &Context<'_, W, M>,
        s: &Stream<Compute>,
        x: &mut CtBuf<'_, Compute, W>,
        top: usize,
        fold: Option<(&[u64], *const c_void, *const c_void)>,
    ) -> Result<()> {
        if x.lay != self.lay {
            return Err(shape("rescale", format!("buffer layout {:?} is not the context's {:?}", x.lay, self.lay)));
        }
        if x.k == 0 || x.k > self.k_max {
            return Err(shape("rescale", format!("{} ciphertexts, the rescaler covers 1..={} (MemCfg::k_max)", x.k, self.k_max)));
        }
        if top < self.d || top >= self.lay.nq {
            return Err(shape("rescale", format!("top limb {top} has no {} primes to drop", self.d)));
        }
        if self.kind == RescaleKind::CompositeOnePass && self.d == 2 {
            return self.composite_from(ctx, s, x, None, top, fold);
        }
        if fold.is_some() {
            return Err(shape("rescale fold", "the scalar fold is a composite one-pass option"));
        }
        for j in 0..self.d {
            self.single(ctx, s, x, top - j)?;
        }
        Ok(())
    }

    /// Out-of-place `rescale`: `dst` (k cts) = the rescale of `src` (live top limb `top`), `src` untouched (no
    /// full copy before an operand's Rescale prelude). Only the two dropped tops are copied into `dst` (their
    /// INTT and `rescale_coef` run there), then `rescale_combine_to` writes `dst = alpha src + tmp`: the same words
    /// as copy + `rescale`. Composite one-pass on a 2-prime unit only (callers fall back to copy + `rescale`).
    pub fn rescale_to<M: DevStrategy<W>>(
        &mut self,
        ctx: &Context<'_, W, M>,
        s: &Stream<Compute>,
        src: PolyRef,
        dst: &mut CtBuf<'_, Compute, W>,
        top: usize,
        fold: Option<(&[u64], *const c_void, *const c_void)>,
    ) -> Result<()> {
        if !self.can_fold() {
            return Err(shape("rescale_to", "out-of-place rescale is a composite one-pass option"));
        }
        if dst.lay != self.lay || dst.k == 0 || dst.k > self.k_max || top < self.d || top >= self.lay.nq {
            return Err(shape("rescale_to", format!("{} cts at top {top}", dst.k)));
        }
        let k = dst.k;
        let n = self.lay.n;
        let off = |r: PolyRef| PolyRef { base: (r.base as *mut W).wrapping_add((top - 1) * n) as *mut c_void, ..r };
        let rs = RowSet { nq: 2, np: 0, qmax: self.lay.nq as i32, pad: 0 };
        let cp = EwArgs { out: off(dst.w()), a: off(src), b: off(src), rs, sc: std::ptr::null(), scs: std::ptr::null(), pr: ctx.pr(), k: k as i32, nparts: 2, sub: 0, zero_p: 0 };
        // SAFETY: rows top-1, top of k cts of two context-layout buffers (the caller's src holds >= k cts at top)
        launch_result("ew_copy", unsafe { (ctx.table.ew_copy)(&cp, s.raw()) })?;
        self.composite_from(ctx, s, dst, Some(src), top, fold)
    }

    /// `src` = Some: x holds only the tops (rows top-1, top); the targets come from `src` (`rescale_combine_to`).
    fn composite_from<M: DevStrategy<W>>(
        &mut self,
        ctx: &Context<'_, W, M>,
        s: &Stream<Compute>,
        x: &mut CtBuf<'_, Compute, W>,
        src: Option<PolyRef>,
        top: usize,
        fold: Option<(&[u64], *const c_void, *const c_void)>,
    ) -> Result<()> {
        let k = x.k;
        let np = self.lay.rows();
        let ct = self.comp.as_ref().ok_or_else(|| shape("rescale", "no composite tables (level unit != 2)"))?;
        let h = self.host[top].as_ref().ok_or_else(|| shape("rescale", format!("no composite drop from top {top}")))?;
        ntt::transform_rows(ctx, s, Dir::Inverse, &mut x.buf, &self.intt2[top][k - 1])?;
        let (a, a_s) = ct.a.at(top)?;
        let (b, b_s) = ct.b.at(top)?;
        let (ca, cb, al, als) = match fold {
            Some((c, alv, als)) => {
                let (ca, cb) = (c[top] % ctx.chain.q[top], c[top - 1] % ctx.chain.q[top - 1]);
                if ca == 1 && cb != 1 {
                    return Err(shape("rescale fold", "c[top] == 1 with c[top-1] != 1: the kernel would skip the top scaling"));
                }
                (ca, cb, alv, als)
            }
            None => {
                let (al, als) = ct.alpha.at(top)?;
                (1, 1, al, als)
            }
        };
        let args = RescaleArgs {
            x: x.w(),
            tmp: self.tmp.w(),
            a,
            b,
            a_s,
            b_s,
            qam: ct.qam.dptr_at(top * np)? as *const c_void,
            qbm: ct.qbm.dptr_at(top * np)? as *const c_void,
            pr: ctx.pr(),
            qinv_ab: h.qinv_ab,
            c_ab: h.c_ab,
            ca,
            cb,
            top: top as i32,
            nrows: (top - 1) as i32,
            k: k as i32,
            ndrop: 2,
        };
        // SAFETY: x (k cts) and tmp (k_max >= k cts) are distinct context-layout buffers, exclusively borrowed; the
        // tables are context constants of this chain
        launch_result("rescale_coef", unsafe { (ctx.table.rescale_coef)(&args, s.raw()) })?;
        match src {
            None => self.ntt_tmp_and_combine(ctx, s, x, top - 2, al, als),
            Some(src) if src.ct_stride == self.lay.ct_words() as u64 && src.part_stride == self.lay.poly_words() as u64 => {
                // x = alpha src + NTT(tmp), the combine in the transform's epilogue (src shares tmp's row offsets)
                let new_top = top - 2;
                let epi = ntt::Epi::MulAdd { out: x.buf.vptr_mut(), x: src.base, s: al, ss: als };
                ntt::transform_rows_prefix_ex(ctx, s, Dir::Forward, &mut self.tmp.buf, &self.fwd_tmp[new_top][k - 1], 2 * k * (new_top + 1), None, Some(epi))
            }
            Some(src) => {
                let new_top = top - 2;
                ntt::transform_rows(ctx, s, Dir::Forward, &mut self.tmp.buf, &self.fwd_tmp[new_top][k - 1])?;
                let a = RescaleCombineArgs { x: src, tmp: self.tmp.r(), alpha: al, alphas: als, pr: ctx.pr(), n: (new_top + 1) as i32, k: k as i32 };
                let args = RescaleCombineToArgs { a, out: x.w() };
                // SAFETY: rows 0..=new_top of k cts in src, tmp and x (x distinct from src: the caller's out-of-place
                // destination); alpha as in `ntt_tmp_and_combine`
                launch_result("rescale_combine_to", unsafe { (ctx.table.rescale_combine_to)(&args, s.raw()) })?;
                Ok(())
            }
        }
    }

    fn single<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, x: &mut CtBuf<'_, Compute, W>, top: usize) -> Result<()> {
        let k = x.k;
        let np = self.lay.rows();
        ntt::transform_rows(ctx, s, Dir::Inverse, &mut x.buf, &self.intt1[top][k - 1])?;
        let (a, a_s) = self.s_a.at(top)?;
        let args = RescaleArgs {
            x: x.w(),
            tmp: self.tmp.w(),
            a,
            b: std::ptr::null(),
            a_s,
            b_s: std::ptr::null(),
            qam: self.s_qam.dptr_at(top * np)? as *const c_void,
            qbm: std::ptr::null(),
            pr: ctx.pr(),
            qinv_ab: 0,
            c_ab: 0,
            ca: 1,
            cb: 1,
            top: top as i32,
            nrows: top as i32,
            k: k as i32,
            ndrop: 1,
        };
        // SAFETY: as in `composite`
        launch_result("rescale_coef", unsafe { (ctx.table.rescale_coef)(&args, s.raw()) })?;
        let (al, als) = self.s_alpha.at(top)?;
        self.ntt_tmp_and_combine(ctx, s, x, top - 1, al, als)
    }

    fn ntt_tmp_and_combine<M: DevStrategy<W>>(
        &mut self,
        ctx: &Context<'_, W, M>,
        s: &Stream<Compute>,
        x: &mut CtBuf<'_, Compute, W>,
        new_top: usize,
        al: *const c_void,
        als: *const c_void,
    ) -> Result<()> {
        let k = x.k;
        // x = alpha x + NTT(tmp): the combine rides the transform's epilogue (x shares tmp's row offsets); alpha is a
        // context table or a constant-ring slot written earlier on this stream
        let epi = ntt::Epi::MulAdd { out: x.buf.vptr_mut(), x: x.buf.vptr(), s: al, ss: als };
        let list = &self.fwd_tmp[new_top][k - 1];
        ntt::transform_rows_prefix_ex(ctx, s, Dir::Forward, &mut self.tmp.buf, list, list.len(), None, Some(epi))
    }
}
