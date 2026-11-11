//! Subring switch: a bottom-level ciphertext switched to a pack secret so that its c1 lies in Z[X^sub]
//! (`pc_client::client::pack`: the key, the relation, the frames).
//!
//! c1 is raised to the bottom key's modulus (the level's ModUp without its NTT: one digit, the special rows in
//! coefficient form), split by coefficient class into `sub` digits (`pack_split`, into the unused digits of
//! `KsWork::digits`), the digits transformed and dotted against the key (`pack_dot`), then the level's ModDown. The
//! key's `a` rows are zero off the subring in coefficient form, so c1 comes out with exact zeros there.
use super::{check_cts, level_scalars, moddown_to, KsPlans, MdOut, View};
use crate::context::{Context, DevStrategy, DevWord};
use crate::error::{shape, Result};
use crate::ntt::{self, Dir, RowList};
use crate::poly::CtBuf;
use crate::Executor;
use pc_cuda::{launch_result, Arena, Compute, ConstBuf, DBuf, Lane, Stream};
use pc_kernels::{PackDotArgs, PackSplitArgs};
use std::ffi::c_void;

/// Tables of the subring switch at the bottom level for `KsPlans::k_max` ciphertexts (setup).
pub struct PackPlan<'a> {
    pub sub: usize,
    /// The bottom level's top limb.
    pub top: usize,
    /// Prime ids of the key's rows: Q rows 0..=top, then the bottom key's special rows.
    pub ids: Vec<usize>,
    rows: ConstBuf<'a, u32>,
    /// Forward rows of the split digits, `[c < k_max][digit][row]` (prefix by ciphertext).
    ntt: RowList<'a>,
}

impl<'a> PackPlan<'a> {
    pub fn ids(chain: &pc_core::params::Chain) -> Vec<usize> {
        let top = chain.level_unit - 1;
        (0..=top).chain(chain.special_ids().into_iter().take(chain.specials_at(top))).collect()
    }
    pub fn bytes(chain: &pc_core::params::Chain, k_max: usize, sub: usize) -> usize {
        let rows = Self::ids(chain).len();
        Arena::bytes_for::<u32>(rows) + super::rowlist_bytes(k_max * sub * rows)
    }
    pub fn new<W: DevWord, M: DevStrategy<W>, S: Lane>(arena: &'a Arena, s: &Stream<S>, ctx: &Context<'_, W, M>, plans: &KsPlans<'_>, sub: usize) -> Result<Self> {
        let (lay, top) = (ctx.lay, ctx.chain.level_unit - 1);
        let ids = Self::ids(&ctx.chain);
        let nrows = ids.len();
        let lv = plans.level(top)?;
        if !(2..=16).contains(&sub) || !sub.is_power_of_two() {
            return Err(shape("pack plan", format!("subring stride {sub} (2, 4, 8 or 16)")));
        }
        // one digit raised to the special rows (contiguous ids), the split digits in the digits after it
        if lv.ndig != 1 || lv.kp != ctx.chain.p_sparse || nrows != top + 1 + lv.kp || ids[top + 1..].windows(2).any(|w| w[1] != w[0] + 1) || plans.dnum < 2 || (plans.dnum - 1) * lay.rows() < sub * nrows {
            return Err(shape("pack plan", format!("bottom level of {} digits over {} special rows, {} digit slots of {} rows", lv.ndig, lv.kp, plans.dnum, lay.rows())));
        }
        let list: Vec<(u32, i32)> = (0..plans.k_max)
            .flat_map(|c| (0..sub * nrows).map(move |i| (c, i)))
            .map(|(c, i)| (((c * plans.dnum + 1) * lay.rows() + i) as u32, ids[i % nrows] as i32))
            .collect();
        let dev: Vec<u32> = ids.iter().map(|&t| t as u32).collect();
        Ok(PackPlan { sub, top, rows: ConstBuf::new(arena, "ks.pack", s, &dev)?, ntt: RowList::new(arena, s, &list)?, ids })
    }
}

/// A pack key on the device: `b` and `a`, `[digit][row][N]` over the plan's rows, evaluation form.
pub struct PackKey<'a, W: DevWord> {
    b: DBuf<'a, Compute, W>,
    a: DBuf<'a, Compute, W>,
}

impl<'a, W: DevWord> PackKey<'a, W> {
    pub fn bytes(n: usize, sub: usize, rows: usize) -> usize {
        2 * Arena::bytes_for::<W>(sub * rows * n)
    }
    pub fn carve(arena: &'a Arena, n: usize, sub: usize, rows: usize) -> Result<Self> {
        Ok(PackKey { b: arena.alloc("keys.pack", sub * rows * n)?, a: arena.alloc("keys.pack", sub * rows * n)? })
    }
    /// The key's words (`Client::pack_keygen`'s `b`, `pack_a_eval`'s `a`).
    pub fn load(&mut self, s: &Stream<Compute>, b: &[u32], a: &[u32]) -> Result<()> {
        if b.len() != self.b.len() || a.len() != self.a.len() {
            return Err(shape("pack key", format!("{} + {} words, the slot holds 2 x {}", b.len(), a.len(), self.b.len())));
        }
        let up = |x: &[u32]| x.iter().map(|&v| W::from_u64(v as u64)).collect::<Vec<W>>();
        self.b.upload(s, &up(b))?;
        self.a.upload(s, &up(a))?;
        Ok(())
    }
}

/// The first `k` ciphertexts of `x` (bottom level) switched to the secret of `key`: c1 in the subring.
pub fn pack_switch<W: DevWord, M: DevStrategy<W>>(ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, plans: &KsPlans<'_>, pp: &PackPlan<'_>, key: &PackKey<'_, W>, x: &mut CtBuf<'_, Compute, W>, k: usize) -> Result<()> {
    plans.check_k(k)?;
    check_cts(x, 0, k, "pack switch")?;
    let (lay, top, sub, nrows) = (ctx.lay, pp.top, pp.sub, pp.ids.len());
    let (n, pw) = (lay.n, lay.poly_words());
    if key.b.len() != sub * nrows * n || ex.bconv_work.len() < k * pw {
        return Err(shape("pack switch", format!("key of {} words for stride {sub} over {nrows} rows; workspace of {} words", key.b.len(), ex.bconv_work.len())));
    }
    let lv = plans.level(top)?;
    let s = &ex.compute;
    let v = View::of(x, 0);
    // c1's rows in coefficient form, its special rows from the level's conversion (left in coefficient form)
    let soff = lv.intt_src.dptr();
    let src = ntt::Src { base: v.part::<W>(1).base, soff, n: 1, stride: 0 };
    ntt::transform_rows_prefix_ex(ctx, s, Dir::Inverse, &mut ex.bconv_work, &lv.intt, k * (top + 1), Some(src), None)?;
    lv.modup.run_prefix(ctx, s, &ex.bconv_work, 0, &mut ex.ks.digits, 0, k)?;
    let digits = ex.ks.digits.vptr_mut() as *mut W;
    let dstride = (plans.dnum * pw) as u64;
    let split = PackSplitArgs {
        q: ex.bconv_work.vptr(),
        p: digits.wrapping_add(pp.ids[top + 1] * n) as *const c_void,
        out: digits.wrapping_add(pw) as *mut c_void,
        q_stride: pw as u64,
        p_stride: dstride,
        out_stride: dstride,
        k: k as i32,
        sub: sub as i32,
        nq: (top + 1) as i32,
        nrows: nrows as i32,
    };
    // SAFETY: rows of the first k polynomials of the coefficient workspace and of digits 0 (sources) and 1.. (the
    // split digits, `PackPlan::new` checked their room) of the first k ciphertexts of `ks.digits` (`check_k`)
    launch_result("pack_split", unsafe { (ctx.table.pack_split)(&split, s.raw()) })?;
    ntt::transform_rows_prefix(ctx, s, Dir::Forward, &mut ex.ks.digits, &pp.ntt, k * sub * nrows)?;
    let [pmod, pmods, _, _] = level_scalars(ctx, lv);
    let out0 = View::of(&ex.ks.ext, 0).base as *mut W;
    let dot = PackDotArgs {
        dig: digits.wrapping_add(pw) as *const c_void,
        kb: key.b.vptr(),
        ka: key.a.vptr(),
        c0: v.base,
        out0: out0 as *mut c_void,
        out1: out0.wrapping_add(pw) as *mut c_void,
        rows: pp.rows.dptr(),
        pmod,
        pmods,
        pr: ctx.pr(),
        dig_stride: dstride,
        ct_stride: v.ct_stride,
        out_ct_stride: lay.ct_words() as u64,
        k: k as i32,
        sub: sub as i32,
        nq: (top + 1) as i32,
        nrows: nrows as i32,
    };
    // SAFETY: the split digits above, the key's rows (length checked), c0 of the first k ciphertexts of x
    // (`check_cts`), the plan's rows of the first k extended ciphertexts (`check_k`: ext holds k_max)
    launch_result("pack_dot", unsafe { (ctx.table.pack_dot)(&dot, s.raw()) })?;
    moddown_to(ctx, s, plans, &mut ex.ks.ext, &mut ex.ks.conv, v, MdOut::Set, k, top)
}
