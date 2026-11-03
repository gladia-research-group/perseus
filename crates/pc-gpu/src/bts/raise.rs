//! Seam `Raise`: the mod raise of a bootstrap (FIDESlib `Bootstrap.cu::ModRaise` with the sparse-secret
//! encapsulation and the exact post-raise constant of `SmallInt.cu`), bit-exact with FIDESlib: (1) bring the input to
//! the composite bottom (FLEXIBLEAUTO adjust + rescale, drop to top limb d - 1); (2) dense -> sparse key switch at the
//! bottom; (3) coefficient form, composite raise (`mod_raise`) and exact small-integer scaling (`small_int`), NTT;
//! (4) sparse -> dense key switch at the raised level. On composite degree 1 (n64) step 3 is one kernel
//! (`small_int1`, exact rounding of x * c; not FIDESlib-bit-exact). `Encapsulated` is the only implementation; the
//! per-drop variants differ only in the target top limb `L - d * raise_drop` and the constant of step 3. Step 3's lift is
//! FIDESlib's unless the plan asks for the centred one (`Lift`).
//! Details: docs/ARCHITECTURE.md "pc-gpu mod raise".
use super::BtsCx;
use crate::context::{DevStrategy, DevWord};
use crate::ct::Ct;
use crate::error::{shape, Result};
use crate::ks::{self, DotIo, DotRows, KeyGroup, KsKeyRef, KsPlans, KskSource, View};
use crate::ntt::{self, Dir, Parts, Span};
use pc_core::encode::Complex;
use pc_core::modmul::{invmod, mulmod};
use pc_cuda::{launch_result, Arena, ConstBuf, Lane, Stream};
use pc_kernels::{ModRaiseArgs, SmallInt1Args, SmallIntArgs, SmallIntTab, SI_MAXK, SI_MAXOUT, SI_MAXR, SI_MAXW, SI_SHIFT};
use pc_shadow::fides::FMeta;
use pc_shadow::CtMeta;
use std::marker::PhantomData;

/// Seam `Raise`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Raise {
    /// Sparse-secret encapsulation around a composite raise with the exact post-raise constant.
    Encapsulated,
}

// ── x87 extended precision (FIDESlib computes the small-integer reciprocal in `long double`) ───────────────

/// A positive x87 extended value: `m * 2^e` with a 64-bit normalized mantissa (bit 63 set).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Ext {
    m: u64,
    e: i32,
}

/// x87 round-to-nearest-even of `2^num_exp / den` to a 64-bit mantissa.
fn div_rn64(num_exp: i32, den: u64) -> Ext {
    let den128 = den as u128;
    let lz = den.leading_zeros() as i32;
    // 2^k / den in [2^63, 2^64): k = 127 - lz (or one less when den is a power of two)
    let mut k = 127 - lz;
    let mut q = (1u128 << k) / den128;
    while q >= 1u128 << 64 {
        k -= 1;
        q = (1u128 << k) / den128;
    }
    let r = (1u128 << k) - q * den128;
    let mut m = q as u64;
    let mut e = num_exp - k;
    let twice = 2 * r;
    if twice > den128 || (twice == den128 && m & 1 == 1) {
        match m.checked_add(1) {
            Some(v) => m = v,
            None => {
                m = 1u64 << 63;
                e += 1;
            }
        }
    }
    Ext { m, e }
}

/// FIDESlib `buildSmallIntTab`'s reciprocal for `D = 1.0L / (long double)c`: R = round(2^192 / D) as the 64-bit x87
/// mantissa of the quotient, shifted into 32-bit words.
fn reciprocal_words(c: f64) -> Result<[u32; SI_MAXR]> {
    if !(c > 0.0) || !c.is_finite() {
        return Err(shape("small_int", format!("scaling constant {c} must be positive")));
    }
    let bits = c.to_bits();
    let exp = ((bits >> 52) & 0x7FF) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    let (mc, ec) = if exp == 0 { (frac, -1074) } else { (frac | (1u64 << 52), exp - 1075) };
    // D = RN64(1 / c) = RN64(2^-ec / mc)
    let d = div_rn64(-ec, mc);
    // V = RN64(2^192 / D) = RN64(2^(192 - e_D) / m_D); R = m_V << e_V
    let v = div_rn64(SI_SHIFT as i32 - d.e, d.m);
    let sh = v.e;
    if sh < 0 || sh as usize + 64 > 32 * SI_MAXR {
        return Err(shape("small_int", format!("reciprocal shift {sh} outside the table (constant {c})")));
    }
    let r = (v.m as u128) << (sh % 32);
    let w0 = (sh / 32) as usize;
    let mut out = [0u32; SI_MAXR];
    for i in 0..4 {
        if w0 + i < SI_MAXR {
            out[w0 + i] = (r >> (32 * i)) as u32;
        }
    }
    Ok(out)
}

/// The small-integer table (FIDESlib `buildSmallIntTab`): source primes `src`, outputs at prime ids `out_rows` with
/// values `out_primes`; `c` = Some: scale by c (D = 1/c), None: lift only.
pub fn small_int_tab(src: &[u64], out_rows: &[usize], out_primes: &[u64], c: Option<f64>) -> Result<SmallIntTab> {
    // SAFETY: plain integers; all-zero is a valid table
    let mut t: SmallIntTab = unsafe { std::mem::zeroed() };
    if src.is_empty() || src.len() > SI_MAXK || out_rows.len() > SI_MAXOUT || out_rows.len() != out_primes.len() {
        return Err(shape("small_int", format!("{} sources (<= {SI_MAXK}), {} outputs (<= {SI_MAXOUT})", src.len(), out_rows.len())));
    }
    t.k = src.len() as i32;
    t.nout = out_rows.len() as i32;
    for (i, &qi) in src.iter().enumerate() {
        if qi >= 1 << 32 {
            return Err(shape("small_int", "source primes must fit 32 bits"));
        }
        t.q[i] = qi as u32;
        for (j, &qj) in src.iter().enumerate().take(i) {
            t.qinv[i][j] = invmod(qj % qi, qi) as u32;
        }
    }
    let mut q: Vec<u32> = vec![1];
    for &qi in src {
        let mut carry = 0u64;
        for w in q.iter_mut() {
            let v = *w as u64 * qi + carry;
            *w = v as u32;
            carry = v >> 32;
        }
        if carry != 0 {
            q.push(carry as u32);
        }
    }
    t.nw = q.len() as i32 + 1;
    if t.nw as usize > SI_MAXW {
        return Err(shape("small_int", "source product too wide"));
    }
    t.qk[..q.len()].copy_from_slice(&q);
    let mut rem = 0u32;
    for i in (0..q.len()).rev() {
        let cur = ((rem as u64) << 32) | q[i] as u64;
        t.qkh[i] = (cur >> 1) as u32;
        rem = (cur & 1) as u32;
    }
    if let Some(c) = c {
        t.divide = 1;
        t.r = reciprocal_words(c)?;
        t.nr = SI_MAXR as i32;
        while t.nr > 1 && t.r[t.nr as usize - 1] == 0 {
            t.nr -= 1;
        }
    }
    for (o, (&row, &p)) in out_rows.iter().zip(out_primes).enumerate() {
        t.orow[o] = row as i32;
        let mut pw = 1 % p;
        let b32 = (1u64 << 32) % p;
        for w in 0..SI_MAXW + SI_MAXR {
            t.pow32[o][w] = pw as u32;
            pw = mulmod(pw, b32, p);
        }
    }
    Ok(t)
}

/// Host twin of the `small_int` kernel for one coefficient (tests; sources of at most 3 primes): the residues of
/// round(|x| R / 2^192) (half up) with x's sign, x reconstructed and centred from `x_rows`.
pub fn small_int_host(t: &SmallIntTab, x_rows: &[u64], out_primes: &[u64]) -> Vec<u64> {
    let k = t.k as usize;
    let mut x: i128 = 0;
    let mut m: i128 = 1;
    for i in 0..k {
        let qi = t.q[i] as i128;
        let minv = invmod(m.rem_euclid(qi) as u64, qi as u64) as i128;
        let d = ((x_rows[i] as i128 - x).rem_euclid(qi) * minv).rem_euclid(qi);
        x += m * d;
        m *= qi;
    }
    let neg = x > m / 2;
    let mag = if neg { m - x } else { x } as u128;
    let qv: u128 = if t.divide == 1 {
        let mut prod = [0u64; 2 * SI_MAXR + 8];
        for i in 0..4 {
            let a = ((mag >> (32 * i)) & 0xFFFF_FFFF) as u64;
            let mut carry = 0u64;
            for j in 0..SI_MAXR {
                let v = a * t.r[j] as u64 + prod[i + j] + carry;
                prod[i + j] = v & 0xFFFF_FFFF;
                carry = v >> 32;
            }
            let mut p = i + SI_MAXR;
            while carry != 0 {
                let v = prod[p] + carry;
                prod[p] = v & 0xFFFF_FFFF;
                carry = v >> 32;
                p += 1;
            }
        }
        let mut carry = 1u64 << 31;
        let mut i = 5;
        while carry != 0 {
            let v = prod[i] + carry;
            prod[i] = v & 0xFFFF_FFFF;
            carry = v >> 32;
            i += 1;
        }
        (0..4).fold(0u128, |a, w| a | ((prod[6 + w] as u128) << (32 * w)))
    } else {
        mag
    };
    out_primes
        .iter()
        .map(|&p| {
            let r = (qv % p as u128) as u64;
            if neg && r != 0 {
                p - r
            } else {
                r
            }
        })
        .collect()
}

/// How the raise lifts a coefficient from the composite bottom Qd. The two differ by a multiple of Qd, which is what the
/// overflow integer I = (c0' + c1' s - m) / Qd counts, so the choice sets the distribution EvalMod's series must cover.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lift {
    /// FIDESlib: `mod_raise` fills the third limb with each Garner term centred on its own prime, `small_int` reconstructs
    /// from three limbs. Every coefficient is a sum of two uniforms: I has sigma 2.35 at h = 32 (66 uniforms).
    PerPrime,
    /// `small_int` reconstructs from the d bottom limbs and centres in Qd: I has sigma 1.66 at h = 32 (33 uniforms), one
    /// kernel less. Not FIDESlib-bit-exact.
    Centred,
}

impl Lift {
    /// Rows `[r0, r1)` the composite `mod_raise` fills before `small_int` reads its sources. Only FIDESlib's lift at d = 2
    /// runs it: d = 1 has its own kernel and at d = 3 the three sources are the bottom.
    pub fn mod_raise_rows(self, d: usize) -> Option<(usize, usize)> {
        match (self, d) {
            (Lift::PerPrime, 2) => Some((2, 3)),
            _ => None,
        }
    }

    /// Source limbs of `small_int` at composite degree d (2 or 3).
    fn sources(self, d: usize) -> usize {
        match self {
            Lift::PerPrime => 3,
            Lift::Centred => d,
        }
    }
}

/// The `mod_raise` constants of a raise to `top` (composite degree d); launch fields left for `launch_mod_raise`.
pub fn raise_args(q: &[u64], d: usize, top: usize) -> ModRaiseArgs {
    // SAFETY: plain data, zero is valid
    let mut ra: ModRaiseArgs = unsafe { std::mem::zeroed() };
    for k in 0..d {
        let qk = q[k];
        let hat_k = (0..d).filter(|&j| j != k).fold(1u64, |h, j| mulmod(h, q[j] % qk, qk));
        ra.qhatinv[k] = invmod(hat_k, qk);
        for (i, &qi) in q.iter().enumerate().take(top + 1) {
            ra.qhat[k][i] = (0..d).filter(|&j| j != k).fold(1u64, |v, j| mulmod(v, q[j] % qi, qi));
        }
    }
    ra.d = d as i32;
    ra
}

/// The `small_int` table of a raise to `top` (composite degree d >= 2): every row 0..=top from the lift's source limbs.
pub fn raise_tab(q: &[u64], d: usize, top: usize, c: Option<f64>, lift: Lift) -> Result<SmallIntTab> {
    let rows: Vec<usize> = (0..=top).collect();
    small_int_tab(&q[..lift.sources(d)], &rows, &q[..=top], c)
}

// ── device plan ──────────────────────────────────────────────────────────────────────────────────────────────────

/// Words of a `SmallIntTab` on the device (it is all 4-byte fields).
const TAB_WORDS: usize = std::mem::size_of::<SmallIntTab>() / 4;
const _: () = assert!(std::mem::size_of::<SmallIntTab>() % 4 == 0);

fn tab_words(t: &SmallIntTab) -> Vec<u32> {
    // SAFETY: SmallIntTab is repr(C) of i32/u32 fields and arrays only (size asserted a multiple of 4)
    unsafe { std::slice::from_raw_parts(t as *const SmallIntTab as *const u32, TAB_WORDS) }.to_vec()
}

/// Raise tables of one precomputation variant.
pub struct RaisePlan<'a, W: DevWord> {
    /// Raise target top limb (`L - d * raise_drop`).
    pub top: usize,
    /// Composite degree d (bottom limbs).
    pub d: usize,
    /// The exact post-raise constant (`cts0_const`) and its shift t; the table scales by `exact * 2^t`.
    pub exact: f64,
    pub t: u32,
    /// The scale of the raised ciphertext before the shift: sf(top).
    pub target_sf: f64,
    /// How the bottom is lifted.
    pub lift: Lift,
    raise: ModRaiseArgs,
    /// `SmallIntTab` as words (device).
    tab: ConstBuf<'a, u32>,
    pub kind: Raise,
    /// composite degree 1: c = mc 2^-sh (the `small_int1` kernel; `tab` unused)
    single: Option<(u64, i32)>,
    _w: PhantomData<W>,
}

/// c = mc 2^-sh for a positive finite double (sh >= 1), refused when round(c |x|) can leave 64 bits for |x| <= q0/2.
pub fn single_scale(c: f64, q0: u64) -> Result<(u64, i32)> {
    if !(c > 0.0) || !c.is_finite() {
        return Err(shape("small_int1", format!("scaling constant {c} must be positive")));
    }
    let bits = c.to_bits();
    let exp = ((bits >> 52) & 0x7FF) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    let (mut mc, mut e) = if exp == 0 { (frac, -1074) } else { (frac | (1u64 << 52), exp - 1075) };
    // strip trailing zeros so sh is as small as possible, then pad to sh >= 1
    while mc & 1 == 0 && e < -1 {
        mc >>= 1;
        e += 1;
    }
    while e > -1 {
        if mc.leading_zeros() == 0 {
            return Err(shape("small_int1", format!("scaling constant {c} too large")));
        }
        mc <<= 1;
        e -= 1;
    }
    let sh = -e;
    if sh > 127 {
        return Err(shape("small_int1", format!("scaling constant {c} needs a shift of {sh} (> 127)")));
    }
    let max = ((q0 / 2) as u128 * mc as u128 + (1u128 << (sh - 1))) >> sh;
    if max >> 64 != 0 {
        return Err(shape("small_int1", format!("round(c |x|) leaves 64 bits for |x| <= q0/2 (c = {c})")));
    }
    Ok((mc, sh))
}

/// Host twin of `small_int1` for one coefficient: row 0 value `v` (mod q0) -> residues of sign(x) round(|x| c).
pub fn small_int1_host(v: u64, q0: u64, mc: u64, sh: i32, out_primes: &[u64]) -> Vec<u64> {
    let neg = v > q0 / 2;
    let mag = if neg { q0 - v } else { v };
    let q = ((mag as u128 * mc as u128 + (1u128 << (sh - 1))) >> sh) as u64;
    out_primes.iter().map(|&p| {
        let r = q % p;
        if neg && r != 0 {
            p - r
        } else {
            r
        }
    }).collect()
}

impl<'a, W: DevWord> RaisePlan<'a, W> {
    /// Device bytes of one variant's tables.
    pub fn table_bytes() -> usize {
        Arena::bytes_for::<u32>(TAB_WORDS)
    }

    /// Tables for a raise to `top` with the exact constant `exact * 2^t`.
    pub fn new<S: Lane, M: DevStrategy<W>>(arena: &'a Arena, s: &Stream<S>, ctx: &crate::Context<'_, W, M>, top: usize, exact: f64, t: u32, lift: Lift) -> Result<Self> {
        let chain = &ctx.chain;
        let d = chain.level_unit;
        let q = &chain.q;
        if !(1..=3).contains(&d) || top >= q.len() || top < 3 || top >= 64 {
            return Err(shape("raise", format!("composite degree {d} / target top {top} (the encapsulated raise needs d in 1..=3)")));
        }
        let ra = raise_args(q, d, top);
        let c = exact * (t as f64).exp2();
        let (tab, single) = if d == 1 {
            // SAFETY: plain integers; the table is unused at d = 1 (carved anyway: the variant's bytes do not depend on d)
            (unsafe { std::mem::zeroed::<SmallIntTab>() }, Some(single_scale(c, q[0])?))
        } else {
            (raise_tab(q, d, top, Some(c), lift)?, None)
        };
        Ok(RaisePlan {
            top,
            d,
            exact,
            t,
            target_sf: ctx.scale.sf(top as i32)?,
            lift,
            raise: ra,
            tab: ConstBuf::new(arena, "bts.raise", s, &tab_words(&tab))?,
            kind: Raise::Encapsulated,
            single,
            _w: PhantomData,
        })
    }

    pub fn bytes(&self) -> usize {
        Self::table_bytes()
    }

    /// `mod_raise` kernel on rows `[r0, r1)` of the first k cts of `x` (both parts, coefficient form).
    pub fn launch_mod_raise<M: DevStrategy<W>>(&self, ctx: &crate::Context<'_, W, M>, s: &Stream<pc_cuda::Compute>, x: View, k: usize, r0: usize, r1: usize) -> Result<()> {
        let mut a = self.raise;
        a.x = x.base;
        a.pr = ctx.pr();
        a.ct_stride = x.ct_stride;
        a.part_stride = x.part_stride;
        a.k = k as i32;
        a.r0 = r0 as i32;
        a.r1 = r1 as i32;
        a.nparts = 2;
        if r1 > self.top + 1 {
            return Err(shape("mod_raise", format!("rows up to {r1} above the target top {}", self.top)));
        }
        // SAFETY: rows [0, r1) of k ciphertexts of a context-layout buffer (r1 <= top + 1 < |Q|)
        launch_result("mod_raise", unsafe { (ctx.table.mod_raise)(&a, s.raw()) })?;
        Ok(())
    }

    /// `small_int` kernel: rows 0..=top of the first k cts of `x` := round(c * value), sources the lift's first rows.
    pub fn launch_small_int<M: DevStrategy<W>>(&self, ctx: &crate::Context<'_, W, M>, s: &Stream<pc_cuda::Compute>, x: View, k: usize) -> Result<()> {
        if let Some((mc, sh)) = self.single {
            let a = SmallInt1Args { x: x.base, pr: ctx.pr(), ct_stride: x.ct_stride, part_stride: x.part_stride, mc, sh, nrows: (self.top + 1) as i32, k: k as i32, nparts: 2 };
            // SAFETY: rows 0..=top of k cts of a context-layout buffer; each thread reads its row-0 coefficient
            // before writing that coefficient of any row
            launch_result("small_int1", unsafe { (ctx.table.small_int1)(&a, s.raw()) })?;
            return Ok(());
        }
        let a = SmallIntArgs {
            src: x.base,
            out: x.base,
            tab: self.tab.dptr() as *const SmallIntTab,
            pr: ctx.pr(),
            src_ct_stride: x.ct_stride,
            src_part_stride: x.part_stride,
            ct_stride: x.ct_stride,
            part_stride: x.part_stride,
            k: k as i32,
            nparts: 2,
        };
        // SAFETY: each thread reads its coefficient of the source rows before writing that coefficient of the output
        // rows (in place is element-wise); rows 0..=top of k cts lie inside the buffer
        launch_result("small_int", unsafe { (ctx.table.small_int)(&a, s.raw()) })?;
        Ok(())
    }
}

// ── key switching without automorphism (the encapsulation keys) ─────────────────────────────────────────────────

/// `x := KeySwitch(x)` for the first k cts of `x` at `top` with key 0 of `group` (FIDESlib `Ciphertext::keySwitch`:
/// ModUp(c1), inner product with P c0 added, ModDown). Keys without a resident `a`, packed or banded keys, and
/// `KskSource::Regen` go through `ks::stage_key`.
pub fn key_switch_inplace<W: DevWord, M: DevStrategy<W>>(
    cx: &mut BtsCx<'_, '_, W, M>,
    group: &KeyGroup<'_>,
    source: KskSource,
    x: &mut crate::poly::CtBuf<'_, pc_cuda::Compute, W>,
    k: usize,
    top: usize,
) -> Result<()> {
    let ctx = cx.ctx;
    let ex = &mut *cx.ex;
    let plans: &KsPlans<'_> = cx.ks;
    let lay = ctx.lay;
    ks::modup(ctx, &ex.compute, plans, &mut ex.ks, &mut ex.bconv_work, x, 0, k, 1, top)?;
    let v = View::of(x, 0);
    let io = DotIo {
        c1: v.part::<W>(1).base,
        c0: v.base,
        ct_stride: v.ct_stride,
        out: View::of(&ex.ks.ext, 0).base,
        part_stride: lay.poly_words() as u64,
        out_ct_stride: lay.ct_words() as u64,
        out_rot_stride: 0,
        c0_mode: 1,
    };
    let staged = source == KskSource::Regen || group.needs_stage() || !group.has_a();
    if staged {
        let ndig = plans.ndig(top)?;
        ks::stage_key(ctx, &ex.compute, plans, &mut ex.ks, group, 0, ndig, true)?;
    }
    ks::dot(ctx, &ex.compute, plans, &ex.ks, group, 0, 1, k, top, DotRows::All, io, staged)?;
    ks::moddown_to(ctx, &ex.compute, plans, &mut ex.ks.ext, &mut ex.ks.conv, View::of(x, 0), ks::MdOut::Set, k, top)
}

/// The key tables of the two encapsulation keys (setup).
pub struct EncapsKeys<'a> {
    pub dense_to_sparse: KeyGroup<'a>,
    pub sparse_to_dense: KeyGroup<'a>,
}

impl<'a> EncapsKeys<'a> {
    pub fn bytes() -> usize {
        2 * KeyGroup::bytes(1, true)
    }
    pub fn new<S: Lane>(arena: &'a Arena, s: &Stream<S>, n: usize, atob: &dyn KsKeyRef, btoa: &dyn KsKeyRef, stage: Option<(*const std::ffi::c_void, *const std::ffi::c_void)>) -> Result<Self> {
        Ok(EncapsKeys { dense_to_sparse: KeyGroup::new(arena, s, n, &[atob], stage)?, sparse_to_dense: KeyGroup::new(arena, s, n, &[btoa], stage)? })
    }
}

/// What the raise leaves (FIDESlib coordinates).
pub fn raised_meta<W: DevWord>(plan: &RaisePlan<'_, W>) -> FMeta {
    FMeta { top: plan.top as i32, nl: 1, nf: plan.target_sf * (plan.t as f64).exp2() }
}

/// Steps of one raise call, for checks that compare intermediate values (pc-check).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RaiseStop {
    /// after the dense -> sparse switch (FIDESlib stash "MR-atob")
    Bottom,
    /// after the composite raise of every limb, NTT form, before the exact constant ("MR-raised"; check path:
    /// raises every limb instead of the third only)
    Raised,
    /// after the exact constant ("X-scaled")
    Scaled,
    /// the whole raise ("MR-btoa")
    Done,
}

/// The encapsulated raise of every batch of `cts` (steps 1-4 of the module doc). `correction`: the call's raise
/// correction for step 1's FLEXIBLEAUTO adjust (None = prescaled path). `keys`: None = no encapsulation.
pub fn raise<'a, W: DevWord, M: DevStrategy<W>>(
    cx: &mut BtsCx<'_, 'a, W, M>,
    plan: &RaisePlan<'_, W>,
    keys: Option<&EncapsKeys<'_>>,
    source: KskSource,
    cts: &mut [Ct<'a, W>],
    correction: Option<u32>,
) -> Result<()> {
    raise_to(cx, plan, keys, source, cts, correction, RaiseStop::Done)
}

/// `raise` up to `stop`: `Scaled` leaves the raised ciphertexts under the sparse secret (the caller's first hoisted
/// rotations switch them, `BtsSetup::merge_s2d`).
pub fn raise_to<'a, W: DevWord, M: DevStrategy<W>>(
    cx: &mut BtsCx<'_, 'a, W, M>,
    plan: &RaisePlan<'_, W>,
    keys: Option<&EncapsKeys<'_>>,
    source: KskSource,
    cts: &mut [Ct<'a, W>],
    correction: Option<u32>,
    stop: RaiseStop,
) -> Result<()> {
    for x in cts.iter_mut() {
        to_bottom(cx, plan, x, correction)?;
        if let Some(kk) = keys {
            let k = x.k();
            key_switch_inplace(cx, &kk.dense_to_sparse, source, &mut x.buf, k, plan.d - 1)?;
        }
        raise_from_bottom(cx, plan, keys, source, x, stop)?;
    }
    Ok(())
}

/// Step 1: the input at the composite bottom with the target's nominal scale.
pub fn to_bottom<'a, W: DevWord, M: DevStrategy<W>>(cx: &mut BtsCx<'_, 'a, W, M>, plan: &RaisePlan<'_, W>, x: &mut Ct<'a, W>, correction: Option<u32>) -> Result<()> {
    let ctx = cx.ctx;
    if x.meta.deg == 2 {
        cx.ops.rescale(ctx, cx.ex, cx.sh, std::slice::from_mut(x))?;
    }
    if let Some(corr) = correction {
        let top_in = x.top(ctx.shape)?;
        let adj = adjust_factor(ctx, plan.target_sf, x.meta.nf, top_in, corr)?;
        cx.ops.mult_scalar_inplace(ctx, cx.ex, cx.sh, std::slice::from_mut(x), &[Complex::new(adj, 0.0)])?;
        cx.ops.rescale(ctx, cx.ex, cx.sh, std::slice::from_mut(x))?;
    }
    let bottom_level = pc_shadow::meta::level_from_top_limb(ctx.shape, (plan.d - 1) as u32)?;
    if x.meta.level > bottom_level {
        return Err(shape("raise", format!("input at level {} is below the composite bottom", x.meta.level)));
    }
    x.meta.level = bottom_level;
    x.meta.nf = plan.target_sf;
    Ok(())
}

/// Steps 3-4 from the switched bottom (`x` at top d - 1, evaluation form), stopping at `stop`.
pub fn raise_from_bottom<'a, W: DevWord, M: DevStrategy<W>>(
    cx: &mut BtsCx<'_, 'a, W, M>,
    plan: &RaisePlan<'_, W>,
    keys: Option<&EncapsKeys<'_>>,
    source: KskSource,
    x: &mut Ct<'a, W>,
    stop: RaiseStop,
) -> Result<()> {
    let ctx = cx.ctx;
    let d = plan.d;
    let nq = ctx.shape.n_q;
    let k = x.k();
    if stop == RaiseStop::Bottom {
        return Ok(());
    }
    let s = &cx.ex.compute;
    ntt::transform_k(ctx, s, Dir::Inverse, &mut x.buf, Span::Q { top: d - 1 }, Parts::Both, k)?;
    let level = pc_shadow::meta::level_from_top_limb(ctx.shape, plan.top as u32)?;
    if stop == RaiseStop::Raised {
        plan.launch_mod_raise(ctx, s, View::of(&x.buf, 0), k, d, plan.top + 1)?;
        ntt::transform_k(ctx, s, Dir::Forward, &mut x.buf, Span::Q { top: plan.top }, Parts::Both, k)?;
        x.meta = CtMeta { level, deg: 1, nf: plan.target_sf, ..x.meta };
        return Ok(());
    }
    if let Some((r0, r1)) = plan.lift.mod_raise_rows(d) {
        plan.launch_mod_raise(ctx, s, View::of(&x.buf, 0), k, r0, r1)?;
    }
    plan.launch_small_int(ctx, s, View::of(&x.buf, 0), k)?;
    ntt::transform_k(ctx, s, Dir::Forward, &mut x.buf, Span::Q { top: plan.top }, Parts::Both, k)?;
    x.meta = CtMeta { level, deg: 1, nf: raised_meta(plan).nf, ..x.meta };
    if stop == RaiseStop::Scaled {
        return Ok(());
    }
    if let Some(kk) = keys {
        key_switch_inplace(cx, &kk.sparse_to_dense, source, &mut x.buf, k, plan.top)?;
    }
    Ok(())
}

/// ModRaise's FLEXIBLEAUTO adjust factor (FIDESlib `ModRaise`, non-prescaled path): the input (after its pending
/// rescale) at `top_in` with scale `nf_in` is brought to `target_sf * 2^-correction` by one scalar product + rescale.
pub fn adjust_factor<W: DevWord, M: DevStrategy<W>>(ctx: &crate::Context<'_, W, M>, target_sf: f64, nf_in: f64, top_in: usize, correction: u32) -> Result<f64> {
    let st = &ctx.scale;
    let mod_to_drop = st.mod_reduce_product(top_in as i32);
    let mut a = (target_sf / nf_in) * (mod_to_drop / st.sf(top_in as i32)?);
    a *= (-(correction as f64)).exp2();
    Ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reciprocal_is_close() {
        for &c in &[2.3167161361161844e-07 * 4096.0, 1.0, 0.5, 3.0, 1e-3, 0.7] {
            let r = reciprocal_words(c).unwrap();
            let v: f64 = r.iter().enumerate().map(|(i, &w)| w as f64 * 2f64.powi(32 * i as i32)).sum();
            let rel = (v / 2f64.powi(192) - c).abs() / c;
            assert!(rel < 1e-15, "c {c}: rel {rel}");
        }
    }

    /// gcc x86-64 `long double`: m = (uint64_t)ldexpl(frexpl(ldexpl(1, 192) / (1.0L / c), &e), 64), sh = e - 64.
    #[test]
    fn reciprocal_matches_x87() {
        let refs: [(f64, u64, i32); 8] = [
            (2.3167161361161844e-07 * 4096.0, 0xf8c16882ef368000, 118),
            (1.0, 0x8000000000000000, 129),
            (0.5, 0x8000000000000000, 128),
            (3.0, 0xc000000000000000, 130),
            (1e-3, 0x83126e978d4fe000, 119),
            (0.7, 0xb333333333333000, 128),
            (0.9489, 0xf2eb1c432ca57800, 128),
            (1.234567e-5, 0xcf203fb7ba39d000, 112),
        ];
        for (c, m, sh) in refs {
            let bits = c.to_bits();
            let (mc, ec) = ((bits & ((1u64 << 52) - 1)) | (1u64 << 52), ((bits >> 52) & 0x7FF) as i32 - 1075);
            let d = div_rn64(-ec, mc);
            let v = div_rn64(SI_SHIFT as i32 - d.e, d.m);
            assert_eq!((v.m, v.e), (m, sh), "c {c}");
        }
    }

    #[test]
    fn div_rn64_rounds_to_nearest_even() {
        // 1/3 = 0.0101.. : 64-bit mantissa 0xAAAA..AB (round up), exponent -65
        let e = div_rn64(0, 3);
        assert_eq!(e.m, 0xAAAA_AAAA_AAAA_AAAB);
        assert_eq!(e.e, -65);
        // exact powers
        let e = div_rn64(10, 1 << 20);
        assert_eq!((e.m, e.e), (1u64 << 63, -73));
    }

    #[test]
    fn single_scale_is_exact() {
        let q0 = 1152921504606584833u64;
        for &c in &[2.3167161361161844e-07 * 4096.0, 1.0, 0.25, 3.0e-5, 0.7] {
            let (mc, sh) = single_scale(c, q0).unwrap();
            assert_eq!(mc as f64 * (-(sh as f64)).exp2(), c);
            for &x in &[0i128, 1, -3, 1000, -1001, 123456789012345, -98765432109879, (q0 / 2) as i128, -((q0 / 2) as i128)] {
                let v = x.rem_euclid(q0 as i128) as u64;
                let got = small_int1_host(v, q0, mc, sh, &[q0, 97, 65537]);
                // exact: round half up of |x| c with c = mc / 2^sh
                let num = x.unsigned_abs() * mc as u128;
                let mag = ((num + (1u128 << (sh - 1))) >> sh) as i128;
                let want_i = if x < 0 { -mag } else { mag };
                let want: Vec<u64> = [q0, 97, 65537].iter().map(|&p| want_i.rem_euclid(p as i128) as u64).collect();
                assert_eq!(got, want, "c {c} x {x}");
            }
        }
        assert!(single_scale(1e30, q0).is_err());
    }

    #[test]
    fn small_int_host_scales() {
        let q = [268042241u64, 265420801, 156499969];
        let out = [268042241u64, 265420801, 156499969, 95027201];
        let t = small_int_tab(&q, &[0, 1, 2, 3], &out, Some(0.25)).unwrap();
        for &x in &[0i128, 1, -3, 1000, -1001, 123456789012345, -98765432109879] {
            let rows: Vec<u64> = q.iter().map(|&p| x.rem_euclid(p as i128) as u64).collect();
            let got = small_int_host(&t, &rows, &out);
            let mag = (x.abs() + 2) / 4; // round half up of |x| / 4
            let want_i = if x < 0 { -mag } else { mag };
            let want: Vec<u64> = out.iter().map(|&p| want_i.rem_euclid(p as i128) as u64).collect();
            assert_eq!(got, want, "x {x}");
        }
    }

    const Q: [u64; 4] = [268042241, 265420801, 156499969, 95027201];
    const QD: i128 = Q[0] as i128 * Q[1] as i128;

    fn tables(lift: Lift, c: Option<f64>) -> (ModRaiseArgs, SmallIntTab) {
        (raise_args(&Q, 2, 3), raise_tab(&Q, 2, 3, c, lift).unwrap())
    }

    /// What `raise_from_bottom` launches for `lift` on one coefficient of residues `bottom` (d = 2): the raised value on
    /// `Q`. The `mod_raise` kernel's host twin is inline.
    fn raised(lift: Lift, ra: &ModRaiseArgs, tab: &SmallIntTab, bottom: [u64; 2]) -> Vec<u64> {
        let mut rows = bottom.to_vec();
        if let Some((r0, r1)) = lift.mod_raise_rows(2) {
            for t in r0..r1 {
                let mut acc = 0u64;
                for k in 0..2 {
                    let y = mulmod(rows[k], ra.qhatinv[k], Q[k]);
                    let v = if y > Q[k] >> 1 { (y as i128 - Q[k] as i128).rem_euclid(Q[t] as i128) as u64 } else { y % Q[t] };
                    acc = (acc + mulmod(v, ra.qhat[k][t], Q[t])) % Q[t];
                }
                rows.push(acc);
            }
        }
        small_int_host(tab, &rows, &Q)
    }

    fn residues(x: i128) -> [u64; 2] {
        [x.rem_euclid(Q[0] as i128) as u64, x.rem_euclid(Q[1] as i128) as u64]
    }

    /// The centred lift raises a bottom value to its representative in (-Qd/2, Qd/2], on every output row.
    #[test]
    fn centred_lift_raises_to_the_centred_value() {
        let (ra, tab) = tables(Lift::Centred, None);
        let mut pr = pc_core::keys::Prng::from_seed([1, 2, 3, 4, 5, 6, 7, 8], 1);
        let mut xs = vec![0i128, 1, -1, QD / 2, -(QD / 2), QD / 2 - 1, QD / 4, -(QD / 4), 3 * QD / 8, -(3 * QD / 8)];
        xs.extend((0..4000).map(|_| (pr.uniform_f64() * QD as f64) as i128 - QD / 2));
        for x in xs {
            let want: Vec<u64> = Q.iter().map(|&p| x.rem_euclid(p as i128) as u64).collect();
            assert_eq!(raised(Lift::Centred, &ra, &tab, residues(x)), want, "x {x}");
        }
    }

    /// Both lifts keep the value modulo the bottom: they differ by a multiple of Qd only.
    #[test]
    fn both_lifts_keep_the_bottom() {
        let mut pr = pc_core::keys::Prng::from_seed([2, 2, 3, 4, 5, 6, 7, 8], 1);
        for lift in [Lift::PerPrime, Lift::Centred] {
            let (ra, tab) = tables(lift, None);
            for _ in 0..4000 {
                let x = (pr.uniform_f64() * QD as f64) as i128 - QD / 2;
                let b = residues(x);
                let r = raised(lift, &ra, &tab, b);
                assert_eq!([r[0], r[1]], b, "{lift:?} x {x}");
            }
        }
    }

    /// The overflow integer I = (c0' + c1' s - m) / Qd under the production sparse secret. FIDESlib's raise centres each
    /// Garner term on its own prime, so every coefficient is a sum of two uniforms: I has sigma 2.35 (66 uniforms). The
    /// centred lift has sigma 1.66 (33 uniforms), the distribution the even EvalMod series is fitted to.
    #[test]
    fn centred_lift_narrows_the_overflow_integer() {
        let n = 1usize << 16;
        let mut pr = pc_core::keys::Prng::from_seed([3, 2, 3, 4, 5, 6, 7, 8], 1);
        let s = pc_core::keys::sample_sparse_ternary(&mut pr, n, pc_core::keys::SecretKey::SPARSE_H).unwrap();
        let s: Vec<(usize, i64)> = s.iter().enumerate().filter(|(_, &v)| v != 0).map(|(i, &v)| (i, v as i64)).collect();
        let rows: Vec<[Vec<u64>; 2]> = (0..2).map(|_| [(0..n).map(|_| pr.uniform_below(Q[0])).collect(), (0..n).map(|_| pr.uniform_below(Q[1])).collect()]).collect();
        let c1 = &rows[0];
        let c0: [Vec<u64>; 2] = [0, 1].map(|r| {
            let p = Q[r] as i64;
            (0..n)
                .map(|j| {
                    let x: i64 = s.iter().map(|&(i, v)| if j >= i { v * c1[r][j - i] as i64 } else { -v * c1[r][n + j - i] as i64 }).sum();
                    (-x).rem_euclid(p) as u64
                })
                .collect()
        });
        let t = Q[3];
        let qdinv = invmod((QD % t as i128) as u64, t);
        let spread = |lift: Lift| -> (f64, i64) {
            let (ra, tab) = tables(lift, None);
            let lift_row = |c: &[Vec<u64>; 2]| -> Vec<i64> { (0..n).map(|j| raised(lift, &ra, &tab, [c[0][j], c[1][j]])[3] as i64).collect() };
            let (l0, l1) = (lift_row(&c0), lift_row(c1));
            let (mut var, mut max) = (0f64, 0i64);
            for j in 0..n {
                let x: i64 = l0[j] + s.iter().map(|&(i, v)| if j >= i { v * l1[j - i] } else { -v * l1[n + j - i] }).sum::<i64>();
                let i = mulmod(x.rem_euclid(t as i64) as u64, qdinv, t) as i64;
                let i = if i > t as i64 / 2 { i - t as i64 } else { i };
                var += (i * i) as f64;
                max = max.max(i.abs());
            }
            ((var / n as f64).sqrt(), max)
        };
        let (fides, max_fides) = spread(Lift::PerPrime);
        let (centred, max_centred) = spread(Lift::Centred);
        assert!((fides - 2.345).abs() < 0.05, "per-prime sigma {fides}");
        assert!((centred - 1.659).abs() < 0.03, "centred sigma {centred}");
        assert!(max_centred <= 10 && max_centred < max_fides, "max |I|: centred {max_centred}, per-prime {max_fides}");
    }

    /// The record's raise is FIDESlib's: `mod_raise` on the third limb, `small_int` from three sources.
    #[test]
    fn per_prime_lift_is_the_shipped_raise() {
        let tab = raise_tab(&Q, 2, 3, Some(0.25), Lift::PerPrime).unwrap();
        assert_eq!(tab_words(&tab), tab_words(&small_int_tab(&Q[..3], &[0, 1, 2, 3], &Q, Some(0.25)).unwrap()));
        assert_eq!(Lift::PerPrime.mod_raise_rows(2), Some((2, 3)));
    }

    /// The composite raise only ever fills the third source at d = 2; the centred lift reads the bottom directly.
    #[test]
    fn lift_rows_by_composite_degree() {
        assert_eq!(Lift::Centred.mod_raise_rows(2), None);
        for lift in [Lift::PerPrime, Lift::Centred] {
            assert_eq!(lift.mod_raise_rows(1), None);
            assert_eq!(lift.mod_raise_rows(3), None);
        }
        assert_eq!(raise_tab(&Q, 2, 3, None, Lift::Centred).unwrap().k, 2);
        assert_eq!(raise_tab(&Q, 2, 3, None, Lift::PerPrime).unwrap().k, 3);
    }
}
