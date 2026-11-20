//! EvalMod as a static program.
//!
//! A metadata replay of FIDESlib's EvalMod control flow (btsEvalMod's complex branch, approxModReduction,
//! evalChebyshevSeries with lazy Paterson-Stockmeyer, the double-angle iterations) that records the device
//! operations with their integer constants. The level/degree/scale logic is the `fides` engine the `Shadow` rules
//! use (FLEXIBLEAUTO adjust of Ciphertext.cpp); the constants come from `pc_core::scale`. The program is executed
//! for a batch by the runtime: every op once for all 2K ciphertexts (K finishes x the Re/Im chains).
//!
//! The Chebyshev long divisions (`PsPlan`) are OpenFHE's own, taken from the reference dump's `params.json`
//! (`cheb_ps`) in the oracle tests; the runtime computes them with `pc_core::cheb` (`PsPlan::from_coeffs`).
use crate::error::{rule, Result};
use crate::fides::{FMeta, Fx, Step};
use pc_core::scale::ScaleTables;
use serde::{Deserialize, Serialize};

/// Relinearization epilogue folded from the following ops: x2 (Add(d, d, d)) and + constant (AddConst(d)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Epi {
    pub dbl: bool,
    pub k: Option<usize>,
}

/// One device op over buffers (`dst`, `a`, ...) of `n` limbs; `k` indexes `EmProgram::consts`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmOp {
    Copy { dst: usize, src: usize, n: usize },
    Add { dst: usize, a: usize, b: usize, n: usize, sub: bool },
    MulConst { x: usize, n: usize, k: usize },
    AddConst { x: usize, n: usize, k: usize },
    /// rescale; `k`: a MulConst folded in front (its constants scale the input first)
    Rescale { x: usize, n: usize, k: Option<usize> },
    Mult { dst: usize, a: usize, b: usize, n: usize, epi: Epi },
    MultAcc { dst: usize, pairs: Vec<(usize, usize)>, n: usize, epi: Epi },
    WSum { dst: usize, srcs: Vec<usize>, n: usize, k: usize },
}

/// The recorded program: ops, constant rows (residues by prime id, padded to |Q|; a WSum row is one padded row per
/// source, concatenated), the final metadata of every buffer, and every `ElemForEvalMult(level, level_in, operand)`
/// request (top limb indices) for the memo gate.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EmProgram {
    pub ops: Vec<EmOp>,
    pub consts: Vec<Vec<u64>>,
    pub reqs: Vec<(i32, i32, f64)>,
}

/// The PS coefficient plan (OpenFHE's long divisions of the Chebyshev series).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PsPlan {
    pub k: usize,
    pub m: usize,
    pub divqr_q: Vec<f64>,
    pub divcs_q: Vec<f64>,
    pub s2: Vec<f64>,
    /// The series itself (needed by the general m > 2 / recursive shape; empty for dumped single-level plans).
    #[serde(default)]
    pub coeffs: Vec<f64>,
    /// Functional bootstrapping: the amplitude A of EvalMod's sine (the series and double angles rescaled, so
    /// EvalMod returns A sin(2 pi x)); 0 = the shipped 1 / (2 pi).
    #[serde(default)]
    pub amp: f64,
    /// Functional bootstrapping: a Chebyshev series on [-1, 1] applied to the EvalMod output (empty: none).
    #[serde(default)]
    pub post: Vec<f64>,
}

impl PsPlan {
    /// The plan for a Chebyshev coefficient table, computed by OpenFHE's long divisions (`pc_core::cheb`).
    pub fn from_coeffs(coefficients: &[f64]) -> pc_core::error::Result<Self> {
        let p = pc_core::cheb::PsPlan::new(coefficients)?;
        Ok(PsPlan { k: p.k, m: p.m, divqr_q: p.divqr.q.clone(), divcs_q: p.divcs.q.clone(), s2: p.s2.clone(), coeffs: coefficients.to_vec(), amp: 0.0, post: vec![] })
    }

    /// Functional bootstrapping (Alexandru-Kim-Polyakov, ePrint 2024/1623; here a real Chebyshev series on the
    /// sine instead of their power series on the complex exponential): `base` (a shipped EvalMod table, amplitude
    /// (2 pi)^(-2^-da_its)) rescaled to amplitude `amp`, the double angles at that amplitude, then `post` on
    /// A sin(2 pi x). A table on |j| <= t/2 at phase j q0 / 2t and A = 1 reads s = sin(pi j / t): the nodes of a
    /// degree-t Chebyshev interpolation; a smooth f on |x| <= X at phase x q0 / G with A = 1 / sin(2 pi X / G) reads
    /// t in [-1, 1] (f(G asin(t / A) / 2 pi), interpolated on [-1, 1]).
    pub fn fbs(base: &[f64], da_its: u32, amp: f64, post: &[f64]) -> pc_core::error::Result<Self> {
        let a = (amp * 2.0 * std::f64::consts::PI).powf(1.0 / (1u64 << da_its) as f64);
        let c: Vec<f64> = base.iter().map(|x| x * a).collect();
        let mut p = Self::from_coeffs(&c)?;
        p.amp = amp;
        p.post = post.to_vec();
        Ok(p)
    }

    /// The EvalMod plan of the shipped bootstrap (degree 14, K=16 table).
    pub fn evalmod() -> Self {
        Self::from_coeffs(pc_core::bts_setup::evalmod_coeffs()).expect("frozen EvalMod table has a PS plan")
    }
}

/// Sources per `WSum` op (the device kernel's bound).
pub const WSUM_MAX: usize = 8;

/// Metadata replay + op recording (one chain).
pub struct Builder<'s> {
    pub st: &'s ScaleTables,
    pub prog: EmProgram,
    pub metas: Vec<FMeta>,
    nq: usize,
    /// Fewer level-alignment rescales than FIDESlib (same output level and scale, other roundings): T_1 for the
    /// odd powers' subtraction from its first realized copy (a scalar and a drop, no rescale), and the series'
    /// giant power realized before it joins the sum (its rescale can ride its relinearization). Off: FIDESlib's
    /// program, request for request.
    pub lean: bool,
}

impl<'s> Builder<'s> {
    pub fn new(st: &'s ScaleTables) -> Self {
        Builder { st, prog: EmProgram::default(), metas: vec![], nq: st.q.len(), lean: false }
    }
    fn fx(&self) -> Fx<'s> {
        Fx { d: self.st.d as i32, st: Some(self.st) }
    }
    pub fn fresh(&mut self) -> usize {
        self.metas.push(FMeta { top: -1, nl: 1, nf: 1.0 });
        self.metas.len() - 1
    }
    pub fn input(&mut self, m: FMeta) -> usize {
        self.metas.push(m);
        self.metas.len() - 1
    }
    fn n(&self, v: usize) -> usize {
        (self.metas[v].top + 1) as usize
    }
    fn konst(&mut self, v: Vec<u64>) -> usize {
        let mut row = vec![0u64; self.nq];
        row[..v.len()].copy_from_slice(&v);
        self.prog.consts.push(row);
        self.prog.consts.len() - 1
    }

    /// Record the device ops of an adjustment: `steps` replayed from `start` (the metadata before them).
    fn emit_steps(&mut self, x: usize, start: FMeta, steps: &[Step]) -> Result<()> {
        let d = self.st.d as i32;
        let mut m = start;
        for s in steps {
            match *s {
                Step::Rescale => {
                    self.prog.ops.push(EmOp::Rescale { x, n: (m.top + 1) as usize, k: None });
                    m.top = if self.st.singles != 0 { self.fx().down(m.top) } else { m.top - d };
                }
                Step::DropTo(t) => m.top = t,
                Step::Scalar(c) => {
                    let e = self.st.elem_for_eval_mult(m.top, c, -1)?;
                    self.prog.reqs.push((m.top, m.top, c));
                    let k = self.konst(e);
                    self.prog.ops.push(EmOp::MulConst { x, n: (m.top + 1) as usize, k });
                }
            }
        }
        Ok(())
    }

    pub fn copy(&mut self, dst: usize, src: usize) {
        let n = self.n(src);
        self.prog.ops.push(EmOp::Copy { dst, src, n });
        self.metas[dst] = self.metas[src];
    }
    pub fn rescale(&mut self, x: usize) -> Result<()> {
        let start = self.metas[x];
        let mut steps = vec![];
        self.fx().rescale(&mut self.metas[x], &mut steps)?;
        self.emit_steps(x, start, &steps)
    }
    pub fn drop_to_level(&mut self, x: usize, top: i32) {
        if top < self.metas[x].top {
            self.metas[x].top = top;
        }
    }
    pub fn grow_to_level(&mut self, x: usize, top: i32) {
        if top > self.metas[x].top {
            self.metas[x].top = top;
        }
    }
    pub fn mult_scalar_noprecheck(&mut self, x: usize, c: f64) -> Result<()> {
        let start = self.metas[x];
        let mut steps = vec![];
        self.fx().scalar_noprecheck(&mut self.metas[x], c, &mut steps)?;
        self.emit_steps(x, start, &steps)
    }
    pub fn mult_scalar(&mut self, x: usize, c: f64) -> Result<()> {
        let start = self.metas[x];
        let mut steps = vec![];
        self.fx().scalar(&mut self.metas[x], c, &mut steps)?;
        self.emit_steps(x, start, &steps)
    }
    pub fn add_scalar(&mut self, x: usize, c: f64) -> Result<()> {
        if c == 0.0 {
            return Ok(());
        }
        let m = self.metas[x];
        let mut e = self.st.elem_for_eval_add(m.top, c.abs(), m.nl)?;
        if c < 0.0 {
            for (i, v) in e.iter_mut().enumerate() {
                *v = self.st.q[i] - *v; // may equal p (FIDESlib keeps it unreduced)
            }
        }
        let k = self.konst(e);
        let n = self.n(x);
        self.prog.ops.push(EmOp::AddConst { x, n, k });
        Ok(())
    }
    pub fn mult_int(&mut self, x: usize, k: u64) {
        let n = self.n(x);
        let e: Vec<u64> = (0..n).map(|i| k % self.st.q[i]).collect();
        let ki = self.konst(e);
        self.prog.ops.push(EmOp::MulConst { x, n, k: ki });
    }
    pub fn adjust_add(&mut self, a: usize, b: usize) -> Result<bool> {
        let (start, mb) = (self.metas[a], self.metas[b]);
        let mut steps = vec![];
        let ok = self.fx().adjust_add(&mut self.metas[a], &mb, &mut steps)?;
        self.emit_steps(a, start, &steps)?;
        Ok(ok)
    }
    pub fn adjust_mult(&mut self, a: usize, b: usize) -> Result<bool> {
        let (start, mb) = (self.metas[a], self.metas[b]);
        let mut steps = vec![];
        let ok = self.fx().adjust_mult(&mut self.metas[a], &mb, &mut steps)?;
        self.emit_steps(a, start, &steps)?;
        Ok(ok)
    }
    fn add_impl(&mut self, a: usize, b: usize, sub: bool) -> Result<()> {
        if !self.adjust_add(a, b)? {
            let b2 = self.fresh();
            self.copy(b2, b);
            if !self.adjust_add(b2, a)? {
                return Err(rule("evalmod add", "adjustForAddOrSub failed both ways"));
            }
            return self.add_impl(a, b2, sub);
        }
        let (ma, mb) = (self.metas[a], self.metas[b]);
        if ma.top != mb.top || ma.nl != mb.nl {
            return Err(rule("evalmod add", format!("adjust left {ma:?} vs {mb:?}")));
        }
        let n = self.n(a);
        self.prog.ops.push(EmOp::Add { dst: a, a, b, n, sub });
        Ok(())
    }
    pub fn add(&mut self, a: usize, b: usize) -> Result<()> {
        self.add_impl(a, b, false)
    }
    pub fn sub(&mut self, a: usize, b: usize) -> Result<()> {
        self.add_impl(a, b, true)
    }
    /// Ciphertext::mult(b, rescale=false) (moddown, not fused)
    pub fn mult(&mut self, a: usize, b: usize) -> Result<()> {
        if !self.adjust_mult(a, b)? {
            let b2 = self.fresh();
            self.copy(b2, b);
            if !self.adjust_mult(b2, a)? {
                return Err(rule("evalmod mult", "adjustForMult failed both ways"));
            }
            return self.mult(a, b2);
        }
        let (ma, mb) = (self.metas[a], self.metas[b]);
        if ma.nl != 1 || ma.top != mb.top {
            return Err(rule("evalmod mult", format!("adjust left {ma:?} vs {mb:?}")));
        }
        let n = self.n(a);
        self.prog.ops.push(EmOp::Mult { dst: a, a, b, n, epi: Epi::default() });
        let m = &mut self.metas[a];
        m.nl += mb.nl;
        m.nf *= mb.nf;
        Ok(())
    }
    /// Ciphertext::square(rescale=false)
    pub fn square(&mut self, a: usize) -> Result<()> {
        if self.metas[a].nl == 2 {
            self.rescale(a)?;
        }
        self.mult(a, a)
    }
    /// Ciphertext::mult(b, c, rescale=false)
    pub fn mult3(&mut self, dst: usize, b: usize, c: usize) -> Result<()> {
        if dst == b && dst == c {
            self.square(dst)
        } else if dst == b {
            self.mult(dst, c)
        } else if dst == c {
            self.mult(dst, b)
        } else if self.metas[b].top <= self.metas[c].top {
            self.copy(dst, b);
            self.mult(dst, c)
        } else {
            self.copy(dst, c);
            self.mult(dst, b)
        }
    }
    /// Ciphertext::square(src, rescale=false)
    pub fn square2(&mut self, dst: usize, src: usize) -> Result<()> {
        if dst != src {
            self.copy(dst, src);
        }
        self.square(dst)
    }
    /// Ciphertext::evalLinearWSumMutable; more than `WSUM_MAX` sources (the kernel bound; the long series of a
    /// functional bootstrap) as sums of chunks added up.
    pub fn wsum(&mut self, dst: usize, srcs: &[usize], w: &[f64]) -> Result<()> {
        if srcs.len() <= WSUM_MAX {
            return self.wsum_one(dst, srcs, w);
        }
        self.wsum_one(dst, &srcs[..WSUM_MAX], &w[..WSUM_MAX])?;
        let lv = self.metas[dst].top;
        for (cs, cw) in srcs[WSUM_MAX..].chunks(WSUM_MAX).zip(w[WSUM_MAX..].chunks(WSUM_MAX)) {
            let t = self.fresh();
            self.metas[t] = FMeta { top: lv, nl: 1, nf: 1.0 };
            self.wsum_one(t, cs, cw)?;
            self.add(dst, t)?;
        }
        Ok(())
    }
    fn wsum_one(&mut self, dst: usize, srcs: &[usize], w: &[f64]) -> Result<()> {
        if self.metas[dst].top == -1 {
            self.metas[dst].top = self.metas[srcs[0]].top;
            self.metas[dst].nl = 1;
        }
        let lv = self.metas[dst].top;
        let mut tab = Vec::new();
        for (i, &s) in srcs.iter().enumerate() {
            let ms = self.metas[s];
            if ms.nl != 1 || ms.top < lv {
                return Err(rule("evalmod wsum", format!("source {ms:?} below the sum level {lv}")));
            }
            let e = self.st.elem_for_eval_mult(lv, w[i], ms.top)?;
            self.prog.reqs.push((lv, ms.top, w[i]));
            let mut row = vec![0u64; self.nq];
            row[..e.len()].copy_from_slice(&e);
            tab.extend(row);
        }
        self.prog.consts.push(tab);
        let k = self.prog.consts.len() - 1;
        let n = self.n(dst);
        self.prog.ops.push(EmOp::WSum { dst, srcs: srcs.to_vec(), n, k });
        let sf = self.st.sf(lv)?;
        let m = &mut self.metas[dst];
        m.nl = 2;
        m.nf = sf * sf;
        Ok(())
    }
    /// Ciphertext::multAccumulateChecked: false = the eager fallback must run (nothing recorded then).
    pub fn mult_acc_checked(&mut self, acc: usize, a: &[usize], b: &[usize]) -> Result<bool> {
        if self.metas[acc].nl != 2 {
            return Ok(false);
        }
        let (ops0, metas0, consts0, reqs0) =
            (self.prog.ops.len(), self.metas.len(), self.prog.consts.len(), self.prog.reqs.len());
        let lv = self.metas[acc].top;
        let mut aa = a.to_vec();
        let mut bb = b.to_vec();
        for j in 0..aa.len() {
            let (pa, pb) = (aa[j], bb[j]);
            let (ma, mb) = (self.metas[pa], self.metas[pb]);
            if ma.nl == 1 && mb.nl == 1 && ma.top == lv && mb.top == lv {
                continue;
            }
            let ta = self.fresh();
            self.copy(ta, pa);
            if !self.adjust_mult(ta, pb)? {
                let tb = self.fresh();
                self.copy(tb, pb);
                self.adjust_mult(tb, ta)?;
                self.adjust_mult(ta, tb)?;
                bb[j] = tb;
            }
            aa[j] = ta;
        }
        let nfacc = self.metas[acc].nf;
        for j in 0..aa.len() {
            let (ma, mb) = (self.metas[aa[j]], self.metas[bb[j]]);
            let nf = ma.nf * mb.nf;
            if ma.nl != 1 || mb.nl != 1 || ma.top != lv || mb.top != lv || !(nf > nfacc * (1.0 - 1e-9) && nf < nfacc * (1.0 + 1e-9)) {
                self.prog.ops.truncate(ops0);
                self.metas.truncate(metas0);
                self.prog.consts.truncate(consts0);
                self.prog.reqs.truncate(reqs0);
                return Ok(false);
            }
        }
        let n = self.n(acc);
        self.prog.ops.push(EmOp::MultAcc { dst: acc, pairs: aa.into_iter().zip(bb).collect(), n, epi: Epi::default() });
        Ok(true)
    }
}

/// OpenFHE Degree (index of the last non-zero coefficient)
pub fn degree(c: &[f64]) -> usize {
    c.iter().rposition(|&x| x != 0.0).unwrap_or(0)
}

/// evalChebyshevSeries (lazy PS) on chain input `x` (modified in place, as FIDESlib's ctxt): the single-level
/// m = 2 shape from the plan's long divisions, any other shape (m > 2, recursive InnerEvalChebyshevPS) from the
/// plan's coefficients (`build_chebyshev_general`).
pub fn build_chebyshev(b: &mut Builder, x: usize, ps: &PsPlan) -> Result<()> {
    let single = ps.m == 2 && degree(&ps.divqr_q) <= ps.k && degree(&ps.s2) <= ps.k;
    if !single {
        if ps.coeffs.is_empty() {
            return Err(rule("evalmod", format!("PS shape k {} m {} needs the series coefficients (PsPlan::from_coeffs)", ps.k, ps.m)));
        }
        return build_chebyshev_general(b, x, &ps.coeffs);
    }
    build_chebyshev_m2(b, x, ps)
}

/// The m = 2 single-level shape (bit-exact vs d3/g0).
fn build_chebyshev_m2(b: &mut Builder, x: usize, ps: &PsPlan) -> Result<()> {
    let (k, m) = (ps.k, ps.m);
    let t: Vec<usize> = (0..k).map(|_| b.fresh()).collect();
    b.copy(t[0], x);
    if b.metas[t[0]].nl == 2 {
        b.rescale(t[0])?;
    }
    let x1 = if b.lean {
        let x1 = b.fresh();
        b.copy(x1, t[0]);
        x1
    } else {
        x
    };
    for i in 2..=k {
        if i % 2 == 1 {
            b.adjust_mult(t[i / 2], t[i / 2 - 1])?;
            b.adjust_mult(t[i / 2 - 1], t[i / 2])?;
            b.mult3(t[i - 1], t[i / 2 - 1], t[i / 2])?;
            b.add(t[i - 1], t[i - 1])?;
            // lean: T_1 for the subtraction from its first realized copy, brought to T_(i-1)'s level and scale by a
            // scalar and a drop (FIDESlib adjusts the input in place: a rescale from the second odd power on)
            let xs = if b.lean {
                let xs = b.fresh();
                b.copy(xs, x1);
                xs
            } else {
                x
            };
            b.adjust_add(xs, t[i - 1])?;
            if b.metas[xs].nl == 1 && b.metas[t[i - 1]].nl == 2 {
                b.rescale(t[i - 1])?;
            }
            b.sub(t[i - 1], xs)?;
        } else {
            b.adjust_mult(t[i / 2 - 1], t[i / 2 - 1])?;
            b.square2(t[i - 1], t[i / 2 - 1])?;
            b.add(t[i - 1], t[i - 1])?;
            b.add_scalar(t[i - 1], -1.0)?;
        }
    }
    for i in 1..=k {
        if b.metas[t[i - 1]].nl == 2 {
            b.rescale(t[i - 1])?;
        }
    }
    let t2: Vec<usize> = (0..m).map(|_| b.fresh()).collect();
    b.copy(t2[0], t[k - 1]);
    for i in 1..m {
        b.square2(t2[i], t2[i - 1])?;
        b.add(t2[i], t2[i])?;
        b.add_scalar(t2[i], -1.0)?;
    }
    let t2km1 = b.fresh();
    b.copy(t2km1, t2[0]);
    // lazy: the single product of the m = 2 loop joins the final sum
    // ---- innerEvalChebyshevPS(ctxt, ctxt, f2, k, m, T, T2, 0, m, &lz)
    let d_ = b.st.d as i32;
    let cu = x;
    let dc = degree(&ps.divcs_q);
    let mut flag_c = false;
    let lvl_t2 = |b: &Builder| b.metas[t2[m - 1]].top + if b.metas[t2[m - 1]].nl == 1 { d_ } else { 0 };
    if dc >= 1 {
        if dc == 1 {
            b.copy(cu, t[0]);
            if ps.divcs_q[1] != 1.0 {
                b.mult_scalar(cu, ps.divcs_q[1])?;
            }
        } else {
            let w: Vec<f64> = (0..dc).map(|i| ps.divcs_q[i + 1]).collect();
            let tl = lvl_t2(b);
            b.drop_to_level(cu, tl);
            b.grow_to_level(cu, tl);
            b.wsum(cu, &t[..dc], &w)?;
        }
        b.add_scalar(cu, ps.divcs_q[0] / 2.0)?;
        flag_c = true;
    }
    let qu = b.fresh();
    if degree(&ps.divqr_q) > k {
        return Err(rule("evalmod", "recursive PS not ported"));
    }
    {
        let mut qcopy = ps.divqr_q.clone();
        qcopy.resize(k, 0.0);
        if degree(&qcopy) > 0 {
            let mut w = vec![];
            let mut cts = vec![];
            for i in 0..ps.divqr_q.len() - 1 {
                if ps.divqr_q[i + 1] != 0.0 {
                    w.push(ps.divqr_q[i + 1]);
                    cts.push(t[i]);
                }
            }
            let tl = lvl_t2(b);
            b.grow_to_level(qu, tl);
            b.drop_to_level(qu, tl);
            b.wsum(qu, &cts, &w)?;
            b.add_scalar(qu, ps.divqr_q[0] / 2.0)?;
            if b.metas[t[k - 1]].nl == 1 {
                b.rescale(qu)?;
            }
            if b.metas[t[k - 1]].nl == 2 {
                b.rescale(qu)?;
            }
        } else {
            b.copy(qu, t[k - 1]);
            let q = *ps.divqr_q.last().unwrap_or(&0.0);
            if !(q > 0.0 && q.round() == q) {
                return Err(rule("evalmod", format!("non-integral leading quotient {q}")));
            }
            b.mult_int(qu, q as u64);
            b.add_scalar(qu, ps.divqr_q[0] / 2.0)?;
            if b.metas[qu].nl == 2 {
                b.rescale(qu)?;
            }
        }
    }
    let su = b.fresh();
    if degree(&ps.s2) > k {
        return Err(rule("evalmod", "s2 degree above k"));
    }
    {
        let mut scopy = ps.s2.clone();
        scopy.resize(k, 0.0);
        if degree(&scopy) > 0 {
            let mut w = vec![];
            let mut cts = vec![];
            for i in 0..ps.s2.len() - 1 {
                if ps.s2[i + 1] != 0.0 {
                    w.push(ps.s2[i + 1]);
                    cts.push(t[i]);
                }
            }
            let tl = lvl_t2(b) - d_;
            b.grow_to_level(su, tl);
            b.drop_to_level(su, tl);
            // the T's of the non-zero weights (FIDESlib passes the first |w| T's: the same program for a series
            // whose s2 has no zero below its top, as the shipped tables; a wrong one otherwise)
            b.wsum(su, &cts, &w)?;
            b.add_scalar(su, ps.s2[0] / 2.0)?;
        } else {
            b.copy(su, t[k - 1]);
            b.add_scalar(su, ps.s2[0] / 2.0)?;
        }
    }
    if !flag_c {
        return Err(rule("evalmod", "addScalar(T2, c) branch not ported"));
    }
    // max_m - m <= 1; lean: the giant power realized before it joins cu (its rescale can ride its relinearization;
    // FIDESlib rescales the sum, then a copy of the power for the lazy product)
    if b.lean && b.metas[t2[m - 1]].nl == 2 {
        b.rescale(t2[m - 1])?;
    }
    b.adjust_add(t2[m - 1], cu)?;
    if b.metas[t2[m - 1]].nl == 1 && b.metas[cu].nl == 2 {
        b.rescale(cu)?;
    }
    b.add(cu, t2[m - 1])?;
    // lazy hand-back -> evalChebyshevSeries' lazy tail
    let neg_x2 = b.fresh();
    b.copy(neg_x2, t2km1);
    b.sub(neg_x2, t2km1)?;
    b.sub(neg_x2, t2km1)?;
    b.sub(neg_x2, t2km1)?;
    let acc = b.fresh();
    b.copy(acc, su);
    if b.mult_acc_checked(acc, &[x, neg_x2], &[qu, t2[m - 1]])? {
        b.adjust_add(t2[0], acc)?;
        if b.metas[t2[0]].nl == 1 && b.metas[acc].nl == 2 {
            b.rescale(acc)?;
        }
        b.add(acc, t2[0])?;
        b.copy(x, acc);
    } else {
        b.mult(x, qu)?;
        b.add(x, su)?;
        b.mult(t2km1, t2[m - 1])?;
        b.add(t2km1, t2km1)?;
        b.adjust_add(t2[0], t2km1)?;
        if b.metas[t2[0]].nl == 1 && b.metas[t2km1].nl == 2 {
            b.rescale(t2km1)?;
        }
        b.sub(t2km1, t2[0])?;
        b.sub(x, t2km1)?;
    }
    Ok(())
}


/// T_1..T_k (FIDESlib evalChebyshevSeries' power loop) into fresh buffers, all realized; `x` untouched.
fn cheb_powers(b: &mut Builder, x: usize, k: usize) -> Result<Vec<usize>> {
    let t: Vec<usize> = (0..k).map(|_| b.fresh()).collect();
    b.copy(t[0], x);
    if b.metas[t[0]].nl == 2 {
        b.rescale(t[0])?;
    }
    let x1 = if b.lean {
        let x1 = b.fresh();
        b.copy(x1, t[0]);
        x1
    } else {
        x
    };
    for i in 2..=k {
        if i % 2 == 1 {
            b.adjust_mult(t[i / 2], t[i / 2 - 1])?;
            b.adjust_mult(t[i / 2 - 1], t[i / 2])?;
            b.mult3(t[i - 1], t[i / 2 - 1], t[i / 2])?;
            b.add(t[i - 1], t[i - 1])?;
            // lean: T_1 for the subtraction from its first realized copy, brought to T_(i-1)'s level and scale by a
            // scalar and a drop (FIDESlib adjusts the input in place: a rescale from the second odd power on)
            let xs = if b.lean {
                let xs = b.fresh();
                b.copy(xs, x1);
                xs
            } else {
                x
            };
            b.adjust_add(xs, t[i - 1])?;
            if b.metas[xs].nl == 1 && b.metas[t[i - 1]].nl == 2 {
                b.rescale(t[i - 1])?;
            }
            b.sub(t[i - 1], xs)?;
        } else {
            b.adjust_mult(t[i / 2 - 1], t[i / 2 - 1])?;
            b.square2(t[i - 1], t[i / 2 - 1])?;
            b.add(t[i - 1], t[i - 1])?;
            b.add_scalar(t[i - 1], -1.0)?;
        }
    }
    for i in 1..=k {
        if b.metas[t[i - 1]].nl == 2 {
            b.rescale(t[i - 1])?;
        }
    }
    Ok(t)
}

/// evalChebyshevSeries for any Paterson-Stockmeyer shape (FIDESlib ApproxModEval.cu: the T / T2 powers,
/// T_{k(2^m-1)} with its last product left to the lazy tail, the recursive InnerEvalChebyshevPS), FLEXIBLEAUTO.
pub fn build_chebyshev_general(b: &mut Builder, x: usize, coeffs: &[f64]) -> Result<()> {
    let n = degree(coeffs);
    let f2: Vec<f64> = coeffs[..=n].to_vec();
    let (k, m) = pc_core::cheb::compute_degrees_ps(n as u32)?;
    let (k, m) = (k as usize, m as usize);
    if m < 2 {
        return Err(rule("evalmod", format!("PS shape m = {m} (degree {n}) not ported")));
    }
    let t = cheb_powers(b, x, k)?;
    let t2: Vec<usize> = (0..m).map(|_| b.fresh()).collect();
    b.copy(t2[0], t[k - 1]);
    for i in 1..m {
        b.square2(t2[i], t2[i - 1])?;
        b.add(t2[i], t2[i])?;
        b.add_scalar(t2[i], -1.0)?;
    }
    let t2km1 = b.fresh();
    b.copy(t2km1, t2[0]);
    for i in 1..m {
        if i == m - 1 {
            break; // lazy: the last product joins the final sum
        }
        b.mult(t2km1, t2[i])?;
        b.add(t2km1, t2km1)?;
        b.adjust_add(t2[0], t2km1)?;
        if b.metas[t2[0]].nl == 1 && b.metas[t2km1].nl == 2 {
            b.rescale(t2km1)?;
        }
        b.sub(t2km1, t2[0])?;
        if b.metas[t2[0]].nl == 2 && i < m - 1 && b.metas[t2km1].nl == 2 {
            b.rescale(t2km1)?;
        }
    }
    let lz = inner_ps(b, x, x, &f2, k, m, &t, &t2, 0, m, true)?;
    let (qu, su) = lz.ok_or_else(|| rule("evalmod", "lazy PS hand-back missing"))?;
    let neg_x2 = b.fresh();
    b.copy(neg_x2, t2km1);
    b.sub(neg_x2, t2km1)?;
    b.sub(neg_x2, t2km1)?;
    b.sub(neg_x2, t2km1)?;
    let acc = b.fresh();
    b.copy(acc, su);
    if b.mult_acc_checked(acc, &[x, neg_x2], &[qu, t2[m - 1]])? {
        b.adjust_add(t2[0], acc)?;
        if b.metas[t2[0]].nl == 1 && b.metas[acc].nl == 2 {
            b.rescale(acc)?;
        }
        b.add(acc, t2[0])?;
        b.copy(x, acc);
    } else {
        b.mult(x, qu)?;
        b.add(x, su)?;
        b.mult(t2km1, t2[m - 1])?;
        b.add(t2km1, t2km1)?;
        b.adjust_add(t2[0], t2km1)?;
        if b.metas[t2[0]].nl == 1 && b.metas[t2km1].nl == 2 {
            b.rescale(t2km1)?;
        }
        b.sub(t2km1, t2[0])?;
        b.sub(x, t2km1)?;
    }
    Ok(())
}

/// InnerEvalChebyshevPS into `out` (`cu`); returns (qu, su) instead of finishing when `lazy` at the top call.
#[allow(clippy::too_many_arguments)]
fn inner_ps(
    b: &mut Builder,
    _x: usize,
    out: usize,
    coefficients: &[f64],
    k: usize,
    m: usize,
    t: &[usize],
    t2: &[usize],
    level_offset: i32,
    max_m: usize,
    lazy: bool,
) -> Result<Option<(usize, usize)>> {
    use pc_core::cheb::long_division;
    let k2m2k = k * (1 << (m - 1)) - k;
    let mut f2 = coefficients.to_vec();
    f2.resize(2 * k2m2k + k + 1, 0.0);
    if f2.len() > coefficients.len() {
        if let Some(l) = f2.last_mut() {
            *l = 1.0;
        }
    }
    let mut tkm = vec![0.0; k2m2k + k + 1];
    if let Some(l) = tkm.last_mut() {
        *l = 1.0;
    }
    let divqr = long_division(&f2, &tkm)?;
    let mut r2 = divqr.r.clone();
    if k2m2k as i64 - degree(&divqr.r) as i64 <= 0 {
        r2[k2m2k] -= 1.0;
        let d = degree(&r2);
        r2.truncate(d + 1);
    } else {
        r2.resize(k2m2k + 1, 0.0);
        if let Some(l) = r2.last_mut() {
            *l = -1.0;
        }
    }
    let divcs = long_division(&r2, &divqr.q)?;
    let mut s2 = divcs.r.clone();
    s2.resize(k2m2k + 1, 0.0);
    if let Some(l) = s2.last_mut() {
        *l = 1.0;
    }
    let d_ = b.st.d as i32;
    let cu = out;
    let lvl_t2 = |b: &Builder| b.metas[t2[m - 1]].top + if b.metas[t2[m - 1]].nl == 1 { d_ } else { 0 } - level_offset * d_;
    let dc = degree(&divcs.q);
    let mut flag_c = false;
    if dc >= 1 {
        if dc == 1 {
            b.copy(cu, t[0]);
            if divcs.q[1] != 1.0 {
                b.mult_scalar(cu, divcs.q[1])?;
            }
        } else {
            let w: Vec<f64> = (0..dc).map(|i| divcs.q[i + 1]).collect();
            let tl = lvl_t2(b);
            b.drop_to_level(cu, tl);
            b.grow_to_level(cu, tl);
            b.wsum(cu, &t[..dc], &w)?;
        }
        b.add_scalar(cu, divcs.q[0] / 2.0)?;
        flag_c = true;
    }
    let qu = b.fresh();
    if degree(&divqr.q) > k {
        if m <= 2 {
            return Err(rule("evalmod", "PS recursion below m = 2"));
        }
        inner_ps(b, _x, qu, &divqr.q, k, m - 1, t, t2, level_offset, max_m, false)?;
        if b.metas[qu].nl == 2 {
            b.rescale(qu)?;
        }
    } else {
        let mut qcopy = divqr.q.clone();
        qcopy.resize(k, 0.0);
        if degree(&qcopy) > 0 {
            let mut w = vec![];
            let mut cts = vec![];
            for i in 0..divqr.q.len() - 1 {
                if divqr.q[i + 1] != 0.0 {
                    w.push(divqr.q[i + 1]);
                    cts.push(t[i]);
                }
            }
            let tl = lvl_t2(b);
            b.grow_to_level(qu, tl);
            b.drop_to_level(qu, tl);
            b.wsum(qu, &cts, &w)?;
            b.add_scalar(qu, divqr.q[0] / 2.0)?;
            if b.metas[t[k - 1]].nl == 1 {
                b.rescale(qu)?;
            }
            if b.metas[t[k - 1]].nl == 2 {
                b.rescale(qu)?;
            }
        } else {
            b.copy(qu, t[k - 1]);
            let q = *divqr.q.last().unwrap_or(&0.0);
            if !(q > 0.0 && q.round() == q) {
                return Err(rule("evalmod", format!("non-integral leading quotient {q}")));
            }
            b.mult_int(qu, q as u64);
            b.add_scalar(qu, divqr.q[0] / 2.0)?;
            if b.metas[qu].nl == 2 {
                b.rescale(qu)?;
            }
        }
    }
    let su = b.fresh();
    if degree(&s2) > k {
        if m <= 2 {
            return Err(rule("evalmod", "PS recursion below m = 2"));
        }
        inner_ps(b, _x, su, &s2, k, m - 1, t, t2, level_offset + 1, max_m, false)?;
    } else {
        let mut scopy = s2.clone();
        scopy.resize(k, 0.0);
        if degree(&scopy) > 0 {
            let mut w = vec![];
            let mut cts = vec![];
            for i in 0..s2.len() - 1 {
                if s2[i + 1] != 0.0 {
                    w.push(s2[i + 1]);
                    cts.push(t[i]);
                }
            }
            let tl = lvl_t2(b) - d_;
            b.grow_to_level(su, tl);
            b.drop_to_level(su, tl);
            // the T's of the non-zero weights (FIDESlib passes the first |w| T's: the same program for a series
            // whose s2 has no zero below its top, as the shipped tables; a wrong one otherwise)
            b.wsum(su, &cts, &w)?;
            b.add_scalar(su, s2[0] / 2.0)?;
        } else {
            b.copy(su, t[k - 1]);
            b.add_scalar(su, s2[0] / 2.0)?;
        }
    }
    // lean, the lazy top call: the giant power realized before it joins cu (as `build_chebyshev_m2`)
    if b.lean && lazy && level_offset == 0 && m == max_m && b.metas[t2[m - 1]].nl == 2 {
        b.rescale(t2[m - 1])?;
    }
    if flag_c {
        if max_m - m <= 1 {
            b.adjust_add(t2[m - 1], cu)?;
        }
        if b.metas[t2[m - 1]].nl == 1 && b.metas[cu].nl == 2 {
            b.rescale(cu)?;
        }
        b.add(cu, t2[m - 1])?;
    } else {
        // the quotient c is a constant (InnerEvalChebyshevPS' EvalAdd(T2[m - 1], c_0 / 2))
        b.copy(cu, t2[m - 1]);
        b.add_scalar(cu, divcs.q[0] / 2.0)?;
    }
    if lazy && level_offset == 0 && m == max_m {
        return Ok(Some((qu, su)));
    }
    b.mult(cu, qu)?;
    b.add(cu, su)?;
    Ok(None)
}

/// applyDoubleAngleIterations
pub fn build_double_angle(b: &mut Builder, x: usize, its: i32) -> Result<()> {
    build_double_angle_amp(b, x, its, 0.0)
}

/// applyDoubleAngleIterations at output amplitude `amp` (0: the shipped 1 / (2 pi)).
pub fn build_double_angle_amp(b: &mut Builder, x: usize, its: i32, amp: f64) -> Result<()> {
    let r = its;
    for j in 1..=r {
        b.square(x)?;
        let scalar = if amp > 0.0 { -amp.powf(2f64.powf((j - r) as f64)) } else { -1.0 / (2.0 * std::f64::consts::PI).powf(2f64.powf((j - r) as f64)) };
        b.add(x, x)?;
        b.add_scalar(x, scalar)?;
    }
    Ok(())
}

/// The whole EvalMod chain of one input: the Chebyshev series, then `double_angle_its` double-angle steps.
pub fn build_evalmod(st: &ScaleTables, input: FMeta, ps: &PsPlan, double_angle_its: i32) -> Result<(EmProgram, FMeta)> {
    build_evalmod_opts(st, input, ps, double_angle_its, false)
}

/// `build_evalmod` with `Builder::lean`.
pub fn build_evalmod_opts(st: &ScaleTables, input: FMeta, ps: &PsPlan, double_angle_its: i32, lean: bool) -> Result<(EmProgram, FMeta)> {
    let mut b = Builder::new(st);
    b.lean = lean;
    let x = b.input(input);
    build_chebyshev(&mut b, x, ps)?;
    build_double_angle_amp(&mut b, x, double_angle_its, ps.amp)?;
    if !ps.post.is_empty() {
        build_chebyshev_general(&mut b, x, &ps.post)?;
    }
    let out = b.metas[x];
    Ok((b.prog, out))
}

impl EmProgram {
    /// Peephole fusion (exact: modular arithmetic is reordered only where it is associative):
    /// Copy(d <- s) + Mult(d, d, b) -> Mult(d, s, b); Mult(d) + Add(d, d, d) + AddConst(d) -> one relin epilogue;
    /// MulConst(x) + Rescale(x) -> Rescale(x, k).
    pub fn fused(&self) -> Vec<EmOp> {
        let ops = &self.ops;
        let mut out: Vec<EmOp> = Vec::with_capacity(ops.len());
        let mut i = 0;
        while i < ops.len() {
            let mut op = ops[i].clone();
            if let (EmOp::Copy { dst, src, n }, Some(EmOp::Mult { dst: md, a, b, n: mn, epi })) = (&ops[i], ops.get(i + 1)) {
                if md == dst && a == dst && n == mn {
                    let b2 = if b == dst { *src } else { *b };
                    op = EmOp::Mult { dst: *dst, a: *src, b: b2, n: *n, epi: *epi };
                    i += 1;
                }
            }
            i += 1;
            let (d0, is_mult) = match &op {
                EmOp::Mult { dst, .. } | EmOp::MultAcc { dst, .. } => (*dst, true),
                _ => (usize::MAX, false),
            };
            if is_mult {
                let mut epi = Epi::default();
                if let Some(EmOp::Add { dst, a, b, sub: false, .. }) = ops.get(i) {
                    if *dst == d0 && *a == d0 && *b == d0 {
                        epi.dbl = true;
                        i += 1;
                    }
                }
                if let Some(EmOp::AddConst { x, k, .. }) = ops.get(i) {
                    if *x == d0 {
                        epi.k = Some(*k);
                        i += 1;
                    }
                }
                match &mut op {
                    EmOp::Mult { epi: e, .. } | EmOp::MultAcc { epi: e, .. } => *e = epi,
                    _ => {}
                }
            }
            if let (EmOp::MulConst { x, n, k }, Some(EmOp::Rescale { x: rx, n: rn, k: None })) = (&op, ops.get(i)) {
                if x == rx && n == rn {
                    op = EmOp::Rescale { x: *x, n: *n, k: Some(*k) };
                    i += 1;
                }
            }
            out.push(op);
        }
        out
    }
}

#[cfg(test)]
mod general_ps_tests {
    use super::*;

    fn tables() -> ScaleTables {
        let chain = pc_core::params::Chain::generate(&pc_core::params::ChainSpec::n32()).expect("n32 chain");
        ScaleTables::precompute(&chain).expect("tables")
    }

    /// The general builder reproduces the single-level m = 2 program op for op (the shipped degree-14 series).
    #[test]
    fn general_equals_m2_on_the_shipped_series() {
        let st = tables();
        let input = FMeta { top: 45, nl: 2, nf: st.sf(45).unwrap() * st.sf(45).unwrap() };
        let ps = PsPlan::evalmod();
        let mut a = Builder::new(&st);
        let xa = a.input(input);
        build_chebyshev_m2(&mut a, xa, &ps).unwrap();
        let mut b = Builder::new(&st);
        let xb = b.input(input);
        build_chebyshev_general(&mut b, xb, &ps.coeffs).unwrap();
        assert_eq!(a.prog.ops, b.prog.ops);
        assert_eq!(a.prog.consts, b.prog.consts);
        assert_eq!(a.metas[xa].top, b.metas[xb].top);
        assert_eq!(a.metas[xa].nf.to_bits(), b.metas[xb].nf.to_bits());
        // the lean builders agree too, with FIDESlib's output level and scale and fewer rescales
        let mut c = Builder::new(&st);
        c.lean = true;
        let xc = c.input(input);
        build_chebyshev_m2(&mut c, xc, &ps).unwrap();
        let mut e = Builder::new(&st);
        e.lean = true;
        let xe = e.input(input);
        build_chebyshev_general(&mut e, xe, &ps.coeffs).unwrap();
        assert_eq!(c.prog.ops, e.prog.ops);
        assert_eq!((c.metas[xc].top, c.metas[xc].nl, c.metas[xc].nf.to_bits()), (a.metas[xa].top, a.metas[xa].nl, a.metas[xa].nf.to_bits()));
        let rescales = |p: &EmProgram| p.ops.iter().filter(|o| matches!(o, EmOp::Rescale { .. })).count();
        assert!(rescales(&c.prog) < rescales(&a.prog));
    }

    /// The shipped tables never reach `wsum`'s chunked path (> WSUM_MAX sources, a functional bootstrap's long series):
    /// every weighted sum reads T's of the baby-step basis, at most k of them, and k <= WSUM_MAX for each.
    #[test]
    fn shipped_tables_below_the_wsum_split() {
        for c in [&pc_core::bts_setup::EVALMOD_CHEB[..], &pc_core::bts_setup::SPARSE_B_CHEB_K24[..], &pc_core::bts_setup::SPARSE_B_K16_W14[..]] {
            let k = PsPlan::from_coeffs(c).unwrap().k;
            assert!(k <= WSUM_MAX, "degree {}: k {k}", c.len() - 1);
        }
    }

    /// SparseB's K = 24 degree-20 series (m = 3, recursive PS) builds; depth as FIDESlib's.
    #[test]
    fn k24_series_builds() {
        let st = tables();
        let input = FMeta { top: 51, nl: 2, nf: st.sf(51).unwrap() * st.sf(51).unwrap() };
        let ps = PsPlan::from_coeffs(&pc_core::bts_setup::SPARSE_B_CHEB_K24).unwrap();
        assert_eq!((ps.k, ps.m), (3, 3), "OpenFHE ComputeDegreesPS(20)");
        let (prog, out) = build_evalmod(&st, input, &ps, 5).unwrap();
        assert!(!prog.ops.is_empty());
        println!("K24 EvalMod: {} ops, out {:?}", prog.ops.len(), out);
    }
}

/// An exact replay of an EvalMod program on a degree-0 plaintext (every buffer one residue per prime, no
/// encryption, no noise): the program's modular arithmetic and rescale roundings as the device runs them, for
/// checking a program's constants and scales against the real-number series it encodes (tests, prototypes).
pub mod sim {
    use super::{EmOp, Epi};
    use pc_core::modmul::{invmod, mulmod};
    use pc_core::scale::ScaleTables;

    /// Residues of the integer nearest `v` mod q (|v| < 2^126).
    pub fn encode(v: f64, q: &[u64]) -> Vec<u64> {
        let x = v.round() as i128;
        q.iter().map(|&p| x.rem_euclid(p as i128) as u64).collect()
    }

    /// The centred integer of `r` on primes `ids` (their product < 2^126).
    fn crt(r: &[u64], q: &[u64], ids: &[usize]) -> i128 {
        let mut x: u128 = 0;
        let mut m: u128 = 1;
        for &i in ids {
            let p = q[i] as u128;
            // x + m t == r_i mod p
            let xm = (x % p) as u64;
            let d = (r[i] + q[i] - xm) % q[i];
            let t = mulmod(d, invmod((m % p) as u64, q[i]), q[i]);
            x += m * t as u128;
            m *= p;
        }
        if x > m / 2 {
            x as i128 - m as i128
        } else {
            x as i128
        }
    }

    /// Rescale `x` (rows 0..n) by the primes a rescale from top n - 1 drops: round(x / Q_dropped).
    pub fn rescale(st: &ScaleTables, x: &mut [u64], n: usize) {
        let top = n - 1;
        let g = st.drop.get(top).copied().filter(|&g| g > 0).unwrap_or(st.d);
        let ids: Vec<usize> = (n - g..n).collect();
        let xd = crt(x, &st.q, &ids);
        for i in 0..n - g {
            let p = st.q[i];
            let mut qd = 1u64;
            for &j in &ids {
                qd = mulmod(qd, st.q[j] % p, p);
            }
            let r = (x[i] + p - xd.rem_euclid(p as i128) as u64) % p;
            x[i] = mulmod(r, invmod(qd, p), p);
        }
        for v in x.iter_mut().take(n).skip(n - g) {
            *v = 0;
        }
    }

    fn epi(x: &mut [u64], e: &Epi, consts: &[Vec<u64>], q: &[u64], n: usize) {
        for i in 0..n {
            if e.dbl {
                x[i] = (x[i] + x[i]) % q[i];
            }
            if let Some(k) = e.k {
                x[i] = (x[i] + consts[k][i] % q[i]) % q[i];
            }
        }
    }

    /// Run `ops` over `bufs` (each |Q| residues).
    pub fn run(st: &ScaleTables, ops: &[EmOp], consts: &[Vec<u64>], bufs: &mut Vec<Vec<u64>>) {
        let q = &st.q;
        let nq = q.len();
        for op in ops {
            let need = super::sim_ids(op).into_iter().max().unwrap_or(0) + 1;
            if bufs.len() < need {
                bufs.resize(need, vec![0; nq]);
            }
            match op {
                EmOp::Copy { dst, src, n } => {
                    let s = bufs[*src].clone();
                    bufs[*dst][..*n].copy_from_slice(&s[..*n]);
                    for v in bufs[*dst][*n..].iter_mut() {
                        *v = 0;
                    }
                }
                EmOp::Add { dst, a, b, n, sub } => {
                    for i in 0..*n {
                        let (x, y) = (bufs[*a][i] % q[i], bufs[*b][i] % q[i]);
                        bufs[*dst][i] = if *sub { (x + q[i] - y) % q[i] } else { (x + y) % q[i] };
                    }
                }
                EmOp::MulConst { x, n, k } => {
                    for i in 0..*n {
                        bufs[*x][i] = mulmod(bufs[*x][i] % q[i], consts[*k][i] % q[i], q[i]);
                    }
                }
                EmOp::AddConst { x, n, k } => {
                    for i in 0..*n {
                        bufs[*x][i] = (bufs[*x][i] % q[i] + consts[*k][i] % q[i]) % q[i];
                    }
                }
                EmOp::Rescale { x, n, k } => {
                    if let Some(k) = k {
                        for i in 0..*n {
                            bufs[*x][i] = mulmod(bufs[*x][i] % q[i], consts[*k][i] % q[i], q[i]);
                        }
                    }
                    rescale(st, &mut bufs[*x], *n);
                }
                EmOp::Mult { dst, a, b, n, epi: e } => {
                    let v: Vec<u64> = (0..*n).map(|i| mulmod(bufs[*a][i] % q[i], bufs[*b][i] % q[i], q[i])).collect();
                    bufs[*dst][..*n].copy_from_slice(&v);
                    epi(&mut bufs[*dst], e, consts, q, *n);
                }
                EmOp::MultAcc { dst, pairs, n, epi: e } => {
                    for i in 0..*n {
                        let mut acc = bufs[*dst][i] % q[i];
                        for (a, b) in pairs {
                            acc = (acc + mulmod(bufs[*a][i] % q[i], bufs[*b][i] % q[i], q[i])) % q[i];
                        }
                        bufs[*dst][i] = acc;
                    }
                    epi(&mut bufs[*dst], e, consts, q, *n);
                }
                EmOp::WSum { dst, srcs, n, k } => {
                    let v: Vec<u64> = (0..*n)
                        .map(|i| {
                            let mut acc = 0u64;
                            for (j, s) in srcs.iter().enumerate() {
                                acc = (acc + mulmod(bufs[*s][i] % q[i], consts[*k][j * nq + i] % q[i], q[i])) % q[i];
                            }
                            acc
                        })
                        .collect();
                    bufs[*dst][..*n].copy_from_slice(&v);
                }
            }
        }
    }

    /// The real value of buffer `x` with metadata (top, nl, nf): rescaled once more when nl = 2 (exact), then the
    /// centred integer on the bottom primes over the scale.
    pub fn decode(st: &ScaleTables, x: &[u64], top: i32, nl: i32, nf: f64) -> f64 {
        let mut v = x.to_vec();
        let (mut top, mut nf) = (top as usize, nf);
        if nl == 2 {
            let g = st.drop.get(top).copied().filter(|&g| g > 0).unwrap_or(st.d);
            nf /= st.mod_reduce_product(top as i32);
            rescale(st, &mut v, top + 1);
            top -= g;
        }
        let mut ids = vec![];
        let mut bits = 0.0;
        for i in 0..=top {
            bits += (st.q[i] as f64).log2();
            if bits > 120.0 {
                break;
            }
            ids.push(i);
        }
        crt(&v, &st.q, &ids) as f64 / nf
    }
}

/// Buffer ids an op names (for `sim::run`).
pub fn sim_ids(op: &EmOp) -> Vec<usize> {
    match op {
        EmOp::Copy { dst, src, .. } => vec![*dst, *src],
        EmOp::Add { dst, a, b, .. } | EmOp::Mult { dst, a, b, .. } => vec![*dst, *a, *b],
        EmOp::MulConst { x, .. } | EmOp::AddConst { x, .. } | EmOp::Rescale { x, .. } => vec![*x],
        EmOp::MultAcc { dst, pairs, .. } => std::iter::once(*dst).chain(pairs.iter().flat_map(|(a, b)| [*a, *b])).collect(),
        EmOp::WSum { dst, srcs, .. } => std::iter::once(*dst).chain(srcs.iter().copied()).collect(),
    }
}

#[cfg(test)]
mod sim_tests {
    use super::*;

    fn tables() -> ScaleTables {
        let chain = pc_core::params::Chain::generate(&pc_core::params::ChainSpec::n32()).expect("n32 chain");
        ScaleTables::precompute(&chain).expect("tables")
    }

    /// OpenFHE convention (c0 not halved).
    fn chebval(x: f64, c: &[f64]) -> f64 {
        let (mut b1, mut b2) = (0.0, 0.0);
        for &ck in c.iter().skip(1).rev() {
            let b0 = 2.0 * x * b1 - b2 + ck;
            b2 = b1;
            b1 = b0;
        }
        x * b1 - b2 + c[0] / 2.0
    }

    /// Max |program - series| over inputs in [-1, 1] for a series alone on an input at (top, nl = 1).
    fn series_err(st: &ScaleTables, top: i32, c: &[f64], stage: &str) -> f64 {
        let input = FMeta { top, nl: 1, nf: st.sf(top).unwrap() };
        let mut b = Builder::new(st);
        b.lean = true;
        let x = b.input(input);
        build_chebyshev_general(&mut b, x, c).unwrap();
        let out = b.metas[x];
        let ops = match stage {
            "raw" => b.prog.ops.clone(),
            _ => b.prog.fused(),
        };
        let mut e: f64 = 0.0;
        for i in 0..=20 {
            let u = -1.0 + i as f64 / 10.0;
            let mut bufs = vec![sim::encode(u * input.nf, &st.q)];
            sim::run(st, &ops, &b.prog.consts, &mut bufs);
            let got = sim::decode(st, &bufs[x], out.top, out.nl, out.nf);
            e = e.max((got - chebval(u, c)).abs());
        }
        e
    }

    #[test]
    fn sim_shipped_series() {
        let st = tables();
        let top = st.l as i32 - if st.drop[st.l] == 0 { 1 } else { 0 };
        let c = pc_core::bts_setup::evalmod_coeffs().to_vec();
        for nl in [1, 2] {
            let input = FMeta { top, nl, nf: st.sf(top).unwrap().powi(nl) };
            let mut b = Builder::new(&st);
            b.lean = true;
            let x = b.input(input);
            build_chebyshev_general(&mut b, x, &c).unwrap();
            let out = b.metas[x];
            for u in [0.0, 0.3, -0.7] {
                let mut bufs = vec![sim::encode(u * input.nf, &st.q)];
                sim::run(&st, &b.prog.ops, &b.prog.consts, &mut bufs);
                let got = sim::decode(&st, &bufs[x], out.top, out.nl, out.nf);
                assert!((got - chebval(u, &c)).abs() < 1e-12, "shipped nl {nl} u {u}: {got} vs {}", chebval(u, &c));
            }
        }
    }

    /// A series whose PS remainder has zero coefficients below its top (here: odd series plus a constant) reads the
    /// T's of its non-zero weights (FIDESlib's first-|w| T's gave 1.94 off on this one); and SparseB's K = 24 table.
    #[test]
    fn sim_sparse_series() {
        let st = tables();
        let top = st.l as i32 - if st.drop[st.l] == 0 { 1 } else { 0 };
        let mut c: Vec<f64> = (0..=15).map(|i| if i % 2 == 1 { 0.4 / (i as f64) } else { 0.0 }).collect();
        c[0] = 0.5;
        c[14] = 1e-3;
        assert!(series_err(&st, top, &c, "raw") < 1e-12);
        assert!(series_err(&st, top, &pc_core::bts_setup::SPARSE_B_CHEB_K24, "raw") < 1e-12);
        // an even series of degree 342 (GELU-like): a recursion level whose c quotient is a constant
        let c: Vec<f64> = (0..=342).map(|i| if i % 2 == 0 { 0.5 * 0.99f64.powi(i as i32) } else { 1e-3 * 0.98f64.powi(i as i32) }).collect();
        assert!(series_err(&st, top, &c, "raw") < 1e-11);
    }

    #[test]
    fn sim_series_constants() {
        let st = tables();
        let top = st.l as i32 - if st.drop[st.l] == 0 { 1 } else { 0 };
        for deg in [14usize, 20, 64, 120, 284] {
            for c0 in [0.0, 0.1, 2.0, 16.0] {
                let mut c: Vec<f64> = (0..=deg).map(|i| 0.3 * 0.9f64.powi(i as i32) * (1.0 + i as f64).sin()).collect();
                c[0] = c0;
                for stage in ["raw", "fused"] {
                    let e = series_err(&st, top, &c, stage);
                    assert!(e < 1e-12, "deg {deg} c0 {c0} {stage}: max err {e:.3e}");
                }
            }
        }
    }
}
