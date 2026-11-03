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
    /// An even series (`build_even_prestep`): the series is in z = 2 (y - s)^2 - 1, not in the input y. None: the
    /// series is in y (every table but `pc_core::bts_setup::EVALMOD_EVEN_CHEB`).
    #[serde(default)]
    pub even_shift: Option<f64>,
}

impl PsPlan {
    /// The plan for a Chebyshev coefficient table, computed by OpenFHE's long divisions (`pc_core::cheb`).
    pub fn from_coeffs(coefficients: &[f64]) -> pc_core::error::Result<Self> {
        let p = pc_core::cheb::PsPlan::new(coefficients)?;
        Ok(PsPlan {
            k: p.k,
            m: p.m,
            divqr_q: p.divqr.q.clone(),
            divcs_q: p.divcs.q.clone(),
            s2: p.s2.clone(),
            coeffs: coefficients.to_vec(),
            even_shift: None,
        })
    }

    /// The EvalMod plan of the shipped bootstrap (degree 14, K=16 table).
    pub fn evalmod() -> Self {
        Self::from_coeffs(pc_core::bts_setup::evalmod_coeffs()).expect("frozen EvalMod table has a PS plan")
    }

    /// The even EvalMod plan (`pc_core::bts_setup::EVALMOD_EVEN_CHEB`, K = 16): degree 5 in z = 2 (y - 1/64)^2 - 1, one
    /// level less than the shipped series at the same output level and scale.
    pub fn evalmod_even() -> Self {
        let mut p = Self::from_coeffs(&pc_core::bts_setup::EVALMOD_EVEN_CHEB).expect("frozen even EvalMod table has a PS plan");
        p.even_shift = Some(pc_core::bts_setup::EVALMOD_EVEN_SHIFT);
        p
    }

    /// The K = 16 EvalMod plan: the even one when `even`, else the shipped one.
    pub fn evalmod_k16(even: bool) -> Self {
        if even {
            Self::evalmod_even()
        } else {
            Self::evalmod()
        }
    }
}

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
    /// Ciphertext::evalLinearWSumMutable
    pub fn wsum(&mut self, dst: usize, srcs: &[usize], w: &[f64]) -> Result<()> {
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
            let mut ncts = 0;
            for i in 0..ps.s2.len() - 1 {
                if ps.s2[i + 1] != 0.0 {
                    ncts += 1;
                    w.push(ps.s2[i + 1]);
                }
            }
            let tl = lvl_t2(b) - d_;
            b.grow_to_level(su, tl);
            b.drop_to_level(su, tl);
            // FIDESlib passes T (not the filtered list): the first ncts T's with the filtered weights
            b.wsum(su, &t[..ncts], &w)?;
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
            let mut ncts = 0;
            for i in 0..s2.len() - 1 {
                if s2[i + 1] != 0.0 {
                    ncts += 1;
                    w.push(s2[i + 1]);
                }
            }
            let tl = lvl_t2(b) - d_;
            b.grow_to_level(su, tl);
            b.drop_to_level(su, tl);
            // FIDESlib passes T (not the filtered list): the first ncts T's with the filtered weights
            b.wsum(su, &t[..ncts], &w)?;
            b.add_scalar(su, s2[0] / 2.0)?;
        } else {
            b.copy(su, t[k - 1]);
            b.add_scalar(su, s2[0] / 2.0)?;
        }
    }
    if !flag_c {
        return Err(rule("evalmod", "addScalar(T2, c) branch not ported"));
    }
    // lean, the lazy top call: the giant power realized before it joins cu (as `build_chebyshev_m2`)
    if b.lean && lazy && level_offset == 0 && m == max_m && b.metas[t2[m - 1]].nl == 2 {
        b.rescale(t2[m - 1])?;
    }
    if max_m - m <= 1 {
        b.adjust_add(t2[m - 1], cu)?;
    }
    if b.metas[t2[m - 1]].nl == 1 && b.metas[cu].nl == 2 {
        b.rescale(cu)?;
    }
    b.add(cu, t2[m - 1])?;
    if lazy && level_offset == 0 && m == max_m {
        return Ok(Some((qu, su)));
    }
    b.mult(cu, qu)?;
    b.add(cu, su)?;
    Ok(None)
}

/// applyDoubleAngleIterations
pub fn build_double_angle(b: &mut Builder, x: usize, its: i32) -> Result<()> {
    let r = its;
    for j in 1..=r {
        b.square(x)?;
        let scalar = -1.0 / (2.0 * std::f64::consts::PI).powf(2f64.powf((j - r) as f64));
        b.add(x, x)?;
        b.add_scalar(x, scalar)?;
    }
    Ok(())
}

/// The whole EvalMod chain of one input: the Chebyshev series, then `double_angle_its` double-angle steps.
pub fn build_evalmod(st: &ScaleTables, input: FMeta, ps: &PsPlan, double_angle_its: i32) -> Result<(EmProgram, FMeta)> {
    build_evalmod_opts(st, input, ps, double_angle_its, false)
}

/// The even series' argument in place: x <- 2 (x - s)^2 - 1 (a constant add, one square, and the x2 and -1 that
/// `EmProgram::fused` folds into the square's relinearization epilogue): one level, so a series of degree n in z
/// costs the depth of PS(n) + 1 where the series of degree 2n in y costs PS(2n).
pub fn build_even_prestep(b: &mut Builder, x: usize, shift: f64) -> Result<()> {
    b.add_scalar(x, -shift)?;
    b.square(x)?;
    b.add(x, x)?;
    b.add_scalar(x, -1.0)
}

/// `build_evalmod` with `Builder::lean`.
pub fn build_evalmod_opts(st: &ScaleTables, input: FMeta, ps: &PsPlan, double_angle_its: i32, lean: bool) -> Result<(EmProgram, FMeta)> {
    let mut b = Builder::new(st);
    b.lean = lean;
    let x = b.input(input);
    if let Some(s) = ps.even_shift {
        build_even_prestep(&mut b, x, s)?;
    }
    build_chebyshev(&mut b, x, ps)?;
    build_double_angle(&mut b, x, double_angle_its)?;
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

#[cfg(test)]
mod even_series_tests {
    use super::*;
    use pc_core::modmul::{invmod, mulmod};
    use pc_core::params::{Chain, ChainSpec};

    fn tables(spec: ChainSpec) -> ScaleTables {
        ScaleTables::precompute(&Chain::generate(&spec).expect("chain")).expect("tables")
    }

    fn specs() -> Vec<(&'static str, ChainSpec)> {
        vec![("n32", ChainSpec::n32()), ("n32a4", ChainSpec::n32a4()), ("n32sp", ChainSpec::n32sp())]
    }

    /// The program of the whole chain with a trailing rescale when the output is still degree 2 (so that the
    /// interpreter decodes at a single scale), and its output metadata.
    fn program(st: &ScaleTables, input: FMeta, ps: &PsPlan, lean: bool, finish: bool) -> Result<(EmProgram, FMeta)> {
        let mut b = Builder::new(st);
        b.lean = lean;
        let x = b.input(input);
        if let Some(s) = ps.even_shift {
            build_even_prestep(&mut b, x, s)?;
        }
        build_chebyshev(&mut b, x, ps)?;
        build_double_angle(&mut b, x, 5)?;
        if finish && b.metas[x].nl == 2 {
            b.rescale(x)?;
        }
        Ok((b.prog, b.metas[x]))
    }

    /// The even plan: degree 5 in z, shifted by 1/64 of y.
    #[test]
    fn even_plan_shape() {
        let ps = PsPlan::evalmod_even();
        assert_eq!(ps.coeffs.len(), 6);
        assert_eq!(ps.even_shift, Some(1.0 / 64.0));
        assert_eq!(PsPlan::evalmod().even_shift, None);
        assert_eq!(PsPlan::evalmod_k16(false), PsPlan::evalmod());
        assert_eq!(PsPlan::evalmod_k16(true), ps);
    }

    /// On every chain the runtime builds, wherever the shipped series builds from an input top and degree, the even
    /// series builds from the same degree one CKKS level lower and ends at the shipped output's level, degree and scale
    /// (bit for bit): the routes raise that level lower and no landing changes.
    #[test]
    fn even_lands_as_shipped_one_level_lower() {
        let (sh, ev) = (PsPlan::evalmod(), PsPlan::evalmod_even());
        for (name, spec) in specs() {
            let st = tables(spec);
            let d = st.d as i32;
            let mut n = 0;
            for top in d..=st.l as i32 {
                let low = top - d;
                if st.drop[top as usize] == 0 || st.drop[low as usize] == 0 {
                    continue;
                }
                let (sf, sf_low) = (st.sf(top).unwrap(), st.sf(low).unwrap());
                for (nl, nf, nf_low) in [(1, sf, sf_low), (2, sf * sf, sf_low * sf_low)] {
                    for lean in [false, true] {
                        let Ok((_, a)) = build_evalmod_opts(&st, FMeta { top, nl, nf }, &sh, 5, lean) else { continue };
                        let (_, e) = build_evalmod_opts(&st, FMeta { top: low, nl, nf: nf_low }, &ev, 5, lean)
                            .unwrap_or_else(|err| panic!("{name} top {top} nl {nl} lean {lean}: even fails: {err}"));
                        assert_eq!((a.top, a.nl, a.nf.to_bits()), (e.top, e.nl, e.nf.to_bits()), "{name} top {top} nl {nl} lean {lean}");
                        n += 1;
                    }
                }
            }
            assert!(n > 0, "{name}: no input top builds");
            println!("[even] {name}: {n} (top, degree, lean) inputs land as the shipped series");
        }
    }

    /// Op census: from one level lower, the even program does fewer products, rescales and weighted-sum terms than the
    /// shipped one, on fewer limbs.
    #[test]
    fn even_is_cheaper() {
        let st = tables(ChainSpec::n32sp());
        let top = (0..=st.l as i32).rev().find(|&t| st.drop[t as usize] > 0 && {
            let sf = st.sf(t).unwrap();
            build_evalmod_opts(&st, FMeta { top: t, nl: 2, nf: sf * sf }, &PsPlan::evalmod(), 5, true).is_ok()
        }).unwrap();
        let sf = st.sf(top).unwrap();
        let low = top - st.d as i32;
        let sf_low = st.sf(low).unwrap();
        let census = |ps: &PsPlan, input: FMeta| {
            let (p, _) = build_evalmod_opts(&st, input, ps, 5, true).unwrap();
            let ops = p.fused();
            let (mut mult, mut resc, mut wsum, mut limbs) = (0, 0, 0, [0usize; 3]);
            for o in &ops {
                match o {
                    EmOp::Mult { n, .. } | EmOp::MultAcc { n, .. } => {
                        mult += 1;
                        limbs[0] += n;
                    }
                    EmOp::Rescale { n, .. } => {
                        resc += 1;
                        limbs[1] += n;
                    }
                    EmOp::WSum { n, srcs, .. } => {
                        wsum += srcs.len();
                        limbs[2] += n * srcs.len();
                    }
                    _ => {}
                }
            }
            (mult, resc, wsum, limbs)
        };
        let a = census(&PsPlan::evalmod(), FMeta { top, nl: 2, nf: sf * sf });
        let e = census(&PsPlan::evalmod_even(), FMeta { top: low, nl: 2, nf: sf_low * sf_low });
        println!("[even] n32sp top {top}: shipped mult/rescale/wsum terms {:?}, limbs {:?}; even top {low}: {:?}, limbs {:?}", (a.0, a.1, a.2), a.3, (e.0, e.1, e.2), e.3);
        assert!(e.0 < a.0 && e.1 < a.1 && e.2 < a.2, "even not cheaper: {a:?} vs {e:?}");
        assert!(e.3.iter().sum::<usize>() < a.3.iter().sum::<usize>());
    }

    // ---- an exact interpreter of the program on constant polynomials (a trivial ciphertext's c0: every ring product
    // is the product of the constant terms, the slots all hold c / scale), residue by residue as the device does

    fn center(v: u64, q: u64) -> i128 {
        if v > q / 2 {
            v as i128 - q as i128
        } else {
            v as i128
        }
    }

    fn run(st: &ScaleTables, prog: &EmProgram, input_top: i32, x: Vec<u64>) -> Vec<Vec<u64>> {
        let nq = st.q.len();
        let mut bufs: Vec<Vec<u64>> = vec![vec![0; nq]; 1];
        bufs[0] = x;
        let q = &st.q;
        let fx = Fx { d: st.d as i32, st: Some(st) };
        let get = |bufs: &mut Vec<Vec<u64>>, i: usize| {
            if bufs.len() <= i {
                bufs.resize(i + 1, vec![0; nq]);
            }
        };
        let _ = input_top;
        for op in &prog.ops {
            match op {
                EmOp::Copy { dst, src, n } => {
                    get(&mut bufs, (*dst).max(*src));
                    let s = bufs[*src].clone();
                    bufs[*dst][..*n].copy_from_slice(&s[..*n]);
                }
                EmOp::Add { dst, a, b, n, sub } => {
                    get(&mut bufs, (*dst).max(*a).max(*b));
                    for i in 0..*n {
                        let (va, vb) = (bufs[*a][i], bufs[*b][i] % q[i]);
                        bufs[*dst][i] = if *sub { (va + q[i] - vb) % q[i] } else { (va + vb) % q[i] };
                    }
                }
                EmOp::MulConst { x, n, k } => {
                    for i in 0..*n {
                        bufs[*x][i] = mulmod(bufs[*x][i], prog.consts[*k][i] % q[i], q[i]);
                    }
                }
                EmOp::AddConst { x, n, k } => {
                    for i in 0..*n {
                        bufs[*x][i] = (bufs[*x][i] + prog.consts[*k][i] % q[i]) % q[i];
                    }
                }
                EmOp::Rescale { x, n, k } => {
                    if let Some(k) = k {
                        for i in 0..*n {
                            bufs[*x][i] = mulmod(bufs[*x][i], prog.consts[*k][i] % q[i], q[i]);
                        }
                    }
                    let top = *n as i32 - 1;
                    let new = fx.down(top);
                    let mut t = top;
                    while t > new {
                        let p = q[t as usize];
                        let r = center(bufs[*x][t as usize], p);
                        for i in 0..t as usize {
                            let rm = r.rem_euclid(q[i] as i128) as u64;
                            let d = (bufs[*x][i] + q[i] - rm) % q[i];
                            bufs[*x][i] = mulmod(d, invmod(p % q[i], q[i]), q[i]);
                        }
                        t -= 1;
                    }
                }
                EmOp::Mult { dst, a, b, n, epi } => {
                    get(&mut bufs, (*dst).max(*a).max(*b));
                    for i in 0..*n {
                        let mut v = mulmod(bufs[*a][i], bufs[*b][i], q[i]);
                        if epi.dbl {
                            v = (v + v) % q[i];
                        }
                        if let Some(k) = epi.k {
                            v = (v + prog.consts[k][i] % q[i]) % q[i];
                        }
                        bufs[*dst][i] = v;
                    }
                }
                EmOp::MultAcc { dst, pairs, n, epi } => {
                    let mx = pairs.iter().map(|&(a, b)| a.max(b)).max().unwrap_or(0).max(*dst);
                    get(&mut bufs, mx);
                    for i in 0..*n {
                        let mut v = bufs[*dst][i];
                        for &(a, b) in pairs {
                            v = (v + mulmod(bufs[a][i], bufs[b][i], q[i])) % q[i];
                        }
                        if epi.dbl {
                            v = (v + v) % q[i];
                        }
                        if let Some(k) = epi.k {
                            v = (v + prog.consts[k][i] % q[i]) % q[i];
                        }
                        bufs[*dst][i] = v;
                    }
                }
                EmOp::WSum { dst, srcs, n, k } => {
                    let mx = srcs.iter().copied().max().unwrap_or(0).max(*dst);
                    get(&mut bufs, mx);
                    for i in 0..*n {
                        let mut v = 0u64;
                        for (j, &s) in srcs.iter().enumerate() {
                            v = (v + mulmod(bufs[s][i], prog.consts[*k][j * nq + i] % q[i], q[i])) % q[i];
                        }
                        bufs[*dst][i] = v;
                    }
                }
            }
        }
        bufs
    }

    /// Garner on the bottom three primes: the centred value of a residue vector whose integer is below their product / 2.
    fn decode3(v: &[u64], q: &[u64]) -> f64 {
        let (q0, q1, q2) = (q[0] as i128, q[1] as i128, q[2] as i128);
        let a0 = v[0] as i128;
        let a1 = ((v[1] as i128 - a0).rem_euclid(q1) * invmod(q[0] % q[1], q[1]) as i128).rem_euclid(q1);
        let t = (a0 + a1 * q0).rem_euclid(q2);
        let a2 = ((v[2] as i128 - t).rem_euclid(q2) * invmod(((q[0] as u128 * q[1] as u128) % q[2] as u128) as u64, q[2]) as i128).rem_euclid(q2);
        let m = q0 * q1 * q2;
        let mut x = a0 + a1 * q0 + a2 * q0 * q1;
        if x > m / 2 {
            x -= m;
        }
        x as f64
    }

    /// P(|I| = i), i = 0..=15, of the overflow integer under the centred lift (`Lift::Centred`, the even series' raise):
    /// I = round(X), X the sum of h = 32 uniforms, so the density of 33 uniforms at i (exact rationals; the raise kernels
    /// on 1.3e8 coefficients under the production sparse secret agree). FIDESlib's per-prime lift is wider (sigma 2.35).
    const P_ABS_I: [f64; 16] = [
        2.394752e-01, 4.005643e-01, 2.337782e-01, 9.453113e-02,
        2.615467e-02, 4.858999e-03, 5.901276e-04, 4.513991e-05,
        2.064469e-06, 5.243321e-08, 6.634776e-10, 3.536999e-12,
        6.012096e-15, 1.953140e-18, 4.120381e-23, 3.279280e-30,
    ];

    /// The executed program against sin(2 pi u) / (2 pi), |delta| <= 2^-4: the worst error's bits at each |I| and the RMS
    /// bits over `P_ABS_I`.
    fn value_bits(st: &ScaleTables, input: FMeta, ps: &PsPlan, lean: bool) -> ([f64; 16], f64) {
        let (prog, out) = program(st, input, ps, lean, true).unwrap();
        let (mut worst, mut mse) = ([0.0f64; 16], 0.0f64);
        for abs_i in 0..16usize {
            let (mut e2, mut n) = (0.0f64, 0.0f64);
            for ii in [abs_i as i32, -(abs_i as i32)] {
                for dj in -8..=8 {
                    let u = ii as f64 + dj as f64 / 8.0 * 0.0625;
                    let c = (u / 16.0 * input.nf).round() as i128;
                    let x: Vec<u64> = (0..st.q.len()).map(|i| if i as i32 <= input.top { c.rem_euclid(st.q[i] as i128) as u64 } else { 0 }).collect();
                    let bufs = run(st, &prog, input.top, x);
                    let e = decode3(&bufs[0], &st.q) / out.nf - (2.0 * std::f64::consts::PI * u).sin() / (2.0 * std::f64::consts::PI);
                    worst[abs_i] = worst[abs_i].max(e.abs());
                    e2 += e * e;
                    n += 1.0;
                }
            }
            mse += P_ABS_I[abs_i] * e2 / n;
        }
        (worst.map(|w| -w.log2()), -mse.sqrt().log2())
    }

    /// The executed program, from the inputs the routes give it. The shipped table keeps 27.4 bits on every |I| <= 15
    /// (the interpreter reproduces them). The even series is fitted to the overflow integer's distribution: its RMS over
    /// `P_ABS_I` is what reaches a slot, and its error grows with |I|, floors from an independent evaluation of the table.
    #[test]
    fn even_series_value() {
        for (name, spec) in [("n32", ChainSpec::n32()), ("n32sp", ChainSpec::n32sp())] {
            let st = tables(spec);
            let top = (0..=st.l as i32).rev().find(|&t| st.drop[t as usize] > 0 && {
                let sf = st.sf(t).unwrap();
                build_evalmod_opts(&st, FMeta { top: t, nl: 2, nf: sf * sf }, &PsPlan::evalmod(), 5, true).is_ok()
            }).unwrap();
            let low = top - st.d as i32;
            let (sf, sf_low) = (st.sf(top).unwrap(), st.sf(low).unwrap());
            for lean in [false, true] {
                let (worst, rms) = value_bits(&st, FMeta { top, nl: 2, nf: sf * sf }, &PsPlan::evalmod(), lean);
                println!("[even] {name} top {top} shipped lean {lean}: rms {rms:.2} bits, worst by |I| {worst:.1?}");
                assert!(worst.iter().all(|&b| b >= 27.0), "{name} shipped lean {lean}: {worst:.1?}");
                let (worst, rms) = value_bits(&st, FMeta { top: low, nl: 2, nf: sf_low * sf_low }, &PsPlan::evalmod_even(), lean);
                println!("[even] {name} top {low} even lean {lean}: rms {rms:.2} bits, worst by |I| {worst:.1?}");
                assert!(rms >= 28.9, "{name} even lean {lean}: rms {rms:.2} bits");
                for (upto, floor) in [(5, 26.8), (8, 20.4), (12, 12.8), (15, 10.6)] {
                    let w = worst[..=upto].iter().cloned().fold(f64::INFINITY, f64::min);
                    assert!(w >= floor, "{name} even lean {lean}: |I| <= {upto} worst {w:.2} bits");
                }
            }
        }
    }
}
