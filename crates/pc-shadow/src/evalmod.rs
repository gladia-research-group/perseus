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
    /// x := i x (`neg`: -i x), the monomial X^(N/2): exact, level, degree and scale kept.
    MulI { x: usize, n: usize, neg: bool },
    /// x := x + conj(x), d := x - conj(x) (both of the old x): one key switch with the conjugation key at x's level.
    ConjSplit { x: usize, d: usize, n: usize },
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
    /// Trigonometric functional bootstrapping (`PsPlan::fbs_trig`): this plan's series is the cosine of the base
    /// angle, `trig` the sine half and the series on it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trig: Option<Box<TrigSeries>>,
    /// Functional bootstrapping, sine form: the base series and `post` on the paired Paterson-Stockmeyer builder
    /// (`build_sine_paired`, one level shallower than FIDESlib's at the same degree) instead of FIDESlib's.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub paired: bool,
}

/// The sine half of a trigonometric FBS plan (`PsPlan::trig`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrigSeries {
    /// The base sine series (amplitude A^(2^-r) sin of the base angle).
    pub sin: PsPlan,
    /// Q of P(t) + s Q(t) (evaluator convention), s = A sin(2 pi u).
    pub post_s: Vec<f64>,
    /// t = A cos(2 pi u) + `off` (added at the last double angle).
    pub off: f64,
}

impl PsPlan {
    /// The plan for a Chebyshev coefficient table, computed by OpenFHE's long divisions (`pc_core::cheb`).
    pub fn from_coeffs(coefficients: &[f64]) -> pc_core::error::Result<Self> {
        let p = pc_core::cheb::PsPlan::new(coefficients)?;
        Ok(PsPlan { k: p.k, m: p.m, divqr_q: p.divqr.q.clone(), divcs_q: p.divcs.q.clone(), s2: p.s2.clone(), coeffs: coefficients.to_vec(), amp: 0.0, post: vec![], trig: None, paired: false })
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

    /// Trigonometric functional bootstrapping (f folded into EvalMod's own cosine / sine: the Fourier extension of
    /// Bian et al., ePrint 2026/367, and the trigonometric minimax of Xiao et al., ePrint 2026/1255): the base pair
    /// `cos_base`, `sin_base` (amplitude 1 cos / sin of the base angle, `trig_base`) rescaled to amplitude `amp`,
    /// `da_its` paired double angles to c = A cos(2 pi u), s = A sin(2 pi u), then P(t) + s Q(t) on t = c + `off`.
    /// A degree-n trigonometric polynomial of the phase u is P of degree n and Q of degree n - 1 in cos(2 pi u); on
    /// a domain |u| <= u_max < 1/2 they are fitted on cos(2 pi u) in [cos(2 pi u_max), 1], mapped to t in [-1, 1].
    #[allow(clippy::too_many_arguments)]
    pub fn fbs_trig(cos_base: &[f64], sin_base: &[f64], da_its: u32, amp: f64, off: f64, post: &[f64], post_s: &[f64]) -> pc_core::error::Result<Self> {
        let a = amp.powf(1.0 / (1u64 << da_its) as f64);
        let mut p = Self::from_coeffs(&cos_base.iter().map(|x| x * a).collect::<Vec<_>>())?;
        let sin = Self::from_coeffs(&sin_base.iter().map(|x| x * a).collect::<Vec<_>>())?;
        p.amp = amp;
        p.post = post.to_vec();
        p.trig = Some(Box::new(TrigSeries { sin, post_s: post_s.to_vec(), off }));
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
    /// x := i x (`neg`: -i x).
    pub fn mul_i(&mut self, x: usize, neg: bool) {
        let n = self.n(x);
        self.prog.ops.push(EmOp::MulI { x, n, neg });
    }
    /// x := x + conj(x) = 2 Re x; returns d = x - conj(x) = 2 i Im x (same metadata, degree 1).
    pub fn conj_split(&mut self, x: usize) -> Result<usize> {
        if self.metas[x].nl == 2 {
            self.rescale(x)?;
        }
        let d = self.fresh();
        self.metas[d] = self.metas[x];
        let n = self.n(x);
        self.prog.ops.push(EmOp::ConjSplit { x, d, n });
        Ok(d)
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
    // OpenFHE's level arithmetic (one level = d primes) as limb tops: `Fx::up` / `down` step by the levels' own primes
    // on a chain with an alternating single / pair user region
    let cu = x;
    let dc = degree(&ps.divcs_q);
    let mut flag_c = false;
    let lvl_t2 = |b: &Builder| {
        let t = b.metas[t2[m - 1]];
        if t.nl == 1 { b.fx().up(t.top) } else { t.top }
    };
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
            let tl = b.fx().down(lvl_t2(b));
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
    let cu = out;
    // as `build_chebyshev`'s: OpenFHE's level arithmetic by the levels' own primes (`Fx::up` / `down`)
    let lvl_t2 = |b: &Builder| {
        let t = b.metas[t2[m - 1]];
        let mut top = if t.nl == 1 { b.fx().up(t.top) } else { t.top };
        for _ in 0..level_offset {
            top = b.fx().down(top);
        }
        top
    };
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
            let tl = b.fx().down(lvl_t2(b));
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

/// A device-time proxy of a program (limb-weighted: a product with its relinearization ~4 rescales, a weighted-sum
/// source ~1/4, the element-wise ops ~1/5).
pub fn op_cost(ops: &[EmOp]) -> f64 {
    ops.iter()
        .map(|op| match op {
            EmOp::Mult { n, .. } => 4.0 * *n as f64,
            EmOp::MultAcc { n, pairs, .. } => (3.0 + pairs.len() as f64) * *n as f64,
            EmOp::Rescale { n, .. } => *n as f64,
            EmOp::WSum { n, srcs, .. } => 0.25 * (*n * srcs.len()) as f64,
            EmOp::Copy { n, .. } | EmOp::Add { n, .. } | EmOp::MulConst { n, .. } | EmOp::AddConst { n, .. } | EmOp::MulI { n, .. } => 0.2 * *n as f64,
            EmOp::ConjSplit { n, .. } => 3.4 * *n as f64,
        })
        .sum()
}

/// A plain EvalMod table (the shipped tables' function): the Chebyshev interpolant on [-1, 1] (evaluator convention)
/// of (2 pi)^(-1/2^r) cos(2 pi (K y - 1/4) / 2^r) at degree `deg`; after r double angles it reads sin(2 pi K y) / 2 pi.
pub fn sine_base(k: f64, da_its: u32, deg: usize) -> Vec<f64> {
    let s = (1u64 << da_its) as f64;
    let two_pi = 2.0 * std::f64::consts::PI;
    let n = deg + 1;
    (0..n)
        .map(|j| {
            let acc: f64 = (0..n)
                .map(|i| {
                    let th = std::f64::consts::PI * (i as f64 + 0.5) / n as f64;
                    two_pi.powf(-1.0 / s) * (two_pi * (k * th.cos() - 0.25) / s).cos() * (j as f64 * th).cos()
                })
                .sum();
            acc * 2.0 / n as f64
        })
        .collect()
}

/// A plain EvalMod table spec "deg:da" (`sine_base` at degree `deg`, `da` double angles): (deg, da).
pub fn parse_em_table(spec: &str) -> Option<(usize, u32)> {
    let (d, r) = spec.trim().split_once(':')?;
    let (d, r): (usize, u32) = (d.trim().parse().ok()?, r.trim().parse().ok()?);
    ((4..=128).contains(&d) && (1..=6).contains(&r)).then_some((d, r))
}

/// Levels a plain EvalMod program on `b` (coefficients, double angles) lands above the one on `a` from the same
/// `input` (FIDESlib's Paterson-Stockmeyer shape, lean; realized: a pending rescale counted as done). The depth of a
/// table is a property of the series, not of where it starts: the compile shifts the dense routes' measured landings
/// by it.
pub fn table_landing_shift(st: &ScaleTables, input: FMeta, a: (&[f64], u32), b: (&[f64], u32)) -> Result<i32> {
    let top = |(c, r): (&[f64], u32)| -> Result<i32> {
        let ps = PsPlan::from_coeffs(c).map_err(|e| rule("evalmod table", e.to_string()))?;
        let (_, out) = build_evalmod_opts(st, input, &ps, r as i32, true)?;
        Ok(if out.nl == 2 { Fx { d: st.d as i32, st: Some(st) }.down(out.top) } else { out.top })
    };
    // in level units (`d` per level: a single-prime level of an alternating user region counts as one level)
    let (ta, tb) = (top(a)?, top(b)?);
    let fx = Fx { d: st.d as i32, st: Some(st) };
    let (lo, hi, sign) = if tb >= ta { (ta, tb, 1) } else { (tb, ta, -1) };
    let (mut t, mut k) = (lo, 0);
    while t < hi {
        t = fx.up(t);
        k += 1;
    }
    if t != hi {
        return Err(rule("evalmod table shift", format!("landings {ta} / {tb} off the level grid")));
    }
    Ok(sign * k * st.d as i32)
}

/// The base pair of a trigonometric FBS (`PsPlan::fbs_trig`): Chebyshev interpolants on [-1, 1] (evaluator convention)
/// of cos and sin of 2 pi K y / 2^r at degree `deg`; after r paired double angles, y = (u + I) / K reads
/// (cos 2 pi u, sin 2 pi u) for every ModRaise integer |I| < K.
pub fn trig_base(k: f64, da_its: u32, deg: usize) -> (Vec<f64>, Vec<f64>) {
    let w = 2.0 * std::f64::consts::PI * k / (1u64 << da_its) as f64;
    let n = deg + 1;
    let interp = |par: usize, f: &dyn Fn(f64) -> f64| -> Vec<f64> {
        (0..n)
            .map(|j| {
                if j % 2 != par {
                    return 0.0;
                }
                let s: f64 = (0..n).map(|i| {
                    let th = std::f64::consts::PI * (i as f64 + 0.5) / n as f64;
                    f(th.cos()) * (j as f64 * th).cos()
                }).sum();
                s * 2.0 / n as f64
            })
            .collect()
    };
    (interp(0, &|y| (w * y).cos()), interp(1, &|y| (w * y).sin()))
}

/// f = q T_g + r (true Chebyshev coefficients, value = sum f_i T_i), deg r < g; deg f < 2 g
/// (T_i = 2 T_(i-g) T_g - T_(2g-i) for g < i < 2 g).
fn cheb_divide(f: &[f64], g: usize) -> (Vec<f64>, Vec<f64>) {
    let mut r = f.to_vec();
    let mut q = vec![0f64; f.len().saturating_sub(g).max(1)];
    for i in (g..f.len()).rev() {
        let c = r[i];
        r[i] = 0.0;
        if c == 0.0 {
            continue;
        }
        let a = i - g;
        if a == 0 {
            q[0] += c;
        } else {
            q[a] += 2.0 * c;
            r[2 * g - i] -= c;
        }
    }
    r.truncate(g);
    (q, r)
}

fn top_deg(c: &[f64]) -> Option<usize> {
    c.iter().rposition(|&x| x != 0.0)
}

/// The baby / giant powers of a paired Paterson-Stockmeyer evaluation of P(t) + s Q(t).
struct PairPowers {
    k: usize,
    /// T_1..T_k of t (index i - 1)
    t: Vec<usize>,
    /// s T_i, i = 0..k-1 (s T_0 = s)
    st: Vec<usize>,
    /// T_(k 2^j), j = 0..m-1
    g: Vec<usize>,
    /// s T_(k 2^j) (the giant placement of s: `st` empty), and s itself for the leaves
    sg: Vec<usize>,
    s: Option<usize>,
}

enum PairVal {
    Ct(usize),
    Const(f64),
}

/// P(t) + s Q(t) of degree < k 2^m (true coefficients) on the powers `pw`: Q enters only at the leaves (s T_i), so
/// the product by s costs no level of its own (the leaves sit above the giant powers).
fn pair_eval(b: &mut Builder, pw: &PairPowers, p: &[f64], q: &[f64], m: usize) -> Result<Option<PairVal>> {
    let deg = top_deg(p).max(top_deg(q));
    let Some(deg) = deg else { return Ok(None) };
    let k = pw.k;
    if deg < k || m == 0 {
        if deg >= k {
            return Err(rule("evalmod trig", format!("leaf of degree {deg} >= k {k}")));
        }
        let mut srcs = vec![];
        let mut w = vec![];
        for i in 1..k.min(p.len()) {
            if p[i] != 0.0 {
                srcs.push(pw.t[i - 1]);
                w.push(p[i]);
            }
        }
        if !pw.st.is_empty() {
            for i in 0..k.min(q.len()) {
                if q[i] != 0.0 {
                    if pw.st[i] == usize::MAX {
                        return Err(rule("evalmod trig", format!("Q coefficient {i} has no leaf power (the exponential form's sine half is odd)")));
                    }
                    srcs.push(pw.st[i]);
                    w.push(q[i]);
                }
            }
        }
        let c0 = p.first().copied().unwrap_or(0.0);
        let leaf = if srcs.is_empty() {
            PairVal::Const(c0)
        } else {
            let dst = b.fresh();
            let top = srcs.iter().map(|&x| b.metas[x].top).min().unwrap_or(-1);
            b.metas[dst] = FMeta { top, nl: 1, nf: 1.0 };
            b.wsum(dst, &srcs, &w)?;
            b.add_scalar(dst, c0)?;
            PairVal::Ct(dst)
        };
        if !pw.st.is_empty() || top_deg(q).is_none() {
            return Ok(Some(leaf));
        }
        // the giant placement: s times the leaf's Q part (one product; the leaf sits a level above its parent's)
        let s_ = pw.s.ok_or_else(|| rule("evalmod trig", "a Q leaf without s"))?;
        let lq = pair_eval(b, pw, q, &[], 0)?;
        let sq = match lq {
            None => return Ok(Some(leaf)),
            Some(PairVal::Const(c)) => {
                let r = b.fresh();
                b.copy(r, s_);
                b.mult_scalar(r, c)?;
                r
            }
            Some(PairVal::Ct(h)) => {
                if b.metas[h].nl == 2 {
                    b.rescale(h)?;
                }
                b.mult(h, s_)?;
                h
            }
        };
        match leaf {
            PairVal::Const(c) => b.add_scalar(sq, c)?,
            PairVal::Ct(l) => b.add(sq, l)?,
        }
        return Ok(Some(PairVal::Ct(sq)));
    }
    let gd = k << (m - 1);
    if deg < gd {
        return pair_eval(b, pw, p, q, m - 1);
    }
    let (pq, pr) = cheb_divide(p, gd);
    let (qq, qr) = cheb_divide(q, gd);
    let giant = pw.st.is_empty() && top_deg(&qq).is_some();
    let hi = pair_eval(b, pw, &pq, if giant { &[] } else { &qq }, m - 1)?;
    // the giant placement: the quotient's Q part times s T_(k 2^(m-1)), a pure series (no s inside)
    let hq = if giant { pair_eval(b, pw, &qq, &[], m - 1)? } else { None };
    let lo = pair_eval(b, pw, &pr, &qr, m - 1)?;
    let times = |b: &mut Builder, v: Option<PairVal>, g: usize| -> Result<Option<usize>> {
        Ok(match v {
            None => None,
            Some(PairVal::Const(c)) => {
                let r = b.fresh();
                b.copy(r, g);
                b.mult_scalar(r, c)?;
                Some(r)
            }
            Some(PairVal::Ct(h)) => {
                if b.metas[h].nl == 2 {
                    b.rescale(h)?;
                }
                b.mult(h, g)?;
                Some(h)
            }
        })
    };
    let r1 = times(b, hi, pw.g[m - 1])?;
    let r2 = if giant { times(b, hq, pw.sg[m - 1])? } else { None };
    let res = match (r1, r2) {
        (Some(a), Some(c)) => {
            b.add(a, c)?;
            a
        }
        (Some(a), None) | (None, Some(a)) => a,
        (None, None) => return Ok(lo),
    };
    match lo {
        None => {}
        Some(PairVal::Const(c)) => b.add_scalar(res, c)?,
        Some(PairVal::Ct(l)) => b.add(res, l)?,
    }
    Ok(Some(PairVal::Ct(res)))
}

fn true_coeffs(c: &[f64]) -> Vec<f64> {
    let mut v = c.to_vec();
    if let Some(c0) = v.first_mut() {
        *c0 /= 2.0;
    }
    v
}

/// The baby steps T_1..T_k of `t`, the giant steps T_(k 2^j) up to the degree `deg` (m of them), and when `s` is
/// given its products: s T_0..s T_(k-1) (`giant` false: Q at the leaves) or s T_(k 2^j) (`giant`: Q through the
/// giant powers, s times the bottom leaves only); returns (powers, m).
fn pair_powers(b: &mut Builder, t: usize, s: Option<usize>, k: usize, deg: usize, giant: bool) -> Result<(PairPowers, usize)> {
    let mut m = 0;
    while (k << m) <= deg {
        m += 1;
    }
    let tp = cheb_powers(b, t, k)?;
    let mut st = Vec::with_capacity(k);
    if let (Some(s), false) = (s, giant) {
        for i in 0..k {
            let v = b.fresh();
            if i == 0 {
                b.copy(v, s);
            } else {
                b.copy(v, tp[i - 1]);
                b.mult(v, s)?;
                if b.metas[v].nl == 2 {
                    b.rescale(v)?;
                }
            }
            st.push(v);
        }
    }
    let mut g = vec![tp[k - 1]];
    for j in 1..m {
        let v = b.fresh();
        b.square2(v, g[j - 1])?;
        b.add(v, v)?;
        b.add_scalar(v, -1.0)?;
        if b.metas[v].nl == 2 {
            b.rescale(v)?;
        }
        g.push(v);
    }
    let mut sg = vec![];
    if let (Some(s), true) = (s, giant) {
        for &gj in &g {
            let v = b.fresh();
            b.copy(v, gj);
            b.mult(v, s)?;
            if b.metas[v].nl == 2 {
                b.rescale(v)?;
            }
            sg.push(v);
        }
    }
    Ok((PairPowers { k, t: tp, st, g, sg, s }, m))
}

fn pair_result(v: Option<PairVal>) -> Result<usize> {
    match v {
        None => Err(rule("evalmod trig", "empty series")),
        Some(PairVal::Const(_)) => Err(rule("evalmod trig", "a constant series")),
        Some(PairVal::Ct(r)) => Ok(r),
    }
}

/// A trigonometric FBS (`PsPlan::fbs_trig`) on input `x`, the result in `x`: the cos / sin base series on shared
/// powers of the input (`kb` baby steps), `its` paired double angles (c, s) -> (2 c^2 - a_j, 2 s c) (one more product
/// per step than the cosine chain, same depth; the last adds `off`), then P(t) + s Q(t) by a paired
/// Paterson-Stockmeyer with `k` baby steps (Q at the leaves, or through the giant powers: `giant`).
#[allow(clippy::too_many_arguments)]
fn build_trig(b: &mut Builder, x: usize, ps: &PsPlan, t: &TrigSeries, its: i32, kb: usize, k: usize, giant: bool) -> Result<()> {
    if b.metas[x].nl == 2 {
        b.rescale(x)?;
    }
    let cb = true_coeffs(&ps.coeffs);
    let sb = true_coeffs(&t.sin.coeffs);
    let dbase = top_deg(&cb).max(top_deg(&sb)).unwrap_or(0);
    let (pw, m) = pair_powers(b, x, None, kb, dbase, false)?;
    let c = pair_result(pair_eval(b, &pw, &cb, &[], m)?)?;
    let mut s = pair_result(pair_eval(b, &pw, &sb, &[], m)?)?;
    b.copy(x, c);
    // (c, s) -> (2 c^2 - a_j, 2 s c): the cosine chain's step (the complex square (c - s)(c + s) would halve its
    // error growth near |c| = a_j but needs c and s at one scale: an alignment level), the last adds `off`
    for j in 1..=its {
        for v in [x, s] {
            if b.metas[v].nl == 2 {
                b.rescale(v)?;
            }
        }
        let s2 = b.fresh();
        b.copy(s2, s);
        b.mult(s2, x)?;
        b.add(s2, s2)?;
        s = s2;
        b.square(x)?;
        let mut scalar = -ps.amp.powf(2f64.powf((j - its) as f64));
        if j == its {
            scalar += t.off;
        }
        b.add(x, x)?;
        b.add_scalar(x, scalar)?;
    }
    for v in [x, s] {
        if b.metas[v].nl == 2 {
            b.rescale(v)?;
        }
    }
    let p = true_coeffs(&ps.post);
    let q = true_coeffs(&t.post_s);
    let deg = top_deg(&p).max(top_deg(&q)).unwrap_or(0);
    let has_q = top_deg(&q).is_some();
    let (pw, m) = pair_powers(b, x, if has_q { Some(s) } else { None }, k, deg, giant)?;
    let r = pair_result(pair_eval(b, &pw, &p, &q, m)?)?;
    b.copy(x, r);
    Ok(())
}

/// The baby / giant powers of `t` as `pair_powers` (Q at the leaves), with i T_i (the monomial X^(N/2), exact: no
/// product, no level) in place of s T_i: P + i Q evaluates as one series (a complex leaf per giant product). The
/// constant i T_0 has no ciphertext (`usize::MAX`): Q must have no constant term at any leaf (an odd Q).
fn pair_powers_i(b: &mut Builder, t: usize, k: usize, deg: usize) -> Result<(PairPowers, usize)> {
    let (mut pw, m) = pair_powers(b, t, None, k, deg, false)?;
    let mut st = vec![usize::MAX];
    for i in 1..k {
        let v = b.fresh();
        b.copy(v, pw.t[i - 1]);
        b.mul_i(v, false);
        st.push(v);
    }
    pw.st = st;
    Ok((pw, m))
}

/// The exponential form of a trigonometric FBS (`PsPlan::fbs_trig`), on the real slots EvalMod reads: z = a e^(i th)
/// of the base angle as ONE series on the input's powers (the sine half's leaves times i: the giant products shared),
/// `its` squarings z <- z^2 (one product per step instead of the pair's two; the error grows 2x per step instead of
/// the cosine chain's up to 4x), then one conjugation: c = z + conj z = A cos 2 pi u (+ `off`), s = -i (z - conj z)
/// = A sin 2 pi u, and P(t) + s Q(t) as `build_trig`. The base amplitude makes z^(2^its) = A / 2 e^(2 pi i u).
#[allow(clippy::too_many_arguments)]
fn build_trig_exp(b: &mut Builder, x: usize, ps: &PsPlan, t: &TrigSeries, its: i32, kb: usize, k: usize, giant: bool) -> Result<()> {
    if b.metas[x].nl == 2 {
        b.rescale(x)?;
    }
    let f = 0.5f64.powf(1.0 / (1u64 << its) as f64);
    let cb: Vec<f64> = true_coeffs(&ps.coeffs).iter().map(|v| v * f).collect();
    let sb: Vec<f64> = true_coeffs(&t.sin.coeffs).iter().map(|v| v * f).collect();
    if cb.iter().skip(1).step_by(2).chain(sb.iter().step_by(2)).any(|v| *v != 0.0) {
        return Err(rule("evalmod trig", "the exponential form needs an even cosine and an odd sine half"));
    }
    let dbase = top_deg(&cb).max(top_deg(&sb)).unwrap_or(0);
    let (pw, m) = pair_powers_i(b, x, kb, dbase)?;
    let z = pair_result(pair_eval(b, &pw, &cb, &sb, m)?)?;
    for _ in 0..its {
        b.square(z)?;
    }
    let d = b.conj_split(z)?;
    b.mul_i(d, true);
    b.add_scalar(z, t.off)?;
    let p = true_coeffs(&ps.post);
    let q = true_coeffs(&t.post_s);
    let deg = top_deg(&p).max(top_deg(&q)).unwrap_or(0);
    let has_q = top_deg(&q).is_some();
    let (pw, m) = pair_powers(b, z, if has_q { Some(d) } else { None }, k, deg, giant)?;
    let r = pair_result(pair_eval(b, &pw, &p, &q, m)?)?;
    b.copy(x, r);
    Ok(())
}

/// The sine form (`PsPlan::fbs`) on the paired Paterson-Stockmeyer builder (`pair_eval`, no Q): the base series on
/// the input's powers (`kb` baby steps), the double angles, the series `post` (`k` baby steps). The shallowest and
/// then cheapest (`op_cost`) over the baby-step counts.
pub fn build_sine_paired(st: &ScaleTables, input: FMeta, ps: &PsPlan, double_angle_its: i32) -> Result<(EmProgram, FMeta)> {
    let mut best: Option<(EmProgram, FMeta)> = None;
    for (kb, k) in [4usize, 8, 16].iter().flat_map(|&kb| [2usize, 4, 8, 16].map(move |k| (kb, k))) {
        let mut b = Builder::new(st);
        b.lean = true;
        let x = b.input(input);
        let run = |b: &mut Builder| -> Result<()> {
            let cb = true_coeffs(&ps.coeffs);
            let (pw, m) = pair_powers(b, x, None, kb, top_deg(&cb).unwrap_or(0), false)?;
            let c = pair_result(pair_eval(b, &pw, &cb, &[], m)?)?;
            b.copy(x, c);
            build_double_angle_amp(b, x, double_angle_its, ps.amp)?;
            if b.metas[x].nl == 2 {
                b.rescale(x)?;
            }
            if ps.post.is_empty() {
                return Ok(());
            }
            let p = true_coeffs(&ps.post);
            let (pw, m) = pair_powers(b, x, None, k, top_deg(&p).unwrap_or(0), false)?;
            let r = pair_result(pair_eval(b, &pw, &p, &[], m)?)?;
            b.copy(x, r);
            Ok(())
        };
        if run(&mut b).is_err() {
            continue;
        }
        let out = b.metas[x];
        if best.as_ref().is_none_or(|(p, o)| out.top > o.top || (out.top == o.top && op_cost(&b.prog.ops) < op_cost(&p.ops))) {
            best = Some((b.prog, out));
        }
    }
    best.ok_or_else(|| rule("evalmod paired", "no program"))
}

/// The whole EvalMod chain of one input: the Chebyshev series, then `double_angle_its` double-angle steps.
pub fn build_evalmod(st: &ScaleTables, input: FMeta, ps: &PsPlan, double_angle_its: i32) -> Result<(EmProgram, FMeta)> {
    build_evalmod_opts(st, input, ps, double_angle_its, false)
}

/// A trigonometric FBS program (`PsPlan::fbs_trig`): over the baby-step counts of the base and the series, Q's
/// placement and the double-angle forms `exps` (false: the paired double angles, `build_trig`; true: the exponential
/// form, `build_trig_exp`), the shallowest program, then the cheapest (`op_cost`).
pub fn build_trig_opts(st: &ScaleTables, input: FMeta, ps: &PsPlan, double_angle_its: i32, lean: bool, exps: &[bool]) -> Result<(EmProgram, FMeta)> {
    let t = ps.trig.as_deref().ok_or_else(|| rule("evalmod trig", "not a trigonometric plan"))?;
    let mut best: Option<(EmProgram, FMeta)> = None;
    let shapes = [4usize, 8, 16].iter().flat_map(|&kb| [2usize, 4, 8, 16].iter().flat_map(move |&k| [(kb, k, false), (kb, k, true)]));
    for ((kb, k, giant), &exp) in shapes.flat_map(|s| exps.iter().map(move |e| (s, e))) {
        let mut b = Builder::new(st);
        b.lean = lean;
        let x = b.input(input);
        let built = if exp { build_trig_exp(&mut b, x, ps, t, double_angle_its, kb, k, giant) } else { build_trig(&mut b, x, ps, t, double_angle_its, kb, k, giant) };
        if built.is_err() {
            continue;
        }
        let out = b.metas[x];
        let better = match &best {
            None => true,
            Some((p, o)) => out.top > o.top || (out.top == o.top && op_cost(&b.prog.ops) < op_cost(&p.ops)),
        };
        if better {
            best = Some((b.prog, out));
        }
    }
    best.ok_or_else(|| rule("evalmod trig", "no program"))
}

/// `build_evalmod` with `Builder::lean`.
pub fn build_evalmod_opts(st: &ScaleTables, input: FMeta, ps: &PsPlan, double_angle_its: i32, lean: bool) -> Result<(EmProgram, FMeta)> {
    let mut b = Builder::new(st);
    b.lean = lean;
    let x = b.input(input);
    // a series after the sine on a chain with an alternating user region always takes the paired builder: FIDESlib's
    // lazy Paterson-Stockmeyer adjusts its operands across the region's single / pair levels with integer scalars that
    // lose ~2^-25 (exact replay: 4e-8 .. 5e-7 at degrees 16 .. 128), the paired one lands them exactly
    if ps.trig.is_none() && (ps.paired || (st.singles != 0 && !ps.post.is_empty())) {
        return build_sine_paired(st, input, ps, double_angle_its);
    }
    if ps.trig.is_some() {
        // PC_FBS_TRIG_EXP=1 (experiments): the exponential form (`build_trig_exp`, ~10 % faster real, ~15 % complex,
        // same landing and RMS) in place of the paired double angles. Off by default: it turns the imaginary part
        // of its input and of its own rescales into an amplitude error, which the variance model does not carry
        // (measured: the slot nearest the root 1 of every ciphertext, up to 20x the paired form's error)
        let exps: &[bool] = match std::env::var("PC_FBS_TRIG_EXP").as_deref() {
            Ok("1") => &[true],
            _ => &[false],
        };
        return build_trig_opts(st, input, ps, double_angle_its, lean, exps);
    }
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

    /// Run `ops` over `bufs` (each |Q| residues): real slot values (a program with `MulI` / `ConjSplit` needs
    /// `run_c`).
    pub fn run(st: &ScaleTables, ops: &[EmOp], consts: &[Vec<u64>], bufs: &mut Vec<Vec<u64>>) {
        let mut im = vec![];
        run_c(st, ops, consts, bufs, &mut im);
    }

    /// `run` on complex slot values: `re` and `im` the residues of the two components (Gaussian integers mod q; a
    /// missing `im` buffer reads zero).
    pub fn run_c(st: &ScaleTables, ops: &[EmOp], consts: &[Vec<u64>], re: &mut Vec<Vec<u64>>, im: &mut Vec<Vec<u64>>) {
        let q = &st.q;
        let nq = q.len();
        let neg = |x: u64, p: u64| if x % p == 0 { 0 } else { p - x % p };
        for op in ops {
            let need = (super::sim_ids(op).into_iter().max().unwrap_or(0) + 1).max(re.len());
            if re.len() < need {
                re.resize(need, vec![0; nq]);
            }
            if im.len() < need {
                im.resize(need, vec![0; nq]);
            }
            match op {
                EmOp::Copy { dst, src, n } => {
                    for b in [&mut *re, &mut *im] {
                        let s = b[*src].clone();
                        b[*dst][..*n].copy_from_slice(&s[..*n]);
                        for v in b[*dst][*n..].iter_mut() {
                            *v = 0;
                        }
                    }
                }
                EmOp::Add { dst, a, b, n, sub } => {
                    for bf in [&mut *re, &mut *im] {
                        for i in 0..*n {
                            let (x, y) = (bf[*a][i] % q[i], bf[*b][i] % q[i]);
                            bf[*dst][i] = if *sub { (x + q[i] - y) % q[i] } else { (x + y) % q[i] };
                        }
                    }
                }
                EmOp::MulConst { x, n, k } => {
                    for b in [&mut *re, &mut *im] {
                        for i in 0..*n {
                            b[*x][i] = mulmod(b[*x][i] % q[i], consts[*k][i] % q[i], q[i]);
                        }
                    }
                }
                EmOp::AddConst { x, n, k } => {
                    for i in 0..*n {
                        re[*x][i] = (re[*x][i] % q[i] + consts[*k][i] % q[i]) % q[i];
                    }
                }
                EmOp::Rescale { x, n, k } => {
                    for b in [&mut *re, &mut *im] {
                        if let Some(k) = k {
                            for i in 0..*n {
                                b[*x][i] = mulmod(b[*x][i] % q[i], consts[*k][i] % q[i], q[i]);
                            }
                        }
                        rescale(st, &mut b[*x], *n);
                    }
                }
                EmOp::Mult { dst, a, b, n, epi: e } => {
                    let (mut vr, mut vi) = (vec![0u64; *n], vec![0u64; *n]);
                    for i in 0..*n {
                        let p = q[i];
                        let (ar, ai, br, bi) = (re[*a][i] % p, im[*a][i] % p, re[*b][i] % p, im[*b][i] % p);
                        vr[i] = (mulmod(ar, br, p) + neg(mulmod(ai, bi, p), p)) % p;
                        vi[i] = (mulmod(ar, bi, p) + mulmod(ai, br, p)) % p;
                    }
                    re[*dst][..*n].copy_from_slice(&vr);
                    im[*dst][..*n].copy_from_slice(&vi);
                    epi(&mut re[*dst], e, consts, q, *n);
                    epi(&mut im[*dst], &Epi { k: None, ..*e }, consts, q, *n);
                }
                EmOp::MultAcc { dst, pairs, n, epi: e } => {
                    for i in 0..*n {
                        let p = q[i];
                        let (mut accr, mut acci) = (re[*dst][i] % p, im[*dst][i] % p);
                        for (a, b) in pairs {
                            let (ar, ai, br, bi) = (re[*a][i] % p, im[*a][i] % p, re[*b][i] % p, im[*b][i] % p);
                            accr = (accr + mulmod(ar, br, p) + neg(mulmod(ai, bi, p), p)) % p;
                            acci = (acci + mulmod(ar, bi, p) + mulmod(ai, br, p)) % p;
                        }
                        re[*dst][i] = accr;
                        im[*dst][i] = acci;
                    }
                    epi(&mut re[*dst], e, consts, q, *n);
                    epi(&mut im[*dst], &Epi { k: None, ..*e }, consts, q, *n);
                }
                EmOp::WSum { dst, srcs, n, k } => {
                    for b in [&mut *re, &mut *im] {
                        let v: Vec<u64> = (0..*n)
                            .map(|i| {
                                let mut acc = 0u64;
                                for (j, s) in srcs.iter().enumerate() {
                                    acc = (acc + mulmod(b[*s][i] % q[i], consts[*k][j * nq + i] % q[i], q[i])) % q[i];
                                }
                                acc
                            })
                            .collect();
                        b[*dst][..*n].copy_from_slice(&v);
                    }
                }
                EmOp::MulI { x, n, neg: ng } => {
                    // i (a + i b) = -b + i a; -i (a + i b) = b - i a
                    for i in 0..*n {
                        let p = q[i];
                        let (a, b) = (re[*x][i] % p, im[*x][i] % p);
                        (re[*x][i], im[*x][i]) = if *ng { (b, neg(a, p)) } else { (neg(b, p), a) };
                    }
                }
                EmOp::ConjSplit { x, d, n } => {
                    for i in 0..*n {
                        let p = q[i];
                        let (a, b) = (re[*x][i] % p, im[*x][i] % p);
                        (re[*x][i], im[*x][i]) = ((a + a) % p, 0);
                        (re[*d][i], im[*d][i]) = (0, (b + b) % p);
                    }
                    for v in re[*d][*n..].iter_mut().chain(im[*d][*n..].iter_mut()) {
                        *v = 0;
                    }
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

/// Noise of a program, per slot, linearized: the program's value in integer units (`sim`'s integers, as f64), its
/// derivative with respect to the input (forward values, reverse-mode adjoints through every op), and the variance the
/// fresh noise sources add at the output: every rescale's rounding (`Fresh::rescale`, added after it) and every
/// relinearization (`Fresh::relin`, at the product's scale), independent of each other and of the input, each times
/// its adjoint squared. The input's own error enters as derivative x error (first order: the products of two errors
/// are dropped, ~2^-40 relative here).
pub mod noise {
    use super::{EmOp, Epi};
    use pc_core::modmul::{invmod, mulmod};
    use pc_core::scale::ScaleTables;

    /// Variances (integer units squared, per real slot component) a rescale's rounding and a relinearization add.
    #[derive(Clone, Copy, Debug)]
    pub struct Fresh {
        pub rescale: f64,
        pub relin: f64,
    }

    /// The centred integer of the residues `r` on primes 0..n (balanced mixed radix), as f64.
    pub fn centred(r: &[u64], q: &[u64], n: usize) -> f64 {
        let mut digits: Vec<i64> = Vec::with_capacity(n);
        for i in 0..n {
            let p = q[i];
            // the value of the digits so far mod p, and M_i = prod_{k < i} q_k mod p
            let (mut acc, mut m) = (0u64, 1u64);
            for (k, &t) in digits.iter().enumerate() {
                acc = (acc + mulmod(t.rem_euclid(p as i64) as u64, m, p)) % p;
                m = mulmod(m, q[k] % p, p);
            }
            let d = mulmod((r[i] % p + p - acc) % p, invmod(m, p), p);
            digits.push(if d > p / 2 { d as i64 - p as i64 } else { d as i64 });
        }
        // the constants are small next to the modulus: the high digits are zero (and their weights overflow f64)
        let (mut v, mut m) = (0f64, 1f64);
        for (k, &t) in digits.iter().enumerate() {
            if t != 0 {
                v += t as f64 * m;
            }
            m *= q[k] as f64;
        }
        v
    }

    fn dropped(st: &ScaleTables, n: usize) -> f64 {
        let g = st.drop.get(n - 1).copied().filter(|&g| g > 0).unwrap_or(st.d);
        (n - g..n).map(|i| st.q[i] as f64).product()
    }

    /// The constants every op reads, decoded once (`run`'s second argument): per op, in its read order.
    pub fn prepare(st: &ScaleTables, ops: &[EmOp], consts: &[Vec<u64>]) -> Vec<Vec<f64>> {
        let q = &st.q;
        let nq = q.len();
        let epi_c = |e: &Epi, n: usize| e.k.map(|k| centred(&consts[k], q, n)).into_iter().collect::<Vec<_>>();
        ops.iter()
            .map(|op| match op {
                EmOp::MulConst { n, k, .. } | EmOp::AddConst { n, k, .. } => vec![centred(&consts[*k], q, *n)],
                EmOp::Rescale { n, k, .. } => vec![k.map_or(1.0, |k| centred(&consts[k], q, *n))],
                EmOp::Mult { n, epi: e, .. } | EmOp::MultAcc { n, epi: e, .. } => epi_c(e, *n),
                EmOp::WSum { srcs, n, k, .. } => (0..srcs.len()).map(|j| centred(&consts[*k][j * nq..(j + 1) * nq], q, *n)).collect(),
                EmOp::Copy { .. } | EmOp::Add { .. } | EmOp::MulI { .. } | EmOp::ConjSplit { .. } => vec![],
            })
            .collect()
    }

    /// The linearized program on input `x` (integer units, buffer 0) read at buffer `out`: (value, d value / d x,
    /// variance of the fresh noise at `out`), integer units; `pre` = `prepare(st, ops, consts)`. Slot values are
    /// complex (a program with `MulI` / `ConjSplit` carries a signal in the imaginary part); the input is real, the
    /// output its real part, every fresh source independent in each real component (`Fresh` is per component), the
    /// adjoints the transposed real 2 x 2 Jacobians (a complex product: times the conjugate).
    pub fn linearize(st: &ScaleTables, ops: &[EmOp], pre: &[Vec<f64>], x: f64, out: usize, fresh: Fresh) -> (f64, f64, f64) {
        type C = (f64, f64);
        let mul = |a: C, b: C| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
        let conj_mul = |a: C, b: C| (a.0 * b.0 + a.1 * b.1, a.0 * b.1 - a.1 * b.0);
        let n2 = |a: C| a.0 * a.0 + a.1 * a.1;
        let nb = ops.iter().flat_map(super::sim_ids).max().unwrap_or(0).max(out) + 1;
        let mut v = vec![(0f64, 0f64); nb];
        v[0] = (x, 0.0);
        // forward: the operand values each product reads
        let mut tape: Vec<Vec<(C, C)>> = Vec::with_capacity(ops.len());
        let epi = |x: C, e: &Epi, c: &[f64]| {
            let x = if e.dbl { (2.0 * x.0, 2.0 * x.1) } else { x };
            if e.k.is_some() {
                (x.0 + c[0], x.1)
            } else {
                x
            }
        };
        for (op, c) in ops.iter().zip(pre) {
            let mut t = vec![];
            match op {
                EmOp::Copy { dst, src, .. } => v[*dst] = v[*src],
                EmOp::Add { dst, a, b, sub, .. } => {
                    let (va, vb) = (v[*a], v[*b]);
                    v[*dst] = if *sub { (va.0 - vb.0, va.1 - vb.1) } else { (va.0 + vb.0, va.1 + vb.1) };
                }
                EmOp::MulConst { x, .. } => v[*x] = (v[*x].0 * c[0], v[*x].1 * c[0]),
                EmOp::AddConst { x, .. } => v[*x].0 += c[0],
                EmOp::Rescale { x, n, .. } => {
                    let f = c[0] / dropped(st, *n);
                    v[*x] = (v[*x].0 * f, v[*x].1 * f);
                }
                EmOp::Mult { dst, a, b, epi: e, .. } => {
                    t.push((v[*a], v[*b]));
                    v[*dst] = epi(mul(v[*a], v[*b]), e, c);
                }
                EmOp::MultAcc { dst, pairs, epi: e, .. } => {
                    let mut acc = v[*dst];
                    for (a, b) in pairs {
                        t.push((v[*a], v[*b]));
                        let p = mul(v[*a], v[*b]);
                        acc = (acc.0 + p.0, acc.1 + p.1);
                    }
                    v[*dst] = epi(acc, e, c);
                }
                EmOp::WSum { dst, srcs, .. } => {
                    v[*dst] = srcs.iter().enumerate().fold((0.0, 0.0), |a, (j, s)| (a.0 + v[*s].0 * c[j], a.1 + v[*s].1 * c[j]))
                }
                EmOp::MulI { x, neg, .. } => {
                    let (a, b) = v[*x];
                    v[*x] = if *neg { (b, -a) } else { (-b, a) };
                }
                EmOp::ConjSplit { x, d, .. } => {
                    let (a, b) = v[*x];
                    v[*x] = (2.0 * a, 0.0);
                    v[*d] = (0.0, 2.0 * b);
                }
            }
            tape.push(t);
        }
        // reverse: adjoints of every buffer, the fresh sources' variance where they enter
        let mut g = vec![(0f64, 0f64); nb];
        g[out] = (1.0, 0.0);
        let mut var = 0.0;
        let take = |g: &mut Vec<C>, i: usize| std::mem::replace(&mut g[i], (0.0, 0.0));
        for ((op, c), t) in ops.iter().zip(pre).rev().zip(tape.iter().rev()) {
            match op {
                EmOp::Copy { dst, src, .. } => {
                    let d = take(&mut g, *dst);
                    g[*src] = (g[*src].0 + d.0, g[*src].1 + d.1);
                }
                EmOp::Add { dst, a, b, sub, .. } => {
                    let d = take(&mut g, *dst);
                    g[*a] = (g[*a].0 + d.0, g[*a].1 + d.1);
                    g[*b] = if *sub { (g[*b].0 - d.0, g[*b].1 - d.1) } else { (g[*b].0 + d.0, g[*b].1 + d.1) };
                }
                EmOp::MulConst { x, .. } => g[*x] = (g[*x].0 * c[0], g[*x].1 * c[0]),
                EmOp::AddConst { .. } => {}
                EmOp::Rescale { x, n, .. } => {
                    var += n2(g[*x]) * fresh.rescale;
                    let f = c[0] / dropped(st, *n);
                    g[*x] = (g[*x].0 * f, g[*x].1 * f);
                }
                EmOp::Mult { dst, a, b, epi: e, .. } => {
                    let m = if e.dbl { 2.0 } else { 1.0 };
                    let d = take(&mut g, *dst);
                    let d = (d.0 * m, d.1 * m);
                    var += n2(d) * fresh.relin;
                    let (va, vb) = t[0];
                    let (ga, gb) = (conj_mul(vb, d), conj_mul(va, d));
                    g[*a] = (g[*a].0 + ga.0, g[*a].1 + ga.1);
                    g[*b] = (g[*b].0 + gb.0, g[*b].1 + gb.1);
                }
                EmOp::MultAcc { dst, pairs, epi: e, .. } => {
                    let m = if e.dbl { 2.0 } else { 1.0 };
                    let d = take(&mut g, *dst);
                    let d = (d.0 * m, d.1 * m);
                    var += n2(d) * fresh.relin;
                    g[*dst] = (g[*dst].0 + d.0, g[*dst].1 + d.1);
                    for ((a, b), (va, vb)) in pairs.iter().zip(t) {
                        let (ga, gb) = (conj_mul(*vb, d), conj_mul(*va, d));
                        g[*a] = (g[*a].0 + ga.0, g[*a].1 + ga.1);
                        g[*b] = (g[*b].0 + gb.0, g[*b].1 + gb.1);
                    }
                }
                EmOp::WSum { dst, srcs, .. } => {
                    let d = take(&mut g, *dst);
                    for (j, s) in srcs.iter().enumerate() {
                        g[*s] = (g[*s].0 + d.0 * c[j], g[*s].1 + d.1 * c[j]);
                    }
                }
                EmOp::MulI { x, neg, .. } => {
                    // the transpose of (a, b) -> (-b, a) is (g0, g1) -> (g1, -g0)
                    let (g0, g1) = g[*x];
                    g[*x] = if *neg { (-g1, g0) } else { (g1, -g0) };
                }
                EmOp::ConjSplit { x, d, .. } => {
                    let (gx, gd) = (take(&mut g, *x), take(&mut g, *d));
                    // the key switch's noise e enters x as + e and d as - e
                    var += n2((gx.0 - gd.0, gx.1 - gd.1)) * fresh.relin;
                    g[*x] = (2.0 * gx.0, 2.0 * gd.1);
                }
            }
        }
        (v[out].0, g[0].0, var)
    }

    /// (value, derivative, fresh variance) of `linearize` read as `sim::decode` reads buffer metadata (top, nl, nf):
    /// in value units, a degree-2 output rescaled once more (its rounding added).
    pub fn decode(st: &ScaleTables, l: (f64, f64, f64), top: i32, nl: i32, nf: f64, fresh: Fresh) -> (f64, f64, f64) {
        if nl == 2 {
            let qd = dropped(st, top as usize + 1);
            let nf = nf / st.mod_reduce_product(top);
            return (l.0 / qd / nf, l.1 / qd / nf, (l.2 / (qd * qd) + fresh.rescale) / (nf * nf));
        }
        (l.0 / nf, l.1 / nf, l.2 / (nf * nf))
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
        EmOp::MulI { x, .. } => vec![*x],
        EmOp::ConjSplit { x, d, .. } => vec![*x, *d],
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

    /// The linearized noise replay reads the program's values as the exact replay does, its derivative is the program's
    /// (central differences), and its fresh variance is at least the last rescale's rounding.
    #[test]
    fn noise_linearize_matches_sim() {
        let st = tables();
        let top = st.l as i32 - if st.drop[st.l] == 0 { 1 } else { 0 };
        let input = FMeta { top, nl: 1, nf: st.sf(top).unwrap() };
        let mut c: Vec<f64> = (0..=40).map(|i| 0.3 * 0.9f64.powi(i as i32) * (1.0 + i as f64).sin()).collect();
        c[0] = 0.5;
        let mut b = Builder::new(&st);
        b.lean = true;
        let x = b.input(input);
        build_chebyshev_general(&mut b, x, &c).unwrap();
        let out = b.metas[x];
        let pre = noise::prepare(&st, &b.prog.ops, &b.prog.consts);
        let fresh = noise::Fresh { rescale: 1e8, relin: 1e8 };
        for u in [-0.9, -0.3, 0.0, 0.45, 0.8] {
            let mut bufs = vec![sim::encode(u * input.nf, &st.q)];
            sim::run(&st, &b.prog.ops, &b.prog.consts, &mut bufs);
            let want = sim::decode(&st, &bufs[x], out.top, out.nl, out.nf);
            let l = noise::linearize(&st, &b.prog.ops, &pre, u * input.nf, x, fresh);
            let (v, d, var) = noise::decode(&st, l, out.top, out.nl, out.nf, fresh);
            assert!((v - want).abs() < 1e-9, "u {u}: {v} vs {want}");
            let h = 1e-5;
            let f = |y: f64| noise::decode(&st, noise::linearize(&st, &b.prog.ops, &pre, y * input.nf, x, fresh), out.top, out.nl, out.nf, fresh).0;
            let fd = (f(u + h) - f(u - h)) / (2.0 * h) / input.nf;
            assert!((d - fd).abs() < 1e-6 * fd.abs().max(1e-12) + 1e-20, "u {u}: d {d:.6e} vs {fd:.6e}");
            assert!(var > 0.0 && var.is_finite());
        }
        // the balanced mixed-radix decode of small constants, either sign, over many primes
        for v in [0i64, 1, -1, 123_456_789, -987_654_321_012] {
            let r: Vec<u64> = st.q.iter().map(|&p| (v as i128).rem_euclid(p as i128) as u64).collect();
            assert_eq!(noise::centred(&r, &st.q, st.q.len()), v as f64);
        }
    }
}
