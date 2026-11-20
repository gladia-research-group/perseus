"""Iterative solvers for symmetric positive definite systems on the ops protocol: conjugate gradients (CG), CG on a
symmetrically preconditioned system (Jacobi, or the client's sketch preconditioner for least squares) and the
Chebyshev semi-iteration, B systems per ciphertext; power iteration with deflation on the same layout.

Layout (`VecLay`). B problems of size n share a ciphertext as the cube (slots / (B n), B n, 1) of impl.linalg: a
matrix in its L face with wrap (A_b[i, j] at (a0, b n + i) for every a0 = j mod n), a vector in its R face with wrap
(v_b[i] at (a0, b n + i') for a0 = i mod n, every i'). A matrix-vector product is one ciphertext product and a window
sum of n terms down the cyclic axis 0 (log2 n rotations), landing in the O face (y_b[i] along a0); one diagonal
transform (`relayout`, one level) takes it back to the R face. A dot product of two R-face vectors is one product
and the same window sum: the result is broadcast over its problem's block, ready to scale a vector. Nothing crosses a
problem's block, so B = slots / n^2 systems run at the cost of one (n = 16: 128 systems per ciphertext).

What CKKS changes. A step of CG divides (alpha = r.r / p.A p, beta = r'.r' / r.r), and CKKS has no division: a
reciprocal is a polynomial on a PUBLIC band (impl.math). The band is the problem: once CG has converged in some
direction its scalars shrink with the residual, so a band that covers them all is kappa / tol^2 wide. Here every
vector is held at a public per-step envelope (the residual bound 2 sqrt(kappa) rho^k of CG's A-norm error, rho =
(sqrt(kappa) - 1) / (sqrt(kappa) + 1)), so p.Ap / envelope^2 stays in [lambda_min tau^2, lambda_max c] while the
iteration runs no more than `tau` ahead of the envelope; past that the reciprocal saturates (its value at 0 is its
value at the band's bottom), the step shrinks and the result stays where it got (the A-norm error is already below
tau x the envelope there). One reciprocal per step: beta = -(r'.A p) / (p.A p) shares alpha's. CG is not
self-correcting: a refresh's error in r is never recomputed away, so its result carries ~the refresh error times the
conditioning (`refine` restarts CG on the recomputed residual b - A x).

The Chebyshev semi-iteration (Golub and Varga 1961; Saad, Iterative Methods, 2003, sec. 12.1) needs no inner product
and no division: its coefficients follow from the public band alone, the residual b - A x is recomputed every step
(self-correcting) and only the iterates x_k, x_{k-1} (unit size by a public scale) are ever refreshed. It takes
~sqrt(kappa) ln(2 / eps) / 2 steps of two levels against CG's ~n steps of a dozen.

Every function returns true values (the public envelopes are folded back in the last step)."""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from . import linalg as la
from . import math as M


@dataclasses.dataclass(frozen=True)
class VecLay:
    """B systems of size n (n, B powers of two, B n^2 <= slots) in one ciphertext (see the module docstring)."""
    n: int
    B: int = 1
    slots: int = 1 << 15

    def __post_init__(self):
        assert self.n & (self.n - 1) == 0 and self.B & (self.B - 1) == 0, (self.n, self.B)
        assert self.B * self.n * self.n <= self.slots, "B n^2 <= slots"

    @property
    def cube(self):
        return la.Cube((self.slots // (self.B * self.n), self.B * self.n, 1), self.slots)

    @property
    def mat(self):
        return la.L(self.cube, (self.n, self.n), self.B, wrap=True)

    @property
    def vec(self):
        return la.R(self.cube, (self.n, 1), self.B, wrap=True)

    @property
    def out(self):
        return la.O(self.cube, (self.n, 1), self.B)

    @property
    def stride(self):
        return self.cube.strides[0]

    def encode_mat(self, A):
        return la.encode(np.asarray(A, dtype=np.float64), self.mat)

    def encode_vec(self, v):
        v = np.asarray(v, dtype=np.float64)
        return la.encode(v.reshape(v.shape[:-1] + (self.n, 1)), self.vec)

    def decode_vec(self, s):
        v = la.decode(s, self.vec)
        return v.reshape(self.B, self.n) if self.B > 1 else v.reshape(self.n)

    @classmethod
    def fill(cls, n, slots=1 << 15):
        """As many systems as fit: B = slots / n^2."""
        return cls(n, slots // (n * n), slots)


def consolidate(ops, v, lay: VecLay, scale: float = 1.0):
    """v (R face) with every copy of an entry set to ONE copy: (a0 < n, the first a1 of its block) masked (one level),
    replicated over the block's a1 (log2 n rotations) and over the a0 tiles (log2 (d0 / n) rotations). A refresh's
    error is independent per slot; without this the copies differ and products read a mix of them. `scale` rides
    the mask."""
    c = lay.cube
    co = c.coords()
    m = ((co[0] < lay.n) & (co[1] % lay.n == 0)).astype(np.float64)
    t = ops.mult_pt(v, np.tile(m * scale, lay.slots // c.P), kind="mask")
    t = ops.rotate_and_sum(t, -1, lay.n)
    s = lay.stride * lay.n
    return ops.rotate_and_sum(t, -s, lay.slots) if s < lay.slots else t


def refresh(ops, v, lay: VecLay):
    """A deliberate refresh of v (real payload) and `consolidate`."""
    la.refresh(ops, v, True)
    return consolidate(ops, v, lay)


def matvec(ops, a_l, v, lay: VecLay, scale: float = 1.0):
    """A v in the R face (`scale` rides the face move): one product, log2 n rotations, the face move: two levels."""
    q = la.mm(ops, a_l, v, lay.cube, lay.n)
    return la.relayout(ops, q, lay.out, lay.vec, scale=scale)


def dot(ops, u, v, lay: VecLay):
    """u . v per system, broadcast over its block (one product, log2 n rotations)."""
    s = lay.stride
    return ops.rotate_and_sum(ops.mult(u, v), s, s * lay.n)


def norm2(ops, u, lay: VecLay):
    s = lay.stride
    return ops.rotate_and_sum(ops.square(u), s, s * lay.n)


# ── conjugate gradients ──────────────────────────────────────────────────────────────

def cg_rate(kappa: float) -> float:
    q = math.sqrt(kappa)
    return (q - 1.0) / (q + 1.0)


def cg_iters(kappa: float, eps: float = 1e-3) -> int:
    """Steps for CG's A-norm error bound 2 rho^k to reach eps (Chebyshev bound)."""
    rho = cg_rate(kappa)
    if rho <= 0.0:
        return 1
    return max(1, int(math.ceil(math.log(eps / 2.0) / math.log(rho))))


def cg_envelope(kappa: float, iters: int) -> list:
    """s_k >= ||r_k|| / ||b||: CG's residual bound (||r|| <= sqrt(lambda_max) ||e||_A, ||e_0||_A <= ||b|| / sqrt(
    lambda_min)), capped at 1."""
    rho = cg_rate(kappa)
    return [min(1.0, 2.0 * math.sqrt(kappa) * rho ** k) for k in range(iters + 1)]


@dataclasses.dataclass(frozen=True)
class CGPlan:
    """The public schedule of a CG solve: bounds, steps, envelope, the reciprocal's band and its approximation."""
    lmin: float
    lmax: float
    bnorm: float
    iters: int
    tau: float
    env: tuple
    recip: object            # impl.math.Approx
    shift: float = 0.0       # added to p.Ap before the reciprocal

    @property
    def kappa(self):
        return self.lmax / self.lmin


def cg_plan(lmin: float, lmax: float, bnorm: float = 1.0, *, iters: int | None = None, eps: float = 1e-3,
            tau: float = 0.1, pmax: float = 1.2, recip_err: float = 1e-2) -> CGPlan:
    """The schedule for eigenvalues of A in [lmin, lmax] and ||b|| <= bnorm (all PUBLIC): `iters` (default
    cg_iters(kappa, eps)), the envelope, and the reciprocal of p.Ap on [lmin tau^2, lmax pmax^2] (relative error
    `recip_err`, chosen on its encrypted error by impl.math). `tau`: how far below the envelope the iteration may run
    before its steps shrink (the band's width is kappa (pmax / tau)^2). The reciprocal is taken of p.Ap + 2 lo (lo the
    band's bottom): a polynomial reciprocal evaluated below its band extrapolates without bound (measured: CG on the
    device returned 1e11..1e19 once its residual fell below the band; on the twin a refresh error of 3e-4 x the
    largest slot took p.Ap + lo below the band); shifted by twice the bottom, the argument stays in it up to a
    refresh error of lo, and the step only shrinks by 2 lo / p.Ap (< 2 tau^2 while the iteration follows the
    envelope)."""
    kappa = lmax / lmin
    it = iters or cg_iters(kappa, eps)
    env = cg_envelope(kappa, it)
    lo = lmin * tau * tau
    rec = M.approx("reciprocal", lo, lmax * pmax * pmax + 2.0 * lo, recip_err)
    return CGPlan(float(lmin), float(lmax), float(bnorm), int(it), float(tau), tuple(env), rec, 2.0 * lo)


def cg(ops, a_l, b, lay: VecLay, plan: CGPlan, x0=None):
    """x ~ A^{-1} b for SPD A (L face of `lay`), b (R face), by CG on `plan`'s schedule. The iterates are held as
    r^ = r / (||b|| s_k), p^ = p / (||b|| s_k) and x^ = x lambda_min / ||b|| (unit size); `x0` (R face, true
    values): start from it (r_0 = b - A x0, the envelope then relative to ||r_0|| <= ||b||). Returns x (R face)."""
    lmin, bn, env = plan.lmin, plan.bnorm, plan.env
    xs = lmin / bn                                       # x^ = xs x
    if x0 is None:
        r = ops.mult(b, 1.0 / (bn * env[0])) if bn * env[0] != 1.0 else b
        x = None
    else:
        r = ops.sub(ops.mult(b, 1.0 / (bn * env[0])), matvec(ops, a_l, x0, lay, scale=1.0 / (bn * env[0])))
        x = ops.mult(x0, xs)
    p = r
    for k in range(plan.iters):
        q = matvec(ops, a_l, p, lay)                     # A p^
        pq = dot(ops, p, q, lay)
        rr = norm2(ops, r, lay)
        y = plan.recip(ops, ops.add(pq, plan.shift))     # 1 / (p^.A p^ + 2 lo): never below the band
        alpha = ops.mult(rr, y)
        upd = ops.mult(p, ops.mult(alpha, env[k] * lmin))     # alpha s_k lambda_min p^ (x^ units)
        x = upd if x is None else ops.add(x, upd)
        if k + 1 == plan.iters:
            break
        g = env[k] / env[k + 1]
        rn = ops.sub(r, ops.mult(q, alpha))              # r_{k+1} / (||b|| s_k)
        beta = ops.mult(dot(ops, rn, q, lay), ops.mult(y, -1.0))
        p = ops.mult(ops.add(rn, ops.mult(p, beta)), g)
        r = ops.mult(rn, g)
    return ops.mult(x, 1.0 / xs)


def cg_refined(ops, a_l, b, lay: VecLay, plan: CGPlan, refine: int = 1, plan2: CGPlan | None = None):
    """CG, then `refine` restarts on the recomputed residual (iterative refinement: x += CG(A, b - A x) with the
    residual's bound `plan2.bnorm`), each correcting the previous pass's refresh errors to first order."""
    x = cg(ops, a_l, b, lay, plan)
    for _ in range(refine):
        x = cg(ops, a_l, b, lay, plan2 or plan, x0=x)
    return x


# ── Chebyshev semi-iteration ─────────────────────────────────────────────────────────

def cheb_iters(kappa: float, eps: float = 1e-3) -> int:
    """Steps for the Chebyshev semi-iteration's error bound 2 rho^k / (1 + rho^2k) to reach eps."""
    return cg_iters(kappa, eps)


def chebyshev(ops, a_l, b, lay: VecLay, lmin: float, lmax: float, *, bnorm: float = 1.0, iters: int | None = None,
              eps: float = 1e-3, refresh_every: int = 2, x0=None):
    """x ~ A^{-1} b for SPD A with eigenvalues in [lmin, lmax] (PUBLIC) by the Chebyshev semi-iteration in its
    two-term form (Saad 2003, Alg. 12.1) with the residual recomputed every step: x_{k+1} = x_k + d_k,
    d_{k+1} = rho_{k+1} rho_k d_k + (2 rho_{k+1} / delta) (b - A x_{k+1}). x is held as x^ = x lmin / bnorm (unit
    size); the scalars ride the face move of A x and one product: two levels a step. `refresh_every` = k > 0: x and
    d refreshed by hand every k steps, d at its public envelope 4 rho^k (|d_k| <= |e_k| + |e_k+1|, |e_k| <= 2 rho^k
    |x*|: the update keeps its relative precision as it shrinks), both CONSOLIDATED (`consolidate`: a refresh's
    error is independent per slot and the product would read a mix of an entry's copies). The two-term form, not the
    three-term one: a refresh error in x_k is then an error the later steps correct through the residual, while
    the three-term recurrence carries inconsistent (x_k, x_k-1) pairs (twin, refresh error 1e-3, kappa 100:
    7e-2 three-term, ~5e-3 two-term). 0: the planner places the refreshes (no consolidation). `x0` (R face, true
    values): start from it (iterative refinement: the residual b - A x0 is the right-hand side). Returns x (R face)."""
    th, de = 0.5 * (lmax + lmin), 0.5 * (lmax - lmin)
    sig = th / de
    it = iters or cheb_iters(lmax / lmin, eps)
    rate = cg_rate(lmax / lmin)
    xs = lmin / bnorm
    env = [min(1.0, 4.0 * rate ** k) for k in range(it + 1)]
    rho = 1.0 / sig
    if x0 is None:
        x = None
        d = ops.mult(b, xs / th)                                   # d_0 = b^ / th
    else:
        x = ops.mult(x0, xs)
        d = ops.sub(ops.mult(b, xs / th), matvec(ops, a_l, x, lay, scale=1.0 / th))
    for k in range(it):
        x = d if x is None else ops.add(x, d)
        if k + 1 == it:
            break
        rn = 1.0 / (2.0 * sig - rho)
        hand = refresh_every and (k + 1) % refresh_every == 0
        e = env[k + 1] if hand else 1.0                               # d_{k+1} computed as d / e at a refresh
        if hand:
            x = refresh(ops, x, lay)
        ax = matvec(ops, a_l, x, lay, scale=-2.0 * rn / (de * e))   # -(2 rn / delta) A x^ / e
        d = ops.add(ops.add(ax, ops.mult(b, 2.0 * rn * xs / (de * e))), ops.mult(d, rn * rho / e))
        if hand:
            la.refresh(ops, d, True)
            d = consolidate(ops, d, lay, scale=e)
        rho = rn
    return ops.mult(x, 1.0 / xs)


def refined(ops, solve, a_l, b, lay: VecLay, passes: int = 1, rbound: float = 0.05):
    """Iterative refinement around any of the solvers here: x = solve(b), then `passes` times x += solve(r) with
    r = b - A x recomputed (no refresh between the residual and the sum, so the last pass's refresh errors are in r
    and get solved for) and `solve` told ||r|| <= rbound (PUBLIC: the previous pass's relative residual): each pass
    multiplies the relative error by the solver's own (the CKKS floor of one pass, ~kappa x the refresh error, is
    not a floor any more). `solve(ops, b, bnorm)` -> x (R face)."""
    x = solve(ops, b, 1.0)
    for _ in range(passes):
        r = ops.sub(b, matvec(ops, a_l, x, lay))
        x = ops.add(x, solve(ops, r, rbound))
    return x


# ── preconditioning ─────────────────────────────────────────────────────────────────

def jacobi_scale(ops, a_l, lay: VecLay, dlo: float, dhi: float, tol: float = 1e-4):
    """The symmetric Jacobi scaling of A: (A^ = D^-1/2 A D^-1/2 in the L face, u = sqrt(dlo) D^-1/2 in the R face)
    for diag(A) in the PUBLIC band [dlo, dhi]: the diagonal gathered to the O and R faces (a face gather each), one
    inverse square root each (impl.linalg.inv_sqrt_unit), two products. A^ = dlo^-1 (u_row A u_col): returned scaled
    so that its true value is A^ (the 1 / dlo folded into the row factor)."""
    d_o = la.relayout(ops, a_l, lay.mat, lay.out, gather=("drow",))
    d_r = la.relayout(ops, a_l, lay.mat, lay.vec, gather=("drow",))
    u_o = la.inv_sqrt_unit(ops, d_o, dlo, dhi, tol)                    # sqrt(dlo / d_i) along a1
    u_r = la.inv_sqrt_unit(ops, d_r, dlo, dhi, tol)                    # sqrt(dlo / d_j) along a0
    ah = ops.mult(ops.mult(a_l, ops.mult(u_o, 1.0 / dlo)), u_r)
    return ah, u_r


def pcg_jacobi(ops, a_l, b, lay: VecLay, *, dlo: float, dhi: float, lmin: float, lmax: float, bnorm: float = 1.0,
               method: str = "cg", **kw):
    """A x = b through the Jacobi-scaled system A^ y = D^-1/2 b, x = D^-1/2 y (PCG with M = diag(A), written
    symmetrically). `lmin`, `lmax`: the PUBLIC band of A^'s spectrum (unit diagonal: lmax <= n, often ~2); `bnorm`
    bounds ||D^-1/2 b||. `method` "cg" (plan from cg_plan, **kw to it) or "chebyshev"."""
    ah, u = jacobi_scale(ops, a_l, lay, dlo, dhi)
    bh = ops.mult(b, ops.mult(u, 1.0 / math.sqrt(dlo)))
    if method == "chebyshev":
        y = chebyshev(ops, ah, bh, lay, lmin, lmax, bnorm=bnorm, **kw)
    else:
        y = cg(ops, ah, bh, lay, cg_plan(lmin, lmax, bnorm, **kw))
    return ops.mult(y, ops.mult(u, 1.0 / math.sqrt(dlo)))


def normal_system(ops, a_r, b_l, m: int, n: int, lay: VecLay, rinv_r=None, rinv_l=None, slots: int | None = None):
    """The normal equations of min ||A x - b|| (A m x n in the R face of linalg.gram_cube(n), b^T in its L face) moved
    to `lay` (B = 1): (G in lay's L face, c = A^T b in its R face) with G = A^T A, or, with the client's sketch
    preconditioner R^-1 (`rinv_r`: R^-1 in the R face of full_cube(p2(m), n); its L face on gram_cube is
    `rinv_l`), G = R^-T A^T A R^-1 and c = R^-T A^T b (cond(G) ~ (1 + eps)^2 / (1 - eps)^2 whatever cond(A)). Then
    x = R^-1 y (`precond_back`)."""
    slots = slots or lay.slots
    g = la.gram_cube(n, slots)
    if rinv_r is not None:
        pc = la.full_cube(la.p2(m), n, slots)
        al = la.relayout(ops, a_r, la.R(g, (m, n)), la.L(pc, (m, n)))
        bm = la.mm(ops, al, rinv_r, pc)                                         # B = A R^-1 (O, m x n)
        a_r = la.relayout(ops, bm, la.O(pc, (m, n)), la.R(g, (m, n)))
    G, at = la.gram(ops, a_r, m, n, slots)                                      # B^T B (O on gram_cube), L(B^T)
    c = la.mm(ops, at, la.relayout(ops, b_l, la.L(g, (1, m)), la.R(g, (m, 1)), transpose=True), g)   # B^T b
    G_l = la.relayout(ops, G, la.O(g, (n, n)), lay.mat)
    c_r = la.relayout(ops, c, la.O(g, (n, 1)), lay.vec)
    return G_l, c_r


def precond_back(ops, y, rinv_l, lay: VecLay):
    """x = R^-1 y (R^-1 in lay's L face, from the client)."""
    return matvec(ops, rinv_l, y, lay)


# ── power iteration ─────────────────────────────────────────────────────────────────

def power_iteration(ops, a_l, lay: VecLay, k: int, iters: int, *, lam: list, v0=None, norm_err: float = 1e-3,
                    refresh_iterate: bool = True):
    """The top-k eigenpairs of a symmetric PSD A (L face of `lay`; B matrices batched) by power iteration with
    Hotelling deflation (A <- A - lambda v v^T after each pair). `lam` = [(lo_j, hi_j)]: PUBLIC bands of the j-th
    eigenvalue (the normalization 1 / ||A v|| is impl.math's rsqrt on [lo_j^2 ... hi_j^2] x the start's slack:
    a power step from a unit v has ||A v|| in [lambda_j |<v, v_j>|, lambda_1]). `v0`: the public start (n, or B x n),
    default all-ones. Each step: v <- A v / ||A v|| (matvec, a norm, rsqrt, one product), the iterate refreshed by
    hand when `refresh_iterate`. Returns ([v_j (R face)], [lambda_j broadcast]) with lambda_j = v_j . A v_j."""
    n = lay.n
    if v0 is None:
        v0 = np.ones(n) / math.sqrt(n)
    v0 = np.asarray(v0, dtype=np.float64)
    if v0.ndim == 1 and lay.B > 1:
        v0 = np.broadcast_to(v0, (lay.B, n))
    vecs, vals = [], []
    a = a_l
    for j in range(k):
        lo, hi = lam[j]
        v = None
        for it in range(iters):
            if v is None:
                av = _matvec_pt(ops, a, v0, lay)
            else:
                if refresh_iterate:
                    v = refresh(ops, v, lay)
                av = matvec(ops, a, v, lay)
            s = norm2(ops, av, lay)
            lo2 = (lo * (0.3 if it == 0 else 0.9)) ** 2
            r = M.rsqrt(ops, s, lo2, (hi * 1.05) ** 2, norm_err)
            v = ops.mult(av, r)
        lam_j = dot(ops, v, matvec(ops, a, v, lay), lay)
        vecs.append(v)
        vals.append(lam_j)
        if j + 1 < k:
            a = deflate(ops, a, v, lam_j, lay)
    return vecs, vals


def _matvec_pt(ops, a_l, v0, lay: VecLay):
    """A v0 for a PUBLIC v0 (plaintext product)."""
    q = ops.mult_pt(a_l, lay.encode_vec(v0), kind="mask")
    s = lay.stride
    q = ops.rotate_and_sum(q, s, s * lay.n)
    return la.relayout(ops, q, lay.out, lay.vec)


def deflate(ops, a_l, v, lam, lay: VecLay):
    """A - lambda v v^T in the L face: v along a1 (row) from its R face by a face move, v along a0 as is."""
    vo = la.relayout(ops, v, lay.vec, lay.out)
    return ops.sub(a_l, ops.mult(ops.mult(vo, lam), v))
