"""Matrix functions on the ops protocol, built on impl.linalg's batched cube (B n x n matrices per ciphertext, n^3 B
<= slots; a matrix lives in the O face, products read the L / R faces with wrap):

* `inv_sqrt` (and `sqrtm`): A^{-1/2} of an SPD A by the coupled Newton-Schulz iteration (Higham, Functions of
  Matrices, 2008, eq. 6.35: Y -> A^{1/2}, Z -> A^{-1/2}, T = a I - b Z Y), each step's (a, b) Chebyshev-scaled
  (poly.scaled_newton_steps: the scalar map z -> z (a - b z)^2 that every eigenvalue of Z Y follows, ~6.7x of the
  spread per step against plain Newton's 2.25x); both iterates held at unit size by public per-step envelopes,
  refreshed and consolidated (impl.linalg.consolidate) once per step. The coupled form is stable but not
  self-correcting, so `polish` ends with Newton-Schulz polar steps on W = A^{1/2} Z: Z <- Z (3 I - Z^T A Z) / 2 (Z
  symmetrised first), which drive the whitening residual Z^T A Z - I to zero quadratically (twin, refresh error
  1e-4: 5.6e-3 -> 2.7e-5 in one step). Z - A^{-1/2} itself keeps its first-order error: A^{-1/2} Q is a whitener
  for any orthogonal Q, and the inverse-square-root Newton step is not self-correcting for an error that does not
  commute with A (Z' - Z* = (E - A^{-1/2} E A^{1/2}) / 2, the uncoupled iteration's known instability)
  coupled steps). `whiten`: X C^{-1/2} with C the data's Gram.
* `sign` / `spectral_projector`: the matrix sign of a symmetric A as its polar factor (impl.linalg.polar_ns: the
  accelerated Newton-Schulz polar iteration, self-correcting), and (I + sign(A - mu I)) / 2.
* `expm`: scaling and squaring (Higham, SIAM J. Matrix Anal. Appl. 26 (2005); Moler and Van Loan, SIAM Rev. 45
  (2003)) with a Taylor polynomial evaluated Paterson-Stockmeyer style (Paterson and Stockmeyer, SIAM J. Comput. 2
  (1973)): exp(A) = e^b exp((A - b I) / 2^s)^(2^s) with b >= ||A||_2 public, so every intermediate has norm <= 1;
  (degree, s) chosen for the fewest levels meeting the error bound (Taylor remainder, x 2^s for the squarings).
* `power_iteration` (impl.solvers' vector layout): the top-k eigenpairs by normalized power steps with Hotelling
  deflation, against impl.linalg.block_power (subspace iteration with the polar factor)."""
from __future__ import annotations

import math

import numpy as np

from . import linalg as la
from .poly import scaled_newton_steps
from .solvers import power_iteration, deflate  # noqa: F401  (re-exported)


class Faces:
    """The batched cube of B n x n matrices and its three faces (wrap: products sum n terms)."""

    def __init__(self, n: int, B: int = 1, slots: int = 1 << 15):
        self.n, self.B, self.slots = n, B, slots
        self.cube = la.batched_cube(n, B, slots)
        assert self.cube.d[0] >= n, "B n^3 <= slots"
        self.o = la.O(self.cube, (n, n), B)
        self.l = la.L(self.cube, (n, n), B, wrap=True)
        self.r = la.R(self.cube, (n, n), B, wrap=True)

    def L(self, ops, x, scale=1.0):
        return la.relayout(ops, x, self.o, self.l, scale=scale)

    def R(self, ops, x, scale=1.0):
        return la.relayout(ops, x, self.o, self.r, scale=scale)

    def mm(self, ops, xl, yr):
        return la.mm(ops, xl, yr, self.cube, self.n)

    def eye(self, ops, x, lay, c):
        return la.add_eye(ops, x, lay, c) if c else x

    def encode(self, M):
        return la.encode(M, self.o)

    def decode(self, v):
        X = la.decode(np.real(v), self.o)
        return np.asarray(X).reshape(self.B, self.n, self.n)


# ── inverse square root ─────────────────────────────────────────────────────────────

def coupled_schedule(kappa: float, tol: float = 1e-6, accel: bool = True):
    """[(a, b)] of the coupled steps T = a I - b Z Y for eigenvalues of A / bound in [1 / kappa, 1], and the public
    envelopes (sy_k, sz_k) >= ||Y_k||, ||Z_k|| (the scalar recursion on a log grid of the band)."""
    if accel:
        steps = scaled_newton_steps(kappa, tol)
    else:
        x, steps = 1.0 / kappa, []
        while abs(1.0 - x) > tol and len(steps) < 80:
            x = x * (1.5 - 0.5 * x) ** 2
            steps.append((1.5, 0.5))
    v = np.geomspace(1.0 / kappa, 1.0, 2048)
    y, z = v.copy(), np.ones_like(v)
    env = [(float(y.max()), float(z.max()))]
    for a, b in steps:
        t = a - b * z * y
        y, z = y * t, z * t
        env.append((float(np.abs(y).max()), float(np.abs(z).max())))
    return steps, env


def inv_sqrt(ops, a, F: Faces, *, bound: float, kappa: float, tol: float = 1e-6, accel: bool = True,
             polish: int = 1, refresh_iterate: bool = True, want: str = "inv"):
    """A^{-1/2} (`want` "inv"), A^{1/2} ("sqrt") or both ("both") of SPD A (O face of F) with eigenvalues in
    [bound / kappa, bound] (PUBLIC). Returns true values (O face)."""
    steps, env = coupled_schedule(kappa, tol, accel)
    o, l, r = F.o, F.l, F.r
    # step 1 in closed form from Y0 = A / bound, Z0 = I: T0 = a I - b A / bound
    a0, b0 = steps[0]
    sy1, sz1 = env[1]
    al = F.L(ops, a, 1.0 / bound)                                     # Y0 (L)
    t0r = F.eye(ops, F.R(ops, a, -b0 / (bound * sy1)), r, a0 / sy1)  # T0 / sy1 (R)
    y = F.mm(ops, al, t0r)                                            # Y1 / sy1
    z = F.eye(ops, la.relayout(ops, a, o, o, scale=-b0 / (bound * sz1)), o, a0 / sz1)   # Z1 / sz1 = T0 / sz1
    for k in range(1, len(steps)):
        ak, bk = steps[k]
        sy, sz = env[k]
        sy2, sz2 = env[k + 1]
        if refresh_iterate:
            y, z = la.refresh(ops, y, True, o), la.refresh(ops, z, True, o)
        zy = F.mm(ops, F.L(ops, z), F.R(ops, y))                     # Z Y / (sy sz) (~unit: Z Y -> I)
        c = sy * sz
        tr = F.eye(ops, F.R(ops, zy, -bk * c), r, ak)               # T (R)
        tl = F.eye(ops, F.L(ops, zy, -bk * c), l, ak)               # T (L)
        y = F.mm(ops, F.L(ops, y, sy / sy2), tr)                    # Y T / sy2
        z = F.mm(ops, tl, F.R(ops, z, sz / sz2))                    # T Z / sz2
    sy, sz = env[-1]
    for _ in range(polish if want != "sqrt" else 0):
        # Z <- Z (3 I - Z^T A~ Z) / 2 on A~ = A / bound (Z ~ A~^{-1/2}, unit-size up to sz): polar NS on A~^{1/2} Z
        if refresh_iterate:
            z = la.refresh(ops, z, True, o)
        # Z's faces from (Z + Z^T) / 2: the polish corrects a symmetric error only (its antisymmetric part, half of an
        # i.i.d. refresh error, commutes away), the symmetrisation removes the rest (same level, more rotations)
        zl = ops.add(F.L(ops, z, 0.5), la.relayout(ops, z, o, l, scale=0.5, transpose=True))
        zr = ops.add(F.R(ops, z, 0.5), la.relayout(ops, z, o, r, scale=0.5, transpose=True))
        az = F.mm(ops, al, zr)                                                # A~ Z / sz
        zaz = F.mm(ops, zl, F.R(ops, az, sz * sz))                            # Z A~ Z (~ I)
        m = F.eye(ops, F.R(ops, zaz, -0.5), r, 1.5)
        z = F.mm(ops, zl, m)                                                  # Z' / sz
    zs = sz / math.sqrt(bound)
    ys = sy * math.sqrt(bound)
    if want == "inv":
        return ops.mult(z, zs)
    if want == "sqrt":
        return ops.mult(y, ys)
    return ops.mult(z, zs), ops.mult(y, ys)


def whiten(ops, x_r, m: int, d: int, *, bound: float, kappa: float, slots: int, **kw):
    """X W with W = C^{-1/2}, C = X^T X / m (X m x d in the R face of gram_cube(d), centred by the caller; the
    eigenvalues of C in [bound / kappa, bound] public): the Gram (linalg.gram), inv_sqrt on it (B = 1: the batched
    cube is gram_cube), one product. Returns (O(X W) on full_cube(p2(m), d), O(W) on gram_cube)."""
    F = Faces(d, 1, slots)
    g = la.gram_cube(d, slots)
    G, _ = la.gram(ops, x_r, m, d, slots)
    G = ops.mult(G, 1.0 / m)
    W = inv_sqrt(ops, G, F, bound=bound, kappa=kappa, **kw)
    pc = la.full_cube(la.p2(m), d, slots)
    xl = la.relayout(ops, x_r, la.R(g, (m, d)), la.L(pc, (m, d)))
    wr = la.relayout(ops, W, la.O(g, (d, d)), la.R(pc, (d, d)))
    return la.mm(ops, xl, wr, pc), W


# ── sign / spectral projector ───────────────────────────────────────────────────────

def sign(ops, a, n: int, *, gap: float, bound: float = 1.0, slots: int = 1 << 15, **kw):
    """sign(A) of a symmetric A (O on gram_cube(n): the row-major matrix tiled) with eigenvalues in
    [-bound, -gap] u [gap, bound] (PUBLIC): its polar factor by linalg.polar_ns (A is its own n x n "Y")."""
    return la.polar_ns(ops, a, n, n, bound=bound, kappa=bound / gap, slots=slots, **kw)


def spectral_projector(ops, a, n: int, mu: float, *, gap: float, bound: float = 1.0, slots: int = 1 << 15, **kw):
    """The orthogonal projector onto the eigenvectors of symmetric A with eigenvalue > mu: (I + sign(A - mu I)) / 2,
    no eigenvalue within `gap` of mu (PUBLIC), ||A|| <= bound."""
    lay = la.O(la.gram_cube(n, slots), (n, n))
    s = sign(ops, la.add_eye(ops, a, lay, -mu), n, gap=gap, bound=bound + abs(mu), slots=slots, **kw)
    return la.add_eye(ops, ops.mult(s, 0.5), lay, 0.5)


# ── exponential ─────────────────────────────────────────────────────────────────────

def ps_powers(deg: int):
    """Paterson-Stockmeyer block size: ~sqrt(deg)."""
    return max(1, int(round(math.sqrt(deg + 1))))


def taylor_depth(deg: int) -> int:
    """Levels of `_poly` for a degree-deg polynomial (face moves included)."""
    c = _CountF()
    _poly(c, c.leaf(), [1.0] * (deg + 1), c)
    return c.depth


def expm_plan(bound: float, err: float = 1e-6, max_deg: int = 24, max_s: int = 12):
    """(deg, s) for exp((A - bound I) / 2^s) with ||A|| <= bound: the fewest levels (taylor_depth + 2 s + 1) whose
    error bound 2^s x (Taylor remainder at theta = 2 bound / 2^s) meets `err` (relative to ||exp(A - bound I)||)."""
    best = None
    for s in range(0, max_s + 1):
        th = 2.0 * bound / 2 ** s
        if th > 50.0:
            continue
        for deg in range(1, max_deg + 1):
            rem = th ** (deg + 1) / math.factorial(deg + 1) * math.exp(th)
            if rem * 2 ** s * math.exp(th) <= err:
                lv = taylor_depth(deg) + 2 * s + 1
                if best is None or lv < best[0] or (lv == best[0] and deg < best[1]):
                    best = (lv, deg, s)
                break
    if best is None:
        raise ValueError(f"expm: no (degree, s) meets {err:g} at bound {bound:g}")
    return best[1], best[2]


def expm(ops, a, F: Faces, *, bound: float, err: float = 1e-6, deg: int | None = None, s: int | None = None):
    """exp(A) of B matrices (O face of F), ||A||_2 <= bound (PUBLIC): X = (A - bound I) / 2^s (||X|| <= 2 bound /
    2^s), the Taylor polynomial of degree `deg` (Paterson-Stockmeyer), s squarings, the e^bound folded into the
    last face move. Every intermediate has norm <= 1 (exp of a matrix with numerical range in the left half plane).
    Returns exp(A) (O face)."""
    if deg is None or s is None:
        d0, s0 = expm_plan(bound, err)
        deg, s = deg or d0, s0 if s is None else s
    x = F.eye(ops, la.relayout(ops, a, F.o, F.o, scale=1.0 / 2 ** s), F.o, -bound / 2 ** s)
    coeffs = [1.0 / math.factorial(k) for k in range(deg + 1)]
    e = _poly(ops, x, coeffs, F)
    for j in range(s):
        sc = math.exp(bound) if j == s - 1 else 1.0
        e = F.mm(ops, F.L(ops, e), F.R(ops, e, sc))
    if s == 0:
        e = ops.mult(e, math.exp(bound))
    return e


def _poly(ops, x, coeffs, F):
    """sum c_k X^k (X in O): powers X^1..X^m (m ~ sqrt(deg)) by products, the blocks q_j = sum_i c_{jm+i} X^i
    (scalar products), then Horner in X^m on the blocks."""
    deg = len(coeffs) - 1
    m = ps_powers(deg)
    P = [None, x]
    for i in range(2, m + 1):
        h = i // 2
        P.append(F.mm(ops, F.L(ops, P[h]), F.R(ops, P[i - h])))
    blocks = []
    for j in range(0, deg + 1, m):
        acc = None
        for i in range(m):
            k = j + i
            if k > deg or coeffs[k] == 0.0:
                continue
            if i == 0:
                continue
            t = ops.mult(P[i], float(coeffs[k]))
            acc = t if acc is None else ops.add(acc, t)
        c0 = coeffs[j] if j <= deg else 0.0
        if acc is None:
            acc = F.eye(ops, ops.mult(P[1], 0.0), F.o, c0)
        elif c0:
            acc = F.eye(ops, acc, F.o, c0)
        blocks.append(acc)
    if len(blocks) == 1:
        return blocks[0]
    pm = P[m] if m > 1 else x
    acc = blocks[-1]
    for q in reversed(blocks[:-1]):
        acc = ops.add(F.mm(ops, F.L(ops, acc), F.R(ops, pm)), q)
    return acc


class _V:
    __slots__ = ("d",)

    def __init__(self, d):
        self.d = d


class _CountF:
    """Depth counter standing in for both the ops and the Faces of `_poly` (a face move or product: one level)."""
    o = None

    def __init__(self):
        self.depth = 0

    def leaf(self):
        return _V(0)

    def _o(self, d):
        self.depth = max(self.depth, d)
        return _V(d)

    def mm(self, ops, a, b): return self._o(max(a.d, b.d) + 1)
    def L(self, ops, x, scale=1.0): return self._o(x.d + 1)
    def R(self, ops, x, scale=1.0): return self._o(x.d + 1)
    def mult(self, a, b): return self._o(a.d + 1)
    def add(self, a, b): return _V(max(a.d, b.d))
    def eye(self, ops, x, lay, c): return x
