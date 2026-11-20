"""Encrypted dense linear algebra on the ops protocol: products, inverse / pseudoinverse, least squares, QR, polar
factor, top-k subspaces (block power, randomized range finder).

Layouts. A matrix lives in ONE ciphertext as a face of a cube of d0 x d1 x d2 slots (powers of two, P = d0 d1 d2
dividing the slot count; slot a0 d1 d2 + a1 d2 + a2, the cube tiled over the slots, so a rotation is cyclic mod P).
`Lay(cube, row, col)` puts the row index on axis `row`, the column index on axis `col` and REPLICATES the matrix along
the third axis. Three faces matter (`O`, `L`, `R`):

    O = (row 1, col 2), replicated along 0: the product's output; with d0 = 1 it is the plain row-major matrix
    L = (row 1, col 0), replicated along 2: a product's left operand
    R = (row 0, col 2), replicated along 1: a product's right operand

`mm(L(X), R(Y)) = O(X Y)`: one ciphertext product and log2(d0) rotations summing axis 0, which is cyclic (its stride
times d0 is the period), so the output comes back replicated along axis 0 without a mask: one level. Moving a matrix
between faces (`relayout`) is a slot permutation: a diagonal transform whose diagonals are found by a greedy cover over
the copies replication offers, evaluated baby-step giant-step with the scale folded into its masks: one level, ~2
sqrt(#diagonals) rotations (O -> R and O -> L take d and 2 d - 1 diagonals). It is the replicate-multiply-sum
product of the hypercube / tile-tensor packings (Huang and Zong, J. Supercomputing 2023; HElayers' tile tensors),
against the 2D diagonal method of Jiang, Kim, Lauter and Song (CCS 2018: O(d) rotations per product). The cube must
fill the slots (`full_cube`): see there for what goes wrong when it is tiled.

Nonlinear steps. CKKS has no division, pivoting or data-dependent control flow: every algorithm here is a fixed
polynomial schedule whose iteration counts and scalings come from PUBLIC bounds (`spectral bound`, condition-number
bound) the caller passes. The only scalar nonlinearities are an inverse square root (MGS / Cholesky pivots, poly's
Chebyshev-scaled Newton steps) and a reciprocal (Newton), each on a range the caller bounds.

Every function takes `ops` (ops.FheOps on the device session or the fake, ops.NumpyOps for the plaintext mirror) and
ciphertexts in the layouts named in its docstring; the client side (`encode` / `decode`) is plain numpy.

In a recorded stage. The functions are leaf ops only, so an ImplModel stage records them like any other code and the
planner places their refreshes. Results are the true values: the iterations hold every iterate at unit size by a public
per-step scale (the planner refuses to refresh a value far below the bootstrap's envelope) and fold it back in their
last step. Their own refreshes (`refresh`) run on the real-payload route. The rotation keys a stage needs:
keyset.stage_rotations; keyset.prefer biases the face moves' BSGS splits toward keys the model already holds.
"""
from __future__ import annotations

import dataclasses
import functools
import math

import numpy as np

from . import keyset
from .poly import scaled_newton_steps


# ── geometry ─────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class Cube:
    """d0 x d1 x d2 slots (powers of two), tiled over `slots`."""
    d: tuple
    slots: int = 1 << 15

    def __post_init__(self):
        assert all(x > 0 and x & (x - 1) == 0 for x in self.d), self.d
        assert self.slots % self.P == 0, (self.d, self.slots)

    @property
    def P(self):
        return self.d[0] * self.d[1] * self.d[2]

    @property
    def strides(self):
        return (self.d[1] * self.d[2], self.d[2], 1)

    def coords(self):
        """(3, P) coordinates of the slots of one tile."""
        p = np.arange(self.P)
        return np.stack([p // self.strides[0], (p // self.strides[1]) % self.d[1], p % self.d[2]])


def full_cube(d1: int, d2: int, slots: int = 1 << 15) -> Cube:
    """The (slots / (d1 d2), d1, d2) cube: one tile fills the ciphertext. Products on it must contract over an axis
    that is the WHOLE rotation cycle: with several tiles the sum over axis 0 crosses into the next tile, whose copy of
    every operand carries different noise, so an output's copies mix different "universes" of the computation and
    their differences, which no iteration corrects, compound (measured x4-8 per Newton-Schulz step)."""
    return Cube((slots // (d1 * d2), d1, d2), slots)


def p2(x: int) -> int:
    """The next power of two >= x."""
    return 1 << max(int(x) - 1, 0).bit_length()


@dataclasses.dataclass(frozen=True)
class Lay:
    """A rows x cols matrix on the cube's (row, col) axes, replicated along the third (zero-padded to the axes).
    `batch` = B: B independent matrices, axis 1 split as (problem, index) blocks of d1 / B (entry b R C + i C + j).
    `wrap`: an index on axis 0 is taken mod the matrix's extent there (the matrix repeats down axis 0), so a product's
    sum over axis 0 needs only a window of that extent (log2 of it rotations, not log2 d0)."""
    cube: Cube
    row: int
    col: int
    shape: tuple
    batch: int = 1
    wrap: bool = False

    def __post_init__(self):
        assert self.row != self.col and {self.row, self.col} <= {0, 1, 2}
        d = self.dims
        assert self.shape[0] <= d[self.row] and self.shape[1] <= d[self.col], (self, self.cube)
        assert self.cube.d[1] % self.batch == 0

    @property
    def dims(self):
        d = list(self.cube.d)
        d[1] //= self.batch
        return d

    @property
    def rep(self):
        return 3 - self.row - self.col

    def entries(self):
        """(P,) the flat entry index b R C + i C + j each slot of a tile holds, -1 for padding."""
        c = self.cube.coords()
        blk = self.cube.d[1] // self.batch
        b = c[1] // blk
        c = np.stack([c[0], c[1] % blk, c[2]])
        i, j = c[self.row], c[self.col]
        if self.wrap:
            if self.row == 0:
                i = i % self.shape[0]
            if self.col == 0:
                j = j % self.shape[1]
        ok = (i < self.shape[0]) & (j < self.shape[1])
        return np.where(ok, (b * self.shape[0] + i) * self.shape[1] + j, -1)

    def __str__(self):
        return (f"{'x'.join(map(str, self.cube.d))}:{self.row}{self.col}:{self.shape[0]}x{self.shape[1]}"
                f"{f'/B{self.batch}' if self.batch > 1 else ''}{'w' if self.wrap else ''}")


def O(cube, shape, batch=1): return Lay(cube, 1, 2, tuple(shape), batch)   # noqa: E741,E302
def L(cube, shape, batch=1, wrap=False): return Lay(cube, 1, 0, tuple(shape), batch, wrap)   # noqa: E302
def R(cube, shape, batch=1, wrap=False): return Lay(cube, 0, 2, tuple(shape), batch, wrap)   # noqa: E302


def batched_cube(n: int, B: int, slots: int = 1 << 15) -> Cube:
    """(slots / (B n^2), B n, n): B n x n problems side by side on axis 1 (B <= slots / n^3 keeps axis 0 >= n)."""
    return Cube((slots // (B * n * n), B * n, n), slots)


def flat(rows: int, cols: int, slots: int = 1 << 15) -> Lay:
    """The plain row-major layout of a rows x cols matrix (O on a d0 = 1 cube, tiled over the slots)."""
    return O(Cube((1, p2(rows), p2(cols)), slots), (rows, cols))


def encode(M, lay: Lay) -> np.ndarray:
    """The slot vector of M (rows x cols, or B x rows x cols for a batched layout; a 2D M is shared by every
    problem) in `lay` (client side)."""
    M = np.asarray(M, dtype=np.float64)
    if M.ndim == 2:
        M = np.broadcast_to(M, (lay.batch,) + M.shape)
    assert M.shape == (lay.batch,) + tuple(lay.shape), (M.shape, lay.shape, lay.batch)
    e = lay.entries()
    v = np.where(e >= 0, M.reshape(-1)[np.maximum(e, 0)], 0.0)
    return np.tile(v, lay.cube.slots // lay.cube.P)


def decode(vec, lay: Lay) -> np.ndarray:
    """The matrix in `lay` read from a decrypted slot vector (the first copy of each entry); B x rows x cols when
    batched."""
    v = np.real(np.asarray(vec))[:lay.cube.P]
    e = lay.entries()
    M = np.zeros(lay.batch * lay.shape[0] * lay.shape[1])
    first = np.unique(e[e >= 0], return_index=True)
    M[first[0]] = v[np.nonzero(e >= 0)[0][first[1]]]
    return M.reshape((lay.batch,) + tuple(lay.shape)) if lay.batch > 1 else M.reshape(lay.shape)


# ── diagonal transforms ──────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Transform:
    """y[p] = sum_r diag_r[p] x[p + r] as baby-step giant-step: babies j g, giants G g (signed slot units)."""
    babies: list
    giants: dict           # giant step -> [(baby step, mask)]
    n_diag: int

    def rotations(self):
        return sorted({b for b in self.babies if b} | {G for G in self.giants if G})


def _signed(r, P):
    r = np.asarray(r) % P
    return np.where(r > P // 2, r - P, r)


def _bsgs(diags: dict, slots: int, pref: frozenset = frozenset()) -> Transform:
    """The baby step b (in units of the offsets' gcd) with the fewest rotations; among equal counts, the one needing
    the fewest keys outside `pref` (keyset.prefer: the keys the model already holds)."""
    rs = sorted(diags)
    nz = [abs(r) for r in rs if r]
    g = functools.reduce(math.gcd, nz, 0) or 1
    q = [r // g for r in rs]
    lo, hi = min(q), max(q)
    best = None
    for b in range(1, hi - lo + 2):
        bab = {x % b for x in q}
        gia = {x - x % b for x in q}
        cost = len(bab - {0}) + len(gia - {0})
        if pref:
            new = {k * g % slots for k in (bab | gia) if k} - {k % slots for k in pref}
            cost += keyset.COST_NEW_KEY * len(new)
        if best is None or cost < best[0]:
            best = (cost, b)
    b = best[1]
    giants = {}
    for r, x in zip(rs, q):
        j = x % b
        G = (x - j) * g
        giants.setdefault(G, []).append((j * g, np.roll(diags[r], G)))
    babies = sorted({j for lst in giants.values() for j, _m in lst})
    return Transform(babies, giants, len(rs))


def _want(gather, i, j, src_shape, dst_shape):
    """The src entry index each dst entry (i, j) reads under `gather`: ("id",), ("T",) (src (j, i)), ("col", c) (src
    (i, c): column c broadcast along the row), ("row", r) (src (r, j): row r broadcast down the column)."""
    kind = gather[0]
    sr, sc = src_shape
    if kind == "id":
        assert tuple(src_shape) == tuple(dst_shape), (src_shape, dst_shape)
        return i * sc + j
    if kind == "T":
        assert (sc, sr) == tuple(dst_shape), (src_shape, dst_shape)
        return j * sc + i
    if kind == "col":
        return i * sc + gather[1]
    if kind == "row":
        return gather[1] * sc + j
    if kind == "drow":                  # the diagonal entry of the row, along the row
        return i * sc + i
    if kind == "dcol":                  # the diagonal entry of the column, down the column
        return j * sc + j
    raise ValueError(gather)


def _plan(src: Lay, dst: Lay, gather: tuple, scale: float, sel: tuple | None) -> Transform:
    """The transform taking `src` to `dst` (entries mapped by `gather`, see _want), scaled. `sel`: (row range, col
    range) of the dst entries kept (the others zero). Its BSGS split prefers the keys of keyset.prefer."""
    return _plan_keys(src, dst, gather, scale, sel, keyset.preferred())


@functools.lru_cache(maxsize=None)
def _plan_keys(src: Lay, dst: Lay, gather: tuple, scale: float, sel: tuple | None, pref: frozenset) -> Transform:
    S = src.cube.slots
    assert dst.cube.slots == S
    Pm = max(src.cube.P, dst.cube.P)
    se = np.tile(src.entries(), Pm // src.cube.P)
    de = np.tile(dst.entries(), Pm // dst.cube.P)
    assert src.batch == dst.batch, (src, dst)
    rows, cols = dst.shape
    bi, j = np.divmod(np.maximum(de, 0), cols)
    bb, i = np.divmod(bi, rows)
    want = _want(gather, i, j, src.shape, dst.shape) + bb * (src.shape[0] * src.shape[1])
    ok = de >= 0
    if sel is not None:
        (r0, r1), (c0, c1) = sel
        ok &= (i >= r0) & (i < r1) & (j >= c0) & (j < c1)
    ps = np.nonzero(ok)[0]
    want = want[ps]
    # every copy of each src entry: (n_entries, copies)
    order = np.argsort(se, kind="stable")
    sev = se[order]
    first = np.searchsorted(sev, 0)
    copies = order[first:]
    n_ent = src.batch * src.shape[0] * src.shape[1]
    c = len(copies) // n_ent
    pos = copies.reshape(n_ent, c)
    cand = (pos[want] - ps[:, None]) % Pm                       # (len(ps), c) shifts that serve p
    left = np.ones(len(ps), dtype=bool)
    chosen = {}
    while left.any():
        cnt = np.bincount(cand[left].ravel(), minlength=Pm)
        r = int(np.argmax(cnt))
        hit = left & (cand == r).any(axis=1)
        chosen[r] = ps[hit]
        left &= ~hit
    diags = {}
    for r, at in chosen.items():
        d = np.zeros(Pm)
        d[at] = scale
        diags[int(_signed(r, Pm))] = np.tile(d, S // Pm)
    return _bsgs(diags, S, pref)


def apply(ops, x, t: Transform):
    """One diagonal transform: one level (its masks), len(babies) hoisted + len(giants) rotations."""
    steps = [b for b in t.babies if b]
    rots = dict(zip(steps, ops.rotate_many(x, steps))) if steps else {}
    rots[0] = x
    acc = None
    for G in sorted(t.giants, key=abs):
        part = None
        for j, m in t.giants[G]:
            term = ops.mult_pt(rots[j], m, kind="mask")
            if part is None:
                part = term
            else:
                ops.inplace_add(part, term)
        if G:
            part = ops.rotate(part, G)
        if acc is None:
            acc = part
        else:
            ops.inplace_add(acc, part)
    return acc


def _gather(transpose, gather):
    return tuple(gather) if gather is not None else (("T",) if transpose else ("id",))


def relayout(ops, x, src: Lay, dst: Lay, scale: float = 1.0, transpose: bool = False, sel=None, gather=None):
    """x (in `src`) moved to `dst` (transposed, or an entry gather, see _want), times `scale`; `sel` = ((r0, r1),
    (c0, c1)) keeps a sub-block (zeros elsewhere). One level."""
    sel = None if sel is None else (tuple(sel[0]), tuple(sel[1]))
    return apply(ops, x, _plan(src, dst, _gather(transpose, gather), float(scale), sel))


def relayout_plan(src: Lay, dst: Lay, scale: float = 1.0, transpose: bool = False, sel=None, gather=None):
    sel = None if sel is None else (tuple(sel[0]), tuple(sel[1]))
    return _plan(src, dst, _gather(transpose, gather), float(scale), sel)


# ── products and elementwise helpers ─────────────────────────────────────────────────

def mm(ops, xl, xr, cube: Cube, terms: int | None = None):
    """O(X Y) from L(X) and R(Y) on `cube` (contraction over axis 0): one product, log2 d0 rotations, one level.
    `terms` = k: the operands repeat down axis 0 with period k (`wrap` faces of an inner dimension k), so a window of
    k sums everything at every copy: log2 k rotations."""
    z = ops.mult(xl, xr)
    s = cube.strides[0]
    t = cube.d[0] if terms is None else int(terms)
    return ops.rotate_and_sum(z, s, s * t) if t > 1 else z


def mm_pt(ops, xl, Yr, cube: Cube):
    """O(X Y) from L(X) and a PUBLIC Y (its R slot vector): a plaintext product, log2 d0 rotations, one level."""
    z = ops.mult_pt(xl, Yr, kind="mask")
    s = cube.strides[0]
    return ops.rotate_and_sum(z, s, s * cube.d[0]) if cube.d[0] > 1 else z


def mm_steps(cube: Cube):
    s = cube.strides[0]
    out, g = [], s
    while g < s * cube.d[0]:
        out.append(g)
        g *= 2
    return out


def add_eye(ops, x, lay: Lay, c: float):
    """x + c I (in `lay`, square)."""
    return ops.add_pt(x, encode(c * np.eye(lay.shape[0], lay.shape[1]), lay), kind="mask")


def lincomb(ops, terms):
    """sum c_i x_i (c_i public): scalar products + adds (one level when a c_i is not 1)."""
    acc = None
    for x, c in terms:
        t = x if c == 1.0 else ops.mult(x, float(c))
        acc = t if acc is None else ops.add(acc, t)
    return acc


def broadcast_sum(ops, x, P: int):
    """Every slot = the sum of one tile (period P, power of two): log2 P rotations, no level."""
    return ops.rotate_and_sum(x, 1, P)


def _sum_steps(start, stop):
    out, g = [], abs(start)
    while g < stop:
        out.append(g if start > 0 else -g)
        g *= 2
    return out


# ── scalar nonlinearities on public ranges ───────────────────────────────────────────

def inv_sqrt(ops, s, lo: float, hi: float, tol: float = 1e-4, out_scale: float = 1.0):
    """out_scale / sqrt(s) for s in [lo, hi] (every slot): x = s / hi in [lo / hi, 1], poly's Chebyshev-scaled Newton
    steps from the seed 1 (`scaled_newton_steps`, each step ~6.7x more range), the 1 / sqrt(hi) and `out_scale`
    folded into the last step. Depth 1 + 2 (steps - 1) (+1 for the x = s / hi product)."""
    steps = scaled_newton_steps(hi / lo, tol)
    from .poly import inv_sqrt_newton_scaled
    x = ops.mult(s, 1.0 / hi)
    return inv_sqrt_newton_scaled(ops, x, steps, scale=out_scale / math.sqrt(hi))


def inv_sqrt_depth(lo, hi, tol=1e-4):
    return 2 * len(scaled_newton_steps(hi / lo, tol))


def recip(ops, s, lo: float, hi: float, iters: int | None = None, out_scale: float = 1.0):
    """out_scale / s for s in [lo, hi]: Newton y <- y (2 - s y) from the affine seed y0 = 4 (lo + hi - s) / (lo + hi)^2
    (s y0 in [4 lo hi / (lo + hi)^2, 1], touching 1 at the band's middle); 2 levels per iteration + 1 for the seed.
    `iters` default: enough for 2^-20."""
    if iters is None:
        e, iters = 1.0 - 4.0 * lo * hi / (lo + hi) ** 2, 0
        while e > 2.0 ** -20 and iters < 40:
            e, iters = e * e, iters + 1
    w = 4.0 / (lo + hi) ** 2
    y = ops.add(ops.mult(s, -w), w * (lo + hi))
    for k in range(iters):
        c = out_scale if k == iters - 1 else 1.0
        y = ops.mult(y, ops.add(ops.mult(ops.mult(s, y), -c), 2.0 * c))
    if iters == 0 and out_scale != 1.0:
        y = ops.mult(y, out_scale)
    return y


# ── inverse / pseudoinverse (Newton-Schulz) ──────────────────────────────────────────

def ns_iters(kappa: float, eps: float = 1e-8, sq: bool = True) -> int:
    """Newton-Schulz steps from X0 = A^T / ||A||^2 (`sq`: the error 1 - (sigma / sigma_max)^2) or, for SPD A,
    X0 = I / lambda_max (error 1 - lambda / lambda_max), to a residual eps."""
    e = 1.0 - (1.0 / kappa) ** (2 if sq else 1)
    k = 0
    while e > eps and k < 200:
        e, k = e * e, k + 1
    return k


@dataclasses.dataclass
class InvState:
    """X in O, L and R faces (Z = X / s held, s = the public scale of the inverse)."""
    xo: object
    xl: object
    xr: object


def faces(ops, a, src: Lay, cube: Cube, alpha: float = 1.0, beta: float = 0.0, transpose: bool = False,
          which: str = "OLR"):
    """alpha A (or alpha A^T) + beta I in the O / L / R faces of `cube` from A in `src`: one level each."""
    n = src.shape[1] if transpose else src.shape[0]
    m = src.shape[0] if transpose else src.shape[1]
    out = []
    for w in which:
        dst = {"O": O, "L": L, "R": R}[w](cube, (n, m))
        x = relayout(ops, a, src, dst, scale=alpha, transpose=transpose)
        out.append(add_eye(ops, x, dst, beta) if beta else x)
    return out


def consolidate(ops, x, lay: Lay):
    """x with every replicated copy of an entry set to ONE copy (the rep-axis index 0): mask (one level) + a
    replicate ladder (log2 d_rep rotations). A bootstrap's error is independent per slot, so after a refresh the
    copies of an entry differ by it; faces moved from different copies then carry different errors, and an iteration
    whose first-order error cancels (Newton-Schulz: Z_L (2 I - A Z_R) at convergence) only cancels it when L(Z) and
    R(Z) carry the SAME error."""
    c = lay.cube
    mask = np.tile((c.coords()[lay.rep] == 0).astype(np.float64), c.slots // c.P)
    st = c.strides[lay.rep]
    return ops.rotate_and_sum(ops.mult_pt(x, mask, kind="mask"), -st, st * c.d[lay.rep])


def refresh(ops, x, on: bool, lay: Lay | None = None, real: bool = True):
    """A deliberate bootstrap of x (in place) when `on`, then (`lay` given) `consolidate`: a self-correcting
    iteration wants its refreshes on the iterate, whose error the next step squares away, not inside the step (the
    planner places refreshes by level alone, so some land on intermediates whose error goes straight into the result).
    `real`: the payload is a real matrix: `bootstrap_real` (the real-payload route when the session runs with
    bts.real, else the ordinary refresh), whatever its pack tag says (a complex plaintext upstream, e.g. a Fourier
    transform's twiddles, leaves the tag complex)."""
    if on:
        (ops.bootstrap_real if real and hasattr(ops, "bootstrap_real") else ops.bootstrap)(x)
        if lay is not None:
            x = consolidate(ops, x, lay)
    return x


def ns_scales(lo: float, hi: float, denom: float, iters: int, spd: bool) -> list:
    """The public envelope s_k >= ||X_k|| of the Newton-Schulz iterates from X0 = A^T / denom (singular values of A
    in [lo, hi]) or, `spd`, X0 = I / denom (eigenvalues in [lo, hi]): the iterate's values are
    g_k(x) = (1 - (1 - x)^(2^k)) / v (x = v^2 / denom, resp. v / denom) on a log grid of v. The iterate is held as
    Z_k = X_k / s_k: unit size at EVERY step (X grows from 1 / denom to 1 / lo over the iteration; one fixed scale
    leaves the early iterates far below the bootstrap envelope, where the planner refuses to refresh them)."""
    v = np.geomspace(lo, hi, 2048)
    x = v / denom if spd else v * v / denom
    out, e = [], np.ones_like(x)
    for k in range(iters + 1):
        g = (1.0 - (1.0 - x) ** (2 ** k)) / v if k else x / v
        out.append(float(g.max()))
    return out


def inverse(ops, a, n: int, *, bound: float = 1.0, kappa: float = 100.0, spd: bool = False, iters: int | None = None,
            method: str = "ns", polish: int = 1, slots: int | None = None, src: Lay | None = None,
            refresh_iterate: bool = True, frob: tuple | None = None, keep_scale: bool = False):
    """A^{-1} (n x n, n a power of two with n^3 <= slots) from A (default: O on the full cube, i.e. the row-major
    matrix tiled), Newton-Schulz on PUBLIC bounds ||A||_2 <= `bound`, cond(A) <= `kappa`. Returns O(A^{-1}): every
    iterate is held at unit size by a public per-step scale (ns_scales) and the last step folds it back, so the
    planner sees refreshable magnitudes and the caller sees none of it (`keep_scale`: (O(Z), s) with A^{-1} = s Z).

    "ns": X <- X (2 I - A X) (Schulz 1933) from X0 = A^T / bound^2 (Ben-Israel and Cohen 1966: A X0 symmetric positive
    definite, so the residual I - A X_k = (I - A X0)^(2^k) and the step count is log2 of log(eps) / log(1 - 1/kappa^2));
    spd=True starts at X0 = I / bound (residual 1 - lambda / bound: half the steps). Self-correcting: an error in the
    iterate is squared by the next step, so with `refresh_iterate` every step opens with a deliberate refresh of the
    iterate, `consolidate`d (one bootstrap per step, the step's levels inside one refresh window) and the result
    carries only the last step's own rounding, not a refresh's. Depth 4 per step (two products, two face moves).
    "gs": the product form (Goldschmidt): E = I - A X0, X <- X (I + E), E <- E^2. Depth 2 per step but NOT
    self-correcting (E is squared, not recomputed, so its noise compounds); `polish` NS steps from the result after.
    `frob` = (lo, hi), a public band of ||A||_F^2: the start X0 = A^T / ||A||_F^2 with the reciprocal encrypted
    (Newton on [lo, hi]), for a loose public norm bound; `kappa` is then sigma_max / sigma_min."""
    slots = slots or ops_slots(ops)
    cube = full_cube(n, n, slots)
    assert cube.d[0] >= n, "n^3 <= slots"
    src = src or O(cube, (n, n))
    lo, ll, lr = O(cube, (n, n)), L(cube, (n, n)), R(cube, (n, n))
    if frob is not None:
        flo, fhi = frob
        assert not spd
        smax = math.sqrt(fhi)
        smin = smax / kappa
        denom = flo                           # the upper envelope (the real ||A||_F^2 >= flo only shrinks X)
        if iters is None:
            iters = ns_iters(kappa * math.sqrt(n))
    else:
        smax, smin, denom = bound, bound / kappa, (bound if spd else bound * bound)
        if iters is None:
            iters = ns_iters(kappa, sq=not spd)
    total = iters + (polish if method == "gs" else 0)
    sc = ns_scales(smin, smax, denom, total, spd)
    (al,) = faces(ops, a, src, cube, which="L")
    if frob is not None:      # X0 = A^T / ||A||_F^2 (encrypted): e0 <= 1 - 1 / (n kappa^2) whatever the public bound
        f = ops.rotate_and_sum(ops.square(a), 1, slots)                       # (slots / P_src) ||A||_F^2
        rf = recip(ops, ops.mult(f, src.cube.P / slots), flo, fhi, out_scale=flo)   # flo / ||A||_F^2 in (0, 1]
        zo, zl, zr = (ops.mult(x, rf) for x in faces(ops, a, src, cube, alpha=1.0 / (flo * sc[0]), transpose=True))
        k0 = 0
    elif spd:   # Z1 = (2 X0 - X0 A X0) / s_1 with X0 = I / bound: closed form, one level
        zo, zl, zr = faces(ops, a, src, cube, alpha=-1.0 / (bound * bound * sc[1]), beta=2.0 / (bound * sc[1]))
        k0 = 1
    else:
        zo, zl, zr = faces(ops, a, src, cube, alpha=1.0 / (denom * sc[0]), transpose=True)
        k0 = 0
    if method == "ns":
        for k in range(k0, iters):
            if k > k0:
                zo = refresh(ops, zo, refresh_iterate, lo)
                zl, zr = relayout(ops, zo, lo, ll), relayout(ops, zo, lo, lr, scale=sc[k])
            else:
                zr = relayout(ops, zo, lo, lr, scale=sc[k])
            last = k == iters - 1 and not keep_scale
            zo = _ns_step(ops, al, zl, zr, cube, sc[k] if last else sc[k] / sc[k + 1])
        if keep_scale:
            return zo, sc[iters]
        return zo if iters > k0 else ops.mult(zo, sc[iters])     # no step to fold the scale into
    if method != "gs":
        raise ValueError(method)
    # E = I - A X0; X <- X (I + E) with Z_k = X_k / s_k, E <- E^2
    t = mm(ops, al, relayout(ops, zo, lo, lr, scale=sc[k0]), cube)  # A X (O)
    el, er = (add_eye(ops, relayout(ops, t, lo, d, scale=-1.0), d, 1.0) for d in (ll, lr))
    for k in range(k0, iters):
        r = sc[k] if (k == total - 1 and not keep_scale) else sc[k] / sc[k + 1]
        zo = mm(ops, zl, add_eye(ops, ops.mult(er, r), lr, r), cube)   # Z (I + E) s_k / s_k+1
        zl = relayout(ops, zo, lo, ll)
        if k + 1 < iters:
            eo = mm(ops, el, er, cube)                              # E^2
            el, er = relayout(ops, eo, lo, ll), relayout(ops, eo, lo, lr)
    for j in range(iters, total):
        zo = refresh(ops, zo, refresh_iterate, lo)
        zl, zr = relayout(ops, zo, lo, ll), relayout(ops, zo, lo, lr, scale=sc[j])
        zo = _ns_step(ops, al, zl, zr, cube, sc[j] if (j == total - 1 and not keep_scale) else sc[j] / sc[j + 1])
    return (zo, sc[total]) if keep_scale else zo


def _ns_step(ops, al, zl, zr, cube, ratio: float = 1.0):
    """O(Z_k (2 I - A X_k) s_k / s_k+1) from L(A), L(Z_k) and R(X_k) (= s_k Z_k: the scale rides the face move), as
    ONE product: an iterate that is a product output has every replicated copy of an entry equal (the cyclic sum gives
    each copy the same terms); 2 Z - Z A Z would add the stored iterate to the product, a second path for the
    iterate's noise that the face moves then read inconsistently. A X_k ~ I stays unit size whatever s_k."""
    n = cube.d[1]
    lo, lr = O(cube, (n, n)), R(cube, (n, n))
    t = mm(ops, al, zr, cube)                                           # A X (O)
    m = add_eye(ops, relayout(ops, t, lo, lr, scale=-ratio), lr, 2.0 * ratio)   # (2 I - A X) s_k / s_k+1 (R)
    return mm(ops, zl, m, cube)


def gram_cube(n: int, slots: int = 1 << 15) -> Cube:
    """The full cube of n x n products: tall operands up to slots / n^2 rows ride its axis 0."""
    return full_cube(n, n, slots)


def gram(ops, a_r, m: int, n: int, slots: int):
    """O(A^T A) (n x n, gram_cube) from R(A) (A m x n on gram_cube, m <= slots / n^2), plus L(A^T) for reuse."""
    cube = gram_cube(n, slots)
    at = relayout(ops, a_r, R(cube, (m, n)), L(cube, (n, m)), transpose=True)
    return mm(ops, at, a_r, cube), at


# ── least squares ────────────────────────────────────────────────────────────────────

def lstsq(ops, a_r, b_r, m: int, n: int, *, bound: float = 1.0, kappa: float = 100.0, ridge: float = 0.0,
          slots: int | None = None, method: str = "ns", refresh_iterate: bool = True):
    """argmin ||A x - b||^2 + ridge ||x||^2 by the normal equations: G = A^T A + ridge I (one product), c = A^T b,
    X ~ G^{-1} by `inverse` (SPD start, cond(G) = cond(A)^2 bounds the steps) stopped one step short, and the last
    Newton-Schulz step applied to the VECTOR: x = (2 I - X G) X c = 2 u - X (G u), u = X c. An error E in X enters x as
    E (I - G X*) c = 0 to first order, so the refresh error of X (relative eps, i.e. cond(G) eps in X c) is squared
    away; X_final c would carry it at cond(G) eps (measured: O(1) relative error at cond(A) = 100). A (m x n) and b
    (m x 1) come in the R face of gram_cube (m <= slots / n^2; more rows: sum the Grams of row blocks). Returns O(x)
    (n x 1)."""
    slots = slots or ops_slots(ops)
    cube = gram_cube(n, slots)
    lo, ll = O(cube, (n, n)), L(cube, (n, n))
    lo1, lr1 = O(cube, (n, 1)), R(cube, (n, 1))
    g, at = gram(ops, a_r, m, n, slots)
    if ridge:
        g = add_eye(ops, g, lo, ridge)
    c = mm(ops, at, b_r, cube)                                              # A^T b (O, n x 1)
    hi = bound * bound + ridge
    kg = hi / (bound * bound / (kappa * kappa) + ridge)
    it = ns_iters(kg, sq=False)
    z, s = inverse(ops, g, n, bound=hi, kappa=kg, spd=True, slots=slots, method=method, iters=it - 1,
                   refresh_iterate=refresh_iterate, keep_scale=True)
    z = refresh(ops, z, refresh_iterate)
    zl = relayout(ops, z, lo, ll)                                           # X / s (both products read this face)
    gl = relayout(ops, g, lo, ll)
    u = mm(ops, zl, relayout(ops, c, lo1, lr1, scale=s), cube)              # u = X c
    gu = mm(ops, gl, relayout(ops, u, lo1, lr1), cube)                      # G u
    w = mm(ops, zl, relayout(ops, gu, lo1, lr1, scale=s), cube)             # X G u
    return ops.sub(ops.add(u, u), w)


def pinv_lstsq(ops, a_r, b_l, m: int, n: int, *, bound: float = 1.0, kappa: float = 100.0, slots: int | None = None,
               iters: int | None = None, refresh_iterate: bool = True, accel: bool = True):
    """argmin ||A x - b|| through the pseudoinverse itself, never forming A^T A: Newton-Schulz for A^+ (Ben-Israel and
    Cohen 1966: X <- (2 I - X A) X from X0 = A^T / bound^2), run on Y = X^T (m x n, the flat layout of
    full_cube(m, n)) as Y <- Y (2 I - A^T Y), whose two products both fit the slots: A^T Y on gram_cube (contraction over
    m, A^T's face built once) and Y M on full_cube(m, n). The last step is applied to b: x^T = u^T (2 I - A^T Y),
    u^T = b^T Y, so an error E in X enters x as E (b - A x*) (the LS residual) to first order and as E A E b to second:
    the refresh error is amplified by cond(A) (twice in the second-order term), not by cond(A)^2 as through
    (A^T A)^{-1}. A comes in the R face of gram_cube (m <= slots / n^2), b^T in its L face (1 x m). Returns O(x^T) on
    gram_cube (1 x n)."""
    slots = slots or ops_slots(ops)
    g, c3 = gram_cube(n, slots), full_cube(p2(m), n, slots)
    assert c3.d[0] >= n
    ar_src = R(g, (m, n))
    o3, l3, r3 = O(c3, (m, n)), L(c3, (m, n)), R(c3, (n, n))
    og, rg = O(g, (n, n)), R(g, (m, n))
    alphas, sc = ns_schedule(bound / kappa, bound, bound * bound, False, accel)
    if iters is not None:
        alphas, sc = alphas[:iters], sc[:iters + 1]
    iters = len(alphas)
    atl = relayout(ops, a_r, ar_src, L(g, (n, m)), transpose=True)                 # A^T (fixed)
    z = relayout(ops, a_r, ar_src, o3, scale=1.0 / (bound * bound * sc[0]))        # Y0 / s0 = A / (b^2 s0)
    for k in range(iters - 1):
        if k:
            z = refresh(ops, z, refresh_iterate, o3)
        t = mm(ops, atl, relayout(ops, z, o3, rg, scale=sc[k]), g)                 # A^T Y_k (~ (X A)^T)
        r, ak = sc[k] / sc[k + 1], alphas[k]
        mr = add_eye(ops, relayout(ops, t, og, r3, scale=-ak * ak * r), r3, 2.0 * ak * r)   # alpha (2 I - alpha A^T Y)
        z = mm(ops, relayout(ops, z, o3, l3), mr, c3)
    z = refresh(ops, z, refresh_iterate, o3)
    yr = relayout(ops, z, o3, rg, scale=sc[iters - 1])                             # Y_{K-1}, both products read it
    t = mm(ops, atl, yr, g)                                                         # A^T Y
    u = mm(ops, b_l, yr, g)                                                         # u^T = b^T Y (1 x n)
    ak = alphas[iters - 1]
    mr = add_eye(ops, relayout(ops, t, og, R(g, (n, n)), scale=-ak * ak), R(g, (n, n)), 2.0 * ak)
    ul = relayout(ops, u, O(g, (1, n)), L(g, (1, n)))
    return mm(ops, ul, mr, g)


def precond_lstsq(ops, a_l, rinv_r, rinvt_r, b_l, m: int, n: int, *, bound_b: float = 2.5, kappa_b: float = 6.0,
                  slots: int | None = None):
    """Sketch-and-precondition least squares (Rokhlin and Tygert 2008; Blendenpik, Avron et al. 2010) with the
    preconditioner made by the CLIENT: it sketches its own A (S A, s ~ 4 n rows: an SRHT costs O(m n log s)), takes
    the small QR S A = Q R (O(s n^2)) and encrypts R^{-1} (and R^{-T}) beside A and b. The server forms B = A R^{-1}
    (cond(B) <= (1 + eps) / (1 - eps) ~ 3 for s = 4 n, whatever cond(A)), solves min ||B y - b|| by pinv_lstsq with
    kappa_b (a handful of steps instead of ~2 log2 cond(A)), and returns x = R^{-1} y. The server learns nothing new
    (R^{-1} is encrypted). Faces: A as L on full_cube(m, n), R^{-1} as R on it, R^{-T} as R on gram_cube(n), b^T as
    L on gram_cube(n). Returns O(x^T) on gram_cube (1 x n)."""
    slots = slots or ops_slots(ops)
    g, pc = gram_cube(n, slots), full_cube(p2(m), n, slots)
    bm = mm(ops, a_l, rinv_r, pc)                              # B = A R^{-1}
    br = relayout(ops, bm, O(pc, (m, n)), R(g, (m, n)))
    y = pinv_lstsq(ops, br, b_l, m, n, bound=bound_b, kappa=kappa_b, slots=slots)   # y^T (O, 1 x n)
    yl = relayout(ops, y, O(g, (1, n)), L(g, (1, n)))
    return mm(ops, yl, rinvt_r, g)


def chebyshev_solve(ops, a_r, b_r, m: int, n: int, lmin: float, lmax: float, iters: int, slots: int | None = None):
    """The Chebyshev semi-iteration on the normal equations G x = c (G = A^T A, eigenvalues in the PUBLIC band
    [lmin, lmax]): no inverse and no division, fixed coefficients; error ~ 2 ((sqrt(k) - 1) / (sqrt(k) + 1))^iters
    with k = lmax / lmin = cond(A)^2, i.e. ~cond(A) ln(2 / eps) / 2 steps (each a matrix-vector product and a face
    move, 2 levels) -- against Newton-Schulz's ~2 log2 cond(A). Returns O(x) (n x 1)."""
    slots = slots or ops_slots(ops)
    cube = gram_cube(n, slots)
    g, at = gram(ops, a_r, m, n, slots)
    gl = relayout(ops, g, O(cube, (n, n)), L(cube, (n, n)))
    c = mm(ops, at, b_r, cube)
    lo1, lr1 = O(cube, (n, 1)), R(cube, (n, 1))
    theta, delta = (lmax + lmin) / 2.0, (lmax - lmin) / 2.0
    # Saad, Iterative Methods for Sparse Linear Systems (2003), Alg. 12.1, with the residual recomputed (c - G x:
    # self-correcting) instead of updated
    sigma = theta / delta
    rho = 1.0 / sigma
    x = None
    dk = ops.mult(c, 1.0 / theta)
    for k in range(iters):
        x = dk if x is None else ops.add(x, dk)
        if k + 1 == iters:
            break
        r = ops.sub(c, mm(ops, gl, relayout(ops, x, lo1, lr1), cube))
        rho_new = 1.0 / (2.0 * sigma - rho)
        dk = ops.add(ops.mult(dk, rho_new * rho), ops.mult(r, 2.0 * rho_new / delta))
        rho = rho_new
    return x


# ── scalar inverse square root with bounded magnitudes ──────────────────────────────

def inv_sqrt_unit(ops, s, lo: float, hi: float, tol: float = 1e-5):
    """sqrt(lo / s) in (0, 1] for s in [lo, hi] (every slot): poly.scaled_newton_steps (Chebyshev-scaled Newton, each
    step covering ~6.7x more of the band) on x = s / hi, the iterate held as y sqrt(t) (t = lo / hi) so it stays in
    (0, 1]: the closed first step, then y <- y (a - (b / t) x y^2) with (b / t) x formed from x off the critical path
    (it reaches b / t: a large but shallow value), 3 levels per step on the chain (2 + 3 (steps - 1) in all)."""
    t = lo / hi
    steps = scaled_newton_steps(hi / lo, tol)
    x = ops.mult(s, 1.0 / hi)
    a, b = steps[0]
    y = ops.add(ops.mult(x, -b * math.sqrt(t)), a * math.sqrt(t))
    for a, b in steps[1:]:
        y = ops.mult(y, ops.add(ops.mult(ops.mult(x, -b / t), ops.square(y)), a))
    return y


def inv_sqrt_unit_depth(lo, hi, tol=1e-5):
    return 2 + 2 * (len(scaled_newton_steps(hi / lo, tol)) - 1)


# ── QR ───────────────────────────────────────────────────────────────────────────────

def mgs_qr(ops, a, m: int, n: int, *, kappa: float = 100.0, bound: float = 1.0, slots: int | None = None,
           tol: float = 1e-5):
    """Modified Gram-Schmidt on A (m x n, row-major O on full_cube(m, n), m n <= slots): column j is broadcast
    along the rows (a face gather, 1 level), its squared norm summed down the rows (cyclic: the rows axis is outermost
    in the tile), normalized by inv_sqrt_unit on [1/kappa^2, bound^2] (R_jj >= sigma_min), projected out of the later
    columns at once (q^T A summed down the rows, an outer-product update). n sequential inverse square roots; no
    division, no pivoting. Returns (O(Q), O(R)) with R in rows 0..n-1 of an m x n layout."""
    slots = slots or ops_slots(ops)
    cube = full_cube(p2(m), p2(n), slots)
    lay = O(cube, (m, n))
    d2, d1 = cube.d[2], cube.d[1]
    lo, hi = (bound / kappa) ** 2, bound * bound
    c = 1.0 / math.sqrt(lo)
    Q = Rm = None
    for j in range(n):
        cb = relayout(ops, a, lay, lay, gather=("col", j))                    # a_j along every row
        nrm = ops.rotate_and_sum(ops.square(cb), d2, d2 * d1)                  # ||a_j||^2
        r = inv_sqrt_unit(ops, nrm, lo, hi, tol)                               # sqrt(lo) / ||a_j||
        qt = ops.mult(cb, r)                                                   # q_j sqrt(lo)
        pt = ops.rotate_and_sum(ops.mult(qt, a), d2, d2 * d1)                  # (q_j . a_k) sqrt(lo), every row
        colj = encode(np.outer(np.ones(m), np.eye(n)[j]), lay)
        rowj = encode(np.outer(np.eye(m)[j], (np.arange(n) >= j).astype(float)), lay)
        qj = ops.mult_pt(qt, c * colj, kind="mask")
        rj = ops.mult_pt(pt, c * rowj, kind="mask")
        Q = qj if Q is None else ops.add(Q, qj)
        Rm = rj if Rm is None else ops.add(Rm, rj)
        if j + 1 < n:
            later = encode(np.outer(np.ones(m), (np.arange(n) > j).astype(float)), lay)
            q = ops.mult_pt(qt, c * np.ones(slots), kind="mask")
            pm = ops.mult_pt(pt, c * later, kind="mask")
            a = ops.sub(a, ops.mult(q, pm))
    return Q, Rm


def cholesky_rinv(ops, g, n: int, lo: float, hi: float, slots: int, tol: float = 1e-5):
    """Right-looking Cholesky G = R^T R of O(G) (gram_cube) with the inverse factor carried along: the row
    operations that reduce G applied to I give W = R^{-T}. Step j: the pivot p = G_jj broadcast (mask + sum),
    sqrt(lo / p) by inv_sqrt_unit (pivots in the PUBLIC band [lo, hi]), the column and row of G broadcast (face
    gathers), the Schur update G -= c c^T / p. n sequential inverse square roots. Returns (O(R), O(W))."""
    cube = gram_cube(n, slots)
    lay = O(cube, (n, n))
    d0 = cube.d[0]
    c = 1.0 / math.sqrt(lo)
    I = np.eye(n)
    Rm = W = None
    for j in range(n):
        e = encode(np.outer(I[j], I[j]), lay) / d0
        p = ops.rotate_and_sum(ops.mult_pt(g, e, kind="mask"), 1, slots)        # G_jj everywhere
        rt = inv_sqrt_unit(ops, p, lo, hi, tol)                                  # sqrt(lo / p)
        rb = relayout(ops, g, lay, lay, gather=("row", j))                       # G_jk along every row
        cb = relayout(ops, g, lay, lay, gather=("col", j), sel=((0, n), (0, n)))  # G_ij along every column
        rj = ops.mult(ops.mult_pt(rb, c * encode(np.outer(I[j], np.ones(n)), lay), kind="mask"), rt)  # R_j.
        Rm = rj if Rm is None else ops.add(Rm, rj)
        r2 = ops.square(rt)                                                      # lo / p
        if j + 1 < n:
            g = ops.sub(g, ops.mult(ops.mult_pt(cb, encode(np.ones((n, n)), lay) / lo, kind="mask"),
                                    ops.mult(rb, r2)))
        # W: rows i != j -= (c_i / p) W_j; row j *= 1 / sqrt(p)
        notj = encode(np.outer(1.0 - I[j], np.ones(n)), lay)
        onlyj = encode(np.outer(I[j], np.ones(n)), lay)
        t = ops.sub(ops.mult_pt(rt, c * onlyj, kind="mask"),
                    ops.mult(ops.mult_pt(cb, notj / lo, kind="mask"), r2))       # T = e_j / sqrt(p) - c_i / p
        if W is None:
            W = ops.add_pt(ops.mult_pt(t, encode(np.outer(np.ones(n), I[j]), lay), kind="mask"),
                           encode(np.diag(1.0 - I[j]), lay), kind="mask")
        else:
            wb = relayout(ops, W, lay, lay, gather=("row", j))                   # W_jk along every row
            W = ops.add(ops.mult_pt(W, notj, kind="mask"), ops.mult(wb, t))
    return Rm, W


def cholesky_qr(ops, a_r, a_l, m: int, n: int, *, lo: float, hi: float, shift: float = 0.0,
                slots: int | None = None, tol: float = 1e-5):
    """One CholeskyQR pass (Yamamoto et al. 2015 / Fukaya et al. 2020 with `shift`): G = A^T A (+ shift I), its
    Cholesky R and W = R^{-T}, Q = A W^T. A comes in two faces: R on gram_cube (the Gram) and L on full_cube(m, n)
    (the product Q = A R^{-1}; the client encrypts both). Returns (O(Q) on full_cube(m, n), O(R) on gram_cube,
    O(W))."""
    slots = slots or ops_slots(ops)
    gc, pc = gram_cube(n, slots), full_cube(p2(m), n, slots)
    g, _ = gram(ops, a_r, m, n, slots)
    if shift:
        g = add_eye(ops, g, O(gc, (n, n)), shift)
    Rm, W = cholesky_rinv(ops, g, n, lo + shift, hi + shift, slots, tol)
    wt = relayout(ops, W, O(gc, (n, n)), R(pc, (n, n)), transpose=True)         # R^{-1} = W^T
    q = mm(ops, a_l, wt, pc)
    return q, Rm, W


def qr_faces(ops, q, m: int, n: int, slots: int):
    """R(Q) on gram_cube and L(Q) on full_cube(m, n) from O(Q) on full_cube(m, n): the faces a next CholeskyQR pass
    reads."""
    gc, pc = gram_cube(n, slots), full_cube(p2(m), n, slots)
    src = O(pc, (m, n))
    return relayout(ops, q, src, R(gc, (m, n))), relayout(ops, q, src, L(pc, (m, n)))


def cholesky_qr2(ops, a_r, a_l, m: int, n: int, *, kappa: float, bound: float = 1.0, shift: float = 0.0,
                 passes: int = 2, slots: int | None = None, tol: float = 1e-5):
    """CholeskyQR2 (passes=2) / shifted CholeskyQR3 (shift > 0, passes=3): the first pass's pivots lie in
    [sigma_min^2 + shift, bound^2 + shift]; later passes see Q^T Q ~ I, pivots in a narrow band around 1. R is the
    product of the passes' R factors (one more product each). Returns (O(Q), O(R) on gram_cube)."""
    slots = slots or ops_slots(ops)
    gc = gram_cube(n, slots)
    lo, ll, lr = O(gc, (n, n)), L(gc, (n, n)), R(gc, (n, n))
    q, Rt, _ = cholesky_qr(ops, a_r, a_l, m, n, lo=(bound / kappa) ** 2, hi=bound * bound, shift=shift,
                           slots=slots, tol=tol)
    smin = bound / kappa
    for k in range(1, passes):
        qr_, ql = qr_faces(ops, q, m, n, slots)
        if k == 1 and shift:     # sigma(Q1) = sigma / sqrt(sigma^2 + shift): the shift's own band
            lo_k, hi_k = 0.5 * smin ** 2 / (smin ** 2 + shift), 1.25
        else:
            lo_k, hi_k = 0.5, 1.5
        q, Rk, _ = cholesky_qr(ops, qr_, ql, m, n, lo=lo_k, hi=hi_k, slots=slots, tol=tol)
        Rt = mm(ops, relayout(ops, Rk, lo, ll), relayout(ops, Rt, lo, lr), gc)
    return q, Rt


# ── orthonormalization (polar factor) ───────────────────────────────────────────────

def inv_sqrt_spd(ops, b, k: int, *, bound: float, kappa: float, iters: int | None = None, slots: int | None = None,
                 refresh_iterate: bool = False):
    """B^{-1/2} * sqrt(bound) for SPD B (k x k, O on gram_cube) with eigenvalues in [bound / kappa, bound]: the coupled
    Newton-Schulz iteration (Higham, Functions of Matrices, 2008, eq. 6.35) on B / bound: T = (3 I - Z Y) / 2,
    Y <- Y T, Z <- T Z (all three polynomials in B: commuting and symmetric, so every product order is free). Each
    update is one product (consistent copies). Returns O(Z) (Z -> (B / bound)^{-1/2})."""
    slots = slots or ops_slots(ops)
    cube = gram_cube(k, slots)
    lo, ll, lr = O(cube, (k, k)), L(cube, (k, k)), R(cube, (k, k))
    if iters is None:
        e, iters = 1.0 - 1.0 / kappa, 0
        while e > 1e-7 and iters < 60:      # the linear phase grows small eigenvalues x1.5 then converges quadratically
            x = 1.0 - e
            x = x * (1.5 - 0.5 * x) ** 2
            e, iters = 1.0 - x, iters + 1
    # step 1 in closed form from Y0 = B / bound, Z0 = I: T0 = (3 I - B / bound) / 2
    yl, = faces(ops, b, lo, cube, alpha=1.0 / bound, which="L")
    t0r, = faces(ops, b, lo, cube, alpha=-0.5 / bound, beta=1.5, which="R")
    y = mm(ops, yl, t0r, cube)                                       # Y1 = Y0 T0
    z = faces(ops, b, lo, cube, alpha=-0.5 / bound, beta=1.5, which="O")[0]   # Z1 = T0
    for it in range(1, iters):
        if refresh_iterate:
            y, z = refresh(ops, y, True, lo), refresh(ops, z, True, lo)
        zy = mm(ops, relayout(ops, z, lo, ll), relayout(ops, y, lo, lr), cube)
        tr = add_eye(ops, relayout(ops, zy, lo, lr, scale=-0.5), lr, 1.5)    # T (R)
        tl = add_eye(ops, relayout(ops, zy, lo, ll, scale=-0.5), ll, 1.5)    # T (L)
        y = mm(ops, relayout(ops, y, lo, ll), tr, cube)
        z = mm(ops, tl, relayout(ops, z, lo, lr), cube)
    return z


def trace(ops, x, lay: Lay):
    """tr(X) in every slot (X square in `lay`): a diagonal mask (one level, 1 / copies folded in) + a full sum."""
    c = lay.cube
    copies = c.slots // (c.d[lay.row] * c.d[lay.col])
    return ops.rotate_and_sum(ops.mult_pt(x, encode(np.eye(lay.shape[0]) / copies, lay), kind="mask"), 1, c.slots)


def orthonormalize(ops, y, m: int, k: int, *, bound: float, kappa: float, slots: int | None = None,
                   iters: int | None = None, trace_band: tuple | None = None, refresh_iterate: bool = True,
                   method: str = "gram"):
    """The polar factor Q = Y (Y^T Y)^{-1/2} of Y (m x k, O on full_cube(m, k)) with sigma(Y) in
    [bound / kappa, bound] (PUBLIC): the Gram B on gram_cube(k), its inverse square root by `inv_sqrt_spd` (iterations
    on k x k only), one product. `trace_band` = (lo, hi), a public band of tr(B) = ||Y||_F^2: B is first normalized
    by its (encrypted) trace, so the iterations follow k cond(Y)^2 (`kappa` is then cond(Y)) instead of
    (bound / sigma_min)^2 -- the scale of Y no longer needs a tight public bound, only its conditioning.
    `method` "ns": polar_ns on Y itself instead (self-correcting orthonormality, no cond(Y) amplification).
    Returns O(Q) on full_cube(m, k)."""
    slots = slots or ops_slots(ops)
    if method == "ns":
        return polar_ns(ops, y, m, k, bound=bound, kappa=kappa, iters=iters, slots=slots,
                        refresh_iterate=refresh_iterate)
    gc, pc = gram_cube(k, slots), full_cube(p2(m), k, slots)
    src = O(pc, (m, k))
    yr = relayout(ops, y, src, R(gc, (m, k)))
    b, _ = gram(ops, yr, m, k, slots)
    if trace_band is None:
        z = inv_sqrt_spd(ops, b, k, bound=bound * bound, kappa=kappa * kappa, slots=slots, iters=iters,
                         refresh_iterate=refresh_iterate)
        zr = relayout(ops, z, O(gc, (k, k)), R(pc, (k, k)), scale=1.0 / bound)
    else:
        tlo, thi = trace_band
        r = inv_sqrt_unit(ops, trace(ops, b, O(gc, (k, k))), tlo, thi)                 # sqrt(tlo / t)
        bh = ops.mult(b, ops.square(r))                                                # B tlo / t
        z = inv_sqrt_spd(ops, bh, k, bound=tlo, kappa=k * kappa * kappa, slots=slots, iters=iters,
                         refresh_iterate=refresh_iterate)
        zr = ops.mult(relayout(ops, z, O(gc, (k, k)), R(pc, (k, k)), scale=1.0 / math.sqrt(tlo)), r)
    return mm(ops, relayout(ops, y, src, L(pc, (m, k))), zr, pc)


def polar_ns(ops, y, m: int, k: int, *, bound: float, kappa: float, iters: int | None = None,
             slots: int | None = None, refresh_iterate: bool = True, accel: bool = True):
    """The polar factor of Y (m x k, O on full_cube(m, k), sigma(Y) in [bound / kappa, bound]) by the Newton-Schulz
    polar iteration on Y itself (Higham 2008, eq. 8.20): X <- X (3 I - X^T X) / 2 from X0 = Y / bound. A small
    singular value grows x1.5 per step until ~1, then converges quadratically: ~log_1.5(kappa) + 4 steps. Unlike
    Y (Y^T Y)^{-1/2} (whose inverse square root, of a cond(Y)^2 matrix, scales a refresh's error by cond(Y)), the
    orthonormality corrects itself: X^T X = I + D gives I + O(D^2) after a step, whatever error D a refresh left.
    Each step: X^T X on gram_cube(k) (X moved to its R face: ~m diagonals) and X M on full_cube(m, k); one consolidated
    refresh per step. `accel`: the Chebyshev-scaled steps (a_k, b_k) of poly.scaled_newton_steps on sigma^2 (~6.7x
    of the singular-value spread squared per step against 2.25x): ~half the steps. Returns O(Q) on full_cube(m, k)."""
    slots = slots or ops_slots(ops)
    gc, pc = gram_cube(k, slots), full_cube(p2(m), k, slots)
    src = O(pc, (m, k))
    if accel:       # X <- X (a I - b X^T X): sigma^2 follows z <- z (a - b z)^2, the map of poly.scaled_newton_steps
        steps = scaled_newton_steps(kappa * kappa, 1e-6)
    else:
        x, steps = 1.0 / kappa, []
        while abs(1.0 - x) > 1e-6 and len(steps) < 80:
            x = x * (1.5 - 0.5 * x * x)
            steps.append((1.5, 0.5))
    if iters is not None:
        steps = steps[:iters]
    x = ops.mult(y, 1.0 / bound)
    for it, (a, b_) in enumerate(steps):
        if it:
            x = refresh(ops, x, refresh_iterate, src)
        xr = relayout(ops, x, src, R(gc, (m, k)))
        b, _ = gram(ops, xr, m, k, slots)                                         # X^T X
        mr = add_eye(ops, relayout(ops, b, O(gc, (k, k)), R(pc, (k, k)), scale=-b_), R(pc, (k, k)), a)
        x = mm(ops, relayout(ops, x, src, L(pc, (m, k))), mr, pc)
    return x


# ── top-k subspaces ──────────────────────────────────────────────────────────────────

def covariance(ops, x_rs, N: int, d: int, slots: int | None = None):
    """The centered covariance C = (X - 1 mu^T)^T (X - 1 mu^T) / N of samples X (N x d) given as row blocks in the R
    face of gram_cube(d) (each <= slots / d^2 rows): sum of the blocks' Grams minus s s^T / N (s the column sums,
    a cyclic sum down axis 0 of the R face; s s^T as the product of its row and column broadcasts). Returns O(C)."""
    slots = slots or ops_slots(ops)
    cube = gram_cube(d, slots)
    G = S_ = None
    for xr in x_rs:
        rows = cube.d[0]
        g, _ = gram(ops, xr, rows, d, slots)
        cs = ops.rotate_and_sum(xr, cube.strides[0], slots)                     # s_k on every (a0, a1, k)
        G = g if G is None else ops.add(G, g)
        S_ = cs if S_ is None else ops.add(S_, cs)
    lay = O(cube, (d, d))
    col = relayout(ops, S_, lay, lay, transpose=True, scale=1.0 / N)            # s_i / N along every column
    row = ops.mult_pt(S_, np.full(slots, 1.0 / N), kind="mask")
    return ops.sub(ops.mult_pt(G, np.full(slots, 1.0 / N), kind="mask"), ops.mult(col, row))


def block_power(ops, c, d: int, k: int, iters: int, *, v0, bound: float, gap_kappa: float, slots: int | None = None,
                trace_band: tuple | None = None, orth: str = "gram"):
    """Top-k invariant subspace of a symmetric PSD C (d x d, O on gram_cube(d), ||C|| <= bound) by subspace
    iteration: Y = C V, V = polar(Y), `iters` times from the public start v0 (d x k, plaintext). The polar factor
    needs sigma(Y) in [bound / gap_kappa, bound] (gap_kappa: a public bound on lambda_1 / lambda_k). Returns O(V) on
    full_cube(d, k)."""
    slots = slots or ops_slots(ops)
    gc, pc = gram_cube(d, slots), full_cube(d, k, slots)
    cl = relayout(ops, c, O(gc, (d, d)), L(pc, (d, d)))
    y = mm_pt(ops, cl, encode(v0, R(pc, (d, k))), pc)
    v = None
    for it in range(iters):
        v = orthonormalize(ops, y, d, k, bound=bound, kappa=gap_kappa, slots=slots, trace_band=trace_band, method=orth)
        if it + 1 < iters:
            y = mm(ops, cl, relayout(ops, v, O(pc, (d, k)), R(pc, (d, k))), pc)
    return v


def rsvd(ops, a_l1, at_l2, m: int, n: int, l: int, q: int, *, omega, bound: float, kappa: float,
         slots: int | None = None, trace_band: tuple | None = None, orth_iters: int | None = None, orth: str = "gram"):
    """Randomized range finder (Halko, Martinsson and Tropp 2011, Alg. 4.4 with q power steps): Y = A Omega with the
    PUBLIC Gaussian Omega (n x l, plaintext product), q times Y = A (A^T Q) re-orthonormalized, Q = polar(Y), and
    B^T = A^T Q (n x l). A comes in two faces the client encrypts: L on full_cube(m, l) (contraction over n) and A^T as
    L on full_cube(n, l) (contraction over m). The small SVD of B = Q^T A is the client's (it decrypts Q and B^T,
    O((m + n) l^2) plaintext work), so Q only needs to keep the power steps well conditioned: `orth_iters` caps the
    polar iterations (the client re-orthonormalizes Q = Q' R' and takes the SVD of R'^{-T} B). Returns (O(Q) on full_cube(m, l), O(B^T) on full_cube(n, l))."""
    slots = slots or ops_slots(ops)
    p1, p2_ = full_cube(p2(m), l, slots), full_cube(p2(n), l, slots)
    y = mm_pt(ops, a_l1, encode(omega, R(p1, (n, l))), p1)
    Q = orthonormalize(ops, y, m, l, bound=bound, kappa=kappa, slots=slots, trace_band=trace_band, iters=orth_iters,
                       method=orth)
    tb2 = trace_band
    for _ in range(q):
        w = mm(ops, at_l2, relayout(ops, Q, O(p1, (m, l)), R(p2_, (m, l))), p2_)      # A^T Q (n x l)
        y = mm(ops, a_l1, relayout(ops, w, O(p2_, (n, l)), R(p1, (n, l))), p1)        # A A^T Q
        Q = orthonormalize(ops, y, m, l, bound=bound * bound, kappa=kappa * kappa, slots=slots, trace_band=tb2,
                           iters=orth_iters, method=orth)
    bt = mm(ops, at_l2, relayout(ops, Q, O(p1, (m, l)), R(p2_, (m, l))), p2_)
    return Q, bt


def ns_schedule(lo: float, hi: float, denom: float, spd: bool, accel: bool = True, eps: float = 1e-8,
                max_iters: int = 200, guard: float = 0.25):
    """(alphas, scales) of the Newton-Schulz steps X <- alpha X (2 I - alpha A X) from X0 = A^T / denom (singular
    values of A in [lo, hi]) or, `spd`, I / denom (eigenvalues in [lo, hi]), until the spectrum of A X is within eps
    of 1. `accel`: alpha_k = 2 / (l_k + u_k) for the public band [l_k, u_k] of A X_k's spectrum, the Chebyshev-optimal
    shift of each quadratic step (Pan and Schreiber 1991): a small eigenvalue grows x4 per step instead of x2, so the
    linear phase takes log4 instead of log2 of cond(A X0); the last steps have alpha -> 1 (plain Newton-Schulz, which
    squares away the refresh errors). `guard`: the shift is computed for the band [l, u + guard (u - l)] (vanishing at convergence): alpha x (2 - alpha x)
    sends x > 2 / alpha negative, and the unguarded shift puts 2 / alpha at l + 1, so a refresh error that lifts an
    eigenvalue just past 1 flips it (measured: 64 batched 8 x 8 inverses at a 3e-4 refresh error, median error 7). scales: s_k >= ||X_k|| on a log grid of the band (the iterate is held as
    X_k / s_k)."""
    v = np.geomspace(lo, hi, 4096)
    e = v / denom if spd else v * v / denom
    alphas, scales = [], [float((e / v).max())]
    while 1.0 - e.min() > eps and len(alphas) < max_iters:
        l, u = float(e.min()), float(e.max())
        al = 2.0 / (l + u + guard * (u - l)) if accel else 1.0
        e = al * e * (2.0 - al * e)
        alphas.append(al)
        scales.append(float((e / v).max()))
    return alphas, scales


def inverse_fast(ops, a, n: int, *, B: int = 1, bound: float = 1.0, kappa: float = 100.0, spd: bool = False,
                 accel: bool = True, wrap: bool = True, slots: int | None = None, src: Lay | None = None,
                 refresh_iterate: bool = True, tag_period: bool = False, eps: float = 1e-8, keep_scale: bool = False):
    """`inverse` ("ns") with its cost levers: B problems per ciphertext (batched_cube: B = slots / n^3 fills it,
    one refresh per step for all of them), `accel` (ns_schedule: ~40 % fewer steps), `wrap` faces (each product's
    axis-0 sum over n terms, not slots / (B n^2)), and (`tag_period`) the refreshed iterate tagged with its slot
    period B n^2 (the replicated O face) so the capture routes it sparse (the twin must land it where the compiled
    route does: FAKECAP_LANDINGS=runtime). A in the batched O face (B x n x n). Returns
    O(A_b^{-1}) (`keep_scale`: (O(Z), s), A_b^{-1} = s Z_b)."""
    slots = slots or ops_slots(ops)
    cube = batched_cube(n, B, slots)
    assert cube.d[0] >= n, "B n^3 <= slots"
    lo, ll, lr = O(cube, (n, n), B), L(cube, (n, n), B, wrap), R(cube, (n, n), B, wrap)
    src = src or lo
    terms = n if wrap else None
    smin, smax, denom = bound / kappa, bound, (bound if spd else bound * bound)
    alphas, sc = ns_schedule(smin, smax, denom, spd, accel, eps)
    al = relayout(ops, a, src, ll)
    if spd:                     # X1 = alpha0 X0 (2 I - alpha0 A X0), X0 = I / denom: closed form, one level
        a0 = alphas[0]
        zo = relayout(ops, a, src, lo, scale=-a0 * a0 / (denom * denom * sc[1]))
        zo = add_eye(ops, zo, lo, 2.0 * a0 / (denom * sc[1]))
        k0 = 1
    else:
        zo = relayout(ops, a, src, lo, scale=1.0 / (denom * sc[0]), transpose=True)
        k0 = 0
    for k in range(k0, len(alphas)):
        if k > k0 or spd:
            if refresh_iterate and tag_period and hasattr(ops, "tag_reduce"):
                ops.tag_reduce(zo, cube.strides[0])
            zo = refresh(ops, zo, refresh_iterate, lo, real=not tag_period)
        zl, zr = relayout(ops, zo, lo, ll), relayout(ops, zo, lo, lr, scale=sc[k])
        t = mm(ops, al, zr, cube, terms)                                    # A X_k
        last = k == len(alphas) - 1 and not keep_scale
        r, ak = (sc[k] if last else sc[k] / sc[k + 1]), alphas[k]
        m = add_eye(ops, relayout(ops, t, lo, lr, scale=-ak * ak * r), lr, 2.0 * ak * r)
        zo = mm(ops, zl, m, cube, terms)
    if keep_scale:
        return zo, sc[len(alphas)]
    return zo if len(alphas) > k0 else ops.mult(zo, sc[len(alphas)])


def ops_count_schedule(kappa, spd=False, accel=True):
    return len(ns_schedule(1.0 / kappa, 1.0, 1.0, spd, accel)[0])


def inverse_jacobi(ops, a, n: int, *, dlo: float, dhi: float, bound: float, kappa: float, slots: int | None = None,
                   src: Lay | None = None, **kw):
    """A^{-1} for SPD A through its Jacobi (diagonal) equilibration: u = diag(A)^{-1/2} by inv_sqrt_unit on the
    PUBLIC band [dlo, dhi] of the diagonal (one vectorized inverse square root: the diagonal broadcast along rows and
    down columns by face gathers), B = D^{-1/2} A D^{-1/2} (unit diagonal; `bound` >= lambda_max(B), `kappa` >= cond(B)),
    B^{-1} by `inverse` (SPD start), A^{-1} = D^{-1/2} B^{-1} D^{-1/2} (two elementwise products). For a badly scaled
    A = D^{1/2} B D^{1/2} the steps follow cond(B), not cond(A) (van der Sluis 1969: the diagonal scaling is within a
    factor n of the optimal one). Returns O(A^{-1})."""
    slots = slots or ops_slots(ops)
    cube = full_cube(n, n, slots)
    src = src or O(cube, (n, n))
    lo = O(cube, (n, n))
    ur = inv_sqrt_unit(ops, relayout(ops, a, src, lo, gather=("drow",)), dlo, dhi)    # sqrt(dlo / d_i)
    uc = inv_sqrt_unit(ops, relayout(ops, a, src, lo, gather=("dcol",)), dlo, dhi)    # sqrt(dlo / d_k)
    ah = ops.mult(ops.mult(relayout(ops, a, src, lo), ur), uc)                          # dlo B
    z = inverse(ops, ah, n, bound=dlo * bound, kappa=kappa, spd=True, slots=slots, src=lo, **kw)
    return ops.mult(ops.mult(z, ur), uc)


def ops_slots(ops):
    inf = getattr(ops, "inf", None)
    s = getattr(inf, "slots", None)
    return int(s) if s else (1 << 15)


def inverse_rotations(n, slots=1 << 15, spd=False, method="ns"):
    cube = full_cube(n, n, slots)
    lo, ll, lr = O(cube, (n, n)), L(cube, (n, n)), R(cube, (n, n))
    st = set(mm_steps(cube))
    for (s_, d_, tr) in ((lo, ll, False), (lo, lr, False), (lo, lo, True), (lo, ll, True), (lo, lr, True)):
        st |= set(relayout_plan(s_, d_, transpose=tr).rotations())
    return sorted(st)
