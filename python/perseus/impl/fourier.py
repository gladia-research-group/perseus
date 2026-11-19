"""Encrypted Fourier-type transforms: DFT / inverse DFT, real-input FFTs, DCT-II / DCT-III, 1-D and 2-D, batched.

In CKKS every one of them is a linear map of the slot vector, so each runs as a short sequence of diagonal linear
transforms (the slot-diagonal method of Halevi-Shoup, BSGS as in Han-Ki and Chen-Chillotti-Song): one plaintext
level per sequence step, the rotations its diagonals need, nothing else. A transform is described as a list of
sparse n x n factors applied in order (the radix-2 butterflies of a decimation-in-frequency FFT, the bit reversal,
the DCT's Makhoul reordering and twiddle, the real-FFT unpacking); `plan` groups consecutive factors into `levels`
steps (a dynamic program over contiguous groups, the level-budget trade of the CKKS bootstrap's collapsed FFT:
fewer levels -> more diagonals per step -> more rotations). Grouped steps are the same matrices, so a plan only
moves cost between depth and rotations; the values are the transform's up to rounding.

Factors may be R-linear (z -> A z + B conj(z)): the real-input FFTs' Hermitian unpacking. Such a step conjugates
its input once (one key switch) and shares the giant rotations between the z and conj(z) paths.

Layouts (`Layout`): a transform of length n acts on `batch` independent vectors. "block": vector b in slots
[b n, (b + 1) n) (the batched transform is I_batch (x) M; its diagonals are not cyclic unless n = slots);
"stride": element j of vector b at slot j batch + b with n batch = slots (M (x) I_batch; diagonals cyclic, offsets
times batch). A 2-D transform on row-major r x c images uses both at once: the row transform is I_r (x) M_c, the
column transform M_r (x) I_c, so no transpose is needed (a transpose is a permutation with ~r c / 2 diagonals).

Complex payload: a real factorization (the DCTs, any real matrix) applied to z = x + i y transforms x and y at
once (C z = C x + i C y), so every real transform here packs two real vectors per complex slot (`pack_real_pair`).

Entry points: `dft`, `idft`, `rfft_pair` / `irfft_pair`, `rfft_half` / `irfft_half`, `dct2`, `dct3`, `dft_2d`,
`dct2_2d`, `dct3_2d`; each returns a `Transform` (its `steps` with their diagonals, `rot_steps()` the keys,
`apply(ops, ct)` on any ops surface, `reference(x)` the float64 map, `pack` / `unpack` the slot layout)."""
from __future__ import annotations

import dataclasses
import math
from functools import lru_cache

import numpy as np
import scipy.sparse as sp

# Relative device costs of one hoisted baby rotation, one plain rotation (giant / conjugation) and one cached
# plaintext product, measured on the n32 chain at a high level (0.30 / 0.49 / 0.03 ms on an RTX PRO 6000): the
# planner's tie breaker between groupings, not a timing model.
COST_HOIST, COST_ROT, COST_MULT = 0.6, 1.0, 0.06
# The price of one level when `plan` picks the depth itself (levels=None), in rotations: a bootstrap (~12 ms) buys
# ~5 levels of the n32 chain, ~2.4 ms or ~5 rotations each; 4 keeps the planner on the shallow side of that.
COST_LEVEL = 4.0


# ── factors ────────────────────────────────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Factor:
    """z -> A z + B conj(z) on one length-n vector (B None: C-linear)."""
    A: sp.csr_matrix
    B: sp.csr_matrix | None = None

    @property
    def n(self):
        return self.A.shape[0]

    def then(self, g: "Factor") -> "Factor":
        """g after self: g(self(z))."""
        A = g.A @ self.A
        B = None
        if self.B is not None:
            B = g.A @ self.B
        if g.B is not None:
            A = A + g.B @ _conj(self.B) if self.B is not None else A
            gb = g.B @ _conj(self.A)
            B = gb if B is None else B + gb
        return Factor(_clean(A), None if B is None else _clean(B))

    def apply(self, z):
        """Plaintext reference on a (..., n) array."""
        y = (self.A @ np.asarray(z).T).T
        if self.B is not None:
            y = y + (self.B @ np.conj(np.asarray(z)).T).T
        return y

    def kron(self, left: int = 1, right: int = 1) -> "Factor":
        """I_left (x) self (x) I_right."""
        def k(m):
            if m is None:
                return None
            if left > 1:
                m = sp.kron(sp.identity(left, format="csr"), m)
            if right > 1:
                m = sp.kron(m, sp.identity(right, format="csr"))
            return m.tocsr()
        return Factor(k(self.A), k(self.B))

    def scale(self, s) -> "Factor":
        return Factor(self.A * s, None if self.B is None else self.B * s)


def _conj(m):
    return m.conj() if m is not None else None


def _clean(m, tol=1e-14):
    m = sp.csr_matrix(m)
    m.data[np.abs(m.data) < tol] = 0
    m.eliminate_zeros()
    return m


def _perm(dst_of_src):
    """The permutation matrix y[dst[j]] = x[j]."""
    n = len(dst_of_src)
    return sp.csr_matrix((np.ones(n, dtype=np.complex128), (np.asarray(dst_of_src), np.arange(n))), shape=(n, n))


@lru_cache(maxsize=None)
def bit_reverse(n: int) -> np.ndarray:
    L = n.bit_length() - 1
    assert 1 << L == n, "power-of-two lengths only"
    r = np.zeros(n, dtype=np.int64)
    for b in range(L):
        r |= ((np.arange(n) >> b) & 1) << (L - 1 - b)
    return r


def bit_swap(n: int, a: int, b: int) -> Factor:
    """The permutation exchanging index bits a and b: three diagonals (0, +-(2^b - 2^a))."""
    i = np.arange(n)
    t = ((i >> a) ^ (i >> b)) & 1
    return Factor(_perm(i ^ (t << a) ^ (t << b)))


def bit_flip_if(n: int, a: int, c: int) -> Factor:
    """The permutation flipping index bit a where bit c is set: three diagonals (0, +-2^a)."""
    i = np.arange(n)
    return Factor(_perm(i ^ (((i >> c) & 1) << a)))


def bit_reverse_factors(n: int) -> list[Factor]:
    """The bit reversal as L/2 bit swaps (a, L-1-a): 3 diagonals each, instead of one ~n^0.8-diagonal permutation
    (235 diagonals at n = 1024)."""
    L = n.bit_length() - 1
    return [bit_swap(n, a, L - 1 - a) for a in range(L // 2)]


def fft_factors(n: int, sign: int = -1, stage_scale: float = 1.0, perm: str = "swaps") -> list[Factor]:
    """Radix-2 decimation in frequency: [S_1 .. S_L, P] with F = P S_L .. S_1 (P the bit reversal), F_jk =
    exp(sign 2 pi i j k / n) stage_scale^L. Stage s pairs slots j, j + h (h = n / 2^s) inside aligned 2h blocks:
    three diagonals (0, +-h; two cyclic ones at s = 1), never across a 2h block. perm="swaps": P as its L/2 bit
    swaps (`bit_reverse_factors`), "dense": one factor."""
    assert n >= 2 and n & (n - 1) == 0, "power-of-two lengths only"
    out = []
    h = n // 2
    while h >= 1:
        j = np.arange(n)
        top = (j % (2 * h)) < h
        w = np.exp(sign * 2j * np.pi * (j % h) / (2 * h))
        rows = np.concatenate([j[top], j[top], j[~top], j[~top]])
        cols = np.concatenate([j[top], j[top] + h, j[~top] - h, j[~top]])
        vals = np.concatenate([np.ones(top.sum()), np.ones(top.sum()), w[~top], -w[~top]]) * stage_scale
        out.append(Factor(sp.csr_matrix((vals.astype(np.complex128), (rows, cols)), shape=(n, n))))
        h //= 2
    out += bit_reverse_factors(n) if perm == "swaps" else [Factor(_perm(bit_reverse(n)))]
    return out


def dense_factor(M) -> Factor:
    return Factor(sp.csr_matrix(np.asarray(M, dtype=np.complex128)))


# ── layouts and diagonals ─────────────────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class Layout:
    """`batch` length-n vectors in `slots` slots: "block" (contiguous) or "stride" (interleaved, n batch = slots)."""
    n: int
    slots: int = 1 << 15
    batch: int = 1
    mode: str = "block"

    def __post_init__(self):
        assert self.n * self.batch <= self.slots
        if self.mode == "stride":
            assert self.n * self.batch == self.slots, "stride layout fills the slots"

    @property
    def cyclic(self):
        return self.mode == "stride" or self.n == self.slots

    def pack(self, xs):
        """(batch, n) -> (slots,) (a missing batch dim: one vector)."""
        xs = np.atleast_2d(np.asarray(xs))
        assert xs.shape == (self.batch, self.n), (xs.shape, self.batch, self.n)
        v = np.zeros(self.slots, dtype=np.result_type(xs.dtype, np.float64))
        if self.mode == "block":
            v[:self.n * self.batch] = xs.reshape(-1)
        else:
            v[:] = xs.T.reshape(-1)
        return v

    def unpack(self, v):
        v = np.asarray(v)
        if self.mode == "block":
            return v[:self.n * self.batch].reshape(self.batch, self.n)
        return v.reshape(self.n, self.batch).T

    def diagonals(self, m: sp.csr_matrix) -> dict[int, np.ndarray]:
        """The slot diagonals of the batched map: {offset: d} with y = sum d * rot(x, offset), offsets signed in
        (-slots / 2, slots / 2]."""
        m = m.tocoo()
        keep = m.data != 0
        r, c, v = m.row[keep].astype(np.int64), m.col[keep].astype(np.int64), m.data[keep]
        S, n, B = self.slots, self.n, self.batch
        off = self._signed(self._raw(r, c))
        uo, inv = np.unique(off, return_inverse=True)
        D = np.zeros((len(uo), n), dtype=np.complex128)
        D[inv, r] = v                      # one entry per (offset, row)
        out = {}
        for k, o in enumerate(uo):
            if self.mode == "block":
                d = np.zeros(S, dtype=np.complex128)
                d[:n * B] = np.tile(D[k], B)
            else:
                d = np.repeat(D[k], B)
            out[int(o)] = d
        return out

    def _raw(self, r, c):
        if self.mode == "block":
            off = c - r
            return off % self.slots if self.cyclic else off
        return ((c - r) % self.n) * self.batch

    def _signed(self, off):
        S = self.slots
        off = np.where(off > S // 2, off - S, off)
        return np.where(off <= -S // 2, off + S, off)

    def offsets(self, m: sp.csr_matrix) -> set[int]:
        """The diagonal offsets alone (planning; same rule as `diagonals`)."""
        m = m.tocoo()
        keep = m.data != 0
        off = self._signed(np.unique(self._raw(m.row[keep].astype(np.int64), m.col[keep].astype(np.int64))))
        return set(int(o) for o in np.unique(off))


# ── one step: a BSGS diagonal transform ────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Bsgs:
    """Offsets d = g + b split into baby steps b (hoisted rotations of the input) and giant steps g (one rotation
    per giant of the inner sum), b = d mod n1."""
    n1: int
    babies: list[int]
    cbabies: list[int]     # baby steps of conj(x) (R-linear steps)
    giants: list[int]

    @property
    def rotations(self):
        """Key switches: hoisted babies, giants, and the conjugation."""
        return (sum(1 for b in self.babies if b) + sum(1 for b in self.cbabies if b)
                + sum(1 for g in self.giants if g) + (1 if self.cbabies else 0))

    def steps(self):
        return {b for b in self.babies + self.cbabies + self.giants if b}


def _split(offs, n1):
    return {d: (d - (d % n1), d % n1) for d in offs}


def bsgs(offs: set[int], coffs: set[int] = frozenset(), n1: int | None = None) -> Bsgs:
    """The cheapest power-of-two baby step for these offsets (COST_* weights), or the given n1."""
    allo = set(offs) | set(coffs)
    span = max(1, max(abs(d) for d in allo) if allo else 1)
    cands = [n1] if n1 else [1 << k for k in range(0, int(math.log2(2 * span)) + 2)]
    best = None
    for c in cands:
        sa, sc = _split(offs, c), _split(coffs, c)
        babies = sorted({b for _, b in sa.values()})
        cb = sorted({b for _, b in sc.values()})
        giants = sorted({g for g, _ in list(sa.values()) + list(sc.values())})
        cost = (COST_HOIST * (sum(1 for b in babies if b) + sum(1 for b in cb if b))
                + COST_ROT * (sum(1 for g in giants if g) + (1 if cb else 0)))
        if best is None or cost < best[0]:
            best = (cost, Bsgs(c, babies, cb, giants))
    return best[1]


@dataclasses.dataclass
class Step:
    """One level: y = sum_d diags[d] * rot(x, d) + sum_d cdiags[d] * rot(conj(x), d)."""
    diags: dict[int, np.ndarray]
    cdiags: dict[int, np.ndarray]
    plan: Bsgs
    name: str = "f"

    @property
    def n_pt(self):
        return len(self.diags) + len(self.cdiags)


def step_cost(offs, coffs) -> tuple[float, Bsgs]:
    p = bsgs(offs, coffs)
    hb = sum(1 for b in p.babies if b) + sum(1 for b in p.cbabies if b)
    gr = sum(1 for g in p.giants if g) + (1 if p.cbabies else 0)
    return COST_HOIST * hb + COST_ROT * gr + COST_MULT * (len(offs) + len(coffs)), p


def apply_step(ops, x, st: Step, cache: bool = True):
    """One BSGS diagonal step on the ops surface (FheOps, NumpyOps, the fake): babies hoisted from x (and from
    conj(x)), one inner plaintext sum per giant, one rotation per giant. Plaintexts are the diagonals pre-rotated by
    -giant; with `cache` they are named (encoded once per level, the weight path), else encoded at use. An input with
    a pending rescale is realized on a copy first (one rescale instead of one per product)."""
    p = st.plan
    # a pending rescale (the previous step's output) realized once here, not inside every product reading a baby
    if hasattr(ops, "realize"):
        x = ops.realize(ops.copy(x)) if getattr(ops, "noise_deg", lambda a: 1)(x) == 2 else x
    xr = dict(zip([b for b in p.babies if b], ops.rotate_many(x, [b for b in p.babies if b])))
    xr[0] = x
    xc = {}
    if p.cbabies:
        c = ops.conjugate(x)
        xc = dict(zip([b for b in p.cbabies if b], ops.rotate_many(c, [b for b in p.cbabies if b])))
        xc[0] = c
    groups: dict[int, list] = {}
    for d, v in st.diags.items():
        g, b = d - d % p.n1, d % p.n1
        groups.setdefault(g, []).append((xr[b], v, f"{st.name}.d{d}"))
    for d, v in st.cdiags.items():
        g, b = d - d % p.n1, d % p.n1
        groups.setdefault(g, []).append((xc[b], v, f"{st.name}.c{d}"))
    acc = None
    for g in sorted(groups):
        terms = groups[g]
        pts = [np.roll(v, g) for _, v, _ in terms]
        if cache:
            inner = ops.mult_pt(terms[0][0], pts[0], key=terms[0][2])
            for (ct, _, k), pt in zip(terms[1:], pts[1:]):
                ops.inplace_add(inner, ops.mult_pt(ct, pt, key=k))
        else:
            inner = ops.mult_pt_sum([t[0] for t in terms], np.asarray(pts))
        if g:
            inner = ops.rotate(inner, g)
        if acc is None:
            acc = inner
        else:
            ops.inplace_add(acc, inner)
    return acc


# ── planning: group factors into levels ────────────────────────────────────────────────────────────────────────────

def _compose(fs: list[Factor]) -> Factor:
    f = fs[0]
    for g in fs[1:]:
        f = f.then(g)
    return f


def plan(factors: list[Factor], layout: Layout, levels: int | None = None, max_diags: int = 1 << 11,
         max_nnz: int = 1 << 24, exact: bool = False):
    """Contiguous groups of `factors` (at most `levels` of them, exactly `levels` with `exact`; None: the cheapest
    at any depth, each level priced COST_LEVEL) minimizing the summed step cost. Returns [(i, j)] factor ranges."""
    L = len(factors)
    if exact and levels > L:
        raise ValueError(f"{levels} levels from {L} factors")
    cost = {}
    for i in range(L):
        f = None
        for j in range(i + 1, L + 1):
            f = factors[i] if f is None else f.then(factors[j - 1])
            if f.A.nnz + (f.B.nnz if f.B is not None else 0) > max_nnz:
                break
            offs = layout.offsets(f.A)
            coffs = layout.offsets(f.B) if f.B is not None else set()
            if len(offs) + len(coffs) > max_diags:
                break
            cost[i, j] = step_cost(offs, coffs)[0]
    D = levels if levels is not None else L
    pen = 0.0 if levels is not None else COST_LEVEL
    INF = float("inf")
    best = [[INF] * (L + 1) for _ in range(D + 1)]
    arg = [[None] * (L + 1) for _ in range(D + 1)]
    best[0][0] = 0.0
    for d in range(1, D + 1):
        for j in range(1, L + 1):
            for i in range(j):
                if (i, j) in cost and best[d - 1][i] + cost[i, j] + pen < best[d][j]:
                    best[d][j] = best[d - 1][i] + cost[i, j] + pen
                    arg[d][j] = i
    d = D if exact else min(range(1, D + 1), key=lambda k: (best[k][L], k))
    if best[d][L] == INF:
        if max_diags < max_nnz:
            return plan(factors, layout, levels, max_diags * 4, max_nnz, exact)
        raise ValueError(f"no grouping in {levels} levels under {max_nnz} nonzeros per step")
    out, j = [], L
    while j > 0:
        i = arg[d][j]
        out.append((i, j))
        j, d = i, d - 1
    return out[::-1]


# ── transforms ─────────────────────────────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Transform:
    """A planned transform: `steps` (one level each) on `layout`; `ref` the composed map (float64 reference)."""
    name: str
    layout: Layout
    factors: list[Factor]
    groups: list[tuple[int, int]]
    steps: list[Step]
    out_layout: Layout | None = None

    @property
    def levels(self):
        return len(self.steps)

    def rot_steps(self) -> set[int]:
        """The rotation keys the transform needs (normalized to (-slots/2, slots/2])."""
        S = self.layout.slots
        out = set()
        for st in self.steps:
            for k in st.plan.steps():
                k %= S
                out.add(k - S if k > S // 2 else k)
        out.discard(0)
        return out

    def needs_conj(self):
        return any(st.plan.cbabies for st in self.steps)

    def rotations(self):
        return sum(st.plan.rotations for st in self.steps)

    def n_pt(self):
        return sum(st.n_pt for st in self.steps)

    def apply(self, ops, x, cache: bool = True):
        for st in self.steps:
            x = apply_step(ops, x, st, cache)
        return x

    def matrix(self) -> Factor:
        return _compose(self.factors)

    def reference(self, xs):
        """The float64 map on (batch, n) inputs."""
        return self.matrix().apply(np.atleast_2d(xs))

    def pack(self, xs):
        return self.layout.pack(xs)

    def unpack(self, v):
        return (self.out_layout or self.layout).unpack(v)

    def summary(self):
        return {"name": self.name, "n": self.layout.n, "batch": self.layout.batch, "mode": self.layout.mode,
                "levels": self.levels, "rotations": self.rotations(), "keys": len(self.rot_steps()) +
                (1 if self.needs_conj() else 0), "plaintexts": self.n_pt(),
                "diags_per_step": [st.n_pt for st in self.steps], "groups": self.groups}


def build(name: str, factors: list[Factor], layout: Layout, levels: int | None = None, groups=None,
          max_diags: int = 1 << 11, exact: bool = False) -> Transform:
    """Plan (or take `groups`) and materialize the steps' diagonals and BSGS splits."""
    groups = groups if groups is not None else plan(factors, layout, levels, max_diags, exact=exact)
    steps = []
    tag = f"fourier.{name}{layout.n}{layout.mode[0]}{layout.batch}.g" + "_".join(f"{i}-{j}" for i, j in groups)
    for k, (i, j) in enumerate(groups):
        f = _compose(factors[i:j])
        dg = layout.diagonals(f.A)
        cd = layout.diagonals(f.B) if f.B is not None else {}
        steps.append(Step(dg, cd, bsgs(set(dg), set(cd)), f"{tag}.s{k}"))
    return Transform(name, layout, factors, list(groups), steps)


def _norm_scale(n: int, norm: str, inverse: bool) -> float:
    """numpy's `norm` as one overall factor."""
    if norm == "ortho":
        return 1 / math.sqrt(n)
    if norm == "backward":
        return 1 / n if inverse else 1.0
    if norm == "forward":
        return 1.0 if inverse else 1 / n
    raise ValueError(norm)


def dft_factors(n: int, inverse: bool = False, norm: str = "backward", order: str = "natural") -> list[Factor]:
    """numpy.fft.fft (ifft with `inverse`) of length n as factors; the overall scale spread evenly over the
    butterfly stages. order="bitrev": the output in bit-reversed order (the permutation dropped: a forward / inverse
    pair through a pointwise product needs none, see `idft(..., order="bitrev")`)."""
    L = n.bit_length() - 1
    s = _norm_scale(n, norm, inverse) ** (1 / L) if L else 1.0
    L = n.bit_length() - 1
    fs = fft_factors(n, -1, s)
    if order != "natural":
        fs = fs[:L]
    if not inverse:
        return fs
    # ifft: conj-transpose of the forward DIF (a DIT on bit-reversed input), scaled
    return [Factor(f.A.conj().T.tocsr()) for f in reversed(fs)]


def dft(n, levels=None, inverse=False, norm="backward", order="natural", layout: Layout | None = None,
        direct=False, **kw) -> Transform:
    """The length-n DFT (numpy.fft.fft) as a planned transform. direct=True: one dense step (the n-diagonal BSGS).
    order="bitrev": output (forward) or input (inverse) in bit-reversed order, one permutation fewer."""
    layout = layout or Layout(n)
    fs = dft_factors(n, inverse, norm, order)
    nm = ("idft" if inverse else "dft") + ("_direct" if direct else "")
    if direct:
        return build(nm, fs, layout, groups=[(0, len(fs))], max_diags=1 << 30)
    return build(nm, fs, layout, levels, **kw)


def idft(n, levels=None, norm="backward", order="natural", layout=None, direct=False, **kw) -> Transform:
    return dft(n, levels, True, norm, order, layout, direct, **kw)


# real-input FFTs: the Hermitian unpacking as an R-linear factor

def _rl(n, entries_a, entries_b):
    def m(ent):
        if not ent:
            return None
        r, c, v = zip(*ent)
        return sp.csr_matrix((np.asarray(v, dtype=np.complex128), (r, c)), shape=(n, n))
    return Factor(m(entries_a) if entries_a else sp.csr_matrix((n, n), dtype=np.complex128), m(entries_b))


def pair_split(n: int) -> Factor:
    """Z = F(x + i y) -> [X packed | Y packed] (each n/2 slots: slot 0 = X_0 + i X_{n/2}, slot k = X_k)."""
    h = n // 2
    a, b = [], []
    a += [(0, 0, .5), (0, h, .5j)]; b += [(0, 0, .5), (0, h, .5j)]
    a += [(h, 0, -.5j), (h, h, .5)]; b += [(h, 0, .5j), (h, h, -.5)]
    for k in range(1, h):
        a += [(k, k, .5), (h + k, k, -.5j)]
        b += [(k, n - k, .5), (h + k, n - k, .5j)]
    return _rl(n, a, b)


def pair_merge(n: int) -> Factor:
    """The inverse of pair_split: [X packed | Y packed] -> Z = X + i Y (full spectrum)."""
    h = n // 2
    a, b = [], []
    # Z_0 = X_0 + i Y_0 = Re P_x + i Re P_y; Z_h = X_h + i Y_h = Im P_x + i Im P_y
    a += [(0, 0, .5), (0, h, .5j)]; b += [(0, 0, .5), (0, h, .5j)]
    a += [(h, 0, -.5j), (h, h, .5)]; b += [(h, 0, .5j), (h, h, -.5)]
    for k in range(1, h):
        a += [(k, k, 1.0), (k, h + k, 1j)]            # Z_k = X_k + i Y_k
        b += [(n - k, k, 1.0), (n - k, h + k, 1j)]    # Z_{n-k} = conj X_k + i conj Y_k
    return _rl(n, a, b)


def half_split(n: int) -> Factor:
    """Z = F_{n/2}(x_even + i x_odd) -> X_0..X_{n/2-1} of the real length-n x, slot 0 = X_0 + i X_{n/2}."""
    h = n // 2
    a, b = [], [(0, 0, 1 + 1j)]
    for k in range(1, h):
        w = np.exp(-2j * np.pi * k / n)
        a.append((k, k, .5 + w / 2j))
        b.append((k, h - k, .5 - w / 2j))
    return _rl(h, a, b)


def half_merge(n: int) -> Factor:
    """The inverse of half_split: packed X -> Z = F_{n/2}(x_even + i x_odd)."""
    h = n // 2
    a, b = [], [(0, 0, (1 + 1j) / 2)]
    for k in range(1, h):
        wi = np.exp(2j * np.pi * k / n)
        a.append((k, k, .5 + 1j * wi / 2))
        b.append((k, h - k, .5 - 1j * wi / 2))
    return _rl(h, a, b)


def rfft_pair(n, levels=None, inverse=False, layout=None, norm="backward", direct=False, **kw) -> Transform:
    """Two real length-n vectors x, y as z = x + i y (`pack_real_pair`) -> their half spectra [X | Y], X packed
    as numpy.fft.rfft(x) with X_{n/2} in the imaginary part of slot 0 (inverse: back to x + i y)."""
    layout = layout or Layout(n)
    if not inverse:
        fs = dft_factors(n, False, norm) + [pair_split(n)]
    else:
        fs = [pair_merge(n)] + dft_factors(n, True, norm)
    nm = "irfft_pair" if inverse else "rfft_pair"
    if direct:
        return build(nm, fs, layout, groups=[(0, len(fs))], max_diags=1 << 30)
    return build(nm, fs, layout, levels, **kw)


def rfft_half(n, levels=None, inverse=False, layout=None, norm="backward", direct=False, **kw) -> Transform:
    """One real length-n vector in n/2 complex slots (z_m = x_2m + i x_2m+1, `pack_real_half`) -> its half spectrum
    in n/2 slots (slot 0 = X_0 + i X_{n/2}); the transform runs at length n/2 (inverse: back to z)."""
    h = n // 2
    layout = layout or Layout(h)
    assert layout.n == h
    sc = {"backward": 1.0, "ortho": 1 / math.sqrt(n), "forward": 1 / n}[norm]
    if not inverse:
        fs = dft_factors(h, False, "backward") + [half_split(n).scale(sc)]
    else:
        fs = [half_merge(n).scale(1 / sc)] + dft_factors(h, True, "backward")
    nm = "irfft_half" if inverse else "rfft_half"
    if direct:
        return build(nm, fs, layout, groups=[(0, len(fs))], max_diags=1 << 30)
    return build(nm, fs, layout, levels, **kw)


# DCT-II / DCT-III by the length-n FFT of Makhoul's reordering (C-linear: a real map, so it also runs on x + i y)

def makhoul(n: int) -> np.ndarray:
    """v = x[makhoul(n)]: v_m = x_2m, v_{n-1-m} = x_2m+1."""
    idx = np.empty(n, dtype=np.int64)
    idx[:(n + 1) // 2] = np.arange(0, n, 2)
    idx[(n + 1) // 2:] = np.arange(n - 1 if n % 2 == 0 else n - 2, 0, -2)
    return idx


def dct_post(n: int, norm: str) -> Factor:
    """V = F v -> scipy.fft.dct(x, 2, norm): y_k = s_k (T_k V_k + conj(T_k) V_{n-k}), T_k = exp(-i pi k / 2n)."""
    k = np.arange(n)
    T = np.exp(-1j * np.pi * k / (2 * n))
    if norm == "ortho":
        s = np.full(n, math.sqrt(1 / (2 * n)))
        s[0] = math.sqrt(1 / (4 * n))
    else:
        s = np.ones(n)
    rows = np.concatenate([k, k])
    cols = np.concatenate([k, (n - k) % n])
    vals = np.concatenate([s * T, s * np.conj(T)])
    return Factor(_clean(sp.csr_matrix((vals, (rows, cols)), shape=(n, n))))


def makhoul_factors(n: int) -> list[Factor]:
    """v = x[makhoul(n)] as 3-diagonal factors: a cyclic right rotation of the index bits (L - 1 adjacent bit swaps)
    then, where the top bit is set, the lower L - 1 bits complemented (L - 1 conditional flips)."""
    L = n.bit_length() - 1
    fs = [bit_swap(n, a, a + 1) for a in range(L - 1)]
    fs += [bit_flip_if(n, a, L - 1) for a in range(L - 1)]
    return fs


def _check_perm(fs, n, dst):
    m = _compose(fs).A if fs else sp.identity(n)
    assert abs(m - _perm(dst)).max() < 1e-12


def dct_re(n: int, a, b) -> Factor:
    """The diagonal R-linear step z -> a z + b conj(z) (the real part of a twiddled spectrum)."""
    return Factor(sp.diags(np.asarray(a, dtype=np.complex128)).tocsr(), sp.diags(np.asarray(b, dtype=np.complex128)).tocsr())


def _dct_scale(n, norm):
    s = np.full(n, math.sqrt(1 / (2 * n)) if norm == "ortho" else 1.0)
    if norm == "ortho":
        s[0] = math.sqrt(1 / (4 * n))
    return s


def dct2_factors(n: int, norm: str = "ortho", packing: str = "real") -> list[Factor]:
    """scipy.fft.dct(x, 2, norm) as [Makhoul reorder, DIF stages, bit reversal, twiddle] (Makhoul 1980).
    packing="real": x real (one per complex slot); the twiddle is y = 2 Re(s T V), a diagonal R-linear step (one
    conjugation). "pair": x + i y (two real vectors per slot); the twiddle needs V_{n-k}, s (T V + conj(T) R V)
    with R the index reversal (~n/2 diagonals: a C-linear map on the pair)."""
    pm = makhoul(n)
    perm = np.empty(n, dtype=np.int64)
    perm[pm] = np.arange(n)           # x_j lands at v[perm[j]]
    mk = makhoul_factors(n)
    _check_perm(mk, n, perm)
    if packing == "pair":
        return mk + fft_factors(n, -1, 1.0) + [dct_post(n, norm)]
    s = _dct_scale(n, norm)
    T = np.exp(-1j * np.pi * np.arange(n) / (2 * n))
    return mk + fft_factors(n, -1, 1.0) + [dct_re(n, s * T, s * np.conj(T))]


def dct3_factors(n: int, norm: str = "ortho", packing: str = "real") -> list[Factor]:
    """scipy.fft.dct(x, 3, norm), the inverse of DCT-II (scipy's backward pair up to 2n). "pair": the transposes of
    the DCT-II factors in reverse order (C real, C^-1 = C^T at norm="ortho"). "real": the real adjoint of
    dct2 "real", C^T y = Pm^T 2 Re(F (s T y)): a twiddle, the DIF FFT, a real part, the inverse reordering."""
    if norm == "backward":
        # scipy's backward DCT-III: y_k = x_0 + 2 sum_j x_j cos(pi j (2k + 1) / 2n) = C2_ortho^T diag(sqrt(n), sqrt(2n), ..)
        d = np.full(n, math.sqrt(2 * n))
        d[0] = math.sqrt(n)
        pre = [Factor(sp.diags(d.astype(np.complex128)).tocsr())]
    else:
        pre = []
    if packing == "pair":
        return pre + [Factor(f.A.T.tocsr()) for f in reversed(dct2_factors(n, "ortho", "pair"))]
    s = _dct_scale(n, "ortho")
    T = np.exp(-1j * np.pi * np.arange(n) / (2 * n))
    mk = [Factor(f.A.T.tocsr()) for f in reversed(makhoul_factors(n))]
    return (pre + [Factor(sp.diags(s * T).tocsr())] + fft_factors(n, -1, 1.0)
            + [dct_re(n, np.ones(n), np.ones(n))] + mk)


def dct2(n, levels=None, norm="ortho", layout=None, direct=False, packing="real", **kw) -> Transform:
    """scipy.fft.dct(x, 2, norm) on real x ("real") or on x + i y ("pair", `pack_real_pair`)."""
    layout = layout or Layout(n)
    fs = dct2_factors(n, norm, packing)
    if direct:
        return build("dct2_direct", fs, layout, groups=[(0, len(fs))], max_diags=1 << 30)
    return build("dct2" + packing[0], fs, layout, levels, **kw)


def dct3(n, levels=None, norm="ortho", layout=None, direct=False, packing="real", **kw) -> Transform:
    layout = layout or Layout(n)
    fs = dct3_factors(n, norm, packing)
    if direct:
        return build("dct3_direct", fs, layout, groups=[(0, len(fs))], max_diags=1 << 30)
    return build("dct3" + packing[0], fs, layout, levels, **kw)


# 2-D: row-major r x c images, rows as I_r (x) M_c, columns as M_r (x) I_c

def separable_2d(row: list[Factor], col: list[Factor], r: int, c: int) -> list[Factor]:
    """Column factors then row factors on the row-major r x c image (they commute; the planner may merge a column
    step with a row step into one level)."""
    return [f.kron(right=c) for f in col] + [f.kron(left=r) for f in row]


def _two_d(name, fac, r, c, levels, layout, direct, separable, **kw):
    layout = layout or Layout(r * c)
    fs = separable_2d(fac(c), fac(r), r, c)
    if direct:
        return build(name + "_direct", fs, layout, groups=[(0, len(fs))], max_diags=1 << 30)
    if separable:
        # one step per direction: the row-column method
        nc = len(fac(r))
        return build(name + "_rowcol", fs, layout, groups=[(0, nc), (nc, len(fs))], max_diags=1 << 30)
    return build(name, fs, layout, levels, **kw)


def dft_2d(r, levels=None, inverse=False, norm="backward", layout=None, direct=False, separable=False,
           c=None, **kw) -> Transform:
    """numpy.fft.fft2 (ifft2) on row-major r x c images (power-of-two sides)."""
    c = c or r
    return _two_d("idft2" if inverse else "dft2", lambda m: dft_factors(m, inverse, norm), r, c, levels, layout,
                  direct, separable, **kw)


def dct2_2d(r, levels=None, norm="ortho", layout=None, direct=False, separable=False, packing="pair",
            c=None, **kw) -> Transform:
    """scipy.fft.dctn(x, 2, norm) on row-major r x c images (the JPEG 8 x 8 block transform at r = c = 8): "pair"
    two images per slot vector as x + i y, "real" one (each direction's real part is real, so the R-linear passes
    compose)."""
    c = c or r
    return _two_d("dct2_2d" + packing[0], lambda m: dct2_factors(m, norm, packing), r, c, levels, layout, direct,
                  separable, **kw)


def dct3_2d(r, levels=None, norm="ortho", layout=None, direct=False, separable=False, packing="pair",
            c=None, **kw) -> Transform:
    c = c or r
    return _two_d("dct3_2d" + packing[0], lambda m: dct3_factors(m, norm, packing), r, c, levels, layout, direct,
                  separable, **kw)


# packing helpers

def pack_real_pair(x, y):
    """Two real vectors as one complex: a real transform (DCT) or rfft_pair transforms both."""
    return np.asarray(x, dtype=np.float64) + 1j * np.asarray(y, dtype=np.float64)


def pack_real_half(x):
    """A real length-n vector as n/2 complex values x_2m + i x_2m+1 (rfft_half's input)."""
    x = np.asarray(x, dtype=np.float64)
    return x[..., 0::2] + 1j * x[..., 1::2]


def unpack_rfft(X, n):
    """Packed half spectrum (slot 0 = X_0 + i X_{n/2}) -> numpy.fft.rfft's n/2 + 1 values."""
    X = np.asarray(X)
    out = np.concatenate([X[..., :n // 2], X[..., :1].imag], axis=-1).astype(np.complex128)
    out[..., 0] = X[..., 0].real
    return out


def pack_rfft(R, n):
    """numpy.fft.rfft's n/2 + 1 values -> the packed half spectrum."""
    R = np.asarray(R)
    X = R[..., :n // 2].astype(np.complex128).copy()
    X[..., 0] = R[..., 0].real + 1j * R[..., n // 2].real
    return X


def release(ops):
    """Drop the transforms' named plaintexts (every `fourier.` slot plaintext) and the ops' name cache of them."""
    if hasattr(ops, "drop_pts"):
        ops.drop_pts("fourier.")
    for attr in ("_slots", "_slot_name"):
        c = getattr(ops, attr, None)
        if c is not None:
            for k in [k for k in c if (k if isinstance(k, str) else k[0]).startswith(("impl.fourier.", "fourier."))]:
                if isinstance(c, dict):
                    del c[k]
                else:
                    c.discard(k)
