"""Reductions and statistics on batched slot vectors.

Layout (`Batch`): B vectors of n values interleaved, value j of vector b at slot j B + b, the pattern tiled over the
slots (period `span` B, span = n, or 2n with `pad=True`: zeros after each vector's values, what a prefix sum needs). A
reduction over j is a rotate-and-sum ladder at strides B, 2B, ..., and on a tiled layout it leaves every slot holding
its vector's result: the result is already broadcast, no mask and no level. Everything below costs log2(span)
rotations per reduction; the levels are the products' (listed per function).

* `total`, `mean` (one level: the 1/n), weighted versions (`w`: public weights, one plaintext product).
* `variance` / `std`: the centred two-pass form, the 1/sqrt(n - ddof) folded into the centring, two levels for the
  variance; `std` and `zscore` add impl.math's sqrt / rsqrt on the variance's public band.
* `covariance` / `correlation` of m samples x d features (`Features` layout: each sample's d features twice, so a
  rotation by s features is cyclic within the sample): d/2 + 1 cyclic diagonals, each one product of the centred
  data with a (hoisted) rotation of itself and one sample ladder; correlation standardises the features first.
* `logsumexp` / `softmax`: max-shifted (an encrypted maximum from impl.sort, or a calibrated public bound): exp on
  [lo - hi, 0], the sum, then log / reciprocal on the sum's band ([1, n] after an exact max shift).
* `prefix_sum`: Hillis-Steele scan on the padded layout, log2(n) rotations and no level, no mask (the zeros after each
  vector absorb the shifts; the padding holds garbage afterwards).
"""
from __future__ import annotations

import dataclasses
import math as _m

import numpy as np

from . import math as M

SLOTS = 1 << 15


def _pow2(v, what):
    v = int(v)
    if v < 1 or v & (v - 1):
        raise ValueError(f"{what} = {v}: a power of two")
    return v


@dataclasses.dataclass(frozen=True)
class Batch:
    """B vectors of n values (module doc). `pad`: each vector followed by n zeros (prefix sums)."""
    n: int
    B: int = 1
    slots: int = SLOTS
    pad: bool = False

    def __post_init__(self):
        _pow2(self.n, "n")
        _pow2(self.B, "B")
        if self.period > self.slots:
            raise ValueError(f"Batch: span {self.span} x B {self.B} > {self.slots} slots")

    @property
    def span(self):
        return 2 * self.n if self.pad else self.n

    @property
    def period(self):
        return self.span * self.B

    def encode(self, X) -> np.ndarray:
        """(B, n) values (or n for B = 1) -> the slot vector."""
        X = np.asarray(X, dtype=np.float64).reshape(self.B, self.n)
        one = np.zeros((self.span, self.B))
        one[: self.n] = X.T
        return np.tile(one.reshape(-1), self.slots // self.period)

    def decode(self, v) -> np.ndarray:
        """The slot vector -> (B, n) (the first period)."""
        v = np.real(np.asarray(v))[: self.period].reshape(self.span, self.B)
        return v[: self.n].T.copy()

    def live_mask(self, scale=1.0) -> np.ndarray:
        return self.encode(np.full((self.B, self.n), float(scale)))

    def rotations(self) -> list:
        """The rotation steps of a reduction and a prefix sum on this layout."""
        out, g = [], self.B
        while g < self.period:
            out += [g, -g]
            g *= 2
        return out


def total(ops, x, lay: Batch):
    """Each vector's sum in every one of its slots (0 levels, log2(span) rotations)."""
    return ops.rotate_and_sum(x, lay.B, lay.period)


def mean(ops, x, lay: Batch, scale: float = 1.0):
    """scale * the mean (one level)."""
    return ops.mult(total(ops, x, lay), float(scale) / lay.n)


def _w(lay, w):
    """Public weights (B, n) / (n,) -> a slot vector, each vector's weights normalised to sum 1."""
    w = np.asarray(w, dtype=np.float64).reshape(-1, lay.n) * np.ones((lay.B, 1))
    s = w.sum(axis=1, keepdims=True)
    if np.any(s <= 0):
        raise ValueError("weights: a non-positive total")
    return lay.encode(w / s)


def weighted_mean(ops, x, lay: Batch, w):
    """sum_j w_j x_j / sum_j w_j for public weights (one plaintext product)."""
    return total(ops, ops.mult_pt(x, _w(lay, w), kind="mask"), lay)


def weighted_sum(ops, x, lay: Batch, w):
    """sum_j w_j x_j, w public (B, n) / (n,) values or a ciphertext on the layout (one product)."""
    if isinstance(w, np.ndarray) or isinstance(w, (list, tuple)):
        v = lay.encode(np.asarray(w, dtype=np.float64).reshape(-1, lay.n) * np.ones((lay.B, 1)))
        return total(ops, ops.mult_pt(x, v, kind="mask"), lay)
    return total(ops, ops.mult(x, w), lay)


def centred(ops, x, lay: Batch, k: float = 1.0, w=None):
    """k (x - mean) (one level: both scalar products sit at the same depth); `w`: the weighted mean (two levels)."""
    if w is None:
        return ops.sub(ops.mult(x, float(k)), ops.mult(total(ops, x, lay), float(k) / lay.n))
    mu = weighted_mean(ops, x, lay, w)
    return ops.mult(ops.sub(x, mu), float(k))


def variance(ops, x, lay: Batch, ddof: int = 0, w=None):
    """The variance, broadcast over each vector's slots: sum (k (x - mean))^2 with k = 1 / sqrt(n - ddof) (two
    levels); `w`: public weights (sum w (x - mu_w)^2 / sum w: four levels). Two-pass (centred) for stability."""
    if w is None:
        c = centred(ops, x, lay, 1.0 / _m.sqrt(lay.n - ddof))
        return total(ops, ops.square(c), lay)
    c = centred(ops, x, lay, 1.0, w)
    return total(ops, ops.mult_pt(ops.square(c), _w(lay, w), kind="mask"), lay)


def std(ops, x, lay: Batch, lo: float, hi: float, err: float = 1e-4, ddof: int = 0, w=None):
    """sqrt(variance) for a variance in the public band [lo, hi] (impl.math.sqrt)."""
    return M.sqrt(ops, variance(ops, x, lay, ddof, w), lo, hi, err)


def zscore(ops, x, lay: Batch, lo: float, hi: float, err: float = 1e-4, ddof: int = 0):
    """(x - mean) / std for a variance in [lo, hi] (the LayerNorm core without its affine): the centred value times
    impl.math.rsqrt of the variance (relative error `err`)."""
    c = centred(ops, x, lay)
    v = variance(ops, x, lay, ddof)
    return ops.mult(c, M.rsqrt(ops, v, lo, hi, err))


# ── prefix sums ─────────────────────────────────────────────────────────────────────────────────────────────────────


def prefix_sum(ops, x, lay: Batch, inclusive: bool = True):
    """The running sums over j (log2(n) rotations, no level) on a padded layout (`Batch(pad=True)`); the padding
    holds garbage afterwards. Exclusive: the inclusive scan of x shifted by one (one more rotation)."""
    if not lay.pad:
        raise ValueError("prefix_sum: a padded layout (Batch(..., pad=True))")
    if not inclusive:
        x = ops.rotate(x, -lay.B)
    k = 1
    while k < lay.n:
        x = ops.add(x, ops.rotate(x, -k * lay.B))
        k *= 2
    return x


def prefix_rotations(lay: Batch) -> list:
    out, k = [-lay.B], 1
    while k < lay.n:
        out.append(-k * lay.B)
        k *= 2
    return out


# ── covariance / correlation ────────────────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Features:
    """m samples x d features (both powers of two), B batches: feature f of sample i of batch b at slot
    b + B (f + 2 d i), and again at f + d (each sample's features twice: a rotation by s features is cyclic within
    the sample), tiled over the slots."""
    m: int
    d: int
    B: int = 1
    slots: int = SLOTS

    def __post_init__(self):
        _pow2(self.m, "m")
        _pow2(self.d, "d")
        _pow2(self.B, "B")
        if self.period > self.slots:
            raise ValueError(f"Features: 2 d m B = {self.period} > {self.slots} slots")

    @property
    def period(self):
        return 2 * self.d * self.m * self.B

    def encode(self, X) -> np.ndarray:
        """(B, m, d) or (m, d) -> the slot vector."""
        X = np.asarray(X, dtype=np.float64).reshape(self.B, self.m, self.d)
        one = np.concatenate([X, X], axis=2)             # (B, m, 2d)
        one = one.transpose(1, 2, 0).reshape(-1)         # slot = b + B (f + 2 d i)
        return np.tile(one, self.slots // self.period)

    def decode_feature_row(self, v) -> np.ndarray:
        """A per-feature result broadcast over the samples (sample 0's features): (B, d)."""
        v = np.real(np.asarray(v))[: self.period].reshape(self.m, 2 * self.d, self.B)
        return v[0, : self.d].T.copy()

    def decode(self, v) -> np.ndarray:
        v = np.real(np.asarray(v))[: self.period].reshape(self.m, 2 * self.d, self.B)
        return v[:, : self.d].transpose(2, 0, 1).copy()

    def sample_total(self, ops, x):
        return ops.rotate_and_sum(x, 2 * self.d * self.B, self.period)

    def rotations(self) -> list:
        out = [s * self.B for s in range(1, self.d // 2 + 1)]
        g = 2 * self.d * self.B
        while g < self.period:
            out.append(g)
            g *= 2
        return out

    def assemble(self, diags) -> np.ndarray:
        """The decoded diagonals (list of (B, d), diagonal s = C[f, f + s mod d]) -> (B, d, d) symmetric matrices."""
        C = np.zeros((self.B, self.d, self.d))
        f = np.arange(self.d)
        for s, D in enumerate(diags):
            C[:, f, (f + s) % self.d] = D
            C[:, (f + s) % self.d, f] = D
        return C


def _feat_centred(ops, X, lay: Features, k: float):
    """k (X - the per-feature mean over the samples): one level."""
    return ops.sub(ops.mult(X, float(k)), ops.mult(lay.sample_total(ops, X), float(k) / lay.m))


def _diagonals(ops, Z, lay: Features):
    """sum_i Z_{i,f} Z_{i,f+s} for s = 0 .. d/2 (one product and one sample ladder each; the rotations hoisted)."""
    rots = ops.rotate_many(Z, [s * lay.B for s in range(1, lay.d // 2 + 1)])
    out = [lay.sample_total(ops, ops.square(Z))]
    for r in rots:
        out.append(lay.sample_total(ops, ops.mult(Z, r)))
    return out


def covariance(ops, X, lay: Features, ddof: int = 1):
    """The d x d covariance's cyclic diagonals s = 0 .. d/2 (`Features.assemble` builds the matrix), each broadcast
    over the samples: two levels (the 1 / (m - ddof) rides the centring as 1 / sqrt(m - ddof))."""
    Z = _feat_centred(ops, X, lay, 1.0 / _m.sqrt(lay.m - ddof))
    return _diagonals(ops, Z, lay)


def correlation(ops, X, lay: Features, lo: float, hi: float, err: float = 1e-4):
    """The correlation matrix's cyclic diagonals (diagonal 0 ~ 1): the features standardised first (their variance in
    the public band [lo, hi], impl.math.rsqrt), then their covariance."""
    Z = _feat_centred(ops, X, lay, 1.0 / _m.sqrt(lay.m))
    v = lay.sample_total(ops, ops.square(Z))
    Z = ops.mult(Z, M.rsqrt(ops, v, lo, hi, err))
    return _diagonals(ops, Z, lay)


# ── logsumexp / softmax ─────────────────────────────────────────────────────────────────────────────────────────────


def _shift(ops, x, lay, lo, hi, shift, method, err):
    """(u = x - m, m, the exp's domain [a, 0], the sum's band [s_lo, n]) for shift "bound" (m = hi, public) or "max"
    (m = the encrypted maximum, impl.sort)."""
    if shift == "bound":
        return ops.add(x, -float(hi)), float(hi), (float(lo) - float(hi), 0.0), lay.n * _m.exp(lo - hi)
    if shift == "max":
        from . import sort
        m = sort.vmax(ops, x, lay, bound=float(hi) - float(lo), method=method, err=err)
        return ops.sub(x, m), m, (float(lo) - float(hi), 0.0), 1.0
    raise ValueError(f"shift {shift!r}: 'bound' or 'max'")


def softmax(ops, x, lay: Batch, lo: float, hi: float, err: float = 1e-3, shift: str = "max", method: str = "tfhe",
            s_lo: float | None = None):
    """softmax over each vector, x in [lo, hi]: exp(x - m) / sum. `shift` "max" (exact maximum: the sum's band is
    [1, n]) or "bound" (m = hi: the band is [n e^(lo - hi), n], or `s_lo` if calibrated); `method`: the max's
    comparisons (impl.sort)."""
    u, _m_, (a, b), sl = _shift(ops, x, lay, lo, hi, shift, method, err)
    sl = float(s_lo) if s_lo is not None else sl
    e = M.exp(ops, u, a, b, 0.25 * err, rel=False)
    s = total(ops, e, lay)
    return ops.mult(e, M.reciprocal(ops, s, sl, float(lay.n), 0.5 * err, rel=True))


def logsumexp(ops, x, lay: Batch, lo: float, hi: float, err: float = 1e-3, shift: str = "max", method: str = "tfhe",
              s_lo: float | None = None):
    """log sum_j exp(x_j), x in [lo, hi], broadcast over each vector's slots (softmax's shift and band)."""
    u, m, (a, b), sl = _shift(ops, x, lay, lo, hi, shift, method, err)
    sl = float(s_lo) if s_lo is not None else sl
    e = M.exp(ops, u, a, b, 0.25 * err / lay.n, rel=False)
    s = total(ops, e, lay)
    L = M.log(ops, s, sl, float(lay.n), 0.5 * err)
    return ops.add(L, m)
