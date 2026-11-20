"""Distances, kernels, nearest neighbours and k-means on the ops protocol.

Layout (`Points`): N points of d features against K references (centroids, or K = 1 for a query) in one
ciphertext, slot = f (K N) + c N + i (feature f outermost, padded to dp = slots / (K N) >= d + 1). A point x_i is
replicated over c, a reference mu_c over i. A squared distance D[c, i] = sum_f (x_if - mu_cf)^2 is one square and a
window sum down the cyclic outermost axis (log2 dp rotations): it lands broadcast over f, i.e. exactly impl.stats'
`Batch(n=K, B=N)` tiled dp times, the layout impl.sort compares in (argmin over c per point, ranks over i per
reference). The last padded feature carries a constant 1 (`Points.encode_points`): a sum over the points of an
assignment times x then also counts the points (the k-means update's denominator) at no extra product.

* `sqdist`: D for every (reference, point) pair; `pairwise_sqdist` / `rbf_kernel` / `poly_kernel`: the N x N matrices
  of a point set from its Gram (impl.linalg's cube product: one level) and the diagonal's two broadcasts.
* `knn`: the k nearest points of a query by all-pairs ranks (impl.sort.rank, n <= 128 per ciphertext) and the
  mean label of the k (a vote: the client thresholds it).
* `kmeans`: Lloyd's iterations: distances, the one-hot of the nearest centroid (impl.sort's comparison: CKKS
  composite sign or the TFHE switch), the per-centroid sums of the assigned points and their counts (one product,
  a masked sum, a replicate ladder), the division by the counts (impl.math.reciprocal on [n_min, N], n_min a
  PUBLIC lower bound of a cluster's size)."""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from . import linalg as la
from . import math as M
from . import sort as SO
from .stats import Batch


@dataclasses.dataclass(frozen=True)
class Points:
    """N points x d features against K references (powers of two; d + 1 <= slots / (K N))."""
    N: int
    d: int
    K: int = 1
    slots: int = 1 << 15

    def __post_init__(self):
        assert self.dp >= self.d + 1, (self.N, self.K, self.d)

    @property
    def dp(self):
        return self.slots // (self.K * self.N)

    @property
    def batch(self):
        """The distances' layout: K values (one per reference) of each of N points."""
        return Batch(self.K, self.N, self.slots)

    def _idx(self):
        s = np.arange(self.slots)
        return s // (self.K * self.N), (s // self.N) % self.K, s % self.N

    def encode_points(self, X, one: bool = True):
        """X (N x d) replicated over the references; feature d = 1 (`one`)."""
        X = np.asarray(X, dtype=np.float64)
        Xp = np.zeros((self.N, self.dp))
        Xp[:, :self.d] = X
        if one:
            Xp[:, self.d] = 1.0
        f, _c, i = self._idx()
        return Xp[i, f]

    def encode_refs(self, Mu):
        """Mu (K x d) replicated over the points (feature d = 1, so x - mu is 0 there)."""
        Mu = np.asarray(Mu, dtype=np.float64).reshape(self.K, self.d)
        Mp = np.zeros((self.K, self.dp))
        Mp[:, :self.d] = Mu
        Mp[:, self.d] = 1.0
        f, c, _i = self._idx()
        return Mp[c, f]

    def decode_refs(self, v):
        v = np.real(np.asarray(v))
        f, c, i = self._idx()
        sel = i == 0
        out = np.zeros((self.K, self.dp))
        out[c[sel], f[sel]] = v[sel]
        return out[:, :self.d]

    def encode_labels(self, y):
        """Per-point values y (N) in the distances' layout (replicated over c and f)."""
        _f, _c, i = self._idx()
        return np.asarray(y, dtype=np.float64)[i]

    def decode_dist(self, v):
        """(K, N) from a distances' slot vector."""
        return self.batch.decode(v).T


def sqdist(ops, x, mu, lay: Points, scale: float = 1.0):
    """scale x D[c, i] broadcast over f (one level: the square; `scale` folded into the difference only when it is
    not 1)."""
    t = ops.sub(x, mu)
    if scale != 1.0:
        t = ops.mult(t, math.sqrt(scale))
    s = lay.K * lay.N
    return ops.rotate_and_sum(ops.square(t), s, lay.slots)


def argmin_onehot(ops, D, lay: Points, bound: float, *, method: str = "tfhe", res: float = 1e-2,
                  err: float = 1e-3):
    """The one-hot over c of each point's smallest D (|D_c - D_c'| <= bound): impl.sort's minimum (tournament) and
    the comparison D <= min + res / 2. TFHE compares the first K N slots (one period) and tiles the result over f;
    CKKS compares every slot at once."""
    b = lay.batch
    m = SO.vmin(ops, D, b, bound, method=method, err=err)
    x = ops.sub(m, D)                                                  # >= 0 at the minimum
    if method == "tfhe":
        heads = np.arange(b.period)
        oh = SO.step(ops, ops.add_pt(x, SO._put(lay.slots, heads, 0.5 * res), kind="mask"), heads, 1.0,
                     method="tfhe", bound=bound + res, tag="kmeans")
        return SO._tile(ops, oh, b, lay.slots)
    heads = np.arange(lay.slots)
    return SO.step(ops, ops.add(x, 0.5 * res), heads, 1.0, method="ckks", bound=bound + res, res=0.5 * res)


def assigned_sums(ops, a, x, lay: Points):
    """S[c, f] = sum_i a[c, i] x[i, f] (feature d: the count) replicated over i: one product, a window sum over i
    (correct at i = 0), a mask (one level) and a replicate ladder."""
    t = ops.rotate_and_sum(ops.mult(a, x), 1, lay.N)
    _f, _c, i = lay._idx()
    t = ops.mult_pt(t, (i == 0).astype(np.float64), kind="mask")
    return ops.rotate_and_sum(t, -1, lay.N)


def _feature(ops, s, lay: Points, f0: int):
    """Feature f0 of every (c, i) broadcast over f: a mask (one level) and the f ladder."""
    f, _c, _i = lay._idx()
    t = ops.mult_pt(s, (f == f0).astype(np.float64), kind="mask")
    return ops.rotate_and_sum(t, lay.K * lay.N, lay.slots)


def kmeans(ops, x, mu, lay: Points, iters: int, *, xbound: float = 1.0, nmin: int = 1, method: str = "tfhe",
           res: float = 1e-2, recip_err: float = 1e-3, refresh_iterate: bool = True):
    """Lloyd's k-means from the centroids `mu` (Points.encode_refs layout, encrypted by the client), points `x`
    (encode_points, with the count feature), features in [-xbound, xbound], every cluster of >= `nmin` points
    (PUBLIC). Returns (mu, assignment one-hot of the last step)."""
    dmax = 4.0 * lay.d * xbound * xbound
    rec = M.approx("reciprocal", float(nmin), float(lay.N), recip_err)
    a = None
    for it in range(iters):
        if refresh_iterate and it:
            mu = la.refresh(ops, mu, True)
        D = sqdist(ops, x, mu, lay, scale=1.0 / dmax)                  # in [0, 1]
        a = argmin_onehot(ops, D, lay, 1.0, method=method, res=res)
        S = assigned_sums(ops, a, x, lay)                              # sums, count at feature d
        cnt = _feature(ops, S, lay, lay.d)
        mu = ops.mult(S, rec(ops, cnt))
    return mu, a


def knn(ops, x, q, y, lay: Points, k: int, *, xbound: float = 1.0, method: str = "ckks", res: float = 1e-3):
    """The k nearest of N points (`x`, K = 1 layout) to the query `q` (encode_refs), and the mean of their labels `y`
    (encode_labels): (vote broadcast, the k-hot mask). Ranks by impl.sort.rank (all pairs; N <= 128)."""
    assert lay.K == 1
    dmax = 4.0 * lay.d * xbound * xbound
    D = sqdist(ops, x, q, lay, scale=1.0 / dmax)
    b = Batch(lay.N, 1, lay.slots)
    r = SO.rank(ops, ops.mult(D, -1.0), b, 1.0, method=method, res=res)      # rank of -D: the nearest is the top
    m = SO.topk_mask(ops, r, b, k, method=method)
    vote = ops.rotate_and_sum(ops.mult(m, y), 1, lay.N)
    return ops.mult(vote, 1.0 / k), m


# ── kernel matrices ─────────────────────────────────────────────────────────────────

def gram_cube(N: int, slots: int):
    return la.full_cube(N, N, slots)


def pairwise_sqdist(ops, x_l, xt_r, N: int, d: int, slots: int, scale: float = 1.0):
    """scale x ||x_i - x_j||^2 (O face, N x N on full_cube(N, N)) from X in its L face and X^T in its R face (the
    client encrypts both; d <= slots / N^2): the Gram G = X X^T (one product), its diagonal broadcast along rows and
    down columns (two face gathers), diag_i + diag_j - 2 G."""
    cube = gram_cube(N, slots)
    lo = la.O(cube, (N, N))
    G = la.mm(ops, x_l, xt_r, cube)
    di = la.relayout(ops, G, lo, lo, gather=("drow",), scale=scale)
    dj = la.relayout(ops, G, lo, lo, gather=("dcol",), scale=scale)
    return ops.add(ops.add(di, dj), ops.mult(G, -2.0 * scale))


def rbf_kernel(ops, x_l, xt_r, N: int, d: int, slots: int, *, gamma: float, dmax: float, err: float = 1e-4):
    """exp(-gamma ||x_i - x_j||^2) (O face N x N), squared distances <= dmax (PUBLIC): impl.math.exp on
    [-gamma dmax, 0]."""
    D = pairwise_sqdist(ops, x_l, xt_r, N, d, slots, scale=gamma)
    return M.exp(ops, ops.mult(D, -1.0), -gamma * dmax * 1.01, 1e-3, err, rel=False)


def poly_kernel(ops, x_l, xt_r, N: int, d: int, slots: int, *, degree: int = 2, c: float = 1.0, scale: float = 1.0):
    """(scale x_i . x_j + c)^degree (O face N x N): the Gram, an affine map, impl.math's integer power (squarings)."""
    cube = gram_cube(N, slots)
    G = la.mm(ops, x_l, xt_r, cube)
    t = ops.add(ops.mult(G, scale), c)
    out = None
    base, p = t, int(degree)
    while p:
        if p & 1:
            out = base if out is None else ops.mult(out, base)
        p >>= 1
        if p:
            base = ops.square(base)
    return out
