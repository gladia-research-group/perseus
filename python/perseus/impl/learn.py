"""Encrypted training of linear models on the ops protocol: logistic regression and a smoothed-hinge linear SVM by
full-batch gradient descent (plain or Nesterov), ridge regression in closed form (impl.linalg / impl.solvers).

Layout (`Design`): N samples x d features in one ciphertext as impl.cluster's Points with K = 1: slot = f N + i,
feature f outermost (dp = slots / N >= d + 1), the last padded feature a constant 1 (the bias). A weight vector w is
replicated over the samples. The margins z_i = sum_f x_if w_f are one product and a window sum down the cyclic
feature axis (broadcast over f, what the next product needs); the gradient g_f = sum_i x_if e_i is one product and
a window sum over the samples, correct at i = 0, masked (one level, the step size folded into the mask) and
replicated back over the samples. An epoch is thus: two products, the link function (impl.math's sigmoid on the
public margin band [-zb, zb], chosen on its encrypted error), the mask and (Nesterov) one scalar product.

The model is the true weights at every step (no public rescaling); the client decrypts w. Labels: y in {0, 1}
(logistic), y' = 2 y - 1 in {-1, 1} (SVM), replicated over f."""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from . import linalg as la
from . import math as M
from .cluster import Points


@dataclasses.dataclass(frozen=True)
class Design(Points):
    """N samples x d features (K = 1 Points; the bias at feature d)."""
    K: int = 1

    def encode_w(self, w):
        """w (d + 1: the bias last) replicated over the samples."""
        w = np.asarray(w, dtype=np.float64)
        wp = np.zeros(self.dp)
        wp[:self.d + 1] = w
        f, _c, _i = self._idx()
        return wp[f]

    def decode_w(self, v):
        v = np.real(np.asarray(v))
        f, _c, i = self._idx()
        sel = i == 0
        out = np.zeros(self.dp)
        out[f[sel]] = v[sel]
        return out[:self.d + 1]


def margins(ops, x, w, lay: Design):
    """z_i = x_i . w broadcast over f (one product, log2 dp rotations)."""
    s = lay.N
    return ops.rotate_and_sum(ops.mult(x, w), s, lay.slots)


def grad_step(ops, x, e, lay: Design, scale: float):
    """scale x sum_i x_if e_i replicated over the samples (one product, the masked sum: two levels)."""
    t = ops.rotate_and_sum(ops.mult(x, e), 1, lay.N)
    _f, _c, i = lay._idx()
    t = ops.mult_pt(t, np.where(i == 0, float(scale), 0.0), kind="mask")
    return ops.rotate_and_sum(t, -1, lay.N)


def nesterov_mu(t: int) -> float:
    """Nesterov's momentum (k - 1) / (k + 2) at step t (Nesterov 1983; the convex schedule)."""
    return t / (t + 3.0)


@dataclasses.dataclass(frozen=True)
class Trainer:
    """A linear model's public schedule: `loss` "logistic" / "svm" (softplus-smoothed hinge with sharpness `beta`),
    learning rate, L2 `lam`, epochs, Nesterov momentum, the margin band zb (|z| <= zb on every epoch: PUBLIC, from
    ||x|| <= 1 and a bound on ||w||) and the link's error target."""
    loss: str = "logistic"
    lr: float = 1.0
    lam: float = 0.0
    epochs: int = 4
    nesterov: bool = True
    zb: float = 8.0
    beta: float = 4.0
    err: float = 1e-3
    fbs: bool | None = None

    def link(self):
        """sigmoid on the margin band ([-zb, zb] for logistic, the hinge's argument [beta (1 - zb), beta (1 + zb)]).
        `fbs`: as a functional bootstrap of the margin (the cheapest that lands no deeper than the series after a
        refresh, impl.math `fbs_choice`) when one meets the target: the epoch's refresh and its link in one. None
        (default): the FBS for the logistic loss (n32 planned, breast cancer / wine / iris: -11..12 % per training,
        the same accuracy, 4 bootstraps instead of 10 besides the links), the series for the hinge (its argument
        reaches the band's edge, past which a series on EvalMod's sine or cosine diverges: w off by 0.6..1.4)."""
        b = self.zb if self.loss == "logistic" else self.beta * (1.0 + self.zb)
        a = M.approx("sigmoid", -b, b, self.err)
        if self.fbs or (self.fbs is None and self.loss == "logistic"):
            f = M.fbs_choice("sigmoid", -b, b, self.err, False, a.depth)
            if f is not None:
                return f
        return a


def train(ops, x, y, lay: Design, tr: Trainer, w0=None, refresh_iterate: bool = False):
    """Full-batch GD of `tr` on samples x (Design.encode_points), labels y (encode_labels: {0, 1}), from w0 (encrypted
    or None: zero). Returns the weights (replicated over the samples)."""
    sig = tr.link()
    eta = tr.lr / lay.N
    w = w0
    v_prev = None
    ysgn = None
    if tr.loss == "svm":
        ysgn = ops.add(ops.mult(y, 2.0), -1.0)                    # y' in {-1, 1}
    for t in range(tr.epochs):
        if w is None:
            z = None
        else:
            if refresh_iterate and t:
                w = la.refresh(ops, w, True)
            z = margins(ops, x, w, lay)
        if tr.loss == "logistic":
            p = sig(ops, z) if z is not None else 0.5
            e = ops.sub(p, y) if z is not None else ops.add(ops.mult(y, -1.0), 0.5)
        else:
            # d/dz softplus(beta (1 - y' z)) / beta = -y' sigmoid(beta (1 - y' z))
            if z is None:
                e = ops.mult(ysgn, -float(sig.numpy(np.array([tr.beta]))[0]))     # the link's own value at z = 0
            else:
                u = ops.add(ops.mult(ops.mult(ysgn, z), -tr.beta), tr.beta)
                e = ops.mult(ysgn, ops.mult(sig(ops, u), -1.0))
        step = grad_step(ops, x, e, lay, -eta)                    # -eta g
        if tr.lam and w is not None:
            v = ops.add(ops.mult(w, 1.0 - tr.lr * tr.lam), step)
        else:
            v = step if w is None else ops.add(w, step)
        if tr.nesterov and v_prev is not None:
            mu = nesterov_mu(t)
            w = ops.add(ops.mult(v, 1.0 + mu), ops.mult(v_prev, -mu))
        else:
            w = v
        v_prev = v
    return v_prev


def train_numpy(X, y, tr: Trainer, link=None):
    """The same schedule in float64 (`link`: the encrypted run's sigmoid approximation, else the exact one)."""
    N, d = X.shape
    Xb = np.c_[X, np.ones(N)]
    w = np.zeros(d + 1)
    v_prev = None
    sg = link or (lambda u: 1.0 / (1.0 + np.exp(-u)))
    ys = 2.0 * y - 1.0
    for t in range(tr.epochs):
        z = Xb @ w
        if tr.loss == "logistic":
            e = sg(z) - y
        else:
            e = -ys * sg(tr.beta * (1.0 - ys * z))
        v = w * (1.0 - tr.lr * tr.lam) - tr.lr / N * (Xb.T @ e)
        w = v + nesterov_mu(t) * (v - v_prev) if (tr.nesterov and v_prev is not None) else v
        v_prev = v
    return v_prev


def ridge_normal(ops, x_r, y_l, m: int, n: int, lam: float, *, bound: float, slots: int, method: str = "ns",
                 eps: float = 1e-4, lay=None):
    """argmin ||X w - y||^2 / m + lam ||w||^2 in closed form (X m x n in the R face of linalg.gram_cube(n), y^T in
    its L face; ||X||^2 / m <= bound, PUBLIC): G = X^T X / m + lam I has cond <= (bound + lam) / lam whatever the data.
    `method` "ns": linalg.lstsq (Newton-Schulz on G, the last step on the vector), "cg" / "cheb": impl.solvers on G
    (B = 1 vector layout). Returns w (lstsq: O face (n x 1) of gram_cube; cg / cheb: lay's R face)."""
    from . import solvers as S
    if method == "ns":
        s = 1.0 / math.sqrt(m)
        xr = ops.mult(x_r, s)
        g = la.gram_cube(n, slots)
        yr = la.relayout(ops, y_l, la.L(g, (1, m)), la.R(g, (m, 1)), transpose=True, scale=s)
        return la.lstsq(ops, xr, yr, m, n, bound=math.sqrt(bound), kappa=math.sqrt((bound + lam) / lam) * 1.01,
                        ridge=lam, slots=slots)
    lay = lay or S.VecLay(n, 1, slots)
    G, c = S.normal_system(ops, x_r, y_l, m, n, lay, slots=slots)
    G = la.add_eye(ops, ops.mult(G, 1.0 / m), lay.mat, lam)
    c = ops.mult(c, 1.0 / m)
    lo, hi = lam, bound + lam
    if method == "cg":
        return S.cg(ops, G, c, lay, S.cg_plan(lo, hi, math.sqrt(bound), eps=eps))
    return S.chebyshev(ops, G, c, lay, lo, hi, bnorm=math.sqrt(bound), eps=eps)
