"""Composite-sign ReLU fit — the configs.json "relu" section.

ReLU(x) = x (1 + sgn(x / B)) / 2 with sgn a composition of odd Chebyshev series (Lee et al., IEEE TDSC 2021: minimax
composite sign). Stage i is the odd degree-d_i polynomial on [-h_i, h_i] minimizing max |p - 1| on [l_i, h_i] while
staying in [-delta, 1 + delta] on the gap [0, l_i]; the next stage's interval is p_i's range on [l_i, h_i]. Stage 1's
gap is eps (relative to B). The last stage carries the (1 + s) / 2, so the runtime's ReLU is x * p_k(...p_1(x)).
B = the site's max |x| times a margin.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from numpy.polynomial import chebyshev as C

from perseus.calibrate.registry import Approximation, register

log = logging.getLogger(__name__)


def _odd_minimax(d, lo, hi, n_grid=2000):
    """Odd Chebyshev coefficients (domain [-hi, hi]) of the degree-d sign stage, and its max |p - 1| on [lo, hi]."""
    from scipy.optimize import linprog
    ks = np.arange(1, d + 1, 2)
    xa = hi * (lo / hi + (1 - lo / hi) * (1 - np.cos(np.linspace(0, np.pi, n_grid))) / 2)   # dense near lo
    xg = np.linspace(0.0, lo, max(n_grid // 4, 50))
    Ta = np.cos(np.outer(np.arccos(np.clip(xa / hi, -1, 1)), ks))
    Tg = np.cos(np.outer(np.arccos(np.clip(xg / hi, -1, 1)), ks))
    n = len(ks)
    # variables (c, delta): minimize delta; |Ta c - 1| <= delta; -delta <= Tg c <= 1 + delta
    one = np.ones((len(xa), 1))
    gone = np.ones((len(xg), 1))
    A = np.vstack([np.hstack([Ta, -one]), np.hstack([-Ta, -one]),
                   np.hstack([Tg, -gone]), np.hstack([-Tg, -gone])])
    b = np.concatenate([np.ones(len(xa)), -np.ones(len(xa)), np.ones(len(xg)), np.zeros(len(xg))])
    cost = np.zeros(n + 1)
    cost[-1] = 1.0
    r = linprog(cost, A_ub=A, b_ub=b, bounds=[(None, None)] * n + [(0, None)], method="highs")
    if not r.success:
        raise RuntimeError(f"relu stage d={d} [{lo}, {hi}]: {r.message}")
    coef = np.zeros(d + 1)
    coef[ks] = r.x[:n]
    return coef, float(r.x[-1])


def fit_composite_sign(degs, eps):
    """[(cheb coeffs, h)] per stage: stage i evaluates on [-h_i, h_i] (stage 1: h = 1, i.e. x / B); the last stage
    returns (1 + sgn) / 2."""
    stages, lo, hi = [], float(eps), 1.0
    for d in degs:
        coef, _delta = _odd_minimax(int(d), lo, hi)
        xs = np.linspace(lo, hi, 4000)
        ys = C.chebval(xs / hi, coef)
        top = float(np.abs(C.chebval(np.linspace(0, hi, 8000) / hi, coef)).max())
        stages.append((coef, hi))
        lo, hi = float(ys.min()), top
    coef, h = stages[-1]
    last = 0.5 * coef
    last[0] += 0.5
    stages[-1] = (last, h)
    return stages


def sign_half(stages, t):
    """(1 + sgn(t)) / 2 of t = x / B (numpy or a float64 tensor) through the composite."""
    y = t
    for coef, h in stages:
        y = C.chebval(np.asarray(y) / h, coef)
    return y


def relu_apply(x, B, stages):
    """The approximate ReLU of x (numpy)."""
    x = np.asarray(x)
    return x * sign_half(stages, x / B)


def depth(degs):
    """Multiplicative depth of the composite ReLU: each Chebyshev stage ceil(log2(d + 1)), plus the product by x."""
    return int(sum(int(np.ceil(np.log2(d + 1))) for d in degs)) + 1


def fit_relu_section(samples, cfg):
    """Per site: B = margin * max |x|, the ladder `cfg.degs` with gap `cfg.eps`."""
    degs = [int(d) for d in cfg.degs]
    stages = fit_composite_sign(degs, float(cfg.eps))
    out = {"degs": degs, "eps": float(cfg.eps), "depth": depth(degs),
           "stages": [{"cheb": c.tolist(), "h": h} for c, h in stages], "sites": {}}
    for n, s in sorted(samples.items()):
        B = float(cfg.margin) * float(s)
        out["sites"][n] = {"B": B}
        log.info(f"[relu] {n:16s} B={B:7.3f}")
    return out


class _ReluCollector:
    """max |input| per site."""

    def __init__(self, cfg):
        del cfg
        self._max: dict[str, float] = {}

    def attach(self, name, module):
        def hook(module_, inp, output):
            del module_, output
            x = inp[0] if isinstance(inp, tuple) else inp
            if torch.is_tensor(x):
                self._max[name] = max(self._max.get(name, 0.0), float(x.detach().abs().max()))

        return [module.register_forward_hook(hook)]

    def on_output(self, output):
        pass

    def finalize(self):
        return dict(self._max) or None


APPROX = register(Approximation(
    kind="relu",
    section="relu",
    matches=lambda m: isinstance(m, nn.ReLU),
    make_collector=_ReluCollector,
    fit_section=fit_relu_section,
))
