"""THOR composite of the exact (erf) GELU in the Chebyshev basis: GELU(x) ~ x (p2(p1(x / S)) + 1/2), p1 a squash of
t = x / S on [-1, 1], p2 on p1's range (perseus.calibrate.approximations.gelu's scheme, erf target instead of tanh,
fitted directly as Chebyshev series: the n32 runtime's thor_p*_cheb keys)."""
from __future__ import annotations

import math
import sys

import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.special import erf


def _wfit(x, y, deg, w, a, b):
    u = (2 * x - (a + b)) / (b - a)
    V = C.chebvander(u, deg)
    coef, *_ = np.linalg.lstsq(V * w[:, None], y * w, rcond=None)
    return coef


def _ev(c, x, a, b):
    return C.chebval((2 * x - (a + b)) / (b - a), c)


def fit(S: float, d1: int = 31, d2: int = 27, pad: float = 0.02):
    """(entry dict, max abs error, median abs error) for GELU on [-S, S]."""
    k = np.arange(1, 6001)
    t = np.sort(np.concatenate([np.cos((2 * k - 1) * np.pi / 12000), np.linspace(-1.0, 1.0, 6000)]))
    x = S * t
    T = 0.5 * erf(x / math.sqrt(2.0))
    ref = x * (T + 0.5)
    wgt = 1.0 + 6.0 * np.exp(-0.5 * (x / 2.5) ** 2)
    best = None
    for name, s in (("cbrt", np.cbrt(x) / np.cbrt(S)), ("lin", t)):
        for gamma in np.linspace(1.0, 20.0, 39):
            p1 = _wfit(t, np.tanh(gamma * s), d1, wgt, -1.0, 1.0)
            y1 = _ev(p1, t, -1.0, 1.0)
            lo, hi = float(y1.min()), float(y1.max())
            a2, b2 = lo - pad * (hi - lo), hi + pad * (hi - lo)
            p2 = _wfit(y1, T, d2, wgt, a2, b2)
            err = np.abs(x * (_ev(p2, y1, a2, b2) + 0.5) - ref)
            e = float(err.max())
            if best is None or e < best[0]:
                best = (e, float(np.median(err)), p1, p2, a2, b2)
    e, med, p1, p2, a2, b2 = best
    entry = {"method": "thor_composite", "xmax": S,
             "thor_p1": [float(v) for v in C.cheb2poly(p1)],
             "thor_p2": [float(v) for v in C.cheb2poly(p2)],
             "thor_p1_cheb": [float(v) for v in p1], "thor_p1_a": -1.0, "thor_p1_b": 1.0,
             "thor_p2_cheb": [float(v) for v in p2], "thor_p2_a": a2, "thor_p2_b": b2}
    return entry, e, med


def main():
    for S in (26.0, 77.0, 111.0, 145.0, 173.0):
        for d1, d2 in ((31, 27), (47, 31)):
            _, e, med = fit(S, d1, d2)
            print(f"S {S:6.1f} d1 {d1} d2 {d2}: max err {e:.3e} median {med:.3e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
