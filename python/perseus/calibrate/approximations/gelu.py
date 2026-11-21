"""THOR composite GELU fit — the configs.json "softgelu" section.

GELU(x) ≈ x·(p2(p1(x/S)) + 0.5) (Moon et al. 2024): a two-stage odd polynomial
composition targeting tanh/2, emitted as ascending coefficient lists. S = xmax
comes from the sample distribution, so the x/S normalization matches the C++
gelu_thor_composite op. (The iterative softsign_inv_sqrt method was removed
from the calibrator; the runtime still parses legacy softsign configs.)
"""

import logging
import math

import numpy as np
import torch
import torch.nn as nn

from perseus.calibrate.numerics import safe_quantile
from perseus.calibrate.registry import Approximation, register

log = logging.getLogger(__name__)


def _thor_g(x):
    return math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)


def _thor_wlsq(x, y, deg, w):
    V = np.vander(x, deg + 1, increasing=True)
    coef, *_ = np.linalg.lstsq(V * w[:, None], y * w, rcond=None)
    return coef


def _fit_thor_composite(S, d1, d2):
    from scipy.special import erf
    k = np.arange(1, 4001)
    cheb = np.cos((2 * k - 1) * np.pi / (2 * 4000))
    t = np.sort(np.concatenate([cheb, np.linspace(-1.0, 1.0, 4000)]))
    x = S * t
    gt = _thor_g(x)
    T_tgt = np.tanh(gt) / 2.0
    ref = 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))
    wgt = 1.0 + 6.0 * np.exp(-0.5 * (gt / 2.5) ** 2)
    smax_g = np.abs(gt).max() ** (1.0 / 3.0)
    squashes = [
        np.sign(gt) * np.abs(gt) ** (1.0 / 3.0) / smax_g,
        gt / np.abs(gt).max(),
        t,
    ]
    best = None
    for s_mono in squashes:
        for gamma in np.linspace(1.0, 14.0, 27):
            p1 = _thor_wlsq(t, np.tanh(gamma * s_mono), d1, wgt)
            y1 = np.polynomial.polynomial.polyval(t, p1)
            p2 = _thor_wlsq(y1, T_tgt, d2, wgt)
            comp = np.polynomial.polynomial.polyval(y1, p2)
            err = np.abs(x * (comp + 0.5) - ref)
            e = float(err.max())
            if best is None or e < best[0]:
                best = (e, float(np.median(err)), p1, p2)
    e_max, e_med, p1, p2 = best
    return p1, p2, e_max, e_med


def fit_thor_gelu(samples, cfg):
    """One site's config entry (method='thor_composite')."""
    abs_s = samples.abs()
    q_hi = float(safe_quantile(abs_s, cfg.range_q_hi))
    S = float(max((1.0 + cfg.range_margin) * q_hi,
                  float(abs_s.max()) * cfg.raw_max_safety))
    p1, p2, e_max, e_med = _fit_thor_composite(
        S, int(cfg.d1), int(cfg.d2))
    log.info(f"[thor-gelu] S={S:8.3f}  d1={cfg.d1} d2={cfg.d2}"
          f"  max|err|={e_max:.3e}  med={e_med:.3e}")
    return {
        "method": "thor_composite",
        "xmax": S,
        "thor_p1": p1.tolist(),
        "thor_p2": p2.tolist(),
    }


def fit_cheb_gelu(X, rel_err=1e-5, max_deg=4095):
    """One Chebyshev series of GELU(X t) / X on t in [-1, 1] (method 'chebyshev', GELU_FOLD's input scale 1 / X): the
    lowest degree 2^k - 1 whose max error is at most rel_err X (the THOR composite cannot reach a wide range)."""
    from numpy.polynomial import chebyshev as C
    from scipy.special import erf
    f = lambda t: 0.5 * t * (1.0 + erf(X * t / math.sqrt(2.0)))
    t = np.linspace(-1.0, 1.0, 200001)
    deg = 127
    while True:
        c = C.chebinterpolate(f, deg)
        e = float(np.abs(C.chebval(t, c) - f(t)).max()) * X
        if e <= rel_err * X or deg >= max_deg:
            break
        deg = 2 * deg + 1
    log.info(f"[cheb-gelu] X={X:8.3f} deg={deg} max|err|={e:.3e}")
    return {"method": "chebyshev", "xmax": float(X), "cheb_coeffs": [float(v) for v in c], "cheb_a": -1.0,
            "cheb_b": 1.0, "cheb_eps": 1e-12, "fit_err_max": e}


def fit_split_gelu(unit_max, nch, safety=1.5, target=5e-4, rel_err=1e-5,
                   thor_degs=((31, 27), (63, 31))):
    """A range-split site (impl.config.GeluCfg perm / blocks) from each hidden unit's max |input|: the units sorted by
    range into output blocks of nch, block k fitted at S = safety x its widest unit by the first THOR composite of
    `thor_degs` meeting `target` (absolute), else by one Chebyshev series (fit_cheb_gelu). The site's own fields are
    its narrowest block's."""
    unit_max = np.asarray(unit_max, dtype=np.float64)
    if len(unit_max) % nch:
        raise ValueError(f"{len(unit_max)} hidden units are not whole blocks of {nch}")
    perm = np.argsort(unit_max, kind="stable")
    blocks = []
    for k in range(len(unit_max) // nch):
        S = float(safety * unit_max[perm[(k + 1) * nch - 1]])
        fit = None
        for d1, d2 in thor_degs:
            p1, p2, e_max, e_med = _fit_thor_composite(S, d1, d2)
            if e_max <= target:
                fit = {"method": "thor_composite", "xmax": S, "thor_p1": p1.tolist(), "thor_p2": p2.tolist(),
                       "fit_err_max": e_max}
                log.info(f"[split-gelu] block {k}: S={S:8.3f} THOR d1={d1} d2={d2} max|err|={e_max:.3e}")
                break
        blocks.append(fit or fit_cheb_gelu(S, rel_err))
    return {**blocks[0], "perm": perm.tolist(), "blocks": blocks}


def fit_gelu_section(gelu_samples, cfg):
    """The configs.json "softgelu" section: per-layer (or "global") THOR fits."""
    if cfg.per_layer:
        return {n: fit_thor_gelu(s, cfg) for n, s in sorted(gelu_samples.items())}
    all_g = torch.cat([gelu_samples[n] for n in sorted(gelu_samples)])
    return {"global": fit_thor_gelu(all_g, cfg)}


_GELU_CLASS_NAMES = {"GELU", "GELUActivation", "NewGELUActivation",
                     "FastGELUActivation", "QuickGELUActivation",
                     "PytorchGELUTanh", "GELUTanh"}


class _GeluCollector:
    """Flattened activation-input samples per site."""

    def __init__(self, cfg):
        del cfg
        self._bufs: dict[str, list] = {}

    def attach(self, name, module):
        def hook(module_, inp, output):
            del module_, output
            x = inp[0] if isinstance(inp, tuple) else inp
            if torch.is_tensor(x):
                self._bufs.setdefault(name, []).append(x.float().flatten().detach().cpu())

        return [module.register_forward_hook(hook)]

    def on_output(self, output):
        pass

    def finalize(self):
        return {n: torch.cat(parts) for n, parts in self._bufs.items()} or None


APPROX = register(Approximation(
    kind="gelu",
    section="softgelu",
    matches=lambda m: isinstance(m, nn.GELU) or type(m).__name__ in _GELU_CLASS_NAMES,
    make_collector=_GeluCollector,
    fit_section=fit_gelu_section,
))
