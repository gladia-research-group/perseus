"""THOR composite GELU (src/algorithms/nonlinear/nonlinear.cu):
GELU(x) = x * (P2(P1(x / xmax)) + 1/2) with the conj-doubling absorbed by a 0.5 mask that
also zeroes the padded MLP lanes."""
from __future__ import annotations

import os

import numpy as np

from .config import GeluCfg
from .layout import active_expanded_lanes, active_expanded_mask
from .poly import eval_chebyshev, eval_polynomial_ps, im_cleanse


# PF_ACT_FOLD=1: fold a constant output mask into the THOR coefficients (changes the captured op stream)
_ACT_FOLD = os.environ.get("PF_ACT_FOLD", "0") not in ("", "0")
# PF_GELU_TFHE=1: the folded GELU as a TFHE LUT (impl/tfhe.py) on the active MLP lanes instead of its polynomial (an
# experiment: the switch's input noise is too large for GELU)
_GELU_TFHE = os.environ.get("PF_GELU_TFHE", "0") not in ("", "0")
# PF_GELU_TFHE=2: the LUT reads Phi (GELU = x Phi(x)) and one CKKS product makes GELU: the switch's input noise then
# enters as x phi(x) sigma (<= 0.24 sigma) instead of GELU's slope (<= 1.13 sigma)
_GELU_TFHE_PHI = os.environ.get("PF_GELU_TFHE", "0") == "2"


def gelu_tfhe(rt, x, cfg: GeluCfg, lanes=None):
    """GELU(x) = xmax f(x / xmax), f(m) = GELU(m xmax) / xmax read by a TFHE LUT on the active lanes (|f| <= 1 on
    [-1, 1)); x arrives as x / xmax (the up-projection carries it), xmax rides the lane mask as in the Chebyshev
    branch. Every other slot comes back zero (the imaginary halves too)."""
    from . import tfhe
    ops, d = rt.ops, rt.dims
    from math import erf, sqrt
    xm = float(cfg.xmax)
    phi = np.vectorize(lambda y: 0.5 * (1.0 + erf(y / sqrt(2.0))))
    name = f"{'phi' if _GELU_TFHE_PHI else 'gelu'}/{xm:.9g}"
    if name not in tfhe.LUTS:
        if _GELU_TFHE_PHI:
            tfhe.register_lut(name, lambda m, xm=xm: phi(np.asarray(m) * xm))
        else:
            tfhe.register_lut(name, lambda m, xm=xm: np.asarray(m) * phi(np.asarray(m) * xm))
    L = () if lanes is None else ("L", int(lanes))
    act = np.nonzero(np.real(_expanded(d, 1.0, lanes)))[0]
    sl = tfhe.register_slots(f"mlp.lanes{L}", act)
    y = tfhe.tfhe_lut(ops, x, name, sl)
    if _GELU_TFHE_PHI:
        y = ops.mult(x, y)
    return rt.mult_mask(y, ("gelu.tfhe", xm) + L, lambda: _expanded(d, xm, lanes))

def gelu_thor_core(rt, t, cfg: GeluCfg, out_scale: float | None = None):
    """nonlinear.cu. With GELU_FOLD the refreshes are the plan's, not these hints. `out_scale` (Chebyshev THOR only):
    out_scale * core, folded into the coefficients."""
    ops = rt.ops
    hints = not ops.gelu_fold
    if hints:
        ops.bootstrap_hint(t, 16)
    cheb = bool(cfg.thor_p1_cheb)
    if out_scale is not None and cheb:
        a2, b2 = cfg.thor_p2_a, cfg.thor_p2_b
        al, be = 2.0 / (b2 - a2), (a2 + b2) / (b2 - a2)
        c1 = [al * float(c) for c in cfg.thor_p1_cheb]
        c1[0] -= be
        p1 = eval_chebyshev(ops, t, c1, cfg.thor_p1_a, cfg.thor_p1_b)
        g = eval_chebyshev(ops, p1, [out_scale * float(c) for c in cfg.thor_p2_cheb], -1.0, 1.0)
        return ops.add(g, 0.5 * out_scale)
    p1 = (eval_chebyshev(ops, t, cfg.thor_p1_cheb, cfg.thor_p1_a, cfg.thor_p1_b) if cheb
          else eval_polynomial_ps(ops, t, cfg.thor_p1))
    if hints:
        ops.bootstrap_hint(p1, ops.headroom(4))
    g = (eval_chebyshev(ops, p1, cfg.thor_p2_cheb, cfg.thor_p2_a, cfg.thor_p2_b) if cheb
         else eval_polynomial_ps(ops, p1, cfg.thor_p2))
    g = ops.add(g, 0.5)
    if out_scale is not None:
        g = ops.mult(g, out_scale)
    if hints and ops.unit > 1:
        ops.bootstrap_hint(g, ops.level_limit() - 4)
    return g


def _expanded(d, scale, lanes):
    return active_expanded_mask(d, scale) if lanes is None else active_expanded_lanes(d, scale, lanes)


def gelu(rt, x, cfg: GeluCfg, lanes: int | None = None, mask=None):
    """nonlinear.cu: gelu_thor_composite, or gelu_chebyshev (plain Chebyshev). `lanes`: prefill lanes filled in the
    expanded layout; `mask`: (key, build(scale)) replacing the expanded-layout mask."""
    ops, d = rt.ops, rt.dims
    L = () if lanes is None else ("L", int(lanes))
    if mask is not None and ops.gelu_fold and cfg.method == "thor_composite":
        if _ACT_FOLD:
            m = np.asarray(mask[1](1.0))
            if np.all(m == m.flat[0]):      # constant mask: rides the core's coefficients
                g = gelu_thor_core(rt, x, cfg, out_scale=0.5 * cfg.xmax * float(np.real(m.flat[0])))
                return ops.mult(g, im_cleanse(ops, x))
        g = ops.mult(gelu_thor_core(rt, x, cfg), im_cleanse(ops, x))
        return rt.mult_mask(g, (mask[0], cfg.xmax), lambda: mask[1](0.5 * cfg.xmax))
    if _GELU_TFHE and cfg.method == "chebyshev" and ops.gelu_fold:
        return gelu_tfhe(rt, x, cfg, lanes)
    if cfg.method == "chebyshev":
        if ops.gelu_fold:   # x arrives as x / xmax (the up-projection carries 1/xmax): GELU(x) = xmax q(x / xmax),
            # q the Chebyshev series on [cheb_a, cheb_b]; xmax rides the lane mask, the refreshes are the plan's
            g = eval_chebyshev(ops, x, cfg.cheb_coeffs, cfg.cheb_a, cfg.cheb_b, cfg.cheb_eps)
            return rt.mult_mask(g, ("gelu.cheb", cfg.xmax) + L, lambda: _expanded(d, cfg.xmax, lanes))
        ops.bootstrap_hint(x, ops.headroom(4))
        return eval_chebyshev(ops, x, cfg.cheb_coeffs, cfg.cheb_a, cfg.cheb_b, cfg.cheb_eps)
    if cfg.method != "thor_composite":
        raise ValueError(f"gelu: method {cfg.method!r} not ported")
    if ops.gelu_fold:   # x arrives as x / xmax (the up-projection carries it): xmax rides the mask
        g = ops.mult(gelu_thor_core(rt, x, cfg), im_cleanse(ops, x))
        return rt.mult_mask(g, ("gelu.half", cfg.xmax) + L, lambda: _expanded(d, 0.5 * cfg.xmax, lanes))
    t = ops.mult(x, 1.0 / cfg.xmax)
    g = gelu_thor_core(rt, t, cfg)
    xs = im_cleanse(ops, x)
    g = ops.mult(g, xs)
    return rt.mult_mask(g, ("gelu.half",) + L if L else "gelu.half", lambda: _expanded(d, 0.5, lanes))
