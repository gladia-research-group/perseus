"""THOR composite GELU (src/algorithms/nonlinear/nonlinear.cu):
GELU(x) = x * (P2(P1(x / xmax)) + 1/2) with the conj-doubling absorbed by a 0.5 mask that
also zeroes the padded MLP lanes."""
from __future__ import annotations

from .config import GeluCfg
from .layout import active_expanded_mask
from .poly import eval_chebyshev, eval_polynomial_ps, im_cleanse


def gelu_thor_core(rt, t, cfg: GeluCfg):
    """nonlinear.cu."""
    ops = rt.ops
    ops.bootstrap_hint(t, 16)
    cheb = bool(cfg.thor_p1_cheb)
    p1 = (eval_chebyshev(ops, t, cfg.thor_p1_cheb, cfg.thor_p1_a, cfg.thor_p1_b) if cheb
          else eval_polynomial_ps(ops, t, cfg.thor_p1))
    ops.bootstrap_hint(p1, ops.headroom(4))
    g = (eval_chebyshev(ops, p1, cfg.thor_p2_cheb, cfg.thor_p2_a, cfg.thor_p2_b) if cheb
         else eval_polynomial_ps(ops, p1, cfg.thor_p2))
    g = ops.add(g, 0.5)
    if ops.unit > 1:
        ops.bootstrap_hint(g, ops.level_limit() - 4)
    return g


def gelu(rt, x, cfg: GeluCfg):
    """nonlinear.cu: gelu_thor_composite, or gelu_chebyshev (plain Chebyshev)."""
    ops, d = rt.ops, rt.dims
    if cfg.method == "chebyshev":
        ops.bootstrap_hint(x, ops.headroom(4))
        return eval_chebyshev(ops, x, cfg.cheb_coeffs, cfg.cheb_a, cfg.cheb_b)
    if cfg.method != "thor_composite":
        raise ValueError(f"gelu: method {cfg.method!r} not ported")
    t = ops.mult(x, 1.0 / cfg.xmax)
    g = gelu_thor_core(rt, t, cfg)
    xs = im_cleanse(ops, x)
    g = ops.mult(g, xs)
    return rt.mult_mask(g, "gelu.half", lambda: active_expanded_mask(d, 0.5))
