"""The plaintext mirror: the same polynomials and iteration schedules as the FHE port, on
plain float64 vectors (one token = [768]). ``*_ref`` applies the approximation the ciphertext
path applies, ``*_exact`` the closed form, so ``fhe - ref`` is FHE noise and ``ref - exact``
approximation error (examples/rmsnorm_from_primitives.py convention).

Doubling conventions: im_cleanse doubles a real payload and every C++ site pairs it with a 0.5
mask; the mirror applies both, i.e. nothing."""
from __future__ import annotations

import math

import numpy as np

from perseus.impl.config import CutMaxCfg, GeluCfg, NormCfg, SoftmaxCfg
from perseus.impl.poly import (NumpyOps, eval_chebyshev, eval_polynomial_ps, eval_remez_31,
                   eval_taylor_inv_sqrt, goldschmidt_inv_ndf, goldschmidt_inv_x0,
                   inv_sqrt_newton, inv_sqrt_newton_safe, pow_odd, taylor_inv_sqrt_coeffs)

_ops = NumpyOps()


def norm_ref(x, cfg: NormCfg, pos: int):
    """norm(): inv_out_scale * LN(x) with the calibrated centering scale and the Remez/Newton
    inverse sqrt (norm.py)."""
    x = np.asarray(x, dtype=np.float64)
    c2 = cfg.c_eff_sq(pos)
    xs = math.sqrt(c2) * x
    centered = xs - xs.mean()
    var = np.array([np.mean(centered * centered) + cfg.epsilon * c2])
    if cfg.method == "remez":
        init = eval_remez_31(_ops, var, cfg.Ncoeffs, cfg.Dcoeffs, cfg.lin_alpha, cfg.lin_beta,
                             cfg.gs_iters)
        nx = 1.0 / (cfg.inv_out_scale ** 2)
    else:
        init = eval_taylor_inv_sqrt(_ops, var, taylor_inv_sqrt_coeffs(cfg.taylor_z0), cfg.taylor_z0)
        nx = 1.0
    inv = inv_sqrt_newton(_ops, var, init, cfg.nr_iters, nx)
    return centered * inv[0]


def norm_exact(x, cfg: NormCfg):
    x = np.asarray(x, dtype=np.float64)
    c = x - x.mean()
    return cfg.inv_out_scale * c / np.sqrt(np.mean(c * c) + cfg.epsilon)


def layer_norm_ref(x, cfg, pos, gamma, beta):
    """The unfolded affine on the descaled gamma: LN(x)*gamma + beta."""
    return norm_ref(x, cfg, pos) * (gamma * cfg.descale) + beta


def gelu_ref(x, cfg: GeluCfg):
    x = np.asarray(x, dtype=np.float64)
    if cfg.method == "chebyshev":
        return eval_chebyshev(_ops, x, cfg.cheb_coeffs, cfg.cheb_a, cfg.cheb_b)
    t = x / cfg.xmax
    if cfg.thor_p1_cheb:
        p1 = eval_chebyshev(_ops, t, cfg.thor_p1_cheb, cfg.thor_p1_a, cfg.thor_p1_b)
        g = eval_chebyshev(_ops, p1, cfg.thor_p2_cheb, cfg.thor_p2_a, cfg.thor_p2_b)
    else:
        p1 = eval_polynomial_ps(_ops, t, cfg.thor_p1)
        g = eval_polynomial_ps(_ops, p1, cfg.thor_p2)
    return x * (g + 0.5)


def gelu_exact(x):
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def softmax_ref(scores, cfg: SoftmaxCfg, kc: int):
    """THOR softmax over the LAST axis of scores[..., kc] (attention.softmax_thor)."""
    s = np.asarray(scores, dtype=np.float64)
    mean = (cfg.clip_hi + cfg.clip_lo) / 2.0
    ct = s - mean
    sf = 2.0 ** (-cfg.log2delta1 - cfg.log2delta2)
    z = eval_chebyshev(_ops, ct, cfg.cheb_coeffs, cfg.cheb_a / sf, cfg.cheb_b / sf)
    for _ in range(cfg.log2delta1):
        z = z * z
    z = z / kc                                   # im_cleanse (x2) then the 0.5/kc active mask
    tot = z.sum(axis=-1, keepdims=True) * np.ones_like(z)
    F = cfg.init_alpha - cfg.init_beta * tot
    y = goldschmidt_inv_ndf(_ops, z, tot, F, cfg.gs_iters_scaled)
    for i in range(cfg.log2delta2):
        y = 2.0 * y                              # im_cleanse
        z = (y * y) * (0.5 * math.sqrt(kc) * 0.25)
        r = cfg.kc_r(i, kc)
        tot = z.sum(axis=-1, keepdims=True) * np.ones_like(z)
        F = cfg.refine_alpha[i] * math.sqrt(r) - cfg.refine_beta[i] * r * tot
        y = goldschmidt_inv_ndf(_ops, z, tot, F, int(cfg.per_step_refine_iters[i]))
    return 2.0 * y * 0.5                         # softmax_v's im_cleanse and tok0 half mask


def softmax_exact(scores):
    s = np.asarray(scores, dtype=np.float64)
    e = np.exp(s - s.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def attention_ref(q, K, V, cfg: SoftmaxCfg, H_real: int, d_head: int, softmax=softmax_ref):
    """q [hid] (head-interleaved feature r = h + H*lane), K/V [kc, hid] in the same order:
    the per-head attention output, head-interleaved again (what softmax_v emits)."""
    kc = K.shape[0]
    H = q.shape[0] // d_head
    out = np.zeros_like(q)
    for h in range(H_real):
        idx = h + H * np.arange(d_head)
        sc = (K[:, idx] @ q[idx]) / math.sqrt(d_head)
        p = softmax(sc, cfg, kc) if softmax is softmax_ref else softmax(sc)
        out[idx] = p @ V[:, idx]
    return out


def cutmax_ref(logits, cfg: CutMaxCfg):
    """cutmax_argmax on a plaintext logit vector (head.cutmax_argmax with bootstraps = identity)."""
    x = np.asarray(logits, dtype=np.float64) * cfg.entry_scale
    n = x.shape[0]
    for it in cfg.iters:
        kf = 1.0 / math.sqrt(n * it.s2_hi)
        R = x.sum()
        cen = x * kf if it.ex2 else x * kf - R * kf / n
        S2 = np.array([np.sum(cen * cen)])
        ones = np.ones(1)
        u_prod = None
        xx = S2
        for j in range(it.passes):
            iters = cfg.newton_per_pass + (cfg.newton_polish if j + 1 == it.passes else 0)
            y0 = (it.ca - it.cb * xx) if (j == 0 and it.ca != 0.0) else ones
            u = inv_sqrt_newton_safe(_ops, xx, y0, iters)
            if j + 1 < it.passes:
                xx = xx * u * u
            u_prod = u if j == 0 else u_prod * u
        f = (u_prod * (0.5 / (math.sqrt(it.s2_hi) * it.c * it.m)))[0]
        y = 2.0 * (x * f - R * f / n + 0.5 / it.m)      # im_cleanse doubles
        x = pow_odd(_ops, y, it.p)
    S = np.array([x.sum()])
    g = 1.0 / math.sqrt(cfg.sum_lo * cfg.sum_hi)
    lo, hi = cfg.sum_lo * g, cfg.sum_hi * g
    bsum = 8.0 / ((lo + hi) ** 2 + 4.0 * lo * hi)
    Sn = S * g
    F = bsum * (lo + hi) - bsum * Sn
    r = goldschmidt_inv_x0(_ops, Sn, F, cfg.gs_sum_iters) * g
    return x * r[0]
