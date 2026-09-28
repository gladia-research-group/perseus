"""LayerNorm from primitives (src/algorithms/norm/norm.cu with the cachemir helpers of
src/packing/cachemir/cachemir_norm_utils.cu and the affine of src/model/layer_norm.cu)."""
from __future__ import annotations

import math

from .config import NormCfg
from .layout import active_token_mask, lane_vec, stride_mask
from .poly import (eval_remez_31, eval_taylor_inv_sqrt, im_cleanse, inv_sqrt_newton, rotsum,
                   taylor_inv_sqrt_coeffs)


def norm(rt, x, cfg: NormCfg, pos: int):
    """norm(): returns inv_out_scale * LN(x) on the lanes (no gamma/beta); pos is the
    absolute token position (capture_t) selecting center_scale_sq."""
    ops, d = rt.ops, rt.dims
    N, t, dr = d.N, d.t, d.dim
    c2 = cfg.c_eff_sq(pos)
    c = math.sqrt(c2)
    # im_cleanse + centering-scale mask (norm.cu)
    x2 = im_cleanse(ops, x)
    xs = rt.mult_mask(x2, ("ln.center", c), lambda: active_token_mask(N, dr, t, 0.5 * c))
    # mean: mod-t class sum (cachemir_norm_utils.cu) times -1/d on the lanes (norm.cu)
    mean = ops.tag_reduce(rotsum(ops, xs, t, N), t)       # t-periodic (mod-t class sum)
    mean = rt.mult_mask(mean, "ln.scalemask", lambda: stride_mask(N, dr, t, -1.0 / dr))
    centered = ops.add(xs, mean)
    # biased variance broadcast to every slot (cachemir_norm_utils.cu) + eps*c^2.
    # FUSED_LN_VAR: the square may sit at the ceiling (no reactive refresh), the all-slot
    # ladder stops at the sparse slot count and the fold bootstrap finishes it and divides
    # by the real width in one refresh
    if ops.fused_ln_var:
        with ops.suppress_auto_bts():
            var = ops.square(centered)
        s_eff = ops.fold_slots_for(1)
        if s_eff > 1:
            var = rotsum(ops, var, 1, s_eff)
        ops.fold_bootstrap(var, s_eff, dr)
        var = ops.tag_reduce(var, 1)
    else:
        var = ops.square(centered)
        var = ops.tag_reduce(rotsum(ops, var, 1, N), 1)   # broadcast constant
        var = ops.mult(var, 1.0 / dr)
    var = ops.add(var, cfg.epsilon * c2)
    # inverse sqrt: init + Newton (norm.cu)
    if cfg.method == "remez":
        init = eval_remez_31(ops, var, cfg.Ncoeffs, cfg.Dcoeffs, cfg.lin_alpha, cfg.lin_beta,
                             cfg.gs_iters)
        nx_scale = 1.0 / (cfg.inv_out_scale * cfg.inv_out_scale)
    else:
        init = eval_taylor_inv_sqrt(ops, var, taylor_inv_sqrt_coeffs(cfg.taylor_z0), cfg.taylor_z0)
        nx_scale = 1.0
    inv = inv_sqrt_newton(ops, var, init, cfg.nr_iters, nx_scale)
    ops.bootstrap_hint(inv, ops.headroom(4))
    return ops.mult(centered, inv)


def _lane(rt, tag, part, v):
    """`lane_vec(v)` memoised per site. The slot plaintext behind it is encoded once, but the
    numpy lane vector was being rebuilt (a scatter over N slots) at every visit."""
    d = rt.dims
    if tag is None:
        return lane_vec(v, d.N, d.t)
    return rt.mask((tag, part), lambda: lane_vec(v, d.N, d.t))


def ln_affine(rt, normed, gamma_desc, beta, tag=None):
    """layer_norm.cu: normed * gamma + beta with the per-feature lane layout;
    gamma_desc is gamma / inv_out_scale (weight_loader.h)."""
    ops = rt.ops
    y = ops.mult_pt(normed, _lane(rt, tag, "gamma", gamma_desc),
                    key=None if tag is None else tag + ".gamma")
    return ops.add_pt(y, _lane(rt, tag, "beta", beta),
                      key=None if tag is None else tag + ".beta")


def ln_shift(rt, normed, shift, tag=None):
    """The folded form (layer_norm.cu): beta/gamma rides the normalized input."""
    return rt.ops.add_pt(normed, _lane(rt, tag, "shift", shift),
                         key=None if tag is None else tag + ".shift")


def layer_norm(rt, x, cfg: NormCfg, pos: int, *, shift=None, gamma_desc=None, beta=None, tag=None):
    """layer_norm.cu: folded (shift) or unfolded (gamma/beta)."""
    normed = norm(rt, x, cfg, pos)
    if shift is not None:
        return ln_shift(rt, normed, shift, tag)
    rt.ops.fhe.maybe_bootstrap(normed)
    return ln_affine(rt, normed, gamma_desc, beta, tag)


def norm_step_masks(rt, cfg: NormCfg, pos: int):
    """The per-step mask of a norm site: the centering scale of position `pos` (norm.cu;
    a run-constant mask when the config has no per-position center_scale_sq)."""
    d = rt.dims
    c = math.sqrt(cfg.c_eff_sq(pos))
    return [(("ln.center", c), lambda: active_token_mask(d.N, d.dim, d.t, 0.5 * c))]
