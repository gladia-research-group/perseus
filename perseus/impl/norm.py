"""LayerNorm from primitives (src/algorithms/norm/norm.cu with the cachemir helpers of
src/packing/cachemir/cachemir_norm_utils.cu and the affine of src/model/layer_norm.cu)."""
from __future__ import annotations

import math

from .config import NormCfg
from .layout import active_token_mask, lane_vec, lanes_mask, lanes_periodic, stride_mask
from .poly import (eval_chebyshev, eval_remez_31, eval_taylor_inv_sqrt, im_cleanse, inv_sqrt_newton,
                   inv_sqrt_newton_d2, rotsum, taylor_inv_sqrt_coeffs)


def _seed_mid(cfg: NormCfg, dr: float = 1.0) -> float:
    """A variance in the inverse-sqrt seed's band (for empty prefill lanes)."""
    if cfg.cheb_coeffs and cfg.cheb_hi > cfg.cheb_lo:
        return dr * 0.5 * (cfg.cheb_lo + cfg.cheb_hi)
    return dr * 1.0


def norm_lanes(rt, x, cfg: NormCfg, lanes):
    """norm() on every lane of a prefill chunk (`lanes`: absolute positions); empty lanes stay in the seed's band."""
    ops, d = rt.ops, rt.dims
    N, t, dr = d.N, d.t, d.dim
    cs = tuple(round(math.sqrt(cfg.c_eff_sq(p)), 12) for p in lanes)
    n = len(cs)
    x2 = im_cleanse(ops, x)
    xs = rt.mult_mask(x2, ("ln.center.L", cs), lambda: lanes_mask(N, dr, t, [0.5 * c for c in cs]))
    mean = ops.tag_reduce(rotsum(ops, xs, t, N), t)
    mean = rt.mult_mask(mean, ("ln.scalemask.L", n), lambda: lanes_mask(N, dr, t, [-1.0 / dr] * n))
    centered = ops.add(xs, mean)
    var = ops.square(centered)
    var = ops.tag_reduce(rotsum(ops, var, t, N), t)
    var = ops.mult(var, 1.0 / dr)
    var = rt.add_mask(var, ("ln.eps.L", cfg.epsilon, cs, _seed_mid(cfg)),
                      lambda: lanes_periodic(N, t, [cfg.epsilon * c * c for c in cs], _seed_mid(cfg)))
    if cfg.cheb_coeffs and ops.ln_cheb:
        init = eval_chebyshev(ops, var, cfg.cheb_coeffs, cfg.cheb_lo, cfg.cheb_hi)
        inv = inv_sqrt_newton_d2(ops, var, init, cfg.cheb_nr_iters,
                                 1.0 / (cfg.inv_out_scale * cfg.inv_out_scale))
    elif cfg.method == "remez":
        init = eval_remez_31(ops, var, cfg.Ncoeffs, cfg.Dcoeffs, cfg.lin_alpha, cfg.lin_beta, cfg.gs_iters)
        inv = inv_sqrt_newton(ops, var, init, cfg.nr_iters, 1.0 / (cfg.inv_out_scale * cfg.inv_out_scale))
    else:
        init = eval_taylor_inv_sqrt(ops, var, taylor_inv_sqrt_coeffs(cfg.taylor_z0), cfg.taylor_z0)
        inv = inv_sqrt_newton(ops, var, init, cfg.nr_iters, 1.0)
    ops.bootstrap_hint(inv, ops.headroom(4))
    return ops.mult(centered, inv)


def rms_norm_lanes(rt, x, cfg: NormCfg, lanes):
    """rms_norm() on every token lane of a prefill chunk (see norm_lanes)."""
    ops, d = rt.ops, rt.dims
    N, t, dr = d.N, d.t, d.dim
    if not cfg.cheb_coeffs:
        raise ValueError("rms_norm: the config has no Chebyshev seed (cheb_coeffs)")
    cs = tuple(round(math.sqrt(cfg.c_eff_sq(p)), 12) for p in lanes)
    x2 = im_cleanse(ops, x)
    xs = rt.mult_mask(x2, ("ln.center.L", cs), lambda: lanes_mask(N, dr, t, [0.5 * c for c in cs]))
    S = ops.tag_reduce(rotsum(ops, ops.square(xs), t, N), t)
    S = rt.add_mask(S, ("rms.eps.L", cfg.epsilon, cs, _seed_mid(cfg, dr)),
                    lambda: lanes_periodic(N, t, [dr * cfg.epsilon * c * c for c in cs], _seed_mid(cfg, dr)))
    init = eval_chebyshev(ops, S, cfg.cheb_coeffs, dr * cfg.cheb_lo, dr * cfg.cheb_hi)
    inv = inv_sqrt_newton_d2(ops, S, init, cfg.cheb_nr_iters,
                             1.0 / (dr * cfg.inv_out_scale * cfg.inv_out_scale))
    ops.bootstrap_hint(inv, ops.headroom(4))
    return ops.mult(xs, inv)


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
    if getattr(ops, "realize_shared", False):
        ops.realize(centered)                             # read by the square and the final product only
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
    # inverse sqrt: init + Newton (norm.cu); under LN_CHEB a cheb_* config replaces the Remez
    # seed with a Chebyshev one on the data band
    if cfg.cheb_coeffs and ops.ln_cheb:
        init = eval_chebyshev(ops, var, cfg.cheb_coeffs, cfg.cheb_lo, cfg.cheb_hi)
        inv = inv_sqrt_newton_d2(ops, var, init, cfg.cheb_nr_iters,
                                 1.0 / (cfg.inv_out_scale * cfg.inv_out_scale))
        ops.bootstrap_hint(inv, ops.headroom(4))
        return ops.mult(centered, inv)
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


def rms_norm(rt, x, cfg: NormCfg, pos: int):
    """RMSNorm without its gamma (the next linear carries gamma / inv_out_scale):
    inv_out_scale * x / sqrt(mean(x^2) + eps) on the lanes. The norm() pipeline minus the mean:
    the centering scale c, the all-slot ladder of (c x)^2, and the Chebyshev seed + Newton
    inverse sqrt of z = c^2 (mean(x^2) + eps) (LN_CHEB's path; the config carries no Remez
    seed). The 1/d of the mean rides the seed's interval and the Newton constant, so the sum
    S = sum (c x)^2 + d eps c^2 enters both directly (examples/rmsnorm_from_primitives.py)."""
    ops, d = rt.ops, rt.dims
    N, t, dr = d.N, d.t, d.dim
    if not cfg.cheb_coeffs:
        raise ValueError("rms_norm: the config has no Chebyshev seed (cheb_coeffs)")
    c2 = cfg.c_eff_sq(pos)
    c = math.sqrt(c2)
    x2 = im_cleanse(ops, x)
    xs = rt.mult_mask(x2, ("ln.center", c), lambda: active_token_mask(N, dr, t, 0.5 * c))
    S = ops.tag_reduce(rotsum(ops, ops.square(xs), 1, N), 1)       # broadcast constant
    S = ops.add(S, dr * cfg.epsilon * c2)
    init = eval_chebyshev(ops, S, cfg.cheb_coeffs, dr * cfg.cheb_lo, dr * cfg.cheb_hi)
    inv = inv_sqrt_newton_d2(ops, S, init, cfg.cheb_nr_iters,
                             1.0 / (dr * cfg.inv_out_scale * cfg.inv_out_scale))
    ops.bootstrap_hint(inv, ops.headroom(4))
    return ops.mult(xs, inv)


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
