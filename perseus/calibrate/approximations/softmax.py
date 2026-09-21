"""THOR softmax fit
"""

import math

import numpy as np
import torch
import torch.nn as nn

from perseus.calibrate.numerics import (
    GS_INITS,
    cheb_exp_series,
    estimate_gs_iters,
    fit_gs_under_noise,
    get_chebyshev_nodes,
    polyval_torch,
    safe_quantile,
)
from perseus.calibrate.registry import Approximation, register


def _fit_exp_poly(delta0, degree):
    x = get_chebyshev_nodes(-0.5 * delta0, 0.5 * delta0, max(8 * (degree + 1), 200))
    return np.polyfit(x, np.exp(x), degree)[::-1].astype(np.float64)


@torch.no_grad()
def _phase1_val(rows, valid, mid, scale, coeffs, n_sq):
    y = ((rows - mid) / scale).double()
    val = polyval_torch(y, coeffs)
    for _ in range(n_sq):
        val = val * val
    return val * valid.double()


@torch.no_grad()
def _refine_denominators(val, ref_iters, k_eff, row_q, C, q):
    kc = (row_q + 1).long()
    sqrt_k = k_eff.sqrt()
    S_steps, S_kc = [], []
    sum_exp = val.sum(dim=-1, keepdim=True)
    for _ in range(ref_iters):
        val = (val * sum_exp.clamp(min=1e-30).reciprocal()) ** 2
        sum_exp = val.sum(dim=-1, keepdim=True)
        s = sum_exp.flatten() * sqrt_k          # [n_rows]
        S_steps.append(s[s > 0])
        S_kc.append([
            float(torch.quantile(s[kc == k].double(), q)) if (kc == k).any()
            else float("nan")
            for k in range(1, C + 1)
        ])
    return S_steps, S_kc


def _next_pow2(x):
    if x <= 1.0:
        return 1
    return 1 << math.ceil(math.log2(x))


def _derive_per_layer_delta(M, cfg):
    delta2 = _next_pow2(M / (2.0 * math.log(cfg.magnitude_cap)))
    delta2 = min(max(delta2, int(cfg.delta2_min)), int(cfg.delta2_max))
    delta1 = _next_pow2(M / (cfg.target_delta0 * delta2))
    delta1 = min(max(delta1, int(cfg.delta1_min)), int(cfg.delta1_max))
    return delta1, delta2


def fit_softmax(rows, row_q, T, cfg):
    win = int(cfg.decode_window) if cfg.decode_window else 0
    if cfg.bidirectional:
        win = 0
        row_q = torch.full_like(row_q, T - 1)
    if 0 < win < T:
        keep = row_q < win
        rows = rows[keep]
        row_q = row_q[keep]

    valid = torch.arange(T).unsqueeze(0) <= row_q.unsqueeze(1)
    flat = rows[valid]
    a_obs = safe_quantile(flat, cfg.range_q_lo)
    b_obs = safe_quantile(flat, cfg.range_q_hi)
    pad = cfg.range_margin * (b_obs - a_obs)
    a, b = a_obs - pad, b_obs + pad
    M, mid = b - a, 0.5 * (a + b)

    rows = rows.where(valid, rows.new_tensor(float(mid)))

    if cfg.per_layer and not cfg.fixed_delta_per_layer:
        delta1, delta2 = _derive_per_layer_delta(float(M), cfg)
    else:
        delta1, delta2 = cfg.delta1, cfg.delta2

    n_sq = int(round(math.log2(delta1)))
    ref_iters = int(round(math.log2(delta2)))
    scale = float(delta1 * delta2)
    delta0 = float(cfg.delta0) if cfg.delta0 is not None else M / scale

    coeffs = torch.tensor(_fit_exp_poly(delta0, cfg.poly_degree), dtype=torch.float64)
    cheb_coeffs = cheb_exp_series(-0.5 * delta0, 0.5 * delta0, cfg.poly_degree)
    val1 = _phase1_val(rows, valid, mid, scale, coeffs, n_sq)
    k_eff = (row_q + 1).double()
    S_mean = val1.sum(-1) / k_eff
    S_mean = S_mean[S_mean > 0]

    d_safety = cfg.d_safety
    init_d_min = safe_quantile(S_mean, cfg.init_q_lo) / d_safety
    init_d_max = safe_quantile(S_mean, cfg.init_q_hi) * d_safety

    gs_init_fn = GS_INITS[cfg.gs_init_method]
    eps = float(cfg.chain_noise)
    if eps > 0.0:
        init_alpha, init_beta, gs_iters_scaled, _e = fit_gs_under_noise(
            S_mean, init_d_min, init_d_max, cfg.gs_max_iters, cfg.gs_init_method, eps,
            iters=(int(cfg.gs_iters) if cfg.gs_iters is not None else None),
        )
    else:
        init_alpha, init_beta = gs_init_fn(init_d_min, init_d_max)
        if cfg.gs_iters is not None:
            gs_iters_scaled = int(cfg.gs_iters)
        else:
            gs_iters_scaled = estimate_gs_iters(
                S_mean, init_d_min, init_d_max,
                cfg.gs_target_err, cfg.gs_max_iters, cfg.gs_init_method,
            )

    C = win if win else int(T)
    refine_S, S_kc = _refine_denominators(
        val1, ref_iters, k_eff, row_q, C, cfg.init_q_hi)
    refine_alpha, refine_beta, per_step_iters = [], [], []
    for s in refine_S:
        d_min_i = safe_quantile(s, cfg.init_q_lo) / d_safety
        d_max_i = safe_quantile(s, cfg.init_q_hi) * d_safety
        d_max_fit = d_max_i * cfg.refine_drift_margin
        if eps > 0.0:
            a_i, b_i, iters_i, _e = fit_gs_under_noise(
                s, d_min_i, d_max_fit, cfg.gs_max_iters, cfg.gs_init_method, eps)
        else:
            iters_i = estimate_gs_iters(
                s, d_min_i, d_max_fit,
                cfg.gs_target_err, cfg.gs_max_iters, cfg.gs_init_method,
            )
            a_i, b_i = gs_init_fn(d_min_i, d_max_fit)
        refine_alpha.append(a_i)
        refine_beta.append(b_i)
        per_step_iters.append(iters_i)
    finite = [it for it in per_step_iters if it is not None]
    if not per_step_iters:
        gs_iters_refine_scaled = 0
    elif len(finite) < len(per_step_iters):
        gs_iters_refine_scaled = None
    else:
        gs_iters_refine_scaled = max(finite)

    denom_floor = float(cfg.denom_floor_frac) * float(init_d_min)

    anchor = max(1, min(int(cfg.sm_kc_anchor), C))
    if cfg.bidirectional:   # uniform kc: the refine bands ARE the fit — no per-kc shrink
        sm_kc_r = [1.0] * (ref_iters * C)
    else:
        sm_kc_r = []
        for i in range(ref_iters):
            lows = [S_kc[i][k] for k in range(anchor)
                    if S_kc[i][k] == S_kc[i][k] and S_kc[i][k] > 0]
            s_ref = max(lows) if lows else 1.0
            last = 1.0
            for k in range(C):
                kc = k + 1
                st = S_kc[i][k]
                if kc <= anchor:
                    last = 1.0
                elif st == st and st > 0.0:
                    st_eff = st * (1.0 + cfg.sm_kc_tail_margin)
                    last = min(1.0, (s_ref / st_eff) ** 2)
                sm_kc_r.append(last)

    return {
        "mid_x": mid,
        "input_scale": scale,
        "squeeze_bound": 0.5 * M,
        "clip_lo": a,
        "clip_hi": b,
        "n_squarings": n_sq,
        "refinement_iters": ref_iters,
        "poly_coeffs": coeffs.tolist(),
        "cheb_coeffs": cheb_coeffs.tolist(),
        "cheb_a": -0.5 * delta0,
        "cheb_b":  0.5 * delta0,
        "init_alpha": init_alpha,
        "init_beta":  init_beta,
        "refine_alpha": refine_alpha,
        "refine_beta":  refine_beta,
        "denom_floor": denom_floor,
        "gs_iters_scaled": gs_iters_scaled,
        "gs_iters_refine_scaled": gs_iters_refine_scaled,
        "per_step_refine_iters": per_step_iters,
        "sm_kc_r": sm_kc_r,
        "delta0": delta0,
        "delta1": int(delta1),
        "delta2": int(delta2),
        "decode_window": win,
        "calib_T": int(T),
    }


def fit_softmax_section(attn_scores, cfg):
    """The configs.json "softmax" section: per-site (or "global") THOR fits."""
    if cfg.per_layer:
        return {
            n: fit_softmax(r, rq, T, cfg)
            for n, (r, rq, T) in sorted(attn_scores.items())
        }
    Ts = {T for _, _, T in attn_scores.values()}
    if len(Ts) != 1:
        raise RuntimeError(f"mixed T across attention layers: {Ts}")
    T = Ts.pop()
    rows = torch.cat([r for r, _, _ in attn_scores.values()], dim=0)
    row_q = torch.cat([rq for _, rq, _ in attn_scores.values()], dim=0)
    return {"global": fit_softmax(rows, row_q, T, cfg)}


class _SoftmaxTapHandle:
    """Removable handle restoring the patched `nn.functional.softmax`."""

    def __init__(self, collector):
        self._collector = collector

    def remove(self):
        self._collector._unpatch()


class _SoftmaxCollector:

    def __init__(self, cfg):
        self.rows_per_batch = cfg.rows_per_batch
        self._bufs: dict[str, dict] = {}
        self._stack: list[str] = []
        self._orig = None

    def _record(self, site, x):
        T = x.shape[-1]
        scores2d = x.detach().reshape(-1, T)
        keep = min(self.rows_per_batch, scores2d.size(0))
        idx = torch.randperm(scores2d.size(0), device=scores2d.device)[:keep]
        e = self._bufs.setdefault(site, {"rows": [], "row_q": [], "T": T})
        e["rows"].append(scores2d[idx].detach())
        e["row_q"].append((idx % T).detach())

    def _patch(self):
        if self._orig is not None:
            return
        self._orig = nn.functional.softmax

        def tapped(input, *args, **kwargs):
            dim = kwargs.get("dim", args[0] if args else None)
            if (self._stack and input.dim() >= 2
                    and input.shape[-1] == input.shape[-2]
                    and dim in (-1, input.dim() - 1)):
                self._record(self._stack[-1], input)
            return self._orig(input, *args, **kwargs)

        nn.functional.softmax = tapped

    def _unpatch(self):
        if self._orig is not None:
            nn.functional.softmax = self._orig
            self._orig = None

    def attach(self, name, module):
        if isinstance(module, nn.Softmax):
            def hook(module_, args, kwargs):
                del module_, kwargs
                if args and torch.is_tensor(args[0]) and args[0].shape[-1] == args[0].shape[-2]:
                    self._record(name, args[0])
            return [module.register_forward_pre_hook(hook, with_kwargs=True)]

        self._patch()

        def push(module_, args, kwargs):
            del module_, args, kwargs
            self._stack.append(name)

        def pop(module_, args, kwargs, output):
            del module_, args, kwargs, output
            if self._stack and self._stack[-1] == name:
                self._stack.pop()

        return [module.register_forward_pre_hook(push, with_kwargs=True),
                module.register_forward_hook(pop, with_kwargs=True),
                _SoftmaxTapHandle(self)]

    def on_output(self, output):
        pass

    def finalize(self):
        self._unpatch()
        return {
            n: (torch.cat(e["rows"], dim=0).float().cpu(),
                torch.cat(e["row_q"], dim=0).cpu(),
                e["T"])
            for n, e in self._bufs.items()
        } or None


APPROX = register(Approximation(
    kind="softmax",
    section="softmax",
    matches=lambda m: isinstance(m, nn.Softmax) or "Attention" in type(m).__name__,
    make_collector=_SoftmaxCollector,
    fit_section=fit_softmax_section,
))
