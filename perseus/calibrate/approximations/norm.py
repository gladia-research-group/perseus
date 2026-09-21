"""Norm inv-sqrt fit — the configs.json "norm" section.

Rational-Remez(d_num/d_den) fit of `s·z^{-1/2}` over the measured per-token
variance domain, refined at runtime by Goldschmidt + Newton. The value-scale
design rule: with s = ln_inv_y_max·sqrt(fit_lo), the output spans
[Y_MAX/ratio, Y_MAX] for every token — above the bootstrap noise floor and
below the range wall. LN(c·x) == LN(x): the runtime scales the centered input
by center_scale = center_target / max|x-mean| and fits over the c²·var domain;
`center_scale_sq` carries the per-position differential rescale k(pos) that
lifts the Goldschmidt operand to its precision optimum at both variance
extremes. Applies unchanged to RMSNorm sites (variance := mean-square, no
centering).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from perseus.calibrate.numerics import (
    GS_INITS, goldschmidt_reciprocal, gs_converged_iters, linear_gs_init,
    nr_converged_iters, polyval_torch, rational_remez,
    safe_quantile,
)
from perseus.calibrate.registry import Approximation, register


def fit_norm(samples, kind, eps, cx_abs_max, pos_stats, cfg,
             max_var_scaled, min_var_scaled):
    """Fit one norm site's inv_sqrt config over its variance samples.

    max_var_scaled / min_var_scaled bound the operating point var_scaled that any
    token may reach (see _center_sq); both are site-dependent (_site_var_bounds):
    - Nonlinearity-feeding LNs (ln_1 -> softmax, ln_2 -> gelu) CAP var_scaled at
      max_var_scaled: an over-scaled q/k doubles the self-attention logit past the
      softmax clip, and an over-scaled gelu input leaves the Remez basin. The
      per-position MAX lands at the cap.
    - The terminal LN (ln_f -> classifier linear, no clip) instead FLOORS at
      min_var_scaled: content-determined patch variance spans a wide range, and
      landing the per-position MIN in the precise band keeps low-variance images
      out of the sub-fit_lo region where bts noise is amplified into garbage.
    Sites with both bounds inert (max=inf, min=0) keep the median at the precise
    target (GPT-2, whose position-determined variance is naturally stable)."""
    s = samples.detach().float().flatten().cpu()
    s = s[s > 0]
    z_lo = safe_quantile(s, cfg.range_q_lo)
    z_hi = safe_quantile(s, cfg.range_q_hi)
    z_lo = max(z_lo, eps)

    z_min = max(z_lo / cfg.z_min_safety, eps)
    z_max = z_hi * (1.0 + cfg.range_margin) * cfg.z_max_safety
    if not (z_min < z_max):
        raise RuntimeError(f"z_min={z_min:.3g} >= z_max={z_max:.3g}")

    center_scale = cfg.center_target / float(cx_abs_max)
    c2 = center_scale * center_scale
    # Floor AND finite cap armed => the per-position rescale lands every token in
    # the ABSOLUTE window [min_var_scaled, max_var_scaled] (the bootstrap precision
    # window: above the ~1e-2 abs noise floor, below the ~10 EvalMod wall). The
    # fit domain must be THAT window (+ safety margins), not the c²-sample band:
    # css[pos] != c², and at outlier-dominated deep sites c² collapses with
    # cx_abs_max, so the c²-band top (basin_safety·fit_hi) crushes the rescale
    # 100x below the noise floor (ViT b6+ detonation). Floor==0 (GPT-2) keeps the
    # legacy sample band byte-identically.
    absolute_band = min_var_scaled > 0.0 and math.isfinite(max_var_scaled)
    if absolute_band:
        fit_lo = float(min_var_scaled) / cfg.z_min_safety
        fit_hi = float(max_var_scaled) * cfg.z_max_safety
        z_min = fit_lo / c2
        z_max = fit_hi / c2
    else:
        fit_lo = z_min * c2
        fit_hi = z_max * c2

    def _fit_band(fit_lo, fit_hi):
        """Remez + Goldschmidt numbers for one candidate fit band."""
        inv_out_scale = float(cfg.inv_y_max) * math.sqrt(fit_lo)
        p_np, q_np, q_min, q_max = rational_remez(
            lambda t: inv_out_scale * t ** -0.5, fit_lo, fit_hi,
            cfg.remez_d_num, cfg.remez_d_den,
        )
        p_np = p_np / q_max
        q_np = q_np / q_max

        gs_lo = float(q_min / q_max)
        gs_hi = 1.0

        gs_lo_fit = gs_lo / cfg.gs_drift_margin
        lin_alpha, lin_beta = GS_INITS[cfg.gs_init_method](gs_lo_fit, gs_hi)
        budget_alpha, budget_beta = linear_gs_init(gs_lo, gs_hi)

        if cfg.gs_iters is not None:
            gs_iters = int(cfg.gs_iters)
        else:
            fit_grid = torch.linspace(fit_lo, fit_hi, 4096, dtype=torch.float64)
            D = polyval_torch(fit_grid, torch.tensor(list(q_np), dtype=torch.float64))
            it = gs_converged_iters(D, budget_alpha, budget_beta,
                                    cfg.gs_target_err, cfg.gs_max_iters)
            # runtime gs_iters counts the init product as iteration 1
            max_it = int(cfg.gs_max_iters)
            gs_iters = max_it if it is None else min(it + 1, max_it)

        # Newton depth: explicit count wins (every shipped config carries one, so
        # this path is inert by default); nr_iters=null runs the same 1e-3
        # criterion that sets gs_iters, on the seed those gs_iters produce.
        if cfg.nr_iters is not None:
            nr_iters = int(cfg.nr_iters)
        else:
            nr_grid = torch.linspace(fit_lo, fit_hi, 4096, dtype=torch.float64)
            N = polyval_torch(nr_grid, torch.tensor(list(p_np), dtype=torch.float64))
            D = polyval_torch(nr_grid, torch.tensor(list(q_np), dtype=torch.float64))
            seed = N * goldschmidt_reciprocal(D, lin_alpha, lin_beta, gs_iters)
            nr_it = nr_converged_iters(nr_grid, seed / inv_out_scale,
                                       cfg.gs_target_err, cfg.nr_max_iters)
            if nr_it is None:
                print(f"[fit_norm] WARNING: Newton does not reach "
                      f"{cfg.gs_target_err:g} within {cfg.nr_max_iters} steps on "
                      f"[{fit_lo:.4g},{fit_hi:.4g}] — the gs_iters={gs_iters} seed "
                      f"is outside the convergence basin at the band edge")
            nr_iters = int(cfg.nr_max_iters) if nr_it is None else nr_it
        return (inv_out_scale, p_np, q_np, gs_lo, gs_hi,
                lin_alpha, lin_beta, gs_iters, nr_iters)

    (inv_out_scale, p_np, q_np, gs_lo, gs_hi,
     lin_alpha, lin_beta, gs_iters, nr_iters) = _fit_band(fit_lo, fit_hi)

    precise_var_bts = False
    finite = float(torch.quantile(s, 0.5))
    if absolute_band:
        # inactive-lane floor mid-window: the c²-median is meaningless on the
        # absolute band (css != c²), and a band-edge floor would park the padding
        # lanes' Newton y² at the bootstrap-wall margin.
        z0_floor = math.sqrt(fit_lo * fit_hi)
    else:
        z0_floor = float(min(max(finite, z_min), z_max)) * c2   # floor on the c²·var domain

    def _center_sq(var_med, var_max, var_min):
        """Per-position rescale: land the median variance on the Goldschmidt
        precision target D·F ≈ rescale_target_df, then clamp so the per-position
        MAX stays ≤ max_var_scaled (nonlinearity range) and the per-position MIN
        stays ≥ min_var_scaled (above the sub-fit_lo garbage region). The rescale
        cancels in the LN output, so this only moves the operating point."""
        a, b = float(lin_alpha), float(lin_beta)
        disc = a * a - 4.0 * b * float(cfg.rescale_target_df)
        if disc <= 0.0 or var_med <= 0.0 or q_np[1] <= 0.0:
            return c2                                     # no rescale -> plain c²
        d_t = (a - math.sqrt(disc)) / (2.0 * b)          # low branch (D < alpha/2beta)
        x_t = (d_t - float(q_np[0])) / float(q_np[1])    # D = D0 + D1*x  ->  target fit input
        # max var_scaled any token may reach. GPT-2: basin_safety*fit_hi (wide, its
        # per-position variance is stable). ViT ln_1/ln_2: rescale_max_var_scaled
        # bounds it to the inv_sqrt's precise low band — content-varying patch
        # variance overshoots the fit basin otherwise (bts-noise-free but
        # structurally imprecise), doubling high-variance tokens' LN output past
        # the softmax/gelu range. min var_scaled any token may reach: ViT ln_f
        # (terminal, linear-feeding) floors here so low-variance images clear
        # fit_lo (the sub-band where bts noise amplifies into garbage).
        max_vs = min(float(cfg.rescale_basin_safety) * fit_hi,
                     float(max_var_scaled))
        # floor OFF (min_var_scaled==0) -> min_vs 0 -> cs2_floor 0, fully inert
        # (GPT-2 unchanged); floor ON -> land the per-position MIN at min_vs, kept
        # inside the fit domain [fit_lo, ...].
        min_vs = max(fit_lo, float(min_var_scaled)) if min_var_scaled > 0.0 else 0.0
        x_t = min(max(x_t, fit_lo), max_vs)
        cs2 = x_t / var_med                              # c_eff² landing var_med at x_target
        cs2_cap = max_vs / max(var_max, var_med)         # per-position MAX at max_vs
        cs2_floor = min_vs / max(var_min, 1e-12)         # per-position MIN at min_vs
        return float(min(max(cs2, cs2_floor), cs2_cap))

    if pos_stats is not None:
        mean = pos_stats["per_pos_mean"]
        pmax = pos_stats["per_pos_max"]
        pmin = pos_stats["per_pos_min"]
        center_scale_sq = [_center_sq(v, pmax[i], pmin[i]) for i, v in enumerate(mean)]

        # Band-coverage fixpoint on the RESCALED domain. The fit band above is
        # derived on the global-c² domain, but the runtime operating point is
        # css[p]·var — with content-determined variance (ViT) the per-position
        # rescale can land tokens far outside the c² band (out-of-band lanes blow
        # the Newton y² past the bootstrap EvalMod wall; sub-band medians sink
        # under the bts noise floor). Grow [fit_lo, fit_hi] until it covers the
        # per-position rescaled range with the same safety margins, refit, and
        # recompute css against the new fit. Position-stable models (GPT-2)
        # converge on pass 1 with the band unchanged.
        for _ in range(3):
            zr_lo = min(cs * pmin[i] for i, cs in enumerate(center_scale_sq))
            zr_hi = max(cs * pmax[i] for i, cs in enumerate(center_scale_sq))
            lo2 = min(fit_lo, zr_lo / cfg.z_min_safety)
            hi2 = max(fit_hi, zr_hi * cfg.z_max_safety)
            if lo2 >= fit_lo * 0.999 and hi2 <= fit_hi * 1.001:
                break
            fit_lo, fit_hi = lo2, hi2
            (inv_out_scale, p_np, q_np, gs_lo, gs_hi,
             lin_alpha, lin_beta, gs_iters, nr_iters) = _fit_band(fit_lo, fit_hi)
            center_scale_sq = [_center_sq(v, pmax[i], pmin[i]) for i, v in enumerate(mean)]

        y_min = float(cfg.inv_y_max) * math.sqrt(fit_lo / fit_hi)
        if y_min < 1e-2:
            print(f"[fit_norm] WARNING: band ratio {fit_hi / fit_lo:.3g} puts the "
                  f"high-variance tokens' inv_sqrt output at {y_min:.3g} < 1e-2 "
                  f"(bts noise floor) — extreme-token LN precision degrades")

        if absolute_band:
            # cap-floor conflict census: positions whose pool var spread exceeds
            # the absolute window ratio land their normal-variance tokens BELOW
            # min_var_scaled (cap wins) — the residual precision tax. Those sites
            # get precise_var_bts: the runtime runs the variance -> GS-init
            # segment at 2-iter bootstraps so the sub-floor tokens keep ~17 bits.
            spread = sorted(pmax[i] / max(pmin[i], 1e-12) for i in range(len(mean)))
            win = float(max_var_scaled) / float(min_var_scaled)
            n_conf = sum(1 for r in spread if r > win)
            q = lambda p: spread[min(len(spread) - 1, int(p * len(spread)))]
            print(f"[fit_norm] window={win:.0f}x spread q50={q(0.5):.0f}x "
                  f"q90={q(0.9):.0f}x max={spread[-1]:.0f}x "
                  f"conflict_pos={n_conf}/{len(spread)}")
            precise_var_bts = n_conf > 0
    else:
        center_scale_sq = [c2]

    denom_floor = float(cfg.denom_floor_frac) * float(gs_lo)

    return {
        "method": "remez",
        "kind": kind,
        "eps": eps,
        "z0": z0_floor,
        "center_scale": float(center_scale),
        "inv_out_scale": float(inv_out_scale),
        "z_min": z_min,
        "z_max": z_max,
        "fit_lo": float(fit_lo),
        "fit_hi": float(fit_hi),
        "Ncoeffs": p_np.tolist(),
        "Dcoeffs": q_np.tolist(),
        "gs_lo": gs_lo,
        "gs_hi": gs_hi,
        "lin_alpha": float(lin_alpha),
        "lin_beta":  float(lin_beta),
        "denom_floor": denom_floor,
        "gs_iters":  int(gs_iters),
        "nr_iters":  int(nr_iters),
        "center_scale_sq": center_scale_sq,
        "precise_var_bts": bool(precise_var_bts),
    }


@dataclass
class _NormData:
    kind: str
    eps: float
    var: torch.Tensor          # per-token variance samples (+eps)
    cx: float                  # input-scale max (LN: max |x-mean|; RMS: max |x|)
    pos: dict | None           # per-position variance profile


def _norm_kind(module) -> str | None:
    if isinstance(module, nn.LayerNorm):
        return "layernorm"
    if type(module).__name__.endswith("RMSNorm"):
        return "rmsnorm"
    return None


def _norm_eps(module) -> float:
    for attr in ("eps", "variance_epsilon"):
        v = getattr(module, attr, None)
        if v is not None:
            return float(v)
    return 1e-5


class _NormCollector:
    """Per-token variance + input-scale samples, and the per-position variance
    profile behind the per-token inv_sqrt rescale k(pos)."""

    def __init__(self, cfg):
        del cfg
        self._var: dict[str, list] = {}
        self._cx: dict[str, list] = {}
        self._pos: dict[str, dict] = {}
        self._meta: dict[str, tuple[str, float]] = {}

    def attach(self, name, module):
        kind, eps = _norm_kind(module), _norm_eps(module)
        self._meta[name] = (kind, eps)

        def hook(module_, inp, output):
            del module_, output
            x = inp[0] if isinstance(inp, tuple) else inp
            if not torch.is_tensor(x):
                return
            xf = x.float()
            if kind == "rmsnorm":
                z = xf.pow(2).mean(dim=-1) + eps          # [B, T] per-token mean-square
                cx = xf
            else:
                z = xf.var(dim=-1, unbiased=False) + eps  # [B, T] per-token variance
                cx = xf - xf.mean(dim=-1, keepdim=True)
            self._var.setdefault(name, []).append(z.flatten().detach().cpu())
            if z.dim() == 2 and z.size(1) > 1:
                zc = z.detach().cpu()                     # [B, T] all positions
                p = self._pos.setdefault(name, {
                    "sum": torch.zeros(zc.size(1)),
                    "max": torch.zeros(zc.size(1)),
                    "min": torch.full((zc.size(1),), float("inf")),
                    "n": 0,
                })
                p["sum"] += zc.sum(dim=0)
                p["max"] = torch.maximum(p["max"], zc.amax(dim=0))
                p["min"] = torch.minimum(p["min"], zc.amin(dim=0))
                p["n"] += zc.size(0)
            self._cx.setdefault(name, []).append(cx.abs().amax().reshape(1).detach().cpu())

        return [module.register_forward_hook(hook)]

    def on_output(self, output):
        pass

    def finalize(self):
        out = {}
        for name, parts in self._var.items():
            kind, eps = self._meta[name]
            p = self._pos.get(name)
            pos = None
            if p is not None:
                pos = {
                    "per_pos_mean": (p["sum"] / max(p["n"], 1)).tolist(),
                    "per_pos_max": p["max"].tolist(),
                    "per_pos_min": p["min"].tolist(),
                }
            out[name] = _NormData(
                kind=kind, eps=eps,
                var=torch.cat(parts),
                cx=float(torch.cat(self._cx[name]).max()),
                pos=pos,
            )
        return out or None


def _site_var_bounds(name, cfg):
    """Per-site (max_var_scaled, min_var_scaled) for the inv_sqrt operating point.
    Fitting runs on the model's own module names (pre section-remap):

    - CAP (max=rescale_max_var_scaled, min=0) for nonlinearity-feeding LNs: GPT-2
      '.ln_1'/'.ln_2', ViT '.layernorm_before'/'.layernorm_after' — an over-scaled
      q/k doubles the self-attention logit past the softmax clip; an over-scaled
      gelu input leaves the Remez basin. (GPT-2 sets the cap to 1e9 == inert.)
    - FLOOR (max=inf, min=rescale_min_var_scaled) for the terminal LN: GPT-2
      '.ln_f', ViT 'vit.layernorm' — linear-feeding, no clip, so land the
      per-position MIN in the precise band instead. (GPT-2 sets the floor 0.)
    - Other sites (MLP / RMS): both bounds inert.

    BERT names BOTH of its post-LN LayerNorms '.LayerNorm', so the ROLE cannot be
    read off the leaf — it is carried by the parent path. Both are nonlinearity-
    feeding (attention.output.LayerNorm feeds the MLP; output.LayerNorm feeds the
    next block's attention), so a single '.output.LayerNorm' suffix covers both and
    routes them to CAP. Note the match is case-sensitive: '.LayerNorm' never hits
    the lowercase ViT/'.layernorm' arms, so without this BERT would silently take
    the inert branch — the legacy c²-sample band that detonated ViT b6+.
    'bert.embeddings.LayerNorm' correctly falls through to inert: it is folded
    client-side in plaintext and never runs under FHE."""
    cap = float(cfg.rescale_max_var_scaled)
    floor = float(cfg.rescale_min_var_scaled)
    floor_nl = float(cfg.rescale_min_var_scaled_nl)
    if name.endswith((".ln_1", ".layernorm_before",
                      ".ln_2", ".layernorm_after",
                      ".output.LayerNorm")):
        # floor+cap both finite => fit_norm switches to the ABSOLUTE band (the
        # bootstrap precision window) and css lands every position inside it.
        return cap, floor_nl
    if name.endswith((".ln_f", ".layernorm")):
        return float("inf"), floor                       # floor: terminal linear
    return float("inf"), 0.0                              # other sites: inert


def fit_norm_section(collected, cfg):
    """The configs.json "norm" section: per-site (or "global") inv_sqrt fits."""
    if cfg.per_layer:
        return {
            name: fit_norm(d.var, d.kind, d.eps, d.cx, d.pos, cfg,
                           *_site_var_bounds(name, cfg))
            for name, d in sorted(collected.items())
        }
    names = sorted(collected)
    all_var = torch.cat([collected[n].var for n in names])
    eps_used = max(d.eps for d in collected.values())
    cx_used = max(d.cx for d in collected.values())
    kind = collected[names[0]].kind
    return {"global": fit_norm(all_var, kind, eps_used, cx_used, None, cfg,
                               float(cfg.rescale_max_var_scaled),
                               float(cfg.rescale_min_var_scaled))}


APPROX = register(Approximation(
    kind="norm",
    section="norm",
    matches=lambda m: _norm_kind(m) is not None,
    make_collector=_NormCollector,
    fit_section=fit_norm_section,
))
