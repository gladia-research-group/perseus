"""The per-site fits of the SANA port's polynomial maps (configs.json, which nl.PolyNL reads), from stats.py's ranges.

  relu      one-sided composite sign (calibrate/approximations/relu.py's stages, the step delayed by its gap and lowered
            by its ripple): ReLU(x) ~ x s(x) with s <= 0 on x <= 0, so f(q).z >= 0 and the attention's division stays
            in its band
  inv_sqrt  Chebyshev seed + Newton on the pre-scaled statistic c^2 z (refit_norm_cheb.fit_site)
  recip     Goldschmidt from the Chebyshev-optimal linear seed on [lo, hi]
  exp       one Chebyshev series of exp((x / 2 - c) 2^-r) (x = 2 s, the split's)
  silu      THOR composite (examples/llama/calibrate.fit_silu) in the Chebyshev basis
  ln        the LayerNorm's NormCfg (center scale, the inverse sqrt's seed + Newton) for blocked.norm_blocked

    .venv/bin/python -m perseus.examples.sana.fit --stats stats_ref.npz --out configs.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

from . import hf
from .nl import _gs, series_units

_UTILS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "utils")


# --deg-align: fits ranked by eval_chebyshev's own cost (nl.series_units: 2 / 4 / 6 / 8 / 10 units up to degree
# 2 / 6 / 14 / 28 / 60, + 4 per Newton iteration) instead of refit_norm_cheb's log2(deg) depth: the default fits often
# land 1-3 degrees above a step (inv_sqrt / ln seeds at 15-17 and 29-31 cost what 28 and 60 cost; Gemma's GELU scans
# odd degrees only, 29 where 28 meets the target). Aligned: the fewest units meeting the target, then the lowest
# degree (no extra multiplications for accuracy past the target).
DEG_ALIGN = False
DEG_STEPS = (2, 6, 14, 28, 60)


def _fit_site(lo, hi, scale, target, max_deg=31):
    if DEG_ALIGN:
        return _fit_site_aligned(lo, hi, scale, target)
    sys.path.insert(0, _UTILS)
    from refit_norm_cheb import fit_site
    return fit_site(lo, hi, scale, target, max_deg)


def _fit_site_aligned(lo, hi, scale, target, max_nr=4):
    """refit_norm_cheb.fit_site's chain (the reweighted minimax relative seed, then Newton) ranked by SANA's units
    (series_units(deg) + 4 nr: nl.inv_sqrt's hint room), then deg + 3 nr: each step's top decides whether the step
    can meet the target with nr iterations, the step is then scanned up from its bottom."""
    from perseus.impl.poly import NumpyOps, eval_chebyshev, inv_sqrt_newton_d2
    ops = NumpyOps()
    t = np.cos(np.pi * (np.arange(400) + 0.5) / 400)
    xs = (t + 1) / 2 * (hi - lo) + lo
    f = scale / np.sqrt(xs)
    grid = np.geomspace(lo, hi, 4001)
    exact = scale / np.sqrt(grid)
    seeds = {}

    def err(deg, nr):
        if deg not in seeds:
            w = 1.0 / f
            c = np.polynomial.chebyshev.chebfit(t, f, deg, w=w)
            for _ in range(30):
                r = np.abs(np.polynomial.chebyshev.chebval(t, c) / f - 1)
                c = np.polynomial.chebyshev.chebfit(t, f, deg, w=w * (r / r.max() + 0.05))
            seeds[deg] = c
        y = inv_sqrt_newton_d2(ops, grid, eval_chebyshev(ops, grid, seeds[deg], lo, hi), nr, 1.0 / (scale * scale))
        return float(np.max(np.abs(y / exact - 1)))

    best = None
    for bot, top in zip((1,) + DEG_STEPS[:-1], DEG_STEPS):
        for nr in range(max_nr + 1):
            u = series_units(top) + 4 * nr
            if best is not None and u > best[0][0]:
                break
            if err(top, nr) <= target:
                deg = next(d for d in range(bot + 1, top + 1) if d == top or err(d, nr) <= target)
                key = (u, deg + 3 * nr)
                if best is None or key < best[0]:
                    best = (key, deg, nr, seeds[deg], err(deg, nr))
                break
    if best is None:
        raise RuntimeError(f"no chain up to degree {DEG_STEPS[-1]} meets {target} on [{lo:.3g}, {hi:.3g}]")
    return best


# ── fits ──

def smoothstep_cheb(m, k):
    """S(h) = h^m sum_{j<k} C(m+j-1, j) (1-h)^j (S(0) = 0, S(1) = 1, flat to orders m and k there) as a Chebyshev series
    in y = 2 h - 1, degree m + k - 1 (exact)."""
    from numpy.polynomial import chebyshev as C
    def S(y):
        h = 0.5 * (y + 1.0)
        return h ** m * sum(math.comb(m + j - 1, j) * (1.0 - h) ** j for j in range(k))
    return [float(v) for v in C.chebinterpolate(S, m + k - 1)]


def fit_relu(degs, eps_bits, margin, polish=None):
    """polish (m, k): the step s(x) = S(h(x)) (smoothstep_cheb on the composite's output): its ripple flattened to
    ~ripple^min(m, k), none subtracted."""
    from perseus.calibrate.approximations.relu import fit_composite_sign, sign_half
    eps = 2.0 ** -eps_bits
    stages = fit_composite_sign(degs, eps)
    t = np.concatenate([np.linspace(-1, -eps, 20001), np.linspace(eps, 1, 20001)])
    h = sign_half(stages, t)
    ripple = float(max(np.abs(h[t < 0]).max(), np.abs(h[t > 0] - 1).max()))
    out = {"degs": list(degs), "eps": eps, "margin": margin, "ripple": ripple,
           "stages": [[[float(c) for c in coef], float(hh)] for coef, hh in stages], "sites": {}}
    if polish:
        out["polish"] = [int(polish[0]), int(polish[1])]
        out["polish_cheb"] = smoothstep_cheb(*polish)
    return out


def fit_inv_sqrt(lo, hi, absmax_hint, target=1e-6, safety=2.0):
    """z in [lo, hi] (data); c puts c^2 z near [0, 1]."""
    c = 1.0 / math.sqrt(hi * safety)
    blo, bhi = c * c * lo / safety, c * c * hi * safety
    (_d, _), deg, nr, coef, err = _fit_site(blo, bhi, c, target)
    return {"c": c, "lo": blo, "hi": bhi, "cheb": [float(v) for v in coef], "nr": int(nr), "fit_err": float(err)}


def fit_recip(lo, hi, target=1e-6, max_iters=40):
    from perseus.calibrate.numerics import chebyshev_gs_init
    a, b = chebyshev_gs_init(lo, hi)
    D = np.geomspace(lo, hi, 4001)
    for it in range(1, max_iters + 1):
        r = _gs(np, D, a, b, it)
        if np.max(np.abs(r * D - 1)) <= target:
            break
    return {"lo": lo, "hi": hi, "alpha": float(a), "beta": float(b), "iters": it,
            "err": float(np.max(np.abs(r * D - 1)))}


def fit_recip_cgs(lo, hi, target=1e-6):
    """1 / x on [lo, hi] by the Chebyshev-scaled Goldschmidt recursion (the shipped SM_CHEB_GS: impl.smcheb,
    poly.goldschmidt_recip with cheb=): k levels for the residual 1 / T_{2^k}(z0) <= target, ~log2(7 sqrt(hi / lo))
    instead of plain Goldschmidt's ~log2(hi / lo)."""
    from perseus.impl import smcheb
    k = smcheb.k_for(lo, hi, target)
    a, b = smcheb.cheb_seed(lo, hi)
    kap, k0, mu = smcheb.step_consts(smcheb.z0_of(lo, hi), k)
    return {"lo": lo, "hi": hi, "cgs": {"alpha": a * k0, "beta": b * k0, "k": int(k), "kap": [float(v) for v in kap],
                                        "mu": float(mu)}, "err": float(target)}


def fit_recip_cheb(lo, hi, target=1e-7, max_deg=63, max_nr=4):
    """1 / x on [lo, hi]: a Chebyshev seed then Newton y <- y (2 - x y), the shallowest (depth) pair meeting `target`
    relative error; None when the band is too wide for a seed (Goldschmidt then)."""
    t = np.cos(np.pi * (np.arange(512) + 0.5) / 512)
    x = (t + 1) / 2 * (hi - lo) + lo
    xs = np.geomspace(lo, hi, 4001)
    ts = (2 * xs - lo - hi) / (hi - lo)
    best = None
    for deg in (6, 14, 28, 60):                     # the series_units breakpoints: the most series per level
        if deg > max_deg:
            break
        c = np.polynomial.chebyshev.chebfit(t, 1.0 / x, deg, w=x)          # relative weighting
        y = np.polynomial.chebyshev.chebval(ts, c)
        for nr in range(max_nr + 1):
            if nr:
                y = y * (2.0 - xs * y)
            err = float(np.abs(y * xs - 1).max())
            if err <= target:
                d = series_units(deg) + 4 * nr
                if best is None or d < best[0]:
                    best = (d, deg, nr, c, err)
                break
    if best is None:
        return None
    d, deg, nr, c, err = best
    return {"lo": lo, "hi": hi, "cheb": [float(v) for v in c], "nr": nr, "depth": d, "err": err}



# Llama prefill's softmax budget (SM_CHEB_GS: the last round to 1e-3; PF_SM_EXP_EPS: a coarser exp), set by main's
# --sm-last / --text-cgs / --exp-target; the defaults are the fits before the port
SM_LAST, SM_CGS_TEXT, EXP_TARGET = 1e-5, False, 1e-9
TEXT_SM_LAST = None                       # --text-sm-last: Gemma's own last-round target (None: SM_LAST)


def _sm_recips(out, site, band0, band, rounds, loose=None, last=1e-5, cgs=False):
    """A softmax's reciprocals sum0 .. sum<rounds> (band0: the exp sum's band, band: the later rounds' sum of p^2).
    With `loose` (the shipped smcheb design): every one but the last to that relative error -- its error is one scale
    per row that the next round's p^2 / sum p^2 cancels -- each band widened by the previous error's square, the last
    to `last`. Without: every one to fit_recip_cheb's default target."""
    eps = 0.0
    for i in range(rounds + 1):
        lo, hi = band0 if i == 0 else band
        lo, hi = lo / (1 + eps) ** 2, hi * (1 + eps) ** 2
        tgt = None if loose is None else (last if i == rounds else loose)
        kw = {} if tgt is None else {"target": tgt}
        fit = fit_recip_cheb(lo, hi, **kw)
        if cgs:                                   # the shallower of the seed + Newton and the scaled recursion
            g = fit_recip_cgs(lo, hi, tgt or 1e-7)
            if fit is None or 2 * (g["cgs"]["k"] + 1) < series_units(len(fit["cheb"]) - 1) + 4 * fit["nr"]:
                fit = g
        out[f"{site}.sum{i}"] = fit or fit_recip(lo, hi, **kw)
        eps = 0.0 if tgt is None else tgt

def fit_exp(lo, hi, c, rounds, target=1e-9, max_deg=63):
    """exp((x / 2 - c) 2^-r) on x in [2 lo, 2 hi]."""
    a, b = 2.0 * lo, 2.0 * hi
    t = np.cos(np.pi * (np.arange(512) + 0.5) / 512)
    x = (t + 1) / 2 * (b - a) + a
    f = np.exp((x / 2 - c) / 2 ** rounds)
    for deg in [d for d in (6, 14, 28, 60) if d <= max_deg] + [max_deg]:   # breakpoints: same levels, more series
        coef = np.polynomial.chebyshev.chebfit(t, f, deg)
        xs = np.linspace(a, b, 4001)
        ex = np.exp((xs / 2 - c) / 2 ** rounds)
        err = float(np.max(np.abs(np.polynomial.chebyshev.chebval((2 * xs - a - b) / (b - a), coef) / ex - 1)))
        if err <= target:
            break
    return {"a": a, "b": b, "c": c, "rounds": rounds, "cheb": [float(v) for v in coef], "err": err}


def fit_silu_site(gmax, safety=1.5, d1=31, d2=27):
    from perseus.calibrate.thor_to_cheb import convert_site
    from perseus.examples.llama.calibrate import fit_silu
    s = fit_silu(gmax, safety, d1, d2)
    convert_site(s, "silu")
    return s


def fit_cheb_fn(f, a, b, target=1e-7, max_deg=127, rel=False, align=False):
    """The lowest-degree Chebyshev series of f on [a, b] meeting `target` (absolute, or relative with rel); odd
    degrees only unless `align` (series_units steps at even degrees 6 / 14 / 28 / 60: 29 costs a step more than 28)."""
    t = np.cos(np.pi * (np.arange(1024) + 0.5) / 1024)
    x = (t + 1) / 2 * (b - a) + a
    xs = np.linspace(a, b, 8001)
    fx = f(xs)
    for deg in range(3, max_deg + 1, 1 if align else 2):
        coef = np.polynomial.chebyshev.chebfit(t, f(x), deg)
        e = np.abs(np.polynomial.chebyshev.chebval((2 * xs - a - b) / (b - a), coef) - fx)
        err = float((e / np.maximum(np.abs(fx), 1e-300)).max() if rel else e.max())
        if err <= target:
            break
    return {"a": float(a), "b": float(b), "cheb": [float(v) for v in coef], "err": err}


def gelu_tanh_np(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def fit_gelu_clip(xmax, C=10.0, degs=(15, 15, 15, 27), eps_bits=11):
    """GELU(x) = x Phi(clip(x, -C, C)) with clip(x) = (|x + C| - |x - C|) / 2, |y| = y sgn(y) by the composite sign
    on [-B, B], B = xmax + C: a wide range costs the sign's depth, not a long series. Phi (the tanh GELU's gate) is one
    Chebyshev series on [-C, C]."""
    from perseus.calibrate.approximations.relu import fit_composite_sign
    stages = fit_composite_sign(degs, 2.0 ** -eps_bits)
    phi = fit_cheb_fn(lambda x: 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3))), -C, C, 1e-8)
    return {"C": C, "B": float(xmax + C), "stages": [[[float(c) for c in coef], float(h)] for coef, h in stages],
            "phi": phi}


def calibrate_text(summ_g, summ_s, n_layers=26, sm_rounds=4, n_pub=207, kc_extra=48, resid=None, sm_loose=None):
    """Gemma's prompt-row sites (gemma.py: g<b>.*) and the caption projection's (ref.py: caption.*)."""
    cfg = {"sm_rounds": sm_rounds, "ln": {}, "softcap": {}, "exp": {}, "recip": {}, "gelu": {}, "gelu_c": {}}
    for b in range(n_layers):
        p = f"g{b}"
        for site in ("ln_in", "ln_post_attn", "ln_ff", "ln_post_ff"):
            st = summ_g[f"{p}.{site}"]
            cfg["ln"][f"{p}.{site}"] = fit_ln(st["lo"], st["hi"], 16 * math.sqrt(st["hi"]))
        lg = summ_g[f"{p}.logit"]
        m = 1.5 * max(abs(lg["lo"]), abs(lg["hi"]))
        # x = 2 s -> 2 * 50 tanh(s / 50) on [-2m, 2m]
        cfg["softcap"][p + ".attn.softcap"] = fit_cheb_fn(lambda x: 100.0 * np.tanh(x / 100.0), -2 * m, 2 * m, 1e-7)
        sm = summ_g[f"{p}.sm"]
        span = sm["hi"] - sm["lo"]
        lo, hi = sm["lo"] - 0.2 * span, sm["hi"] + 0.2 * span
        c = 0.5 * (lo + hi)
        cfg["exp"][p + ".attn.sm.exp"] = fit_exp(lo, hi, c, sm_rounds, target=EXP_TARGET)
        mm = (hi - c) / 2 ** sm_rounds
        kc_lo, kc_hi = n_pub + 1, n_pub + kc_extra
        # every row reads the n_pub public keys: the first sum's band is narrow, a Chebyshev seed + Newton
        lo0, hi0 = kc_lo * math.exp(-mm), kc_hi * math.exp(mm)
        _sm_recips(cfg["recip"], p + ".attn.sm", (lo0, hi0), (0.5 / kc_hi, 1.05), sm_rounds, sm_loose,
                   last=SM_LAST if TEXT_SM_LAST is None else TEXT_SM_LAST, cgs=SM_CGS_TEXT)
        a = summ_g[f"{p}.act"]
        xm = 1.5 * max(abs(a["lo"]), abs(a["hi"]))
        cfg["gelu"][p + ".act"] = fit_cheb_fn(gelu_tanh_np, -xm, xm, 1e-6, align=DEG_ALIGN)
    st = summ_g["g.norm"]
    cfg["ln"]["g.norm"] = fit_ln(st["lo"], st["hi"], 16 * math.sqrt(st["hi"]))
    a = summ_s["caption.act"]
    cfg["gelu_clip"] = {"caption.act": fit_gelu_clip(1.5 * max(abs(a["lo"]), abs(a["hi"])))}
    st = summ_s["caption.norm"]
    cfg["ln"]["caption.norm"] = fit_ln(st["lo"], st["hi"], summ_s["caption.norm.in"]["absmax"])
    cfg["resid_scale"] = (resid_scale(resid) if resid is not None else np.ones(2304)).tolist()
    return cfg


def fit_ln(lo, hi, absmax, target=1e-6, safety=2.0, margin=1.25):
    c = 1.0 / (margin * absmax)
    blo, bhi = c * c * lo / safety, c * c * hi * safety
    (_d, _), deg, nr, coef, err = _fit_site(blo, bhi, 1.0, target)
    return {"method": "remez", "eps": 1e-6, "center_scale": c, "z0": 0.0, "inv_out_scale": 1.0, "Ncoeffs": [],
            "Dcoeffs": [], "lin_alpha": 0.0, "lin_beta": 0.0, "gs_lo": 0.0, "gs_hi": 1.0, "gs_iters": 0,
            "nr_iters": 0, "cheb_coeffs": [float(v) for v in coef], "cheb_lo": blo, "cheb_hi": bhi,
            "cheb_nr_iters": int(nr), "fit_err": float(err)}


def resid_scale(resid, target=128.0):
    """(D,) power-of-two r >= 1: the residual stream carried as x / r stays within ~target on every channel."""
    m = np.asarray(resid).max(0)
    return np.exp2(np.ceil(np.log2(np.maximum(1.0, m / target))))


def calibrate(summ, xa_center, n_layers=28, la_delta=1e-3, relu=((13, 13, 13, 13), 11, 1.5, (3, 3)), sm_rounds=4,
              kc_max=12, resid=None, silu_degs=(15, 27), den_margin=16.0, sm_loose=0.3, den_zero_margin=256.0,
              center_margin=2.0, umax=7.0, qnorm_margin=1.5, qnorm_eps=0.01, qnorm_deg=14):
    """SANA's sites from stats.py's summary; `xa_center` {b<i>: [L, U]}: the cross-attention's row-centred score range
    per block."""
    cfg = {"la_delta": la_delta, "sm_rounds": sm_rounds, "relu": fit_relu(*relu), "inv_sqrt": {}, "recip": {},
           "exp": {}, "silu": {}, "ln": {},
           "resid_scale": (resid_scale(resid) if resid is not None else np.ones(1152)).tolist()}
    for b in range(n_layers):
        p = f"b{b}"
        for site in (p + ".la.q", p + ".la.k"):
            cfg["relu"]["sites"][site] = {"B": relu[2] * summ[site]["absmax"]}
        for site in (p + ".la.knorm", p + ".xa.qnorm", p + ".xa.knorm"):
            cfg["inv_sqrt"][site] = fit_inv_sqrt(summ[site]["lo"], summ[site]["hi"], None)
        den = summ[p + ".la.den"]
        # 2 (den + delta), the scale sigma carried by S and z (kernels): the band [2 lo, 2 max] / sigma, lo the
        # smallest recorded quantile above the zero class (a query with no positive feature: den = num = 0, any finite
        # reciprocal) over den_margin, so the band is the data's, not delta's
        sigma = max(den["hi"], 1.0) / 4.0
        q0 = float(den["q"][0])                      # the 1e-4 quantile
        lo_den = max(la_delta, q0 / den_margin) if q0 > 1e-6 * den["hi"] else la_delta
        if lo_den == la_delta:
            # the zero class reaches the 1e-4 quantile: from the first quantile above it over den_zero_margin (delta's
            # band puts the reciprocal near 1 / delta on those rows: too large to refresh; below the floor the
            # recursion under-converges, bounded, and those rows' numerators are 0)
            qs = [float(v) for v in den["q"] if v > 1e-6 * den["hi"]]
            lo_den = max(la_delta, qs[0] / den_zero_margin) if qs else la_delta
        # the Chebyshev-scaled Goldschmidt recursion
        cfg["recip"][p + ".la.den"] = dict(fit_recip_cgs(2 * lo_den / sigma, 2 * den["hi"] * 2 / sigma), sigma=sigma)
        if p + ".la.hq" in summ:
            # SANA_LA_QNORM (kernels._la_qnorm_map): H the top of the kernel's 2 sum relu(q) per head (u = that / H in
            # [0, 1], qnorm_margin over the data: the series' domain); gain a puts the top of 2 den / sigma / (u + eps)
            # <= (2 / sigma) zbar H / 2 / (1 + eps) on den's own data top 8 (the band's top is 16)
            H = 2.0 * qnorm_margin * summ[p + ".la.hq"]["hi"]
            cfg.setdefault("la_qnorm", {"eps": qnorm_eps, "deg": qnorm_deg})[p + ".la"] = {
                "H": H, "a": 8.0 * (1.0 + qnorm_eps) * sigma / (summ[p + ".la.zbar"]["hi"] * H)}
        # the scores less their row mean (kernels.center_keys) on [L, U] (xa_center's, times center_margin); the fewest
        # rounds that put U / 2^r under umax; exp anchored at the top (every value <= 1); the row's largest centred
        # score is >= 0, so its sum >= e^(-U / 2^r)
        L, U = (center_margin * float(v) for v in xa_center[p])
        r = 0
        while U / 2 ** r > umax:
            r += 1
        cfg["exp"][p + ".xa.sm.exp"] = fit_exp(L, U, U, r, target=EXP_TARGET)
        cfg.setdefault("sm_rounds_site", {})[p + ".xa.sm"] = r
        _sm_recips(cfg["recip"], p + ".xa.sm", (0.5 * math.exp(-U / 2 ** r), 1.05 * kc_max),
                   (0.5 / kc_max, 1.05), r, sm_loose, cgs=True, last=SM_LAST)
        for site in (p + ".ff.act1", p + ".ff.act2"):
            # act1 reads a PCMM output (10 products of room): p1 at degree 13 (6 levels) fits the series under the guard
            degs = (13, silu_degs[1]) if site.endswith("act1") else silu_degs
            cfg["silu"][site] = fit_silu_site(summ[site]["absmax"], 1.5, *degs)
        for site in (p + ".ln1", p + ".ln2"):
            cfg["ln"][site] = fit_ln(summ[site]["lo"], summ[site]["hi"], summ[site + ".in"]["absmax"])
    cfg["ln"]["out.ln"] = fit_ln(summ["out.ln"]["lo"], summ["out.ln"]["hi"], summ["out.ln.in"]["absmax"])
    return cfg


def main():
    """The shipped configs' recipe by default (hf.CONFIGS: its `meta` records the arguments)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="stats_ref.npz", help="stats.py's SANA sites (under the scratch root)")
    ap.add_argument("--stats-gemma", default="stats_gemma.npz", help="stats.py --gemma's (absent: no text side)")
    ap.add_argument("--out", default="configs.json")
    ap.add_argument("--la-delta", type=float, default=1e-3)
    ap.add_argument("--silu-degs", default="15,27", help="THOR p1, p2 degrees (the sites no single series fits; "
                    "15,27: depth 12, the room a PCMM output lands with)")
    ap.add_argument("--relu-degs", default="13,13,13,13", help="the one-sided sign composite's stage degrees")
    ap.add_argument("--relu-eps-bits", type=float, default=11, help="its margin 2^-bits (of the site bound B)")
    ap.add_argument("--relu-margin", type=float, default=1.5, help="the ReLU sites' bound over their max |x|")
    ap.add_argument("--relu-polish", default="3,3", help="m,k: a smoothstep S(m, k) on the composite's step (its "
                    "ripple flattened; none subtracted); '' for none")
    ap.add_argument("--relu-k", default="13,13,27/10/3,3", help="degs/eps_bits[/m,k]: the keys' own composite "
                    "(their features are summed over the tokens: a wider step is tolerated); '' for relu's")
    ap.add_argument("--sm-last", type=float, default=1e-5, help="the softmax's last reciprocal round's target "
                    "(Llama prefill's SM_CHEB_GS: 1e-3)")
    ap.add_argument("--text-sm-last", type=float, default=None, help="Gemma's softmax last round's target (default: "
                    "--sm-last; 1e-3 there cost T rows 1e-5 -> 2.6e-3, 11-11)")
    ap.add_argument("--text-cgs", action="store_true", help="Gemma's softmax reciprocals: the shallower of seed + "
                    "Newton and the Chebyshev-scaled Goldschmidt recursion (Llama prefill's SM_CHEB_GS)")
    ap.add_argument("--exp-target", type=float, default=1e-9, help="the softmax exp series' relative error "
                    "(Llama prefill's coarser exp: PF_SM_EXP_EPS)")
    ap.add_argument("--sm-rounds", type=int, default=4, help="softmax squaring rounds (exp on the 2^-r scaled scores)")
    ap.add_argument("--sm-loose", type=float, default=0.3, help="the softmax's intermediate reciprocals to this "
                    "relative error (the next round cancels it), the last to 1e-5")
    ap.add_argument("--kc-max", type=int, default=12, help="SANA cross-attention keys at most (BOS + prompt tokens)")
    ap.add_argument("--text-kc-extra", type=int, default=12, help="Gemma's prompt keys at most past the public ones")
    ap.add_argument("--xa-center-margin", type=float, default=2.0, help="the centred scores' range over the recorded")
    ap.add_argument("--xa-umax", type=float, default=7.0, help="the largest top of a round's exp domain")
    ap.add_argument("--den-margin", type=float, default=16.0, help="la.den's band from its smallest recorded "
                    "quantile above the zero class over this margin")
    ap.add_argument("--den-zero-margin", type=float, default=256.0, help="la.den: blocks whose zero class reaches the "
                    "1e-4 quantile from the first quantile above it over this margin")
    ap.add_argument("--deg-align", action="store_true", help="inv_sqrt / ln seeds ranked by eval_chebyshev's "
                    "units (series_units + 4 per Newton) and Gemma's GELU over every degree: the fewest units, then "
                    "the lowest degree, meeting the same targets")
    a = ap.parse_args()
    global SM_LAST, SM_CGS_TEXT, EXP_TARGET, TEXT_SM_LAST, DEG_ALIGN
    SM_LAST, SM_CGS_TEXT, EXP_TARGET, TEXT_SM_LAST = a.sm_last, a.text_cgs, a.exp_target, a.text_sm_last
    DEG_ALIGN = a.deg_align
    st = np.load(hf.scratch(a.stats), allow_pickle=True)
    summ = st["summary"].item()
    # the cross-attention's row-centred score range per block (ref.cross_attention's site)
    xa_center = {f"b{b}": [summ[f"b{b}.xa.sm.c"]["lo"], summ[f"b{b}.xa.sm.c"]["hi"]] for b in range(28)}
    polish = lambda v: tuple(int(x) for x in v.split(",")) if v else None          # noqa: E731
    cfg = calibrate(summ, xa_center, la_delta=a.la_delta, resid=st["resid"],
                    relu=(tuple(int(d) for d in a.relu_degs.split(",")), a.relu_eps_bits, a.relu_margin,
                          polish(a.relu_polish)),
                    sm_rounds=a.sm_rounds, kc_max=a.kc_max, den_margin=a.den_margin, sm_loose=a.sm_loose,
                    den_zero_margin=a.den_zero_margin, center_margin=a.xa_center_margin, umax=a.xa_umax,
                    silu_degs=tuple(int(d) for d in a.silu_degs.split(",")))
    if a.relu_k:
        parts = a.relu_k.split("/")
        cfg["relu_k"] = fit_relu(tuple(int(d) for d in parts[0].split(",")), float(parts[1]), a.relu_margin,
                                 polish(parts[2]) if len(parts) > 2 else None)
        cfg["relu_k"].pop("sites")                      # the bounds stay relu's
    if os.path.exists(hf.scratch(a.stats_gemma)):
        sg = np.load(hf.scratch(a.stats_gemma), allow_pickle=True)
        txt = calibrate_text(sg["summary"].item(), summ, sm_rounds=a.sm_rounds, sm_loose=a.sm_loose,
                             kc_extra=a.text_kc_extra, resid=sg["resid"][None])
        for key in ("ln", "exp", "recip"):
            cfg[key].update(txt.pop(key))
        cfg["resid_scale_text"] = txt.pop("resid_scale")
        cfg.update(txt)
    cfg["meta"] = {"tool": "python -m perseus.examples.sana.fit", "args": vars(a)}
    with open(hf.scratch(a.out), "w") as f:
        json.dump(cfg, f, indent=1)
    r = cfg["recip"]
    print(f"[fit] {hf.scratch(a.out)}: relu ripple {cfg['relu']['ripple']:.2e}; la.den iters "
          f"{sorted(set(v.get('iters', v.get('cgs', {}).get('k', -1)) for k, v in r.items() if k.endswith('la.den')))}"
          f" (-1: Chebyshev seed); exp degs {sorted(set(len(v['cheb']) - 1 for v in cfg['exp'].values()))}")


if __name__ == "__main__":
    main()
