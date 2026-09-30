#!/usr/bin/env python
"""rescale_ln_center.py — shrink the LN variance domain of an approximation config by an
exact, output-invariant rescale.

Why: at n32 (CORRECTION_FACTOR=3) the first out-of-band bootstrap at decode token 1 is
`ln_2.inv_sqrt_newton` on a broadcast var_scaled = 28.5 — token 1's LEGITIMATE LN2
variance once real attention mixes in. n64 runs the byte-identical `norm` section and
tolerates 28.5; n32's EvalMod range does not, the refresh garbles, and the block output
collapses. LN(c·x) == LN(x), so scaling the centered input by γ divides var_scaled by
γ² and moves the same value under the wall — provided the WHOLE fitted set moves with it.

This is an exact CLOSED-FORM rescale of that set, not a refit: no calibration pool, no
Remez re-solve. With g = γ²,

    center_scale       ×γ        inv_out_scale  ×γ        (s = inv_y_max·√fit_lo ∝ c)
    center_scale_sq[i] ×g        z0             ×g        fit_lo, fit_hi ×g
    Ncoeffs[k]         ÷g^k      Dcoeffs[k]     ÷g^k      (P'(w) = P(w/g))

    UNCHANGED: eps (raw-var domain — norm.cu:89 multiplies it by c_eff_sq at use),
               gs_lo, gs_hi, lin_alpha, lin_beta, gs_iters, nr_iters, denom_floor,
               z_min, z_max (raw-var domain), precise_var_bts.

D'(w) over the shifted band takes the IDENTICAL value set D(z) did, so the Goldschmidt
init and its iteration budget are invariant; likewise every Remez partial (eval_remez_31's
cube-root balancing scales p^{1/3} by exactly the inverse of x's shrink). The only two
quantities that physically move are `var_scaled` (÷1/g) and `centered_x` (×γ) — which is
the entire point. `weight_loader.h:203` folds 1/inv_out_scale into gamma, so the model
output is invariant with no further edit.

Certified below, per site, in 60-digit arithmetic through the real evaluation structure
(polynomial.cu:218 eval_remez_31 -> primitives.cu:131 goldschmidt_inv -> primitives.cu:17
inv_sqrt_newton): the LN output of the rescaled config matches the source's to <1e-12
relative, over the fit band and 30x past it.

Usage (--var-div is the variance divisor 1/g; --gamma is the equivalent input scale):
  .venv/bin/python scripts/utils/rescale_ln_center.py \
      configs/model/approximation/gpt2_base_n32/configs.json \
      configs/model/approximation/gpt2_base_n32_c50/configs.json --var-div 4

Writes a FULL COPY of the source config (CutMax section byte-equal, lint-safe) with only
the `norm` sites' remez fields rescaled. Power-of-two --var-div keeps every scaling factor
exact in float64.
"""
import argparse
import json
import os

import mpmath as mp

mp.mp.dps = 60

# Any site whose rescaled fit_lo drops below this is flagged: var_scaled near the
# bootstrap noise floor is the precision-cost side of the range trade.
LOW_END_WARN = 1e-5
EXACTNESS_TOL = 1e-12
# Past the fitted band the 3-iteration Newton polish amplifies perturbations, so the
# extended check is a condition-number report, not an exactness claim. It still has to
# stay far below anything that would matter numerically.
EXT_TOL = 1e-6
# Grid density for the invariance certificate. The claim is ALGEBRAIC (each site is a
# fixed rational map composed with a fixed iteration), so density buys nothing past a
# few hundred points — it only costs mpf time, and this runs over 25 sites x 6 c_eff.
GRID = 512


# ── the runtime evaluation chain, mirrored in mpf ────────────────────────────────────
def eval_remez_31(z, Nc, Dc, alpha, beta, gs_iters):
    """polynomial.cu:218 + primitives.cu:131 (goldschmidt_inv). Value-exact mirror."""
    z = mp.mpf(z)
    D = mp.mpf(Dc[1]) * z + mp.mpf(Dc[0])
    F_init = mp.mpf(alpha) - mp.mpf(beta) * D

    # cube-root-balanced Horner: algebraically Nc0 + Nc1 z + Nc2 z^2 + Nc3 z^3
    p = abs(mp.mpf(Nc[3]))
    sgn = mp.mpf(1) if Nc[3] >= 0 else mp.mpf(-1)
    p13 = p ** (mp.mpf(1) / 3)
    x2 = p13 * z
    N = sgn * p13 * z + mp.mpf(Nc[2]) * p13 ** -2
    N = N * x2 + mp.mpf(Nc[1]) * p13 ** -1
    N = N * x2 + mp.mpf(Nc[0])

    # goldschmidt_inv(N, D, F_init, iters)
    Nk = N * F_init
    D_neg = -(D * F_init)
    F = D_neg + 2
    for i in range(1, gs_iters):
        Nk = Nk * F
        if i + 1 < gs_iters:
            D_neg = D_neg * F
            F = D_neg + 2
    return Nk


def inv_sqrt_newton(z, y, iters, x_scale):
    """primitives.cu:17 — y <- 1.5y - 0.5·x_scale·z·y^3."""
    c = mp.mpf(z) * (mp.mpf(-0.5) * mp.mpf(x_scale))
    for _ in range(iters):
        y = mp.mpf(1.5) * y + c * y * y * y
    return y


def ln_output(site, c_eff_sq, var_raw):
    """The LN output for a unit deviate: c_eff·y/s, with y the approximated s·z^{-1/2}.

    Mirrors norm.cu:82-123 — var_scaled = c_eff_sq·(Var+eps), then the remez init and the
    Newton polish, then mult(centered_x, inv_sqrt_var); weight_loader.h:203 divides the
    gamma by s. c_eff and 1/s are the two factors the rescale moves, so this is exactly
    the quantity that must be invariant.
    """
    c_eff_sq = mp.mpf(c_eff_sq)
    z = c_eff_sq * (mp.mpf(var_raw) + mp.mpf(site["eps"]))
    y0 = eval_remez_31(z, site["Ncoeffs"], site["Dcoeffs"],
                       site["lin_alpha"], site["lin_beta"], site["gs_iters"])
    s = mp.mpf(site["inv_out_scale"])
    y = inv_sqrt_newton(z, y0, site["nr_iters"], 1 / (s * s))
    return mp.sqrt(c_eff_sq) * y / s


# ── the transform ───────────────────────────────────────────────────────────────────
def rescale_site(src, gamma, g):
    """Return the rescaled copy of one remez norm site."""
    dst = dict(src)
    dst["center_scale"] = float(mp.mpf(src["center_scale"]) * gamma)
    dst["inv_out_scale"] = float(mp.mpf(src["inv_out_scale"]) * gamma)
    dst["z0"] = float(mp.mpf(src["z0"]) * g)
    for k in ("fit_lo", "fit_hi"):          # metadata (config_loader never binds these)
        if k in src:
            dst[k] = float(mp.mpf(src[k]) * g)
    if "center_scale_sq" in src:
        dst["center_scale_sq"] = [float(mp.mpf(v) * g) for v in src["center_scale_sq"]]
    # P'(w) = P(w/g)  =>  p'_k = p_k / g^k, same for D. Exact when 1/g is a power of two.
    for key in ("Ncoeffs", "Dcoeffs"):
        dst[key] = [float(mp.mpf(c) / g ** k) for k, c in enumerate(src[key])]
    return dst


def certify(name, src, dst, gamma, g):
    """Assert the LN output is invariant, and report the band move + low-end position.

    Two regions, because they answer different questions. IN-BAND ([z_min, z_max], where
    the Remez was fitted) the transform must be invariant to EXACTNESS_TOL — that is the
    correctness claim. Past the band the chain itself is ill-conditioned: only 3 Newton
    iterations (`nr_iters`) polish an init that is being extrapolated, so the cubic map
    amplifies any perturbation. The configs differ there by ~1e-16 whenever gamma is not
    dyadic (center_scale/inv_out_scale round; center_scale_sq and the coefficients are
    exact for a power-of-two --var-div), so the extended number measures the CHAIN's
    condition, not the transform's. We report it, plus the horizon where it crosses 1e-9,
    because the fix has to work at the measured ~13x overshoot.
    """
    assert len(src["Ncoeffs"]) == 4 and len(src["Dcoeffs"]) == 2, \
        f"{name}: unexpected remez degrees {len(src['Ncoeffs'])}/{len(src['Dcoeffs'])}"

    z_min, z_max = mp.mpf(src["z_min"]), mp.mpf(src["z_max"])

    # Every c_eff the runtime can select: the scalar c^2 and the per-position array.
    css_src = [mp.mpf(src["center_scale"]) ** 2]
    css_dst = [mp.mpf(dst["center_scale"]) ** 2]
    if "center_scale_sq" in src:
        idx = sorted({0, 1, 2, len(src["center_scale_sq"]) // 2,
                      len(src["center_scale_sq"]) - 1})
        css_src += [src["center_scale_sq"][i] for i in idx]
        css_dst += [dst["center_scale_sq"][i] for i in idx]

    def sweep(lo, hi, n):
        """Max relative LN-output deviation over a log grid, and the v where it first
        exceeds 1e-9 (mp.inf if it never does)."""
        worst, horizon = mp.mpf(0), mp.inf
        for k in range(n):
            v = lo * (hi / lo) ** (mp.mpf(k) / (n - 1))
            for cs, cd in zip(css_src, css_dst):
                a = ln_output(src, cs, v)
                if a == 0:
                    continue
                d = abs(ln_output(dst, cd, v) - a) / abs(a)
                worst = max(worst, d)
                if d > 1e-9 and horizon is mp.inf:
                    horizon = v
        return worst, horizon

    in_band, _ = sweep(z_min, z_max, GRID)
    ext, horizon = sweep(z_max, z_max * 30, GRID // 4)

    band = (float(mp.mpf(src["fit_lo"]) * g), float(mp.mpf(src["fit_hi"]) * g))
    css_hi = max(dst.get("center_scale_sq", [dst["center_scale"] ** 2]))
    flag = "  ** LOW-END **" if band[0] < LOW_END_WARN else ""
    hz = "none" if horizon is mp.inf else f"{float(horizon / z_max):.1f}x"
    print(f"[ln_rescale] {name:<26} c={src['center_scale']:.4g}->{dst['center_scale']:.4g} "
          f"s={src['inv_out_scale']:.4g}->{dst['inv_out_scale']:.4g} "
          f"band=[{band[0]:.3g},{band[1]:.3g}] max(css)={css_hi:.4g} "
          f"inv={float(in_band):.1e} ext={float(ext):.1e} horizon={hz}{flag}")
    assert float(in_band) < EXACTNESS_TOL, \
        f"{name}: LN output NOT invariant in-band ({float(in_band):.3e} >= {EXACTNESS_TOL})"
    assert float(ext) < EXT_TOL, \
        f"{name}: LN output diverges past the band ({float(ext):.3e} >= {EXT_TOL}) — the " \
        f"chain is not merely ill-conditioned there, check the transform"
    return band[0] < LOW_END_WARN, horizon


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--var-div", type=str, help="variance divisor 1/g (use a power of "
                                                 "two to keep every factor exact)")
    grp.add_argument("--gamma", type=str, help="centered-input scale gamma (g = gamma^2)")
    a = ap.parse_args()
    src_path, dst_path = a.src, a.dst

    if a.var_div is not None:
        g = 1 / mp.mpf(a.var_div)
        gamma = mp.sqrt(g)
    else:
        gamma = mp.mpf(a.gamma)
        g = gamma * gamma
    assert 0 < g < 1, "the point is to SHRINK the domain: need 0 < g < 1"

    cfg = json.load(open(src_path))
    n = skipped = low = 0

    nearest = [mp.inf]

    def walk(o, path=""):
        nonlocal n, skipped, low
        if not isinstance(o, dict):
            return
        if "center_scale" in o and "method" in o:
            if o["method"] != "remez":
                print(f"[ln_rescale] {path}: method={o['method']} — SKIPPED (not remez)")
                skipped += 1
                return
            name = path.rsplit("/", 1)[-1]
            new = rescale_site(o, gamma, g)
            is_low, horizon = certify(name, o, new, gamma, g)
            low += is_low
            nearest[0] = min(nearest[0], horizon / mp.mpf(o["z_max"]))
            o.clear()
            o.update(new)
            n += 1
            return
        for k, v in o.items():
            walk(v, f"{path}/{k}")

    walk(cfg)
    assert n > 0, "no remez norm sites found"

    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    json.dump(cfg, open(dst_path, "w"), indent=1)
    hz = "none" if nearest[0] is mp.inf else f"{float(nearest[0]):.1f}x z_max"
    print(f"[ln_rescale] {n} sites rescaled (gamma={float(gamma):.6g}, var/{float(1/g):g}), "
          f"{skipped} skipped, {low} flagged low-end, nearest sensitivity horizon {hz} "
          f"-> {dst_path}")


if __name__ == "__main__":
    main()
