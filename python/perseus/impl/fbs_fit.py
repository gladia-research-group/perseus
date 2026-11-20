"""Series for smooth functions in a functional bootstrap (impl/fbs.py): the forms EvalMod can carry f in.

The FBS input x (centred: x' = x - center, |x'| <= X) is read as the phase u = x' / 2^cf of the ModRaise period.

* ``sine``: EvalMod's sine s = A sin(2 pi u), then a Chebyshev series R(s) (the asin warp: R(s) = f(center +
  2^cf asin(s / A) / 2 pi), A = 1 / sin(2 pi u_max), u_max <= 0.23). R of degree d is a trigonometric polynomial of the
  phase with only the harmonics symmetric about u = 1/4 (cos 2j 2 pi u, sin (2j+1) 2 pi u). ``sine-paired``: the same
  series on the paired Paterson-Stockmeyer builder (pc_shadow `build_sine_paired`): one composite level shallower than
  FIDESlib's at the same degree, a few % more device time.
* ``trig``: f folded into EvalMod's own cosine and sine (Bian et al., ePrint 2026/367: f as a trigonometric series of
  the phase over a Fourier extension; Xiao et al., ePrint 2026/1255: its minimax over the domain alone): EvalMod returns
  the pair c = A cos(2 pi u), s = A sin(2 pi u) (paired double angles), and f = P(t) + s Q(t), t = c + off. Every
  degree-n trigonometric polynomial of u is P of degree n and Q of degree n - 1 in cos(2 pi u); on |u| <= u_max < 1/2
  the fit is over cos(2 pi u) in [cos(2 pi u_max), 1] only (no bridge: the rest of the period is free), mapped to
  t in [-1, 1] by A = 2 / (1 - cos 2 pi u_max), off = -(1 + cos 2 pi u_max) / (1 - cos 2 pi u_max). Q enters the
  paired Paterson-Stockmeyer through s times the giant powers (or at the leaves), so a degree-n trig series lands
  where the paired sine form of degree n does, and approximates like a degree-2n one (singularities in the u plane
  reach t through cos, a quadratic map near u = 0: the Bernstein parameter squares), far better where a singularity
  sits near the domain's end (log, reciprocal, rsqrt on positive ranges: the asin warp's branch point lands there).
  It costs the pair: the sine base series and one more product per double angle (n32 real payload: +3..4 ms).

Both fits are Lawson-weighted least squares on a dense grid of the domain (the minimax limit, `lawson`), in the
Chebyshev basis of the evaluated variable (well conditioned: no large coefficients for the CKKS noise to amplify).
Coefficients are returned in output units, evaluator convention (c0 not halved), as impl/fbs.py's `register` takes
them.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.polynomial import chebyshev as C

TWO_PI = 2.0 * math.pi
SINE_UMAX = 0.23        # today's asin warp: the sine is monotone up to 1/4, the warp's branch point at A


def _grid_u(umax, n):
    """Chebyshev-clustered plus uniform points on [-umax, umax] (fit and check grid)."""
    k = np.arange(n)
    c = umax * np.cos(np.pi * (k + 0.5) / n)
    return np.unique(np.concatenate([c, np.linspace(-umax, umax, n)]))


def lawson(V, y, its=30, w0=None):
    """min_c max |V c - y| by Lawson's iteration (weighted least squares, weights *= |residual|); the first step is
    plain least squares. Returns (c, max residual)."""
    w = np.ones(len(y)) if w0 is None else w0.copy()
    best = None
    for _ in range(its):
        sw = np.sqrt(w)
        c = np.linalg.lstsq(V * sw[:, None], y * sw, rcond=None)[0]
        r = np.abs(V @ c - y)
        m = float(r.max())
        if best is None or m < best[1]:
            best = (c, m)
        w = w * (r + 1e-300)
        s = w.sum()
        if not np.isfinite(s) or s == 0.0:
            break
        w = w / s
    return best


def sine_cf(X, umax=SINE_UMAX):
    """The smallest cf with X / 2^cf <= umax."""
    return max(0, math.ceil(math.log2(X / umax)))


def trig_cf(X, umax=0.25):
    return max(0, math.ceil(math.log2(X / umax)))


def fit_sine(f, lo, hi, deg, cf=None, its=30, rel=False):
    """R(s) on s = A sin(2 pi (x - center) / 2^cf) for f on [lo, hi]: dict(cf, amp, post, center, xmax, err); `rel`:
    minimax of the pointwise relative error (err relative)."""
    center, X = 0.5 * (lo + hi), 0.5 * (hi - lo)
    cf = sine_cf(X) if cf is None else int(cf)
    G = 2.0 ** cf
    umax = X / G
    if not umax < 0.25:
        raise ValueError(f"sine form: |u| <= {umax:.3f} reaches the sine's turning point 1/4 (cf {cf} too small)")
    A = 1.0 / math.sin(TWO_PI * umax)
    u = _grid_u(umax, max(2000, 16 * deg))
    s = A * np.sin(TWO_PI * u)
    y = f(center + G * u)
    V = C.chebvander(s, deg)
    w = 1.0 / np.abs(y) if rel else 1.0
    c, e = lawson(V * np.reshape(w, (-1, 1)), y * w, its)
    post = np.array(c, dtype=np.float64)
    post[0] *= 2.0
    return {"form": "sine", "cf": cf, "amp": A, "post": post, "center": center, "xmax": X, "err": e, "deg": deg}


def fit_trig(f, lo, hi, deg, cf=None, its=30, rel=False):
    """P(t) + s Q(t) for f on [lo, hi] (module doc): dict(cf, amp, off, post, post_s, center, xmax, err); `rel`:
    minimax of the pointwise relative error (err relative)."""
    center, X = 0.5 * (lo + hi), 0.5 * (hi - lo)
    cf = trig_cf(X) if cf is None else int(cf)
    G = 2.0 ** cf
    umax = X / G
    if not umax < 0.5:
        raise ValueError(f"trig form: |u| <= {umax:.3f} covers the whole period (cf {cf} too small)")
    cmin = math.cos(TWO_PI * umax)
    A = 2.0 / (1.0 - cmin)
    off = -(1.0 + cmin) / (1.0 - cmin)
    u = _grid_u(umax, max(2000, 32 * deg))
    th = TWO_PI * u
    t = np.clip(A * np.cos(th) + off, -1.0, 1.0)
    sv = A * np.sin(th)
    y = f(center + G * u)
    VP = C.chebvander(t, deg)
    VQ = C.chebvander(t, max(deg - 1, 0)) * sv[:, None]
    V = np.hstack([VP, VQ])
    w = 1.0 / np.abs(y) if rel else 1.0
    c, e = lawson(V * np.reshape(w, (-1, 1)), y * w, its)
    P = np.array(c[:deg + 1], dtype=np.float64)
    Q = np.array(c[deg + 1:], dtype=np.float64)
    P[0] *= 2.0
    Q[0] *= 2.0
    return {"form": "trig", "cf": cf, "amp": A, "off": off, "post": P, "post_s": Q, "center": center, "xmax": X,
            "err": e, "deg": deg}


def eval_fit(r, x):
    """The fit's value at inputs x (original units, float64)."""
    G = 2.0 ** r["cf"]
    th = TWO_PI * (np.asarray(x, dtype=np.float64) - r["center"]) / G
    p = np.array(r["post"], dtype=np.float64)
    p[0] /= 2.0
    if r["form"] == "sine":
        return C.chebval(r["amp"] * np.sin(th), p)
    q = np.array(r["post_s"], dtype=np.float64)
    q[0] /= 2.0
    t = r["amp"] * np.cos(th) + r["off"]
    return C.chebval(t, p) + r["amp"] * np.sin(th) * C.chebval(t, q)


# ── registration ────────────────────────────────────────────────────────────────────────────────────────────────────

#: The trigonometric form's base pairs (pc_shadow `trig_base`: degree, double angles). "std": degree 27, 5 double
#: angles (a degree-n series lands where the paired sine form of degree n does); "low": degree 59, 3 double angles,
#: one level shallower (the larger base series runs at the top levels: more ms).
TRIG_BASES = {"std": (27, 5), "low": (59, 3)}
SINE_DEGREES = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384)
TRIG_DEGREES = (4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128)

_FITS: dict = {}


def _make_fn(name, f, r, base, complex):
    from . import fbs
    c = float(r["center"])

    def twin(x, f=f, c=c):
        return f(c + np.asarray(x, dtype=np.float64))
    meta = {"xmax": float(r["xmax"]), "center": c, "form": r["form"], "fit_err": float(r["err"]), "deg": int(r["deg"])}
    if r["form"] == "sine" and r.get("paired"):
        meta["paired"] = True
    if r["form"] == "trig":
        d, da = TRIG_BASES[base]
        meta["trig"] = {"post_s": [float(v) for v in r["post_s"]], "off": float(r["off"]), "deg": d, "da": da}
        meta["base"] = base
    return fbs.Fn(twin, r["cf"], r["amp"], [float(v) for v in r["post"]], complex, "smooth", **meta)


def _install(name, fn, sigma_in, check):
    from . import fbs
    for cache in (fbs._LAND, fbs._NOISE):
        for k in [k for k in cache if name in k]:
            cache.pop(k)
    fbs._register(name, fn, sigma_in, check)
    _FITS[name] = fn
    return name


def candidates(f, lo, hi, err, rel=False, forms=("sine", "sine-paired", "trig", "trig-low"), degrees=None, cfs=None):
    """Per form (and cf), the smallest degree whose float64 fit error is at most half the target (`err`, absolute; or
    pointwise relative when `rel`): a list of fit dicts (impl `fit_sine` / `fit_trig`) with "base"."""
    X = 0.5 * (hi - lo)
    tgt = 0.5 * err
    out = []
    for form in forms:
        base = form[5:] if form.startswith("trig-") else "std"
        sine = form.startswith("sine")
        fit = fit_sine if sine else fit_trig
        c0 = sine_cf(X) if sine else trig_cf(X)
        for cf in (cfs or (c0, c0 + 1)):
            for d in (degrees or (SINE_DEGREES if sine else TRIG_DEGREES)):
                try:
                    r = fit(f, lo, hi, d, cf=cf, rel=rel)
                except ValueError:
                    break
                if r["err"] <= tgt:
                    r["base"] = base
                    r["paired"] = form == "sine-paired"
                    out.append(r)
                    break
    return out


def feasible(name, f, lo, hi, err, rel=False, sigma_in=None, log2_tail=-128.0, complex=False):
    """Every form of `candidates` whose noise-model bound (`fbs.error_bound` at `log2_tail`) meets `err`: a list of
    (fit, landing level, bound), registered in turn under `name` (the last one stays registered)."""
    from . import fbs
    out = []
    for r in candidates(f, lo, hi, err, rel):
        _install(name, _make_fn(name, f, r, r["base"], complex), sigma_in, False)
        lv = fbs.landing(name)[0]
        b = bound(name, log2_tail, sigma_in, rel)
        if b <= err:
            out.append((r, lv, b))
    return out


def bound(name, log2_tail=-128.0, sigma_in=None, rel=False):
    """`fbs.error_bound` of a registered function; `rel`: pointwise relative (max over the model's grid of
    (deviation + z sigma) / |f|)."""
    from . import fbs
    if not rel:
        return fbs.error_bound(name, log2_tail=log2_tail, sigma_in=sigma_in)["bound"]
    r = fbs.noise(name, sigma_in)
    want = fbs.FUNCS[name].f(r["xs"])
    dev = np.maximum(np.abs(r["lo"] - want), np.abs(r["hi"] - want))
    z = fbs._z_of_log2_tail(log2_tail)
    return float(np.max((dev + z * r["sigma"]) / np.abs(want)))


def install(name, f, fit, landing=None, bound=None, complex=False, sigma_in=None, check=True):
    """Register a fit dict (`fit_sine` / `fit_trig` / an entry of `feasible`) under `name`."""
    fn = _make_fn(name, f, fit, fit.get("base", "std"), complex)
    if landing is not None:
        fn.meta["landing"], fn.meta["bound"] = landing, bound
    return _install(name, fn, sigma_in, check)


def form_rank(r):
    """The forms by device cost at one landing: sine, paired sine, trig std, trig low."""
    if r["form"] == "sine":
        return 1 if r.get("paired") else 0
    return {"std": 2, "low": 3}[r.get("base", "std")]


def form_name(r):
    if r["form"] == "sine":
        return "sine-paired" if r.get("paired") else "sine"
    return "trig" if r.get("base", "std") == "std" else f"trig-{r['base']}"


def register_smooth(name, f, lo, hi, err=None, rel=False, form="auto", deg=None, cf=None, base="std", complex=False,
                    sigma_in=None, log2_tail=-128.0, check=True):
    """Register f on [lo, hi] (a vectorized numpy function) as an FBS (impl/fbs.py `fbs(ops, x, name)`).

    `form` "sine" / "sine-paired" / "trig" (with `deg`, optional `cf` and trig `base` "std" | "low") fits that form;
    "auto" (with an error target `err`, absolute or pointwise relative when `rel`) fits every form at the smallest
    degree meeting half the target, keeps those whose noise-model bound (`bound` at `log2_tail`, the input noise
    `sigma_in`) meets it, and picks the shallowest landing, then the cheaper form (`form_rank`), then the lower
    degree. The choice is in `FUNCS[name].meta` (form, deg, fit_err, base, paired, landing, bound)."""
    lo, hi = float(lo), float(hi)
    if not hi > lo:
        raise ValueError(f"fbs {name!r}: domain [{lo}, {hi}]")
    if form != "auto":
        if deg is None:
            raise ValueError(f"fbs {name!r}: form {form!r} needs a degree")
        r = (fit_sine if form.startswith("sine") else fit_trig)(f, lo, hi, int(deg), cf=cf, rel=rel)
        r["paired"] = form == "sine-paired"
        return _install(name, _make_fn(name, f, r, base, complex), sigma_in, check)
    if err is None:
        raise ValueError(f"fbs {name!r}: form 'auto' needs an error target")
    ok = feasible(name, f, lo, hi, err, rel, sigma_in, log2_tail, complex)
    if not ok:
        raise ValueError(f"fbs {name!r}: no form meets {err:g} on [{lo}, {hi}] (noise-model bound at 2^{log2_tail:g})")
    r, lv, b = min(ok, key=lambda e: (e[1], form_rank(e[0]), e[0]["deg"]))
    return install(name, f, r, lv, b, complex, sigma_in, check)
