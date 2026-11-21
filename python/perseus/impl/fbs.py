"""CKKS functional bootstrapping (FBS; Alexandru-Kim-Polyakov, ePrint 2024/1623): y = f(x) slot-wise as one refresh.
The refresh's EvalMod computes A sin(2 pi u) of the input's phase u = x / 2^cf (a period of 2^cf in x), and a Chebyshev
series on [-1, 1] applied to it gives f (pc-gpu `StcFirstRoute::run_fbs`, pc-py fbs.rs; the session builds the
functions its programs name, no registration call). The ModRaise runs at K = 32, the route's other tables are the
dense StC-first route's.

Functions (`fbs(ops, x, name)` evaluates one):
- `register_table(name, values)`: a lookup table on the integers |j| <= t/2 (t = len(values) - 1, a power of two):
  phase j q0 / 2t, the degree-t interpolant on the Chebyshev extrema sin(pi j / t);
- `register_sign(name, lo, hi, gap)`: the step (lo for x < 0, hi for x >= 0) on |x| < 2, exact to ~2e-5 outside
  |x| < gap (wrapping with period 8);
- `register(name, f, cf, amp, post, xmax)`: any series `post` on t = amp sin(2 pi x / 2^cf), `f` its numpy twin, valid
  on |x| <= xmax;
- impl/fbs_fit.py `register_smooth(name, f, lo, hi, err | form, deg)`: a smooth f on [lo, hi] (centred: the bootstrap
  reads x - center), fitted in the sine form (the asin warp, on FIDESlib's or the paired Paterson-Stockmeyer, one
  level shallower) or the trigonometric form (f folded into EvalMod's own cosine and sine: P(t) + s Q(t); Bian et
  al., ePrint 2026/367; Xiao et al., ePrint 2026/1255), the shallowest landing meeting the error target.
Real payload (the real-payload chain: f of the real part, Im ignored) unless complex=True. The capture records one
external refresh `{op: "fbs", fn, cf, amp, post, real}` landing where the FBS EvalMod lands (`_pc.fbs_landing`); the
numpy mirror and the fakes compute the series exactly (no noise). The output lands deeper than a plain refresh by
the series' depth (n32, realized: degree 16 at level 40, 64 at 44, 256 at 48), which its consumers pay for.

Chains: n32, n32a4 and n32sph / n32sp (composite degree 2, no reduced special-prime tiers); anything else is refused
(`_pc.fbs_landing` raises). On the alternating single / pair user region of n32sph the EvalMod program steps the
region's levels by their own primes and a sine-form series takes the paired builder; a series landing past the
region's 2^36-scale single levels rounds its intermediates there (the variance model carries it).

Failure model (`p_fail`, `error_bound`): the ModRaise overflow |I| > K is exact (Irwin-Hall: the composite raise lifts
each coefficient as a sum of 2(h + 1) centred uniforms, h the encapsulation secret's weight; 2^-291.1 per bootstrap at
K = 32, h = 32). The rest is Gaussian in a variance model (`_pc.fbs_noise`, pc-gpu bts/fbs_noise.rs): the phase noise
at EvalMod's input (the caller's input noise `sigma_in` plus the split's StC stages, the rescale to q0, the
encapsulation key switch and the CtS stages: rounding and key-switch noise carried by the stages' operator norms) and
the output noise (that phase noise through EvalMod and the series by their derivative, every rescale's rounding and
relinearization through the program's adjoints, and the paired lookup's chain sharing the complex slot), at every
ModRaise integer |I| <= K. A table / sign lookup fails when its phase crosses the decision point (margin: half a grid
step / the gap) or its output moves more than half the output spacing; p_fail per ciphertext = lookups x per-lookup +
the overflow. The predicted std bounds the measured one on n32 (sign t = 16 / 64, identity series; 2^23 outputs each:
model / measured >= 3.3 per table node, >= 1.7 per input bin). Registration warns when p_fail per ciphertext is not
below 2^-128 at `sigma_in` (default `SIGMA_IN`), as the TFHE parameters warn (`lwe_warning`). For a smooth function
`error_bound` gives the deviation bound at a stated tail probability instead.
"""
from __future__ import annotations

import math
import os
import warnings
from fractions import Fraction

import numpy as np

FUNCS: dict = {}      # name -> Fn
_N32_CAP = 44         # the deepest input level that reaches the StC-first entry after a pending rescale (as tfhe.py)
_LAND: dict = {}
_NOISE: dict = {}

K = 32                # ModRaise bound of the FBS EvalMod table (pc_core::bts_setup::EVALMOD_K32)
H_SPARSE = 32         # encapsulation secret weight (pc_core::keys::SecretKey::SPARSE_H)
LOG_N = 16            # ring degree of the validated chains
P_FAIL_MAX = -128.0   # log2 bound p_fail per ciphertext must stay below
SIGMA_IN = 2.0 ** -20  # default caller input noise std (input units) for the registration check
SIGN_CF = 3           # sign inputs |x| < 2 at phase x q0 / 8: u = x / 8, t = sin(pi x / 4) monotone on |x| < 2


class Fn:
    """A registered function: `f` the numpy twin of x -> output, the series `post` (output units, c0 not halved) on
    t = amp sin(2 pi x / 2^cf); `kind` table | sign | smooth with its decision data in `meta`."""

    def __init__(self, f, cf, amp, post, complex, kind, **meta):
        self.f, self.cf, self.amp, self.post, self.complex, self.kind = f, int(cf), float(amp), post, bool(complex), kind
        self.meta = meta


def _preset():
    return os.environ.get("CHAIN_PRESET", "n32") or "n32"


def _register(name, fn, sigma_in, check):
    if len(fn.post) < 3:
        raise ValueError(f"fbs {name!r}: a series of degree {len(fn.post) - 1}; degree >= 2 is supported")
    FUNCS[name] = fn
    if check and fn.kind != "smooth":
        r = p_fail(name, sigma_in=sigma_in)
        if not r["ok"]:
            warnings.warn(
                f"fbs {name!r}: p_fail per ciphertext 2^{r['log2_ct']:.1f} is not below 2^{P_FAIL_MAX:.0f} at input "
                f"noise {r['sigma_in']:.3g} (log2 per lookup: phase {r['log2_phase']:.1f}, output {r['log2_out']:.1f}, "
                f"overflow {r['log2_overflow']:.1f})", RuntimeWarning, stacklevel=3)
    return name


def register(name, f, cf, amp, post, complex=False, xmax=None, sigma_in=None, check=True):
    """`post`: Chebyshev coefficients (c0 not halved, the evaluator's convention) of the output as a function of
    t = amp sin(2 pi x / 2^cf) on [-1, 1]; `f`: the numpy twin of the whole map x -> output; `xmax`: the inputs it is
    valid on (default a quarter period, 2^cf / 4)."""
    xm = float(xmax) if xmax is not None else 2.0 ** int(cf) / 4.0
    fn = Fn(f, cf, amp, [float(c) for c in post], complex, "smooth", xmax=xm)
    return _register(name, fn, sigma_in, check)


def _cheb_fit_step(deg, gap_t, lo, hi):
    """Least squares (iteratively reweighted towards minimax) Chebyshev fit of lo / hi for t < 0 / t >= 0 on [-1, 1]
    outside |t| < gap_t."""
    from numpy.polynomial import chebyshev as C
    t = np.cos(np.pi * (np.arange(16 * deg) + 0.5) / (16 * deg))
    t = t[np.abs(t) >= gap_t]
    y = np.where(t >= 0, hi, lo)
    V = C.chebvander(t, deg)
    w = np.ones_like(t)
    c = np.linalg.lstsq(V, y, rcond=None)[0]
    for _ in range(40):
        r = np.abs(V @ c - y)
        w = w * (1.0 + 20.0 * r / max(r.max(), 1e-300))
        w /= w.mean()
        c = np.linalg.lstsq(V * w[:, None], y * w, rcond=None)[0]
    c = np.asarray(c, dtype=np.float64)
    c[0] *= 2.0          # evaluator convention: c0 not halved
    return c


def register_sign(name, lo=0.0, hi=1.0, deg=127, gap=0.1, sigma_in=None, check=True):
    """The step (`lo` for x < 0, `hi` for x >= 0) on |x| < 2, exact to ~2e-5 outside |x| < `gap` (in between it moves
    from lo to hi); beyond |x| = 2 it follows sin(pi x / 4) (period 8: inputs must stay in (-2, 2))."""
    gap_t = float(np.sin(np.pi * gap / 4.0))
    post = _cheb_fit_step(int(deg), gap_t, float(lo), float(hi))
    from numpy.polynomial import chebyshev as C
    c = post.copy()
    c[0] /= 2.0

    def f(x, c=c):
        return C.chebval(np.sin(np.pi * np.asarray(x, dtype=np.float64) / 4.0), c)
    fn = Fn(f, SIGN_CF, 1.0, [float(v) for v in post], False, "sign", lo=float(lo), hi=float(hi), gap=float(gap))
    return _register(name, fn, sigma_in, check)


def register_table(name, values, complex=False, sigma_in=None, check=True):
    """A lookup table: input j (an integer, |j| <= t/2, t = len(values) - 1 a power of two >= 4) -> values[j + t/2].
    Phase j q0 / 2t (cf = log2 2t), A = 1: EvalMod reads s = sin(pi j / t), the Chebyshev extrema of degree t, and the
    series is the table's degree-t interpolant on them (DCT-I). Inputs off the integers move along the interpolant."""
    v = np.asarray(values, dtype=np.float64)
    t = len(v) - 1
    if t < 4 or t & (t - 1):
        raise ValueError(f"fbs table {name!r}: {len(v)} values; 2^k + 1 (k >= 2) are supported")
    h = t // 2
    k = np.arange(t + 1)
    w = np.where((k == 0) | (k == t), 0.5, 1.0)
    vals = v[t - k]                          # node k <-> j = h - k, values[j + h]
    post = np.array([2.0 / t * np.sum(w * vals * np.cos(np.pi * d * k / t)) for d in range(t + 1)])
    post[t] *= 0.5
    cf = int(math.log2(2 * t))
    from numpy.polynomial import chebyshev as C
    c = post.copy()
    c[0] /= 2.0

    def f(x, c=c, t=t):
        return C.chebval(np.sin(np.pi * np.asarray(x, dtype=np.float64) / t), c)
    fn = Fn(f, cf, 1.0, [float(x) for x in post], complex, "table", values=v, h=h)
    return _register(name, fn, sigma_in, check)


def sign_error(name, gap):
    """The registered sign's largest deviation from lo / hi on gap <= |x| < 2 (the fit's)."""
    f = FUNCS[name].f
    x = np.concatenate([np.linspace(-2.0, -gap, 200001), np.linspace(gap, 2.0, 200001)])
    y = f(x)
    want = np.where(x >= 0, f(np.array([1.0]))[0].round(6), f(np.array([-1.0]))[0].round(6))
    return float(np.abs(y - want).max())


# ── failure model ──────────────────────────────────────────────────────────────────────────────────────────────────

def _irwin_hall_tail(n, k):
    """P(S > k) for S the sum of n independent U(-1/2, 1/2), exact (a Fraction)."""
    y = Fraction(n, 2) - k                   # P(S > k) = P(IH_n < n/2 - k)
    if y <= 0:
        return Fraction(0)
    top = int(y) + 1 if y != int(y) else int(y)
    s = sum((-1) ** j * math.comb(n, j) * (y - j) ** n for j in range(top))
    return s / math.factorial(n)


def _log2_fraction(p):
    if p == 0:
        return -math.inf
    return math.log2(p.numerator) - math.log2(p.denominator)


def log2_overflow(h=H_SPARSE, k=K):
    """log2 P(|I| > K) per coefficient: the composite ModRaise lifts each coefficient mod q0 = p0 p1 as a sum of
    2(h + 1) centred uniforms (two per secret coefficient plus the constant term). Per bootstrap: + LOG_N (union)."""
    return _log2_fraction(2 * _irwin_hall_tail(2 * (int(h) + 1), int(k)))


def _log2_q(z):
    """log2 P(N(0,1) > z), z >= 0 (the asymptotic series past 8)."""
    if z < 8.0:
        return math.log2(0.5 * math.erfc(z / math.sqrt(2.0)))
    ln = -z * z / 2.0 - math.log(z) - 0.5 * math.log(2.0 * math.pi) + math.log1p(-1.0 / z**2 + 3.0 / z**4)
    return ln / math.log(2.0)


def _log2_add(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log2(1.0 + 2.0 ** (min(a, b) - m))


def _z_of_log2_tail(log2p):
    """z with P(|N(0,1)| > z) = 2^log2p (bisection)."""
    lo, hi = 0.0, 64.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 1.0 + _log2_q(mid) > log2p:
            lo = mid
        else:
            hi = mid
    return hi


def _grid(fn):
    if fn.kind == "table":
        h = fn.meta["h"]
        return np.arange(-h, h + 1, dtype=np.float64)
    if fn.kind == "sign":
        return np.linspace(-2.0, 2.0, 401)
    return np.linspace(-fn.meta["xmax"], fn.meta["xmax"], 513)


def noise(name, sigma_in=None, h=H_SPARSE, preset=None):
    """The variance model on the function's input grid (`_pc.fbs_noise`, cached): per input `lo` / `hi` (the
    noise-free output over the ModRaise integers) and `sigma` (std bound), the phase std `phase` (phase units), `xs`."""
    fn = FUNCS[name]
    s = SIGMA_IN if sigma_in is None else float(sigma_in)
    preset = preset or _preset()
    key = (name, s, int(h), preset)
    if key not in _NOISE:
        from perseus import _pc
        xs = _grid(fn)
        r = dict(_pc.fbs_noise(preset, fn.cf, fn.amp, [c / 2.0 ** fn.cf for c in fn.post], [float(x) for x in xs], s,
                               int(h), **_trig_kw(fn)))
        r.pop("by_i", None)
        r["xs"] = xs
        for k_ in ("lo", "hi", "sigma"):
            r[k_] = np.asarray(r[k_])
        _NOISE[key] = r
    return _NOISE[key]


def p_fail(name, margin=None, sigma_in=None, h=H_SPARSE, slots=None, preset=None):
    """Failure probability of a table / sign lookup (module doc). `margin`: the inputs' distance from the decision
    point, input units (default: 1/2 for a table, the gap for a sign); `sigma_in`: the input's own noise std (input
    units, default `SIGMA_IN`); `h`: the encapsulation secret's weight; `slots`: lookups per ciphertext (default N / 2
    real payload, N complex). Returns log2 probabilities per lookup (`log2_phase`: the phase crosses the decision point,
    `log2_out`: the output moves past half the output spacing, `log2_overflow`: |I| > K, `log2_lookup`: their sum) and
    per ciphertext (`log2_ct`: slots x the Gaussian terms + the overflow per bootstrap, union bound, at most 0), `ok`
    (log2_ct < -128), the model's stds at the decisions (`phase_sigma` in input units, `out_sigma`) and the series'
    largest deviation there (`approx`)."""
    fn = FUNCS[name]
    if fn.kind not in ("table", "sign"):
        raise ValueError(f"fbs {name!r}: p_fail is for table / sign lookups; a smooth function has error_bound")
    r = noise(name, sigma_in, h, preset)
    xs, lo, hi, sg = r["xs"], r["lo"], r["hi"], r["sigma"]
    phase_x = r["phase"] * 2.0 ** fn.cf
    if fn.kind == "table":
        v = fn.meta["values"]
        want, dec = v, np.ones_like(xs, dtype=bool)
        d = np.unique(np.round(v, 12))
        spacing = float(np.diff(d).min()) if len(d) > 1 else math.inf
        m = 0.5 if margin is None else float(margin)
    else:
        a, b, gap = fn.meta["lo"], fn.meta["hi"], fn.meta["gap"]
        want, dec = np.where(xs >= 0, b, a), np.abs(xs) >= gap
        spacing = abs(b - a)
        m = gap if margin is None else float(margin)
    dev = float(np.max(np.maximum(np.abs(lo - want), np.abs(hi - want))[dec]))
    out_sigma = float(sg[dec].max())
    lp = 1.0 + _log2_q(m / phase_x)
    room = spacing / 2.0 - dev
    lout = (1.0 + _log2_q(room / out_sigma)) if room > 0 else 0.0
    lov = log2_overflow(h)
    gauss = _log2_add(lp, lout)
    n = int(slots) if slots is not None else (2 ** LOG_N if fn.complex else 2 ** (LOG_N - 1))
    ct = min(0.0, _log2_add(gauss + math.log2(n), lov + LOG_N))
    return {"log2_phase": lp, "log2_out": lout, "log2_overflow": lov, "log2_lookup": _log2_add(gauss, lov),
            "log2_ct": ct, "ok": ct < P_FAIL_MAX, "phase_sigma": phase_x, "out_sigma": out_sigma, "approx": dev,
            "margin": m, "sigma_in": SIGMA_IN if sigma_in is None else float(sigma_in), "slots": n}


def error_bound(name, log2_tail=-128.0, sigma_in=None, h=H_SPARSE, preset=None):
    """A function's deviation bound per output: max |series - f| over its inputs (the fit and EvalMod's approximation,
    at every ModRaise integer) + z sigma_max, z the two-sided Gaussian quantile of 2^log2_tail; the overflow it does
    not cover in `log2_overflow` (per bootstrap)."""
    fn = FUNCS[name]
    r = noise(name, sigma_in, h, preset)
    want = fn.f(r["xs"])
    dev = float(np.max(np.maximum(np.abs(r["lo"] - want), np.abs(r["hi"] - want))))
    sg = float(r["sigma"].max())
    z = _z_of_log2_tail(log2_tail)
    return {"bound": dev + z * sg, "approx": dev, "sigma": sg, "z": z, "log2_tail": log2_tail,
            "log2_overflow": log2_overflow(h) + LOG_N}


# ── capture ────────────────────────────────────────────────────────────────────────────────────────────────────────

def landing(name):
    """(level, deg) of the FBS output on the session's chain (CHAIN_PRESET, default n32); refused off n32 / n32a4."""
    fn = FUNCS[name]
    preset = _preset()
    key = (preset, name)
    if key not in _LAND:
        from perseus import _pc
        _LAND[key] = tuple(_pc.fbs_landing(preset, fn.amp, [c / 2.0 ** fn.cf for c in fn.post], **_trig_kw(fn)))
    return _LAND[key]


def _trig_kw(fn):
    """The trigonometric form's arguments of `_pc.fbs_landing` / `_pc.fbs_noise` (impl/fbs_fit.py): Q (scaled by 2^-cf
    as `post`), t's offset, the base pair's degree and double angles; the paired sine form's flag; none for the
    sine form."""
    t = fn.meta.get("trig")
    if t is None:
        return {"paired": True} if fn.meta.get("paired") else {}
    return {"post_s": [c / 2.0 ** fn.cf for c in t["post_s"]], "off": float(t["off"]), "tdeg": int(t["deg"]),
            "tda": int(t["da"])}


def desc(name):
    fn = FUNCS[name]
    # the StC-first output multiplies by 2^cf: the series carries 2^-cf
    d = {"op": "fbs", "fn": name, "cf": fn.cf, "amp": fn.amp, "post": [c / 2.0 ** fn.cf for c in fn.post],
         "real": not fn.complex}
    kw = _trig_kw(fn)
    if "post_s" in kw:
        d["trig"] = {"post_s": kw["post_s"], "off": kw["off"], "deg": kw["tdeg"], "da": kw["tda"]}
    if kw.get("paired"):
        d["paired"] = True
    return d


def _values(name, vec, novec=True):
    """The function at the FBS input `vec` (centred: x - center for a centred function). `novec`: a vector of one
    entry is the recorder twin's CapCt.novec (the op stream only: zeros); the plaintext mirror passes False (a public
    scalar, e.g. a link's value at a known margin)."""
    fn = FUNCS[name]
    v = np.asarray(vec)
    if novec and v.shape[0] <= 1:
        return np.zeros_like(v, dtype=np.float64)
    if fn.complex:
        return fn.f(np.real(v)) + 1j * fn.f(np.imag(v))
    return np.asarray(fn.f(np.real(v)), dtype=np.float64)


def fbs(ops, x, name):
    """f(x) slot-wise as one refresh (module doc)."""
    if name not in FUNCS:
        raise KeyError(f"fbs: {name!r} not registered")
    from .ops import NumpyOps
    if isinstance(ops, NumpyOps):
        return _values(name, np.asarray(x) - float(FUNCS[name].meta.get("center", 0.0)), novec=False)
    fhe = ops.fhe
    if getattr(ops.core, "bts", None) is not None:
        raise NotImplementedError("fbs runs in planned sessions, not on the eager device path")
    lv, dg = landing(name)             # refused here on a chain FBS is not validated on, before anything is recorded
    center = float(FUNCS[name].meta.get("center", 0.0))
    if center != 0.0:                  # the bootstrap reads x - center (impl/fbs_fit.py: the domain's midpoint)
        if FUNCS[name].complex:
            raise NotImplementedError(f"fbs {name!r}: a centred function on the complex payload")
        x = ops.add(x, -center)
    ops.bootstrap_hint(x, _N32_CAP, True)
    fhe.level_hint(x, _N32_CAP)
    names, levels = [fhe.var_name(x)], [ops.level(x)]
    from .fake import FakeCt
    y = FakeCt(_values(name, x.vec), lv, dg, getattr(x, "packing", "cachemir"))
    if hasattr(x, "ptag"):
        y = type(x)(y.vec, y.meta, packing=y.packing)     # the capture's own ciphertext type (CapCt)
        y.ptag = x.ptag
    d = desc(name)
    vals = np.asarray(y.vec)
    d["range"] = float(max(np.abs(vals).max() if vals.size > 1 else 0.0, 1.0))
    fhe.record_external_bootstrap(y, names, levels, landing=(lv, dg), desc=d)
    return y
