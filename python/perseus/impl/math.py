"""Elementwise functions on ciphertexts with explicit domains and error control.

Every function takes its input domain [lo, hi] (public: a calibrated bound) and an error target, picks the cheapest
method that meets the target over the whole domain, and returns the value. The choice is made on the host, once per
(function, domain, target) and cached: each candidate method is run on a dense grid of the domain in float64 (the
same code, on numpy) for its max / RMS error, and on a counting backend (`CountOps`) for its multiplicative depth and
ciphertext products; the feasible candidate with the fewest levels wins, then the fewest products. `approx(...)`
returns that choice (`Approx`: method, parameters, depth, error) without evaluating anything; calling it on a ciphertext
evaluates it (`Approx.__call__(ops, x)`), and the module-level functions do both.

Methods (all built from impl/poly.py's kernels, so they record the standard op stream and plan like any stage):

* ``cheb``: one Chebyshev interpolant of degree d on [lo, hi] (baby-step giant-step, poly.eval_chebyshev), the
  smallest d meeting the target. Depth ceil(log2(d + 1)) + 1, plus one level for the affine map unless the domain
  is [-1, 1].
* ``exp``: ``cheb+sq``: exp(x / 2^k) by a short series, then k squarings (range reduction by squaring: the relative
  error doubles per squaring, the series' degree falls much faster for wide domains). An `out_scale` rides the series'
  coefficients (no level), so the result can be held at a chosen size (e.g. exp(x - hi) <= 1 before a refresh).
* ``log``: ``cheb`` on [lo, hi], or ``atanh``: log x = log c + 2 atanh((x - c) / (x + c)) with c = sqrt(lo hi), the
  scaling that halves the log-range seen by the series (the division by `reciprocal`).
* ``rsqrt``: ``cheb``; ``cheb+newton``: a low-degree Chebyshev seed refined by Newton y <- y (3 - x y^2) / 2 (two
  levels per iteration, poly.inv_sqrt_newton_d2); ``scaled-newton``: the Chebyshev-scaled Newton steps from the seed 1
  (poly.scaled_newton_steps / inv_sqrt_newton_scaled, the LayerNorm's wide-band inverse square root).
* ``reciprocal``: ``cheb``; ``cheb+newton`` (y <- y (2 - x y)); ``goldschmidt`` (linear Chebyshev seed,
  poly.goldschmidt_recip, the softmax's division); ``rsqrt^2``.
* ``sqrt``: ``cheb`` (also on [0, hi]); ``x*rsqrt``.
* ``tanh``, ``sigmoid``, ``erf``: ``cheb`` with the function's parity enforced (odd series: half the products).
* ``pow``: integer exponents by squaring (exact); 1/2, -1/2, -1 routed to sqrt / rsqrt / reciprocal; any other real
  exponent on a positive domain by ``cheb`` or ``exp(p log x)``.
* ``sign``, ``abs``, ``relu``, ``max``, ``min``, ``clip``: ``ckks``: a composite of odd minimax polynomials (Lee,
  Lee, Kim, No, IEEE TDSC 2021: calibrate/approximations/relu.py's fit), the ladder chosen for the resolution `eps`
  and the target (abs / relu fold the product by x into the last stage: relu.relu_folded); ``tfhe``: the sign from
  the scheme switch (impl/tfhe.py, tfhe_select.msign: a PBS sign LUT, exact up to the switch's input noise band
  ~1e-2 of the range; p_fail per PBS < 2^-128 at the default n 840 parameters), then one product. The TFHE path runs
  in planned sessions only (the switch is a refresh the planner places).

The target is held against the encrypted error, not the float64 one: `Approx.err_enc` = the float64 error plus the
noise model's term (NOISE_BTS x the method's noise gain when the evaluation is deeper than BUDGET levels and the
planner must refresh inside it; see NOISE_BTS). A deep high-degree series refreshed inside amplifies the refresh's
error by its Chebyshev recurrence (measured on the device: a degree-318 rsqrt on [1e-3, 10] read 2.8e-2 relative
for a float64 error of 1e-3), so wide domains at tight targets are refused (TargetError) rather than chosen.
"""
from __future__ import annotations

import dataclasses
import functools
import math as _m
from typing import Callable

import builtins

import numpy as np
from numpy.polynomial import chebyshev as _C

from . import poly
from .ops import NumpyOps

_abs = builtins.abs

# ── counting and plaintext backends ─────────────────────────────────────────────────────────────────────────────────


class _V:
    """A value on CountOps: its multiplicative depth."""
    __slots__ = ("d",)

    def __init__(self, d):
        self.d = d


class CountOps:
    """The ops protocol on depth counters: levels consumed (`depth`, the deepest value), ciphertext products
    (`ct_mults`), plaintext / scalar products (`pt_mults`), key switches (`rots`: rotations and conjugations). A scalar
    product costs a level (the runtime rescales it like a plaintext product); additions, i x and rotations are free.
    The same switches as a default FheOps (BSGS Chebyshev, dead-op elision)."""
    complex_payload = True
    cheb_bsgs = True
    dead_ops = True
    realize_shared = False
    cheb_align = False
    neg_free = False
    fused_sm_den = fused_ln_var = folds = False
    unit = 2

    def __init__(self):
        self.ct_mults = self.pt_mults = self.rots = self.depth = self.bts = 0

    def _v(self, d):
        self.depth = max(self.depth, d)
        return _V(d)

    @staticmethod
    def _d(x):
        return x.d if isinstance(x, _V) else 0

    @staticmethod
    def _sc(b):
        return isinstance(b, (int, float, np.floating, np.integer, complex))

    def leaf(self, d=0):
        return _V(d)

    def add(self, a, b): return _V(max(self._d(a), self._d(b)))
    sub = add

    def mult(self, a, b):
        if self._sc(b):
            self.pt_mults += 1
            return self._v(self._d(a) + 1)
        self.ct_mults += 1
        return self._v(max(self._d(a), self._d(b)) + 1)

    def square(self, a):
        self.ct_mults += 1
        return self._v(self._d(a) + 1)

    def negate(self, a): return _V(self._d(a))
    def mult_i(self, a): return _V(self._d(a))

    def conjugate(self, a):
        self.rots += 1
        return _V(self._d(a))

    def rotate(self, a, k):
        self.rots += 1
        return _V(self._d(a))

    def rotate_many(self, a, steps):
        return [self.rotate(a, k) for k in steps]

    def rotate_and_sum(self, a, start, stop):
        g = _abs(int(start))
        while g < stop:
            self.rots += 1
            g *= 2
        return _V(self._d(a))

    def inplace_add(self, a, b): a.d = max(a.d, self._d(b))
    def copy(self, a): return _V(self._d(a))

    def scalar_sum(self, ts, cs):
        self.pt_mults += len(ts)
        return self._v(max(self._d(t) for t in ts) + 1)

    def mult_pt(self, a, v, key=None, kind="weight", tagged=True):
        self.pt_mults += 1
        return self._v(self._d(a) + 1)

    def add_pt(self, a, v, key=None, kind="weight", tagged=True): return _V(self._d(a))

    def mult_const(self, a, re, im):
        self.pt_mults += 1
        return self._v(self._d(a) + 1)

    def realize(self, a): return a
    def bootstrap_hint(self, a, thr, acct=False): return a
    def tag_reduce(self, a, stride): return a

    def bootstrap(self, a):
        self.bts += 1
        a.d = 0
        return a

    bootstrap_real = bootstrap


class _NpOps(NumpyOps):
    """The plaintext mirror with FheOps' default switches (BSGS Chebyshev, dead ops), so the error measured on the
    grid is the error of the evaluation order the device runs."""
    cheb_bsgs = True
    dead_ops = True
    complex_payload = True


def count(fn, *args, **kw) -> CountOps:
    """Run `fn(ops, x, *args, **kw)` on a fresh CountOps (x at depth 0); returns the counter."""
    c = CountOps()
    fn(c, c.leaf(), *args, **kw)
    return c


# ── the choice ──────────────────────────────────────────────────────────────────────────────────────────────────────


#: The encrypted evaluation's noise model (n32 chain, GPU 3, 2026-11-20): without a refresh inside, the device matches
#: float64 to ~NOISE_FRESH; an evaluation deeper than BUDGET levels gets refreshed inside (a fresh input or a refresh
#: landing sits at 32 and the planner refreshes by 48: 8 units of the composite chain), and the planner's min cut
#: refreshes the values most others are built from (a Chebyshev ladder's early T_k, whose error the recurrence then
#: multiplies): the output error is ~ noise_gain x NOISE_BTS. Measured against it: log degree 311 on [0.01, 100]
#: 5.2e-2 (gain 6657), sqrt degree 460 on [0, 1] 5.5e-4 (gain 128), rsqrt degree 318 on [1e-3, 10] 0.89 (gain 80829),
#: i.e. 0.4-1.1 x NOISE_BTS per unit of gain.
NOISE_BTS = 1e-5
NOISE_FRESH = 1e-9
BUDGET = 8


@dataclasses.dataclass(frozen=True)
class Approx:
    """One method for f on [lo, hi]: its parameters, its depth (levels, `CountOps`), its ciphertext / plaintext products,
    its float64 error over the domain (max and RMS; relative to |f| when `rel`) and its noise gain (the output's
    absolute change per unit of i.i.d. noise on every product, max over the domain). `err_enc` = err + gain x the
    noise model (NOISE_BTS past BUDGET levels, else NOISE_FRESH; relative targets: over max |f|): what the choice
    is held to. Call it on a ciphertext."""
    fn: str
    method: str
    lo: float
    hi: float
    depth: int
    ct_mults: int
    pt_mults: int
    err: float
    rms: float
    rel: bool
    params: tuple = ()
    gain: float = 0.0
    fmax: float = 1.0
    run: Callable = dataclasses.field(default=None, repr=False, compare=False)

    @property
    def err_enc(self) -> float:
        noise = self.gain * (NOISE_BTS if self.depth > BUDGET else NOISE_FRESH)
        return self.err + (noise / self.fmax if self.rel else noise)

    def __call__(self, ops, x):
        return self.run(ops, x)

    def unit(self):
        """A `cheb` method as (alpha, beta, g): g(ops, t) is the series on t = alpha x + beta in [-1, 1], one level
        less than the call (the affine map is the caller's: e.g. folded into a product before a refresh, so the series
        starts on a refreshed unit-size value). None for the other methods."""
        c = getattr(self.run, "cheb", None)
        if c is None:
            return None
        lo, hi = self.lo, self.hi
        return (2.0 / (hi - lo), -(hi + lo) / (hi - lo),
                lambda ops, t, c=c: poly.eval_chebyshev(ops, t, list(c), -1.0, 1.0, eps=0.0))

    def numpy(self, x):
        """The approximation in float64 (what the encrypted evaluation computes, without the CKKS noise)."""
        y = self.run(_NpOps(), np.asarray(x, dtype=np.float64))
        return np.real(y)

    def describe(self) -> dict:
        return {"fn": self.fn, "method": self.method, "lo": self.lo, "hi": self.hi, "depth": self.depth,
                "ct_mults": self.ct_mults, "pt_mults": self.pt_mults, "err": self.err, "rms": self.rms,
                "rel": self.rel, "params": dict(self.params), "gain": self.gain, "err_enc": self.err_enc}


def grid(lo: float, hi: float, n: int = 4097) -> np.ndarray:
    """The error grid: uniform plus Chebyshev-clustered points (the interpolants' error peaks at the ends)."""
    u = np.linspace(lo, hi, n)
    k = np.arange(n)
    c = 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.cos(np.pi * k / (n - 1))
    return np.unique(np.concatenate([u, c]))


def _measure(f, run, lo, hi, rel):
    x = grid(lo, hi)
    y = np.real(run(_NpOps(), x.copy()))
    t = f(x)
    e = np.abs(y - t)
    if rel:
        e = e / np.maximum(np.abs(t), 1e-300)
    return float(e.max()), float(np.sqrt(np.mean(e * e)))


class _Noisy(_NpOps):
    """The mirror with i.i.d. Gaussian noise of std `s` on every product's output (the noise gain's probe)."""

    def __init__(self, s, seed=0):
        self.s, self.rng = float(s), np.random.default_rng(seed)

    def _n(self, x):
        # CKKS noise is absolute at the scale of a value near 1 and grows with larger values (a refresh prescales them)
        return x + self.s * self.rng.standard_normal(np.shape(x))

    def mult(self, a, b):
        return self._n(super().mult(a, b))

    def square(self, a):
        return self._n(super().square(a))


def noise_gain(run, lo, hi, s=1e-6):
    """max |run_noisy - run| / s on the domain: how much the products' noise (CKKS rounding, a bootstrap's error on an
    intermediate) is amplified to the output. Self-correcting iterations (Newton, Goldschmidt) keep it small, a
    high-degree series with large coefficients makes it large."""
    x = grid(lo, hi, 1025)
    y0 = np.real(run(_NpOps(), x.copy()))
    y1 = np.real(run(_Noisy(s), x.copy()))
    return float(np.abs(y1 - y0).max() / s)


def _make(fn, method, f, lo, hi, rel, run, params=()):
    err, rms = _measure(f, run, lo, hi, rel)
    c = count(run)
    fmax = float(np.abs(f(grid(lo, hi, 257))).max()) or 1.0
    return Approx(fn, method, float(lo), float(hi), int(c.depth), int(c.ct_mults), int(c.pt_mults), err, rms,
                  bool(rel), tuple(params), noise_gain(run, lo, hi), fmax, run)


def _pick(cands, err):
    """The cheapest candidate meeting `err` with its noise (err_enc; depth, then ciphertext products, then error);
    None if none does."""
    ok = [a for a in cands if a is not None and a.err_enc <= err]
    if not ok:
        return None
    return min(ok, key=lambda a: (a.depth, a.ct_mults, a.pt_mults, a.err))


class TargetError(ValueError):
    """No method reaches the error target within the degree / iteration limits."""


_LAST: dict = {}


def _best(name, cands, err):
    _LAST["cands"] = [c for c in cands if c is not None]
    a = _pick(cands, err)
    if a is None:
        got = min((c.err_enc for c in cands if c is not None), default=float("inf"))
        raise TargetError(f"{name}: no method reaches {err:.3g} (best {got:.3g}); widen the target or narrow the domain")
    return a


# ── Chebyshev interpolants ──────────────────────────────────────────────────────────────────────────────────────────

MAX_DEGREE = 1023
_DEGREES = (1, 2, 3, 5, 7, 11, 15, 23, 31, 47, 63, 95, 127, 191, 255, 383, 511, 767, 1023)


def cheb_fit(f, lo: float, hi: float, deg: int, parity: int | None = None) -> np.ndarray:
    """Chebyshev coefficients (c0 unhalved, poly.eval_chebyshev's convention) of the degree-`deg` interpolant of f at
    the Chebyshev points of [lo, hi]; `parity` 1 / 0: an odd / even f on a symmetric domain (the other half zeroed)."""
    c = _C.chebinterpolate(lambda t: f(0.5 * (hi - lo) * t + 0.5 * (hi + lo)), int(deg))
    if parity is not None:
        c[(1 - parity)::2] = 0.0
    return np.asarray(c, dtype=np.float64)


def _trim(c, err):
    """Drop the tail coefficients whose total is below a hundredth of the target (each changes the value by <= |c|)."""
    c = np.array(c, dtype=np.float64)
    tail = np.cumsum(np.abs(c[::-1]))[::-1]
    keep = len(c)
    while keep > 2 and tail[keep - 1] <= 0.01 * err:
        keep -= 1
    c = c[:keep]
    c[np.abs(c) <= 1e-3 * err / max(len(c), 1)] = 0.0
    return c


def _cheb_run(c, lo, hi):
    c = [float(v) for v in c]

    def run(ops, x):
        return poly.eval_chebyshev(ops, x, c, lo, hi, eps=0.0)
    run.cheb = tuple(c)
    return run


def _cheb_degree(f, lo, hi, err, rel, parity=None, tgt=None, degrees=_DEGREES):
    """The smallest-degree interpolant of f meeting `err` against `tgt` (default f): a scan of `degrees`, then a
    bisection below the first one that meets it. Returns (coeffs, err, rms) or None."""
    tgt = f if tgt is None else tgt
    x = grid(lo, hi)
    t = tgt(x)
    den = np.maximum(np.abs(t), 1e-300) if rel else 1.0

    def e_of(d):
        c = _trim(cheb_fit(f, lo, hi, d, parity), err if not rel else err * float(np.abs(t).min()))
        y = _C.chebval((2.0 * x - (lo + hi)) / (hi - lo), c)
        e = np.abs(y - t) / den
        return c, float(e.max()), float(np.sqrt(np.mean(e * e)))

    prev = 0
    for d in degrees:
        c, e, r = e_of(d)
        if e <= err:
            lo_d, hi_d, best = prev, d, (c, e, r)
            while hi_d - lo_d > 1:
                mid = (lo_d + hi_d) // 2
                cm, em, rm = e_of(mid)
                if em <= err:
                    hi_d, best = mid, (cm, em, rm)
                else:
                    lo_d = mid
            return best
        prev = d
    return None


def _cheb_cand(fn, f, lo, hi, err, rel, parity=None, scale=1.0):
    """`cheb`: scale * f by the cheapest interpolant meeting err (None if degree MAX_DEGREE does not)."""
    g = (lambda x: scale * f(x)) if scale != 1.0 else f
    tgt = err * (_abs(scale) if not rel else 1.0)
    a = None
    for frac in (0.8, 0.5, 0.25, 0.1):          # room for the noise term: a tighter fit when the noise needs it
        r = _cheb_degree(g, lo, hi, frac * tgt, rel, parity)
        if r is None:
            break
        c = r[0]
        a = _make(fn, "cheb", g, lo, hi, rel, _cheb_run(c, lo, hi), (("degree", len(c) - 1),))
        if a.err_enc <= tgt:
            break
    if a is None:
        return None
    if scale != 1.0 and not rel:
        a = dataclasses.replace(a, err=a.err / _abs(scale), rms=a.rms / _abs(scale))
    return a


def _check(lo, hi, positive=False, nonneg=False):
    lo, hi = float(lo), float(hi)
    if not hi > lo:
        raise ValueError(f"domain [{lo}, {hi}]: need lo < hi")
    if positive and not lo > 0.0:
        raise ValueError(f"domain [{lo}, {hi}]: need lo > 0")
    if nonneg and lo < 0.0:
        raise ValueError(f"domain [{lo}, {hi}]: need lo >= 0")
    return lo, hi


def _memo(f):
    return functools.lru_cache(maxsize=512)(f)


# ── exp / log ───────────────────────────────────────────────────────────────────────────────────────────────────────


@_memo
def approx_exp(lo: float, hi: float, err: float = 1e-4, rel: bool = True, out_scale: float = 1.0) -> Approx:
    """out_scale * exp(x) on [lo, hi]: `cheb`, or `cheb+sq` (k squarings of out_scale^(1/2^k) exp(x / 2^k))."""
    lo, hi = _check(lo, hi)
    f = lambda x: out_scale * np.exp(x)
    cands = [_cheb_cand("exp", np.exp, lo, hi, err, rel, scale=out_scale)]
    for k in range(1, 13):
        s = 2.0 ** -k
        g = lambda x, s=s: out_scale ** s * np.exp(s * x)
        # the series' relative error doubles per squaring; its absolute target follows from the relative one
        e_k = err * s if rel else err * s / max(float(f(np.array([hi]))[0]), 1e-300)
        r = _cheb_degree(g, lo, hi, 0.8 * e_k, True)
        if r is None:
            continue
        c, kk = r[0], k

        def run(ops, x, c=[float(v) for v in c], kk=kk):
            y = poly.eval_chebyshev(ops, x, c, lo, hi, eps=0.0)
            for _ in range(kk):
                y = ops.square(y)
            return y
        cands.append(_make("exp", "cheb+sq", f, lo, hi, rel, run, (("degree", len(c) - 1), ("squarings", kk))))
    return _best(f"exp on [{lo}, {hi}]", cands, err)


def exp(ops, x, lo, hi, err=1e-4, rel=True, out_scale=1.0):
    """out_scale * exp(x) for x in [lo, hi] (module doc). The output's size is out_scale exp(hi)."""
    return approx_exp(float(lo), float(hi), float(err), bool(rel), float(out_scale))(ops, x)


@_memo
def approx_log(lo: float, hi: float, err: float = 1e-4, rel: bool = False) -> Approx:
    """log x on [lo, hi] (lo > 0): `cheb`, or `atanh` (log c + 2 atanh((x - c) / (x + c)), c = sqrt(lo hi))."""
    lo, hi = _check(lo, hi, positive=True)
    cands = [_cheb_cand("log", np.log, lo, hi, err, rel)]
    c0 = _m.sqrt(lo * hi)
    if hi / lo > 4.0:
        lc = _m.log(c0)
        # the division: 1 / (x + c) on [lo + c, hi + c] to a quarter of the target (t's error enters 2 / (1 - t^2) <=
        # (hi + c)^2 / (2 hi c) amplified)
        amp = (hi + c0) ** 2 / (2.0 * hi * c0)
        tb = (hi - c0) / (hi + c0)
        try:
            rc = approx_reciprocal(lo + c0, hi + c0, 0.25 * err / amp / 2.0 / max(1.0, _abs(hi - c0)), rel=False)
        except TargetError:
            rc = None
        r = _cheb_degree(lambda t: 2.0 * np.arctanh(t), -tb, tb, 0.5 * err if not rel else 0.5 * err * 1e-3, False,
                         parity=1) if rc is not None else None
        if r is not None:
            ca = [float(v) for v in r[0]]

            def run(ops, x, rc=rc, ca=ca, tb=tb, lc=lc):
                inv = rc(ops, ops.add(x, c0))
                t = ops.mult(ops.add(x, -c0), inv)
                return ops.add(poly.eval_chebyshev(ops, t, ca, -tb, tb, eps=0.0), lc)
            cands.append(_make("log", "atanh", np.log, lo, hi, rel, run,
                               (("reciprocal", rc.method), ("degree", len(ca) - 1))))
    return _best(f"log on [{lo}, {hi}]", cands, err)


def log(ops, x, lo, hi, err=1e-4, rel=False):
    """log x for x in [lo, hi], lo > 0."""
    return approx_log(float(lo), float(hi), float(err), bool(rel))(ops, x)


# ── inverse square root, reciprocal, square root ────────────────────────────────────────────────────────────────────


def _rsqrt_newton_run(c, lo, hi, iters):
    c = [float(v) for v in c]

    def run(ops, x):
        y = poly.eval_chebyshev(ops, x, c, lo, hi, eps=0.0)
        return poly.inv_sqrt_newton_d2(ops, x, y, iters) if iters else y
    return run


def _scaled_newton_run(lo, hi, steps):
    """poly.inv_sqrt_newton_scaled on x / hi, the 1 / hi riding every step's b (no level) and 1 / sqrt(hi) the last
    step's constants."""
    st = [(a, b / hi) for a, b in steps]

    def run(ops, x):
        return poly.inv_sqrt_newton_scaled(ops, x, st, 1.0 / _m.sqrt(hi))
    return run


@_memo
def approx_rsqrt(lo: float, hi: float, err: float = 1e-4, rel: bool = True) -> Approx:
    """1 / sqrt(x) on [lo, hi], lo > 0: `cheb`, `cheb+newton`, `scaled-newton`."""
    lo, hi = _check(lo, hi, positive=True)
    f = lambda x: 1.0 / np.sqrt(x)
    cands = [_cheb_cand("rsqrt", f, lo, hi, err, rel)]
    er = 0.8 * (err if rel else err * _m.sqrt(lo))  # relative target that meets an absolute one (room for noise)
    for iters in (1, 2, 3, 4):
        # Newton: e -> 1.5 e^2 (relative); the seed's relative error that reaches er after `iters` steps
        e0 = er
        for _ in range(iters):
            e0 = _m.sqrt(e0 / 1.5)
        e0 = min(0.5, 0.8 * e0)
        r = _cheb_degree(f, lo, hi, e0, True, degrees=_DEGREES[:12])
        if r is None:
            continue
        cands.append(_make("rsqrt", "cheb+newton", f, lo, hi, rel, _rsqrt_newton_run(r[0], lo, hi, iters),
                           (("degree", len(r[0]) - 1), ("newton", iters))))
    try:
        steps = poly.scaled_newton_steps(hi / lo, 0.5 * er)
        cands.append(_make("rsqrt", "scaled-newton", f, lo, hi, rel, _scaled_newton_run(lo, hi, steps),
                           (("steps", len(steps)),)))
    except (ValueError, ZeroDivisionError, OverflowError):
        pass
    return _best(f"rsqrt on [{lo}, {hi}]", cands, err)


def rsqrt(ops, x, lo, hi, err=1e-4, rel=True):
    """1 / sqrt(x) for x in [lo, hi], lo > 0."""
    return approx_rsqrt(float(lo), float(hi), float(err), bool(rel))(ops, x)


def _recip_newton_run(c, lo, hi, iters):
    c = [float(v) for v in c]

    def run(ops, x):
        y = poly.eval_chebyshev(ops, x, c, lo, hi, eps=0.0)
        nx = ops.mult(x, -1.0)
        for _ in range(iters):                      # y (2 - x y)
            y = ops.mult(y, ops.add(ops.mult(nx, y), 2.0))
        return y
    return run


def _goldschmidt_run(lo, hi, iters):
    """poly.goldschmidt_recip with the Chebyshev-optimal linear seed of 1 / x on [lo, hi] (calibrate.numerics
    chebyshev_gs_init): F_init(x) = alpha - beta x."""
    # chebyshev_gs_init: max |1 - x F_init| = (hi - lo)^2 / (lo^2 + 6 lo hi + hi^2), equioscillating at lo, hi and
    # the vertex
    beta = 8.0 / (lo * lo + 6.0 * lo * hi + hi * hi)
    alpha = beta * (lo + hi)

    def run(ops, x):
        return poly.goldschmidt_recip(ops, x, alpha, beta, iters)
    return run


@_memo
def approx_reciprocal(lo: float, hi: float, err: float = 1e-4, rel: bool = True) -> Approx:
    """1 / x on [lo, hi], lo > 0: `cheb`, `cheb+newton`, `goldschmidt`, `rsqrt^2`."""
    lo, hi = _check(lo, hi, positive=True)
    f = lambda x: 1.0 / x
    cands = [_cheb_cand("reciprocal", f, lo, hi, err, rel)]
    er = 0.8 * (err if rel else err * lo)
    for iters in (1, 2, 3, 4):
        e0 = er                                     # Newton: e -> e^2
        for _ in range(iters):
            e0 = _m.sqrt(e0)
        e0 = min(0.5, 0.8 * e0)
        r = _cheb_degree(f, lo, hi, e0, True, degrees=_DEGREES[:12])
        if r is None:
            continue
        cands.append(_make("reciprocal", "cheb+newton", f, lo, hi, rel, _recip_newton_run(r[0], lo, hi, iters),
                           (("degree", len(r[0]) - 1), ("newton", iters))))
    e0 = (hi - lo) ** 2 / (lo * lo + 6.0 * lo * hi + hi * hi)   # the linear seed's |1 - x F| peak
    for iters in range(1, 24):
        if e0 ** (2 ** (iters - 1)) <= 0.5 * er:
            cands.append(_make("reciprocal", "goldschmidt", f, lo, hi, rel, _goldschmidt_run(lo, hi, iters),
                               (("iters", iters),)))
            break
    try:
        rs = approx_rsqrt(lo, hi, 0.45 * er, True)
        cands.append(_make("reciprocal", "rsqrt^2", f, lo, hi, rel, lambda ops, x, rs=rs: ops.square(rs(ops, x)),
                           (("rsqrt", rs.method),)))
    except TargetError:
        pass
    return _best(f"reciprocal on [{lo}, {hi}]", cands, err)


def reciprocal(ops, x, lo, hi, err=1e-4, rel=True):
    """1 / x for x in [lo, hi], lo > 0."""
    return approx_reciprocal(float(lo), float(hi), float(err), bool(rel))(ops, x)


@_memo
def approx_sqrt(lo: float, hi: float, err: float = 1e-4, rel: bool = False) -> Approx:
    """sqrt x on [lo, hi], lo >= 0: `cheb` (any lo >= 0), `x*rsqrt` (lo > 0)."""
    lo, hi = _check(lo, hi, nonneg=True)
    cands = [_cheb_cand("sqrt", np.sqrt, lo, hi, err, rel)]
    if lo > 0.0:
        er = err if rel else err / _m.sqrt(hi)
        try:
            rs = approx_rsqrt(lo, hi, 0.9 * er, True)
            cands.append(_make("sqrt", "x*rsqrt", np.sqrt, lo, hi, rel,
                               lambda ops, x, rs=rs: ops.mult(x, rs(ops, x)), (("rsqrt", rs.method),)))
        except TargetError:
            pass
    return _best(f"sqrt on [{lo}, {hi}]", cands, err)


def sqrt(ops, x, lo, hi, err=1e-4, rel=False):
    """sqrt x for x in [lo, hi], lo >= 0."""
    return approx_sqrt(float(lo), float(hi), float(err), bool(rel))(ops, x)


# ── saturating functions ────────────────────────────────────────────────────────────────────────────────────────────


def _erf(x):
    from scipy.special import erf as e
    return e(x)


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x)))


_SAT = {"tanh": (np.tanh, 1, 0.0), "erf": (_erf, 1, 0.0), "sigmoid": (_sigmoid, 1, 0.5)}


@_memo
def approx_saturating(name: str, lo: float, hi: float, err: float = 1e-4) -> Approx:
    """tanh / erf / sigmoid on [lo, hi] by `cheb` (absolute error); on a symmetric domain the odd part's parity is
    enforced (sigmoid = 1/2 + an odd function: its constant rides the series)."""
    lo, hi = _check(lo, hi)
    f, par, c0 = _SAT[name]
    sym = _abs(lo + hi) <= 1e-12 * max(_abs(lo), _abs(hi))
    if sym:
        cands = []
        for frac in (0.8, 0.5, 0.25, 0.1):
            r = _cheb_degree(lambda x: f(x) - c0, lo, hi, frac * err, False, parity=par)
            if r is None:
                break
            c = np.array(r[0])
            c[0] += c0
            cands.append(_make(name, "cheb", f, lo, hi, False, _cheb_run(c, lo, hi),
                               (("degree", len(c) - 1), ("parity", "odd"))))
            if cands[-1].err_enc <= err:
                break
    else:
        cands = [_cheb_cand(name, f, lo, hi, err, False)]
    return _best(f"{name} on [{lo}, {hi}]", cands, err)


def tanh(ops, x, lo, hi, err=1e-4):
    return approx_saturating("tanh", float(lo), float(hi), float(err))(ops, x)


def sigmoid(ops, x, lo, hi, err=1e-4):
    return approx_saturating("sigmoid", float(lo), float(hi), float(err))(ops, x)


def erf(ops, x, lo, hi, err=1e-4):
    return approx_saturating("erf", float(lo), float(hi), float(err))(ops, x)


# ── powers ──────────────────────────────────────────────────────────────────────────────────────────────────────────


def _pow_int_run(p):
    """x^p, p >= 1, by binary powering (squares, then the set bits multiplied in, balanced)."""
    def run(ops, x):
        sq, terms, b, k = x, [], p, 0
        while b:
            if b & 1:
                terms.append(sq)
            b >>= 1
            if b:
                sq = ops.square(sq)
        while len(terms) > 1:
            nxt = [ops.mult(terms[i], terms[i + 1]) for i in range(0, len(terms) - 1, 2)]
            if len(terms) % 2:
                nxt.append(terms[-1])
            terms = nxt
        return terms[0]
    return run


@_memo
def approx_pow(p: float, lo: float, hi: float, err: float = 1e-4, rel: bool = True) -> Approx:
    """x^p: integer p >= 1 on any domain by squarings (exact); p in {1/2, -1/2, -1} as sqrt / rsqrt / reciprocal; any
    other real p on lo > 0 (lo >= 0 for p > 0) by `cheb` or `exp(p log x)`."""
    p = float(p)
    if p == int(p) and p >= 1:
        lo, hi = _check(lo, hi)
        f = lambda x: np.asarray(x) ** int(p)
        return _make("pow", "squarings", f, lo, hi, rel if lo > 0 else False, _pow_int_run(int(p)),
                     (("p", int(p)),))
    if p == 0.5:
        return approx_sqrt(lo, hi, err, rel)
    if p == -0.5:
        return approx_rsqrt(lo, hi, err, rel)
    if p == -1.0:
        return approx_reciprocal(lo, hi, err, rel)
    lo, hi = _check(lo, hi, positive=p < 0, nonneg=p > 0)
    f = lambda x: np.asarray(x) ** p
    cands = [_cheb_cand("pow", f, lo, hi, err, rel)]
    if lo > 0.0:
        try:
            lg = approx_log(lo, hi, 0.3 * err / _abs(p) if rel else 0.3 * err / _abs(p) / max(hi ** p, lo ** p), False)
            a, b = sorted((p * _m.log(lo), p * _m.log(hi)))
            ex = approx_exp(a, b, 0.6 * err if rel else 0.6 * err / max(hi ** p, lo ** p), True)

            def run(ops, x, lg=lg, ex=ex):
                return ex(ops, ops.mult(lg(ops, x), p))
            cands.append(_make("pow", "exp(p log x)", f, lo, hi, rel, run, (("log", lg.method), ("exp", ex.method))))
        except TargetError:
            pass
    return _best(f"pow {p} on [{lo}, {hi}]", cands, err)


def pow(ops, x, p, lo, hi, err=1e-4, rel=True):  # noqa: A001  (the module's name space, like numpy.power)
    """x^p for x in [lo, hi] (approx_pow)."""
    return approx_pow(float(p), float(lo), float(hi), float(err), bool(rel))(ops, x)


# ── sign and its relatives: CKKS composites ─────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class SignLadder:
    """A composite odd minimax sign on [-1, 1] \\ (-eps, eps): `coefs` per stage on [-1, 1] (stage i's coefficients
    carry 1 / h_{i+1}, relu.unit_stages), |sgn - p| <= `delta` on [eps, 1]; `half` = the last stage as (1 + s) / 2
    (relu_folded's form)."""
    degs: tuple
    eps: float
    delta: float
    depth: int
    ct_mults: int
    coefs: tuple
    half: tuple


def _sign_value(coefs, t):
    y = np.asarray(t, dtype=np.float64)
    for c in coefs:
        y = _C.chebval(y, c)
    return y


def _odd_minimax(d, lo, hi, n_grid=2000):
    """calibrate/approximations/relu.py `_odd_minimax` (that module imports torch at load): the odd Chebyshev
    coefficients (domain [-hi, hi]) minimising max |p - 1| on [lo, hi] with p in [-delta, 1 + delta] on [0, lo]."""
    from scipy.optimize import linprog
    if not (0.0 < lo < hi):
        raise TargetError(f"sign stage d={d}: interval [{lo}, {hi}]")
    ks = np.arange(1, d + 1, 2)
    xa = hi * (lo / hi + (1 - lo / hi) * (1 - np.cos(np.linspace(0, np.pi, n_grid))) / 2)
    xg = np.linspace(0.0, lo, max(n_grid // 4, 50))
    Ta = np.cos(np.outer(np.arccos(np.clip(xa / hi, -1, 1)), ks))
    Tg = np.cos(np.outer(np.arccos(np.clip(xg / hi, -1, 1)), ks))
    n = len(ks)
    one, gone = np.ones((len(xa), 1)), np.ones((len(xg), 1))
    A = np.vstack([np.hstack([Ta, -one]), np.hstack([-Ta, -one]), np.hstack([Tg, -gone]), np.hstack([-Tg, -gone])])
    b = np.concatenate([np.ones(len(xa)), -np.ones(len(xa)), np.ones(len(xg)), np.zeros(len(xg))])
    cost = np.zeros(n + 1)
    cost[-1] = 1.0
    r = linprog(cost, A_ub=A, b_ub=b, bounds=[(None, None)] * n + [(0, None)], method="highs")
    if not r.success:
        raise TargetError(f"sign stage d={d} [{lo}, {hi}]: {r.message}")
    coef = np.zeros(d + 1)
    coef[ks] = r.x[:n]
    return coef


@functools.lru_cache(maxsize=4096)
def _stage(d, lo, hi):
    """One composite-sign stage: the odd minimax polynomial of degree d on [-hi, hi] for sgn on [lo, hi]
    (calibrate/approximations/relu.py `_odd_minimax`) and its image (min on [lo, hi], max |p| on [0, hi])."""
    coef = _odd_minimax(int(d), lo, hi)
    ys = _C.chebval(np.linspace(lo, hi, 4000) / hi, coef)
    top = float(np.abs(_C.chebval(np.linspace(0.0, hi, 8000) / hi, coef)).max())
    return coef, float(ys.min()), top


def _cache_path():
    import os
    d = os.environ.get("PERSEUS_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "perseus")
    return os.path.join(d, "math_sign_ladders.json")


def _disk_cached(f):
    """sign_ladder's results kept on disk (the search fits ~50-300 minimax LPs: seconds per new (eps, delta)); the
    file is a cache only (PERSEUS_CACHE_DIR, default ~/.cache/perseus), deterministic to recompute."""
    import json
    import os

    @functools.wraps(f)
    def g(eps, delta, relu=False):
        key = f"{float(eps)!r}|{float(delta)!r}|{int(bool(relu))}|m{LADDER_MARGIN!r}"
        path = _cache_path()
        try:
            with open(path) as fh:
                d = json.load(fh).get(key)
            if d is not None:
                return SignLadder(tuple(d["degs"]), d["eps"], d["delta"], d["depth"], d["ct_mults"],
                                  tuple(tuple(v) for v in d["coefs"]), tuple(tuple(v) for v in d["half"]))
        except (OSError, ValueError, KeyError):
            pass
        lad = f(eps, delta, relu)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                with open(path) as fh:
                    allc = json.load(fh)
            except (OSError, ValueError):
                allc = {}
            allc[key] = dataclasses.asdict(lad)
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w") as fh:
                json.dump(allc, fh)
            os.replace(tmp, path)
        except OSError:
            pass
        return lad
    return g


# A stage past the first is fitted on its input's image widened by this fraction, so its Chebyshev domain is not
# the image itself: in unit form every intermediate's maximum sat exactly at 1, the edge of the domain, and a refresh
# between stages (relative error ~1e-3 on unit-size data, plan/btserr.py) evaluated a degree 15..63 stage outside
# [-1, 1], where T_k grows like (t + sqrt(t^2 - 1))^k: each later stage compounded it to ~1e19 (CKKS bitonic n >= 1024,
# threshold n 4096, measured 2026-11-20). With the margin an input up to (1 + margin) x the image still maps to
# +-1 within delta. 1/64 keeps every ladder stable under 3e-3 of noise between stages (the dense route's worst
# unit-size refresh error is ~2.8e-3 at CF 7) and leaves the ladders' shapes as they were, but sign at 1e-4 (+1 level);
# 1/16 also holds 1e-2 but costs the ReLU at 1e-3 16 more products.
LADDER_MARGIN = 1.0 / 64


@_memo
@_disk_cached
def sign_ladder(eps: float, delta: float, relu: bool = False) -> SignLadder:
    """The cheapest composite sign (fewest levels, then products) with |p - sgn| <= delta on [eps, 1]: a first stage of
    degree d1 (7, 15, 31, 63), middle stages of one degree d (7, 15, 31, 63, <= d1 past the first) and, optionally, a
    last stage of another degree (15, 27, 31, 63: a wide last stage, as the ResNet-20 ladder 7, 7, 27), each stage
    the minimax odd polynomial on the previous stage's image (calibrate/approximations/relu.py, Lee et al. IEEE TDSC
    2021), grown one stage at a time (at most 12) until the target holds or the cost passes the best so far. `relu`:
    the target is the ReLU's instead, max over t in [0, 1] (the gap included) of t |1 - p(t)| / 2 <= delta."""
    eps, delta = float(eps), float(delta)
    if relu:
        xs = np.concatenate([np.geomspace(1e-7, 1.0, 4000), np.linspace(0.0, 1.0, 4000)])
    else:
        xs = np.concatenate([np.geomspace(eps, 1.0, 3000), np.linspace(eps, 1.0, 3000)])
    wt = 0.5 * xs if relu else 1.0
    shapes = [(d, d) for d in (7, 15, 31, 63)] + [(d1, d) for d1 in (15, 31, 63) for d in (7, 15) if d < d1]
    best = None

    def unit(stages):
        # unit form: stage i's coefficients carry 1 / h_{i+1}; the last one maps to +-1
        return [list(np.asarray(c) / (stages[q + 1][1] if q + 1 < len(stages) else 1.0))
                for q, (c, _h) in enumerate(stages)]

    last = [np.inf]

    def offer(coefs, degs):
        nonlocal best
        err = float((wt * np.abs(_sign_value(coefs, xs) - 1.0)).max())
        last[0] = err
        if err > delta:
            return False
        dep, cm = _ladder_cost(coefs)
        if best is None or (dep, cm) < (best.depth, best.ct_mults):
            half = [list(c) for c in coefs[:-1]] + [list(0.5 * np.asarray(coefs[-1]))]
            half[-1][0] += 0.5
            best = SignLadder(tuple(int(v) for v in degs), eps, err, dep, cm, tuple(tuple(v) for v in coefs),
                              tuple(tuple(v) for v in half))
        return True

    for d1, d in shapes:
        stages, degs, lo, hi = [], [], eps, 1.0
        for m in range(12):
            dm = d1 if m == 0 else d
            # past the first stage the domain reaches LADDER_MARGIN beyond the previous image (see LADDER_MARGIN)
            H = hi if m == 0 else hi * (1.0 + LADDER_MARGIN)
            if m:       # a different last stage on the prefix
                for dl in (15, 27, 31, 63):
                    if dl != d and lo > 0.0:
                        coef, _l, _t = _stage(dl, lo, H)
                        offer(unit(stages + [(coef, H)]), degs + [dl])
            coef, lo2, top = _stage(dm, lo, H)
            if not (0.0 < lo2 <= top and np.isfinite(top)):
                break               # a degenerate stage (the gap's image collapsed): this shape cannot reach it
            stages.append((coef, H))
            degs.append(dm)
            lo, hi = lo2, top
            coefs = unit(stages)
            prev = last[0]
            if offer(coefs, degs):
                break
            if best is not None and _ladder_cost(coefs) >= (best.depth, best.ct_mults):
                break
            if m >= 2 and last[0] > 0.9 * prev:
                break               # stalled: the gap's own error floor (a wider gap than the target allows)
    if best is None:
        raise TargetError(f"sign: no ladder reaches delta {delta:.3g} at eps {eps:.3g}")
    return best


def _ladder_cost(coefs):
    from .relu import odd_series
    c = CountOps()
    z = c.leaf()
    for cc in coefs:
        z = odd_series(c, z, list(cc))
    return c.depth, c.ct_mults


def sign_ckks(ops, t, lad: SignLadder):
    """sgn(t) for t in [-1, 1] (|t| >= eps: within delta) by the ladder; an exact 0 maps to 0 (odd stages)."""
    from .relu import odd_series
    for c in lad.coefs:
        t = odd_series(ops, t, list(c))
    return t


def _resolution(bound, err):
    """(eps, delta) of a sign whose product with x (|x| <= bound) is within err: the gap and the band's error."""
    r = min(0.25, float(err) / float(bound))
    return r, r


def _scaled(ops, x, s):
    return x if s == 1.0 else ops.mult(x, s)


@_memo
def relu_ladder(bound: float, err: float) -> SignLadder:
    """The cheapest sign ladder whose ReLU x (1 + p(x / bound)) / 2 is within err on [-bound, bound]: the error is
    |x| |1 - p(|x|)| / 2, small inside the gap too (the composite rises through it), so the ladders are fitted for
    gaps from err / bound upwards and each one judged by that error directly (sign_ladder relu=True)."""
    r = float(err) / float(bound)
    best = None
    for f in (1.0, 4.0, 16.0):   # inside the gap the error is ~eps / 8 at best: wider gaps cannot pass
        eps = min(0.25, f * r)
        try:
            lad = sign_ladder(eps, r, True)
        except TargetError:
            if best is not None:
                break               # a wider gap only raises the floor
            continue
        if best is None or (lad.depth, lad.ct_mults) < (best.depth, best.ct_mults):
            best = lad
        if eps >= 0.25:
            break
    if best is None:
        raise TargetError(f"relu: no ladder reaches {err:.3g} on [-{bound}, {bound}]")
    return best


def relu_ckks(ops, x, bound, err, gain=1.0):
    """gain * max(x, 0) for |x| <= bound within gain * err: x * (1 + p(x / bound)) / 2 (relu.relu_folded: the product
    by x rides the last stage)."""
    from .relu import relu_folded
    lad = relu_ladder(float(bound), float(err))
    return relu_folded(ops, x if bound == 1.0 else ops.mult(x, 1.0 / bound), [list(c) for c in lad.half],
                       gain=gain * bound)


# ── sign and its relatives: the TFHE switch ─────────────────────────────────────────────────────────────────────────


def _heads(ops, x, slots):
    n = int(getattr(getattr(ops, "inf", None), "slots", 0) or (np.shape(x)[0] if isinstance(x, np.ndarray) else 0))
    if n == 0:
        n = int(getattr(x, "vec", np.zeros(1)).shape[0])
    idx = np.arange(n) if slots is None else np.asarray(slots, dtype=np.int64)
    return n, idx


def step_tfhe(ops, d, bound, slots=None, levels=1, tag="m"):
    """[d >= 0] (0 / 1) on `slots` (default all), 0 elsewhere, for |d| < bound, by the scheme switch
    (tfhe_select.msign: a sign LUT, cleaned; `levels` > 1: finer scales in the same switch, needs free slots)."""
    from .tfhe_select import msign
    n, idx = _heads(ops, d, slots)
    return msign(ops, d, n, idx, float(bound), int(levels), f"math.{tag}")


# ── the public sign family ──────────────────────────────────────────────────────────────────────────────────────────


def sign(ops, x, bound, err=1e-3, method="ckks", slots=None):
    """sgn(x) for |x| <= bound: within err for |x| >= err * bound (ckks); exact (0 / 1 then 2 s - 1) outside the
    switch's noise band (tfhe)."""
    if method == "tfhe":
        s = step_tfhe(ops, x, bound, slots, tag="sign")
        return ops.add(ops.add(s, s), -1.0)
    eps, delta = _resolution(1.0, err)
    return sign_ckks(ops, _scaled(ops, x, 1.0 / bound), sign_ladder(eps, delta))


def relu(ops, x, bound, err=1e-3, method="ckks", slots=None):
    """max(x, 0) for |x| <= bound, within err (ckks; the error is largest at |x| ~ err), or x [x >= 0] (tfhe: exact
    outside the switch's noise band, where the error is <= |x|)."""
    if method == "tfhe":
        return ops.mult(x, step_tfhe(ops, x, bound, slots, tag="relu"))
    return relu_ckks(ops, x, float(bound), float(err))


def abs(ops, x, bound, err=1e-3, method="ckks", slots=None):  # noqa: A001
    """|x| for |x| <= bound: 2 relu(x) - x."""
    if method == "tfhe":
        s = step_tfhe(ops, x, bound, slots, tag="abs")
        xs = ops.mult(x, s)
        return ops.sub(ops.add(xs, xs), x)
    return ops.sub(relu_ckks(ops, x, float(bound), 0.5 * float(err), gain=2.0), x)


def maximum(ops, a, b, bound, err=1e-3, method="ckks", slots=None):
    """max(a, b) for |a - b| <= bound: b + relu(a - b)."""
    return ops.add(b, relu(ops, ops.sub(a, b), bound, err, method, slots))


def minimum(ops, a, b, bound, err=1e-3, method="ckks", slots=None):
    """min(a, b) for |a - b| <= bound: a - relu(a - b)."""
    return ops.sub(a, relu(ops, ops.sub(a, b), bound, err, method, slots))


def clip(ops, x, lo, hi, bound, err=1e-3, method="ckks", slots=None):
    """min(max(x, lo), hi) for |x| <= bound: lo + relu(x - lo) - relu(x - hi) (two signs at the same depth)."""
    lo, hi = float(lo), float(hi)
    B = float(bound) + max(_m.fabs(lo), _m.fabs(hi))
    r1 = relu(ops, ops.add(x, -lo), B, 0.5 * err, method, slots)
    r2 = relu(ops, ops.add(x, -hi), B, 0.5 * err, method, slots)
    return ops.add(ops.sub(r1, r2), lo)


def approx_relu(bound: float, err: float = 1e-3):
    """The CKKS relu's ladder and cost for |x| <= bound within err (no evaluation): (SignLadder, depth, products,
    float64 max error on [-bound, bound])."""
    lad = relu_ladder(float(bound), float(err))
    x = grid(-bound, bound)
    y = np.real(relu_ckks(_NpOps(), x.copy(), float(bound), float(err)))
    c = count(lambda o, v: relu_ckks(o, v, float(bound), float(err)))
    return lad, c.depth, c.ct_mults, float(np.abs(y - np.maximum(x, 0.0)).max())


# ── the catalogue ───────────────────────────────────────────────────────────────────────────────────────────────────


def approx(fn: str, lo: float, hi: float, err: float = 1e-4, **kw) -> Approx:
    """The chosen method for `fn` on [lo, hi] (exp, log, sqrt, rsqrt, reciprocal, tanh, sigmoid, erf, pow (p=...))."""
    if fn == "exp":
        return approx_exp(lo, hi, err, kw.get("rel", True), kw.get("out_scale", 1.0))
    if fn == "log":
        return approx_log(lo, hi, err, kw.get("rel", False))
    if fn == "sqrt":
        return approx_sqrt(lo, hi, err, kw.get("rel", False))
    if fn == "rsqrt":
        return approx_rsqrt(lo, hi, err, kw.get("rel", True))
    if fn == "reciprocal":
        return approx_reciprocal(lo, hi, err, kw.get("rel", True))
    if fn in _SAT:
        return approx_saturating(fn, lo, hi, err)
    if fn == "pow":
        return approx_pow(kw["p"], lo, hi, err, kw.get("rel", True))
    raise KeyError(fn)


def candidates(fn: str, lo: float, hi: float, err: float = 1e-4, **kw) -> list:
    """Every method tried for `fn` with its cost and error (the table behind `approx`'s choice), feasible or not."""
    _LAST.clear()
    try:
        _approx_uncached(fn, lo, hi, err, **kw)
    except TargetError:
        pass
    return list(_LAST.get("cands", []))


def _approx_uncached(fn, lo, hi, err, **kw):
    f = {"exp": approx_exp, "log": approx_log, "sqrt": approx_sqrt, "rsqrt": approx_rsqrt,
         "reciprocal": approx_reciprocal}.get(fn)
    if f is not None:
        rest = {k: v for k, v in kw.items() if k in ("rel", "out_scale")}
        return f.__wrapped__(lo, hi, err, **rest)
    if fn in _SAT:
        return approx_saturating.__wrapped__(fn, lo, hi, err)
    return approx_pow.__wrapped__(kw["p"], lo, hi, err, kw.get("rel", True))


# ── functional bootstrapping: a refresh that evaluates the function ─────────────────────────────────────────────────

#: Where a candidate replaces a refresh followed by a series (n32, real payload, GPU 3, 2026-11-20): the plain
#: real-payload StC-first refresh lands at level REFRESH_LANDING (realized) for REFRESH_MS per ciphertext (K = 1); an
#: FBS lands at its own level (impl/fbs.py `landing`) for REFRESH_MS + `fbs_extra_ms`; a series after the refresh costs
#: about SERIES_MS per ciphertext product (the shipped EvalMod: 4.07 ms for ~11 products at 46..34 limbs, scaled to
#: the 24..10 limbs below the refresh's landing), its levels below the refresh's landing being the Approx's depth.
REFRESH_LANDING = 26
REFRESH_MS = 14.13
SERIES_MS = 0.25
#: The FBS's EvalMod cost over a plain refresh (ms, real payload, K = 1; least squares over the 15-function campaign,
#: residual <= 0.7 ms for the paired sine, <= 3.7 ms for the others at degree 256 / 96): a + b x degree per form
#: (impl/fbs_fit.py `form_name`).
FBS_EXTRA_MS = {"sine": (1.96, 0.0114), "sine-paired": (2.35, 0.0197), "trig": (5.74, 0.0712),
                "trig-low": (8.62, 0.0757)}

#: The FBS fit's margin past the domain (fraction of its width; a positive domain's low end: of itself).
FBS_PAD = 0.02
_FBS_POSITIVE = ("log", "sqrt", "rsqrt", "reciprocal")
_FBS_F = {"exp": np.exp, "log": np.log, "sqrt": np.sqrt, "rsqrt": lambda x: 1.0 / np.sqrt(x),
          "reciprocal": lambda x: 1.0 / x, "tanh": np.tanh, "sigmoid": lambda x: _sigmoid(x), "erf": lambda x: _erf(x)}


def fbs_extra_ms(form: str, deg: int) -> float:
    a, b = FBS_EXTRA_MS[form]
    return a + b * deg


def _fbs_approx(fn, lo, hi, rel, name, fit, lv, bound, f):
    from . import fbs, fbs_fit
    fbs_fit.install(name, f, fit, lv, bound)
    fmax = float(np.abs(f(grid(lo, hi, 257))).max()) or 1.0
    depth = max(0, (lv - REFRESH_LANDING + 1) // CountOps.unit)
    form = fbs_fit.form_name(fit)
    ms = fbs_extra_ms(form, fit["deg"])

    def run(ops, x, name=name):
        return fbs.fbs(ops, x, name)
    return Approx(fn, f"fbs-{form}", lo, hi, depth, 0, 0, bound, float(fit["err"]), bool(rel),
                  (("form", form), ("degree", fit["deg"]), ("landing", lv), ("extra_ms", ms), ("name", name)), 0.0,
                  fmax, run)


@_memo
def fbs_candidates(fn: str, lo: float, hi: float, err: float = 1e-4, rel: bool = False, scale: float = 1.0) -> tuple:
    """f(scale x) on [lo, hi] as functional bootstraps (impl/fbs_fit.py: the sine and the trigonometric forms at the
    smallest degree meeting `err`, absolute or pointwise relative when `rel`, by the noise model's bound): Approx
    objects, each registered under its own name. Their `depth` counts the levels below a plain refresh's landing;
    `err` is the model's bound (fit + 13 sigma: 2^-128 per output), `params` the form, degree, landing, extra ms.
    The fit covers the domain widened by FBS_PAD of its width on each side (a positive domain's low end by FBS_PAD of
    itself): past its fitted interval a series on EvalMod's sine / cosine leaves [-1, 1] and its high-degree
    Chebyshev terms explode (a band overshoot of 0.7 % broke an SVM link of degree 96), where a series on x / X would
    only drift."""
    from . import fbs_fit
    lo, hi = _check(lo, hi)
    f0 = _FBS_F[fn]
    f = (lambda x: f0(scale * x)) if scale != 1.0 else f0
    tag = f"math.{fn}.{lo:.9g}.{hi:.9g}.{err:.3g}.{int(rel)}.{scale:.9g}"
    w = hi - lo
    flo = lo * (1.0 - FBS_PAD) if fn in _FBS_POSITIVE else lo - FBS_PAD * w
    fhi = hi + FBS_PAD * w
    out = []
    for i, (fit, lv, b) in enumerate(fbs_fit.feasible(tag, f, flo, fhi, err, rel)):
        name = f"{tag}.{fbs_fit.form_name(fit)}{fit['deg']}c{fit['cf']}"
        out.append(_fbs_approx(fn, lo, hi, rel, name, fit, lv, b, f))
    return tuple(out)


def approx_fbs(fn: str, lo: float, hi: float, err: float = 1e-4, rel: bool = False, scale: float = 1.0):
    """The shallowest-landing functional bootstrap of `fbs_candidates` (then the cheapest), or None."""
    c = fbs_candidates(fn, float(lo), float(hi), float(err), bool(rel), float(scale))
    return min(c, key=lambda a: (a.depth, dict(a.params)["extra_ms"])) if c else None


def fbs_choice(fn: str, lo: float, hi: float, err: float, rel: bool, plain_depth: int, scale: float = 1.0,
               series_ms: float | None = None):
    """The functional bootstrap of f(scale x), x in [lo, hi], that replaces a refresh followed by a series landing
    `plain_depth` levels below the refresh: the cheapest candidate (`fbs_candidates`: extra ms, then depth) that lands
    no deeper, meets `err` by its bound and, with `series_ms`, costs no more than the series; None if none does."""
    ok = [f for f in fbs_candidates(fn, float(lo), float(hi), float(err), bool(rel), float(scale))
          if f.depth <= plain_depth and f.err <= err
          and (series_ms is None or dict(f.params)["extra_ms"] <= series_ms)]
    return min(ok, key=lambda f: (dict(f.params)["extra_ms"], f.depth)) if ok else None


def refreshed(fn: str, lo: float, hi: float, err: float = 1e-4, fbs: bool = True, **kw):
    """The cheaper of a refresh followed by `approx(fn, ...)` and a functional bootstrap of fn (`approx_fbs`), for a
    value the caller refreshes anyway (a deliberate refresh, a stage entry the plan refreshes): the FBS is chosen only
    if it lands no deeper than the refreshed series and costs no more (REFRESH_MS + its extra ms against REFRESH_MS +
    SERIES_MS per product), with its bound within `err`. Returns (Approx, uses_fbs); calling the Approx on the
    unrefreshed value does the whole step (the refresh is inside: `ops.bootstrap_real`, or the FBS)."""
    rel = kw.get("rel", fn in ("exp", "rsqrt", "reciprocal"))
    a = approx(fn, lo, hi, err, **kw)
    plain = dataclasses.replace(a, run=lambda ops, x, a=a: a(ops, ops.bootstrap_real(x)))
    if not fbs or fn not in _FBS_F:
        return plain, False
    f = fbs_choice(fn, lo, hi, err, rel, a.depth, series_ms=SERIES_MS * a.ct_mults)
    return (plain, False) if f is None else (f, True)
