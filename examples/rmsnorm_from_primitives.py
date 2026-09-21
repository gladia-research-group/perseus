"""RMSNorm built from the session's leaf primitives.

    y = x * rsqrt(mean(x^2) + eps) * gamma

is computed on a token ciphertext with the primitives `Context.square / mult`,
`Inference.sum_slots`, `Inference.eval_chebyshev` and `Inference.mult_pt`, and mirrored
in numpy with the SAME Chebyshev approximation so that the approximation error
(`rmsnorm_ref` vs `rmsnorm_exact`) is separable from the FHE noise (`rmsnorm` vs
`rmsnorm_ref`).

The chain, with the CKKS levels each step consumes:

    sq  = fhe.square(ct)                                        # x^2 slot-wise        1
    S   = inf.sum_slots(sq, inf.slots)                          # all-slot sum, bcast  0
    r   = inf.eval_chebyshev(S, coeffs, (a-eps)*d, (b-eps)*d)   # rsqrt(S/d + eps)     2 + ceil(log2 degree)
    xg  = inf.mult_pt(ct, gamma_slots)                          # x * gamma (parallel) 1
    out = fhe.mult(xg, r)                                       # ct * ct              1

Total depth: 4 + ceil(log2 degree) levels (`rmsnorm_depth`). `rmsnorm` returns the
output together with `ct.level` before and after.

Packing facts this example relies on (all in the cachemir packing, the session's default
for the gpt2 family; `Session.encrypt` -> `_core.encode_token_input`):

* A token ciphertext keeps real lane ``i`` at slot ``i * t`` with
  ``t = inf.slots // inf.size.hidDim`` -- `cachemir::encode_linear_input`
  (src/packing/cachemir/cachemir_linear_utils.cu: ``p.t = N / p.d`` with ``p.d = hidDim``,
  ``ptx[i * p.t] = x[i]``) and `decode_tokens` (``y[tok][i] = cy[i * t + tok]``, same file).
  `pack_tokens` (src/model/gpt2/gpt2_io.cu) zero-pads the ``d`` real features to
  ``hidDim`` first, so lanes ``d .. hidDim-1`` AND every off-lane slot ``i*t + k`` (k > 0)
  are exactly zero in a fresh encode.
* Because every non-lane slot is zero, the full rotate-and-add ladder
  ``sum_slots(ct, inf.slots)`` (steps 1, 2, 4, ..., slots/2) yields the sum over the ``d``
  real lanes broadcast to EVERY slot -- byte-for-byte the ladder the runtime's own
  LayerNorm variance uses (src/packing/cachemir/cachemir_norm_utils.cu,
  `compute_variance_interleaved`: "This ladder sums EVERY slot, so the result is a
  broadcast constant").
* The rotation keys 1, 2, 4, ..., slots/2 are in the gpt2 band
  (`collect_norm_rots`, src/packing/cachemir/cachemir_rot_indices.cu); `rmsnorm` checks
  them against `Context.loaded_rot_steps` before rotating.
* A plaintext vector for `mult_pt` follows the same layout: ``gamma[i]`` at slot ``i * t``
  (`pack_per_feature_vec`, cachemir_norm_utils.cu) -- `gamma_to_slots`.

Precondition: the input must be a FRESH token-basis ciphertext (what `Session.encrypt`
returns) or one rebased to it. A ciphertext that went through a cachemir linear carries
non-zero partial sums in the off-lane slots (include/slot_layout.h, "Derived" basis) and
the all-slot ladder would sum them too; `PackedCtx` does not expose the layout kind, so
this example cannot detect it.

Chebyshev convention: `rsqrt_cheb_coeffs` fits ``u ** -0.5`` on ``[a, b]`` with
`numpy.polynomial.chebyshev.chebinterpolate`; ``c0`` is NOT halved, which is both numpy's
`chebval` convention and the runtime's (`eval_chebyshev_series`,
src/primitives/polynomial.cu: ``r = sum_{k>=1} c_k T_k(y) + c_0``), so
``[0.5, 0, 0.5]`` on ``[-1, 1]`` is ``x^2``. The evaluator maps its argument onto
``[-1, 1]`` with ``y = (2x - (a + b)) / (b - a)``; feeding it the sum of squares ``S``
with the interval ``[(a-eps)d, (b-eps)d]`` gives the same ``y`` as feeding
``u = S/d + eps`` with ``[a, b]`` (`fhe_interval`), so the ``1/d`` and ``+eps`` cost no
level. The interval is a calibration window: `rmsnorm_ref` raises outside it; the FHE
side cannot check and the series diverges there.

Level budget: the runtime bootstraps reactively when ``ct.level >= fhe.level_limit()``
(include/fideslib_wrapper.h `maybe_bootstrap`; on the n32 chain both are PRIME counts and
one CKKS level is ``composite_degree`` primes). `level_budget(sess, ct)` returns the whole
levels left; under `SessionProfile.custom_n32()` a fresh encrypt lands at 34 with the
ceiling at 49 (AUTO_BTS_LEVEL; scripts/local_env.sh exports 46 instead), i.e. 7 levels
-> ``degree <= 7`` (5 levels -> ``degree <= 2`` with the ceiling at 46).

Usage (a live session; the GPU test in tests/gpu/test_primitives.py is the executable
form):

    from perseus import session
    from perseus.profile import SessionProfile
    from rmsnorm_from_primitives import rmsnorm, rmsnorm_ref, level_budget

    with session(profile=SessionProfile.custom_n32()) as s:
        ct = s.encrypt(x)                                   # x: 768 real features
        res = rmsnorm(s, ct, gamma, eps=1e-5, d=768, interval=(0.04, 0.16), degree=7)
        y = s.decrypt(res.ct, d=768)
        print(res.level_before, "->", res.level_after)       # 34 -> 48 (46: rescale pending)
        err = abs(y - rmsnorm_ref(x, gamma, eps=1e-5, interval=(0.04, 0.16), degree=7)).max()
"""
from __future__ import annotations

import functools
import logging
import math
from typing import NamedTuple

import numpy as np
from numpy.polynomial import chebyshev as _cheb

log = logging.getLogger(__name__)

DEFAULT_EPS = 1e-5
DEFAULT_INTERVAL = (0.04, 0.16)     # mean(x^2)+eps for x ~ 0.3 N(0,1), d=768: [0.074, 0.109]
DEFAULT_DEGREE = 7                  # the deepest degree that fits custom_n32's 7 levels

_SLOT_LAYOUT_PACKINGS = ("cachemir", "cachemir_complex")   # lane i at slot i*t (packed_ctx.h)


def _check_interval(interval) -> tuple[float, float]:
    """Validate a fitting interval: finite, ``0 < a < b`` (rsqrt needs a positive argument)."""
    try:
        a, b = (float(v) for v in interval)
    except (TypeError, ValueError):
        raise ValueError(f"interval must be an (a, b) pair, got {interval!r}") from None
    if not (math.isfinite(a) and math.isfinite(b)) or not 0.0 < a < b:
        raise ValueError(f"interval must satisfy 0 < a < b with finite bounds, got {interval!r}")
    return a, b


@functools.cache
def _fit(a: float, b: float, degree: int) -> tuple[float, ...]:
    mid, half = (a + b) / 2.0, (b - a) / 2.0
    c = _cheb.chebinterpolate(lambda y: 1.0 / np.sqrt(mid + half * y), degree)
    return tuple(float(v) for v in c)


def rsqrt_cheb_coeffs(interval, degree: int) -> tuple[float, ...]:
    """Chebyshev coefficients ``c_0 .. c_degree`` of ``u ** -0.5`` on ``interval``.

    Fitted with `numpy.polynomial.chebyshev.chebinterpolate` (interpolation at the
    Chebyshev points of the first kind); ``c_0`` is unhalved -- `chebval` and the runtime's
    `eval_chebyshev` agree on that convention. Cached per (interval, degree).
    """
    a, b = _check_interval(interval)
    degree = int(degree)
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")
    return _fit(a, b, degree)


DEFAULT_COEFFS = rsqrt_cheb_coeffs(DEFAULT_INTERVAL, DEFAULT_DEGREE)


def fhe_interval(interval, eps: float, d: int) -> tuple[float, float]:
    """The evaluator interval that makes ``eval_chebyshev(S, coeffs, a', b')`` on the sum of
    squares ``S`` equal the series on ``u = S/d + eps`` over ``[a, b]``.

    With ``a' = (a - eps) d`` and ``b' = (b - eps) d``:
    ``(2S - (a' + b')) / (b' - a') = (2 (S/d + eps) - (a + b)) / (b - a)``, the affine map
    the runtime applies (src/primitives/polynomial.cu) -- so the ``1/d`` and ``+eps`` are
    absorbed into a map that is evaluated anyway, saving one level.
    """
    a, b = _check_interval(interval)
    return (a - eps) * d, (b - eps) * d


def rmsnorm_depth(degree: int) -> int:
    """CKKS levels the chain consumes: square 1, affine map 1, ``T_degree`` tree
    ``ceil(log2 degree)``, weighted sum 1, final ciphertext product 1 (the gamma product
    runs in parallel and is absorbed by the final product's level alignment)."""
    degree = int(degree)
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")
    return 4 + math.ceil(math.log2(degree))


def _level_unit(sess) -> int:
    """Primes per CKKS level (`composite_degree`; 1 when the session does not say)."""
    ckks = getattr(getattr(sess, "options", None), "ckks", None)
    return int(getattr(ckks, "composite_degree", 1) or 1)


def level_budget(sess, ct) -> int:
    """Whole CKKS levels ``ct`` can consume before the runtime's reactive bootstrap fires
    (which happens once ``ct.level >= fhe.level_limit()``; both prime-granular on n32)."""
    limit = int(sess.inf.fhe.level_limit())
    return (limit - int(ct.level) - 1) // _level_unit(sess)


def gamma_to_slots(gamma, slots: int, t: int) -> np.ndarray:
    """Lay ``gamma`` out as the token packing does: ``gamma[i]`` at slot ``i * t``, zeros
    elsewhere (the `mult_pt` operand)."""
    g = np.asarray(gamma, dtype=np.float64).ravel()
    if g.shape[0] * t > slots:
        raise ValueError(f"{g.shape[0]} lanes at stride {t} do not fit in {slots} slots")
    out = np.zeros(int(slots))
    out[np.arange(g.shape[0]) * int(t)] = g
    return out


def rmsnorm_exact(x, gamma, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Closed-form ``x * rsqrt(mean(x^2) + eps) * gamma``."""
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    return x / np.sqrt(np.mean(x * x) + eps) * gamma


def rmsnorm_ref(x, gamma, *, eps: float = DEFAULT_EPS, interval=DEFAULT_INTERVAL,
                degree: int = DEFAULT_DEGREE) -> np.ndarray:
    """numpy mirror of `rmsnorm`: the SAME Chebyshev series for the rsqrt, so
    ``rmsnorm_ref - rmsnorm_exact`` is the approximation error and ``fhe - rmsnorm_ref`` the
    FHE noise. Raises ValueError when ``mean(x^2) + eps`` leaves ``interval``."""
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    a, b = _check_interval(interval)
    coeffs = rsqrt_cheb_coeffs((a, b), degree)
    u = float(np.mean(x * x) + eps)
    if not a <= u <= b:
        raise ValueError(f"mean(x^2)+eps = {u:.4g} lies outside the fitted interval [{a}, {b}]; "
                         "the Chebyshev series diverges there and the FHE side cannot check it")
    r = _cheb.chebval((2.0 * u - (a + b)) / (b - a), coeffs)
    return x * r * gamma


class RMSNormResult(NamedTuple):
    """`rmsnorm`'s output ciphertext with its level report."""
    ct: object
    level_before: int
    level_after: int
    degree: int
    coeffs: tuple[float, ...]


def rmsnorm(sess, ct, gamma, *, eps: float = DEFAULT_EPS, d: int, interval=DEFAULT_INTERVAL,
            degree: int = DEFAULT_DEGREE) -> RMSNormResult:
    """``x * rsqrt(mean(x^2) + eps) * gamma`` on a token ciphertext, from the primitives.

    sess: a `perseus.Session` (anything with ``.inf`` and ``.options``); ct: a fresh
    token-basis ciphertext (`Session.encrypt`) holding ``d`` real features; gamma: length
    ``d``; interval: the ``[a, b]`` window of ``mean(x^2) + eps`` the rsqrt series is fitted
    on (a calibration window -- the mirror refuses inputs outside it, the FHE side cannot);
    degree: the series degree (``rmsnorm_depth(degree)`` levels; see `level_budget`).

    Returns an `RMSNormResult` with ``ct.level`` before and after; the level report is
    logged at INFO. A warning is logged when the degree does not fit the level budget (the
    runtime then bootstraps mid-chain and the level report stops being a pure count).
    """
    inf = sess.inf
    fhe = inf.fhe
    d = int(d)
    degree = int(degree)
    dim, hid = int(inf.size.dim), int(inf.size.hidDim)
    slots = int(inf.slots)
    if not 0 < d <= dim:
        raise ValueError(f"d must be in 1..{dim} (the session's real width), got {d}")
    if hid <= 0 or slots % hid != 0:
        raise ValueError(f"{slots} slots are not a multiple of hidDim {hid}: not the cachemir "
                         "token layout this example relies on")
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps}")
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")
    interval = _check_interval(interval)
    g = np.asarray(gamma, dtype=np.float64)
    if g.ndim != 1 or g.shape[0] != d:
        raise ValueError(f"gamma must be 1-D of length d={d}, got shape {g.shape}")
    if not np.isfinite(g).all():
        raise ValueError("gamma contains NaN/inf")
    packing = str(ct.packing)
    if packing not in _SLOT_LAYOUT_PACKINGS:
        raise ValueError(f"ct is packed as {packing!r}; this example relies on the cachemir "
                         "token layout (lane i at slot i*t) and would mis-sum any other")
    t = slots // hid
    needed = {1 << k for k in range(int(math.log2(slots)))}
    loaded = {int(s) for s in (getattr(fhe, "loaded_rot_steps", None) or ())}
    if loaded:
        missing = sorted(needed - loaded)
        if missing:
            raise ValueError(f"sum_slots over {slots} slots needs rotation keys "
                             f"{sorted(needed)}; not loaded: {missing}")
    depth = rmsnorm_depth(degree)
    budget = level_budget(sess, ct)
    if depth > budget:
        log.warning("RMSNorm degree %d needs %d levels but only %d remain before the reactive "
                    "bootstrap (level %d, limit %d): the runtime will bootstrap mid-chain and "
                    "the level report is not a pure count",
                    degree, depth, budget, int(ct.level), int(fhe.level_limit()))

    before = int(ct.level)
    sq = fhe.square(ct)                                    # x^2 on the lanes, 0 elsewhere
    total = inf.sum_slots(sq, slots)                       # sum(x^2) broadcast to every slot
    coeffs = rsqrt_cheb_coeffs(interval, degree)
    a2, b2 = fhe_interval(interval, eps, d)
    r = inf.eval_chebyshev(total, [float(c) for c in coeffs], float(a2), float(b2))
    xg = inf.mult_pt(ct, gamma_to_slots(g, slots, t))      # x * gamma, lane layout
    out = fhe.mult(xg, r)
    after = int(out.level)
    unit = _level_unit(sess)
    log.info("rmsnorm: degree %d, levels %d -> %d (consumed %d; formula %d levels x %d primes)",
             degree, before, after, after - before, depth, unit)
    return RMSNormResult(out, before, after, degree, coeffs)
