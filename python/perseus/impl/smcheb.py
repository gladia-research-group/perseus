"""SM_CHEB_GS: Chebyshev-scaled Goldschmidt for the softmax reciprocal rounds (SM_DEN_RECIP=1).

Round 0 keeps the calibrated seed; refine round j >= 1 runs on the exact band of its denominator and uses the step
d <- c_j d (2 - d), c_j = 2 t_j^2 / (2 t_j^2 - 1), t_j = T_{2^(j+1)}(z0), leaving residual 1 / T_{2^k}(z0) after k
levels (plain Goldschmidt: (1 / T_2(z0))^(2^(k-1))) at no extra level or product. Per-round counts (recip_iters_cheb)
and residual targets (recip_targets) are per-layer config keys from design(); seeds and step constants are per-kc
host constants. Derivation: docs/ARCHITECTURE.md.
"""
from __future__ import annotations

import functools
import itertools
import math

TOP = 1.05
GRID = (0.01, 0.03, 0.1, 0.2, 0.3, 0.5)
KC_DESIGN = (1, 2, 4, 8, 16, 32, 64, 96, 128, 160)
TGT_LAST = 1e-3


def cheb_T(n: int, z: float) -> float:
    return math.cosh(n * math.acosh(z)) if z >= 1 else math.cos(n * math.acos(z))


def cheb_seed(lo: float, hi: float):
    """The minimax linear seed F(x) = A - B x of 1/x on [lo, hi]."""
    s = lo + hi
    b = 8.0 / (s * s + 4.0 * lo * hi)
    return b * s, b


def band_from_seed(a: float, b: float):
    """The band [lo, hi] a linear seed a - b x is the minimax seed of."""
    s = a / b
    p = (8.0 / b - s * s) / 4.0
    disc = s * s - 4.0 * p
    if disc <= 0 or p <= 0:
        raise ValueError(f"softmax seed ({a}, {b}) is not a minimax seed of any band")
    r = math.sqrt(disc)
    return (s - r) / 2.0, (s + r) / 2.0


def refine_band(kc: int, e: float, top: float = TOP):
    return 0.5 * (1.0 - e) ** 2 / math.sqrt(kc), 0.5 * (1.0 + e) ** 2 * math.sqrt(kc) * top


def z0_of(lo: float, hi: float) -> float:
    return (hi + lo) / (hi - lo)


def resid(lo: float, hi: float, k: int) -> float:
    return 1.0 / cheb_T(2 ** k, z0_of(lo, hi))


def k_for(lo: float, hi: float, tgt: float) -> int:
    for k in range(1, 40):
        if resid(lo, hi, k) <= tgt:
            return k
    raise ValueError(f"band [{lo}, {hi}] needs more than 40 levels for {tgt}")


def design(init_alpha: float, init_beta: float, n_refine: int, tgt_last: float = TGT_LAST, kcs=KC_DESIGN):
    """(counts per round, residual targets per round): the mid-round targets on GRID minimizing the total count, the
    last round at tgt_last; refine counts are the max over kcs."""
    lo0, hi0 = band_from_seed(init_alpha, init_beta)
    best = None
    for mids in itertools.product(GRID, repeat=n_refine):
        tg = list(mids) + [tgt_last]
        ks = [k_for(lo0, hi0, tg[0])]
        for j in range(1, n_refine + 1):
            ks.append(max(k_for(*refine_band(kc, tg[j - 1]), tg[j]) for kc in kcs))
        if best is None or sum(ks) < sum(best[0]):
            best = (ks, tg)
    return best


@functools.lru_cache(maxsize=4096)
def step_consts(z0: float, k: int):
    """(kappa_0..kappa_{k-2}, kappa_0, 1/lambda_K) of the packed Chebyshev-scaled recursion with k levels."""
    t = cheb_T(2, z0)
    cs = []
    for _ in range(k - 1):
        cs.append(2.0 * t * t / (2.0 * t * t - 1.0))
        t = 2.0 * t * t - 1.0
    k0 = math.prod(c ** (2.0 ** -(i + 1)) for i, c in enumerate(cs))
    kap, lam = [k0], [k0]
    for c in cs:
        kap.append(kap[-1] ** 2 / c)
        lam.append(lam[-1] * kap[-2] / c)
    # lambda_K is 1 by construction (kappa_0 = prod c_i^(2^-(i+1))): mu = 1 exactly, not 1 +- a few ulps that differ
    # between cache lengths (a planned decode takes a scalar its probe tokens kept equal for a constant)
    return tuple(kap[:k - 1]), k0, 1.0


def rounds(cfg, kc: int):
    """Per reciprocal round: (alpha, beta, iters, scale, (kappas, mu)) for poly.goldschmidt_recip, alpha and beta
    already scaled by kappa_0 (the seed mask value 0.5 beta scale^2 follows)."""
    if not cfg.recip_iters_cheb or len(cfg.recip_iters_cheb) != 1 + cfg.log2delta2:
        raise ValueError("SM_CHEB_GS=1 needs the softmax config keys recip_iters_cheb / recip_targets "
                         "(scripts/utils/sm_cheb_design.py)")
    out = []
    lo, hi = band_from_seed(cfg.init_alpha, cfg.init_beta)
    a, b, scale = cfg.init_alpha, cfg.init_beta, 1.0
    for j, k in enumerate(cfg.recip_iters_cheb):
        if j:
            lo, hi = refine_band(kc, float(cfg.recip_targets[j - 1]))
            a, b = cheb_seed(lo, hi)
            scale = 0.5 * math.sqrt(kc) * 0.25
        kap, k0, mu = step_consts(z0_of(lo, hi), int(k))
        out.append((a * k0, b * k0, int(k), scale, (kap, mu)))
    return out
