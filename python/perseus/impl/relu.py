"""Composite-sign ReLU (calibrate/approximations/relu.py): ReLU(x) = x * p_k(... p_1(x / B)), each p_i an odd
Chebyshev series on [-h_i, h_i], the last one carrying (1 + sgn) / 2."""
from __future__ import annotations

from .poly import eval_chebyshev


def relu_composite(ops, x, B, stages):
    """`stages` = [(cheb coeffs, h)] from the calibration; stage 1 reads x on [-B h_1, B h_1]."""
    s = x
    for i, (coef, h) in enumerate(stages):
        r = B * h if i == 0 else h
        s = eval_chebyshev(ops, s, coef, -r, r)
    return ops.mult(x, s)


def unit_stages(stages):
    """The stages with every affine folded away: stage i's coefficients carry 1 / h_{i+1}, so each series reads its
    input on [-1, 1] (no level for the map) and stage 1 reads x / B (the 1 / B rides the producing conv)."""
    out = []
    for i, (coef, _h) in enumerate(stages):
        nxt = stages[i + 1][1] if i + 1 < len(stages) else 1.0
        out.append([float(c) / nxt for c in coef])
    return out


def relu_unit(ops, x, coefs, gain=1.0):
    """gain * ReLU(B x) / B of x = (input / B): x * p_k(... p_1(x)), each series on [-1, 1] (`unit_stages`), the gain
    in the last one's coefficients."""
    s = x
    for i, c in enumerate(coefs):
        if i == len(coefs) - 1 and gain != 1.0:
            c = [gain * v for v in c]
        s = eval_chebyshev(ops, s, c, -1.0, 1.0)
    return ops.mult(x, s)


def relu_config(section):
    """The configs.json "relu" section -> (stages, {site: B})."""
    stages = [(st["cheb"], float(st["h"])) for st in section["stages"]]
    return stages, {n: float(v["B"]) for n, v in section["sites"].items()}
