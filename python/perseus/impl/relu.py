"""Composite-sign ReLU (calibrate/approximations/relu.py): ReLU(x) = x * p_k(... p_1(x / B)), each p_i an odd
Chebyshev series on [-h_i, h_i], the last one carrying (1 + sgn) / 2."""
from __future__ import annotations

from .poly import _cheb_split, _cheb_trim, eval_chebyshev


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


def odd_series(ops, z, c, w=None, pw=None):
    """w sum_k c_k T_k(z) (w = None: 1) of an odd series c (c_0 = 0) of degree d < 2^L in L levels above z: split at
    k = 2^(L-1) as r + T_k q (both odd, degree < k) down to degree 3, a z + b z^3 = a z + (b z) z^2, so every constant
    rides the shallower factor of a product (no level of its own). w (the ReLU's x, several levels shallower than z)
    enters a degree-3 piece as (a w) z + ((b w) z) z^2 (two levels, three products), or as (w z) (a + b z^2) with w z
    shared (three levels, one product) where the piece's depth allows. `pw`: z's powers (z2: z^2, k: T_k)."""
    c = _cheb_trim(c)
    pw = {} if pw is None else pw

    def z2():
        if "z2" not in pw:
            pw["z2"] = ops.square(z)
        return pw["z2"]

    def T(k):
        if k not in pw:
            h = z2() if k == 2 else ops.square(T(k // 2))
            pw[k] = ops.add(ops.add(h, h), -1.0)
        return pw[k]

    def wz():
        if "wz" not in pw:
            pw["wz"] = ops.mult(w, z)
        return pw["wz"]

    def rec(c, depth):
        c = list(c) + [0.0] * (4 - len(c)) if len(c) < 4 else list(c)
        while len(c) > 4 and c[-1] == 0.0:
            c.pop()
        d = len(c) - 1
        if d <= 3:
            a, b = c[1] - 3.0 * c[3], 4.0 * c[3]
            if w is not None and depth >= 3 and (a != 0.0 or b != 0.0):
                if b == 0.0:
                    return ops.mult(wz(), a)
                u = ops.mult(z2(), b)
                return ops.mult(wz(), ops.add(u, a) if a != 0.0 else u)
            out = None
            if a != 0.0:
                out = ops.mult(z, a) if w is None else ops.mult(ops.mult(w, a), z)
            if b != 0.0:
                t = ops.mult(z, b) if w is None else ops.mult(ops.mult(w, b), z)
                t = ops.mult(t, z2())
                out = t if out is None else ops.add(out, t)
            return out
        k = 1 << (d.bit_length() - 1)
        q, r = _cheb_split(c, k)
        lo, hi = rec(r, depth), rec(q, depth - 1)
        if hi is None:
            return lo
        hi = ops.mult(T(k), hi)
        return hi if lo is None else ops.add(lo, hi)

    assert all(c[i] == 0.0 for i in range(0, len(c), 2)), "odd_series: an even coefficient"
    return rec(c, max(len(c) - 1, 1).bit_length())


def relu_folded(ops, x, coefs, gain=1.0):
    """`relu_unit` with every series by `odd_series` and the product by x folded into the last one: stage i of
    degree d_i costs ceil(log2(d_i + 1)) levels and the ReLU no level of its own (the last stage's (1 + s) / 2
    constant is c_0 x)."""
    s = x
    for c in coefs[:-1]:
        s = odd_series(ops, s, c)
    last = [gain * v for v in coefs[-1]]
    y = odd_series(ops, s, [0.0] + last[1:], w=x)
    return ops.add(y, ops.mult(x, last[0])) if last[0] != 0.0 else y


def relu_config(section):
    """The configs.json "relu" section -> (stages, {site: B})."""
    stages = [(st["cheb"], float(st["h"])) for st in section["stages"]]
    return stages, {n: float(v["B"]) for n, v in section["sites"].items()}
