#!/usr/bin/env python3
"""EvalMod Chebyshev refit generator.

The ENCAPS EvalMod polynomial approximates
    f_r(y) = (2*pi)**(-1/2**r) * cos(2*pi*(K*y - 0.25) / 2**r),   y in [-1, 1], K = 16
(verified against the shipped degree-32 / r=3 series to 7e-12). Because each double-angle
iteration is an EXACT identity, raising r shrinks the fitted range and the required degree
at zero approximation cost. The shipped default is r=5 / degree 14 (fit err 2.6e-10).

Output convention matches the FIDESlib evaluator: f = c0/2 + sum_{i>=1} c_i T_i
(i.e. c0 is stored DOUBLED). Feed the file to FIDESLIB_CHEB_COEFFS_FILE with
FIDESLIB_DA_ITS=<r>; keep both consistent or the level plan shifts (illegal access in the
CtS LT dot).

Usage: python3 fit_evalmod_cheby.py <r> <degree> [outfile]
Pure python on purpose: no numpy dependency.
"""
import math
import sys

K = 16.0


def target(y: float, r: int) -> float:
    return (2 * math.pi) ** (-1 / 2**r) * math.cos(2 * math.pi * (K * y - 0.25) / 2**r)


def cheb_fit(f, d: int):
    """Chebyshev interpolation at Gauss nodes; returns c with c0 doubled."""
    n = d + 1
    xs = [math.cos(math.pi * (k + 0.5) / n) for k in range(n)]
    fs = [f(x) for x in xs]
    return [sum(fs[k] * math.cos(math.pi * j * (k + 0.5) / n) for k in range(n)) * 2.0 / n
            for j in range(n)]


def cheb_eval(c, x: float) -> float:
    b1 = b2 = 0.0
    for ci in reversed(c[1:]):
        b1, b2 = 2 * x * b1 - b2 + ci, b1
    return x * b1 - b2 + c[0] / 2


def main():
    r, d = int(sys.argv[1]), int(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else f"cheb_r{r}_d{d}.txt"
    c = cheb_fit(lambda y: target(y, r), d)
    err = max(abs(cheb_eval(c, -1 + 2 * i / 4000) - target(-1 + 2 * i / 4000, r))
              for i in range(4001))
    with open(out, "w") as fh:
        fh.write("\n".join("%.17e" % x for x in c))
    # Measured error amplification fit->bootstrap output: ~7.7e3 (r=4), ~1.5e4 (r=5).
    # Keep err * amplification below the chain's ~1e-4 noise floor.
    print(f"r={r} d={d}: fit err {err:.3e} ({-math.log2(err):.1f} bits) -> {out}")


if __name__ == "__main__":
    main()
