#!/usr/bin/env python3
"""EvalMod ZOO — generate Chebyshev series for a grid of (r, degree) shapes.

    python3 scripts/utils/gen_evalmod_coeffs.py --validate         # reproduce the shipped series
    python3 scripts/utils/gen_evalmod_coeffs.py --zoo out_dir/     # emit the grid + a table

WHY. The EvalMod approximates x mod q0 as a Chebyshev series followed by `r` double-angle
squarings. Each squaring is an EXACT identity (cos 2t = 2cos^2 t - 1), so raising r shrinks the
fitted range and therefore the degree needed — at zero approximation cost. Applying that
once (r=3/deg32 -> r=5/deg14) is worth -2.96 ms and is depth-neutral; the shipped default
stops at the precision-neutral point.

Sparse routing changes the calculus: it hands back 3-5 bits,
the fused reduction hands back more, and the chain only needs ~10
bits at the bootstrap. That surplus is spendable, and it buys two things at once:

    latency  - a smaller Paterson-Stockmeyer tree is fewer multiplications
    LEVELS   - depth = ceil(log2(deg+1)) + r, so a cheaper shape frees compute levels
               for the model, which is worth more than milliseconds

TARGET FUNCTION (RawCiphertext.cu:530-542, verified against the shipped series to 7e-12):

    f_r(y) = (2*pi)^(-1/2^r) * cos(2*pi*(K*y - 0.25) / 2^r),   y in [-1, 1],  K = bootK = 16

r is baked into the coefficients while the CtS masks carry only 1/K, so a series and its r MUST
be shipped together — FIDESLIB_CHEB_COEFFS_FILE and FIDESLIB_DA_ITS are a pair, never one alone.

 These knobs are CONTEXT-GLOBAL (param.raw), so a zoo entry applies to every bootstrap in the
run. Using a cheap shape on the sparse lanes and an accurate one elsewhere needs per-precomp
EvalMod parameters in FIDESlib — the same gap that already blocks per-site arcsine and per-site
(r, degree). This zoo is the measurement that says whether that plumbing is worth building.
"""
import argparse
import math
import os

import numpy as np

K = 16.0

# The shipped r=5 / degree-14 series (RawCiphertext.cu), evaluator convention (c0 doubled).
SHIPPED_R5_D14 = [
    -5.73829476916553172e-01, 2.63718536771466207e-02, -9.15574236933522245e-01,
    -3.08975417541565676e-02, 2.85601052580403802e-01, 4.83129148738224799e-03,
    -2.74350673185783482e-02, -3.16919293037672828e-04, 1.31295181914509542e-03,
    1.15825536926191591e-05, -3.79010152666429525e-05, -2.71035793019274615e-07,
    7.33952552563662122e-07, 4.41686895092293209e-09, -1.03169742989480305e-08,
]


def target(y, r):
    """f_r(y) — the function the Chebyshev series must approximate."""
    return (2.0 * math.pi) ** (-1.0 / 2 ** r) * np.cos(2.0 * math.pi * (K * y - 0.25) / 2 ** r)


def fit(r, degree):
    """Chebyshev coefficients of f_r, in the EVALUATOR convention (c0 doubled).

    Fitted at Chebyshev nodes of the second kind via numpy's Chebyshev basis, which is the same
    transform §1.16 used. Returns (coeffs, max |error| on a dense grid).
    """
    n = degree + 1
    nodes = np.cos(np.pi * (np.arange(n) + 0.5) / n)          # Chebyshev-Gauss nodes
    vals = target(nodes, r)
    c = np.polynomial.chebyshev.chebfit(nodes, vals, degree)

    grid = np.linspace(-1.0, 1.0, 20001)
    err = float(np.max(np.abs(np.polynomial.chebyshev.chebval(grid, c) - target(grid, r))))

    out = c.copy()
    out[0] *= 2.0                                              # evaluator convention
    return out, err


def depth(degree, r):
    """Multiplicative depth: the PS tree over deg+1 coefficients, plus one per double angle.

    Calibrated against §1.16: depth(15 coeffs)=4, +5 DA = 9; depth(33 coeffs)=6, +3 = 9. Both
    shapes cost 9, which is why that refit was depth-neutral.
    """
    return int(math.ceil(math.log2(degree + 1))) + r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="refit r=5/deg14 and diff against the shipped series")
    ap.add_argument("--zoo", metavar="OUTDIR", help="emit the (r, degree) grid")
    ap.add_argument("--r", type=int, nargs="*", default=[4, 5, 6, 7])
    ap.add_argument("--deg", type=int, nargs="*", default=[4, 6, 8, 10, 12, 14, 16, 20])
    args = ap.parse_args()

    if args.validate:
        c, err = fit(5, 14)
        d = np.abs(np.array(c) - np.array(SHIPPED_R5_D14))
        rel = d / np.maximum(np.abs(SHIPPED_R5_D14), 1e-30)
        print(f"refit r=5 deg=14: fit_err={err:.3e}")
        print(f"  vs shipped: max_abs_diff={d.max():.3e}  max_rel_diff={rel.max():.3e}")
        print("  " + ("MATCH — the generator reproduces the shipped series"
                      if d.max() < 1e-9 else
                      "*** MISMATCH — do NOT trust the zoo until this is resolved ***"))
        return

    if not args.zoo:
        ap.error("pass --validate or --zoo OUTDIR")
    os.makedirs(args.zoo, exist_ok=True)

    # Measured fit->output error amplification: ~7.7e3 at r=4, ~1.5e4 at r=5. The
    # trend is ~2x per r (each double angle squares, so it doubles the error exponent's reach).
    amp = {4: 7.7e3, 5: 1.5e4, 6: 3.0e4, 7: 6.0e4}
    base_depth = depth(14, 5)   # the shipped shape

    rows = []
    for r in args.r:
        for dg in args.deg:
            c, err = fit(r, dg)
            dep = depth(dg, r)
            a = amp.get(r, 1.5e4)
            pred_bits = -math.log2(err * a) if err * a > 0 else 99.0
            name = f"cheb_r{r}_d{dg}.txt"
            with open(os.path.join(args.zoo, name), "w") as fh:
                fh.write("\n".join(f"{v:.17e}" for v in c) + "\n")
            rows.append((r, dg, dep, dep - base_depth, err, pred_bits, name))

    rows.sort(key=lambda t: (t[2], -t[5]))
    print(f"{'r':>2} {'deg':>4} {'depth':>6} {'vs_ship':>8} {'fit_err':>11} "
          f"{'pred_bits':>10}  file")
    print("-" * 78)
    for r, dg, dep, dd, err, pb, name in rows:
        mark = ""
        if pb < 10.0:
            mark = "  (under the 10-bit bar)"
        elif dd < 0:
            mark = "  <-- CHEAPER DEPTH, still >=10 bits"
        print(f"{r:>2} {dg:>4} {dep:>6} {dd:>+8} {err:>11.2e} {pb:>10.1f}  {name}{mark}")
    print("\npred_bits uses the measured fit->output amplification (7.7e3 at r=4, 1.5e4 at"
          "\nr=5; r=6 and r=7 are extrapolated and unverified). Treat it as a screen rather"
          "\nthan a measurement.")


if __name__ == "__main__":
    main()
