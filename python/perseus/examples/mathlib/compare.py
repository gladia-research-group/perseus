"""impl.math against the ad-hoc approximations in the model code, on the same domain and at the same error (host
side: float64 error on the domain, levels and products counted by impl.math.CountOps on the code each one runs).

    python -m perseus.examples.mathlib.compare [--json out.jsonl]

Rows: the GPT-2 LayerNorm inverse square root seeds (LN_CHEB, every ln site of gpt2_base_n32_cm), the THOR softmax
exp (Chebyshev + squarings, every layer), the softmax's first Goldschmidt division (init seed, gs_iters_scaled), the
sampler head's exp (tfhe_select._exp_ckks), the ResNet-20 composite ReLU (degrees 7, 7, 27) and the selection's CKKS
sign (tfhe_select._composite: Cheon et al. g3 / f3 iterations). A row's `ours` is impl.math's choice for the ad-hoc
version's domain and its measured max error (relative where the function's range is wide)."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from perseus.impl import math as M
from perseus.impl import poly
from perseus.impl.math import CountOps, _NpOps

RUNS: dict = {}       # site -> the ad-hoc code (ops, x) -> y (examples/mathlib/tasks.py AdHoc)
REFS: dict = {}       # site -> its reference function

CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs", "model", "approximation")


def _cost(fn):
    c = CountOps()
    fn(c, c.leaf())
    return c.depth, c.ct_mults


def _row(name, dom, adhoc_err, adhoc_cost, ours, rel, run=None, ref=None):
    RUNS[name], REFS[name] = run, ref
    return {"site": name, "domain": [float(dom[0]), float(dom[1])], "rel": rel, "adhoc_err": adhoc_err,
            "adhoc_depth": adhoc_cost[0], "adhoc_ct": adhoc_cost[1], "ours_method": ours.method,
            "ours_params": dict(ours.params), "ours_err": ours.err, "ours_depth": ours.depth, "ours_ct": ours.ct_mults,
            "no_worse": (ours.depth, ours.ct_mults) <= tuple(adhoc_cost)}


def ln_rows(cfgname="gpt2_base_n32_cm"):
    c = json.load(open(os.path.join(CFG, cfgname, "configs.json")))
    out = []
    for site, s in sorted(c["norm"].items()):
        if not s.get("cheb_coeffs"):
            continue
        lo, hi, k = float(s["cheb_lo"]), float(s["cheb_hi"]), float(s["inv_out_scale"])
        cf = [float(v) for v in s["cheb_coeffs"]]
        nr = int(s.get("cheb_nr_iters", 0) or 0)

        def run(ops, x, cf=cf, lo=lo, hi=hi, nr=nr, k=k):
            y = poly.eval_chebyshev(ops, x, cf, lo, hi)
            return poly.inv_sqrt_newton_d2(ops, x, y, nr, 1.0 / (k * k)) if nr else y
        x = M.grid(lo, hi)
        e = float(np.max(np.abs(np.real(run(_NpOps(), x.copy())) / (k / np.sqrt(x)) - 1.0)))
        ours = M.approx_rsqrt(lo, hi, e, True)
        out.append(_row(f"LN rsqrt {site}", (lo, hi), e, _cost(run), ours, True, run, lambda x, k=k: k / np.sqrt(x)))
    return out


def softmax_rows(cfgname="gpt2_base_n32_cm"):
    c = json.load(open(os.path.join(CFG, cfgname, "configs.json")))
    out = []
    for site, s in sorted(c["softmax"].items()):
        cf = [float(v) for v in s["cheb_coeffs"]]
        a, b, sq = float(s["cheb_a"]), float(s["cheb_b"]), int(round(np.log2(float(s["delta1"]))))
        k = 2 ** sq
        lo, hi = k * a, k * b                  # u = 2^sq t: the series reads t = u / 2^sq through its affine

        def run(ops, u, cf=cf, a=a, b=b, sq=sq, k=k):
            z = poly.eval_chebyshev(ops, u, cf, k * a, k * b)
            for _ in range(sq):
                z = ops.square(z)
            return z
        x = M.grid(lo, hi)
        y = np.real(run(_NpOps(), x.copy()))
        e = float(np.max(np.abs(y / np.exp(x) - 1.0)))
        ours = M.approx_exp(lo, hi, e, True)
        out.append(_row(f"softmax exp {site}", (lo, hi), e, _cost(run), ours, True, run, np.exp))
        # the first division: Goldschmidt from the config's seed on its band (chebyshev_gs_init inverted)
        al, be, it = float(s["init_alpha"]), float(s["init_beta"]), int(s["gs_iters_scaled"])
        sm = al / be
        p = (8.0 / be - sm * sm) / 4.0
        dlo, dhi = (sm - np.sqrt(sm * sm - 4 * p)) / 2, (sm + np.sqrt(sm * sm - 4 * p)) / 2

        def runr(ops, x, al=al, be=be, it=it):
            return poly.goldschmidt_recip(ops, x, al, be, it)
        x = M.grid(dlo, dhi)
        e = float(np.max(np.abs(np.real(runr(_NpOps(), x.copy())) * x - 1.0)))
        ours = M.approx_reciprocal(float(dlo), float(dhi), max(e, 1e-12), True)
        out.append(_row(f"softmax recip {site}", (dlo, dhi), e, _cost(runr), ours, True, runr, lambda x: 1.0 / x))
    return out


def sampler_exp_rows():
    from perseus.impl.tfhe_select import NAT_PAD, _exp_ckks
    out = []
    for W in (8.0, 32.0):
        lo, hi = -W, NAT_PAD

        def run(ops, x, lo=lo, hi=hi):
            return _exp_ckks(ops, x, lo, hi)
        x = M.grid(lo, hi)
        e = float(np.max(np.abs(np.real(run(_NpOps(), x.copy())) / np.exp(x) - 1.0)))
        out.append(_row(f"sampler exp W={W:g}", (lo, hi), e, _cost(run), M.approx_exp(lo, hi, e, True), True))
    return out


def relu_rows():
    from perseus.impl.relu import relu_folded, unit_stages
    from perseus.calibrate.approximations.relu import fit_composite_sign
    out = []
    for degs, eps in (((7, 7, 27), 2.0 ** -7), ((15, 15, 27), 2.0 ** -10)):
        coefs = unit_stages(fit_composite_sign(list(degs), eps))

        def run(ops, x, coefs=coefs):
            return relu_folded(ops, x, coefs)
        x = M.grid(-1.0, 1.0)
        e = float(np.max(np.abs(np.real(run(_NpOps(), x.copy())) - np.maximum(x, 0.0))))
        lad, dep, cm, oe = M.approx_relu(1.0, e)
        r = {"site": f"ReLU composite {degs} gap {eps:g}", "domain": [-1.0, 1.0], "rel": False, "adhoc_err": e,
             "adhoc_depth": _cost(run)[0], "adhoc_ct": _cost(run)[1], "ours_method": f"composite {lad.degs}",
             "ours_params": {"eps": lad.eps}, "ours_err": oe, "ours_depth": dep, "ours_ct": cm}
        r["no_worse"] = (dep, cm) <= (r["adhoc_depth"], r["adhoc_ct"])
        RUNS[r["site"]], REFS[r["site"]] = run, (lambda x: np.maximum(x, 0.0))
        out.append(r)
    return out


def sign_rows():
    from perseus.impl.tfhe_select import _F3, _G3, _composite
    from perseus.impl.relu import odd_series
    out = []
    cheb = np.polynomial.chebyshev.poly2cheb
    for eps in (2.0 ** -6, 2.0 ** -8, 2.0 ** -10):
        a, b = _composite(eps)
        cs = [list(cheb(c)) for c in [_G3] * a + [_F3] * b]

        def run(ops, x, cs=cs):
            for c in cs:
                x = poly.eval_chebyshev(ops, x, c, -1.0, 1.0)
            return x
        xs = np.concatenate([np.geomspace(eps, 1.0, 3000), np.linspace(eps, 1.0, 3000)])
        e = float(np.max(np.abs(np.real(run(_NpOps(), xs.copy())) - 1.0)))
        lad = M.sign_ladder(eps, max(e, 1e-12))

        def ours(ops, x, lad=lad):
            for c in lad.coefs:
                x = odd_series(ops, x, list(c))
            return x
        dep, cm = _cost(ours)
        r = {"site": f"CKKS sign g3^{a} f3^{b} eps {eps:g}", "domain": [eps, 1.0], "rel": False, "adhoc_err": e,
             "adhoc_depth": _cost(run)[0], "adhoc_ct": _cost(run)[1], "ours_method": f"composite {lad.degs}",
             "ours_params": {}, "ours_err": lad.delta, "ours_depth": dep, "ours_ct": cm}
        r["no_worse"] = (dep, cm) <= (r["adhoc_depth"], r["adhoc_ct"])
        out.append(r)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    rows = ln_rows() + softmax_rows() + sampler_exp_rows() + relu_rows() + sign_rows()
    for r in rows:
        print(f"{r['site']:44s} [{r['domain'][0]:.4g}, {r['domain'][1]:.4g}] err {r['adhoc_err']:.2e}: "
              f"ad-hoc depth {r['adhoc_depth']:2d} ct {r['adhoc_ct']:3d} | ours {r['ours_method']:14s} "
              f"depth {r['ours_depth']:2d} ct {r['ours_ct']:3d} err {r['ours_err']:.2e} {'' if r['no_worse'] else 'WORSE'}")
    if a.json:
        with open(a.json, "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
