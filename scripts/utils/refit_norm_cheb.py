#!/usr/bin/env python
"""Chebyshev seed for the LayerNorm inverse sqrt on each site's observed variance band.

The band is the min/max of c_eff_sq(pos)*var + eps*c2 over the token pool, widened by --safety
on each side; the seed is the shallowest Chebyshev degree + Newton count meeting --target-err.
Written as the cheb_* keys of each norm site (read under LN_CHEB; the Remez fields stay).

  PYTHONPATH=$REPO python scripts/utils/refit_norm_cheb.py --pool $POOL \
      --ref-config configs/model/approximation/gpt2_base_n32/configs.json \
      --out-config configs/model/approximation/gpt2_base_n32/configs.json
"""
import argparse
import json
import math

import numpy as np

from perseus.impl.config import parse_norm
from perseus.impl.poly import NumpyOps, eval_chebyshev, inv_sqrt_newton_d2


def collect(model_name, checkpoint, pool, cfgs, n_win, window, offset):
    """{site: z per token} over n_win windows spread across the pool."""
    import torch

    from perseus.hub import load_model

    model = load_model(model_name, device="cpu")
    if checkpoint:
        from perseus.export import load_trained_backbone
        load_trained_backbone(model, checkpoint)
    model.eval()
    out = {}
    c2 = {n: torch.tensor([parse_norm(cfgs[n]).c_eff_sq(p) for p in range(window)],
                          dtype=torch.float64) for n in cfgs}

    def hook(name):
        def f(mod, inp, _):
            x = inp[0][0].double()
            c = x - x.mean(-1, keepdim=True)
            z = c2[name] * (c * c).mean(-1) + float(cfgs[name]["eps"]) * c2[name]
            out.setdefault(name, []).append(z.numpy())
        return f

    for n, m in model.named_modules():
        if n in cfgs:
            m.register_forward_hook(hook(n))
    stride = max(window, (len(pool) - offset - window) // max(n_win, 1))
    with torch.no_grad():
        for w in range(n_win):
            off = offset + w * stride
            toks = np.asarray(pool[off:off + window], dtype=np.int64)
            if len(toks) < window:
                break
            model(torch.tensor(toks).unsqueeze(0))
    return {n: np.concatenate(v) for n, v in out.items()}


def depth(deg, nr):
    """eval_chebyshev (affine, T_deg, coefficient) + two levels per Newton iteration."""
    return 2 + math.ceil(math.log2(deg)) + 2 * nr if deg > 1 else 2 + 2 * nr


def fit_site(lo, hi, scale, target, max_deg):
    """The shallowest (deg, nr) whose seed + Newton meets `target` relative error on [lo, hi]."""
    ops = NumpyOps()
    t = np.cos(np.pi * (np.arange(400) + 0.5) / 400)
    xs = (t + 1) / 2 * (hi - lo) + lo
    f = scale / np.sqrt(xs)
    grid = np.geomspace(lo, hi, 4001)
    exact = scale / np.sqrt(grid)
    best = None
    for deg in range(2, max_deg + 1):
        w = 1.0 / f
        c = np.polynomial.chebyshev.chebfit(t, f, deg, w=w)
        for _ in range(30):     # reweight toward minimax relative error
            r = np.abs(np.polynomial.chebyshev.chebval(t, c) / f - 1)
            c = np.polynomial.chebyshev.chebfit(t, f, deg, w=w * (r / r.max() + 0.05))
        for nr in range(0, 5):
            y = inv_sqrt_newton_d2(ops, grid, eval_chebyshev(ops, grid, c, lo, hi), nr,
                                   1.0 / (scale * scale))
            err = float(np.max(np.abs(y / exact - 1)))
            if err <= target:
                key = (depth(deg, nr), deg + 3 * nr)
                if best is None or key < best[0]:
                    best = (key, deg, nr, c, err)
                break
    if best is None:
        raise RuntimeError(f"no chain up to degree {max_deg} meets {target} on [{lo:.3g}, {hi:.3g}]")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="openai-community/gpt2")
    ap.add_argument("--checkpoint", default=None,
                    help="HE-aware-trained weights for --model (perseus-export --checkpoint)")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--ref-config", required=True)
    ap.add_argument("--out-config", required=True)
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--holdout", type=int, default=64, help="windows checked after the fit")
    ap.add_argument("--safety", type=float, default=2.0, help="band widening on each side")
    ap.add_argument("--target-err", type=float, default=1e-5)
    ap.add_argument("--max-deg", type=int, default=31)
    ap.add_argument("--dry-run", action="store_true", help="fit + validate, do not write")
    args = ap.parse_args()

    j = json.load(open(args.ref_config))
    cfgs = j["norm"]
    pool = np.load(args.pool, mmap_mode="r")
    z = collect(args.model, args.checkpoint, pool, cfgs, args.windows, args.window, 4096)
    zh = collect(args.model, args.checkpoint, pool, cfgs, args.holdout, args.window,
                 len(pool) // 2 + 4096)
    ops = NumpyOps()
    for n in sorted(cfgs):
        cfg = parse_norm(cfgs[n])
        lo, hi = float(z[n].min()) / args.safety, float(z[n].max()) * args.safety
        (d, _), deg, nr, c, err = fit_site(lo, hi, cfg.inv_out_scale, args.target_err, args.max_deg)
        h = zh[n]
        y = inv_sqrt_newton_d2(ops, h, eval_chebyshev(ops, h, c, lo, hi), nr,
                               1.0 / cfg.inv_out_scale ** 2)
        herr = float(np.max(np.abs(y * np.sqrt(h) / cfg.inv_out_scale - 1)))
        oob = int(((h < lo) | (h > hi)).sum())
        print(f"{n:24s} data [{z[n].min():.3g}, {z[n].max():.3g}] band ratio {hi / lo:6.1f}: "
              f"deg {deg:2d} + NR{nr} depth {d:2d} err {err:.1e} | holdout err {herr:.1e} "
              f"out-of-band {oob}/{h.size}")
        cfgs[n].update(cheb_coeffs=[float(v) for v in c], cheb_lo=lo, cheb_hi=hi, cheb_nr_iters=nr)
    if args.dry_run:
        return
    with open(args.out_config, "w") as fh:
        json.dump(j, fh, indent=1)
    print(f"[write] {args.out_config}")


if __name__ == "__main__":
    main()
