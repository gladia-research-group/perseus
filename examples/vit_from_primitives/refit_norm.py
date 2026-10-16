"""LN_CHEB seeds for a ViT approximation config: scripts/utils/refit_norm_cheb.py's fit on the LayerNorm inputs of an
image pool (the calibrator's preprocessed pool, or images on disk).

    .venv/bin/python -m examples.vit_from_primitives.refit_norm CONFIG.json [--pool POOL.npy] [--images DIR]
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os

import numpy as np


def _fit_module():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "utils", "refit_norm_cheb.py")
    spec = importlib.util.spec_from_file_location("refit_norm_cheb", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gpt2_names(j):
    """The calibrator's site names (vit.layers.N.*) as the GPT-2 names Configs.block reads."""
    import re
    rules = [(r"^vit\.layers\.(\d+)\.layernorm_before$", r"transformer.h.\1.ln_1"),
             (r"^vit\.layers\.(\d+)\.layernorm_after$", r"transformer.h.\1.ln_2"),
             (r"^vit\.layers\.(\d+)\.attention$", r"transformer.h.\1.attn"),
             (r"^vit\.layers\.(\d+)\.mlp\.activation_fn$", r"transformer.h.\1.mlp.act")]
    for sec in ("norm", "softmax", "softgelu"):
        out = {}
        for k, v in j.get(sec, {}).items():
            for pat, rep in rules:
                k = re.sub(pat, rep, k)
            out[k] = v
        j[sec] = out
    return j


def hf_name(site):
    p = site.split(".")
    if site == "transformer.ln_f":
        return "vit.layernorm"
    return f"vit.layers.{p[2]}." + ("layernorm_before" if p[3] == "ln_1" else "layernorm_after")


def collect(cfgs, batches):
    import torch
    from transformers import ViTForImageClassification
    from perseus.impl.config import parse_norm
    from .weights import MODEL
    model = ViTForImageClassification.from_pretrained(MODEL, attn_implementation="eager").eval().double()
    out = {}
    mods = dict(model.named_modules())
    for site in cfgs:
        c = parse_norm(cfgs[site])
        c2 = torch.tensor([c.c_eff_sq(p) for p in range(197)], dtype=torch.float64)

        def hook(mod, inp, _, site=site, c2=c2):
            x = inp[0].double()
            cen = x - x.mean(-1, keepdim=True)
            z = c2 * (cen * cen).mean(-1) + float(cfgs[site]["eps"]) * c2
            out.setdefault(site, []).append(z.reshape(-1).numpy())
        mods[hf_name(site)].register_forward_hook(hook)
    with torch.no_grad():
        for b in batches:
            model(torch.tensor(b, dtype=torch.float64))
    return {k: np.concatenate(v) for k, v in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--pool", default=None, help="(n, 3, 224, 224) preprocessed images")
    ap.add_argument("--images", default=None)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--holdout", type=int, default=64)
    ap.add_argument("--safety", type=float, default=2.0)
    ap.add_argument("--target-err", type=float, default=1e-5)
    ap.add_argument("--max-deg", type=int, default=31)
    ap.add_argument("--flat", action="store_true",
                    help="drop the per-position centering scales (high-norm artifact tokens land on any position)")
    a = ap.parse_args(argv)
    from perseus.impl.config import parse_norm
    from perseus.impl.poly import NumpyOps, eval_chebyshev, inv_sqrt_newton_d2
    from . import weights as V
    fit = _fit_module().fit_site
    if a.pool:
        pool = np.load(a.pool, mmap_mode="r")
    else:
        pool = np.stack([V.pixels(f).transpose(2, 0, 1) for f in sorted(glob.glob(os.path.join(a.images, "*")))])
    n = min(a.n, len(pool) - (a.holdout if len(pool) > a.holdout else 0))
    batches = [np.asarray(pool[i:i + 8]) for i in range(0, n, 8)]
    hold = [np.asarray(pool[i:i + 8]) for i in range(n, min(len(pool), n + a.holdout), 8)]
    j = gpt2_names(json.load(open(a.config)))
    cfgs = j["norm"]
    if a.flat:
        for v in cfgs.values():
            v.pop("center_scale_sq", None)
    z = collect(cfgs, batches)
    zh = collect(cfgs, hold) if hold else z
    ops = NumpyOps()
    for site in sorted(cfgs):
        cfg = parse_norm(cfgs[site])
        lo, hi = float(z[site].min()) / a.safety, float(z[site].max()) * a.safety
        (d, _), deg, nr, c, err = fit(lo, hi, cfg.inv_out_scale, a.target_err, a.max_deg)
        h = zh[site]
        y = inv_sqrt_newton_d2(ops, h, eval_chebyshev(ops, h, c, lo, hi), nr, 1.0 / cfg.inv_out_scale ** 2)
        herr = float(np.max(np.abs(y * np.sqrt(h) / cfg.inv_out_scale - 1)))
        oob = int(((h < lo) | (h > hi)).sum())
        print(f"{site:24s} data [{z[site].min():.3g}, {z[site].max():.3g}] deg {deg:2d} + NR{nr} depth {d:2d} "
              f"err {err:.1e} | holdout err {herr:.1e} out-of-band {oob}/{h.size}", flush=True)
        cfgs[site].update(cheb_coeffs=[float(v) for v in c], cheb_lo=lo, cheb_hi=hi, cheb_nr_iters=nr)
    with open(a.config, "w") as fh:
        json.dump(j, fh, indent=1)
    print(f"[write] {a.config}")


if __name__ == "__main__":
    main()
