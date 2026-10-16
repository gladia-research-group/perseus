"""The BERT-base SST-2 approximation config at the runtime row length T (bidirectional rows: every query sees the
n real keys), from float64 runs on SST-2 sentences padded to T:
  LayerNorm sites: Chebyshev inverse-sqrt seeds on the observed c_eff_sq(pos) var + eps c2 band (refit_norm_cheb's fit);
  GELU: the THOR composite of the exact (erf) GELU (thor_erf.py) on [-S, S], S = margin x the observed max |input|
    (BERT's massive activations reach |x| ~ 140 in a few channels: a single series cannot resolve the unit-scale
    curvature over that range; GELU_FOLD=1 puts 1 / S on the up-projection);
  softmax: SM_CHEB_GS reciprocal counts for kc <= T (smcheb.design), the score clip checked against the data.

    .venv/bin/python -m examples.bert_from_primitives.calib --out configs/model/approximation/bert_base_n32_t128/configs.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np

from perseus.impl import smcheb
from perseus.impl.config import parse_norm, parse_softmax

from . import data as D
from . import thor_erf

BASE = "configs/model/approximation/bert_base_n32/configs.json"


def check_torch(texts, T, st, cl):
    """Max |float64 reference - HF torch (float64)| over the classifier logits."""
    import torch
    from transformers import AutoModelForSequenceClassification
    m = AutoModelForSequenceClassification.from_pretrained(D.MODEL).double().eval()
    worst = 0.0
    for text in texts:
        ids, n = D.encode(text, T)
        with torch.no_grad():
            want = m(input_ids=torch.tensor(ids[None]),
                     attention_mask=torch.tensor((np.arange(T) < n)[None].astype(np.int64))).logits[0].numpy()
        got = D.head(D.exact_blocks(st, D.embed(ids, cl), n)[-1][0], cl)
        worst = max(worst, float(np.abs(got - want).max()))
    return worst


def ln_fit(lo, hi, scale, target, max_deg=31, max_nr=7):
    """refit_norm_cheb.fit_site with more Newton steps (BERT's LN2 bands span ~1e4)."""
    from perseus.impl.poly import NumpyOps, eval_chebyshev, inv_sqrt_newton_d2
    from refit_norm_cheb import depth
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
        for _ in range(30):
            r = np.abs(np.polynomial.chebyshev.chebval(t, c) / f - 1)
            c = np.polynomial.chebyshev.chebfit(t, f, deg, w=w * (r / r.max() + 0.05))
        for nr in range(0, max_nr + 1):
            y = inv_sqrt_newton_d2(ops, grid, eval_chebyshev(ops, grid, c, lo, hi), nr, 1.0 / (scale * scale))
            err = float(np.max(np.abs(y / exact - 1)))
            if err <= target:
                key = (depth(deg, nr), deg + 3 * nr)
                if best is None or key < best[0]:
                    best = (key, deg, nr, c, err)
                break
    if best is None:
        raise RuntimeError(f"no seed up to degree {max_deg} + {max_nr} Newton meets {target} on [{lo:.3g}, {hi:.3g}]")
    return best


def main(argv=None):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "utils"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--n", type=int, default=48, help="SST-2 validation sentences (plus the 128-token text)")
    ap.add_argument("--safety", type=float, default=2.0, help="LN band widening on each side")
    ap.add_argument("--ln-target", type=float, default=1e-4)
    ap.add_argument("--gelu-margin", type=float, default=1.15)
    ap.add_argument("--gelu-wide", type=float, default=100.0, help="S above which the composite takes degrees 47 / 31")
    ap.add_argument("--sm-tgt-last", type=float, default=1e-5)
    ap.add_argument("--sm-margin", type=float, default=1.1, help="exp domain half-width over the observed scores")
    a = ap.parse_args(argv)
    j = json.load(open(a.base))
    st, cl = D.store(), D.client()
    texts = [s for s, _ in D.sentences(a.n)] + [D.TEXT128.read_text().strip()]
    print(f"[calib] float64 reference vs HF torch: max logit diff {check_torch(texts[:4], a.T, st, cl):.2e}", flush=True)
    stats = {}
    ns = []
    for text in texts:
        ids, n = D.encode(text, a.T)
        ns.append(n)
        D.exact_blocks(st, D.embed(ids, cl), n, stats=stats)
    print(f"[calib] {len(texts)} rows, real lengths {min(ns)}..{max(ns)} (median {int(np.median(ns))})", flush=True)
    out = copy.deepcopy(j)
    L = int(j["model"]["n_layers"])
    for b in range(L):
        for site, key in (("ln_1", "ln1"), ("ln_2", "ln2")):
            name = f"transformer.h.{b}.{site}"
            cj = out["norm"][name]
            cs = list(cj.get("center_scale_sq") or [])
            if cs and len(cs) < a.T:
                cj["center_scale_sq"] = cs + [cs[-1]] * (a.T - len(cs))
            cfg = parse_norm(cj)
            c2 = np.array([cfg.c_eff_sq(p) for p in range(a.T)])
            zs = []
            for y in stats[(key, b)]:
                c = y - y.mean(-1, keepdims=True)
                zs.append(c2 * (c * c).mean(-1) + cfg.epsilon * c2)
            z = np.concatenate(zs)
            lo, hi = float(z.min()) / a.safety, float(z.max()) * a.safety
            (dp, _), deg, nr, coef, err = ln_fit(lo, hi, cfg.inv_out_scale, a.ln_target)
            cj.update(cheb_coeffs=[float(v) for v in coef], cheb_lo=lo, cheb_hi=hi, cheb_nr_iters=nr)
            print(f"[calib] {name:22s} z [{z.min():.3g}, {z.max():.3g}] deg {deg} + NR{nr} depth {dp} err {err:.1e}",
                  flush=True)
        name = f"transformer.h.{b}.mlp.act"
        u = np.concatenate([x.ravel() for x in stats[("gelu", b)]])
        X = a.gelu_margin * float(np.abs(u).max())
        d1, d2 = (47, 31) if X > a.gelu_wide else (31, 27)
        entry, err, _ = thor_erf.fit(X, d1, d2)
        entry["note"] = (f"THOR composite (d1 {d1}, d2 {d2}) of the erf GELU on [-S, S], S = {a.gelu_margin} x max|u| "
                         f"{np.abs(u).max():.4g} at T={a.T}; max err {err:.2e}")
        out["softgelu"][name] = entry
        name = f"transformer.h.{b}.attn"
        sj = out["softmax"][name]
        s = np.concatenate([x.ravel() for x in stats[("scores", b)]])
        sm = parse_softmax(sj)
        ks, tg = smcheb.design(sm.init_alpha, sm.init_beta, sm.log2delta2, a.sm_tgt_last,
                               tuple(sorted({k for k in smcheb.KC_DESIGN if k <= a.T} | {a.T})))
        sj["recip_iters_cheb"], sj["recip_targets"] = ks, tg
        # the exp's domain around the calibrated centre widened to the T-length scores: exp(s sf) on +-half sf
        mean = (sm.clip_hi + sm.clip_lo) / 2.0
        sf = 2.0 ** (-sm.log2delta1 - sm.log2delta2)
        half = max(float(sj["squeeze_bound"]), a.sm_margin * max(mean - s.min(), s.max() - mean))
        lo, hi = mean - half, mean + half
        u = np.cos(np.pi * (np.arange(64) + 0.5) / 64)
        cc = np.polynomial.chebyshev.chebfit(u, np.exp(half * sf * u), len(sj["cheb_coeffs"]) - 1)
        e_exp = float(np.abs(np.polynomial.chebyshev.chebval(np.linspace(-1, 1, 4001), cc)
                             - np.exp(half * sf * np.linspace(-1, 1, 4001))).max())
        sj.update(clip_lo=lo, clip_hi=hi, squeeze_bound=half, cheb_a=-half * sf, cheb_b=half * sf,
                  cheb_coeffs=[float(v) for v in cc])
        print(f"[calib] {name:22s} scores [{s.min():.3g}, {s.max():.3g}] clip [{sm.clip_lo:.3g}, {sm.clip_hi:.3g}] "
              f"-> [{lo:.3g}, {hi:.3g}] exp err {e_exp:.1e} | gelu S {X:.3g} err {err:.1e} | recip {ks}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[calib] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
