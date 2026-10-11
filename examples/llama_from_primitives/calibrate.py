"""Calibrate the approximation configs of the Llama port on teacher-forced text windows.

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m examples.llama_from_primitives.calibrate \
        --out configs/model/approximation/llama31_8b_n32/configs.json

Per site, from every position but 0 (the BOS prefix runs in plaintext, see model.py):
  norm     RMSNorm inverse sqrt: centering scale c (c * max|x| = center_target), the band of
           c^2 (mean(x^2) + eps) widened by `--norm-safety`, the output scale inv_out_scale =
           inv_y_max * sqrt(band lo) and the shallowest Chebyshev seed + Newton count meeting
           `--norm-err` (scripts/utils/refit_norm_cheb.py's fit)
  silu     THOR composite x * (p2(p1(x / xmax)) + 1/2) with p2 o p1 ~ tanh(x / 2) / 2 (the
           sigmoid's odd part), xmax = `--silu-safety` * max|gate|
  softmax  the THOR softmax fit of perseus.calibrate (approximation gpt2 knobs) on the score
           rows of the decode queries
"""
from __future__ import annotations

import argparse
import json
import math
import os
import types

import numpy as np

from . import hf


def _sigmoid_composite(S, d1, d2):
    """_fit_thor_composite (perseus/calibrate/approximations/gelu.py) retargeted at SiLU."""
    from perseus.calibrate.approximations.gelu import _thor_wlsq
    k = np.arange(1, 4001)
    cheb = np.cos((2 * k - 1) * np.pi / (2 * 4000))
    t = np.sort(np.concatenate([cheb, np.linspace(-1.0, 1.0, 4000)]))
    x = S * t
    g = x / 2.0
    T_tgt = np.tanh(g) / 2.0
    ref = x / (1.0 + np.exp(-x))
    wgt = 1.0 + 6.0 * np.exp(-0.5 * (g / 2.5) ** 2)
    smax = np.abs(g).max() ** (1.0 / 3.0)
    best = None
    for s_mono in (np.sign(g) * np.abs(g) ** (1.0 / 3.0) / smax, t):
        for gamma in np.linspace(1.0, 14.0, 27):
            p1 = _thor_wlsq(t, np.tanh(gamma * s_mono), d1, wgt)
            y1 = np.polynomial.polynomial.polyval(t, p1)
            p2 = _thor_wlsq(y1, T_tgt, d2, wgt)
            comp = np.polynomial.polynomial.polyval(y1, p2)
            err = np.abs(x * (comp + 0.5) - ref)
            e = float(err.max())
            if best is None or e < best[0]:
                best = (e, float(np.median(err)), p1, p2)
    return best


def fit_silu(gmax, safety, d1, d2):
    S = float(safety * gmax)
    e_max, e_med, p1, p2 = _sigmoid_composite(S, d1, d2)
    # odd p1 / p2: zero the even coefficients the least squares leaves at ~1e-10
    p1 = np.where(np.arange(len(p1)) % 2 == 1, p1, 0.0)
    p2 = np.where(np.arange(len(p2)) % 2 == 1, p2, 0.0)
    return {"method": "thor_composite", "xmax": S, "thor_p1": p1.tolist(), "thor_p2": p2.tolist(),
            "fit_err_max": e_max, "fit_err_med": e_med, "data_max": float(gmax)}


def fit_norm(ms, absmax, eps, a):
    """ms, absmax: per-token mean(x^2) and max|x| over positions >= 1."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "utils"))
    from refit_norm_cheb import fit_site
    c = a.center_target / float(absmax.max() * a.norm_absmax_margin)
    c2 = c * c
    z = c2 * (ms + eps)
    lo, hi = float(z.min()) / a.norm_safety, float(z.max()) * a.norm_safety
    ios = a.inv_y_max * math.sqrt(lo)
    (d, _), deg, nr, coef, err = fit_site(lo, hi, ios, a.norm_err, 31)
    return {"method": "remez", "kind": "rmsnorm", "eps": eps, "z0": 0.0, "center_scale": c,
            "inv_out_scale": ios, "Ncoeffs": [], "Dcoeffs": [], "lin_alpha": 0.0, "lin_beta": 0.0,
            "gs_lo": 0.0, "gs_hi": 1.0, "gs_iters": 0, "nr_iters": 0,
            "cheb_coeffs": [float(v) for v in coef], "cheb_lo": lo, "cheb_hi": hi, "cheb_nr_iters": nr,
            "fit_err": err, "depth": d, "data_lo": float(z.min()), "data_hi": float(z.max())}


def _gs_iters(lo, hi, alpha, beta, target, max_iters):
    """The Goldschmidt count (goldschmidt_inv_ndf) whose worst relative error on [lo, hi] meets
    `target` from the seed F0 = alpha - beta D."""
    from perseus.impl.poly import goldschmidt_inv_ndf
    from perseus.impl.ops import NumpyOps
    D = np.geomspace(lo, hi, 2001)
    for it in range(1, max_iters + 1):
        y = goldschmidt_inv_ndf(NumpyOps(), np.ones_like(D), D, alpha - beta * D, it)
        if np.max(np.abs(y * D - 1)) <= target:
            return it
    return max_iters


def fit_softmax_llama(rows, row_q, T, knobs, safety=2.0, target=1e-5, max_iters=24, kc_max=256):
    """The THOR softmax with every division sized on the data. perseus.calibrate's fit gives the
    clip range, the squaring split and the exp series; its division seeds assume GPT-2's flat
    attention (the per-kc refine shrink, a fixed count), which Llama's attention-sink heads break:
    their first denominators span ~e^(M/d2) and the refine sums sum p^2 do not shrink with kc. Here
    each stage runs the mirror (ref.softmax_ref's arithmetic) on the calibration rows and seeds its
    Goldschmidt on [lo, hi]: lo the smallest measured denominator / `safety`, hi a BOUND, not a
    measurement (a denominator past hi makes the linear seed's D F0 leave (0, 2) and the iteration
    diverge; one below lo only converges slower): the first sum is at most the exp at clip_hi,
    a refine sum 0.5 sqrt(kc) sum p^2 at most 0.5 sqrt(kc_max). The count is the shallowest meeting
    `target` on the band; each stage's actual output feeds the next. The per-kc refine scaling is
    pinned to 1."""
    import torch
    from perseus.calibrate.approximations.softmax import fit_softmax
    from perseus.calibrate.numerics import chebyshev_gs_init
    from perseus.impl.config import parse_softmax
    from perseus.impl.ops import NumpyOps
    from perseus.impl.poly import eval_chebyshev, goldschmidt_inv_ndf
    j = fit_softmax(torch.as_tensor(rows), torch.as_tensor(row_q), T, knobs)
    cfg = parse_softmax(j)
    ops = NumpyOps()
    s = np.asarray(rows, dtype=np.float64)
    valid = np.arange(T)[None, :] <= np.asarray(row_q)[:, None]
    kc = valid.sum(-1, keepdims=True).astype(np.float64)
    mean = (cfg.clip_hi + cfg.clip_lo) / 2.0
    sf = 2.0 ** (-cfg.log2delta1 - cfg.log2delta2)
    z = eval_chebyshev(ops, s - mean, cfg.cheb_coeffs, cfg.cheb_a / sf, cfg.cheb_b / sf)
    for _ in range(cfg.log2delta1):
        z = z * z
    z = np.where(valid, z / kc, 0.0)
    seeds, iters, bands = [], [], []
    for k in range(cfg.log2delta2 + 1):
        if k:
            y = 2.0 * y
            z = (y * y) * (0.5 * np.sqrt(kc) * 0.25)
        tot = z.sum(-1, keepdims=True)
        lo = float(tot.min()) / safety
        hi = math.exp((cfg.clip_hi - mean) * sf * 2 ** cfg.log2delta1) if k == 0 else 0.5 * math.sqrt(kc_max)
        hi = max(hi, float(tot.max())) * 1.05
        a, b = chebyshev_gs_init(lo, hi)
        n = _gs_iters(lo, hi, a, b, target, max_iters)
        y = goldschmidt_inv_ndf(ops, z, tot, a - b * tot, n)
        seeds.append((float(a), float(b))); iters.append(n); bands.append((lo, hi))
    j.update(init_alpha=seeds[0][0], init_beta=seeds[0][1], gs_iters_scaled=iters[0], gs_iters_first=iters[0],
             refine_alpha=[a for a, _ in seeds[1:]], refine_beta=[b for _, b in seeds[1:]],
             per_step_refine_iters=iters[1:], gs_iters_refine_scaled=max(iters[1:], default=0),
             sm_kc_r=[1.0] * cfg.log2delta2, den_bands=bands, kc_max=kc_max)
    return j


def softmax_knobs():
    from omegaconf import OmegaConf
    p = os.path.join(os.path.dirname(__file__), "..", "..", "perseus", "configs", "approximation", "gpt2.yaml")
    return OmegaConf.load(p).softmax


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", default=None, help="token windows .npy (default: hf.text_windows)")
    ap.add_argument("--nseq", type=int, default=40)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--center-target", type=float, default=12.0)
    ap.add_argument("--norm-absmax-margin", type=float, default=1.25)
    ap.add_argument("--inv-y-max", type=float, default=3.0)
    ap.add_argument("--norm-safety", type=float, default=2.0)
    ap.add_argument("--norm-err", type=float, default=1e-5)
    ap.add_argument("--silu-safety", type=float, default=1.5)
    ap.add_argument("--silu-d1", type=int, default=31)
    ap.add_argument("--silu-d2", type=int, default=27)
    ap.add_argument("--sm-safety", type=float, default=2.0, help="softmax denominator band widening")
    ap.add_argument("--sm-err", type=float, default=1e-5, help="softmax per-division relative error")
    ap.add_argument("--device", default="cuda", help="where the fp32 model runs (cpu when the GPU is busy)")
    ap.add_argument("--rotate", type=int, default=None, metavar="SEED",
                    help="rotate the residual stream by the seeded randomized Hadamard (weights.residual_rotation)")
    ap.add_argument("--sm-kc-max", type=int, default=256, help="longest cache the softmax seeds cover")
    ap.add_argument("--sm-margin", type=float, default=0.3, help="clip range margin (fraction of the span)")
    ap.add_argument("--sm-rows", type=int, default=16384, help="score rows sampled per layer")
    a = ap.parse_args(argv)

    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
    toks = np.load(a.tokens) if a.tokens else hf.text_windows(a.nseq, a.T)
    toks = toks[:a.nseq, :a.T]
    model = hf.load_model(device=a.device)
    cfg = model.config
    L = cfg.num_hidden_layers
    rot = None
    if a.rotate is not None:
        from .weights import residual_rotation
        rot = residual_rotation(cfg.hidden_size, a.rotate)
    st = hf.collect_sites(model, toks, rows_per_layer=a.sm_rows, rot=rot)          # per site: arrays over (seq, pos >= 1)

    out = {"model": {"name": hf.NAME, "n_layers": L, "n_embd": cfg.hidden_size,
                     "n_head": cfg.num_attention_heads, "n_kv_head": cfg.num_key_value_heads,
                     "n_inner": cfg.intermediate_size, "vocab": cfg.vocab_size,
                     # the rotary frequencies as HF computes them (llama3 scaling included)
                     "rope_inv_freq": model.model.rotary_emb.inv_freq.double().cpu().tolist(),
                     "rms_norm_eps": cfg.rms_norm_eps,
                     "residual_rotation": a.rotate},
           "norm": {}, "silu": {}, "softmax": {},
           "meta": {"calib_windows": int(toks.shape[0]), "T": int(toks.shape[1]), "args": vars(a)}}
    eps = float(cfg.rms_norm_eps)
    for name in hf.norm_sites(L):
        out["norm"][name] = fit_norm(st[name + ".ms"], st[name + ".absmax"], eps, a)
        n = out["norm"][name]
        print(f"[norm] {name:12s} c={n['center_scale']:.3g} band [{n['cheb_lo']:.3g},{n['cheb_hi']:.3g}] "
              f"ios={n['inv_out_scale']:.3g} depth={n['depth']} err={n['fit_err']:.1e}", flush=True)
    for b in range(L):
        s = fit_silu(float(st[f"layers.{b}.gate.absmax"].max()), a.silu_safety, a.silu_d1, a.silu_d2)
        # the Chebyshev basis the kernel evaluates: the power basis' coefficients reach ~1e8 on wide
        # ranges (layer 31, xmax ~50) and CKKS noise through them destroys the output
        from perseus.calibrate.thor_to_cheb import convert_site
        convert_site(s, f"layers.{b}.mlp.act")
        out["silu"][f"layers.{b}.mlp.act"] = s
        print(f"[silu] {b:2d} xmax={s['xmax']:.3g} err max={s['fit_err_max']:.2e} med={s['fit_err_med']:.2e}", flush=True)
    knobs = softmax_knobs()
    knobs.range_margin = a.sm_margin
    for b in range(L):
        rows, row_q = st[f"layers.{b}.scores"]
        sm = fit_softmax_llama(rows, row_q, int(toks.shape[1]), knobs, a.sm_safety, a.sm_err, kc_max=a.sm_kc_max)
        out["softmax"][f"layers.{b}.attn"] = sm
        print(f"[softmax] {b:2d} clip [{sm['clip_lo']:.3g},{sm['clip_hi']:.3g}] d1={sm['delta1']} d2={sm['delta2']} "
              f"gs={sm['gs_iters_scaled']} refine={sm['per_step_refine_iters']}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[write] {a.out}")


if __name__ == "__main__":
    main()
