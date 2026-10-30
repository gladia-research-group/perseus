#!/usr/bin/env python
"""The decode softmax centred on the sink key (key 0) at any cache length, on a base config (attention.complex_qkt).

Every head subtracts key 0's score before the exp (u = s - s_0 <= U), and key 0's own term keeps the head sum at
least exp(-U / delta2) of its largest value at every length: the band of the first reciprocal round no longer spans
the scores' spread, only U and the cache length. The exponentials are exp((u + c) / delta2), c placing that band
around 1 (fit_sink). Key 0 comes off on the K side (every key is pushed minus
key 0), so the bucketed K cache and the 512-key layouts need nothing more; the self / previous-key references of
fit_gpt2_decode16_ref.py need its 16-slot ladder and stay there.

On the starts of OpenWebText shard 0042's documents (up to --kmax tokens, through float64 GPT-2), a layer's U* is the
largest over its heads and queries of max_j u; U = margin x U*, u_lo the data's least u widened by 5 % of the range,
delta2 = 2^ceil(log2(U* / tau)), and the reciprocal counts are impl.smcheb.design's over cache lengths up to --kmax
(sm_cheb_design.py's design lengths), as fit_gpt2_decode16_ref.py. A layer keeps the base softmax unless the centred
one is shallower (depth()). The documents of the second half are held out: their max u per layer is printed against
U, and a layer whose held-out u passes U keeps the base softmax.

    python scripts/utils/fit_gpt2_decode_sink.py --base python/perseus/configs/model/approximation/gpt2_base_n32_ctx1024_g3/configs.json \\
        --out python/perseus/configs/model/approximation/gpt2_base_n32_ctx1024_g3s/configs.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "python"))
from gpt2_ref import _gelu, _ln, load_safetensors      # noqa: E402
from perseus.impl import smcheb                        # noqa: E402

OWT = ("Skylion007/openwebtext", "plain_text/train/0042.parquet", "433fe0f44ed7894fea29c08b3202aa348ccc6369")


def next_pow2(x):
    return 1 if x <= 1 else 1 << math.ceil(math.log2(x))


def cheb_exp_series(lo, hi, degree, min_nodes=200):
    """calibrate.numerics.cheb_exp_series (without its torch import)."""
    n = max(8 * (degree + 1), min_nodes)
    k = np.arange(1, n + 1)
    x = 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.cos((2 * k - 1) * np.pi / (2 * n))
    series = np.polynomial.chebyshev.Chebyshev.fit(x, np.exp(x), degree, domain=[lo, hi])
    return (series.coef / float(series(0.0))).astype(np.float64)


def sink_ranges(w, docs, n_layer=12, n_head=12):
    """Per layer, per head: (max u, min u) over every query of every document, u = s - s_0 (causal rows)."""
    hi = np.full((n_layer, n_head), -np.inf)
    lo = np.full((n_layer, n_head), np.inf)
    for ids in docs:
        T = len(ids)
        x = w["wte.weight"][ids][None] + w["wpe.weight"][:T][None]
        tri = np.tril(np.ones((T, T), bool))
        for i in range(n_layer):
            p = f"h.{i}."
            h = _ln(x, w[p + "ln_1.weight"], w[p + "ln_1.bias"]) @ w[p + "attn.c_attn.weight"] + w[p + "attn.c_attn.bias"]
            q, k, v = (a.reshape(1, T, n_head, -1).transpose(0, 2, 1, 3) for a in np.split(h, 3, axis=-1))
            s = q @ k.transpose(0, 1, 3, 2) / math.sqrt(q.shape[-1])
            u = s - s[..., :1]
            hi[i] = np.maximum(hi[i], np.where(tri, u, -np.inf).max((-1, -2))[0])
            lo[i] = np.minimum(lo[i], np.where(tri, u, np.inf).min((-1, -2))[0])
            sm = np.where(tri, s, -np.inf)
            att = np.exp(sm - sm.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            x = x + (att @ v).transpose(0, 2, 1, 3).reshape(1, T, -1) @ w[p + "attn.c_proj.weight"] + w[p + "attn.c_proj.bias"]
            m = _ln(x, w[p + "ln_2.weight"], w[p + "ln_2.bias"]) @ w[p + "mlp.c_fc.weight"] + w[p + "mlp.c_fc.bias"]
            x = x + _gelu(m) @ w[p + "mlp.c_proj.weight"] + w[p + "mlp.c_proj.bias"]
    return hi, lo


def fit_sink(top, low, base, kmax, tau, margin, lo_margin=0.05, d0=5.0, tgt=1e-3):
    """The centred softmax of one layer from its heads' (max u, min u)."""
    ustar, ulo = float(top.max()), float(low.min())
    U = margin * ustar
    lo = ulo - lo_margin * (U - ulo)
    W = U - lo
    delta2 = next_pow2(ustar / tau)
    delta1 = max(next_pow2(W / (d0 * delta2)), 1)
    delta0 = W / (delta1 * delta2)
    # the exponentials are exp((u + c) / delta2): the head sum's band [exp(c / delta2) / kmax, exp((U + c) / delta2) 1.05]
    # keeps its ratio (the rounds) for any c; c puts it around 1, so neither the sum nor its reciprocal outgrows the
    # ratio's square root (at c = -U, 1 at the top, the reciprocal reaches kmax exp(U / delta2): ~2^16 at 512 keys)
    c = 0.5 * (delta2 * math.log(kmax / 1.05) - U)
    mid = 0.5 * (lo + U)
    coeffs = cheb_exp_series(-0.5 * delta0, 0.5 * delta0, 8) * math.exp((mid + c) / (delta1 * delta2))
    band = (math.exp(c / delta2) / kmax, math.exp((U + c) / delta2) * 1.05)
    ia, ib = smcheb.cheb_seed(*band)
    R = int(round(math.log2(delta2)))
    kcs = tuple(sorted({k for k in smcheb.KC_DESIGN if k <= kmax} | {kmax}))
    ks, tg = smcheb.design(ia, ib, R, tgt_last=tgt, kcs=kcs)
    refine = smcheb.cheb_seed(*smcheb.refine_band(kmax, 0.3))
    cfg = dict(base)
    cfg.pop("poly_coeffs", None)
    cfg.update({
        "mid_x": mid, "input_scale": float(delta1 * delta2), "squeeze_bound": 0.5 * W,
        "clip_lo": lo, "clip_hi": U, "n_squarings": int(round(math.log2(delta1))), "refinement_iters": R,
        "cheb_coeffs": coeffs.tolist(), "cheb_a": -0.5 * delta0, "cheb_b": 0.5 * delta0,
        "init_alpha": ia, "init_beta": ib, "refine_alpha": [refine[0]] * R, "refine_beta": [refine[1]] * R,
        "gs_iters_scaled": int(ks[0]), "gs_iters_first": int(ks[0]), "gs_iters_refine_scaled": int(max(ks[1:], default=0)),
        "per_step_refine_iters": [int(k) for k in ks[1:]], "sm_kc_r": [1.0] * (R * kmax),
        "recip_iters_cheb": [int(k) for k in ks], "recip_targets": [float(t) for t in tg],
        "delta0": delta0, "delta1": int(delta1), "delta2": int(delta2), "decode_window": kmax, "calib_T": kmax,
        "ref_keys": ["sink"] * len(top), "ref_window": kmax,
        "note": (f"centred on key 0 (<= {kmax} keys): u = s - s_0 over OpenWebText shard 0042 document starts reach "
                 f"{ustar:.2f}, U = {margin:g} x that, u_lo {ulo:.2f} widened by {lo_margin:g} of the range; delta2 = "
                 f"2^ceil(log2(U* / {tau:g})); exp((u + c) / delta2), c = {c:.2f} (the band around 1); round-0 band "
                 f"[{band[0]:.3g}, {band[1]:.3g}]; SM_CHEB_GS counts "
                 f"(impl.smcheb.design, kc up to {kmax}, last target {tgt:g})"),
        "poly_coeffs": np.polynomial.chebyshev.cheb2poly(coeffs).tolist(),     # unused under cheb_coeffs
    })
    return cfg


def depth(c):
    """Levels of a softmax config from its exp to its last normalisation (fit_gpt2_decode16_ref.depth)."""
    k = c["recip_iters_cheb"]
    return 4 + int(c["n_squarings"]) + k[0] + 2 + sum(int(x) + 3 for x in k[1:])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--owt", help="OpenWebText shard 0042 (default: the HF cache's, downloaded if absent)")
    ap.add_argument("--gpt2", help="GPT-2's model.safetensors and tokenizer.json directory (default: the HF cache's)")
    ap.add_argument("--kmax", type=int, default=1024)
    ap.add_argument("--docs", type=int, default=48, help="documents fitted; as many again are held out")
    ap.add_argument("--tau", type=float, default=4.6)
    ap.add_argument("--margin", type=float, default=1.25)
    a = ap.parse_args()
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    hub = os.path.join(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")), "hub")
    snap = a.gpt2 or os.path.dirname(sorted(glob.glob(os.path.join(
        hub, "models--openai-community--gpt2", "snapshots", "*", "model.safetensors")))[0])
    tok = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))
    w = load_safetensors(os.path.join(snap, "model.safetensors"))
    if a.owt is None:
        from huggingface_hub import hf_hub_download
        a.owt = hf_hub_download(OWT[0], OWT[1], repo_type="dataset", revision=OWT[2])
    texts = pq.read_table(a.owt, columns=["text"]).column("text").to_pylist()
    rng = np.random.default_rng(1234)
    docs = []
    for i in rng.permutation(len(texts)):
        ids = tok.encode(texts[i][:8 * a.kmax]).ids
        if len(ids) >= a.kmax // 2:
            docs.append(np.array(ids[:a.kmax], np.int64))
        if len(docs) == 2 * a.docs:
            break
    hi, lo = sink_ranges(w, docs[:a.docs])
    ho, _ = sink_ranges(w, docs[a.docs:])
    print(f"[fit] {a.docs} documents fitted, {len(docs) - a.docs} held out (lengths {min(map(len, docs))}.."
          f"{max(map(len, docs))})", flush=True)
    cfg = json.load(open(a.base))
    for i in range(hi.shape[0]):
        name = f"transformer.h.{i}.attn"
        base = cfg["softmax"][name]
        c = fit_sink(hi[i], lo[i], base, a.kmax, a.tau, a.margin)
        over = float(ho[i].max()) > c["clip_hi"]
        keep = over or depth(c) >= depth(base)
        print(f"[fit] layer {i}: U {c['clip_hi']:.2f} (held-out max u {ho[i].max():.2f}) delta2 {c['delta2']} "
              f"delta1 {c['delta1']} counts {c['recip_iters_cheb']} depth {depth(c)} vs {depth(base)}"
              f"{': keeps the base softmax' if keep else ''}", flush=True)
        if not keep:
            cfg["softmax"][name] = c
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(cfg, f, indent=1)


if __name__ == "__main__":
    main()
