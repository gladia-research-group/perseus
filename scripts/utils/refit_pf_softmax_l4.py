#!/usr/bin/env python
"""Prefill-128's layer-4 softmax at delta2 16 / delta1 8 instead of 32 / 4 (one reciprocal round fewer).

delta1 x delta2 stays 128, so the exp series and its domain are the served ones bit for bit; the squarings move one
from the normalising rounds to the exp. The first reciprocal round's band is the head sums x3 at delta2 16 (D0 =
the key mean of exp((s - mid) / 16) per head and query) over 128-token windows of OpenWebText shard 0042 (document
starts + random offsets: fit_gpt2_decode16.windows with 128-token windows; never the dev prompts, which are C4), run
through float64 GPT-2 up to layer 4; the Chebyshev-scaled Goldschmidt counts come from impl.smcheb.design over kc =
1..128. (fit_softmax's rule, delta2 = next_pow2(M / 2 ln 2885), puts the served range M = 256.5 just past 16 x 15.94.)

    python scripts/utils/refit_pf_softmax_l4.py gpt2_base_n32_pf_mf gpt2_base_n32_pf_cm_mf     -> <name>_s16
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "speedrun", "harness"))
import fit_gpt2_decode16 as F                             # noqa: E402
from perseus.impl import smcheb                       # noqa: E402

A = os.path.join(REPO, "python/perseus/configs/model/approximation")
T, LAYER, N_DOCS = 128, 4, 1500


def layer_scores(w, ids):
    """Layer LAYER's causal scores [B, 12, T, T] (NaN above the diagonal), float64 GPT-2."""
    x = w["wte.weight"][ids] + w["wpe.weight"][:T]
    B = x.shape[0]
    causal = np.tril(np.ones((T, T), bool))
    for i in range(LAYER + 1):
        p = f"h.{i}."
        h = F.ln(x, w[p + "ln_1.weight"], w[p + "ln_1.bias"]) @ w[p + "attn.c_attn.weight"] + w[p + "attn.c_attn.bias"]
        q, k, v = (a.reshape(B, T, 12, 64).transpose(0, 2, 1, 3) for a in np.split(h, 3, axis=-1))
        s = q @ k.transpose(0, 1, 3, 2) / 8.0
        if i == LAYER:
            return np.where(causal, s, np.nan)
        sm = np.where(causal, s, -np.inf)
        e = np.exp(sm - sm.max(-1, keepdims=True))
        x = x + ((e / e.sum(-1, keepdims=True)) @ v).transpose(0, 2, 1, 3).reshape(B, T, 768) @ w[p + "attn.c_proj.weight"] \
            + w[p + "attn.c_proj.bias"]
        u = F.ln(x, w[p + "ln_2.weight"], w[p + "ln_2.bias"]) @ w[p + "mlp.c_fc.weight"] + w[p + "mlp.c_fc.bias"]
        x = x + F.gelu(u) @ w[p + "mlp.c_proj.weight"] + w[p + "mlp.c_proj.bias"]


def main():
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    from gpt2_ref import load_safetensors
    F.KMAX = T
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    snap = os.path.dirname(sorted(glob.glob(f"{hub}/models--openai-community--gpt2/snapshots/*/model.safetensors"))[0])
    tok = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))
    w = load_safetensors(os.path.join(snap, "model.safetensors"))
    owt = pq.read_table(F.data_file(F.OWT), columns=["text"]).column("text").to_pylist()
    start, rand = F.windows(owt, 1234, N_DOCS, 8000, 300, tok)
    ids = np.concatenate([start, rand])
    names = sys.argv[1:]
    base = json.load(open(f"{A}/{names[0]}/configs.json"))["softmax"][f"transformer.h.{LAYER}.attn"]
    mid, clo, chi = base["mid_x"], base["clip_lo"], base["clip_hi"]
    lo0, hi0 = np.inf, -np.inf
    for s0 in range(0, len(ids), 50):
        d0 = np.nanmean(np.exp((np.clip(layer_scores(w, ids[s0:s0 + 50]), clo, chi) - mid) / 16.0), axis=-1)
        lo0, hi0 = min(lo0, float(d0.min())), max(hi0, float(d0.max()))
    lo, hi = lo0 / 3.0, hi0 * 3.0
    ia, ib = smcheb.cheb_seed(lo, hi)
    ks, tg = smcheb.design(ia, ib, 4, kcs=tuple(range(1, T + 1)))
    refine = smcheb.cheb_seed(*smcheb.refine_band(T, 0.3))
    for name in names:
        cfg = json.load(open(f"{A}/{name}/configs.json"))
        c = dict(cfg["softmax"][f"transformer.h.{LAYER}.attn"])
        if (c["delta1"], c["delta2"], c["mid_x"]) != (4, 32, mid):
            raise SystemExit(f"{name}: layer {LAYER} is not the served delta2 32 / delta1 4 softmax")
        c.update({"delta1": 8, "delta2": 16, "n_squarings": 3, "refinement_iters": 4, "init_alpha": ia, "init_beta": ib,
                  "recip_iters_cheb": [int(k) for k in ks], "recip_targets": [float(t) for t in tg],
                  "gs_iters_scaled": int(ks[0]), "gs_iters_first": int(ks[0]), "gs_iters_refine_scaled": int(max(ks[1:])),
                  "per_step_refine_iters": [int(k) for k in ks[1:]], "refine_alpha": [refine[0]] * 4,
                  "refine_beta": [refine[1]] * 4, "sm_kc_r": c["sm_kc_r"][:4 * T]})
        c["note"] = (c.get("note", "") + f"; delta2 32 -> 16, delta1 4 -> 8 (the same exp series and domain), round-0 band "
                     f"[{lo:.4g}, {hi:.4g}] = the head sums x3 at delta2 16 over {len(ids)} OpenWebText shard 0042 "
                     f"128-token windows, counts {ks} at targets {tg} (scripts/utils/refit_pf_softmax_l4.py)").lstrip("; ")
        cfg["softmax"][f"transformer.h.{LAYER}.attn"] = c
        os.makedirs(f"{A}/{name}_s16", exist_ok=True)
        with open(f"{A}/{name}_s16/configs.json", "w") as f:
            json.dump(cfg, f, indent=1)
        print(f"{name}_s16: band [{lo:.4g}, {hi:.4g}] counts {ks} targets {tg}")


if __name__ == "__main__":
    main()
