#!/usr/bin/env python
"""The Llama decode softmax centred on key 0 (BOS) at any cache length up to --kmax, on a base config: fit_gpt2_decode_sink.py
for Llama-3.1-8B. Every query head subtracts its key-0 score (the GQA cache holds each KV head's keys minus its key 0).

On the starts of OpenWebText shard 0042's documents (--kmax tokens, BOS first, through the fp32 model with eager
attention on the visible GPU), a layer's U* is the largest over its query heads and queries of max_j u, u = s - s_0;
the rest is fit_gpt2_decode_sink.fit_sink. The second half of the documents is held out; a layer whose held-out u passes U,
or whose centred softmax is not shallower (depth()), keeps the base softmax.

    python scripts/utils/fit_llama_decode_sink.py --base python/perseus/configs/model/approximation/llama31_8b_n32_ctx512_silu/configs.json \\
        --out python/perseus/configs/model/approximation/llama31_8b_n32_ctx512_silus/configs.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fit_gpt2_decode_sink import OWT, depth, fit_sink   # noqa: E402

NAME = os.environ.get("LLAMA_MODEL", "meta-llama/Llama-3.1-8B")


def sink_ranges(model, docs):
    """Per layer, per query head: (max u, min u) over every causal row of every document, u = s - s_0."""
    import torch
    from transformers.models.llama import modeling_llama as ML
    stats = {}
    orig = ML.eager_attention_forward

    def hooked(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        s = (query @ ML.repeat_kv(key, module.num_key_value_groups).transpose(2, 3)) * scaling
        tri = torch.tril(torch.ones(s.shape[-2:], dtype=torch.bool, device=s.device))
        u = s - s[..., :1]
        hi = torch.where(tri, u, -torch.inf).amax((-1, -2))[0].double().cpu().numpy()
        lo = torch.where(tri, u, torch.inf).amin((-1, -2))[0].double().cpu().numpy()
        h0, l0 = stats.get(module.layer_idx, (hi, lo))
        stats[module.layer_idx] = (np.maximum(h0, hi), np.minimum(l0, lo))
        return orig(module, query, key, value, attention_mask, scaling, dropout=dropout, **kw)
    ML.eager_attention_forward = hooked
    try:
        with torch.no_grad():
            for d in docs:
                model(torch.tensor([d], device=model.device))
    finally:
        ML.eager_attention_forward = orig
    L = len(stats)
    return np.stack([stats[i][0] for i in range(L)]), np.stack([stats[i][1] for i in range(L)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--owt", help="OpenWebText shard 0042 (default: the HF cache's, downloaded if absent)")
    ap.add_argument("--kmax", type=int, default=512)
    ap.add_argument("--docs", type=int, default=48, help="documents fitted; as many again are held out")
    ap.add_argument("--tau", type=float, default=4.6)
    ap.add_argument("--margin", type=float, default=1.25)
    a = ap.parse_args()
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(NAME)
    if a.owt is None:
        from huggingface_hub import hf_hub_download
        a.owt = hf_hub_download(OWT[0], OWT[1], repo_type="dataset", revision=OWT[2])
    texts = pq.read_table(a.owt, columns=["text"]).column("text").to_pylist()
    rng = np.random.default_rng(1234)
    docs = []
    for i in rng.permutation(len(texts)):
        ids = tok(texts[i][:8 * a.kmax])["input_ids"]
        if len(ids) >= a.kmax:
            docs.append(ids[:a.kmax])
        if len(docs) == 2 * a.docs:
            break
    assert all(d[0] == tok.bos_token_id for d in docs)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="eager").cuda().eval()
    hi, lo = sink_ranges(model, docs[:a.docs])
    ho, _ = sink_ranges(model, docs[a.docs:])
    print(f"[fit] {a.docs} documents fitted, {len(docs) - a.docs} held out ({a.kmax} tokens)", flush=True)
    cfg = json.load(open(a.base))
    for i in range(hi.shape[0]):
        name = f"layers.{i}.attn"
        base = cfg["softmax"][name]
        c = fit_sink(hi[i], lo[i], base, a.kmax, a.tau, a.margin)
        over = float(ho[i].max()) > c["clip_hi"]
        keep = over or depth(c) >= depth(base)
        c["note"] = c["note"].replace("OpenWebText shard 0042 document starts", "OpenWebText shard 0042 document starts "
                                      "(Llama-3.1-8B, fp32)")
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
