#!/usr/bin/env python
"""Decode-8's softmax centred on per-head reference keys, on gpt2_base_n32_kc16_cm (attention._thor_exp).

Per layer and head the key whose score is subtracted before the exp: the first key (sink), the query's own (self) or the
one before it (prev), so the exponentials are exp((u - U) / delta2) <= 1 with u = s - s_ref and the reference's own term
keeps the head sum >= exp(-U / delta2). On 16-token windows of OpenWebText shard 0042 (fit_gpt2_decode16.py's
OpenWebText windows, run through float64 GPT-2), the layer's U* is the largest over its heads of the best reference's
max u; every head takes the first of sink / self / prev whose max u stays within U*. The rounds follow from U*: delta2 =
2^ceil(log2(U* / tau)), so a refreshed exp vector stays within e^tau of its head sum. The exp series runs on [u_lo, U]
(U = margin x U*, u_lo the data's least u widened by 5 % of the range), normalised to 1 at its top; the first
reciprocal round's band is the exact one, [exp(-U / delta2) / 16, 1.05], and the Chebyshev-scaled Goldschmidt counts come
from impl.smcheb.design over kc = 1..16. A layer keeps the base softmax unless the centred one is shallower (depth()).

    python scripts/utils/fit_gpt2_decode16_ref.py --out python/perseus/configs/model/approximation/gpt2_base_n32_kc16r_cm/configs.json
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
import fit_gpt2_decode16 as F                             # noqa: E402
from perseus.impl import smcheb                       # noqa: E402

BASE = os.path.join(REPO, "python/perseus/configs/model/approximation/gpt2_base_n32_kc16_cm/configs.json")
KMAX = F.KMAX
REFS = ("sink", "self", "prev")


def ranges(spec):
    """'0-10,12' -> [0, ..., 10, 12]; 'none' -> []."""
    if spec == "none":
        return []
    return sorted({i for part in spec.split(",") for i in range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1)})


def ref_ranges(rows):
    """Per reference kind: (max u, min u) per head over the score rows {kc: [B, H, kc]}."""
    out = {r: [np.full(12, -np.inf), np.full(12, np.inf)] for r in REFS}
    for kc, s in rows.items():
        idx = {"sink": 0, "self": kc - 1, "prev": max(kc - 2, 0)}
        for r in REFS:
            u = s - s[..., idx[r], None]
            out[r][0] = np.maximum(out[r][0], u.max(-1).max(0))
            out[r][1] = np.minimum(out[r][1], u.min(-1).min(0))
    return out


def fit_centred(rows, base, tau, margin, lo_margin=0.05, d0=5.0, tgt=1e-3):
    rng = ref_ranges(rows)
    top = np.stack([rng[r][0] for r in REFS])                  # [3, H]
    ustar = float(top.min(0).max())
    refs = [REFS[int(np.argmax(top[:, h] <= ustar))] for h in range(top.shape[1])]
    ulo = float(min(rng[r][1][h] for h, r in enumerate(refs)))
    U = margin * ustar
    lo = ulo - lo_margin * (U - ulo)
    W = U - lo
    delta2 = F.next_pow2(ustar / tau)
    delta1 = max(F.next_pow2(W / (d0 * delta2)), 1)
    delta0 = W / (delta1 * delta2)
    coeffs = F.cheb_exp_series(-0.5 * delta0, 0.5 * delta0, 8) * math.exp(-0.5 * delta0)   # 1 at the top, not mid
    band = (math.exp(-U / delta2) / KMAX, 1.05)
    ia, ib = smcheb.cheb_seed(*band)
    R = int(round(math.log2(delta2)))
    ks, tg = smcheb.design(ia, ib, R, tgt_last=tgt, kcs=tuple(range(1, KMAX + 1)))
    refine = smcheb.cheb_seed(*smcheb.refine_band(KMAX, 0.3))
    cfg = dict(base)
    cfg.pop("poly_coeffs", None)
    cfg.update({
        "mid_x": 0.5 * (lo + U), "input_scale": float(delta1 * delta2), "squeeze_bound": 0.5 * W,
        "clip_lo": lo, "clip_hi": U, "n_squarings": int(round(math.log2(delta1))), "refinement_iters": R,
        "cheb_coeffs": coeffs.tolist(), "cheb_a": -0.5 * delta0, "cheb_b": 0.5 * delta0,
        "init_alpha": ia, "init_beta": ib, "refine_alpha": [refine[0]] * R, "refine_beta": [refine[1]] * R,
        "gs_iters_scaled": int(ks[0]), "gs_iters_first": int(ks[0]), "gs_iters_refine_scaled": int(max(ks[1:], default=0)),
        "per_step_refine_iters": [int(k) for k in ks[1:]], "sm_kc_r": [1.0] * (R * KMAX),
        "recip_iters_cheb": [int(k) for k in ks], "recip_targets": [float(t) for t in tg],
        "delta0": delta0, "delta1": int(delta1), "delta2": int(delta2), "decode_window": KMAX, "calib_T": KMAX,
        "ref_keys": refs, "ref_window": KMAX,
        "note": (f"Decode-8 geometry (<= {KMAX} keys), centred on each head's reference key: u = s - s_ref over "
                 f"16-token windows of OpenWebText shard 0042 reach {ustar:.2f} (the best reference per head), "
                 f"U = {margin:g} x that, u_lo {ulo:.2f} widened by {lo_margin:g} of the range; delta2 = "
                 f"2^ceil(log2(U* / {tau:g})); round-0 band [exp(-U / delta2) / {KMAX}, 1.05]; SM_CHEB_GS counts "
                 f"(impl.smcheb.design, kc 1..{KMAX}, last target {tgt:g})"),
        "poly_coeffs": np.polynomial.chebyshev.cheb2poly(coeffs).tolist(),     # unused under cheb_coeffs
    })
    return cfg


def depth(c):
    """Levels of a softmax config from its exp to its last normalisation: the degree-8 series (4), the squarings,
    and per reciprocal round its count plus the head sum's mask and the product (+ the square past round 0)."""
    k = c["recip_iters_cheb"]
    return 4 + int(c["n_squarings"]) + k[0] + 2 + sum(int(x) + 3 for x in k[1:])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--owt", help="Skylion007/openwebtext plain_text/train/0042.parquet (downloaded if absent; must "
                    "match its sha256)")
    ap.add_argument("--gpt2", help="GPT-2's model.safetensors and tokenizer.json directory (default: the HF cache's "
                    "openai-community/gpt2 snapshot)")
    ap.add_argument("--layers", default="0-11", help="layers whose softmax may be centred (e.g. 0-11, or none); a "
                    "layer keeps the base softmax unless the centred one is shallower (depth())")
    ap.add_argument("--tau", type=float, default=4.6)
    ap.add_argument("--margin", type=float, default=1.25)
    a = ap.parse_args()
    cfg = json.load(open(a.base))
    layers = ranges(a.layers)
    if layers:
        import pyarrow.parquet as pq
        from tokenizers import Tokenizer
        hub = os.path.join(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")), "hub")
        snap = a.gpt2 or os.path.dirname(sorted(glob.glob(os.path.join(
            hub, "models--openai-community--gpt2", "snapshots", "*", "model.safetensors")))[0])
        tok = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))
        w = F.load_safetensors(os.path.join(snap, "model.safetensors"))
        owt = pq.read_table(F.data_file(F.OWT, a.owt), columns=["text"]).column("text").to_pylist()
        o_start, o_rand = F.windows(owt, 1234, 6000, 4000, 160, tok)
        scores, gelu_max = [dict() for _ in range(12)], [0.0] * 12
        for ids in (np.concatenate([o_start[:1500], o_rand[:1500]]), np.concatenate([o_start[1500:], o_rand[1500:]])):
            for s0 in range(0, len(ids), 50):
                F.taps(w, ids[s0:s0 + 50], scores, gelu_max)
            print(f"[fit] {len(ids)} windows", flush=True)
        for i in layers:
            name = f"transformer.h.{i}.attn"
            rows = {kc: np.concatenate(r, 0) for kc, r in scores[i].items()}
            c, base = fit_centred(rows, cfg["softmax"][name], a.tau, a.margin), cfg["softmax"][name]
            keep = depth(c) >= depth(base)
            print(f"[fit] layer {i}: refs {''.join(r[1] if r == 'self' else r[0] for r in c['ref_keys'])} "
                  f"U {c['clip_hi']:.2f} delta2 {c['delta2']} delta1 {c['delta1']} counts {c['recip_iters_cheb']} depth "
                  f"{depth(c)} vs {depth(base)}{': keeps the base softmax' if keep else ''}", flush=True)
            if not keep:
                cfg["softmax"][name] = c
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(cfg, f, indent=1)


if __name__ == "__main__":
    main()
