"""Generate the plaintext per-tap reference for pool[0] (RES=80), written as raw
float64[slots] in the filling layout (slot = feat*t + tok, t=32) — the exact format
the C++ tap dump uses. With VIT_TAP_REF=<dir> pointing here, the encrypted run loads
these and prints a per-block rel-error INLINE/LIVE (see dump_tap in vit_model.cu).

Usage:  VIT_TAP_REF=<dir> python scripts/vit_tap_ref.py
"""
import os
import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
torch.set_num_threads(8)
from transformers import AutoModelForImageClassification

RES, K = 80, 12
HID_PAD, T_STRIDE, N_REAL, T_REAL = 1024, 32, 768, 26
REF = os.environ["VIT_TAP_REF"]
os.makedirs(REF, exist_ok=True)
pool = np.load(os.environ["VIT_POOL"], mmap_mode="r")
hf = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", attn_implementation="eager").eval()


def write_tap(b, name, arr):   # arr: [T,768] -> filling slot vector
    v = np.zeros(HID_PAD * T_STRIDE, dtype=np.float64).reshape(HID_PAD, T_STRIDE)
    v[:N_REAL, :T_REAL] = arr.T                      # [feat, tok]
    v.reshape(-1).tofile(os.path.join(REF, f"b{b}_{name}.f64"))


def block_taps(layer, inp):
    with torch.no_grad():
        ln1 = layer.layernorm_before(inp)
        ctx = layer.attention.attention(ln1)[0]
        attn = layer.attention.output.dense(ctx)
        r1 = attn + inp
        ln2 = layer.layernorm_after(r1)
        mlp = layer.output.dense(layer.intermediate(ln2))
        res = mlp + r1
    return {"inp": inp, "ln1": ln1, "attn": attn, "r1": r1,
            "ln2": ln2, "mlp": mlp, "res": res}


pix = torch.nn.functional.interpolate(
    torch.from_numpy(np.stack([pool[0]])), size=(RES, RES), mode="bilinear", antialias=True)
with torch.no_grad():
    h = hf.vit.embeddings(pix, interpolate_pos_encoding=True)   # [1,T,768]
for b in range(K):
    layer = hf.vit.encoder.layer[b]
    taps = block_taps(layer, h)
    for name, tap in taps.items():
        write_tap(b, name, tap[0].numpy())
    with torch.no_grad():
        o = layer(h); h = o[0] if isinstance(o, tuple) else o
with torch.no_grad():
    lnf = hf.vit.layernorm(h)
write_tap(11, "resid_final", h[0].numpy())
write_tap(11, "lnf", lnf[0].numpy())
print(f"[ref] wrote plaintext taps for pool[0] -> {REF}")
