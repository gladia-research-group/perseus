"""Per-tap FHE-vs-plaintext rel-error table for the ViT encoder gate.

The C++ forward (VIT_TAP_DUMP=<dir>) dumps each intermediate as raw float64[slots]
in the filling layout (slot = feat*t + tok). This script recomputes the SAME taps
in plaintext from the HF model for pool[0] and prints a per-block, per-op relative
error, so the first block/op where the encrypted forward diverges is obvious.

Usage:  VIT_TAP_DUMP=<dir> python scripts/vit_tap_compare.py
Taps per block: inp, ln1, attn_core (pre out-proj), attn (post out-proj), r1,
ln2, mlp (post down-proj), res.  Tail: resid_final, lnf.
"""
import os
import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
torch.set_num_threads(8)
from transformers import AutoModelForImageClassification

RES, K = 80, 12
HID_PAD, T_STRIDE, N_REAL, T_REAL = 1024, 32, 768, 26   # slots=32768, hidDim=1024
DUMP = os.environ["VIT_TAP_DUMP"]
pool = np.load(os.environ["VIT_POOL"], mmap_mode="r")
hf = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", attn_implementation="eager").eval()


def load_fhe(b, name):
    p = os.path.join(DUMP, f"b{b}_{name}.f64")
    if not os.path.exists(p):
        return None
    v = np.fromfile(p, dtype=np.float64)
    if v.size < HID_PAD * T_STRIDE:
        return None
    m = v[: HID_PAD * T_STRIDE].reshape(HID_PAD, T_STRIDE)   # [feat, tok]
    return m[:N_REAL, :T_REAL].T                             # [tok, feat]


def relerr(fhe, ref):
    if fhe is None:
        return None
    num = np.linalg.norm(fhe - ref, axis=1)
    den = np.linalg.norm(ref, axis=1) + 1e-12
    return num / den   # per-token


def scale_fit(fhe, ref):
    """Best-fit uniform scale c (over all real slots) and the residual rel-error
    after removing it. A pure scale bug -> c!=1 with a SMALL residual; a structural
    error -> residual stays large regardless of c. Layout-tolerant for scale bugs."""
    if fhe is None:
        return None, None
    f, r = fhe.ravel(), ref.ravel()
    c = float(f @ r / (r @ r + 1e-12))
    resid = float(np.linalg.norm(f - c * r) / (np.linalg.norm(r) + 1e-12))
    return c, resid


def block_taps(layer, inp):
    """Recompute the FHE tap points for one ViT encoder layer (inp: [T,768])."""
    with torch.no_grad():
        ln1 = layer.layernorm_before(inp)
        ctx = layer.attention.attention(ln1)[0]          # pre out-proj
        attn = layer.attention.output.dense(ctx)         # out-proj
        r1 = attn + inp
        ln2 = layer.layernorm_after(r1)
        gelu = layer.intermediate(ln2)                   # gelu(up_proj)
        mlp = layer.output.dense(gelu)                   # down-proj (pre-residual)
        res = mlp + r1
    return {"inp": inp, "ln1": ln1, "attn_core": ctx, "attn": attn,
            "r1": r1, "ln2": ln2, "mlp": mlp, "res": res}, res


pix = torch.nn.functional.interpolate(
    torch.from_numpy(np.stack([pool[0]])), size=(RES, RES), mode="bilinear", antialias=True)
with torch.no_grad():
    h = hf.vit.embeddings(pix, interpolate_pos_encoding=True)   # [1,T,768] (keep batch dim)

ORDER = ["inp", "ln1", "attn_core", "attn", "r1", "ln2", "mlp", "res"]
print(f"{'blk':>3} {'tap':<10} {'cls_rel':>9} {'mean_rel':>9} {'scale_c':>8} {'resid':>8}")
print("-" * 52)
first_bad = None
for b in range(K):
    layer = hf.vit.encoder.layer[b]
    ref, _ = block_taps(layer, h)
    with torch.no_grad():                       # authoritative next-h from the real layer
        o = layer(h); h = o[0] if isinstance(o, tuple) else o
    for name in ORDER:
        r = ref[name][0].numpy()                # drop batch -> [T,768]
        fhe = load_fhe(b, name)
        re = relerr(fhe, r)
        if re is None:
            print(f"{b:>3} {name:<10} {'(missing)':>9}")
            continue
        c, resid = scale_fit(fhe, r)
        print(f"{b:>3} {name:<10} {re[0]:>9.3f} {re.mean():>9.3f} {c:>8.3f} {resid:>8.3f}")
        if first_bad is None and re[0] > 0.15:
            first_bad = (b, name, round(float(re[0]), 3))

# tail: ln_f on the final residual (all tokens)
with torch.no_grad():
    lnf = hf.vit.layernorm(h)          # [1,T,768]
for name, ref in (("resid_final", h[0].numpy()), ("lnf", lnf[0].numpy())):
    re = relerr(load_fhe(11, name), ref)
    if re is None:
        print(f"{'  T':>3} {name:<10} {'(missing)':>9}")
    else:
        print(f"{'  T':>3} {name:<10} {re[0]:>9.3f} {re.mean():>9.3f} {re.max():>9.3f} {int(np.argmax(re)):>6}")

print("-" * 52)
print(f"FIRST CLS divergence (rel>0.15): {first_bad}")
