"""HF ViT weights under the GPT-2 export's tensor names (examples/gpt2_from_primitives/weights.py reads them), the
image -> encrypted-input rows, the per-token embedding offsets and the float64 reference."""
from __future__ import annotations

import json
import os
import types

import numpy as np

MODEL = os.environ.get("VIT_MODEL", "google/vit-base-patch16-224")
# VIT_MASSIVE="187,382,759,183": the residual stream carries these channels (ViT-B/16's massive activations, up to
# ~1.7e3 from block 5 on) times VIT_MASSIVE_SCALE, so every ciphertext's range (the refreshes' precision) stays ~1e2;
# the producing linears carry the scale, each LayerNorm restores it at its input, the client on the decrypted rows
MASSIVE = [int(c) for c in os.environ.get("VIT_MASSIVE", "").split(",") if c]
MASSIVE_SCALE = float(os.environ.get("VIT_MASSIVE_SCALE", "0.03125"))


def residual_scale(dim=768):
    r = np.ones(dim)
    r[MASSIVE] = MASSIVE_SCALE
    return r
T, P, D = 197, 16, 768


def snapshot():
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors"])


class VitStore:
    """tensor(name) for the GPT-2 names gpt2_from_primitives.weights / .model read (float64)."""

    def __init__(self, path=None):
        from safetensors.numpy import load_file
        self.path = path or snapshot()
        self.raw = load_file(os.path.join(self.path, "model.safetensors"))
        self.config = json.load(open(os.path.join(self.path, "config.json")))
        self._zip = types.SimpleNamespace(filename=os.path.join(self.path, "model.safetensors"))
        self._cache = {}

    def r(self, n):
        return np.asarray(self.raw[n], dtype=np.float64)

    def _map(self, name):
        p = name.split(".")
        if name == "transformer.wte.weight":
            return self.r("classifier.weight")
        if name == "transformer.wpe.weight":
            return self.r("vit.embeddings.position_embeddings")[0]
        if name.startswith("transformer.ln_f."):
            return self.r("vit.layernorm." + p[-1])
        b, rest = int(p[2]), ".".join(p[3:])
        L = f"vit.encoder.layer.{b}."
        a = L + "attention.attention."
        if rest.startswith("ln_1."):
            return self.r(L + "layernorm_before." + p[-1])
        if rest.startswith("ln_2."):
            return self.r(L + "layernorm_after." + p[-1])
        if rest.startswith("attn.c_attn."):
            return np.concatenate([self.r(a + f"{x}.{p[-1]}") for x in ("query", "key", "value")], axis=0)
        if rest.startswith("attn.c_proj."):
            return self.r(L + "attention.output.dense." + p[-1])
        if rest.startswith("mlp.c_fc."):
            return self.r(L + "intermediate.dense." + p[-1])
        if rest.startswith("mlp.c_proj."):
            return self.r(L + "output.dense." + p[-1])
        raise KeyError(name)

    def tensor(self, name):
        if name not in self._cache:
            self._cache[name] = self._map(name)
        return self._cache[name]

    def has(self, name):
        try:
            self.tensor(name)
            return True
        except KeyError:
            return False

    def shape(self, name):
        return list(self.tensor(name).shape)

    def names(self):
        return sorted(self.raw)


def pixels(path):
    """ViTImageProcessor: RGB, bilinear 224 x 224, (x / 255 - 0.5) / 0.5."""
    from PIL import Image
    im = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    return (np.asarray(im, dtype=np.float64) / 255.0 - 0.5) / 0.5          # (224, 224, 3)


def patch_rows(px):
    """(T, 768): row 0 (CLS) zero, row 1 + 14 i + j the patch (i, j) flattened (channel, y, x) as the conv weight."""
    g = px.reshape(14, P, 14, P, 3).transpose(0, 2, 4, 1, 3).reshape(196, 3 * P * P)
    return np.vstack([np.zeros((1, 3 * P * P)), g])


def patch_matrix(store):
    """The conv as (768 in, 768 out)."""
    return store.r("vit.embeddings.patch_embeddings.projection.weight").reshape(D, -1).T


def embed_offsets(store):
    """(T, 768) added after the patch PCMM: CLS + pos 0 on row 0, conv bias + pos t on the patch rows."""
    pos = store.r("vit.embeddings.position_embeddings")[0]
    off = pos.copy()
    off[0] += store.r("vit.embeddings.cls_token")[0, 0]
    off[1:] += store.r("vit.embeddings.patch_embeddings.projection.bias")
    return off


def embed(store, rows):
    return rows @ patch_matrix(store) + embed_offsets(store)


def unscale(rows):
    """Decrypted residual rows (VIT_MASSIVE scaled) -> the model's."""
    return np.asarray(rows) / residual_scale(np.asarray(rows).shape[1])


def exact_blocks(store, x, n_layers=12):
    """Bidirectional HF ViT blocks in float64 on the embedded rows: the per-block hidden states."""
    from scipy.special import erf
    from examples.gpt2_from_primitives import weights as GW
    x = np.asarray(x, dtype=np.float64)
    eps = float(store.config.get("layer_norm_eps", 1e-12))

    def ln(h, g, b):
        mu = h.mean(axis=1, keepdims=True)
        return (h - mu) / np.sqrt(((h - mu) ** 2).mean(axis=1, keepdims=True) + eps) * g + b

    t = store.tensor
    out = []
    for blk in range(n_layers):
        n = GW.block_names(blk)
        h = ln(x, t(n["ln1_w"]), t(n["ln1_b"]))
        q, k, v = np.split(h @ t(n["qkv_w"]).T + t(n["qkv_b"]), 3, axis=1)
        o = np.zeros_like(q)
        for hh in range(12):
            sl = slice(hh * 64, (hh + 1) * 64)
            s = q[:, sl] @ k[:, sl].T / 8.0
            if os.environ.get("VIT_REF_CAUSAL"):
                s = s + np.triu(np.full(s.shape, -np.inf), 1)
            p = np.exp(s - s.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            o[:, sl] = p @ v[:, sl]
        x = x + o @ t(n["out_w"]).T + t(n["out_b"])
        h = ln(x, t(n["ln2_w"]), t(n["ln2_b"]))
        u = h @ t(n["up_w"]).T + t(n["up_b"])
        u = 0.5 * u * (1.0 + erf(u / np.sqrt(2.0)))
        x = x + u @ t(n["down_w"]).T + t(n["down_b"])
        out.append(x.copy())
    return out


def logits(store, hidden):
    """Final LayerNorm + the classifier on the CLS row of decrypted hidden rows."""
    eps = float(store.config.get("layer_norm_eps", 1e-12))
    h = np.asarray(hidden, dtype=np.float64)[:1]
    mu = h.mean(axis=1, keepdims=True)
    n = (h - mu) / np.sqrt(((h - mu) ** 2).mean(axis=1, keepdims=True) + eps) * store.tensor("transformer.ln_f.weight") \
        + store.tensor("transformer.ln_f.bias")
    return n @ store.r("classifier.weight").T + store.r("classifier.bias")


def block0_taps(store, x, b=0):
    """Block b's float64 values at tapped_block's points (x: the block's input rows)."""
    from scipy.special import erf
    from examples.gpt2_from_primitives import weights as GW
    eps = float(store.config.get("layer_norm_eps", 1e-12))
    t = store.tensor
    n = GW.block_names(b)
    nrm = lambda h: (h - h.mean(1, keepdims=True)) / np.sqrt(((h - h.mean(1, keepdims=True)) ** 2).mean(1, keepdims=True) + eps)
    out = {"ln1": nrm(x)}
    h = out["ln1"] * t(n["ln1_w"]) + t(n["ln1_b"])
    q, k, v = np.split(h @ t(n["qkv_w"]).T + t(n["qkv_b"]), 3, axis=1)
    o = np.zeros_like(q)
    for hh in range(12):
        sl = slice(hh * 64, (hh + 1) * 64)
        s = q[:, sl] @ k[:, sl].T / 8.0
        if os.environ.get("VIT_REF_CAUSAL"):
            s = s + np.triu(np.full(s.shape, -np.inf), 1)
        p = np.exp(s - s.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        o[:, sl] = p @ v[:, sl]
    out["attn_out"] = o @ t(n["out_w"]).T + t(n["out_b"])
    x = x + out["attn_out"]
    out["ln2"] = nrm(x)
    u = (out["ln2"] * t(n["ln2_w"]) + t(n["ln2_b"])) @ t(n["up_w"]).T + t(n["up_b"])
    out["mlp_out"] = (0.5 * u * (1.0 + erf(u / np.sqrt(2.0)))) @ t(n["down_w"]).T + t(n["down_b"])
    out["u_max"] = np.abs(u).max()
    out["u"] = u
    return out
