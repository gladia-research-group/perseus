"""Float64 GPT-2 small: the reference every encrypted output is checked against, from HuggingFace's weights.

    ref = GPT2("model.safetensors")       # openai-community/gpt2
    rows = ref.embed(ids)                  # wte[ids] + wpe[0..T): the inputs the client encrypts
    logits = ref.logits(rows)              # (T, vocab), causal

The forward is HuggingFace's GPT2LMHeadModel in float64 (gelu_new, LayerNorm eps 1e-5, the LM head tied to wte).
It reads the safetensors file itself, so it needs numpy only.
"""
from __future__ import annotations

import json
import struct

import numpy as np

_DTYPES = {"F32": np.float32, "F64": np.float64, "F16": np.float16}


def load_safetensors(path: str) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        data = f.read()
    out = {}
    for name, h in header.items():
        if name == "__metadata__":
            continue
        a, b = h["data_offsets"]
        out[name] = np.frombuffer(data[a:b], dtype=_DTYPES[h["dtype"]]).reshape(h["shape"]).astype(np.float64)
    return out


def _ln(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + eps) * g + b


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


class GPT2:
    def __init__(self, path: str, n_layer: int = 12, n_head: int = 12):
        self.w, self.n_layer, self.n_head = load_safetensors(path), n_layer, n_head

    def embed(self, ids) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        return self.w["wte.weight"][ids] + self.w["wpe.weight"][:len(ids)]

    def logits(self, rows) -> np.ndarray:
        w, x = self.w, np.asarray(rows, dtype=np.float64)
        T, d = x.shape
        hd = d // self.n_head
        mask = np.triu(np.full((T, T), -np.inf), 1)
        for i in range(self.n_layer):
            p = f"h.{i}."
            q, k, v = np.split(_ln(x, w[p + "ln_1.weight"], w[p + "ln_1.bias"]) @ w[p + "attn.c_attn.weight"]
                               + w[p + "attn.c_attn.bias"], 3, axis=-1)
            q, k, v = (a.reshape(T, self.n_head, hd).transpose(1, 0, 2) for a in (q, k, v))
            s = q @ k.transpose(0, 2, 1) / np.sqrt(hd) + mask
            s = np.exp(s - s.max(-1, keepdims=True))
            a = (s / s.sum(-1, keepdims=True)) @ v
            x = x + a.transpose(1, 0, 2).reshape(T, d) @ w[p + "attn.c_proj.weight"] + w[p + "attn.c_proj.bias"]
            h = _gelu(_ln(x, w[p + "ln_2.weight"], w[p + "ln_2.bias"]) @ w[p + "mlp.c_fc.weight"] + w[p + "mlp.c_fc.bias"])
            x = x + h @ w[p + "mlp.c_proj.weight"] + w[p + "mlp.c_proj.bias"]
        return _ln(x, w["ln_f.weight"], w["ln_f.bias"]) @ w["wte.weight"].T


def kl(ref_logits, logits) -> float:
    """KL(reference || encrypted) of one position's next-token distributions."""
    p, q = (np.asarray(a, dtype=np.float64) for a in (ref_logits, logits))
    p = np.exp(p - p.max()); p /= p.sum()
    q = np.exp(q - q.max()); q /= q.sum()
    return float(np.sum(p * (np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(q, 1e-300)))))
