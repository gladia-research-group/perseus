"""BERT-base SST-2: the exported weights, the client ends (embeddings + LayerNorm, pooler + tanh + classifier), the
inputs (SST-2 sentences padded to T) and the float64 reference encoder."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

MODEL = "textattack/bert-base-uncased-SST-2"
TEXT128 = Path(__file__).with_name("bert128_text.txt")


def export_dir():
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return Path(hf) / "perseus" / "models" / MODEL / "classic"


def store():
    """The perseus export (weights.bin.zip, transformer.h.<b>.* grammar: perseus/export.py _bert_tensors)."""
    from examples.gpt2_from_primitives.weights import RawStore
    return RawStore(str(export_dir() / "weights.bin.zip"))


def client():
    return dict(np.load(export_dir() / "client.npz"))


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL)


def sentences(n: int, split: str = "validation"):
    """The first n SST-2 sentences of `split` with their labels."""
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", "sst2", split=split)
    return [(r["sentence"], int(r["label"])) for r in ds.select(range(min(n, len(ds))))]


def encode(text: str, T: int):
    """Token ids padded with [PAD] to T, and the real length (truncated to T)."""
    tok = tokenizer()
    ids = tok(text, truncation=True, max_length=T)["input_ids"]
    n = len(ids)
    return np.asarray(ids + [tok.pad_token_id] * (T - n), dtype=np.int64), n


def text128():
    ids, n = encode(TEXT128.read_text().strip(), 128)
    if n != 128:
        raise ValueError(f"{TEXT128} tokenizes to {n} tokens, not 128")
    return ids, n


def _ln(x, g, b, eps=1e-12):
    mu = x.mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + eps) * g + b


def embed(ids, cl=None):
    """Client front: word + position + token-type (0) embeddings, then the embeddings LayerNorm: (T, 768)."""
    cl = client() if cl is None else cl
    T = len(ids)
    x = cl["word_embeddings"][ids] + cl["position_embeddings"][:T] + cl["token_type_embeddings"][0]
    return _ln(x.astype(np.float64), cl["emb_ln_weight"], cl["emb_ln_bias"])


def head(hidden_cls, cl=None):
    """Client back: pooler dense + tanh + classifier on the CLS row: the 2 logits."""
    cl = client() if cl is None else cl
    p = np.tanh(cl["pooler_weight"] @ hidden_cls + cl["pooler_bias"])
    return cl["classifier_weight"] @ p + cl["classifier_bias"]


def gelu(x):
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def exact_blocks(st, rows, n_keys, n_layers=12, stats=None):
    """Post-LN BERT blocks in float64 on the embedded rows (T, 768), keys >= n_keys masked: per-block hidden states.
    `stats` (dict) collects per-site arrays: ln1 / ln2 inputs, the GELU input, the attention scores."""
    from examples.gpt2_from_primitives import weights as W
    x = np.asarray(rows, dtype=np.float64)
    T = x.shape[0]
    t = st.tensor
    H, dh = 12, 64
    out = []
    for b in range(n_layers):
        n = W.block_names(b)
        qkv = x @ t(n["qkv_w"]).T + t(n["qkv_b"])
        q, k, v = np.split(qkv, 3, axis=1)
        o = np.zeros_like(q)
        for hh in range(H):
            sl = slice(hh * dh, (hh + 1) * dh)
            s = q[:, sl] @ k[:n_keys, sl].T / np.sqrt(dh)
            if stats is not None:
                stats.setdefault(("scores", b), []).append(s)
            p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            o[:, sl] = p @ v[:n_keys, sl]
        y = x + o @ t(n["out_w"]).T + t(n["out_b"])
        if stats is not None:
            stats.setdefault(("ln1", b), []).append(y)
        x = _ln(y, t(n["ln1_w"]), t(n["ln1_b"]))
        u = x @ t(n["up_w"]).T + t(n["up_b"])
        if stats is not None:
            stats.setdefault(("gelu", b), []).append(u)
        y = x + gelu(u) @ t(n["down_w"]).T + t(n["down_b"])
        if stats is not None:
            stats.setdefault(("ln2", b), []).append(y)
        x = _ln(y, t(n["ln2_w"]), t(n["ln2_b"]))
        out.append(x.copy())
    return out


def kl2(logits, ref):
    """KL(ref || logits) of the 2-class softmaxes."""
    def lsm(z):
        z = np.asarray(z, dtype=np.float64)
        m = z.max()
        return z - m - np.log(np.exp(z - m).sum())
    a, b = lsm(ref), lsm(logits)
    return float(np.sum(np.exp(a) * (a - b)))
