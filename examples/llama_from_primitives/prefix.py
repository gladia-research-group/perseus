"""The BOS prefix: position 0 of every sequence is the same token, so its keys and values are
public constants. Llama-3 parks massive activations on it (residual entries ~300, mean square
~60, against ~1e-3 everywhere else from layer 2 on): run encrypted, it would set the range of
every approximation and bootstrap. The port computes its K/V exactly in plaintext (float64,
one token: attention over itself is its own V, RoPE at 0 is the identity) and writes them into
the caches as encryptions of public vectors; the encrypted decode starts at position 1."""
from __future__ import annotations

import numpy as np

from perseus.impl.attention import ComplexKVCache, KVCache, cache_k_push, cache_kv_push_pair, cache_kv_push_packed_complex, cache_v_push
from perseus.impl.layout import encode_linear_input, rearrange_qkv_biases
from perseus.impl.ops import NumpyOps
from perseus.impl.rt import Rt


def _rms(x, g, eps):
    return x / np.sqrt(np.mean(x * x) + eps) * g


def bos_kv(store, token: int | None = None, n_layers: int | None = None):
    """[(k, v)] per layer (the first `n_layers`) for the BOS token at position 0: (H_kv * d_head,)
    each, float64."""
    c = store.config
    tok = int(c["bos_token_id"] if token is None else token)
    eps = float(c["rms_norm_eps"])
    H, H_kv = int(c["num_attention_heads"]), int(c["num_key_value_heads"])
    dh = int(c["hidden_size"]) // H
    t = lambda n: store.tensor(n).astype(np.float64)
    x = t("model.embed_tokens.weight")[tok]
    out = []
    for b in range(int(c["num_hidden_layers"]) if n_layers is None else int(n_layers)):
        p = f"model.layers.{b}."
        n = _rms(x, t(p + "input_layernorm.weight"), eps)
        k = t(p + "self_attn.k_proj.weight") @ n
        v = t(p + "self_attn.v_proj.weight") @ n
        out.append((k, v))
        attn = np.repeat(v.reshape(H_kv, dh), H // H_kv, axis=0).ravel()     # softmax over itself = 1
        x = x + t(p + "self_attn.o_proj.weight") @ attn
        n = _rms(x, t(p + "post_attention_layernorm.weight"), eps)
        g = t(p + "mlp.gate_proj.weight") @ n
        x = x + t(p + "mlp.down_proj.weight") @ (g / (1.0 + np.exp(-g)) * (t(p + "mlp.up_proj.weight") @ n))
    return out


class PlainOps(NumpyOps):
    """NumpyOps plus the batched / bookkeeping members the attention kernels call."""
    complex_payload = True

    def mult_pt_many(self, a, vecs, keys, kind="weight"):
        return [a * np.asarray(v) for v in vecs]

    def rotate_many(self, a, steps):
        return [self.rotate(a, k) for k in steps]

    def mult_add_many(self, acc, vs, ss):
        for v, s in zip(vs, ss):
            acc += v * s
        return False

    def tag_reduce(self, a, stride):
        return a


def plain_cache(dims, k_feat, v_feat, packing: str, rope=None):
    """The cache structure one push of (k, v) at position 0 leaves, on plain slot vectors.
    k_feat / v_feat: (hid,) in the head-interleaved order (GQA-expanded)."""
    rt = Rt(PlainOps(), dims)
    N, hid = dims.N, dims.hid
    lin = lambda f: encode_linear_input(f, N, hid, hid)      # a linear's output layout: f[r] at r t
    if packing == "cachemir_complex":
        kv = ComplexKVCache(dims.d_head)
        cache_kv_push_packed_complex(rt, kv, lin(k_feat) + 1j * lin(v_feat), rope=rope, pos=0)
    else:
        kv = KVCache(dims.d_head)
        if packing == "cachemir":
            cache_kv_push_pair(rt, kv, lin(k_feat).astype(complex), lin(v_feat).astype(complex), rope=rope, pos=0)
        else:
            cache_k_push(rt, kv, lin(k_feat), rope=rope, pos=0)
            cache_v_push(rt, kv, lin(v_feat))
    return kv


def expanded_heads(vec, H, H_kv, d_head):
    """(H_kv d_head,) -> head-interleaved (H d_head,) like the K/V linears' outputs."""
    full = np.repeat(np.asarray(vec).reshape(H_kv, d_head), H // H_kv, axis=0).ravel()
    return rearrange_qkv_biases(full, H)
