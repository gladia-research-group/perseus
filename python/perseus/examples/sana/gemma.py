"""Gemma-2-2B (SANA's text encoder) as float64 torch functions of its state dict, split the way the port runs it: the
n_pub positions the instruction prefix fixes are public (causal: their states do not depend on the prompt), so their
keys / values per layer are plaintext constants; the prompt's n_tok positions are the encrypted rows, attending the
public keys and the earlier prompt keys. Every nonlinearity goes through a ref.Approx (exact here), which also sees
each site's input on the prompt rows (the calibration statistics).

    .venv/bin/python -m perseus.examples.sana.gemma --check ref      # vs the dumps' Gemma states
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from . import hf
from .ref import Approx

H, H_KV, DH, D, FF = 8, 4, 256, 2304, 9216
EPS = 1e-6
SOFTCAP = 50.0
THETA = 10000.0


class GemmaWeights:
    def __init__(self, path=None):
        import torch
        from transformers import AutoModel
        m = AutoModel.from_pretrained(path or hf.snapshot(), subfolder="text_encoder", dtype=torch.float64)
        self.sd = {n: p.detach() for n, p in m.state_dict().items()}
        self.n_layers = m.config.num_hidden_layers
        del m

    def __getitem__(self, name):
        return self.sd[name]

    def layer(self, b, name):
        return self.sd[f"layers.{b}.{name}"]


def rms(x, w, ap: Approx, site):
    """Gemma's RMSNorm: x / sqrt(mean x^2 + eps) (1 + w)."""
    return x * ap.inv_sqrt((x * x).mean(-1, keepdim=True) + EPS, site) * (1.0 + w)


def rope(x, pos):
    """Rotate-half RoPE of x (..., T, DH) at absolute positions pos (T,)."""
    import torch
    inv = 1.0 / THETA ** (torch.arange(0, DH, 2, dtype=torch.float64) / DH)
    ang = pos[:, None].double() * inv[None]
    cos, sin = torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)
    x1, x2 = x[..., :DH // 2], x[..., DH // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


def layer_kv(W, b, h, pos):
    """(K, V) (H_KV, T, DH) of the normed rows h at positions pos, K post-RoPE."""
    k = (h @ W.layer(b, "self_attn.k_proj.weight").T).reshape(-1, H_KV, DH).transpose(0, 1)
    v = (h @ W.layer(b, "self_attn.v_proj.weight").T).reshape(-1, H_KV, DH).transpose(0, 1)
    return rope(k, pos), v


def attention(W, b, x, pos, kv_pub, ap: Approx, site):
    """Self-attention of the rows x at positions pos (the prompt, causal among themselves) over the public keys /
    values kv_pub (H_KV, n_pub, DH) and their own."""
    import torch
    h = rms(x, W.layer(b, "input_layernorm.weight"), ap, site + ".ln_in")
    q = rope((h @ W.layer(b, "self_attn.q_proj.weight").T).reshape(-1, H, DH).transpose(0, 1), pos)
    k, v = layer_kv(W, b, h, pos)
    K = torch.cat([kv_pub[0], k], 1).repeat_interleave(H // H_KV, 0)          # (H, n_pub + T, DH)
    V = torch.cat([kv_pub[1], v], 1).repeat_interleave(H // H_KV, 0)
    s = q @ K.transpose(1, 2) / math.sqrt(DH)                                   # query_pre_attn_scalar 256
    ap.see(site + ".logit", s)
    s = SOFTCAP * ap.tanh(s / SOFTCAP, site + ".softcap")
    n_pub, T = kv_pub[0].shape[1], x.shape[0]
    valid = torch.ones(T, n_pub + T, dtype=torch.bool)
    valid[:, n_pub:] = torch.tril(torch.ones(T, T, dtype=torch.bool))
    P = ap.softmax(s, valid, site + ".sm")
    o = (P @ V).transpose(0, 1).reshape(T, H * DH)
    return o @ W.layer(b, "self_attn.o_proj.weight").T


def mlp(W, b, x, ap: Approx, site):
    h = rms(x, W.layer(b, "pre_feedforward_layernorm.weight"), ap, site + ".ln_ff")
    g = ap.gelu_tanh(h @ W.layer(b, "mlp.gate_proj.weight").T, site + ".act")
    u = h @ W.layer(b, "mlp.up_proj.weight").T
    return (g * u) @ W.layer(b, "mlp.down_proj.weight").T


def public_prefix(W, ids):
    """The exact states of the public positions: per layer (K, V) for the prompt rows to attend, and nothing else."""
    import torch
    x = W["embed_tokens.weight"][torch.as_tensor(ids)] * math.sqrt(D)
    pos = torch.arange(len(ids))
    ex = Approx()
    kvs = []
    empty = (torch.zeros(H_KV, 0, DH, dtype=torch.float64),) * 2
    for b in range(W.n_layers):
        h = rms(x, W.layer(b, "input_layernorm.weight"), ex, None)
        kvs.append(layer_kv(W, b, h, pos))
        x = x + rms(attention(W, b, x, pos, empty, ex, "pub"), W.layer(b, "post_attention_layernorm.weight"), ex, None)
        x = x + rms(mlp(W, b, x, ex, "pub"), W.layer(b, "post_feedforward_layernorm.weight"), ex, None)
    return kvs


def prompt_states(W, ids, n_pub, kvs, ap: Approx = None, on_layer=None):
    """The final-norm rows of the prompt positions ids[n_pub:] given the public prefix's (K, V)."""
    import torch
    ap = ap or Approx()
    x = W["embed_tokens.weight"][torch.as_tensor(ids[n_pub:])] * math.sqrt(D)
    pos = torch.arange(n_pub, len(ids))
    for b in range(W.n_layers):
        p = f"g{b}"
        ap.see(p + ".resid", x)
        a = attention(W, b, x, pos, kvs[b], ap, p)
        x = x + rms(a, W.layer(b, "post_attention_layernorm.weight"), ap, p + ".ln_post_attn")
        f = mlp(W, b, x, ap, p)
        x = x + rms(f, W.layer(b, "post_feedforward_layernorm.weight"), ap, p + ".ln_post_ff")
        if on_layer is not None:
            on_layer(b, x)
    return rms(x, W["norm.weight"], ap, "g.norm")


def main():
    import torch
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--check", default="ref")
    ap_.add_argument("--n", type=int, default=2)
    ap_.add_argument("--threads", type=int, default=4)
    a = ap_.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(a.threads)
    W = GemmaWeights()
    for i in range(a.n):
        z = np.load(hf.scratch(a.check, f"p{i}.npz"))
        n_pub, n_tok = int(z["n_pub"]), int(z["n_tok"])
        ids = z["ids"][:n_pub + n_tok]
        kvs = public_prefix(W, ids[:n_pub])
        layers = {}
        out = prompt_states(W, ids, n_pub, kvs, on_layer=lambda b, x: layers.__setitem__(b, x))
        ref = torch.as_tensor(z["gemma_out_sel"][1:1 + n_tok]).double()
        e = float((out - ref).abs().max() / ref.abs().max())
        el = max(float((layers[b] - torch.as_tensor(z["gemma"][b + 1]).double()).abs().max()
                       / torch.as_tensor(z["gemma"][b + 1]).abs().max()) for b in range(W.n_layers))
        print(f"p{i}: n_pub {n_pub} n_tok {n_tok}: final rel err {e:.2e}, worst layer {el:.2e}", flush=True)


if __name__ == "__main__":
    main()
