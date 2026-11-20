"""A decoder (Llama / Gemma-2, arch.py) as an encrypted encoder of a short prompt after a public prefix: SANA-Sprint's
text encoder (Gemma-2-2B-IT, <= 16 prompt tokens after its 207-token instruction prefix) on the general blocked prefill.

The prompt rides `nch`-channel blocks (2048: 16 token lanes, one chunk pair), every block is
prefill.llama_prefill_block_blocked with the prompt at positions P .. (Chunks base P), and the prefix's keys / values
enter as plaintext: impl.prefill.PublicPrefix, two PCMMs per head group and layer (q K_pub^T onto the score
ciphertext's free key offsets, P V_pub back), K_pub / V_pub from one float64 pass of the checkpoint over the prefix.
The linears are named <lin_prefix><layer>.* (`g`) so a host model's session (SANA's) carries them beside its own; the
encoder's Rt wraps the host's inference session (same ciphertexts, its own dims)."""
from __future__ import annotations

import numpy as np

from perseus.impl import blocked as BL
from perseus.impl import pcmm as PC
from perseus.impl.prefill import Chunks, PublicPrefix, public_prefix_mats

from .prefill import LlamaPrefill, llama_prefill_block_blocked


def rope_rows(x, pos, inv_freq):
    """HF's rotate-half RoPE of x (..., P, d_head) at positions `pos`."""
    inv = np.asarray(inv_freq, dtype=np.float64)
    ang = np.asarray(pos, dtype=np.float64)[:, None] * inv[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def public_kv(store, inv_freq, ids, n_layers=None):
    """Per layer (K, V) of the prefix tokens `ids` (positions 0 ..): (H_kv, P, d_head) float64, K after RoPE; one fp64
    forward of the checkpoint in `store` (a directory: HF's eager attention, the soft-cap included)."""
    import torch
    from transformers import AutoModel
    m = AutoModel.from_pretrained(store.dir, dtype=torch.float64, attn_implementation="eager")
    out = {}

    def hook(b, kind):
        def f(_mod, _inp, o):
            out[(b, kind)] = o[0].double().cpu().numpy()
        return f
    layers = m.layers if hasattr(m, "layers") else m.model.layers
    hs = [h for b, layer in enumerate(layers) for h in (layer.self_attn.k_proj.register_forward_hook(hook(b, "k")),
                                                         layer.self_attn.v_proj.register_forward_hook(hook(b, "v")))]
    with torch.no_grad():
        m(torch.as_tensor(np.asarray(ids)[None]))
    for h in hs:
        h.remove()
    c = store.config
    hkv = int(c["num_key_value_heads"])
    dh = int(c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"])
    P = len(ids)
    res = []
    for b in range(len(layers) if n_layers is None else n_layers):
        k = out[(b, "k")].reshape(P, hkv, dh).transpose(1, 0, 2)
        v = out[(b, "v")].reshape(P, hkv, dh).transpose(1, 0, 2)
        res.append((rope_rows(k, np.arange(P), inv_freq), v))
    del m
    return res


def encoder_rot_steps(dims, nch=2048):
    """The encoder's rotation keys (signed): the prefill's, the PCMM's, the radix ladders', the blocked layout's."""
    from perseus.impl.layout import ladder_radix_bits, ladder_radix_steps
    from perseus.impl.prefill import prefill_rot_steps
    bits = ladder_radix_bits()
    return (prefill_rot_steps(dims) | PC.rot_steps() | ladder_radix_steps(dims, bits)
            | BL.blocked_rot_steps(BL.Blocks(dims, nch, 1), bits))


class Encoder(LlamaPrefill):
    """LlamaPrefill as an encoder: `nch`-channel blocks (no GQA grouping: K / V per query head), a PublicPrefix of
    `prefix_ids` on every layer, linears named <lin_prefix><b>.*. `block(Xs, b)` is one layer on the encoded prompt."""

    def __init__(self, inf, weights, cfgs, prefix_ids, *, nch=2048, lin_prefix="g", **kw):
        kw.setdefault("packing", "cachemir_complex")
        super().__init__(inf, weights, cfgs, **kw)
        self.nch, self.lin_prefix = int(nch), lin_prefix
        self.prefix_ids = [int(i) for i in prefix_ids]
        self.P = len(self.prefix_ids)
        self._kv = None
        self._pub = {}

    @property
    def blocks(self):
        return BL.Blocks(self.rt.dims, self.nch, 1)

    def chunks(self, T):
        return Chunks(int(T), self.blocks.t, base=self.P)

    def encode(self, rows):
        """The prompt's embedding rows (T, dim) as the blocked residual stream (flat: chunk pairs x hidden blocks)."""
        B = self.blocks
        return [x for g in BL.encode_blocked(self.rt, np.asarray(rows), self.chunks(len(rows)), B) for x in g]

    def public_kv(self):
        if self._kv is None:
            self._kv = public_kv(self.w.store, self.cfgs.model["rope_inv_freq"], self.prefix_ids, self.n_layers)
        return self._kv

    def public_prefix(self, b, ch):
        """Layer b's PublicPrefix per head group (query head h reads KV head h // (H / H_kv)), PCMMs on this rt."""
        key = (b, ch.t, ch.n_chunks)
        if key not in self._pub:
            B = self.blocks
            K, V = self.public_kv()[b]
            rep = self.H // self.H_kv
            gd = B.group_dims(0)
            gscale = 1.0 / np.sqrt(gd.d_head_real)
            rt = self.rt
            pres = []
            for k in range(B.Ka):
                heads = [h for h in range(k * B.hg, min((k + 1) * B.hg, self.H))]
                Kq, Vq = K[[h // rep for h in heads]], V[[h // rep for h in heads]]
                lins = {}

                def lin(c, i, which, k=k, Kq=Kq, Vq=Vq, lins=lins):
                    if c not in lins:
                        Ws, Wv = public_prefix_mats(B.group_dims(k), ch, c, Kq, Vq, gscale)
                        p = f"{self.lin_prefix}{b}"
                        lins[c] = (PC.PcmmLinear.encode(Ws, B.t, name=f"{p}.s_pub", shift=PC.band_shift(f"{p}.s_pub")),
                                   PC.PcmmLinear.encode(Wv, B.t, name=f"{p}.pv_pub", shift=PC.band_shift(f"{p}.pv_pub")))
                    return lins[c][0 if which == "s" else 1]
                # the base rt's PCMM engine (attention_blocked hands prefill_attention a fresh sub-rt per call)
                pres.append(PublicPrefix(self.P, lambda c, i, lin=lin: lin(c, i, "s"),
                                         lambda c, i, lin=lin: lin(c, i, "v"),
                                         lambda _rt, xs, ln: PC.run(rt, xs, ln)))
            self._pub[key] = pres
        return self._pub[key]

    def block(self, Xs, b, T):
        """Layer b on the encoded prompt Xs (flat, as encode's) of T tokens -> the layer's output (flat)."""
        B = self.blocks
        ch = self.chunks(T)
        nested = [Xs[g * B.Kh:(g + 1) * B.Kh] for g in range(len(Xs) // B.Kh)]
        with self.rt.step(f"{self.lin_prefix}blk{b}"):
            out = llama_prefill_block_blocked(self.rt, nested, self.pcmm_weights(b), self.cfgs.block(b), ch, self.rope,
                                              self.public_prefix(b, ch), B, None, layer=b, last=None,
                                              post=self.cfgs.post(b), softcap=self.cfgs.softcap(b), act_fn=self.arch.act)
        return [x for g in out for x in g]

    def linears(self, T):
        """Every PcmmLinear the layers use on a T-token prompt (a host session's specs): the block linears and the
        public prefix's, encoded now."""
        out = []
        ch = self.chunks(T)
        for b in range(self.n_layers):
            P = self.pcmm_weights(b)
            out += [P[k] for k in ("qkv", "out", "gate_up", "down")]
            for pre in self.public_prefix(b, ch):
                out += [pre.s_lin(c, 0) for c in range(ch.n_chunks)] + [pre.pv_lin(c, 0) for c in range(ch.n_chunks)]
        return out

    def decode(self, Xs, T):
        """The residual stream's rows (T, dim) of a flat Xs (fake / decrypting sessions)."""
        B = self.blocks
        nested = [Xs[g * B.Kh:(g + 1) * B.Kh] for g in range(len(Xs) // B.Kh)]
        return BL.decrypt_blocked(self.rt, nested, self.chunks(T), B, self.rt.dims.dim)
