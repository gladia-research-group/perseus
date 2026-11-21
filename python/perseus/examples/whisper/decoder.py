"""Whisper's decoder on GPT-2's decode (examples/gpt2/model.py: cachemir_complex packing, the encrypted CutMax argmax
and its feedback): each block's cross-attention sub-block (gpt2.block.transformer_block `cross`) reads a cache the
encoder's output filled once per utterance (impl.cross_attention.fill_cache), and the LM head covers only the tokens
greedy decoding may emit (weights.Generation.allowed): the client maps its column back to the token."""
from __future__ import annotations

import dataclasses

import numpy as np

from perseus.impl.cross_attention import cross_masks
from perseus.impl.layout import rearrange_qkv_biases, rearrange_qkv_weights, rearrange_wo_weights
from perseus.impl.linear import EncodedLinear

from perseus.examples.gpt2 import weights as GW
from perseus.examples.gpt2.model import Gpt2Model

from . import weights as W


@dataclasses.dataclass
class CrossWeights:
    """One block's cross-attention: the query and out projections (EncodedLinear), ln_x's folded shift."""
    q: EncodedLinear
    out: EncodedLinear
    shift: np.ndarray


def cross_weights(store, cfgs, b, dims):
    """Block b's CrossWeights (gpt2.weights.block_matrices' rules: ln_x's gain, descaled, folded into the query's
    rows, its bias as the normalised input's shift; the heads in the decode's order)."""
    p, dr, dp, H = f"transformer.h.{b}.", dims.dim, dims.hid, dims.H
    lnx = cfgs.cross(b)[0]
    g, g_pad = (v * lnx.descale for v in GW.load_vector(store, p + "ln_x.weight", dr, dp))
    beta = GW.load_vector(store, p + "ln_x.bias", dr, dp)[0]
    Q = GW.load_qkv(store, p + "xattn.c_attn.weight", dr, dp)[0] * g_pad[:, None]
    bq = GW.load_qkv_bias(store, p + "xattn.c_attn.bias", dr, dp)[0]
    O = rearrange_wo_weights(GW.load_linear(store, p + "xattn.c_proj.weight", dr, dr, dp, dp), H)
    bo = GW.load_vector(store, p + "xattn.c_proj.bias", dr, dp)[1]
    enc = EncodedLinear.encode
    return CrossWeights(q=enc(rearrange_qkv_weights(Q, H), dims.N, dp, dp, rearrange_qkv_biases(bq, H),
                              name=f"b{b}.xq"),
                        out=enc(O, dims.N, dp, dp, bo, name=f"b{b}.xout"), shift=GW.ln_input_shift(g, beta, dr))


class WhisperDecoder(Gpt2Model):
    """Gpt2Model over WhisperStore("decoder") with the cross-attention caches (`fill`) and the emitted tokens' LM head
    (`tokens`: column -> token id)."""

    def __init__(self, inf, store, cfgs, *, gen=None, **kw):
        kw.setdefault("packing", "cachemir_complex")
        super().__init__(inf, store, cfgs, **kw)
        self.gen = gen or W.Generation(store.path)
        self.tokens = np.flatnonzero(self.gen.allowed(GW.vocab_size(store)))
        self.vocab = len(self.tokens)
        self.xkv = [None] * self.n_layers
        self.cross_metas = None          # adopt_values: [(level, deg)] the handed cache states arrive at, cache_cts'
                                         # order over the layers (the encoder plan's xkv_metas.json; None: the landing)
        # (level, deg) every token enters block 0 at: the fed-back token's (planner.fed_back_entry), so the forced
        # start's rows and the generated tokens share block 0's plan (None: a prompt row's fresh encryption)
        self.entry = None
        self._xw = {}

    def lm_matrix(self, fold_lnf):
        full = GW.lm_head_matrix(self.store, self.cfgs.ln_f, self.rt.dims, GW.vocab_size(self.store), fold_lnf)
        return full[:, self.tokens]

    def cross_weights(self, b):
        if b not in self._xw:
            self._xw[b] = cross_weights(self.store, self.cfgs, b, self.rt.dims)
        return self._xw[b]

    def preload(self):
        super().preload()
        for b in range(self.n_layers):
            self.cross_weights(b)

    def adopt(self, caches):
        """The cross caches of one utterance as the encoder's last stage filled them (WhisperEncoder.cross_caches)."""
        if len(caches) < self.n_layers:
            raise ValueError(f"{len(caches)} cross caches for {self.n_layers} blocks")
        self.xkv = list(caches[:self.n_layers])

    def adopt_values(self, vecs, n=W.T):
        """`adopt` from the caches' slot values (WhisperEncoder.cross_caches' ciphertexts decrypted, cache_cts' order per
        layer), each encrypted here at the refresh landing, as the device hand-off leaves them (every state refreshed
        once in the decoder's session): the decoder's twin on the encrypted encoder's output (run_encoder --save)."""
        from perseus.impl.cross_attention import cache_from
        ops, d = self.rt.ops, self.rt.dims
        per = len(vecs) // self.n_layers

        metas = self.cross_metas
        if metas is not None and len(metas) != len(vecs):
            raise ValueError(f"adopt_values: {len(vecs)} cache states, {len(metas)} handed metas")

        def enc(i, v):                    # as blocked.encode_blocked; at the encoder's handed (level, deg) if given
            ct = ops.add_pt(ops.encode_token(np.zeros(d.dim)), v)
            if metas is not None:
                ops.fhe.drop_to_level(ct, metas[i][0])
                ct.noise_deg = metas[i][1]
            return ct
        self.adopt([cache_from([enc(l * per + j, v) for j, v in enumerate(vecs[l * per:(l + 1) * per])], n, d)
                    for l in range(self.n_layers)])

    def fill_plain(self, enc_out):
        """`adopt_values` from the float encoder output (ln_post's): the caches the encoder's last stage would fill
        (weights.cross_cache_ref; the padding keys' lanes zero), a decoder check without the encrypted encoder."""
        vecs = [np.nan_to_num(re) + 1j * np.nan_to_num(im) for K, V in W.cross_kv(self.store, enc_out)
                for re, im in W.cross_cache_ref(K, V, len(enc_out), self.rt.dims)]
        self.adopt_values(vecs, n=len(enc_out))

    # ── Gpt2Model's hooks ──
    def block_cross(self, b):
        if self.xkv[b] is None:
            raise RuntimeError(f"block {b}: the cross cache is empty (fill it from the encoder's K / V first)")
        return self.cross_weights(b), self.xkv[b], self.cfgs.cross(b)

    def stage_caches(self, i):
        out = super().stage_caches(i)
        if out is None or self.xkv[i] is None:
            return out
        cts, keys = self.xkv[i].cts(f"impl.b{i}.x")
        return out[0] + cts, out[1] + keys

    def fixed_cache_keys(self, b):
        """Block b's cross cache keys: filled before the first token, only read (driver.PlannedRun's fixed states)."""
        return set(self.xkv[b].cts(f"impl.b{b}.x")[1]) if self.xkv[b] is not None else set()

    def step_masks(self, pos):
        items, seen = super().step_masks(pos), None
        for b in range(self.n_layers):
            if self.xkv[b] is not None:
                if seen is None:
                    seen = {k for k, _ in items}
                for key, build in cross_masks(self.rt, self.cfgs.cross(b)[1], self.xkv[b]):
                    if key not in seen:
                        seen.add(key)
                        items.append((key, build))
        return items

    def encode_input(self, row):
        """A row as the client encrypts it, at `entry` when set (the twin records only the metadata; the device run
        encrypts at its program's input metadata)."""
        x = super().encode_input(row)
        if self.entry is not None:
            x.meta = x.meta.replace(level=int(self.entry[0]), deg=int(self.entry[1]))
        return x

    # ── the client's side ──
    def prompt_rows(self):
        """The forced start's embedding rows (public tokens: the client encrypts them as it would any)."""
        return [GW.token_embedding(self.store, tok, p, self.rt.dims.dim) for p, tok in enumerate(self.gen.start)]

    def token_of(self, col):
        return int(self.tokens[int(col)])
