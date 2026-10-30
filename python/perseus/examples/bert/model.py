"""BERT-base on ``perseus.impl``: the GPT-2 prefill machinery (channel-blocked layout, PCMM linears) with the post-LN
block body and bidirectional attention (Chunks(bidir=True): every query over the unpadded keys).

Block (post-LN; the export maps attention.output.LayerNorm to ln_1, output.LayerNorm to ln_2, perseus/export.py):
    a = Out(Attn(QKV(x)));  x1 = LN1(x + a);  x2 = LN2(x1 + Down(GELU(Up(x1))))
The LayerNorm affine stays explicit (gamma on a mask, beta added): its output is the next residual, so it cannot fold
into the following linear as in GPT-2's pre-LN. Scales follow perseus/examples/gpt2/prefill.py
prefill_block_blocked: Q / K / V at 1/2, the up-projection at 1/2 (times 1 / S under GELU_FOLD), the norm output at
2 n / descale."""
from __future__ import annotations

import os
import time

import numpy as np

from perseus.impl import blocked as BL
from perseus.impl.driver import ImplModel
from perseus.impl.pcmm import PcmmLinear, band_shift, run as pcmm_run
from perseus.impl.prefill import Chunks, evict_block_masks, name_inputs

from perseus.examples.gpt2 import weights as W
from perseus.examples.gpt2.prefill import _bias_blocks, _gelu_blocks, qkv_encode, qkv_split

# PF_LAST_QUERY=1 (blocked attention): the last block computes [CLS] alone past its K / V (the head reads nothing else)
_LAST_QUERY = os.environ.get("PF_LAST_QUERY", "0") not in ("", "0")


def affine_blocked(rt, Xs, gamma, beta, ch: Chunks, B, site):
    """LN output rows: gamma / 2 x (the norm's 2 n / descale, gamma descaled) + beta on the filled lanes."""
    ops, N, t = rt.ops, B.d.N, B.t
    g = np.zeros(B.d.hid)
    g[:len(gamma)] = gamma
    out = []
    for gi, row in enumerate(Xs):
        r = []
        for k, x in enumerate(row):
            if B.real(k) == 0:
                r.append(x)
                continue
            gk = g[k * B.nch:(k + 1) * B.nch]
            y = rt.mult_mask(x, (site + ".g", k), lambda gk=gk: np.repeat(0.5 * gk, t))
            r.append(BL.shift_blocked(rt, y, beta, ch, gi, k, B, 1.0))
        out.append(r)
    return out


def bert_block(rt, Xs, w, cfgs, ch: Chunks, B, b: int, last=None):
    """`last`: the only position the block's output must hold (blocked.attention_last)."""
    ops = rt.ops
    ln1, ln2, sm, ge = cfgs
    with rt.step("qkv"):
        Q, Kc, V = [], [], []
        for g, x in enumerate(Xs):
            k, v, q = qkv_split(rt, _bias_blocks(rt, pcmm_run(rt, x, w["qkv"]), w["qkv"], ch, g, B), B)
            Kc.append(k); V.append(v); Q.append(q)
    with rt.step("attn"):
        if os.environ.get("PF_ATTN", "") == "ccmm":     # CCMM attention on the d_head-64 SinC packing (ccmm_mha.py)
            if last is not None:
                raise ValueError("PF_LAST_QUERY: blocked attention only")
            from perseus.impl import ccmm_mha
            O = ccmm_mha.attention(rt, Q, Kc, V, sm, ch, layer=b, h_real=B.d.H_real)
        elif last is not None:
            O = BL.attention_last(rt, Q, Kc, V, sm, ch, B, last)
            Xs = [BL.token_bcast(rt, row, ch, B, last) for row in Xs]
        else:
            O = BL.attention_blocked(rt, Q, Kc, V, sm, ch, B)
    with rt.step("out_proj"):
        A = [_bias_blocks(rt, pcmm_run(rt, o, w["out"]), w["out"], ch, g, B) for g, o in enumerate(O)]
    Y = [[ops.add(a, s) for a, s in zip(ra, rs)] for ra, rs in zip(A, Xs)]
    with rt.step("ln_1"):
        X1 = affine_blocked(rt, BL.norm_blocked(rt, Y, ln1, ch, B), w["g1"], w["b1"], ch, B, f"bert.ln1.{b}")
    out = []
    for g, x in enumerate(X1):
        with rt.step("up"):
            H = _bias_blocks(rt, pcmm_run(rt, x, w["up"]), w["up"], ch, g, B)
        with rt.step("gelu"):
            G = _gelu_blocks(rt, H, ge)
        with rt.step("down"):
            out.append(_bias_blocks(rt, pcmm_run(rt, G, w["down"]), w["down"], ch, g, B))
    Y = [[ops.add(a, s) for a, s in zip(ra, rs)] for ra, rs in zip(out, X1)]
    with rt.step("ln_2"):
        return affine_blocked(rt, BL.norm_blocked(rt, Y, ln2, ch, B), w["g2"], w["b2"], ch, B, f"bert.ln2.{b}")


class BertPrefill(ImplModel):
    def __init__(self, inf, store, cfgs, *, core=None, n_layers=None, profile=False):
        super().__init__(inf, core=core, profile=profile, kv_offload=False)
        self.store, self.cfgs = store, cfgs
        self.n_layers = int(n_layers if n_layers is not None else cfgs.model.n_layers)
        self._w = {}

    @property
    def blocks(self):
        return BL.Blocks(self.rt.dims, int(os.environ.get("PF_NCH", "512")))

    def weights(self, b):
        """Block b's PCMM linears (blocked head-grouped order, unfolded LayerNorms) and the descaled LN affines."""
        if b not in self._w:
            d, B = self.rt.dims, self.blocks
            ge = self.cfgs.block(b)[3]
            us = 1.0 / ge.xmax if (self.rt.ops.gelu_fold and ge.method in ("thor_composite", "chebyshev")) else 1.0
            m = W.block_matrices(self.store, self.cfgs, b, d, fold_ln1=False, fold_ln2=False)
            Er = d.E_real

            def enc(name, Wm, sc, bias):
                return PcmmLinear.encode(Wm, B.t, sc, bias, shift=band_shift(f"b{b}.{name}"), name=f"b{b}.{name}")
            qkv_cols, qkv_bias = qkv_encode(B, m)
            self._w[b] = {
                "qkv": enc("qkv", qkv_cols, 0.5, qkv_bias),
                "out": enc("out", B.group_rows(m.O), 1.0, m.bo),
                "up": enc("up", (m.Up * us)[:, :Er], 0.5, (m.bup * us)[:Er]),
                "down": enc("down", m.Down[:Er], 0.5, 2.0 * m.bdown),
                "g1": m.g1, "b1": m.b1, "g2": m.g2, "b2": m.b2}
        return self._w[b]

    def preload(self):
        for b in range(self.n_layers):
            self.weights(b)

    def stages(self, ch: Chunks):
        B = self.blocks

        def block_fn(b):
            def compute(Xs):
                cap = self.enter_stage(b, 0)
                name_inputs(self.inf, [x for row in Xs for x in row])
                with self.rt.step(f"blk{b}"):
                    last = 0 if _LAST_QUERY and b == self.n_layers - 1 else None
                    Xs = bert_block(self.rt, Xs, self.weights(b), self.cfgs.block(b), ch, B, b, last)
                self.exit_stage(b, cap)
                evict_block_masks(self.rt)
                return Xs
            return compute
        return [(f"blk{b}", None, block_fn(b)) for b in range(self.n_layers)]

    def encode(self, rows, n_keys):
        """The embedded rows (T, 768; the client's embeddings + LayerNorm) as the blocked prompt."""
        ch = Chunks(len(rows), self.blocks.t, bidir=True, keys=int(n_keys))
        return BL.encode_blocked(self.rt, rows, ch, self.blocks), ch

    def forward(self, rows, n_keys, on_block=None):
        Xs, ch = self.encode(rows, n_keys)
        self.start()
        self.token_begin(0)
        st = self.stages(ch)
        if on_block is not None:
            st = [(lab, pre, _tapped(fn, b, on_block)) for b, (lab, pre, fn) in enumerate(st)]
        Xs = self.run_stages(Xs, 0, st)
        return Xs, ch, self.token_end(0)

    def hidden(self, Xs, ch):
        return BL.decrypt_blocked(self.rt, Xs, ch, self.blocks, self.rt.dims.dim)


def _tapped(fn, b, cb):
    def compute(x):
        t0 = time.time()
        x = fn(x)
        cb(b, x, time.time() - t0)
        return x
    return compute
