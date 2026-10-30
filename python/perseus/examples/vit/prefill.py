"""ViT on the GPT-2 blocked prefill (perseus/examples/gpt2/prefill.py, PF_LAYOUT=blocked, PF_LINEAR=pcmm,
bidirectional Chunks): the client encrypts the patch pixels (CLS row zero); stage 0 opens with the patch embedding as a PCMM
plus the per-token offsets (CLS, position embeddings, conv bias) as plaintext, then the 12 pre-LN blocks."""
from __future__ import annotations

import os

import numpy as np

from perseus.impl import blocked as BL
from perseus.impl.pcmm import PcmmLinear, band_shift, run as pcmm_run
from perseus.impl.prefill import Chunks, evict_block_masks, name_inputs

from perseus.examples.gpt2.prefill import Gpt2Prefill, _bias_blocks, _gelu_blocks, qkv_encode, qkv_split

from . import weights as V

# PF_LAST_QUERY=1 (blocked attention): the last block computes [CLS] alone past its K / V (the head reads nothing else)
_LAST_QUERY = os.environ.get("PF_LAST_QUERY", "0") not in ("", "0")


def _pad(Wm, r, c):
    out = np.zeros((r, c))
    out[:Wm.shape[0], :Wm.shape[1]] = Wm
    return out


def add_rows(rt, Xg, rows, ch: Chunks, B, site):
    """Xg[g][k] + the plaintext (T, dim) rows in encode_blocked's layout."""
    ops, N, t = rt.ops, B.d.N, B.t
    if rows.shape[0] < ch.T:                   # the padded rows repeat the last ones (VitPrefill.prefill)
        n = rows.shape[0]
        rows = np.vstack([rows, rows[n - (ch.T - n):]])
    full = np.zeros((rows.shape[0], B.d.hid))
    full[:, :rows.shape[1]] = rows
    idx = np.arange(B.nch) * t
    out = []
    for g, blk in enumerate(Xg):
        row = []
        for k, x in enumerate(blk):
            def build(g=g, k=k):
                v = np.zeros(N, dtype=np.complex128)
                for half in (0, 1):
                    if ch.has(2 * g + half):
                        for lane, p in enumerate(ch.idx(2 * g + half)):
                            v[idx + lane] += (1j if half else 1.0) * full[p, k * B.nch:(k + 1) * B.nch]
                return v
            row.append(rt.add_mask(x, (site, g, k), build))
        out.append(row)
    return out


def unscale_blocked(rt, Xg, B, site="vit.unscale"):
    """Xg times 1 / residual_scale per channel (VIT_MASSIVE): the LayerNorm's true input."""
    if not V.MASSIVE:
        return Xg
    inv = np.zeros(B.d.hid)
    inv[:B.d.dim] = 1.0 / V.residual_scale(B.d.dim)
    return [[rt.mult_mask(x, (site, k), lambda k=k: np.repeat(inv[k * B.nch:(k + 1) * B.nch], B.t))
             for k, x in enumerate(row)] for row in Xg]


def vit_block(rt, Xs, w, cfgs, ch: Chunks, B, tap=None, layer=0, last=None):
    """prefill_block_blocked with the VIT_MASSIVE unscale at both LayerNorm inputs; tap(name, rows) at the
    hidden-width points (diagnostic). `last`: the only position the block's output must hold ([CLS],
    blocked.attention_last): the rest of the block runs on chunk pair 0 alone."""
    ops = rt.ops
    ln1, ln2, sm, ge = cfgs
    dec = (lambda Xg: BL.decrypt_blocked(rt, Xg, ch, B, rt.dims.dim)) if tap else None
    skip = Xs
    with rt.step("ln_1"):
        N1 = BL.norm_blocked(rt, unscale_blocked(rt, Xs, B), ln1, ch, B)
        if tap:
            tap("ln1", dec(N1))
        Y = [[BL.shift_blocked(rt, n, w["shift1"], ch, g, k, B, 2.0) for k, n in enumerate(row)]
             for g, row in enumerate(N1)]
    with rt.step("qkv"):
        Q, Kc, Vv = [], [], []
        for g, y in enumerate(Y):
            k, v, q = qkv_split(rt, _bias_blocks(rt, pcmm_run(rt, y, w["qkv"]), w["qkv"], ch, g, B), B)
            Kc.append(k); Vv.append(v); Q.append(q)
    with rt.step("attn"):
        if os.environ.get("PF_ATTN", "") == "ccmm":     # two-page bidirectional CCMM (ccmm_mha_paged.py)
            if last is not None:
                raise ValueError("PF_LAST_QUERY: blocked attention only")
            from perseus.impl import ccmm_mha_paged
            O = ccmm_mha_paged.attention(rt, Q, Kc, Vv, sm, ch, layer=layer, h_real=B.d.H_real)
        elif last is not None:
            O = BL.attention_last(rt, Q, Kc, Vv, sm, ch, B, last)
            skip = [BL.token_bcast(rt, skip[0], ch, B, last)]
        else:
            O = BL.attention_blocked(rt, Q, Kc, Vv, sm, ch, B)
    with rt.step("out_proj"):
        A = [_bias_blocks(rt, pcmm_run(rt, o, w["out"]), w["out"], ch, g, B) for g, o in enumerate(O)]
        if tap:
            tap("attn_out", V.unscale(dec(A)))
    Xs = [[ops.add(a, s) for a, s in zip(ra, rs)] for ra, rs in zip(A, skip)]
    skip = Xs
    with rt.step("ln_2"):
        N2 = BL.norm_blocked(rt, unscale_blocked(rt, Xs, B), ln2, ch, B)
        if tap:
            tap("ln2", dec(N2))
        Y = [[BL.shift_blocked(rt, n, w["shift2"], ch, g, k, B, 2.0) for k, n in enumerate(row)]
             for g, row in enumerate(N2)]
    out = []
    for g, y in enumerate(Y):
        with rt.step("up"):
            H = _bias_blocks(rt, pcmm_run(rt, y, w["up"]), w["up"], ch, g, B)
        with rt.step("gelu"):
            G = _gelu_blocks(rt, H, ge)
        with rt.step("down"):
            out.append(_bias_blocks(rt, pcmm_run(rt, G, w["down"]), w["down"], ch, g, B))
    if tap:
        tap("mlp_out", V.unscale(dec(out)))
    return [[ops.add(a, s) for a, s in zip(ra, rs)] for ra, rs in zip(out, skip)]


class VitPrefill(Gpt2Prefill):
    def embed_weights(self):
        if not hasattr(self, "_emb"):
            d, B = self.rt.dims, self.blocks
            r = V.residual_scale(d.dim)
            self._emb = PcmmLinear.encode(_pad(V.patch_matrix(self.store) * r, d.hid, d.hid), B.t, 1.0,
                                          shift=band_shift("emb"), name="emb")
            self._emb_off = V.embed_offsets(self.store) * r
        return self._emb

    def diag_weights(self, b):
        """Gpt2Prefill.diag_weights with the residual scale on the out / down projections (VIT_MASSIVE)."""
        cache = self.__dict__.setdefault("_pf_diag", {})
        if b in cache or not V.MASSIVE:
            return super().diag_weights(b)
        from perseus.examples.gpt2 import weights as GW
        d, B = self.rt.dims, self.blocks
        r = np.ones(d.hid)
        r[:d.dim] = V.residual_scale(d.dim)
        m = GW.block_matrices(self.store, self.cfgs, b, d, self.fold[0], self.fold[1])
        ge = self.cfgs.block(b)[3]
        us = 1.0 / ge.xmax if (self.rt.ops.gelu_fold and ge.method in ("thor_composite", "chebyshev")) else 1.0
        enc = lambda name, Wm, sc, bias: PcmmLinear.encode(Wm, B.t, sc, bias, shift=band_shift(f"b{b}.{name}"),
                                                           name=f"b{b}.{name}")
        Er = d.E_real
        qkv_cols, qkv_bias = qkv_encode(B, m)
        cache[b] = {
            "qkv": enc("qkv", qkv_cols, 0.25, 2.0 * qkv_bias),
            "out": enc("out", B.group_rows(m.O) * r, 1.0, m.bo * r),
            "up": enc("up", (m.Up * us)[:, :Er], 0.25, 2.0 * (m.bup * us)[:Er]),
            "down": enc("down", m.Down[:Er] * r, 0.5, 2.0 * m.bdown * r),
            "shift1": self.block_weights(b).shift1, "shift2": self.block_weights(b).shift2}
        return cache[b]

    def prefill(self, rows, on_block=None):
        """Gpt2Prefill.prefill on a bidirectional prompt (every token attends the T real keys). VIT_PAD=1 (default):
        the rows padded to whole 128-token pages with copies of the last ones, keys >= T masked, so no ciphertext
        slice is identically zero (a zero-magnitude value is not refreshable by the planner)."""
        from perseus.examples.gpt2.prefill import _tapped
        n = len(rows)
        if os.environ.get("VIT_PAD", "1") not in ("", "0"):
            full = -(-n // 128) * 128
            rows = np.vstack([rows, rows[n - (full - n):]]) if full > n else rows
        ch = Chunks(len(rows), self.blocks.t, bidir=True, keys=n)
        self.last_ch = ch
        self.start()
        self.token_begin(0)
        Xs = [x for row in BL.encode_blocked(self.rt, rows, ch, self.blocks) for x in row]
        stages = self.prefill_stages(ch)
        if on_block is not None:
            stages = [(lab, pre, _tapped(fn, b, on_block)) for b, (lab, pre, fn) in enumerate(stages)]
        Xs = self.run_stages(Xs, 0, stages)
        return Xs, ch, self.token_end(0)

    def hidden(self, Xs, ch):
        return super().hidden(Xs, ch)[:ch.n_keys]

    def preload_prefill(self):
        self.embed_weights()
        super().preload_prefill()

    def prefill_stages(self, ch: Chunks):
        B = self.blocks

        def block_fn(b):
            def compute(Xs):
                cap = self.enter_stage(b, 0)
                name_inputs(self.inf, Xs)
                with self.rt.step(f"blk{b}"):
                    Xg = [Xs[g * B.K:(g + 1) * B.K] for g in range(len(Xs) // B.K)]
                    if b == 0:
                        with self.rt.step("embed"):
                            Xg = [pcmm_run(self.rt, x, self.embed_weights()) for x in Xg]
                            Xg = add_rows(self.rt, Xg, self._emb_off, ch, B, "vit.emb")
                        if getattr(self, "dbg_embed", None) is not None:
                            self.dbg_embed(V.unscale(BL.decrypt_blocked(self.rt, Xg, ch, B, self.rt.dims.dim)))
                    tap = self.tap if b == getattr(self, "tap_block", 0) and getattr(self, "tap", None) else None
                    last = 0 if _LAST_QUERY and b == self.n_layers - 1 else None
                    Xg = vit_block(self.rt, Xg, self.diag_weights(b), self.cfgs.block(b), ch, B, tap, layer=b,
                                   last=last)
                    Xs = [x for row in Xg for x in row]
                self.exit_stage(b, cap)
                evict_block_masks(self.rt)
                return Xs
            return compute
        return [(f"blk{b}", f"b{b}.", block_fn(b)) for b in range(self.n_layers)]
