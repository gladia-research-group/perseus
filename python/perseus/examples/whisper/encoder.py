"""Whisper's encoder on ViT's blocked prefill (examples/vit/prefill.py: channel-blocked layout, PCMM linears,
bidirectional attention): the client encrypts the rows of five mel frames (weights.mel_rows), stage 0 opens with the
convolutional front-end as two PCMMs with their GELUs, then the six pre-LN blocks. The 1500 positions are padded to
whole 128-token pages (VitPrefill.prefill), the padding's keys masked. The last stage hands the decoder its
cross-attention caches: ln_post, then every decoder layer's K and V projections as one PCMM whose output channels are in
the decode's head order, so each chunk pair's output block is a cache pair, and each layer's cache filled from them
(impl.cross_attention.fill_cache): the decoder adopts the stage's output as it is."""
from __future__ import annotations

import numpy as np

from perseus.impl import blocked as BL
from perseus.impl.activation import gelu
from perseus.impl.cross_attention import cache_cts, cache_from, fill_cache
from perseus.impl.pcmm import PcmmLinear, band_shift, run as pcmm_run
from perseus.impl.poly import im_cleanse
from perseus.impl.prefill import Chunks, join, name_inputs, split

from perseus.examples.gpt2 import weights as GW
from perseus.examples.gpt2.prefill import _bias_blocks
from perseus.examples.vit.prefill import VitPrefill, add_rows, unscale_blocked

from . import weights as W

CONV1, CONV2 = "frontend.conv1.act", "frontend.conv2.act"
XKV = "xkv"


def stem_gelu(rt, H, ge, ch: Chunks, B, g, pad=None, out_scale=1.0):
    """_gelu_blocks on one chunk pair's blocks H[k] with the lane mask carrying `out_scale` (rides the series'
    coefficients when constant) and, `pad` = (row, block), zero at that row's lanes of that block (conv2's padding
    frame: one mask product on that ciphertext alone)."""
    ops, N, t = rt.ops, B.d.N, B.t
    out = []
    for k, h in enumerate(H):
        halves = []
        for half, x in enumerate(split(ops, h)):
            zero = None
            if pad is not None and k == pad[1] and ch.has(2 * g + half) and pad[0] in ch.idx(2 * g + half):
                zero = ch.idx(2 * g + half).index(pad[0])
            if zero is None:
                mask = ("pf.gelu.dense", lambda s: np.full(N, out_scale * s))
            else:
                def build(s, lane=zero):
                    m = np.full(N, out_scale * s)
                    m[np.arange(B.nch) * t + lane] = 0.0
                    return m
                mask = (("whisper.stem.pad", g, half, zero), build)
            halves.append(im_cleanse(ops, gelu(rt, x, ge, mask=mask)))
        out.append(join(ops, *halves))
    return out


class WhisperEncoder(VitPrefill):
    """VitPrefill with Whisper's front-end as stage 0's stem; the blocks are GPT-2's grammar (WhisperStore)."""

    def stem_weights(self):
        """(conv1, conv2) PCMMs. conv1 reads the encrypted rows (scale 1) and returns 0.5 h1 / X1 (the GELU's folded
        input, as an MLP's up-projection returns it); conv2 reads the GELU's output as a down-projection does and
        returns 0.5 h2 / X2."""
        if not hasattr(self, "_stem"):
            d, B = self.rt.dims, self.blocks
            A, a, C, c = W.stem_matrices(self.store)
            u1, u2 = (self._gelu_in_scale(n) for n in (CONV1, CONV2))
            A_pad = np.zeros((d.hid * (-(-A.shape[0] // d.hid)), A.shape[1]))
            A_pad[:A.shape[0]] = A
            self._stem = (
                PcmmLinear.encode(A_pad * u1, B.t, 0.5, a * u1, shift=band_shift("conv1"), name="conv1"),
                PcmmLinear.encode(C * u2, B.t, 0.25, 2.0 * c * u2, shift=band_shift("conv2"), name="conv2"))
            self._pos = W.positions(self.store)
        return self._stem

    def _gelu_in_scale(self, site):
        ge = self.cfgs.gelu[site]
        return 1.0 / ge.xmax if self.gelu_folded(ge) else 1.0

    def xkv_weights(self):
        """(the cross K / V PCMM, ln_post's shift): every decoder layer's K and V projections (weights.cross_kv_matrices)
        with ln_post's gain, descaled, folded into the rows and its bias as the normalised input's shift. The PCMM reads
        2 (LN + shift) (blocked.norm_blocked) at scale 0.5, so K and V come out as they are."""
        if not hasattr(self, "_xkv"):
            d, B = self.rt.dims, self.blocks
            Wm, b = W.cross_kv_matrices(W.WhisperStore("decoder", self.store.path), d.H)
            g, g_pad = (v * self.cfgs.ln_f.descale for v in GW.load_vector(self.store, "transformer.ln_f.weight",
                                                                           d.dim, d.hid))
            beta = GW.load_vector(self.store, "transformer.ln_f.bias", d.dim, d.hid)[0]
            self._xkv = (PcmmLinear.encode(Wm * g_pad[:, None], B.t, 0.5, 2.0 * b, shift=band_shift(XKV), name=XKV),
                         GW.ln_input_shift(g, beta, d.dim))
        return self._xkv

    def extra_linears(self):
        return list(self.stem_weights()) + [self.xkv_weights()[0]]

    def preload_prefill(self):
        self.xkv_weights()
        super().preload_prefill()

    def prefill_stages(self, ch: Chunks):
        """The blocks, then the cross K / V stage (`xkv`, stage n_layers): ln_post, the projections, each decoder
        layer's cache filled; its output the caches' ciphertexts, flat (`cross_caches`)."""
        rt, B = self.rt, self.blocks
        lin, shift = self.xkv_weights()

        def xkv(Xs):
            b = self.n_layers
            cap = self.enter_stage(b, 0)
            name_inputs(self.inf, Xs)
            with rt.step(XKV):
                Xg = [Xs[g * B.K:(g + 1) * B.K] for g in range(len(Xs) // B.K)]
                with rt.step("ln_f"):
                    Nf = BL.norm_blocked(rt, unscale_blocked(rt, Xg, B), self.cfgs.ln_f, ch, B)
                    Y = [[BL.shift_blocked(rt, n, shift, ch, g, k, B, 2.0) for k, n in enumerate(row)]
                         for g, row in enumerate(Nf)]
                with rt.step("proj"):
                    out = [_bias_blocks(rt, pcmm_run(rt, y, lin), lin, ch, g, B) for g, y in enumerate(Y)]
                caches = []
                for l in range(lin.n_out // 2):              # output block 2l: layer l's K pair, 2l + 1 its V pair
                    with rt.step(f"l{l}"):
                        caches.append(fill_cache(rt, [o[2 * l] for o in out], [o[2 * l + 1] for o in out],
                                                 ch.n_keys, tag=XKV))
            self.exit_stage(b, cap)
            return [x for kv in caches for x in cache_cts(kv)]
        return super().prefill_stages(ch) + [(XKV, None, xkv)]

    def cross_caches(self, Xs, n=W.T):
        """The xkv stage's flat output as the decoder's cross caches (one ComplexKVCache per decoder layer)."""
        per = len(Xs) // (self.xkv_weights()[0].n_out // 2)
        return [cache_from(Xs[l * per:(l + 1) * per], n, self.rt.dims) for l in range(len(Xs) // per)]

    def embed(self, Xg, ch: Chunks):
        """conv1 (PCMM + bias), GELU with conv2's zero padding frame, conv2 (PCMM + bias), GELU at half scale (the
        residual stream's scale), + the position embedding."""
        rt, B = self.rt, self.blocks
        conv1, conv2 = self.stem_weights()
        ge1, ge2 = self.cfgs.gelu[CONV1], self.cfgs.gelu[CONV2]
        out = []
        for g, x in enumerate(Xg):
            with rt.step("conv1"):
                H = _bias_blocks(rt, pcmm_run(rt, x, conv1), conv1, ch, g, B)
            with rt.step("gelu1"):
                G = stem_gelu(rt, H, ge1, ch, B, g, pad=W.STEM_PAD)
            with rt.step("conv2"):
                H = _bias_blocks(rt, pcmm_run(rt, G, conv2), conv2, ch, g, B)
            with rt.step("gelu2"):
                out.append(stem_gelu(rt, H, ge2, ch, B, g, out_scale=0.5))
        return add_rows(rt, out, self._pos, ch, B, "whisper.pos")
