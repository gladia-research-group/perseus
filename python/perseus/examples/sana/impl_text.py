"""The whole text-to-image forward encrypted: SanaImpl with the text side on the server. The client's ciphertexts are
the model input's and the carry's rows and the prompt's Gemma embedding rows (E[id] sqrt(2304) / r, the n_tok tokens
on the real lanes of one chunk); 26 Gemma-2 stages (gemma_fhe.layer: the instruction prefix's keys / values
plaintext) and the caption stage (final norm, caption_projection, caption_norm) produce SANA's text rows T, then
SanaImpl's steps."""
from __future__ import annotations

import math

import numpy as np

from . import gemma as GM
from . import gemma_fhe as GF
from .impl import SanaImpl
from .layout import G


class SanaTextImpl(SanaImpl):
    def __init__(self, inf, W, nl, Wg, *, n_pub, prefix_ids, core=None, n_layers=None, profile=False,
                 text_layers=None):
        super().__init__(inf, W, nl, core=core, n_layers=n_layers, profile=profile)
        self.Wg, self.n_pub = Wg, n_pub
        self.text_layers = Wg.n_layers if text_layers is None else text_layers      # fewer: a debug cut
        self.kvs = GM.public_prefix(Wg, prefix_ids)          # the instruction prefix: public, the same every prompt
        self.r_text = np.asarray(getattr(nl, "cfg", {}).get("resid_scale_text", np.ones(GM.D)))
        self._lw = {}

    def lw(self, b):
        if b not in self._lw:
            site = f"g{b}.act"
            cfg = getattr(self.nl, "cfg", {})
            xm = (cfg["gelu"][site]["b"] if GF.gelu_fold() and site in cfg.get("gelu", {})
                  and site not in cfg.get("gelu_clip", {}) and cfg["gelu"][site]["a"] == -cfg["gelu"][site]["b"] else None)
            sc = cfg.get("softcap", {}).get(f"g{b}.attn.softcap")
            scs = (1.0 / sc["b"] if GF.sm_fold() and sc is not None and abs(sc["a"] + sc["b"]) < 1e-12 * abs(sc["b"])
                   and f"g{b}.attn.sm.exp" in cfg.get("exp", {}) else 1.0)
            self._lw[b] = GF.fold_layer(self.Wg, b, self.kvs[b], self.r_text, gelu_xm=xm, sc_scale=scs)
        return self._lw[b]

    def set_prompt_ids(self, x_in, prompt_ids, txt_bos, z2=None):
        """The prompt by its token ids after the public prefix (the client embeds them)."""
        import torch
        rows = (self.Wg["embed_tokens.weight"][torch.as_tensor(prompt_ids)] * math.sqrt(GM.D)).numpy()
        self.emb_rows = rows / self.r_text
        self.set_prompt(x_in, np.zeros((len(prompt_ids), 1152)), txt_bos, z2)

    def input_values(self, x_in=None, txt_rows=None, z2=None):
        """The model input's and the carry's rows (2 G ciphertexts), then the prompt's embedding rows (gemma_fhe's 5
        blocks)."""
        noise = super().input_values(x_in, np.zeros((len(self.emb_rows), 1152)), z2)[:2 * G]
        return noise + GF.encode(self.emb_rows, GF.HB)

    def stages(self):
        rt, nl = self.rt, self.nl
        n_tok, n_pub = len(self.emb_rows), self.n_pub
        nx = 2 * G                                    # the model input's and the carry's ciphertexts ride ahead

        def tlin(rt_, X, L):                                     # the prompt's real lanes only (bias included)
            return self.lin(rt_, X, L, (n_tok, 0))

        def gemma(b):
            def fn(state):
                xs, X = state[:nx], state[nx:]
                X = GF.layer(rt, X, self.lw(b), n_tok, n_pub, nl, tlin, b)
                return [rt.ops.copy(x) for x in xs] + [rt.ops.realize(x) for x in X]
            return fn

        def caption(state):
            xs, X = state[:nx], state[nx:]
            T = GF.caption(rt, X, self.r_text, self.Wg, self.W, n_tok, nl, tlin)
            return [rt.ops.copy(x) for x in xs] + [rt.ops.realize(t) for t in T]
        text = [(f"g{b}", gemma(b)) for b in range(self.text_layers)] + [("caption", caption)]
        return text + super().stages()
