"""The whole text-to-image forward encrypted: SanaImpl with the text side on the server. The client's ciphertexts are
the model input's and the carry's rows and the prompt's Gemma embedding rows (E[id] sqrt(2304), the n_tok tokens on the
real lanes of one chunk); the 26 Gemma-2 stages run the general encoder (examples/llama/encoder.py: the general blocked
prefill at 16 lanes, the instruction prefix's keys / values plaintext), then the caption stage (text.caption: final
norm, caption_projection, caption_norm) produces SANA's text rows T, then SanaImpl's steps."""
from __future__ import annotations

import os

import numpy as np

from perseus.examples.llama import weights as LW
from perseus.examples.llama.config import load_configs as llama_configs
from perseus.examples.llama.encoder import Encoder

from . import hf
from . import text as TX
from .impl import SanaImpl
from .layout import G

# the encoder's Gemma-2 configs (SANA_TEXT_CONFIGS): calibrate.py --first 207 on instruction + prompt windows, tune.py,
# the single SiLU series baked (the softmax design covers the prefix's 207 keys plus the prompt's)
TEXT_CONFIGS = os.environ.get("SANA_TEXT_CONFIGS", "configs/model/approximation/gemma2_2b_it_t2i/configs.json")


def text_store():
    """SANA's Gemma-2-2B-IT (the pipeline's text_encoder) as the general weight store."""
    return LW.SafetensorsStore(os.path.join(hf.snapshot(), "text_encoder"))


class SanaTextImpl(SanaImpl):
    def __init__(self, inf, W, nl, Wg=None, *, n_pub, prefix_ids, core=None, n_layers=None, profile=False,
                 text_layers=None, text_configs=None):
        super().__init__(inf, W, nl, core=core, n_layers=n_layers, profile=profile)
        from perseus._paths import resolve
        self.n_pub = n_pub
        store = text_store()
        path = resolve(text_configs or TEXT_CONFIGS)
        cfgs = llama_configs(path)
        dims = LW.llama_dims(store.config)
        self.lw = LW.LlamaWeights(store, cfgs, path, dims, "cachemir_complex", pinned=None)
        n = int(store.config["num_hidden_layers"])
        self.text_layers = n if text_layers is None else text_layers      # fewer: a debug cut
        self.enc = Encoder(inf, self.lw, cfgs, prefix_ids, core=core, n_layers=self.text_layers)
        assert self.enc.P == n_pub, (self.enc.P, n_pub)
        self.Wg = {"norm.weight": store.tensor("model.norm.weight")}     # the caption's final norm

    def embed(self, prompt_ids):
        """The client's embedding rows of the prompt ids (E[id] sqrt(d))."""
        return np.stack([self.lw.embedding(int(t)) for t in prompt_ids])

    def set_prompt_ids(self, x_in, prompt_ids, txt_bos, z2=None):
        """The prompt by its token ids after the public prefix (the client embeds them)."""
        self.emb_rows = self.embed(prompt_ids)
        self.set_prompt(x_in, np.zeros((len(prompt_ids), 1152)), txt_bos, z2)

    def input_values(self, x_in=None, txt_rows=None, z2=None):
        """The model input's and the carry's rows (2 G ciphertexts), then the prompt's embedding rows (the encoder's
        Kh blocks)."""
        noise = super().input_values(x_in, np.zeros((len(self.emb_rows), 1152)), z2)[:2 * G]
        return noise + TX.encode(self.emb_rows, self.enc.blocks.Kh)

    def text_linears(self):
        """The encoder's PCMM linears (the session carries them beside SANA's)."""
        return {lin.name: lin for lin in self.enc.linears(len(self.emb_rows))}

    def stages(self):
        rt, nl = self.rt, self.nl
        n_tok = len(self.emb_rows)
        nx = 2 * G                                    # the model input's and the carry's ciphertexts ride ahead

        def tlin(rt_, X, L):                                     # the prompt's real lanes only (bias included)
            return self.lin(rt_, X, L, (n_tok, 0))

        def gemma(b):
            def fn(state):
                xs, X = state[:nx], state[nx:]
                X = self.enc.block(X, b, n_tok)
                return [rt.ops.copy(x) for x in xs] + [rt.ops.realize(x) for x in X]
            return fn

        def caption(state):
            xs, X = state[:nx], state[nx:]
            T = TX.caption(rt, X, None, self.Wg, self.W, n_tok, nl, tlin)
            return [rt.ops.copy(x) for x in xs] + [rt.ops.realize(t) for t in T]
        text = [(f"g{b}", gemma(b)) for b in range(self.text_layers)] + [("caption", caption)]
        return text + super().stages()
