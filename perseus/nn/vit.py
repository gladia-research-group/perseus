from .. import _core
from .head import EncLMHead
from .module import EncModule


def chunk_sizes(T, t_stride):
    return [min(t_stride, T - base) for base in range(0, T, t_stride)]


class EncViTBlock(EncModule):
    """ViT encoder block on the filling packing; bidirectional two-phase MHA.

    forward_chunks takes the per-chunk ct list. Phase 1 runs ln_1 + QKV + K/V push
    for EVERY chunk before any attention (the inf.bidirectional schedule attends all
    T keys); phase 2 runs attention + out-proj + MLP per chunk.
    """

    def __init__(self, softmax_cfg="attn"):
        super().__init__()
        self.softmax_cfg = softmax_cfg

    def forward_chunks(self, xs, ns):
        inf, fhe = self.inf, self.inf.fhe
        d = inf.size.hidDim
        _core.prepare_mha_masks(inf)
        _core.prepare_vcache(inf)

        skips, qs = [], []
        for x, n in zip(xs, ns):
            inf.n_tok = n
            skips.append(x)
            with inf.step("ln_1"):
                x = _core.layer_norm(inf, x, "ln_1")
            with inf.step("qkv"):
                fhe.bootstrap_hint(x, fhe.level_limit() - 1, True)
                fhe.level_hint(x, fhe.level_limit() - 1)
                k, v, q = _core.linear_multi(inf, x, ["k", "v", "q"], d, d, stream_pt=True)
                _core.cache_kv_push(inf, k, v)
            qs.append(q)

        out = []
        for skip, q, n in zip(skips, qs, ns):
            inf.n_tok = n
            with inf.step("attn_core"):
                fhe.level_hint(q, fhe.level_limit() - 3)
                s = _core.qkt(inf, q)
                s = _core.attention_softmax_thor(inf, s, self.softmax_cfg)
                x = _core.softmax_v(inf, s)
            with inf.step("out_proj"):
                fhe.bootstrap_hint(x, fhe.level_limit() - 1, True)
                fhe.level_hint(x, fhe.level_limit() - 1)
                x = _core.linear(inf, x, "out", d, d, stream_pt=True)
            with inf.step("attn_residual"):
                x = fhe.add(x, skip)
                skip = x
            with inf.step("ln_2"):
                x = _core.layer_norm(inf, x, "ln_2")
            x = _core.mlp_block(inf, x)   # canonical MLP: tiled/plain dispatch
            with inf.step("mlp_residual"):
                x = fhe.add(x, skip)
            out.append(x)
        return out


class EncViT(EncModule):
    """ViT forward: chunked filling tokens -> the C++ encoder driver -> CLS logits.

    Client-side (plaintext): patch embed + position embeddings + CLS token, then
    encode_prefill_input per chunk. forward delegates to the C++ vit driver —
    residency (weight streaming, eviction, host reclaim) lives there, mirroring
    the GPT-2 standard. EncViTBlock above is the fine-grained authoring/capture
    mirror of the C++ encoder block, not the production path.
    """

    def __init__(self, store, configs, n_layers=None):
        super().__init__()
        self.store = store
        self.configs = configs
        self.n_layers = n_layers if n_layers is not None else configs.model.n_layers
        self.block = EncViTBlock()
        self.head = EncLMHead(store)   # decode_logits over the classifier tiles

    def bind(self, inf):
        super().bind(inf)
        inf.bidirectional = True
        return self

    def encode_tokens(self, embeddings):
        """-> (chunk cts, A-half counts, B-half counts).

        Token-pair (complex payload): one ct carries up to 2*t tokens — tokens
        [0,t) on the Re lanes, [t,2t) on the Im lanes. Real arm: t per ct, and
        the B counts come back all-zero.
        """
        inf = self.inf
        t = inf.slots // inf.size.hidDim
        cap = 2 * t if inf.token_pair else t
        sizes = chunk_sizes(len(embeddings), cap)
        xs, ns, ns_im, base = [], [], [], 0
        for n in sizes:
            inf.n_tok = min(n, t)
            xs.append(_core.encode_prefill_input(inf, embeddings[base:base + n]))
            ns.append(min(n, t))
            ns_im.append(max(0, n - t))
            base += n
        return xs, ns, ns_im

    def forward(self, xs, ns, ns_im=None):
        return _core.vit_forward(self.inf, xs, ns, self.store, self.configs,
                                 self.n_layers, ns_im or [])

    def decode_logits(self, tiles):
        return self.head.decode_logits(tiles)
