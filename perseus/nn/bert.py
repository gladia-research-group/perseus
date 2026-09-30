from .. import _core
from .module import EncModule
from .vit import chunk_sizes


class EncBert(EncModule):
    """BERT encoder forward: chunked filling tokens -> the C++ bert driver -> CLS.

    Client-side (plaintext): word + position + token_type embeddings summed and
    passed through the embeddings LayerNorm.

    forward delegates to `_core.bert_forward`. Residency (weight streaming,
    eviction, host reclaim), staging, plan loading and graph capture live in the
    cuda residency pipeline.
    """

    def __init__(self, store, configs, n_layers=None):
        super().__init__()
        self.store = store
        self.configs = configs
        self.n_layers = n_layers if n_layers is not None else configs.model.n_layers

    @classmethod
    def from_pretrained(cls, name, *, tag="classic", weights=None, configs=None, n_layers=None,
                        strict=True):
        """Build from the artifacts `perseus-export` / `perseus-calibrate` produced for
        `name` (perseus cache or explicit paths; see perseus.nn.pretrained)."""
        from .pretrained import from_pretrained
        return from_pretrained(cls, name, tag=tag, weights=weights, configs=configs,
                               n_layers=n_layers, strict=strict, model_type="bert")

    def extra_repr(self):
        return f"n_layers={self.n_layers}"

    def encode_tokens(self, embeddings):
        """[T][d] plaintext rows -> (chunk cts, A-half counts, B-half counts).

        Token-pair (complex payload) packs 2*t tokens per ct: [0,t) on the Re
        lanes and [t,2t) on the Im lanes. On the real arm the B counts are zero.
        """
        inf = self.inf
        t = inf.slots // inf.size.hidDim
        cap = 2 * t if inf.token_pair else t
        xs, ns, ns_im, base = [], [], [], 0
        for n in chunk_sizes(len(embeddings), cap):
            inf.n_tok = min(n, t)
            xs.append(_core.encode_prefill_input(inf, embeddings[base:base + n]))
            ns.append(min(n, t))
            ns_im.append(max(0, n - t))
            base += n
        return xs, ns, ns_im

    def forward(self, xs, ns, ns_im=None):
        """outputs the CLS token ciphertext (cachemir-packed)"""
        return _core.bert_forward(self.inf, xs, ns, self.store, self.configs,
                                  self.n_layers, ns_im or [])

    def decode_cls(self, cls_ct):
        return _core.decode_token_output(self.inf, cls_ct)[:self.inf.size.dim]
