from .. import _core
from .module import EncModule


class EncAttention(EncModule):
    """Causal MHA over the KV cache: qkv -> cache push -> qkt -> THOR softmax -> P·V -> out proj.

    A GPT-2 mirror, not a composable layer: it reads the canonical production weight names
    ("kv"/"q"/"k"/"v"/"out") installed by the GPT-2 loader and mirrors the C++ mha_ops
    (step labels, hints, complex arm) so captures/plans bind.
    """

    def __init__(self, softmax_cfg="attn"):
        super().__init__()
        self.softmax_cfg = softmax_cfg

    def extra_repr(self):
        return f"softmax_cfg={self.softmax_cfg!r}"

    def forward(self, x):
        inf, fhe = self.inf, self.inf.fhe
        d = inf.size.hidDim

        with inf.step("qkv"):
            inf.name_ct_if_absent(x, "mha_block.x")
            fhe.bootstrap_hint(x, fhe.level_limit() - 1, True)
            fhe.level_hint(x, fhe.level_limit() - 1)
            if inf.complex:
                kv, q = _core.linear_multi(inf, x, ["kv", "q"], d, d)
                _core.cache_kv_push_packed(inf, kv)
                x = q
            else:
                k, v, q = _core.linear_multi(inf, x, ["k", "v", "q"], d, d)
                _core.cache_kv_push(inf, k, v)
                x = q

        with inf.step("attn_core"):
            fhe.level_hint(x, fhe.level_limit() - 3)
            scores = _core.qkt(inf, x)
            scores = _core.attention_softmax_thor(inf, scores, self.softmax_cfg)
            x = _core.softmax_v(inf, scores)

        with inf.step("out_proj"):
            fhe.bootstrap_hint(x, fhe.level_limit() - 1, True)
            fhe.level_hint(x, fhe.level_limit() - 1)
            x = _core.linear(inf, x, "out", d, d)

        return x
