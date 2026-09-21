from .. import _core
from .module import EncModule


class EncLMHead(EncModule):
    """Tiled vocab head over the tied wte weight; returns encrypted logit tiles."""

    def __init__(self, store, plan=None):
        super().__init__()
        self.store = store
        self.plan = plan if plan is not None else _core.BootstrapPlan()
        self.vocab = _core.lm_head_vocab(store)
        self._cache = _core.LMHeadCache()

    def forward(self, x):
        return _core.lm_head(self.inf, x, self.store, self.vocab, self.plan, self._cache)

    def decode_logits(self, tiles):
        return _core.decode_lm_head_logits(self.inf, tiles, self.vocab)


class EncCutMax(EncModule):
    """Encrypted argmax over logit tiles (CutMax); returns one-hot Z tiles."""

    def __init__(self, vocab, config=None):
        super().__init__()
        self.vocab = vocab
        self.config = config if config is not None else _core.default_cutmax_config()

    @classmethod
    def from_configs(cls, vocab, parsed):
        cfg = (_core.cutmax_config_from_calib(parsed.cutmax) if parsed.has_cutmax
               else _core.default_cutmax_config())
        return cls(vocab, cfg)

    def forward(self, tiles):
        return _core.cutmax_argmax(self.inf, tiles, self.vocab, self.config)
