from .. import _core
from .module import EncModule


class EncLinear(EncModule):
    """Packed linear over a pre-installed weight name; optional own weight at bind.

    Refreshes the input first (bootstrap/level hints, as every production linear op
    does): an eager bootstrap firing mid-linear amplifies the noise floor.
    """

    def __init__(self, name, d_in, d_out, weight=None, hint=True):
        super().__init__()
        self.name = name
        self.d_in = d_in
        self.d_out = d_out
        self.weight = weight
        self.hint = hint

    def bind(self, inf):
        super().bind(inf)
        if self.weight is not None:
            inf.set_weight(self.name, self.weight, self.d_in, self.d_out)
        return self

    def forward(self, x):
        if self.hint:
            fhe = self.inf.fhe
            fhe.bootstrap_hint(x, fhe.level_limit() - 1, True)
            fhe.level_hint(x, fhe.level_limit() - 1)
        return _core.linear(self.inf, x, self.name, self.d_in, self.d_out)
