from .. import _core
from .module import EncModule


class EncGELU(EncModule):
    """FHE GELU approximation; cfg_name selects the calibrated section."""

    def __init__(self, cfg_name="mlp.act"):
        super().__init__()
        self.cfg_name = cfg_name

    def to_config(self):
        return {"cfg_name": self.cfg_name}

    @classmethod
    def from_config(cls, cfg, params):
        return cls(cfg["cfg_name"])

    def extra_repr(self):
        return repr(self.cfg_name)

    def torch_mirror(self):
        import torch.nn as nn
        return nn.GELU(approximate="tanh")

    def forward(self, x):
        return _core.gelu_approx(self.inf, x, self.cfg_name)
