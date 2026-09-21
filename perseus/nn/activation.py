from .. import _core
from .module import EncModule


class EncGELU(EncModule):
    """FHE GELU approximation; cfg_name selects the calibrated section."""

    def __init__(self, cfg_name="mlp.act"):
        super().__init__()
        self.cfg_name = cfg_name

    def forward(self, x):
        return _core.gelu_approx(self.inf, x, self.cfg_name)
