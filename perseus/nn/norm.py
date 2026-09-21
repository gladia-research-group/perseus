from .. import _core
from .module import EncModule


class EncLayerNorm(EncModule):
    """FHE LayerNorm approximation; cfg_name selects the calibrated section."""

    def __init__(self, cfg_name):
        super().__init__()
        self.cfg_name = cfg_name

    def forward(self, x):
        return _core.layer_norm(self.inf, x, self.cfg_name)
