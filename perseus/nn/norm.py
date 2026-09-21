import logging

import numpy as np

from .. import _core
from .module import EncModule

log = logging.getLogger(__name__)


def _check_scale(cfg_name, scale):
    if scale is None:
        return None
    try:
        r_g, r_b = (float(v) for v in scale)
    except (TypeError, ValueError):
        raise ValueError(f"EncLayerNorm({cfg_name!r}): scale must be a (r_g, r_b) pair, "
                         f"got {scale!r}") from None
    if not (np.isfinite(r_g) and np.isfinite(r_b)) or r_g == 0 or r_b == 0:
        raise ValueError(f"EncLayerNorm({cfg_name!r}): scale entries must be finite and "
                         f"non-zero, got {scale!r}")
    return (r_g, r_b)


class EncLayerNorm(EncModule):
    """FHE LayerNorm approximation; cfg_name selects the calibrated section.

    d: normalized width (needed by torch_mirror). weight/bias: the affine gamma/beta
    (length d, real features first); installed at bind as <cfg_name>.weight/.bias —
    the tiles ln_affine applies. A custom cfg_name never folds, so both are required
    for the affine to run; omit both for a plain normalize.
    """

    def __init__(self, cfg_name: str, d: int | None = None, weight=None, bias=None,
                 scale=None):
        super().__init__()
        self.cfg_name = cfg_name
        self._scale = _check_scale(cfg_name, scale)
        if (weight is None) != (bias is None):
            raise ValueError(f"EncLayerNorm({cfg_name!r}): weight and bias come together")
        if weight is not None:
            weight = np.asarray(weight, dtype=np.float64)
            bias = np.asarray(bias, dtype=np.float64)
            if weight.ndim != 1 or bias.shape != weight.shape:
                raise ValueError(f"EncLayerNorm({cfg_name!r}): weight and bias must be 1-D "
                                 f"of the same length; got {weight.shape} and {bias.shape}")
            if d is None:
                d = weight.shape[0]
            elif weight.shape[0] != d:
                raise ValueError(f"EncLayerNorm({cfg_name!r}): weight has {weight.shape[0]} "
                                 f"entries but d={d}")
            if not (np.isfinite(weight).all() and np.isfinite(bias).all()):
                raise ValueError(f"EncLayerNorm({cfg_name!r}): weight/bias contain NaN/inf")
        self.d = None if d is None else int(d)
        self.weight = weight
        self.bias = bias

    @property
    def scale(self):
        """(r_g, r_b): the measured gamma/beta-path output scales the affine tiles are
        descaled by, or None until fit_scale() ran or scale= was given."""
        return self._scale

    def set_scale(self, scale):
        """Install a previously fitted (r_g, r_b) — what the key-holding calibration
        session measured — so this module can be bound on a session that cannot decrypt
        (an EncServer). Re-installs the affine when already bound. Returns self."""
        self._scale = _check_scale(self.cfg_name, scale)
        if self.inf is not None and self.weight is not None:
            self._install_affine(self.inf, *self._scale)
        return self

    def to_config(self):
        return {"cfg_name": self.cfg_name, "d": self.d, "has_affine": self.weight is not None,
                "scale": None if self._scale is None else list(self._scale)}

    @classmethod
    def from_config(cls, cfg, params):
        scale = cfg.get("scale")
        if cfg.get("has_affine"):
            return cls(cfg["cfg_name"], cfg["d"], weight=params["weight"], bias=params["bias"],
                       scale=scale)
        return cls(cfg["cfg_name"], cfg["d"], scale=scale)

    def extra_repr(self):
        s = f"{self.cfg_name!r}, d={self.d}"
        if self.weight is not None:
            s += ", affine=True"
        if self._scale is not None:
            s += f", scale=({self._scale[0]:.4g}, {self._scale[1]:.4g})"
        return s

    def bind(self, inf):
        super().bind(inf)
        if self.weight is not None:
            if self._scale is not None:
                self._install_affine(self.inf, *self._scale)
            else:
                self._install_affine(self.inf, 1.0)
        return self

    def _install_affine(self, inf, scale_g=1.0, scale_b=1.0, gamma=None, bias=None):
        d = self.d or (len(self.weight) if self.weight is not None else 0)
        if gamma is None:
            gamma = self.weight.tolist() if self.weight is not None else [1.0] * d
        if bias is None:
            bias = self.bias.tolist() if self.bias is not None else [0.0] * d
        _core.set_ln_affine(inf, self.cfg_name,
                            [g / scale_g for g in gamma], [b / scale_b for b in bias])

    def apply_calibration(self, section, cfg, probe=None):
        if section == "norm":
            return self.apply_cfg(cfg, probe=probe)
        return super().apply_calibration(section, cfg, probe=probe)

    def fit_scale(self, inf=None, probe=None):
        import numpy as np
        inf = inf or self.inf
        d = self.d
        if d is None:
            raise ValueError("EncLayerNorm.fit_scale needs d (the normalized width)")
        x = (np.random.default_rng(0).standard_normal(d) * 0.5
             if probe is None else np.asarray(probe, dtype=np.float64)[:d])
        ln = (x - x.mean()) / np.sqrt(x.var() + 1e-5)
        gamma = np.asarray(self.weight, np.float64) if self.weight is not None else np.ones(d)
        beta = np.asarray(self.bias, np.float64) if self.bias is not None else np.zeros(d)

        def run():
            pc = _core.encode_token_input(inf, x.tolist())
            inf.fhe.bootstrap_hint(pc, int(inf.fhe.bootstrap_output_level()) + 2)
            return np.array(_core.decode_token_output(
                inf, _core.layer_norm(inf, pc, self.cfg_name)))[:d]

        self._install_affine(inf, gamma=gamma.tolist(), bias=[0.0] * d)
        y_g = run()
        ref_g = ln * gamma
        m = np.abs(ref_g) > 0.2 * np.abs(ref_g).max()
        r_g = float(np.median(y_g[m] / ref_g[m]))
        r_g2 = float(np.median(run()[m] / ref_g[m]))
        if abs(r_g2 - r_g) > 0.02 * max(abs(r_g), abs(r_g2), 1e-9):
            raise RuntimeError(
                f"EncLayerNorm.fit_scale({self.cfg_name}): the norm chain is not "
                f"run-to-run repeatable on this chain (r_g={r_g:.4g} then "
                f"{r_g2:.4g}) — the standalone inv-sqrt is chaotic at this "
                f"precision; use the block-context LN (folded gamma) or a plan-"
                f"managed chain. See docs/PYTHON_BINDING_STATUS.md (LN, n32).")
        r_b = 1.0
        if np.abs(beta).max() > 0:
            self._install_affine(inf, gamma=gamma.tolist(), bias=beta.tolist())
            m = np.abs(beta) > 0.2 * np.abs(beta).max()
            r_b = float(np.median((run() - y_g)[m] / beta[m]))
        if not (0.01 < abs(r_g) < 100.0) or not (0.01 < abs(r_b) < 100.0):
            raise RuntimeError(f"EncLayerNorm.fit_scale({self.cfg_name}): measured "
                               f"scales out of range (r_g={r_g:.4g}, r_b={r_b:.4g}) — "
                               f"probe off-distribution or the norm cfg is unevaluable")
        log.info(f"[fit_scale] {self.cfg_name}: r_g={r_g:.5g} r_b={r_b:.5g}")
        self._scale = (r_g, r_b)
        self._install_affine(inf, r_g, r_b)
        return self._scale

    def apply_cfg(self, cfg, probe=None, refit=None):
        self.inf.set_norm_cfg(self.cfg_name, cfg)
        if refit is None:
            refit = self._scale is None
        if refit:
            self.fit_scale(self.inf, probe=probe)
        elif self._scale is None:
            raise ValueError(f"EncLayerNorm({self.cfg_name!r}).apply_cfg(refit=False): no "
                             f"scale is known; pass scale= or call fit_scale on a key-holding session")
        elif self.weight is not None:
            self._install_affine(self.inf, *self._scale)

    def residency(self):
        if self.weight is None:
            return None
        return [f"{self.cfg_name}.weight", f"{self.cfg_name}.bias"]

    def torch_mirror(self):
        if self.d is None:
            return None
        import numpy as np
        import torch
        import torch.nn as nn

        class _PaddedLN(nn.Module):
            """LayerNorm over the first d features of a padded row; the pad stays."""

            def __init__(self, d, ln):
                super().__init__()
                self.d, self.ln = d, ln   # the nn.LayerNorm leaf is what calibration attaches to

            def forward(self, x):
                if x.shape[-1] == self.d:
                    return self.ln(x)
                return torch.cat([self.ln(x[..., :self.d]), x[..., self.d:]], dim=-1)

        ln = nn.LayerNorm(self.d)
        if self.weight is not None:
            with torch.no_grad():
                ln.weight.copy_(torch.as_tensor(np.asarray(self.weight, dtype=np.float64)[:self.d],
                                                dtype=torch.float32))
                ln.bias.copy_(torch.as_tensor(np.asarray(self.bias, dtype=np.float64)[:self.d],
                                              dtype=torch.float32))
        return _PaddedLN(self.d, ln)

    def forward(self, x):
        self.inf.fhe.bootstrap_hint(x, int(self.inf.fhe.bootstrap_output_level()) + 2)
        return _core.layer_norm(self.inf, x, self.cfg_name)
