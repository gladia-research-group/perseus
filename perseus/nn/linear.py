import numpy as np

from .. import _core
from .module import EncModule


def _check_weight(name, weight, d_in, d_out):
    W = np.asarray(weight, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError(f"EncLinear({name!r}): weight must be a 2-D (d_in, d_out) = "
                         f"({d_in}, {d_out}) matrix; got shape {W.shape}")
    if W.shape != (d_in, d_out):
        hint = ""
        if W.shape == (d_out, d_in):
            hint = (" — that is torch's (out_features, in_features) layout; perseus computes "
                    "y = x @ W with W stored as (d_in, d_out): pass weight.T, or use "
                    "EncLinear.from_torch(...)")
        raise ValueError(f"EncLinear({name!r}): weight shape {W.shape} != (d_in, d_out) = "
                         f"({d_in}, {d_out}){hint}")
    if not np.isfinite(W).all():
        raise ValueError(f"EncLinear({name!r}): weight contains NaN/inf")
    return W


def _check_bias(name, bias, d_out):
    b = np.asarray(bias, dtype=np.float64)
    if b.shape != (d_out,):
        raise ValueError(f"EncLinear({name!r}): bias must have shape (d_out,) = ({d_out},); "
                         f"got {b.shape}")
    if not np.isfinite(b).all():
        raise ValueError(f"EncLinear({name!r}): bias contains NaN/inf")
    return b


def _pad(arr, shape):
    if arr.shape == shape:
        return arr
    out = np.zeros(shape, dtype=arr.dtype)
    out[tuple(slice(0, n) for n in arr.shape)] = arr
    return out


class EncLinear(EncModule):
    """Packed linear over a pre-installed weight name; optional own weight/bias at bind.

    y = x @ W (+ bias) with W of shape (d_in, d_out) — the transpose of torch's
    nn.Linear.weight; `EncLinear.from_torch` converts (and zero-pads to the packed
    widths). d_in / d_out are the PACKED widths (powers of two: 768 real features
    ride a 1024-wide lane); pad the real values with zeros.

    Refreshes the input first (bootstrap/level hints, as every production linear op
    does): an eager bootstrap firing mid-linear amplifies the noise floor.
    bias installs as "<name>_bias" — the packed linear applies it automatically,
    the same path the canonical q/out/up/down biases ride. weight=None means the
    name is pre-installed on the session (a shared production weight): nothing is
    installed or evicted by this module.
    """

    def __init__(self, name: str, d_in: int, d_out: int, weight=None, bias=None, hint: bool = True):
        super().__init__()
        self.name = name
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.weight = None if weight is None else _check_weight(name, weight, self.d_in, self.d_out)
        self.bias = None if bias is None else _check_bias(name, bias, self.d_out)
        self.hint = hint

    @classmethod
    def from_torch(cls, name: str, linear, d_in: int | None = None, d_out: int | None = None,
                   hint: bool = True) -> "EncLinear":
        """Build from a torch.nn.Linear: transposes to (d_in, d_out) and zero-pads to
        the packed widths (default: the layer's own in/out features)."""
        W = linear.weight.detach().cpu().numpy().astype(np.float64).T
        d_in = int(d_in if d_in is not None else W.shape[0])
        d_out = int(d_out if d_out is not None else W.shape[1])
        if W.shape[0] > d_in or W.shape[1] > d_out:
            raise ValueError(f"EncLinear.from_torch({name!r}): layer is {W.shape}, larger than "
                             f"the packed (d_in, d_out) = ({d_in}, {d_out})")
        bias = None
        if linear.bias is not None:
            bias = _pad(linear.bias.detach().cpu().numpy().astype(np.float64), (d_out,))
        return cls(name, d_in, d_out, weight=_pad(W, (d_in, d_out)), bias=bias, hint=hint)

    def to_config(self):
        return {"name": self.name, "d_in": self.d_in, "d_out": self.d_out, "hint": self.hint,
                "has_weight": self.weight is not None, "has_bias": self.bias is not None}

    @classmethod
    def from_config(cls, cfg, params):
        return cls(cfg["name"], cfg["d_in"], cfg["d_out"], hint=cfg.get("hint", True),
                   weight=params.get("weight") if cfg.get("has_weight") else None,
                   bias=params.get("bias") if cfg.get("has_bias") else None)

    def extra_repr(self):
        s = f"{self.name!r}, {self.d_in}->{self.d_out}"
        if self.weight is None:
            return s + ", shared"
        return s + f", bias={self.bias is not None}"

    def bind(self, inf):
        super().bind(inf)
        if self.weight is not None:      # ndarray: the binding's buffer fast path
            inf.set_weight(self.name, self.weight, self.d_in, self.d_out)
        if self.bias is not None:
            inf.set_bias(f"{self.name}_bias", self.bias, self.d_in, self.d_out)
        return self

    def residency(self):
        if self.weight is None:
            return None
        keys = [self.name]
        if self.bias is not None:
            keys.append(f"{self.name}_bias")
        return keys

    def torch_mirror(self):
        if self.weight is None:
            return None   # pre-installed shared name: values not visible here
        import torch
        import torch.nn as nn
        m = nn.Linear(self.d_in, self.d_out, bias=self.bias is not None)
        with torch.no_grad():
            # our convention is y = x @ W with W (d_in, d_out); torch stores W.T
            m.weight.copy_(torch.as_tensor(self.weight.T, dtype=torch.float32))
            if self.bias is not None:
                m.bias.copy_(torch.as_tensor(self.bias, dtype=torch.float32))
        return m

    def forward(self, x):
        if self.hint:
            fhe = self.inf.fhe
            fhe.bootstrap_hint(x, fhe.level_limit() - 1, True)
            fhe.level_hint(x, fhe.level_limit() - 1)
        return _core.linear(self.inf, x, self.name, self.d_in, self.d_out)
