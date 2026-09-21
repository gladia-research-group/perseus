"""EncModule — the torch.nn.Module analogue for encrypted modules."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np


class EncModule:
    """Base for encrypted modules; children register on attribute assignment.

    The torch conventions that matter for authoring are mirrored: assigning an
    EncModule attribute registers a child, `bind(inf)` recurses, `named_modules()` /
    `modules()` / `apply()` walk the tree, `repr()` prints it, and calling the module
    forwards every positional and keyword argument to `forward`. Weights live on the
    session (`inf`) once bound, not on the module — see `residency()` for the hook the
    residency pipeline uses to stream them.
    """

    def __init__(self):
        object.__setattr__(self, "_children", {})
        object.__setattr__(self, "inf", None)

    def __setattr__(self, name, value):
        children = self.__dict__.get("_children")
        if children is None:
            raise AttributeError(
                f"{type(self).__name__}.__init__ must call super().__init__() before "
                f"assigning attributes (got {name!r})")
        if isinstance(value, EncModule):
            children[name] = value
        elif name in children:      # a child slot reassigned to a non-module leaves the tree
            del children[name]
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        self.__dict__.get("_children", {}).pop(name, None)
        object.__delattr__(self, name)

    def children(self):
        """Direct children (unnamed), in registration order."""
        return self._children.values()

    def named_children(self):
        """(name, child) pairs of the direct children, in registration order."""
        return self._children.items()

    def named_modules(self, prefix: str = ""):
        """(dotted name, module) for this module and every descendant, pre-order."""
        yield prefix, self
        for name, child in self._children.items():
            yield from child.named_modules(f"{prefix}.{name}" if prefix else name)

    def modules(self):
        """This module and every descendant, pre-order."""
        for _, m in self.named_modules():
            yield m

    def apply(self, fn: Callable[[EncModule], object]):
        """Call `fn(module)` on this module and every descendant; returns self."""
        for m in self.modules():
            fn(m)
        return self

    def bind(self, inf):
        """Attach a session (an `_core.Inference`) to this module and its children,
        installing any owned weights. Returns self."""
        inf = getattr(inf, "inf", inf)          # a perseus.Session is accepted too
        object.__setattr__(self, "inf", inf)
        for child in self._children.values():
            child.bind(inf)
        return self

    def unbind(self):
        """Detach the session from this module and its children (weights already
        installed on the session stay there). Returns self."""
        object.__setattr__(self, "inf", None)
        for child in self._children.values():
            child.unbind()
        return self

    @property
    def bound(self) -> bool:
        return self.inf is not None

    def residency(self):
        return None

    def torch_mirror(self):
        return None

    PARAM_NAMES = ("weight", "bias")

    def state_dict(self) -> dict[str, np.ndarray]:
        """Plaintext parameters of this tree, keyed torch style ("0.weight", "ln.bias").
        Modules whose weights are pre-installed on the session (shared names) or come
        from a WeightStore contribute nothing."""
        import numpy as np
        out = {}
        for prefix, m in self.named_modules():
            for p in self.PARAM_NAMES:
                v = m.__dict__.get(p)
                if isinstance(v, np.ndarray):
                    out[f"{prefix}.{p}" if prefix else p] = v
        return out

    def load_state_dict(self, state: dict[str, np.ndarray], strict: bool = True):
        """Inverse of state_dict(); shapes must match the existing parameters."""
        import numpy as np
        mods = dict(self.named_modules())
        missing, unexpected = [], []
        for key, v in state.items():
            prefix, _, p = key.rpartition(".")
            m = mods.get(prefix)
            cur = m.__dict__.get(p) if m is not None else None
            if m is None or p not in self.PARAM_NAMES or not isinstance(cur, np.ndarray):
                unexpected.append(key)
                continue
            v = np.asarray(v, dtype=np.float64)
            if v.shape != cur.shape:
                raise ValueError(f"load_state_dict: {key} has shape {v.shape}, expected {cur.shape}")
            object.__setattr__(m, p, v)
        for key in self.state_dict():
            if key not in state:
                missing.append(key)
        if strict and (missing or unexpected):
            raise KeyError(f"load_state_dict: missing {missing}, unexpected {unexpected}")
        return self

    def to_config(self) -> dict:
        """Constructor arguments as plain JSON (see perseus.nn.serialization). Modules
        built from a WeightStore/ParsedConfigs are not serializable this way."""
        raise NotImplementedError(f"{type(self).__name__} cannot be rebuilt from a config")

    CALIBRATION_SETTERS = {"softgelu": "set_gelu_cfg", "norm": "set_norm_cfg",
                           "softmax": "set_softmax_cfg"}

    def apply_calibration(self, section: str, cfg, probe=None) -> None:
        """Install one fitted approximation section on the bound session under this
        module's cfg_name. The default covers every section the runtime keys by name;
        modules with extra state to close (EncLayerNorm's affine scale) override."""
        cfg_name = getattr(self, "cfg_name", None)
        if cfg_name is None:
            raise TypeError(f"{type(self).__name__} carries no calibrated section "
                            f"(no cfg_name); nothing to apply")
        if section not in self.CALIBRATION_SETTERS:
            raise ValueError(f"unknown calibration section {section!r}; expected one of "
                             f"{sorted(self.CALIBRATION_SETTERS)}")
        getattr(self.inf, self.CALIBRATION_SETTERS[section])(cfg_name, cfg)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.forward")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        name = type(self).__name__
        extra = self.extra_repr()
        if not self._children:
            return f"{name}({extra})"
        lines = [extra] if extra else []
        for cname, child in self._children.items():
            lines.append(f"({cname}): " + repr(child).replace("\n", "\n  "))
        return f"{name}(\n  " + "\n  ".join(lines) + "\n)"
