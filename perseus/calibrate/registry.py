from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch.nn as nn


@runtime_checkable
class Collector(Protocol):
    """Gathers calibration samples for one approximation kind."""

    def attach(self, name: str, module: nn.Module) -> list:
        """Hook one matched site; return the removable hook handles."""
        ...

    def on_output(self, output: Any) -> None:
        """Observe one batch's model output (model-level collectors)."""
        ...

    def finalize(self) -> Any:
        """Return the collected samples, ready for `fit_section`."""
        ...


@dataclass(frozen=True)
class Approximation:
    kind: str                                       # e.g. "gelu", "softmax"
    section: str                                    # configs.json section it emits
    matches: Callable[[nn.Module], bool]
    make_collector: Callable[[Any], Collector]      # kind subtree -> Collector
    fit_section: Callable[[Any, Any], dict]         # (collected, subtree) -> section
    model_level: bool = False                       # attach to model outputs, not sites


_REGISTRY: dict[str, Approximation] = {}


def register(approx: Approximation) -> Approximation:
    if approx.kind in _REGISTRY:
        raise ValueError(f"approximation kind {approx.kind!r} already registered")
    _REGISTRY[approx.kind] = approx
    return approx


def registered() -> dict[str, Approximation]:
    return dict(_REGISTRY)
