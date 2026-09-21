"""The approximation registry — the framework's extension point.

An `Approximation` bundles everything the engine needs to support one
FHE-approximable operation kind:

    matches(module)      site detector over the model's module tree
    make_collector(cfg)  one collector per kind; hooks its sites, gathers samples
    fit_section(...)     collected samples -> one configs.json section

A model is calibratable iff every nonlinearity it contains is matched by a
registered approximation — `discovery.discover` enforces that and reports
precisely what is missing otherwise. Registering a new `Approximation` is all
it takes to support a new op kind; the engine, discovery, and CLI never change.

A registered kind PARTICIPATES in a run iff the selected approximation set
(`configs/approximation/*.yaml`) contains a subtree named after it; the `cfg`
handed to `make_collector` / `fit_section` is that subtree, nothing else.

Model-level approximations (`model_level=True`, e.g. the cutmax argmax
schedule) attach to the model's outputs instead of matched sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

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
