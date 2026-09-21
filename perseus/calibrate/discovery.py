"""Model discovery: classify any torch module tree against the registry.

`discover(model, approximations)` walks `named_modules()` once and assigns
every matched module to its approximation kind (keeping only leaf matches, so
wrapper classes never shadow the module that actually computes). It then
verifies COVERAGE: any remaining activation-family nonlinearity is unsupported
and aborts calibration with a precise report — a model either calibrates fully
or not at all.

Works for anything from an `nn.Sequential` to a hub transformer; the optional
HF helpers (`infer_dims`, `infer_block_size`) read the transformers config
when one exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from perseus.calibrate.registry import Approximation

_NONLINEAR_MODULES = ("torch.nn.modules.activation", "transformers.activations")


class UnsupportedModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class Site:
    name: str
    kind: str


@dataclass
class ModelMap:
    model_name: str
    sites: list[Site]
    dims: dict | None          # transformer dims when inferable (HF config)
    block_size: int | None     # context length when inferable

    def by_kind(self, kind: str) -> list[Site]:
        return [s for s in self.sites if s.kind == kind]


def _leaf_matches(model: nn.Module, matches) -> list[str]:
    hits = [name for name, m in model.named_modules() if matches(m)]
    return [n for n in hits if not any(o != n and o.startswith(n + ".") for o in hits)]


def _unmatched_nonlinearities(model: nn.Module, matched: set[str]) -> list[str]:
    out = []
    for name, m in model.named_modules():
        if type(m).__module__ in _NONLINEAR_MODULES and not any(
            name == s or name.startswith(s + ".") or s.startswith(name + ".")
            for s in matched
        ):
            out.append(f"{name} ({type(m).__name__})")
    return out


def infer_dims(model) -> dict | None:
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "hidden_size"):
        return None
    n_embd = config.hidden_size
    n_inner = getattr(config, "n_inner", None) or getattr(config, "intermediate_size", None)
    return {
        "n_layers": config.num_hidden_layers,
        "n_embd":   n_embd,
        "n_head":   config.num_attention_heads,
        "n_inner":  n_inner if n_inner is not None else 4 * n_embd,
    }


def infer_block_size(model) -> int | None:
    config = getattr(model, "config", None)
    for attr in ("n_positions", "max_position_embeddings"):
        v = getattr(config, attr, None)
        if v:
            return int(v)
    return None


def discover(model: nn.Module, approximations: list[Approximation]) -> ModelMap:
    sites: list[Site] = []
    for approx in approximations:
        if approx.model_level:
            continue
        for name in _leaf_matches(model, approx.matches):
            sites.append(Site(name, approx.kind))

    unmatched = _unmatched_nonlinearities(model, {s.name for s in sites})
    if unmatched:
        supported = ", ".join(a.kind for a in approximations)
        raise UnsupportedModelError(
            "model contains nonlinearities with no registered FHE approximation:\n  "
            + "\n  ".join(unmatched)
            + f"\nsupported kinds: {supported}")

    config = getattr(model, "config", None)
    model_name = getattr(config, "name_or_path", None) or type(model).__name__
    mm = ModelMap(
        model_name=model_name,
        sites=sites,
        dims=infer_dims(model),
        block_size=infer_block_size(model),
    )

    counts = {k: len(mm.by_kind(k)) for k in sorted({s.kind for s in sites})}
    dims = f" dims={mm.dims}" if mm.dims else ""
    print(f"[discover] {model_name}: sites={counts}{dims} ctx={mm.block_size}")
    if not sites:
        raise UnsupportedModelError("discovery matched no approximation sites — "
                                    "is this a supported model?")
    return mm
