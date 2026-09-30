import importlib

from .remote import EncClient, EncServer
from .serve import EncGenerationClient, EncGenerationServer

__all__ = [
    "EncAttention", "EncBert", "EncBlock", "EncClient", "EncCutMax", "EncGELU",
    "EncGenerationClient", "EncGenerationServer", "EncGPT2",
    "EncLMHead", "EncLayerNorm", "EncLinear", "EncModule", "EncModuleList", "EncSequential",
    "EncServer", "EncViT", "EncViTBlock", "Stage", "calibrate_sequential", "load", "run_stages",
    "save", "stage_of",
]

# name -> submodule of the lazily loaded surface (everything that needs perseus._core)
_LAZY = {
    "EncGELU": ".activation",
    "EncAttention": ".attention",
    "EncBert": ".bert",
    "EncBlock": ".block",
    "calibrate_sequential": ".calibrate",
    "EncModuleList": ".container",
    "EncSequential": ".container",
    "EncGPT2": ".gpt2",
    "EncCutMax": ".head",
    "EncLMHead": ".head",
    "EncLinear": ".linear",
    "EncModule": ".module",
    "EncLayerNorm": ".norm",
    "Stage": ".pipeline",
    "run_stages": ".pipeline",
    "stage_of": ".pipeline",
    "load": ".serialization",
    "save": ".serialization",
    "EncViT": ".vit",
    "EncViTBlock": ".vit",
}


def __getattr__(name):
    try:
        modname = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(modname, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
