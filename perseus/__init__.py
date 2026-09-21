"""perseus — GPU FHE inference for transformer LMs under CKKS.

Python side of the pipeline: HF model acquisition (`perseus.hub`), approximation
calibration (`perseus.calibrate`), weight packing (`perseus.export`), bootstrap planning
(`perseus.plan`), and the torch-style authoring surface (`perseus.nn`). The CUDA/C++
runtime is the `cuda_cachemir` binary; the same runtime is exposed to Python as the
`perseus._core` extension, built separately (README: Install). A GPU-less client machine
builds `perseus._client` instead (OpenFHE only, no CUDA): the client role — EncClient /
EncGenerationClient — runs on it and interchanges keys and ciphertexts with a _core server.

Heavy deps (torch/transformers/hydra) are imported inside submodules, not here.
"""

import logging as _logging

__version__ = "0.1.0"

_logging.getLogger(__name__).addHandler(_logging.NullHandler())

from ._backend import _CLIENT_HELP, _CORE_HELP  # noqa: E402, F401  (imports nothing from perseus)
from .session import Session, session  # noqa: E402  (imports no heavy deps)

__all__ = ["Session", "__version__", "session"]


def __getattr__(name):
    # `from perseus import _core` (what every perseus.nn module does) first asks
    # hasattr(perseus, "_core"), which lands here before the submodule is imported —
    # so import it ourselves, and turn a genuinely missing extension into the recipe.
    # `_client` is served the same way (its own recipe).
    if name in ("_core", "_client"):
        import importlib

        try:
            return importlib.import_module(f"perseus.{name}")
        except ModuleNotFoundError as e:
            if e.name == f"perseus.{name}":
                raise ImportError(_CORE_HELP if name == "_core" else _CLIENT_HELP) from None
            raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
