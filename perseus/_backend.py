from __future__ import annotations

import importlib
import logging
import os
import types

log = logging.getLogger(__name__)

_CORE_HELP = (
    "perseus._core (the CUDA extension) is not built for this interpreter.\n"
    "Everything under perseus.nn needs it; perseus.plan, perseus.hub, perseus.export and\n"
    "perseus.calibrate do not.\n"
    "Build it into the checkout (needs the FIDESlib deps tree, see README 'Install'):\n"
    "    CHAIN=n32 bash scripts/local_build_core.sh\n"
    "or configure CMake yourself with -DCACHEMIR_BUILD_PYTHON=ON and build the `_core` target.\n"
    "GPU-less client machine: CHAIN=n32 bash scripts/local_build_client.sh builds perseus._client\n"
    "(EncClient / EncGenerationClient only; needs the patched OpenFHE of the deps tree, no CUDA)."
)

_CLIENT_HELP = (
    "perseus._client (the CUDA-free client extension) is not built for this interpreter.\n"
    "Build it into the checkout (needs the patched OpenFHE of the deps tree, no CUDA toolchain):\n"
    "    CHAIN=n32 bash scripts/local_build_client.sh\n"
    "or configure CMake with -DCACHEMIR_CLIENT_ONLY=ON and build the `_client` target."
)

#: overlap name -> InferenceMode member name (the one table both extensions resolve against)
_MODE_NAMES = {
    "sync": "Sync",
    "prefetch": "Prefetch",
    "stream": "Prefetch",
    "cached": "Prefetch",   # legacy name for the Stream overlap
    "threaded": "Threaded",
}


def load(name: str) -> types.ModuleType | None:
    """Import ``perseus.<name>`` (``"_core"`` | ``"_client"``); None when that extension is
    not built (or cannot be loaded on this machine, e.g. libcuda missing for _core); any
    other import failure propagates."""
    modname = f"perseus.{name}"
    try:
        return importlib.import_module(modname)
    except ImportError as e:
        if getattr(e, "name", None) in (modname, name):
            if not isinstance(e, ModuleNotFoundError):
                log.debug("%s is present but cannot be loaded here: %s", modname, e)
            return None
        raise


def default() -> types.ModuleType:
    """The extension the package runs on: _core when it imports, else _client; ImportError
    with the build recipe when neither is built."""
    mod = load("_core") or load("_client")
    if mod is None:
        raise ImportError(_CORE_HELP)
    return mod


def client_extension(choice: str | None = None) -> types.ModuleType:
    """The extension backing the client role. ``choice``: ``"core"`` | ``"client"`` forces
    one (ImportError with its recipe when it is not built); None reads
    ``PERSEUS_CLIENT_EXTENSION`` (same values) and otherwise falls back to ``default()``."""
    if choice is None:
        choice = os.environ.get("PERSEUS_CLIENT_EXTENSION") or None
    if choice is None:
        return default()
    if choice not in ("core", "client"):
        raise ValueError(f"backend must be None, 'core' or 'client', got {choice!r}")
    mod = load(f"_{choice}")
    if mod is None:
        raise ImportError(_CORE_HELP if choice == "core" else _CLIENT_HELP)
    return mod


def is_client(mod: types.ModuleType) -> bool:
    """True when ``mod`` is the CUDA-free client extension."""
    return getattr(mod, "__name__", "") == "perseus._client"


def mode(ext: types.ModuleType, name):
    """None | "sync"/"prefetch"/"stream"/"cached"/"threaded" | ext.InferenceMode -> that
    extension's InferenceMode (None passes through: "use inf.mode at run time")."""
    if name is None or isinstance(name, ext.InferenceMode):
        return name
    try:
        return getattr(ext.InferenceMode, _MODE_NAMES[name.lower()])
    except (KeyError, AttributeError):
        raise ValueError(f"overlap must be None, one of {sorted(set(_MODE_NAMES))}, "
                         f"or an InferenceMode; got {name!r}") from None
