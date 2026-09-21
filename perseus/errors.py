"""The error taxonomy in one importable place.

C++-raised (markers translated by the bindings; all subclass _core.FHEError):
    FHEError      any runtime error from the FHE core
    PlanError     [plan_*]: strict planned-mode violations (level/weight/bts)
    MaskError     [mask_*]: strict decode-mask misses / mask generation
    LayoutError   [layout_error]: slot-layout basis mismatch (_core.set_strict_layout)

Python-raised:
    PlanContractError   plan loaded under a different env than it was captured in
    BundleError         a key bundle's manifest disagrees with the session asked to load it
    SecurityError       a trust-boundary invariant would be violated (e.g. a server session
                        that holds a secret key)

This module imports without the compiled extension: the C++ classes are then stand-ins
with the same names and hierarchy, so `except perseus.errors.FHEError` is always valid.

Two extensions can carry the classes: perseus._core (exported here when it imports) and
perseus._client (exported when only it is built). With both built, _client raises its own
FHEError (a RuntimeError subclass, not this module's); EncClient re-raises _client errors
as these classes, so `except perseus.errors.FHEError` stays valid for the client role.
"""
try:
    from ._core import FHEError, LayoutError, MaskError, PlanError
except ImportError:
    try:
        from ._client import FHEError, LayoutError, MaskError, PlanError
    except ImportError:   # no extension built: keep the names and the hierarchy importable

        class FHEError(Exception):
            """Stand-in for perseus._core.FHEError when the extension is absent."""

        class PlanError(FHEError):
            """Stand-in for perseus._core.PlanError when the extension is absent."""

        class MaskError(FHEError):
            """Stand-in for perseus._core.MaskError when the extension is absent."""

        class LayoutError(FHEError):
            """Stand-in for perseus._core.LayoutError when the extension is absent."""


from .plan.contract import PlanContractError  # noqa: E402


class BundleError(RuntimeError):
    """A key bundle cannot be used by this session: its manifest (parameters, chain,
    model family) disagrees with what the caller asked for, or is missing."""


class SecurityError(RuntimeError):
    """A trust-boundary invariant would be violated."""


__all__ = [
    "BundleError",
    "FHEError",
    "LayoutError",
    "MaskError",
    "PlanContractError",
    "PlanError",
    "SecurityError",
]
