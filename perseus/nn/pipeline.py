from .. import _core
from .._backend import _MODE_NAMES

Stage = _core.Stage

_OVERLAP = {k: getattr(_core.InferenceMode, v) for k, v in _MODE_NAMES.items()}


def resolve_overlap(overlap):
    """None | "sync"/"prefetch"/"threaded" | InferenceMode -> InferenceMode | None.

    None means "use inf.mode at run time", matching the C++ drivers.
    """
    if overlap is None or isinstance(overlap, _core.InferenceMode):
        return overlap
    try:
        return _OVERLAP[overlap.lower()]
    except (KeyError, AttributeError):
        raise ValueError(f"overlap must be None, one of {sorted(set(_OVERLAP))}, "
                         f"or an InferenceMode; got {overlap!r}") from None


def stage_of(module, label=None):
    """One pipeline stage from a bound module; module.residency() decides the hooks.

    residency() -> None       : opaque stage (forward runs, nothing to overlap)
                -> list[str]  : inf.w keys streamed around forward
                -> EncodedBlock: cached state acquired/installed/evicted around forward
                -> Stage      : used as-is (full control over compute/release)
    """
    r = module.residency()
    if isinstance(r, _core.Stage):
        return r
    kw = {"label": label or type(module).__name__.lower()}
    if isinstance(r, _core.EncodedBlock):
        kw["state"] = r
    elif r is not None:
        kw["weights"] = list(r)
    state = kw.get("state")

    def compute(x, _m=module, _state=state):
        if _state is not None and _state.prefix:
            _m.inf.block_prefix = _state.prefix
        return _m(x)

    return _core.Stage(compute=compute, **kw)


def run_stages(inf, x, stages, overlap=None):
    """Run stages through the residency pipeline (the C++ cached-block runner).

    x is opaque to the pipeline (a PackedCtx, a list of chunk cts, ...); each
    stage's compute maps x -> x on the main thread while the runner overlaps the
    next stage's weight residency (CPU extraction on the worker, uploads on a
    side stream). overlap None uses inf.mode, like the C++ drivers.
    """
    return _core.run_stages(inf, x, list(stages), resolve_overlap(overlap))
