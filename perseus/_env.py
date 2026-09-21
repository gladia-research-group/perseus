import contextlib
import os


@contextlib.contextmanager
def scoped_env(**values):
    """Set environment variables for the duration of the block (None unsets), then
    restore every touched name to its previous state."""
    saved = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def current_chain(default: str | None = None) -> str | None:
    """The native-integer chain the loaded extension was built for ("n32" | "n64"):
    the module's own stamp when it carries one (perseus._core, else perseus._client), else
    the CHAIN environment variable."""
    from ._backend import load

    chain = None
    for name in ("_core", "_client"):
        mod = load(name)
        chain = getattr(mod, "chain", None) if mod is not None else None
        if chain:
            break
    return chain or os.environ.get("CHAIN") or default
