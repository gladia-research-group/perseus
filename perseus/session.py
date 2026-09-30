"""A session is a live FHE context plus the model-shaped state the runtime keeps for it
(an `_core.Inference`), with a Python lifecycle around it.

    from perseus import session
    from perseus.profile import SessionProfile

    with session(profile=SessionProfile.custom_n32()) as s:
        model = EncSequential(...).bind(s)        # bind() accepts the session
        y = model(x)

`close()` (or leaving the block) releases what the session holds on the device —
installed weights, the KV/mask/encode caches and the loaded rotation keys — and drops
this handle; the CKKS context object itself lives until process exit (ciphertexts
reference it). A second `session()` while one is open is refused rather than silently
sharing global state; after close() a new one can be created.
"""
from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

_LIVE = threading.Lock()
_live_session = None


class Session:
    """Handle on a live `_core.Inference` with `close()` / context-manager support."""

    def __init__(self, inf, options=None, family: str | None = None):
        self._inf = inf
        self.options = options
        self.family = family

    @property
    def inf(self):
        if self._inf is None:
            raise RuntimeError("this perseus.Session is closed")
        return self._inf

    @property
    def closed(self) -> bool:
        return self._inf is None

    def close(self) -> None:
        """Idempotent. Releases what the session holds (weights, KV/mask/encode caches,
        rotation keys) and detaches the handle. The memory returns to the runtime's device
        pool — reused by the next session in this process, given back to the device at
        exit — so a driver-level free-memory counter does not move; `_core.close_session`
        documents why."""
        global _live_session
        inf, self._inf = self._inf, None
        if inf is None:
            return
        try:
            from . import _core
            if hasattr(_core, "close_session"):
                try:
                    _core.close_session(inf)      # weights, caches, rotation keys
                except TypeError:                 # not a runtime Inference (a test double)
                    inf.clear_enc_cache()
            else:
                inf.clear_enc_cache()
        except Exception:      # a session already torn down by the runtime; nothing to free
            pass
        with _LIVE:
            if _live_session is self:
                _live_session = None

    def encrypt(self, values):
        """Plaintext vector (<= size.dim real features; shorter is zero-padded) -> ciphertext."""
        import numpy as np

        from . import _core
        x = np.asarray(values, dtype=np.float64)
        d = self.inf.size.dim
        if x.ndim != 1:
            raise ValueError(f"encrypt: expected a 1-D vector of up to {d} features, got "
                             f"shape {x.shape}")
        if x.shape[0] > d:
            raise ValueError(f"encrypt: {x.shape[0]} features do not fit the session's "
                             f"{d}-wide token (values past {d} would be dropped)")
        if not np.isfinite(x).all():
            raise ValueError("encrypt: input contains NaN/inf")
        return _core.encode_token_input(self.inf, x)

    def decrypt(self, ct, d: int | None = None) -> np.ndarray:
        """Ciphertext -> plaintext vector; d trims to the first d lanes."""
        import numpy as np

        from . import _core
        out = np.array(_core.decode_token_output(self.inf, ct))
        return out[:d] if d is not None else out

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        state = "closed" if self.closed else "open"
        return f"Session({state}, family={self.family!r})"


def session(profile=None, options=None, family: str = "gpt2", mode: str | None = None,
            device: int | str | None = None) -> Session:
    """Create the process's FHE session.

    profile: a perseus.profile.SessionProfile (its env view is applied as defaults,
    then CKKSOptions are read); options: an explicit _core.InferenceOptions (wins over
    profile); neither: the environment. family picks the rotation-key band
    ("gpt2" | "vit" | "bert" | "generic"). mode overrides the residency scheduling
    ("sync" | "prefetch" | "threaded"). device: the GPU index this process should use
    (sets CUDA_VISIBLE_DEVICES before the runtime's first CUDA call; the runtime has no
    per-context device selection, so this must be the first session of the process).
    """
    global _live_session
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    from . import _core
    from .nn.pipeline import resolve_overlap
    from .nn.remote import _FAMILIES, _copy_options, _resolve_options

    if family not in _FAMILIES:
        raise ValueError(f"family must be one of {sorted(_FAMILIES)}, got {family!r}")
    with _LIVE:
        if _live_session is not None and not _live_session.closed:
            raise RuntimeError("a perseus.Session is already open in this process; close it "
                               "first (the runtime keeps one context per process)")
    opts = _copy_options(_resolve_options(options, profile), family)
    if mode is not None:
        opts.mode = resolve_overlap(mode)
    inf = getattr(_core, _FAMILIES[family])(opts)
    s = Session(inf, options=opts, family=family)
    with _LIVE:
        _live_session = s
    return s
