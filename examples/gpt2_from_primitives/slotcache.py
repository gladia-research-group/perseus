"""Disk cache of the prepared slot vectors (block weights, LM-head and feedback tiles): numpy work that does not
depend on the input and costs ~100 s at token 0. An entry is keyed by the weights' content, the configs, the build
arguments and the source of the modules that build it, and stores that key: a missing, stale or unreadable entry is
rebuilt, never reused. Entries are written atomically. PERSEUS_SLOTVEC_CACHE=0 disables the cache."""
from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
from pathlib import Path

SCHEMA = 1
_HERE = Path(__file__).resolve().parent
_SOURCES = (_HERE / "weights.py", _HERE.parents[1] / "perseus" / "impl" / "layout.py",
            _HERE.parents[1] / "perseus" / "impl" / "linear.py")
_file_hashes: dict = {}


def enabled() -> bool:
    return os.environ.get("PERSEUS_SLOTVEC_CACHE", "1") not in ("", "0")


def _file_hash(path) -> str:
    st = os.stat(path)
    memo = (str(path), st.st_size, st.st_mtime_ns)
    if memo not in _file_hashes:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        _file_hashes[memo] = h.hexdigest()
    return _file_hashes[memo]


def model_key(weights_path, cfgs) -> str:
    """The weights' content, the parsed configs and the builder sources."""
    h = hashlib.sha256(f"schema {SCHEMA}\n".encode())
    h.update(_file_hash(weights_path).encode())
    h.update(repr(cfgs).encode())
    for p in _SOURCES:
        h.update(_file_hash(p).encode())
    return h.hexdigest()


def cached(model: str, kind: str, args: tuple, build):
    """build() through the cache: `kind` and `args` name the entry within the model key."""
    if not enabled():
        return build()
    key = hashlib.sha256(f"{model}\n{kind}\n{args!r}".encode()).hexdigest()
    from perseus.hub import cache_dir
    path = Path(cache_dir("slotvec")) / f"{kind}-{key[:32]}.pkl"
    try:
        with open(path, "rb") as f:
            entry = pickle.load(f)
        if entry.get("key") == key:
            return entry["value"]
    except Exception:   # missing, truncated or from another layout: rebuild
        pass
    value = build()
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump({"key": key, "value": value}, f, protocol=5)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return value
