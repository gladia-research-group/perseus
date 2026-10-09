"""Disk cache of the prepared slot vectors (block weights, LM-head and feedback tiles): numpy work that does not
depend on the input and costs ~100 s at token 0.

An entry belongs to a model FAMILY (the model being served: its weights' location by default) and a VARIANT of it
(the weights' content, the configs and the builder sources), and is keyed within the variant by its kind and build
arguments. It stores that key: a missing, stale or unreadable entry is rebuilt, never reused. Entries are written
atomically. After each write the eviction RULES run in order over all entries:
  keep_latest_variants  per family, the PERSEUS_SLOTVEC_CACHE_VARIANTS (default 2) most recently used variants
  size_cap              then least recently used variants go until the cache fits PERSEUS_SLOTVEC_CACHE_GB (default 32)
plus the leftovers of killed runs and of older entry formats. PERSEUS_SLOTVEC_CACHE=0 disables the cache."""
from __future__ import annotations

import dataclasses
import hashlib
import os
import pickle
import re
import tempfile
import time
from pathlib import Path

SCHEMA = 2
_HERE = Path(__file__).resolve().parent
_SOURCES = (_HERE / "weights.py", _HERE.parents[1] / "perseus" / "impl" / "layout.py",
            _HERE.parents[1] / "perseus" / "impl" / "linear.py")
_ENTRY = re.compile(r"^(?P<family>[\w.-]+?)\.(?P<variant>[0-9a-f]{16})\.(?P<kind>\w+)-(?P<key>[0-9a-f]{32})\.pkl$")
_file_hashes: dict = {}


def enabled() -> bool:
    return os.environ.get("PERSEUS_SLOTVEC_CACHE", "1") not in ("", "0")


@dataclasses.dataclass(frozen=True)
class ModelRef:
    family: str      # the model being served, e.g. "openai-community_gpt2_classic"
    key: str         # its variant: weights content, configs, builder sources


@dataclasses.dataclass
class Entry:
    path: Path
    family: str
    variant: str
    size: int
    used: float      # mtime, refreshed on every hit


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


def family_of(weights_path) -> str:
    """The weights' location as a family name: its last three parent directories."""
    parts = Path(weights_path).resolve().parent.parts[-3:]
    return re.sub(r"[^\w.-]", "_", "_".join(parts))


def model_ref(weights_path, cfgs, family: str | None = None) -> ModelRef:
    """The weights' content, the parsed configs and the builder sources, within `family` (default: family_of)."""
    h = hashlib.sha256(f"schema {SCHEMA}\n".encode())
    h.update(_file_hash(weights_path).encode())
    h.update(repr(cfgs).encode())
    for p in _SOURCES:
        h.update(_file_hash(p).encode())
    return ModelRef(family or family_of(weights_path), h.hexdigest())


def cached(model: ModelRef, kind: str, args: tuple, build):
    """build() through the cache: `kind` and `args` name the entry within the model variant."""
    if not enabled():
        return build()
    key = hashlib.sha256(f"{model.key}\n{kind}\n{args!r}".encode()).hexdigest()
    from perseus.hub import cache_dir
    path = Path(cache_dir("slotvec")) / f"{model.family}.{model.key[:16]}.{kind}-{key[:32]}.pkl"
    try:
        with open(path, "rb") as f:
            entry = pickle.load(f)
        if entry.get("key") == key:
            os.utime(path)
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
    prune(path.parent, keep=path)
    return value


# ── eviction rules: (entries, keep) -> the entries to drop; `keep` is the entry just written ──

def _by_variant(entries):
    variants: dict = {}
    for e in entries:
        variants.setdefault((e.family, e.variant), []).append(e)
    return variants


def keep_latest_variants(n: int):
    """Per family, keep the n most recently used variants."""
    def rule(entries, keep):
        families: dict = {}
        for (family, _), es in _by_variant(entries).items():
            families.setdefault(family, []).append(es)
        drop = []
        for variants in families.values():
            variants.sort(key=lambda es: max(e.used for e in es), reverse=True)
            for es in variants[n:]:
                drop += [e for e in es if e.path != keep]
        return drop
    return rule


def size_cap(gb: float):
    """Drop least recently used variants, whole, until the cache fits `gb`."""
    def rule(entries, keep):
        total = sum(e.size for e in entries)
        drop = []
        for es in sorted(_by_variant(entries).values(), key=lambda es: max(e.used for e in es)):
            if total <= gb * 2**30:
                break
            if any(e.path == keep for e in es):
                continue
            drop += es
            total -= sum(e.size for e in es)
        return drop
    return rule


def default_rules():
    return [keep_latest_variants(int(os.environ.get("PERSEUS_SLOTVEC_CACHE_VARIANTS", "2"))),
            size_cap(float(os.environ.get("PERSEUS_SLOTVEC_CACHE_GB", "32")))]


def prune(d: Path, keep: Path | None = None, rules=None):
    """Apply the eviction rules to the cache in `d`; drop temp files of killed runs and older entry formats."""
    now = time.time()
    entries = []
    for f in d.iterdir():
        try:
            st = f.stat()
        except FileNotFoundError:
            continue
        m = _ENTRY.match(f.name)
        if m is None:
            if f.suffix == ".pkl" or now - st.st_mtime > 3600:   # an older format, or a killed run's temp file
                f.unlink(missing_ok=True)
            continue
        entries.append(Entry(f, m["family"], m["variant"], st.st_size, st.st_mtime))
    for rule in rules if rules is not None else default_rules():
        gone = {e.path for e in rule(entries, keep)}
        for p in gone:
            p.unlink(missing_ok=True)
        entries = [e for e in entries if e.path not in gone]
