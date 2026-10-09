"""The slot-vector cache (examples/gpt2_from_primitives/slotcache.py): a hit equals a fresh build; the configs, the
weights, the build arguments and the builder sources are part of the key; an unreadable entry is rebuilt; the
eviction rules keep the latest variants per family under a size cap."""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.gpt2_from_primitives import slotcache  # noqa: E402
from perseus.impl.config import load_configs  # noqa: E402

CFG = ROOT / "configs/model/approximation/gpt2_base_n32/configs.json"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    import huggingface_hub.constants as c
    monkeypatch.setattr(c, "HF_HOME", str(tmp_path))
    monkeypatch.setenv("PERSEUS_SLOTVEC_CACHE", "1")
    return tmp_path / "perseus" / "slotvec"


def _weights(tmp_path, name="w.bin", data=b"weights"):
    w = tmp_path / "models" / "org" / "model" / "variant" / name
    w.parent.mkdir(parents=True, exist_ok=True)
    w.write_bytes(data)
    return w


def test_hit_equals_build_and_key_parts(cache, tmp_path):
    w = _weights(tmp_path, data=b"weights v1")
    cfgs = load_configs(str(CFG))
    calls = []

    def build(tag):
        def f():
            calls.append(tag)
            return {"pts": np.arange(8.0) * len(calls), "tag": tag}
        return f

    m = slotcache.model_ref(w, cfgs)
    assert m.family == "org_model_variant"
    a = slotcache.cached(m, "block", (0, "dims"), build("a"))
    b = slotcache.cached(m, "block", (0, "dims"), build("b"))
    assert calls == ["a"] and b["tag"] == "a" and np.array_equal(a["pts"], b["pts"])
    slotcache.cached(m, "block", (1, "dims"), build("c"))                 # another argument: a new entry
    assert calls == ["a", "c"]
    cfgs.norm["transformer.h.0.ln_1"].epsilon *= 2                        # configs are in the key
    slotcache.cached(slotcache.model_ref(w, cfgs), "block", (0, "dims"), build("d"))
    assert calls[-1] == "d"
    w.write_bytes(b"weights v2")                                           # so are the weights' contents
    slotcache.cached(slotcache.model_ref(w, load_configs(str(CFG))), "block", (0, "dims"), build("e"))
    assert calls[-1] == "e"


def test_unreadable_entry_is_rebuilt(cache, tmp_path):
    m = slotcache.model_ref(_weights(tmp_path), load_configs(str(CFG)))
    slotcache.cached(m, "lm", (), lambda: {"x": 1})
    (entry,) = list(cache.glob("*.lm-*.pkl"))
    entry.write_bytes(b"truncated")
    assert slotcache.cached(m, "lm", (), lambda: {"x": 2}) == {"x": 2}
    assert slotcache.cached(m, "lm", (), lambda: {"x": 3}) == {"x": 2}    # and the rewrite is a hit


def test_sources_are_in_the_key(cache, tmp_path, monkeypatch):
    w = _weights(tmp_path)
    cfgs = load_configs(str(CFG))
    k1 = slotcache.model_ref(w, cfgs).key
    src = tmp_path / "builder.py"
    src.write_text("v = 1\n")
    monkeypatch.setattr(slotcache, "_SOURCES", slotcache._SOURCES + (src,))
    k2 = slotcache.model_ref(w, cfgs).key
    src.write_text("v = 2\n")
    k3 = slotcache.model_ref(w, cfgs).key
    assert len({k1, k2, k3}) == 3


def test_disabled(cache, tmp_path, monkeypatch):
    monkeypatch.setenv("PERSEUS_SLOTVEC_CACHE", "0")
    n = []
    for _ in range(2):
        slotcache.cached(slotcache.ModelRef("f", "0" * 64), "fb", (), lambda: n.append(1))
    assert len(n) == 2 and not cache.exists()


def _variant(family, i):
    return slotcache.ModelRef(family, f"{i:x}" * 64)


def test_latest_variants_per_family(cache, monkeypatch):
    monkeypatch.setenv("PERSEUS_SLOTVEC_CACHE_VARIANTS", "2")
    for i in range(3):                                   # three variants of gpt2, each with two entries
        for kind in ("block", "lm"):
            slotcache.cached(_variant("gpt2", i), kind, (), lambda: {"x": 1})
        time.sleep(0.01)
    slotcache.cached(_variant("vit", 0), "block", (), lambda: {"x": 1})   # another family is untouched
    left = {(p.name.split(".")[0], p.name.split(".")[1]) for p in cache.glob("*.pkl")}
    assert left == {("gpt2", "1" * 16), ("gpt2", "2" * 16), ("vit", "0" * 16)}


def test_size_cap_drops_whole_variants(cache, monkeypatch):
    monkeypatch.setenv("PERSEUS_SLOTVEC_CACHE_GB", str(2.5 / 1024))   # ~2.5 MB
    blob = lambda: {"x": np.zeros(1 << 17)}                            # ~1 MB pickled
    for fam in ("bert", "llama", "vit"):
        slotcache.cached(_variant(fam, 0), "block", (), blob)
        time.sleep(0.01)
    left = sorted(p.name.split(".")[0] for p in cache.glob("*.pkl"))
    assert left == ["llama", "vit"]                                     # the least recently used family went


def test_leftovers_are_cleaned(cache):
    cache.mkdir(parents=True, exist_ok=True)
    old = cache / "block-0123456789abcdef0123456789abcdef.pkl"       # the previous entry format
    tmp = cache / ("gpt2." + "0" * 16 + ".block-x.pkl.abc")            # a killed run's temp file
    for f in (old, tmp):
        f.write_bytes(b"x")
    os.utime(tmp, (1, 1))
    slotcache.cached(_variant("gpt2", 0), "block", (), lambda: {"x": 1})
    assert not old.exists() and not tmp.exists()
