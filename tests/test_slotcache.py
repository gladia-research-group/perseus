"""The slot-vector cache (examples/gpt2_from_primitives/slotcache.py): a hit equals a fresh build; the configs, the
build arguments and the builder sources are part of the key; an unreadable entry is rebuilt."""
import sys
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


def test_hit_equals_build_and_key_parts(cache, tmp_path):
    w = tmp_path / "w.bin"
    w.write_bytes(b"weights v1")
    cfgs = load_configs(str(CFG))
    calls = []

    def build(tag):
        def f():
            calls.append(tag)
            return {"pts": np.arange(8.0) * len(calls), "tag": tag}
        return f

    k = slotcache.model_key(w, cfgs)
    a = slotcache.cached(k, "block", (0, "dims"), build("a"))
    b = slotcache.cached(k, "block", (0, "dims"), build("b"))
    assert calls == ["a"] and b["tag"] == "a" and np.array_equal(a["pts"], b["pts"])
    slotcache.cached(k, "block", (1, "dims"), build("c"))                 # another argument: a new entry
    assert calls == ["a", "c"]
    cfgs.norm["transformer.h.0.ln_1"].epsilon *= 2                        # configs are in the key
    slotcache.cached(slotcache.model_key(w, cfgs), "block", (0, "dims"), build("d"))
    assert calls[-1] == "d"
    w.write_bytes(b"weights v2")                                           # so are the weights' contents
    slotcache.cached(slotcache.model_key(w, load_configs(str(CFG))), "block", (0, "dims"), build("e"))
    assert calls[-1] == "e"


def test_unreadable_entry_is_rebuilt(cache, tmp_path):
    w = tmp_path / "w.bin"
    w.write_bytes(b"weights")
    k = slotcache.model_key(w, load_configs(str(CFG)))
    slotcache.cached(k, "lm", (), lambda: {"x": 1})
    (entry,) = list(cache.glob("lm-*.pkl"))
    entry.write_bytes(b"truncated")
    assert slotcache.cached(k, "lm", (), lambda: {"x": 2}) == {"x": 2}
    assert slotcache.cached(k, "lm", (), lambda: {"x": 3}) == {"x": 2}    # and the rewrite is a hit


def test_sources_are_in_the_key(cache, tmp_path, monkeypatch):
    w = tmp_path / "w.bin"
    w.write_bytes(b"weights")
    cfgs = load_configs(str(CFG))
    k1 = slotcache.model_key(w, cfgs)
    src = tmp_path / "builder.py"
    src.write_text("v = 1\n")
    monkeypatch.setattr(slotcache, "_SOURCES", slotcache._SOURCES + (src,))
    k2 = slotcache.model_key(w, cfgs)
    src.write_text("v = 2\n")
    k3 = slotcache.model_key(w, cfgs)
    assert len({k1, k2, k3}) == 3


def test_disabled(cache, tmp_path, monkeypatch):
    monkeypatch.setenv("PERSEUS_SLOTVEC_CACHE", "0")
    n = []
    for _ in range(2):
        slotcache.cached("k", "fb", (), lambda: n.append(1))
    assert len(n) == 2 and not cache.exists()
