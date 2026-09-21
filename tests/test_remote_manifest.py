"""CPU tests for the client/server bundle manifest: no keygen (52 s), no GPU.

The manifest is what lets a server rebuild the client's context without trusting its
own environment; these tests pin the round trip, the diff, and the refusals that
happen BEFORE any session is created — and the fresh-encode level handshake going
the other way (`EncServer.session_manifest()` -> `EncClient.accept()`), driven with a
fake `_core` and keygen-free doubles of both roles.
"""
import json
import logging
import types

import numpy as np
import pytest

_core = pytest.importorskip("perseus._core")
from perseus.errors import BundleError
from perseus.nn import EncClient, EncServer, remote
from perseus.nn.remote import (
    MANIFEST_NAME,
    manifest_diff,
    options_from_manifest,
    options_to_manifest,
)


def _opts():
    o = _core.InferenceOptions()
    o.ckks = _core.CKKSOptions()
    o.ckks.logN = 15
    o.ckks.depth = 9
    o.ckks.level_budget = [3, 3]
    o.ckks.sparse_bts_slots_list = [512, 1]
    o.ckks.extra_rot_steps = [1, 2, 4]
    o.ckks.ckks_complex_payload = True
    o.dim = 512
    o.hidDim = 512
    o.aux_packing_kinds = [_core.PackingKind.CachemirFilling]
    o.packing_kind = _core.PackingKind.CachemirComplex
    return o


def test_manifest_round_trip_is_json_and_lossless():
    m = options_to_manifest(_opts(), "gpt2")
    m2 = json.loads(json.dumps(m))                     # must be plain JSON
    back = options_from_manifest(m2)
    assert options_to_manifest(back, "gpt2") == m
    assert back.ckks.logN == 15 and back.ckks.level_budget == [3, 3]
    assert back.ckks.sparse_bts_slots_list == [512, 1] and back.ckks.ckks_complex_payload
    assert back.dim == 512 and back.packing_kind == _core.PackingKind.CachemirComplex
    assert back.aux_packing_kinds == [_core.PackingKind.CachemirFilling]
    # process-local knobs never travel
    assert "keys_dir" not in m["ckks"] and "skip_gpu_load" not in m["ckks"]


def test_manifest_diff_names_the_fields():
    a = options_to_manifest(_opts(), "gpt2")
    o = _opts()
    o.ckks.depth = 10
    o.dim = 768
    b = options_to_manifest(o, "generic")
    d = manifest_diff(a, b)
    assert any(x.startswith("family:") for x in d)
    assert any(x.startswith("ckks.depth:") for x in d)
    assert any(x.startswith("inference.dim:") for x in d)
    assert manifest_diff(a, options_to_manifest(_opts(), "gpt2")) == []


def test_bad_family_rejected():
    with pytest.raises(ValueError, match="family must be one of"):
        options_to_manifest(_opts(), "llama")


def test_server_refuses_a_bundle_of_another_family_before_keygen(tmp_path):
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(options_to_manifest(_opts(), "generic")))
    with pytest.raises(BundleError, match="keyed for family 'generic'"):
        EncServer(str(tmp_path), family="gpt2")


def test_server_refuses_mismatched_caller_options_before_keygen(tmp_path):
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(options_to_manifest(_opts(), "gpt2")))
    other = _opts()
    other.ckks.logN = 16
    with pytest.raises(BundleError, match="ckks.logN: bundle=15 caller=16"):
        EncServer(str(tmp_path), options=other)


def test_unsupported_manifest_format_is_a_bundle_error():
    with pytest.raises(BundleError, match="format"):
        options_from_manifest({"format": 99, "ckks": {}, "inference": {}})


# ---- the fresh-encode level handshake (server -> client) ----

_D = 8


class _FakeCore:
    """Records the level every client encode is pinned to; everything else (the option
    structs the manifest helpers build) is the real extension."""

    def __init__(self):
        self.levels = []
        self.__name__ = "perseus._fake"

    def __getattr__(self, name):
        return getattr(_core, name)

    def pack_tokens(self, inf, m, level):
        assert m.shape == (1, _D)
        self.levels.append(int(level))
        return "ct"

    def serialize_ct(self, inf, ct):
        return b"ct"


@pytest.fixture
def fake_core(monkeypatch):
    fc = _FakeCore()
    monkeypatch.setattr(remote, "_ext", fc)     # the module default the client role runs on
    return fc


class _CpuClient(EncClient):
    """An EncClient without keygen: a GPU-less session whose bootstrap_output_level()
    is the parameter formula."""

    def __init__(self, formula=32):
        self.family = "gpt2"
        self.options = _opts()
        self.inf = types.SimpleNamespace(
            fhe=types.SimpleNamespace(bootstrap_output_level=lambda: formula),
            size=types.SimpleNamespace(dim=_D))
        self.d = _D


def _server(manifest, probed):
    """An EncServer without a session: the bundle it loaded and the level it probed."""
    s = EncServer.__new__(EncServer)
    s.manifest = manifest
    s.family = "gpt2"
    s.fresh_encode_level = probed
    return s


def test_client_adopts_the_servers_probed_level_and_the_manifest_round_trips(fake_core):
    bundle = options_to_manifest(_opts(), "gpt2") | {"bootstrap_output_level": 32}
    sm = json.loads(json.dumps(_server(bundle, 34).session_manifest()))   # plain JSON
    assert sm["bootstrap_output_level"] == 34                    # the probe replaces the formula
    assert sm["ckks"] == bundle["ckks"] and sm["inference"] == bundle["inference"]
    assert options_from_manifest(sm).ckks.logN == 15             # still a valid manifest
    assert manifest_diff(sm, options_to_manifest(_opts(), "gpt2")) == []   # no format bump

    c = _CpuClient(formula=32)
    assert c.encode_level == 32
    c.encrypt(np.zeros(_D))
    assert c.accept(sm) == 34
    assert c.encode_level == 34
    c.encrypt(np.ones(_D))
    assert fake_core.levels == [32, 34]                          # the level the encode was pinned to
    assert c.manifest()["bootstrap_output_level"] == 34


def test_client_without_a_probed_level_keeps_the_formula_and_logs_once(fake_core, caplog):
    caplog.set_level(logging.INFO, logger="perseus.nn.remote")
    c = _CpuClient(formula=32)
    # absent: an empty / missing manifest, or a bundle manifest handed over by mistake
    assert c.accept({}) == 32
    assert c.accept(None) == 32
    assert c.accept(options_to_manifest(_opts(), "gpt2")) == 32
    assert c.encode_level == 32
    c.encrypt(np.zeros(_D))
    assert fake_core.levels == [32]
    infos = [r for r in caplog.records if r.levelno == logging.INFO
             and "bootstrap_output_level" in r.getMessage()]
    assert len(infos) == 1                                       # one INFO across all three
    # present: adopted
    assert c.accept({"bootstrap_output_level": 34}) == 34
    assert c.encode_level == 34


def test_session_manifest_of_a_manifest_less_bundle():
    assert _server(None, 34).session_manifest() == {"bootstrap_output_level": 34}


def test_accept_validates_the_level():
    c = _CpuClient(formula=32)
    for bad in (0, -2, "34", True, 3.5):
        with pytest.raises(ValueError, match="positive integer"):
            c.accept({"bootstrap_output_level": bad})
    assert c.encode_level == 32                                  # nothing adopted
    c.options.ckks.composite_degree = 2
    with pytest.raises(ValueError, match="multiple of 2"):
        c.accept({"bootstrap_output_level": 33})
    assert c.accept({"bootstrap_output_level": 34}) == 34
    assert c.accept({"bootstrap_output_level": np.int64(36)}) == 36
    assert isinstance(c.encode_level, int)


def test_keygen_free_client_runs_on_the_module_default(fake_core):
    """A double that skips __init__ (no _backend chosen) is backed by `remote._ext`, which
    is what the fixture monkeypatches — so the level pinning above sees the fake."""
    c = _CpuClient(formula=32)
    assert c.backend is remote._ext is fake_core
    assert "backend=perseus._fake" in repr(c)


def test_backend_choice_is_validated():
    with pytest.raises(ValueError, match="backend must be"):
        EncClient(backend="gpu")
