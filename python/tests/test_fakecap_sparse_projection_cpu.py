"""The capture twin's sparse refresh returns the s-periodic projection of its input, as the device does: a 1-slot
refresh of a token's values gives their slot mean (what a LayerNorm's statistic from one 1-slot refresh reads), an
input that is already s-periodic comes back bit for bit, and FAKECAP_SPARSE_EXACT=0 passes the input through."""
from __future__ import annotations

import numpy as np
import pytest

from perseus import config
from perseus.impl import fakecap
from perseus.impl.layout import Dims


def _refresh(monkeypatch, tmp_path, vec, s, env=None):
    """The vector a capture-twin refresh on the s-slot route leaves in place of `vec`."""
    monkeypatch.setenv("FHE_GRAPH_DIR", str(tmp_path))
    cfg = config.preset("n32_gpt2_decode")
    if env:
        cfg = cfg.with_env(env)
    fhe, inf = fakecap.make_session(Dims(1024, 32, 24, 4, 3, 128, 96), range(1, 1024), config=cfg)
    assert fakecap.core.begin_subgraph_capture(inf, 0)
    x = fakecap.core.encode_token_input(inf, np.linspace(-1, 1, 24))
    x.vec = np.array(vec, copy=True)
    monkeypatch.setattr(fhe, "auto_route", lambda ct: s)
    fhe.inner_bootstrap(x)
    return np.asarray(x.vec)


def test_one_slot_refresh_returns_the_slot_mean(monkeypatch, tmp_path):
    v = np.random.default_rng(0).normal(size=1024)
    out = _refresh(monkeypatch, tmp_path, v, 1)
    assert np.allclose(out, v.mean()) and np.ptp(out) == 0


def test_sparse_refresh_projects_onto_its_period(monkeypatch, tmp_path):
    v = np.random.default_rng(1).normal(size=1024)
    out = _refresh(monkeypatch, tmp_path, v, 512)
    assert np.allclose(out, np.tile(v.reshape(2, 512).mean(0), 2))


@pytest.mark.parametrize("s", [1, 512])
@pytest.mark.parametrize("eps", [0.0, 4e-16, 1e-11])
def test_periodic_input_is_kept_bit_for_bit(monkeypatch, tmp_path, s, eps):
    """An s-periodic input, exactly or to rounding (eps of its scale), comes back unchanged."""
    v = np.tile(np.random.default_rng(2).normal(size=s) / 3.0, 1024 // s)
    v = v + eps * np.random.default_rng(4).normal(size=1024)
    out = _refresh(monkeypatch, tmp_path, v, s)
    assert np.array_equal(out, v)


def test_pass_through_on_request(monkeypatch, tmp_path):
    v = np.random.default_rng(3).normal(size=1024)
    out = _refresh(monkeypatch, tmp_path, v, 1, env={"FAKECAP_SPARSE_EXACT": "0"})
    assert np.array_equal(out, v)
