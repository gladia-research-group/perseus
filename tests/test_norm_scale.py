"""EncLayerNorm's fitted output scale is a parameter: given, saved/loaded, used at bind
without a decrypt — the calibrate-once, deploy-through-the-roles path (CC-04)."""
import pytest

pytest.importorskip("perseus._core")
from perseus.nn import EncLayerNorm, EncSequential, load, save
from perseus.nn import norm as norm_mod


class _FakeInf:
    """Records set_ln_affine / set_norm_cfg; no key, no GPU."""

    def __init__(self):
        self.affine = []
        self.cfgs = []

    def set_norm_cfg(self, name, cfg):
        self.cfgs.append((name, cfg))


@pytest.fixture
def capture(monkeypatch):
    calls = []
    monkeypatch.setattr(norm_mod._core, "set_ln_affine", lambda inf, name, g, b: calls.append((name, g, b)))
    return calls


def test_scale_is_validated_and_shown():
    m = EncLayerNorm("ln", weight=[1, 2], bias=[0.5, 0.5], scale=(2.0, 4.0))
    assert m.scale == (2.0, 4.0) and "scale=(2, 4)" in repr(m)
    with pytest.raises(ValueError, match="pair"):
        EncLayerNorm("ln", scale=3)
    with pytest.raises(ValueError, match="non-zero"):
        EncLayerNorm("ln", scale=(0.0, 1.0))


def test_known_scale_is_installed_at_bind_without_a_fit(capture):
    m = EncLayerNorm("ln", weight=[1.0, 2.0], bias=[0.5, 1.0], scale=(2.0, 4.0))
    m.bind(_FakeInf())
    name, g, b = capture[-1]
    assert name == "ln" and g == [0.5, 1.0] and b == [0.125, 0.25]     # gamma/r_g, beta/r_b
    m.set_scale((1.0, 1.0))
    assert capture[-1][1] == [1.0, 2.0]                                # re-installed live


def test_apply_cfg_keeps_a_known_scale_and_refuses_without_one(capture):
    inf = _FakeInf()
    m = EncLayerNorm("ln", weight=[1.0, 1.0], bias=[0.0, 0.0], scale=(2.0, 1.0)).bind(inf)
    m.apply_cfg("cfg-object")                 # no decrypt: the known scale is installed
    assert inf.cfgs == [("ln", "cfg-object")] and capture[-1][1] == [0.5, 0.5]
    m2 = EncLayerNorm("ln2", weight=[1.0, 1.0], bias=[0.0, 0.0]).bind(inf)
    with pytest.raises(ValueError, match="no scale is known"):
        m2.apply_cfg("cfg", refit=False)


def test_scale_survives_save_load(tmp_path):
    m = EncSequential(EncLayerNorm("ln", weight=[1.0, 2.0], bias=[0.0, 1.0], scale=(1.5, 0.75)))
    save(m, tmp_path / "m")
    m2 = load(tmp_path / "m")
    assert m2[0].scale == (1.5, 0.75) and repr(m2) == repr(m)
    assert m2[0].to_config()["scale"] == [1.5, 0.75]
    assert EncLayerNorm("ln", 4).to_config()["scale"] is None
