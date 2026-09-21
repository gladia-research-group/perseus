"""state_dict / save / load for custom Enc models (CPU)."""
import numpy as np
import pytest

pytest.importorskip("perseus._core")
from perseus.nn import EncGELU, EncLayerNorm, EncLinear, EncSequential, load, save


def _model():
    rng = np.random.default_rng(0)
    return EncSequential(
        EncLayerNorm("ln", weight=1 + 0.1 * rng.standard_normal(8), bias=rng.standard_normal(8)),
        EncLinear("fc1", 8, 16, weight=rng.standard_normal((8, 16)), bias=rng.standard_normal(16)),
        EncGELU("act"),
        EncLinear("fc2", 16, 8, weight=rng.standard_normal((16, 8))),
        EncLinear("shared", 8, 8),                      # pre-installed name: no parameters
        overlap="sync",
    )


def test_state_dict_keys_are_torch_style():
    sd = _model().state_dict()
    assert sorted(sd) == ["0.bias", "0.weight", "1.bias", "1.weight", "3.weight"]
    assert sd["1.weight"].shape == (8, 16)


def test_load_state_dict_checks_shapes_and_keys():
    m = _model()
    sd = m.state_dict()
    sd["1.weight"] = sd["1.weight"] * 2
    m.load_state_dict(sd)
    np.testing.assert_array_equal(m[1].weight, sd["1.weight"])
    with pytest.raises(ValueError, match="shape"):
        m.load_state_dict({**sd, "1.weight": np.zeros((16, 8))})
    with pytest.raises(KeyError, match="unexpected"):
        m.load_state_dict({**sd, "9.weight": np.zeros(3)})
    with pytest.raises(KeyError, match="missing"):
        m.load_state_dict({"0.weight": sd["0.weight"]})
    m.load_state_dict({"0.weight": sd["0.weight"]}, strict=False)


def test_save_load_round_trip(tmp_path):
    m = _model()
    save(m, tmp_path / "mlp")
    assert (tmp_path / "mlp" / "model.json").exists() and (tmp_path / "mlp" / "weights.npz").exists()
    m2 = load(tmp_path / "mlp")
    assert repr(m2) == repr(m)
    for k, v in m.state_dict().items():
        np.testing.assert_array_equal(m2.state_dict()[k], v)
    assert m2[4].weight is None and m2.overlap == m.overlap
    assert m2.torch_mirror() is None                  # the shared linear has no values


def test_unserializable_module_says_so(tmp_path):
    from perseus.nn import EncModule

    class Custom(EncModule):
        pass

    with pytest.raises(NotImplementedError, match="Custom cannot be rebuilt"):
        save(Custom(), tmp_path / "x")
