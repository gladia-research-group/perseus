"""CPU-only tests for the Python authoring surface: no GPU, no context, seconds.

Covers what does not need ciphertexts: torch mirrors, the calibration fit +
Chebyshev conversion + configs.json parse round-trip, plan/env contract stamping
and validation, session profiles, the error taxonomy, and the pipeline sugar's
stage construction. The GPU probes (scripts/utils/probe_*.py) cover the rest.
"""
import json
import os

import numpy as np
import pytest

_core = pytest.importorskip("perseus._core")   # the .so must be importable (any chain)
from perseus.nn import EncGELU, EncLayerNorm, EncLinear, EncSequential, Stage  # noqa: E402
from perseus.plan import contract  # noqa: E402

D_PAD, D_REAL, D_EXP, E_REAL = 1024, 768, 4096, 3072


def _w(rng, di, do, ir, orr, s=0.5):
    w = rng.standard_normal((di, do)) * s / np.sqrt(ir)
    w[ir:, :] = 0.0
    w[:, orr:] = 0.0
    return w


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    W1, W2 = _w(rng, D_PAD, D_EXP, D_REAL, E_REAL), _w(rng, D_EXP, D_PAD, E_REAL, D_REAL)
    b1 = np.zeros(D_EXP); b1[:E_REAL] = 0.05 * rng.standard_normal(E_REAL)
    g = 1.0 + 0.1 * rng.standard_normal(D_REAL); b = 0.05 * rng.standard_normal(D_REAL)
    model = EncSequential(
        EncLayerNorm("cln", D_REAL, weight=g.tolist(), bias=b.tolist()),
        EncLinear("fc1", D_PAD, D_EXP, weight=W1.tolist(), bias=b1.tolist()),
        EncGELU("act"),
        EncLinear("fc2", D_EXP, D_PAD, weight=W2.tolist()),
    )
    return rng, model, (W1, b1, W2, g, b)


def test_torch_mirror_matches_plaintext(toy):
    torch = pytest.importorskip("torch")
    rng, model, (W1, b1, W2, g, b) = toy
    x = rng.standard_normal((4, D_REAL)) * 0.3
    xp = np.concatenate([x, np.zeros((4, D_PAD - D_REAL))], axis=1)
    ln = (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True) + 1e-5) * g + b
    lnp = np.concatenate([ln, np.zeros((4, D_PAD - D_REAL))], axis=1)
    h = lnp @ W1 + b1
    gelu = 0.5 * h * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (h + 0.044715 * h ** 3)))
    ref = gelu @ W2
    out = model.torch_mirror()(torch.as_tensor(xp, dtype=torch.float32)).detach().numpy()
    assert np.linalg.norm(out - ref) / np.linalg.norm(ref) < 1e-4


def test_residency_protocol(toy):
    _, model, _ = toy
    assert model[0].residency() == ["cln.weight", "cln.bias"]
    assert model[1].residency() == ["fc1", "fc1_bias"]
    assert model[2].residency() is None
    assert model[3].residency() == ["fc2"]
    assert EncLinear("shared", D_PAD, D_PAD).residency() is None   # not owned: never evicted


def test_stage_construction_without_binding(toy):
    _, model, _ = toy
    s = Stage(compute=lambda x: x, weights=["a"], label="t")
    assert s.weights == ["a"] and s.state is None and s.loader is None
    with pytest.raises(TypeError):
        Stage()   # compute is required


def test_calibrate_fit_convert_parse(toy):
    pytest.importorskip("torch")
    from perseus.nn.calibrate import calibrate_sequential
    rng, model, _ = toy
    parsed = calibrate_sequential(model, rng.standard_normal((32, D_REAL)) * 0.3,
                                  apply=False, cheb_basis=True)
    assert "2" in parsed.softgelu and parsed.softgelu["2"].method == _core.GeLUMethod.THOR_COMPOSITE
    assert "0.ln" in parsed.norm


def test_contract_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("GPT2_FOLD_LN1", "1")
    monkeypatch.delenv("GPT2_FOLD_LN2", raising=False)
    c = contract.capture_contract()
    assert c["env"]["GPT2_FOLD_LN1"] == "1" and c["env"]["GPT2_FOLD_LN2"] is None
    f = tmp_path / "block_0_placement.json"
    f.write_text(json.dumps({"placements": []}))
    contract.stamp_file(f, c)
    assert contract.read_stamp(f) == c
    assert contract.validate_contract(contract.read_stamp(f)) == []
    monkeypatch.setenv("GPT2_FOLD_LN2", "1")   # unset -> set is a mismatch
    with pytest.raises(contract.PlanContractError):
        contract.validate_contract(contract.read_stamp(f), source=str(f))
    assert contract.validate_contract(None) == []   # unstamped plans pass


def test_session_profile_env_and_options(monkeypatch):
    from perseus.profile import SessionProfile
    for k in ("CHAIN", "AUTO_BTS_LEVEL", "SPARSE_AUTO", "FUSED_SM_DEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AUTO_BTS_LEVEL", "49")            # exported env wins
    p = SessionProfile.gpt2_decode_n32().apply()
    assert os.environ["CHAIN"] == "n32" and os.environ["AUTO_BTS_LEVEL"] == "49"
    # the preset carries the levers the paper's plan is bound to
    assert os.environ["SPARSE_AUTO"] == "2" and os.environ["FUSED_SM_DEN"] == "1"
    assert p.options().mode == _core.InferenceMode.Threaded


def test_error_taxonomy():
    from perseus.errors import FHEError, LayoutError, MaskError, PlanContractError, PlanError
    assert issubclass(PlanError, FHEError) and issubclass(MaskError, FHEError)
    assert issubclass(LayoutError, FHEError) and issubclass(PlanContractError, RuntimeError)
    with pytest.raises(FHEError):
        _core.throw_test()
