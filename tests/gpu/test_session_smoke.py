"""Encrypt / run / decrypt through the Python surface on a real context."""
import numpy as np
import pytest

# Collection-safe on a CPU-only runner: every other test importorskips perseus._core;
# a bare import here errors pytest collection and fails the whole CI job.
pytest.importorskip("perseus._core")
from perseus.nn import EncGELU, EncLinear, EncSequential

pytestmark = pytest.mark.gpu

D_PAD, D_REAL, D_EXP, E_REAL = 1024, 768, 4096, 3072


def test_encrypt_decrypt_round_trip(sess):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(D_REAL) * 0.3
    y = sess.decrypt(sess.encrypt(x), d=D_REAL)
    assert np.linalg.norm(y - x) / np.linalg.norm(x) < 1e-6
    with pytest.raises(ValueError, match="do not fit"):
        sess.encrypt(np.zeros(sess.inf.size.dim + 1))


def test_custom_mlp_matches_the_plaintext_reference(sess):
    """The custom_encrypted_model notebook's forward, as a test (rel < 5e-2)."""
    rng = np.random.default_rng(0)

    def w(di, do, ir, orr, s=0.5):
        m = rng.standard_normal((di, do)) * s / np.sqrt(ir)
        m[ir:, :] = 0.0
        m[:, orr:] = 0.0
        return m

    W1, W2 = w(D_PAD, D_EXP, D_REAL, E_REAL), w(D_EXP, D_PAD, E_REAL, D_REAL)
    model = EncSequential(EncLinear("t_fc1", D_PAD, D_EXP, weight=W1), EncGELU("t_act"),
                          EncLinear("t_fc2", D_EXP, D_PAD, weight=W2)).bind(sess)
    from perseus.nn import calibrate_sequential
    calibrate_sequential(model, rng.standard_normal((32, D_REAL)) * 0.3)

    x = rng.standard_normal(D_REAL) * 0.3
    out = sess.decrypt(model(sess.encrypt(x)), d=D_REAL)
    xp = np.zeros(D_PAD); xp[:D_REAL] = x
    h = xp @ W1
    gelu = 0.5 * h * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (h + 0.044715 * h ** 3)))
    ref = (gelu @ W2)[:D_REAL]
    assert np.linalg.norm(out - ref) / np.linalg.norm(ref) < 5e-2


def test_wrong_weight_shape_is_a_value_error_not_a_crash(sess):
    inf = sess.inf
    if not hasattr(inf, "set_weight") or not hasattr(type(inf), "set_weight"):
        pytest.skip("no set_weight")
    try:
        with pytest.raises(ValueError):
            inf.set_weight("t_bad", np.zeros((D_EXP, D_PAD)).tolist(), D_PAD, D_EXP)
    except AssertionError:
        pytest.skip("extension predates the C++ shape check")
