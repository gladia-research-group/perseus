"""The Python-authorable leaf primitives against numpy on a real session:
rotate, conjugate, negate, plaintext-vector mult/add, sum_slots, Chebyshev evaluation.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.gpu

D_REAL = 768


def _vec(rng, n=D_REAL, scale=0.3):
    return rng.standard_normal(n) * scale


def _dec(sess, ct, n=D_REAL):
    return sess.decrypt(ct, d=n)


def test_rotate_conjugate_negate(sess):
    from perseus import _core
    rng = np.random.default_rng(1)
    x = _vec(rng)
    ct = sess.encrypt(x)
    fhe = sess.inf.fhe
    # rotation by 1: the real lanes are interleaved with a stride, so compare on the
    # packed slot vector the session exposes through decrypt_slots (the research tap)
    slots = np.array(_core._debug.decrypt_slots(sess.inf, ct))
    rot = np.array(_core._debug.decrypt_slots(sess.inf, fhe.rotate(ct, 1)))
    np.testing.assert_allclose(rot, np.roll(slots, -1), atol=1e-5)
    np.testing.assert_allclose(_dec(sess, fhe.conjugate(ct)), x, atol=1e-5)   # real payload
    np.testing.assert_allclose(_dec(sess, fhe.negate(ct)), -x, atol=1e-5)


def test_plaintext_vector_operands(sess):
    from perseus import _core
    rng = np.random.default_rng(2)
    x, w, b = _vec(rng), _vec(rng, scale=1.0), _vec(rng, scale=0.5)
    ct = sess.encrypt(x)
    slots = np.array(_core._debug.decrypt_slots(sess.inf, ct))
    n = slots.shape[0]
    w_full = np.zeros(n); w_full[:len(w)] = w
    b_full = np.zeros(n); b_full[:len(b)] = b
    y = np.array(_core._debug.decrypt_slots(sess.inf, sess.inf.mult_pt(ct, w_full)))
    np.testing.assert_allclose(y, slots * w_full, atol=1e-4)
    z = np.array(_core._debug.decrypt_slots(sess.inf, sess.inf.add_pt(ct, b_full)))
    np.testing.assert_allclose(z, slots + b_full, atol=1e-5)


def test_sum_slots_and_chebyshev(sess):
    from perseus import _core
    rng = np.random.default_rng(3)
    x = _vec(rng)
    ct = sess.encrypt(x)
    slots = np.array(_core._debug.decrypt_slots(sess.inf, ct))
    s = np.array(_core._debug.decrypt_slots(sess.inf, sess.inf.sum_slots(ct, 4)))
    ref = sum(np.roll(slots, -k) for k in range(4))
    np.testing.assert_allclose(s, ref, atol=1e-4)
    # x^2 on [-1, 1] in the Chebyshev basis: T0/2 + T2/2
    y = np.array(_core._debug.decrypt_slots(sess.inf, sess.inf.eval_chebyshev(ct, [0.5, 0.0, 0.5], -1.0, 1.0)))
    np.testing.assert_allclose(y, slots ** 2, atol=1e-3)


def _load_rmsnorm_example():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "examples" / "rmsnorm_from_primitives.py"
    spec = importlib.util.spec_from_file_location("rmsnorm_from_primitives", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rmsnorm_example(sess):
    """The worked RMSNorm (examples/rmsnorm_from_primitives.py) against its numpy mirror,
    which applies the same Chebyshev series: the residual is FHE noise only."""
    from perseus import _core
    ex = _load_rmsnorm_example()
    rng = np.random.default_rng(4)
    x = _vec(rng)                       # mean(x^2)+eps = 0.092: inside the window [0.04, 0.16]
    gamma = _vec(rng, scale=1.0)
    ct = sess.encrypt(x)
    fhe = sess.inf.fhe
    # the deepest degree the reactive-bootstrap budget allows: 7 under custom_n32's
    # ceiling of 49 primes (34 -> 48), 2 when the shell exports AUTO_BTS_LEVEL=46
    budget = ex.level_budget(sess, ct)
    degree = max((k for k in (15, 7, 3, 2) if ex.rmsnorm_depth(k) <= budget), default=None)
    assert degree is not None, (f"no degree fits: level {ct.level}, limit {fhe.level_limit()}, "
                                f"budget {budget} levels")
    res = ex.rmsnorm(sess, ct, gamma, eps=ex.DEFAULT_EPS, d=D_REAL, interval=ex.DEFAULT_INTERVAL,
                     degree=degree)
    got = _dec(sess, res.ct)
    want = ex.rmsnorm_ref(x, gamma, eps=ex.DEFAULT_EPS, interval=ex.DEFAULT_INTERVAL,
                          degree=degree)
    # atol 1e-2: the output is x/rms * gamma (entries O(1), max ~10 for this draw). The
    # deg-2 pin above allows 1e-3 on values <= 0.1 (1% relative); 1e-2 on this output is the
    # same relative budget after the degree-7 series (sum |c_k| ~ 5) and two more ciphertext
    # products. Every structural failure this test exists for is O(1): a wrong stride
    # mis-sums the mean by up to t = 32x, a halved c0 shifts rsqrt by ~50%, a wrong interval
    # fold moves the argument off [-1, 1].
    np.testing.assert_allclose(got, want, atol=1e-2)
    # level report: levels were consumed, the reactive bootstrap did not fire, and the count
    # matches rmsnorm_depth within one unit (a deg-2 output's rescale is pending:
    # _core.realize_pending_rescale)
    assert res.level_before < res.level_after < fhe.level_limit(), res
    unit = sess.options.ckks.composite_degree
    expected = ex.rmsnorm_depth(degree) * unit
    consumed = res.level_after - res.level_before
    assert expected - unit <= consumed <= expected, (consumed, expected, unit)
    # packing: the mean is broadcast to every slot but the payload is zero off the lanes,
    # so the product stays zero there
    slots = np.array(_core._debug.decrypt_slots(sess.inf, res.ct))
    t = sess.inf.slots // sess.inf.size.hidDim
    off = np.ones(len(slots), bool)
    off[np.arange(D_REAL) * t] = False
    assert np.abs(slots[off]).max() < 1e-2
