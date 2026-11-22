"""The digit decomposition (impl/tfhe_digits.py) on the numpy mirror (impl/tfhe.py `Prog.run`: window loads of the
phase, the negacyclic tables, the switch's input noise and twice it): large-precision signs exact at their resolution
(edges: zero, the resolution, the windows' rounding boundaries, the range ends), exact words with every digit compared
(ties: either rounding), the words feeding tfhe_bitwise ops, the impl.math entry points, and the noise model's
p_fail per read / lookup below 2^-128."""
from __future__ import annotations

import numpy as np
import pytest

from perseus.impl import tfhe, tfhe_bitwise, tfhe_digits as D


@pytest.fixture
def noise():
    old = tfhe._NOISE
    yield lambda v: setattr(tfhe, "_NOISE", v)
    tfhe._NOISE = old


def _sign_values(bits, rng, k=600):
    n = len(D.sign_windows(bits))
    edge = [0.0, 1e-12, -1e-12, 1.0, -1.0, D.sign_resolution(n), -D.sign_resolution(n)]
    for i in range(n):
        for j in range(-6, 6):
            b = (j + 0.5) / (4.0 * 8.0 ** i) / D.SIGN_SPAN
            edge += [b, b * (1 + 1e-9), b * (1 - 1e-9)]
    rnd = rng.uniform(-1, 1, k) * 2.0 ** -rng.integers(0, bits + 2, k)
    return np.concatenate([edge, rnd])


@pytest.mark.parametrize("bits", [12, 16, 20, 24])
@pytest.mark.parametrize("sd", [0.0, 5.3e-3, 1.06e-2])
def test_sign_exact(noise, bits, sd):
    noise(sd)
    x = _sign_values(bits, np.random.default_rng(bits))
    p = D.sign_prog("t", len(x), bits, "o")
    out = np.asarray(p.run(x * D.SIGN_SPAN, noise=0.0))
    dec = np.abs(x) >= D.sign_resolution(len(D.sign_windows(bits)))
    assert set(np.unique(out)) <= {-1.0, 1.0}
    assert not np.any((out != np.where(x >= 0, 1.0, -1.0)) & dec)
    assert D.sign_resolution(len(D.sign_windows(bits))) <= 2.0 ** -bits


def test_sign_rounds_and_model():
    a16, a20 = D.analyze_sign(16), D.analyze_sign(20)
    assert (a16["windows"], a20["windows"]) == (6, 7)
    assert a16["rounds"] == a20["rounds"] == 3                    # the load, one combine, the store
    assert max(a16["log2_pfail_read"], a20["log2_pfail_read"]) < -128
    for bits in (16, 20):
        p = D.sign_prog("t", 4, bits, "o")
        an = tfhe_bitwise.analyze(p, np.linspace(-1.7, 1.7, 4))
        assert an["log2_pfail_lookup"] < -128


def _word_values(nd, rng, k=400):
    ulp = 2.0 ** -(2 * nd - 1)
    kk = rng.integers(-(2 ** (2 * nd - 1)), 2 ** (2 * nd - 1) - 1, 40)
    edge = [0.0, ulp / 2, -ulp / 2, ulp, -ulp, -1.0, 1.0 - ulp]
    return np.concatenate([edge, kk * ulp, (kk + 0.5) * ulp, rng.uniform(-1, 1, k) * 2.0 ** -rng.integers(0, 2 * nd, k)])


@pytest.mark.parametrize("nd", [1, 2, 4, 8, 10, 16])
@pytest.mark.parametrize("sd", [0.0, 5.3e-3, 1.06e-2])
def test_word_exact(noise, nd, sd):
    noise(sd)
    x = _word_values(nd, np.random.default_rng(nd))
    p = D.word_prog("t", len(x), nd, "o")
    got = np.rint(np.asarray(p.run(x, noise=0.0)) * 4).astype(np.int64).reshape(len(x), nd)
    y = x * 2.0 ** (2 * nd - 1)
    # either rounding within the reads' reach of a tie (the mirror's grid 2^-10 of a native unit, the noise's tail)
    tie = np.abs(y - np.floor(y) - 0.5) < 0.01 + 30.0 * sd
    want = D.word_reference(x, nd)
    alt = D.word_reference(np.floor(y) / 2.0 ** (2 * nd - 1), nd)
    alt2 = D.word_reference((np.floor(y) + 1) / 2.0 ** (2 * nd - 1), nd)
    ok = np.all(got == want, 1) | (tie & (np.all(got == alt, 1) | np.all(got == alt2, 1)))
    assert ok.all(), x[~ok][:4]


def test_word_model():
    for nd in (8, 10):
        assert D.analyze_word(nd)["log2_pfail_read"] < -128
        p = D.word_prog("t", 8, nd, "o")
        an = tfhe_bitwise.analyze(p, np.linspace(-1, 0.99, 8))
        assert an["log2_pfail_lookup"] < -128


@pytest.mark.parametrize("op", ["add", "sub", "cmp", "xor"])
def test_values_load_feeds_word_ops(noise, op):
    noise(5.3e-3)
    rng = np.random.default_rng(5)
    nd, w = 8, 48
    W = 2 * nd
    a, b = rng.integers(0, 1 << W, w), rng.integers(0, 1 << W, w)
    a[:3], b[:3] = [0, (1 << W) - 1, 1 << (W - 1)], [(1 << W) - 1, 0, (1 << (W - 1)) - 1]
    sv = lambda u: np.where(u >= 1 << (W - 1), u - (1 << W), u) / 2.0 ** (W - 1)
    wp = tfhe_bitwise.WordProg("t", w, nd, load="values")
    if op == "xor":
        wp.bitwise("xor", "o")
    else:
        getattr(wp, op)("o")
    assert wp.rows_in() == 2 * w
    out = np.asarray(wp.pr.run(np.concatenate([sv(a), sv(b)]), noise=0.0))
    if op == "cmp":
        got, want = np.rint(out).astype(np.int64), (a < b).astype(np.int64)
    else:
        d = np.rint(out * 4).astype(np.int64).reshape(w, nd) % 4
        got = (d * (4 ** np.arange(nd))).sum(1)
        want = {"add": (a + b) % (1 << W), "sub": (a - b) % (1 << W), "xor": a ^ b}[op]
    assert np.array_equal(got, want)


def test_impl_math_sign_digit():
    from perseus.impl import math as M
    from perseus.impl.ops import NumpyOps
    rng = np.random.default_rng(2)
    x = rng.uniform(-3, 3, 64) * 2.0 ** -rng.integers(0, 18, 64)
    x[:3] = (0.0, 3.0, -3.0)
    s = np.real(M.sign(NumpyOps(), x, 3.0, 2.0 ** -16, method="digit", slots=np.arange(48)))
    keep = np.abs(x[:48]) >= 2.0 ** -16 * 3.0
    assert np.array_equal(s[:48][keep], np.where(x[:48] >= 0, 1.0, -1.0)[keep])
    assert np.all(s[48:] == 0.0)
    r = np.real(M.relu(NumpyOps(), x, 3.0, 2.0 ** -16, method="digit"))
    assert np.allclose(r, np.maximum(x, 0.0), atol=1e-12)
    a = np.real(M.abs(NumpyOps(), x, 3.0, 2.0 ** -16, method="digit"))
    assert np.allclose(a, np.abs(x), atol=1e-12)
    # the latency model: the digit sign at 2^-16 beats the composite (depth 22, 118 products) at batch 1
    assert M.sign_method(2.0 ** -16, 1) == "digit"
    assert M.sign_est_ms("digit", 2.0 ** -16, 1) < M.sign_est_ms("ckks", 2.0 ** -16, 1)


def test_sort_step_digit():
    from perseus.impl import sort
    from perseus.impl.ops import NumpyOps
    rng = np.random.default_rng(4)
    n = 256
    d = rng.uniform(-2, 2, n) * 2.0 ** -rng.integers(0, 20, n)
    heads = np.arange(0, n, 2)
    sc = np.where(heads % 4 == 0, 1.0, -1.0)
    st = np.real(sort.step(NumpyOps(), d, heads, sc, method="digit", bound=2.0, res=2.0 ** -20, tag="t"))
    keep = np.abs(d[heads]) >= 2.0 ** -20
    assert np.array_equal(st[heads][keep], (sc * d[heads] >= 0).astype(float)[keep])
    assert np.all(st[np.setdiff1d(np.arange(n), heads)] == 0.0)
