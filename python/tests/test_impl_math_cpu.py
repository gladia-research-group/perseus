"""impl/math.py: the method choice meets its error target on the domain (float64), the counted depth is the depth the
CPU twin spends, domains and impossible targets are refused, and the sign family (CKKS composites, TFHE switch) is
right on the plaintext mirror and the twin."""
import numpy as np
import pytest

from perseus.impl import math as M
from perseus.impl.ops import NumpyOps

CASES = [
    ("exp", -8.0, 0.0, 1e-4, {}), ("exp", -30.0, 0.0, 1e-3, {}), ("exp", -1.0, 1.0, 1e-6, {}),
    ("exp", -4.0, 4.0, 1e-4, {"out_scale": float(np.exp(-4.0))}),
    ("log", 1.0, 100.0, 1e-4, {}), ("log", 0.01, 100.0, 5e-2, {}), ("log", 0.5, 2.0, 1e-6, {}),
    ("rsqrt", 0.01, 1.0, 1e-4, {}), ("rsqrt", 0.0613, 0.9735, 1e-5, {}), ("rsqrt", 1e-3, 10.0, 2e-2, {}),
    ("reciprocal", 0.1, 1.0, 1e-4, {}), ("reciprocal", 1.0, 256.0, 1e-3, {}), ("reciprocal", 0.231, 27.55, 1e-4, {}),
    ("sqrt", 0.0, 1.0, 3e-3, {}), ("sqrt", 0.1, 10.0, 1e-4, {}),
    ("tanh", -4.0, 4.0, 1e-4, {}), ("tanh", -1.0, 3.0, 1e-4, {}), ("sigmoid", -8.0, 8.0, 1e-4, {}),
    ("erf", -3.0, 3.0, 1e-5, {}),
    ("pow", 0.1, 2.0, 1e-4, {"p": 1.5}), ("pow", 0.5, 4.0, 1e-4, {"p": -0.3}), ("pow", -2.0, 2.0, 1e-6, {"p": 5}),
]

REF = {"exp": np.exp, "log": np.log, "rsqrt": lambda x: 1 / np.sqrt(x), "reciprocal": lambda x: 1 / x,
       "sqrt": np.sqrt, "tanh": np.tanh, "sigmoid": lambda x: 1 / (1 + np.exp(-x))}


def _ref(fn, x, kw):
    if fn == "erf":
        from scipy.special import erf
        return erf(x)
    if fn == "pow":
        return x ** kw["p"]
    return kw.get("out_scale", 1.0) * REF[fn](x) if fn == "exp" else REF[fn](x)


@pytest.mark.parametrize("fn,lo,hi,err,kw", CASES)
def test_target_met_on_domain(fn, lo, hi, err, kw):
    a = M.approx(fn, lo, hi, err, **kw)
    assert a.err <= a.err_enc <= err and (a.depth >= 1 or a.method == "squarings")
    x = np.random.default_rng(1).uniform(lo, hi, 20000)
    x[:2] = lo, hi
    y = a.numpy(x)
    t = _ref(fn, x, kw)
    e = np.abs(y - t) / (np.abs(t) if a.rel else 1.0)
    assert e.max() <= err * 1.001, (a.method, e.max())


def test_cheapest_and_candidates():
    """The choice is the feasible candidate with the fewest levels; every candidate's own numbers are honest."""
    cs = M.candidates("reciprocal", 0.231, 27.55, 1e-4)
    assert {c.method for c in cs} >= {"cheb", "cheb+newton", "goldschmidt", "rsqrt^2"}
    ok = [c for c in cs if c.err_enc <= 1e-4]
    best = M.approx("reciprocal", 0.231, 27.55, 1e-4)
    assert (best.depth, best.ct_mults) == min((c.depth, c.ct_mults) for c in ok)
    for c in cs:
        x = M.grid(0.231, 27.55)
        assert np.max(np.abs(c.numpy(x) * x - 1.0)) == pytest.approx(c.err, rel=1e-6)


def test_exp_squaring_beats_cheb_on_wide_domains():
    a = M.approx("exp", -60.0, 0.0, 1e-3)
    cheb = [c for c in M.candidates("exp", -60.0, 0.0, 1e-3) if c.method == "cheb"]
    assert a.method == "cheb+sq"
    assert not cheb or (a.depth, a.ct_mults) <= (cheb[0].depth, cheb[0].ct_mults)


def test_parity_halves_the_series():
    a = M.approx("tanh", -4.0, 4.0, 1e-4)
    b = M.approx("tanh", -4.0, 4.0 + 1e-9, 1e-4)        # not symmetric: the full series
    assert dict(a.params).get("parity") == "odd" and a.ct_mults <= b.ct_mults


@pytest.mark.parametrize("fn,lo,hi", [("log", 0.0, 1.0), ("rsqrt", -1.0, 1.0), ("reciprocal", 0.0, 2.0),
                                      ("sqrt", -0.1, 1.0), ("exp", 1.0, 1.0), ("pow", 0.0, 1.0)])
def test_domain_refused(fn, lo, hi):
    with pytest.raises(ValueError):
        M.approx(fn, lo, hi, 1e-3, **({"p": -0.5} if fn == "pow" else {}))


def test_impossible_target():
    with pytest.raises(M.TargetError):
        M.approx("sqrt", 0.0, 1.0, 1e-9)


def test_noise_model():
    """A deep, high-degree series refreshed inside amplifies the refresh's error (noise_gain x NOISE_BTS, measured on
    the device: rsqrt degree 318 on [1e-3, 10] read a relative 2.8e-2): such a choice is refused at a tight target;
    a shallow one (no refresh inside) is held to its float64 error."""
    with pytest.raises(M.TargetError):
        M.approx("rsqrt", 1e-3, 10.0, 1e-3)
    a = M.approx("rsqrt", 1e-3, 10.0, 2e-2)
    assert a.depth > M.BUDGET and a.err_enc <= 2e-2
    shallow = M.approx("exp", -8.0, 0.0, 1e-4)
    assert shallow.depth <= M.BUDGET and shallow.err_enc == pytest.approx(shallow.err, rel=1e-3)



def test_pow_integer_exact():
    a = M.approx("pow", -3.0, 3.0, 1e-12, p=7)
    assert a.method == "squarings" and a.depth == 3 and a.err < 1e-12


@pytest.mark.parametrize("eps", [2.0 ** -6, 2.0 ** -10])
def test_sign_ladder(eps):
    lad = M.sign_ladder(eps, eps)
    x = np.concatenate([np.linspace(eps, 1.0, 5000), np.geomspace(eps, 1.0, 5000)])
    y = M.sign_ckks(NumpyOps(), np.concatenate([x, -x, [0.0]]), lad)
    assert np.abs(y[: 2 * len(x)] - np.concatenate([np.ones_like(x), -np.ones_like(x)])).max() <= eps
    assert y[-1] == 0.0                             # odd stages: an exact zero stays zero
    assert np.all(np.abs(y) <= 1.0 + eps)


@pytest.mark.parametrize("eps", [1.2e-4, 4.9e-4, 2e-3])
def test_sign_ladder_survives_refresh_noise(eps):
    """Noise of 3e-3 (the dense route's worst unit-size refresh error) between stages keeps every output within the
    ladder's delta: each stage past the first is fitted LADDER_MARGIN beyond its input's image (without the margin a
    1e-3 perturbation took these ladders to 1e11..1e32, the device's 1e19)."""
    from numpy.polynomial import chebyshev as C
    lad = M.sign_ladder(eps, min(eps, 2.0 ** -10))
    rng = np.random.default_rng(0)
    x = np.concatenate([np.linspace(-1, 1, 20001), np.geomspace(eps, 1, 2000)])
    y = x
    for i, c in enumerate(lad.coefs):
        if i:
            y = y + 3e-3 * rng.standard_normal(y.shape)
        y = C.chebval(y, c)
    assert np.abs(y).max() <= 1.0 + 2 * lad.delta + 1e-9


@pytest.mark.parametrize("method", ["ckks", "tfhe"])
def test_sign_family_mirror(method):
    ops = NumpyOps()
    rng = np.random.default_rng(0)
    N = 1 << 15
    x = rng.uniform(-1.0, 1.0, N)
    x[np.abs(x) < 0.02] = 0.5                       # outside the TFHE noise band and the CKKS gap
    y = rng.uniform(-1.0, 1.0, N)
    tol = 2e-3 if method == "ckks" else 1e-9
    np.testing.assert_allclose(np.real(M.relu(ops, x, 1.0, 1e-3, method)), np.maximum(x, 0), atol=tol)
    np.testing.assert_allclose(np.real(M.abs(ops, x, 1.0, 1e-3, method)), np.abs(x), atol=tol)
    np.testing.assert_allclose(np.real(M.sign(ops, x, 1.0, 1e-3, method)), np.sign(x), atol=tol)
    d = x - y
    keep = np.abs(d) > 0.02
    np.testing.assert_allclose(np.real(M.maximum(ops, x, y, 2.0, 1e-3, method))[keep], np.maximum(x, y)[keep], atol=tol)
    np.testing.assert_allclose(np.real(M.minimum(ops, x, y, 2.0, 1e-3, method))[keep], np.minimum(x, y)[keep], atol=tol)
    c = np.real(M.clip(ops, x, -0.3, 0.4, 1.0, 1e-3, method))
    keep = (np.abs(x + 0.3) > 0.02) & (np.abs(x - 0.4) > 0.02)
    np.testing.assert_allclose(c[keep], np.clip(x, -0.3, 0.4)[keep], atol=2 * tol)


def test_relu_ckks_error_bound():
    """approx_relu's error: within err over the whole domain, the gap included (|x| err at most there)."""
    for bound, err in [(1.0, 1e-3), (8.0, 1e-2)]:
        _lad, depth, cm, e = M.approx_relu(bound, err)
        assert e <= err * 1.01 and depth >= 1 and cm >= 1


# ── the CPU twin: levels and values ─────────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def twin():
    from perseus.examples.gpt2 import env
    from perseus.impl import fakecap
    from perseus.impl.layout import Dims
    from perseus.impl.ops import FheOps
    over = {"AUTO_BTS_LEVEL": "50"}
    env.export_env(None, chain="n32", CKKS_COMPLEX="1", **over)
    fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over})
    _, inf = fakecap.make_session(Dims.gpt2(), {1, 2, 4}, limit=200, bts_out=32)
    ops = FheOps(inf, fakecap.core)

    def fresh(v):
        z = ops.core.encode_token_input(ops.inf, np.zeros(Dims.gpt2().dim))
        return ops.add_pt(z, v)
    return ops, fresh


@pytest.mark.parametrize("fn,lo,hi,err,kw", [c for c in CASES if c[0] in ("exp", "log", "rsqrt", "reciprocal",
                                                                          "tanh", "sqrt")][:10])
def test_twin_depth_and_value(twin, fn, lo, hi, err, kw):
    ops, fresh = twin
    a = M.approx(fn, lo, hi, err, **kw)
    x = np.random.default_rng(2).uniform(lo, hi, ops.inf.slots)
    c = fresh(x)
    l0 = ops.lvl(c)
    y = a(ops, c)
    assert ops.lvl(y) - l0 == ops.unit * a.depth
    out = np.real(ops.decrypt_slots_complex(y))
    np.testing.assert_allclose(out, a.numpy(x), rtol=1e-9, atol=1e-12)
