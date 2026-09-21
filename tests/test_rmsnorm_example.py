"""examples/rmsnorm_from_primitives.py on the CPU: the Chebyshev fit, the numpy mirror
against the closed form, the out-of-interval guard, the interval fold the FHE path relies
on, the gamma slot layout, and the whole primitive call sequence against a 64-slot numpy
fake session (so the single GPU run in tests/gpu/test_primitives.py only adds noise).
"""
import importlib.util
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.polynomial import chebyshev as cheb


def _load():
    path = Path(__file__).resolve().parents[1] / "examples" / "rmsnorm_from_primitives.py"
    spec = importlib.util.spec_from_file_location("rmsnorm_from_primitives", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ex = _load()

UNIT = 2      # primes per CKKS level on the fake (the n32 chain's composite_degree)


# ---- a numpy fake of the session surface the example uses -----------------------------

class _Ct:
    def __init__(self, vec, level, packing="cachemir"):
        self.vec = np.asarray(vec, dtype=np.float64)
        self.level = int(level)
        self.packing = packing


class _Fhe:
    """Slot-vector arithmetic with the runtime's level accounting: every product costs UNIT
    primes, adds and rotations none; rotate(ct, k) is out[i] = in[i + k] (ops.cu)."""

    def __init__(self, rot_steps, limit):
        self.loaded_rot_steps = list(rot_steps)
        self._limit = limit
        self.calls = []

    def level_limit(self):
        return self._limit

    def bootstrap_output_level(self):
        return 34

    def square(self, a):
        self.calls.append("square")
        return _Ct(a.vec * a.vec, a.level + UNIT, a.packing)

    def mult(self, a, b):
        if isinstance(b, _Ct):
            self.calls.append("mult_cc")
            return _Ct(a.vec * b.vec, max(a.level, b.level) + UNIT, a.packing)
        self.calls.append("mult_sc")
        return _Ct(a.vec * float(b), a.level + UNIT, a.packing)

    def add(self, a, b):
        if isinstance(b, _Ct):
            return _Ct(a.vec + b.vec, max(a.level, b.level), a.packing)
        return _Ct(a.vec + float(b), a.level, a.packing)

    def rotate(self, a, steps):
        assert steps in self.loaded_rot_steps, f"no rotation key for {steps}"
        self.calls.append(f"rotate{steps}")
        return _Ct(np.roll(a.vec, -steps), a.level, a.packing)


class _Inf:
    def __init__(self, fhe, slots, dim, hid_dim):
        self.fhe = fhe
        self.slots = slots
        self.size = SimpleNamespace(dim=dim, hidDim=hid_dim)

    def sum_slots(self, ct, width):          # the ladder in src/bindings/ops.cu
        assert width >= 1 and width & (width - 1) == 0
        acc, step = ct, 1
        while step < width:
            acc = self.fhe.add(acc, self.fhe.rotate(acc, step))
            step <<= 1
        return acc

    def eval_chebyshev(self, ct, coeffs, a, b):   # src/primitives/polynomial.cu
        # the binding takes std::vector<double>: the example must pass a list of floats
        assert isinstance(coeffs, list) and all(type(c) is float for c in coeffs)
        n = len(coeffs) - 1
        while n > 0 and coeffs[n] == 0.0:
            n -= 1
        y = (2.0 * ct.vec - (a + b)) / (b - a)
        depth = 2 + math.ceil(math.log2(n)) if n >= 1 else 1
        self.fhe.calls.append(f"cheb{n}")
        return _Ct(cheb.chebval(y, coeffs), ct.level + depth * UNIT, ct.packing)

    def mult_pt(self, ct, values):
        values = np.asarray(values, dtype=np.float64)
        assert values.ndim == 1 and values.shape[0] <= self.slots
        full = np.zeros(self.slots)
        full[:values.shape[0]] = values          # shorter than the slot count is zero-filled
        self.fhe.calls.append("mult_pt")
        return _Ct(ct.vec * full, ct.level + UNIT, ct.packing)


def _fake_session(rot_steps=(1, 2, 4, 8, 16, 32), limit=49):
    inf = _Inf(_Fhe(rot_steps, limit), slots=64, dim=12, hid_dim=16)
    return SimpleNamespace(inf=inf,
                           options=SimpleNamespace(ckks=SimpleNamespace(composite_degree=UNIT)))


def _token_ct(x, inf, level=34, packing="cachemir"):
    """A fresh token encode: lane i at slot i*t, zeros elsewhere (encode_linear_input)."""
    t = inf.slots // inf.size.hidDim
    slots = np.zeros(inf.slots)
    slots[np.arange(len(x)) * t] = x
    return _Ct(slots, level, packing)


# ---- tests --------------------------------------------------------------------------------

@pytest.mark.parametrize("interval,degree,tol", [((0.04, 0.16), 7, 2e-4), ((0.5, 2.0), 15, 1e-7)])
def test_coefficients_fit_rsqrt(interval, degree, tol):
    c = ex.rsqrt_cheb_coeffs(interval, degree)
    assert len(c) == degree + 1 and all(type(v) is float for v in c)
    a, b = interval
    y = np.linspace(-1.0, 1.0, 4001)
    u = (a + b) / 2 + (b - a) / 2 * y
    rel = np.abs(cheb.chebval(y, c) - u ** -0.5) / u ** -0.5
    assert rel.max() < tol, rel.max()
    # c0 unhalved (the runtime's and numpy's convention): the series at the midpoint is
    # rsqrt(mid); a halved c0 would be off by c0/2 (~1.7 on the default window)
    assert cheb.chebval(0.0, c) == pytest.approx(1 / math.sqrt((a + b) / 2), rel=1e-3)


def test_chebyshev_convention_and_defaults():
    # the convention the runtime's GPU pin relies on: [0.5, 0, 0.5] on [-1, 1] is x^2
    y = np.linspace(-1.0, 1.0, 101)
    np.testing.assert_allclose(cheb.chebval(y, [0.5, 0.0, 0.5]), y ** 2, atol=1e-15)
    assert ex.DEFAULT_COEFFS == ex.rsqrt_cheb_coeffs(ex.DEFAULT_INTERVAL, ex.DEFAULT_DEGREE)
    assert len(ex.DEFAULT_COEFFS) == ex.DEFAULT_DEGREE + 1 == 8
    with pytest.raises(ValueError, match="degree"):
        ex.rsqrt_cheb_coeffs((0.04, 0.16), 0)
    for bad in [(0.0, 1.0), (0.2, 0.1), (-1.0, 1.0), (0.1, float("inf")), "ab"]:
        with pytest.raises(ValueError, match="interval"):
            ex.rsqrt_cheb_coeffs(bad, 3)


def test_mirror_matches_exact_inside_the_interval():
    rng = np.random.default_rng(0)
    d = 768
    worst = {7: 0.0, 15: 0.0}
    for scale in np.linspace(0.23, 0.36, 100):     # mean(x^2)+eps stays inside [0.04, 0.16]
        x = rng.standard_normal(d) * scale
        gamma = rng.standard_normal(d)
        exact = ex.rmsnorm_exact(x, gamma, 1e-5)
        for degree in worst:
            ref = ex.rmsnorm_ref(x, gamma, eps=1e-5, interval=(0.04, 0.16), degree=degree)
            worst[degree] = max(worst[degree], np.abs(ref - exact).max())
    # degree 7: series rel err 8.1e-5 on the window, measured worst 6.4e-4 on the payload;
    # degree 15: series 9e-9, measured 6.6e-8
    assert worst[7] < 2e-3, worst
    assert worst[15] < 1e-5, worst


def test_mirror_refuses_out_of_interval():
    rng = np.random.default_rng(1)
    gamma = np.ones(768)
    for scale in (1.0, 0.05):                       # mean(x^2) ~ 1 and ~ 0.0025
        x = rng.standard_normal(768) * scale
        with pytest.raises(ValueError, match="outside"):
            ex.rmsnorm_ref(x, gamma, eps=1e-5, interval=(0.04, 0.16), degree=7)
    # the closed form has no window
    assert np.isfinite(ex.rmsnorm_exact(x, gamma)).all()


def test_interval_fold_identity():
    """The one algebraic step the FHE path does differently from the mirror: feeding the
    SUM of squares S with the interval [(a-eps)d, (b-eps)d] must give the evaluator's
    affine map the same argument as mean(x^2)+eps with [a, b]."""
    rng = np.random.default_rng(2)
    d, eps, (a, b) = 768, 1e-5, (0.04, 0.16)
    S = rng.uniform(20.0, 130.0, 1000)
    a2, b2 = ex.fhe_interval((a, b), eps, d)
    y_fhe = (2 * S - (a2 + b2)) / (b2 - a2)
    y_ref = (2 * (S / d + eps) - (a + b)) / (b - a)
    np.testing.assert_allclose(y_fhe, y_ref, atol=1e-12, rtol=0)
    assert a2 == pytest.approx((a - eps) * d) and b2 == pytest.approx((b - eps) * d)


def test_depth_and_gamma_layout():
    assert [ex.rmsnorm_depth(k) for k in (1, 2, 3, 4, 7, 8, 15)] == [4, 5, 6, 6, 7, 7, 8]
    with pytest.raises(ValueError, match="degree"):
        ex.rmsnorm_depth(0)
    gamma = np.arange(1.0, 13.0)
    g = ex.gamma_to_slots(gamma, slots=64, t=4)
    assert g.shape == (64,)
    np.testing.assert_array_equal(g[np.arange(12) * 4], gamma)
    off = np.ones(64, bool)
    off[np.arange(12) * 4] = False
    assert np.all(g[off] == 0.0)
    with pytest.raises(ValueError, match="do not fit"):
        ex.gamma_to_slots(np.ones(17), slots=64, t=4)


def test_fhe_call_sequence_on_a_numpy_fake():
    sess = _fake_session()
    inf = sess.inf
    t = inf.slots // inf.size.hidDim
    assert t == 4
    rng = np.random.default_rng(6)
    d = 12
    x = rng.standard_normal(d) * 0.3
    gamma = rng.standard_normal(d)
    ct = _token_ct(x, inf)
    u = float(np.mean(x * x) + 1e-5)
    interval = (u / 2, 2 * u)
    assert ex.level_budget(sess, ct) == (49 - 34 - 1) // 2 == 7

    res = ex.rmsnorm(sess, ct, gamma, eps=1e-5, d=d, interval=interval, degree=7)

    out = res.ct.vec
    want = ex.rmsnorm_ref(x, gamma, eps=1e-5, interval=interval, degree=7)
    np.testing.assert_allclose(out[::t][:d], want, atol=1e-10, rtol=0)
    off = np.ones(inf.slots, bool)
    off[np.arange(d) * t] = False
    assert np.all(out[off] == 0.0)                      # nothing leaks off the lanes
    assert (res.level_before, res.level_after) == (34, 34 + UNIT * ex.rmsnorm_depth(7))
    assert res.degree == 7 and res.coeffs == ex.rsqrt_cheb_coeffs(interval, 7)
    assert inf.fhe.calls == ["square", "rotate1", "rotate2", "rotate4", "rotate8", "rotate16",
                             "rotate32", "cheb7", "mult_pt", "mult_cc"]
    with pytest.raises(ValueError, match="cachemir"):
        ex.rmsnorm(sess, _token_ct(x, inf, packing="diagonal"), gamma, eps=1e-5, d=d,
                   interval=interval, degree=7)


def test_guards_rotation_keys_arguments_and_level_budget(caplog):
    d = 12
    rng = np.random.default_rng(7)
    x = rng.standard_normal(d) * 0.3
    gamma = rng.standard_normal(d)
    interval = (0.02, 0.3)
    sess = _fake_session(rot_steps=(1, 2, 4, 8, 32))           # 16 missing
    ct = _token_ct(x, sess.inf)
    with pytest.raises(ValueError, match=r"not loaded: \[16\]"):
        ex.rmsnorm(sess, ct, gamma, eps=1e-5, d=d, interval=interval, degree=7)
    assert sess.inf.fhe.calls == []                             # refused before any op

    sess = _fake_session()
    for kwargs, msg in [(dict(gamma=gamma[:-1]), "gamma"), (dict(d=13), "d must"),
                        (dict(eps=-1.0), "eps"), (dict(degree=0), "degree"),
                        (dict(interval=(0.3, 0.02)), "interval")]:
        kw = dict(gamma=gamma, eps=1e-5, d=d, interval=interval, degree=7)
        kw.update(kwargs)
        with pytest.raises(ValueError, match=msg):
            ex.rmsnorm(sess, ct, kw.pop("gamma"), **kw)

    sess = _fake_session(limit=40)                              # (40 - 34 - 1) // 2 = 2 levels
    assert ex.level_budget(sess, ct) == 2
    with caplog.at_level(logging.WARNING, logger="rmsnorm_from_primitives"):
        ex.rmsnorm(sess, ct, gamma, eps=1e-5, d=d, interval=interval, degree=7)
    assert any("needs 7 levels but only 2 remain" in r.getMessage() for r in caplog.records)
