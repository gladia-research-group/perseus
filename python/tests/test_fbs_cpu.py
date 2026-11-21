"""impl/fbs.py on the CPU: the table / sign series, the failure model (ModRaise overflow, the variance model's p_fail
and error bounds) and the refusals (chains FBS is not validated on, unsupported series)."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from perseus.impl import fbs

_pc = pytest.importorskip("perseus._pc")


def test_overflow_exact():
    """Irwin-Hall over 2(h + 1) uniforms, per bootstrap (x 2^16): the K = 32 table at h = 32 is 2^-291.1; K = 16 (the
    plain bootstrap's table) 2^-23.4; heavier sparse secrets overflow more."""
    per_bts = lambda h, k: fbs.log2_overflow(h, k) + fbs.LOG_N      # noqa: E731
    assert per_bts(32, 32) == pytest.approx(-291.06, abs=0.01)
    assert per_bts(32, 16) == pytest.approx(-23.37, abs=0.01)
    assert per_bts(48, 32) == pytest.approx(-94.31, abs=0.01)
    assert per_bts(32, 40) == -math.inf


def test_table_series_hits_the_table():
    rng = np.random.default_rng(0)
    for t in (4, 16, 64):
        v = rng.integers(-8, 9, t + 1) / 8.0
        name = fbs.register_table(f"cpu.rand.{t}", v, check=False)
        j = np.arange(-(t // 2), t // 2 + 1)
        assert np.abs(fbs.FUNCS[name].f(j) - v).max() < 1e-12
        assert fbs.FUNCS[name].cf == int(math.log2(2 * t))
    with pytest.raises(ValueError, match="2\\^k \\+ 1"):
        fbs.register_table("cpu.bad", np.zeros(12))


def test_sign_fit():
    name = fbs.register_sign("cpu.sign", 0.0, 1.0, gap=0.1, check=False)
    assert fbs.sign_error(name, 0.1) < 3e-5


def test_numpy_ops_is_the_twin():
    from perseus.impl.ops import NumpyOps
    name = fbs.register_table("cpu.step.16", (np.arange(-8, 9) >= 0).astype(float), check=False)
    x = np.arange(-8, 9, dtype=float)
    assert np.abs(fbs.fbs(NumpyOps(), x, name) - (x >= 0)).max() < 1e-12
    # a one-entry vector is a public scalar on the mirror (the hinge's link at a known margin), not the twin's novec
    assert abs(fbs.fbs(NumpyOps(), np.array([3.0]), name)[0] - 1.0) < 1e-12


@pytest.mark.parametrize("t", [16, 64, 256])
def test_defaults_meet_128(t):
    """At the defaults (K = 32, h = 32, input noise SIGMA_IN) the Gaussian terms are negligible and p_fail per
    ciphertext is the overflow's 2^-291.1."""
    name = fbs.register_table(f"cpu.sign.{t}", np.sign(np.arange(-(t // 2), t // 2 + 1)))
    r = fbs.p_fail(name)
    assert r["ok"] and r["log2_ct"] == pytest.approx(-291.06, abs=0.01)
    assert r["log2_phase"] < -1e6 and r["log2_out"] < -1e6
    # the bootstrap's own phase noise: well below the half step 1/2 (input units) even at t = 256 (cf 9)
    assert r["phase_sigma"] < 1e-4
    assert r["approx"] < 1e-4


def test_sign_default_and_margin():
    name = fbs.register_sign("cpu.sign01", 0.0, 1.0, gap=0.1)
    r = fbs.p_fail(name)
    assert r["ok"] and r["margin"] == 0.1
    # a margin of a few input stds is a real failure probability
    s = r["phase_sigma"]
    m = fbs.p_fail(name, margin=5 * s)
    assert -30 < m["log2_phase"] < -15 and not m["ok"]


def test_warns_past_128():
    """The TFHE convention (lwe_warning): a registration whose p_fail per ciphertext is not below 2^-128 warns."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fbs.register_sign("cpu.sign.noisy", gap=0.1, sigma_in=0.02)
    assert any("not below 2^-128" in str(x.message) for x in w)
    name = fbs.register_table("cpu.sign.h", np.sign(np.arange(-8, 9)), check=False)
    assert not fbs.p_fail(name, h=64)["ok"]               # the overflow alone: 2^-61.6 per bootstrap at h = 64


def test_noise_grows_with_input_noise():
    name = fbs.register_table("cpu.sign.n", np.sign(np.arange(-8, 9)), check=False)
    a, b = fbs.noise(name, sigma_in=0.0), fbs.noise(name, sigma_in=1e-4)
    assert b["phase"] > a["phase"] > 0.0
    assert (b["sigma"] >= a["sigma"]).all()
    assert a["phase_own"] == b["phase_own"]


def test_smooth_error_bound():
    """exp on [-2, 2] (cf 4, the asin warp): the bound covers the fit and the noise at the stated tail; p_fail is for
    decisions only."""
    from numpy.polynomial import chebyshev as C
    X, cf = 2.0, 4
    A = 1.0 / math.sin(2 * math.pi * X / 2**cf)
    t = np.cos(np.pi * (np.arange(81) + 0.5) / 81)
    c = C.chebfit(t, np.exp(2**cf * np.arcsin(t / A) / (2 * math.pi)), 30)
    c[0] *= 2.0
    name = fbs.register("cpu.exp", np.exp, cf, A, c, xmax=X)
    b128, b40 = fbs.error_bound(name), fbs.error_bound(name, log2_tail=-40.0)
    assert b128["bound"] > b40["bound"] > b128["approx"] > 0.0
    assert b128["bound"] < 1e-3 and b128["log2_overflow"] < -290
    with pytest.raises(ValueError, match="error_bound"):
        fbs.p_fail(name)


def test_refusals():
    post = [0.0, 0.5, 0.0, 0.1]
    for preset in ("n64", "n32/lp4b3"):
        with pytest.raises(ValueError, match="validated on n32 / n32a4 / n32sph"):
            _pc.fbs_landing(preset, 1.0, post)
        with pytest.raises(ValueError, match="validated on n32 / n32a4 / n32sph"):
            _pc.fbs_noise(preset, 3, 1.0, post, [0.0])
    for preset in ("n32", "n32a4", "n32sph"):
        assert _pc.fbs_landing(preset, 1.0, post)[1] == 1
    with pytest.raises(ValueError, match="degree >= 2"):
        fbs.register("cpu.lin", lambda x: x, 3, 1.0, [0.0, 1.0])
