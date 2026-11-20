"""impl/fbs_fit.py on the CPU: the sine and trigonometric forms of a smooth functional bootstrap (fits, centring,
landings, the noise model's bound) and the impl.math candidates built on them."""
from __future__ import annotations

import math

import numpy as np
import pytest

from perseus.impl import fbs
from perseus.impl import fbs_fit as F

_pc = pytest.importorskip("perseus._pc")


def _bits(r, f, lo, hi):
    x = np.linspace(lo, hi, 20001)
    fm = np.abs(f(x)).max()
    return -math.log2(np.abs(F.eval_fit(r, x) - f(x)).max() / fm)


def test_trig_fit_beats_sine_at_half_the_degree():
    """A degree-n trigonometric series (P of degree n, Q of n - 1 in cos 2 pi u) approximates like a degree-2n sine
    series on an interior-singularity function (sigmoid), and far better on a domain end near a singularity (log on
    a positive range: the asin warp's branch point sits at the end)."""
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    t16 = _bits(F.fit_trig(sig, -8, 8, 16, cf=5), sig, -8, 8)
    s32 = _bits(F.fit_sine(sig, -8, 8, 32), sig, -8, 8)
    assert t16 > s32 - 1.0 and t16 > 20.0
    t16 = _bits(F.fit_trig(np.log, 1, 8, 16, cf=4), np.log, 1, 8)
    s32 = _bits(F.fit_sine(np.log, 1, 8, 32), np.log, 1, 8)
    assert t16 > s32 + 15.0


def test_trig_fit_coefficients_and_domain():
    r = F.fit_trig(np.exp, -2, 2, 12, cf=4)
    assert r["err"] < 1e-12 and len(r["post"]) == 13 and len(r["post_s"]) == 12
    # t = A cos(2 pi u) + off maps [cos 2 pi u_max, 1] onto [-1, 1]
    umax = 2.0 / 16
    assert r["amp"] * math.cos(2 * math.pi * umax) + r["off"] == pytest.approx(-1.0)
    assert r["amp"] + r["off"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="whole period"):
        F.fit_trig(np.exp, -2, 2, 12, cf=2)
    with pytest.raises(ValueError, match="turning point"):
        F.fit_sine(np.exp, -2, 2, 12, cf=3)


def test_register_trig_twin_centred():
    """The numpy twin reads the original input (the bootstrap reads x - center), the descriptor carries the
    trigonometric part and the landing comes from the trigonometric program."""
    from perseus.impl.ops import NumpyOps
    name = F.register_smooth("cpu.trig.log", np.log, 1.0, 8.0, form="trig", deg=12, cf=4)
    fn = fbs.FUNCS[name]
    assert fn.meta["center"] == 4.5 and fn.meta["form"] == "trig"
    x = np.linspace(1.0, 8.0, 101)
    assert np.abs(fbs.fbs(NumpyOps(), x, name) - np.log(x)).max() < 1e-6
    d = fbs.desc(name)
    assert d["trig"]["deg"] == 27 and d["trig"]["da"] == 5 and len(d["trig"]["post_s"]) == 12
    s = F.register_smooth("cpu.sine.log", np.log, 1.0, 8.0, form="sine", deg=12)
    sp = F.register_smooth("cpu.sinep.log", np.log, 1.0, 8.0, form="sine-paired", deg=12)
    assert "trig" not in fbs.desc(s) and "paired" not in fbs.desc(s) and fbs.desc(sp)["paired"]
    # a degree-n trig series lands where the paired sine form of degree n does: one composite level above the sine
    # form on FIDESlib's Paterson-Stockmeyer
    lv_t, lv_s, lv_p = fbs.landing(name)[0], fbs.landing(s)[0], fbs.landing(sp)[0]
    assert lv_t == lv_p == lv_s - 2
    low = fbs.landing(F.register_smooth("cpu.trig.log.low", np.log, 1.0, 8.0, form="trig", deg=12, cf=4,
                                        base="low"))[0]
    assert low == lv_t - 2


def test_error_bound_trig():
    """The bound is the fit plus z sigma; the bootstrap's own noise at zero input noise, the caller's input noise
    through the derivative otherwise; the overflow is the K32 table's."""
    name = F.register_smooth("cpu.trig.exp", np.exp, -2.0, 2.0, form="trig", deg=12, cf=4)
    b0 = fbs.error_bound(name, sigma_in=0.0)
    b1 = fbs.error_bound(name, sigma_in=1e-5)
    assert b0["approx"] < 1e-8
    assert 1e-8 < b0["sigma"] < 1e-5
    assert b1["sigma"] >= 0.99 * math.exp(2.0) * 1e-5 and b1["bound"] > b0["bound"]   # e^x at the top, input noise
    assert b0["log2_overflow"] == pytest.approx(-291.06, abs=0.01)


def test_register_auto_picks_shallowest():
    name = F.register_smooth("cpu.auto.log", np.log, 1.0, 8.0, err=1e-4, sigma_in=0.0)
    m = fbs.FUNCS[name].meta
    assert m["form"] == "trig" and m["bound"] <= 1e-4
    with pytest.raises(ValueError, match="no form meets"):
        F.register_smooth("cpu.auto.tight", np.log, 1.0, 8.0, err=1e-9)
