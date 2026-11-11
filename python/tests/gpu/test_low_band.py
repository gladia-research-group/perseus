"""A chain with two low-P tiers (`n32/lp4b3/lp2b1`) and a key source with rotation keys at two bands
(`KeySource(low_steps=, low_band=)`).

Key switches at the bottom limb tops (<= 1) run over 2 special primes, those at tops 2..3 over 4: the low steps' keys
(and conj) are banded at top 1 and rotate bottom ciphertexts only, the other steps and relin stay at band 3 and serve
levels above the bottom. Gates: every rotation, conjugation and product decrypts to its plaintext; the batched and
tree rotations equal `rotate` word for word; a key used at the other tier is refused; the bundle is smaller than the
one-tier one.

    tools/gpu.sh out/low_band.log .venv/bin/python -m pytest -m gpu python/tests/gpu/test_low_band.py -q -s
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

_pc = pytest.importorskip("perseus._pc")

LOW, TOP, BAND = 52, 50, 3
LOW_STEPS, TOP_STEPS = [1, 2, 3], [5, 7]
TOL = 1e-4


def _session(chain, tmp, **low):
    cfg = _pc.Config.preset("n32_gpt2_decode")
    cfg.set("bts.auto_bts_level", LOW)
    cfg.set("chain.preset", chain)
    sec = _pc.ClientSecret.generate(chain=chain, seed=5, sparse_h=72)
    cl = _pc.Client(sec, chain=chain, complex=True, seed=6)
    ks = _pc.KeySource.generate(LOW_STEPS + TOP_STEPS, groups=[[1, 2]], conj=True, seed=3, secret=sec.to_bytes(),
                                sparse_out=True, band=BAND, lean=True, layout="packed", **low)
    kb = os.path.join(tmp, chain.replace("/", "_") + ".pckb")
    cl.keygen(_pc.Session.key_manifest(cfg, ks), kb, threads=8)
    return cfg, ks, kb, cl


@pytest.fixture(scope="module")
def setup():
    tmp = tempfile.mkdtemp(prefix="lowband_", dir=os.environ.get("PC_WORK_DIR"))
    cfg, ks, kb, cl = _session("n32/lp4b3/lp2b1", tmp, low_steps=LOW_STEPS, low_band=1)
    s = _pc.Session.from_bundle(cfg, ks, kb, size={"lowPool": 32, "lowRows": 2}, pool=64, pt_slots=8).complete_setup()
    yield dict(s=s, cl=cl, n=cl.slots, kb=kb, tmp=tmp)
    s.close()


def _vals(seed, n):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / 4


def _dec(S, x):
    return np.asarray(S["cl"].decrypt(_pc.export_ct(S["s"], x))[0])


def _limbs(f, x):
    d = f.export_limbs(x)
    return np.concatenate([d["c0"].ravel(), d["c1"].ravel()])


def test_bottom_rotations(setup):
    s, f, cl, n = setup["s"], setup["s"].fhe, setup["cl"], setup["n"]
    a = _vals(0, n)
    x = _pc.import_ct(s, cl.encrypt(a, LOW))
    ref = {}
    for r in LOW_STEPS:
        y = f.rotate(x, r)
        ref[r] = _limbs(f, y)
        assert np.abs(_dec(setup, y) - np.roll(a, -r)).max() < TOL, r
    assert np.abs(_dec(setup, f.conjugate(x)) - a.conj()).max() < TOL
    xs = [x] + [_pc.import_ct(s, cl.encrypt(_vals(i, n), LOW)) for i in (1, 2)]
    for batch in (None, 1, 2):
        got = f.rotate_many(xs, 3, batch=batch)
        assert np.array_equal(_limbs(f, got[0]), ref[3]) and all(np.array_equal(_limbs(f, y), _limbs(f, f.rotate(z, 3))) for y, z in zip(got, xs))
    tree = f.rotate_tree(x, [[1, 2]])                           # offsets 0, 1, 2
    assert np.array_equal(_limbs(f, tree[1]), ref[1]) and np.array_equal(_limbs(f, tree[2]), ref[2])
    for o, y in enumerate(f.rotate_tree(x, [[1, 2]], low=True)):
        assert np.abs(_dec(setup, y) - np.roll(a, -o)).max() < TOL, o


def test_upper_tier(setup):
    s, f, cl, n = setup["s"], setup["s"].fhe, setup["cl"], setup["n"]
    a, b = _vals(3, n), _vals(4, n)
    x, y = _pc.import_ct(s, cl.encrypt(a, TOP)), _pc.import_ct(s, cl.encrypt(b, TOP))
    for r in TOP_STEPS:
        assert np.abs(_dec(setup, f.rotate(x, r)) - np.roll(a, -r)).max() < TOL, r
    with f.suppress_auto_bts():
        z = f.mult(x, y)
        f.realize(z)
    assert np.abs(_dec(setup, z) - a * b).max() < TOL


def test_wrong_tier_refused(setup):
    s, f, cl, n = setup["s"], setup["s"].fhe, setup["cl"], setup["n"]
    lo, hi = _pc.import_ct(s, cl.encrypt(_vals(5, n), LOW)), _pc.import_ct(s, cl.encrypt(_vals(6, n), TOP))
    with pytest.raises(Exception):
        f.rotate(lo, TOP_STEPS[0])                              # a band-3 key (4 special primes) at top 1
    with pytest.raises(Exception):
        f.rotate(hi, LOW_STEPS[0])                              # a top-1 key at top 3
    with pytest.raises(ValueError):
        _pc.KeySource.generate([1, 5], band=BAND, lean=True, low_steps=[1])
    with pytest.raises(ValueError):
        _pc.KeySource.generate([1, 5], band=BAND, lean=True, low_steps=[1], low_band=BAND)


def test_bundle_smaller(setup):
    _, _, kb, _ = _session("n32/lp4b3", setup["tmp"])
    one, two = _pc.bundle_info(kb)["bytes"], _pc.bundle_info(setup["kb"])["bytes"]
    assert two < 0.8 * one, (one, two)
