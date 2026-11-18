"""Low ciphertext slots (`size={"lowPool", "lowRows"}`, `import_ct(..., low=True)`).

A low slot holds only Q rows 0..lowRows-1 of one ciphertext. Gates: `mult_sum` over low operands exports the same
bytes as over the same frames in full slots; every other op refuses a low ciphertext; the slots are their size.

    tools/gpu.sh out/low_slots.log .venv/bin/python -m pytest -m gpu python/tests/gpu/test_low_slots.py -q -s
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

_pc = pytest.importorskip("perseus._pc")

LEVEL = 50                     # q0 + one level: 4 Q rows
ROWS = 4
NLOW = 32


@pytest.fixture(scope="module")
def setup():
    cfg = _pc.Config.preset("n32_gpt2_decode")
    cfg.set("bts.auto_bts_level", 52)
    secret = _pc.ClientSecret.generate(seed=5)
    client = _pc.Client(secret, seed=6, complex=True)
    ks = _pc.KeySource.generate([1], conj=False, seed=7, secret=secret.to_bytes(), sparse_out=True)
    s = _pc.Session(cfg, ks, size={"lowPool": NLOW, "lowRows": ROWS}, pool=64, pt_slots=8).complete_setup()
    yield s, client
    s.close()


def frames(client, rng, n):
    v = (rng.standard_normal((n, 1 << 15)) + 1j * rng.standard_normal((n, 1 << 15))) / 8
    return v, [client.encrypt(x, LEVEL) for x in v]


def test_mult_sum_low_equals_full(setup):
    s, client = setup
    rng = np.random.default_rng(0)
    for n in (1, 2, 5, 10):
        xv, xf = frames(client, rng, n)
        yv, yf = frames(client, rng, n)
        full = _pc.export_ct(s, s.fhe.mult_sum([_pc.import_ct(s, f) for f in xf], [_pc.import_ct(s, f) for f in yf]))
        xl = [_pc.import_ct(s, f, low=True) for f in xf]
        yl = [_pc.import_ct(s, f, low=True) for f in yf]
        assert all(c.low for c in xl + yl)
        low = _pc.export_ct(s, s.fhe.mult_sum(xl, yl))
        assert low == full, n
        got = client.decrypt(low)[0]
        assert np.abs(got - (xv * yv).sum(0)).max() < 1e-6
        # mixed: low operands with a full one
        mixed = _pc.export_ct(s, s.fhe.mult_sum(xl, [_pc.import_ct(s, f) for f in yf]))
        assert mixed == full, n
        np.testing.assert_array_equal(client.decrypt(_pc.export_ct(s, xl[0]))[0], client.decrypt(xf[0])[0])
        del xl, yl


def test_mult_sum_many_equals_mult_sum(setup):
    """Groups sharing b (giant steps over one query) or a (queries over one corpus), 5 groups (two output launches),
    20 pairs (three pair launches), low and full operands: every output exports the bytes of its own mult_sum."""
    s, client = setup
    rng = np.random.default_rng(3)
    _, fa = frames(client, rng, 20)
    _, fb = frames(client, rng, 5)
    full = [_pc.import_ct(s, f) for f in fa[:20]]
    low = [_pc.import_ct(s, f, low=True) for f in fa[:20]]
    q = [_pc.import_ct(s, f) for f in fb]
    shared_b = ([low[i:i + 4] for i in range(0, 20, 4)], [q[:4]] * 5)
    shared_a = ([full[:4]] * 5, [[q[g]] * 4 for g in range(5)])
    for ga, gb in (shared_b, shared_a):
        got = s.fhe.mult_sum_many(ga, gb)
        for o, x, y in zip(got, ga, gb):
            assert _pc.export_ct(s, o) == _pc.export_ct(s, s.fhe.mult_sum(x, y))
    long = s.fhe.mult_sum_many([full, low], [low, full])
    assert _pc.export_ct(s, long[0]) == _pc.export_ct(s, s.fhe.mult_sum(full, low))
    assert _pc.export_ct(s, long[1]) == _pc.export_ct(s, s.fhe.mult_sum(low, full))


def test_compressed_export(setup):
    """export_ct(bits=32) of a product at the bottom (2 rows): half the frame, decrypting to within ~1e-4 of it."""
    s, client = setup
    rng = np.random.default_rng(4)
    xv, xf = frames(client, rng, 3)
    yv, yf = frames(client, rng, 3)
    p = s.fhe.mult_sum([_pc.import_ct(s, f, low=True) for f in xf], [_pc.import_ct(s, f) for f in yf])
    full, small = _pc.export_ct(s, p), _pc.export_ct(s, p, bits=32)
    assert len(small) < 0.51 * len(full)
    a, b = client.decrypt(full)[0], client.decrypt(small)[0]
    # the rounding of c1 is multiplied by the secret: ~2^-13 of the slots' scale at 32 bits
    assert np.abs(a - b).max() < 1e-3
    assert np.abs(b - (xv * yv).sum(0)).max() < 1e-3
    q = s.fhe.mult_sum([_pc.import_ct(s, f) for f in xf[:2]], [_pc.import_ct(s, f) for f in yf[:2]])
    assert _pc.export_ct_many(s, [p, q], 32) == [_pc.export_ct(s, p, bits=32), _pc.export_ct(s, q, bits=32)]
    with pytest.raises(Exception):
        _pc.export_ct(s, _pc.import_ct(s, xf[0]), bits=32)        # 4 rows: the modulus is too wide for 32 bits


def test_seeded_frames(setup):
    """Client.encrypt(seeded=True): half the frame; the client decrypts it, the server imports it (c1 expanded from
    its seeds) into full and low slots, and products of imported seeded frames match numpy."""
    s, client = setup
    rng = np.random.default_rng(5)
    v = (rng.standard_normal((2, 1 << 15)) + 1j * rng.standard_normal((2, 1 << 15))) / 8
    full, seeded = client.encrypt(v[0], LEVEL), client.encrypt(v[0], LEVEL, seeded=True)
    assert len(seeded) < 0.51 * len(full)
    assert np.abs(client.decrypt(seeded)[0] - v[0]).max() < 1e-6
    a = _pc.import_ct(s, seeded)
    assert np.abs(client.decrypt(_pc.export_ct(s, a))[0] - v[0]).max() < 1e-6
    b = _pc.import_ct(s, client.encrypt(v[1], LEVEL, seeded=True), low=True)
    p = s.fhe.mult_sum([b], [a])
    assert np.abs(client.decrypt(_pc.export_ct(s, p))[0] - v[0] * v[1]).max() < 1e-6


def test_sparse_export(setup):
    """export_ct_many(sparse=True): switched to the client's sparse secret, c0 / c1 at their own bits (bit-packed):
    24/30 bits matches the dense 32-bit error in 0.84 of its bytes; 20/26 (0.72 of them) stays within 2e-3."""
    s, client = setup
    rng = np.random.default_rng(6)
    xv, xf = frames(client, rng, 3)
    yv, yf = frames(client, rng, 3)
    p = s.fhe.mult_sum([_pc.import_ct(s, f) for f in xf], [_pc.import_ct(s, f) for f in yf])
    ref = client.decrypt(_pc.export_ct(s, p))[0]
    dense = _pc.export_ct_many(s, [p], 32)[0]
    for b0, b1, tol, size in ((24, 30, 2e-4, 0.85), (20, 26, 2e-3, 0.73)):
        fr = _pc.export_ct_many(s, [p], b0, bits1=b1, sparse=True)[0]
        assert len(fr) < size * len(dense)
        assert np.abs(client.decrypt(fr)[0] - ref).max() < tol, (b0, b1)


def test_other_ops_refuse_low(setup):
    s, client = setup
    rng = np.random.default_rng(1)
    _, (f,) = frames(client, rng, 1)
    a, b = _pc.import_ct(s, f, low=True), _pc.import_ct(s, f)
    f_ = s.fhe
    for op in (lambda: f_.rotate(a, 1), lambda: f_.add(a, b), lambda: f_.add(b, a), lambda: f_.mult(a, b),
               lambda: f_.mult(a, 2.0), lambda: f_.rotate_hoisted(a, [1]), lambda: f_.drop_to_level(a, LEVEL + 1),
               lambda: f_.mult_pt(a, np.ones(1 << 15)), lambda: f_.export_limbs(a), lambda: f_.inplace_add(b, a)):
        with pytest.raises(ValueError, match="low-slot"):
            op()


def test_low_slots_are_small(setup):
    s, client = setup
    rep = s.mem_report
    low = float(rep.split("ct_low ")[1].split(" MB")[0])
    full = float(rep.split("ct_pool ")[1].split(" MB")[0]) / 64
    assert low / NLOW <= full * (ROWS + 1) / 53, rep
    rng = np.random.default_rng(2)
    _, fs = frames(client, rng, NLOW)
    held = [_pc.import_ct(s, f, low=True) for f in fs]
    with pytest.raises(Exception, match="low slot"):
        _pc.import_ct(s, fs[0], low=True)
    del held
    with pytest.raises(Exception, match="low slot"):
        _pc.import_ct(s, client.encrypt(np.zeros(1 << 15), LEVEL - 2), low=True)   # 6 rows do not fit 4
