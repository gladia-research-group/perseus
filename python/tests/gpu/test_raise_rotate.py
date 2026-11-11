"""`Fhe.mod_raise` (the bootstrap's composite raise alone), the second low-slot class (`size={"lowPool2",
"lowRows2"}`) and `Fhe.rotate_many` (rotations of many ciphertexts by one step, batched).

A bottom (q0) ciphertext raised to level 50 keeps its rows, scale and degree; the rows above q0 hold its coefficients'
centred lift, so its phase is the input's + q0 I for a small integer polynomial I. Gates: the raised rows against a host
CRT lift (equal mod q0, the difference a multiple of q0 in {-1, 0, 1} per coefficient: the Garner terms are centred one
by one); the low-slot result equal to the full-slot one word for word; and the exact product the raise is for: the
sum over c of raise(S_c) times weights encrypting (Q_50 / q0) e_j (a coefficient-encoded unit vector, expanded),
rescaled twice, decrypts to S_j to the export's precision, the q0 I terms cancelled. `rotate_many` equals `rotate` one by one
word for word at every batch size, at the bottom and at level 50.

    tools/gpu.sh out/raise.log .venv/bin/python -m pytest -m gpu python/tests/gpu/test_raise_rotate.py -q -s
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

_pc = pytest.importorskip("perseus._pc")

LOW, TOP = 52, 50


def _steps(n_ring, levels):
    """Galois rotation steps of the coefficient expansion (X -> X^(N / 2^j + 1))."""
    m, out = 2 * n_ring, []
    for j in range(levels):
        g, x = n_ring // 2**j + 1, 1
        for r in range(n_ring // 2):
            if x == g % m:
                out.append(r)
                break
            x = x * 5 % m
    return out


def _expand(f, c, levels, n_ring):
    """2^levels ciphertexts, output i holding 2^levels times coefficient i of c's plaintext."""
    steps, cts = _steps(n_ring, levels), [c]
    for j in range(levels):
        nxt = [None] * (2 * len(cts))
        for i, x in enumerate(cts):
            y = f.rotate(x, steps[j])
            nxt[i] = f.add(x, y)
            nxt[i + len(cts)] = f.mult_monomial(f.sub(x, y), 2 * n_ring - 2**j)
        cts = nxt
    return cts


@pytest.fixture(scope="module", params=["n32", "n32/lp4b3"])
def setup(request):
    chain = request.param
    cfg = _pc.Config.preset("n32_gpt2_decode")
    cfg.set("bts.auto_bts_level", LOW)
    if chain != "n32":
        cfg.set("chain.preset", chain)
    sec = _pc.ClientSecret.generate(chain=chain, seed=5, sparse_h=72)
    cl = _pc.Client(sec, chain=chain, complex=True, seed=6)
    n_ring = 2 * cl.slots
    ks = _pc.KeySource.generate(_steps(n_ring, 3), conj=False, seed=3, secret=sec.to_bytes(), sparse_out=True, band=3,
                                lean=True)
    tmp = tempfile.mkdtemp(prefix="raise_", dir=os.environ.get("PC_WORK_DIR"))
    kb = os.path.join(tmp, "k.pckb")
    cl.keygen(_pc.Session.key_manifest(cfg, ks), kb, threads=8)
    s = _pc.Session.from_bundle(cfg, ks, kb, size={"lowPool": 16, "lowRows": 2, "lowPool2": 16, "lowRows2": 4},
                                pool=96, pt_slots=8).complete_setup()
    yield dict(s=s, cl=cl, n_ring=n_ring)
    s.close()


def _bottom_cts(S, count):
    """`count` bottom ciphertexts: each a q0 product of two fresh ones, settled (values ~2^-50)."""
    s, f, cl, n = S["s"], S["s"].fhe, S["cl"], S["n_ring"]
    rng = np.random.default_rng(0)
    out, want = [], []
    with f.suppress_auto_bts():
        for _ in range(count):
            a = (rng.standard_normal(n // 2) + 1j * rng.standard_normal(n // 2)) / 4
            b = (rng.standard_normal(n // 2) + 1j * rng.standard_normal(n // 2)) / 4
            x = _pc.import_ct(s, cl.encrypt(a * 2.0**-26, LOW), low=True)
            y = f.mult(_pc.import_ct(s, cl.encrypt(b, TOP)), 2.0**-24)
            f.realize(y)
            z = f.mult_sum_many([[x]], [[f.to_low(y)]])[0]
            out.append(f.settle(z) if z.deg == 2 else z)
            want.append(a * b * 2.0**-50)
    return out, want


def test_raise_rows(setup):
    f = setup["s"].fhe
    (x,), _ = _bottom_cts(setup, 1)
    r = f.mod_raise(x, TOP)
    assert (r.level, r.deg, r.nf) == (TOP, x.deg, x.nf)
    a, b = f.export_limbs(x), f.export_limbs(r)
    q = [int(v) for v in b["moduli"]]
    assert len(q) == 4 and [int(v) for v in a["moduli"]] == q[:2]
    q0, inv = q[0] * q[1], pow(q[0], -1, q[1])
    for part in ("c0", "c1"):
        assert np.array_equal(a[part], b[part][:2])
        r0, r1 = a[part][0].astype(object), a[part][1].astype(object)
        v = r0 + q[0] * ((r1 - r0) * inv % q[1])
        v = np.where(v > q0 // 2, v - q0, v)                    # the exact centred lift
        for row in (2, 3):
            p = q[row]
            d = (b[part][row].astype(object) - v % p) * pow(q0 % p, -1, p) % p
            d = np.where(d > p // 2, d - p, d)
            assert set(np.unique(d.astype(np.int64))) <= {-1, 0, 1}, (part, row)
    lo = f.mod_raise(x, TOP, low=True)
    assert lo.low and lo.level == TOP
    with pytest.raises(ValueError):
        f.mod_raise(r, TOP)                                     # not above the input


def test_raised_product(setup):
    s, f, cl, n = setup["s"], setup["s"].fhe, setup["cl"], setup["n_ring"]
    C, L = 8, 3
    S, want = _bottom_cts(setup, C)
    q = [int(v) for v in f.export_limbs(S[0])["moduli"]]
    qtop = f.export_limbs(f.mod_raise(S[0], TOP))["moduli"]
    w_nf = float(qtop[2]) * float(qtop[3])                      # Q_50 / q0
    raised = [f.mod_raise(x, TOP, low=True) for x in S]
    full = [f.mod_raise(x, TOP) for x in S]
    ref = _pc.export_ct_many(s, [S[5]], 12, bits1=18, sparse=True)[0]
    ref_err = np.abs(np.asarray(cl.decrypt_many([ref])[0][0]).real - want[5].real).max()
    for j in (5, 2):
        pw = np.zeros(n)
        pw[j] = 1.0 / 2**L
        w = _expand(f, _pc.import_ct(s, cl.encrypt_coeffs(list(pw), TOP, nf=w_nf)), L, n)
        outs = []
        for src in (raised, full):
            o = f.mult_sum_many([src], [w])[0]
            o = f.settle(o) if o.deg == 2 else o
            while o.level < LOW:
                f.rescale_one(o)
            outs.append(o)
        la, lb = f.export_limbs(outs[0]), f.export_limbs(outs[1])
        assert np.array_equal(la["c0"], lb["c0"]) and np.array_equal(la["c1"], lb["c1"])     # low = full slots
        assert outs[0].level == LOW and abs(outs[0].nf / S[j].nf - 1) < 1e-12
        fr = _pc.export_ct_many(s, [outs[0]], 12, bits1=18, sparse=True)[0]
        got = np.asarray(cl.decrypt_many([fr])[0][0]).real
        err = np.abs(got - want[j].real).max()
        assert err < 1.5 * ref_err, (j, err, ref_err)
        assert np.abs(got - want[(j + 1) % C].real).max() > 5 * ref_err


def test_rotate_many(setup):
    s, f, cl, n = setup["s"], setup["s"].fhe, setup["cl"], setup["n_ring"]
    step = _steps(n, 3)[1]
    rng = np.random.default_rng(3)

    def limbs(x):
        d = f.export_limbs(x)
        return np.concatenate([d["c0"].ravel(), d["c1"].ravel()])

    for level in (LOW, TOP):
        xs = [_pc.import_ct(s, cl.encrypt(rng.standard_normal(n // 2) / 4, level)) for _ in range(11)]
        ref = [limbs(f.rotate(x, step)) for x in xs]
        for batch in (None, 1, 3, 4, 8):
            got = f.rotate_many(xs, step, batch=batch)
            assert all(y.level == level and y.nf == x.nf for x, y in zip(xs, got))
            assert all(np.array_equal(limbs(y), r) for y, r in zip(got, ref)), (level, batch)
        assert all(np.array_equal(limbs(y), limbs(x)) for x, y in zip(xs, f.rotate_many(xs, 0)))
    with pytest.raises(ValueError):
        f.rotate_many([xs[0], _pc.import_ct(s, cl.encrypt(np.zeros(n // 2), LOW))], step)     # two levels
