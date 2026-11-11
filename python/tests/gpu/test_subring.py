"""Ciphertexts from a subring client (`Client(sub=True)`: a secret of N' = N / k coefficients, the chain's primes)
brought to the main secret through entry keys (`Client.keygen(entry_steps=, entry_secret=)`).

A subring frame is short (c0 in coefficient form, low bits dropped) with c1's seed rows in coefficient form too;
`import_ct` scatters both parts at stride k (X -> X^k) and transforms them, a ciphertext of the lifted plaintext under
the lifted subring secret. An entry step's rotation key switches from that secret, so one `rotate` by it gives the
rotated plaintext under the main secret. Gates: a slot vector (of period N' / 2, rolled ahead by the client) at the
bottom level through a low-band entry key and a coefficient vector at level 50 through a band-3 one decrypt, under the
main secret, to what full-ring frames give; the frames are an eighth of the full ring's; the host's expansion
(`expand_frame(n=)`) imports to the same words.

    tools/gpu.sh out/subring.log .venv/bin/python -m pytest -m gpu python/tests/gpu/test_subring.py -q -s
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

_pc = pytest.importorskip("perseus._pc")

CHAIN, N, NS = "n32/lp4b3/lp2b1", 65536, 8192
LOW, TOP = 52, 50
R_A, R_B = 3, 7                                                 # the entry steps: low tier, band 3
TOL = 1e-6


@pytest.fixture(scope="module")
def setup():
    tmp = tempfile.mkdtemp(prefix="subring_", dir=os.environ.get("PC_WORK_DIR"))
    cfg = _pc.Config.preset("n32_gpt2_decode")
    cfg.set("bts.auto_bts_level", LOW)
    cfg.set("chain.preset", CHAIN)
    sec = _pc.ClientSecret.generate(chain=CHAIN, seed=5, sparse_h=72)
    cl = _pc.Client(sec, chain=CHAIN, complex=True, seed=6)
    usec = _pc.ClientSecret.generate(chain=CHAIN, seed=7, n=NS)
    up = _pc.Client(usec, chain=CHAIN, complex=True, seed=8, sub=True)
    ks = _pc.KeySource.generate([1, R_A, 5, R_B], conj=True, seed=3, secret=sec.to_bytes(), sparse_out=True, band=3,
                                lean=True, layout="packed", low_steps=[1, R_A], low_band=1)
    kb = os.path.join(tmp, "k.pckb")
    cl.keygen(_pc.Session.key_manifest(cfg, ks), kb, threads=8, entry_steps=[R_A, R_B], entry_secret=usec)
    s = _pc.Session.from_bundle(cfg, ks, kb, pool=32, pt_slots=8).complete_setup()
    yield dict(s=s, cl=cl, up=up)
    s.close()


def _limbs(f, x):
    d = f.export_limbs(x)
    return np.concatenate([d["c0"].ravel(), d["c1"].ravel()])


def test_slots_entry(setup):
    s, f, cl, up = setup["s"], setup["s"].fhe, setup["cl"], setup["up"]
    assert up.slots == NS // 2 and up.params_hash == cl.params_hash
    rng = np.random.default_rng(0)
    t = (rng.standard_normal(NS // 2) + 1j * rng.standard_normal(NS // 2)) / 4
    want = np.tile(t, N // NS)
    fr, full = up.encrypt(np.roll(t, R_A), LOW, seeded=True, drop=8), cl.encrypt(want, LOW, seeded=True, drop=8)
    assert len(fr) < len(full) / 7
    x = _pc.import_ct(s, fr)
    y = f.rotate(x, R_A)
    assert np.abs(np.asarray(cl.decrypt(_pc.export_ct(s, y))[0]) - want).max() < TOL
    assert np.abs(np.asarray(cl.decrypt(_pc.export_ct(s, x))[0]) - np.roll(want, R_A)).max() > 1e-2     # not under the main secret yet
    assert np.abs(np.asarray(cl.decrypt(_pc.export_ct(s, f.rotate(y, 1)))[0]) - np.roll(want, -1)).max() < TOL
    qm = [int(q) for q in f.export_limbs(x)["moduli"]]
    assert np.array_equal(_limbs(f, _pc.import_ct(s, _pc.expand_frame(fr, qm, n=N))), _limbs(f, x))
    with pytest.raises(Exception):
        up.encrypt(t, LOW, seeded=True)                         # a subring client sends short frames only


def test_coefficient_entry(setup):
    s, f, cl, up = setup["s"], setup["s"].fhe, setup["cl"], setup["up"]
    k, nf = N // NS, cl.fresh_meta(TOP)["nf"]
    rng = np.random.default_rng(1)
    w = np.zeros(NS)
    w[:64] = rng.integers(-8, 9, 64) / 64
    gi = pow(pow(5, R_B, 2 * N), -1, 2 * N) % (2 * NS)          # the client's side of the entry rotation: X -> X^(1/g)
    idx = np.arange(NS) * gi % (2 * NS)
    pre = np.zeros(NS)
    pre[idx % NS] = np.where(idx >= NS, -w, w)
    fr = up.encrypt_coeffs(pre, TOP, drop=12)
    assert len(fr) < len(cl.encrypt_coeffs(np.zeros(N), TOP, drop=12)) / 7
    y = f.rotate(_pc.import_ct(s, fr), R_B)
    got = np.asarray(cl.decrypt_coeffs(_pc.export_ct(s, y))[0])
    want = np.zeros(N)
    want[::k] = w
    assert np.abs(got - want).max() * nf < 2.0**14              # the dropped bits and the switch
