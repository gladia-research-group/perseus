"""Packed exports (`export_ct_many(..., pack=)`): a bottom-level ciphertext switched through a pack key
(`Client.pack_keygen`, `Session.add_pack_key`) to the client's pack secret with c1 in the subring Z[X^8], framed with
only those coefficients of c1.

Gates: the packed frame decrypts (under the pack secret) to what the plain export gives, within the error of the
sparse export at the same c0 bits; it is about half its size; the deferred frames are the same bytes; the key's
relation holds.

    tools/gpu.sh out/pack.log .venv/bin/python -m pytest -m gpu python/tests/gpu/test_pack.py -q -s
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

_pc = pytest.importorskip("perseus._pc")

CHAIN, N, SUB = "n32/lp4b3/lp2b1", 65536, 8
LOW = 52


@pytest.fixture(scope="module")
def setup():
    tmp = tempfile.mkdtemp(prefix="pack_", dir=os.environ.get("PC_WORK_DIR"))
    cfg = _pc.Config.preset("n32_gpt2_decode")
    cfg.set("bts.auto_bts_level", LOW)
    cfg.set("chain.preset", CHAIN)
    sec = _pc.ClientSecret.generate(chain=CHAIN, seed=5, sparse_h=72).with_pack(seed=9)
    assert sec.has_pack and _pc.ClientSecret.from_bytes(sec.to_bytes()).has_pack
    cl = _pc.Client(sec, chain=CHAIN, complex=True, seed=6)
    ks = _pc.KeySource.generate([1, 3], conj=True, seed=3, secret=sec.to_bytes(), sparse_out=True, band=3, lean=True,
                                layout="packed", low_steps=[1, 3], low_band=1)
    kb, pk = os.path.join(tmp, "k.pckb"), os.path.join(tmp, "p.pckb")
    cl.keygen(_pc.Session.key_manifest(cfg, ks), kb, threads=8)
    st = cl.pack_keygen(pk, sub=SUB)
    assert cl.pack_key_error(pk) < 64
    info = _pc.bundle_info(pk)
    assert info["entries"] == {"pack_ksk": 1} and st["bytes"] == info["bytes"] < 4.1 * SUB * 4 * N
    s = _pc.Session.from_bundle(cfg, ks, kb, size={"packKeys": 2}, pool=32, pt_slots=8).complete_setup()
    s.add_pack_key("u", pk)
    yield dict(s=s, cl=cl, pk=pk)
    s.close()


def test_packed_export(setup):
    s, f, cl = setup["s"], setup["s"].fhe, setup["cl"]
    rng = np.random.default_rng(0)
    t = (rng.standard_normal(N // 2) + 1j * rng.standard_normal(N // 2)) / 4
    x = f.rotate(_pc.import_ct(s, cl.encrypt(t, LOW)), 1)       # under the session's key, at the bottom level
    want = np.asarray(cl.decrypt(_pc.export_ct(s, x))[0])
    assert np.abs(want - np.roll(t, -1)).max() < 1e-6
    err = lambda fr: float(np.abs(np.asarray(cl.decrypt(fr)[0]) - want).max())
    full = _pc.export_ct_many(s, [x], 32, pack="u")[0]
    e_full = err(full)
    sparse = _pc.export_ct_many(s, [x], 16, bits1=22, sparse=True)[0]
    packed = _pc.export_ct_many(s, [x], 16, bits1=28, pack="u")[0]
    e_sparse, e_packed = err(sparse), err(packed)
    print(f"\nframes: sparse 16/22 {len(sparse)} B err {e_sparse:.3g}; packed 16/28 {len(packed)} B err {e_packed:.3g}; "
          f"packed 32/32 {len(full)} B err {e_full:.3g}")
    assert e_full < e_packed / 100                              # 32-bit frames: 2^24 on c1 against the dense pack secret
    assert e_packed < 1.25 * e_sparse + 1e-9
    assert len(packed) < 0.52 * len(sparse)
    # several at once, deferred: the same frames
    y = f.rotate(x, 3)
    now = _pc.export_ct_many(s, [x, y], 16, bits1=28, pack="u")
    later = _pc.export_ct_many(s, [x, y], 16, bits1=28, pack="u", defer=True).frames()
    assert [bytes(a) for a in now] == [bytes(a) for a in later] and bytes(now[0]) == bytes(packed)
    assert err(now[0]) == e_packed
    w2 = np.asarray(cl.decrypt(_pc.export_ct(s, y))[0])
    assert np.abs(np.asarray(cl.decrypt(now[1])[0]) - w2).max() < 1.25 * e_sparse + 1e-9
    assert np.abs(np.asarray(cl.decrypt_many(list(now))[1][0]) - w2).max() < 1.25 * e_sparse + 1e-9


def test_pack_key_slots(setup):
    s = setup["s"]
    with pytest.raises(Exception):
        s.add_pack_key("u", setup["pk"])                        # registered
    s.add_pack_key("v", setup["pk"])
    with pytest.raises(Exception):
        s.add_pack_key("w", setup["pk"])                        # no slot left
    s.remove_pack_key("v")
    s.add_pack_key("w", setup["pk"])
    s.remove_pack_key("w")
    x = _pc.import_ct(s, setup["cl"].encrypt(np.zeros(N // 2), 50))
    with pytest.raises(Exception):
        _pc.export_ct_many(s, [x], 16, pack="u")                # not at the bottom level
    with pytest.raises(Exception):
        _pc.export_ct_many(s, [x], 16, pack="nobody")
