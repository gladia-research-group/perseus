"""Sparse (export / encapsulation) secrets of any Hamming weight (`ClientSecret.generate(sparse_h=)`).

Keys made by the client for a secret of weight 72, the dense -> sparse export, and the client's decryption of the
sparse frame are as accurate as at the default weight 32.

    tools/gpu.sh out/sparse_weight.log .venv/bin/python -m pytest -m gpu python/tests/gpu/test_sparse_weight.py -q -s
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

_pc = pytest.importorskip("perseus._pc")

N = 1 << 15


def test_heavier_sparse_secret():
    cfg = _pc.Config.preset("n32_gpt2_decode")
    cfg.set("bts.auto_bts_level", 52)
    tmp = tempfile.mkdtemp(prefix="sparse_h_", dir=os.environ.get("PC_WORK_DIR"))
    rng = np.random.default_rng(5)
    x = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / 8
    errs = {}
    for h in (32, 72):
        sec = _pc.ClientSecret.generate(seed=12, sparse_h=h)
        b = sec.to_bytes()
        n = int.from_bytes(b[4:8], "little")
        assert sum(v != 0 for v in b[8 + n:]) == h
        c = _pc.Client(sec, seed=22, complex=True)
        ks = _pc.KeySource.generate([1], conj=False, seed=3, secret=b, sparse_out=True)
        path = os.path.join(tmp, f"h{h}.pckb")
        c.keygen(_pc.Session.key_manifest(cfg, ks), path, threads=8)
        s = _pc.Session.from_bundle(cfg, ks, path, pool=16, pt_slots=8).complete_setup()
        try:
            y = s.fhe.rotate(_pc.import_ct(s, c.encrypt(x, 52)), 1)       # the bottom level (q0)
            fr = _pc.export_ct_many(s, [y], 20, bits1=26, sparse=True)[0]
            errs[h] = float(np.abs(c.decrypt(fr)[0] - np.roll(x, -1)).max())
        finally:
            s.close()
            os.remove(path)
    os.rmdir(tmp)
    print(f"\nsparse export max err by weight: {errs}")
    assert errs[72] < 4 * errs[32] + 1e-6
