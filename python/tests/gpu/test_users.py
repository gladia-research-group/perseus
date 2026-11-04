"""Per-user result keys (`size={"userKeys"}`, `Session.add_user`, `export_ct_many(..., user=)`).

The data sits under a master key; a user holds only its own secret and the master public key. Its query is a public-key
encryption under the master key; the server switches the result to the user's (sparse) secret with a master -> user key
registered at run time. Gates: the user decrypts its result; the master and other users cannot; a removed user is
refused; the user's key bundle holds that one key and nothing that would expose the master secret.

    tools/gpu.sh out/users.log .venv/bin/python -m pytest -m gpu python/tests/gpu/test_users.py -q -s
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

_pc = pytest.importorskip("perseus._pc")

LEVEL, N = 50, 1 << 15


def user_bundle(cfg, ks, master, user, path):
    """The key service's job: the master -> user key, made with the user's sparse secret in the sparse place."""
    m, u = master.to_bytes(), user.to_bytes()
    n = int.from_bytes(m[4:8], "little")
    hybrid = _pc.Client(_pc.ClientSecret.from_bytes(m[:8 + n] + u[8 + n:]), seed=3)
    spec = json.loads(_pc.Session.key_manifest(cfg, _pc.KeySource.generate([1], conj=False, sparse_out=True)))
    spec["keys"] = [k for k in spec["keys"] if k["role"] == "enc_dense_to_sparse"]
    spec["public_key"] = False
    hybrid.keygen(json.dumps(spec), path, threads=8)


@pytest.fixture(scope="module")
def setup():
    cfg = _pc.Config.preset("n32_gpt2_decode")
    cfg.set("bts.auto_bts_level", 52)
    tmp = tempfile.mkdtemp(prefix="users_", dir=os.environ.get("PC_WORK_DIR"))
    master = _pc.ClientSecret.generate(seed=1)
    owner = _pc.Client(master, seed=2, complex=True)
    ks = _pc.KeySource.generate([1], conj=False, seed=3, secret=master.to_bytes())
    mb = os.path.join(tmp, "master.pckb")
    owner.keygen(_pc.Session.key_manifest(cfg, ks), mb, threads=8)
    s = _pc.Session.from_bundle(cfg, ks, mb, size={"userKeys": 2, "lowPool": 8, "lowRows": 4}, pool=48, pt_slots=8).complete_setup()
    users = {}
    for i, name in enumerate(("alice", "bob")):
        sec = _pc.ClientSecret.generate(seed=10 + i)
        c = _pc.Client(sec, seed=20 + i, complex=True)
        c.load_public_key(mb)
        path = os.path.join(tmp, f"{name}.pckb")
        user_bundle(cfg, ks, master, sec, path)
        s.add_user(name, path)
        users[name] = (c, path)
    yield s, owner, users, cfg, ks, tmp
    s.close()


def test_user_round_trip(setup):
    s, owner, users, *_ = setup
    rng = np.random.default_rng(0)
    data = [(rng.standard_normal(N) + 1j * rng.standard_normal(N)) / 8 for _ in range(4)]
    q = [(rng.standard_normal(N) + 1j * rng.standard_normal(N)) / 8 for _ in range(4)]
    stored = [_pc.import_ct(s, owner.encrypt(v, LEVEL, seeded=True), low=True) for v in data]   # under the master key
    alice, _ = users["alice"]
    p = s.fhe.mult_sum(stored, [_pc.import_ct(s, alice.encrypt(v, LEVEL, public=True)) for v in q])
    want = sum(a * b for a, b in zip(data, q))
    fa = _pc.export_ct_many(s, [p], 20, bits1=26, user="alice")[0]
    assert np.abs(alice.decrypt(fa)[0] - want).max() < 2e-3
    for other in (owner, users["bob"][0]):                      # the master key and another user see noise
        assert np.abs(other.decrypt(fa)[0] - want).max() > 1e3 * 2e-3
    fb = _pc.export_ct_many(s, [p], 20, bits1=26, user="bob")[0]
    assert np.abs(users["bob"][0].decrypt(fb)[0] - want).max() < 2e-3


def test_remove_user(setup):
    s, owner, users, cfg, ks, tmp = setup
    rng = np.random.default_rng(1)
    p = s.fhe.mult_sum([_pc.import_ct(s, owner.encrypt(rng.standard_normal(N) / 8, LEVEL))],
                       [_pc.import_ct(s, owner.encrypt(rng.standard_normal(N) / 8, LEVEL))])
    s.remove_user("bob")
    assert sorted(s.users()) == ["alice"]
    with pytest.raises(KeyError):
        _pc.export_ct_many(s, [p], 20, bits1=26, user="bob")
    s.add_user("bob", users["bob"][1])                          # a free slot again
    with pytest.raises(Exception, match="slot"):
        s.add_user("carol", users["bob"][1])


def test_user_bundle_holds_one_key(setup):
    *_, users, _, _, _ = setup
    info = _pc.bundle_info(users["alice"][1])
    assert info["entries"] == {"ksk": 1}, info
    assert info["bytes"] < 4 << 20, info
