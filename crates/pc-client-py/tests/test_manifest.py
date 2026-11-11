"""perseus_client's eager key manifest against pc-py's (`perseus._pc.Session.key_manifest`), byte for byte, over
key sources like the ones clients use: run where both are installed (a server's build plus `maturin build -m
crates/pc-client-py/Cargo.toml`):  python crates/pc-client-py/tests/test_manifest.py
Also `pytest` collects it (skipped without perseus._pc)."""
import itertools

import perseus_client as cl
import pytest

try:
    from perseus import _pc as full
except ImportError:
    full = None
pytestmark = pytest.mark.skipif(full is None, reason="perseus._pc (a server's build) is not installed")

CHAINS = ["n32", "n32/lp4b3", "n32/lp4b3/lp2b1"]
STEPS = [1, 2, 3, 4, 8, 16, 64, 256, 1024, -1, 32768, 16384 + 5]


def sources():
    for chain, layout, conj, sparse_out in itertools.product(CHAINS, ["dense", "packed"], [True, False], [True, False]):
        yield chain, dict(rot_steps=STEPS, layout=layout, conj=conj, sparse_out=sparse_out)
        yield chain, dict(rot_steps=STEPS, groups=[[1, 2, 3], [4, 8, 12]], layout=layout, conj=conj, sparse_out=sparse_out, band=3)
        yield chain, dict(rot_steps=STEPS, layout=layout, conj=conj, sparse_out=sparse_out, band=3, lean=True)
        yield chain, dict(rot_steps=STEPS, groups=[[1, 2]], layout=layout, conj=conj, sparse_out=sparse_out, band=3, lean=True,
                          low_steps=[1, 2, 3, -1], low_band=1)


def manifests(mod, chain, kw):
    cfg = mod.Config.preset("n32_gpt2_decode")
    cfg.set("bts.auto_bts_level", 51)
    cfg.set("chain.preset", chain)
    return mod.Session.key_manifest(cfg, mod.KeySource.generate(**kw))


def test_eager_manifests_match():
    n = 0
    for chain, kw in sources():
        a, b = manifests(full, chain, kw), manifests(cl, chain, kw)
        assert a == b, f"{chain} {kw}:\n pc-py  {a[:400]}\n client {b[:400]}"
        n += 1
    assert n


if __name__ == "__main__":
    test_eager_manifests_match()
    print("eager manifests: equal")
