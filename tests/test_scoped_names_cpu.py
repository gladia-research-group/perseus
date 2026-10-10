"""IMPL_SCOPED_NAMES on the CPU: the op-order change it implies (the K pair pack after the V half of the push)
leaves the K/V cache the same."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perseus.impl import fake, layout  # noqa: E402
from perseus.impl.ops import FheOps  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402


@pytest.mark.parametrize("kv_lanes", [True, False])
def test_late_k_pack_same_cache(monkeypatch, kv_lanes):
    """IMPL_SCOPED_NAMES moves the K pair pack after the V half of the push: the cache comes out the same."""
    from perseus.impl import attention
    caches = []
    for scoped in ("0", "1"):
        monkeypatch.setenv("IMPL_SCOPED_NAMES", scoped)
        monkeypatch.setenv("KV_LANES", "1" if kv_lanes else "0")
        fhe, inf = fake.make_fake(complex_payload=True)
        rt = Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))
        d = rt.dims
        kv = attention.ComplexKVCache(d.d_head)
        rng = np.random.default_rng(7)
        for _ in range(2 * d.t + 3):                       # across the first pair pack
            v = rng.standard_normal(d.N) + 1j * rng.standard_normal(d.N)
            attention.cache_kv_push_packed_complex(rt, kv, fake.FakeCt(v, 34))
        caches.append(kv)
    a, b = caches
    assert len(a.k_buckets) == len(b.k_buckets) == 1 and len(a.k_pend) == len(b.k_pend)
    for x, y in zip(a.k_buckets + a.k_pend + a.v_buckets, b.k_buckets + b.k_pend + b.v_buckets):
        np.testing.assert_allclose(x.vec, y.vec, atol=1e-12)
