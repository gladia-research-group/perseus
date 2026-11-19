"""SM_QKT_SHARE (attention._qkt_share): with the model's heads in at most half the head slots (Gemma-2-2B: 8 of 16),
two K buckets' q.K^T products share one head-dimension rotation sum, the second on the first's zero padding heads. The
scores equal the unshared ones for caches of one or two pending groups, an odd or even bucket count and a Gp period."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perseus.impl import attention, fake, layout  # noqa: E402
from perseus.impl.ops import FheOps  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402


def _scores(kc, share, monkeypatch, seed=11):
    monkeypatch.setenv("SM_QKT_SHARE", "1" if share else "0")
    fhe, inf = fake.make_fake(complex_payload=True, N=1024, hid=32, dim=16, H=4, H_real=2, E=128, E_real=96)
    rt = Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))
    ops, d = rt.ops, rt.dims
    ops.sm_periodic = ops.sm_periodic_general = ops.kv_lanes = True
    rng = np.random.default_rng(seed)
    kv = attention.ComplexKVCache(d.d_head)
    pad = (np.arange(d.hid) % d.H) >= d.H_real
    for _ in range(kc):
        k, v = rng.standard_normal(d.hid), rng.standard_normal(d.hid)
        k[pad] = v[pad] = 0.0
        attention.cache_kv_push_packed_complex(rt, kv, fake.FakeCt(np.repeat(k, d.t) + 1j * np.repeat(v, d.t), 34))
    q = rng.standard_normal(d.hid)
    q[pad] = 0.0
    n0 = len(fhe.calls)
    out = attention.complex_qkt(rt, kv, fake.FakeCt(layout.lane_vec(q, d.N, d.t), 34), 0.7)
    sums = sum(c.startswith("rotate_sum") for c in fhe.calls[n0:])     # the fake books a ladder as one call
    return out.vec, sums, (len(kv.k_buckets), len(kv.k_pend))


@pytest.mark.parametrize("kc", [3, 33, 70, 100, 200, 260, 300])     # t = 32 lanes: 0..4 buckets, 1..2 pending
def test_shared_sum_matches(kc, monkeypatch):
    base, r0, _ = _scores(kc, False, monkeypatch)
    got, r1, (nb, npend) = _scores(kc, True, monkeypatch)
    np.testing.assert_allclose(got, base, rtol=0, atol=1e-10 * max(1.0, np.abs(base).max()))
    assert r0 - r1 == nb // 2 + npend // 2       # one head-dimension sum per bucket pair and per pending pair


def test_share_needs_free_head_slots(monkeypatch):
    monkeypatch.setenv("SM_QKT_SHARE", "1")
    fhe, inf = fake.make_fake(complex_payload=True, N=1024, hid=32, dim=24, H=4, H_real=3, E=128, E_real=96)
    rt = Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))
    assert attention._qkt_share(rt) == 0
