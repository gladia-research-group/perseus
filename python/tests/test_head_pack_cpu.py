"""SM_HEAD_PACK (attention._hp): with the model's heads in at most half the head slots, K group c of the decode cache
sits on head half c % 2 of packed group c // 2, so a cache ciphertext and the scores hold two groups where they held
one. On the numpy fake, for caches of one to several packed buckets plus pending groups: every (group, key, head) score
of q.K^T lands on its packed slot (block c // 2 of the packed period, head h + (c % 2) H_real) with the unpacked
value, the empty slots stay 0, and the period halves."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perseus.impl import attention, fake, layout  # noqa: E402
from perseus.impl.ops import FheOps  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402


def _scores(kc, pack, monkeypatch, seed=5):
    monkeypatch.setenv("SM_HEAD_PACK", "1" if pack else "0")
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
    out = attention.complex_qkt(rt, kv, fake.FakeCt(layout.lane_vec(q, d.N, d.t), 34), 1.0)
    return out.vec, attention.sm_periodic(rt, kc), d


@pytest.mark.parametrize("kc", [5, 40, 70, 130, 200, 250])     # t = 32 lanes: 1 .. 8 groups (Gp 8 x tH = N)
def test_packed_scores(kc, monkeypatch):
    base, per_u, d = _scores(kc, False, monkeypatch)
    got, per_p, _ = _scores(kc, True, monkeypatch)
    groups = -(-kc // d.t)
    if groups > 1:
        assert per_p == max(1, 1 << ((-(-groups // 2)) - 1).bit_length())
    want = np.zeros(d.N, dtype=complex)
    for g in range(groups):
        for lane in range(min(d.t, kc - g * d.t)):
            for h in range(d.H_real):
                src = (g % per_u) * d.tH + h * d.t + lane
                for blk in range(d.N // d.tH):
                    if blk % per_p == (g // 2) % per_p:
                        want[blk * d.tH + (h + (g % 2) * d.H_real) * d.t + lane] = base[src]
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-10 * max(1.0, np.abs(base).max()))


def test_pack_needs_free_head_slots(monkeypatch):
    monkeypatch.setenv("SM_HEAD_PACK", "1")
    fhe, inf = fake.make_fake(complex_payload=True, N=1024, hid=32, dim=24, H=4, H_real=3, E=128, E_real=96)
    rt = Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))
    assert attention._hp(rt) == 0
