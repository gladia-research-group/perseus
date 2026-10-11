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


@pytest.mark.parametrize("complex_packing", [True, False])
def test_periodic_softmax_names_independent_of_the_period(monkeypatch, complex_packing):
    """SM_PERIODIC=2 with step-scoped names: outside the Gp sub-steps the softmax and P.V issue the same op stream at
    every period (Gp = 1, 2, 4), so their ciphertexts -- the refresh sites a plan pins -- carry the same names at every
    cache length and one token-0 plan binds throughout."""
    from perseus.impl import attention, magtape
    from perseus.impl.config import load_configs
    monkeypatch.setenv("IMPL_SCOPED_NAMES", "1")
    cfg = load_configs(str(ROOT / "configs/model/approximation/gpt2_base_n32/configs.json")).softmax["transformer.h.0.attn"]
    streams = {}
    for kc in (1, 33, 65):
        fhe, inf = fake.make_fake(complex_payload=True)
        fhe.graph_scope, fhe.graph_scope_counter = "", 0
        tape = magtape.attach(fhe, inf)
        rt = Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))
        assert rt.scoped_names
        rt.ops.sm_den_recip, rt.ops.sm_periodic, rt.ops.sm_periodic_general = True, True, True
        d = rt.dims
        rng = np.random.default_rng(5)
        kv = attention.ComplexKVCache(d.d_head) if complex_packing else attention.KVCache(d.d_head)
        for _ in range(kc):
            k, v = (layout.lane_vec(rng.standard_normal(d.hid) * 0.5, d.N, d.t) for _ in range(2))
            if complex_packing:
                attention.cache_kv_push_packed_complex(rt, kv, fake.FakeCt(k + 1j * v, 34))
            else:
                attention.cache_k_push(rt, kv, fake.FakeCt(k, 34))
                attention.cache_v_push(rt, kv, fake.FakeCt(v, 34))
        q = fake.FakeCt(layout.lane_vec(rng.standard_normal(d.hid), d.N, d.t), 34)
        scores = (attention.complex_qkt if complex_packing else attention.qkt)(rt, kv, q)
        tape.active = True
        with rt.step("softmax"):
            probs = attention.softmax_thor(rt, scores, cfg, kc)
        with rt.step("softmax_v"):
            (attention.complex_softmax_v if complex_packing else attention.softmax_v)(rt, kv, probs)
        streams[kc] = [(op, step) for op, step, _ in tape.records if "gp" not in step.split(".")]
        assert attention.sm_periodic(rt, kc) == {1: 1, 33: 2, 65: 4}[kc]
    assert streams[1] == streams[33] == streams[65]
    assert len(streams[1]) > 100
