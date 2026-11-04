"""SoftmaxCfg.gs_num: the reciprocal rounds' scores folded into the packed Goldschmidt recursion (poly.goldschmidt_recip
num=) against the product after it, on the numpy fake: the same probabilities and attention output."""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perseus._paths import PKG  # noqa: E402

from perseus.impl import config, fake, layout  # noqa: E402
from perseus.impl.ops import FheOps  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402

# main's 512-context decode config (SM_CHEB_GS, centred on key 0)
CFG = PKG / "configs/model/approximation/gpt2_base_n32_ctx512s/configs.json"


def _run(cfg, kc, seed=5, general=False, cleanse=False):
    from perseus.examples.gpt2 import ref  # noqa: F401
    from perseus.impl import attention
    fhe, inf = fake.make_fake(complex_payload=True)
    rt = Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))
    ops = rt.ops
    n = [0]
    mult = ops.mult

    def counted(*a, **kw):
        n[0] += 1
        return mult(*a, **kw)
    ops.mult = counted
    ops.sm_den_recip = ops.sm_periodic = ops.sm_fold = ops.sm_cheb_gs = ops.hrs_pack = ops.kv_lanes = True
    ops.sm_periodic_general, ops.hrs_cleanse = general, cleanse     # SM_PERIODIC=2, SM_HRS_CLEANSE (main's decode)
    d = rt.dims
    rng = np.random.default_rng(seed)
    kv = attention.ComplexKVCache(d.d_head, centre=bool(cfg.ref_keys))
    for _ in range(kc):
        k, v = rng.standard_normal(d.hid), rng.standard_normal(d.hid) * 0.5
        for arr in (k, v):
            arr[(np.arange(d.hid) % d.H) >= d.H_real] = 0.0
        attention.cache_kv_push_packed_complex(rt, kv, fake.FakeCt(np.repeat(k, d.t) + 1j * np.repeat(v, d.t), 34))
    q = rng.standard_normal(d.hid)
    q[(np.arange(d.hid) % d.H) >= d.H_real] = 0.0
    a = attention.sm_fold_affine(rt, cfg)[0]
    scores = attention.complex_qkt(rt, kv, fake.FakeCt(layout.lane_vec(q, d.N, d.t), 34), a, cfg=cfg)
    probs = attention.softmax_thor(rt, scores, cfg, kc, folded=True)
    pv = attention.complex_softmax_v(rt, kv, probs, attention._key_lanes(d, cfg, kc), pair=attention._pair(rt, cfg, kc))
    return probs, pv, n[0]


@pytest.mark.parametrize("layer", [0, 4, 8, 11])
@pytest.mark.parametrize("kc,general,cleanse", [(1, False, False), (5, False, False), (16, False, False),
                                                (40, True, True), (200, True, True)])
def test_gs_num_matches_product(monkeypatch, layer, kc, general, cleanse):
    monkeypatch.setenv("SM_CHEB_GS", "1")
    # the shipped entry with the gs_num switches off (the shipped GPT-2 decode config turns them on)
    base = dataclasses.replace(config.load_configs(str(CFG)).softmax[f"transformer.h.{layer}.attn"], gs_num=0, gs_num_refresh=0, gs_num_first=0)
    p0, v0, n0 = _run(base, kc, general=general, cleanse=cleanse)
    p1, v1, n1 = _run(dataclasses.replace(base, gs_num=1), kc, general=general, cleanse=cleanse)
    assert n1 < n0, (n0, n1)          # gs_num took effect: a product per reciprocal round folded into the recursion
    np.testing.assert_allclose(p1.vec, p0.vec, rtol=1e-9, atol=1e-12 * np.abs(p0.vec).max())
    np.testing.assert_allclose(v1.vec, v0.vec, rtol=1e-9, atol=1e-12 * np.abs(v0.vec).max())
