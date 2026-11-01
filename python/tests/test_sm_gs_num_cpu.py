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

CFG = PKG / "configs/model/approximation/gpt2_base_n32_kc16r_cmfl_lhs2_mflp/configs.json"


def _run(cfg, kc, seed=5):
    from perseus.examples.gpt2 import ref  # noqa: F401
    from perseus.impl import attention
    fhe, inf = fake.make_fake(complex_payload=True)
    rt = Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))
    ops = rt.ops
    ops.sm_den_recip = ops.sm_periodic = ops.sm_fold = ops.sm_cheb_gs = ops.hrs_pack = ops.kv_lanes = True
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
    return probs, pv


@pytest.mark.parametrize("layer", [0, 4, 8, 11])
@pytest.mark.parametrize("kc", [1, 5, 16])
def test_gs_num_matches_product(monkeypatch, layer, kc):
    monkeypatch.setenv("SM_CHEB_GS", "1")
    base = config.load_configs(str(CFG)).softmax[f"transformer.h.{layer}.attn"]
    p0, v0 = _run(base, kc)
    p1, v1 = _run(dataclasses.replace(base, gs_num=1), kc)
    np.testing.assert_allclose(p1.vec, p0.vec, rtol=1e-9, atol=1e-12 * np.abs(p0.vec).max())
    np.testing.assert_allclose(v1.vec, v0.vec, rtol=1e-9, atol=1e-12 * np.abs(v0.vec).max())
