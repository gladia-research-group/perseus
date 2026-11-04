"""SoftmaxCfg.gs_num on the grouped-query softmax (attention.py's GQA merge, Llama's decode below t keys: G query groups'
scores in one G tH-periodic ciphertext, the head sum over G tH), on the numpy fake: the same probabilities with and
without the scores folded into the Goldschmidt recursion, and fewer products."""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perseus._paths import PKG  # noqa: E402

from perseus.impl import attention, config, fake, layout  # noqa: E402
from perseus.impl.ops import FheOps  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402

CFG = PKG / "configs/model/approximation/gpt2_base_n32_ctx512s/configs.json"


def _merged_probs(cfg, kc, G, seed=3):
    """attention.py's GQA merge on the fake: G groups' scores summed into one G tH-periodic ciphertext, one softmax
    under gq_merge = G; returns its probabilities and the number of products."""
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
    d = rt.dims
    rng = np.random.default_rng(seed)
    kv = attention.ComplexKVCache(d.d_head, centre=bool(cfg.ref_keys))
    for _ in range(kc):
        k, v = rng.standard_normal(d.hid), rng.standard_normal(d.hid) * 0.5
        for arr in (k, v):
            arr[(np.arange(d.hid) % d.H) >= d.H_real] = 0.0
        attention.cache_kv_push_packed_complex(rt, kv, fake.FakeCt(np.repeat(k, d.t) + 1j * np.repeat(v, d.t), 34))
    a = attention.sm_fold_affine(rt, cfg)[0]
    scores = None
    for g in range(G):
        q = rng.standard_normal(d.hid)
        q[(np.arange(d.hid) % d.H) >= d.H_real] = 0.0
        s_g = attention.complex_qkt(rt, kv, fake.FakeCt(layout.lane_vec(q, d.N, d.t), 34), a, merge=(G, g), cfg=cfg)
        scores = s_g if scores is None else ops.add(scores, s_g)
    scores = ops.tag_reduce(scores, G * d.tH)
    rt.gq_merge = G
    try:
        probs = attention.softmax_thor(rt, scores, cfg, kc, folded=True)
    finally:
        rt.gq_merge = 1
    return probs, n[0]


@pytest.mark.parametrize("layer", [0, 7, 11])
@pytest.mark.parametrize("kc,G", [(3, 2), (16, 2), (9, 4)])
def test_gs_num_on_the_merged_gqa_softmax(monkeypatch, layer, kc, G):
    monkeypatch.setenv("SM_CHEB_GS", "1")
    # the shipped entry with the gs_num switches off (the shipped GPT-2 decode config turns them on)
    base = dataclasses.replace(config.load_configs(str(CFG)).softmax[f"transformer.h.{layer}.attn"], gs_num=0, gs_num_refresh=0, gs_num_first=0)
    p0, n0 = _merged_probs(base, kc, G)
    p1, n1 = _merged_probs(dataclasses.replace(base, gs_num=1), kc, G)
    np.testing.assert_allclose(p1.vec, p0.vec, rtol=1e-9, atol=1e-12 * np.abs(p0.vec).max())
    assert np.abs(p0.vec).max() > 1e-3
    assert n1 < n0, (n0, n1)
