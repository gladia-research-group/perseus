"""The decode softmax centred on key 0 past one K group (fit_gpt2_decode_sink.py's configs): the K side alone centres
every bucket, so q.K^T returns each head's scores minus its key-0 score and the softmax matches the uncentred one."""
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perseus._paths import PKG  # noqa: E402
from perseus.impl import attention, config, fake, layout  # noqa: E402
from perseus.impl.ops import FheOps  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402

APPROX = PKG / "configs/model/approximation"
CFG_SINK = APPROX / "gpt2_base_n32_ctx512s/configs.json"
CFG_BASE = APPROX / "gpt2_base_n32_ctx1024_g3/configs.json"
LAYER = "transformer.h.3.attn"


def _rt():
    fhe, inf = fake.make_fake(complex_payload=True)
    rt = Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))
    ops = rt.ops
    ops.sm_den_recip = ops.sm_periodic = ops.sm_fold = ops.sm_cheb_gs = ops.hrs_pack = ops.kv_lanes = True
    return rt


def _attend(cfg, kc, centre, seed=41):
    """(scores, attention output, key 0's score per head, K, q) on kc random keys pushed into a (centred) cache."""
    rt = _rt()
    d = rt.dims
    rng = np.random.default_rng(seed)
    kv = attention.ComplexKVCache(d.d_head, centre=centre)
    pad = (np.arange(d.hid) % d.H) >= d.H_real
    Ks = []
    for _ in range(kc):
        k, v = rng.standard_normal(d.hid) * 0.4, rng.standard_normal(d.hid) * 0.5
        k[pad] = v[pad] = 0.0
        Ks.append(k)
        attention.cache_kv_push_packed_complex(rt, kv, fake.FakeCt(np.repeat(k, d.t) + 1j * np.repeat(v, d.t), 34))
    q = rng.standard_normal(d.hid) * 0.4
    q[pad] = 0.0
    a = attention.sm_fold_affine(rt, cfg)[0]
    scores = attention.complex_qkt(rt, kv, fake.FakeCt(layout.lane_vec(q, d.N, d.t), 34), a,
                                   cfg=cfg if centre else None)
    sc = scores.vec.copy()
    probs = attention.softmax_thor(rt, scores, cfg, kc, folded=True)
    out = attention.complex_softmax_v(rt, kv, probs)
    s0 = np.array([Ks[0][h + d.H * np.arange(d.d_head)] @ q[h + d.H * np.arange(d.d_head)]
                   for h in range(d.H_real)]) / math.sqrt(d.d_head_real)
    return rt, sc, out.vec, a * s0, len(kv.k_buckets)


@pytest.mark.parametrize("kc", [3, 65])
def test_sink_centred_matches_uncentred(monkeypatch, kc):
    """kc = 65 = 2 t + 1 spans a full K bucket and a pending group: the centred scores are the uncentred ones minus key 0's on
    each head's key slots, and the attention outputs agree to the approximations' accuracy."""
    monkeypatch.setenv("SM_CHEB_GS", "1")
    sink = config.load_configs(str(CFG_SINK)).softmax[LAYER]
    base = config.load_configs(str(CFG_BASE)).softmax[LAYER]
    assert sink.ref_keys and all(r == config.REF_SINK for r in sink.ref_keys) and sink.ref_window >= 512
    rt, sc_c, out_c, s0, nb = _attend(sink, kc, True)
    _, sc_u, out_u, _, _ = _attend(base, kc, False)
    d = rt.dims
    assert nb == (1 if kc > 2 * d.t else 0)
    # q.K^T's affine (SM_FOLD) differs between the configs: compare the raw scores per unit of it
    a_c, a_u = attention.sm_fold_affine(rt, sink)[0], attention.sm_fold_affine(rt, base)[0]
    diff = sc_u.real / a_u - sc_c.real / a_c
    head = (np.arange(d.N) % d.tH) // d.t
    on = np.abs(sc_u) > 1e-12
    assert on.sum() >= kc * d.H_real
    np.testing.assert_allclose(diff[on], (s0 / a_c)[head[on]], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(out_c.real, out_u.real, rtol=0, atol=2e-3 * np.abs(out_u.real).max())


def test_sink_centred_refusals():
    """Key 0 alone holds to its ref_window; a self / previous-key head keeps the one-group limit."""
    sink = config.load_configs(str(CFG_SINK)).softmax[LAYER]
    rt = _rt()
    d = rt.dims
    assert attention._ref_covers(d, sink, sink.ref_window) and not attention._ref_covers(d, sink, sink.ref_window + 1)
    mixed = dataclasses.replace(sink, ref_keys=(config.REF_SELF,) + sink.ref_keys[1:])
    assert not attention._ref_covers(d, mixed, d.t + 1)
    with pytest.raises(ValueError, match="self / previous key"):
        attention.mha_gqa(rt, None, 2, None, None, None, mixed)
