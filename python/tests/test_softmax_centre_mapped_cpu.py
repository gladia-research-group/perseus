"""A decode softmax centred on key 0 under a score map (attention._centre_mapped, Gemma-2's logit soft-cap): the K cache
stays raw, key 0's score comes from its public K (the BOS) in a step of its own, and the centred scores are f(s_j) - f(s_0) on every active key slot
(f the map, here a strongly nonlinear soft-cap series) and f(s) = 0 on the others, at f's own depth (its
interval folded into the q.K^T masks), for a cache of
one pending group, of a full K bucket and a pending group, and of a Gp = 8 period."""
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perseus._paths import PKG  # noqa: E402
from perseus.examples.llama.block import softcap_map  # noqa: E402
from perseus.examples.llama.config import SoftcapCfg  # noqa: E402
from perseus.impl import attention, config, fake, layout  # noqa: E402
from perseus.impl.ops import FheOps  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402

CFG_SINK = PKG / "configs/model/approximation/gpt2_base_n32_ctx512s/configs.json"
LAYER = "transformer.h.3.attn"


def _centred(kc, seed=7):
    """(centred scores, raw scores f maps, f's numpy form, key 0's raw score per head, dims, levels) on kc random keys."""
    fhe, inf = fake.make_fake(complex_payload=True)
    rt = Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))
    ops, d = rt.ops, rt.dims
    ops.sm_den_recip = ops.sm_periodic = ops.sm_fold = ops.sm_cheb_gs = ops.hrs_pack = ops.kv_lanes = True
    ops.sm_periodic_general = ops.cheb_bsgs = True
    cfg = dataclasses.replace(config.load_configs(str(CFG_SINK)).softmax[LAYER], key_lanes=0, pair_heads=0)
    rng = np.random.default_rng(seed)
    kv = attention.ComplexKVCache(d.d_head, centre=False)
    pad = (np.arange(d.hid) % d.H) >= d.H_real
    Ks, k0 = [], None
    for _ in range(kc):
        k, v = rng.standard_normal(d.hid) * 0.6, rng.standard_normal(d.hid) * 0.5
        k[pad] = v[pad] = 0.0
        Ks.append(k)
        attention.cache_kv_push_packed_complex(rt, kv, fake.FakeCt(np.repeat(k, d.t) + 1j * np.repeat(v, d.t), 34))
        if k0 is None:
            k0 = np.array(kv.k_pend[0].vec)     # key 0's K as the cache holds it (later keys add into its group)
    q = rng.standard_normal(d.hid) * 0.6
    q[pad] = 0.0
    sel = lambda h: h + d.H * np.arange(d.d_head)
    s = np.array([[Ks[j][sel(h)] @ q[sel(h)] for j in range(kc)] for h in range(d.H_real)]) / math.sqrt(d.d_head_real)
    m = 1.5 * np.abs(s).max()
    cap = 0.5 * np.abs(s).max()                     # tanh well past its linear range
    coeffs = np.polynomial.chebyshev.chebinterpolate(lambda y: cap * np.tanh(m * y / cap), 15)
    sc = SoftcapCfg(cap, -m, m, tuple(float(c) for c in coeffs))
    f = lambda x: np.polynomial.chebyshev.chebval(np.asarray(x) / m, coeffs)
    a, b = attention.sm_fold_affine(rt, cfg)
    fmap = softcap_map(sc)
    a_in = attention._map_in_scale(fmap, a)
    assert a_in == 1.0 / m                          # the series' interval rides the group masks
    qp = attention.qkt_query(rt, fake.FakeCt(layout.lane_vec(q, d.N, d.t), 34), cfg, kc)
    raw = attention.complex_qkt(rt, kv, qp, a_in, cfg=cfg, ref0=True, prepared=True)
    r0 = attention._ref0_score(rt, qp, k0, kc, a_in, ("test.k0", 0))
    mapped = fmap(rt, raw, a, b)
    out = attention._centre_mapped(rt, mapped, r0, fmap, (a, b), kc)
    return out, mapped, f, a, s, d, (ops.level(raw), ops.level(mapped), ops.level(out))


@pytest.mark.parametrize("kc", [3, 65, 200])
def test_centre_after_map(kc):
    out, mapped, f, a, s, d, (_, lm, lo) = _centred(kc)
    assert lo == lm, "the centring costs no level past the map"
    per = 1 if kc <= d.t else 1 << ((kc + d.t - 1) // d.t - 1).bit_length()
    want = np.zeros(d.N)
    for blk in range(d.N // d.tH):
        g = blk % per
        for h in range(d.H_real):
            for lane in range(d.t):
                j = g * d.t + lane
                if j < kc:
                    want[blk * d.tH + h * d.t + lane] = a * (f(s[h, j]) - f(s[h, 0]))
    np.testing.assert_allclose(out.vec.real, want, rtol=0, atol=1e-9 * max(1.0, np.abs(want).max()))
    # key 0's own slots come out exactly 0, the map is far from shift-invariant on these scores
    assert np.abs(f(s + 1.0) - f(s) - 1.0).max() > 0.1


def test_centre_after_map_needs_periodic_scores():
    fhe, inf = fake.make_fake(complex_payload=True)
    rt = Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))
    rt.ops.sm_periodic = False
    with pytest.raises(ValueError, match="tH-periodic"):
        attention._centre_mapped(rt, None, None, None, (1.0, 0.0), 3)
