"""LayerNorm with its statistics from one 1-slot refresh (NormCfg.spru, norm._norm_spru) against the ladder form, on
the numpy fake (whose refresh of a value stamped one slot returns its slot mean, as SPRU does)."""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perseus._paths import PKG  # noqa: E402

from perseus.impl import config, fake, layout, norm  # noqa: E402
from perseus.impl.ops import FheOps  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402

CFG = PKG / "configs/model/approximation/gpt2_base_n32_kc16r_cmfl/configs.json"


def _rt():
    fhe, inf = fake.make_fake(complex_payload=True)
    return Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf)), fhe, inf


@pytest.mark.parametrize("site", ["transformer.h.3.ln_1", "transformer.h.10.ln_2", "transformer.h.11.ln_2"])
@pytest.mark.parametrize("pos", [0, 5])
def test_norm_spru_matches_ladders(site, pos):
    cfg = config.load_configs(str(CFG)).norm[site]
    rng = np.random.default_rng(7)
    for scale in (0.15, 0.3):
        rt, fhe, inf = _rt()
        x = rng.standard_normal(inf.size.dim) * scale + 0.05      # a nonzero mean: the m2 - mu^2 form must cancel it
        want = fake.core.decode_token_output(inf, norm.norm(rt, fake.core.encode_token_input(inf, x), cfg, pos))
        n_bts = fhe.n_bootstraps if hasattr(fhe, "n_bootstraps") else None
        rt, fhe, inf = _rt()
        y = norm.norm(rt, fake.core.encode_token_input(inf, x), dataclasses.replace(cfg, spru=1), pos)
        got = fake.core.decode_token_output(inf, y)
        np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-10 * np.abs(want).max())
        d = rt.dims
        off = np.ones(d.N, bool)
        off[np.arange(d.dim) * d.t] = False
        assert np.abs(y.vec[off]).max() < 1e-12
        del n_bts


def test_norm_spru_step_masks():
    cfg = dataclasses.replace(config.load_configs(str(CFG)).norm["transformer.h.3.ln_1"], spru=1)
    rt, _fhe, _inf = _rt()
    keys = [k for k, _ in norm.norm_step_masks(rt, cfg, 3)]
    assert [k[0] for k in keys] == ["ln.center", "ln.spru.a", "ln.spru.b"]


CFG_REF = PKG / "configs/model/approximation/gpt2_base_n32_kc16r_cm/configs.json"


@pytest.mark.parametrize("kc", [1, 2, 9, 16])
def test_key_lanes_match_full_ladders(monkeypatch, kc):
    """SoftmaxCfg.key_lanes = 16: q's replicate, the head sums and P.V's key sum over 16 lanes give the scores, the
    probabilities and the attention output of the t-lane ladders while kc <= 16."""
    from perseus.examples.gpt2 import ref  # noqa: F401
    from perseus.impl import attention
    monkeypatch.setenv("SM_CHEB_GS", "1")
    base = config.load_configs(str(CFG_REF)).softmax["transformer.h.0.attn"]
    base = dataclasses.replace(base, ref_keys=(config.REF_SELF, config.REF_PREV, config.REF_SINK) + base.ref_keys[3:])
    outs = []
    for lanes in (0, 16):
        cfg = dataclasses.replace(base, key_lanes=lanes)
        rt, fhe, inf = _rt()
        ops = rt.ops
        ops.sm_den_recip = ops.sm_periodic = ops.sm_fold = ops.sm_cheb_gs = ops.hrs_pack = ops.kv_lanes = True
        d = rt.dims
        assert d.t == 32
        rng = np.random.default_rng(41)
        kv = attention.ComplexKVCache(d.d_head, centre=True)
        for tok in range(kc):
            k, v = rng.standard_normal(d.hid), rng.standard_normal(d.hid) * 0.5
            for arr in (k, v):
                arr[(np.arange(d.hid) % d.H) >= d.H_real] = 0.0
            attention.cache_kv_push_packed_complex(rt, kv, fake.FakeCt(np.repeat(k, d.t) + 1j * np.repeat(v, d.t), 34))
        q = rng.standard_normal(d.hid)
        q[(np.arange(d.hid) % d.H) >= d.H_real] = 0.0
        a = attention.sm_fold_affine(rt, cfg)[0]
        scores = attention.complex_qkt(rt, kv, fake.FakeCt(layout.lane_vec(q, d.N, d.t), 34), a, cfg=cfg)
        probs = attention.softmax_thor(rt, scores, cfg, kc, folded=True)
        pv = attention.complex_softmax_v(rt, kv, probs, attention._key_lanes(d, cfg, kc))
        outs.append((scores.vec.copy(), probs.vec.copy(), pv.vec.copy()))
    (s0, p0, v0), (s1, p1, v1) = outs
    keys = (np.arange(d.N) % d.t) < kc
    np.testing.assert_allclose(s1[keys], s0[keys], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(p1, p0, rtol=1e-9, atol=1e-12 * np.abs(p0).max())
    np.testing.assert_allclose(v1, v0, rtol=1e-9, atol=1e-12 * np.abs(v0).max())
