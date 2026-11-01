"""SoftmaxCfg.pair_heads (attention._pair): the decode softmax on scores packed two heads to a t-lane block, tH/2-periodic,
against the tH layout, on the numpy fake (key_lanes 16, centred heads of every kind)."""
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

CFG_REF = PKG / "configs/model/approximation/gpt2_base_n32_kc16r_cm/configs.json"


def _run(cfg, kc, seed=41):
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
    rt._used = set()
    scores = attention.complex_qkt(rt, kv, fake.FakeCt(layout.lane_vec(q, d.N, d.t), 34), a, cfg=cfg)
    probs = attention.softmax_thor(rt, scores, cfg, kc, folded=True)
    pv = attention.complex_softmax_v(rt, kv, probs, attention._key_lanes(d, cfg, kc),
                                     pair=attention._pair(rt, cfg, kc))
    step = {k for k, _ in attention.complex_attention_step_masks(rt, cfg, kc, (kc - 1) % d.t)}
    return rt, scores, probs, pv, set(rt._used), step


def _cfg(ref):
    base = config.load_configs(str(CFG_REF)).softmax["transformer.h.0.attn"]
    if ref:
        base = dataclasses.replace(base, ref_keys=(config.REF_SELF, config.REF_PREV, config.REF_SINK) + base.ref_keys[3:],
                                   ref_window=16)
    else:
        base = dataclasses.replace(base, ref_keys=(), ref_window=0)
    return dataclasses.replace(base, key_lanes=16)


@pytest.mark.parametrize("ref", [True, False])
@pytest.mark.parametrize("kc", [1, 2, 9, 16])
def test_pair_heads_match_th_layout(monkeypatch, kc, ref):
    """The paired softmax returns the tH layout's scores and probabilities on every head's keys (moved to its paired
    slot), tH/2-periodic, and the same attention output; every per-step mask it reads is a declared step mask."""
    monkeypatch.setenv("SM_CHEB_GS", "1")
    base = _cfg(ref)
    rt0, s0, p0, v0, used0, step0 = _run(base, kc)
    rt1, s1, p1, v1, used1, step1 = _run(dataclasses.replace(base, pair_heads=1), kc)
    d = rt1.dims
    P = d.tH // 2
    assert s1.period == P
    for x in (s1.vec, p1.vec):
        np.testing.assert_allclose(x, np.tile(x[:P], d.N // P), rtol=0, atol=1e-14 * np.abs(x).max())
    # the active key slots of every real head, paired vs tH layout
    act = [(layout.pair_pos(d, h, l), h * d.t + l) for h in range(d.H_real) for l in range(kc)]
    i1, i0 = np.array(act).T
    np.testing.assert_allclose(s1.vec[i1], s0.vec[i0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(p1.vec[i1], p0.vec[i0], rtol=1e-9, atol=1e-12 * np.abs(p0.vec).max())
    # every slot of P off the keys of every head (the padded ones' included) is 0 in both
    allk = np.array([(layout.pair_pos(d, h, l), h * d.t + l) for h in range(d.H) for l in range(kc)]).T
    off1 = np.ones(P, bool); off1[allk[0]] = False
    off0 = np.ones(d.tH, bool); off0[allk[1]] = False
    assert np.abs(p1.vec[:P][off1]).max(initial=0) < 1e-12 and np.abs(p0.vec[:d.tH][off0]).max(initial=0) < 1e-12
    np.testing.assert_allclose(v1.vec, v0.vec, rtol=1e-9, atol=1e-12 * np.abs(v0.vec).max())
    # per-step masks: whatever the paired run reads beyond the tH run's static masks is declared for the step
    static0 = used0 - step0
    extra = {k for k in used1 - step1 if (k[:-1] if k[-1] == "pair" else k) not in static0}
    assert {k[0] for k in extra} <= {"sv.pair", "tok0.h.pair"}, extra
    assert not (step1 - used1 - {k for k in step1 if k[0].startswith("kpush") or k[0].startswith("v.cpc")})


def test_pair_heads_off_past_the_window():
    """Past the key window (kc > t/2) the config runs the tH layout."""
    from perseus.impl import attention
    fhe, inf = fake.make_fake(complex_payload=True)
    rt = Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))
    rt.ops.sm_periodic = True
    cfg = dataclasses.replace(_cfg(False), pair_heads=1)
    assert attention._pair(rt, cfg, 16) and not attention._pair(rt, cfg, 17)
