"""SoftmaxCfg.gs_num_refresh (with gs_num): each reciprocal round's numerator is refreshed right before its last squaring
(round 0: the exp's last squaring; a refine round: its square of 2 Re y), on the numpy fake: the same probabilities and
attention output as gs_num alone, exactly one deliberate refresh per round, and the planner's round-square rule
(plan/placer/round_sq) still finds each refine round's square behind the refresh."""
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
from perseus.plan.placer import round_sq  # noqa: E402

CFG = PKG / "configs/model/approximation/gpt2_base_n32_ctx512s/configs.json"


def _run(cfg, kc, seed=5, general=False, cleanse=False):
    from perseus.examples.gpt2 import ref  # noqa: F401
    from perseus.impl import attention
    fhe, inf = fake.make_fake(complex_payload=True)
    rt = Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))
    ops = rt.ops
    n = [0]
    bts = ops.bootstrap

    def counted(a):
        n[0] += 1
        return bts(a)
    ops.bootstrap = counted
    ops.sm_den_recip = ops.sm_periodic = ops.sm_fold = ops.sm_cheb_gs = ops.hrs_pack = ops.kv_lanes = True
    ops.sm_periodic_general, ops.hrs_cleanse = general, cleanse
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
    n[0] = 0                                   # the K/V pushes refresh too: count the softmax's own
    probs = attention.softmax_thor(rt, scores, cfg, kc, folded=True)
    nb = n[0]
    pv = attention.complex_softmax_v(rt, kv, probs, attention._key_lanes(d, cfg, kc), pair=attention._pair(rt, cfg, kc))
    return probs, pv, nb, len(attention._recip_rounds_ops(ops, cfg, kc))


@pytest.mark.parametrize("layer", [0, 4, 11])
@pytest.mark.parametrize("kc,general,cleanse", [(5, False, False), (40, True, True), (200, True, True)])
def test_refresh_keeps_values_and_counts_rounds(monkeypatch, layer, kc, general, cleanse):
    monkeypatch.setenv("SM_CHEB_GS", "1")
    base = config.load_configs(str(CFG)).softmax[f"transformer.h.{layer}.attn"]
    num = dataclasses.replace(base, gs_num=1)
    p0, v0, n0, _ = _run(num, kc, general=general, cleanse=cleanse)
    p1, v1, n1, rounds = _run(dataclasses.replace(num, gs_num_refresh=1), kc, general=general, cleanse=cleanse)
    np.testing.assert_allclose(np.asarray(p1.vec), np.asarray(p0.vec), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(v1.vec), np.asarray(v0.vec), rtol=0, atol=1e-12)
    assert n0 == 0 and n1 == rounds, (n0, n1, rounds)


def test_no_refresh_without_gs_num(monkeypatch):
    monkeypatch.setenv("SM_CHEB_GS", "1")
    base = config.load_configs(str(CFG)).softmax["transformer.h.4.attn"]
    _, _, n, _ = _run(dataclasses.replace(base, gs_num=0, gs_num_refresh=1), 40, general=True, cleanse=True)
    assert n == 0


def test_round_square_found_behind_the_refresh():
    def node(op, inputs, out):
        return {"op_type": op, "inputs": inputs, "output": out, "step": "blk0.softmax"}
    nodes = [node("conjugate", ["y"], "yc"), node("add", ["y", "yc"], "re"),
             node("deliberate_bootstrap", ["re"], "re_b"), node("square", ["re_b"], "z1")]
    assert round_sq.tag_round_squares(nodes) == 1
    assert nodes[-1]["step"] == "blk0.softmax" + round_sq.SQ


def test_config_field_parses():
    j = {"gs_num": 1, "gs_num_refresh": 1}
    base = config.load_configs(str(CFG)).softmax["transformer.h.4.attn"]
    assert base.gs_num_refresh == 0
    assert dataclasses.replace(base, **j).gs_num_refresh == 1
