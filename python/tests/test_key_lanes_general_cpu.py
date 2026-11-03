"""SoftmaxCfg.key_lanes and pair_heads on the 512-token decode layout (SM_PERIODIC=2): while the cache fits the key
window (Gp = 1, kc <= key_lanes) the lane ladders run over that window and two heads share a t-lane block, with the
same attention output as the full ladders; past it the op stream is the one of a config without the levers, so the
plans of the later positions bind unchanged. GQA's merged softmax (Llama) takes the key window, not the paired heads.
On the numpy fake (t = 32 lanes, key_lanes 16)."""
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
TRACED = ("rotate", "rotate_many", "rotate_and_sum", "mult", "mult_add_many", "add", "sub", "conjugate", "mult_i", "square", "copy",
          "inplace_add", "tag_reduce")


def _rt(trace):
    fhe, inf = fake.make_fake(complex_payload=True)
    rt = Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))
    ops = rt.ops
    for name in TRACED:
        fn = getattr(ops, name)

        def wrapped(*a, _fn=fn, _name=name, **kw):
            if _name == "rotate_and_sum":     # the ladder's key switches: gap = |start|, 2 |start|, ... < stop
                trace.append(f"rotate_many:{(abs(a[2]) // abs(a[1])).bit_length() - 1}")
            else:
                trace.append(f"{_name}:{len(a[1])}" if _name == "rotate_many" else _name)
            return _fn(*a, **kw)
        setattr(ops, name, wrapped)
    ops.sm_den_recip = ops.sm_periodic = ops.sm_periodic_general = ops.sm_fold = ops.sm_cheb_gs = True
    ops.hrs_pack = ops.kv_lanes = True
    return rt


def _cache(rt, cfg, kc, rng):
    d = rt.dims
    kv = attention.ComplexKVCache(d.d_head, centre=bool(cfg.ref_keys))
    for _ in range(kc):
        k, v = rng.standard_normal(d.hid), rng.standard_normal(d.hid) * 0.5
        for arr in (k, v):
            arr[(np.arange(d.hid) % d.H) >= d.H_real] = 0.0
        attention.cache_kv_push_packed_complex(rt, kv, fake.FakeCt(np.repeat(k, d.t) + 1j * np.repeat(v, d.t), 34))
    return kv


def _query(rt, rng):
    d = rt.dims
    q = rng.standard_normal(d.hid)
    q[(np.arange(d.hid) % d.H) >= d.H_real] = 0.0
    return fake.FakeCt(layout.lane_vec(q, d.N, d.t), 34)


def _rots(trace):
    """The key switches of the rotations in a trace (a hoisted rotate_many and a ladder count each step)."""
    return sum(int(t.split(":")[1]) if t.startswith("rotate_many:") else t == "rotate" for t in trace)


def _mha(cfg, kc, seed=5):
    """complex_qkt -> softmax_thor -> complex_softmax_v as the decode's mha runs them; (output, op trace)."""
    trace = []
    rt = _rt(trace)
    rng = np.random.default_rng(seed)
    kv = _cache(rt, cfg, kc, rng)
    a = attention.sm_fold_affine(rt, cfg)[0]
    del trace[:]
    scores = attention.complex_qkt(rt, kv, _query(rt, rng), a, cfg=cfg)
    probs = attention.softmax_thor(rt, scores, cfg, kc, folded=True)
    out = attention.complex_softmax_v(rt, kv, probs, attention._key_lanes(rt.dims, cfg, kc),
                                      pair=attention._pair(rt, cfg, kc))
    return out.vec, trace


def _gqa_merged(cfg, kc, G=4, seed=7):
    """mha_gqa's merged branch: G groups' scores in one G tH-periodic softmax, P split per group for P.V."""
    trace = []
    rt = _rt(trace)
    ops, d = rt.ops, rt.dims
    rng = np.random.default_rng(seed)
    kv = _cache(rt, cfg, kc, rng)
    a = attention.sm_fold_affine(rt, cfg)[0]
    del trace[:]
    scores = None
    for g in range(G):
        s_g = attention.complex_qkt(rt, kv, _query(rt, rng), a, merge=(G, g), cfg=cfg)
        scores = s_g if scores is None else ops.add(scores, s_g)
    scores = ops.tag_reduce(scores, G * d.tH)
    rt.gq_merge = G
    try:
        probs = attention.softmax_thor(rt, scores, cfg, kc, folded=True)
    finally:
        rt.gq_merge = 1
    outs = []
    for g in range(G):
        p = rt.mult_mask(probs, ("gqa.psel", G, g), lambda g=g: attention._gqa_psel(d, G, g))
        p = ops.tag_reduce(attention.rotsum(ops, p, d.tH, G * d.tH), d.tH)
        outs.append(attention.complex_softmax_v(rt, kv, p, attention._key_lanes(d, cfg, kc)).vec)
    return np.concatenate(outs), trace


def _cfgs():
    base = config.load_configs(str(CFG)).softmax["transformer.h.0.attn"]
    base = dataclasses.replace(base, key_lanes=0, pair_heads=0)
    return base, dataclasses.replace(base, key_lanes=16, pair_heads=1)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("SM_CHEB_GS", "1")


@pytest.mark.parametrize("kc", [1, 2, 9, 16])
def test_levers_inside_the_key_window_keep_the_output(kc):
    base, lev = _cfgs()
    x0, t0 = _mha(base, kc)
    x1, t1 = _mha(lev, kc)
    np.testing.assert_allclose(x1, x0, rtol=0, atol=1e-9 * np.abs(x0).max())
    assert _rots(t1) < _rots(t0), (_rots(t1), _rots(t0))


@pytest.mark.parametrize("kc", [17, 32, 33, 64])
def test_levers_past_the_key_window_leave_the_op_stream(kc):
    base, lev = _cfgs()
    x0, t0 = _mha(base, kc)
    x1, t1 = _mha(lev, kc)
    assert t1 == t0
    np.testing.assert_array_equal(x1, x0)


@pytest.mark.parametrize("kc", [1, 9, 16])
def test_gqa_merged_softmax_takes_the_key_window(kc):
    base, lev = _cfgs()
    x0, t0 = _gqa_merged(base, kc)
    x1, t1 = _gqa_merged(lev, kc)
    np.testing.assert_allclose(x1, x0, rtol=0, atol=1e-9 * np.abs(x0).max())
    assert _rots(t1) < _rots(t0), (_rots(t1), _rots(t0))


@pytest.mark.parametrize("kc", [17, 32])
def test_gqa_merged_past_the_key_window_leaves_the_op_stream(kc):
    base, lev = _cfgs()
    x0, t0 = _gqa_merged(base, kc)
    x1, t1 = _gqa_merged(lev, kc)
    assert t1 == t0
    np.testing.assert_array_equal(x1, x0)
