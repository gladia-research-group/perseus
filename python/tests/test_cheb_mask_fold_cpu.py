"""A plaintext mask folded into a BSGS Chebyshev series (poly.eval_chebyshev_masked): the GELU's lane mask
(GeluCfg.mask_fold) and the softmax's active mask (SoftmaxCfg.exp_mask_fold) ride the series' giants and leaves at its
own depth instead of a product after it."""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perseus._paths import PKG  # noqa: E402

from perseus.impl import attention, config, fake, layout  # noqa: E402
from perseus.impl.ops import FheOps  # noqa: E402
from perseus.impl.poly import cheb_mask_scales, cheb_mask_scales_terms, eval_chebyshev, eval_chebyshev_masked  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402

APPROX = PKG / "configs/model/approximation"
# every shipped row's series: the GELUs of GPT-2 (decode, prefill), BERT and ViT, the exps of all six rows
CONFIGS = ("gpt2_base_n32_ctx512s", "gpt2_base_n32_pf_cms", "bert_base_n32_t128", "vit_base_224",
           "llama31_8b_n32_ctx512_silus", "llama31_8b_n32_cheb1e-5_kc256s")
CFG = APPROX / "gpt2_base_n32_ctx512s/configs.json"


class _V:
    def __init__(self, v, d):
        self.v, self.d = v, d


class _DepthOps:
    """Values with the depth of each product (a ciphertext or constant product is one level past its deeper input)."""
    cheb_bsgs = dead_ops = True
    realize_shared = cheb_align = False

    def mult(self, a, b):
        return _V(a.v * b.v, max(a.d, b.d) + 1) if isinstance(b, _V) else _V(a.v * b, a.d + 1)

    def square(self, a):
        return _V(a.v * a.v, a.d + 1)

    def add(self, a, b):
        return _V(a.v + b.v, max(a.d, b.d)) if isinstance(b, _V) else _V(a.v + b, a.d)

    def sub(self, a, b):
        return _V(a.v - b.v, max(a.d, b.d)) if isinstance(b, _V) else _V(a.v - b, a.d)

    def inplace_add(self, a, b):
        a.v, a.d = a.v + b.v, max(a.d, b.d)


def _series(kind):
    """(row:site, coefficients, eps) of every shipped row's GELU ("softgelu") or exp ("softmax") series."""
    out = []
    for name in CONFIGS:
        c = json.load(open(APPROX / name / "configs.json"))
        out += [(f"{name}:{k}", g["cheb_coeffs"], g.get("cheb_eps")) for k, g in c.get(kind, {}).items()
                if isinstance(g, dict) and g.get("cheb_coeffs")]
    return out


@pytest.mark.parametrize("name,coeffs,eps", _series("softgelu") + _series("softmax"))
def test_masked_series_value_and_depth(name, coeffs, eps):
    rng = np.random.default_rng(3)
    x = rng.uniform(-1, 1, 4096)
    M = (rng.uniform(0, 1, x.size) > 0.3) * 0.7
    used = []

    def mask(ct, s):
        used.append(s)
        return _V(ct.v * (s * M), ct.d + 1)

    ops = _DepthOps()
    ref = eval_chebyshev(ops, _V(x, 0), coeffs, -1.0, 1.0, eps)
    got = eval_chebyshev_masked(ops, _V(x, 0), coeffs, -1.0, 1.0, eps, mask)
    np.testing.assert_allclose(got.v, ref.v * M, rtol=0, atol=1e-13 * np.abs(ref.v).max())
    # the fold never costs more than the product after the series did, and nothing where the series has slack
    assert got.d <= ref.d + 1
    assert set(used) <= set(cheb_mask_scales(coeffs, eps))


def _rt():
    fhe, inf = fake.make_fake(complex_payload=True)
    return Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf)), fhe, inf


@pytest.mark.parametrize("kc", [1, 9, 16])
@pytest.mark.parametrize("block", [1, 4, 6])
def test_softmax_exp_mask_fold(monkeypatch, kc, block):
    """exp_mask_fold: the same exponentials and probabilities, the exponentials one level shallower (the decode's
    centred complex q.K^T on a random cache, as test_ln_spru_cpu's key-lanes test)."""
    monkeypatch.setenv("SM_CHEB_GS", "1")
    base = config.load_configs(str(CFG)).softmax[f"transformer.h.{block}.attn"]
    outs = []
    for fold in (0, 1):
        cfg = dataclasses.replace(base, exp_mask_fold=fold)
        rt, _fhe, _inf = _rt()
        ops = rt.ops
        ops.sm_den_recip = ops.sm_periodic = ops.sm_fold = ops.sm_cheb_gs = ops.hrs_pack = ops.kv_lanes = True
        d = rt.dims
        rng = np.random.default_rng(41)
        kv = attention.ComplexKVCache(d.d_head, centre=bool(cfg.ref_keys))
        for _tok in range(kc):
            k, v = rng.standard_normal(d.hid), rng.standard_normal(d.hid) * 0.5
            for arr in (k, v):
                arr[(np.arange(d.hid) % d.H) >= d.H_real] = 0.0
            attention.cache_kv_push_packed_complex(rt, kv, fake.FakeCt(np.repeat(k, d.t) + 1j * np.repeat(v, d.t), 18))
        q = rng.standard_normal(d.hid)
        q[(np.arange(d.hid) % d.H) >= d.H_real] = 0.0
        a = attention.sm_fold_affine(rt, cfg)[0]
        scores = attention.complex_qkt(rt, kv, fake.FakeCt(layout.lane_vec(q, d.N, d.t), 18), a, cfg=cfg)
        z = attention._thor_exp(rt, scores, cfg, kc, folded=True)
        probs = attention._softmax_recip(rt, z, cfg, kc)
        outs.append((z.vec.copy(), probs.vec.copy()))
    (z0, p0), (z1, p1) = outs
    np.testing.assert_allclose(z1, z0, rtol=1e-9, atol=1e-12 * np.abs(z0).max())
    np.testing.assert_allclose(p1, p0, rtol=1e-9, atol=1e-12 * np.abs(p0).max())
    # the folded series lands no lower than the unfolded series (the fake's reactive refreshes keep the whole exp from
    # comparing levels: the series alone, from a low level)
    rt, _fhe, _inf = _rt()
    x = fake.FakeCt(np.random.default_rng(2).uniform(-0.9, 0.9, rt.dims.N).astype(complex), 10)
    plain = poly_eval(rt.ops, x, base.cheb_coeffs)
    folded = poly_eval_masked(rt, x, base.cheb_coeffs)
    assert rt.ops.lvl(folded) >= rt.ops.lvl(plain)


def poly_eval(ops, x, c):
    return eval_chebyshev(ops, x, c, -1.0, 1.0)


def poly_eval_masked(rt, x, c):
    return eval_chebyshev_masked(rt.ops, x, c, -1.0, 1.0, None,
                                 lambda ct, s: rt.mult_mask(ct, ("t.m", s), lambda: np.full(rt.dims.N, s)))


@pytest.mark.parametrize("name,coeffs,_eps", _series("softmax"))
def test_masked_series_leaf_terms_keep_depth(name, coeffs, _eps):
    """With add_mask (the prefill's exp: degree-8 series whose trimmed top leaves no slack) every leaf takes the mask
    term by term: the masked series lands at the series' own depth, on every row's exp."""
    rng = np.random.default_rng(4)
    x = rng.uniform(-1, 1, 1024)
    M = (rng.uniform(0, 1, x.size) > 0.3) * 0.37
    ops = _DepthOps()
    ref = eval_chebyshev(ops, _V(x, 0), coeffs, -1.0, 1.0)
    got = eval_chebyshev_masked(ops, _V(x, 0), coeffs, -1.0, 1.0, None, lambda ct, s: _V(ct.v * (s * M), ct.d + 1),
                                lambda ct, s: _V(ct.v + s * M, ct.d))
    np.testing.assert_allclose(got.v, ref.v * M, rtol=0, atol=1e-13 * np.abs(ref.v).max())
    assert got.d == ref.d


@pytest.mark.parametrize("name,coeffs,eps", _series("softgelu"))
def test_masked_series_budget_leaves_keep_depth(name, coeffs, eps):
    """leaf_terms="budget" (GeluCfg.mask_fold 2): a leaf takes the mask term by term only where one product after its
    sum would deepen the series, so every GELU series of GPT-2, BERT and ViT keeps its own depth."""
    rng = np.random.default_rng(5)
    x = rng.uniform(-1, 1, 1024)
    M = (rng.uniform(0, 1, x.size) > 0.3) * 1.0
    ops = _DepthOps()
    ref = eval_chebyshev(ops, _V(x, 0), coeffs, -1.0, 1.0, eps)
    got = eval_chebyshev_masked(ops, _V(x, 0), coeffs, -1.0, 1.0, eps, lambda ct, s: _V(ct.v * (s * M), ct.d + 1),
                                lambda ct, s: _V(ct.v + s * M, ct.d), leaf_terms="budget")
    np.testing.assert_allclose(got.v, ref.v * M, rtol=0, atol=1e-13 * np.abs(ref.v).max())
    assert got.d == ref.d


@pytest.mark.parametrize("kc", [1, 16])
@pytest.mark.parametrize("block", [0, 4])
def test_softmax_exp_mask_fold_terms(monkeypatch, kc, block):
    """exp_mask_fold 2 on a degree-6 exp (no leaf slack): the same exponentials as unfolded, and the masked series
    lands at the plain series' depth (fold 1 would cost a level there)."""
    monkeypatch.setenv("SM_CHEB_GS", "1")
    base = config.load_configs(str(CFG)).softmax[f"transformer.h.{block}.attn"]
    base = dataclasses.replace(base, cheb_coeffs=list(base.cheb_coeffs[:7]))
    zs = []
    for fold in (0, 2):
        cfg = dataclasses.replace(base, exp_mask_fold=fold)
        rt, _fhe, _inf = _rt()
        ops = rt.ops
        ops.sm_den_recip = ops.sm_periodic = ops.sm_fold = ops.sm_cheb_gs = ops.hrs_pack = ops.kv_lanes = True
        d = rt.dims
        rng = np.random.default_rng(43)
        kv = attention.ComplexKVCache(d.d_head, centre=bool(cfg.ref_keys))
        for _tok in range(kc):
            k, v = rng.standard_normal(d.hid), rng.standard_normal(d.hid) * 0.5
            for arr in (k, v):
                arr[(np.arange(d.hid) % d.H) >= d.H_real] = 0.0
            attention.cache_kv_push_packed_complex(rt, kv, fake.FakeCt(np.repeat(k, d.t) + 1j * np.repeat(v, d.t), 18))
        q = rng.standard_normal(d.hid)
        q[(np.arange(d.hid) % d.H) >= d.H_real] = 0.0
        a = attention.sm_fold_affine(rt, cfg)[0]
        scores = attention.complex_qkt(rt, kv, fake.FakeCt(layout.lane_vec(q, d.N, d.t), 18), a, cfg=cfg)
        zs.append(attention._thor_exp(rt, scores, cfg, kc, folded=True).vec.copy())
    np.testing.assert_allclose(zs[1], zs[0], rtol=1e-9, atol=1e-12 * np.abs(zs[0]).max())
    ops = _DepthOps()
    x = np.random.default_rng(6).uniform(-1, 1, 512)
    M = (np.arange(512) % 5 > 0) * 1.0
    ref = eval_chebyshev(ops, _V(x, 0), base.cheb_coeffs, -1.0, 1.0)
    used = []

    def mask(ct, s):
        used.append(s)
        return _V(ct.v * (s * M), ct.d + 1)

    def add_mask(ct, s):
        used.append(s)
        return _V(ct.v + s * M, ct.d)
    got = eval_chebyshev_masked(ops, _V(x, 0), base.cheb_coeffs, -1.0, 1.0, None, mask, add_mask, leaf_terms="budget")
    np.testing.assert_allclose(got.v, ref.v * M, rtol=0, atol=1e-13 * np.abs(ref.v).max())
    assert got.d == ref.d == 3
    assert set(used) <= set(cheb_mask_scales_terms(base.cheb_coeffs))
