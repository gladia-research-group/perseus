"""The CCMM prefill softmax (sinc_softmax.softmax, Llama's prefill attention) with the active mask folded into the
exp's series (SoftmaxCfg.exp_mask_fold, as the packed prefill's) against the unfolded one, on the numpy fake: the same
probabilities, and no active-mask product after the exp."""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perseus._paths import PKG  # noqa: E402

from perseus.impl import config, fake, layout  # noqa: E402
from perseus.impl import sinc as sc  # noqa: E402
from perseus.impl import sinc_softmax as sm  # noqa: E402
from perseus.impl.ops import FheOps  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402

CFG = PKG / "configs/model/approximation/llama31_8b_n32_cheb1e-5_kc256s/configs.json"


def _softmax(fold, layer, prescaled):
    """The packed probabilities of 16 random score ciphertexts, and the active-mask keys the run used."""
    steps = {1 << b for b in range(15)} | {sc.SLOTS - (1 << b) for b in range(15)}
    fhe, inf = fake.make_fake(N=sc.SLOTS, complex_payload=True, rot_steps=steps)
    rt = Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))
    rt.ops.cheb_bsgs = rt.ops.sm_cheb_gs = True
    keys = []
    mm = rt.mult_mask

    def mult_mask(ct, key, build, *a, **k):
        keys.append(key)
        return mm(ct, key, build, *a, **k)
    rt.mult_mask = mult_mask
    cfg = dataclasses.replace(config.load_configs(str(CFG)).softmax[f"layers.{layer}.attn"], exp_mask_fold=fold)
    rng = np.random.default_rng(7 + layer)
    al = sm.score_affine(cfg)[0] if prescaled else 1.0
    S = [fake.FakeCt((al * rng.uniform(cfg.clip_lo, cfg.clip_hi, sc.SLOTS)).astype(complex), 40) for _ in range(sc.NCT)]
    P = sm.softmax(rt, S, cfg, layer=layer, T=127, prescaled=prescaled)
    return np.array([p.vec for p in P]), keys


@pytest.mark.parametrize("layer", [0, 17])
@pytest.mark.parametrize("prescaled", [False, True])
def test_ccmm_softmax_exp_mask_fold(layer, prescaled):
    p0, k0 = _softmax(0, layer, prescaled)
    p2, k2 = _softmax(2, layer, prescaled)
    np.testing.assert_allclose(p2, p0, rtol=1e-8, atol=1e-10 * np.abs(p0).max())
    active = lambda keys: [k for k in keys if len(k) > 2 and k[2] == "active"]
    assert len(active(k0)) == sc.NCT and all(len(k) == 4 for k in active(k0))     # one product per exp, after it
    assert all(k[4] == "fold" for k in active(k2)) and active(k2)                  # only the series' masks
