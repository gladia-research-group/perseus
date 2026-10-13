"""SM_CHEB_GS (impl smcheb): the packed Chebyshev-scaled goldschmidt_recip equals the direct recursion and reaches the
theoretical residual 1/T_{2^k}(z0) on its band; the config's rounds stay inside their exact bands; SM_CHEB_GS=0 keeps
today's rounds."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from perseus.impl import smcheb
from perseus.impl.attention import _recip_rounds, _recip_rounds_ops
from perseus.impl.config import parse_softmax
from perseus.impl.poly import NumpyOps, goldschmidt_recip

REPO = Path(__file__).resolve().parents[1]
CFG = json.loads((REPO / "configs/model/approximation/gpt2_base_n32/configs.json").read_text())["softmax"]


@pytest.mark.parametrize("lo,hi,k", [(0.02, 6.3, 6), (0.04, 5.7, 5), (0.5, 40.0, 4), (0.0395, 6.32, 7), (0.3, 0.9, 1)])
def test_packed_matches_direct_and_theory(lo, hi, k):
    ops = NumpyOps()
    a, b = smcheb.cheb_seed(lo, hi)
    z0 = smcheb.z0_of(lo, hi)
    kap, k0, mu = smcheb.step_consts(z0, k)
    D = np.geomspace(lo, hi, 2001)
    r = np.real(goldschmidt_recip(ops, D, a * k0, b * k0, k, 1.0, Dh=0.5 * b * k0 * D, cheb=(kap, mu)))
    d = D * (a - b * D)                                   # the direct recursion d <- c d (2 - d)
    t = smcheb.cheb_T(2, z0)
    for _ in range(k - 1):
        c = 2 * t * t / (2 * t * t - 1)
        d = c * d * (2 - d)
        t = 2 * t * t - 1
    assert np.allclose(D * r, d, rtol=0, atol=1e-12)
    assert np.abs(1 - D * r).max() <= smcheb.resid(lo, hi, k) * (1 + 1e-9) + 1e-14
    rr = goldschmidt_recip(ops, D, a * k0, b * k0, k, 1.0, cheb=(kap, mu))         # the unfolded seed: same result
    assert np.allclose(np.real(rr), r, rtol=0, atol=1e-12)


def test_real_payload_branch():
    class RealOps(NumpyOps):
        complex_payload = False
    lo, hi, k = 0.05, 4.0, 5
    a, b = smcheb.cheb_seed(lo, hi)
    kap, k0, mu = smcheb.step_consts(smcheb.z0_of(lo, hi), k)
    D = np.geomspace(lo, hi, 501)
    r = goldschmidt_recip(RealOps(), D, a * k0, b * k0, k, 1.0, cheb=(kap, mu))
    assert np.abs(1 - D * r).max() <= smcheb.resid(lo, hi, k) * (1 + 1e-9) + 1e-14


@pytest.mark.parametrize("layer", range(12))
def test_config_rounds_meet_targets(layer):
    cfg = parse_softmax(CFG[f"transformer.h.{layer}.attn"])
    assert len(cfg.recip_iters_cheb) == 1 + cfg.log2delta2
    for kc in (1, 5, 32, 33, 64, 160):
        for j, (_a, _b, k, _s, _c) in enumerate(smcheb.rounds(cfg, kc)):
            lo, hi = (smcheb.band_from_seed(cfg.init_alpha, cfg.init_beta) if j == 0
                      else smcheb.refine_band(kc, cfg.recip_targets[j - 1]))
            assert smcheb.resid(lo, hi, k) <= cfg.recip_targets[j] * (1 + 1e-9)


def test_off_keeps_today():
    cfg = parse_softmax(CFG["transformer.h.4.attn"])
    ops = NumpyOps()
    ops.sm_gs_first = True
    got = _recip_rounds_ops(ops, cfg, 17)
    assert [r[:4] for r in got] == _recip_rounds(cfg, 17, True) and all(r[4] is None for r in got)
