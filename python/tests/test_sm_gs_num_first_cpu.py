"""SoftmaxCfg.gs_num_first (with gs_num, gs_num_refresh): the rounds before it keep their product (round 0 with
gs_num_first=1, no exp-side refresh), the rest fold their scores into the recursion behind a refresh before their
square; on the numpy fake: the same probabilities and attention output as gs_num alone and as the product path, and
one deliberate refresh per folded round."""
import dataclasses

import pytest

from perseus.impl import config
from test_sm_gs_num_refresh_cpu import CFG, _run

import numpy as np


def _cfg(layer, **kw):
    base = config.load_configs(str(CFG)).softmax[f"transformer.h.{layer}.attn"]
    return dataclasses.replace(base, **kw)


@pytest.mark.parametrize("layer", [0, 4, 11])
@pytest.mark.parametrize("kc,general,cleanse", [(5, False, False), (40, True, True), (200, True, True)])
def test_refine_rounds_only(monkeypatch, layer, kc, general, cleanse):
    monkeypatch.setenv("SM_CHEB_GS", "1")
    p_prod, v_prod, n_prod, rounds = _run(_cfg(layer, gs_num=0), kc, general=general, cleanse=cleanse)
    p_num, v_num, _, _ = _run(_cfg(layer, gs_num=1), kc, general=general, cleanse=cleanse)
    p1, v1, n1, _ = _run(_cfg(layer, gs_num=1, gs_num_refresh=1, gs_num_first=1), kc, general=general,
                         cleanse=cleanse)
    for p, v in ((p_prod, v_prod), (p_num, v_num)):
        np.testing.assert_allclose(np.asarray(p1.vec), np.asarray(p.vec), rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(v1.vec), np.asarray(v.vec), rtol=0, atol=1e-12)
    assert n_prod == 0 and n1 == rounds - 1, (n_prod, n1, rounds)


def test_first_zero_is_every_round(monkeypatch):
    monkeypatch.setenv("SM_CHEB_GS", "1")
    _, _, n, rounds = _run(_cfg(4, gs_num=1, gs_num_refresh=1, gs_num_first=0), 40, general=True, cleanse=True)
    assert n == rounds


def test_first_past_the_rounds_is_the_product(monkeypatch):
    monkeypatch.setenv("SM_CHEB_GS", "1")
    p0, v0, _, rounds = _run(_cfg(4, gs_num=0), 40, general=True, cleanse=True)
    p1, v1, n1, _ = _run(_cfg(4, gs_num=1, gs_num_refresh=1, gs_num_first=rounds), 40, general=True, cleanse=True)
    np.testing.assert_allclose(np.asarray(p1.vec), np.asarray(p0.vec), rtol=0, atol=1e-12)
    assert n1 == 0


def test_config_key_parses():
    assert _cfg(4).gs_num_first == 0
    assert _cfg(4, gs_num_first=1).gs_num_first == 1
