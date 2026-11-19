"""tune.capfold (SoftmaxCfg.cap_fold): under a logit soft-cap the exp's series refitted on the raw scores, so the
soft-cap's own series goes. For a Gemma-2-like layer: the folded series, run the way the runtime runs the exp (the
score mask's centre, eval_chebyshev on the config's interval, the squarings), equals the soft-capped exp of the raw
score within the soft-cap fit's error, at no more depth than the exp series it replaces; the parsed config carries the
flag; a centred layer (ref_keys) keeps its softmax."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perseus.examples.llama import tune  # noqa: E402
from perseus.impl.config import parse_softmax  # noqa: E402
from perseus.impl.ops import NumpyOps  # noqa: E402
from perseus.impl.poly import eval_chebyshev  # noqa: E402

CAP = 50.0


def _layer(sq=3, ri=2, clip=(-34.87, 22.47)):
    """A softmax entry as perseus-calibrate writes it (the exp's degree-8 series on the capped scores) and a soft-cap."""
    from numpy.polynomial import chebyshev as C
    sf = 2.0 ** (-sq - ri)
    half = (clip[1] - clip[0]) / 2.0
    co = C.chebinterpolate(lambda x: np.exp(sf * half * x), 8)
    sm = {"n_squarings": sq, "refinement_iters": ri, "clip_lo": clip[0], "clip_hi": clip[1],
          "poly_coeffs": [1.0 / math.factorial(k) for k in range(9)],
          "cheb_coeffs": [float(v) for v in co], "cheb_a": -half * sf, "cheb_b": half * sf,
          "init_alpha": 0.006, "init_beta": 4e-6, "refine_alpha": [0.67, 0.67], "refine_beta": [0.056, 0.056],
          "gs_iters_scaled": 13, "gs_iters_refine_scaled": 11, "per_step_refine_iters": [5, 6]}
    sc = {"cap": CAP, "lo": -40.68, "hi": 40.68, "degree": 9, "fit_err": 3.8e-5}
    return sm, sc


def _exp_as_runtime(cfg, s):
    """prefill._sm_cheb + _sm_tail on valid slots, unmasked: the series of s - centre, then the squarings."""
    mean = (cfg.clip_hi + cfg.clip_lo) / 2.0
    sf = 2.0 ** (-cfg.log2delta1 - cfg.log2delta2)
    z = eval_chebyshev(NumpyOps(), s - mean, cfg.cheb_coeffs, cfg.cheb_a / sf, cfg.cheb_b / sf)
    for _ in range(cfg.log2delta1):
        z = z * z
    return z


def test_folded_series_matches_capped_exp():
    sm, sc = _layer()
    c = (sm["clip_hi"] + sm["clip_lo"]) / 2.0
    cfg_old = parse_softmax(sm)
    cfg = {"softmax": {"layers.0.attn": dict(sm)}, "softcap": {"layers.0.attn": sc}}
    tune.tune_capfold(cfg, log=lambda m: None)
    new = cfg["softmax"]["layers.0.attn"]
    assert new["cap_fold"] == 1
    cfg_new = parse_softmax(new)
    assert cfg_new.cap_fold == 1 and cfg_old.cap_fold == 0
    s = np.linspace(new["clip_lo"], new["clip_hi"], 20001)                 # raw scores
    capped = CAP * np.tanh(s / CAP)
    sf_out = 2.0 ** (-cfg_old.log2delta2)                                   # the squarings' output scale
    want = np.exp(sf_out * (capped - c))
    got = _exp_as_runtime(cfg_new, s)
    gain = 2.0 ** cfg_old.log2delta2                                        # the refine rounds' powers
    assert np.abs(got / want - 1.0).max() * gain <= sc["fit_err"]
    # the unfolded pipeline on the capped scores lands on the same values
    ref = _exp_as_runtime(cfg_old, capped)
    assert np.abs(got / ref - 1.0).max() * gain <= sc["fit_err"]
    assert tune.series_depth(new["cheb_coeffs"])[1] <= tune.series_depth(sm["cheb_coeffs"])[1]


def test_centred_layer_kept():
    sm, sc = _layer()
    sm["ref_keys"] = ["sink"] * 8
    cfg = {"softmax": {"layers.0.attn": dict(sm)}, "softcap": {"layers.0.attn": sc}}
    tune.tune_capfold(cfg, log=lambda m: None)
    assert "cap_fold" not in cfg["softmax"]["layers.0.attn"]


def test_no_softcap_no_fold():
    sm, _ = _layer()
    cfg = {"softmax": {"layers.0.attn": dict(sm)}}
    tune.tune_capfold(cfg, log=lambda m: None)
    assert "cap_fold" not in cfg["softmax"]["layers.0.attn"]
