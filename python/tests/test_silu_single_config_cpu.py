"""A thor_composite SiLU site's single series from the config (single_cheb_coeffs): the prefill's joint GLU path runs it
instead of deriving one over PF_SILU_SINGLE_DEGS, and the parser carries it."""
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial import chebyshev as C

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perseus.impl import config, prefill  # noqa: E402


def _site(xmax, single=None):
    t = np.linspace(-1, 1, 64)
    j = {"method": "thor_composite", "xmax": xmax, "thor_p1": [0.0, 1.0], "thor_p2": [0.0, 0.25],
         "thor_p1_cheb": list(C.chebfit(t, t, 7)), "thor_p2_cheb": list(C.chebfit(t, 0.25 * t, 7)),
         "thor_p2_a": -1.0, "thor_p2_b": 1.0}
    if single is not None:
        j["single_cheb_coeffs"] = single
    return j


def test_parser_carries_the_single_series():
    s = [0.0, 0.5, 0.0, -0.01]
    assert config.parse_gelu(_site(4.0, s)).single_cheb == tuple(s)
    assert config.parse_gelu(_site(4.0)).single_cheb == ()


def test_the_config_series_wins_over_the_runtime_derivation():
    s = [0.0, 0.4, 0.0, -0.02, 0.0, 0.001]
    assert prefill.silu_single_coeffs(config.parse_gelu(_site(4.0, s))) == s


def test_without_the_field_the_series_is_derived():
    cfg = config.parse_gelu(_site(4.0))
    c = prefill.silu_single_coeffs(cfg)
    assert c is None or (len(c) > 4 and all(abs(v) < 1e-12 for v in c[0::2]))   # odd series, or the composite kept
