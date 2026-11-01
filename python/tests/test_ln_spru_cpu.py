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

