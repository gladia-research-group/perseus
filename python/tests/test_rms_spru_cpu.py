"""RMSNorm with its mean square from one 1-slot refresh (NormCfg.spru, norm._rms_spru) against the ladder form, on the
numpy fake (whose refresh of a value stamped one slot returns its slot mean, as SPRU does)."""
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

CFG = PKG / "configs/model/approximation/llama31_8b_n32_ctx512_silus/configs.json"


def _rms(site, spru, scale, newton):
    """rms_norm of one token on the fake, decoded, and its ladder / conjugation counts."""
    fhe, inf = fake.make_fake(complex_payload=True)
    rt = Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))
    counts = {"rotate_and_sum": 0, "conjugate": 0}
    for name in counts:
        op = getattr(rt.ops, name)

        def wrapped(*a, _op=op, _name=name, **kw):
            counts[_name] += 1
            return _op(*a, **kw)
        setattr(rt.ops, name, wrapped)
    cfg = dataclasses.replace(config.load_configs(str(CFG)).norm[site], spru=spru)
    if newton:
        cfg = dataclasses.replace(cfg, cheb_nr_iters=2)
    # a magnitude inside the seed's interval: mean square of c x near the middle of [cheb_lo, cheb_hi]
    c2 = cfg.c_eff_sq(5)
    target = 0.5 * (cfg.cheb_lo + cfg.cheb_hi) * scale
    x = np.random.default_rng(3).standard_normal(inf.size.dim) * np.sqrt(target / c2) + 0.02
    y = norm.rms_norm(rt, fake.core.encode_token_input(inf, x), cfg, 5)
    return fake.core.decode_token_output(inf, y), counts


@pytest.mark.parametrize("site", ["layers.3.ln1", "layers.17.ln1", "layers.30.ln1"])
@pytest.mark.parametrize("scale", [0.6, 1.4])
@pytest.mark.parametrize("newton", [False, True])
def test_rms_spru_matches_the_ladder(site, scale, newton):
    want, base = _rms(site, 0, scale, newton)
    got, sp = _rms(site, 1, scale, newton)
    np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-10 * np.abs(want).max())
    assert base["rotate_and_sum"] - sp["rotate_and_sum"] == 1      # the all-slot ladder is gone
    assert sp["conjugate"] - base["conjugate"] == 1                 # one conjugation keeps the real part
