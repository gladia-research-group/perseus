"""The scaled CutMax's op-stream switches (configs.json cutmax: spru_mean, newton_fold, conj_once, entry_dd) on the
numpy fake: the same one-hot as the record's form and the plaintext mirror, and their pieces against numpy."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perseus._paths import PKG
from perseus.examples.gpt2 import head, ref
from perseus.impl import config, fake, layout, poly
from perseus.impl.ops import FheOps
from perseus.impl.rt import Rt


def _cutmax(cfgs, logits):
    """head.cutmax_argmax_packed of `logits` on a production-size fake under configs `cfgs`: (z, refreshes)."""
    fhe, inf = fake.make_fake(N=32768, hid=1024, dim=768, H=16, H_real=12, E=4096, E_real=3072, complex_payload=True)
    rt = Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))
    d = rt.dims
    col = layout.cutmax_tile_col_of_slot(np.arange(d.N), d.hid, d.N)
    t1 = np.zeros(d.N)
    ok = d.N + col < len(logits)
    t1[ok] = logits[d.N + col[ok]]
    cfg = config.load_configs(str(PKG / f"configs/model/approximation/{cfgs}/configs.json")).cutmax
    Z = head.cutmax_argmax_packed(rt, fake.FakeCt(logits[col] + 1j * t1, 34), len(logits), cfg)
    return head.decode_z(rt, [Z], len(logits), d.N), fhe.n_bootstraps


@pytest.mark.parametrize("cfgs", ["gpt2_base_n32_kc16r_cmf", "gpt2_base_n32_kc16r_cmfl"])
def test_switched_cutmax_is_the_record_one(cfgs):
    """Every switch is exact arithmetic: the switched CutMax's one-hot is the record's (gpt2_base_n32_kc16r_cm) and the
    plaintext mirror's."""
    rng = np.random.default_rng(11)
    logits = rng.standard_normal(50257) * 3.0
    logits[4321] = logits.max() + 2.0
    z0, _ = _cutmax("gpt2_base_n32_kc16r_cm", logits)
    z1, _ = _cutmax(cfgs, logits)
    np.testing.assert_allclose(z1, z0, rtol=1e-9, atol=1e-12)
    cfg = config.load_configs(str(PKG / f"configs/model/approximation/{cfgs}/configs.json")).cutmax
    np.testing.assert_allclose(z1, ref.cutmax_ref(logits, cfg), rtol=1e-6, atol=1e-9)
    assert int(np.argmax(z1)) == 4321 and z1[4321] > 0.99


def test_newton_folded_is_newton_scaled():
    """poly.inv_sqrt_newton_folded from w = -b1 x is inv_sqrt_newton_scaled of x, served steps and scales alike."""
    with open(PKG / "configs/model/approximation/gpt2_base_n32_kc16r_cm/configs.json") as f:
        raw = json.load(f)["cutmax"]
    x = np.geomspace(1e-5, 1.0, 2001)
    for steps in raw["newton"] + [raw["sum_newton"]]:
        for scale in (1.0, 1.7):
            a = poly.inv_sqrt_newton_scaled(poly.NumpyOps(), x, steps, scale)
            b = poly.inv_sqrt_newton_folded(poly.NumpyOps(), -steps[0][1] * x, steps, scale)
            np.testing.assert_allclose(b, a, rtol=1e-13)


def test_fake_one_slot_refresh_returns_the_mean():
    """A value stamped as one slot refreshes to its slot mean (the 1-slot route), a constant to itself."""
    fhe, inf = fake.make_fake(N=32768, hid=1024, dim=768, H=16, H_real=12, E=4096, E_real=3072, complex_payload=True)
    v = np.random.default_rng(2).standard_normal(32768) + 1j
    a = fake.FakeCt(v.copy(), 30)
    inf.tag_reduce(a, 1)
    fhe.bootstrap(a)
    np.testing.assert_allclose(a.vec, np.full(32768, v.mean()), rtol=1e-12)
    c = fake.FakeCt(np.full(32768, 0.25 + 0.5j), 30)
    inf.tag_reduce(c, 1)
    fhe.bootstrap(c)
    np.testing.assert_allclose(c.vec, 0.25 + 0.5j, rtol=1e-15)
