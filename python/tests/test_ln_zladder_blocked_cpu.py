"""The blocked prefill LayerNorm on one complex ladder (zladder) on the numpy fake: the same output as the two-ladder
norm at no deeper level, with one stride-t ladder per half instead of two."""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perseus._paths import PKG  # noqa: E402

from perseus.impl import blocked, config, fake, layout, prefill  # noqa: E402
from perseus.impl.ops import FheOps  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402

CFG = PKG / "configs/model/approximation/gpt2_base_n32/configs.json"
DIMS = dict(N=4096, hid=256, dim=192, H=4, H_real=3, E=512, E_real=384)


def _norm(zladder, T, mean_shift):
    """norm_blocked's output, its deepest output level, and its ladder / conjugation counts."""
    fhe, inf = fake.make_fake(complex_payload=True, **DIMS)
    fhe.slots = None
    rt = Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))
    counts = {"rotate_and_sum": 0, "conjugate": 0}
    for name in counts:
        op = getattr(rt.ops, name)

        def wrapped(*a, _op=op, _name=name, **kw):
            counts[_name] += 1
            return _op(*a, **kw)
        setattr(rt.ops, name, wrapped)
    B = blocked.Blocks(rt.dims, nch=64)
    cfg = dataclasses.replace(config.load_configs(str(CFG)).block(0)[0], zladder=zladder)
    rows = np.random.default_rng(T).standard_normal((T, rt.dims.dim)) * 0.8 + mean_shift
    ch = prefill.Chunks(T, B.t)
    out = blocked.norm_blocked(rt, blocked.encode_blocked(rt, rows, ch, B), cfg, ch, B)
    level = max(c.level for row in out for c in row)
    return blocked.decrypt_blocked(rt, out, ch, B, rt.dims.dim), level, counts, ch


@pytest.mark.parametrize("T", [40, 128, 180])
@pytest.mark.parametrize("mean_shift", [0.1, 1.5])
def test_zladder_matches_two_ladders(T, mean_shift):
    want, lvl_base, base, ch = _norm(0, T, mean_shift)
    got, lvl_zl, zl, _ = _norm(1, T, mean_shift)
    np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-7)
    assert lvl_zl <= lvl_base            # the output is no deeper than the two-ladder norm's
    halves = sum(ch.has(c) for c in range(2 * ch.n_ct))
    assert base["rotate_and_sum"] - zl["rotate_and_sum"] == halves
    assert zl["conjugate"] - base["conjugate"] == halves
