"""PF_PCMM_INFOLD (examples/llama/prefill._infold): a PCMM whose stamp w_scale 2^shift passes the pre-division limit
2^PREDIV_STAMP_LOG2 takes 2^k more in its weights (its stamp lands on the limit) while its inputs' producer folds in
2^-k on a plaintext it multiplies anyway. The weight side: k is the stamp's excess, none at band shift <= PREDIV_SHIFT,
and the linear's effective weights times 2^-k inputs give the same product. The producer side: blocked.norm_blocked's
out_scale scales the norm's output exactly (ln_1 / ln_2 before qkv / gate_up), at the same depth on the fake."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perseus._paths import PKG  # noqa: E402

from perseus.examples.llama import prefill as LP  # noqa: E402
from perseus.impl import blocked, config, fake, layout, prefill  # noqa: E402
from perseus.impl.ops import FheOps  # noqa: E402
from perseus.impl.pcmm import PcmmLinear  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402

CFG = PKG / "configs/model/approximation/gpt2_base_n32/configs.json"
DIMS = dict(N=4096, hid=256, dim=192, H=4, H_real=3, E=512, E_real=384)


def _lin(ws_log2, shift):
    return PcmmLinear(1, 1, None, 2.0 ** ws_log2, shift, name="b0.down")


@pytest.mark.parametrize("ws_log2, shift, k", [(17, 7, 4), (16, 6, 2), (13, 7, 0), (10, 9, 0), (30, 1, 0), (20, 2, 2)])
def test_infold_lands_the_stamp_on_the_limit(ws_log2, shift, k):
    lin = _lin(ws_log2, shift)
    f = LP._infold(lin)
    assert f == 2.0 ** -k
    stamp = math.log2(lin.w_scale * 2.0 ** lin.shift)
    assert stamp == ws_log2 + shift - k         # band shift 1 (never pre-divided) untouched: k = 0
    assert stamp <= LP.PREDIV_STAMP_LOG2 or shift <= LP.PREDIV_SHIFT


def test_infold_keeps_the_product():
    """x W with W's digits read at w_scale / 2^k, on inputs 2^-k x: the same y."""
    rng = np.random.default_rng(0)
    Wd = rng.integers(-2 ** 15, 2 ** 15, size=(8, 8)).astype(np.float64)
    x = rng.standard_normal(8)
    lin = _lin(17, 7)
    want = x @ (Wd / lin.w_scale)
    f = LP._infold(lin)
    np.testing.assert_allclose((f * x) @ (Wd / lin.w_scale), want, rtol=1e-12)


@pytest.mark.parametrize("T", [40, 128])
def test_norm_out_scale(T):
    fhe, inf = fake.make_fake(complex_payload=True, **DIMS)
    fhe.slots = None
    rt = Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))
    d = rt.dims
    B = blocked.Blocks(d, nch=64)
    cfg = config.load_configs(str(CFG)).block(0)[0]
    rows = np.random.default_rng(T).standard_normal((T, d.dim)) * 0.8 + 0.1
    ch = prefill.Chunks(T, B.t)
    outs = {}
    for s in (1.0, 2.0 ** -3):
        Y = blocked.norm_blocked(rt, blocked.encode_blocked(rt, rows, ch, B), cfg, ch, B, rms=True, out_scale=s)
        outs[s] = (blocked.decrypt_blocked(rt, Y, ch, B, d.dim), max(y.level for blk in Y for y in blk if y is not None))
    np.testing.assert_allclose(outs[2.0 ** -3][0], outs[1.0][0] * 2.0 ** -3, rtol=1e-9, atol=1e-12)
    assert outs[2.0 ** -3][1] == outs[1.0][1]
