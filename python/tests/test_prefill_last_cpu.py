"""The last-position prefill block (PF_LAST_QUERY) on the plaintext mirror: blocked.attention_last against
blocked.attention_blocked at the last position, blocked.token_bcast against the row it broadcasts."""
import contextlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perseus._paths import PKG  # noqa: E402

from perseus.impl import blocked as BL  # noqa: E402
from perseus.impl import config, layout, prefill  # noqa: E402
from perseus.impl.ops import NumpyOps  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402

CFG = PKG / "configs/model/approximation/gpt2_base_n32_thor/configs.json"
T = 128


class _Inf:
    def step(self, label):
        return contextlib.nullcontext()


class _Ops(NumpyOps):
    sm_gs_first = True
    inf = _Inf()

    def __init__(self, cheb):
        self.sm_cheb_gs = cheb

    def tag_reduce(self, a, stride):
        return a

    def decrypt_slots_complex(self, a):
        return np.asarray(a)


def _setup(cheb):
    d = layout.Dims.gpt2()
    B = BL.Blocks(d, 512)
    return Rt(_Ops(cheb), d), d, B, prefill.Chunks(T, B.t)


def _rows(rng, d, scale):
    r = np.zeros((T, d.hid))
    r[:, :d.dim] = scale * rng.standard_normal((T, d.dim))
    return r


@pytest.mark.parametrize("cheb", [False, True])
@pytest.mark.parametrize("layer", [0, 4, 11])
def test_attention_last_matches_blocked(cheb, layer):
    rt, d, B, ch = _setup(cheb)
    rng = np.random.default_rng(layer)
    q, k, v = _rows(rng, d, 1.2), _rows(rng, d, 1.2), _rows(rng, d, 1.0)

    def enc(r):
        return [[0.5 * x for x in row] for row in BL.blocked_slots(r, ch, B)]
    sm = config.load_configs(str(CFG)).block(layer)[2]
    ref = BL.decrypt_blocked(rt, BL.attention_blocked(rt, enc(q), enc(k), enc(v), sm, ch, B), ch, B, d.dim)
    got = BL.decrypt_blocked(rt, BL.attention_last(rt, enc(q), enc(k), enc(v), sm, ch, B, T - 1), ch, B, d.dim)
    assert np.abs(got - ref[T - 1][None, :]).max() < 1e-12 * max(1.0, np.abs(ref).max())


def test_token_bcast():
    rt, d, B, ch = _setup(False)
    X = [list(row) for row in BL.blocked_slots(_rows(np.random.default_rng(1), d, 3.0), ch, B)]
    got = BL.decrypt_blocked(rt, [BL.token_bcast(rt, row, ch, B, T - 1) for row in X], ch, B, d.dim)
    want = BL.decrypt_blocked(rt, X, ch, B, d.dim)[T - 1]
    assert np.abs(got - want[None, :]).max() == 0.0
