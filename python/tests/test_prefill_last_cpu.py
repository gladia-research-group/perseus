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


def test_qkv_pack_columns():
    """PF_QKV_PACK: V's last channel block, carried on K's padding heads, comes back by a rotation of half a group."""
    pf = pytest.importorskip("perseus.examples.gpt2.prefill")
    d = layout.Dims.gpt2()
    B = BL.Blocks(d, 512)
    ch = prefill.Chunks(T, B.t)
    rng = np.random.default_rng(2)
    real = np.arange(d.hid) % d.H < d.H_real            # interleaved features j H + h: padding heads are zero
    Wk, Wv = rng.standard_normal((2, d.hid, d.hid)) * real
    bk, bv = rng.standard_normal((2, d.hid)) * real
    rows = _rows(rng, d, 1.0)
    kc, vc, kv, vv = pf._pack_kv(B, B.group_cols(Wk), B.group_cols(Wv), B.group_vec(bk), B.group_vec(bv))
    K = BL.blocked_slots(rows @ B.group_cols(Wk) + B.group_vec(bk), ch, B)[0]
    V = BL.blocked_slots(rows @ B.group_cols(Wv) + B.group_vec(bv), ch, B)[0]
    Kp = BL.blocked_slots(rows @ kc + kv, ch, B)[0]
    V0 = BL.blocked_slots(np.hstack([rows @ vc + vv, np.zeros((T, B.nch))]), ch, B)[0][0]
    V1 = np.roll(Kp[1], -(B.hg // 2) * B.t)
    heads = (np.arange(d.N) % (B.hg * B.t)) // B.t < B.group_dims(1).H_real
    assert np.abs(Kp[0] - K[0]).max() < 1e-9 and np.abs((Kp[1] - K[1])[heads]).max() < 1e-9
    assert np.abs(V0 - V[0]).max() < 1e-9 and np.abs((V1 - V[1])[heads]).max() < 1e-9
