"""GPT-2's blocked prefill attention on the plaintext mirror: the last-position block (PF_LAST_QUERY:
blocked.attention_last against blocked.attention_blocked at the last position, blocked.token_bcast against the row it
broadcasts), the last head group's K and V in one PCMM output (PF_QKV_PACK) and the packed q.K^T / P.V reductions
(PF_ATTN_PACK) against the plain attention."""
import contextlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perseus._paths import PKG
from perseus.impl import blocked as BL
from perseus.impl import config, layout, prefill
from perseus.impl.ops import NumpyOps
from perseus.impl.rt import Rt

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


@pytest.mark.parametrize("layer", [0, 4, 11])
def test_attn_pack_matches_plain(monkeypatch, layer):
    """PF_ATTN_PACK=3 (8 products per ciphertext over windows of 8 blocks), with K / V plain and QKV-packed (K's last
    block carries V's heads on its padding heads, V's last block is that block rotated by half a group): the real heads'
    outputs equal the plain attention's (Chebyshev-scaled GS, the entry's softmax configs)."""
    sm = config.load_configs(str(PKG / "configs/model/approximation/gpt2_base_n32_pf/configs.json")).block(layer)[2]
    d = layout.Dims.gpt2()
    B = BL.Blocks(d, 512)
    ch = prefill.Chunks(T, B.t)
    real = [np.ones(B.nch, bool), np.arange(B.nch) % B.hg < B.group_dims(1).H_real]
    rng = np.random.default_rng(100 + layer)

    def enc(scale):
        r = scale * rng.standard_normal((T, d.hid))
        for k in range(B.K):
            r[:, k * B.nch:(k + 1) * B.nch] *= real[k]
        return [[0.5 * x for x in row] for row in BL.blocked_slots(r, ch, B)]
    Q, K, V = enc(1.2), enc(1.2), enc(1.0)
    h2t = B.hg // 2 * B.t
    Kp = [[K[g][0], K[g][1] + np.roll(V[g][1], h2t)] for g in range(len(K))]
    Vp = [[V[g][0], np.roll(Kp[g][1], -h2t)] for g in range(len(K))]

    def attend(pack, k, v):
        monkeypatch.setattr(prefill, "_PACK", pack)
        rt = Rt(_Ops(True), d)
        return np.array([[np.asarray(x) for x in row] for row in BL.attention_blocked(rt, Q, k, v, sm, ch, B)])
    ref = attend(0, K, V)
    slot_real = [np.ones(d.N, bool), (np.arange(d.N) // B.t) % B.hg < B.group_dims(1).H_real]
    for pack, k, v in ((3, K, V), (0, Kp, Vp), (3, Kp, Vp)):
        got = attend(pack, k, v)
        err = max(np.abs(got[g, j][slot_real[j]] - ref[g, j][slot_real[j]]).max()
                  for g in range(ref.shape[0]) for j in range(B.K))
        assert err < 1e-12 * max(1.0, np.abs(ref).max()), (pack, err)
