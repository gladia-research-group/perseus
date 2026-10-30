"""GPT-2's blocked prefill attention on the plaintext mirror: the last-position block (PF_LAST_QUERY:
blocked.attention_last against blocked.attention_blocked at the last position, blocked.token_bcast against the row it
broadcasts), the last head group's K and V in one PCMM output (PF_QKV_PACK) and the packed q.K^T / P.V reductions
(PF_ATTN_PACK, with PF_ATTN_BSGS's baby-step giant-step lane rotations) against the plain attention."""
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

    def attend(pack, k, v, bsgs=False, hpack=False):
        monkeypatch.setattr(prefill, "_PACK", pack)
        monkeypatch.setattr(prefill, "_BSGS", bsgs)
        monkeypatch.setattr(prefill, "_SM_HPACK", hpack)
        rt = Rt(_Ops(True), d)
        return np.array([[np.asarray(x) for x in row] for row in BL.attention_blocked(rt, Q, k, v, sm, ch, B)])
    ref = attend(0, K, V)
    slot_real = [np.ones(d.N, bool), (np.arange(d.N) // B.t) % B.hg < B.group_dims(1).H_real]
    for pack, k, v, bsgs, hpack in ((3, K, V, False, False), (0, Kp, Vp, False, False), (3, Kp, Vp, False, False),
                                    (3, K, V, True, False), (3, Kp, Vp, True, False), (2, Kp, Vp, True, False),
                                    (3, Kp, Vp, False, True), (3, Kp, Vp, True, True)):
        got = attend(pack, k, v, bsgs, hpack)
        err = max(np.abs(got[g, j][slot_real[j]] - ref[g, j][slot_real[j]]).max()
                  for g in range(ref.shape[0]) for j in range(B.K))
        assert err < 1e-12 * max(1.0, np.abs(ref).max()), (pack, bsgs, hpack, err)


@pytest.mark.parametrize("pack", [1, 2, 3])
def test_bsgs_block_offsets(monkeypatch, pack):
    """PF_ATTN_BSGS's score block order is a permutation of the offsets of a score ciphertext."""
    monkeypatch.setattr(prefill, "_PACK", pack)
    assert sorted(prefill._bsgs_block_offset(np.arange(64), 64).tolist()) == list(range(64))


@pytest.mark.parametrize("layer", [0, 4, 10])
def test_padding_heads_do_not_leak(monkeypatch, layer):
    """Garbage on every padding head of the last group's Q, K and V (K's also carrying V's heads, as PF_QKV_PACK serves
    them) leaves the real heads' outputs of the packed attention, with and without PF_ATTN_BSGS / PF_SM_HPACK, equal
    to the plain attention's on clean inputs."""
    sm = config.load_configs(str(PKG / "configs/model/approximation/gpt2_base_n32_pf/configs.json")).block(layer)[2]
    d = layout.Dims.gpt2()
    B = BL.Blocks(d, 512)
    ch = prefill.Chunks(T, B.t)
    real = [np.ones(B.nch, bool), np.arange(B.nch) % B.hg < B.group_dims(1).H_real]
    rng = np.random.default_rng(1000 + layer)

    def enc(scale):
        r = scale * rng.standard_normal((T, d.hid))
        for k in range(B.K):
            r[:, k * B.nch:(k + 1) * B.nch] *= real[k]
        return [[0.5 * x for x in row] for row in BL.blocked_slots(r, ch, B)]
    Q, K, V = enc(1.2), enc(1.2), enc(1.0)
    h2t = B.hg // 2 * B.t
    pad = (np.arange(d.N) // B.t) % B.hg >= B.group_dims(1).H_real

    def garbage(x):
        y = np.array(x, dtype=complex)
        y[pad] += 10.0 * (rng.standard_normal(pad.sum()) + 1j * rng.standard_normal(pad.sum()))
        return y
    Kp = [[K[g][0], garbage(K[g][1] + np.roll(V[g][1], h2t))] for g in range(len(K))]
    Vp = [[V[g][0], garbage(np.roll(K[g][1] + np.roll(V[g][1], h2t), -h2t))] for g in range(len(K))]
    Qg = [[Q[g][0], garbage(Q[g][1])] for g in range(len(Q))]

    def attend(pack, q, k, v, bsgs=False, hpack=0):
        monkeypatch.setattr(prefill, "_PACK", pack)
        monkeypatch.setattr(prefill, "_BSGS", bsgs)
        monkeypatch.setattr(prefill, "_SM_HPACK", hpack)
        rt = Rt(_Ops(True), d)
        return np.array([[np.asarray(x) for x in row] for row in BL.attention_blocked(rt, q, k, v, sm, ch, B)])
    ref = attend(0, Q, K, V)
    slot_real = [np.ones(d.N, bool), ~pad]
    for bsgs, hpack in ((False, 0), (True, 0), (False, 1), (True, 1)):
        got = attend(3, Qg, Kp, Vp, bsgs, hpack)
        err = max(np.abs(got[g, j][slot_real[j]] - ref[g, j][slot_real[j]]).max()
                  for g in range(ref.shape[0]) for j in range(B.K))
        assert err < 1e-12 * max(1.0, np.abs(ref).max()), (bsgs, hpack, err)


@pytest.mark.parametrize("keys", [T, 100])
@pytest.mark.parametrize("layer", [0, 11])
def test_attention_first_bidirectional(keys, layer):
    """An encoder's [CLS] (position 0 of a bidirectional prompt, padding past `keys`) alone against attention_blocked."""
    rt, d, B, _ = _setup(False)
    ch = prefill.Chunks(T, B.t, bidir=True, keys=keys)
    rng = np.random.default_rng(10 + layer)
    q, k, v = _rows(rng, d, 1.2), _rows(rng, d, 1.2), _rows(rng, d, 1.0)

    def enc(r):
        return [[0.5 * x for x in row] for row in BL.blocked_slots(r, ch, B)]
    sm = config.load_configs(str(CFG)).block(layer)[2]
    ref = BL.decrypt_blocked(rt, BL.attention_blocked(rt, enc(q), enc(k), enc(v), sm, ch, B), ch, B, d.dim)
    got = BL.decrypt_blocked(rt, BL.attention_last(rt, enc(q), enc(k), enc(v), sm, ch, B, 0), ch, B, d.dim)
    assert np.abs(got - ref[0][None, :]).max() < 1e-12 * max(1.0, np.abs(ref).max())


def test_token_bcast_first():
    rt, d, B, ch = _setup(False)
    X = [list(row) for row in BL.blocked_slots(_rows(np.random.default_rng(2), d, 3.0), ch, B)]
    got = BL.decrypt_blocked(rt, [BL.token_bcast(rt, row, ch, B, 0) for row in X], ch, B, d.dim)
    want = BL.decrypt_blocked(rt, X, ch, B, d.dim)[0]
    assert np.abs(got - want[None, :]).max() == 0.0


@pytest.mark.parametrize("T,keys", [(128, 128), (128, 100), (256, 197)])
def test_attn_pack_bidirectional(monkeypatch, T, keys):
    """An encoder's attention (bidirectional, padding past `keys`; T = 256: two chunk pairs, ViT's): PF_ATTN_PACK with
    PF_ATTN_BSGS (every chunk rotated cyclically) and PF_SM_HPACK (one softmax per query chunk pair on the half-empty
    group), K / V plain and QKV-packed, against the plain attention on the real heads."""
    sm = config.load_configs(str(PKG / "configs/model/approximation/gpt2_base_n32_pf/configs.json")).block(4)[2]
    d = layout.Dims.gpt2()
    B = BL.Blocks(d, 512)
    ch = prefill.Chunks(T, B.t, bidir=True, keys=keys)
    real = [np.ones(B.nch, bool), np.arange(B.nch) % B.hg < B.group_dims(1).H_real]
    rng = np.random.default_rng(T + keys)

    def enc(scale):
        r = scale * rng.standard_normal((T, d.hid))
        for k in range(B.K):
            r[:, k * B.nch:(k + 1) * B.nch] *= real[k]
        return [[0.5 * x for x in row] for row in BL.blocked_slots(r, ch, B)]
    Q, K, V = enc(1.2), enc(1.2), enc(1.0)
    h2t = B.hg // 2 * B.t
    Kp = [[K[g][0], K[g][1] + np.roll(V[g][1], h2t)] for g in range(len(K))]
    Vp = [[V[g][0], np.roll(Kp[g][1], -h2t)] for g in range(len(K))]

    def attend(pack, k, v, bsgs=False, hpack=False):
        monkeypatch.setattr(prefill, "_PACK", pack)
        monkeypatch.setattr(prefill, "_BSGS", bsgs)
        monkeypatch.setattr(prefill, "_SM_HPACK", hpack)
        rt = Rt(_Ops(True), d)
        return np.array([[np.asarray(x) for x in row] for row in BL.attention_blocked(rt, Q, k, v, sm, ch, B)])
    ref = attend(0, K, V)
    slot_real = [np.ones(d.N, bool), (np.arange(d.N) // B.t) % B.hg < B.group_dims(1).H_real]
    for pack, k, v, bsgs, hpack in ((3, K, V, True, False), (2, K, V, True, False), (3, Kp, Vp, True, False),
                                    (0, K, V, False, True), (3, Kp, Vp, False, True), (3, Kp, Vp, True, True)):
        got = attend(pack, k, v, bsgs, hpack)
        err = max(np.abs(got[g, j][slot_real[j]] - ref[g, j][slot_real[j]]).max()
                  for g in range(ref.shape[0]) for j in range(B.K))
        assert err < 1e-12 * max(1.0, np.abs(ref).max()), (pack, bsgs, hpack, err)


@pytest.mark.parametrize("keys", [256, 197])
@pytest.mark.parametrize("layer", [0, 11])
def test_attention_first_two_chunk_pairs(keys, layer):
    """[CLS] alone on a bidirectional prompt over two chunk pairs (T = 256, ViT's: padding past `keys`) against
    attention_blocked at position 0."""
    T2 = 256
    rt, d, B, _ = _setup(False)
    ch = prefill.Chunks(T2, B.t, bidir=True, keys=keys)
    rng = np.random.default_rng(20 + layer)

    def rows(scale):
        r = np.zeros((T2, d.hid))
        r[:, :d.dim] = scale * rng.standard_normal((T2, d.dim))
        return r

    def enc(r):
        return [[0.5 * x for x in row] for row in BL.blocked_slots(r, ch, B)]
    q, k, v = rows(1.2), rows(1.2), rows(1.0)
    sm = config.load_configs(str(CFG)).block(layer)[2]
    ref = BL.decrypt_blocked(rt, BL.attention_blocked(rt, enc(q), enc(k), enc(v), sm, ch, B), ch, B, d.dim)
    out = BL.attention_last(rt, enc(q), enc(k), enc(v), sm, ch, B, 0)
    assert len(out) == 1
    got = BL.decrypt_blocked(rt, out, ch, B, d.dim)[:2 * B.t]
    assert np.abs(got - ref[0][None, :]).max() < 1e-12 * max(1.0, np.abs(ref).max())


@pytest.mark.parametrize("share", [False, True])
def test_attn_bsgs_llama_geometry(monkeypatch, share):
    """PF_ATTN_BSGS on Llama's prefill geometry (d_head 128 over t = 64 lanes: a score ciphertext's giant steps span two
    key chunks; GQA, four query blocks per K / V block, PF_KV_SHARE or not; the BOS prefix key; 127 prompt tokens)
    against the plain attention, with and without the packed reductions."""
    sm = config.load_configs(str(PKG / "configs/model/approximation/gpt2_base_n32_pf/configs.json")).block(4)[2]
    d = layout.Dims(1 << 15, 4096, 4096, 32, 32, 16384, 14336)
    B = BL.Blocks(d, 512, G=4)
    Tq = 127
    ch = prefill.Chunks(Tq, B.t)
    gd = B.group_dims(0)
    rng = np.random.default_rng(7)

    def enc(scale, width):
        r = np.zeros((Tq, d.hid))
        r[:, :width] = scale * rng.standard_normal((Tq, width))
        return [[0.5 * x for x in row[:width // B.nch]] for row in BL.blocked_slots(r, ch, B)]
    Q, K, V = enc(1.2, d.hid), enc(1.2, B.Kkv * B.nch), enc(1.0, B.Kkv * B.nch)

    def pre_vec(scale):
        v = scale * rng.standard_normal((gd.d_head, gd.H))
        return np.repeat(v.reshape(-1), gd.t).astype(complex)       # feature j of head h on every lane
    prefixes = [(pre_vec(1.2), pre_vec(1.0)) for _ in range(B.Kkv)]
    monkeypatch.setattr(prefill, "_KV_SHARE", share)

    def attend(pack, bsgs):
        monkeypatch.setattr(prefill, "_PACK", pack)
        monkeypatch.setattr(prefill, "_BSGS", bsgs)
        rt = Rt(_Ops(True), d)
        return np.array([[np.asarray(x) for x in row]
                         for row in BL.attention_blocked(rt, Q, K, V, sm, ch, B, prefixes=prefixes)])
    ref = attend(0, False)
    for pack, bsgs in ((3, False), (3, True), (2, True)):
        got = attend(pack, bsgs)
        err = np.abs(got - ref).max()
        assert err < 1e-12 * max(1.0, np.abs(ref).max()), (pack, bsgs, err)


def test_attention_last_llama_geometry():
    """The last position alone on Llama's prefill geometry (127 prompt tokens: position 126 on lane 62 of chunk 1; GQA,
    four query blocks per K / V block; RoPE; the BOS key / value on the first padding lane) against attention_blocked
    at that position, with the post-RoPE K / V of every position handed over as prefill_attention's."""
    from perseus.impl.attention import Rope
    sm = config.load_configs(str(PKG / "configs/model/approximation/gpt2_base_n32_pf/configs.json")).block(4)[2]
    d = layout.Dims(1 << 15, 4096, 4096, 32, 32, 16384, 14336)
    B = BL.Blocks(d, 512, G=4)
    Tq = 127
    ch = prefill.Chunks(Tq, B.t, base=1)
    gd = B.group_dims(0)
    rng = np.random.default_rng(11)
    rope = Rope(1.0 / 500000.0 ** (np.arange(0, gd.d_head, 2) / gd.d_head))

    def enc(scale, width):
        r = np.zeros((Tq, d.hid))
        r[:, :width] = scale * rng.standard_normal((Tq, width))
        return [[0.5 * x for x in row[:width // B.nch]] for row in BL.blocked_slots(r, ch, B)]
    Q, K, V = enc(1.2, d.hid), enc(1.2, B.Kkv * B.nch), enc(1.0, B.Kkv * B.nch)

    def pre_vec(scale):
        v = scale * rng.standard_normal((gd.d_head, gd.H))
        return np.repeat(v.reshape(-1), gd.t).astype(complex)
    prefixes = [(pre_vec(1.2), pre_vec(1.0)) for _ in range(B.Kkv)]
    rt = Rt(_Ops(True), d)
    kv_ref = [dict() for _ in range(B.Kkv)]
    ref = BL.decrypt_blocked(rt, BL.attention_blocked(rt, Q, K, V, sm, ch, B, rope=rope, prefixes=prefixes, kv=kv_ref),
                             ch, B, d.dim)
    rt = Rt(_Ops(True), d)
    kv = [dict() for _ in range(B.Kkv)]
    got = BL.decrypt_blocked(rt, BL.attention_last(rt, Q, K, V, sm, ch, B, Tq - 1, rope=rope, prefixes=prefixes, kv=kv),
                             ch, B, d.dim)
    assert np.abs(got - ref[Tq - 1][None, :]).max() < 1e-12 * max(1.0, np.abs(ref).max())
    for b in range(B.Kkv):
        for name in ("K", "V"):
            for x, y in zip(kv[b][name], kv_ref[b][name]):
                assert np.abs(np.asarray(x) - np.asarray(y)).max() < 1e-12
