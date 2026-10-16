"""perseus.impl.prefill on the numpy fake against float64 references."""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perseus.impl import config, fake, layout, prefill  # noqa: E402
from perseus.impl.linear import EncodedLinear  # noqa: E402
from perseus.impl.ops import FheOps  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402

CFG = ROOT / "configs/model/approximation/gpt2_base_n32/configs.json"


def _rt(**kw):
    fhe, inf = fake.make_fake(complex_payload=True, **kw)
    fhe.slots = None                                   # any rotation step
    return Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf)), fhe, inf


def _tokens(rng, T, d):
    return rng.standard_normal((T, d))


def _dense_rows(rt, Xs, ch, stride, d_feat):
    """(T, d_feat) rows of dense complex ciphertexts in a stride layout."""
    out = np.zeros((ch.T, d_feat))
    for c, X in enumerate(Xs):
        for half, part in ((0, X.vec.real), (1, X.vec.imag)):
            for lane in range(stride):
                p = (2 * c + half) * ch.t + lane if stride == ch.t else None
                if p is not None and p < ch.T:
                    out[p] = part[np.arange(d_feat) * stride + lane]
    return out


@pytest.mark.parametrize("T", [5, 32, 47, 64, 96])
def test_encode_and_split_roundtrip(T):
    rt, fhe, inf = _rt()
    rng = np.random.default_rng(T)
    d = rt.dims
    rows = _tokens(rng, T, d.dim)
    ch = prefill.Chunks(T, d.t)
    Xs = prefill.encode_prompt(rt, rows, ch)
    np.testing.assert_allclose(prefill.decrypt_prompt(rt, Xs, ch, d.dim), rows, atol=1e-12)
    A, B = prefill.split(rt.ops, Xs[0])
    np.testing.assert_allclose(A.vec, 2 * Xs[0].vec.real, atol=1e-12)
    np.testing.assert_allclose(B.vec, 2 * Xs[0].vec.imag, atol=1e-12)


@pytest.mark.parametrize("T", [7, 32, 64, 80])
def test_dense_linear_pairs(T):
    """hid -> E (up) and E -> hid (down) on complex token pairs equal x @ W + b per token."""
    rt, fhe, inf = _rt()
    rng = np.random.default_rng(T + 1)
    d = rt.dims
    ch = prefill.Chunks(T, d.t)
    rows = _tokens(rng, T, d.hid)
    Wu, bu = rng.standard_normal((d.hid, d.E)), rng.standard_normal(d.E)
    Wd, bd = rng.standard_normal((d.E, d.hid)), rng.standard_normal(d.hid)
    up = EncodedLinear.encode(Wu, d.N, d.hid, d.E, bu, name="u")
    down = EncodedLinear.encode(Wd, d.N, d.E, d.hid, bd, name="d")
    v = np.zeros(d.N, dtype=np.complex128)
    Xs = []
    for c in range(ch.n_ct):
        v = np.zeros(d.N, dtype=np.complex128)
        for half in (0, 1):
            for lane, p in enumerate(ch.lanes(2 * c + half)):
                v[np.arange(d.hid) * d.t + lane] += (1j if half else 1.0) * rows[p]
        Xs.append(fake.FakeCt(v, 30))
    (H,) = prefill.dense_linear(rt, Xs, [up], ch, d.t, d.tp_E, 1.0, 1.0)
    (Y,) = prefill.dense_linear(rt, H, [down], ch, d.tp_E, d.t, 1.0, 1.0)
    want = (rows @ Wu + bu) @ Wd + bd
    np.testing.assert_allclose(_dense_rows(rt, Y, ch, d.t, d.hid), want, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("T", [8, 40, 64])
def test_attention_matches_softmax(T):
    """Causal attention through the THOR softmax of GPT-2 block 0 against float64 softmax attention."""
    rt, fhe, inf = _rt(N=4096, hid=128, dim=96, H=4, H_real=3)
    d = rt.dims
    rng = np.random.default_rng(T + 2)
    ch = prefill.Chunks(T, d.t)
    sm = config.load_configs(str(CFG)).block(0)[2]
    q, k, v = (0.5 * rng.standard_normal((T, d.hid)) for _ in range(3))
    for M in (q, k, v):
        M[:, d.H_real * d.d_head:] = 0.0
        for r in range(d.hid):                       # padded heads (r % H >= H_real) are zero
            if r % d.H >= d.H_real:
                M[:, r] = 0.0

    def dense(M):
        out = []
        for c in range(ch.n_ct):
            vec = np.zeros(d.N, dtype=np.complex128)
            for half in (0, 1):
                for lane, p in enumerate(ch.lanes(2 * c + half)):
                    vec[np.arange(d.hid) * d.t + lane] += 0.5 * (1j if half else 1.0) * M[p]
            out.append(fake.FakeCt(vec, 30))
        return out
    O = prefill.prefill_attention(rt, dense(q), dense(k), dense(v), sm, ch)
    got = _dense_rows(rt, O, ch, d.t, d.hid)
    want = np.zeros((T, d.hid))
    for h in range(d.H_real):
        cols = [j * d.H + h for j in range(d.d_head)]
        s = q[:, cols] @ k[:, cols].T / math.sqrt(d.d_head_real)
        s += np.triu(np.full((T, T), -np.inf), 1)
        p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        want[:, cols] = p @ v[:, cols]
    err = np.abs(got - want)                     # THOR's own error grows with kc
    assert err.max() < 3e-2 and np.median(err) < 1e-3


@pytest.mark.parametrize("T,keys,pack", [(8, 0, 0), (40, 0, 0), (64, 50, 0), (100, 0, 0), (128, 97, 0), (128, 0, 3)])
def test_attention_bidirectional(T, keys, pack, monkeypatch):
    """Chunks(bidir=True): every query over the first `keys` keys (all T if 0), against float64 softmax attention
    (SM_CHEB_GS reciprocal: the per-kc design holds at kc = keys for every row)."""
    monkeypatch.setattr(prefill, "_PACK", pack)
    rt, fhe, inf = _rt(N=4096, hid=128, dim=96, H=4, H_real=3)
    rt.ops.sm_cheb_gs = True
    d = rt.dims
    rng = np.random.default_rng(T + keys + 7)
    ch = prefill.Chunks(T, d.t, bidir=True, keys=keys)
    sm = config.load_configs(str(CFG)).block(0)[2]
    q, k, v = (0.5 * rng.standard_normal((T, d.hid)) for _ in range(3))
    for M in (q, k, v):
        for r in range(d.hid):
            if r % d.H >= d.H_real:
                M[:, r] = 0.0

    def dense(M):
        out = []
        for c in range(ch.n_ct):
            vec = np.zeros(d.N, dtype=np.complex128)
            for half in (0, 1):
                for lane, p in enumerate(ch.lanes(2 * c + half)):
                    vec[np.arange(d.hid) * d.t + lane] += 0.5 * (1j if half else 1.0) * M[p]
            out.append(fake.FakeCt(vec, 30))
        return out
    O = prefill.prefill_attention(rt, dense(q), dense(k), dense(v), sm, ch)
    got = _dense_rows(rt, O, ch, d.t, d.hid)
    n = ch.n_keys
    want = np.zeros((T, d.hid))
    for h in range(d.H_real):
        cols = [j * d.H + h for j in range(d.d_head)]
        s = q[:, cols] @ k[:n, cols].T / math.sqrt(d.d_head_real)
        p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        want[:, cols] = p @ v[:n, cols]
    err = np.abs(got - want)
    assert err.max() < 3e-2 and np.median(err) < 1e-3, (err.max(), np.median(err))


@pytest.mark.parametrize("T,pack", [(5, 1), (47, 2), (64, 3), (96, 2), (128, 3)])
def test_attention_packed_matches_unpacked(T, pack, monkeypatch):
    rt, fhe, inf = _rt(N=4096, hid=128, dim=128, H=4, H_real=4)
    d = rt.dims
    rng = np.random.default_rng(T)
    ch = prefill.Chunks(T, d.t)
    sm = config.load_configs(str(CFG)).block(0)[2]
    q, k, v = (0.5 * rng.standard_normal((T, d.hid)) for _ in range(3))

    def dense(M):
        out = []
        for c in range(ch.n_ct):
            vec = np.zeros(d.N, dtype=np.complex128)
            for half in (0, 1):
                for lane, p in enumerate(ch.lanes(2 * c + half)):
                    vec[np.arange(d.hid) * d.t + lane] += 0.5 * (1j if half else 1.0) * M[p]
            out.append(fake.FakeCt(vec, 30))
        return out
    pre = (rng.standard_normal(d.N) * 0.3, rng.standard_normal(d.N) * 0.3)
    outs = []
    for p in (0, pack):
        monkeypatch.setattr(prefill, "_PACK", p)
        O = prefill.prefill_attention(rt, dense(q), dense(k), dense(v), sm, ch, prefix=pre)
        outs.append(_dense_rows(rt, O, ch, d.t, d.hid))
    assert np.abs(outs[0] - outs[1]).max() < 1e-9


@pytest.mark.parametrize("pack", [0, 3])
def test_attention_kv_share(pack, monkeypatch):
    rt, fhe, inf = _rt(N=4096, hid=128, dim=128, H=4, H_real=4)
    d = rt.dims
    T = 96
    rng = np.random.default_rng(5)
    ch = prefill.Chunks(T, d.t)
    sm = config.load_configs(str(CFG)).block(0)[2]

    def dense(M):
        out = []
        for c in range(ch.n_ct):
            vec = np.zeros(d.N, dtype=np.complex128)
            for half in (0, 1):
                for lane, p in enumerate(ch.lanes(2 * c + half)):
                    vec[np.arange(d.hid) * d.t + lane] += 0.5 * (1j if half else 1.0) * M[p]
            out.append(fake.FakeCt(vec, 30))
        return out
    qs = [dense(0.5 * rng.standard_normal((T, d.hid))) for _ in range(3)]
    k, v = (dense(0.5 * rng.standard_normal((T, d.hid))) for _ in range(2))
    pre = (rng.standard_normal(d.N) * 0.3, rng.standard_normal(d.N) * 0.3)
    monkeypatch.setattr(prefill, "_PACK", pack)
    outs = []
    for share in (False, True):
        monkeypatch.setattr(prefill, "_KV_SHARE", share)
        O = prefill.prefill_attention(rt, qs, k, v, sm, ch, prefix=pre, shared=True)
        outs.append([_dense_rows(rt, o, ch, d.t, d.hid) for o in O])
    for a, b in zip(*outs):
        assert np.abs(a - b).max() < 1e-12


@pytest.mark.parametrize("T,pack,share", [(32, 0, False), (64, 3, True), (96, 0, True), (128, 2, False)])
def test_attention_periodic_prefix(T, pack, share, monkeypatch):
    rt, fhe, inf = _rt(N=4096, hid=128, dim=128, H=4, H_real=4)
    d = rt.dims
    rng = np.random.default_rng(T + 7)
    ch = prefill.Chunks(T, d.t)
    sm = config.load_configs(str(CFG)).block(0)[2]

    def dense(M):
        out = []
        for c in range(ch.n_ct):
            vec = np.zeros(d.N, dtype=np.complex128)
            for half in (0, 1):
                for lane, p in enumerate(ch.lanes(2 * c + half)):
                    vec[np.arange(d.hid) * d.t + lane] += 0.5 * (1j if half else 1.0) * M[p]
            out.append(fake.FakeCt(vec, 30))
        return out
    qs = [dense(0.5 * rng.standard_normal((T, d.hid))) for _ in range(2)]
    k, v = (dense(0.5 * rng.standard_normal((T, d.hid))) for _ in range(2))
    pre = (np.repeat(rng.standard_normal(d.hid) * 0.3, d.t), np.repeat(rng.standard_normal(d.hid) * 0.3, d.t))
    monkeypatch.setattr(prefill, "_PACK", pack)
    monkeypatch.setattr(prefill, "_KV_SHARE", share)
    outs = []
    for per in (False, True):
        monkeypatch.setattr(prefill, "_PRE_PERIODIC", per)
        O = prefill.prefill_attention(rt, qs, k, v, sm, ch, prefix=pre, shared=True)
        outs.append([_dense_rows(rt, o, ch, d.t, d.hid) for o in O])
    for a, b in zip(*outs):
        assert np.abs(a - b).max() < 1e-9


@pytest.mark.skipif(not os.environ.get("WEIGHTS_PATH") or not os.path.exists(os.environ.get("WEIGHTS_PATH", "")),
                    reason="needs the GPT-2 export (WEIGHTS_PATH)")
def test_gpt2_two_blocks():
    from examples.gpt2_from_primitives import weights
    from examples.gpt2_from_primitives.prefill import Gpt2Prefill, exact_blocks
    d = layout.Dims.gpt2()
    fhe, inf = fake.make_fake(N=d.N, hid=d.hid, dim=d.dim, H=d.H, H_real=d.H_real, E=d.E, E_real=d.E_real,
                              complex_payload=True)
    fhe.slots = None
    store = weights.RawStore(os.environ["WEIGHTS_PATH"])
    m = Gpt2Prefill(inf, store, config.load_configs(str(CFG)), core=fake.core, n_layers=2)
    rng = np.random.default_rng(0)
    T = 40
    wte, wpe = store.tensor(weights.WTE), store.tensor(weights.WPE)
    ids = rng.integers(0, wte.shape[0], T)
    rows = wte[ids] + wpe[:T]
    Xs, ch, _ = m.prefill(rows)
    got = m.hidden(Xs, ch)
    want = exact_blocks(store, rows, 2)[-1]
    rel = np.abs(got - want).max(1) / np.abs(want).max(1)
    assert np.median(rel) < 0.02 and rel.max() < 0.1


@pytest.mark.parametrize("op", [False, True])
def test_pair_linears_complex_weights(op):
    """The decode's complex weights on token pairs: a fused W0 + i W1 (two outputs) or output-packed W."""
    rt, fhe, inf = _rt()
    d = rt.dims
    rng = np.random.default_rng(7 + op)
    T = 40
    ch = prefill.Chunks(T, d.t)
    rows = _tokens(rng, T, d.hid)
    Xs = []
    for c in range(ch.n_ct):
        v = np.zeros(d.N, dtype=np.complex128)
        for half in (0, 1):
            for lane, p in enumerate(ch.lanes(2 * c + half)):
                v[np.arange(d.hid) * d.t + lane] += (1j if half else 1.0) * rows[p]
        Xs.append(fake.FakeCt(v, 30))
    if op:
        W = rng.standard_normal((d.hid, d.E))
        w = EncodedLinear.encode(W, d.N, d.hid, d.E, outputpack=True)
        (H,) = prefill.dense_linear(rt, Xs, [w], ch, d.t, d.tp_E, 1.0, 1.0)
        (Y,) = prefill.dense_linear(rt, H, [EncodedLinear.encode(np.eye(d.E, d.hid), d.N, d.E, d.hid)], ch,
                                    d.tp_E, d.t, 1.0, 1.0)
        np.testing.assert_allclose(_dense_rows(rt, Y, ch, d.t, d.hid), (rows @ W)[:, :d.hid], atol=1e-8)
    else:
        W0, W1 = rng.standard_normal((2, d.hid, d.hid))
        w = EncodedLinear.encode(W0 + 1j * W1, d.N, d.hid, d.hid)
        A, B = prefill.dense_linear(rt, Xs, [w], ch, d.t, d.t, 1.0, 1.0)
        np.testing.assert_allclose(_dense_rows(rt, A, ch, d.t, d.hid), rows @ W0, atol=1e-8)
        np.testing.assert_allclose(_dense_rows(rt, B, ch, d.t, d.hid), rows @ W1, atol=1e-8)


@pytest.mark.parametrize("n_in,n_out", [(1, 1), (1, 3), (3, 1)])
def test_diag_linear(n_in, n_out):
    """The dense diagonal BSGS linear on complex token-layout blocks equals x @ W per token, both chunks."""
    rt, fhe, inf = _rt()
    d = rt.dims
    B = d.N // d.t
    rng = np.random.default_rng(n_in * 10 + n_out)
    T = 2 * d.t - 5
    ch = prefill.Chunks(T, d.t)
    X = rng.standard_normal((T, n_in * B))
    W = rng.standard_normal((n_in * B, n_out * B))
    bias = rng.standard_normal(n_out * B)
    lin = prefill.DiagLinear.encode(W, d.N, d.t, 0.5, bias)
    blocks = []
    for n in range(n_in):
        v = np.zeros(d.N, dtype=np.complex128)
        for half in (0, 1):
            for lane, p in enumerate(ch.lanes(half)):
                v[np.arange(B) * d.t + lane] += (1j if half else 1.0) * X[p, n * B:(n + 1) * B]
        blocks.append(fake.FakeCt(v, 30))
    (Y,) = prefill.diag_apply(rt, blocks, [lin])
    want = 0.5 * (X @ W + bias)
    for m in range(n_out):
        y = prefill.diag_bias(rt, Y[m], lin, m, ch, 0)
        got = _dense_rows(rt, [y], ch, d.t, B)
        np.testing.assert_allclose(got, want[:, m * B:(m + 1) * B], atol=1e-8)
