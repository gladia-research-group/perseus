"""perseus.impl.blocked (channel-blocked prefill) on the numpy fake."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perseus.impl import blocked, config, fake, layout, prefill  # noqa: E402
from perseus.impl.ops import FheOps  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402

CFG = ROOT / "configs/model/approximation/gpt2_base_n32/configs.json"
DIMS = dict(N=4096, hid=256, dim=192, H=4, H_real=3, E=512, E_real=384)   # d_head 64 >= t = 64


def _rt(**kw):
    fhe, inf = fake.make_fake(complex_payload=True, **kw)
    fhe.slots = None
    return Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf)), fhe, inf


@pytest.mark.parametrize("T", [5, 64, 100, 128, 200])
def test_encode_roundtrip(T):
    rt, _, _ = _rt(**DIMS)
    B = blocked.Blocks(rt.dims, nch=64)
    ch = prefill.Chunks(T, B.t)
    rows = np.random.default_rng(T).standard_normal((T, rt.dims.dim))
    Xs = blocked.encode_blocked(rt, rows, ch, B)
    assert len(Xs) == ch.n_ct and all(len(x) == B.K for x in Xs)
    np.testing.assert_allclose(blocked.decrypt_blocked(rt, Xs, ch, B, rt.dims.dim), rows, atol=1e-12)


def test_head_perm_groups_whole_heads():
    d = layout.Dims(32768, 4096, 4096, 32, 32, 16384, 14336)
    B = blocked.Blocks(d, nch=512)
    p = B.head_perm()
    assert sorted(p) == list(range(d.hid))
    r = np.arange(d.hid)
    j, h = r // d.H, r % d.H
    assert np.all(p // B.nch == h // B.hg)                      # head h lives in block h // H'
    loc = p % B.nch
    assert np.all(loc // B.hg == j) and np.all(loc % B.hg == h % B.hg)


@pytest.mark.parametrize("rms", [False, True])
@pytest.mark.parametrize("T", [40, 128, 180])
def test_norm_blocked_matches_norm_all(rms, T):
    rt, _, _ = _rt(**DIMS)
    d = rt.dims
    B = blocked.Blocks(d, nch=64)
    cfg = config.load_configs(str(CFG)).block(0)[0]
    rows = np.random.default_rng(T + 7).standard_normal((T, d.dim)) * 0.8 + 0.1
    ch_b = prefill.Chunks(T, B.t)
    got = blocked.decrypt_blocked(rt, blocked.norm_blocked(rt, blocked.encode_blocked(rt, rows, ch_b, B), cfg, ch_b,
                                                           B, rms=rms), ch_b, B, d.dim)
    ch = prefill.Chunks(T, d.t)
    want = prefill.decrypt_prompt(rt, prefill.norm_all(rt, prefill.encode_prompt(rt, rows, ch), cfg, ch, rms=rms),
                                  ch, d.dim)
    np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("T", [40, 64, 128, 150, 256])
def test_attention_blocked_matches_unblocked(T):
    """Per head group == the one-ciphertext attention (same THOR ops); float softmax while THOR is accurate."""
    rt, _, _ = _rt(**DIMS)
    d = rt.dims
    B = blocked.Blocks(d, nch=64)
    sm = config.load_configs(str(CFG)).block(0)[2]
    rng = np.random.default_rng(T + 2)
    q, k, v = (0.5 * rng.standard_normal((T, d.hid)) for _ in range(3))
    for M in (q, k, v):
        for r in range(d.hid):                       # padded heads (r % H >= H_real) are zero
            if r % d.H >= d.H_real:
                M[:, r] = 0.0
    perm = B.head_perm()
    chb = prefill.Chunks(T, B.t)

    def grouped(M):                                  # interleaved -> head-grouped, at the attention's 1/2
        G = np.zeros_like(M)
        G[:, perm] = M
        return blocked.encode_blocked(rt, 0.5 * G, chb, B)
    O = blocked.attention_blocked(rt, grouped(q), grouped(k), grouped(v), sm, chb, B)
    got = blocked.decrypt_blocked(rt, O, chb, B, d.hid)[:, perm]
    ch = prefill.Chunks(T, d.t)
    enc = lambda M: prefill.encode_prompt(rt, 0.5 * M, ch)
    ref = prefill.decrypt_prompt(rt, prefill.prefill_attention(rt, enc(q), enc(k), enc(v), sm, ch), ch, d.hid)
    np.testing.assert_allclose(got, ref, atol=1e-10)
    if T <= 40:
        want = np.zeros((T, d.hid))
        for h in range(d.H_real):
            cols = [j * d.H + h for j in range(d.d_head)]
            s = q[:, cols] @ k[:, cols].T / math.sqrt(d.d_head_real)
            s += np.triu(np.full((T, T), -np.inf), 1)
            p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            want[:, cols] = p @ v[:, cols]
        err = np.abs(got - want)
        assert err.max() < 3e-2 and np.median(err) < 1e-3
