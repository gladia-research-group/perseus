"""perseus.impl.sinc_mha on the numpy fake: the d_head-64 SinC conversions against the explicit matrices, and S = Q K^T /
O = P V per head through the exact CKL product."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perseus.impl import ccmm as cm  # noqa: E402
from perseus.impl import fake, layout  # noqa: E402
from perseus.impl import sinc as sc  # noqa: E402
from perseus.impl import sinc_mha as sm  # noqa: E402
from perseus.impl.ops import FheOps  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402

H = 12


def _rt():
    fhe, inf = fake.make_fake(complex_payload=True, N=1 << 15, hid=1024, dim=768, H=16, H_real=12, E=4096,
                              E_real=3072)
    fhe.slots = None
    return Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))


def _blocked(M):
    """(16 heads, 128 tokens, 64 dims) -> 2 complex block vectors (Re / Im = chunk)."""
    out = []
    for k in range(2):
        v = np.zeros(sc.SLOTS, dtype=np.complex128)
        for m in range(sm.HB):
            h = sm.HB * k + m
            for j in range(sm.DH):
                v[j * 512 + m * 64 + np.arange(64)] = M[h, :64, j] + 1j * M[h, 64:, j]
        out.append(v)
    return out


def _chunks(M):
    """(16, 128, 64) -> X[k][c] real per-chunk vectors."""
    X = []
    for k in range(2):
        row = []
        for c in range(2):
            v = np.zeros(sc.SLOTS)
            for m in range(sm.HB):
                for j in range(sm.DH):
                    v[j * 512 + m * 64 + np.arange(64)] = M[sm.HB * k + m, 64 * c:64 * c + 64, j]
            row.append(v)
        X.append(row)
    return X


def test_qk_pv_round_trip():
    rt = _rt()
    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal((16, 128, 64)) for _ in range(3))
    q[H:] = k[H:] = v[H:] = 0.0
    Q = [fake.FakeCt(x, 30) for x in _blocked(q)]
    K = [[fake.FakeCt(x, 30) for x in row] for row in _chunks(k)]
    V = [[fake.FakeCt(x, 30) for x in row] for row in _chunks(v)]
    Qs, Ks, Vs = sm.to_sinc_q(rt, Q), sm.to_sinc_kv(rt, K), sm.to_sinc_kv(rt, V)
    S = cm.exact([x.vec for x in Qs], [x.vec for x in Ks], True)
    Sm = cm.to_mats(S)
    for h in range(H):
        hp, l = h % 2, sm.batch_of(h)
        rows = sc.brev(np.arange(64), 6) + 64 * hp
        keys = sc.brev(np.arange(128) % 64, 6) + 64 * (np.arange(128) // 64)
        want = q[h] @ k[h].T                                    # (128 queries, 128 keys)
        got = Sm[rows][:, keys, l]                              # Re = chunk 0 queries, Im = chunk 1
        np.testing.assert_allclose(got.real, want[:64], atol=1e-9)
        np.testing.assert_allclose(got.imag, want[64:], atol=1e-9)
    p = rng.random((16, 128, 128))
    P = np.zeros((128, 128, 32), dtype=np.complex128)
    for h in range(H):
        hp, l = h % 2, sm.batch_of(h)
        rows = sc.brev(np.arange(64), 6) + 64 * hp
        keys = sc.brev(np.arange(128) % 64, 6) + 64 * (np.arange(128) // 64)
        P[np.ix_(rows, keys, [l])] = (p[h, :64] + 1j * p[h, 64:])[:, :, None]
    O = cm.exact(cm.from_mats(P), [x.vec for x in Vs], False)
    out = sm.from_sinc_q(rt, [fake.FakeCt(x, 30) for x in O])
    want = np.stack([p[h] @ v[h] for h in range(16)])
    want[H:] = 0.0
    got = _blocked(want)
    for a, b in zip(out, got):
        np.testing.assert_allclose(a.vec, b, atol=1e-9)
