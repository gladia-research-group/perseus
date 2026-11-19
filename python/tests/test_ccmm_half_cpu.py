"""The CCMM attention's half-head view (sinc.half_perm: d_head 256 as Llama's KV heads of 128), its n-way output merge
(ccmm PF_CCMM_MERGE4: quadruples per SinCBoot) and the compressed softmax layout (sinc_softmax PF_SM_HALF_PACK), on
numpy: the channel permutation is a bijection keeping RoPE's pairs N / 2 apart; unmerge (n = 2, 4) and unmerge_half
recover the outputs of a duplicated-operand CCMM exactly; the compressed layout tiles the batch slots, keeps every key,
and its denominator ladder equals the direct per-row key sum."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perseus.impl import ccmm as cm  # noqa: E402
from perseus.impl import sinc as sc  # noqa: E402
from perseus.impl import sinc_softmax as ss  # noqa: E402

S = sc.SLOTS
B = (np.arange(S) - sc.OFFSET) % 32           # batch slot of every SinC slot


class _Ops:
    def rotate(self, x, k): return np.roll(x, -k)
    def rotate_many(self, x, ks): return [np.roll(x, -k) for k in ks]
    def add(self, a, b): return a + b
    def sub(self, a, b): return a - b


class _Rt:
    ops = _Ops()
    def mult_mask(self, x, key, f): return x * f()
    def mult_mask_sum(self, xs, keys, fs): return sum(x * f() for x, f in zip(xs, fs))


def test_half_perm_bijection_and_rope_pairs():
    p = sc.half_perm(512)
    assert sorted(p) == list(range(512))
    c = np.arange(512)
    j, kvlo = c >> 1, c & 1
    partner = ((j ^ 128) << 1) | kvlo            # RoPE pairs dims j, j + 128 of the same head
    assert np.all(np.abs(p[partner] - p) * 64 == 512 * 64 // 2)


@pytest.mark.parametrize("n", [2, 4])
def test_unmerge_exact(n):
    rng = np.random.default_rng(n)
    step = 32 // n
    w = cm.w_theory()
    Y = []
    for _ in range(16):                           # outputs of operands duplicated on l + step k: replicated
        y = rng.standard_normal(S) + 1j * rng.standard_normal(S)
        z = y.copy()
        for k in range(1, n):
            z = np.where((B >= k * step) & (B < (k + 1) * step), np.roll(y, step * k), z)
        Y.append(z)
    M = cm._group(np.array(Y), w, n)
    got = cm.unmerge(_Rt(), list(M), w, n=n)
    sel = B < step
    for g, y in zip(got, Y):
        np.testing.assert_allclose(g[sel], y[sel], atol=1e-12)
        assert np.abs(g[~sel]).max() == 0.0
    keep = (B < 8) & (((B // sc.HALF_STEP) % 2) == 0)
    for g, y in zip(cm.unmerge_half(_Rt(), list(M), w, n=n), Y):
        np.testing.assert_allclose(g, np.where(keep, y + np.roll(y, -sc.HALF_STEP), 0), atol=1e-12)


def test_compressed_layout_and_denominator():
    r, l = sc.half_regions()
    assert np.bincount(r).tolist() == [S // 4] * 4
    tq, tk = sc.half_meta()
    assert sorted(set(tk.ravel().tolist())) == list(range(128))
    rng = np.random.default_rng(3)
    V = np.isin(B, [0, 1, 4, 5, 16, 17, 20, 21])
    Spk = [np.where(V, rng.random(S), 0.0) for _ in range(16)]
    C = ss.compress_half(_Rt(), Spk)
    D = sum(C)
    for k in (1, 2, 4):                           # the key ladder over slot bits 12..14
        D = D + np.roll(D, -(k << 12))
    D = ss._half_den(_Rt(), D)
    want = sum(Spk)
    for k in (1, 2, 4):
        want = want + np.roll(want, -(k << 12))
    for rr, dl in enumerate(sc.HALF_DELTAS):      # every region carries its rows' total
        sel = r == rr
        np.testing.assert_allclose(D[sel], np.roll(want, dl)[sel], rtol=1e-12)
