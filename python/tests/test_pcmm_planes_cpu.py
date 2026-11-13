"""PcmmLinear.encode's digit planes: the int32 path and deferred_planes give the int64 encoding's planes exactly."""
import numpy as np
import pytest

from perseus.impl import pcmm


def _planes_int64(W, t, ws, nb):
    """The int64 encoding (encode before the int32 path), for reference."""
    sh = pcmm.Shape(t)
    n_in, n_out = W.shape[0] // sh.nch, W.shape[1] // sh.nch
    Wt = np.rint(W.T * ws).astype(np.int64)
    perm = sh.ch_of_row()
    Wi = np.empty_like(Wt)
    for o in range(n_out):
        for c in range(n_in):
            blk = Wt[o * sh.nch:(o + 1) * sh.nch, c * sh.nch:(c + 1) * sh.nch]
            Wi[o * sh.nch:(o + 1) * sh.nch, c * sh.nch:(c + 1) * sh.nch] = blk[perm[:, None], perm[None, :]]
    v, out = Wi.copy(), np.empty((nb,) + Wi.shape, dtype=np.int8)
    for u in range(nb):
        r = ((v + 128) % 256) - 128
        out[u] = r
        v = (v - r) // 256
    assert not np.any(v)
    return out


@pytest.mark.parametrize("wbits", [8, 16, 24])
def test_planes_match_int64(wbits):
    t = 64
    nch = pcmm.Shape(t).nch
    rng = np.random.default_rng(wbits)
    W = rng.standard_normal((2 * nch, 3 * nch)) * 0.05
    W[0, 0] = 0.5                                  # the largest weight sits on the digits' reach
    lin = pcmm.PcmmLinear.encode(W, t, wbits=wbits)
    ref = _planes_int64(W, t, lin.w_scale, (wbits + 7) // 8)
    assert lin.Wd.dtype == np.int8 and np.array_equal(lin.Wd, ref)
    np.testing.assert_array_equal(pcmm.from_planes(lin.Wd), pcmm.from_planes(ref))


def test_deferred_planes():
    t = 64
    nch = pcmm.Shape(t).nch
    rng = np.random.default_rng(0)
    Ws = [rng.standard_normal((nch, 2 * nch)) for _ in range(5)]
    with pcmm.deferred_planes(3) as jobs:
        lins = [pcmm.PcmmLinear.encode(W, t, bias=np.ones(2 * nch), name=f"l{i}") for i, W in enumerate(Ws)]
        assert all(lin.Wd is None for lin in lins) and len(jobs) == len(lins)
    for W, lin in zip(Ws, lins):
        now = pcmm.PcmmLinear.encode(W, t, bias=np.ones(2 * nch))
        assert np.array_equal(lin.Wd, now.Wd) and lin.w_scale == now.w_scale and lin.n_out == now.n_out
        assert all(np.array_equal(a, b) for a, b in zip(lin.bias, now.bias))
