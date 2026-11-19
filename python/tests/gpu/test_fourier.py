"""impl/fourier.py encrypted on the GPU: DFT (direct, planned natural order, bit-reversed pair), the real-input FFTs,
DCT-II / DCT-III (real and pair packings), a 2-D DCT on 32 x 32 images, against numpy / scipy in float64 (relative
error on the decoded slots <= 1e-8; measured ~1e-10) and the levels each one spends.

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -m gpu python/tests/gpu/test_fourier.py -q
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.fft as sf

pytestmark = pytest.mark.gpu

_pc = pytest.importorskip("perseus._pc")

S = 1 << 15
TOL = 1e-8


def _cases():
    from perseus.impl import fourier as F
    rng = np.random.default_rng(3)
    lay64 = F.Layout(64, S, S // 64, "stride")
    lay1k = F.Layout(1024, S, S // 1024, "stride")
    cx = rng.standard_normal((S // 64, 64)) + 1j * rng.standard_normal((S // 64, 64))
    ck = rng.standard_normal((S // 1024, 1024)) + 1j * rng.standard_normal((S // 1024, 1024))
    xr, yr = rng.standard_normal((S // 64, 64)), rng.standard_normal((S // 64, 64))
    xh = rng.standard_normal((S // 32, 64))
    img = rng.standard_normal((S // 1024, 32, 32))
    br = np.argsort(F.bit_reverse(1024))
    h = 32
    return {
        "dft64_direct": (F.dft(64, layout=lay64, direct=True, norm="ortho"), cx, np.fft.fft(cx, norm="ortho"), None),
        "dft1024_natural": (F.dft(1024, layout=lay1k, norm="ortho"), ck, np.fft.fft(ck, norm="ortho"),
                            F.idft(1024, layout=lay1k, norm="ortho")),
        "dft1024_bitrev": (F.dft(1024, layout=lay1k, norm="ortho", order="bitrev"), ck,
                           np.fft.fft(ck, norm="ortho")[:, F.bit_reverse(1024)],
                           F.idft(1024, layout=lay1k, norm="ortho", order="bitrev")),
        "rfft64_pair": (F.rfft_pair(64, layout=lay64, norm="ortho"), F.pack_real_pair(xr, yr),
                        np.concatenate([F.pack_rfft(np.fft.rfft(xr, norm="ortho"), 64),
                                        F.pack_rfft(np.fft.rfft(yr, norm="ortho"), 64)], 1),
                        F.rfft_pair(64, layout=lay64, norm="ortho", inverse=True)),
        "rfft64_half": (F.rfft_half(64, layout=F.Layout(h, S, S // h, "stride"), norm="ortho"), F.pack_real_half(xh),
                        F.pack_rfft(np.fft.rfft(xh, norm="ortho"), 64),
                        F.rfft_half(64, layout=F.Layout(h, S, S // h, "stride"), norm="ortho", inverse=True)),
        "dct64_real": (F.dct2(64, layout=lay64), xr + 0j, sf.dct(xr, 2, norm="ortho"), F.dct3(64, layout=lay64)),
        "dct64_pair": (F.dct2(64, layout=lay64, packing="pair"), F.pack_real_pair(xr, yr),
                       sf.dct(xr, 2, norm="ortho") + 1j * sf.dct(yr, 2, norm="ortho"),
                       F.dct3(64, layout=lay64, packing="pair")),
        "dct2d_32": (F.dct2_2d(32, layout=lay1k, separable=True), img.reshape(-1, 1024) + 0j,
                     sf.dctn(img, 2, axes=(1, 2), norm="ortho").reshape(-1, 1024),
                     F.dct3_2d(32, layout=lay1k, separable=True)),
    }


@pytest.fixture(scope="module")
def env():
    cases = _cases()
    steps = set()
    for T, _, _, inv in cases.values():
        steps |= T.rot_steps() | (inv.rot_steps() if inv else set())
    cfg = _pc.Config.preset("n32_gpt2_decode")
    keys = _pc.KeySource.generate(sorted(steps), seed=13)
    sess = _pc.Session(cfg, keys, pool=128, pt_slots=160).complete_setup()
    from perseus.impl.ops import FheOps
    yield cases, sess, FheOps.from_session(sess)
    sess.close()


def _rel(got, want):
    return float(np.max(np.abs(got - want)) / np.sqrt(np.mean(np.abs(want) ** 2)))


@pytest.mark.parametrize("name", ["dft64_direct", "dft1024_natural", "dft1024_bitrev", "rfft64_pair", "rfft64_half",
                                  "dct64_real", "dct64_pair", "dct2d_32"])
def test_encrypted(env, name):
    cases, sess, ops = env
    T, x, want, inv = cases[name]
    fhe = sess.fhe
    a = fhe.encrypt(T.pack(x), 0)
    y = T.apply(ops, a)
    assert ops.lvl(y) == T.levels * ops.unit
    got = T.unpack(fhe.decrypt_slots_complex(y))
    assert _rel(got, want) < TOL
    if inv is not None:
        z = inv.apply(ops, y)
        assert _rel(inv.unpack(fhe.decrypt_slots_complex(z)), x) < TOL
    from perseus.impl import fourier as F
    F.release(ops)
