"""The model-independent pieces the Llama port added to perseus.impl, on the CPU: the shape-cached weight gather,
RoPE folded into the head-interleaved layout, RMSNorm on the fake session, and the BOS-prefix cache layout.
(The end-to-end check against the HF model needs the checkpoint: examples/llama_from_primitives.)"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perseus.impl import attention, config, fake, layout, norm  # noqa: E402
from perseus.impl.ops import FheOps  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402


@pytest.mark.parametrize("N,d_in,d_out", [(1024, 32, 32), (1024, 32, 128), (1024, 128, 32), (1024, 32, 1024),
                                          (4096, 64, 256), (4096, 256, 64)])
def test_weight_gather_matches_encoder(N, d_in, d_out):
    rng = np.random.default_rng(d_in * d_out)
    W = rng.standard_normal((d_in, d_out))
    assert np.array_equal(layout.encode_weight_matrix_fast(W, N, d_in, d_out),
                          layout.encode_weight_matrix(W, N, d_in, d_out))
    Wc = W + 1j * rng.standard_normal((d_in, d_out))
    assert np.array_equal(layout.encode_weight_matrix_fast(Wc, N, d_in, d_out),
                          layout.encode_weight_matrix(Wc, N, d_in, d_out))
    if layout.cm_params(N, d_in, d_out).r_o % 2 == 0:
        assert np.array_equal(layout.encode_weight_matrix_fast(W, N, d_in, d_out, True),
                              layout.encode_weight_matrix_outputpack(W, N, d_in, d_out))


def _rope_ref(x, inv_freq, pos):
    """HF Llama: x cos + rotate_half(x) sin on one head (d_head,)."""
    ang = pos * np.asarray(inv_freq)
    c, s = np.concatenate([np.cos(ang)] * 2), np.concatenate([np.sin(ang)] * 2)
    h = x.shape[-1] // 2
    return x * c + np.concatenate([-x[h:], x[:h]]) * s


def test_rope_mask_matches_rotate_half():
    N, hid, H = 1024, 128, 4                     # t = 8, d_head = 32, tH = 32
    fhe, inf = fake.make_fake(N=N, hid=hid, dim=hid, H=H, H_real=H, E=256, E_real=256, complex_payload=True)
    rt = Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))
    d = rt.dims
    rng = np.random.default_rng(1)
    feats = rng.standard_normal(hid)             # head-interleaved: feature lane * H + h
    inv_freq = 1.0 / (500000.0 ** (np.arange(0, d.d_head, 2) / d.d_head))
    rope = attention.Rope(inv_freq)
    for pos in (0, 1, 7, 123):
        ct = fake.FakeCt(layout.encode_linear_input(feats, N, hid, hid), 34)
        out = attention.rope_mask(rt, ct, "tok0", lambda: layout.real_head_tok0_mask(d), rope, pos).vec
        got = out[np.arange(hid) * d.t].real.reshape(d.d_head, H)          # [lane, h]
        want = np.stack([_rope_ref(feats.reshape(d.d_head, H)[:, h], inv_freq, pos) for h in range(H)], 1)
        np.testing.assert_allclose(got, want, atol=1e-12)
        off = np.ones(N, bool); off[np.arange(hid) * d.t] = False
        assert np.abs(out[off]).max() == 0.0          # the mask still selects tok0 only


def test_rms_norm_on_fake():
    N, hid = 1024, 128
    fhe, inf = fake.make_fake(N=N, hid=hid, dim=hid, H=4, H_real=4, E=256, E_real=256, complex_payload=True)
    rt = Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))
    x = np.random.default_rng(2).standard_normal(hid) * 0.3
    c, eps, ios = 4.0, 1e-5, 0.5
    z = c * c * (np.mean(x * x) + eps)
    lo, hi = z / 3, z * 3
    t = np.cos(np.pi * (np.arange(200) + 0.5) / 200)
    coeffs = np.polynomial.chebyshev.chebfit(t, ios / np.sqrt((t + 1) / 2 * (hi - lo) + lo), 12)
    cfg = config.NormCfg(method="remez", epsilon=eps, center_scale=c, taylor_z0=0.0, inv_out_scale=ios,
                         cheb_coeffs=tuple(coeffs), cheb_lo=lo, cheb_hi=hi, cheb_nr_iters=2)
    ct = fake.core.encode_token_input(inf, x)
    y = fake.core.decode_token_output(inf, norm.rms_norm(rt, ct, cfg, 1))
    want = ios * x / math.sqrt(np.mean(x * x) + eps)
    np.testing.assert_allclose(y, want, rtol=1e-6, atol=1e-8)


def test_magtape_coeffs_invert_the_embedding():
    """node_stats' coefficients: sum_k Re(u_k) X^k + Im(u_k) X^(k+n) evaluates to slot j at zeta^(5^j)."""
    from perseus.impl import magtape
    n = 16
    rng = np.random.default_rng(3)
    z = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    m, tw = magtape._coeff_perm(n)
    y = np.empty(n, complex); y[m] = z
    u = np.fft.fft(y) * tw / n
    coeffs = np.concatenate([u.real, u.imag])                       # degree < 2n
    zeta = np.exp(2j * np.pi / (4 * n))
    for j in range(n):
        x = zeta ** pow(5, j, 4 * n)
        assert abs(np.polyval(coeffs[::-1], x) - z[j]) < 1e-12
    st = magtape.node_stats(z)
    assert np.isclose(st["output_max_coeff"], np.abs(coeffs).max())
    assert np.isclose(st["output_max_abs"], np.abs(z).max())
