"""Numpy model of the cachemir VMM packing (encode x, encode W, rotate/accumulate, decode).

The algebra the CUDA packer implements, checked against x @ W to 1e-10 on the three
shapes a transformer MLP needs (square, up-projection, down-projection). Promoted from
scripts/dev/linear.py; shapes are scaled down so the pure-Python encode loops stay in
the seconds range (the algorithm is shape-generic in N, d, alpha).
"""
import numpy as np
import pytest


def rot(v, k):
    return np.roll(v, -k)


def compute_params(N, d, alpha, is_up):
    t = N // d
    tp = N // (alpha * d)
    tp_in = t if is_up else tp
    tp_out = tp if is_up else t
    d_in = d if is_up else alpha * d
    n_pt = d_in // tp_out
    r_i = min(max(1, d * d // N), n_pt)
    r_o = n_pt // r_i
    return t, tp, tp_in, tp_out, r_i, r_o, n_pt


def interleave_idx(m, d, dim):
    a = dim // d if dim > d else 1
    return (m // a + (m % a) * d) % dim


def encode_x(x, N, d, alpha, is_up):
    t, tp, *_ = compute_params(N, d, alpha, is_up)
    d_in = d if is_up else alpha * d
    ptx = np.zeros(N)
    if is_up:
        ptx[np.arange(d) * t] = x
    else:
        M = N // tp
        for m in range(M):
            ptx[m * tp] = x[interleave_idx(m, d, d_in)]
    return ptx


def encode_W(W, N, d, alpha, is_up):
    d_in, d_out = W.shape
    t, tp, tp_in, tp_out, r_i, r_o, n_pt = compute_params(N, d, alpha, is_up)
    M_out = N // tp_out
    cascade_shift = (t * tp) // tp_out
    i = np.arange(N)
    row = ((i // t)[None, :] + (np.arange(r_i) * t)[:, None] + (i % tp_in)[None, :]) % d \
        + (((i % t) // tp_in) * d)[None, :]                                  # (r_i, N)
    pt = np.zeros((n_pt, N))
    for k in range(r_o):
        m_shifted = (i // tp_out - k * cascade_shift) % M_out
        col = np.array([interleave_idx(int(m), d, d_out) for m in m_shifted])
        for j in range(r_i):
            pt[j * r_o + k] = W[row[j], col]
    return pt


def decode_output(cy, N, t, tp, alpha, d, d_out, is_up):
    if is_up and alpha > 1:
        M = N // tp
        y = np.zeros(d_out)
        for m in range(M):
            idx = interleave_idx(m, d, d_out)
            if idx < d_out:
                y[idx] = cy[m * tp]
        return y
    return cy[::t][:d_out]


def cachemir_vmm(x, W, N, params):
    t, tp, tp_in, tp_out, r_i, r_o, _ = params
    ptx = x.copy()
    step = 1
    while step < tp_in:
        ptx = ptx + rot(ptx, step * (t - 1))
        step *= 2
    ptx_rot = [rot(ptx, j * t * t) for j in range(r_i)]
    cy = np.zeros((r_o, N))
    for k in range(r_o):
        for j in range(r_i):
            cy[k] += ptx_rot[j] * W[j * r_o + k]
    for k in range(r_o - 1, 0, -1):
        cy[k - 1] += rot(cy[k], t * tp)
    out = cy[0]
    step = 1
    while step < tp_out:
        out = out + rot(out, step)
        step *= 2
    return out


@pytest.mark.parametrize("label, N, d, alpha, is_up", [
    ("square", 4096, 64, 1, True),
    ("up", 4096, 64, 4, True),
    ("down", 4096, 64, 4, False),
    ("square-wide", 16384, 256, 1, True),     # r_i > 1: the cascade path
    ("up-wide", 16384, 128, 4, True),
])
def test_packed_vmm_matches_x_at_w(label, N, d, alpha, is_up):
    rng = np.random.default_rng(42)
    d_in, d_out = (d, alpha * d) if is_up else (alpha * d, d)
    x, W = rng.standard_normal(d_in), rng.standard_normal((d_in, d_out))
    params = compute_params(N, d, alpha, is_up)
    cy = cachemir_vmm(encode_x(x, N, d, alpha, is_up), encode_W(W, N, d, alpha, is_up), N, params)
    y = decode_output(cy, N, params[0], params[1], alpha, d, d_out, is_up)
    assert np.max(np.abs(x @ W - y)) < 1e-10, label
