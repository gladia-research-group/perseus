"""cachemir geometry, weight/bias slot layouts and every mask vector, in numpy.

Ports (all indices are slot indices into the N = slots vector):
  compute_cm_params / interleave_idx / encode_weight_matrix / encode_bias_vector /
  encode_linear_input / decode_linear_output / decode_tokens
      -> src/packing/cachemir/cachemir_linear_utils.cu
  stride / active-token / active-expanded masks -> include/inference.h,
      include/packing/cachemir/cachemir_masks.h
  attention masks -> include/packing/cachemir/cachemir_attention_utils.h
  rearrange_* -> src/packing/cachemir/cachemir_attention_utils.cu
  cutmax_tile_col_of_slot -> include/cutmax.h
"""
from __future__ import annotations

import dataclasses
import functools
import math

import numpy as np


@dataclasses.dataclass(frozen=True)
class Dims:
    """Model geometry (ModelSize, include/inference.h) plus the derived strides."""
    N: int            # slots
    hid: int          # hidDim (padded hidden width)
    dim: int          # real hidden width
    H: int            # numHeads (padded)
    H_real: int       # numHeadsReal
    E: int            # expDim (padded MLP width)
    E_real: int       # expanded (real MLP width)

    @property
    def t(self): return self.N // self.hid
    @property
    def tH(self): return self.t * self.H
    @property
    def d_head(self): return self.hid // self.H
    @property
    def d_head_real(self): return self.dim // self.H_real
    @property
    def tp_E(self): return self.N // self.E

    @classmethod
    def from_inf(cls, inf):
        s = inf.size
        return cls(int(inf.slots), int(s.hidDim), int(s.dim), int(s.numHeads),
                   int(s.numHeadsReal), int(s.expDim), int(s.expanded))

    @classmethod
    def gpt2(cls, logN=16):
        return cls(1 << (logN - 1), 1024, 768, 16, 12, 4096, 3072)


# ── linear packing ───────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class CmParams:
    is_up: bool
    d: int
    alpha: int
    t: int
    tp: int
    tp_in: int
    tp_out: int
    n_pt: int
    r_i: int
    r_o: int
    bstep_c: int
    gstep_c: int


@functools.lru_cache(maxsize=None)
def cm_params(N: int, d_in: int, d_out: int) -> CmParams:
    """compute_cm_params (cachemir_linear_utils.cu)."""
    is_up = d_in <= d_out
    d = d_in if is_up else d_out
    alpha = max(d_in, d_out) // d
    t = N // d
    tp = N // (alpha * d)
    tp_in = t if is_up else tp
    tp_out = tp if is_up else t
    d_ = d if is_up else alpha * d
    n_pt = d_ // tp_out
    r_i = max(1, d * d // N)
    r_i = min(r_i, n_pt)
    r_o = n_pt // r_i
    bstep_c, gstep_c = r_i, 1
    for b in range(1, r_i + 1):
        if r_i % b:
            continue
        g = r_i // b
        if b + g < bstep_c + gstep_c or (b + g == bstep_c + gstep_c and b > bstep_c):
            bstep_c, gstep_c = b, g
    return CmParams(is_up, d, alpha, t, tp, tp_in, tp_out, n_pt, r_i, r_o, bstep_c, gstep_c)


def interleave_idx(m, d: int, dim: int):
    """interleave_idx (cachemir_linear_utils.cu); works on ints and int arrays."""
    a = dim // d if dim > d else 1
    return (m // a + (m % a) * d) % dim


def linear_rot_steps(N: int, d_in: int, d_out: int) -> set[int]:
    """Rotation steps a cachemir linear of this shape uses (cachemir_rot_indices.cu)."""
    p = cm_params(N, d_in, d_out)
    steps = set()
    step = 1
    while step < p.tp_in:
        steps.add(step * (p.t - 1)); step *= 2
    for b in range(1, p.bstep_c):
        steps.add(b * p.t * p.t)
    for g in range(1, p.gstep_c):
        steps.add(g * p.bstep_c * p.t * p.t)
    if p.r_o > 1:
        steps.add(p.t * p.tp)
    step = 1
    while step < p.tp_out:
        steps.add(step); step *= 2
    return {s % N for s in steps if s % N}


def encode_linear_input(x, N: int, d_in: int, d_out: int) -> np.ndarray:
    """The slot vector encode_linear_input (cachemir_linear_utils.cu) encrypts."""
    p = cm_params(N, d_in, d_out)
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros(N)
    if p.is_up:
        out[np.arange(p.d) * p.t] = x[:p.d]
    else:
        d_x = p.alpha * p.d
        M = N // p.tp
        m = np.arange(M)
        out[m * p.tp] = x[interleave_idx(m, p.d, d_x)]
    return out


def encode_weight_matrix(W, N: int, d_in: int, d_out: int) -> np.ndarray:
    """encode_weight_matrix (cachemir_linear_utils.cu) without the CKKS encode:
    returns the (n_pt, N) plaintext slot vectors, index [j*r_o + k]."""
    W = np.asarray(W)
    W = np.asarray(W, dtype=np.complex128 if np.iscomplexobj(W) else np.float64)
    if W.shape != (d_in, d_out):
        raise ValueError(f"weight must be ({d_in}, {d_out}), got {W.shape}")
    p = cm_params(N, d_in, d_out)
    M_out = N // p.tp_out
    cascade_shift = (p.t * p.tp) // p.tp_out
    i = np.arange(N)
    pt = np.zeros((p.n_pt, N), dtype=W.dtype)   # complex W (W_re + i W_im): complex plaintexts
    for j in range(p.r_i):
        g = j // p.bstep_c
        s_g = g * p.bstep_c * p.t * p.t
        ip = (i - s_g) % N
        row = ((ip // p.t + j * p.t + ip % p.tp_in) % p.d) + ((ip % p.t) // p.tp_in) * p.d
        for k in range(p.r_o):
            ms = (ip // p.tp_out - k * cascade_shift) % M_out
            col = interleave_idx(ms, p.d, d_out)
            pt[j * p.r_o + k] = W[row, col]
    return pt


def encode_weight_matrix_outputpack(W, N: int, d_in: int, d_out: int) -> np.ndarray:
    """encode_weight_matrix_outputpack (cachemir_linear_utils.cu): output blocks 2k' and
    2k'+1 share one complex plaintext (real and imaginary lane), so the linear runs half the
    plaintext products; apply_linear_outputpack unpacks them (r_o must be even)."""
    W = np.asarray(W, dtype=np.float64)
    if W.shape != (d_in, d_out):
        raise ValueError(f"weight must be ({d_in}, {d_out}), got {W.shape}")
    p = cm_params(N, d_in, d_out)
    if p.r_o % 2:
        raise ValueError(f"output pack needs an even r_o, got {p.r_o}")
    rop = p.r_o // 2
    M_out = N // p.tp_out
    cascade_shift = (p.t * p.tp) // p.tp_out
    i = np.arange(N)
    pt = np.zeros((p.r_i * rop, N), dtype=np.complex128)
    for j in range(p.r_i):
        g = j // p.bstep_c
        s_g = g * p.bstep_c * p.t * p.t
        ip = (i - s_g) % N
        row = ((ip // p.t + j * p.t + ip % p.tp_in) % p.d) + ((ip % p.t) // p.tp_in) * p.d
        for kp in range(rop):
            cols = [interleave_idx((ip // p.tp_out - k * cascade_shift) % M_out, p.d, d_out)
                    for k in (2 * kp, 2 * kp + 1)]
            pt[j * rop + kp] = W[row, cols[0]] + 1j * W[row, cols[1]]
    return pt


def encode_bias_vector(b, N: int, d_in: int, d_out: int, fill: bool = True) -> np.ndarray:
    """encode_bias_vector (cachemir_linear_utils.cu)."""
    b = np.asarray(b).ravel()
    b = np.asarray(b, dtype=np.complex128 if np.iscomplexobj(b) else np.float64)
    p = cm_params(N, d_in, d_out)
    out = np.zeros(N, dtype=b.dtype)              # complex b (b_re + i b_im): a complex bias
    if p.is_up and p.alpha > 1:
        M = N // p.tp
        for m in range(M):
            idx = interleave_idx(m, p.d, d_out)
            if idx < b.shape[0]:
                out[m * p.tp: m * p.tp + (p.tp if fill else 1)] = b[idx]
    else:
        for i in range(b.shape[0]):
            out[i * p.t: i * p.t + (p.t if fill else 1)] = b[i]
    return out


def decode_linear_output(cy, N: int, d_in: int, d_out: int) -> np.ndarray:
    """decode_linear_output (cachemir_linear_utils.cu)."""
    cy = np.asarray(cy, dtype=np.float64)
    p = cm_params(N, d_in, d_out)
    y = np.zeros(d_out)
    if p.is_up and p.alpha > 1:
        M = N // p.tp
        m = np.arange(M)
        idx = interleave_idx(m, p.d, d_out)
        ok = idx < d_out
        y[idx[ok]] = cy[m[ok] * p.tp]
    else:
        y[:] = cy[np.arange(d_out) * p.t]
    return y


def decode_tokens(cy, N: int, d_pad: int, d_real: int, T: int) -> np.ndarray:
    """decode_tokens (cachemir_linear_utils.cu): y[tok][i] = cy[i*t + tok]."""
    t = cm_params(N, d_pad, d_pad).t
    cy = np.asarray(cy, dtype=np.float64)
    i = np.arange(d_real)
    return np.stack([cy[i * t + tok] for tok in range(T)])


# ── per-feature (token-basis) vectors and masks ───────────────────────────────────────

def lane_vec(v, N: int, t: int) -> np.ndarray:
    """v[i] at slot i*t (pack_per_feature_vec, cachemir_norm_utils.cu; also
    encode_stride_values_at, inference.h)."""
    v = np.asarray(v, dtype=np.float64).ravel()
    out = np.zeros(N)
    out[np.arange(v.shape[0]) * t] = v
    return out


def stride_mask(N: int, d: int, stride: int, scale: float = 1.0, fill: float = 0.0,
                offset: int = 0) -> np.ndarray:
    """stride_mask_vec (inference.h) on the cachemir stride slots i*stride+offset."""
    out = np.full(N, float(fill))
    out[np.arange(d) * stride + offset] = scale
    return out


def active_token_mask(N: int, d: int, t: int, scale: float) -> np.ndarray:
    """active_token_mask_vec (inference.h): on cachemir every lane is active."""
    return stride_mask(N, d, t, scale)


def active_expanded_mask(dims: Dims, scale: float, fill: float = 0.0) -> np.ndarray:
    """active_expanded_mask_vec (inference.h + cachemir_masks.h): slot m*tp_E
    for every m whose interleaved feature lies below the real MLP width."""
    out = np.full(dims.N, float(fill))
    m = np.arange(dims.E)
    ok = interleave_idx(m, dims.hid, dims.E) < dims.E_real
    out[m[ok] * dims.tp_E] = scale
    return out


def real_head_tok0_mask(dims: Dims, tok_offset: int = 0) -> np.ndarray:
    """cachemir_attention_utils.h."""
    out = np.zeros(dims.N)
    for lane in range(dims.d_head_real):
        for h in range(dims.H_real):
            out[lane * dims.tH + h * dims.t + tok_offset] = 1.0
    return out


def real_head_half_mask(dims: Dims) -> np.ndarray:
    """cachemir_attention_utils.h."""
    return 0.5 * real_head_tok0_mask(dims)


def _tok_slot(dims: Dims, h: int, tok: int) -> int:
    return tok // dims.t * dims.t * dims.H + h * dims.t + tok % dims.t


def _tile_tH(dims: Dims, out: np.ndarray, periodic: bool) -> np.ndarray:
    """`periodic`: block 0 repeated over every tH block (the scores of a single group, kc <= t, kept tH-periodic)."""
    return np.tile(out[:dims.tH], dims.N // dims.tH) if periodic else out


def score_mask(dims: Dims, clip_lo: float, mean: float, kc: int, periodic: bool = False) -> np.ndarray:
    """score_mask_vec (cachemir_attention_utils.h)."""
    out = np.full(dims.N, clip_lo - mean)
    for h in range(dims.H):
        for tok in range(kc):
            out[_tok_slot(dims, h, tok)] = -mean
    return _tile_tH(dims, out, periodic)


def active_mask(dims: Dims, kc: int, periodic: bool = False) -> np.ndarray:
    """active_mask_vec (cachemir_attention_utils.h): 0.5/kc on the active slots."""
    out = np.zeros(dims.N)
    for h in range(dims.H):
        for tok in range(kc):
            out[_tok_slot(dims, h, tok)] = 0.5 / kc
    return _tile_tH(dims, out, periodic)


def qkt_group_mask(dims: Dims, num_tok: int, g: int, periodic: bool = False) -> np.ndarray:
    """qkt_group_mask_vec (cachemir_attention_utils.h); `periodic` (group 0 only) keeps every tH block."""
    gscale = 0.5 / math.sqrt(dims.d_head_real)
    i = np.arange(dims.N)
    h = (i % dims.tH) // dims.t
    ok = ((i // dims.tH == g) | periodic) & (i % dims.t < num_tok) & (h < dims.H_real)
    return np.where(ok, gscale, 0.0)


def hrs_pos0(dims: Dims) -> np.ndarray:
    """hrs_pos0_vec (cachemir_attention_utils.h)."""
    out = np.zeros(dims.N)
    out[::dims.t] = 1.0
    return out


def vlane_mask(dims: Dims, i: int, right_rot: int, scale: float = 0.5) -> np.ndarray:
    """vlane_mask_vec (cachemir_attention_utils.h)."""
    out = np.zeros(dims.N)
    for h in range(dims.H_real):
        out[i * dims.tH + h * dims.t + right_rot] = scale
    return out


def complex_vlane_mask(dims: Dims, i: int, right_rot: int, imag_scale: float = -0.5) -> np.ndarray:
    """complex_vlane_mask_vec (cachemir_attention_utils.h): the V-lane selector on
    the imaginary axis, for a V that came out of the K + iV pair bootstrap as 2i V."""
    return 1j * imag_scale * vlane_mask(dims, i, right_rot, 0.5)


def qkt_complex_odd_mask(dims: Dims, num_tok: int, g: int) -> np.ndarray:
    """qkt_complex_odd_mask_vec: the group mask on the negative imaginary axis, for the odd
    group of a complex K bucket (its scores come out as 2i q.K_odd)."""
    return -1j * qkt_group_mask(dims, num_tok, g)


def vpair_mask_complex(dims: Dims, i_re: int, i_im: int, right_rot: int, scale: float = 0.5) -> np.ndarray:
    """vpair_mask_complex_vec: one complex selector extracting source lane i_re onto the real
    axis and i_im onto the imaginary axis of a V pair bucket (disjoint slots)."""
    return vlane_mask(dims, i_re, right_rot, scale) + 1j * vlane_mask(dims, i_im, right_rot, scale)


def attention_rot_steps(dims: Dims) -> set[int]:
    """Every rotation the attention/norm ports issue (collect_mha_rots / collect_norm_rots)."""
    steps = set()
    step = 1
    while step < dims.t:
        steps.add(step); steps.add(-step); step *= 2
    s = dims.tH
    while s < dims.N:
        steps.add(s); s *= 2
    for i in range(1, dims.d_head_real):
        steps.add(i * dims.tH)
    for k in range(1, dims.t):
        steps.add(-k)
    gap = 1
    while gap < dims.N:
        steps.add(gap); gap *= 2
    return {s % dims.N for s in steps if s % dims.N}


# ── weight rearrangement (head-interleaved columns) ──────────────────────────────────

def rearrange_qkv_weights(W, H: int) -> np.ndarray:
    """out[:, r] = W[:, (r % H) * d_head + r // H] (cachemir_attention_utils.cu)."""
    W = np.asarray(W, dtype=np.float64)
    d_out = W.shape[1]
    d_head = d_out // H
    r = np.arange(d_out)
    return W[:, (r % H) * d_head + r // H]


def rearrange_qkv_biases(b, H: int) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64).ravel()
    d_head = b.shape[0] // H
    r = np.arange(b.shape[0])
    return b[(r % H) * d_head + r // H]


def rearrange_wo_weights(W, H: int) -> np.ndarray:
    """out[r, :] = W[(r % H) * d_head + r // H, :] (cachemir_attention_utils.cu)."""
    W = np.asarray(W, dtype=np.float64)
    d_in = W.shape[0]
    d_head = d_in // H
    r = np.arange(d_in)
    return W[(r % H) * d_head + r // H, :]


# ── CutMax tile masks ────────────────────────────────────────────────────────────────

def cutmax_tile_col_of_slot(m, d: int, W_tile: int):
    """include/cutmax.h (the interleave of a (d, W_tile) linear output)."""
    a = W_tile // d
    return (m // a + (m % a) * d) % W_tile


def cutmax_tile_mask(N: int, k: int, vocab: int, W_tile: int, d: int) -> np.ndarray:
    """cutmax.cu: 1 on the slots that carry a real vocabulary column of tile k."""
    wreal = min(W_tile, vocab - k * W_tile)
    m = np.arange(W_tile)
    out = np.zeros(N)
    out[:W_tile] = (cutmax_tile_col_of_slot(m, d, W_tile) < wreal).astype(np.float64)
    return out
