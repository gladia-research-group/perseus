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


@functools.lru_cache(maxsize=8)
def weight_index(N: int, d_in: int, d_out: int, outputpack: bool = False):
    """The gather behind encode_weight_matrix (and _outputpack): the flat index row * d_out +
    col of W each plaintext slot reads, as an (n_pt, N) int32 array (a pair of them, the real
    and imaginary lanes, for `outputpack`). It depends on the shape only, so a model whose layers
    share shapes builds each once and encodes a matrix as one take (a Llama-sized layer is ~4.6k
    plaintexts; the per-plaintext index arithmetic dominated its encode)."""
    p = cm_params(N, d_in, d_out)
    M_out = N // p.tp_out
    cascade_shift = (p.t * p.tp) // p.tp_out
    i = np.arange(N)
    ks = [(2 * k, 2 * k + 1) for k in range(p.r_o // 2)] if outputpack else [(k,) for k in range(p.r_o)]
    if outputpack and p.r_o % 2:
        raise ValueError(f"output pack needs an even r_o, got {p.r_o}")
    out = [np.empty((p.r_i * len(ks), N), dtype=np.int32) for _ in ks[0]]
    for j in range(p.r_i):
        g = j // p.bstep_c
        ip = (i - g * p.bstep_c * p.t * p.t) % N
        row = ((ip // p.t + j * p.t + ip % p.tp_in) % p.d) + ((ip % p.t) // p.tp_in) * p.d
        for kk, group in enumerate(ks):
            for lane, k in enumerate(group):
                col = interleave_idx((ip // p.tp_out - k * cascade_shift) % M_out, p.d, d_out)
                out[lane][j * len(ks) + kk] = row * d_out + col
    for a in out:
        a.setflags(write=False)
    return tuple(out) if outputpack else out[0]


def encode_weight_matrix_fast(W, N: int, d_in: int, d_out: int, outputpack: bool = False) -> np.ndarray:
    """encode_weight_matrix / encode_weight_matrix_outputpack through weight_index: the same
    values (a gather), without the per-plaintext index arithmetic."""
    W = np.asarray(W)
    if W.shape != (d_in, d_out):
        raise ValueError(f"weight must be ({d_in}, {d_out}), got {W.shape}")
    flat = W.reshape(-1)
    if outputpack:
        if np.iscomplexobj(W):
            raise ValueError("output pack needs a real weight")
        re, im = weight_index(N, d_in, d_out, True)
        out = np.empty(re.shape, dtype=np.complex128)
        out.real = flat.take(re)
        out.imag = flat.take(im)
        return out
    idx = weight_index(N, d_in, d_out)
    return flat.astype(np.complex128 if np.iscomplexobj(W) else np.float64, copy=False).take(idx)


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


def lanes_mask(N: int, d: int, stride: int, vals) -> np.ndarray:
    """Prefill: vals[l] on the slots i*stride + l (i < d) of every token lane l < len(vals)."""
    out = np.zeros(N, dtype=np.complex128 if np.iscomplexobj(np.asarray(vals)) else np.float64)
    idx = np.arange(d) * stride
    for lane, v in enumerate(vals):
        out[idx + lane] = v
    return out


def lanes_periodic(N: int, stride: int, vals, fill: float = 0.0) -> np.ndarray:
    """Prefill: vals[s % stride] on every slot s (lanes past len(vals) at `fill`)."""
    per = np.full(stride, float(fill))
    per[:len(vals)] = vals
    return np.tile(per, N // stride)


def active_expanded_mask(dims: Dims, scale: float, fill: float = 0.0) -> np.ndarray:
    """active_expanded_mask_vec (inference.h + cachemir_masks.h): slot m*tp_E
    for every m whose interleaved feature lies below the real MLP width."""
    out = np.full(dims.N, float(fill))
    m = np.arange(dims.E)
    ok = interleave_idx(m, dims.hid, dims.E) < dims.E_real
    out[m[ok] * dims.tp_E] = scale
    return out


def active_expanded_lanes(dims: Dims, scale: float, n_lanes: int) -> np.ndarray:
    """active_expanded_mask on token lanes 0..n_lanes-1 of the stride tp_E (prefill)."""
    base = active_expanded_mask(dims, scale)
    return sum(np.roll(base, lane) for lane in range(n_lanes))


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


def _tile_tH(dims: Dims, out: np.ndarray, periodic) -> np.ndarray:
    """`periodic` = Gp (True = 1): the first Gp tH blocks repeated over the vector, i.e. group g's scores in every block
    b with b % Gp == g (Gp = 1: the scores of a single group, kc <= t, in every block)."""
    gp = int(periodic)
    return np.tile(out[:gp * dims.tH], dims.N // (gp * dims.tH)) if gp else out


def score_mask(dims: Dims, clip_lo: float, mean: float, kc: int, periodic=False) -> np.ndarray:
    """score_mask_vec (cachemir_attention_utils.h)."""
    out = np.full(dims.N, clip_lo - mean)
    for h in range(dims.H):
        for tok in range(kc):
            out[_tok_slot(dims, h, tok)] = -mean
    return _tile_tH(dims, out, periodic)


def active_mask(dims: Dims, kc: int, periodic=False) -> np.ndarray:
    """active_mask_vec (cachemir_attention_utils.h): 0.5/kc on the active slots."""
    out = np.zeros(dims.N)
    for h in range(dims.H):
        for tok in range(kc):
            out[_tok_slot(dims, h, tok)] = 0.5 / kc
    return _tile_tH(dims, out, periodic)


def qkt_group_mask(dims: Dims, num_tok: int, g: int, periodic=False) -> np.ndarray:
    """qkt_group_mask_vec (cachemir_attention_utils.h); `periodic` = Gp (True = 1) keeps every tH block b with
    b % Gp == g instead of block g only."""
    gscale = 0.5 / math.sqrt(dims.d_head_real)
    i = np.arange(dims.N)
    h = (i % dims.tH) // dims.t
    gp = int(periodic)
    blk = (i // dims.tH) % gp == g if gp else i // dims.tH == g
    ok = blk & (i % dims.t < num_tok) & (h < dims.H_real)
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


def qkt_complex_odd_mask(dims: Dims, num_tok: int, g: int, periodic=False) -> np.ndarray:
    """qkt_complex_odd_mask_vec: the group mask on the negative imaginary axis, for the odd
    group of a complex K bucket (its scores come out as 2i q.K_odd)."""
    return -1j * qkt_group_mask(dims, num_tok, g, periodic)


def vpair_mask_complex(dims: Dims, i_re: int, i_im: int, right_rot: int, scale: float = 0.5) -> np.ndarray:
    """vpair_mask_complex_vec: one complex selector extracting source lane i_re onto the real
    axis and i_im onto the imaginary axis of a V pair bucket (disjoint slots)."""
    return vlane_mask(dims, i_re, right_rot, scale) + 1j * vlane_mask(dims, i_im, right_rot, scale)


def ladder_chunks(start: int, stop: int, bits: int) -> list[list[int]]:
    """The radix grouping of the rotate-and-sum ladder x += rot(x, g), g = start, 2 start, ... (|g| < stop) under
    LADDER_RADIX_BITS = bits: consecutive rungs in groups of `bits` from the bottom, each group done as
    x + sum_j rot(x, j h) for j = 1 .. 2^len - 1 (h its first gap) from one hoisted decomposition. Returns the
    step lists; a 1-rung group is the plain rung. src/bindings/ops.cu rotate_and_sum mirrors this rule."""
    sign, gap = (1 if start > 0 else -1), abs(int(start))
    gaps = []
    while gap < stop:
        gaps.append(gap)
        gap *= 2
    out = []
    for i in range(0, len(gaps), max(1, bits)):
        n = min(max(1, bits), len(gaps) - i)
        out.append([sign * j * gaps[i] for j in range(1, 1 << n)])
    return out


def gpt2_ladders(dims: Dims) -> list[tuple[int, int]]:
    """The (start, stop) of every rotate-and-sum ladder the GPT-2 port issues (head sums, LN mean and
    variance, the all-slot sums, the linear input/output ladders; census of a fake decode)."""
    t, tH, N = dims.t, dims.tH, dims.N
    return [(-1, t), (1, t), (1, 8), (1, N), (t - 1, 8 * (t - 1)), (t - 1, t * (t - 1)), (t, N), (tH, N)]


def ladder_radix_steps(dims: Dims, bits: int) -> set[int]:
    """The extra rotation keys the radix ladders need (every step of every multi-rung group), mod N."""
    steps = set()
    if bits < 2:
        return steps
    for start, stop in gpt2_ladders(dims):
        for g in ladder_chunks(start, stop, bits):
            if len(g) > 1:
                steps.update(s % dims.N for s in g)
    return {s for s in steps if s}


def ladder_radix_bits() -> int:
    """LADDER_RADIX_BITS (default 3; 0/1 = the plain ladder)."""
    import os
    try:
        return int(os.environ.get("LADDER_RADIX_BITS", "3") or 0)
    except ValueError:
        return 0


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


_COEFF_TABLES = {}


def _coeff_tables(slots: int):
    """The FFT positions of slot j (zeta^{5^j}) and of its conjugate, and the twist zeta^{-k}, for ring 2 * slots."""
    if slots not in _COEFF_TABLES:
        n = 2 * slots
        t = np.empty(slots, dtype=np.int64)
        t[0] = 1
        for j in range(1, slots):
            t[j] = t[j - 1] * 5 % (2 * n)
        _COEFF_TABLES[slots] = ((t - 1) // 2, (2 * n - t - 1) // 2, np.exp(-1j * np.pi * np.arange(n) / n))
    return _COEFF_TABLES[slots]


def coeff_rows(values, out=None, block=64):
    """The CKKS plaintext polynomials of slot-value rows (n, slots), divided by the scaling factor: (n, 2 slots)
    float32 (or written into `out`). The device encoder's map (FIDESlib DeviceEncode: the values at the FFT
    positions of zeta^{5^j}, their conjugates at those of zeta^{-5^j}, a forward FFT, the twist zeta^{-k}, 1/N),
    so round(row * scale) is the plaintext encode would produce up to the float32 rounding of the row."""
    values = np.asarray(values)
    n, slots = values.shape
    up, un, tw = _coeff_tables(slots)
    N = 2 * slots
    if out is None:
        out = np.empty((n, N), dtype=np.float32)
    for r0 in range(0, n, block):
        v = np.asarray(values[r0:r0 + block], dtype=np.complex128)
        w = np.zeros((len(v), N), dtype=np.complex128)
        w[:, up] = v
        w[:, un] = np.conj(v)
        out[r0:r0 + len(v)] = (np.fft.fft(w, axis=1) * tw).real / N
    return out
