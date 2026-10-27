"""Prefill: a T-token prompt through the decode kernels, 2t tokens per complex ciphertext.

Chunk k fills lanes l < t (token k t + l, feature i at slot i t + l); ciphertext c carries chunk 2c
(real) and chunk 2c+1 (imaginary).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from . import smcheb
from .attention import _recip_rounds, first_iters
from .layout import lanes_mask, lanes_periodic, stride_mask
from .layout import cm_params
from .linear import _baby_sum, linear_multi, prepare_linear_input
from .poly import eval_chebyshev, im_cleanse, rotsum
import os
_DBG = bool(os.environ.get("PF_DEBUG"))
# PF_ATTN_PAIR=1: two offsets per q.K^T / P.V product on the complex axis
_ATTN_PAIR = os.environ.get("PF_ATTN_PAIR", "1") not in ("", "0")
# PF_SM_JOINT=1: both chunks of a ciphertext share one refresh point after the softmax exp's Chebyshev
_SM_JOINT = os.environ.get("PF_SM_JOINT", "0") not in ("", "0")
# PF_LANE_HOIST=B: hoisted K / V lane rotations, B offsets per call (all at once exhausts the pool at t = 64)
_LANE_HOIST = int(os.environ.get("PF_LANE_HOIST", "0") or 0)
# PF_ATTN_PACK=l: paired q.K^T / P.V reductions packed 2^l products per ciphertext over windows of d_head / 2^l blocks
_PACK = int(os.environ.get("PF_ATTN_PACK", "0") or 0)
# PF_ATTN_BSGS=1 (with PF_ATTN_PACK): lane offsets b + 2^(l+1) g as baby rotations of K / V and giant ones of Q and of
# the P.V sums, the giant lane shift riding the packed placing / unpacking rotations (_bsgs_block_offset); bidirectional
# prompts rotate every chunk cyclically
_BSGS = os.environ.get("PF_ATTN_BSGS", "0") not in ("", "0")
# PF_SM_HPACK=1: a head group whose real heads fill at most half of it runs one softmax per query chunk pair
# (2k, 2k + 1), chunk 2k + 1's scores moved onto the group's empty half of chunk 2k's (one chain instead of two; causal:
# two chunks)
_SM_HPACK = os.environ.get("PF_SM_HPACK", "0") not in ("", "0")
# PF_KV_SHARE=1: GQA query sets walk each K / V pair together (one complex K / V combination per pair, not per set)
_KV_SHARE = os.environ.get("PF_KV_SHARE", "0") not in ("", "0")
# PF_PRE_PERIODIC=1: a prefix score alone in its score ciphertext stays tH-periodic (sparse refreshes, no block select);
# the prefix key / value plaintexts are encoded once per level
_PRE_PERIODIC = os.environ.get("PF_PRE_PERIODIC", "0") not in ("", "0")


@dataclasses.dataclass(frozen=True)
class Chunks:
    """A prompt of T tokens in chunks of t lanes; ciphertext c carries chunks 2c (real) and 2c+1 (imaginary)."""
    T: int
    t: int
    base: int = 0            # absolute position of token 0
    bidir: bool = False      # every query sees every key (encoders), not only the earlier ones
    keys: int = 0            # bidir: the first `keys` tokens are attended (0: all T; the rest is padding)

    @property
    def n_keys(self):
        return self.keys or self.T

    @property
    def key_tag(self):
        """The geometry part of a mask key: (T, base), plus (keys,) on a bidirectional prompt."""
        return (self.T, self.base) + ((("bidir", self.n_keys),) if self.bidir else ())

    @property
    def n_chunks(self):
        return (self.T + self.t - 1) // self.t

    @property
    def n_ct(self):
        return (self.n_chunks + 1) // 2

    def idx(self, k):
        """Prompt indices of chunk k's filled lanes."""
        return list(range(k * self.t, min(self.T, (k + 1) * self.t)))

    def lanes(self, k):
        """Absolute positions of chunk k's filled lanes."""
        return [self.base + i for i in self.idx(k)]

    def has(self, k):
        return k < self.n_chunks

    @property
    def n_pairs(self):
        """Token pairs (lane l of ciphertext c = pair c t + l)."""
        return sum(len(self.lanes(2 * c)) for c in range(self.n_ct))


# ── complex payload: two real chunks per ciphertext ─────────────────────────────────────

def split(ops, X):
    """(2 Re X, 2 Im X): one conjugate, no level."""
    cj = ops.conjugate(X)
    return ops.add(X, cj), ops.mult_i(ops.sub(cj, X))


def join(ops, A, B):
    """A + i B (the monomial: no level); B None leaves A."""
    return A if B is None else ops.add(A, ops.mult_i(B))


def name_inputs(inf, Xs):
    """Name inputs after the first pf_in<c>: anon_<n> numbering differs between a capture and its planned run."""
    if hasattr(inf, "name_ct"):
        for c, X in enumerate(Xs[1:], 1):
            inf.name_ct(X, f"pf_in{c}")


def prompt_slots(rows, ch: Chunks, d):
    """encode_prompt's slot vectors (chunk pair c: real half 2c, imaginary half 2c + 1, stride t)."""
    rows = np.asarray(rows, dtype=np.float64)
    out = []
    idx = np.arange(rows.shape[1]) * d.t
    for c in range(ch.n_ct):
        v = np.zeros(d.N, dtype=np.complex128)
        for half in (0, 1):
            if ch.has(2 * c + half):
                for lane, p in enumerate(ch.idx(2 * c + half)):
                    v[idx + lane] += (1j if half else 1.0) * rows[p]
        out.append(v)
    return out


def encode_prompt(rt, rows, ch: Chunks):
    """The prompt rows as ch.n_ct complex ciphertexts at the bootstrap output level (encrypted zero + slot vector)."""
    ops, d = rt.ops, rt.dims
    out = []
    for v in prompt_slots(rows, ch, d):
        zero = ops.core.encode_token_input(ops.inf, np.zeros(d.dim))
        out.append(ops.add_pt(zero, v))
    return out


def decrypt_prompt(rt, Xs, ch: Chunks, d_real: int):
    """(T, d_real) rows of a hid-layout complex stream (stride t)."""
    ops, d = rt.ops, rt.dims
    rows = np.zeros((ch.T, d_real))
    idx = np.arange(d_real) * d.t
    for c, X in enumerate(Xs):
        v = ops.decrypt_slots_complex(X)
        for half, part in ((0, v.real), (1, v.imag)):
            for lane, p in enumerate(ch.idx(2 * c + half)):
                rows[p] = part[idx + lane]
    return rows


# ── dense <-> per-pair (the decode linears) ───────────────────────────────────────────

def _lane0(rt, stride, scale, site):
    N = rt.dims.N
    return (site, stride, scale), lambda: stride_mask(N, N // stride, stride, scale)


def unpack(rt, X, stride: int, n: int, scale: float = 1.0):
    """Lanes 0..n-1 of a dense ciphertext as n ciphertexts with the lane on lane 0 (stride mask times `scale`)."""
    ops = rt.ops
    rots = [X] + ops.rotate_many(X, list(range(1, n)))
    key, build = _lane0(rt, stride, scale, "pf.unpack")
    return [rt.mult_mask(r, key, build) for r in rots]


def pack(rt, ys, stride: int, scale: float = 1.0):
    """sum_l rotate(lane0(ys[l]), -l): the inverse of unpack (lane 0 of each, times `scale`)."""
    ops = rt.ops
    key, build = _lane0(rt, stride, scale, "pf.pack")
    acc = None
    for lane, y in enumerate(ys):
        m = rt.mult_mask(y, key, build)
        if lane:
            m = ops.rotate(m, -lane)
        if acc is None:
            acc = m
        else:
            ops.inplace_add(acc, m)
    return acc


def dense_linear(rt, Xs, ws, ch: Chunks, in_stride: int, out_stride: int, in_scale=1.0, out_scale=1.0,
                 headroom=3):
    """Decode linears `ws` on every token pair of dense Xs (layout in_stride); per weight, dense outputs (out_stride).
    Inputs are refreshed when fewer than `headroom` units are left."""
    ops = rt.ops
    P = ch.n_pairs
    pairs = []
    for X in Xs:
        if len(pairs) >= P:
            break
        ops.bootstrap_hint(X, ops.headroom(headroom), True)
        n = min(in_stride, P - len(pairs))
        pairs += unpack(rt, X, in_stride, n, in_scale)
    # a bias would land on the real chunk only: added to the dense outputs on both axes
    nobias = [dataclasses.replace(w, bias=None) if w.bias is not None else w for w in ws]
    if any(np.iscomplexobj(w.pts) for w in ws):
        ys = [pair_linears(rt, p, nobias) for p in pairs]
        # an unhalved fused weight's outputs carry a factor 2
        fac = [0.5 if _n_out(w) == 2 else 1.0 for w in ws for _ in range(_n_out(w))]
        ws = [w for w in ws for _ in range(_n_out(w))]
    else:
        ys = [linear_multi(rt, p, nobias) for p in pairs]
        fac = [1.0] * len(ws)
    outs = []
    for j, w in enumerate(ws):
        dense = []
        for g in range(0, P, out_stride):
            y = pack(rt, [y[j] for y in ys[g:g + out_stride]], out_stride, out_scale * fac[j])
            if w.bias is not None:
                y = _add_bias(rt, y, w, ch, g, out_stride, out_scale)
            dense.append(y)
        outs.append(dense)
    return outs


def _n_out(w):
    """Outputs per token pair: two for a complex weight that is not output-packed."""
    return 2 if np.iscomplexobj(w.pts) and not w.outputpack else 1


def _bsgs(rt, x_rot, w, ks, idx_of):
    """BSGS accumulators of output blocks `ks`; idx_of(g, k) gives the plaintext indices."""
    ops = rt.ops
    p = cm_params(rt.dims.N, w.d_in, w.d_out)
    giant_rot = p.bstep_c * p.t * p.t
    out = []
    for k in ks:
        acc = None
        for g in range(p.gstep_c):
            tmp = _baby_sum(ops, x_rot, w, idx_of(g, k))
            if g > 0:
                tmp = ops.rotate(tmp, g * giant_rot)
            if acc is None:
                acc = tmp
            else:
                ops.inplace_add(acc, tmp)
        out.append(acc)
    return out


def _tail(rt, w, cy):
    """apply_linear's cascade over the output blocks and its tp_out ladder."""
    ops = rt.ops
    p = cm_params(rt.dims.N, w.d_in, w.d_out)
    cascade_rot = p.t * p.tp
    for k in range(len(cy) - 1, 0, -1):
        ops.inplace_add(cy[k - 1], ops.rotate(cy[k], cascade_rot))
    y = cy[0]
    if p.tp_out > 1:
        y = ops.rotate_and_sum(y, 1, p.tp_out)
    return y


def pair_linears(rt, x, ws):
    """Decode linears on a token pair z = a + i b; a complex weight W0 + i W1 uses S = BSGS(z), S' = BSGS(conj z):
    S + conj S' = a W0 + i b W0, -i (S - conj S') = a W1 + i b W1."""
    ops = rt.ops
    w0 = ws[0]
    # conjugates before the realize: a conjugate's degree is its input's, as the planner predicts
    p0 = cm_params(rt.dims.N, w0.d_in, w0.d_out)
    xi = ops.copy(x) if p0.tp_in == 1 else ops.rotate_and_sum(x, p0.t - 1, p0.tp_in * (p0.t - 1))
    x_rot = [xi] + ops.rotate_many(xi, [b * p0.t * p0.t for b in range(1, p0.bstep_c)])
    x_cj = None
    if any(np.iscomplexobj(w.pts) for w in ws):
        x_cj = [ops.conjugate(r) for r in x_rot]
        for r in x_cj:
            ops.realize(r)
    for r in x_rot:
        ops.realize(r)
    outs = []
    for w in ws:
        p = cm_params(rt.dims.N, w.d_in, w.d_out)
        if not np.iscomplexobj(w.pts):
            outs.append(_tail(rt, w, _bsgs(rt, x_rot, w, range(p.r_o),
                                           lambda g, k: [(g * p.bstep_c + b) * p.r_o + k for b in range(p.bstep_c)])))
            continue
        if w.outputpack:
            rop = p.r_o // 2
            idx = lambda g, k: [(g * p.bstep_c + b) * rop + k for b in range(p.bstep_c)]
            Sa = _bsgs(rt, x_rot, w, range(rop), idx)
            Sb = _bsgs(rt, x_cj, w, range(rop), idx)
            cy = []
            for a, b in zip(Sa, Sb):
                cb = ops.conjugate(b)
                cy += [ops.add(a, cb), ops.mult_i(ops.sub(cb, a))]
            outs.append(_tail(rt, w, cy))
        else:
            idx = lambda g, k: [(g * p.bstep_c + b) * p.r_o + k for b in range(p.bstep_c)]
            Sa = _bsgs(rt, x_rot, w, range(p.r_o), idx)
            Sb = _bsgs(rt, x_cj, w, range(p.r_o), idx)
            c0, c1 = [], []
            for a, b in zip(Sa, Sb):
                cb = ops.conjugate(b)
                c0.append(ops.add(a, cb))
                c1.append(ops.mult_i(ops.sub(cb, a)))
            outs += [_tail(rt, w, c0), _tail(rt, w, c1)]
    return outs


def _add_bias(rt, y, w, ch: Chunks, g0: int, stride: int, scale: float):
    """Bias on the lanes of pairs g0.. of a dense output: scale (b + i b) where the real / imaginary token exists."""
    d = rt.dims
    P = ch.n_pairs
    fac = np.zeros(stride, dtype=np.complex128)
    for lane in range(stride):
        p = g0 + lane
        if p < P:
            c, l = divmod(p, ch.t)
            fac[lane] = scale * (1.0 + (1j if (2 * c + 1) * ch.t + l < ch.T else 0.0))
    key = ("pf.bias", w.name or id(w), g0, stride, scale, ch.T)
    return rt.add_mask(y, key, lambda: np.asarray(w.bias) * fac[np.arange(d.N) % stride])


# ── nonlinear sites on the two halves ───────────────────────────────────────────────

def shift_lanes(rt, X2, shift, ch: Chunks, c: int, d_real: int, scale: float = 1.0):
    """X2 + scale (shift + i shift) on every filled lane (the folded LayerNorm shift)."""
    d = rt.dims
    na, nb = len(ch.lanes(2 * c)), len(ch.lanes(2 * c + 1)) if ch.has(2 * c + 1) else 0
    sh = np.asarray(shift, dtype=np.float64).ravel()[:d_real]

    def build():
        out = np.zeros(d.N, dtype=np.complex128)
        idx = np.arange(sh.shape[0]) * d.t
        for lane in range(d.t):
            out[idx + lane] = scale * sh * ((1.0 if lane < na else 0.0) + (1j if lane < nb else 0.0))
        return out
    return rt.add_mask(X2, ("pf.shift", id(shift), na, nb, scale), build)


def norm_pair(rt, X, cfg, ch: Chunks, c: int, norm_fn):
    """2 (LN(A) + i LN(B)) of X = A + i B; each half is cleansed so its imaginary noise stays out of the other."""
    ops = rt.ops
    nA = im_cleanse(ops, norm_fn(rt, X, cfg, ch.lanes(2 * c)))
    if not ch.has(2 * c + 1):
        return nA
    nB = im_cleanse(ops, norm_fn(rt, ops.mult_i(X), cfg, ch.lanes(2 * c + 1)))
    return join(ops, nA, ops.negate(nB))


def gelu_pair(rt, H, cfg, gelu_fn):
    """2 (g(A) + i g(B)) of H = (A + i B) / 2."""
    ops, d = rt.ops, rt.dims
    A, B = split(ops, H)
    gA = im_cleanse(ops, gelu_fn(rt, A, cfg, lanes=d.tp_E))
    gB = im_cleanse(ops, gelu_fn(rt, B, cfg, lanes=d.tp_E))
    return join(ops, gA, gB)


# ── attention ──────────────────────────────────────────────────────────────────────────

def lane_rotate(rt, x, e: int, site: str):
    """Lane i <- lane (i - e) mod t in every feature group (one level)."""
    ops, d = rt.ops, rt.dims
    if e == 0:
        return x
    N, t = d.N, d.t
    hi = rt.mult_mask(ops.rotate(x, -e), (site + ".hi", e),
                      lambda: lanes_mask(N, N // t, t, [1.0 if i >= e else 0.0 for i in range(t)]))
    lo = rt.mult_mask(ops.rotate(x, t - e), (site + ".lo", e),
                      lambda: lanes_mask(N, N // t, t, [1.0 if i < e else 0.0 for i in range(t)]))
    return ops.add(hi, lo)


_EVICT = os.environ.get("PF_EVICT_MASKS", "1") not in ("", "0")
_EVICT_SITES = ("",)        # every mask: each block re-encodes at its own levels


_POOL_TRIM = os.environ.get("PF_POOL_TRIM", "1") not in ("", "0")
# PF_POOL_TRIM_BELOW_GB=x: trim the pools (a device-wide sync) only when free device memory is below x GB
_POOL_TRIM_BELOW_GB = float(os.environ.get("PF_POOL_TRIM_BELOW_GB", "1e9"))
# PF_POOL_TRIM_EVERY=k: the trim (a device-wide sync) on every k-th block end only
_POOL_TRIM_EVERY = int(os.environ.get("PF_POOL_TRIM_EVERY", "1"))
_trim_count = [0]
_EVICT_BELOW_GB = float(os.environ.get("PF_EVICT_BELOW_GB", "12"))     # evict/trim only below this
# PF_EVICT_UNUSED=1: evict only the masks the block did not use (all of them below PF_EVICT_ALL_BELOW_GB)
_EVICT_UNUSED = os.environ.get("PF_EVICT_UNUSED", "1") not in ("", "0")
_EVICT_ALL_BELOW_GB = float(os.environ.get("PF_EVICT_ALL_BELOW_GB", "4"))


def evict_block_masks(rt):
    """End of a prefill block, under memory pressure: evict mask plaintexts and trim the stream-ordered pools."""
    core = rt.ops.core
    if hasattr(core, "device_free_gb") and core.device_free_gb() >= _EVICT_BELOW_GB:
        rt._used = set()
        return 0          # re-encoding the masks costs a few %
    can = _EVICT and hasattr(rt.ops, "erase_pts")
    if _EVICT_UNUSED and can:
        n = rt.evict_unused()
        _trim_count[0] += 1
        if _POOL_TRIM and hasattr(core, "gpu_pool_trim") and core.device_free_gb() < _POOL_TRIM_BELOW_GB \
                and _trim_count[0] % _POOL_TRIM_EVERY == 0:
            core.gpu_pool_trim()
        if not hasattr(core, "device_free_gb") or core.device_free_gb() >= _EVICT_ALL_BELOW_GB:
            return n
    n = rt.evict_masks(_EVICT_SITES) if can else 0
    if _POOL_TRIM and hasattr(core, "gpu_pool_trim"):
        core.gpu_pool_trim()
    return n


def lane_rotate_all(rt, x, site, batch: int):
    """lane_rotate(x, e) for e = 1 .. t - 1, `batch` offsets per hoisted rotation call: {e: ct}."""
    ops, d = rt.ops, rt.dims
    N, t = d.N, d.t
    out = {}
    for e0 in range(1, t, batch):
        es = list(range(e0, min(t, e0 + batch)))
        rots = ops.rotate_many(x, [-e for e in es] + [t - e for e in es])
        for i, e in enumerate(es):
            hi = rt.mult_mask(rots[i], (site + ".hi", e),
                              lambda e=e: lanes_mask(N, N // t, t, [1.0 if j >= e else 0.0 for j in range(t)]))
            lo = rt.mult_mask(rots[len(es) + i], (site + ".lo", e),
                              lambda e=e: lanes_mask(N, N // t, t, [1.0 if j < e else 0.0 for j in range(t)]))
            out[e] = ops.add(hi, lo)
    return out


def _offsets(ch: Chunks, c: int):
    """(u, c', e) of query chunk c: every earlier chunk at all t offsets, its own chunk at e < its lane count.
    Bidirectional: every chunk c' at every cyclic lane offset e (key lane (lane - e) mod t), u = c' t + e."""
    if ch.bidir:
        return [(cp * ch.t + e, cp, e) for cp in range(ch.n_chunks) for e in range(ch.t)]
    out = []
    nq = len(ch.lanes(c))
    for cp in range(c, -1, -1):
        for e in range(nq if cp == c else ch.t):
            out.append(((c - cp) * ch.t + e, cp, e))
    return out


def _kcs(ch: Chunks, pos):
    """Keys each query lane attends: its position + 1 (causal), or every unpadded key (bidirectional)."""
    return [ch.n_keys] * len(pos) if ch.bidir else [p + 1 for p in pos]


def prefix_offset(ch: Chunks, c: int) -> int:
    """Score offset of the public prefix key for chunk c: first unused own-chunk offset if partial, else (c + 1) t."""
    nq = len(ch.lanes(c))
    return nq if nq < ch.t else c * ch.t + ch.t


def _bsgs_on(ch: Chunks) -> bool:
    return _BSGS and _ATTN_PAIR and _PACK > 0


def _bsgs_block_offset(blk, d_head: int):
    """PF_ATTN_BSGS: the key offset (in its score ciphertext) of score block blk. Window a (W = d_head / 2^l blocks)
    holds giant step g of the pair (2a, 2a + 1) at blocks a W + 2 g, a W + 2 g + 1: offsets 2a + 2^(l+1) g + (0, 1)."""
    W = d_head >> _PACK
    rem = blk % W
    return 2 * (blk // W) + rem % 2 + (2 << _PACK) * (rem // 2)


def _bsgs_block_of(off: int, d_head: int) -> int:
    """PF_ATTN_BSGS: the score block of key offset `off` (< d_head), the inverse of _bsgs_block_offset."""
    return int(np.flatnonzero(_bsgs_block_offset(np.arange(d_head), d_head) == off)[0])


def _layout_tag(ch: Chunks) -> tuple:
    """Mask-key part of the score block order."""
    return ("bsgs", _PACK) if _bsgs_on(ch) else ()


def _score_valid(d, ch: Chunks, c: int, m: int, prefix: bool = False):
    """(N,) bool: slots of score ciphertext m of query chunk c holding a real score (`prefix`: at prefix_offset)."""
    nq = len(ch.lanes(c))
    i = np.arange(d.N)
    blk, h, lane = i // d.tH, (i % d.tH) // d.t, i % d.t
    u = m * d.d_head + (_bsgs_block_offset(blk, d.d_head) if _bsgs_on(ch) else blk)
    if ch.bidir:
        key = (u // ch.t) * ch.t + (lane - u % ch.t) % ch.t
        return (h < d.H_real) & (lane < nq) & (u < ch.n_chunks * ch.t) & (key < ch.n_keys)
    same = u < ch.t
    reg = (u < c * ch.t + ch.t) & np.where(same, lane >= u, True)
    if prefix:
        bos = u == prefix_offset(ch, c)
        reg = (reg & ~bos) | bos
    return (h < d.H_real) & (lane < nq) & reg


def _hp(k: int):
    """PF_SM_HPACK's chunk name of query chunk pair (2k, 2k + 1) ("hp" for the first)."""
    return "hp" if k == 0 else ("hp", k)


def _hp_pair(c):
    """The first chunk of a head-packed chunk name, or None."""
    if c == "hp":
        return 0
    return 2 * c[1] if isinstance(c, tuple) and c[0] == "hp" else None


def _hp_valid(d, ch: Chunks, m: int, c0: int = 0):
    """PF_SM_HPACK: score ciphertext m of the head-packed softmax: chunk c0's on the first half of the heads, chunk
    c0 + 1's moved half a group up."""
    return _score_valid(d, ch, c0, m) | np.roll(_score_valid(d, ch, c0 + 1, m), d.tH // 2)


def _valid_of(d, ch: Chunks, c, m: int, prefix: bool = False):
    c0 = _hp_pair(c)
    return _hp_valid(d, ch, m, c0) if c0 is not None else _score_valid(d, ch, c, m, prefix)


def _kcs_of(ch: Chunks, c):
    """Keys per query lane of chunk c; a head-packed pair: chunk c0's lanes, then chunk c0 + 1's (t each)."""
    c0 = _hp_pair(c)
    if c0 is not None:
        k0 = _kcs(ch, ch.lanes(c0))
        return k0 + [1] * (ch.t - len(k0)) + _kcs(ch, ch.lanes(c0 + 1))
    return _kcs(ch, ch.lanes(c))


def _periodic_valid(d, ch: Chunks, c: int):
    """(N,) bool: the slots of a tH-periodic prefix score ciphertext holding a real score."""
    i = np.arange(d.N)
    return ((i % d.tH) // d.t < d.H_real) & (i % d.t < len(ch.lanes(c)))


def _vkey(v):
    import hashlib
    return hashlib.blake2b(np.ascontiguousarray(v).tobytes(), digest_size=8).hexdigest()


def rope_lanes(rt, x, rope, positions, site):
    """RoPE of every lane at its own position: x cos + rotate(x, N/2) sin' (one level)."""
    ops, d = rt.ops, rt.dims
    pos = tuple(int(p) for p in positions)

    def vec(k):
        def build():
            out = np.zeros((d.d_head, d.H, d.t))
            for lane, p in enumerate(pos):
                out[:, :, lane] = rope.lane_cs(d, p)[k][:, None]
            return out.reshape(-1)
        return build
    a = rt.mult_mask(x, (site + ".c", pos), vec(0))
    b = rt.mult_mask(ops.rotate(x, d.N // 2), (site + ".s", pos), vec(1))
    return ops.add(a, b)


def _lane_vec(d, vals, fill=0.0):
    """A per-query-lane value on every slot (t-periodic); 2t values: the second t on the second half of the heads
    (PF_SM_HPACK)."""
    if len(vals) <= d.t:
        return lanes_periodic(d.N, d.t, vals, fill)
    lo, hi = lanes_periodic(d.N, d.t, vals[:d.t], fill), lanes_periodic(d.N, d.t, vals[d.t:], fill)
    return np.where(_upper_heads(d), hi, lo)


def _upper_heads(d):
    """(N,) bool: the slots of the second half of a group's heads."""
    return (np.arange(d.N) % d.tH) // d.t >= d.H // 2


class _KcExtrapolated:
    """Softmax config whose kc_r scales by C / kc past the calibrated counts instead of holding the last r."""

    def __init__(self, cfg):
        self._c = cfg

    def __getattr__(self, name):
        return getattr(self._c, name)

    def kc_r(self, i, kc):
        r = self._c.kc_r(i, kc)
        if self._c.sm_kc_r and self._c.log2delta2 > 0:
            C = len(self._c.sm_kc_r) // self._c.log2delta2
            if kc > C and r < 1.0:
                r *= C / float(kc)
        return r


def recip_lanes(rt, D, a_vals, b_vals, iters: int, key, cheb=None):
    """1 / D by Goldschmidt with per-lane seed a - b D; D and the reciprocal share one complex ciphertext.
    `cheb` = (per-step kappa lanes, mu lanes): SM_CHEB_GS's scaled steps (poly.goldschmidt_recip)."""
    ops, d = rt.ops, rt.dims
    av = np.asarray(a_vals, dtype=np.float64)
    bv = np.asarray(b_vals, dtype=np.float64)
    hb = rt.mult_mask(D, ("pf.rcp.hb",) + key, lambda: _lane_vec(d, 0.5 * bv))
    G = rt.add_mask(hb, ("pf.rcp.ha",) + key, lambda: _lane_vec(d, -0.5 * av))
    if cheb is not None:
        kap, mu = cheb
        X = rt.add_mask(ops.mult_i(D), ("pf.rcp.mu",) + key, lambda: _lane_vec(d, mu))
        P = ops.negate(ops.mult_i(ops.mult(G, X)))
        for j in range(iters - 1):
            F = rt.add_mask(ops.add(P, ops.conjugate(P)), ("pf.rcp.kap", j) + key,
                            lambda j=j: _lane_vec(d, 2.0 * np.asarray(kap[j])))
            P = ops.mult(P, F)
        return ops.mult_i(ops.sub(ops.conjugate(P), P))
    nb = rt.mult_mask(D, ("pf.rcp.nb",) + key, lambda: _lane_vec(d, -0.5 * bv))
    im = rt.add_mask(nb, ("pf.rcp.pa",) + key, lambda: _lane_vec(d, 0.5 * av))
    P = ops.add(ops.mult(D, G), ops.mult_i(im))
    for _ in range(1, iters):
        P = ops.mult(P, ops.add(ops.add(P, ops.conjugate(P)), 2.0))
    return ops.mult_i(ops.sub(ops.conjugate(P), P))


def _sm_cheb(rt, S, cfg, ch: Chunks, c: int, prefix: bool, m0: int = 0, periodic=None):
    """Score masks and the exp's Chebyshev of chunk c's score ciphertexts m0, m0 + 1, ...: [(z, valid)]."""
    ops, d = rt.ops, rt.dims
    mean = (cfg.clip_hi + cfg.clip_lo) / 2.0
    sf = 2.0 ** (-cfg.log2delta1 - cfg.log2delta2)
    out = []
    for m, s in enumerate(S, m0):
        per = m == periodic
        valid = _periodic_valid(d, ch, c) if per else _valid_of(d, ch, c, m, prefix)
        s = rt.add_mask(s, ("pf.sm.score", c, m, cfg.clip_lo, mean) + ch.key_tag + (prefix,) + (("per",) if per else ())
                        + _layout_tag(ch),
                        lambda valid=valid: np.where(valid, -mean, cfg.clip_lo - mean))
        out.append((eval_chebyshev(ops, s, cfg.cheb_coeffs, cfg.cheb_a / sf, cfg.cheb_b / sf), valid))
    return out


def _sm_tail(rt, zs, cfg, ch: Chunks, c: int, prefix: bool, factor=None, m0: int = 0, periodic=None):
    """Squarings, cleanse and active mask (0.5 / kc per lane, times factor[j]) of chunk c's exps."""
    ops, d = rt.ops, rt.dims
    kcs = _kcs_of(ch, c)
    Z = []
    for j, (z, valid) in enumerate(zs):
        m = m0 + j
        for _ in range(cfg.log2delta1):
            z = ops.square(z)
        z = im_cleanse(ops, z)
        f = 1.0 if factor is None else factor[j]
        lane_scale = [f * 0.5 / kc for kc in kcs]
        key = ("pf.sm.active", c, m) + ch.key_tag + (prefix,) + ((f,) if f != 1.0 else ()) + \
            (("per",) if m == periodic else ()) + _layout_tag(ch)
        Z.append(rt.mult_mask(z, key,
                              lambda valid=valid, ls=lane_scale: np.where(valid, _lane_vec(d, ls), 0.0)))
    return Z


def _sm_joint(rt, za, zb, cfg):
    """A chunk pair's exp Chebyshevs joined into one refresh point and split back; returns the active masks' factors."""
    ops = rt.ops
    n = min(len(za), len(zb))
    fa, fb = [1.0] * len(za), [1.0] * len(zb)
    back = 2.0 ** (-(2 ** cfg.log2delta1))
    for m in range(n):
        X = ops.add(za[m][0], ops.mult_i(zb[m][0]))
        if _JOINT_FORCE:
            ops.bootstrap(X)            # binding under a plan
        else:
            ops.bootstrap_hint(X, ops.headroom(2), True)
        a, b = split(ops, X)
        za[m] = (a, za[m][1]); zb[m] = (b, zb[m][1])
        fa[m] = fb[m] = back
    return fa, fb


def prefill_softmax(rt, S, cfg, ch: Chunks, c: int, prefix: bool = False, Z=None, periodic=None):
    """softmax_thor over chunk c's packed scores S with per-lane (kc-dependent) reciprocal seeds.
    `Z`: precomputed masked exps (the joint path)."""
    ops, d = rt.ops, rt.dims
    c0 = _hp_pair(c)
    pos = ch.lanes(c if c0 is None else c0)
    kcs = _kcs_of(ch, c)
    if Z is None:     # cheb then tail per score ct: the captured op order
        Z = []
        for m in range(len(S)):
            (zc,) = _sm_cheb(rt, [S[m]], cfg, ch, c, prefix, m0=m, periodic=periodic)
            Z += _sm_tail(rt, [zc], cfg, ch, c, prefix, m0=m, periodic=periodic)
    lean = ops.sm_gs_first
    cheb = getattr(ops, "sm_cheb_gs", False)
    per_lane = [smcheb.rounds(cfg, kc) if cheb else _recip_rounds(_KcExtrapolated(cfg), kc, lean) for kc in kcs]
    Y = Z
    for k in range(len(per_lane[0])):
        if k:
            Z = [ops.square(im_cleanse(ops, y)) for y in Y]
        Zr = [z for m, z in enumerate(Z) if m != periodic]
        D = Zr[0] if len(Zr) == 1 else ops.add(Zr[0], Zr[1])
        for z in Zr[2:]:
            ops.inplace_add(D, z)
        D = rotsum(ops, D, d.tH, d.N)
        if periodic is not None:
            D = ops.add(D, Z[periodic])
        D = ops.tag_reduce(D, d.tH)
        rnd = [pl[k] for pl in per_lane]
        iters = rnd[0][2]
        a_vals = [r[0] * r[3] for r in rnd]
        b_vals = [r[1] * r[3] * r[3] for r in rnd]
        cv = None if not cheb else ([[r[4][0][j] for r in rnd] for j in range(iters - 1)], [r[4][1] for r in rnd])
        R = recip_lanes(rt, D, a_vals, b_vals, iters,
                        (tuple(round(v, 12) for v in a_vals), tuple(round(v, 15) for v in b_vals)), cv)
        if _DBG:
            dv = np.real(D.vec).reshape(-1, d.t)[:, :len(pos)]; rv = np.real(R.vec).reshape(-1, d.t)[:, :len(pos)]
            hv = (np.arange(d.N) % d.tH) // d.t < d.H_real
            dvv = np.real(D.vec)[hv].reshape(-1)
            if k == 0:
                per = np.real(D.vec).reshape(-1, d.tH)[0].reshape(d.H, d.t)[:d.H_real, :len(pos)]
                print("      D>8 at (head,pos):", [(int(h), int(pos[l])) for h, l in zip(*np.nonzero(per > 8))][:20])
            print(f"      round {k} c={c} D [{dvv.min():.3g},{dvv.max():.3g}] a {min(a_vals):.3g}..{max(a_vals):.3g} b {min(b_vals):.3g}..{max(b_vals):.3g} iters {iters} R*D [{np.nanmin(np.real(R.vec*D.vec)[hv]):.3g},{np.nanmax(np.real(R.vec*D.vec)[hv]):.3g}]")
        Y = [ops.mult(z, R) for z in Z]
    return Y


def prefill_attention(rt, Qd, Kd, Vd, cfg, ch: Chunks, kv=None, rope=None, prefix=None, shared=False):
    """Causal attention on dense complex Q/K/V (scale 1/2); returns dense outputs (scale 1) as the decode's P.V.
    `kv` receives the per-chunk real K/V; `prefix`: (k, v) slot vectors of one public key / value before the prompt.
    `shared`: Qd is a list of query sets on the same K / V (GQA), one output list each."""
    ops, d = rt.ops, rt.dims
    nc = ch.n_chunks

    def halves(src):
        dst = [None] * nc
        for c, X in enumerate(src):
            ops.bootstrap_hint(X, ops.headroom(2), True)
            A, B = split(ops, X)
            dst[2 * c] = A
            if ch.has(2 * c + 1):
                dst[2 * c + 1] = B
        return dst
    Qs = [halves(q) for q in (Qd if shared else [Qd])]
    qset = {id(Q): j for j, Q in enumerate(Qs)}      # a query set's key in the rotation caches
    K, V = halves(Kd), halves(Vd)
    if rope is not None:
        for k in range(nc):
            for Q in Qs:
                Q[k] = rope_lanes(rt, Q[k], rope, ch.lanes(k), "pf.rope.q")
            K[k] = rope_lanes(rt, K[k], rope, ch.lanes(k), "pf.rope.k")
    if kv is not None:
        kv["K"], kv["V"] = K, V
    gscale = 1.0 / math.sqrt(d.d_head_real)
    rotK, rotV = {}, {}

    def k_rot(cp, e, c):
        if cp == c:
            return None
        if (cp, e) not in rotK:
            if _LANE_HOIST and e:
                for f, y in lane_rotate_all(rt, K[cp], "pf.krot", _LANE_HOIST).items():
                    rotK[(cp, f)] = y
            else:
                rotK[(cp, e)] = lane_rotate(rt, K[cp], e, "pf.krot")
        return rotK[(cp, e)]

    def v_rot(cp, e, c):
        if cp == c:
            return None
        if (cp, e) not in rotV:
            if _LANE_HOIST and e:
                for f, y in lane_rotate_all(rt, V[cp], "pf.vrot", _LANE_HOIST).items():
                    rotV[(cp, f)] = y
            else:
                rotV[(cp, e)] = lane_rotate(rt, V[cp], e, "pf.vrot")
        return rotV[(cp, e)]

    owns, vowns = {}, {}

    nw = 1 << _PACK
    W = d.d_head // nw
    blk_of = np.arange(d.N) // d.tH
    bsgs = _bsgs_on(ch)
    Bb = 2 * nw                             # baby offsets b < Bb; a packed ciphertext holds one giant step
    if bsgs and (d.d_head % ch.t or ch.t % Bb):
        raise ValueError("PF_ATTN_BSGS: needs d_head a multiple of t lanes, t a multiple of the baby range")
    lane_of = np.arange(d.N) % d.t
    start = blk_of % W == 0
    rots, cycs = {}, {}

    def prime(tag, x, steps):
        """Plain rotations of x (one hoisted call for the new steps)."""
        new = sorted({s % d.N for s in steps} - {0} - {s for t_, s in rots if t_ == tag})
        for s, y in zip(new, ops.rotate_many(x, new) if new else []):
            rots[(tag, s)] = y

    def rot_of(tag, x, s):
        s %= d.N
        if s == 0:
            return x
        if (tag, s) not in rots:
            prime(tag, x, [s])
        return rots[(tag, s)]

    def cyc(tag, x, e, site):
        """lane_rotate(x, e) on the cached plain rotations of x."""
        e %= d.t
        if e == 0:
            return x
        if (tag, e) not in cycs:
            hi = rt.mult_mask(rot_of(tag, x, -e), (site + ".hi", e),
                              lambda: lanes_mask(d.N, d.N // d.t, d.t, [1.0 if j >= e else 0.0 for j in range(d.t)]))
            lo = rt.mult_mask(rot_of(tag, x, d.t - e), (site + ".lo", e),
                              lambda: lanes_mask(d.N, d.N // d.t, d.t, [1.0 if j < e else 0.0 for j in range(d.t)]))
            cycs[(tag, e)] = ops.add(hi, lo)
        return cycs[(tag, e)]

    def lane_parts(shift, L, cyclic):
        """(slot shift, lane select) of a lane shift by L riding a block shift: the lanes below t - L move L up in
        their own lane group; cyclic: the others move L - t (wrap)."""
        out = [(shift + L, lane_of < d.t - L)]
        if cyclic and L:
            out.append((shift + L - d.t, lane_of >= d.t - L))
        return out

    def baby(src, name, c, cp, b):
        """src[cp] lane-rotated by baby offset b: plain within chunk c, cyclic across chunks (and within a
        bidirectional chunk)."""
        if cp == c and not ch.bidir:
            return rot_of((name, cp), src[cp], -b)
        prime((name, cp), src[cp], [d.t - j for j in range(1, Bb)] + ([-j for j in range(1, Bb)] if ch.bidir else []))
        return cyc((name, cp), src[cp], b, f"pf.{name.lower()}rot")

    def win_mask(a, lo=0):
        """Blocks [aW - lo, aW - lo + W): window a (pack), lo = W - 1: what the forward broadcast fills from aW."""
        return ((blk_of - a * W + lo) % d.d_head < W).astype(float)

    def qkt_packed(c, QL, SL, groups, own, pre):
        """Pair products summed over the nw windows (leaf ladder), packed one per window, summed inside the windows,
        shifted to their score blocks with the real / imaginary parts split to u, u + 1 (per query set of QL)."""
        by = {}
        for grp in groups:
            u = grp[0][0]
            m, r = divmod(u, d.d_head)
            a, o = divmod(r, W)
            krs = [own[e] if cp == c else k_rot(cp, e, c) for _, cp, e in grp]
            kc = krs[0] if len(krs) == 1 else ops.add(krs[0], ops.mult_i(krs[1]))
            for i, Q in enumerate(QL):
                p = rotsum(ops, ops.mult(Q[c], kc), W * d.tH, d.N)
                v = rt.mult_mask(p, ("pf.qpk", a, W), lambda a=a: win_mask(a))
                if (i, m, o) in by:
                    ops.inplace_add(by[(i, m, o)], v)
                else:
                    by[(i, m, o)] = v
        for (i, m, o), w in sorted(by.items()):
            S = SL[i]
            w = rotsum(ops, w, d.tH, W * d.tH)

            def build(sh, coef, m=m):
                start = (blk_of % W == 0)
                valid = np.roll(_score_valid(d, ch, c, m, pre), -sh * d.tH)
                return np.where(start & valid, coef, 0.0)
            for sh, coef in ((o, 0.5 * gscale), (o + 1, -0.5j * gscale)):
                x = rt.mult_mask(w, ("pf.qpl", c, m, sh, W, complex(coef)) + ch.key_tag + (pre,),
                                 lambda sh=sh, coef=coef: build(sh, coef))
                if sh:
                    x = ops.rotate(x, d.N - sh * d.tH)
                if S[m] is None:
                    S[m] = x
                else:
                    ops.inplace_add(S[m], x)

    def qkt_bsgs(c, QL, SL, groups, pre):
        """qkt_packed with offset e = b + Bb g as K rotated by b and Q by Bb g (cyclic across chunks): the nw pairs of a
        giant step share a packed ciphertext, whose lane shift back by Bb g rides each window's placing rotation. With
        d_head > t a score ciphertext's giant steps span several key chunks: block step g, lane step g mod t / Bb."""
        by, lanes = {}, {}
        for grp in groups:
            u, cp, _ = grp[0]
            m, r = divmod(u, d.d_head)
            g, b = divmod(r, Bb)
            gl = (r % d.t) // Bb
            cy = cp != c or ch.bidir
            lanes[(m, g)] = (gl, cy)
            krs = [baby(K, "K", c, cp_, e % Bb) for _, cp_, e in grp]
            kc = krs[0] if len(krs) == 1 else ops.add(krs[0], ops.mult_i(krs[1]))
            for i, Q in enumerate(QL):
                if not cy:
                    qg = rot_of(("Q", qset[id(Q)], c), Q[c], Bb * gl)
                else:
                    qg = cyc(("Q", qset[id(Q)], c), Q[c], d.t - Bb * gl, "pf.qrot")
                p = rotsum(ops, ops.mult(qg, kc), W * d.tH, d.N)
                v = rt.mult_mask(p, ("pf.qpk", b // 2, W), lambda a=b // 2: win_mask(a))
                if (i, m, g) in by:
                    ops.inplace_add(by[(i, m, g)], v)
                else:
                    by[(i, m, g)] = v
        for (i, m, g), w in sorted(by.items()):
            S = SL[i]
            gl, cy = lanes[(m, g)]
            w = rotsum(ops, w, d.tH, W * d.tH)
            valid = _score_valid(d, ch, c, m, pre)
            for par, coef in ((0, 0.5 * gscale), (1, -0.5j * gscale)):
                for part, (sh, sel) in enumerate(lane_parts((2 * g + par) * d.tH, Bb * gl, cy)):
                    x = rt.mult_mask(w, ("pf.qpl.b", c, m, sh, part, Bb * gl, W, complex(coef)) + ch.key_tag
                                     + (("pre",) if pre else ()),
                                     lambda sh=sh, sel=sel, coef=coef, valid=valid:
                                     np.where(start & sel & np.roll(valid, -sh), coef, 0.0))
                    if sh % d.N:
                        x = ops.rotate(x, -sh % d.N)
                    if S[m] is None:
                        S[m] = x
                    else:
                        ops.inplace_add(S[m], x)

    def pv_bsgs(c, PL, groups):
        """pv_packed with offset e = b + Bb g: each giant step's probabilities unpacked shifted by Bb g lanes, times V
        rotated by b, summed per giant step and shifted back by Bb g (cyclic once chunk c attends an earlier chunk)."""
        by, lanes = {}, {}
        for grp in groups:
            u, cp = grp[0][0], grp[0][1]
            m, r = divmod(u, d.d_head)
            by.setdefault((m, r // Bb), []).append((r % Bb // 2, grp))
            lanes[(m, r // Bb)] = ((r % d.t) // Bb, cp != c or ch.bidir)
        sums = {}
        for (m, g), items in sorted(by.items()):
            gl, cy = lanes[(m, g)]
            ys = []
            for P in PL:
                y = None
                for par, coef in ((0, 0.5), (1, -0.5j)):
                    for part, (sh, sel) in enumerate(lane_parts((2 * g + par) * d.tH, Bb * gl, cy)):
                        x = P[m] if sh % d.N == 0 else ops.rotate(P[m], sh % d.N)
                        x = rt.mult_mask(x, ("pf.vpk.b", W, Bb * gl, part, complex(coef)),
                                         lambda sel=sel, coef=coef: np.where(start & sel, coef, 0.0))
                        y = x if y is None else ops.add(y, x)
                ys.append(rotsum(ops, y, d.tH, W * d.tH))
            for a, grp in items:
                pbs = [rotsum(ops, rt.mult_mask(y, ("pf.vup", a, W), lambda a=a: win_mask(a, W - 1)), W * d.tH, d.N)
                       for y in ys]
                vrs = [baby(V, "V", c, cp, e % Bb) for _, cp, e in grp]
                vc = vrs[0] if len(vrs) == 1 else ops.add(vrs[0], ops.mult_i(vrs[1]))
                for i, pb in enumerate(pbs):
                    prod = ops.mult(pb, vc)
                    if (i, gl) in sums:
                        ops.inplace_add(sums[(i, gl)], prod)
                    else:
                        sums[(i, gl)] = prod
        cyclic = c > 0 or ch.bidir
        accs = [None] * len(PL)
        for (i, g), x in sorted(sums.items()):
            for part, (sh, sel) in enumerate(lane_parts(0, Bb * g, cyclic) if g else [(0, None)]):
                y = x if sel is None or not cyclic else rt.mult_mask(
                    x, ("pf.vgb", Bb * g, part), lambda sel=sel: sel.astype(float))
                if sh % d.N:
                    y = ops.rotate(y, -sh % d.N)
                accs[i] = y if accs[i] is None else ops.add(accs[i], y)
        return accs

    def pv_packed(c, PL, groups, vown):
        """Inverse of qkt_packed: each pair's 0.5 p_u - 0.5i p_(u+1) moved to its window's first block, broadcast over
        the window, unpacked per window and broadcast over the nw windows (leaf ladder); per query set of PL
        sum_pairs pb * vc."""
        accs = [None] * len(PL)
        by = {}
        for grp in groups:
            u = grp[0][0]
            m, r = divmod(u, d.d_head)
            a, o = divmod(r, W)
            by.setdefault((m, o), []).append((a, grp))
        for (m, o), items in sorted(by.items()):
            ys = []
            for P in PL:
                y = None
                for sh, coef in ((o, 0.5), (o + 1, -0.5j)):
                    x = P[m] if sh == 0 else ops.rotate(P[m], sh * d.tH)
                    x = rt.mult_mask(x, ("pf.vpk", W, complex(coef)),
                                     lambda coef=coef: np.where(blk_of % W == 0, coef, 0.0))
                    y = x if y is None else ops.add(y, x)
                ys.append(rotsum(ops, y, d.tH, W * d.tH))
            for a, grp in items:
                pbs = [rotsum(ops, rt.mult_mask(y, ("pf.vup", a, W), lambda a=a: win_mask(a, W - 1)), W * d.tH, d.N)
                       for y in ys]
                vrs = [vown[e] if cp == c else v_rot(cp, e, c) for _, cp, e in grp]
                vc = vrs[0] if len(vrs) == 1 else ops.add(vrs[0], ops.mult_i(vrs[1]))
                for i, pb in enumerate(pbs):
                    prod = ops.mult(pb, vc)
                    if accs[i] is None:
                        accs[i] = prod
                    else:
                        ops.inplace_add(accs[i], prod)
        return accs

    def qkt_stage(c, QL):
        """Chunk c's packed scores and what its P.V needs, per query set of QL (one K pass)."""
        offs = _offsets(ch, c)
        nq = len(ch.lanes(c))
        if bsgs:
            bs = range(1, Bb) if ch.bidir else range(1, min(Bb, nq))
            prime(("K", c), K[c], [-b for b in bs] + ([d.t - b for b in bs] if ch.bidir else []))
        elif c not in owns:
            owns[c] = ([K[c]] + [lane_rotate(rt, K[c], e, "pf.krot") for e in range(1, ch.t)] if ch.bidir
                       else [K[c]] + ops.rotate_many(K[c], [-e for e in range(1, nq)]))
        own = owns.get(c)
        u_pre = prefix_offset(ch, c)                               # prefix key's offset
        M = (max([u for u, _, _ in offs] + ([u_pre] if prefix is not None else [])) // d.d_head) + 1
        SL = [[None] * M for _ in QL]
        pre = prefix is not None
        per_m = (u_pre // d.d_head) if (pre and _PRE_PERIODIC and u_pre % d.d_head == 0
                                        and all(u < u_pre for u, _, _ in offs)) else None
        pair = _ATTN_PAIR
        half = 0.5 if pair else 1.0          # paired: scores at 1/2, one cleanse per score ct
        packed = pair and _PACK > 0
        if packed:
            by_u = {}
            for o in offs:
                by_u.setdefault(o[0] // 2, []).append(o)
            groups = [sorted(g) for _, g in sorted(by_u.items())]
        else:
            groups = [offs[i:i + 2] for i in range(0, len(offs), 2)] if pair else [[o] for o in offs]

        def smask(m, blk, coef):
            def build():
                valid = _score_valid(d, ch, c, m, pre)
                return np.where(valid & (np.arange(d.N) // d.tH == blk), coef, 0.0)
            return build

        def place(dst, ct, parts, tag):
            """ct times the block masks of `parts` [(m, blk, coef)], accumulated into dst[m]."""
            by_m = {}
            for m, blk, coef in parts:
                by_m.setdefault(m, []).append((blk, coef))
            for m, lst in by_m.items():
                builds = [smask(m, blk, coef) for blk, coef in lst]
                key = (tag, c, m, tuple((blk, complex(coef)) for blk, coef in lst)) + ch.key_tag + (pre,)
                v = rt.mult_mask(ct, key, lambda builds=builds: sum(b() for b in builds))
                if dst[m] is None:
                    dst[m] = v
                else:
                    ops.inplace_add(dst[m], v)

        with rt.step("qkt"):
            if pre:
                for Q, S in zip(QL, SL):
                    m, blk = divmod(u_pre, d.d_head)
                    if bsgs:
                        blk = _bsgs_block_of(blk, d.d_head)
                    if per_m is not None:
                        k0 = np.asarray(prefix[0]) * (half * gscale)
                        S[m] = ops.tag_reduce(rotsum(ops, rt.mult_mask(Q[c], ("pf.pre.k", _vkey(k0)), lambda: k0),
                                                     d.tH, d.N), d.tH)
                        continue
                    q0 = rt.mult_mask(Q[c], ("pf.pre.k", _vkey(prefix[0])), lambda: np.asarray(prefix[0])) \
                        if _PRE_PERIODIC else ops.mult_pt(Q[c], prefix[0])
                    s = rotsum(ops, q0, d.tH, d.N)
                    place(S, s, [(m, blk, half * gscale)], "pf.qkt")
            if packed and bsgs:
                gs = range(1, d.t // Bb) if ch.bidir else range(1, -(-nq // Bb))
                for i, Q in enumerate(QL):
                    prime(("Q", qset[id(Q)], c), Q[c],
                          [Bb * g for g in gs] + ([Bb * g - d.t for g in gs] if ch.bidir else []))
                qkt_bsgs(c, QL, SL, groups, pre)
            elif packed:
                qkt_packed(c, QL, SL, groups, own, pre)
            else:
                for grp in groups:
                    krs = [own[e] if cp == c else k_rot(cp, e, c) for _, cp, e in grp]
                    kc = krs[0] if len(krs) == 1 else ops.add(krs[0], ops.mult_i(krs[1]))
                    for Q, S in zip(QL, SL):
                        s = rotsum(ops, ops.mult(Q[c], kc), d.tH, d.N)
                        coefs = (half * gscale, -0.5j * gscale)
                        place(S, s, [(*divmod(u, d.d_head), coefs[i]) for i, (u, _, _) in enumerate(grp)], "pf.qkt")
            if pair:
                SL = [[im_cleanse(ops, x) for x in S] for S in SL]
        return [dict(S=S, offs=offs, nq=nq, u_pre=u_pre, pre=pre, groups=groups, per=per_m) for S in SL]

    def pv_stage(c, sts, PL):
        st = sts[0]
        offs, nq, u_pre, pre, groups = st["offs"], st["nq"], st["u_pre"], st["pre"], st["groups"]
        with rt.step("softmax_v"):
            if bsgs:
                bs = range(1, Bb) if ch.bidir else range(1, min(Bb, nq))
                prime(("V", c), V[c], [-b for b in bs] + ([d.t - b for b in bs] if ch.bidir else []))
            elif c not in vowns:
                vowns[c] = ([V[c]] + [lane_rotate(rt, V[c], e, "pf.vrot") for e in range(1, ch.t)] if ch.bidir
                            else [V[c]] + ops.rotate_many(V[c], [-e for e in range(1, nq)]))
            vown = vowns.get(c)
            accs = [None] * len(PL)
            if pre:
                m, blk = divmod(u_pre, d.d_head)
                if bsgs:
                    blk = _bsgs_block_of(blk, d.d_head)
                for i, P in enumerate(PL):
                    if st["per"] is not None:
                        v0 = 0.5 * np.asarray(prefix[1])
                        accs[i] = rt.mult_mask(P[m], ("pf.pre.v", _vkey(v0)), lambda: v0)
                        continue
                    pb = rt.mult_mask(P[m], ("pf.bsel", blk), lambda blk=blk: np.where(
                        np.arange(d.N) // d.tH == blk, 0.5, 0.0))
                    pb = rotsum(ops, pb, d.tH, d.N)
                    accs[i] = rt.mult_mask(pb, ("pf.pre.v", _vkey(prefix[1])), lambda: np.asarray(prefix[1])) \
                        if _PRE_PERIODIC else ops.mult_pt(pb, prefix[1])
            if _ATTN_PAIR and _PACK > 0:
                pas = pv_bsgs(c, PL, groups) if bsgs else pv_packed(c, PL, groups, vown)
                accs = [pa if acc is None else ops.add(acc, pa) for acc, pa in zip(accs, pas)]
                groups = []
            for grp in groups:
                pbs = []
                for P in PL:
                    sel = [None] * len(P)
                    coefs = (0.5, -0.5j)
                    for i, (u, _, _) in enumerate(grp):
                        m, blk = divmod(u, d.d_head)
                        mk = lambda blk=blk, cf=coefs[i]: np.where(np.arange(d.N) // d.tH == blk, cf, 0.0)
                        v = rt.mult_mask(P[m], ("pf.bsel", blk, complex(coefs[i])), mk)
                        if sel[m] is None:
                            sel[m] = v
                        else:
                            ops.inplace_add(sel[m], v)
                    pb = None
                    for v in sel:
                        if v is not None:
                            pb = v if pb is None else ops.add(pb, v)
                    pbs.append(rotsum(ops, pb, d.tH, d.N))
                vrs = [vown[e] if cp == c else v_rot(cp, e, c) for _, cp, e in grp]
                vc = vrs[0] if len(vrs) == 1 else ops.add(vrs[0], ops.mult_i(vrs[1]))
                for i, pb in enumerate(pbs):
                    prod = ops.mult(pb, vc)
                    if accs[i] is None:
                        accs[i] = prod
                    else:
                        ops.inplace_add(accs[i], prod)
            return [im_cleanse(ops, acc) for acc in accs]

    hpack = (_SM_HPACK and not _SM_JOINT and prefix is None and 2 * d.H_real <= d.H
             and (nc % 2 == 0 if ch.bidir else nc == 2))

    def attend(QL):
        outs = [[None] * nc for _ in QL]
        if _SM_JOINT:
            for k in range(0, nc, 2):
                cs = [c for c in (k, k + 1) if c < nc]
                sts = {c: qkt_stage(c, QL) for c in cs}
                for i in range(len(QL)):
                    with rt.step("softmax"):
                        zs = {c: _sm_cheb(rt, sts[c][i]["S"], cfg, ch, c, sts[c][i]["pre"]) for c in cs}
                        fac = {c: None for c in cs}
                        if len(cs) == 2:
                            fac[cs[0]], fac[cs[1]] = _sm_joint(rt, zs[cs[0]], zs[cs[1]], cfg)
                        Ps = {c: prefill_softmax(rt, sts[c][i]["S"], cfg, ch, c, sts[c][i]["pre"],
                                                 Z=_sm_tail(rt, zs[c], cfg, ch, c, sts[c][i]["pre"], fac[c]))
                              for c in cs}
                    for c in cs:
                        outs[i][c] = pv_stage(c, [sts[c][i]], [Ps[c]])[0]
        elif hpack:
            h2 = d.tH // 2
            for k in range(0, nc, 2):
                sts = [qkt_stage(c, QL) for c in (k, k + 1)]
                PL = [[], []]
                for s0, s1 in zip(*sts):
                    with rt.step("softmax"):
                        n0 = len(s0["S"])
                        S = [ops.add(a, ops.rotate(b, d.N - h2)) for a, b in zip(s0["S"], s1["S"])] + \
                            [ops.rotate(x, d.N - h2) for x in s1["S"][n0:]]
                        P = prefill_softmax(rt, S, cfg, ch, _hp(k // 2))
                    PL[0].append(P[:n0])
                    PL[1].append([ops.rotate(x, h2) for x in P])
                for j, c in enumerate((k, k + 1)):
                    for i, o in enumerate(pv_stage(c, sts[j], PL[j])):
                        outs[i][c] = o
        else:
            for c in range(nc):
                sts = qkt_stage(c, QL)
                PL = []
                for st in sts:
                    with rt.step("softmax"):
                        PL.append(prefill_softmax(rt, st["S"], cfg, ch, c, st["pre"], periodic=st["per"]))
                for i, o in enumerate(pv_stage(c, sts, PL)):
                    outs[i][c] = o
        return [[join(ops, o[2 * c], o[2 * c + 1] if ch.has(2 * c + 1) else None) for c in range(ch.n_ct)]
                for o in outs]

    res = attend(Qs) if (shared and _KV_SHARE) else [attend([Q])[0] for Q in Qs]
    return res if shared else res[0]


def prefill_rot_steps(dims) -> set[int]:
    """Rotations the prefill adds to the decode band."""
    steps = set()
    for k in range(1, dims.t):
        steps |= {k, -k, dims.t - k}
    for k in range(1, dims.tp_E):
        steps |= {k, -k}
    r = dims.t
    while r < 512:                           # norm_all's region spread ladder
        steps.add(-r)
        r *= 2
    return {s % dims.N for s in steps if s % dims.N}


# ── dense diagonal linears ──────────────────────────────────────────────────────────────────────────────────────

class DiagLinear:
    """y = x W on dense token-layout ciphertexts, BSGS diagonal method (Halevi-Shoup) on B x B blocks, B = N / t:
        y_m = sum_g rot( sum_{n, j} P[n, m, g, j] * rot(x_n, j t), g b t ); `rows` (pinned) or `pts` (slots)."""

    def __init__(self, n_in, n_out, b, g, pts=None, rows=None, tags=None, bias=None, name=None):
        self.n_in, self.n_out, self.b, self.g = n_in, n_out, b, g
        self.pts, self.rows, self.tags, self.bias, self.name = pts, rows, tags, bias, name

    @staticmethod
    def split(B):
        b = 1 << ((B.bit_length() - 1 + 1) // 2)          # b ~ sqrt(B), a power of two, b >= g
        return b, B // b

    @classmethod
    def slot_vectors(cls, W, N, t, scale=1.0):
        """(n_in, n_out, g, b, N) plaintext slot vectors of W (n_in B, n_out B) in the order the call consumes."""
        W = np.asarray(W, dtype=np.float64) * scale
        B = N // t
        n_in, n_out = W.shape[0] // B, W.shape[1] // B
        b, g = cls.split(B)
        i = np.arange(B)
        out = np.empty((n_out, g, n_in, b, N))
        for m in range(n_out):
            for n in range(n_in):
                blk = W[n * B:(n + 1) * B, m * B:(m + 1) * B]
                for gg in range(g):
                    for j in range(b):
                        k = gg * b + j
                        v = np.repeat(blk[(i + k) % B, i], t)
                        out[m, gg, n, j] = np.roll(v, gg * b * t)
        return out

    @classmethod
    def encode(cls, W, N, t, scale=1.0, bias=None, name=None, pinned=None, core=None, cache=None):
        """From dense W (n_in B, n_out B): pinned coefficient rows (cached at `cache`) when `pinned`, else slots."""
        B = N // t
        n_in, n_out = W.shape[0] // B, W.shape[1] // B
        b, g = cls.split(B)
        bvec = None
        if bias is not None:
            bvec = [np.repeat(np.asarray(bias, dtype=np.float64)[m * B:(m + 1) * B] * scale, t) for m in range(n_out)]
        if pinned is None:
            return cls(n_in, n_out, b, g, pts=cls.slot_vectors(W, N, t, scale), bias=bvec, name=name)
        import os
        from .layout import coeff_rows
        n_rows = n_out * g * n_in * b
        if cache is not None and os.path.exists(cache + ".coeff.npy") and os.path.exists(cache + ".tags.npy"):
            src = np.load(cache + ".coeff.npy", mmap_mode="r")
            tags = np.load(cache + ".tags.npy")
        else:
            pts = cls.slot_vectors(W, N, t, scale).reshape(n_rows, N)
            src = coeff_rows(pts)
            tags = core.pt_tag_fields(np.ascontiguousarray(pts))
            if cache is not None:
                for path, arr in ((cache + ".coeff.npy", src), (cache + ".tags.npy", tags)):
                    tmp = path + f".{os.getpid()}.tmp.npy"
                    np.save(tmp, arr)
                    os.replace(tmp, path)
        rows = pinned(n_rows, 2 * N)
        rows[:] = src
        lin = cls(n_in, n_out, b, g, rows=rows, tags=np.ascontiguousarray(tags), bias=bvec, name=name)
        lin.src = (np.asarray(W, dtype=np.float64), N, t, scale)
        return lin

    def _sum(self, ops, xs, m, gg):
        """sum_{n, j} P[n, m, gg, j] * xs[n][j]."""
        cts = [x for xn in xs for x in xn]
        k = self.n_in * self.b
        r0 = (m * self.g + gg) * k
        if self.rows is not None and hasattr(ops, "mult_coeff_sum"):
            return ops.mult_coeff_sum(cts, self.rows[r0:r0 + k], self.tags[r0:r0 + k], _LazyRows(self, m, gg))
        vecs = self.pts[m, gg].reshape(k, -1)
        return ops.mult_pt_sum(cts, vecs)


class _LazyRows:
    """Slot vectors of one (m, g) group, built only when mult_coeff_sum falls back to mult_pt_sum."""

    def __init__(self, lin, m, gg):
        self.lin, self.m, self.gg = lin, m, gg

    def __array__(self, dtype=None, copy=None):
        lin = self.lin
        W, N, t, scale = lin.src
        B = N // t
        i = np.arange(B)
        out = np.empty((lin.n_in * lin.b, N))
        for n in range(lin.n_in):
            blk = W[n * B:(n + 1) * B, self.m * B:(self.m + 1) * B] * scale
            for j in range(lin.b):
                k = self.gg * lin.b + j
                out[n * lin.b + j] = np.roll(np.repeat(blk[(i + k) % B, i], t), self.gg * lin.b * t)
        return out if dtype is None else out.astype(dtype)


def diag_apply(rt, Xs, lins, scale_bias=None):
    """DiagLinears `lins` (same n_in) on input blocks Xs of one ciphertext group; per linear, its n_out blocks."""
    ops, d = rt.ops, rt.dims
    b = lins[0].b
    for x in Xs:
        ops.bootstrap_hint(x, ops.headroom(2), True)       # the products (one unit deeper) stay under the limit
    xs = [[x] + ops.rotate_many(x, [j * d.t for j in range(1, b)]) for x in Xs]
    for xn in xs:
        for x in xn:
            ops.realize(x)
    out = []
    for lin in lins:
        blocks = []
        for m in range(lin.n_out):
            acc = None
            for gg in range(lin.g):
                tmp = lin._sum(ops, xs, m, gg)
                if gg:
                    tmp = ops.rotate(tmp, gg * lin.b * d.t)
                if acc is None:
                    acc = tmp
                else:
                    ops.inplace_add(acc, tmp)
            blocks.append(acc)
        out.append(blocks)
    return out


def diag_bias(rt, y, lin, m, ch: Chunks, c: int):
    """Output block m of ciphertext c plus (b + i b) on the filled lanes."""
    if lin.bias is None:
        return y
    d = rt.dims
    na = len(ch.lanes(2 * c))
    nb = len(ch.lanes(2 * c + 1)) if ch.has(2 * c + 1) else 0
    lane = np.arange(d.N) % d.t
    fac = (lane < na).astype(np.float64) + 1j * (lane < nb)
    return rt.add_mask(y, ("pf.dbias", lin.name or id(lin), m, na, nb), lambda: lin.bias[m] * fac)


def diag_rot_steps(dims) -> set[int]:
    b, g = DiagLinear.split(dims.N // dims.t)
    steps = {j * dims.t for j in range(1, b)} | {k * b * dims.t for k in range(1, g)}
    return {s % dims.N for s in steps if s % dims.N}


# ── lockstep halves ─────────────────────────────────────────────────────────────────────────────────────────────

class Pair:
    """The two real halves (chunks 2c, 2c + 1) of a value under LockstepOps."""
    __slots__ = ("a", "b")

    def __init__(self, a, b):
        self.a, self.b = a, b


class LockstepOps:
    """Ops on Pairs, each op on both halves; an operand near the ceiling is joined, bootstrapped once and split back."""

    total = 0                             # joint refreshes since import

    def __init__(self, ops):
        self._o = ops
        self.refreshes = 0

    def __getattr__(self, name):
        return getattr(self._o, name)

    def _two(self, f, x, y=None):
        if y is None:
            return Pair(f(x.a), f(x.b))
        if isinstance(y, Pair):
            return Pair(f(x.a, y.a), f(x.b, y.b))
        return Pair(f(x.a, y), f(x.b, y))

    def _ceiling(self):
        """Two units under the refresh envelope (PF_LOCKSTEP_TOP, BTS_MAX_INPUT_LEVEL, or the session limit)."""
        import os
        o = self._o
        top = int(os.environ.get("PF_LOCKSTEP_TOP") or os.environ.get("BTS_MAX_INPUT_LEVEL") or o.level_limit())
        return top - 2 * o.unit

    def _guard(self, x):
        o = self._o
        if isinstance(x, Pair) and max(o.lvl(x.a), o.lvl(x.b)) > self._ceiling():
            X = o.add(x.a, o.mult_i(x.b))
            if getattr(o, "folds", False):
                o.fold_bootstrap(X, self._o.inf.slots, 2)     # the split's 1/2 inside the refresh
                half = None
            else:
                o.bootstrap(X)
                half = 0.5
            o.device_sync()               # a conjugate right after a complex refresh must see it landed (poly.py)
            A, B = split(o, X)
            x = Pair(A, B) if half is None else Pair(o.mult(A, half), o.mult(B, half))
            self.refreshes += 1
            LockstepOps.total += 1
        return x

    def add(self, x, y): return self._two(self._o.add, x, y)
    def sub(self, x, y): return self._two(self._o.sub, x, y)
    def negate(self, x): return self._two(self._o.negate, x)
    def conjugate(self, x): return self._two(self._o.conjugate, x)
    def copy(self, x): return self._two(self._o.copy, x)

    def mult(self, x, y):
        x = self._guard(x)
        y = self._guard(y) if isinstance(y, Pair) else y
        return self._two(self._o.mult, x, y)

    def square(self, x):
        return self._two(self._o.square, self._guard(x))

    def inplace_add(self, x, y):
        if isinstance(y, Pair):
            self._o.inplace_add(x.a, y.a); self._o.inplace_add(x.b, y.b)
        else:
            self._o.inplace_add(x.a, y); self._o.inplace_add(x.b, y)

    def mult_pt(self, x, v, key=None, kind="weight", tagged=True):
        x = self._guard(x)
        return Pair(self._o.mult_pt(x.a, v, key=key, kind=kind, tagged=tagged),
                    self._o.mult_pt(x.b, v, key=key, kind=kind, tagged=tagged))

    def add_pt(self, x, v, key=None, kind="weight", tagged=True):
        return Pair(self._o.add_pt(x.a, v, key=key, kind=kind, tagged=tagged),
                    self._o.add_pt(x.b, v, key=key, kind=kind, tagged=tagged))

    def lvl(self, x): return max(self._o.lvl(x.a), self._o.lvl(x.b)) if isinstance(x, Pair) else self._o.lvl(x)
    def level(self, x): return max(self._o.level(x.a), self._o.level(x.b)) if isinstance(x, Pair) else self._o.level(x)

    def bootstrap_hint(self, x, thr, acct=False):
        return x                          # the refreshes are _guard's

    def tag_reduce(self, x, stride):
        return self._two(lambda c: self._o.tag_reduce(c, stride), x)

    def realize(self, x):
        return self._two(self._o.realize, x)


def lockstep_rt(rt):
    """rt with LockstepOps (shared mask caches)."""
    import copy
    r = copy.copy(rt)
    r.ops = LockstepOps(rt.ops)
    return r


# ── shared-chain norms ──────────────────────────────────────────────────────────────────────────────────────────

def _region_mask(d, R, k, val):
    """val on the slots of region k (slot s: (s // t) % R == k), zero elsewhere."""
    return np.where((np.arange(d.N) // d.t) % R == k, val, 0.0)


def norm_all(rt, Xs, cfg, ch: Chunks, rms: bool = False, max_period: int = 512):
    """norm_pair on every ciphertext with ONE inverse-sqrt chain: chunk k's statistic goes to region k of an
    (R t)-periodic ciphertext (R t <= max_period) and is spread back by a log R ladder."""
    ops, d = rt.ops, rt.dims
    N, t, dr = d.N, d.t, d.dim
    halves = []                              # (ct, half, lane scales, input, statistic)
    for c, X in enumerate(Xs):
        for half in (0, 1):
            if not ch.has(2 * c + half):
                continue
            pos = ch.lanes(2 * c + half)
            cs = tuple(round(math.sqrt(cfg.c_eff_sq(p)), 12) for p in pos)
            x2 = im_cleanse(ops, X if half == 0 else ops.mult_i(X))      # 2 Re X, or 2 Re(iX) = -2 B
            xs = rt.mult_mask(x2, ("ln.center.L", cs), lambda cs=cs: lanes_mask(N, dr, t, [0.5 * c_ for c_ in cs]))
            if rms:
                S = ops.tag_reduce(rotsum(ops, ops.square(xs), t, N), t)
                halves.append((c, half, cs, xs, S))
            else:
                mean = ops.tag_reduce(rotsum(ops, xs, t, N), t)
                mean = rt.mult_mask(mean, ("ln.scalemask.L", len(cs)),
                                    lambda n=len(cs): lanes_mask(N, dr, t, [-1.0 / dr] * n))
                centered = ops.add(xs, mean)
                var = ops.tag_reduce(rotsum(ops, ops.square(centered), t, N), t)
                halves.append((c, half, cs, centered, var))
    per = max(1, max_period // t)
    outs = {}
    for g0 in range(0, len(halves), per):
        grp = halves[g0:g0 + per]
        R = 1 << (len(grp) - 1).bit_length()
        V = None
        eps = np.zeros(N)
        for k, (_c, _h, cs, _x, st) in enumerate(grp):
            sc = 1.0 if rms else 1.0 / dr
            m = rt.mult_mask(st, ("pf.nreg", R, k, sc), lambda R=R, k=k, sc=sc: _region_mask(d, R, k, sc))
            V = m if V is None else ops.add(V, m)
            e = [(dr if rms else 1.0) * cfg.epsilon * c_ * c_ for c_ in cs]
            lane_e = lanes_periodic(N, t, e, (dr if rms else 1.0) * 0.5 * (cfg.cheb_lo + cfg.cheb_hi)
                                    if cfg.cheb_coeffs else 1.0)
            eps += np.where((np.arange(N) // t) % R == k, lane_e, 0.0)
        for k in range(len(grp), R):         # empty regions: a value in the seed's band
            eps += _region_mask(d, R, k, (dr if rms else 1.0) * 0.5 * (cfg.cheb_lo + cfg.cheb_hi)
                                if cfg.cheb_coeffs else 1.0)
        V = rt.add_mask(V, ("pf.neps", rms, R, tuple(g[2] for g in grp), cfg.epsilon), lambda eps=eps: eps)
        V = ops.tag_reduce(V, R * t)
        lo, hi = (dr * cfg.cheb_lo, dr * cfg.cheb_hi) if rms else (cfg.cheb_lo, cfg.cheb_hi)
        if not (cfg.cheb_coeffs and ops.ln_cheb):
            raise ValueError("norm_all: needs the Chebyshev inverse-sqrt seed (LN_CHEB=1, cheb_* config)")
        init = eval_chebyshev(ops, V, cfg.cheb_coeffs, lo, hi)
        xsc = 1.0 / ((dr if rms else 1.0) * cfg.inv_out_scale * cfg.inv_out_scale)
        from .poly import inv_sqrt_newton_d2
        inv = inv_sqrt_newton_d2(ops, V, init, cfg.cheb_nr_iters, xsc)
        ops.bootstrap_hint(inv, ops.headroom(4))
        for k, (c, half, _cs, x, _st) in enumerate(grp):
            ik = rt.mult_mask(inv, ("pf.nreg", R, k, 1.0), lambda R=R, k=k: _region_mask(d, R, k, 1.0))
            if R > 1:
                ik = ops.rotate_and_sum(ik, t, R * t)
            outs[(c, half)] = im_cleanse(ops, ops.mult(x, ik))
    res = []
    for c in range(len(Xs)):
        nA = outs[(c, 0)]
        res.append(nA if (c, 1) not in outs else join(ops, nA, ops.negate(outs[(c, 1)])))
    return res


# PF_JOINT_FORCE=1: the joint GELU / softmax refresh is a deliberate bootstrap instead of a hint
_JOINT_FORCE = os.environ.get("PF_JOINT_FORCE", "0") not in ("", "0")

# PF_SILU_SINGLE=1 (with `us`): sigma(xmax t) - 1/2 as one odd Chebyshev series of degree 63 / 95 / 127, the smallest
# within PF_SILU_SINGLE_TOL x the composite's error (both as evaluated, trimmed; floor 1e-7) on t in [-1, 1]:
# no refresh in the GLU
_SILU_SINGLE = os.environ.get("PF_SILU_SINGLE", "0") not in ("", "0")
_SILU_SINGLE_TOL = float(os.environ.get("PF_SILU_SINGLE_TOL", "2"))
_SILU_SINGLE_EPS = float(os.environ.get("PF_SILU_SINGLE_EPS", "1e-8"))       # the series' coefficient trim
_SILU_SINGLE_DEGS = [int(v) for v in os.environ.get("PF_SILU_SINGLE_DEGS", "63,95,127").split(",")]
_SILU_SINGLE_ABS = float(os.environ.get("PF_SILU_SINGLE_ABS", "0"))   # else the last degree if within this abs error
# PF_SILU_SINGLE_ERR=<abs>: instead, the smallest degree whose |x| |series - f| stays within <abs> (composite-free)
_SILU_SINGLE_ERR = float(os.environ.get("PF_SILU_SINGLE_ERR", "0"))
# PF_GLU_LEAN=1 (forced joint, single series): x u from the split halves as they are (no second cleanse; its 2 rides
# the series' coefficients)
_GLU_LEAN = os.environ.get("PF_GLU_LEAN", "0") not in ("", "0")
_silu_single_cache = {}


def silu_single_coeffs(cfg):
    """The single-series coefficients of sigma(xmax t) - 1/2 on t in [-1, 1], or None (keep the composite)."""
    from numpy.polynomial import chebyshev as C
    from .poly import _cheb_trim
    key = (float(cfg.xmax), tuple(cfg.thor_p1_cheb), tuple(cfg.thor_p2_cheb))
    if key not in _silu_single_cache:
        xm = float(cfg.xmax)
        t = np.linspace(-1.0, 1.0, 40001)
        f = 1.0 / (1.0 + np.exp(-xm * t)) - 0.5
        a1, b1, a2, b2 = cfg.thor_p1_a, cfg.thor_p1_b, cfg.thor_p2_a, cfg.thor_p2_b
        y1 = C.chebval((2 * t - (a1 + b1)) / (b1 - a1), _cheb_trim(cfg.thor_p1_cheb))
        comp = C.chebval((2 * y1 - (a2 + b2)) / (b2 - a2), _cheb_trim(cfg.thor_p2_cheb))
        tol = _SILU_SINGLE_ERR or max(_SILU_SINGLE_TOL * float((np.abs(comp - f) * np.abs(t) * xm).max()), 1e-7)
        best = None
        for n in _SILU_SINGLE_DEGS:
            c = C.chebinterpolate(lambda t: 1.0 / (1.0 + np.exp(-xm * t)) - 0.5, n)
            c[0::2] = 0.0
            c = np.array(_cheb_trim(c, _SILU_SINGLE_EPS))
            err = float((np.abs(C.chebval(t, c) - f) * np.abs(t) * xm).max())
            if err <= tol or (n == _SILU_SINGLE_DEGS[-1] and err <= _SILU_SINGLE_ABS):
                best = [float(v) for v in c]
                break
        _silu_single_cache[key] = best
    return _silu_single_cache[key]


def gelu_joint(rt, A, B, cfg, mask, us=None):
    """Folded THOR GELU of real halves A, B (x / xmax) with one refresh of p1(A) + i p1(B) between P1 and P2.
    Returns (g(A), g(B)) times the mask; with `us` (forced joint only), (g(A) uA, g(B) uB) without the final cleanse."""
    from .poly import eval_polynomial_ps
    ops = rt.ops
    if not (cfg.method == "thor_composite" and cfg.thor_p1_cheb and ops.gelu_fold):
        raise ValueError("gelu_joint: the folded THOR composite with Chebyshev P1 / P2 only")
    m = np.asarray(mask[1](1.0))
    if us is not None and not (_JOINT_FORCE and np.all(m == m.flat[0])):
        raise ValueError("gelu_joint: us needs the forced joint with a constant mask")
    if _JOINT_FORCE and np.all(m == m.flat[0]):
        # constant mask: P2's domain map and the split's 2 ride P1's coefficients, the mask P2's
        s_out = 0.5 * cfg.xmax * float(np.real(m.flat[0]))
        a2, b2 = cfg.thor_p2_a, cfg.thor_p2_b
        al, be = 2.0 / (b2 - a2), (a2 + b2) / (b2 - a2)
        c1 = [0.5 * al * float(c) for c in cfg.thor_p1_cheb]
        c1[0] -= 0.5 * be
        if us is not None:
            cs = silu_single_coeffs(cfg) if _SILU_SINGLE else None
            lean = _GLU_LEAN and cs is not None
            xu = [ops.mult(x if lean else im_cleanse(ops, x), u) for x, u in zip((A, B), us)]
            if cs is not None:
                so = 2.0 * s_out if lean else s_out
                cs = [so * c for c in cs]
                return [ops.mult(ops.add(eval_chebyshev(ops, x, cs, -1.0, 1.0, _SILU_SINGLE_EPS), 0.5 * so), w)
                        for x, w in zip((A, B), xu)]
        p1 = [eval_chebyshev(ops, x, c1, cfg.thor_p1_a, cfg.thor_p1_b) for x in (A, B)]
        X = ops.add(p1[0], ops.mult_i(p1[1]))
        ops.bootstrap(X)
        ya, yb = split(ops, X)                                 # P2's domain variable
        c2 = [s_out * float(c) for c in cfg.thor_p2_cheb]
        if us is not None:
            return [ops.mult(ops.add(eval_chebyshev(ops, y, c2, -1.0, 1.0), 0.5 * s_out), w)
                    for w, y in zip(xu, (ya, yb))]
        return [ops.mult(ops.add(eval_chebyshev(ops, y, c2, -1.0, 1.0), 0.5 * s_out), im_cleanse(ops, x))
                for x, y in ((A, ya), (B, yb))]
    p1 = [eval_chebyshev(ops, x, cfg.thor_p1_cheb, cfg.thor_p1_a, cfg.thor_p1_b) for x in (A, B)]
    X = ops.add(p1[0], ops.mult_i(p1[1]))
    ops.bootstrap_hint(X, ops.headroom(4), True)
    p1a, p1b = split(ops, X)                                   # 2 p1(A), 2 p1(B)
    out = []
    for x, p in ((A, p1a), (B, p1b)):
        g = eval_chebyshev(ops, p, cfg.thor_p2_cheb, 2.0 * cfg.thor_p2_a, 2.0 * cfg.thor_p2_b)
        g = ops.add(g, 0.5)
        g = ops.mult(g, im_cleanse(ops, x))
        out.append(rt.mult_mask(g, (mask[0], cfg.xmax), lambda: mask[1](0.5 * cfg.xmax)))
    return out


# ── planned prefill on the new runtime: one Program per block, PCMM finishes served in Rust ─────────────────────────
#
# The prefill is one forward: its op stream depends on T and the layout, never on the prompt's values. `PlannedPrefill`
# records that stream once on the model's CPU twin (impl.driver.Recorder: the fakecap model, structure + plaintext
# values), compiles every block against the plan (pc-program; the PCMM linears are `Finish` ops whose descriptor is
# the capture's `args.external` dict), and binds them on one planned Session that also holds the PCMM service: its
# keys made at setup from the secret (dropped afterwards), every linear's digit planes resident (GPT-2: 48 linears,
# ~250 MB at t = 32). The device run encrypts block 0's inputs at their compiled level and runs the blocks
# (`Program.run`, GIL released, eager: a one-shot forward gains nothing from a CUDA-graph capture), carrying the
# block outputs in registers.

class PlannedPrefill:
    """PLANNED prefill on the new runtime (see above). `model`: the prefill model on the CPU twin (fakecap session);
    `plan_dir`: the block plans; `config`: the session Config (plan_runtime over the prefill preset)."""

    def __init__(self, model, plan_dir, config, keys, *, work_dir, pt_rows=28, stc_cache=8, pool=16, pcmm_cf=6,
                 check_meta=False, log=print, schedule="siblings"):
        self.model, self.plan_dir, self.cfg, self.keys = model, str(plan_dir), config, keys
        self.work_dir, self.pt_rows, self.stc_cache, self.pool = work_dir, pt_rows, stc_cache, pool
        self.pcmm_cf, self.check_meta, self.log = pcmm_cf, check_meta, log
        self.schedule = schedule
        self.n_layers = model.n_layers
        self.n_stages = self.n_layers               # record sets it from the model's stages
        self.labels = [f"block_{b}" for b in range(self.n_layers)]

    # ── setup (CPU) ──
    def record(self, rows):
        """The forward's per-block streams on the CPU twin: {b: BlockRec} (`out_names`: the stage outputs' graph
        names); `self.in_names[b]` the stage inputs' graph names, `self.in_metas[b]` [(level, deg)] of the inputs when the stage starts, `self.in_vals` block
        0's input slot vectors."""
        import time
        from .driver import Recorder
        t0 = time.time()
        m = self.model
        rec = Recorder(m, self.n_layers)
        orig = m.prefill_stages
        io = {}

        def stages(ch):
            out = []
            for i, (lab, pre, fn) in enumerate(orig(ch)):
                def wrap(x, i=i, fn=fn):
                    metas = [(int(c.level), int(c.noise_deg)) for c in x]
                    vals = [np.array(c.vec) for c in x] if i == 0 else None
                    y = fn(x)
                    # graph names as this block's capture wrote them (the next block renames what it reads)
                    io[i] = ([c.name for c in x], [c.name for c in y], metas, vals)
                    return y
                out.append((lab, pre, wrap))
            return out
        m.prefill_stages = stages
        self.rows = np.asarray(rows, dtype=np.float64)
        try:
            _xs, self.ch, _ = m.prefill(rows)
        finally:
            del m.prefill_stages
        self.in_metas = {}
        self.in_names = {}
        for i, r in rec.blocks.items():
            names_in, names_out, metas, vals = io[i]
            r.out_names = names_out
            self.in_names[i] = names_in
            self.in_metas[i] = metas
            if i == 0:
                self.in_vals = vals
        self.recs = rec.blocks
        self.n_stages = len(self.recs)             # the blocks, plus the model's extra stages (GPT-2: with_tail)
        self.labels = [f"block_{b}" for b in range(self.n_stages)]
        self.record_s = time.time() - t0
        return self.recs

    def prepare(self, rows):
        """`record` + the programs compiled from the recorded streams against the plan."""
        import json
        import time
        from perseus import _pc
        if not hasattr(self, "recs"):
            self.record(rows)
        t0 = time.time()
        os.makedirs(self.work_dir, exist_ok=True)
        live, metas, self.in_pos = {}, {}, {}
        for b in range(self.n_stages):
            r, lab = self.recs[b], self.labels[b]
            d = os.path.join(self.work_dir, f"block_{b}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "graph.json"), "w") as f:
                json.dump({"version": 2, "nodes": r.nodes}, f)
            names = self.in_names[b]
            ins = r.inputs()
            missing = [n for n in ins if n not in names]
            if missing:
                raise RuntimeError(f"block {b}: inputs {missing} are not the stage's input ciphertexts {names}")
            self.in_pos[b] = [names.index(n) for n in ins]          # program input i = stage input in_pos[i]
            # the twin knows the entries up to the first extra stage; past it (GPT-2's CutMax after the tail, which the
            # twin ran unplanned) a stage enters at the previous stage's compiled exit
            if b <= self.n_layers:
                metas[lab] = {n: self.in_metas[b][names.index(n)] for n in ins}
            live[lab] = list(r.out_names)
        self.progs = _pc.Program.compile(self.work_dir, self.plan_dir, config=self.cfg, blocks=self.labels,
                                         live_out=live, input_metas=metas, schedule=self.schedule)
        # registers: every block's outputs (the stage order), kept to the end (the per-block check reads them)
        self.reg, n = {}, 0
        for b in range(self.n_stages):
            p = self.progs[self.labels[b]]
            self.reg[b] = []
            for o in self.recs[b].out_names:
                self.reg[b].append((p.output_index(o), n))
                n += 1
        self.n_regs = n
        self.compile_s = time.time() - t0
        return self

    def linears(self):
        """The PCMM linears the blocks' finishes name: {name: PcmmLinear} (the model's encoded weights)."""
        from .pcmm import PcmmLinear
        out = {}
        for b in range(self.n_layers):
            for lin in self.model.diag_weights(b).values():
                if isinstance(lin, PcmmLinear):
                    out[lin.name] = lin
        return out

    def session_kwargs(self):
        """The planned Session's arguments after (config, keys), which `Session.key_manifest` and `from_bundle` take
        too; records the plaintext aliases `setup` binds."""
        import hashlib
        from .driver import register_levels
        progs = [self.progs[lab] for lab in self.labels]
        # plaintexts by value: the blocks' masks repeat (one resident slot per distinct (values, level))
        self.alias, uniq = {}, {}
        for b, p in enumerate(progs):
            r, lab = self.recs[b], self.labels[b]
            for nm, lv, _d, form in p.pt_refs:
                v = np.ascontiguousarray(r.pts[nm])
                key = "v/" + hashlib.blake2b(v.tobytes(), digest_size=12).hexdigest()
                # bind's store name: the block's own `pt_N` namespaced, a shared name as is
                store = f"{lab}/{nm}" if (form == "at_use" or nm.startswith("pt_")) else nm
                self.alias.setdefault(lab, {})[store] = key
                uniq.setdefault((key, lv), v)
        self.uniq = uniq
        # a compact plaintext slot holds the rows of its level (n32: 54 - level Q primes); the deepest plaintext sizes it
        pt_rows = self.pt_rows and max(self.pt_rows, 54 - min(lv for _k, lv in uniq))
        t = self.model.rt.dims.t
        specs = [(lin.name, int(lin.t or t), int(lin.n_in), int(lin.n_out), int(lin.Wd.shape[0]), float(lin.w_scale),
                  int(lin.shift)) for lin in self.linears().values()]
        reg_levels = register_levels(self.n_regs, [(progs[b], j, r) for b in range(self.n_stages) for j, r in self.reg[b]])
        return dict(pool=self.pool, pt_slots=len(uniq) + 8, programs=progs, regs=self.n_regs, pt_rows=pt_rows,
                    stc_cache=self.stc_cache, pcmm=specs or None, pcmm_cf=self.pcmm_cf, check_meta=self.check_meta,
                    reg_levels=reg_levels, pt_levels=[lv for _k, lv in uniq])

    def open_session(self, kw):
        from perseus import _pc
        return _pc.Session(self.cfg, self.keys, **kw)

    def setup(self, rows):
        """`prepare` (CPU) + the planned Session: routes, keys (+ the PCMM keys, then the secret is dropped), slab,
        registers, the PCMM service with every linear resident; every block bound, its plaintexts staged."""
        import time
        if not hasattr(self, "progs"):
            self.prepare(rows)
        t0 = time.time()
        kw = self.session_kwargs()
        progs, uniq, lins, specs = kw["programs"], self.uniq, self.linears(), kw["pcmm"] or []
        n_pts = len(uniq)
        from .driver import plan_rot_scope
        self.rot_keys = plan_rot_scope(self.plan_dir, progs)
        self.sess = self.open_session(kw)
        self.store = self.sess.pt_store
        for b, p in enumerate(progs):
            p.bind(self.sess, self.store)
            p.alias_pts(self.alias.get(self.labels[b], {}))
        self.sess.complete_setup()
        if specs:
            for lin in lins.values():
                self.sess.pcmm_upload(lin.name, np.ascontiguousarray(lin.Wd, dtype=np.int8).tobytes())
        self.sess.drop_secret()
        staged = self.store.stage([(key, lv, v) for (key, lv), v in uniq.items()])
        n_refs = sum(len({(nm, lv) for nm, lv, _d, _f in p.pt_refs}) for p in progs)
        self.fhe = self.sess.fhe
        self.setup_s = time.time() - t0
        self.log(f"[planned] {len(progs)} programs, {self.n_regs} registers, {n_refs} plaintext keys = {n_pts} distinct "
                 f"(values, level) ({staged} staged), "
                 f"{len(specs)} PCMM linears, rotation keys {self.rot_keys[0]} (plan_rot_steps.json {self.rot_keys[1]}); "
                 f"record {self.record_s:.1f} s, compile {self.compile_s:.1f} s, "
                 f"session {self.sess.setup_seconds()[0]:.1f} s (keygen {self.sess.setup_seconds()[1]:.1f}), "
                 f"setup total {self.setup_s:.1f} s")
        self.log("[planned] " + self.sess.mem_report.replace("\n", " | "))
        return self

    def input_values(self, rows):
        """Block 0's input slot vectors for another prompt of the recorded length (the warm-up runners: GPT2_PF_WARMUP
        semantics), packed as the model's prefill packs them; checked against the recorded prompt's own."""
        from .blocked import blocked_slots

        def pack(r):
            if os.environ.get("PF_LAYOUT", "") == "blocked":
                return [v for row in blocked_slots(r, self.ch, self.model.blocks) for v in row]
            return prompt_slots(r, self.ch, self.model.rt.dims)
        if not hasattr(self, "_rows_ok"):
            mine = pack(self.rows)
            if len(mine) != len(self.in_vals) or not all(np.allclose(a, b, atol=1e-9) for a, b in zip(mine, self.in_vals)):
                raise RuntimeError("PlannedPrefill.input_values: the packing does not reproduce the recorded block-0 inputs")
            self._rows_ok = True
        if np.asarray(rows).shape != np.asarray(self.rows).shape:
            raise ValueError(f"warm-up rows {np.asarray(rows).shape}: the programs were compiled for {np.asarray(self.rows).shape}")
        return pack(rows)

    # ── the device run ──
    def run(self, on_block=None, vals=None, x0=None):
        """The forward on the device (asynchronous until the caller decrypts); `on_block(b, info)` after each block's
        issue; `vals`: block 0's input slot vectors (default: the recorded prompt's; `input_values` for another);
        `x0`: block 0's inputs already encrypted (a client's) instead. Returns the last stage's output registers."""
        if x0 is None:
            vals = self.in_vals if vals is None else vals
            x0 = [self.fhe.encrypt(v, lv, dg) for v, (lv, dg) in zip(vals, self.in_metas[0])]
        info = []
        for b in range(self.n_stages):
            p = self.progs[self.labels[b]]
            if b == 0:
                ins = [x0[j] for j in self.in_pos[0]]
            else:
                prev = [reg for _j, reg in self.reg[b - 1]]
                ins = [prev[j] for j in self.in_pos[b]]
            info.append(p.run(ins, list(self.reg[b])))
            if on_block is not None:
                on_block(b, info[-1])
        return [reg for _j, reg in self.reg[self.n_stages - 1]]

    def ct(self, reg):
        from perseus import _pc
        return _pc.Program.reg_ct(self.sess, reg)

    def decrypt(self, regs):
        """Slot vectors (complex) of the registers (synchronous)."""
        return [np.asarray(self.fhe.decrypt_slots_complex(self.ct(r))) for r in regs]

    def hidden(self, b=None):
        """(T, dim) rows of block b's output (default: the last), decrypted and unpacked like `model.hidden`."""
        import types
        b = self.n_layers - 1 if b is None else b
        vals = self.decrypt([reg for _j, reg in self.reg[b]])
        shim = types.SimpleNamespace(ops=types.SimpleNamespace(decrypt_slots_complex=lambda v: v),
                                     dims=self.model.rt.dims)
        if os.environ.get("PF_LAYOUT", "") == "blocked":
            from .blocked import decrypt_blocked
            B = self.model.blocks
            return decrypt_blocked(shim, [vals[g * B.K:(g + 1) * B.K] for g in range(len(vals) // B.K)], self.ch, B,
                                   self.model.rt.dims.dim)
        return decrypt_prompt(shim, vals, self.ch, self.model.rt.dims.dim)

    def close(self):
        self.sess.close()
