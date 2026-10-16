"""SinC operand packing of the CCMM attention for multi-head attention with d_head = 64 (BERT / ViT-base: 768 = 12 x 64,
padded to 16 heads), and its maps to the channel-blocked prefill layout (blocked.py, nch = 512: 8 heads per block).

Blocked source: block k = h // 8, slot j 512 + (h % 8) 64 + lane (j < 64 the head dim), Re / Im = chunk 0 / 1 of a
128-token pair (token 64 c + lane). K and V enter per chunk as real ciphertexts (prefill split).

The CKL batch CCMM multiplies 32 batch 128 x 128 matrices (sinc.py's SinC_{1024, 2^16}). A head pair (2 p, 2 p + 1)
shares batch slot l = 2 p (even, < 16: PF_CCMM_MERGE's duplicate on l + 16 stays free):
  Q: row = brev6(lane) + 64 hp (hp = h % 2), col = col_of(j, hp), Re / Im = chunk; the other head's columns stay 0
  K, V: row = brev6(lane) + 64 c (keys), col = col_of(j, hp): both heads' 64 dims side by side, real
so S = Q K^T keeps every (query of head h, key) score inside its own row half, and O = P V is valid where the row half
equals the column's hp (the cross entries are dropped on the way back). col_of(j, hp) = (j5 + 2 j4 + 4 j3) + 8 (j0 + 2 j1 +
4 j2 + 8 hp): the bits stage A leaves on slot bits 12..14 stay put.

Conversions: sinc.stage_a (slot fields [0, 6) <-> [6, 12)) then a routed stage B: per (source v, target ciphertext o)
one constant shift, factored as a rotation per source and one per target (masks from the explicit maps, numpy only)."""
from __future__ import annotations

import functools

import numpy as np

from . import sinc as sc

HB = 8                 # heads per 512-channel block
DH = 64
SLOTS, NCT, OFFSET = sc.SLOTS, sc.NCT, sc.OFFSET


def col_of(j, hp):
    j, hp = np.asarray(j, dtype=np.int64), np.asarray(hp, dtype=np.int64)
    lo = ((j >> 5) & 1) | (((j >> 4) & 1) << 1) | (((j >> 3) & 1) << 2)
    return lo | ((j & 7) << 3) | (hp << 6)


def dim_of(col):
    """(j, hp) of a SinC column."""
    col = np.asarray(col, dtype=np.int64)
    j = ((col >> 3) & 7) | (((col >> 2) & 1) << 3) | (((col >> 1) & 1) << 4) | ((col & 1) << 5)
    return j, (col >> 6) & 1


def batch_of(h):
    return 2 * (np.asarray(h) // 2)


def _q_entries(n_blocks):
    """(source block k, stage-A slot, target ct, target slot) of every Q entry (h, lane, j)."""
    h, lane, j = (a.ravel() for a in np.meshgrid(np.arange(HB * n_blocks), np.arange(64), np.arange(DH),
                                                 indexing="ij"))
    src = j * 512 + (h % HB) * 64 + lane
    ct, sl = sc.sinc_slot(sc.brev(lane, 6) + 64 * (h % 2), col_of(j, h % 2), batch_of(h))
    return h // HB, sc.swap_fields(src), ct, sl


def _kv_entries(n_blocks):
    """(source (block k, chunk c) index 2 k + c, stage-A slot, target ct, target slot) of every K / V entry."""
    h, tok, j = (a.ravel() for a in np.meshgrid(np.arange(HB * n_blocks), np.arange(128), np.arange(DH),
                                                indexing="ij"))
    lane, c = tok % 64, tok // 64
    src = j * 512 + (h % HB) * 64 + lane
    ct, sl = sc.sinc_slot(sc.brev(lane, 6) + 64 * c, col_of(j, h % 2), batch_of(h))
    return 2 * (h // HB) + c, sc.swap_fields(src), ct, sl


def _route(v, s, o, t, n_src):
    """Factor slot s of source v -> slot t of target o as rotate(x_v, r_v), mask, sum, rotate(., q_o):
    t = s - r_v - q_o. Returns (r per source, q per target, masks {(o, v): vector})."""
    d = (t - s) % SLOTS
    r = np.zeros(n_src, dtype=np.int64)
    q = np.zeros(NCT, dtype=np.int64)
    base = {}
    for vv in range(n_src):
        for oo in range(NCT):
            sel = (v == vv) & (o == oo)
            if not sel.any():
                continue
            u = np.unique(d[sel])
            if len(u) != 1:
                raise AssertionError(f"route ({vv}, {oo}): {len(u)} shifts")
            base[(vv, oo)] = int(u[0])
    v0 = min(vv for vv, _ in base)
    for oo in range(NCT):
        if (v0, oo) in base:
            q[oo] = (-base[(v0, oo)]) % SLOTS
    for (vv, oo), dd in base.items():
        rv = (-dd - q[oo]) % SLOTS
        if vv == v0:
            assert rv == 0
        r[vv] = rv
    for (vv, oo), dd in base.items():
        assert (-r[vv] - q[oo]) % SLOTS == dd, "shift not separable per source and target"
    masks = {}
    for (vv, oo) in base:
        sel = (v == vv) & (o == oo)
        m = np.zeros(SLOTS)
        m[(s[sel] - r[vv]) % SLOTS] = 1.0
        masks[(oo, vv)] = m
    return r, q, masks


@functools.lru_cache(maxsize=None)
def q_route(n_blocks=2):
    k, s, o, t = _q_entries(n_blocks)
    return _route(k, s, o, t, n_blocks)


@functools.lru_cache(maxsize=None)
def kv_route(n_blocks=2):
    v, s, o, t = _kv_entries(n_blocks)
    return _route(v, s, o, t, 2 * n_blocks)


@functools.lru_cache(maxsize=None)
def q_inv_route(n_blocks=2):
    """The way back for O (Q's layout): SinC ct o, slot t -> block k, stage-A slot s (valid entries only)."""
    k, s, o, t = _q_entries(n_blocks)
    r, q, m = _route(o, t, k, s, NCT)
    return r, q, m


def _routed(rt, xs, route, n_out, site):
    ops = rt.ops
    r, q, masks = route
    ys = [x if r[v] == 0 else ops.rotate(x, int(r[v])) for v, x in enumerate(xs)]
    outs = []
    for o in range(n_out):
        vs = [v for v in range(len(xs)) if (o, v) in masks]
        if not vs:
            outs.append(None)
            continue
        acc = rt.mult_mask_sum([ys[v] for v in vs], [(site, o, v) for v in vs],
                               [lambda o=o, v=v: masks[(o, v)] for v in vs])
        outs.append(ops.rotate(acc, int(q[o])) if q[o] else acc)
    return outs


def to_sinc_q(rt, Q):
    """Blocked Q (one complex ciphertext per 512-channel block) -> the 16 SinC ciphertexts of M = Q."""
    return _routed(rt, [sc.stage_a(rt, x) for x in Q], q_route(len(Q)), NCT, "smha.q")


def to_sinc_kv(rt, X):
    """Per-chunk real K or V, X[k][c] -> the 16 SinC ciphertexts of (rows = keys, cols = both heads' dims)."""
    xs = [sc.stage_a(rt, x) for chunks in X for x in chunks]
    return _routed(rt, xs, kv_route(len(X)), NCT, "smha.kv")


def from_sinc_q(rt, Y, n_blocks=2):
    """O in Q's SinC layout (16 ciphertexts) -> n_blocks blocked complex ciphertexts (the valid diagonal only)."""
    zs = _routed(rt, list(Y), q_inv_route(n_blocks), n_blocks, "smha.qi")
    return [sc.stage_a(rt, z, site="sinc.ai") for z in zs]


def rot_steps(n_blocks=2) -> set[int]:
    st = set()
    for r, q, _ in (q_route(n_blocks), kv_route(n_blocks), q_inv_route(n_blocks)):
        st |= {int(x) for x in r if x} | {int(x) for x in q if x}
    return st | sc.rot_steps()


def packed_meta(T: int, h_real: int = 12):
    """The packed real score layout (sinc_softmax.pack: Re on batch l < 16, Im on l + 16): per slot the query token
    (-1: none) and whether the slot is a real head's row; per (ciphertext, slot) the key token."""
    row, col_lo, l = sc.slot_coords()
    chunk, lb = l >> 4, l & 15
    h = lb + (row >> 6)
    head = (lb % 2 == 0) & (h < h_real)
    tq = 64 * chunk + sc.brev(row & 63, 6)
    tq = np.where(head & (tq < T), tq, -1)
    col = 8 * np.arange(NCT)[:, None] + col_lo[None, :]
    tk = 64 * (col >> 6) + sc.brev(col & 63, 6)
    return tq, tk
