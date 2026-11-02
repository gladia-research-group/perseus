"""SinC slot layout of the CCMM attention (CKL batch CCMM, ePrint 2025/1957) and its maps to the blocked GQA prefill
layout (blocked.py), numpy only.

SinC_{1024, 2^16} after bts.stc_high(skip_mask=1): entry (row i, col c, batch l) of a 128 x 128 matrix over R_64 sits in
ciphertext c // 8 at slot (32 brev10(c % 8 + 8 i) + l - 30) mod 2^15.

Blocked source (Llama, nch 512, t 64, 4 heads per block, G 4): block q, slot j 256 + m 64 + lane, Re / Im = chunk 0 / 1
(token 64 c + lane); query head h = 4 kv + u', kv = 4 (q % 2) + m, u' = q // 2.
CCMM batch: l = kv + 8 (u' >> 1) (16 of 32 slots); rows of Q = brev6(lane) + 64 (u' & 1) with Re / Im = chunk;
K and V enter as (rows = keys, cols = dims): row = brev6(lane) + 64 c, real; cols = dims in COL_OF_DIM order.
Q to SinC = stage A (swap slot bit fields [0, 6) <-> [6, 12), within each ciphertext) then stage B (slot bits 2..5 <->
the ciphertext index, one rotation per (source, target) pair, the -30 folded in)."""
from __future__ import annotations

import numpy as np

SLOTS = 1 << 15
D = 128
NCT = 16
BATCH = 32
OFFSET = -30


def brev(x, bits):
    x = np.asarray(x, dtype=np.int64)
    out = np.zeros_like(x)
    for b in range(bits):
        out |= ((x >> b) & 1) << (bits - 1 - b)
    return out


BR10 = brev(np.arange(1024), 10)


def sinc_slot(row, col, l):
    """(row, col, batch) -> (ciphertext, slot)."""
    row, col, l = (np.asarray(v, dtype=np.int64) for v in (row, col, l))
    return col // 8, (32 * BR10[col % 8 + 8 * row] + l + OFFSET) % SLOTS


def col_of_dim(j):
    """Dim j -> SinC column: column bits 0..2 = j6, j5, j4 (slot bits 14..12 stay put), bits 3..6 = j0..j3."""
    j = np.asarray(j, dtype=np.int64)
    return ((j >> 6) & 1) | (((j >> 5) & 1) << 1) | (((j >> 4) & 1) << 2) | ((j & 15) << 3)


def batch_of(kv, u1):
    return np.asarray(kv) + 8 * np.asarray(u1)


def q_source(h, tok, j):
    """Query head h, token row tok = 64 c + lane, dim j -> (block, slot, chunk)."""
    h, tok, j = (np.asarray(v, dtype=np.int64) for v in (h, tok, j))
    kv, up = h // 4, h % 4
    q = 2 * up + kv // 4
    m = kv % 4
    return q, j * 256 + m * 64 + tok % 64, tok // 64


def q_target(h, tok, j):
    """The same entry in the SinC operand M = Q: (ciphertext, slot); Re / Im = chunk."""
    h, tok, j = (np.asarray(v, dtype=np.int64) for v in (h, tok, j))
    kv, up = h // 4, h % 4
    row = brev(tok % 64, 6) + 64 * (up & 1)
    return sinc_slot(row, col_of_dim(j), batch_of(kv, up >> 1))


def kv_source(kv, tok, j):
    """KV head kv, token tok, dim j -> (block b, slot, chunk) in the per-chunk real K / V ciphertexts."""
    kv, tok, j = (np.asarray(v, dtype=np.int64) for v in (kv, tok, j))
    return kv // 4, j * 256 + (kv % 4) * 64 + tok % 64, tok // 64


def kv_target(kv, tok, j, u1):
    """K or V as (rows = keys, cols = dims), replica u1 of KV head kv: (ciphertext, slot), real."""
    kv, tok, j = (np.asarray(v, dtype=np.int64) for v in (kv, tok, j))
    row = brev(tok % 64, 6) + 64 * (tok // 64)
    return sinc_slot(row, col_of_dim(j), batch_of(kv, u1))


def swap_fields(s):
    """Stage A: slot bit fields [0, 6) <-> [6, 12)."""
    s = np.asarray(s, dtype=np.int64)
    return (s & ~0xFFF) | ((s & 63) << 6) | ((s >> 6) & 63)


def stage_a_shifts():
    """Distinct rotation amounts of stage A (target - source, mod 2^15): 63 d, d in [-63, 63]."""
    return sorted({(63 * d) % SLOTS for d in range(-63, 64)})


def stage_b(src_ct_bits, s):
    """Stage B on a stage-A slot s of a source whose ciphertext-index bits (placed on slot bits 2..5) are
    src_ct_bits: -> (target ciphertext = old slot bits 2..5, target slot with the offset)."""
    s = np.asarray(s, dtype=np.int64)
    tgt_ct = (s >> 2) & 15
    t = (s & ~(15 << 2)) | (np.asarray(src_ct_bits, dtype=np.int64) << 2)
    return tgt_ct, (t + OFFSET) % SLOTS


def q_ct_bits(q):
    """Block q -> the bits stage B writes on slot bits 2..5: (kv2, u1, 0, u0)."""
    q = np.asarray(q, dtype=np.int64)
    kv2, up = q % 2, q // 2
    return kv2 | ((up >> 1) << 1) | ((up & 1) << 3)


def kv_ct_bits(b, chunk, u1):
    """K / V source (block b, chunk) written as replica u1: (kv2, u1, 0, chunk)."""
    return np.asarray(b) | (np.asarray(u1) << 1) | (np.asarray(chunk) << 3)


def stage_b_shift(src_bits, tgt_ct):
    """Rotation amount (target - source slot) of the (source, target ciphertext) pair in stage B."""
    return ((int(src_bits) - int(tgt_ct)) * 4 + OFFSET) % SLOTS


A_BABY = 12
A_GIANT = range(-6, 6)


def _stage_a_masks():
    """mask_{g,b} = roll(m_d, 63 b) for d = 12 g + b, m_d[s] = [swap_fields(s) - s == 63 d]."""
    s = np.arange(SLOTS)
    dd = (swap_fields(s) - s) // 63
    out = {}
    for g in A_GIANT:
        for b in range(A_BABY):
            d = A_BABY * g + b
            if -63 <= d <= 63:
                out[(g, b)] = np.roll((dd == d).astype(float), 63 * b)
    return out


_MA = None


def stage_a(rt, x, site="sinc.a"):
    """Swap slot fields [0, 6) <-> [6, 12) of x (BSGS: 12 hoisted baby rotations, 11 giant)."""
    global _MA
    if A2:
        return stage_a2(rt, x, site)
    if _MA is None:
        _MA = _stage_a_masks()
    ops = rt.ops
    baby = [x] + ops.rotate_many(x, [-63 * b for b in range(1, A_BABY)])
    out = None
    for g in A_GIANT:
        bs = [b for b in range(A_BABY) if (g, b) in _MA]
        if not bs:
            continue
        acc = rt.mult_mask_sum([baby[b] for b in bs], [(site, g, b) for b in bs],
                               [lambda m=_MA[(g, b)]: m for b in bs])
        if g:
            acc = ops.rotate(acc, -63 * A_BABY * g)
        out = acc if out is None else ops.add(out, acc)
    return out


# PF_SINC_A2=1: stage A as two 3-bit field swaps ([0, 3) <-> [6, 9), then [3, 6) <-> [9, 12)): 2 x 15 diagonals and
# one more level instead of 127 diagonals
A2 = __import__("os").environ.get("PF_SINC_A2", "0") not in ("", "0")
A2_BABY = 4
A2_GIANT = range(-2, 2)


def swap3(s, lo):
    """Slot bit fields [lo, lo + 3) <-> [lo + 6, lo + 9)."""
    s = np.asarray(s, dtype=np.int64)
    a, b = (s >> lo) & 7, (s >> (lo + 6)) & 7
    return (s & ~((7 << lo) | (7 << (lo + 6)))) | (b << lo) | (a << (lo + 6))


_MA2 = {}


def _swap3_masks(lo):
    """mask_{g,b} = roll(m_d, u b) for d = 4 g + b, u = 63 2^lo, m_d[s] = [swap3(s) - s == u d]."""
    if lo not in _MA2:
        u = 63 << lo
        s = np.arange(SLOTS)
        dd = (swap3(s, lo) - s) // u
        _MA2[lo] = {(g, b): np.roll((dd == A2_BABY * g + b).astype(float), u * b)
                    for g in A2_GIANT for b in range(A2_BABY) if -7 <= A2_BABY * g + b <= 7}
    return _MA2[lo]


def _swap3(rt, x, lo, site):
    ops, u = rt.ops, 63 << lo
    ms = _swap3_masks(lo)
    baby = [x] + ops.rotate_many(x, [-u * b for b in range(1, A2_BABY)])
    out = None
    for g in A2_GIANT:
        bs = [b for b in range(A2_BABY) if (g, b) in ms]
        if not bs:
            continue
        acc = rt.mult_mask_sum([baby[b] for b in bs], [(site, lo, g, b) for b in bs],
                               [lambda m=ms[(g, b)]: m for b in bs])
        if g:
            acc = ops.rotate(acc, -u * A2_BABY * g)
        out = acc if out is None else ops.add(out, acc)
    return out


def stage_a2(rt, x, site="sinc.a"):
    return _swap3(rt, _swap3(rt, x, 0, site), 3, site)


def _b_mask(o, v):
    """Stage-B mask for target ciphertext o, read off a source pre-rotated by -4 v: slots with bits 2..5 == o."""
    s = np.arange(SLOTS)
    return np.roll((((s >> 2) & 15) == o).astype(float), 4 * v)


def stage_b(rt, xs, vs, n_out=NCT, site="sinc.b"):
    """xs: stage-A outputs, vs: their ciphertext bits for slot bits 2..5 -> n_out ciphertexts (offset included)."""
    ops = rt.ops
    ys = [x if v == 0 else ops.rotate(x, -4 * v) for x, v in zip(xs, vs)]
    outs = []
    for o in range(n_out):
        acc = rt.mult_mask_sum(ys, [(site, o, v) for v in vs], [lambda o=o, v=v: _b_mask(o, v) for v in vs])
        sh = (4 * o - OFFSET) % SLOTS
        outs.append(ops.rotate(acc, sh) if sh else acc)
    return outs


def _bi_mask(v):
    """Inverse stage B: slots whose bits 2..5 == v."""
    s = np.arange(SLOTS)
    return (((s >> 2) & 15) == v).astype(float)


# PF_SINC_BI_HOIST=1: stage_b_inv with the rotations hoisted to the 16 sources and the 8 sums (24 instead of 128)
BI_HOIST = __import__("os").environ.get("PF_SINC_BI_HOIST", "0") not in ("", "0")


def stage_b_inv(rt, ys, vs, site="sinc.bi"):
    """Inverse of stage_b: SinC ciphertexts ys (target o = index) -> one stage-A-layout ciphertext per v in vs."""
    ops = rt.ops
    if BI_HOIST:
        ws = [ops.rotate(y, (OFFSET - 4 * o) % SLOTS) for o, y in enumerate(ys)]
        outs = []
        for v in vs:
            acc = rt.mult_mask_sum(ws, [(site, "h", v, o) for o in range(len(ws))],
                                   [lambda v=v, o=o: np.roll(_bi_mask(v), 4 * o) for o in range(len(ws))])
            outs.append(ops.rotate(acc, 4 * v) if v else acc)
        return outs
    zs = [ops.rotate(y, OFFSET) for y in ys]
    outs = []
    for v in vs:
        acc = None
        for o, z in enumerate(zs):
            t = rt.mult_mask(z, (site, v), lambda v=v: _bi_mask(v))
            t = ops.rotate(t, 4 * (v - o)) if v != o else t
            acc = t if acc is None else ops.add(acc, t)
        outs.append(acc)
    return outs


def to_sinc_q(rt, Q):
    """Blocked Q (8 complex block ciphertexts, Re / Im = chunks) -> the 16 SinC ciphertexts of M = Q."""
    return stage_b(rt, [stage_a(rt, x) for x in Q], [int(q_ct_bits(q)) for q in range(len(Q))])


def to_sinc_kv(rt, X):
    """Per-chunk real K or V, X[b][c] (2 blocks x 2 chunks) -> the 16 SinC ciphertexts of (rows = keys, cols = dims),
    each KV head on both replicas u1."""
    xs, vs = [], []
    for b, chunks in enumerate(X):
        for c, x in enumerate(chunks):
            a = stage_a(rt, x)
            for u1 in (0, 1):
                xs.append(a)
                vs.append(int(kv_ct_bits(b, c, u1)))
    return stage_b(rt, xs, vs)


def dup(rt, xs):
    """Batch l < 16 copied onto l + 16 (the operands carry nothing on l >= 16): no level."""
    ops = rt.ops
    return [ops.add(x, ops.rotate(x, -16)) for x in xs]


def stage_b_inv_merged(rt, Ms, wl, vs, site="sinc.bim"):
    """stage_b_inv (hoisted) of the unmerged pairs of the merged CCMM outputs Ms ((e + w o) / 2, wl: w per SinC slot):
    the unmerge's halves ride the stage's masks (no extra level)."""
    ops = rt.ops
    lo = (((np.arange(SLOTS) - OFFSET) % 32) < 16).astype(float)
    us = []
    for p, m in enumerate(Ms):
        for o in (2 * p, 2 * p + 1):
            sh = (OFFSET - 4 * o) % SLOTS
            a, b = ops.rotate(m, sh), ops.rotate(m, (sh + 16) % SLOTS)
            us.append(ops.add(a, b) if o % 2 == 0 else ops.sub(a, b))
    um = [lo, lo / np.asarray(wl)]
    outs = []
    for v in vs:
        acc = rt.mult_mask_sum(us, [(site, v, o) for o in range(len(us))],
                               [lambda v=v, o=o: np.roll(_bi_mask(v), 4 * o) * np.roll(um[o % 2], (4 * o - OFFSET) % SLOTS)
                                for o in range(len(us))])
        outs.append(ops.rotate(acc, 4 * v) if v else acc)
    return outs


def from_sinc_q_merged(rt, Ms, wl):
    """from_sinc_q of the 16 unmerged outputs of the 8 merged CCMM outputs Ms."""
    vs = [int(q_ct_bits(q)) for q in range(8)]
    return [stage_a(rt, z, site="sinc.ai") for z in stage_b_inv_merged(rt, Ms, wl, vs)]


def from_sinc_q(rt, Y):
    """Inverse of to_sinc_q (the attention output O, in Q's SinC layout) -> 8 blocked ciphertexts."""
    vs = [int(q_ct_bits(q)) for q in range(8)]
    return [stage_a(rt, z, site="sinc.ai") for z in stage_b_inv(rt, Y, vs)]


def rot_steps() -> set[int]:
    """Rotation keys of the conversions."""
    st = {-63 * b for b in range(1, A_BABY)} | {-63 * A_BABY * g for g in A_GIANT if g}
    st |= {-4 * v for v in range(1, 16)} | {4 * v for v in range(1, 16)} | {4 * v - 4 * o for v in range(16) for o in range(16)}
    st |= {(4 * o - OFFSET) % SLOTS for o in range(16)} | {OFFSET}
    if BI_HOIST:
        st |= {(OFFSET - 4 * o) % SLOTS for o in range(16)} | {4 * v for v in range(1, 16)}
    if __import__("os").environ.get("PF_CCMM_MERGE", "0") not in ("", "0"):
        st |= {(OFFSET - 4 * o + 16) % SLOTS for o in range(16)} | {(OFFSET - 4 * o) % SLOTS for o in range(16)}
        st |= {16, SLOTS - 16} | {4 * v for v in range(1, 16)}
    if A2:
        for lo in (0, 3):
            u = 63 << lo
            st |= {-u * b for b in range(1, A2_BABY)} | {-u * A2_BABY * g for g in A2_GIANT if g}
    return {s for s in st if s % SLOTS}


def slot_coords():
    """Per slot of a SinC ciphertext: (row, col_lo = col % 8, batch l)."""
    j = (np.arange(SLOTS) - OFFSET) % SLOTS
    blk = BR10[j >> 5]
    return blk >> 3, blk & 7, j & 31


BOS_TOK = 127


def packed_meta(T=127):
    """The softmax's packed real score layout (Re half on batch l < 16, Im half moved to l + 16): per slot the query
    token (-1: none), and per (ciphertext o, slot) the key token (BOS_TOK = the BOS key); batch l -> kv = l % 8."""
    row, col_lo, l = slot_coords()
    chunk = l >> 4
    tq = 64 * chunk + brev(row & 63, 6)
    tq = np.where(tq < T, tq, -1)
    col = 8 * np.arange(NCT)[:, None] + col_lo[None, :]
    tk = 64 * (col >> 6) + brev(col & 63, 6)
    return tq, tk


def dim_of_col(col):
    col = np.asarray(col, dtype=np.int64)
    return ((col >> 3) & 15) | (((col >> 2) & 1) << 4) | (((col >> 1) & 1) << 5) | ((col & 1) << 6)


def kv_rows_pattern(vals, toks):
    """kv_row_pattern summed over key rows `toks`."""
    return sum(kv_row_pattern(vals, t) for t in toks)


def kv_row_pattern(vals, tok=BOS_TOK):
    """Slot vectors (16 ciphertexts) holding vals[kv][dim] on key row `tok` of a (rows = keys, cols = dims) SinC
    operand, both replicas."""
    out = np.zeros((NCT, SLOTS))
    col = np.arange(D)
    for l in range(16):
        ct, sl = sinc_slot(np.full(D, brev(tok % 64, 6) + 64 * (tok // 64)), col, np.full(D, l))
        out[ct, sl] = np.asarray(vals)[l % 8][dim_of_col(col)]
    return out
