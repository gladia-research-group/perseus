"""The SANA block's own kernels on the blocked layout (layout.py), written on the leaf ops (perseus.impl.ops): they run
unchanged on the plaintext mirror (NumpyOps), the numpy fake and the device. Masks go through rt.mult_mask / add_mask
(memoised, keyed per site)."""
from __future__ import annotations


import os
import zlib

import numpy as np

from perseus.impl.prefill import join, split

from .layout import G, IMG, LANES, N, NCH, TIGHT, TOK, VALID, col_of_lane, lane_vec, slot_of


# the deepest nominal level a value may reach before its refresh: the StC-first dense refresh's input cap
# (PLAN_STC_FIRST_ENTRY)
GUARD = 46


def center_keys(rt, Kb, site):
    """Kb[j][k] less the keys' mean per block k (ciphertexts or Pt; the result is ciphertexts): one masked product.
    The cross-attention's scores are then centered per query (exact: the softmax is shift-invariant), so exp's domain
    is the centered one (~10x narrower: fewer squaring rounds)."""
    ops = rt.ops
    n, nblk = len(Kb), len(Kb[0])
    out = [[None] * nblk for _ in range(n)]
    for k in range(nblk):
        cts = [Kb[j][k] for j in range(n) if not isinstance(Kb[j][k], Pt)]
        acc = cts[0]
        for c in cts[1:]:
            acc = ops.add(acc, c)
        for j in range(n):
            x = Kb[j][k]
            if isinstance(x, Pt):
                acc = rt.add_mask(acc, (site, "kc.pt") + tuple(x.tag), lambda x=x: x.v)
        mean = rt.mult_mask(acc, (site, "kc.mean", n), lambda: np.full(N, 1.0 / n))
        for j in range(n):
            x = Kb[j][k]
            out[j][k] = (rt.add_mask(ops.negate(mean), (site, "kc.pt0") + tuple(x.tag), lambda x=x: x.v)
                         if isinstance(x, Pt) else ops.sub(x, mean))
    return out


# PF_XA_KEY_FOLD=1: the cross-attention key rows keep their RMS scale (max ~1) and the scores' 1 / (sqrt(DX) M) rides
# the score window's mask instead of the keys. The keys' refresh site then sees values of order 1, not 1e-3 (which
# the refresh's fixed absolute error turned into ~2.5% relative), and their scale mask is gone (one level)
XA_KEY_FOLD = os.environ.get("PF_XA_KEY_FOLD", "0") not in ("", "0")


# SANA_KV_HINT_EARLY=1: the text keys' / values' refresh sites sit on the 4 + 4 ciphertexts that hold every prompt
# token (one more product of room: the token mask), not on each token's broadcast row (n_tok x 8 sites)
KV_HINT_EARLY = os.environ.get("SANA_KV_HINT_EARLY", "0") not in ("", "0")


# SANA_PREDIV_FOLD=1: a pre-divided linear (stamp w_scale 2^shift above 2^20) costs its inputs a level: they must come
# in one level above the StC-first entry so the runtime can lower their scale by the stamp over 2^20. The same is had
# for free by feeding it x / 2^s (s = log2 stamp - 20) from a plaintext product its producer already has, and reading
# the same integer planes at w_scale / 2^s: the stamp is then 2^20, nothing is pre-divided, and the input may sit at
# the entry itself. For the linears named here (their producers: the reciprocals' unpack mask, the gate's series)
PREDIV_FOLD = tuple(x for x in os.environ.get("SANA_PREDIV_FOLD", "").split(",") if x and x != "0")


def xa_scale(nl, site):
    """M of a cross-attention site: what the keys are divided by, the query's RMS scale multiplied by."""
    return 1.0 / nl.stat_scale(site + ".qnorm")


def room_hint(rt, x, k):
    """A refresh site on x where fewer than k products (2 levels each) are left below GUARD."""
    return rt.ops.bootstrap_hint(x, GUARD - 2 * int(k), True)


# ── packed scalar maps ──────────────────────────────────────────────────────────────────────────────────────────────
#
# A per-token statistic (on every channel of its lane) or a per-head value (on every channel of the head) fills a
# ciphertext with copies. Up to R of them share one evaluation of the scalar map: input i keeps the channels of
# class i (channel % R == i for a lane statistic, channel % P == i within a P-channel group for a head value), the
# map runs once on the sum (unused classes hold `fill`, inside the map's band), and output i is its class spread back
# (norm_blocked's regions).

def packed_lane_map(rt, xs, fn, site, R=8, fill=1.0, lanes=None):
    """[fn(x) for x in xs]: real ciphertexts each constant over the channels of a lane; `lanes` ((LANES,) per x): the
    outputs times their lane pattern (taken into the class masks: no level)."""
    lps = [np.ones(LANES)] * len(xs) if lanes is None else [np.asarray(l, dtype=np.float64) for l in lanes]
    ops = rt.ops
    out = []
    for i0 in range(0, len(xs), R):
        grp = xs[i0:i0 + R]
        acc = None
        for i, x in enumerate(grp):
            m = rt.mult_mask(x, (site, "plm.in", R, i), lambda i=i: lane_vec(np.ones(LANES), np.arange(NCH) % R == i))
            acc = m if acc is None else ops.add(acc, m)
        if len(grp) < R:                      # the unused classes in the map's band
            acc = rt.add_mask(acc, (site, "plm.fill", R, len(grp), fill), lambda n=len(grp): lane_vec(
                np.ones(LANES), np.where(np.arange(NCH) % R >= n, fill, 0.0)))
        y = fn(acc)
        for i in range(len(grp)):
            lp = lps[i0 + i]
            m = rt.mult_mask(y, (site, "plm.out", R, i) + (() if lanes is None else ("lanes", lp.tobytes())),
                             lambda i=i, lp=lp: lane_vec(lp, np.arange(NCH) % R == i))
            out.append(ops.rotate_and_sum(m, LANES, R * LANES) if R > 1 else m)
    return out


def packed_group_map(rt, xs, fn, site, P=32, fill=1.0, out_scale=1.0):
    """[fn(x) for x in xs]: real ciphertexts each constant over every aligned group of P channels; `out_scale`: the
    outputs times it (on the unpack masks: no level)."""
    ops = rt.ops
    out = []
    for i0 in range(0, len(xs), P):
        grp = xs[i0:i0 + P]
        acc = None
        for i, x in enumerate(grp):
            m = rt.mult_mask(x, (site, "pgm.in", P, i), lambda i=i: lane_vec(np.ones(LANES), np.arange(NCH) % P == i))
            acc = m if acc is None else ops.add(acc, m)
        if len(grp) < P:                      # the unused classes in the map's band
            acc = rt.add_mask(acc, (site, "pgm.fill", P, len(grp), fill), lambda n=len(grp): lane_vec(
                np.ones(LANES), np.where(np.arange(NCH) % P >= n, fill, 0.0)))
        y = fn(acc)
        for i in range(len(grp)):
            m = rt.mult_mask(y, (site, "pgm.out", P, i) + (() if out_scale == 1.0 else (out_scale,)),
                             lambda i=i: lane_vec(np.ones(LANES), out_scale * (np.arange(NCH) % P == i)))
            m = ops.rotate(m, i * LANES) if i else m                  # to the group's first channel
            out.append(ops.rotate_and_sum(m, -LANES, P * LANES))
    return out


# ── 3 x 3 depthwise convolution ─────────────────────────────────────────────────────────────────────────────────────
#
# Token slot p = 128 g + 64 half + lane sits on image row p // 32, column lane % 32 (a token where the column is under
# IMG). A horizontal neighbour is one lane away (masked at the image's edges). A vertical neighbour is 32 lanes away
# in the same half, or crosses to the other half: row 4g + 1 (real, lanes 32-63) continues in row 4g + 2 (imaginary,
# lanes 0-31), row 4g + 3 (imaginary) in row 4g + 4 (ciphertext g + 1, real). So the row below of X_g = A_g + i B_g
# is X_g rotated by 32 lanes on lanes 0-31 and W_g = B_g + i A_{g+1} rotated by -32 lanes on lanes 32-63; the row
# above is X_g rotated by -32 on lanes 32-63 and W_{g-1} rotated by 32 on lanes 0-31 (A_G = B_{-1} = 0: the zero
# padding). Every term is one rotation of X_g or W
# times one plaintext (the 3 x 3 weight of the channel, the column mask and the half-lane mask folded): one level.

def _dw_terms():
    """[(source, lane rotation, dx, lane mask)]: source 'x' (X_g), 'w_below' (W_g), 'w_above' (W_{g-1})."""
    lo = (np.arange(LANES) < 32).astype(float)
    hi = 1.0 - lo
    out = []
    for dx in (-1, 0, 1):
        out.append(("x", dx, 1, dx, np.ones(LANES)))                 # same row: dy = 0 -> kernel row 1
        out.append(("x", 32 + dx, 2, dx, lo))                          # row below, same half
        out.append(("w_below", dx - 32, 2, dx, hi))                    # row below, next half
        out.append(("x", dx - 32, 0, dx, hi))                          # row above, same half
        out.append(("w_above", 32 + dx, 0, dx, lo))                    # row above, previous half
    return out


def _dw_mask(w, bias_free, k, ky, dx, lanes_m):
    """The plaintext of one term on channel block k: weight w[c, ky, dx + 1] times the column-validity and lane masks,
    evaluated at the source lane's destination (the mask is on the output lane): a token reads its neighbours inside
    the image only (an empty slot, outside it, reads its own row's as if the image went on)."""
    col = col_of_lane()
    valid = ((col + dx >= 0) & ((col + dx < IMG) | (col >= IMG)) & (col + dx < 32)).astype(float) * lanes_m
    wc = np.zeros(NCH)
    c0 = k * NCH
    n = max(0, min(NCH, w.shape[0] - c0))
    wc[:n] = w[c0:c0 + n, ky, dx + 1]
    return lane_vec(valid, wc)


def depthwise3x3(rt, U, w, bias, site):
    """U[g][k] (channels k NCH + r, real-valued features in both halves) -> the zero-padded 3 x 3 depthwise
    convolution with per-channel filters w (C, 3, 3) plus bias (C,)."""
    if TIGHT:
        return _depthwise3x3_tight(rt, U, w, bias, site)
    ops = rt.ops
    nblk = len(U[0])
    # Wv[j] = B_{j-1} + i A_j for j = 0 .. G (W_g = Wv[g + 1]; B_{-1} = A_G = 0). split gives (2A, 2B): the masks
    # carry the 1/2
    halves = [[split(ops, x) for x in row] for row in U]
    Wv = []
    for j in range(G + 1):
        row = []
        for k in range(nblk):
            b2 = halves[j - 1][k][1] if j > 0 else None
            a2 = halves[j][k][0] if j < G else None
            row.append(ops.mult_i(a2) if b2 is None else join(ops, b2, a2))
        Wv.append(row)
    out = []
    terms = _dw_terms()
    for g in range(G):
        row = []
        for k in range(nblk):
            acc = None
            for src, rot, ky, dx, lanes_m in terms:
                if src == "x":
                    x, sc = U[g][k], 1.0
                elif src == "w_below":
                    x, sc = Wv[g + 1][k], 0.5
                else:
                    x, sc = Wv[g][k], 0.5
                xr = ops.rotate(x, rot) if rot else x
                m = rt.mult_mask(xr, (site, "dw", k, src, rot, ky, dx),
                                 lambda k=k, ky=ky, dx=dx, lm=lanes_m, sc=sc: sc * _dw_mask(w, None, k, ky, dx, lm))
                acc = m if acc is None else ops.add(acc, m)
            bc = np.zeros(NCH)
            c0 = k * NCH
            n = max(0, min(NCH, len(bias) - c0))
            bc[:n] = bias[c0:c0 + n]
            if np.any(bc):
                acc = rt.add_mask(acc, (site, "dw.bias", k),
                                  lambda bc=bc: lane_vec(np.ones(LANES), bc) * (1.0 + 1.0j))
            row.append(acc)
        out.append(row)
    return out


# ── the same on tight rows (SANA_TIGHT_ROWS=1) ──────────────────────────────────────────────────────────────────────
#
# Token slot p = 128 g + 64 half + lane sits on image row p // IMG, column p % IMG (a token where p < IMG^2). The
# neighbour (dy, dx) is d = IMG dy + dx slots away: lane + d of the same half where that is a lane, else of the next
# half (lane + d - 64) or the previous one (lane + d + 64). The half after the real part of X_g = A_g + i B_g is its
# imaginary part and the half after that the real part of X_{g+1}, so the next halves of both are W_g = B_g + i A_{g+1}
# and the previous ones W_{g-1} (A_G = B_{-1} = 0: the zero padding above the image; below it the masks). Every term is
# one rotation of X_g or W times one plaintext (the 3 x 3 weight of the channel and the masks folded): one level.
# A token reads a neighbour inside the image only (its column + dx on the row, the slot + d a token), and the two
# halves sit at different image columns unless 64 is a multiple of IMG, so the mask m_re of the real part differs from
# the imaginary part's m_im: m_re Re(t) + i m_im Im(t) = t (m_re + m_im) / 2 + conj(t) (m_re - m_im) / 2, the second
# product on the rotation of the source's conjugate (skipped where the masks agree).

def _dwt_terms():
    """[(source, lane rotation, kernel row, dx)]: source 'x' (X_g), 'w_below' (W_g), 'w_above' (W_{g-1})."""
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            d = IMG * dy + dx
            out.append(("x", d, dy + 1, dx))
            if d > 0:
                out.append(("w_below", d - LANES, dy + 1, dx))
            if d < 0:
                out.append(("w_above", d + LANES, dy + 1, dx))
    return out


def _dw_lanes(g, src, rot, dx):
    """(m_re, m_im): per lane of chunk pair g's halves, 1 where the term's source lane is this neighbour and a token
    may read it (a slot past the image reads the slots around it as if they were its rows)."""
    d = rot + (LANES if src == "w_below" else -LANES if src == "w_above" else 0)
    lane = np.arange(LANES)
    here = {"x": (lane + d >= 0) & (lane + d < LANES), "w_below": lane + d >= LANES, "w_above": lane + d < 0}[src]
    out = []
    for half in (0, 1):
        p = slot_of(g, half)
        col = p % IMG
        end = np.where(p < IMG * IMG, IMG * IMG, TOK)
        out.append((here & (col + dx >= 0) & (col + dx < IMG) & (p + d >= 0) & (p + d < end)).astype(float))
    return out


def _dw_w(w, k, ky, dx):
    wc = np.zeros(NCH)
    c0 = k * NCH
    n = max(0, min(NCH, w.shape[0] - c0))
    wc[:n] = w[c0:c0 + n, ky, dx + 1]
    return wc


def _depthwise3x3_tight(rt, U, w, bias, site):
    """U[g][k] (channels k NCH + r, real-valued features in both halves) -> the zero-padded 3 x 3 depthwise
    convolution with per-channel filters w (C, 3, 3) plus bias (C,)."""
    ops = rt.ops
    nblk = len(U[0])
    # Wv[j] = B_{j-1} + i A_j for j = 0 .. G (W_g = Wv[g + 1]; B_{-1} = A_G = 0). split gives (2A, 2B): the masks
    # carry the 1/2
    halves = [[split(ops, x) for x in row] for row in U]
    Wv = []
    for j in range(G + 1):
        row = []
        for k in range(nblk):
            b2 = halves[j - 1][k][1] if j > 0 else None
            a2 = halves[j][k][0] if j < G else None
            row.append(ops.mult_i(a2) if b2 is None else join(ops, b2, a2))
        Wv.append(row)
    conj = {}

    def conj_of(kind, j, k):
        """2 conj(X_j) or 2 conj(Wv[j]) of block k from the halves (no key switch)."""
        if (kind, j, k) not in conj:
            if kind == "x":
                a2, b2 = halves[j][k]
            else:
                a2 = halves[j - 1][k][1] if j > 0 else None            # Wv[j] = a2 / 2 + i b2 / 2
                b2 = halves[j][k][0] if j < G else None
            conj[(kind, j, k)] = (ops.negate(ops.mult_i(b2)) if a2 is None else a2 if b2 is None
                                  else ops.sub(a2, ops.mult_i(b2)))
        return conj[(kind, j, k)]
    out = []
    terms = _dwt_terms()
    for g in range(G):
        row = []
        for k in range(nblk):
            acc = None
            if DW_HOIST:                     # each source's rotations back to back (a grid: one ModUp per source)
                plan = []
                for src, rot, ky, dx in terms:
                    m_re, m_im = _dw_lanes(g, src, rot, dx)
                    if src == "x":
                        x, sc, kind, j = U[g][k], 1.0, "x", g
                    elif src == "w_below":
                        x, sc, kind, j = Wv[g + 1][k], 0.5, "w", g + 1
                    else:
                        x, sc, kind, j = Wv[g][k], 0.5, "w", g
                    for part, lm in (("s", 0.5 * (m_re + m_im)), ("a", 0.5 * (m_re - m_im))):
                        if np.any(lm):
                            y, s2 = (x, sc) if part == "s" else (conj_of(kind, j, k), 0.5)
                            plan.append((y, rot, src, ky, dx, part, lm, s2))
                rots, order = {}, []
                for y, rot, *_ in plan:
                    if id(y) not in rots:
                        rots[id(y)] = (y, [])
                        order.append(id(y))
                    if rot and rot not in rots[id(y)][1]:
                        rots[id(y)][1].append(rot)
                done = {}
                for i in order:
                    y, rl = rots[i]
                    for r, yr in zip(rl, ops.rotate_many(y, rl) if rl else []):
                        done[(i, r)] = yr
                for y, rot, src, ky, dx, part, lm, s2 in plan:
                    yr = done[(id(y), rot)] if rot else y
                    m = rt.mult_mask(yr, (site, "dw", k, src, rot, ky, dx, part, lm.tobytes()),
                                     lambda k=k, ky=ky, dx=dx, lm=lm, s2=s2: s2 * lane_vec(lm, _dw_w(w, k, ky, dx)))
                    acc = m if acc is None else ops.add(acc, m)
                terms_here = ()
            else:
                terms_here = terms
            for src, rot, ky, dx in terms_here:
                m_re, m_im = _dw_lanes(g, src, rot, dx)
                if src == "x":
                    x, sc, kind, j = U[g][k], 1.0, "x", g
                elif src == "w_below":
                    x, sc, kind, j = Wv[g + 1][k], 0.5, "w", g + 1
                else:
                    x, sc, kind, j = Wv[g][k], 0.5, "w", g
                for part, lm in (("s", 0.5 * (m_re + m_im)), ("a", 0.5 * (m_re - m_im))):
                    if not np.any(lm):
                        continue
                    y, s2 = (x, sc) if part == "s" else (conj_of(kind, j, k), 0.5)
                    yr = ops.rotate(y, rot) if rot else y
                    m = rt.mult_mask(yr, (site, "dw", k, src, rot, ky, dx, part, lm.tobytes()),
                                     lambda k=k, ky=ky, dx=dx, lm=lm, s2=s2: s2 * lane_vec(lm, _dw_w(w, k, ky, dx)))
                    acc = m if acc is None else ops.add(acc, m)
            bc = np.zeros(NCH)
            c0 = k * NCH
            n = max(0, min(NCH, len(bias) - c0))
            bc[:n] = bias[c0:c0 + n]
            if np.any(bc):
                acc = rt.add_mask(acc, (site, "dw.bias", k),
                                  lambda bc=bc: lane_vec(np.ones(LANES), bc) * (1.0 + 1.0j))
            row.append(acc)
        out.append(row)
    return out


def dw_rotations():
    """The lane rotations of the depthwise convolution on the layout in use."""
    return {t[1] for t in (_dwt_terms() if TIGHT else _dw_terms())}


# ── ReLU linear attention (attn1) ───────────────────────────────────────────────────────────────────────────────────
#
# Heads of DH = 32 channels, 16 per block (block 2: heads 32-35 on channels 0-127). With f = ReLU and the key's
# per-token RMS scale rk folded into f(k), out_i = sum_a f(q_i)_a S[a, :] / sum_a f(q_i)_a z_a per head, where
# S[a, b] = sum_j f(k_j)_a v_j,b and z_a = sum_j f(k_j)_a. S is taken by diagonals: S_d[a] = S[a, a + d] (in-head
# indices mod 32) = the lane sum of f(K) . V rotated by d channels in-head; the token sum over both halves rides one
# product, Re(f(K) conj(V_d)) = f(K)_A V_A + f(K)_B V_B. Then num[b] = sum_d (f(q) S_d)[b - d].

DH = 32


def _ih_masks(shift):
    """(lo, hi) channel masks of an in-head rotation by `shift` channels to the left: lo where r % DH < DH - shift
    (a plain rotation by shift), hi elsewhere (rotation by shift - DH)."""
    r = np.arange(NCH) % DH
    lo = (r < DH - shift).astype(float)
    return lo, 1.0 - lo


def rotate_in_head(rt, x, shift, site):
    """y[r] = x[head(r) + (r + shift) % DH] on every lane: two rotations, one level."""
    ops = rt.ops
    shift %= DH
    if shift == 0:
        return x
    lo, hi = _ih_masks(shift)
    a = rt.mult_mask(ops.rotate(x, shift * LANES), (site, "ih.lo", shift), lambda: lane_vec(np.ones(LANES), lo))
    b = rt.mult_mask(ops.rotate(x, (shift - DH) * LANES), (site, "ih.hi", shift),
                     lambda: lane_vec(np.ones(LANES), hi))
    return ops.add(a, b)


# SANA_LA_HOIST=1: the S sums' value side conjugated once per ciphertext (not once per diagonal: 465 key switches a
# block) and its 62 in-head rotations issued back to back, a rotation grid the compiler hoists (one ModUp). Exact. Its own
# switch: plans captured before it keep their op stream
LA_HOIST = os.environ.get("SANA_LA_HOIST", "0") not in ("", "0")
# SANA_DW_HOIST=1: the depthwise convolution's rotations of each source issued back to back per output (a rotation grid);
# the masked terms then accumulate in the same order. Exact. Its own switch (op order changes)
DW_HOIST = os.environ.get("SANA_DW_HOIST", "0") not in ("", "0")
# SANA_LA_NUM_SUM=1: la.num as q S_0 + sum_d rot(q) A_d + rot(q) B_d (num_rhs / num_lhs). Measured a LOSS on GPU 0 (11-12
# 09:44, 1-block cut: +465 products at ~112 us each, the compiler's mult-sum fusion did not take the chain; rotations
# -372): kept for reference, off
LA_NUM_SUM = os.environ.get("SANA_LA_NUM_SUM", "0") not in ("", "0")
# SANA_LA_QNORM=1 (needs the configs' la_qnorm: fit.py from stats' la.hq / la.zbar): attn1's den times s~, a series
# of the head's own sum of relu(q), and the reciprocal times s~ after. num / den is unchanged and s~'s error cancels
# (it multiplies both); den s~ is a weighted mean of z over the head, so its spread is z's, about a decade under
# den's (2.4 -> 1.4 decades max / q.001): a floor rho then damps what a raw floor of ~rho / 30 damps, at the
# reciprocal depth of rho. Only the packed den ciphertexts change (one series, two products); the numerators do not.
LA_QNORM = os.environ.get("SANA_LA_QNORM", "0") not in ("", "0")


def s_sums(rt, pairs, site, sigma):
    """[S_d for d < DH]: S_d = lane_total(Re sum_i a_i conj(rot_in_head(v_i, d))) over (a_i, v_i) in `pairs` (the
    per-ciphertext f(K) and V of one channel block)."""
    ops = rt.ops
    acc = [None] * DH
    if LA_HOIST:
        steps = [d * LANES for d in range(1, DH)] + [(d - DH) * LANES for d in range(1, DH)]
        for a, v in pairs:
            vc = ops.conjugate(v)
            rots = ops.rotate_many(vc, steps)
            for d in range(DH):
                if d == 0:
                    vd = vc
                else:
                    lo, hi = _ih_masks(d)
                    vd = ops.add(rt.mult_mask(rots[d - 1], (site + ".v", "ih.lo", d), lambda lo=lo: lane_vec(np.ones(LANES), lo)),
                                 rt.mult_mask(rots[DH - 1 + d - 1], (site + ".v", "ih.hi", d),
                                              lambda hi=hi: lane_vec(np.ones(LANES), hi)))
                p = ops.mult(a, vd)
                acc[d] = p if acc[d] is None else ops.add(acc[d], p)
        return [lane_total(rt, ops.add(x, ops.conjugate(x)), site + ".S", 0.5 / sigma) for x in acc]
    out = []                                 # the op stream of plans captured before SANA_LA_HOIST, unchanged
    for d in range(DH):
        x = None
        for a, v in pairs:
            vd = rotate_in_head(rt, v, d, site + ".v")
            p = ops.mult(a, ops.conjugate(vd))
            x = p if x is None else ops.add(x, p)
        out.append(lane_total(rt, ops.add(x, ops.conjugate(x)), site + ".S", 0.5 / sigma))
    return out


def num_rhs(rt, Sk, site):
    """SANA_LA_HOIST's la.num right-hand sides of one channel block, shared by every pair: per d > 0 the in-head
    rotation by -d of S_d split into its two masked rotations, (lo(s) rot(S_d, s), hi(s) rot(S_d, -d)), s = DH - d."""
    ops = rt.ops
    out = [None]
    for d in range(1, DH):
        sh = DH - d
        lo, hi = _ih_masks(sh)
        a = rt.mult_mask(ops.rotate(Sk[d], sh * LANES), (site + ".num", "ih.lo", sh), lambda lo=lo: lane_vec(np.ones(LANES), lo))
        b = rt.mult_mask(ops.rotate(Sk[d], -d * LANES), (site + ".num", "ih.hi", sh), lambda hi=hi: lane_vec(np.ones(LANES), hi))
        out.append((a, b))
    return out


def num_lhs(rt, q, S0, rhs):
    """num = sum_d rot_in_head(q S_d, -d) as q S_0 + sum_d rot(q, s) A_d + rot(q, -d) B_d: q's 62 rotations one grid,
    the 63 products one chain of sums (one relinearisation under PC_MULT_SUM). Exact; q's path one level shallower."""
    ops = rt.ops
    steps = [(DH - d) * LANES for d in range(1, DH)] + [-d * LANES for d in range(1, DH)]
    rots = ops.rotate_many(q, steps)
    # the 62 masked products first (one meta: msum::fuse chains them, one relinearisation), q S_0 (one level higher:
    # S_0 has no mask) added last on its own (starting the chain with it refused all 63)
    acc = None
    for d in range(1, DH):
        a, b = rhs[d]
        p = ops.mult(rots[d - 1], a)
        acc = p if acc is None else ops.add(acc, p)
        acc = ops.add(acc, ops.mult(rots[DH - 1 + d - 1], b))
    return ops.add(acc, ops.mult(q, S0))


def lane_total(rt, x, site, scale=1.0):
    """The sum over the LANES lanes of each channel, times `scale`, on every lane (real input)."""
    ops = rt.ops
    s = ops.rotate_and_sum(x, 1, LANES)                               # lane 0 holds the channel's sum
    s = rt.mult_mask(s, (site, "lane0", scale), lambda: lane_vec(scale * (np.arange(LANES) == 0)))
    return ops.rotate_and_sum(s, -1, LANES)


def head_total(rt, x, site, scale=1.0):
    """The sum over each head's DH channels, times `scale`, on all of the head's channels (lane-wise)."""
    ops = rt.ops
    s = ops.rotate_and_sum(x, LANES, DH * LANES)                      # the head's first channel holds the sum
    s = rt.mult_mask(s, (site, "head0", scale),
                     lambda: lane_vec(np.ones(LANES), scale * (np.arange(NCH) % DH == 0)))
    return ops.rotate_and_sum(s, -LANES, DH * LANES)


def token_stat(rt, Xh, inv_w, site, row0=None):
    """Per token of one half (Xh[k]: 2x the half's real features), mean_c (x_c / w_c)^2 over the D real channels on
    every slot of the token's lane (inv_w: (n_blk NCH,) 1 / w on the real channels, 0 on the padding). `row0[k]`: the
    channel row block k's real channels start at in Xh[k] (a partly filled block shared by several chunk pairs: this
    pair's rows; the weight mask selects them, the other pairs' rows drop out of the sum at no level)."""
    ops = rt.ops
    sq = None
    for k, x in enumerate(Xh):
        wk = 0.5 * inv_w[k * NCH:(k + 1) * NCH] / np.sqrt(D_REAL)
        if not np.any(wk):
            continue
        r0 = int(row0[k]) if row0 is not None else 0
        wk = np.roll(wk, r0) if r0 else wk
        y = rt.mult_mask(x, (site, "stat.w", k) + ((r0,) if r0 else ()), lambda wk=wk: lane_vec(np.ones(LANES), wk))
        y = ops.square(y)
        sq = y if sq is None else ops.add(sq, y)
    return ops.rotate_and_sum(sq, LANES, N)


D_REAL = 1152


def linear_attention(rt, Q, K, V, wk, nl, site, delta=1e-15, out_scale=1.0):
    """attn1 before to_out. Q[g][k]: q * norm_q.weight, K[g][k]: k * wk (wk: norm_k.weight times the unit forms'
    input scale), V[g][k]; Q and K carry the ReLUs' input scales (model.fold_block). ReLU(k_hat) = rk ReLU(k * w)
    (rk > 0): the ReLU reads the projection, the key's per-token RMS scale rk (its statistic on k = K / wk) is taken
    into the ReLU's last product (relu_u2 mul=: a one-refresh key composite ends with no product left for it; the
    packed block-2 chains keep it as their tail). nl: relu_u / relu_u2, inv_sqrt(rt, z, site), recip(rt, d, site) on
    real-valued ciphertexts, stat_scale(site) (c: the statistic read as c^2 z), la_sigma(site) (S and z carried
    divided by it). Returns O[g][k] = num / (den + delta) per head, delta in den / sigma units."""
    ops = rt.ops
    nblk = len(Q[0])
    inv_w = np.zeros(nblk * NCH)
    inv_w[:D_REAL] = 1.0 / np.asarray(wk)
    eps = 1e-5
    kh = [[split(ops, x) for x in K[g]] for g in range(G)]
    qh = [[split(ops, x) for x in Q[g]] for g in range(G)]
    with rt.step("la.knorm"):
        rks = _la_knorm(rt, kh, inv_w, nl, site, eps)
    kin = {(g, half, k): kh[g][k][half] for g in range(G) for half in (0, 1) for k in range(nblk)}
    qin = {(g, half, k): qh[g][k][half] for g in range(G) for half in (0, 1) for k in range(nblk)}
    with rt.step("la.relu_k"):                                                    # a PCMM output
        fk = relu_packed(rt, kin, nl, site + ".k", room=10, mul=lambda key: rks[2 * key[0] + key[1]])
    with rt.step("la.relu_q"):
        fq = relu_packed(rt, qin, nl, site + ".q", room=10)

    if RELU_NOMASK:      # block 2's rk on its live channels only: the product with it then cleans the packed chains'
        live = lambda: lane_vec(np.ones(LANES), (np.arange(NCH) < SPARSE_CH).astype(float))    # noqa: E731
        rkl = [rt.mult_mask(r, (site, "rk.live"), live) for r in rks]      # outputs (their unpack masks are gone)
    else:
        rkl = rks

    def fkr(g, half, k):                   # the packed chains' RMS scale (the dense ones carry it already)
        y = fk[(g, half, k)]
        return y if k != SPARSE_BLOCK else ops.mult(y, rkl[2 * g + half])
    # then one complex refresh per pair of halves with the room the S / num chains take (the query's: 4 products,
    # the shallower polished ReLU leaves less)
    fK = [[room_hint(rt, join(ops, fkr(g, 0, k), fkr(g, 1, k)), 6) for k in range(nblk)] for g in range(G)]
    fQ = [[room_hint(rt, join(ops, fq[(g, 0, k)], fq[(g, 1, k)]), 4) for k in range(nblk)] for g in range(G)]
    return _la_tail(rt, fQ, fK, V, nl, site, delta, nblk, out_scale)


SPARSE_BLOCK, SPARSE_CH = 2, D_REAL - 2 * NCH         # block 2 holds 128 real channels (heads 32-35)
# SANA_RELU_NOMASK=1: block 2's packed ReLU chains come back four to one; each output was masked to its 128 live
# channels (a level on both q and k, so block 2's attention output arrived one level past the others and out1 refreshed
# it). The key side's RMS scale rk is masked once instead: f(k) rk is then clean, so S and z are zero past the live
# channels, and what f(q) holds there never survives its products with them
RELU_NOMASK = os.environ.get("SANA_RELU_NOMASK", "0") not in ("", "0")
# SANA_NUM_HINT=1 (p17, with SANA_PREDIV_FOLD on out1): the folded out1 input's refresh site on la_tail's numerator at
# its normal size instead of the PCMM's input hint on values ~2^-s (block 2: 5 per block, same count). Its own switch
# so that plans captured before it keep their op stream
NUM_HINT = os.environ.get("SANA_NUM_HINT", "0") not in ("", "0")


def relu_packed(rt, xs, nl, site, room, mul=None):
    """{(g, half, k): real ct of unit-scaled inputs} -> ReLU each (nl.relu_u), the halves (g, 0, k) and (g, 1, k)
    refreshed jointly (relu_u2); block 2's ciphertexts (128 live channels, zero elsewhere) go four to one (channel
    rotations: no level), unpacked with one mask. `mul`: key -> a ciphertext the dense outputs come back multiplied
    by (relu_u2 mul=); the packed chains then size their tail for the unpack mask and that product (the caller's)."""
    ops = rt.ops
    per = NCH // SPARSE_CH
    sparse = sorted(key for key in xs if key[2] == SPARSE_BLOCK)
    live = lane_vec(np.ones(LANES), (np.arange(NCH) < SPARSE_CH).astype(float))
    groups = [sparse[i0:i0 + per] for i0 in range(0, len(sparse), per)]

    def pack(grp):
        acc = None
        for i, key in enumerate(grp):
            y = ops.rotate(xs[key], -i * SPARSE_CH * LANES) if i else xs[key]
            acc = y if acc is None else ops.add(acc, y)
        return acc

    def unpack(grp, r):
        for i, key in enumerate(grp):
            y = ops.rotate(r, i * SPARSE_CH * LANES) if i else r
            out[key] = y if RELU_NOMASK else rt.mult_mask(y, (site, "unpack"), lambda: live)
    out = {}                               # (g, 0, k) with (g, 1, k): the chain's refreshes joint
    for key in sorted(k_ for k_ in xs if k_[2] != SPARSE_BLOCK and k_[1] == 0):
        twin = (key[0], 1, key[2])
        kw = {} if mul is None else {"mul": (mul(key), mul(twin))}
        out[key], out[twin] = nl.relu_u2(rt, xs[key], xs[twin], site, room=room, **kw)
    accs = [pack(grp) for grp in groups]
    rs = []                                # the packed ciphertexts two by two
    for j in range(0, len(accs) - 1, 2):
        rs.extend(nl.relu_u2(rt, accs[j], accs[j + 1], site, room=room, **({} if mul is None else {"tail": 1 if RELU_NOMASK else 2})))
    if len(accs) % 2:
        rs.append(nl.relu_u(rt, accs[-1], site, room=room))
    for grp, r in zip(groups, rs):
        unpack(grp, r)
    return out


def _la_knorm(rt, kh, inv1, nl, site, eps):
    """Per (g, half) the key's RMS scale 1 / sqrt(mean k^2 + eps) on every slot of each lane (inv1: 1 / wk)."""
    c = nl.stat_scale(site + ".knorm")                                  # the statistic's pre-scale
    stats = []
    for g in range(G):
        for half in (0, 1):
            z = token_stat(rt, [h[half] for h in kh[g]], c * inv1, site + ".knorm")
            stats.append(rt.add_mask(z, (site, "knorm.eps", c), lambda: np.full(N, c * c * eps)))
    # zero on the empty token slots: every key feature carries rk, so they drop out of the attention's sums
    return packed_lane_map(rt, stats, lambda z: nl.inv_sqrt(rt, z, site + ".knorm"), site + ".knorm",
                           lanes=[VALID[slot_of(g, half)].astype(float) for g in range(G) for half in (0, 1)])


def _la_tail(rt, fQ, fK, V, nl, site, delta, nblk, out_scale=1.0):
    ops = rt.ops
    # S_d and z per block, replicated on every lane, both divided by sigma (num / den unchanged, den in its band)
    sigma = nl.la_sigma(site)
    S = []
    Z = []
    st = rt.step("la.S")
    st.__enter__()
    for k in range(nblk):
        S.append(s_sums(rt, [(fK[g][k], V[g][k]) for g in range(G)], site, sigma))
        acc = None
        for g in range(G):
            p = ops.mult_const(fK[g][k], 1.0, -1.0)
            acc = p if acc is None else ops.add(acc, p)
        Z.append(lane_total(rt, ops.add(acc, ops.conjugate(acc)), site + ".z", 0.5 / sigma))
    st.__exit__(None, None, None)
    st = rt.step("la.num")
    st.__enter__()
    nums, dens = [], []
    rhs = [num_rhs(rt, S[k], site) for k in range(nblk)] if LA_NUM_SUM else None
    for g in range(G):
        for k in range(nblk):
            if LA_NUM_SUM:
                num = num_lhs(rt, fQ[g][k], S[k][0], rhs[k])
            else:
                num = None
                for d in range(DH):
                    t = rotate_in_head(rt, ops.mult(fQ[g][k], S[k][d]), -d, site + ".num")
                    num = t if num is None else ops.add(num, t)
            den = head_total(rt, ops.mult(fQ[g][k], Z[k]), site + ".den")
            if out_scale != 1.0 and NUM_HINT:
                # the reciprocals carry the out1 fold (2^-s): a refresh after their product would be one of values
                # ~1e-3 (the PCMM's own input hint, on block 2's deeper chain). Its site sits here instead, on the
                # numerator at its normal size, with the room of that last product
                num = room_hint(rt, num, 2)
            nums.append(split(ops, num))
            if LA_QNORM:
                hq = head_total(rt, fQ[g][k], site + ".hq")
                dens += [(dh, hh, k) for dh, hh in zip(split(ops, den), split(ops, hq))]
                continue
            for dh in split(ops, den):
                # 2 den + 2 delta, and a large value on the padding channels (zero num there): the map's input stays
                # in its band
                dens.append(rt.add_mask(dh, (site, "den.delta", delta, k), lambda k=k: lane_vec(
                    np.ones(LANES), np.where(np.arange(NCH) + k * NCH < D_REAL, 2.0 * delta, 1.0))))
    st.__exit__(None, None, None)
    with rt.step("la.recip"):
        recips = _la_qnorm_map(rt, dens, nl, site, delta, out_scale, DH) if LA_QNORM else packed_group_map(
            rt, dens, lambda x: nl.recip(rt, x, site + ".den"), site + ".den", DH, out_scale=out_scale)
    out = []
    for g in range(G):
        row = []
        for k in range(nblk):
            i = g * nblk + k
            na, nb = nums[i]
            row.append(join(ops, ops.mult(na, recips[2 * i]), ops.mult(nb, recips[2 * i + 1])))   # (2 num) / (2 den)
        out.append(row)
    return out


def _la_qnorm_map(rt, qn, nl, site, delta, out_scale, P):
    """SANA_LA_QNORM. qn: [(2 den / sigma, 2 hq, k)], real ciphertexts constant over each head (hq the head's sum of
    relu(q), k the channel block). Returns out_scale / (2 den / sigma + its floor) per entry, as packed_group_map
    does: with u = 2 hq / H in [0, 1] and s~ = eps / (u + eps) (bounded, positive, analytic: one Chebyshev series on
    the packed ciphertext), D = (a / eps) (2 den / sigma) s~ + 2 delta lies in the reciprocal's band and
    1 / (2 den / sigma) = (a / eps) s~ / D. The gain a / eps rides the pack and unpack masks (no level); the floor
    is delta in D's units."""
    from numpy.polynomial import chebyshev as C

    from .nl import Room, eval_chebyshev
    ops = rt.ops
    q, c = nl.cfg["la_qnorm"][site], nl.cfg["la_qnorm"]
    H, eps = q["H"], c["eps"]
    gain = q["a"] / eps
    coef = [float(v) for v in C.chebinterpolate(lambda x: eps / (0.5 * (x + 1.0) + eps), c["deg"])]
    out = []
    for i0 in range(0, len(qn), P):
        grp = qn[i0:i0 + P]
        ks = [k for _d, _h, k in grp]
        accd = acch = None
        for i, (d, h, _k) in enumerate(grp):
            m = rt.mult_mask(d, (site, "qn.d", P, i, gain),
                             lambda i=i: lane_vec(np.ones(LANES), gain * (np.arange(NCH) % P == i)))
            accd = m if accd is None else ops.add(accd, m)
            m = rt.mult_mask(h, (site, "qn.h", P, i, H),
                             lambda i=i: lane_vec(np.ones(LANES), (np.arange(NCH) % P == i) / H))
            acch = m if acch is None else ops.add(acch, m)

        def floor(ks=ks):
            # 2 delta on the real channels of each class's block, 1 on the padding and on the unused classes (den =
            # hq = 0 there: s~ = 1, D in its band, num = 0). A key ("fold", n) (sana.fold): the folded block, its
            # first n sub-blocks of 128 channels live
            ch = np.arange(NCH)
            i = ch % P
            live = np.zeros(NCH, dtype=bool)
            for j, kk in enumerate(ks):
                rows = (ch < kk[1] * (NCH // 4)) if isinstance(kk, tuple) else (ch + kk * NCH < D_REAL)
                live |= (i == j) & rows
            return lane_vec(np.ones(LANES), np.where(live, 2.0 * delta, 1.0))
        # the series' entry with the room of its two products and the reciprocal's own hint after them
        sv = eval_chebyshev(ops, Room(rt).need(acch, coef, 4), coef, 0.0, 1.0)
        D = rt.add_mask(ops.mult(accd, sv), (site, "qn.delta", delta, P, tuple(ks)), floor)
        y = ops.mult(nl.recip(rt, D, site + ".den"), sv)
        for i in range(len(grp)):
            m = rt.mult_mask(y, (site, "qn.out", P, i, out_scale * gain),
                             lambda i=i: lane_vec(np.ones(LANES), out_scale * gain * (np.arange(NCH) % P == i)))
            m = ops.rotate(m, i * LANES) if i else m                  # to the group's first channel
            out.append(ops.rotate_and_sum(m, -LANES, P * LANES))
    return out


# ── cross-attention (attn2) ─────────────────────────────────────────────────────────────────────────────────────────
#
# 16 heads of 72 channels, head h on block h // 4 at channels [128 (h % 4), 128 (h % 4) + 72) ("x-head layout"; the
# q / k / v projections and to_out carry the permutation). The n_key text keys are few, so each key's row is a
# lane-broadcast vector Kb[j][k] (plaintext for the public BOS, a ciphertext for the prompt's): key j's scores are
# the 128-channel window sums of Q Kb[j], on each head's first channel. They are packed for the softmax at channel
# offset J gg + j of the head's window (J >= n_key a power of two, gg = g mod 128 / J), one packed ciphertext per
# block and group of 128 / J chunk pairs; the softmax runs there; each probability is moved back and spread over its
# head's window before the product with the key's value row Vb[j][k].

XW = 128            # head window (channels)
DX = 72             # head width


class Pt:
    """A public slot vector among ciphertext operands (the BOS key / value rows); `tag` keys its encoding."""

    def __init__(self, v, tag):
        self.v, self.tag = np.asarray(v), tag


def bcast_rows(vec, nblk):
    """(nblk NCH,) channel values -> per block the lane-broadcast slot vector."""
    vec = np.asarray(vec, dtype=np.float64)
    return [lane_vec(np.ones(LANES), vec[k * NCH:(k + 1) * NCH]) for k in range(nblk)]


def xhead_perm():
    """(1152,) destination channel (in 4 blocks of NCH) of each source channel h DX + c."""
    h, c = np.divmod(np.arange(16 * DX), DX)
    return (h // 4) * NCH + XW * (h % 4) + c


def window_total(rt, x, site, scale=1.0):
    """Each head window's sum on its first channel (other channels zero), lane-wise, times `scale`."""
    s = rt.ops.rotate_and_sum(x, LANES, XW * LANES)
    return rt.mult_mask(s, (site, "win0", scale), lambda: lane_vec(np.ones(LANES), scale * (np.arange(NCH) % XW == 0)))


def window_spread(rt, x):
    """A value on each window's first channel copied over the window."""
    return rt.ops.rotate_and_sum(x, -LANES, XW * LANES)


def _key_group(n_key):
    J = 1 << max(0, (n_key - 1).bit_length())
    if J > XW:
        raise ValueError(f"cross-attention: {n_key} keys exceed one {XW}-channel window")
    return J, XW // J            # keys per group, chunk pairs per packed ciphertext


def softmax_masked(rt, S, vmask, J, nl, site, lanes=LANES, exp_folded=False):
    """Softmax over each aligned J-channel group of the real scores S where `vmask` (lane x channel: a causal row) is
    set: exp((s - c) 2^-r) by nl.exp, then r rounds p <- p^2 / sum p^2 (nl.recip), the division first by the group
    sum. `lanes`: the layout's token lanes (N / lanes channels per block)."""
    ops = rt.ops
    first = lane_vec(np.ones(lanes), (np.arange(N // lanes) % J == 0).astype(float))
    tag = zlib.crc32(vmask.tobytes())                      # stable across processes (hash(bytes) is salted)

    def group_sum(x, i):
        s = ops.rotate_and_sum(x, lanes, J * lanes)
        s = rt.mult_mask(s, (site, "gsum", J, i), lambda: first)
        return ops.rotate_and_sum(s, -lanes, J * lanes)
    z = nl.exp(rt, S, site + ".exp", folded=exp_folded)                 # exp_folded: S is the exp's variable
    z = rt.mult_mask(z, (site, "valid", tag), lambda: vmask)
    p = ops.mult(z, nl.recip(rt, group_sum(z, 0), site + ".sum0"))
    for i in range(nl.sm_rounds(site)):
        q = ops.square(p)
        p = ops.mult(q, nl.recip(rt, group_sum(q, i + 1), f"{site}.sum{i + 1}"))
    return p


def softmax_pair(rt, a, b, valid, J, nl, site, lanes=LANES, exp_prescaled=False):
    """softmax_masked on two real score ciphertexts a, b at once (Llama prefill's PF_SM_PAIR): each group sum on their
    join (one ladder for both) with its refresh there, each round's p refresh on the join, split back (2 Re, 2 Im).
    The split's 2 costs nothing: a round normalizes (q / sum q is homogeneous in p) and the reciprocal reads the sum
    it is given (nl.recip's dscale); p after a normalization is half the softmax, the last one doubled by an add.
    `valid`: the slot validity (lane x channel); `lanes`: the layout's token lanes. Returns (pa, pb)."""
    from .nl import Room
    ops = rt.ops
    first = lane_vec(np.ones(lanes), (np.arange(N // lanes) % J == 0).astype(float))
    vtag = zlib.crc32(valid.tobytes())

    def norm(xa, xb, c, i):
        """(xa, xb) / their group sums: xa, xb carry c times the true values; returns half the normalized pair."""
        s = ops.rotate_and_sum(join(ops, xa, xb), lanes, J * lanes)
        s = rt.mult_mask(s, (site, "gsum", J, i), lambda: first)
        s = ops.rotate_and_sum(s, -lanes, J * lanes)
        need = nl.recip_cost(f"{site}.sum{i}")
        if need is not None:
            s = Room(rt).need(s, *need)
        da, db = split(ops, s)                                      # 2 c sum
        return (ops.mult(xa, nl.recip(rt, da, f"{site}.sum{i}", 2.0 * c)),
                ops.mult(xb, nl.recip(rt, db, f"{site}.sum{i}", 2.0 * c)))
    za, zb = (rt.mult_mask(nl.exp(rt, x, site + ".exp", prescaled=exp_prescaled), (site, "valid", vtag),
                           lambda: valid) for x in (a, b))
    pa, pb = norm(za, zb, 1.0, 0)
    c = 0.5
    for i in range(nl.sm_rounds(site)):
        # p's refresh on the join where the round (the square, the reciprocal's product) does not fit
        if ops.level(pa) + 4 + (2 if ops.noise_deg(pa) > 1 else 0) > GUARD:
            from .nl import joint_force
            pa, pb = split(ops, ops.bootstrap(join(ops, pa, pb)) if joint_force()
                           else ops.bootstrap_hint(join(ops, pa, pb), GUARD - 4, True))
            c *= 2.0
        pa, pb = norm(ops.square(pa), ops.square(pb), c * c, i + 1)
        c = 0.5
    return ops.add(pa, pa), ops.add(pb, pb)


def cross_attention(rt, Q, Kb, Vb, wq, nl, site, out_scale=1.0):
    """attn2 before to_out. Q[g][k] (4 blocks, x-head layout): q * norm_q.weight, its RMS scale taken here (inv_w =
    1 / norm_q.weight); Kb[j][k]: key j's normalized row / sqrt(DX), lane-broadcast; Vb[j][k]: its value row. Kb / Vb
    entries are ciphertexts or Pt (public rows). Returns O[g][k] (x-head layout)."""
    ops = rt.ops
    nblk = len(Q[0])
    n_key = len(Kb)
    J, gpc = _key_group(n_key)
    Kb = center_keys(rt, Kb, site)
    inv_w = np.zeros(nblk * NCH)
    inv_w[xhead_perm()] = 1.0 / np.asarray(wq)

    def mul(x, y, key):
        if isinstance(y, Pt):
            return rt.mult_mask(x, key + (y.tag,), lambda: y.v)
        return ops.mult(x, y)
    # the query's RMS scale per token
    qh = [[split(ops, x) for x in Q[g]] for g in range(G)]
    c = nl.stat_scale(site + ".qnorm")
    stats = []
    for g in range(G):
        for half in (0, 1):
            z = token_stat(rt, [h[half] for h in qh[g]], c * inv_w, site + ".qnorm")
            stats.append(rt.add_mask(z, (site, "qnorm.eps", c), lambda: np.full(N, c * c * 1e-5)))
    rqs = packed_lane_map(rt, stats, lambda z: nl.inv_sqrt(rt, z, site + ".qnorm"), site + ".qnorm")
    n_pack = -(-G // gpc)
    # the scores on the raw q, its RMS scale taken once per packed score ciphertext (the scores are linear in it);
    # the keys carry 1 / M and the scale masks M (M = 1 / the q statistic's scale c: the raw scores stay inside the
    # normalized ones' band)
    Msc = xa_scale(nl, site)
    kfold = 1.0 / (np.sqrt(DX) * Msc) if XA_KEY_FOLD else 1.0
    # PF_SM_FOLD=1: the exp's domain slope rides this mask too (softmax_pair's exp then only shifts: a free add)
    from .gemma_fhe import sm_fold
    efold = sm_fold()
    if efold:
        e = nl.cfg["exp"][site + ".sm.exp"]
        Msc *= 2.0 / (e["b"] - e["a"])
    RQ = []                                  # [c][half]: rq of chunk pair c gpc + i on group i's channels, times M
    for c_ in range(n_pack):
        halves = []
        for half in (0, 1):
            acc = None
            for i in range(gpc):
                g = c_ * gpc + i
                if g >= G:
                    break
                t = rt.mult_mask(rqs[2 * g + half], (site, "rq.grp", i, J, Msc), lambda i=i: lane_vec(
                    np.ones(LANES), ((np.arange(NCH) % XW) // J == i) * Msc))
                acc = t if acc is None else ops.add(acc, t)
            halves.append(acc)
        RQ.append(halves)
    # scores, packed: key j of chunk pair g on channel J (g % gpc) + j of its head window, ciphertext g // gpc
    packed = [[None] * nblk for _ in range(n_pack)]
    for g in range(G):
        for k in range(nblk):
            for j in range(n_key):
                s = window_total(rt, mul(Q[g][k], Kb[j][k], (site, "kb", j, k)), site + ".s", 0.5 * kfold)
                off = J * (g % gpc) + j
                s = ops.rotate(s, -off * LANES) if off else s
                c = packed[g // gpc][k]
                packed[g // gpc][k] = s if c is None else ops.add(c, s)
    for c_ in range(n_pack):                 # 2 (Re P rq0 + i Im P rq1) M: what the q_hat scores packed to
        for k in range(nblk):
            a, b = split(ops, packed[c_][k])
            packed[c_][k] = join(ops, ops.mult(a, RQ[c_][0]), ops.mult(b, RQ[c_][1]))
    valid = lane_vec(np.ones(LANES), (np.arange(NCH) % J < n_key).astype(float))
    probs = []
    for c in range(n_pack):
        row = []
        for k in range(nblk):
            # one entry refresh, the two halves' sums and rounds refreshed jointly; then the unpack's room (the window
            # mask, the value product)
            a, b = split(ops, nl.exp_need(rt, packed[c][k], site + ".sm.exp"))
            pa, pb = softmax_pair(rt, a, b, valid, J, nl, site + ".sm", exp_prescaled=efold)
            row.append(room_hint(rt, join(ops, pa, pb), 3))
        probs.append(row)
    out = []
    for g in range(G):
        row = []
        for k in range(nblk):
            acc = None
            for j in range(n_key):
                off = J * (g % gpc) + j
                p = probs[g // gpc][k]
                p = ops.rotate(p, off * LANES) if off else p
                # out_scale: the next linear's pre-division (SANA_ATTN_MERGE's out), on the unpack mask: no level
                p = rt.mult_mask(p, (site, "win0", out_scale),
                                 lambda: lane_vec(np.ones(LANES), out_scale * (np.arange(NCH) % XW == 0).astype(float)))
                t = mul(window_spread(rt, p), Vb[j][k], (site, "vb", j, k))
                acc = t if acc is None else ops.add(acc, t)
            row.append(acc)
        out.append(row)
    return out
