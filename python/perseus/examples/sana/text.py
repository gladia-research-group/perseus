"""SANA's side of its text encoder: the prompt rows' layout (16 lanes, 2048 channels per block: the general encoder's,
examples/llama/encoder.py, which runs the 26 Gemma-2 layers), the caption stage (Gemma's final norm,
caption_projection, caption_norm) and its relayout to SANA's 64 lanes (to_sana)."""
from __future__ import annotations

import os

import numpy as np

from perseus.impl.prefill import split

from . import kernels as KN
from .layout import LANES, N, NCH
from .model import Lin, _pad, _padv

D, FF = 2304, 9216                   # Gemma-2-2B's residual and MLP widths
GL = 16
GNCH = N // GL
FB = -(-FF // GNCH)
HB = -(-D // GNCH)                # the residual stream's blocks
C1 = -(-1152 // GNCH)                # the caption's blocks (SANA's 1152 channels) in the Gemma layout


def rot_steps():
    """The Gemma layout's rotations (a superset: every channel step, every lane): the twin's keys; a planned run
    loads only its programs' (prefill.plan_rot_scope)."""
    return {s * m * GL for m in range(1, GNCH) for s in (1, -1)} | {s * j for j in range(1, GL) for s in (1, -1)}


def gvec(per_lane, per_channel):
    """A slot vector of the Gemma layout: a (GL,) lane pattern times a (GNCH,) channel pattern."""
    return np.outer(np.asarray(per_channel), np.asarray(per_lane)).reshape(N)


def geometry(n_tok):
    from perseus.impl.blocked import Blocks
    from perseus.impl.layout import Dims
    from perseus.impl.prefill import Chunks
    d = Dims(N, HB * GNCH, D, HB * GNCH // 64, D // 64, 2 * FB * GNCH, 2 * FF)
    B = Blocks(d, GNCH)
    return d, B, Chunks(n_tok, B.t)


def encode(rows, nblk):
    """(n, C) rows (n <= GL) -> per block the slot vector (token on the real part of its lane)."""
    rows = np.asarray(rows, dtype=np.float64)
    full = np.zeros((GL, nblk * GNCH))
    full[:rows.shape[0], :rows.shape[1]] = rows
    return [np.ascontiguousarray(full[:, k * GNCH:(k + 1) * GNCH].T.reshape(N)).astype(complex) for k in range(nblk)]


def decode(cts, n, C=None):
    x = np.concatenate([np.asarray(c).real.reshape(GNCH, GL).T for c in cts], 1)[:n]
    return x if C is None else x[:, :C]


def decode_sana(cts, n, C=None):
    """Rows of SANA's layout (tokens on the real lanes of LANES, NCH channels per block): the caption's output."""
    x = np.concatenate([np.asarray(c).real.reshape(NCH, LANES).T for c in cts], 1)[:n]
    return x if C is None else x[:, :C]


def sm_fold():
    """PF_SM_FOLD=1 (Llama's SM_FOLD idea on Gemma's attention): the softcap's domain map (1 / its half-width) folded
    into the score producers (s_pub's matrix, the prompt scores' 1 / sqrt(d) mask) and the exp's domain map folded
    into the softcap series' coefficients (it outputs the exp's [-1, 1] variable): two levels per layer, exact."""
    return os.environ.get("PF_SM_FOLD", "0") not in ("", "0")


def KN_unscale(rt, X, r, site):
    """X (one row of blocks) times r per channel: the norm's true input."""
    if r is None or np.all(r == 1.0):
        return X
    rv = _padv(r, HB * GNCH)
    return [rt.mult_mask(x, (site, "unscale", i), lambda i=i: gvec(np.ones(GL), rv[i * GNCH:(i + 1) * GNCH]))
            for i, x in enumerate(X)]


def caption_lins(Wg, Ws):
    """Gemma's final norm (1 + w) folded into caption_projection.linear_1 (input 2 rms(x): in_scale 0.5), then
    linear_2; both onto the C1-block caption rows (the Gemma layout)."""
    w_n = 1.0 + np.asarray(Wg["norm.weight"], dtype=np.float64)
    H1 = C1 * GNCH
    l1 = Lin("cap.l1", _pad(Ws["caption_projection.linear_1.weight"].numpy().T * w_n[:, None], HB * GNCH, H1),
             _padv(Ws["caption_projection.linear_1.bias"].numpy(), H1), 0.5, GNCH)
    l2 = Lin("cap.l2", _pad(Ws["caption_projection.linear_2.weight"].numpy().T, H1, H1),
             _padv(Ws["caption_projection.linear_2.bias"].numpy(), H1), 1.0, GNCH)
    return l1, l2


def caption(rt, X, r, Wg, Ws, n_tok, nl, lin):
    """Gemma's final norm, caption_projection and caption_norm on the prompt rows X (HB blocks, x / r): SANA's text
    rows T (3 blocks of SANA's layout, the prompt tokens on the real lanes)."""
    from perseus.impl.blocked import Blocks, norm_blocked
    from perseus.impl.layout import Dims
    from perseus.impl.prefill import Chunks
    ops = rt.ops
    _d, B, ch = geometry(n_tok)
    l1, l2 = caption_lins(Wg, Ws)
    with rt.step("g.norm"):
        Hn = norm_blocked(rt, [KN_unscale(rt, X, r, "g.norm")], nl.ln("g.norm"), ch, B, rms=True)[0]
    with rt.step("cap.l1"):
        U = lin(rt, [Hn], l1)[0]
    with rt.step("cap.act"):
        U = [nl.gelu_tanh(rt, split(ops, u)[0], "caption.act", 0.5) for u in U]
    with rt.step("cap.l2"):
        U = lin(rt, [U], l2)[0]
    B1 = Blocks(Dims(N, C1 * GNCH, 1152, C1 * GNCH // 64, 18, 12 * NCH, 5760), GNCH)
    with rt.step("cap.norm"):
        T = norm_blocked(rt, [U], nl.ln("caption.norm"), Chunks(n_tok, B1.t), B1, rms=True)[0]
    w = _padv(Ws["caption_norm.weight"].numpy() / 2.0, C1 * GNCH)
    T = [rt.mult_mask(t, ("cap.norm.w", i), lambda i=i: gvec(np.ones(GL), w[i * GNCH:(i + 1) * GNCH]))
         for i, t in enumerate(T)]
    if GL == LANES:
        return T
    with rt.step("cap.relayout"):
        return to_sana(rt, T, n_tok)


def to_sana(rt, T, n_tok, C=1152):
    """The caption rows from the Gemma layout (channel c at slot (c % GNCH) GL + lane of block c // GNCH) to SANA's
    (slot (c % NCH) LANES + lane of block c // NCH): per SANA block, its channels rotated to the front, then spread
    from stride GL to stride LANES bit by bit of the channel index (the high bits first: no two values meet), one
    masked product (a level) per bit."""
    ops = rt.ops
    nbits = (NCH - 1).bit_length()
    gap = LANES - GL
    out = []
    for kb in range(-(-C // NCH)):
        c0 = kb * NCH
        src, off = divmod(c0, GNCH)                            # the block, and the first channel's offset in it
        x = ops.rotate(T[src], off * GL) if off else T[src]
        x = KN.room_hint(rt, x, nbits + 1)
        nch = min(NCH, C - c0, GNCH - off)
        cs = np.repeat(np.arange(nch), GL)
        ls = np.tile(np.arange(GL), nch)
        live = ls < n_tok
        cs, ls = cs[live], ls[live]
        pos = cs * GL + ls                                     # where each value sits now
        for i in reversed(range(nbits)):
            mv = (cs >> i) & 1 == 1
            m_mv = np.zeros(N)
            m_mv[pos[mv]] = 1.0
            y = rt.mult_mask(x, ("g.relayout.mv", kb, i, n_tok), lambda m=m_mv: m)
            if i == nbits - 1:                                 # the first step also drops what is not this block's
                m_st = np.zeros(N)
                m_st[pos[~mv]] = 1.0
                x = rt.mult_mask(x, ("g.relayout.st", kb, n_tok), lambda m=m_st: m)
            else:
                x = ops.sub(x, y)
            x = ops.add(x, ops.rotate(y, -gap * (1 << i)))
            pos = np.where(mv, pos + gap * (1 << i), pos)
        assert np.array_equal(pos, cs * LANES + ls)
        out.append(x)
    return out
