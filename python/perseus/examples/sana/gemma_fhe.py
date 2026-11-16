"""Gemma-2-2B on the prompt rows, encrypted dataflow (a blocked layout with the n_tok <= GL prompt tokens on the real
lanes: one ciphertext per channel block). The 207 public positions enter as plaintext: their keys and values per layer
are the weights of two PCMMs (scores = q K_pub^T / 16, P V_pub), block-diagonal per head; the prompt's own keys /
values are lane-broadcast rows (kernels.py's cross-attention pattern) at score slots 207 + j.

Layouts (GL = 16 lanes, GNCH = N / GL channels per block): hidden 2304 on HB blocks; q (8 heads x 256)
on QB blocks, head h at channels [256 (h % HPB), +256) of block h // HPB; scores: the same blocks, head h's window
holding key c at channel 256 (h % HPB) + c (c < 207 public, 207 + j prompt key j); k / v (4 KV heads) on KB blocks,
or with KV_DUP every q head's KV head in the q layout (its columns duplicated in the projection: no GQA expansion).
RoPE: q = Q1 cos + Q2 sin with Q2 the rotate-half projection (its permutation and sign folded into the weights),
cos / sin per (lane = position, channel). The residual is carried as x / r. Norms (Gemma's (1 + w) RMSNorm) by
blocked.norm_blocked(rms=True): the input / pre-feed-forward norms' (1 + w) ride the next linear's rows, the post
norms' a channel mask. The caption's rows leave in SANA's layout (to_sana: LANES lanes, NCH channels)."""
from __future__ import annotations

import dataclasses
import math
import os

import numpy as np

from perseus.impl.prefill import split

from . import gemma as GM
from . import kernels as KN
from .layout import LANES, N, NCH
from .model import Lin, _pad, _padv

# the prompt's <= 16 tokens on 16 lanes, 2048 channels per block (hidden on 2 blocks, q / scores / k / v on 1, the FFN
# on 5); SANA's own 64-lane layout would leave ~5% of the slots live at 5-11 tokens
GL = 16
GNCH = N // GL
WIN = 256                            # a head's window (q channels, score slots)
HPB = GNCH // WIN                    # heads per block
HB, QB, KB, FB = (-(-n // GNCH) for n in (GM.D, GM.H * GM.DH, GM.H_KV * GM.DH, GM.FF))
# the KV heads duplicated per q head when that costs no block (the q layout's KB = QB): no GQA expansion
KV_DUP = QB == KB
KBX = QB if KV_DUP else KB           # k / v blocks as laid out


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
    d = Dims(N, HB * GNCH, GM.D, HB * GNCH // 64, GM.D // 64, 2 * FB * GNCH, 2 * GM.FF)
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


def q_perm():
    """Destination channel (QB blocks) of q's source channel h DH + d."""
    h, d = np.divmod(np.arange(GM.H * GM.DH), GM.DH)
    return (h // HPB) * GNCH + WIN * (h % HPB) + d


def kv_perm():
    g, d = np.divmod(np.arange(GM.H_KV * GM.DH), GM.DH)
    return (g // HPB) * GNCH + WIN * (g % HPB) + d


def rotate_half_matrix(nh):
    """(nh DH, nh DH) R with x R = rotate_half(x) per head: [-x2, x1]."""
    R = np.zeros((nh * GM.DH, nh * GM.DH))
    h2 = GM.DH // 2
    for h in range(nh):
        o = h * GM.DH
        for i in range(h2):
            R[o + h2 + i, o + i] = -1.0           # out[i] = -x[h2 + i]
            R[o + i, o + h2 + i] = 1.0            # out[h2 + i] = x[i]
    return R


def rope_masks(pos0, n, perm, nblk):
    """(cos, sin) slot vectors per block: lane = token at position pos0 + lane (lanes >= n: 0), channel by perm."""
    inv = 1.0 / GM.THETA ** (np.arange(0, GM.DH, 2) / GM.DH)
    freq = np.concatenate([inv, inv])                         # per in-head channel
    src = np.arange(len(perm))
    f = np.zeros(nblk * GNCH)
    f[perm] = freq[src % GM.DH]
    live = np.zeros(nblk * GNCH, dtype=bool)
    live[perm] = True
    pos = pos0 + np.arange(GL)
    ang = np.outer(f, pos)                                    # (channels, lanes)
    on = (np.arange(GL) < n)[None] & live[:, None]
    cos, sin = np.where(on, np.cos(ang), 0.0), np.where(on, np.sin(ang), 0.0)
    return ([cos[k * GNCH:(k + 1) * GNCH].reshape(N) for k in range(nblk)],
            [sin[k * GNCH:(k + 1) * GNCH].reshape(N) for k in range(nblk)])


@dataclasses.dataclass
class LayerWeights:
    qkv: Lin                 # h -> [Q1 | Q2 | K1 | K2 | V]  (QB, QB, KBX, KBX, KBX blocks)
    s_pub: Lin               # rotated q -> scores on the public keys (/ 16)
    pv_pub: Lin              # probabilities -> q layout (public values)
    o: Lin                   # attention output -> hidden (r_out folded)
    gate_up: Lin             # 2 rms(x) -> [gate | up] (FB + FB blocks)
    down: Lin                # 2 gelu(gate) up -> hidden
    post_attn: np.ndarray    # (1 + w) / 2 / r on the hidden channels
    post_ff: np.ndarray
    r: np.ndarray
    gelu_folded: bool = False
    sc_scale: float = 1.0


def sm_fold():
    """PF_SM_FOLD=1 (Llama's SM_FOLD idea on Gemma's attention): the softcap's domain map (1 / its half-width) folded
    into the score producers (s_pub's matrix, the prompt scores' 1 / sqrt(d) mask) and the exp's domain map folded
    into the softcap series' coefficients (it outputs the exp's [-1, 1] variable): two levels per layer, exact."""
    return os.environ.get("PF_SM_FOLD", "0") not in ("", "0")


def gelu_fold():
    """PF_GELU_FOLD=1 (Llama's GELU_FOLD): each layer's gate columns carry 1 / (2 xm) of its GELU series' symmetric
    domain [-xm, xm], so the series reads its input on [-1, 1] as is: no affine map (a level per gate ciphertext)."""
    return os.environ.get("PF_GELU_FOLD", "0") not in ("", "0")


def fold_layer(W, b, kv_pub, r, gelu_xm=None, sc_scale=1.0):
    """Layer b's linears; kv_pub: (K, V) (H_KV, n_pub, DH) of the public prefix (K post-RoPE); gelu_xm: the layer's
    GELU domain half-width when its gate is folded (gelu_fold); sc_scale: the scores' fold (sm_fold: 1 / the softcap's
    half-width)."""
    t = lambda name: W.layer(b, name).numpy()                 # noqa: E731
    D, H3 = GM.D, HB * GNCH
    rep = GM.H // GM.H_KV
    w_in = 1.0 + t("input_layernorm.weight")
    Wq, Wk, Wv = t("self_attn.q_proj.weight").T, t("self_attn.k_proj.weight").T, t("self_attn.v_proj.weight").T
    qp = q_perm()
    if KV_DUP:                                                # q head h's KV head h // rep in its own window
        dup = np.concatenate([np.arange((h // rep) * GM.DH, (h // rep + 1) * GM.DH) for h in range(GM.H)])
        Wk, Wv, Rk, kp = Wk[:, dup], Wv[:, dup], rotate_half_matrix(GM.H), qp
    else:
        Rk, kp = rotate_half_matrix(GM.H_KV), kv_perm()
    Rq = rotate_half_matrix(GM.H)
    cols = np.concatenate([qp, QB * GNCH + qp, 2 * QB * GNCH + kp, 2 * QB * GNCH + KBX * GNCH + kp,
                           2 * QB * GNCH + 2 * KBX * GNCH + kp])
    M = np.concatenate([Wq, Wq @ Rq, Wk, Wk @ Rk, Wv], 1) * w_in[:, None]          # rows read 2 rms(x)
    qkv = Lin(f"g{b}.qkv", _pad(M, H3, (2 * QB + 3 * KBX) * GNCH, None, cols), np.zeros((2 * QB + 3 * KBX) * GNCH),
              0.5, GNCH)
    Kp, Vp = (np.asarray(x) for x in kv_pub)
    n_pub = Kp.shape[1]
    S = np.zeros((QB * GNCH, QB * GNCH))
    PV = np.zeros((QB * GNCH, QB * GNCH))
    for h in range(GM.H):
        g = h // rep
        o = (h // HPB) * GNCH + WIN * (h % HPB)
        S[o:o + GM.DH, o:o + n_pub] = Kp[g].T / math.sqrt(GM.DH) * sc_scale
        PV[o:o + n_pub, o:o + GM.DH] = Vp[g]
    s_pub = Lin(f"g{b}.s_pub", S, np.zeros(QB * GNCH), 1.0, GNCH)
    pv_pub = Lin(f"g{b}.pv_pub", PV, np.zeros(QB * GNCH), 1.0, GNCH)
    o = Lin(f"g{b}.o", _pad(t("self_attn.o_proj.weight").T, QB * GNCH, H3, qp, None), np.zeros(H3), 1.0, GNCH)
    w_ff = 1.0 + t("pre_feedforward_layernorm.weight")
    GU = np.zeros((D, 2 * FB * GNCH))                         # gate on blocks [0, FB), up on [FB, 2 FB)
    GU[:, :GM.FF] = t("mlp.gate_proj.weight").T * w_ff[:, None] / (2.0 * gelu_xm if gelu_xm else 1.0)
    GU[:, FB * GNCH:FB * GNCH + GM.FF] = t("mlp.up_proj.weight").T * w_ff[:, None]
    gate_up = Lin(f"g{b}.gate_up", _pad(GU, H3, 2 * FB * GNCH), np.zeros(2 * FB * GNCH), 0.5, GNCH)
    down = Lin(f"g{b}.down", _pad(t("mlp.down_proj.weight").T, FB * GNCH, H3), np.zeros(H3), 1.0, GNCH)
    pa = _padv((1.0 + t("post_attention_layernorm.weight")) / (2 * r), H3)        # the norm's 2 rms
    pf = _padv((1.0 + t("post_feedforward_layernorm.weight")) / (2 * r), H3)
    return LayerWeights(qkv, s_pub, pv_pub, o, gate_up, down, pa, pf, r, bool(gelu_xm), float(sc_scale))


def _bcast_lane(rt, x, j):
    """Lane j of x (real) on every lane."""
    ops = rt.ops
    y = rt.mult_mask(x, ("g.tok", j), lambda: (np.arange(N) % GL == j).astype(float))
    y = ops.rotate(y, j) if j else y
    return ops.rotate_and_sum(y, -1, GL)


def _gqa_expand(rt, kvb, b, site):
    """kv head b (block b // 2, window b % 2) on both windows of a q block (HPB = 2: SANA's layout)."""
    ops = rt.ops
    w = b % 2
    m = rt.mult_mask(kvb[b // 2], (site, "win", w),
                     lambda: gvec(np.ones(GL), (np.arange(GNCH) // WIN == w).astype(float)))
    return ops.add(m, ops.rotate(m, (WIN if w == 0 else -WIN) * GL))


def attention(rt, q, k, v, lw: LayerWeights, n_tok, n_pub, nl, lin, site):
    """q (QB blocks, rotated), k / v (KBX blocks, k rotated) of the prompt rows -> the attention output (q layout)."""
    ops = rt.ops
    kv_of = (lambda xs, blk, _s: xs[blk]) if KV_DUP else (lambda xs, blk, s: _gqa_expand(rt, xs, blk, s))
    with rt.step("s_pub"):
        S = lin(rt, [q], lw.s_pub)[0]                                    # public keys' scores
    st = rt.step("s_prompt")
    st.__enter__()
    for j in range(n_tok):
        kj = [_bcast_lane(rt, x, j) for x in k]
        for blk in range(QB):
            p = ops.mult(q[blk], kv_of(kj, blk, site + ".k"))
            s = ops.rotate_and_sum(p, GL, WIN * GL)                     # the head window's sum on its first channel
            f = lw.sc_scale / math.sqrt(GM.DH)
            s = rt.mult_mask(s, (site, "s.first", f), lambda f=f: gvec(np.ones(GL), (np.arange(GNCH) % WIN == 0) * f))
            S[blk] = ops.add(S[blk], ops.rotate(s, -(n_pub + j) * GL))
    st.__exit__(None, None, None)
    # the causal validity: lane i reads the public keys and the prompt keys j <= i
    c = np.arange(GNCH) % WIN
    valid = ((c[:, None] < n_pub) | ((c[:, None] >= n_pub) & (c[:, None] - n_pub <= np.arange(GL)[None]))) & \
        (np.arange(GL)[None] < n_tok)
    valid |= (np.arange(GL)[None] >= n_tok) & (c[:, None] == 0)          # empty lanes: a nonzero row sum
    vmask = valid.astype(float).reshape(N)
    P = []
    for blk in range(QB):
        s = split(ops, S[blk])[0]                                         # 2 s (real)
        folded = lw.sc_scale != 1.0
        with rt.step("softcap"):                                          # 2 * 50 tanh(s / 50), or its exp variable
            s = nl.softcap(rt, s, site + ".softcap", folded=folded, exp_site=site + ".sm.exp" if folded else None)
        with rt.step("softmax"):
            P.append(KN.softmax_masked(rt, s, vmask, WIN, nl, site + ".sm", lanes=GL, exp_folded=folded))
    with rt.step("pv_pub"):
        O = lin(rt, [P], lw.pv_pub)[0]
    st = rt.step("pv_prompt")
    st.__enter__()
    for j in range(n_tok):
        vj = [_bcast_lane(rt, x, j) for x in v]
        for blk in range(QB):
            pj = ops.rotate(P[blk], (n_pub + j) * GL)
            pj = rt.mult_mask(pj, (site, "p.first"), lambda: gvec(np.ones(GL),
                                                                  (np.arange(GNCH) % WIN == 0).astype(float)))
            pj = ops.rotate_and_sum(pj, -GL, WIN * GL)
            O[blk] = ops.add(O[blk], ops.mult(pj, kv_of(vj, blk, site + ".v")))
    st.__exit__(None, None, None)
    return O


def layer(rt, X, lw: LayerWeights, n_tok, n_pub, nl, lin, b):
    """One Gemma layer on the prompt rows X (HB blocks, carried as x / r)."""
    from perseus.impl.blocked import norm_blocked
    ops = rt.ops
    _d, B, ch = geometry(n_tok)
    p = f"g{b}"
    with rt.step("ln_in"):
        Hn = norm_blocked(rt, [KN_unscale(rt, X, lw.r, p + ".ln_in")], nl.ln(p + ".ln_in"), ch, B, rms=True)[0]
    with rt.step("qkv"):
        o = lin(rt, [Hn], lw.qkv)[0]
    q1, q2 = o[:QB], o[QB:2 * QB]
    k1, k2, v = o[2 * QB:2 * QB + KBX], o[2 * QB + KBX:2 * QB + 2 * KBX], o[2 * QB + 2 * KBX:]
    cq, sq = rope_masks(n_pub, n_tok, q_perm(), QB)
    ck, sk = rope_masks(n_pub, n_tok, q_perm() if KV_DUP else kv_perm(), KBX)
    q = [ops.add(rt.mult_mask(a, (p, "rope.qc", i), lambda i=i: cq[i]),
                 rt.mult_mask(c, (p, "rope.qs", i), lambda i=i: sq[i])) for i, (a, c) in enumerate(zip(q1, q2))]
    k = [ops.add(rt.mult_mask(a, (p, "rope.kc", i), lambda i=i: ck[i]),
                 rt.mult_mask(c, (p, "rope.ks", i), lambda i=i: sk[i])) for i, (a, c) in enumerate(zip(k1, k2))]
    with rt.step("attn"):
        O = attention(rt, q, k, v, lw, n_tok, n_pub, nl, lin, p + ".attn")
    with rt.step("o"):
        A = lin(rt, [O], lw.o)[0]
    with rt.step("ln_post_attn"):
        A = norm_blocked(rt, [A], nl.ln(p + ".ln_post_attn"), ch, B, rms=True)[0]
        A = [rt.mult_mask(a, (p, "post_attn", i), lambda i=i: gvec(np.ones(GL), lw.post_attn[i * GNCH:(i + 1) * GNCH]))
             for i, a in enumerate(A)]
    X = [ops.add(x, a) for x, a in zip(X, A)]
    with rt.step("ln_ff"):
        Hn = norm_blocked(rt, [KN_unscale(rt, X, lw.r, p + ".ln_ff")], nl.ln(p + ".ln_ff"), ch, B, rms=True)[0]
    with rt.step("gate_up"):
        GU = lin(rt, [Hn], lw.gate_up)[0]
    with rt.step("glu"):
        nb = len(GU) // 2
        # the gate reads 2 x (scale 0.5), or x / xm with the gate folded (gelu_fold: scale xm, the series on [-1, 1])
        sc = nl.cfg["gelu"][p + ".act"]["b"] if lw.gelu_folded else 0.5
        Gl = [ops.mult(nl.gelu_tanh(rt, split(ops, g)[0], p + ".act", sc), u) for g, u in zip(GU[:nb], GU[nb:])]
    with rt.step("down"):
        F = lin(rt, [Gl], lw.down)[0]
    with rt.step("ln_post_ff"):
        F = norm_blocked(rt, [F], nl.ln(p + ".ln_post_ff"), ch, B, rms=True)[0]
        F = [rt.mult_mask(f, (p, "post_ff", i), lambda i=i: gvec(np.ones(GL), lw.post_ff[i * GNCH:(i + 1) * GNCH]))
             for i, f in enumerate(F)]
    return [ops.add(x, f) for x, f in zip(X, F)]


def KN_unscale(rt, X, r, site):
    """X (one row of blocks) times r per channel: the norm's true input."""
    if np.all(r == 1.0):
        return X
    rv = _padv(r, HB * GNCH)
    return [rt.mult_mask(x, (site, "unscale", i), lambda i=i: gvec(np.ones(GL), rv[i * GNCH:(i + 1) * GNCH]))
            for i, x in enumerate(X)]


C1 = -(-1152 // GNCH)                # the caption's blocks (SANA's 1152 channels) in the Gemma layout


def caption_lins(Wg, Ws):
    """Gemma's final norm (1 + w) folded into caption_projection.linear_1 (input 2 rms(x): in_scale 0.5), then
    linear_2; both onto the C1-block caption rows (the Gemma layout)."""
    w_n = 1.0 + Wg["norm.weight"].numpy()
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
