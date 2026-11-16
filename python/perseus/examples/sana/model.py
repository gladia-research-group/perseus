"""The SANA block on the blocked layout: the folded weights (adaLN modulation, gates, norm weights, head layouts and
the residual's per-channel scale in the linears) and the block's dataflow on the kernels. `lin(rt, X, Lin)` applies
a linear (the dense mirror here; the PCMM engine encrypted), `nl` the nonlinear maps (kernels.py's hooks plus
silu and the LayerNorm config).

The residual stream is carried as x / r (r: (D,) per-channel scale, the massive channels' range folded away); the
LayerNorms read r x (one plaintext product), the linears that read the stream carry r on their rows, the linears that
write it carry 1 / r on their columns."""
from __future__ import annotations

import dataclasses
import os

import numpy as np

from perseus.impl.prefill import join, split

from . import kernels as KN
from . import layout as L
from .layout import NCH

PCMM_ROOM, REFRESH_ROOM = 11, 6     # products a PCMM output / a refresh leaves below the guard (level 48)

MLP, MLP_PAD = 2880, 3072        # GLUMBConv hidden half and its padded width (6 blocks)


@dataclasses.dataclass
class Lin:
    """y = (in_scale x) W + b on the padded blocked layouts: W (n_in nch, n_out nch), b (n_out nch,); nch: the
    layout's channels per block (N / nch token lanes: NCH for SANA's, the Gemma layout's GNCH)."""
    name: str
    W: np.ndarray
    b: np.ndarray
    in_scale: float = 1.0
    nch: int = NCH

    @property
    def n_in(self): return self.W.shape[0] // self.nch
    @property
    def n_out(self): return self.W.shape[1] // self.nch


def dense_lin(rt, X, lin: Lin):
    """The mirror's linear: decode, product, encode (NumpyOps only)."""
    if lin.nch != NCH:                      # tokens on the real lanes of N / nch (the Gemma layout), X[g][k]
        t = L.N // lin.nch
        out = []
        for row in X:
            rows = np.concatenate([np.asarray(c).reshape(lin.nch, t).T for c in row], 1)
            y = (lin.in_scale * rows) @ lin.W + lin.b
            out.append([np.ascontiguousarray(y[:, k * lin.nch:(k + 1) * lin.nch].T.reshape(L.N))
                        for k in range(lin.n_out)])
        return out
    rows = L.decode(X)
    return L.encode((lin.in_scale * rows) @ lin.W + lin.b, lin.n_out)


def _pad(M, rows, cols, row_idx=None, col_idx=None):
    out = np.zeros((rows, cols))
    ri = np.arange(M.shape[0]) if row_idx is None else row_idx
    ci = np.arange(M.shape[1]) if col_idx is None else col_idx
    out[np.ix_(ri, ci)] = M
    return out


def _padv(v, n, idx=None):
    out = np.zeros(n)
    out[np.arange(len(v)) if idx is None else idx] = v
    return out


@dataclasses.dataclass
class BlockWeights:
    qkv: Lin
    out1: Lin
    qx: Lin
    out2: Lin
    ff_in: Lin
    ff_out: Lin
    wk1: np.ndarray            # attn1.norm_k.weight
    wq2: np.ndarray            # attn2.norm_q.weight
    dw_w: np.ndarray           # (12 NCH, 3, 3) depthwise filters in the ff layout
    dw_b: np.ndarray
    r: np.ndarray              # (D,) residual scale
    qxm: Lin | None = None     # SANA_ATTN_MERGE: qx on [x, attn1 out] (out1 merged in)
    outm: Lin | None = None    # SANA_ATTN_MERGE: out1 + out2 on [attn1 out, attn2 out]


# SANA_ATTN_MERGE=1: the cross-attention reads the residual with no norm, so out1 need not be finished on its own:
# q_x = (x + o1 W1 + b1) Wq + bq = x Wq + o1 (W1 Wq) + (b1 Wq + bq) and x + o1 W1 + o2 W2 + b1 + b2 is ONE linear on
# [o1, o2]: out1's finishes per block (15; 12 folded) go, exact. With SANA_PREDIV_FOLD's "outm" both attention outputs
# take the merged linear's pre-division (attn1 on its reciprocals, attn2 on its unpack mask). Names: {pre}b<k>.qx (W1
# carries the step's gate) and {pre}b<k>.outm; their bands: qx's, and out1's + out2's
ATTN_MERGE = os.environ.get("SANA_ATTN_MERGE", "0") not in ("", "0")


def ff_perm():
    """Destination channel (12 blocks) of conv_inverted's 5760 outputs: the value half on blocks 0-5, the gate half
    on blocks 6-11."""
    c = np.arange(2 * MLP)
    return np.where(c < MLP, c, c - MLP + MLP_PAD)


def fold_block(W, b, r, nl, pre="") -> BlockWeights:
    """Block b's linears for the blocked dataflow, from ref.SanaWeights (float64 numpy) at its step's time (`pre`
    names the linears the time enters: hf.step_tag); the producers of the ReLU / SiLU inputs carry the unit forms'
    input scales from `nl` (nl.PolyNL; times 1/2: the split doubles)."""
    t = lambda name: W[f"transformer_blocks.{b}.{name}"].numpy()            # noqa: E731
    D = L.D
    r = np.asarray(r, dtype=np.float64)
    sh1, sc1, g1, sh2, sc2, g2 = (m.numpy() for m in W.modulation(b))
    H3 = 3 * NCH
    uq = 0.5 * nl.relu_unit_scale(f"b{b}.la.q")
    uk = 0.5 * nl.relu_unit_scale(f"b{b}.la.k")
    u1 = 0.5 * nl.silu_unit_scale(f"b{b}.ff.act1")
    u2 = 0.5 * nl.silu_unit_scale(f"b{b}.ff.act2")
    # attn1 q / k / v from 2 LN(x): the modulation on the rows, norm_q.weight on q's columns
    # k carries norm_k.weight (and the unit scale): its ReLU reads the projection directly, the RMS scale after
    M = np.concatenate([t("attn1.to_q.weight").T * t("attn1.norm_q.weight") * uq,
                        t("attn1.to_k.weight").T * t("attn1.norm_k.weight") * uk, t("attn1.to_v.weight").T], 1)
    cols = np.concatenate([np.arange(D), H3 + np.arange(D), 2 * H3 + np.arange(D)])
    qkv = Lin(f"{pre}b{b}.qkv", _pad((1 + sc1)[:, None] * M, H3, 3 * H3, None, cols), _padv(sh1 @ M, 3 * H3, cols), 0.5)
    out1 = Lin(f"{pre}b{b}.out1", _pad(t("attn1.to_out.0.weight").T * (g1 / r), H3, H3),
               _padv(t("attn1.to_out.0.bias") * g1 / r, H3))
    xp = KN.xhead_perm()
    wq2 = t("attn2.norm_q.weight")
    qx = Lin(f"b{b}.qx", _pad(r[:, None] * t("attn2.to_q.weight").T * wq2, H3, 4 * NCH, None, xp),
             _padv(t("attn2.to_q.bias") * wq2, 4 * NCH, xp))
    out2 = Lin(f"b{b}.out2", _pad(t("attn2.to_out.0.weight").T / r, 4 * NCH, H3, xp, None),
               _padv(t("attn2.to_out.0.bias") / r, H3))
    fp = ff_perm()
    Wi = t("ff.conv_inverted.weight")[:, :, 0, 0].T                          # (D, 5760)
    ff_in = Lin(f"{pre}b{b}.ff_in", _pad((1 + sc2)[:, None] * Wi * u1, H3, 12 * NCH, None, fp),
                _padv((sh2 @ Wi + t("ff.conv_inverted.bias")) * u1, 12 * NCH, fp), 0.5)
    ff_out = Lin(f"{pre}b{b}.ff_out", _pad(t("ff.conv_point.weight")[:, :, 0, 0].T * (g2 / r), 6 * NCH, H3),
                 np.zeros(H3), 0.5)                                     # its input: 2 h silu(g)
    dw = np.zeros((12 * NCH, 3, 3))
    dw[fp] = t("ff.conv_depth.weight")[:, 0]
    dwb = _padv(t("ff.conv_depth.bias"), 12 * NCH, fp)
    dw[6 * NCH:] *= u2                                                    # the gate half feeds act2
    dwb[6 * NCH:] *= u2
    qxm = outm = None
    if ATTN_MERGE:
        qxm = Lin(f"{pre}b{b}.qx", np.vstack([qx.W, out1.W @ qx.W]), qx.b + out1.b @ qx.W)
        outm = Lin(f"{pre}b{b}.outm", np.vstack([out1.W, out2.W]), out1.b + out2.b)
        qxm.row_div = [(slice(H3, 2 * H3), outm)]       # attn1's output arrives divided by outm's pre-division
    return BlockWeights(qkv, out1, qx, out2, ff_in, ff_out, t("attn1.norm_k.weight") * uk, wq2, dw, dwb, r, qxm, outm)


def text_keys(W, b, txt, enc, r=None, msc=1.0):
    """Block b's cross-attention key / value rows from the text rows txt (n, D) (float64, the plaintext side of the
    mirror): (Kb, Vb) per key in x-head layout, lane-broadcast; rows not in `enc` (BOS) as public Pt. `msc`: the keys
    divided by it (kernels.xa_scale's M)."""
    t = lambda name: W[f"transformer_blocks.{b}.{name}"].numpy()            # noqa: E731
    k = txt @ t("attn2.to_k.weight").T + t("attn2.to_k.bias")
    v = txt @ t("attn2.to_v.weight").T + t("attn2.to_v.bias")
    k = k / np.sqrt((k * k).mean(-1, keepdims=True) + 1e-5) * t("attn2.norm_k.weight") / (np.sqrt(KN.DX) * msc)
    xp = KN.xhead_perm()
    Kb, Vb = [], []
    for j in range(len(txt)):
        kr = KN.bcast_rows(_padv(k[j], 4 * NCH, xp), 4)
        vr = KN.bcast_rows(_padv(v[j], 4 * NCH, xp), 4)
        if enc[j]:
            Kb.append([x.astype(complex) for x in kr])
            Vb.append([x.astype(complex) for x in vr])
        else:
            Kb.append([KN.Pt(x, (f"b{b}.k", j, i)) for i, x in enumerate(kr)])
            Vb.append([KN.Pt(x, (f"b{b}.v", j, i)) for i, x in enumerate(vr)])
    return Kb, Vb


def _unscale(rt, X, r, site):
    if np.all(r == 1.0):
        return X
    rv = _padv(r, 3 * NCH)
    return [[rt.mult_mask(x, (site, "unscale", k), lambda k=k: L.lane_vec(np.ones(L.LANES), rv[k * NCH:(k + 1) * NCH]))
             for k, x in enumerate(row)] for row in X]


def _silu_blocks(rt, U, nl, site, in_room):
    """SiLU on both real halves of each ciphertext, their refreshes joint (nl.PolyNL.silu_u2), joined back."""
    return [[join(rt.ops, *nl.silu_u2(rt, x, site, room=in_room)) for x in row] for row in U]


def block(rt, X, bw: BlockWeights, Kb, Vb, nl, lin, b):
    """One SANA block on the residual X[g][k] (3 blocks, carried as x / r)."""
    from perseus.impl.blocked import norm_blocked
    ops = rt.ops
    _d, B, ch = L.geometry()
    site = f"b{b}"
    with rt.step("ln1"):
        N1 = norm_blocked(rt, _unscale(rt, X, bw.r, site + ".ln1"), nl.ln(site + ".ln1"), ch, B)   # 2 LN
    with rt.step("qkv"):
        QKV = lin(rt, N1, bw.qkv)
    Q = [row[0:3] for row in QKV]
    K = [row[3:6] for row in QKV]
    V = [row[6:9] for row in QKV]
    if bw.outm is not None:                  # SANA_ATTN_MERGE
        from perseus.impl.pcmm import hint_entry
        div = 1.0 / lin.in_div(bw.outm) if hasattr(lin, "in_div") else 1.0
        with rt.step("attn1"):
            O = KN.linear_attention(rt, Q, K, V, bw.wk1, nl, site + ".la", nl.la_delta(site), out_scale=div)
        O = [[hint_entry(rt, o, L.N // NCH) for o in row] for row in O]       # read by both merged linears: hinted once
        with rt.step("qx"):
            Qx = lin(rt, [rx + ro for rx, ro in zip(X, O)], bw.qxm)
        with rt.step("attn2"):
            O2 = KN.cross_attention(rt, Qx, Kb, Vb, bw.wq2, nl, site + ".xa", out_scale=div)
        with rt.step("out2"):
            A = lin(rt, [ro + ro2 for ro, ro2 in zip(O, O2)], bw.outm)
        X = [[ops.add(x, a) for x, a in zip(rx, ra)] for rx, ra in zip(X, A)]
    else:
        with rt.step("attn1"):
            O = KN.linear_attention(rt, Q, K, V, bw.wk1, nl, site + ".la", nl.la_delta(site),
                                    out_scale=1.0 / lin.in_div(bw.out1) if hasattr(lin, "in_div") else 1.0)
        with rt.step("out1"):
            A = lin(rt, O, bw.out1)
        X = [[ops.add(x, a) for x, a in zip(rx, ra)] for rx, ra in zip(X, A)]
        with rt.step("qx"):
            Qx = lin(rt, X, bw.qx)
        with rt.step("attn2"):
            O = KN.cross_attention(rt, Qx, Kb, Vb, bw.wq2, nl, site + ".xa")
        with rt.step("out2"):
            A = lin(rt, O, bw.out2)
        X = [[ops.add(x, a) for x, a in zip(rx, ra)] for rx, ra in zip(X, A)]
    with rt.step("ln2"):
        N2 = norm_blocked(rt, _unscale(rt, X, bw.r, site + ".ln2"), nl.ln(site + ".ln2"), ch, B)
    with rt.step("ff_in"):
        U = lin(rt, N2, bw.ff_in)
    with rt.step("act1"):
        # from the PCMM's landing, no hint after it: the single series leaves the conv under the refresh envelope,
        # the plan refreshes after it
        U = _silu_blocks(rt, U, nl, site + ".ff.act1", in_room=PCMM_ROOM)
    with rt.step("dw"):
        U = KN.depthwise3x3(rt, U, bw.dw_w, bw.dw_b, site + ".dw")
    ff_div = 1.0 / lin.in_div(bw.ff_out) if hasattr(lin, "in_div") else 1.0
    with rt.step("glu"):
        # the gate after the conv (a refresh's room less the conv's product), h u taken before its series ends
        P = [[join(ops, *nl.silu_u2(rt, row[6 + k], site + ".ff.act2", room=REFRESH_ROOM - 1,
                                    mul=split(ops, row[k]), out_scale=ff_div)) for k in range(6)] for row in U]
    with rt.step("ff_out"):
        F = lin(rt, P, bw.ff_out)
    return [[ops.add(x, f) for x, f in zip(rx, rf)] for rx, rf in zip(X, F)]


def out_lin(W, c=1.0, name="out"):
    """norm_out's modulation and proj_out on 2 LN(x), times c: 3 blocks in, 1 block out (32 channels)."""
    shift, scale = (v.numpy() for v in W.out_modulation())
    Wp = W["proj_out.weight"].numpy().T * c                            # (D, 32)
    return Lin(name, _pad((1 + scale)[:, None] * Wp, 3 * L.NCH, L.NCH),
               _padv(shift @ Wp + W["proj_out.bias"].numpy() * c, L.NCH), 0.5)
