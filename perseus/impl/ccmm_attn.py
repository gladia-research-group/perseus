"""Blocked-GQA prefill attention through the CCMM (PF_ATTN=ccmm): Q K^T and P V as batch ciphertext-ciphertext
matrix products (ccmm.py) on the SinC layout (sinc.py), the softmax there (sinc_softmax.py). One 128-token ciphertext
pair (BOS + 127 prompt tokens); the BOS key / value sit on key token sinc.BOS_TOK. RoPE and the per-chunk real K / V
(the decode hand-off's) are prefill_attention's."""
from __future__ import annotations

import math
import os

import numpy as np

from . import ccmm as cm
from . import sinc as sc
from . import sinc_softmax as sm
from .blocked import Blocks, sub_rt
from .prefill import Chunks, join, rope_lanes, split

SHIFT_QK = int(os.environ.get("PF_CCMM_SHIFT_QK", "5"))
SHIFT_PV = int(os.environ.get("PF_CCMM_SHIFT_PV", "3"))
QMAX = float(os.environ.get("PF_CCMM_QMAX", "4"))
KMAX = float(os.environ.get("PF_CCMM_KMAX", "24"))
VMAX = float(os.environ.get("PF_CCMM_VMAX", "8"))
# PF_SM_UNPACK_FOLD=1: the unpack masks on the softmax's last Z (P one level shallower)
UNPACK_FOLD = os.environ.get("PF_SM_UNPACK_FOLD", "0") not in ("", "0")


def _halves(ops, X):
    ops.bootstrap_hint(X, ops.headroom(2), True)
    return split(ops, X)


def _bos(prefixes):
    """prefixes[b] = (k, v) slot vectors of KV block b (feature j 4 + m on lanes) -> (k[kv][dim], v[kv][dim])."""
    ks, vs = np.zeros((8, sc.D)), np.zeros((8, sc.D))
    for b, (k, v) in enumerate(prefixes):
        kk = np.asarray(k).reshape(sc.D, 4, 64)[:, :, 0]
        vv = np.asarray(v).reshape(sc.D, 4, 64)[:, :, 0]
        for m in range(4):
            ks[4 * b + m], vs[4 * b + m] = kk[:, m], vv[:, m]
    return ks, vs


def attention_ccmm(rt, Q, K, V, cfg, ch: Chunks, B: Blocks, rope=None, prefixes=None, kv=None, layer=0):
    ops = rt.ops
    if len(Q) != 1 or ch.n_chunks != 2 or B.Kkv != 2 or B.K != 8 or ch.T != 127:
        raise ValueError("attention_ccmm: one ciphertext pair of BOS + 127 tokens, Llama blocked GQA geometry")
    eng = cm.engine(rt)
    eng.prepare(rt.dims.dim)
    g0 = sub_rt(rt, B, 0)
    gscale = 1.0 / math.sqrt(g0.dims.d_head_real)
    kb, vb = _bos(prefixes)
    with rt.step("qkt"):
        Qc = []
        for x in Q[0]:
            a, b = _halves(ops, x)
            if rope is not None:
                a = rope_lanes(g0, a, rope, ch.lanes(0), "pf.rope.q")
                b = rope_lanes(g0, b, rope, ch.lanes(1), "pf.rope.q")
            Qc.append(join(ops, a, b))
        Kr, Vr = [], []
        for bb in range(B.Kkv):
            k0, k1 = _halves(ops, K[0][bb])
            if rope is not None:
                k0 = rope_lanes(g0, k0, rope, ch.lanes(0), "pf.rope.k")
                k1 = rope_lanes(g0, k1, rope, ch.lanes(1), "pf.rope.k")
            Kr.append([k0, k1])
            Vr.append(list(_halves(ops, V[0][bb])))
            if kv is not None:
                kv[bb]["K"], kv[bb]["V"] = Kr[-1], Vr[-1]
        Qs = sc.to_sinc_q(rt, Qc)
        Ks = sc.to_sinc_kv(rt, Kr)
        kpat = sc.kv_row_pattern(kb)
        Ks = [rt.add_mask(x, ("ccmm.kbos", layer, o), lambda o=o: kpat[o]) for o, x in enumerate(Ks)]
        if cm.MERGE:
            Qs, Ks = sc.dup(rt, Qs), sc.dup(rt, Ks)
        S = eng.apply(Qs, Ks, b_transposed=True, amax=QMAX, bmax=KMAX, shift=SHIFT_QK, out_scale=0.5 * gscale,
                      site=f"qk{layer}", merge=cm.MERGE)
        if cm.MERGE:
            S = cm.unmerge(rt, S, eng.w, scale=sm.score_affine(cfg)[0])
    with rt.step("softmax"):
        if UNPACK_FOLD:
            P = sm.softmax(rt, sm.pack(rt, S), cfg, layer=layer, T=ch.T, unpacked=True, prescaled=cm.MERGE)
        else:
            P = sm.unpack(rt, sm.softmax(rt, sm.pack(rt, S), cfg, layer=layer, T=ch.T, prescaled=cm.MERGE))
    with rt.step("softmax_v"):
        Vs = sc.to_sinc_kv(rt, Vr)
        vpat = sc.kv_row_pattern(vb)
        Vs = [rt.add_mask(x, ("ccmm.vbos", layer, o), lambda o=o: vpat[o]) for o, x in enumerate(Vs)]
        if cm.MERGE:
            P, Vs = sc.dup(rt, P), sc.dup(rt, Vs)
        O = eng.apply(P, Vs, b_transposed=False, amax=1.5, bmax=VMAX, shift=SHIFT_PV, site=f"pv{layer}", merge=cm.MERGE)
        out = sc.from_sinc_q_merged(rt, O, cm.w_slots(eng.w)) if cm.MERGE else sc.from_sinc_q(rt, O)
    return [out]
