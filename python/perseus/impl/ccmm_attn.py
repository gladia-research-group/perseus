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
# PF_SM_HALF_PACK=1 (the half-head view, with PF_SM_UNPACK_FOLD): the softmax on the 16 score ciphertexts compressed to
# 4 (sinc_softmax.compress_half; its first halves fill 8 of the 32 batch slots)
HALF_PACK = os.environ.get("PF_SM_HALF_PACK", "0") not in ("", "0")
# PF_CCMM_CAUSAL=1 (the half-head view, merged): every CCMM score off its causal keys zeroed on the unmerge's masks, so
# the exp's series reads only in-domain inputs (a non-causal product plus the invalid-key offset lands past -1: its
# Chebyshev terms grow to ~10^2 and every refresh in the softmax pays for that range)
CAUSAL = os.environ.get("PF_CCMM_CAUSAL", "0") not in ("", "0")


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
    """Llama's geometry (4 heads of D per block, 8 KV heads, GQA 4), or the half-head view of d_head = 2 D under GQA 2
    (Gemma-2: sinc.half_perm; the caller's Q / K / V / O and prefixes in that channel order, Q blocks 0..3)."""
    ops = rt.ops
    g0 = sub_rt(rt, B, 0)
    half = g0.dims.d_head == 2 * sc.D
    nq = len(Q[0]) if Q else 0
    if len(Q) != 1 or ch.n_chunks != 2 or B.Kkv != 2 or B.K != 8 or ch.T != 127 or nq != (4 if half else 8):
        raise ValueError("attention_ccmm: one ciphertext pair of BOS + 127 tokens, Llama blocked GQA geometry (or its "
                         "half-head view)")
    eng = cm.engine(rt)
    eng.prepare(rt.dims.dim)
    gscale = 1.0 / math.sqrt(g0.dims.d_head_real)
    perm = sc.half_perm_slots(B.t, B.nch) if half else None
    reps = (0,) if half else (0, 1)
    if half and prefixes is not None:
        prefixes = [(np.asarray(k)[perm], np.asarray(v)[perm]) for k, v in prefixes]
    kb, vb = _bos(prefixes)
    with rt.step("qkt"):
        Qc = []
        for x in Q[0]:
            a, b = _halves(ops, x)
            if rope is not None:
                a = rope_lanes(g0, a, rope, ch.lanes(0), "pf.rope.q", perm)
                b = rope_lanes(g0, b, rope, ch.lanes(1), "pf.rope.q", perm)
            Qc.append(join(ops, a, b))
        Kr, Vr = [], []
        for bb in range(B.Kkv):
            k0, k1 = _halves(ops, K[0][bb])
            if rope is not None:
                k0 = rope_lanes(g0, k0, rope, ch.lanes(0), "pf.rope.k", perm)
                k1 = rope_lanes(g0, k1, rope, ch.lanes(1), "pf.rope.k", perm)
            Kr.append([k0, k1])
            Vr.append(list(_halves(ops, V[0][bb])))
            if kv is not None:
                kv[bb]["K"], kv[bb]["V"] = Kr[-1], Vr[-1]
        Qs = sc.to_sinc_q(rt, Qc)
        Ks = sc.to_sinc_kv(rt, Kr, reps)
        if cfg.ref_keys:
            # a softmax centred on key 0: every prompt key row minus the (public) BOS key, the BOS row left 0, so
            # q.K^T returns s - s_BOS (no level: a plaintext sum)
            kpat = sc.kv_rows_pattern(-kb, range(ch.T), len(reps))
            Ks = [rt.add_mask(x, ("ccmm.kref", layer, o), lambda o=o: kpat[o]) for o, x in enumerate(Ks)]
        else:
            kpat = sc.kv_row_pattern(kb, n_rep=len(reps))
            Ks = [rt.add_mask(x, ("ccmm.kbos", layer, o), lambda o=o: kpat[o]) for o, x in enumerate(Ks)]
        mn = cm.MERGE_N
        if mn == 4 and not half:
            raise ValueError("PF_CCMM_MERGE=4: the half-head view's operands alone sit on l < 8")
        if cm.MERGE:
            Qs, Ks = sc.dup(rt, Qs, mn), sc.dup(rt, Ks, mn)
        # the half view, merged: the exp's domain scale alpha in the CCMM's own output scale (its operand factors, exact),
        # so the unmerge's masks are the same for every layer (the causal ones: 16 outputs x 4 n terms, shared)
        al = sm.score_affine(cfg)[0] if half and cm.MERGE else 1.0
        S = eng.apply(Qs, Ks, b_transposed=True, amax=QMAX, bmax=KMAX, shift=SHIFT_QK, out_scale=0.5 * gscale * al,
                      site=f"qk{layer}", merge=mn)
        if cm.MERGE:        # half view: a head's scores = its halves' sum, on the first halves' batch slots alone
            if half:        # PF_CCMM_CAUSAL: the non-causal products zeroed on the unmerge's masks (no level)
                S = cm.unmerge_half(rt, S, eng.w, scale=1.0, n=mn, T=ch.T if CAUSAL else None)
            else:
                S = cm.unmerge(rt, S, eng.w, scale=sm.score_affine(cfg)[0], n=mn)
        elif half:
            S = [rt.mult_mask(ops.add(s, ops.rotate(s, sc.HALF_STEP)), ("ccmm.half",), lambda: sm.LO_HALF) for s in S]
    lo = sm.LO_HALF if half else None
    with rt.step("softmax"):
        if UNPACK_FOLD:
            P = sm.softmax(rt, sm.pack(rt, S), cfg, layer=layer, T=ch.T, unpacked=True, prescaled=cm.MERGE, lo=lo,
                           half=half and HALF_PACK)
        else:
            P = sm.unpack(rt, sm.softmax(rt, sm.pack(rt, S), cfg, layer=layer, T=ch.T, prescaled=cm.MERGE), lo=lo)
        if half:            # both halves' P.V read the head's P
            P = [ops.add(p, ops.rotate(p, -sc.HALF_STEP)) for p in P]
    with rt.step("softmax_v"):
        Vs = sc.to_sinc_kv(rt, Vr, reps)
        vpat = sc.kv_row_pattern(vb, n_rep=len(reps))
        Vs = [rt.add_mask(x, ("ccmm.vbos", layer, o), lambda o=o: vpat[o]) for o, x in enumerate(Vs)]
        if cm.MERGE:
            P, Vs = sc.dup(rt, P, mn), sc.dup(rt, Vs, mn)
        O = eng.apply(P, Vs, b_transposed=False, amax=1.5, bmax=VMAX, shift=SHIFT_PV, site=f"pv{layer}", merge=mn)
        if mn == 4:         # the general unmerge, then the inverse stage B (its masks do not fold the quadruples)
            out = sc.from_sinc_q(rt, cm.unmerge(rt, O, eng.w, n=4), nq)
        else:
            out = sc.from_sinc_q_merged(rt, O, cm.w_slots(eng.w), nq) if cm.MERGE else sc.from_sinc_q(rt, O, nq)
    return [out]
