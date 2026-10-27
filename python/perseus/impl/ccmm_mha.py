"""CCMM attention for multi-head attention with d_head = 64 (BERT / ViT-base geometry; PF_ATTN=ccmm under a model whose
blocked layout holds 8 heads of 64 per 512-channel block): Q K^T and P V as CKL batch CCMMs (ccmm.py) on the SinC
packing of sinc_mha.py, the softmax on the SinC layout with per-slot validity (bidirectional: every key < n, no
causal triangle; kc = n), one 128-token chunk pair. The softmax mirrors sinc_softmax.softmax (same switches:
PF_SM_UNPACK_FOLD, PF_SM_R_REFRESH / R_SCALE / R_ITER, PF_SM_PAIR / PAIR_ROUNDS, PF_CCMM_MERGE)."""
from __future__ import annotations

import math
import os

import numpy as np

from . import ccmm as cm
from . import sinc as sc
from . import sinc_mha as smh
from . import sinc_softmax as ss
from . import smcheb
from .attention import _recip_rounds, uncentred_only
from .poly import eval_chebyshev, im_cleanse
from .prefill import Chunks, _KcExtrapolated, split

SHIFT_QK = int(os.environ.get("PF_CCMM_MHA_SHIFT_QK", "6"))
SHIFT_PV = int(os.environ.get("PF_CCMM_MHA_SHIFT_PV", "3"))
QMAX = float(os.environ.get("PF_CCMM_MHA_QMAX", "6"))
KMAX = float(os.environ.get("PF_CCMM_MHA_KMAX", "12"))
VMAX = float(os.environ.get("PF_CCMM_MHA_VMAX", "6"))
UNPACK_FOLD = os.environ.get("PF_SM_UNPACK_FOLD", "0") not in ("", "0")


_DBG = os.environ.get("PF_MHA_DBG", "0") not in ("", "0")


def _dbg(rt, name, cts):
    if _DBG:
        v = np.array([rt.ops.decrypt_slots_complex(x) for x in cts if x is not None])
        print(f"[mha_dbg] {name}: finite {bool(np.isfinite(v).all())} max {np.nanmax(np.abs(v)):.4g}", flush=True)


def _halves(ops, X):
    ops.bootstrap_hint(X, ops.headroom(2), True)
    return split(ops, X)


def softmax(rt, Spk, cfg, valid, kc, tag, layer=0, unpacked=False, prescaled=False):
    """sinc_softmax.softmax with the validity per (ciphertext, slot) `valid[o]` and the key count per slot `kc`
    given; `tag` names the masks of this geometry."""
    uncentred_only(cfg, "ccmm_mha.softmax")
    ops = rt.ops
    mean = (cfg.clip_hi + cfg.clip_lo) / 2.0
    sf = 2.0 ** (-cfg.log2delta1 - cfg.log2delta2)
    rowv = np.zeros(sc.SLOTS, dtype=bool)
    for v in valid:
        rowv |= v

    def exp_part(s, vo, site):
        if prescaled:
            al, be = ss.score_affine(cfg)
            s = rt.add_mask(s, site + ("score", cfg.clip_lo, mean, "pre") + tag,
                            lambda: al * np.where(vo, -mean, cfg.clip_lo - mean) - be)
            z = eval_chebyshev(ops, s, cfg.cheb_coeffs, -1.0, 1.0)
        else:
            s = rt.add_mask(s, site + ("score", cfg.clip_lo, mean) + tag,
                            lambda: np.where(vo, -mean, cfg.clip_lo - mean))
            z = eval_chebyshev(ops, s, cfg.cheb_coeffs, cfg.cheb_a / sf, cfg.cheb_b / sf)
        for _ in range(cfg.log2delta1):
            z = ops.square(z)
        z = im_cleanse(ops, z)
        return rt.mult_mask(z, site + ("active",) + tag, lambda: np.where(vo, 0.5 / kc, 0.0))
    Z = [exp_part(s, valid[o], ("smha", o)) for o, s in enumerate(Spk)]
    cheb = getattr(ops, "sm_cheb_gs", False)
    ks = sorted(set(int(k) for k in kc[rowv])) or [1]
    rnd = {k: (smcheb.rounds(cfg, k) if cheb else _recip_rounds(_KcExtrapolated(cfg), k, ops.sm_gs_first)) for k in ks}
    n_rounds = len(rnd[ks[0]])
    Y = Z
    for r in range(n_rounds):
        if r and ss.PAIR and len(Y) % 2 == 0 and (not ss.PAIR_ROUNDS or r in ss.PAIR_ROUNDS):
            Z = []
            for a, b in zip(Y[0::2], Y[1::2]):
                X = ops.add(a, ops.mult_i(b))
                ops.bootstrap(X)
                Z += [ops.square(v) for v in split(ops, X)]
        elif r:
            Z = [ops.square(im_cleanse(ops, y)) for y in Y]
        D = Z[0]
        for z in Z[1:]:
            D = ops.add(D, z)
        D = ops.rotate_and_sum(D, 1 << 12, sc.SLOTS)
        # slots of no row (odd batch slots, padded heads, unused queries) sum to 0: a 1 there keeps their
        # reciprocal bounded (R's refresh reads every slot), P stays 0 (Z = 0)
        D = rt.add_mask(D, ("smha.dfill",) + tag, lambda: np.where(rowv, 0.0, 1.0))
        per = [rnd[int(k)][r] if v else rnd[ks[0]][r] for k, v in zip(kc, rowv)]
        av = np.array([p[0] * p[3] for p in per])
        bv = np.array([p[1] * p[3] * p[3] for p in per])
        iters = per[0][2]
        cv = None
        if cheb:
            cv = ([np.array([p[4][0][j] for p in per]) for j in range(iters - 1)], np.array([p[4][1] for p in per]))
        R = ss.recip_slots(rt, D, av, bv, iters, ("smha", layer, r) + tag, cv)
        if _DBG:
            dv, rv = np.real(ops.decrypt_slots_complex(D)), np.real(ops.decrypt_slots_complex(R))
            print(f"[mha_dbg] round {r}: D valid [{dv[rowv].min():.4g}, {dv[rowv].max():.4g}] invalid max "
                  f"{np.abs(dv[~rowv]).max():.3g} | R*D valid [{(rv * dv)[rowv].min():.4g}, {(rv * dv)[rowv].max():.4g}]"
                  f" R invalid max {np.abs(rv[~rowv]).max():.3g} | R valid max {np.abs(rv[rowv]).max():.4g} | layer {layer}"
                  f" rows {int(rowv.sum())}", flush=True)
        if unpacked and r == n_rounds - 1:
            sc_r = ss.R_SCALE if ss.R_REFRESH else 1.0
            if ss.R_REFRESH:
                if sc_r != 1.0:
                    R = ops.mult(R, 1.0 / sc_r)
                if ss.R_ITER:
                    y = ops.copy(R)
                    ops.bootstrap_real(y)
                    e = ops.add(ops.mult(ops.sub(R, y), float(2 ** ss.R_ITER_P)), 0.25)
                    ops.bootstrap_real(e)
                    R = ops.add(y, ops.mult(ops.add(e, -0.25), float(2.0 ** -ss.R_ITER_P)))
                else:
                    ops.bootstrap_real(R)
            Rh = ops.rotate(R, 16)
            mk = ("sm.unpack",) if sc_r == 1.0 else ("sm.unpack", sc_r)
            return [ops.add(ops.mult(rt.mult_mask(z, mk, lambda: sc_r * ss.LO), R),
                            ops.mult_i(ops.mult(rt.mult_mask(ops.rotate(z, 16), mk, lambda: sc_r * ss.LO), Rh)))
                    for z in Z]
        Y = [ops.mult(z, R) for z in Z]
    return Y


def attention(rt, Q, K, V, cfg, ch: Chunks, layer=0, h_real=12):
    """Q[0], K[0], V[0]: the blocked complex ciphertexts (scale 1/2) of one 128-token chunk pair, one per 512-channel
    block; returns [O] (blocked, scale 1). ch.bidir with ch.n_keys real keys."""
    ops = rt.ops
    if len(Q) != 1 or ch.n_chunks > 2 or not ch.bidir:
        raise ValueError("ccmm_mha: one bidirectional chunk pair")
    eng = cm.engine(rt)
    eng.prepare(rt.dims.dim)
    nb = len(Q[0])
    gscale = 1.0 / math.sqrt(smh.DH)
    merge = cm.MERGE
    al = ss.score_affine(cfg)[0]
    with rt.step("qkt"):
        Kr = [list(_halves(ops, x)) for x in K[0]]
        Vr = [list(_halves(ops, x)) for x in V[0]]
        _dbg(rt, "Q", Q[0]); _dbg(rt, "K", [x for r in Kr for x in r])
        Qs = smh.to_sinc_q(rt, Q[0])
        Ks = smh.to_sinc_kv(rt, Kr)
        _dbg(rt, "Qs", Qs); _dbg(rt, "Ks", Ks)
        if merge:
            Qs, Ks = sc.dup(rt, Qs), sc.dup(rt, Ks)
        # Q at 1/2, K at 1: S = q k gscale / 2 (pack's 2 Re restores true units)
        S = eng.apply(Qs, Ks, b_transposed=True, amax=QMAX, bmax=KMAX, shift=SHIFT_QK, out_scale=gscale,
                      site=f"mqk{layer}", merge=merge)
        if merge:
            S = cm.unmerge(rt, S, eng.w, scale=al)
        _dbg(rt, "S", S)
    tq, tk = smh.packed_meta(ch.T, h_real)
    qv = tq >= 0
    n = ch.n_keys
    valid = [qv & (tk[o] < n) for o in range(sc.NCT)]
    kc = np.where(qv, n, 1)
    tag = ("mha", ch.T, n, h_real)
    with rt.step("softmax"):
        P = softmax(rt, ss.pack(rt, S), cfg, valid, kc, tag, layer=layer, unpacked=UNPACK_FOLD, prescaled=merge)
        if not UNPACK_FOLD:
            P = ss.unpack(rt, P)
        _dbg(rt, "P", P)
    with rt.step("softmax_v"):
        Vs = smh.to_sinc_kv(rt, Vr)
        if merge:
            P, Vs = sc.dup(rt, P), sc.dup(rt, Vs)
        O = eng.apply(P, Vs, b_transposed=False, amax=1.5, bmax=VMAX, shift=SHIFT_PV, site=f"mpv{layer}", merge=merge)
        if merge:
            O = cm.unmerge(rt, O, eng.w)
        _dbg(rt, "O", O)
        out = smh.from_sinc_q(rt, O, nb)
        _dbg(rt, "out", out)
    return [out]
