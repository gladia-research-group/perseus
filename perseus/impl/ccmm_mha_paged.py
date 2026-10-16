"""ccmm_mha's CCMM attention over several 128-token pages of a bidirectional prompt (ViT-B/16: 197 tokens = 128 + 69):
every query page against every key page of the same layer, the softmax's row sums added over the key pages before
each reciprocal, the P.V products added (merged, before the unmerge) over them. Keys past ch.n_keys are masked out;
kc = ch.n_keys on every row. Same switches as ccmm_mha (PF_CCMM_MERGE, PF_SM_UNPACK_FOLD, PF_SM_R_*, PF_SM_PAIR*)."""
from __future__ import annotations

import math

import numpy as np

from . import ccmm as cm
from . import sinc as sc
from . import sinc_mha as smh
from . import sinc_softmax as ss
from . import smcheb
from .attention import _recip_rounds
from .ccmm_mha import KMAX, QMAX, SHIFT_PV, SHIFT_QK, UNPACK_FOLD, VMAX, _halves
from .poly import eval_chebyshev, im_cleanse
from .prefill import Chunks, _KcExtrapolated, split

PAGE = 128


def _acc(rt, acc, xs):
    if acc is None:
        return list(xs)
    return [rt.ops.add(a, x) for a, x in zip(acc, xs)]


def softmax_pages(rt, Spks, cfg, valids, kc, tag, layer=0, prescaled=False):
    """ccmm_mha.softmax over key blocks Spks[j] (16 packed score ciphertexts each, validity valids[j][o]): the
    denominator sums every block; returns the unpacked P per block (PF_SM_UNPACK_FOLD form) or the packed Y."""
    ops = rt.ops
    mean = (cfg.clip_hi + cfg.clip_lo) / 2.0
    sf = 2.0 ** (-cfg.log2delta1 - cfg.log2delta2)
    rowv = np.zeros(sc.SLOTS, dtype=bool)
    for vj in valids:
        for v in vj:
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
    # a score ciphertext whose 8 key columns are all padding is identically 0: skipped (its refreshes would have
    # nothing to refresh), its P is a stand-in the P.V meets with zeroed value rows
    live = [[o for o in range(len(Spk)) if valids[j][o].any()] for j, Spk in enumerate(Spks)]
    Z = [[exp_part(Spk[o], valids[j][o], ("smhp", j, o)) for o in live[j]] for j, Spk in enumerate(Spks)]
    cheb = getattr(ops, "sm_cheb_gs", False)
    ks = sorted(set(int(k) for k in kc[rowv])) or [1]
    rnd = {k: (smcheb.rounds(cfg, k) if cheb else _recip_rounds(_KcExtrapolated(cfg), k, ops.sm_gs_first)) for k in ks}
    n_rounds = len(rnd[ks[0]])
    Y = Z
    for r in range(n_rounds):
        Zr = []
        for Yj in Y:
            if r and ss.PAIR and len(Yj) % 2 == 0 and (not ss.PAIR_ROUNDS or r in ss.PAIR_ROUNDS):
                zj = []
                for a, b in zip(Yj[0::2], Yj[1::2]):
                    X = ops.add(a, ops.mult_i(b))
                    ops.bootstrap(X)
                    zj += [ops.square(v) for v in split(ops, X)]
            elif r:
                zj = [ops.square(im_cleanse(ops, y)) for y in Yj]
            else:
                zj = Yj
            Zr.append(zj)
        D = None
        for zj in Zr:
            for z in zj:
                D = z if D is None else ops.add(D, z)
        D = ops.rotate_and_sum(D, 1 << 12, sc.SLOTS)
        # slots of no row sum to 0: a 1 there keeps their reciprocal bounded (P stays 0 there, Z = 0)
        D = rt.add_mask(D, ("smhp.dfill",) + tag, lambda: np.where(rowv, 0.0, 1.0))
        per = [rnd[int(k)][r] if v else rnd[ks[0]][r] for k, v in zip(kc, rowv)]
        av = np.array([p[0] * p[3] for p in per])
        bv = np.array([p[1] * p[3] * p[3] for p in per])
        iters = per[0][2]
        cv = None
        if cheb:
            cv = ([np.array([p[4][0][i] for p in per]) for i in range(iters - 1)], np.array([p[4][1] for p in per]))
        R = ss.recip_slots(rt, D, av, bv, iters, ("smhp", layer, r) + tag, cv)
        if __import__("os").environ.get("PF_MHA_DBG"):
            rv = np.real(ops.decrypt_slots_complex(R))
            print(f"[mhap_dbg] layer {layer} round {r} R valid max {np.abs(rv[rowv]).max():.4g} invalid max {np.abs(rv[~rowv]).max():.3g}", flush=True)
        if UNPACK_FOLD and r == n_rounds - 1:
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
            P = [[ops.add(ops.mult(rt.mult_mask(z, mk, lambda: sc_r * ss.LO), R),
                          ops.mult_i(ops.mult(rt.mult_mask(ops.rotate(z, 16), mk, lambda: sc_r * ss.LO), Rh)))
                  for z in zj] for zj in Zr]
            return [_full(ops, pj, lj, len(Spk)) for pj, lj, Spk in zip(P, live, Spks)]
        Y = [[ops.mult(z, R) for z in zj] for zj in Zr]
    return [_full(ops, yj, lj, len(Spk)) for yj, lj, Spk in zip(Y, live, Spks)]


def _full(ops, xs, live, n):
    """The live ciphertexts at their indices, a copy of the first live one elsewhere."""
    out = [None] * n
    for o, x in zip(live, xs):
        out[o] = x
    return [x if x is not None else ops.copy(xs[0]) for x in out]


def _key_rows(n_keys):
    """K / V SinC operand slots on a real key row (row = brev6(lane) + 64 chunk: token 64 chunk + lane)."""
    row = sc.slot_coords()[0]
    return ((64 * (row >> 6) + sc.brev(row & 63, 6)) < n_keys).astype(float)


def attention(rt, Q, K, V, cfg, ch: Chunks, layer=0, h_real=12):
    """Q[g], K[g], V[g]: page g's blocked complex ciphertexts (scale 1/2), one per 512-channel block; returns O[g]
    (blocked, scale 1)."""
    ops = rt.ops
    if not ch.bidir or ch.t * 2 != PAGE:
        raise ValueError("ccmm_mha_paged: a bidirectional prompt in 64-lane chunk pairs")
    eng = cm.engine(rt)
    eng.prepare(rt.dims.dim)
    pages = len(Q)
    nb = len(Q[0])
    gscale = 1.0 / math.sqrt(smh.DH)
    merge = cm.MERGE
    al = ss.score_affine(cfg)[0]
    n = ch.n_keys
    n_pg = [min(PAGE, max(0, ch.T - PAGE * g)) for g in range(pages)]          # tokens per page
    k_pg = [min(PAGE, max(0, n - PAGE * g)) for g in range(pages)]             # real keys per page
    with rt.step("qkt"):
        Ks, Vs = [], []
        for g in range(pages):
            Kr = [list(_halves(ops, x)) for x in K[g]]
            Vr = [list(_halves(ops, x)) for x in V[g]]
            kj, vj = smh.to_sinc_kv(rt, Kr), smh.to_sinc_kv(rt, Vr)
            if k_pg[g] < PAGE:          # padded keys: their value rows zeroed (P's stand-ins meet them)
                vj = [rt.mult_mask(x, ("smhp.vrows", k_pg[g]), lambda g=g: _key_rows(k_pg[g])) for x in vj]
            if merge:
                kj, vj = sc.dup(rt, kj), sc.dup(rt, vj)
            Ks.append(kj)
            Vs.append(vj)
        Qs = []
        for g in range(pages):
            q = smh.to_sinc_q(rt, Q[g])
            Qs.append(sc.dup(rt, q) if merge else q)
        S = [[None] * pages for _ in range(pages)]
        for i in range(pages):
            for j in range(pages):
                s = eng.apply([ops.copy(x) for x in Qs[i]], [ops.copy(x) for x in Ks[j]], b_transposed=True,
                              amax=QMAX, bmax=KMAX, shift=SHIFT_QK, out_scale=gscale, site=f"mqk{layer}.{i}.{j}",
                              merge=merge)
                if merge:
                    s = cm.unmerge(rt, s, eng.w, scale=al)
                S[i][j] = ss.pack(rt, s)
    outs = []
    for i in range(pages):
        tq, tk = smh.packed_meta(n_pg[i], h_real)
        qv = tq >= 0
        valids = [[qv & (tk[o] < k_pg[j]) for o in range(sc.NCT)] for j in range(pages)]
        kc = np.where(qv, n, 1)
        tag = ("mhap", ch.T, n, h_real, i)
        with rt.step("softmax"):
            P = softmax_pages(rt, S[i], cfg, valids, kc, tag, layer=layer, prescaled=merge)
            if not UNPACK_FOLD:
                P = [ss.unpack(rt, p) for p in P]
        with rt.step("softmax_v"):
            O = None
            for j in range(pages):
                pj = sc.dup(rt, P[j]) if merge else P[j]
                O = _acc(rt, O, eng.apply(pj, [ops.copy(x) for x in Vs[j]], b_transposed=False, amax=1.5, bmax=VMAX,
                                          shift=SHIFT_PV, site=f"mpv{layer}.{i}.{j}", merge=merge))
            if merge:
                O = cm.unmerge(rt, O, eng.w)
            outs.append(smh.from_sinc_q(rt, O, nb))
    return outs
