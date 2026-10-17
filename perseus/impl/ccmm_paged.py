"""Paged CCMM prefill attention (PF_PAGED=1): the prompt in pages of one ciphertext pair, page p's queries against
the key blocks of pages 0..p. Page 0 = BOS + 127 tokens (BOS on key token sinc.BOS_TOK), page p >= 1 = 128 tokens.
Every per-block op runs in a repeated step (rt.step(repeat=True)), so the plan of page 0 binds any page count; the
softmax's row sums add over the blocks before each reciprocal and the P.V outputs add after their refreshes."""
from __future__ import annotations

import math

import numpy as np

from . import ccmm as cm
from . import sinc as sc
from . import sinc_softmax as sm
from . import smcheb
from .attention import _recip_rounds
from .blocked import Blocks, sub_rt
from .ccmm_attn import KMAX, QMAX, SHIFT_PV, SHIFT_QK, UNPACK_FOLD, VMAX, _bos, _halves
from .poly import eval_chebyshev, im_cleanse
from .prefill import Chunks, _KcExtrapolated, join, rope_lanes, split

PAGE = 128
# page p refreshes R / (PF_SM_R_SCALE (p + 1)): the last round's |R| grows about linearly with the key count (a
# plaintext constant, the same op stream); PF_PAGED_R_SCALES=f0,f1,... overrides (the last entry for later pages)
_R_SCALES = [float(v) for v in __import__("os").environ.get("PF_PAGED_R_SCALES", "").split(",") if v]


def r_scale(p: int) -> float:
    return _R_SCALES[min(p, len(_R_SCALES) - 1)] if _R_SCALES else float(p + 1)


def page_tokens(p: int) -> int:
    return PAGE - 1 if p == 0 else PAGE


def page_meta(p: int):
    """(tq, tk, query valid, kc): kc = the keys a query row sees over pages 0..p (BOS included)."""
    tq, tk = sc.packed_meta(page_tokens(p))
    qv = tq >= 0
    kc = np.where(qv, PAGE * p + tq + (2 if p == 0 else 1), 1)
    return tq, tk, qv, kc


def _valid(p: int, diag: bool, tq, tk, qv, o: int):
    if not diag:
        return qv
    if p == 0:
        return qv & ((tk[o] == sc.BOS_TOK) | (tk[o] <= tq))
    return qv & (tk[o] <= tq)


_DBG_LAYER = int(__import__("os").environ.get("PF_PAGED_DBG", "-1"))
_RSTAT = __import__("os").environ.get("PF_PAGED_RSTAT", "0") not in ("", "0")
# PF_PAGED_STREAM=1: each block's P is built inside its P.V step and dropped after it (memory flat in the page count)
_STREAM = __import__("os").environ.get("PF_PAGED_STREAM", "0") not in ("", "0")
# PF_PAGED_DEVKV=1: a page's own K / V stay on the device for its diagonal block; host copies only for later pages
_DEVKV = __import__("os").environ.get("PF_PAGED_DEVKV", "0") not in ("", "0")


def _dbg_softmax(rt, Sd, Pd, p, al, tq, tk, qv):
    """PF_PAGED_DBG=layer: the decrypted probabilities vs the exact softmax of the decrypted scores, per block."""
    row, col_lo, l = sc.slot_coords()
    grp = row * 32 + l                                  # one attention row per (row, batch) slot class
    S = np.stack([np.stack(b) for b in Sd]) / al          # (blocks, 16, slots) true scores
    P = np.stack([np.stack(b) for b in Pd])
    valid = np.stack([np.stack([_valid(p, j == p, tq, tk, qv, o) & np.ones(sc.SLOTS, bool) for o in range(sc.NCT)])
                      for j in range(p + 1)])
    E = np.where(valid, np.exp(np.where(valid, S, -np.inf) - np.nanmax(np.where(valid, S, -np.inf))), 0.0)
    tot = np.zeros(128 * 32)
    np.add.at(tot, np.broadcast_to(grp, E.shape).ravel(), E.ravel())
    ex = np.where(valid, E / np.where(tot[grp] > 0, tot[grp], 1.0), 0.0)
    m = valid & (np.broadcast_to(qv, valid.shape))
    c = float(np.vdot(ex[m], P[m]) / np.vdot(ex[m], ex[m]))
    for j in range(p + 1):
        mj = m[j]
        err = np.abs(P[j][mj] - c * ex[j][mj])
        print(f"[paged_dbg] page {p} block {j}: scale {c:.4g} max|p| {np.abs(ex[j][mj]).max():.3g} "
              f"maxerr {err.max() / c:.3g} meanerr {err.mean() / c:.3g} |S| {np.abs(S[j][mj]).max():.3g}", flush=True)


def _acc(rt, acc, xs):
    """acc + xs, element-wise; the adds in their own sub-step (only blocks after the first issue them)."""
    if acc is None:
        return xs
    with rt.step("acc"):
        return [rt.ops.add(a, x) for a, x in zip(acc, xs)]


def attention_paged(rt, Q, K, V, cfg, ch: Chunks, B: Blocks, page: int, store: list, rope=None, prefixes=None,
                    kv=None, layer=0, final=False):
    """`store`: this layer's [(K operands, V operands)] of the pages before `page`; this page's are appended."""
    ops, p = rt.ops, page
    if len(Q) != 1 or ch.n_chunks != 2 or B.Kkv != 2 or B.K != 8 or ch.T != page_tokens(p):
        raise ValueError("attention_paged: one ciphertext pair per page, Llama blocked GQA geometry")
    if not UNPACK_FOLD:
        raise ValueError("attention_paged needs PF_SM_UNPACK_FOLD=1")
    if len(store) != p:
        raise ValueError(f"attention_paged: page {p} with {len(store)} stored pages")
    eng = cm.engine(rt)
    eng.prepare(rt.dims.dim)
    g0 = sub_rt(rt, B, 0)
    gscale = 1.0 / math.sqrt(g0.dims.d_head_real)
    kb, vb = _bos(prefixes)
    merge = cm.MERGE
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
        Ks, Vs = sc.to_sinc_kv(rt, Kr), sc.to_sinc_kv(rt, Vr)
        # the BOS key / value only on page 0; later pages add zeros (the same op stream)
        kpat = sc.kv_row_pattern(kb) if p == 0 else np.zeros((sc.NCT, sc.SLOTS))
        vpat = sc.kv_row_pattern(vb) if p == 0 else np.zeros((sc.NCT, sc.SLOTS))
        Ks = [rt.add_mask(x, ("ccmm.kbos", layer, o, p == 0), lambda o=o: kpat[o]) for o, x in enumerate(Ks)]
        Vs = [rt.add_mask(x, ("ccmm.vbos", layer, o, p == 0), lambda o=o: vpat[o]) for o, x in enumerate(Vs)]
        if merge:
            Qs, Ks, Vs = sc.dup(rt, Qs), sc.dup(rt, Ks), sc.dup(rt, Vs)
        # K / V in the CCMM's input form (4 limbs): every later page reads them as they are
        kw = dict(keep=True, export=not final) if _DEVKV else {}
        store.append((eng.prep_b(Ks, amax=QMAX, bmax=KMAX, shift=SHIFT_QK, out_scale=0.5 * gscale, merge=merge, **kw),
                      eng.prep_b(Vs, amax=1.5, bmax=VMAX, shift=SHIFT_PV, merge=merge, **kw)))

    tq, tk, qv, kc = page_meta(p)
    mean = (cfg.clip_hi + cfg.clip_lo) / 2.0
    sf = 2.0 ** (-cfg.log2delta1 - cfg.log2delta2)
    al, be = sm.score_affine(cfg)

    def exp_part(s, valid, site):
        if merge:           # the scores carry alpha (unmerge's scale): the domain map's product rode its masks
            s = rt.add_mask(s, site + ("score", cfg.clip_lo, mean, "pre"),
                            lambda: al * np.where(valid, -mean, cfg.clip_lo - mean) - be)
            z = eval_chebyshev(ops, s, cfg.cheb_coeffs, -1.0, 1.0)
        else:
            s = rt.add_mask(s, site + ("score", cfg.clip_lo, mean),
                            lambda: np.where(valid, -mean, cfg.clip_lo - mean))
            z = eval_chebyshev(ops, s, cfg.cheb_coeffs, cfg.cheb_a / sf, cfg.cheb_b / sf)
        for _ in range(cfg.log2delta1):
            z = ops.square(z)
        z = im_cleanse(ops, z)
        return rt.mult_mask(z, site + ("active",), lambda: np.where(valid, 0.5 / kc, 0.0))

    Z, Sdbg = [], []
    with rt.step("qkt"):
        for j in range(p + 1):
            diag = j == p
            with rt.step("kb", repeat=True):
                qv_ = [ops.copy(x) for x in Qs]
                S = eng.apply(qv_, store[j][0], b_transposed=True, amax=QMAX, bmax=KMAX, shift=SHIFT_QK,
                              out_scale=0.5 * gscale, site=f"qk{layer}", merge=merge)
                if merge:
                    S = cm.unmerge(rt, S, eng.w, scale=al)
                tag = ("pg", p, diag)
                Spk = sm.pack(rt, S)
                if layer == _DBG_LAYER:
                    Sdbg.append([np.real(ops.decrypt_slots_complex(x)) for x in Spk])
                Z.append([exp_part(s, _valid(p, diag, tq, tk, qv, o), ("sm", o) + tag)
                          for o, s in enumerate(Spk)])

    with rt.step("softmax"):
        cheb = getattr(ops, "sm_cheb_gs", False)
        ks = sorted(set(kc.tolist()))
        rnd = {k: (smcheb.rounds(cfg, k) if cheb else _recip_rounds(_KcExtrapolated(cfg), k, ops.sm_gs_first))
               for k in ks}
        n_rounds = len(rnd[ks[0]])
        Y = Z
        P = None
        for r in range(n_rounds):
            Zr, acc = [], None
            for j in range(p + 1):
                with rt.step(f"r{r}", repeat=True):
                    if r and sm.PAIR and len(Y[j]) % 2 == 0 and (not sm.PAIR_ROUNDS or r in sm.PAIR_ROUNDS):
                        zj = []
                        for a, b in zip(Y[j][0::2], Y[j][1::2]):
                            X = ops.add(a, ops.mult_i(b))
                            ops.bootstrap(X)
                            zj += [ops.square(v) for v in split(ops, X)]
                    elif r:
                        zj = [ops.square(im_cleanse(ops, y)) for y in Y[j]]
                    else:
                        zj = Y[j]
                    Y[j] = None
                    D = zj[0]
                    for z in zj[1:]:
                        D = ops.add(D, z)
                    Zr.append(zj)
                    acc = _acc(rt, acc, [D])
            D = ops.rotate_and_sum(acc[0], 1 << 12, sc.SLOTS)
            per = [rnd[int(k)][r] for k in kc]
            av = np.array([q[0] * q[3] for q in per])
            bv = np.array([q[1] * q[3] * q[3] for q in per])
            iters = per[0][2]
            cv = None
            if cheb:
                cv = ([np.array([q[4][0][i] for q in per]) for i in range(iters - 1)], np.array([q[4][1] for q in per]))
            R = sm.recip_slots(rt, D, av, bv, iters, ("sm", layer, r, "pg", p), cv)
            if layer == _DBG_LAYER or (_RSTAT and r == n_rounds - 1):
                rv = np.real(ops.decrypt_slots_complex(R))[qv]
                print(f"[paged_dbg] layer {layer} page {p} round {r}: max|R| {np.abs(rv).max():.4g} median {np.median(np.abs(rv)):.4g}",
                      flush=True)
            if r == n_rounds - 1:
                sc_r = sm.R_SCALE * r_scale(p) if sm.R_REFRESH else 1.0
                if sm.R_REFRESH:
                    if sc_r != 1.0:
                        R = ops.mult(R, 1.0 / sc_r)
                    if sm.R_ITER:
                        y = ops.copy(R)
                        ops.bootstrap_real(y)
                        e = ops.add(ops.mult(ops.sub(R, y), float(2 ** sm.R_ITER_P)), 0.25)
                        ops.bootstrap_real(e)
                        R = ops.add(y, ops.mult(ops.add(e, -0.25), float(2.0 ** -sm.R_ITER_P)))
                    else:
                        ops.bootstrap_real(R)
                Rh = ops.rotate(R, 16)
                mk = ("sm.unpack",) if sc_r == 1.0 else ("sm.unpack", sc_r)

                def unpack_p(zs):
                    return [ops.add(ops.mult(rt.mult_mask(z, mk, lambda: sc_r * sm.LO), R),
                                    ops.mult_i(ops.mult(rt.mult_mask(ops.rotate(z, 16), mk, lambda: sc_r * sm.LO), Rh)))
                            for z in zs]
                P = []
                if not _STREAM:
                    for j in range(p + 1):
                        with rt.step(f"u{r}", repeat=True):
                            P.append(unpack_p(Zr[j]))
                        Zr[j] = None
            else:
                Y = []
                for j in range(p + 1):
                    with rt.step(f"y{r}", repeat=True):
                        Y.append([ops.mult(z, R) for z in Zr[j]])
                    Zr[j] = None

    if layer == _DBG_LAYER and P:
        Pdbg = []
        for pj in P:
            blk = []
            for x in pj:
                v = ops.decrypt_slots_complex(x)
                blk.append(np.where(sc.slot_coords()[2] < 16, np.real(v), np.imag(np.roll(v, 16))))
            Pdbg.append(blk)
        _dbg_softmax(rt, Sdbg, Pdbg, p, al, tq, tk, qv)
    with rt.step("softmax_v"):
        O = None
        for j in range(p + 1):
            with rt.step("pv", repeat=True):
                if _STREAM:
                    P.append(unpack_p(Zr[j]))
                    Zr[j] = None
                pj = sc.dup(rt, P[j]) if merge else P[j]
                P[j] = None
                Oj = eng.apply(pj, store[j][1], b_transposed=False, amax=1.5, bmax=VMAX, shift=SHIFT_PV, site=f"pv{layer}",
                               merge=merge)
                O = _acc(rt, O, Oj)
        out = sc.from_sinc_q_merged(rt, O, cm.w_slots(eng.w)) if merge else sc.from_sinc_q(rt, O)
    return [out]
