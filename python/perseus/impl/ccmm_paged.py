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
from .attention import _recip_rounds, sink_only
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
    if _META_CACHE:
        m = _meta_cache.get(p)
        if m is None:
            m = _meta_cache[p] = tuple(_frozen(a) for a in _page_meta(p))
        return m
    return _page_meta(p)


def _page_meta(p: int):
    tq, tk = sc.packed_meta(page_tokens(p))
    qv = tq >= 0
    kc = np.where(qv, PAGE * p + tq + (2 if p == 0 else 1), 1)
    return tq, tk, qv, kc


# PF_SM_META_CACHE=1 (default 0): page_meta and the reciprocal rounds' per-slot seed vectors are built once per page
# (and softmax config) and reused: pure functions of their key, ~12 ms of host Python each that the GPU waits on. The
# warm-up pass builds them; the cached arrays are read-only.
_META_CACHE = __import__("os").environ.get("PF_SM_META_CACHE", "0") not in ("", "0")
_meta_cache = {}
_seed_cache = {}


def _frozen(a):
    a.setflags(write=False)
    return a


def _round_seeds(cfg, p: int, kc, cheb: bool, gs_first):
    """Per reciprocal round: (av, bv, iters, cv) over the page's slots (the values the round loop builds inline)."""
    key = (p, cheb, gs_first, repr(cfg))
    out = _seed_cache.get(key)
    if out is not None:
        return out
    ks = sorted(set(kc.tolist()))
    rnd = {k: (smcheb.rounds(cfg, k) if cheb else _recip_rounds(_KcExtrapolated(cfg), k, gs_first)) for k in ks}
    out = []
    for r in range(len(rnd[ks[0]])):
        per = [rnd[int(k)][r] for k in kc]
        av = _frozen(np.array([q[0] * q[3] for q in per]))
        bv = _frozen(np.array([q[1] * q[3] * q[3] for q in per]))
        cv = None
        if cheb:
            cv = ([_frozen(np.array([q[4][0][i] for q in per])) for i in range(per[0][2] - 1)],
                  _frozen(np.array([q[4][1] for q in per])))
        out.append((av, bv, per[0][2], cv))
    _seed_cache[key] = out
    return out


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
# PF_SM_EXP_EPS=2e-6: the softmax exp series' coefficient trim (relative to the largest; default poly.CHEB_COEFF_EPS
# 1e-6). Layers whose degree-7 coefficient sits just above 1e-6 (14, 29 at d1 = 8) evaluate one level deeper and
# refresh every score ciphertext before the first normalisation; 2e-6 cuts them to degree 6 like their neighbours
_EXP_EPS = float(__import__("os").environ.get("PF_SM_EXP_EPS", "0") or 0) or None
# PF_SM_R_RAISE=d: the last round's deliberate R refreshes (PF_SM_R_REFRESH / PF_SM_R_ITER) raise d composite levels
# below the top (the route's raise variant; FIDESLIB_BTS_RAISE_DROPS must list d): their outputs have 3-4 levels of
# slack before the unpack products. The capture lands them d levels lower too (CapFhe.raise_drop)
_R_RAISE = int(__import__("os").environ.get("PF_SM_R_RAISE", "0") or 0)


def _r_refresh(ops, x):
    if not _R_RAISE:
        return ops.bootstrap_real(x)
    bts, fhe = getattr(ops.core, "bts", None), ops.fhe
    if bts is not None:
        old = bts.raise_scope(fhe, _R_RAISE)
        try:
            return ops.bootstrap_real(x)
        finally:
            bts.raise_scope(fhe, old)
    old = getattr(fhe, "raise_drop", 0)
    fhe.raise_drop = _R_RAISE
    try:
        return ops.bootstrap_real(x)
    finally:
        fhe.raise_drop = old


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
    sink_only(cfg, "attention_paged")
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
        # the BOS key / value only on page 0; later pages add zeros (the same op stream). A softmax centred on key 0:
        # every page's token rows minus the (public) BOS key instead, page 0's BOS row left 0 (q.K^T = s - s_BOS)
        if cfg.ref_keys:
            kpat = sc.kv_rows_pattern(-kb, range(page_tokens(p)))
            Ks = [rt.add_mask(x, ("ccmm.kref", layer, o, p == 0), lambda o=o: kpat[o]) for o, x in enumerate(Ks)]
        else:
            kpat = sc.kv_row_pattern(kb) if p == 0 else np.zeros((sc.NCT, sc.SLOTS))
            Ks = [rt.add_mask(x, ("ccmm.kbos", layer, o, p == 0), lambda o=o: kpat[o]) for o, x in enumerate(Ks)]
        vpat = sc.kv_row_pattern(vb) if p == 0 else np.zeros((sc.NCT, sc.SLOTS))
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
            z = eval_chebyshev(ops, s, cfg.cheb_coeffs, -1.0, 1.0, _EXP_EPS)
        else:
            s = rt.add_mask(s, site + ("score", cfg.clip_lo, mean),
                            lambda: np.where(valid, -mean, cfg.clip_lo - mean))
            z = eval_chebyshev(ops, s, cfg.cheb_coeffs, cfg.cheb_a / sf, cfg.cheb_b / sf, _EXP_EPS)
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
        seeds = _round_seeds(cfg, p, kc, cheb, ops.sm_gs_first) if _META_CACHE else None
        if seeds is None:
            ks = sorted(set(kc.tolist()))
            rnd = {k: (smcheb.rounds(cfg, k) if cheb else _recip_rounds(_KcExtrapolated(cfg), k, ops.sm_gs_first))
                   for k in ks}
        n_rounds = len(seeds) if seeds is not None else len(rnd[ks[0]])
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
            if seeds is not None:
                av, bv, iters, cv = seeds[r]
            else:
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
                        _r_refresh(ops, y)
                        e = ops.add(ops.mult(ops.sub(R, y), float(2 ** sm.R_ITER_P)), 0.25)
                        _r_refresh(ops, e)
                        R = ops.add(y, ops.mult(ops.add(e, -0.25), float(2.0 ** -sm.R_ITER_P)))
                    else:
                        _r_refresh(ops, R)
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
