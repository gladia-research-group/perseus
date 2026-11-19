"""prefill_softmax's numerics on CCMM scores in the SinC slot layout: complex scores packed real (Im half on batch
l + 16), normalised per (query, head) row with per-slot seeds, unpacked for the P.V CCMM."""
from __future__ import annotations

import os

import numpy as np

from . import sinc as sc
from . import smcheb
from .attention import _recip_rounds, sink_only
from .poly import eval_chebyshev, eval_chebyshev_masked, im_cleanse
from .prefill import _KcExtrapolated, split

LO = (sc.slot_coords()[2] < 16).astype(float)
# the half-head view's unpack mask: the first half's batch slots (kv' with e = 0) alone, P copied onto e = 1 after
LO_HALF = ((sc.slot_coords()[2] < 8) & (((sc.slot_coords()[2] // sc.HALF_STEP) % 2) == 0)).astype(float)
# PF_SM_R_REFRESH=1 (with unpacked): one refresh of the last reciprocal R instead of P's 16 CCMM-entry refreshes
R_REFRESH = os.environ.get("PF_SM_R_REFRESH", "0") not in ("", "0")
R_SCALE = float(os.environ.get("PF_SM_R_SCALE", "1"))   # refresh R / R_SCALE (|R| <= 32.5), R_SCALE rides the Z masks
# PF_SM_R_ITER=1: the R refresh as two passes (the residual refreshed at 2^PF_SM_R_ITER_P, bts2's form from leaf ops)
R_ITER = os.environ.get("PF_SM_R_ITER", "0") not in ("", "0")
# PF_SM_PAIR=1: the later rounds' Z refreshes as one complex bootstrap per score ciphertext pair (y_a + i y_b, split
# after: 2 Re y_a, 2 Re y_b = the cleanses' values) instead of two real-route ones
PAIR = os.environ.get("PF_SM_PAIR", "0") not in ("", "0")
# PF_SM_PAIR_ROUNDS=1: pair only those rounds (the ones the plan refreshes); unset = every round >= 1
PAIR_ROUNDS = {int(v) for v in os.environ.get("PF_SM_PAIR_ROUNDS", "").split(",") if v}
R_ITER_P = int(os.environ.get("PF_SM_R_ITER_P", "8"))
_DBG_X = os.environ.get("PF_SM_DBG_X", "0") not in ("", "0")


def pack(rt, S):
    """complex -> 2 (Re on l < 16, Im on l + 16), real."""
    ops = rt.ops
    out = []
    for s in S:
        c = ops.conjugate(s)
        out.append(ops.add(ops.add(s, c), ops.rotate(ops.mult_i(ops.sub(c, s)), -16)))
    return out


def unpack(rt, Y, site="sm.unpack", lo=None):
    """packed real -> complex (Re / Im = chunk 0 / 1); `lo`: the batch slots kept (default LO)."""
    ops = rt.ops
    lo = LO if lo is None else lo
    key = (site,) if lo is LO else (site, "half")
    out = []
    for y in Y:
        re = rt.mult_mask(y, key, lambda: lo)
        im = rt.mult_mask(ops.rotate(y, 16), key, lambda: lo)
        out.append(ops.add(re, ops.mult_i(im)))
    return out


def compress_half(rt, Spk):
    """The half view's 16 packed real score ciphertexts (zero off sinc's V) -> 4: ciphertext 4 p + r onto p at batch
    offset HALF_DELTAS[r] (rotations only)."""
    ops = rt.ops
    out = []
    for p in range(len(Spk) // 4):
        acc = None
        for r, dl in enumerate(sc.HALF_DELTAS):
            x = Spk[4 * p + r]
            x = x if dl == 0 else ops.rotate(x, -dl)
            acc = x if acc is None else ops.add(acc, x)
        out.append(acc)
    return out


def _half_den(rt, D, site="sm.hden"):
    """The compressed layout's denominator: the four regions' sums (offsets 2, 8) on region 0, selected (a level),
    broadcast back to every region."""
    ops = rt.ops
    D = ops.add(D, ops.rotate(D, 2))
    D = ops.add(D, ops.rotate(D, 8))
    D = rt.mult_mask(D, (site,), lambda: (sc.half_regions()[0] == 0).astype(float))
    D = ops.add(D, ops.rotate(D, -2))
    return ops.add(D, ops.rotate(D, -8))


def recip_slots(rt, D, av, bv, iters, key, cheb=None):
    """prefill.recip_lanes with per-slot seed vectors (av, bv, kappas, mu over all slots)."""
    ops = rt.ops
    hb = rt.mult_mask(D, ("sm.rcp.hb",) + key, lambda: 0.5 * bv)
    G = rt.add_mask(hb, ("sm.rcp.ha",) + key, lambda: -0.5 * av)
    if cheb is not None:
        kap, mu = cheb
        X = rt.add_mask(ops.mult_i(D), ("sm.rcp.mu",) + key, lambda: mu)
        P = ops.negate(ops.mult_i(ops.mult(G, X)))
        for j in range(iters - 1):
            F = rt.add_mask(ops.add(P, ops.conjugate(P)), ("sm.rcp.kap", j) + key, lambda j=j: 2.0 * kap[j])
            P = ops.mult(P, F)
        return ops.mult_i(ops.sub(ops.conjugate(P), P))
    nb = rt.mult_mask(D, ("sm.rcp.nb",) + key, lambda: -0.5 * bv)
    im = rt.add_mask(nb, ("sm.rcp.pa",) + key, lambda: 0.5 * av)
    P = ops.add(ops.mult(D, G), ops.mult_i(im))
    for _ in range(1, iters):
        P = ops.mult(P, ops.add(ops.add(P, ops.conjugate(P)), 2.0))
    return ops.mult_i(ops.sub(ops.conjugate(P), P))


def score_affine(cfg):
    """(alpha, beta) of the exp Chebyshev's domain map y = alpha s - beta."""
    sf = 2.0 ** (-cfg.log2delta1 - cfg.log2delta2)
    a, b = cfg.cheb_a / sf, cfg.cheb_b / sf
    return 2.0 / (b - a), (a + b) / (b - a)


def softmax(rt, Spk, cfg, layer=0, T=127, unpacked=False, prescaled=False, lo=None, half=False):
    """Spk: 16 packed real score ciphertexts in true units, the BOS key on key token sc.BOS_TOK.
    Returns the packed probabilities, or (`unpacked`) unpack's output with its masks on the Z side of the last
    product: (Z LO) R + i (rot(Z, 16) LO) rot(R, 16), one level shallower when R is the deeper factor."""
    sink_only(cfg, "sinc_softmax.softmax")   # a key-0 config: the caller's K rows came off the BOS key (attention_ccmm)
    ops = rt.ops
    if half:                # the half view: the 16 score ciphertexts compressed to 4 (sinc.half_meta)
        if not unpacked:
            raise ValueError("sinc_softmax.softmax: the compressed half view needs unpacked (PF_SM_UNPACK_FOLD=1)")
        Spk = compress_half(rt, Spk)
        tq, tk = sc.half_meta(T)
    else:
        tq, tk = sc.packed_meta(T)
    hk = ("half",) if half else ()
    qv = tq >= 0
    kc = np.where(qv, tq + 2, 1)
    mean = (cfg.clip_hi + cfg.clip_lo) / 2.0
    sf = 2.0 ** (-cfg.log2delta1 - cfg.log2delta2)

    def exp_part(s, valid, site):
        if prescaled:       # Spk = alpha S: the domain map's product rode the caller's masks
            al, be = score_affine(cfg)
            s = rt.add_mask(s, site + ("score", cfg.clip_lo, mean, T, "pre"),
                            lambda: al * np.where(valid, -mean, cfg.clip_lo - mean) - be)
            a, b = -1.0, 1.0
        else:
            s = rt.add_mask(s, site + ("score", cfg.clip_lo, mean, T),
                            lambda: np.where(valid, -mean, cfg.clip_lo - mean))
            a, b = cfg.cheb_a / sf, cfg.cheb_b / sf
        if _DBG_X and hasattr(s, "vec"):     # PF_SM_DBG_X=1 (fake): the series' input on valid / invalid slots
            xv = np.real(s.vec) if prescaled else (2 * np.real(s.vec) - (a + b)) / (b - a)
            print(f"[sm_dbg_x] {site}: valid |x| max {np.abs(xv[valid]).max():.3g}, invalid |x| max "
                  f"{np.abs(xv[~valid]).max() if (~valid).any() else 0:.3g}", flush=True)
        if cfg.exp_mask_fold:
            # the active mask's 2^log2delta1-th root rides the series, as the packed prefill's (prefill._sm_cheb):
            # (p M^(1/2^s))^(2^s) = p^(2^s) M, a level less after the exp
            pw = 2 ** cfg.log2delta1
            act = np.where(valid, 0.5 / kc, 0.0)
            root = lambda sc_: (lambda: sc_ * act ** (1.0 / pw))
            z = eval_chebyshev_masked(ops, s, cfg.cheb_coeffs, a, b, None,
                                      lambda ct, sc_: rt.mult_mask(ct, site + ("active", T, "fold", pw, sc_), root(sc_)),
                                      lambda ct, sc_: rt.add_mask(ct, site + ("active", T, "fold+", pw, sc_), root(sc_)))
            for _ in range(cfg.log2delta1):
                z = ops.square(z)
            return im_cleanse(ops, z)
        z = eval_chebyshev(ops, s, cfg.cheb_coeffs, a, b)
        for _ in range(cfg.log2delta1):
            z = ops.square(z)
        z = im_cleanse(ops, z)
        return rt.mult_mask(z, site + ("active", T), lambda: np.where(valid, 0.5 / kc, 0.0))
    Z = [exp_part(s, qv & ((tk[o] == sc.BOS_TOK) | (tk[o] <= tq)), ("sm", o) + hk) for o, s in enumerate(Spk)]
    cheb = getattr(ops, "sm_cheb_gs", False)
    ks = sorted(set(kc.tolist()))
    rnd = {k: (smcheb.rounds(cfg, k) if cheb else _recip_rounds(_KcExtrapolated(cfg), k, ops.sm_gs_first)) for k in ks}
    n_rounds = len(rnd[ks[0]])
    Y = Z
    for r in range(n_rounds):
        if r and PAIR and len(Y) % 2 == 0 and (not PAIR_ROUNDS or r in PAIR_ROUNDS):
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
        if half:
            D = _half_den(rt, D)
        per = [rnd[int(k)][r] for k in kc]
        av = np.array([p[0] * p[3] for p in per])
        bv = np.array([p[1] * p[3] * p[3] for p in per])
        iters = per[0][2]
        cv = None
        if cheb:
            cv = ([np.array([p[4][0][j] for p in per]) for j in range(iters - 1)], np.array([p[4][1] for p in per]))
        R = recip_slots(rt, D, av, bv, iters, ("sm", layer, r, T), cv)
        if unpacked and r == n_rounds - 1:
            sc_r = R_SCALE if R_REFRESH else 1.0
            if R_REFRESH:
                if sc_r != 1.0:
                    R = ops.mult(R, 1.0 / sc_r)
                if R_ITER:
                    y = ops.copy(R)
                    ops.bootstrap_real(y)
                    # + 1/4: the residual is 0 on an exact backend, its refresh needs a captured magnitude
                    e = ops.add(ops.mult(ops.sub(R, y), float(2 ** R_ITER_P)), 0.25)
                    ops.bootstrap_real(e)
                    R = ops.add(y, ops.mult(ops.add(e, -0.25), float(2.0 ** -R_ITER_P)))
                else:
                    ops.bootstrap_real(R)
            lm = LO if lo is None else lo
            mk = (("sm.unpack",) if sc_r == 1.0 else ("sm.unpack", sc_r)) + (() if lo is None else ("half",))
            if half:        # back to 16: region r of ciphertext p onto V (the unpack masks select it, no level)
                Rs = [R if dl == 0 else ops.rotate(R, dl) for dl in sc.HALF_DELTAS]
                Rhs = [ops.rotate(R, dl + 16) for dl in sc.HALF_DELTAS]
                out = []
                for z in Z:
                    for r, dl in enumerate(sc.HALF_DELTAS):
                        zr = z if dl == 0 else ops.rotate(z, dl)
                        out.append(ops.add(ops.mult(rt.mult_mask(zr, mk, lambda: sc_r * lm), Rs[r]),
                                           ops.mult_i(ops.mult(rt.mult_mask(ops.rotate(zr, 16), mk,
                                                                            lambda: sc_r * lm), Rhs[r]))))
                return out
            Rh = ops.rotate(R, 16)
            return [ops.add(ops.mult(rt.mult_mask(z, mk, lambda: sc_r * lm), R),
                            ops.mult_i(ops.mult(rt.mult_mask(ops.rotate(z, 16), mk, lambda: sc_r * lm), Rh)))
                    for z in Z]
        Y = [ops.mult(z, R) for z in Z]
    return Y
