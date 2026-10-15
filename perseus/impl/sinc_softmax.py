"""prefill_softmax's numerics on CCMM scores in the SinC slot layout: complex scores packed real (Im half on batch
l + 16), normalised per (query, head) row with per-slot seeds, unpacked for the P.V CCMM."""
from __future__ import annotations

import os

import numpy as np

from . import sinc as sc
from . import smcheb
from .attention import _recip_rounds
from .poly import eval_chebyshev, im_cleanse
from .prefill import _KcExtrapolated

LO = (sc.slot_coords()[2] < 16).astype(float)
# PF_SM_R_REFRESH=1 (with unpacked): one refresh of the last reciprocal R instead of P's 16 CCMM-entry refreshes
R_REFRESH = os.environ.get("PF_SM_R_REFRESH", "0") not in ("", "0")


def pack(rt, S):
    """complex -> 2 (Re on l < 16, Im on l + 16), real."""
    ops = rt.ops
    out = []
    for s in S:
        c = ops.conjugate(s)
        out.append(ops.add(ops.add(s, c), ops.rotate(ops.mult_i(ops.sub(c, s)), -16)))
    return out


def unpack(rt, Y, site="sm.unpack"):
    """packed real -> complex (Re / Im = chunk 0 / 1)."""
    ops = rt.ops
    out = []
    for y in Y:
        re = rt.mult_mask(y, (site,), lambda: LO)
        im = rt.mult_mask(ops.rotate(y, 16), (site,), lambda: LO)
        out.append(ops.add(re, ops.mult_i(im)))
    return out


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


def softmax(rt, Spk, cfg, layer=0, T=127, unpacked=False):
    """Spk: 16 packed real score ciphertexts in true units, the BOS key on key token sc.BOS_TOK.
    Returns the packed probabilities, or (`unpacked`) unpack's output with its masks on the Z side of the last
    product: (Z LO) R + i (rot(Z, 16) LO) rot(R, 16), one level shallower when R is the deeper factor."""
    ops = rt.ops
    tq, tk = sc.packed_meta(T)
    qv = tq >= 0
    kc = np.where(qv, tq + 2, 1)
    mean = (cfg.clip_hi + cfg.clip_lo) / 2.0
    sf = 2.0 ** (-cfg.log2delta1 - cfg.log2delta2)

    def exp_part(s, valid, site):
        s = rt.add_mask(s, site + ("score", cfg.clip_lo, mean, T),
                        lambda: np.where(valid, -mean, cfg.clip_lo - mean))
        z = eval_chebyshev(ops, s, cfg.cheb_coeffs, cfg.cheb_a / sf, cfg.cheb_b / sf)
        for _ in range(cfg.log2delta1):
            z = ops.square(z)
        z = im_cleanse(ops, z)
        return rt.mult_mask(z, site + ("active", T), lambda: np.where(valid, 0.5 / kc, 0.0))
    Z = [exp_part(s, qv & ((tk[o] == sc.BOS_TOK) | (tk[o] <= tq)), ("sm", o)) for o, s in enumerate(Spk)]
    cheb = getattr(ops, "sm_cheb_gs", False)
    ks = sorted(set(kc.tolist()))
    rnd = {k: (smcheb.rounds(cfg, k) if cheb else _recip_rounds(_KcExtrapolated(cfg), k, ops.sm_gs_first)) for k in ks}
    n_rounds = len(rnd[ks[0]])
    Y = Z
    for r in range(n_rounds):
        if r:
            Z = [ops.square(im_cleanse(ops, y)) for y in Y]
        D = Z[0]
        for z in Z[1:]:
            D = ops.add(D, z)
        D = ops.rotate_and_sum(D, 1 << 12, sc.SLOTS)
        per = [rnd[int(k)][r] for k in kc]
        av = np.array([p[0] * p[3] for p in per])
        bv = np.array([p[1] * p[3] * p[3] for p in per])
        iters = per[0][2]
        cv = None
        if cheb:
            cv = ([np.array([p[4][0][j] for p in per]) for j in range(iters - 1)], np.array([p[4][1] for p in per]))
        R = recip_slots(rt, D, av, bv, iters, ("sm", layer, r, T), cv)
        if unpacked and r == n_rounds - 1:
            if R_REFRESH:
                ops.bootstrap_real(R)
            Rh = ops.rotate(R, 16)
            return [ops.add(ops.mult(rt.mult_mask(z, ("sm.unpack",), lambda: LO), R),
                            ops.mult_i(ops.mult(rt.mult_mask(ops.rotate(z, 16), ("sm.unpack",), lambda: LO), Rh)))
                    for z in Z]
        Y = [ops.mult(z, R) for z in Z]
    return Y
