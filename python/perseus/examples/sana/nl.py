"""The SANA port's polynomial maps on ops, read from a configs.json (fit.py): PolyNL's hooks the kernels call
(kernels.py, model.py, gemma_fhe.py), and Room / series_units, which size the refresh sites the plan may take.

  relu_u / relu_u2   one-sided composite sign on the unit input (a polished last stage optional): ReLU(x) ~ x s(x)
                     with s <= 0 on x <= 0, so f(q).z >= 0 and the attention's division stays in its band
  silu_u2            the single odd series (or THOR's composite) on the unit input, both real halves, refreshes joint
  inv_sqrt           Chebyshev seed + Newton on the pre-scaled statistic c^2 z
  recip              Goldschmidt from a Chebyshev-optimal seed (or a Chebyshev seed + Newton)
  exp / exp_need     one Chebyshev series of exp((x / 2 - c) 2^-r) (x = 2 s, the split's)
  ln                 the LayerNorm's NormCfg for blocked.norm_blocked
"""
from __future__ import annotations


import os

import numpy as np

from perseus.impl.config import NormCfg, parse_norm
from perseus.impl.poly import _cheb_trim, eval_chebyshev, inv_sqrt_newton_d2

from .kernels import GUARD


def joint_force():
    """PF_JOINT_FORCE=1 (Llama prefill's): the joint (complex-packed) refreshes a deliberate bootstrap, not a hint."""
    import os
    return os.environ.get("PF_JOINT_FORCE", "0") not in ("", "0")

# Llama prefill's PF_SILU_SINGLE on the unit forms: a site's SiLU as x (s(t) + 1/2), s the odd series of
# sigma(xmax t) - 1/2 on t = u, of the smallest degree in SILU_SINGLE_DEGS (series_units breakpoints: 8 / 10 / 12 / 14
# levels; 123 only where 119 misses the tolerance) whose error max |x| |s - f| stays within SILU_SINGLE_TOL x the THOR
# composite's (Llama's rule, perseus.impl.prefill.silu_single_coeffs); one series, no refresh between the composite's
# two. A site no degree fits keeps the composite.
SILU_SINGLE_DEGS = (27, 59, 119, 123)
SILU_SINGLE_TOL, SILU_SINGLE_EPS = 2.0, 1e-8
# SANA_GLU_LATE=1 (with PF_PCMM_RAISE's ff_in at 0): the GLU's value half h multiplies the gate's SiLU last,
# h (u s(u)), not first, (h u) s(u). Taking h u before the series (Llama's PF_GLU_XU) saves a product when h is
# shallow; here h comes out of act1 and the conv at the series' own depth, so the early product only forced a refresh
# of h (30 per block). Taken last, and with ff_in landing at 26, h needs none where act1's series costs 5 levels
# (degree <= 28): 36 FFN refreshes in those blocks instead of 60. Without ff_in:0 it costs refreshes: set both
GLU_LATE = os.environ.get("SANA_GLU_LATE", "0") not in ("", "0")
# SANA_SILU_XFOLD=1: the single series' u s(u) as ONE Chebyshev series of degree n + 1 (x T_k = (T_{k+1} + T_{k-1}) / 2,
# numpy chebmulx): the same polynomial a level shallower than s(u) then times u. The degree lists sit at the depth
# classes' tops minus one (27 / 59 / 119 / 123; 15 / 23 / 31 / 63 / 127 under PF_SILU_SINGLE_ERR), so n + 1 keeps
# the series' depth. With GLU_LATE, h (u s(u)) lands a level shallower: ff_out's entry hints stop firing where
# act1's and the gate's products were the one level over the StC-first cap.
SILU_XFOLD = os.environ.get("SANA_SILU_XFOLD", "0") not in ("", "0")


def _gs(ops_or_np, D, a, b, iters, rt=None):
    """Goldschmidt 1 / D from the seed a - b D (numpy arrays or an ops object); with rt, a refresh site on the two
    running ciphertexts every 3 steps."""
    if ops_or_np is np:
        R = a - b * D
        E = D * R
        for _ in range(iters):
            F = 2.0 - E
            R, E = R * F, E * F
        return R
    ops = ops_or_np
    R = ops.add(ops.mult(D, -b), a)
    E = ops.mult(D, R)
    for i in range(iters):
        if rt is not None and i and i % 3 == 0:
            R, E = hint(rt, R, 4), hint(rt, E, 4)
        F = ops.add(ops.negate(E), 2.0)
        R = ops.mult(R, F)
        if i + 1 < iters:
            E = ops.mult(E, F)
    return R


# ── the maps on ops ──

def hint(rt, x, k):
    """A refresh site the plan may take: x with room for k products (blocked kernels' bootstrap_hint)."""
    ops = rt.ops
    return ops.bootstrap_hint(x, ops.headroom(k), True)


def series_units(deg):
    """Levels eval_chebyshev (CHEB_BSGS) spends on a degree-deg series from a deg-1 input, measured on the twin:
    2 per step of the baby-step giant-step ladder (degrees 121..124 take 14, not 12)."""
    for top, units in ((2, 2), (6, 4), (14, 6), (28, 8), (60, 10), (120, 12), (248, 14)):
        if deg <= top:
            return units
    raise ValueError(f"degree {deg}")


class Room:
    """Refresh sites only where a series does not fit: `need(x, coef, tail)` hints x when its level, plus the series'
    levels (+2 for a pending rescale on x, + `tail` for products after the series) would pass GUARD. On the twin the
    levels are the ciphertexts'; on the mirror (level 0) nothing fires."""

    def __init__(self, rt):
        self.rt = rt

    def need(self, x, coef, tail=0):
        ops = self.rt.ops
        cost = series_units(len(coef) - 1) + tail + (2 if ops.noise_deg(x) > 1 else 0)
        if ops.level(x) + cost > GUARD:
            x = ops.bootstrap_hint(x, GUARD - cost, True)
        return x


class PolyNL:
    """kernels.py's nonlinear hooks from a configs dict (calibrate's). Every series starts at a hint sized to its depth:
    the plan's cuts land on the one ciphertext entering it, not on the power ladder's branches."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._ln = {k: parse_norm(v) for k, v in cfg["ln"].items()}
        # SANA_LA_FLOOR=rho (default off): attn1's den floor delta at rho times the block's largest den (delta / sigma
        # = 4 rho), so 1 / (den + delta) <= 1 / delta: the rows whose den the refresh noise (relative to den's band
        # top) swamps come out ~0 instead of noise / den. The reciprocals keep their fits (their range [lo, top] holds
        # [2 delta / sigma, top]: the op stream, hence a plan, unchanged); SANA_LA_FLOOR_REFIT=1 refits each la.den
        # reciprocal on [2 delta / sigma, its top] (fewer iterations: a new capture and plan).
        import os
        self.la_floor = float(os.environ.get("SANA_LA_FLOOR") or 0.0)
        if self.la_floor and os.environ.get("SANA_LA_FLOOR_REFIT") == "1":
            from .fit import fit_recip_cgs
            # SANA_LA_RECIP_TGT (default 1e-6, fit_recip_cgs's): the refit's relative error; 1e-5 takes one Goldschmidt
            # iteration (a level) off 26 of the 28 blocks at rho 1e-4, far under a refresh's own error
            tgt = float(os.environ.get("SANA_LA_RECIP_TGT") or 1e-6)
            cfg = self.cfg = dict(cfg, recip=dict(cfg["recip"]))
            for k, p in cfg["recip"].items():
                if k.endswith(".la.den") and (2 * self._la_delta(p) > p["lo"] or tgt != 1e-6):
                    cfg["recip"][k] = dict(fit_recip_cgs(max(2 * self._la_delta(p), p["lo"]), p["hi"], tgt),
                                           sigma=p["sigma"])

    @property
    def resid_scale(self):
        return np.asarray(self.cfg.get("resid_scale", np.ones(1152)))

    def _la_delta(self, p):
        return max(self.cfg["la_delta"] / p["sigma"], 4.0 * self.la_floor)

    def la_delta(self, site):
        return self._la_delta(self.cfg["recip"][site + ".la.den"])

    def la_sigma(self, site):
        return self.cfg["recip"][site + ".den"]["sigma"]

    def sm_rounds(self, site):
        return self.cfg.get("sm_rounds_site", {}).get(site, self.cfg["sm_rounds"])

    def ln(self, site) -> NormCfg:
        return self._ln[site]

    def stat_scale(self, site):
        return self.cfg["inv_sqrt"][site]["c"]

    # ── unit-input forms: the producer folds the input map, every stage reads [-1, 1] (no level for a map) ──

    def relu_unit_scale(self, site):
        """What the producer multiplies x by: relu_u reads u = x / B."""
        return 1.0 / self.cfg["relu"]["sites"][site]["B"]

    def _relu_unit_stages(self, site):
        """The unit forms' series [(coef on [-1, 1], tail)]: each stage's coefficients carrying the next stage's 1 / h,
        the last one B (s - ripple); with a polish, the composite's last one 2 s - 1 and the polish B S (fit_relu)."""
        B = self.cfg["relu"]["sites"][site]["B"]
        r = self.cfg["relu_k"] if site.endswith(".k") and "relu_k" in self.cfg else self.cfg["relu"]
        st, pol = r["stages"], r.get("polish_cheb")
        out = []
        for i, (coef, _h) in enumerate(st):
            nxt = st[i + 1][1] if i + 1 < len(st) else None
            coef = list(coef)
            if nxt is not None:
                coef = [v / nxt for v in coef]
            elif pol is not None:
                coef = [2.0 * v for v in coef]
                coef[0] -= 1.0
            else:
                coef[0] -= r["ripple"]
                coef = [B * v for v in coef]
            out.append((coef, 2 if nxt is None and pol is None else 0))
        if pol is not None:
            out.append(([B * v for v in pol], 2))
        return out, r["eps"]

    def relu_u(self, rt, u, site, room=None):
        """ReLU(B u) of u = x / B: stage 1 on u - eps (a free shift), then _relu_unit_stages."""
        ops = rt.ops
        stages, eps = self._relu_unit_stages(site)
        rm = Room(rt)
        s = None
        for i, (coef, tail) in enumerate(stages):
            if i == 0:
                u = rm.need(u, coef)
                s = eval_chebyshev(ops, ops.add(u, -eps), coef, -1.0, 1.0)
            else:
                s = eval_chebyshev(ops, rm.need(s, coef, tail), coef, -1.0, 1.0)
        return ops.mult(u, s)

    # ── paired forms (Llama prefill's PF_GELU_JOINT / PF_SM_PAIR): the two real halves of one complex ciphertext run
    # the same chain; where a stage boundary needs a refresh, the halves (the stage's coefficients halved) join, take
    # ONE complex refresh, and split back (2 Re, 2 Im: the halving undone, no level) ──

    def _pair_chain(self, rt, ua, ub, stages):
        """stages: [(coef, tail)], each series on [-1, 1] of the previous one's output (the first on ua / ub); `tail`:
        the products after that series that its entry refresh must leave room for. Returns the last outputs."""
        ops = rt.ops
        cur = [ua, ub]
        for i, (coef, _tail) in enumerate(stages):
            coef = list(coef)
            joint, cost = False, 0
            if i + 1 < len(stages):
                nxt, ntail = stages[i + 1]
                # the series' output level (exact on the twin: series_units, +2 for a pending rescale on its input)
                lvl = ops.level(cur[0]) + series_units(len(coef) - 1) + (2 if ops.noise_deg(cur[0]) > 1 else 0)
                cost = series_units(len(nxt) - 1) + ntail + 2           # the output carries a pending rescale
                joint = lvl + cost > GUARD
                if joint:
                    coef = [0.5 * v for v in coef]
            cur = [eval_chebyshev(ops, x, coef, -1.0, 1.0) for x in cur]
            if joint:
                from perseus.impl.prefill import join, split
                # PF_JOINT_FORCE=1 (Llama prefill's): the joint refresh deliberate (a fixed point of the plan), not a hint
                X = (ops.bootstrap(join(ops, cur[0], cur[1])) if joint_force()
                     else ops.bootstrap_hint(join(ops, cur[0], cur[1]), GUARD - cost, True))
                cur = list(split(ops, X))
        return cur

    def relu_u2(self, rt, ua, ub, site, room=None, mul=None, tail=0):
        """relu_u on both real halves, the stage boundaries' refreshes joint. `mul` (ma, mb): (ma relu(a), mb relu(b))
        with ma u taken before the series ends (silu_u2's); `tail`: products the outputs take after (the last joint
        boundary sized for them)."""
        ops = rt.ops
        stages, eps = self._relu_unit_stages(site)
        if tail:
            stages[-1] = (stages[-1][0], stages[-1][1] + 2 * tail)
        rm = Room(rt)
        ua, ub = rm.need(ua, stages[0][0]), rm.need(ub, stages[0][0])
        sa, sb = self._pair_chain(rt, ops.add(ua, -eps), ops.add(ub, -eps), stages)
        if mul is not None:
            ua, ub = ops.mult(mul[0], ua), ops.mult(mul[1], ub)
        return ops.mult(ua, sa), ops.mult(ub, sb)

    def silu_single(self, site):
        """The site's single-series coefficients (x (xmax s(u) + xmax / 2) / xmax folded: the unit form's), or None."""
        cache = self.__dict__.setdefault("_silu_single", {})
        if site not in cache:
            cache[site] = None
            g = self.cfg["silu"][site]
            if abs(g["thor_p1_a"] + 1.0) < 1e-12 and abs(g["thor_p1_b"] - 1.0) < 1e-12:
                from numpy.polynomial import chebyshev as C
                xm = float(g["xmax"])
                t = np.linspace(-1.0, 1.0, 40001)
                f = 1.0 / (1.0 + np.exp(-xm * t)) - 0.5
                a2, b2 = g["thor_p2_a"], g["thor_p2_b"]
                y1 = C.chebval(t, _cheb_trim(g["thor_p1_cheb"]))
                comp = C.chebval((2 * y1 - (a2 + b2)) / (b2 - a2), _cheb_trim(g["thor_p2_cheb"]))
                tol = max(SILU_SINGLE_TOL * float((np.abs(comp - f) * np.abs(t) * xm).max()), 1e-7)
                degs = SILU_SINGLE_DEGS
                # PF_SILU_SINGLE_ERR=<abs> (Llama prefill's, shipped 2e-2): instead the smallest degree of
                # PF_SILU_SINGLE_DEGS (Llama's 15,23,31,63,127) whose max |x| |s - f| stays within <abs>
                import os
                if float(os.environ.get("PF_SILU_SINGLE_ERR", "0") or 0):
                    tol = float(os.environ["PF_SILU_SINGLE_ERR"])
                    # SANA_SILU_ERR_SITES=<site>:<abs>,...: those sites' own tolerance (a site the tighter budget
                    # moved a degree class up without a gain on held-out prompts goes back)
                    for ent in filter(None, os.environ.get("SANA_SILU_ERR_SITES", "").split(",")):
                        if ent.rsplit(":", 1)[0] == site:
                            tol = float(ent.rsplit(":", 1)[1])
                    degs = tuple(int(v) for v in os.environ.get("PF_SILU_SINGLE_DEGS", "15,23,31,63,127").split(","))
                for n in degs:
                    c = C.chebinterpolate(lambda t: 1.0 / (1.0 + np.exp(-xm * t)) - 0.5, n)
                    c[0::2] = 0.0
                    c = np.array(_cheb_trim(c, SILU_SINGLE_EPS))
                    if float((np.abs(C.chebval(t, c) - f) * np.abs(t) * xm).max()) <= tol:
                        c = [xm * float(v) for v in c]
                        c[0] += 0.5 * xm
                        cache[site] = c
                        break
        return cache[site]

    def silu_u2(self, rt, x, site, room=None, mul=None, out_scale=1.0):
        """SiLU of xmax m1 u on the real halves (2 Re x, 2 Im x) of x (the unit form: the producer folded
        1 / (xmax m1)): the site's single series (silu_single), its entry refresh on x itself and none inside; else
        THOR's p1 then p2, the refresh between them joint. Returns the two halves' outputs; with `mul` (ha, hb),
        (ha silu(a), hb silu(b)) with ha u taken before the series ends (Llama's PF_GLU_XU: the GLU output one product
        shallower)."""
        from perseus.impl.prefill import split
        ops = rt.ops
        cs = self.silu_single(site)
        if cs is not None and out_scale != 1.0:  # the outputs times out_scale: on the coefficients (free)
            cs = [float(out_scale) * v for v in cs]
        if cs is not None and SILU_XFOLD and (mul is None or GLU_LATE):     # u s(u) as one series
            from numpy.polynomial import chebyshev as C
            qs = [float(v) for v in C.chebmulx(np.asarray(cs, dtype=np.float64))]
            ua, ub = split(ops, Room(rt).need(x, qs, 2 if mul is not None else 0))
            qa, qb = (eval_chebyshev(ops, u, qs, -1.0, 1.0, SILU_SINGLE_EPS) for u in (ua, ub))
            if mul is not None:
                return ops.mult(mul[0], qa), ops.mult(mul[1], qb)
            return qa, qb
        if cs is not None:                      # one series: its entry refresh on x, none inside
            ua, ub = split(ops, Room(rt).need(x, cs, 4 if mul is not None else 2))
            pa, pb = (eval_chebyshev(ops, u, cs, -1.0, 1.0, SILU_SINGLE_EPS) for u in (ua, ub))
            if mul is not None and GLU_LATE:    # h (u s(u)): h may then be as deep as the series' output
                return ops.mult(mul[0], ops.mult(ua, pa)), ops.mult(mul[1], ops.mult(ub, pb))
            if mul is not None:
                ua, ub = ops.mult(mul[0], ua), ops.mult(mul[1], ub)
            return ops.mult(ua, pa), ops.mult(ub, pb)
        g = self.cfg["silu"][site]
        if abs(g["thor_p1_a"] + g["thor_p1_b"]) > 1e-12:
            raise ValueError(f"{site}: p1's domain is not symmetric")
        a2, b2 = g["thor_p2_a"], g["thor_p2_b"]
        c1 = [2.0 * v / (b2 - a2) for v in g["thor_p1_cheb"]]
        c1[0] -= (a2 + b2) / (b2 - a2)
        k = g["xmax"] * g["thor_p1_b"]
        c2 = [k * v for v in g["thor_p2_cheb"]]
        c2[0] += 0.5 * k
        c2 = [float(out_scale) * v for v in c2]
        ua, ub = split(ops, Room(rt).need(x, c1))
        pa, pb = self._pair_chain(rt, ua, ub, [(c1, 0), (c2, 2)])
        if mul is not None:
            ua, ub = ops.mult(mul[0], ua), ops.mult(mul[1], ub)
        return ops.mult(ua, pa), ops.mult(ub, pb)

    def silu_unit_scale(self, site):
        """What the producer multiplies x by: silu_u2 reads u = x / (xmax m1), p1's domain [-1, 1]."""
        g = self.cfg["silu"][site]
        return 1.0 / (g["xmax"] * g["thor_p1_b"])

    def inv_sqrt(self, rt, z, site):
        """1 / sqrt(z / c^2) of the pre-scaled statistic z = c^2 (mean + eps)."""
        p = self.cfg["inv_sqrt"][site]
        ops = rt.ops
        z = Room(rt).need(z, p["cheb"], 2)
        y = eval_chebyshev(ops, z, p["cheb"], p["lo"], p["hi"])
        y = ops.bootstrap_hint(y, GUARD - 4 * p["nr"] - 2, True) if p["nr"] else y
        return inv_sqrt_newton_d2(ops, z, y, p["nr"], 1.0 / (p["c"] * p["c"]))

    def recip(self, rt, d, site, dscale=1.0):
        """1 / d, d read as dscale times the calibrated statistic (the seed's domain and coefficients scaled: free)."""
        p = self.cfg["recip"][site]
        ops = rt.ops
        s = float(dscale)
        if "cgs" in p:                          # the Chebyshev-scaled Goldschmidt recursion (k + 1 levels)
            from perseus.impl.poly import goldschmidt_recip
            g = p["cgs"]
            d = Room(rt).need(d, [0.0, 0.0], 2 * g["k"])
            return goldschmidt_recip(ops, d, g["alpha"], g["beta"], g["k"], 1.0 / s, cheb=(tuple(g["kap"]), g["mu"]))
        if "cheb" in p:                         # Chebyshev seed + Newton (a narrow band)
            d = Room(rt).need(d, p["cheb"], 2 + 4 * p["nr"])
            coef = p["cheb"] if s == 1.0 else [v / s for v in p["cheb"]]
            y = eval_chebyshev(ops, d, coef, p["lo"] * s, p["hi"] * s)
            for _ in range(p["nr"]):
                y = ops.mult(y, ops.add(ops.negate(ops.mult(d, y)), 2.0))
            return y
        return _gs(ops, hint(rt, d, 2), p["alpha"] / s, p["beta"] / (s * s), p["iters"], rt=rt)

    def recip_cost(self, site):
        """The levels recip's entry needs (its Room.need: the seed series plus the Newton steps)."""
        p = self.cfg["recip"][site]
        return (p["cheb"], 2 + 4 * p["nr"]) if "cheb" in p else None

    def exp(self, rt, x, site, folded=False, prescaled=False):
        """exp's series; `folded`: x is already its [-1, 1] variable (the producer folded the domain map); `prescaled`:
        x carries the map's slope (the producer's mask), the shift is added here (free)."""
        p = self.cfg["exp"][site]
        a, b = (-1.0, 1.0) if (folded or prescaled) else (p["a"], p["b"])
        if prescaled:
            sh = (p["a"] + p["b"]) / (p["b"] - p["a"])
            x = rt.ops.add(x, -sh) if sh != 0.0 else x
        return eval_chebyshev(rt.ops, Room(rt).need(x, p["cheb"], 2), p["cheb"], a, b)

    def exp_need(self, rt, x, site):
        """exp's entry refresh on x (a complex join of two halves: one refresh for both)."""
        return Room(rt).need(x, self.cfg["exp"][site]["cheb"], 2)

    def softcap(self, rt, x, site, folded=False, exp_site=None):
        """The softcap series; `folded`: x already carries 1 / its half-width (the series on [-1, 1], no map);
        `exp_site`: the exp's domain map folded into the coefficients (the output is the exp's [-1, 1] variable)."""
        p = self.cfg["softcap"][site]
        coef, a, b = list(p["cheb"]), p["a"], p["b"]
        if folded:
            assert abs(a + b) < 1e-12 * abs(b), f"{site}: the fold needs a symmetric domain"
            a, b = -1.0, 1.0
        if exp_site is not None:
            e = self.cfg["exp"][exp_site]
            al, be = 2.0 / (e["b"] - e["a"]), (e["a"] + e["b"]) / (e["b"] - e["a"])
            coef = [al * c for c in coef]
            coef[0] -= be
        return eval_chebyshev(rt.ops, Room(rt).need(x, coef, 2), coef, a, b)

    def gelu_tanh(self, rt, x, site, scale=1.0):
        """The tanh GELU of scale x: one Chebyshev series (a narrow range), else clipped to x Phi(clip(x)) (gelu_clip:
        two sign composites)."""
        if site in self.cfg.get("gelu_clip", {}):
            return self._gelu_clip(rt, x, self.cfg["gelu_clip"][site], scale)
        p = self.cfg["gelu"][site]
        return eval_chebyshev(rt.ops, Room(rt).need(x, p["cheb"], 2), p["cheb"], p["a"] / scale, p["b"] / scale)

    def _sgn(self, ops, x, st, lo, hi):
        """sgn of the affine image of x mapping [lo, hi] onto the composite's [-1, 1]."""
        s = None
        for i, (coef, h) in enumerate(st):
            coef = list(coef)
            if i == len(st) - 1:
                coef = [2.0 * v for v in coef]
                coef[0] -= 1.0
            s = eval_chebyshev(ops, x if i == 0 else s, coef, lo if i == 0 else -h, hi if i == 0 else h)
        return s

    def _gelu_clip(self, rt, x, p, scale):
        ops = rt.ops
        C, B, st = p["C"], p["B"], p["stages"]
        xs = ops.mult(x, scale)
        ap, am = ops.add(xs, C), ops.add(xs, -C)
        # |x' + C| = (x' + C) sgn: x' + C on [-B, B] <=> x on [(-B - C) / scale, (B - C) / scale]
        abs_p = ops.mult(ap, self._sgn(ops, x, st, (-B - C) / scale, (B - C) / scale))
        abs_m = ops.mult(am, self._sgn(ops, x, st, (-B + C) / scale, (B + C) / scale))
        clip = ops.mult(ops.sub(abs_p, abs_m), 0.5)
        g = eval_chebyshev(ops, clip, p["phi"]["cheb"], p["phi"]["a"], p["phi"]["b"])
        return ops.mult(xs, g)
