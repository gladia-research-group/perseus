"""Exact bootstrap placement: the stage-4 decision as a mixed-integer program (HiGHS via `scipy.optimize.milp`).

A transcription of `sim.simulate`: each var's effective level is a unary LADDER of 0/1 "eff >= k" quantities, so
max / refresh / hint / budget become exact logic over the refresh decisions x (all declared integer: continuous
logic variables accumulate tolerance through fan-in). Objectives: "total" (placed + fired hints), "placed" (the
min-cut's), "ms" (route latency). The placement is re-simulated and kept only if the sim agrees on every level,
degree and hint; the min-cut runs first as incumbent and fallback. Model details: docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import bisect
import logging
import math
import os
import time
from dataclasses import dataclass, field

from .ir import ADD_FAMILY, MULT_FAMILY, is_kv_cache_read, is_literal_input, is_plaintext_name
from .place import QUALITY_LAMBDA, Placer, PlanInfeasible, refresh_env_cap
from .sim import SimResult

log = logging.getLogger(__name__)

_INF = float("inf")
_TOL = 1e-4           # agreement tolerance between the solved model and the sim
_ENV_PENALTY = 1.0e4  # `_capacity`'s payable price for a site past the refresh envelope


class ModelMismatch(RuntimeError):
    """The solved model disagrees with `simulate()` on a placement."""


# ── a minimal linear-model builder ────────────────────────────────────────────────────

class Expr:
    """const + sum(coef * var), with variables as column indices of a `_Model`."""

    __slots__ = ("terms", "const")

    def __init__(self, terms: dict[int, float] | None = None, const: float = 0.0):
        self.terms = terms or {}
        self.const = float(const)

    @property
    def is_const(self) -> bool:
        return not self.terms

    def key(self):
        return (tuple(sorted(self.terms.items())), self.const)

    def __add__(self, o):
        if not isinstance(o, Expr):
            return Expr(dict(self.terms), self.const + o)
        t = dict(self.terms)
        for k, c in o.terms.items():
            t[k] = t.get(k, 0.0) + c
            if t[k] == 0.0:
                del t[k]
        return Expr(t, self.const + o.const)

    __radd__ = __add__

    def __neg__(self):
        return Expr({k: -c for k, c in self.terms.items()}, -self.const)

    def __sub__(self, o):
        return self + (-o if isinstance(o, Expr) else -o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, s: float):
        if s == 0:
            return Expr()
        return Expr({k: c * s for k, c in self.terms.items()}, self.const * s)

    __rmul__ = __mul__


def K(c: float) -> Expr:
    return Expr(None, c)


ONE, ZERO = K(1.0), K(0.0)


def _is(e: Expr, c: float) -> bool:
    return e.is_const and e.const == c


@dataclass
class _Model:
    lb: list[float] = field(default_factory=list)
    ub: list[float] = field(default_factory=list)
    integer: list[int] = field(default_factory=list)
    rows: list[tuple[dict[int, float], float, float]] = field(default_factory=list)
    infeasible: str | None = None
    #: column -> how its value follows from earlier columns (the logic variables); the
    #: columns without one are the decisions. Creation order is a topological order.
    defs: dict = field(default_factory=dict)
    _or: dict = field(default_factory=dict)
    _and: dict = field(default_factory=dict)

    def var(self, lo: float = 0.0, hi: float = 1.0, integer: bool = False) -> Expr:
        i = len(self.lb)
        self.lb.append(lo)
        self.ub.append(hi)
        self.integer.append(1 if integer else 0)
        return Expr({i: 1.0})

    def bounds(self, e: Expr) -> tuple[float, float]:
        lo = hi = e.const
        for k, c in e.terms.items():
            if c > 0:
                lo += c * self.lb[k]
                hi += c * self.ub[k]
            else:
                lo += c * self.ub[k]
                hi += c * self.lb[k]
        return lo, hi

    def add(self, e: Expr, lo: float = -_INF, hi: float = _INF, what: str = "") -> None:
        """lo <= e <= hi. A row the bounds already imply is dropped; a constant row that
        fails marks the model infeasible (with `what` as the reason)."""
        blo, bhi = self.bounds(e)
        if blo >= lo - 1e-9 and bhi <= hi + 1e-9:
            return
        if e.is_const or bhi < lo - 1e-9 or blo > hi + 1e-9:
            if self.infeasible is None:
                self.infeasible = what or "a constant row is violated"
            return
        self.rows.append((dict(e.terms), lo - e.const, hi - e.const))

    def propagate(self, decisions: dict[int, float]):
        """Every column's value from the decisions alone, in creation order (no solver:
        a pinned model is fully determined). Returns (values, the first violated row's
        index or None)."""
        import numpy as np
        sol = np.zeros(len(self.lb))

        def ev(e: Expr) -> float:
            return e.const + sum(w * sol[k] for k, w in e.terms.items())

        for i in range(len(self.lb)):
            d = self.defs.get(i)
            if d is None:
                sol[i] = decisions.get(i, 0.0)
            elif d[0] == "or":
                sol[i] = 1.0 if any(ev(b) > 0.5 for b in d[1]) else 0.0
            elif d[0] == "and":
                sol[i] = 1.0 if all(ev(b) > 0.5 for b in d[1]) else 0.0
            else:                                       # "addz": 1 - A_k where E == k
                sol[i] = next((1.0 - round(ev(A)) for t, A in d[1] if ev(t) > 0.5), 1.0)
        for r, (terms, lo, hi) in enumerate(self.rows):
            v = sum(w * sol[k] for k, w in terms.items())
            if v < lo - 1e-6 or v > hi + 1e-6:
                return sol, r
        return sol, None

    # 0/1 logic, exact at integer points
    def OR(self, bits: list[Expr]) -> Expr:
        live = {}
        for b in bits:
            if _is(b, 1.0):
                return ONE
            if not _is(b, 0.0):
                live.setdefault(b.key(), b)
        if not live:
            return ZERO
        if len(live) == 1:
            return next(iter(live.values()))
        key = tuple(sorted(live))
        got = self._or.get(key)
        if got is None:
            y = self.var(integer=True)
            tot = ZERO
            for b in live.values():
                self.add(b - y, hi=0.0)
                tot = tot + b
            self.add(y - tot, hi=0.0)
            self.defs[next(iter(y.terms))] = ("or", list(live.values()))
            self._or[key] = got = y
        return got

    def AND(self, a: Expr, b: Expr) -> Expr:
        if _is(a, 0.0) or _is(b, 0.0):
            return ZERO
        if _is(a, 1.0):
            return b
        if _is(b, 1.0):
            return a
        if a.key() == b.key():
            return a
        key = tuple(sorted((a.key(), b.key())))
        got = self._and.get(key)
        if got is None:
            y = self.var(integer=True)
            self.add(y - a, hi=0.0)
            self.add(y - b, hi=0.0)
            self.add(y - a - b, lo=-1.0)
            self.defs[next(iter(y.terms))] = ("and", [a, b])
            self._and[key] = got = y
        return got

    def ITE(self, c: Expr, a: Expr, b: Expr) -> Expr:
        """c ? a : b over 0/1 quantities."""
        if _is(c, 1.0):
            return a
        if _is(c, 0.0):
            return b
        if a.key() == b.key():
            return a
        return self.OR([self.AND(c, a), self.AND(1 - c, b)])


class Ladder:
    """A level in unary: `vals` are the values it can take (ascending), `bits[i]` is the
    0/1 quantity [value >= vals[i + 1]]."""

    __slots__ = ("vals", "bits")

    def __init__(self, vals: list[float], bits: list[Expr]):
        self.vals, self.bits = vals, bits

    @classmethod
    def const(cls, v: float) -> Ladder:
        return cls([float(v)], [])

    def ge(self, k: float) -> Expr:
        """[value >= k]"""
        if k <= self.vals[0]:
            return ONE
        i = bisect.bisect_left(self.vals, k)
        return self.bits[i - 1] if i < len(self.vals) else ZERO

    def gt(self, k: float) -> Expr:
        """[value > k]"""
        i = bisect.bisect_right(self.vals, k)
        if i == 0:
            return ONE
        return self.bits[i - 1] if i < len(self.vals) else ZERO

    def shift(self, s: float) -> Ladder:
        return Ladder([v + s for v in self.vals], self.bits)

    def value(self, sol) -> float:
        """The level at a solution. Every rung is exactly 0/1 once the decisions are
        integral; rounding strips the solver's feasibility tolerance (~1e-4) off the
        continuous logic variables before the result is compared with the sim."""
        out = self.vals[0]
        for i, b in enumerate(self.bits):
            bv = round(b.const + sum(w * sol[k] for k, w in b.terms.items()))
            out += (self.vals[i + 1] - self.vals[i]) * bv
        return out


# ── the placer ────────────────────────────────────────────────────────────────────────

@dataclass
class IlpPlacer(Placer):
    #: solver wall-clock cap per block, seconds
    time_limit: float = 300.0
    #: relative optimality gap at which the solver may stop (0 = prove the optimum)
    mip_gap: float = 0.0
    #: "total" prices placed refreshes AND fired hints; "placed" only the former (min-cut's);
    #: "ms" prices both at their route's measured bootstrap latency (`bts_ms`)
    objective: str = "total"
    #: raise ModelMismatch instead of falling back when the sim disagrees (tests set it)
    strict: bool = False
    #: forbid exiting the block deeper than the min-cut's plan does. Set on the model's
    #: final block: its exit feeds the encrypted argmax, which no plan prices, so a
    #: deeper exit there is a saving the planner books and the argmax pays back
    #: (measured: n32 block 12 exited 42 instead of 40, argmax +0.13 s/token)
    cap_exit: bool = False
    #: a hint's firing is a decision too: the plan binds every hint (`hint_fire`), so the
    #: model may veto one its threshold fires (as PLAN_HINT_ENV_VETO does past the envelope)
    #: when a refresh elsewhere serves better. A vetoed hint is pinned off for the sim.
    free_hints: bool = False
    #: among the optima, take the one a fixed weight per decision (from the site's graph position) ranks first: a
    #: second solve under the optimal cost, so the plan does not depend on the solver build's tie-breaking
    tiebreak: bool = False
    meta: dict = field(default_factory=dict)
    _exit_cap: tuple[str, float] | None = None
    #: the min-cut incumbent's hint pins (its envelope vetoes), restored with its plan
    _mc_force: dict | None = None

    heuristic_config = "ilp"

    # ── pricing ──
    def _route_ms(self, v: str) -> float:
        """Measured latency of the refresh at v: its route's, 0 = dense. A site's route is
        fixed by its packing (`RefreshPlanner.route_for`), so this is a constant per site
        and the "ms" objective stays linear. The min-cut's `ms_discount` credit for a
        richer landing is not applied: the runway it buys is modelled exactly here, as the
        later refreshes it makes unnecessary."""
        route = self.refresh.spec(v).route or 0
        return self.bts_ms.get(route) or self.bts_ms.get(0, 1.0)

    def _fallback_ms(self) -> float:
        """Extra latency of a dense refresh that takes the shipped order (its input past the
        StC-first entry) over the StC-first one, when the recipe prices the two (`bts_ms`
        key -1); 0 otherwise."""
        shipped = self.bts_ms.get(-1)
        return max(0.0, shipped - self.bts_ms.get(0, shipped)) if shipped else 0.0

    def _site_cost(self, v: str) -> float:
        spec = self.refresh.spec(v)
        cost = 1.0 + self.miss_penalty * spec.overshoot(self.err_target)
        if self.objective == "ms":
            cost *= self._route_ms(v)
        if self.quality_weight > 0.0:
            q = min(1.0, max(0.0, spec.rel_err / max(self.err_target, 1e-300)))
            cost += QUALITY_LAMBDA * self.quality_weight * q
        return cost

    def _hint_cost(self, v: str) -> float:
        return 0.0 if self.objective == "placed" else self._site_cost(v)

    def _env_hard(self) -> bool:
        ov = os.environ.get("PLAN_HARD_ENV_CAP")          # same override as _capacity
        if ov == "1":
            return True
        if ov == "0":
            return False
        return self.hard_env_cap

    def _candidate(self, v: str) -> bool:
        """Mirror of the finite branch of `_capacity`, minus its state-dependent
        "already at a refreshed level" test (the objective makes a useless refresh
        unattractive, and the sim judges the rest). Fold/deliberate outputs are refresh
        points already; the runtime does not fire a planted bootstrap after them."""
        g = self.g
        p = g.producer_of.get(v)
        if p is None and not (v in g.inputs and is_kv_cache_read(v)):
            return False
        if p is not None and (p.is_deliberate_bts or p.is_fold_bts):
            return False
        if self._forbidden(v):
            return False
        return not self.refresh.spec(v).hopeless

    # ── the model ──
    def _build(self, candidates: set[str], fixed: set[str] | None,
               env_hard: bool | None = None, constrain: bool = True):
        """Transcribe `simulate()`. With `fixed`, every x is pinned to membership in it
        (evaluation mode). `env_hard` overrides the envelope mode (True once the
        incumbent is cheaper than one envelope penalty, which then is never paid).
        `constrain=False` drops the budget and envelope rows, leaving the pure level
        algebra (`twin` uses it to compare against the sim on any placement)."""
        g, m = self.g, _Model()
        unit = float(g.level_unit)
        bts = float(self.bootstrap_level)
        forced = set(self.placed)
        hint_force = self.hint_force or {}
        realized = self.realized or set()
        offs = self._step_offsets or {}
        cost: dict[int, float] = {}

        x: dict[str, Expr] = {}

        def x_of(v: str) -> Expr:
            if v not in x:
                if v in forced:
                    x[v] = ONE
                elif fixed is not None:
                    x[v] = ONE if v in fixed else ZERO
                elif v in candidates:
                    x[v] = e = m.var(integer=True)
                    (k,) = e.terms
                    cost[k] = self._site_cost(v)
                else:
                    x[v] = ZERO
            return x[v]

        def bit(d: int) -> Expr:
            return ONE if d == 2 else ZERO

        def landing(v: str) -> tuple[Ladder, Expr]:
            s = self.refresh.spec(v)
            if s.hopeless:                      # sim: a hopeless hint lands at (0, deg 2)
                return Ladder.const(unit), ONE
            d = bit(s.out_deg)
            return Ladder.const(s.out_consumed + unit * d.const), d

        sh = self.refresh.shadow
        fb = self._dense_fallback()
        fb_deep: dict[str, Expr] = {}           # [input past the StC-first entry] per dense site

        def landing_from(v: str, pe: Ladder) -> tuple[Ladder, Expr]:
            """`landing(v)` for a refresh whose input is `pe`: a dense site whose input passed
            the StC-first entry runs the shipped order and lands where it lands (sim
            `dense_fallback`)."""
            le, ld = landing(v)
            s = self.refresh.spec(v)
            if fb is None or (s.route and not s.hopeless):
                return le, ld
            deep = pe.gt(fb[0] - bts)
            if _is(deep, 0.0):
                return le, ld
            fb_deep[v] = deep
            fd = bit(fb[2])
            return lite(deep, Ladder.const(fb[1] + unit * fd.const), le), m.ITE(deep, fd, ld)

        def rungs(vals: list[float], bits: list[Expr]) -> Ladder:
            """A ladder, with its rungs stated to be monotone: [v >= k+1] implies [v >= k].
            Every integer solution satisfies this by construction, but saying it keeps the
            solver out of the subtrees where a rung pattern denotes no level at all."""
            for lo, hi in zip(bits, bits[1:]):
                m.add(hi - lo, hi=0.0)
            return Ladder(vals, bits)

        def lmax(ls: list[Ladder]) -> Ladder:
            uniq = {}
            for la in ls:
                uniq.setdefault((tuple(la.vals), tuple(b.key() for b in la.bits)), la)
            ls = list(uniq.values())
            if len(ls) == 1:
                return ls[0]
            floor = max(la.vals[0] for la in ls)
            vals = sorted({v for la in ls for v in la.vals if v >= floor})
            return rungs(vals, [m.OR([la.ge(k) for la in ls]) for k in vals[1:]])

        def lite(c: Expr, a: Ladder, b: Ladder) -> Ladder:
            """c ? a : b"""
            if _is(c, 1.0):
                return a
            if _is(c, 0.0):
                return b
            vals = sorted(set(a.vals) | set(b.vals))
            return rungs(vals, [m.ITE(c, a.ge(k), b.ge(k)) for k in vals[1:]])

        # state: var -> (eff ladder, deg-2 bit)
        E: dict[str, Ladder] = {}
        D: dict[str, Expr] = {}
        pre: dict[str, Ladder] = {}             # the input of each refresh x can place

        def eff(v: str) -> Ladder:
            return E.get(v) or Ladder.const(0.0)

        def dg(v: str) -> Expr:
            return D.get(v, ZERO)

        fires: dict[str, Expr] = {}             # [the refresh at v runs], per site the model refreshes
        hard = self._env_hard() if env_hard is None else env_hard

        def env_rel(v: str) -> float:
            return refresh_env_cap(self._step_of(v), route=self.refresh.spec(v).route or 0) - bts

        def refresh(v: str, e: Ladder, d: Expr, xv: Expr, seed: bool = False) -> tuple[Ladder, Expr]:
            if _is(xv, 0.0):
                return e, d
            pre[v] = e
            fires[v] = xv
            # a refreshed graph input is seeded at its landing (the sim's StC-first test reads op outputs only);
            # a hard envelope at or inside the StC-first entry leaves a placed refresh no input past it
            inside = constrain and hard and fb is not None and env_rel(v) <= fb[0] - bts
            le, ld = landing(v) if seed or inside else landing_from(v, e)
            return lite(xv, le, e), m.ITE(xv, ld, d)

        for v in g.inputs:
            d0 = bit(self.seed_deg(v))
            e0 = Ladder.const(self.seed_consumed(v) + unit * d0.const)
            E[v], D[v] = refresh(v, e0, d0, x_of(v), seed=True)

        terminal = None
        for n in g.nodes:
            if n.output and not n.is_deliberate_bts:
                terminal = n

        def budget(n, c_e: Ladder, c_d: Expr, xo: Expr) -> Ladder:
            """consumed = eff - unit*d <= b, where b depends on "out is refreshed" (`xo`;
            always true for a hint output, whose name is in the sim's refreshed map).
            Returns the ladder cut at the loosest ceiling: the rows just added force every
            rung above it to 0, so dropping them changes nothing but the model's size."""
            if not constrain:
                return c_e
            kw = dict(is_terminal=(n is terminal), bootstrap_level=bts)
            b_of = {False: self.budget.node_budget(n, output_refreshed=False, **kw),
                    True: self.budget.node_budget(n, output_refreshed=True, **kw)}
            xs = ([(False, ONE)] if _is(xo, 0.0) else [(True, ONE)] if _is(xo, 1.0)
                  else [(False, 1 - xo), (True, xo)])
            ds = ([(c_d.const == 1.0, ONE)] if c_d.is_const
                  else [(False, 1 - c_d), (True, c_d)])
            top = -_INF
            for xr, x_lit in xs:
                for dv, d_lit in ds:
                    ceil = b_of[xr] + (unit if dv else 0.0)
                    top = max(top, ceil)
                    slack = (ZERO if _is(x_lit, 1.0) else 1 - x_lit) + \
                            (ZERO if _is(d_lit, 1.0) else 1 - d_lit)
                    m.add(c_e.gt(ceil) - slack, hi=0.0,
                          what=f"budget at node {n.idx} ({n.op} -> {n.output})")
            j = bisect.bisect_right(c_e.vals, top)
            if 0 < j < len(c_e.vals):
                return Ladder(c_e.vals[:j], c_e.bits[:j - 1])
            return c_e

        hints: dict[int, Expr] = {}
        vetoes: dict[str, tuple[Expr, Expr]] = {}   # hint output -> (its threshold, its "keep" decision)
        for n in g.nodes:
            ins = n.cipher_inputs
            out = n.output
            if n.is_deliberate_bts:
                if out:
                    base = (n.output_level - bts if n.output_level is not None
                            else max(0.0, offs.get(n.step, 0.0)))
                    if self.deliberate_clamp0 and base + unit < 0:
                        base = 0.0
                    # the captured landing degree (a PCMM finish pends its rescale, a CCMM's ends
                    # rescaled); StC-first runs push K/V in the shipped order (kv_landing)
                    dd = bit(n.output_deg if n.output_deg is not None else 2)
                    if sh.kv_landing is not None:
                        base, dd = sh.kv_landing[0], bit(sh.kv_landing[1])
                    E[out], D[out] = Ladder.const(base + unit * dd.const), dd
                # a captured StC-first refresh past the entry takes the shipped order: re-routed to its raw
                # landing (plan.stc_first_reroute), else a budget violation
                if n.stc_first and fb is not None and ins:
                    deep = lmax([eff(v) for v in ins]).gt(fb[0] - bts)
                    if sh.stc_first_reroute and out:
                        c_fb, d_fb = (fb[1] - unit, 2) if fb[2] == 1 else (fb[1], fb[2])
                        rd = bit(d_fb)
                        E[out] = lite(deep, Ladder.const(c_fb + unit * rd.const), E[out])
                        D[out] = m.ITE(deep, rd, D[out])
                    elif constrain:
                        m.add(deep, hi=0.0, what=f"StC-first entry at node {n.idx} ({out})")
                continue

            xo = ZERO
            if n.hint_level is not None:
                ie = eff(ins[0]) if ins else Ladder.const(0.0)
                idg = dg(ins[0]) if ins else ZERO
                if out and out in hint_force:
                    h = ONE if hint_force[out] else ZERO
                else:
                    h = ie.gt(n.hint_level - bts)
                    if self.free_hints and out and fixed is None and not _is(h, 0.0):
                        keep = m.var(integer=True)
                        vetoes[out] = (h, keep)
                        h = m.AND(h, keep)
                hints[n.idx] = h
                if out:
                    xo = x_of(out)
                    y = m.OR([h, xo])           # fired, or placed: the refreshed value
                    if not _is(y, 0.0):
                        pre[out] = ie
                        fires[out] = y
                    le, ld = landing_from(out, ie)
                    e, d = lite(y, le, ie), m.ITE(y, ld, idg)
                else:
                    e, d = ie, idg
                e = budget(n, e, d, ONE)        # the sim's node_out is the post value here
                if out:
                    E[out], D[out] = e, d
            else:
                if not ins:
                    # Shadow.step on a node with no cipher input
                    if n.op == "level_reduce":
                        d = bit(n.output_deg if n.output_deg is not None else 1)
                        c0 = n.output_level - bts if n.output_level is not None else 0.0
                    elif n.op == "fold_bootstrap":
                        d, c0 = ONE, (n.output_level - bts if n.output_level is not None else unit)
                    else:
                        raise PlanInfeasible(f"ilp: node {n.idx} ({n.op}) has no cipher input; "
                                             "not modelled")
                    e = Ladder.const(c0 + unit * d.const)
                elif n.op in MULT_FAMILY:
                    e, d = lmax([eff(v) for v in ins]).shift(unit), ONE
                elif n.op in ADD_FAMILY:
                    if len(ins) >= 2:
                        e, d = self._add_join(m, [eff(v) for v in ins], [dg(v) for v in ins],
                                              lmax)
                    elif any(is_plaintext_name(xx) for xx in n.inputs):
                        realize = True
                        if unit == 1:
                            ct_cap = pt_cap = None
                            for xx, lv in zip(n.inputs, n.input_levels):
                                if is_plaintext_name(xx):
                                    if lv is not None:
                                        pt_cap = lv if pt_cap is None else max(pt_cap, lv)
                                elif not is_literal_input(xx) and lv is not None:
                                    ct_cap = lv if ct_cap is None else max(ct_cap, lv)
                            realize = (pt_cap is not None and ct_cap is not None
                                       and pt_cap > ct_cap)
                        # realize: consumed = eff, deg 1 -- eff itself is unchanged
                        e, d = (eff(ins[0]), ZERO) if realize else (eff(ins[0]), dg(ins[0]))
                    else:
                        e, d = eff(ins[0]), dg(ins[0])
                elif n.op in ("negate", "negate_inplace"):
                    e, d = eff(ins[0]).shift(unit), ONE
                elif n.op == "level_reduce":
                    # the wrapper's drop_to_level: max(cur, target), degree kept
                    e, d = eff(ins[0]), dg(ins[0])
                    if n.output_level is not None:
                        c0 = n.output_level - bts
                        e = (lmax([e, Ladder.const(c0 + unit * d.const)]) if d.is_const
                             else lite(d, lmax([e, Ladder.const(c0 + unit)]), lmax([e, Ladder.const(c0)])))
                elif n.op == "fold_bootstrap":
                    c0 = n.output_level - bts if n.output_level is not None else unit
                    e, d = Ladder.const(c0 + unit), ONE
                else:
                    e, d = eff(ins[0]), dg(ins[0])
                xo = x_of(out) if out else ZERO
                e = budget(n, e, d, xo)
                if out:
                    E[out], D[out] = refresh(out, e, d, xo)
            if out and out in realized:
                D[out] = ZERO                   # a deg-2 value rescales now; eff unchanged

        # refresh envelope, on the input of every refresh a decision can place: the site's own
        # (its step's and its route's, as the min-cut's _capacity reads it)
        env_pen: dict[int, float] = {}
        hint_veto = os.environ.get("PLAN_HINT_ENV_VETO") == "1"
        for v, pe in (pre.items() if constrain else ()):
            xv = x.get(v, ZERO)
            deep = pe.gt(env_rel(v))
            if v in vetoes and hint_veto:
                # a hint the model may veto: vetoed when it would fire past the envelope
                m.add(deep + fires[v], hi=1.0, what=f"hint envelope at {v}")
            elif hard:
                m.add(deep + xv, hi=1.0, what=f"refresh envelope at {v}")
            elif not xv.is_const and not _is(deep, 0.0):
                p = m.AND(deep, xv)             # a refresh past the envelope: payable
                (k,) = p.terms
                (kx,) = xv.terms
                env_pen[k] = env_pen.get(k, 0.0) + _ENV_PENALTY - cost[kx]

        obj_const = 0.0
        terms: list[tuple[Expr, float]] = []
        for i, h in hints.items():
            out = g.nodes[i].output
            w = self._hint_cost(out) if out else 0.0
            if w:
                terms.append((h, w))
        # "ms": a dense refresh past the StC-first entry pays the shipped order's latency
        w_fb = self._fallback_ms() if self.objective == "ms" else 0.0
        for v, deep in (fb_deep.items() if w_fb else ()):
            if v in fires:
                terms.append((m.AND(fires[v], deep), w_fb))
        for e, w in terms:
            obj_const += w * e.const
            for k, c in e.terms.items():
                cost[k] = cost.get(k, 0.0) + w * c
        for k, pen in env_pen.items():
            cost[k] = cost.get(k, 0.0) + pen
        if constrain and fixed is None and self._exit_cap is not None:
            term, cap = self._exit_cap
            if term in E:
                m.add(E[term].gt(cap), hi=0.0, what=f"exit cap at {term}")
        return m, x, E, D, hints, pre, cost, obj_const, vetoes

    @staticmethod
    def _add_join(m: _Model, effs: list[Ladder], degs: list[Expr], lmax):
        """The sim's multi-operand ADD: eff_out = max eff, and deg 2 iff that max is
        attained by deg-2 operands only (a tie with a deg-1 operand goes deg 1).

        z = 1 - A(E), where A(k) = [some deg-1 operand reaches k]. With t_k = [E == k]
        (a difference of adjacent rungs, exactly one of them 1), z = 1 - A_k at the
        k where t_k = 1:   z >= t_k - A_k   and   z <= 2 - t_k - A_k."""
        Eo = lmax(effs)
        if all(_is(d, 1.0) for d in degs):
            return Eo, ONE
        if all(_is(d, 0.0) for d in degs):
            return Eo, ZERO
        vals = Eo.vals
        tA = []
        for j, k in enumerate(vals):
            t = (ONE if j == 0 else Eo.bits[j - 1]) - (Eo.bits[j] if j < len(Eo.bits) else ZERO)
            tA.append((t, m.OR([m.AND(e.ge(k), 1 - d) for e, d in zip(effs, degs)])))
        z = m.var(integer=True)                         # after its operands: see propagate
        m.defs[next(iter(z.terms))] = ("addz", tA)
        for t, A in tA:
            m.add(z - t + A, lo=0.0)
            m.add(z + t + A, hi=2.0)
        return Eo, z

    # ── solve ──
    def _solve(self, m: _Model, cost: dict[int, float], cutoff: float | None,
               obj_const: float, time_limit: float, tie: dict[int, float] | None = None):
        """min cost (or, with `tie`, min tie subject to cost <= cutoff); returns (x, status, cost objective,
        bound)."""
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csr_array

        nv = len(m.lb)
        if nv == 0:                                 # nothing left to decide
            return np.zeros(0), 0, obj_const, obj_const
        c = np.zeros(nv)
        for k, w in (tie if tie is not None else cost).items():
            c[k] = w
        rows = list(m.rows)
        if cutoff is not None and cost:
            rows.append((dict(cost), -_INF, cutoff - obj_const + 1e-6))
        data, ri, ci, lo, hi = [], [], [], [], []
        for r, (terms, a, b) in enumerate(rows):
            for k, w in terms.items():
                data.append(w)
                ri.append(r)
                ci.append(k)
            lo.append(a)
            hi.append(b)
        cons = []
        if rows:
            A = csr_array((data, (ri, ci)), shape=(len(rows), nv))
            cons = [LinearConstraint(A, np.array(lo), np.array(hi))]
        res = milp(c, constraints=cons, integrality=np.array(m.integer),
                   bounds=Bounds(np.array(m.lb), np.array(m.ub)),
                   options={"time_limit": float(time_limit),
                            "mip_rel_gap": float(self.mip_gap), "disp": False})
        bound = getattr(res, "mip_dual_bound", None)
        if tie is not None:
            fun = None if res.x is None else obj_const + sum(w * res.x[k] for k, w in cost.items())
            return res.x, res.status, fun, None
        return (res.x, res.status, (res.fun + obj_const if res.x is not None else None),
                bound + obj_const if bound is not None and math.isfinite(bound) else None)

    def _check(self, sim: SimResult, sol, E, D, hints) -> list[str]:
        """Every place the solved model and the sim disagree (empty = the twin holds)."""
        unit = float(self.g.level_unit)

        def val(e: Expr) -> float:
            return e.const + sum(w * sol[k] for k, w in e.terms.items())

        bad = []
        if sim.over_budget:
            bad.append(f"{len(sim.over_budget)} op(s) over budget")
        for i, h in hints.items():
            if bool(round(val(h))) != bool(sim.hint_fired.get(i, False)):
                bad.append(f"hint {self.g.nodes[i].output}: model {round(val(h))} "
                           f"sim {sim.hint_fired.get(i)}")
        for v, la in E.items():
            md = round(val(D[v]))
            mc = la.value(sol) - unit * md
            sc, sd = sim.consumed.get(v, 0.0), 1 if sim.deg.get(v, 1) == 2 else 0
            if abs(mc - sc) > _TOL or md != sd:
                bad.append(f"{v}: model ({mc:g}, d{md + 1}) sim ({sc:g}, d{sd + 1})")
        return bad

    def twin(self, placed: set[str]) -> list[str]:
        """The model with x pinned to `placed` (feasible or not) vs `simulate()`: every
        disagreement on a level, degree or hint decision. Empty = the transcription
        holds on this placement. Over-budget ops are not a disagreement here."""
        forced = set(self.placed)
        m, x, E, D, hints, pre, cost, oc, vetoes = self._build(set(placed), set(placed),
                                                       constrain=False)
        sol, bad_row = m.propagate({})
        if bad_row is not None:
            return [f"the pinned logic violates its own row {bad_row}"]
        self.placed = forced | set(placed)
        try:
            return [b for b in self._check(self._sim(), sol, E, D, hints)
                    if "over budget" not in b]
        finally:
            self.placed = forced

    def _evaluate(self, placed: set[str]) -> float:
        """Pin x to `placed`, solve the (then fully determined) model and check it
        against the sim. Returns the plan's cost under the ILP objective."""
        forced = set(self.placed)
        m, x, E, D, hints, pre, cost, oc, vetoes = self._build(set(placed), set(placed))
        if m.infeasible:
            raise ModelMismatch(f"model rejects the min-cut plan: {m.infeasible}")
        sol, bad_row = m.propagate({})                  # pinned: no solver needed
        if bad_row is not None:
            raise ModelMismatch(f"model rejects the min-cut plan (row {bad_row})")
        obj = oc + sum(w * sol[k] for k, w in cost.items())
        self.placed = forced | set(placed)
        bad = self._check(self._sim(), sol, E, D, hints)
        self.placed = forced
        if bad:
            raise ModelMismatch("model/sim disagree on the min-cut plan: " + "; ".join(bad[:6]))
        # pinned sites are constants of this model: add their price back
        return obj + sum(self._site_cost(v) for v in placed - forced)

    def run(self) -> SimResult:
        if self.depth_weight > 0.0 or self.level_weight > 0.0:
            raise PlanInfeasible("placer 'ilp' prices count + overshoot only; the depth/level "
                                 "pricing forms depend on levels and are not linear")
        if self.objective not in ("total", "placed", "ms"):
            raise ValueError("ilp objective must be 'total', 'placed' or 'ms', "
                             f"not {self.objective!r}")
        t0 = time.monotonic()
        forced = set(self.placed)
        caller_force = dict(self.hint_force or {})

        # 1. the incumbent: the min-cut, run as is
        mc_placed: set[str] | None = None
        mc_in_eff: dict[str, float] = {}
        try:
            mc_sim = super().run()
            mc_placed = set(self.placed)
            mc_in_eff = dict(self.placed_in_eff)
            if self.cap_exit:
                term = next((n.output for n in reversed(self.g.nodes)
                             if n.output and not n.is_deliberate_bts), None)
                if term is not None:
                    self._exit_cap = (term, mc_sim.effective(term, self.g.level_unit))
        except PlanInfeasible as e:
            log.info(f"[ilp] min-cut incumbent infeasible ({str(e).splitlines()[0]}); "
                     "solving without a cutoff")
        self.placed = set(forced)
        self._mc_force = dict(self.hint_force or {})
        mc_cost = None
        if mc_placed is not None:
            try:
                mc_cost = self._evaluate(mc_placed - forced)
            except (ModelMismatch, PlanInfeasible) as e:
                if self.strict:
                    raise
                log.error(f"[ilp] {e} -- keeping the min-cut plan")
                return self._keep_mincut(mc_placed, mc_in_eff, "model-mismatch", t0)

        # 2. the free model, with the incumbent as a cutoff
        cands = {v for v in (set(self.g.producer_of) | set(self.g.inputs))
                 if v not in forced and self._candidate(v)}
        if mc_placed:
            cands |= mc_placed - forced         # dominance: the incumbent stays reachable
        if self.free_hints:
            # the min-cut's envelope vetoes are decisions of the model now (its own vetoes included)
            self.hint_force = dict(caller_force)
        try:
            m, x, E, D, hints, pre, cost, oc, vetoes = self._build(
                cands, None,
                env_hard=True if mc_cost is not None and mc_cost < _ENV_PENALTY else None)
        except PlanInfeasible as e:
            if mc_placed is None:
                raise
            log.error(f"[ilp] {e} -- keeping the min-cut plan")
            return self._keep_mincut(mc_placed, mc_in_eff, "unmodelled", t0)
        n_int = sum(m.integer)
        self.meta.update(n_vars=len(m.lb), n_int=n_int, n_rows=len(m.rows),
                         mincut_cost=mc_cost,
                         mincut_placed=(len(mc_placed - forced) if mc_placed else None))
        if m.infeasible:
            self.meta.update(reason=m.infeasible)
            if mc_placed is not None:
                return self._keep_mincut(mc_placed, mc_in_eff, "infeasible-model", t0)
            raise PlanInfeasible(f"ilp: no placement can satisfy {m.infeasible}")
        left = max(1.0, self.time_limit - (time.monotonic() - t0))
        sol, status, obj, bound = self._solve(m, cost, mc_cost, oc, left)
        if sol is None and status == 2 and mc_cost is not None:
            # the cutoff priced the incumbent without its envelope penalties; retry bare
            left = max(1.0, self.time_limit - (time.monotonic() - t0))
            sol, status, obj, bound = self._solve(m, cost, None, oc, left)
        if sol is not None and status == 0 and self.tiebreak:
            # a weight per decision column fixed by its var's graph position (no solver state): the optimum is unique
            idx = {next(iter(e.terms)): self.g.producer_of[v].idx if v in self.g.producer_of else 0
                   for v, e in x.items() if not e.is_const}
            idx.update({next(iter(keep.terms)): self.g.producer_of[v].idx for v, (_, keep) in vetoes.items()})
            tie = {k: float(1 + (i * 2654435761 + k) % 1000003) for k, i in idx.items()}
            left = max(1.0, self.time_limit - (time.monotonic() - t0))
            sol2, status2, obj2, _ = self._solve(m, cost, obj, oc, left, tie=tie)
            if sol2 is not None and status2 == 0:
                sol, obj = sol2, obj2
        solve_s = time.monotonic() - t0
        if sol is None:
            what = {1: "time-limit-no-incumbent", 2: "infeasible"}.get(status, f"status-{status}")
            if mc_placed is not None:
                return self._keep_mincut(mc_placed, mc_in_eff, what, t0)
            raise PlanInfeasible(f"ilp: solver returned no placement ({what})")

        def val(e: Expr) -> float:
            return e.const + sum(w * sol[k] for k, w in e.terms.items())

        chosen = {v for v, e in x.items() if round(val(e)) == 1}
        self.placed = forced | chosen
        if vetoes:
            # every free hint is pinned to the model's decision, which the plan binds anyway
            fired = {self.g.nodes[i].output: bool(round(val(h))) for i, h in hints.items()}
            self.hint_force = {**caller_force, **{v: fired[v] for v in vetoes}}
            self.meta["hints_vetoed"] = sorted(v for v, (thr, keep) in vetoes.items()
                                               if round(val(thr)) == 1 and round(val(keep)) == 0)
        sim = self._sim()
        bad = self._check(sim, sol, E, D, hints)
        if bad:
            if self.strict:
                raise ModelMismatch("model/sim disagree on the ILP plan: " + "; ".join(bad[:6]))
            log.error("[ilp] model/sim disagree on the ILP plan: " + "; ".join(bad[:6])
                      + " -- keeping the min-cut plan")
            if mc_placed is not None:
                return self._keep_mincut(mc_placed, mc_in_eff, "model-mismatch", t0)
            raise PlanInfeasible("ilp: model/sim mismatch and no min-cut fallback")

        self.placed_in_eff = {v: pre[v].value(sol) for v in self.placed if v in pre}
        gap = (None if bound is None or obj is None else
               max(0.0, obj - bound) / max(abs(obj), 1e-9))
        self.meta.update(status="optimal" if status == 0 else "time-limit",
                         objective=obj, bound=bound, gap=gap, solve_s=round(solve_s, 2),
                         objective_kind=self.objective)
        if self.verbose:
            log.info(f"[ilp] {self.meta['status']}: cost {obj:.4g} (bound "
                     f"{bound if bound is None else round(bound, 4)}) vs min-cut "
                     f"{mc_cost if mc_cost is None else round(mc_cost, 4)}; "
                     f"{len(chosen)} placed, {sum(1 for f in sim.hint_fired.values() if f)} "
                     f"hint(s) fire; {n_int} integer vars, {solve_s:.1f}s")
        return sim

    def _keep_mincut(self, placed, in_eff, why, t0) -> SimResult:
        self.placed = set(placed)
        self.placed_in_eff = dict(in_eff)
        if self._mc_force is not None:
            self.hint_force = dict(self._mc_force)
        self.meta.update(status=f"fallback:{why}", solve_s=round(time.monotonic() - t0, 2))
        return self._sim()
