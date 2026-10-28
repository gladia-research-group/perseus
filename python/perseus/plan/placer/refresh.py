from __future__ import annotations

import dataclasses
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, field

from .. import btserr
from .ir import FIELD_REAL, Graph, is_kv_cache_read
from .sim import PlannerShadow

log = logging.getLogger(__name__)


def mag_safety_of(step: str | None, base: float) -> float:
    """The drift margin κ a site of `step` is sized with: `base` (PLAN_MAG_SAFETY), or the largest value of
    PLAN_MAG_SAFETY_STEPS (`name:kappa,...`) whose `name` begins the full step name (`blk11.attn` covers blk11.attn and
    blk11.attn.softmax; `tail` the tail), never below `base`. For the sites whose capture holds few independent values:
    one position's (a prefill's last query and its tail) varies across prompts far more than a max over all of them."""
    k = base
    if step:
        for kv in (os.environ.get("PLAN_MAG_SAFETY_STEPS") or "").split(","):
            name, _, v = kv.partition(":")
            if name.strip() and v.strip() and step.startswith(name.strip()):
                k = max(k, float(v))
    return k


@dataclass(frozen=True)
class RefreshSpec:
    var: str
    feasible: bool
    hopeless: bool
    rel_err: float
    cf: int | None = None
    route: int = 0
    prescale: float | None = None
    offset: float | None = None
    out_consumed: float = 0.0
    out_deg: int = 1
    reason: str = ""
    raise_drop: int = 0
    # `cf == policy.cf_max` because the search ran out of range (btserr.ceiling_binds).
    # Never enters a plan file: emit reads the named fields it needs, nothing dumps a spec.
    clamped: bool = False
    # StC-first runs: a dense site bootstrapped in the shipped order (plan 'shipped_order', PLAN_STC_FIRST_MAG_MAX)
    shipped: bool = False

    def overshoot(self, target: float) -> float:
        import math
        if self.feasible or not math.isfinite(self.rel_err) or self.rel_err <= 0:
            return 0.0
        return max(0.0, math.log10(self.rel_err / max(target, 1e-300)))


@dataclass(frozen=True)
class CfClampReport:
    """Where the CF ceiling binds, over the typed sites that carry a CF in the plan.

    Also written, as plain numbers and lists, into the plan file's `summary.bts_quality`;
    these are summary keys only and do not affect what the runtime executes.
    `num_pinned` counts `cf == cf_max` for any reason; `clamped` is the subset where the
    ceiling binds (`btserr.ceiling_binds`), so `num_clamped <= num_pinned <= num_sites`.
    """
    cf_max: int
    num_sites: int                           # typed sites with a CF (= sum of cf_histogram)
    num_pinned: int                          # cf == cf_max, for any reason
    clamped: tuple[str, ...]                 # pinned AND cf_max + 1 would win
    clamped_missing_target: tuple[str, ...]  # subset of `clamped` whose refresh misses err_target

    @property
    def num_clamped(self) -> int:
        return len(self.clamped)


def _step_landing_floor(spec: str, step: str):
    """The plan.step_landing (PLAN_STEP_LANDING) floor (absolute level) of a refresh in `step`, or None."""
    best = None
    for item in spec.split(","):
        if ":" not in item:
            continue
        key, lv = item.rsplit(":", 1)
        if key and key in step.split("."):
            best = max(best or 0, int(lv))
    return best


@dataclass
class RefreshPlanner:
    table: btserr.AccuracyTable
    policy: btserr.SitePolicy
    graph: Graph
    sparse_precomps: tuple[int, ...] = ()
    sparse_out_levels: dict[int, int] = field(default_factory=dict)
    site_out_levels: dict[str, int] = field(default_factory=dict)
    out_deg: int = 1
    bootstrap_level: int = 16
    step_bts_offset: dict[str, float] = field(default_factory=dict)
    #: the route landings and the recipe knobs (pc-shadow, planner mode); None = the recipe of the fields above
    shadow: PlannerShadow | None = None
    #: dense refreshes of real-tagged slots take the real route (plan.real_route): priced on its real read
    real_route: bool = False
    #: the drop each route runs at by default (PlanConfig.raise_drop_base): a site's emitted raise_drop is absolute
    raise_drop_base: dict[int, int] = field(default_factory=dict)
    _cache: dict[str, RefreshSpec] = field(default_factory=dict)
    _warned_routes: set = field(default_factory=set)

    def __post_init__(self):
        if self.shadow is None:
            from perseus import _pc
            spec = ",".join(f"{s}:{lv}" for s, lv in sorted(self.sparse_out_levels.items()))
            plan = _pc.PlanConfig.from_legacy_env({"PLAN_SPARSE_BTS_OUT": spec} if spec else {})
            self.shadow = PlannerShadow(self.bootstrap_level, self.graph.level_unit, plan)

    def route_for(self, v: str) -> int | None:
        p = self.graph.producer_of.get(v)
        per = p.pack_period if p else None
        if per is None or per <= 0:
            return None
        for s in sorted(self.sparse_precomps):
            if s > 0 and s % per == 0:
                return s
        return None

    def spec(self, v: str) -> RefreshSpec:
        got = self._cache.get(v)
        if got is not None:
            return got
        spec = self._build(v)
        self._cache[v] = spec
        return spec

    def spec_shipped(self, v: str) -> RefreshSpec:
        """`spec` of a dense site that runs the shipped order (an input past the StC-first entry): priced without the
        StC-first sine term, landed where that order lands."""
        got = self._cache.get(("shipped", v))
        if got is None:
            got = self._build(v, force_shipped=True)
            self._cache[("shipped", v)] = got
        return got

    def _build(self, v: str, force_shipped: bool = False) -> RefreshSpec:
        p = self.graph.producer_of.get(v)
        if p is None and not is_kv_cache_read(v):
            return RefreshSpec(v, False, True, float("inf"),
                               reason="fresh graph input — never refreshable")
        route = self.route_for(v)
        pol = self.policy
        kappa = mag_safety_of(p.step if p is not None else None, pol.mag_safety)
        if kappa != pol.mag_safety:
            pol = dataclasses.replace(pol, mag_safety=kappa)
        site = dict(
            slot_mag=p.max_abs if p else None,
            period=p.pack_period if p else None,
            coeff_mag=p.max_coeff if p else None,
            coeff_mag_ac=p.max_coeff_ac if p else None,
            mean=p.mean if p else None,
            residual_mag=p.max_dev if p else None,
            routes_avail=(route,) if route else (),
            levels_free=1 if pol.allow_prescale else 0,
            real=bool(self.real_route and p is not None and p.pack_field == FIELD_REAL),
            slot_mag_lo=p.max_abs_lo if p else None,
            coeff_mag_lo=p.max_coeff_lo if p else None,
        )
        # PLAN_STEP_ROUTE_PRICING=1: a sparse site of a step whose route follows the cache length (a PLAN_STEP_LANDING
        # step, SM_PERIODIC=2) runs on the bigger routes too once its value's period outgrows the planned one: its CF is
        # priced on the worst of them the accuracy table measured
        if route and p is not None and os.environ.get("PLAN_STEP_ROUTE_PRICING", "0") not in ("", "0") \
                and _step_landing_floor(self.shadow.plan.get("plan.step_landing") or "", p.step) is not None:
            # with the period range captured over several positions, only a value whose period outgrows the planned
            # route moves, and only up to the route its largest period fills
            pmax = p.pack_period_max
            site["price_routes"] = tuple(r for r in self.table.routes() if r > route and (pmax is None or r <= pmax))
        choice = btserr.choose_site(self.table, pol, **site)
        # StC-first (policy.slim_cubic > 0): a dense site whose slot magnitude the StC-first order cannot hold at the
        # shipped order's precision (plan.stc_first_mag_max, after the drift margin) bootstraps in the shipped order:
        # priced without the slim term, landed where that order lands (the Shadow's shipped-order dense landing,
        # plan.stc_first_fallback)
        shipped = False
        sh = self.shadow
        _mag_max = sh.plan.get("plan.stc_first_mag_max")
        # plan.stc_first_steps (PLAN_STC_FIRST_STEPS, comma list of step names, the last dotted part): only those
        # steps' dense sites keep StC-first, every other dense site takes the shipped order
        _steps = sh.plan.get("plan.stc_first_steps")
        _step_ok = True
        if _steps is not None and p is not None:
            _step_ok = (p.step or "").split(".")[-1] in {x.strip() for x in _steps.split(",") if x.strip()}
        _too_big = bool(_mag_max) and p is not None and (p.max_abs or 0.0) * pol.mag_safety > float(_mag_max)
        # PLAN_STC_FIRST_STAGED_ONLY=1: the runtime runs the StC-first order on the staged PCMM / CCMM finishes only
        # (FIDESLIB_BTS_STC_FIRST_STAGED_ONLY=1, no FIDESLIB_BTS_STC_FIRST on the plan's runtime line): every refresh
        # site of the plan bootstraps in the shipped order, so none takes the slim (slot) law; landed where that order
        # lands when the recipe names it (plan.stc_first_fallback), else at the recipe's dense landing as before
        _staged_only = os.environ.get("PLAN_STC_FIRST_STAGED_ONLY", "0") not in ("", "0")
        if (pol.slim_cubic > 0 and not choice.route and p is not None
                and (_staged_only or (sh.has_shipped and (_too_big or not _step_ok or force_shipped)))):
            import dataclasses as _dc
            choice = btserr.choose_site(self.table, _dc.replace(pol, slim_cubic=0.0), **site)
            shipped = not choice.route
        # one extra search, only for a site pinned at cf_max; the choice above is unchanged
        clamped = btserr.ceiling_binds(self.table, pol, choice.cf, **site)

        route_land = sh.landing(choice.route) if choice.route else None
        if route_land is not None:
            base_route = float(route_land[0]) - self.bootstrap_level
        elif choice.route:
            if choice.route not in self._warned_routes:
                self._warned_routes.add(choice.route)
                log.info(f"[plan]  routing sites sparse at s={choice.route} with no measured "
                      f"landing level; assuming dense. Measure it from a "
                      f"`[planted_bts] ... out=` ledger and pass --sparse-bts-out.")
            base_route = max(0.0, self.step_bts_offset.get(p.step, 0.0)) if p else 0.0
        else:
            base_route = max(0.0, self.step_bts_offset.get(p.step, 0.0)) if p else 0.0
        raise_drop = 0
        if v in self.site_out_levels:
            base = float(self.site_out_levels[v]) - self.bootstrap_level
            # a landing deeper than the route's own is a level-aware (partial) ModRaise
            unit = float(self.graph.level_unit) or 1.0
            raise_drop = max(0, int(round((base - base_route) / unit)))
            if raise_drop:
                raise_drop += self.raise_drop_base.get(choice.route or 0, 0)
        else:
            base = base_route
        # plan.step_landing="softmax:32,...": refreshes in those steps are planned to land no shallower than that
        # (their route changes with the cache length under SM_PERIODIC=2; a shallower landing is headroom)
        floor = _step_landing_floor(sh.plan.get("plan.step_landing") or "", p.step if p else "")
        if floor is not None:
            base = max(base, float(floor) - self.bootstrap_level)
        out_deg = self.out_deg
        if shipped and sh.has_shipped:
            _lv, _dg = sh.shipped_landing()
            base, out_deg = float(_lv) - self.bootstrap_level, int(_dg)
        restore = float(self.graph.level_unit) if choice.prescale is not None else 0.0
        return RefreshSpec(
            var=v,
            feasible=choice.feasible,
            hopeless=choice.hopeless,
            rel_err=choice.rel_err,
            cf=choice.cf,
            route=choice.route,
            prescale=choice.prescale,
            offset=choice.offset,
            out_consumed=base + restore,
            out_deg=out_deg,
            reason=choice.reason,
            raise_drop=raise_drop,
            clamped=clamped,
            shipped=shipped,
        )

    def cf_clamp_report(self, sites: Iterable[str]) -> CfClampReport:
        """Count the CF ceiling over `sites` (the placed + fired-hint sites emit gives a CF).

        Reads the memoised specs, so it costs nothing beyond what the cut already priced.
        """
        typed = [self.spec(v) for v in sites]
        typed = [s for s in typed if s.cf is not None]
        pinned = [s for s in typed if s.cf == self.policy.cf_max]
        clamped = tuple(sorted(s.var for s in pinned if s.clamped))
        missing = tuple(v for v in clamped if not self.spec(v).feasible)
        return CfClampReport(cf_max=self.policy.cf_max, num_sites=len(typed),
                             num_pinned=len(pinned), clamped=clamped,
                             clamped_missing_target=missing)
