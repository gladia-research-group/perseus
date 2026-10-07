from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from .. import btserr
from .ir import Graph, is_kv_cache_read

log = logging.getLogger(__name__)


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
    _cache: dict[str, RefreshSpec] = field(default_factory=dict)
    _warned_routes: set = field(default_factory=set)

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

    def _build(self, v: str) -> RefreshSpec:
        p = self.graph.producer_of.get(v)
        if p is None and not is_kv_cache_read(v):
            return RefreshSpec(v, False, True, float("inf"),
                               reason="fresh graph input — never refreshable")
        route = self.route_for(v)
        site = dict(
            slot_mag=p.max_abs if p else None,
            period=p.pack_period if p else None,
            coeff_mag=p.max_coeff if p else None,
            coeff_mag_ac=p.max_coeff_ac if p else None,
            mean=p.mean if p else None,
            residual_mag=p.max_dev if p else None,
            routes_avail=(route,) if route else (),
            levels_free=1 if self.policy.allow_prescale else 0,
        )
        choice = btserr.choose_site(self.table, self.policy, **site)
        # one extra search, only for a site pinned at cf_max; the choice above is unchanged
        clamped = btserr.ceiling_binds(self.table, self.policy, choice.cf, **site)

        if choice.route and choice.route in self.sparse_out_levels:
            base_route = float(self.sparse_out_levels[choice.route]) - self.bootstrap_level
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
        else:
            base = base_route
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
            out_deg=self.out_deg,
            reason=choice.reason,
            raise_drop=raise_drop,
            clamped=clamped,
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
