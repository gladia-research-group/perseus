"""Fhelipe-style depth-boundary DP placer — CLEAN-ROOM REIMPLEMENTATION.

Upstream (github.com/fhelipe-compiler/fhelipe) is GPL-3. This module was written from
the algorithm description in the PLDI 2024 paper (Krastev et al., "A Tensor Compiler
with Automatic Data Packing for Simple and Efficient Fully Homomorphic Encryption",
Sec. 6) and from prose behavioral notes; NO upstream source code was read into it.

The algorithm, as published:
  * Nodes are partitioned by multiplicative DEPTH; bootstrapping decisions are made at
    depth boundaries only — bootstrap all values crossing a chosen boundary, or none.
  * dp[i] = min over recent boundaries j of dp[j] + cost(bootstrap boundary j) + the
    shortcut values crossing (j, i] that cannot ride through un-refreshed; window depth
    is capped by the usable levels l0. Backtracking yields the boundary set.
  * Boundaries are NARROWED to chokepoints (a single value through which all of a
    depth-component's flow passes) where one exists.
  * SHORTCUT edges (skipping multiple depths, e.g. residuals) are greedily omitted from
    boundary bootstrap sets when the window's level budget can carry them.
  * A post-pass greedily prunes bootstraps that turn out to be unnecessary.

Adaptations to this repo (documented in summary.placer_meta.deviations):
  * depth = monotone accumulation of per-op level cost in composite units (hint sites
    do NOT reset depth here — hints are runtime-owned in every arm; deliberate/fold
    refreshes DO reset, they always execute). The shared sim remains the feasibility
    oracle, and it is strictly more permissive than this model (hints only lower
    levels), so DP-feasible stays twin-feasible.
  * ct count = 1 per var (our IR is ciphertext-level, not tensor-level).
  * shortcut omission uses a per-edge carry bound (carry + window residue <= l0)
    instead of the paper's persistent level-vector shaving; omissions decided
    independently, which the paper's tie-break approximates anyway.
  * the pruning post-pass consults the shared simulate() twin directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx

from ..sim import SimResult
from .base import BaselinePlacer, accum_depths, var_dag


@dataclass
class FhelipePlacer(BaselinePlacer):
    NAME = "fhelipe_depthdp_v1"
    NAME_KEY = "fhelipe"

    def _depths(self) -> dict[str, float]:
        return accum_depths(self.g, self.seed_consumed, self.bootstrap_level)

    # ── the DP ──────────────────────────────────────────────────────────────────────
    def choose_sites(self, sim0: SimResult) -> set[str]:
        unit = max(1, self.g.level_unit)
        # usable composite levels after a dense refresh, honoring the 1-prime guard
        l0 = max(1, int((self.budget.L - 1) // unit))

        dprime = self._depths()
        D: dict[str, int] = {v: int(math.ceil(c / unit - 1e-9)) for v, c in dprime.items()}
        vdag = var_dag(self.g)
        maxD = max(D.values(), default=0)
        self.meta.update(l0=l0, max_depth=maxD, deviations=[
            "monotone depth (hint/deliberate/fold sites reset — runtime-owned)",
            "ct_count=1 per var",
            "ShaveLevels subset enumeration (greedy above 8 shortcuts/window)",
            "twin-based pruning post-pass"])

        if maxD <= l0:
            return set()               # whole block fits one window

        # crossing edges per boundary j: (v -> u) with D[v] <= j < D[u]
        cross_vars: dict[int, set[str]] = {j: set() for j in range(maxD)}
        deepest_reach: dict[str, int] = {}   # v -> max D[u] over consumers (for shortcuts)
        edges: list[tuple[str, str]] = []
        for (v, u) in vdag.edges():
            dv, du = D.get(v, 0), D.get(u, 0)
            if du <= dv:
                continue               # reset/lateral edge: crosses nothing
            edges.append((v, u))
            deepest_reach[v] = max(deepest_reach.get(v, dv), du)
            for j in range(dv, min(du, maxD)):
                cross_vars[j].add(v)

        # Only same-depth crossers form the (chokepoint-narrowed) boundary. Upstream
        # as published defers earlier-produced crossers (the residuals) to the shortcut
        # omission pass, but that rule emits an infeasible leveling on this IR; the
        # port folds them into the boundary instead, which is the configuration
        # upstream itself needs to execute (its shortcut omission disabled).
        defer_mode = False
        self.meta["defer_shortcuts"] = defer_mode
        frontier: dict[int, set[str]] = {}
        deferred: dict[int, set[str]] = {}
        for j in range(maxD):
            stratum_cross = {v for v in cross_vars[j] if D.get(v, 0) == j}
            frontier[j] = self._narrow_boundary(j, stratum_cross, D, vdag)
            if defer_mode:
                deferred[j] = cross_vars[j] - stratum_cross
            else:
                frontier[j] |= (cross_vars[j] - stratum_cross)
                deferred[j] = set()

        # dp over boundaries; dp[i] = cost to run to depth i, prev[i] = last boundary
        INF = float("inf")
        dp = [0.0] * (maxD + 1)
        prev: list[int | None] = [None] * (maxD + 1)
        omitted_at: dict[int, set[str]] = {}
        for i in range(l0 + 1, maxD + 1):
            best, bestj, bestom, bestpain = INF, None, set(), 1 << 30
            for j in range(max(0, i - l0), i):
                if dp[j] == INF:
                    continue
                omitted, extra, levels = self._window_shortcuts(
                    j, i, deferred.get(j, set()), deepest_reach,
                    D, vdag, l0, prev, dp)
                cost = dp[j] + len(frontier.get(j, set())) + extra
                pain = self._pain(levels, extra, l0)
                # upstream SelectMinimum: min cost, ties broken by least ShortcutPain
                if cost < best or (cost == best and pain < bestpain):
                    best, bestj, bestom, bestpain = cost, j, omitted, pain
            dp[i], prev[i] = best, bestj
            omitted_at[i] = bestom
        if dp[maxD] == INF:
            # no feasible boundary chain in-model; place every frontier (dense fallback,
            # the twin + P1 still judge it)
            self.meta["dp_infeasible_fallback"] = True
            return set().union(*frontier.values()) if frontier else set()

        boundaries: list[int] = []
        i = maxD
        while prev[i] is not None:
            boundaries.append(prev[i])
            i = prev[i]
        boundaries.reverse()
        self.meta["boundaries"] = boundaries

        sites: set[str] = set()
        bset = list(boundaries) + [maxD]
        for k, j in enumerate(boundaries):
            i_next = bset[k + 1]
            omitted, _, _lv = self._window_shortcuts(
                j, i_next, deferred.get(j, set()), deepest_reach,
                D, vdag, l0, prev, dp)
            # bootstrap the boundary frontier + every non-omitted terminal-window
            # deferred crosser (deep-reaching ones are charged in later windows)
            sites |= frontier.get(j, set())
            sites |= {v for v in deferred.get(j, set())
                      if v not in omitted and deepest_reach.get(v, i_next) <= i_next}
        sites = self._prune(sites)
        return sites

    # ── boundary narrowing (the paper's "simple min-cut" chokepoints) ───────────────
    def _narrow_boundary(self, j: int, crossing: set[str], D: dict[str, int],
                         vdag: nx.DiGraph) -> set[str]:
        if len(crossing) <= 1:
            return set(crossing)
        # crossing here = SAME-DEPTH crossers only; deferred crossers
        # are handled by the caller as shortcut candidates.
        stratum = set(crossing)
        if not stratum:
            return set()
        sub = vdag.subgraph(stratum).to_undirected()
        out: set[str] = set()
        for comp in nx.connected_components(sub):
            comp_cross = comp & crossing
            if len(comp_cross) == 1:
                out |= comp_cross
                continue
            choke = self._chokepoint(comp, vdag)
            if choke is not None and choke in comp_cross:
                out.add(choke)
            else:
                out |= comp_cross
        return out

    @staticmethod
    def _chokepoint(comp: set[str], vdag: nx.DiGraph) -> str | None:
        """A var through which all of the component's internal flow passes (the
        paper's FindChokepoint): the common dominator of all component sinks that
        lies nearest to them — computed on the induced DAG with a virtual source."""
        sinks = [v for v in comp
                 if not any(u in comp for u in vdag.successors(v))]
        if len(sinks) == 1:
            return sinks[0]
        if not sinks:
            return None
        sub = vdag.subgraph(comp).copy()
        src = "__src__"
        sub.add_node(src)
        for v in comp:
            if not any(u in comp for u in vdag.predecessors(v)):
                sub.add_edge(src, v)
        try:
            idom = nx.immediate_dominators(sub, src)
        except Exception:
            return None
        # dominator chain of the first sink, walked sink->source; the first node on
        # it that dominates EVERY sink is the chokepoint
        def chain(v):
            out = [v]
            while idom.get(v) is not None and idom[v] != v:
                v = idom[v]
                out.append(v)
            return out
        chains = [chain(s) for s in sinks]
        common = set(chains[0])
        for c in chains[1:]:
            common &= set(c)
        common.discard(src)
        if not common:
            return None
        for v in chains[0]:                 # nearest to the sinks first
            if v in common:
                return v
        return None

    # ── shortcut omission (the paper's ShaveLevels + largest acceptable subset) ─────
    @staticmethod
    def _shave(levels: list[int], bottom: int, top: int) -> list[int]:
        """Depress the window's level budget vector to account for carrying an
        un-refreshed value of age `top` used at window position `bottom`.
        No shaving when the carried value's use lies at/after the window end
        (the x<=0 guard)."""
        out = list(levels)
        x = len(out) - bottom
        if x <= 0:
            return out
        for idx in range(x, 0, -1):
            out[idx - 1] = min(out[idx - 1], bottom - top + idx)
        return out

    def _window_shortcuts(self, j: int, i: int, riders_in: set[str],
                          reach: dict[str, int], D: dict[str, int],
                          vdag: nx.DiGraph, l0: int,
                          prev: list[int | None], dp: list[float],
                          ) -> tuple[set[str], int, list[int]]:
        """Deferred (residual) crossers of boundary j whose last deep use lies
        within the window (j, i]: the LARGEST ShaveLevels-acceptable subset rides
        free (omitted); the rest are bootstrapped too (costed). Crossers reaching
        beyond i are charged in their terminal window only."""
        levels0 = list(range(1, l0 + 2))     # length l0+1, floor 1
        riders = sorted(v for v in riders_in if reach.get(v, i) <= i)
        if not riders:
            return set(), 0, levels0
        path: list[int] = [0]
        k = j
        while k is not None and k > 0:
            p = prev[k]
            if p is None:
                break
            path.append(p)
            k = p
        path = sorted(set(path))

        infos: list[tuple[str, int, int]] = []       # (var, top, bottom)
        for v in riders:
            dv = D.get(v, 0)
            b = max((p for p in path if p <= dv), default=0)
            top = min(dv - b, l0 + 1)                # age at window entry
            first_use = min((D.get(u, i) for u in vdag.successors(v)
                             if D.get(u, 0) > j), default=i)
            bottom = max(1, min(first_use, i) - j)
            infos.append((v, top, bottom))

        width_idx = l0 - (i - j)                     # deepest node's budget slot

        def acceptable(subset: tuple[tuple[str, int, int], ...]) -> list[int] | None:
            lv = levels0
            for (_v, top, bottom) in subset:
                lv = self._shave(lv, bottom, top)
                if lv[width_idx] < 1:
                    return None
            return lv

        best_subset: tuple[tuple[str, int, int], ...] = ()
        best_levels = levels0
        if len(infos) <= 8:                          # upstream enumerates all subsets
            from itertools import combinations
            found = False
            for size in range(len(infos), 0, -1):
                for subset in combinations(infos, size):
                    lv = acceptable(subset)
                    if lv is not None:
                        best_subset, best_levels, found = subset, lv, True
                        break
                if found:
                    break
        else:                                        # capped: greedy by span ascending
            cur: list[tuple[str, int, int]] = []
            lv = levels0
            for info in sorted(infos, key=lambda t: t[1]):
                trial = self._shave(lv, info[2], info[1])
                if trial[width_idx] >= 1:
                    cur.append(info)
                    lv = trial
            best_subset, best_levels = tuple(cur), lv
        omitted = {v for (v, _t, _b) in best_subset}
        extra = len(infos) - len(best_subset)
        return omitted, extra, best_levels

    @staticmethod
    def _pain(levels: list[int], omitted_infos: int, l0: int) -> int:
        """ShortcutPain analog: how much window budget the kept shortcuts consumed."""
        return sum(l0 - lv for lv in levels)

    # ── twin-based greedy pruning (the paper's post-pass, via the shared sim) ───────
    def _prune(self, sites: set[str]) -> set[str]:
        keep = set(v for v in sites if self._placeable_now(v))
        self.placed = set(keep)
        base = self._sim()
        if base.over_budget:
            self.placed = set()
            return keep                # infeasible anyway; rescue/refusal handles it
        pruned = 0
        for v in sorted(keep):
            self.placed = keep - {v}
            if not self._sim().over_budget:
                keep.discard(v)
                pruned += 1
        self.placed = set()
        self.meta["pruned"] = pruned
        return keep

    def _placeable_now(self, v: str) -> bool:
        from ..ir import is_kv_cache_read
        if v in self.g.inputs and not is_kv_cache_read(v):
            return False
        return v in self.g.producer_of or v in self.g.inputs
