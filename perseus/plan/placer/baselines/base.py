"""Shared seam for the baseline placers — the controlled-variable design.

A baseline placer reimplements another compiler's BOOTSTRAP PLACEMENT decision (WHERE
refreshes go) while sharing every other stage of plan/placer unchanged: graph ingest,
reactive-bootstrap erasure, the runtime-validated `simulate()` twin as the feasibility
oracle, `RefreshPlanner` pricing (HOW each site is refreshed), `emit.assemble()`, and
postconditions P1-P3. That makes cross-placer numbers a comparison of placement
algorithms, not of runtimes.

Baseline algorithms are magnitude-blind (their home compilers have no EvalMod wall), so a
site they pick can be `hopeless` (destructive refresh) or their plan can still be over
budget under the twin. Two policies, both honest:

  * raw (default):  PlanInfeasible, loudly — "infeasible as captured" IS the offline
                    result for that arm, not an error to hide.
  * rescue:         repair with counted, labeled bootstraps: hopeless sites move up the
                    binding-producer chain (`rescue_moved`), residual over-budget ops
                    get refreshes at their binding inputs (`rescue_added`). Every rescue
                    bootstrap counts against the baseline's total; the counts travel in
                    summary.placer_meta and the heuristic_config gains "+rescued".

Hints stay runtime-owned for every arm: the shared sim decides firing from whatever
levels the baseline's placement produces, and emit exports hint_fire so the plan binds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import networkx as nx

from ..ir import Graph, is_kv_cache_read
from ..place import Placer, PlanInfeasible
from ..sim import SimResult, step_bts_offsets

log = logging.getLogger(__name__)


def var_dag(g: Graph) -> nx.DiGraph:
    """The var-level DAG: edge producer-var -> consumer-var. Deliberate-bts outputs are
    ordinary nodes (they are fixed refresh points; baselines see them as depth resets via
    the sim, not as cuttable sites)."""
    d = nx.DiGraph()
    for v in g.inputs:
        d.add_node(v)
    for n in g.nodes:
        if not n.output:
            continue
        d.add_node(n.output)
        for u in n.cipher_inputs:
            if u != n.output:
                d.add_edge(u, n.output)
    return d


def accum_depths(g: Graph, seed_consumed, bootstrap_level: float,
                 hint_reset: bool | None = None) -> dict[str, float]:
    """Monotone accumulated consumed level per var, PRIME units — the baseline-internal
    depth model. Deliberate/fold refreshes reset to their captured landing, and (by
    default) hint sites reset to 0 — all three refresh kinds execute in every arm, so
    a baseline that could not see them would model whole steps (e.g. inv_sqrt_newton,
    which refreshes through hints mid-step) as deeper than the chain itself. The
    shared sim remains the sole feasibility oracle."""
    if hint_reset is None:
        hint_reset = BaselinePlacer.hint_aware()
    d: dict[str, float] = {}
    for v in g.inputs:
        d[v] = max(0.0, float(seed_consumed(v)))
    for n in g.nodes:
        if not n.output:
            continue
        if n.is_deliberate_bts or n.is_fold_bts:
            base = (n.output_level - bootstrap_level
                    if n.output_level is not None else 0.0)
            d[n.output] = max(0.0, base)
            continue
        if hint_reset and n.hint_level is not None:
            d[n.output] = 0.0
            continue
        ins = [d.get(v, 0.0) for v in n.cipher_inputs]
        d[n.output] = (max(ins) if ins else 0.0) + n.cost
    return d


@dataclass
class BaselinePlacer(Placer):
    """Base class: subclasses implement `choose_sites(sim0)` and set NAME/meta."""

    NAME = "baseline"

    rescue: bool = False
    #: cap on a placed refresh's effective input depth (absolute prime level); None = the
    #: baseline as published. The dense Fhelipe arm of the paper is planned with 48.
    depth_cap: float | None = None
    meta: dict = field(default_factory=dict)

    MAX_RESCUE_HOPS = 10
    MAX_REPAIR_ITERS = 20
    MAX_FIXPOINT_ITERS = 8

    @staticmethod
    def hint_aware() -> bool:
        """Baselines are ported hint-aware: hint-conditional resets in the baseline's
        world-model plus fixpoint re-invocation as placement shifts hint firing.
        Hints execute at runtime in every arm; only the model is aware of them."""
        return True

    # ── what a baseline must provide ────────────────────────────────────────────────────
    def choose_sites(self, sim0: SimResult) -> set[str]:
        """Return the target vars to refresh, given the no-placement simulation
        (sim0.consumed is the unrefreshed level trajectory in PRIME units; hint and
        deliberate refresh landings are already applied — they happen in every arm)."""
        raise NotImplementedError

    # ── shared plumbing ──────────────────────────────────────────────────────────────
    @property
    def heuristic_config(self) -> str:
        return self.NAME + ("+rescued" if self.rescue and (
            self.meta.get("rescue_added") or self.meta.get("rescue_moved")) else "")

    def _placeable(self, v: str, consumed: dict[str, float]) -> bool:
        if v in self.placed:
            return False
        if consumed.get(v, 0.0) == 0.0:
            return False               # already at a refreshed level: cutting is noise
        if v in self.g.inputs and not is_kv_cache_read(v):
            return False
        if self._forbidden(v):
            return False
        p = self.g.producer_of.get(v)
        if p is not None and (p.is_fold_bts or p.is_deliberate_bts):
            # already a refresh point; the runtime does NOT fire planted bootstraps
            # after fold/deliberate nodes (measured: fhelipe block_8 v_599 placement
            # silently skipped -> [plan_level_error] +2), so placing here is a lie.
            return False
        return not self.refresh.spec(v).hopeless

    def _cross_step_vars(self) -> set[str]:
        """Vars consumed outside their producing step — the between-module vocabulary
        every upstream placer actually uses. Cached."""
        cached = getattr(self, "_xstep_cache", None)
        if cached is not None:
            return cached
        step_of = {n.output: n.step for n in self.g.nodes if n.output}
        out: set[str] = set()
        for n in self.g.nodes:
            for v in n.cipher_inputs:
                if v in step_of and step_of[v] != n.step:
                    out.add(v)
        self._xstep_cache = out
        return out

    def _rescue_move(self, sim) -> str | None:
        """One atomic repair MOVE for a starved over-budget op: try (remove a placed
        refresh on the offender's ancestry) x (add the closest placeable site to the
        offender), accept the first combination that strictly reduces the number of
        over-budget ops. Returns the removed site or None. Runs only under rescue."""
        if not self.rescue:
            return None
        for i, _excess in sim.over_budget:
            n = self.g.nodes[i]
            chain: list[str] = []
            cur_vs = list(n.cipher_inputs)
            seen_r: set[str] = set()
            for _hop in range(self.MAX_RESCUE_HOPS * 3):
                nxt_vs: list[str] = []
                for u in cur_vs:
                    if u in seen_r:
                        continue
                    seen_r.add(u)
                    if u in self.placed:
                        chain.append(u)
                    p = self.g.producer_of.get(u)
                    if p is not None:
                        nxt_vs.extend(p.cipher_inputs)
                if not nxt_vs:
                    break
                cur_vs = nxt_vs
            base_ob = len(sim.over_budget)
            for u in chain:
                saved = set(self.placed)
                self.placed.discard(u)
                trial = self._sim()
                # removal alone helps?
                if len(trial.over_budget) < base_ob:
                    self.meta["rescue_moved"] = self.meta.get("rescue_moved", 0) + 1
                    self._origin.setdefault(u, "chosen")
                    return u
                # removal + closest re-add (the actual MOVE)
                for v in n.cipher_inputs:
                    site = self._binding_ancestor(v, trial.consumed)
                    if site is not None and site not in self.placed:
                        self.placed.add(site)
                        trial2 = self._sim()
                        if len(trial2.over_budget) < base_ob:
                            self._origin.setdefault(site, "rescue")
                            self.meta["rescue_moved"] = \
                                self.meta.get("rescue_moved", 0) + 1
                            return u
                        self.placed.discard(site)
                self.placed = saved
        return None

    def _depth_cap(self) -> float | None:
        """Cap on a placed refresh's effective input depth (abs prime level): the
        constraint the min-cut enforces via REFRESH_ENV_CAP_ABS and that the baseline
        placers bypass, since they never call _capacity. None = the baseline as published."""
        return self.depth_cap

    def _site_input_depth(self, v: str) -> float:
        """Effective ABS input depth of the refresh at placed var v: simulate with v
        UNPLACED (everything else pinned) and read v's consumed there — the exact
        depth the runtime's [planted_bts] in= reports. Post-placement sim.consumed[v]
        is the POST-refresh trajectory and reads ~landing (measured: v_75 -10 vs
        runtime in=50), which is why P3c never fires on placed sites."""
        saved = self.placed
        self.placed = saved - {v}
        sim = self._sim()
        self.placed = saved
        eff = sim.consumed.get(v, 0.0) + (
            self.g.level_unit if sim.deg.get(v, 1) == 2 else 0)
        return self.bootstrap_level + eff

    def _slide_deep_sites(self) -> int:
        """Replace every placed refresh whose input depth exceeds `_depth_cap()`
        with its nearest binding ancestor that fits under the cap (same relocation
        vocabulary as _binding_ancestor: binding-producer chain, deepest cipher input
        first). Returns the number of sites moved; the caller re-repairs afterwards —
        a shallower refresh shortens downstream runway, and the standard repair loop
        owns that fallout."""
        cap = self._depth_cap()
        if cap is None:
            return 0
        moved = 0
        for v in sorted(self.placed):
            if self._site_input_depth(v) <= cap + 1e-9:
                continue
            # probe world: v unplaced, so ancestors read the unrefreshed-at-v trajectory
            saved = self.placed
            self.placed = saved - {v}
            sim_wo = self._sim()
            cons = sim_wo.consumed
            target: str | None = None
            cur = v
            for _ in range(self.MAX_RESCUE_HOPS * 3):
                p = self.g.producer_of.get(cur)
                if p is None or not p.cipher_inputs:
                    break
                cur = max(p.cipher_inputs, key=lambda u: cons.get(u, 0.0))
                eff_u = cons.get(cur, 0.0) + (
                    self.g.level_unit if sim_wo.deg.get(cur, 1) == 2 else 0)
                if (self.bootstrap_level + eff_u <= cap + 1e-9
                        and cur not in saved and self._placeable(cur, cons)):
                    target = cur
                    break
            self.placed = saved
            if target is None:
                # nothing reachable fits: leave the site (better a deep refresh than a
                # starved path); the count is reported so the arm is honest about it
                self.meta["depth_cap_unmovable"] = \
                    self.meta.get("depth_cap_unmovable", 0) + 1
                continue
            self.placed = (self.placed - {v}) | {target}
            self._origin.pop(v, None)
            self._origin[target] = "moved"
            moved += 1
        if moved:
            self.meta["depth_cap_moved"] = self.meta.get("depth_cap_moved", 0) + moved
        return moved

    def _binding_ancestor(self, v: str, consumed: dict[str, float]) -> str | None:
        """Nearest placeable var walking up the binding-producer chain from v
        (inclusive), preferring CROSS-STEP vars (the baselines' own between-module site
        vocabulary) so repairs stay in-model; falls back to a bounded BFS over ALL
        cipher-input ancestors. Returns None if nothing in reach is placeable."""
        xstep = self._cross_step_vars()
        first_any: str | None = None
        cur = v
        for _ in range(self.MAX_RESCUE_HOPS + 1):
            if self._placeable(cur, consumed):
                if cur in xstep:
                    return cur
                if first_any is None:
                    first_any = cur
            p = self.g.producer_of.get(cur)
            if p is None or not p.cipher_inputs:
                break
            cur = max(p.cipher_inputs, key=lambda u: consumed.get(u, 0.0))
        if first_any is not None:
            return first_any
        # BFS fallback across every ancestor branch, bounded
        seen: set[str] = {v}
        frontier = [v]
        best: str | None = None
        for _ in range(self.MAX_RESCUE_HOPS * 3):
            nxt: list[str] = []
            for x in frontier:
                p = self.g.producer_of.get(x)
                if p is None:
                    continue
                for u in p.cipher_inputs:
                    if u in seen:
                        continue
                    seen.add(u)
                    nxt.append(u)
                    if self._placeable(u, consumed) and (
                            best is None
                            or consumed.get(u, 0.0) > consumed.get(best, 0.0)):
                        best = u
            if best is not None or not nxt:
                break
            frontier = nxt
        return best

    def run(self) -> SimResult:
        self._step_offsets = step_bts_offsets(self.g, self.bootstrap_level)
        self.refresh.step_bts_offset = self._step_offsets

        self._pending_moved: set[str] = set()
        sim0 = self._sim()             # placed is empty: the unrefreshed trajectory
        sites = set(self.choose_sites(sim0))
        self.meta.setdefault("algorithm", self.NAME)
        self.meta["num_sites_chosen"] = len(sites)
        self.meta["sparse_routing"] = bool(getattr(self.refresh, "sparse_precomps", ()))

        # sanitize: forbidden / fresh / free-noise sites are dropped for every arm
        # (identical to our _capacity's infinite-capacity rules), hopeless sites are
        # policy (raw refuses, rescue relocates).
        dropped, hopeless, kept = [], [], []
        for v in sorted(sites):
            if v not in self.g.producer_of and not (
                    v in self.g.inputs and is_kv_cache_read(v)):
                dropped.append(v)
            elif v in self.g.producer_of and self.refresh.spec(v).hopeless \
                    and sim0.consumed.get(v, 0.0) > 0.0 and not self._forbidden(v):
                hopeless.append(v)
            elif not self._placeable(v, sim0.consumed):
                dropped.append(v)
            else:
                kept.append(v)
        self.meta["num_sites_dropped"] = len(dropped)

        moved = 0
        if hopeless:
            if not self.rescue:
                raise PlanInfeasible(
                    f"{self.NAME}: {len(hopeless)} chosen site(s) have destructive "
                    "refreshes (magnitude-blind placement); raw mode refuses — "
                    "run with baseline-rescue for a repaired artifact",
                    [self.refresh.spec(v) for v in hopeless[:12]])
            dropped_hopeless = 0
            for v in hopeless:
                sub = self._binding_ancestor(v, sim0.consumed)
                if sub is not None:
                    if sub not in kept:
                        kept.append(sub)
                        self._pending_moved = getattr(self, "_pending_moved", set())
                        self._pending_moved.add(sub)
                        moved += 1
                else:
                    # nothing placeable upstream (e.g. a magnitude-less KV entry the
                    # baseline picked): drop the site; the repair loop below handles any
                    # violation this leaves, and the drop is counted.
                    dropped_hopeless += 1
            self.meta["rescue_dropped_hopeless"] = dropped_hopeless
        self.meta["rescue_moved"] = moved

        # Per-site provenance: the addition counters below tally
        # per-iteration additions, but the zero-gain prune and re-solves remove many
        # of them, so counters do NOT describe the FINAL plan. This map does: every
        # surviving site carries who put it there — the baseline's own first pass
        # ("chosen"), its own re-solve ("fixpoint"), or our magnitude machinery
        # ("moved" = relocated off a destructive site, "rescue" = abstraction-gap
        # repair). summary.placer_meta.provenance counts SURVIVORS only.
        self._origin: dict[str, str] = {v: "chosen" for v in kept}
        for v in getattr(self, "_pending_moved", set()):
            self._origin[v] = "moved"          # our magnitude relocation, not theirs
        self.placed = set(kept)
        sim = self._sim()

        # ── FIXPOINT: the baseline's own algorithm closes its residuals ─────────────────
        # Upstream placers are feasibility-total for the world they simulate; residual
        # over-budget here means their internal model diverged from the twin (hint
        # conditionality above all). The honest closure is the same discipline our
        # min-cut uses: re-run the BASELINE on the updated twin trajectory with prior
        # placements pinned, until fixpoint. The naive repair below then only covers
        # true abstraction gaps.
        fixpoint_added = 0
        fixpoint_iters = 0
        max_fixpoint = self.MAX_FIXPOINT_ITERS if self.hint_aware() else 0
        self.meta["hint_aware"] = self.hint_aware()
        for _ in range(max_fixpoint):
            if not sim.over_budget:
                break
            fixpoint_iters += 1
            new_sites = set(self.choose_sites(sim))
            fresh = []
            for v in sorted(new_sites - self.placed):
                if v in self.g.producer_of and self._placeable(v, sim.consumed):
                    fresh.append(v)
                elif self.rescue and v in self.g.producer_of \
                        and self.refresh.spec(v).hopeless:
                    sub = self._binding_ancestor(v, sim.consumed)
                    if sub is not None and sub not in self.placed:
                        fresh.append(sub)
            if not fresh:
                break                        # the baseline has nothing more to say
            self.placed |= set(fresh)
            for v in fresh:
                self._origin.setdefault(v, "fixpoint")
            fixpoint_added += len(fresh)
            sim = self._sim()
        self.meta["fixpoint_iters"] = fixpoint_iters
        self.meta["fixpoint_added"] = fixpoint_added

        # Depth-cap slide (off unless `_depth_cap()` is set): move too-deep refresh INPUTS
        # under the envelope BEFORE the repair loop, so repair owns any runway the
        # slide costs downstream.
        if self._slide_deep_sites():
            sim = self._sim()

        # residual over-budget ops: raw refuses; rescue repairs boundedly (this is
        # now the measure of the baseline's ABSTRACTION GAP, not of model divergence).
        added = 0
        if sim.over_budget and self.rescue:
            for _ in range(self.MAX_REPAIR_ITERS):
                if not sim.over_budget:
                    break
                progressed = False
                for i, _excess in sim.over_budget:
                    n = self.g.nodes[i]
                    cands = [v for v in n.cipher_inputs] + ([n.output] if n.output else [])
                    cands.sort(key=lambda u: -sim.consumed.get(u, 0.0))
                    for v in cands:
                        site = self._binding_ancestor(v, sim.consumed)
                        if site is not None and site not in self.placed:
                            self.placed.add(site)
                            self._origin.setdefault(site, "rescue")
                            added += 1
                            progressed = True
                            break
                if not progressed:
                    progressed = self._rescue_move(sim) is not None
                    if progressed:
                        sim = self._sim()
                if not progressed:
                    break
                sim = self._sim()
            # Final MOVE phase:
            # the additions loop can exhaust its iteration budget while "progressing"
            # on sites that never fix a starved op. A ceiling-type level_hint whose
            # input arrives over-consumed cannot be fixed by ADDING refreshes when the
            # existing ones sit too far up its ancestry (bts-out 36 < the hint's 38+
            # requirement after the 08-20 +2 shift) — the repair is a MOVE: drop the
            # far refresh, add the binding ancestor nearest the offender, atomically.
            for _ in range(self.MAX_REPAIR_ITERS):
                if not sim.over_budget:
                    break
                if self._rescue_move(sim) is None:
                    break
                sim = self._sim()
        self.meta["rescue_added"] = added

        # Re-check the depth cap: repair/move phases may have ADDED deep sites (they
        # pick binding ancestors by max consumed, i.e. deepest-first). One more slide,
        # then one bounded re-repair for its fallout.
        if self._slide_deep_sites():
            sim = self._sim()
            if sim.over_budget and self.rescue:
                for _ in range(self.MAX_REPAIR_ITERS):
                    if not sim.over_budget:
                        break
                    if self._rescue_move(sim) is None:
                        break
                    sim = self._sim()

        # zero-gain pruning (uniform, all baselines): a refresh whose input is already at
        # or below its own landing buys nothing — boundary-style placers (Fhelipe
        # bootstraps EVERY var crossing a boundary) emit chains of adjacent refreshes
        # where the runtime then bootstraps an already-fresh ct, a path that segfaults
        # the sparse runtime (measured, fhelipe sparse decode block_0). Dropping them
        # cannot raise any downstream level (gain <= 0), so a single pass suffices.
        zero_gain = 0
        for v in sorted(self.placed):
            keep = self.placed - {v}
            saved, self.placed = self.placed, keep
            unref = self._sim().consumed.get(v, 0.0)
            self.placed = saved
            if unref <= self.refresh.spec(v).out_consumed + 1e-9:
                self.placed = self.placed - {v}
                zero_gain += 1
        if zero_gain:
            sim = self._sim()
        self.meta["zero_gain_dropped"] = zero_gain

        # SURVIVOR provenance — what the FINAL plan is actually made of. Unlike the
        # addition counters above (which tally per-iteration adds that the prune and
        # re-solves later remove), these sum to exactly len(self.placed).
        prov: dict[str, int] = {}
        for v in self.placed:
            prov[self._origin.get(v, "chosen")] = \
                prov.get(self._origin.get(v, "chosen"), 0) + 1
        own = prov.get("chosen", 0) + prov.get("fixpoint", 0)
        ours = prov.get("moved", 0) + prov.get("rescue", 0)
        self.meta["provenance"] = {
            # sorted: `placed` is a set, so its iteration order would otherwise reach the
            # shipped JSON and a regeneration would not be byte-identical
            **dict(sorted(prov.items())),
            "own_total": own,                       # the baseline's own decisions
            "assisted_total": ours,                 # our magnitude machinery
            "own_frac": round(own / max(1, own + ours), 4),
        }
        self.meta["site_origin"] = {v: self._origin.get(v, "chosen")
                                    for v in sorted(self.placed)}

        if sim.over_budget:
            offenders = [self.g.nodes[i] for i, _ in sim.over_budget[:6]]
            raise PlanInfeasible(
                f"{self.NAME}: plan leaves {len(sim.over_budget)} op(s) over budget"
                + ("" if self.rescue else " (raw mode: no repair attempted)")
                + "; first offenders: "
                + ", ".join(f"{n.op}@{n.step[-40:]}" for n in offenders))

        if self.verbose:
            log.info(f"[{self.NAME}] {len(self.placed)} refresh(es) placed "
                  f"(chosen={self.meta['num_sites_chosen']} dropped={len(dropped)} "
                  f"moved={moved} added={added})")
        return sim
