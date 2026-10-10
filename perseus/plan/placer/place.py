from __future__ import annotations

import logging
from dataclasses import dataclass, field

import networkx as nx

from .ir import Graph, Node, is_kv_cache_read
from .refresh import RefreshPlanner, RefreshSpec
from .sim import Budget, SimResult, simulate

log = logging.getLogger(__name__)

QUALITY_LAMBDA = 1.0 / 4096.0
#: Deepest absolute level (in PRIMES) a refresh may start at. The runtime's own guard is
#: chain-relative -- one CKKS level above the reactive ceiling, so 46+2=48 on the 32-bit
#: composite chain and 24+1=25 on the 64-bit one (fideslib_wrapper.h, BTS_MAX_INPUT_LEVEL).
#: The planner cannot see that ceiling: no plan recipe sources scripts/local_env.sh, so
#: AUTO_BTS_LEVEL is not in its environment. Hence a knob with the 32-bit value as the
#: default, which is what every shipped recipe was planned under.
REFRESH_ENV_CAP_ABS = 48.0


def refresh_env_cap() -> float:
    """REFRESH_ENV_CAP_ABS, or PLAN_REFRESH_ENV_CAP when the recipe pins the chain's own."""
    import os as _os
    e = _os.environ.get("PLAN_REFRESH_ENV_CAP")
    if e and e.strip():
        try:
            return float(e)
        except ValueError:
            pass
    return REFRESH_ENV_CAP_ABS


class PlanInfeasible(RuntimeError):
    """The block cannot be planned. Carries the refused sites so the error is actionable."""

    def __init__(self, msg: str, refusals: list[RefreshSpec] | None = None):
        self.refusals = refusals or []
        detail = ""
        if self.refusals:
            lines = [f"    {s.var}: {s.reason}" for s in self.refusals[:12]]
            more = len(self.refusals) - len(lines)
            detail = ("\n  refused site(s):\n" + "\n".join(lines)
                      + (f"\n    ... and {more} more" if more > 0 else ""))
        super().__init__(msg + detail)


@dataclass
class EraseReport:
    n_erased: int
    n_deliberate_kept: int


def erase_reactive_bootstraps(g: Graph) -> tuple[Graph, EraseReport]:
    """Drop every auto_bootstrap; rewire consumers to its input. Unconditional."""
    remap: dict[str, str] = {}
    kept_raw: list[Node] = []
    for n in g.nodes:
        if n.is_auto_bts and n.cipher_inputs and n.output:
            remap[n.output] = n.cipher_inputs[0]
            continue
        kept_raw.append(n)

    def resolve(v: str) -> str:
        seen: set[str] = set()
        while v in remap and v not in seen:
            seen.add(v)
            v = remap[v]
        return v

    from .ir import _node_to_raw, is_literal_input, is_plaintext_name
    out_level_of = {n.output: n.output_level for n in kept_raw if n.output}
    raws = []
    for n in kept_raw:
        d = _node_to_raw(n)
        new_in, new_lv = [], []
        for x, lv in zip(d["inputs"], d["input_levels"]):
            if not is_literal_input(x) and not is_plaintext_name(x):
                r = resolve(x)
                new_in.append(r)
                rl = out_level_of.get(r)
                new_lv.append(rl if (r != x and rl is not None) else lv)
            else:
                new_in.append(x)
                new_lv.append(lv)
        d["inputs"], d["input_levels"] = new_in, new_lv
        raws.append(d)
    g2 = Graph.from_nodes(raws, level_unit=g.level_unit)
    n_delib = sum(1 for n in g2.nodes if n.is_deliberate_bts)
    return g2, EraseReport(n_erased=len(remap), n_deliberate_kept=n_delib)


@dataclass
class Placer:
    g: Graph
    refresh: RefreshPlanner
    budget: Budget
    bootstrap_level: float
    seed_consumed: object
    seed_deg: object
    forbid_steps: tuple[str, ...] = ()
    err_target: float = 1e-2
    miss_penalty: float = 4.0
    quality_weight: float = 0.0    # the "pre-score" weight; 0 = count-only (default)
    # Depth-aware pricing. A site's capacity is scaled by how many levels its refresh
    # RESTORES: restored = effective input depth - the spec's landing, so a sparse route
    # landing richer than dense restores more, and a prescaled site less. Three forms:
    #   ratio   budget / restored, blended    cap *= (1 - w) + w * ratio
    #   linear  the levels NOT regained, W = M - restored, normalised by M, same blend
    #   ab      W = a * min_runway / (M - R) + b * (M - L) / M, used directly
    # 0 disables it and the cut is count-only, which is what every shipped plan uses.
    # Above 0 count no longer strictly dominates: two shallow refreshes can outprice one
    # deep one, which is the point of the knob.
    depth_weight: float = 0.0
    depth_form: str = "ratio"       # "ratio" | "linear" | "ab"
    depth_a: float = 1.0            # ab form: landing-yield coefficient
    depth_b: float = 1.0            # ab form: early-refresh coefficient
    # Input-level pricing. A bootstrap's own error grows with the depth of its input, and
    # the accuracy table does not price that. Linear ramp on the effective absolute input
    # level: 0 at bootstrap_level, `level_weight` at the refresh envelope. Pulls against
    # depth_weight, which rewards deep high-yield refreshes.
    level_weight: float = 0.0
    # Measured per-route bootstrap latency in milliseconds, keyed by sparse slot count with
    # 0 for a dense refresh (paper Table 7). With depth_form="ms" the cut minimises the
    # bootstrap milliseconds a plan will actually spend instead of the number of refreshes,
    # which are not the same thing: a dense refresh costs nearly twice a 1-slot one.
    bts_ms: dict[int, float] = field(default_factory=lambda: {0: 28.77, 512: 18.59, 1: 16.10})
    # How much of a sparse route's extra restored runway to credit back. A refresh landing
    # richer buys depth later refreshes then do not have to buy, which its own latency does
    # not show. cost = ms / (restored / L) ** ms_discount: 0 prices pure milliseconds, 1
    # prices milliseconds per unit of runway regained.
    ms_discount: float = 1.0
    # Refresh-input envelope hardness. MUST be True exactly when the cut is the SOLE
    # refresher (hints dissolved) — see _capacity. The planner sets it from
    # cfg.dissolve_hints; PLAN_HARD_ENV_CAP overrides either way.
    hard_env_cap: bool = False
    verbose: bool = True

    #: Set when `run` had to allow a refresh past REFRESH_ENV_CAP_ABS because no plan that
    #: respects the envelope exists. The deep site is then forced, not chosen, so the
    #: planner's P3c check reports it instead of refusing.
    env_cap_relaxed: bool = False
    #: Resolved per `run`; `_capacity` reads this, never `hard_env_cap` directly.
    _env_cap_hard: bool = False
    #: Hints vetoed because the sim predicts them firing past the envelope. A hint never goes
    #: through `_capacity`, so the envelope policy cannot price it; forcing it off hands the
    #: branch to the cut, which can.
    vetoed_hints: set[str] = field(default_factory=set)
    #: Cut-chosen sites whose refresh input drifted past the envelope after they were priced (a later cut
    #: disarmed a hint upstream) and that PLAN_DRIFT_RECUT could not re-cover: kept, reported by P3c as forced.
    drift_forced: set[str] = field(default_factory=set)
    _drift_rounds: int = 0

    placed: set[str] = field(default_factory=set)
    #: effective (pre-refresh) consumed depth of each placed site, as _capacity saw it
    #: when the cut chose it — what the pricing knobs above are judged on.
    placed_in_eff: dict[str, float] = field(default_factory=dict)
    realized: set[str] = field(default_factory=set)
    hint_force: dict[str, bool] | None = None
    deliberate_clamp0: bool = False
    _step_offsets: dict[str, float] | None = None
    ever_eligible: set[str] = field(default_factory=set)
    ever_pressured: set[str] = field(default_factory=set)

    def _forbidden(self, v: str) -> bool:
        p = self.g.producer_of.get(v)
        step = p.step if p else ""
        return any(k in step for k in self.forbid_steps if k)

    @staticmethod
    def _wire_operand(v: str, eff: dict[str, float], binding: float, ceiling: float,
                      deep_operands: bool) -> bool:
        """Which operands of a pressured op get a flow-network edge.

        The binding operand (the one at the max effective level) is always wired; with
        `deep_operands` an operand that by itself sits past the op's ceiling is wired
        too, so a deep non-binding branch is reachable in one cut pass (the P4 class).
        The extra edge is a hint to the cut, not a proven obligation: operand levels are
        coupled, and routed through an uncuttable var it can make `nx.minimum_cut` read
        the network as unbounded — hence `_one_cut`'s binding-only fallback.
        """
        e = eff.get(v, 0.0)
        return e >= binding or (deep_operands and e > ceiling)

    def _refreshed_map(self) -> dict[str, tuple[float, int]]:
        out: dict[str, tuple[float, int]] = {}
        for v in self.placed:
            s = self.refresh.spec(v)
            out[v] = (s.out_consumed, s.out_deg)

        for n in self.g.nodes:
            if n.hint_level is not None and n.output and n.output not in out:
                s = self.refresh.spec(n.output)
                if not s.hopeless:
                    out[n.output] = (s.out_consumed, s.out_deg)
                else:
                    out[n.output] = (0.0, 2)
        return out

    def _sim(self) -> SimResult:
        refreshed = {}
        sparse_refreshed = set()
        for v in self.placed:
            s = self.refresh.spec(v)
            refreshed[v] = (s.out_consumed, s.out_deg)
            if s.route:
                sparse_refreshed.add(v)
        for n in self.g.nodes:
            if n.hint_level is not None and n.output:
                s = self.refresh.spec(n.output)
                refreshed.setdefault(n.output, (s.out_consumed, s.out_deg)
                                     if not s.hopeless else (0.0, 2))
                if s.route and not s.hopeless:
                    sparse_refreshed.add(n.output)
        return simulate(
            self.g, bootstrap_level=self.bootstrap_level, budget=self.budget,
            seed_consumed=self.seed_consumed, seed_deg=self.seed_deg,
            refreshed=refreshed, step_offsets=self._step_offsets,
            placed=set(self.placed), sparse_refreshed=sparse_refreshed,
            realized=set(self.realized) if self.realized else None,
            hint_force=self.hint_force,
            deliberate_clamp0=self.deliberate_clamp0)

    def _min_runway(self, M: float) -> float:
        """min over routes of (M - R): the runway the worst landing leaves."""
        lands = [float(self.bootstrap_level)] + [float(lv) for lv in self.refresh.sparse_out_levels.values()]
        return M - max(lands)

    def _capacity(self, v: str, sim: SimResult) -> float:
        self._last_sim = sim
        if v in self.placed:
            return float("inf")
        if (sim.consumed.get(v, 0.0)
                + (self.g.level_unit if sim.deg.get(v, 1) == 2 else 0)) == 0.0:
            return float("inf")
        if v in self.g.inputs and not is_kv_cache_read(v):
            return float("inf")
        if self._forbidden(v):
            return float("inf")
        spec = self.refresh.spec(v)
        if spec.hopeless:
            return float("inf")
        eff_v = (self._last_sim.consumed.get(v, 0.0) + (self.g.level_unit if self._last_sim.deg.get(v, 1) == 2 else 0)) if self._last_sim is not None else 0.0
        if self.bootstrap_level + eff_v > refresh_env_cap():
            # Past the refresh envelope. Payable (1e4, large but finite) by default: an
            # inf here can starve the cut entirely when whole paths sit past the envelope,
            # and with hints live they absorb the deep values so the penalty is never
            # exercised. Hard (inf) when the cut is the sole refresher (`hard_env_cap`,
            # set by the planner from dissolve_hints): a bootstrap started past the
            # envelope silently returns garbage. PLAN_HARD_ENV_CAP=0/1 overrides either way.
            return float("inf") if self._env_cap_hard else 1.0e4
        cap = 1.0 + self.miss_penalty * spec.overshoot(self.err_target)
        if self.depth_weight > 0.0:
            restored = eff_v - float(spec.out_consumed)
            unit = float(self.g.level_unit)
            M = float(self.bootstrap_level) + float(self.budget.L)
            if self.depth_form == "ms":
                route = spec.route or 0
                ms = self.bts_ms.get(route) or self.bts_ms.get(0, 1.0)
                yield_ = max(restored, unit) / max(float(self.budget.L), unit)
                cap *= ms / max(yield_, 1e-9) ** self.ms_discount
            elif self.depth_form == "ab":
                R = float(self.bootstrap_level) + float(spec.out_consumed)
                L = float(self.bootstrap_level) + eff_v
                cap *= (self.depth_a * self._min_runway(M) / max(M - R, unit)
                        + self.depth_b * max(M - L, 0.0) / max(M, 1.0))
            else:
                ratio = (max(M - restored, 0.0) / max(M, 1.0) if self.depth_form == "linear"
                         else float(self.budget.L) / max(restored, unit))
                cap *= (1.0 - self.depth_weight) + self.depth_weight * ratio
        if self.level_weight > 0.0:
            span = max(REFRESH_ENV_CAP_ABS - self.bootstrap_level, 1.0)
            cap *= 1.0 + self.level_weight * min(1.0, max(0.0, eff_v / span))
        if self.quality_weight > 0.0:
            q = min(1.0, max(0.0, spec.rel_err / max(self.err_target, 1e-300)))
            cap += QUALITY_LAMBDA * self.quality_weight * q
        return cap

    def run(self) -> SimResult:
        """Place the refreshes, keeping every one inside the refresh envelope if any plan can.

        A bootstrap started past REFRESH_ENV_CAP_ABS returns garbage silently, so the envelope
        is what the cut should respect. Making it an outright refusal is too blunt: an `inf`
        there starves the cut on graphs whose deep paths have no shallower cut point, and the
        plan dies with nothing placed. So: price it as unreachable first, and only if THAT
        cannot be planned fall back to the payable cap, which is the old behaviour. A deep
        refresh then means there was nowhere else to put one, and `env_cap_relaxed` says so.

        PLAN_HARD_ENV_CAP pins one pass: 1 = envelope absolute (no fallback), 0 = payable
        throughout (what the shipped 32-bit recipe asked for before this was two-pass).
        """
        import os as _os
        _ov = _os.environ.get("PLAN_HARD_ENV_CAP")
        if _ov in ("0", "1"):
            self._env_cap_hard = (_ov == "1")
            return self._run_with_hint_veto()
        if self.hard_env_cap:
            # the cut is the sole refresher (hints dissolved): there is no hint to absorb a
            # deep value, so the envelope is absolute and a fallback would only hide it
            self._env_cap_hard = True
            return self._run_with_hint_veto()

        saved = (set(self.placed), set(self.realized),
                 set(self.ever_eligible), set(self.ever_pressured))
        self._env_cap_hard = True
        try:
            return self._run_with_hint_veto()
        except PlanInfeasible:
            self.placed, self.realized, self.ever_eligible, self.ever_pressured = (
                set(saved[0]), set(saved[1]), set(saved[2]), set(saved[3]))
            self._env_cap_hard = False
            self.env_cap_relaxed = True
            res = self._run_with_hint_veto()  # its own refusal is the honest one: nowhere to place
            if self.verbose:
                log.info("[plan] refresh envelope RELAXED: no placement keeps every refresh at "
                         f"or below absolute level {refresh_env_cap():g}; the deep sites below "
                         "are forced, not chosen")
            return res

    def _deep_hints(self, sim: SimResult) -> list[str]:
        """Hints the sim has firing on an input already past the envelope.

        The hint's own output is post-refresh and always shallow; what matters is the level of
        the ciphertext entering it, which is what the runtime's bts_depth_error guard sees.
        """
        out = []
        unit = self.g.level_unit
        for n in self.g.nodes:
            if n.hint_level is None or not sim.hint_fired.get(n.idx, False) or not n.output:
                continue
            ins = n.cipher_inputs
            if not ins:
                continue
            eff = sim.consumed.get(ins[0], 0.0) + (unit if sim.deg.get(ins[0], 1) == 2 else 0)
            if self.bootstrap_level + eff > refresh_env_cap():
                out.append(n.output)
        return out

    def _run_with_hint_veto(self, passes: int = 4) -> SimResult:
        """`_run_once`, then force off any hint predicted to fire past the envelope and replan.

        Vetoing makes the value keep accumulating, so the op below it goes over budget and the
        cut has to cover the branch -- under `_capacity`, which respects the envelope. If the
        veto makes the graph unplannable the veto is taken back: a plan with a deep hint beats
        no plan, and the runtime guard still covers it.

        OFF unless PLAN_HINT_ENV_VETO=1. It moves refreshes, so it would rewrite every shipped
        plan; the arms that want the envelope guarantee to cover hints ask for it.
        """
        res = self._run_once()
        import os as _os
        if _os.environ.get("PLAN_HINT_ENV_VETO") != "1":
            return res
        for _ in range(passes):
            deep = [v for v in self._deep_hints(res) if v not in self.vetoed_hints]
            if not deep:
                return res
            saved_force = dict(self.hint_force) if self.hint_force else None
            saved_state = (set(self.placed), set(self.realized),
                           set(self.ever_eligible), set(self.ever_pressured))
            self.hint_force = dict(self.hint_force or {})
            for v in deep:
                self.hint_force[v] = False
            self.vetoed_hints.update(deep)
            self.placed, self.realized = set(), set()
            try:
                res = self._run_once()
            except PlanInfeasible:
                self.hint_force = saved_force
                self.vetoed_hints.difference_update(deep)
                (self.placed, self.realized,
                 self.ever_eligible, self.ever_pressured) = (set(saved_state[0]),
                    set(saved_state[1]), set(saved_state[2]), set(saved_state[3]))
                if self.verbose:
                    log.info(f"[plan] hint veto taken back for {len(deep)} hint(s): the cut "
                             "cannot cover the branch, so the deep hint stays")
                return self._run_once()
            if self.verbose:
                log.info(f"[plan] vetoed {len(deep)} hint(s) predicted to fire past absolute "
                         f"level {refresh_env_cap():g}; the cut covers them instead")
        return res

    def _run_once(self) -> SimResult:
        # useless if the graph has been cleaned of auto_bootstrap nodes (as we do)
        from .sim import step_bts_offsets
        self._step_offsets = step_bts_offsets(self.g, self.bootstrap_level)
        self.refresh.step_bts_offset = self._step_offsets

        max_iters = 2 * len(self.g.nodes)
        import os as _os
        recut = _os.environ.get("PLAN_DRIFT_RECUT") == "1" and self._env_cap_hard
        for it in range(max_iters):
            sim = self._sim()
            if not sim.over_budget:
                # A site is priced by `_capacity` at the level the sim showed when it was cut. A later cut can
                # disarm a hint upstream of it (the hint's input drops below its trigger), and the depth the
                # hint absorbed then lands on the site: legal when chosen, past the envelope now. With
                # PLAN_DRIFT_RECUT=1, un-place such sites and let the cut cover the branch again under
                # `_capacity`; if no envelope-respecting cover exists, keep them and report them as forced.
                # Off by default: the shipped recipes carry such sites (the runtime guard reads the nominal
                # level, which is what their refreshes meet) and a re-cut would rewrite their plans.
                deep = [v for v in (self._deep_placed(sim) if recut and self._drift_rounds < 32 else [])
                        if v not in self.drift_forced]
                if deep:
                    self._drift_rounds += 1
                    saved = (set(self.placed), dict(self.placed_in_eff))
                    for v in deep:
                        self.placed.discard(v)
                        self.placed_in_eff.pop(v, None)
                    if self.verbose:
                        log.info(f"[plan] {len(deep)} placed refresh(es) drifted past the envelope after a hint "
                                 f"upstream stopped firing; un-placed, the cut re-covers them: "
                                 + ", ".join(deep[:6]))
                    try:
                        return self._run_once()
                    except PlanInfeasible:
                        self.placed, self.placed_in_eff = saved
                        self.drift_forced.update(deep)
                        if self.verbose:
                            log.info(f"[plan] no envelope-respecting cover for {len(deep)} drifted site(s); "
                                     "kept as forced: " + ", ".join(deep[:6]))
                        continue
                if self.verbose:
                    log.info(f"[plan] feasible after {it} cut pass(es); "
                          f"{len(self.placed)} refresh(es) placed")
                return sim
            cut_vars = self._one_cut(sim)
            progressed = False
            for v in cut_vars:
                if v not in self.placed:
                    spec = self.refresh.spec(v)
                    if spec.hopeless:
                        raise PlanInfeasible(
                            f"min-cut selected '{v}' but its refresh is destructive",
                            [spec])
                    self.placed.add(v)
                    self.placed_in_eff[v] = (sim.consumed.get(v, 0.0)
                        + (self.g.level_unit if sim.deg.get(v, 1) == 2 else 0))
                    progressed = True
            if not progressed:
                raise PlanInfeasible(
                    f"cut made no progress at iteration {it} "
                    f"({len(sim.over_budget)} op(s) still over budget)")

        sim = self._sim()
        offenders = [self.g.nodes[i] for i, _ in sim.over_budget]
        refusals = []
        for n in offenders:
            for v in n.cipher_inputs:
                s = self.refresh.spec(v)
                if s.hopeless:
                    refusals.append(s)
        raise PlanInfeasible(
            f"still {len(sim.over_budget)} op(s) over budget after {max_iters} passes — "
            "every remaining cut point is unrefreshable", refusals)

    def _deep_placed(self, sim: SimResult) -> list[str]:
        """Placed (cut-chosen) DENSE sites whose refresh now starts past the envelope.

        The measured envelope is the dense route's. A sparse route raises less and tolerates a deeper input
        (the shipped n32 plan runs 512-slot sites at 48 with a rescale pending), and its ceiling is not
        measured, so those stay with the runtime guard.
        """
        out = []
        for v in self.placed:
            p = self.g.producer_of.get(v)
            if p is None or p.is_deliberate_bts or p.hint_level is not None or self.refresh.spec(v).route:
                continue
            if self.bootstrap_level + sim.input_eff(self.g, v, self.g.level_unit) > refresh_env_cap() + 1e-9:
                out.append(v)
        return sorted(out)

    def _one_cut(self, sim: SimResult) -> list[str]:
        """Deep-operand wiring first, binding-operand-only as the fallback.

        Wiring deep non-binding operands lets the cut see P4's blind branches in one
        pass, but such an edge can route through an uncuttable var and make
        `nx.minimum_cut` read the network as unbounded on graphs the binding-only
        network plans fine. If the binding-only pass also fails, its refusal is the one
        reported.
        """
        try:
            return self._one_cut_impl(sim, deep_operands=True)
        except PlanInfeasible:
            return self._one_cut_impl(sim, deep_operands=False)

    def _one_cut_impl(self, sim: SimResult, *, deep_operands: bool) -> list[str]:
        g = self.g
        unit = g.level_unit

        target: dict[str, float] = {}
        over_idx = {i for i, _ in sim.over_budget}
        for i, _ in sim.over_budget:
            n = g.nodes[i]
            B = self.budget.node_budget(
                n, is_terminal=False, output_refreshed=(n.output in self.placed),
                bootstrap_level=self.bootstrap_level)
            for v in n.cipher_inputs:
                if sim.consumed.get(v, 0.0) + n.cost > B:
                    target[v] = min(target.get(v, float("inf")), B - n.cost)
        for n in reversed(g.nodes):
            if n.is_deliberate_bts or not n.output or n.output not in target:
                continue
            t_u = target[n.output]
            for v in n.cipher_inputs:
                # A hint absorbs its input only when it actually fires; gating on the
                # threshold instead would delete this edge under hint_force={v: False}
                # and leave the cut no way to reach the branch.
                if n.hint_level is not None and sim.hint_fired.get(n.idx, False):
                    continue
                if sim.consumed.get(v, 0.0) + n.cost > t_u:
                    target[v] = min(target.get(v, float("inf")), t_u - n.cost)

        frontier = list(target.keys())
        seen = set(frontier)
        while frontier:
            v = frontier.pop()
            p = g.producer_of.get(v)
            if p is None:
                continue
            for u in p.cipher_inputs:
                if u not in seen:
                    seen.add(u)
                    target[u] = target.get(u, float("inf"))
                    frontier.append(u)

        F = nx.DiGraph()
        F.add_node("S")
        F.add_node("T")
        in_net: set[str] = set()
        for v in target:
            cap = self._capacity(v, sim)
            F.add_edge(f"i_{v}", f"o_{v}", capacity=cap)
            in_net.add(v)
            if cap != float("inf"):
                self.ever_eligible.add(v)
        for i, _ in sim.over_budget:
            self.ever_pressured.update(self.g.nodes[i].cipher_inputs)

        eff = {v: sim.effective(v, unit) for v in
               {x for n in g.nodes for x in n.cipher_inputs} | set(sim.consumed)}

        for n in g.nodes:
            if n.is_deliberate_bts:
                continue
            out = n.output
            if n.idx in over_idx:
                op_node = f"op_{n.idx}"
                F.add_edge(op_node, "T", capacity=float("inf"))
                out_level = sim.node_out.get(n.idx, 0.0)
                ceiling = (bool(out) and out_level == self.budget.L)
                if ceiling and out not in in_net:
                    F.add_edge(f"i_{out}", f"o_{out}", capacity=self._capacity(out, sim))
                    in_net.add(out)
                if ceiling:
                    F.add_edge(f"o_{out}", op_node, capacity=float("inf"))
                elif out:
                    F.add_edge(op_node, f"i_{out}", capacity=float("inf"))
                    if out not in in_net:
                        F.add_edge(f"i_{out}", f"o_{out}", capacity=float("inf"))
                        in_net.add(out)
                binding = max((eff.get(v, 0.0) for v in n.cipher_inputs), default=0.0)
                b_n = self.budget.node_budget(
                    n, is_terminal=False, output_refreshed=(out in self.placed),
                    bootstrap_level=self.bootstrap_level)
                wired = 0
                for v in n.cipher_inputs:
                    if n.hint_level is not None and sim.hint_fired.get(n.idx, False):
                        continue
                    if v not in in_net:
                        continue
                    if not self._wire_operand(v, eff, binding, b_n, deep_operands):
                        continue
                    F.add_edge(f"o_{v}", f"i_{out}" if ceiling else op_node,
                               capacity=float("inf"))
                    wired += 1
                if wired == 0 and n.cipher_inputs:
                    for v in n.cipher_inputs:
                        if v in in_net and self._wire_operand(v, eff, binding, b_n, deep_operands):
                            F.add_edge(f"o_{v}", f"i_{out}" if ceiling else op_node,
                                       capacity=float("inf"))
            elif out in target and out not in self.placed:
                op_node = f"op_{n.idx}"
                F.add_edge(op_node, f"i_{out}", capacity=float("inf"))
                binding = max((eff.get(v, 0.0) for v in n.cipher_inputs), default=0.0)
                t_out = target[out]
                for v in n.cipher_inputs:
                    if n.hint_level is not None and sim.hint_fired.get(n.idx, False):
                        continue
                    if v in in_net and self._wire_operand(v, eff, binding, t_out, deep_operands):
                        F.add_edge(f"o_{v}", op_node, capacity=float("inf"))

        for node in list(F.nodes):
            if node not in ("S", "T") and F.in_degree(node) == 0:
                F.add_edge("S", node, capacity=float("inf"))

        if not nx.has_path(F, "S", "T"):
            raise PlanInfeasible(
                "no S->T path in the cut network — the violating ops have no cuttable "
                "ancestors at all: refused rather than silently falling back)")
        try:
            cut_value, (side_s, _) = nx.minimum_cut(F, "S", "T")
        except nx.NetworkXUnbounded as e:
            refusals = [self.refresh.spec(v) for v in sorted(in_net)
                        if self.refresh.spec(v).hopeless]
            over = [f"{self.g.nodes[i].op}@{self.g.nodes[i].step[-60:]}"
                    for i, _ in list(sim.over_budget)[:6]]
            finite = [v for v in sorted(in_net)
                      if not self.refresh.spec(v).hopeless and not self._forbidden(v)][:8]
            raise PlanInfeasible(
                "every S->T path crosses only unrefreshable sites (infinite cut); "
                f"over-budget ops: {over}; finite candidates in net: {finite}",
                refusals) from e
        cut = []
        for v in in_net:
            if f"i_{v}" in side_s and f"o_{v}" not in side_s:
                cut.append(v)
        if not cut:
            raise PlanInfeasible(f"minimum cut is empty (cut_value={cut_value})")
        return sorted(cut)
