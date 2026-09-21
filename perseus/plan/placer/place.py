from __future__ import annotations

import logging
from dataclasses import dataclass, field

import networkx as nx

from .ir import Graph, Node, is_kv_cache_read
from .refresh import RefreshPlanner, RefreshSpec
from .sim import Budget, SimResult, simulate

log = logging.getLogger(__name__)

QUALITY_LAMBDA = 1.0 / 4096.0
REFRESH_ENV_CAP_ABS = 48.0


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
    # Refresh-input envelope hardness. MUST be True exactly when the cut is the SOLE
    # refresher (hints dissolved) — see _capacity. The planner sets it from
    # cfg.dissolve_hints; PLAN_HARD_ENV_CAP overrides either way.
    hard_env_cap: bool = False
    verbose: bool = True

    placed: set[str] = field(default_factory=set)
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
        if self.bootstrap_level + eff_v > REFRESH_ENV_CAP_ABS:
            # Past the refresh envelope. Payable (1e4, large but finite) by default: an
            # inf here can starve the cut entirely when whole paths sit past the envelope,
            # and with hints live they absorb the deep values so the penalty is never
            # exercised. Hard (inf) when the cut is the sole refresher (`hard_env_cap`,
            # set by the planner from dissolve_hints): a bootstrap started past the
            # envelope silently returns garbage. PLAN_HARD_ENV_CAP=0/1 overrides either way.
            import os as _os
            _ov = _os.environ.get("PLAN_HARD_ENV_CAP")
            if _ov == "1":
                return float("inf")
            if _ov == "0":
                return 1.0e4
            return float("inf") if self.hard_env_cap else 1.0e4
        cap = 1.0 + self.miss_penalty * spec.overshoot(self.err_target)
        if self.quality_weight > 0.0:
            q = min(1.0, max(0.0, spec.rel_err / max(self.err_target, 1e-300)))
            cap += QUALITY_LAMBDA * self.quality_weight * q
        return cap

    def run(self) -> SimResult:
        # useless if the graph has been cleaned of auto_bootstrap nodes (as we do)
        from .sim import step_bts_offsets
        self._step_offsets = step_bts_offsets(self.g, self.bootstrap_level)
        self.refresh.step_bts_offset = self._step_offsets

        max_iters = 2 * len(self.g.nodes)
        for it in range(max_iters):
            sim = self._sim()
            if not sim.over_budget:
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
