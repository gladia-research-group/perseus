from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from perseus import _pc

from .ir import Graph, Node


@dataclass
class SimResult:
    consumed: dict[str, float]
    deg: dict[str, int]
    node_out: dict[int, float]
    hint_fired: dict[int, bool]
    over_budget: list[tuple[int, float]]
    #: pending-rescale degree of each node's output BEFORE any refresh override (node_out holds the level)
    node_deg: dict[int, int] = field(default_factory=dict)

    def effective(self, v: str, unit: int) -> float:
        return self.consumed.get(v, 0.0) + (unit if self.deg.get(v, 1) == 2 else 0)

    def input_state(self, g: Graph, v: str) -> tuple[float, int]:
        """(consumed depth, pending-rescale degree) a refresh of `v` STARTS at: the producing node's own output
        before the sim overwrote it with the landing, or, for a hint node, the ciphertext entering the hint.
        `consumed[v]`/`deg[v]` are the landing for a placed site, which is the one level a refresh-envelope
        check must not read. Unknown degree counts as pending (deg 2)."""
        p = g.producer_of.get(v)
        if p is not None and p.hint_level is not None and p.cipher_inputs:
            src = p.cipher_inputs[0]
            return self.consumed.get(src, 0.0), self.deg.get(src, 1)
        if p is None or p.idx not in self.node_out:
            return self.consumed.get(v, 0.0), self.deg.get(v, 1)
        return self.node_out[p.idx], self.node_deg.get(p.idx, 2)

    def input_eff(self, g: Graph, v: str, unit: int) -> float:
        """Effective consumed depth a refresh of `v` STARTS at (`input_state`, plus one unit when a rescale is
        pending): what the refresh envelope is measured in."""
        c, d = self.input_state(g, v)
        return c + (unit if d == 2 else 0)


@dataclass(frozen=True)
class Budget:
    """The level budget: L = max_level - bootstrap_level, with per-node overrides."""
    L: float
    unit: int
    guard_margin: bool = True

    _GUARDED = frozenset({"mult", "mult_inplace", "square", "square_inplace",
                          "sub", "sub_inplace", "sub_ct", "sub_inplace_ct",
                          "negate", "negate_inplace"})

    #: the StC-first entry (absolute effective level) when the recipe sets one: a deliberate StC-first refresh's input
    #: budget (`node_budget` of such a node bounds its input's consumed depth, one unit kept for a pending rescale)
    stc_first_entry: float | None = None
    #: plan.stc_first_reroute: past the entry the refresh runs the shipped order, so the entry bounds only a K/V push's
    stc_first_reroute: bool = False

    def node_budget(self, n: Node, *, is_terminal: bool, output_refreshed: bool,
                    bootstrap_level: float) -> float:
        if n.is_deliberate_bts:
            if (n.stc_first and self.stc_first_entry is not None
                    and (not self.stc_first_reroute or is_kv_push(n.step))):
                return self.stc_first_entry - bootstrap_level - self.unit
            return float("inf")
        b = self.L
        if n.lvl_cap is not None:
            b = min(b, max(0.0, n.lvl_cap - bootstrap_level) + n.cost)
        if (self.guard_margin and not output_refreshed
                and (n.op in self._GUARDED or is_terminal)):
            b = min(b, self.L - 1)
        return b


def step_bts_offsets(g: Graph, bootstrap_level: float) -> dict[str, float]:
    """Per-step captured bootstrap landing offset."""
    out: dict[str, float] = {}
    for n in g.nodes:
        if (n.is_deliberate_bts or n.is_auto_bts) and n.output_level is not None:
            out[n.step] = max(out.get(n.step, 0.0), n.output_level - bootstrap_level)
    return out


class PlannerShadow:
    """The planner's view of the pc-shadow rules (`_pc.Shadow` in "planner" mode: `Quirks::planner()`, a bottomless
    chain, landings exactly the recipe's) plus the typed recipe knobs the sim reads (`_pc.PlanConfig`).

    The per-op rules of `simulate` are `Shadow.step`; this object is the thin adapter between the planner's
    coordinates (consumed depth below the bootstrap level, pending-rescale degree) and `CtMeta` (absolute level,
    degree), through pc-shadow's own conversions (`_pc.level_from_planner` refuses a fractional depth,
    `_pc.planner_consumed`). Refresh landings per route (`landing`) come from the same engine: the dense landing is
    the bootstrap level, the sparse ones `plan.sparse_bts_out`, the shipped-order (StC-last) dense landing
    `plan.stc_first_fallback`, the StC-first entry `plan.stc_first_entry`.
    """

    def __init__(self, bootstrap_level: float, unit: int, plan: _pc.PlanConfig | None = None):
        bl = float(bootstrap_level)
        if not bl.is_integer() or bl < 0:
            raise ValueError(f"bootstrap_level {bootstrap_level!r} is not a level")
        self.bootstrap_level = int(bl)
        self.unit = int(unit)
        env = dict(plan.to_legacy_env()) if plan is not None else {}
        env["BTS_LEVEL"] = str(self.bootstrap_level)
        self.plan = _pc.PlanConfig.from_legacy_env(env)
        cfg = _pc.Config()
        cfg["chain.composite_degree"] = self.unit
        self.shadow = _pc.Shadow(cfg, "planner", plan=self.plan)
        self.routes = tuple(self.shadow.routes)
        self.delib_deg_captured = (self.plan.get("plan.delib_deg") or "") == "captured"
        # PLAN_STC_FIRST_REROUTE: a deliberate StC-first refresh whose input passed the entry lands where the shipped
        # order lands (the compile re-routes it) instead of being over budget
        self.stc_first_reroute = bool(self.plan.get("plan.stc_first_reroute"))
        entry, fallback = self.shadow.stc_first
        self.stc_first_entry = entry
        self.has_shipped = fallback is not None
        self._nodes: dict[int, tuple[Node, _pc.SimNode]] = {}
        self._probe = _pc.CtMeta(self.bootstrap_level, 1)
        # PLAN_STC_FIRST_KV_LANDING: StC-first runs push K/V in the shipped order (IMPL_STC_FIRST_KV=0), so that
        # deliberate refresh lands where the shipped order lands before its realize (one level up, degree 2)
        self.kv_landing: tuple[float, int] | None = None
        kvl = self.plan.get("plan.stc_first_kv_landing")
        if kvl:
            if not self.has_shipped:
                raise ValueError("plan.stc_first_kv_landing (PLAN_STC_FIRST_KV_LANDING) needs "
                                 "plan.stc_first_fallback (PLAN_STC_FIRST_FALLBACK): the shipped-order landing")
            lv, dg = self.shipped_landing()
            raw = (lv - self.unit, 2) if dg == 1 else (lv, dg)
            lvs, _, dgs = kvl.partition(":")
            if (int(lvs), int(dgs or 2)) != raw:
                raise ValueError(f"plan.stc_first_kv_landing {kvl!r} is not the shipped order's raw landing "
                                 f"{raw[0]}:{raw[1]} (plan.stc_first_fallback realized at {lv}:{dg})")
            self.kv_landing = (float(raw[0] - self.bootstrap_level), raw[1])

    # ── coordinates ──
    def meta(self, consumed: float, deg: int) -> _pc.CtMeta:
        return _pc.CtMeta(_pc.level_from_planner(consumed, self.bootstrap_level), deg)

    def consumed(self, level: int) -> float:
        return float(_pc.planner_consumed(level, self.bootstrap_level))

    # ── landings ──
    def landing(self, route: int) -> tuple[int, int] | None:
        """(absolute level, deg) a planned refresh on `route` (slot count, 0 = dense StC-first/default order) lands
        at; None when the recipe measured no landing for that route."""
        if route not in self.routes:
            return None
        return self.shadow.landing(_pc.Route(slots=route, stc_first=route == 0 and self.has_shipped), self._probe)

    def shipped_landing(self) -> tuple[int, int]:
        """The shipped-order (StC-last) dense landing (`BtsConfig.dense_shipped`)."""
        return self.shadow.landing(_pc.Route.dense(), self._probe)

    def dense_fallback(self) -> tuple[float, float, int] | None:
        """(StC-first entry, landing consumed depth, landing deg): a dense refresh whose effective input sits past the
        entry runs the shipped order and lands there. None unless the recipe names both."""
        if self.stc_first_entry is None or not self.has_shipped:
            return None
        lv, dg = self.shipped_landing()
        return float(self.stc_first_entry), float(lv - self.bootstrap_level), dg

    # ── rules ──
    def node(self, n: Node) -> _pc.SimNode:
        got = self._nodes.get(id(n))
        if got is None or got[0] is not n:
            sn = _pc.SimNode(n.idx, n.op, list(n.inputs), list(n.input_levels), n.output, n.output_level,
                             n.output_deg, n.step, list(n.cipher_inputs), n.hint_level, n.lvl_cap, n.cost,
                             n.cost_forced)
            got = self._nodes[id(n)] = (n, sn)
        return got[1]

    def rule(self, n: Node, ins: tuple[str, ...], lc: dict[str, float], dg: dict[str, int]) -> tuple[float, int]:
        """`Shadow.step` on node `n` whose cipher inputs `ins` are at (lc, dg): its output (consumed, deg)."""
        metas = [self.meta(lc.get(v, 0.0), dg.get(v, 1)) for v in ins]
        out = self.shadow.step(self.node(n), metas).out
        return self.consumed(out.level), out.deg


def is_kv_push(step: str) -> bool:
    """A step that writes the K/V cache (the decode's `blk<b>.kv_push`)."""
    return step.endswith(".kv_push") or ".kv_push." in step


def simulate(
    g: Graph,
    *,
    bootstrap_level: float,
    budget: Budget,
    seed_consumed: Callable[[str], float],
    seed_deg: Callable[[str], int],
    refreshed: dict[str, tuple[float, int]],
    sparse_refreshed: set[str] | None = None,
    step_offsets: dict[str, float] | None = None,
    placed: set[str] | None = None,
    realized: set[str] | None = None,
    hint_force: dict[str, bool] | None = None,
    deliberate_clamp0: bool = False,
    dense_fallback: tuple[float, float, int] | None = None,
    shadow: PlannerShadow | None = None,
) -> SimResult:
    """The planner's level/degree simulation of one block: seeds, refresh overrides, hint decisions, realize
    anchors and the per-node budget here; every other op's rule is pc-shadow's (`shadow.rule` = `Shadow.step`).
    `shadow` defaults to the bare recipe (bootstrap level only, no sparse landings, no StC-first)."""
    unit = g.level_unit
    sh = shadow if shadow is not None else PlannerShadow(bootstrap_level, unit)
    if sh.unit != unit or sh.bootstrap_level != bootstrap_level:
        raise ValueError(f"shadow (bootstrap level {sh.bootstrap_level}, unit {sh.unit}) does not match the "
                         f"simulation (bootstrap level {bootstrap_level}, unit {unit})")
    offs = step_offsets if step_offsets is not None else step_bts_offsets(g, bootstrap_level)
    placed_set: set[str] = placed if placed is not None else set()

    lc: dict[str, float] = {} # level consumed
    dg: dict[str, int] = {} # noise scale degree
    for v in g.inputs:
        lc[v] = seed_consumed(v)
        dg[v] = seed_deg(v)
    for v, (c, d) in refreshed.items():
        lc[v] = c
        dg[v] = d

    node_out: dict[int, float] = {}
    node_deg: dict[int, int] = {}
    hint_fired: dict[int, bool] = {}
    over: list[tuple[int, float]] = []

    terminal: Node | None = None
    for n in g.nodes:
        if n.output and not n.is_deliberate_bts:
            terminal = n

    for n in g.nodes:
        ins = n.cipher_inputs
        if n.is_deliberate_bts:
            if n.output:
                base = (n.output_level - bootstrap_level
                        if n.output_level is not None
                        else max(0.0, offs.get(n.step, 0.0)))
                if deliberate_clamp0 and base + unit < 0:
                    # a landing richer than the bts level even with its pending rescale is a
                    # sparse one; a dense run lands it at the bts level. A dense landing
                    # (bts level - unit, degree 2: the K/V push) is left alone.
                    base = 0.0
                lc[n.output] = base
                # the captured landing degree (a CCMM's SinCBoot ends rescaled, a PCMM finish pends its rescale)
                dg[n.output] = n.output_deg if n.output_deg is not None else 2
                # StC-first runs with the K/V push in the shipped order (plan.stc_first_kv_landing): it lands where
                # that order lands
                if sh.kv_landing is not None:
                    lc[n.output], dg[n.output] = sh.kv_landing
            # a captured StC-first refresh keeps its landing only within the entry; past it the runtime takes the
            # shipped order: a budget violation the cut repairs upstream, or (plan.stc_first_reroute) that order's
            # raw landing
            # (plan.stc_first_reroute) never a K/V push's: that refresh lands the cache state every later position
            # reads at the captured landing, and the shipped order lands it deeper
            if n.stc_first and dense_fallback is not None and ins:
                in_eff = max(lc.get(v, 0.0) + (unit if dg.get(v, 1) == 2 else 0) for v in ins)
                if bootstrap_level + in_eff > dense_fallback[0] + 1e-9:
                    if sh.stc_first_reroute and n.output and not is_kv_push(n.step):
                        c_fb, d_fb = dense_fallback[1], dense_fallback[2]
                        lc[n.output], dg[n.output] = (c_fb - unit, 2) if d_fb == 1 else (c_fb, d_fb)
                    else:
                        over.append((n.idx, bootstrap_level + in_eff - dense_fallback[0]))
            continue

        if n.hint_level is not None:
            ic = lc.get(ins[0], 0.0) if ins else 0.0
            ic_eff = ic + (unit if (ins and dg.get(ins[0], 1) == 2) else 0)
            fired = bootstrap_level + ic_eff > n.hint_level
            # an explicit pin overrides the threshold decision
            if hint_force is not None and n.output in hint_force:
                fired = hint_force[n.output]
            hint_fired[n.idx] = fired
            if fired or n.output in placed_set:
                if n.output in refreshed:
                    c, d = refreshed[n.output]
                else:
                    c, d = 0.0, 2
                # StC-first (PLAN_STC_FIRST_ENTRY): a dense refresh whose input has no room for the StC levels takes
                # the shipped order at run time and lands where that order lands
                if (dense_fallback is not None and n.output not in (sparse_refreshed or ())
                        and bootstrap_level + ic_eff > dense_fallback[0] + 1e-9):
                    c, d = dense_fallback[1], dense_fallback[2]
            else:
                c, d = ic, (dg.get(ins[0], 1) if ins else 1)
        else:
            c, d = sh.rule(n, ins, lc, dg)

        if n.output:
            if n.output in refreshed and n.hint_level is None:
                lc[n.output], dg[n.output] = refreshed[n.output]
                if (dense_fallback is not None and n.output not in (sparse_refreshed or ())
                        and bootstrap_level + c + (unit if d == 2 else 0) > dense_fallback[0] + 1e-9):
                    lc[n.output], dg[n.output] = dense_fallback[1], dense_fallback[2]
            else:
                lc[n.output] = c
                dg[n.output] = d
            if realized and n.output in realized and dg[n.output] == 2:
                lc[n.output] += unit
                dg[n.output] = 1
        node_out[n.idx] = c
        node_deg[n.idx] = d

        b = budget.node_budget(
            n, is_terminal=(n is terminal),
            output_refreshed=bool(n.output and n.output in refreshed),
            bootstrap_level=bootstrap_level)
        if c > b:
            over.append((n.idx, c - b))

    return SimResult(consumed=lc, deg=dg, node_out=node_out,
                     hint_fired=hint_fired, over_budget=over, node_deg=node_deg)
