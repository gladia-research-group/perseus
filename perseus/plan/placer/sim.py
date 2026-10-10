from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .ir import ADD_FAMILY, MULT_FAMILY, Graph, Node, is_literal_input, is_plaintext_name


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

    def node_budget(self, n: Node, *, is_terminal: bool, output_refreshed: bool,
                    bootstrap_level: float) -> float:
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
) -> SimResult:
    unit = g.level_unit
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

    def eff(v: str) -> float:
        return lc.get(v, 0.0) + (unit if dg.get(v, 1) == 2 else 0)

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
                dg[n.output] = 2
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
            else:
                c, d = ic, (dg.get(ins[0], 1) if ins else 1)
        elif not ins:
            c, d = 0.0, 1
        elif n.op in MULT_FAMILY:
            c, d = max(eff(v) for v in ins), 2
        elif n.op in ADD_FAMILY:
            has_enc_pt = any(is_plaintext_name(x) for x in n.inputs)
            if len(ins) >= 2:
                degs = {dg.get(v, 1) for v in ins}
                if degs == {2}:
                    c, d = max(lc.get(v, 0.0) for v in ins), 2
                elif degs == {1}:
                    c, d = max(lc.get(v, 0.0) for v in ins), 1
                else:
                    d2eff = max(eff(v) for v in ins if dg.get(v, 1) == 2)
                    d1max = max(lc.get(v, 0.0) for v in ins if dg.get(v, 1) != 2)
                    if d2eff > d1max:
                        c, d = d2eff - unit, 2
                    else:
                        c, d = d1max, 1
            elif has_enc_pt:
                if unit == 1:
                    ct_cap = pt_cap = None
                    for x, lv in zip(n.inputs, n.input_levels):
                        if is_plaintext_name(x):
                            if lv is not None:
                                pt_cap = lv if pt_cap is None else max(pt_cap, lv)
                        elif not is_literal_input(x) and lv is not None:
                            ct_cap = lv if ct_cap is None else max(ct_cap, lv)
                    if (pt_cap is not None and ct_cap is not None
                            and pt_cap > ct_cap):
                        c, d = eff(ins[0]), 1
                    else:
                        c, d = lc.get(ins[0], 0.0), dg.get(ins[0], 1)
                else:
                    c, d = eff(ins[0]), 1
            else:
                c, d = lc.get(ins[0], 0.0), dg.get(ins[0], 1)
        elif n.op in ("negate", "negate_inplace"):
            c, d = eff(ins[0]), 2
        elif n.op == "level_reduce":
            c = (n.output_level - bootstrap_level if n.output_level is not None
                 else (lc.get(ins[0], 0.0) if ins else 0.0))
            d = n.output_deg if n.output_deg is not None else (dg.get(ins[0], 1) if ins else 1)
        elif n.op == "fold_bootstrap":
            c = (n.output_level - bootstrap_level
                 if n.output_level is not None else float(unit))
            d = 2
        else:
            c, d = (lc.get(ins[0], 0.0), dg.get(ins[0], 1)) if ins else (0.0, 1)

        if n.output:
            if n.output in refreshed and n.hint_level is None:
                lc[n.output], dg[n.output] = refreshed[n.output]
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
