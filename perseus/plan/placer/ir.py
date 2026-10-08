from __future__ import annotations

import heapq
import json
import re
from collections import defaultdict
import os
from dataclasses import dataclass, field
from pathlib import Path

# packtag::Field::Real (include/packing/pack_tag.h): the slots hold a real payload.
FIELD_REAL = 1

# Table depth costs, in CKKS levels (x level_unit at use sites).
DEPTH_COSTS: dict[str, int] = {
    "mult": 1, "mult_inplace": 1, "square": 1, "square_inplace": 1,
    "sub_ct": 0, "sub_inplace_ct": 0, "negate": 0, "negate_inplace": 0,
    "level_hint": 0,
}

MULT_FAMILY = frozenset({"mult", "mult_inplace", "square", "square_inplace"})
ADD_FAMILY = frozenset({"add", "add_inplace", "sub", "sub_inplace",
                        "sub_ct", "sub_inplace_ct"})

_LVL_SUFFIX_RE = re.compile(r"-lvl=(-?\d+(?:\.\d+)?)$")


def is_literal_input(name: str) -> bool:
    return (name.startswith("rot(") or name.startswith("const(")
            or name in ("ct_null", "pt_null")
            or name.startswith("hint_lev(") or name.startswith("lvl_cap("))


def is_plaintext_name(name: str) -> bool:
    return name.startswith("pt_") or name.startswith("mask.")


def is_kv_cache_read(var: str) -> bool:
    return "cache.k" in var or "cache.v" in var


def parse_lvl_suffix(name: str) -> float | None:
    m = _LVL_SUFFIX_RE.search(name)
    return float(m.group(1)) if m else None


def _parse_token(inputs: tuple[str, ...], prefix: str) -> float | None:
    for x in inputs:
        if x.startswith(prefix) and x.endswith(")"):
            try:
                return float(x[len(prefix):-1])
            except ValueError:
                pass
    return None


@dataclass(frozen=True)
class Node:
    """One captured op."""
    idx: int                     # position in topological order
    op: str
    inputs: tuple[str, ...]
    input_levels: tuple[float | None, ...]
    output: str
    output_level: float | None
    output_deg: int | None    # captured noise
    step: str
    # magnitude & stats
    max_abs: float | None
    mean: float | None
    max_dev: float | None
    max_coeff: float | None
    pack_period: int | None
    max_coeff_ac: float | None = None   # max_coeff with the DC (X^0) coefficient excluded
    pack_field: int = 0                 # packtag::Field of the slots (FIELD_REAL: a real payload)
    # derived, precomputed once
    cipher_inputs: tuple[str, ...] = ()
    hint_level: float | None = None
    lvl_cap: float | None = None
    cost: int = 0
    cost_forced: bool = False

    @property
    def is_deliberate_bts(self) -> bool:
        return self.op == "deliberate_bootstrap"

    @property
    def is_auto_bts(self) -> bool:
        return self.op == "auto_bootstrap"

    @property
    def is_fold_bts(self) -> bool:
        return self.op == "fold_bootstrap"


@dataclass
class Graph:
    """A captured block in topological order."""
    nodes: list[Node]
    producer_of: dict[str, Node] = field(default_factory=dict)
    inputs: frozenset = frozenset()
    level_unit: int = 1

    @classmethod
    def load(cls, path: Path | str, level_unit: int = 1) -> Graph:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_nodes(doc.get("nodes", []), level_unit=level_unit)

    @classmethod
    def from_nodes(cls, raw: list[dict], level_unit: int = 1) -> Graph:
        if not raw:
            raise ValueError("graph has no nodes")

        # PLAN_DELIB_LANDING_REMAP="old:new,...": captured landings of the code's own (deliberate) bootstraps re-mapped,
        # for a library whose route landings differ from the capture's (e.g. FIDESLIB_BTS_RAISE_DROP moves the s=1 route);
        # the planner re-routes auto bootstraps itself, but a deliberate one keeps its captured output level.
        remap: dict[float, float] = {}
        for tok in os.environ.get("PLAN_DELIB_LANDING_REMAP", "").split(","):
            if ":" in tok:
                a, b = tok.split(":", 1)
                remap[float(a)] = float(b)

        drafts: list[dict] = []
        for n in raw:
            inputs = tuple(str(x) for x in n.get("inputs", []))
            ilv = list(n.get("input_levels", []))
            input_levels = tuple(
                float(ilv[i]) if i < len(ilv) and isinstance(ilv[i], (int, float))
                and ilv[i] >= 0 else None
                for i in range(len(inputs)))
            drafts.append(dict(
                op=str(n.get("op_type", "")),
                inputs=inputs,
                input_levels=input_levels,
                output=str(n.get("output", "")),
                output_level=(remap.get(float(n["output_level"]), float(n["output_level"]))
                              if str(n.get("op_type", "")) == "deliberate_bootstrap" and remap
                              and isinstance(n.get("output_level"), (int, float)) and n["output_level"] >= 0
                              else float(n["output_level"])
                              if isinstance(n.get("output_level"), (int, float))
                              and n["output_level"] >= 0 else None),
                output_deg=(int(n["output_noise_level"])
                            if isinstance(n.get("output_noise_level"), (int, float))
                            else None),
                step=str(n.get("step", "")),
                max_abs=(float(n["output_max_abs"])
                         if isinstance(n.get("output_max_abs"), (int, float)) else None),
                mean=(float(n["output_mean"])
                      if isinstance(n.get("output_mean"), (int, float)) else None),
                max_dev=(float(n["output_max_dev"])
                         if isinstance(n.get("output_max_dev"), (int, float)) else None),
                max_coeff=(float(n["output_max_coeff"])
                           if isinstance(n.get("output_max_coeff"), (int, float)) else None),
                pack_period=(int(n["pack_period"])
                             if isinstance(n.get("pack_period"), (int, float)) else None),
                max_coeff_ac=(float(n["output_max_coeff_ac"])
                              if isinstance(n.get("output_max_coeff_ac"), (int, float)) else None),
                pack_field=int(n.get("pack_field") or 0),
            ))

        for d in drafts:
            lv = list(d["input_levels"])
            for i, name in enumerate(d["inputs"]):
                exp = parse_lvl_suffix(name)
                if exp is None:
                    continue
                if lv[i] is not None and exp < lv[i]:
                    raise ValueError(
                        f"input pin {name} declares level {exp} below the captured "
                        f"{lv[i]} (op {d['op']} -> {d['output']}): pins may only deepen")
                lv[i] = exp
            d["input_levels"] = tuple(lv)

        out_to_i = {d["output"]: i for i, d in enumerate(drafts) if d["output"]}
        indeg: dict[int, int] = defaultdict(int)
        deps: dict[int, list[int]] = defaultdict(list)
        cipher_ins_of: list[tuple[str, ...]] = []
        for i, d in enumerate(drafts):
            cins = tuple(x for x in d["inputs"]
                         if not is_literal_input(x) and not is_plaintext_name(x))
            cipher_ins_of.append(cins)
            for v in cins:
                j = out_to_i.get(v)
                if j is not None:
                    deps[j].append(i)
                    indeg[i] += 1
        heap = [i for i in range(len(drafts)) if indeg[i] == 0]
        heapq.heapify(heap)
        order: list[int] = []
        while heap:
            i = heapq.heappop(heap)
            order.append(i)
            for j in sorted(deps[i]):
                indeg[j] -= 1
                if indeg[j] == 0:
                    heapq.heappush(heap, j)
        if len(order) != len(drafts):
            raise ValueError(f"graph has a cycle ({len(drafts) - len(order)} nodes unreachable)")

        mult_lazy_out = {d["output"]: d["output_level"] for d in drafts
                         if d["op"] in MULT_FAMILY and d["output"]
                         and d["output_level"] is not None}
        nodes: list[Node] = []
        for pos, i in enumerate(order):
            d = drafts[i]
            cins = cipher_ins_of[i]
            base = DEPTH_COSTS.get(d["op"], 0) * level_unit
            forced = False
            if base == 0 and d["op"] not in ("auto_bootstrap", "deliberate_bootstrap"):
                known = [lv for v, lv in zip(d["inputs"], d["input_levels"])
                         if not is_literal_input(v) and not is_plaintext_name(v)
                         and lv is not None]
                n_cipher = len(cins)
                if known and len(known) == n_cipher and d["output_level"] is not None:
                    max_in = max(known)
                    if d["output_level"] > max_in:
                        lazy = any(
                            lv == max_in and mult_lazy_out.get(v) == max_in
                            for v, lv in zip(d["inputs"], d["input_levels"])
                            if v in mult_lazy_out)
                        if not lazy:
                            base = (int(d["output_level"] - max_in)
                                    if level_unit != 1 else 1)
                            forced = True
            nodes.append(Node(
                idx=pos,
                cipher_inputs=cins,
                hint_level=_parse_token(d["inputs"], "hint_lev("),
                lvl_cap=_parse_token(d["inputs"], "lvl_cap("),
                cost=base,
                cost_forced=forced,
                **d,
            ))

        producer = {n.output: n for n in nodes if n.output}
        consumed = {v for n in nodes for v in n.cipher_inputs}
        g_inputs = frozenset(v for v in consumed if v not in producer)
        return cls(nodes=nodes, producer_of=producer, inputs=g_inputs,
                   level_unit=level_unit)

    def max_abs_of(self, v: str) -> float | None:
        p = self.producer_of.get(v)
        return p.max_abs if p else None

    def replace_nodes(self, nodes: list[Node]) -> Graph:
        """A new Graph over a transformed node list (used by the erase pass)."""
        return Graph.from_nodes(
            [_node_to_raw(n) for n in nodes], level_unit=self.level_unit)


def _node_to_raw(n: Node) -> dict:
    """Node -> the raw dict shape from_nodes ingests (round-trip for transforms)."""
    return {
        "op_type": n.op,
        "inputs": list(n.inputs),
        "input_levels": [lv if lv is not None else -1 for lv in n.input_levels],
        "output": n.output,
        "output_level": n.output_level if n.output_level is not None else -1,
        "output_noise_level": n.output_deg,
        "step": n.step,
        "output_max_abs": n.max_abs,
        "output_mean": n.mean,
        "output_max_dev": n.max_dev,
        "output_max_coeff": n.max_coeff,
        "output_max_coeff_ac": n.max_coeff_ac,
        "pack_period": n.pack_period,
        **({"pack_field": n.pack_field} if n.pack_field else {}),
    }
