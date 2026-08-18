"""CKKS bootstrap-placement planner: min-cut + range awareness + white-box iterative
resets + FLEXIBLE_AUTO level pinning. Parameter-agnostic: the captured graph is the
single source of truth. Entry point: optimize_global (driven by perseus.plan.driver)."""

from __future__ import annotations

import heapq
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx


DEFAULT_DEPTH_COSTS: Dict[str, int] = {
    "mult":             1,
    "mult_inplace":     1,
    "square":           1,
    "square_inplace":   1,
    "sub_ct":           0,
    "sub_inplace_ct":   0,
    "negate":           0,
    "negate_inplace":   0,
    "level_hint":       0,
}


def cut_capacity(var: str, level_consumed: Dict[str, float], producer_of=None) -> float:
    """Fresh/input vars get ∞ (never cut); everything else gets uniform cost 1."""
    return float('inf') if level_consumed.get(var, 0) == 0 else 1.0


def node_depth_cost(node: dict, depth_costs: Dict[str, int]) -> int:
    if "effective_cost" in node:
        return int(node["effective_cost"])
    op = str(node.get("op_type", ""))
    return int(depth_costs.get(op, 0))


def node_budget(node: dict, L: float, depth_costs: Dict[str, int]) -> float:
    """Per-op output budget: min(L, cap_rel + cost, production ceiling)."""
    cap_rel = node.get("cap_rel")
    base = L if cap_rel is None else min(L, cap_rel + node_depth_cost(node, depth_costs))
    pc = node.get("_prod_ceiling")
    return base if pc is None else min(base, pc)


def compute_effective_cost(
    node: dict,
    depth_costs: Dict[str, int],
    warn_cb: Callable[[str], None],
    mult_lazy_out: Optional[Dict[str, int]] = None,
) -> int:
    op = str(node.get("op_type", ""))
    if is_bootstrap_op(op):
        return int(depth_costs.get(op, 0))
    base_cost = int(depth_costs.get(op, 0))
    if base_cost != 0:
        return base_cost

    input_levels = node.get("input_levels")
    output_level = node.get("output_level", -1)
    if not isinstance(input_levels, list) or output_level is None:
        return base_cost

    if output_level < 0:
        return base_cost

    inputs = node.get("inputs", [])
    known_inputs: List[float] = []
    has_unknown_cipher = False
    for name, lvl in zip(inputs, input_levels):
        if is_plaintext_name(str(name)):
            continue
        if isinstance(lvl, (int, float)) and lvl >= 0:
            known_inputs.append(lvl)
        else:
            has_unknown_cipher = True

    if not known_inputs or has_unknown_cipher:
        return base_cost

    max_in = max(known_inputs)
    if output_level > max_in:
        if mult_lazy_out and output_level == max_in + 1:
            for name, lvl in zip(inputs, input_levels):
                nm = str(name)
                if (isinstance(lvl, (int, float)) and lvl == max_in
                        and mult_lazy_out.get(nm) == int(max_in)):
                    return base_cost
        warn_cb(
            "Zero-cost op shows level change; forcing cost=1. "
            f"op={op} output={node.get('output', '')} "
            f"max_in={max_in} output_level={output_level}"
        )
        return 1

    return base_cost


HINT_LEVEL_SHIFT: float = 0.0
HINT_SHIFT_FLOOR: float = 18.0


def _shifted(level: float) -> float:
    return level + HINT_LEVEL_SHIFT if level > HINT_SHIFT_FLOOR else level


def parse_hint_level(inputs: List[str]) -> Optional[float]:
    """Extract l from any 'hint_lev(l)' input token."""
    for x in inputs:
        x_str = str(x)
        if x_str.startswith("hint_lev(") and x_str.endswith(")"):
            try:
                return _shifted(float(x_str[len("hint_lev("):-1]))
            except ValueError:
                pass
    return None


def parse_lvl_cap(inputs: List[str]) -> Optional[int]:
    """Extract L from a 'lvl_cap(L)' token: the op must run at <= L."""
    for x in inputs:
        xs = str(x)
        if xs.startswith("lvl_cap(") and xs.endswith(")"):
            try:
                return int(_shifted(float(xs[len("lvl_cap("):-1])))
            except ValueError:
                pass
    return None


def parse_lvl_suffix(name: str) -> Optional[float]:
    """Parse a trailing '-lvl=N' level annotation from a variable name."""
    match = re.search(r"-lvl=(-?\d+(?:\.\d+)?)$", name)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def boundary_any_over(
    ordered_nodes: List[dict],
    level_consumed: Dict[str, float],
    L: float,
    depth_costs: Dict[str, int],
    bootstrap_level: float = 16.0,
) -> set:
    """Ops whose simulated output exceeds their per-node budget."""
    T_ops = set()
    for idx, n in enumerate(ordered_nodes):
        op = str(n.get("op_type", ""))
        if is_bootstrap_op(op):
            continue
        out_level = n.get("_sim_out")
        if out_level is None:
            ins = ciphertext_inputs(n)
            max_in = max((level_consumed.get(v, 0) for v in ins), default=0)
            cost = node_depth_cost(n, depth_costs)
            hint_l = parse_hint_level(n.get("inputs", []))
            if hint_l is not None and bootstrap_level + max_in > hint_l:
                out_level = 0.0 + cost
            else:
                out_level = max_in + cost

        if out_level > node_budget(n, L, depth_costs):
            T_ops.add(idx)
    return T_ops


def _dump_over_budget_ops(ordered_nodes, level_consumed, L, depth_costs, cfg_name,
                          limit: int = 12) -> None:
    """Print the ops still over their per-node budget (uses _sim_out)."""
    shown = 0
    for idx, n in enumerate(ordered_nodes):
        if is_bootstrap_op(str(n.get("op_type", ""))):
            continue
        c = n.get("_sim_out", 0.0)
        b = node_budget(n, L, depth_costs)
        if c > b:
            print(f"[{cfg_name}] OVER idx={idx} op={n.get('op_type')} out={n.get('output')} "
                  f"sim_out={c} budget={b} ins={ciphertext_inputs(n)} step={n.get('step','')}")
            shown += 1
            if shown >= limit:
                break


def target_prop_one_pass(
    ordered_nodes: List[dict],
    T_ops: set,
    level_consumed: Dict[str, float],
    L: float,
    depth_costs: Dict[str, int],
    bootstrap_level: float = 16.0,
) -> Dict[str, float]:
    """One backward pass from T_ops, propagating target levels up."""
    target_level: Dict[str, float] = {}

    for idx in T_ops:
        n = ordered_nodes[idx]
        cost = node_depth_cost(n, depth_costs)
        B = node_budget(n, L, depth_costs)
        for v in ciphertext_inputs(n):
            if level_consumed.get(v, 0) + cost > B:
                target_level[v] = min(target_level.get(v, float('inf')), B - cost)

    for n in reversed(ordered_nodes):
        op = str(n.get("op_type", ""))
        if is_bootstrap_op(op):
            continue
        out = str(n.get("output", ""))
        if not out or out not in target_level:
            continue
        t_u = target_level[out]
        cost = node_depth_cost(n, depth_costs)

        hint_l = parse_hint_level(n.get("inputs", []))
        for v in ciphertext_inputs(n):
            if hint_l is not None:
                in_lev = level_consumed.get(v, 0)
                if in_lev + bootstrap_level > hint_l:
                    continue
            
            if level_consumed.get(v, 0) + cost > t_u:
                target_level[v] = min(target_level.get(v, float('inf')), t_u - cost)

    return target_level


def is_literal_input(name: str) -> bool:
    return (
        name.startswith("rot(")
        or name.startswith("const(")
        or name in {"ct_null", "pt_null"}
        or name.startswith("hint_lev(")
        or name.startswith("lvl_cap(")
    )

def is_plaintext_name(name: str) -> bool:
    return name.startswith("pt_") or name.startswith("mask.")

def is_bootstrap_op(op_type: str) -> bool:
    return op_type.lower() == "deliberate_bootstrap"

def is_auto_bootstrap_op(op_type: str) -> bool:
    return op_type.lower() == "auto_bootstrap"

def is_kv_cache_read(var: str) -> bool:
    return "cache.k" in var or "cache.v" in var

def ciphertext_inputs(node: dict) -> List[str]:
    return [
        str(x)
        for x in node.get("inputs", [])
        if not is_literal_input(str(x)) and not is_plaintext_name(str(x))
    ]

def topological_sort(nodes: List[dict]) -> List[dict]:
    output_to_idx: Dict[str, int] = {
        str(n["output"]): i for i, n in enumerate(nodes) if n.get("output")
    }
    in_degree: Dict[int, int] = defaultdict(int)
    dependents: Dict[int, List[int]] = defaultdict(list)
    for i, n in enumerate(nodes):
        for v in ciphertext_inputs(n):
            prod_idx = output_to_idx.get(v)
            if prod_idx is not None:
                dependents[prod_idx].append(i)
                in_degree[i] += 1
    heap = [i for i in range(len(nodes)) if in_degree[i] == 0]
    heapq.heapify(heap)
    ordered: List[dict] = []
    while heap:
        idx = heapq.heappop(heap)
        ordered.append(nodes[idx])
        for child in sorted(dependents[idx]):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                heapq.heappush(heap, child)
    if len(ordered) != len(nodes):
        raise ValueError(f"Graph has a cycle ({len(nodes) - len(ordered)} unreachable nodes)")
    return ordered

def all_cipher_vars(nodes: List[dict]) -> List[str]:
    seen: Dict[str, None] = {}
    for n in nodes:
        out = str(n.get("output", ""))
        if out:
            seen[out] = None
        for v in ciphertext_inputs(n):
            seen[v] = None
    return list(seen.keys())

def relabel_auto_as_deliberate(nodes: List[dict]) -> Tuple[List[dict], int]:
    out_nodes: List[dict] = []
    n_relabeled = 0
    for n in nodes:
        if is_auto_bootstrap_op(str(n.get("op_type", ""))):
            n = dict(n)
            n["op_type"] = "deliberate_bootstrap"
            n["_from_auto"] = True
            n_relabeled += 1
        out_nodes.append(n)
    return out_nodes, n_relabeled


def erase_auto_bootstrap_nodes(nodes: List[dict], max_abs: float,
                               keep_steps: Optional[List[str]] = None,
                               min_abs: float = 0.0,
                               erase_oob_inband: bool = False) -> Tuple[List[dict], int]:
    out_level_of: Dict[str, object] = {
        str(n.get("output", "")): n.get("output_level", -1) for n in nodes if n.get("output")
    }
    producer: Dict[str, dict] = {
        str(n.get("output", "")): n for n in nodes if n.get("output")
    }

    def _has_inband_ancestor(start: str, max_hops: int = 10) -> bool:
        cur = start
        for _ in range(max_hops):
            p = producer.get(cur)
            if p is None:
                return False
            op = str(p.get("op_type", ""))
            if is_bootstrap_op(op) or is_auto_bootstrap_op(op):
                return False
            pma = p.get("output_max_abs")
            if isinstance(pma, (int, float)) and min_abs <= pma <= max_abs:
                return True
            cins = ciphertext_inputs(p)
            if not cins:
                return False
            cur = str(cins[0])
        return False

    remap: Dict[str, str] = {}
    pruned: List[dict] = []
    for n in nodes:
        if is_auto_bootstrap_op(str(n.get("op_type", ""))):
            ins = ciphertext_inputs(n)
            out = str(n.get("output", ""))
            ma = n.get("output_max_abs")
            step = str(n.get("step", ""))
            precision_keep = bool(keep_steps) and any(k in step for k in keep_steps)
            erasable = (isinstance(ma, (int, float)) and ma <= max_abs and not precision_keep)
            if (erase_oob_inband and not erasable and not precision_keep and ins
                    and isinstance(ma, (int, float)) and ma > max_abs
                    and _has_inband_ancestor(ins[0])):
                erasable = True
            if ins and out and erasable:
                remap[out] = ins[0]
                continue
        pruned.append(n)

    def resolve(v: str) -> str:
        seen: set = set()
        while v in remap and v not in seen:
            seen.add(v)
            v = remap[v]
        return v

    result: List[dict] = []
    for n in pruned:
        m = dict(n)
        ils = list(n.get("input_levels", []))
        new_in: List[str] = []
        new_lv: List[object] = []
        for i, x in enumerate(n.get("inputs", [])):
            xs = str(x)
            lvl = ils[i] if i < len(ils) else -1
            if not is_literal_input(xs) and not is_plaintext_name(xs):
                r = resolve(xs)
                new_in.append(r)
                rl = out_level_of.get(r)
                new_lv.append(rl if (r != xs and isinstance(rl, (int, float))) else lvl)
            else:
                new_in.append(x)
                new_lv.append(lvl)
        m["inputs"] = new_in
        m["input_levels"] = new_lv
        result.append(m)
    return result, len(remap)


_LINEAR_STEP_KEYS: Dict[str, List[str]] = {
    "qkv":         ["q", "k", "v", "kv"],
    "out_proj":    ["out"],
    "up_linear":   ["up"],
    "down_linear": ["down"],
    "lm_head_tile": ["lm_head_tile_0", "lm_head_tile_1"],
    "feedback":    ["fb_tile_0", "fb_tile_1"],
}


def weight_keys_for_node(step: str, op_type: str) -> List[str]:
    """The wl() weight key(s) a graph node contributes to, or [] if it isn't a
    per-layer weight apply. mult/square => weight matrix, add => affine bias."""
    is_mult = op_type in ("mult", "mult_inplace", "square", "square_inplace")
    is_add  = op_type in ("add", "add_inplace")
    last = step.rsplit(".", 1)[-1] if step else ""
    if is_mult and last in _LINEAR_STEP_KEYS:
        return list(_LINEAR_STEP_KEYS[last])
    if step.endswith("ln_affine"):
        which = ("ln_1" if "ln_1" in step else
                 "ln_2" if "ln_2" in step else
                 "ln_f" if "ln_f" in step else None)
        if which is None:
            return []
        if is_mult:
            return [f"{which}.weight"]
        if is_add:
            return [f"{which}.bias"]
    if last == "ln_shift" and is_add:
        which = ("ln_1" if "ln_1" in step else
                 "ln_2" if "ln_2" in step else
                 "ln_f" if "ln_f" in step else None)
        if which is not None:
            return [f"{which}.shift"]
    return []


def compute_weight_levels(
    ordered_nodes: List[dict],
    level_consumed: Dict[str, float],
    bootstrap_level: int,
    seed_input_consumed: Callable[[str], float],
    max_level: int,
    level_deg: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """Planned per-weight encode level: planned_abs(ct input) + FLEXIBLE_AUTO lazy offset."""
    def planned_abs(v: str) -> int:
        c = level_consumed[v] if v in level_consumed else seed_input_consumed(v)
        return bootstrap_level + int(math.ceil(c))

    if level_deg is not None:
        planned_deg = level_deg
    else:
        _MULT_OPS = {"mult", "mult_inplace", "square", "square_inplace"}
        planned_deg = {}
        for n in ordered_nodes:
            out = str(n.get("output", ""))
            if not out:
                continue
            op = str(n.get("op_type", ""))
            if level_consumed.get(out, 0) == 0:
                planned_deg[out] = 1
            elif op in _MULT_OPS:
                planned_deg[out] = 2
            else:
                planned_deg[out] = max((planned_deg.get(v, 1) for v in ciphertext_inputs(n)), default=1)

    weight_levels: Dict[str, int] = {}
    for n in ordered_nodes:
        op = str(n.get("op_type", ""))
        if is_bootstrap_op(op):
            continue
        keys = weight_keys_for_node(str(n.get("step", "")), op)
        if not keys:
            continue
        inputs = [str(x) for x in n.get("inputs", [])]
        in_levels = list(n.get("input_levels", []))

        pt_level: Optional[float] = None
        for name, lvl in zip(inputs, in_levels):
            if is_plaintext_name(name) and isinstance(lvl, (int, float)) and lvl >= 0:
                pt_level = float(lvl)
                break
        if pt_level is None:
            continue
        ct_var: Optional[str] = None
        for name, lvl in zip(inputs, in_levels):
            if is_literal_input(name) or is_plaintext_name(name):
                continue
            if isinstance(lvl, (int, float)) and lvl >= 0:
                ct_var = name
                break
        if ct_var is None:
            continue

        lazy_offset = 1 if planned_deg.get(ct_var, 1) == 2 else 0
        enc_level = min(planned_abs(ct_var) + lazy_offset, max_level)
        for k in keys:
            weight_levels[k] = max(weight_levels.get(k, -1), enc_level)
    return weight_levels


def mask_site_for_node(step: str, op_type: str) -> Optional[str]:
    is_mult = op_type in ("mult", "mult_inplace")
    is_add  = op_type in ("add", "add_inplace")
    if not (is_mult or is_add):
        return None
    s = step or ""
    last = s.rsplit(".", 1)[-1]
    if is_add  and last == "score_mask_add":   return "sm.score"
    if is_mult and last == "active_mask_mult":  return "sm.active"
    if is_mult and last == "gmask_mult":        return "qkt.gmask"
    if is_mult and last == "lane_mask_mult":    return "v.lane"
    if is_mult and last == "softmax_v_pack":    return "rpk.pack"
    if is_mult and last == "q_tok0_mask_mult":  return "tok0"
    if is_mult and last == "tok0_mask_mult":
        if "softmax_v"    in s: return "tok0.h"
        if "cache_k_push" in s or "cache_kv_push" in s: return "kpush.tok0h"
        return None
    if is_mult and last == "head_reduce_sum":   return "hrs.pos0"
    if is_mult and last == "mean":              return "ln.scalemask"
    if is_mult and last in ("attn_residual", "mlp_residual"): return "resid"
    if is_mult and last == "im_cleanse":
        if "ln_1" in s: return "ln.center.ln_1"
        if "ln_2" in s: return "ln.center.ln_2"
        if "gelu" in s: return "gelu.half"
        return None
    return None


def compute_mask_levels(
    ordered_nodes: List[dict],
    level_consumed: Dict[str, float],
    bootstrap_level: int,
    seed_input_consumed: Callable[[str], float],
    level_deg: Dict[str, int],
) -> Dict[str, List[int]]:
    """Planned encode level(s) per mask site (compute_weight_levels grouped by site, uncapped)."""
    def planned_abs(v: str) -> int:
        c = level_consumed[v] if v in level_consumed else seed_input_consumed(v)
        return bootstrap_level + int(math.ceil(c))

    sites: Dict[str, set] = {}
    for n in ordered_nodes:
        op = str(n.get("op_type", ""))
        site = mask_site_for_node(str(n.get("step", "")), op)
        if site is None:
            continue
        inputs = [str(x) for x in n.get("inputs", [])]
        in_levels = list(n.get("input_levels", []))
        pt_level = None
        for name, lvl in zip(inputs, in_levels):
            if is_plaintext_name(name) and isinstance(lvl, (int, float)) and lvl >= 0:
                pt_level = float(lvl); break
        if pt_level is None:
            continue
        ct_var = None
        for name, lvl in zip(inputs, in_levels):
            if is_literal_input(name) or is_plaintext_name(name):
                continue
            if isinstance(lvl, (int, float)) and lvl >= 0:
                ct_var = name; break
        if ct_var is None:
            continue
        lazy_offset = 1 if level_deg.get(ct_var, 1) == 2 else 0
        enc_level = planned_abs(ct_var) + lazy_offset
        sites.setdefault(site, set()).add(int(enc_level))
    return {k: sorted(v) for k, v in sites.items()}


@dataclass(frozen=True)
class Placement:
    placement_type: str
    node_op: str
    target_var: str
    def to_dict(self) -> dict:
        return {"type": self.placement_type, "node_op": self.node_op, "target_var": self.target_var}


_VAR_INDEX_RE = re.compile(r"^v_(\d+)$")

def _placement_order(p: Placement) -> tuple:
    """Deterministic emission order (set-collected placements; bytes only, not semantics)."""
    m = _VAR_INDEX_RE.match(p.target_var)
    return (0, int(m.group(1))) if m else (1, p.target_var)


_MULT_FAMILY = {"mult", "mult_inplace", "square", "square_inplace"}
_ADD_FAMILY  = {"add", "add_inplace", "sub", "sub_inplace", "sub_ct", "sub_inplace_ct"}


BTS_OUT_DEG = 2


def sim_forward_deg(
    ordered_nodes: List[dict],
    graph_inputs: set,
    bootstrapped_vars: set,
    bootstrap_level: float,
    seed_consumed_fn: Callable[[str], float],
    seed_deg_fn: Callable[[str], int],
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """FLEXIBLEAUTO (consumed level, noise degree) forward simulator.

    Rules: bootstrap/hint refresh -> (bts_level, 2); rotate/conj/clone pass through;
    mult/square rescale deg-2 inputs then land deg-2; ct+ct add aligns mixed degs to
    deg-1; ct+encoded-pt rescales a deg-2 ct; ct+const(double) is free.
    """
    lc: Dict[str, float] = {}
    dg: Dict[str, int] = {}
    for v in graph_inputs:
        lc[v] = seed_consumed_fn(v)
        dg[v] = seed_deg_fn(v)
    for v in bootstrapped_vars:
        lc[v] = 0
        dg[v] = BTS_OUT_DEG

    step_bts_offset: Dict[str, float] = {}
    for n in ordered_nodes:
        op_t = str(n.get("op_type", ""))
        if not (is_bootstrap_op(op_t) or is_auto_bootstrap_op(op_t)):
            continue
        ol = n.get("output_level")
        if isinstance(ol, (int, float)) and ol >= 0:
            s = str(n.get("step", ""))
            step_bts_offset[s] = max(step_bts_offset.get(s, 0.0),
                                     float(ol) - bootstrap_level)

    def bts_out_consumed(node: dict) -> float:
        """Consumed level a bootstrap at this node actually lands at."""
        ol = node.get("output_level")
        if isinstance(ol, (int, float)) and ol >= 0:
            return max(0.0, float(ol) - bootstrap_level)
        return max(0.0, step_bts_offset.get(str(node.get("step", "")), 0.0))

    def eff(v: str) -> float:
        """Level after the pending rescale this op would force."""
        return lc.get(v, 0) + (1 if dg.get(v, 1) == 2 else 0)

    for n in ordered_nodes:
        op = str(n.get("op_type", ""))
        out = str(n.get("output", ""))
        if is_bootstrap_op(op):
            if out:
                lc[out] = bts_out_consumed(n)
                dg[out] = BTS_OUT_DEG
            continue
        ins = ciphertext_inputs(n)
        has_enc_pt = any(is_plaintext_name(str(x)) for x in n.get("inputs", []))
        hint_l = parse_hint_level(n.get("inputs", []))
        if hint_l is not None:
            ic = lc.get(ins[0], 0) if ins else 0.0
            ic_eff = ic + (1 if (ins and dg.get(ins[0], 1) == 2) else 0)
            fired = bootstrap_level + ic_eff > hint_l
            n["_hint_fired"] = fired
            if fired:
                c, d = 0.0, BTS_OUT_DEG
            else:
                c, d = ic, dg.get(ins[0], 1) if ins else 1
        elif not ins:
            c, d = 0.0, 1
        elif op in _MULT_FAMILY:
            c, d = max(eff(v) for v in ins), 2
        elif op in _ADD_FAMILY:
            ct_ins = ins
            if len(ct_ins) >= 2:
                degs = {dg.get(v, 1) for v in ct_ins}
                if degs == {2}:
                    c, d = max(lc.get(v, 0) for v in ct_ins), 2
                elif degs == {1}:
                    c, d = max(lc.get(v, 0) for v in ct_ins), 1
                else:
                    c, d = max(eff(v) for v in ct_ins), 1
            elif has_enc_pt:
                c, d = eff(ct_ins[0]), 1
            else:
                c, d = lc.get(ct_ins[0], 0), dg.get(ct_ins[0], 1)
        elif op in ("negate", "negate_inplace"):
            c, d = eff(ins[0]), 2
        elif op == "level_reduce":
            ol = n.get("output_level")
            od = n.get("output_noise_level")
            c = (float(ol) - bootstrap_level) if ol is not None else (lc.get(ins[0], 0) if ins else 0.0)
            d = int(od) if od is not None else (dg.get(ins[0], 1) if ins else 1)
        else:
            c, d = lc.get(ins[0], 0), dg.get(ins[0], 1)

        if out:
            if out in bootstrapped_vars:
                lc[out] = max(0.0, step_bts_offset.get(str(n.get("step", "")), 0.0))
                dg[out] = 2
            else:
                lc[out] = c
                dg[out] = d
        n["_sim_out"] = c
    return lc, dg


def optimize_global(
    graph: dict,
    bootstrap_level: int,
    max_level: int,
    source_level: int,
    cache_read_level: int,
    max_abs_threshold: float = 10.0,
    min_abs_threshold: float = 1e-3,
    verbose: bool = True,
    entry_level_override: Optional[int] = None,
    entry_deg_override: Optional[int] = None,
    emit_deliberate_placements: bool = False,
    erase_autobts: bool = False,
    erase_keep_steps: Optional[List[str]] = None,
    erase_oob_inband: bool = False,
    no_relocate: bool = False,
    reloc_min_abs: float = 0.0,
    reloc_safety_margin: float = 1.0,
    forbid_steps: Optional[List[str]] = None,
) -> dict:
    cfg_name = "min_cut"
    depth_costs = DEFAULT_DEPTH_COSTS
    boundary_fn = boundary_any_over
    target_prop_fn = target_prop_one_pass
    capacity_fn = cut_capacity
    level_agg_fn = max
    L = max_level - bootstrap_level

    _forbid = [s for s in (forbid_steps or []) if s]
    def _step_forbidden(prod) -> bool:
        if not _forbid or not prod:
            return False
        step = str(prod.get("step", ""))
        return any(k in step for k in _forbid)

    raw_nodes_input = [
        {
            "op_type": str(n.get("op_type", "")),
            "inputs": [str(x) for x in n.get("inputs", [])],
            "input_levels": list(n.get("input_levels", [])),
            "output": str(n.get("output", "")),
            "output_level": n.get("output_level", -1),
            "output_max_abs": n.get("output_max_abs"),
            "step": str(n.get("step", "")),
        }
        for n in graph.get("nodes", [])
    ]
    if not raw_nodes_input:
        raise ValueError("Graph has no nodes")

    for node in raw_nodes_input:
        inputs = node["inputs"]
        input_levels = node["input_levels"]
        for idx, name in enumerate(inputs):
            if idx < len(input_levels):
                expected_lvl = parse_lvl_suffix(name)
                if expected_lvl is not None:
                    actual_lvl = input_levels[idx]
                    if isinstance(actual_lvl, (int, float)):
                        if expected_lvl < actual_lvl:
                            raise ValueError(
                                f"Variable '{name}' specifies level {expected_lvl} via suffix, "
                                f"which is lower than its actual level {actual_lvl} in op '{node['op_type']}'."
                            )
                    input_levels[idx] = expected_lvl

    if erase_autobts:
        raw_nodes, n_erased = erase_auto_bootstrap_nodes(raw_nodes_input, max_abs_threshold,
                                                         keep_steps=erase_keep_steps,
                                                         min_abs=min_abs_threshold,
                                                         erase_oob_inband=erase_oob_inband)
        raw_nodes, n_kept = relabel_auto_as_deliberate(raw_nodes)
        if verbose:
            print(f"[{cfg_name}] erase-autobts: dropped {n_erased} reactive auto_bootstrap "
                  f"(min-cut re-places{', incl. out-of-band w/ in-band ancestor' if erase_oob_inband else ''}); "
                  f"kept {n_kept} (unsafe |abs|>{max_abs_threshold} + "
                  f"precision-keep {erase_keep_steps or []}) as deliberate")
    else:
        raw_nodes, _ = relabel_auto_as_deliberate(raw_nodes_input)
    deliberate_bootstraps = sum(1 for n in raw_nodes if is_bootstrap_op(str(n.get("op_type", ""))))
    ordered_nodes = topological_sort(raw_nodes)
    cipher_vars = all_cipher_vars(ordered_nodes)

    deliberate_input_vars: set = set()
    if emit_deliberate_placements:
        for n in ordered_nodes:
            if is_bootstrap_op(str(n.get("op_type", ""))):
                ins = n.get("inputs") or []
                if ins:
                    deliberate_input_vars.add(str(ins[0]))


    warned_nodes: set = set()
    def warn_once(node: dict, msg: str) -> None:
        if not verbose:
            return
        key = (node.get("op_type", ""), node.get("output", ""), msg)
        if key in warned_nodes:
            return
        warned_nodes.add(key)
        print(f"[{cfg_name}] {msg}")

    producer_of: Dict[str, dict] = {}
    for n in ordered_nodes:
        out = str(n.get("output", ""))
        if out: producer_of[out] = n

    mult_lazy_out: Dict[str, int] = {}
    for n in ordered_nodes:
        if str(n.get("op_type", "")) in ("mult", "mult_inplace", "square", "square_inplace"):
            out = str(n.get("output", ""))
            olv = n.get("output_level")
            if out and isinstance(olv, (int, float)):
                mult_lazy_out[out] = int(olv)

    for n in ordered_nodes:
        n["effective_cost"] = compute_effective_cost(
            n,
            depth_costs,
            lambda message, node=n: warn_once(node, message),
            mult_lazy_out,
        )

    graph_inputs: set = set()
    for v in cipher_vars:
        if v not in producer_of:
            graph_inputs.add(v)

    input_level_of: Dict[str, int] = {}
    first_consumer_seed: Dict[str, int] = {}
    for n in ordered_nodes:
        ilv = n.get("input_levels", [])
        out_lv = n.get("output_level")
        for idx, v in enumerate(n.get("inputs", [])):
            v = str(v)
            if v not in graph_inputs:
                continue
            if v not in input_level_of and idx < len(ilv):
                lv = ilv[idx]
                if isinstance(lv, (int, float)) and lv >= 0:
                    input_level_of[v] = int(lv)
            if v not in first_consumer_seed and isinstance(out_lv, (int, float)):
                first_consumer_seed[v] = int(out_lv) - node_depth_cost(n, depth_costs)

    def seed_input_consumed(v: str) -> float:
        exp = parse_lvl_suffix(v)
        if "cache." in v:
            lv = exp if exp is not None else cache_read_level
            return lv - bootstrap_level
        if entry_level_override is not None:
            return entry_level_override - bootstrap_level
        if v in first_consumer_seed:
            return first_consumer_seed[v] - bootstrap_level
        if exp is not None:
            return exp - bootstrap_level
        if v in input_level_of:
            return input_level_of[v] - bootstrap_level
        return source_level - bootstrap_level

    def seed_input_deg(v: str) -> int:
        """Noise degree of a graph input (block-entry residuals deg-2, fresh deg-1)."""
        if "cache." in v:
            return 2
        if entry_deg_override is not None:
            return entry_deg_override
        exp = parse_lvl_suffix(v)
        fc = first_consumer_seed.get(v)
        if exp is not None and fc is not None:
            return 2 if exp > fc else 1
        return 2 if seed_input_consumed(v) > 0 else 1

    bootstrapped_vars: set = set()
    added_bootstraps: List[Placement] = []

    n_capped = 0
    for n in ordered_nodes:
        lc = parse_lvl_cap(n.get("inputs", []))
        if lc is not None and int(lc) < max_level:
            n["cap_rel"] = max(0, int(lc) - bootstrap_level)
            n_capped += 1
    if verbose and n_capped:
        print(f"[{cfg_name}] per-op caps active on {n_capped} nodes (level_hint)")

    max_iters = len(ordered_nodes) * 2

    if verbose:
        print(f"[{cfg_name}] Running layer-by-layer min-cut (L={L})...")

    _GUARDED_OPS = {"mult", "mult_inplace", "square", "square_inplace",
                    "sub", "sub_inplace", "sub_ct", "sub_inplace_ct",
                    "negate", "negate_inplace"}

    terminal_node = None
    for n in ordered_nodes:
        if str(n.get("output", "")) and not is_bootstrap_op(str(n.get("op_type", ""))):
            terminal_node = n

    for iters in range(max_iters):
        for n in ordered_nodes:
            out = str(n.get("output", ""))
            checked = str(n.get("op_type", "")) in _GUARDED_OPS
            if (out and out not in bootstrapped_vars
                    and out not in deliberate_input_vars
                    and not is_bootstrap_op(str(n.get("op_type", "")))
                    and (checked or n is terminal_node)):
                n["_prod_ceiling"] = L - 1
            else:
                n.pop("_prod_ceiling", None)

        level_consumed, level_deg = sim_forward_deg(
            ordered_nodes, graph_inputs, bootstrapped_vars,
            bootstrap_level, seed_input_consumed, seed_input_deg)

        max_level_consumed = 0.0
        max_over = -1e9
        for n in ordered_nodes:
            if is_bootstrap_op(str(n.get("op_type", ""))):
                continue
            c = n.get("_sim_out", 0.0)
            max_level_consumed = max(max_level_consumed, c)
            max_over = max(max_over, c - node_budget(n, L, depth_costs))

        if max_over <= 0:
            break

        T_ops = boundary_fn(ordered_nodes, level_consumed, L, depth_costs, bootstrap_level)
        if not T_ops:
            raise RuntimeError("Over budget but no boundary ops found.")

        eff_consumed = {v: level_consumed[v] + (1 if level_deg.get(v, 1) == 2 else 0)
                        for v in level_consumed}

        target_level = target_prop_fn(ordered_nodes, T_ops, eff_consumed, L, depth_costs, bootstrap_level)

        _stack = []
        for _idx in T_ops:
            _stack.extend(ciphertext_inputs(ordered_nodes[_idx]))
        _seen_anc = set()
        while _stack:
            v = _stack.pop()
            if v in _seen_anc:
                continue
            _seen_anc.add(v)
            target_level.setdefault(v, eff_consumed.get(v, 0))
            if v in bootstrapped_vars:
                continue
            p = producer_of.get(v)
            if p is not None and not is_bootstrap_op(str(p.get("op_type", ""))):
                _stack.extend(ciphertext_inputs(p))

        G_flow = nx.DiGraph()
        G_flow.add_node('S'); G_flow.add_node('T')

        vars_in_network = set(target_level.keys())
        for v in vars_in_network:
            prod = producer_of.get(v)
            if prod is not None:
                max_abs = prod.get("output_max_abs")
                if max_abs is None or max_abs > max_abs_threshold or max_abs < min_abs_threshold:
                    cap = float('inf')
                elif _step_forbidden(prod):
                    cap = float('inf')
                else:
                    cap = capacity_fn(v, level_consumed, producer_of)
            else:
                cap = (capacity_fn(v, level_consumed, producer_of)
                       if is_kv_cache_read(v) else float('inf'))

            G_flow.add_edge(f"var_{v}_in",  f"var_{v}_out", capacity=cap)

        for idx, n in enumerate(ordered_nodes):
            op = str(n.get("op_type", ""))
            if is_bootstrap_op(op): continue
            out = str(n.get("output", ""))
            ins = ciphertext_inputs(n)
            cost = node_depth_cost(n, depth_costs)

            hint_l = parse_hint_level(n.get("inputs", []))

            if idx in T_ops:
                op_node = f"op_{idx}"
                B = node_budget(n, L, depth_costs)
                G_flow.add_edge(op_node, 'T', capacity=float('inf'))
                out_v = str(n.get("output", ""))
                out_level = n.get("_sim_out")
                if out_level is None:
                    max_in = max((level_consumed.get(v, 0) for v in ins), default=0)
                    out_level = max_in + cost
                    if hint_l is not None and ins and level_consumed.get(ins[0], 0) + bootstrap_level > hint_l:
                        out_level = cost
                pc = n.pop("_prod_ceiling", None)
                base_B = node_budget(n, L, depth_costs)
                if pc is not None:
                    n["_prod_ceiling"] = pc
                ceiling = bool(out_v) and out_level == L and base_B >= L
                if ceiling and out_v not in vars_in_network:
                    o_prod = producer_of.get(out_v)
                    o_ma = o_prod.get("output_max_abs") if o_prod else None
                    o_cap = (float('inf') if (o_ma is None or o_ma > max_abs_threshold
                                              or o_ma < min_abs_threshold
                                              or _step_forbidden(o_prod))
                             else capacity_fn(out_v, level_consumed, producer_of))
                    G_flow.add_edge(f"var_{out_v}_in", f"var_{out_v}_out", capacity=o_cap)
                    vars_in_network.add(out_v)
                if ceiling:
                    G_flow.add_edge(f"var_{out_v}_out", op_node, capacity=float('inf'))
                elif out_v:
                    G_flow.add_edge(op_node, f"var_{out_v}_in", capacity=float('inf'))
                    G_flow.add_edge(f"var_{out_v}_in", f"var_{out_v}_out", capacity=float('inf'))
                    vars_in_network.add(out_v)
                binding = max((eff_consumed.get(v, 0) for v in ins), default=0)
                for v in ins:
                    if hint_l is not None:
                        in_lev = level_consumed.get(v, 0)
                        if in_lev + bootstrap_level > hint_l:
                            continue
                    if v not in vars_in_network or eff_consumed.get(v, 0) < binding:
                        continue
                    if ceiling:
                        G_flow.add_edge(f"var_{v}_out", f"var_{out_v}_in", capacity=float('inf'))
                    else:
                        G_flow.add_edge(f"var_{v}_out", op_node, capacity=float('inf'))
            elif out in target_level and out not in bootstrapped_vars:
                op_node = f"op_{idx}"
                G_flow.add_edge(op_node, f"var_{out}_in", capacity=float('inf'))
                binding = max((eff_consumed.get(v, 0) for v in ins), default=0)
                for v in ins:
                    if hint_l is not None:
                        in_lev = level_consumed.get(v, 0)
                        if in_lev + bootstrap_level > hint_l:
                            continue
                    if v in vars_in_network and eff_consumed.get(v, 0) >= binding:
                        G_flow.add_edge(f"var_{v}_out", op_node, capacity=float('inf'))

        for node in list(G_flow.nodes):
            if node in ('S', 'T'):
                continue
            if G_flow.in_degree(node) == 0:
                G_flow.add_edge('S', node, capacity=float('inf'))

        if not G_flow.has_node('S') or not nx.has_path(G_flow, 'S', 'T'):
            def _safe(v: str) -> bool:
                prod = producer_of.get(v)
                if prod is None:
                    return False
                ma = prod.get("output_max_abs")
                return ma is not None and ma <= max_abs_threshold
            cut_vars = []
            for idx in T_ops:
                n = ordered_nodes[idx]
                ins_safe = [v for v in ciphertext_inputs(n) if _safe(v)]
                if ins_safe:
                    cut_vars.extend(ins_safe)
                elif _safe(str(n.get("output", ""))):
                    cut_vars.append(str(n.get("output", "")))
            if verbose:
                print(f"[{cfg_name}] no S->T path; fallback selected "
                      f"{len(cut_vars)} safe direct-input/output bts")
                if not cut_vars:
                    for idx in sorted(T_ops)[:6]:
                        n = ordered_nodes[idx]
                        def _mi(v):
                            p = producer_of.get(v)
                            ma = p.get("output_max_abs") if p else None
                            return f"{v}(ma={None if ma is None else round(ma,2)},eff={eff_consumed.get(v)})"
                        print(f"  nopath T-op op_{idx} {n.get('op_type')} sim={n.get('_sim_out')} "
                              f"B={node_budget(n, L, depth_costs)} out={_mi(str(n.get('output','')))} "
                              f"ins={[_mi(v) for v in ciphertext_inputs(n)]} step={str(n.get('step',''))[-60:]}")
        else:
            try:
                cut_value, (reachable, non_reachable) = nx.minimum_cut(G_flow, 'S', 'T')
                if cut_value >= float('inf') or cut_value > 1e12:
                    raise ValueError("No bootstrap placement is possible within the max_abs constraint")
            except (nx.NetworkXError, Exception) as e:
                inf_g = nx.DiGraph()
                for u, w, dd in G_flow.edges(data=True):
                    if dd.get("capacity", 0) == float('inf'):
                        inf_g.add_edge(u, w)
                if inf_g.has_node('S') and inf_g.has_node('T') and nx.has_path(inf_g, 'S', 'T'):
                    path = nx.shortest_path(inf_g, 'S', 'T')
                    desc = []
                    for nname in path:
                        if nname.startswith("var_") and nname.endswith("_in"):
                            vv = nname[4:-3]
                            prod = producer_of.get(vv)
                            ma = prod.get("output_max_abs") if prod else None
                            opn = prod.get("op_type") if prod else "INPUT"
                            desc.append(f"{vv}({opn},max_abs={ma},consumed={level_consumed.get(vv, 0)})")
                        elif nname.startswith("op_"):
                            desc.append(nname)
                    print(f"[unbounded-path L={L}] " + " -> ".join(desc))
                raise ValueError(f"No bootstrap placement is possible within the max_abs constraint: {e}")

            cut_vars = []
            for u in reachable:
                for v_edge in G_flow[u]:
                    if v_edge in non_reachable:
                        if u.startswith("var_") and u.endswith("_in") and v_edge.endswith("_out"):
                            if G_flow[u][v_edge].get("capacity", 0) == float('inf'):
                                continue
                            cut_vars.append(u[4:-3])

        if not cut_vars:
            raise RuntimeError(f"Min-cut found 0 variables to cut (cut_value={cut_value}).")

        progressed = False
        for v in cut_vars:
            if v in bootstrapped_vars:
                continue
            prod = producer_of.get(v)
            if prod is not None:
                max_abs = prod.get("output_max_abs")
                if max_abs is None or max_abs > max_abs_threshold or max_abs < min_abs_threshold:
                    raise ValueError(f"No bootstrap placement is possible: variable '{v}' has output_max_abs {max_abs} outside the safe window [{min_abs_threshold}, {max_abs_threshold}]")
            if v in graph_inputs and not is_kv_cache_read(v):
                raise ValueError(f"No bootstrap placement is possible: variable '{v}' is a fresh input")

            bootstrapped_vars.add(v)
            progressed = True
            added_bootstraps.append(Placement(
                placement_type="bootstrap_after_node",
                node_op=str(prod.get("op_type", "")) if prod else "input",
                target_var=v,
            ))
        if not progressed:
            _dump_over_budget_ops(ordered_nodes, level_consumed, L, depth_costs, cfg_name)
            for idx in list(T_ops)[:4]:
                n = ordered_nodes[idx]
                out = str(n.get("output", ""))
                def _info(v):
                    p = producer_of.get(v)
                    ma = p.get("output_max_abs") if p else None
                    return f"{v}(ma={None if ma is None else round(ma,2)},bts={v in bootstrapped_vars},lc={level_consumed.get(v)})"
                print(f"[{cfg_name}] STUCK op_{idx} {n.get('op_type')} out={_info(out)} "
                      f"ins={[ _info(v) for v in ciphertext_inputs(n) ]}")
            raise RuntimeError("Cut made no progress: every selected var is already "
                               "bootstrapped but pressure persists (see offender dump).")
    else:
        _dump_over_budget_ops(ordered_nodes, level_consumed, L, depth_costs, cfg_name)
        raise RuntimeError("Did not converge within max iterations.")


    def _forward_max_over(bset: set) -> float:
        lc: Dict[str, float] = {}
        for gv in graph_inputs:
            lc[gv] = seed_input_consumed(gv)
        for gv in bset:
            lc[gv] = 0
        mo = -1e9
        for nn in ordered_nodes:
            op = str(nn.get("op_type", "")); o = str(nn.get("output", ""))
            if is_bootstrap_op(op):
                if o: lc[o] = 0
                continue
            ins = ciphertext_inputs(nn)
            inl = [lc.get(x, 0) for x in ins]
            hl = parse_hint_level(nn.get("inputs", []))
            if hl is not None:
                ilc = inl[0] if inl else 0.0
                c = 0.0 if (bootstrap_level + ilc > hl) else ilc
                c += node_depth_cost(nn, depth_costs)
            else:
                c = (level_agg_fn(inl) if inl else 0.0) + node_depth_cost(nn, depth_costs)
            if o:
                lc[o] = 0 if o in bset else c
            mo = max(mo, c - node_budget(nn, L, depth_costs))
        return mo

    uf: Dict[str, str] = {}
    def _find(x):
        uf.setdefault(x, x)
        while uf[x] != x:
            uf[x] = uf[uf[x]]; x = uf[x]
        return x
    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb: uf[ra] = rb
    for nn in ordered_nodes:
        o = str(nn.get("output", ""))
        if o and node_depth_cost(nn, depth_costs) == 0:
            for x in ciphertext_inputs(nn):
                _union(x, o)
    comps: Dict[str, List[str]] = defaultdict(list)
    for v in list(uf.keys()):
        comps[_find(v)].append(v)

    reloc_min_abs = reloc_min_abs if reloc_min_abs > 0 else min_abs_threshold

    def _abs_of(v):
        p = producer_of.get(v)
        return None if p is None else p.get("output_max_abs")
    def _feasible(v):
        a = _abs_of(v)
        return (v in producer_of and v not in graph_inputs and a is not None
                and reloc_min_abs <= a <= max_abs_threshold
                and not _step_forbidden(producer_of.get(v)))

    n_relocated = 0
    for X in ([] if no_relocate else list(bootstrapped_vars)):
        comp = comps.get(_find(X), [X]) if X in uf else [X]
        aX = _abs_of(X)
        if aX is None:
            continue
        cands = sorted(
            (y for y in comp
             if y != X and y not in bootstrapped_vars and _feasible(y)
             and _abs_of(y) is not None and _abs_of(y) < aX - 1e-9),
            key=_abs_of,
        )
        for Y in cands:
            trial = (bootstrapped_vars - {X}) | {Y}
            if _forward_max_over(trial) <= -reloc_safety_margin:
                bootstrapped_vars = trial
                n_relocated += 1
                break
    if n_relocated:
        added_bootstraps = [
            Placement(placement_type="bootstrap_after_node",
                      node_op=str(producer_of[v].get("op_type", "")) if v in producer_of else "input",
                      target_var=v)
            for v in bootstrapped_vars
        ]
        if verbose:
            print(f"[{cfg_name}] range-aware: relocated {n_relocated} bootstrap(s) to "
                  f"level-equivalent friendlier-|abs| vars (count unchanged)")

    level_consumed, level_deg = sim_forward_deg(
        ordered_nodes, graph_inputs, bootstrapped_vars,
        bootstrap_level, seed_input_consumed, seed_input_deg)

    hint_bootstraps = 0
    hint_total = 0
    hint_fire_vars: List[str] = []
    participation_nodes = []
    exit_var, exit_level, exit_deg = None, None, None
    for idx, n in enumerate(ordered_nodes):
        op = str(n.get("op_type", ""))
        out = str(n.get("output", ""))
        if is_bootstrap_op(op):
            continue

        ins = ciphertext_inputs(n)
        participation_nodes.append({
            "op_index": idx,
            "op_type": op,
            "inputs": ins,
            "output": out,
        })

        hint_l = parse_hint_level(n.get("inputs", []))
        if hint_l is not None:
            hint_total += 1
            if n.get("_hint_fired"):
                hint_bootstraps += 1
                if out:
                    hint_fire_vars.append(out)

        if out:
            exit_var = out
            exit_level = int(bootstrap_level + math.ceil(level_consumed.get(out, 0)))
            exit_deg = int(level_deg.get(out, 1))

    final_named_levels = {
        str(n["output"]): bootstrap_level + math.ceil(level_consumed.get(str(n["output"]), 0))
        for n in ordered_nodes if n.get("output")
    }

    weight_levels = compute_weight_levels(
        ordered_nodes, level_consumed, bootstrap_level, seed_input_consumed, max_level,
        level_deg=level_deg,
    )
    if verbose:
        print(f"[{cfg_name}] weight levels: "
              + (", ".join(f"{k}={v}" for k, v in sorted(weight_levels.items()))
                 if weight_levels else "(none — weight steps not found in graph)"))

    mask_levels = compute_mask_levels(
        ordered_nodes, level_consumed, bootstrap_level, seed_input_consumed,
        level_deg=level_deg,
    )
    if verbose:
        print(f"[{cfg_name}] mask levels: "
              + (", ".join(f"{k}={v}" for k, v in sorted(mask_levels.items()))
                 if mask_levels else "(none — mask steps not found in graph)"))

    cache_pin_level = cache_read_level

    placement_dicts = [p.to_dict() for p in sorted(added_bootstraps, key=_placement_order)]
    if emit_deliberate_placements:
        producer_op = {n.get("output"): str(n.get("op_type", "")) for n in ordered_nodes}
        node_mag = {n.get("output"): n.get("output_max_abs") for n in ordered_nodes}
        already = {p.target_var for p in added_bootstraps}
        n_unsafe = 0
        for n in ordered_nodes:
            if str(n.get("op_type", "")).lower() == "deliberate_bootstrap" and n.get("_from_auto"):
                ins = n.get("inputs") or []
                if ins and ins[0] not in already:
                    m = node_mag.get(ins[0])
                    if isinstance(m, (int, float)) and m > max_abs_threshold:
                        n_unsafe += 1
                    placement_dicts.append({
                        "type": "bootstrap_after_node",
                        "node_op": producer_op.get(ins[0], "input"),
                        "target_var": ins[0],
                    })
                    already.add(ins[0])
        if verbose and n_unsafe:
            print(f"[{cfg_name}] emit-deliberate: {n_unsafe} forced bts on |abs|>"
                  f"{max_abs_threshold} (no range-safe alternative — lm_head/residual)")

    return {
        "version": 1,
        "heuristic_config": cfg_name,
        "input_graph": {
            "version": graph.get("version"),
            "num_nodes": len(graph.get("nodes", [])),
            "num_nodes_deliberate_bootstrap": deliberate_bootstraps,
            "num_nodes_optimized": len(ordered_nodes),
        },
        "rules": {
            "bootstrap_level": bootstrap_level,
            "max_level": max_level,
            "source_level": source_level,
            "cache_read_level": cache_read_level,
            "cache_pin_level": cache_pin_level,
        },
        "placements": placement_dicts,
        "weight_levels": weight_levels,
        "mask_levels": mask_levels,
        "hint_fire": hint_fire_vars,
        "hint_decisions_bound": True,
        "var_participation_graph": {
            "version": 1,
            "num_nodes": len(participation_nodes),
            "nodes": participation_nodes,
        },
        "summary": {
            "num_placements": len(added_bootstraps),
            "num_hint_bootstraps": hint_bootstraps,
            "num_hints_total": hint_total,
            "num_deliberate_bootstrap_nodes": deliberate_bootstraps,
            "final_named_levels": final_named_levels,
            "total_bootstraps": len(added_bootstraps) + hint_bootstraps + deliberate_bootstraps,
            "num_weight_levels": len(weight_levels),
            "exit_var": exit_var,
            "exit_level": exit_level,
            "exit_deg": exit_deg,
        },
    }

