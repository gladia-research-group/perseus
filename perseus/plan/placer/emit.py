from __future__ import annotations

import math
import re
from collections.abc import Callable

from .ir import FIELD_REAL, Graph, Node, is_literal_input, is_plaintext_name
from .refresh import RefreshPlanner
from .sim import SimResult

_VAR_INDEX_RE = re.compile(r"^v_(\d+)$")

_LINEAR_STEP_KEYS: dict[str, list[str]] = {
    "qkv":          ["q", "k", "v", "kv"],
    "out_proj":     ["out"],
    "up_linear":    ["up"],
    "down_linear":  ["down"],
    "lm_head_tile": ["lm_head_tile_0", "lm_head_tile_1"],
    "feedback":     ["fb_tile_0", "fb_tile_1"],
}


def _weight_keys(step: str, op: str) -> list[str]:
    is_mult = op in ("mult", "mult_inplace", "square", "square_inplace")
    is_add = op in ("add", "add_inplace")
    last = step.rsplit(".", 1)[-1] if step else ""
    if is_mult and last in _LINEAR_STEP_KEYS:
        return list(_LINEAR_STEP_KEYS[last])
    if step.endswith("ln_affine"):
        which = ("ln_1" if "ln_1" in step else "ln_2" if "ln_2" in step
                 else "ln_f" if "ln_f" in step else None)
        if which is None:
            return []
        if is_mult:
            return [f"{which}.weight"]
        if is_add:
            return [f"{which}.bias"]
    if last == "ln_shift" and is_add:
        which = ("ln_1" if "ln_1" in step else "ln_2" if "ln_2" in step
                 else "ln_f" if "ln_f" in step else None)
        if which is not None:
            return [f"{which}.shift"]
    return []


def _mask_site(step: str, op: str) -> str | None:
    is_mult = op in ("mult", "mult_inplace")
    is_add = op in ("add", "add_inplace")
    if not (is_mult or is_add):
        return None
    s = step or ""
    last = s.rsplit(".", 1)[-1]
    if is_add and last == "score_mask_add":
        return "sm.score"
    if not is_mult:
        return None
    return {
        "active_mask_mult": "sm.active", "gmask_mult": "qkt.gmask",
        "lane_mask_mult": "v.lane", "softmax_v_pack": "rpk.pack",
        "q_tok0_mask_mult": "tok0", "head_reduce_sum": "hrs.pos0",
        "mean": "ln.scalemask",
        "attn_residual": "resid", "mlp_residual": "resid",
    }.get(last) or _mask_site_ctx(s, last)


def _mask_site_ctx(s: str, last: str) -> str | None:
    if last == "tok0_mask_mult":
        if "softmax_v" in s:
            return "tok0.h"
        if "cache_k_push" in s or "cache_kv_push" in s:
            return "kpush.tok0h"
        return None
    if last == "im_cleanse":
        if "ln_1" in s:
            return "ln.center.ln_1"
        if "ln_2" in s:
            return "ln.center.ln_2"
        if "gelu" in s:
            return "gelu.half"
        return None
    return None


def _pt_and_ct(n: Node) -> tuple[float | None, str | None]:
    pt_level = None
    for name, lv in zip(n.inputs, n.input_levels):
        if is_plaintext_name(name) and lv is not None:
            pt_level = lv
            break
    ct_var = None
    for name, lv in zip(n.inputs, n.input_levels):
        if is_literal_input(name) or is_plaintext_name(name):
            continue
        if lv is not None:
            ct_var = name
            break
    return pt_level, ct_var


def weight_levels(g: Graph, sim: SimResult, bootstrap_level: int,
                  seed_consumed: Callable[[str], float], max_level: int) -> dict[str, int]:
    unit = g.level_unit
    out: dict[str, int] = {}
    for n in g.nodes:
        if n.is_deliberate_bts:
            continue
        keys = _weight_keys(n.step, n.op)
        if not keys:
            continue
        pt_level, ct_var = _pt_and_ct(n)
        if pt_level is None or ct_var is None:
            continue
        c = sim.consumed.get(ct_var, seed_consumed(ct_var))
        lazy = unit if sim.deg.get(ct_var, 1) == 2 else 0
        enc = min(bootstrap_level + int(math.ceil(c)) + lazy, max_level)
        for k in keys:
            out[k] = max(out.get(k, -1), enc)
    return out


def mask_levels(g: Graph, sim: SimResult, bootstrap_level: int,
                seed_consumed: Callable[[str], float]) -> dict[str, list[int]]:
    unit = g.level_unit
    sites: dict[str, set] = {}
    for n in g.nodes:
        site = _mask_site(n.step, n.op)
        if site is None:
            continue
        pt_level, ct_var = _pt_and_ct(n)
        if pt_level is None or ct_var is None:
            continue
        c = sim.consumed.get(ct_var, seed_consumed(ct_var))
        lazy = unit if sim.deg.get(ct_var, 1) == 2 else 0
        sites.setdefault(site, set()).add(int(bootstrap_level + math.ceil(c) + lazy))
        sites[site].add(int(bootstrap_level + lazy))
    return {k: sorted(v) for k, v in sites.items()}


def _placement_order(var: str) -> tuple:
    m = _VAR_INDEX_RE.match(var)
    return (0, int(m.group(1))) if m else (1, var)


def assemble(
    g: Graph,
    sim: SimResult,
    refresh: RefreshPlanner,
    placed: list[str],
    *,
    bootstrap_level: int,
    max_level: int,
    source_level: int,
    cache_read_level: int,
    seed_consumed: Callable[[str], float],
    err_target: float,
    num_deliberate: int,
    num_nodes_raw: int,
    heuristic_config: str = "min_cut",
    placer_meta: dict | None = None,
    boundary_realize: bool = False,
    realize_anchors: set | None = None,
    real_route: bool = False,
) -> dict:
    unit = g.level_unit

    hint_fire: list[str] = []
    hint_total = 0
    for n in g.nodes:
        if n.hint_level is not None:
            hint_total += 1
            if sim.hint_fired.get(n.idx) and n.output:
                hint_fire.append(n.output)

    exit_var = exit_level = exit_deg = None
    for n in g.nodes:
        if n.output and not n.is_deliberate_bts:
            exit_var = n.output
            exit_level = int(bootstrap_level + math.ceil(sim.consumed.get(n.output, 0.0)))
            exit_deg = int(sim.deg.get(n.output, 1))

    final_named_levels = {
        n.output: bootstrap_level + math.ceil(sim.consumed.get(n.output, 0.0))
        for n in g.nodes if n.output
    }

    final_named_degs = {
        n.output: int(sim.deg.get(n.output, 1)) for n in g.nodes if n.output
    }

    all_sites = sorted(set(placed), key=_placement_order)
    quality_sites = all_sites + [v for v in hint_fire if v not in set(all_sites)]

    rescale_after = list(quality_sites) if refresh.out_deg == 1 else []

    if realize_anchors:
        rescale_after.extend(v for v in realize_anchors if v not in set(rescale_after))
    if (boundary_realize and exit_var is not None and exit_deg == 2
            and exit_var not in set(rescale_after)):

        rescale_after.append(exit_var)
        exit_deg = 1
        exit_level += unit
        final_named_levels[exit_var] += unit
        final_named_degs[exit_var] = 1
    cf_map: dict[str, int] = {}
    off_map: dict[str, float] = {}
    ps_map: dict[str, float] = {}
    sp_map: dict[str, int] = {}
    rd_map: dict[str, int] = {}
    real: list[str] = []
    errs: list[float] = []
    cf_hist: dict[int, int] = {}
    n_missed = 0
    out_level_hist: dict[int, int] = {}
    for v in quality_sites:
        s = refresh.spec(v)
        if s.cf is not None:
            cf_map[v] = int(s.cf)
            cf_hist[int(s.cf)] = cf_hist.get(int(s.cf), 0) + 1
        if s.offset is not None:
            off_map[v] = float(s.offset)
        if s.prescale is not None:
            ps_map[v] = float(s.prescale)
        if s.route:
            sp_map[v] = int(s.route)
        if s.raise_drop:
            rd_map[v] = int(s.raise_drop)
        if real_route and not s.route and v in g.producer_of and g.producer_of[v].pack_field == FIELD_REAL:
            real.append(v)
        if math.isfinite(s.rel_err):
            errs.append(s.rel_err)
        if not s.feasible:
            n_missed += 1
        lvl = int(bootstrap_level + round(s.out_consumed))
        out_level_hist[lvl] = out_level_hist.get(lvl, 0) + 1
    errs.sort()

    def q(p: float) -> float | None:
        return errs[min(len(errs) - 1, int(p * len(errs)))] if errs else None

    worst = sorted(((refresh.spec(v).rel_err, v) for v in quality_sites),
                   reverse=True)[:10]

    placements = [{
        "type": "bootstrap_after_node",
        "node_op": (g.producer_of[v].op if v in g.producer_of else "input"),
        "target_var": v,
    } for v in all_sites]

    sparse_avail = sorted({int(s) for s in refresh.sparse_precomps if int(s) > 0})

    return {
        "version": 2,
        "heuristic_config": heuristic_config,
        "input_graph": {
            "version": 1,
            "num_nodes": num_nodes_raw,
            "num_nodes_deliberate_bootstrap": num_deliberate,
            "num_nodes_optimized": len(g.nodes),
        },
        "rules": {
            "bootstrap_level": bootstrap_level,
            "max_level": max_level,
            "source_level": source_level,
            "cache_read_level": cache_read_level,
            "cache_pin_level": cache_read_level,
            **({"sparse_bts_slots": sparse_avail} if sparse_avail else {}),
        },
        "placements": placements,
        "weight_levels": weight_levels(g, sim, bootstrap_level, seed_consumed, max_level),
        "mask_levels": mask_levels(g, sim, bootstrap_level, seed_consumed),
        "hint_fire": hint_fire,
        "hint_decisions_bound": True,
        "rescale_after": sorted(set(rescale_after), key=_placement_order),
        **({"sparse_slots": sp_map} if sp_map else {}),
        **({"prescale": ps_map} if ps_map else {}),
        **({"correction_factor": cf_map} if cf_map else {}),
        **({"offset": off_map} if off_map else {}),
        **({"raise_drop": rd_map} if rd_map else {}),
        **({"real_route": real} if real else {}),
        "summary": {
            **({"placer_meta": placer_meta} if placer_meta else {}),
            "num_placements": len(all_sites),
            "num_hint_bootstraps": len(hint_fire),
            "num_hints_total": hint_total,
            "num_deliberate_bootstrap_nodes": num_deliberate,
            "final_named_levels": final_named_levels,
            **({"final_named_degs": final_named_degs} if refresh.out_deg == 1 else {}),
            "total_bootstraps": len(all_sites) + len(hint_fire) + num_deliberate,
            "num_weight_levels": len(
                weight_levels(g, sim, bootstrap_level, seed_consumed, max_level)),
            "exit_var": exit_var,
            "exit_level": exit_level,
            "exit_deg": exit_deg,
            "bts_quality": {
                "err_target": err_target,
                "table_measured": refresh.table.measured,
                "num_sites": len(quality_sites),
                "predicted_rel_err": {"median": q(0.5), "p90": q(0.9),
                                      "p99": q(0.99),
                                      "max": errs[-1] if errs else None},
                "num_sites_missing_target": n_missed,
                "cf_histogram": {str(k): v for k, v in sorted(cf_hist.items())},
                "out_level_histogram": {str(k): v for k, v in
                                        sorted(out_level_hist.items())},
                "num_offset": len(off_map),
                "num_prescale": len(ps_map),
            "num_raise_drop": len(rd_map),
                "num_real_route": len(real),
                "num_sparse": len(sp_map),
                "worst_sites": [
                    {"var": v, "rel_err": e,
                     "max_abs": g.max_abs_of(v),
                     "pack_period": (g.producer_of[v].pack_period
                                     if v in g.producer_of else None),
                     "step": (g.producer_of[v].step if v in g.producer_of else None)}
                    for e, v in worst],
            },
        },
    }
