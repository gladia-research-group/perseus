from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Callable

from .. import btserr
from .ir import FIELD_COMPLEX, FIELD_REAL, Graph, Node, is_literal_input, is_plaintext_name
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


# ops that carry a complex pack from their one ciphertext input to their output, both parts alike
_PACK_LINEAR = frozenset({"rotate", "rotate_sum", "mult", "mult_inplace", "mult_scalar", "rescale", "level_reduce"})


def split_packs(g: Graph) -> dict[str, float]:
    """The vars holding a complex pack whose imaginary part a conjugation splits off downstream (conj(z) - z) and is
    small next to the pack: the split value and its one-input ancestors back to the op that made the pack complex,
    each with that ratio (the split's captured imaginary max over the pack's max). Small means below the band's own
    range (BAND_LO_FRAC / BAND_HI_FRAC): with the pack at the top of a band, that payload sits under its bottom."""
    prod = g.producer_of
    out: dict[str, float] = {}
    for n in g.nodes:
        if n.op not in ("sub_ct", "sub") or len(n.cipher_inputs) != 2 or n.max_abs is None:
            continue
        a, b = n.cipher_inputs
        pa, pb = prod.get(a), prod.get(b)
        z = (b if pa is not None and pa.op == "conjugate" and pa.cipher_inputs == (b,) else
             a if pb is not None and pb.op == "conjugate" and pb.cipher_inputs == (a,) else None)
        pz = prod.get(z) if z is not None else None
        if pz is None or not pz.max_abs:
            continue
        ratio = 0.5 * n.max_abs / pz.max_abs            # conj(z) - z = -2i Im z
        if ratio >= btserr.BAND_LO_FRAC / btserr.BAND_HI_FRAC:
            continue
        while z is not None and z not in out:
            p = prod.get(z)
            if p is None or p.pack_field != FIELD_COMPLEX:
                break
            out[z] = ratio
            ins = p.cipher_inputs
            z = ins[0] if len(ins) == 1 and p.op in _PACK_LINEAR else None
    return out


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
    fallback_cf: bool = False,
    pack_cf: bool = False,
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
    # plan.site_cf (PLAN_SITE_CF, "softmax_1:12,softmax:+2"): over the priced choice, the correction factor of every
    # site named `..._<step>_<k>` (key `<step>_<k>`, else `<step>`), absolute or relative (+n / -n): a drift margin the
    # capture's one trajectory cannot show
    site_cf: dict[str, str] = {}
    for item in filter(None, (refresh.shadow.plan.get("plan.site_cf") or "").split(",")):
        key, _, val = item.rpartition(":")
        site_cf[key.strip()] = val.strip()

    def _site_cf(v: str, cf):
        parts = v.split("_")
        val = site_cf.get("_".join(parts[-2:])) or (site_cf.get(parts[-2]) if len(parts) >= 2 else None)
        if cf is None or val is None:
            return cf
        return int(cf) + int(val) if val[0] in "+-" else int(val)
    off_map: dict[str, float] = {}
    ps_map: dict[str, float] = {}
    sp_map: dict[str, int] = {}
    rd_map: dict[str, int] = {}
    real: list[str] = []
    shipped: list[str] = []
    errs: list[float] = []
    cf_hist: dict[int, int] = {}
    n_missed = 0
    out_level_hist: dict[int, int] = {}
    # a dense site whose input passed the StC-first entry runs the shipped order: with plan.stc_first_reroute its CF is
    # that order's choice (priced on the coefficients, not on the slots the StC-first order meets) and the compile
    # routes it to the plain dense route (`shipped_order`); with `fallback_cf` the CF only, the StC-first route running
    # that order itself below its entry (btsBegin)
    _fb = refresh.shadow.dense_fallback() if refresh.shadow.stc_first_reroute or fallback_cf else None
    # `pack_cf`: a refresh of a complex pack whose small imaginary part is split off downstream (`split_packs`) adds the
    # same absolute noise to both parts, so that payload's error is the noise floor: the lowest CF whose band holds
    # kappa x the site's coefficient max, not the choice sized on the pack. Only where the route's EvalMod sees the
    # coefficients (a sparse route, or a dense refresh in the shipped order): the StC-first order's sees the slots
    packs = split_packs(g) if pack_cf else {}
    kappa = max(refresh.policy.mag_safety, 1.0)
    _dfb = refresh.shadow.dense_fallback() if packs else None
    for v in quality_sites:
        s = refresh.spec(v)
        if (_fb is not None and not s.route and not s.shipped
                and bootstrap_level + sim.input_eff(g, v, unit) > _fb[0] + 1e-9 and refresh.spec_shipped(v).shipped):
            s = refresh.spec_shipped(v)
            if not refresh.shadow.stc_first_reroute:
                s = dataclasses.replace(s, shipped=False)
        cf = _site_cf(v, s.cf)
        p = g.producer_of.get(v)
        coeffs = (bool(s.route) or s.shipped or _dfb is None
                  or bootstrap_level + sim.input_eff(g, v, unit) > _dfb[0] + 1e-9)
        if (cf is not None and v in packs and coeffs and p is not None and p.max_coeff and s.offset is None
                and s.prescale is None):
            cf = min(int(cf), next((c for c in range(refresh.policy.cf_min, int(cf))
                                    if btserr.band(c)[1] >= kappa * p.max_coeff), int(cf)))
        if cf is not None:
            cf_map[v] = int(cf)
            cf_hist[int(cf)] = cf_hist.get(int(cf), 0) + 1
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
        if getattr(s, "shipped", False):
            shipped.append(v)
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
        **({"shipped_order": sorted(shipped)} if shipped else {}),
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


def stamp(doc: dict, capture_stamp: dict | None) -> dict:
    """Write the capture's Config stamp (`contract.capture_stamp`: `Config.load_bearing()` as JSON) into a plan
    document as `doc["config"]`, the contract the run validates the plan against; None leaves the plan unstamped."""
    from ..contract import CONTRACT_KEY, LEGACY_KEY
    doc.pop(LEGACY_KEY, None)
    if capture_stamp is not None:
        doc[CONTRACT_KEY] = capture_stamp
    return doc
