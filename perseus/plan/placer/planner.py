from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

from .. import btserr
from ..contract import CONTRACT_KEY
from .ir import ADD_FAMILY, Graph, parse_lvl_suffix
from .place import Placer, PlanInfeasible, erase_reactive_bootstraps
from .refresh import CfClampReport, RefreshPlanner
from .sim import Budget, SimResult, step_bts_offsets

log = logging.getLogger(__name__)


def _capture_stamp(block_dir: Path):
    for f in (block_dir / "capture_env.json", block_dir.parent / "capture_env.json"):
        if f.is_file():
            return json.loads(f.read_text(encoding="utf-8"))
    return None
from . import emit


def _raise_candidates(g: Graph, sim: SimResult, forbid_steps, hint_outs, placed) -> set:
    unit = g.level_unit
    out = set()
    for n in g.nodes:
        if n.op not in ADD_FAMILY or n.is_deliberate_bts or n.hint_level is not None:
            continue
        ins = n.cipher_inputs
        if len(ins) < 2:
            continue
        degs = {sim.deg.get(v, 1) for v in ins}
        if degs != {1, 2}:
            continue
        d2 = [v for v in ins if sim.deg.get(v, 1) == 2]
        d2eff = max(sim.consumed.get(v, 0.0) + unit for v in d2)
        d1max = max(sim.consumed.get(v, 0.0) for v in ins if sim.deg.get(v, 1) != 2)
        if d2eff <= d1max:
            continue
        for v in d2:
            if v in placed or v in hint_outs or v in g.inputs:
                continue
            p = g.producer_of.get(v)
            if p is None:
                continue
            step = p.step or ""
            if any(k in step for k in forbid_steps if k):
                continue
            out.add(v)
    return out


@dataclass(frozen=True)
class PlanConfig:
    bootstrap_level: int = 16
    bts_out_deg: int = 1
    max_level: int = 24
    source_level: int = 16
    cache_read_level: int = 17
    level_unit: int = 1
    acc_chain: str = "n32"
    acc_table_path: str | None = None
    cf_min: int = 2
    cf_max: int = 20
    err_target: float = 1e-2
    err_hopeless: float = 0.5
    prescale_reach: float = 2000.0
    mag_safety: float = 2.0
    emit_offset: bool = False
    allow_prescale: bool = True
    miss_penalty: float = 4.0
    prescale_bits_max: float = 9.5
    quality_weight: float = 0.0
    # Cut pricing experiments (place.py): 0 leaves the cut count-only, which is what every
    # shipped plan uses. depth_weight prices a site by the levels its refresh restores,
    # level_weight by how deep its input sits.
    depth_weight: float = 0.0
    depth_form: str = "ratio"       # "ratio" | "linear" | "ab"
    depth_a: float = 1.0
    depth_b: float = 1.0
    level_weight: float = 0.0
    #: measured bootstrap latency per route in ms, 0 = dense (paper Table 7, 32-bit chain).
    #: The 64-bit chain is {0: 36.69, 512: 23.59, 1: 19.95}.
    bts_ms: tuple[tuple[int, float], ...] = ((0, 28.77), (512, 18.59), (1, 16.10))
    ms_discount: float = 1.0
    sparse_precomps: tuple[int, ...] = ()
    sparse_out_levels: tuple[tuple[int, int], ...] = ()
    forbid_steps: tuple[str, ...] = (".var", ".mean", ".qkv")
    boundary_realize: bool = False
    rescale_opt: bool = False
    # Landing feedback: {block_dir_name: {var: absolute_out_level}} of measured per-site
    # refresh landings from a prior run's [planted_bts] ledger. Overrides the route table
    # at exactly the named sites.
    site_bts_out: dict[str, dict[str, int]] | None = None
    # Magnitude keep-anchors: force a refresh at every node whose step contains one of
    # these substrings. The min-cut prices sites by their own captured |m| but is blind
    # to the counterfactual of skipping a cleansing refresh (level-legal, quality-fatal).
    force_step_refresh: tuple[str, ...] = ()
    # Hint veto by step substring: hint sites whose step matches are pinned NOT to fire;
    # the sim re-derives everything else around them. Explicit hint_force pins override.
    hint_veto_steps: tuple[str, ...] = ()
    # Hint dissolution (opt-in, --dissolve-hints). A `bootstrap_hint` is a runtime
    # threshold trigger, so a plan that leans on one carries a refresh whose level is
    # decided at runtime. Dissolution pins every hint NOT to fire and lets the min-cut
    # place the refreshes instead (deterministic, priced, depth-checked in `_capacity`);
    # hints no placement can replace are retained and named in summary.hints_retained.
    # Off by default: bootstrap-neutral and no more accurate than hints-live. Implies the
    # hard refresh envelope, because the cut becomes the sole refresher (see place.py).
    dissolve_hints: bool = False
    # Seed deliberate landings clamped at the bts level (never richer) instead of the
    # signed captured landing.
    deliberate_clamp0: bool = False
    first_entry_level: int | None = None
    first_entry_deg: int | None = None
    verbose: bool = True
    placer: str = "min_cut"
    baseline_rescue: bool = False
    #: baseline placers only: cap a placed refresh's input depth (absolute prime level).
    #: The paper's dense Fhelipe arm uses 48, the chain's measured refresh envelope.
    baseline_depth_cap: float | None = None
    latency_table_path: str | None = None
    #: placer "ilp" (ilp.py): solver time cap per block (s), relative optimality gap at
    #: which it may stop, and what it minimises -- "total" (placed + fired hints) or
    #: "placed" (the min-cut's own objective) or "ms" (per-route latency, `bts_ms`)
    ilp_time_limit: float = 300.0
    ilp_gap: float = 0.0
    ilp_objective: str = "total"
    #: the model's final block may not exit deeper than the min-cut's plan (its exit feeds
    #: the unplanned encrypted argmax; see IlpPlacer.cap_exit)
    ilp_cap_exit: bool = True
    #: remove redundant refreshes from the final plan (any placer, after rescue): greedily,
    #: each placed refresh whose removal keeps the block within budget, lowers the bootstrap
    #: count and adds no refresh input past the envelope.
    prune: bool = False
    #: vars the prune never removes (a caller's forced refreshes, e.g. an upstream deploy's
    #: block-exit refresh)
    prune_keep: tuple = ()


@dataclass
class PlanDiagnostics:
    """In-memory side channel of `plan_block`: the typed clamp report.

    The same numbers are written into the plan file under `summary.bts_quality`
    (`num_cf_clamped`, `num_at_cf_max`, `cf_clamped_sites`, ...) — summary keys are
    additive under the baseline contract (scripts/utils/plan_equiv.py) — but the typed
    object with the site tuples is only available here.
    """
    cf_clamp: CfClampReport | None = None


def _prune_redundant(placer, sim: SimResult, keep=()) -> tuple[SimResult, int]:
    """Drop redundant refreshes from `placer.placed` in name order (see PlanConfig.prune)."""
    from .place import refresh_env_cap
    g = placer.g
    env = refresh_env_cap() - placer.bootstrap_level

    def count(s):
        return len(placer.placed) + sum(1 for f in s.hint_fired.values() if f)

    def past_env(s):
        # a refresh's input level: its producer's output before the refresh, which for a
        # refresh on a hint's output is the hint's input (the sim records a placed hint's
        # output as already refreshed)
        k = 0
        for v in placer.placed:
            p = g.producer_of.get(v)
            if p is None:
                continue
            if p.hint_level is not None and p.cipher_inputs:
                lvl = s.effective(p.cipher_inputs[0], g.level_unit)
            else:
                lvl = s.node_out.get(p.idx, 0.0)
            if lvl > env + 1e-9:
                k += 1
        for n in g.nodes:
            if n.hint_level is not None and s.hint_fired.get(n.idx) and n.cipher_inputs:
                if s.consumed.get(n.cipher_inputs[0], 0.0) > env + 1e-9:
                    k += 1
        return k

    best, deep, removed = count(sim), past_env(sim), 0
    origin = getattr(placer, "_origin", None)
    for v in sorted(placer.placed - set(keep)):
        kept = placer.placed
        placer.placed = kept - {v}
        s2 = placer._sim()
        if not s2.over_budget and count(s2) < best and past_env(s2) <= deep:
            sim, best = s2, count(s2)
            removed += 1
            if origin is not None:
                origin.pop(v, None)
        else:
            placer.placed = kept
    return placer._sim(), removed


def _hint_nodes(g: Graph) -> list:
    return [n for n in g.nodes if n.hint_level is not None and n.output]


def _dissolve_hints(g: Graph, cfg: PlanConfig, refresh: RefreshPlanner
                    ) -> tuple[dict[str, bool], dict[str, str]]:
    """Seed the all-vetoed hint pin set, pre-retaining the irreplaceable ones.

    A hint can only be dissolved if the cut is allowed to place a refresh at the same
    point instead. Two classes never can, and pinning them off would starve the plan
    rather than improve it:
      * `forbidden` — the hint's step is a runtime-owned site (`forbid_steps`, e.g.
        `.qkv`'s aliased fused buffer, where a planted bootstrap is a native segfault);
      * `hopeless` — the refresh spec at that var is destructive (past the EvalMod
        wall), so `_capacity` would refuse the placement anyway.
    Returns (pins, retained{var: reason}); a retained hint simply keeps its threshold
    decision.
    """
    pins: dict[str, bool] = {}
    retained: dict[str, str] = {}
    for n in _hint_nodes(g):
        step = n.step or ""
        if any(k in step for k in cfg.forbid_steps if k):
            retained[n.output] = "forbidden"
        elif refresh.spec(n.output).hopeless:
            retained[n.output] = "hopeless"
        else:
            pins[n.output] = False
    return pins, retained


def _relax_hints(g: Graph, pins: dict[str, bool], retained: dict[str, str],
                 err: PlanInfeasible) -> list[str]:
    """Un-veto the minimum set of hints that could unblock a starved dissolution.

    First tier: hints whose step is named in the refusal (the message carries the
    over-budget ops as `op@step`). Second tier: everything still vetoed — the
    bind-hints behaviour, reached only when the refusal names no hint-bearing step.
    Mutates `pins`/`retained`; returns the vars released this round.
    """
    msg = str(err)
    released = [n.output for n in _hint_nodes(g)
                if pins.get(n.output) is False and n.step and n.step[-60:] in msg]
    if not released:
        released = [v for v, f in pins.items() if f is False]
    for v in released:
        pins.pop(v, None)
        retained[v] = "starved"
    return released


def _load_table(cfg: PlanConfig) -> btserr.AccuracyTable:
    if cfg.acc_table_path:
        return btserr.AccuracyTable.load(cfg.acc_table_path)
    return btserr.AccuracyTable.for_chain(cfg.acc_chain)


def fed_back_entry(graph_file: Path | str) -> tuple[int, int]:
    """(level, deg) at which a token fed back by generation enters the block: the landing of
    the feedback's refresh, as the capture's own refreshes record it (the dense landing; a
    sparse one is richer). A block planned from this entry serves that token."""
    nodes = json.loads(Path(graph_file).read_text())["nodes"]
    n = max((n for n in nodes if n["op_type"].endswith("bootstrap")),
            key=lambda n: n["output_level"])
    return int(n["output_level"]), int(n["output_noise_level"])



def plan_fed_back_block0(plan_fn, graph_dir: Path | str, out_dir: Path | str) -> Path:
    """Block 0 once more, for a token fed back by generation: ``plan_fn(0, entry)`` from
    :func:`fed_back_entry`, written as block_0_feedback_placement.json next to the plan. Its
    exit must be block 0's, since block 1's plan binds to that exit."""
    lvl, deg = fed_back_entry(Path(graph_dir) / "block_0" / "graph.json")
    r = plan_fn(0, dict(entry_level=lvl, entry_deg=deg))
    if r is None:
        raise ValueError(f"block 0 is infeasible from the fed-back entry ({lvl}, deg {deg})")
    key = lambda s: (s.get("exit_var"), s.get("exit_level"), s.get("exit_deg") or 1)  # noqa: E731
    b0 = json.loads((Path(out_dir) / "block_0_placement.json").read_text())["summary"]
    if key(r["summary"]) != key(b0):
        raise ValueError(f"the fed-back entry moves block 0's exit {key(b0)} -> "
                         f"{key(r['summary'])}: block 1's plan would not bind")
    p = Path(out_dir) / "block_0_feedback_placement.json"
    p.write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(f"block_0 fed back from ({lvl}, deg {deg}): total={r['summary']['total_bootstraps']}",
          flush=True)
    return p


def plan_block(graph_file: Path | str, cfg: PlanConfig, *,
               entry_level: int | None = None,
               entry_deg: int | None = None,
               table: btserr.AccuracyTable | None = None,
               force_place: set | None = None,
               hint_force: dict[str, bool] | None = None,
               diagnostics: PlanDiagnostics | None = None,
               final_block: bool = False) -> dict:
    """Plan one block's graph.json; returns the plan document, written to disk verbatim.

    `diagnostics`, if given, is filled with what must stay OUT of that document (the
    cf_max clamp report); everything else about the plan is in the returned dict.
    """
    raw = json.loads(Path(graph_file).read_text(encoding="utf-8"))
    num_nodes_raw = len(raw.get("nodes", []))
    g0 = Graph.from_nodes(raw.get("nodes", []), level_unit=cfg.level_unit)

    g, erase = erase_reactive_bootstraps(g0)
    if cfg.verbose:
        log.info(f"[plan] erased {erase.n_erased} reactive auto_bootstrap(s); "
              f"{erase.n_deliberate_kept} deliberate fixed point(s) kept")

    input_level_of: dict[str, float] = {}
    first_consumer_seed: dict[str, float] = {}
    for n in g.nodes:
        for i, v in enumerate(n.inputs):
            if v not in g.inputs:
                continue
            if v not in input_level_of and i < len(n.input_levels) \
                    and n.input_levels[i] is not None:
                input_level_of[v] = float(n.input_levels[i])
            if v not in first_consumer_seed and n.output_level is not None:
                first_consumer_seed[v] = float(n.output_level) - n.cost

    def seed_consumed(v: str) -> float:
        exp = parse_lvl_suffix(v)
        if "cache." in v:
            lv = exp if exp is not None else cfg.cache_read_level
            return lv - cfg.bootstrap_level
        if entry_level is not None:
            return entry_level - cfg.bootstrap_level
        if v in first_consumer_seed:
            return first_consumer_seed[v] - cfg.bootstrap_level
        if exp is not None:
            return exp - cfg.bootstrap_level
        if v in input_level_of:
            return input_level_of[v] - cfg.bootstrap_level
        return cfg.source_level - cfg.bootstrap_level

    def seed_deg(v: str) -> int:
        if "cache." in v:
            return 2
        if entry_deg is not None:
            return entry_deg
        exp = parse_lvl_suffix(v)
        fc = first_consumer_seed.get(v)
        if exp is not None and fc is not None:
            return 2 if exp > fc else 1
        return 2 if seed_consumed(v) > 0 else 1

    policy = btserr.SitePolicy(
        cf_min=cfg.cf_min, cf_max=cfg.cf_max,
        err_target=cfg.err_target, err_hopeless=cfg.err_hopeless,
        prescale_reach=cfg.prescale_reach,
        allow_prescale=cfg.allow_prescale,
        allow_offset=cfg.emit_offset,
        allow_sparse=bool(cfg.sparse_precomps),
        mag_safety=cfg.mag_safety,
        miss_penalty=cfg.miss_penalty,
        prescale_bits_max=cfg.prescale_bits_max,
    )
    block_name = Path(graph_file).parent.name
    refresh = RefreshPlanner(
        table=table if table is not None else _load_table(cfg),
        policy=policy,
        graph=g,
        out_deg=cfg.bts_out_deg,
        sparse_precomps=tuple(cfg.sparse_precomps),
        sparse_out_levels=dict(cfg.sparse_out_levels),
        site_out_levels=dict((cfg.site_bts_out or {}).get(block_name, {})),
        bootstrap_level=cfg.bootstrap_level,
        step_bts_offset=step_bts_offsets(g, cfg.bootstrap_level),
    )

    # compile the step substrings into per-var hint_force=False pins, so the pin
    # plumbing (Placer, the rescale_opt trial placers, emit) carries the veto and
    # the sim re-derives every other decision around the vetoed sites
    if cfg.hint_veto_steps:
        veto = {n.output: False for n in g.nodes
                if n.hint_level is not None and n.output
                and any(k in n.step for k in cfg.hint_veto_steps if k)}
        if veto:
            if cfg.verbose:
                log.info(f"[plan] hint_veto_steps: {len(veto)} hint site(s) vetoed")
            hint_force = {**veto, **(hint_force or {})}

    # Hint dissolution (default): veto every dissolvable hint up front. The pins are
    # kept SEPARATE from `hint_force` so the starvation retry can release dissolution's
    # own pins without ever touching a caller's (or hint_veto_steps') — releasing an
    # experiment's veto to unstarve a block would silently void the experiment.
    hints_retained: dict[str, str] = {}
    dissolve_pins: dict[str, bool] = {}
    n_hints_total = len(_hint_nodes(g))
    # min_cut only: the baseline placers model hints inside their own world-model
    # (`hint_aware`) with hints live; dissolving them there would change what the
    # benchmark measures.
    exact = cfg.placer in ("min_cut", "ilp")     # ours: the cut, or its exact twin
    if cfg.dissolve_hints and exact:
        dissolve_pins, hints_retained = _dissolve_hints(g, cfg, refresh)

    # stage 4 — ours, or a baseline placer (same seam, different site selection)
    budget = Budget(L=float(cfg.max_level - cfg.bootstrap_level), unit=g.level_unit)
    placer_kwargs = dict(
        g=g, refresh=refresh, budget=budget,
        bootstrap_level=float(cfg.bootstrap_level),
        seed_consumed=seed_consumed, seed_deg=seed_deg,
        forbid_steps=cfg.forbid_steps,
        err_target=cfg.err_target, miss_penalty=cfg.miss_penalty,
        quality_weight=cfg.quality_weight, verbose=cfg.verbose,
        depth_weight=cfg.depth_weight, level_weight=cfg.level_weight,
        depth_form=cfg.depth_form, depth_a=cfg.depth_a, depth_b=cfg.depth_b,
        bts_ms=dict(cfg.bts_ms), ms_discount=cfg.ms_discount,
        deliberate_clamp0=cfg.deliberate_clamp0,
        # Coupled to dissolution: with hints live they absorb the deep values and a
        # payable envelope penalty is never exercised; with them dissolved the cut is the
        # sole refresher and must be forbidden from deep sites, not merely discouraged.
        hard_env_cap=bool(cfg.dissolve_hints and exact),
    )
    PlacerCls = Placer
    if cfg.placer == "ilp":
        from functools import partial

        from .ilp import IlpPlacer
        PlacerCls = partial(IlpPlacer, time_limit=cfg.ilp_time_limit, mip_gap=cfg.ilp_gap,
                            objective=cfg.ilp_objective,
                            cap_exit=bool(final_block and cfg.ilp_cap_exit))
    if exact:
        # force_place: pre-seeded placements (the terminal-exit refresh retry in
        # plan_graph_dir, and the magnitude keep-anchors). The cut still adds
        # whatever else it needs; every forced site goes through P2.
        forced = set(force_place or ())
        if cfg.force_step_refresh:
            # One anchor per matching step SCOPE — its last node, i.e. the scope's
            # result (where the canonical/era plans refreshed), not every interior.
            last_of_step: dict[str, str] = {}
            for n in g.nodes:
                if (n.output and not n.is_deliberate_bts and n.hint_level is None
                        and any(k in n.step for k in cfg.force_step_refresh if k)):
                    last_of_step[n.step] = n.output
            forced |= set(last_of_step.values())
            if cfg.verbose and forced:
                log.info(f"[plan] force_step_refresh: {len(forced)} keep-anchor site(s)")
        # Dissolution retry: a starved cut releases the hints nearest the starvation
        # and tries again, so a block that genuinely needs a hint keeps exactly the
        # ones it needs instead of falling back to all of them (or refusing).
        caller_force = dict(hint_force or {})
        for _round in range(4):
            hint_force = {**dissolve_pins, **caller_force}
            placer = PlacerCls(**placer_kwargs, placed=set(forced),
                               hint_force=hint_force)
            try:
                sim = placer.run()
                break
            except PlanInfeasible as _e:
                released = (_relax_hints(g, dissolve_pins, hints_retained, _e)
                            if cfg.dissolve_hints else [])
                if not released:
                    raise
                if cfg.verbose:
                    log.info(f"[plan] hint dissolution starved — retaining "
                          f"{len(released)} hint(s): {', '.join(released[:6])}")
        else:
            raise PlanInfeasible(
                "hint dissolution could not converge in 4 rounds; retained "
                f"{len(hints_retained)}/{n_hints_total} hint(s)")
    else:
        from .baselines import make_placer
        placer = make_placer(cfg.placer, rescue=cfg.baseline_rescue,
                             depth_cap=cfg.baseline_depth_cap,
                             latency_table_path=cfg.latency_table_path,
                             **placer_kwargs)
        sim = placer.run()

    rescale_anchors: set = set()
    if cfg.rescale_opt and exact:
        hint_outs = {n.output for n in g.nodes if n.hint_level is not None and n.output}

        def _count(s: SimResult, p: Placer) -> int:
            return len(p.placed) + sum(1 for f in s.hint_fired.values() if f)

        best = _count(sim, placer)
        base = best
        for _ in range(4):
            cands = _raise_candidates(g, sim, cfg.forbid_steps, hint_outs,
                                      placer.placed) - rescale_anchors
            if not cands:
                break
            trial = rescale_anchors | cands
            p2 = PlacerCls(**{**placer_kwargs, "verbose": False}, realized=set(trial),
                           hint_force=hint_force)
            try:
                s2 = p2.run()
            except PlanInfeasible:
                break
            c2 = _count(s2, p2)
            if c2 <= best:
                rescale_anchors, sim, placer, best = trial, s2, p2, c2
            else:
                break
        if cfg.verbose:
            log.info(f"[plan] rescale_opt: {len(rescale_anchors)} realize anchor(s), "
                  f"bootstraps {base} -> {best}")

    if cfg.prune:
        sim, n_pruned = _prune_redundant(placer, sim, cfg.prune_keep)
        if hasattr(placer, "meta"):
            placer.meta["pruned"] = n_pruned
        if cfg.verbose:
            log.info(f"[plan] prune: {n_pruned} redundant refresh(es) removed")

    assert not sim.over_budget, "P1 violated: final sim has over-budget ops"
    # P2 covers every refresh the plan will execute, not only the min-cut placements:
    # fired hint sites and deliberate fixed points bypass the cut, so without P2 a forced
    # high correction factor could emit a sign-inverting refresh at a hint without refusing
    # (measured: predicted rel_err 3.66 at CF=14, err_hopeless 0.5).
    # A placed site that merely misses err_target is priced by the cut; a hint or
    # deliberate site has no capacity in the cut network, so "priced" means nothing there
    # and it is reported as a warning instead.
    fired_hints = {g.nodes[i].output for i, f in sim.hint_fired.items()
                   if f and g.nodes[i].output}
    deliberate = {n.output for n in g.nodes if n.is_deliberate_bts and n.output}
    missed, lossy = [], []
    for v in sorted(set(placer.placed) | fired_hints | deliberate):
        s = refresh.spec(v)
        kind = ("placed" if v in placer.placed else
                "hint" if v in fired_hints else "deliberate")
        if s.hopeless:
            raise PlanInfeasible(f"P2 violated: destructive refresh emitted at {v} ({kind})", [s])
        if not s.feasible:
            (missed if kind == "placed" else lossy).append((kind, s))
    if missed and cfg.verbose:
        log.info(f"[plan]  {len(missed)} placed site(s) miss err_target="
              f"{cfg.err_target:g} (priced, not refused): "
              + ", ".join(f"{s.var}({s.rel_err:.2g})" for _, s in missed[:6]))
    if lossy:
        log.warning(f"[plan]  {len(lossy)} hint/deliberate site(s) miss err_target="
                    f"{cfg.err_target:g} and cannot be priced by the cut: "
                    + ", ".join(f"{s.var}[{k}]({s.rel_err:.2g})" for k, s in lossy[:6]))
    # The CF ceiling. choose_site is an argmin over cf_min..cf_max, so a site at cf_max is
    # either the cheapest CF or merely the last one priced; only the second is a clamp
    # (btserr.ceiling_binds). Counted over the sites that carry a CF in the plan (placed +
    # fired hints = emit's quality_sites, so num_sites == sum(cf_histogram)). Written into
    # summary.bts_quality after assemble (additive summary keys; see plan_equiv.py).
    clamp = refresh.cf_clamp_report(sorted(set(placer.placed) | fired_hints))
    if diagnostics is not None:
        diagnostics.cf_clamp = clamp
    if clamp.clamped_missing_target or cfg.verbose:
        ids = ", ".join(clamp.clamped[:6])
        if clamp.num_clamped > 6:
            ids += f" (+{clamp.num_clamped - 6} more)"
        # WARNING only when a clamped site also misses err_target: the ceiling binds AND
        # costs the target, so raising --cf-max is the action. A clamped site inside the
        # target is a diagnostic only.
        say = log.warning if clamp.clamped_missing_target else log.info
        say(f"[plan] cf clamp: {clamp.num_clamped} of {clamp.num_sites} typed sites at "
            f"cf_max={clamp.cf_max} ({clamp.num_pinned} pinned; "
            f"{len(clamp.clamped_missing_target)} miss err_target={cfg.err_target:g})"
            + (f": {ids}" if ids else ""))
    eff_deep = []
    for v, c in sim.consumed.items():
        lvl = cfg.bootstrap_level + math.ceil(c)
        if lvl > cfg.max_level:
            raise PlanInfeasible(
                f"P3 violated: {v} predicted at level {lvl} > max_level {cfg.max_level}")
        if lvl + (cfg.level_unit if sim.deg.get(v, 1) == 2 else 0) > cfg.max_level:
            eff_deep.append((v, lvl))
    if eff_deep and cfg.verbose:
        log.info(f"[plan]  P3b: {len(eff_deep)} var(s) EFFECTIVELY past max_level "
              f"(deg-2 at the nominal ceiling) — the chain was never measured there: "
              + ", ".join(f"{v}@{l}+u" for v, l in eff_deep[:6]))
    # P3c — refresh depth, a hard refusal. A bootstrap started at absolute level
    # >= REFRESH_ENV_CAP_ABS + level_unit silently returns garbage (round-trip rel_err
    # 3.7e-1 at level 48 -> 1.96e4 at level 50) and the runtime raises nothing.
    # `_capacity` already refuses these for cut-chosen sites; hint-fired refreshes never
    # pass through it, hence this check covers both.
    from .place import refresh_env_cap as _refresh_env_cap
    _ENV_CAP = _refresh_env_cap()
    _bad_cut, _bad_hint = [], []
    # A var past the envelope that nothing refreshes is NOT harmless: the runtime meets it
    # there and fires a REACTIVE bootstrap, which is the one class neither the cut nor the
    # hint check covers, and it stops the run with [bts_depth_error]. Collect it.
    _hint_covered = {n.output for n in g.nodes
                     if n.hint_level is not None and sim.hint_fired.get(n.idx) and n.output}
    _bad_reactive = []
    for _v, _c in sim.consumed.items():
        _eff = cfg.bootstrap_level + _c + (cfg.level_unit if sim.deg.get(_v, 1) == 2 else 0)
        if _eff <= _ENV_CAP:
            continue
        if _v in placer.placed:
            _bad_cut.append((_v, _eff))
        elif _v not in _hint_covered:
            _bad_reactive.append((_v, _eff))
    # Hint-fired refreshes: `hint_fired` is keyed by node index, and the depth that
    # matters is the level of the ct entering the hint (its input var) — the hint's own
    # output is post-refresh and always shallow. Warning, not a refusal: the sim
    # over-predicts hint input levels, so the runtime [bts_depth_error] guard remains
    # the authority for hint depth.
    for _n in g.nodes:
        if _n.hint_level is None or not sim.hint_fired.get(_n.idx):
            continue
        _in = _n.cipher_inputs[0] if _n.cipher_inputs else None
        if _in is None:
            continue
        _eff = (cfg.bootstrap_level + sim.consumed.get(_in, 0.0)
                + (cfg.level_unit if sim.deg.get(_in, 1) == 2 else 0))
        if _eff > _ENV_CAP:
            _bad_hint.append((_n.output or _in, _eff))
    if _bad_hint and cfg.verbose:
        log.info(f"[plan] P3c(hint): {len(_bad_hint)} hint-fired refresh(es) PREDICTED past the "
              f"envelope (abs level > {_ENV_CAP:g}): "
              + ", ".join(f"{v}@{l:g}" for v, l in _bad_hint[:6])
              + " — WARNING ONLY: this prediction is known to over-fire; the runtime "
                "[bts_depth_error] guard is the authority.")
    if _bad_cut and getattr(placer, "env_cap_relaxed", False):
        # the placer already proved no envelope-respecting placement exists (it tried that
        # pass first and it could not be planned), so these sites are forced, not chosen
        log.warning(
            f"[plan] P3c: {len(_bad_cut)} refresh(es) start past the measured envelope "
            f"(abs level > {_ENV_CAP:g}) and NO placement avoids them -- a bootstrap there "
            f"returns garbage silently, so the runtime [bts_depth_error] guard is what "
            f"stands between this plan and a wrong answer:\n  "
            + ", ".join(f"{v}@{l:g}" for v, l in _bad_cut[:6]))
    elif _bad_cut:
        _msg = (f"P3c violated: {len(_bad_cut)} CUT-CHOSEN refresh(es) start past the "
                f"measured envelope (abs level > {_ENV_CAP:g}) — a bootstrap there returns "
                f"GARBAGE SILENTLY (rel_err 1.96e4 at level 50 vs 3.7e-1 at 48).\n  "
                + ", ".join(f"{v}@{l:g}" for v, l in _bad_cut[:6])
                + "  -> lower --max-level (ceiling is REFRESH_ENV_CAP_ABS + level_unit"
                  f" = {_ENV_CAP + cfg.level_unit:g}), or PLAN_HARD_ENV_CAP=0 to allow"
                  " (the 32-bit recipe does; the envelope is then the runtime guard's).")
        raise PlanInfeasible(_msg)

    # P4 — LINEAGE COVERAGE.
    # A branch that fed an over-budget op but was NEVER eligible for a cut is a branch the
    # optimizer is structurally blind to: the flow network wires only the binding operand,
    # so a non-binding input can need refreshing forever and no cut can reach it. Every
    # such var is named. It is a WARNING rather than a
    # refusal because being non-binding is not by itself a defect — the branch may be
    # perfectly safe — but an unexplained absence of refreshes must never look clean.
    # (min-cut-specific: baseline placers do not use the binding-operand flow network, so
    # ever_eligible/ever_pressured stay empty there and P4 is skipped by construction.)
    blind = sorted(placer.ever_pressured - placer.ever_eligible
                   - set(placer.placed) - set(g.inputs))
    blind = [v for v in blind if v in g.producer_of]
    if blind:
        worst = sorted(blind, key=lambda v: -(g.max_abs_of(v) or 0.0))[:6]
        if cfg.verbose:
            log.info(f"[plan]  P4: {len(blind)} branch(es) fed an over-budget op but were "
                  f"never cuttable (non-binding operands the min-cut cannot see). "
                  f"Largest |m|: "
                  + ", ".join(f"{v}({g.max_abs_of(v):.3g})" for v in worst
                              if g.max_abs_of(v) is not None))
    result_blind = blind

    result = emit.assemble(
        g, sim, refresh, sorted(placer.placed),
        bootstrap_level=cfg.bootstrap_level, max_level=cfg.max_level,
        source_level=cfg.source_level, cache_read_level=cfg.cache_read_level,
        seed_consumed=seed_consumed, err_target=cfg.err_target,
        num_deliberate=erase.n_deliberate_kept, num_nodes_raw=num_nodes_raw,
        heuristic_config=getattr(placer, "heuristic_config", "min_cut"),
        placer_meta=getattr(placer, "meta", None),
        boundary_realize=cfg.boundary_realize,
        realize_anchors=rescale_anchors,
    )
    # Refresh-input absolute levels of the placed sites: what the pricing knobs move.
    # eff = nominal + pending rescale, as _capacity sees it.
    _hist: dict[str, int] = {}
    for _v in placer.placed:
        _c = placer.placed_in_eff.get(_v)          # forced/pinned sites never went through the cut
        if _c is None:
            _c = sim.consumed.get(_v, 0.0) + (cfg.level_unit if sim.deg.get(_v, 1) == 2 else 0)
        _key = str(int(cfg.bootstrap_level + _c))
        _hist[_key] = _hist.get(_key, 0) + 1
    _hist = dict(sorted(_hist.items(), key=lambda kv: int(kv[0])))
    result["summary"]["bts_quality"]["placed_input_level_hist"] = _hist
    if cfg.verbose:
        log.info("[plan] placed_input_level_hist " + " ".join(f"{k}:{c}" for k, c in _hist.items()))
    result["summary"]["bts_quality"]["blind_branches"] = {
        "count": len(result_blind),
        "vars": result_blind[:32],
    }
    # The absolute level the sim PREDICTS each placed refresh starts at, so a run can be
    # audited against it (scripts/utils/bts_level_audit.py). The runtime prints the level it
    # actually met as `[planted_bts] ... in=`; the two disagreeing is how a refresh ends up
    # past the envelope on a plan that looked clean, which no placement policy can catch.
    # (placed_input_level_hist holds the histogram of the same levels.)
    result["summary"]["bts_quality"]["placed_input_levels"] = {
        v: cfg.bootstrap_level + sim.consumed.get(v, 0.0)
           + (cfg.level_unit if sim.deg.get(v, 1) == 2 else 0)
        for v in sorted(placer.placed)
    }
    if _bad_reactive:
        log.warning(
            f"[plan] P3c(reactive): {len(_bad_reactive)} var(s) are predicted past the envelope "
            f"(abs level > {_ENV_CAP:g}) with NO placed or hinted refresh -- the runtime will "
            f"refresh them there and stop with [bts_depth_error]: "
            + ", ".join(f"{v}@{l:g}" for v, l in _bad_reactive[:6])
            + "  -> lower --max-level. (--baseline-depth-cap does NOT cover this class: it "
            "bounds a PLACED refresh's input depth, and these are precisely the vars nothing "
            "places.)")
    result["summary"]["bts_quality"]["env_cap"] = {
        "limit": _ENV_CAP,
        "relaxed": bool(getattr(placer, "env_cap_relaxed", False)),
        "vetoed_hints": sorted(getattr(placer, "vetoed_hints", ()) or ()),
        "placed_past_cap": [v for v, _ in _bad_cut],
        "hints_predicted_past_cap": [v for v, _ in _bad_hint],
        "reactive_past_cap": [v for v, _ in _bad_reactive],
    }
    # Which hints survived dissolution, and why: that residue is exactly the set whose
    # refresh level is still decided at runtime.
    result["summary"]["hints_retained"] = [
        {"var": v, "reason": r,
         "step": next((n.step for n in _hint_nodes(g) if n.output == v), "")}
        for v, r in sorted(hints_retained.items())
    ]
    # The clamp report also rides in the plan file, under summary.bts_quality. Summary
    # keys are additive under the baseline contract (scripts/utils/plan_equiv.py).
    # `cf_clamp_probe_cf` is the one step past the ceiling that defines a clamp and
    # `cf_table_max` the last measured CF: past it the probe prices analytically.
    cf_lo_hi = refresh.table.cf_range()
    result["summary"]["bts_quality"].update({
        "num_at_cf_max": clamp.num_pinned,
        "num_cf_clamped": clamp.num_clamped,
        "cf_clamped_sites": list(clamp.clamped),
        "cf_clamped_missing_target": list(clamp.clamped_missing_target),
        "cf_clamp_probe_cf": clamp.cf_max + 1,
        "cf_table_max": cf_lo_hi[1] if cf_lo_hi else None,
    })
    if cfg.verbose:
        s = result["summary"]
        bq = s["bts_quality"]
        if cfg.dissolve_hints:
            by_reason: dict[str, int] = {}
            for r in hints_retained.values():
                by_reason[r] = by_reason.get(r, 0) + 1
            log.info(f"[plan] hints dissolved {n_hints_total - len(hints_retained)}"
                  f"/{n_hints_total}; retained {len(hints_retained)}"
                  + (f" ({', '.join(f'{k} {c}' for k, c in sorted(by_reason.items()))})"
                     if by_reason else ""))
        log.info(f"[plan] placements={s['num_placements']} hints={s['num_hint_bootstraps']}"
              f"/{s['num_hints_total']} deliberate={s['num_deliberate_bootstrap_nodes']} "
              f"total={s['total_bootstraps']}")
        rng = refresh.table.cf_range()
        if rng is not None:
            used = [int(k) for k in bq['cf_histogram']]
            outside = sorted(c for c in used if c < rng[0] or c > rng[1])
            if outside:
                log.info(f"[plan]  CF {outside} lies OUTSIDE the measured table (cf {rng[0]}..{rng[1]}) "
                      f"— those sites are priced by the ANALYTIC fallback, not measurement. "
                      f"Extend the table with BTS_ACC_CFS before trusting them.")
        log.info(f"[plan] CF {bq['cf_histogram']}  out-levels {bq['out_level_histogram']}  "
              f"pred rel_err median={bq['predicted_rel_err']['median']:.3g} "
              f"p90={bq['predicted_rel_err']['p90']:.3g}"
              + (f"   {bq['num_sites_missing_target']} MISS TARGET"
                 if bq['num_sites_missing_target'] else ""))
    return result


def _block_index(d: Path) -> int:
    return int(d.name.rsplit("_", 1)[1])


def plan_graph_dir(graph_dir: Path | str, out_dir: Path | str,
                   cfg: PlanConfig) -> list[dict]:
    graph_dir, out_dir = Path(graph_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = _load_table(cfg)
    if cfg.verbose:
        log.info(f"[plan] out={out_dir} bts_level={cfg.bootstrap_level} "
              f"max_level={cfg.max_level} unit={cfg.level_unit} "
              f"table_measured={table.measured} err_target={cfg.err_target:g}")

    summaries: list[dict] = []
    entry_level: int | None = cfg.first_entry_level
    entry_deg: int | None = cfg.first_entry_deg
    blocks = sorted((d for d in graph_dir.glob("block_*") if d.is_dir()),
                    key=_block_index)
    prev: dict | None = None
    for bd in blocks:
        gf = bd / "graph.json"
        if not gf.is_file():
            log.info(f"[plan] skipping {bd} (no graph.json)")
            continue
        if cfg.verbose:
            entry_desc = (f"entry={entry_level}/d{entry_deg}"
                          if entry_level is not None else "capture-derived")
            log.info(f"[plan] [{bd.name}] ({entry_desc})")
        # the in-memory diagnostics ride on `summaries` (never on the plan file)
        diag = PlanDiagnostics()
        try:
            result = plan_block(gf, cfg, entry_level=entry_level, entry_deg=entry_deg,
                                table=table, diagnostics=diag, final_block=bd == blocks[-1])
        except PlanInfeasible:
            if prev is None or prev.get("retried") or not prev.get("exit_var"):
                raise
            log.warning(f"[plan] [{bd.name}] infeasible at entry {entry_level}/d{entry_deg} — "
                  f"retrying {prev['name']} with a terminal-exit refresh on "
                  f"{prev['exit_var']}")
            pdiag = PlanDiagnostics()
            pres = plan_block(prev["gf"], cfg, entry_level=prev["entry_level"],
                              entry_deg=prev["entry_deg"], table=table,
                              force_place={prev["exit_var"]}, diagnostics=pdiag)
            if (st := _capture_stamp(Path(prev["gf"]).parent)) is not None:
                pres[CONTRACT_KEY] = st
            prev["out_file"].write_text(json.dumps(pres, indent=1), encoding="utf-8")
            ps = pres["summary"]
            summaries[-1] = {"block": prev["name"], "output": str(prev["out_file"]),
                             **ps, "cf_clamp": pdiag.cf_clamp}
            prev["retried"] = True
            entry_level = ps.get("exit_level")
            entry_deg = (ps.get("exit_deg") or 1) if entry_level is not None else None
            if cfg.verbose:
                log.info(f"[plan] [{bd.name}] (retry entry={entry_level}/d{entry_deg})")
            diag = PlanDiagnostics()
            result = plan_block(gf, cfg, entry_level=entry_level, entry_deg=entry_deg,
                                table=table, diagnostics=diag, final_block=bd == blocks[-1])
        out_file = out_dir / f"{bd.name}_placement.json"
        if (st := _capture_stamp(bd)) is not None:   # plan/env contract (perseus.plan.contract)
            result[CONTRACT_KEY] = st
        out_file.write_text(json.dumps(result, indent=1), encoding="utf-8")
        s = result["summary"]
        summaries.append({"block": bd.name, "output": str(out_file), **s,
                          "cf_clamp": diag.cf_clamp})
        prev = {"name": bd.name, "gf": gf, "out_file": out_file,
                "entry_level": entry_level, "entry_deg": entry_deg,
                "exit_var": s.get("exit_var"), "retried": False}
        entry_level = s.get("exit_level")
        entry_deg = (s.get("exit_deg") or 1) if entry_level is not None else None
    return summaries
