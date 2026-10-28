from __future__ import annotations

import dataclasses
import os
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

from .. import btserr
from .. import contract
from .ir import ADD_FAMILY, MULT_FAMILY, Graph, parse_lvl_suffix
from .place import Placer, PlanInfeasible, erase_reactive_bootstraps
from .refresh import CfClampReport, RefreshPlanner
from .sim import Budget, PlannerShadow, SimResult, step_bts_offsets

log = logging.getLogger(__name__)


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


KS_OPS = frozenset({"rotate", "conjugate", "rotate_sum"})


def _ks_candidates(g: Graph, sim: SimResult, hint_outs, placed) -> set:
    """Deg-2 values a key switch reads (rotations, conjugations; a rotate-and-sum ladder is recorded as its rotates).
    Realizing the pending rescale first runs every key switch on one composite level (2 limbs) fewer, sometimes one
    gadget digit fewer; the rescale itself happens anyway at the next product, so effective levels are unchanged."""
    out = set()
    for n in g.nodes:
        if n.op not in KS_OPS or n.is_deliberate_bts or n.hint_level is not None:
            continue
        for v in n.cipher_inputs:
            if sim.deg.get(v, 1) != 2 or v in placed or v in hint_outs or v in g.inputs:
                continue
            if g.producer_of.get(v) is None:
                continue
            out.add(v)
    return out


def _shared_candidates(g: Graph, sim: SimResult, hint_outs, placed) -> set:
    """Deg-2 values read only by products and key switches, at least two of them: every product rescales its own
    copy of a deg-2 operand (FIDESlib mult / multPt / multScalar / square) and a key switch carries its 2 extra limbs,
    so realizing the pending rescale once serves them all (the generic form of IMPL_REALIZE_SHARED). Effective levels
    are unchanged; the planner's own simulation tracks the degrees, so the plan's degree pins stay exact."""
    consumers = {}
    for n in g.nodes:
        if n.is_deliberate_bts or n.hint_level is not None:
            continue
        for v in set(n.cipher_inputs):
            consumers.setdefault(v, []).append(n)
    out = set()
    for v, cs in consumers.items():
        if sim.deg.get(v, 1) != 2 or v in placed or v in hint_outs or v in g.inputs:
            continue
        if g.producer_of.get(v) is None or len(cs) < 2:
            continue
        if all(n.op in MULT_FAMILY or n.op in KS_OPS for n in cs):
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
    slim_cubic: float = 0.0   # StC-first order: price the sine cubic (btserr.SitePolicy.slim_cubic)
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
    # Realize anchors on the deg-2 inputs of key switches (_ks_candidates), kept when the re-cut plan needs no more
    # bootstraps (PLAN_KS_REALIZE): cheaper rotations and conjugations, same effective levels.
    ks_realize: bool = False
    # Realize anchors on deg-2 values read by two or more products / key switches (_shared_candidates), kept when the
    # re-cut plan needs no more bootstraps (PLAN_SHARED_REALIZE): one rescale instead of one per consumer.
    shared_realize: bool = False
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
    # Level-aware ModRaise (opt-in, --raise-drop-max K). After the cut, every placed site whose
    # lineage reaches its next refresh with slack hands that slack back: its raise stops
    # `drop` composite levels below the chain top (the bootstrap runs on fewer limbs, ~6%
    # per level on the n32 chain) and its landing is deeper by the same amount. Assigned on
    # the FIXED placement from the sim's own levels, then re-simulated so the plan's expected
    # levels and the P3/P3c checks see the deeper landings. Sites whose lineage reaches a
    # block output, a KV-cache write, or a hint (runtime-decided) are left at the full raise.
    raise_drop_max: int = 0
    # Search the drops on the simulation instead (--raise-drop-search): site by site in program order, the largest
    # drop whose re-simulation keeps every hint decision, every block output and cache write, every refresh input
    # inside its envelope (or no deeper than before) and every level within max_level. A deeper landing absorbed by a
    # max() downstream costs nothing there, which the walk cannot see.
    raise_drop_search: bool = False
    # Envelope rule for the slack: "effective" (planner's P3c convention, refresh input + pending
    # rescale <= cap) or "nominal" (the runtime [bts_depth_error] guard: refresh input <= cap, the
    # level the shipped plans already run their deepest refreshes at).
    raise_drop_env_rule: str = "effective"
    # Deepest landing a raise-dropped site may take (absolute level). The runtime's reactive
    # safety net fires at AUTO_BTS_LEVEL (46 on the n32 recipe) for any value it did not expect
    # there, so landings must stay strictly below it: default AUTO_BTS_LEVEL - level_unit = 44.
    raise_drop_landing_max: int = 44
    # Routes (sparse slot counts, 0 = dense) a raise drop may be assigned to. The 1-slot route (SPRU on n32) builds a
    # variant for any drop; it is not in the default, which the shipped plans were made with.
    raise_drop_routes: tuple[int, ...] = (0, 512)
    # The drops the runtime builds plaintext variants for (FIDESLIB_BTS_RAISE_DROPS); a site's
    # drop is rounded down to the largest of them (None: any drop up to raise_drop_max). Each
    # variant is a full CtS/StC plaintext set per route, so the runtime cannot afford them all; the 1-slot route has
    # no plaintext set and takes any drop.
    raise_drop_set: tuple[int, ...] | None = None
    # The drop each route already runs at by default (the runtime's FIDESLIB_BTS_RAISE_DROP, e.g. 512:1): its
    # landing (PLAN_SPARSE_BTS_OUT) includes it, and the plan's raise_drop is absolute, so a site's assigned drop adds
    # to it and rounds within raise_drop_set as that total (PLAN_RAISE_DROP_BASE).
    raise_drop_base: tuple[tuple[int, int], ...] = ()
    # Dense refreshes whose slots the capture tagged real run one EvalMod chain (plan 'real_route'; the runtime
    # builds the real StC stage under FIDESLIB_BTS_REAL).
    real_route: bool = False
    first_entry_level: int | None = None
    first_entry_deg: int | None = None
    #: (level, deg) every K/V cache read of a decode block arrives at, when it is not the capture's: a decode
    #: after a prefill hand-off reads the handed states, which sit deeper than the decode's own cache writes
    kv_entry_level: int | None = None
    kv_entry_deg: int = 2
    #: a plan dir whose placements every block keeps (forced into the cut, which adds only what the new seeds
    #: still need): a plan re-derived for other entry levels with the refresh sites it was validated with
    keep_placements: str | None = None
    #: with keep_placements: the kept file's decisions are the plan's (PLAN_KEEP_PLACEMENTS_EXACT): its placements, its
    #: fired hints (every hint pinned as the file has it; no PLAN_HINT_ENV_VETO, no dissolution) and its realize anchors
    #: (no realize-anchor trials, no prune); the cut may add nothing. The block stops when a kept site is not in the
    #: graph or the plan does not end with exactly the kept refreshes and rescale_after (another capture or recipe).
    #: With PLAN_RAISE_SEARCH it is a drops-only post-pass over a finished plan.
    keep_placements_exact: bool = False
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
    #: a dense refresh whose input is past the StC-first entry runs the shipped order at the plan's CF: give it that
    #: order's CF (PLAN_FALLBACK_CF; off: the StC-first choice, sized on slot magnitudes, as every shipped plan has)
    fallback_cf: bool = False
    #: a refresh of a complex pack whose small imaginary part a conjugation splits off downstream gets the lowest CF its
    #: band allows, where its route's EvalMod sees the coefficients: its noise is absolute, the same on both parts
    #: (PLAN_PACK_CF; emit.split_packs)
    pack_cf: bool = False
    #: placer "ilp": one optimum, the same on every machine and solver build (IlpPlacer.tiebreak, PLAN_ILP_TIEBREAK)
    ilp_tiebreak: bool = False
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
    additive under the baseline contract (perseus.plan.equiv) — but the typed
    object with the site tuples is only available here.
    """
    cf_clamp: CfClampReport | None = None


def _prune_redundant(placer, sim: SimResult, keep=()) -> tuple[SimResult, int]:
    """Drop redundant refreshes from `placer.placed` in name order (see PlanConfig.prune)."""
    from .place import refresh_env_cap
    g = placer.g

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
            if lvl > refresh_env_cap(route=placer.refresh.spec(v).route or 0) - placer.bootstrap_level + 1e-9:
                k += 1
        for n in g.nodes:
            if n.hint_level is not None and s.hint_fired.get(n.idx) and n.cipher_inputs:
                env_n = refresh_env_cap(n.step, route=placer.refresh.route_for(n.output) or 0) - placer.bootstrap_level
                if s.consumed.get(n.cipher_inputs[0], 0.0) > env_n + 1e-9:
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


def _bts_ms(cfg: PlanConfig, table: btserr.AccuracyTable) -> dict[int, float]:
    """Per-route refresh latency (ms, keyed by slot count, 0 = dense) the "ms" pricings read: the recipe's, unless it
    left the default and the accuracy table carries the runtime's measured costs (the dense route is the StC-first
    order when the recipe prices it)."""
    ms = dict(cfg.bts_ms)
    if cfg.bts_ms != PlanConfig.bts_ms or not table.bts_ms:
        return ms
    t = table.bts_ms
    dense = t.get("stcf") if cfg.slim_cubic > 0 else t.get("dense")
    out = {0: dense or ms.get(0, 1.0)}
    out.update({int(k): v for k, v in t.items() if k.isdigit()})
    return out


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


def _assign_raise_drops(g: Graph, sim: SimResult, placed: set, cfg: PlanConfig,
                        refresh=None) -> tuple[dict[str, int], dict[str, int]]:
    """Per placed site, the composite levels its raise can stop short of the chain top.

    Walks the refreshed value forward to the next refresh on every path. slack = the room
    the deepest reached refresh input leaves under max_level (nominal) and under the refresh
    envelope (effective, pending rescale included). Paths ending at a block output / cache
    write (nothing refreshes them inside the block), or at a hint (its firing level is
    decided at runtime), disqualify the site. Deliberate (fold) bootstraps count as refreshes.
    Returns ({var: drop}, {reason: count}).
    """
    import collections
    from .place import refresh_env_cap as _refresh_env_cap
    unit = float(cfg.level_unit) or 1.0
    L = float(cfg.max_level - cfg.bootstrap_level)
    env_rel = _refresh_env_cap() - cfg.bootstrap_level
    consumers: dict[str, list] = collections.defaultdict(list)
    for n in g.nodes:
        for i in n.inputs:
            consumers[i].append(n)
    drops: dict[str, int] = {}
    skipped: dict[str, int] = collections.Counter()
    base = dict(cfg.raise_drop_base)
    for p in sorted(placed):
        stack, seen, ends, sink, hint = [p], {p}, [], False, False
        while stack:
            x = stack.pop()
            cs = consumers.get(x, [])
            if not cs:
                sink = True
                continue
            for n in cs:
                if n.hint_level is not None:
                    hint = True
                    continue
                if getattr(n, "is_deliberate_bts", False):
                    ends.append(x)          # the fold's INPUT is what meets the envelope
                    continue
                y = n.output
                if not y:
                    sink = True             # cache write / store: leaves the block at this level
                    continue
                if y in seen:
                    continue
                seen.add(y)
                if y in placed:
                    ends.append(y)
                    continue
                stack.append(y)
        if sink:
            skipped["sink"] += 1
            continue
        if hint:
            skipped["hint_downstream"] += 1
            continue
        if not ends:
            skipped["no_consumer"] += 1
            continue
        # A placed site's `consumed` is its LANDING (the sim overwrites the output with the
        # refreshed level); the level its refresh STARTS at is node_out of the producing node.
        # The pending-rescale degree at that point is not kept, so assume deg 2 (one unit).
        def in_level(c: str) -> float:
            if c in placed and c in g.producer_of:
                return sim.node_out.get(g.producer_of[c].idx, sim.consumed.get(c, 0.0))
            return sim.consumed.get(c, 0.0)
        # each downstream refresh meets its own route's envelope (PLAN_REFRESH_ENV_CAP_ROUTES); a deliberate one is
        # dense
        def env_of(c: str) -> float:
            if refresh is None:
                return env_rel
            return _refresh_env_cap(route=int(refresh.spec(c).route or 0) if c in placed else 0) - cfg.bootstrap_level
        rule = float(cfg.level_unit) if cfg.raise_drop_env_rule != "nominal" else 0.0
        slack = min(min(L - in_level(c), env_of(c) - in_level(c) - rule) for c in ends)
        k = min(cfg.raise_drop_max, int(math.floor(slack / unit + 1e-9)))
        if refresh is not None and int(refresh.spec(p).route or 0) not in cfg.raise_drop_routes:
            skipped["route"] += 1
            continue
        _fb = refresh.shadow.dense_fallback() if refresh is not None else None
        if (_fb is not None and not refresh.spec(p).route
                and cfg.bootstrap_level + sim.input_eff(g, p, g.level_unit) > _fb[0] + 1e-9):
            skipped["shipped_order"] += 1       # past the StC-first entry: the shipped order runs it
            continue
        if refresh is not None and cfg.raise_drop_landing_max is not None:
            s = refresh.spec(p)
            restore = float(g.level_unit) if s.prescale is not None else 0.0
            landing = cfg.bootstrap_level + s.out_consumed - restore
            k = min(k, int(math.floor((cfg.raise_drop_landing_max - landing) / unit + 1e-9)))
        if cfg.raise_drop_set and (refresh is None or int(refresh.spec(p).route or 0) != 1):
            b = base.get(int(refresh.spec(p).route or 0), 0) if refresh is not None else 0
            k = max((a - b for a in cfg.raise_drop_set if b < a <= b + k), default=0)
        if k <= 0:
            skipped["no_slack"] += 1
            continue
        drops[p] = k
    return drops, dict(skipped)


def _search_raise_drops(g: Graph, sim: SimResult, placer, refresh, cfg: PlanConfig
                        ) -> tuple[dict[str, int], dict[str, int], SimResult]:
    """`PlanConfig.raise_drop_search`: per refresh of a raise route (placed sites and fired hints), in program order,
    the largest drop whose re-simulation leaves every hint decision, every block output and cache-write operand, and
    every other refresh's landing as they were, keeps every refresh input inside its envelope (or no deeper than
    before) and every level within max_level. The drops found are kept in `refresh.site_out_levels`. Returns (drops,
    skipped, final sim)."""
    import collections
    from .place import refresh_env_cap as _refresh_env_cap
    unit = int(cfg.level_unit) or 1
    consumers: dict[str, int] = collections.Counter()
    stores: set[str] = set()
    for n in g.nodes:
        for i in n.inputs:
            consumers[i] += 1
            if not n.output:
                stores.add(i)
    pinned = sorted(stores | {v for v in sim.consumed if not consumers[v]})

    def inputs(s: SimResult) -> dict[str, tuple[float, float]]:
        """(absolute effective input level, envelope) of every refresh: placed sites, fired hints and the code's
        deliberate refreshes (dense: past the StC-first entry the runtime takes the shipped order, which lands
        elsewhere)."""
        out = {}
        for v in placer.placed:
            p = g.producer_of.get(v)
            out[v] = (cfg.bootstrap_level + s.input_eff(g, v, unit),
                      _refresh_env_cap(route=int(refresh.spec(v).route or 0), step=getattr(p, "step", None)))
        for n in g.nodes:
            if not n.cipher_inputs:
                continue
            if n.hint_level is not None and s.hint_fired.get(n.idx):
                out[f"hint:{n.idx}"] = (cfg.bootstrap_level + s.effective(n.cipher_inputs[0], unit),
                                        _refresh_env_cap(route=0, step=n.step))
            elif n.is_deliberate_bts:
                out[f"delib:{n.idx}"] = (cfg.bootstrap_level + max(s.effective(x, unit) for x in n.cipher_inputs),
                                         _refresh_env_cap(route=0, step=n.step))
        return out

    def past_cap(s: SimResult) -> dict[str, float]:
        """Effective level of every value past its step's envelope (P3c's reactive class: the runtime would refresh
        it there)."""
        out = {}
        for v, c in s.consumed.items():
            p = g.producer_of.get(v)
            lv = cfg.bootstrap_level + s.effective(v, unit)
            if lv > _refresh_env_cap(step=getattr(p, "step", None)) + 1e-9:
                out[v] = lv
        return out

    fb = placer._dense_fallback()

    def past_entry(s: SimResult, v: str) -> bool:
        return fb is not None and cfg.bootstrap_level + s.input_eff(g, v, unit) > fb[0] + 1e-9

    base_in, base_past = inputs(sim), past_cap(sim)
    drops: dict[str, int] = {}
    skipped: dict[str, int] = collections.Counter()
    fired = {n.output for n in g.nodes if n.hint_level is not None and n.output and sim.hint_fired.get(n.idx)}
    sites = set(placer.placed) | fired
    order = sorted(sites, key=lambda v: g.producer_of[v].idx if v in g.producer_of else -1)
    for p in order:
        s = refresh.spec(p)
        route = int(s.route or 0)
        if route not in cfg.raise_drop_routes:
            skipped["route"] += 1
            continue
        if not route and past_entry(sim, p):
            # its input passed the StC-first entry: the shipped order runs it, at its own landing
            skipped["shipped_order"] += 1
            continue
        restore = float(g.level_unit) if s.prescale is not None else 0.0
        base_abs = cfg.bootstrap_level + s.out_consumed - restore
        k_max = cfg.raise_drop_max
        if cfg.raise_drop_landing_max is not None:
            k_max = min(k_max, int(math.floor((cfg.raise_drop_landing_max - base_abs) / unit + 1e-9)))
        saved = refresh.site_out_levels.get(p)
        found = 0
        for k in range(k_max, 0, -1):
            refresh.site_out_levels[p] = int(round(base_abs + k * unit))
            refresh._cache.pop(p, None)
            # the plan's drop is absolute (the route's default and any step landing floor under base_abs included):
            # it must be a built variant
            if route != 1 and cfg.raise_drop_set and refresh.spec(p).raise_drop not in cfg.raise_drop_set:
                continue
            s2 = placer._sim()
            in2 = inputs(s2)
            ok = (s2.hint_fired == sim.hint_fired
                  and all(s2.consumed.get(v) == sim.consumed.get(v) and s2.deg.get(v) == sim.deg.get(v) for v in pinned)
                  and all(cfg.bootstrap_level + math.ceil(c) <= cfg.max_level for c in s2.consumed.values())
                  and all(l <= max(c, base_in[key][0]) + 1e-9 for key, (l, c) in in2.items())
                  and all(l <= base_past.get(v, 0.0) + 1e-9 for v, l in past_cap(s2).items() if v not in sites)
                  and all(s2.consumed.get(v) == sim.consumed.get(v) for v in sites if v != p)
                  and set(i for i, _ in s2.over_budget) <= set(i for i, _ in sim.over_budget))
            if ok:
                found, sim, base_in = k, s2, in2
                break
        if found:
            drops[p] = found
        else:
            if saved is None:
                refresh.site_out_levels.pop(p, None)
            else:
                refresh.site_out_levels[p] = saved
            refresh._cache.pop(p, None)
            skipped["no_slack"] += 1
    return drops, dict(skipped), placer._sim()


def recipe_plan_config(cfg: PlanConfig):
    """The planner recipe as the typed `_pc.PlanConfig`: the PLAN_* knobs of this process's environment (the
    planner's recipe knobs stay CLI/env for now, docs/ENV_INVENTORY.md `plan.*`) with the CLI-resolved values the
    planner Shadow is built from (bootstrap level, measured sparse landings, CF ceiling) taking precedence."""
    from perseus import _pc
    names = {f["env"] for f in _pc.PlanConfig.fields()}
    env = {k: v for k, v in os.environ.items() if k in names}
    env["BTS_LEVEL"] = str(cfg.bootstrap_level)
    env["PLAN_CF_MAX"] = str(cfg.cf_max)
    spec = ",".join(f"{s}:{lv}" for s, lv in cfg.sparse_out_levels)
    if spec:
        env["PLAN_SPARSE_BTS_OUT"] = spec
    else:
        env.pop("PLAN_SPARSE_BTS_OUT", None)
    return _pc.PlanConfig.from_legacy_env(env)


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

    kept_sites: set[str] = set()
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

    # With an `entry_level`, the unnamed graph inputs other than the block input are K/V cache reads: they keep the
    # captured level, at degree 2
    entry_var = next((v for n in g.nodes for v in n.inputs if v in g.inputs), None)

    def is_cache_side_input(v: str) -> bool:
        return (entry_level is not None and v != entry_var and v.startswith("anon_")
                and v in input_level_of)

    def is_kv_entry(v: str) -> bool:
        return cfg.kv_entry_level is not None and v in g.inputs and v != entry_var and v.startswith("anon_")

    def seed_consumed(v: str) -> float:
        if is_kv_entry(v):
            return cfg.kv_entry_level - cfg.bootstrap_level
        exp = parse_lvl_suffix(v)
        if "cache." in v:
            lv = exp if exp is not None else cfg.cache_read_level
            return lv - cfg.bootstrap_level
        if is_cache_side_input(v):
            return input_level_of[v] - cfg.bootstrap_level
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
        if is_kv_entry(v):
            return cfg.kv_entry_deg
        if "cache." in v or is_cache_side_input(v):
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
        slim_cubic=cfg.slim_cubic,
        miss_penalty=cfg.miss_penalty,
        prescale_bits_max=cfg.prescale_bits_max,
    )
    block_name = Path(graph_file).parent.name
    shadow = PlannerShadow(cfg.bootstrap_level, g.level_unit, recipe_plan_config(cfg))
    if shadow.dense_fallback() is not None:
        # a v1 capture records no route: a code-owned refresh landing where the dense StC-first route lands before
        # its realize is that route, and past the StC-first entry the compile runs it in the shipped order
        # (`route_for_landing`), so the entry binds it as it binds a v2 StC-first refresh
        raw = (float(cfg.bootstrap_level - g.level_unit), 2)
        marked = [dataclasses.replace(n, stc_first=True) if (n.is_deliberate_bts and not n.stc_first and not n.external
                                                             and (n.output_level, n.output_deg) == raw) else n
                  for n in g.nodes]
        if any(a is not b for a, b in zip(marked, g.nodes)):
            g = g.replace_nodes(marked)
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
        shadow=shadow,
        real_route=cfg.real_route,
        raise_drop_base=dict(cfg.raise_drop_base),
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
    keep_exact = bool(cfg.keep_placements and cfg.keep_placements_exact)
    kept_doc: dict | None = None              # PLAN_KEEP_PLACEMENTS_EXACT: the kept plan
    kept_hints: set[str] | None = None        # ... its fired hints
    kept_anchors: set[str] = set()            # ... and its realize anchors
    if keep_exact and not exact:
        raise ValueError("keep_placements_exact needs placer min_cut or ilp")
    if keep_exact:
        # the kept plan's realize anchors and refreshes as they are: no realize-anchor trials (another trial path can
        # end on other anchors at the same count), no prune
        cfg = dataclasses.replace(cfg, rescale_opt=False, shared_realize=False, ks_realize=False, prune=False)
    if cfg.dissolve_hints and exact and not keep_exact:
        dissolve_pins, hints_retained = _dissolve_hints(g, cfg, refresh)

    # stage 4 — ours, or a baseline placer (same seam, different site selection)
    _fb = refresh.shadow.dense_fallback()
    budget = Budget(L=float(cfg.max_level - cfg.bootstrap_level), unit=g.level_unit,
                    stc_first_entry=_fb[0] if _fb is not None else None,
                    stc_first_reroute=bool(refresh.shadow.stc_first_reroute))
    placer_kwargs = dict(
        g=g, refresh=refresh, budget=budget,
        bootstrap_level=float(cfg.bootstrap_level),
        seed_consumed=seed_consumed, seed_deg=seed_deg,
        forbid_steps=cfg.forbid_steps,
        err_target=cfg.err_target, miss_penalty=cfg.miss_penalty,
        quality_weight=cfg.quality_weight, verbose=cfg.verbose,
        depth_weight=cfg.depth_weight, level_weight=cfg.level_weight,
        depth_form=cfg.depth_form, depth_a=cfg.depth_a, depth_b=cfg.depth_b,
        bts_ms=_bts_ms(cfg, refresh.table), ms_discount=cfg.ms_discount,
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
                            cap_exit=bool(final_block and cfg.ilp_cap_exit), tiebreak=cfg.ilp_tiebreak)
    if exact:
        # force_place: pre-seeded placements (the terminal-exit refresh retry in
        # plan_graph_dir, and the magnitude keep-anchors). The cut still adds
        # whatever else it needs; every forced site goes through P2.
        forced = set(force_place or ())
        if cfg.keep_placements:
            # kept sites also survive the prune below
            kept = Path(cfg.keep_placements) / f"{block_name}_placement.json"
            if not kept.is_file():
                raise FileNotFoundError(f"keep_placements: no {kept}")
            doc = json.loads(kept.read_text(encoding="utf-8"))
            kept_sites = {p["target_var"] for p in doc.get("placements", []) if p.get("target_var") in g.producer_of}
            if keep_exact:
                kept_doc, kept_hints = doc, set(doc.get("hint_fire", []))
                quality = (kept_sites | kept_hints) if refresh.out_deg == 1 else set()
                exit_v = doc.get("summary", {}).get("exit_var")
                kept_anchors = (set(doc.get("rescale_after", [])) - quality
                                - ({exit_v} if cfg.boundary_realize and exit_v else set()))
                absent = sorted(({p["target_var"] for p in doc.get("placements", []) if p.get("target_var")}
                                 | kept_hints | kept_anchors) - set(g.producer_of))
                if absent:
                    raise RuntimeError(f"keep_placements_exact: {len(absent)} kept site(s) of {kept} are not in "
                                       f"{graph_file}: {', '.join(map(str, absent[:6]))} (another capture?)")
                # every hint as the kept plan has it, over any other pin (veto steps, a caller's)
                hint_force = {n.output: n.output in kept_hints for n in g.nodes
                              if n.hint_level is not None and n.output}
            forced |= kept_sites
            if cfg.verbose:
                log.info(f"[plan] keep_placements: {len(forced)} site(s) kept from {kept}")
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
            placer = PlacerCls(**placer_kwargs, placed=set(forced), hint_force=hint_force,
                               **({"realized": set(kept_anchors), "hint_veto": False} if keep_exact else {}))
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
    # the realize-anchor trials re-run the cut: kept and forced sites stay placed in them too (the terminal-exit refresh
    # of the block-loop retry was dropped by those trials, so the retry re-planned the same exit)
    _kept_all = set(kept_sites) | (set(force_place or ()) if exact else set())
    _kept = {"placed": _kept_all} if _kept_all else {}
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
                           hint_force=hint_force, **_kept)
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

    if cfg.shared_realize and exact:
        hint_outs = {n.output for n in g.nodes if n.hint_level is not None and n.output}
        best_sr = len(placer.placed) + sum(1 for f in sim.hint_fired.values() if f)
        base_sr, n_sr = best_sr, 0
        for _ in range(6):
            cands = _shared_candidates(g, sim, hint_outs, placer.placed) - rescale_anchors
            if not cands:
                break
            trial = rescale_anchors | cands
            p2 = PlacerCls(**{**placer_kwargs, "verbose": False}, realized=set(trial), hint_force=hint_force, **_kept)
            try:
                s2 = p2.run()
            except PlanInfeasible:
                break
            c2 = len(p2.placed) + sum(1 for f in s2.hint_fired.values() if f)
            if c2 <= best_sr:
                rescale_anchors, sim, placer, best_sr = trial, s2, p2, c2
                n_sr += len(cands)
            else:
                break
        if cfg.verbose:
            log.info(f"[plan] shared_realize: {n_sr} shared-operand realize anchor(s), bootstraps {base_sr} -> {best_sr}")

    if cfg.ks_realize and exact:
        hint_outs = {n.output for n in g.nodes if n.hint_level is not None and n.output}

        def _count_ks(s: SimResult, p: Placer) -> int:
            return len(p.placed) + sum(1 for f in s.hint_fired.values() if f)

        best_ks = _count_ks(sim, placer)
        base_ks, n_ks = best_ks, 0
        for _ in range(6):               # a ladder realized at its root makes its later steps deg-1: re-derive
            cands = _ks_candidates(g, sim, hint_outs, placer.placed) - rescale_anchors
            if not cands:
                break
            trial = rescale_anchors | cands
            p2 = PlacerCls(**{**placer_kwargs, "verbose": False}, realized=set(trial),
                           hint_force=hint_force, **_kept)
            try:
                s2 = p2.run()
            except PlanInfeasible:
                break
            c2 = _count_ks(s2, p2)
            if c2 <= best_ks:
                rescale_anchors, sim, placer, best_ks = trial, s2, p2, c2
                n_ks += len(cands)
            else:
                break
        if cfg.verbose:
            log.info(f"[plan] ks_realize: {n_ks} key-switch realize anchor(s), bootstraps {base_ks} -> {best_ks}")

    if cfg.prune:
        # forced sites stay (the exit refresh of the terminal-exit retry is needed by the next block)
        sim, n_pruned = _prune_redundant(placer, sim, tuple(cfg.prune_keep) + tuple(force_place or ())
                                         + tuple(sorted(kept_sites)))
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
    # ---- level-aware ModRaise post-pass (see PlanConfig.raise_drop_max) ----
    raise_hist: dict[int, int] = {}
    raise_skipped: dict[str, int] = {}
    if cfg.raise_drop_max > 0 and exact and cfg.raise_drop_search:
        drops, raise_skipped, sim = _search_raise_drops(g, sim, placer, refresh, cfg)
        for k in drops.values():
            raise_hist[k] = raise_hist.get(k, 0) + 1
        if cfg.verbose:
            log.info(f"[plan] raise_drop (search): {len(drops)} refreshes (placed + fired hints) raise below the "
                     f"chain top; " + " ".join(f"{v}:{k}" for v, k in drops.items())
                     + (f"; skipped {raise_skipped}" if raise_skipped else ""))
    elif cfg.raise_drop_max > 0 and exact:
        drops, raise_skipped = _assign_raise_drops(g, sim, set(placer.placed), cfg, refresh)
        if drops:
            saved_cache = dict(refresh._cache)
            saved_overrides = dict(refresh.site_out_levels)
            for v, k in drops.items():
                s = refresh.spec(v)
                restore = float(g.level_unit) if s.prescale is not None else 0.0
                base_abs = cfg.bootstrap_level + s.out_consumed - restore
                refresh.site_out_levels[v] = int(round(base_abs + k * cfg.level_unit))
                refresh._cache.pop(v, None)
            sim2 = placer._sim()
            bad = [v for v, c in sim2.consumed.items()
                   if cfg.bootstrap_level + math.ceil(c) > cfg.max_level]
            if bad:
                log.warning(f"[plan] raise_drop: re-simulation pushed {len(bad)} var(s) past "
                            f"max_level ({', '.join(bad[:4])}) — keeping the full raise everywhere")
                refresh._cache = saved_cache
                refresh.site_out_levels = saved_overrides
                drops = {}
            else:
                sim = sim2
        for k in drops.values():
            raise_hist[k] = raise_hist.get(k, 0) + 1
        if cfg.verbose:
            log.info(f"[plan] raise_drop: {len(drops)} of {len(placer.placed)} placed sites raise "
                     f"below the chain top; levels->sites "
                     + " ".join(f"{k}:{c}" for k, c in sorted(raise_hist.items()))
                     + (f"; skipped {raise_skipped}" if raise_skipped else ""))

    if kept_hints is not None:
        # PLAN_KEEP_PLACEMENTS_EXACT: the final plan (after the cut, the realize trials, the prune and the raise pass)
        # refreshes exactly where the kept plan does
        fired_now = {g.nodes[i].output for i, f in sim.hint_fired.items() if f and g.nodes[i].output}
        diff = [(what, sorted(a - b)) for what, a, b in (
            ("placements added", set(placer.placed), kept_sites),
            ("placements dropped", kept_sites, set(placer.placed)),
            ("hints fired", fired_now, kept_hints), ("hints no longer fired", kept_hints, fired_now)) if a - b]
        if diff:
            raise RuntimeError(f"keep_placements_exact: {block_name} differs from {cfg.keep_placements} (the kept "
                               f"decisions are infeasible here?): "
                               + "; ".join(f"{what} {', '.join(vs[:6])}" for what, vs in diff))
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

    def _cap_of(v: str) -> float:          # the envelope of the step that produces `v` (PLAN_REFRESH_ENV_CAP_STEPS)
        _pn = g.producer_of.get(v)
        return _refresh_env_cap(getattr(_pn, "step", None) if _pn is not None else None)
    _bad_cut, _bad_hint = [], []
    # A var past the envelope that nothing refreshes is NOT harmless: the runtime meets it
    # there and fires a REACTIVE bootstrap, which is the one class neither the cut nor the
    # hint check covers, and it stops the run with [bts_depth_error]. Collect it.
    _hint_covered = {n.output for n in g.nodes
                     if n.hint_level is not None and sim.hint_fired.get(n.idx) and n.output}
    _bad_reactive = []
    _bad_nom: dict[str, float] = {}      # nominal input level of each entry in _bad_cut (eff = nominal + pending unit)
    _refresh_obj = getattr(placer, "refresh", None)
    for _v, _c in sim.consumed.items():
        if _v in placer.placed:
            # a placed site's `consumed` is its LANDING; the envelope applies to the level its refresh STARTS
            # at. Cut-chosen dense sites only: a deliberate fixed point is the capture's, a hint has its own
            # check below, and the envelope was measured on the dense route (a sparse route tolerates a
            # deeper input; its ceiling stays the runtime guard's, as before).
            _p = g.producer_of.get(_v)
            _sparse = _refresh_obj is not None and bool(_refresh_obj.spec(_v).route)
            if _p is None or _p.is_deliberate_bts or _p.hint_level is not None or _sparse:
                continue
            _nom, _deg = sim.input_state(g, _v)
            _eff = cfg.bootstrap_level + _nom + (cfg.level_unit if _deg == 2 else 0)
            if _eff > _cap_of(_v) + 1e-9:
                _bad_cut.append((_v, _eff))
                _bad_nom[_v] = cfg.bootstrap_level + _nom
            continue
        _eff = cfg.bootstrap_level + _c + (cfg.level_unit if sim.deg.get(_v, 1) == 2 else 0)
        if _eff <= _cap_of(_v):
            continue
        if _v not in _hint_covered:
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
        if _eff > _refresh_env_cap(_n.step):
            _bad_hint.append((_n.output or _in, _eff))
    if _bad_hint and cfg.verbose:
        log.info(f"[plan] P3c(hint): {len(_bad_hint)} hint-fired refresh(es) PREDICTED past the "
              f"envelope (abs level > {_ENV_CAP:g}): "
              + ", ".join(f"{v}@{l:g}" for v, l in _bad_hint[:6])
              + " — WARNING ONLY: this prediction is known to over-fire; the runtime "
                "[bts_depth_error] guard is the authority.")
    if _bad_cut and getattr(placer, "env_cap_relaxed", False):
        # the placer already proved no envelope-respecting placement exists (it tried that
        # pass first and it could not be planned), so these sites are forced, not chosen.
        # PLAN_ENVELOPE_STRICT=1: refuse instead (the block loop then re-plans the block before with a refresh on its
        # exit, the entry this block could not plan from: the tail after a deep last block)
        if os.environ.get("PLAN_ENVELOPE_STRICT", "0") not in ("", "0"):
            raise PlanInfeasible(f"P3c: {len(_bad_cut)} refresh(es) start past the measured envelope with no "
                                 "envelope-respecting placement: " + ", ".join(f"{v}@{l:g}" for v, l in _bad_cut[:6]))
        log.warning(
            f"[plan] P3c: {len(_bad_cut)} refresh(es) start past the measured envelope "
            f"(abs level > {_ENV_CAP:g}) and NO placement avoids them -- a bootstrap there "
            f"returns garbage silently, so the runtime [bts_depth_error] guard is what "
            f"stands between this plan and a wrong answer:\n  "
            + ", ".join(f"{v}@{l:g}" for v, l in _bad_cut[:6]))
    elif _bad_cut:
        # Cut-chosen sites priced inside the envelope that the FINAL sim puts past it: a later cut disarmed
        # a hint upstream, or the raise-drop pass moved a landing below them. The cut never saw them
        # there. The runtime guard reads the nominal level, so a site past the envelope only by its
        # pending rescale runs (the shipped recipes carry such sites: report), while a nominal level past
        # the cap stops the run with [bts_depth_error] (warn). PLAN_DRIFT_RECUT=1 re-cuts both.
        _forced = getattr(placer, "drift_forced", set())
        _stop = [(v, l) for v, l in _bad_cut if _bad_nom.get(v, l) > _cap_of(v) + 1e-9]
        _pend = [(v, l) for v, l in _bad_cut if _bad_nom.get(v, l) <= _cap_of(v) + 1e-9]
        _tail = (f" ({sum(1 for v, _ in _bad_cut if v in _forced)} with no envelope-respecting cover)"
                 if _forced else "") + "; PLAN_DRIFT_RECUT=1 re-cuts them."
        if _stop:
            log.warning(
                f"[plan] P3c(drift): {len(_stop)} cut-chosen refresh(es) priced inside the envelope now "
                f"start past it NOMINALLY (level > {_ENV_CAP:g}) in the final sim -- the runtime "
                "[bts_depth_error] guard stops the run there: "
                + ", ".join(f"{v}@{l:g}" for v, l in _stop[:6]) + _tail)
        if _pend and cfg.verbose:
            log.info(
                f"[plan] P3c(drift): {len(_pend)} cut-chosen refresh(es) priced inside the envelope now "
                f"start past it by their pending rescale (nominal <= {_ENV_CAP:g}, within the runtime "
                "guard) in the final sim: " + ", ".join(f"{v}@{l:g}" for v, l in _pend[:6]) + _tail)

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
        realize_anchors=kept_anchors if keep_exact else rescale_anchors,
        real_route=cfg.real_route,
        fallback_cf=cfg.fallback_cf,
        pack_cf=cfg.pack_cf,
    )
    if kept_doc is not None:
        if sorted(result["rescale_after"]) != sorted(kept_doc.get("rescale_after", [])):
            a, b = set(result["rescale_after"]), set(kept_doc.get("rescale_after", []))
            raise RuntimeError(f"keep_placements_exact: {block_name}'s rescale_after differs from the kept plan's: "
                               f"+{sorted(a - b)[:6]} -{sorted(b - a)[:6]}")
        if cfg.verbose:
            log.info(f"[plan] keep_placements_exact: {len(kept_sites)} placement(s), {len(kept_hints or ())} fired "
                     f"hint(s) and {len(kept_anchors)} realize anchor(s) as kept")
    # Refresh-input absolute levels of the placed sites: what the pricing knobs move.
    # eff = nominal + pending rescale, as _capacity sees it.
    _hist: dict[str, int] = {}
    for _v in placer.placed:
        _c = sim.input_eff(g, _v, cfg.level_unit)   # the final sim, not the level the cut priced
        _key = str(int(cfg.bootstrap_level + _c))
        _hist[_key] = _hist.get(_key, 0) + 1
    _hist = dict(sorted(_hist.items(), key=lambda kv: int(kv[0])))
    result["summary"]["bts_quality"]["placed_input_level_hist"] = _hist
    result["summary"]["bts_quality"]["raise_drop_hist"] = {str(k): c for k, c in sorted(raise_hist.items())}
    result["summary"]["bts_quality"]["raise_drop_skipped"] = raise_skipped
    if cfg.verbose:
        log.info("[plan] placed_input_level_hist " + " ".join(f"{k}:{c}" for k, c in _hist.items()))
    result["summary"]["bts_quality"]["blind_branches"] = {
        "count": len(result_blind),
        "vars": result_blind[:32],
    }
    # The absolute level and pending-rescale degree the sim PREDICTS each placed refresh starts
    # at, so a run can be audited against it (scripts/utils/bts_level_audit.py). The runtime
    # prints what it actually met as `[planted_bts] ... in=<level> ... in_deg=<deg>`; the two
    # disagreeing is how a refresh ends up past the envelope on a plan that looked clean, which
    # no placement policy can catch. (placed_input_level_hist holds the effective levels, level
    # plus one unit when a rescale is pending, which is what the envelope is measured in.)
    result["summary"]["bts_quality"]["placed_input_levels"] = {}
    result["summary"]["bts_quality"]["placed_input_degs"] = {}
    for v in sorted(placer.placed):
        _lvl, _deg = sim.input_state(g, v)
        result["summary"]["bts_quality"]["placed_input_levels"][v] = float(cfg.bootstrap_level + _lvl)
        result["summary"]["bts_quality"]["placed_input_degs"][v] = int(_deg)
    if _bad_reactive:
        log.warning(
            f"[plan] P3c(reactive): {len(_bad_reactive)} var(s) are predicted past the envelope "
            f"(abs level > {_ENV_CAP:g}) with NO placed or hinted refresh -- the runtime will "
            f"refresh them there and stop with [bts_depth_error]: "
            + ", ".join(f"{v}@{l:g}" for v, l in _bad_reactive[:6])
            + "  -> lower --max-level. (--baseline-depth-cap does NOT cover this class: it "
            "bounds a PLACED refresh's input depth, and these are precisely the vars nothing "
            "places.)")
        # PLAN_REACTIVE_STRICT=1: refuse the block, so the chain retries the previous one with an exit refresh
        if os.environ.get("PLAN_REACTIVE_STRICT", "0") not in ("", "0"):
            raise PlanInfeasible(f"P3c(reactive): {len(_bad_reactive)} var(s) past the envelope with no refresh: "
                                 + ", ".join(f"{v}@{l:g}" for v, l in _bad_reactive[:6]))
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
    # keys are additive under the baseline contract (perseus.plan.equiv).
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


def _write_whole(path: Path, doc: dict) -> None:
    """Write a plan file under a temporary name and rename it: a file that exists is complete (PLAN_RESUME)."""
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _resumed(out_file: Path, block_dir: Path) -> dict | None:
    """PLAN_RESUME: the block's plan file already in the out dir, when it parses and carries this capture's stamp."""
    try:
        doc = json.loads(out_file.read_text(encoding="utf-8"))
        doc["summary"]["total_bootstraps"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    want = json.loads(json.dumps(contract.capture_stamp(block_dir)))
    return doc if doc.get(contract.CONTRACT_KEY) == want else None


def _stage_outputs(gf: Path, level_unit: int) -> list[str]:
    """The block's stage outputs as the compile takes them (every final state nothing reads): named, not a deliberate
    refresh or a hint pass-through, read by no other node once the reactive refreshes are erased. One for a decode
    block; several for a packed prefill stage (BERT's LN2 halves)."""
    raw = json.loads(Path(gf).read_text(encoding="utf-8"))
    g, _ = erase_reactive_bootstraps(Graph.from_nodes(raw.get("nodes", []), level_unit=level_unit))
    read: set[str] = set()
    for n in g.nodes:
        read.update(n.cipher_inputs)
    return [n.output for n in g.nodes
            if n.output and not n.is_deliberate_bts and n.hint_level is None and n.output not in read]


def _next_entry(s: dict, gf: Path, cfg: PlanConfig, strict: bool) -> tuple[int | None, int | None]:
    """(level, deg) the next block enters at: the plan's exit, or under PLAN_ENVELOPE_STRICT the deepest of the block's
    stage outputs (the next block reads all of them; the exit var is only the last one)."""
    lvl = s.get("exit_level")
    if lvl is None:
        return None, None
    deg = s.get("exit_deg") or 1
    if not strict:
        return lvl, deg
    unit = cfg.level_unit
    levels, degs = s.get("final_named_levels") or {}, s.get("final_named_degs") or {}
    for v in _stage_outputs(gf, unit):
        if v in levels:
            lv, dv = int(levels[v]), int(degs.get(v, 1))
            if lv + (unit if dv == 2 else 0) > lvl + (unit if deg == 2 else 0):
                lvl, deg = lv, dv
    return lvl, deg


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
    # PLAN_RESUME=1: a block whose plan file is already in out_dir (written whole, with this capture's stamp) is taken
    # as planned and its exit enters the next block, so a preempted run restarts at the block it was planning; the
    # plans are the chain's (each block's depends on its graph, the config and its entry alone)
    resume = os.environ.get("PLAN_RESUME", "0") not in ("", "0")
    # PLAN_ENVELOPE_STRICT=1: a block enters at the deepest of the previous block's stage outputs, and the terminal-exit
    # retry refreshes every stage output (a packed stage hands several to the next block, not only its exit var)
    strict = os.environ.get("PLAN_ENVELOPE_STRICT", "0") not in ("", "0")
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
        out_file = out_dir / f"{bd.name}_placement.json"
        done = _resumed(out_file, bd) if resume else None
        if done is not None:
            log.info(f"[plan] [{bd.name}] resumed: {out_file}")
            s = done["summary"]
            summaries.append({"block": bd.name, "output": str(out_file), **s, "cf_clamp": None})
            prev = {"name": bd.name, "gf": gf, "out_file": out_file,
                    "entry_level": entry_level, "entry_deg": entry_deg,
                    "exit_var": s.get("exit_var"), "retried": False}
            entry_level, entry_deg = _next_entry(s, gf, cfg, strict)
            continue
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
            exits = {prev["exit_var"]}
            if strict:
                exits |= set(_stage_outputs(prev["gf"], cfg.level_unit))
                if cfg.verbose and len(exits) > 1:
                    log.info(f"[plan] [{prev['name']}] terminal-exit refreshes on its {len(exits)} stage outputs: "
                             + ", ".join(sorted(exits)))
            pres = plan_block(prev["gf"], cfg, entry_level=prev["entry_level"],
                              entry_deg=prev["entry_deg"], table=table,
                              force_place=exits, diagnostics=pdiag)
            emit.stamp(pres, contract.capture_stamp(Path(prev["gf"]).parent))
            _write_whole(prev["out_file"], pres)
            ps = pres["summary"]
            summaries[-1] = {"block": prev["name"], "output": str(prev["out_file"]),
                             **ps, "cf_clamp": pdiag.cf_clamp}
            prev["retried"] = True
            entry_level, entry_deg = _next_entry(ps, prev["gf"], cfg, strict)
            if cfg.verbose:
                log.info(f"[plan] [{bd.name}] (retry entry={entry_level}/d{entry_deg})")
            diag = PlanDiagnostics()
            result = plan_block(gf, cfg, entry_level=entry_level, entry_deg=entry_deg,
                                table=table, diagnostics=diag, final_block=bd == blocks[-1])
        emit.stamp(result, contract.capture_stamp(bd))   # the plan/config contract (perseus.plan.contract)
        _write_whole(out_file, result)
        s = result["summary"]
        summaries.append({"block": bd.name, "output": str(out_file), **s,
                          "cf_clamp": diag.cf_clamp})
        prev = {"name": bd.name, "gf": gf, "out_file": out_file,
                "entry_level": entry_level, "entry_deg": entry_deg,
                "exit_var": s.get("exit_var"), "retried": False}
        entry_level, entry_deg = _next_entry(s, gf, cfg, strict)
    return summaries
