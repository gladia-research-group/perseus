"""Orion-style level-DAG shortest-path placer.

Port of the placement algorithm of baahl-nyu/orion (MIT license, (c) 2025 Austin
Ebel; `orion/core/level_dag.py` + `orion/core/auto_bootstrap.py`, the algorithm of
arXiv 2311.03470 Sec. 5.2), adapted to this repo's captured-graph IR.

Upstream algorithm, implemented here structurally 1:1:
  * every network LAYER gets a column of "run at level l" vertices; edges between
    consecutive layers price the bootstrap needed for that level transition, weighted
    by the number of ciphertexts crossing (get_num_input_cts);
  * residual regions: fork = node with >1 successor, join = first node common to the
    paths from every successor to the sink (find_residuals); each region is solved
    innermost-first into an AGGREGATE edge matrix — for every (fork level i, join
    level j) the summed per-branch shortest path, with the traversed bootstrap
    decisions stored on the cell (LevelDAG.__add__ + the `path` bookkeeping);
  * nested regions are spliced into branch paths when encountered
    (build_level_dag_from_path); the full network is one final shortest path and
    bootstraps are marked on producer outputs (mark_bootstrap_locations).

Adaptations (documented in summary.placer_meta.deviations):
  * layer = capture STEP (Orion's nodes are torch modules; our step tags are that
    granularity). State = consumed composite level AFTER the step; a bootstrap of a
    step's outputs costs #crossing vars (ciphertext count), count objective —
    upstream's own comment: the linear-layer latency term does not change bootstrap
    counts, and t_boot is a constant multiplier at fixed l_eff.
  * branch representatives: one path per fork child (upstream dedups all_simple_paths
    by first child; we take the DEEPEST-cost path per child — the binding branch, so
    the depth the region must pay is the depth the model prices; a fewest-hops
    representative routes around the deep approximation chains and leaves them
    unpriced). Branch costs sum over branches, so shared sub-steps double-count in
    the OBJECTIVE but dedupe at PLACEMENT — exactly upstream's __add__ behavior.
  * equal join levels across branches (upstream's rule; conservative-valid here since
    our runtime max-joins).
  * no free level-drop edges: our runtime places only bootstraps, never modswitches.
  * steps whose own depth exceeds l_eff are capped for the solver and left to the
    shared repair (counted in placer_meta.unfixable_steps) — upstream would declare
    the whole program unplannable ("increase your LogQ chain").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx

from ..sim import SimResult
from .base import BaselinePlacer, accum_depths

# a fragment matrix: (entry level, exit level) -> (cost, bootstrapped steps)
Cell = tuple[float, frozenset[str]]
Frag = dict[tuple[int, int], Cell]


def _seq(A: Frag, B: Frag, l0: int, w_conn: float | None,
         conn_site: str | None, reset: int = 0) -> Frag:
    """A then B; between them, optionally bootstrap the connecting producer's
    outputs (cost w_conn). The refresh lands at `reset` — 0 when every live-cut
    member is placeable, else the residual depth of the unplaceable members
    (fold/deliberate outputs, runtime-owned steps), so the model never counts on
    a refresh execution cannot deliver."""
    C: Frag = {}
    by_entry: dict[int, list[tuple[int, Cell]]] = {}
    for (m, j), cell in B.items():
        by_entry.setdefault(m, []).append((j, cell))
    for (i, k), (ca, sa) in A.items():
        for j, (cb, sb) in by_entry.get(k, ()):          # carry through
            c = ca + cb
            cur = C.get((i, j))
            if cur is None or c < cur[0]:
                C[(i, j)] = (c, sa | sb)
        if w_conn is not None:
            for j, (cb, sb) in by_entry.get(reset, ()):  # bootstrap between
                c = ca + w_conn + cb
                cur = C.get((i, j))
                if cur is None or c < cur[0]:
                    C[(i, j)] = (c, sa | sb | {conn_site})
    return C


@dataclass
class OrionPlacer(BaselinePlacer):
    NAME = "orion_leveldag_v1"
    NAME_KEY = "orion"
    # chain hand-off: prefer exits at most this many composite levels below the usable
    # depth l0, so the successor block's first ops stay feasible (0 = uncapped, as
    # upstream, which plans whole programs).
    EXIT_CAP_SLACK = 2

    # ── step graph: one layer per (capture step, composite level) ──────────────────
    # Costs come from the TWIN trajectory (sim0.consumed — deg-aware), not the naive
    # accumulator: the twin charges the pending-rescale effective level, and a model
    # that ignores it underplaces by 1 level per lazy hop. Steps deeper than one
    # composite level split into one sub-layer per level — the module decomposition
    # upstream networks express anyway (activations are Mult/Add module compositions),
    # which gives the solver legitimate interior boundaries inside approximations.
    def _step_graph(self, sim0: SimResult | None = None):
        g = self.g
        unit = max(1, g.level_unit)
        if sim0 is not None:
            dprime = dict(sim0.consumed)
        else:
            dprime = accum_depths(g, self.seed_consumed, self.bootstrap_level)

        # pass 1: per capture step, the entry depth (max over external inputs)
        raw_step_of: dict[str, str] = {}
        raw_first: dict[str, int] = {}
        for n in g.nodes:
            s0 = n.step or "_"
            raw_first.setdefault(s0, n.idx)
            if n.output:
                raw_step_of[n.output] = s0
        step_entry: dict[str, float] = {}
        for n in g.nodes:
            s0 = n.step or "_"
            for v in n.cipher_inputs:
                if raw_step_of.get(v) != s0:
                    step_entry[s0] = max(step_entry.get(s0, 0.0), dprime.get(v, 0.0))

        # pass 2: sub-layer id = composite level within the step
        def sub(n) -> str:
            s0 = n.step or "_"
            rel = dprime.get(n.output, 0.0) - step_entry.get(s0, 0.0)
            k = max(0, int(rel // unit))
            return f"{s0}#{k}" if k > 0 else s0

        step_of_var: dict[str, str] = {}
        first_idx: dict[str, int] = {}
        for n in g.nodes:
            s = sub(n) if n.output else (n.step or "_")
            first_idx.setdefault(s, n.idx)
            if n.output:
                step_of_var[n.output] = s
        # graph inputs consumed by >1 step become pseudo-source layers, so a fork at
        # a block input (branches with no producing step) is still discovered as a
        # region instead of being serialized by the topo walk.
        in_consumers: dict[str, set[str]] = {}
        for n in g.nodes:
            s = sub(n) if n.output else (n.step or "_")
            for v in n.cipher_inputs:
                if v in g.inputs:
                    in_consumers.setdefault(v, set()).add(s)
        pseudo_sources: set[str] = set()
        for k, (v, cons) in enumerate(sorted(in_consumers.items())):
            if len(cons) > 1:
                ps = f"__in__{v}"
                step_of_var[v] = ps
                first_idx[ps] = (-len(in_consumers) + k, 0)
                pseudo_sources.add(ps)
        self._pseudo_sources = pseudo_sources
        # Order sub-layers by (capture position of the step, level index)
        # so the #k split can never invert an edge; plain steps key as (idx, 0).
        def _order_key(item):
            s, fi = item
            if isinstance(fi, tuple):
                return fi
            base, _, k = s.partition("#")
            return (raw_first.get(base, fi), int(k) if k else 0)
        order = [s for s, _ in sorted(first_idx.items(), key=_order_key)]
        pos = {s: i for i, s in enumerate(order)}

        sg = nx.DiGraph()
        sg.add_nodes_from(order)
        max_in: dict[str, float] = {s: 0.0 for s in order}
        max_out: dict[str, float] = {s: 0.0 for s in order}
        last_use: dict[str, int] = {}
        for n in g.nodes:
            s = step_of_var.get(n.output) if n.output else None
            if s is None:
                continue
            max_out[s] = max(max_out[s], dprime.get(n.output, 0.0))
            for v in n.cipher_inputs:
                ps = step_of_var.get(v)
                if ps is None:
                    max_in[s] = max(max_in[s], dprime.get(v, 0.0))
                elif ps != s and pos[ps] < pos[s]:
                    sg.add_edge(ps, s)
                    last_use[v] = max(last_use.get(v, -1), pos[s])
                    max_in[s] = max(max_in[s], dprime.get(v, 0.0))
        # crossing[s] = EVERY ciphertext live across the boundary after layer s, not
        # just s's own outputs: a boot arm in the solver resets the composed level to
        # 0, which is only sound if the whole live state is refreshed. Outputs-only
        # placement leaves parallel live vars (residual copies, sibling goldschmidt /
        # newton branches) deep while the model thinks they reset, so the solver
        # under-places at tiny 1-ct sites and repair takes over (measured collapse:
        # 24 chosen sites / 447 repairs vs ~125 sites / ~0 repairs with live-cut
        # placement). The OBJECTIVE weight stays upstream's get_num_input_cts — the
        # consumed-later outputs of the producing layer (out_x below) — so arm prices
        # match upstream's edge weights while placement refreshes the full boundary.
        crossing: dict[str, set[str]] = {s: set() for s in order}
        out_x: dict[str, set[str]] = {s: set() for s in order}
        # unplaceable live members (fold/deliberate outputs, runtime-owned steps,
        # non-KV inputs) are EXCLUDED from crossing (a site sanitize would drop must
        # not be modeled as refreshed) and instead raise the boundary's reset floor
        # to their residual depth — the model prices exactly what execution delivers.
        from ..ir import is_kv_cache_read as _kv
        l0cap0 = int((self.budget.L - 1) // unit)

        def _static_placeable(v: str) -> bool:
            pnode = g.producer_of.get(v)
            if pnode is None:
                return _kv(v)
            if pnode.is_fold_bts or pnode.is_deliberate_bts:
                return False
            st = pnode.step or ""
            return not any(k in st for k in self.forbid_steps if k)

        reset_floor: dict[str, int] = {s: 0 for s in order}
        for v, lc in last_use.items():
            pp = pos[step_of_var[v]]
            out_x[step_of_var[v]].add(v)
            ok = _static_placeable(v)
            dv = min(l0cap0, int(math.ceil(dprime.get(v, 0.0) / unit - 1e-9)))
            for i in range(pp, lc):
                if ok:
                    crossing[order[i]].add(v)
                else:
                    reset_floor[order[i]] = max(reset_floor[order[i]], dv)
        cost = {s: max(0, int(math.ceil((max_out[s] - max_in[s]) / unit - 1e-9)))
                if max_out[s] > max_in[s] else 0 for s in order}
        # Only steps containing an UNCONDITIONAL interior refresh
        # (deliberate/fold — they always execute) become landing fragments. A drop
        # caused by hints firing in sim0 is conditional: under the baseline's own
        # placement the hints may not fire, and modeling that landing as free lets
        # the solver skip real refreshes (measured: 468-repair collapse).
        l0cap = int((self.budget.L - 1) // unit)
        uncond_reset_steps: set[str] = set()
        for n in g.nodes:
            if (n.is_deliberate_bts or n.is_fold_bts) and n.output:
                s = step_of_var.get(n.output)
                if s is not None:
                    uncond_reset_steps.add(s)
        self._out_x = out_x
        resets = {s: max(0, min(l0cap, int(math.ceil(max_out[s] / unit - 1e-9))))
                  for s in order
                  if max_out[s] < max_in[s] and s in uncond_reset_steps}
        entry = max((self.seed_consumed(v) for v in g.inputs), default=0.0)
        # Clamp: an unclamped deep entry matches no fragment cell and
        # silently degrades the whole solve to the greedy fallback.
        entry_level = max(0, min(l0cap, int(math.ceil(entry / unit - 1e-9))))
        return sg, order, cost, crossing, entry_level, pos, resets, reset_floor

    # ── the solver ─────────────────────────────────────────────────────────────────
    def choose_sites(self, sim0: SimResult) -> set[str]:
        """One level-DAG solve, iterated to self-consistency against the twin.

        Hint refreshes are runtime-conditional: sim0 models them as firing, but the
        placement this solve produces changes the very levels that decide firing, so
        a single pass under-places around hint-carrying approximation steps
        (goldschmidt/newton/remez ladders). Upstream Orion likewise re-validates the placed network and
        re-solves until it passes, so the iteration belongs to the placer, not to
        the shared repair machinery."""
        sites = set(self._solve_once(sim0))
        if self.placed:
            return sites            # re-entry from the shared fixpoint: single pass
        if not self.hint_aware():
            return sites            # hint-blind: one shot, as published
        iters = 0
        for _ in range(self.MAX_FIXPOINT_ITERS):
            placeable = {v for v in sites if v in self.g.producer_of
                         and self._placeable(v, sim0.consumed)}
            self.placed = placeable
            sim = self._sim()
            self.placed = set()
            if not sim.over_budget:
                break
            more = set(self._solve_once(sim)) - sites
            if not more:
                break
            iters += 1
            sites |= more
        self.meta["selfconsistency_iters"] = iters
        return sites

    def _solve_once(self, sim0: SimResult) -> set[str]:
        unit = max(1, self.g.level_unit)
        l0 = max(1, int((self.budget.L - 1) // unit))
        # Layer = capture step: Orion's network nodes are torch modules (a Linear is one
        # atomic node of depth 1 hiding many primitive ops), and our capture step is
        # that granularity. Joins drop freely (landing fragments allowed).
        gran = "step"
        self.meta["join_freedrop"] = True
        sg, order, cost, crossing, entry_level, pos, resets, reset_floor = \
            self._step_graph(sim0 if self.hint_aware() else None)
        unfixable = sorted(s for s in order if cost.get(s, 0) > l0)
        self.meta.update(l0=l0, granularity=gran, num_steps=len(order), deviations=[
            f"layer granularity = {gran}",
            "count objective (cts crossing per bootstrap)",
            "deepest-cost path as the representative branch path per fork child",
            "equal-level joins with free level-drops (upstream Case 1/4)",
            "entry level pinned to the capture/chain seed (runtime hand-off)",
            "over-deep layers capped for the solver, left to counted repair"],
            unfixable_steps=unfixable[:16])

        # Non-KV pseudo-sources are unbootstrappable at runtime — give
        # them no bootstrap arm so the solver optimizes over executable sites only.
        from ..ir import is_kv_cache_read
        pseudo = getattr(self, "_pseudo_sources", set())
        # arm price = upstream's get_num_input_cts (the layer's consumed-later
        # outputs), NOT the full live-cut width used for placement — pricing arms at
        # cut width makes the solver dodge wide boundaries and under-place.
        wsrc = self._out_x
        w = {s: float(max(1, len(wsrc.get(s) or crossing.get(s, ())))) for s in order
             if crossing.get(s)
             and not (s in pseudo and not any(
                 is_kv_cache_read(v) for v in crossing.get(s, ())))}

        def rf(site: str | None) -> int:
            return reset_floor.get(site, 0) if site else 0

        def frag_step(s: str) -> Frag:
            # exact exits mid-chain: a free label raise between consecutive layers
            # lets the solver hide off-path depth (measured collapse: 8 chosen sites,
            # 468 repairs). Upstream's free level-drop (Case 1/4) is applied where it
            # is actually sound for our max-join runtime: at region JOINS, via
            # _raise_exits on branch fragments below.
            if s in resets:
                land = resets[s]
                return {(i, land): (0.0, frozenset()) for i in range(l0 + 1)}
            k = min(cost.get(s, 0), l0)
            return {(i, i + k): (0.0, frozenset()) for i in range(l0 + 1 - k)}

        def _raise_exits(B: Frag) -> Frag:
            # A branch may arrive at the join at any DEEPER label for
            # free (the join max-selects anyway) — running min over exit labels.
            out: Frag = dict(B)
            for i in range(l0 + 1):
                best: Cell | None = None
                for j in range(l0 + 1):
                    cur = out.get((i, j))
                    if cur is not None and (best is None or cur[0] < best[0]):
                        best = cur
                    elif best is not None and (cur is None or best[0] < cur[0]):
                        out[(i, j)] = best
            return out

        # ── find_residuals: fork -> join, representative path per child ────────────
        sink = "__sink__"
        sgx = sg.copy()
        sgx.add_node(sink)
        for s in order:
            if sg.out_degree(s) == 0:
                sgx.add_edge(s, sink)

        def _deep_path(src: str, dst: str) -> list[str] | None:
            """The DEEPEST (max summed layer cost) src..dst path — the binding
            branch representative. A fewest-hops representative routes AROUND the
            deep approximation chains (goldschmidt/newton/remez ladders) the model
            exists to price, leaving them claimed-but-unpriced (measured: the whole
            ln_1 chain unpriced, 38 residual violations)."""
            if src == dst:
                return [src]
            anc = nx.ancestors(sgx, dst) | {dst}
            if src not in anc:
                return None
            nodes = (nx.descendants(sgx, src) | {src}) & anc
            best_c: dict[str, float] = {src: float(cost.get(src, 0))}
            best_p: dict[str, str | None] = {src: None}
            for s in sorted(nodes, key=lambda x: pos.get(x, len(order))):
                if s == src:
                    continue
                preds = [p for p in sgx.predecessors(s) if p in best_c]
                if not preds:
                    continue
                pb = max(preds, key=lambda p: best_c[p])
                best_c[s] = best_c[pb] + float(cost.get(s, 0))
                best_p[s] = pb
            if dst not in best_p:
                return None
            path, cur = [], dst
            while cur is not None:
                path.append(cur)
                cur = best_p[cur]
            return path[::-1]

        regions: dict[str, tuple[str, list[list[str]]]] = {}
        for f in order:
            succs = list(sg.successors(f))
            if len(succs) <= 1:
                continue
            # first common step over ALL paths (descendant intersection), not over
            # one shortest path per successor: an unweighted shortest path escapes
            # to the nearest side sink (KV-cache pushes, reduce dead ends), so the
            # intersection misses reconvergences that every real trunk path shares
            # and forks that DO re-converge lose their region.
            reach = [nx.descendants(sgx, si) | {si} for si in succs]
            common = set.intersection(*reach) - {sink}
            # no common real step: the branches never re-converge (multi-sink DAG —
            # side exits like KV-cache pushes); the virtual sink is the join and the
            # branches run to the program end with equal exit levels (the closest
            # generalization of upstream's model). Only sound with DEEP-path
            # representatives: a fewest-hops branch to sink covers a sliver of the
            # span the region then swallows from the topo walk (measured collapse).
            join = min(common, key=lambda s: pos[s]) if common else sink
            branches: list[list[str]] = []
            ok = True
            for si in succs:
                if si == join:
                    branches.append([])              # identity/residual edge
                    continue
                p = _deep_path(si, join)
                if p is None:
                    ok = False
                    break
                branches.append(p[:-1])              # steps strictly before join
            if ok:
                regions[f] = (join, branches)
        # innermost-first: smallest fork->join span solved first, so outer branch
        # paths can splice the solved fragment (build_level_dag_from_path)
        pos[sink] = len(order)
        region_order = sorted(regions, key=lambda f: pos[regions[f][0]] - pos[f])
        solved: dict[str, tuple[str, Frag]] = {}   # fork -> (join, fragment incl. F..J)

        def build_chain(steps: list[str], lead_site: str | None) -> Frag:
            """Compose step fragments along a path, splicing solved nested regions.
            lead_site = producer feeding the first step (its outputs may bootstrap)."""
            M: Frag = {(i, i): (0.0, frozenset()) for i in range(l0 + 1)}
            prev = lead_site
            i = 0
            while i < len(steps):
                s = steps[i]
                if s in solved and s != steps[-1]:
                    join, RF = solved[s]
                    # splice the solved region only when its join lies ON this path;
                    # otherwise (overlapping region, F12) splicing would fold steps
                    # beyond this chain's span into the fragment and then keep
                    # composing this chain's remaining steps on top of it — free
                    # off-path depth and a dangling prev. Price the fork as a plain
                    # step instead.
                    ji = -1
                    try:
                        ji = steps.index(join, i)
                    except ValueError:
                        pass
                    if ji >= 0:
                        M = _seq(M, RF, l0, w.get(prev) if prev else None, prev,
                                 reset=rf(prev))
                        prev = join
                        i = ji + 1
                        continue
                M = _seq(M, frag_step(s), l0, w.get(prev) if prev else None, prev,
                         reset=rf(prev))
                prev = s
                i += 1
            return M

        for f in region_order:
            join, branches = regions[f]
            per_branch: list[Frag] = []
            for chain in branches:
                if not chain:
                    # identity/residual edge: free drop to any deeper join label
                    # (upstream-faithful Case 1).
                    B: Frag = {(i, j): (0.0, frozenset())
                               for i in range(l0 + 1) for j in range(i, l0 + 1)}
                    per_branch.append(B)
                    continue
                # fork bootstrap priced once at region entry (F4), not per branch.
                B = build_chain(chain, lead_site=None)
                tail = chain[-1]
                # allow bootstrapping the branch tail before the join
                extra: Frag = {}
                for (i, k), (c, ss) in B.items():
                    cur = extra.get((i, 0))
                    wt = w.get(tail)
                    if wt is None:
                        continue          # nothing placeable at this tail boundary
                    cand = (c + wt, ss | {tail})
                    if k != 0 and (cur is None or cand[0] < cur[0]):
                        extra[(i, 0)] = cand
                for cell, val in extra.items():
                    if cell not in B or val[0] < B[cell][0]:
                        B[cell] = val
                per_branch.append(_raise_exits(B))
            # aggregate: equal (i, j) across branches, costs summed, sites unioned
            agg: Frag = {}
            keys = set(per_branch[0])
            for B in per_branch[1:]:
                keys &= set(B)
            for cell in keys:
                tot, sites = 0.0, frozenset()
                for B in per_branch:
                    c, ss = B[cell]
                    tot += c
                    sites = sites | ss
                agg[cell] = (tot, sites)
            # F4: the fork's refresh feeds ALL branches, priced ONCE on the
            # region-entry connector (upstream's aux-fork structure).
            R = _seq(frag_step(f), agg, l0, w.get(f), f)         # F then branches
            if join != sink:
                R = _seq(R, frag_step(join), l0, None, None)     # then the join
            solved[f] = (join, R)

        # ── full network: topo walk splicing solved regions ────────────────────────
        # No blanket "claimed" skip: the splice jump below already skips exactly the
        # [fork..join] span of every region that IS spliced; skipping all claimed
        # steps additionally hides steps of regions never spliced anywhere, which
        # then get NO pricing and no bootstrap arm.
        M: Frag = {(entry_level, entry_level): (0.0, frozenset())}
        prev: str | None = None
        i = 0
        while i < len(order):
            s = order[i]
            if s in solved:
                join, RF = solved[s]
                M = _seq(M, RF, l0, w.get(prev) if prev else None, prev,
                         reset=rf(prev))
                prev = join
                i = pos[join] + 1
                continue
            M = _seq(M, frag_step(s), l0, w.get(prev) if prev else None, prev,
                     reset=rf(prev))
            prev = s
            i += 1

        if not M:
            self.meta["solver_empty_fallback"] = True
            decisions = self._greedy(order, sg, cost, crossing, entry_level, l0)
            return {v for (s, _lv) in decisions for v in crossing.get(s, ())}

        # chain hand-off: upstream plans whole programs (nothing after the output),
        # but here the exit seeds the NEXT block's entry — a count-optimal deep exit
        # makes the successor's first ops infeasible before any boundary exists. The
        # hand-off is a module boundary in Orion's own vocabulary: offer a terminal
        # boundary refresh and prefer exits the successor can absorb.
        exit_cap = max(0, l0 - self.EXIT_CAP_SLACK)
        if prev is not None and w.get(prev) is not None:
            r_term = rf(prev)
            for (i0, j), (c, ss) in list(M.items()):
                if j > exit_cap:
                    cell = (i0, r_term)
                    cand = (c + w[prev], ss | {prev})
                    if cell not in M or cand[0] < M[cell][0]:
                        M[cell] = cand
        feas_exits = {k: v for k, v in M.items() if k[1] <= exit_cap}
        pick_from = feas_exits or M
        best = min(pick_from.items(), key=lambda kv: kv[1][0])
        (_i, exit_lv), (total_cost, boot_steps) = best
        self.meta.update(solver_cost=total_cost, solver_exit_level=exit_lv,
                         num_boot_steps=len(boot_steps),
                         num_regions=len(solved))
        sites: set[str] = set()
        for s in boot_steps:
            sites |= set(crossing.get(s, ()))
        return sites

    # ── emergency fallback only (solver produced no feasible matrix) ───────────────
    def _greedy(self, order, sg, cost, crossing, entry_level, l0):
        refreshed: set[str] = set()
        decisions: list[tuple[str, int]] = []
        unfixable: set[str] = set()

        def propagate():
            lv: dict[str, int] = {}
            binding: dict[str, str | None] = {}
            first_bad = None
            for s in order:
                preds = list(sg.predecessors(s))
                if preds:
                    b = max(preds, key=lambda p: 0 if p in refreshed else lv.get(p, 0))
                    base = 0 if b in refreshed else lv.get(b, 0)
                else:
                    b, base = None, entry_level
                binding[s] = b
                total = base + cost.get(s, 0)
                if total > l0 and first_bad is None and s not in unfixable:
                    first_bad = s
                lv[s] = min(total, l0)
            return lv, binding, first_bad

        for _ in range(4 * len(order)):
            lv, binding, bad = propagate()
            if bad is None:
                return decisions
            k = cost.get(bad, 0)
            b = binding[bad]
            base = entry_level if b is None else (0 if b in refreshed else lv.get(b, 0))
            need = base + k - l0
            cand, cur, deepest = None, b, None
            while cur is not None:
                cur_lv = 0 if cur in refreshed else lv.get(cur, 0)
                if crossing.get(cur) and cur not in refreshed and cur_lv > 0:
                    if deepest is None:
                        deepest = cur
                    if cur_lv >= need:
                        cand = cur
                if cur_lv < need and cand is not None:
                    break
                if cur_lv == 0 and cur in refreshed:
                    break
                cur = binding.get(cur)
            cand = cand or deepest
            if cand is None:
                unfixable.add(bad)
                continue
            refreshed.add(cand)
            decisions.append((cand, 0))
        return decisions
