"""DaCapo-style segment-DP placer.

Port of the placement algorithm of corelab-src/dacapo (MIT license, (c) 2024
corelab-src; USENIX Security 2024 "DaCapo: Automatic Bootstrapping Management for
Efficient Fully Homomorphic Encryption" — `DaCapoPlanner.cpp`, `CandidateSelection.cpp`,
`BypassDetection.cpp`, `CoverageRecorder.cpp`), adapted to this repo's IR.

Upstream algorithm: cut candidates are program points with few live-out ciphertexts;
a forward DP over candidates treats each pair (from -> to) as one "bootstrapping
segment" (bootstrap all live-outs of `from`, compute to `to`), legal only while
`to` lies inside `from`'s bootstrappability coverage; segment weight comes from a
per-op latency table; bypass detection exempts live values whose consumers lie beyond
a scale threshold from the boundary bootstrap; candidate selection widens the allowed
live-out width only until the program compiles.

Adaptations (documented in summary.placer_meta.deviations):
  * program points = topo positions after each level-consuming node (the rescale
    boundaries; DaCapo's SMU boundaries are exactly the scale-management points).
  * "accumulated scale" maps to consumed prime levels: budget L is the scale budget,
    threshold*L the bypass wall; coverage passes reset every live var to fresh at the
    cut (upstream: segment args enter at waterline) and accumulate per-op costs.
  * latency table is a flat per-op-class ms table (latency_n32.json knob) instead of
    per-level tables; bootstrap dominates either way.
  * width search validates with the shared simulate() twin ("compiles" upstream).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..ir import ADD_FAMILY, MULT_FAMILY, is_kv_cache_read
from ..sim import SimResult
from .base import BaselinePlacer


def _op_class(op: str) -> str:
    if op in MULT_FAMILY:
        return "mult"
    if op == "rotate":
        return "rotate"
    if op in ADD_FAMILY:
        return "add"
    return "other"


@dataclass
class DaCapoPlacer(BaselinePlacer):
    NAME = "dacapo_segdp_v1"
    NAME_KEY = "dacapo"

    # upstream's --threshold CLI knob ("scale coverage threshold", 0..1; their code
    # defaults 0.5).
    threshold: float = 0.5
    latency: dict[str, float] = field(default_factory=lambda: {
        "bootstrap": 28.5, "mult": 0.5, "rotate": 0.3, "add": 0.05, "other": 0.05})
    latency_table_name: str = "builtin"
    max_width_iters: int = 12

    def choose_sites(self, sim0: SimResult) -> set[str]:
        g = self.g
        L = self.budget.L - 1                      # usable primes (guard honored)
        nodes = g.nodes
        N = len(nodes)

        # last consumer position per var (liveness). Audit fix 6: produced vars with
        # no in-block consumer (block outputs, KV writes) are RETURN-LIVE — upstream
        # keeps them in every subsequent liveOut snapshot, so they stay live to N.
        last_use: dict[str, int] = {}
        first_use_after: dict[str, list[int]] = {}
        for n in nodes:
            for v in n.cipher_inputs:
                last_use[v] = n.idx
                first_use_after.setdefault(v, []).append(n.idx)
        prod_pos: dict[str, int] = {v: p.idx for v, p in g.producer_of.items()}
        for v in g.producer_of:
            last_use.setdefault(v, N)
        for v in g.inputs:
            prod_pos.setdefault(v, -1)

        # candidates: after every level-consuming node, plus the entry
        cands = [-1] + [n.idx for n in nodes if n.cost > 0]
        cands = sorted(set(cands))
        self.meta.update(num_candidates=len(cands), threshold=self.threshold,
                         latency_table=self.latency_table_name, deviations=[
            "cut points = post-rescale topo positions",
            "accumulated scale = consumed primes; coverage resets live vars fresh",
            "per-op-class latency table, level-indexed by limb fraction",
            "width search validated by the shared sim twin"])

        # A var the runtime can never refresh
        # (non-KV graph input — no producer, so no site to fire at) must not count in
        # a candidate's live-out WIDTH: it pushes candidates out of the width search
        # and forces an extra cut where upstream places one bootstrap. Upstream's
        # liveOuts are function args it CAN bootstrap; ours cannot, so excluding them
        # is the faithful adaptation of the same rule.
        def _refreshable(v: str) -> bool:
            if v in g.producer_of:
                return True
            return v in g.inputs and is_kv_cache_read(v)

        live_outs: dict[int, list[str]] = {}
        for i in cands:
            live_outs[i] = sorted(
                v for v, p in prod_pos.items()
                if p <= i < last_use.get(v, -1) and _refreshable(v))

        # forward pass per candidate (all live vars fresh at the cut, upstream's
        # "segment args at waterline"): bootstrappability coverage, bypass threshold,
        # and the LEVEL-INDEXED op latency accumulation (upstream latencyTable[op][lvl]
        # — a fresher ct carries more limbs and costs more; bootstrap cost is flat
        # because upstream modswitches the bootstrap input to the floor first).
        boot_cov: dict[int, int] = {}
        full_cov: dict[int, int] = {}     # laxer bound (nothing after the
        thresh_pos: dict[int, int] = {}   # last cut needs to be bootstrappable)
        seg_lat: dict[tuple[int, int], float] = {}   # (from, to) -> op latency sum
        thr = self.threshold * L
        cand_set = set(cands)

        # Upstream pricing axis (hecate-opt): hecate prices every op as
        #   getLatency() = getLatencyOf(getCipherLevel())
        #   getCipherLevel() = init_level - getRescaleLevel()      (EarthOps.td:101)
        # where getRescaleLevel() = scaleType.getLevel() is a RESCALE COUNTER that
        # RescaleOp increments (`lScale.switchLevel(lScale.getLevel() + 1)`,
        # EarthDialect.cpp) — it counts levels CONSUMED. So the table index is
        # REMAINING levels and the tables rise with it (mul_double = [0, 751, 869,
        # ... 2750]): an op is DEARER on a fresher ciphertext. Our `down` factor has
        # that same sign, so the "our sign is inverted" claim is FALSE as stated.
        # What our port was missing is WHERE the level profile is anchored. hecate
        # runs EarlyModswitch inside every DP segment, so ops execute at the lowest
        # level that still finishes the segment: a latency-table spike probe on
        # Upstream bills the mults of a segment at its END indices, so segment cost is
        # sum_{j=1..k} f(j) — convex in segment length k — and chains split evenly.
        # `segend` anchors the level factor at the segment end (index = levels left to
        # consume before the cut) and is the faithful model.
        lvl_sign = "segend"
        self.meta["lvl_sign"] = lvl_sign

        def lvl_factor(c: float) -> float:
            # limb-proportional: consumed c of the binding operand -> fewer limbs
            return max(0.25, 1.0 - 0.5 * c / max(1.0, L))

        Lf = max(1.0, float(L))

        def seg_end_lat(sw: float, swc: float, k: float) -> float:
            # `segend`: price each op at the level it will ACTUALLY run at once the
            # segment's early-modswitch has pushed it down — index = levels the value
            # still has to consume before the segment ends, k - c. Linear in (k - c),
            # so the sum telescopes into two prefix sums and stays O(N) per `from`.
            return 0.5 * sw + 0.5 * (k * sw - swc) / Lf

        for i in cands:
            lc: dict[str, float] = {v: 0.0 for v in live_outs[i]}
            cov = fcov = N
            tpos = N
            acc = 0.0
            sw = swc = cmax = 0.0
            for n in nodes[i + 1:]:
                # Candidates AT reset nodes still get their seg_lat
                if n.idx in cand_set:
                    seg_lat[(i, n.idx)] = (seg_end_lat(sw, swc, cmax)
                                           if lvl_sign == "segend" else acc)
                if n.is_deliberate_bts or n.is_fold_bts:
                    # Reset to the captured landing, not to fresh
                    if n.output:
                        lc[n.output] = max(0.0, (n.output_level - self.bootstrap_level)
                                           if n.output_level is not None else 0.0)
                    continue
                if n.hint_level is not None and self.hint_aware():
                    # hint refreshes execute in every arm — mirror accum_depths and
                    # reset the hint output (hint-aware porting)
                    if n.output:
                        lc[n.output] = 0.0
                    continue
                ins = [lc.get(v, 0.0) for v in n.cipher_inputs]
                c = (max(ins) if ins else 0.0) + n.cost
                w = self.latency.get(_op_class(n.op),
                                     self.latency.get("other", 0.05))
                acc += w * lvl_factor(c)
                sw += w
                swc += w * c
                cmax = max(cmax, c)
                if n.output:
                    lc[n.output] = c
                if tpos == N and c > thr:
                    tpos = n.idx
                # Upstream's exit segment uses a laxer
                # bound (output needs only decryptability), but our block exits feed
                # the next block and the twin's guard applies — a laxer exit is
                # twin-infeasible by construction. Both bounds coincide here.
                if c > L:
                    cov = fcov = n.idx
                    break
            seg_lat[(i, N)] = (seg_end_lat(sw, swc, cmax)
                               if lvl_sign == "segend" else acc)
            boot_cov[i] = cov
            full_cov[i] = fcov
            thresh_pos[i] = tpos

        # The bypass window belongs to the VALUE, not the cut — v's
        # threshold is the one scanned from v's own production context (nearest
        # candidate at/after its producer), a value whose window closed before the
        # cut bypasses unconditionally, and the cut's own value never bypasses.
        import bisect
        def _own_thresh(v: str) -> int:
            p = prod_pos.get(v, -1)
            k = bisect.bisect_left(cands, p)
            return thresh_pos[cands[min(k, len(cands) - 1)]]

        # "cut" is the faithful mode: upstream's BypassDetection marks EDGES
        # (is_bypassed on operands), which cut-side accounting reproduces.
        bypass_mode = "cut"
        self.meta["bypass_mode"] = bypass_mode

        def bypassed(i: int) -> set[str]:
            if bypass_mode == "cut":
                t = thresh_pos[i]
                return {v for v in live_outs[i]
                        if not any(i < u <= t for u in first_use_after.get(v, ()))}
            out = set()
            for v in live_outs[i]:
                if prod_pos.get(v, -1) == i:
                    continue                      # own value never bypasses
                t = _own_thresh(v)
                if t <= i:
                    # stale window: upstream bypasses unconditionally. Guard: only
                    # when some candidate existed inside the window — a value with
                    # NO refresh opportunity before its window closed would then be
                    # unbootstrappable everywhere (never occurs in upstream's nets).
                    p = prod_pos.get(v, -1)
                    if any(p <= c < t for c in cands):
                        out.add(v)
                        continue
                uses = first_use_after.get(v, ())
                if not any(i < u <= t for u in uses):
                    out.add(v)
            return out

        byp: dict[int, set[str]] = {i: bypassed(i) for i in cands}
        widths = {i: len(live_outs[i]) - len(byp[i]) for i in cands}

        # per-segment op latency prefix sums
        op_lat = [0.0] * (N + 1)
        for n in nodes:
            op_lat[n.idx + 1] = op_lat[n.idx] + self.latency.get(
                _op_class(n.op), self.latency.get("other", 0.05))

        lat_b = self.latency.get("bootstrap", 28.5)

        def run_dp(maxw: int) -> list[int] | None:
            """DP over candidates of width <= maxw; returns chosen boundary list."""
            use = [i for i in cands if i == -1 or widths[i] <= maxw]
            INF = float("inf")
            best: dict[int, float] = {-1: 0.0}
            prev: dict[int, int | None] = {-1: None}
            # end state: reach position N (the block exit)
            best_end, prev_end = INF, None
            for j in use:
                if j == -1:
                    continue
                bj, pj = INF, None
                for i in use:
                    if i >= j or best.get(i, INF) == INF:
                        continue
                    if j >= boot_cov[i]:
                        continue
                    w = best[i] + (len(live_outs[i]) - len(byp[i])) * (
                        lat_b if i != -1 else 0.0) \
                        + seg_lat.get((i, j), op_lat[j + 1] - op_lat[i + 1])
                    if w < bj:
                        bj, pj = w, i
                if bj < INF:
                    best[j], prev[j] = bj, pj
            for i in use:
                if best.get(i, INF) == INF:
                    continue
                # The EXIT segment needs only decryptability (the laxer
                # coverage), not bootstrappability
                if full_cov[i] >= N:              # this segment reaches the exit
                    w = best[i] + (len(live_outs[i]) - len(byp[i])) * (
                        lat_b if i != -1 else 0.0) \
                        + seg_lat.get((i, N), op_lat[N] - op_lat[i + 1])
                    if w < best_end:
                        best_end, prev_end = w, i
            if best_end == INF:
                return None
            chain: list[int] = []
            k = prev_end
            while k is not None and k != -1:
                chain.append(k)
                k = prev.get(k)
            chain.reverse()
            return chain

        # width search: smallest width whose DP plan the twin accepts
        all_w = sorted({w for i, w in widths.items() if i != -1 and w > 0})
        tried = 0
        chosen: list[int] | None = None
        for w in all_w:
            tried += 1
            chain = run_dp(w)
            if chain is None:
                if tried >= self.max_width_iters:
                    break
                continue
            sites = set()
            for b in chain:
                sites |= set(live_outs[b]) - byp[b]
            self.placed = {v for v in sites if self._placeable(v, sim0.consumed)}
            ok = not self._sim().over_budget
            self.placed = set()
            if ok:
                chosen = chain
                self.meta["selected_width"] = w
                break
            if tried >= self.max_width_iters:
                break
        if chosen is None:
            chain = run_dp(max(all_w) if all_w else 0)
            chosen = chain or []
            self.meta["selected_width"] = "unbounded"
        self.meta["num_segments"] = len(chosen) + 1

        sites: set[str] = set()
        for b in chosen:
            sites |= set(live_outs[b]) - byp[b]
        return sites
