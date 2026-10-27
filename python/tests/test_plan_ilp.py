"""The exact placer (perseus/plan/placer/ilp.py): its model is the sim, and its answer is optimal.

TWIN: with decisions pinned, the MILP's levels / degrees / hints equal `simulate()`'s on every var. OPTIMAL: no
cheaper placement exists on the synthetic graphs (enumerated). DOMINANT: never costlier than the min-cut.
PLUMBING: plan_block(placer="ilp") emits a labelled plan with the solver's status in summary.placer_meta.

  python -m pytest tests/test_plan_ilp.py
"""
from __future__ import annotations

import itertools
import os
import random
import unittest
from pathlib import Path
from unittest import mock

from perseus.plan.placer.ilp import IlpPlacer
from perseus.plan.placer.place import Placer
from perseus.plan.placer.planner import PlanConfig, plan_block

try:
    import scipy  # noqa: F401
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

FIX = Path(__file__).resolve().parent / "fixtures"
ASSETS = Path(__file__).resolve().parents[2]
# the record decode plan's recipe (StC-first dense refreshes, the shipped-order fallback past the entry, per-route and
# per-step envelopes) on its own graph, the blocks chained at the min-cut plan's exits
DECODE_GRAPH = ASSETS / "graphs" / "gpt2_decode_python_n32_gcheb"
DECODE_PLAN = ASSETS / "bootstrap_placements" / "gpt2_decode_python_n32"
SYNTH = sorted((FIX / "placer_fidelity" / "graphs").glob("*/block_0/graph.json"))
REAL = FIX / "real_graph" / "n32_gpt2_base" / "block_0" / "graph.json"

# the placer-fidelity recipe (tests/test_placer_fidelity.py)
SYNTH_CFG = dict(bootstrap_level=16, max_level=22, source_level=16, cache_read_level=16,
                 level_unit=1, acc_chain="n32", allow_prescale=True, verbose=False)
SYNTH_ENTRY = dict(entry_level=16, entry_deg=2)
# the paper's n32 recipe (scripts/make_plans.sh), dense, with a short solver cap: proving block 0 optimal takes ~100 s, and every
# property checked on it holds for a time-limited incumbent too
REAL_CFG = dict(bootstrap_level=36, source_level=36, cache_read_level=36, level_unit=2,
                max_level=50, cf_max=20, mag_safety=2.0, allow_prescale=False,
                acc_chain="n32", verbose=False, ilp_time_limit=20.0)


def _plan(graph: Path, cfg: dict, entry: dict | None = None, **kw):
    """plan_block with placer ilp in strict mode; returns (plan, the placer instance)."""
    seen = []
    orig = IlpPlacer.run

    def run(self):
        self.strict = True
        seen.append(self)
        return orig(self)

    with mock.patch.object(IlpPlacer, "run", run):
        res = plan_block(graph, PlanConfig(placer="ilp", **cfg, **kw), **(entry or {}))
    return res, seen[-1]


def _cost(p: IlpPlacer, placed: set[str]):
    """The ILP objective of a placement, from the sim alone (None if infeasible)."""
    p.placed = set(placed)
    sim = p._sim()
    if sim.over_budget:
        return None
    c = sum(p._site_cost(v) for v in placed)
    for i, fired in sim.hint_fired.items():
        out = p.g.nodes[i].output
        if fired and out:
            c += p._hint_cost(out)
    return c


@unittest.skipUnless(HAVE_SCIPY, "placer ilp needs scipy")
class IlpTwin(unittest.TestCase):
    def _twin(self, graph, cfg, entry=None, draws=12, seed=0):
        _, p = _plan(graph, cfg, entry)
        cands = sorted(v for v in set(p.g.producer_of) | set(p.g.inputs) if p._candidate(v))
        rng = random.Random(seed)
        for _ in range(draws):
            k = rng.randint(0, min(len(cands), 40))
            placed = set(rng.sample(cands, k))
            with self.subTest(graph=graph.parent.parent.name, k=k):
                self.assertEqual(p.twin(placed), [], "the model is not the sim")

    def test_synthetic_graphs(self):
        for g in SYNTH:
            self._twin(g, SYNTH_CFG, SYNTH_ENTRY)

    def test_real_graph(self):
        if not REAL.is_file():
            self.skipTest("real-graph fixture not present")
        self._twin(REAL, REAL_CFG, draws=4)


def _recipe_env(plan_dir: Path) -> dict[str, str]:
    """The PLAN_CMD.txt recipe= line (later tokens win) plus make_plan.sh's defaults."""
    env = {"PLAN_REAL_ROUTE": "1", "PLAN_RAISE_DROP_SET": "1,3,5", "PLAN_HINT_ENV_VETO": "1"}
    for line in (plan_dir / "PLAN_CMD.txt").read_text().splitlines():
        if line.startswith("recipe="):
            env.update(tok.split("=", 1) for tok in line[len("recipe="):].split() if "=" in tok)
    return env


@unittest.skipUnless(HAVE_SCIPY, "placer ilp needs scipy")
class IlpTwinRecord(unittest.TestCase):
    """TWIN on the record decode recipe: a dense refresh past the StC-first entry lands where the shipped order lands,
    every site has its route's and step's envelope, a captured dense refresh past the entry is over budget."""

    def test_record_decode_blocks(self):
        if not (DECODE_GRAPH / "block_1" / "graph.json").is_file() or not DECODE_PLAN.is_dir():
            self.skipTest("record decode graph / plan not present")
        import json
        env = _recipe_env(DECODE_PLAN)
        cfg = dict(bootstrap_level=int(env["BTS_LEVEL"]), max_level=int(env["MAX_LEVEL"]),
                   source_level=int(env["SRC_LEVEL"]), cache_read_level=int(env["CACHE_READ_LEVEL"]),
                   level_unit=int(env["PLAN_LEVEL_UNIT"]), cf_max=int(env["PLAN_CF_MAX"]),
                   mag_safety=float(env["PLAN_MAG_SAFETY"]), allow_prescale=False, acc_chain=env["PLAN_ACC_CHAIN"],
                   sparse_precomps=(512, 1), sparse_out_levels=((1, 16), (512, 26)),
                   slim_cubic=float(env["PLAN_SLIM_CUBIC"]), real_route=True,
                   forbid_steps=tuple(env["FORBID_STEPS"].split(",")), hint_veto_steps=("tail",), verbose=False)
        seen = []

        def run(self_):                     # the incumbent only: the twin needs the min-cut's hint pins
            Placer.run(self_)
            seen.append(self_)
            return self_._sim()

        rng = random.Random(0)
        with mock.patch.dict(os.environ, env), mock.patch.object(IlpPlacer, "run", run):
            for b in (1, 5, 12):            # a down hint past the entry; the ILP's deep dense sites; the LM head
                prev = json.loads((DECODE_PLAN / f"block_{b - 1}_placement.json").read_text())["summary"]
                plan_block(DECODE_GRAPH / f"block_{b}" / "graph.json", PlanConfig(placer="ilp", **cfg),
                           entry_level=prev["exit_level"], entry_deg=prev["exit_deg"])
                p = seen[-1]
                cands = sorted(v for v in set(p.g.producer_of) | set(p.g.inputs) if p._candidate(v))
                for placed in [set(p.placed)] + [set(rng.sample(cands, rng.randint(0, 40))) for _ in range(6)]:
                    with self.subTest(block=b, k=len(placed)):
                        self.assertEqual(p.twin(placed), [], "the model is not the sim")


@unittest.skipUnless(HAVE_SCIPY, "placer ilp needs scipy")
class IlpOptimal(unittest.TestCase):
    def test_no_cheaper_placement_exists(self):
        for g in SYNTH:
            _, p = _plan(g, SYNTH_CFG, SYNTH_ENTRY)
            best = p.meta["objective"]
            self.assertEqual(p.meta["status"], "optimal")
            cands = sorted(v for v in set(p.g.producer_of) | set(p.g.inputs)
                           if p._candidate(v))
            for k in range(int(best) + 1):
                for combo in itertools.combinations(cands, k):
                    c = _cost(p, set(combo))
                    if c is not None:
                        with self.subTest(graph=g.parent.parent.name, placed=combo):
                            self.assertGreaterEqual(c, best - 1e-6,
                                                    "a cheaper placement exists")

    def test_never_worse_than_min_cut(self):
        graphs = [(g, SYNTH_CFG, SYNTH_ENTRY) for g in SYNTH]
        if REAL.is_file():
            graphs.append((REAL, REAL_CFG, None))
        for g, cfg, entry in graphs:
            for objective in ("total", "placed"):
                _, p = _plan(g, cfg, entry, ilp_objective=objective)
                with self.subTest(graph=g.parent.parent.name, objective=objective):
                    self.assertIsNotNone(p.meta.get("mincut_cost"))
                    self.assertLessEqual(p.meta["objective"], p.meta["mincut_cost"] + 1e-6)


@unittest.skipUnless(HAVE_SCIPY, "placer ilp needs scipy")
class IlpTiebreak(unittest.TestCase):
    def test_lightest_optimum(self):
        """With the tie-break (PLAN_ILP_TIEBREAK) both solves are proven and the plan is the optimum the fixed
        weights rank first: every other optimal placement (enumerated) weighs more."""
        for g in SYNTH:
            _, p = _plan(g, SYNTH_CFG, SYNTH_ENTRY, ilp_tiebreak=True)
            self.assertEqual((p.meta["status"], p.meta["tie"]), ("optimal", "optimal"))
            best, chosen = p.meta["objective"], set(p.placed)
            cands = sorted(v for v in set(p.g.producer_of) | set(p.g.inputs) if p._candidate(v))
            p.placed = set()                    # the model's decisions, as the solve saw them
            _, x, *_ = p._build(set(cands), None)
            wt = p._tie_weights(x)
            mine = sum(wt[v] for v in chosen)
            for k in range(int(best) + 1):
                for combo in itertools.combinations(cands, k):
                    c = _cost(p, set(combo))
                    if c is not None and abs(c - best) < 1e-9 and set(combo) != chosen:
                        with self.subTest(graph=g.parent.parent.name, placed=combo):
                            self.assertGreater(sum(wt[v] for v in combo), mine)


@unittest.skipUnless(HAVE_SCIPY, "placer ilp needs scipy")
class IlpPlumbing(unittest.TestCase):
    def test_plan_is_labelled_and_carries_the_solver_status(self):
        if not REAL.is_file():
            self.skipTest("real-graph fixture not present")
        res, p = _plan(REAL, REAL_CFG)
        self.assertEqual(res["heuristic_config"], "ilp")
        meta = res["summary"]["placer_meta"]
        self.assertIn(meta["status"], ("optimal", "time-limit"))
        self.assertEqual(res["summary"]["num_placements"], len(p.placed))

    def test_refuses_level_dependent_pricing(self):
        from perseus.plan.placer.place import PlanInfeasible
        with self.assertRaises(PlanInfeasible):
            _plan(SYNTH[0], SYNTH_CFG, SYNTH_ENTRY, depth_weight=0.5)


if __name__ == "__main__":
    unittest.main()
