"""The exact placer (perseus/plan/placer/ilp.py): its model is the sim, and its answer is optimal.

  * TWIN     — with the refresh decisions pinned to arbitrary placements (feasible or
               not), the MILP's levels, degrees and hint decisions equal `simulate()`'s
               on every var. This is what makes "optimal" mean optimal for the planner.
  * OPTIMAL  — on the synthetic fixture graphs, no placement is cheaper than the ILP's:
               checked by enumerating every placement that could be (a site costs >= 1,
               so only placements of at most `objective` sites can beat it).
  * DOMINANT — never costlier than the min-cut it starts from, on every graph here.
  * PLUMBING — plan_block(placer="ilp") emits a plan through the usual postconditions,
               labelled, with the solver's status in summary.placer_meta.

  python -m pytest tests/test_plan_ilp.py
"""
from __future__ import annotations

import itertools
import random
import unittest
from pathlib import Path
from unittest import mock

from perseus.plan.placer.ilp import IlpPlacer
from perseus.plan.placer.planner import PlanConfig, plan_block

try:
    import scipy  # noqa: F401
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

FIX = Path(__file__).resolve().parent / "fixtures"
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
