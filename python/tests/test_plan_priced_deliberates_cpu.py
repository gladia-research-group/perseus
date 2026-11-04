"""plan/placer/planner.priced_deliberates and RefreshPlanner.spec_dense: the code's own refreshes of a cache-following
step (PLAN_STEP_LANDING) on the dense StC-first route are the ones the planner gives a correction factor, priced on the
dense route only and sized by their captured magnitude; K/V pushes, external finishes, sparse captures and recipes
without a step landing get none."""
from __future__ import annotations

import unittest

from perseus.plan import btserr
from perseus.plan.placer.ir import Graph
from perseus.plan.placer.planner import priced_deliberates
from perseus.plan.placer.refresh import RefreshPlanner

SPEC = "softmax:32,softmax_v:32"


def _delib(out, step, slots, stc_first, max_abs=0.2, external=False, period=512):
    route = {"slots": slots, "stc_first": stc_first}
    return {"op_type": "deliberate_bootstrap", "inputs": ["x"], "input_levels": [48], "output": out,
            "output_level": 26, "output_noise_level": 2, "step": step, "output_max_abs": max_abs,
            "output_max_coeff": max_abs / 40, "pack_period": period,
            "args": {"route": route, **({"external": True} if external else {})}}


def _graph():
    return Graph.from_nodes([
        _delib("sm_small", "blk0.softmax", 0, True, max_abs=0.04),
        _delib("sm_big", "blk0.softmax", 0, True, max_abs=1.9),
        _delib("smv", "blk0.softmax_v", 0, True),
        _delib("kv", "blk0.kv_push", 0, True),                          # not a cache-following step
        _delib("sm_x", "blk0.softmax", 0, True, external=True),        # an external finish
        _delib("sm_sparse", "blk0.softmax", 256, False),               # not re-routed (no dense landing)
    ], level_unit=2)


class Selection(unittest.TestCase):
    def test_only_dense_stc_first_refreshes_of_cache_following_steps(self):
        self.assertEqual(set(priced_deliberates(_graph(), SPEC)), {"sm_small", "sm_big", "smv"})

    def test_no_step_landing_none(self):
        self.assertEqual(priced_deliberates(_graph(), ""), ())


class DensePricing(unittest.TestCase):
    def _planner(self):
        # no measured table: the analytic law (btserr.analytic_rel_err); the shadow from the fields (no recipe)
        return RefreshPlanner(table=btserr.AccuracyTable(None), policy=btserr.SitePolicy(), graph=_graph(),
                              out_deg=2, sparse_precomps=(256, 512), bootstrap_level=28)

    def test_dense_only_and_cf_follows_magnitude(self):
        rp = self._planner()
        small, big = rp.spec_dense("sm_small"), rp.spec_dense("sm_big")
        self.assertFalse(small.route)
        self.assertFalse(big.route)
        self.assertLess(small.cf, big.cf)
        self.assertIs(rp.spec_dense("sm_small"), small)                 # memoised
