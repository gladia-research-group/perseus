"""plan/placer/planner.cache_route_deliberates: a deliberate refresh in a cache-following step (PLAN_STEP_LANDING) captured
on a sparse route is planned on the dense StC-first route (its landing, degree 2, and the StC-first entry on its input);
K/V pushes, external finishes, refreshes already on that route and recipes without a step landing are left alone."""
from __future__ import annotations

import unittest

from perseus.plan.placer.ir import Graph
from perseus.plan.placer.planner import cache_route_deliberates

RAW = (26.0, 2)            # the dense StC-first landing before its realize (bootstrap level 28, unit 2)
SPEC = "softmax:32,softmax_v:32"


def _delib(out, step, level, slots, stc_first=False, external=False, inp="x"):
    route = {"slots": slots, "stc_first": stc_first}
    return {"op_type": "deliberate_bootstrap", "inputs": [inp], "input_levels": [48], "output": out,
            "output_level": level, "output_noise_level": 2, "step": step, "output_max_abs": 1.0, "pack_period": 256,
            "args": {"route": route, **({"external": True} if external else {})}}


def _graph():
    nodes = [
        _delib("sm_r", "blk0.softmax", 22, 256),                      # the gs_num_refresh one, captured sparse
        _delib("smv_r", "blk0.softmax_v", 24, 512),                   # another cache-following step
        _delib("kv_r", "blk0.kv_push", 22, 256),                      # not a cache-following step
        _delib("sm_d", "blk0.softmax", 26, 0, stc_first=True),        # already on the dense StC-first route
        _delib("sm_x", "blk0.softmax", 22, 256, external=True),       # an external finish (PCMM / CCMM)
        _delib("sm_hi", "blk0.softmax", 30, 256),                     # a landing already deeper than the route's
    ]
    return Graph.from_nodes(nodes, level_unit=2)


class CacheRouteDeliberates(unittest.TestCase):
    def test_reroutes_only_sparse_refreshes_of_cache_following_steps(self):
        g, k = cache_route_deliberates(_graph(), SPEC, RAW)
        by = {n.output: n for n in g.nodes}
        self.assertEqual(k, 2)
        for v in ("sm_r", "smv_r"):
            self.assertEqual((by[v].output_level, by[v].output_deg, by[v].stc_first), (26.0, 2, True), v)
        self.assertEqual((by["kv_r"].output_level, by["kv_r"].stc_first), (22.0, False))
        self.assertEqual((by["sm_d"].output_level, by["sm_d"].stc_first), (26.0, True))
        self.assertEqual((by["sm_x"].output_level, by["sm_x"].stc_first), (22.0, False))
        self.assertEqual((by["sm_hi"].output_level, by["sm_hi"].stc_first), (30.0, False))

    def test_no_step_landing_no_change(self):
        g0 = _graph()
        g, k = cache_route_deliberates(g0, "", RAW)
        self.assertEqual(k, 0)
        self.assertIs(g, g0)

    def test_idempotent(self):
        g, _ = cache_route_deliberates(_graph(), SPEC, RAW)
        g2, k2 = cache_route_deliberates(g, SPEC, RAW)
        self.assertEqual(k2, 0)
        self.assertIs(g2, g)


if __name__ == "__main__":
    unittest.main()
