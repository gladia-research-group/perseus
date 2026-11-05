"""Forced refresh sites in plan_block (perseus/plan/placer/planner.py): sites refreshed whatever the levels need
(force_place, keep_placements, force_step_refresh) stay refreshed through the re-cut passes (rescale_opt,
shared_realize, ks_realize) and the prune.

Generated blocks mix the shapes the passes act on (a product added to a fresher operand, a product read by two
products, a product read by a rotation); each check also runs an unforced control, so a site the levels need anyway
does not count.

  .venv/bin/python python/tests/test_plan_forced_refresh.py
"""
from __future__ import annotations

import dataclasses
import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from test_plan_placer import _node

from perseus.plan.placer.planner import PlanConfig, plan_block

try:
    import scipy  # noqa: F401
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

PASSES = ("rescale_opt", "shared_realize", "ks_realize")
SEEDS = range(24)


def block(seed: int, n_shapes: int = 7, keep_step: int | None = None) -> list[dict]:
    """A block of `n_shapes` random shapes from x0 at level 16 (one level per product); the products of shape
    `keep_step` carry the step tag "keep"."""
    rnd = random.Random(seed)
    nodes, prev, lv = [], "x0", 16

    def mult(a, b, out_lv, ilv, tag):
        nodes.append(_node("mult", [a, b], f"v{len(nodes)}", out_lv, ilv=ilv, step=tag, period=1))
        return nodes[-1]["output"]

    for k in range(n_shapes):
        tag = f"b.{'keep' if k == keep_step else 's'}{k}"
        shape = rnd.choice("mafr")
        p = mult(prev, "pt_w", lv + 1, [lv, lv], tag)
        if shape == "a":          # the product summed with its own (fresher) input
            nodes.append(_node("add", [p, prev], f"v{len(nodes)}", lv + 1, ilv=[lv + 1, lv], step=tag, period=1))
            prev, lv = nodes[-1]["output"], lv + 1
        elif shape == "f":        # the product read by two products, then joined
            q = mult(p, "pt_w", lv + 2, [lv + 1, lv + 1], tag)
            r = mult(p, "pt_w", lv + 2, [lv + 1, lv + 1], tag)
            prev, lv = mult(q, r, lv + 3, [lv + 2, lv + 2], tag), lv + 3
        elif shape == "r":        # the product rotated
            nodes.append(_node("rotate", [p], f"v{len(nodes)}", lv + 1, ilv=[lv + 1], step=tag, period=1))
            prev, lv = nodes[-1]["output"], lv + 1
        else:
            prev, lv = p, lv + 1
    return nodes


class ForcedRefreshTest(unittest.TestCase):
    CFG = PlanConfig(bootstrap_level=16, max_level=24, source_level=16, cache_read_level=17, verbose=False,
                     forbid_steps=())

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="perseus_forced_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def graph(self, nodes, name="block_0") -> Path:
        d = self.tmp / f"g{len(list(self.tmp.iterdir()))}" / name
        d.mkdir(parents=True)
        (d / "graph.json").write_text(json.dumps({"version": 1, "nodes": nodes}))
        return d / "graph.json"

    @staticmethod
    def refreshed(plan) -> set:
        return {p["target_var"] for p in plan["placements"]}

    def cfg(self, **kw):
        return dataclasses.replace(self.CFG, prune=True, **{p: True for p in PASSES}, **kw)

    def check_force_place(self, **cfg_kw):
        dropped = 0                   # forced sites the unforced control does not refresh
        for seed in SEEDS:
            nodes = block(seed)
            outs = [n["output"] for n in nodes]
            sites = {outs[-1], random.Random(seed).choice(outs[:-1])}
            gf = self.graph(nodes)
            cfg = self.cfg(**cfg_kw)
            plan = plan_block(gf, cfg, force_place=set(sites))
            with self.subTest(seed=seed):
                self.assertLessEqual(sites, self.refreshed(plan))
            dropped += len(sites - self.refreshed(plan_block(gf, cfg)))
        self.assertGreater(dropped, 0, "every forced site was needed anyway: the check is vacuous")

    def test_force_place(self):
        self.check_force_place()

    @unittest.skipUnless(HAVE_SCIPY, "placer ilp needs scipy")
    def test_force_place_ilp(self):
        self.check_force_place(placer="ilp")

    def test_each_pass_realizes(self):
        """Each pass adds realize anchors on some generated block (else the forced checks would not exercise it)."""
        for name in PASSES:
            hits = 0
            for seed in SEEDS:
                nodes = block(seed)
                plan = plan_block(self.graph(nodes), dataclasses.replace(self.CFG, **{name: True}),
                                  force_place={nodes[-1]["output"]})
                hits += bool(set(plan["rescale_after"]) - self.refreshed(plan))
            with self.subTest(recut=name):
                self.assertGreater(hits, 0)

    def test_keep_placements(self):
        dropped = 0
        for seed in SEEDS:
            nodes = block(seed)
            site = random.Random(seed + 1).choice([n["output"] for n in nodes])
            kept = self.tmp / f"kept{seed}"
            kept.mkdir()
            (kept / "block_0_placement.json").write_text(json.dumps({"placements": [{"target_var": site}]}))
            gf = self.graph(nodes)
            plan = plan_block(gf, self.cfg(keep_placements=str(kept)))
            with self.subTest(seed=seed):
                self.assertIn(site, self.refreshed(plan))
            dropped += site not in self.refreshed(plan_block(gf, self.cfg()))
        self.assertGreater(dropped, 0)

    def test_force_step_refresh(self):
        """The anchor is the last node of the tagged step."""
        dropped = 0
        for seed in SEEDS:
            nodes = block(seed, keep_step=3)
            anchor = [n["output"] for n in nodes if ".keep" in n["step"]][-1]
            gf = self.graph(nodes)
            plan = plan_block(gf, self.cfg(force_step_refresh=("keep",)))
            with self.subTest(seed=seed):
                self.assertIn(anchor, self.refreshed(plan))
            dropped += anchor not in self.refreshed(plan_block(gf, self.cfg()))
        self.assertGreater(dropped, 0)


if __name__ == "__main__":
    unittest.main()
