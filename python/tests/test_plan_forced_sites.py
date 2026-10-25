"""Tests for the planner's forced refresh sites (perseus/plan/placer/planner.py, plan_block's forced set).

A forced site survives each re-cut pass (rescale_opt, shared_realize, ks_realize): one from force_place under the
min-cut and the ILP placer, one from keep_placements, which survives the prune too. A force_step_refresh keep-anchor
survives the prune. plan_graph_dir's terminal-exit retry survives each pass on a two-block fixture.

  .venv/bin/python python/tests/test_plan_forced_sites.py
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from test_plan_placer import _chain_graph, _node

from perseus.plan.placer.planner import PlanConfig, plan_block, plan_graph_dir

try:
    import scipy  # noqa: F401
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


class ForcedSitesSurviveTest(unittest.TestCase):
    """A forced refresh site survives every pass that runs after the first cut.

    `plan_block` seeds the cut with sites that must be refreshed whatever the levels say: the
    terminal-exit refresh plan_graph_dir's retry adds (`force_place`) when the next block was
    refused at its entry, the force_step_refresh keep-anchors, and the sites of a keep_placements
    plan. The three re-cut passes (rescale_opt, shared_realize, ks_realize) rebuild the placer
    around a trial set of realize anchors and keep the trial when it needs no more bootstraps than
    the incumbent, an equal count included; a trial that does not start from the forced sites can
    drop them, and is kept whenever that costs nothing (on these fixtures it needs one fewer), so
    the forced site is gone. The prune removes the placed refreshes the levels do not need, and a
    keep-anchor is placed whether or not they need it.
    """

    # the planner's unit test recipe, a copy of test_plan_placer.PlanBlockTest.CFG (importing that class would
    # collect its tests here too): unit 1, refreshes land at 16, the ceiling is 24
    CFG = PlanConfig(bootstrap_level=16, max_level=24, source_level=16,
                     cache_read_level=17, verbose=False, forbid_steps=())
    RECUT_PASSES = ("rescale_opt", "shared_realize", "ks_realize")

    def _block0(self, recut_pass) -> tuple[list[dict], str]:
        """Block 0 for `recut_pass`: x0 -> v_0, a product (its rescale pending) read in the one
        shape the pass realizes, then a chain of products long enough that the cut must refresh
        once and short enough that the exit then sits one level under the ceiling at degree 2,
        i.e. effectively at the ceiling. Returns the nodes and the exit var."""
        nodes = [_node("mult", ["x0", "pt_w"], "v_0", 17, ilv=[16, 16], step="b0.m0", period=1)]
        if recut_pass == "shared_realize":    # v_0 read by two products: one rescale serves both
            nodes += [_node("mult", ["v_0", "pt_w"], "v_1a", 18, ilv=[17, 17], step="b0.m1a", period=1),
                      _node("mult", ["v_0", "pt_w"], "v_1b", 18, ilv=[17, 17], step="b0.m1b", period=1),
                      _node("mult", ["v_1a", "v_1b"], "v_2", 19, ilv=[18, 18], step="b0.m2", period=1)]
            prev, lv, last = "v_2", 19, 16
        elif recut_pass == "rescale_opt":     # v_0 (degree 2) summed with x0 (degree 1)
            nodes += [_node("add", ["v_0", "x0"], "v_1", 17, ilv=[17, 16], step="b0.a1", period=1),
                      _node("mult", ["v_1", "pt_w"], "v_2", 18, ilv=[17, 17], step="b0.m2", period=1)]
            prev, lv, last = "v_2", 18, 17
        else:                                 # ks_realize: a key switch reads v_0 (degree 2)
            assert recut_pass == "ks_realize", recut_pass
            nodes += [_node("rotate", ["v_0"], "v_1", 17, ilv=[17], step="b0.r1", period=1),
                      _node("mult", ["v_1", "pt_w"], "v_2", 18, ilv=[17, 17], step="b0.m2", period=1)]
            prev, lv, last = "v_2", 18, 17
        for i in range(3, last + 1):
            nodes.append(_node("mult", [prev, "pt_w"], f"v_{i}", lv + 1, ilv=[lv, lv],
                               step=f"b0.m{i}", period=1))
            prev, lv = f"v_{i}", lv + 1
        return nodes, prev

    def _graph_dir(self) -> Path:
        d = Path(tempfile.mkdtemp(prefix="perseus_plan_forced_"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        return d

    @staticmethod
    def _write_block(d: Path, name: str, nodes) -> Path:
        (d / name).mkdir()
        gf = d / name / "graph.json"
        gf.write_text(json.dumps({"version": 1, "nodes": nodes}))
        return gf

    @staticmethod
    def _refreshed(plan: dict) -> set[str]:
        return {p["target_var"] for p in plan["placements"]}

    def _check_forced_site_survives_each_recut_pass(self, **placer_cfg):
        for recut_pass in self.RECUT_PASSES:
            with self.subTest(recut_pass=recut_pass):
                nodes, exit_var = self._block0(recut_pass)
                gf = self._write_block(self._graph_dir(), "block_0", nodes)
                cfg = dataclasses.replace(self.CFG, **placer_cfg, **{recut_pass: True})
                plan = plan_block(gf, cfg, force_place={exit_var})
                # the pass found a candidate and kept a trial: its realize anchor is in the plan
                # over and above the placed refreshes (which rescale_after lists as well).
                # Otherwise the pass ran empty and the test would pass whatever the trial
                # started from.
                self.assertTrue(set(plan["rescale_after"]) - self._refreshed(plan),
                                f"{recut_pass} realized nothing: the fixture does not exercise it")
                self.assertIn(exit_var, self._refreshed(plan))
                self.assertEqual(plan["summary"]["exit_level"], self.CFG.bootstrap_level)

    def test_a_forced_site_survives_each_recut_pass(self):
        self._check_forced_site_survives_each_recut_pass()

    @unittest.skipUnless(HAVE_SCIPY, "placer ilp needs scipy")
    def test_a_forced_site_survives_each_recut_pass_under_the_ilp_placer(self):
        """The other exact placer: the re-cut trials are built from the same constructor."""
        self._check_forced_site_survives_each_recut_pass(placer="ilp", ilp_time_limit=30.0)

    def test_terminal_exit_retry_survives_each_recut_pass(self):
        """plan_graph_dir: block 1 is refused at the entry block 0 leaves (its only product sits in
        a step the configuration forbids refreshing in, and a block input is never a site), block 0
        is re-planned with a forced refresh on its exit, and that refresh must still be in block 0's
        plan after the pass. Without it block 1 is refused again at the same entry and, the retry
        being once per block, the chain ends in PlanInfeasible."""
        for recut_pass in self.RECUT_PASSES:
            with self.subTest(recut_pass=recut_pass):
                nodes, _ = self._block0(recut_pass)
                d = self._graph_dir()
                self._write_block(d, "block_0", nodes)
                self._write_block(d, "block_1", [
                    _node("mult", ["x0", "pt_w"], "y_0", 17, ilv=[16, 16], step="b1.m0", period=1)])
                cfg = dataclasses.replace(self.CFG, forbid_steps=("b1.",), **{recut_pass: True})
                out = d / "plans"
                with self.assertLogs("perseus.plan.placer.planner", level="WARNING") as cm:
                    summaries = plan_graph_dir(d, out, cfg)
                self.assertTrue(any("terminal-exit refresh" in m for m in cm.output),
                                "the fixture must exercise the retry")
                self.assertEqual([s["block"] for s in summaries], ["block_0", "block_1"])
                b0 = json.loads((out / "block_0_placement.json").read_text())
                self.assertTrue(set(b0["rescale_after"]) - self._refreshed(b0),
                                f"{recut_pass} realized nothing on the retried block: "
                                "the fixture does not exercise it")
                self.assertIn(summaries[0]["exit_var"], self._refreshed(b0))
                self.assertEqual(summaries[0]["exit_level"], self.CFG.bootstrap_level)
                self.assertTrue((out / "block_1_placement.json").is_file())

    def test_a_kept_placement_survives_each_recut_pass(self):
        """keep_placements seeds the cut from another plan's sites, which the re-cut trials and the prune keep as
        well: here the exit, which block 0's own levels do not need."""
        for recut_pass in self.RECUT_PASSES:
            with self.subTest(recut_pass=recut_pass):
                nodes, exit_var = self._block0(recut_pass)
                d = self._graph_dir()
                gf = self._write_block(d, "block_0", nodes)
                (d / "kept").mkdir()
                (d / "kept" / "block_0_placement.json").write_text(
                    json.dumps({"placements": [{"target_var": exit_var}]}))
                cfg = dataclasses.replace(self.CFG, keep_placements=str(d / "kept"), prune=True,
                                          **{recut_pass: True})
                plan = plan_block(gf, cfg)
                self.assertTrue(set(plan["rescale_after"]) - self._refreshed(plan),
                                f"{recut_pass} realized nothing: the fixture does not exercise it")
                self.assertIn(exit_var, self._refreshed(plan))

    def test_a_keep_anchor_survives_the_prune(self):
        """force_step_refresh forces a refresh whether or not the levels need it (the cut is blind to
        what a cleansing refresh buys), here one they do not need; the prune removes placed refreshes
        the levels do not need."""
        nodes = _chain_graph(4)               # 4 products from 16 fit the 8-level budget unrefreshed
        nodes[1]["step"] = "chain.keep.m1"    # the anchor: the last node of the step chain.keep.m1
        gf = self._write_block(self._graph_dir(), "block_0", nodes)
        cfg = dataclasses.replace(self.CFG, force_step_refresh=("keep",), prune=True)
        plan = plan_block(gf, cfg)
        self.assertEqual(self._refreshed(plan), {"v_1"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
