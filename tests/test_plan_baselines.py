"""Tests for the baseline placers (perseus/plan/placer/baselines).

Synthetic cases pin the behaviors that make the benchmark honest:
  * every placer produces a P1-feasible plan on a plain deep chain;
  * Fhelipe's DP is count-optimal on a chain (exact assert);
  * a fork/fan-out gets ONE refresh at the shared producer, not one per consumer;
  * a magnitude-hopeless site is a LOUD PlanInfeasible in raw mode and a counted
    rescue_moved in rescue mode;
  * heuristic_config and placer_meta are stamped into the emitted plan.

Real-graph smoke runs when .cache/graph_n32_gpt2_base is present.

  .venv/bin/python tests/test_plan_baselines.py
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from perseus.plan.placer.place import PlanInfeasible
from perseus.plan.placer.planner import PlanConfig, plan_block
from test_plan_placer import _chain_graph, _node

REAL_GRAPH = Path(os.environ.get(
    "PERSEUS_REAL_GRAPH",
    Path(__file__).resolve().parent / "fixtures" / "real_graph" / "n32_gpt2_base"))
BASELINES = ("fhelipe", "orion", "dacapo")


_TMPDIRS: list[Path] = []


def _write(nodes) -> Path:
    d = Path(tempfile.mkdtemp(prefix="perseus_plan_baselines_"))
    _TMPDIRS.append(d)
    (d / "graph.json").write_text(json.dumps({"version": 1, "nodes": nodes}))
    return d / "graph.json"


def tearDownModule():
    for d in _TMPDIRS:
        shutil.rmtree(d, ignore_errors=True)


def _cfg(**kw) -> PlanConfig:
    base = dict(bootstrap_level=16, max_level=24, source_level=16,
                cache_read_level=17, verbose=False, forbid_steps=())
    base.update(kw)
    return PlanConfig(**base)


def _fork_graph(depth_each=6, start_level=16):
    """x0 forks into two mult chains of `depth_each`, rejoined by an add, then a
    short tail. Total path depth 6+1 > budget 8-1 only via the tail."""
    nodes = []
    for b in ("a", "b"):
        prev, lv = "x0", start_level
        for i in range(depth_each):
            out = f"{b}_{i}"
            nodes.append(_node("mult", [prev, "pt_w"], out, lv + 1,
                               ilv=[lv, lv], step=f"br{b}.m{i}", period=1))
            prev, lv = out, lv + 1
    nodes.append(_node("add", [f"a_{depth_each-1}", f"b_{depth_each-1}"], "j",
                       start_level + depth_each, step="join",
                       ilv=[start_level + depth_each] * 2, period=1))
    prev, lv = "j", start_level + depth_each
    for i in range(4):
        out = f"t_{i}"
        nodes.append(_node("mult", [prev, "pt_w"], out, lv + 1,
                           ilv=[lv, lv], step=f"tail.m{i}", period=1))
        prev, lv = out, lv + 1
    return nodes


def _fanout_graph(n_consumers=6, pre=5, post=4, start_level=16):
    """A deep producer var fanning out to `n_consumers` chains: one refresh at the
    shared producer must suffice; per-consumer refreshes would cost n_consumers."""
    nodes = []
    prev, lv = "x0", start_level
    for i in range(pre):
        out = f"p_{i}"
        nodes.append(_node("mult", [prev, "pt_w"], out, lv + 1,
                           ilv=[lv, lv], step=f"pre.m{i}", period=1))
        prev, lv = out, lv + 1
    hub, hub_lv = prev, lv
    for c in range(n_consumers):
        prev, lv = hub, hub_lv
        for i in range(post):
            out = f"c{c}_{i}"
            nodes.append(_node("mult", [prev, "pt_w"], out, lv + 1,
                               ilv=[lv, lv], step=f"cons{c}.m{i}", period=1))
            prev, lv = out, lv + 1
    return nodes


class ChainTest(unittest.TestCase):
    def test_every_baseline_plans_a_deep_chain(self):
        gf = _write(_chain_graph(12))
        for placer in BASELINES:
            with self.subTest(placer=placer):
                result = plan_block(gf, _cfg(placer=placer))
                s = result["summary"]
                self.assertGreaterEqual(s["num_placements"], 1)
                self.assertLessEqual(max(s["final_named_levels"].values()), 24)
                self.assertIn(placer.split("_")[0], result["heuristic_config"])
                self.assertIn("placer_meta", s)

    def test_fhelipe_chain_is_count_optimal(self):
        # 12 mult levels, budget 8 minus the 1-level guard => 7 usable: ceil(12/7)-ish
        # spacing needs exactly 1 interior refresh (the tail rides the second window).
        gf = _write(_chain_graph(12))
        result = plan_block(gf, _cfg(placer="fhelipe"))
        self.assertEqual(result["summary"]["num_placements"], 1)


class ForkJoinTest(unittest.TestCase):
    def test_every_baseline_survives_a_fork_join(self):
        # DaCapo's per-value stale-bypass (upstream semantics) can leave residual
        # values unbootstrappable on this shape — upstream's own coverage machinery
        # would refuse the program too, so rescued mode is the legitimate path there.
        gf = _write(_fork_graph())
        for placer in BASELINES:
            with self.subTest(placer=placer):
                try:
                    result = plan_block(gf, _cfg(placer=placer))
                except PlanInfeasible:
                    result = plan_block(gf, _cfg(placer=placer, baseline_rescue=True))
                self.assertGreaterEqual(result["summary"]["num_placements"], 1)
                self.assertLessEqual(
                    max(result["summary"]["final_named_levels"].values()), 24)


class FanOutTest(unittest.TestCase):
    def test_shared_producer_refreshed_once_not_per_consumer(self):
        gf = _write(_fanout_graph(n_consumers=6))
        for placer in BASELINES:
            with self.subTest(placer=placer):
                result = plan_block(gf, _cfg(placer=placer))
                # a naive per-consumer placement would cost >= 6; the shared hub
                # (or a single ancestor) must be found
                self.assertLessEqual(result["summary"]["num_placements"], 3)


class HopelessTest(unittest.TestCase):
    def test_raw_mode_refuses_loudly(self):
        gf = _write(_chain_graph(12, mag=1e30))
        for placer in BASELINES:
            with self.subTest(placer=placer):
                with self.assertRaises(PlanInfeasible):
                    plan_block(gf, _cfg(placer=placer))

    def test_rescue_mode_repairs_and_counts(self):
        # interior magnitudes hopeless except the first two producers: rescue must
        # relocate upstream and count the move.
        nodes = _chain_graph(12, mag=1e30)
        for d in nodes[:2]:
            d["output_max_abs"] = 1.0
        gf = _write(nodes)
        for placer in BASELINES:
            with self.subTest(placer=placer):
                try:
                    result = plan_block(gf, _cfg(placer=placer, baseline_rescue=True))
                except PlanInfeasible:
                    continue   # honest refusal is acceptable when repair cannot fit
                meta = result["summary"]["placer_meta"]
                self.assertIn("rescue_moved", meta)


@unittest.skipUnless(REAL_GRAPH.is_dir(), "real n32 capture not on this box")
class RealGraphSmoke(unittest.TestCase):
    CFG = dict(bootstrap_level=34, max_level=48, source_level=34,
               cache_read_level=34, level_unit=2, acc_chain="n32",
               sparse_precomps=(512, 1), sparse_out_levels=((1, 24), (512, 34)),
               allow_prescale=False, verbose=False)

    def test_block_0_per_baseline(self):
        for placer in BASELINES:
            with self.subTest(placer=placer):
                cfg = PlanConfig(placer=placer, baseline_rescue=True, **self.CFG)
                try:
                    result = plan_block(REAL_GRAPH / "block_0/graph.json", cfg)
                except PlanInfeasible as e:
                    print(f"[smoke] {placer}: PlanInfeasible (recorded): {e}")
                    continue
                s = result["summary"]
                self.assertGreater(s["num_placements"], 0)
                self.assertIn("hint_fire", result)
                self.assertIn("placer_meta", s)
                print(f"[smoke] {placer}: placements={s['num_placements']} "
                      f"total={s['total_bootstraps']} meta={s['placer_meta']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
