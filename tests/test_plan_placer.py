"""Smoke tests for the planner (perseus/plan/placer).

Runs against a small synthetic graph always, and against a real 32-bit capture: the
single-block fixture under tests/fixtures, or whatever PERSEUS_REAL_GRAPH names. The
synthetic cases pin the behaviours the planner is built on:

  * reactive bootstraps are erased unconditionally — no keep-list survives;
  * an unrefreshable-but-needed site is a LOUD PlanInfeasible, never a silent drop;
  * per-site refreshes carry per-site OUTPUT levels and the sim uses them.

  python tests/test_plan_placer.py
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from perseus.plan.placer.ir import Graph
from perseus.plan.placer.place import PlanInfeasible, erase_reactive_bootstraps
from perseus.plan.placer.planner import PlanConfig, PlanDiagnostics, plan_block

REAL_GRAPH = Path(os.environ.get(
    "PERSEUS_REAL_GRAPH",
    Path(__file__).resolve().parent / "fixtures" / "real_graph" / "n32_gpt2_base"))


def _node(op, inputs, output, out_level, max_abs=1.0, step="s", ilv=None, deg=1,
          period=None):
    return {
        "op_type": op, "inputs": inputs,
        "input_levels": ilv if ilv is not None else [-1] * len(inputs),
        "output": output, "output_level": out_level, "output_noise_level": deg,
        "step": step, "output_max_abs": max_abs,
        **({"pack_period": period} if period is not None else {}),
    }


def _chain_graph(n_mults=8, mag=1.0, start_level=16):
    """x0 -> mult chain, each mult costs 1 level; magnitudes constant."""
    nodes = []
    lv = start_level
    prev = "x0"
    for i in range(n_mults):
        out = f"v_{i}"
        nodes.append(_node("mult", [prev, "pt_w"], out, lv + 1, max_abs=mag,
                           ilv=[lv, lv], step=f"chain.m{i}", period=1))
        prev, lv = out, lv + 1
    return nodes


class EraseTest(unittest.TestCase):
    def test_all_reactive_bootstraps_erased_no_keep_list(self):
        nodes = _chain_graph(2)
        nodes.insert(1, _node("auto_bootstrap", ["v_0"], "v_0_ab", 16,
                              max_abs=5e4, step="chain.keepme.goldschmidt"))
        nodes[2]["inputs"] = ["v_0_ab", "pt_w"]
        g = Graph.from_nodes(nodes)
        g2, rep = erase_reactive_bootstraps(g)
        self.assertEqual(rep.n_erased, 1)
        # the consumer was rewired to the erased bootstrap's input, whatever the step is
        # named and however large |m| is.
        (m1,) = [n for n in g2.nodes if n.output == "v_1"]
        self.assertIn("v_0", m1.cipher_inputs)

    def test_deliberate_bootstraps_survive(self):
        nodes = _chain_graph(2)
        nodes.append(_node("deliberate_bootstrap", ["v_1"], "v_1_db", 16,
                           step="lm_head.tail"))
        g2, rep = erase_reactive_bootstraps(Graph.from_nodes(nodes))
        self.assertEqual(rep.n_deliberate_kept, 1)
        self.assertTrue(any(n.is_deliberate_bts for n in g2.nodes))


class PlanBlockTest(unittest.TestCase):
    def _write(self, nodes) -> Path:
        d = Path(tempfile.mkdtemp(prefix="perseus_plan_placer_"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        (d / "graph.json").write_text(json.dumps({"version": 1, "nodes": nodes}))
        return d / "graph.json"

    CFG = PlanConfig(bootstrap_level=16, max_level=24, source_level=16,
                     cache_read_level=17, verbose=False, forbid_steps=())

    def test_long_chain_gets_refreshes_without_a_keep_list(self):
        # 12 mults against an 8-level budget: impossible without interior refreshes.
        gf = self._write(_chain_graph(12))
        result = plan_block(gf, self.CFG)
        self.assertGreaterEqual(result["summary"]["num_placements"], 1)
        self.assertLessEqual(
            max(result["summary"]["final_named_levels"].values()), 24)

    def test_unrefreshable_chain_is_a_loud_error_not_a_plan(self):
        # every interior site far past the CF=9 prescale reach => nothing cuttable.
        gf = self._write(_chain_graph(12, mag=1e30))
        with self.assertRaises(PlanInfeasible):
            plan_block(gf, self.CFG)

    def test_per_site_output_levels_reach_the_plan(self):
        # a sparse-routed site (period 1, s=1 precomp, measured out level 12 => consumed
        # offset -4 relative to bts_level 16) must let a LONGER chain fit than dense.
        cfg = PlanConfig(bootstrap_level=16, max_level=24, source_level=16,
                         cache_read_level=17, verbose=False, forbid_steps=(),
                         sparse_precomps=(1,), sparse_out_levels=((1, 12),))
        gf = self._write(_chain_graph(12))
        result = plan_block(gf, cfg)
        self.assertTrue(result.get("sparse_slots"),
                        "period-1 sites with an s=1 precomp should route sparse")
        # the quality block must SHOW the non-dense landing
        hist = result["summary"]["bts_quality"]["out_level_histogram"]
        self.assertIn("12", hist)


class CfClampTest(unittest.TestCase):
    """The cf_max ceiling is counted when the search HITS it, and never enters the plan.

    choose_site is an argmin over cf_min..cf_max, so a site at cf_max is either the
    cheapest CF or merely the last one priced; the counter (btserr.ceiling_binds: the same
    search with cf_max+1 would move up) separates the two. On this fixture the 12-mult
    chain against the 8-level budget places exactly ONE site (v_8), so the counter moves
    0 -> 1: |m|=1 centres a band (cf 7); |m|=400 (x1.2 kappa = 480) sits at the top of the
    cf=14 band, where cf 15 would price 2.9e-3 against 6.4e-3.
    """
    _write = PlanBlockTest._write
    # The clamp fixture was built on the pre-liberation window (cf_max=14, kappa=1.2); the
    # shipping defaults are cf_max=20, kappa=2, under which nothing on it clamps.
    CFG = dataclasses.replace(PlanBlockTest.CFG, cf_max=14, mag_safety=1.2)

    def _plan(self, mag, cfg=None):
        diag = PlanDiagnostics()
        result = plan_block(self._write(_chain_graph(12, mag=mag)), cfg or self.CFG,
                            diagnostics=diag)
        return result, diag.cf_clamp

    def test_inflated_magnitudes_clamp_at_the_ceiling(self):
        _, calm = self._plan(1.0)
        self.assertEqual(calm.num_clamped, 0)
        self.assertEqual(calm.num_pinned, 0)
        self.assertEqual(calm.cf_max, 14)
        self.assertGreater(calm.num_sites, 0)
        result, hot = self._plan(400.0)
        self.assertGreater(hot.num_clamped, 0)
        self.assertEqual(hot.clamped, ("v_8",))
        self.assertLessEqual(hot.num_clamped, hot.num_pinned)
        self.assertLessEqual(hot.num_pinned, hot.num_sites)
        # the denominator is the set of sites that carry a CF in the plan
        hist = result["summary"]["bts_quality"]["cf_histogram"]
        self.assertEqual(hot.num_sites, sum(hist.values()))
        self.assertEqual(result["correction_factor"]["v_8"], 14)
        self.assertEqual(hot.clamped_missing_target, ())   # 6.4e-3 < err_target: INFO, not WARNING

    def test_report_is_written_to_summary_bts_quality(self):
        result, hot = self._plan(400.0)
        self.assertGreater(hot.num_clamped, 0)
        bq = result["summary"]["bts_quality"]
        self.assertEqual(bq["num_cf_clamped"], hot.num_clamped)
        self.assertEqual(bq["num_at_cf_max"], hot.num_pinned)
        self.assertEqual(bq["cf_clamped_sites"], list(hot.clamped))
        self.assertEqual(bq["cf_clamp_probe_cf"], hot.cf_max + 1)
        self.assertEqual(bq["cf_table_max"], 20)          # the n32 table's last measured CF
        # additive keys only: a plan file that gained them is still equivalent to one that
        # did not (the baseline contract of scripts/utils/plan_equiv.py)
        json.dumps(result)
        # and the document is the same whether or not the side channel was requested
        plain = plan_block(self._write(_chain_graph(12, mag=400.0)), self.CFG)
        self.assertEqual(json.dumps(plain, sort_keys=True), json.dumps(result, sort_keys=True))

    def test_clamp_is_one_log_line(self):
        cfg = dataclasses.replace(self.CFG, verbose=True)
        with self.assertLogs("perseus.plan.placer.planner", level="INFO") as cm:
            plan_block(self._write(_chain_graph(12, mag=400.0)), cfg)
        lines = [m for m in cm.output if "cf clamp:" in m]
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0], r"^INFO:.*\[plan\] cf clamp: [1-9]\d* of \d+ typed sites "
                                   r"at cf_max=14 \(\d+ pinned; 0 miss err_target=0\.01\): v_8$")
        # verbose=False and nothing missing the target: the diagnostic stays silent
        with self.assertLogs("perseus.plan.placer.planner", level="DEBUG") as cm:
            logging.getLogger("perseus.plan.placer.planner").debug("sentinel")   # assertLogs needs one
            plan_block(self._write(_chain_graph(12, mag=400.0)), self.CFG)
        self.assertFalse([m for m in cm.output if "cf clamp:" in m])

    def test_clamped_site_missing_target_is_a_warning(self):
        # band placement only (no prescale), |m|=1000: cf 14 is lossy (3.5e-2, priced by
        # the cut) and cf 15 would meet the target (9.4e-3) — the ceiling binds AND costs
        # the target, so this is the one case the line escalates, verbose or not
        cfg = dataclasses.replace(self.CFG, allow_prescale=False, verbose=False)
        with self.assertLogs("perseus.plan.placer.planner", level="WARNING") as cm:
            result, hot = self._plan(1000.0, cfg)
        self.assertEqual(hot.clamped_missing_target, ("v_8",))
        self.assertEqual(result["summary"]["bts_quality"]["num_sites_missing_target"], 1)
        lines = [m for m in cm.output if "cf clamp:" in m]
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0], r"^WARNING:.*cf clamp: 1 of 1 typed sites at cf_max=14 "
                                   r"\(1 pinned; 1 miss err_target=0\.01\): v_8$")


class HintPlacementTest(unittest.TestCase):
    """A placement ON a hint output refreshes UNCONDITIONALLY.

    The runtime runs maybe_apply_planned_bootstrap_after on a hint node's output
    whether or not the hint fired ("plans may cut a hint output",
    fideslib_wrapper.h). Simulating that as a pass-through predicts a STALER ct
    than the runtime produces; the level check tolerates it (it only refuses
    runtime levels DEEPER than predicted) and the first pre-encoded weight then
    dies with [plan_weight_error].
    """

    def _sim(self, placed):
        from perseus.plan.placer.sim import Budget, simulate
        # x0 -> 3 mults -> hint(threshold far above the trajectory: never fires)
        nodes = _chain_graph(3)
        nodes.append(_node("hint", ["v_2", "hint_lev(30)"], "h", 19,
                           ilv=[19, -1], step="s.h"))
        g = Graph.from_nodes(nodes)
        return simulate(
            g, bootstrap_level=16.0, budget=Budget(L=8.0, unit=1),
            seed_consumed=lambda v: 0.0, seed_deg=lambda v: 1,
            refreshed={"h": (0.0, 2)}, placed=placed)

    def test_unfired_hint_passes_through_without_a_placement(self):
        sim = self._sim(placed=set())
        self.assertFalse(any(sim.hint_fired.values()))
        self.assertEqual(sim.consumed["h"], sim.consumed["v_2"])
        self.assertGreater(sim.consumed["h"], 0.0)

    def test_unfired_hint_still_refreshes_when_placed(self):
        sim = self._sim(placed={"h"})
        self.assertFalse(any(sim.hint_fired.values()))   # the hint itself skips
        self.assertEqual(sim.consumed["h"], 0.0)         # the placement executes
        self.assertEqual(sim.deg["h"], 2)


class HintP2Test(unittest.TestCase):
    """P2 (no destructive refresh) covers fired hint sites, not only min-cut placements.

    A hint that fires on a 1e6-magnitude output would refresh at a correction factor
    no band covers, so checking only placer.placed would let it through silently.
    Hints and deliberate sites are a substantial share of a real plan's refreshes.
    """

    def _write(self, nodes) -> Path:
        d = Path(tempfile.mkdtemp(prefix="perseus_plan_placer_"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        (d / "graph.json").write_text(json.dumps({"version": 1, "nodes": nodes}))
        return d / "graph.json"

    # Pinned to the pre-liberation window: under the shipping cf_max=20 + prescale the
    # 1e6 hint output is plannable, so the destructive case needs cf_max=14 to arise.
    CFG = PlanConfig(bootstrap_level=16, max_level=24, source_level=16,
                     cache_read_level=17, verbose=False, forbid_steps=(),
                     cf_max=14, mag_safety=1.2)

    def _graph(self, mag):
        nodes = _chain_graph(3)
        # threshold 17 < 3 + 16: the hint fires; its output magnitude decides the refresh
        nodes.append(_node("hint", ["v_2", "hint_lev(17)"], "h", 19, max_abs=mag,
                           ilv=[19, -1], step="s.h"))
        nodes.append(_node("mult", ["h", "pt_w"], "v_3", 20, max_abs=mag,
                           ilv=[19, 19], step="chain.m3", period=1))
        return self._write(nodes)

    def test_destructive_hint_refresh_is_refused(self):
        with self.assertRaisesRegex(PlanInfeasible, r"P2 violated.*\(hint\)"):
            plan_block(self._graph(mag=1e6), self.CFG)

    def test_benign_hint_still_plans(self):
        result = plan_block(self._graph(mag=1.0), self.CFG)
        self.assertIn("placements", result)

class HintDissolutionTest(unittest.TestCase):
    """Planned mode must not depend on threshold-fired refreshes.

    A hint's firing level is decided at runtime, so a plan that leans on one carries a
    bootstrap whose depth is unknown until the GPU run. Dissolution pins every hint off
    and lets the min-cut place the refreshes instead.
    """

    def _graph_with_hint(self, threshold=19):
        """x0 -> 6 mults -> hint(threshold) -> 6 more mults. Budget L=8 at bootstrap
        level 16, so the 12-mult chain overruns unless SOMETHING refreshes, and the
        hint's threshold is low enough that it fires when left to itself."""
        nodes = _chain_graph(6)
        nodes.append(_node("hint", ["v_5", f"hint_lev({threshold})"], "h", 22,
                           ilv=[22, -1], step="s.mid"))
        lv, prev = 22, "h"
        for i in range(6, 12):
            nodes.append(_node("mult", [prev, "pt_w"], f"v_{i}", lv + 1,
                               ilv=[lv, lv], step=f"chain.m{i}", period=1))
            lv += 1
            prev = f"v_{i}"
        return nodes

    def _cfg(self, **kw):
        # Dissolution is not the planner default — this class tests it, so it opts in
        # explicitly; individual cases pass dissolve_hints=False for the control.
        kw.setdefault("forbid_steps", ())
        kw.setdefault("dissolve_hints", True)
        return PlanConfig(bootstrap_level=16, max_level=24, source_level=16,
                          cache_read_level=17, verbose=False, **kw)

    def _write(self, nodes):
        d = Path(tempfile.mkdtemp())
        f = d / "graph.json"
        f.write_text(json.dumps({"version": 1, "nodes": nodes}), encoding="utf-8")
        return f

    def test_dissolution_empties_hint_fire_and_places_instead(self):
        gf = self._write(self._graph_with_hint())
        bound = plan_block(gf, self._cfg(dissolve_hints=False))
        dissolved = plan_block(gf, self._cfg())
        self.assertTrue(bound["hint_fire"], "control must exercise the hint")
        self.assertEqual(dissolved["hint_fire"], [],
                         "a dissolvable hint must not fire in the plan")
        self.assertEqual(dissolved["summary"]["hints_retained"], [])
        # the refresh did not vanish — it became an ordinary, depth-checked placement
        self.assertGreater(dissolved["summary"]["num_placements"],
                           bound["summary"]["num_placements"])

    def test_a_forbidden_hint_is_retained_and_named(self):
        gf = self._write(self._graph_with_hint())
        result = plan_block(gf, self._cfg(forbid_steps=("s.mid",)))
        retained = result["summary"]["hints_retained"]
        self.assertEqual([r["var"] for r in retained], ["h"])
        self.assertEqual(retained[0]["reason"], "forbidden")
        self.assertEqual(retained[0]["step"], "s.mid")
        # retained means it keeps its threshold decision, i.e. it still fires
        self.assertIn("h", result["hint_fire"])

    def test_vetoed_hint_keeps_its_input_reachable_by_the_cut(self):
        """A vetoed hint must not starve the cut.

        The flow network must keep a hint node's input edge when the input sits above
        the hint THRESHOLD but hint_force={h: False} says the hint does not refresh it.
        Dropping the edge leaves the cut no way to reach the branch.
        """
        gf = self._write(self._graph_with_hint())
        result = plan_block(gf, self._cfg(dissolve_hints=False),
                            hint_force={"h": False})
        self.assertEqual(result["hint_fire"], [])
        self.assertTrue(result["placements"],
                        "with the hint vetoed the cut must place the refresh itself")


@unittest.skipUnless(REAL_GRAPH.is_dir(), "real n32 capture not on this box")
class RealGraphTest(unittest.TestCase):
    def test_block_0_plans_with_postconditions(self):
        cfg = PlanConfig(bootstrap_level=34, max_level=46, source_level=34,
                         cache_read_level=34, level_unit=2, acc_chain="n32",
                         sparse_precomps=(512, 1), verbose=False)
        result = plan_block(REAL_GRAPH / "block_0/graph.json", cfg)
        s = result["summary"]
        self.assertGreater(s["num_placements"], 0)
        self.assertEqual(s["bts_quality"]["num_sites_missing_target"], 0)
        self.assertLessEqual(max(s["final_named_levels"].values()), 46)
        # every placement carries a CF
        placed = {p["target_var"] for p in result["placements"]}
        self.assertTrue(placed.issubset(set(result["correction_factor"])))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class CutPricingKnobs(unittest.TestCase):
    """depth_weight / level_weight: off by default, and each form moves the cut when on.

    Off is what every shipped plan uses, so the default must reproduce the count-only cut
    exactly; tests/test_paper_plans.py is the end-to-end form of that guarantee.
    """

    GRAPH = REAL_GRAPH / "block_0" / "graph.json"

    def _cfg(self, **kw):
        return PlanConfig(bootstrap_level=36, source_level=36, cache_read_level=36,
                          level_unit=2, max_level=50, cf_max=20, mag_safety=2.0,
                          allow_prescale=False, acc_chain="n32", verbose=False, **kw)

    def _plan(self, **kw):
        if not self.GRAPH.is_file():
            self.skipTest("real-graph fixture not present")
        return plan_block(self.GRAPH, self._cfg(**kw))["summary"]

    def test_defaults_are_off(self):
        cfg = PlanConfig()
        for name in ("depth_weight", "level_weight"):
            self.assertEqual(getattr(cfg, name), 0.0, f"{name} must default to off")
        self.assertEqual(cfg.depth_form, "ratio")

    def test_zero_weight_reproduces_the_count_only_cut(self):
        base = self._plan()
        for kw in ({"depth_weight": 0.0}, {"level_weight": 0.0},
                   {"depth_weight": 0.0, "depth_form": "linear"}):
            self.assertEqual(self._plan(**kw)["num_placements"], base["num_placements"],
                             f"a zero weight changed the cut: {kw}")

    def test_each_form_is_reachable_and_prices_differently(self):
        base = self._plan()["num_placements"]
        moved = {f: self._plan(depth_weight=1.0, depth_form=f)["num_placements"]
                 for f in ("ratio", "linear", "ab", "ms")}
        moved["level"] = self._plan(level_weight=1.0)["num_placements"]
        self.assertTrue(any(v != base for v in moved.values()),
                        f"no pricing form moved the cut off {base}: {moved}")

    def test_placed_input_levels_is_reported(self):
        hist = self._plan()["bts_quality"].get("placed_input_level_hist")
        self.assertIsInstance(hist, dict)
        self.assertTrue(all(k.isdigit() for k in hist), hist)

    def test_ms_form_prefers_the_cheaper_route(self):
        """Pricing by measured latency plus the runway credit must not route MORE sites
        dense than the count-only cut does."""
        def dense_share(**kw):
            cfg = self._cfg(**kw)
            r = plan_block(self.GRAPH, cfg)
            placed = r["summary"]["num_placements"]
            routed = len(r.get("sparse_slots") or {})
            return placed - routed
        if not self.GRAPH.is_file():
            self.skipTest("real-graph fixture not present")
        self.assertLessEqual(dense_share(depth_weight=1.0, depth_form="ms"),
                             dense_share())
