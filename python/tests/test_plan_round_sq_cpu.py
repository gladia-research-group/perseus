"""FORBID_STEPS `.sq` (plan/placer/round_sq): a softmax reciprocal refine round's numerator square, square(x + conj x)
in a `*.softmax` step, is planned as the sub-step `<step>.sq`, so no refresh lands on it; other squares and steps are
left alone, and without `.sq` in the forbid list the graph is planned as captured."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from perseus.plan.placer import round_sq
from perseus.plan.placer.planner import PlanConfig, plan_block


def _node(op, inputs, output, out_level, step, ilv=None, max_abs=1.0):
    return {"op_type": op, "inputs": inputs, "input_levels": ilv if ilv is not None else [-1] * len(inputs),
            "output": output, "output_level": out_level, "output_noise_level": 1, "step": step,
            "output_max_abs": max_abs, "pack_period": 1}


def _graph(n_after=9, start=16):
    """x0 -> conj -> x0 + conj -> square (blk0.softmax) -> a chain of products deep enough to need refreshes; plus an
    exp-style square (of a product, not of 2 Re) in the same step and a 2 Re square outside softmax steps."""
    lv = start
    nodes = [
        _node("conjugate", ["x0"], "c0", lv, "blk0.softmax", ilv=[lv]),
        _node("add", ["x0", "c0"], "re0", lv, "blk0.softmax", ilv=[lv, lv]),
        _node("square", ["re0"], "sq0", lv + 1, "blk0.softmax", ilv=[lv]),
    ]
    prev, lv = "sq0", lv + 1
    for i in range(n_after):
        nodes.append(_node("mult", [prev, "pt_w"], f"m{i}", lv + 1, "blk0.softmax", ilv=[lv, lv]))
        prev, lv = f"m{i}", lv + 1
    nodes += [
        _node("square", [prev], "sq_exp", lv + 1, "blk0.softmax", ilv=[lv]),
        _node("conjugate", ["x1"], "c1", start, "blk0.ln_2", ilv=[start]),
        _node("add", ["c1", "x1"], "re1", start, "blk0.ln_2", ilv=[start, start]),
        _node("square", ["re1"], "sq_ln", start + 1, "blk0.ln_2", ilv=[start]),
    ]
    return nodes


class TagTest(unittest.TestCase):
    def test_tags_only_the_round_square(self):
        nodes = _graph()
        self.assertEqual(round_sq.tag_round_squares(nodes), 1)
        steps = {n["output"]: n["step"] for n in nodes}
        self.assertEqual(steps["sq0"], "blk0.softmax.sq")
        self.assertEqual(steps["sq_exp"], "blk0.softmax")     # the square of a product (exp's doubling)
        self.assertEqual(steps["sq_ln"], "blk0.ln_2")          # 2 Re squared outside a softmax step
        self.assertEqual(steps["re0"], "blk0.softmax")         # the 2 Re itself stays refreshable

    def test_conjugate_on_either_side(self):
        nodes = _graph()
        add = next(n for n in nodes if n["output"] == "re0")
        add["inputs"] = ["c0", "x0"]
        self.assertEqual(round_sq.tag_round_squares(nodes), 1)


class PlanTest(unittest.TestCase):
    def _plan(self, forbid):
        d = Path(tempfile.mkdtemp())
        gf = d / "graph.json"
        # the square enters one level below the chain's top: the products after it need a refresh at or before it
        gf.write_text(json.dumps({"version": 1, "nodes": _graph(n_after=4, start=21)}), encoding="utf-8")
        cfg = PlanConfig(bootstrap_level=16, max_level=22, source_level=16, cache_read_level=17, verbose=False,
                         forbid_steps=forbid)
        return plan_block(gf, cfg), json.loads(gf.read_text())

    @staticmethod
    def _placed(doc):
        return {p["target_var"] for p in doc["placements"]}

    def test_no_refresh_on_the_round_square(self):
        doc, raw = self._plan((".sq",))
        self.assertNotIn("sq0", self._placed(doc))
        self.assertIn("re0", self._placed(doc))                    # the refresh moves before the square
        self.assertIn("sq_ln", self._placed(doc))                  # squares outside softmax steps unaffected
        self.assertEqual(next(n for n in raw["nodes"] if n["output"] == "sq0")["step"], "blk0.softmax",
                         "the graph file on disk is not rewritten")

    def test_without_sq_the_graph_plans_as_captured(self):
        nodes = _graph()
        before = copy.deepcopy(nodes)
        doc, _ = self._plan(())
        self.assertEqual(nodes, before)
        self.assertIn("sq0", self._placed(doc))                    # the control: the planner's choice is the square


if __name__ == "__main__":
    unittest.main()
