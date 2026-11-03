"""FORBID_STEPS `.num` with PLAN_NUM_BAND=1 (plan/placer/round_sq.tag_numerators `below`): only the softmax values
that carry per-key numerators AND sit under the refresh band at the smallest CF are kept off refreshes; numerator values
inside the band stay refreshable, values before the numerator's root and outside softmax steps are never tagged."""
from __future__ import annotations

import copy
import unittest

from perseus.plan import btserr
from perseus.plan.placer import round_sq


def _node(op, inputs, output, step="blk0.softmax", max_abs=1.0):
    return {"op_type": op, "inputs": inputs, "output": output, "step": step, "output_max_abs": max_abs}


def _graph():
    """exp -> square -> 2 Re (round 0's numerator) -> a gs_num packed state growing x4 per step from 5e-4 -> a key
    reduction (rotate_sum) -> a value after it; plus a 2 Re of a square in a LayerNorm step."""
    nodes = [
        _node("mult", ["s", "pt"], "e0", max_abs=0.5),              # the exp series' value: before the root
        _node("square", ["e0"], "e1", max_abs=0.25),
        _node("conjugate", ["e1"], "e1c", max_abs=0.25),
        _node("add", ["e1", "e1c"], "n0", max_abs=0.5),            # 2 Re of the last squaring: the root
    ]
    prev, mag = "n0", 5e-4
    for i in range(6):
        nodes.append(_node("mult", [prev, "g"], f"p{i}", max_abs=mag))
        prev, mag = f"p{i}", mag * 4
    nodes += [
        _node("rotate_sum", [prev], "hs", max_abs=1.0),            # a head sum: the taint stops
        _node("mult", ["hs", "pt"], "after", max_abs=1e-4),
        _node("square", ["x"], "lsq", step="blk0.ln_2", max_abs=1e-4),
        _node("conjugate", ["lsq"], "lsqc", step="blk0.ln_2", max_abs=1e-4),
        _node("add", ["lsq", "lsqc"], "lre", step="blk0.ln_2", max_abs=1e-4),
    ]
    return nodes


def _tagged(nodes):
    return {n["output"] for n in nodes if n["step"].endswith(round_sq.NUM)}


class NumBand(unittest.TestCase):
    def test_all_numerators_without_band(self):
        nodes = copy.deepcopy(_graph())
        round_sq.tag_numerators(nodes)
        self.assertEqual(_tagged(nodes), {"n0", "p0", "p1", "p2", "p3", "p4", "p5"})

    def test_only_under_the_band(self):
        below = btserr.band(2)[0] / 2.0          # band_lo(cf_min = 2) / mag_safety 2: 0.006
        self.assertAlmostEqual(below, 0.006)
        nodes = copy.deepcopy(_graph())
        n = round_sq.tag_numerators(nodes, below)
        # 5e-4, 2e-3 under the floor; 8e-3 and up inside the band; the root n0 (0.5) inside the band
        self.assertEqual(_tagged(nodes), {"p0", "p1"})
        self.assertEqual(n, 2)

    def test_never_outside_the_numerator(self):
        nodes = copy.deepcopy(_graph())
        round_sq.tag_numerators(nodes, 1.0)
        tagged = _tagged(nodes)
        for v in ("e0", "e1", "e1c", "hs", "after", "lsq", "lsqc", "lre"):
            self.assertNotIn(v, tagged)


if __name__ == "__main__":
    unittest.main()
