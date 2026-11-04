"""FORBID_STEPS `.kred` (plan/placer/round_sq.tag_key_reduced): in a `*.softmax_v` step the attention output from the
P.V product on (the product of two ciphertexts, its key reduction and every later value of the step) becomes
`<step>.kred`; the packed probabilities before the product stay refreshable, a product with a plaintext starts
nothing, and other steps are never tagged."""
from __future__ import annotations

import copy
import unittest

from perseus.plan.placer import round_sq


def _node(op, inputs, output, step="blk0.softmax_v"):
    return {"op_type": op, "inputs": inputs, "output": output, "step": step}


def _graph():
    """P (the softmax's result) -> its packing in softmax_v -> V lanes summed (.gp) -> P.V -> the key reduction ->
    2 Re -> a plaintext product; plus a plaintext product before P.V and a value in another step."""
    return [
        _node("mult_i", ["d"], "p", step="blk0.softmax"),
        _node("add", ["p", "const(0)"], "v1"),
        _node("rotate", ["v1", "rot(256)"], "v2"),
        _node("mult_i", ["v2"], "v3"),
        _node("sub_ct", ["v1", "v3"], "v4"),
        _node("mult", ["v4", "pt_7"], "v4s"),                     # a plaintext product: not P.V
        _node("add", ["k0", "const(0)"], "g1", step="blk0.softmax_v.gp"),
        _node("mult", ["g1", "v4"], "v5"),                         # P.V
        _node("add", ["v5", "const(0)"], "v6"),
        _node("rotate_sum", ["v6", "rot(1,2,3)"], "v7"),
        _node("conjugate", ["v7"], "v8"),
        _node("add", ["v7", "v8"], "v9"),
        _node("mult", ["v9", "pt_9"], "v10"),
        _node("add", ["v10", "x"], "o1", step="blk0.attn_out"),
    ]


def _tagged(nodes):
    return {n["output"] for n in nodes if n["step"].endswith(round_sq.KRED)}


class KeyReduced(unittest.TestCase):
    def test_from_the_pv_product_on(self):
        nodes = copy.deepcopy(_graph())
        n = round_sq.tag_key_reduced(nodes)
        self.assertEqual(_tagged(nodes), {"v5", "v6", "v7", "v8", "v9", "v10"})
        self.assertEqual(n, 6)

    def test_before_the_product_and_other_steps_untouched(self):
        nodes = copy.deepcopy(_graph())
        round_sq.tag_key_reduced(nodes)
        tagged = _tagged(nodes)
        for v in ("p", "v1", "v2", "v3", "v4", "v4s", "g1", "o1"):
            self.assertNotIn(v, tagged)

    def test_forbid_matches_the_sub_step(self):
        nodes = copy.deepcopy(_graph())
        round_sq.tag_key_reduced(nodes)
        steps = {n["output"]: n["step"] for n in nodes}
        self.assertEqual(steps["v7"], "blk0.softmax_v.kred")
        self.assertEqual(steps["g1"], "blk0.softmax_v.gp")


if __name__ == "__main__":
    unittest.main()
