"""FORBID_STEPS `.sq`: each softmax reciprocal refine round's numerator square, `square(x + conjugate(x))` inside a
`*.softmax` step (attention._softmax_recip: the round's 2 Re of the previous probabilities, squared), becomes the
sub-step `<step>.sq`, which the forbid list then keeps refreshes off.

Why: the squares feed the head sum directly. On flat heads (Llama decode at 512 keys) they sit ~300x below a
refresh's band, so a refresh there leaves every slot an absolute floor error that the head sum adds up over the keys
and the reciprocal passes on; the placer prices a refresh against the slot's maximum, not against the sum. The head
sums themselves are never refreshed for the same reason. The capture does not scope the square (a sub-step there
would shift every later scoped name of the step, so plans captured before would not bind): the planner derives the
sub-step from the graph instead, only when the forbid list names `.sq`."""
from __future__ import annotations

SQ = ".sq"


def tag_round_squares(nodes: list[dict]) -> int:
    """Move every round square of `nodes` (graph.json node dicts, edited in place) to the sub-step `<step>.sq`;
    returns how many were tagged."""
    prod = {n["output"]: n for n in nodes if n.get("output")}
    k = 0
    for n in nodes:
        if n.get("op_type") != "square" or not n.get("step", "").endswith(".softmax") or not n.get("inputs"):
            continue
        a = prod.get(n["inputs"][0])
        if a is None or a.get("op_type") != "add" or len(a.get("inputs", ())) != 2:
            continue
        x, y = a["inputs"]
        cx, cy = prod.get(x), prod.get(y)
        if (cy and cy.get("op_type") == "conjugate" and cy["inputs"][0] == x) or \
           (cx and cx.get("op_type") == "conjugate" and cx["inputs"][0] == y):
            n["step"] = n["step"] + SQ
            k += 1
    return k
