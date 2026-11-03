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
NUM = ".num"


def tag_round_squares(nodes: list[dict]) -> int:
    """Move every round square of `nodes` (graph.json node dicts, edited in place) to the sub-step `<step>.sq`;
    returns how many were tagged."""
    prod = {n["output"]: n for n in nodes if n.get("output")}
    k = 0
    for n in nodes:
        if n.get("op_type") != "square" or not n.get("step", "").endswith(".softmax") or not n.get("inputs"):
            continue
        a = prod.get(n["inputs"][0])
        if a is not None and a.get("op_type") in ("deliberate_bootstrap", "bootstrap") and a.get("inputs"):
            a = prod.get(a["inputs"][0])        # SoftmaxCfg.gs_num_refresh: the square reads the refreshed 2 Re
        if a is None or a.get("op_type") != "add" or len(a.get("inputs", ())) != 2:
            continue
        x, y = a["inputs"]
        cx, cy = prod.get(x), prod.get(y)
        if (cy and cy.get("op_type") == "conjugate" and cy["inputs"][0] == x) or \
           (cx and cx.get("op_type") == "conjugate" and cx["inputs"][0] == y):
            n["step"] = n["step"] + SQ
            k += 1
    return k


def _is_2re(n: dict, prod: dict) -> bool:
    """n = x + conjugate(x) (im_cleanse: 2 Re x)."""
    if n.get("op_type") != "add" or len(n.get("inputs", ())) != 2:
        return False
    x, y = n["inputs"]
    cx, cy = prod.get(x), prod.get(y)
    return bool((cy and cy.get("op_type") == "conjugate" and cy["inputs"][0] == x) or
                (cx and cx.get("op_type") == "conjugate" and cx["inputs"][0] == y))


def tag_numerators(nodes: list[dict], below: float | None = None) -> int:
    """FORBID_STEPS `.num`, the round-square rule extended to every value that carries the softmax's per-key
    numerators: from round 0's numerator (2 Re of the exp's last squaring) and each refine round's square on, every
    value of the `*.softmax` step computed from them, up to a key reduction (rotate_sum: a head sum is a sum, its
    refresh is relative to it). Under SoftmaxCfg.gs_num the numerator rides the Goldschmidt recursion's packed state
    (P = G D - i G mu N), whose refresh would leave each key's numerator an absolute floor error that the head sums and
    P.V add up over the keys (Llama decode, 512 keys: KL 3.9e-3 vs 5e-6); off gs_num the same values are the round's
    probabilities y = z R, by the same mechanism. Values before the squarings (the exp series) stay refreshable: the
    squarings shrink a relative error's floor with the value. Tags those values `<step>.num` (the round squares
    `<step>.sq` as tag_round_squares); returns how many were tagged `.num`.

    `below` (the refresh band's floor, plan_block: band_lo(cf_min) / mag_safety): only the numerator values whose
    captured magnitude sits under it are kept off refreshes. Inside the band a refresh's floor error is below the
    value's own scale, and under gs_num the recursion is longer than one refresh window, so it must be refreshed
    somewhere: the rule keeps it where the state has grown into the band (Llama decode: the packed state starts at
    5e-4 and grows ~4x per Goldschmidt step; GPT-2's smallest refreshed state is 0.024, so its plans are unchanged)."""
    tag_round_squares(nodes)
    prod = {n["output"]: n for n in nodes if n.get("output")}
    taint: set = set()
    k = 0
    for n in nodes:
        st = n.get("step", "")
        base = st[:-len(SQ)] if st.endswith(SQ) else st
        if not base.endswith(".softmax") or not n.get("output"):
            continue
        root = (_is_2re(n, prod) and any((prod.get(i) or {}).get("op_type") == "square" for i in n["inputs"])) \
            or st.endswith(SQ)
        hot = root or (n.get("op_type") != "rotate_sum" and any(i in taint for i in n.get("inputs", ())))
        if not hot:
            continue
        taint.add(n["output"])
        mag = n.get("output_max_abs")
        if below is not None and mag is not None and mag >= below:
            continue
        if not st.endswith(SQ):
            n["step"] = st + NUM
            k += 1
    return k
