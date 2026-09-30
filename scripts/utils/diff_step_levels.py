"""Diff PER-STEP LEVEL CONSUMPTION between two captured graphs.

Built to answer: why does an n32 (composite d=2) block burn ~2x the CKKS levels of the
same n64 block, doing 99 bootstraps where n64 does 39?

Levels in a capture are PRIME-granular, so a d=2 chain moves in steps of 2 for the SAME
logical operation. Everything below is therefore reported in CKKS LEVELS (primes / d) so
the two chains are directly comparable — an op that legitimately consumes one level shows
1 on both. A step that shows 2 on n32 and 1 on n64 is rescaling twice per level, which is
the defect this script exists to find.

  python scripts/utils/diff_step_levels.py <graph_a.json> <d_a> <graph_b.json> <d_b>

e.g. .cache/graph_n32_vit_blk0/block_0/graph.json 2 \
     .cache/graph_vit_heat_80/block_0/graph.json  1
"""
import json
import sys
from collections import defaultdict


def load(path, d):
    nodes = json.load(open(path))["nodes"]
    # per step: total CKKS levels consumed, op count, bootstrap count
    cons = defaultdict(float)
    ops = defaultdict(int)
    bts = defaultdict(int)
    for n in nodes:
        step = n.get("step") or "<none>"
        op = str(n.get("op_type", ""))
        ops[step] += 1
        if "bootstrap" in op:
            bts[step] += 1
            continue                      # a bootstrap RESETS level; not consumption
        ins = [l for l in (n.get("input_levels") or []) if isinstance(l, int) and l >= 0]
        out = n.get("output_level")
        if not ins or not isinstance(out, int) or out < 0:
            continue
        delta = (out - max(ins)) / float(d)   # primes -> CKKS levels
        if delta > 0:
            cons[step] += delta
    return cons, ops, bts


def leaf(step, depth=3):
    """Collapse to the last `depth` scope components so the two chains group alike."""
    return ":".join(step.split(":")[-depth:])


def agg(cons, ops, bts, depth):
    c, o, b = defaultdict(float), defaultdict(int), defaultdict(int)
    for k, v in cons.items():
        c[leaf(k, depth)] += v
    for k, v in ops.items():
        o[leaf(k, depth)] += v
    for k, v in bts.items():
        b[leaf(k, depth)] += v
    return c, o, b


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(2)
    pa, da, pb, db = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    ca, oa, ba = agg(*load(pa, da), depth=3)
    cb, ob, bb = agg(*load(pb, db), depth=3)

    print(f"A = {pa}  (d={da})")
    print(f"B = {pb}  (d={db})")
    print(f"{'levels A':>9} {'levels B':>9} {'ratio':>6} {'btsA':>5} {'btsB':>5}  step")
    print("-" * 100)

    rows = []
    for step in set(ca) | set(cb):
        la, lb = ca.get(step, 0.0), cb.get(step, 0.0)
        if la < 0.5 and lb < 0.5:
            continue
        ratio = (la / lb) if lb > 0 else float("inf")
        rows.append((la - lb, la, lb, ratio, ba.get(step, 0), bb.get(step, 0), step))
    rows.sort(reverse=True)                      # biggest EXCESS first — the suspects

    for _, la, lb, ratio, bta, btb, step in rows[:30]:
        r = "inf" if ratio == float("inf") else f"{ratio:.2f}"
        print(f"{la:9.1f} {lb:9.1f} {r:>6} {bta:5d} {btb:5d}  {step}")

    print("-" * 100)
    print(f"{sum(ca.values()):9.1f} {sum(cb.values()):9.1f} "
          f"{sum(ca.values())/max(sum(cb.values()),1e-9):6.2f} "
          f"{sum(ba.values()):5d} {sum(bb.values()):5d}  TOTAL")


if __name__ == "__main__":
    main()
