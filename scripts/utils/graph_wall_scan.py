#!/usr/bin/env python3
"""Scan a captured graph for EvalMod-wall breaches at bootstrap inputs.

The CKKS bootstrap (EvalMod) is only faithful while |value| stays inside its
window (~10 for this chain). A ciphertext that enters a bootstrap above the wall
comes out smeared — and per the LN-envelope lesson a SINGLE out-of-wall slot
poisons that bootstrap for every slot, so one bad node destroys the block.

Captures record `output_max_abs` per produced var, so we can ask directly: which
bootstrap consumed a var whose magnitude was already past the wall, and in which
block does that first happen?

Usage:
  python scripts/utils/graph_wall_scan.py .cache/graph_bert_base [--wall 10]
  python scripts/utils/graph_wall_scan.py .cache/graph_vit_base   # reference
"""
import argparse
import glob
import json
import os
import re


def block_index(path):
    m = re.search(r"block_(\d+)", path)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph_dir")
    ap.add_argument("--wall", type=float, default=10.0,
                    help="EvalMod magnitude window (chain default ~10)")
    ap.add_argument("--top", type=int, default=5, help="worst offenders per block")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.graph_dir, "**", "block_*", "graph.json"),
                             recursive=True) +
                   glob.glob(os.path.join(args.graph_dir, "block_*", "graph.json")),
                   key=block_index)
    if not files:
        raise SystemExit(f"no block_*/graph.json under {args.graph_dir}")

    print(f"{'blk':>4} {'nodes':>7} {'bts':>5} {'max|out|':>11} {'max_bts_in':>11} "
          f"{'>wall':>6}  worst bootstrap inputs")
    first_bad = None
    worst_overall = (0.0, None)
    for f in sorted(set(files), key=block_index):
        b = block_index(f)
        nodes = json.load(open(f))["nodes"]
        mag = {n["output"]: n["output_max_abs"]
               for n in nodes if "output_max_abs" in n and n.get("output")}
        bts = [n for n in nodes if "bootstrap" in n.get("op_type", "")]
        # magnitude each bootstrap consumed (max over its inputs that we know)
        breaches, all_bts_in = [], []
        for n in bts:
            ins = [(i, mag[i]) for i in n.get("inputs", []) if i in mag]
            if not ins:
                continue
            v, m = max(ins, key=lambda t: t[1])
            all_bts_in.append((m, v, n.get("step", "?"), n["op_type"]))
            if m > args.wall:
                breaches.append((m, v, n.get("step", "?"), n["op_type"]))
        breaches.sort(reverse=True)
        all_bts_in.sort(reverse=True)
        overall = max(mag.values()) if mag else 0.0
        max_in = all_bts_in[0][0] if all_bts_in else 0.0
        if max_in > worst_overall[0]:
            worst_overall = (max_in, (b,) + all_bts_in[0][1:])
        worst = "  ".join(f"{v}={m:.1f}@{s.split('.')[-1]}" for m, v, s, _ in breaches[:args.top])
        print(f"{b:>4} {len(nodes):>7} {len(bts):>5} {overall:>11.1f} {max_in:>11.1f} "
              f"{len(breaches):>6}  {worst}")
        if breaches and first_bad is None:
            first_bad = (b, breaches[0])

    print()
    m, info = worst_overall
    if info:
        b, v, s, op = info
        print(f"WORST bootstrap input overall: |abs|={m:.1f} in block {b} ({op} <- {v})  step={s}")
    if first_bad:
        b, (m, v, s, op) = first_bad
        print(f"first input over |abs|={args.wall}: block {b} {op} <- {v} at {m:.1f}")


if __name__ == "__main__":
    main()
