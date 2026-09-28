#!/usr/bin/env python
"""Audit a run's actual refresh input levels against what the plan predicted.

A placement policy can only keep refreshes inside the envelope where the planner's level model
is right. It is not always right: a site the sim puts at 48 can meet the runtime at 50, and the
plan looks clean until `[bts_depth_error]` stops the run. This pairs the two so the gap is
visible before a run is trusted, and so a regression in the model has somewhere to show up.

  plan      summary.bts_quality.placed_input_levels  (absolute level, per placed var)
  run       [planted_bts] var=<v> ... in=<level>     (what the runtime actually met)

usage: bts_level_audit.py <plan-dir> <run.log> [--cap 48]
"""
import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict

RE_PLANT = re.compile(r"\[planted_bts\] (?:hint )?var=(\S+) .*?in=(\d+)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("plan_dir")
    ap.add_argument("run_log")
    ap.add_argument("--cap", type=float, default=48.0)
    a = ap.parse_args()

    predicted: dict[str, float] = {}
    blocks = 0
    for f in sorted(pathlib.Path(a.plan_dir).glob("block_*_placement.json")):
        q = (json.load(open(f)).get("summary", {}).get("bts_quality") or {})
        lv = q.get("placed_input_levels")
        if lv:
            blocks += 1
            # var names are per block, so keep the first (blocks are planned independently and
            # a later block reusing a name would otherwise overwrite an earlier prediction)
            for v, l in lv.items():
                predicted.setdefault(f"{f.name}:{v}", float(l))
    if not predicted:
        print(f"{a.plan_dir}: no placed_input_levels in any plan — regenerate the plan with a "
              f"planner that emits it, or there are no placed refreshes")
        return 2

    actual: dict[str, set] = defaultdict(set)
    for ln in open(a.run_log, encoding="utf-8", errors="replace"):
        m = RE_PLANT.search(ln)
        if m:
            actual[m.group(1)].add(int(m.group(2)))

    print(f"plan {a.plan_dir}: {len(predicted)} placed site(s) with a predicted level "
          f"across {blocks} block(s)")
    print(f"run  {a.run_log}: {len(actual)} distinct var(s) refreshed, "
          f"{sum(len(v) for v in actual.values())} distinct (var, level) pair(s)")

    over = sorted((v, sorted(ls)) for v, ls in actual.items() if max(ls) > a.cap)
    print(f"\nrefreshes the RUN started past level {a.cap:g}: {len(over)}")
    for v, ls in over[:20]:
        print(f"  {v:<12} actual in={ls}")

    by_var: dict[str, float] = {}
    for k, l in predicted.items():
        by_var.setdefault(k.split(":", 1)[1], l)
    mism = [(v, by_var[v], sorted(ls)) for v, ls in sorted(actual.items())
            if v in by_var and max(ls) != by_var[v]]
    print(f"\npredicted vs actual, where they disagree: {len(mism)} of {len(by_var)} named sites")
    for v, pred, ls in mism[:20]:
        flag = "  <-- past the cap" if max(ls) > a.cap >= pred else ""
        print(f"  {v:<12} predicted {pred:g}  actual {ls}{flag}")
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
