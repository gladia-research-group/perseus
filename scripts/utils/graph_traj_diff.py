#!/usr/bin/env python3
"""Diff two captured graph dirs NODE BY NODE and report where the trajectories depart.

The `[bts_input]` probe only prints out-of-window BOOTSTRAP inputs, so it can say where a
run has already blown up but never where it first departed. A capture records every node's
magnitude, so capturing the same workload twice — once eager, once under a plan
(`CAPTURE_UNDER_PLAN=1`) — localises the first divergence to a single op.

Alignment is by `(step, op, ordinal-within-step)`, never by var name: a planned run inserts
refresh nodes and eager does not reset the per-block var counter, so var names do not
correspond across arms. Steps are compared in execution order and the FIRST node whose
magnitude ratio leaves [1/tol, tol] is reported with its full local context.

  python scripts/utils/graph_traj_diff.py \
      --ref .cache/graph_n32_gpt2_base --run .cache/graph_n32_underplan [--tol 1.05]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Nodes a planned run legitimately adds/removes relative to eager: comparing them is
# meaningless, but the nodes AROUND them must still line up.
_REFRESH_OPS = {"auto_bootstrap", "deliberate_bootstrap"}


def load_block(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["nodes"]


def key_stream(nodes: list[dict]) -> list[tuple[tuple[str, str, int], dict]]:
    """[(step, op, n-th occurrence), node] in execution order, refresh nodes dropped."""
    seen: dict[tuple[str, str], int] = {}
    out = []
    for n in nodes:
        op = str(n.get("op_type", ""))
        if op in _REFRESH_OPS:
            continue
        step = str(n.get("step", ""))
        k = seen.get((step, op), 0)
        seen[(step, op)] = k + 1
        out.append(((step, op, k), n))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, type=Path, help="reference capture (eager)")
    ap.add_argument("--run", required=True, type=Path, help="capture under test (planned)")
    ap.add_argument("--tol", type=float, default=1.05,
                    help="ratio outside [1/tol, tol] counts as divergence (default 1.05)")
    ap.add_argument("--min-abs", type=float, default=0.0,
                    help="ignore nodes where BOTH arms are below this |m| — a ratio at a "
                         "near-cancellation swings on noise and is not a divergence")
    ap.add_argument("--context", type=int, default=6,
                    help="nodes of context to print around the first divergence")
    ap.add_argument("--blocks", type=str, default="",
                    help="comma list of block indices (default: all present in BOTH)")
    args = ap.parse_args()

    ref_blocks = {int(d.name.rsplit("_", 1)[1]): d / "graph.json"
                  for d in args.ref.glob("block_*") if (d / "graph.json").is_file()}
    run_blocks = {int(d.name.rsplit("_", 1)[1]): d / "graph.json"
                  for d in args.run.glob("block_*") if (d / "graph.json").is_file()}
    want = ([int(x) for x in args.blocks.split(",") if x] if args.blocks
            else sorted(set(ref_blocks) & set(run_blocks)))
    print(f"ref={args.ref} ({len(ref_blocks)} blocks)")
    print(f"run={args.run} ({len(run_blocks)} blocks)")
    print(f"comparing blocks {want} at tol={args.tol}\n")

    first_global = None
    for b in want:
        r = key_stream(load_block(ref_blocks[b]))
        u = key_stream(load_block(run_blocks[b]))
        ref_by = dict(r)
        shared = [(k, n) for k, n in u if k in ref_by]
        n_only_run = len(u) - len(shared)
        n_only_ref = len(r) - len(shared)

        first = None
        n_div = 0
        for i, (k, n) in enumerate(shared):
            a = n.get("output_max_abs")
            e = ref_by[k].get("output_max_abs")
            if a is None or e is None:
                continue
            if e == 0 and a == 0:
                continue
            if max(abs(a), abs(e)) < args.min_abs:
                continue
            ratio = (a / e) if e not in (0, None) else float("inf")
            if ratio > args.tol or ratio < 1.0 / args.tol:
                n_div += 1
                if first is None:
                    first = (i, k, n, ref_by[k], ratio)
        status = "IDENTICAL" if first is None else f"diverges at shared node #{first[0]}"
        print(f"block_{b}: {len(shared)} shared nodes "
              f"(+{n_only_run} run-only, +{n_only_ref} ref-only) — {status}"
              + (f", {n_div} divergent" if n_div else ""))
        if first is not None and first_global is None:
            first_global = (b, first, shared, ref_by)

    if first_global is None:
        print("\nNo divergence anywhere: the planned trajectory reproduces the reference.")
        return

    b, (i, k, n, rn, ratio), shared, ref_by = first_global
    print(f"\n=== FIRST DIVERGENCE: block_{b}, shared node #{i} ===")
    print(f"  step = {k[0]}")
    print(f"  op   = {k[1]}  (occurrence {k[2]})")
    print(f"  run |m| = {n.get('output_max_abs'):.6g}   "
          f"ref |m| = {rn.get('output_max_abs'):.6g}   ratio = {ratio:.6g}")
    print(f"  run out_level={n.get('output_level')} deg={n.get('output_noise_level')}  |  "
          f"ref out_level={rn.get('output_level')} deg={rn.get('output_noise_level')}")
    print(f"  run inputs={n.get('inputs')} levels={n.get('input_levels')}")
    print(f"  ref inputs={rn.get('inputs')} levels={rn.get('input_levels')}")

    lo = max(0, i - args.context)
    print(f"\n  context (shared nodes {lo}..{i + args.context}):")
    print(f"    {'#':>5} {'ratio':>11} {'run |m|':>11} {'ref |m|':>11} {'lvl r/e':>9}  op / step")
    for j in range(lo, min(len(shared), i + args.context + 1)):
        kk, nn = shared[j]
        rr = ref_by[kk]
        a, e = nn.get("output_max_abs"), rr.get("output_max_abs")
        ratio_j = (a / e) if (a is not None and e) else float("nan")
        mark = "  <<<" if j == i else ""
        print(f"    {j:>5} {ratio_j:>11.4g} {(a if a is not None else float('nan')):>11.4g} "
              f"{(e if e is not None else float('nan')):>11.4g} "
              f"{str(nn.get('output_level'))+'/'+str(rr.get('output_level')):>9}  "
              f"{kk[1]} / {kk[0].split(':')[-1]}{mark}")


if __name__ == "__main__":
    main()
