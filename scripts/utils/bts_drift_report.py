#!/usr/bin/env python3
"""Join `[bts_input]` runtime magnitudes against the CAPTURED graph magnitudes.

The planner places bootstraps using `output_max_abs` from a capture that is an EAGER
trajectory. A planned run removes ~half the refreshes, so the magnitudes it actually
meets are not the ones the plan was built from. This script measures that gap — it is
the quantity `mag_safety` (κ) is supposed to cover, and nothing measured it before.

  DEBUG=1 (not BTS_DEBUG) emits one `[bts_input]` line per bootstrap:
    [bts_input] step=<step>.bootstrap var=<v> level=<L> deg=<d> min=.. max=.. avg=.. |abs|max=<A>

Usage
  python scripts/utils/bts_drift_report.py --graph-dir .cache/graph_n32_gpt2_base \
      --log logs/core/dbg2_planned.log [--ref-log logs/core/dbg2_eager.log] \
      [--plan bootstrap_placements/planned_n32_keep] [--cf 6] [--top 20]

With --ref-log it also reports the first bootstrap at which the two runs diverge on a
var they SHARE, which is the honest way to localise "the trajectory stopped matching"
(a site the reference never bootstraps proves nothing on its own).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

# [bts_input] step=<s> var=<v> level=<n> deg=<n> min=<f> max=<f> avg=<f> |abs|max=<f>
LINE_RE = re.compile(
    r"\[bts_input\]\s+step=(?P<step>\S+)\s+var=(?P<var>\S+)\s+level=(?P<level>-?\d+)\s+"
    r"deg=(?P<deg>-?\d+)\s+min=(?P<min>\S+)\s+max=(?P<max>\S+)\s+avg=(?P<avg>\S+)\s+"
    r"\|abs\|max=(?P<absmax>\S+)"
)
BLK_RE = re.compile(r"\.blk(\d+)\.")
TOK_RE = re.compile(r"^tok(\d+)\.")


def parse_log(path: Path) -> list[dict]:
    """One record per bootstrap, in execution order."""
    out: list[dict] = []
    for line in path.read_text(errors="replace").splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        step = m.group("step")
        blk = BLK_RE.search(step)
        tok = TOK_RE.match(step)
        try:
            absmax = float(m.group("absmax"))
        except ValueError:
            continue
        out.append({
            "i": len(out),
            "step": step,
            # the recorder appends `.bootstrap` to the step scope; the graph node's
            # `step` is the enclosing scope, so strip it before joining.
            "step_base": step[:-len(".bootstrap")] if step.endswith(".bootstrap") else step,
            "var": m.group("var"),
            "level": int(m.group("level")),
            "deg": int(m.group("deg")),
            "avg": _f(m.group("avg")),
            "absmax": absmax,
            "block": int(blk.group(1)) if blk else None,
            "tok": int(tok.group(1)) if tok else None,
        })
    return out


def _f(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def load_graph_mags(graph_dir: Path) -> dict[int, dict[str, dict]]:
    """block index -> {var: node}. Only blocks with a graph.json."""
    blocks: dict[int, dict[str, dict]] = {}
    for d in sorted(graph_dir.glob("block_*")):
        gf = d / "graph.json"
        if not gf.is_file():
            continue
        idx = int(d.name.rsplit("_", 1)[1])
        nodes = json.loads(gf.read_text())["nodes"]
        blocks[idx] = {str(n.get("output")): n for n in nodes}
    return blocks


def load_plan_sites(plan_dir: Path) -> dict[int, set[str]]:
    """block index -> planned target_vars (placements only, not hints)."""
    sites: dict[int, set[str]] = {}
    for f in sorted(plan_dir.glob("block_*_placement.json")):
        idx = int(f.name.split("_")[1])
        p = json.loads(f.read_text())
        sites[idx] = {str(x.get("target_var")) for x in p.get("placements", [])}
    return sites


def band(cf: float) -> tuple[float, float]:
    """The measured EvalMod usable band."""
    return 0.003 * 2.0 ** cf, 0.03 * 2.0 ** cf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph-dir", required=True, type=Path)
    ap.add_argument("--log", required=True, type=Path, help="the run under test")
    ap.add_argument("--ref-log", type=Path, help="reference run (usually eager)")
    ap.add_argument("--plan", type=Path, help="plan dir, to split planted vs reactive sites")
    ap.add_argument("--cf", type=float, default=6.0, help="CORRECTION_FACTOR the run used")
    ap.add_argument("--tok", type=int, default=0, help="token index to join (capture is tok0)")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    recs = parse_log(args.log)
    graphs = load_graph_mags(args.graph_dir)
    plan_sites = load_plan_sites(args.plan) if args.plan else None
    lo, hi = band(args.cf)

    print(f"log={args.log}  bootstraps={len(recs)}")
    print(f"graph={args.graph_dir}  blocks={sorted(graphs)}")
    print(f"CF={args.cf:g} -> EvalMod band [{lo:.4g}, {hi:.4g}]\n")

    joined: list[dict] = []
    unjoined = 0
    for r in recs:
        if r["tok"] != args.tok or r["block"] is None:
            continue
        node = graphs.get(r["block"], {}).get(r["var"])
        if node is None or node.get("output_max_abs") is None:
            unjoined += 1
            continue
        cap = float(node["output_max_abs"])
        r = dict(r, captured=cap, ratio=(r["absmax"] / cap) if cap > 0 else float("inf"),
                 step_graph=node.get("step"), op=node.get("op_type"),
                 pack_period=node.get("pack_period"))
        if plan_sites is not None:
            r["planted"] = r["var"] in plan_sites.get(r["block"], set())
        joined.append(r)

    print(f"joined {len(joined)} bootstraps to captured magnitudes "
          f"({unjoined} tok{args.tok} sites had no captured magnitude)\n")
    if not joined:
        return

    # ── drift ────────────────────────────────────────────────────────────────────
    ratios = sorted(r["ratio"] for r in joined if r["ratio"] not in (float("inf"),))
    def pct(p: float) -> float:
        return ratios[min(len(ratios) - 1, int(p / 100 * len(ratios)))]
    print("RUNTIME / CAPTURED magnitude ratio  (1.0 = the plan met what it planned for)")
    print(f"  n={len(ratios)}  median={statistics.median(ratios):.4g}  "
          f"p90={pct(90):.4g}  p99={pct(99):.4g}  max={ratios[-1]:.4g}")
    over = [r for r in ratios if r > 2.0]
    print(f"  sites above 2x: {len(over)}/{len(ratios)} ({100*len(over)/len(ratios):.1f}%)\n")

    # ── band occupancy, captured vs runtime ──────────────────────────────────────
    def occupancy(key: str) -> tuple[int, int, int]:
        below = sum(1 for r in joined if r[key] < lo)
        inside = sum(1 for r in joined if lo <= r[key] <= hi)
        above = sum(1 for r in joined if r[key] > hi)
        return below, inside, above
    cb, ci, ca = occupancy("captured")
    rb, ri, ra = occupancy("absmax")
    print(f"EvalMod band occupancy of the SAME {len(joined)} sites")
    print(f"  captured : below={cb:5d}  IN BAND={ci:5d}  above={ca:5d}")
    print(f"  runtime  : below={rb:5d}  IN BAND={ri:5d}  above={ra:5d}")
    print("  -> a placer thresholding the CAPTURED value cannot refuse what the "
          "runtime column shows.\n")

    if plan_sites is not None:
        for label, want in (("planted (plan placements)", True), ("reactive/hint", False)):
            sel = [r for r in joined if r.get("planted") is want]
            if not sel:
                continue
            rs = sorted(r["ratio"] for r in sel)
            ab = sum(1 for r in sel if r["absmax"] > hi)
            print(f"  {label}: n={len(sel)} median_ratio={statistics.median(rs):.4g} "
                  f"above_band={ab}")
        print()

    # ── worst offenders ──────────────────────────────────────────────────────────
    print(f"TOP {args.top} by runtime magnitude")
    print(f"  {'#':>4} {'blk':>3} {'var':>10} {'captured':>11} {'runtime':>11} "
          f"{'ratio':>10}  {'op':<12} step")
    for r in sorted(joined, key=lambda x: -x["absmax"])[:args.top]:
        print(f"  {r['i']:>4} {r['block']:>3} {r['var']:>10} {r['captured']:>11.4g} "
              f"{r['absmax']:>11.4g} {r['ratio']:>10.4g}  {str(r['op']):<12} "
              f"{r['step_base'].split(':')[-1]}")
    print()

    # ── divergence against a reference run ───────────────────────────────────────
    #
    #  KEYED BY (block, step, ordinal-within-step), NEVER BY VAR. Eager does not reset the
    # per-block ct var counter while the capture and planned runs do, so the same computation
    # carries completely different var names in the two logs — an eager/planned pair shares
    # about ONE var name by accident. A var-keyed diff reports "the runs never diverge"
    # because it has nothing to compare.
    if args.ref_log:
        ref = parse_log(args.ref_log)

        def index(recs: list[dict]) -> dict[tuple, dict]:
            seen: dict[tuple, int] = {}
            out: dict[tuple, dict] = {}
            for r in recs:
                if r["tok"] != args.tok or r["block"] is None:
                    continue
                base = (r["block"], r["step_base"])
                k = seen.get(base, 0)
                seen[base] = k + 1
                out[(r["block"], r["step_base"], k)] = r
            return out

        run_ix, ref_ix = index(recs), index(ref)
        shared = sorted(set(run_ix) & set(ref_ix), key=lambda k: run_ix[k]["i"])
        print(f"vs {args.ref_log}: {len(ref)} bootstraps; "
              f"{len(shared)} (block, step, n) sites bootstrapped by BOTH runs")

        print("\n  execution-order comparison on the shared prefix "
              "(ratio = this run / reference):")
        print(f"    {'#':>4} {'blk':>3} {'run |m|':>11} {'ref |m|':>11} {'ratio':>10}  step")
        first = None
        for k in shared[:args.top]:
            r, q = run_ix[k], ref_ix[k]
            ratio = r["absmax"] / q["absmax"] if q["absmax"] > 0 else float("inf")
            flag = ""
            if first is None and (ratio > 1.05 or ratio < 0.95):
                first, flag = (k, ratio), "  <<< FIRST DIVERGENCE"
            print(f"    {r['i']:>4} {r['block']:>3} {r['absmax']:>11.4g} "
                  f"{q['absmax']:>11.4g} {ratio:>10.4g}  "
                  f"{r['step_base'].split(':')[-1]}{flag}")
        if first is None:
            print("    (no divergence above 5% in the window shown)")

        # Sites this run refreshes that the reference never does, and vice versa. The
        # dangerous direction is the second one: where eager refreshes and the plan does
        # not, a value can climb out of band.
        only_run = [k for k in run_ix if k not in ref_ix]
        only_ref = [k for k in ref_ix if k not in run_ix]
        print(f"\n  bootstrapped ONLY by this run: {len(only_run)} "
              f"({sum(1 for k in only_run if run_ix[k]['absmax'] > hi)} above the band)")
        print(f"  bootstrapped ONLY by the reference: {len(only_ref)} "
              f"({sum(1 for k in only_ref if ref_ix[k]['absmax'] > hi)} above the band)")


if __name__ == "__main__":
    main()
