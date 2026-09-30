#!/usr/bin/env python
"""The full baseline-placer lever matrix.

Arms: 3 placers x 2 variants x {hints aware|blind} x {sparse granted|dense}
      (variants: orion/fhelipe = tight|faithful; dacapo = faithful|pervalue, since
       cross-validation showed the "cut" bypass mode is the one reproducing upstream)
      x magnitude lever — dacapo/fhelipe: {rescued, raw} (rescue IS their magnitude
      machinery; upstream has none); orion: RAW ONLY (natively magnitude-aware via
      its calibrated prescale — it gets no rescue anywhere, infeasible is reported
      as infeasible).
Plus the min_cut reference row.

  .venv/bin/python scripts/utils/placer_matrix.py --graph-dir .cache/graph_n32_gpt2_base

Plan dirs land under bootstrap_placements/matrix/<label>/ (never clobbers canonical
dirs). Output: results/placer_matrix_<graph>.csv + .md (one row per arm).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "utils"))

from compare_placers import n32_cfg, run_arm  # noqa: E402

# Variants RETIRED as a headline axis: each placer runs its FAITHFUL published
# semantics by default (see the placer modules). What remains is the repair axis
# (faithful vs working) plus hints and sparse. Our deviations stay as opt-in
# ablations: PLAN_ORION_FREEDROP=0 / PLAN_FHELIPE_DEFER=0 / PLAN_DACAPO_BYPASS=value.
VARIANT_ENV = {
    "orion": ("PLAN_ORION_FREEDROP", {"faithful": "1"}),
    "fhelipe": ("PLAN_FHELIPE_DEFER", {"faithful": "1"}),
    "dacapo": ("PLAN_DACAPO_BYPASS", {"faithful": "cut"}),
}


def arms():
    yield ("min_cut", "-", "-", "-", "-", {}, dict(placer="min_cut"))
    for placer in ("orion", "dacapo", "fhelipe"):
        vkey, vmap = VARIANT_ENV[placer]
        rescue_opts = ((False,) if placer == "orion" else (True, False))
        for variant in vmap:                       # per-placer variant names
            for hints in ("aware", "blind"):
                for sparse in ("sparse", "dense"):
                    for rescued in rescue_opts:
                        env = {vkey: vmap[variant], "PLAN_BASELINE_HINTS": hints}
                        over = dict(placer=placer, baseline_rescue=rescued)
                        if sparse == "dense":
                            over["sparse_precomps"] = ()
                            over["sparse_out_levels"] = ()
                        mag = ("native" if placer == "orion"
                               else ("rescued" if rescued else "raw"))
                        label = f"{placer}({variant},{hints},{sparse},{mag})"
                        yield (placer, variant, hints, sparse, mag, env, over)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", required=True)
    args = ap.parse_args()
    graph_dir = Path(args.graph_dir)
    out_prefix = Path(f"results/placer_matrix_{graph_dir.name}")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summary = []
    for placer, variant, hints, sparse, mag, env, over in arms():
        label = (placer if placer == "min_cut"
                 else f"{placer}({variant},{hints},{sparse},{mag})")
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            cfg = n32_cfg(**over)
            out_dir = REPO / "bootstrap_placements" / "matrix" / \
                label.replace("(", "_").replace(")", "").replace(",", "_")
            chunks = sorted(graph_dir.glob("chunk_*"))
            rows = []
            if chunks:
                for cd in chunks:
                    crows = run_arm(cd, out_dir / cd.name, cfg)
                    for r in crows:
                        r["block"] = f"{cd.name}/{r['block']}"
                    rows += crows
            else:
                rows = run_arm(graph_dir, out_dir, cfg)
            for r in rows:
                r["placer"] = label
            all_rows += rows
            ok = [r for r in rows if r["status"] == "ok"]
            infeas = len(rows) - len(ok)
            tot = sum(r["total_bts"] for r in ok)
            own = sum(r.get("own_sites") or 0 for r in ok)
            asst = sum(r.get("assisted_sites") or 0 for r in ok)
            summary.append((label, len(ok), len(rows), tot, own, asst))
            print(f"[matrix] {label}: ok={len(ok)}/{len(rows)} total={tot} "
                  f"own={own} assisted={asst} "
                  f"({own/max(1,own+asst):.0%} own)", flush=True)
        finally:
            for k, old in saved.items():
                if old is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old

    fields = ["placer", "mode", "block", "status", "placements", "hint_bts",
              "total_bts", "own_sites", "assisted_sites", "err_median", "err_p90",
              "err_max", "miss_target", "fixpoint_added", "rescue_moved",
              "rescue_added", "rescue_dropped", "chain_break", "reason"]
    with open(f"{out_prefix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    lines = ["| arm | blocks ok | TOTAL bts | own sites | assisted (ours) | % own |",
             "|---|---|---|---|---|---|"]
    for label, ok, n, tot, own, asst in summary:
        frac = f"{own/max(1,own+asst):.0%}"
        lines.append(f"| {label} | {ok}/{n} | **{tot}** | {own} | {asst} | {frac} |")
    Path(f"{out_prefix}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
