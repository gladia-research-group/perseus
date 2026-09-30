#!/usr/bin/env python
"""Offline placer comparison: run every placer over a graph dir, tabulate.

  .venv/bin/python scripts/utils/compare_placers.py \
      --graph-dir .cache/graph_n32_gpt2_base [--out results/placer_compare_base] \
      [--placers min_cut,fhelipe,orion,dacapo] [--modes raw,rescued]

Emits per-arm plan dirs under bootstrap_placements/ (suffix _<placer>[_raw]) plus a
CSV + markdown table. PlanInfeasible on a block is a RECORDED RESULT (the chain to the
next block re-seeds from the capture, marked chain_break), never a crash.

Env: the N32V2 recipe is the default; every knob can be overridden with the same
PLAN_* envs run_bootstrap_all_blocks.sh uses.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from perseus.plan.placer.place import PlanInfeasible  # noqa: E402
from perseus.plan.placer.planner import PlanConfig, _load_table, plan_block  # noqa: E402

# Sparse routing is GRANTED to every baseline arm. Orion's upstream genuinely supports
# sparse bootstrapping; DaCapo and Fhelipe do not, but running them dense handicaps
# them ~2x on refresh runway (dense lands at 34 against s=1 at 24) and turns the
# comparison into a sparse-vs-dense story instead of a placement story.
# placer_meta.sparse_routing records the grant, and the paper labels it "granted,
# beyond upstream". Set PLAN_DENSE_PLACERS=dacapo,fhelipe to revert.
DENSE_ONLY_PLACERS = set(
    p for p in os.environ.get("PLAN_DENSE_PLACERS", "").split(",") if p)

# Orion upstream is magnitude-aware at refresh REALIZATION (Bootstrap.fit wraps each
# bootstrap in a calibrated prescale into [-1,1], restored level-free by an integer
# postscale). Our input-prescale machinery is the closest analogue but NOT the same
# mechanism, and granting it measures worse on both frames: count -0.35% at ML48 and
# identical at ML50, with accuracy degrading from 4/4 KL 0.09 to 3/4 KL 0.21 (ML48) and
# from 4/4 KL 0.12 to 1/4 KL 0.25 (ML50). The Orion arms therefore run NO-PRESCALE like
# every other arm. Re-grant with PLAN_PRESCALE_PLACERS=orion.
PRESCALE_PLACERS = set(
    p for p in os.environ.get("PLAN_PRESCALE_PLACERS", "").split(",") if p)


def _sparse_out(spec: str) -> tuple:
    # "1:26,512:36" -> ((1, 26), (512, 36)) — same syntax as the v2 CLI's
    # --sparse-bts-out. The default is the 24/34 frame; the 36-pin chains pass
    # PLAN_SPARSE_BTS_OUT=1:26,512:36 (the recipes live in each plan's PLAN_CMD.txt).
    return tuple(tuple(int(x) for x in part.split(":")) for part in spec.split(","))


def n32_cfg(**over) -> PlanConfig:
    env = os.environ.get
    base = dict(
        bootstrap_level=int(env("BTS_LEVEL", 34)),
        max_level=int(env("MAX_LEVEL", 48)),
        source_level=int(env("SRC_LEVEL", 34)),
        cache_read_level=int(env("CACHE_READ_LEVEL", 34)),
        level_unit=int(env("PLAN_LEVEL_UNIT", 2)),
        acc_chain=env("PLAN_ACC_CHAIN", "n32"),
        cf_min=int(env("PLAN_CF_MIN", 2)),
        cf_max=int(env("PLAN_CF_MAX", 14)),
        # Escape hatch: prefill baseline arms planned before the deg-1 landing contract
        # are reproduced with PLAN_BTS_OUT_DEG=2.
        bts_out_deg=int(env("PLAN_BTS_OUT_DEG", 1)),
        allow_prescale=not env("PLAN_NO_PRESCALE", "1"),
        sparse_precomps=(512, 1),
        sparse_out_levels=_sparse_out(env("PLAN_SPARSE_BTS_OUT", "1:24,512:34")),
        verbose=False,
    )
    base.update(over)
    if base.get("placer") in DENSE_ONLY_PLACERS:
        base["sparse_precomps"] = ()
        base["sparse_out_levels"] = ()
    if base.get("placer") in PRESCALE_PLACERS:
        base["allow_prescale"] = True
    return PlanConfig(**base)


def run_arm(graph_dir: Path, out_dir: Path, cfg: PlanConfig) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = _load_table(cfg)
    rows: list[dict] = []
    entry_level = entry_deg = None
    blocks = sorted((d for d in graph_dir.glob("block_*") if d.is_dir()),
                    key=lambda d: int(d.name.rsplit("_", 1)[1]))
    for bd in blocks:
        gf = bd / "graph.json"
        if not gf.is_file():
            continue
        row = {"block": bd.name, "placer": cfg.placer,
               "mode": "rescued" if cfg.baseline_rescue else "raw",
               "chain_break": entry_level is None and bd.name != blocks[0].name}
        try:
            result = plan_block(gf, cfg, entry_level=entry_level,
                                entry_deg=entry_deg, table=table)
        except PlanInfeasible as e:
            row.update(status="infeasible", reason=str(e).splitlines()[0][:160],
                       placements=None, hint_bts=None, total_bts=None)
            rows.append(row)
            entry_level = entry_deg = None       # re-seed next block from capture
            continue
        (out_dir / f"{bd.name}_placement.json").write_text(
            json.dumps(result, indent=1), encoding="utf-8")
        s = result["summary"]
        bq = s["bts_quality"]
        meta = s.get("placer_meta") or {}
        row.update(
            status="ok", reason="",
            placements=s["num_placements"], hint_bts=s["num_hint_bootstraps"],
            total_bts=s["total_bootstraps"],
            err_median=bq["predicted_rel_err"]["median"],
            err_p90=bq["predicted_rel_err"]["p90"],
            err_max=bq["predicted_rel_err"]["max"],
            miss_target=bq["num_sites_missing_target"],
            rescue_moved=meta.get("rescue_moved", 0),
            rescue_added=meta.get("rescue_added", 0),
            fixpoint_added=meta.get("fixpoint_added", 0),
            # SURVIVOR provenance (reconciles to num_placements, unlike the
            # per-iteration addition counters):
            # min_cut carries no placer_meta: every one of its sites is its own
            # decision (single integrated optimization, no repair stage at all).
            own_sites=(meta.get("provenance") or {}).get(
                "own_total", s["num_placements"] if not meta else 0),
            assisted_sites=(meta.get("provenance") or {}).get("assisted_total", 0),
            rescue_dropped=meta.get("rescue_dropped_hopeless", 0),
        )
        rows.append(row)
        entry_level, entry_deg = s.get("exit_level"), s.get("exit_deg") or 1
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", required=True)
    ap.add_argument("--out", default=None, help="output prefix (csv/md)")
    ap.add_argument("--plan-prefix", default=None,
                    help="plan dir prefix; default planned_<graph_dir basename>")
    ap.add_argument("--placers", default="min_cut,fhelipe,orion,dacapo")
    ap.add_argument("--modes", default="raw,rescued",
                    help="baseline modes to run (min_cut always runs once)")
    args = ap.parse_args()

    graph_dir = Path(args.graph_dir)
    name = graph_dir.name.replace("graph_", "planned_")
    prefix = args.plan_prefix or name
    out_prefix = Path(args.out or f"results/placer_compare_{graph_dir.name}")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # There is no variant axis: the baselines run their FAITHFUL published semantics
    # (the defaults in each placer module) plus hint/deliberate awareness for
    # correctness and the sparse flag. The only added axis is REPAIR:
    #   modes=raw      -> faithful as published; infeasible is a reported RESULT
    #   modes=rescued  -> "working": the same algorithm + the minimal repair that
    #                     makes it produce a valid plan.
    # Our own semantic deviations remain available as an opt-in ABLATION via
    # PLAN_ORION_FREEDROP=0 / PLAN_FHELIPE_DEFER=0 / PLAN_DACAPO_BYPASS=pervalue.
    VARIANTS = {}

    all_rows: list[dict] = []
    for placer in args.placers.split(","):
        modes = [None] if placer == "min_cut" else args.modes.split(",")
        variants = [("", {})]
        for mode in modes:
          for vname, venv in variants:
            rescued = (mode == "rescued")
            saved_env = {k: os.environ.get(k) for k in venv}
            os.environ.update(venv)
            try:
                label = placer if not vname else f"{placer}({vname})"
                # tight rescued plans get the canonical online A/B dir name; other
                # variants suffix; our own arm gets _cmp so the shipping plan dir is
                # never clobbered.
                # canonical (unsuffixed) dir = the first variant listed for
                # this placer: its best-faith headline arm.
                canonical = variants[0][0]
                vtag = "" if vname in ("", canonical) else f"_{vname}"
                tag = ("_mincut_cmp" if placer == "min_cut"
                       else f"_{placer}{vtag}" + ("" if rescued else "_raw"))
                out_dir = REPO / "bootstrap_placements" / f"{prefix}{tag}"
                cfg = n32_cfg(placer=placer, baseline_rescue=rescued)
                print(f"[compare] {label}{'' if mode is None else '/' + mode} "
                      f"-> {out_dir.name}")
                chunks = sorted(graph_dir.glob("chunk_*"))
                rows = []
                if chunks:                   # prefill layout: per-chunk plan dirs
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
            finally:
                for k, old in saved_env.items():
                    if old is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = old
            ok = [r for r in rows if r["status"] == "ok"]
            tot = sum(r["total_bts"] for r in ok)
            print(f"[compare]   blocks ok={len(ok)}/{len(rows)} total_bts={tot} "
                  f"rescue(+{sum(r.get('rescue_added') or 0 for r in ok)}"
                  f"/~{sum(r.get('rescue_moved') or 0 for r in ok)})")

    fields = ["placer", "mode", "block", "status", "placements", "hint_bts",
              "total_bts", "own_sites", "assisted_sites", "err_median", "err_p90",
              "err_max", "miss_target", "fixpoint_added", "rescue_moved",
              "rescue_added", "rescue_dropped", "chain_break", "reason"]
    with open(f"{out_prefix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    # markdown summary: one line per arm
    lines = ["| placer | mode | blocks ok | placements | hint bts | TOTAL bts | "
             "pred err p90 (worst blk) | miss target | rescues (moved/added/dropped) |",
             "|---|---|---|---|---|---|---|---|---|"]
    seen = []
    for r in all_rows:
        key = (r["placer"], r["mode"])
        if key in seen:
            continue
        seen.append(key)
        arm = [x for x in all_rows if (x["placer"], x["mode"]) == key]
        ok = [x for x in arm if x["status"] == "ok"]
        p90s = [x["err_p90"] for x in ok if x.get("err_p90") is not None]
        lines.append(
            f"| {key[0]} | {key[1] or '-'} | {len(ok)}/{len(arm)} "
            f"| {sum(x['placements'] for x in ok)} "
            f"| {sum(x['hint_bts'] for x in ok)} "
            f"| **{sum(x['total_bts'] for x in ok)}** "
            f"| {max(p90s) if p90s else float('nan'):.3g} "
            f"| {sum(x.get('miss_target') or 0 for x in ok)} "
            f"| {sum(x.get('rescue_moved') or 0 for x in ok)}"
            f"/{sum(x.get('rescue_added') or 0 for x in ok)}"
            f"/{sum(x.get('rescue_dropped') or 0 for x in ok)} |")
    Path(f"{out_prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[compare] wrote {out_prefix}.csv / .md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
