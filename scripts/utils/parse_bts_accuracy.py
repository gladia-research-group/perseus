#!/usr/bin/env python3
"""Parse a `[bts_acc]`/`[bts_off]` sweep log into perseus/plan/data/bts_accuracy_<chain>.json.

The table is the placer's model of `rel_err(|m|, CF, data period, route)`. It is TRACKED in
the repo on purpose: `docs/` is gitignored and four cited documents have already been lost
that way, and a plan is only as reproducible as the error model behind it.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ACC_RE = re.compile(
    r"\[bts_acc\]\s+cf=(?P<cf>\d+)\s+period=(?P<period>\d+)\s+route=(?P<route>\d+)\s+"
    r"A=(?P<A>\S+)\s+rel_err=(?P<err>\S+)")
OFF_RE = re.compile(
    r"\[bts_off\]\s+cf=(?P<cf>\d+)\s+dc=(?P<dc>\S+)\s+as_is=(?P<as_is>\S+)\s+"
    r"offset=(?P<offset>\S+)")
META_RE = re.compile(
    r"\[bts_acc\]\s+meta\s+chain_d=(?P<d>\d+)\s+total_depth=(?P<depth>\d+)\s+"
    r"level_limit=(?P<limit>\d+)")
SWEEP_RE = re.compile(r"\[sweep\]\s+(?P<key>\w+)=(?P<val>\S*)")


def _f(s: str) -> float | None:
    try:
        v = float(s)
    except ValueError:
        return None
    return None if v != v else v          # drop NaN (the cf < deg refusals)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--chain", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing table that has MORE rows than this one")
    args = ap.parse_args()

    text = args.log.read_text(errors="replace")

    meta: dict = {"chain": args.chain}
    for m in SWEEP_RE.finditer(text):
        if m.group("key") in ("repo_sha", "fideslib_sha", "stamp", "SPARSE_BTS_SLOTS"):
            meta[m.group("key").lower()] = m.group("val")
    if (m := META_RE.search(text)):
        meta.update(composite_degree=int(m.group("d")),
                    total_depth=int(m.group("depth")),
                    level_limit=int(m.group("limit")))

    wall = []
    for m in ACC_RE.finditer(text):
        err, amp = _f(m.group("err")), _f(m.group("A"))
        if err is None or amp is None:
            continue
        wall.append({"cf": int(m.group("cf")), "period": int(m.group("period")),
                     "route": int(m.group("route")), "amp": amp, "rel_err": err})

    offset = []
    for m in OFF_RE.finditer(text):
        a, o, dc = _f(m.group("as_is")), _f(m.group("offset")), _f(m.group("dc"))
        if a is None or o is None or dc is None:
            continue
        offset.append({"cf": int(m.group("cf")), "dc": dc, "as_is": a, "offset": o})

    if not wall:
        raise SystemExit(f"no [bts_acc] rows in {args.log} — did the sweep run?")

    cfs = sorted({r["cf"] for r in wall})
    meta["cf_min"], meta["cf_max"] = cfs[0], cfs[-1]
    doc = {"version": 1, "meta": meta, "wall": wall, "offset": offset}

    # A one-cell debug sweep must not silently replace the full measured grid the planner
    # plans against. This nearly happened: the table is the plan's evidence, and a plan is
    # only as reproducible as the model behind it.
    if args.out.is_file() and not args.force:
        try:
            prev = json.loads(args.out.read_text(encoding="utf-8"))
            n_prev = len(prev.get("wall", []))
        except Exception:
            n_prev = 0
        if n_prev > len(wall):
            raise SystemExit(
                f"refusing to overwrite {args.out}: it has {n_prev} wall rows, this sweep "
                f"produced {len(wall)}. Re-run the FULL grid, or pass --force "
                f"(SWEEP_FORCE=1) if you really mean to shrink it.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"[parse] {len(wall)} wall rows, {len(offset)} offset rows, "
          f"cf {cfs[0]}..{cfs[-1]} -> {args.out}")


if __name__ == "__main__":
    main()
