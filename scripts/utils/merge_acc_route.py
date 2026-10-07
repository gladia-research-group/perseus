#!/usr/bin/env python3
"""Replace one route's rows of a base bts accuracy table with the same route's rows from another table."""
import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--src", required=True)
ap.add_argument("--route", type=int, required=True); ap.add_argument("--chain", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
base = json.load(open(a.base)); src = json.load(open(a.src))
keep = [r for r in base["wall"] if int(r["route"]) != a.route]
new = [r for r in src["wall"] if int(r["route"]) == a.route]
if not new:
    raise SystemExit(f"no route {a.route} rows in {a.src}")
base["wall"] = keep + new
base["meta"]["chain"] = a.chain
base["meta"]["merged_route"] = {"route": a.route, "from": a.src, "rows": len(new)}
json.dump(base, open(a.out, "w"), indent=1)
print(f"[merge] {a.out}: {len(keep)} base rows + {len(new)} route-{a.route} rows from {a.src}")
