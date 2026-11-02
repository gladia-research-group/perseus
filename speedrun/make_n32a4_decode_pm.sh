#!/bin/bash
# The pre-mask refresh pass over a finished Decode-8 plan (CPU only): in the blocks whose GELU output reaches the down
# projection's hint past the StC-first entry (the plan fires the hint there, the runtime runs the shipped order: ~3.8
# ms more per refresh on the RTX PRO 6000), the refresh moves one op up, onto the GELU's Chebyshev sum before its lane
# mask (x xmax), whose input sits one level fresher: an StC-first refresh at raise drop 2 lands where the mask then
# leaves the down projection's input as before (same block exit, every other refresh, drop and CF as the plan's).
# Same maths (the mask is applied after the refresh instead of before), not the same rounding.
# Each edited block is re-run kept-exact (PLAN_KEEP_PLACEMENTS_EXACT: the planner's sim checks the edit, sets the new
# site's CF, and the raise-drop search runs once more with the new site preset at drop 2: PLAN_RAISE_PRESET).
#   bash speedrun/make_n32a4_decode_pm.sh [PLAN [OUT]]   -> bootstrap_placements/OUT (PLAN with those blocks edited:
#   2, 9 and 11 on gpt2_decode_python_n32a4_kc16r_cmfl and _lh)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/local_env.sh
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1
PLAN=${1:-gpt2_decode_python_n32a4_kc16r_lh} OUT=${2:-${PLAN}_pm}
GR=$(sed -n 's/^graph=\([^ ]*\).*/\1/p' "bootstrap_placements/$PLAN/PLAN_CMD.txt")
RECIPE="$(sed -n 's/^recipe=//p' "bootstrap_placements/$PLAN/PLAN_CMD.txt")"
# the sites: every fired hint on the GELU's masked output (the lane mask x xmax product) whose input passes the
# StC-first entry (PLAN_STC_FIRST_ENTRY, effective level: + a unit for a pending rescale) -> block:sum:hint
ENTRY=$(echo "$RECIPE" | tr ' ' '\n' | sed -n 's/^PLAN_STC_FIRST_ENTRY=//p' | tail -1)
SITES=$("$PYTHON" - "bootstrap_placements/$PLAN" "$GR" "${ENTRY:-46}" <<'PY'
import json, sys
d, gr, entry = sys.argv[1], sys.argv[2], float(sys.argv[3])
out = []
for b in range(12):
    p = json.load(open(f"{d}/block_{b}_placement.json"))
    g = {n["output"]: n for n in json.load(open(f"{gr}/block_{b}/graph.json"))["nodes"]}
    lv, dg, unit = p["summary"]["final_named_levels"], p["summary"]["final_named_degs"], 2
    for h in p["hint_fire"]:
        n = g.get(h)
        if not n or n["op_type"] != "hint":
            continue
        m = g.get(n["inputs"][0].removesuffix("_auto_bootstrapped"))
        if not (m and m["op_type"] == "mult" and m.get("args", {}).get("pt", {}).get("name", "").startswith("mask.gelu")):
            continue
        eff = lv[m["output"]] + (unit if dg.get(m["output"]) == 2 else 0)
        if eff > entry:
            out.append(f"{b}:{m['inputs'][0]}:{h}")
if any(e.startswith("0:") for e in out):
    sys.exit("block 0 has a site: its feedback variant would need the same edit")
print(" ".join(out))
PY
)
echo "sites (block:refreshed sum:hint no longer fired): $SITES"
# a block whose K/V push the runtime raises (KV_PUSH_RAISE) was planned on a capture copy with that landing: not here
KVR=$(sed -n 's/^runtime=.*KV_PUSH_RAISE=\([^ ]*\).*/\1/p' "bootstrap_placements/$PLAN/PLAN_CMD.txt")
for e in $SITES; do
    case ",$KVR," in *",${e%%:*}:"*) echo "block ${e%%:*}: its K/V push is raised (KV_PUSH_RAISE=$KVR)" >&2; exit 1;; esac
done
EDIT=_pm_edit_$OUT
rm -rf "bootstrap_placements/$OUT" "bootstrap_placements/$EDIT"
cp -r "bootstrap_placements/$PLAN" "bootstrap_placements/$OUT"
cp -r "bootstrap_placements/$PLAN" "bootstrap_placements/$EDIT"
"$PYTHON" - "bootstrap_placements/$EDIT" "$GR" $SITES <<'PY'
import json, sys
d, gr, *sites = sys.argv[1:]
for e in sites:
    b, v, h = e.split(":")
    g = {n["output"]: n for n in json.load(open(f"{gr}/block_{b}/graph.json"))["nodes"]}
    m = [n for n in g.values() if v in n["inputs"]]
    assert len(m) == 1 and m[0]["op_type"] == "mult" and m[0]["args"]["pt"]["name"].startswith("mask.gelu"), (b, v)
    # the hint reads the mask's output (or the capture's own refresh of it, which the planner erases)
    assert g[h]["op_type"] == "hint" and g[h]["inputs"][0].removesuffix("_auto_bootstrapped") == m[0]["output"], (b, h)
    f = f"{d}/block_{b}_placement.json"
    p = json.load(open(f))
    assert h in p["hint_fire"], (b, h)
    p["hint_fire"] = [x for x in p["hint_fire"] if x != h]
    p["placements"].append({"type": "bootstrap_after_node", "node_op": g[v]["op_type"], "target_var": v})
    p["real_route"] = [v if x == h else x for x in p["real_route"]]
    p["correction_factor"].pop(h, None)
    p.get("raise_drop", {}).pop(h, None)
    p["rescale_after"] = sorted(set(p["rescale_after"]) | {v})
    json.dump(p, open(f, "w"), indent=1)
PY
for e in $SITES; do
    IFS=: read -r b v _h <<< "$e"
    read -r L D < <("$PYTHON" -c 'import json, sys; s = json.load(open(sys.argv[1]))["summary"]
print(s["exit_level"], s.get("exit_deg") or 1)' "bootstrap_placements/$PLAN/block_$((b - 1))_placement.json")
    TMP="$(mktemp -d graphs/_pm_XXXX)"
    ln -s "$PWD/$GR/block_$b" "$TMP/block_$b" && cp "$GR/capture_env.json" "$TMP/"
    # shellcheck disable=SC2086
    env $RECIPE PLAN_KEEP_PLACEMENTS="bootstrap_placements/$EDIT" PLAN_KEEP_PLACEMENTS_EXACT=1 PLAN_PLACER=min_cut \
        PLAN_RAISE_PRESET="$v:2" FIRST_ENTRY_LEVEL="$L" FIRST_ENTRY_DEG="$D" \
        bash python/perseus/examples/gpt2/make_plan.sh "$TMP" "${EDIT}_b$b" blocks
    rm -rf "$TMP"
    "$PYTHON" - "bootstrap_placements/$PLAN/block_${b}_placement.json" "bootstrap_placements/${EDIT}_b$b/block_${b}_placement.json" "$v" <<'PY'
import json, sys
a, c = (json.load(open(f)) for f in sys.argv[1:3])
v = sys.argv[3]
ex = lambda p: (p["summary"]["exit_level"], p["summary"].get("exit_deg") or 1)
assert ex(a) == ex(c), f"exit {ex(a)} -> {ex(c)}"
assert {k: x for k, x in c["raise_drop"].items() if k != v} == a.get("raise_drop", {}), (a.get("raise_drop"), c["raise_drop"])
assert c["raise_drop"].get(v) == 2 and v in c["real_route"], c["raise_drop"]
print(f"{sys.argv[2]}: {v} drop 2 CF {c['correction_factor'][v]}, exit {ex(c)}, refreshes {c['summary']['total_bootstraps']}")
PY
    cp "bootstrap_placements/${EDIT}_b$b/block_${b}_placement.json" "bootstrap_placements/$OUT/"
    rm -rf "bootstrap_placements/${EDIT}_b$b"
done
rm -rf "bootstrap_placements/$EDIT"
echo "pm=$SITES (block:site:hint) by speedrun/make_n32a4_decode_pm.sh: the down projection's hint refresh moved before \
the GELU's lane mask (StC-first at raise drop 2 instead of the shipped order; PLAN_RAISE_PRESET)" >> "bootstrap_placements/$OUT/PLAN_CMD.txt"
