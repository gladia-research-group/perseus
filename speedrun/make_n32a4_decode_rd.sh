#!/bin/bash
# The raise-drop pass over a finished Decode-8 plan (CPU only), as speedrun/make_prefill_plan.sh's ${OUT}rd: every
# block keeps PLAN's program (PLAN_KEEP_PLACEMENTS_EXACT: its placements, fired hints and realize anchors; the min-cut
# placer, no solver) and the planner's raise-drop search runs once more on the final program, which moves no refresh
# and changes no block output (blocks 0-11 and block 0's feedback variant; stages 12-13 kept). On
# gpt2_decode_python_n32a4_kc16r_cm the pass finds the slack the realize-anchor trials leave after the plan's own search:
# 16 SPRU refreshes per token (the LayerNorms' variance) raise 2-4 levels lower.
# KV_PUSH_RAISE=b:d,... (the runtime switch, impl/attention.py) also raises those blocks' K/V push refreshes d levels
# below the top (the planner's deliberate-drop probe, PLAN_RAISE_DELIB=1, finds 1 level of slack in blocks 0, 4, 5, 7,
# 10; block 0 is left out, its feedback variant enters elsewhere): the pass runs on a copy of the capture with those
# refreshes landed there (fakecap's landing, CapFhe.raise_drop; KV_PUSH_RAISE_LANDED=1: the capture was taken with the
# switch and already lands them, its graph is not edited) and
# the plan's runtime line gains the switch. The pass refuses a raise that moves a block's exit or starts a refresh
# downstream past the chain (PLAN_CHAIN_STRICT: the planner checks every placed refresh's effective input level).
#   bash speedrun/make_n32a4_decode_rd.sh [PLAN]   -> PLAN's blocks 0-11 and block_0_feedback rewritten in place
#   KV_PUSH_RAISE=4:1,5:1,7:1,10:1 bash speedrun/make_n32a4_decode_rd.sh   (the served plan's; its runtime line also
#   carries FIDESLIB_CTS_ALT_BUDGET=5: the dense raise variants of drop >= 1 on a 5-stage CtS)
# RD_FREE_CF=1: the pass may also change the refreshes' CFs (a plan's decisions carried onto another capture by
# scripts/utils/map_plan_decisions.py: new levels, new CFs)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/local_env.sh
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1
PLAN=${1:-gpt2_decode_python_n32a4_kc16r_cm}
GR=$(sed -n 's/^graph=\([^ ]*\).*/\1/p' "bootstrap_placements/$PLAN/PLAN_CMD.txt")
RECIPE="$(sed -n 's/^recipe=//p' "bootstrap_placements/$PLAN/PLAN_CMD.txt")"
KEEP="PLAN_KEEP_PLACEMENTS=bootstrap_placements/$PLAN PLAN_KEEP_PLACEMENTS_EXACT=1 PLAN_PLACER=min_cut"
# the kept program's refresh inputs are checked against the chain once more (a raised K/V push lands its block deeper:
# a refresh downstream can start past max_level, which the compile refuses): the planner refuses such a block
export PLAN_CHAIN_STRICT=${PLAN_CHAIN_STRICT-1}
OUT=_rd_$PLAN
rm -rf "bootstrap_placements/$OUT"
GR0=$GR
if [ -n "${KV_PUSH_RAISE:-}" ]; then
    GR=${GR0}_kvraise
    rm -rf "$GR" && cp -r "$GR0" "$GR"
    "$PYTHON" - "$GR" "$KV_PUSH_RAISE" <<'PY'
import json, os, sys
d, spec = sys.argv[1:]
for e in spec.split(","):
    b, k = (int(x) for x in e.split(":"))
    f = f"{d}/block_{b}/graph.json"
    g = json.load(open(f))
    hit = [n for n in g["nodes"] if n["op_type"] == "deliberate_bootstrap" and n["step"].endswith("kv_push")]
    assert len(hit) == 1, f"block {b}: {len(hit)} K/V push refreshes"
    n = hit[0]
    if n["args"]["route"].get("raise_drop") == k or os.environ.get("KV_PUSH_RAISE_LANDED", "0") not in ("", "0"):
        continue                                  # a capture taken with KV_PUSH_RAISE already lands it there
    n["output_level"] += 2 * k
    n["shadow"]["level"] += 2 * k
    n["args"]["route"]["raise_drop"] = k
    json.dump(g, open(f, "w"))
PY
fi
TMP="$(mktemp -d graphs/_blocks_XXXX)"
for b in $(seq 0 11); do ln -s "$PWD/$GR/block_$b" "$TMP/block_$b"; done
cp "$GR/capture_env.json" "$TMP/"
# shellcheck disable=SC2086
env $RECIPE $KEEP bash python/perseus/examples/gpt2/make_plan.sh "$TMP" "$OUT" blocks
rm -rf "$TMP"
# shellcheck disable=SC2086
env $RECIPE $KEEP bash python/perseus/examples/gpt2/make_plan.sh "$GR" "$OUT" feedback
"$PYTHON" - "bootstrap_placements/$PLAN" "bootstrap_placements/$OUT" <<'PY'
import json, os, sys
a, b = sys.argv[1:]
keys = ("placements", "hint_fire", "sparse_slots", "real_route", "rescale_after") + \
    (() if os.environ.get("RD_FREE_CF") else ("correction_factor",))
for k in [str(i) for i in range(12)] + ["0_feedback"]:
    x, y = (json.load(open(f"{d}/block_{k}_placement.json")) for d in (a, b))
    # rescale_after is a set (a carried plan lists it in the mapping's order)
    norm = lambda f, v: sorted(v) if f == "rescale_after" and v else v
    bad = [f for f in keys if norm(f, x.get(f)) != norm(f, y.get(f))]
    ex = lambda s: (s["exit_level"], s.get("exit_deg") or 1)
    # RD_EXIT_SHALLOWER=1: a block may exit at a lower level (same degree) than the kept plan's (a carried plan on a
    # shallower capture: the next stage enters with more room)
    moved = ex(x["summary"]) != ex(y["summary"])
    if moved and os.environ.get("RD_EXIT_SHALLOWER") and ex(y["summary"])[1] == ex(x["summary"])[1] \
            and ex(y["summary"])[0] <= ex(x["summary"])[0]:
        moved = False
    if bad or moved:
        sys.exit(f"block {k}: the pass changed {bad or 'the exit'}")
    print(f"block {k}: raise drops {x.get('raise_drop', {})} -> {y.get('raise_drop', {})}")
PY
for k in $(seq 0 11) 0_feedback; do cp "bootstrap_placements/$OUT/block_${k}_placement.json" "bootstrap_placements/$PLAN/"; done
rm -rf "bootstrap_placements/$OUT"
[ "$GR" = "$GR0" ] || rm -rf "$GR"
sed -i -e "/^rd=/d" -e "s/ KV_PUSH_RAISE=[^ ]*//" "bootstrap_placements/$PLAN/PLAN_CMD.txt"
[ -z "${KV_PUSH_RAISE:-}" ] || sed -i "/^runtime=/s|\$| KV_PUSH_RAISE=$KV_PUSH_RAISE|" "bootstrap_placements/$PLAN/PLAN_CMD.txt"
echo "rd=blocks 0-11 and block_0_feedback by speedrun/make_n32a4_decode_rd.sh${KV_PUSH_RAISE:+ (KV_PUSH_RAISE=$KV_PUSH_RAISE)}: \
the raise-drop search once more on the final program (PLAN_KEEP_PLACEMENTS_EXACT)" >> "bootstrap_placements/$PLAN/PLAN_CMD.txt"
