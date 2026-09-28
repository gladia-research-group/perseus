#!/bin/bash
# make_plan.sh — plan the bootstraps of a graph captured by the primitives port (CPU only).
#   bash examples/gpt2_from_primitives/make_plan.sh graphs/gpt2_decode_python_n32 gpt2_decode_python_n32
#   bash examples/gpt2_from_primitives/make_plan.sh graphs/gpt2_decode_python_n32 gpt2_decode_python_n32 argmax
# Recipe: the n32 recipe of scripts/make_plans.sh (a planted refresh lands at 36 on the
# composite chain, the block input / a KV-cache read arrive at 34 + a pending rescale = 36,
# MAX_LEVEL=50, kappa=2, CF window up to 20, the measured n32 accuracy table, sparse routes
# {512, 1} with their landings). PLAN_* knobs pass through the env (PLAN_SPARSE_SLOTS= for dense).
#
# The CutMax stage (block n_layers+1) is planned in a SECOND pass, `make_plan.sh <graph> <name>
# argmax`, after the blocks: from the same eager capture, with its entry (level, degree) taken
# from the tail plan's exit instead of the capture (what enters CutMax is what the PLANNED tail
# produces; from the capture's own levels the plan fails the strict level check at its first
# op), and with the hints dissolved, which makes the refresh envelope hard: planned live, its
# hints fire at the top of the chain, where a bootstrap returns garbage silently; dissolved,
# the cut places the refreshes on the same sites with per-site factors and sparse routes.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
GRAPH_DIR="${1:?graph dir (block_<b>/graph.json)}"
OUT_NAME="${2:?out name under bootstrap_placements/}"
STAGE="${3:-blocks}"
export CUDA_VISIBLE_DEVICES=""
export PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
# a dense plan (PLAN_SPARSE_SLOTS set and empty) runs every refresh on the dense route, so the
# code's own bootstraps land at the bootstrap level whatever route they took in the capture
if [[ -n "${PLAN_SPARSE_SLOTS+x}" && -z "$PLAN_SPARSE_SLOTS" ]]; then
    export PLAN_DELIBERATE_CLAMP0="${PLAN_DELIBERATE_CLAMP0-1}"
fi
if [ "$STAGE" = argmax ]; then
    [ -d "$GRAPH_DIR/block_13" ] || { echo "no $GRAPH_DIR/block_13 (capture with --argmax)"; exit 1; }
    TAIL="bootstrap_placements/$OUT_NAME/block_12_placement.json"
    [ -f "$TAIL" ] || { echo "no $TAIL (plan the blocks first)"; exit 1; }
    read -r ENTRY_LEVEL ENTRY_DEG < <("$PYTHON" -c 'import json,sys; s=json.load(open(sys.argv[1]))["summary"]; print(s["exit_level"], s.get("exit_deg") or 1)' "$TAIL")
    TMP="$(mktemp -d "$GRAPH_DIR/../_argmax_XXXX")"; ln -s "$(cd "$GRAPH_DIR/block_13" && pwd)" "$TMP/block_13"
    cp "$GRAPH_DIR/capture_env.json" "$TMP/" 2>/dev/null || true
    env GRAPH_DIR="$TMP" OUT_NAME="${OUT_NAME}_argmax" PLAN_DISSOLVE_HINTS=1 PLAN_HARD_ENV_CAP=1 \
        FIRST_ENTRY_LEVEL="${FIRST_ENTRY_LEVEL:-$ENTRY_LEVEL}" FIRST_ENTRY_DEG="${FIRST_ENTRY_DEG:-$ENTRY_DEG}" \
        MAX_LEVEL="${MAX_LEVEL:-50}" BTS_LEVEL="${BTS_LEVEL:-36}" SRC_LEVEL="${SRC_LEVEL:-36}" \
        CACHE_READ_LEVEL="${CACHE_READ_LEVEL:-36}" PLAN_LEVEL_UNIT="${PLAN_LEVEL_UNIT:-2}" \
        PLAN_CF_MAX="${PLAN_CF_MAX:-20}" PLAN_NO_PRESCALE="${PLAN_NO_PRESCALE:-1}" \
        PLAN_ACC_CHAIN="${PLAN_ACC_CHAIN-n32}" PLAN_MAG_SAFETY="${PLAN_MAG_SAFETY:-2}" \
        PLAN_SPARSE_SLOTS="${PLAN_SPARSE_SLOTS-512,1}" PLAN_SPARSE_BTS_OUT="${PLAN_SPARSE_BTS_OUT-1:26,512:36}" \
        bash scripts/utils/run_bootstrap_all_blocks.sh
    cp "bootstrap_placements/${OUT_NAME}_argmax/block_13_placement.json" "bootstrap_placements/$OUT_NAME/"
    rm -rf "$TMP" "bootstrap_placements/${OUT_NAME}_argmax"
    "$PYTHON" - "$GRAPH_DIR" "bootstrap_placements/$OUT_NAME/block_13_placement.json" <<'PY'
import json, os, sys
from perseus.plan import contract
g, p = sys.argv[1], sys.argv[2]
try:
    c = json.load(open(os.path.join(g, "capture_env.json")))
    if os.environ.get("PLAN_SPARSE_SLOTS", "512,1") == "":   # dense plan: runs without sparse keys
        c["env"].update(SPARSE_AUTO="0", SPARSE_BTS_SLOTS="0")
    contract.stamp_file(p, c)
except OSError:
    pass
print(p, "placements:", len(json.load(open(p)).get("placements", [])))
PY
    exit 0
fi
env GRAPH_DIR="$GRAPH_DIR" OUT_NAME="$OUT_NAME" \
    MAX_LEVEL="${MAX_LEVEL:-50}" BTS_LEVEL="${BTS_LEVEL:-36}" SRC_LEVEL="${SRC_LEVEL:-36}" \
    CACHE_READ_LEVEL="${CACHE_READ_LEVEL:-36}" PLAN_LEVEL_UNIT="${PLAN_LEVEL_UNIT:-2}" \
    PLAN_CF_MAX="${PLAN_CF_MAX:-20}" PLAN_NO_PRESCALE="${PLAN_NO_PRESCALE:-1}" \
    PLAN_ACC_CHAIN="${PLAN_ACC_CHAIN-n32}" PLAN_MAG_SAFETY="${PLAN_MAG_SAFETY:-2}" \
    PLAN_HINT_ENV_VETO="${PLAN_HINT_ENV_VETO:-1}" \
    PLAN_SPARSE_SLOTS="${PLAN_SPARSE_SLOTS-512,1}" PLAN_SPARSE_BTS_OUT="${PLAN_SPARSE_BTS_OUT-1:26,512:36}" \
    bash scripts/utils/run_bootstrap_all_blocks.sh
# stamp the capture contract into every placement so the run refuses a mismatched env
"$PYTHON" - "$GRAPH_DIR" "bootstrap_placements/$OUT_NAME" <<'PY'
import glob, json, os, sys
from perseus.plan import contract
g, o = sys.argv[1], sys.argv[2]
try:
    c = json.load(open(os.path.join(g, "capture_env.json")))
except OSError:
    c = None
if c is not None and os.environ.get("PLAN_SPARSE_SLOTS", "512,1") == "":
    # dense plan from any capture (routing is a plan-time choice): it runs without sparse keys
    c["env"].update(SPARSE_AUTO="0", SPARSE_BTS_SLOTS="0")
for p in sorted(glob.glob(os.path.join(o, "block_*_placement.json"))):
    if c is not None:
        contract.stamp_file(p, c)
    d = json.load(open(p))
    print(p, "placements:", len(d.get("placements", d.get("bootstraps", []))))
PY
