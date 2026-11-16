#!/bin/bash
# make_plan.sh — plan a SANA / text-to-image capture (CPU only) with the shipped prefill recipe (levers.RECIPE: ViT's)
# through the GPT-2 port's blocks driver, then write PLAN_CMD.txt (bind_plan.write_cmd): the recipe, as runtime= the
# levers the capture ran with (<graph dir>/sana_runtime.json, written by run.py --capture), which a --plan
# run applies, and as configs= / pcmm_calib.json the capture's inputs (sana_capture.json).
#   bash python/perseus/examples/sana/make_plan.sh <graph dir> <out name>
# Exported PLAN_* values win over the recipe. JOBS=n (> 1): stage 0 first (from its capture), then the others n at a
# time, each alone: stage 1 from stage 0's exit, the later ones from ENTRY_LEVEL / ENTRY_DEG (default 32 / 1: a
# block's deepest stage output under PLAN_ENVELOPE_STRICT); then the chain bound input by input from those first plans
# (bind_plan.py, n at a time; SEED=a:z its first pass's seed), which writes PLAN_CMD.txt and compiles the chain offline.
# ONLY="1 11" (with JOBS): stage 0 and those stages alone (a typical block's plan; not bound).
set -euo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; REPO="$(dirname "$(dirname "$PKG")")"
G="$(cd "${1:?graph dir}" && pwd)"; OUT="${2:?out name under bootstrap_placements/}"
cd "$REPO"
export PYTHON="${PYTHON:-$REPO/.venv/bin/python}" PYTHONPATH="$(dirname "$PKG")${PYTHONPATH:+:$PYTHONPATH}"
[ -f "$G/sana_runtime.json" ] || { echo "no $G/sana_runtime.json (capture with run.py --capture)"; exit 1; }
RECIPE="$("$PYTHON" -c 'import os, shlex
from perseus.examples.sana import levers
print(" ".join(f"{k}={shlex.quote(os.environ.get(k, v))}" for k, v in levers.RECIPE.items()))')"
[ -f "$G/pcmm_stamps.json" ] && RECIPE="$RECIPE PLAN_PCMM_STAMPS=$G/pcmm_stamps.json"
D="bootstrap_placements/$OUT"
JOBS="${JOBS:-1}"
# WARM=<bound plan> (or a one-line <graph dir>/bind_warm naming it), with JOBS > 1: no first plans; bind_plan's first
# pass plans every stage from that plan's converged entry maps (a capture of the same layout with other levers)
WARM="${WARM:-$(cat "$G/bind_warm" 2>/dev/null || true)}"
if [ -n "$WARM" ] && [ "$JOBS" -gt 1 ] && [ -z "${ONLY:-}" ]; then
    mkdir -p "$D/logs"; rm -f "$D/logs/entry_maps.json"
    exec "$PYTHON" -m perseus.examples.sana.bind_plan "$G" "$OUT" --jobs "$JOBS" --warm "$WARM"
fi
if [ "$JOBS" -le 1 ]; then
    env $RECIPE bash "$PKG/examples/gpt2/make_plan.sh" "$G" "$OUT"
else
    mkdir -p "$D"
    TMP="$(mktemp -d "$G/../_par_XXXX")"
    n=$(ls -d "$G"/block_* | wc -l)
    next_entry() {  # next_entry <stage>: "level deg" its successor enters at (the planner's strict rule)
        "$PYTHON" - "$D/block_$1_placement.json" "$G/block_$1/graph.json" <<'PY'
import json, sys
from pathlib import Path
from perseus.plan.placer.planner import _next_entry
class C: level_unit = 2
print(*_next_entry(json.load(open(sys.argv[1]))["summary"], Path(sys.argv[2]), C, True))
PY
    }
    plan_one() {  # plan_one <stage> [level deg]: stage b alone as block_0 of its own graph dir
        local b=$1 t="$TMP/s$1" o="${OUT}__s$1"
        mkdir -p "$t"; ln -s "$G/block_$b" "$t/block_0"
        for f in capture_env.json capture_config.json; do [ -f "$G/$f" ] && cp "$G/$f" "$t/"; done
        if [ -z "${2:-}" ]; then
            env $RECIPE bash "$PKG/examples/gpt2/make_plan.sh" "$t" "$o" > "$TMP/s$b.log" 2>&1
        else
            env $RECIPE BLOCKS_ENTRY_LEVEL="$2" BLOCKS_ENTRY_DEG="$3" \
                bash "$PKG/examples/gpt2/make_plan.sh" "$t" "$o" > "$TMP/s$b.log" 2>&1
        fi
        mv "bootstrap_placements/$o/block_0_placement.json" "$D/block_${b}_placement.json"
        rm -rf "bootstrap_placements/$o"
        echo "$b ${2:-capture} ${3:-}" >> "$TMP/entries"
        echo "stage $b: $(grep -h 'placements=' "$TMP/s$b.log" | tail -1)"
    }
    export -f plan_one; export TMP G D OUT RECIPE PKG PYTHON
    plan_one 0
    read -r E1 D1 < <(next_entry 0)
    { echo "1 $E1 $D1"; seq 2 $((n - 1)) | sed "s/\$/ ${ENTRY_LEVEL:-32} ${ENTRY_DEG:-1}/"; } \
        | awk -v only="${ONLY:-}" 'BEGIN { n = split(only, o, " "); for (i = 1; i <= n; i++) keep[o[i]] = 1 }
                                   only == "" || ($1 in keep)' \
        | xargs -P "$JOBS" -L1 bash -c 'plan_one "$@"' _
    mkdir -p "$D/logs"; cp "$TMP"/s*.log "$TMP/entries" "$D/logs/"; rm -rf "$TMP"
    if [ -z "${ONLY:-}" ]; then
        rm -f "$D/logs/entry_maps.json"
        exec "$PYTHON" -m perseus.examples.sana.bind_plan "$G" "$OUT" --jobs "$JOBS" --seed "${SEED:-0:0}"
    fi
fi
"$PYTHON" -m perseus.examples.sana.bind_plan "$G" "$OUT" --write-only
echo "wrote $D/PLAN_CMD.txt"
