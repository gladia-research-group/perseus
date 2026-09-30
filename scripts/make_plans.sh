#!/bin/bash
# make_plans.sh — regenerate the shipped bootstrap plans from their recipes (CPU only).
#   bash scripts/make_plans.sh                    # every plan directory with a recipe
#   bash scripts/make_plans.sh gpt2_decode_python_n32 python/dacapo   # just these
# Each bootstrap_placements/<dir>/PLAN_CMD.txt names the graph, the tool (make_plan.sh, or the
# released DaCapo / Orion tools under scripts/utils/*_upstream/), the recipe and, for an
# upstream tool's dense plan, the route; tests/test_paper_plans.py checks the committed plans
# against this regeneration. The DaCapo and Orion plans need hecate-opt
# (scripts/utils/dacapo_upstream/build_hecate.sh) and an Orion clone (fetched on first use).
set -e -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
export PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES=""            # the planner never touches the GPU
mkdir -p logs/plans

if [ $# -gt 0 ]; then
    dirs=("$@")
else
    mapfile -t dirs < <(cd bootstrap_placements && find . -name PLAN_CMD.txt -printf '%h\n' | sed 's#^\./##' | sort)
fi

for d in "${dirs[@]}"; do
    cmd="bootstrap_placements/$d/PLAN_CMD.txt"
    [ -f "$cmd" ] || { echo "no $cmd"; exit 1; }
    graph=$(sed -n 's/^graph=\([^ ]*\).*/\1/p' "$cmd")
    tool=$(sed -n 's/^tool=\([^ ]*\).*/\1/p' "$cmd")
    recipe=$(sed -n 's/^recipe=//p' "$cmd")
    route=$(sed -n 's/^route=//p' "$cmd")
    log="logs/plans/${d//\//_}.log"
    keep="$(mktemp)"; cp "$cmd" "$keep"
    if [ "$tool" = examples/gpt2_from_primitives/make_plan.sh ]; then
        # shellcheck disable=SC2086
        env $recipe bash "$tool" "$graph" "$d" > "$log" 2>&1
        # shellcheck disable=SC2086
        env $recipe bash "$tool" "$graph" "$d" argmax >> "$log" 2>&1
    else
        # shellcheck disable=SC2086
        env $recipe bash "$tool" "$graph" "$d" ${route:-sparse} > "$log" 2>&1
    fi
    cp "$keep" "$cmd"; rm -f "$keep"            # the tools rewrite the directory
    echo "$d: $(tail -n 1 "$log")"
done
