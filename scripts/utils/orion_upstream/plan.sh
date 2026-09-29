#!/bin/bash
# plan.sh — the Orion arm: the released Orion bootstrap solver's sites on a captured graph (CPU only).
#   bash scripts/utils/orion_upstream/plan.sh graphs/gpt2_decode_python_n32 python/orion [dense]
# 1. marks.py runs baahl-nyu/orion's BootstrapSolver (commit be8a827 + orion_be8a827.patch,
#    cloned into .cache/orion_upstream unless ORION_SRC points at such a clone) on each block's
#    step graph; 2. deploy.py replays the marked steps through the planner (level bookkeeping,
#    chain hand-off, rescue) and stamps the capture contract, the argmax stage (block 13)
#    included, entered from the tail's exit.
# The solver iterates Python sets of node names, so its marks depend on the hash seed:
# PYTHONHASHSEED is pinned (0 unless exported). ML (default 48) is the top of the deployed
# chain: the solver marks sites in its own 50-level frame (marks.py), and deployed at 50 some
# seeds place a refresh at level 50, past the measured bootstrap envelope (48), which the
# runtime refuses.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$REPO"
GRAPH_DIR="${1:?graph dir (block_<b>/graph.json)}"
OUT_NAME="${2:?out name under bootstrap_placements/}"
ROUTE="${3:-sparse}"
ML="${ML:-48}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
HERE=scripts/utils/orion_upstream
if [ -z "${ORION_SRC:-}" ]; then
    ORION_SRC="$REPO/.cache/orion_upstream"
    if [ ! -d "$ORION_SRC/.git" ]; then
        git clone -q https://github.com/baahl-nyu/orion "$ORION_SRC"
        git -C "$ORION_SRC" checkout -q be8a827
        git -C "$ORION_SRC" apply "$REPO/$HERE/orion_be8a827.patch"
    fi
fi
export ORION_SRC
DENSE=0; [ "$ROUTE" = dense ] && DENSE=1
MARKS="$(mktemp --suffix=.json)"; trap 'rm -f "$MARKS"' EXIT
rm -rf "bootstrap_placements/$OUT_NAME"
GRAPH_DIR="$GRAPH_DIR" OUT_JSON="$MARKS" PLAN_DENSE=$DENSE "$PYTHON" $HERE/marks.py
RESULTS="$MARKS" GRAPH_DIR="$GRAPH_DIR" OUT="$OUT_NAME" PLAN_DENSE=$DENSE ML=$ML "$PYTHON" $HERE/deploy.py
