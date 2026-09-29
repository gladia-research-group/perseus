#!/bin/bash
# plan.sh — the DaCapo arm: the released DaCapo placer's bootstrap targets on a captured graph (CPU only).
#   bash scripts/utils/dacapo_upstream/plan.sh graphs/gpt2_decode_python_n32 python/dacapo [dense]
# 1. sites.py translates each block to hecate's earth dialect and runs hecate-opt's DaCapo
#    passes (HECATE_OPT, or .cache/dacapo_upstream/build/bin/hecate-opt from build_hecate.sh);
#    2. deploy.py replays the targets through the planner (level bookkeeping, chain hand-off,
#    rescue) and stamps the capture contract, the argmax stage (block 13) included, entered
#    from the tail's exit. ML (default 48) is the top of the deployed chain; at 50 the
#    planner's cap slides refresh inputs under 48.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$REPO"
GRAPH_DIR="${1:?graph dir (block_<b>/graph.json)}"
OUT_NAME="${2:?out name under bootstrap_placements/}"
ROUTE="${3:-sparse}"
ML="${ML:-48}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
HERE=scripts/utils/dacapo_upstream
export HECATE_OPT="${HECATE_OPT:-$REPO/.cache/dacapo_upstream/build/bin/hecate-opt}"
[ -x "$HECATE_OPT" ] || { echo "no hecate-opt at $HECATE_OPT (run $HERE/build_hecate.sh)"; exit 1; }
DENSE=0; [ "$ROUTE" = dense ] && DENSE=1
SITES="$(mktemp --suffix=.json)"; trap 'rm -f "$SITES"' EXIT
rm -rf "bootstrap_placements/$OUT_NAME"
GRAPH_DIR="$GRAPH_DIR" OUT_JSON="$SITES" ML=$ML "$PYTHON" $HERE/sites.py
SITES="$SITES" GRAPH_DIR="$GRAPH_DIR" OUT="$OUT_NAME" PLAN_DENSE=$DENSE ML=$ML "$PYTHON" $HERE/deploy.py
