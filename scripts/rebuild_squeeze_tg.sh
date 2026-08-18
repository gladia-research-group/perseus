#!/bin/bash
# rebuild_squeeze_tg.sh — rebuild the paper's L_range-only decode row with the THOR-composite
# GELU, so the squeeze arm differs from HEAT only in LN/softmax iteration counts.
#
# RUN ON LEONARDO (A100-64GB). The baseline and HEAT rows were measured there
# (job 50604662 header: "GPU 0: NVIDIA A100-SXM-64GB"); a different device makes the
# latency cells incomparable.  Submit from a login node, from anywhere:
#
#   bash <repo>/src/perseus/scripts/rebuild_squeeze_tg.sh
#
# Overridable: WEIGHTS, CFG, ACCOUNT.  Everything else is pinned to the shape that
# produced the row being replaced (scripts/chain_g2sq_adaptive.sh).
set -e -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # .../src/perseus
ROOT="$(cd "$REPO/../.." && pwd)"                                # repo root
cd "$REPO"

ACCOUNT="${ACCOUNT:-EUHPC_D34_099}"                              # CLAUDE.md: all jobs
GRAPH="$REPO/.cache/graph_gpt2_squeeze_tg"
PLAN_NAME="planned_gpt2_squeeze_tg"
PLAN="$REPO/bootstrap_placements/$PLAN_NAME"

# run_task.sh's own PYTHON default ($REPO/../he-aware-training/.venv/...) does not resolve in
# this layout on either machine; derive it from the repo root instead.
PY="${PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || { echo "missing venv python: $PY"; exit 1; }

# The config lane reorg of 2026-08-14 archived gpt2_squeeze_tg; accept either location.
CFG="${CFG:-}"
if [ -z "$CFG" ]; then
  for c in "$REPO/configs/model/approximation/gpt2_squeeze_tg/configs.json" \
           "$REPO"/configs/model/approximation/_backup/_campaign_*/gpt2_squeeze_tg/configs.json; do
    [ -f "$c" ] && { CFG="$c"; break; }
  done
fi
[ -n "$CFG" ] && [ -f "$CFG" ] || {
  echo "gpt2_squeeze_tg config not found. It is committed at perseus 80b555a:"
  echo "  git checkout 80b555a -- configs/model/approximation/gpt2_squeeze_tg"; exit 1; }

# The exact bundle job 50604662 read — recorded in scripts/chain_g2sq_adaptive.sh:41, NOT in
# the job log. Guessing here is the dangerous failure: run_task.sh would otherwise default to
# the CLASSIC (plain GPT-2) bundle, which completes and yields a plausible row for an arm that
# was never measured.
WEIGHTS="${WEIGHTS:-$ROOT/checkpoints/openai-community/gpt2/lm_eval/squeezed/weights.bin.zip}"
[ -f "$WEIGHTS" ] || { echo "missing squeezed weights: $WEIGHTS"; exit 1; }

echo "[cfg]     $CFG"
echo "[weights] $WEIGHTS"
echo "[python]  $PY"

# ---------------------------------------------------------------- 1. capture
# Proven shape: python driver at MULTI_T=1 (RUNNER=cuda capture has no logged run).
# NOTE FHE_GRAPH_DIR does not wipe: run_task.sh:305-306 ARCHIVES any existing dir to
# .cache/_graph_archive/<name>_<timestamp> before re-tracing.
CAP=$(TASK=decode STAGE=capture MULTI_T=1 PYTHON="$PY" \
      CONFIGS_PATH="$CFG" WEIGHTS_PATH="$WEIGHTS" FHE_GRAPH_DIR="$GRAPH" \
      sbatch --parsable --export=ALL -A "$ACCOUNT" --job-name cap_g2sq_tg scripts/run_task.sh)
echo "[1/3] capture  job $CAP  -> $GRAPH"

# ------------------------------------------------------------------- 2. plan
# CPU pass (~8 s, 62 MB): no GPU, qos normal, and PYTHON exported or the planner dies on
# `import networkx` under the system python3.  afterok + explicit gates: run_task.sh's capture
# stage exits 0 even on a partial trace, and the planner exits 0 on an empty graph dir having
# already created the output directory — that is the fail-open shape that has bitten this
# project three times.
PLN=$(sbatch --parsable --dependency=afterok:$CAP --job-name plan_g2sq_tg \
      -A "$ACCOUNT" -p boost_usr_prod --qos=normal -t 00:20:00 --cpus-per-task=8 --mem=16G \
      -o logs/core/%x_%j.out -e logs/core/%x_%j.err \
      --wrap "set -e; cd $REPO; \
              n=\$(ls -d $GRAPH/block_* 2>/dev/null | wc -l); \
              [ \"\$n\" -eq 13 ] || { echo \"[gate] capture incomplete: \$n/13 block graphs\"; exit 1; }; \
              export PYTHON=$PY; \
              env GRAPH_DIR=.cache/graph_gpt2_squeeze_tg OUT_NAME=$PLAN_NAME bash scripts/utils/run_bootstrap_all_blocks.sh; \
              p=\$(ls $PLAN/block_*_placement.json 2>/dev/null | wc -l); \
              [ \"\$p\" -eq 13 ] || { echo \"[gate] plan incomplete: \$p/13 placement files\"; exit 1; }")
echo "[2/3] plan     job $PLN  -> $PLAN"

# ----------------------------------------------------------------- 3. decode
# 4 h wall: run_task.sh's own 2 h leaves ~10% headroom over the reference run (6340.8 s in the
# decode loop plus ~134 s session setup), and the composite GELU changes the per-token cost.
DEC=$(TASK=decode RUNNER=cuda MULTI_T=128 STEPS_T=128 PYTHON="$PY" \
      CONFIGS_PATH="$CFG" WEIGHTS_PATH="$WEIGHTS" FHE_BOOTSTRAP_PLACEMENTS_DIR="$PLAN" \
      sbatch --parsable --export=ALL -A "$ACCOUNT" -t 04:00:00 \
      --dependency=afterok:$PLN --job-name t128_g2sq_tg scripts/run_task.sh)
echo "[3/3] decode   job $DEC  -> logs/core/t128_g2sq_tg_$DEC.out"
echo
echo "GATE before trusting anything: the new graph must carry the composite GELU —"
echo "  grep -ho 'gelu_[a-z_]*' $GRAPH/block_0/graph.json | sort | uniq -c"
echo "  expect gelu_thor_composite, and ZERO gelu_softsign_inv_sqrt"
echo
echo "COMPARE against the row being replaced (job 50604662, single 128-token chain):"
echo "  bts/tok 466.0 | transformer 39.923 | argmax 9.397 | e2e 49.320 s/tok"
echo "expected iterations/forward = 454 (LN 209+75, softmax 88+82, GELU 0)"

# ---------------------------------------------------------------------------
# WHY NO CALIBRATION JOB: the old deploy config already carried the squeezed checkpoint's GELU
# DOMAINS (xmax identical site by site to gpt2_heat's); only the approximant differed. The
# composite fit for exactly those domains already exists as heat's softgelu section, so
# gpt2_squeeze_tg is that section spliced in — no refit, nothing else touched.
#
# AFTER VALIDATION: archive the old artifacts (they are the record of what produced the
# published number) and promote the new ones to the canonical names.
