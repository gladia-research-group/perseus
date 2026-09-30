#!/bin/bash
# macro_run.sh — one provenance-stamped model run.
#
#   usage: TAG=<name> [any run_task.sh env] bash scripts/macro_run.sh
#
# run_task.sh's internal log is stdout-only and is removed on both the success and the
# failure path, so nothing survives a run unless the caller captures it — and half the
# interesting markers ([decode] per-token, [cutmax_bts], [sparse_*], [bts_*]) are on
# stderr and never reach it at all. This wrapper keeps both streams and records, in the
# log itself, what the run was: the repo SHA, the FIDESlib SHA, which chain's .so is
# loaded, the plan directory's mtime and checksum, and the full env. A plan directory can
# be regenerated afterwards, so its path alone does not identify what a run bound to.
#
# Writes $OUTDIR/$TAG.{out,err,meta}. Never deletes anything.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${TAG:?usage: TAG=<name> bash scripts/macro_run.sh}"
OUTDIR="${OUTDIR:-$REPO/logs/macro}"
mkdir -p "$OUTDIR"

OUT="$OUTDIR/$TAG.out"
ERR="$OUTDIR/$TAG.err"
META="$OUTDIR/$TAG.meta"

SO="$REPO/perseus/_core.cpython-312-x86_64-linux-gnu.so"
PLAN="${FHE_BOOTSTRAP_PLACEMENTS_DIR:-}"

{
  echo "=== macro_run provenance: $TAG ==="
  echo "date_utc          $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host              $(hostname)"
  echo "repo_sha          $(git -C "$REPO" rev-parse HEAD 2>/dev/null)"
  echo "repo_dirty        $(git -C "$REPO" status --porcelain 2>/dev/null | wc -l) files"
  echo "fideslib_sha      $(git -C "$REPO/third_party/FIDESlib" rev-parse HEAD 2>/dev/null)"
  # Which chain is loaded: the .so is a symlink to _core.<chain>.so. A plain file here is an
  # unstashed build that a chain switch would destroy.
  echo "core_so_link      $(readlink "$SO" 2>/dev/null || echo 'NOT-A-SYMLINK(unstashed build!)')"
  echo "core_so_mtime     $(stat -c '%y' "$SO" 2>/dev/null)"
  echo "core_so_md5       $(md5sum "$SO" 2>/dev/null | cut -d' ' -f1)"
  if [ -n "$PLAN" ]; then
    echo "plan_dir          $PLAN"
    echo "plan_mtime        $(stat -c '%y' "$PLAN" 2>/dev/null)"
    # content hash over the placement files: proves WHICH plan this run bound, so a later
    # regeneration cannot silently reattribute this log to a different plan.
    echo "plan_md5          $(cat "$PLAN"/*.json 2>/dev/null | md5sum | cut -d' ' -f1)"
    echo "plan_files        $(ls "$PLAN"/*.json 2>/dev/null | wc -l)"
  else
    echo "plan_dir          (unset -- run_task.sh default or eager)"
  fi
  echo "accuracy_table_md5 $(md5sum "$REPO/perseus/plan/data/bts_accuracy_n32.json" 2>/dev/null | cut -d' ' -f1)"
  echo "gpu_visible       ${CUDA_VISIBLE_DEVICES:-unset}"
  echo "--- other GPUs at launch ---"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
  echo "--- caller env overrides ---"
  # Every knob that can move the wall belongs in this list: an A/B whose .meta does not
  # record which arm it was cannot be compared against anything later.
  for v in TASK STAGE RUNNER CHAIN MULTI_T AUTO_BTS_LEVEL GPT2_INFERENCE_MODE CONFIGS_PATH \
           WEIGHTS_PATH FHE_GRAPH_DIR \
           FHE_BOOTSTRAP_PLACEMENTS_DIR FHE_PROFILE FHE_PROFILE_TOKEN FHE_PT_STAGE_BLOCK \
           FHE_KV_OVERLAP FHE_KV_RESIDENT SPARSE_AUTO CORRECTION_FACTOR FHE_LMHEAD_CAP \
           OMP_NUM_THREADS \
           FHE_BLOCK_CIRCULAR FHE_BLOCK_PREFETCH FHE_PIN_STAGE FHE_STAGE_ARENA_GB \
           FHE_WEIGHTS_RESIDENT NO_INLINE_PRIME FHE_MASK_PREFETCH FHE_MASK_PREFETCH_STRICT \
           FHE_RING_FILL_AT FHE_WORKER_OMP FHE_LMHEAD_PROBE \
           FHE_PREFILL_ENCODE_THREADS PREFILL_ROT_KEY_BAND PREFILL_STAGE_ARENA_GB \
           FHE_PT_COEFF_ENCODE FHE_COEFF_LOG FHE_COEFF_PRESCALE; do
    [ -n "${!v:-}" ] && echo "  $v=${!v}"
  done
  echo "=== end provenance ==="
} | tee "$META"

t0=$(date +%s)
bash "$REPO/scripts/run_task.sh" > "$OUT" 2> "$ERR"
rc=$?
t1=$(date +%s)

{
  echo "wall_total_s      $((t1 - t0))"
  echo "exit_rc           $rc   (NOT evidence -- teardown can exit(0) on fatal CUDA errors)"
  # The run's own marker is the gate. MODE defaults to TASK for decode.
  if grep -qE '^\[[a-z_]+\] PASS$' "$OUT"; then echo "marker            PASS"; else echo "marker            NO-PASS"; fi
} | tee -a "$META"

echo
echo "--- summary ($TAG) ---"
grep -aE '^\[decode\] (top1|pos=)' "$OUT" "$ERR" 2>/dev/null | sed 's/^/  /'
grep -aE '^\[tokstat\]' "$OUT" 2>/dev/null | sed 's/^/  /'
echo "logs: $OUT / $ERR / $META"
exit $rc
