#!/bin/bash
# Profile a run and render an interactive flame chart — one command, repeatable.
#
#   bash scripts/flamechart.sh <TAG> [ENV=VAL ...]
#
#   # the shipped arm — the default if you pass nothing but a tag
#   bash scripts/flamechart.sh B
#
#   # any other arm: every env is forwarded verbatim to run_task.sh, same as macro_run.sh
#   bash scripts/flamechart.sh lnvar FUSED_LN_VAR=1 \
#        FHE_BOOTSTRAP_PLACEMENTS_DIR=$PWD/bootstrap_placements/lnvar_ml46
#   bash scripts/flamechart.sh eager STAGE=eager
#   bash scripts/flamechart.sh n64   CHAIN=n64 STAGE=eager
#
#   # re-render an existing log without touching the GPU (edit the template, re-run this)
#   RENDER_ONLY=1 bash scripts/flamechart.sh B
#
# Output: logs/macro/<TAG>.flamechart.html — self-contained, open it straight from disk.
# To publish it to claude.ai, ask Claude to republish that file: updating the SAME artifact
# keeps its URL, a new file path mints a new one. That step needs the Artifact tool and so
# cannot live in this script.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:?usage: flamechart.sh <TAG> [ENV=VAL ...]}"; shift || true
OUTDIR="${OUTDIR:-$REPO/logs/macro}"
LOG="$OUTDIR/$TAG.out"

if [ "${RENDER_ONLY:-0}" = "0" ]; then
  # Profiling defaults. Every one is overridable on the command line because they are passed
  # BEFORE "$@" — a later assignment to the same name wins in `env`.
  #   FHE_PROFILE_TRACE=1   per-INSTANCE intervals -> a true flame CHART (x = real elapsed time).
#                         Without it only per-path totals exist, which can only be packed from
#                         the parent's left edge — an ordering the data never contained.
#   FHE_PROFILE=wall      the only mode whose totals are self-consistent; `events` inflates
  #                         the token 2.19x vs wall's 1.39x, so it is not the default
  #   FHE_PROFILE_TOKEN=1   token 1 = WARM. Token 0 is the cold token, and its ~8.5 s of
  #                         one-time pinned-arena first touch is charged to whatever runs
  #                         first: profiling it reads one-time cost as steady state
  #   BTS_SCOPE_ROUTE=1     name each bootstrap scope by its routed slot count, so the graph
  #                         separates dense from s=1 / s=512 instead of one undifferentiated mass
  #   MULTI_T=2             enough to reach a warm token; more just costs GPU time
  C_DEFAULT="$REPO/configs/model/approximation/gpt2_base_n32/configs.json"
  P_DEFAULT="$REPO/bootstrap_placements/planned_n32_L48"
  set -- FHE_PROFILE=wall FHE_PROFILE_TRACE=1 FHE_PROFILE_TOKEN=1 BTS_SCOPE_ROUTE=1 MULTI_T=2 \
         CHAIN=n32 TASK=decode GPT2_INFERENCE_MODE=threaded AUTO_BTS_LEVEL=48 \
         CONFIGS_PATH="$C_DEFAULT" FHE_BOOTSTRAP_PLACEMENTS_DIR="$P_DEFAULT" "$@"

  echo "[flamechart] running $TAG …"
  env "$@" TAG="$TAG" OUTDIR="$OUTDIR" bash "$REPO/scripts/macro_run.sh" >/dev/null 2>&1
  rc=$?
  # A failed run is still worth rendering — the partial tree usually shows where it died — but
  # say so, because a profile of a FAILing arm is not evidence about a passing one.
  [ $rc -ne 0 ] && echo "[flamechart]   run did NOT pass its gate (rc=$rc); rendering anyway — see $OUTDIR/$TAG.err"
fi

[ -f "$LOG" ] || { echo "[flamechart] no log at $LOG (RENDER_ONLY=1 with no prior run?)"; exit 1; }

"$REPO/.venv/bin/python" "$REPO/scripts/utils/make_flamechart.py" --log "$LOG" || exit 1

echo "[flamechart] gate:"
grep -ahE '^\[[a-z_]+\] (top1|pos=0)' "$OUTDIR/$TAG.out" "$OUTDIR/$TAG.err" 2>/dev/null | sed 's/^/         /'
echo "[flamechart] other GPUs at launch (wall figures are unusable if any were busy):"
sed -n '/other GPUs/,/caller env/p' "$OUTDIR/$TAG.meta" 2>/dev/null | grep -E '^[0-7],' | sed 's/^/         /'
