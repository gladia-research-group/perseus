#!/usr/bin/env bash
# Measure the bootstrap accuracy table the placer plans against.
#
# Emits the raw log AND the parsed table:
#   logs/core/bts_accuracy_<chain>_<stamp>.log
#   perseus/plan/data/bts_accuracy_<chain>.json     (the table the planner reads)
#
# CF is runtime-only (CorrectionScope), so the WHOLE grid runs in ONE context: the
# ~85 s bootstrap setup is paid once, not once per correction factor.
#
#   CHAIN=n32 bash scripts/utils/sweep_bts_accuracy.sh
#   CHAIN=n64 bash scripts/utils/sweep_bts_accuracy.sh
#
# Env passthrough: BTS_ACC_CFS BTS_ACC_AMPS BTS_ACC_PERIODS BTS_ACC_RIPPLE
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

export CHAIN="${CHAIN:-n32}"
# shellcheck disable=SC1091
source scripts/local_env.sh >/dev/null 2>&1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

BIN="$REPO/build_py_$CHAIN/bin/bts_accuracy_sweep"
if [ ! -x "$BIN" ]; then
    echo "missing $BIN — build it with:" >&2
    echo "  CHAIN=$CHAIN source scripts/local_env.sh && cmake --build build_py_$CHAIN --target bts_accuracy_sweep -j 12" >&2
    exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$REPO/logs/core/bts_accuracy_${CHAIN}_${STAMP}.log"
mkdir -p "$(dirname "$LOG")"

# Provenance: a table is only valid for the binary that produced it (plans are
# binary-bound, and so is the error model behind them).
{
    echo "[sweep] chain=$CHAIN stamp=$STAMP"
    echo "[sweep] repo_sha=$(git -C "$REPO" rev-parse HEAD)"
    echo "[sweep] fideslib_sha=$(git -C "$REPO/third_party/FIDESlib" rev-parse HEAD)"
    echo "[sweep] bin=$BIN mtime=$(date -r "$BIN" +%Y-%m-%dT%H:%M:%S)"
    echo "[sweep] SPARSE_BTS_SLOTS=${SPARSE_BTS_SLOTS:-} CORRECTION_FACTOR=${CORRECTION_FACTOR:-}"
    echo "[sweep] BTS_ACC_CFS=${BTS_ACC_CFS:-default} BTS_ACC_AMPS=${BTS_ACC_AMPS:-default}"
    echo "[sweep] BTS_ACC_PERIODS=${BTS_ACC_PERIODS:-default} BTS_ACC_RIPPLE=${BTS_ACC_RIPPLE:-default}"
} | tee "$LOG"

# `|| true`: the binary's FHE teardown can rewrite the exit status after printing `[  PASSED  ]`,
# so the sweep is gated on the log, not on the exit code.

"$BIN" --gtest_filter='BtsAccuracySweep.AccuracyTable:BtsAccuracySweep.OffsetTable' 2>&1 \
    | tee -a "$LOG" || true

if ! grep -q '^\[  PASSED  \]' "$LOG"; then
    echo "[sweep] FAILED: no PASSED marker in $LOG — table NOT written" >&2
    exit 1
fi

OUT="$REPO/perseus/plan/data/bts_accuracy_${CHAIN}.json"
"${PYTHON:-$REPO/.venv/bin/python}" "$REPO/scripts/utils/parse_bts_accuracy.py" \
    --log "$LOG" --chain "$CHAIN" --out "$OUT" ${SWEEP_FORCE:+--force}
echo "[sweep] log   -> $LOG"
