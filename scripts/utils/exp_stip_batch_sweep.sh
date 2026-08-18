#!/bin/bash
#SBATCH --job-name stip_batch_sweep
#SBATCH -A EUHPC_D34_099
#SBATCH --time 01:00:00
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=256G
#SBATCH --output=logs/core/%x_%j.out
#SBATCH --error=logs/core/%x_%j.err

# =============================================================================
#  exp_stip_batch_sweep.sh — deployment-relevance experiment for the paper.
#
#  CLAIM UNDER TEST: STIP's compact column/AMCP packing (ePrint 2026/174) is a
#  THROUGHPUT layout. Its cost is set by the ciphertext count and the op counts,
#  both of which are independent of how many queries actually occupy the slots.
#  Therefore its advantage is pure batch amortization and vanishes at batch=1,
#  which is the interactive-deployment regime our library targets.
#
#  METHOD: run STIP's OWN binary, their target 9 ("InputProjection: AMCP with
#  DDH" = the full BERT-base Q/K/V projection, all 12 heads via DHP), sweeping
#  ONLY the batch size t. Two edits to their tree, both recorded in-place:
#    1. docs/STIP/GPU/src/STIP_func/ckks_evaluator.cu:735 — a missing '{' that
#       stops the released artifact from compiling at all. One char, no semantics.
#    2. main.cu target 9 — `t` read from $STIP_T instead of the hardcoded 32
#       (their -b/--batchsize flag is parsed but never read by any target).
#       k stays 6: their DHP weight construction pairs head h with head h+6, so
#       k == H/2 is structural, not a tunable. G=128, m=128, every op count and
#       every ciphertext count is IDENTICAL across the sweep — only the number
#       of active interleaved streams W = k*t changes, i.e. only the OCCUPANCY.
#
#  PREDICTION: wall is flat in t  =>  per-query latency scales as wall/t  =>  a
#  32x per-query penalty at t=1 relative to their evaluated t=32.
#
#  Usage:  sbatch scripts/exp_stip_batch_sweep.sh        (from a CLEAN shell)
# =============================================================================
set -uo pipefail

module load cuda/12.6 gcc cmake nccl
module use /leonardo/prod/spack/06/modules/0.22.2_preprod_base
module load libarchive/3.7.1--gcc--12.2.0-sw6t2mm

# NOTE: /scratch_local is NODE-LOCAL on Leonardo — the binary, its .so's and the
# Data tree must all live on shared scratch or the compute node cannot see them.
# Layout must be <root>/GPU/build/bin/STIP_main, because their myread.cu resolves
# the data dir as exeDir.parent^3/"Data".
BIN="${STIP_BIN:?set STIP_BIN to the built STIP_main (on SHARED scratch)}"
REPS="${REPS:-2}"
TS="${TS:-1 2 4 8 16 32}"
export LD_LIBRARY_PATH="$(dirname "$BIN")/../lib:${LD_LIBRARY_PATH:-}"

[ -x "$BIN" ] || { echo "[exp] FATAL: $BIN not executable/visible from $(hostname)"; exit 1; }

echo "[exp] host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "[exp] bin=$BIN"
echo "[exp] target=9 (InputProjection: AMCP with DDH)  model=BERT  reps=$REPS"
echo "[exp] sweep t = $TS   (k=6, m=128, G=128 fixed)"
echo

runs=0
ok=0
for t in $TS; do
  for r in $(seq 1 "$REPS"); do
    echo "===== [exp] STIP_T=$t rep=$r ====="
    runs=$((runs + 1))
    out=$(STIP_T="$t" "$BIN" -p 9 -m BERT -b "$t" 2>&1)
    echo "$out" | sed "s/^/[t=$t r=$r] /"
    # gate on the target actually reporting its projection timings, NOT on exit
    # code (see CLAUDE.md: SLURM/exit status is not evidence)
    echo "$out" | grep -q "Q projection:" && ok=$((ok + 1))
    echo
  done
done

echo "[exp] runs=$runs produced_timings=$ok"
if [ "$runs" -gt 0 ] && [ "$ok" -eq "$runs" ]; then
  echo "[exp] PASS sweep complete"
else
  echo "[exp] FAIL — $((runs - ok))/$runs runs produced no 'Q projection:' timing"
  exit 1
fi
