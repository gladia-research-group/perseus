#!/bin/bash
#SBATCH --job-name lin_split
#SBATCH -A EUHPC_D34_099
#SBATCH --time 00:30:00
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=256G
#SBATCH --output=logs/core/%x_%j.out
#SBATCH --error=logs/core/%x_%j.err

# =============================================================================
#  exp_linear_split.sh — OUR side of the STIP compute-only comparison.
#
#  Companion to exp_stip_batch_sweep.sh. That one measured STIP's Q projection
#  and split it into prep_encode (~21.9 s) vs mul_only (~0.92 s). To compare
#  like with like we need OUR linear decomposed the same way: how much of the
#  wall is homomorphic compute (rotations + ct x pt mults) and how much is
#  weight residency (coeff expand / load).
#
#  tests/dev/test_linear_timing.cu already does exactly this — it times the full
#  linear with cudaEvents AND times one isolated rotation and one isolated
#  ct x pt mult at the same level, so
#        compute_only = n_rot * rot_ms + n_mult * mult_ms
#  and the residual against the measured linear is the weight-residency term.
#
#  Chain matched to the frozen production chain (CLAUDE.md): depth 11, dnum 7,
#  logN 16. HID_DIM=768 = BERT-base / GPT-2-base width, the same width STIP's
#  target 9 uses. NOTE the test hardcodes num_heads=16 (production is 12) and
#  benchmarks a SINGLE-token linear; under cachemir one ct carries
#  t = slots/d = 32768/768 = 42 token lanes, so STIP's m=128 needs 4 such cts.
#  Both facts are stated in the write-up rather than silently scaled away.
#
#  Usage:  sbatch scripts/exp_linear_split.sh          (from a CLEAN shell)
# =============================================================================
set -uo pipefail

module load cuda/12.6 gcc cmake nccl
module use /leonardo/prod/spack/06/modules/0.22.2_preprod_base
module load libarchive/3.7.1--gcc--12.2.0-sw6t2mm

BIN="${LIN_BIN:-build-py/bin/test_linear_timing}"
[ -x "$BIN" ] || { echo "[exp] FATAL: $BIN not built"; exit 1; }

export HID_DIM="${HID_DIM:-768}"
export DEPTH="${DEPTH:-11}"
export DNUM="${DNUM:-7}"
export REPS="${REPS:-20}"

echo "[exp] host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "[exp] HID_DIM=$HID_DIM DEPTH=$DEPTH DNUM=$DNUM REPS=$REPS"
echo

ok=0
for arm in materialized coeff; do
  echo "############### [exp] ARM=$arm ###############"
  if [ "$arm" = "coeff" ]; then
    export FHE_PT_COEFF_ENCODE=1
  else
    unset FHE_PT_COEFF_ENCODE
  fi
  out=$("$BIN" 2>&1)
  echo "$out" | sed "s/^/[$arm] /"
  echo "$out" | grep -q "isolated primitives" && ok=$((ok + 1))
  echo
done

echo "[exp] arms_with_primitive_timings=$ok/2"
if [ "$ok" -eq 2 ]; then echo "[exp] PASS"; else echo "[exp] FAIL"; exit 1; fi
