#!/bin/bash
#SBATCH --job-name he128x
#SBATCH -A EUHPC_D34_099
#SBATCH --time 03:00:00
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=256G
#SBATCH --array=0-127%24
#SBATCH --output=logs/core/%x_%a_%A.out
#SBATCH --error=logs/core/%x_%a_%A.err

# HE128 encrypted fidelity campaign (F7 crossover + tab:main-results top1/KL columns),
# new-library edition: 128 paired teacher-forced prompts x 128 tokens per arm.
# Plaintext GT oracles pre-gathered per sample (cluster/gather_he128_*.slurm, seeds
# 1000+TID, SAME OWT windows across arms => paired). Arm env comes via --export=ALL:
#   ORACLE_BASE  $SCRATCH/he128 | he128_heat  (per-sample all_blocks_* GT dirs)
#   CONFIGS_PATH / WEIGHTS_PATH / FHE_BOOTSTRAP_PLACEMENTS_DIR   the arm's deploy set
TID=${SLURM_ARRAY_TASK_ID:-0}
export ALL_BLOCKS_IO_DIR="$ORACLE_BASE/sample_$(printf '%03d' "$TID")"
[ -f "$ALL_BLOCKS_IO_DIR/all_blocks_L00_T128.json" ] || { echo "[he128x] missing oracle $ALL_BLOCKS_IO_DIR"; exit 1; }
export TASK=decode STAGE=run RUNNER=cuda MULTI_T=128 STEPS_T=128
echo "[he128x] tid=$TID oracle=$ALL_BLOCKS_IO_DIR cfg=$CONFIGS_PATH"
exec bash scripts/run_task.sh
