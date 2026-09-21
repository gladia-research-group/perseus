#!/bin/bash
#SBATCH --job-name cj_sweep
#SBATCH -A IscrC_eff-SAM2
#SBATCH --time 03:00:00
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=256G
#SBATCH --output=logs/core/%x_%j.out
#SBATCH --error=logs/core/%x_%j.err

# Crown-jewel accuracy sweep, ONE PROCESS PER IMAGE: strict planned mode is only
# valid for the first forward of a session (per-forward state leaks the entry
# levels off-plan on forward #2 — enc-cache clear + wrapper rebuild both
# insufficient, open library lead). Fresh process per image = fresh keys (~3.5
# min overhead) + the exact deployed planned path. Arm env (TASK/STAGE/PYTHON/
# CONFIGS_PATH/VIT_*/FHE_BOOTSTRAP_PLACEMENTS_DIR) comes via --export=ALL.
for i in $(seq "${IMG_START:-0}" "${IMG_END:-19}"); do
  echo "=== [cj] image $i ==="
  # run_task re-loads modules every iteration; without a purge the stacks
  # accumulate and by ~image 6 the poisoned env breaks the _core import.
  module purge 2>/dev/null || true
  env VIT_IMG_IDXS="$i" bash scripts/run_task.sh || echo "[cj] image $i FAILED"
done
echo "[cj] sweep done"
