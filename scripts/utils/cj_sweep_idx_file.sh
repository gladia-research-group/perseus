#!/bin/bash
#SBATCH --job-name cj_sweep_idx
#SBATCH -A EUHPC_D34_099
#SBATCH --time 03:00:00
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=256G
#SBATCH --output=logs/core/%x_%j.out
#SBATCH --error=logs/core/%x_%j.err

# RANDOM-SAMPLE variant of cj_sweep_per_image.sh (which swept a contiguous IMG_START..IMG_END
# range). The EuroSAT val tensor set is CLASS-ORDERED (2700 imgs: 317 of class 0, 312 of
# class 1, ...), so a contiguous prefix is not a sample of the task: the n=128 pass was ONE
# class and the n=512 pass was TWO of ten. Indices now come from a FILE of pre-sampled
# image ids (one per line), sliced into NSHARDS contiguous shards; SHARD picks this job's.
# Everything else is unchanged from the original: ONE PROCESS PER IMAGE (strict planned mode
# is only valid for a session's first forward), fresh keys per image, module purge per
# iteration (accumulated stacks break the _core import by ~image 6). Arm env
# (TASK/STAGE/PYTHON/CONFIGS_PATH/VIT_*/FHE_BOOTSTRAP_PLACEMENTS_DIR) via --export=ALL.
#   IMG_FILE  file of image indices, one per line (e.g. scripts/utils/cj_random512_seed0.txt)
#   SHARD     0-based shard index
#   NSHARDS   total shards (default 32)
set -u
IMG_FILE="${IMG_FILE:?set IMG_FILE}"
SHARD="${SHARD:?set SHARD}"
NSHARDS="${NSHARDS:-32}"

mapfile -t ALL < "$IMG_FILE"
n=${#ALL[@]}
per=$(( (n + NSHARDS - 1) / NSHARDS ))
start=$(( SHARD * per ))
echo "[cj] shard $SHARD/$NSHARDS: images $start..$(( start + per - 1 )) of $n from $IMG_FILE"

for (( k=start; k<start+per && k<n; k++ )); do
  i="${ALL[$k]}"
  echo "=== [cj] image $i ==="
  module purge 2>/dev/null || true
  env VIT_IMG_IDXS="$i" bash scripts/run_task.sh || echo "[cj] image $i FAILED"
done
echo "[cj] sweep done shard $SHARD"
