#!/bin/bash
#SBATCH --job-name cjf
#SBATCH -A EUHPC_D34_099
#SBATCH --time 03:00:00
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=256G
#SBATCH --array=0-31%8
#SBATCH --output=logs/core/%x_%a_%A.out
#SBATCH --error=logs/core/%x_%a_%A.err

# Array wrapper around cj_sweep_idx_file.sh, whose SHARD is a plain env var and so
# cannot be driven by an array index directly. Exists only to get a %N throttle on the
# full 512-image EuroSAT sweep: 3 arms x 32 shards unthrottled is 96 simultaneous jobs,
# which both floods the queue and raises the chance two of our own jobs share a host --
# and a shared host voids the latency column (CLAUDE.md: never compare timings across
# runs that shared the machine).
#   NSHARDS must match the --array range. Arm env (CONFIGS_PATH/VIT_*/plan) via --export=ALL.
set -u
export SHARD="${SLURM_ARRAY_TASK_ID:-0}"
export NSHARDS="${NSHARDS:-32}"
exec bash scripts/utils/cj_sweep_idx_file.sh
