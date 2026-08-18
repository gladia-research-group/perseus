#!/bin/bash
#SBATCH --job-name bert_calibrate
#SBATCH -A EUHPC_D34_099
#SBATCH --time 00:30:00
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --output=logs/core/%x_%j.out
#SBATCH --error=logs/core/%x_%j.err

# BERT-base SST-2 calibration — the twin of calibrate_vit.sh.
#
# BLOCK_T is the calibration row length, and it MUST equal the deployment
# envelope. It sets two things that are indexed by position/row-length:
#   * softmax `calib_T` / `sm_kc_r` — the bidirectional denominator range. Fitting
#     a 64-token row and deploying a <=32-token one sizes the reciprocal for an
#     8x-too-wide domain: MEASURED at calib_T=64 the softmax burned 22
#     bootstraps/block (47% of BERT's bootstrap time) vs ViT-80's 7 at
#     calib_T=26, with n_squarings 3 vs 2 and refinement_iters 2 vs 1.
#   * norm `center_scale_sq[pos]` — one rescale per position, `block_size` long.
#
# DEFAULT 32 = the REAL arm's hard cap (slots/hidDim); bert_forward throws above
# it and the multi-chunk bidirectional arm is unvalidated. NOTE this covers only
# ~76% of SST-2 validation (median 24, p95 44, max 55) — raising it requires the
# multi-chunk arm first, not a bigger BLOCK_T.
#
# The token pool must be pre-cached on a LOGIN node (compute nodes have no
# network): perseus/calibrate/data.py build_token_pool.
set -e -o pipefail
REPO="$(pwd)"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
# UNCONDITIONAL: the login profile exports another project's unwritable HF_HOME
# and sbatch inherits it (a ${HF_HOME:-...} default silently keeps the bad value).
export HF_HOME="$SCRATCH/.cache"
export HF_HUB_CACHE="$SCRATCH/.cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PYTHON="${PYTHON:-$REPO/../he-aware-training/.venv/bin/python}"

BLOCK_T="${BLOCK_T:-32}"
OUT="${OUT:-$REPO/configs/model/approximation/bert_base/configs.json}"
echo "=== BERT calibration: block_size=$BLOCK_T -> $OUT ==="
"$PYTHON" -m perseus.calibrate \
    model=bert_base dataset=sst2 approximation=bert_base \
    model.block_size="$BLOCK_T" calib_out_path="$OUT"
echo "=== Done === $(date)"
