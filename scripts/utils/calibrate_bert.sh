#!/bin/bash
# calibrate_bert.sh — fit the BERT-base approximation config on SST-2 with
# perseus-calibrate. The twin of calibrate_vit.sh.
#   bash scripts/utils/calibrate_bert.sh [hydra overrides...]
# Output: configs/model/approximation/bert_base/configs.json.
#
# BLOCK_T is the calibration row length and MUST equal the deployment envelope. It
# sets two things that are indexed by position or by row length:
#   * softmax `calib_T` / `sm_kc_r` — the bidirectional denominator range. Fitting a
#     64-token row and deploying a <=32-token one sizes the reciprocal for a domain
#     8x too wide: measured at calib_T=64 the softmax burned 22 bootstraps/block,
#     47% of BERT's bootstrap time, against 7 for ViT-80 at calib_T=26, with
#     n_squarings 3 vs 2 and refinement_iters 2 vs 1.
#   * norm `center_scale_sq[pos]` — one rescale per position, `block_size` long.
#
# The default 32 is the real arm's hard cap (slots/hidDim): bert_forward throws above
# it, and the multi-chunk bidirectional arm is unvalidated. That covers about 76% of
# SST-2 validation (median 24, p95 44, max 55); covering the rest needs the
# multi-chunk arm, not a larger BLOCK_T.
#
# The token pool must already be cached; see build_token_pool in perseus/calibrate/data.py.
set -e -o pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" HYDRA_FULL_ERROR=1

BLOCK_T="${BLOCK_T:-32}"
OUT="${OUT:-$REPO/configs/model/approximation/bert_base/configs.json}"
echo "=== BERT calibration: block_size=$BLOCK_T -> $OUT ==="
exec "${PYTHON:-$REPO/.venv/bin/python}" -m perseus.calibrate \
    model=bert_base dataset=sst2 approximation=bert_base \
    model.block_size="$BLOCK_T" calib_out_path="$OUT" "$@"
