#!/bin/bash
# calibrate_gpt2.sh — fit the GPT-2 approximation config (LayerNorm inverse-sqrt, softmax, GELU,
# CutMax) on OpenWebText with perseus-calibrate. CPU/GPU, a few hours on CPU.
#   bash scripts/utils/calibrate_gpt2.sh [hydra overrides...]     # e.g. n_calib_batches=8
# Output: configs/model/approximation/<name>/configs.json (perseus/configs/calibrate.yaml).
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" HYDRA_FULL_ERROR=1
exec "${PYTHON:-$REPO/.venv/bin/python}" -m perseus.calibrate model=gpt2 dataset=openwebtext "$@"
