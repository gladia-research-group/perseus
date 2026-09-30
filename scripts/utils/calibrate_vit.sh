#!/bin/bash
# calibrate_vit.sh — fit the ViT-B/16 approximation config on Tiny-ImageNet with
# perseus-calibrate.
#   RES=80  bash scripts/utils/calibrate_vit.sh   # -> vit_base     (T=26, 1 chunk)
#   RES=112 bash scripts/utils/calibrate_vit.sh   # -> vit_base_112 (complex/token-pair, T=50, 2 chunks)
# RES is the sub-native input size and selects the token count the config is fitted
# for; leaving it unset calibrates the native 224 (T=197, 7 chunks), which no shipped
# arm uses. Pass RES.
# Output: configs/model/approximation/vit_base[_<RES>]/configs.json.
#
# The image pool and the hub checkpoint must already be cached; see build_image_pool
# in perseus/calibrate/data.py.
set -e -o pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" HYDRA_FULL_ERROR=1

RES="${RES:-}"
OUT="${OUT:-$REPO/configs/model/approximation/vit_base${RES:+_$RES}/configs.json}"
exec "${PYTHON:-$REPO/.venv/bin/python}" -m perseus.calibrate \
    model=vit_base_224 dataset=tiny_imagenet approximation=vit_base \
    ${RES:+model.resolution=$RES} calib_out_path="$OUT" "$@"
