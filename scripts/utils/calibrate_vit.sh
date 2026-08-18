#!/bin/bash
#SBATCH --job-name vit_calibrate
#SBATCH -A IscrC_eff-SAM2
#SBATCH --time 00:30:00
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --output=logs/core/%x_%j.out
#SBATCH --error=logs/core/%x_%j.err

# ViT-B/16 calibration. RES=80 -> vit_base (ex vit_base_80_ft; live, real arm, T=26, 1 chunk);
# RES=112 -> vit_base_112 (complex/token-pair arm, T=50, 2 chunks).
# RES unset writes the NATIVE vit_base (T=197, 7 chunks) — RETIRED to
# configs/model/approximation/_backup/ 2026-07-20. Always pass RES.
# Image pool + hub checkpoint must be pre-cached (login node): see
# perseus/calibrate/data.py build_image_pool.
set -e -o pipefail
REPO="$(pwd)"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_CACHE="$SCRATCH/.cache/huggingface/hub"
export HF_DATASETS_CACHE="$SCRATCH/.cache/huggingface/datasets"
export HF_HUB_OFFLINE=1
PYTHON="${PYTHON:-$REPO/../he-aware-training/.venv/bin/python}"

# RES: sub-native input size (112 -> 50 tokens, 80 -> 26); empty = native 224.
RES="${RES:-}"
OUT="$REPO/configs/model/approximation/vit_base${RES:+_$RES}/configs.json"
"$PYTHON" -m perseus.calibrate \
    model=vit_base_224 dataset=tiny_imagenet approximation=vit_base \
    ${RES:+model.resolution=$RES} calib_out_path="$OUT"
echo "=== Done === $(date)"
