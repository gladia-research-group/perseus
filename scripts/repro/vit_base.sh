#!/bin/bash
# ViT-B/16 at 224, encrypted, 4 EuroSAT images (pre-normalized (224, 224, 3) arrays): the number is SUMMARY s/image
# (median of the image walls); every block replayed as CUDA graphs, 32 StC-first cache entries.
#   GPU=0 NODE=1 MEMS=1,2 bash scripts/repro/vit_base.sh
source "$(dirname "$0")/common.sh"
bind $PY -m perseus.examples.vit.run_prefill --images scripts/repro/data/vit_eurosat --n 4 --stc-cache 32 --graphs \
    --device "$GPU" "$@"
