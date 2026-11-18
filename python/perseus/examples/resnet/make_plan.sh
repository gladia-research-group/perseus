#!/bin/bash
# make_plan.sh — calibrate, capture and plan ResNet-20 / CIFAR-10 (CPU only):
#   bash python/perseus/examples/resnet/make_plan.sh [graphs/resnet20_cifar10] [resnet20_cifar10]
# The ReLU: composite sign (7, 7, 27), gap 2^-7, B = 1.5 x the max |x| of 10 000 training images (examples/resnet/
# calibrate.py). The capture runs on the CPU twin with its magnitudes (the planner prices the refreshes on them) and
# the twin's reactive ceiling at the planner's MAX_LEVEL. The plan is dense (every ciphertext fills the slots) with
# the GPT-2 port's n32 recipe (examples/gpt2/make_plan.sh blocks): ~3 refreshes per ReLU per ciphertext, 74 in all.
set -euo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; REPO="$(dirname "$(dirname "$PKG")")"
GRAPH_DIR="${1:-graphs/resnet20_cifar10}"
OUT_NAME="${2:-resnet20_cifar10}"
cd "$REPO"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$(dirname "$PKG")${PYTHONPATH:+:$PYTHONPATH}"
_venv="$REPO/.venv/bin/python"
[ -x "$_venv" ] || _venv=python3
PYTHON="${PYTHON:-$_venv}"
"$PYTHON" -m perseus.examples.resnet.calibrate --degs 7,7,27 --eps 0.0078125 --margin 1.5 --n 10000
rm -rf "$GRAPH_DIR"
"$PYTHON" -m perseus.examples.resnet.run --fake --n 1 --capture "$GRAPH_DIR"
PLAN_SPARSE_SLOTS= PYTHON="$PYTHON" bash "$PKG/examples/gpt2/make_plan.sh" "$GRAPH_DIR" "$OUT_NAME"
cat > "bootstrap_placements/$OUT_NAME/PLAN_CMD.txt" <<EOF
graph=$GRAPH_DIR
tool=python/perseus/examples/resnet/make_plan.sh
configs=configs/model/approximation/resnet20_cifar10/configs.json
replanned=dense (PLAN_SPARSE_SLOTS=), the n32 blocks recipe
EOF
