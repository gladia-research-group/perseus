#!/bin/bash
# ResNet-20 on CIFAR-10, encrypted, the first 64 test images: the number is s/image median (of all 64 walls; the first
# image also captures the CUDA graphs); SUMMARY gives top-1 and the agreement with the float64 model.
#   GPU=0 bash scripts/repro/resnet20.sh
source "$(dirname "$0")/common.sh"
bind $PY -m perseus.examples.resnet.run --graphs --n 64 "$@"
