#!/bin/bash
# Llama-3.1-8B, encrypted 128-token prefill. Every switch rides in the plan's runtime= line. The wall (SUMMARY wall=) is
# the timed pass after an untimed warm-up on another window: keys, masks and block-0 setup are excluded. Every block
# replayed as CUDA graphs, 4 StC-first cache entries.
#   GPU=0 NODE=1 MEMS=1,2 bash scripts/repro/llama_prefill128.sh
source "$(dirname "$0")/common.sh"
bind $PY -m perseus.examples.llama.run_prefill --tokens 128 --graphs --stc-cache 4 --device "$GPU" "$@"
