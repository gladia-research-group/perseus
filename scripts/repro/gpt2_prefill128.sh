#!/bin/bash
# GPT-2, encrypted 128-token prefill to the first token (the default plan): the 12 blocks, the last position's final
# LayerNorm and LM head, and the encrypted argmax, replayed as CUDA graphs; SUMMARY wall= times the pass after an untimed
# warm-up on another prompt.
#   GPU=0 NODE=1 MEMS=1,2 bash scripts/repro/gpt2_prefill128.sh
source "$(dirname "$0")/common.sh"
bind $PY -m perseus.examples.gpt2.run_prefill --tokens 128 --tail --graphs --check --device "$GPU" "$@"
