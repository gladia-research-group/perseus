#!/bin/bash
# GPT-2, encrypted 128-token prefill (the Prefill-128 record's plan, with its tail; the warm-up captures the stages as
# CUDA graphs, as the server does): SUMMARY wall= times the pass after an untimed warm-up on another prompt.
#   GPU=0 NODE=1 MEMS=1,2 bash scripts/repro/gpt2_prefill128.sh
source "$(dirname "$0")/common.sh"
bind $PY -m perseus.examples.gpt2.run_prefill --tokens 128 --plan bootstrap_placements/gpt2_prefill128_n32a4_pfbhrd_cmw_mfsab2q --tail \
    --graphs --check --device "$GPU" "$@"
