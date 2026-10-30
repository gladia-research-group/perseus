#!/bin/bash
# Llama-3.1-8B, encrypted decode up to a 512-token context (weights streamed from pinned host memory), teacher-forced on
# the first 512-token OpenWebText window: 511 positions after the BOS prefix, one default plan per 128-position window
# (python/perseus/examples/defaults.py); the number is the e2e of the positions near 512. TOKENS=16: the short-context
# decode. PP=<n>: positions 0..n-1 computed in plaintext (the exact fp32 model, --plain-prefix), the encrypted decode
# from n.
#   GPU=0 NODE=1 MEMS=1,2 bash scripts/repro/llama_decode.sh
#   GPU=0 PP=500 bash scripts/repro/llama_decode.sh
source "$(dirname "$0")/common.sh"
bind $PY -m perseus.examples.llama.run_decode --tokens "${TOKENS:-511}" --window 0 --window-tokens 512 \
    ${PP:+--plain-prefix "$PP"} --device "$GPU" "$@"
