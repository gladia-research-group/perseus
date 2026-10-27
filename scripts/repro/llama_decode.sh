#!/bin/bash
# Llama-3.1-8B, encrypted decode up to a 512-token context (weights streamed from pinned host memory), teacher-forced on
# the first 512-token OpenWebText window: 511 positions after the BOS prefix; the number is the e2e of the positions
# near 512. TOKENS=16: the short-context decode.
#   GPU=0 NODE=1 MEMS=1,2 bash scripts/repro/llama_decode.sh
source "$(dirname "$0")/common.sh"
bind $PY -m perseus.examples.llama.run_decode --tokens "${TOKENS:-511}" --window 0 --window-tokens 512 \
    --device "$GPU" "$@"
