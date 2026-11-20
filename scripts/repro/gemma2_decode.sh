#!/bin/bash
# Gemma-2-2B, encrypted decode up to a 512-token context on the decoder port (scripts/repro/llama_decode.sh with
# MODEL=gemma2_2b: its window plans bootstrap_placements/gemma2_2b_decode_ctx512_w<B>, configs gemma2_2b_n32_ctx512).
#   GPU=0 bash scripts/repro/gemma2_decode.sh
MODEL=gemma2_2b exec bash "$(dirname "$0")/llama_decode.sh" "$@"
