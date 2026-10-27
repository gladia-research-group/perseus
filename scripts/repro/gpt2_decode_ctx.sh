#!/bin/bash
# GPT-2 encrypted decode up to a CTX-token context (default 128): CTX positions with the encrypted argmax, teacher-forced
# on a CTX-token reference; the number is the e2e of the positions near CTX in `[precise] tok<p>`. The plan binds at
# every cache length up to GPT-2's 1,024: every position is recorded on the CPU twin at setup and grouped into program
# classes (the softmax's sparse route grows with the cache: Gp x 512 slots). Past 128 the reference is generated:
#   .venv/bin/python scripts/utils/gen_gpt2_oracle.py --model openai-community/gpt2 \
#       --pool .cache/pools/openwebtext_gpt2.npy --out .cache/oracle/gpt2_ctx1024/all_blocks_io --T 16 1024
#   CTX=1024 ALL_BLOCKS_IO_DIR=.cache/oracle/gpt2_ctx1024/all_blocks_io bash scripts/repro/gpt2_decode_ctx.sh
source "$(dirname "$0")/common.sh"
CTX=${CTX:-128}
export STEPS_T=$CTX
bind $PY -m perseus.examples.gpt2.run_decode --tokens "$CTX" --argmax --check \
    --plan bootstrap_placements/gpt2_decode_ctx1024 --device "$GPU" "$@"
