#!/bin/bash
# GPT-2 encrypted decode up to a CTX-token context (default 512): CTX positions with the encrypted argmax, teacher-forced
# on a CTX-token reference; the number is the e2e of the positions near CTX in `[precise] tok<p>`. The default plans
# (python/perseus/examples/defaults.py) bind at every cache length: up to 512 one per 128-position window, past it one
# up to GPT-2's 1,024. Every position is recorded on the CPU twin at setup and grouped into program classes (the
# softmax's sparse route grows with the cache: Gp x 512 slots). PP=<n>: positions 0..n-1 computed in plaintext on the
# CPU (--plain-prefix), the encrypted decode from n. Past 128 the reference of the CTX-token context is generated:
#   .venv/bin/python scripts/utils/gen_gpt2_oracle.py --model openai-community/gpt2 \
#       --pool .cache/pools/openwebtext_gpt2.npy --out .cache/oracle/gpt2_ctx$CTX/all_blocks_io --T 16 $CTX
#   GPU=0 PP=500 bash scripts/repro/gpt2_decode_ctx.sh
#   CTX=1024 bash scripts/repro/gpt2_decode_ctx.sh
CTX=${CTX:-512}
# past 128 positions the reference of that context (before local_env.sh's default)
[ "$CTX" -le 128 ] || export ALL_BLOCKS_IO_DIR="${ALL_BLOCKS_IO_DIR:-${PERSEUS_DATA:-.cache}/oracle/gpt2_ctx$CTX/all_blocks_io}"
source "$(dirname "$0")/common.sh"
export STEPS_T=$CTX
bind $PY -m perseus.examples.gpt2.run_decode --tokens "$CTX" --argmax --check ${PP:+--plain-prefix "$PP"} \
    --device "$GPU" "$@"
