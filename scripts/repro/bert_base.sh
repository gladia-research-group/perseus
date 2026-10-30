#!/bin/bash
# BERT-base, encrypted 128-token text (SST-2 head): SUMMARY wall= times the pass after an untimed warm-up on another
# input, every block replayed as CUDA graphs. Needs scipy (float64 reference GELU).
#   GPU=0 NODE=1 MEMS=1,2 bash scripts/repro/bert_base.sh
source "$(dirname "$0")/common.sh"
bind $PY -m perseus.examples.bert.run_bert --text 128 --graphs --device "$GPU" "$@"
