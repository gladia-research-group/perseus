#!/bin/bash
# make_plans.sh — regenerate every plan the paper reports from the shipped graphs (CPU only).
#   bash scripts/make_plans.sh            # all of them (~15 s per plan)
#   bash scripts/make_plans.sh main       # just gpt2_decode_n32
# Each plan directory gets a PLAN_CMD.txt with the recipe; tests/test_paper_plans.py checks the
# committed plans against a regeneration.
set -e -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
export PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES=""            # the planner never touches the GPU
mkdir -p logs/core
ONLY="${1:-all}"

G32=graphs/gpt2_decode_n32
G64=graphs/gpt2_decode_n64
# the n32 recipe (paper Section 3: kappa = 2, target 1e-2, CF window 2..20, sparse routes {512, 1})
N32=(BTS_LEVEL=36 SRC_LEVEL=36 CACHE_READ_LEVEL=36 PLAN_LEVEL_UNIT=2 PLAN_CF_MAX=20
     PLAN_NO_PRESCALE=1 PLAN_SPARSE_SLOTS=512,1 PLAN_SPARSE_BTS_OUT=1:26,512:36 PLAN_ACC_CHAIN=n32
     MAX_LEVEL=50 PLAN_HARD_ENV_CAP=0 PLAN_MAG_SAFETY=2)
# the 64-bit reference chain (28 levels of 53-bit primes). PLAN_ACC_CHAIN= selects the
# analytic error model, which is what the paper's 64-bit rows were planned with; planning
# them against the measured table (PLAN_ACC_CHAIN=n64) moves the counts slightly.
N64=(BTS_LEVEL=18 SRC_LEVEL=17 CACHE_READ_LEVEL=18 PLAN_LEVEL_UNIT=1 PLAN_ACC_CHAIN=
     PLAN_NO_PRESCALE=1 PLAN_CF_MIN=7 PLAN_CF_MAX=20 PLAN_SPARSE_SLOTS=512,1 PLAN_SPARSE_BTS_OUT=1:13,512:18
     MAX_LEVEL=24 PLAN_MAG_SAFETY=2)

plan() {  # plan <graph_dir> <out_name> <recipe...>
    local g="$1" o="$2"; shift 2
    local log="logs/core/plan_$(echo "$o" | tr / _).log"
    [ -d "$g" ] || { echo "== $o: missing graph $g"; return 1; }
    echo "== $o"
    env GRAPH_DIR="$g" OUT_NAME="$o" "$@" bash scripts/utils/run_bootstrap_all_blocks.sh > "$log" 2>&1 \
        || { echo "   FAILED ($log)"; return 1; }
    local chain; chain="$(printf '%s\n' "$@" | sed -n 's/^PLAN_ACC_CHAIN=//p' | tail -1)"
    local acc="analytic error model (no measured table)"
    [ -n "$chain" ] && acc="$(md5sum "perseus/plan/data/bts_accuracy_$chain.json" | cut -d' ' -f1)"
    { echo "graph=$g (md5 $(cat "$g"/block_*/graph.json | md5sum | cut -d' ' -f1))"
      echo "recipe=$*"
      echo "acc_table=$acc"
    } > "bootstrap_placements/$o/PLAN_CMD.txt"
    grep -a '^\[plan\] planned' "$log" | tail -1
}

plan "$G32" gpt2_decode_n32 "${N32[@]}"
[ "$ONLY" = main ] && exit 0
plan "$G64" gpt2_decode_n64 "${N64[@]}"
plan "$G32" gpt2_decode_n32_dense "${N32[@]}" PLAN_SPARSE_SLOTS= PLAN_SPARSE_BTS_OUT=
plan "$G64" gpt2_decode_n64_dense "${N64[@]}" PLAN_SPARSE_SLOTS= PLAN_SPARSE_BTS_OUT=
# baselines: the ported placers run inside our pipeline with the rescue pass (paper Section B)
plan "$G32" baselines/dacapo  "${N32[@]}" MAX_LEVEL=48 PLAN_PLACER=dacapo PLAN_BASELINE_RESCUE=1
plan "$G32" baselines/fhelipe "${N32[@]}" BTS_LEVEL=34 SRC_LEVEL=34 CACHE_READ_LEVEL=34 \
     PLAN_SPARSE_BTS_OUT=1:24,512:34 BTS_OUT_DEG=2 PLAN_PLACER=fhelipe PLAN_BASELINE_RESCUE=1
# dense Fhelipe is planned with the refresh-envelope cap (paper Table 6): without it the arm
# places refreshes past the chain's measured input envelope, where a bootstrap returns garbage
plan "$G32" baselines/fhelipe_dense "${N32[@]}" BTS_LEVEL=34 SRC_LEVEL=34 CACHE_READ_LEVEL=34 \
     BTS_OUT_DEG=2 PLAN_SPARSE_SLOTS= PLAN_SPARSE_BTS_OUT= PLAN_PLACER=fhelipe PLAN_BASELINE_RESCUE=1 \
     PLAN_BASELINE_DEPTH_CAP=48
# baselines/orion is the released Orion tool's output (not regenerable here)
# ablations: one fixed correction factor (Table 4) and the margin kappa (Table 5)
for k in 7 8 9 10; do plan "$G32" ablations/cf_fixed_$k "${N32[@]}" PLAN_CF_MIN=$k PLAN_CF_MAX=$k; done
for k in 1 2 4 8 16 32 64 128; do plan "$G32" ablations/kappa_$k "${N32[@]}" PLAN_MAG_SAFETY=$k; done
