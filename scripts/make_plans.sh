#!/usr/bin/env bash
# make_plans.sh — the login-side plan pass (run after run_fleet.sh capture lands).
# FROZEN SCHEME 2026-07-22: ONE plan per matrix case, all emitted by the
# hint-bound planner (plans carry "hint_fire"; the planned runtime obeys them —
# see memory planned-mode-hint-contract). Regenerating a plan is ALWAYS safe
# login-side; a plan predating the current planner binds by luck only.
#   planned_gpt2_base              decode smart cut            (-42%)
#   planned_gpt2_gen               gen keep cut, blocks 0-12 ONLY: the deg-2
#                                  autoregressive entry needs FIRST_ENTRY 16/2 +
#                                  the keep recipe (mid-Goldschmidt auto), and the
#                                  cutmax/feedback tail (block_13/14) runs EAGER —
#                                  its entry-bts iteration count is binary-fragile.
#   planned_vit_base_80            keep-recipe cut             (-63%)
#   planned_vit_complex_112        keep-recipe cut             (-76%)
#   planned_gpt2_prefill_delta_T32/chunk_0   relax cut (.var keep only, -27%)
#     (VALIDATED 2026-07-22: top1 262==262, unplanned=0, 605 bts vs 679 eager;
#      the fuller keep recipe (-8%) is the fallback if a re-capture won't bind)
#   planned_gpt2_prefill_delta_T128/chunk_*  smart cut from the RANGED graph
#     (graph_gpt2_prefill_delta_T128_rng; falls back to skip if not captured yet.
#      chunk_1 needs the annotate pass for its anon K/V cache reads.)
set -e -o pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
PY="${PYTHON:-$REPO/../he-aware-training/.venv/bin/python}"
export PYTHON="$PY"

FAILED=""
plan() {  # plan <graph_dir> <out_name> [env overrides...]
    local g="$1" o="$2"; shift 2
    if [ ! -d "$g" ]; then echo "== plan $o: SKIP (missing graph $g)"; FAILED="$FAILED $o(missing)"; return 0; fi
    echo "== plan $g -> $o"
    if env GRAPH_DIR="$g" OUT_NAME="$o" "$@" bash scripts/utils/run_bootstrap_all_blocks.sh \
        > "logs/core/plan_$(echo "$o" | tr '/' '_').log" 2>&1; then
        echo "   placements: $(env python3 - "$o" <<'EOF'
import json,glob,sys
print(sum(len(json.load(open(f)).get('placements',[])) for f in glob.glob(f'bootstrap_placements/{sys.argv[1]}/**/block_*_placement.json',recursive=True)))
EOF
)"
    else
        echo "   PLAN FAILED: $o (see logs/core/plan_*.log)"; FAILED="$FAILED $o"
    fi
}

KEEP="goldschmidt,inv_sqrt_newton,.var,ln_affine,remez"

plan .cache/graph_gpt2_base planned_gpt2_base
# GPT-2 medium: same bare smart cut as base — same padded tier, no recipe change.
plan .cache/graph_gpt2_medium planned_gpt2_medium
plan .cache/graph_gpt2_gen  planned_gpt2_gen  FIRST_ENTRY_LEVEL=16 FIRST_ENTRY_DEG=2 ERASE_KEEP_STEPS="$KEEP"
# gen: strip the tail blocks -> cutmax + feedback run EAGER (the canon)
rm -f bootstrap_placements/planned_gpt2_gen/block_1[34]_placement.json \
      bootstrap_placements/planned_gpt2_gen/block_1[34]_var_graph.json
# vit80: RELAX (2026-07-27). The keep recipe here was never chosen over relax — it was
# the survivor of the bring-up variant search (docs/HANDOFF_vit_cachefree_perf.txt §8,
# whose aggressive variants predate the hint-bound planner) and was inherited ever since.
# Measured head-to-head, same binary, keep n=4 vs relax n=6:
#   e2e 117.6s -> 110.1s (-6.4%, ranges do not overlap), bts 491 -> 422, placements
#   275 -> 204, unplanned_bts=0, top1=664 canon in all 10 runs, and w_mape IMPROVES
#   0.409 -> 0.337 (relax's whole range sits inside keep's). Known: top5_overlap spreads
#   2-4/5 where keep pinned 3/5 (mean 3.17 vs 3.0) — recorded, not a regression.
# vit112 stays on KEEP until the same A/B is run: it is the complex arm, different
# level behaviour, and untested. BERT stays on KEEP — relax cannot place its blocks 10-11.
# vit112 also RELAX (2026-07-27), but on WEAKER evidence than vit80 — record it as such.
# keep n=4 vs relax n=4: e2e 185.3s -> 180.5s (-2.6%, ranges still do not overlap),
# bts 914 -> 864, placements 550 -> 435, unplanned_bts=0, top1=1 in all 8 runs. Unlike
# vit80, w_mape is FLAT (0.483 -> 0.487) rather than improved, and top5 goes 4,4,4,4 ->
# 4,3,4,4. Real gain, no measured precision cost, but a third the size of vit80's.
# NAMING 2026-08-13: vit_base_80_ft -> vit_base (deployed EuroSAT-ft baseline);
# the old tiny-imagenet graph_vit_base_80 is retired. ⚠ On the bert-heat-branch
# binary the RELAX recipe below does NOT bind for fresh ViT plans
# (plan_level_error, measured on vit_atlas 08-13) — fresh-plan with the KEEP
# recipe ($KEEP) instead; the shipped planned_vit_base is a KEEP plan.
plan .cache/graph_vit_base         planned_vit_base         ERASE_KEEP_STEPS="$KEEP"
plan .cache/graph_vit_complex_112  planned_vit_complex_112  ERASE_KEEP_STEPS=".var"
# BERT: encoder, same keep recipe as ViT. (The earlier note here blamed eager mode for
# the block-2 death — that was wrong: the cause was a calibration-corpus mismatch that
# left ln_2's per-position rescale 5.6x too aggressive, and eager PASSES all 12 blocks
# once calibrated on SST-2. See docs/STATUS_bert_sst2.md. Requires a capture taken
# AFTER that fix — the pre-fix graphs carry 1e35 magnitudes and plan to garbage.)
plan .cache/graph_bert_base        planned_bert_base        ERASE_KEEP_STEPS="$KEEP"
plan .cache/graph_gpt2_prefill_delta_T32/chunk_0 planned_gpt2_prefill_delta_T32/chunk_0 ERASE_KEEP_STEPS=".var"

# T128: RANGED graph -> RELAX smart cut (.var keep only — the validated aggressive
# recipe, chunk_0 -33% vs eager; annotate chunk_1's anon KV reads first).
RNG=.cache/graph_gpt2_prefill_delta_T128
[ -d "$RNG/chunk_1" ] || RNG=.cache/graph_gpt2_prefill_delta_T128_rng
if [ -d "$RNG/chunk_1" ]; then
    "$PY" -m perseus.plan.annotate "$RNG/chunk_1"
    plan "$RNG/chunk_0" planned_gpt2_prefill_delta_T128/chunk_0 ERASE_KEEP_STEPS=".var"
    plan "$RNG/chunk_1" planned_gpt2_prefill_delta_T128/chunk_1 ERASE_KEEP_STEPS=".var"
else
    echo "== T128: ranged graph not captured yet ($RNG) — keeping the current plan set"
fi

# T96 (mixed bucket: packed chunk_0 + REAL tail chunk_1): chunk_0 = the same
# template/recipe as T128 chunk_0; chunk_1 planned from its own ranged capture
# (submit.sh --task prefill96 --stage capture pre-seeds chunk_0 automatically).
T96=.cache/graph_gpt2_prefill_delta_T96
if [ -d "$T96/chunk_1" ]; then
    "$PY" -m perseus.plan.annotate "$T96/chunk_1"
    plan "$T96/chunk_0" planned_gpt2_prefill_delta_T96/chunk_0 ERASE_KEEP_STEPS=".var"
    plan "$T96/chunk_1" planned_gpt2_prefill_delta_T96/chunk_1 ERASE_KEEP_STEPS=".var"
else
    echo "== T96: not captured yet ($T96) — run submit.sh --task prefill96 --stage capture"
fi

echo "=== plan pass done ===  failed/skipped:${FAILED:- none}"
ls bootstrap_placements/
