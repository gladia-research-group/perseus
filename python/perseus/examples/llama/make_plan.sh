#!/bin/bash
# make_plan.sh — plan the bootstraps of a Llama capture (CPU only), with the recipe and runtime levers of the shipped
# GPT-2 arm (bootstrap_placements/gpt2_decode: SparseB on the sparse route (256 slots: Llama's tH), SPRU on the 1-slot route,
# the exact post-raise scaling, the real dense route, raise drops 1,3,5):
#   bash python/perseus/examples/llama/make_plan.sh graphs/llama31_8b_decode llama31_8b_decode
# To reproduce a shipped plan use scripts/make_plans.sh <plan> (or python/tests/test_paper_plans.py <plan>): it exports
# the plan's PLAN_CMD.txt recipe= line, runs its stages and its rd= pass. Run bare, this script falls back to the
# built-in recipe below, which no shipped plan uses as is.
# RUNTIME_EXTRA="KEY=VALUE ..." appends switches the capture ran with to the plan's runtime= line (the run
# applies them), e.g. LLAMA_LINEAR_BOTTOM=1.
# The blocks are 0..n_layers-1 plus the tail (final RMSNorm + LM head); the planner's driver is the GPT-2 port's
# make_plan.sh blocks stage (model-independent: one plan per block_<b>/graph.json).
# The runtime= line carries the levers a capture at HEAD runs with (SM_HRS_PACK .. LADDER_RADIX_BITS; without them the
# contract refuses the plan). No raise drops: a dense-route drop 1 on a SiLU refresh returns garbage (the ctx-512
# recalibration's plan planted one in blocks 20-31; KL ~1e19 from the first token). FIDESLIB_DISCARD_SCRATCH=0: the L2
# discard of dead NTT scratch (FIDESlib default since 797d26f) returns garbage on decode (KL ~1e19 from position 1).
#   ctx-512 plan (CFG=configs/model/approximation/llama31_8b_n32_ctx512/configs.json):
#     python -m perseus.examples.llama.run_decode --tokens 1 --kv-offload --set KV_ARENA_GB=56 \
#         --set FIDESLIB_BTS_RAISE_DROPS=1,3,5 --set SM_CHEB_GS=0 --capture $GR --structure-only --configs $CFG
#     python -m perseus.examples.llama.capture_mags $GR --configs $CFG
#     bash python/perseus/examples/llama/make_plan.sh $GR llama31_8b_decode_ctx512
# A third argument `prefill`: the 128-token prefill (bootstrap_placements/llama31_8b_prefill128), the prefill recipe and
# runtime below (PCMM linears, CCMM attention, paged blocked GQA; the finishes' raise drops PF_PCMM_RAISE / _HI and
# PF_CCMM_RAISE land where the Shadow's `finish_landing` says, on the variants FIDESLIB_BTS_RAISE_DROPS builds), planned
# from a fake capture under that runtime (CPU, ~8 min, ~21 GB):
#     RT=$(python -c "from perseus import config; print(' '.join(k+'='+v for k,v in \
#         config.runtime_env('bootstrap_placements/llama31_8b_prefill128').items()))")
#     env $RT PF_PCMM_CALIB_IN=bootstrap_placements/llama31_8b_prefill128/pcmm_calib.json \
#         python -m perseus.examples.llama.run_prefill --fake --tokens 127 --capture graphs/llama31_8b_prefill128 \
#         --configs configs/model/approximation/llama31_8b_n32_cheb1e-5_kc256/configs.json $(printf -- '--set %s ' $RT)
#     bash python/perseus/examples/llama/make_plan.sh graphs/llama31_8b_prefill128 llama31_8b_prefill128 prefill
set -euo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; REPO="$(dirname "$(dirname "$PKG")")"
GRAPH_DIR="${1:?graph dir (block_<b>/graph.json)}"
OUT_NAME="${2:?out name under bootstrap_placements/}"
PHASE="${3:-decode}"
case "$PHASE" in decode | prefill) ;; *) echo "make_plan.sh: stage $PHASE (decode | prefill)" >&2; exit 2 ;; esac
if [[ -d "$GRAPH_DIR" && ! -d "$REPO/$GRAPH_DIR" ]]; then GRAPH_DIR="$(cd "$GRAPH_DIR" && pwd)"; fi
cd "$REPO"
RECIPE="MAX_LEVEL=50 BTS_LEVEL=34 SRC_LEVEL=36 CACHE_READ_LEVEL=36 PLAN_LEVEL_UNIT=2 PLAN_CF_MAX=20 PLAN_NO_PRESCALE=1 \
PLAN_ACC_CHAIN=n32l PLAN_MAG_SAFETY=2 PLAN_SPARSE_BTS_OUT=1:16,256:24 PLAN_DELIB_LANDING_REMAP=24:14 \
PLAN_HINT_ENV_VETO=1 PLAN_PRUNE=1 PLAN_RAISE_DROP_MAX=0 PLAN_RAISE_ENV_RULE=nominal PLAN_RAISE_DROP_SET=1,3,5 \
PLAN_REAL_ROUTE=1 PLAN_HINT_VETO_STEPS=tail PLAN_RAISE_ROUTES=0 PLAN_REFRESH_ENV_CAP=50 PLAN_REFRESH_ENV_CAP_STEPS=tail:48 \
FORBID_STEPS=.qkv,.gp PLAN_STEP_LANDING=softmax:32,softmax_v:32"
RUNTIME="FIDESLIB_BTS_SHIFT=1 FIDESLIB_SPRU=64 FIDESLIB_BTS_RAISE_DROPS=1,3,5 SM_DEN_RECIP=1 FIDESLIB_BTS_REAL=1 \
GELU_FOLD=1 LN_CHEB=1 SM_GS_FIRST=1 SM_PERIODIC=2 SM_FOLD=1 FIDESLIB_BTS_SPARSE_B=1 FIDESLIB_ROT_KEY_BAND=32 \
KV_LANES=1 CHEB_BSGS=1 QKV_LANES=1 BTS_MAX_INPUT_LEVEL=50 IMPL_SCOPED_NAMES=1 \
SPARSE_BTS_SLOTS=256,1 FIDESLIB_BTS_SPARSE_B_MAX=1024 SPARSE_LEVEL_BUDGET=3,3 \
SM_HRS_PACK=1 CUTMAX_PAIR=1 IMPL_DEAD_OPS=1 LADDER_RADIX_BITS=3 FIDESLIB_DISCARD_SCRATCH=0 ${RUNTIME_EXTRA:-}"
CONFIGS=""
if [ "$PHASE" = prefill ]; then
    RECIPE="MAX_LEVEL=48 BTS_LEVEL=34 SRC_LEVEL=36 CACHE_READ_LEVEL=36 PLAN_LEVEL_UNIT=2 PLAN_CF_MAX=20 \
PLAN_NO_PRESCALE=1 PLAN_ACC_CHAIN=n32l PLAN_MAG_SAFETY=2 PLAN_SPARSE_BTS_OUT=1:16,256:24 PLAN_DELIB_LANDING_REMAP=24:14 \
PLAN_HINT_ENV_VETO=1 PLAN_PRUNE=1 PLAN_RAISE_DROP_MAX=6 PLAN_RAISE_ENV_RULE=nominal PLAN_RAISE_DROP_SET=1,3,5 \
PLAN_REAL_ROUTE=1 PLAN_HINT_VETO_STEPS=tail PLAN_RAISE_ROUTES=0 PLAN_REFRESH_ENV_CAP=48 PLAN_REFRESH_ENV_CAP_STEPS=tail:48 \
FORBID_STEPS=.qkv,.gp PLAN_STEP_LANDING=softmax:32,softmax_v:32 PLAN_DELIB_DEG=captured PLAN_DRIFT_RECUT=1"
    RUNTIME="FIDESLIB_BTS_SHIFT=1 FIDESLIB_SPRU=64 FIDESLIB_BTS_RAISE_DROPS=1,2,4,5,6 SM_DEN_RECIP=1 FIDESLIB_BTS_REAL=1 \
GELU_FOLD=1 LN_CHEB=1 SM_GS_FIRST=1 SM_PERIODIC=2 SM_FOLD=1 FIDESLIB_BTS_SPARSE_B=1 FIDESLIB_ROT_KEY_BAND=32 KV_LANES=1 \
CHEB_BSGS=1 QKV_LANES=1 BTS_MAX_INPUT_LEVEL=50 IMPL_SCOPED_NAMES=1 SPARSE_BTS_SLOTS=256,1 FIDESLIB_BTS_SPARSE_B_MAX=1024 \
SPARSE_LEVEL_BUDGET=3,3 SM_HRS_PACK=1 CUTMAX_PAIR=1 IMPL_DEAD_OPS=1 LADDER_RADIX_BITS=3 PF_LINEAR=pcmm PF_ACT_FOLD=1 \
PF_GELU_JOINT=1 PF_JOINT_FORCE=1 PF_PCMM_WBITS=16 PF_LAYOUT=blocked PF_GQA=1 SM_CHEB_GS=1 PF_ATTN_PACK=3 PF_KV_SHARE=1 \
PF_ATTN=ccmm PF_GLU_XU=1 PF_SINC_A2=1 PF_SILU_SINGLE=1 PF_SM_UNPACK_FOLD=1 PF_SINC_BI_HOIST=1 \
PF_SILU_SINGLE_DEGS=15,23,31,63,127 PF_SILU_SINGLE_ABS=2e-5 PF_SM_R_REFRESH=1 PF_SM_R_SCALE=64 PF_SM_R_ITER=0 \
PF_SM_R_ITER_P=8 FIDESLIB_PCMM_ASYNC=1 PF_CCMM_MERGE=1 PF_SM_PAIR=1 PF_SM_PAIR_ROUNDS=1 PF_GLU_LEAN=1 \
PF_PCMM_RAISE=qkv:4,gate_up:2:0-14,gate_up:2:17-17,gate_up:1:0-23,out:5,down:5 FIDESLIB_GPU_ENCODE=1 \
PF_CHEB_LEAF_NATIVE=1 IMPL_WSUM=1 PF_SILU_SINGLE_ERR=2e-2 PF_PAGED=1 PF_PAGED_STREAM=1 PF_PCMM_RAISE_HI=gate_up:6 \
PF_CCMM_RAISE=pv:4 FIDESLIB_BTS_RAISE_NOSTC=2,6 PF_SM_EXP_EPS=2e-6 PF_PAGED_DEVKV=1 FIDESLIB_STC_POW2_KEY=1 \
FIDESLIB_PCMM_OUT_CHUNK=8 FIDESLIB_PCMM_GEMM_MB=256 FIDESLIB_STC_CACHE_RELEASE=1 FIDESLIB_STC_LAST_CACHE_FREE_GB=4 \
PF_WARMUP=1 FIDESLIB_ENCODE_NOSYNC=1 FIDESLIB_CCMM_ASYNC=1 FIDESLIB_PCMM_NOSYNC=1 FIDESLIB_ENCODE_PINNED_RING=256 \
PF_MASK_HOST_CACHE=1 FIDESLIB_DUMP_FULL=1 FIDESLIB_BTS_RAISE_STC_ONUSE=1 PF_EVICT_ALL_BELOW_GB=3 \
FIDESLIB_CTS_ALT_BUDGET=5 ${RUNTIME_EXTRA:-}"
    CONFIGS=configs/model/approximation/llama31_8b_n32_cheb1e-5_kc256/configs.json
fi
# exported values win over the recipe (PLAN_* experiments)
for kv in $RECIPE; do k=${kv%%=*}; [ -n "${!k+x}" ] || export "$kv"; done
# the sparse routes follow from Llama's layout (tH = 256), not from the shell: scripts/local_env.sh exports GPT-2's
# 512,1, which the recipe loop above would keep (LLAMA_PLAN_SPARSE_SLOTS overrides)
export PLAN_SPARSE_SLOTS="${LLAMA_PLAN_SPARSE_SLOTS:-256,1}"
RECIPE="$RECIPE PLAN_SPARSE_SLOTS=$PLAN_SPARSE_SLOTS"
# KV_ENTRY_LEVEL/DEG: the level the K/V cache reads arrive at (a decode cut for a prefill hand-off's cache);
# PLAN_KEEP_PLACEMENTS: a plan dir (under bootstrap_placements/) whose refresh sites every block keeps
for k in KV_ENTRY_LEVEL KV_ENTRY_DEG PLAN_KEEP_PLACEMENTS; do if [ -n "${!k:-}" ]; then RECIPE="$RECIPE $k=${!k}"; fi; done
bash "$PKG/examples/gpt2/make_plan.sh" "$GRAPH_DIR" "$OUT_NAME"
MD5=$(cat "$GRAPH_DIR"/block_*/graph.json | md5sum | cut -d' ' -f1)
{
    echo "graph=$GRAPH_DIR (md5 $MD5)"
    echo "tool=python/perseus/examples/llama/make_plan.sh$([ "$PHASE" = decode ] || echo " ($PHASE)")"
    echo "recipe=$(for kv in $RECIPE; do k=${kv%%=*}; echo -n "$k=${!k} "; done | sed 's/ $//')"
    echo "runtime=$(echo $RUNTIME)"
    [ -z "$CONFIGS" ] || echo "configs=$CONFIGS"
} > "bootstrap_placements/$OUT_NAME/PLAN_CMD.txt"
# a prefill run loads the graph's rotation keys: written next to the plan, which then runs without the recording
if [ -n "$CONFIGS" ]; then
    PYTHONPATH="$(dirname "$PKG")${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON:-$REPO/.venv/bin/python}" -c \
        'import sys; from perseus.impl.magtape import write_plan_rot_steps as w; w(sys.argv[1])' "bootstrap_placements/$OUT_NAME"
fi
cat "bootstrap_placements/$OUT_NAME/PLAN_CMD.txt"
