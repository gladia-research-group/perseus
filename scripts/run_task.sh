#!/bin/bash
#SBATCH --job-name fhe_task
#SBATCH -A EUHPC_D34_099
#SBATCH --time 02:00:00
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=256G
#SBATCH --output=logs/core/%x_%j.out
#SBATCH --error=logs/core/%x_%j.err

# =============================================================================
#  run_task.sh — THE capture/run script. One task = one job, discriminated by
#  env only. Companion scripts: submit.sh (flag CLI to submit many), make_plans.sh (login
#  plan pass). Old per-case scripts: scripts/_backup/.
#
#    TASK   = decode | decode_medium | gen | gen_prefill | handoff | prefill32
#           | prefill96 | prefill128 | vit80 | vit112 | bert                     (required)
#    STAGE  = run      planned threaded run — the matrix bar         (default)
#           | eager    threaded run without a plan
#           | capture  sync graph capture -> FHE_GRAPH_DIR
#    RUNNER = python   perseus._core via scripts/modes_baseline.py — the ONE
#                      driver for both models (MODEL=gpt2|vit)       (default)
#           | cuda     the native cuda_cachemir CLI (same env contract;
#                      GPT-2 tasks only — ViT has no CLI driver)
#
#  Every preset uses the REAL env names with ${VAR:-default}: whatever you pass
#  at sbatch time wins. Diagnostics are env overrides, not new scripts:
#    TASK=decode sbatch scripts/run_task.sh                       # the bar
#    TASK=gen GPT2_INFERENCE_MODE=sync sbatch scripts/run_task.sh # strict throw
#    TASK=prefill32 FHE_BOOTSTRAP_PLACEMENTS_DIR=$PWD/bootstrap_placements/x \
#        sbatch scripts/run_task.sh                               # plan A/B
#    TASK=prefill128 STAGE=capture PREFILL_T=64 sbatch scripts/run_task.sh
#  Submit from a CLEAN shell (no modules loaded) — see CLAUDE.md.
# =============================================================================
set -e -o pipefail

# ---------------------------------------------------------------- boilerplate
REPO="$(pwd)"; DEPS="$REPO/deps"
# The he-aware-training checkout sits in one of two places depending on the site: a SIBLING of
# this repo (Leonardo) or its PARENT two levels up (behemoth, where this repo is nested at
# he-aware-training/src/perseus). Resolve it instead of assuming, so paths built from it —
# the venv, the ViT plaintext backbones — do not silently point at nothing.
HOME_REPO="${HOME_REPO:-}"
if [ -z "$HOME_REPO" ]; then
  for _c in "$REPO/../he-aware-training" "$REPO/../.." "$REPO/.."; do
    [ -d "$_c/.venv" ] && { HOME_REPO="$(cd "$_c" && pwd)"; break; }
  done
fi
[ -n "$HOME_REPO" ] || echo "[run_task] WARN: he-aware-training checkout not found; set HOME_REPO explicitly" >&2
module load cuda/12.6 gcc cmake nccl
module use /leonardo/prod/spack/06/modules/0.22.2_preprod_base
module load libarchive/3.7.1--gcc--12.2.0-sw6t2mm
export LD_LIBRARY_PATH="$DEPS/lib:$DEPS/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1
export TMPDIR="$SCRATCH/tmp.${SLURM_JOB_ID:-login}"
mkdir -p "$TMPDIR" logs/core
PYTHON="${PYTHON:-$HOME_REPO/.venv/bin/python}"

TASK="${TASK:?set TASK=decode|decode_medium|gen|gen_prefill|handoff|prefill32|prefill96|prefill128|vit80|vit112|bert}"
STAGE="${STAGE:-run}"
RUNNER="${RUNNER:-python}"
BP="$REPO/bootstrap_placements"; CFG="$REPO/configs/model/approximation"

[ "${BUILD:-0}" = "1" ] && cmake --build build-py --parallel 16 --target _core

# ------------------------------------------------- frozen CKKS chain (shared)
export LOGN="${LOGN:-16}"
export AUTO_BTS_LEVEL="${AUTO_BTS_LEVEL:-24}"
export BTS_ITERATIONS="${BTS_ITERATIONS:-1}"

# ------------------------------------------------------------ per-task preset
# DRIVER: gpt2 -> modes_baseline.py $MODE | vit -> encvit_forward.py
# PLAN_DEFAULT/GRAPH_DEFAULT: the canonical plan / capture dir for the task.
DRIVER=gpt2
case "$TASK" in
  decode)
    MODE=decode
    export CKKS_COMPLEX="${CKKS_COMPLEX:-1}"
    export GPT2_PACKING="${GPT2_PACKING:-cachemir}"
    export FIDESLIB_ROT_KEY_BAND="${FIDESLIB_ROT_KEY_BAND:-11}"
    export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
    PLAN_DEFAULT="$BP/planned_gpt2_base"
    GRAPH_DEFAULT="$REPO/.cache/graph_gpt2_base"
    ;;
  decode_medium)
    # GPT-2 MEDIUM (24L/1024/16H/4096). Same padded tier as base (hidDim=1024,
    # t=32) => identical packing, keys and per-block cost; only n_layers differs,
    # so the ONLY deltas vs `decode` are the model paths + the KV arena.
    MODE=decode
    export CKKS_COMPLEX="${CKKS_COMPLEX:-1}"
    export GPT2_PACKING="${GPT2_PACKING:-cachemir}"
    export FIDESLIB_ROT_KEY_BAND="${FIDESLIB_ROT_KEY_BAND:-11}"
    export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
    # 24 layers ~= 2x base's decode KV (~9GB): the 12GB default EXHAUSTS the
    # pinned arena at token 0 (job 50175030, kv_slot_ensure throw).
    export KV_ARENA_GB="${KV_ARENA_GB:-24}"
    export CONFIGS_PATH="${CONFIGS_PATH:-$CFG/gpt2_medium/configs.json}"
    export WEIGHTS_PATH="${WEIGHTS_PATH:-$SCRATCH/.cache/perseus/models/openai-community/gpt2-medium/classic/weights.bin.zip}"
    export ALL_BLOCKS_IO_DIR="${ALL_BLOCKS_IO_DIR:-$SCRATCH/.cache/perseus/oracle/gpt2-medium/all_blocks_io}"
    PLAN_DEFAULT="$BP/planned_gpt2_medium"
    GRAPH_DEFAULT="$REPO/.cache/graph_gpt2_medium"
    ;;
  gen)
    MODE=gen
    export GEN_TOKENS="${GEN_TOKENS:-8}"
    export CKKS_COMPLEX="${CKKS_COMPLEX:-1}"
    export GPT2_PACKING="${GPT2_PACKING:-cachemir}"
    export FIDESLIB_ROT_KEY_BAND="${FIDESLIB_ROT_KEY_BAND:-11}"
    export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
    PLAN_DEFAULT="$BP/planned_gpt2_gen"           # keep cut, blocks 0-12; cutmax EAGER
    GRAPH_DEFAULT="$REPO/.cache/graph_gpt2_gen"
    ;;
  gen_prefill|handoff)
    MODE="$TASK"                                  # correctness rows: eager by design
    export GEN_TOKENS="${GEN_TOKENS:-8}"
    export CKKS_COMPLEX="${CKKS_COMPLEX:-1}"
    export GPT2_PACKING="${GPT2_PACKING:-cachemir}"
    export FHE_DELTA_BLOCK="${FHE_DELTA_BLOCK:-1}"
    export FIDESLIB_ROT_KEY_BAND="${FIDESLIB_ROT_KEY_BAND:-11}"
    export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
    # NO staging here (2026-07-22): the staged+released prefill weights trip the
    # second-extraction guard in the handoff DECODE phase (job 50098456) — these
    # are correctness rows, wall is irrelevant; they keep the validated unstaged
    # form. Staging the mixed prefill->decode session is a recorded open lead.
    export FHE_PT_STAGE_BLOCK="${FHE_PT_STAGE_BLOCK:-0}"
    PLAN_DEFAULT=""; GRAPH_DEFAULT=""
    ;;
  prefill32)
    # REAL arm (T<=32): complex payload is pure overhead at ns_im=0 and trips the
    # mixed-arm tail bug; band=-1 (real filling rotations dip below L16, banded
    # keys throw there); weight granularity MUST stay Plaintext (linear corrupts
    # the real arm — open bug, memory prefill-linear-gran-real-arm-bug).
    MODE=prefill
    export PREFILL_T="${PREFILL_T:-32}"
    export CKKS_COMPLEX="${CKKS_COMPLEX:-0}"
    export GPT2_PACKING="${GPT2_PACKING:-cachemir}"
    export FHE_DELTA_BLOCK="${FHE_DELTA_BLOCK:-1}"
    export GPT2_LMHEAD_GRANULARITY="${GPT2_LMHEAD_GRANULARITY:-linear}"
    export FIDESLIB_ROT_KEY_BAND="${FIDESLIB_ROT_KEY_BAND:--1}"
    export MALLOC_TOP_PAD_="${MALLOC_TOP_PAD_:-134217728}"; unset MALLOC_ARENA_MAX
    # staging+hybrid-coeff+release-CPU are CODE DEFAULTS since the freeze; only sizing stays:
    export FHE_STAGE_ARENA_GB="${FHE_STAGE_ARENA_GB:-56}"   # non-coeff pts are full-size: ~54GB/block
    export FHE_PT_STAGE_BLOCK="${FHE_PT_STAGE_BLOCK:-16}"   # streamed-weight arm
    export FHE_STAGE_RELEASE_CPU="${FHE_STAGE_RELEASE_CPU:-1}"  # arm-coupled: staged blocks free host RAM
    PLAN_DEFAULT="$BP/planned_gpt2_prefill_delta_T32"
    GRAPH_DEFAULT="$REPO/.cache/graph_gpt2_prefill_delta_T32"
    ;;
  prefill128)
    # COMPLEX token-pair arm + ViT-regime weights: linear granularity + top-pad
    # cut prefill+handoff 809.6 -> 618.4 s, plan-neutral. band=11 fits memory.
    MODE=prefill
    export PREFILL_T="${PREFILL_T:-128}"
    export CKKS_COMPLEX="${CKKS_COMPLEX:-1}"
    export GPT2_PACKING="${GPT2_PACKING:-cachemir}"
    export FHE_DELTA_BLOCK="${FHE_DELTA_BLOCK:-1}"
    export KV_ARENA_GB="${KV_ARENA_GB:-24}"
    export GPT2_LMHEAD_GRANULARITY="${GPT2_LMHEAD_GRANULARITY:-linear}"
    export FIDESLIB_ROT_KEY_BAND="${FIDESLIB_ROT_KEY_BAND:-11}"
    export MALLOC_TOP_PAD_="${MALLOC_TOP_PAD_:-134217728}"; unset MALLOC_ARENA_MAX
    # staging+hybrid-coeff+release-CPU are CODE DEFAULTS since the freeze; only sizing stays:
    export FHE_STAGE_ARENA_GB="${FHE_STAGE_ARENA_GB:-56}"   # non-coeff pts are full-size: ~54GB/block
    export FHE_PT_STAGE_BLOCK="${FHE_PT_STAGE_BLOCK:-16}"   # streamed-weight arm
    export FHE_STAGE_RELEASE_CPU="${FHE_STAGE_RELEASE_CPU:-1}"  # arm-coupled: staged blocks free host RAM
    PLAN_DEFAULT="$BP/planned_gpt2_prefill_delta_T128"
    GRAPH_DEFAULT="$REPO/.cache/graph_gpt2_prefill_delta_T128"
    ;;
  prefill96)
    # MIXED bucket (64<T<=96): packed complex chunk_0 (the T128 chunk_0 template)
    # + a REAL tail chunk_1, all inside the COMPLEX build (greedy one-payload
    # chunking flips token_pair per chunk). Env = the complex-prefill best set.
    # OPEN QUESTION recorded for the validation run: the real TAIL chunk under
    # linear granularity may hit the real-arm off-level value bug — if it does,
    # this task drops to GPT2_PREFILL_GRANULARITY=plaintext (record the outcome).
    MODE=prefill
    export PREFILL_T="${PREFILL_T:-96}"
    export CKKS_COMPLEX="${CKKS_COMPLEX:-1}"
    export GPT2_PACKING="${GPT2_PACKING:-cachemir}"
    export FHE_DELTA_BLOCK="${FHE_DELTA_BLOCK:-1}"
    export KV_ARENA_GB="${KV_ARENA_GB:-24}"
    export GPT2_LMHEAD_GRANULARITY="${GPT2_LMHEAD_GRANULARITY:-linear}"
    export FIDESLIB_ROT_KEY_BAND="${FIDESLIB_ROT_KEY_BAND:-11}"
    export MALLOC_TOP_PAD_="${MALLOC_TOP_PAD_:-134217728}"; unset MALLOC_ARENA_MAX
    # staging+hybrid-coeff+release-CPU are CODE DEFAULTS since the freeze; only sizing stays:
    export FHE_STAGE_ARENA_GB="${FHE_STAGE_ARENA_GB:-56}"   # non-coeff pts are full-size: ~54GB/block
    export FHE_PT_STAGE_BLOCK="${FHE_PT_STAGE_BLOCK:-16}"   # streamed-weight arm
    export FHE_STAGE_RELEASE_CPU="${FHE_STAGE_RELEASE_CPU:-1}"  # arm-coupled: staged blocks free host RAM
    PLAN_DEFAULT="$BP/planned_gpt2_prefill_delta_T96"
    GRAPH_DEFAULT="$REPO/.cache/graph_gpt2_prefill_delta_T96"
    ;;
  bert)
    # BERT-base SST-2 (12L/768/12H/3072) — the POST-LN encoder. Shares the ViT
    # encoder context (filling packing, bidirectional, no KV cache) and its
    # residency/staging envelope; only the block ORDERING differs (LN wraps the
    # residual). One packed chunk: SST-2 sentences are short, and the multi-chunk
    # bidirectional arm is unvalidated. Head (pooler+tanh+classifier) is client-side.
    DRIVER=bert
    MODE=forward
    export MODEL=bert
    export GATE_BLOCKS="${GATE_BLOCKS:-12}"
    export CONFIGS_PATH="${CONFIGS_PATH:-$CFG/bert_base/configs.json}"
    export BERT_MODEL="${BERT_MODEL:-textattack/bert-base-uncased-SST-2}"
    export BERT_MODEL_DIR="${BERT_MODEL_DIR:-$SCRATCH/.cache/perseus/models/textattack/bert-base-uncased-SST-2/classic}"
    # REAL arm (like vit80). The token-pair tail needs im_cleanse + the paired 0.5
    # before extract_token_i_cachemir (vit_model.cu:344-352) and neither primitive
    # is bound to Python, so the complex arm is unreachable from this driver. The
    # real arm gives slots/hidDim = 32 tokens/chunk, ample for SST-2 (8-12 tokens).
    export CKKS_COMPLEX="${CKKS_COMPLEX:-0}"
    # UNCONDITIONAL (like the ViT arm): the login profile exports an HF_HOME under
    # another project's scratch, which sbatch inherits and which is not writable —
    # a ${HF_HOME:-...} default would silently keep it (job 50189967).
    export HF_HOME="$SCRATCH/.cache"
    export HF_HUB_CACHE="$SCRATCH/.cache"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export FHE_DELTA_BLOCK="${FHE_DELTA_BLOCK:-1}"
    export FHE_KV_OVERLAP="${FHE_KV_OVERLAP:-0}"
    export FHE_STAGE_ARENA_GB="${FHE_STAGE_ARENA_GB:-12}"
    export FHE_PT_STAGE_BLOCK="${FHE_PT_STAGE_BLOCK:-16}"
    export FHE_STAGE_RELEASE_CPU="${FHE_STAGE_RELEASE_CPU:-1}"
    export MALLOC_TOP_PAD_="${MALLOC_TOP_PAD_:-134217728}"; unset MALLOC_ARENA_MAX
    unset GPT2_FOLD_LN1 GPT2_FOLD_LN2 GPT2_FOLD_LNF
    # capture dips below L16 (eager sync trajectories) -> full keys; run bands.
    if [ "$STAGE" = "capture" ]; then
        export FIDESLIB_ROT_KEY_BAND="${FIDESLIB_ROT_KEY_BAND:--1}"
        export BERT_MODE=sync
    else
        export FIDESLIB_ROT_KEY_BAND="${FIDESLIB_ROT_KEY_BAND:-11}"
    fi
    PLAN_DEFAULT="$BP/planned_bert_base"
    GRAPH_DEFAULT="$REPO/.cache/graph_bert_base"
    ;;
  vit80|vit112)
    DRIVER=vit
    MODE=forward
    export MODEL=vit
    if [ "$TASK" = "vit80" ]; then
        export GATE_RES="${GATE_RES:-80}"                      # 26 tok, one REAL chunk
        export CONFIGS_PATH="${CONFIGS_PATH:-$CFG/vit_base/configs.json}"
        unset CKKS_COMPLEX
        PLAN_DEFAULT="$BP/planned_vit_base"
        GRAPH_DEFAULT="$REPO/.cache/graph_vit_base"
        # ASSET TIER — must match the config. `vit_base` is the EuroSAT-ft 80px arm (it was
        # `vit_base_80_ft` before the 2026-08-13 rename); the tiny-imagenet `classic` weights,
        # the tiny-imagenet pool and canon class 664 belong to a DIFFERENT arm. Pairing them
        # is a config-vs-weights mismatch, which does not error — it produces a complete run
        # with a wrong answer. vit112 below still legitimately uses the tiny-imagenet tier.
        VIT_ASSET_TIER=eurosat
        export VIT_MODEL_DIR="${VIT_MODEL_DIR:-$SCRATCH/.cache/perseus/models/google/vit-base-patch16-224/ft80}"
        export VIT_MODEL="${VIT_MODEL:-$HOME_REPO/checkpoints/vit/eurosat_vit80/hf}"
        export VIT_POOL="${VIT_POOL:-$SCRATCH/.cache/huggingface/perseus/pools/eurosat_vit-base-patch16-224_512.npy}"
        # NO VIT_CANON_TOP1 default: 664 is the tiny-imagenet canon class and is WRONG for
        # every EuroSAT arm. Set it explicitly only when a canon class has been established.
    else
        export GATE_RES="${GATE_RES:-112}"                     # 50 tok, one token-pair chunk
        export CONFIGS_PATH="${CONFIGS_PATH:-$CFG/vit_base_112/configs.json}"
        export CKKS_COMPLEX="${CKKS_COMPLEX:-1}"
        PLAN_DEFAULT="$BP/planned_vit_complex_112"
        GRAPH_DEFAULT="$REPO/.cache/graph_vit_complex_112"
    fi
    export GATE_BLOCKS="${GATE_BLOCKS:-12}"
    # staging+hybrid-coeff+release-CPU are CODE DEFAULTS since the freeze; only sizing stays:
    export FHE_STAGE_ARENA_GB="${FHE_STAGE_ARENA_GB:-12}"

    export VIT_MODEL_DIR="${VIT_MODEL_DIR:-$SCRATCH/.cache/perseus/models/google/vit-base-patch16-224/classic}"
    export VIT_POOL="${VIT_POOL:-$SCRATCH/.cache/huggingface/perseus/pools/tiny_imagenet_vit-base-patch16-224_512.npy}"
    export HF_HOME="$SCRATCH/.cache/huggingface"
    export HF_HUB_CACHE="$SCRATCH/.cache/huggingface/hub"
    export HF_HUB_OFFLINE=1
    export FHE_DELTA_BLOCK="${FHE_DELTA_BLOCK:-1}"
    export FHE_KV_OVERLAP="${FHE_KV_OVERLAP:-0}"
    export MALLOC_TOP_PAD_="${MALLOC_TOP_PAD_:-134217728}"; unset MALLOC_ARENA_MAX
    export FHE_PT_STAGE_BLOCK="${FHE_PT_STAGE_BLOCK:-16}"   # streamed-weight arm
    export FHE_STAGE_RELEASE_CPU="${FHE_STAGE_RELEASE_CPU:-1}"  # arm-coupled: staged blocks free host RAM
    unset GPT2_FOLD_LN1 GPT2_FOLD_LN2 GPT2_FOLD_LNF
    # run: band=11 (min plan pin >=16 = level-safe; full keys OOM the 112 chunk).
    # capture: band=-1 (eager sync trajectories dip below L16).
    if [ "$STAGE" = "capture" ]; then
        export FIDESLIB_ROT_KEY_BAND="${FIDESLIB_ROT_KEY_BAND:--1}"
    else
        export FIDESLIB_ROT_KEY_BAND="${FIDESLIB_ROT_KEY_BAND:-11}"
    fi
    ;;
  *) echo "unknown TASK $TASK"; exit 1 ;;
esac

# -------------------------------------- GPT-2 shared model envs (post-preset)
if [ "$DRIVER" = "gpt2" ]; then
    export MULTI_T="${MULTI_T:-4}"
    export STEPS_T="${STEPS_T:-128}"
    export GPT2_CACHE="${GPT2_CACHE:-1}"
    export GPT2_FOLD_LN1="${GPT2_FOLD_LN1:-1}"
    export GPT2_FOLD_LN2="${GPT2_FOLD_LN2:-1}"
    export GPT2_FOLD_LNF="${GPT2_FOLD_LNF:-0}"
    export CACHE_READ_LEVEL_K="${CACHE_READ_LEVEL_K:-17}"
    export CACHE_READ_LEVEL_V="${CACHE_READ_LEVEL_V:-17}"
    export FHE_LMHEAD_CAP="${FHE_LMHEAD_CAP:-22}"
    export CUTMAX_PRECISE_SCOPED="${CUTMAX_PRECISE_SCOPED:-1}"
    export CUTMAX_VEC_BTS_ITERS="${CUTMAX_VEC_BTS_ITERS:-1}"
    export CONFIGS_PATH="${CONFIGS_PATH:-$CFG/gpt2_base/configs.json}"
    export WEIGHTS_PATH="${WEIGHTS_PATH:-$SCRATCH/.cache/perseus/models/openai-community/gpt2/classic/weights.bin.zip}"
fi

# ---------------------------------------------------------------- stage wiring
# FHE_DECODE_PLACEMENTS_DIR (the decode-PHASE plan for handoff/gen_prefill runs)
# is honored when the caller sets it; plan-free otherwise.
# FHE_PROFILE honored when the caller sets it (wall|events; wall syncs per op —
# attribution numbers, NOT canon walls); clean runs stay unprofiled.
[ -n "${FHE_PROFILE:-}" ] && export FHE_PROFILE || unset FHE_PROFILE
[ -n "${FHE_DECODE_PLACEMENTS_DIR:-}" ] && export FHE_DECODE_PLACEMENTS_DIR || unset FHE_DECODE_PLACEMENTS_DIR
case "$STAGE" in
  capture)
    export GPT2_INFERENCE_MODE=sync VIT_MODE=sync FIDESLIB_LAZY_CPU_SHADOW=0
    unset FHE_BOOTSTRAP_PLACEMENTS_DIR
    export FHE_GRAPH_DIR="${FHE_GRAPH_DIR:-$GRAPH_DEFAULT}"
    [ -n "$FHE_GRAPH_DIR" ] || { echo "TASK=$TASK has no capture graph"; exit 1; }
    [ "$MODE" = "prefill" ] && export FHE_PREFILL_CAPTURE_RANGES="${FHE_PREFILL_CAPTURE_RANGES:-1}"
    [ "$MODE" = "gen" ]     && export GEN_TOKENS=2   # tok0 + the entry-pinning feedback token
    if [ "${RESUME_GRAPH:-0}" != "1" ]; then
        # never destroy a prior capture: archive it (user ruling 2026-07-28) —
        # graphs are the reproducibility record of every measured row
        if [ -d "$FHE_GRAPH_DIR" ]; then
            mkdir -p "$REPO/.cache/_graph_archive"
            mv "$FHE_GRAPH_DIR" "$REPO/.cache/_graph_archive/$(basename "$FHE_GRAPH_DIR")_$(date +%Y%m%d_%H%M%S)"
        fi
    else
        # resume: a COMPLETE chunk (13 block graphs) is pre-seeded and skipped by
        # the capture; an incomplete one would be skipped wrongly — drop it.
        for c in "$FHE_GRAPH_DIR"/chunk_*; do
            [ -d "$c" ] || continue
            n=$(ls "$c"/block_*/graph.json 2>/dev/null | wc -l)
            [ "$n" -lt 13 ] && { echo "[task] resume: dropping incomplete $c ($n/13)"; rm -rf "$c"; }
        done
    fi
    mkdir -p "$FHE_GRAPH_DIR"
    # pre-capture lint: adaptive counts on base/squeeze tiers + frozen cutmax
    # (2026-07-28: fixed-count gpt2 baselines slipped into a capture round)
    "$PYTHON" scripts/utils/lint_approx_config.py "$CONFIGS_PATH" \
        || { echo "[task] config lint FAILED — refusing to capture"; exit 1; }
    echo "[task] CAPTURE $TASK -> $FHE_GRAPH_DIR (sync, ranged=${FHE_PREFILL_CAPTURE_RANGES:-native})"
    ;;
  run)
    export GPT2_INFERENCE_MODE="${GPT2_INFERENCE_MODE:-threaded}"
    export VIT_MODE="${VIT_MODE:-threaded}"
    unset FHE_GRAPH_DIR
    # a caller-provided plan dir ALWAYS wins — even for tasks whose preset default
    # is eager (e.g. a planned-handoff experiment passes the prefill plan explicitly)
    if [ -n "${FHE_BOOTSTRAP_PLACEMENTS_DIR:-}" ] || [ -n "$PLAN_DEFAULT" ]; then
        export FHE_BOOTSTRAP_PLACEMENTS_DIR="${FHE_BOOTSTRAP_PLACEMENTS_DIR:-$PLAN_DEFAULT}"
        [ -d "$FHE_BOOTSTRAP_PLACEMENTS_DIR" ] || { echo "missing plan dir $FHE_BOOTSTRAP_PLACEMENTS_DIR"; exit 1; }
        echo "[task] PLANNED RUN $TASK <- $FHE_BOOTSTRAP_PLACEMENTS_DIR"
    else
        unset FHE_BOOTSTRAP_PLACEMENTS_DIR
        echo "[task] RUN $TASK (eager by design)"
    fi
    ;;
  eager)
    export GPT2_INFERENCE_MODE="${GPT2_INFERENCE_MODE:-threaded}"
    export VIT_MODE="${VIT_MODE:-threaded}"
    unset FHE_GRAPH_DIR FHE_BOOTSTRAP_PLACEMENTS_DIR
    echo "[task] EAGER RUN $TASK"
    ;;
  *) echo "unknown STAGE $STAGE"; exit 1 ;;
esac

# ------------------------------------------------------------------ exec + gate
# Both runners share the SAME env contract; only the entry differs. The CLI's
# flags are pure env overrides (see src/app/cli.cu usage), so the env set built
# above drives either. PASS gating: exit codes are NOT evidence (FIDESlib
# teardown can exit(0) on fatal CUDA errors) — require the run's own marker.
LOGT="logs/core/.task_${TASK}_${SLURM_JOB_ID:-$$}.out"

run_python() {
    # ONE driver for both models: modes_baseline.py (MODEL env selects gpt2/vit;
    # vit accepts only MODE=forward). Uniform acceptance marker for every mode.
    OKPAT="^\[$MODE\] PASS$"
    env "$PYTHON" scripts/modes_baseline.py "$MODE" | tee "$LOGT" || true
}

run_cuda() {
    [ "$DRIVER" = "gpt2" ] || { echo "[task] RUNNER=cuda: ViT has no CLI driver"; exit 1; }
    local BIN=""
    for c in "${CUDA_CLI:-}" build-py/cuda_cachemir build-py/bin/cuda_cachemir; do
        [ -n "$c" ] && [ -x "$c" ] && { BIN="$c"; break; }
    done
    [ -n "$BIN" ] || { echo "[task] cuda_cachemir not built: cmake --build build-py --target cuda_cachemir"; exit 1; }
    # env aliases the CLI reads (python's modes_baseline derives these itself)
    local SUB
    case "$MODE" in
        decode)              SUB=decode ;;
        prefill)             SUB=prefill; export PREFILL_TOKENS="$PREFILL_T" DECODE_TOKENS="${DECODE_TOKENS:-0}" ;;
        handoff)             SUB=prefill; export PREFILL_TOKENS="${PREFILL_T:-8}" DECODE_TOKENS="${DECODE_TOKENS:-2}" ;;
        gen)                 SUB=generate; export GEN_PROMPT="${GEN_PROMPT:-1}" ;;
        gen_prefill)         SUB=generate; export GEN_PROMPT="${GEN_PROMPT:-8}" ;;
    esac
    [ "$STAGE" = "capture" ] && SUB=capture
    local FLAGS=""
    [ "$STAGE" = "eager" ] && [ "$SUB" = "decode" ] && FLAGS="--eager"
    OKPAT="top1|PASS"   # CLI predates the [mode] PASS marker; python is gate-authoritative
    env "$BIN" "$SUB" $FLAGS | tee "$LOGT" || true
}

case "$RUNNER" in
  python) run_python ;;
  cuda)   run_cuda ;;
  *) echo "unknown RUNNER $RUNNER"; exit 1 ;;
esac

if [ "$STAGE" = "capture" ]; then
    echo "[task] captured graphs: $(ls "$FHE_GRAPH_DIR"/block_*/graph.json "$FHE_GRAPH_DIR"/chunk_*/block_*/graph.json 2>/dev/null | wc -l)"
else
    grep -qE "$OKPAT" "$LOGT" || { echo "[task] FAIL: $TASK produced no PASS marker"; rm -f "$LOGT"; exit 1; }
fi
rm -f "$LOGT"
echo "=== Done === $(date)"
