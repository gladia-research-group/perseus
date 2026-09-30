#!/bin/bash
# run_task.sh — run, gate, or capture one GPT-2 task with the paper's configuration.
#
#   TASK=decode [STAGE=run|eager|capture] [RUNNER=python|cuda] [CHAIN=n32|n64] bash scripts/run_task.sh
#
#   TASK      decode       oracle-fed decode, MULTI_T tokens (the paper's row; planned by default)
#             gen          prompt + GEN_TOKENS generated tokens, encrypted argmax feedback (eager)
#             gen_prefill  prefill of GEN_PROMPT tokens then generation (eager)
#             handoff      prefill -> decode hand-off correctness check (eager)
#   STAGE     run          planned run with the shipped plan (FHE_BOOTSTRAP_PLACEMENTS_DIR overrides)
#             eager        no plan: reactive bootstraps only
#             capture      record the op-graph of one sync forward into FHE_GRAPH_DIR (for the planner)
#   RUNNER    python       perseus._core through scripts/modes_baseline.py (the correctness gate)
#             cuda         the native CLI build_py_<chain>/bin/cuda_cachemir (the wall-clock vehicle)
#
# Every preset below is a default (`${VAR:-value}`): exported env always wins, so an ablation is
# an env override, e.g. FIDESLIB_KSK_REGEN=0 or SPARSE_BTS_SLOTS=0. A run passes only if the
# driver prints its PASS marker; the exit code alone is not evidence.
set -e -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
source scripts/local_env.sh
mkdir -p logs/core

TASK="${TASK:?set TASK=decode|gen|gen_prefill|handoff}"
STAGE="${STAGE:-run}"
RUNNER="${RUNNER:-python}"
BP="$REPO/bootstrap_placements"; CFG="$REPO/configs/model/approximation"

# ---- GPT-2 (124M) on the cachemir packing, complex lanes on, LN1/LN2 folded into the weights
export CKKS_COMPLEX="${CKKS_COMPLEX:-1}"
export GPT2_PACKING="${GPT2_PACKING:-cachemir}"
export GPT2_CACHE="${GPT2_CACHE:-1}"
export STEPS_T="${STEPS_T:-128}"
export GPT2_FOLD_LN1="${GPT2_FOLD_LN1:-1}"
export GPT2_FOLD_LN2="${GPT2_FOLD_LN2:-1}"
export GPT2_FOLD_LNF="${GPT2_FOLD_LNF:-0}"
export CUTMAX_PRECISE_SCOPED="${CUTMAX_PRECISE_SCOPED:-1}"
export CUTMAX_VEC_BTS_ITERS="${CUTMAX_VEC_BTS_ITERS:-1}"
# The C++ decode ships no capture or plan: STAGE=capture writes GRAPH_DEFAULT, and
# `perseus-plan --graph-dir <graph> --out-dir <plan>` writes the plan a planned run reads.
if [ "$CHAIN" = "n32" ]; then
    export CONFIGS_PATH="${CONFIGS_PATH:-$CFG/gpt2_base_n32/configs.json}"
    PLAN_DEFAULT="$BP/gpt2_decode_n32"; GRAPH_DEFAULT="$REPO/graphs/gpt2_decode_n32"
else
    export CONFIGS_PATH="${CONFIGS_PATH:-$CFG/gpt2_base/configs.json}"
    PLAN_DEFAULT="$BP/gpt2_decode_n64"; GRAPH_DEFAULT="$REPO/graphs/gpt2_decode_n64"
fi

case "$TASK" in
  decode)
    MODE=decode
    export MULTI_T="${MULTI_T:-16}"
    # the decode scheduling of the paper's row (block ring, mask prefetch, dropped-tower
    # decrypt) is the code's behaviour, not a set of knobs
    ;;
  gen)
    MODE=gen; PLAN_DEFAULT=""
    export GEN_TOKENS="${GEN_TOKENS:-8}"
    ;;
  gen_prefill|handoff)
    MODE="$TASK"; PLAN_DEFAULT=""
    export GEN_TOKENS="${GEN_TOKENS:-8}"
    export FHE_DELTA_BLOCK="${FHE_DELTA_BLOCK:-1}"
    ;;
  *) echo "unknown TASK $TASK"; exit 1 ;;
esac

case "$STAGE" in
  capture)
    export GPT2_INFERENCE_MODE=sync FIDESLIB_LAZY_CPU_SHADOW=0
    export FHE_ASYNC_MAG="${FHE_ASYNC_MAG:-1}"           # magnitudes measured on a worker, off the critical path
    export FHE_MAG_WORKERS="${FHE_MAG_WORKERS:-8}"
    export OPENFHE_DECODE_NO_THROW="${OPENFHE_DECODE_NO_THROW:-1}"
    unset FHE_BOOTSTRAP_PLACEMENTS_DIR
    export FHE_GRAPH_DIR="${FHE_GRAPH_DIR:-$GRAPH_DEFAULT}"
    [ "$MODE" = gen ] && export GEN_TOKENS=2
    if [ -d "$FHE_GRAPH_DIR" ]; then                        # never overwrite a capture: archive it
        mkdir -p "$REPO/.cache/_graph_archive"
        mv "$FHE_GRAPH_DIR" "$REPO/.cache/_graph_archive/$(basename "$FHE_GRAPH_DIR")_$(date +%Y%m%d_%H%M%S)"
    fi
    mkdir -p "$FHE_GRAPH_DIR"
    "$PYTHON" scripts/utils/lint_approx_config.py "$CONFIGS_PATH" || { echo "[task] config lint failed"; exit 1; }
    echo "[task] CAPTURE $TASK -> $FHE_GRAPH_DIR"
    ;;
  run)
    export GPT2_INFERENCE_MODE="${GPT2_INFERENCE_MODE:-threaded}"
    unset FHE_GRAPH_DIR
    if [ -n "${FHE_BOOTSTRAP_PLACEMENTS_DIR:-}" ] || [ -n "$PLAN_DEFAULT" ]; then
        export FHE_BOOTSTRAP_PLACEMENTS_DIR="${FHE_BOOTSTRAP_PLACEMENTS_DIR:-$PLAN_DEFAULT}"
        [ -d "$FHE_BOOTSTRAP_PLACEMENTS_DIR" ] || { echo "missing plan dir $FHE_BOOTSTRAP_PLACEMENTS_DIR"; exit 1; }
        echo "[task] PLANNED RUN $TASK <- $FHE_BOOTSTRAP_PLACEMENTS_DIR"
    else
        unset FHE_BOOTSTRAP_PLACEMENTS_DIR
        echo "[task] RUN $TASK (eager)"
    fi
    ;;
  eager)
    export GPT2_INFERENCE_MODE="${GPT2_INFERENCE_MODE:-threaded}"
    unset FHE_GRAPH_DIR FHE_BOOTSTRAP_PLACEMENTS_DIR
    echo "[task] EAGER RUN $TASK"
    ;;
  *) echo "unknown STAGE $STAGE"; exit 1 ;;
esac

for f in "$WEIGHTS_PATH" "$CONFIGS_PATH"; do [ -e "$f" ] || { echo "[task] missing $f (README: Artifacts)"; exit 1; }; done
[ -d "$ALL_BLOCKS_IO_DIR" ] || { echo "[task] missing oracle dir $ALL_BLOCKS_IO_DIR (README: Artifacts)"; exit 1; }

LOGT="logs/core/.task_${TASK}_$$.out"
case "$RUNNER" in
  python)
    # perseus._core is a symlink to the active chain's build: with both chains built it is
    # whichever was linked last, and CHAIN alone would not move it. The two are numerically
    # different libraries, so refuse rather than run the wrong one.
    _loaded="$("$PYTHON" -c 'import perseus._core as c; print(c.chain)' 2>/dev/null || echo unknown)"
    if [ "$_loaded" != "$CHAIN" ]; then
        _so="_core$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
        echo "[task] perseus._core is the '$_loaded' build but CHAIN=$CHAIN. Relink it:" >&2
        echo "         ln -sfn _core.$CHAIN.so perseus/$_so" >&2
        echo "       (or rebuild: CHAIN=$CHAIN bash scripts/local_build_core.sh)" >&2
        exit 1
    fi
    OKPAT="^\[$MODE\] PASS$"
    "$PYTHON" scripts/modes_baseline.py "$MODE" | tee "$LOGT" || true
    ;;
  cuda)
    BIN="${CUDA_CLI:-$FHE_BUILD_DIR/bin/cuda_cachemir}"
    [ -x "$BIN" ] || { echo "[task] cuda_cachemir not built: cmake --build $FHE_BUILD_DIR --target cuda_cachemir"; exit 1; }
    case "$MODE" in
        decode)      SUB=decode ;;
        handoff)     SUB=prefill; export PREFILL_TOKENS="${PREFILL_T:-8}" DECODE_TOKENS="${DECODE_TOKENS:-2}" ;;
        gen)         SUB=generate; export GEN_PROMPT="${GEN_PROMPT:-1}" ;;
        gen_prefill) SUB=generate; export GEN_PROMPT="${GEN_PROMPT:-8}" ;;
    esac
    [ "$STAGE" = capture ] && SUB=capture
    FLAGS=""; [ "$STAGE" = eager ] && [ "$SUB" = decode ] && FLAGS="--eager"
    OKPAT="SUMMARY cmd=.* completed=([0-9]+)/\1 "
    "$BIN" "$SUB" $FLAGS | tee "$LOGT" || true
    ;;
  *) echo "unknown RUNNER $RUNNER"; exit 1 ;;
esac

if [ "$STAGE" = capture ]; then
    echo "[task] captured graphs: $(ls "$FHE_GRAPH_DIR"/block_*/graph.json 2>/dev/null | wc -l)"
else
    grep -qE "$OKPAT" "$LOGT" || { echo "[task] FAIL: $TASK produced no PASS marker"; rm -f "$LOGT"; exit 1; }
fi
rm -f "$LOGT"
echo "[task] done $(date)"
