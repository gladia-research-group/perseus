#!/bin/bash
# local_env.sh — environment for building and running perseus on one machine.
#   source scripts/local_env.sh          # CHAIN=n32 (default) or CHAIN=n64
# Every value is a default: anything already exported by the caller wins.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- chain: n32 (27 composite levels over 54 primes, the paper's chain) or n64 (28 levels of 53-bit)
export CHAIN="${CHAIN:-n32}"
export FHE_DEPS_DIR="${FHE_DEPS_DIR:-$REPO/deps_$CHAIN}"     # FIDESlib + patched OpenFHE (scripts/install_deps.sh)
export FHE_BUILD_DIR="${FHE_BUILD_DIR:-$REPO/build_py_$CHAIN}"

# ---- toolchain
if [ -z "${CUDA_HOME:-}" ]; then
    # prefer a toolkit under /usr/local (newest), then the nvcc on PATH
    for c in /usr/local/cuda $(ls -d /usr/local/cuda-* 2>/dev/null | sort -V -r); do
        [ -x "$c/bin/nvcc" ] && { CUDA_HOME="$c"; break; }; done
    if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
        CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"; fi
    CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$FHE_DEPS_DIR/lib:$FHE_DEPS_DIR/lib64:${NCCL_HOME:+$NCCL_HOME/lib:}${LD_LIBRARY_PATH:-}"
export PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TMPDIR="${TMPDIR:-$REPO/.cache/tmp}"; mkdir -p "$TMPDIR"

# ---- host resources: one FHE process at a time; 16 OpenMP threads is the measured default
_ncpu="$(nproc 2>/dev/null || echo 16)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( _ncpu < 16 ? _ncpu : 16 ))}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

# ---- data: model artifacts and the decode oracle (README: "Artifacts")
export PERSEUS_DATA="${PERSEUS_DATA:-$REPO/.cache}"
export WEIGHTS_PATH="${WEIGHTS_PATH:-$PERSEUS_DATA/models/openai-community/gpt2/classic/weights.bin.zip}"
export ALL_BLOCKS_IO_DIR="${ALL_BLOCKS_IO_DIR:-$PERSEUS_DATA/oracle/gpt2/all_blocks_io}"

# ---- CKKS parameters
export LOGN="${LOGN:-16}"
export BTS_ITERATIONS="${BTS_ITERATIONS:-1}"
if [ "$CHAIN" = "n32" ]; then
    # 32-bit composite chain: pairs of 27-bit primes per level, q0 a pair of 28-bit primes
    export COMPOSITE_DEGREE="${COMPOSITE_DEGREE:-2}"
    export CKKS_DEPTH="${CKKS_DEPTH:-10}"
    export BTP_DEPTH_OVERHEAD="${BTP_DEPTH_OVERHEAD:-16}"
    export SCALE_BITS="${SCALE_BITS:-54}"
    export BTP_SCALE_BITS="${BTP_SCALE_BITS:-54}"
    export FIRST_MOD_BITS="${FIRST_MOD_BITS:-56}"
    export NUM_LARGE_DIGITS="${NUM_LARGE_DIGITS:-6}"
    export LEVEL_BUDGET="${LEVEL_BUDGET:-4:3}"
    export CORRECTION_FACTOR="${CORRECTION_FACTOR:-6}"
    export AUTO_BTS_LEVEL="${AUTO_BTS_LEVEL:-46}"
    export FIDESLIB_ROT_KEY_BAND="${FIDESLIB_ROT_KEY_BAND:-22}"
    export FHE_STAGE_ARENA_GB="${FHE_STAGE_ARENA_GB:-24}"
    export KV_ARENA_GB="${KV_ARENA_GB:-24}"
    export FHE_LMHEAD_CAP="${FHE_LMHEAD_CAP:-44}"
    export CACHE_READ_LEVEL_K="${CACHE_READ_LEVEL_K:-34}"
    export CACHE_READ_LEVEL_V="${CACHE_READ_LEVEL_V:-34}"
    export SPARSE_BTS_SLOTS="${SPARSE_BTS_SLOTS:-512,1}"    # sparse bootstrap routes the plans use
    export SPARSE_AUTO="${SPARSE_AUTO:-2}"                  # route periodic payloads to them automatically
    export FUSED_SM_DEN="${FUSED_SM_DEN:-1}"                # softmax denominator refreshed inside the fold
    export FUSED_LN_VAR="${FUSED_LN_VAR:-0}"
    export PLAN_SPARSE_SLOTS="${PLAN_SPARSE_SLOTS:-$SPARSE_BTS_SLOTS}"
else
    export AUTO_BTS_LEVEL="${AUTO_BTS_LEVEL:-24}"
    export FIDESLIB_ROT_KEY_BAND="${FIDESLIB_ROT_KEY_BAND:-11}"
    export FHE_LMHEAD_CAP="${FHE_LMHEAD_CAP:-22}"
    export CACHE_READ_LEVEL_K="${CACHE_READ_LEVEL_K:-17}"
    export CACHE_READ_LEVEL_V="${CACHE_READ_LEVEL_V:-17}"
    export SPARSE_BTS_SLOTS="${SPARSE_BTS_SLOTS:-512,1}"    # sparse bootstrap routes the plans use
    export SPARSE_AUTO="${SPARSE_AUTO:-2}"                  # route periodic payloads to them automatically
    # A sparse route needs a non-zero correction factor on this chain. The chain default is 0
    # ("OpenFHE auto"), which a dense refresh tolerates and a sparse one does not: at 0 the run
    # diverges silently, with no throw and no change to the op sequence.
    export CORRECTION_FACTOR="${CORRECTION_FACTOR:-7}"
    export PLAN_SPARSE_SLOTS="${PLAN_SPARSE_SLOTS:-$SPARSE_BTS_SLOTS}"
fi
# coefficient staging needs |w| < 2, which GPT-2's LN-folded weights exceed: both chains off
export FHE_PT_COEFF_ENCODE="${FHE_PT_COEFF_ENCODE:-0}"
# key-switching keys: seed-regenerated in-kernel + bit-packed (paper Table 2; =0 restores stored / unpacked keys)
export FIDESLIB_KSK_REGEN="${FIDESLIB_KSK_REGEN:-2}"
export FIDESLIB_KSK_PACK="${FIDESLIB_KSK_PACK:-1}"
