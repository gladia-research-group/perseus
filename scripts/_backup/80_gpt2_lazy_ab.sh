#!/bin/bash
#SBATCH --job-name gpt2_lazy_ab
#SBATCH -A IscrC_eff-SAM2
#SBATCH --time 01:30:00
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=256G
#SBATCH --output=logs/core/%x_%j.out
#SBATCH --error=logs/core/%x_%j.err

# SAME-NODE A/B of FIDESLIB_LAZY_CPU_SHADOW on GPT-2 planned decode.
# (Cross-JOB timing on this cluster is not comparable — both arms in ONE job.)
#
# The ViT saw 2.4x from this flag. GPT-2 should benefit for the same reason and less of
# it: its decode is bootstrap-heavy (604 bts/tok) where the ViT's forward is mult-heavy,
# and a bootstrap's internal work dwarfs one output-ciphertext copy. But every EvalMult /
# EvalAdd / EvalRotate in the 12 blocks + lm_head still mints an output ciphertext, and
# each one was deep-copying the stale OpenFHE CPU shadow.
#
# ACCEPTANCE: (1) per-position top1 IDENTICAL between arms and to ref (this is a
# teacher-forced trajectory, so refs are fixed); (2) relevels=0 and no plan_level_error in
# both; (3) arm 1 decode s/tok < arm 0. A top1 change means the flag is NOT safe on the
# GPT-2 path and must go back to ViT-only — report it, do not tune around it.
set -e -o pipefail
REPO="$(pwd)"; DEPS="$REPO/deps"
module load cuda/12.6 gcc cmake nccl
module use /leonardo/prod/spack/06/modules/0.22.2_preprod_base
module load libarchive/3.7.1--gcc--12.2.0-sw6t2mm
export LD_LIBRARY_PATH="$DEPS/lib:$DEPS/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=16
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TMPDIR="$SCRATCH/tmp.${SLURM_JOB_ID:-login}"
mkdir -p "$TMPDIR" logs/core
PYTHON="${PYTHON:-$REPO/../he-aware-training/.venv/bin/python}"

export STEPS_T=128 LOGN=16 AUTO_BTS_LEVEL=24 BTS_ITERATIONS=1
export CKKS_COMPLEX=1 GPT2_PACKING="${GPT2_PACKING:-cachemir}" FIDESLIB_ROT_KEY_BAND=11
export GPT2_CACHE=1 GPT2_FOLD_LN1=1 GPT2_FOLD_LN2=1 GPT2_FOLD_LNF=0
export FHE_LMHEAD_CAP=22 MALLOC_ARENA_MAX=2
export CACHE_READ_LEVEL_K=17 CACHE_READ_LEVEL_V=17
export CONFIGS_PATH="${CONFIGS_PATH:-$REPO/configs/model/approximation/gpt2_base/configs.json}"
export WEIGHTS_PATH="${WEIGHTS_PATH:-$SCRATCH/.cache/perseus/models/openai-community/gpt2/classic/weights.bin.zip}"
unset FHE_DECODE_PLACEMENTS_DIR FHE_PROFILE FHE_GRAPH_DIR

PLAN_DIR="${PLAN_DIR:-$REPO/bootstrap_placements/planned_${TAG:-gpt2_base}}"
[ -d "$PLAN_DIR" ] || { echo "missing plan dir $PLAN_DIR"; exit 1; }

if [ "${BUILD:-0}" = "1" ]; then
    cmake --build build-py --parallel 16 --target _core
fi

echo "=== node: $(hostname) ==="
for ARM in ${ARMS:-0 1}; do
    echo ""
    echo "################ ARM: FIDESLIB_LAZY_CPU_SHADOW=$ARM ################"
    FIDESLIB_LAZY_CPU_SHADOW=$ARM MULTI_T="${MULTI_T:-4}" GPT2_INFERENCE_MODE=threaded \
        FHE_BOOTSTRAP_PLACEMENTS_DIR="$PLAN_DIR" \
        "$PYTHON" scripts/modes_baseline.py decode || echo "ARM $ARM FAILED"
done
echo "=== Done === $(date)"
