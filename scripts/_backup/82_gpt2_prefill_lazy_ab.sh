#!/bin/bash
#SBATCH --job-name gpt2_prefill_lazy_ab
#SBATCH -A IscrC_eff-SAM2
#SBATCH --time 02:00:00
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=256G
#SBATCH --output=logs/core/%x_%j.out
#SBATCH --error=logs/core/%x_%j.err

# Python session baseline across all inference modes; one process per mode.
set -e -o pipefail
REPO="$(pwd)"; DEPS="$REPO/deps"
module load cuda/12.6 gcc cmake nccl
module use /leonardo/prod/spack/06/modules/0.22.2_preprod_base
module load libarchive/3.7.1--gcc--12.2.0-sw6t2mm
export LD_LIBRARY_PATH="$DEPS/lib:$DEPS/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=16
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
PYTHON="${PYTHON:-$REPO/../he-aware-training/.venv/bin/python}"

export MULTI_T=4 STEPS_T=128 LOGN=16
export AUTO_BTS_LEVEL=24 BTS_ITERATIONS=1
export GPT2_INFERENCE_MODE=threaded GPT2_CACHE=1
export GPT2_FOLD_LN1=1 GPT2_FOLD_LN2=1 GPT2_FOLD_LNF=0
export FHE_LMHEAD_CAP=22 FIDESLIB_ROT_KEY_BAND=11
export CONFIGS_PATH="${CONFIGS_PATH:-$REPO/configs/model/approximation/gpt2_base/configs.json}"
export WEIGHTS_PATH="${WEIGHTS_PATH:-$SCRATCH/.cache/perseus/models/openai-community/gpt2/classic/weights.bin.zip}"
unset FHE_BOOTSTRAP_PLACEMENTS_DIR FHE_DECODE_PLACEMENTS_DIR FHE_PROFILE FHE_GRAPH_DIR

# decode = the proven Mode-A arm; prefill/handoff = the frozen real-mode arm with
# the canonical 04_decode_handoff memory envelope (band 11, streamed lm_head tiles).
export MALLOC_ARENA_MAX=2
# SAME-NODE A/B of FIDESLIB_LAZY_CPU_SHADOW on GPT-2 PREFILL.
#
# WHY PREFILL AND NOT DECODE. Decode is bootstrap-bound (~604 bts/tok = 82-94% of its wall)
# and got only -15% from the lazy shadow. PREFILL is structurally the ViT: cachemir_filling
# packing, T tokens in one ciphertext, and linear.cu:15 routes it to the SAME
# diagonal::linear BSGS as the ViT — ~10240 plaintext mults per block against ~129 bts/block
# (chunk_0 = 1678 bts / 13 blocks). That is ~80% linear, i.e. the ViT profile, where the
# lazy shadow was worth 2.4x. So prefill should gain far more than decode did.
#
# ACCEPTANCE: identical top1 in both arms (the flag is value-neutral — proven on ViT and on
# GPT-2 decode); arm 1 s/tok materially lower. unplanned_bts/weight_relevels equal.
export MODES="${MODES:-prefill}"
for ARM in ${ARMS:-0 1}; do
    echo ""
    echo "################ ARM: FIDESLIB_LAZY_CPU_SHADOW=$ARM ################"
    for m in $MODES; do
        FIDESLIB_LAZY_CPU_SHADOW=$ARM CKKS_COMPLEX=0 GPT2_PACKING=cachemir \
            FIDESLIB_ROT_KEY_BAND=11 GPT2_LMHEAD_GRANULARITY=linear \
            "$PYTHON" scripts/modes_baseline.py "$m" || echo "ARM $ARM $m FAILED"
    done
done
echo "=== Done === $(date)"
