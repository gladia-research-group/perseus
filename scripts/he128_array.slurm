#!/bin/bash
#SBATCH --job-name he128
#SBATCH -A EUHPC_D34_099
#SBATCH --qos normal
#SBATCH --time 10:00:00
#SBATCH -p boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=128G
#SBATCH --array=0-7
#SBATCH --output=logs/core/%x_%A_%a.out
#SBATCH --error=logs/core/%x_%A_%a.err
#
# HE128 fidelity arrays on the perseus CANON (Mode-A + scope-fix binary): paired
# teacher-forced chains x 128 positions, per-sample GT from $SCRATCH/he128* (library-
# invariant, gathered 07-15 era, seeds 1000+id). Adapts pycudafideslib.bkp/scripts/
# 63_decode_he128_heat_chunked.slurm to the canon env contract (run_task.sh decode
# preset). Session built ONCE per chunk (decode_multi), CHUNK_SIZE samples each.
#
#   ARM=heat|base|squeeze  selects config/weights/plan/GT-dir/log-dir  (required)
#   Scope: 32 chains = --array=0-7 (default, CHUNK_SIZE=4); 128 chains = --array=0-31.
#
# Full heat arm:  ARM=heat sbatch scripts/he128_array.slurm
# Full base arm:  ARM=base sbatch --time=12:00:00 scripts/he128_array.slurm
# Wave form (2026-08-14 Leonardo resume): first 8 chains = --array=0-7 CHUNK_SIZE=1
# (one sample per task, 8 in parallel); the remaining 120 = --array=0-29 CHUNK_SIZE=4
# HE128_START=8, which offsets into the master list instead of restarting at 0.
# Smoke (1 sample, 8 tok, dbg):
#   ARM=heat MULTI_T=8 CHUNK_SIZE=1 sbatch --qos=boost_qos_dbg --time=00:30:00 --array=0-0 scripts/he128_array.slurm
# Aggregate: python pycudafideslib.bkp/scripts/aggregate_he128.py --logdir logs/he128_<arm> --outdir logs/he128_<arm>/results
set -e -o pipefail
REPO="$(pwd)"; DEPS="$REPO/deps"
module load cuda/12.6 gcc cmake nccl
module use /leonardo/prod/spack/06/modules/0.22.2_preprod_base
module load libarchive/3.7.1--gcc--12.2.0-sw6t2mm
export LD_LIBRARY_PATH="$DEPS/lib:$DEPS/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=8

ARM=${ARM:?set ARM=heat|base}
CFG="$REPO/configs/model/approximation"
case "$ARM" in
  heat)
    export CONFIGS_PATH="${CONFIGS_PATH:-$CFG/gpt2_heat/configs.json}"
    export WEIGHTS_PATH="${WEIGHTS_PATH:-$SCRATCH/.cache/perseus/models/openai-community/gpt2/heat/weights.bin.zip}"
    export FHE_BOOTSTRAP_PLACEMENTS_DIR="${FHE_BOOTSTRAP_PLACEMENTS_DIR:-$REPO/bootstrap_placements/planned_gpt2_heat}"
    HE128_DIR="${HE128_DIR:-$SCRATCH/he128_heat}"
    ;;
  base)
    export CONFIGS_PATH="${CONFIGS_PATH:-$CFG/gpt2_base/configs.json}"
    export WEIGHTS_PATH="${WEIGHTS_PATH:-$SCRATCH/.cache/perseus/models/openai-community/gpt2/classic/weights.bin.zip}"
    export FHE_BOOTSTRAP_PLACEMENTS_DIR="${FHE_BOOTSTRAP_PLACEMENTS_DIR:-$REPO/bootstrap_placements/planned_gpt2_base}"
    HE128_DIR="${HE128_DIR:-$SCRATCH/he128}"
    ;;
  squeeze)
    # Third arm of the 1e-4 recount ladder. All four assets are tier-matched: the
    # squeezed export (scratch tag mirrors checkpoints/.../lm_eval/squeezed) and the
    # squeezed-checkpoint oracle set gathered HERE on the real train corpus
    # (cluster/gather_he128_squeezed.slurm, seeds 1000+id) — so unlike behemoth's
    # regenerated oracle, this one IS cross-comparable with the base arm.
    export CONFIGS_PATH="${CONFIGS_PATH:-$CFG/gpt2_squeeze/configs.json}"
    export WEIGHTS_PATH="${WEIGHTS_PATH:-$SCRATCH/.cache/perseus/models/openai-community/gpt2/squeezed/weights.bin.zip}"
    export FHE_BOOTSTRAP_PLACEMENTS_DIR="${FHE_BOOTSTRAP_PLACEMENTS_DIR:-$REPO/bootstrap_placements/planned_gpt2_squeeze}"
    HE128_DIR="${HE128_DIR:-$SCRATCH/he128_squeezed}"
    ;;
  *) echo "unknown ARM $ARM"; exit 2 ;;
esac

LOG_DIR="${HE128_LOG_DIR:-$REPO/logs/he128_$ARM}"; mkdir -p "$LOG_DIR"
MASTER="${HE128_MASTER:-$LOG_DIR/list_master.txt}"
if [ ! -s "$MASTER" ]; then
    for d in "$HE128_DIR"/sample_*; do
        id=$(basename "$d"); echo "${id#sample_} $d"
    done | sort > "$MASTER"
fi
CHUNK_SIZE=${CHUNK_SIZE:-4}
TID=${SLURM_ARRAY_TASK_ID:-0}
# HE128_START offsets into the master list so a second wave can continue where the
# first stopped (wave 1 = samples 0-7, wave 2 = HE128_START=8 covering 8-127).
START=$(( ${HE128_START:-0} + TID * CHUNK_SIZE ))
CHUNK_LIST="$LOG_DIR/list_chunk_${TID}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}.txt"
sed -n "$((START + 1)),$((START + CHUNK_SIZE))p" "$MASTER" > "$CHUNK_LIST"
export HE128_SAMPLE_LIST="$CHUNK_LIST"
export HE128_LOG_DIR="$LOG_DIR"

# ---- canon decode env (== run_task.sh decode preset, Mode-A) --------------------------
export MULTI_T=${MULTI_T:-128}
export STEPS_T=128                 # only the T128 per-sample oracle exists
export LOGN=16 AUTO_BTS_LEVEL=24 BTS_ITERATIONS=1
export CKKS_COMPLEX=1
export GPT2_PACKING=cachemir
export FIDESLIB_ROT_KEY_BAND=11
export MALLOC_ARENA_MAX=2
export GPT2_INFERENCE_MODE=threaded
export GPT2_CACHE=1
export GPT2_FOLD_LN1=1 GPT2_FOLD_LN2=1 GPT2_FOLD_LNF=0
export CACHE_READ_LEVEL_K=17 CACHE_READ_LEVEL_V=17
export FHE_LMHEAD_CAP=22
export CUTMAX_PRECISE_SCOPED=1 CUTMAX_VEC_BTS_ITERS=1

echo "=== HE128 $ARM chunk=$TID samples=[$START,$((START + CHUNK_SIZE - 1))] MULTI_T=$MULTI_T ==="
echo "list=$CHUNK_LIST plan=$FHE_BOOTSTRAP_PLACEMENTS_DIR config=$CONFIGS_PATH weights=$WEIGHTS_PATH"
cat "$CHUNK_LIST"
ls "$FHE_BOOTSTRAP_PLACEMENTS_DIR"/block_0_placement.json >/dev/null || { echo "MISSING PLAN"; exit 3; }
ls "$WEIGHTS_PATH" >/dev/null || { echo "MISSING WEIGHTS"; exit 3; }

"${BIN:-$REPO/build-py/bin/cuda_cachemir}" decode_multi 2>&1 || true
echo "=== chunk=$TID done === $(date)"
