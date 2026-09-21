#!/bin/bash
#SBATCH --job-name encvit_lazy_ab
#SBATCH -A IscrC_eff-SAM2
#SBATCH --time 01:00:00
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=256G
#SBATCH --output=logs/core/%x_%j.out
#SBATCH --error=logs/core/%x_%j.err

# SAME-NODE A/B of FIDESLIB_LAZY_CPU_SHADOW on the planned k=12 ViT forward.
# (Cross-JOB timing on this cluster is not comparable — see scripts/73_encvit_ab.sh.)
#
# OFF arm = the current 800.5 s baseline. ON arm gives every GPU-path result
# ciphertext a metadata-only OpenFHE shadow instead of a deep copy of up to ~29 MB of
# DCRTPoly limbs — the copy that arms A/G of test_ptmult_isolation showed to be ~100%
# of a fresh-output op's wall.
#
# GRAPH-NEUTRAL: no call site changes, no op count changes, so the SAME plan
# (planned_vit_base_80) binds in both arms. unplanned_bts must stay 0 in both.
#
# ACCEPTANCE: (1) w_mape and top5 identical between arms (host bookkeeping only);
#             (2) unplanned_bts=0 in both; (3) ON arm's [vit-time] blocks much faster.
set -e -o pipefail
REPO="$(pwd)"; DEPS="$REPO/deps"
module load cuda/12.6 gcc cmake nccl
module use /leonardo/prod/spack/06/modules/0.22.2_preprod_base
module load libarchive/3.7.1--gcc--12.2.0-sw6t2mm
export LD_LIBRARY_PATH="$DEPS/lib:$DEPS/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=16
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TMPDIR="$SCRATCH/tmp.${SLURM_JOB_ID:-login}"
mkdir -p "$TMPDIR"
PYTHON="${PYTHON:-$REPO/../he-aware-training/.venv/bin/python}"

export LOGN=16 AUTO_BTS_LEVEL=24 BTS_ITERATIONS=1
export CONFIGS_PATH="${CONFIGS_PATH:-$REPO/configs/model/approximation/vit_base_80/configs.json}"
export VIT_MODEL_DIR="${VIT_MODEL_DIR:-$SCRATCH/.cache/perseus/models/google/vit-base-patch16-224/classic}"
export VIT_POOL="${VIT_POOL:-$SCRATCH/.cache/huggingface/perseus/pools/tiny_imagenet_vit-base-patch16-224_512.npy}"
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_CACHE="$SCRATCH/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1
export FHE_KV_OVERLAP=0
export FIDESLIB_ROT_KEY_BAND="${FIDESLIB_ROT_KEY_BAND:-11}"
# glibc: the weight encode/load path BETWEEN blocks is allocator-bound (it was ~42% of
# the run). MALLOC_TOP_PAD_ stops glibc trimming the heap back to the kernel, so the
# ~49 GB/block of fresh OpenFHE plaintexts stop re-faulting their pages in.
# MEASURED isolated (job 49904313): outside-bodies 196.4 -> 115.7 s, e2e -12.6%,
# quality unchanged. TRIM_THRESHOLD adds nothing (0.3%). NEVER set MALLOC_ARENA_MAX
# here — it serialises malloc against OMP_NUM_THREADS + the residency worker (+42%).
export MALLOC_TOP_PAD_="${MALLOC_TOP_PAD_:-134217728}"
export VIT_MODE="${VIT_MODE:-threaded}"
export GATE_BLOCKS="${GATE_BLOCKS:-12}" GATE_RES="${GATE_RES:-80}"
unset GPT2_FOLD_LN1 GPT2_FOLD_LN2 GPT2_FOLD_LNF CKKS_COMPLEX FHE_PROFILE FHE_GRAPH_DIR

export FHE_BOOTSTRAP_PLACEMENTS_DIR="${PLAN_DIR:-$REPO/bootstrap_placements/planned_vit_base_80}"

if [ "${BUILD:-1}" = "1" ]; then
    cmake --build build-py --parallel 16 --target _core
fi
export PYTHONFAULTHANDLER=1

echo "=== node: $(hostname) ==="
for ARM in ${ARMS:-0 1}; do
    echo ""
    echo "################ ARM: FIDESLIB_LAZY_CPU_SHADOW=$ARM ################"
    FIDESLIB_LAZY_CPU_SHADOW=$ARM "$PYTHON" scripts/encvit_forward.py \
        || echo "ARM $ARM FAILED"
done
echo "=== Done === $(date)"
