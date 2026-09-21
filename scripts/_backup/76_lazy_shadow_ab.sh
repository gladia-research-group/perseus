#!/bin/bash
#SBATCH --job-name lazyshadow_ab
#SBATCH -A IscrC_eff-SAM2
#SBATCH --time 00:30:00
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=256G
#SBATCH --output=logs/core/%x_%j.out
#SBATCH --error=logs/core/%x_%j.err

# FIDESLIB_LAZY_CPU_SHADOW gate — rebuild FIDESlib, then A/B the flag BACK-TO-BACK on
# ONE node (cross-job timing on this cluster is not comparable).
#
# The claim under test: ~100% of a fresh-output GPU op's wall is the copy ctor
# deep-copying the source ciphertext's OpenFHE CPU-side DCRTPoly limbs, which the device
# kernel immediately makes stale and which nothing reads before Decrypt overwrites it.
# With the flag armed, CryptoContextImpl::MakeGpuResultLike hands GPU-path results a
# metadata-only (CloneEmpty) shadow instead.
#
# ACCEPTANCE:
#   1. [lazyshadow] checksum BIT-IDENTICAL between the OFF and ON arms  (correctness)
#   2. [ptmult] A / G collapse in the ON arm                            (the win)
#   3. OFF arm identical to the pre-change baseline                     (neutrality)
set -e -o pipefail
REPO="$(pwd)"; DEPS="$REPO/deps"; FB="$REPO/third_party/FIDESlib/build"
module load cuda/12.6 gcc cmake nccl
module use /leonardo/prod/spack/06/modules/0.22.2_preprod_base
module load libarchive/3.7.1--gcc--12.2.0-sw6t2mm
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export LD_LIBRARY_PATH="$DEPS/lib:$DEPS/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=16
export TMPDIR="$SCRATCH/tmp.${SLURM_JOB_ID:-login}"
mkdir -p "$TMPDIR" logs/core
export LOGN=16 AUTO_BTS_LEVEL=24 BTS_ITERATIONS=1

if [ "${REBUILD_FIDESLIB:-1}" = "1" ]; then
    cp -f "$DEPS/lib64/fideslib.a" "$DEPS/lib64/fideslib.a.bak-prelazy"
    # Touched TUs: api/CryptoContext.cpp (MakeGpuResultLike + the two shadow guards),
    # api/Ciphertext.cpp (the lazy copy ctor). Force both out; headers pull the rest.
    find "$FB" -name 'CryptoContext.cpp.o' -delete
    find "$FB" -name 'Ciphertext.cpp.o' -delete
    cmake --build "$FB" --parallel 16
    cmake --install "$FB"
    echo "=== fideslib rebuilt: $(ls -la $DEPS/lib64/fideslib.a) ==="
fi

rm -f build/bin/test_ptmult_isolation
cmake --build build --parallel 16 --target test_ptmult_isolation

for ARM in 0 1; do
    echo ""
    echo "################ FIDESLIB_LAZY_CPU_SHADOW=$ARM ################"
    FIDESLIB_LAZY_CPU_SHADOW=$ARM ITERS="${ITERS:-100}" LEVEL="${LEVEL:-19}" \
        CHUNK="${CHUNK:-64}" build/bin/test_ptmult_isolation \
        --gtest_filter="${FILTER:-PtMultIsolation.*}" 2>&1 \
        | grep -E "ptmult|ptcorrect|lazyshadow|OK \]|FAILED"
done
echo "=== Done === $(date)"
