#!/bin/bash
#SBATCH --job-name primitive_timing
#SBATCH -A IscrC_TVU
#SBATCH --time 00:30:00
#SBATCH --qos=boost_qos_dbg
#SBATCH -p boost_usr_prod
#SBATCH --mem=128G
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=logs/core/%x_%j.out
#SBATCH --error=logs/core/%x_%j.err

# app:primitives — build ONLY tests/dev/test_primitive_timing.cu and run it.
# Separate from scripts/build.sh on purpose: that one wipes build/bin/test_* and rebuilds the
# whole suite (long, and it would disturb a shared build tree); this touches one target.
# Env: REPS (20), LEVEL_SWEEP (comma list; default = post-bootstrap down to nearly exhausted),
#      HID_DIM (768). Context params are NOT knobs here — the point is default_ckks_options().
set -e -o pipefail
REPO=/leonardo_work/IscrC_eff-SAM2/azirilli/he-aware-training/src/perseus
cd "$REPO"
DEPS="$REPO/deps"
BUILD_DIR="${BUILD_DIR:-$REPO/build}"

module load cuda/12.6 gcc cmake nccl
module use /leonardo/prod/spack/06/modules/0.22.2_preprod_base
module load libarchive/3.7.1--gcc--12.2.0-sw6t2mm
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export CUDAHOSTCXX=$(which g++) CXX=$(which g++) CC=$(which gcc)
export OMP_NUM_THREADS=16
export LD_LIBRARY_PATH="$DEPS/lib:$DEPS/lib64:${LD_LIBRARY_PATH:-}"

_nccl_lib=""; _nccl_inc=""
for d in "${NCCL_HOME:+$NCCL_HOME/lib}" "$CUDA_HOME/lib64" "$CUDA_HOME/targets/x86_64-linux/lib" /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib; do
    [ -n "$d" ] && [ -z "$_nccl_lib" ] && [ -f "$d/libnccl.so" ] && _nccl_lib="$d"
done
for d in "${NCCL_HOME:+$NCCL_HOME/include}" "$CUDA_HOME/include" "$CUDA_HOME/targets/x86_64-linux/include" /usr/include /usr/local/include; do
    [ -n "$d" ] && [ -z "$_nccl_inc" ] && [ -f "$d/nccl.h" ] && _nccl_inc="$d"
done
[ -n "$_nccl_lib" ] && [ -n "$_nccl_inc" ] || { echo "ERROR: NCCL not found"; exit 1; }
export NCCL_HOME="${NCCL_HOME:-$(dirname "$_nccl_inc")}"

_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1)
GPU_ARCH="80-real"; [[ "$_cap" =~ ^[0-9]+\.[0-9]+$ ]] && GPU_ARCH="$(echo "$_cap" | tr -d '.')-real"
echo "[primitive_timing] GPU_ARCH=$GPU_ARCH  build=$BUILD_DIR"

# configure (CONFIGURE_DEPENDS picks up the new tests/dev/test_*.cu), then ONE target
cmake -S "$REPO" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" \
    -DCMAKE_LIBRARY_PATH="$_nccl_lib" -DCMAKE_INCLUDE_PATH="$_nccl_inc" \
    -DCUDA_PATH="$CUDA_HOME" -DCMAKE_CUDA_ARCHITECTURES="$GPU_ARCH" \
    -DFIDESLIB_ROOT="$DEPS" \
    -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$HOME/src/googletest" \
    -DCACHEMIR_BUILD_TESTS=ON

cmake --build "$BUILD_DIR" --parallel 16 --target test_primitive_timing

echo ""
echo "=== running ==="
"$BUILD_DIR/bin/test_primitive_timing" --gtest_color=no
