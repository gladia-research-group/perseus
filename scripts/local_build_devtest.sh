#!/bin/bash
# local_build_devtest.sh — build one (or several) of the GPU test binaries under tests/dev.
#
#   bash scripts/local_build_devtest.sh [target ...]      # default: test_bts_profile
#
# Needs the chain's deps tree (scripts/install_deps.sh -> deps_<chain>/). Tests only: the
# python bindings are off, so this does not touch perseus/_core. Kernel edits inside FIDESlib
# need a deps rebuild first; this tree links the prebuilt package in deps_<chain>/.
# Env: CHAIN (n32 default), JOBS (default 16), CUDA_ARCH (default = the first visible GPU's
# compute capability, else 80-real).
set -e -o pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
source scripts/local_env.sh              # CHAIN, FHE_DEPS_DIR, FHE_BUILD_DIR, CUDA_HOME

[ -d "$FHE_DEPS_DIR" ] || { echo "missing deps tree $FHE_DEPS_DIR — run 'NATIVE_SIZE=${CHAIN#n} bash scripts/install_deps.sh'"; exit 1; }
BUILD_DIR="${FHE_BUILD_DIR}_tests"
TARGETS=("${@:-test_bts_profile}")
if [ -z "${CUDA_ARCH:-}" ]; then
    _cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')"
    CUDA_ARCH="${_cap:+${_cap}-real}"; CUDA_ARCH="${CUDA_ARCH:-80-real}"
fi

if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    echo "== configure $BUILD_DIR (chain=$CHAIN, deps=$FHE_DEPS_DIR, arch=$CUDA_ARCH)"
    cmake -S "$REPO" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
        -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
        -DFIDESLIB_ROOT="$FHE_DEPS_DIR" \
        -DCACHEMIR_BUILD_PYTHON=OFF -DCACHEMIR_BUILD_TESTS=ON \
        ${LibArchive_INCLUDE_DIR:+-DLibArchive_INCLUDE_DIR=$LibArchive_INCLUDE_DIR} \
        ${LibArchive_LIBRARY:+-DLibArchive_LIBRARY=$LibArchive_LIBRARY}
fi

cmake --build "$BUILD_DIR" --parallel "${JOBS:-16}" --target "${TARGETS[@]}"
ls -lh "$BUILD_DIR/bin/" | grep -E "$(IFS='|'; echo "${TARGETS[*]}")" || true
