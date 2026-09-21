#!/bin/bash
# build_core.sh — configure (first run) and build the _core / cuda_cachemir targets.
#
# Env: DEPS (default ./deps, where install_deps.sh puts OpenFHE + FIDESlib), BUILD_DIR
# (default ./build-py), PYTHON (default ./.venv/bin/python), CUDA_HOME, GPU_ARCH
# (default: the first visible GPU's compute capability, else 80-real), JOBS (default 8),
# LibArchive_INCLUDE_DIR / LibArchive_LIBRARY when libarchive is not installed system-wide.
set -e
cd "$(dirname "$0")/.."

# Module preamble: loads the toolchain on an HPC system, a no-op anywhere else.
module load cuda/12.6 gcc cmake nccl 2>/dev/null || true
module use /leonardo/prod/spack/06/modules/0.22.2_preprod_base 2>/dev/null || true
module load libarchive/3.7.1--gcc--12.2.0-sw6t2mm 2>/dev/null || true

REPO="$(pwd)"
DEPS="${DEPS:-$REPO/deps}"
BUILD_DIR="${BUILD_DIR:-$REPO/build-py}"
export TMPDIR="${TMPDIR:-${SCRATCH:-/tmp}/tmp.build}"
mkdir -p "$TMPDIR"

if [ -z "${CUDA_HOME:-}" ]; then
    if [ -d /usr/local/cuda ]; then
        CUDA_HOME=/usr/local/cuda
    elif command -v nvcc >/dev/null 2>&1; then
        CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
    else
        CUDA_HOME=/usr/local/cuda-12.6
    fi
    export CUDA_HOME
fi

PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"
"$PYTHON" -c 'import pybind11' 2>/dev/null || {
    echo "pybind11 not found in $PYTHON — run 'uv sync' (or pip install pybind11) first" >&2
    exit 1
}

if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    _cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '. ')"
    GPU_ARCH="${GPU_ARCH:-${_cap:-80}-real}"
    echo "== configure $BUILD_DIR (deps=$DEPS, cuda=$CUDA_HOME, arch=$GPU_ARCH)"
    # a failed configure still writes CMakeCache.txt, which would make the next run skip
    # configure and fail on a missing Makefile — drop the directory instead
    cmake -S "$REPO" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="$GPU_ARCH" \
        -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
        -DCUDA_PATH="$CUDA_HOME" \
        -DFIDESLIB_ROOT="$DEPS" \
        -DCACHEMIR_BUILD_PYTHON=ON -DCACHEMIR_BUILD_TESTS=OFF \
        -Dpybind11_DIR="$("$PYTHON" -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
        -DPYBIND11_FINDPYTHON=NEW \
        -DPython_EXECUTABLE="$PYTHON" \
        -DPython_INCLUDE_DIR="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')" \
        ${LibArchive_INCLUDE_DIR:+-DLibArchive_INCLUDE_DIR=$LibArchive_INCLUDE_DIR} \
        ${LibArchive_LIBRARY:+-DLibArchive_LIBRARY=$LibArchive_LIBRARY} \
        || { rm -rf "$BUILD_DIR"; exit 1; }
fi

# rename-aside so running jobs keep their mapped inode; ld writes a fresh file in place
SO=perseus/_core.cpython-311-x86_64-linux-gnu.so
[ -f "$SO" ] && mv "$SO" "$SO.stale.$$"
cmake --build "$BUILD_DIR" --parallel "${JOBS:-8}" --target _core cuda_cachemir
rm -f "$SO.stale.$$"
