#!/bin/bash
set -e
echo "=== Installing OpenFHE + FIDESlib ==="
echo "Host: $(hostname)  |  Date: $(date)"

# ── Proxy ──────────────────────────────────────────────────────────────────

export OMP_NUM_THREADS=16

# ── Modules (best-effort; no-op outside Leonardo) ─────────────────────────
module load cuda/12.6 gcc cmake nccl 2>/dev/null || true

# ── Paths ─────────────────────────────────────────────────────────────────
REPO="$(pwd)"
DEPS="$REPO/deps"
FIDESLIB_SRC="$REPO/third_party/FIDESlib"
FIDESLIB_BUILD="$FIDESLIB_SRC/build"
OPENFHE_PATCH="$FIDESLIB_SRC/deps/openfhe-1.4.2.patch"
OPENFHE_TMP="${OPENFHE_TMP:-/tmp/openfhe_build_$$}"

mkdir -p "$DEPS"

# ── Modules ───────────────────────────────────────────────────────────────
# first check if module is available as a command
# if cuda_home is not set, try setting it like so
if [ -z "$CUDA_HOME" ] &>/dev/null; then
    if [ -d /usr/local/cuda ]; then
        export CUDA_HOME="/usr/local/cuda"
    elif command -v nvcc >/dev/null 2>&1; then
        export CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
    else
        export CUDA_HOME="/usr/local/cuda-12.6"
    fi
fi
export CUDA_NVCC="$CUDA_HOME/bin/nvcc"
export CUDAHOSTCXX=$(which g++)
export CXX=$(which g++)
export CC=$(which gcc)
echo "CUDA_HOME:   $CUDA_HOME"
echo "CUDA_NVCC:   $CUDA_NVCC"
echo "CUDAHOSTCXX: $CUDAHOSTCXX"
cmake --version | head -1
g++ --version | head -1

# ── NCCL preflight ─────────────────────────────────────────────────────────
_nccl_lib=""
_nccl_inc=""

if [ -n "$NCCL_HOME" ]; then
    if [ -f "$NCCL_HOME/lib/libnccl.so" ]; then
        _nccl_lib="$NCCL_HOME/lib"
    fi
    if [ -f "$NCCL_HOME/include/nccl.h" ]; then
        _nccl_inc="$NCCL_HOME/include"
    fi
fi

for d in \
    "$CUDA_HOME/lib64" \
    "$CUDA_HOME/targets/x86_64-linux/lib" \
    /usr/lib/x86_64-linux-gnu \
    /usr/lib \
    /usr/local/lib
do
    if [ -z "$_nccl_lib" ] && [ -f "$d/libnccl.so" ]; then
        _nccl_lib="$d"
    fi
done

for d in \
    "$CUDA_HOME/include" \
    "$CUDA_HOME/targets/x86_64-linux/include" \
    /usr/include \
    /usr/local/include
do
    if [ -z "$_nccl_inc" ] && [ -f "$d/nccl.h" ]; then
        _nccl_inc="$d"
    fi
done

if [ -z "$_nccl_lib" ] || [ -z "$_nccl_inc" ]; then
    echo ""
    echo "ERROR: NCCL not found (required for this FIDESlib build)."
    echo "Missing lib path:    ${_nccl_lib:-<not found>}"
    echo "Missing include path:${_nccl_inc:-<not found>}"
    echo ""
    echo "Install NCCL runtime and headers (e.g. libnccl2 + libnccl-dev),"
    echo "or export NCCL_HOME to a valid prefix containing lib/libnccl.so"
    echo "and include/nccl.h before running this script."
    exit 1
fi

export NCCL_HOME="$(dirname "$_nccl_inc")"
echo "NCCL_HOME:   $NCCL_HOME"
echo "NCCL lib:    $_nccl_lib"
echo "NCCL include:$_nccl_inc"

# ── GPU architecture ───────────────────────────────────────────────────────
_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1)
if [[ "$_cap" =~ ^[0-9]+\.[0-9]+$ ]]; then
    GPU_ARCH="$(echo "$_cap" | tr -d '.')-real"
    echo "Auto-detected GPU arch: $GPU_ARCH (compute cap ${_cap})"
else
    GPU_ARCH="80-real"
    echo "nvidia-smi unavailable, defaulting to $GPU_ARCH"
fi

# ── Init FIDESlib submodule ────────────────────────────────────────────────
echo ""
echo "--- Initialising FIDESlib submodule ---"
cd "$REPO"
GIT_SSL_NO_VERIFY=true git submodule update --init third_party/FIDESlib 2>&1

# ══════════════════════════════════════════════════════════════════════════
# STAGE 1: OpenFHE v1.4.2
# ══════════════════════════════════════════════════════════════════════════
OPENFHE_VER_FILE="$DEPS/lib/OpenFHE/OpenFHEConfigVersion.cmake"
OPENFHE_LIB="$DEPS/lib/libOPENFHEcore_static.a"

if [ "${REBUILD_OPENFHE:-0}" != 1 ] && [ -f "$OPENFHE_VER_FILE" ] && grep -qE 'PACKAGE_VERSION.*1\.4\.2' "$OPENFHE_VER_FILE" && [ -f "$OPENFHE_LIB" ]; then
    echo ""
    echo "--- OpenFHE 1.4.2 already installed, skipping (REBUILD_OPENFHE=1 forces) ---"
else
    echo ""
    mkdir -p "$OPENFHE_TMP"
    # Compute nodes have no internet: prefer a pre-fetched source (OPENFHE_SRC,
    # cloned on a login node) over the proxy-dependent clone.
    if [ -n "${OPENFHE_SRC:-}" ] && [ -d "$OPENFHE_SRC/.git" ]; then
        echo "--- Copying OpenFHE v1.4.2 source from $OPENFHE_SRC (no network) ---"
        cp -r "$OPENFHE_SRC" "$OPENFHE_TMP/src"
    else
        echo "--- Cloning OpenFHE v1.4.2 ---"
        GIT_SSL_NO_VERIFY=true git clone \
            https://github.com/openfheorg/openfhe-development.git \
            --branch v1.4.2 --depth 1 --recurse-submodules --shallow-submodules "$OPENFHE_TMP/src" 2>&1
    fi

    echo "--- Applying FIDESlib patches (base + mixedlimb) ---"
    cd "$OPENFHE_TMP/src"
    git config user.email "build@leonardo"
    git config user.name "build"
    git am "$OPENFHE_PATCH" 2>&1
    # FIDESlib HEAD requires SetScalingModSizePerLevel (api/CCParams.cpp) — added
    # by the mixedlimb patch; without it the FIDESlib build fails.
    git am "$FIDESLIB_SRC/deps/openfhe-1.4.2-mixedlimb.patch" 2>&1

    echo "--- Building OpenFHE ---"
    # -fPIC on the static archives: they get linked into the perseus._core Python .so.
    cmake -S "$OPENFHE_TMP/src" -B "$OPENFHE_TMP/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$DEPS" \
        -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_STATIC=ON \
        -DWITH_BE2=OFF -DWITH_BE4=OFF \
        -DGIT_SUBMOD_AUTO=OFF \
        2>&1
    cmake --build "$OPENFHE_TMP/build" --parallel 16 2>&1

    echo "--- Installing OpenFHE to $DEPS ---"
    cmake --install "$OPENFHE_TMP/build" 2>&1
    rm -rf "$OPENFHE_TMP"
    echo "OpenFHE installed."
fi

cd "$REPO"   # return after possible cd into deleted OPENFHE_TMP

# ══════════════════════════════════════════════════════════════════════════
# STAGE 2: FIDESlib (against already-installed OpenFHE)
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "--- Configuring FIDESlib (FIDESLIB_INSTALL_OPENFHE=OFF) ---"
rm -rf "$FIDESLIB_BUILD"
cmake -S "$FIDESLIB_SRC" -B "$FIDESLIB_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$DEPS" \
    -DFIDESLIB_INSTALL_PREFIX="$DEPS" \
    -DOPENFHE_INSTALL_PREFIX="$DEPS" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_CUDA_COMPILER="$CUDA_NVCC" \
    -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" \
    -DCUDA_PATH="$CUDA_HOME" \
    -DFIDESLIB_ARCH="$GPU_ARCH" \
    -DNVTX3_INCLUDE="$CUDA_HOME/targets/x86_64-linux/include" \
    -DCMAKE_LIBRARY_PATH="$_nccl_lib" \
    -DCMAKE_INCLUDE_PATH="$_nccl_inc" \
    -DFIDESLIB_INSTALL_OPENFHE=OFF \
    -DFIDESLIB_COMPILE_TESTS=OFF \
    -DFIDESLIB_COMPILE_BENCHMARKS=OFF \
    -DFIDESLIB_COMPILE_PYTHON_BINDINGS=OFF \
    2>&1

echo ""
echo "--- Building FIDESlib (16 parallel jobs) ---"
cmake --build "$FIDESLIB_BUILD" --parallel 16 2>&1

echo ""
echo "--- Installing FIDESlib to $DEPS ---"
cmake --install "$FIDESLIB_BUILD" 2>&1

# ── Verify ────────────────────────────────────────────────────────────────
echo ""
echo "--- Installed files ---"
ls "$DEPS/lib/" | grep -E "fideslib|OPENFHE|openfhe" || echo "WARNING: no lib files found"
ls "$DEPS/include/" | head -10

echo ""
echo "=== Install complete: $DEPS ==="
echo "Date: $(date)"
