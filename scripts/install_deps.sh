#!/bin/bash
# install_deps.sh — build the patched OpenFHE and FIDESlib32bits into deps_<chain>/.
#
#   NATIVE_SIZE=32 bash scripts/install_deps.sh     # the paper's 32-bit composite chain -> deps_n32/
#   NATIVE_SIZE=64 bash scripts/install_deps.sh     # the 64-bit reference chain            -> deps_n64/
#
# The 32-bit OpenFHE is a fixed upstream commit plus the patch series in
# third_party/openfhe-n32/patches (every shipped plan was measured against it); the 64-bit
# one is v1.4.2 plus FIDESlib's patches. Env: DEPS_DIR (default deps_n<NATIVE_SIZE>), CUDA_HOME
# (default: the nvcc on PATH, else /usr/local/cuda), NCCL_HOME (a prefix holding lib/libnccl.so and include/nccl.h,
# when NCCL is not installed system-wide), JOBS (default 16), GPU_ARCH (default: the first visible GPU's compute
# capability, else 80-real), OPENFHE_SRC (a local openfhe-development clone, for offline use),
# REBUILD_OPENFHE=1 (rebuild even if deps already holds OpenFHE 1.4.2).
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
NATIVE_SIZE="${NATIVE_SIZE:-32}"
DEPS="${DEPS_DIR:-$REPO/deps_n$NATIVE_SIZE}"
FIDESLIB_SRC="$REPO/third_party/FIDESlib"
FIDESLIB_BUILD="$FIDESLIB_SRC/build_n$NATIVE_SIZE"
OPENFHE_TMP="${OPENFHE_TMP:-${TMPDIR:-/tmp}/openfhe_build_$$}"
OPENFHE_N32_BASE="aa391988d354d4360f390f223a90e0d1b98839d7"
OPENFHE_N32_SERIES="$REPO/third_party/openfhe-n32/patches"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
echo "=== OpenFHE + FIDESlib32bits, NATIVE_SIZE=$NATIVE_SIZE -> $DEPS ==="

[ -f "$FIDESLIB_SRC/CMakeLists.txt" ] || git submodule update --init third_party/FIDESlib
if [ "$NATIVE_SIZE" = 32 ] && [ ! -f "$OPENFHE_N32_SERIES/0001-Minimal-Compatibility-with-FIDESlib.patch" ]; then
    echo "ERROR: $OPENFHE_N32_SERIES is missing (the 32-bit OpenFHE patch series)" >&2; exit 1
fi
mkdir -p "$DEPS"

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
export CC="${CC:-$(command -v gcc)}" CXX="${CXX:-$(command -v g++)}" CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
[ -x "$CUDA_HOME/bin/nvcc" ] || { echo "ERROR: nvcc not found under CUDA_HOME=$CUDA_HOME" >&2; exit 1; }
echo "CUDA_HOME=$CUDA_HOME  CXX=$CXX  $(cmake --version | head -1)"

_nccl_lib=""; _nccl_inc=""
for d in "${NCCL_HOME:-/nonexistent}/lib" "$CUDA_HOME/lib64" "$CUDA_HOME/targets/x86_64-linux/lib" /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib; do
    [ -z "$_nccl_lib" ] && [ -f "$d/libnccl.so" ] && _nccl_lib="$d"; done
for d in "${NCCL_HOME:-/nonexistent}/include" "$CUDA_HOME/include" "$CUDA_HOME/targets/x86_64-linux/include" /usr/include /usr/local/include; do
    [ -z "$_nccl_inc" ] && [ -f "$d/nccl.h" ] && _nccl_inc="$d"; done
if [ -z "$_nccl_lib" ] || [ -z "$_nccl_inc" ]; then
    echo "ERROR: NCCL not found (libnccl.so: ${_nccl_lib:-missing}, nccl.h: ${_nccl_inc:-missing})." >&2
    echo "       FIDESlib needs it even for single-GPU builds. Install libnccl2 + libnccl-dev," >&2
    echo "       or export NCCL_HOME=<prefix with lib/libnccl.so and include/nccl.h>." >&2
    exit 1
fi
echo "NCCL: $_nccl_lib"
NCCL_ARGS=(-DCMAKE_LIBRARY_PATH="$_nccl_lib" -DCMAKE_INCLUDE_PATH="$_nccl_inc")

if [ -z "${GPU_ARCH:-}" ]; then
    _cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1)"
    if [[ "$_cap" =~ ^[0-9]+\.[0-9]+$ ]]; then GPU_ARCH="$(echo "$_cap" | tr -d '.')-real"; else GPU_ARCH="80-real"; fi
fi
echo "GPU arch: $GPU_ARCH"

# ---- OpenFHE
OPENFHE_VER_FILE="$DEPS/lib/OpenFHE/OpenFHEConfigVersion.cmake"
if [ "${REBUILD_OPENFHE:-0}" != 1 ] && [ -f "$OPENFHE_VER_FILE" ] && grep -qE 'PACKAGE_VERSION.*1\.4\.2' "$OPENFHE_VER_FILE" && [ -f "$DEPS/lib/libOPENFHEcore_static.a" ]; then
    echo "--- OpenFHE already installed in $DEPS (REBUILD_OPENFHE=1 forces) ---"
else
    mkdir -p "$OPENFHE_TMP"
    if [ -n "${OPENFHE_SRC:-}" ] && [ -d "$OPENFHE_SRC/.git" ]; then
        echo "--- Copying OpenFHE source from $OPENFHE_SRC ---"; cp -r "$OPENFHE_SRC" "$OPENFHE_TMP/src"
    elif [ "$NATIVE_SIZE" = 32 ]; then
        echo "--- Cloning OpenFHE (full history; base $OPENFHE_N32_BASE) ---"
        git clone https://github.com/openfheorg/openfhe-development.git --recurse-submodules "$OPENFHE_TMP/src"
    else
        echo "--- Cloning OpenFHE v1.4.2 ---"
        git clone https://github.com/openfheorg/openfhe-development.git --branch v1.4.2 --depth 1 --recurse-submodules --shallow-submodules "$OPENFHE_TMP/src"
    fi
    [ -f "$OPENFHE_TMP/src/third-party/cereal/include/cereal/cereal.hpp" ] || { echo "ERROR: OpenFHE source lacks third-party/cereal (clone with --recurse-submodules)" >&2; exit 1; }
    cd "$OPENFHE_TMP/src"
    git config user.email "build@perseus" && git config user.name "perseus build"
    if [ "$NATIVE_SIZE" = 32 ]; then
        git cat-file -e "$OPENFHE_N32_BASE" 2>/dev/null || { echo "ERROR: commit $OPENFHE_N32_BASE not in the OpenFHE source (full clone needed)" >&2; exit 1; }
        git checkout -q "$OPENFHE_N32_BASE"
        echo "--- Applying the 32-bit patch series ---"
        git am "$OPENFHE_N32_SERIES"/*.patch
        # the key-switching-key seed expander exists on both sides and must be byte-identical
        cmp -s src/core/include/utils/kskseedexpand.h "$FIDESLIB_SRC/src/CKKS/KskSeedExpand.cuh" \
            || { echo "ERROR: kskseedexpand.h (OpenFHE patch) differs from FIDESlib's KskSeedExpand.cuh" >&2; exit 1; }
    else
        echo "--- Applying FIDESlib's OpenFHE patches (base, mixed-limb, seed-expanded keys) ---"
        git am "$FIDESLIB_SRC/deps/openfhe-1.4.2.patch"
        git am "$FIDESLIB_SRC/deps/openfhe-1.4.2-mixedlimb.patch"
        git am "$FIDESLIB_SRC/deps/openfhe-1.4.2-kska-seed.patch"
        cmp -s src/core/include/utils/kskseedexpand.h "$FIDESLIB_SRC/src/CKKS/KskSeedExpand.cuh" \
            || { echo "ERROR: kskseedexpand.h (OpenFHE patch) differs from FIDESlib's KskSeedExpand.cuh" >&2; exit 1; }
    fi
    echo "--- Building OpenFHE ---"
    cmake -S "$OPENFHE_TMP/src" -B "$OPENFHE_TMP/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$DEPS" \
        -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_STATIC=ON -DWITH_BE2=OFF -DWITH_BE4=OFF -DNATIVE_SIZE="$NATIVE_SIZE" -DGIT_SUBMOD_AUTO=OFF
    cmake --build "$OPENFHE_TMP/build" --parallel "${JOBS:-16}"
    cmake --install "$OPENFHE_TMP/build"
    cd "$REPO"; rm -rf "$OPENFHE_TMP"
    echo "OpenFHE installed."
fi

# ---- FIDESlib32bits
echo "--- Building FIDESlib (NATIVEINT=$NATIVE_SIZE, arch $GPU_ARCH) ---"
rm -rf "$FIDESLIB_BUILD"
cmake -S "$FIDESLIB_SRC" -B "$FIDESLIB_BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$DEPS" -DFIDESLIB_INSTALL_PREFIX="$DEPS" -DOPENFHE_INSTALL_PREFIX="$DEPS" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" -DCUDA_PATH="$CUDA_HOME" \
    -DFIDESLIB_ARCH="$GPU_ARCH" -DFIDESLIB_OPENFHE_NATIVE_SIZE="$NATIVE_SIZE" "${NCCL_ARGS[@]}" \
    -DFIDESLIB_INSTALL_OPENFHE=OFF -DFIDESLIB_COMPILE_TESTS=OFF -DFIDESLIB_COMPILE_BENCHMARKS=OFF
cmake --build "$FIDESLIB_BUILD" --parallel "${JOBS:-16}"
cmake --install "$FIDESLIB_BUILD"
echo "=== Installed: $(ls "$DEPS/lib" | grep -cE 'fideslib|OPENFHE') libraries in $DEPS ==="
