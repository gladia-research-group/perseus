#!/bin/bash
# build_hecate.sh — build the released DaCapo compiler's hecate-opt with dacapo_4616402.patch.
#   LLVM_PREFIX=/usr/lib/llvm-18 bash scripts/utils/dacapo_upstream/build_hecate.sh [dest]
# Needs LLVM/MLIR 18 development files (Ubuntu: llvm-18-dev libmlir-18-dev mlir-18-tools)
# under LLVM_PREFIX (lib/cmake/llvm, lib/cmake/mlir). Upstream pins LLVM 16; with 18 its
# sources need only llvm18_compat.h (unqualified cast helpers) and -Wno-error. Its CMake
# requires SEAL 4.0 (SEAL_DIR, or built here from microsoft/SEAL v4.0.0); HEaaN is optional.
# Prints the path of hecate-opt.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HERE="$REPO/scripts/utils/dacapo_upstream"
DEST="${1:-$REPO/.cache/dacapo_upstream}"
LLVM_PREFIX="${LLVM_PREFIX:-/usr/lib/llvm-18}"
if [ ! -d "$DEST/.git" ]; then
    git clone -q https://github.com/corelab-src/dacapo "$DEST"
    git -C "$DEST" checkout -q 4616402
    git -C "$DEST" apply "$HERE/dacapo_4616402.patch"
fi
if [ -z "${SEAL_DIR:-}" ]; then
    SEAL_DIR="$DEST/seal/lib/cmake/SEAL-4.0"
    if [ ! -d "$SEAL_DIR" ]; then
        [ -d "$DEST/seal-src" ] || git clone -q --branch v4.0.0 --depth 1 https://github.com/microsoft/SEAL "$DEST/seal-src"
        cmake -S "$DEST/seal-src" -B "$DEST/seal-src/build" -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX="$DEST/seal" -DSEAL_BUILD_TESTS=OFF -DSEAL_BUILD_EXAMPLES=OFF \
            > "$DEST/seal.log" 2>&1
        cmake --build "$DEST/seal-src/build" --parallel "${JOBS:-8}" >> "$DEST/seal.log" 2>&1
        cmake --install "$DEST/seal-src/build" >> "$DEST/seal.log" 2>&1
    fi
fi
cmake -S "$DEST" -B "$DEST/build" -DCMAKE_BUILD_TYPE=Release -DSEAL_DIR="$SEAL_DIR" \
    -DLLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm" -DMLIR_DIR="$LLVM_PREFIX/lib/cmake/mlir" \
    -DCMAKE_CXX_FLAGS="-include $HERE/llvm18_compat.h -Wno-error" > "$DEST/build.log" 2>&1
cmake --build "$DEST/build" --target hecate-opt --parallel "${JOBS:-8}" >> "$DEST/build.log" 2>&1
echo "$DEST/build/bin/hecate-opt"
