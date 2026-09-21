#!/bin/bash
# local_build_core.sh — build perseus._core (the CUDA extension) and cuda_cachemir for ONE chain.
#
#   CHAIN=n32 bash scripts/local_build_core.sh      # the paper's 32-bit composite chain (default)
#   CHAIN=n64 bash scripts/local_build_core.sh      # the 64-bit reference chain
#
# Needs the chain's deps tree (scripts/install_deps.sh -> deps_<chain>/) and the project venv
# (uv sync). pybind11 fixes the module filename, so each chain's build is stashed as
# perseus/_core.<chain>.so and the import name is a symlink to the active one: the two chains
# are numerically different libraries, never variants of one build.
# Env: JOBS (parallel compile jobs, default 16), CUDA_ARCH (e.g. 120-real; default = the first
# visible GPU's compute capability, else 80-real), TMPDIR, PYTHON.
set -e -o pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
CHAIN_REQUESTED="${CHAIN:-}"
source scripts/local_env.sh                       # CHAIN, FHE_DEPS_DIR, FHE_BUILD_DIR, CUDA_HOME, PYTHON

[ -x "$PYTHON" ] || { echo "no python at $PYTHON — run 'uv sync' first"; exit 1; }
[ -d "$FHE_DEPS_DIR" ] || { echo "missing deps tree $FHE_DEPS_DIR — run 'NATIVE_SIZE=${CHAIN#n} bash scripts/install_deps.sh'"; exit 1; }
[ -n "$CHAIN_REQUESTED" ] || echo "== CHAIN not set by the caller; building the default chain '$CHAIN'"

SO_NAME="_core$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
STASH="perseus/_core.${CHAIN}.so"
PY_INC="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
[ -f "$PY_INC/Python.h" ] || { echo "Python headers not found at $PY_INC (install the python-dev package or use a uv-managed python)"; exit 1; }
if [ -z "${CUDA_ARCH:-}" ]; then
    _cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')"
    CUDA_ARCH="${_cap:+${_cap}-real}"; CUDA_ARCH="${CUDA_ARCH:-80-real}"
fi

if [ ! -f "$FHE_BUILD_DIR/CMakeCache.txt" ]; then
    echo "== configure $FHE_BUILD_DIR (chain=$CHAIN, deps=$FHE_DEPS_DIR, arch=$CUDA_ARCH)"
    cmake -S . -B "$FHE_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
        -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
        -DFIDESLIB_ROOT="$FHE_DEPS_DIR" \
        -DCACHEMIR_BUILD_PYTHON=ON -DCACHEMIR_BUILD_TESTS=OFF \
        -Dpybind11_DIR="$("$PYTHON" -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
        -DPYBIND11_FINDPYTHON=NEW \
        -DPython_EXECUTABLE="$PYTHON" \
        -DPython_INCLUDE_DIR="$PY_INC" \
        ${LibArchive_INCLUDE_DIR:+-DLibArchive_INCLUDE_DIR=$LibArchive_INCLUDE_DIR} \
        ${LibArchive_LIBRARY:+-DLibArchive_LIBRARY=$LibArchive_LIBRARY}
fi

# a running process keeps its mapped inode: rename the old module aside, link a fresh one
[ -f "perseus/$SO_NAME" ] && [ ! -L "perseus/$SO_NAME" ] && mv "perseus/$SO_NAME" "perseus/$SO_NAME.stale.$$"
rm -f "perseus/$SO_NAME"
cmake --build "$FHE_BUILD_DIR" --parallel "${JOBS:-16}" --target _core cuda_cachemir
rm -f "perseus/$SO_NAME.stale.$$"
mv "perseus/$SO_NAME" "$STASH"

OTHER_CHAIN=$([ "$CHAIN" = "n64" ] && echo n32 || echo n64)
if [ -f "perseus/_core.${OTHER_CHAIN}.so" ] && cmp -s "$STASH" "perseus/_core.${OTHER_CHAIN}.so"; then
    echo "== FATAL: $STASH is byte-identical to the $OTHER_CHAIN stash: this build did not target chain=$CHAIN" >&2
    echo "==   (export CHAIN; a 'CHAIN=x source' prefix does not persist). Symlink left unchanged." >&2
    exit 1
fi
ln -sfn "$(basename "$STASH")" "perseus/$SO_NAME"
echo "== built chain=$CHAIN -> $STASH ; perseus/$SO_NAME -> $(readlink "perseus/$SO_NAME") ; $FHE_BUILD_DIR/bin/cuda_cachemir"
