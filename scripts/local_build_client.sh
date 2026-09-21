#!/bin/bash
# local_build_client.sh — build perseus._client, the CUDA-free client extension (OpenFHE only).
#
#   CHAIN=n32 bash scripts/local_build_client.sh
#
# Links only the chain's patched OpenFHE static archives from deps_<chain>/ (no CUDA toolchain,
# no FIDESlib), so a client machine without a GPU can generate keys, encrypt and decrypt.
# Stash + import-symlink convention as in local_build_core.sh (perseus/_client.<chain>.so).
# Env: JOBS (default 8), TMPDIR, PYTHON, CLIENT_BUILD_DIR.
set -e -o pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
CHAIN="${CHAIN:-n32}"
FHE_DEPS_DIR="${FHE_DEPS_DIR:-$REPO/deps_$CHAIN}"
CLIENT_BUILD_DIR="${CLIENT_BUILD_DIR:-$REPO/build_client_$CHAIN}"
PY="${PYTHON:-$REPO/.venv/bin/python}"
export TMPDIR="${TMPDIR:-$REPO/.cache/tmp}"; mkdir -p "$TMPDIR"

[ -x "$PY" ] || { echo "no python at $PY — run 'uv sync' first"; exit 1; }
[ -d "$FHE_DEPS_DIR" ] || { echo "missing deps tree $FHE_DEPS_DIR (patched OpenFHE: include/openfhe + lib/libOPENFHE*_static.a)"; exit 1; }
SO_NAME="_client$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
STASH="perseus/_client.${CHAIN}.so"
PY_INC="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
[ -f "$PY_INC/Python.h" ] || { echo "Python headers not found at $PY_INC"; exit 1; }

if [ ! -f "$CLIENT_BUILD_DIR/CMakeCache.txt" ]; then
    echo "== configure $CLIENT_BUILD_DIR (chain=$CHAIN, deps=$FHE_DEPS_DIR, client-only)"
    cmake -S . -B "$CLIENT_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCACHEMIR_CLIENT_ONLY=ON -DCACHEMIR_BUILD_TESTS=OFF \
        -DFIDESLIB_ROOT="$FHE_DEPS_DIR" \
        -Dpybind11_DIR="$("$PY" -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
        -DPYBIND11_FINDPYTHON=NEW \
        -DPython_EXECUTABLE="$PY" \
        -DPython_INCLUDE_DIR="$PY_INC"
fi
[ -f "perseus/$SO_NAME" ] && [ ! -L "perseus/$SO_NAME" ] && mv "perseus/$SO_NAME" "perseus/$SO_NAME.stale.$$"
rm -f "perseus/$SO_NAME"
cmake --build "$CLIENT_BUILD_DIR" --parallel "${JOBS:-8}" --target _client
rm -f "perseus/$SO_NAME.stale.$$"
mv "perseus/$SO_NAME" "$STASH"

OTHER_CHAIN=$([ "$CHAIN" = "n64" ] && echo n32 || echo n64)
if [ -f "perseus/_client.${OTHER_CHAIN}.so" ] && cmp -s "$STASH" "perseus/_client.${OTHER_CHAIN}.so"; then
    echo "== FATAL: $STASH is byte-identical to the $OTHER_CHAIN stash: this build did not target chain=$CHAIN" >&2
    exit 1
fi
if readelf -d "$STASH" | grep -qE 'libcuda|libcudart|libnccl'; then
    echo "== FATAL: $STASH links a CUDA library; perseus._client must be CUDA-free" >&2
    exit 1
fi
ln -sfn "$(basename "$STASH")" "perseus/$SO_NAME"
echo "== built chain=$CHAIN -> $STASH ; perseus/$SO_NAME -> $(readlink "perseus/$SO_NAME")"
