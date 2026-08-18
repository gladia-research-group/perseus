#!/bin/bash
#SBATCH --job-name=rebuild_pin509990c
#SBATCH -A EUHPC_D34_099
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH -p boost_usr_prod
#SBATCH --mem=64G
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --error=logs/build/%j.err
#SBATCH --output=logs/build/%j.out
# Pin bump 4619590 -> 509990c (behemoth-port): rebuild FIDESlib, install to deps, relink _core.
# Compute-node only: a login build leaves build/ mixed and the next compute build silently runs stale.
set -e
REPO="$(pwd)"; DEPS="$REPO/deps"; FB="$REPO/third_party/FIDESlib/build"
PIN=509990c0091ee2f9ccc505f5b650dfc2c79f5cce
SO=perseus/_core.cpython-311-x86_64-linux-gnu.so

echo "=== host $(hostname) | $(date) ==="
HAVE=$(git -C third_party/FIDESlib rev-parse HEAD)
[ "$HAVE" = "$PIN" ] || { echo "FATAL: FIDESlib at $HAVE, expected $PIN"; exit 1; }
echo "FIDESlib pin OK: $HAVE ($(git -C third_party/FIDESlib rev-parse --abbrev-ref HEAD))"

module load cuda/12.6 gcc cmake nccl
module use /leonardo/prod/spack/06/modules/0.22.2_preprod_base
module load libarchive/3.7.1--gcc--12.2.0-sw6t2mm
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export LD_LIBRARY_PATH="$DEPS/lib:$DEPS/lib64:${LD_LIBRARY_PATH:-}"
export TMPDIR=$SCRATCH/tmp.build && mkdir -p "$TMPDIR"

echo ""
echo "--- 1/3 FIDESlib incremental build (6 TUs changed vs 4619590) ---"
cmake --build "$FB" --parallel 16
cmake --install "$FB"
ls -la "$DEPS/lib64/fideslib.a"

echo ""
echo "--- 2/3 relink _core + cuda_cachemir against the new fideslib.a ---"
# rename-aside so any running job keeps its mapped inode; ld writes a fresh file in place
[ -f "$SO" ] && mv "$SO" "$SO.stale.$$"
cmake --build build-py --parallel 8 --target _core cuda_cachemir
rm -f "$SO.stale.$$"

echo ""
echo "--- 3/3 verify the artifact is fresh ---"
ls -la "$SO"
python - <<'PY'
import os, time
so = "perseus/_core.cpython-311-x86_64-linux-gnu.so"
age = time.time() - os.path.getmtime(so)
print(f"_core.so age: {age:.0f}s")
assert age < 3600, "STALE: _core.so was not relinked by this job"
print("BUILD OK")
PY
