#!/bin/bash
#SBATCH --job-name=rebuild_fideslib_arena
#SBATCH -A EUHPC_D34_099
#SBATCH --qos=normal
#SBATCH --time=00:40:00
#SBATCH -p boost_usr_prod
#SBATCH --mem=64G
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --error=logs/decode_handoff/%j.err
#SBATCH --output=logs/decode_handoff/%j.out
set -e
REPO="$(pwd)"; DEPS="$REPO/deps"; FB="$REPO/third_party/FIDESlib/build"
module load cuda/12.6 gcc cmake nccl
module use /leonardo/prod/spack/06/modules/0.22.2_preprod_base
module load libarchive/3.7.1--gcc--12.2.0-sw6t2mm
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export LD_LIBRARY_PATH="$DEPS/lib:$DEPS/lib64:${LD_LIBRARY_PATH:-}"
# Incremental FIDESlib relink: KV_ARENA_GB env made kKvArenaBytes runtime (CryptoContext.cpp).
# Only that TU changed; force its .o out so cmake recompiles, then relink + install to deps.
find "$FB" -name 'CryptoContext.cpp.o' -delete
cmake --build "$FB" --parallel 16
cmake --install "$FB"
echo "=== fideslib rebuilt + installed: $(ls -la $DEPS/lib64/fideslib.a) ==="
