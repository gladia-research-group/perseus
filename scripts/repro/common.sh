# Shared setup of the result runs: the data paths from the repo root, the GPU (GPU=<index>, the device the run may
# use), an optional second GPU for the prefills (REMOTE=<index>, the two-GPU split, PC_REMOTE_DEVICE), and an optional
# NUMA binding (NODE=host node, MEMS=its memory nodes) so the host side runs next to the GPU.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONUNBUFFERED=1
source scripts/local_env.sh
# GPU defaults to the caller's first visible device; a GPU outside a set CUDA_VISIBLE_DEVICES is refused
: "${GPU:=${CUDA_VISIBLE_DEVICES%%,*}}"; : "${GPU:=0}"
case ",${CUDA_VISIBLE_DEVICES:-$GPU}," in *",$GPU,"*) ;;
    *) echo "GPU=$GPU is outside CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2; exit 1 ;; esac
if [ -n "${REMOTE:-}" ]; then
    case ",${CUDA_VISIBLE_DEVICES:-$REMOTE}," in *",$REMOTE,"*) ;;
        *) echo "REMOTE=$REMOTE is outside CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2; exit 1 ;; esac
    [ "$REMOTE" != "$GPU" ] || { echo "REMOTE=$REMOTE is the run's own GPU" >&2; exit 1; }
    export CUDA_VISIBLE_DEVICES="$GPU,$REMOTE" PC_REMOTE_DEVICE=1
else
    export CUDA_VISIBLE_DEVICES="$GPU"
fi
PY=${PYTHON:-python}
bind() { if [ -n "${NODE:-}" ]; then numactl --cpunodebind="$NODE" --membind="${MEMS:-$NODE}" "$@"; else "$@"; fi; }
