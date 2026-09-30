#!/bin/bash
# run_notebooks.sh — execute one demo notebook headless with the paper's GPT-2 environment.
#   NB=setup|gpt2_torch|gpt2_nn|custom|client_server bash scripts/run_notebooks.sh
# Outputs are written back into the notebook; a cell error fails the run ([notebook] FAIL).
set -e -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
source scripts/local_env.sh
export PYTHONFAULTHANDLER=1 HYDRA_FULL_ERROR=1
NB="${NB:?set NB=setup|gpt2_torch|gpt2_nn|custom|client_server}"
BP="$REPO/bootstrap_placements"; CFG="$REPO/configs/model/approximation"

gpt2_env() {   # the decode row's GPT-2 environment (scripts/run_task.sh)
    export CKKS_COMPLEX="${CKKS_COMPLEX:-1}" GPT2_PACKING="${GPT2_PACKING:-cachemir}" GPT2_CACHE="${GPT2_CACHE:-1}"
    export STEPS_T="${STEPS_T:-128}" MULTI_T="${MULTI_T:-4}"
    export GPT2_FOLD_LN1="${GPT2_FOLD_LN1:-1}" GPT2_FOLD_LN2="${GPT2_FOLD_LN2:-1}" GPT2_FOLD_LNF="${GPT2_FOLD_LNF:-0}"
    export CUTMAX_PRECISE_SCOPED="${CUTMAX_PRECISE_SCOPED:-1}" CUTMAX_VEC_BTS_ITERS="${CUTMAX_VEC_BTS_ITERS:-1}"
    export GPT2_INFERENCE_MODE="${GPT2_INFERENCE_MODE:-threaded}"
    if [ "$CHAIN" = n32 ]; then export CONFIGS_PATH="${CONFIGS_PATH:-$CFG/gpt2_base_n32/configs.json}"
    else export CONFIGS_PATH="${CONFIGS_PATH:-$CFG/gpt2_base/configs.json}"; fi
}
case "$NB" in
  setup)         NBPATH=notebooks/setup_artifacts.ipynb ;;
  gpt2_torch)    NBPATH=notebooks/gpt2_torch_forward.ipynb;   gpt2_env; export GEN_PROMPT="${GEN_PROMPT:-8}" GEN_TOKENS="${GEN_TOKENS:-10}" FHE_DELTA_BLOCK="${FHE_DELTA_BLOCK:-1}" ;;
  gpt2_nn)       NBPATH=notebooks/gpt2_perseus_nn.ipynb;      gpt2_env; export FHE_DELTA_BLOCK="${FHE_DELTA_BLOCK:-1}" ;;
  custom)        NBPATH=notebooks/custom_encrypted_model.ipynb; gpt2_env ;;
  client_server) NBPATH=notebooks/client_server_minimal.ipynb; gpt2_env ;;
  *) echo "unknown NB $NB"; exit 1 ;;
esac
unset FHE_GRAPH_DIR FHE_BOOTSTRAP_PLACEMENTS_DIR   # the notebooks pick their own plan (or run eager)
# perseus._core is a symlink to the active chain's build and the two are numerically
# different libraries: the environment above is this chain's, so refuse a mismatch.
_loaded="$("$PYTHON" -c 'import perseus._core as c; print(c.chain)' 2>/dev/null || echo unknown)"
if [ "$_loaded" != "$CHAIN" ]; then
    _so="_core$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
    echo "[nb] perseus._core is the '$_loaded' build but CHAIN=$CHAIN. Relink it:" >&2
    echo "       ln -sfn _core.$CHAIN.so perseus/$_so" >&2
    exit 1
fi
echo "[nb] $NB -> $NBPATH (chain $CHAIN)"
NB_PATH="$NBPATH" exec "$PYTHON" scripts/utils/run_notebook.py
