#!/bin/bash
#SBATCH --job-name nb
#SBATCH -A EUHPC_D34_099
#SBATCH --time 01:00:00
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=256G
#SBATCH --output=logs/core/%x_%j.out
#SBATCH --error=logs/core/%x_%j.err

# =============================================================================
#  run_notebooks.sh — execute a notebook in place on GPU with the CURRENT baseline
#  plan/env, writing outputs (and any error traceback) back into the .ipynb. One
#  notebook = one job, env-discriminated by NB (mirrors run_task.sh). Every preset
#  uses ${VAR:-default}, so any env passed at sbatch time overrides it.
#
#    NB = custom      custom_encrypted_model.ipynb  torch-alike custom model; self-plans
#       | gpt2_torch  gpt2_torch_forward.ipynb      prefill->hand-off->generate (run_generate)
#       | vit_torch   vit_torch_forward.ipynb       EncViT forward; planned_vit_complex_112
#       | bert_torch  bert_torch_forward.ipynb      EncBert forward; planned_bert_base
#       | setup       setup_artifacts.ipynb         ARTIFACTS ONLY (export/calibrate/plan);
#                                                   FAMILY=bert|vit|gpt2, runs no model
#
#  Submit from a CLEAN shell (no modules loaded) — see CLAUDE.md.
# =============================================================================
set -e -o pipefail

REPO="$(pwd)"; DEPS="$REPO/deps"; BP="$REPO/bootstrap_placements"
CFG="$REPO/configs/model/approximation"
module load cuda/12.6 gcc cmake nccl
module use /leonardo/prod/spack/06/modules/0.22.2_preprod_base
module load libarchive/3.7.1--gcc--12.2.0-sw6t2mm
export LD_LIBRARY_PATH="$DEPS/lib:$DEPS/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1
export TMPDIR="$SCRATCH/tmp.${SLURM_JOB_ID:-login}"; mkdir -p "$TMPDIR" logs/core
PYTHON="${PYTHON:-$REPO/../he-aware-training/.venv/bin/python}"

NB="${NB:?set NB=setup|custom|gpt2_torch|vit_torch|bert_torch}"

# frozen CKKS chain (shared)
export LOGN="${LOGN:-16}" AUTO_BTS_LEVEL="${AUTO_BTS_LEVEL:-24}" BTS_ITERATIONS="${BTS_ITERATIONS:-1}"

gpt2_common() {   # the decode-baseline GPT-2 env (run_task.sh decode preset)
    export MULTI_T="${MULTI_T:-4}" STEPS_T="${STEPS_T:-128}"
    export CKKS_COMPLEX="${CKKS_COMPLEX:-1}" GPT2_PACKING="${GPT2_PACKING:-cachemir}"
    export FIDESLIB_ROT_KEY_BAND="${FIDESLIB_ROT_KEY_BAND:-11}"
    export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
    export GPT2_CACHE="${GPT2_CACHE:-1}"
    export GPT2_FOLD_LN1="${GPT2_FOLD_LN1:-1}" GPT2_FOLD_LN2="${GPT2_FOLD_LN2:-1}" \
           GPT2_FOLD_LNF="${GPT2_FOLD_LNF:-0}"
    export CACHE_READ_LEVEL_K="${CACHE_READ_LEVEL_K:-17}" CACHE_READ_LEVEL_V="${CACHE_READ_LEVEL_V:-17}"
    export FHE_LMHEAD_CAP="${FHE_LMHEAD_CAP:-22}"
    export CUTMAX_PRECISE_SCOPED="${CUTMAX_PRECISE_SCOPED:-1}" CUTMAX_VEC_BTS_ITERS="${CUTMAX_VEC_BTS_ITERS:-1}"
    export CONFIGS_PATH="${CONFIGS_PATH:-$CFG/gpt2_base/configs.json}"
    export WEIGHTS_PATH="${WEIGHTS_PATH:-$SCRATCH/.cache/perseus/models/openai-community/gpt2/classic/weights.bin.zip}"
    export GPT2_INFERENCE_MODE="${GPT2_INFERENCE_MODE:-threaded}"
}

case "$NB" in
  custom)
    NBPATH="notebooks/custom_encrypted_model.ipynb"; TIMEOUT=1500
    gpt2_common
    # NOTE the toy fires 0 eager bootstraps and that is CORRECT: one MLP block runs
    # L16->L17, nowhere near AUTO_BTS_LEVEL=24. Do not "fix" it —
    #   * deepening breaks it: chaining needs the head-style weight rearrangement the
    #     attention path uses (see the notebook), and a stack walks the GELU past its
    #     fitted xmax (rel_err 24 at DEPTH=3, 4.2e6 at DEPTH=10);
    #   * lowering AUTO_BTS_LEVEL breaks it worse: at 17 the forced bootstrap lands on the
    #     4096-wide expand intermediate, past the EvalMod wall -> rel_err 1.5e4. The chain
    #     is frozen at 24 exactly (CLAUDE.md).
    # The planner correctly places NOTHING here (placements=0); the old "0 -> 3" reading
    # was a metric bug (summary["total_bootstraps"] != num_placements), now fixed.
    unset FHE_BOOTSTRAP_PLACEMENTS_DIR   # the notebook builds (captures+plans) its own
    ;;
  gpt2_torch)
    # prefill -> hand-off -> generate (run_generate), eager by design (correctness row);
    # mirrors run_task.sh gen_prefill + prefill-phase ship staging.
    NBPATH="notebooks/gpt2_torch_forward.ipynb"; TIMEOUT=5400
    gpt2_common
    # offline HF cache for the GPT-2 tokenizer (detokenizing the generated ids)
    export HF_HOME="$SCRATCH/.cache/huggingface" HF_HUB_CACHE="$SCRATCH/.cache/huggingface/hub" HF_HUB_OFFLINE=1
    # GEN_PROMPT+GEN_TOKENS must stay <= t = slots/hidDim = 32 TOTAL positions. The gen
    # capture ran 2 tokens from position 0, so it only ever recorded ONE attention group;
    # past position 32 the runtime executes qkt_group.g1 ops that graph never saw and the
    # plan cannot place them (plan_level_error). 8+10=18 keeps it inside group 0.
    export GEN_PROMPT="${GEN_PROMPT:-8}" GEN_TOKENS="${GEN_TOKENS:-10}"
    export FHE_DELTA_BLOCK="${FHE_DELTA_BLOCK:-1}"
    # staging OFF: gen = prefill + decode, and the prefill-phase FHE_STAGE_RELEASE_CPU frees
    # weights the decode phase re-extracts -> assertion. Plain weight load is correct here.
    export FHE_PT_STAGE_BLOCK="${FHE_PT_STAGE_BLOCK:-0}" FHE_PT_COEFF_ENCODE="${FHE_PT_COEFF_ENCODE:-0}"
    # run_generate (pipeline.cu:309) takes ONE plan dir and loads it with
    # load_block_plans(plan_dir, N_blocks+3) — flat block_*_placement.json. It does NOT
    # read FHE_DECODE_PLACEMENTS_DIR (that belongs to run_prefill, pipeline.cu:505), and
    # it cannot load the chunked prefill dirs (planned_gpt2_prefill_delta_T* hold only
    # chunk_0/), which load silently as EMPTY -> eager. planned_gpt2_gen is the flat
    # 13-block autoregressive plan and the only one shaped for this path.
    export FHE_BOOTSTRAP_PLACEMENTS_DIR="${FHE_BOOTSTRAP_PLACEMENTS_DIR:-$BP/planned_gpt2_gen}"
    unset FHE_DECODE_PLACEMENTS_DIR   # unused by run_generate
    ;;
  vit_torch)
    NBPATH="notebooks/vit_torch_forward.ipynb"; TIMEOUT=1500
    export GATE_BLOCKS="${GATE_BLOCKS:-12}" GATE_RES="${GATE_RES:-112}"
    export CONFIGS_PATH="${CONFIGS_PATH:-$CFG/vit_base_112/configs.json}"
    export CKKS_COMPLEX="${CKKS_COMPLEX:-1}"
    export VIT_WEIGHT_GRANULARITY="${VIT_WEIGHT_GRANULARITY:-plaintext}"
    export FHE_PT_STAGE_BLOCK="${FHE_PT_STAGE_BLOCK:-16}" FHE_STAGE_ARENA_GB="${FHE_STAGE_ARENA_GB:-12}"
    export FHE_STAGE_RELEASE_CPU="${FHE_STAGE_RELEASE_CPU:-1}" FHE_PT_COEFF_ENCODE="${FHE_PT_COEFF_ENCODE:-1}"
    export VIT_MODEL_DIR="${VIT_MODEL_DIR:-$SCRATCH/.cache/perseus/models/google/vit-base-patch16-224/classic}"
    export VIT_POOL="${VIT_POOL:-$SCRATCH/.cache/huggingface/perseus/pools/tiny_imagenet_vit-base-patch16-224_512.npy}"
    export HF_HOME="$SCRATCH/.cache/huggingface" HF_HUB_CACHE="$SCRATCH/.cache/huggingface/hub" HF_HUB_OFFLINE=1
    export FHE_DELTA_BLOCK="${FHE_DELTA_BLOCK:-1}" FHE_KV_OVERLAP="${FHE_KV_OVERLAP:-0}"
    export MALLOC_TOP_PAD_="${MALLOC_TOP_PAD_:-134217728}"; unset MALLOC_ARENA_MAX
    export FIDESLIB_ROT_KEY_BAND="${FIDESLIB_ROT_KEY_BAND:-11}"
    export VIT_MODE="${VIT_MODE:-threaded}"
    unset GPT2_FOLD_LN1 GPT2_FOLD_LN2 GPT2_FOLD_LNF
    export FHE_BOOTSTRAP_PLACEMENTS_DIR="${FHE_BOOTSTRAP_PLACEMENTS_DIR:-$BP/planned_vit_complex_112}"
    ;;
  setup)
    # setup_artifacts.ipynb — ARTIFACTS ONLY: export -> pool -> calibrate -> probe ->
    # plan. It never runs the encrypted model (the model notebooks do that), so it needs
    # none of the FHE runtime env the other arms set — only the HF cache and FAMILY.
    # The GPU allocation is for calibration, which runs the plaintext model on device.
    # Each step is guarded by an existence check, so re-running is cheap; TIMEOUT covers
    # a cold calibration. FAMILY=bert|vit|gpt2.
    NBPATH="notebooks/setup_artifacts.ipynb"; TIMEOUT=5400
    export FAMILY="${FAMILY:-bert}"
    # UNCONDITIONAL: the login profile exports another project's unwritable HF_HOME and
    # sbatch inherits it (a ${HF_HOME:-...} default silently keeps the bad value).
    export HF_HOME="$SCRATCH/.cache" HF_HUB_CACHE="$SCRATCH/.cache"
    export HF_DATASETS_CACHE="$SCRATCH/.cache"
    # Pool building needs network and compute nodes have none — so an uncached pool must
    # be built on a login node first. Everything else here is offline-safe.
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    export HYDRA_FULL_ERROR=1
    unset FHE_BOOTSTRAP_PLACEMENTS_DIR FHE_GRAPH_DIR
    ;;
  bert_torch)
    # EncBert forward on planned_bert_base. Mirrors run_task.sh TASK=bert: REAL arm
    # (no CKKS_COMPLEX), ship staging, band 11. Head is client-side (tanh has no FHE
    # approximation), so there is no encrypted tail and no cutmax.
    NBPATH="notebooks/bert_torch_forward.ipynb"; TIMEOUT=1500
    export GATE_BLOCKS="${GATE_BLOCKS:-12}"
    export CONFIGS_PATH="${CONFIGS_PATH:-$CFG/bert_base/configs.json}"
    export BERT_MODEL="${BERT_MODEL:-textattack/bert-base-uncased-SST-2}"
    export BERT_MODEL_DIR="${BERT_MODEL_DIR:-$SCRATCH/.cache/perseus/models/textattack/bert-base-uncased-SST-2/classic}"
    export CKKS_COMPLEX="${CKKS_COMPLEX:-0}"
    export BERT_WEIGHT_GRANULARITY="${BERT_WEIGHT_GRANULARITY:-plaintext}"
    export FHE_PT_STAGE_BLOCK="${FHE_PT_STAGE_BLOCK:-16}" FHE_STAGE_ARENA_GB="${FHE_STAGE_ARENA_GB:-12}"
    export FHE_STAGE_RELEASE_CPU="${FHE_STAGE_RELEASE_CPU:-1}" FHE_PT_COEFF_ENCODE="${FHE_PT_COEFF_ENCODE:-1}"
    # UNCONDITIONAL (like run_task.sh's bert arm): the login profile exports another
    # project's unwritable HF_HOME and sbatch inherits it.
    export HF_HOME="$SCRATCH/.cache" HF_HUB_CACHE="$SCRATCH/.cache"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export FHE_DELTA_BLOCK="${FHE_DELTA_BLOCK:-1}" FHE_KV_OVERLAP="${FHE_KV_OVERLAP:-0}"
    export MALLOC_TOP_PAD_="${MALLOC_TOP_PAD_:-134217728}"; unset MALLOC_ARENA_MAX
    export FIDESLIB_ROT_KEY_BAND="${FIDESLIB_ROT_KEY_BAND:-11}"
    export BERT_MODE="${BERT_MODE:-threaded}"
    unset GPT2_FOLD_LN1 GPT2_FOLD_LN2 GPT2_FOLD_LNF
    export FHE_BOOTSTRAP_PLACEMENTS_DIR="${FHE_BOOTSTRAP_PLACEMENTS_DIR:-$BP/planned_bert_base}"
    ;;
  *) echo "unknown NB $NB (expected custom|gpt2_session|gpt2_torch|vit_torch|bert_torch)"; exit 1 ;;
esac
unset FHE_GRAPH_DIR FHE_PROFILE

# Unique kernel name per job: `ipykernel install --user` rmtree+reinstalls a SHARED
# home kernelspec dir; concurrent jobs race there (FileNotFoundError on the logo files).
KNAME="perseus_${SLURM_JOB_ID:-$$}"
echo "[nb] $NB -> $NBPATH  plan=${FHE_BOOTSTRAP_PLACEMENTS_DIR:-(self/eager)}  "\
     "mode=${GPT2_INFERENCE_MODE:-${VIT_MODE:-}}  kernel=$KNAME"
set +e   # capture the notebook exit code ourselves (heredoc exits 1 on cell failure)
"$PYTHON" -m ipykernel install --user --name "$KNAME" >/dev/null 2>&1

# execute in place; write outputs (incl. any failing-cell traceback) back either way.
env NBPATH="$NBPATH" TIMEOUT="$TIMEOUT" KNAME="$KNAME" "$PYTHON" - <<'EOF'
import os, sys
import nbformat
from nbclient import NotebookClient

path = os.environ["NBPATH"]
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=int(os.environ["TIMEOUT"]), kernel_name=os.environ["KNAME"])
ok, err = True, ""
try:
    client.execute()
except Exception as e:                       # CellExecutionError etc.
    ok, err = False, f"{type(e).__name__}: {e}"
nbformat.write(nb, path)                      # persist outputs regardless
print(f"[notebook] {'executed and saved' if ok else 'FAILED (saved partial): ' + err}: {path}",
      flush=True)
sys.exit(0 if ok else 1)
EOF
NB_RC=$?
"$PYTHON" -m jupyter kernelspec remove -f "$KNAME" >/dev/null 2>&1 || true

echo "=== Done === $(date)"
exit $NB_RC
