#!/bin/bash
# perseus calibration on a compute node. Extra args are hydra overrides, e.g.:
#   sbatch scripts/utils/calibrate_gpt2.sh n_calib_batches=8 approximation=gpt2_cutmax \
#       calib_out_path=$PWD/configs/model/approximation/gpt2_smoke/configs.json
# First run per (dataset, tokenizer): build the token pool on a login node
# (perseus.calibrate.data) or rely on the compute-node proxy below.

#SBATCH --job-name=perseus_calibrate
#SBATCH -A IscrC_eff-SAM2
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=logs/calibrate/%x_%j.out
#SBATCH --error=logs/calibrate/%x_%j.err

set -e
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export http_proxy='http://login05:3140'
export https_proxy='http://login05:3140'

module load gcc cuda/12.6

REPO=/leonardo_work/IscrC_eff-SAM2/azirilli/pycudafideslib
cd "$REPO"

export CACHE_DIR=${CACHE_DIR:-$SCRATCH/.cache}
export HF_HOME=$CACHE_DIR
export HF_DATASETS_CACHE=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HYDRA_FULL_ERROR=1

VENV=${VENV:-/leonardo_work/IscrC_eff-SAM2/azirilli/he-aware-training/.venv}
source "$VENV/bin/activate"

PYTHONPATH=$REPO python -m perseus.calibrate "$@"
