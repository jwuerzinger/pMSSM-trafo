#!/bin/bash -l
# ==============================================================================
# Slurm job: Transformer AL — top_k selection, n_select=20000
#
# Equivalent to run_active_learning_top_k_20000.sh for Slurm.
#
# Submit from repo root:
#   sbatch slurm/submit_al_transformer_top_k_20k.sh
#
# Override partition/account on the command line (takes precedence over #SBATCH):
#   sbatch --partition=rvs --account=mpp slurm/submit_al_transformer_top_k_20k.sh
# ==============================================================================
#SBATCH --job-name=al_transformer_top_k_20k
#SBATCH --constraint="gpu"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=36
#SBATCH --mem=250000
#SBATCH --time=24:00:00
#SBATCH --output=/raven/u/jwuerzin/pMSSM-trafo/logs/%x_%j.out
#SBATCH --error=/raven/u/jwuerzin/pMSSM-trafo/logs/%x_%j.err

set -euo pipefail

# ---- Resolve repo root -------------------------------------------------------
REPO_ROOT="${SLURM_SUBMIT_DIR:-/raven/u/jwuerzin/pMSSM-trafo}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/al_pmssmwithgp/model:${PYTHONPATH:-}"

echo "=========================================="
echo " Job:     ${SLURM_JOB_NAME} | ID: ${SLURM_JOB_ID}"
echo " Node:    $(hostname)"
echo " Started: $(date)"
echo " Repo:    ${REPO_ROOT}"
echo "=========================================="

# ---- Pixi environment --------------------------------------------------------
PYTHON="${REPO_ROOT}/.pixi/envs/default/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "[setup] pixi env not found — running: pixi install"
    /u/jwuerzin/.pixi/bin/pixi install
fi
if [[ ! -x "${PYTHON}" ]]; then
    echo "[error] Python executable not found after pixi install: ${PYTHON}"
    exit 1
fi
echo "[env] $(${PYTHON} --version)"

# ---- GPU count detection -----------------------------------------------------
if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" -gt 0 ]]; then
    N_GPUS="${SLURM_GPUS_ON_NODE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
    N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
else
    N_GPUS=1
fi
GPU_IDS=$( [[ "${N_GPUS}" -ge 2 ]] && echo "0,1" || echo "0" )

echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "[gpu] SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<not set>}"
echo "[gpu] Using --gpu-ids ${GPU_IDS} (${N_GPUS} GPU(s) detected)"

# ---- Run active learning -----------------------------------------------------
"${PYTHON}" active_learning.py \
    --y-transform log \
    --epochs 10000 \
    --generate-data \
    --n-samples 2000 \
    --n-iterations 40 \
    --n-candidates 1000000 \
    --entropy-pool-size 5000 \
    --gen-workers 20 \
    --mcmc-data-dir /ptmp/jwuerzin/data \
    --data-dir /ptmp/jwuerzin/data \
    --static-eval-size 100000 \
    --warm-starting \
    --early-stopping \
    --selection-strategy top_k \
    --n-select 20000 \
    --output-dir /ptmp/jwuerzin/active_learning_output_top_k_n_select_20k \
    --gpu-ids "${GPU_IDS}"

echo "=========================================="
echo " Done: $(date)"
echo "=========================================="
