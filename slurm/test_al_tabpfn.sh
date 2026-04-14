#!/bin/bash -l
# ==============================================================================
# Test job: TabPFN active learning smoke test on gpudev (15 min, 1 GPU)
#
# Submit from repo root:
#   mkdir -p logs
#   sbatch slurm/test_al_tabpfn.sh
#
# Expected output: /ptmp/jwuerzin/test_al_tabpfn_output/ with 2 AL iterations
# ==============================================================================
#SBATCH --job-name=test_al_tabpfn
#SBATCH --partition=gpudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#SBATCH --time=00:15:00
#SBATCH --output=/raven/u/jwuerzin/pMSSM-trafo/logs/%x_%j.out
#SBATCH --error=/raven/u/jwuerzin/pMSSM-trafo/logs/%x_%j.err

set -euo pipefail

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

echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"

"${PYTHON}" active_learning_tabpfn.py \
    --n-samples 500 \
    --n-iterations 2 \
    --n-select 5 \
    --n-candidates 5000 \
    --n-ensemble-samples 4 \
    --selection-strategy top_k \
    --proximity-sampling 0.1 \
    --tolerance-sampling 1.0 \
    --mcmc-data-dir /ptmp/jwuerzin/data/19250082 \
    --static-eval-size 10000 \
    --output-dir /ptmp/jwuerzin/test_al_tabpfn_output \
    --gpu-id 0

echo "=========================================="
echo " Done: $(date)"
echo "=========================================="
