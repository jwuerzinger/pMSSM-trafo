#!/bin/bash -l
# ==============================================================================
# Test job: ExactGP active learning smoke test on gpudev (15 min, 1 GPU)
#
# Submit from repo root:
#   mkdir -p logs
#   sbatch slurm/test_al_gp_exact.sh
#
# Expected output: /ptmp/jwuerzin/test_al_gp_exact_output/ with 2 AL iterations
# ==============================================================================
#SBATCH --job-name=test_al_gp_exact
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

# ---- Resolve repo root (safe regardless of where sbatch was called from) ----
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

# ---- GPU report --------------------------------------------------------------
echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "[gpu] SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<not set>}"

# ---- Run active learning (exact_gp, 5 iterations, no data generation) --------
# --gpu-ids 0,1: AL model on cuda:0, baseline on cuda:1 (parallel processes)
# Slurm remaps physical GPUs via CUDA_VISIBLE_DEVICES, so 0,1 are always correct
"${PYTHON}" active_learning_gp.py \
    --model-type exact_gp \
    --n-samples 500 \
    --n-iterations 2 \
    --epochs 100 \
    --early-stopping \
    --patience 50 \
    --learning-rate 1e-3 \
    --n-candidates 5000 \
    --n-select 5 \
    --kernel RBF \
    --lengthscale 1.0 \
    --noise 1e-2 \
    --jitter 1e-3 \
    --use-ard \
    --warm-starting \
    --output-dir /ptmp/jwuerzin/test_al_gp_exact_output \
    --gpu-ids 0

echo "=========================================="
echo " Done: $(date)"
echo "=========================================="
