#!/bin/bash
# ==============================================================================
# Test job: GP-based active learning, ~30 min, 2 GPUs (no data generation)
#
# Submit from repo root (partition/account/gres come from cluster.conf):
#   sbatch $(slurm/cluster_flags.sh 2gpu) slurm/test_job.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/al_test_<JOBID>.out
#
# Expected output: active_learning_test_output/ with 5 AL iterations
# ==============================================================================
#SBATCH --job-name=al_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --constraint=normalmem
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Partition, account, and gres are set via sbatch flags from cluster.conf.
# See slurm/cluster.conf.template for details.

set -euo pipefail

# ---- Resolve repo root (safe regardless of where sbatch was called from) ----
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs"

# ---- Cluster config ----------------------------------------------------------
if [[ -f "${REPO_ROOT}/slurm/cluster.conf" ]]; then
    source "${REPO_ROOT}/slurm/cluster.conf"
else
    echo "[warn] slurm/cluster.conf not found — using defaults"
    echo "       cp slurm/cluster.conf.template slurm/cluster.conf"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/al_pmssmwithgp/model:${PYTHONPATH:-}"

echo "=========================================="
echo " Job:     ${SLURM_JOB_NAME} | ID: ${SLURM_JOB_ID}"
echo " Node:    $(hostname)"
echo " Started: $(date)"
echo " Repo:    ${REPO_ROOT}"
echo "=========================================="

# ---- Pixi environment --------------------------------------------------------
PIXI_ENV="${PIXI_ENV:-cuda}"
PYTHON="${REPO_ROOT}/.pixi/envs/${PIXI_ENV}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "[setup] pixi env '${PIXI_ENV}' not found — running: pixi install -e ${PIXI_ENV}"
    /u/jwuerzin/.pixi/bin/pixi install -e "${PIXI_ENV}"
fi
if [[ ! -x "${PYTHON}" ]]; then
    echo "[error] Python executable not found after pixi install: ${PYTHON}"
    exit 1
fi
echo "[env] $(${PYTHON} --version)"

# ---- GPU report --------------------------------------------------------------
echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "[gpu] ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<not set>}"
echo "[gpu] SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<not set>}"
echo "[gpu] PIXI_ENV=${PIXI_ENV}"

# ---- Run active learning (exact_gp, 5 iterations, no data generation) --------
# --gpu-ids 0,1: AL model on cuda:0, baseline on cuda:1 (parallel processes)
# Slurm remaps physical GPUs via CUDA_VISIBLE_DEVICES, so 0,1 are always correct
"${PYTHON}" active_learning_gp.py \
    --model-type exact_gp \
    --n-samples 2000 \
    --n-iterations 5 \
    --epochs 500 \
    --early-stopping \
    --patience 100 \
    --learning-rate 1e-3 \
    --n-candidates 20000 \
    --n-select 10 \
    --kernel RBF \
    --lengthscale 1.0 \
    --noise 1e-2 \
    --jitter 1e-3 \
    --use-ard \
    --warm-starting \
    --data-dir /ptmp/jwuerzin/data \
    --output-dir /ptmp/jwuerzin/active_learning_test_output \
    --gpu-ids 0,1

echo "=========================================="
echo " Done: $(date)"
echo "=========================================="
