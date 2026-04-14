#!/bin/bash
# ==============================================================================
# Test job: TabPFN active learning smoke test (1 GPU)
#
# Submit from repo root (partition/gres come from cluster.conf):
#   sbatch --partition="${CLUSTER_PARTITION_DEV}" \
#          --gres="${CLUSTER_GPU_GRES_1}" slurm/test_al_tabpfn.sh
#
# Expected output: /ptmp/jwuerzin/test_al_tabpfn_output/ with 2 AL iterations
# ==============================================================================
#SBATCH --job-name=test_al_tabpfn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Partition and gres are set via sbatch flags from cluster.conf.
# See slurm/cluster.conf.template for details.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs"

if [[ -f "${REPO_ROOT}/slurm/cluster.conf" ]]; then
    source "${REPO_ROOT}/slurm/cluster.conf"
else
    echo "[warn] slurm/cluster.conf not found — using defaults"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/al_pmssmwithgp/model:${PYTHONPATH:-}"

echo "=========================================="
echo " Job:     ${SLURM_JOB_NAME} | ID: ${SLURM_JOB_ID}"
echo " Node:    $(hostname)"
echo " Started: $(date)"
echo " Repo:    ${REPO_ROOT}"
echo "=========================================="

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

echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "[gpu] ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<not set>}"
echo "[gpu] SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<not set>}"
echo "[gpu] PIXI_ENV=${PIXI_ENV}"

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
    --data-dir /ptmp/jwuerzin/data/18387358 \
    --static-eval-size 10000 \
    --output-dir /ptmp/jwuerzin/test_al_tabpfn_output \
    --gpu-id 0

echo "=========================================="
echo " Done: $(date)"
echo "=========================================="
