#!/bin/bash
# ==============================================================================
# Slurm job: GP active learning — ExactGP
#
# Equivalent to run_active_learning_gp.sh with --model-type exact_gp.
#
# Setup: cp slurm/cluster.conf.template slurm/cluster.conf  (edit for your cluster)
#
# Submit from repo root:
#   source slurm/cluster.conf
#   sbatch --partition="${CLUSTER_PARTITION}" --gres="${CLUSTER_GPU_GRES_2}" slurm/submit_al_gp_exact.sh
# ==============================================================================
#SBATCH --job-name=al_gp_exact
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Partition and gres are set via sbatch flags from cluster.conf.
# See slurm/cluster.conf.template for details.

set -euo pipefail

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

if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" -gt 0 ]]; then
    N_GPUS="${SLURM_GPUS_ON_NODE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
    N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
elif [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    N_GPUS=$(echo "${ROCR_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
elif [[ -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
    N_GPUS=$(echo "${HIP_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
else
    N_GPUS=1
fi
GPU_IDS=$( [[ "${N_GPUS}" -ge 2 ]] && echo "0,1" || echo "0" )

echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "[gpu] ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<not set>}"
echo "[gpu] PIXI_ENV=${PIXI_ENV}"
echo "[gpu] Using --gpu-ids ${GPU_IDS} (${N_GPUS} GPU(s) detected)"

"${PYTHON}" active_learning_gp.py \
    --model-type exact_gp \
    --epochs 10000 \
    --early-stopping \
    --patience 100 \
    --learning-rate 1e-3 \
    --generate-data \
    --n-samples 2000 \
    --n-iterations 40 \
    --n-select 500 \
    --n-candidates 1000000 \
    --entropy-pool-size 5000 \
    --tolerance-sampling 1.0 \
    --gen-workers 20 \
    --mcmc-data-dir ${CLUSTER_DATA_DIR}/19250082 \
    --data-dir ${CLUSTER_DATA_DIR}/18387358 \
    --static-eval-size 100000 \
    --kernel RBF \
    --lengthscale 1.0 \
    --noise 1e-2 \
    --jitter 1e-3 \
    --use-ard \
    --warm-starting \
    --output-dir /ptmp/jwuerzin/active_learning_exact_gp_output \
    --gpu-ids "${GPU_IDS}"

echo "=========================================="
echo " Done: $(date)"
echo "=========================================="
