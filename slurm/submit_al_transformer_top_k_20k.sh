#!/bin/bash
# ==============================================================================
# Slurm job: Transformer AL — top_k selection, n_select=20000
#
# Equivalent to run_active_learning_top_k_20000.sh for Slurm.
#
# Submit from repo root (partition/gres come from cluster.conf):
#   sbatch $(slurm/cluster_flags.sh 2gpu) slurm/submit_al_transformer_top_k_20k.sh
#
# Or override on the command line:
#   sbatch --partition="${CLUSTER_PARTITION}" --gres="${CLUSTER_GPU_GRES_2}" slurm/submit_al_transformer_top_k_20k.sh
# ==============================================================================
#SBATCH --job-name=al_transformer_top_k_20k
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

# ---- Resolve repo root -------------------------------------------------------
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mv "${REPO_ROOT}/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "${REPO_ROOT}/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${TIMESTAMP}.out" 2>/dev/null || true
mv "${REPO_ROOT}/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "${REPO_ROOT}/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${TIMESTAMP}.err" 2>/dev/null || true


# ---- Cluster config ----------------------------------------------------------
if [[ -f "${REPO_ROOT}/slurm/cluster.conf" ]]; then
    source "${REPO_ROOT}/slurm/cluster.conf"
else
    echo "[warn] slurm/cluster.conf not found — using defaults"
    echo "       cp slurm/cluster.conf.template slurm/cluster.conf"
fi

export PYTHONUNBUFFERED=1
# SModelS results database. Run3ModelGen's build/setup.sh otherwise points
# SMODELS_CACHEDIR at the repo checkout (no /eos on this cluster), which holds no
# database, and compute nodes cannot download one. setup.sh honours a pre-set
# value. Only the r-value target's step list needs this; harmless otherwise.
export SMODELS_CACHEDIR="${SMODELS_CACHEDIR:-/ptmp/jwuerzin/cache/smodels}"
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
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

# ---- GPU count detection -----------------------------------------------------
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
echo "[gpu] SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<not set>}"
echo "[gpu] PIXI_ENV=${PIXI_ENV}"
echo "[gpu] Using --gpu-ids ${GPU_IDS} (${N_GPUS} GPU(s) detected)"

# ---- Run active learning -----------------------------------------------------
source "${REPO_ROOT}/slurm/resume_args.sh"

"${PYTHON}" active_learning.py \
    --y-transform log \
    --epochs 10000 \
    --generate-data \
    --n-samples 2000 \
    --n-iterations 40 \
    --n-candidates 1000000 \
    --entropy-pool-size 5000 \
    --gen-workers 20 \
    --mcmc-data-dir ${CLUSTER_DATA_DIR}/neutralino_v4 \
    --data-dir ${CLUSTER_DATA_DIR}/18387358 \
    --static-eval-size 100000 \
    --warm-starting \
    --early-stopping \
    --selection-strategy top_k \
    --n-select 20000 \
    --output-dir "${AL_OUTPUT_DIR:-/ptmp/jwuerzin/output/active_learning_output_top_k_n_select_20k_${TIMESTAMP}}" ${RESUME_ARGS} \
    --gpu-ids "${GPU_IDS}"

echo "=========================================="
echo " Done: $(date)"
echo "=========================================="
