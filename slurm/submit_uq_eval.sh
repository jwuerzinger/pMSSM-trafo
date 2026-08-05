#!/bin/bash
# ==============================================================================
# Slurm job: post-hoc uncertainty-quantification evaluation (scripts/evaluate_uq.py).
#
# For each best-per-model pick this reloads per-iteration AL checkpoints,
# evaluates the full predictive distribution on seeded static-random and MCMC
# eval sets, and scores calibration (z-stats, PIT coverage), proper scores
# (NLPD, CRPS), ranking quality (Spearman, AUSE) and sharpness trajectories.
# Results are cached at <run_dir>/uq_eval_cache.json; outputs
# (uq_evaluation.json + uq_*.png) land in the analysis directory.
#
# Submit from repo root:
#   sbatch --partition=$(. slurm/cluster.conf; echo ${CLUSTER_PARTITION}) \
#          --gres=gpu:1 slurm/submit_uq_eval.sh
#
# Env knobs (all optional):
#   REFRESH=1          ignore per-run caches
#   VETO=1             --require-neutralino-lsp
#   MODELS=...         comma list of picks (default: all)
#   MC_SAMPLES=30,100  dropout-pass ablation
#   OUTPUT_DIR=...     override analysis output dir
# ==============================================================================
#SBATCH --job-name=uq_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Partition and gres are set via sbatch flags from cluster.conf.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs"

# ---- Cluster config ----------------------------------------------------------
if [[ -f "${REPO_ROOT}/slurm/cluster.conf" ]]; then
    source "${REPO_ROOT}/slurm/cluster.conf"
else
    echo "[warn] slurm/cluster.conf not found — using defaults"
fi

export PYTHONUNBUFFERED=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO_ROOT}/al_pmssmwithgp/model:${PYTHONPATH:-}"

echo "=========================================="
echo " Job:     ${SLURM_JOB_NAME} | ID: ${SLURM_JOB_ID}"
echo " Node:    $(hostname)"
echo " Started: $(date)"
echo " Repo:    ${REPO_ROOT}"
echo "=========================================="

PIXI_ENV="${PIXI_ENV:-rocm}"
PYTHON="${REPO_ROOT}/.pixi/envs/${PIXI_ENV}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "[setup] pixi env '${PIXI_ENV}' not found — running: pixi install -e ${PIXI_ENV}"
    /u/jwuerzin/.pixi/bin/pixi install -e "${PIXI_ENV}"
fi

EXTRA_FLAGS=()
if [[ "${REFRESH:-0}" == "1" ]]; then
    EXTRA_FLAGS+=(--refresh)
    echo "[opt] cache refresh: ON"
fi
if [[ "${VETO:-0}" == "1" ]]; then
    EXTRA_FLAGS+=(--require-neutralino-lsp)
    echo "[opt] neutralino-LSP veto: ON"
fi
if [[ -n "${MODELS:-}" ]]; then
    EXTRA_FLAGS+=(--models "${MODELS}")
    echo "[opt] models: ${MODELS}"
fi
if [[ -n "${INCLUDE_STATUS:-}" ]]; then
    EXTRA_FLAGS+=(--include-status "${INCLUDE_STATUS}")
    echo "[opt] include-status: ${INCLUDE_STATUS}"
fi
if [[ -n "${MC_SAMPLES:-}" ]]; then
    EXTRA_FLAGS+=(--mc-samples "${MC_SAMPLES}")
    echo "[opt] mc-samples: ${MC_SAMPLES}"
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
    EXTRA_FLAGS+=(--output-dir "${OUTPUT_DIR}")
    echo "[opt] output dir: ${OUTPUT_DIR}"
fi

DATA_DIR="${CLUSTER_DATA_DIR:-/ptmp/jwuerzin/data}"

"${PYTHON}" scripts/evaluate_uq.py \
    --baseline-data-dir "${DATA_DIR}/18387358" \
    --mcmc-data-dir "${DATA_DIR}/neutralino_v4" \
    "${EXTRA_FLAGS[@]}"

echo "=========================================="
echo " Done: $(date)"
echo "=========================================="
