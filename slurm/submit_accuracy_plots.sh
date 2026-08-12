#!/bin/bash
# ==============================================================================
# Slurm job: render classification-accuracy plots alongside the existing
# best-per-model hit-rate plots.
#
# This walks each best-per-model pick's seed runs, loads every iteration's AL
# and baseline checkpoints, evaluates accuracy on (static_random, MCMC, train,
# val), and writes four PNGs into the analysis directory. Results are cached
# at <run_dir>/accuracy_trajectory.json so re-runs are near-instant.
#
# Submit from repo root:
#   sbatch slurm/submit_accuracy_plots.sh
#
# To force re-evaluation (e.g. after a code change in the predict path):
#   sbatch --export=ALL,REFRESH=1 slurm/submit_accuracy_plots.sh
#
# Note: TabPFN is intentionally not evaluated here -- AL TabPFN runs save no
# per-iteration weight file, so reproducing AL-time predictions would require
# re-running the AL pipeline. The plotter skips TabPFN picks.
# ==============================================================================
#SBATCH --job-name=acc_plots
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Partition and gres are set via sbatch flags from cluster.conf.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mv "${REPO_ROOT}/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
   "${REPO_ROOT}/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${TIMESTAMP}.out" 2>/dev/null || true
mv "${REPO_ROOT}/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" \
   "${REPO_ROOT}/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${TIMESTAMP}.err" 2>/dev/null || true

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
echo "[env] $(${PYTHON} --version)"

# Pick a single visible GPU index (0 if anything was allocated, else fall back
# to cpu — the script's --accuracy-device defaulting handles cuda-vs-cpu).
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" || -n "${HIP_VISIBLE_DEVICES:-}" \
      || -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    ACC_DEVICE="cuda:0"
else
    ACC_DEVICE="cpu"
fi
echo "[gpu] ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<not set>}"
echo "[gpu] HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-<not set>}"
echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "[gpu] accuracy device: ${ACC_DEVICE}"

# Optional flags via sbatch --export
EXTRA_FLAGS=()
if [[ "${REFRESH:-0}" == "1" ]]; then
    EXTRA_FLAGS+=(--accuracy-cache-refresh)
    echo "[opt] cache refresh: ON"
fi
if [[ -n "${SWEEP_ID:-}" ]]; then
    EXTRA_FLAGS+=(--sweep-id "${SWEEP_ID}")
    echo "[opt] sweep-id: ${SWEEP_ID}"
fi
if [[ "${VETO:-0}" == "1" ]]; then
    EXTRA_FLAGS+=(--require-neutralino-lsp)
    echo "[opt] neutralino-LSP veto: ON"
fi
# The per-run accuracy_trajectory.json caches carry no fingerprint of the
# reference set, so they go stale invisibly whenever the eval data or a run's
# iterations change. REFRESH=1 forces re-evaluation.
if [[ "${REFRESH:-0}" == "1" ]]; then
    EXTRA_FLAGS+=(--accuracy-cache-refresh)
    echo "[opt] accuracy cache: FORCED REFRESH"
fi
# A custom manifest lets this wrapper drive a focused comparison (e.g. the
# Laplace cells against their MC-dropout counterparts) without touching the
# main sweep's figures.
if [[ -n "${MANIFEST:-}" ]]; then
    EXTRA_FLAGS+=(--manifest "${MANIFEST}")
    echo "[opt] manifest: ${MANIFEST}"
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
    EXTRA_FLAGS+=(--output-dir "${OUTPUT_DIR}")
    echo "[opt] output dir: ${OUTPUT_DIR}"
fi

DATA_DIR="${CLUSTER_DATA_DIR:-/ptmp/jwuerzin/data}"

"${PYTHON}" scripts/plot_hit_rate_trajectories_multiseed.py \
    --compute-accuracy \
    --accuracy-device "${ACC_DEVICE}" \
    --baseline-data-dir "${DATA_DIR}/18387358" \
    --mcmc-data-dir "${DATA_DIR}/neutralino_v4" \
    "${EXTRA_FLAGS[@]}"

echo "=========================================="
echo " Done: $(date)"
echo "=========================================="
