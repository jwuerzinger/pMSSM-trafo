#!/bin/bash
# ==============================================================================
# Slurm job array: why is the AL-trained Deep GP's prediction spread in the
# posterior region larger than its random baseline's?
#
# Retrains the same architecture, cold and at matched size, on training sets
# that differ in one controlled way at a time (see the docstring of
# scripts/dispersion_experiment.py for the arms and what each isolates).
#
# 6 arms x 3 seeds = 18 tasks, one GPU each.
#
# Submit from repo root:
#   source slurm/cluster.conf
#   sbatch --partition="${CLUSTER_PARTITION}" --gres="${CLUSTER_GPU_GRES_1}" \
#          slurm/submit_dispersion_experiment.sh
# ==============================================================================
#SBATCH --job-name=dispersion_exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=110000
#SBATCH --time=08:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs"

if [[ -f "${REPO_ROOT}/slurm/cluster.conf" ]]; then
    source "${REPO_ROOT}/slurm/cluster.conf"
else
    echo "[warn] slurm/cluster.conf not found, using defaults"
fi

export PYTHONUNBUFFERED=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO_ROOT}/al_pmssmwithgp/model:${PYTHONPATH:-}"

ARMS=(al random stratified random_plus_inband al_ind1024 random_ind1024)
SEEDS=(1 2 3)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
ARM="${ARMS[$(( IDX / ${#SEEDS[@]} ))]}"
SEED="${SEEDS[$(( IDX % ${#SEEDS[@]} ))]}"

PIXI_ENV="${PIXI_ENV:-rocm}"
PYTHON="${REPO_ROOT}/.pixi/envs/${PIXI_ENV}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "[error] Python not found: ${PYTHON}" >&2
    exit 1
fi

echo "=========================================="
echo " Job:     ${SLURM_JOB_NAME} | ${SLURM_ARRAY_JOB_ID:-}_${IDX}"
echo " Arm:     ${ARM}   Seed: ${SEED}"
echo " Node:    $(hostname)"
echo " Started: $(date)"
echo " Python:  $(${PYTHON} --version 2>&1)"
echo "=========================================="

"${PYTHON}" scripts/dispersion_experiment.py \
    --arm "${ARM}" \
    --seed "${SEED}" \
    --output-dir /ptmp/jwuerzin/analysis/dispersion_experiment

echo "=========================================="
echo " Done: $(date)"
echo "=========================================="
