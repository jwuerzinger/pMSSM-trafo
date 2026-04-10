#!/bin/bash
# ==============================================================================
# Slurm array job: hyperparameter sweep for any active learning script
#
# Each array task runs one combination from a YAML sweep config.
# The Python scripts handle sweep logic via --config-file + --sweep-index.
#
# Usage (submit from repo root):
#   sbatch --array=0-8 slurm/submit_sweep.sh <config.yaml>
#
# The config file path is the first positional argument ($1). It can be
# relative (resolved from repo root) or absolute. Example configs:
#   test_config.yaml   — 8 combinations (epochs × dropout × strategy)
#
# Customize via environment variables:
#   AL_SWEEP_SCRIPT   Script to run (default: active_learning.py)
#                       options: active_learning.py, active_learning_gp.py, active_learning_tabpfn.py
#   AL_OUTPUT_DIR     Base output dir (default: sweep_output)
#                       each task writes to <AL_OUTPUT_DIR>/task_<TASK_ID>/
#   AL_EXTRA_ARGS     Any additional flags passed verbatim to the script
#
# Examples:
#   # Transformer sweep (8 combinations in test_config.yaml, indices 0-7):
#   sbatch --array=0-7 slurm/submit_sweep.sh test_config.yaml
#
#   # GP sweep with custom config:
#   sbatch --array=0-5 \
#       --export=ALL,AL_SWEEP_SCRIPT=active_learning_gp.py,AL_OUTPUT_DIR=gp_sweep \
#       slurm/submit_sweep.sh my_gp_sweep.yaml
#
#   # TabPFN sweep:
#   sbatch --array=0-3 \
#       --export=ALL,AL_SWEEP_SCRIPT=active_learning_tabpfn.py \
#       slurm/submit_sweep.sh tabpfn_sweep.yaml
#
# Note: The sweep script handles all model-specific flags via the YAML config.
#       Only --config-file, --sweep-index, --output-dir, and --gpu-ids are
#       injected here. Put everything else in the YAML.
# ==============================================================================
#SBATCH --job-name=al_sweep
#SBATCH --partition=gpu
#SBATCH --account=mpp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/raven/u/jwuerzin/pMSSM-trafo/logs/%x_%A_%a.out
#SBATCH --error=/raven/u/jwuerzin/pMSSM-trafo/logs/%x_%A_%a.err

set -euo pipefail

# ---- Resolve repo root -------------------------------------------------------
REPO_ROOT="${SLURM_SUBMIT_DIR:-/raven/u/jwuerzin/pMSSM-trafo}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/al_pmssmwithgp/model:${PYTHONPATH:-}"

echo "=========================================="
echo " Job:      ${SLURM_JOB_NAME} | Array: ${SLURM_ARRAY_JOB_ID} | Task: ${SLURM_ARRAY_TASK_ID}"
echo " Node:     $(hostname)"
echo " Started:  $(date)"
echo " Repo:     ${REPO_ROOT}"
echo "=========================================="

# ---- Config file (positional arg or AL_CONFIG_FILE env var) ------------------
CONFIG_ARG="${1:-${AL_CONFIG_FILE:-}}"
if [[ -z "${CONFIG_ARG}" ]]; then
    echo "[error] No config file specified."
    echo "[error] Usage: sbatch --array=0-N slurm/submit_sweep.sh <config.yaml>"
    exit 1
fi

# Resolve to absolute path (try as-is, then relative to repo root)
if [[ -f "${CONFIG_ARG}" ]]; then
    CONFIG_FILE="$(realpath "${CONFIG_ARG}")"
elif [[ -f "${REPO_ROOT}/${CONFIG_ARG}" ]]; then
    CONFIG_FILE="${REPO_ROOT}/${CONFIG_ARG}"
else
    echo "[error] Config file not found: ${CONFIG_ARG}"
    echo "[error] Tried: ${CONFIG_ARG} and ${REPO_ROOT}/${CONFIG_ARG}"
    exit 1
fi
echo "[config] Config file: ${CONFIG_FILE}"
echo "[config] Sweep index: ${SLURM_ARRAY_TASK_ID}"

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

echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "[gpu] Using --gpu-ids 0 (single GPU per array task)"

# ---- Output directory (one subdirectory per task) ----------------------------
SWEEP_SCRIPT="${AL_SWEEP_SCRIPT:-active_learning.py}"
BASE_OUTPUT_DIR="${AL_OUTPUT_DIR:-/ptmp/jwuerzin/sweep_output}"
TASK_OUTPUT_DIR="${BASE_OUTPUT_DIR}/task_${SLURM_ARRAY_TASK_ID}"
AL_DATA_DIR="${AL_DATA_DIR:-/ptmp/jwuerzin/data}"

# TabPFN uses --gpu-id (singular); the other two use --gpu-ids
if [[ "${SWEEP_SCRIPT}" == *tabpfn* ]]; then
    GPU_FLAG="--gpu-id"
else
    GPU_FLAG="--gpu-ids"
fi

echo "[params] Script:     ${SWEEP_SCRIPT}"
echo "[params] Output dir: ${TASK_OUTPUT_DIR}"
echo "[params] Data dir:   ${AL_DATA_DIR}"

# ---- Run sweep task ----------------------------------------------------------
"${PYTHON}" "${SWEEP_SCRIPT}" \
    --config-file "${CONFIG_FILE}" \
    --sweep-index "${SLURM_ARRAY_TASK_ID}" \
    --output-dir "${TASK_OUTPUT_DIR}" \
    --data-dir "${AL_DATA_DIR}" \
    ${GPU_FLAG} 0 \
    ${AL_EXTRA_ARGS:-}

echo "=========================================="
echo " Done: $(date)"
echo "=========================================="
