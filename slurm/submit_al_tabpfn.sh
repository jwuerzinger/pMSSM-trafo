#!/bin/bash
# ==============================================================================
# Slurm job: TabPFN-based active learning
#
# Defaults mirror run_active_learning_tabpfn.sh (full production run).
#
# TabPFN is a single-GPU model (ensemble members run sequentially on one device).
# This script requests 1 GPU. Pass --gpu-id 0 — Slurm always allocates the
# device as cuda:0 inside the job via CUDA_VISIBLE_DEVICES.
#
# Submit from repo root (partition/gres come from cluster.conf):
#   sbatch $(slurm/cluster_flags.sh 1gpu) slurm/submit_al_tabpfn.sh
#
# Or override on the command line:
#   sbatch --partition="${CLUSTER_PARTITION}" --gres="${CLUSTER_GPU_GRES_1}" slurm/submit_al_tabpfn.sh
#
# Customize via environment variables (export or --export=ALL,VAR=val):
#   AL_N_SAMPLES              Initial dataset size               (default: 2000)
#   AL_N_ITERATIONS           Number of AL iterations            (default: 40)
#   AL_N_SELECT               Points selected per iteration      (default: 500)
#   AL_N_CANDIDATES           Candidate pool size                (default: 1000000)
#   AL_N_ENSEMBLE_SAMPLES     TabPFN ensemble runs per point     (default: 16)
#   AL_SELECTION_STRATEGY     top_k or entropy_batch             (default: top_k)
#   AL_PROXIMITY_SAMPLING     Proximity weighting width          (default: 0.1)
#   AL_TOLERANCE_SAMPLING     Hard candidate cut width           (default: 1.0)
#   AL_GEN_WORKERS            Parallel generation workers        (default: 20)
#   AL_MCMC_DATA_DIR          MCMC evaluation dataset dir        (default: ${CLUSTER_DATA_DIR}/19250082)
#   AL_STATIC_EVAL_SIZE       Static eval set size               (default: 100000)
#   AL_OUTPUT_DIR             Output directory                   (default: active_learning_tabpfn_output_slurm)
#   AL_GENERATE_DATA          Set empty to disable physics simulation (default: --generate-data)
#   AL_EXTRA_ARGS             Any additional flags passed verbatim to the script
#
# Note: TabPFN is a pre-trained model; --warm-starting and --early-stopping
#       are not applicable and are not passed here.
#
# Examples:
#   # Standard production run (with data generation):
#   sbatch slurm/submit_al_tabpfn.sh
#
#   # Without data generation (ML pipeline only):
#   sbatch --export=ALL,AL_GENERATE_DATA= slurm/submit_al_tabpfn.sh
#
#   # Entropy batch selection:
#   sbatch --export=ALL,AL_SELECTION_STRATEGY=entropy_batch slurm/submit_al_tabpfn.sh
# ==============================================================================
#SBATCH --job-name=al_tabpfn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
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

# ---- Cluster config ----------------------------------------------------------
if [[ -f "${REPO_ROOT}/slurm/cluster.conf" ]]; then
    source "${REPO_ROOT}/slurm/cluster.conf"
else
    echo "[warn] slurm/cluster.conf not found — using defaults"
    echo "       cp slurm/cluster.conf.template slurm/cluster.conf"
fi

# TabPFN requires a license token to download model weights non-interactively.
# Set it via: export TABPFN_TOKEN="<key>"  (get key at https://ux.priorlabs.ai/account)
if [[ -z "${TABPFN_TOKEN:-}" ]]; then
    echo "[error] TABPFN_TOKEN is not set. Cannot run TabPFN without a license token."
    echo "        Submit via submit_slurm.sh (prompts automatically), or:"
    echo "        export TABPFN_TOKEN='<your-api-key>' && bash submit_slurm.sh"
    exit 1
fi
export TABPFN_TOKEN

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

echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "[gpu] ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<not set>}"
echo "[gpu] PIXI_ENV=${PIXI_ENV}"
echo "[gpu] Using --gpu-id 0 (single GPU, Slurm-remapped)"

# ---- Parameters (all overridable via environment variables) ------------------
echo "[params] AL_N_SAMPLES=${AL_N_SAMPLES:-2000}"
echo "[params] AL_N_ITERATIONS=${AL_N_ITERATIONS:-40}"
echo "[params] AL_N_SELECT=${AL_N_SELECT:-500}"
echo "[params] AL_N_CANDIDATES=${AL_N_CANDIDATES:-1000000}"
echo "[params] AL_N_ENSEMBLE_SAMPLES=${AL_N_ENSEMBLE_SAMPLES:-16}"
echo "[params] AL_SELECTION_STRATEGY=${AL_SELECTION_STRATEGY:-top_k}"
echo "[params] AL_OUTPUT_DIR=${AL_OUTPUT_DIR:-/ptmp/jwuerzin/output/active_learning_tabpfn_output_slurm}"
echo "[params] AL_GENERATE_DATA=${AL_GENERATE_DATA:---generate-data}"

# ---- Run active learning -----------------------------------------------------
"${PYTHON}" active_learning_tabpfn.py \
    --n-samples "${AL_N_SAMPLES:-2000}" \
    --n-iterations "${AL_N_ITERATIONS:-40}" \
    --n-select "${AL_N_SELECT:-500}" \
    --n-candidates "${AL_N_CANDIDATES:-1000000}" \
    --n-ensemble-samples "${AL_N_ENSEMBLE_SAMPLES:-16}" \
    --selection-strategy "${AL_SELECTION_STRATEGY:-top_k}" \
    --proximity-sampling "${AL_PROXIMITY_SAMPLING:-0.1}" \
    --tolerance-sampling "${AL_TOLERANCE_SAMPLING:-1.0}" \
    --mcmc-data-dir "${AL_MCMC_DATA_DIR:-${CLUSTER_DATA_DIR}/19250082}" \
    --data-dir "${AL_DATA_DIR:-${CLUSTER_DATA_DIR}/18387358}" \
    --static-eval-size "${AL_STATIC_EVAL_SIZE:-100000}" \
    --gen-workers "${AL_GEN_WORKERS:-20}" \
    --output-dir "${AL_OUTPUT_DIR:-/ptmp/jwuerzin/output/active_learning_tabpfn_output_slurm}" \
    --gpu-id 0 \
    ${AL_GENERATE_DATA:---generate-data} \
    ${AL_EXTRA_ARGS:-}

echo "=========================================="
echo " Done: $(date)"
echo "=========================================="
