#!/bin/bash
# ==============================================================================
# Slurm job: Transformer-based active learning (PMSSMTransformerTabular + MC Dropout)
#
# Defaults mirror run_active_learning.sh with top_k selection.
#
# Submit from repo root:
#   sbatch slurm/submit_al_transformer.sh
#
# Override partition/account on the command line (takes precedence over #SBATCH):
#   sbatch --partition=rvs --account=mpp slurm/submit_al_transformer.sh
#
# Customize via environment variables (export or --export=ALL,VAR=val):
#   AL_N_SAMPLES          Initial dataset size           (default: 2000)
#   AL_N_ITERATIONS       Number of AL iterations        (default: 40)
#   AL_N_SELECT           Points selected per iteration  (default: 500)
#   AL_N_CANDIDATES       Candidate pool size            (default: 1000000)
#   AL_EPOCHS             Training epochs per iteration  (default: 10000)
#   AL_GEN_WORKERS        Parallel generation workers    (default: 20)
#   AL_PATIENCE           Early stopping patience        (default: 200)
#   AL_ENTROPY_POOL_SIZE  Focused pool for entropy sel.  (default: 5000)
#   AL_TOLERANCE_SAMPLING Hard candidate cut width       (default: 1.0)
#   AL_MCMC_DATA_DIR      MCMC evaluation dataset dir    (default: /ptmp/jwuerzin/data)
#   AL_STATIC_EVAL_SIZE   Static eval set size           (default: 100000)
#   AL_OUTPUT_DIR         Output directory               (default: active_learning_output_slurm)
#   AL_GENERATE_DATA      Set empty to disable physics simulation (default: --generate-data)
#   AL_EXTRA_ARGS         Any additional flags passed verbatim to the script
#
# Examples:
#   # Standard production run (with data generation):
#   sbatch slurm/submit_al_transformer.sh
#
#   # Without data generation (ML pipeline only):
#   sbatch --export=ALL,AL_GENERATE_DATA= slurm/submit_al_transformer.sh
#
#   # Custom output directory:
#   sbatch --export=ALL,AL_OUTPUT_DIR=my_run slurm/submit_al_transformer.sh
# ==============================================================================
#SBATCH --job-name=al_transformer_top_k
#SBATCH --partition=gpu
#SBATCH --account=mpp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/raven/u/jwuerzin/pMSSM-trafo/logs/%x_%j.out
#SBATCH --error=/raven/u/jwuerzin/pMSSM-trafo/logs/%x_%j.err

set -euo pipefail

# ---- Resolve repo root -------------------------------------------------------
REPO_ROOT="${SLURM_SUBMIT_DIR:-/raven/u/jwuerzin/pMSSM-trafo}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs"

# Flush Python stdout immediately so logs are complete even if the job is killed
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

# ---- GPU count detection -----------------------------------------------------
# Slurm sets CUDA_VISIBLE_DEVICES to allocated GPU indices (always 0-based).
# Use 0,1 for 2 GPUs (AL model on cuda:0, baseline on cuda:1 in parallel).
if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" -gt 0 ]]; then
    N_GPUS="${SLURM_GPUS_ON_NODE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
    N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
else
    N_GPUS=1
fi
GPU_IDS=$( [[ "${N_GPUS}" -ge 2 ]] && echo "0,1" || echo "0" )

echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "[gpu] SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<not set>}"
echo "[gpu] Using --gpu-ids ${GPU_IDS} (${N_GPUS} GPU(s) detected)"

# ---- Parameters (all overridable via environment variables) ------------------
echo "[params] AL_N_SAMPLES=${AL_N_SAMPLES:-2000}"
echo "[params] AL_N_ITERATIONS=${AL_N_ITERATIONS:-40}"
echo "[params] AL_N_SELECT=${AL_N_SELECT:-500}"
echo "[params] AL_N_CANDIDATES=${AL_N_CANDIDATES:-1000000}"
echo "[params] AL_EPOCHS=${AL_EPOCHS:-10000}"
echo "[params] AL_GEN_WORKERS=${AL_GEN_WORKERS:-20}"
echo "[params] AL_OUTPUT_DIR=${AL_OUTPUT_DIR:-/ptmp/jwuerzin/active_learning_output_top_k_slurm}"
echo "[params] AL_GENERATE_DATA=${AL_GENERATE_DATA:---generate-data}"

# ---- Run active learning -----------------------------------------------------
"${PYTHON}" active_learning.py \
    --y-transform log \
    --n-samples "${AL_N_SAMPLES:-2000}" \
    --n-iterations "${AL_N_ITERATIONS:-40}" \
    --n-select "${AL_N_SELECT:-500}" \
    --n-candidates "${AL_N_CANDIDATES:-1000000}" \
    --epochs "${AL_EPOCHS:-10000}" \
    --entropy-pool-size "${AL_ENTROPY_POOL_SIZE:-5000}" \
    --tolerance-sampling "${AL_TOLERANCE_SAMPLING:-1.0}" \
    --mcmc-data-dir "${AL_MCMC_DATA_DIR:-/ptmp/jwuerzin/data}" \
    --data-dir "${AL_DATA_DIR:-/ptmp/jwuerzin/data}" \
    --static-eval-size "${AL_STATIC_EVAL_SIZE:-100000}" \
    --gen-workers "${AL_GEN_WORKERS:-20}" \
    --output-dir "${AL_OUTPUT_DIR:-/ptmp/jwuerzin/active_learning_output_top_k_slurm}" \
    --early-stopping \
    --patience "${AL_PATIENCE:-200}" \
    --warm-starting \
    --selection-strategy top_k \
    --gpu-ids "${GPU_IDS}" \
    ${AL_GENERATE_DATA:---generate-data} \
    ${AL_EXTRA_ARGS:-}

echo "=========================================="
echo " Done: $(date)"
echo "=========================================="
