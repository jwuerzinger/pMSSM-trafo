#!/bin/bash
# ==============================================================================
# Slurm job: GP-based active learning (exact_gp / deep_gp / sparse_gp / mlp)
#
# Defaults mirror run_active_learning_gp.sh (full production run).
#
# Submit from repo root:
#   sbatch slurm/submit_al_gp.sh
#
# Override partition/account on the command line (takes precedence over #SBATCH):
#   sbatch --partition=rvs --account=mpp slurm/submit_al_gp.sh
#
# Customize via environment variables (export or --export=ALL,VAR=val):
#   AL_MODEL_TYPE         GP model type                  (default: exact_gp)
#                           options: exact_gp, deep_gp, sparse_gp, mlp
#   AL_KERNEL             GP kernel                      (default: RBF)
#                           options: RBF, Matern, RQK, SpectralMixture
#   AL_LEARNING_RATE      Optimizer learning rate        (default: 1e-3)
#   AL_LENGTHSCALE        Initial GP lengthscale         (default: 1.0)
#   AL_NOISE              Initial GP noise level         (default: 1e-2)
#   AL_JITTER             Cholesky jitter                (default: 1e-3)
#   AL_USE_ARD            ARD flag (default: --use-ard; set empty to disable)
#   AL_N_SAMPLES          Initial dataset size           (default: 2000)
#   AL_N_ITERATIONS       Number of AL iterations        (default: 40)
#   AL_N_SELECT           Points selected per iteration  (default: 500)
#   AL_N_CANDIDATES       Candidate pool size            (default: 1000000)
#   AL_EPOCHS             Training epochs per iteration  (default: 10000)
#   AL_GEN_WORKERS        Parallel generation workers    (default: 20)
#   AL_PATIENCE           Early stopping patience        (default: 100)
#   AL_ENTROPY_POOL_SIZE  Focused pool for entropy sel.  (default: 5000)
#   AL_TOLERANCE_SAMPLING Hard candidate cut width       (default: 1.0)
#   AL_MCMC_DATA_DIR      MCMC evaluation dataset dir    (default: /ptmp/jwuerzin/data)
#   AL_OUTPUT_DIR         Output directory               (default: active_learning_exact_gp_output_slurm)
#   AL_GENERATE_DATA      Set empty to disable physics simulation (default: --generate-data)
#   AL_EXTRA_ARGS         Any additional flags passed verbatim to the script
#
# Examples:
#   # Standard production run (exact_gp with data generation):
#   sbatch slurm/submit_al_gp.sh
#
#   # DeepGP, no data generation:
#   sbatch --export=ALL,AL_MODEL_TYPE=deep_gp,AL_GENERATE_DATA=,AL_OUTPUT_DIR=deep_gp_run \
#       slurm/submit_al_gp.sh
#
#   # Disable ARD:
#   sbatch --export=ALL,AL_USE_ARD= slurm/submit_al_gp.sh
# ==============================================================================
#SBATCH --job-name=al_gp
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
echo "[params] AL_MODEL_TYPE=${AL_MODEL_TYPE:-exact_gp}"
echo "[params] AL_N_SAMPLES=${AL_N_SAMPLES:-2000}"
echo "[params] AL_N_ITERATIONS=${AL_N_ITERATIONS:-40}"
echo "[params] AL_N_SELECT=${AL_N_SELECT:-500}"
echo "[params] AL_N_CANDIDATES=${AL_N_CANDIDATES:-1000000}"
echo "[params] AL_EPOCHS=${AL_EPOCHS:-10000}"
echo "[params] AL_GEN_WORKERS=${AL_GEN_WORKERS:-20}"
echo "[params] AL_OUTPUT_DIR=${AL_OUTPUT_DIR:-/ptmp/jwuerzin/active_learning_exact_gp_output_slurm}"
echo "[params] AL_GENERATE_DATA=${AL_GENERATE_DATA:---generate-data}"

# ---- Run active learning -----------------------------------------------------
# AL_USE_ARD: defaults to --use-ard; set AL_USE_ARD= (empty) to disable ARD
"${PYTHON}" active_learning_gp.py \
    --model-type "${AL_MODEL_TYPE:-exact_gp}" \
    --n-samples "${AL_N_SAMPLES:-2000}" \
    --n-iterations "${AL_N_ITERATIONS:-40}" \
    --n-select "${AL_N_SELECT:-500}" \
    --n-candidates "${AL_N_CANDIDATES:-1000000}" \
    --epochs "${AL_EPOCHS:-10000}" \
    --early-stopping \
    --patience "${AL_PATIENCE:-100}" \
    --learning-rate "${AL_LEARNING_RATE:-1e-3}" \
    --kernel "${AL_KERNEL:-RBF}" \
    --lengthscale "${AL_LENGTHSCALE:-1.0}" \
    --noise "${AL_NOISE:-1e-2}" \
    --jitter "${AL_JITTER:-1e-3}" \
    ${AL_USE_ARD:---use-ard} \
    --entropy-pool-size "${AL_ENTROPY_POOL_SIZE:-5000}" \
    --tolerance-sampling "${AL_TOLERANCE_SAMPLING:-1.0}" \
    --mcmc-data-dir "${AL_MCMC_DATA_DIR:-/ptmp/jwuerzin/data}" \
    --data-dir "${AL_DATA_DIR:-/ptmp/jwuerzin/data}" \
    --gen-workers "${AL_GEN_WORKERS:-20}" \
    --output-dir "${AL_OUTPUT_DIR:-/ptmp/jwuerzin/active_learning_exact_gp_output_slurm}" \
    --warm-starting \
    --gpu-ids "${GPU_IDS}" \
    ${AL_GENERATE_DATA:---generate-data} \
    ${AL_EXTRA_ARGS:-}

echo "=========================================="
echo " Done: $(date)"
echo "=========================================="
