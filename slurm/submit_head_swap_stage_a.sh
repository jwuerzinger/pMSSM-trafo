#!/bin/bash
# ==============================================================================
# Slurm job: Stage A of the classification-versus-regression test.
#
# Retrains both acquisition heads (pmssm.heads: regression, classification) on
# labelled sets the existing AL runs already acquired, and scores the labelled
# random pool with each head's own acquisition rule. No simulator calls, so this
# decides whether a full AL ablation is worth running before any is launched.
#
# Queue choice: apu1, not apu. apu is OverSubscribe=EXCLUSIVE (a one-GPU request
# blocks a whole node and queues behind every large job); apu1 is MaxNodes=1 with
# OverSubscribe=NO, so a single-GCD job shares a node and backfills. Walltime is
# kept short for the same reason (bf_window=1500 min).
#
# Submit from repo root, e.g.:
#   TAG=dnn_i40 AL_MODELS=dnn_expr AL_ITERATIONS=40 \
#     sbatch --time=04:00:00 slurm/submit_head_swap_stage_a.sh
#
# Every parameter is an env override; the defaults run the decisive cell
# (DNN, ExpR, iteration 40, five seeds, three inits, both heads, one-step on).
# Results are checkpointed to the output JSON after every snapshot, so a
# walltime kill still leaves whatever completed.
# ==============================================================================
#SBATCH --job-name=head_stage_a
#SBATCH --partition=apu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs"

# ---- Parameters (all overridable via --export) --------------------------------
TAG="${TAG:-dnn_i40}"
AL_TARGET="${AL_TARGET:-ExpR}"
AL_MANIFEST="${AL_MANIFEST:-/ptmp/jwuerzin/analysis/joint/manifest_expr.csv}"
AL_MODELS="${AL_MODELS:-dnn_expr}"
AL_SEEDS="${AL_SEEDS:-1,2,3,4,5}"
AL_ITERATIONS="${AL_ITERATIONS:-40}"
AL_HEADS="${AL_HEADS:-regression,classification}"
AL_N_INITS="${AL_N_INITS:-3}"
AL_EVAL_SIZE="${AL_EVAL_SIZE:-100000}"
AL_SCORE_POOL="${AL_SCORE_POOL:-500000}"
AL_ONE_STEP="${AL_ONE_STEP:-1}"
AL_EXTRA_ARGS="${AL_EXTRA_ARGS:-}"
OUT_DIR="${OUT_DIR:-/ptmp/jwuerzin/analysis/head_swap}"
OUT_JSON="${OUT_JSON:-${OUT_DIR}/stage_a_${TAG}.json}"

export PYTHONUNBUFFERED=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO_ROOT}/al_pmssmwithgp/model:${PYTHONPATH:-}"
# The login/compute nodes hand out 128 OpenBLAS threads by default, which fails
# to spawn under the process limits; the work here is torch-side anyway.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=========================================="
echo " Job:     ${SLURM_JOB_NAME} | ID: ${SLURM_JOB_ID}"
echo " Node:    $(hostname)"
echo " Started: $(date)"
echo " Tag:     ${TAG}"
echo " Target:  ${AL_TARGET} | models ${AL_MODELS} | seeds ${AL_SEEDS}"
echo " Iters:   ${AL_ITERATIONS} | heads ${AL_HEADS} | inits ${AL_N_INITS}"
echo " Output:  ${OUT_JSON}"
echo "=========================================="

PIXI_ENV="${PIXI_ENV:-rocm}"
PYTHON="${REPO_ROOT}/.pixi/envs/${PIXI_ENV}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "[setup] pixi env '${PIXI_ENV}' not found — running: pixi install -e ${PIXI_ENV}"
    /u/jwuerzin/.pixi/bin/pixi install -e "${PIXI_ENV}"
fi
echo "[env] $(${PYTHON} --version)"

if [[ -n "${ROCR_VISIBLE_DEVICES:-}${HIP_VISIBLE_DEVICES:-}${CUDA_VISIBLE_DEVICES:-}" ]]; then
    DEVICE="cuda:0"
else
    DEVICE="cpu"
    echo "[warn] no GPU visible; falling back to CPU (this will be slow)"
fi
echo "[gpu] ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>} device=${DEVICE}"

ONE_STEP_FLAG="--one-step"
[[ "${AL_ONE_STEP}" == "1" ]] || ONE_STEP_FLAG="--no-one-step"

mkdir -p "${OUT_DIR}"

set -x
"${PYTHON}" scripts/head_swap_stage_a.py \
    --target "${AL_TARGET}" \
    --manifest "${AL_MANIFEST}" \
    --models "${AL_MODELS}" \
    --seeds "${AL_SEEDS}" \
    --iterations "${AL_ITERATIONS}" \
    --heads "${AL_HEADS}" \
    --n-inits "${AL_N_INITS}" \
    --eval-size "${AL_EVAL_SIZE}" \
    --score-pool-size "${AL_SCORE_POOL}" \
    ${ONE_STEP_FLAG} \
    --device "${DEVICE}" \
    --output "${OUT_JSON}" \
    ${AL_EXTRA_ARGS}
set +x

echo "=========================================="
echo " Finished: $(date)"
echo " Results:  ${OUT_JSON}"
echo "=========================================="
