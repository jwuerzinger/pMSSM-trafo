#!/bin/bash
#SBATCH --job-name=cap_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
# Half-node memory: the cluster refuses 1 of 2 GPUs with more than half the
# node's RAM ("requested only 1 of two apus but more than 1/2 of memory").
# 64G is ample here, a 13k labelled set and a 500k-row eval subsample.
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Transformer capacity sweep: is the acquisition-uncertainty deficit a
# consequence of over-parametrisation? Trains four capacities on ONE fixed
# labelled set and scores each with evaluate_uq's own metrics, so rho is
# directly comparable with the paper's Table 4.
#
# Submit with partition and gres from cluster.conf, as the per-model scripts do:
#   sbatch --partition="${CLUSTER_PARTITION}" --gres="${CLUSTER_GPU_GRES_1}" \
#          slurm/submit_capacity_sweep.sh
set -uo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
PYTHON="${REPO_ROOT}/.pixi/envs/${PIXI_ENV:-rocm}/bin/python"

echo "=========================================="
echo " Job:     ${SLURM_JOB_NAME} | ID: ${SLURM_JOB_ID}"
echo " Node:    $(hostname)"
echo " Started: $(date)"
echo "=========================================="
"${PYTHON}" -c "import torch;print('[gpu] available:',torch.cuda.is_available(),
      'count:',torch.cuda.device_count())" || true

# One BLAS thread per process: the work is GPU-bound and oversubscribing threads
# on a shared node buys nothing.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

"${PYTHON}" -u scripts/transformer_capacity_sweep.py \
    --run-dir "${CAP_RUN_DIR:?CAP_RUN_DIR must be set}" \
    --output-dir "${CAP_OUT_DIR:-/ptmp/jwuerzin/analysis/all_runs}" \
    --eval-size "${CAP_EVAL_SIZE:-20000}" \
    --mc-samples "${CAP_MC_SAMPLES:-30}" \
    --epochs "${CAP_EPOCHS:-10000}" \
    --patience "${CAP_PATIENCE:-200}" \
    --device cuda

echo "=========================================="
echo " Done: $(date)"
echo "=========================================="
