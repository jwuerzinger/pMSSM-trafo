#!/bin/bash
#SBATCH --job-name=lap_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
# Half-node memory: the cluster refuses 1 of 2 GPUs with more than half the
# node's RAM. 64G covers the 500k-row MCMC load and the pool cache.
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Stage 0 of the "more principled posterior" test: can the ALREADY TRAINED
# networks produce an acquisition uncertainty that orders their own errors
# better than MC dropout does? Post-hoc only, nothing is retrained, so this is
# a go/no-go gate on whether a full Laplace AL run is worth submitting.
#
# Submit with partition and gres from cluster.conf, as the per-model scripts do:
#   sbatch --partition="${CLUSTER_PARTITION}" --gres="${CLUSTER_GPU_GRES_1}" \
#          slurm/submit_laplace_probe.sh
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

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

"${PYTHON}" -u scripts/laplace_uq_probe.py \
    --models "${LAP_MODELS:-transformer,dnn,dnn_match_trafo}" \
    --output-dir "${LAP_OUT_DIR:-/ptmp/jwuerzin/analysis/all_runs}" \
    --eval-size "${LAP_EVAL_SIZE:-20000}" \
    --mc-samples "${LAP_MC_SAMPLES:-30}" \
    --knn-k "${LAP_KNN_K:-10}" \
    --max-seeds "${LAP_MAX_SEEDS:-0}" \
    --device cuda

echo "=========================================="
echo " Done: $(date)"
echo "=========================================="
