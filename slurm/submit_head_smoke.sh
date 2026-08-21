#!/bin/bash
# ==============================================================================
# Slurm job: 3-iteration GPU smoke of the two new acquisition paths, on apudev.
#
# Validates, on the real device and with the real pipeline, that
#   --selection-strategy tol_only_random          (regression head, mean-guided)
#   --head classification --selection-strategy bald
# both run end to end, before any 40-iteration production run is committed to
# them. No data generation (--no-generate-data): the point is the training and
# selection path, not the simulator.
#
#   sbatch slurm/submit_head_smoke.sh
# ==============================================================================
#SBATCH --job-name=head_smoke
#SBATCH --partition=apudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO_ROOT}/al_pmssmwithgp/model:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
PYTHON="${REPO_ROOT}/.pixi/envs/rocm/bin/python"
POOL="${CLUSTER_DATA_DIR:-/ptmp/jwuerzin/data}/260804"
OUT="/ptmp/jwuerzin/output/head_smoke_${SLURM_JOB_ID}"

rc=0
for spec in "regression:tol_only_random" "classification:bald"; do
    HEAD="${spec%%:*}"
    STRAT="${spec##*:}"
    echo "=============================================================="
    echo " smoke: head=${HEAD} strategy=${STRAT}"
    echo "=============================================================="
    "${PYTHON}" active_learning_dnn.py \
        --testing \
        --target ExpR --y-transform log \
        --head "${HEAD}" \
        --selection-strategy "${STRAT}" \
        --n-iterations 3 --n-select 20 --n-candidates 2000 \
        --n-samples 400 --data-dir "${POOL}" \
        --static-eval-size 2000 \
        --no-mcmc-eval --no-generate-data --no-warm-starting \
        --output-dir "${OUT}/${HEAD}_${STRAT}" \
        --seed 1 --gpu-ids 0
    status=$?
    echo "[smoke] head=${HEAD} strategy=${STRAT} exit=${status}"
    [[ ${status} -ne 0 ]] && rc=${status}
done

# The combinations that must be rejected rather than silently mis-run.
echo "=============================================================="
echo " smoke: guards"
echo "=============================================================="
"${PYTHON}" active_learning_dnn.py --target ExpR --head classification \
    --selection-strategy entropy_batch --n-iterations 1 2>&1 | tail -3
"${PYTHON}" active_learning_dnn.py --target ExpR --head regression \
    --selection-strategy bald --n-iterations 1 2>&1 | tail -3

echo "[smoke] overall exit=${rc}"
exit ${rc}
