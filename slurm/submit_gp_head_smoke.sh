#!/bin/bash
# ==============================================================================
# Slurm job: GPU smoke of the GP classification heads, with PRODUCTION
# hyperparameters, on apudev.
#
# Validates before any long run is committed to them that
#   exact GP  --head lsq_classification --selection-strategy bald|cls_entropy
#   deep GP   --head classification     --selection-strategy bald|cls_entropy
# train, produce a posterior and select, end to end. Each is run beside its
# REGRESSION control, so a failure can be attributed to the head rather than to
# the harness.
#
# The GP hyperparameters below are copied from submit_al_gp_{exact,deep}.sh and
# are not adjustable knobs here. An earlier version of this script used the CLI
# defaults instead and both deep GP arms died in NotPSDError -- including the
# regression control, which is what identified it: production caps inducing
# points at 256, the default is 512, and 512 inducing locations drawn from 1600
# training points are close enough together to make the inducing-inducing kernel
# singular. A smoke test on defaults tests the defaults, not the run.
#
# Only the epoch budget is shortened (200/patience 20 against 10000/patience
# 100), since the fault modes under test appear in the first forward pass.
#
#   sbatch slurm/submit_gp_head_smoke.sh
# ==============================================================================
#SBATCH --job-name=gp_head_smoke
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
export SMODELS_CACHEDIR="${SMODELS_CACHEDIR:-/ptmp/jwuerzin/cache/smodels}"
export PYTHONPATH="${REPO_ROOT}/al_pmssmwithgp/model:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
PYTHON="${REPO_ROOT}/.pixi/envs/rocm/bin/python"
POOL="/ptmp/jwuerzin/data/260804"
OUT="/ptmp/jwuerzin/output/gp_head_smoke_${SLURM_JOB_ID}"

# --n-datasets 3 gives production's initial n_train (1600) while parsing 3 ROOT
# files instead of all 1499, which on its own outran a 15-minute wall clock.
run () {   # run <label> <model_type> <head> <strategy> <extra...>
    local label="$1" mt="$2" head="$3" strat="$4"; shift 4
    echo "=============================================================="
    echo " smoke [${label}]: ${mt} head=${head} strategy=${strat}"
    echo "=============================================================="
    "${PYTHON}" active_learning_gp.py \
        --model-type "${mt}" \
        --target ExpR \
        --head "${head}" --selection-strategy "${strat}" \
        --n-iterations 2 --n-select 20 --n-candidates 3000 \
        --n-datasets 3 --n-samples 2000 \
        --epochs 200 --early-stopping --patience 20 \
        --learning-rate 1e-3 \
        --kernel RBF --lengthscale 1.0 --noise 1e-2 --jitter 1e-3 --use-ard \
        "$@" \
        --data-dir "${POOL}" --static-eval-size 2000 \
        --no-mcmc-eval --no-generate-data \
        --output-dir "${OUT}/${mt}_${head}_${strat}" \
        --seed 1
    echo "[smoke] ${label} ${mt}/${head}/${strat} exit=$?"
}

run CONTROL exact_gp regression         entropy_batch
run TEST    exact_gp lsq_classification bald
run TEST    exact_gp lsq_classification cls_entropy
run CONTROL deep_gp  regression         entropy_batch --num-inducing-max 256
run TEST    deep_gp  classification     bald          --num-inducing-max 256
run TEST    deep_gp  classification     cls_entropy   --num-inducing-max 256

# The combinations that must be REJECTED rather than silently mis-run.
echo "=============================================================="
echo " smoke: guards (each must print a UsageError, not run)"
echo "=============================================================="
"${PYTHON}" active_learning_gp.py --model-type exact_gp --target ExpR \
    --head classification --selection-strategy bald --n-iterations 1 2>&1 | tail -3
"${PYTHON}" active_learning_gp.py --model-type deep_gp --target ExpR \
    --head classification --selection-strategy entropy_batch --n-iterations 1 2>&1 | tail -3
"${PYTHON}" active_learning_gp.py --model-type exact_gp --target ExpR \
    --head regression --selection-strategy bald --n-iterations 1 2>&1 | tail -3
echo "[smoke] done"
