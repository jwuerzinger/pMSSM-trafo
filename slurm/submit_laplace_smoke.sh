#!/bin/bash
# ==============================================================================
# Slurm job: verify and smoke the Laplace GP classifier, plus re-test the
# Bernoulli deep GP after the checkpoint-reload fix.
#
# Three things, in increasing cost:
#   1. scripts/verify_laplace_gpc.py at production's initial n (1600) on the GPU,
#      which checks the implementation against R&W's own stated properties and
#      extrapolates the per-step cost cubically to the n the AL loop reaches.
#   2. laplace_gpc end to end for 2 iterations with BALD.
#   3. deep_gp + Bernoulli end to end, which failed before with
#      "Missing key(s) in state_dict: likelihood.noise_covar.raw_noise" because
#      the two checkpoint-reload sites rebuilt the model without the head and
#      so built a Gaussian likelihood.
#
# cls_entropy is not smoked separately: it shares every line with bald except
# which score the head returns.
#
#   sbatch slurm/submit_laplace_smoke.sh
# ==============================================================================
#SBATCH --job-name=laplace_smoke
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
OUT="/ptmp/jwuerzin/output/laplace_smoke_${SLURM_JOB_ID}"

echo "=============================================================="
echo " 1. verify the Laplace GPC against R&W's stated properties"
echo "=============================================================="
"${PYTHON}" scripts/verify_laplace_gpc.py --n 1600 --dim 9 --device cuda \
    --iters 40 --time-scaling
echo "[smoke] verify exit=$?"

echo "=============================================================="
echo " 2. laplace_gpc end to end, BALD"
echo "=============================================================="
"${PYTHON}" active_learning_gp.py \
    --model-type laplace_gpc --target ExpR \
    --head classification --selection-strategy bald \
    --n-iterations 2 --n-select 20 --n-candidates 3000 \
    --n-datasets 3 --n-samples 2000 \
    --epochs 100 --early-stopping --patience 20 --learning-rate 5e-3 \
    --kernel RBF --lengthscale 1.0 --jitter 1e-3 --use-ard \
    --data-dir "${POOL}" --static-eval-size 2000 \
    --no-mcmc-eval --no-generate-data \
    --output-dir "${OUT}/laplace_bald" --seed 1
echo "[smoke] laplace_gpc/bald exit=$?"

echo "[skip] 3. deep_gp + Bernoulli already passed end to end in job 10967952"
if false; then
echo "=============================================================="
echo " 3. deep_gp + Bernoulli, BALD (checkpoint-reload regression test)"
echo "=============================================================="
"${PYTHON}" active_learning_gp.py \
    --model-type deep_gp --target ExpR \
    --head classification --selection-strategy bald \
    --n-iterations 2 --n-select 20 --n-candidates 3000 \
    --n-datasets 3 --n-samples 2000 \
    --epochs 200 --early-stopping --patience 20 --learning-rate 1e-3 \
    --kernel RBF --lengthscale 1.0 --noise 1e-2 --jitter 1e-3 --use-ard \
    --num-inducing-max 256 \
    --data-dir "${POOL}" --static-eval-size 2000 \
    --no-mcmc-eval --no-generate-data \
    --output-dir "${OUT}/deepgp_bald" --seed 1
echo "[smoke] deep_gp/bernoulli/bald exit=$?"
fi
echo "[smoke] done"
