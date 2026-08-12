#!/bin/bash
#SBATCH --job-name=dgp_cost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Measures (a) how much of a Deep GP training iteration is the unbatched
# validation pass at the benchmark's own scale, and (b) where the size wall sits
# once that pass is batched. The apparent ceiling of n_train ~ 5.8e4 is set by
# n_val, not by the method, so it is not the Deep GP's limit.
set -uo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"; cd "${REPO_ROOT}"
PY="${REPO_ROOT}/.pixi/envs/${PIXI_ENV:-rocm}/bin/python"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="${REPO_ROOT}/al_pmssmwithgp/model:${PYTHONPATH:-}"
echo "=== timing at the benchmark scale ==="
"${PY}" -u scripts/deepgp_validation_cost.py --mode time --n-train "${DGP_NTRAIN:-12581}" \
    --out /ptmp/jwuerzin/analysis/all_runs/deepgp_validation_cost_time.json
echo
echo "=== size sweep (one subprocess per size, so a GPU abort is survivable) ==="
"${PY}" -u scripts/deepgp_validation_cost.py --mode sweep --sizes "${DGP_SIZES:-20000,40000,60000,80000,120000,160000}" \
    --out /ptmp/jwuerzin/analysis/all_runs/deepgp_validation_cost_sweep.json
