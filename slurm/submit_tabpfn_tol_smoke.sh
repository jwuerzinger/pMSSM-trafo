#!/bin/bash
# Does TabPFN + tol_only_random now survive selection and write state.pt?
# Before the al_needs_cand fix it raised TypeError after training and before
# save_state, so the cell sat at iteration 1 forever. The assertion here is
# therefore not "it runs" but "iteration_002 exists and state.pt is on disk".
#SBATCH --job-name=tabpfn_tol_smoke
#SBATCH --partition=apudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:14:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
[[ -f slurm/cluster.conf ]] && source slurm/cluster.conf
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/al_pmssmwithgp/model:${PYTHONPATH:-}"
export TABPFN_TOKEN="${TABPFN_TOKEN:-}"
PY="${SLURM_SUBMIT_DIR}/.pixi/envs/rocm/bin/python"
OUT=/ptmp/jwuerzin/output/tabpfn_tol_smoke_${SLURM_JOB_ID}
"${PY}" active_learning_tabpfn.py \
    --target DMRD --n-iterations 2 --n-select 20 --n-candidates 5000 \
    --n-samples 400 --data-dir /ptmp/jwuerzin/data/18387358 \
    --static-eval-size 2000 --no-mcmc-eval --no-generate-data \
    --selection-strategy tol_only_random --seed 1 --gpu-ids 0 \
    --output-dir "${OUT}"
echo "[smoke] driver exit=$?"
echo "=== the assertion ==="
echo "  iterations : $(ls -d ${OUT}/iteration_[0-9][0-9][0-9] 2>/dev/null | wc -l)  (need 2)"
echo "  state.pt   : $([ -f ${OUT}/state.pt ] && echo present || echo MISSING)"
