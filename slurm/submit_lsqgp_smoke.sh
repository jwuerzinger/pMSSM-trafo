#!/bin/bash
# apudev smoke of the least-squares GP classifier arm (R&W section 6.5) on both
# acquisition rules, before its four 5-seed cells are allowed to start.
# --no-generate-data: the question is whether the head trains and whether bald /
# cls_entropy can score from a conjugate exact-GP posterior, not the simulator.
#SBATCH --job-name=lsq_smoke
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
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/al_pmssmwithgp/model:${PYTHONPATH:-}"
PY="${SLURM_SUBMIT_DIR}/.pixi/envs/rocm/bin/python"
POOL=/ptmp/jwuerzin/data/260804
OUT=/ptmp/jwuerzin/output/lsq_smoke_${SLURM_JOB_ID}
rc=0
for strat in bald cls_entropy; do
    echo "=============== lsq_classification / ${strat} ==============="
    "${PY}" active_learning_gp.py \
        --target ExpR --no-mcmc-eval --data-dir "${POOL}" \
        --model-type exact_gp --head lsq_classification \
        --selection-strategy "${strat}" \
        --n-iterations 2 --n-select 20 --n-candidates 4000 --n-samples 400 \
        --static-eval-size 2000 --no-generate-data --epochs 200 --patience 50 \
        --seed 1 --gpu-ids 0 --output-dir "${OUT}/${strat}"
    st=$?; echo "[smoke] ${strat} exit=${st}"; [[ ${st} -ne 0 ]] && rc=${st}
done
echo "[smoke] overall rc=${rc}"; exit ${rc}
