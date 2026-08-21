#!/bin/bash
# A/B: the same tiny-N command against the pre-change and post-change dnn driver.
# Settles whether the silent exit at the entropy_batch selection step is mine or
# a pre-existing small-N edge case exposed by --testing (which pins
# n_candidates to 100, against production's 1,000,000).
#SBATCH --job-name=ab_sel
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
OUT=/ptmp/jwuerzin/output/ab_sel_${SLURM_JOB_ID}
PRE=/tmp/claude-54825/-viper-u2-jwuerzin-pMSSM-trafo/1fdc4dc0-ebd7-455b-a4bf-4bce2e5236f8/scratchpad/al_dnn_prechange.py

ARGS=(--target ExpR --target-value 1.0 --y-transform log
      --n-iterations 1 --n-select 20 --n-candidates 2000 --n-samples 400
      --data-dir "${POOL}" --static-eval-size 2000 --no-mcmc-eval
      --no-generate-data --seed 1 --gpu-ids 0 --testing --epochs 40
      --no-warm-starting)

echo "### PRE-CHANGE driver (HEAD~1) ###"
"${PY}" "${PRE}" "${ARGS[@]}" --output-dir "${OUT}/pre" >"${OUT}_pre.txt" 2>&1
echo "pre exit=$?"
echo "### POST-CHANGE driver (working tree) ###"
"${PY}" active_learning_dnn.py "${ARGS[@]}" --output-dir "${OUT}/post" >"${OUT}_post.txt" 2>&1
echo "post exit=$?"
for f in "${OUT}_pre.txt" "${OUT}_post.txt"; do
  echo "--- tail of $(basename $f) ---"; tail -4 "$f"
done
