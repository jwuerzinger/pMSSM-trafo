#!/bin/bash
# Recompute verdict accuracy from checkpoints for the classification GP arms,
# whose run-time cache holds the positive-class fraction rather than an accuracy
# (see scripts/deepgp_posthoc_accuracy.py). A GPU is needed because the deep GP
# posterior has to be evaluated on the 100k static eval set per iteration.
#
#   sbatch slurm/submit_deepgp_posthoc_acc.sh
#SBATCH --job-name=dgp_acc
#SBATCH --partition=apudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}/al_pmssmwithgp/model:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=8

.pixi/envs/rocm/bin/python scripts/deepgp_posthoc_accuracy.py \
    --runs '/ptmp/jwuerzin/output/headtest_deepgp_*_seed1_20260821_11*' \
    --data-dir /ptmp/jwuerzin/data/260804 --target ExpR \
    --device cuda --write
echo "[exit] $?"
