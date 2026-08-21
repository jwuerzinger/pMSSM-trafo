#!/bin/bash
# In-band support covered per head/strategy arm, one panel per model.
#
# apudev, because the run is now short: the support is built from the parsed-pool
# .npy cache instead of re-parsing 1499 ROOT files (19 min -> seconds), and the
# per-arm curves are only a state.pt read plus cell assignment. apudev caps the
# wall clock at 15 minutes and has idle nodes, so this starts immediately.
#
# --expect-cells 1067 is a guard, not decoration: n_bins must be 12 over
# AXES = (M_1, M_2, mu) to reproduce the published 1067-of-1728 support. A
# coarser grid still "works" and silently produces a support on which every arm
# saturates (n_bins=3 gives 27 cells), so the count is asserted.
#SBATCH --job-name=prelim_support
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
.pixi/envs/rocm/bin/python scripts/plot_prelim_support.py \
    --headtest-glob '/ptmp/jwuerzin/output/headtest_*_20260821_*' \
    --exclude-runs '_111422' \
    --n-bins 12 --min-cell 20 --expect-cells 1067 \
    --output-dir /ptmp/jwuerzin/analysis/joint/prelim_20260821
echo "[exit] $?"
