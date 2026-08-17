#!/bin/bash
#SBATCH --job-name=disp_oob
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=110000
#SBATCH --time=08:00:00
#SBATCH --array=1-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-/viper/u2/jwuerzin/pMSSM-trafo}"
cd "${REPO_ROOT}"; mkdir -p logs
source "${REPO_ROOT}/slurm/cluster.conf"
export PYTHONUNBUFFERED=1 PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO_ROOT}/al_pmssmwithgp/model:${PYTHONPATH:-}"
"${REPO_ROOT}/.pixi/envs/${PIXI_ENV:-rocm}/bin/python" scripts/dispersion_experiment.py \
    --arm random_plus_al_outofband --seed "${SLURM_ARRAY_TASK_ID}" \
    --output-dir /ptmp/jwuerzin/analysis/dispersion_experiment
