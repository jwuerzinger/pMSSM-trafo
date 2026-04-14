#!/bin/bash -l
# Quick environment check — verifies imports and CUDA before submitting full jobs.
# Uses gpudev partition (short queue, starts almost immediately).
#
# Submit: sbatch slurm/check_env.sh
#SBATCH --job-name=al_check_env
#SBATCH --partition=gpudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=/raven/u/jwuerzin/pMSSM-trafo/logs/%x_%j.out
#SBATCH --error=/raven/u/jwuerzin/pMSSM-trafo/logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/raven/u/jwuerzin/pMSSM-trafo}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/al_pmssmwithgp/model:${PYTHONPATH:-}"

echo "=========================================="
echo " Job:     ${SLURM_JOB_NAME} | ID: ${SLURM_JOB_ID}"
echo " Node:    $(hostname)"
echo " Started: $(date)"
echo "=========================================="

PYTHON="${REPO_ROOT}/.pixi/envs/default/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "[setup] pixi env not found — running: pixi install"
    /u/jwuerzin/.pixi/bin/pixi install
fi
if [[ ! -x "${PYTHON}" ]]; then
    echo "[error] Python not found after pixi install"
    exit 1
fi
echo "[env] $(${PYTHON} --version)"

echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"

"${PYTHON}" - <<'EOF'
import torch
import gpytorch
import numpy as np
import pandas as pd
import sklearn
import uproot
import tabpfn

# Check CUDA
assert torch.cuda.is_available(), "CUDA not available!"
print(f"[ok] torch       {torch.__version__}  |  CUDA {torch.version.cuda}  |  GPU: {torch.cuda.get_device_name(0)}")
print(f"[ok] gpytorch    {gpytorch.__version__}")
print(f"[ok] numpy       {np.__version__}")
print(f"[ok] pandas      {pd.__version__}")
print(f"[ok] scikit-learn {sklearn.__version__}")
print(f"[ok] uproot      {uproot.__version__}")
print(f"[ok] tabpfn      {tabpfn.__version__}")

# Check pmssm package
from pmssm.models import PMSSMTransformerTabular
from pmssm.data import load_pmssm_data
print("[ok] pmssm package imports OK")

# Check GP submodule
from gp_pipeline.models.exact_gp import ExactGP
print("[ok] gp_pipeline imports OK")

print("\n=== All checks passed ===")
EOF

echo "=========================================="
echo " Done: $(date)"
echo "=========================================="
