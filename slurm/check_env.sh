#!/bin/bash
# Quick environment check — verifies imports and GPU before submitting full jobs.
#
# Setup: cp slurm/cluster.conf.template slurm/cluster.conf  (edit for your cluster)
#
# Submit from repo root:
#   source slurm/cluster.conf
#   sbatch --partition="${CLUSTER_PARTITION_DEV}" \
#          --gres="${CLUSTER_GPU_GRES_1}" slurm/check_env.sh
#SBATCH --job-name=al_check_env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Partition and gres are set via sbatch flags from cluster.conf.
# See slurm/cluster.conf.template for details.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs"

# ---- Cluster config ----------------------------------------------------------
if [[ -f "${REPO_ROOT}/slurm/cluster.conf" ]]; then
    source "${REPO_ROOT}/slurm/cluster.conf"
else
    echo "[warn] slurm/cluster.conf not found — using defaults"
    echo "       cp slurm/cluster.conf.template slurm/cluster.conf"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/al_pmssmwithgp/model:${PYTHONPATH:-}"

echo "=========================================="
echo " Job:     ${SLURM_JOB_NAME} | ID: ${SLURM_JOB_ID}"
echo " Node:    $(hostname)"
echo " Started: $(date)"
echo "=========================================="

PIXI_ENV="${PIXI_ENV:-cuda}"
PYTHON="${REPO_ROOT}/.pixi/envs/${PIXI_ENV}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "[setup] pixi env '${PIXI_ENV}' not found — running: pixi install -e ${PIXI_ENV}"
    /u/jwuerzin/.pixi/bin/pixi install -e "${PIXI_ENV}"
fi
if [[ ! -x "${PYTHON}" ]]; then
    echo "[error] Python not found after pixi install"
    exit 1
fi
echo "[env] $(${PYTHON} --version)"

echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "[gpu] ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<not set>}"
echo "[gpu] SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<not set>}"
echo "[gpu] PIXI_ENV=${PIXI_ENV}"

"${PYTHON}" - <<'EOF'
import torch
import gpytorch
import numpy as np
import pandas as pd
import sklearn
import uproot
import tabpfn

# Check GPU (works for both CUDA and ROCm — PyTorch ROCm maps torch.cuda.* to HIP)
assert torch.cuda.is_available(), "GPU not available (neither CUDA nor ROCm detected)!"

if hasattr(torch.version, 'hip') and torch.version.hip:
    backend = f"ROCm {torch.version.hip}"
elif torch.version.cuda:
    backend = f"CUDA {torch.version.cuda}"
else:
    backend = "Unknown"

print(f"[ok] torch       {torch.__version__}  |  {backend}  |  GPU: {torch.cuda.get_device_name(0)}")
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
