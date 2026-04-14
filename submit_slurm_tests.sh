#!/bin/bash
# Submit all architecture smoke-test jobs to Slurm (dev partition).
# Cluster-specific settings (partition, gres, pixi env) are read from
# slurm/cluster.conf — copy slurm/cluster.conf.template and edit as needed.
#
# Run from repo root: bash submit_slurm_tests.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -f "${REPO_ROOT}/slurm/cluster.conf" ]]; then
    echo "Error: slurm/cluster.conf not found."
    echo "       cp slurm/cluster.conf.template slurm/cluster.conf"
    exit 1
fi
source "${REPO_ROOT}/slurm/cluster.conf"
export PIXI_ENV

mkdir -p "${REPO_ROOT}/logs"

COMMON="--partition=${CLUSTER_PARTITION_DEV}"

# Transformer (1 GPU)
sbatch ${COMMON} --gres=${CLUSTER_GPU_GRES_1} slurm/test_al_transformer.sh

# ExactGP (2 GPUs)
sbatch ${COMMON} --gres=${CLUSTER_GPU_GRES_2} slurm/test_al_gp_exact.sh

# DeepGP (1 GPU)
sbatch ${COMMON} --gres=${CLUSTER_GPU_GRES_1} slurm/test_al_gp_deep.sh

# TabPFN requires a license token to download model weights.
# Loaded from slurm/cluster.conf if saved there; otherwise prompt and save for next time.
if [[ -z "${TABPFN_TOKEN:-}" ]]; then
    read -rsp "TABPFN_TOKEN not set. Enter your token (input hidden): " TABPFN_TOKEN
    echo
    echo "TABPFN_TOKEN=\"${TABPFN_TOKEN}\"" >> "${REPO_ROOT}/slurm/cluster.conf"
    echo "[info] Token saved to slurm/cluster.conf — will be loaded automatically next time."
fi
export TABPFN_TOKEN

# TabPFN (1 GPU)
sbatch ${COMMON} --gres=${CLUSTER_GPU_GRES_1} slurm/test_al_tabpfn.sh
