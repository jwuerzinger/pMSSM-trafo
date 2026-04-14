#!/bin/bash
# Submit all active learning jobs to Slurm.
# Cluster-specific settings (partition, gres, pixi env) are read from
# slurm/cluster.conf — copy slurm/cluster.conf.template and edit as needed.
#
# Run from repo root: bash submit_slurm.sh

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

COMMON="--partition=${CLUSTER_PARTITION}"

# Transformer (2 GPUs)
sbatch ${COMMON} --gres=${CLUSTER_GPU_GRES_2} slurm/submit_al_transformer.sh
sbatch ${COMMON} --gres=${CLUSTER_GPU_GRES_2} slurm/submit_al_transformer_top_k.sh
sbatch ${COMMON} --gres=${CLUSTER_GPU_GRES_2} slurm/submit_al_transformer_top_k_20k.sh

# ExactGP (2 GPUs)
sbatch ${COMMON} --gres=${CLUSTER_GPU_GRES_2} slurm/submit_al_gp_exact.sh
sbatch ${COMMON} --gres=${CLUSTER_GPU_GRES_2} slurm/submit_al_gp_exact_top_k.sh

# DeepGP (2 GPUs)
sbatch ${COMMON} --gres=${CLUSTER_GPU_GRES_2} slurm/submit_al_gp_deep.sh
sbatch ${COMMON} --gres=${CLUSTER_GPU_GRES_2} slurm/submit_al_gp_deep_top_k.sh

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
sbatch ${COMMON} --gres=${CLUSTER_GPU_GRES_1} --export=ALL,TABPFN_TOKEN="${TABPFN_TOKEN}" slurm/submit_al_tabpfn.sh
sbatch ${COMMON} --gres=${CLUSTER_GPU_GRES_1} --export=ALL,TABPFN_TOKEN="${TABPFN_TOKEN}" slurm/submit_al_tabpfn_entropy.sh
