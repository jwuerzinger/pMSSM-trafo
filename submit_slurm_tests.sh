#!/bin/bash
# Submit all architecture smoke-test jobs to Slurm (dev partition).
# Cluster-specific settings (partition, gres, pixi env) are read from
# slurm/cluster.conf — copy slurm/cluster.conf.template and edit as needed.
#
# Each job's output directory is automatically timestamped so successive
# submissions never overwrite earlier results.
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

TS="$(date +%Y%m%d_%H%M%S)"
COMMON="--partition=${CLUSTER_PARTITION_DEV}"
OUT="/ptmp/jwuerzin/output"

sub() {
    # sub <output_dir_name> <gres> <extra_export> <script>
    local name="$1" gres="$2" extra="$3" script="$4"
    local dir="${OUT}/${name}_${TS}"
    local exp="ALL,AL_OUTPUT_DIR=${dir}${extra:+,$extra}"
    echo "[submit] ${script} → ${dir}"
    sbatch ${COMMON} --gres="${gres}" --export="${exp}" "${script}"
}

# Transformer (1 GPU)
sub test_al_transformer_output "${CLUSTER_GPU_GRES_1}" "" slurm/test_al_transformer.sh

# ExactGP (2 GPUs)
sub test_al_gp_exact_output    "${CLUSTER_GPU_GRES_2}" "" slurm/test_al_gp_exact.sh

# DeepGP (1 GPU)
sub test_al_gp_deep_output     "${CLUSTER_GPU_GRES_1}" "" slurm/test_al_gp_deep.sh

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
sub test_al_tabpfn_output "${CLUSTER_GPU_GRES_1}" "TABPFN_TOKEN=${TABPFN_TOKEN}" slurm/test_al_tabpfn.sh
