#!/bin/bash
# Submit all active learning jobs to Slurm.
# Cluster-specific settings (partition, gres, pixi env) are read from
# slurm/cluster.conf — copy slurm/cluster.conf.template and edit as needed.
#
# Each job's output directory is automatically timestamped so successive
# submissions never overwrite earlier results.
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

TS="$(date +%Y%m%d_%H%M%S)"
COMMON="--partition=${CLUSTER_PARTITION}"
OUT="/ptmp/jwuerzin/output"

sub() {
    # sub <output_dir_name> <gres> <extra_export> <script>
    local name="$1" gres="$2" extra="$3" script="$4"
    local dir="${OUT}/${name}_${TS}"
    local exp="ALL,AL_OUTPUT_DIR=${dir}${extra:+,$extra}"
    echo "[submit] ${script} → ${dir}"
    sbatch ${COMMON} --gres="${gres}" --export="${exp}" "${script}"
}

# Transformer (2 GPUs)
sub active_learning_output_slurm              "${CLUSTER_GPU_GRES_2}" "" slurm/submit_al_transformer.sh
sub active_learning_output_top_k_slurm        "${CLUSTER_GPU_GRES_2}" "" slurm/submit_al_transformer_top_k.sh
sub active_learning_output_top_k_n_select_20k "${CLUSTER_GPU_GRES_2}" "" slurm/submit_al_transformer_top_k_20k.sh

# DNN (2 GPUs)
sub active_learning_dnn_output_slurm              "${CLUSTER_GPU_GRES_2}" "" slurm/submit_al_dnn.sh
sub active_learning_dnn_output_top_k_slurm        "${CLUSTER_GPU_GRES_2}" "" slurm/submit_al_dnn_top_k.sh
sub active_learning_dnn_output_top_k_n_select_20k "${CLUSTER_GPU_GRES_2}" "" slurm/submit_al_dnn_top_k_20k.sh

# ExactGP (2 GPUs)
sub active_learning_exact_gp_output       "${CLUSTER_GPU_GRES_2}" "" slurm/submit_al_gp_exact.sh
sub active_learning_exact_gp_top_k_output "${CLUSTER_GPU_GRES_2}" "" slurm/submit_al_gp_exact_top_k.sh

# DeepGP (2 GPUs)
sub active_learning_deep_gp_output       "${CLUSTER_GPU_GRES_2}" "" slurm/submit_al_gp_deep.sh
sub active_learning_deep_gp_top_k_output "${CLUSTER_GPU_GRES_2}" "" slurm/submit_al_gp_deep_top_k.sh

# TabPFN requires a license token to download model weights.
# Loaded from slurm/cluster.conf if saved there; otherwise prompt and save for next time.
if [[ -z "${TABPFN_TOKEN:-}" ]]; then
    read -rsp "TABPFN_TOKEN not set. Enter your token (input hidden): " TABPFN_TOKEN
    echo
    echo "TABPFN_TOKEN=\"${TABPFN_TOKEN}\"" >> "${REPO_ROOT}/slurm/cluster.conf"
    echo "[info] Token saved to slurm/cluster.conf — will be loaded automatically next time."
fi
export TABPFN_TOKEN

# TabPFN (2 GPUs — AL and Baseline fit+eval run in parallel)
sub active_learning_tabpfn_output_slurm         "${CLUSTER_GPU_GRES_2}" "TABPFN_TOKEN=${TABPFN_TOKEN}" slurm/submit_al_tabpfn.sh
sub active_learning_tabpfn_entropy_output_slurm "${CLUSTER_GPU_GRES_2}" "TABPFN_TOKEN=${TABPFN_TOKEN}" slurm/submit_al_tabpfn_entropy.sh
