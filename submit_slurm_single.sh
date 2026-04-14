#!/bin/bash
# Submit all active learning jobs to Slurm, requesting ONLY ONE GPU per job.
# Sequential AL+baseline training inside each job (no within-job parallelism);
# the tradeoff is shorter queue waits, which often wins overall walltime.
#
# Each job's output directory is automatically timestamped so successive
# submissions never overwrite earlier results.
#
# Run from repo root: bash submit_slurm_single.sh

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
    # sub <output_dir_name> <extra_export> <script>
    local name="$1" extra="$2" script="$3"
    local dir="${OUT}/${name}_single_${TS}"
    local exp="ALL,AL_OUTPUT_DIR=${dir}${extra:+,$extra}"
    # Derive a "<jobname>_single" name from the script filename so Slurm log
    # files (%x_%j.out / .err) are distinguishable from 2-GPU runs.
    local base
    base="$(basename "${script}" .sh)"
    base="${base#submit_}"   # submit_al_foo → al_foo
    local jobname="${base}_single"
    echo "[submit] ${script} → ${dir}  (job name: ${jobname})"
    sbatch ${COMMON} --gres="${CLUSTER_GPU_GRES_1}" \
           --job-name="${jobname}" --export="${exp}" "${script}"
}

# Transformer (1 GPU — sequential AL+baseline)
sub active_learning_output_slurm              "" slurm/submit_al_transformer.sh
sub active_learning_output_top_k_slurm        "" slurm/submit_al_transformer_top_k.sh
sub active_learning_output_top_k_n_select_20k "" slurm/submit_al_transformer_top_k_20k.sh

# ExactGP (1 GPU)
sub active_learning_exact_gp_output       "" slurm/submit_al_gp_exact.sh
sub active_learning_exact_gp_top_k_output "" slurm/submit_al_gp_exact_top_k.sh

# DeepGP (1 GPU)
sub active_learning_deep_gp_output       "" slurm/submit_al_gp_deep.sh
sub active_learning_deep_gp_top_k_output "" slurm/submit_al_gp_deep_top_k.sh

# TabPFN requires a license token.
if [[ -z "${TABPFN_TOKEN:-}" ]]; then
    read -rsp "TABPFN_TOKEN not set. Enter your token (input hidden): " TABPFN_TOKEN
    echo
    echo "TABPFN_TOKEN=\"${TABPFN_TOKEN}\"" >> "${REPO_ROOT}/slurm/cluster.conf"
    echo "[info] Token saved to slurm/cluster.conf — will be loaded automatically next time."
fi
export TABPFN_TOKEN

# TabPFN (1 GPU — already single-GPU by design)
sub active_learning_tabpfn_output_slurm         "TABPFN_TOKEN=${TABPFN_TOKEN}" slurm/submit_al_tabpfn.sh
sub active_learning_tabpfn_entropy_output_slurm "TABPFN_TOKEN=${TABPFN_TOKEN}" slurm/submit_al_tabpfn_entropy.sh
