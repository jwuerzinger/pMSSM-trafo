#!/bin/bash
# Resume a previous active learning run from its state.pt checkpoint.
#
# Usage:
#   bash resume_slurm.sh <slurm_script> <prev_output_dir> <n_additional_iterations>
#
# Examples:
#   bash resume_slurm.sh slurm/submit_al_transformer.sh \
#       /ptmp/jwuerzin/output/active_learning_output_slurm_20260414_143005 20
#
#   bash resume_slurm.sh slurm/submit_al_gp_exact.sh \
#       /ptmp/jwuerzin/output/active_learning_exact_gp_output_20260414_143005 10
#
# The previous output dir (NOT a timestamped new one) is reused so plots,
# gifs, and iteration_NNN/ dirs accumulate in place. state.pt inside that
# dir tells the script where to continue from.

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: bash resume_slurm.sh <slurm_script> <prev_output_dir> <n_additional>"
    exit 2
fi

SLURM_SCRIPT="$1"
PREV_DIR="$2"
N_ADD="$3"

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -f "${SLURM_SCRIPT}" ]]; then
    # Allow short names like "submit_al_transformer.sh"
    if [[ -f "${REPO_ROOT}/slurm/${SLURM_SCRIPT}" ]]; then
        SLURM_SCRIPT="${REPO_ROOT}/slurm/${SLURM_SCRIPT}"
    else
        echo "Error: slurm script '${SLURM_SCRIPT}' not found."
        exit 1
    fi
fi

if [[ ! -f "${PREV_DIR}/state.pt" ]]; then
    echo "Error: ${PREV_DIR}/state.pt not found — cannot resume."
    exit 1
fi

if [[ ! -f "${REPO_ROOT}/slurm/cluster.conf" ]]; then
    echo "Error: slurm/cluster.conf not found."
    exit 1
fi
source "${REPO_ROOT}/slurm/cluster.conf"
export PIXI_ENV

# Pick gres based on script name (heuristic; override with CLUSTER_GPU_GRES env)
if [[ -n "${CLUSTER_GPU_GRES:-}" ]]; then
    GRES="${CLUSTER_GPU_GRES}"
elif [[ "${SLURM_SCRIPT}" == *tabpfn* ]]; then
    GRES="${CLUSTER_GPU_GRES_1}"
else
    GRES="${CLUSTER_GPU_GRES_2}"
fi

# Forward the token for tabpfn resumes
EXP="ALL,AL_RESUME_FROM=${PREV_DIR},AL_N_ADDITIONAL_ITERATIONS=${N_ADD},AL_OUTPUT_DIR=${PREV_DIR}"
if [[ "${SLURM_SCRIPT}" == *tabpfn* && -n "${TABPFN_TOKEN:-}" ]]; then
    EXP="${EXP},TABPFN_TOKEN=${TABPFN_TOKEN}"
fi

echo "[resume] script:   ${SLURM_SCRIPT}"
echo "[resume] prev dir: ${PREV_DIR}"
echo "[resume] +${N_ADD} iterations"
sbatch --partition="${CLUSTER_PARTITION}" --gres="${GRES}" --export="${EXP}" "${SLURM_SCRIPT}"
