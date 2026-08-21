#!/bin/bash
# =============================================================================
# The campaign's resume loop. Runs scripts/campaign_chase.py, then resubmits
# itself six hours later, until a horizon date.
#
# Why a self-resubmitting job and not a dependency chain: reaching 200
# iterations takes several 24 h jobs per cell, and 44 cells x 4 links would be
# 176 pre-submitted jobs against a MaxSubmit of 300, all of them guessing in
# advance how many links each cell needs. A loop that measures what each cell
# has actually reached needs neither the guess nor the job slots.
#
# Why apu1 and a 15-minute wall clock: the work is a directory count and a few
# sbatch calls, seconds of it. Single-node jobs schedule freely here (QOS a0001,
# MaxJobsPU 32) while the multi-node bundles queue behind the association's
# 8-job limit, so this always gets to run. Keeping it short also keeps it from
# holding a slot the bundles want.
#
# Submit:
#   CAMPAIGN_CELLS=/ptmp/jwuerzin/analysis/campaign/<id>_cells.csv \
#   CAMPAIGN_HORIZON=2026-09-30 \
#       sbatch --partition=apu1 --gres=gpu:1 --mem=16G slurm/submit_campaign_chase.sh
#
# Env:
#   CAMPAIGN_CELLS    the cells CSV written by submit_campaign_200.sh (required)
#   CAMPAIGN_HORIZON  YYYY-MM-DD; the loop stops resubmitting after this
#   CHASE_RESUME_TO   iteration target (default 200)
#   CHASE_QUEUE_CAP   stop submitting past this many of my queued jobs (default 24)
#   CHASE_EVERY       resubmission delay (default 6hours)
#   DRY_RUN=1         chase in dry-run AND validate the resubmission with
#                     `sbatch --test-only` instead of queueing it
# =============================================================================
#SBATCH --job-name=c200_chase
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs

if [[ -f "${REPO_ROOT}/slurm/cluster.conf" ]]; then
    source "${REPO_ROOT}/slurm/cluster.conf"
fi
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
PYTHON="${REPO_ROOT}/.pixi/envs/${PIXI_ENV:-rocm}/bin/python"

CELLS="${CAMPAIGN_CELLS:-}"
HORIZON="${CAMPAIGN_HORIZON:-2026-09-30}"
RESUME_TO="${CHASE_RESUME_TO:-200}"
QUEUE_CAP="${CHASE_QUEUE_CAP:-24}"
EVERY="${CHASE_EVERY:-6hours}"
DRY_RUN="${DRY_RUN:-0}"

echo "=========================================="
echo " campaign chase   job=${SLURM_JOB_ID:-local}  node=$(hostname)"
echo " started  = $(date)"
echo " cells    = ${CELLS:-<none>}"
echo " horizon  = ${HORIZON}   every = ${EVERY}   target = ${RESUME_TO}"
echo " dry_run  = ${DRY_RUN}"
echo "=========================================="

if [[ -z "${CELLS}" || ! -f "${CELLS}" ]]; then
    echo "[error] CAMPAIGN_CELLS is unset or missing: '${CELLS}'" >&2
    echo "        Without it only the continuation cells would be chased and the" >&2
    echo "        32 campaign cells would stall at the wall clock. Aborting." >&2
    exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
    echo "[error] ${PYTHON} is not executable; aborting rather than looping idle" >&2
    exit 1
fi

CHASE_FLAGS=(--cells "${CELLS}" --resume-to "${RESUME_TO}"
             --queue-cap "${QUEUE_CAP}")
[[ "${DRY_RUN}" == "1" ]] && CHASE_FLAGS+=(--dry-run) || CHASE_FLAGS+=(--submit)
"${PYTHON}" scripts/campaign_chase.py "${CHASE_FLAGS[@]}"
RC=$?
echo "[chase] campaign_chase.py exited ${RC}"

# ---- resubmit, unless past the horizon --------------------------------------
TODAY="$(date +%Y-%m-%d)"
if [[ "${TODAY}" > "${HORIZON}" ]]; then
    echo "[chase] ${TODAY} is past the horizon ${HORIZON}; the loop ends here."
    echo "[chase] Resume it with: CAMPAIGN_CELLS=${CELLS} CAMPAIGN_HORIZON=<date> \\"
    echo "        sbatch --partition=apu1 --gres=gpu:1 --mem=16G slurm/submit_campaign_chase.sh"
    exit ${RC}
fi

export CAMPAIGN_CELLS="${CELLS}" CAMPAIGN_HORIZON="${HORIZON}"
export CHASE_RESUME_TO="${RESUME_TO}" CHASE_QUEUE_CAP="${QUEUE_CAP}"
export CHASE_EVERY="${EVERY}" DRY_RUN="${DRY_RUN}"
SB=(sbatch --parsable "--begin=now+${EVERY}" --partition=apu1 --gres=gpu:1
    --mem=16G --export=ALL slurm/submit_campaign_chase.sh)
# In dry-run, --test-only exercises the whole path (sbatch on a compute node,
# the controller connection, the flag set) without queueing anything. That is
# the one assumption this design rests on, so it is checked rather than assumed.
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[chase] dry-run: validating the resubmission with --test-only"
    sbatch --test-only "--begin=now+${EVERY}" --partition=apu1 --gres=gpu:1 \
        --mem=16G --export=ALL slurm/submit_campaign_chase.sh 2>&1 | sed 's/^/        /'
    echo "[chase] --test-only rc=$?"
    exit ${RC}
fi
NEXT="$("${SB[@]}")"
if [[ -z "${NEXT}" ]]; then
    echo "[chase] WARNING: resubmission failed. The loop has STOPPED." >&2
    echo "        Restart it by hand with the command in this script's header." >&2
    exit 1
fi
echo "[chase] next wake queued as job ${NEXT}, begins in ${EVERY}"
exit ${RC}
