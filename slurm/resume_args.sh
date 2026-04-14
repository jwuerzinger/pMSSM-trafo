# Source this from a slurm submit script to build RESUME_ARGS from env vars.
#
# If AL_RESUME_FROM is set, sets RESUME_ARGS to the matching --resume-from /
# --n-additional-iterations flags; otherwise RESUME_ARGS is empty.
#
# Usage:
#   source "${REPO_ROOT}/slurm/resume_args.sh"
#   "${PYTHON}" active_learning.py ... ${RESUME_ARGS}

RESUME_ARGS=""
if [[ -n "${AL_RESUME_FROM:-}" ]]; then
    : "${AL_N_ADDITIONAL_ITERATIONS:?AL_N_ADDITIONAL_ITERATIONS must be set when resuming}"
    RESUME_ARGS="--resume-from ${AL_RESUME_FROM} --n-additional-iterations ${AL_N_ADDITIONAL_ITERATIONS}"
    echo "[resume] ${RESUME_ARGS}"
fi
