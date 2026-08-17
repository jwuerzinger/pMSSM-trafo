#!/bin/bash
# =============================================================================
# Bundled-seed active-learning job.
#
# Allocates N nodes × 2 GPUs and runs N seeds of the SAME (model, strategy,
# warm) cell in parallel, one seed per node. Each seed is a full 2-GPU run
# (AL + baseline) — identical to the single-seed per-model submit scripts —
# just fired off via srun inside a larger sbatch allocation.
#
# Invoked by slurm/submit_strategy_sweep.sh in BUNDLE_SEEDS mode. Not intended
# for direct use, but can be called manually by PRE-EXPORTING the env vars
# (never via --export=KEY=VAL,...; see note below):
#
#   export AL_MODEL=transformer AL_STRATEGY=top_k AL_WARM=warm \
#          AL_SEEDS=1,2,3,4,5 \
#          AL_OUTPUT_BASE=/ptmp/jwuerzin/output/active_learning_transformer_top_k_warm \
#          AL_SWEEP_ID=20260422_190000
#   sbatch --partition=${CLUSTER_PARTITION} --nodes=5 --gres=gpu:2 --exclusive \
#          --export=ALL slurm/submit_al_bundled.sh
#
# Note: AL_SEEDS MUST be pre-exported. SLURM's --export uses commas as entry
# separators with no escape mechanism, so passing `AL_SEEDS=1,2,3,4,5` inline
# would be parsed as AL_SEEDS=1 plus four bare-name entries (2,3,4,5), and the
# job would silently only run seed 1.
#
# Required env vars:
#   AL_MODEL       : transformer | dnn | dnn_match_trafo | deep_gp | exact_gp | tabpfn
#   AL_STRATEGY    : top_k | top_k_tol_only | entropy_batch
#   AL_WARM        : warm | cold | tabpfn
#   AL_SEEDS       : comma-separated integers, one per allocated node
#   AL_OUTPUT_BASE : dir prefix; per-seed dirs are ${AL_OUTPUT_BASE}_seed${N}_${AL_SWEEP_ID}
#   AL_SWEEP_ID    : timestamp string used in output dir names and manifest
# =============================================================================
#SBATCH --job-name=al_bundled
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Note: --nodes, --gres, and --mem must be set by the sbatch caller, not here,
# since they depend on AL_SEEDS count and per-model GPU+memory footprint.
# (TabPFN uses gpu:1 and --mem=64G; transformer/GP use gpu:2 and --mem=128G.)

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${REPO_ROOT}"

# ---- Validate required env ---------------------------------------------------
for required in AL_MODEL AL_STRATEGY AL_WARM AL_SEEDS AL_OUTPUT_BASE AL_SWEEP_ID; do
    if [[ -z "${!required:-}" ]]; then
        echo "[error] required env var ${required} is unset" >&2
        exit 1
    fi
done

# ---- Dispatch to per-model submit script ------------------------------------
case "${AL_MODEL}" in
    transformer) PER_MODEL_SCRIPT=slurm/submit_al_transformer.sh;;
    dnn)         PER_MODEL_SCRIPT=slurm/submit_al_dnn.sh;;
    dnn_match_trafo) PER_MODEL_SCRIPT=slurm/submit_al_dnn.sh;;
    deep_gp)     PER_MODEL_SCRIPT=slurm/submit_al_gp_deep.sh;;
    exact_gp)    PER_MODEL_SCRIPT=slurm/submit_al_gp_exact.sh;;
    tabpfn)      PER_MODEL_SCRIPT=slurm/submit_al_tabpfn.sh;;
    *) echo "[error] unknown AL_MODEL: ${AL_MODEL}" >&2; exit 1;;
esac

# ---- Translate warm tag to Click flag ---------------------------------------
case "${AL_WARM}" in
    warm)   WARM_FLAG="--warm-starting";;
    cold)   WARM_FLAG="--no-warm-starting";;
    tabpfn) WARM_FLAG="";;
    *) echo "[error] unknown AL_WARM: ${AL_WARM}" >&2; exit 1;;
esac

# ---- Parse seed list --------------------------------------------------------
IFS=',' read -ra SEEDS_ARR <<< "${AL_SEEDS}"
N_SEEDS=${#SEEDS_ARR[@]}

# Fail loudly if the seed count doesn't match the allocated nodes. This
# catches the --export=...,AL_SEEDS=1,2,3,... comma-truncation footgun: if
# the submit side accidentally passes AL_SEEDS inline through --export,
# only the first seed survives and the job would otherwise silently run
# one seed on a multi-node allocation and return rc=0.
if [[ -n "${SLURM_NNODES:-}" && "${N_SEEDS}" -ne "${SLURM_NNODES}" ]]; then
    echo "[error] AL_SEEDS count (${N_SEEDS}, value='${AL_SEEDS}') != allocated nodes (${SLURM_NNODES})." >&2
    echo "        Did --export= truncate AL_SEEDS at its first comma?" >&2
    echo "        Pre-export AL_SEEDS in the caller shell instead." >&2
    exit 2
fi

echo "=========================================="
echo "[bundle] job_id=${SLURM_JOB_ID} job_name=${SLURM_JOB_NAME}"
echo "[bundle] model=${AL_MODEL} strategy=${AL_STRATEGY} warm=${AL_WARM}"
echo "[bundle] seeds=${AL_SEEDS}  (${N_SEEDS} parallel sruns)"
echo "[bundle] nodes=${SLURM_NNODES:-?} allocated"
echo "[bundle] sweep_id=${AL_SWEEP_ID}"
echo "[bundle] per_model_script=${PER_MODEL_SCRIPT}"
echo "[bundle] started: $(date)"
echo "=========================================="

if [[ "${SLURM_NNODES:-0}" -lt "${N_SEEDS}" ]]; then
    echo "[warn] allocated nodes (${SLURM_NNODES:-?}) < seeds (${N_SEEDS}); " \
         "srun will serialise on the shared nodes" >&2
fi

# Always gpu:2 per srun to match the outer allocation. TabPFN runs AL on
# cuda:0 and Baseline on cuda:1 concurrently via a ThreadPoolExecutor in
# active_learning_tabpfn.py — threads share the parent's CUDA context so
# both GPUs get utilised without the spawn-after-CUDA-init deadlock that an
# earlier mp.Process implementation hit on ROCm/MI300A. Multi-node
# --gres=gpu:1 is rejected on apu, so gpu:2 is both the useful and the only
# allowed choice.
PER_SEED_GRES="gpu:2"

# ---- Optional resume mode -----------------------------------------------------
# With AL_RESUME_TO=<n>, each seed continues its EXISTING per-seed directory
# until it has n iterations in total: the per-seed n_additional is derived from
# that directory's state.pt, and seeds already at >= n are skipped. This keeps
# one bundle job per cell instead of one job per seed (the per-user job limit
# makes 5 single-seed resumes expensive), and it is idempotent, so a second
# resume round can reuse the same command.
if [[ -n "${AL_RESUME_TO:-}" ]]; then
    RESUME_PYTHON="${REPO_ROOT}/.pixi/envs/${PIXI_ENV:-rocm}/bin/python"
    echo "[bundle] resume mode: continuing each seed to ${AL_RESUME_TO} iterations"
fi

_iters_done() {  # $1 = run dir; echoes iteration count, or empty on failure
    "${RESUME_PYTHON}" - "$1" <<'PY' 2>/dev/null
import sys, torch
s = torch.load(f"{sys.argv[1]}/state.pt", weights_only=False, map_location="cpu")
print(len(list(s.get("al_n_train") or [])))
PY
}

# ---- Fork one srun per seed on its own node -----------------------------------
# Each srun inherits the exported AL_OUTPUT_DIR / AL_EXTRA_ARGS set on its line
# (bash `VAR=val cmd &` applies VAR only to that command, per-iteration).
# The per-model submit script then drives `python active_learning*.py` as usual.
for seed in "${SEEDS_ARR[@]}"; do
    per_seed_dir="${AL_OUTPUT_BASE}_seed${seed}_${AL_SWEEP_ID}"
    per_seed_extra="--seed ${seed} --selection-strategy ${AL_STRATEGY}"
    if [[ -n "${WARM_FLAG}" ]]; then
        per_seed_extra="${per_seed_extra} ${WARM_FLAG}"
    fi
    # Preserve any extra args the caller passed (e.g. --use-mcmc-loader).
    if [[ -n "${AL_EXTRA_ARGS_BASE:-}" ]]; then
        per_seed_extra="${per_seed_extra} ${AL_EXTRA_ARGS_BASE}"
    fi

    # Resume mode: derive this seed's n_additional from its own state.pt.
    per_seed_resume_from=""
    per_seed_n_add=""
    if [[ -n "${AL_RESUME_TO:-}" ]]; then
        if [[ ! -f "${per_seed_dir}/state.pt" ]]; then
            echo "[bundle] seed=${seed}: no state.pt in ${per_seed_dir} — skipped" >&2
            continue
        fi
        done_iters="$(_iters_done "${per_seed_dir}")"
        if [[ -z "${done_iters}" ]]; then
            echo "[bundle] seed=${seed}: could not read state.pt — skipped" >&2
            continue
        fi
        if (( done_iters >= AL_RESUME_TO )); then
            echo "[bundle] seed=${seed}: already at ${done_iters} iterations — skipped"
            continue
        fi
        per_seed_resume_from="${per_seed_dir}"
        per_seed_n_add=$(( AL_RESUME_TO - done_iters ))
        echo "[bundle] seed=${seed}: at ${done_iters}, resuming +${per_seed_n_add}"
    fi

    echo "[bundle] launching seed=${seed} -> ${per_seed_dir}"

    AL_OUTPUT_DIR="${per_seed_dir}" \
    AL_EXTRA_ARGS="${per_seed_extra}" \
    AL_RESUME_FROM="${per_seed_resume_from}" \
    AL_N_ADDITIONAL_ITERATIONS="${per_seed_n_add}" \
    srun \
        --nodes=1 \
        --ntasks=1 \
        --exclusive \
        --gres="${PER_SEED_GRES}" \
        --output="logs/al_bundled_${SLURM_JOB_ID}_seed${seed}.out" \
        --error="logs/al_bundled_${SLURM_JOB_ID}_seed${seed}.err" \
        bash "${PER_MODEL_SCRIPT}" &
done

wait
RC=$?

echo "=========================================="
echo "[bundle] all sruns completed: $(date)  rc=${RC}"
echo "=========================================="

# Archive this bundle's runs to non-purged storage (ptmp deletes files
# unaccessed for ~12 weeks). Pattern-scoped to this bundle's own run dirs so
# concurrently finishing bundles never race on each other's tarballs.
bash "${REPO_ROOT}/scripts/archive_runs.sh" \
    "$(dirname "${AL_OUTPUT_BASE}")" \
    /viper/u2/jwuerzin/pmssm-archive/runs \
    "$(basename "${AL_OUTPUT_BASE}")_seed*" \
    || echo "[bundle] WARN: run archiving failed"

exit ${RC}
