#!/bin/bash
# =============================================================================
# Multi-seed strategy sweep launcher for pMSSM active learning.
#
# Submits the full grid of (model x strategy x warm/cold x seed) jobs and
# appends one row per job to the sweep manifest. Lets downstream bookkeeping
# scripts map (config, seed) -> (job_id, run_dir, status).
#
# Grid (per user decisions, 2026-04-22):
#   models     : transformer, deep_gp, exact_gp, tabpfn
#   strategies : top_k, top_k_tol_only, entropy_batch
#              : tabpfn SKIPS entropy_batch (prohibitively expensive)
#   warm_modes : warm, cold
#              : tabpfn has NO warm-start axis (warm_tag="tabpfn")
#   seeds      : 1,2,3,4,5
#
# Full sweep size: 3 models x 3 strategies x 2 warm x 5 seeds
#                  + tabpfn: 2 strategies x 1 "warm" x 5 seeds  = 100 jobs
#
# Env overrides (all optional):
#   SEEDS       (default "1,2,3,4,5")
#   MODELS      (default "transformer,deep_gp,exact_gp,tabpfn")
#   STRATEGIES  (default "top_k,top_k_tol_only,entropy_batch"; tabpfn still
#                auto-skips entropy_batch unless TABPFN_ALLOW_ENTROPY=1)
#   WARM_MODES  (default "warm,cold"; tabpfn always uses "tabpfn" sentinel)
#   DRY_RUN     (1 = do not sbatch, just preview and append DRY rows)
#   MANIFEST    (default /ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv)
#   SUBMIT_SLEEP_SEC (default 0; optional throttle between sbatch calls)
#
# Usage:
#   bash slurm/submit_strategy_sweep.sh                     # full sweep
#   DRY_RUN=1 bash slurm/submit_strategy_sweep.sh           # dry-run
#   MODELS=transformer STRATEGIES=top_k_tol_only WARM_MODES=warm SEEDS=1 \
#       bash slurm/submit_strategy_sweep.sh                 # 1 job
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

MANIFEST="${MANIFEST:-/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv}"
mkdir -p "$(dirname "${MANIFEST}")"
if [[ ! -f "${MANIFEST}" ]]; then
    echo "sweep_id,submit_time,model,strategy,warm_start,seed,job_id,expected_run_dir,status,slurm_log" > "${MANIFEST}"
    echo "[manifest] created ${MANIFEST}"
fi

# ---- Cluster config ----------------------------------------------------------
if [[ -f "${REPO_ROOT}/slurm/cluster.conf" ]]; then
    source "${REPO_ROOT}/slurm/cluster.conf"
else
    echo "[warn] slurm/cluster.conf not found — using defaults (apu, gpu:2, rocm)"
    CLUSTER_PARTITION="${CLUSTER_PARTITION:-apu}"
    CLUSTER_GPU_GRES_2="${CLUSTER_GPU_GRES_2:-gpu:2}"
    CLUSTER_GPU_GRES_1="${CLUSTER_GPU_GRES_1:-gpu:1}"
fi

SWEEP_ID="$(date +%Y%m%d_%H%M%S)"
SEEDS="${SEEDS:-1,2,3,4,5}"
MODELS="${MODELS:-transformer,deep_gp,exact_gp,tabpfn}"
STRATEGIES="${STRATEGIES:-top_k,top_k_tol_only,entropy_batch}"
WARM_MODES="${WARM_MODES:-warm,cold}"
SUBMIT_SLEEP_SEC="${SUBMIT_SLEEP_SEC:-0}"
DRY_RUN="${DRY_RUN:-0}"
TABPFN_ALLOW_ENTROPY="${TABPFN_ALLOW_ENTROPY:-0}"

echo "[sweep] sweep_id=${SWEEP_ID}"
echo "[sweep] models=${MODELS}"
echo "[sweep] strategies=${STRATEGIES}"
echo "[sweep] warm_modes=${WARM_MODES}"
echo "[sweep] seeds=${SEEDS}"
echo "[sweep] dry_run=${DRY_RUN}"
echo "[sweep] manifest=${MANIFEST}"

N_SUBMITTED=0
N_SKIPPED=0

for model in ${MODELS//,/ }; do
    case "${model}" in
        transformer) submit_script="slurm/submit_al_transformer.sh"; model_tag="transformer";;
        deep_gp)     submit_script="slurm/submit_al_gp_deep.sh";     model_tag="deep_gp";;
        exact_gp)    submit_script="slurm/submit_al_gp_exact.sh";    model_tag="exact_gp";;
        tabpfn)      submit_script="slurm/submit_al_tabpfn.sh";      model_tag="tabpfn";;
        *) echo "[error] unknown model: ${model}"; exit 1;;
    esac

    # Per-model axis gating
    if [[ "${model}" == "tabpfn" ]]; then
        model_warm_modes="tabpfn"   # sentinel: no real warm axis
    else
        model_warm_modes="${WARM_MODES}"
    fi

    for strategy in ${STRATEGIES//,/ }; do
        # Skip entropy_batch for tabpfn (very slow) unless user opts in
        if [[ "${model}" == "tabpfn" && "${strategy}" == "entropy_batch" && "${TABPFN_ALLOW_ENTROPY}" != "1" ]]; then
            echo "[skip] ${model_tag} + entropy_batch (set TABPFN_ALLOW_ENTROPY=1 to include)"
            continue
        fi

        for warm in ${model_warm_modes//,/ }; do
            # Translate warm tag to Click flag
            case "${warm}" in
                warm)   WARM_FLAG="--warm-starting";;
                cold)   WARM_FLAG="--no-warm-starting";;
                tabpfn) WARM_FLAG="";;
                *) echo "[error] unknown warm mode: ${warm}"; exit 1;;
            esac

            for seed in ${SEEDS//,/ }; do
                # Dir name uses the shared sweep timestamp; seed already makes
                # the path unique per (config, seed). submit_time below captures
                # the actual per-job sbatch time for the manifest.
                submit_time="$(date +%Y%m%d_%H%M%S)"
                expected_dir="/ptmp/jwuerzin/output/active_learning_${model_tag}_${strategy}_${warm}_seed${seed}_${SWEEP_ID}"

                # Assemble the args we inject. AL_EXTRA_ARGS is spliced at the
                # very end of each per-model submit script, so last-wins Click
                # parsing lets our --selection-strategy / warm-start override
                # whatever the script body hardcodes.
                extra_args="--seed ${seed} --selection-strategy ${strategy}"
                if [[ -n "${WARM_FLAG}" ]]; then
                    extra_args="${extra_args} ${WARM_FLAG}"
                fi

                # TabPFN driver also reads AL_SELECTION_STRATEGY; align it.
                tabpfn_strategy_env=""
                if [[ "${model}" == "tabpfn" ]]; then
                    tabpfn_strategy_env=",AL_SELECTION_STRATEGY=${strategy}"
                fi

                # Choose GRES (tabpfn runs on 1 GPU; others on 2)
                if [[ "${model}" == "tabpfn" ]]; then
                    gres="${CLUSTER_GPU_GRES_1:-gpu:1}"
                else
                    gres="${CLUSTER_GPU_GRES_2:-gpu:2}"
                fi

                if [[ "${DRY_RUN}" == "1" ]]; then
                    JOB_ID="DRY"
                    echo "[dry-run] ${submit_script} -> ${expected_dir}"
                    echo "          extra_args='${extra_args}'  gres=${gres}"
                else
                    JOB_ID=$(sbatch --parsable \
                        --partition="${CLUSTER_PARTITION}" \
                        --gres="${gres}" \
                        --export=ALL,AL_OUTPUT_DIR="${expected_dir}",AL_EXTRA_ARGS="${extra_args}"${tabpfn_strategy_env} \
                        "${submit_script}")
                    echo "[submitted] ${JOB_ID}  ${model_tag}/${strategy}/${warm}/seed${seed}"
                fi

                slurm_log="logs/al_${model_tag}_${JOB_ID}.out"
                echo "${SWEEP_ID},${submit_time},${model_tag},${strategy},${warm},${seed},${JOB_ID},${expected_dir},submitted,${slurm_log}" >> "${MANIFEST}"
                N_SUBMITTED=$((N_SUBMITTED + 1))

                sleep "${SUBMIT_SLEEP_SEC}"
            done
        done
    done
done

echo "=========================================="
echo "[sweep] submitted: ${N_SUBMITTED}  skipped: ${N_SKIPPED}"
echo "[sweep] manifest: ${MANIFEST}"
echo "=========================================="
