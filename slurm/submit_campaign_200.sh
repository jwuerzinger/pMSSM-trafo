#!/bin/bash
# =============================================================================
# The 200-iteration campaign: 32 new bundled cells, 5 seeds each, both targets.
#
# Queues every acquisition arm the head/strategy probes covered, at the full
# five seeds and 200 iterations, on BOTH the relic density (DMRD) and the
# exclusion boundary (ExpR):
#
#   regression   tol_only_random   6 models x 2 targets = 12 cells
#   classification bald            5 models x 2 targets = 10 cells
#   classification cls_entropy     5 models x 2 targets = 10 cells
#
# TabPFN is absent from the classification arms on purpose:
# active_learning_tabpfn.py has no --head and offers neither bald nor
# cls_entropy in its --selection-strategy choices.
#
# entropy_batch is NOT here. Its 200-iteration continuations are the 12 b200*/
# b200e* bundles queued on 2026-08-18, which keep their queue position; the
# chaser (slurm/submit_campaign_chase.sh) tops those cells up too.
#
# This script only STARTS the cells. No fresh run reaches 200 iterations inside
# the 24 h wall clock, so slurm/submit_campaign_chase.sh must be running to
# resume them; submit it right after this.
#
# Why it is a wrapper rather than a loop of sbatch calls: submit_strategy_sweep.sh
# already handles bundling, the per-model resource footprint, dnn_match_trafo's
# architecture override and the manifest rows. What it cannot express is that the
# flag set is not uniform across drivers, which is the entire content below.
#
# Usage:
#   DRY_RUN=1 bash slurm/submit_campaign_200.sh     # print the 32 cells only
#   bash slurm/submit_campaign_200.sh               # submit
#   ARMS=tol_only_random bash slurm/submit_campaign_200.sh   # one arm
#   TARGETS=ExpR bash slurm/submit_campaign_200.sh           # one target
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

N_ITERS="${N_ITERS:-200}"
SEEDS="${SEEDS:-1,2,3,4,5}"
DRY_RUN="${DRY_RUN:-0}"
CAMPAIGN_ID="${CAMPAIGN_ID:-c200_$(date +%Y%m%d_%H%M%S)}"
ARMS="${ARMS:-bald,cls_entropy,tol_only_random}"
TARGETS="${TARGETS:-ExpR,DMRD}"

EXPR_POOL=/ptmp/jwuerzin/data/260804
DMRD_POOL=/ptmp/jwuerzin/data/18387358
EXPR_MANIFEST=/ptmp/jwuerzin/analysis/expr_runs/sweep_manifest.csv
DMRD_MANIFEST=/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv
CELLS_CSV="${CELLS_CSV:-/ptmp/jwuerzin/analysis/campaign/${CAMPAIGN_ID}_cells.csv}"
mkdir -p "$(dirname "${CELLS_CSV}")" logs

# AL_DATA_DIR must not be exported. submit_strategy_sweep.sh submits with
# --export=ALL, so a value set for the ExpR invocations would follow the DMRD
# ones onto the compute node and silently point them at the wrong pool. That is
# the bug that sent six GP jobs to the relic-density pool on 2026-08-21, where
# they died on a missing branch. The pool is passed as --data-dir inside
# EXTRA_AL_ARGS instead, for both targets, where Click's last-wins resolves it
# over whatever default the per-model submit script carries.
unset AL_DATA_DIR

# --n-iterations likewise travels in EXTRA_AL_ARGS rather than AL_N_ITERATIONS:
# submit_al_gp_{exact,deep}.sh:98 hardcode 40 and never read that variable.
#
# Per-driver flag sets. The GP driver has no --target-value and no
# --y-transform (it reads both from TARGET_CONFIG), and TabPFN has no
# --y-transform either; passing the neural spelling to them is an immediate
# "No such option" exit.
common_for () {  # $1 = target, $2 = family (neural|gp|tabpfn)
    local t="$1" fam="$2" pool
    case "${t}" in
        ExpR) pool="${EXPR_POOL}";;
        DMRD) pool="${DMRD_POOL}";;
        *) echo "[error] unknown target ${t}" >&2; exit 1;;
    esac
    # ExpR has no posterior, so its MCMC evaluation split would be scored
    # against the relic-density chains. DMRD keeps its reference.
    local mcmc=""
    [[ "${t}" == "ExpR" ]] && mcmc="--no-mcmc-eval"
    case "${fam}" in
        neural) echo "--target ${t} --target-value $( [[ ${t} == ExpR ]] && echo 1.0 || echo 0.12 ) ${mcmc} --y-transform log --data-dir ${pool} --n-iterations ${N_ITERS}";;
        gp)     echo "--target ${t} ${mcmc} --data-dir ${pool} --n-iterations ${N_ITERS}";;
        tabpfn) echo "--target ${t} --target-value $( [[ ${t} == ExpR ]] && echo 1.0 || echo 0.12 ) ${mcmc} --data-dir ${pool} --n-iterations ${N_ITERS}";;
    esac
}

manifest_for () { [[ "$1" == "ExpR" ]] && echo "${EXPR_MANIFEST}" || echo "${DMRD_MANIFEST}"; }
tag_for ()      { [[ "$1" == "ExpR" ]] && echo "expr" || echo ""; }

N_GROUPS=0
# One invocation per (target, arm, driver family). Every argument that differs
# between drivers is resolved here so nothing downstream has to guess.
run_group () {  # $1 target  $2 arm  $3 family  $4 models  $5 warm  $6 head_args
                #  $7 (optional) output-tag suffix
    local t="$1" arm="$2" fam="$3" models="$4" warm="$5" head_args="$6"
    local suffix="${7:-}"
    [[ ",${TARGETS}," == *",${t},"* ]] || return 0
    [[ ",${ARMS},"    == *",${arm},"* ]] || return 0
    local extra; extra="$(common_for "${t}" "${fam}") ${head_args}"
    # A variant of a driver that already has cells here (lsqgp is exact_gp with a
    # different head) must not reuse the base model's names, or its run
    # directories collide with them and AL_RESUME_TO would extend the wrong
    # thing. The suffix goes through OUTPUT_TAG, which names both the manifest
    # model column and the output dir.
    local otag; otag="$(tag_for "${t}")"
    if [[ -n "${suffix}" ]]; then
        otag="${otag:+${otag}_}${suffix}"
    fi
    # Refuse to resubmit a cell that already has a job. Re-running this script
    # to add ONE new group would otherwise resubmit every earlier group too,
    # and a second bundle against the same run directories races the first on
    # state.pt. (That is exactly what happened when the lsq group was added on
    # 2026-08-21; 32 duplicates had to be cancelled by hand.) Cell names here
    # must match job_name() in scripts/campaign_chase.py.
    local pending=""
    for m in ${models//,/ }; do
        local short jn tt
        case "${arm}" in
            tol_only_random) short=tol;; cls_entropy) short=clsent;;
            bald) short=bald;; *) short="${arm}";;
        esac
        [[ "${t}" == "ExpR" ]] && tt=e || tt=d
        jn="c200_${tt}_${m}${suffix:+_${suffix}}_${short}"
        if squeue -h -u "${USER}" -o "%j" 2>/dev/null | grep -qx "${jn}"; then
            pending="${pending} ${jn}"
        fi
    done
    if [[ -n "${pending}" && "${FORCE_RESUBMIT:-0}" != "1" ]]; then
        echo
        echo "--- [skip] ${t} / ${arm} / ${fam}${suffix:+ [${suffix}]}: already queued:${pending}"
        echo "           (FORCE_RESUBMIT=1 to override, but check for run-dir collisions first)"
        return 0
    fi
    N_GROUPS=$((N_GROUPS + 1))
    echo
    echo "--- [${N_GROUPS}] ${t} / ${arm} / ${fam}: ${models} (${warm})${suffix:+ [${suffix}]}"
    echo "    OUTPUT_TAG:    ${otag:-<none>}"
    echo "    EXTRA_AL_ARGS: ${extra}"
    for m in ${models//,/ }; do
        echo "${t},${m}${suffix:+_${suffix}},${arm},${warm},${CAMPAIGN_ID},$(manifest_for "${t}")" >> "${CELLS_CSV}"
    done
    SWEEP_ID="${CAMPAIGN_ID}" \
    MANIFEST="$(manifest_for "${t}")" \
    OUTPUT_TAG="${otag}" \
    BUNDLE_SEEDS=1 \
    SEEDS="${SEEDS}" \
    MODELS="${models}" \
    STRATEGIES="${arm}" \
    WARM_MODES="${warm}" \
    EXTRA_AL_ARGS="${extra}" \
    DRY_RUN="${DRY_RUN}" \
        bash slurm/submit_strategy_sweep.sh 2>&1 | sed 's/^/    /' \
        | grep -Ev "^\s*\[sweep\] (models|strategies|warm_modes|seeds|dry_run|manifest)=" || true
}

# The verdict head follows the strategy: bald and cls_entropy are both defined
# on a Bernoulli likelihood. The exact GP cannot take a Bernoulli likelihood
# through gpytorch's exact marginal, so its classification arm is the Laplace
# GP classifier (Rasmussen & Williams Algorithm 3.1/3.2) with the optimiser
# budget that mode finding needs.
CLS="--head classification"
CLS_EXACT="--model-type laplace_gpc --head classification --epochs 3000 --patience 200 --learning-rate 1e-2"
# The least-squares alternative of Rasmussen & Williams section 6.5: regress
# +-1 targets under the ordinary Gaussian likelihood and read the verdict off a
# probit. It stays CONJUGATE, so it needs neither laplace_gpc nor the enlarged
# optimiser budget that mode finding does, and it runs on the plain exact GP
# with the production settings. Paired with CLS_EXACT it separates two things
# the Laplace arm changes at once: discretising the target, and replacing the
# likelihood.
CLS_LSQ="--head lsq_classification"
NEURAL=transformer,dnn,dnn_match_trafo

if [[ ! -f "${CELLS_CSV}" ]]; then
    echo "target,model,strategy,warm,campaign_id,manifest" > "${CELLS_CSV}"
fi

echo "=========================================="
echo " 200-iteration campaign"
echo " campaign_id = ${CAMPAIGN_ID}"
echo " seeds       = ${SEEDS}    iterations = ${N_ITERS}"
echo " arms        = ${ARMS}"
echo " targets     = ${TARGETS}"
echo " cells csv   = ${CELLS_CSV}"
echo " dry_run     = ${DRY_RUN}"
echo "=========================================="

# Submission order is the priority order: SLURM breaks priority ties on age, so
# what goes in first starts first. The classification arms are the decisive new
# measurement, tol + uniform is the control that isolates the uncertainty rule,
# and exact_gp goes last because it is the one model whose O(n^3) cost means it
# will not reach 200 whatever it is given.
for t in ExpR DMRD; do
    run_group "${t}" bald        neural "${NEURAL}" cold "${CLS}"
    run_group "${t}" cls_entropy neural "${NEURAL}" cold "${CLS}"
    run_group "${t}" bald        gp     deep_gp     warm "${CLS}"
    run_group "${t}" cls_entropy gp     deep_gp     warm "${CLS}"
done
for t in ExpR DMRD; do
    run_group "${t}" tol_only_random neural "${NEURAL}"     cold ""
    run_group "${t}" tol_only_random gp     deep_gp,exact_gp warm ""
    run_group "${t}" tol_only_random tabpfn tabpfn          tabpfn ""
done
for t in ExpR DMRD; do
    run_group "${t}" bald        gp exact_gp warm "${CLS_EXACT}"
    run_group "${t}" cls_entropy gp exact_gp warm "${CLS_EXACT}"
done
# Last, alongside the Laplace arm it is the control for.
for t in ExpR DMRD; do
    run_group "${t}" bald        gp exact_gp warm "${CLS_LSQ}" lsq
    run_group "${t}" cls_entropy gp exact_gp warm "${CLS_LSQ}" lsq
done

echo
echo "=========================================="
echo " ${N_GROUPS} groups, $(( $(wc -l < "${CELLS_CSV}") - 1 )) cells recorded in ${CELLS_CSV}"
if [[ "${DRY_RUN}" == "1" ]]; then
    echo " DRY RUN: nothing was submitted"
else
    echo " Next: sbatch the chaser, or these cells stop at the 24 h wall clock:"
    echo "   CAMPAIGN_CELLS=${CELLS_CSV} bash slurm/submit_campaign_chase.sh"
fi
echo "=========================================="
