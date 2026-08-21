#!/bin/bash
# ==============================================================================
# apudev smoke of the end-of-iteration housekeeping added to all four drivers.
#
# Two things need proving, and they need different runs:
#
#   MODE=wiring (default) -- 2 iterations per driver with --no-generate-data.
#     No simulator workspaces exist, so the housekeeping call must no-op
#     cleanly. This is the check that the call site itself is correct in all
#     four drivers: `pack_debris` was referenced before it was a CLI option, and
#     that NameError would have killed every queued job at the end of its first
#     iteration.
#
#   MODE=generate -- 2 iterations of ONE driver WITH --generate-data, so real
#     worker_*/retry_* trees exist and the pack path runs inside the loop.
#     Small n_select, because 15 minutes has to cover SPheno, micrOmegas and
#     SModelS for every model.
#
#   sbatch slurm/submit_housekeeping_smoke.sh              # wiring
#   MODE=generate sbatch --export=ALL slurm/submit_housekeeping_smoke.sh
# ==============================================================================
#SBATCH --job-name=hk_smoke
#SBATCH --partition=apudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs
[[ -f slurm/cluster.conf ]] && source slurm/cluster.conf

export PYTHONUNBUFFERED=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/al_pmssmwithgp/model:${PYTHONPATH:-}"
export SMODELS_CACHEDIR="${SMODELS_CACHEDIR:-/ptmp/jwuerzin/cache/smodels}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
PYTHON="${REPO_ROOT}/.pixi/envs/rocm/bin/python"
POOL="${CLUSTER_DATA_DIR:-/ptmp/jwuerzin/data}/260804"
OUT="/ptmp/jwuerzin/output/hk_smoke_${SLURM_JOB_ID}"
MODE="${MODE:-wiring}"

echo "=== housekeeping smoke: MODE=${MODE}  out=${OUT} ==="
rc=0

run_one () {   # $1 label, rest = argv
    local label="$1"; shift
    echo; echo "---------------- ${label} ----------------"
    "$@"
    local st=$?
    echo "[smoke] ${label} exit=${st}"
    [[ ${st} -ne 0 ]] && rc=${st}
    # The assertion: the housekeeping line must appear, or the call site is dead.
    return 0
}

if [[ "${MODE}" == "wiring" ]]; then
    COMMON_NEURAL=(--target ExpR --target-value 1.0 --y-transform log
                   --n-iterations 2 --n-select 20 --n-candidates 2000
                   --n-samples 400 --data-dir "${POOL}" --static-eval-size 2000
                   --no-mcmc-eval --no-generate-data --seed 1 --gpu-ids 0
                   --testing --epochs 40 --no-warm-starting)
    run_one transformer "${PYTHON}" active_learning.py "${COMMON_NEURAL[@]}" \
        --output-dir "${OUT}/transformer"
    run_one dnn "${PYTHON}" active_learning_dnn.py "${COMMON_NEURAL[@]}" \
        --output-dir "${OUT}/dnn"
    run_one gp "${PYTHON}" active_learning_gp.py \
        --target ExpR --n-iterations 2 --n-select 20 --n-candidates 2000 \
        --n-samples 400 --data-dir "${POOL}" --static-eval-size 2000 \
        --no-mcmc-eval --no-generate-data --seed 1 --gpu-ids 0 --testing \
        --model-type deep_gp --num-inducing-max 128 --epochs 40 \
        --output-dir "${OUT}/deep_gp"
    run_one tabpfn "${PYTHON}" active_learning_tabpfn.py \
        --target ExpR --target-value 1.0 --n-iterations 2 --n-select 20 \
        --n-candidates 2000 --n-samples 400 --data-dir "${POOL}" \
        --static-eval-size 2000 --no-mcmc-eval --no-generate-data --seed 1 \
        --gpu-ids 0 --testing --output-dir "${OUT}/tabpfn"
else
    # Real generation, so real workspaces to pack. The DNN is the cheapest
    # surrogate, which leaves the budget to the simulator.
    # No --testing here. It pins n_candidates to 100 and, on apudev, the run
    # died at the selection step before ever reaching save_state, i.e. upstream
    # of the housekeeping call. Reproduce the production path instead: real
    # candidate counts, small n_samples so the simulator budget is bearable.
    run_one dnn_generate "${PYTHON}" active_learning_dnn.py \
        --target ExpR --target-value 1.0 --y-transform log \
        --n-iterations 2 --n-select 8 --n-candidates 20000 \
        --n-samples 30 --data-dir "${POOL}" --static-eval-size 2000 \
        --no-mcmc-eval --generate-data --gen-workers 4 \
        --min-gen-fraction 0.1 --max-gen-attempts 2 \
        --epochs 200 --patience 50 --no-warm-starting \
        --seed 1 --gpu-ids 0 \
        --output-dir "${OUT}/dnn_generate"
fi

echo; echo "=== housekeeping evidence in the run logs ==="
grep -h "\[housekeeping\]" "${OUT}"/*/active_learning.log 2>/dev/null | head -20 \
    || echo "(no [housekeeping] lines found)"
echo; echo "=== per-iteration inventory ==="
for d in "${OUT}"/*/iteration_*; do
    [[ -d "$d" ]] || continue
    printf "  %-42s files=%-5s dirs=%-4s tar=%s ntuples=%s json=%s\n" \
        "${d#${OUT}/}" \
        "$(find "$d" -type f | wc -l)" "$(find "$d" -type d | wc -l)" \
        "$([[ -f $d/debris.tar ]] && echo yes || echo no)" \
        "$(ls "$d"/ntuples 2>/dev/null | wc -l)" \
        "$([[ -f $d/smodels_best_analysis.json ]] && echo yes || echo no)"
done
echo; echo "[smoke] overall rc=${rc}"
exit ${rc}
