#!/bin/bash
# ==============================================================================
# Slurm job: re-render the classification-accuracy figures over the FULL
# iteration count once the 200-iteration continuation bundles have finished,
# and persist the numbers so the figures can be rebuilt without a GPU.
#
# Why it is gated: the continuation bundles resume the seed-1 runs IN PLACE, so
# the run directories do not change and a manifest built today stays valid; what
# changes is how many iterations each directory holds. Running the accuracy pass
# before they finish would freeze the curves at today's horizon.
#
# What "without a GPU" rests on: the AL driver writes each iteration's accuracy
# into <run_dir>/accuracy_trajectory.json as it runs, so a finished run needs no
# checkpoint reloads. This job harvests those caches, computes only what is
# missing (hence the GPU allocation), and then writes
#   - accuracy_trajectories.json  full [n_seeds, n_iters] matrices per
#                                 (cell, role, dataset), NaN-padded
#   - accuracy_trajectories.csv   long form: mean, SEM, n_seeds per iteration
# into the output dir and copies both into analysis/accuracy/ in the repo, which
# survives scratch cleanup. With those, or with --accuracy-cache-only against
# the run dirs, the figure redraws on any CPU.
#
# Submit (dependency list is built by the caller; afterany so one failed bundle
# does not strand this job forever):
#   sbatch --partition=apu --gres=gpu:1 \
#          --dependency=afterany:<id>:<id>:... slurm/submit_accuracy_after_ext.sh
# ==============================================================================
#SBATCH --job-name=acc_after_ext
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
# apu rejects >1/2 the node's 220G when only 1 of its 2 GPUs is requested.
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/analysis/accuracy"

if [[ -f "${REPO_ROOT}/slurm/cluster.conf" ]]; then
    source "${REPO_ROOT}/slurm/cluster.conf"
fi

export PYTHONUNBUFFERED=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO_ROOT}/al_pmssmwithgp/model:${PYTHONPATH:-}"

# The in-repo .pixi env has been unusable on this project for a while; the
# conda-forge build below is the interpreter every figure in the paper was made
# with. Fall back to .pixi only if it is missing.
PYTHON=/ptmp/jwuerzin/pixi-envs/pytorch-conda-forge-2863954108128992291/envs/rocm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="${REPO_ROOT}/.pixi/envs/${PIXI_ENV:-rocm}/bin/python"
fi

if [[ -n "${ROCR_VISIBLE_DEVICES:-}" || -n "${HIP_VISIBLE_DEVICES:-}" \
      || -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    ACC_DEVICE="cuda:0"
else
    ACC_DEVICE="cpu"
fi

J=/ptmp/jwuerzin/analysis/joint
DP=/ptmp/jwuerzin/data/18387358
EP=/ptmp/jwuerzin/data/260804
# One shared parsed-pool cache. Without it each step writes its own into its
# fresh output dir and re-parses the pool from ROOT first, which on GPFS cost
# ~30 min per exclusion-boundary pass before anything was drawn.
PC=/ptmp/jwuerzin/analysis/pool_cache
STAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo " Job:     ${SLURM_JOB_NAME:-local} | ID: ${SLURM_JOB_ID:-none}"
echo " Node:    $(hostname)"
echo " Started: $(date)"
echo " Python:  ${PYTHON}"
echo " Device:  ${ACC_DEVICE}"
echo "=========================================="

# ---- 1. re-point the joint manifests at whatever is now longest -------------
echo "### 1/4 rebuilding joint manifests"
"${PYTHON}" scripts/build_joint_manifest.py \
    --manifest /ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv \
    --out "${J}/manifest_dmrd.csv" 2>&1 | tail -4
"${PYTHON}" scripts/build_joint_manifest.py \
    --manifest /ptmp/jwuerzin/analysis/expr_runs/sweep_manifest.csv \
    --model-tag expr --out "${J}/manifest_expr.csv" 2>&1 | tail -4

# ---- 2. relic density ------------------------------------------------------
# --min-seeds 1 keeps every iteration any seed reached, which is the point of
# this pass: past the benchmark horizon a cell is one resumed seed.
echo "### 2/4 relic-density accuracy + hit rate over the full horizon"
"${PYTHON}" scripts/plot_hit_rate_trajectories_multiseed.py \
    --manifest "${J}/manifest_dmrd.csv" --output-dir "${J}/dmrd_final" \
    --min-seeds 1 --mark-iteration 40 --pool-cache-dir "${PC}" \
    --baseline-data-dir "${DP}" --tolerances 0.10,0.20,0.50 \
    --compute-accuracy --accuracy-device "${ACC_DEVICE}" 2>&1 | tail -40

# ---- 3. exclusion boundary -------------------------------------------------
# --mcmc-data-dir '' : this target has no sampler reference, so an MCMC accuracy
# panel would be scored against the relic-density posterior.
echo "### 3/4 exclusion-boundary accuracy + hit rate over the full horizon"
"${PYTHON}" scripts/plot_hit_rate_trajectories_multiseed.py \
    --manifest "${J}/manifest_expr.csv" --output-dir "${J}/expr_final" \
    --target ExpR --min-seeds 1 --mark-iteration 40 --pool-cache-dir "${PC}" \
    --baseline-data-dir "${EP}" --no-baseline-require-neutralino-lsp \
    --tolerances 0.10,0.20,0.50 --mcmc-yield-json "" --mcmc-data-dir "" \
    --compute-accuracy --accuracy-device "${ACC_DEVICE}" 2>&1 | tail -40

# ---- 4. persist the numbers ------------------------------------------------
echo "### 4/4 archiving the numeric record"
for t in dmrd expr; do
    for f in accuracy_trajectories.json accuracy_trajectories.csv \
             data_efficiency_best_per_model.json; do
        if [[ -f "${J}/${t}_final/${f}" ]]; then
            cp "${J}/${t}_final/${f}" \
               "${REPO_ROOT}/analysis/accuracy/${t}_${f}"
            echo "  kept analysis/accuracy/${t}_${f}"
        else
            echo "  [warn] missing ${J}/${t}_final/${f}"
        fi
    done
done

# The per-run caches are the upstream source of all of the above. Tar them next
# to the figures so a scratch cleanup of the run dirs is survivable.
ARCH="${J}/accuracy_caches_${STAMP}"
mkdir -p "${ARCH}"
"${PYTHON}" - "${J}/manifest_dmrd.csv" "${J}/manifest_expr.csv" "${ARCH}" <<'PY'
import csv, shutil, sys
from pathlib import Path
*manifests, arch = sys.argv[1:]
arch = Path(arch)
n = 0
for m in manifests:
    for row in csv.DictReader(open(m)):
        d = Path(row.get("expected_run_dir") or "")
        src = d / "accuracy_trajectory.json"
        if src.exists():
            shutil.copy2(src, arch / f"{d.name}.json")
            n += 1
        else:
            print(f"  [warn] no cache: {d.name}")
print(f"  copied {n} per-run accuracy caches")
PY
tar czf "${ARCH}.tar.gz" -C "$(dirname "${ARCH}")" "$(basename "${ARCH}")" \
    && rm -rf "${ARCH}" \
    && echo "  archived ${ARCH}.tar.gz"

echo "=========================================="
echo " Done: $(date)"
echo "=========================================="
