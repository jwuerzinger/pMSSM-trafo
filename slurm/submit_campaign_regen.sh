#!/bin/bash
# =============================================================================
# Recurring re-make of every figure in the paper, both targets, plus the new
# arm figures with seed uncertainty. Self-resubmits every three days until a
# horizon date, then stops.
#
# Why recurring rather than gated on the campaign finishing: 44 cells at 200
# iterations will not all finish, so a job with a dependency on all of them
# might never fire and there would be nothing to come back to. A snapshot every
# three days means the figures on disk are always current for whatever has
# landed, and the last snapshot before the horizon is the one that matters.
#
# Order follows the documented regeneration sequence (slurm/submit_expr_final_regen.sh):
#   0. back up the paper's figures BEFORE anything overwrites them
#   1. refresh the manifests, or every manifest-driven step is blind to whatever
#      finished last
#   2. hit rate FIRST: it writes random_baseline_prevalence.json that later steps
#      divide by, and --compute-accuracy folds the GPU accuracy pass into the
#      same walk over the checkpoints
#   3. the remaining figure families, per target
#   4. the new arm figures, read from the sweep manifests so they carry bands
#   5. copy and rename the renders into the paper's figure directories
#   6. rebuild the PDF
#
# Step 10 of submit_expr_final_regen.sh (the SModelS figure companion) is
# deliberately absent: ALPaper commit b0561f6 deleted pmssm-expr-figures.tex, so
# build_expr_figure_report.py now writes an orphan and its latexmk call fails.
#
# Submit:
#   CAMPAIGN_ID=c200_... REGEN_HORIZON=2026-10-15 \
#   sbatch --partition=apu1 --gres=gpu:1 --mem=110000 slurm/submit_campaign_regen.sh
#
# Env: CAMPAIGN_ID (arm sweep id), REGEN_HORIZON (YYYY-MM-DD),
#      REGEN_EVERY (default 3days), COPY_TO_PAPER=0 to render without touching
#      ALPaper, SKIP_SLOW=1 to drop the UQ/rank/diagnostics families.
# =============================================================================
#SBATCH --job-name=c200_regen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
# Half the node's 220G: apu refuses more when only 1 of its 2 GPUs is requested.
#SBATCH --mem=110000
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs analysis/accuracy

[[ -f "${REPO_ROOT}/slurm/cluster.conf" ]] && source "${REPO_ROOT}/slurm/cluster.conf"
export PYTHONUNBUFFERED=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/scripts:${REPO_ROOT}/al_pmssmwithgp/model:${PYTHONPATH:-}"
PY="${REPO_ROOT}/.pixi/envs/${PIXI_ENV:-rocm}/bin/python"

CAMPAIGN_ID="${CAMPAIGN_ID:-}"
HORIZON="${REGEN_HORIZON:-2026-10-15}"
EVERY="${REGEN_EVERY:-3days}"
COPY_TO_PAPER="${COPY_TO_PAPER:-1}"
SKIP_SLOW="${SKIP_SLOW:-0}"

PAPER=/viper/u2/jwuerzin/ALPaper
ARCHIVE=/viper/u2/jwuerzin/pmssm-archive/plots
# One shared parsed-pool cache across every step. Without it each step re-parses
# ~1500 ROOT files from GPFS before drawing anything, ~30 min per pass.
PC=/ptmp/jwuerzin/analysis/pool_cache
J=/ptmp/jwuerzin/analysis/joint
STAMP=$(date +%Y%m%d)

if [[ -n "${ROCR_VISIBLE_DEVICES:-}${HIP_VISIBLE_DEVICES:-}${CUDA_VISIBLE_DEVICES:-}" ]]; then
    DEV="cuda:0"
else
    DEV="cpu"
fi

echo "=========================================="
echo " campaign regen   job=${SLURM_JOB_ID:-local}  node=$(hostname)"
echo " started = $(date)   device=${DEV}"
echo " campaign_id=${CAMPAIGN_ID:-<none>}  horizon=${HORIZON}  every=${EVERY}"
echo "=========================================="
step () { echo; echo "########## $* ##########"; }

# ---- 0. back up before overwriting ------------------------------------------
# Standing rule: never overwrite a paper figure without a dated copy and a tar
# on non-purged storage first. This job overwrites dozens of them every cycle.
step "0/6 back up the paper's figures"
mkdir -p "${ARCHIVE}"
for d in figures figures_expr; do
    if [[ -d "${PAPER}/${d}" && ! -d "${PAPER}/${d}_backup_${STAMP}" ]]; then
        cp -a "${PAPER}/${d}" "${PAPER}/${d}_backup_${STAMP}"
        echo "  ${PAPER}/${d}_backup_${STAMP}"
    else
        echo "  ${d}: backup for ${STAMP} already exists, kept"
    fi
done
if [[ ! -f "${ARCHIVE}/alpaper-figures-${STAMP}.tar.gz" ]]; then
    tar czf "${ARCHIVE}/alpaper-figures-${STAMP}.tar.gz" -C "${PAPER}" \
        figures figures_expr && echo "  ${ARCHIVE}/alpaper-figures-${STAMP}.tar.gz"
fi

# ---- 1. manifests ------------------------------------------------------------
step "1/6 refresh the manifests"
for M in /ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv \
         /ptmp/jwuerzin/analysis/expr_runs/sweep_manifest.csv; do
    "${PY}" scripts/update_sweep_manifest.py --manifest "${M}" --all 2>&1 | tail -3
done
"${PY}" scripts/build_joint_manifest.py \
    --manifest /ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv \
    --out "${J}/manifest_dmrd.csv" 2>&1 | tail -3
"${PY}" scripts/build_joint_manifest.py \
    --manifest /ptmp/jwuerzin/analysis/expr_runs/sweep_manifest.csv \
    --model-tag expr --out "${J}/manifest_expr.csv" 2>&1 | tail -3

# ---- 1b. benchmark-only manifests --------------------------------------------
# The campaign writes its cells into the SAME sweep manifests the paper's
# figures are built from. Left alone, bald / cls_entropy / tol_only_random would
# appear in Fig. 1, Fig. 2 and every per-strategy panel as soon as their status
# flips to running, silently changing figures whose captions describe five
# strategies. The paper's figures are therefore built from a manifest with the
# campaign rows removed, and the new arms appear only in the arm figures of
# step 4, which are about them.
step "1b/6 split the manifests: benchmark rows only for the paper's figures"
"${PY}" - <<'PYSPLIT'
import csv
from pathlib import Path
for src, dst in (("/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
                  "/ptmp/jwuerzin/analysis/all_runs/sweep_manifest_benchmark.csv"),
                 ("/ptmp/jwuerzin/analysis/expr_runs/sweep_manifest.csv",
                  "/ptmp/jwuerzin/analysis/expr_runs/sweep_manifest_benchmark.csv")):
    if not Path(src).exists():
        continue
    rows = list(csv.DictReader(open(src)))
    keep = [r for r in rows if not str(r.get("sweep_id", "")).startswith("c200_")]
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(keep)
    print(f"  {Path(dst).name}: {len(keep)} of {len(rows)} rows "
          f"({len(rows) - len(keep)} campaign rows held back)")
PYSPLIT

# ---- 2+3. per-target figure families ----------------------------------------
# One function, two targets. Every switch that differs is an argument, so the
# relic-density branch cannot silently inherit an exclusion-boundary default
# (the failure mode that put relic-density data into five ExpR figures).
regen_target () {
    local T="$1" M="$2" O="$3" POOL="$4" TAG="$5"
    shift 5
    local TARGET_ARGS=("$@")

    step "2/6 [${T}] hit rate, hits/desired and verdict accuracy"
    # FIRST: writes random_baseline_prevalence.json that later steps divide by.
    "${PY}" scripts/plot_hit_rate_trajectories_multiseed.py \
        --manifest "${M}" --output-dir "${O}" --pool-cache-dir "${PC}" \
        --baseline-data-dir "${POOL}" --tolerances 0.10,0.20,0.50 \
        --mark-iteration 40 --min-seeds 2 --min-seeds-axis 1 \
        "${TARGET_ARGS[@]}" \
        --compute-accuracy --accuracy-device "${DEV}" 2>&1 | tail -20

    step "3/6 [${T}] per-seed, R2, MSE, compute, coverage, support, inputs"
    "${PY}" scripts/plot_hit_rate_seeds_per_model.py \
        --manifest "${M}" --output-dir "${O}" --baseline-data-dir "${POOL}" \
        --tolerances 0.10,0.20,0.50 $( [[ ${T} == ExpR ]] && echo "--target ExpR" ) \
        2>&1 | tail -3
    "${PY}" scripts/plot_r2_trajectories_multiseed.py \
        --manifest "${M}" --output-dir "${O}" 2>&1 | tail -2
    "${PY}" scripts/plot_mse_trajectories_multiseed.py \
        --manifest "${M}" --output-dir "${O}" --model-tag "${TAG}" 2>&1 | tail -2
    "${PY}" scripts/plot_compute_vs_dataset.py \
        --manifest "${M}" --output-dir "${O}" --model-tag "${TAG}" \
        --baseline-data-dir "${POOL}" \
        $( [[ ${T} == ExpR ]] && echo "--target ExpR --mcmc-data-dir ''" ) \
        2>&1 | tail -3
    "${PY}" scripts/coverage_saturation.py \
        --manifest "${M}" --output-dir "${O}" --cache-dir "${O}" \
        --baseline-data-dir "${POOL}" --support-source pool --model-tag "${TAG}" \
        $( [[ ${T} == ExpR ]] && echo "--target ExpR" ) 2>&1 | tail -3
    for side in in out; do
        "${PY}" scripts/plot_support_efficiency.py \
            --manifest "${M}" --output-dir "${O}" --cache-dir "${O}" \
            --baseline-data-dir "${POOL}" --model-tag "${TAG}" \
            --band-side "${side}" --run-set-label joint \
            $( [[ ${T} == ExpR ]] && echo "--target ExpR" ) 2>&1 | tail -3
    done
    "${PY}" scripts/plot_diversity_vs_budget.py \
        --manifest "${M}" --output-dir "${O}" --model-tag "${TAG}" \
        --run-set-label joint $( [[ ${T} == ExpR ]] && echo "--target ExpR --no-mcmc" ) \
        2>&1 | tail -3
    "${PY}" scripts/plot_al_input_target_diagnostics.py \
        --manifest "${M}" --output-dir "${O}" --cache-dir "${O}" \
        --model-tag "${TAG}" --baseline-data-dir "${POOL}" \
        $( [[ ${T} == ExpR ]] && echo "--target ExpR --mcmc-data-dir ''" ) \
        2>&1 | tail -3
    "${PY}" scripts/plot_pairwise_input_summary.py \
        --manifest "${M}" --output-dir "${O}" \
        $( [[ ${T} == ExpR ]] && echo "--mcmc-data-dir ''" ) 2>&1 | tail -2

    if [[ "${SKIP_SLOW}" != "1" ]]; then
        "${PY}" scripts/compute_yield_comparison.py \
            --manifest "${M}" --output-dir "${O}" --baseline-data-dir "${POOL}" \
            --model-tag "${TAG}" --tolerance 0.10 \
            $( [[ ${T} == ExpR ]] && echo "--target ExpR --mcmc-data-dir '' --no-baseline-require-neutralino-lsp" ) \
            2>&1 | tail -6
        "${PY}" scripts/evaluate_uq.py \
            --manifest "${M}" --output-dir "${O}" --cache-dir "${O}" \
            --baseline-data-dir "${POOL}" --model-tag "${TAG}" \
            $( [[ ${T} == ExpR ]] && echo "--target ExpR --mcmc-data-dir ''" ) \
            2>&1 | tail -6
    fi
}

regen_target DMRD /ptmp/jwuerzin/analysis/all_runs/sweep_manifest_benchmark.csv \
    /ptmp/jwuerzin/analysis/all_runs /ptmp/jwuerzin/data/18387358 ""
# ExpR has no posterior, so its MCMC yield reference and neutralino veto are off:
# the population deliberately keeps non-neutralino LSPs.
regen_target ExpR /ptmp/jwuerzin/analysis/expr_runs/sweep_manifest_benchmark.csv \
    /ptmp/jwuerzin/analysis/expr_runs /ptmp/jwuerzin/data/260804 expr \
    --target ExpR --no-baseline-require-neutralino-lsp \
    --mcmc-yield-json "" --mcmc-data-dir ""

# ---- 4. the new arm figures, with seed bands ---------------------------------
step "4/6 arm comparison figures (seed means and bands)"
ARM_ID_ARGS=()
[[ -n "${CAMPAIGN_ID}" ]] && ARM_ID_ARGS=(--arm-sweep-id "${CAMPAIGN_ID}")
for spec in "ExpR:/ptmp/jwuerzin/analysis/expr_runs/sweep_manifest.csv:${J}/manifest_expr.csv:1.0:/ptmp/jwuerzin/data/260804:expr" \
            "DMRD:/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv:${J}/manifest_dmrd.csv:0.12:/ptmp/jwuerzin/data/18387358:dmrd"; do
    IFS=: read -r T SWEEP JOINT TV POOL SLUG <<< "${spec}"
    OUT="${J}/arms_${SLUG}"
    mkdir -p "${OUT}"
    # The probe glob stays in for ExpR: those single-seed runs are what the
    # figures currently in the paper were made from, so keeping them means the
    # curves extend rather than jump when the campaign's seeds arrive.
    GLOB='/ptmp/jwuerzin/output/headtest_*_20260821_*'
    [[ "${T}" == "DMRD" ]] && GLOB='/ptmp/jwuerzin/output/__none__*'
    "${PY}" scripts/plot_prelim_paper_style.py \
        --headtest-glob "${GLOB}" --exclude-runs '_111422' \
        --manifest "${JOINT}" --arm-manifest "${SWEEP}" "${ARM_ID_ARGS[@]}" \
        --true-value "${TV}" --x-axis both --output-dir "${OUT}" 2>&1 | tail -12
    "${PY}" scripts/plot_prelim_support.py \
        --headtest-glob "${GLOB}" --exclude-runs '_111422' \
        --manifest "${JOINT}" --arm-manifest "${SWEEP}" "${ARM_ID_ARGS[@]}" \
        --pool-dir "${POOL}" --target "${T}" --true-value "${TV}" \
        --n-bins 12 --min-cell 20 --expect-cells 0 \
        --pool-cache-dir "${PC}" --output-dir "${OUT}" 2>&1 | tail -8
done

# ---- 5. into the paper -------------------------------------------------------
step "5/6 copy the renders into the paper"
if [[ "${COPY_TO_PAPER}" == "1" ]]; then
    "${PY}" - "${PAPER}" "${J}" <<'PYCOPY'
"""Copy renders into the paper, refusing to invent or drop figures.

Only basenames the paper ALREADY carries are overwritten. A render with a new
name is reported and left alone: adding it would require a figure environment
that does not exist, and a silent copy would grow the figure directory without
anything referencing it. The `_joint` composites are the exception, because the
paper's name for them differs from the render's.
"""
import shutil, sys
from pathlib import Path
paper, joint = Path(sys.argv[1]), Path(sys.argv[2])
SRC = {"figures": Path("/ptmp/jwuerzin/analysis/all_runs"),
       "figures_expr": Path("/ptmp/jwuerzin/analysis/expr_runs")}
FINAL = {"figures": joint / "arms_dmrd", "figures_expr": joint / "arms_expr"}
# render name -> paper name, for the main-text composites
RENAME = {
    "hit_rate_best_per_model.png": "hit_rate_joint.png",
    "hits_per_desired_best_per_model.png": "hits_per_desired_joint.png",
    "accuracy_best_per_model_static_random.png": "accuracy_joint.png",
    "mse_best_per_model.png": "mse_joint.png",
    "compute_vs_dataset.png": "compute_joint.png",
    "support_efficiency_joint.png": "support_efficiency_joint.png",
    "support_efficiency_joint_offband.png": "support_efficiency_joint_offband.png",
    "coverage_diversity_joint.png": "coverage_diversity_joint.png",
    # the arm figures, from the arms_<target> dirs
    "prelim_hit_rate_vs_size.png": "prelim_arms_hit_rate_size.png",
    "prelim_hits_per_desired_vs_size.png": "prelim_arms_hits_per_desired_size.png",
    "prelim_accuracy_static_random_vs_size.png": "prelim_arms_accuracy_size.png",
    "prelim_support_inband.png": "prelim_arms_support.png",
}
for dest_dir, src_dir in SRC.items():
    dest = paper / dest_dir
    have = {p.name for p in dest.glob("*.png")}
    copied = skipped = unknown = 0
    for src_root in (src_dir, FINAL[dest_dir], src_dir / "al_diag"):
        if not src_root.is_dir():
            continue
        for p in sorted(src_root.glob("*.png")):
            target = RENAME.get(p.name, p.name)
            if target in have:
                shutil.copy2(p, dest / target)
                copied += 1
            elif p.name in RENAME:
                unknown += 1     # a composite the paper does not carry yet
            else:
                skipped += 1
    print(f"  {dest_dir}: {copied} updated, {skipped} renders the paper does not "
          f"use, {unknown} composite(s) not yet referenced")
PYCOPY
else
    echo "  COPY_TO_PAPER=0: renders left on /ptmp"
fi

# ---- 6. rebuild the documents -----------------------------------------------
step "6/6 rebuild the PDFs"
# Paper first: the supplementary reads its .aux through xr for cross-refs.
# -outdir must stay relative; this TeX Live refuses absolute bibtex output paths.
for doc in pmssm-active-learning pmssm-supplementary; do
    ( cd "${PAPER}" && \
      env PATH="/mpcdf/soft/RHEL_9/packages/x86_64/texlive/2021/2021/bin/x86_64-linux:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin" \
      latexmk -synctex=1 -interaction=nonstopmode -file-line-error -pdf \
              -outdir=build "${doc}" 2>&1 \
      | grep -E "Output written|Latexmk: (All|Errors)" | tail -2 )
done

# ---- resubmit ----------------------------------------------------------------
TODAY="$(date +%Y-%m-%d)"
if [[ "${TODAY}" > "${HORIZON}" ]]; then
    echo; echo "[regen] ${TODAY} is past ${HORIZON}; this was the last cycle."
    exit 0
fi
export CAMPAIGN_ID REGEN_HORIZON="${HORIZON}" REGEN_EVERY="${EVERY}"
export COPY_TO_PAPER SKIP_SLOW
NEXT=$(sbatch --parsable "--begin=now+${EVERY}" --partition=apu1 --gres=gpu:1 \
       --mem=110000 --export=ALL slurm/submit_campaign_regen.sh)
if [[ -z "${NEXT}" ]]; then
    echo "[regen] WARNING: resubmission failed; the cycle has STOPPED." >&2
    exit 1
fi
echo; echo "[regen] next cycle queued as ${NEXT}, begins in ${EVERY}"
echo "[regen] done: $(date)"
