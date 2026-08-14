#!/bin/bash
# =============================================================================
# Post-drain regeneration for the ExpR (SModelS exclusion boundary) sweep.
#
# Meant to be submitted with a dependency on every outstanding sweep and probe
# job, so it runs once the cluster work is finished and needs no supervision:
#
#   DEPS=$(squeue -h -u $USER -o "%i" | tr '\n' ':' | sed 's/:$//')
#   sbatch --partition=apu1 --gres=gpu:1 --mem=110000 \
#          --dependency=afterany:${DEPS} --export=ALL \
#          slurm/submit_expr_final_regen.sh
#
# afterany, not afterok: one failed bundle must not leave this queued forever.
#
# Order matters and follows the documented regeneration sequence:
#   1. refresh the manifest, or every manifest-driven step is blind to whatever
#      finished last
#   2. build the probe manifest, or the 160-iteration runs are invisible
#   3. hit rate FIRST, because it writes random_baseline_prevalence.json that
#      later steps divide by; --compute-accuracy folds the GPU accuracy pass
#      into the same walk over the checkpoints
#   4. the remaining figure families
#   5. rebuild the figure-only companion document
#
# Every step is target-aware and passes the r-value pool explicitly. The
# relic-density MCMC yield reference and the neutralino veto are switched off:
# this target has no posterior, and its population deliberately keeps
# non-neutralino LSPs.
# =============================================================================
#SBATCH --job-name=expr_final_regen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=110000
#SBATCH --time=12:00:00
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
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/scripts:${REPO_ROOT}/al_pmssmwithgp/model:${PYTHONPATH:-}"
PY="${REPO_ROOT}/.pixi/envs/${PIXI_ENV:-rocm}/bin/python"

M=/ptmp/jwuerzin/analysis/expr_runs/sweep_manifest.csv
O=/ptmp/jwuerzin/analysis/expr_runs
POOL="${CLUSTER_DATA_DIR:-/ptmp/jwuerzin/data}/260804"
TEX=/viper/u2/jwuerzin/ALPaper/pmssm-expr-figures.tex

echo "=========================================="
echo " ExpR final regeneration"
echo " node=$(hostname)  started=$(date)"
echo " manifest=${M}  pool=${POOL}"
echo "=========================================="

step () { echo; echo "########## $* ##########"; }

step "1/10 refresh the manifest"
"${PY}" scripts/update_sweep_manifest.py --manifest "${M}" --all || true

step "2/10 probe manifest for the 160-iteration runs"
"${PY}" scripts/build_probe_manifest.py --help >/dev/null 2>&1 \
  && "${PY}" scripts/build_probe_manifest.py 2>&1 | tail -5 \
  || echo "[skip] build_probe_manifest.py needs its own arguments; see the README"

step "3/10 hit rate + hits/desired + classification accuracy"
# --mcmc-yield-json '' suppresses the relic-density reference lines; the veto is
# off because this population keeps non-neutralino LSPs by design.
"${PY}" scripts/plot_hit_rate_trajectories_multiseed.py \
    --manifest "${M}" --output-dir "${O}" --target ExpR \
    --baseline-data-dir "${POOL}" --no-baseline-require-neutralino-lsp \
    --tolerances 0.10,0.20,0.50 --mcmc-yield-json "" \
    --compute-accuracy --accuracy-device cuda:0 2>&1 | tail -25

step "4/10 per-seed hit rate"
"${PY}" scripts/plot_hit_rate_seeds_per_model.py \
    --manifest "${M}" --output-dir "${O}" --target ExpR \
    --baseline-data-dir "${POOL}" --tolerances 0.10,0.20,0.50 2>&1 | tail -5

step "5/10 R2 / MSE / compute / coverage / inputs"
"${PY}" scripts/plot_r2_trajectories_multiseed.py --manifest "${M}" --output-dir "${O}" 2>&1 | tail -3
"${PY}" scripts/plot_mse_trajectories_multiseed.py --manifest "${M}" --output-dir "${O}" \
    --model-tag expr 2>&1 | tail -3
"${PY}" scripts/plot_compute_vs_dataset.py --manifest "${M}" --output-dir "${O}" \
    --model-tag expr --target ExpR --baseline-data-dir "${POOL}" 2>&1 | tail -5
"${PY}" scripts/coverage_saturation.py --manifest "${M}" --output-dir "${O}" \
    --cache-dir "${O}" --baseline-data-dir "${POOL}" \
    --target ExpR --support-source pool --model-tag expr 2>&1 | tail -4
# Same support as coverage_saturation, budget axis divided by the pool's own
# size, so the gap to the random-scan curve is the cost ratio. The support build
# is a full pool ingest but it is cached in ${O}, so this is fast on a re-run.
"${PY}" scripts/plot_support_efficiency.py --manifest "${M}" --output-dir "${O}" \
    --cache-dir "${O}" --baseline-data-dir "${POOL}" \
    --target ExpR --model-tag expr --run-set-label 40iter 2>&1 | tail -8
"${PY}" scripts/plot_al_input_target_diagnostics.py --manifest "${M}" --output-dir "${O}" \
    --model-tag expr --target ExpR --mcmc-data-dir "" \
    --baseline-data-dir "${POOL}" --cache-dir "${O}" 2>&1 | tail -5
"${PY}" scripts/plot_pairwise_input_summary.py --manifest "${M}" --output-dir "${O}" \
    --mcmc-data-dir "" 2>&1 | tail -3

step "6/10 N1 composition per best model, plus the random pool"
# The LSP's bino/wino/higgsino mixing fractions per scan. --target is the numeric
# band centre (1.0 = the exclusion boundary) while --target-name is the registry
# key used to read the pool; they are deliberately separate options here.
# --baseline-data-dir supplies the random-pool row, and the emcee row is skipped
# since this target has no posterior. The neutralino veto stays OFF to match the
# population the sweep actually trained on; rows whose LSP is not a neutralino
# carry no mixing fractions and are dropped by classify_lsp_type.
"${PY}" scripts/composition_fractions.py \
    --manifest "${M}" --output-dir "${O}" --cache-dir "${O}" \
    --baseline-data-dir "${POOL}" --mcmc-data-dir "" \
    --target 1.0 --target-name ExpR --tolerance 0.10 \
    --model-tag expr --no-validate-against-spheno 2>&1 | tail -30

step "7/10  R-hat and ESS per AL cell (no reference row)"
# argparse, not click, and it takes explicit --al-picks, so the OUTPUT_TAG'd
# names go in directly. --skip-mcmc drops the reference row: this target has no
# emcee chains, and the reference pass is what dominates runtime anyway.
# CAUTION: never pass a base run and a separate continuation dir as two rows --
# their X overlaps verbatim, so R-hat would compare a run with itself and ESS
# inflates. The 160-iteration probes resume IN PLACE, so each run is one row here.
"${PY}" scripts/mcmc_diagnostics.py --skip-mcmc \
    --al-manifest "${M}" --output-dir "${O}" \
    --al-picks "transformer_expr:entropy_batch:cold,dnn_expr:entropy_batch:cold,deep_gp_expr:entropy_batch:warm,exact_gp_expr:entropy_batch:warm,tabpfn_expr:top_k:tabpfn" \
    2>&1 | tail -25

step "8/10  uncertainty quantification: calibration, NLPD, CRPS, Spearman, AUSE"
# Needs the GPU. Scores the predictive distribution each model's selection
# actually consumed (dropout draws / GP posterior / TabPFN quantiles). Falls back
# to the static random slice alone, since there is no emcee eval set here. This
# is the measurement that speaks to whether MC-dropout uncertainty degrades on
# this harder target: compare Var(z) and the sigma-vs-error ranking against the
# Omega branch's transformer ~7 / deep GP ~1.1.
"${PY}" scripts/evaluate_uq.py \
    --manifest "${M}" --output-dir "${O}" --cache-dir "${O}" \
    --baseline-data-dir "${POOL}" --mcmc-data-dir "" \
    --target ExpR --model-tag expr 2>&1 | tail -25

step "9/10  yield table and per-seed rank uniformity"
# The yield table's emcee column is absent for this target; the random-scan
# multiplier is the meaningful one.
"${PY}" scripts/compute_yield_comparison.py \
    --manifest "${M}" --output-dir "${O}" \
    --baseline-data-dir "${POOL}" --mcmc-data-dir "" \
    --target ExpR --model-tag expr --tolerance 0.10 \
    --no-baseline-require-neutralino-lsp 2>&1 | tail -20
"${PY}" scripts/rank_uniformity_al.py \
    --manifest "${M}" --output-dir "${O}" --model-tag expr 2>&1 | tail -8

step "10/10  rebuild the companion document"
"${PY}" scripts/build_expr_figure_report.py --fig-dir "${O}" --out "${TEX}" \
    --gendate "$(date '+%Y-%m-%d %H:%M %Z')" 2>&1 | tail -3
# Same recipe VSCode uses, so both share one build tree. -outdir must stay
# relative: this TeX Live refuses absolute bibtex output paths.
( cd /viper/u2/jwuerzin/ALPaper && \
  env PATH="/mpcdf/soft/RHEL_9/packages/x86_64/texlive/2021/2021/bin/x86_64-linux:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin" \
  latexmk -synctex=1 -interaction=nonstopmode -file-line-error -pdf \
          -outdir=build pmssm-expr-figures 2>&1 | grep -E "Output written|Latexmk: (All|Errors)" | tail -2 )

step "archive the runs to non-purged storage"
bash "${REPO_ROOT}/scripts/archive_runs.sh" /ptmp/jwuerzin/output \
     /viper/u2/jwuerzin/pmssm-archive/runs "active_learning_*_expr_*" \
     2>&1 | tail -3 || echo "[warn] archiving failed"

echo
echo "=========================================="
echo " done: $(date)"
echo " NOTE: the 160-iteration probe runs share their 40-iteration siblings'"
echo " directories, so any seed-pooling figure above mixes run lengths. Truncate"
echo " to 40 or exclude the probes before quoting cross-model numbers."
echo "=========================================="
