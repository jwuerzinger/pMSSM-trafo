#!/bin/bash
# =============================================================================
# Re-run the ExpR steps that the 2026-08-16 regeneration got wrong or skipped.
#
#   sbatch --partition=apu1 --gres=gpu:1 --mem=110000 --export=ALL \
#          slurm/submit_expr_fix_regen.sh
#
# What was wrong, and why each step is here:
#
#  1. The accuracy pass built its MCMC eval set from the RELIC-DENSITY posterior
#     for every target: the load call omitted `target` and --mcmc-data-dir
#     defaults to neutralino_v4. It therefore scored r-surrogates against Omega
#     labels and wrote a plausible-looking accuracy_best_per_model_mcmc.png.
#     Fixed in the plotter (it now refuses by registry), so the accuracy pass is
#     re-run to regenerate the static-random, train and val panels cleanly.
#  2. evaluate_uq never ran: --model-tag was declared as an option but missing
#     from main()'s signature, so click raised TypeError. This is the measurement
#     that speaks to whether MC-dropout uncertainty degrades on this target.
#  3. rank_uniformity_al needs emcee_diagnostics, which lives in the
#     Run3ModelGen environment, not the torch one. Two passes: --export-only
#     under torch to build the chain caches, then the analysis under R3MG.
#
# The support-efficiency, pairwise and yield steps were already fixed and re-run
# interactively, so they are not repeated here.
# =============================================================================
#SBATCH --job-name=expr_fix_regen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=110000
#SBATCH --time=08:00:00
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
PY_R3MG="${REPO_ROOT}/Run3ModelGen/.pixi/envs/default/bin/python"

M=/ptmp/jwuerzin/analysis/expr_runs/sweep_manifest.csv
O=/ptmp/jwuerzin/analysis/expr_runs
POOL="${CLUSTER_DATA_DIR:-/ptmp/jwuerzin/data}/260804"
TEX=/viper/u2/jwuerzin/ALPaper/pmssm-expr-figures.tex

echo "=========================================="
echo " ExpR fix-up regeneration"
echo " node=$(hostname)  started=$(date)"
echo "=========================================="
step () { echo; echo "########## $* ##########"; }

step "1/4 accuracy pass (MCMC panel now refused by registry, not by luck)"
# --mcmc-yield-json '' keeps the relic-density reference lines off the hit-rate
# panels. --mcmc-data-dir is left at its default deliberately: the point of the
# fix is that the plotter now IGNORES it for a target with no posterior and says
# so in the log, rather than silently loading Omega. Expect a line reading
# "target 'ExpR' has no emcee reference; skipping the MCMC panel".
"${PY}" scripts/plot_hit_rate_trajectories_multiseed.py \
    --manifest "${M}" --output-dir "${O}" --target ExpR \
    --baseline-data-dir "${POOL}" --no-baseline-require-neutralino-lsp \
    --tolerances 0.10,0.20,0.50 --mcmc-yield-json "" \
    --compute-accuracy --accuracy-device cuda:0 2>&1 | tail -30

step "2/4 uncertainty quantification"
"${PY}" scripts/evaluate_uq.py \
    --manifest "${M}" --output-dir "${O}" --cache-dir "${O}" \
    --baseline-data-dir "${POOL}" --mcmc-data-dir "" \
    --target ExpR --model-tag expr 2>&1 | tail -30

step "3/4 rank uniformity: export the caches under torch, analyse under R3MG"
"${PY}" scripts/rank_uniformity_al.py --export-only \
    --manifest "${M}" --output-dir "${O}" --model-tag expr 2>&1 | tail -6
if [[ -x "${PY_R3MG}" ]]; then
    "${PY_R3MG}" scripts/rank_uniformity_al.py \
        --manifest "${M}" --output-dir "${O}" --model-tag expr 2>&1 | tail -10
else
    echo "[skip] ${PY_R3MG} not executable; rank uniformity analysis not run"
fi

step "4/4 rebuild the companion document"
"${PY}" scripts/build_expr_figure_report.py --fig-dir "${O}" --out "${TEX}" \
    --gendate "$(date '+%Y-%m-%d %H:%M %Z')" 2>&1 | tail -3
( cd /viper/u2/jwuerzin/ALPaper && \
  env PATH="/mpcdf/soft/RHEL_9/packages/x86_64/texlive/2021/2021/bin/x86_64-linux:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin" \
  latexmk -synctex=1 -interaction=nonstopmode -file-line-error -pdf \
          -outdir=build pmssm-expr-figures 2>&1 | grep -E "Output written|Latexmk: (All|Errors)" | tail -2 )

echo
echo "=========================================="
echo " done: $(date)"
echo "=========================================="
