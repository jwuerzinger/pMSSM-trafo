#!/bin/bash
# =============================================================================
# run_analysis.sh — Wrapper for analyse_runs.py
#
# Runs the cross-run quality and diversity analysis for the pMSSM active
# learning pipeline.  Edit the sections below to select which runs to compare,
# then execute from the repo root:
#
#   bash run_analysis.sh
#
# Output (plots + summary.csv) is written to the directory set by OUTPUT_DIR.
#
# Prerequisites
# -------------
# The rocm pixi environment must be installed:
#   pixi install -e rocm
#
# What analyse_runs.py produces
# ------------------------------
#   quality_summary.png       — bar chart: hit rate at 10/20/50 % tolerance
#                               and final R² (val / static-random / MCMC)
#   r2_trajectories.png       — R² vs iteration for every run
#   hit_rate_trajectories.png — hit rate vs iteration for every run
#   relic_density_summary.png — mean ± std of Ω per run (target = 0.12)
#   diversity_summary.png     — mean entropy, k-NN distance, MMD, cluster coverage
#   param_entropy_heatmap.png — per-parameter Shannon entropy per run
#   param_variance_heatmap.png— per-parameter variance per run
#   pairwise_scatter.png      — 2-D parameter projections overlaid with MCMC
#   summary.csv               — one row per run, all scalar metrics ± uncertainties
#
# Key flags
# ---------
#   --run-dirs   DIR [DIR ...]   Paths to run output dirs (must contain state.pt)
#   --labels     NAME [NAME ...] Short names for each run (auto-derived if omitted)
#   --mcmc-data-dir DIR          MCMC ROOT file dir for MMD + cluster coverage
#   --output-dir DIR             Where to write plots and summary.csv
#   --n-bootstrap N              Bootstrap resamples for ±uncertainty (default 500;
#                                set to 0 to skip for a quick preview)
#   --n-permutations N           Permutation-test resamples for MMD CI (default 200)
#   --n-clusters N               K-means clusters for cluster-coverage (default 20)
#   --knn-k N                    Nearest neighbours for k-NN distance (default 5)
#   --include-baseline           Also draw baseline model curves in R² trajectories
#   --seed N                     RNG seed for reproducibility (default 0)
#   --target {DMRD,CrossSection,CLs}  Target variable (default DMRD)
#
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${REPO_ROOT}/.pixi/envs/rocm/bin/python"
SCRIPT="${REPO_ROOT}/analyse_runs.py"
MCMC_DIR="/ptmp/jwuerzin/data/19250082"
OUT_BASE="/ptmp/jwuerzin/analysis"

# =============================================================================
# Completed runs (state.pt present, ≥ 40 iterations)
# Edit these paths if you have newer/different runs.
# =============================================================================

# ── Transformer (MC Dropout) ──────────────────────────────────────────────────
TRANSFORMER_ENTROPY_2GPU="/ptmp/jwuerzin/output/active_learning_output_slurm_20260414_152657"
TRANSFORMER_ENTROPY_1GPU="/ptmp/jwuerzin/output/active_learning_output_slurm_single_20260414_152701"
TRANSFORMER_TOPK_2GPU="/ptmp/jwuerzin/output/active_learning_output_top_k_slurm_20260414_152657"
TRANSFORMER_TOPK_1GPU="/ptmp/jwuerzin/output/active_learning_output_top_k_slurm_single_20260414_152701"

# ── ExactGP ───────────────────────────────────────────────────────────────────
EXACT_GP_ENTROPY_2GPU="/ptmp/jwuerzin/output/active_learning_exact_gp_output_20260415_113302"
EXACT_GP_TOPK_2GPU="/ptmp/jwuerzin/output/active_learning_exact_gp_top_k_output_20260415_133709"
EXACT_GP_TOPK_1GPU="/ptmp/jwuerzin/output/active_learning_exact_gp_top_k_output_single_20260414_183723"

# ── DeepGP ────────────────────────────────────────────────────────────────────
DEEP_GP_ENTROPY_2GPU="/ptmp/jwuerzin/output/active_learning_deep_gp_output_20260415_113302"
DEEP_GP_TOPK_2GPU="/ptmp/jwuerzin/output/active_learning_deep_gp_top_k_output_20260415_133935"
DEEP_GP_TOPK_1GPU="/ptmp/jwuerzin/output/active_learning_deep_gp_top_k_output_single_20260414_183027"

# ── TabPFN ────────────────────────────────────────────────────────────────────
TABPFN_TOPK_2GPU="/ptmp/jwuerzin/output/active_learning_tabpfn_output_slurm_20260414_152657"
TABPFN_TOPK_1GPU="/ptmp/jwuerzin/output/active_learning_tabpfn_output_slurm_single_20260414_152701"

# =============================================================================
# Helper — run the analysis script
# Usage: run_analysis <output_subdir> [extra_flags...] -- DIR LABEL [DIR LABEL ...]
# =============================================================================
run_analysis() {
    local subdir="$1"; shift
    local outdir="${OUT_BASE}/${subdir}"
    mkdir -p "${outdir}"

    # Collect extra flags until "--"
    local flags=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do
        flags+=("$1"); shift
    done
    [[ "$1" == "--" ]] && shift

    # Remaining args are DIR LABEL pairs
    local dirs=() labels=()
    while [[ $# -ge 2 ]]; do
        dirs+=("$1"); labels+=("$2"); shift 2
    done

    echo ""
    echo "══════════════════════════════════════════════════"
    echo " Analysis: ${subdir}"
    echo " Output:   ${outdir}"
    echo " Runs:     ${labels[*]}"
    echo "══════════════════════════════════════════════════"

    "${PYTHON}" "${SCRIPT}" \
        --run-dirs  "${dirs[@]}" \
        --labels    "${labels[@]}" \
        --mcmc-data-dir "${MCMC_DIR}" \
        --output-dir    "${outdir}" \
        "${flags[@]}"
}

# =============================================================================
# Comparison 1: Selection strategy — entropy vs top-k (transformer)
#
# Both runs use the same model (MC Dropout transformer) and the same 2-GPU
# setup, differing only in how candidates are selected each iteration.
# Entropy-batch selects the most collectively informative set; top-k selects
# the individually most uncertain points.
# =============================================================================
run_analysis "transformer_strategy_comparison" \
    --n-bootstrap 500 --n-permutations 200 \
    -- \
    "${TRANSFORMER_ENTROPY_2GPU}" "transformer_entropy" \
    "${TRANSFORMER_TOPK_2GPU}"    "transformer_top_k"

# =============================================================================
# Comparison 2: Model type — transformer vs ExactGP vs DeepGP vs TabPFN
#
# All single-GPU runs so the hardware is comparable.  Compares the effect of
# the surrogate model choice on quality and diversity of selected pMSSM models.
# =============================================================================
run_analysis "model_type_comparison" \
    --n-bootstrap 500 --n-permutations 200 \
    -- \
    "${TRANSFORMER_ENTROPY_1GPU}" "transformer" \
    "${EXACT_GP_TOPK_1GPU}"       "exact_gp" \
    "${DEEP_GP_TOPK_1GPU}"        "deep_gp" \
    "${TABPFN_TOPK_1GPU}"         "tabpfn"

# =============================================================================
# Comparison 2b: Model type — 2-GPU entropy runs
#
# Fairer comparison using the same selection strategy (entropy) for GP models.
# Requires exact_gp_entropy_2gpu to finish (job 8033834, ~32/40 as of writing).
# =============================================================================
run_analysis "model_type_entropy_comparison" \
    --n-bootstrap 500 --n-permutations 200 \
    -- \
    "${TRANSFORMER_ENTROPY_2GPU}" "transformer" \
    "${EXACT_GP_ENTROPY_2GPU}"    "exact_gp" \
    "${DEEP_GP_ENTROPY_2GPU}"     "deep_gp" \
    "${TABPFN_TOPK_2GPU}"         "tabpfn"

# =============================================================================
# Comparison 2c: GP strategy — entropy vs top-k for GP models
#
# Direct strategy comparison within each GP model type.
# =============================================================================
run_analysis "gp_strategy_comparison" \
    --n-bootstrap 500 --n-permutations 200 \
    -- \
    "${EXACT_GP_ENTROPY_2GPU}"  "exact_gp_entropy" \
    "${EXACT_GP_TOPK_2GPU}"     "exact_gp_top_k" \
    "${DEEP_GP_ENTROPY_2GPU}"   "deep_gp_entropy" \
    "${DEEP_GP_TOPK_1GPU}"      "deep_gp_top_k"

# =============================================================================
# Comparison 3: Parallelism — 1-GPU (sequential) vs 2-GPU (parallel)
#
# The 2-GPU setup trains AL and baseline models in parallel, potentially giving
# more iterations within the same walltime.  This comparison checks whether
# parallelism affects the quality or diversity of the resulting dataset.
# =============================================================================
run_analysis "parallelism_comparison" \
    --n-bootstrap 500 --n-permutations 200 --include-baseline \
    -- \
    "${TRANSFORMER_ENTROPY_1GPU}" "transformer_1gpu" \
    "${TRANSFORMER_ENTROPY_2GPU}" "transformer_2gpu" \
    "${EXACT_GP_TOPK_1GPU}"       "exact_gp_1gpu" \
    "${EXACT_GP_TOPK_2GPU}"       "exact_gp_2gpu" \
    "${DEEP_GP_TOPK_1GPU}"        "deep_gp_1gpu" \
    "${DEEP_GP_TOPK_2GPU}"        "deep_gp_2gpu"

# =============================================================================
# Comparison 4: Full cross-run overview — all completed runs
#
# Useful for a broad overview plot.  Bootstrap is reduced to keep runtime
# manageable; increase --n-bootstrap for publication-quality error bars.
# =============================================================================
run_analysis "all_runs" \
    --n-bootstrap 200 --n-permutations 100 \
    -- \
    "${TRANSFORMER_ENTROPY_2GPU}" "transformer_entropy_2gpu" \
    "${TRANSFORMER_TOPK_2GPU}"    "transformer_top_k_2gpu" \
    "${TRANSFORMER_ENTROPY_1GPU}" "transformer_entropy_1gpu" \
    "${TRANSFORMER_TOPK_1GPU}"    "transformer_top_k_1gpu" \
    "${EXACT_GP_ENTROPY_2GPU}"    "exact_gp_entropy_2gpu" \
    "${EXACT_GP_TOPK_2GPU}"       "exact_gp_top_k_2gpu" \
    "${EXACT_GP_TOPK_1GPU}"       "exact_gp_top_k_1gpu" \
    "${DEEP_GP_ENTROPY_2GPU}"     "deep_gp_entropy_2gpu" \
    "${DEEP_GP_TOPK_2GPU}"        "deep_gp_top_k_2gpu" \
    "${DEEP_GP_TOPK_1GPU}"        "deep_gp_top_k_1gpu" \
    "${TABPFN_TOPK_2GPU}"         "tabpfn_top_k_2gpu" \
    "${TABPFN_TOPK_1GPU}"         "tabpfn_top_k_1gpu"

echo ""
echo "All analyses complete.  Results in ${OUT_BASE}/"
