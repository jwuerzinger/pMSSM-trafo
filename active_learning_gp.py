"""
Active Learning pipeline for pMSSM relic density prediction using Gaussian Process models.

Uses ExactGP, DeepGP, SparseGP, or MLP models from the al_pmssmwithgp submodule with
native GP posterior variance for uncertainty estimation.

Harmonized with active_learning.py to use the unified pmssm package.
"""

import warnings
import random
import re
warnings.filterwarnings('ignore', message='.*enable_nested_tensor.*')

import sys
from pathlib import Path

# Add gp_pipeline to import path
_GP_PIPELINE_ROOT = Path(__file__).parent / "al_pmssmwithgp" / "model"
sys.path.insert(0, str(_GP_PIPELINE_ROOT))

# Import from unified pmssm package
from pmssm import (
    # Configuration
    PARAM_ORDER,
    PARAM_RANGES,
    CSV_TO_MODELGEN,
    TARGET_CONFIG,
    GP_RANGE_DICT,
    # Data operations
    load_pmssm_data,
    load_mcmc_data,
    build_norm_tensors,
    normalize_x,
    unnormalize_x,
    transform_y,
    split_mcmc_for_oracle,
    # Selection
    generate_candidate_pool,
    select_top_uncertain,
    select_top_uncertain_filtered,
    select_top_uncertain_tol_only,
    # Uncertainty
    compute_uncertainty_gp,
    # Training
    create_gp_model,
    train_gp_worker,
    model_has_likelihood,
    # Evaluation
    compute_gp_r2,
    compute_comprehensive_metrics,
    # Visualization
    plot_data_histograms,
    plot_parallel_coordinates,
    plot_candidate_uncertainty,
    plot_iteration_metrics,
    plot_advanced_diagnostics,
    plot_eval_scatterplots,
    pick_representative_points,
    plot_representative_trajectories,
    # Logging
    setup_logging,
    # Model generation
    generate_models_from_csv,
    load_generated_data,
    save_selected_points,
    load_true_eval_dataset,
    # Classification accuracy capture
    binary_accuracy,
    write_iter_accuracies,
)

from datetime import datetime
import logging
import structlog
import json
import multiprocessing as mp

import click
import numpy as np
import pandas as pd
import torch
import gpytorch

import yaml
from itertools import product as iterproduct

# Import al_pmssmwithgp models and utilities
from gp_pipeline.models.exact_gp import ExactGP
from gp_pipeline.models.deep_gp import DeepGP
from gp_pipeline.models.sparse_gp import SparseGP
from gp_pipeline.models.mlp import MLP
from gp_pipeline.utils.selection import EntropySelectionStrategy
from gp_pipeline.utils.evaluation import (
    compute_accuracy,
    compute_gof_metrics,
    compute_weighted_accuracy,
    misclassified,
)


# All configuration, data processing, training, evaluation, and visualization
# functions are now imported from the unified pmssm package above


def cross_evaluate_gp(model, X_other, Y_other, data_min, data_max, model_type,
                      jitter=1e-3, num_samples=8, target='DMRD',
                      return_predictions=False):
    """Evaluate a trained GP model on an arbitrary dataset.

    Returns (mse_loss, r2) where mse_loss is in transformed space
    and r2 is in physical space (matching the regular metric computations).
    If return_predictions=True, returns (mse_loss, r2, y_true_phys, y_pred_phys).
    """
    from pmssm.visualization import gp_predict
    from pmssm.data import inverse_transform_y

    x_norm = normalize_x(X_other, data_min, data_max)
    y_t = transform_y(Y_other, target=target).view(-1)

    y_pred_t = gp_predict(model, x_norm, model_type, jitter=jitter, num_samples=num_samples)

    # MSE in transformed space (same space as training loss)
    mse = ((y_t - y_pred_t.cpu()) ** 2).mean().item()

    # R² in physical space (same as regular R² computation)
    y_true_phys = inverse_transform_y(y_t.cpu(), target=target)
    y_pred_phys = inverse_transform_y(y_pred_t.cpu(), target=target)

    ss_res = ((y_true_phys - y_pred_phys) ** 2).sum()
    ss_tot = ((y_true_phys - y_true_phys.mean()) ** 2).sum()
    r2 = (1 - (ss_res / ss_tot)).item()
    if return_predictions:
        return mse, r2, y_true_phys.squeeze(), y_pred_phys.squeeze()
    return mse, r2


# ---------------------------------------------------------------------------
# Entropy-based batch active learning selection (Phase 6)
# ---------------------------------------------------------------------------

def select_entropy_batch(model, X_candidates_norm, n_select, model_type,
                         threshold=0.0, blur=0.15, beta=50.0,
                         tolerance_sampling=1.0, proximity_sampling=0.1,
                         use_dkl=False, jitter=1e-3, num_samples=8,
                         entropy_pool_size=10_000,
                         device=None, logger=None):
    """
    Entropy-based batch active learning selection.

    Adapts EntropySelectionStrategy.select_new_points from al_pmssmwithgp
    for the main repo's data flow (explicit model + args instead of pipeline object).

    Args:
        model: Trained GP model (ExactGP, DeepGP, or SparseGP). Not MLP.
        X_candidates_norm: Normalized candidate pool (N, D) on device.
        n_select: Number of points to select.
        model_type: Model type string.
        threshold: Decision threshold in transformed space.
        blur: Entropy smoothing parameter.
        beta: Gibbs sampling temperature (high=deterministic, low=random).
        tolerance_sampling: Threshold proximity filter width (0 to skip filtering).
        proximity_sampling: Gaussian proximity weighting width.
        use_dkl: Whether model uses Deep Kernel Learning.
        jitter: Cholesky jitter.
        num_samples: Likelihood samples for DeepGP.
        device: Torch device.
        logger: Logger instance.

    Returns:
        selected_X: Selected points from X_candidates_norm (n_select, D).
        per_point_entropy: Entropy scores for each selected point.
        mean_all: Mean predictions over the full candidate pool (N,).
        var_all: Variance predictions over the full candidate pool (N,).
    """
    strategy = EntropySelectionStrategy(
        blur=blur, beta=beta,
        tolerance_sampling=tolerance_sampling,
        proximity_sampling=proximity_sampling,
    )

    if device is None:
        device = next(model.parameters()).device
    thr = torch.tensor([threshold], device=device)

    n_pool = 1000 if use_dkl else entropy_pool_size
    is_deep = model_type == "deep_gp"
    ns = 1 if not is_deep else num_samples

    model.eval()
    if model_has_likelihood(model_type):
        model.likelihood.eval()

    # Evaluate mean and variance on the passed-in candidates
    if logger:
        logger.info(f"Entropy selection: evaluating {len(X_candidates_norm)} candidates...")

    batch_size = 10_000 if is_deep else 5_000
    means_list, vars_list = [], []
    for i in range(0, len(X_candidates_norm), batch_size):
        x_batch = X_candidates_norm[i:i + batch_size].to(device)
        with torch.no_grad(), \
             gpytorch.settings.eval_cg_tolerance(1e-4), \
             gpytorch.settings.max_cg_iterations(300), \
             gpytorch.settings.fast_pred_var(False), \
             gpytorch.settings.fast_pred_samples(True), \
             gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter), \
             gpytorch.settings.num_likelihood_samples(ns):
            preds = model.likelihood(model(x_batch))
            m = preds.mean.detach()
            v = preds.variance.detach()
            if is_deep:
                means_list.append(m.mean(dim=0).squeeze())
                vars_list.append(v.mean(dim=0))
            else:
                means_list.append(m)
                vars_list.append(v)

    mean_all = torch.cat(means_list)
    var_all = torch.cat(vars_list)

    if logger:
        logger.info(f"Candidate variance stats: mean={var_all.mean():.6f}, "
                    f"max={var_all.max():.6f}, min={var_all.min():.6f}, "
                    f"std_of_var={var_all.std():.6f}")

    # Step 1: Hard tolerance cut (if enabled)
    if tolerance_sampling != 0:
        mask = (mean_all > thr - tolerance_sampling) & (mean_all < thr + tolerance_sampling)
        surviving_indices = torch.where(mask)[0]
        if logger:
            logger.info(f"Tolerance filter (±{tolerance_sampling:.1f}): "
                       f"{len(surviving_indices)}/{len(mean_all)} candidates survive")
        if len(surviving_indices) == 0:
            if logger:
                logger.warning("No candidates survived tolerance filter, using all candidates")
            surviving_indices = torch.arange(len(mean_all), device=device)
    else:
        surviving_indices = torch.arange(len(mean_all), device=device)

    # Step 2: Proximity-weighted variance ranking on survivors
    surv_mean = mean_all[surviving_indices]
    surv_var = var_all[surviving_indices]

    if proximity_sampling != 0:
        proximity = torch.exp(-((surv_mean - thr) ** 2) / proximity_sampling)
        entropy_score = proximity * surv_var
    else:
        entropy_score = surv_var

    k = min(n_pool, len(surviving_indices))
    topk = torch.topk(entropy_score, k=k, largest=True)
    pool_indices = surviving_indices[topk.indices]
    x_pool = X_candidates_norm[pool_indices].to(device)

    if logger:
        logger.info(f"Focused pool: {len(x_pool)} candidates")

    # Get full covariance matrix on focused pool
    if logger:
        logger.info("Computing covariance on focused pool...")

    with torch.no_grad(), \
         gpytorch.settings.eval_cg_tolerance(1e-4), \
         gpytorch.settings.max_cg_iterations(300), \
         gpytorch.settings.fast_pred_var(False), \
         gpytorch.settings.fast_pred_samples(True), \
         gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter), \
         gpytorch.settings.num_likelihood_samples(ns):
        preds_focus = model.likelihood(model(x_pool))
        if is_deep:
            mean = preds_focus.mean.detach().mean(dim=0).reshape(-1)
            covar = preds_focus.covariance_matrix.detach().mean(dim=0)
        else:
            mean = preds_focus.mean.detach()
            covar = preds_focus.covariance_matrix.detach()

    # Iterative batch entropy selection
    score_function = strategy.smoothed_batch_entropy(blur=blur, device=device)
    choice_function = lambda score, indices: strategy.gibbs_sample(score, beta, device)

    selected_indices = strategy.iterative_batch_selector(
        score_function, choice_function, mean - thr, covar, n_select, device
    )

    # Compute per-point entropy
    sel_mean = mean[selected_indices]
    sel_covar = covar[selected_indices][:, selected_indices]
    per_point_entropy = []
    for i in range(sel_mean.shape[0]):
        m1 = sel_mean[i].view(1, 1)
        c1 = sel_covar[i, i].view(1, 1, 1)
        s1 = score_function(m1, c1).item()
        per_point_entropy.append(s1)

    if logger:
        logger.info(f"Entropy selection complete: {len(selected_indices)} points selected")

    # Map pool-local indices back to original candidate indices
    original_indices = pool_indices[selected_indices]
    return X_candidates_norm[original_indices], per_point_entropy, mean_all, var_all


# ---------------------------------------------------------------------------
# YAML config + parameter sweep
# ---------------------------------------------------------------------------

def load_config_with_sweep(config_file, sweep_index=None):
    """
    Load YAML config and optionally apply sweep combination.

    List-valued parameters are treated as sweep dimensions.
    The sweep_index selects one combination from the Cartesian product.

    Returns:
        dict of parameter name -> resolved value.
    """
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    sweep_params = {k: v for k, v in cfg.items() if isinstance(v, list)}
    if sweep_index is not None and sweep_params:
        combinations = list(iterproduct(*sweep_params.values()))
        if sweep_index >= len(combinations):
            raise ValueError(
                f"Sweep index {sweep_index} out of range "
                f"(max {len(combinations)-1}, {len(combinations)} total combinations)"
            )
        for key, value in zip(sweep_params.keys(), combinations[sweep_index]):
            cfg[key] = value

    return cfg


# train_gp_worker is now imported from pmssm.training

# ---------------------------------------------------------------------------
# CLI and main loop
# ---------------------------------------------------------------------------

@click.command()
@click.option('--testing/--no-testing', default=False, help="Run in testing mode (small data).")
@click.option('--n-iterations', default=1, type=int, help="Number of active learning iterations.")
@click.option('--n-candidates', default=1000, type=int, help="Candidate pool size.")
@click.option('--n-select', default=10, type=int, help="Number of points to select per iteration.")
@click.option('--n-datasets', default=None, type=int, help="Number of ROOT datasets to load.")
@click.option('--n-samples', default=None, type=int, help="Number of samples to use from data.")
@click.option('--val-fraction', default=0.2, type=float, help="Fraction of data reserved for validation (default: 0.2). Applied to initial data and each batch of new points.")
@click.option('--output-dir', default='active_learning_gp_output', type=str, help="Output directory.")
@click.option('--generate-data/--no-generate-data', default=False, help="Generate new models using Run3ModelGen.")
@click.option('--min-gen-fraction', default=0.6, type=float, help="Minimum fraction of n-select that must be generated successfully before stopping retries (default: 0.6).")
@click.option('--max-gen-attempts', default=10, type=int, help="Maximum number of generation attempts per iteration (default: 10).")
@click.option('--gen-workers', default=1, type=int, help="Number of parallel genModels.py workers per generation attempt (default: 1).")
# Target & model options
@click.option('--target', default='DMRD', type=click.Choice(['DMRD', 'CrossSection', 'CLs']),
              help="Target function to predict.")
@click.option('--model-type', default='exact_gp',
              type=click.Choice(['exact_gp', 'deep_gp', 'sparse_gp', 'mlp']),
              help="Model type.")
@click.option('--kernel', default='RBF', type=str, help="Kernel type (RBF, Matern, RQK, SpectralMixture, RBF+Matern).")
@click.option('--lengthscale', default=1.0, type=float, help="Initial kernel lengthscale.")
@click.option('--noise', default=1e-2, type=float, help="Initial noise level.")
@click.option('--jitter', default=1e-3, type=float, help="Cholesky jitter.")
@click.option('--learning-rate', default=1e-3, type=float, help="Optimizer learning rate.")
@click.option('--epochs', default=2000, type=int, help="Max training epochs per AL iteration (default: 2000).")
@click.option('--early-stopping/--no-early-stopping', default=True, help="Enable early stopping on validation loss.")
@click.option('--patience', default=200, type=int, help="Early stopping patience (epochs without improvement, default: 200).")
@click.option('--use-ard/--no-ard', default=True, help="Use ARD (Automatic Relevance Determination).")
@click.option('--use-dkl/--no-dkl', default=False, help="Use Deep Kernel Learning (ExactGP only).")
@click.option('--feature-dim', default=2, type=int, help="DKL feature dimension.")
@click.option('--num-hidden-dims', default=10, type=int, help="DeepGP hidden layer dimensions.")
@click.option('--num-middle-dims', default=0, type=int, help="DeepGP middle layer dimensions (0 = no middle layer).")
@click.option('--num-inducing-max', default=512, type=int, help="Max inducing points (DeepGP/SparseGP).")
@click.option('--inducing-strategy', default='kmeans', type=click.Choice(['kmeans', 'vanilla']),
              help="Inducing point initialization (sparse_gp only).")
@click.option('--gp-num-samples', default=8, type=int, help="Number of likelihood samples (DeepGP).")
@click.option('--batch-size', default=256, type=int, help="Batch size (DeepGP/SparseGP).")
@click.option('--warm-starting/--no-warm-starting', default=True, help="Warm-start from previous iteration.")
@click.option('--m-nu', default=1.5, type=float, help="Matern nu parameter.")
@click.option('--num-mixtures', default=4, type=int, help="Number of mixtures for SpectralMixture kernel.")
# Selection strategy options
@click.option('--selection-strategy', default='entropy_batch', type=click.Choice(['top_k', 'top_k_tol_only', 'entropy_batch']),
              help="Selection strategy: top_k or entropy_batch (default).")
@click.option('--entropy-blur', default=0.15, type=float, help="Entropy smoothing parameter (entropy_batch only).")
@click.option('--entropy-beta', default=50.0, type=float, help="Gibbs sampling temperature (entropy_batch only).")
@click.option('--tolerance-sampling', default=1.0, type=float,
              help="Hard cut: keep only candidates within ±tolerance of threshold in transformed space (0 to disable, default: 1.0).")
@click.option('--proximity-sampling', default=0.1, type=float,
              help="Gaussian proximity weighting width around target value (0 to disable, default: 0.1).")
@click.option('--entropy-pool-size', default=5_000, type=int, help="Focused pool size for entropy_batch pre-filtering.")
@click.option('--candidate-source', default='pool',
              type=click.Choice(['pool', 'mcmc']),
              help="Candidate source: 'pool' (random sampling, default) or 'mcmc' "
                   "(theoretical-limit / oracle mode — restrict candidates to the "
                   "pre-loaded MCMC dataset; --generate-data is forced off).")
# Evaluation options
@click.option('--compute-full-metrics/--no-compute-full-metrics', default=False,
              help="Compute comprehensive GoF metrics (accuracy, chi2, pulls, etc.).")
@click.option('--eval-data-path', default=None, type=str, help="Path to true eval dataset (ROOT/CSV).")
@click.option('--mcmc-data-dir', default=None, type=str,
              help="Directory containing MCMC ROOT files for static evaluation (e.g. data/neutralino_v4).")
@click.option('--mcmc-max-samples', default=500_000, type=int,
              help="Seeded uniform subsample cap on the MCMC set (emcee chains "
                   "are ~96% repeated rows; the subsample preserves multiplicity "
                   "weighting). 0 disables.")
@click.option('--static-eval-size', default=100_000, type=int,
              help="Number of models to reserve from the random pool as a static evaluation set (default: 100000).")
@click.option('--track-lengthscales/--no-track-lengthscales', default=True,
              help="Save learned ARD lengthscales per iteration.")
@click.option('--advanced-plots/--no-advanced-plots', default=False,
              help="Generate advanced diagnostic plots (heatmaps, residuals).")
# Config file / sweep options
@click.option('--config-file', default=None, type=str,
              help="YAML config file (overrides CLI args). Supports parameter sweeps.")
@click.option('--sweep-index', default=None, type=int, help="Sweep combination index (requires --config-file).")
@click.option('--data-dir', default='data/18387358', type=str,
              help="Directory containing training ROOT files (default: data/18387358).")
@click.option('--resume-from', default=None, type=str,
              help="Path to previous output dir to resume from (loads state.pt).")
@click.option('--n-additional-iterations', default=None, type=int,
              help="If --resume-from given, run this many more iterations.")
@click.option('--gpu-ids', default='0,1', type=str,
              help="Comma-separated GPU IDs for AL and baseline models (default: 0,1).")
@click.option('--seed', default=42, type=int,
              help="Master random seed propagated to torch / numpy / candidate pool (default: 42).")
def main(testing, n_iterations, n_candidates, n_select, n_datasets, n_samples, val_fraction,
         output_dir, generate_data, min_gen_fraction, max_gen_attempts, gen_workers,
         target, model_type, kernel, lengthscale, noise, jitter, learning_rate,
         epochs, early_stopping, patience,
         use_ard, use_dkl, feature_dim,
         num_hidden_dims, num_middle_dims, num_inducing_max, inducing_strategy,
         gp_num_samples, batch_size, warm_starting, m_nu, num_mixtures,
         selection_strategy, entropy_blur, entropy_beta,
         tolerance_sampling, proximity_sampling, entropy_pool_size,
         candidate_source,
         compute_full_metrics, eval_data_path, mcmc_data_dir, mcmc_max_samples, static_eval_size,
         track_lengthscales, advanced_plots,
         config_file, sweep_index, data_dir, resume_from, n_additional_iterations, gpu_ids, seed):
    """
    Active learning pipeline for pMSSM relic density prediction using GP models.

    Trains ExactGP, DeepGP, SparseGP, or MLP; computes uncertainty via GP posterior
    variance (or entropy-based batch selection); and selects most informative points
    for data generation.
    """
    # ---- Config file override ----
    if config_file is not None:
        cfg = load_config_with_sweep(config_file, sweep_index)
        # Map config keys to local variables (only override if present in config)
        _cfg_map = {
            'target': 'target', 'model_type': 'model_type', 'kernel': 'kernel',
            'lengthscale': 'lengthscale', 'noise': 'noise', 'jitter': 'jitter',
            'learning_rate': 'learning_rate', 'epochs': 'epochs',
            'n_iterations': 'n_iterations', 'n_candidates': 'n_candidates',
            'n_select': 'n_select', 'selection_strategy': 'selection_strategy',
            'entropy_blur': 'entropy_blur', 'entropy_beta': 'entropy_beta',
            'tolerance_sampling': 'tolerance_sampling', 'proximity_sampling': 'proximity_sampling',
            'candidate_source': 'candidate_source',
        }
        _locals = locals()
        for cfg_key, local_key in _cfg_map.items():
            if cfg_key in cfg:
                _locals[local_key] = cfg[cfg_key]
        # Re-assign after locals hack
        target = _locals.get('target', target)
        model_type = _locals.get('model_type', model_type)
        kernel = _locals.get('kernel', kernel)
        lengthscale = float(_locals.get('lengthscale', lengthscale))
        noise = float(_locals.get('noise', noise))
        jitter = float(_locals.get('jitter', jitter))
        learning_rate = float(_locals.get('learning_rate', learning_rate))
        epochs = int(_locals.get('epochs', epochs))
        n_iterations = int(_locals.get('n_iterations', n_iterations))
        n_candidates = int(_locals.get('n_candidates', n_candidates))
        n_select = int(_locals.get('n_select', n_select))
        selection_strategy = _locals.get('selection_strategy', selection_strategy)
        entropy_blur = float(_locals.get('entropy_blur', entropy_blur))
        entropy_beta = float(_locals.get('entropy_beta', entropy_beta))
        tolerance_sampling = float(_locals.get('tolerance_sampling', tolerance_sampling))
        proximity_sampling = float(_locals.get('proximity_sampling', proximity_sampling))
        candidate_source = _locals.get('candidate_source', candidate_source)

    # Propagate master seed to torch / numpy / python-random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    threshold = TARGET_CONFIG[target]["threshold"]

    if n_candidates < n_select:
        n_candidates = n_select

    output_dir = Path(output_dir)
    # Collision-free dir suffix (same contract as active_learning.py).
    warm_tag = "warm" if warm_starting else "cold"
    auto_suffix = f"_{selection_strategy}_{warm_tag}_seed{seed}_{timestamp}"
    if not re.search(r"_\d{8}_\d{6}$", output_dir.name):
        output_dir = output_dir.with_name(output_dir.name + auto_suffix)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file, logger = setup_logging(timestamp, output_dir=output_dir)

    # Route submodule loggers (gp_pipeline.*) through the root logger's handlers
    # so they appear with proper formatting instead of raw text
    gp_logger = logging.getLogger("gp_pipeline")
    gp_logger.setLevel(logging.INFO)

    logger.info("=" * 60)
    logger.info("Active Learning Pipeline for pMSSM (GP Models)")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Output directory: {output_dir}")
    if config_file:
        logger.info(f"Config file: {config_file}" +
                    (f" (sweep index {sweep_index})" if sweep_index is not None else ""))

    # Compute effective patience (None disables early stopping)
    effective_patience = patience if early_stopping else None

    # Apply testing mode defaults
    if testing:
        n_datasets = n_datasets if n_datasets is not None else 3
        n_samples = n_samples if n_samples is not None else 30
        epochs = 50
        n_candidates = 100
        logger.info("Testing mode enabled")
    else:
        n_datasets = n_datasets if n_datasets is not None else -1
        n_samples = n_samples if n_samples is not None else None

    logger.info(f"Configuration:")
    logger.info(f"  target: {target}")
    logger.info(f"  model_type: {model_type}")
    logger.info(f"  n_iterations: {n_iterations}")
    logger.info(f"  n_candidates: {n_candidates}")
    logger.info(f"  n_select: {n_select}")
    logger.info(f"  val_fraction: {val_fraction}")
    logger.info(f"  selection_strategy: {selection_strategy}")
    logger.info(f"  epochs: {epochs}")
    logger.info(f"  early_stopping: {early_stopping} (patience={patience})")
    logger.info(f"  learning_rate: {learning_rate}")
    logger.info(f"  kernel: {kernel}")
    logger.info(f"  lengthscale: {lengthscale}")
    logger.info(f"  noise: {noise}")
    logger.info(f"  jitter: {jitter}")
    logger.info(f"  use_ard: {use_ard}")
    logger.info(f"  use_dkl: {use_dkl}")
    logger.info(f"  warm_starting: {warm_starting}")
    logger.info(f"  n_datasets: {n_datasets}")
    logger.info(f"  n_samples: {n_samples if n_samples else 'all'}")
    logger.info(f"  generate_data: {generate_data}")
    logger.info(f"  compute_full_metrics: {compute_full_metrics}")
    logger.info(f"  track_lengthscales: {track_lengthscales}")
    logger.info(f"  advanced_plots: {advanced_plots}")
    if model_type in ("deep_gp", "sparse_gp"):
        logger.info(f"  num_hidden_dims: {num_hidden_dims}")
        logger.info(f"  num_middle_dims: {num_middle_dims}")
        logger.info(f"  num_inducing_max: {num_inducing_max}")
        logger.info(f"  gp_num_samples: {gp_num_samples}")
        logger.info(f"  batch_size: {batch_size}")
    if model_type == "sparse_gp":
        logger.info(f"  inducing_strategy: {inducing_strategy}")
    if selection_strategy == "entropy_batch":
        logger.info(f"  entropy_blur: {entropy_blur}")
        logger.info(f"  entropy_beta: {entropy_beta}")
        logger.info(f"  tolerance_sampling: {tolerance_sampling}")
        logger.info(f"  proximity_sampling: {proximity_sampling}")
    if generate_data:
        logger.info(f"  min_gen_fraction: {min_gen_fraction}")
        logger.info(f"  max_gen_attempts: {max_gen_attempts}")
        logger.info(f"  gen_workers: {gen_workers}")
    if eval_data_path:
        logger.info(f"  eval_data_path: {eval_data_path}")

    gpu_id_list = [int(x.strip()) for x in gpu_ids.split(',')]
    AL_GPU_ID = gpu_id_list[0]
    BASELINE_GPU_ID = gpu_id_list[1] if len(gpu_id_list) > 1 else gpu_id_list[0]
    device = f"cuda:{AL_GPU_ID}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(AL_GPU_ID)}")

    # Build normalization tensors
    data_min, data_max = build_norm_tensors()
    logger.info(f"Normalization ranges built for {len(PARAM_ORDER)} parameters")

    # Model kwargs (constant across iterations)
    gp_kwargs = dict(
        kernel=kernel, lengthscale=lengthscale, noise=noise,
        use_ard=use_ard, m_nu=m_nu, num_mixtures=num_mixtures,
        use_dkl=use_dkl, feature_dim=feature_dim,
        num_hidden_dims=num_hidden_dims, num_middle_dims=num_middle_dims,
        num_inducing_max=num_inducing_max,
        inducing_strategy=inducing_strategy,
    )

    logger.info("Loading data...")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    X, Y, F = load_pmssm_data(n_datasets=n_datasets, logger=logger,
                              plot_dir=str(plots_dir), data_dir=data_dir,
                              return_lsp_fracs=True)

    # Shuffle once up-front: loaders concatenate ROOT files (= MCMC chains)
    # in file order, so X[:n_samples] would otherwise draw from a single chain.
    _load_perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))
    X = X[_load_perm]
    Y = Y[_load_perm]
    F = F[_load_perm]
    logger.info(f"Shuffled loaded dataset ({len(X)} samples, seed={seed})")

    # Store full dataset for baseline random sampling
    X_full, Y_full, F_full = X.clone(), Y.clone(), F.clone()

    # Select n_samples from loaded data (or use all), then split 80/20 into train/val.
    if n_samples is not None:
        if n_samples > len(X):
            raise ValueError(f"n_samples={n_samples} exceeds available data ({len(X)})")
        X = X[:n_samples].clone()
        Y = Y[:n_samples].clone()
        F = F[:n_samples].clone()
    else:
        X = X.clone()
        Y = Y.clone()
        F = F.clone()

    # Split initial data into train/val using val_fraction
    n_total_init = len(X)
    n_val_init = max(1, int(n_total_init * val_fraction))
    n_train_init = n_total_init - n_val_init

    # Use a fixed permutation so the split is reproducible
    perm = torch.randperm(n_total_init, generator=torch.Generator().manual_seed(seed))
    idx_train_perm = perm[:n_train_init]
    idx_val_perm = perm[n_train_init:]

    X_val = X[idx_val_perm].clone()
    Y_val = Y[idx_val_perm].clone()
    F_val = F[idx_val_perm].clone()
    X = X[idx_train_perm].clone()
    Y = Y[idx_train_perm].clone()
    F = F[idx_train_perm].clone()

    logger.info(f"Initial split ({1-val_fraction:.0%} train / {val_fraction:.0%} val): "
                f"{len(X)} train + {len(X_val)} val = {n_total_init} total")

    # Track which indices from X_full are reserved (train + val).
    # Baseline random sampling will exclude all of these.
    initial_reserved = n_samples if n_samples is not None else len(X_full)
    initial_al_indices = torch.arange(initial_reserved)

    logger.info(f"Baseline pool: X_full={X_full.shape}, reserved [0..{initial_reserved-1}] excluded from sampling")

    # Load MCMC evaluation data if provided
    X_mcmc, Y_mcmc, F_mcmc = None, None, None
    if mcmc_data_dir is not None:
        X_mcmc, Y_mcmc, F_mcmc = load_mcmc_data(data_dir=mcmc_data_dir, logger=logger,
                                                return_lsp_fracs=True,
                                                max_samples=mcmc_max_samples or None)
        logger.info(f"Loaded MCMC evaluation data: {len(X_mcmc)} samples")

    # ---- Theoretical-limit / oracle mode ---------------------------------
    # When --candidate-source=mcmc, restrict per-iteration candidates to
    # points already in the MCMC dataset. Hold out 10% of MCMC as eval so
    # training data and eval data stay disjoint; force --no-generate-data.
    X_mcmc_pool = Y_mcmc_pool = F_mcmc_pool = None
    mcmc_pool_idx = mcmc_eval_idx = None
    mcmc_consumed_mask = None
    if candidate_source == 'mcmc':
        if mcmc_data_dir is None or X_mcmc is None:
            raise click.UsageError(
                "--candidate-source=mcmc requires --mcmc-data-dir to load the candidate pool"
            )
        if generate_data:
            logger.info("Forced --no-generate-data because --candidate-source=mcmc "
                        "(MCMC points are already labelled).")
            generate_data = False
        (X_mcmc_pool, Y_mcmc_pool, F_mcmc_pool,
         X_mcmc, Y_mcmc, F_mcmc,
         mcmc_pool_idx, mcmc_eval_idx) = split_mcmc_for_oracle(
            X_mcmc, Y_mcmc, F_mcmc, eval_fraction=0.1, seed=seed,
        )
        mcmc_consumed_mask = torch.zeros(len(X_mcmc_pool), dtype=torch.bool)
        logger.info(f"Oracle mode: {len(X_mcmc_pool)} candidates / {len(X_mcmc)} eval "
                    f"(MCMC split 90/10, seed={seed})")

    # ------------------------------------------------------------------
    # Representative-points trajectory tracker (seeded, deterministic).
    # Picks 1 point per LSP class nearest the target Ωh², plus the Ωh²-median
    # row, from the MCMC eval pool (fallback: X_full). Same anchors are
    # evaluated by the AL GP every iteration so we can plot mean ± 1σ vs
    # iteration. Symmetric to the transformer driver's tracker.
    # ------------------------------------------------------------------
    if X_mcmc is not None:
        _repr_pool_X, _repr_pool_Y, _repr_pool_F = X_mcmc, Y_mcmc, F_mcmc
        _repr_pool_source = "MCMC eval set"
    else:
        _repr_pool_X, _repr_pool_Y, _repr_pool_F = X_full, Y_full, F_full
        _repr_pool_source = "X_full"
    _target_value = TARGET_CONFIG[target]["true_value"]
    repr_points = pick_representative_points(
        _repr_pool_X, _repr_pool_Y, _repr_pool_F, _target_value, seed=seed
    )
    logger.info(
        f"Representative points (from {_repr_pool_source}): "
        + ", ".join(f"{lbl}@idx={i}:Ω={y.item():.4f}"
                    for lbl, i, y in zip(repr_points['labels'],
                                          repr_points['indices'],
                                          repr_points['Y'].reshape(-1)))
    )
    repr_log = []

    # Carve out static random evaluation set from X_full (after initial reserved block)
    X_static_random, Y_static_random, F_static_random = None, None, None
    static_random_indices = torch.tensor([], dtype=torch.long)
    if static_eval_size > 0:
        available_for_static = len(X_full) - initial_reserved
        actual_static_size = min(static_eval_size, available_for_static)
        if actual_static_size < static_eval_size:
            logger.warning(
                f"Requested static_eval_size={static_eval_size} but only "
                f"{available_for_static} available after reserving {initial_reserved} "
                f"for initial AL data. Using {actual_static_size}."
            )
        if actual_static_size > 0:
            g_static = torch.Generator().manual_seed(123)
            perm_static = torch.randperm(available_for_static, generator=g_static)
            static_random_indices = perm_static[:actual_static_size] + initial_reserved
            X_static_random = X_full[static_random_indices].clone()
            Y_static_random = Y_full[static_random_indices].clone()
            F_static_random = F_full[static_random_indices].clone()
            logger.info(
                f"Static random evaluation set: {len(X_static_random)} samples "
                f"(carved from indices {initial_reserved}..{len(X_full)-1})"
            )

    # Determine if we can train AL and Baseline in parallel (2+ GPUs)
    use_parallel = torch.cuda.is_available() and torch.cuda.device_count() >= 2
    if use_parallel:
        logger.info(f"Parallel training enabled: {torch.cuda.device_count()} GPUs available")
        logger.info(f"  - Active Learning model on cuda:{AL_GPU_ID}")
        logger.info(f"  - Baseline model on cuda:{BASELINE_GPU_ID}")
        mp.set_start_method('spawn', force=True)
    else:
        logger.info("Sequential training (need 2+ GPUs for parallel)")

    # Active learning iterations
    all_selected_points = []
    iteration_numbers = []

    al_train_losses, al_val_losses, al_r2_scores, al_train_r2_scores = [], [], [], []
    al_n_train, al_n_val = [], []
    baseline_train_losses, baseline_val_losses, baseline_r2_scores, baseline_train_r2_scores = [], [], [], []
    baseline_n_train, baseline_n_val = [], []

    # Cross-evaluation metrics (each model on the other's validation set)
    al_on_base_val_losses = []
    al_on_base_val_r2 = []
    base_on_al_val_losses = []
    base_on_al_val_r2 = []

    # MCMC and static random evaluation metrics
    al_on_mcmc_losses, al_on_mcmc_r2 = [], []
    baseline_on_mcmc_losses, baseline_on_mcmc_r2 = [], []
    al_on_static_random_losses, al_on_static_random_r2 = [], []
    baseline_on_static_random_losses, baseline_on_static_random_r2 = [], []

    # Lengthscale tracking
    lengthscale_rows = []

    # Full eval dataset (loaded lazily on first use)
    X_eval_full, Y_eval_full = None, None
    eval_r2_scores = []

    # Previous model state for warm starting
    prev_al_checkpoint = None
    prev_baseline_checkpoint = None

    # Persistent baseline augmentation indices (grows each iteration)
    baseline_add_indices = torch.tensor([], dtype=torch.long)
    prev_n_add_train = 0
    prev_n_add_val = 0

    # ---- Resume handling ----------------------------------------------------
    start_iteration = 1
    if resume_from is not None:
        from pmssm.resume import load_state, restore_rng
        if n_additional_iterations is None:
            raise click.UsageError("--n-additional-iterations is required with --resume-from")
        saved = load_state(resume_from)
        if saved is None:
            raise click.UsageError(f"No state.pt found in {resume_from}")
        logger.info(f"Resuming from {resume_from} (last completed iteration: {saved['iteration']})")
        X, Y = saved["X"], saved["Y"]
        X_val, Y_val = saved["X_val"], saved["Y_val"]
        F = saved.get("F", torch.full((len(X), 3), float('nan')))
        F_val = saved.get("F_val", torch.full((len(X_val), 3), float('nan')))
        baseline_add_indices = saved["baseline_add_indices"]
        prev_n_add_train = saved["prev_n_add_train"]
        prev_n_add_val = saved["prev_n_add_val"]
        all_selected_points = saved["all_selected_points"]
        iteration_numbers = saved["iteration_numbers"]
        al_train_losses = saved["al_train_losses"]; al_val_losses = saved["al_val_losses"]
        al_r2_scores = saved["al_r2_scores"]; al_train_r2_scores = saved["al_train_r2_scores"]
        al_n_train = saved["al_n_train"]; al_n_val = saved["al_n_val"]
        baseline_train_losses = saved["baseline_train_losses"]
        baseline_val_losses = saved["baseline_val_losses"]
        baseline_r2_scores = saved["baseline_r2_scores"]
        baseline_train_r2_scores = saved["baseline_train_r2_scores"]
        baseline_n_train = saved["baseline_n_train"]; baseline_n_val = saved["baseline_n_val"]
        al_on_base_val_losses = saved["al_on_base_val_losses"]
        al_on_base_val_r2 = saved["al_on_base_val_r2"]
        base_on_al_val_losses = saved["base_on_al_val_losses"]
        base_on_al_val_r2 = saved["base_on_al_val_r2"]
        al_on_mcmc_losses = saved["al_on_mcmc_losses"]; al_on_mcmc_r2 = saved["al_on_mcmc_r2"]
        baseline_on_mcmc_losses = saved["baseline_on_mcmc_losses"]
        baseline_on_mcmc_r2 = saved["baseline_on_mcmc_r2"]
        al_on_static_random_losses = saved["al_on_static_random_losses"]
        al_on_static_random_r2 = saved["al_on_static_random_r2"]
        baseline_on_static_random_losses = saved["baseline_on_static_random_losses"]
        baseline_on_static_random_r2 = saved["baseline_on_static_random_r2"]
        eval_r2_scores = saved.get("eval_r2_scores", [])
        lengthscale_rows = saved.get("lengthscale_rows", [])
        # Restore oracle-mode state if the resumed run was an oracle run.
        if candidate_source == 'mcmc' and "mcmc_consumed_mask" in saved:
            mcmc_consumed_mask = saved["mcmc_consumed_mask"]
            mcmc_pool_idx = saved.get("mcmc_pool_idx", mcmc_pool_idx)
            mcmc_eval_idx = saved.get("mcmc_eval_idx", mcmc_eval_idx)
            logger.info(f"Resumed oracle state: {int(mcmc_consumed_mask.sum())}/"
                        f"{len(mcmc_consumed_mask)} candidates consumed")
        prev_iter_dir = Path(resume_from) / f"iteration_{saved['iteration']:03d}"
        prev_al_checkpoint = prev_iter_dir / "al_model_checkpoint.pt"
        prev_baseline_checkpoint = prev_iter_dir / "baseline_model_checkpoint.pt"
        if not prev_al_checkpoint.exists(): prev_al_checkpoint = None
        if not prev_baseline_checkpoint.exists(): prev_baseline_checkpoint = None
        restore_rng(saved["rng"])
        start_iteration = saved["iteration"] + 1
        n_iterations = saved["iteration"] + n_additional_iterations
        logger.info(f"Resuming at iteration {start_iteration}, will run through {n_iterations}")

    for iteration in range(start_iteration, n_iterations + 1):
        logger.info(f"=== GP Active Learning Iteration {iteration} ===")

        iter_dir = output_dir / f"iteration_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        iter_plots_dir = iter_dir / "plots"
        iter_plots_dir.mkdir(parents=True, exist_ok=True)

        # ---- Build baseline dataset (train + val) ----
        if iteration == 1:
            X_baseline_train = X.clone()
            Y_baseline_train = Y.clone()
            X_baseline_val = X_val.clone()
            Y_baseline_val = Y_val.clone()
            F_baseline_train = F.clone()
            F_baseline_val = F_val.clone()
            logger.info(f"Iteration 1: Both models use identical data "
                       f"({len(X_baseline_train)} train + {len(X_baseline_val)} val)")
        else:
            # Iteration 2+: Baseline grows incrementally by sampling NEW random
            # points from X_full, keeping all previously sampled points.
            n_add_train = len(X) - n_train_init
            n_add_val = len(X_val) - n_val_init
            n_new_train = n_add_train - prev_n_add_train
            n_new_val = n_add_val - prev_n_add_val
            n_new_total = n_new_train + n_new_val

            all_indices = torch.arange(len(X_full))
            mask = torch.ones(len(X_full), dtype=torch.bool)
            mask[initial_al_indices] = False  # Exclude initial reserved data
            if len(static_random_indices) > 0:
                mask[static_random_indices] = False  # Exclude static eval set
            if len(baseline_add_indices) > 0:
                mask[baseline_add_indices] = False  # Exclude already-sampled baseline points
            available_indices = all_indices[mask]

            logger.info(f"Baseline sampling: need {n_new_total} new points "
                       f"({n_new_train} train + {n_new_val} val), "
                       f"{len(available_indices)} available from X_full")

            if n_new_total <= len(available_indices):
                new_idx = available_indices[
                    torch.randperm(len(available_indices))[:n_new_total]
                ]
            else:
                logger.info(f"Baseline: sampling with replacement "
                           f"({n_new_total} needed, {len(available_indices)} available)")
                new_idx = available_indices[
                    torch.randint(0, len(available_indices), (n_new_total,))
                ]

            # Append new indices to persistent baseline indices
            # Layout: [train_indices... | val_indices...]
            baseline_add_indices = torch.cat([
                baseline_add_indices[:prev_n_add_train],
                new_idx[:n_new_train],
                baseline_add_indices[prev_n_add_train:],
                new_idx[n_new_train:],
            ])
            prev_n_add_train = n_add_train
            prev_n_add_val = n_add_val

            X_add = X_full[baseline_add_indices]
            Y_add = Y_full[baseline_add_indices]
            F_add = F_full[baseline_add_indices]

            X_baseline_train = torch.cat([X[:n_train_init], X_add[:n_add_train]])
            Y_baseline_train = torch.cat([Y[:n_train_init], Y_add[:n_add_train]])
            F_baseline_train = torch.cat([F[:n_train_init], F_add[:n_add_train]])
            X_baseline_val = torch.cat([X_val[:n_val_init], X_add[n_add_train:]])
            Y_baseline_val = torch.cat([Y_val[:n_val_init], Y_add[n_add_train:]])
            F_baseline_val = torch.cat([F_val[:n_val_init], F_add[n_add_train:]])
            logger.info(f"Baseline dataset: {n_train_init}+{n_add_train}={len(X_baseline_train)} train, "
                       f"{n_val_init}+{n_add_val}={len(X_baseline_val)} val")

        # ---- Assign train/val tensors for this iteration ----
        X_train_al = X
        Y_train_al = Y
        X_val_al = X_val
        Y_val_al = Y_val
        F_val_al = F_val

        X_train_base = X_baseline_train
        Y_train_base = Y_baseline_train
        X_val_base = X_baseline_val
        Y_val_base = Y_baseline_val

        logger.info(f"AL: n_train={len(X_train_al)}, n_val={len(X_val_al)}")
        logger.info(f"Baseline: n_train={len(X_train_base)}, n_val={len(X_val_base)}")

        # Plot data distribution histograms for AL
        al_hist_dir = iter_plots_dir / "al"
        al_hist_dir.mkdir(parents=True, exist_ok=True)
        X_al_combined = torch.cat([X, X_val], dim=0)
        Y_al_combined = torch.cat([Y, Y_val], dim=0)
        idx_train_plot = torch.arange(len(X))
        idx_val_plot = torch.arange(len(X), len(X_al_combined))
        plot_data_histograms(X_al_combined, Y_al_combined, idx_train_plot, idx_val_plot, al_hist_dir, "AL", iteration, logger,
                             reference_X=X_mcmc, reference_Y=Y_mcmc, reference_label="MCMC")
        plot_parallel_coordinates(X_al_combined, idx_train_plot, idx_val_plot, al_hist_dir, "AL", iteration, logger)

        # Plot data distribution histograms and parallel coordinates for Baseline
        baseline_hist_dir = iter_plots_dir / "baseline"
        baseline_hist_dir.mkdir(parents=True, exist_ok=True)
        X_base_combined = torch.cat([X_baseline_train, X_baseline_val], dim=0)
        Y_base_combined = torch.cat([Y_baseline_train, Y_baseline_val], dim=0)
        idx_train_base_plot = torch.arange(len(X_baseline_train))
        idx_val_base_plot = torch.arange(len(X_baseline_train), len(X_base_combined))
        plot_data_histograms(X_base_combined, Y_base_combined, idx_train_base_plot, idx_val_base_plot,
                             baseline_hist_dir, "Baseline", iteration, logger,
                             reference_X=X_static_random, reference_Y=Y_static_random, reference_label="Static Random")
        plot_parallel_coordinates(X_base_combined, idx_train_base_plot, idx_val_base_plot, baseline_hist_dir, "Baseline", iteration, logger)

        # Plot new-points-only histograms for baseline (iteration 2+)
        if iteration > 1 and len(new_idx) > 0:
            X_base_new = X_full[new_idx]
            Y_base_new = Y_full[new_idx]
            idx_train_base_new = torch.arange(n_new_train)
            idx_val_base_new = torch.arange(n_new_train, len(new_idx))
            plot_data_histograms(X_base_new, Y_base_new, idx_train_base_new, idx_val_base_new,
                                 baseline_hist_dir, "Baseline_new", iteration, logger,
                                 fixed_axes=True)

        al_checkpoint_path = iter_dir / "al_model_checkpoint.pt"
        baseline_checkpoint_path = iter_dir / "baseline_model_checkpoint.pt"

        if use_parallel:
            # ---- Train AL and Baseline in parallel on different GPUs ----
            al_queue = mp.Queue()
            baseline_queue = mp.Queue()

            al_warm_start = prev_al_checkpoint if warm_starting else None
            baseline_warm_start = prev_baseline_checkpoint if warm_starting else None
            al_process = mp.Process(
                target=train_gp_worker,
                args=(AL_GPU_ID, X_train_al, Y_train_al, X_val_al, Y_val_al,
                      data_min, data_max, model_type, len(PARAM_ORDER), gp_kwargs,
                      learning_rate, epochs, batch_size, jitter,
                      al_queue, "AL", iter_dir, iter_plots_dir, al_checkpoint_path,
                      gp_num_samples, al_warm_start, target, effective_patience,
                      F, F_val),
            )
            baseline_process = mp.Process(
                target=train_gp_worker,
                args=(BASELINE_GPU_ID, X_train_base, Y_train_base, X_val_base, Y_val_base,
                      data_min, data_max, model_type, len(PARAM_ORDER), gp_kwargs,
                      learning_rate, epochs, batch_size, jitter,
                      baseline_queue, "Baseline", iter_dir, iter_plots_dir,
                      baseline_checkpoint_path,
                      gp_num_samples, baseline_warm_start, target, effective_patience,
                      F_baseline_train, F_baseline_val),
            )

            logger.info(f"Launching parallel training: AL on cuda:{AL_GPU_ID}, "
                        f"Baseline on cuda:{BASELINE_GPU_ID}")
            al_process.start()
            baseline_process.start()

            # Read from queues BEFORE joining — joining first can deadlock
            # if the result object is large enough to fill the pipe buffer.
            al_results = al_queue.get()
            baseline_results = baseline_queue.get()

            al_process.join()
            baseline_process.join()
        else:
            # ---- Train AL and Baseline sequentially ----
            al_warm_start = prev_al_checkpoint if warm_starting else None
            baseline_warm_start = prev_baseline_checkpoint if warm_starting else None
            logger.info(f"Training AL {model_type} model ({len(X_train_al)} train, {len(X_val_al)} val)...")
            al_queue = mp.Queue()
            train_gp_worker(
                device, X_train_al, Y_train_al, X_val_al, Y_val_al,
                data_min, data_max, model_type, len(PARAM_ORDER), gp_kwargs,
                learning_rate, epochs, batch_size, jitter,
                al_queue, "AL", iter_dir, iter_plots_dir, al_checkpoint_path,
                num_samples=gp_num_samples, warm_start_path=al_warm_start,
                target=target, patience=effective_patience,
                lsp_fracs=F, lsp_fracs_val=F_val,
            )
            al_results = al_queue.get()

            logger.info(f"Training Baseline {model_type} model ({len(X_train_base)} train, {len(X_val_base)} val)...")
            baseline_queue = mp.Queue()
            train_gp_worker(
                device, X_train_base, Y_train_base, X_val_base, Y_val_base,
                data_min, data_max, model_type, len(PARAM_ORDER), gp_kwargs,
                learning_rate, epochs, batch_size, jitter,
                baseline_queue, "Baseline", iter_dir, iter_plots_dir,
                baseline_checkpoint_path,
                num_samples=gp_num_samples, warm_start_path=baseline_warm_start,
                target=target, patience=effective_patience,
                lsp_fracs=F_baseline_train, lsp_fracs_val=F_baseline_val,
            )
            baseline_results = baseline_queue.get()

        # ---- Log results ----
        logger.info(f"AL metrics: train_loss={al_results['best_train_loss']:.6f}, "
                   f"val_loss={al_results['best_val_loss']:.6f}, "
                   f"R²={al_results['r2_score']:.4f}, "
                   f"train_R²={al_results['train_r2_score']:.4f}")
        logger.info(f"Baseline metrics: train_loss={baseline_results['best_train_loss']:.6f}, "
                   f"val_loss={baseline_results['best_val_loss']:.6f}, "
                   f"R²={baseline_results['r2_score']:.4f}, "
                   f"train_R²={baseline_results['train_r2_score']:.4f}")

        # Track metrics
        iteration_numbers.append(iteration)
        al_train_losses.append(al_results['best_train_loss'])
        al_val_losses.append(al_results['best_val_loss'])
        al_r2_scores.append(al_results['r2_score'])
        al_train_r2_scores.append(al_results['train_r2_score'])
        al_n_train.append(len(X_train_al))
        al_n_val.append(len(X_val_al))
        baseline_train_losses.append(baseline_results['best_train_loss'])
        baseline_val_losses.append(baseline_results['best_val_loss'])
        baseline_r2_scores.append(baseline_results['r2_score'])
        baseline_train_r2_scores.append(baseline_results['train_r2_score'])
        baseline_n_train.append(len(X_train_base))
        baseline_n_val.append(len(X_val_base))

        # ---- Track lengthscales ----
        if track_lengthscales and al_results.get("lengthscales"):
            lengthscale_rows.append({"iteration": iteration, **al_results["lengthscales"]})

        # ---- Compute uncertainty on candidates using AL model ----
        logger.info("Loading AL model checkpoint for uncertainty computation...")

        # Reload the trained AL model
        x_train_norm = normalize_x(X_train_al, data_min, data_max)
        x_val_norm = normalize_x(X_val_al, data_min, data_max)
        y_train_t = transform_y(Y_train_al, target=target).view(-1)
        y_val_t = transform_y(Y_val_al, target=target).view(-1)

        al_model = create_gp_model(
            model_type, x_train_norm, y_train_t, x_val_norm, y_val_t,
            n_dim=len(PARAM_ORDER), num_samples=gp_num_samples,
            target=target, device=device, **gp_kwargs
        )
        checkpoint = torch.load(al_checkpoint_path, map_location=device)
        al_model.load_state_dict(checkpoint['model_state_dict'])
        if model_has_likelihood(model_type):
            al_model.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        al_model = al_model.to(device)
        if model_has_likelihood(model_type):
            al_model.likelihood = al_model.likelihood.to(device)
        logger.info("AL model reloaded for uncertainty estimation")

        # ---- Cross-evaluation: each model on the other's validation set ----
        _gp_eval_kw = dict(data_min=data_min, data_max=data_max, model_type=model_type,
                           jitter=jitter, num_samples=gp_num_samples, target=target,
                           return_predictions=True)

        # Representative-points capture: GP posterior mean + variance at the fixed
        # anchors chosen before the iteration loop. Output is in *transformed*
        # (z-score) space — plot_representative_trajectories maps it back to physical
        # Ωh² with `y_transform='zscore'`.
        try:
            _rep_mean_t, _rep_var_t = compute_uncertainty_gp(
                al_model, repr_points['X'], data_min, data_max,
                model_type=model_type, jitter=jitter, num_samples=gp_num_samples,
                logger=None,
            )
            repr_log.append({
                'iteration': int(iteration),
                'mean': _rep_mean_t.detach().cpu().numpy().reshape(-1).tolist(),
                'var':  _rep_var_t.detach().cpu().numpy().reshape(-1).tolist(),
            })
        except Exception as _e:
            logger.warning(f"Representative-points capture failed at iter {iteration}: {_e}")

        al_cross_loss, al_cross_r2, al_cross_yt, al_cross_yp = cross_evaluate_gp(
            al_model, X_baseline_val, Y_baseline_val, **_gp_eval_kw)

        # Load baseline model for cross-eval
        x_train_base_norm = normalize_x(X_train_base, data_min, data_max)
        x_val_base_norm = normalize_x(X_val_base, data_min, data_max)
        y_train_base_t = transform_y(Y_train_base, target=target).view(-1)
        y_val_base_t = transform_y(Y_val_base, target=target).view(-1)

        baseline_model = create_gp_model(
            model_type, x_train_base_norm, y_train_base_t,
            x_val_base_norm, y_val_base_t,
            n_dim=len(PARAM_ORDER), num_samples=gp_num_samples,
            target=target, device=device, **gp_kwargs
        )
        base_ckpt = torch.load(baseline_checkpoint_path, map_location=device)
        baseline_model.load_state_dict(base_ckpt['model_state_dict'])
        if model_has_likelihood(model_type):
            baseline_model.likelihood.load_state_dict(base_ckpt['likelihood_state_dict'])
        baseline_model = baseline_model.to(device)
        if model_has_likelihood(model_type):
            baseline_model.likelihood = baseline_model.likelihood.to(device)

        base_cross_loss, base_cross_r2, base_cross_yt, base_cross_yp = cross_evaluate_gp(
            baseline_model, X_val_al, Y_val_al, **_gp_eval_kw)

        logger.info(f"Cross-eval: AL_on_base_val_loss={al_cross_loss:.6f}, AL_on_base_val_R²={al_cross_r2:.4f}, base_on_al_val_loss={base_cross_loss:.6f}, base_on_al_val_R²={base_cross_r2:.4f}")
        al_on_base_val_losses.append(al_cross_loss)
        al_on_base_val_r2.append(al_cross_r2)
        base_on_al_val_losses.append(base_cross_loss)
        base_on_al_val_r2.append(base_cross_r2)

        # Own-val predictions for scatterplots
        al_own_loss, al_own_r2, al_own_yt, al_own_yp = cross_evaluate_gp(
            al_model, X_val_al, Y_val_al, **_gp_eval_kw)
        base_own_loss, base_own_r2, base_own_yt, base_own_yp = cross_evaluate_gp(
            baseline_model, X_baseline_val, Y_baseline_val, **_gp_eval_kw)

        scatter_results = [
            dict(model_name="AL", dataset_name="AL Val", y_true=al_own_yt, y_pred=al_own_yp,
                 loss=al_own_loss, r2=al_own_r2, n=len(X_val_al), lsp_fracs=F_val_al),
            dict(model_name="AL", dataset_name="Base Val", y_true=al_cross_yt, y_pred=al_cross_yp,
                 loss=al_cross_loss, r2=al_cross_r2, n=len(X_baseline_val), lsp_fracs=F_baseline_val),
            dict(model_name="Baseline", dataset_name="AL Val", y_true=base_cross_yt, y_pred=base_cross_yp,
                 loss=base_cross_loss, r2=base_cross_r2, n=len(X_val_al), lsp_fracs=F_val_al),
            dict(model_name="Baseline", dataset_name="Base Val", y_true=base_own_yt, y_pred=base_own_yp,
                 loss=base_own_loss, r2=base_own_r2, n=len(X_baseline_val), lsp_fracs=F_baseline_val),
        ]

        # ---- Evaluate on MCMC static dataset ----
        if X_mcmc is not None:
            mcmc_loss_al, mcmc_r2_al, mcmc_yt_al, mcmc_yp_al = cross_evaluate_gp(
                al_model, X_mcmc, Y_mcmc, **_gp_eval_kw)
            mcmc_loss_base, mcmc_r2_base, mcmc_yt_base, mcmc_yp_base = cross_evaluate_gp(
                baseline_model, X_mcmc, Y_mcmc, **_gp_eval_kw)
            al_on_mcmc_losses.append(mcmc_loss_al)
            al_on_mcmc_r2.append(mcmc_r2_al)
            baseline_on_mcmc_losses.append(mcmc_loss_base)
            baseline_on_mcmc_r2.append(mcmc_r2_base)
            logger.info(f"MCMC eval: AL_loss={mcmc_loss_al:.6f}, AL_R²={mcmc_r2_al:.4f}, "
                        f"Base_loss={mcmc_loss_base:.6f}, Base_R²={mcmc_r2_base:.4f}")
            scatter_results.append(dict(model_name="AL", dataset_name="MCMC",
                y_true=mcmc_yt_al, y_pred=mcmc_yp_al, loss=mcmc_loss_al, r2=mcmc_r2_al, n=len(X_mcmc), lsp_fracs=F_mcmc))
            scatter_results.append(dict(model_name="Baseline", dataset_name="MCMC",
                y_true=mcmc_yt_base, y_pred=mcmc_yp_base, loss=mcmc_loss_base, r2=mcmc_r2_base, n=len(X_mcmc), lsp_fracs=F_mcmc))

        # ---- Evaluate on static random dataset ----
        if X_static_random is not None:
            static_loss_al, static_r2_al, static_yt_al, static_yp_al = cross_evaluate_gp(
                al_model, X_static_random, Y_static_random, **_gp_eval_kw)
            static_loss_base, static_r2_base, static_yt_base, static_yp_base = cross_evaluate_gp(
                baseline_model, X_static_random, Y_static_random, **_gp_eval_kw)
            al_on_static_random_losses.append(static_loss_al)
            al_on_static_random_r2.append(static_r2_al)
            baseline_on_static_random_losses.append(static_loss_base)
            baseline_on_static_random_r2.append(static_r2_base)
            logger.info(f"Static random eval: AL_loss={static_loss_al:.6f}, AL_R²={static_r2_al:.4f}, "
                        f"Base_loss={static_loss_base:.6f}, Base_R²={static_r2_base:.4f}")
            scatter_results.append(dict(model_name="AL", dataset_name="Static Rnd",
                y_true=static_yt_al, y_pred=static_yp_al, loss=static_loss_al, r2=static_r2_al, n=len(X_static_random), lsp_fracs=F_static_random))
            scatter_results.append(dict(model_name="Baseline", dataset_name="Static Rnd",
                y_true=static_yt_base, y_pred=static_yp_base, loss=static_loss_base, r2=static_r2_base, n=len(X_static_random), lsp_fracs=F_static_random))

        plot_eval_scatterplots(scatter_results, iteration, iter_plots_dir, logger)

        # ---- Classification-accuracy capture --------------------------------
        # Compute binary accuracy at the constraint threshold (target_value in
        # physical space) from the predictions already produced above, and
        # write them into <output_dir>/accuracy_trajectory.json in the schema
        # consumed by scripts/plot_hit_rate_trajectories_multiseed.py. With
        # this cache populated in-training, the offline accuracy plotter
        # downgrades to a pure cache read (no checkpoint reload, no inference).
        # Failures are non-fatal — accuracy is diagnostic.
        try:
            _acc_thr = float(_target_value)
            al_accs: dict = {"val": binary_accuracy(al_own_yt, al_own_yp, _acc_thr)}
            base_accs: dict = {"val": binary_accuracy(base_own_yt, base_own_yp, _acc_thr)}
            if X_mcmc is not None:
                al_accs["mcmc"] = binary_accuracy(mcmc_yt_al, mcmc_yp_al, _acc_thr)
                base_accs["mcmc"] = binary_accuracy(mcmc_yt_base, mcmc_yp_base, _acc_thr)
            if X_static_random is not None:
                al_accs["static_random"] = binary_accuracy(static_yt_al, static_yp_al, _acc_thr)
                base_accs["static_random"] = binary_accuracy(static_yt_base, static_yp_base, _acc_thr)
            # Train: extra inference on each role's own training set.
            # _gp_eval_kw already has return_predictions=True.
            _, _, _al_tr_yt, _al_tr_yp = cross_evaluate_gp(
                al_model, X_train_al, Y_train_al, **_gp_eval_kw)
            al_accs["train"] = binary_accuracy(_al_tr_yt, _al_tr_yp, _acc_thr)
            _, _, _bs_tr_yt, _bs_tr_yp = cross_evaluate_gp(
                baseline_model, X_baseline_train, Y_baseline_train, **_gp_eval_kw)
            base_accs["train"] = binary_accuracy(_bs_tr_yt, _bs_tr_yp, _acc_thr)
            write_iter_accuracies(output_dir, iteration, al_accs=al_accs,
                                  baseline_accs=base_accs)
            _acc_summary = "  ".join(
                f"{role}=" + "/".join(f"{k}={v:.4f}" for k, v in (d or {}).items())
                for role, d in (("AL", al_accs), ("Base", base_accs)) if d
            )
            logger.info(f"Accuracy@{_acc_thr}: {_acc_summary}")
        except Exception as _acc_exc:
            logger.warning(f"Accuracy capture failed (non-fatal): {_acc_exc}")

        # ---- Full true dataset evaluation ----
        if eval_data_path is not None:
            if X_eval_full is None:
                X_eval_full, Y_eval_full = load_true_eval_dataset(
                    eval_data_path, target=target, logger=logger
                )
            x_eval_norm = normalize_x(X_eval_full, data_min, data_max)
            y_eval_t = transform_y(Y_eval_full, target=target).view(-1)
            eval_r2 = compute_gp_r2(al_model, x_eval_norm, y_eval_t, model_type,
                                    jitter=jitter, num_samples=gp_num_samples, target=target)
            logger.info(f"Full eval R²: {eval_r2:.4f}")
            eval_r2_scores.append(eval_r2)

            if compute_full_metrics:
                eval_metrics = compute_comprehensive_metrics(
                    al_model, x_eval_norm, y_eval_t, model_type,
                    threshold=threshold, jitter=jitter, num_samples=gp_num_samples,
                    logger=logger,
                )
                eval_metrics_path = iter_dir / "gof_eval.csv"
                pd.DataFrame([eval_metrics]).to_csv(eval_metrics_path, index=False)
                logger.info(f"Eval metrics saved to {eval_metrics_path}")

        # ---- Compute full metrics on validation set ----
        if compute_full_metrics:
            val_metrics = compute_comprehensive_metrics(
                al_model, x_val_norm, y_val_t, model_type,
                threshold=threshold, jitter=jitter, num_samples=gp_num_samples,
                logger=logger,
            )
            val_metrics_path = iter_dir / "gof_al.csv"
            pd.DataFrame([val_metrics]).to_csv(val_metrics_path, index=False)
            logger.info(f"AL validation metrics saved to {val_metrics_path}")

        # ---- Select new points ----
        # In oracle mode the candidate pool is a freshly drawn slice of the
        # held-out MCMC tensor; we keep `cand_pool_idx` around so the
        # post-selection augmentation can look up the labels directly.
        cand_pool_idx = None
        if candidate_source == 'mcmc':
            avail = (~mcmc_consumed_mask).nonzero(as_tuple=False).squeeze(-1)
            if len(avail) == 0:
                raise RuntimeError("MCMC candidate pool exhausted before all iterations completed")
            take = min(n_candidates, len(avail))
            _g_cand = torch.Generator().manual_seed(seed * 10_000 + iteration)
            cand_pool_idx = avail[torch.randperm(len(avail), generator=_g_cand)[:take]]
            _candidates_phys = X_mcmc_pool[cand_pool_idx]
            logger.info(f"Oracle: drew {take}/{len(avail)} unconsumed MCMC candidates "
                        f"(consumed so far: {int(mcmc_consumed_mask.sum())}/{len(mcmc_consumed_mask)})")
        else:
            _candidates_phys = generate_candidate_pool(n_candidates, seed=seed * 10_000 + iteration)

        if selection_strategy == "entropy_batch" and model_has_likelihood(model_type):
            logger.info("Using entropy-based batch selection...")
            candidates_norm = normalize_x(_candidates_phys, data_min, data_max).to(device)
            selected_points_norm, per_point_entropy, _ent_mean_all, _ent_var_all = select_entropy_batch(
                al_model, candidates_norm, n_select, model_type,
                threshold=threshold, blur=entropy_blur, beta=entropy_beta,
                tolerance_sampling=tolerance_sampling,
                proximity_sampling=proximity_sampling,
                use_dkl=use_dkl, jitter=jitter, num_samples=gp_num_samples,
                entropy_pool_size=entropy_pool_size,
                device=device, logger=logger,
            )
            # Unnormalize selected points back to physical space (move to CPU first)
            selected_points_phys = unnormalize_x(selected_points_norm.cpu(), data_min, data_max)
            top_indices = torch.arange(len(selected_points_phys))
            candidates = selected_points_phys  # For downstream compatibility

            # Compute variance for logging (already have entropy)
            pred_var = torch.tensor(per_point_entropy)

            # In oracle mode, map the selected points back to MCMC pool indices
            # so we can mark them consumed and pull labels. We match in
            # *normalized* space — selected_points_norm came directly from
            # candidates_norm inside select_entropy_batch, so float drift is
            # negligible. Matching in physical space failed because the
            # round-trip selected_points_norm → unnormalize_x → physical
            # introduced ~1e-7 drift, breaking exact equality.
            if candidate_source == 'mcmc':
                _cands_norm_np = candidates_norm.cpu().numpy()
                _sel_norm_np = selected_points_norm.cpu().numpy()
                _sel_pool_idx = []
                for row in _sel_norm_np:
                    diffs = np.abs(_cands_norm_np - row).max(axis=1)
                    nearest = int(diffs.argmin())
                    if diffs[nearest] > 1e-6:
                        raise RuntimeError(
                            f"Oracle: drift {diffs[nearest]:.3e} too large at row "
                            f"{row[:3]}…; selected entropy-batch point not from MCMC pool"
                        )
                    _sel_pool_idx.append(int(cand_pool_idx[nearest].item()))
                _oracle_selected_pool_idx = torch.tensor(_sel_pool_idx, dtype=torch.long)
            else:
                _oracle_selected_pool_idx = None

            logger.info(f"Entropy selection: {len(selected_points_phys)} points selected")
        else:
            # Standard top-K by variance
            if selection_strategy == "entropy_batch" and not model_has_likelihood(model_type):
                logger.warning(f"entropy_batch not supported for {model_type}; "
                              "falling back to top_k")

            candidates = _candidates_phys
            logger.info(f"Generating {n_candidates} candidate points...")

            _pred_mean, pred_var = compute_uncertainty_gp(
                al_model, candidates, data_min, data_max,
                model_type=model_type, jitter=jitter, num_samples=gp_num_samples,
                logger=logger,
            )

            if selection_strategy == "top_k_tol_only":
                top_indices = select_top_uncertain_tol_only(
                    candidates, _pred_mean, pred_var.unsqueeze(1), n_select,
                    threshold=threshold,
                    tolerance_sampling=tolerance_sampling,
                    logger=logger,
                )
            else:
                top_indices = select_top_uncertain_filtered(
                    candidates, _pred_mean, pred_var.unsqueeze(1), n_select,
                    threshold=threshold,
                    tolerance_sampling=tolerance_sampling,
                    proximity_sampling=proximity_sampling,
                    logger=logger,
                )

            # Oracle mode: top_indices is a 1-D tensor into the candidate pool.
            if candidate_source == 'mcmc':
                _oracle_selected_pool_idx = cand_pool_idx[top_indices]
            else:
                _oracle_selected_pool_idx = None

        logger.info(f"Selected {len(top_indices)} most uncertain points")

        csv_path = save_selected_points(candidates, pred_var.unsqueeze(1),
                                        top_indices, output_dir, iteration)
        logger.info(f"Saved selected points to {csv_path}")

        # ---- Candidate uncertainty plots (AL + Baseline) ----
        # Reuse the variance already computed during selection instead of
        # running a separate inference pass. For deep_gp, subsample to keep
        # the baseline inference call affordable.
        try:
            _plot_size = min(50_000, n_candidates)
            if selection_strategy == "entropy_batch" and model_has_likelihood(model_type):
                # select_entropy_batch already evaluated all candidates — reuse.
                _all_phys = unnormalize_x(candidates_norm.cpu(), data_min, data_max)
                _all_var = _ent_var_all.cpu()
            else:
                # top_k path: candidates (physical) and pred_var already computed.
                _all_phys = candidates
                _all_var = pred_var

            _n_all = len(_all_phys)
            if _n_all > _plot_size:
                _plot_idx = torch.randperm(_n_all)[:_plot_size]
                _al_plot_pool = _all_phys[_plot_idx]
                _al_plot_var = _all_var[_plot_idx]
            else:
                _al_plot_pool = _all_phys
                _al_plot_var = _all_var

            plot_candidate_uncertainty(
                _al_plot_pool, _al_plot_var,
                al_hist_dir, "AL", iteration, logger,
            )
            _, _base_plot_var = compute_uncertainty_gp(
                baseline_model, _al_plot_pool, data_min, data_max,
                model_type=model_type, jitter=jitter, num_samples=gp_num_samples,
                logger=logger,
            )
            plot_candidate_uncertainty(
                _al_plot_pool, _base_plot_var,
                baseline_hist_dir, "Baseline", iteration, logger,
            )
        except Exception as _e:
            logger.warning(f"Candidate uncertainty plot failed: {_e}")

        all_selected_points.append({
            "iteration": iteration,
            "points": candidates[top_indices].numpy().tolist(),
            "uncertainties": pred_var[top_indices].numpy().tolist() if len(pred_var.shape) > 0 else [],
            "al_best_val_loss": al_results['best_val_loss'],
            "al_r2_score": al_results['r2_score'],
            "baseline_best_val_loss": baseline_results['best_val_loss'],
            "baseline_r2_score": baseline_results['r2_score'],
        })

        # ---- Advanced plots ----
        if advanced_plots:
            plot_advanced_diagnostics(
                al_model, x_val_norm, y_val_t, x_train_norm,
                model_type, threshold, len(PARAM_ORDER),
                iter_plots_dir, iteration, jitter=jitter,
                num_samples=gp_num_samples, logger=logger,
                lsp_fracs_eval=F_val_al,
            )

        # ---- Generate new models if requested ----
        new_X, new_Y, new_F = None, None, None
        if candidate_source == 'mcmc':
            # Oracle mode: every selected MCMC point is already labelled.
            # Skip SPheno entirely and feed the labels straight into augmentation.
            if _oracle_selected_pool_idx is None:
                raise RuntimeError("Oracle mode: missing _oracle_selected_pool_idx after selection")
            mcmc_consumed_mask[_oracle_selected_pool_idx] = True
            new_X = X_mcmc_pool[_oracle_selected_pool_idx].clone()
            new_Y = Y_mcmc_pool[_oracle_selected_pool_idx].clone()
            new_F = F_mcmc_pool[_oracle_selected_pool_idx].clone()
            all_selected_points[-1]["n_generated"] = len(new_X)
            logger.info(f"Oracle: marked {len(_oracle_selected_pool_idx)} MCMC pool indices as consumed "
                        f"({int(mcmc_consumed_mask.sum())}/{len(mcmc_consumed_mask)} cumulative)")
        elif generate_data:
            n_target = max(1, int(n_select * min_gen_fraction))
            logger.info(f"Generation target: {n_target} valid models "
                       f"({min_gen_fraction*100:.0f}% of {n_select})")

            collected_X, collected_Y, collected_F = [], [], []

            for attempt in range(max_gen_attempts):
                if attempt == 0:
                    attempt_candidates = candidates
                    attempt_pred_var = pred_var
                    attempt_indices = top_indices
                    attempt_csv = csv_path
                    attempt_dir = iter_dir
                else:
                    attempt_dir = iter_dir / f"retry_{attempt:03d}"
                    attempt_dir.mkdir(parents=True, exist_ok=True)

                    attempt_seed = seed * 10_000 + iteration * 1000 + attempt
                    attempt_candidates = generate_candidate_pool(n_candidates, seed=attempt_seed)
                    attempt_mean, attempt_pred_var = compute_uncertainty_gp(
                        al_model, attempt_candidates, data_min, data_max,
                        model_type=model_type, jitter=jitter, num_samples=gp_num_samples,
                        logger=logger,
                    )
                    if selection_strategy == "top_k_tol_only":
                        attempt_indices = select_top_uncertain_tol_only(
                            attempt_candidates, attempt_mean, attempt_pred_var.unsqueeze(1), n_select,
                            threshold=threshold,
                            tolerance_sampling=tolerance_sampling,
                            logger=logger,
                        )
                    else:
                        attempt_indices = select_top_uncertain_filtered(
                            attempt_candidates, attempt_mean, attempt_pred_var.unsqueeze(1), n_select,
                            threshold=threshold,
                            tolerance_sampling=tolerance_sampling,
                            proximity_sampling=proximity_sampling,
                            logger=logger,
                        )

                    param_names = [p.replace("IN_", "") for p in PARAM_ORDER]
                    df = pd.DataFrame(attempt_candidates[attempt_indices].numpy(),
                                     columns=param_names)
                    df["uncertainty"] = attempt_pred_var[attempt_indices].numpy()
                    attempt_csv = attempt_dir / "selected_points.csv"
                    df.to_csv(attempt_csv, index=False)

                logger.info(f"Generation attempt {attempt + 1}/{max_gen_attempts}...")
                ntuple_paths = generate_models_from_csv(
                    attempt_csv, attempt_dir, logger, n_workers=gen_workers
                )

                for ntuple_path in ntuple_paths:
                    batch_X, batch_Y, batch_F = load_generated_data(
                        ntuple_path, logger, return_lsp_fracs=True)
                    if batch_X is not None and len(batch_X) > 0:
                        collected_X.append(batch_X)
                        collected_Y.append(batch_Y)
                        collected_F.append(batch_F)

                n_collected = sum(len(x) for x in collected_X)
                logger.info(f"After attempt {attempt + 1}: "
                           f"{n_collected}/{n_target} target models collected")

                if n_collected >= n_target:
                    logger.info(f"Generation target reached after {attempt + 1} attempt(s)")
                    break
                if attempt < max_gen_attempts - 1:
                    logger.info("Below target, retrying...")

            if collected_X:
                new_X = torch.cat(collected_X)
                new_Y = torch.cat(collected_Y)
                new_F = torch.cat(collected_F)
                # Deduplicate: identical X rows from SPheno rounding can leak across train/val
                _, unique_idx = np.unique(new_X.numpy(), axis=0, return_index=True)
                if len(unique_idx) < len(new_X):
                    logger.info(f"Removing {len(new_X) - len(unique_idx)} duplicate generated points")
                    unique_idx = torch.from_numpy(np.sort(unique_idx))
                    new_X = new_X[unique_idx]
                    new_Y = new_Y[unique_idx]
                    new_F = new_F[unique_idx]
                logger.info(f"Total generated: {len(new_X)} unique training points")
                all_selected_points[-1]["n_generated"] = len(new_X)
            else:
                logger.warning("No valid models generated after all attempts")

        # ---- Augment AL data with newly generated points, split 80/20 into train/val ----
        if new_X is not None and new_Y is not None and len(new_X) > 0:
            # Filter new data too (Y > 0)
            new_valid = (new_Y.squeeze(-1) > 0)
            new_X = new_X[new_valid]
            new_Y = new_Y[new_valid]
            new_F = new_F[new_valid]

            # Remove new points that duplicate existing train or val data
            existing = torch.cat([X, X_val], dim=0).numpy()
            new_np = new_X.numpy()
            combined = np.concatenate([existing, new_np], axis=0)
            _, first_idx = np.unique(combined, axis=0, return_index=True)
            novel_mask = np.zeros(len(new_np), dtype=bool)
            for idx in first_idx:
                if idx >= len(existing):
                    novel_mask[idx - len(existing)] = True
            n_existing_dups = len(new_X) - novel_mask.sum()
            if n_existing_dups > 0:
                logger.info(f"Removing {n_existing_dups} generated points that duplicate existing data")
                new_X = new_X[novel_mask]
                new_Y = new_Y[novel_mask]
                new_F = new_F[novel_mask]

            if len(new_X) == 0:
                logger.warning("All generated points were duplicates of existing data")
            else:
                # Shuffle before splitting to avoid ordering bias from generation attempts
                perm_new = torch.randperm(len(new_X))
                new_X = new_X[perm_new]
                new_Y = new_Y[perm_new]
                new_F = new_F[perm_new]
                n_new_val = max(1, int(len(new_X) * val_fraction))
                n_new_train = len(new_X) - n_new_val
                logger.info(f"Augmenting AL: +{n_new_train} train, +{n_new_val} val "
                           f"(train: {len(X)}->{len(X)+n_new_train}, "
                           f"val: {len(X_val)}->{len(X_val)+n_new_val})")
                X = torch.cat([X, new_X[:n_new_train]], dim=0)
                Y = torch.cat([Y, new_Y[:n_new_train]], dim=0)
                F = torch.cat([F, new_F[:n_new_train]], dim=0)
                X_val = torch.cat([X_val, new_X[n_new_train:]], dim=0)
                Y_val = torch.cat([Y_val, new_Y[n_new_train:]], dim=0)
                F_val = torch.cat([F_val, new_F[n_new_train:]], dim=0)

                # Plot new-points-only histograms for AL
                idx_train_al_new = torch.arange(n_new_train)
                idx_val_al_new = torch.arange(n_new_train, len(new_X))
                plot_data_histograms(new_X, new_Y, idx_train_al_new, idx_val_al_new,
                                     al_hist_dir, "AL_new", iteration, logger,
                                     fixed_axes=True)

        prev_al_checkpoint = al_checkpoint_path

        # ---- Checkpoint run state for resume -------------------------------
        from pmssm.resume import save_state, capture_rng
        save_state(output_dir, {
            "iteration": iteration,
            "X": X, "Y": Y, "X_val": X_val, "Y_val": Y_val,
            "F": F, "F_val": F_val,
            "baseline_add_indices": baseline_add_indices,
            "prev_n_add_train": prev_n_add_train,
            "prev_n_add_val": prev_n_add_val,
            "all_selected_points": all_selected_points,
            "iteration_numbers": iteration_numbers,
            "al_train_losses": al_train_losses, "al_val_losses": al_val_losses,
            "al_r2_scores": al_r2_scores, "al_train_r2_scores": al_train_r2_scores,
            "al_n_train": al_n_train, "al_n_val": al_n_val,
            "baseline_train_losses": baseline_train_losses,
            "baseline_val_losses": baseline_val_losses,
            "baseline_r2_scores": baseline_r2_scores,
            "baseline_train_r2_scores": baseline_train_r2_scores,
            "baseline_n_train": baseline_n_train,
            "baseline_n_val": baseline_n_val,
            "al_on_base_val_losses": al_on_base_val_losses,
            "al_on_base_val_r2": al_on_base_val_r2,
            "base_on_al_val_losses": base_on_al_val_losses,
            "base_on_al_val_r2": base_on_al_val_r2,
            "al_on_mcmc_losses": al_on_mcmc_losses, "al_on_mcmc_r2": al_on_mcmc_r2,
            "baseline_on_mcmc_losses": baseline_on_mcmc_losses,
            "baseline_on_mcmc_r2": baseline_on_mcmc_r2,
            "al_on_static_random_losses": al_on_static_random_losses,
            "al_on_static_random_r2": al_on_static_random_r2,
            "baseline_on_static_random_losses": baseline_on_static_random_losses,
            "baseline_on_static_random_r2": baseline_on_static_random_r2,
            "eval_r2_scores": eval_r2_scores,
            "lengthscale_rows": lengthscale_rows,
            # Oracle (mcmc-candidate) state — present only when the run was
            # launched with --candidate-source mcmc.
            "mcmc_consumed_mask": mcmc_consumed_mask,
            "mcmc_pool_idx": mcmc_pool_idx,
            "mcmc_eval_idx": mcmc_eval_idx,
            "rng": capture_rng(),
        })
        logger.info(f"[resume] state.pt saved (iteration {iteration})")
        prev_baseline_checkpoint = baseline_checkpoint_path

    # ---- Plot iteration metrics ----
    al_metrics = {
        'train_losses': al_train_losses,
        'val_losses': al_val_losses,
        'r2_scores': al_r2_scores,
        'train_r2_scores': al_train_r2_scores,
        'cross_val_losses': al_on_base_val_losses,
        'cross_val_r2': al_on_base_val_r2,
        'mcmc_eval_losses': al_on_mcmc_losses,
        'mcmc_eval_r2': al_on_mcmc_r2,
        'static_random_eval_losses': al_on_static_random_losses,
        'static_random_eval_r2': al_on_static_random_r2,
        'n_train': al_n_train,
        'n_val': al_n_val,
    }
    baseline_metrics = {
        'train_losses': baseline_train_losses,
        'val_losses': baseline_val_losses,
        'r2_scores': baseline_r2_scores,
        'train_r2_scores': baseline_train_r2_scores,
        'cross_val_losses': base_on_al_val_losses,
        'cross_val_r2': base_on_al_val_r2,
        'mcmc_eval_losses': baseline_on_mcmc_losses,
        'mcmc_eval_r2': baseline_on_mcmc_r2,
        'static_random_eval_losses': baseline_on_static_random_losses,
        'static_random_eval_r2': baseline_on_static_random_r2,
        'n_train': baseline_n_train,
        'n_val': baseline_n_val,
    }

    if n_iterations > 1:
        plot_iteration_metrics(iteration_numbers, al_metrics, baseline_metrics,
                              output_dir, logger)

        from make_iteration_gifs import generate_gifs
        generate_gifs(output_dir, logger=logger)
    else:
        logger.info(f"Single iteration - AL: val_loss={al_val_losses[0]:.6f}, "
                   f"R²={al_r2_scores[0]:.4f}")
        logger.info(f"Single iteration - Baseline: val_loss={baseline_val_losses[0]:.6f}, "
                   f"R²={baseline_r2_scores[0]:.4f}")

    # ---- Save lengthscales ----
    if track_lengthscales and lengthscale_rows:
        ls_path = output_dir / "lengthscales.csv"
        pd.DataFrame(lengthscale_rows).to_csv(ls_path, index=False)
        logger.info(f"Lengthscales saved to {ls_path}")

    # ---- Representative-points trajectory plot (+ CSV) ----
    if repr_log:
        try:
            _rep_out, _rep_csv = plot_representative_trajectories(
                repr_log, repr_points['Y'], repr_points['cls'], repr_points['labels'],
                _target_value, output_dir, y_transform='zscore', target=target,
            )
            logger.info(f"Representative-points trajectory: {_rep_out}")
            logger.info(f"Representative-points CSV: {_rep_csv}")
        except Exception as _e:
            logger.warning(f"Representative-points plot failed: {_e}")

    # ---- Save summary ----
    summary = {
        "timestamp": timestamp,
        "config": {
            "target": target,
            "model_type": model_type,
            "n_iterations": n_iterations,
            "n_candidates": n_candidates,
            "n_select": n_select,
            "selection_strategy": selection_strategy,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "kernel": kernel,
            "lengthscale": lengthscale,
            "noise": noise,
            "jitter": jitter,
            "use_ard": use_ard,
            "use_dkl": use_dkl,
            "generate_data": generate_data,
            "compute_full_metrics": compute_full_metrics,
            "eval_data_path": eval_data_path,
            "warm_starting": warm_starting,
            "seed": seed,
        },
        "iterations": all_selected_points,
        "final_dataset_size": len(X),
    }
    if eval_r2_scores:
        summary["eval_r2_scores"] = eval_r2_scores

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("GP Active Learning Complete")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
