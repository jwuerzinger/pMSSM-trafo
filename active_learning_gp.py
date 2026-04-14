"""
Active Learning pipeline for pMSSM relic density prediction using Gaussian Process models.

Uses ExactGP, DeepGP, SparseGP, or MLP models from the al_pmssmwithgp submodule with
native GP posterior variance for uncertainty estimation.

Harmonized with active_learning.py to use the unified pmssm package.
"""

import warnings
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
    # Selection
    generate_candidate_pool,
    select_top_uncertain,
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
    # Logging
    setup_logging,
    # Model generation
    generate_models_from_csv,
    load_generated_data,
    save_selected_points,
    load_true_eval_dataset,
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


def plot_eval_scatterplots(eval_results, iteration, plot_dir, logger, max_points=10_000):
    """Plot a grid of true-vs-predicted scatterplots for all model/dataset combinations.

    Args:
        eval_results: list of dicts with keys:
            'model_name', 'dataset_name', 'y_true', 'y_pred', 'loss', 'r2', 'n'
        iteration: current iteration number
        plot_dir: directory to save the plot
        logger: logger instance
        max_points: max points to plot per panel (subsampled for speed)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(eval_results)
    if n == 0:
        return

    # Layout: one row per model, one column per dataset
    model_names = list(dict.fromkeys(r['model_name'] for r in eval_results))
    dataset_names = list(dict.fromkeys(r['dataset_name'] for r in eval_results))
    n_rows = len(model_names)
    n_cols = len(dataset_names)

    # Build lookup
    lookup = {(r['model_name'], r['dataset_name']): r for r in eval_results}

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)

    for row, model_name in enumerate(model_names):
        for col, dataset_name in enumerate(dataset_names):
            ax = axes[row, col]
            key = (model_name, dataset_name)
            if key not in lookup:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        fontsize=14, color="gray", transform=ax.transAxes)
                ax.set_xlabel("True")
                ax.set_ylabel("Predicted")
                ax.set_title(f"{model_name} on {dataset_name}", fontsize=10)
                continue

            r = lookup[key]
            y_true = r['y_true'].detach().numpy() if hasattr(r['y_true'], 'numpy') else r['y_true']
            y_pred = r['y_pred'].detach().numpy() if hasattr(r['y_pred'], 'numpy') else r['y_pred']

            # Subsample for plotting speed
            if len(y_true) > max_points:
                idx = np.random.default_rng(42).choice(len(y_true), max_points, replace=False)
                y_true = y_true[idx]
                y_pred = y_pred[idx]

            ax.scatter(y_true, y_pred, alpha=0.3, s=4, rasterized=True)
            vmin = min(y_true.min(), y_pred.min())
            vmax = max(y_true.max(), y_pred.max())
            ax.plot([vmin, vmax], [vmin, vmax], '--', color='grey', lw=1)
            ax.set_xlabel("True Omega h^2", fontsize=9)
            ax.set_ylabel("Predicted Omega h^2", fontsize=9)
            ax.set_title(
                f"{model_name} on {dataset_name}\n"
                f"MSE={r['loss']:.4f}, R2={r['r2']:.4f}, n={r['n']}",
                fontsize=10
            )
            ax.grid(True, alpha=0.2)

    fig.suptitle(f"Iteration {iteration} — True vs Predicted", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = plot_dir / f"scatterplots_iter_{iteration:03d}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved evaluation scatterplots to {out_path}")


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
        selected_indices: Indices into X_candidates_norm of selected points.
        per_point_entropy: Entropy scores for each selected point.
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

    batch_size = 100_000
    means_list, vars_list = [], []
    for i in range(0, len(X_candidates_norm), batch_size):
        x_batch = X_candidates_norm[i:i + batch_size].to(device)
        with torch.no_grad(), \
             gpytorch.settings.eval_cg_tolerance(1e-4), \
             gpytorch.settings.max_cg_iterations(300), \
             gpytorch.settings.fast_pred_var(False), \
             gpytorch.settings.fast_pred_samples(True), \
             gpytorch.settings.cholesky_jitter(jitter), \
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
         gpytorch.settings.cholesky_jitter(jitter), \
         gpytorch.settings.num_likelihood_samples(ns):
        preds_focus = model.likelihood(model(x_pool))
        if is_deep:
            mean = preds_focus.mean.detach().mean(dim=0).squeeze()
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
    return X_candidates_norm[original_indices], per_point_entropy


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
@click.option('--selection-strategy', default='entropy_batch', type=click.Choice(['top_k', 'entropy_batch']),
              help="Selection strategy: top_k or entropy_batch (default).")
@click.option('--entropy-blur', default=0.15, type=float, help="Entropy smoothing parameter (entropy_batch only).")
@click.option('--entropy-beta', default=50.0, type=float, help="Gibbs sampling temperature (entropy_batch only).")
@click.option('--tolerance-sampling', default=1.0, type=float,
              help="Hard cut: keep only candidates within ±tolerance of threshold in transformed space (0 to disable, default: 1.0).")
@click.option('--proximity-sampling', default=0.1, type=float,
              help="Gaussian proximity weighting width around target value (0 to disable, default: 0.1).")
@click.option('--entropy-pool-size', default=10_000, type=int, help="Focused pool size for entropy_batch pre-filtering.")
# Evaluation options
@click.option('--compute-full-metrics/--no-compute-full-metrics', default=False,
              help="Compute comprehensive GoF metrics (accuracy, chi2, pulls, etc.).")
@click.option('--eval-data-path', default=None, type=str, help="Path to true eval dataset (ROOT/CSV).")
@click.option('--mcmc-data-dir', default=None, type=str,
              help="Directory containing MCMC ROOT files for static evaluation (e.g. data/19250082).")
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
@click.option('--gpu-ids', default='0,1', type=str,
              help="Comma-separated GPU IDs for AL and baseline models (default: 0,1).")
def main(testing, n_iterations, n_candidates, n_select, n_datasets, n_samples, val_fraction,
         output_dir, generate_data, min_gen_fraction, max_gen_attempts, gen_workers,
         target, model_type, kernel, lengthscale, noise, jitter, learning_rate,
         epochs, early_stopping, patience,
         use_ard, use_dkl, feature_dim,
         num_hidden_dims, num_middle_dims, num_inducing_max, inducing_strategy,
         gp_num_samples, batch_size, warm_starting, m_nu, num_mixtures,
         selection_strategy, entropy_blur, entropy_beta,
         tolerance_sampling, proximity_sampling, entropy_pool_size,
         compute_full_metrics, eval_data_path, mcmc_data_dir, static_eval_size,
         track_lengthscales, advanced_plots,
         config_file, sweep_index, data_dir, gpu_ids):
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    threshold = TARGET_CONFIG[target]["threshold"]

    if n_candidates < n_select:
        n_candidates = n_select

    output_dir = Path(output_dir)
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
    X, Y = load_pmssm_data(n_datasets=n_datasets, logger=logger,
                           plot_dir=str(plots_dir), data_dir=data_dir)

    # Store full dataset for baseline random sampling
    X_full, Y_full = X.clone(), Y.clone()

    # Select n_samples from loaded data (or use all), then split 80/20 into train/val.
    if n_samples is not None:
        if n_samples > len(X):
            raise ValueError(f"n_samples={n_samples} exceeds available data ({len(X)})")
        X = X[:n_samples].clone()
        Y = Y[:n_samples].clone()
    else:
        X = X.clone()
        Y = Y.clone()

    # Split initial data into train/val using val_fraction
    n_total_init = len(X)
    n_val_init = max(1, int(n_total_init * val_fraction))
    n_train_init = n_total_init - n_val_init

    # Use a fixed permutation so the split is reproducible
    perm = torch.randperm(n_total_init, generator=torch.Generator().manual_seed(42))
    idx_train_perm = perm[:n_train_init]
    idx_val_perm = perm[n_train_init:]

    X_val = X[idx_val_perm].clone()
    Y_val = Y[idx_val_perm].clone()
    X = X[idx_train_perm].clone()
    Y = Y[idx_train_perm].clone()

    logger.info(f"Initial split ({1-val_fraction:.0%} train / {val_fraction:.0%} val): "
                f"{len(X)} train + {len(X_val)} val = {n_total_init} total")

    # Track which indices from X_full are reserved (train + val).
    # Baseline random sampling will exclude all of these.
    initial_reserved = n_samples if n_samples is not None else len(X_full)
    initial_al_indices = torch.arange(initial_reserved)

    logger.info(f"Baseline pool: X_full={X_full.shape}, reserved [0..{initial_reserved-1}] excluded from sampling")

    # Load MCMC evaluation data if provided
    X_mcmc, Y_mcmc = None, None
    if mcmc_data_dir is not None:
        X_mcmc, Y_mcmc = load_mcmc_data(data_dir=mcmc_data_dir, logger=logger)
        logger.info(f"Loaded MCMC evaluation data: {len(X_mcmc)} samples")

    # Carve out static random evaluation set from X_full (after initial reserved block)
    X_static_random, Y_static_random = None, None
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

    for iteration in range(1, n_iterations + 1):
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

            X_baseline_train = torch.cat([X[:n_train_init], X_add[:n_add_train]])
            Y_baseline_train = torch.cat([Y[:n_train_init], Y_add[:n_add_train]])
            X_baseline_val = torch.cat([X_val[:n_val_init], X_add[n_add_train:]])
            Y_baseline_val = torch.cat([Y_val[:n_val_init], Y_add[n_add_train:]])
            logger.info(f"Baseline dataset: {n_train_init}+{n_add_train}={len(X_baseline_train)} train, "
                       f"{n_val_init}+{n_add_val}={len(X_baseline_val)} val")

        # ---- Assign train/val tensors for this iteration ----
        X_train_al = X
        Y_train_al = Y
        X_val_al = X_val
        Y_val_al = Y_val

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
                      gp_num_samples, al_warm_start, target, effective_patience),
            )
            baseline_process = mp.Process(
                target=train_gp_worker,
                args=(BASELINE_GPU_ID, X_train_base, Y_train_base, X_val_base, Y_val_base,
                      data_min, data_max, model_type, len(PARAM_ORDER), gp_kwargs,
                      learning_rate, epochs, batch_size, jitter,
                      baseline_queue, "Baseline", iter_dir, iter_plots_dir,
                      baseline_checkpoint_path,
                      gp_num_samples, baseline_warm_start, target, effective_patience),
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
                 loss=al_own_loss, r2=al_own_r2, n=len(X_val_al)),
            dict(model_name="AL", dataset_name="Base Val", y_true=al_cross_yt, y_pred=al_cross_yp,
                 loss=al_cross_loss, r2=al_cross_r2, n=len(X_baseline_val)),
            dict(model_name="Baseline", dataset_name="AL Val", y_true=base_cross_yt, y_pred=base_cross_yp,
                 loss=base_cross_loss, r2=base_cross_r2, n=len(X_val_al)),
            dict(model_name="Baseline", dataset_name="Base Val", y_true=base_own_yt, y_pred=base_own_yp,
                 loss=base_own_loss, r2=base_own_r2, n=len(X_baseline_val)),
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
                y_true=mcmc_yt_al, y_pred=mcmc_yp_al, loss=mcmc_loss_al, r2=mcmc_r2_al, n=len(X_mcmc)))
            scatter_results.append(dict(model_name="Baseline", dataset_name="MCMC",
                y_true=mcmc_yt_base, y_pred=mcmc_yp_base, loss=mcmc_loss_base, r2=mcmc_r2_base, n=len(X_mcmc)))

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
                y_true=static_yt_al, y_pred=static_yp_al, loss=static_loss_al, r2=static_r2_al, n=len(X_static_random)))
            scatter_results.append(dict(model_name="Baseline", dataset_name="Static Rnd",
                y_true=static_yt_base, y_pred=static_yp_base, loss=static_loss_base, r2=static_r2_base, n=len(X_static_random)))

        plot_eval_scatterplots(scatter_results, iteration, iter_plots_dir, logger)

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

        # ---- Candidate uncertainty plots (AL + Baseline) ----
        # Compute uncertainty on the full candidate pool for both models and
        # plot std vs each input parameter. Done BEFORE selection to avoid
        # `candidates` being overwritten by `selected_points_phys` in the
        # entropy_batch path.
        try:
            _candidate_plot_pool = generate_candidate_pool(
                min(n_candidates, 50_000), seed=iteration,
            )
            _al_plot_mean, _al_plot_var = compute_uncertainty_gp(
                al_model, _candidate_plot_pool, data_min, data_max,
                model_type=model_type, jitter=jitter, num_samples=gp_num_samples,
                logger=logger,
            )
            plot_candidate_uncertainty(
                _candidate_plot_pool, _al_plot_var,
                al_hist_dir, "AL", iteration, logger,
            )
            _, _base_plot_var = compute_uncertainty_gp(
                baseline_model, _candidate_plot_pool, data_min, data_max,
                model_type=model_type, jitter=jitter, num_samples=gp_num_samples,
                logger=logger,
            )
            plot_candidate_uncertainty(
                _candidate_plot_pool, _base_plot_var,
                baseline_hist_dir, "Baseline", iteration, logger,
            )
        except Exception as _e:
            logger.warning(f"Candidate uncertainty plot failed: {_e}")

        # ---- Select new points ----
        if selection_strategy == "entropy_batch" and model_has_likelihood(model_type):
            logger.info("Using entropy-based batch selection...")
            candidates_norm = normalize_x(
                generate_candidate_pool(n_candidates, seed=iteration),
                data_min, data_max,
            ).to(device)
            selected_points_norm, per_point_entropy = select_entropy_batch(
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

            logger.info(f"Entropy selection: {len(selected_points_phys)} points selected")
        else:
            # Standard top-K by variance
            if selection_strategy == "entropy_batch" and not model_has_likelihood(model_type):
                logger.warning(f"entropy_batch not supported for {model_type}; "
                              "falling back to top_k")

            candidates = generate_candidate_pool(n_candidates, seed=iteration)
            logger.info(f"Generating {n_candidates} candidate points...")

            _pred_mean, pred_var = compute_uncertainty_gp(
                al_model, candidates, data_min, data_max,
                model_type=model_type, jitter=jitter, num_samples=gp_num_samples,
                logger=logger,
            )

            top_indices = select_top_uncertain(candidates, pred_var.unsqueeze(1), n_select)

        logger.info(f"Selected {len(top_indices)} most uncertain points")

        csv_path = save_selected_points(candidates, pred_var.unsqueeze(1),
                                        top_indices, output_dir, iteration)
        logger.info(f"Saved selected points to {csv_path}")

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
            )

        # ---- Generate new models if requested ----
        new_X, new_Y = None, None
        if generate_data:
            n_target = max(1, int(n_select * min_gen_fraction))
            logger.info(f"Generation target: {n_target} valid models "
                       f"({min_gen_fraction*100:.0f}% of {n_select})")

            collected_X, collected_Y = [], []

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

                    attempt_seed = iteration * 1000 + attempt
                    attempt_candidates = generate_candidate_pool(n_candidates, seed=attempt_seed)
                    _, attempt_pred_var = compute_uncertainty_gp(
                        al_model, attempt_candidates, data_min, data_max,
                        model_type=model_type, jitter=jitter, num_samples=gp_num_samples,
                        logger=logger,
                    )
                    attempt_indices = select_top_uncertain(
                        attempt_candidates, attempt_pred_var.unsqueeze(1), n_select
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
                    batch_X, batch_Y = load_generated_data(ntuple_path, logger)
                    if batch_X is not None and len(batch_X) > 0:
                        collected_X.append(batch_X)
                        collected_Y.append(batch_Y)

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
                # Deduplicate: identical X rows from SPheno rounding can leak across train/val
                _, unique_idx = np.unique(new_X.numpy(), axis=0, return_index=True)
                if len(unique_idx) < len(new_X):
                    logger.info(f"Removing {len(new_X) - len(unique_idx)} duplicate generated points")
                    unique_idx = torch.from_numpy(np.sort(unique_idx))
                    new_X = new_X[unique_idx]
                    new_Y = new_Y[unique_idx]
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

            if len(new_X) == 0:
                logger.warning("All generated points were duplicates of existing data")
            else:
                # Shuffle before splitting to avoid ordering bias from generation attempts
                perm_new = torch.randperm(len(new_X))
                new_X = new_X[perm_new]
                new_Y = new_Y[perm_new]
                n_new_val = max(1, int(len(new_X) * val_fraction))
                n_new_train = len(new_X) - n_new_val
                logger.info(f"Augmenting AL: +{n_new_train} train, +{n_new_val} val "
                           f"(train: {len(X)}->{len(X)+n_new_train}, "
                           f"val: {len(X_val)}->{len(X_val)+n_new_val})")
                X = torch.cat([X, new_X[:n_new_train]], dim=0)
                Y = torch.cat([Y, new_Y[:n_new_train]], dim=0)
                X_val = torch.cat([X_val, new_X[n_new_train:]], dim=0)
                Y_val = torch.cat([Y_val, new_Y[n_new_train:]], dim=0)

                # Plot new-points-only histograms for AL
                idx_train_al_new = torch.arange(n_new_train)
                idx_val_al_new = torch.arange(n_new_train, len(new_X))
                plot_data_histograms(new_X, new_Y, idx_train_al_new, idx_val_al_new,
                                     al_hist_dir, "AL_new", iteration, logger,
                                     fixed_axes=True)

        prev_al_checkpoint = al_checkpoint_path
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
