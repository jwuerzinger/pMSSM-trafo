"""
Active Learning pipeline for pMSSM relic density prediction using transformer models.

Uses MC Dropout for uncertainty estimation and supports both top-k variance
and entropy-based batch selection strategies.

Harmonized with active_learning_gp.py to use the unified pmssm package.
"""

import warnings
warnings.filterwarnings('ignore', message='.*enable_nested_tensor.*')

from pathlib import Path
from datetime import datetime
import logging
import structlog
import json
import multiprocessing as mp

import click
import numpy as np
import pandas as pd
import yaml
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import mean_squared_error, r2_score

# Import from unified pmssm package
from pmssm import (
    # Configuration
    PARAM_ORDER,
    PARAM_RANGES,
    CSV_TO_MODELGEN,
    TARGET_CONFIG,
    # Data operations
    load_pmssm_data,
    load_mcmc_data,
    make_split,
    compute_stats,
    PMSSMDataset,
    # Models
    PMSSMTransformer,
    PMSSMTransformerTabular,
    PMSSMFeedForward,
    is_transformer,
    get_model_name,
    # Selection
    generate_candidate_pool,
    select_top_uncertain,
    select_entropy_batch_mc,
    # Uncertainty
    compute_uncertainty_mc_dropout,
    # Training
    train_with_validation,
    train_model_worker,
    # Visualization
    plot_data_histograms,
    plot_parallel_coordinates,
    plot_iteration_metrics,
    # Logging
    setup_logging,
    setup_worker_logging,
    # Model generation
    generate_models_from_csv,
    load_generated_data,
    save_selected_points,
)
from pmssm.data import inverse_transform_y



# All helper functions (model generation, selection, uncertainty,
# visualization, logging) are now imported from the unified pmssm package


def cross_evaluate_transformer(model, stats, X_other, Y_other,
                                y_transform='log', target='DMRD', device='cpu',
                                return_predictions=False):
    """Evaluate a trained transformer on an arbitrary dataset.

    Uses the model's own training stats for normalization.
    Returns (mse_loss, r2) where mse_loss is in transformed space
    (matching the regular training/validation loss) and r2 is in physical space.
    If return_predictions=True, returns (mse_loss, r2, Y_true_phys, Y_pred_phys).
    """
    model.eval()
    model.to(device)
    mean_X, std_X, mean_Y, std_Y = stats

    # Normalize using the model's training stats
    X_norm = (X_other - mean_X) / std_X

    if y_transform == 'log':
        from pmssm.data import transform_y
        Y_transformed = transform_y(Y_other, target=target)
    else:
        Y_transformed = (Y_other - mean_Y) / std_Y

    # Batch inference to avoid OOM on large datasets (e.g., 410k MCMC samples)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_norm), 1024):
            preds.append(model(X_norm[i:i+1024].to(device)).cpu())
    Y_pred_transformed = torch.cat(preds, dim=0)

    # MSE in transformed space (same space as training loss)
    mse = ((Y_transformed.squeeze() - Y_pred_transformed.squeeze()) ** 2).mean().item()

    # R² in physical space (same as regular R² computation)
    if y_transform == 'log':
        Y_true_phys = inverse_transform_y(Y_transformed, target=target)
        Y_pred_phys = inverse_transform_y(Y_pred_transformed, target=target)
    else:
        Y_true_phys = Y_transformed * std_Y + mean_Y
        Y_pred_phys = Y_pred_transformed * std_Y + mean_Y

    ss_res = ((Y_true_phys - Y_pred_phys) ** 2).sum()
    ss_tot = ((Y_true_phys - Y_true_phys.mean()) ** 2).sum()
    r2 = (1 - (ss_res / ss_tot)).item()
    if return_predictions:
        return mse, r2, Y_true_phys.squeeze(), Y_pred_phys.squeeze()
    return mse, r2


def plot_eval_scatterplots(eval_results, iteration, plot_dir, logger, max_points=10_000):
    """Plot a grid of true-vs-predicted scatterplots for all model/dataset combinations.

    Args:
        eval_results: list of dicts with keys:
            'model_name', 'dataset_name', 'y_true', 'y_pred', 'loss', 'r2'
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


def load_config_with_sweep(config_file, sweep_index=None):
    """
    Load YAML config and optionally apply sweep combination.

    List-valued parameters are treated as sweep dimensions.
    The sweep_index selects one combination from the Cartesian product.

    Args:
        config_file: Path to YAML configuration file
        sweep_index: Optional index to select from parameter sweep

    Returns:
        dict of parameter name -> resolved value

    Example:
        config.yaml:
            epochs: [100, 200, 500]
            dropout: [0.1, 0.2]
            # Total: 3 × 2 = 6 combinations

        load_config_with_sweep('config.yaml', sweep_index=0)
        # Returns: {'epochs': 100, 'dropout': 0.1}
    """
    from itertools import product as iterproduct

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


@click.command()
@click.option('--testing/--no-testing', default=False, help="Run in testing mode (small data, few epochs).")
@click.option('--n-iterations', default=1, type=int, help="Number of active learning iterations.")
@click.option('--n-candidates', default=1000, type=int, help="Candidate pool size.")
@click.option('--n-select', default=10, type=int, help="Number of points to select per iteration.")
@click.option('--mc-samples', default=30, type=int, help="Number of MC dropout forward passes.")
@click.option('--epochs', default=2000, type=int, help="Training epochs per iteration (default: 2000).")
@click.option('--dropout', default=0.1, type=float, help="Dropout rate for MC dropout.")
@click.option('--n-datasets', default=None, type=int, help="Number of ROOT datasets to load.")
@click.option('--n-samples', default=None, type=int, help="Number of samples to use from data.")
@click.option('--val-fraction', default=0.2, type=float, help="Fraction of data reserved for validation (default: 0.2). Applied to initial data and each batch of new points.")
@click.option('--output-dir', default='active_learning_output', type=str, help="Output directory.")
@click.option('--generate-data/--no-generate-data', default=False, help="Generate new models using Run3ModelGen.")
@click.option('--min-gen-fraction', default=0.6, type=float, help="Minimum fraction of n-select that must be generated successfully before stopping retries (default: 0.6).")
@click.option('--max-gen-attempts', default=10, type=int, help="Maximum number of generation attempts per iteration (default: 10).")
@click.option('--gen-workers', default=1, type=int, help="Number of parallel genModels.py workers per generation attempt (default: 1).")
@click.option('--selection-strategy', default='entropy_batch', type=click.Choice(['top_k', 'entropy_batch']), help="Selection strategy: top_k or entropy_batch (default).")
@click.option('--entropy-blur', default=0.15, type=float, help="Entropy smoothing parameter (entropy_batch only).")
@click.option('--entropy-beta', default=50.0, type=float, help="Gibbs sampling temperature (entropy_batch only).")
@click.option('--entropy-pool-size', default=5000, type=int, help="Focused pool size for entropy_batch pre-filtering.")
@click.option('--candidate-generation', default='lhs', type=click.Choice(['uniform', 'lhs']),
              help="Candidate pool generation method: uniform random or Latin Hypercube Sampling (default: lhs).")
@click.option('--proximity-sampling', default=0.1, type=float,
              help="Gaussian proximity weighting width around target value (0 to disable, default: 0.1).")
@click.option('--tolerance-sampling', default=0.0, type=float,
              help="Hard cut: keep only candidates within ±tolerance of threshold in transformed space (0 to disable, default: 0).")
@click.option('--target-value', default=0.12, type=float,
              help="Target relic density value for proximity weighting (default: 0.12).")
@click.option('--config-file', default=None, type=str,
              help="YAML config file (overrides CLI args). Supports parameter sweeps.")
@click.option('--sweep-index', default=None, type=int,
              help="Sweep combination index (requires --config-file).")
@click.option('--early-stopping/--no-early-stopping', default=True,
              help="Enable early stopping on validation loss (default: enabled).")
@click.option('--patience', default=200, type=int,
              help="Early stopping patience (epochs without improvement, default: 200).")
@click.option('--warm-starting/--no-warm-starting', default=True,
              help="Warm-start from previous iteration checkpoint (default: enabled).")
@click.option('--eval-data-path', default=None, type=str,
              help="Path to external eval dataset (ROOT/CSV) for validation.")
@click.option('--compute-full-metrics/--no-compute-full-metrics', default=False,
              help="Compute comprehensive evaluation metrics (accuracy, MSE, RMSE).")
@click.option('--y-transform', default='log', type=click.Choice(['zscore', 'log']),
              help="Y transformation: 'log' (default, recommended) or 'zscore' (legacy).")
@click.option('--mcmc-data-dir', default=None, type=str,
              help="Directory containing MCMC ROOT files for static evaluation (e.g., data/19250082).")
@click.option('--static-eval-size', default=100_000, type=int,
              help="Number of models to reserve from the random pool as a static evaluation set (default: 100000).")
@click.option('--gpu-ids', default='2,3', type=str,
              help="Comma-separated GPU IDs for AL and baseline models (default: 2,3).")
def main(testing, n_iterations, n_candidates, n_select, mc_samples, epochs, dropout, n_datasets, n_samples, val_fraction, output_dir, generate_data, min_gen_fraction, max_gen_attempts, gen_workers, selection_strategy, entropy_blur, entropy_beta, entropy_pool_size, candidate_generation, proximity_sampling, tolerance_sampling, target_value, config_file, sweep_index, early_stopping, patience, warm_starting, eval_data_path, compute_full_metrics, y_transform, mcmc_data_dir, static_eval_size, gpu_ids):
    """
    Active learning pipeline for pMSSM relic density prediction.

    Trains PMSSMTransformerTabular, computes uncertainty via MC Dropout,
    and selects most informative points for data generation.
    """
    # Load config file and override parameters if provided
    if config_file is not None:
        cfg = load_config_with_sweep(config_file, sweep_index)

        # Map config keys to local variables
        _cfg_map = {
            'n_iterations': 'n_iterations',
            'n_candidates': 'n_candidates',
            'n_select': 'n_select',
            'mc_samples': 'mc_samples',
            'epochs': 'epochs',
            'dropout': 'dropout',
            'selection_strategy': 'selection_strategy',
            'entropy_blur': 'entropy_blur',
            'entropy_beta': 'entropy_beta',
            'entropy_pool_size': 'entropy_pool_size',
            'candidate_generation': 'candidate_generation',
            'proximity_sampling': 'proximity_sampling',
            'tolerance_sampling': 'tolerance_sampling',
            'target_value': 'target_value',
            'early_stopping': 'early_stopping',
            'patience': 'patience',
            'warm_starting': 'warm_starting',
            'compute_full_metrics': 'compute_full_metrics',
            'y_transform': 'y_transform',
        }

        # Override locals with config values
        for cfg_key, local_key in _cfg_map.items():
            if cfg_key in cfg:
                # Type conversion
                val = cfg[cfg_key]
                if cfg_key in ('n_iterations', 'n_candidates', 'n_select', 'mc_samples', 'epochs', 'entropy_pool_size', 'patience'):
                    val = int(val)
                elif cfg_key in ('dropout', 'entropy_blur', 'entropy_beta', 'proximity_sampling', 'target_value'):
                    val = float(val)
                elif cfg_key in ('early_stopping', 'warm_starting', 'compute_full_metrics'):
                    val = bool(val)
                # String values: selection_strategy, candidate_generation, y_transform
                locals()[local_key] = val

        # Re-assign variables from locals (Python locals() quirk workaround)
        n_iterations = locals().get('n_iterations', n_iterations)
        n_candidates = locals().get('n_candidates', n_candidates)
        n_select = locals().get('n_select', n_select)
        mc_samples = locals().get('mc_samples', mc_samples)
        epochs = locals().get('epochs', epochs)
        dropout = locals().get('dropout', dropout)
        selection_strategy = locals().get('selection_strategy', selection_strategy)
        entropy_blur = locals().get('entropy_blur', entropy_blur)
        entropy_beta = locals().get('entropy_beta', entropy_beta)
        entropy_pool_size = locals().get('entropy_pool_size', entropy_pool_size)
        candidate_generation = locals().get('candidate_generation', candidate_generation)
        proximity_sampling = locals().get('proximity_sampling', proximity_sampling)
        tolerance_sampling = locals().get('tolerance_sampling', tolerance_sampling)
        target_value = locals().get('target_value', target_value)
        early_stopping = locals().get('early_stopping', early_stopping)
        patience = locals().get('patience', patience)
        warm_starting = locals().get('warm_starting', warm_starting)
        compute_full_metrics = locals().get('compute_full_metrics', compute_full_metrics)
        y_transform = locals().get('y_transform', y_transform)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Increase n_candidates if needed:
    if n_candidates < n_select: n_candidates = n_select

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up main logging to output_dir/active_learning.log
    log_file, logger = setup_logging(timestamp, output_dir=output_dir)

    logger.info("=" * 60)
    logger.info("Active Learning Pipeline for pMSSM")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Output directory: {output_dir}")
    if config_file:
        logger.info(f"Config file: {config_file}" +
                    (f" (sweep index {sweep_index})" if sweep_index is not None else ""))

    # Apply testing mode defaults
    if testing:
        n_datasets = n_datasets if n_datasets is not None else 3
        n_samples = n_samples if n_samples is not None else 30
        epochs = 50  # Aligned with GP script (was 10)
        n_candidates = 100
        mc_samples = 10
        logger.info("Testing mode enabled")
    else:
        n_datasets = n_datasets if n_datasets is not None else -1
        n_samples = n_samples if n_samples is not None else None

    logger.info(f"Configuration:")
    logger.info(f"  n_iterations: {n_iterations}")
    logger.info(f"  n_candidates: {n_candidates}")
    logger.info(f"  n_select: {n_select}")
    logger.info(f"  mc_samples: {mc_samples}")
    logger.info(f"  epochs: {epochs}")
    logger.info(f"  dropout: {dropout}")
    logger.info(f"  n_datasets: {n_datasets}")
    logger.info(f"  n_samples: {n_samples if n_samples else 'all'}")
    logger.info(f"  val_fraction: {val_fraction}")
    logger.info(f"  selection_strategy: {selection_strategy}")
    if selection_strategy == 'entropy_batch':
        logger.info(f"  entropy_blur: {entropy_blur}")
        logger.info(f"  entropy_beta: {entropy_beta}")
        logger.info(f"  entropy_pool_size: {entropy_pool_size}")
    logger.info(f"  candidate_generation: {candidate_generation}")
    logger.info(f"  proximity_sampling: {proximity_sampling}")
    logger.info(f"  tolerance_sampling: {tolerance_sampling}")
    logger.info(f"  target_value: {target_value}")
    logger.info(f"  early_stopping: {early_stopping} (patience={patience})")
    logger.info(f"  warm_starting: {warm_starting}")
    logger.info(f"  y_transform: {y_transform}")
    logger.info(f"  compute_full_metrics: {compute_full_metrics}")
    if eval_data_path:
        logger.info(f"  eval_data_path: {eval_data_path}")
    logger.info(f"  generate_data: {generate_data}")
    if generate_data:
        logger.info(f"  min_gen_fraction: {min_gen_fraction} (target: {int(n_select * min_gen_fraction)} valid models per iteration)")
        logger.info(f"  max_gen_attempts: {max_gen_attempts}")
        logger.info(f"  gen_workers: {gen_workers}")

    gpu_id_list = [int(x.strip()) for x in gpu_ids.split(',')]
    AL_GPU_ID = gpu_id_list[0]
    BASELINE_GPU_ID = gpu_id_list[1] if len(gpu_id_list) > 1 else gpu_id_list[0]
    device = f"cuda:{AL_GPU_ID}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(AL_GPU_ID)}")

    # Load initial data
    logger.info("Loading data...")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    X, Y = load_pmssm_data(n_datasets=n_datasets, logger=logger, plot_dir=str(plots_dir))

    # Load MCMC evaluation dataset if provided
    X_mcmc, Y_mcmc = None, None
    if mcmc_data_dir is not None:
        X_mcmc, Y_mcmc = load_mcmc_data(data_dir=mcmc_data_dir, logger=logger)
        logger.info(f"MCMC evaluation dataset: {len(X_mcmc)} samples from {mcmc_data_dir}")

    # Store full dataset for baseline random sampling (before any truncation)
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

    # Determine if we can use parallel training (2+ GPUs for AL + baseline)
    use_parallel = torch.cuda.is_available() and torch.cuda.device_count() >= 2
    if use_parallel:
        logger.info(f"Parallel training enabled: {torch.cuda.device_count()} GPUs available")
        logger.info(f"  - Active Learning model on cuda:{AL_GPU_ID}")
        logger.info(f"  - Baseline model on cuda:{BASELINE_GPU_ID}")
        mp.set_start_method('spawn', force=True)
    else:
        logger.info("Sequential training (need 2+ GPUs for parallel)")

    # Run active learning iterations
    all_selected_points = []
    iteration_numbers = []

    # AL metrics
    al_train_losses = []
    al_val_losses = []
    al_r2_scores = []
    al_train_r2_scores = []
    al_n_train = []
    al_n_val = []

    # Baseline metrics
    baseline_train_losses = []
    baseline_val_losses = []
    baseline_r2_scores = []
    baseline_train_r2_scores = []
    baseline_n_train = []
    baseline_n_val = []

    # Cross-evaluation metrics (each model on the other's validation set)
    al_on_base_val_losses = []
    al_on_base_val_r2 = []
    base_on_al_val_losses = []
    base_on_al_val_r2 = []

    # Checkpoint tracking for warm-starting
    prev_al_checkpoint = None
    prev_baseline_checkpoint = None

    # Persistent baseline augmentation indices (grows each iteration)
    baseline_add_indices = torch.tensor([], dtype=torch.long)
    prev_n_add_train = 0
    prev_n_add_val = 0

    # External eval dataset (loaded lazily on first use)
    X_eval_full, Y_eval_full = None, None
    eval_r2_scores = []

    # Static evaluation metrics
    al_on_mcmc_losses, al_on_mcmc_r2 = [], []
    baseline_on_mcmc_losses, baseline_on_mcmc_r2 = [], []
    al_on_static_random_losses, al_on_static_random_r2 = [], []
    baseline_on_static_random_losses, baseline_on_static_random_r2 = [], []

    for iteration in range(1, n_iterations + 1):
        logger.info(f"=== Global Iteration {iteration} ===")

        # Create iteration directory for logs and plots
        iter_dir = output_dir / f"iteration_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        iter_plots_dir = iter_dir / "plots"
        iter_plots_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Iteration directory: {iter_dir}")

        # Build baseline dataset (train + val).
        # Baseline validation grows in parallel with AL validation.
        if iteration == 1:
            # First iteration: Baseline is an exact copy of AL (same train/val split).
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
                new_idx = available_indices[torch.randperm(len(available_indices))[:n_new_total]]
            else:
                logger.info(f"Baseline: sampling with replacement "
                            f"({n_new_total} needed, {len(available_indices)} available)")
                new_idx = available_indices[torch.randint(0, len(available_indices), (n_new_total,))]

            # Append new indices to persistent baseline indices
            # Layout: [train_indices... | val_indices...]
            baseline_add_indices = torch.cat([
                baseline_add_indices[:prev_n_add_train],   # existing train indices
                new_idx[:n_new_train],                     # new train indices
                baseline_add_indices[prev_n_add_train:],   # existing val indices
                new_idx[n_new_train:],                     # new val indices
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

        # Combine training + validation into single tensors for transformer workers.
        # Train indices come first, then val — contiguous and non-overlapping by construction.
        X_combined = torch.cat([X, X_val], dim=0)
        Y_combined = torch.cat([Y, Y_val], dim=0)
        idx_train_al = torch.arange(len(X))
        idx_val_al = torch.arange(len(X), len(X_combined))

        X_baseline_combined = torch.cat([X_baseline_train, X_baseline_val], dim=0)
        Y_baseline_combined = torch.cat([Y_baseline_train, Y_baseline_val], dim=0)
        idx_train_base = torch.arange(len(X_baseline_train))
        idx_val_base = torch.arange(len(X_baseline_train), len(X_baseline_combined))

        logger.info(f"AL: n_train={len(idx_train_al)}, n_val={len(idx_val_al)}")
        logger.info(f"Baseline: n_train={len(idx_train_base)}, n_val={len(idx_val_base)}")

        # Plot data distribution histograms and parallel coordinates for AL
        al_hist_dir = iter_plots_dir / "al"
        al_hist_dir.mkdir(parents=True, exist_ok=True)
        plot_data_histograms(X_combined, Y_combined, idx_train_al, idx_val_al, al_hist_dir, "AL", iteration, logger,
                             reference_X=X_mcmc, reference_Y=Y_mcmc, reference_label="MCMC")
        plot_parallel_coordinates(X_combined, idx_train_al, idx_val_al, al_hist_dir, "AL", iteration, logger)

        # Plot data distribution histograms and parallel coordinates for Baseline
        baseline_hist_dir = iter_plots_dir / "baseline"
        baseline_hist_dir.mkdir(parents=True, exist_ok=True)
        plot_data_histograms(X_baseline_combined, Y_baseline_combined, idx_train_base, idx_val_base, baseline_hist_dir, "Baseline", iteration, logger,
                             reference_X=X_static_random, reference_Y=Y_static_random, reference_label="Static Random")
        plot_parallel_coordinates(X_baseline_combined, idx_train_base, idx_val_base, baseline_hist_dir, "Baseline", iteration, logger)

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

        # Determine warm-start paths
        al_warm_start = prev_al_checkpoint if (iteration > 1 and warm_starting) else None
        baseline_warm_start = prev_baseline_checkpoint if (iteration > 1 and warm_starting) else None

        if use_parallel:
            # Train AL and Baseline in parallel on different GPUs
            al_queue = mp.Queue()
            baseline_queue = mp.Queue()

            al_process = mp.Process(
                target=train_model_worker,
                args=(AL_GPU_ID, X_combined, Y_combined, idx_train_al, idx_val_al, epochs, dropout, al_queue, "AL",
                      iter_dir, iter_plots_dir, al_checkpoint_path, al_warm_start,
                      early_stopping, patience, y_transform, "DMRD")
            )
            baseline_process = mp.Process(
                target=train_model_worker,
                args=(BASELINE_GPU_ID, X_baseline_combined, Y_baseline_combined, idx_train_base, idx_val_base, epochs,
                      dropout, baseline_queue, "Baseline", iter_dir, iter_plots_dir,
                      baseline_checkpoint_path, baseline_warm_start, early_stopping, patience,
                      y_transform, "DMRD")
            )

            al_process.start()
            baseline_process.start()

            # Read from queues BEFORE joining — joining first can deadlock
            # if the result object is large enough to fill the pipe buffer.
            al_results = al_queue.get()
            baseline_results = baseline_queue.get()

            al_process.join()
            baseline_process.join()
        else:
            # Sequential training
            logger.info("Training Active Learning model...")
            al_queue = mp.Queue()
            train_model_worker(device, X_combined, Y_combined, idx_train_al, idx_val_al, epochs, dropout,
                             al_queue, "AL", iter_dir, iter_plots_dir, al_checkpoint_path,
                             al_warm_start, early_stopping, patience, y_transform, "DMRD")
            al_results = al_queue.get()

            logger.info("Training Baseline model (random samples)...")
            baseline_queue = mp.Queue()
            train_model_worker(device, X_baseline_combined, Y_baseline_combined, idx_train_base, idx_val_base,
                             epochs, dropout, baseline_queue, "Baseline", iter_dir,
                             iter_plots_dir, baseline_checkpoint_path, baseline_warm_start,
                             early_stopping, patience, y_transform, "DMRD")
            baseline_results = baseline_queue.get()

        # Log results
        logger.info(f"AL metrics: train_loss={al_results['best_train_loss']:.6f}, val_loss={al_results['best_val_loss']:.6f}, R²={al_results['r2_score']:.4f}, train_R²={al_results['train_r2_score']:.4f}")
        logger.info(f"Baseline metrics: train_loss={baseline_results['best_train_loss']:.6f}, val_loss={baseline_results['best_val_loss']:.6f}, R²={baseline_results['r2_score']:.4f}, train_R²={baseline_results['train_r2_score']:.4f}")

        # Track metrics
        iteration_numbers.append(iteration)
        al_train_losses.append(al_results['best_train_loss'])
        al_val_losses.append(al_results['best_val_loss'])
        al_r2_scores.append(al_results['r2_score'])
        al_train_r2_scores.append(al_results['train_r2_score'])
        al_n_train.append(len(idx_train_al))
        al_n_val.append(len(idx_val_al))
        baseline_train_losses.append(baseline_results['best_train_loss'])
        baseline_val_losses.append(baseline_results['best_val_loss'])
        baseline_r2_scores.append(baseline_results['r2_score'])
        baseline_train_r2_scores.append(baseline_results['train_r2_score'])
        baseline_n_train.append(len(idx_train_base))
        baseline_n_val.append(len(idx_val_base))

        # Load the AL model checkpoint saved by the worker for MC Dropout uncertainty estimation.
        # This avoids training the AL model a second time.
        stats = compute_stats(X_combined, Y_combined, idx_train_al)
        model = PMSSMTransformerTabular(
            d_model=128, nhead=4, num_layers=3, dim_feedforward=512, dropout=dropout
        )
        model.load_state_dict(torch.load(al_checkpoint_path, map_location=device))
        logger.info(f"Loaded AL model from {al_checkpoint_path} for MC Dropout uncertainty estimation")

        # Cross-evaluation: each model on the other's validation set
        al_cross_loss, al_cross_r2, al_cross_yt, al_cross_yp = cross_evaluate_transformer(
            model, stats, X_baseline_val, Y_baseline_val,
            y_transform=y_transform, target='DMRD', device=device, return_predictions=True)

        baseline_stats = compute_stats(X_baseline_combined, Y_baseline_combined, idx_train_base)
        baseline_model = PMSSMTransformerTabular(
            d_model=128, nhead=4, num_layers=3, dim_feedforward=512, dropout=dropout)
        baseline_model.load_state_dict(torch.load(baseline_checkpoint_path, map_location=device))
        base_cross_loss, base_cross_r2, base_cross_yt, base_cross_yp = cross_evaluate_transformer(
            baseline_model, baseline_stats, X_val, Y_val,
            y_transform=y_transform, target='DMRD', device=device, return_predictions=True)

        logger.info(f"Cross-eval: AL_on_base_val_loss={al_cross_loss:.6f}, AL_on_base_val_R²={al_cross_r2:.4f}, base_on_al_val_loss={base_cross_loss:.6f}, base_on_al_val_R²={base_cross_r2:.4f}")
        al_on_base_val_losses.append(al_cross_loss)
        al_on_base_val_r2.append(al_cross_r2)
        base_on_al_val_losses.append(base_cross_loss)
        base_on_al_val_r2.append(base_cross_r2)

        # Collect scatterplot data: AL model on AL val, Baseline model on Base val
        # (own-val predictions via cross_evaluate for consistency)
        al_own_loss, al_own_r2, al_own_yt, al_own_yp = cross_evaluate_transformer(
            model, stats, X_val, Y_val,
            y_transform=y_transform, target='DMRD', device=device, return_predictions=True)
        base_own_loss, base_own_r2, base_own_yt, base_own_yp = cross_evaluate_transformer(
            baseline_model, baseline_stats, X_baseline_val, Y_baseline_val,
            y_transform=y_transform, target='DMRD', device=device, return_predictions=True)

        scatter_results = [
            dict(model_name="AL", dataset_name="AL Val", y_true=al_own_yt, y_pred=al_own_yp,
                 loss=al_own_loss, r2=al_own_r2, n=len(X_val)),
            dict(model_name="AL", dataset_name="Base Val", y_true=al_cross_yt, y_pred=al_cross_yp,
                 loss=al_cross_loss, r2=al_cross_r2, n=len(X_baseline_val)),
            dict(model_name="Baseline", dataset_name="AL Val", y_true=base_cross_yt, y_pred=base_cross_yp,
                 loss=base_cross_loss, r2=base_cross_r2, n=len(X_val)),
            dict(model_name="Baseline", dataset_name="Base Val", y_true=base_own_yt, y_pred=base_own_yp,
                 loss=base_own_loss, r2=base_own_r2, n=len(X_baseline_val)),
        ]

        # Evaluate on MCMC static dataset
        if X_mcmc is not None:
            mcmc_loss_al, mcmc_r2_al, mcmc_yt_al, mcmc_yp_al = cross_evaluate_transformer(
                model, stats, X_mcmc, Y_mcmc,
                y_transform=y_transform, target='DMRD', device=device, return_predictions=True)
            mcmc_loss_base, mcmc_r2_base, mcmc_yt_base, mcmc_yp_base = cross_evaluate_transformer(
                baseline_model, baseline_stats, X_mcmc, Y_mcmc,
                y_transform=y_transform, target='DMRD', device=device, return_predictions=True)
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

        # Evaluate on static random dataset
        if X_static_random is not None:
            static_loss_al, static_r2_al, static_yt_al, static_yp_al = cross_evaluate_transformer(
                model, stats, X_static_random, Y_static_random,
                y_transform=y_transform, target='DMRD', device=device, return_predictions=True)
            static_loss_base, static_r2_base, static_yt_base, static_yp_base = cross_evaluate_transformer(
                baseline_model, baseline_stats, X_static_random, Y_static_random,
                y_transform=y_transform, target='DMRD', device=device, return_predictions=True)
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

        # Evaluate on external dataset if provided
        if eval_data_path is not None:
            # Lazy load eval dataset on first use
            if X_eval_full is None:
                logger.info(f"Loading external eval dataset from {eval_data_path}")
                from pmssm import load_true_eval_dataset
                X_eval_full, Y_eval_full = load_true_eval_dataset(
                    eval_data_path, target=None, logger=logger  # No target transformation for transformer
                )

            # Compute R² on eval dataset
            from sklearn.metrics import r2_score
            model.eval()
            model.to(device)

            # Normalize eval data using training stats
            mean_X, std_X, mean_Y, std_Y = stats
            X_eval_norm = (X_eval_full - mean_X) / std_X
            Y_eval_norm = (Y_eval_full - mean_Y) / std_Y

            with torch.no_grad():
                y_pred_norm = model(X_eval_norm.to(device)).cpu()

            eval_r2 = r2_score(Y_eval_norm.numpy(), y_pred_norm.numpy())
            logger.info(f"External eval R²: {eval_r2:.4f}")
            eval_r2_scores.append(eval_r2)

        # Compute comprehensive metrics if requested
        if compute_full_metrics:

            # Get validation predictions
            model.eval()
            model.to(device)
            val_dataset = PMSSMDataset(X_combined, Y_combined, idx_val_al, stats)
            val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

            all_preds = []
            all_true = []
            for x_batch, y_batch in val_loader:
                with torch.no_grad():
                    preds = model(x_batch.to(device)).cpu()
                all_preds.append(preds)
                all_true.append(y_batch)

            y_pred_norm = torch.cat(all_preds).squeeze()
            y_true_norm = torch.cat(all_true).squeeze()

            # Denormalize for physical space metrics
            mean_X, std_X, mean_Y, std_Y = stats
            y_true = y_true_norm * std_Y + mean_Y
            y_pred = y_pred_norm * std_Y + mean_Y

            # Compute metrics in normalized space
            mse_norm = mean_squared_error(y_true_norm.numpy(), y_pred_norm.numpy())
            rmse_norm = np.sqrt(mse_norm)
            r2_norm = r2_score(y_true_norm.numpy(), y_pred_norm.numpy())

            # Compute metrics in physical space
            mse_phys = mean_squared_error(y_true.numpy(), y_pred.numpy())
            rmse_phys = np.sqrt(mse_phys)
            r2_phys = r2_score(y_true.numpy(), y_pred.numpy())

            # Classification accuracy (threshold at target_value in physical space)
            threshold_phys = target_value  # 0.12
            if y_transform == 'log':
                from pmssm.data import transform_y
                threshold_transformed_acc = transform_y(torch.tensor([threshold_phys]), target="DMRD").item()
            else:  # zscore
                threshold_transformed_acc = (threshold_phys - mean_Y) / std_Y
            acc = ((y_pred_norm >= threshold_transformed_acc) == (y_true_norm >= threshold_transformed_acc)).float().mean().item()

            metrics = {
                "iteration": iteration,
                "n_val": len(y_true),
                "mse_normalized": float(mse_norm),
                "rmse_normalized": float(rmse_norm),
                "r2_normalized": float(r2_norm),
                "mse_physical": float(mse_phys),
                "rmse_physical": float(rmse_phys),
                "r2_physical": float(r2_phys),
                "accuracy": float(acc),
                "threshold": float(threshold_phys),
            }

            # Save to CSV
            metrics_path = iter_dir / "metrics_al.csv"
            pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
            logger.info(f"Comprehensive metrics: MSE={mse_phys:.6f}, RMSE={rmse_phys:.6f}, "
                       f"R²={r2_phys:.4f}, Acc={acc:.4f}")
            logger.info(f"Metrics saved to {metrics_path}")

        # Generate candidate pool and select uncertain points
        logger.info(f"Generating {n_candidates} candidate points using {candidate_generation} sampling...")
        candidates = generate_candidate_pool(n_candidates, method=candidate_generation, seed=iteration)

        # Convert target value to transformed space for threshold
        mean_X, std_X, mean_Y, std_Y = stats
        if proximity_sampling > 0:
            if y_transform == 'log':
                from pmssm.data import transform_y
                threshold_transformed = transform_y(torch.tensor([target_value]), target="DMRD").item()
            else:  # zscore
                threshold_transformed = (target_value - mean_Y) / std_Y
        else:
            threshold_transformed = 0.0

        if selection_strategy == 'entropy_batch':
            pred_mean, pred_var, predictions = compute_uncertainty_mc_dropout(
                model, candidates, stats, mc_samples, device, logger, return_predictions=True
            )
            top_indices = select_entropy_batch_mc(
                candidates, predictions, pred_mean, pred_var,
                n_select, blur=entropy_blur, beta=entropy_beta,
                n_pool=entropy_pool_size,
                threshold=threshold_transformed, tolerance_sampling=tolerance_sampling,
                proximity_sampling=proximity_sampling,
                device=device, logger=logger
            )
        else:
            pred_mean, pred_var = compute_uncertainty_mc_dropout(
                model, candidates, stats, mc_samples, device, logger
            )
            if proximity_sampling > 0:
                # Apply proximity weighting to variance for top_k selection
                proximity = torch.exp(-((pred_mean.squeeze() - threshold_transformed) ** 2) / proximity_sampling)
                weighted_var = proximity.unsqueeze(1) * pred_var
                top_indices = select_top_uncertain(candidates, weighted_var, n_select)
                logger.info(f"Applied proximity weighting (σ={proximity_sampling:.3f}) around target={target_value:.3f}")
            else:
                top_indices = select_top_uncertain(candidates, pred_var, n_select)

        logger.info(f"Selected {len(top_indices)} points via {selection_strategy} (requested: {n_select}, available: {len(candidates)})")

        csv_path = save_selected_points(candidates, pred_var, top_indices, output_dir, iteration)
        logger.info(f"Saved selected points to {csv_path}")

        all_selected_points.append({
            "iteration": iteration,
            "points": candidates[top_indices].numpy().tolist(),
            "uncertainties": pred_var[top_indices].squeeze().numpy().tolist(),
            "al_best_val_loss": al_results['best_val_loss'],
            "al_r2_score": al_results['r2_score'],
            "baseline_best_val_loss": baseline_results['best_val_loss'],
            "baseline_r2_score": baseline_results['r2_score'],
        })

        # Generate new models if requested, with retry logic
        new_X, new_Y = None, None
        if generate_data:
            n_target = max(1, int(n_select * min_gen_fraction))
            logger.info(f"Generation target: {n_target} valid models ({min_gen_fraction*100:.0f}% of {n_select} selected, max {max_gen_attempts} attempts)")

            collected_X, collected_Y = [], []

            for attempt in range(max_gen_attempts):
                if attempt == 0:
                    # First attempt: use the already-generated candidates and saved CSV
                    attempt_candidates = candidates
                    attempt_pred_var = pred_var
                    attempt_indices = top_indices
                    attempt_csv = csv_path
                    attempt_dir = iter_dir
                else:
                    # Retry: draw a fresh random candidate pool and recompute uncertainty
                    attempt_dir = iter_dir / f"retry_{attempt:03d}"
                    attempt_dir.mkdir(parents=True, exist_ok=True)

                    attempt_seed = iteration * 1000 + attempt
                    attempt_candidates = generate_candidate_pool(n_candidates, method=candidate_generation, seed=attempt_seed)

                    if selection_strategy == 'entropy_batch':
                        attempt_mean, attempt_pred_var, attempt_preds = compute_uncertainty_mc_dropout(
                            model, attempt_candidates, stats, mc_samples, device, logger, return_predictions=True
                        )
                        attempt_indices = select_entropy_batch_mc(
                            attempt_candidates, attempt_preds, attempt_mean, attempt_pred_var,
                            n_select, blur=entropy_blur, beta=entropy_beta,
                            n_pool=entropy_pool_size,
                            threshold=threshold_transformed, tolerance_sampling=tolerance_sampling,
                            proximity_sampling=proximity_sampling,
                            device=device, logger=logger
                        )
                    else:
                        attempt_mean, attempt_pred_var = compute_uncertainty_mc_dropout(
                            model, attempt_candidates, stats, mc_samples, device, logger
                        )
                        if proximity_sampling > 0:
                            proximity = torch.exp(-((attempt_mean.squeeze() - threshold_transformed) ** 2) / proximity_sampling)
                            weighted_var = proximity.unsqueeze(1) * attempt_pred_var
                            attempt_indices = select_top_uncertain(attempt_candidates, weighted_var, n_select)
                        else:
                            attempt_indices = select_top_uncertain(attempt_candidates, attempt_pred_var, n_select)

                    param_names = [p.replace("IN_", "") for p in PARAM_ORDER]
                    df = pd.DataFrame(attempt_candidates[attempt_indices].numpy(), columns=param_names)
                    if selection_strategy != 'entropy_batch':
                        df["uncertainty"] = attempt_pred_var[attempt_indices].squeeze().numpy()
                    attempt_csv = attempt_dir / "selected_points.csv"
                    df.to_csv(attempt_csv, index=False)

                logger.info(f"Generation attempt {attempt + 1}/{max_gen_attempts} ({len(attempt_indices)} points)...")
                ntuple_paths = generate_models_from_csv(attempt_csv, attempt_dir, logger, n_workers=gen_workers)

                for ntuple_path in ntuple_paths:
                    batch_X, batch_Y = load_generated_data(ntuple_path, logger)
                    if batch_X is not None and len(batch_X) > 0:
                        collected_X.append(batch_X)
                        collected_Y.append(batch_Y)

                n_collected = sum(len(x) for x in collected_X)
                logger.info(f"After attempt {attempt + 1}: {n_collected}/{n_target} target models collected")

                if n_collected >= n_target:
                    logger.info(f"Generation target reached after {attempt + 1} attempt(s)")
                    break
                if attempt < max_gen_attempts - 1:
                    logger.info(f"Below target, retrying with next most-uncertain batch...")

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
                logger.info(f"Total generated: {len(new_X)} unique training points across {min(attempt + 1, max_gen_attempts)} attempt(s)")
                all_selected_points[-1]["n_generated"] = len(new_X)
            else:
                logger.warning("No valid models generated after all attempts")

        # Augment AL data with newly generated points, split 80/20 into train/val
        if new_X is not None and new_Y is not None and len(new_X) > 0:
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
                            f"(train: {len(X)}->{len(X)+n_new_train}, val: {len(X_val)}->{len(X_val)+n_new_val})")
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

        # Update previous checkpoints for warm-starting next iteration
        prev_al_checkpoint = al_checkpoint_path
        prev_baseline_checkpoint = baseline_checkpoint_path

    # Plot iteration metrics
    al_metrics = {
        'train_losses': al_train_losses,
        'val_losses': al_val_losses,
        'r2_scores': al_r2_scores,
        'train_r2_scores': al_train_r2_scores,
        'cross_val_losses': al_on_base_val_losses,
        'cross_val_r2': al_on_base_val_r2,
        'n_train': al_n_train,
        'n_val': al_n_val,
        'mcmc_eval_losses': al_on_mcmc_losses,
        'mcmc_eval_r2': al_on_mcmc_r2,
        'static_random_eval_losses': al_on_static_random_losses,
        'static_random_eval_r2': al_on_static_random_r2,
    }
    baseline_metrics = {
        'train_losses': baseline_train_losses,
        'val_losses': baseline_val_losses,
        'r2_scores': baseline_r2_scores,
        'train_r2_scores': baseline_train_r2_scores,
        'cross_val_losses': base_on_al_val_losses,
        'cross_val_r2': base_on_al_val_r2,
        'n_train': baseline_n_train,
        'n_val': baseline_n_val,
        'mcmc_eval_losses': baseline_on_mcmc_losses,
        'mcmc_eval_r2': baseline_on_mcmc_r2,
        'static_random_eval_losses': baseline_on_static_random_losses,
        'static_random_eval_r2': baseline_on_static_random_r2,
    }

    if n_iterations > 1:
        plot_iteration_metrics(iteration_numbers, al_metrics, baseline_metrics, output_dir, logger)

        from make_iteration_gifs import generate_gifs
        generate_gifs(output_dir, logger=logger)
    else:
        logger.info(f"Single iteration - AL: val_loss={al_val_losses[0]:.6f}, R²={al_r2_scores[0]:.4f}")
        logger.info(f"Single iteration - Baseline: val_loss={baseline_val_losses[0]:.6f}, R²={baseline_r2_scores[0]:.4f}")

    # Save summary
    summary = {
        "timestamp": timestamp,
        "config": {
            "n_iterations": n_iterations,
            "n_candidates": n_candidates,
            "n_select": n_select,
            "mc_samples": mc_samples,
            "epochs": epochs,
            "dropout": dropout,
            "generate_data": generate_data,
            "selection_strategy": selection_strategy,
            "early_stopping": early_stopping,
            "patience": patience,
            "warm_starting": warm_starting,
            "compute_full_metrics": compute_full_metrics,
        },
        "iterations": all_selected_points,
        "final_dataset_size": len(X),
        "al_metrics": al_metrics,
        "baseline_metrics": baseline_metrics,
    }

    if eval_r2_scores:
        summary["eval_r2_scores"] = eval_r2_scores

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("Active Learning Complete")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
