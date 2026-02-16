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

# Import from unified pmssm package
from pmssm import (
    # Configuration
    PARAM_ORDER,
    PARAM_RANGES,
    CSV_TO_MODELGEN,
    TARGET_CONFIG,
    # Data operations
    load_pmssm_data,
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
    plot_iteration_metrics,
    # Logging
    setup_logging,
    setup_worker_logging,
    # Model generation
    generate_models_from_csv,
    load_generated_data,
    save_selected_points,
)



# All helper functions (model generation, selection, uncertainty,
# visualization, logging) are now imported from the unified pmssm package


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
def main(testing, n_iterations, n_candidates, n_select, mc_samples, epochs, dropout, n_datasets, n_samples, output_dir, generate_data, min_gen_fraction, max_gen_attempts, gen_workers, selection_strategy, entropy_blur, entropy_beta, entropy_pool_size, candidate_generation, proximity_sampling, target_value, config_file, sweep_index, early_stopping, patience, warm_starting, eval_data_path, compute_full_metrics, y_transform):
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
    logger.info(f"  selection_strategy: {selection_strategy}")
    if selection_strategy == 'entropy_batch':
        logger.info(f"  entropy_blur: {entropy_blur}")
        logger.info(f"  entropy_beta: {entropy_beta}")
        logger.info(f"  entropy_pool_size: {entropy_pool_size}")
    logger.info(f"  candidate_generation: {candidate_generation}")
    logger.info(f"  proximity_sampling: {proximity_sampling}")
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load initial data
    logger.info("Loading data...")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    X, Y = load_pmssm_data(n_datasets=n_datasets, logger=logger, plot_dir=str(plots_dir))

    # Store full dataset for baseline random sampling (before any truncation)
    X_full, Y_full = X.clone(), Y.clone()

    # Track which indices from X_full are used for initial AL dataset
    initial_al_size = n_samples if n_samples is not None else len(X)
    initial_al_indices = torch.arange(initial_al_size)  # Indices [0, 1, ..., n_samples-1]

    if n_samples is not None:
        X = X[:n_samples]
        Y = Y[:n_samples]
        logger.info(f"Using first {n_samples} samples for initial AL training")

    logger.info(f"AL dataset shape: X={X.shape}, Y={Y.shape}")
    logger.info(f"Baseline pool shape: X_full={X_full.shape}, Y_full={Y_full.shape}")
    logger.info(f"Initial AL indices from X_full: [0, ..., {initial_al_size-1}] (will be excluded from baseline sampling)")

    # Determine if we can use parallel training (2+ GPUs for AL + baseline)
    AL_GPU_ID = 2
    BASELINE_GPU_ID = 3
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
    al_n_train = []
    al_n_val = []

    # Baseline metrics
    baseline_train_losses = []
    baseline_val_losses = []
    baseline_r2_scores = []
    baseline_n_train = []
    baseline_n_val = []

    # Checkpoint tracking for warm-starting
    prev_al_checkpoint = None
    prev_baseline_checkpoint = None

    # External eval dataset (loaded lazily on first use)
    X_eval_full, Y_eval_full = None, None
    eval_r2_scores = []

    for iteration in range(1, n_iterations + 1):
        logger.info(f"=== Global Iteration {iteration} ===")

        # Create iteration directory for logs and plots
        iter_dir = output_dir / f"iteration_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        iter_plots_dir = iter_dir / "plots"
        iter_plots_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Iteration directory: {iter_dir}")

        # Create baseline dataset
        if iteration == 1:
            # First iteration: Baseline is an exact copy of the AL dataset.
            # Use a direct clone to guarantee byte-for-byte identity without relying
            # on index construction logic.
            X_baseline = X.clone()
            Y_baseline = Y.clone()
            assert X.shape == X_baseline.shape and torch.allclose(X, X_baseline), \
                "BUG: AL and Baseline datasets must be identical at iteration 1!"
            logger.info(f"Iteration 1: Both models use identical dataset ({len(X_baseline)} samples) — verified by allclose check")
        else:
            # Iteration 2+: Baseline grows by sampling from X_full excluding initial AL indices
            n_current = len(X)
            all_indices = torch.arange(len(X_full))
            mask = torch.ones(len(X_full), dtype=torch.bool)
            mask[initial_al_indices] = False
            available_indices = all_indices[mask]

            logger.info(f"Baseline sampling: {len(available_indices)} indices available from X_full (excluding initial {len(initial_al_indices)} AL indices)")

            n_additional = n_current - len(initial_al_indices)

            if n_additional <= len(available_indices):
                additional_indices = available_indices[torch.randperm(len(available_indices))[:n_additional]]
            else:
                logger.info(f"Baseline needs {n_additional} samples but only {len(available_indices)} available - sampling with replacement")
                additional_indices = available_indices[torch.randint(0, len(available_indices), (n_additional,))]

            baseline_indices = torch.cat([initial_al_indices, additional_indices])
            X_baseline = X_full[baseline_indices]
            Y_baseline = Y_full[baseline_indices]
            logger.info(f"Baseline dataset: {len(initial_al_indices)} initial + {n_additional} random = {len(baseline_indices)} samples")

        # Create train/val split indices based on AL dataset size
        # Use the same indices for both AL and Baseline to ensure identical dataset sizes
        idx_train, idx_val = make_split(X, logger=logger)
        idx_train_al = idx_train
        idx_val_al = idx_val
        idx_train_base = idx_train
        idx_val_base = idx_val

        # Plot data distribution histograms for AL
        al_hist_dir = iter_plots_dir / "al"
        al_hist_dir.mkdir(parents=True, exist_ok=True)
        plot_data_histograms(X, Y, idx_train_al, idx_val_al, al_hist_dir, "AL", iteration, logger)

        # Plot data distribution histograms for Baseline
        baseline_hist_dir = iter_plots_dir / "baseline"
        baseline_hist_dir.mkdir(parents=True, exist_ok=True)
        plot_data_histograms(X_baseline, Y_baseline, idx_train_base, idx_val_base, baseline_hist_dir, "Baseline", iteration, logger)

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
                args=(AL_GPU_ID, X, Y, idx_train_al, idx_val_al, epochs, dropout, al_queue, "AL",
                      iter_dir, iter_plots_dir, al_checkpoint_path, al_warm_start,
                      early_stopping, patience, y_transform, "DMRD")
            )
            baseline_process = mp.Process(
                target=train_model_worker,
                args=(BASELINE_GPU_ID, X_baseline, Y_baseline, idx_train_base, idx_val_base, epochs,
                      dropout, baseline_queue, "Baseline", iter_dir, iter_plots_dir,
                      baseline_checkpoint_path, baseline_warm_start, early_stopping, patience,
                      y_transform, "DMRD")
            )

            al_process.start()
            baseline_process.start()

            al_process.join()
            baseline_process.join()

            al_results = al_queue.get()
            baseline_results = baseline_queue.get()
        else:
            # Sequential training
            logger.info("Training Active Learning model...")
            al_queue = mp.Queue()
            train_model_worker(device, X, Y, idx_train_al, idx_val_al, epochs, dropout,
                             al_queue, "AL", iter_dir, iter_plots_dir, al_checkpoint_path,
                             al_warm_start, early_stopping, patience, y_transform, "DMRD")
            al_results = al_queue.get()

            logger.info("Training Baseline model (random samples)...")
            baseline_queue = mp.Queue()
            train_model_worker(device, X_baseline, Y_baseline, idx_train_base, idx_val_base,
                             epochs, dropout, baseline_queue, "Baseline", iter_dir,
                             iter_plots_dir, baseline_checkpoint_path, baseline_warm_start,
                             early_stopping, patience, y_transform, "DMRD")
            baseline_results = baseline_queue.get()

        # Log results
        logger.info(f"AL metrics: train_loss={al_results['best_train_loss']:.6f}, val_loss={al_results['best_val_loss']:.6f}, R²={al_results['r2_score']:.4f}")
        logger.info(f"Baseline metrics: train_loss={baseline_results['best_train_loss']:.6f}, val_loss={baseline_results['best_val_loss']:.6f}, R²={baseline_results['r2_score']:.4f}")

        # Track metrics
        iteration_numbers.append(iteration)
        al_train_losses.append(al_results['best_train_loss'])
        al_val_losses.append(al_results['best_val_loss'])
        al_r2_scores.append(al_results['r2_score'])
        al_n_train.append(len(idx_train_al))
        al_n_val.append(len(idx_val_al))
        baseline_train_losses.append(baseline_results['best_train_loss'])
        baseline_val_losses.append(baseline_results['best_val_loss'])
        baseline_r2_scores.append(baseline_results['r2_score'])
        baseline_n_train.append(len(idx_train_base))
        baseline_n_val.append(len(idx_val_base))

        # Load the AL model checkpoint saved by the worker for MC Dropout uncertainty estimation.
        # This avoids training the AL model a second time.
        stats = compute_stats(X, Y, idx_train_al)
        model = PMSSMTransformerTabular(
            d_model=128, nhead=4, num_layers=3, dim_feedforward=512, dropout=dropout
        )
        model.load_state_dict(torch.load(al_checkpoint_path, map_location=device))
        logger.info(f"Loaded AL model from {al_checkpoint_path} for MC Dropout uncertainty estimation")

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
            from sklearn.metrics import mean_squared_error, r2_score
            import numpy as np

            # Get validation predictions
            model.eval()
            model.to(device)
            val_dataset = PMSSMDataset(X, Y, idx_val_al, stats)
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
                threshold=threshold_transformed, proximity_sampling=proximity_sampling,
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
                            threshold=threshold_transformed, proximity_sampling=proximity_sampling,
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
                logger.info(f"Total generated: {len(new_X)} valid training points across {min(attempt + 1, max_gen_attempts)} attempt(s)")
                all_selected_points[-1]["n_generated"] = len(new_X)
            else:
                logger.warning("No valid models generated after all attempts")

        # Augment training data with newly generated points for next iteration
        if new_X is not None and new_Y is not None and len(new_X) > 0:
            logger.info(f"Augmenting training data: {len(X)} + {len(new_X)} = {len(X) + len(new_X)} samples")
            X = torch.cat([X, new_X], dim=0)
            Y = torch.cat([Y, new_Y], dim=0)

        # Update previous checkpoints for warm-starting next iteration
        prev_al_checkpoint = al_checkpoint_path
        prev_baseline_checkpoint = baseline_checkpoint_path

    # Plot iteration metrics
    al_metrics = {
        'train_losses': al_train_losses,
        'val_losses': al_val_losses,
        'r2_scores': al_r2_scores,
        'n_train': al_n_train,
        'n_val': al_n_val,
    }
    baseline_metrics = {
        'train_losses': baseline_train_losses,
        'val_losses': baseline_val_losses,
        'r2_scores': baseline_r2_scores,
        'n_train': baseline_n_train,
        'n_val': baseline_n_val,
    }

    if n_iterations > 1:
        plot_iteration_metrics(iteration_numbers, al_metrics, baseline_metrics, output_dir, logger)
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
