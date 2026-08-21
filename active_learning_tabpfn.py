"""
Active Learning pipeline for pMSSM relic density prediction using TabPFN.

Uses TabPFN's native predictive variance for uncertainty estimation and
ensemble diversity (multiple random_state runs) for entropy-based batch selection.

Based on active_learning.py — same loop structure, data generation, and evaluation.
"""

from pathlib import Path
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import structlog
import json
import random
import re
import time

import click
import numpy as np
import pandas as pd
import yaml
import torch

from sklearn.metrics import mean_squared_error, r2_score
from tabpfn import TabPFNRegressor

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
    # Selection
    generate_candidate_pool,
    select_top_uncertain,
    select_top_uncertain_filtered,
    select_top_uncertain_tol_only,
    select_tol_only_random,
    select_entropy_batch_mc,
    # Visualization
    plot_data_histograms,
    plot_parallel_coordinates,
    plot_candidate_uncertainty,
    plot_iteration_metrics,
    plot_eval_scatterplots,
    pick_representative_points,
    plot_representative_trajectories,
    # Logging
    setup_logging,
    # Model generation
    generate_models_from_csv,
    load_generated_data,
    save_selected_points,
)
from pmssm.data import transform_y, inverse_transform_y
from pmssm.accuracy import binary_accuracy, write_iter_accuracies



# All helper functions (model generation, selection, uncertainty,
# visualization, logging) are now imported from the unified pmssm package


def fit_tabpfn(X_train, Y_train, device="cuda:0", target="DMRD"):
    """Fit a TabPFN model on training data.

    Args:
        X_train: (N, 19) tensor in physical space
        Y_train: (N, 1) tensor in physical space (raw Omega h^2)
    Returns:
        model: Fitted TabPFNRegressor
        y_train_t: log-transformed training targets (numpy, 1D)
    """
    y_train_t = transform_y(Y_train, target=target).squeeze().numpy()
    model = TabPFNRegressor(device=device)
    model.fit(X_train.numpy(), y_train_t)
    return model, y_train_t


def tabpfn_predict(model, X, batch_size=10_000):
    """Predict with a fitted TabPFN model, batched to avoid OOM.

    Args:
        model: Fitted TabPFNRegressor
        X: (N, D) tensor in physical space
        batch_size: Max samples per predict call
    Returns:
        y_pred: (N,) numpy array in log-transformed space
    """
    X_np = X.numpy()
    if len(X_np) <= batch_size:
        return model.predict(X_np)
    predictions = []
    for i in range(0, len(X_np), batch_size):
        predictions.append(model.predict(X_np[i:i + batch_size]))
    return np.concatenate(predictions)


def tabpfn_predict_with_variance(model, X, batch_size=10_000):
    """Predict with variance from TabPFN's native predictive distribution, batched.

    Args:
        model: Fitted TabPFNRegressor
        X: (N, D) tensor in physical space
        batch_size: Max samples per predict call
    Returns:
        y_pred: (N,) numpy array in log-transformed space
        variance: (N,) numpy array
    """
    X_np = X.numpy()
    all_means, all_vars = [], []
    for i in range(0, len(X_np), batch_size):
        batch = X_np[i:i + batch_size]
        results = model.predict(batch, output_type="full")
        all_means.append(results['mean'])
        var = results['criterion'].variance(results['logits']).detach().cpu().numpy()
        if var.ndim > 1:
            var = var.mean(axis=0)
        all_vars.append(var)
    return np.concatenate(all_means), np.concatenate(all_vars)


def tabpfn_ensemble_predictions(X_train, Y_train, X_candidates, n_samples, device, logger,
                                target="DMRD"):
    """Generate T diverse prediction sets by varying TabPFN's random_state.

    Returns pred_mean, pred_var, and a (T, N, 1) predictions tensor
    suitable for select_entropy_batch_mc.
    """
    y_train_t = transform_y(Y_train, target=target).squeeze().numpy()
    X_cand_np = X_candidates.numpy()
    X_train_np = X_train.numpy()

    predictions = []
    batch_size = 10_000
    logger.info(f"Running {n_samples} TabPFN ensemble forward passes...")
    for t in range(n_samples):
        model_t = TabPFNRegressor(device=device, random_state=t)
        model_t.fit(X_train_np, y_train_t)
        # Batch predictions to avoid OOM on large candidate pools
        preds_chunks = []
        for i in range(0, len(X_cand_np), batch_size):
            preds_chunks.append(model_t.predict(X_cand_np[i:i + batch_size]))
        predictions.append(np.concatenate(preds_chunks))

    predictions = np.stack(predictions)  # (T, N)
    pred_mean = predictions.mean(axis=0)  # (N,)
    pred_var = predictions.var(axis=0)    # (N,)

    logger.info(f"Uncertainty stats: mean={pred_var.mean():.6f}, max={pred_var.max():.6f}")

    # Convert to torch tensors matching select_entropy_batch_mc interface
    pred_mean_t = torch.from_numpy(pred_mean).float().unsqueeze(1)   # (N, 1)
    pred_var_t = torch.from_numpy(pred_var).float().unsqueeze(1)     # (N, 1)
    predictions_t = torch.from_numpy(predictions).float().unsqueeze(2)  # (T, N, 1)

    return pred_mean_t, pred_var_t, predictions_t


def cross_evaluate_tabpfn(model, X_other, Y_other,
                           target='DMRD', return_predictions=False):
    """Evaluate a fitted TabPFN model on an arbitrary dataset.

    Returns (mse_loss, r2) where mse_loss is in transformed space
    and r2 is in physical space.
    If return_predictions=True, returns (mse_loss, r2, Y_true_phys, Y_pred_phys).
    """
    Y_transformed = transform_y(Y_other, target=target).squeeze()
    Y_pred_transformed = tabpfn_predict(model, X_other)
    Y_pred_transformed_t = torch.from_numpy(Y_pred_transformed).float()

    # MSE in transformed space
    mse = ((Y_transformed - Y_pred_transformed_t) ** 2).mean().item()

    # R² in physical space
    Y_true_phys = inverse_transform_y(Y_transformed, target=target)
    Y_pred_phys = inverse_transform_y(Y_pred_transformed_t, target=target)

    ss_res = ((Y_true_phys - Y_pred_phys) ** 2).sum()
    ss_tot = ((Y_true_phys - Y_true_phys.mean()) ** 2).sum()
    r2 = (1 - (ss_res / ss_tot)).item()
    if return_predictions:
        return mse, r2, Y_true_phys.squeeze(), Y_pred_phys.squeeze()
    return mse, r2


def _tabpfn_eval_worker(gpu_id, X_train, Y_train, eval_sets, candidates,
                        target, name, repr_X=None):
    """Fit a TabPFN model on its training set and evaluate on N held-out sets.

    Returns a dict keyed by eval-set name, each holding loss / r2 /
    Y_true / Y_pred. ``train`` is the self-eval on the training set.
    Designed to be called either directly (sequential) or via
    ``concurrent.futures.ThreadPoolExecutor.submit`` for two-GPU concurrency
    on the same node — threads share the parent's CUDA context, so binding
    is by ``cuda:{gpu_id}`` argument with no spawn deadlock.

    eval_sets: list of ``(name, X, Y)`` tuples. Entries with X/Y == None are
    skipped — convenient for optional MCMC / static-random sets.

    candidates: optional tensor of candidate points. If provided, the worker
    also returns native-variance predictions on them, so the main process
    doesn't have to re-fit the model and re-run inference.

    repr_X: optional (k, D) tensor of representative anchor points. If given,
    the worker returns mean+var predictions on them in log-transformed space
    (matching y_transform='log' for the trajectory plotter).
    """
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model, _ = fit_tabpfn(X_train, Y_train, device=device, target=target)
    fit_time = time.time() - t0

    result = {
        '_name': name, '_fit_time': fit_time, '_device': device,
    }
    # Self-eval on the train set, retaining predictions so the caller can
    # compute classification accuracy from this same train-conditioned model
    # without re-fitting (TabPFN inference is non-deterministic across fits).
    tr_loss, tr_r2, tr_yt, tr_yp = cross_evaluate_tabpfn(
        model, X_train, Y_train, target=target, return_predictions=True
    )
    result['train'] = {'loss': tr_loss, 'r2': tr_r2, 'yt': tr_yt, 'yp': tr_yp}

    for set_name, X_eval, Y_eval in eval_sets:
        if X_eval is None or Y_eval is None:
            result[set_name] = None
            continue
        loss, r2, yt, yp = cross_evaluate_tabpfn(
            model, X_eval, Y_eval, target=target, return_predictions=True
        )
        result[set_name] = {'loss': loss, 'r2': r2, 'yt': yt, 'yp': yp}

    if candidates is not None:
        cand_mean, cand_var = tabpfn_predict_with_variance(model, candidates)
        result['candidates'] = {'mean': cand_mean, 'var': cand_var}
    else:
        result['candidates'] = None

    if repr_X is not None:
        repr_mean, repr_var = tabpfn_predict_with_variance(model, repr_X)
        result['repr'] = {'mean': repr_mean, 'var': repr_var}
    else:
        result['repr'] = None

    return result


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
@click.option('--testing/--no-testing', default=False, help="Run in testing mode (small data).")
@click.option('--n-iterations', default=1, type=int, help="Number of active learning iterations.")
@click.option('--n-candidates', default=1000, type=int, help="Candidate pool size.")
@click.option('--n-select', default=10, type=int, help="Number of points to select per iteration.")
@click.option('--n-ensemble-samples', default=16, type=int, help="Number of TabPFN ensemble runs for uncertainty (default: 16).")
@click.option('--n-datasets', default=None, type=int, help="Number of ROOT datasets to load.")
@click.option('--n-samples', default=None, type=int, help="Number of samples to use from data.")
@click.option('--val-fraction', default=0.2, type=float, help="Fraction of data reserved for validation (default: 0.2). Applied to initial data and each batch of new points.")
@click.option('--output-dir', default='active_learning_tabpfn_output', type=str, help="Output directory.")
@click.option('--generate-data/--no-generate-data', default=False, help="Generate new models using Run3ModelGen.")
@click.option('--min-gen-fraction', default=0.6, type=float, help="Minimum fraction of n-select that must be generated successfully before stopping retries (default: 0.6).")
@click.option('--max-gen-attempts', default=10, type=int, help="Maximum number of generation attempts per iteration (default: 10).")
@click.option('--gen-workers', default=1, type=int, help="Number of parallel genModels.py workers per generation attempt (default: 1).")
@click.option('--selection-strategy', default='top_k', type=click.Choice(['top_k', 'top_k_tol_only', 'tol_only_random', 'entropy_batch']), help="Selection strategy: top_k (default), top_k_tol_only (short-circuit, no proximity), or entropy_batch (prohibitively expensive for TabPFN).")
@click.option('--entropy-blur', default=0.15, type=float, help="Entropy smoothing parameter (entropy_batch only).")
@click.option('--entropy-beta', default=50.0, type=float, help="Gibbs sampling temperature (entropy_batch only).")
@click.option('--entropy-pool-size', default=5000, type=int, help="Focused pool size for entropy_batch pre-filtering.")
@click.option('--candidate-generation', default='lhs', type=click.Choice(['uniform', 'lhs']),
              help="Candidate pool generation method: uniform random or Latin Hypercube Sampling (default: lhs).")
@click.option('--proximity-sampling', default=0.1, type=float,
              help="Gaussian proximity weighting width around target value (0 to disable, default: 0.1).")
@click.option('--tolerance-sampling', default=1.0, type=float,
              help="Hard cut: keep only candidates within ±tolerance of threshold in transformed space (0 to disable, default: 1.0).")
@click.option('--target', 'target', default='DMRD',
              type=click.Choice(sorted(TARGET_CONFIG)),
              help="Observable the surrogate learns and the AL loop targets "
                   "(default: DMRD). See active_learning.py for details.")
@click.option('--no-mcmc-eval', is_flag=True, default=False,
              help="Ignore --mcmc-data-dir and run without an MCMC reference "
                   "set, for targets that have no posterior reference.")
@click.option('--target-value', default=None, type=float,
              help="Target relic density value for proximity weighting (default: 0.12).")
@click.option('--config-file', default=None, type=str,
              help="YAML config file (overrides CLI args). Supports parameter sweeps.")
@click.option('--sweep-index', default=None, type=int,
              help="Sweep combination index (requires --config-file).")
@click.option('--mcmc-data-dir', default=None, type=str,
              help="Directory containing MCMC ROOT files for static evaluation (e.g., data/neutralino_v4).")
@click.option('--mcmc-max-samples', default=500_000, type=int,
              help="Seeded uniform subsample cap on the MCMC set (emcee chains "
                   "are ~96% repeated rows; the subsample preserves multiplicity "
                   "weighting). 0 disables.")
@click.option('--static-eval-size', default=100_000, type=int,
              help="Number of models to reserve from the random pool as a static evaluation set (default: 100000).")
@click.option('--data-dir', default='data/18387358', type=str,
              help="Directory containing training ROOT files (default: data/18387358).")
@click.option('--resume-from', default=None, type=str,
              help="Path to previous output dir to resume from (loads state.pt).")
@click.option('--n-additional-iterations', default=None, type=int,
              help="If --resume-from given, run this many more iterations.")
@click.option('--gpu-ids', '--gpu-id', 'gpu_id', default='0', type=str,
              help="Comma-separated GPU IDs. One: sequential AL+baseline on that GPU. "
                   "Two: AL on the first, baseline on the second, in parallel (default: 0). "
                   "--gpu-id is kept as an alias for backward compatibility.")
@click.option('--pack-debris/--keep-debris', 'pack_debris', default=True,
              help='At the end of each iteration, summarise the SModelS winners '
                   'and the LSP composition into the log plus a small JSON, keep '
                   'the ntuples, and pack iteration_NNN/{worker_*,retry_*} into '
                   'one debris.tar. On by default: each iteration otherwise '
                   'leaves ~6,200 files and /ptmp has a hard filesystem inode '
                   'ceiling that does not grow. --keep-debris restores the old '
                   'behaviour of leaving every workspace in place.')
@click.option('--seed', default=42, type=int,
              help="Master random seed propagated to torch / numpy / candidate pool (default: 42).")
def main(testing, n_iterations, n_candidates, n_select, n_ensemble_samples, n_datasets, n_samples, val_fraction, output_dir, generate_data, min_gen_fraction, max_gen_attempts, gen_workers, selection_strategy, entropy_blur, entropy_beta, entropy_pool_size, candidate_generation, proximity_sampling, tolerance_sampling, target, target_value, no_mcmc_eval, config_file, sweep_index, mcmc_data_dir, mcmc_max_samples, static_eval_size, data_dir, resume_from, n_additional_iterations, gpu_id, seed, pack_debris):
    """
    Active learning pipeline for pMSSM relic density prediction using TabPFN.

    Uses TabPFN's native predictive variance for uncertainty and
    ensemble diversity for entropy-based batch selection.
    """
    # Load config file and override parameters if provided
    if config_file is not None:
        cfg = load_config_with_sweep(config_file, sweep_index)

        _cfg_map = {
            'n_iterations': 'n_iterations',
            'n_candidates': 'n_candidates',
            'n_select': 'n_select',
            'n_ensemble_samples': 'n_ensemble_samples',
            'selection_strategy': 'selection_strategy',
            'entropy_blur': 'entropy_blur',
            'entropy_beta': 'entropy_beta',
            'entropy_pool_size': 'entropy_pool_size',
            'candidate_generation': 'candidate_generation',
            'proximity_sampling': 'proximity_sampling',
            'target': 'target',
            'target_value': 'target_value',
        }

        for cfg_key, local_key in _cfg_map.items():
            if cfg_key in cfg:
                val = cfg[cfg_key]
                if cfg_key in ('n_iterations', 'n_candidates', 'n_select', 'n_ensemble_samples', 'entropy_pool_size'):
                    val = int(val)
                elif cfg_key in ('entropy_blur', 'entropy_beta', 'proximity_sampling', 'target_value'):
                    val = float(val)
                locals()[local_key] = val

        n_iterations = locals().get('n_iterations', n_iterations)
        n_candidates = locals().get('n_candidates', n_candidates)
        n_select = locals().get('n_select', n_select)
        n_ensemble_samples = locals().get('n_ensemble_samples', n_ensemble_samples)
        selection_strategy = locals().get('selection_strategy', selection_strategy)
        entropy_blur = locals().get('entropy_blur', entropy_blur)
        entropy_beta = locals().get('entropy_beta', entropy_beta)
        entropy_pool_size = locals().get('entropy_pool_size', entropy_pool_size)
        candidate_generation = locals().get('candidate_generation', candidate_generation)
        proximity_sampling = locals().get('proximity_sampling', proximity_sampling)
        target = locals().get('target', target)
        target_value = locals().get('target_value', target_value)

    # ---- Resolve the target's physical value ---------------------------------
    # After the config-file block, before the banner and any resume check.
    # TabPFN is log-transform-only, so there is no zscore branch to guard.
    if target_value is None:
        target_value = float(TARGET_CONFIG[target]["true_value"])

    if no_mcmc_eval and mcmc_data_dir is not None:
        mcmc_data_dir = None

    # Propagate master seed to torch / numpy / python-random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Increase n_candidates if needed:
    if n_candidates < n_select: n_candidates = n_select

    output_dir = Path(output_dir)
    # Collision-free dir suffix. TabPFN has no warm-start, use "tabpfn" sentinel
    # so the manifest parser can still split on the 4-token pattern.
    auto_suffix = f"_{selection_strategy}_tabpfn_seed{seed}_{timestamp}"
    if not re.search(r"_\d{8}_\d{6}$", output_dir.name):
        output_dir = output_dir.with_name(output_dir.name + auto_suffix)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up main logging to output_dir/active_learning.log
    log_file, logger = setup_logging(timestamp, output_dir=output_dir)

    logger.info("=" * 60)
    logger.info("Active Learning Pipeline for pMSSM (TabPFN)")
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
        n_candidates = 100
        n_ensemble_samples = 4
        logger.info("Testing mode enabled")
    else:
        n_datasets = n_datasets if n_datasets is not None else -1
        n_samples = n_samples if n_samples is not None else None

    logger.info(f"Configuration:")
    logger.info(f"  model: TabPFN")
    logger.info(f"  n_iterations: {n_iterations}")
    logger.info(f"  n_candidates: {n_candidates}")
    logger.info(f"  n_select: {n_select}")
    logger.info(f"  n_ensemble_samples: {n_ensemble_samples}")
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
    logger.info(f"  target: {target}")
    logger.info(f"  target_value: {target_value}")
    logger.info(f"  generate_data: {generate_data}")
    if generate_data:
        logger.info(f"  min_gen_fraction: {min_gen_fraction} (target: {int(n_select * min_gen_fraction)} valid models per iteration)")
        logger.info(f"  max_gen_attempts: {max_gen_attempts}")
        logger.info(f"  gen_workers: {gen_workers}")

    gpu_id_list = [int(s.strip()) for s in gpu_id.split(',') if s.strip()]
    if not gpu_id_list:
        gpu_id_list = [0]
    AL_GPU_ID = gpu_id_list[0]
    BASELINE_GPU_ID = gpu_id_list[1] if len(gpu_id_list) > 1 else gpu_id_list[0]
    device = f"cuda:{AL_GPU_ID}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(AL_GPU_ID)}")

    # Parallel fit+eval via ThreadPoolExecutor when 2+ distinct GPUs are
    # visible; otherwise sequential on the single device. Threads (rather
    # than mp.Process) avoid the ROCm/MI300A spawn-after-CUDA-init deadlock
    # — the two threads share the parent's CUDA context, with binding
    # selected per call by the worker's ``cuda:{gpu_id}`` argument.
    use_parallel = (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= 2
        and AL_GPU_ID != BASELINE_GPU_ID
    )
    if use_parallel:
        logger.info(f"Parallel TabPFN enabled (threads): AL on cuda:{AL_GPU_ID}, Baseline on cuda:{BASELINE_GPU_ID}")
    else:
        logger.info(f"Sequential TabPFN: both models on {device}")

    # Load initial data
    logger.info("Loading data...")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    X, Y, F = load_pmssm_data(n_datasets=n_datasets, logger=logger,
                              plot_dir=str(plots_dir), data_dir=data_dir,
                              target=target, return_lsp_fracs=True)

    # Shuffle once up-front: loaders concatenate ROOT files (= MCMC chains)
    # in file order, so X[:n_samples] would otherwise draw from a single chain.
    _load_perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))
    X = X[_load_perm]
    Y = Y[_load_perm]
    F = F[_load_perm]
    logger.info(f"Shuffled loaded dataset ({len(X)} samples, seed={seed})")

    # Load MCMC evaluation dataset if provided
    X_mcmc, Y_mcmc, F_mcmc = None, None, None
    if mcmc_data_dir is not None:
        X_mcmc, Y_mcmc, F_mcmc = load_mcmc_data(data_dir=mcmc_data_dir, logger=logger,
                                                target=target,
                                                return_lsp_fracs=True,
                                                max_samples=mcmc_max_samples or None)
        logger.info(f"MCMC evaluation dataset: {len(X_mcmc)} samples from {mcmc_data_dir}")

    # Store full dataset for baseline random sampling (before any truncation)
    X_full, Y_full, F_full = X.clone(), Y.clone(), F.clone()

    # ------------------------------------------------------------------
    # Representative-points trajectory tracker (seeded, deterministic).
    # Picks 1 point per LSP class nearest the target Ωh², plus the Ωh²-median
    # row, from the MCMC eval pool (fallback: X_full). Symmetric to the
    # transformer + GP drivers' trackers; predictions are in log-transformed
    # space (TabPFN's training space).
    # ------------------------------------------------------------------
    if X_mcmc is not None:
        _repr_pool_X, _repr_pool_Y, _repr_pool_F = X_mcmc, Y_mcmc, F_mcmc
        _repr_pool_source = "MCMC eval set"
    else:
        _repr_pool_X, _repr_pool_Y, _repr_pool_F = X_full, Y_full, F_full
        _repr_pool_source = "X_full"
    _target_value = target_value
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

    logger.info(f"TabPFN model — no training loop, fit is in-context learning")

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
        # A resume that changes the target would append labels of one observable
        # to a dataset of another. state.pt written before --target was persisted
        # has no key, so treat that as the historical DMRD default.
        _saved_target = saved.get("target", "DMRD")
        if _saved_target != target:
            raise click.UsageError(
                f"Refusing to resume: this run was trained on target "
                f"{_saved_target!r} but --target is {target!r}. Pass "
                f"--target {_saved_target} to continue it."
            )
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
        restore_rng(saved["rng"])
        start_iteration = saved["iteration"] + 1
        n_iterations = saved["iteration"] + n_additional_iterations
        logger.info(f"Resuming at iteration {start_iteration}, will run through {n_iterations}")

    for iteration in range(start_iteration, n_iterations + 1):
        logger.info(f"=== Global Iteration {iteration} ===")

        # Reset any per-iteration lazy state (see retry block for details).
        al_model_retry = None

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
            F_add = F_full[baseline_add_indices]

            X_baseline_train = torch.cat([X[:n_train_init], X_add[:n_add_train]])
            Y_baseline_train = torch.cat([Y[:n_train_init], Y_add[:n_add_train]])
            F_baseline_train = torch.cat([F[:n_train_init], F_add[:n_add_train]])
            X_baseline_val = torch.cat([X_val[:n_val_init], X_add[n_add_train:]])
            Y_baseline_val = torch.cat([Y_val[:n_val_init], Y_add[n_add_train:]])
            F_baseline_val = torch.cat([F_val[:n_val_init], F_add[n_add_train:]])
            logger.info(f"Baseline dataset: {n_train_init}+{n_add_train}={len(X_baseline_train)} train, "
                        f"{n_val_init}+{n_add_val}={len(X_baseline_val)} val")

        logger.info(f"AL: n_train={len(X)}, n_val={len(X_val)}")
        logger.info(f"Baseline: n_train={len(X_baseline_train)}, n_val={len(X_baseline_val)}")

        # Plot data distribution histograms and parallel coordinates for AL
        X_combined = torch.cat([X, X_val], dim=0)
        Y_combined = torch.cat([Y, Y_val], dim=0)
        idx_train_al = torch.arange(len(X))
        idx_val_al = torch.arange(len(X), len(X_combined))

        X_baseline_combined = torch.cat([X_baseline_train, X_baseline_val], dim=0)
        Y_baseline_combined = torch.cat([Y_baseline_train, Y_baseline_val], dim=0)
        idx_train_base = torch.arange(len(X_baseline_train))
        idx_val_base = torch.arange(len(X_baseline_train), len(X_baseline_combined))

        al_hist_dir = iter_plots_dir / "al"
        al_hist_dir.mkdir(parents=True, exist_ok=True)
        plot_data_histograms(X_combined, Y_combined, idx_train_al, idx_val_al, al_hist_dir, "AL", iteration, logger,
                             reference_X=X_mcmc, reference_Y=Y_mcmc, reference_label="MCMC")
        plot_parallel_coordinates(X_combined, idx_train_al, idx_val_al, al_hist_dir, "AL", iteration, logger)

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

        # Generate candidate pool BEFORE the parallel fit+eval phase, so each
        # worker thread can also predict (mean, var) on the candidates in the
        # same call. This avoids re-fitting either model in the main thread
        # just to run inference for selection and uncertainty plots.
        logger.info(f"Generating {n_candidates} candidate points using {candidate_generation} sampling...")
        candidates = generate_candidate_pool(n_candidates, method=candidate_generation, seed=seed * 10_000 + iteration)

        if proximity_sampling > 0 or tolerance_sampling > 0:
            threshold_transformed = transform_y(torch.tensor([target_value]), target=target).item()
        else:
            threshold_transformed = 0.0

        # Only request candidate variance predictions from the AL worker when
        # the selection strategy needs them. entropy_batch generates its own
        # ensemble predictions from scratch, so skipping here saves one
        # inference pass on the candidate pool.
        al_needs_cand = selection_strategy in ('top_k', 'top_k_tol_only')
        al_candidates = candidates if al_needs_cand else None

        # Fit + evaluate AL and Baseline TabPFN models. In parallel on 2 GPUs
        # when available; otherwise sequentially on the same GPU. Each worker
        # returns: train (self-eval) + own_val + cross_val + optional mcmc +
        # optional static + candidate predictions — the same set the sequential
        # path computes, plus the candidate inference.
        al_eval_sets = [
            ('own_val',   X_val,            Y_val),
            ('cross_val', X_baseline_val,   Y_baseline_val),
            ('mcmc',      X_mcmc,           Y_mcmc),
            ('static',    X_static_random,  Y_static_random),
        ]
        base_eval_sets = [
            ('own_val',   X_baseline_val,   Y_baseline_val),
            ('cross_val', X_val,            Y_val),
            ('mcmc',      X_mcmc,           Y_mcmc),
            ('static',    X_static_random,  Y_static_random),
        ]

        t_fit0 = time.time()
        if use_parallel:
            logger.info(f"Fitting+evaluating TabPFN AL on cuda:{AL_GPU_ID} and Baseline on cuda:{BASELINE_GPU_ID} in parallel threads...")
            with ThreadPoolExecutor(max_workers=2) as ex:
                al_fut = ex.submit(
                    _tabpfn_eval_worker,
                    AL_GPU_ID, X, Y, al_eval_sets, al_candidates, target, 'AL',
                    repr_X=repr_points['X'],
                )
                base_fut = ex.submit(
                    _tabpfn_eval_worker,
                    BASELINE_GPU_ID, X_baseline_train, Y_baseline_train, base_eval_sets,
                    candidates, target, 'Baseline',
                )
                al_res = al_fut.result()
                base_res = base_fut.result()
        else:
            logger.info(f"Fitting+evaluating TabPFN AL then Baseline sequentially on {device}...")
            al_res = _tabpfn_eval_worker(
                AL_GPU_ID, X, Y, al_eval_sets, al_candidates, target, 'AL',
                repr_X=repr_points['X'],
            )
            base_res = _tabpfn_eval_worker(
                BASELINE_GPU_ID, X_baseline_train, Y_baseline_train, base_eval_sets,
                candidates, target, 'Baseline',
            )

        # Representative-points capture from this iteration's AL fit.
        if al_res.get('repr') is not None:
            repr_log.append({
                'iteration': int(iteration),
                'mean': np.asarray(al_res['repr']['mean']).reshape(-1).tolist(),
                'var':  np.asarray(al_res['repr']['var']).reshape(-1).tolist(),
            })
        logger.info(f"TabPFN fit+eval wall-clock: {time.time() - t_fit0:.1f}s  "
                   f"(AL fit {al_res['_fit_time']:.1f}s, Baseline fit {base_res['_fit_time']:.1f}s)")

        # Unpack worker results into the flat names the rest of the loop expects.
        al_train_loss        = al_res['train']['loss']
        al_train_r2_val      = al_res['train']['r2']
        al_val_loss_val      = al_res['own_val']['loss']
        al_r2_val            = al_res['own_val']['r2']
        al_own_yt            = al_res['own_val']['yt']
        al_own_yp            = al_res['own_val']['yp']
        al_cross_loss        = al_res['cross_val']['loss']
        al_cross_r2          = al_res['cross_val']['r2']
        al_cross_yt          = al_res['cross_val']['yt']
        al_cross_yp          = al_res['cross_val']['yp']

        base_train_loss      = base_res['train']['loss']
        base_train_r2_val    = base_res['train']['r2']
        base_val_loss_val    = base_res['own_val']['loss']
        base_r2_val          = base_res['own_val']['r2']
        base_own_yt          = base_res['own_val']['yt']
        base_own_yp          = base_res['own_val']['yp']
        base_cross_loss      = base_res['cross_val']['loss']
        base_cross_r2        = base_res['cross_val']['r2']
        base_cross_yt        = base_res['cross_val']['yt']
        base_cross_yp        = base_res['cross_val']['yp']

        logger.info(f"AL metrics: train_loss={al_train_loss:.6f}, val_loss={al_val_loss_val:.6f}, R²={al_r2_val:.4f}, train_R²={al_train_r2_val:.4f}")
        logger.info(f"Baseline metrics: train_loss={base_train_loss:.6f}, val_loss={base_val_loss_val:.6f}, R²={base_r2_val:.4f}, train_R²={base_train_r2_val:.4f}")

        # Track metrics
        iteration_numbers.append(iteration)
        al_train_losses.append(al_train_loss)
        al_val_losses.append(al_val_loss_val)
        al_r2_scores.append(al_r2_val)
        al_train_r2_scores.append(al_train_r2_val)
        al_n_train.append(len(X))
        al_n_val.append(len(X_val))
        baseline_train_losses.append(base_train_loss)
        baseline_val_losses.append(base_val_loss_val)
        baseline_r2_scores.append(base_r2_val)
        baseline_train_r2_scores.append(base_train_r2_val)
        baseline_n_train.append(len(X_baseline_train))
        baseline_n_val.append(len(X_baseline_val))

        logger.info(f"Cross-eval: AL_on_base_val_loss={al_cross_loss:.6f}, AL_on_base_val_R²={al_cross_r2:.4f}, base_on_al_val_loss={base_cross_loss:.6f}, base_on_al_val_R²={base_cross_r2:.4f}")
        al_on_base_val_losses.append(al_cross_loss)
        al_on_base_val_r2.append(al_cross_r2)
        base_on_al_val_losses.append(base_cross_loss)
        base_on_al_val_r2.append(base_cross_r2)

        scatter_results = [
            dict(model_name="AL", dataset_name="AL Val", y_true=al_own_yt, y_pred=al_own_yp,
                 loss=al_val_loss_val, r2=al_r2_val, n=len(X_val), lsp_fracs=F_val),
            dict(model_name="AL", dataset_name="Base Val", y_true=al_cross_yt, y_pred=al_cross_yp,
                 loss=al_cross_loss, r2=al_cross_r2, n=len(X_baseline_val), lsp_fracs=F_baseline_val),
            dict(model_name="Baseline", dataset_name="AL Val", y_true=base_cross_yt, y_pred=base_cross_yp,
                 loss=base_cross_loss, r2=base_cross_r2, n=len(X_val), lsp_fracs=F_val),
            dict(model_name="Baseline", dataset_name="Base Val", y_true=base_own_yt, y_pred=base_own_yp,
                 loss=base_val_loss_val, r2=base_r2_val, n=len(X_baseline_val), lsp_fracs=F_baseline_val),
        ]

        # MCMC eval (optional — present only if mcmc_data_dir was provided)
        if al_res.get('mcmc') is not None:
            mcmc_loss_al  = al_res['mcmc']['loss']
            mcmc_r2_al    = al_res['mcmc']['r2']
            mcmc_yt_al    = al_res['mcmc']['yt']
            mcmc_yp_al    = al_res['mcmc']['yp']
            mcmc_loss_base = base_res['mcmc']['loss']
            mcmc_r2_base   = base_res['mcmc']['r2']
            mcmc_yt_base   = base_res['mcmc']['yt']
            mcmc_yp_base   = base_res['mcmc']['yp']
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

        # Static-random eval (optional)
        if al_res.get('static') is not None:
            static_loss_al  = al_res['static']['loss']
            static_r2_al    = al_res['static']['r2']
            static_yt_al    = al_res['static']['yt']
            static_yp_al    = al_res['static']['yp']
            static_loss_base = base_res['static']['loss']
            static_r2_base   = base_res['static']['r2']
            static_yt_base   = base_res['static']['yt']
            static_yp_base   = base_res['static']['yp']
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
        # All predictions used here come from the SAME TabPFN model fitted on
        # the iteration's training set (the model produced inside
        # _tabpfn_eval_worker, then discarded). Re-fitting would change
        # predictions because TabPFN inference is non-deterministic across
        # fits. We piggyback on the worker's already-computed (yt, yp) tuples.
        # Failures are non-fatal — accuracy is diagnostic.
        try:
            _acc_thr = float(target_value)
            al_accs = {
                "val":   binary_accuracy(al_own_yt, al_own_yp, _acc_thr),
                "train": binary_accuracy(
                    al_res['train']['yt'], al_res['train']['yp'], _acc_thr),
            }
            base_accs = {
                "val":   binary_accuracy(base_own_yt, base_own_yp, _acc_thr),
                "train": binary_accuracy(
                    base_res['train']['yt'], base_res['train']['yp'], _acc_thr),
            }
            if al_res.get('mcmc') is not None:
                al_accs["mcmc"] = binary_accuracy(mcmc_yt_al, mcmc_yp_al, _acc_thr)
                base_accs["mcmc"] = binary_accuracy(mcmc_yt_base, mcmc_yp_base, _acc_thr)
            if al_res.get('static') is not None:
                al_accs["static_random"] = binary_accuracy(static_yt_al, static_yp_al, _acc_thr)
                base_accs["static_random"] = binary_accuracy(static_yt_base, static_yp_base, _acc_thr)
            write_iter_accuracies(output_dir, iteration, al_accs=al_accs,
                                  baseline_accs=base_accs)
            _acc_summary = "  ".join(
                f"{role}=" + "/".join(f"{k}={v:.4f}" for k, v in (d or {}).items())
                for role, d in (("AL", al_accs), ("Base", base_accs)) if d
            )
            logger.info(f"Accuracy@{_acc_thr}: {_acc_summary}")
        except Exception as _acc_exc:
            logger.warning(f"Accuracy capture failed (non-fatal): {_acc_exc}")

        # Selection. entropy_batch still runs its own ensemble (independent of
        # the fitted models). top_k / top_k_tol_only reuse the AL worker's
        # candidate predictions — no re-fit, no extra inference in main.
        if selection_strategy == 'entropy_batch':
            pred_mean, pred_var, predictions = tabpfn_ensemble_predictions(
                X, Y, candidates, n_ensemble_samples, device, logger,
                target=target
            )
            top_indices = select_entropy_batch_mc(
                candidates, predictions, pred_mean, pred_var,
                n_select, blur=entropy_blur, beta=entropy_beta,
                n_pool=entropy_pool_size,
                threshold=threshold_transformed, tolerance_sampling=tolerance_sampling,
                proximity_sampling=proximity_sampling,
                device=device, logger=logger
            )
        elif selection_strategy == 'tol_only_random':
            # Mean-guided arm: tolerance cut then a uniform draw. Cheap for
            # TabPFN, since only the in-context mean is needed and the ensemble
            # variance is never consulted (see pmssm.selection).
            pred_mean = torch.from_numpy(al_res['candidates']['mean']).float().unsqueeze(1)
            pred_var = torch.from_numpy(al_res['candidates']['var']).float().unsqueeze(1)
            top_indices = select_tol_only_random(
                candidates, pred_mean, n_select,
                threshold=threshold_transformed,
                tolerance_sampling=tolerance_sampling,
                seed=seed * 100_000 + iteration,
                logger=logger,
            )
        elif selection_strategy == 'top_k_tol_only':
            pred_mean = torch.from_numpy(al_res['candidates']['mean']).float().unsqueeze(1)
            pred_var = torch.from_numpy(al_res['candidates']['var']).float().unsqueeze(1)
            logger.info(f"Uncertainty stats: mean={pred_var.mean():.6f}, max={pred_var.max():.6f}")
            top_indices = select_top_uncertain_tol_only(
                candidates, pred_mean, pred_var, n_select,
                threshold=threshold_transformed,
                tolerance_sampling=tolerance_sampling,
                logger=logger,
            )
        else:
            pred_mean = torch.from_numpy(al_res['candidates']['mean']).float().unsqueeze(1)
            pred_var = torch.from_numpy(al_res['candidates']['var']).float().unsqueeze(1)
            logger.info(f"Uncertainty stats: mean={pred_var.mean():.6f}, max={pred_var.max():.6f}")

            top_indices = select_top_uncertain_filtered(
                candidates, pred_mean, pred_var, n_select,
                threshold=threshold_transformed,
                tolerance_sampling=tolerance_sampling,
                proximity_sampling=proximity_sampling,
                logger=logger,
            )

        logger.info(f"Selected {len(top_indices)} points via {selection_strategy} (requested: {n_select}, available: {len(candidates)})")

        # ---- Candidate uncertainty plots (AL + Baseline) ----
        # Baseline variance comes from the baseline worker's candidate pass.
        # AL variance comes from `pred_var` above (ensemble for entropy_batch,
        # native for top_k).
        try:
            _plot_pool_size = min(len(candidates), 20_000)
            _plot_idx = torch.randperm(len(candidates))[:_plot_pool_size]
            _plot_cands = candidates[_plot_idx]
            _plot_al_var = pred_var[_plot_idx]
            plot_candidate_uncertainty(_plot_cands, _plot_al_var, al_hist_dir,
                                       "AL", iteration, logger)
            _base_pred_var = torch.from_numpy(base_res['candidates']['var']).float().unsqueeze(1)
            _base_pred_var = _base_pred_var[_plot_idx]
            plot_candidate_uncertainty(_plot_cands, _base_pred_var, baseline_hist_dir,
                                       "Baseline", iteration, logger)
        except Exception as e:
            logger.warning(f"Candidate uncertainty plot failed: {e}")

        csv_path = save_selected_points(candidates, pred_var, top_indices, output_dir, iteration)
        logger.info(f"Saved selected points to {csv_path}")

        all_selected_points.append({
            "iteration": iteration,
            "points": candidates[top_indices].numpy().tolist(),
            "uncertainties": pred_var[top_indices].squeeze().numpy().tolist(),
            "al_best_val_loss": al_val_loss_val,
            "al_r2_score": al_r2_val,
            "baseline_best_val_loss": base_val_loss_val,
            "baseline_r2_score": base_r2_val,
        })

        # Generate new models if requested, with retry logic
        new_X, new_Y, new_F = None, None, None
        if generate_data:
            n_target = max(1, int(n_select * min_gen_fraction))
            logger.info(f"Generation target: {n_target} valid models ({min_gen_fraction*100:.0f}% of {n_select} selected, max {max_gen_attempts} attempts)")

            collected_X, collected_Y, collected_F = [], [], []

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

                    attempt_seed = seed * 10_000 + iteration * 1000 + attempt
                    attempt_candidates = generate_candidate_pool(n_candidates, method=candidate_generation, seed=attempt_seed)

                    if selection_strategy == 'entropy_batch':
                        attempt_mean, attempt_pred_var, attempt_preds = tabpfn_ensemble_predictions(
                            X, Y, attempt_candidates, n_ensemble_samples, device, logger,
                            target=target
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
                        # Retry path for top_k variants: the per-iteration AL
                        # worker (thread or sequential call) only predicted on
                        # the original candidate pool, so we need a local AL
                        # model to evaluate the retry pool. Lazy-fit once per
                        # iteration (al_model_retry is reset to None at the
                        # top of the iteration loop).
                        if al_model_retry is None:
                            al_model_retry, _ = fit_tabpfn(X, Y, device=device)
                        attempt_y_pred, attempt_var = tabpfn_predict_with_variance(al_model_retry, attempt_candidates)
                        attempt_mean = torch.from_numpy(attempt_y_pred).float().unsqueeze(1)
                        attempt_pred_var = torch.from_numpy(attempt_var).float().unsqueeze(1)
                        if selection_strategy == 'tol_only_random':
                            attempt_indices = select_tol_only_random(
                                attempt_candidates, attempt_mean, n_select,
                                threshold=threshold_transformed,
                                tolerance_sampling=tolerance_sampling,
                                seed=attempt_seed,
                                logger=logger,
                            )
                        elif selection_strategy == 'top_k_tol_only':
                            attempt_indices = select_top_uncertain_tol_only(
                                attempt_candidates, attempt_mean, attempt_pred_var, n_select,
                                threshold=threshold_transformed,
                                tolerance_sampling=tolerance_sampling,
                                logger=logger,
                            )
                        else:
                            attempt_indices = select_top_uncertain_filtered(
                                attempt_candidates, attempt_mean, attempt_pred_var, n_select,
                                threshold=threshold_transformed,
                                tolerance_sampling=tolerance_sampling,
                                proximity_sampling=proximity_sampling,
                                logger=logger,
                            )

                    param_names = [p.replace("IN_", "") for p in PARAM_ORDER]
                    df = pd.DataFrame(attempt_candidates[attempt_indices].numpy(), columns=param_names)
                    if selection_strategy != 'entropy_batch':
                        df["uncertainty"] = attempt_pred_var[attempt_indices].squeeze().numpy()
                    attempt_csv = attempt_dir / "selected_points.csv"
                    df.to_csv(attempt_csv, index=False)

                logger.info(f"Generation attempt {attempt + 1}/{max_gen_attempts} ({len(attempt_indices)} points)...")
                ntuple_paths = generate_models_from_csv(attempt_csv, attempt_dir, logger, n_workers=gen_workers, target=target)

                for ntuple_path in ntuple_paths:
                    batch_X, batch_Y, batch_F = load_generated_data(
                        ntuple_path, logger, return_lsp_fracs=True, target=target)
                    if batch_X is not None and len(batch_X) > 0:
                        collected_X.append(batch_X)
                        collected_Y.append(batch_Y)
                        collected_F.append(batch_F)

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
                new_F = torch.cat(collected_F)
                # Deduplicate: identical X rows from SPheno rounding can leak across train/val
                _, unique_idx = np.unique(new_X.numpy(), axis=0, return_index=True)
                if len(unique_idx) < len(new_X):
                    logger.info(f"Removing {len(new_X) - len(unique_idx)} duplicate generated points")
                    unique_idx = torch.from_numpy(np.sort(unique_idx))
                    new_X = new_X[unique_idx]
                    new_Y = new_Y[unique_idx]
                    new_F = new_F[unique_idx]
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
                            f"(train: {len(X)}->{len(X)+n_new_train}, val: {len(X_val)}->{len(X_val)+n_new_val})")
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

        # No checkpoints to update — TabPFN is re-fitted each iteration

        # ---- Checkpoint run state for resume -------------------------------
        from pmssm.resume import save_state, capture_rng
        save_state(output_dir, {
            "target": target,
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
            "rng": capture_rng(),
        })

        # ---- End-of-iteration housekeeping ---------------------------------
        # Runs AFTER save_state, so the iteration's training data is already
        # durable before the simulator workspaces are touched. It summarises the
        # SModelS winners and the LSP composition into the log and a small JSON,
        # keeps the ntuples, and packs iteration_NNN/{worker_*,retry_*} into one
        # debris.tar. Each iteration otherwise leaves ~6,200 files behind, and
        # /ptmp has a hard filesystem inode ceiling that does not grow. Never
        # raises: on failure the workspaces simply stay.
        from pmssm.iteration_housekeeping import finalise_iteration
        finalise_iteration(iter_dir, logger,
                           target_branch=TARGET_CONFIG[target]["branch"],
                           true_value=TARGET_CONFIG[target]["true_value"],
                           enabled=pack_debris)
        logger.info(f"[resume] state.pt saved (iteration {iteration})")

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

    # ---- Representative-points trajectory plot (+ CSV) ----
    if repr_log:
        try:
            _rep_out, _rep_csv = plot_representative_trajectories(
                repr_log, repr_points['Y'], repr_points['cls'], repr_points['labels'],
                _target_value, output_dir, y_transform='log', target=target,
            )
            logger.info(f"Representative-points trajectory: {_rep_out}")
            logger.info(f"Representative-points CSV: {_rep_csv}")
        except Exception as _e:
            logger.warning(f"Representative-points plot failed: {_e}")

    # Save summary
    summary = {
        "timestamp": timestamp,
        "config": {
            "target": target,
            "target_value": target_value,
            "model": "TabPFN",
            "n_iterations": n_iterations,
            "n_candidates": n_candidates,
            "n_select": n_select,
            "n_ensemble_samples": n_ensemble_samples,
            "generate_data": generate_data,
            "selection_strategy": selection_strategy,
            "seed": seed,
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
    logger.info("TabPFN Active Learning Complete")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
