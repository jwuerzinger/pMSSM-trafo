"""
Candidate generation and selection strategies for active learning.

This module provides unified functions for:
- Generating candidate pools (uniform random or LHS)
- Selecting points based on uncertainty (top-k, entropy-based)
- Proximity weighting to focus on target regions
"""

import numpy as np
import torch
from scipy.stats import qmc

from .config import PARAM_ORDER, PARAM_RANGES


# ===== Candidate Generation =====

def generate_candidate_pool(n_candidates, param_order=PARAM_ORDER,
                            param_ranges=PARAM_RANGES, method='lhs', seed=None):
    """
    Generate candidate points in pMSSM parameter space.

    Args:
        n_candidates: Number of candidates to generate
        param_order: List of parameter names
        param_ranges: Dict mapping parameter names to (min, max)
        method: 'uniform' for random sampling or 'lhs' for Latin Hypercube
        seed: Random seed for reproducibility

    Returns:
        candidates: Tensor (n_candidates, n_params) in physical units
    """
    if seed is not None:
        np.random.seed(seed)

    n_params = len(param_order)

    if method == 'lhs':
        # Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=n_params, seed=seed)
        candidates_norm = sampler.random(n=n_candidates)  # Values in [0, 1]

        # Scale to parameter ranges
        candidates = np.zeros((n_candidates, n_params))
        for i, param in enumerate(param_order):
            low, high = param_ranges[param]
            if low == high:  # Fixed parameter
                candidates[:, i] = low
            else:
                candidates[:, i] = candidates_norm[:, i] * (high - low) + low

    elif method == 'uniform':
        # Uniform random sampling
        candidates = np.zeros((n_candidates, n_params))
        for i, param in enumerate(param_order):
            low, high = param_ranges[param]
            if low == high:  # Fixed parameter
                candidates[:, i] = low
            else:
                candidates[:, i] = np.random.uniform(low, high, n_candidates)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'uniform' or 'lhs'.")

    return torch.from_numpy(candidates).float()


# ===== Uncertainty-Based Selection =====

def select_top_uncertain(X_candidates, uncertainties, n_select):
    """
    Select points with highest uncertainty (variance).

    Args:
        X_candidates: Candidate pool tensor (N, D)
        uncertainties: Uncertainty values (N, 1) or (N,)
        n_select: Number of points to select

    Returns:
        top_indices: Indices of selected points (n_select,)
    """
    uncertainties_flat = uncertainties.squeeze().numpy()
    all_sorted = np.argsort(uncertainties_flat)[::-1].copy()
    return all_sorted[:n_select].copy()


# ===== Entropy-Based Selection with Proximity Weighting =====

def select_entropy_batch_mc(X_candidates, predictions, pred_mean, pred_var,
                            n_select, blur=0.15, beta=50.0, n_pool=5000,
                            threshold=0.0, proximity_sampling=0.0,
                            device='cpu', logger=None):
    """
    Entropy-based batch selection using MC Dropout sample covariance.

    Pre-filters candidates by variance to a focused pool, computes sample
    covariance from MC Dropout predictions, then uses EntropySelectionStrategy
    for iterative batch selection with diversity.

    NEW: Optional proximity weighting to focus on regions near threshold.

    Args:
        X_candidates: (N, D) candidate points
        predictions: (T, N, 1) MC Dropout predictions tensor
        pred_mean: (N, 1) mean predictions in transformed space
        pred_var: (N, 1) prediction variance
        n_select: Number of points to select
        blur: Entropy smoothing parameter
        beta: Gibbs sampling temperature (high=deterministic, low=random)
        n_pool: Focused pool size (pre-filtered by variance)
        threshold: Decision threshold in transformed space (default: 0.0)
        proximity_sampling: Gaussian proximity weighting width (0 to disable)
        device: Torch device
        logger: Logger instance

    Returns:
        selected_indices: Indices into X_candidates of selected points
    """
    # Import here to avoid circular dependency
    import sys
    from pathlib import Path
    _GP_PIPELINE_ROOT = Path(__file__).parent.parent / "al_pmssmwithgp" / "model"
    if str(_GP_PIPELINE_ROOT) not in sys.path:
        sys.path.insert(0, str(_GP_PIPELINE_ROOT))
    from gp_pipeline.utils.selection import EntropySelectionStrategy

    N = X_candidates.shape[0]
    n_pool = min(n_pool, N)
    n_select = min(n_select, n_pool)

    # Apply proximity weighting if enabled
    if proximity_sampling > 0.0:
        # Gaussian weighting: exp(-((pred_mean - threshold)^2) / proximity_sampling)
        # Favors points near the threshold (e.g., 0.12 in physical space for DMRD)
        proximity = torch.exp(-((pred_mean.squeeze() - threshold) ** 2) / proximity_sampling)
        weighted_var = proximity.unsqueeze(1) * pred_var

        if logger:
            logger.info(f"Proximity weighting enabled: "
                       f"focusing on predictions near threshold={threshold:.3f}")
            logger.info(f"Proximity weights: mean={proximity.mean():.4f}, "
                       f"max={proximity.max():.4f}, min={proximity.min():.4f}")
    else:
        weighted_var = pred_var

    # Pre-filter to focused pool: top n_pool candidates by weighted variance
    var_flat = weighted_var.squeeze()
    pool_indices = torch.argsort(var_flat, descending=True)[:n_pool]

    if logger:
        logger.info(f"Entropy batch: focused pool of {n_pool} candidates (from {N} total)")

    # Extract predictions for the focused pool: (T, n_pool, 1)
    pool_preds = predictions[:, pool_indices, :]  # (T, n_pool, 1)
    pool_mean = pred_mean[pool_indices].squeeze()  # (n_pool,)

    # Compute sample covariance from MC Dropout predictions
    preds_2d = pool_preds.squeeze(-1)  # (T, n_pool)
    mean_2d = preds_2d.mean(dim=0)     # (n_pool,)
    centered = preds_2d - mean_2d      # (T, n_pool)
    T = preds_2d.shape[0]
    sample_cov = (centered.T @ centered) / (T - 1)  # (n_pool, n_pool)

    # Regularize: sample covariance is rank-deficient when T < n_pool
    sample_cov += 1e-4 * torch.eye(n_pool)

    if logger:
        logger.info(f"Sample covariance: shape={sample_cov.shape}, rank≤{T}, "
                   f"diag range=[{sample_cov.diag().min():.6f}, {sample_cov.diag().max():.6f}]")

    # Move to device for entropy computation
    pool_mean_dev = pool_mean.to(device)
    sample_cov_dev = sample_cov.to(device)

    # Run entropy-based iterative batch selection
    strategy = EntropySelectionStrategy(blur=blur, beta=beta)
    score_function = strategy.smoothed_batch_entropy(blur=blur, device=device)
    choice_function = lambda score, indices: strategy.gibbs_sample(score, beta, device)

    if logger:
        logger.info(f"Running iterative batch selector for {n_select} points...")

    # Subtract threshold from mean (entropy is relative to threshold)
    selected_in_pool = strategy.iterative_batch_selector(
        score_function, choice_function, pool_mean_dev - threshold, sample_cov_dev, n_select, device
    )

    # Map back to original candidate indices
    selected_indices = pool_indices[selected_in_pool].numpy()

    if logger:
        logger.info(f"Entropy batch selection complete: {len(selected_indices)} points selected")
        if proximity_sampling > 0.0:
            selected_means = pred_mean[selected_indices].squeeze()
            logger.info(f"Selected point predictions: mean={selected_means.mean():.4f}, "
                       f"std={selected_means.std():.4f}, "
                       f"range=[{selected_means.min():.4f}, {selected_means.max():.4f}]")

    return selected_indices


# ===== Unified Selection Interface =====

def select_points(strategy='top_k', **kwargs):
    """
    Unified interface for all selection strategies.

    Args:
        strategy: Selection strategy name
            - 'top_k': Pure variance-based selection
            - 'entropy_batch': Entropy-based selection with optional proximity weighting
        **kwargs: Strategy-specific arguments

    Returns:
        selected_indices: Indices of selected points
    """
    if strategy == 'top_k':
        return select_top_uncertain(
            kwargs['X_candidates'],
            kwargs['uncertainties'],
            kwargs['n_select']
        )
    elif strategy == 'entropy_batch':
        return select_entropy_batch_mc(
            kwargs['X_candidates'],
            kwargs['predictions'],
            kwargs['pred_mean'],
            kwargs['pred_var'],
            kwargs['n_select'],
            blur=kwargs.get('blur', 0.15),
            beta=kwargs.get('beta', 50.0),
            n_pool=kwargs.get('n_pool', 5000),
            threshold=kwargs.get('threshold', 0.0),
            proximity_sampling=kwargs.get('proximity_sampling', 0.0),
            device=kwargs.get('device', 'cpu'),
            logger=kwargs.get('logger', None)
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
