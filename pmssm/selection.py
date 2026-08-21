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


def select_top_uncertain_filtered(X_candidates, pred_mean, pred_var, n_select,
                                  threshold=0.0, tolerance_sampling=0.0,
                                  proximity_sampling=0.0, logger=None):
    """
    Variance-based top-k selection with optional tolerance and proximity filters.

    Mirrors the pre-filter stages of :func:`select_entropy_batch_mc` so both
    strategies honor the same pre-filtering semantics.

    Stages:
      1. Hard tolerance cut: keep candidates with ``pred_mean`` in
         ``[threshold - tolerance_sampling, threshold + tolerance_sampling]``.
      2. Proximity-weighted variance:
         ``weighted_var = var * exp(-(pred_mean - threshold)^2 / proximity_sampling)``.
      3. Return the top ``n_select`` by (weighted) variance.

    Args:
        X_candidates: Candidate pool tensor (N, D)
        pred_mean: Predicted mean in transformed space (N, 1) or (N,)
        pred_var: Prediction variance (N, 1) or (N,)
        n_select: Number to select
        threshold: Decision threshold in transformed space
        tolerance_sampling: ± width around threshold for hard cut (0 to disable)
        proximity_sampling: Gaussian proximity width (0 to disable)
        logger: Logger instance

    Returns:
        Numpy array of indices into ``X_candidates`` (len ≤ ``n_select``).
    """
    N = X_candidates.shape[0]
    mean_flat = pred_mean.squeeze()

    # Step 1: hard tolerance cut
    if tolerance_sampling > 0.0:
        mask = ((mean_flat > threshold - tolerance_sampling) &
                (mean_flat < threshold + tolerance_sampling))
        surviving_indices = torch.where(mask)[0]
        if logger:
            logger.info(f"Tolerance filter (±{tolerance_sampling:.2f}): "
                       f"{len(surviving_indices)}/{N} candidates survive")
        if len(surviving_indices) == 0:
            if logger:
                logger.warning("No candidates survived tolerance filter, "
                              "falling back to all candidates")
            surviving_indices = torch.arange(N)
    else:
        surviving_indices = torch.arange(N)

    # Step 2: proximity-weighted variance on survivors
    surv_var = pred_var[surviving_indices]
    if proximity_sampling > 0.0:
        surv_mean = mean_flat[surviving_indices]
        proximity = torch.exp(-((surv_mean - threshold) ** 2) / proximity_sampling)
        if surv_var.dim() == 2:
            weighted_var = proximity.unsqueeze(1) * surv_var
        else:
            weighted_var = proximity * surv_var
        if logger:
            logger.info(f"Proximity weighting (σ={proximity_sampling:.3f}): "
                       f"mean={proximity.mean():.4f}, max={proximity.max():.4f}")
    else:
        weighted_var = surv_var

    # Step 3: top-k by (weighted) variance
    var_flat = weighted_var.squeeze().numpy()
    k = min(n_select, len(surviving_indices))
    topk_in_surv = np.argsort(var_flat)[::-1][:k].copy()
    return surviving_indices.numpy()[topk_in_surv]


def select_top_uncertain_tol_only(X_candidates, pred_mean, pred_var, n_select,
                                  threshold=0.0, tolerance_sampling=0.0, logger=None):
    """Tolerance-filtered top-k by raw variance. No proximity weighting.

    Short-circuit variant of :func:`select_top_uncertain_filtered` that stops
    the selection pipeline after the tolerance cut — ranks survivors by raw
    ``pred_var`` instead of proximity-weighted variance.
    """
    return select_top_uncertain_filtered(
        X_candidates, pred_mean, pred_var, n_select,
        threshold=threshold,
        tolerance_sampling=tolerance_sampling,
        proximity_sampling=0.0,
        logger=logger,
    )


# ===== Entropy-Based Selection with Proximity Weighting =====

def select_top_score(X_candidates, score, n_select, pred_mean=None,
                     threshold=0.0, tolerance_sampling=0.0, logger=None):
    """Top-k by an arbitrary acquisition score, with the optional tolerance cut.

    The head-agnostic counterpart of :func:`select_top_uncertain_filtered`: the
    ranking quantity is whatever the head produced (predictive entropy, mutual
    information, variance), so a new head needs no new selector.

    The tolerance cut is available but off by default, because a
    boundary-anchored score such as the predictive entropy of a verdict
    classifier already concentrates on the decision surface and does not need
    the mean-based pre-filter that a variance ranking does. Passing
    ``tolerance_sampling`` reinstates it, which is how a matched-anchor
    comparison against the regression path is run.

    Args:
        X_candidates: Candidate pool tensor (N, D)
        score: Acquisition score (N,) or (N, 1); higher is more worth labelling
        n_select: Number to select
        pred_mean: Mean in transformed space, required only for the cut
        threshold: Decision threshold in transformed space
        tolerance_sampling: +/- width around threshold for the hard cut (0 = off)
        logger: Logger instance

    Returns:
        Numpy array of indices into ``X_candidates`` (len <= ``n_select``).
    """
    N = X_candidates.shape[0]
    score_flat = score.squeeze().to(torch.float64)

    if tolerance_sampling > 0.0:
        if pred_mean is None:
            raise ValueError("tolerance_sampling requires pred_mean")
        mean_flat = pred_mean.squeeze()
        mask = ((mean_flat > threshold - tolerance_sampling) &
                (mean_flat < threshold + tolerance_sampling))
        surviving = torch.where(mask)[0]
        if logger:
            logger.info(f"Tolerance filter (+/-{tolerance_sampling:.2f}): "
                        f"{len(surviving)}/{N} candidates survive")
        if len(surviving) == 0:
            if logger:
                logger.warning("No candidates survived tolerance filter, "
                               "falling back to all candidates")
            surviving = torch.arange(N)
    else:
        surviving = torch.arange(N)

    take = min(n_select, len(surviving))
    top = torch.topk(score_flat[surviving], take).indices
    chosen = surviving[top]
    if logger:
        logger.info(f"Selected {take} candidates by score "
                    f"(range {float(score_flat[chosen].min()):.4g} to "
                    f"{float(score_flat[chosen].max()):.4g})")
    return chosen.numpy()


def select_tol_only_random(X_candidates, pred_mean, n_select, threshold=0.0,
                           tolerance_sampling=0.0, seed=0, logger=None):
    """Tolerance cut, then a uniform draw among the survivors: no uncertainty.

    The mean-guided arm the strategy grid was missing. Every other strategy
    applies this same cut and then *ranks* the survivors by some function of the
    predictive uncertainty, so this one isolates what the cut contributes on its
    own, which is the question "how much of the yield is the prefilter?".

    It needs only the predicted mean, so it is available to every surrogate,
    including those whose uncertainty is expensive (TabPFN) or absent.

    Args:
        X_candidates: Candidate pool tensor (N, D)
        pred_mean: Predicted mean in transformed space (N, 1) or (N,)
        n_select: Number to select
        threshold: Decision threshold in transformed space
        tolerance_sampling: +/- width around threshold for the hard cut (0 = off,
            which degenerates to uniform random selection over the whole pool)
        seed: Draw seed; vary it per iteration so replicas are not correlated
        logger: Logger instance

    Returns:
        Numpy array of indices into ``X_candidates`` (len <= ``n_select``).
    """
    N = X_candidates.shape[0]
    mean_flat = pred_mean.squeeze()

    if tolerance_sampling > 0.0:
        mask = ((mean_flat > threshold - tolerance_sampling) &
                (mean_flat < threshold + tolerance_sampling))
        surviving = torch.where(mask)[0]
        if logger:
            logger.info(f"Tolerance filter (+/-{tolerance_sampling:.2f}): "
                        f"{len(surviving)}/{N} candidates survive")
        if len(surviving) == 0:
            if logger:
                logger.warning("No candidates survived tolerance filter, "
                               "falling back to all candidates")
            surviving = torch.arange(N)
    else:
        surviving = torch.arange(N)

    take = min(n_select, len(surviving))
    g = torch.Generator().manual_seed(int(seed))
    chosen = surviving[torch.randperm(len(surviving), generator=g)[:take]]
    if logger:
        logger.info(f"Selected {take} survivors uniformly at random "
                    f"(seed {int(seed)}); no uncertainty was consulted")
    return chosen.numpy()


def select_entropy_batch_mc(X_candidates, predictions, pred_mean, pred_var,
                            n_select, blur=0.15, beta=50.0, n_pool=5000,
                            threshold=0.0, tolerance_sampling=0.0,
                            proximity_sampling=0.0,
                            device='cpu', logger=None):
    """
    Entropy-based batch selection using MC Dropout/ensemble sample covariance.

    Pre-filters candidates (hard tolerance cut, then proximity-weighted variance
    ranking) to a focused pool, computes sample covariance from predictions,
    then uses EntropySelectionStrategy for iterative batch selection with diversity.

    Args:
        X_candidates: (N, D) candidate points
        predictions: (T, N, 1) MC Dropout/ensemble predictions tensor
        pred_mean: (N, 1) mean predictions in transformed space
        pred_var: (N, 1) prediction variance
        n_select: Number of points to select
        blur: Entropy smoothing parameter
        beta: Gibbs sampling temperature (high=deterministic, low=random)
        n_pool: Focused pool size (pre-filtered by variance)
        threshold: Decision threshold in transformed space (default: 0.0)
        tolerance_sampling: Hard cut width around threshold (0 to disable).
            Keeps only candidates with pred_mean in [threshold ± tolerance].
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
    n_select = min(n_select, N)

    # Failsafe: prevent duplicates by ensuring pool size >= selection size
    if n_select > n_pool:
        if logger:
            logger.warning(
                f"Failsafe triggered: n_select ({n_select}) > entropy_pool_size ({n_pool}). "
                f"This would cause duplicate points to be selected. "
                f"Automatically increasing pool size to {n_select}."
            )
            logger.warning(
                f"Note: Larger pool size may increase memory usage. "
                f"If OOM errors occur, reduce --n-select or increase --entropy-pool-size explicitly."
            )
        n_pool = n_select

    # Step 1: Hard tolerance cut (if enabled) — keep only candidates near threshold
    if tolerance_sampling > 0.0:
        mean_flat = pred_mean.squeeze()
        mask = (mean_flat > threshold - tolerance_sampling) & (mean_flat < threshold + tolerance_sampling)
        surviving_indices = torch.where(mask)[0]
        if logger:
            logger.info(f"Tolerance filter (±{tolerance_sampling:.2f}): "
                       f"{len(surviving_indices)}/{N} candidates survive")
        if len(surviving_indices) == 0:
            if logger:
                logger.warning("No candidates survived tolerance filter, falling back to all candidates")
            surviving_indices = torch.arange(N)
    else:
        surviving_indices = torch.arange(N)

    # Step 2: Proximity-weighted variance ranking on survivors
    surv_mean = pred_mean[surviving_indices]
    surv_var = pred_var[surviving_indices]

    if proximity_sampling > 0.0:
        proximity = torch.exp(-((surv_mean.squeeze() - threshold) ** 2) / proximity_sampling)
        weighted_var = proximity.unsqueeze(1) * surv_var

        if logger:
            logger.info(f"Proximity weighting (σ={proximity_sampling:.3f}): "
                       f"mean={proximity.mean():.4f}, max={proximity.max():.4f}")
    else:
        weighted_var = surv_var

    # Step 3: Take top n_pool by weighted variance
    k = min(n_pool, len(surviving_indices))
    var_flat = weighted_var.squeeze()
    topk = torch.argsort(var_flat, descending=True)[:k]
    pool_indices = surviving_indices[topk]

    if logger:
        logger.info(f"Focused pool: {len(pool_indices)} candidates (from {len(surviving_indices)} after tolerance filter)")

    # Extract predictions for the focused pool: (T, n_pool, 1)
    pool_preds = predictions[:, pool_indices, :]  # (T, n_pool, 1)
    pool_mean = pred_mean[pool_indices].squeeze()  # (n_pool,)

    # Compute sample covariance from MC Dropout predictions
    preds_2d = pool_preds.squeeze(-1)  # (T, n_pool)
    mean_2d = preds_2d.mean(dim=0)     # (n_pool,)
    centered = preds_2d - mean_2d      # (T, n_pool)
    T = preds_2d.shape[0]
    sample_cov = (centered.T @ centered) / (T - 1)  # (n_pool, n_pool)

    # Regularize: sample covariance is rank-deficient when T < pool size.
    # The identity must be sized on the ACTUAL pool, not on the requested
    # n_pool: the tolerance filter above can leave fewer survivors than
    # entropy_pool_size, in which case k = min(n_pool, len(surviving_indices))
    # shrinks the pool and torch.eye(n_pool) no longer matches. In production
    # the two are equal (1e6 candidates against a 5000 pool), so this only ever
    # fired on reduced-size runs, which is to say on every smoke test:
    # "The size of tensor a (34) must match the size of tensor b (100)".
    sample_cov += 1e-4 * torch.eye(sample_cov.shape[0], dtype=sample_cov.dtype)

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
        # candidate_batch_size=100  # Process 100 candidates at a time to reduce memory usage
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
            - 'top_k': Tolerance + proximity-weighted variance + top-k
            - 'top_k_tol_only': Tolerance cut + raw top-variance (short-circuit, no proximity)
            - 'entropy_batch': Full DPP-style entropy pipeline with MC covariance
            - 'head_score': Top-k by a head-supplied score (see pmssm.heads),
              with the tolerance cut optional
            - 'tol_only_random': Tolerance cut, then a uniform draw among the
              survivors; the mean-guided arm with no uncertainty at all
        **kwargs: Strategy-specific arguments

    Returns:
        selected_indices: Indices of selected points
    """
    if strategy == 'top_k':
        # Prefer the filtered variant when pred_mean is supplied so tolerance
        # and proximity filters can be applied consistently with entropy_batch.
        if 'pred_mean' in kwargs and 'pred_var' in kwargs:
            return select_top_uncertain_filtered(
                kwargs['X_candidates'],
                kwargs['pred_mean'],
                kwargs['pred_var'],
                kwargs['n_select'],
                threshold=kwargs.get('threshold', 0.0),
                tolerance_sampling=kwargs.get('tolerance_sampling', 0.0),
                proximity_sampling=kwargs.get('proximity_sampling', 0.0),
                logger=kwargs.get('logger', None),
            )
        return select_top_uncertain(
            kwargs['X_candidates'],
            kwargs['uncertainties'],
            kwargs['n_select']
        )
    elif strategy == 'tol_only_random':
        return select_tol_only_random(
            kwargs['X_candidates'],
            kwargs['pred_mean'],
            kwargs['n_select'],
            threshold=kwargs.get('threshold', 0.0),
            tolerance_sampling=kwargs.get('tolerance_sampling', 0.0),
            seed=kwargs.get('seed', 0),
            logger=kwargs.get('logger', None),
        )
    elif strategy == 'head_score':
        # Head-supplied ranking (e.g. a classifier's predictive entropy or its
        # mutual information). The head decides what `score` means; the selector
        # only ranks. Existing strategies are untouched.
        return select_top_score(
            kwargs['X_candidates'],
            kwargs['score'],
            kwargs['n_select'],
            pred_mean=kwargs.get('pred_mean'),
            threshold=kwargs.get('threshold', 0.0),
            tolerance_sampling=kwargs.get('tolerance_sampling', 0.0),
            logger=kwargs.get('logger', None),
        )
    elif strategy == 'top_k_tol_only':
        return select_top_uncertain_tol_only(
            kwargs['X_candidates'],
            kwargs['pred_mean'],
            kwargs['pred_var'],
            kwargs['n_select'],
            threshold=kwargs.get('threshold', 0.0),
            tolerance_sampling=kwargs.get('tolerance_sampling', 0.0),
            logger=kwargs.get('logger', None),
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
