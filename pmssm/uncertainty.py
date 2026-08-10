"""
Uncertainty estimation for active learning.

This module provides functions to compute predictive uncertainty using:
- MC Dropout (for transformer/neural network models)
- GP posterior variance (for Gaussian Process models)
"""

import torch
import gpytorch


# ===== MC Dropout Uncertainty (Transformer Models) =====

def compute_uncertainty_mc_dropout(model, X_candidates, stats, n_samples, device, logger,
                                   return_predictions=False):
    """
    Compute predictive uncertainty using MC Dropout.

    Args:
        model: Trained PMSSMTransformerTabular with dropout
        X_candidates: Non-normalized candidates (N, 19)
        stats: (mean_X, std_X, mean_Y, std_Y) normalization stats
        n_samples: Number of stochastic forward passes
        device: Compute device
        logger: Logger instance
        return_predictions: If True, also return the raw (T, N, 1) predictions tensor

    Returns:
        pred_mean: (N, 1) mean predictions (normalized)
        pred_var: (N, 1) prediction variance (uncertainty)
        predictions: (T, N, 1) raw predictions (only if return_predictions=True)
    """
    model.to(device)
    model.train()  # Keep dropout active for MC sampling

    mean_X, std_X, mean_Y, std_Y = stats

    # Normalize candidates
    X_norm = (X_candidates - mean_X) / std_X
    X_norm = X_norm.to(device)

    predictions = []
    if logger:
        logger.info(f"Running {n_samples} MC Dropout forward passes...")

    # Batch size limit: PyTorch efficient attention fails above 65535
    batch_size = 8192
    with torch.no_grad():
        for _ in range(n_samples):
            if len(X_norm) <= batch_size:
                y_pred = model(X_norm)
            else:
                y_pred = torch.cat([model(X_norm[i:i+batch_size])
                                    for i in range(0, len(X_norm), batch_size)], dim=0)
            predictions.append(y_pred.cpu())

    predictions = torch.stack(predictions, dim=0)  # (T, N, 1)

    pred_mean = predictions.mean(dim=0)  # (N, 1)
    pred_var = predictions.var(dim=0)    # (N, 1) - uncertainty

    if logger:
        logger.info(f"Uncertainty stats: mean={pred_var.mean():.6f}, max={pred_var.max():.6f}")

    if return_predictions:
        return pred_mean, pred_var, predictions
    return pred_mean, pred_var


def compute_uncertainty_ensemble(models, X_candidates, stats, device, logger,
                                 return_predictions=False):
    """Predictive uncertainty from disagreement across independently trained members.

    Deliberately mirrors ``compute_uncertainty_mc_dropout``'s return contract, so
    the selection path is indifferent to which produced the numbers: the (K, N, 1)
    stack of member predictions plays the role of the (T, N, 1) stack of dropout
    passes.

    The difference is what varies between rows. Dropout perturbs one trained
    optimum with Bernoulli masks whose scale is set by the dropout rate, a
    regularisation hyperparameter; the spread it reports is the network's local
    sensitivity to masking. Ensemble members are separate optima reached from
    different initialisations, so their disagreement answers the epistemic
    question directly, namely how much functions that all fit the labelled data
    still differ here, and it contracts where labels accumulate without that
    having to be assumed.

    Members are evaluated in eval mode, so dropout is OFF: the variance reported
    is across-member only and does not mix in mask noise.

    Args:
        models: list of K trained models, same architecture, independent inits
        X_candidates: non-normalised candidates (N, 19)
        stats: (mean_X, std_X, mean_Y, std_Y)
        device, logger: as for the dropout version
        return_predictions: also return the (K, N, 1) member stack

    Returns:
        pred_mean, pred_var[, predictions] with the same shapes as the MC-dropout
        version. NOTE the sample covariance across K members has rank at most
        K-1, so batch scores depending on log det Sigma are degenerate for
        batches larger than that; use top-k selection unless K is large.
    """
    import torch
    mean_X, std_X, mean_Y, std_Y = stats
    X_norm = (X_candidates - mean_X) / std_X
    if not torch.is_tensor(X_norm):
        X_norm = torch.as_tensor(X_norm)
    X_norm = X_norm.float()
    batch_size = 8192
    predictions = []
    for k, model in enumerate(models):
        model.to(device)
        model.eval()                     # dropout off: across-member spread only
        with torch.no_grad():
            y_pred = torch.cat([model(X_norm[i:i + batch_size].to(device)).cpu()
                                for i in range(0, len(X_norm), batch_size)], dim=0)
        predictions.append(y_pred)
    predictions = torch.stack(predictions, dim=0)          # (K, N, 1)
    pred_mean = predictions.mean(dim=0)
    # unbiased across-member variance; K is small so the correction matters
    pred_var = predictions.var(dim=0, unbiased=True) if len(models) > 1 \
        else torch.zeros_like(pred_mean)
    if logger:
        logger.info(f"Ensemble uncertainty over K={len(models)} members: "
                    f"mean={pred_var.mean():.6f}, max={pred_var.max():.6f}")
        if pred_var.mean() == 0:
            logger.warning("Ensemble variance is identically zero: the members "
                           "are not distinct. Check that each was trained with "
                           "its own seed and a fresh initialisation.")
    if return_predictions:
        return pred_mean, pred_var, predictions
    return pred_mean, pred_var


# ===== GP Posterior Uncertainty (GP Models) =====

def compute_uncertainty_gp(model, X_candidates, data_min, data_max,
                          model_type, jitter=1e-3, num_samples=8, logger=None):
    """
    Compute GP posterior variance on candidate points.

    For MLP models (no native uncertainty), returns zero variance and a warning.

    Args:
        model: Trained GP model (ExactGP, DeepGP, SparseGP) or MLP
        X_candidates: Non-normalized candidates (N, 19) in physical units
        data_min, data_max: Normalization tensors
        model_type: One of "exact_gp", "deep_gp", "sparse_gp", "mlp"
        jitter: Cholesky jitter
        num_samples: Number of likelihood samples for DeepGP
        logger: Logger instance

    Returns:
        pred_mean: (N,) mean predictions (in transformed space)
        pred_var: (N,) prediction variance (uncertainty)
    """
    from .data import normalize_x

    device = next(model.parameters()).device
    X_norm = normalize_x(X_candidates, data_min, data_max).to(device)

    model.eval()

    if model_type == "mlp":
        if logger:
            logger.warning("MLP has no native uncertainty; returning zero variance. "
                          "Selection will fall back to random.")
        with torch.no_grad():
            pred_mean = model(X_norm).squeeze().cpu()
        pred_var = torch.zeros(len(X_norm))
        return pred_mean, pred_var

    if model_type == "deep_gp":
        model.likelihood.eval()
        batch_size = 10000
        means, variances = [], []
        for i in range(0, len(X_norm), batch_size):
            x_batch = X_norm[i:i + batch_size]
            with torch.no_grad(), \
                 gpytorch.settings.fast_pred_var(False), \
                 gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter), \
                 gpytorch.settings.num_likelihood_samples(num_samples):
                preds = model.likelihood(model(x_batch))
                means.append(preds.mean.detach().mean(dim=0).squeeze())
                variances.append(preds.variance.detach().mean(dim=0).squeeze())
        pred_mean = torch.cat(means).cpu()
        pred_var = torch.cat(variances).cpu()
    else:
        # exact_gp, sparse_gp
        model.likelihood.eval()
        batch_size = 5000
        means, variances = [], []
        for i in range(0, len(X_norm), batch_size):
            x_batch = X_norm[i:i + batch_size]
            with torch.no_grad(), \
                 gpytorch.settings.fast_pred_var(), \
                 gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter):
                preds = model.likelihood(model(x_batch))
                means.append(preds.mean.detach())
                variances.append(preds.variance.detach())
        pred_mean = torch.cat(means).cpu()
        pred_var = torch.cat(variances).cpu()

    if logger:
        logger.info(f"GP uncertainty stats: mean_var={pred_var.mean():.6f}, "
                   f"max_var={pred_var.max():.6f}, min_var={pred_var.min():.6f}")

    return pred_mean, pred_var
