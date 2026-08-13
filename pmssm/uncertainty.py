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


def _last_linear_layer(model):
    """The final nn.Linear, whose weights carry the Laplace posterior."""
    import torch.nn as nn
    last = None
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            last = mod
    if last is None:
        raise ValueError("model has no nn.Linear to linearise")
    return last


def _penultimate_features(model, X_norm, device, layer, batch_size=8192):
    """Deterministic output and the features feeding ``layer``.

    Returns (mu (N,), Phi (N, d+1)) in float64, with a trailing column of ones
    so the layer's bias is treated as one more posterior parameter. Runs in eval
    mode: these are the MAP function and its features, with dropout OFF.
    """
    import torch
    grab = {}

    def _hook(_mod, inputs, _out):
        grab["phi"] = inputs[0].detach()

    handle = layer.register_forward_hook(_hook)
    model.to(device)
    model.eval()
    mus, phis = [], []
    try:
        with torch.no_grad():
            for i in range(0, len(X_norm), batch_size):
                y = model(X_norm[i:i + batch_size].to(device))
                mus.append(y.detach().cpu().reshape(-1))
                phis.append(grab["phi"].cpu())
    finally:
        handle.remove()
    Phi = torch.cat(phis).to(torch.float64)
    ones = torch.ones(len(Phi), 1, dtype=torch.float64)
    return torch.cat(mus).to(torch.float64), torch.cat([Phi, ones], dim=1)


def compute_uncertainty_laplace(model, X_candidates, X_fit, y_fit_t, stats,
                                n_samples, device, logger,
                                X_noise=None, y_noise_t=None,
                                prior_precision=1.0, return_predictions=False,
                                draw_seed=None):
    """Predictive uncertainty from a last-layer linearised Laplace posterior.

    Mirrors ``compute_uncertainty_mc_dropout``'s return contract so it is a
    drop-in third option in the acquisition dispatcher.

    A Gaussian posterior is placed on the final linear layer's weights,

        Sigma = (Phi^T Phi / sn2 + prior_precision * I)^-1,
        sigma^2(x) = phi(x)^T Sigma phi(x),

    with phi the penultimate features and Phi their matrix over the labelled
    set. Because the output layer is linear and the loss is MSE, this GGN is
    EXACT: the expression above is Bayesian linear regression on the learned
    features, and the only approximation is holding those features fixed
    (Kristiadi et al. 2020; Immer et al. 2021; Daxberger et al. 2021).

    Why this rather than dropout or ensembles. Both of those perturb one trained
    optimum and report how unstable the fitted function is under those
    perturbations, which is not a function of distance from the labelled set.
    sigma^2(x) here is the Mahalanobis norm of phi(x) under the training feature
    covariance, so it grows where the features are unlike anything labelled.
    Two practical consequences:

      Cost. One forward pass, plus O(d^3) once and O(d^2) per candidate, against
      dropout's ``n_samples`` forward passes. d is fixed by the architecture, so
      unlike a GP nothing here grows with the labelled-set size.

      Rank. Draws are generated from the analytic posterior, so the candidate
      covariance has rank min(n_samples - 1, d+1) rather than dropout's
      n_samples - 1. That matters for entropy_batch, whose batch score needs
      log det Sigma over a batch far larger than a dropout stack can span.

    The mean and variance returned are ANALYTIC (no Monte-Carlo error); the
    sample stack exists only for callers that need a covariance. As with the
    dropout version the variance is model spread alone, with no observation
    noise added, so the two are on the same footing.

    Args:
        model: trained network ending in a linear layer
        X_candidates: non-normalised candidates (N, D)
        X_fit: non-normalised labelled inputs used to build Phi^T Phi
        y_fit_t: labelled targets in the MODEL'S OUTPUT SPACE (i.e. already
            passed through the y-transform), used only to estimate sn2
        stats: (mean_X, std_X, mean_Y, std_Y)
        n_samples: number of posterior draws for the returned stack
        X_noise, y_noise_t: optional held-out set for sn2. Preferred, since
            training residuals understate the noise scale.
        prior_precision: Gaussian prior precision on the last-layer weights.
            Ranking is insensitive to it in this regime (Phi^T Phi / sn2 is
            O(1e4) here, so any sane value is negligible against it).
        return_predictions: also return the (n_samples, N, 1) draw stack

    Returns:
        pred_mean (N, 1), pred_var (N, 1)[, predictions (n_samples, N, 1)]
    """
    import torch
    mean_X, std_X, _mean_Y, _std_Y = stats

    def _norm(x):
        xn = x if torch.is_tensor(x) else torch.as_tensor(x)
        return ((xn.float() - mean_X) / std_X).float()

    layer = _last_linear_layer(model)
    mu_fit, Phi_fit = _penultimate_features(model, _norm(X_fit), device, layer)

    # Observation-noise scale. Held-out residuals if available, else the fit
    # residuals with a degrees-of-freedom correction.
    d = Phi_fit.shape[1]
    if X_noise is not None and y_noise_t is not None and len(X_noise) > 0:
        mu_n, _Phi_n = _penultimate_features(model, _norm(X_noise), device, layer)
        r = torch.as_tensor(y_noise_t).to(torch.float64).reshape(-1) - mu_n
        sn2 = float((r ** 2).mean())
    else:
        r = torch.as_tensor(y_fit_t).to(torch.float64).reshape(-1) - mu_fit
        dof = max(1, len(r) - d)
        sn2 = float((r ** 2).sum() / dof)
    sn2 = max(sn2, 1e-8)

    # Posterior precision, Cholesky-factorised. Jitter only if the features are
    # rank-deficient (dead units), which the prior term normally already fixes.
    A = Phi_fit.T @ Phi_fit / sn2 + float(prior_precision) * torch.eye(
        d, dtype=torch.float64)
    L = None
    for extra in (0.0, 1e-8, 1e-6, 1e-4, 1e-2):
        try:
            L = torch.linalg.cholesky(
                A + extra * torch.eye(d, dtype=torch.float64))
            break
        except Exception:
            continue
    if L is None:
        raise RuntimeError("Laplace posterior precision is not factorisable")

    # MAP weights, so the analytic mean can be cross-checked against the net.
    w_map = torch.cat([layer.weight.detach().cpu().reshape(-1),
                       layer.bias.detach().cpu().reshape(-1)]).to(torch.float64)

    mu_c, Phi_c = _penultimate_features(model, _norm(X_candidates), device, layer)

    # Exact predictive variance: q(x) = ||L^-1 phi(x)||^2.
    V = torch.linalg.solve_triangular(L, Phi_c.T, upper=False)
    q = (V ** 2).sum(dim=0)
    pred_mean = mu_c.reshape(-1, 1).float()
    pred_var = q.reshape(-1, 1).float()

    if logger:
        # Phi @ w_map must reproduce the network's own output exactly.
        drift = float((Phi_c @ w_map - mu_c).abs().max())
        logger.info(
            f"Laplace uncertainty: d={d} sn2={sn2:.5f} "
            f"prior_precision={prior_precision:g} "
            f"mean_var={pred_var.mean():.6f} max_var={pred_var.max():.6f} "
            f"(MAP reconstruction error {drift:.2e})")

    if not return_predictions:
        return pred_mean, pred_var

    # Draws for callers needing a covariance: w_s = w_MAP + L^-T z_s has
    # covariance L^-T L^-1 = A^-1 = Sigma, exactly.
    #
    # draw_seed must be threaded through from the run's seed (and iteration).
    # A fixed constant here, which is what the published runs used, gives every
    # replica of a cell the identical z at every iteration, so the arms' draws
    # are common random numbers rather than independent samples. The analytic
    # mean and variance above are unaffected -- they never touch z -- but the
    # entropy_batch covariance is built from these draws, so seed-to-seed
    # spread on any batch-selected quantity is understated.
    g = torch.Generator().manual_seed(0 if draw_seed is None else int(draw_seed))
    z = torch.randn(d, int(n_samples), generator=g, dtype=torch.float64)
    W = w_map.reshape(-1, 1) + torch.linalg.solve_triangular(L.T, z, upper=True)
    predictions = (Phi_c @ W).T.unsqueeze(-1).float()      # (n_samples, N, 1)

    if logger:
        samp = predictions.squeeze(-1).var(dim=0, unbiased=True).mean()
        logger.info(
            f"Laplace draws: n_samples={n_samples} "
            f"covariance rank <= {min(int(n_samples) - 1, d)} "
            f"(sample mean_var={samp:.6f} vs analytic {pred_var.mean():.6f})")
    return pred_mean, pred_var, predictions
