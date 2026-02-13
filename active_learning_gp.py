"""
Active Learning pipeline for pMSSM relic density prediction using Gaussian Process models.

Mirrors active_learning.py structure but uses ExactGP or DeepGP from the al_pmssmwithgp repo
instead of the tabular transformer, and uses native GP posterior variance for uncertainty
estimation instead of MC Dropout.
"""

import warnings
warnings.filterwarnings('ignore', message='.*enable_nested_tensor.*')

import sys
from pathlib import Path

# Add gp_pipeline to import path
_GP_PIPELINE_ROOT = Path(__file__).parent / "al_pmssmwithgp" / "model"
sys.path.insert(0, str(_GP_PIPELINE_ROOT))

import pmssm
from active_learning import (
    PARAM_ORDER,
    PARAM_RANGES,
    CSV_TO_MODELGEN,
    _run_modelgen,
    generate_models_from_csv,
    load_generated_data,
    setup_logging,
    setup_worker_logging,
    generate_candidate_pool,
    save_selected_points,
    plot_iteration_metrics,
    select_top_uncertain,
)

from datetime import datetime
import logging
import structlog
import json
import multiprocessing as mp
import copy

import click
import pandas as pd
import torch
import gpytorch

import yaml
from itertools import product as iterproduct

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


# ---------------------------------------------------------------------------
# GP-specific constants
# ---------------------------------------------------------------------------

# Target-specific configuration (true values, thresholds, ROOT branch names)
TARGET_CONFIG = {
    "DMRD": {"true_value": 0.12, "threshold": 0.0, "branch": "MO_Omega"},
    "CrossSection": {"true_value": 0.03, "threshold": 0.0, "branch": "xsec_TOTAL"},
    "CLs": {"true_value": 0.05, "threshold": 0.05, "branch": "Final__CLs"},
}

# Backward-compatible alias
DMRD_TRUE_VALUE = TARGET_CONFIG["DMRD"]["true_value"]

# Min-max normalization ranges from the GP pipeline (base.py:90-99)
# Keys use the GP repo's naming convention (no IN_ prefix, AT instead of At)
GP_RANGE_DICT = {
    "M_1": [-2000, 2000], "M_2": [-2000, 2000], "tanb": [1, 60],
    "mu": [-2000, 2000], "M_3": [1000, 5000], "AT": [-8000, 8000],
    "Ab": [-2000, 2000], "Atau": [-2000, 2000], "mA": [0, 5000],
    "mqL3": [2000, 5000], "mtR": [2000, 5000], "mbR": [2000, 5000],
    "meL": [0, 10000], "mtauL": [0, 10000], "meR": [0, 10000],
    "mtauR": [0, 10000], "mqL1": [0, 10000], "muR": [0, 10000],
    "mdR": [0, 10000],
}

# Map from PARAM_ORDER names (IN_xxx) to GP_RANGE_DICT keys
_PARAM_TO_RANGE_KEY = {
    "IN_meL": "meL", "IN_meR": "meR", "IN_mtauL": "mtauL", "IN_mtauR": "mtauR",
    "IN_mqL1": "mqL1", "IN_muR": "muR", "IN_mdR": "mdR", "IN_mqL3": "mqL3",
    "IN_mtR": "mtR", "IN_mbR": "mbR", "IN_M_1": "M_1", "IN_M_2": "M_2",
    "IN_mu": "mu", "IN_M_3": "M_3", "IN_At": "AT", "IN_Ab": "Ab",
    "IN_Atau": "Atau", "IN_mA": "mA", "IN_tanb": "tanb",
}


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def build_norm_tensors():
    """Build data_min and data_max tensors in PARAM_ORDER ordering."""
    mins, maxs = [], []
    for param in PARAM_ORDER:
        key = _PARAM_TO_RANGE_KEY[param]
        lo, hi = GP_RANGE_DICT[key]
        mins.append(lo)
        maxs.append(hi)
    return torch.tensor(mins, dtype=torch.float32), torch.tensor(maxs, dtype=torch.float32)


def normalize_x(X, data_min, data_max):
    """Min-max normalize inputs to [0, 1]."""
    return (X - data_min) / (data_max - data_min)


def unnormalize_x(X_norm, data_min, data_max):
    """Reverse min-max normalization."""
    return X_norm * (data_max - data_min) + data_min


def transform_y(Y, target="DMRD"):
    """Transform target values. DMRD/CrossSection: log(Y / true_value). CLs: untransformed."""
    if target == "CLs":
        return Y.clone()
    true_value = TARGET_CONFIG[target]["true_value"]
    return torch.log(Y / true_value)


def inverse_transform_y(Y_t, target="DMRD"):
    """Inverse of transform_y: convert transformed values back to physical space."""
    if target == "CLs":
        return Y_t.clone()
    true_value = TARGET_CONFIG[target]["true_value"]
    return true_value * torch.exp(Y_t)


# ---------------------------------------------------------------------------
# GP model creation, training, and evaluation
# ---------------------------------------------------------------------------

def create_gp_model(model_type, x_train, y_train, x_val, y_val, n_dim,
                    kernel="RBF", lengthscale=1.0, noise=1e-2, use_ard=True,
                    m_nu=1.5, num_mixtures=4, use_dkl=False, feature_dim=2,
                    num_hidden_dims=10, num_middle_dims=0,
                    num_inducing_max=512, num_samples=8, seed=42, device=None,
                    target="DMRD", inducing_strategy="kmeans"):
    """Create and return a GP model (ExactGP, DeepGP, SparseGP, or MLP).

    Data is moved to the model's device before construction so that
    both self.x_train (stored attribute) and self.train_inputs (GPyTorch's
    internal storage after self.to(device)) are on the same device.

    Args:
        device: Target device string (e.g. "cuda:0", "cuda:1", "cpu"). If None,
                auto-detects CUDA. Pass explicitly when running on a specific GPU
                in a multiprocessing worker.
        target: Target function name (for threshold in model construction).
        inducing_strategy: Inducing point initialization for SparseGP ("kmeans" or "vanilla").
    """
    threshold = TARGET_CONFIG.get(target, TARGET_CONFIG["DMRD"])["threshold"]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    x_train = x_train.to(device)
    y_train = y_train.view(-1).to(device)
    x_val = x_val.to(device)
    y_val = y_val.view(-1).to(device)

    if model_type == "exact_gp":
        model = ExactGP(
            x_train, y_train, x_val, y_val, n_dim,
            lengthscale=lengthscale, use_ard=use_ard, noise=noise,
            kernel=kernel, m_nu=m_nu, num_mixtures=num_mixtures,
            use_dkl=use_dkl, feature_dim=feature_dim, thr=threshold, epsilon=0,
            seed=seed,
        )
    elif model_type == "deep_gp":
        model = DeepGP(
            x_train, y_train, x_val, y_val, n_dim,
            lengthscale=lengthscale, noise=noise,
            num_hidden_dims=num_hidden_dims, num_middle_dims=num_middle_dims,
            num_inducing_max=num_inducing_max, kernel=kernel, m_nu=m_nu,
            num_samples=num_samples, seed=seed,
        )
    elif model_type == "sparse_gp":
        model = SparseGP(
            x_train, y_train, x_val, y_val, n_dim,
            lengthscale=lengthscale, noise=noise,
            num_inducing_max=num_inducing_max, kernel=kernel, m_nu=m_nu,
            thr=threshold, seed=seed, inducing_strategy=inducing_strategy,
        )
    elif model_type == "mlp":
        model = MLP(
            x_train, y_train, x_val, y_val,
            input_dim=n_dim, seed=seed,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model


def train_gp_model(model, model_type, lr=1e-3, iters=2000,
                   batch_size=256, jitter=1e-3, patience=None):
    """Train a GP model and return (model, train_losses, val_losses).

    Args:
        patience: Early stopping patience. None disables early stopping.
    """
    if model_type == "exact_gp":
        model, train_losses, val_losses = model.do_train_loop(
            lr=lr, iters=iters, jitter=jitter, patience=patience
        )
    elif model_type == "deep_gp":
        model, train_losses, val_losses = model.do_train_loop(
            lr=lr, iters=iters, batch_size=batch_size, jitter=jitter,
            patience=patience
        )
    elif model_type == "sparse_gp":
        model, train_losses, val_losses = model.do_train_loop(
            lr=lr, iters=iters, batch_size=batch_size, jitter=jitter,
            patience=patience
        )
    elif model_type == "mlp":
        model, train_losses, val_losses = model.do_train_loop(
            lr=lr, iters=iters, patience=patience
        )
    return model, train_losses, val_losses


def model_has_likelihood(model_type):
    """Return True if the model type uses a GP likelihood."""
    return model_type in ("exact_gp", "deep_gp", "sparse_gp")


def model_is_variational(model_type):
    """Return True if the model type uses variational inference."""
    return model_type in ("deep_gp", "sparse_gp")


def compute_gp_r2(model, x_val, y_val, model_type, jitter=1e-3, num_samples=8):
    """Compute R² score on validation set using model predictions.

    Args:
        model_type: One of "exact_gp", "deep_gp", "sparse_gp", "mlp".
    """
    device = next(model.parameters()).device
    model.eval()

    if model_type == "mlp":
        with torch.no_grad():
            y_pred = model(x_val.to(device)).squeeze()
    elif model_type == "deep_gp":
        model.likelihood.eval()
        with torch.no_grad(), \
             gpytorch.settings.fast_pred_var(False), \
             gpytorch.settings.cholesky_jitter(jitter), \
             gpytorch.settings.num_likelihood_samples(num_samples):
            preds = model.likelihood(model(x_val.to(device)))
            y_pred = preds.mean.detach().mean(dim=0).squeeze()
    else:
        # exact_gp, sparse_gp
        model.likelihood.eval()
        with torch.no_grad(), \
             gpytorch.settings.fast_pred_var(), \
             gpytorch.settings.cholesky_jitter(jitter):
            preds = model.likelihood(model(x_val.to(device)))
            y_pred = preds.mean.detach()

    y_true = y_val.view(-1).to(device)
    y_pred = y_pred.view(-1)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = (1 - ss_res / ss_tot).item()
    return r2


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
                 gpytorch.settings.cholesky_jitter(jitter), \
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
                 gpytorch.settings.cholesky_jitter(jitter):
                preds = model.likelihood(model(x_batch))
                means.append(preds.mean.detach())
                variances.append(preds.variance.detach())
        pred_mean = torch.cat(means).cpu()
        pred_var = torch.cat(variances).cpu()

    if logger:
        logger.info(f"GP uncertainty stats: mean_var={pred_var.mean():.6f}, "
                    f"max_var={pred_var.max():.6f}, min_var={pred_var.min():.6f}")

    return pred_mean, pred_var


# ---------------------------------------------------------------------------
# Comprehensive evaluation metrics (Phase 2)
# ---------------------------------------------------------------------------

def compute_comprehensive_metrics(model, X_eval_norm, Y_eval_true, model_type,
                                  threshold=0.0, jitter=1e-3, num_samples=8,
                                  logger=None):
    """
    Compute comprehensive evaluation metrics on an evaluation dataset.

    Adapts evaluate_and_log from al_pmssmwithgp for standalone use.

    Args:
        model: Trained GP or MLP model.
        X_eval_norm: Normalized evaluation inputs (N, D) on model's device.
        Y_eval_true: True target values in transformed space (N,).
        model_type: One of "exact_gp", "deep_gp", "sparse_gp", "mlp".
        threshold: Classification threshold in transformed space.
        jitter: Cholesky jitter.
        num_samples: DeepGP likelihood samples.
        logger: Logger instance.

    Returns:
        dict with accuracy, weighted_acc, mse, rmse, r2, chi2, chi2_red,
        mean_abs_pull, rms_pull, weighted variants, confusion matrix info.
    """
    device = next(model.parameters()).device
    X_eval_norm = X_eval_norm.to(device)
    Y_eval_true = Y_eval_true.view(-1).to(device)

    model.eval()

    if model_type == "mlp":
        with torch.no_grad():
            mean = model(X_eval_norm).squeeze()
        # MLP has no confidence region
        upper = lower = mean
    elif model_type == "deep_gp":
        model.likelihood.eval()
        with torch.no_grad(), \
             gpytorch.settings.fast_pred_var(False), \
             gpytorch.settings.cholesky_jitter(jitter), \
             gpytorch.settings.num_likelihood_samples(num_samples):
            preds = model.likelihood(model(X_eval_norm))
            mean = preds.mean.detach().mean(dim=0).squeeze()
            lower, upper = preds.confidence_region()
            lower = lower.detach().mean(dim=0).squeeze()
            upper = upper.detach().mean(dim=0).squeeze()
    else:
        # exact_gp, sparse_gp
        model.likelihood.eval()
        with torch.no_grad(), \
             gpytorch.settings.fast_pred_var(), \
             gpytorch.settings.cholesky_jitter(jitter):
            preds = model.likelihood(model(X_eval_norm))
            mean = preds.mean.detach()
            lower, upper = preds.confidence_region()
            lower = lower.detach()
            upper = upper.detach()

    # Classification accuracy
    acc, _conf = compute_accuracy(mean, Y_eval_true, threshold)
    results = {"accuracy": acc, "n_eval": len(Y_eval_true)}

    # Weighted accuracy at multiple alpha values
    for alpha in (1.0, 2.0, 5.0, 10.0):
        acc_w, _ = compute_weighted_accuracy(mean, Y_eval_true, threshold, alpha)
        results[f"weighted_acc_alpha_{alpha}"] = acc_w

    # Regression GoF metrics (need confidence region — skip for MLP)
    if model_type != "mlp":
        gof = compute_gof_metrics(
            predictions=mean, truths=Y_eval_true,
            up=upper, low=lower, thr=threshold,
        )
        results.update(gof)

    # Misclassified counts
    results["n_false_positives"] = int(((mean >= threshold) & (Y_eval_true < threshold)).sum())
    results["n_false_negatives"] = int(((mean < threshold) & (Y_eval_true >= threshold)).sum())

    if logger:
        logger.info(f"Comprehensive metrics: acc={acc:.4f}, "
                    f"r2={results.get('r2', 'N/A')}, "
                    f"FP={results['n_false_positives']}, FN={results['n_false_negatives']}")

    return results


# ---------------------------------------------------------------------------
# Lengthscale tracking (Phase 3)
# ---------------------------------------------------------------------------

def extract_lengthscales(model, model_type):
    """Extract learned ARD lengthscales from the GP kernel.

    Returns:
        dict mapping parameter names (GP_RANGE_DICT keys) to lengthscale values,
        or None if model has no lengthscales (MLP, DeepGP).
    """
    if model_type in ("mlp", "deep_gp"):
        return None

    # ExactGP, SparseGP
    try:
        kernel = model.covar_module.base_kernel
        if hasattr(kernel, "lengthscale") and kernel.lengthscale is not None:
            ls = kernel.lengthscale.detach().cpu().numpy().ravel()
            # Map to GP_RANGE_DICT keys in PARAM_ORDER ordering
            range_keys = [_PARAM_TO_RANGE_KEY[p] for p in PARAM_ORDER]
            if len(ls) == len(range_keys):
                return {range_keys[i]: float(ls[i]) for i in range(len(ls))}
            return {f"dim_{i}": float(ls[i]) for i in range(len(ls))}
    except AttributeError:
        pass
    return None


# ---------------------------------------------------------------------------
# Full true dataset evaluation (Phase 4)
# ---------------------------------------------------------------------------

def load_true_eval_dataset(eval_data_path, target="DMRD", logger=None):
    """Load a dedicated evaluation dataset (separate from training data).

    Supports ROOT files (via pmssm.load_pmssm_data logic) and CSV files.

    Args:
        eval_data_path: Path to ROOT file or CSV containing true evaluation data.
        target: Target function name (determines branch name for ROOT files).
        logger: Logger instance.

    Returns:
        X_eval: (N, 19) tensor in physical units.
        Y_eval: (N,) tensor of target values (untransformed).
    """
    eval_path = Path(eval_data_path)

    if eval_path.suffix == ".csv":
        df = pd.read_csv(eval_path)
        branch = TARGET_CONFIG[target]["branch"]
        param_cols = [p.replace("IN_", "") for p in PARAM_ORDER]
        X_eval = torch.tensor(df[param_cols].values, dtype=torch.float32)
        Y_eval = torch.tensor(df[branch].values, dtype=torch.float32)
    else:
        # ROOT file — reuse pmssm loading
        X_eval, Y_eval = pmssm.load_pmssm_data(
            data_dir=str(eval_path.parent),
            n_datasets=-1,
            logger=logger,
        )

    if logger:
        logger.info(f"Loaded true eval dataset: {len(X_eval)} points from {eval_path}")

    return X_eval, Y_eval.view(-1)


# ---------------------------------------------------------------------------
# Entropy-based batch active learning selection (Phase 6)
# ---------------------------------------------------------------------------

def select_entropy_batch(model, X_candidates_norm, n_select, model_type,
                         threshold=0.0, blur=0.15, beta=50.0,
                         tolerance_sampling=1.0, proximity_sampling=0.1,
                         use_dkl=False, jitter=1e-3, num_samples=8,
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
    from scipy.stats import qmc

    strategy = EntropySelectionStrategy(
        blur=blur, beta=beta,
        tolerance_sampling=tolerance_sampling,
        proximity_sampling=proximity_sampling,
    )

    if device is None:
        device = next(model.parameters()).device
    thr = torch.tensor([threshold], device=device)

    n_pool = 1000 if use_dkl else 10000
    is_deep = model_type == "deep_gp"
    ns = 1 if not is_deep else num_samples

    model.eval()
    if model_has_likelihood(model_type):
        model.likelihood.eval()

    if tolerance_sampling != 0:
        # Sample large initial pool with LHS, filter near threshold
        n_large = 1_000_000
        n_dim = X_candidates_norm.shape[1]
        x_large = torch.tensor(
            qmc.LatinHypercube(d=n_dim).random(n=n_large),
            dtype=torch.float32,
        ).to(device)

        if logger:
            logger.info(f"Entropy selection: evaluating {n_large} LHS candidates...")

        batch_size = 100_000
        means_list, vars_list = [], []
        for i in range(0, len(x_large), batch_size):
            x_batch = x_large[i:i + batch_size]
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

        # Filter near threshold
        mask = (mean_all > thr - tolerance_sampling) & (mean_all < thr + tolerance_sampling)

        if proximity_sampling != 0:
            proximity = torch.exp(-((mean_all[mask] - thr) ** 2) / proximity_sampling)
            entropy_score = proximity * var_all[mask]
            k = min(n_pool, int(mask.sum()))
            topk = torch.topk(entropy_score, k=k, largest=True)
            x_pool = x_large[mask][topk.indices]
        else:
            candidates_filtered = x_large[mask]
            if len(candidates_filtered) > n_pool:
                idx = torch.randperm(len(candidates_filtered), device=device)[:n_pool]
                x_pool = candidates_filtered[idx]
            else:
                x_pool = candidates_filtered

        if logger:
            logger.info(f"Focused pool: {len(x_pool)} candidates near threshold")
    else:
        n_dim = X_candidates_norm.shape[1]
        x_pool = torch.tensor(
            qmc.LatinHypercube(d=n_dim).random(n=n_pool),
            dtype=torch.float32,
        ).to(device)

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

    # Return the actual normalized points and their indices in the pool
    return x_pool[selected_indices], per_point_entropy


# ---------------------------------------------------------------------------
# Advanced plotting (Phase 7)
# ---------------------------------------------------------------------------

def plot_advanced_diagnostics(model, X_eval_norm, Y_eval_true, X_train_norm,
                              model_type, threshold, n_dim,
                              plots_dir, iteration, new_points_norm=None,
                              jitter=1e-3, num_samples=8, logger=None):
    """
    Generate advanced diagnostic plots adapted from al_pmssmwithgp.

    Produces: GP mean vs truth scatter, residuals plot.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    device = next(model.parameters()).device
    model.eval()
    X_eval_norm = X_eval_norm.to(device)
    Y_eval_true = Y_eval_true.view(-1).to(device)

    # Get predictions
    if model_type == "mlp":
        with torch.no_grad():
            mean = model(X_eval_norm).squeeze().cpu()
    elif model_type == "deep_gp":
        model.likelihood.eval()
        with torch.no_grad(), \
             gpytorch.settings.fast_pred_var(False), \
             gpytorch.settings.cholesky_jitter(jitter), \
             gpytorch.settings.num_likelihood_samples(num_samples):
            preds = model.likelihood(model(X_eval_norm))
            mean = preds.mean.detach().mean(dim=0).squeeze().cpu()
    else:
        model.likelihood.eval()
        with torch.no_grad(), \
             gpytorch.settings.fast_pred_var(), \
             gpytorch.settings.cholesky_jitter(jitter):
            preds = model.likelihood(model(X_eval_norm))
            mean = preds.mean.detach().cpu()

    y_true = Y_eval_true.cpu()

    plots_dir = Path(plots_dir) / "advanced"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. GP mean vs truth scatter with difference
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # True vs predicted
    axes[0].scatter(y_true.numpy(), mean.numpy(), alpha=0.3, s=5)
    lims = [min(y_true.min().item(), mean.min().item()),
            max(y_true.max().item(), mean.max().item())]
    axes[0].plot(lims, lims, 'r--', linewidth=1)
    axes[0].set_xlabel('True (transformed)')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title(f'Iteration {iteration}: True vs Predicted')
    axes[0].grid(True, alpha=0.3)

    # Residuals
    residuals = (mean - y_true).numpy()
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='r', linestyle='--')
    axes[1].set_xlabel('Residual (pred - true)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Residual Distribution (std={residuals.std():.4f})')
    axes[1].grid(True, alpha=0.3)

    # Residuals vs true
    axes[2].scatter(y_true.numpy(), residuals, alpha=0.3, s=5)
    axes[2].axhline(0, color='r', linestyle='--')
    axes[2].axhline(threshold, color='g', linestyle=':', label=f'threshold={threshold}')
    axes[2].set_xlabel('True (transformed)')
    axes[2].set_ylabel('Residual')
    axes[2].set_title('Residuals vs True Values')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / f'gp_diagnostics_iter{iteration:03d}.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"Advanced diagnostic plots saved to {plots_dir}")


# ---------------------------------------------------------------------------
# Diagnostic plots matching active_learning.py
# ---------------------------------------------------------------------------

def gp_predict(model, X_norm, model_type, jitter=1e-3, num_samples=8):
    """Get mean predictions from a GP/MLP model. Returns CPU tensor."""
    device = next(model.parameters()).device
    model.eval()
    if model_type == "mlp":
        with torch.no_grad():
            return model(X_norm.to(device)).squeeze().cpu()
    elif model_type == "deep_gp":
        model.likelihood.eval()
        with torch.no_grad(), \
             gpytorch.settings.fast_pred_var(False), \
             gpytorch.settings.cholesky_jitter(jitter), \
             gpytorch.settings.num_likelihood_samples(num_samples):
            preds = model.likelihood(model(X_norm.to(device)))
            return preds.mean.detach().mean(dim=0).squeeze().cpu()
    else:
        model.likelihood.eval()
        with torch.no_grad(), \
             gpytorch.settings.fast_pred_var(), \
             gpytorch.settings.cholesky_jitter(jitter):
            preds = model.likelihood(model(X_norm.to(device)))
            return preds.mean.detach().cpu()


def plot_gp_losses(train_losses, val_losses, model_type, plot_dir):
    """Rolling-average loss curves matching pmssm.plot_losses()."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    def rolling_average(x, window=30):
        x = np.asarray(x)
        return np.convolve(x, np.ones(window) / window, mode="valid")

    plt.figure()
    plt.plot(rolling_average(train_losses), label="Train loss")
    plt.plot(rolling_average(val_losses), label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"{model_type} Training for pMSSM Relic Density")
    plt.yscale('log')
    plt.grid(True, which='major', alpha=0.3)
    plt.grid(True, which='minor', alpha=0.15)
    plt.savefig(f"{plot_dir}/losses_{model_type}.png", dpi=150, bbox_inches='tight')
    plt.close()


def scatter_true_vs_pred_gp(y_true_phys, y_pred_phys, mode, model_type, plot_dir):
    """Scatter plot of true vs predicted in physical space, matching pmssm.scatter_true_vs_pred()."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    title = "Validation set" if mode == "validation" else "Training set"
    color = 'orange' if mode == 'validation' else None

    y_true = y_true_phys.numpy() if hasattr(y_true_phys, 'numpy') else y_true_phys
    y_pred = y_pred_phys.numpy() if hasattr(y_pred_phys, 'numpy') else y_pred_phys

    plt.figure()
    plt.scatter(y_true[:10_000], y_pred[:10_000], alpha=0.5, color=color)
    plt.plot(
        [min(y_true), max(y_true)],
        [min(y_true), max(y_true)],
        linestyle='--', color='grey'
    )
    plt.xlabel("True Ωh²")
    plt.ylabel("Predicted Ωh²")
    plt.title(f"True vs Predicted Ωh² ({title})")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{model_type}_true_vs_pred_{mode}.png",
                dpi=150, bbox_inches='tight')
    plt.close()


def hist_true_vs_pred_gp(y_true_phys, y_pred_phys, mode, model_type, plot_dir):
    """2D histogram of true vs predicted in physical space, matching pmssm.hist_true_vs_pred()."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import numpy as np

    title = "Validation set" if mode == "validation" else "Training set"

    y_true = np.asarray(
        y_true_phys.numpy() if hasattr(y_true_phys, 'numpy') else y_true_phys,
        dtype=np.float64
    ).reshape(-1)
    y_pred = np.asarray(
        y_pred_phys.numpy() if hasattr(y_pred_phys, 'numpy') else y_pred_phys,
        dtype=np.float64
    ).reshape(-1)

    plt.figure()
    plt.hist2d(y_true, y_pred, bins=30, cmap="inferno", norm=LogNorm())
    plt.colorbar(label="Counts (log scale)")

    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    plt.plot([vmin, vmax], [vmin, vmax], linestyle="--", color="white")

    plt.xlabel("True Ωh²")
    plt.ylabel("Predicted Ωh²")
    plt.title(f"True vs Predicted Ωh² ({model_type} {title})")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{model_type}_hist_true_vs_pred_{mode}.png",
                dpi=150, bbox_inches='tight')
    plt.close()


def compare_random_predictions_gp(y_true_phys, y_pred_phys, mode, model_type,
                                   n_points=3, logger=None):
    """Log random predictions vs truth, matching pmssm.compare_random_predictions()."""
    import random as _random

    if logger is None:
        return

    label = "validation" if mode == "validation" else "training"
    logger.info("")
    logger.info(f"{model_type}: Comparison on random {label} points:")
    logger.info("-" * 60)
    logger.info(f"{'Index':>6} | {'True Ωh²':>12} | {'Predicted Ωh²':>15}")
    logger.info("-" * 60)

    n = len(y_true_phys)
    indices = _random.sample(range(n), min(n_points, n))
    for idx in indices:
        yt = y_true_phys[idx].item() if hasattr(y_true_phys[idx], 'item') else float(y_true_phys[idx])
        yp = y_pred_phys[idx].item() if hasattr(y_pred_phys[idx], 'item') else float(y_pred_phys[idx])
        logger.info(f"{idx:6d} | {yt:12.6f} | {yp:15.6f}")
    logger.info("-" * 60)


# ---------------------------------------------------------------------------
# YAML config + parameter sweep (Phase 8)
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


# ---------------------------------------------------------------------------
# Training worker (for parallel or sequential training)
# ---------------------------------------------------------------------------

def train_gp_worker(gpu_id, X, Y, X_val, Y_val, data_min, data_max,
                    model_type, n_dim, gp_kwargs,
                    lr, iters, batch_size, jitter,
                    result_queue, model_name="model",
                    log_dir=None, plots_dir=None,
                    checkpoint_path=None, num_samples=8,
                    warm_start_path=None, target="DMRD",
                    patience=None):
    """
    Worker function for GP training (analogous to train_model_worker).

    Args:
        gpu_id: GPU device ID (int) or "cpu". Used to pin the worker to a
                specific GPU when running in parallel (spawn) mode.
        X, Y: Raw training data tensors (non-normalized)
        X_val, Y_val: Raw validation data tensors (non-normalized)
        data_min, data_max: Normalization tensors
        model_type: "exact_gp", "deep_gp", "sparse_gp", or "mlp"
        n_dim: Number of input dimensions
        gp_kwargs: Dict of GP-specific arguments (kernel, lengthscale, noise, etc.)
        lr: Learning rate
        iters: Training iterations
        batch_size: Batch size (DeepGP only)
        jitter: Cholesky jitter
        result_queue: multiprocessing.Queue to return results
        model_name: Name identifier (e.g., "AL", "Baseline")
        log_dir: Directory for log files
        plots_dir: Directory for diagnostic plots
        checkpoint_path: Path to save model state dict after training
        num_samples: Number of samples for DeepGP
        warm_start_path: Path to a previously saved checkpoint to warm-start from.
        patience: Early stopping patience. None disables early stopping.
                         If provided and the file exists, model + likelihood state
                         dicts are loaded before training begins.
    """
    # Resolve device string (same pattern as active_learning.py).
    # Also call set_device so that torch.device("cuda") inside the GP model
    # constructor resolves to the correct GPU (ExactGP/DeepGP auto-detect via
    # torch.device("cuda"), which always returns the default CUDA device).
    device = f"cuda:{gpu_id}" if isinstance(gpu_id, int) else gpu_id
    if isinstance(gpu_id, int) and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    # Set up logging
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{model_name.lower()}_training.log"
        logger = setup_worker_logging(log_file, model_name)
    else:
        logger = logging.getLogger(model_name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(
                f'%(asctime)s [{model_name}] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
            ))
            logger.addHandler(handler)

    # Route submodule loggers (gp_pipeline.*) through the same handlers
    gp_logger = logging.getLogger("gp_pipeline")
    gp_logger.setLevel(logging.INFO)
    gp_logger.handlers = []
    for h in logger.handlers:
        gp_logger.addHandler(h)
    gp_logger.propagate = False

    # Normalize data
    x_train_norm = normalize_x(X, data_min, data_max)
    x_val_norm = normalize_x(X_val, data_min, data_max)
    y_train_t = transform_y(Y, target=target).view(-1)
    y_val_t = transform_y(Y_val, target=target).view(-1)

    logger.info(f"Training set size: {len(x_train_norm)}, Validation set size: {len(x_val_norm)}")
    logger.info(f"Model type: {model_type}")

    # Create model (pinned to the requested device)
    model = create_gp_model(
        model_type, x_train_norm, y_train_t, x_val_norm, y_val_t,
        n_dim=n_dim, num_samples=num_samples, device=device,
        target=target, **gp_kwargs
    )

    # Warm-start: load previous iteration's state dict if provided
    if warm_start_path is not None and Path(warm_start_path).exists():
        ckpt = torch.load(warm_start_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if model_has_likelihood(model_type):
            model.likelihood.load_state_dict(ckpt['likelihood_state_dict'])
        logger.info(f"Warm-started from checkpoint: {warm_start_path}")
    elif warm_start_path is not None:
        logger.warning(f"Warm-start checkpoint not found: {warm_start_path}, training from scratch")

    # Train
    es_msg = f" (early stopping patience={patience})" if patience is not None else ""
    logger.info(f"Starting GP training for up to {iters} iterations{es_msg}...")
    model, train_losses, val_losses = train_gp_model(
        model, model_type, lr=lr, iters=iters, batch_size=batch_size, jitter=jitter,
        patience=patience
    )
    logger.info(f"GP training complete after {len(train_losses)} iterations.")

    # Compute metrics
    best_train_loss = min(train_losses)
    best_val_loss = min(val_losses)
    r2 = compute_gp_r2(model, x_val_norm, y_val_t, model_type, jitter=jitter,
                       num_samples=num_samples)

    logger.info(f"Best train loss: {best_train_loss:.6f}")
    logger.info(f"Best val loss: {best_val_loss:.6f}")
    logger.info(f"R² score: {r2:.4f}")

    # Extract lengthscales
    lengthscales = extract_lengthscales(model, model_type)
    if lengthscales:
        logger.info(f"Learned lengthscales: {lengthscales}")

    # Generate diagnostic plots
    if plots_dir is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plots_dir = Path(plots_dir) / model_name.lower()
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Get predictions for both train and validation sets
        y_pred_val = gp_predict(model, x_val_norm, model_type,
                                jitter=jitter, num_samples=num_samples)
        y_pred_train = gp_predict(model, x_train_norm, model_type,
                                  jitter=jitter, num_samples=num_samples)

        # # Combined diagnostics plot (loss curves + val scatter in transformed space)
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # ax1.plot(train_losses, label='Train')
        # ax1.plot(val_losses, label='Validation')
        # ax1.set_xlabel('Iteration')
        # ax1.set_ylabel('Loss')
        # ax1.set_title(f'{model_name} Loss Curves')
        # ax1.legend()
        # ax1.grid(True, alpha=0.3)

        # ax2.scatter(y_val_t.cpu().numpy(), y_pred_val.numpy(), alpha=0.5, s=10)
        # lims = [min(y_val_t.min().item(), y_pred_val.min().item()),
        #         max(y_val_t.max().item(), y_pred_val.max().item())]
        # ax2.plot(lims, lims, 'r--', linewidth=1)
        # ax2.set_xlabel('True (transformed)')
        # ax2.set_ylabel('Predicted')
        # ax2.set_title(f'{model_name} True vs Predicted (R²={r2:.4f})')
        # ax2.grid(True, alpha=0.3)

        # plt.tight_layout()
        # plt.savefig(plots_dir / 'diagnostics.png', dpi=150, bbox_inches='tight')
        # plt.close()

        # --- Plots matching active_learning.py ---
        # Rolling average loss curve (separate file)
        plot_gp_losses(train_losses, val_losses, model_type, str(plots_dir))

        # Convert to physical space for scatter / histogram plots
        y_true_val_phys = inverse_transform_y(y_val_t.cpu(), target=target)
        y_pred_val_phys = inverse_transform_y(y_pred_val, target=target)
        y_true_train_phys = inverse_transform_y(y_train_t.cpu(), target=target)
        y_pred_train_phys = inverse_transform_y(y_pred_train, target=target)

        # Scatter: true vs predicted (train & validation)
        scatter_true_vs_pred_gp(
            y_true_train_phys, y_pred_train_phys, "train", model_type, str(plots_dir))
        scatter_true_vs_pred_gp(
            y_true_val_phys, y_pred_val_phys, "validation", model_type, str(plots_dir))

        # 2D histogram: true vs predicted (train & validation)
        hist_true_vs_pred_gp(
            y_true_train_phys, y_pred_train_phys, "train", model_type, str(plots_dir))
        hist_true_vs_pred_gp(
            y_true_val_phys, y_pred_val_phys, "validation", model_type, str(plots_dir))

        # Random predictions comparison (text log)
        compare_random_predictions_gp(
            y_true_train_phys, y_pred_train_phys, "train", model_type,
            n_points=min(10, len(y_true_train_phys)), logger=logger)
        compare_random_predictions_gp(
            y_true_val_phys, y_pred_val_phys, "validation", model_type,
            n_points=min(3, len(y_true_val_phys)), logger=logger)

        logger.info(f"Diagnostic plots saved to {plots_dir}")

    # Save model checkpoint
    if checkpoint_path is not None:
        ckpt_data = {'model_state_dict': model.state_dict()}
        if model_has_likelihood(model_type):
            ckpt_data['likelihood_state_dict'] = model.likelihood.state_dict()
        torch.save(ckpt_data, checkpoint_path)
        logger.info(f"Saved model checkpoint to {checkpoint_path}")

    # Return results via queue
    result_queue.put({
        "model_name": model_name,
        "best_train_loss": best_train_loss,
        "best_val_loss": best_val_loss,
        "r2_score": r2,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "lengthscales": lengthscales,
    })


# ---------------------------------------------------------------------------
# CLI and main loop
# ---------------------------------------------------------------------------

@click.command()
@click.option('--testing', is_flag=True, help="Run in testing mode (small data, few iterations).")
@click.option('--n-iterations', default=1, type=int, help="Number of active learning iterations.")
@click.option('--n-candidates', default=1000, type=int, help="Candidate pool size.")
@click.option('--n-select', default=10, type=int, help="Number of points to select per iteration.")
@click.option('--n-datasets', default=None, type=int, help="Number of ROOT datasets to load.")
@click.option('--n-samples', default=None, type=int, help="Number of samples to use from data.")
@click.option('--output-dir', default='active_learning_gp_output', type=str, help="Output directory.")
@click.option('--generate-data', is_flag=True, help="Generate new models using Run3ModelGen.")
@click.option('--min-gen-fraction', default=0.6, type=float, help="Minimum fraction of n-select that must be generated successfully.")
@click.option('--max-gen-attempts', default=10, type=int, help="Maximum generation attempts per iteration.")
@click.option('--gen-workers', default=1, type=int, help="Number of parallel genModels.py workers.")
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
@click.option('--training-iterations', default=2000, type=int, help="Max training iterations per AL iteration (default: 2000).")
@click.option('--early-stopping/--no-early-stopping', default=True, help="Enable early stopping on validation loss.")
@click.option('--patience', default=200, type=int, help="Early stopping patience (iterations without improvement).")
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
              help="Point selection strategy: top_k (variance) or entropy_batch.")
@click.option('--entropy-blur', default=0.15, type=float, help="Entropy smoothing (entropy_batch only).")
@click.option('--entropy-beta', default=50.0, type=float, help="Gibbs sampling temperature (entropy_batch only).")
@click.option('--tolerance-sampling', default=1.0, type=float, help="Threshold filter width (entropy_batch only).")
@click.option('--proximity-sampling', default=0.1, type=float, help="Proximity weighting width (entropy_batch only).")
# Evaluation options
@click.option('--compute-full-metrics/--no-full-metrics', default=False,
              help="Compute comprehensive GoF metrics (accuracy, chi2, pulls, etc.).")
@click.option('--eval-data-path', default=None, type=str, help="Path to true eval dataset (ROOT/CSV).")
@click.option('--track-lengthscales/--no-track-lengthscales', default=True,
              help="Save learned ARD lengthscales per iteration.")
@click.option('--advanced-plots/--no-advanced-plots', default=False,
              help="Generate advanced diagnostic plots (heatmaps, residuals).")
# Config file / sweep options
@click.option('--config-file', default=None, type=str, help="YAML config file (overrides CLI args).")
@click.option('--sweep-index', default=None, type=int, help="Sweep combination index (requires --config-file).")
def main(testing, n_iterations, n_candidates, n_select, n_datasets, n_samples,
         output_dir, generate_data, min_gen_fraction, max_gen_attempts, gen_workers,
         target, model_type, kernel, lengthscale, noise, jitter, learning_rate,
         training_iterations, early_stopping, patience,
         use_ard, use_dkl, feature_dim,
         num_hidden_dims, num_middle_dims, num_inducing_max, inducing_strategy,
         gp_num_samples, batch_size, warm_starting, m_nu, num_mixtures,
         selection_strategy, entropy_blur, entropy_beta,
         tolerance_sampling, proximity_sampling,
         compute_full_metrics, eval_data_path, track_lengthscales, advanced_plots,
         config_file, sweep_index):
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
            'learning_rate': 'learning_rate', 'training_iterations': 'training_iterations',
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
        training_iterations = int(_locals.get('training_iterations', training_iterations))
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
        training_iterations = 50
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
    logger.info(f"  selection_strategy: {selection_strategy}")
    logger.info(f"  training_iterations: {training_iterations}")
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

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
    X, Y = pmssm.load_pmssm_data(n_datasets=n_datasets, logger=logger,
                                  plot_dir=str(plots_dir))

    # Store full dataset for baseline random sampling
    X_full, Y_full = X.clone(), Y.clone()

    initial_al_size = n_samples if n_samples is not None else len(X)
    initial_al_indices = torch.arange(initial_al_size)

    if n_samples is not None:
        X = X[:n_samples]
        Y = Y[:n_samples]
        logger.info(f"Using first {n_samples} samples for initial AL training")

    logger.info(f"AL dataset shape: X={X.shape}, Y={Y.shape}")
    logger.info(f"Baseline pool shape: X_full={X_full.shape}, Y_full={Y_full.shape}")

    # Determine if we can train AL and Baseline in parallel (2+ GPUs)
    AL_GPU_ID = 0
    BASELINE_GPU_ID = 1
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

    al_train_losses, al_val_losses, al_r2_scores = [], [], []
    al_n_train, al_n_val = [], []
    baseline_train_losses, baseline_val_losses, baseline_r2_scores = [], [], []
    baseline_n_train, baseline_n_val = [], []

    # Lengthscale tracking
    lengthscale_rows = []

    # Full eval dataset (loaded lazily on first use)
    X_eval_full, Y_eval_full = None, None
    eval_r2_scores = []

    # Previous model state for warm starting
    prev_al_checkpoint = None
    prev_baseline_checkpoint = None

    for iteration in range(1, n_iterations + 1):
        logger.info(f"=== GP Active Learning Iteration {iteration} ===")

        iter_dir = output_dir / f"iteration_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        iter_plots_dir = iter_dir / "plots"
        iter_plots_dir.mkdir(parents=True, exist_ok=True)

        # ---- Build baseline dataset ----
        if iteration == 1:
            X_baseline = X.clone()
            Y_baseline = Y.clone()
            logger.info(f"Iteration 1: Both models use identical dataset "
                       f"({len(X_baseline)} samples)")
        else:
            n_current = len(X)
            all_indices = torch.arange(len(X_full))
            mask = torch.ones(len(X_full), dtype=torch.bool)
            mask[initial_al_indices] = False
            available_indices = all_indices[mask]

            n_additional = n_current - len(initial_al_indices)
            if n_additional <= len(available_indices):
                additional_indices = available_indices[
                    torch.randperm(len(available_indices))[:n_additional]
                ]
            else:
                logger.info(f"Baseline: sampling with replacement "
                           f"({n_additional} needed, {len(available_indices)} available)")
                additional_indices = available_indices[
                    torch.randint(0, len(available_indices), (n_additional,))
                ]

            baseline_indices = torch.cat([initial_al_indices, additional_indices])
            X_baseline = X_full[baseline_indices]
            Y_baseline = Y_full[baseline_indices]
            logger.info(f"Baseline dataset: {len(initial_al_indices)} initial "
                       f"+ {n_additional} random = {len(baseline_indices)} samples")

        # ---- Train/val split ----
        n_total = len(X)
        n_val = min(int(0.2 * n_total), 5000)  # 20% for validation, capped at 5000
        n_train = n_total - n_val
        perm = torch.randperm(n_total)
        idx_train = perm[:n_train]
        idx_val = perm[n_train:]

        X_train_al = X[idx_train]
        Y_train_al = Y[idx_train]
        X_val_al = X[idx_val]
        Y_val_al = Y[idx_val]

        # Same split indices for baseline (ensures same sizes)
        n_total_base = len(X_baseline)
        n_val_base = min(int(0.2 * n_total_base), 5000)
        n_train_base = n_total_base - n_val_base
        perm_base = torch.randperm(n_total_base)
        X_train_base = X_baseline[perm_base[:n_train_base]]
        Y_train_base = Y_baseline[perm_base[:n_train_base]]
        X_val_base = X_baseline[perm_base[n_train_base:]]
        Y_val_base = Y_baseline[perm_base[n_train_base:]]

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
                      learning_rate, training_iterations, batch_size, jitter,
                      al_queue, "AL", iter_dir, iter_plots_dir, al_checkpoint_path,
                      gp_num_samples, al_warm_start, target, effective_patience),
            )
            baseline_process = mp.Process(
                target=train_gp_worker,
                args=(BASELINE_GPU_ID, X_train_base, Y_train_base, X_val_base, Y_val_base,
                      data_min, data_max, model_type, len(PARAM_ORDER), gp_kwargs,
                      learning_rate, training_iterations, batch_size, jitter,
                      baseline_queue, "Baseline", iter_dir, iter_plots_dir,
                      baseline_checkpoint_path,
                      gp_num_samples, baseline_warm_start, target, effective_patience),
            )

            logger.info(f"Launching parallel training: AL on cuda:{AL_GPU_ID}, "
                        f"Baseline on cuda:{BASELINE_GPU_ID}")
            al_process.start()
            baseline_process.start()
            al_process.join()
            baseline_process.join()

            al_results = al_queue.get()
            baseline_results = baseline_queue.get()
        else:
            # ---- Train AL and Baseline sequentially ----
            al_warm_start = prev_al_checkpoint if warm_starting else None
            baseline_warm_start = prev_baseline_checkpoint if warm_starting else None
            logger.info(f"Training AL {model_type} model ({n_train} train, {n_val} val)...")
            al_queue = mp.Queue()
            train_gp_worker(
                device, X_train_al, Y_train_al, X_val_al, Y_val_al,
                data_min, data_max, model_type, len(PARAM_ORDER), gp_kwargs,
                learning_rate, training_iterations, batch_size, jitter,
                al_queue, "AL", iter_dir, iter_plots_dir, al_checkpoint_path,
                num_samples=gp_num_samples, warm_start_path=al_warm_start,
                target=target, patience=effective_patience,
            )
            al_results = al_queue.get()

            logger.info(f"Training Baseline {model_type} model ({n_train_base} train, {n_val_base} val)...")
            baseline_queue = mp.Queue()
            train_gp_worker(
                device, X_train_base, Y_train_base, X_val_base, Y_val_base,
                data_min, data_max, model_type, len(PARAM_ORDER), gp_kwargs,
                learning_rate, training_iterations, batch_size, jitter,
                baseline_queue, "Baseline", iter_dir, iter_plots_dir,
                baseline_checkpoint_path,
                num_samples=gp_num_samples, warm_start_path=baseline_warm_start,
                target=target, patience=effective_patience,
            )
            baseline_results = baseline_queue.get()

        # ---- Log results ----
        logger.info(f"AL metrics: train_loss={al_results['best_train_loss']:.6f}, "
                   f"val_loss={al_results['best_val_loss']:.6f}, "
                   f"R²={al_results['r2_score']:.4f}")
        logger.info(f"Baseline metrics: train_loss={baseline_results['best_train_loss']:.6f}, "
                   f"val_loss={baseline_results['best_val_loss']:.6f}, "
                   f"R²={baseline_results['r2_score']:.4f}")

        # Track metrics
        iteration_numbers.append(iteration)
        al_train_losses.append(al_results['best_train_loss'])
        al_val_losses.append(al_results['best_val_loss'])
        al_r2_scores.append(al_results['r2_score'])
        al_n_train.append(n_train)
        al_n_val.append(n_val)
        baseline_train_losses.append(baseline_results['best_train_loss'])
        baseline_val_losses.append(baseline_results['best_val_loss'])
        baseline_r2_scores.append(baseline_results['r2_score'])
        baseline_n_train.append(n_train_base)
        baseline_n_val.append(n_val_base)

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
            target=target, **gp_kwargs
        )
        checkpoint = torch.load(al_checkpoint_path, map_location=device)
        al_model.load_state_dict(checkpoint['model_state_dict'])
        if model_has_likelihood(model_type):
            al_model.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        logger.info("AL model reloaded for uncertainty estimation")

        # ---- Full true dataset evaluation ----
        if eval_data_path is not None:
            if X_eval_full is None:
                X_eval_full, Y_eval_full = load_true_eval_dataset(
                    eval_data_path, target=target, logger=logger
                )
            x_eval_norm = normalize_x(X_eval_full, data_min, data_max)
            y_eval_t = transform_y(Y_eval_full, target=target).view(-1)
            eval_r2 = compute_gp_r2(al_model, x_eval_norm, y_eval_t, model_type,
                                    jitter=jitter, num_samples=gp_num_samples)
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
        if selection_strategy == "entropy_batch" and model_has_likelihood(model_type):
            logger.info("Using entropy-based batch selection...")
            candidates_norm = normalize_x(
                generate_candidate_pool(n_candidates, seed=iteration),
                data_min, data_max,
            )
            selected_points_norm, per_point_entropy = select_entropy_batch(
                al_model, candidates_norm, n_select, model_type,
                threshold=threshold, blur=entropy_blur, beta=entropy_beta,
                tolerance_sampling=tolerance_sampling,
                proximity_sampling=proximity_sampling,
                use_dkl=use_dkl, jitter=jitter, num_samples=gp_num_samples,
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
                logger.info(f"Total generated: {len(new_X)} valid training points")
                all_selected_points[-1]["n_generated"] = len(new_X)
            else:
                logger.warning("No valid models generated after all attempts")

        # ---- Augment training data ----
        if new_X is not None and new_Y is not None and len(new_X) > 0:
            # Filter new data too (Y > 0)
            new_valid = (new_Y.squeeze(-1) > 0)
            new_X = new_X[new_valid]
            new_Y = new_Y[new_valid]
            logger.info(f"Augmenting: {len(X)} + {len(new_X)} = "
                       f"{len(X) + len(new_X)} samples")
            X = torch.cat([X, new_X], dim=0)
            Y = torch.cat([Y, new_Y], dim=0)

        prev_al_checkpoint = al_checkpoint_path
        prev_baseline_checkpoint = baseline_checkpoint_path

    # ---- Plot iteration metrics ----
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
        plot_iteration_metrics(iteration_numbers, al_metrics, baseline_metrics,
                              output_dir, logger)
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
            "training_iterations": training_iterations,
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
