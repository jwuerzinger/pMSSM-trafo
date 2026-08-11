"""
Evaluation metrics and model assessment for pMSSM models.

This module provides functions to:
- Compute R² scores for GP and transformer models
- Calculate comprehensive evaluation metrics (accuracy, GoF, pull statistics)
- Extract learned lengthscales from GP kernels
- Compare predictions on random validation points
"""

import random
import torch
import torch.nn as nn
import gpytorch

from .config import PARAM_ORDER, TARGET_CONFIG
from .data import inverse_transform_y


# ===== Transformer Model Utilities =====

def is_transformer(model: nn.Module) -> bool:
    """Check if model is a transformer architecture."""
    return any(
        isinstance(m, (nn.MultiheadAttention, nn.TransformerEncoderLayer))
        for m in model.modules()
    )


def get_model_name(model: nn.Module) -> str:
    """Get a clean name for the model for use in plot filenames."""
    class_name = model.__class__.__name__
    if class_name == "PMSSMTransformer":
        return "transformer"
    elif class_name == "PMSSMTransformerTabular":
        return "transformer_tabular"
    elif class_name == "PMSSMFeedForward":
        return "MLP"
    else:
        # Fallback for unknown models
        return "transformer" if is_transformer(model) else "MLP"


# ===== Transformer Model Evaluation =====

def compare_random_predictions(model, stats, subset, mode='validation', device="cpu", n_points=3, logger=None):
    """
    Print comparison of true vs predicted values on random validation/training points.

    Args:
        model: Trained transformer model
        stats: Tuple of (mean_X, std_X, mean_Y, std_Y) for denormalization
        subset: PyTorch dataset (validation or training subset)
        mode: 'validation' or 'train'
        device: Compute device
        n_points: Number of random points to sample
        logger: Logger instance
    """
    model.eval()
    model.to(device)

    if mode == 'validation':
        logger.info("")
        logger.info(f"{get_model_name(model)}: Comparison on random validation points:")
    elif mode == 'train':
        logger.info("")
        logger.info(f"{get_model_name(model)}: Comparison on random training points:")
    else:
        raise ValueError("Unsupported mode! Should be validation, train.")

    indices = random.sample(range(len(subset)), n_points)

    logger.info("-" * 90)
    logger.info(f"{'Index':>6} | {'True Ωh² (norm.)':>20} | {'Predicted Ωh² (norm.)':>23} | {'True Ωh²':>12} | {'Predicted Ωh²':>15}")
    logger.info("-" * 90)

    with torch.no_grad():
        for idx in indices:
            x, y_true = subset[idx]
            x = x.unsqueeze(0).to(device)      # (1, 19)
            y_pred = model(x).cpu().item()

            # Revert normalization: Y_norm = (Y - mean_Y) / std_Y
            mean_X, std_X, mean_Y, std_Y = stats
            y_true_nonorm = y_true * std_Y + mean_Y
            y_pred_nonorm = y_pred * std_Y + mean_Y

            logger.info(
                f"{idx:6d} | "
                f"{y_true.item():20.6f} | "
                f"{y_pred:23.6f} | "
                f"{y_true_nonorm.item():12.6f} | "
                f"{y_pred_nonorm.item():15.6f}"
            )

    logger.info("-" * 90)


# ===== GP Model Evaluation =====

def compute_gp_r2(model, x_val, y_val, model_type, jitter=1e-3, num_samples=8, target="DMRD"):
    """
    Compute R² score on validation set using GP model predictions.

    R² is computed in physical space (after inverse transformation) to be
    comparable with transformer pipeline metrics.

    Args:
        model: Trained GP or MLP model
        x_val: Normalized validation inputs
        y_val: True validation targets (in transformed space)
        model_type: One of "exact_gp", "deep_gp", "sparse_gp", "mlp"
        jitter: Cholesky jitter for GP inference
        num_samples: Number of likelihood samples for DeepGP
        target: Target name for inverse transformation (default: "DMRD")

    Returns:
        r2: R² score (float) computed in physical space
    """
    device = next(model.parameters()).device
    model.eval()

    # Predict in chunks. This is called on the TRAINING set as well as the
    # validation set (see train_gp_model), so an unbatched call scales with the
    # labelled set and eventually asks the GPU for a workspace it cannot serve.
    # On ROCm that surfaced as "Device memory allocation size is too small for
    # TRSM" followed by "Memory access fault by GPU", killing the training
    # worker after it had logged completion but before it wrote its checkpoint,
    # which left the parent blocked on a result that never arrived. It bit the
    # Deep GP at n_train = 12740 while 12487 had been fine, so the limit is not
    # a clean threshold. gp_predict already chunks at this size.
    #
    # Chunking is safe because a point's posterior mean depends only on the
    # (fixed) training data, not on the other points in its batch: verified
    # exact to float32 round-off (4.8e-7) for the deterministic exact-GP path.
    # For the Deep GP each chunk draws its own likelihood samples, so values are
    # not bitwise reproducible, but they were never reproducible anyway: the R2
    # it reports is a Monte-Carlo estimate that already moves by ~0.01 between
    # identical calls.
    batch = 1024

    def _predict_chunked(fn):
        out = []
        for start in range(0, len(x_val), batch):
            out.append(fn(x_val[start:start + batch].to(device)))
        return torch.cat(out, dim=0) if out else torch.empty(0)

    if model_type == "mlp":
        with torch.no_grad():
            y_pred_transformed = _predict_chunked(
                lambda xb: model(xb).squeeze(-1).view(-1).cpu())
    elif model_type == "deep_gp":
        model.likelihood.eval()
        with torch.no_grad(), \
             gpytorch.settings.fast_pred_var(False), \
             gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter), \
             gpytorch.settings.num_likelihood_samples(num_samples):
            y_pred_transformed = _predict_chunked(
                lambda xb: model.likelihood(model(xb)).mean.detach()
                .mean(dim=0).view(-1).cpu())
    else:
        # exact_gp, sparse_gp
        model.likelihood.eval()
        with torch.no_grad(), \
             gpytorch.settings.fast_pred_var(), \
             gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter):
            y_pred_transformed = _predict_chunked(
                lambda xb: model.likelihood(model(xb)).mean.detach().view(-1).cpu())

    # Convert to physical space for R² calculation
    y_true_transformed = y_val.view(-1).to(device)
    y_pred_transformed = y_pred_transformed.view(-1)

    y_true_physical = inverse_transform_y(y_true_transformed.cpu(), target=target).to(device)
    y_pred_physical = inverse_transform_y(y_pred_transformed.cpu(), target=target).to(device)

    ss_res = ((y_true_physical - y_pred_physical) ** 2).sum()
    ss_tot = ((y_true_physical - y_true_physical.mean()) ** 2).sum()
    r2 = (1 - ss_res / ss_tot).item()
    return r2


def compute_comprehensive_metrics(model, X_eval_norm, Y_eval_true, model_type,
                                  threshold=0.0, jitter=1e-3, num_samples=8,
                                  logger=None):
    """
    Compute comprehensive evaluation metrics on an evaluation dataset.

    Requires al_pmssmwithgp.gp_pipeline.utils.evaluation functions:
    - compute_accuracy
    - compute_weighted_accuracy
    - compute_gof_metrics

    Args:
        model: Trained GP or MLP model
        X_eval_norm: Normalized evaluation inputs (N, D) on model's device
        Y_eval_true: True target values in transformed space (N,)
        model_type: One of "exact_gp", "deep_gp", "sparse_gp", "mlp"
        threshold: Classification threshold in transformed space
        jitter: Cholesky jitter
        num_samples: DeepGP likelihood samples
        logger: Logger instance

    Returns:
        dict with accuracy, weighted_acc, mse, rmse, r2, chi2, chi2_red,
        mean_abs_pull, rms_pull, weighted variants, confusion matrix info
    """
    # Import al_pmssmwithgp evaluation utilities
    try:
        import sys
        from pathlib import Path
        _GP_PIPELINE_ROOT = Path(__file__).parent.parent / "al_pmssmwithgp" / "model"
        if str(_GP_PIPELINE_ROOT) not in sys.path:
            sys.path.insert(0, str(_GP_PIPELINE_ROOT))
        from gp_pipeline.utils.evaluation import (
            compute_accuracy,
            compute_weighted_accuracy,
            compute_gof_metrics,
        )
    except ImportError as e:
        if logger:
            logger.error(f"Failed to import al_pmssmwithgp evaluation utils: {e}")
            logger.error("compute_comprehensive_metrics requires al_pmssmwithgp submodule")
        return {"error": "al_pmssmwithgp not available"}

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
             gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter), \
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
             gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter):
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


# ===== GP Kernel Inspection =====

def extract_lengthscales(model, model_type):
    """
    Extract learned ARD lengthscales from the GP kernel.

    Args:
        model: Trained GP model
        model_type: One of "exact_gp", "deep_gp", "sparse_gp", "mlp"

    Returns:
        dict mapping parameter names (GP_RANGE_DICT keys) to lengthscale values,
        or None if model has no lengthscales (MLP, DeepGP)
    """
    # Mapping from PARAM_ORDER names to GP_RANGE_DICT keys
    _PARAM_TO_RANGE_KEY = {
        "IN_meL": "meL", "IN_meR": "meR", "IN_mtauL": "mtauL", "IN_mtauR": "mtauR",
        "IN_mqL1": "mqL1", "IN_muR": "muR", "IN_mdR": "mdR", "IN_mqL3": "mqL3",
        "IN_mtR": "mtR", "IN_mbR": "mbR", "IN_M_1": "M_1", "IN_M_2": "M_2",
        "IN_mu": "mu", "IN_M_3": "M_3", "IN_At": "AT", "IN_Ab": "Ab",
        "IN_Atau": "Atau", "IN_mA": "mA", "IN_tanb": "tanb",
    }

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
