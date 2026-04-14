"""
Visualization utilities for pMSSM active learning pipelines.

This module provides unified plotting functions for both transformer and GP models,
including training diagnostics, prediction comparisons, and iteration metrics.
"""

import random
from pathlib import Path
import numpy as np
import torch
import gpytorch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .config import PARAM_ORDER, PARAM_RANGES, TARGET_CONFIG


# ===== Utilities =====

def running_in_notebook():
    """Check if code is running in a Jupyter notebook."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        return ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def gp_predict(model, X_norm, model_type, jitter=1e-3, num_samples=8):
    """
    Get mean predictions from a GP/MLP model.

    Args:
        model: Trained GP or MLP model
        X_norm: Normalized input tensor
        model_type: One of "exact_gp", "deep_gp", "sparse_gp", "mlp"
        jitter: Cholesky jitter for GP inference
        num_samples: Number of likelihood samples for DeepGP

    Returns:
        Mean predictions as CPU tensor
    """
    device = next(model.parameters()).device
    model.eval()

    if model_type == "mlp":
        with torch.no_grad():
            return model(X_norm.to(device)).squeeze().cpu()
    elif model_type == "deep_gp":
        model.likelihood.eval()
        batch_size = 1024
        all_means = []
        with torch.no_grad(), \
             gpytorch.settings.fast_pred_var(False), \
             gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter), \
             gpytorch.settings.num_likelihood_samples(num_samples):
            for start in range(0, len(X_norm), batch_size):
                x_batch = X_norm[start:start + batch_size].to(device)
                preds = model.likelihood(model(x_batch))
                all_means.append(preds.mean.detach().mean(dim=0).view(-1).cpu())
            return torch.cat(all_means, dim=0)
    else:
        # exact_gp, sparse_gp
        model.likelihood.eval()
        batch_size = 1024
        all_means = []
        with torch.no_grad(), \
             gpytorch.settings.fast_pred_var(), \
             gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter):
            for start in range(0, len(X_norm), batch_size):
                x_batch = X_norm[start:start + batch_size].to(device)
                preds = model.likelihood(model(x_batch))
                all_means.append(preds.mean.detach().view(-1).cpu())
            return torch.cat(all_means, dim=0)


# ===== Training Diagnostics =====

def plot_losses(train_losses, val_losses, model_name, plot_dir="plots"):
    """
    Plot rolling-average training and validation losses.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        model_name: Model name string for plot title and filename
        plot_dir: Directory to save plot
    """
    def rolling_average(x, window=30):
        x = np.asarray(x)
        return np.convolve(x, np.ones(window) / window, mode="valid")

    plt.figure()
    plt.plot(rolling_average(train_losses), label="Train loss")
    plt.plot(rolling_average(val_losses), label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"{model_name} Training for pMSSM Relic Density")
    plt.yscale('log')
    plt.grid(True, which='major', alpha=0.3)
    plt.grid(True, which='minor', alpha=0.15)

    if not running_in_notebook():
        plt.savefig(f"{plot_dir}/losses_{model_name}.png", dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ===== Prediction Comparisons =====

def scatter_true_vs_pred(y_true, y_pred, mode, model_name, plot_dir="plots"):
    """
    Scatter plot of true vs predicted values.

    Args:
        y_true: True values (array-like) in physical units
        y_pred: Predicted values (array-like) in physical units
        mode: 'validation' or 'train'
        model_name: Model name string for filename
        plot_dir: Directory to save plot
    """
    title = "Validation set" if mode == "validation" else "Training set"
    color = 'orange' if mode == 'validation' else None

    # Convert to numpy if tensor
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()

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

    if not running_in_notebook():
        plt.savefig(f"{plot_dir}/{model_name}_true_vs_pred_{mode}.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def hist_true_vs_pred(y_true, y_pred, mode, model_name, plot_dir="plots"):
    """
    2D histogram of true vs predicted values.

    Args:
        y_true: True values (array-like) in physical units
        y_pred: Predicted values (array-like) in physical units
        mode: 'validation' or 'train'
        model_name: Model name string for filename
        plot_dir: Directory to save plot
    """
    title = "Validation set" if mode == "validation" else "Training set"

    # Convert to numpy if tensor
    y_true_arr = np.asarray(
        y_true.numpy() if hasattr(y_true, 'numpy') else y_true,
        dtype=np.float64
    ).reshape(-1)
    y_pred_arr = np.asarray(
        y_pred.numpy() if hasattr(y_pred, 'numpy') else y_pred,
        dtype=np.float64
    ).reshape(-1)

    plt.figure()
    plt.hist2d(y_true_arr, y_pred_arr, bins=30, cmap="inferno", norm=LogNorm())
    plt.colorbar(label="Counts (log scale)")

    # y = x reference line
    vmin = min(y_true_arr.min(), y_pred_arr.min())
    vmax = max(y_true_arr.max(), y_pred_arr.max())
    plt.plot([vmin, vmax], [vmin, vmax], linestyle="--", color="white")

    plt.xlabel("True Ωh²")
    plt.ylabel("Predicted Ωh²")
    plt.title(f"True vs Predicted Ωh² ({model_name} {title})")
    plt.tight_layout()

    if not running_in_notebook():
        plt.savefig(f"{plot_dir}/{model_name}_hist_true_vs_pred_{mode}.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compare_random_predictions(y_true, y_pred, mode, model_name, n_points=3, logger=None):
    """
    Log random predictions vs truth for visual inspection.

    Args:
        y_true: True values (array-like) in physical units
        y_pred: Predicted values (array-like) in physical units
        mode: 'validation' or 'train'
        model_name: Model name string
        n_points: Number of random points to sample
        logger: Logger instance
    """
    if logger is None:
        return

    label = "validation" if mode == "validation" else "training"
    logger.info("")
    logger.info(f"{model_name}: Comparison on random {label} points:")
    logger.info("-" * 60)
    logger.info(f"{'Index':>6} | {'True Ωh²':>12} | {'Predicted Ωh²':>15}")
    logger.info("-" * 60)

    n = len(y_true)
    indices = random.sample(range(n), min(n_points, n))

    for idx in indices:
        yt = y_true[idx].item() if hasattr(y_true[idx], 'item') else float(y_true[idx])
        yp = y_pred[idx].item() if hasattr(y_pred[idx], 'item') else float(y_pred[idx])
        logger.info(f"{idx:6d} | {yt:12.6f} | {yp:15.6f}")

    logger.info("-" * 60)


# ===== Data Distribution Plots =====

def plot_data_histograms(X, Y, idx_train, idx_val, output_dir, model_name, iteration, logger,
                         fixed_axes=False,
                         reference_X=None, reference_Y=None, reference_label="Reference"):
    """
    Plot histograms of all input parameters and target for training and validation sets.

    Works for both transformer and GP pipelines - expects raw physical units.

    Args:
        X: Full input tensor (N, 19) in physical units
        Y: Full target tensor (N, 1) or (N,) in physical units
        idx_train: Training indices
        idx_val: Validation indices
        output_dir: Directory to save plots
        model_name: Name identifier (e.g., "AL", "Baseline")
        iteration: Current iteration number
        logger: Logger instance
        fixed_axes: If True, fix x-axis ranges (from PARAM_RANGES / [0,1] for target)
                    and use fixed bin edges so plots are comparable across iterations.
        reference_X: Optional reference input data (M, 19) for overlay (e.g., MCMC dataset)
        reference_Y: Optional reference target data (M, 1) or (M,) for overlay
        reference_label: Label for the reference data in legends
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to numpy
    X_train = X[idx_train].numpy() if hasattr(X, 'numpy') else X[idx_train]
    X_val = X[idx_val].numpy() if hasattr(X, 'numpy') else X[idx_val]
    Y_train = Y[idx_train].reshape(-1).numpy() if hasattr(Y, 'numpy') else np.asarray(Y[idx_train]).reshape(-1)
    Y_val = Y[idx_val].reshape(-1).numpy() if hasattr(Y, 'numpy') else np.asarray(Y[idx_val]).reshape(-1)

    # Convert reference data if provided
    X_ref = None
    Y_ref = None
    if reference_X is not None:
        X_ref = reference_X.numpy() if hasattr(reference_X, 'numpy') else np.asarray(reference_X)
    if reference_Y is not None:
        Y_ref = reference_Y.reshape(-1).numpy() if hasattr(reference_Y, 'numpy') else np.asarray(reference_Y).reshape(-1)

    # Plot input parameters
    param_names = [p.replace("IN_", "") for p in PARAM_ORDER]
    n_params = len(param_names)

    # Create grid for input parameters (5 columns)
    n_cols = 5
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    for i, param_name in enumerate(param_names):
        ax = axes[i]

        # Determine bins and x-limits
        full_key = PARAM_ORDER[i]  # e.g. "IN_meL"
        lo, hi = PARAM_RANGES[full_key]
        if fixed_axes and lo < hi:
            bins = np.linspace(lo, hi, 31)
        else:
            bins = 30

        # Plot histograms
        ax.hist(X_train[:, i], bins=bins, alpha=0.5, label='Train', color='blue', density=True)
        ax.hist(X_val[:, i], bins=bins, alpha=0.5, label='Val', color='orange', density=True)
        if X_ref is not None:
            ax.hist(X_ref[:, i], bins=bins, alpha=0.35, label=reference_label, color='green',
                    density=True, histtype='step', linewidth=2)

        if fixed_axes and lo < hi:
            ax.set_xlim(lo, hi)

        ax.set_xlabel(param_name, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{param_name}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plot_path = output_dir / f'{model_name.lower()}_input_histograms_iter{iteration:03d}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Plot target (MO_Omega) with log Y-axis
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    target_bins = np.linspace(0, 1, 31) if fixed_axes else 30
    ax.hist(Y_train, bins=target_bins, alpha=0.5, label='Train', color='blue', density=True)
    ax.hist(Y_val, bins=target_bins, alpha=0.5, label='Val', color='orange', density=True)
    if Y_ref is not None:
        ax.hist(Y_ref, bins=target_bins, alpha=0.35, label=reference_label, color='green',
                density=True, histtype='step', linewidth=2)
    ax.axvline(TARGET_CONFIG["DMRD"]["true_value"], color='red', linestyle='--',
               linewidth=1.5, label=f'Ωh² = {TARGET_CONFIG["DMRD"]["true_value"]}')
    if fixed_axes:
        ax.set_xlim(0, 1)
    ax.set_xlabel('MO_Omega (Ωh²)', fontsize=12)
    ax.set_ylabel('Density (log scale)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title(f'{model_name} Target Distribution - Iteration {iteration}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plot_path = output_dir / f'{model_name.lower()}_target_histogram_iter{iteration:03d}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"{model_name} histograms saved to {output_dir}")


def plot_parallel_coordinates(X, idx_train, idx_val, output_dir, model_name, iteration, logger,
                              max_lines=300):
    """
    Plot parallel coordinate chart for input parameters (train vs validation).

    Only non-fixed parameters (where lo < hi in PARAM_RANGES) are shown.
    Values are normalized to [0, 1] using PARAM_RANGES for comparability.
    Train and validation sets are shown in separate side-by-side subplots
    to avoid visual overlap.

    Args:
        X: Full input tensor (N, 19) in physical units.
        idx_train: Training indices.
        idx_val: Validation indices.
        output_dir: Directory to save the plot.
        model_name: Name identifier (e.g., "AL", "Baseline").
        iteration: Current iteration number.
        logger: Logger instance.
        max_lines: Maximum number of lines per set (train/val) to avoid clutter.
    """
    output_dir = Path(output_dir)

    X_train = X[idx_train].numpy() if hasattr(X, 'numpy') else np.asarray(X[idx_train])
    X_val = X[idx_val].numpy() if hasattr(X, 'numpy') else np.asarray(X[idx_val])

    # Select only non-fixed parameters
    var_indices = []
    var_names = []
    var_lo = []
    var_hi = []
    for i, key in enumerate(PARAM_ORDER):
        lo, hi = PARAM_RANGES[key]
        if lo < hi:
            var_indices.append(i)
            var_names.append(key.replace("IN_", ""))
            var_lo.append(lo)
            var_hi.append(hi)

    if not var_indices:
        return

    var_lo = np.array(var_lo)
    var_hi = np.array(var_hi)
    n_axes = len(var_indices)

    # Extract and normalize to [0, 1]
    X_train_sel = X_train[:, var_indices]
    X_val_sel = X_val[:, var_indices]
    X_train_norm = (X_train_sel - var_lo) / (var_hi - var_lo)
    X_val_norm = (X_val_sel - var_lo) / (var_hi - var_lo)

    n_train_total = len(X_train_norm)
    n_val_total = len(X_val_norm)

    # Subsample if too many lines
    rng = np.random.default_rng(iteration)
    if n_train_total > max_lines:
        X_train_norm = X_train_norm[rng.choice(n_train_total, max_lines, replace=False)]
    if n_val_total > max_lines:
        X_val_norm = X_val_norm[rng.choice(n_val_total, max_lines, replace=False)]

    # Alpha scales with sample count so dense plots aren't overwhelming
    alpha_train = max(0.05, min(0.35, 30.0 / max(1, len(X_train_norm))))
    alpha_val = max(0.08, min(0.45, 40.0 / max(1, len(X_val_norm))))

    fig, axes = plt.subplots(1, 2, figsize=(max(14, n_axes * 1.4), 5.5), sharey=True)
    xs = np.arange(n_axes)

    panels = [
        (axes[0], X_train_norm, 'steelblue', alpha_train, f'Train  (n={n_train_total})'),
        (axes[1], X_val_norm, 'darkorange', alpha_val, f'Val  (n={n_val_total})'),
    ]
    for ax, data, color, alpha, title in panels:
        for row in data:
            ax.plot(xs, row, color=color, alpha=alpha, linewidth=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(var_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        for x in xs:
            ax.axvline(x, color='grey', linewidth=0.5, alpha=0.5)

    axes[0].set_ylabel('Normalized value [0, 1]', fontsize=11)
    fig.suptitle(f'{model_name} Parallel Coordinates — Iteration {iteration}', fontsize=14)

    plt.tight_layout()
    plot_path = output_dir / f'{model_name.lower()}_parallel_coords_iter{iteration:03d}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"{model_name} parallel coordinates saved to {output_dir}")


def plot_candidate_uncertainty(candidates, pred_var, output_dir, model_name, iteration,
                                logger, max_points=5000, n_bins=20):
    """
    Plot predicted uncertainty on candidate points vs each input parameter.

    Creates a grid of subplots (one per non-fixed parameter). Each subplot shows
    candidate uncertainty (standard deviation) as a function of the parameter
    value, overlaid with a binned running mean to reveal structure.

    Args:
        candidates: Candidate tensor (N, 19) in physical units.
        pred_var: Predicted variance tensor (N,) or (N, 1).
        output_dir: Directory to save the plot.
        model_name: Name identifier (e.g. "AL", "Baseline"). Used in filename and title.
        iteration: Current iteration number.
        logger: Logger instance.
        max_points: Maximum number of points to scatter (subsample if more).
        n_bins: Number of bins for the running mean line.
    """
    output_dir = Path(output_dir)

    cand_np = candidates.detach().cpu().numpy() if hasattr(candidates, 'detach') else np.asarray(candidates)
    var_np = pred_var.detach().cpu().numpy() if hasattr(pred_var, 'detach') else np.asarray(pred_var)
    var_np = np.asarray(var_np).reshape(-1)
    # Use standard deviation for plotting (more interpretable than variance)
    std_np = np.sqrt(np.clip(var_np, a_min=0.0, a_max=None))

    # Select only non-fixed parameters
    var_indices = []
    var_names = []
    var_lo = []
    var_hi = []
    for i, key in enumerate(PARAM_ORDER):
        lo, hi = PARAM_RANGES[key]
        if lo < hi:
            var_indices.append(i)
            var_names.append(key.replace("IN_", ""))
            var_lo.append(lo)
            var_hi.append(hi)

    if not var_indices:
        return

    n_params = len(var_indices)

    # Subsample points if there are too many to scatter efficiently
    n_total = len(cand_np)
    rng = np.random.default_rng(iteration)
    if n_total > max_points:
        sel = rng.choice(n_total, max_points, replace=False)
        cand_plot = cand_np[sel]
        std_plot = std_np[sel]
    else:
        cand_plot = cand_np
        std_plot = std_np

    # Layout: 3 columns, enough rows to fit all parameters
    n_cols = min(3, n_params)
    n_rows = int(np.ceil(n_params / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.2 * n_rows),
                              sharey=True)
    axes = np.atleast_1d(axes).flatten()

    # Fixed y-range for visual comparison across parameters
    y_max = float(np.quantile(std_plot, 0.995)) * 1.05 if len(std_plot) else 1.0
    if not np.isfinite(y_max) or y_max <= 0:
        y_max = 1.0

    color = 'steelblue' if model_name.lower().startswith('al') else 'darkorange'

    for ax_idx, (param_idx, name, lo, hi) in enumerate(
            zip(var_indices, var_names, var_lo, var_hi)):
        ax = axes[ax_idx]
        x = cand_plot[:, param_idx]
        ax.scatter(x, std_plot, s=4, alpha=0.15, color=color, linewidths=0)

        # Binned running mean of std vs parameter value
        bin_edges = np.linspace(lo, hi, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_means = np.full(n_bins, np.nan)
        # Use the full (unsubsampled) arrays for accurate binning
        x_full = cand_np[:, param_idx]
        for b in range(n_bins):
            mask = (x_full >= bin_edges[b]) & (x_full < bin_edges[b + 1])
            if b == n_bins - 1:
                mask |= (x_full == bin_edges[-1])
            if mask.any():
                bin_means[b] = std_np[mask].mean()
        ax.plot(bin_centers, bin_means, color='black', linewidth=1.6,
                marker='o', markersize=3, label='binned mean')

        ax.set_xlabel(name, fontsize=10)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.3)
        if ax_idx % n_cols == 0:
            ax.set_ylabel('Candidate std', fontsize=10)

    # Hide unused axes
    for j in range(n_params, len(axes)):
        axes[j].set_visible(False)

    # Single legend on the first axis
    axes[0].legend(loc='upper right', fontsize=8, framealpha=0.9)
    fig.suptitle(
        f'{model_name} Candidate Uncertainty vs Input Parameters — Iteration {iteration}  '
        f'(N={n_total})',
        fontsize=13,
    )

    plt.tight_layout()
    plot_path = output_dir / f'{model_name.lower()}_candidate_uncertainty_iter{iteration:03d}.png'
    plt.savefig(plot_path, dpi=130, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"{model_name} candidate uncertainty plot saved to {plot_path}")


def plot_iteration_metrics(iterations, al_metrics, baseline_metrics, output_dir, logger):
    """
    Plot validation loss, R² score, and dataset sizes across active learning iterations.

    Args:
        iterations: List of iteration numbers
        al_metrics: Dict with 'train_losses', 'val_losses', 'r2_scores', 'n_train', 'n_val' for active learning
        baseline_metrics: Dict with 'train_losses', 'val_losses', 'r2_scores', 'n_train', 'n_val' for baseline
        output_dir: Output directory for plots
        logger: Logger instance
    """
    _, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax1, ax2, ax3 = axes

    # Show a label every 5 iterations; fall back to every iteration for short runs
    tick_step = 5 if len(iterations) > 10 else 1
    label_ticks = [i for i in iterations if i % tick_step == 0]
    if iterations[0] not in label_ticks:
        label_ticks = [iterations[0]] + label_ticks

    # Plot 1: Train/Validation Loss
    ax1.plot(iterations, al_metrics['train_losses'], 'b-', linewidth=2, marker='o', markersize=6, label='AL Train')
    ax1.plot(iterations, al_metrics['val_losses'], 'b--', linewidth=2, marker='s', markersize=6, label='AL Validation')
    ax1.plot(iterations, baseline_metrics['train_losses'], 'r-', linewidth=2, marker='o', markersize=6, label='Baseline Train')
    ax1.plot(iterations, baseline_metrics['val_losses'], 'r--', linewidth=2, marker='s', markersize=6, label='Baseline Validation')
    if al_metrics.get('cross_val_losses'):
        ax1.plot(iterations, al_metrics['cross_val_losses'], 'g:', linewidth=2, marker='^', markersize=6, label='AL on Base Val')
    if baseline_metrics.get('cross_val_losses'):
        ax1.plot(iterations, baseline_metrics['cross_val_losses'], 'g--', linewidth=2, marker='v', markersize=6, label='Base on AL Val')
    # Static evaluation datasets
    if al_metrics.get('mcmc_eval_losses'):
        ax1.plot(iterations, al_metrics['mcmc_eval_losses'], 'm-', linewidth=1.5, marker='D', markersize=5, label='AL on MCMC')
    if baseline_metrics.get('mcmc_eval_losses'):
        ax1.plot(iterations, baseline_metrics['mcmc_eval_losses'], 'm--', linewidth=1.5, marker='d', markersize=5, label='Base on MCMC')
    if al_metrics.get('static_random_eval_losses'):
        ax1.plot(iterations, al_metrics['static_random_eval_losses'], 'c-', linewidth=1.5, marker='P', markersize=5, label='AL on Static Rnd')
    if baseline_metrics.get('static_random_eval_losses'):
        ax1.plot(iterations, baseline_metrics['static_random_eval_losses'], 'c--', linewidth=1.5, marker='p', markersize=5, label='Base on Static Rnd')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Best Loss (MSE)', fontsize=12)
    ax1.set_title('Train/Validation Loss vs Iteration', fontsize=14)
    ax1.grid(True, which='major', alpha=0.3)
    ax1.grid(True, which='minor', alpha=0.15)
    ax1.set_xticks(label_ticks)
    ax1.set_xticks(iterations, minor=True)
    ax1.set_yscale('log')
    ax1.legend(fontsize=9)

    # Plot 2: R² Score (training, validation, and cross-validation)
    if al_metrics.get('train_r2_scores'):
        ax2.plot(iterations, al_metrics['train_r2_scores'], 'b-', linewidth=2, marker='o', markersize=6, label='AL Train')
    ax2.plot(iterations, al_metrics['r2_scores'], 'b--', linewidth=2, marker='s', markersize=6, label='AL Validation')
    if baseline_metrics.get('train_r2_scores'):
        ax2.plot(iterations, baseline_metrics['train_r2_scores'], 'r-', linewidth=2, marker='o', markersize=6, label='Baseline Train')
    ax2.plot(iterations, baseline_metrics['r2_scores'], 'r--', linewidth=2, marker='s', markersize=6, label='Baseline Validation')
    if al_metrics.get('cross_val_r2'):
        ax2.plot(iterations, al_metrics['cross_val_r2'], 'g:', linewidth=2, marker='^', markersize=6, label='AL on Base Val')
    if baseline_metrics.get('cross_val_r2'):
        ax2.plot(iterations, baseline_metrics['cross_val_r2'], 'g--', linewidth=2, marker='v', markersize=6, label='Base on AL Val')
    # Static evaluation datasets
    if al_metrics.get('mcmc_eval_r2'):
        ax2.plot(iterations, al_metrics['mcmc_eval_r2'], 'm-', linewidth=1.5, marker='D', markersize=5, label='AL on MCMC')
    if baseline_metrics.get('mcmc_eval_r2'):
        ax2.plot(iterations, baseline_metrics['mcmc_eval_r2'], 'm--', linewidth=1.5, marker='d', markersize=5, label='Base on MCMC')
    if al_metrics.get('static_random_eval_r2'):
        ax2.plot(iterations, al_metrics['static_random_eval_r2'], 'c-', linewidth=1.5, marker='P', markersize=5, label='AL on Static Rnd')
    if baseline_metrics.get('static_random_eval_r2'):
        ax2.plot(iterations, baseline_metrics['static_random_eval_r2'], 'c--', linewidth=1.5, marker='p', markersize=5, label='Base on Static Rnd')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('R² Score vs Iteration', fontsize=14)
    ax2.grid(True, which='major', alpha=0.3)
    ax2.grid(True, which='minor', alpha=0.15)
    ax2.set_xticks(label_ticks)
    ax2.set_xticks(iterations, minor=True)
    all_r2 = al_metrics['r2_scores'] + baseline_metrics['r2_scores']
    all_r2 += al_metrics.get('train_r2_scores', []) + baseline_metrics.get('train_r2_scores', [])
    all_r2 += al_metrics.get('cross_val_r2', []) + baseline_metrics.get('cross_val_r2', [])
    all_r2 += al_metrics.get('mcmc_eval_r2', []) + baseline_metrics.get('mcmc_eval_r2', [])
    all_r2 += al_metrics.get('static_random_eval_r2', []) + baseline_metrics.get('static_random_eval_r2', [])
    finite_r2 = [v for v in all_r2 if v is not None and not (v != v) and abs(v) != float('inf')]
    r2_lower = -1.0
    ax2.set_ylim(r2_lower, 1.05)

    # Collect all (iteration, r2_value, color) pairs to check for below-limit points
    _r2_series = [
        (al_metrics.get('train_r2_scores', []), 'b'),
        (al_metrics['r2_scores'], 'b'),
        (baseline_metrics.get('train_r2_scores', []), 'r'),
        (baseline_metrics['r2_scores'], 'r'),
        (al_metrics.get('cross_val_r2', []), 'g'),
        (baseline_metrics.get('cross_val_r2', []), 'g'),
        (al_metrics.get('mcmc_eval_r2', []), 'm'),
        (baseline_metrics.get('mcmc_eval_r2', []), 'm'),
        (al_metrics.get('static_random_eval_r2', []), 'c'),
        (baseline_metrics.get('static_random_eval_r2', []), 'c'),
    ]
    for r2_vals, color in _r2_series:
        if not r2_vals:
            continue
        for it, val in zip(iterations, r2_vals):
            if val is not None and val == val and val < r2_lower:
                ax2.annotate('', xy=(it, r2_lower), xytext=(it, r2_lower + 0.08),
                             arrowprops=dict(arrowstyle='->', color=color, lw=2))

    ax2.legend(fontsize=9)

    # Plot 3: Dataset Sizes
    ax3.plot(iterations, al_metrics['n_train'], 'b-', linewidth=2, marker='o', markersize=6, label='AL Train')
    ax3.plot(iterations, al_metrics['n_val'], 'b--', linewidth=2, marker='s', markersize=6, label='AL Validation')
    ax3.plot(iterations, baseline_metrics['n_train'], 'r-', linewidth=2, marker='o', markersize=6, label='Baseline Train')
    ax3.plot(iterations, baseline_metrics['n_val'], 'r--', linewidth=2, marker='s', markersize=6, label='Baseline Validation')
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Number of Samples', fontsize=12)
    ax3.set_title('Dataset Size vs Iteration', fontsize=14)
    ax3.grid(True, which='major', alpha=0.3)
    ax3.grid(True, which='minor', alpha=0.15)
    ax3.set_xticks(label_ticks)
    ax3.set_xticks(iterations, minor=True)
    ax3.legend(fontsize=9)

    plt.tight_layout()

    plot_path = Path(output_dir) / "iteration_metrics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved iteration metrics plot to {plot_path}")


# ===== Advanced Diagnostics (GP-specific) =====

def plot_advanced_diagnostics(model, X_eval_norm, Y_eval_true, X_train_norm,
                              model_type, threshold, n_dim,
                              plots_dir, iteration, new_points_norm=None,
                              jitter=1e-3, num_samples=8, logger=None):
    """
    Generate advanced diagnostic plots for GP models.

    Produces: GP mean vs truth scatter, residuals plot, residuals vs true.

    Args:
        model: Trained GP or MLP model
        X_eval_norm: Normalized evaluation inputs
        Y_eval_true: True evaluation targets (in transformed space)
        X_train_norm: Normalized training inputs (unused, kept for compatibility)
        model_type: One of "exact_gp", "deep_gp", "sparse_gp", "mlp"
        threshold: Classification threshold in transformed space
        n_dim: Input dimensionality (unused, kept for compatibility)
        plots_dir: Directory to save plots
        iteration: Current iteration number
        new_points_norm: Newly added points this iteration (unused, kept for compatibility)
        jitter: Cholesky jitter for GP inference
        num_samples: Number of likelihood samples for DeepGP
        logger: Logger instance
    """
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
             gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter), \
             gpytorch.settings.num_likelihood_samples(num_samples):
            preds = model.likelihood(model(X_eval_norm))
            mean = preds.mean.detach().mean(dim=0).squeeze().cpu()
    else:
        # exact_gp, sparse_gp
        model.likelihood.eval()
        with torch.no_grad(), \
             gpytorch.settings.fast_pred_var(), \
             gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter):
            preds = model.likelihood(model(X_eval_norm))
            mean = preds.mean.detach().cpu()

    y_true = Y_eval_true.cpu()

    plots_dir = Path(plots_dir) / "advanced"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. GP mean vs truth scatter with residuals
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

    # Residuals histogram
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
