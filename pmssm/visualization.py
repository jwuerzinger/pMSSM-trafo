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

# LSP neutralino composition is classified from the mixing-matrix fractions
# (bino, wino, higgsino), computed by the ntupler as N_11^2, N_12^2, N_13^2+N_14^2.
# The dominant fraction wins when it reaches LSP_PURITY_MIN; otherwise the row
# is "mixed" (no single component dominates). Rows with NaN fractions
# (non-neutralino LSP or missing branches) are labelled -1 and skipped in plots.
LSP_TYPE_NAMES = {0: 'bino', 1: 'wino', 2: 'higgsino', 3: 'mixed'}
LSP_TYPE_COLORS = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green', 3: 'tab:red'}
LSP_PURITY_MIN = 0.5


def classify_lsp_type(lsp_fracs):
    """Classify LSP type (0=bino, 1=wino, 2=higgsino, 3=mixed, -1=unknown).

    Mixed when the dominant fraction is below LSP_PURITY_MIN (default 0.5).

    Args:
        lsp_fracs: (N, 3) tensor/array with columns
            [bino_frac, wino_frac, higgsino_frac]. NaN rows (non-neutralino
            LSP / missing branches) are labelled -1.

    Returns:
        numpy int array of shape (N,) with values in {-1, 0, 1, 2, 3}.
    """
    F = lsp_fracs.detach().cpu().numpy() if hasattr(lsp_fracs, 'detach') else np.asarray(lsp_fracs)
    out = np.full(F.shape[0], -1, dtype=int)
    valid = np.isfinite(F).all(axis=1)
    if valid.any():
        Fv = F[valid]
        winner = np.argmax(Fv, axis=1)
        top = np.take_along_axis(Fv, winner[:, None], axis=1).squeeze(1)
        mixed = top < LSP_PURITY_MIN
        out[valid] = np.where(mixed, 3, winner).astype(int)
    return out


def _scatter_colored_by_lsp(ax, y_true, y_pred, lsp_fracs, alpha=0.5, s=4,
                            legend=True, **kwargs):
    """Scatter y_pred vs y_true on `ax`, colored by LSP type from lsp_fracs.

    Rows with NaN fractions (classified -1) are skipped.
    """
    labels = classify_lsp_type(lsp_fracs)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    plotted = []
    for k in (0, 1, 2, 3):
        m = labels == k
        if not m.any():
            continue
        ax.scatter(y_true[m], y_pred[m], color=LSP_TYPE_COLORS[k],
                   label=f"{LSP_TYPE_NAMES[k]} (n={int(m.sum())})",
                   alpha=alpha, s=s, **kwargs)
        plotted.append(k)
    if legend and plotted:
        ax.legend(fontsize=8, loc='best', framealpha=0.8)
    return plotted


def scatter_true_vs_pred(y_true, y_pred, mode, model_name, plot_dir="plots",
                          lsp_fracs=None):
    """
    Scatter plot of true vs predicted values.

    Args:
        y_true: True values (array-like) in physical units
        y_pred: Predicted values (array-like) in physical units
        mode: 'validation' or 'train'
        model_name: Model name string for filename
        plot_dir: Directory to save plot
        lsp_fracs: Optional (N, 3) tensor of [bino, wino, higgsino] fractions;
            when provided, points are colored by dominant LSP component.
    """
    title = "Validation set" if mode == "validation" else "Training set"

    # Convert to numpy if tensor
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()

    # Subsample consistently so LSP labels match plotted points
    n = len(y_true)
    if n > 10_000:
        idx = np.random.default_rng(42).choice(n, 10_000, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
        if lsp_fracs is not None:
            F_sub = lsp_fracs.detach().cpu().numpy() if hasattr(lsp_fracs, 'detach') else np.asarray(lsp_fracs)
            lsp_fracs = F_sub[idx]

    fig, ax = plt.subplots()
    if lsp_fracs is not None:
        _scatter_colored_by_lsp(ax, y_true, y_pred, lsp_fracs, alpha=0.5, s=10)
    else:
        color = 'orange' if mode == 'validation' else None
        ax.scatter(y_true, y_pred, alpha=0.5, color=color)
    ax.plot(
        [min(y_true), max(y_true)],
        [min(y_true), max(y_true)],
        linestyle='--', color='grey'
    )
    ax.set_xlabel("True Ωh²")
    ax.set_ylabel("Predicted Ωh²")
    ax.set_title(f"True vs Predicted Ωh² ({title})")
    fig.tight_layout()

    if not running_in_notebook():
        fig.savefig(f"{plot_dir}/{model_name}_true_vs_pred_{mode}.png",
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_eval_scatterplots(eval_results, iteration, plot_dir, logger,
                           max_points=10_000):
    """Plot a grid of true-vs-predicted scatterplots for all model/dataset combinations.

    Args:
        eval_results: list of dicts with keys:
            'model_name', 'dataset_name', 'y_true', 'y_pred', 'loss', 'r2', 'n'.
            Optional 'lsp_fracs' (N, 3) → enables mixing-matrix LSP coloring.
        iteration: current iteration number
        plot_dir: directory to save the plot
        logger: logger instance
        max_points: max points to plot per panel (subsampled for speed)
    """
    if not eval_results:
        return

    model_names = list(dict.fromkeys(r['model_name'] for r in eval_results))
    dataset_names = list(dict.fromkeys(r['dataset_name'] for r in eval_results))
    n_rows = len(model_names)
    n_cols = len(dataset_names)
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
            fracs = r.get('lsp_fracs', None)
            if fracs is not None and hasattr(fracs, 'detach'):
                fracs = fracs.detach().cpu().numpy()
            elif fracs is not None:
                fracs = np.asarray(fracs)

            if len(y_true) > max_points:
                idx = np.random.default_rng(42).choice(len(y_true), max_points, replace=False)
                y_true = y_true[idx]
                y_pred = y_pred[idx]
                if fracs is not None:
                    fracs = fracs[idx]

            if fracs is not None:
                _scatter_colored_by_lsp(ax, y_true, y_pred, fracs,
                                        alpha=0.3, s=4, rasterized=True)
            else:
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
    if logger is not None:
        logger.info(f"Saved evaluation scatterplots to {out_path}")


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


def plot_iteration_metrics(iterations, al_metrics, baseline_metrics, output_dir,
                           logger, title_suffix=None, filename="iteration_metrics.png",
                           show_raw=False):
    """Per-run trajectory plot — 2×3 grid for readability.

    Layout::

        ┌──────────────────┬──────────────────┬──────────────────┐
        │ (0,0)            │ (0,1)            │ (0,2)            │
        │ Own-set Loss     │ Shared-eval Loss │ Dataset size     │
        │ (4 lines)        │ (≤6 lines)       │ (4 lines)        │
        ├──────────────────┼──────────────────┼──────────────────┤
        │ (1,0)            │ (1,1)            │ (1,2)            │
        │ Own-set R²       │ Shared-eval R²   │ Δ R²             │
        │ (4 lines, linear │ (symlog: linear  │ (AL − Baseline   │
        │  ylim auto)      │  in [-1,1])      │  on each eval)   │
        └──────────────────┴──────────────────┴──────────────────┘

    Conventions
    -----------
    Per-pipeline (own-set) panels (col 0):
      Colour:    blue = AL,  red = Baseline
      Linestyle: solid = train, dashed = val
    Shared-eval / Δ panels (cols 1 + 2):
      Colour:    green = cross-val, magenta = MCMC, cyan = static_random
      Linestyle: solid = AL, dashed = Baseline   (Δ panel: solid only)

    `title_suffix` (e.g. "exact_gp / entropy_batch / warm / seed1") is appended
    to the figure suptitle so the file is self-describing.

    Args:
        iterations: List of iteration numbers.
        al_metrics: dict — keys (all lists indexed by iter):
            train_losses, val_losses, train_r2_scores (optional), r2_scores,
            cross_val_losses, cross_val_r2, mcmc_eval_losses, mcmc_eval_r2,
            static_random_eval_losses, static_random_eval_r2, n_train, n_val.
        baseline_metrics: same schema.
        output_dir: directory to save into.
        logger: Logger instance.
        title_suffix: optional string appended to figure title.
        filename: output PNG filename (default 'iteration_metrics.png').
    """
    import numpy as _np

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=False)
    (ax_loss_own, ax_loss_shr, ax_n) = axes[0]
    (ax_r2_own, ax_r2_shr, ax_delta) = axes[1]

    # Tick step
    tick_step = 5 if len(iterations) > 10 else 1
    label_ticks = [i for i in iterations if i % tick_step == 0]
    if iterations and iterations[0] not in label_ticks:
        label_ticks = [iterations[0]] + label_ticks

    # Rolling-mean window. Five iterations smooths the typical 3-5-iter noise
    # cycles in the per-iteration metrics (especially own-set loss / R² for
    # transformer + DNN, and Δ R² for catastrophic regimes) while preserving
    # genuine ~10-iter trends like exact_gp's late-iter collapse. For short
    # runs we drop the smoothing entirely.
    ROLL_WIN = 5 if len(iterations) >= 10 else 1

    def _rolling_mean(ys, w):
        """Centred moving average ignoring NaNs. Returns same length as `ys`."""
        if w <= 1:
            return ys
        n = len(ys)
        out = [_np.nan] * n
        half = w // 2
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            window = [v for v in ys[lo:hi] if v == v]
            if window:
                out[i] = sum(window) / len(window)
        return out

    def _plot(ax, ys, *, color, linestyle, label):
        """Plot the rolling-mean curve as the primary line; optionally also
        draw the raw data behind it (faint).

        Controlled by the outer `show_raw` kwarg. Markers are intentionally
        omitted — every-nth markers were confusing per user feedback.
        """
        if not ys or all(v is None for v in ys):
            return
        ys_clean = [_np.nan if (v is None or v != v) else float(v) for v in ys]
        if show_raw:
            # Faint raw line so outliers remain visible
            ax.plot(iterations, ys_clean,
                    color=color, linestyle=linestyle, linewidth=0.9,
                    alpha=0.30, label=None)
        # Rolling-mean overlay (the primary visual element)
        smoothed = _rolling_mean(ys_clean, ROLL_WIN)
        ax.plot(iterations, smoothed,
                color=color, linestyle=linestyle, linewidth=2.0,
                label=label)

    def _style_axis(ax, ylabel, title):
        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, which="major", alpha=0.3)
        ax.grid(True, which="minor", alpha=0.15)
        ax.set_xticks(label_ticks)
        if iterations:
            ax.set_xticks(iterations, minor=True)

    # ─── (0,0) Own-set Loss ────────────────────────────────────────────────
    _plot(ax_loss_own, al_metrics.get("train_losses"),
          color="tab:blue", linestyle="-", label="AL train")
    _plot(ax_loss_own, al_metrics.get("val_losses"),
          color="tab:blue", linestyle="--", label="AL val")
    _plot(ax_loss_own, baseline_metrics.get("train_losses"),
          color="tab:red", linestyle="-", label="Base train")
    _plot(ax_loss_own, baseline_metrics.get("val_losses"),
          color="tab:red", linestyle="--", label="Base val")
    _style_axis(ax_loss_own, "MSE (own-set)", "Own-set loss")
    # Always log-scale loss panels — MSE is positive-definite and typically
    # spans multiple decades; linear scale bunches everything at the bottom.
    # Compute y-limits excluding iter 1 (commonly an undertraining outlier
    # that drags one axis edge by 4+ decades).
    _loss_for_ylim = []
    for k in ("train_losses", "val_losses"):
        for m in (al_metrics, baseline_metrics):
            vs = m.get(k, []) or []
            for v in vs[1:] if len(vs) > 1 else vs:
                if v is not None and v == v and v > 0:
                    _loss_for_ylim.append(float(v))
    if _loss_for_ylim:
        ax_loss_own.set_yscale("log")
        lo = min(_loss_for_ylim)
        hi = max(_loss_for_ylim)
        ax_loss_own.set_ylim(lo * 0.5, hi * 2)
    ax_loss_own.legend(fontsize=8, loc="best", framealpha=0.85, ncol=2)

    # ─── (0,1) Shared-eval Loss ──────────────────────────────────────────
    # green = cross-val, magenta = MCMC, cyan = static_random; solid=AL,
    # dashed=Baseline.
    eval_sets = [
        ("cross_val_losses", "tab:green", "Cross-val"),
        ("mcmc_eval_losses", "tab:purple", "MCMC"),
        ("static_random_eval_losses", "tab:cyan", "Static rnd"),
    ]
    has_shr_loss = False
    for key, color, label in eval_sets:
        if al_metrics.get(key):
            _plot(ax_loss_shr, al_metrics[key],
                  color=color, linestyle="-",
                  label=f"AL on {label}")
            has_shr_loss = True
        if baseline_metrics.get(key):
            _plot(ax_loss_shr, baseline_metrics[key],
                  color=color, linestyle="--",
                  label=f"Base on {label}")
            has_shr_loss = True
    _style_axis(ax_loss_shr, "MSE (transformed)", "Shared-eval loss")
    # Always log-scale (positive-definite MSE); exclude iter 1 from ylim.
    _shr_loss_vals = []
    for key, _c, _l in eval_sets:
        for m in (al_metrics, baseline_metrics):
            vs = m.get(key, []) or []
            for v in vs[1:] if len(vs) > 1 else vs:
                if v is not None and v == v and v > 0:
                    _shr_loss_vals.append(float(v))
    if _shr_loss_vals:
        ax_loss_shr.set_yscale("log")
        lo = min(_shr_loss_vals)
        hi = max(_shr_loss_vals)
        ax_loss_shr.set_ylim(lo * 0.5, hi * 2)
    if has_shr_loss:
        ax_loss_shr.legend(fontsize=8, loc="best", framealpha=0.85, ncol=2)
    else:
        ax_loss_shr.text(0.5, 0.5, "no shared-eval data",
                          transform=ax_loss_shr.transAxes,
                          ha="center", va="center", color="gray")

    # ─── (0,2) Dataset size ───────────────────────────────────────────────
    _plot(ax_n, al_metrics.get("n_train"),
          color="tab:blue", linestyle="-", label="AL train")
    _plot(ax_n, al_metrics.get("n_val"),
          color="tab:blue", linestyle="--", label="AL val")
    _plot(ax_n, baseline_metrics.get("n_train"),
          color="tab:red", linestyle="-", label="Base train")
    _plot(ax_n, baseline_metrics.get("n_val"),
          color="tab:red", linestyle="--", label="Base val")
    _style_axis(ax_n, "Number of samples", "Dataset size")
    ax_n.legend(fontsize=8, loc="best", framealpha=0.85, ncol=2)

    # ─── (1,0) Own-set R² ─────────────────────────────────────────────────
    _plot(ax_r2_own, al_metrics.get("train_r2_scores"),
          color="tab:blue", linestyle="-", label="AL train")
    _plot(ax_r2_own, al_metrics.get("r2_scores"),
          color="tab:blue", linestyle="--", label="AL val")
    _plot(ax_r2_own, baseline_metrics.get("train_r2_scores"),
          color="tab:red", linestyle="-", label="Base train")
    _plot(ax_r2_own, baseline_metrics.get("r2_scores"),
          color="tab:red", linestyle="--", label="Base val")
    _style_axis(ax_r2_own, "R² (own-set)", "Own-set R²")
    # Compute y-limits excluding iter 1 to ignore undertraining outliers.
    _own_r2 = []
    for k in ("train_r2_scores", "r2_scores"):
        for m in (al_metrics, baseline_metrics):
            vs = m.get(k, []) or []
            for v in vs[1:] if len(vs) > 1 else vs:
                if v is not None and v == v and abs(v) != float("inf"):
                    _own_r2.append(float(v))
    if _own_r2:
        ylo, yhi = min(_own_r2), max(_own_r2)
        pad = max(0.05, 0.1 * (yhi - ylo))
        ax_r2_own.set_ylim(max(-1.05, ylo - pad), min(1.05, yhi + pad))
    ax_r2_own.legend(fontsize=8, loc="best", framealpha=0.85, ncol=2)
    ax_r2_own.axhline(0, color="gray", linewidth=0.8, alpha=0.5)

    # ─── (1,1) Shared-eval R² (symlog so catastrophic collapses are legible) ──
    r2_sets = [
        ("cross_val_r2", "tab:green", "Cross-val"),
        ("mcmc_eval_r2", "tab:purple", "MCMC"),
        ("static_random_eval_r2", "tab:cyan", "Static rnd"),
    ]
    has_shr_r2 = False
    _shr_r2_vals = []
    for key, color, label in r2_sets:
        if al_metrics.get(key):
            _plot(ax_r2_shr, al_metrics[key],
                  color=color, linestyle="-",
                  label=f"AL on {label}")
            has_shr_r2 = True
            _shr_r2_vals.extend(
                float(v) for v in al_metrics[key]
                if v is not None and v == v and abs(v) != float("inf")
            )
        if baseline_metrics.get(key):
            _plot(ax_r2_shr, baseline_metrics[key],
                  color=color, linestyle="--",
                  label=f"Base on {label}")
            has_shr_r2 = True
            _shr_r2_vals.extend(
                float(v) for v in baseline_metrics[key]
                if v is not None and v == v and abs(v) != float("inf")
            )
    _style_axis(ax_r2_shr, "R² (shared eval)", "Shared-eval R²")
    if has_shr_r2 and _shr_r2_vals:
        # Exclude iter 1 for ylim determination (avoid undertraining outliers
        # pushing the symlog axis to absurd ranges).
        _ylim_vals = []
        for key, _c, _l in r2_sets:
            for m in (al_metrics, baseline_metrics):
                vs = m.get(key, []) or []
                for v in vs[1:] if len(vs) > 1 else vs:
                    if v is not None and v == v and abs(v) != float("inf"):
                        _ylim_vals.append(float(v))
        if not _ylim_vals:
            _ylim_vals = _shr_r2_vals
        ymin, ymax = min(_ylim_vals), max(_ylim_vals)
        ax_r2_shr.set_yscale("symlog", linthresh=1.0)
        # Tight ylim: only pad the side that has data extending toward it.
        # For exact_gp's catastrophic case, ymin ≈ -160 (heavy pad needed)
        # while ymax ≈ 0.6 (don't pad up to 1.0+ — that's empty whitespace).
        upper = max(ymax * 1.05, ymax + 0.05) if ymax > 0 else 0.05
        lower = ymin - 0.10 * max(abs(ymin), 1)
        ax_r2_shr.set_ylim(lower, upper)
        ax_r2_shr.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
        ax_r2_shr.legend(fontsize=8, loc="best", framealpha=0.85, ncol=2)
    else:
        ax_r2_shr.text(0.5, 0.5, "no shared-eval data",
                       transform=ax_r2_shr.transAxes,
                       ha="center", va="center", color="gray")

    # ─── (1,2) Δ R²  (AL minus Baseline on each shared eval) ──────────────
    delta_sets = [
        ("cross_val_r2", "tab:green", "Cross-val"),
        ("mcmc_eval_r2", "tab:purple", "MCMC"),
        ("static_random_eval_r2", "tab:cyan", "Static rnd"),
    ]
    has_delta = False
    _delta_vals = []
    for key, color, label in delta_sets:
        al_v = al_metrics.get(key) or []
        bs_v = baseline_metrics.get(key) or []
        if not al_v or not bs_v:
            continue
        n = min(len(al_v), len(bs_v), len(iterations))
        if n == 0:
            continue
        delta = []
        for i in range(n):
            a, b = al_v[i], bs_v[i]
            if a is None or b is None or a != a or b != b:
                delta.append(_np.nan)
            else:
                delta.append(float(a) - float(b))
        if show_raw:
            # Faint raw line so outlier spikes remain visible
            ax_delta.plot(iterations[:n], delta,
                          color=color, linestyle="-", linewidth=0.9,
                          alpha=0.30)
        # Rolling-mean overlay carries the legend entry
        smoothed_delta = _rolling_mean(delta, ROLL_WIN)
        ax_delta.plot(iterations[:n], smoothed_delta,
                      color=color, linestyle="-", linewidth=2.0,
                      label=f"{label}")
        has_delta = True
        _delta_vals.extend(v for v in delta if v == v)
    _style_axis(ax_delta, "Δ R² = AL − Baseline", "Δ R² across shared evals")
    ax_delta.axhline(0, color="black", linewidth=0.8, alpha=0.7)
    if has_delta and _delta_vals:
        # Drop iter 1 from ylim calculation (commonly very noisy on small data).
        _delta_for_ylim = [v for v in _delta_vals if v == v]
        if len(_delta_for_ylim) > 5:
            # Use 95th percentile of |Δ| to set the ylim, so iter-1 spikes
            # don't blow up the axis range
            absvals = sorted(abs(v) for v in _delta_for_ylim)
            absmax = absvals[int(len(absvals) * 0.95)]
        else:
            absmax = max(abs(v) for v in _delta_for_ylim)
        ax_delta.set_yscale("symlog", linthresh=1.0)
        ax_delta.set_ylim(-absmax * 1.2 - 0.5, absmax * 1.2 + 0.5)
        ax_delta.legend(fontsize=8, loc="best", framealpha=0.85, ncol=2)
    else:
        ax_delta.text(0.5, 0.5, "no shared-eval data",
                      transform=ax_delta.transAxes,
                      ha="center", va="center", color="gray")

    # Suptitle (model/strategy/warm/seed if provided)
    suptitle = "Active-learning iteration metrics"
    if title_suffix:
        suptitle += f"\n{title_suffix}"
    fig.suptitle(suptitle, fontsize=13)

    plt.tight_layout(rect=(0, 0, 1, 0.97))

    plot_path = Path(output_dir) / filename
    plt.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    if logger:
        logger.info(f"Saved iteration metrics plot to {plot_path}")
    return plot_path


# ===== Advanced Diagnostics (GP-specific) =====

def plot_advanced_diagnostics(model, X_eval_norm, Y_eval_true, X_train_norm,
                              model_type, threshold, n_dim,
                              plots_dir, iteration, new_points_norm=None,
                              jitter=1e-3, num_samples=8, logger=None,
                              lsp_fracs_eval=None):
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
    Y_eval_true = Y_eval_true.view(-1).to(device)

    # Chunked: peak memory must not scale with n_eval, which for a GP path on
    # ROCm otherwise ends in an uncatchable GPU memory-access fault.
    from .evaluation import predict_chunked  # noqa: PLC0415  (circular at module level)
    mean, _lower, _upper = predict_chunked(
        model, X_eval_norm, model_type, jitter=jitter, num_samples=num_samples)

    y_true = Y_eval_true.cpu()

    plots_dir = Path(plots_dir) / "advanced"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. GP mean vs truth scatter with residuals
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # True vs predicted
    if lsp_fracs_eval is not None:
        _scatter_colored_by_lsp(axes[0], y_true.numpy(), mean.numpy(),
                                lsp_fracs_eval, alpha=0.3, s=5)
    else:
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
    if lsp_fracs_eval is not None:
        _scatter_colored_by_lsp(axes[2], y_true.numpy(), residuals,
                                lsp_fracs_eval, alpha=0.3, s=5, legend=False)
    else:
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


# ===== Representative-point trajectory tracking =====

def pick_representative_points(X, Y, lsp_fracs, target_value, seed=42):
    """Pick up to 5 anchor points to track mean±var across AL iterations.

    One point per LSP class (bino, wino, higgsino, mixed) — the point nearest
    `target_value` within its class — plus the dataset median row. Classes not
    represented in the pool are silently skipped.

    Args:
        X: (N, D) input tensor.
        Y: (N, 1) target tensor in physical units.
        lsp_fracs: (N, 3) tensor of [bino, wino, higgsino] fractions.
        target_value: physical Ωh² target (anchors the per-class pick).
        seed: unused tiebreaker RNG (kept for future extension).

    Returns:
        dict with X (k, D), Y (k, 1), lsp_fracs (k, 3), cls (k-list of int
        class ids, -1=median), labels (k-list of str), indices (k-list).
    """
    X_np = X.detach().cpu().numpy() if hasattr(X, 'detach') else np.asarray(X)
    Y_np = Y.detach().cpu().numpy().reshape(-1) if hasattr(Y, 'detach') else np.asarray(Y).reshape(-1)
    cls_labels = classify_lsp_type(lsp_fracs)

    picks = []  # (index, cls_id)
    for cls in (0, 1, 2, 3):
        mask = cls_labels == cls
        if not mask.any():
            continue
        cls_idx = np.where(mask)[0]
        best = int(cls_idx[np.argmin(np.abs(Y_np[cls_idx] - target_value))])
        picks.append((best, cls))

    median_idx = int(np.argmin(np.abs(Y_np - np.median(Y_np))))
    picks.append((median_idx, -1))  # -1 marks "median row" (ignore LSP class)

    idxs = [p[0] for p in picks]
    cls_ids = [p[1] for p in picks]
    names = [LSP_TYPE_NAMES.get(c, 'median') for c in cls_ids]
    # Subscript consistently whether inputs are tensors or arrays
    idx_t = torch.tensor(idxs, dtype=torch.long)
    return {
        'X': X[idx_t] if hasattr(X, '__getitem__') else torch.as_tensor(X_np[idx_t.numpy()]),
        'Y': Y[idx_t] if hasattr(Y, '__getitem__') else torch.as_tensor(Y_np[idx_t.numpy()]).unsqueeze(1),
        'lsp_fracs': lsp_fracs[idx_t],
        'cls': cls_ids,
        'labels': names,
        'indices': idxs,
    }


def plot_representative_trajectories(repr_log, Y_true, cls_ids, labels,
                                      target_value, plot_dir,
                                      y_transform='zscore', target='DMRD'):
    """Plot mean ± 1σ vs iteration for each representative point.

    `repr_log` entries are computed in the model's **transformed** output space
    (same space the MC-Dropout variance is measured in); this function maps
    mean and mean±std back to physical Ωh² so the per-point true value is
    directly comparable.

    Args:
        repr_log: list of dicts with keys 'iteration', 'mean' (k-list),
            'var' (k-list). mean/var are in transformed space.
        Y_true: (k,) or (k, 1) physical true Ωh² for the k anchor points.
        cls_ids: k-list of LSP class ids (0-3) or -1 (median row).
        labels: k-list of human-readable class names.
        target_value: physical Ωh² reference line.
        plot_dir: directory to save the figure (and CSV).
        y_transform: 'log' or 'zscore' — matches the training y_transform.
        target: target name for inverse_transform_y when y_transform='log'.
    """
    import pandas as pd
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    iters = np.array([e['iteration'] for e in repr_log])
    mean_t = np.array([e['mean'] for e in repr_log], dtype=float)   # (n_iter, k)
    std_t = np.sqrt(np.clip(np.array([e['var'] for e in repr_log], dtype=float), 0, None))

    # Map to physical space. For log-transform this is inverse_transform_y;
    # the ±1σ bands become asymmetric, which is correct for log-space noise.
    if y_transform == 'log':
        from .data import inverse_transform_y
        mean_p = inverse_transform_y(torch.from_numpy(mean_t), target=target).numpy()
        upper_p = inverse_transform_y(torch.from_numpy(mean_t + std_t), target=target).numpy()
        lower_p = inverse_transform_y(torch.from_numpy(mean_t - std_t), target=target).numpy()
    else:
        mean_p = mean_t
        upper_p = mean_t + std_t
        lower_p = mean_t - std_t

    Y_phys = (Y_true.detach().cpu().numpy() if hasattr(Y_true, 'detach')
              else np.asarray(Y_true)).reshape(-1)

    # CSV persistence (long format)
    rows = []
    for ep, e in enumerate(repr_log):
        for j in range(len(labels)):
            rows.append({
                'iteration': int(e['iteration']),
                'point_idx': j,
                'label': labels[j],
                'cls': cls_ids[j],
                'Y_true_phys': float(Y_phys[j]),
                'mean_transformed': float(mean_t[ep, j]),
                'var_transformed': float(std_t[ep, j] ** 2),
                'mean_phys': float(mean_p[ep, j]),
                'lower_phys': float(lower_p[ep, j]),
                'upper_phys': float(upper_p[ep, j]),
            })
    csv_path = plot_dir / 'representative_trajectory.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    k = len(labels)
    n_cols = min(k, 3)
    n_rows = int(np.ceil(k / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False, sharex=True)
    axes = axes.flatten()
    for i in range(k):
        ax = axes[i]
        c = LSP_TYPE_COLORS.get(cls_ids[i], 'gray')
        ax.fill_between(iters, lower_p[:, i], upper_p[:, i], alpha=0.25, color=c)
        ax.plot(iters, mean_p[:, i], 'o-', color=c, label='predicted mean')
        ax.axhline(Y_phys[i], color='k', linestyle='--', linewidth=1,
                   label=f'true Ωh² = {Y_phys[i]:.4f}')
        ax.axhline(target_value, color='r', linestyle=':', linewidth=1,
                   alpha=0.6, label=f'target = {target_value}')
        ax.set_title(f'{labels[i]}  (Ωh²={Y_phys[i]:.4f})')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Predicted Ωh²')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    for j in range(k, len(axes)):
        axes[j].axis('off')

    fig.suptitle('Representative points: predicted mean ± 1σ vs iteration',
                 fontsize=13, y=1.00)
    fig.tight_layout()
    out_path = plot_dir / 'representative_points_trajectory.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path, csv_path
