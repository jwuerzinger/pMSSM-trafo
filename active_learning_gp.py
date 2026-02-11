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

from gp_pipeline.models.exact_gp import ExactGP
from gp_pipeline.models.deep_gp import DeepGP


# ---------------------------------------------------------------------------
# GP-specific constants
# ---------------------------------------------------------------------------

# True value for log-division of DMRD target (matching base.py:65)
DMRD_TRUE_VALUE = 0.12

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


def transform_y(Y):
    """Transform DMRD target: log(Y / true_value). Returns transformed Y."""
    return torch.log(Y / DMRD_TRUE_VALUE)


# ---------------------------------------------------------------------------
# GP model creation, training, and evaluation
# ---------------------------------------------------------------------------

def create_gp_model(model_type, x_train, y_train, x_val, y_val, n_dim,
                    kernel="RBF", lengthscale=1.0, noise=1e-2, use_ard=True,
                    m_nu=1.5, num_mixtures=4, use_dkl=False, feature_dim=2,
                    num_hidden_dims=10, num_middle_dims=0,
                    num_inducing_max=512, num_samples=8, seed=42, device=None):
    """Create and return a GP model (ExactGP or DeepGP).

    Data is moved to the model's device before construction so that
    both self.x_train (stored attribute) and self.train_inputs (GPyTorch's
    internal storage after self.to(device)) are on the same device.

    Args:
        device: Target device string (e.g. "cuda:0", "cuda:1", "cpu"). If None,
                auto-detects CUDA. Pass explicitly when running on a specific GPU
                in a multiprocessing worker.
    """
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
            use_dkl=use_dkl, feature_dim=feature_dim, thr=0, epsilon=0,
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
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model


def train_gp_model(model, model_type, lr=1e-3, iters=1000,
                   batch_size=256, jitter=1e-3):
    """Train a GP model and return (model, train_losses, val_losses)."""
    if model_type == "exact_gp":
        model, train_losses, val_losses = model.do_train_loop(
            lr=lr, iters=iters, jitter=jitter
        )
    elif model_type == "deep_gp":
        model, train_losses, val_losses = model.do_train_loop(
            lr=lr, iters=iters, batch_size=batch_size, jitter=jitter
        )
    return model, train_losses, val_losses


def compute_gp_r2(model, x_val, y_val, is_deep, jitter=1e-3, num_samples=8):
    """Compute R² score on validation set using GP posterior predictions."""
    device = next(model.parameters()).device
    model.eval()

    if is_deep:
        model.likelihood.eval()
        with torch.no_grad(), \
             gpytorch.settings.fast_pred_var(False), \
             gpytorch.settings.cholesky_jitter(jitter), \
             gpytorch.settings.num_likelihood_samples(num_samples):
            preds = model.likelihood(model(x_val.to(device)))
            y_pred = preds.mean.detach().mean(dim=0).squeeze()
    else:
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
                           is_deep, jitter=1e-3, num_samples=8, logger=None):
    """
    Compute GP posterior variance on candidate points.

    Args:
        model: Trained GP model (ExactGP or DeepGP)
        X_candidates: Non-normalized candidates (N, 19) in physical units
        data_min, data_max: Normalization tensors
        is_deep: Whether model is DeepGP
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

    if is_deep:
        model.likelihood.eval()
        # Process in batches to avoid OOM for large candidate pools
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
        model.likelihood.eval()
        # Process in batches for ExactGP too (covariance can be large)
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
# Training worker (for parallel or sequential training)
# ---------------------------------------------------------------------------

def train_gp_worker(gpu_id, X, Y, X_val, Y_val, data_min, data_max,
                    model_type, n_dim, gp_kwargs,
                    lr, iters, batch_size, jitter,
                    result_queue, model_name="model",
                    log_dir=None, plots_dir=None,
                    checkpoint_path=None, num_samples=8,
                    warm_start_path=None):
    """
    Worker function for GP training (analogous to train_model_worker).

    Args:
        gpu_id: GPU device ID (int) or "cpu". Used to pin the worker to a
                specific GPU when running in parallel (spawn) mode.
        X, Y: Raw training data tensors (non-normalized)
        X_val, Y_val: Raw validation data tensors (non-normalized)
        data_min, data_max: Normalization tensors
        model_type: "exact_gp" or "deep_gp"
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

    is_deep = model_type == "deep_gp"

    # Normalize data
    x_train_norm = normalize_x(X, data_min, data_max)
    x_val_norm = normalize_x(X_val, data_min, data_max)
    y_train_t = transform_y(Y).view(-1)
    y_val_t = transform_y(Y_val).view(-1)

    logger.info(f"Training set size: {len(x_train_norm)}, Validation set size: {len(x_val_norm)}")
    logger.info(f"Model type: {model_type}")

    # Create model (pinned to the requested device)
    model = create_gp_model(
        model_type, x_train_norm, y_train_t, x_val_norm, y_val_t,
        n_dim=n_dim, num_samples=num_samples, device=device, **gp_kwargs
    )

    # Warm-start: load previous iteration's state dict if provided
    if warm_start_path is not None and Path(warm_start_path).exists():
        ckpt = torch.load(warm_start_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.likelihood.load_state_dict(ckpt['likelihood_state_dict'])
        logger.info(f"Warm-started from checkpoint: {warm_start_path}")
    elif warm_start_path is not None:
        logger.warning(f"Warm-start checkpoint not found: {warm_start_path}, training from scratch")

    # Train
    logger.info("Starting GP training...")
    model, train_losses, val_losses = train_gp_model(
        model, model_type, lr=lr, iters=iters, batch_size=batch_size, jitter=jitter
    )
    logger.info("GP training complete.")

    # Compute metrics
    best_train_loss = min(train_losses)
    best_val_loss = min(val_losses)
    r2 = compute_gp_r2(model, x_val_norm, y_val_t, is_deep, jitter=jitter,
                       num_samples=num_samples)

    logger.info(f"Best train loss: {best_train_loss:.6f}")
    logger.info(f"Best val loss: {best_val_loss:.6f}")
    logger.info(f"R² score: {r2:.4f}")

    # Generate diagnostic plots
    if plots_dir is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plots_dir = Path(plots_dir) / model_name.lower()
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Plot loss curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(train_losses, label='Train')
        ax1.plot(val_losses, label='Validation')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss (-MLL)')
        ax1.set_title(f'{model_name} Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Scatter plot: true vs predicted on validation set
        device = next(model.parameters()).device
        model.eval()
        if is_deep:
            model.likelihood.eval()
            with torch.no_grad(), \
                 gpytorch.settings.fast_pred_var(False), \
                 gpytorch.settings.cholesky_jitter(jitter), \
                 gpytorch.settings.num_likelihood_samples(num_samples):
                preds = model.likelihood(model(x_val_norm.to(device)))
                y_pred = preds.mean.detach().mean(dim=0).squeeze().cpu()
        else:
            model.likelihood.eval()
            with torch.no_grad(), \
                 gpytorch.settings.fast_pred_var(), \
                 gpytorch.settings.cholesky_jitter(jitter):
                preds = model.likelihood(model(x_val_norm.to(device)))
                y_pred = preds.mean.detach().cpu()

        ax2.scatter(y_val_t.cpu().numpy(), y_pred.numpy(), alpha=0.5, s=10)
        lims = [min(y_val_t.min().item(), y_pred.min().item()),
                max(y_val_t.max().item(), y_pred.max().item())]
        ax2.plot(lims, lims, 'r--', linewidth=1)
        ax2.set_xlabel('True (log-transformed)')
        ax2.set_ylabel('Predicted')
        ax2.set_title(f'{model_name} True vs Predicted (R²={r2:.4f})')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / 'diagnostics.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Diagnostic plots saved to {plots_dir}")

    # Save model checkpoint
    if checkpoint_path is not None:
        if is_deep:
            torch.save({
                'model_state_dict': model.state_dict(),
                'likelihood_state_dict': model.likelihood.state_dict(),
            }, checkpoint_path)
        else:
            torch.save({
                'model_state_dict': model.state_dict(),
                'likelihood_state_dict': model.likelihood.state_dict(),
            }, checkpoint_path)
        logger.info(f"Saved model checkpoint to {checkpoint_path}")

    # Return results via queue
    result_queue.put({
        "model_name": model_name,
        "best_train_loss": best_train_loss,
        "best_val_loss": best_val_loss,
        "r2_score": r2,
        "train_losses": train_losses,
        "val_losses": val_losses,
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
# GP-specific options
@click.option('--model-type', default='exact_gp', type=click.Choice(['exact_gp', 'deep_gp']), help="GP model type.")
@click.option('--kernel', default='RBF', type=str, help="Kernel type (RBF, Matern, RQK, SpectralMixture, RBF+Matern).")
@click.option('--lengthscale', default=1.0, type=float, help="Initial kernel lengthscale.")
@click.option('--noise', default=1e-2, type=float, help="Initial noise level.")
@click.option('--jitter', default=1e-3, type=float, help="Cholesky jitter.")
@click.option('--learning-rate', default=1e-3, type=float, help="GP optimizer learning rate.")
@click.option('--training-iterations', default=1000, type=int, help="GP training iterations per AL iteration.")
@click.option('--use-ard/--no-ard', default=True, help="Use ARD (Automatic Relevance Determination).")
@click.option('--use-dkl/--no-dkl', default=False, help="Use Deep Kernel Learning (ExactGP only).")
@click.option('--feature-dim', default=2, type=int, help="DKL feature dimension.")
@click.option('--num-hidden-dims', default=10, type=int, help="DeepGP hidden layer dimensions.")
@click.option('--num-middle-dims', default=0, type=int, help="DeepGP middle layer dimensions (0 = no middle layer).")
@click.option('--num-inducing-max', default=512, type=int, help="Max inducing points (DeepGP).")
@click.option('--gp-num-samples', default=8, type=int, help="Number of likelihood samples (DeepGP).")
@click.option('--batch-size', default=256, type=int, help="Batch size (DeepGP).")
@click.option('--warm-starting/--no-warm-starting', default=True, help="Warm-start from previous iteration.")
@click.option('--m-nu', default=1.5, type=float, help="Matern nu parameter.")
@click.option('--num-mixtures', default=4, type=int, help="Number of mixtures for SpectralMixture kernel.")
def main(testing, n_iterations, n_candidates, n_select, n_datasets, n_samples,
         output_dir, generate_data, min_gen_fraction, max_gen_attempts, gen_workers,
         model_type, kernel, lengthscale, noise, jitter, learning_rate,
         training_iterations, use_ard, use_dkl, feature_dim,
         num_hidden_dims, num_middle_dims, num_inducing_max, gp_num_samples,
         batch_size, warm_starting, m_nu, num_mixtures):
    """
    Active learning pipeline for pMSSM relic density prediction using GP models.

    Trains ExactGP or DeepGP, computes uncertainty via GP posterior variance,
    and selects most informative points for data generation.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if n_candidates < n_select:
        n_candidates = n_select

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file, logger = setup_logging(timestamp, output_dir=output_dir)

    logger.info("=" * 60)
    logger.info("Active Learning Pipeline for pMSSM (GP Models)")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Output directory: {output_dir}")

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

    is_deep = model_type == "deep_gp"

    logger.info(f"Configuration:")
    logger.info(f"  model_type: {model_type}")
    logger.info(f"  n_iterations: {n_iterations}")
    logger.info(f"  n_candidates: {n_candidates}")
    logger.info(f"  n_select: {n_select}")
    logger.info(f"  training_iterations: {training_iterations}")
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
    if is_deep:
        logger.info(f"  num_hidden_dims: {num_hidden_dims}")
        logger.info(f"  num_middle_dims: {num_middle_dims}")
        logger.info(f"  num_inducing_max: {num_inducing_max}")
        logger.info(f"  gp_num_samples: {gp_num_samples}")
        logger.info(f"  batch_size: {batch_size}")
    if generate_data:
        logger.info(f"  min_gen_fraction: {min_gen_fraction}")
        logger.info(f"  max_gen_attempts: {max_gen_attempts}")
        logger.info(f"  gen_workers: {gen_workers}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Build normalization tensors
    data_min, data_max = build_norm_tensors()
    logger.info(f"Normalization ranges built for {len(PARAM_ORDER)} parameters")

    # GP model kwargs (constant across iterations)
    gp_kwargs = dict(
        kernel=kernel, lengthscale=lengthscale, noise=noise,
        use_ard=use_ard, m_nu=m_nu, num_mixtures=num_mixtures,
        use_dkl=use_dkl, feature_dim=feature_dim,
        num_hidden_dims=num_hidden_dims, num_middle_dims=num_middle_dims,
        num_inducing_max=num_inducing_max,
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
    BASELINE_GPU_ID = 3
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

    # Previous model state for warm starting
    prev_al_checkpoint = None

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

        if use_parallel:
            # ---- Train AL and Baseline in parallel on different GPUs ----
            al_queue = mp.Queue()
            baseline_queue = mp.Queue()

            al_warm_start = prev_al_checkpoint if warm_starting else None
            al_process = mp.Process(
                target=train_gp_worker,
                args=(AL_GPU_ID, X_train_al, Y_train_al, X_val_al, Y_val_al,
                      data_min, data_max, model_type, len(PARAM_ORDER), gp_kwargs,
                      learning_rate, training_iterations, batch_size, jitter,
                      al_queue, "AL", iter_dir, iter_plots_dir, al_checkpoint_path,
                      gp_num_samples, al_warm_start),
            )
            baseline_process = mp.Process(
                target=train_gp_worker,
                args=(BASELINE_GPU_ID, X_train_base, Y_train_base, X_val_base, Y_val_base,
                      data_min, data_max, model_type, len(PARAM_ORDER), gp_kwargs,
                      learning_rate, training_iterations, batch_size, jitter,
                      baseline_queue, "Baseline", iter_dir, iter_plots_dir, None,
                      gp_num_samples),
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
            logger.info(f"Training AL {model_type} model ({n_train} train, {n_val} val)...")
            al_queue = mp.Queue()
            train_gp_worker(
                device, X_train_al, Y_train_al, X_val_al, Y_val_al,
                data_min, data_max, model_type, len(PARAM_ORDER), gp_kwargs,
                learning_rate, training_iterations, batch_size, jitter,
                al_queue, "AL", iter_dir, iter_plots_dir, al_checkpoint_path,
                num_samples=gp_num_samples, warm_start_path=al_warm_start,
            )
            al_results = al_queue.get()

            logger.info(f"Training Baseline {model_type} model ({n_train_base} train, {n_val_base} val)...")
            baseline_queue = mp.Queue()
            train_gp_worker(
                device, X_train_base, Y_train_base, X_val_base, Y_val_base,
                data_min, data_max, model_type, len(PARAM_ORDER), gp_kwargs,
                learning_rate, training_iterations, batch_size, jitter,
                baseline_queue, "Baseline", iter_dir, iter_plots_dir,
                num_samples=gp_num_samples,
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

        # ---- Compute uncertainty on candidates using AL model ----
        logger.info(f"Loading AL model checkpoint for uncertainty computation...")

        # Reload the trained AL model
        x_train_norm = normalize_x(X_train_al, data_min, data_max)
        x_val_norm = normalize_x(X_val_al, data_min, data_max)
        y_train_t = transform_y(Y_train_al).view(-1)
        y_val_t = transform_y(Y_val_al).view(-1)

        al_model = create_gp_model(
            model_type, x_train_norm, y_train_t, x_val_norm, y_val_t,
            n_dim=len(PARAM_ORDER), num_samples=gp_num_samples, **gp_kwargs
        )
        checkpoint = torch.load(al_checkpoint_path, map_location=device)
        al_model.load_state_dict(checkpoint['model_state_dict'])
        al_model.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        logger.info("AL model reloaded for uncertainty estimation")

        # Generate candidate pool and compute uncertainty
        logger.info(f"Generating {n_candidates} candidate points...")
        candidates = generate_candidate_pool(n_candidates, seed=iteration)

        pred_mean, pred_var = compute_uncertainty_gp(
            al_model, candidates, data_min, data_max,
            is_deep=is_deep, jitter=jitter, num_samples=gp_num_samples,
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
            "uncertainties": pred_var[top_indices].numpy().tolist(),
            "al_best_val_loss": al_results['best_val_loss'],
            "al_r2_score": al_results['r2_score'],
            "baseline_best_val_loss": baseline_results['best_val_loss'],
            "baseline_r2_score": baseline_results['r2_score'],
        })

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
                        is_deep=is_deep, jitter=jitter, num_samples=gp_num_samples,
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
            new_valid = (new_Y.squeeze() > 0)
            new_X = new_X[new_valid]
            new_Y = new_Y[new_valid]
            logger.info(f"Augmenting: {len(X)} + {len(new_X)} = "
                       f"{len(X) + len(new_X)} samples")
            X = torch.cat([X, new_X], dim=0)
            Y = torch.cat([Y, new_Y], dim=0)

        prev_al_checkpoint = al_checkpoint_path

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

    # ---- Save summary ----
    summary = {
        "timestamp": timestamp,
        "config": {
            "model_type": model_type,
            "n_iterations": n_iterations,
            "n_candidates": n_candidates,
            "n_select": n_select,
            "training_iterations": training_iterations,
            "learning_rate": learning_rate,
            "kernel": kernel,
            "lengthscale": lengthscale,
            "noise": noise,
            "jitter": jitter,
            "use_ard": use_ard,
            "use_dkl": use_dkl,
            "generate_data": generate_data,
        },
        "iterations": all_selected_points,
        "final_dataset_size": len(X),
    }

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
