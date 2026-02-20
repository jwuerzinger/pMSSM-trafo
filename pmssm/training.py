"""
Training loops and workers for pMSSM models.

This module provides:
- Transformer training with early stopping
- GP model creation and training
- Multiprocessing training workers for both transformer and GP models
"""

import sys
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import gpytorch

from .config import TARGET_CONFIG
from .datasets import PMSSMDataset
from .data import compute_stats, normalize_x, transform_y, inverse_transform_y
from .evaluation import get_model_name, compute_gp_r2, extract_lengthscales
from .logging_utils import setup_worker_logging
from .visualization import (
    plot_losses, scatter_true_vs_pred, hist_true_vs_pred,
    compare_random_predictions, gp_predict
)
from .models import PMSSMTransformerTabular


# ===== Transformer Training =====

def train_with_validation(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device="cpu",
    epochs=30,
    early_stopping=True,
    patience=500,
    scheduler=None,
    grad_clip=None,
    logger=None,
):
    """
    Train transformer model with validation and early stopping.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        optimizer: Optimizer instance
        criterion: Loss function
        device: Compute device
        epochs: Maximum number of epochs
        early_stopping: Whether to use early stopping
        patience: Early stopping patience (epochs)
        scheduler: Optional learning rate scheduler
        grad_clip: Optional gradient clipping max norm
        logger: Logger instance

    Returns:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """
    model.to(device)

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_model_state = None
    counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()

            # Gradient clipping if specified
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss += loss.item() * x.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # Step the scheduler if provided
        if scheduler is not None:
            scheduler.step()

        # Log progress
        lr = optimizer.param_groups[0]['lr']
        msg = (
            f"{get_model_name(model):<20} | "
            f"Epoch {epoch:04d} | "
            f"Train MSE = {train_loss:.6f} | "
            f"Val MSE = {val_loss:.6f} | "
            f"LR = {lr:.6e}"
        )
        if logger:
            logger.info(msg)
        else:
            print(msg)

        # Early stopping check
        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    msg = f"Early stopping triggered at epoch {epoch}"
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)
                    break

    # Load best weights before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses


def train_model_worker(gpu_id, X, Y, idx_train, idx_val, epochs, dropout,
                      result_queue, model_name="model", log_dir=None,
                      plots_dir=None, checkpoint_path=None,
                      warm_start_path=None, early_stopping=True, patience=200,
                      y_transform='zscore', target='DMRD'):
    """
    Worker function for multiprocessing transformer training.

    Args:
        gpu_id: GPU device ID or "cpu"
        X, Y: Full data tensors
        idx_train, idx_val: Train/val indices
        epochs: Number of training epochs
        dropout: Dropout rate
        result_queue: multiprocessing.Queue to return results
        model_name: Name identifier for this model (e.g., "AL", "Baseline")
        log_dir: Directory to save log files
        plots_dir: Directory to save diagnostic plots
        checkpoint_path: If provided, save model state dict to this path after training
        warm_start_path: If provided, load model weights from this checkpoint before training
        early_stopping: Whether to use early stopping (default: True)
        patience: Early stopping patience in epochs (default: 200)
        y_transform: Y transformation type: 'zscore' (default) or 'log'
        target: Target function name (e.g., 'DMRD') for log transformation
    """
    device = f"cuda:{gpu_id}" if isinstance(gpu_id, int) else gpu_id

    # Set up logging for this worker
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{model_name.lower()}_training.log"
        logger = setup_worker_logging(log_file, model_name)
    else:
        # Fallback to basic logging
        logger = logging.getLogger(model_name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(
                f'%(asctime)s [{model_name}] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            logger.addHandler(handler)

    # Compute stats and create datasets
    stats = compute_stats(X, Y, idx_train)
    train_dataset = PMSSMDataset(X, Y, idx_train, stats, y_transform=y_transform, target=target)
    val_dataset = PMSSMDataset(X, Y, idx_val, stats, y_transform=y_transform, target=target)

    # Log device and dataset sizes
    logger.info(f"Training on device: {device}")
    logger.info(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

    # Check for duplicate entries in the dataset
    X_train = X[idx_train]
    X_val = X[idx_val]

    # Check for duplicates within training set
    unique_train = torch.unique(X_train, dim=0)
    n_duplicates_train = len(X_train) - len(unique_train)
    if n_duplicates_train > 0:
        logger.warning(f"Found {n_duplicates_train} duplicate entries in training set!")

    # Check for duplicates within validation set
    unique_val = torch.unique(X_val, dim=0)
    n_duplicates_val = len(X_val) - len(unique_val)
    if n_duplicates_val > 0:
        logger.warning(f"Found {n_duplicates_val} duplicate entries in validation set!")

    # Check for overlap between train and val indices
    idx_overlap = set(idx_train.tolist()) & set(idx_val.tolist())
    if len(idx_overlap) > 0:
        logger.warning(f"Found {len(idx_overlap)} overlapping indices between train and val sets!")

    # Check if any train samples appear in val set (data leakage check)
    if len(idx_overlap) == 0:
        train_tuples = set(tuple(row.tolist()) for row in X_train)
        val_tuples = set(tuple(row.tolist()) for row in X_val)
        data_overlap = train_tuples & val_tuples
        if len(data_overlap) > 0:
            logger.warning(f"Found {len(data_overlap)} identical data samples in both train and val sets (different indices)!")

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    model = PMSSMTransformerTabular(
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        dropout=dropout,
    )

    # Warm-start from previous checkpoint if provided
    if warm_start_path is not None:
        if Path(warm_start_path).exists():
            model.load_state_dict(torch.load(warm_start_path, map_location='cpu'))
            logger.info(f"Warm-started from checkpoint: {warm_start_path}")
        else:
            logger.warning(f"Warm-start checkpoint not found: {warm_start_path}, training from scratch")

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    train_losses, val_losses = train_with_validation(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device=device,
        epochs=epochs,
        early_stopping=early_stopping,  # Use parameter from function args
        patience=patience,  # Use parameter from function args
        scheduler=scheduler,
        grad_clip=1.0,
        logger=logger,
    )

    # Compute metrics — all correspond to the best-validation-loss epoch
    best_val_epoch = min(range(len(val_losses)), key=lambda i: val_losses[i])
    best_val_loss = val_losses[best_val_epoch]
    best_train_loss = train_losses[best_val_epoch]

    # Compute R² scores on both validation and training sets
    # (model already has best-val-loss weights restored by train_with_validation)
    model.eval()
    model.to(device)
    _, _, mean_Y, std_Y = stats

    # Validation R²
    loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    X_batch, Y_batch = next(iter(loader))
    X_batch = X_batch.to(device)
    with torch.no_grad():
        Y_pred_transformed = model(X_batch).cpu()

    if y_transform == 'log':
        Y_true = inverse_transform_y(Y_batch, target=target)
        Y_pred = inverse_transform_y(Y_pred_transformed, target=target)
    else:  # zscore
        Y_true = Y_batch * std_Y + mean_Y
        Y_pred = Y_pred_transformed * std_Y + mean_Y

    ss_res = ((Y_true - Y_pred) ** 2).sum()
    ss_tot = ((Y_true - Y_true.mean()) ** 2).sum()
    r2 = (1 - (ss_res / ss_tot)).item()

    # Training R²
    train_loader_r2 = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    X_train_r2, Y_train_r2 = next(iter(train_loader_r2))
    X_train_r2 = X_train_r2.to(device)
    with torch.no_grad():
        Y_train_pred_transformed = model(X_train_r2).cpu()

    if y_transform == 'log':
        Y_train_true_r2 = inverse_transform_y(Y_train_r2, target=target)
        Y_train_pred_r2 = inverse_transform_y(Y_train_pred_transformed, target=target)
    else:  # zscore
        Y_train_true_r2 = Y_train_r2 * std_Y + mean_Y
        Y_train_pred_r2 = Y_train_pred_transformed * std_Y + mean_Y

    ss_res_train = ((Y_train_true_r2 - Y_train_pred_r2) ** 2).sum()
    ss_tot_train = ((Y_train_true_r2 - Y_train_true_r2.mean()) ** 2).sum()
    train_r2 = (1 - (ss_res_train / ss_tot_train)).item()

    # Log final metrics
    logger.info(f"Training complete!")
    logger.info(f"Best val epoch: {best_val_epoch}")
    logger.info(f"Best train loss: {best_train_loss:.6f}")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"R² score: {r2:.4f}")
    logger.info(f"Train R² score: {train_r2:.4f}")

    # Generate diagnostic plots if plots_dir is provided
    if plots_dir is not None:
        plots_dir = Path(plots_dir) / model_name.lower()
        plots_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating diagnostic plots in {plots_dir}")

        # Plot losses
        plot_losses(train_losses, val_losses, get_model_name(model), plot_dir=str(plots_dir))

        # Get predictions for plotting
        model.eval()
        model.to(device)

        # Training set predictions
        train_loader_full = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
        X_train_batch, Y_train_batch = next(iter(train_loader_full))
        X_train_batch = X_train_batch.to(device)
        with torch.no_grad():
            Y_train_pred_transformed = model(X_train_batch).cpu()

        # Convert to physical space for plotting
        if y_transform == 'log':
            Y_train_true = inverse_transform_y(Y_train_batch, target=target)
            Y_train_pred = inverse_transform_y(Y_train_pred_transformed, target=target)
        else:  # zscore
            Y_train_true = Y_train_batch * std_Y + mean_Y
            Y_train_pred = Y_train_pred_transformed * std_Y + mean_Y

        # Validation set predictions (already computed in physical space)
        # Y_true, Y_pred from R² calculation

        # Compare random predictions (text log)
        compare_random_predictions(
            Y_train_true, Y_train_pred, mode='train',
            model_name=get_model_name(model),
            n_points=min(10, len(train_dataset)), logger=logger
        )
        compare_random_predictions(
            Y_true, Y_pred, mode='validation',
            model_name=get_model_name(model),
            n_points=min(3, len(val_dataset)), logger=logger
        )

        # Scatter plots
        scatter_true_vs_pred(
            Y_train_true, Y_train_pred, mode='train',
            model_name=get_model_name(model), plot_dir=str(plots_dir)
        )
        scatter_true_vs_pred(
            Y_true, Y_pred, mode='validation',
            model_name=get_model_name(model), plot_dir=str(plots_dir)
        )

        # Histogram plots
        hist_true_vs_pred(
            Y_train_true, Y_train_pred, mode='train',
            model_name=get_model_name(model), plot_dir=str(plots_dir)
        )
        hist_true_vs_pred(
            Y_true, Y_pred, mode='validation',
            model_name=get_model_name(model), plot_dir=str(plots_dir)
        )

        logger.info(f"Diagnostic plots saved to {plots_dir}")

    # Save model checkpoint if requested
    if checkpoint_path is not None:
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Saved model checkpoint to {checkpoint_path}")

    # Return results via queue
    result_queue.put({
        "model_name": model_name,
        "best_train_loss": best_train_loss,
        "best_val_loss": best_val_loss,
        "r2_score": r2,
        "train_r2_score": train_r2,
        "train_losses": train_losses,
        "val_losses": val_losses,
    })


# ===== GP Model Training =====

def model_has_likelihood(model_type):
    """Return True if the model type uses a GP likelihood."""
    return model_type in ("exact_gp", "deep_gp", "sparse_gp")


def model_is_variational(model_type):
    """Return True if the model type uses variational inference."""
    return model_type in ("deep_gp", "sparse_gp")


def create_gp_model(model_type, x_train, y_train, x_val, y_val, n_dim,
                    kernel="RBF", lengthscale=1.0, noise=1e-2, use_ard=True,
                    m_nu=1.5, num_mixtures=4, use_dkl=False, feature_dim=2,
                    num_hidden_dims=10, num_middle_dims=0,
                    num_inducing_max=512, num_samples=8, seed=42, device=None,
                    target="DMRD", inducing_strategy="kmeans"):
    """
    Create and return a GP model (ExactGP, DeepGP, SparseGP, or MLP).

    Data is moved to the model's device before construction so that
    both self.x_train and GPyTorch's internal storage are on the same device.

    Args:
        model_type: One of "exact_gp", "deep_gp", "sparse_gp", "mlp"
        x_train: Normalized training inputs
        y_train: Transformed training targets
        x_val: Normalized validation inputs
        y_val: Transformed validation targets
        n_dim: Input dimensionality
        kernel: Kernel type ("RBF", "Matern", "Spectral")
        lengthscale: Initial lengthscale
        noise: Noise variance
        use_ard: Use ARD (Automatic Relevance Determination) for lengthscales
        m_nu: Smoothness parameter for Matern kernel
        num_mixtures: Number of mixtures for Spectral kernel
        use_dkl: Use Deep Kernel Learning
        feature_dim: Feature dimension for DKL
        num_hidden_dims: Number of hidden dimensions for DeepGP/MLP
        num_middle_dims: Number of middle dimensions for DeepGP
        num_inducing_max: Maximum number of inducing points for SparseGP/DeepGP
        num_samples: Number of samples for DeepGP
        seed: Random seed
        device: Target device (if None, auto-detects CUDA)
        target: Target function name (for threshold)
        inducing_strategy: Inducing point initialization ("kmeans" or "vanilla")

    Returns:
        Initialized GP or MLP model
    """
    # Import al_pmssmwithgp models
    try:
        _GP_PIPELINE_ROOT = Path(__file__).parent.parent / "al_pmssmwithgp" / "model"
        if str(_GP_PIPELINE_ROOT) not in sys.path:
            sys.path.insert(0, str(_GP_PIPELINE_ROOT))
        from gp_pipeline.models.exact_gp import ExactGP
        from gp_pipeline.models.deep_gp import DeepGP
        from gp_pipeline.models.sparse_gp import SparseGP
        from gp_pipeline.models.mlp import MLP
    except ImportError as e:
        raise ImportError(f"Failed to import al_pmssmwithgp GP models: {e}")

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
    """
    Train a GP model and return training history.

    Args:
        model: GP or MLP model to train
        model_type: One of "exact_gp", "deep_gp", "sparse_gp", "mlp"
        lr: Learning rate
        iters: Training iterations
        batch_size: Batch size (for variational models)
        jitter: Cholesky jitter
        patience: Early stopping patience (None disables early stopping)

    Returns:
        model: Trained model
        train_losses: List of training losses
        val_losses: List of validation losses
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


def train_gp_worker(gpu_id, X, Y, X_val, Y_val, data_min, data_max,
                    model_type, n_dim, gp_kwargs,
                    lr, iters, batch_size, jitter,
                    result_queue, model_name="model",
                    log_dir=None, plots_dir=None,
                    checkpoint_path=None, num_samples=8,
                    warm_start_path=None, target="DMRD",
                    patience=None):
    """
    Worker function for GP training (multiprocessing).

    Args:
        gpu_id: GPU device ID (int) or "cpu"
        X, Y: Raw training data tensors (non-normalized)
        X_val, Y_val: Raw validation data tensors (non-normalized)
        data_min, data_max: Normalization tensors
        model_type: "exact_gp", "deep_gp", "sparse_gp", or "mlp"
        n_dim: Number of input dimensions
        gp_kwargs: Dict of GP-specific arguments (kernel, lengthscale, noise, etc.)
        lr: Learning rate
        iters: Training iterations
        batch_size: Batch size (for variational models)
        jitter: Cholesky jitter
        result_queue: multiprocessing.Queue to return results
        model_name: Name identifier (e.g., "AL", "Baseline")
        log_dir: Directory for log files
        plots_dir: Directory for diagnostic plots
        checkpoint_path: Path to save model state dict after training
        num_samples: Number of samples for DeepGP
        warm_start_path: Path to a checkpoint to warm-start from
        target: Target function name (e.g., "DMRD")
        patience: Early stopping patience (None disables)
    """
    # Resolve device string and set default CUDA device for this worker
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
                f'%(asctime)s [{model_name}] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            logger.addHandler(handler)

    # Route gp_pipeline loggers through the same handlers
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

    # Compute metrics using the best model (already restored by train_gp_model).
    # Report MSE in transformed space (not MLL) so losses are comparable
    # across models and with cross-evaluation.
    best_val_iter = min(range(len(val_losses)), key=lambda i: val_losses[i])
    y_pred_val = gp_predict(model, x_val_norm, model_type,
                            jitter=jitter, num_samples=num_samples)
    y_pred_train = gp_predict(model, x_train_norm, model_type,
                              jitter=jitter, num_samples=num_samples)
    best_val_loss = ((y_val_t - y_pred_val.cpu()) ** 2).mean().item()
    best_train_loss = ((y_train_t - y_pred_train.cpu()) ** 2).mean().item()
    r2 = compute_gp_r2(model, x_val_norm, y_val_t, model_type, jitter=jitter,
                       num_samples=num_samples, target=target)
    train_r2 = compute_gp_r2(model, x_train_norm, y_train_t, model_type, jitter=jitter,
                             num_samples=num_samples, target=target)

    logger.info(f"Best val epoch (by MLL): {best_val_iter}")
    logger.info(f"Best train loss (MSE): {best_train_loss:.6f}")
    logger.info(f"Best val loss (MSE): {best_val_loss:.6f}")
    logger.info(f"R² score: {r2:.4f}")
    logger.info(f"Train R² score: {train_r2:.4f}")

    # Extract lengthscales
    lengthscales = extract_lengthscales(model, model_type)
    if lengthscales:
        logger.info(f"Learned lengthscales: {lengthscales}")

    # Generate diagnostic plots
    if plots_dir is not None:
        plots_dir = Path(plots_dir) / model_name.lower()
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Rolling average loss curve (MLL, from training loop)
        plot_losses(train_losses, val_losses, model_type, str(plots_dir))

        # Convert to physical space for scatter/histogram plots
        y_true_val_phys = inverse_transform_y(y_val_t.cpu(), target=target)
        y_pred_val_phys = inverse_transform_y(y_pred_val, target=target)
        y_true_train_phys = inverse_transform_y(y_train_t.cpu(), target=target)
        y_pred_train_phys = inverse_transform_y(y_pred_train, target=target)

        # Scatter: true vs predicted (train & validation)
        scatter_true_vs_pred(
            y_true_train_phys, y_pred_train_phys, "train", model_type, str(plots_dir))
        scatter_true_vs_pred(
            y_true_val_phys, y_pred_val_phys, "validation", model_type, str(plots_dir))

        # 2D histogram: true vs predicted (train & validation)
        hist_true_vs_pred(
            y_true_train_phys, y_pred_train_phys, "train", model_type, str(plots_dir))
        hist_true_vs_pred(
            y_true_val_phys, y_pred_val_phys, "validation", model_type, str(plots_dir))

        # Random predictions comparison (text log)
        compare_random_predictions(
            y_true_train_phys, y_pred_train_phys, "train", model_type,
            n_points=min(10, len(y_true_train_phys)), logger=logger)
        compare_random_predictions(
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
        "train_r2_score": train_r2,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "lengthscales": lengthscales,
    })
