import warnings
warnings.filterwarnings('ignore', message='.*enable_nested_tensor.*')

import pmssm

from pathlib import Path
import sys
from datetime import datetime
import logging
import structlog
import json
import multiprocessing as mp

import click
import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# pMSSM parameter ranges (from physics constraints)
# Order matches load_pmssm_data branches
PARAM_ORDER = [
    "IN_meL", "IN_meR", "IN_mtauL", "IN_mtauR",
    "IN_mqL1", "IN_muR", "IN_mdR", "IN_mqL3",
    "IN_mtR", "IN_mbR", "IN_M_1", "IN_M_2",
    "IN_mu", "IN_M_3", "IN_At", "IN_Ab",
    "IN_Atau", "IN_mA", "IN_tanb"
]

PARAM_RANGES = {
    "IN_meL": (0, 2000),
    "IN_meR": (0, 2000),
    "IN_mtauL": (2000, 2000),  # Fixed
    "IN_mtauR": (2000, 2000),  # Fixed
    "IN_mqL1": (4000, 4000),   # Fixed
    "IN_muR": (4000, 4000),    # Fixed
    "IN_mdR": (4000, 4000),    # Fixed
    "IN_mqL3": (0, 2000),
    "IN_mtR": (0, 2000),
    "IN_mbR": (0, 2000),
    "IN_M_1": (-2000, 2000),
    "IN_M_2": (-2000, 2000),
    "IN_mu": (-2000, 2000),
    "IN_M_3": (1000, 4000),
    "IN_At": (-8000, 8000),
    "IN_Ab": (-2000, 2000),
    "IN_Atau": (-2000, 2000),
    "IN_mA": (0, 2000),
    "IN_tanb": (1, 60),
}

# Mapping from our CSV column names to Run3ModelGen parameter names
CSV_TO_MODELGEN = {
    "meL": "meL",
    "meR": "meR",
    "mtauL": "mtauL",
    "mtauR": "mtauR",
    "mqL1": "mqL1",
    "muR": "muR",
    "mdR": "mdR",
    "mqL3": "mqL3",
    "mtR": "mtR",
    "mbR": "mbR",
    "M_1": "M_1",
    "M_2": "M_2",
    "mu": "mu",
    "M_3": "M_3",
    "At": "AT",  # Note: Run3ModelGen uses uppercase AT
    "Ab": "Ab",
    "Atau": "Atau",
    "mA": "mA",
    "tanb": "tanb",
}


def generate_models_from_csv(csv_path, output_dir, logger):
    """
    Generate pMSSM models using Run3ModelGen from selected points CSV.

    Args:
        csv_path: Path to selected_points.csv
        output_dir: Directory to save generated models (iteration_XXX)
        logger: Logger instance

    Returns:
        Path to generated ROOT ntuple, or None if generation failed
    """
    import subprocess

    # Read the CSV file
    df = pd.read_csv(csv_path)
    n_models = len(df)
    logger.info(f"Read {n_models} points from {csv_path}")

    # Create config for fixed prior mode
    config = {
        "prior": "fixed",
        "num_models": n_models,
        "isGMSB": False,
        "parameters": {},
        "steps": [
            {"name": "prep_input", "output_dir": "input", "prefix": "IN"},
            {"name": "SPheno", "input_dir": "input", "output_dir": "SPheno", "log_dir": "SPheno_log", "prefix": "SP"},
            {"name": "micromegas", "input_dir": "SPheno", "output_dir": "micromegas", "prefix": "MO"},
        ],
    }

    # Map CSV columns to Run3ModelGen parameters
    for csv_col, modelgen_param in CSV_TO_MODELGEN.items():
        if csv_col in df.columns:
            config["parameters"][modelgen_param] = df[csv_col].tolist()
        else:
            logger.warning(f"Column {csv_col} not found in CSV")

    # Write config to file (use absolute paths since subprocess runs from Run3ModelGen dir)
    output_dir = Path(output_dir).resolve()
    config_path = output_dir / "modelgen_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=None)
    logger.info(f"Wrote ModelGen config to {config_path}")

    # Set up scan directory (absolute path)
    scan_dir = output_dir / "scan"
    scan_dir.mkdir(parents=True, exist_ok=True)

    # Get paths
    project_root = Path(__file__).parent.resolve()
    run3modelgen_dir = project_root / "Run3ModelGen"
    setup_script = run3modelgen_dir / "build" / "setup.sh"

    # Check if Run3ModelGen is built
    if not setup_script.exists():
        logger.error(f"Run3ModelGen setup script not found: {setup_script}")
        logger.error("Please build Run3ModelGen first: cd Run3ModelGen && pixi run build")
        return None

    # Run model generation using pixi with setup script
    logger.info(f"Starting model generation in {scan_dir}...")

    # Command to run genModels.py with proper environment
    cmd = f"source {setup_script} && genModels.py --config_file {config_path} --scan_dir {scan_dir}"

    # Find pixi executable
    import shutil
    pixi_path = shutil.which("pixi")
    if pixi_path is None:
        # Try common locations
        for path in [Path.home() / ".pixi" / "bin" / "pixi", Path("/u/jwuerzin/.pixi/bin/pixi")]:
            if path.exists():
                pixi_path = str(path)
                break
    if pixi_path is None:
        logger.error("Could not find pixi executable")
        return None

    try:
        # Run in Run3ModelGen's pixi environment
        result = subprocess.run(
            [pixi_path, "run", "bash", "-c", cmd],
            cwd=str(run3modelgen_dir),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        if result.returncode != 0:
            logger.error(f"genModels.py failed with return code {result.returncode}")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            return None

        logger.info("Model generation complete")
        logger.info(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)

    except subprocess.TimeoutExpired:
        logger.error("Model generation timed out after 1 hour")
        return None
    except Exception as e:
        logger.error(f"Model generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

    # Find the generated ntuple
    ntuple_files = list(scan_dir.glob("*.root"))
    if ntuple_files:
        ntuple_path = ntuple_files[0]
        logger.info(f"Generated ntuple: {ntuple_path}")
        return ntuple_path
    else:
        logger.warning("No ROOT ntuple found after generation")
        return None


def load_generated_data(ntuple_path, logger):
    """
    Load newly generated data from ROOT ntuple.

    Args:
        ntuple_path: Path to generated ROOT file
        logger: Logger instance

    Returns:
        X, Y tensors or None if loading failed
    """
    import uproot

    try:
        root_file = uproot.open(str(ntuple_path))
        # Run3ModelGen uses 'susy' as tree name
        tree = root_file["susy"]

        # Check if MO_Omega exists (micromegas may have failed)
        if "MO_Omega" not in tree.keys():
            logger.warning("MO_Omega not found in ntuple - SPheno or micromegas may have failed for all models")
            logger.warning("No new training data available from this generation")
            return None, None

        # Extract input parameters (same order as PARAM_ORDER)
        branches = [
            "IN_meL", "IN_meR", "IN_mtauL", "IN_mtauR",
            "IN_mqL1", "IN_muR", "IN_mdR", "IN_mqL3",
            "IN_mtR", "IN_mbR", "IN_M_1", "IN_M_2",
            "IN_mu", "IN_M_3", "IN_At", "IN_Ab",
            "IN_Atau", "IN_mA", "IN_tanb"
        ]

        X = np.column_stack([tree[b].array(library="np") for b in branches])
        Y = tree["MO_Omega"].array(library="np")

        # Filter valid entries (Omega != -1 and < 1)
        mask = (Y != -1.0) & (Y < 1.0)
        X = torch.from_numpy(X[mask]).float()
        Y = torch.from_numpy(Y[mask]).float().unsqueeze(1)

        if len(X) == 0:
            logger.warning("No valid models found after filtering (Omega != -1 and < 1)")
            return None, None

        logger.info(f"Loaded {len(X)} valid models from ntuple (filtered from {len(mask)} total)")
        return X, Y

    except Exception as e:
        logger.error(f"Failed to load generated data: {e}")
        return None, None


def setup_logging(timestamp, output_dir=None):
    """Set up structlog to write to both file and console.

    Args:
        timestamp: Timestamp string for log file naming
        output_dir: If provided, logs are saved to output_dir/active_learning.log
                   Otherwise, logs are saved to logs/active_learning_{timestamp}.log
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / "active_learning.log"
    else:
        Path("logs/").mkdir(parents=True, exist_ok=True)
        log_file = f"logs/active_learning_{timestamp}.log"

    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=False),
    )

    for handler in logging.root.handlers:
        handler.setFormatter(formatter)

    return str(log_file), structlog.get_logger()


def setup_worker_logging(log_file_path, model_name):
    """Set up logging for a worker process.

    Args:
        log_file_path: Path to the log file
        model_name: Name identifier for this model (e.g., "AL", "Baseline")

    Returns:
        Logger instance
    """
    # Create a logger specific to this worker
    logger = logging.getLogger(f"worker_{model_name}")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers

    # File handler for this worker
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Format with timestamp and model name
    formatter = logging.Formatter(
        f'%(asctime)s [{model_name}] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def train_model_worker(gpu_id, X, Y, idx_train, idx_val, epochs, dropout, result_queue, model_name="model", log_dir=None, plots_dir=None, checkpoint_path=None):
    """
    Worker function for multiprocessing training.

    Args:
        gpu_id: GPU device ID or "cpu"
        X, Y: Full data tensors
        idx_train, idx_val: Train/val indices
        epochs: Number of training epochs
        dropout: Dropout rate
        result_queue: multiprocessing.Queue to return results
        model_name: Name identifier for this model (e.g., "AL", "Baseline")
        log_dir: Directory to save log files. If provided, logs are saved to
                 log_dir/{model_name.lower()}_training.log
        plots_dir: Directory to save diagnostic plots. If provided, plots are saved to
                   plots_dir/{model_name.lower()}/
        checkpoint_path: If provided, save model state dict to this path after training
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
            handler.setFormatter(logging.Formatter(f'%(asctime)s [{model_name}] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
            logger.addHandler(handler)

    # Compute stats and create datasets
    stats = pmssm.compute_stats(X, Y, idx_train)
    train_dataset = pmssm.PMSSMDataset(X, Y, idx_train, stats)
    val_dataset = pmssm.PMSSMDataset(X, Y, idx_val, stats)

    # Log device and dataset sizes for verification
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
    # This is a more expensive check, so we only do it if no index overlap was found
    if len(idx_overlap) == 0:
        # Create a set of unique train samples as tuples for fast lookup
        train_tuples = set(tuple(row.tolist()) for row in X_train)
        val_tuples = set(tuple(row.tolist()) for row in X_val)
        data_overlap = train_tuples & val_tuples
        if len(data_overlap) > 0:
            logger.warning(f"Found {len(data_overlap)} identical data samples appearing in both train and val sets (different indices)!")

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    model = pmssm.PMSSMTransformerTabular(
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        dropout=dropout,
    )

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    train_losses, val_losses = pmssm.train_with_validation(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device=device,
        epochs=epochs,
        early_stopping=True,
        patience=500,
        scheduler=scheduler,
        grad_clip=1.0,
        logger=logger,
    )

    # Compute metrics
    best_train_loss = min(train_losses)
    best_val_loss = min(val_losses)

    # Compute R² score
    model.eval()
    model.to(device)
    _, _, mean_Y, std_Y = stats
    loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    X_batch, Y_batch = next(iter(loader))
    X_batch = X_batch.to(device)
    with torch.no_grad():
        Y_pred_norm = model(X_batch).cpu()
    Y_true = Y_batch * std_Y + mean_Y
    Y_pred = Y_pred_norm * std_Y + mean_Y
    ss_res = ((Y_true - Y_pred) ** 2).sum()
    ss_tot = ((Y_true - Y_true.mean()) ** 2).sum()
    r2 = (1 - (ss_res / ss_tot)).item()

    # Log final metrics
    logger.info(f"Training complete!")
    logger.info(f"Best train loss: {best_train_loss:.6f}")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"R² score: {r2:.4f}")

    # Generate diagnostic plots if plots_dir is provided
    if plots_dir is not None:
        plots_dir = Path(plots_dir) / model_name.lower()
        plots_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating diagnostic plots in {plots_dir}")

        # Plot losses
        pmssm.plot_losses(train_losses, val_losses, model, plot_dir=str(plots_dir))

        # Compare random predictions
        pmssm.compare_random_predictions(
            model, stats=stats, subset=train_dataset, mode='train',
            device=device, n_points=min(10, len(train_dataset)), logger=logger
        )
        pmssm.compare_random_predictions(
            model, stats=stats, subset=val_dataset, mode='validation',
            device=device, n_points=min(3, len(val_dataset)), logger=logger
        )

        # Scatter plots
        pmssm.scatter_true_vs_pred(
            model, stats=stats, subset=train_dataset, mode='train',
            device=device, plot_dir=str(plots_dir)
        )
        pmssm.scatter_true_vs_pred(
            model, stats=stats, subset=val_dataset, mode='validation',
            device=device, plot_dir=str(plots_dir)
        )

        # Histogram plots
        pmssm.hist_true_vs_pred(
            model, stats=stats, subset=train_dataset, mode='train',
            device=device, plot_dir=str(plots_dir)
        )
        pmssm.hist_true_vs_pred(
            model, stats=stats, subset=val_dataset, mode='validation',
            device=device, plot_dir=str(plots_dir)
        )

        logger.info(f"Diagnostic plots saved to {plots_dir}")

    # Save model checkpoint if requested (used by AL worker so main process can load
    # the trained model for MC Dropout without retraining from scratch)
    if checkpoint_path is not None:
        torch.save(model.state_dict(), checkpoint_path)
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
    import matplotlib.pyplot as plt

    _, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax1, ax2, ax3 = axes

    # Plot 1: Train/Validation Loss
    ax1.plot(iterations, al_metrics['train_losses'], 'b-', linewidth=2, marker='o', markersize=6, label='AL Train')
    ax1.plot(iterations, al_metrics['val_losses'], 'b--', linewidth=2, marker='s', markersize=6, label='AL Validation')
    ax1.plot(iterations, baseline_metrics['train_losses'], 'r-', linewidth=2, marker='o', markersize=6, label='Baseline Train')
    ax1.plot(iterations, baseline_metrics['val_losses'], 'r--', linewidth=2, marker='s', markersize=6, label='Baseline Validation')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Best Loss (MSE)', fontsize=12)
    ax1.set_title('Train/Validation Loss vs Iteration', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(iterations)
    ax1.set_yscale('log')
    ax1.legend(fontsize=9)

    # Plot 2: R² Score
    ax2.plot(iterations, al_metrics['r2_scores'], 'b-', linewidth=2, marker='o', markersize=6, label='Active Learning')
    ax2.plot(iterations, baseline_metrics['r2_scores'], 'r-', linewidth=2, marker='o', markersize=6, label='Baseline (Random)')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('R² Score vs Iteration', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(iterations)
    all_r2 = al_metrics['r2_scores'] + baseline_metrics['r2_scores']
    finite_r2 = [v for v in all_r2 if v is not None and not (v != v) and abs(v) != float('inf')]
    if finite_r2:
        ax2.set_ylim(min(0, min(finite_r2) - 0.1), 1.05)
    ax2.legend(fontsize=10)

    # Plot 3: Dataset Sizes
    ax3.plot(iterations, al_metrics['n_train'], 'b-', linewidth=2, marker='o', markersize=6, label='AL Train')
    ax3.plot(iterations, al_metrics['n_val'], 'b--', linewidth=2, marker='s', markersize=6, label='AL Val')
    ax3.plot(iterations, baseline_metrics['n_train'], 'r-', linewidth=2, marker='o', markersize=6, label='Baseline Train')
    ax3.plot(iterations, baseline_metrics['n_val'], 'r--', linewidth=2, marker='s', markersize=6, label='Baseline Val')
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Number of Samples', fontsize=12)
    ax3.set_title('Dataset Size vs Iteration', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(iterations)
    ax3.legend(fontsize=9)

    plt.tight_layout()

    plot_path = output_dir / "iteration_metrics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved iteration metrics plot to {plot_path}")


def generate_candidate_pool(n_candidates, seed=None):
    """
    Generate candidate points in pMSSM parameter space.

    Returns non-normalized points in original physical units.
    """
    if seed is not None:
        np.random.seed(seed)

    candidates = np.zeros((n_candidates, 19))

    for i, param in enumerate(PARAM_ORDER):
        low, high = PARAM_RANGES[param]
        if low == high:  # Fixed parameter
            candidates[:, i] = low
        else:
            candidates[:, i] = np.random.uniform(low, high, n_candidates)

    return torch.from_numpy(candidates).float()


def compute_uncertainty_mc_dropout(model, X_candidates, stats, n_samples, device, logger):
    """
    Compute predictive uncertainty using MC Dropout.

    Args:
        model: Trained PMSSMTransformerTabular with dropout
        X_candidates: Non-normalized candidates (N, 19)
        stats: (mean_X, std_X, mean_Y, std_Y) normalization stats
        n_samples: Number of stochastic forward passes
        device: Compute device
        logger: Logger instance

    Returns:
        pred_mean: (N, 1) mean predictions (normalized)
        pred_var: (N, 1) prediction variance (uncertainty)
    """
    model.to(device)
    model.train()  # Keep dropout active for MC sampling

    mean_X, std_X, mean_Y, std_Y = stats

    # Normalize candidates
    X_norm = (X_candidates - mean_X) / std_X
    X_norm = X_norm.to(device)

    predictions = []
    logger.info(f"Running {n_samples} MC Dropout forward passes...")

    with torch.no_grad():
        for i in range(n_samples):
            y_pred = model(X_norm)
            predictions.append(y_pred.cpu())

    predictions = torch.stack(predictions, dim=0)  # (T, N, 1)

    pred_mean = predictions.mean(dim=0)  # (N, 1)
    pred_var = predictions.var(dim=0)    # (N, 1) - uncertainty

    logger.info(f"Uncertainty stats: mean={pred_var.mean():.6f}, max={pred_var.max():.6f}")

    return pred_mean, pred_var


def select_top_uncertain(X_candidates, uncertainties, n_select):
    """Select indices of points with highest uncertainty."""
    uncertainties_flat = uncertainties.squeeze().numpy()
    all_sorted = np.argsort(uncertainties_flat)[::-1].copy()
    return all_sorted[:n_select].copy()


def save_selected_points(X_candidates, uncertainties, indices, output_dir, iteration):
    """Save selected points to CSV file."""
    iter_dir = output_dir / f"iteration_{iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # Create DataFrame with parameter names
    param_names = [p.replace("IN_", "") for p in PARAM_ORDER]

    selected_X = X_candidates[indices].numpy()
    selected_unc = uncertainties[indices].squeeze().numpy()

    df = pd.DataFrame(selected_X, columns=param_names)
    df["uncertainty"] = selected_unc

    csv_path = iter_dir / "selected_points.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


@click.command()
@click.option('--testing', is_flag=True, help="Run in testing mode (small data, few epochs).")
@click.option('--n-iterations', default=1, type=int, help="Number of active learning iterations.")
@click.option('--n-candidates', default=1000, type=int, help="Candidate pool size.")
@click.option('--n-select', default=10, type=int, help="Number of points to select per iteration.")
@click.option('--mc-samples', default=30, type=int, help="Number of MC dropout forward passes.")
@click.option('--epochs', default=2000, type=int, help="Training epochs per iteration (default: 2000).")
@click.option('--dropout', default=0.1, type=float, help="Dropout rate for MC dropout.")
@click.option('--n-datasets', default=None, type=int, help="Number of ROOT datasets to load.")
@click.option('--n-samples', default=None, type=int, help="Number of samples to use from data.")
@click.option('--output-dir', default='active_learning_output', type=str, help="Output directory.")
@click.option('--generate-data', is_flag=True, help="Generate new models using Run3ModelGen.")
@click.option('--min-gen-fraction', default=0.6, type=float, help="Minimum fraction of n-select that must be generated successfully before stopping retries (default: 0.6).")
@click.option('--max-gen-attempts', default=10, type=int, help="Maximum number of generation attempts per iteration (default: 10).")
def main(testing, n_iterations, n_candidates, n_select, mc_samples, epochs, dropout, n_datasets, n_samples, output_dir, generate_data, min_gen_fraction, max_gen_attempts):
    """
    Active learning pipeline for pMSSM relic density prediction.

    Trains PMSSMTransformerTabular, computes uncertainty via MC Dropout,
    and selects most informative points for data generation.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Increase n_candidates if needed:
    if n_candidates < n_select: n_candidates = n_select

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up main logging to output_dir/active_learning.log
    log_file, logger = setup_logging(timestamp, output_dir=output_dir)

    logger.info("=" * 60)
    logger.info("Active Learning Pipeline for pMSSM")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Output directory: {output_dir}")

    # Apply testing mode defaults
    if testing:
        n_datasets = n_datasets if n_datasets is not None else 3
        n_samples = n_samples if n_samples is not None else 30
        epochs = 10
        n_candidates = 100
        mc_samples = 10
        logger.info("Testing mode enabled")
    else:
        n_datasets = n_datasets if n_datasets is not None else -1
        n_samples = n_samples if n_samples is not None else None

    logger.info(f"Configuration:")
    logger.info(f"  n_iterations: {n_iterations}")
    logger.info(f"  n_candidates: {n_candidates}")
    logger.info(f"  n_select: {n_select}")
    logger.info(f"  mc_samples: {mc_samples}")
    logger.info(f"  epochs: {epochs}")
    logger.info(f"  dropout: {dropout}")
    logger.info(f"  n_datasets: {n_datasets}")
    logger.info(f"  n_samples: {n_samples if n_samples else 'all'}")
    logger.info(f"  generate_data: {generate_data}")
    if generate_data:
        logger.info(f"  min_gen_fraction: {min_gen_fraction} (target: {int(n_select * min_gen_fraction)} valid models per iteration)")
        logger.info(f"  max_gen_attempts: {max_gen_attempts}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load initial data
    logger.info("Loading data...")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    X, Y = pmssm.load_pmssm_data(n_datasets=n_datasets, logger=logger, plot_dir=str(plots_dir))

    # Store full dataset for baseline random sampling (before any truncation)
    X_full, Y_full = X.clone(), Y.clone()

    # Track which indices from X_full are used for initial AL dataset
    initial_al_size = n_samples if n_samples is not None else len(X)
    initial_al_indices = torch.arange(initial_al_size)  # Indices [0, 1, ..., n_samples-1]

    if n_samples is not None:
        X = X[:n_samples]
        Y = Y[:n_samples]
        logger.info(f"Using first {n_samples} samples for initial AL training")

    logger.info(f"AL dataset shape: X={X.shape}, Y={Y.shape}")
    logger.info(f"Baseline pool shape: X_full={X_full.shape}, Y_full={Y_full.shape}")
    logger.info(f"Initial AL indices from X_full: [0, ..., {initial_al_size-1}] (will be excluded from baseline sampling)")

    # Determine if we can use parallel training (2+ GPUs for AL + baseline)
    use_parallel = torch.cuda.is_available() and torch.cuda.device_count() >= 2
    if use_parallel:
        logger.info(f"Parallel training enabled: {torch.cuda.device_count()} GPUs available")
        logger.info("  - Active Learning model on cuda:0")
        logger.info("  - Baseline model on cuda:1")
        mp.set_start_method('spawn', force=True)
    else:
        logger.info("Sequential training (need 2+ GPUs for parallel)")

    # Run active learning iterations
    all_selected_points = []
    iteration_numbers = []

    # AL metrics
    al_train_losses = []
    al_val_losses = []
    al_r2_scores = []
    al_n_train = []
    al_n_val = []

    # Baseline metrics
    baseline_train_losses = []
    baseline_val_losses = []
    baseline_r2_scores = []
    baseline_n_train = []
    baseline_n_val = []

    for iteration in range(1, n_iterations + 1):
        logger.info(f"=== Global Iteration {iteration} ===")

        # Create iteration directory for logs and plots
        iter_dir = output_dir / f"iteration_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        iter_plots_dir = iter_dir / "plots"
        iter_plots_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Iteration directory: {iter_dir}")

        # Create baseline dataset
        if iteration == 1:
            # First iteration: Baseline is an exact copy of the AL dataset.
            # Use a direct clone to guarantee byte-for-byte identity without relying
            # on index construction logic.
            X_baseline = X.clone()
            Y_baseline = Y.clone()
            assert X.shape == X_baseline.shape and torch.allclose(X, X_baseline), \
                "BUG: AL and Baseline datasets must be identical at iteration 1!"
            logger.info(f"Iteration 1: Both models use identical dataset ({len(X_baseline)} samples) — verified by allclose check")
        else:
            # Iteration 2+: Baseline grows by sampling from X_full excluding initial AL indices
            n_current = len(X)
            all_indices = torch.arange(len(X_full))
            mask = torch.ones(len(X_full), dtype=torch.bool)
            mask[initial_al_indices] = False
            available_indices = all_indices[mask]

            logger.info(f"Baseline sampling: {len(available_indices)} indices available from X_full (excluding initial {len(initial_al_indices)} AL indices)")

            n_additional = n_current - len(initial_al_indices)

            if n_additional <= len(available_indices):
                additional_indices = available_indices[torch.randperm(len(available_indices))[:n_additional]]
            else:
                logger.info(f"Baseline needs {n_additional} samples but only {len(available_indices)} available - sampling with replacement")
                additional_indices = available_indices[torch.randint(0, len(available_indices), (n_additional,))]

            baseline_indices = torch.cat([initial_al_indices, additional_indices])
            X_baseline = X_full[baseline_indices]
            Y_baseline = Y_full[baseline_indices]
            logger.info(f"Baseline dataset: {len(initial_al_indices)} initial + {n_additional} random = {len(baseline_indices)} samples")

        # Create train/val split indices based on AL dataset size
        # Use the same indices for both AL and Baseline to ensure identical dataset sizes
        idx_train, idx_val = pmssm.make_split(X, logger=logger)
        idx_train_al = idx_train
        idx_val_al = idx_val
        idx_train_base = idx_train
        idx_val_base = idx_val

        al_checkpoint_path = iter_dir / "al_model_checkpoint.pt"

        if use_parallel:
            # Train AL and Baseline in parallel on different GPUs
            al_queue = mp.Queue()
            baseline_queue = mp.Queue()

            al_process = mp.Process(
                target=train_model_worker,
                args=(1, X, Y, idx_train_al, idx_val_al, epochs, dropout, al_queue, "AL", iter_dir, iter_plots_dir, al_checkpoint_path)
            )
            baseline_process = mp.Process(
                target=train_model_worker,
                args=(2, X_baseline, Y_baseline, idx_train_base, idx_val_base, epochs, dropout, baseline_queue, "Baseline", iter_dir, iter_plots_dir)
            )

            al_process.start()
            baseline_process.start()

            al_process.join()
            baseline_process.join()

            al_results = al_queue.get()
            baseline_results = baseline_queue.get()
        else:
            # Sequential training
            logger.info("Training Active Learning model...")
            al_queue = mp.Queue()
            train_model_worker(device, X, Y, idx_train_al, idx_val_al, epochs, dropout, al_queue, "AL", iter_dir, iter_plots_dir, al_checkpoint_path)
            al_results = al_queue.get()

            logger.info("Training Baseline model (random samples)...")
            baseline_queue = mp.Queue()
            train_model_worker(device, X_baseline, Y_baseline, idx_train_base, idx_val_base, epochs, dropout, baseline_queue, "Baseline", iter_dir, iter_plots_dir)
            baseline_results = baseline_queue.get()

        # Log results
        logger.info(f"AL metrics: train_loss={al_results['best_train_loss']:.6f}, val_loss={al_results['best_val_loss']:.6f}, R²={al_results['r2_score']:.4f}")
        logger.info(f"Baseline metrics: train_loss={baseline_results['best_train_loss']:.6f}, val_loss={baseline_results['best_val_loss']:.6f}, R²={baseline_results['r2_score']:.4f}")

        # Track metrics
        iteration_numbers.append(iteration)
        al_train_losses.append(al_results['best_train_loss'])
        al_val_losses.append(al_results['best_val_loss'])
        al_r2_scores.append(al_results['r2_score'])
        al_n_train.append(len(idx_train_al))
        al_n_val.append(len(idx_val_al))
        baseline_train_losses.append(baseline_results['best_train_loss'])
        baseline_val_losses.append(baseline_results['best_val_loss'])
        baseline_r2_scores.append(baseline_results['r2_score'])
        baseline_n_train.append(len(idx_train_base))
        baseline_n_val.append(len(idx_val_base))

        # Load the AL model checkpoint saved by the worker for MC Dropout uncertainty estimation.
        # This avoids training the AL model a second time.
        stats = pmssm.compute_stats(X, Y, idx_train_al)
        model = pmssm.PMSSMTransformerTabular(
            d_model=128, nhead=4, num_layers=3, dim_feedforward=512, dropout=dropout
        )
        model.load_state_dict(torch.load(al_checkpoint_path, map_location=device))
        logger.info(f"Loaded AL model from {al_checkpoint_path} for MC Dropout uncertainty estimation")

        # Generate candidate pool and select uncertain points
        logger.info(f"Generating {n_candidates} candidate points...")
        candidates = generate_candidate_pool(n_candidates, seed=iteration)

        _, pred_var = compute_uncertainty_mc_dropout(
            model, candidates, stats, mc_samples, device, logger
        )

        top_indices = select_top_uncertain(candidates, pred_var, n_select)
        logger.info(f"Selected {len(top_indices)} most uncertain points (requested: {n_select}, available: {len(candidates)})")

        csv_path = save_selected_points(candidates, pred_var, top_indices, output_dir, iteration)
        logger.info(f"Saved selected points to {csv_path}")

        all_selected_points.append({
            "iteration": iteration,
            "points": candidates[top_indices].numpy().tolist(),
            "uncertainties": pred_var[top_indices].squeeze().numpy().tolist(),
            "al_best_val_loss": al_results['best_val_loss'],
            "al_r2_score": al_results['r2_score'],
            "baseline_best_val_loss": baseline_results['best_val_loss'],
            "baseline_r2_score": baseline_results['r2_score'],
        })

        # Generate new models if requested, with retry logic
        new_X, new_Y = None, None
        if generate_data:
            n_target = max(1, int(n_select * min_gen_fraction))
            logger.info(f"Generation target: {n_target} valid models ({min_gen_fraction*100:.0f}% of {n_select} selected, max {max_gen_attempts} attempts)")

            collected_X, collected_Y = [], []

            for attempt in range(max_gen_attempts):
                if attempt == 0:
                    # First attempt: use the already-generated candidates and saved CSV
                    attempt_candidates = candidates
                    attempt_pred_var = pred_var
                    attempt_indices = top_indices
                    attempt_csv = csv_path
                    attempt_dir = iter_dir
                else:
                    # Retry: draw a fresh random candidate pool and recompute uncertainty
                    attempt_dir = iter_dir / f"retry_{attempt:03d}"
                    attempt_dir.mkdir(parents=True, exist_ok=True)

                    attempt_seed = iteration * 1000 + attempt
                    attempt_candidates = generate_candidate_pool(n_candidates, seed=attempt_seed)
                    _, attempt_pred_var = compute_uncertainty_mc_dropout(
                        model, attempt_candidates, stats, mc_samples, device, logger
                    )
                    attempt_indices = select_top_uncertain(attempt_candidates, attempt_pred_var, n_select)

                    param_names = [p.replace("IN_", "") for p in PARAM_ORDER]
                    df = pd.DataFrame(attempt_candidates[attempt_indices].numpy(), columns=param_names)
                    df["uncertainty"] = attempt_pred_var[attempt_indices].squeeze().numpy()
                    attempt_csv = attempt_dir / "selected_points.csv"
                    df.to_csv(attempt_csv, index=False)

                logger.info(f"Generation attempt {attempt + 1}/{max_gen_attempts} ({len(attempt_indices)} points)...")
                ntuple_path = generate_models_from_csv(attempt_csv, attempt_dir, logger)

                if ntuple_path:
                    batch_X, batch_Y = load_generated_data(ntuple_path, logger)
                    if batch_X is not None and len(batch_X) > 0:
                        collected_X.append(batch_X)
                        collected_Y.append(batch_Y)

                n_collected = sum(len(x) for x in collected_X)
                logger.info(f"After attempt {attempt + 1}: {n_collected}/{n_target} target models collected")

                if n_collected >= n_target:
                    logger.info(f"Generation target reached after {attempt + 1} attempt(s)")
                    break
                if attempt < max_gen_attempts - 1:
                    logger.info(f"Below target, retrying with next most-uncertain batch...")

            if collected_X:
                new_X = torch.cat(collected_X)
                new_Y = torch.cat(collected_Y)
                logger.info(f"Total generated: {len(new_X)} valid training points across {min(attempt + 1, max_gen_attempts)} attempt(s)")
                all_selected_points[-1]["n_generated"] = len(new_X)
            else:
                logger.warning("No valid models generated after all attempts")

        # Augment training data with newly generated points for next iteration
        if new_X is not None and new_Y is not None and len(new_X) > 0:
            logger.info(f"Augmenting training data: {len(X)} + {len(new_X)} = {len(X) + len(new_X)} samples")
            X = torch.cat([X, new_X], dim=0)
            Y = torch.cat([Y, new_Y], dim=0)

    # Plot iteration metrics
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
        plot_iteration_metrics(iteration_numbers, al_metrics, baseline_metrics, output_dir, logger)
    else:
        logger.info(f"Single iteration - AL: val_loss={al_val_losses[0]:.6f}, R²={al_r2_scores[0]:.4f}")
        logger.info(f"Single iteration - Baseline: val_loss={baseline_val_losses[0]:.6f}, R²={baseline_r2_scores[0]:.4f}")

    # Save summary
    summary = {
        "timestamp": timestamp,
        "config": {
            "n_iterations": n_iterations,
            "n_candidates": n_candidates,
            "n_select": n_select,
            "mc_samples": mc_samples,
            "epochs": epochs,
            "dropout": dropout,
            "generate_data": generate_data,
        },
        "iterations": all_selected_points,
        "final_dataset_size": len(X),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("Active Learning Complete")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
