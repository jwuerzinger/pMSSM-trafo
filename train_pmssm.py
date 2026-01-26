import os
# Set GPU before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import warnings
# Suppress nested tensor warning (expected with norm_first=True)
warnings.filterwarnings('ignore', message='.*enable_nested_tensor.*')

import pmssm

from pathlib import Path
import sys
from datetime import datetime
import logging
import structlog

import click
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import multiprocessing as mp


def setup_logging(timestamp):
    """Set up structlog to write to both file and console."""
    Path("logs/").mkdir(parents=True, exist_ok=True)
    log_file = f"logs/training_{timestamp}.log"

    # Configure standard library logging to write to file and console
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Configure structlog for human-readable output
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

    # Set up formatters for console and file
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=False),
    )

    # Apply formatter to handlers
    for handler in logging.root.handlers:
        handler.setFormatter(formatter)

    return log_file, structlog.get_logger()


def train_transformer(gpu_id, train_dataset, val_dataset, stats, epochs, plots_dir, log_file, early_stopping=False, patience=500):
    """Train PMSSMTransformer on specified GPU."""
    # Set up device
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    # Set up logging for this process
    _, logger = setup_logging(Path(log_file).stem.replace("training_", "transformer_"))

    logger.info("="*60)
    logger.info(f"Training PMSSMTransformer on {device}")
    logger.info("="*60)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Create model
    model = pmssm.PMSSMTransformer(
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.0,
        use_prenorm=True,
    )

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    # Train
    train_losses, val_losses = pmssm.train_with_validation(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device=device,
        epochs=epochs,
        early_stopping=early_stopping,
        patience=patience,
        scheduler=scheduler,
        grad_clip=1.0,
        logger=logger,
    )

    # Plot results
    pmssm.plot_losses(train_losses, val_losses, model, plot_dir=plots_dir)
    pmssm.compare_random_predictions(model, stats=stats, subset=train_dataset, mode='train', device=device, n_points=10, logger=logger)
    pmssm.compare_random_predictions(model, stats=stats, subset=val_dataset, mode='validation', device=device, n_points=3, logger=logger)
    pmssm.scatter_true_vs_pred(model, stats=stats, subset=train_dataset, mode='train', device=device, plot_dir=plots_dir)
    pmssm.scatter_true_vs_pred(model, stats=stats, subset=val_dataset, mode='validation', device=device, plot_dir=plots_dir)
    pmssm.hist_true_vs_pred(model, stats=stats, subset=train_dataset, mode='train', device=device, plot_dir=plots_dir)
    pmssm.hist_true_vs_pred(model, stats=stats, subset=val_dataset, mode='validation', device=device, plot_dir=plots_dir)

    logger.info("PMSSMTransformer training complete")


def train_transformer_tabular(gpu_id, train_dataset, val_dataset, stats, epochs, plots_dir, log_file, early_stopping=False, patience=500):
    """Train PMSSMTransformerTabular on specified GPU."""
    # Set up device
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    # Set up logging for this process
    _, logger = setup_logging(Path(log_file).stem.replace("training_", "transformer_tabular_"))

    logger.info("="*60)
    logger.info(f"Training PMSSMTransformerTabular on {device}")
    logger.info("="*60)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Create model
    model = pmssm.PMSSMTransformerTabular(
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.0,
    )

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    # Train
    train_losses, val_losses = pmssm.train_with_validation(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device=device,
        epochs=epochs,
        early_stopping=early_stopping,
        patience=patience,
        scheduler=scheduler,
        grad_clip=1.0,
        logger=logger,
    )

    # Plot results
    pmssm.plot_losses(train_losses, val_losses, model, plot_dir=plots_dir)
    pmssm.compare_random_predictions(model, stats=stats, subset=train_dataset, mode='train', device=device, n_points=10, logger=logger)
    pmssm.compare_random_predictions(model, stats=stats, subset=val_dataset, mode='validation', device=device, n_points=3, logger=logger)
    pmssm.scatter_true_vs_pred(model, stats=stats, subset=train_dataset, mode='train', device=device, plot_dir=plots_dir)
    pmssm.scatter_true_vs_pred(model, stats=stats, subset=val_dataset, mode='validation', device=device, plot_dir=plots_dir)
    pmssm.hist_true_vs_pred(model, stats=stats, subset=train_dataset, mode='train', device=device, plot_dir=plots_dir)
    pmssm.hist_true_vs_pred(model, stats=stats, subset=val_dataset, mode='validation', device=device, plot_dir=plots_dir)

    logger.info("PMSSMTransformerTabular training complete")


def train_mlp(gpu_id, train_dataset, val_dataset, stats, epochs, plots_dir, log_file, early_stopping=False, patience=500):
    """Train MLP Baseline on specified GPU."""
    # Set up device
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    # Set up logging for this process
    _, logger = setup_logging(Path(log_file).stem.replace("training_", "mlp_"))

    logger.info("="*60)
    logger.info(f"Training MLP Baseline on {device}")
    logger.info("="*60)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Create model
    model = pmssm.PMSSMFeedForward(
        d_model=64,
        num_layers=4,
        dim_feedforward=256*2,
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = nn.MSELoss()

    # Train
    train_losses, val_losses = pmssm.train_with_validation(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device=device,
        epochs=epochs,
        early_stopping=early_stopping,
        patience=patience,
        logger=logger,
    )

    # Plot results
    pmssm.plot_losses(train_losses, val_losses, model, plot_dir=plots_dir)
    pmssm.compare_random_predictions(model, stats=stats, subset=train_dataset, mode='train', device=device, n_points=10, logger=logger)
    pmssm.compare_random_predictions(model, stats=stats, subset=val_dataset, mode='validation', device=device, n_points=3, logger=logger)
    pmssm.scatter_true_vs_pred(model, stats=stats, subset=train_dataset, mode='train', device=device, plot_dir=plots_dir)
    pmssm.scatter_true_vs_pred(model, stats=stats, subset=val_dataset, mode='validation', device=device, plot_dir=plots_dir)
    pmssm.hist_true_vs_pred(model, stats=stats, subset=train_dataset, mode='train', device=device, plot_dir=plots_dir)
    pmssm.hist_true_vs_pred(model, stats=stats, subset=val_dataset, mode='validation', device=device, plot_dir=plots_dir)

    logger.info("MLP Baseline training complete")


@click.command()
@click.option('--testing', is_flag=True, help="Run in testing mode (uses n_datasets=3, n_samples=30).")
@click.option('--epochs', default=2_000, type=int, help="Number of training epochs (default: 2000).")
@click.option('--n-datasets', default=None, type=int, help="Number of datasets to load (-1 for all, overrides --testing).")
@click.option('--n-samples', default=None, type=int, help="Number of samples per dataset (None for all, overrides --testing).")
@click.option('--no-parallel', is_flag=True, help="Disable parallel training (train models sequentially even if multiple GPUs available).")
@click.option('--early-stopping', is_flag=True, help="Enable early stopping based on validation loss.")
@click.option('--patience', default=500, type=int, help="Early stopping patience (epochs without improvement before stopping, default: 500).")
def main(testing, epochs, n_datasets, n_samples, no_parallel, early_stopping, patience):
    # Create timestamped directories for logs and plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file, logger = setup_logging(timestamp)

    # Create plot directory with same timestamp as log file
    plots_dir = Path(f"plots/run_{timestamp}")
    plots_dir.mkdir(parents=True, exist_ok=True)

    if pmssm.running_in_notebook():
        logger.info("Running in Jupyter")
    else:
        logger.info("Running as a script")

    logger.info("="*60)
    logger.info(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Plots directory: {plots_dir}")
    logger.info("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Check GPU availability
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"CUDA available: {num_gpus} GPUs visible")
        for i in range(num_gpus):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("CUDA not available - using CPU")

    # Determine n_datasets: explicit option > testing mode > full data
    if n_datasets is None:
        n_datasets = 3 if testing else -1

    # Determine n_samples: explicit option > testing mode > all samples
    if n_samples is None:
        n_samples = 30 if testing else None

    logger.info(f"Loading data: n_datasets={n_datasets}, n_samples={n_samples if n_samples else 'all'}")

    # Load once
    X, Y = pmssm.load_pmssm_data(n_datasets=n_datasets, logger=logger, plot_dir=plots_dir)

    # Split once
    idx_train, idx_val = pmssm.make_split(X, logger=logger)

    # Stats from training only
    stats = pmssm.compute_stats(X, Y, idx_train)

    # Datasets
    train_dataset = pmssm.PMSSMDataset(X, Y, idx_train, stats, n_samples=n_samples)
    val_dataset   = pmssm.PMSSMDataset(X, Y, idx_val, stats, n_samples=n_samples)

    # ========================================
    # Parallel Training of All Models
    # ========================================
    logger.info("")
    logger.info("="*60)

    # Determine if we can and should use parallel training
    use_parallel = (not no_parallel and
                    torch.cuda.is_available() and
                    torch.cuda.device_count() >= 3)

    if use_parallel:
        logger.info("Starting PARALLEL training on 3 GPUs")
        logger.info("  - PMSSMTransformer on cuda:0")
        logger.info("  - PMSSMTransformerTabular on cuda:1")
        logger.info("  - MLP Baseline on cuda:2")
        logger.info("="*60)

        # Start all three training processes in parallel
        mp.set_start_method('spawn', force=True)

        process1 = mp.Process(
            target=train_transformer,
            args=(0, train_dataset, val_dataset, stats, epochs, plots_dir, log_file, early_stopping, patience)
        )
        process2 = mp.Process(
            target=train_transformer_tabular,
            args=(1, train_dataset, val_dataset, stats, epochs, plots_dir, log_file, early_stopping, patience)
        )
        process3 = mp.Process(
            target=train_mlp,
            args=(2, train_dataset, val_dataset, stats, epochs, plots_dir, log_file, early_stopping, patience)
        )

        process1.start()
        process2.start()
        process3.start()

        # Wait for all three to complete
        process1.join()
        process2.join()
        process3.join()

        logger.info("="*60)
        logger.info("Parallel training complete!")
        logger.info("="*60)
    else:
        # Sequential training
        if no_parallel:
            logger.info("Sequential training (--no-parallel flag set)")
        elif not torch.cuda.is_available():
            logger.info("Sequential training (CUDA not available)")
        else:
            logger.info(f"Sequential training (only {torch.cuda.device_count()} GPU(s) available, need 3 for parallel)")
        logger.info("="*60)

        # Determine device for sequential training
        if torch.cuda.is_available():
            device_seq = "cuda:0"
            logger.info(f"Using {device_seq} for sequential training")
        else:
            device_seq = "cpu"
            logger.info("Using CPU for sequential training")

        # Train PMSSMTransformer
        logger.info("")
        logger.info("Training PMSSMTransformer...")
        train_transformer(0 if torch.cuda.is_available() else "cpu", train_dataset, val_dataset, stats, epochs, plots_dir, log_file, early_stopping, patience)

        # Train PMSSMTransformerTabular
        logger.info("")
        logger.info("Training PMSSMTransformerTabular...")
        train_transformer_tabular(0 if torch.cuda.is_available() else "cpu", train_dataset, val_dataset, stats, epochs, plots_dir, log_file, early_stopping, patience)

        # Train MLP Baseline
        logger.info("")
        logger.info("Training MLP Baseline...")
        train_mlp(0 if torch.cuda.is_available() else "cpu", train_dataset, val_dataset, stats, epochs, plots_dir, log_file, early_stopping, patience)

    # ========================================
    # Summary
    # ========================================
    logger.info("")
    logger.info("="*60)
    logger.info(f"Training Complete - Check {plots_dir} for results")
    logger.info("="*60)
    logger.info("")
    logger.info("Compare the validation loss curves to see which model")
    logger.info("performs best on your pMSSM regression task.")
    logger.info("")
    logger.info("Models trained:")
    logger.info("  1. PMSSMTransformer (improved)")
    logger.info("  2. PMSSMTransformerTabular (tabular-specific)")
    logger.info("  3. PMSSMFeedForward (MLP baseline)")

    logger.info("")
    logger.info("="*60)
    logger.info(f"Training finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*60)

if __name__ == "__main__":
    main()