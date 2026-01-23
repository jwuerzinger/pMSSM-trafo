import os
# Set GPU before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

@click.command()
@click.option('--testing', is_flag=True, help="Run in testing mode (uses n_datasets=3, n_samples=30).")
@click.option('--epochs', default=2_000, type=int, help="Number of training epochs (default: 2000).")
@click.option('--n-datasets', default=None, type=int, help="Number of datasets to load (-1 for all, overrides --testing).")
@click.option('--n-samples', default=None, type=int, help="Number of samples per dataset (None for all, overrides --testing).")
def main(testing, epochs, n_datasets, n_samples):
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
    logger.info(f"Set device to: {device}")
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

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

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=256, shuffle=False
    )

    # ========================================
    # Test 1: Improved PMSSMTransformer
    # ========================================
    logger.info("")
    logger.info("="*60)
    logger.info("Training Improved PMSSMTransformer")
    logger.info("="*60)

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

    train_losses, val_losses = pmssm.train_with_validation(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device=device,
        epochs=epochs,
        early_stopping=False,
        scheduler=scheduler,
        grad_clip=1.0,
        logger=logger,
    )

    pmssm.plot_losses(train_losses, val_losses, model, plot_dir=plots_dir)

    # compare training points
    pmssm.compare_random_predictions(model, stats=stats, subset=train_dataset, mode='train', device=device, n_points=10, logger=logger)
    # compare validation points:
    pmssm.compare_random_predictions(model, stats=stats, subset=val_dataset, mode='validation', device=device, n_points=3, logger=logger)

    pmssm.scatter_true_vs_pred(model, stats=stats, subset=train_dataset, mode='train', device=device, plot_dir=plots_dir)
    pmssm.scatter_true_vs_pred(model, stats=stats, subset=val_dataset, mode='validation', device=device, plot_dir=plots_dir)

    pmssm.hist_true_vs_pred(model, stats=stats, subset=train_dataset, mode='train', device=device, plot_dir=plots_dir)
    pmssm.hist_true_vs_pred(model, stats=stats, subset=val_dataset, mode='validation', device=device, plot_dir=plots_dir)

    # ========================================
    # Test 2: PMSSMTransformerTabular
    # ========================================
    logger.info("")
    logger.info("="*60)
    logger.info("Training PMSSMTransformerTabular (Tabular-Specific)")
    logger.info("="*60)

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

    train_losses, val_losses = pmssm.train_with_validation(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device=device,
        epochs=epochs,
        early_stopping=False,
        scheduler=scheduler,
        grad_clip=1.0,
        logger=logger,
    )

    pmssm.plot_losses(train_losses, val_losses, model, plot_dir=plots_dir)

    # compare training points
    pmssm.compare_random_predictions(model, stats=stats, subset=train_dataset, mode='train', device=device, n_points=10, logger=logger)
    # compare validation points:
    pmssm.compare_random_predictions(model, stats=stats, subset=val_dataset, mode='validation', device=device, n_points=3, logger=logger)

    pmssm.scatter_true_vs_pred(model, stats=stats, subset=train_dataset, mode='train', device=device, plot_dir=plots_dir)
    pmssm.scatter_true_vs_pred(model, stats=stats, subset=val_dataset, mode='validation', device=device, plot_dir=plots_dir)

    pmssm.hist_true_vs_pred(model, stats=stats, subset=train_dataset, mode='train', device=device, plot_dir=plots_dir)
    pmssm.hist_true_vs_pred(model, stats=stats, subset=val_dataset, mode='validation', device=device, plot_dir=plots_dir)

    # ========================================
    # Test 3: MLP Baseline
    # ========================================
    logger.info("")
    logger.info("="*60)
    logger.info("Training MLP Baseline")
    logger.info("="*60)

    # train MLP
    model = pmssm.PMSSMFeedForward(
        d_model = 64,
        num_layers = 4,
        dim_feedforward = 256*2,
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = nn.MSELoss()

    train_losses, val_losses = pmssm.train_with_validation(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device=device,
        epochs=epochs,
        early_stopping=False,
        logger=logger,
    )

    pmssm.plot_losses(train_losses, val_losses, model, plot_dir=plots_dir)

    # compare random training & validation points points
    pmssm.compare_random_predictions(model, stats=stats, subset=train_dataset, mode='train', device=device, n_points=10, logger=logger)
    pmssm.compare_random_predictions(model, stats=stats, subset=val_dataset, mode='validation', device=device, n_points=3, logger=logger)
    
    # scatterplot for training & validation sample
    pmssm.scatter_true_vs_pred(model, stats=stats, subset=train_dataset, mode='train', device=device, plot_dir=plots_dir)
    pmssm.scatter_true_vs_pred(model, stats=stats, subset=val_dataset, mode='validation', device=device, plot_dir=plots_dir)
    
    # 2D hists for training & validation samples
    pmssm.hist_true_vs_pred(model, stats=stats, subset=train_dataset, mode='train', device=device, plot_dir=plots_dir)
    pmssm.hist_true_vs_pred(model, stats=stats, subset=val_dataset, mode='validation', device=device, plot_dir=plots_dir)

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