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

import click
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

@click.command()
@click.option('--testing', is_flag=True, help="Run in testing mode (uses n_datasets=3, n_samples=30).")
@click.option('--epochs', default=2_000, type=int, help="Number of training epochs (default: 2000).")
@click.option('--n-datasets', default=None, type=int, help="Number of datasets to load (-1 for all, overrides --testing).")
@click.option('--n-samples', default=None, type=int, help="Number of samples per dataset (None for all, overrides --testing).")
def main(testing, epochs, n_datasets, n_samples):
    # Create directories
    Path("plots/").mkdir(parents=True, exist_ok=True)

    print("="*60)
    print(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Set device to:", device)
    print(f"Using GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    # Determine n_datasets: explicit option > testing mode > full data
    if n_datasets is None:
        n_datasets = 3 if testing else -1

    # Determine n_samples: explicit option > testing mode > all samples
    if n_samples is None:
        n_samples = 30 if testing else None

    print(f"Loading data: n_datasets={n_datasets}, n_samples={n_samples if n_samples else 'all'}")

    # Load once
    X, Y = pmssm.load_pmssm_data(n_datasets=n_datasets)

    # Split once
    idx_train, idx_val = pmssm.make_split(X)

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
    print("\n" + "="*60)
    print("Training Improved PMSSMTransformer")
    print("="*60)

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
    )

    pmssm.plot_losses(train_losses, val_losses, model)

    # compare training points
    pmssm.compare_random_predictions(model, stats=stats, subset=train_dataset, mode='train', device=device, n_points=10)
    # compare validation points:
    pmssm.compare_random_predictions(model, stats=stats, subset=val_dataset, mode='validation', device=device, n_points=3)

    pmssm.scatter_true_vs_pred(model, stats=stats, subset=train_dataset, mode='train', device=device)
    pmssm.scatter_true_vs_pred(model, stats=stats, subset=val_dataset, mode='validation', device=device)

    pmssm.hist_true_vs_pred(model, stats=stats, subset=train_dataset, mode='train', device=device)
    pmssm.hist_true_vs_pred(model, stats=stats, subset=val_dataset, mode='validation', device=device)

    # ========================================
    # Test 2: PMSSMTransformerTabular
    # ========================================
    print("\n" + "="*60)
    print("Training PMSSMTransformerTabular (Tabular-Specific)")
    print("="*60)

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
    )

    pmssm.plot_losses(train_losses, val_losses, model)

    # compare training points
    pmssm.compare_random_predictions(model, stats=stats, subset=train_dataset, mode='train', device=device, n_points=10)
    # compare validation points:
    pmssm.compare_random_predictions(model, stats=stats, subset=val_dataset, mode='validation', device=device, n_points=3)

    pmssm.scatter_true_vs_pred(model, stats=stats, subset=train_dataset, mode='train', device=device)
    pmssm.scatter_true_vs_pred(model, stats=stats, subset=val_dataset, mode='validation', device=device)

    pmssm.hist_true_vs_pred(model, stats=stats, subset=train_dataset, mode='train', device=device)
    pmssm.hist_true_vs_pred(model, stats=stats, subset=val_dataset, mode='validation', device=device)

    # ========================================
    # Test 3: MLP Baseline
    # ========================================
    print("\n" + "="*60)
    print("Training MLP Baseline")
    print("="*60)

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
        early_stopping=False
    )

    pmssm.plot_losses(train_losses, val_losses, model)

    # compare random training & validation points points
    pmssm.compare_random_predictions(model, stats=stats, subset=train_dataset, mode='train', device=device, n_points=10)
    pmssm.compare_random_predictions(model, stats=stats, subset=val_dataset, mode='validation', device=device, n_points=3)
    
    # scatterplot for training & validation sample
    pmssm.scatter_true_vs_pred(model, stats=stats, subset=train_dataset, mode='train', device=device)
    pmssm.scatter_true_vs_pred(model, stats=stats, subset=val_dataset, mode='validation', device=device)
    
    # 2D hists for training & validation samples
    pmssm.hist_true_vs_pred(model, stats=stats, subset=train_dataset, mode='train', device=device)
    pmssm.hist_true_vs_pred(model, stats=stats, subset=val_dataset, mode='validation', device=device)

    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*60)
    print("Training Complete - Check plots/ directory for results")
    print("="*60)
    print("\nCompare the validation loss curves to see which model")
    print("performs best on your pMSSM regression task.")
    print("\nModels trained:")
    print("  1. PMSSMTransformer (improved)")
    print("  2. PMSSMTransformerTabular (tabular-specific)")
    print("  3. PMSSMFeedForward (MLP baseline)")

    print("\n" + "="*60)
    print(f"Training finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

if __name__ == "__main__":
    main()