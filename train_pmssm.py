import pmssm

from pathlib import Path

import os
import click
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

@click.command()
@click.option('--testing', is_flag=True, help="Run in testing mode only (less data).")
def main(testing):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    print("Set device to:", device)

    # Create plotting dir if needed:
    Path("plots/").mkdir(parents=True, exist_ok=True)

    # Load once
    n_datasets=3 if testing else -1
    X, Y = pmssm.load_pmssm_data(n_datasets=n_datasets)

    # Split once
    idx_train, idx_val = pmssm.make_split(X)

    # Stats from training only
    stats = pmssm.compute_stats(X, Y, idx_train)

    # Datasets - testimg
    n_samples = 30 if testing else None
    train_dataset = pmssm.PMSSMDataset(X, Y, idx_train, stats, n_samples=n_samples)
    val_dataset   = pmssm.PMSSMDataset(X, Y, idx_val, stats, n_samples=n_samples)
    # datasets - full
    # train_dataset = pmssm.PMSSMDataset(X, Y, idx_train, stats)
    # val_dataset   = pmssm.PMSSMDataset(X, Y, idx_val, stats)

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=256, shuffle=False
    )

    model = pmssm.PMSSMTransformer(
        d_model=16,
        nhead=1,
        num_layers=2,
        dim_feedforward=64,
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
        epochs=5_000,
        early_stopping=False
    )

    pmssm.plot_losses(train_losses, val_losses, model)

    # compare training points
    pmssm.compare_random_predictions(
        model,
        stats=stats,
        subset=train_dataset,
        mode='train',
        device=device,
        n_points=10,
    )

    # compare validation points:
    pmssm.compare_random_predictions(
        model,
        stats=stats,
        subset=val_dataset,
        mode='validation',
        device=device,
        n_points=3,
    )

    pmssm.scatter_true_vs_pred(
        model,
        stats=stats,
        subset=train_dataset,
        mode='train',
        device=device
    )

    pmssm.scatter_true_vs_pred(
        model,
        stats=stats,
        subset=val_dataset,
        mode='validation',
        device=device
    )

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
        epochs=2_000,
        early_stopping=False
    )

    pmssm.plot_losses(train_losses, val_losses, model)

    # compare training points
    pmssm.compare_random_predictions(
        model,
        stats=stats,
        subset=train_dataset,
        mode='train',
        device=device,
        n_points=10,
    )

    # compare validation points:
    pmssm.compare_random_predictions(
        model,
        stats=stats,
        subset=val_dataset,
        mode='validation',
        device=device,
        n_points=3,
    )

    pmssm.scatter_true_vs_pred(
        model,
        stats=stats,
        subset=train_dataset,
        mode='train',
        device=device
    )

    pmssm.scatter_true_vs_pred(
        model,
        stats=stats,
        subset=val_dataset,
        mode='validation',
        device=device
    )
    
if __name__ == "__main__":
    main()