"""
PyTorch Dataset classes for pMSSM data.

This module provides Dataset wrappers that handle normalization
and sampling for training neural network models.
"""

import torch
from torch.utils.data import Dataset


class Dummy_PMSSMDataset(Dataset):
    """
    Dummy dataset for testing and development.

    Generates synthetic data with nonlinear relationships between
    inputs and target to simulate physics-inspired patterns.

    Args:
        n_samples: Number of samples to generate (default: 10,000)
    """

    def __init__(self, n_samples=10_000):
        super().__init__()
        self.x = torch.randn(n_samples, 19)

        # Dummy "physics-inspired" target: nonlinear function + noise
        self.y = (
            torch.sum(self.x[:, :5] ** 2, dim=1)
            + torch.sin(self.x[:, 5:10]).sum(dim=1)
            + 0.1 * torch.randn(n_samples)
        )
        self.y = self.y.unsqueeze(1)  # (N, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class PMSSMDataset(Dataset):
    """
    pMSSM dataset with z-score normalization.

    Applies z-score normalization using training set statistics:
    X_norm = (X - mean_X) / std_X
    Y_norm = (Y - mean_Y) / std_Y

    Args:
        X: Input tensor (N, 19) in physical units
        Y: Target tensor (N, 1) in physical units
        indices: Indices to select from X and Y
        stats: Tuple of (mean_X, std_X, mean_Y, std_Y)
        n_samples: Optional limit on number of samples to use
    """

    def __init__(self, X, Y, indices, stats, n_samples=None):
        super().__init__()

        mean_X, std_X, mean_Y, std_Y = stats

        # Normalize using training statistics
        X = (X[indices] - mean_X) / std_X
        Y = (Y[indices] - mean_Y) / std_Y

        if n_samples is not None:
            X = X[:n_samples]
            Y = Y[:n_samples]

        self.x = X
        self.y = Y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
