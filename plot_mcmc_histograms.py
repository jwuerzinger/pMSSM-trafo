#!/usr/bin/env python
"""
Plot input parameter and target histograms for the MCMC dataset,
matching the format of the AL iteration plots.
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pmssm.config import PARAM_ORDER, TARGET_CONFIG
from pmssm.data import load_mcmc_data

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-8s] %(message)s")

    output_dir = Path("plots/mcmc_histograms_19250082")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading MCMC data...")
    X_t, Y_t = load_mcmc_data(logger=logger)
    # Convert tensors to numpy for plotting
    X, Y = X_t.numpy(), Y_t.squeeze(-1).numpy()
    logger.info(f"Loaded {len(X)} MCMC samples with {X.shape[1]} parameters")

    # --- Input parameter histograms ---
    param_names = [p.replace("IN_", "") for p in PARAM_ORDER]
    n_params = len(param_names)
    n_cols = 5
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    fig.suptitle("MCMC Dataset — Input Parameter Distributions", fontsize=16, y=1.01)
    axes = axes.flatten()

    for i, param_name in enumerate(param_names):
        ax = axes[i]
        ax.hist(X[:, i], bins=30, alpha=0.7, color="green", density=True)
        ax.set_xlabel(param_name, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(param_name, fontsize=11)
        ax.grid(True, alpha=0.3)

    for i in range(n_params, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plot_path = output_dir / "mcmc_input_histograms.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {plot_path}")

    # --- Target histogram ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.hist(Y, bins=50, alpha=0.7, color="green", density=True)
    ax.axvline(TARGET_CONFIG["DMRD"]["true_value"], color="red", linestyle="--",
               linewidth=1.5, label=f'Ωh² = {TARGET_CONFIG["DMRD"]["true_value"]}')
    ax.set_xlabel("MO_Omega (Ωh²)", fontsize=12)
    ax.set_ylabel("Density (log scale)", fontsize=12)
    ax.set_yscale("log")
    ax.set_title("MCMC Dataset — Target Distribution", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plot_path = output_dir / "mcmc_target_histogram.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
