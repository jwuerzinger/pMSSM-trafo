#!/usr/bin/env python
"""
Plot input parameter and target histograms for the MCMC dataset,
matching the format of the AL iteration plots.
"""

import glob
from pathlib import Path

import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pmssm.config import PARAM_ORDER, TARGET_CONFIG


def load_mcmc_data(data_dir="data/18732958", target="DMRD"):
    """Load MCMC ROOT data with the same filters as load_pmssm_data."""
    files = sorted(glob.glob(f"{data_dir}/*.root"))
    print(f"Found {len(files)} ROOT files in {data_dir}")

    trees = [uproot.open(f)["susy"] for f in files]

    X_raw = np.column_stack([
        np.concatenate([t[b].array(library="np") for t in trees])
        for b in PARAM_ORDER
    ])

    target_branch = TARGET_CONFIG[target]["branch"]
    Y_raw = np.concatenate([t[target_branch].array(library="np") for t in trees])
    sp_mh = np.concatenate([t["SP_m_h"].array(library="np") for t in trees])

    mask = (Y_raw > 0) & (Y_raw < 1.0) & (sp_mh != -1)
    print(f"Filter ({target_branch} > 0 & < 1 & SP_m_h != -1): "
          f"{mask.sum()} / {len(Y_raw)} samples kept")

    return X_raw[mask], Y_raw[mask]


def main():
    output_dir = Path("plots/mcmc_histograms_18732958")
    output_dir.mkdir(parents=True, exist_ok=True)

    X, Y = load_mcmc_data()

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
    print(f"Saved {plot_path}")

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
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
