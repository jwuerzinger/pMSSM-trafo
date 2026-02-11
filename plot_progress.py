"""
plot_progress.py - Plot iteration metrics from an ongoing active learning run.

Usage:
    python plot_progress.py [--log active_learning_output/active_learning.log]
                            [--output iteration_metrics_progress.png]
"""

import re
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def parse_log(log_path):
    """Parse iteration metrics from the active learning log file."""
    iterations = []
    al_train_losses, al_val_losses, al_r2_scores = [], [], []
    baseline_train_losses, baseline_val_losses, baseline_r2_scores = [], [], []
    n_trains, n_vals = [], []

    current_iteration = None
    current_n_train = None
    current_n_val = None

    iter_re     = re.compile(r"=== Global Iteration (\d+) ===")
    split_re    = re.compile(r"Have n_train=(\d+), n_val=(\d+)")
    al_re       = re.compile(r"AL metrics: train_loss=([\d.]+), val_loss=([\d.]+), R²=([-\d.]+)")
    base_re     = re.compile(r"Baseline metrics: train_loss=([\d.]+), val_loss=([\d.]+), R²=([-\d.]+)")

    with open(log_path) as f:
        for line in f:
            m = iter_re.search(line)
            if m:
                current_iteration = int(m.group(1))
                current_n_train = None
                current_n_val = None
                continue

            m = split_re.search(line)
            if m and current_iteration is not None:
                current_n_train = int(m.group(1))
                current_n_val = int(m.group(2))
                continue

            m = al_re.search(line)
            if m and current_iteration is not None:
                al_train_losses.append(float(m.group(1)))
                al_val_losses.append(float(m.group(2)))
                al_r2_scores.append(float(m.group(3)))
                continue

            m = base_re.search(line)
            if m and current_iteration is not None:
                baseline_train_losses.append(float(m.group(1)))
                baseline_val_losses.append(float(m.group(2)))
                baseline_r2_scores.append(float(m.group(3)))
                # Both AL and Baseline metrics are now available: record iteration
                iterations.append(current_iteration)
                n_trains.append(current_n_train)
                n_vals.append(current_n_val)
                current_iteration = None  # reset until next iteration header

    return dict(
        iterations=iterations,
        al_train_losses=al_train_losses,
        al_val_losses=al_val_losses,
        al_r2_scores=al_r2_scores,
        baseline_train_losses=baseline_train_losses,
        baseline_val_losses=baseline_val_losses,
        baseline_r2_scores=baseline_r2_scores,
        n_trains=n_trains,
        n_vals=n_vals,
    )


def plot(data, output_path):
    iters = data["iterations"]
    if not iters:
        print("No completed iterations found in log.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax1, ax2, ax3 = axes

    # --- Loss ---
    ax1.plot(iters, data["al_train_losses"],       "b-",  lw=2, marker="o", ms=6, label="AL Train")
    ax1.plot(iters, data["al_val_losses"],         "b--", lw=2, marker="s", ms=6, label="AL Validation")
    ax1.plot(iters, data["baseline_train_losses"], "r-",  lw=2, marker="o", ms=6, label="Baseline Train")
    ax1.plot(iters, data["baseline_val_losses"],   "r--", lw=2, marker="s", ms=6, label="Baseline Validation")
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Best Loss (MSE)", fontsize=12)
    ax1.set_title("Train/Validation Loss vs Iteration", fontsize=14)
    ax1.set_yscale("log")
    ax1.set_xticks(iters)
    ax1.grid(True, which="major", alpha=0.3)
    ax1.grid(True, which="minor", alpha=0.15)
    ax1.legend(fontsize=9)

    # --- R² ---
    ax2.plot(iters, data["al_r2_scores"],       "b-", lw=2, marker="o", ms=6, label="Active Learning")
    ax2.plot(iters, data["baseline_r2_scores"], "r-", lw=2, marker="o", ms=6, label="Baseline (Random)")
    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("R² Score", fontsize=12)
    ax2.set_title("R² Score vs Iteration", fontsize=14)
    ax2.set_xticks(iters)
    ax2.grid(True, alpha=0.3)
    all_r2 = data["al_r2_scores"] + data["baseline_r2_scores"]
    finite_r2 = [v for v in all_r2 if v == v and abs(v) != float("inf")]
    if finite_r2:
        ax2.set_ylim(min(0, min(finite_r2) - 0.1), 1.05)
    ax2.legend(fontsize=10)

    # --- Dataset sizes ---
    ax3.plot(iters, data["n_trains"], "b-",  lw=2, marker="o", ms=6, label="Train")
    ax3.plot(iters, data["n_vals"],   "b--", lw=2, marker="s", ms=6, label="Val")
    ax3.set_xlabel("Iteration", fontsize=12)
    ax3.set_ylabel("Number of Samples", fontsize=12)
    ax3.set_title("Dataset Size vs Iteration", fontsize=14)
    ax3.set_xticks(iters)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)

    fig.suptitle(
        f"Active Learning Progress  —  {len(iters)} iteration(s) completed",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}  ({len(iters)} iteration(s))")


def main():
    parser = argparse.ArgumentParser(description="Plot AL progress from log file.")
    parser.add_argument(
        "--log", default="active_learning_output/active_learning.log",
        help="Path to active_learning.log",
    )
    parser.add_argument(
        "--output", default="active_learning_output/iteration_metrics_progress.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return

    data = parse_log(log_path)
    plot(data, args.output)


if __name__ == "__main__":
    main()
