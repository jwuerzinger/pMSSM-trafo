"""
plot_progress.py - Plot iteration metrics from an ongoing active learning run.

Works with both the transformer pipeline (active_learning.py) and the GP
pipeline (active_learning_gp.py).

Usage:
    python plot_progress.py [--log active_learning_output/active_learning.log]
                            [--output iteration_metrics_progress.png]

    # Auto-detect GP output directory:
    python plot_progress.py --log active_learning_gp_output/active_learning.log
"""

import re
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def _read_worker_log_sizes(log_dir, iteration):
    """Read n_train, n_val from the per-iteration AL worker log (GP pipeline).

    The GP pipeline logs 'Training set size: N, Validation set size: M'
    in each iteration's al_training.log.
    """
    worker_log = log_dir / f"iteration_{iteration:03d}" / "al_training.log"
    if not worker_log.exists():
        return None, None
    size_re = re.compile(
        r"Training set size:\s*(\d+),\s*Validation set size:\s*(\d+)"
    )
    with open(worker_log) as f:
        for line in f:
            m = size_re.search(line)
            if m:
                return int(m.group(1)), int(m.group(2))
    return None, None


def parse_log(log_path):
    """Parse iteration metrics from an active learning log file.

    Auto-detects transformer vs GP pipeline from the iteration header format.
    """
    iterations = []
    al_train_losses, al_val_losses, al_r2_scores = [], [], []
    baseline_train_losses, baseline_val_losses, baseline_r2_scores = [], [], []
    n_trains, n_vals = [], []

    current_iteration = None
    current_n_train = None
    current_n_val = None
    pipeline = None  # auto-detected: "transformer" or "gp"

    # Iteration headers — accept both formats
    iter_re = re.compile(
        r"=== (?:Global|GP Active Learning) Iteration (\d+) ==="
    )
    # Dataset size — transformer main log
    split_re = re.compile(r"Split: n_train=(\d+), n_val=(\d+)")
    # Dataset size — GP sequential mode
    gp_seq_re = re.compile(r"Training AL \S+ model \((\d+) train, (\d+) val\)")
    # Metric lines (same format for both pipelines)
    al_re = re.compile(
        r"AL metrics: train_loss=([\d.]+), val_loss=([\d.]+), "
        r"R²=([-\d.]+)"
    )
    base_re = re.compile(
        r"Baseline metrics: train_loss=([\d.]+), val_loss=([\d.]+), "
        r"R²=([-\d.]+)"
    )
    # Auto-detect pipeline type
    gp_header_re = re.compile(r"GP Active Learning")

    log_dir = log_path.parent

    with open(log_path) as f:
        for line in f:
            # Auto-detect pipeline on first match
            if pipeline is None and gp_header_re.search(line):
                pipeline = "gp"

            m = iter_re.search(line)
            if m:
                if pipeline is None:
                    pipeline = "transformer"
                current_iteration = int(m.group(1))
                current_n_train = None
                current_n_val = None
                continue

            # Transformer: "Have n_train=N, n_val=N"
            m = split_re.search(line)
            if m and current_iteration is not None:
                current_n_train = int(m.group(1))
                current_n_val = int(m.group(2))
                continue

            # GP sequential: "Training AL exact_gp model (N train, M val)..."
            m = gp_seq_re.search(line)
            if m and current_iteration is not None and current_n_train is None:
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

                # GP parallel mode: sizes aren't in main log, read worker log
                if current_n_train is None and pipeline == "gp":
                    current_n_train, current_n_val = _read_worker_log_sizes(
                        log_dir, current_iteration
                    )

                iterations.append(current_iteration)
                n_trains.append(current_n_train)
                n_vals.append(current_n_val)
                current_iteration = None

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
        pipeline=pipeline or "transformer",
    )


def plot(data, output_path):
    iters = data["iterations"]
    if not iters:
        print("No completed iterations found in log.")
        return

    pipeline = data["pipeline"]
    pipeline_label = "GP" if pipeline == "gp" else "Transformer"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax1, ax2, ax3 = axes

    # Show a label every 5 iterations; fall back to every iteration for short runs
    tick_step = 5 if len(iters) > 10 else 1
    label_ticks = [i for i in iters if i % tick_step == 0]
    if iters[0] not in label_ticks:
        label_ticks = [iters[0]] + label_ticks

    # --- Loss ---
    ax1.plot(iters, data["al_train_losses"],       "b-",  lw=2, marker="o", ms=6, label="AL Train")
    ax1.plot(iters, data["al_val_losses"],         "b--", lw=2, marker="s", ms=6, label="AL Validation")
    ax1.plot(iters, data["baseline_train_losses"], "r-",  lw=2, marker="o", ms=6, label="Baseline Train")
    ax1.plot(iters, data["baseline_val_losses"],   "r--", lw=2, marker="s", ms=6, label="Baseline Validation")
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Best Loss (MSE)", fontsize=12)
    ax1.set_title("Train/Validation Loss vs Iteration", fontsize=14)
    ax1.set_yscale("log")
    ax1.set_xticks(label_ticks)
    ax1.set_xticks(iters, minor=True)
    ax1.grid(True, which="major", alpha=0.3)
    ax1.grid(True, which="minor", alpha=0.15)
    ax1.legend(fontsize=9)

    # --- R² ---
    ax2.plot(iters, data["al_r2_scores"],       "b-", lw=2, marker="o", ms=6, label="Active Learning")
    ax2.plot(iters, data["baseline_r2_scores"], "r-", lw=2, marker="o", ms=6, label="Baseline (Random)")
    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("R² Score", fontsize=12)
    ax2.set_title("R² Score vs Iteration", fontsize=14)
    ax2.set_xticks(label_ticks)
    ax2.set_xticks(iters, minor=True)
    ax2.grid(True, which="major", alpha=0.3)
    ax2.grid(True, which="minor", alpha=0.15)
    all_r2 = data["al_r2_scores"] + data["baseline_r2_scores"]
    finite_r2 = [v for v in all_r2 if v == v and abs(v) != float("inf")]
    if finite_r2:
        ax2.set_ylim(min(0, min(finite_r2) - 0.1), 1.05)
    ax2.legend(fontsize=10)

    # --- Dataset sizes ---
    has_sizes = all(v is not None for v in data["n_trains"])
    if has_sizes:
        ax3.plot(iters, data["n_trains"], "b-",  lw=2, marker="o", ms=6, label="Train")
        ax3.plot(iters, data["n_vals"],   "b--", lw=2, marker="s", ms=6, label="Val")
        ax3.set_ylabel("Number of Samples", fontsize=12)
        ax3.legend(fontsize=9)
    else:
        ax3.text(0.5, 0.5, "Dataset sizes\nnot available\nin main log",
                 ha="center", va="center", fontsize=12, color="gray",
                 transform=ax3.transAxes)
    ax3.set_xlabel("Iteration", fontsize=12)
    ax3.set_title("Dataset Size vs Iteration", fontsize=14)
    ax3.set_xticks(label_ticks)
    ax3.set_xticks(iters, minor=True)
    ax3.grid(True, which="major", alpha=0.3)
    ax3.grid(True, which="minor", alpha=0.15)

    fig.suptitle(
        f"{pipeline_label} Active Learning Progress  —  "
        f"{len(iters)} iteration(s) completed",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}  ({len(iters)} iteration(s), {pipeline_label} pipeline)")


def main():
    parser = argparse.ArgumentParser(
        description="Plot AL progress from log file (transformer or GP pipeline)."
    )
    parser.add_argument(
        "--log", default=None,
        help="Path to active_learning.log (auto-detects pipeline type). "
             "Default: tries active_learning_output/ then active_learning_gp_output/.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output PNG path (default: <log_dir>/iteration_metrics_progress.png)",
    )
    args = parser.parse_args()

    # Auto-detect log path
    if args.log is None:
        candidates = [
            Path("active_learning_output/active_learning.log"),
            Path("active_learning_gp_output/active_learning.log"),
        ]
        for c in candidates:
            if c.exists():
                log_path = c
                break
        else:
            print("No active_learning.log found. Use --log to specify path.")
            return
    else:
        log_path = Path(args.log)

    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return

    # Auto-detect output path
    if args.output is None:
        output_path = log_path.parent / "iteration_metrics_progress.png"
    else:
        output_path = Path(args.output)

    data = parse_log(log_path)
    plot(data, output_path)


if __name__ == "__main__":
    main()
