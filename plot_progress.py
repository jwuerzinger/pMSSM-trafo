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


_SIZE_RE = re.compile(
    r"Training set size:\s*(\d+),\s*Validation set size:\s*(\d+)"
)


def _read_worker_log_sizes(log_dir, iteration, worker="al"):
    """Read n_train, n_val from a per-iteration worker log.

    Both pipelines log 'Training set size: N, Validation set size: M'
    in each iteration's {worker}_training.log.
    """
    worker_log = log_dir / f"iteration_{iteration:03d}" / f"{worker}_training.log"
    if not worker_log.exists():
        return None, None
    with open(worker_log) as f:
        for line in f:
            m = _SIZE_RE.search(line)
            if m:
                return int(m.group(1)), int(m.group(2))
    return None, None


def parse_log(log_path):
    """Parse iteration metrics from an active learning log file.

    Auto-detects transformer vs GP pipeline from the iteration header format.
    Dataset sizes are always read from per-iteration worker logs
    (al_training.log / baseline_training.log).
    """
    iterations = []
    al_train_losses, al_val_losses, al_r2_scores, al_train_r2_scores = [], [], [], []
    baseline_train_losses, baseline_val_losses, baseline_r2_scores, baseline_train_r2_scores = [], [], [], []
    al_on_base_val_losses, al_on_base_val_r2 = [], []
    base_on_al_val_losses, base_on_al_val_r2 = [], []
    al_on_mcmc_losses, al_on_mcmc_r2 = [], []
    base_on_mcmc_losses, base_on_mcmc_r2 = [], []
    al_on_static_losses, al_on_static_r2 = [], []
    base_on_static_losses, base_on_static_r2 = [], []
    al_n_trains, al_n_vals = [], []
    baseline_n_trains, baseline_n_vals = [], []

    current_iteration = None
    pipeline = None  # auto-detected: "transformer", "gp", or "tabpfn"

    # Iteration headers — accept both formats
    iter_re = re.compile(
        r"=== (?:Global|GP Active Learning|DNN Active Learning) Iteration (\d+) ==="
    )
    # Metric lines (same format for both pipelines)
    # train_R² is optional for backward compatibility with older logs
    al_re = re.compile(
        r"AL metrics: train_loss=([\d.]+), val_loss=([\d.]+), "
        r"R²=([-\d.]+)(?:, train_R²=([-\d.]+))?"
    )
    base_re = re.compile(
        r"Baseline metrics: train_loss=([\d.]+), val_loss=([\d.]+), "
        r"R²=([-\d.]+)(?:, train_R²=([-\d.]+))?"
    )
    # Cross-evaluation line (optional, for newer logs)
    cross_re = re.compile(
        r"Cross-eval: AL_on_base_val_loss=([\d.]+), AL_on_base_val_R²=([-\d.]+), "
        r"base_on_al_val_loss=([\d.]+), base_on_al_val_R²=([-\d.]+)"
    )
    # Static evaluation lines (optional)
    mcmc_re = re.compile(
        r"MCMC eval: AL_loss=([\d.]+), AL_R²=([-\d.]+), "
        r"Base_loss=([\d.]+), Base_R²=([-\d.]+)"
    )
    static_re = re.compile(
        r"Static random eval: AL_loss=([\d.]+), AL_R²=([-\d.]+), "
        r"Base_loss=([\d.]+), Base_R²=([-\d.]+)"
    )
    # Auto-detect pipeline type
    gp_header_re = re.compile(r"GP Active Learning")
    tabpfn_header_re = re.compile(r"pMSSM \(TabPFN\)")
    dnn_header_re = re.compile(r"DNN Active Learning Iteration")
    # TabPFN logs dataset sizes inline: "AL: n_train=1600, n_val=400"
    al_size_re = re.compile(r"AL: n_train=(\d+), n_val=(\d+)")
    base_size_re = re.compile(r"Baseline: n_train=(\d+), n_val=(\d+)")

    log_dir = log_path.parent

    # Track inline sizes for TabPFN (no worker logs)
    _pending_al_nt, _pending_al_nv = None, None
    _pending_base_nt, _pending_base_nv = None, None

    with open(log_path) as f:
        for line in f:
            # Auto-detect pipeline on first match
            if pipeline is None and tabpfn_header_re.search(line):
                pipeline = "tabpfn"
            elif pipeline is None and gp_header_re.search(line):
                pipeline = "gp"
            elif pipeline is None and dnn_header_re.search(line):
                pipeline = "dnn"

            m = iter_re.search(line)
            if m:
                if pipeline is None:
                    pipeline = "transformer"
                current_iteration = int(m.group(1))
                _pending_al_nt, _pending_al_nv = None, None
                _pending_base_nt, _pending_base_nv = None, None
                continue

            # Parse inline dataset sizes (TabPFN logs these in the main log)
            m = al_size_re.search(line)
            if m and current_iteration is not None:
                _pending_al_nt = int(m.group(1))
                _pending_al_nv = int(m.group(2))
                continue

            m = base_size_re.search(line)
            if m and current_iteration is not None:
                _pending_base_nt = int(m.group(1))
                _pending_base_nv = int(m.group(2))
                continue

            m = al_re.search(line)
            if m and current_iteration is not None:
                al_train_losses.append(float(m.group(1)))
                al_val_losses.append(float(m.group(2)))
                al_r2_scores.append(float(m.group(3)))
                al_train_r2_scores.append(float(m.group(4)) if m.group(4) else None)
                continue

            m = base_re.search(line)
            if m and current_iteration is not None:
                baseline_train_losses.append(float(m.group(1)))
                baseline_val_losses.append(float(m.group(2)))
                baseline_r2_scores.append(float(m.group(3)))
                baseline_train_r2_scores.append(float(m.group(4)) if m.group(4) else None)

                # Read dataset sizes: try worker logs first, fall back to inline
                al_nt, al_nv = _read_worker_log_sizes(
                    log_dir, current_iteration, "al"
                )
                base_nt, base_nv = _read_worker_log_sizes(
                    log_dir, current_iteration, "baseline"
                )
                # Use inline sizes if worker logs unavailable (TabPFN pipeline)
                if al_nt is None:
                    al_nt = _pending_al_nt
                if al_nv is None:
                    al_nv = _pending_al_nv
                if base_nt is None:
                    base_nt = _pending_base_nt
                if base_nv is None:
                    base_nv = _pending_base_nv

                iterations.append(current_iteration)
                al_n_trains.append(al_nt)
                al_n_vals.append(al_nv)
                baseline_n_trains.append(base_nt)
                baseline_n_vals.append(base_nv)
                current_iteration = None
                continue

            # Cross-eval line (optional, appears after baseline metrics)
            m = cross_re.search(line)
            if m:
                al_on_base_val_losses.append(float(m.group(1)))
                al_on_base_val_r2.append(float(m.group(2)))
                base_on_al_val_losses.append(float(m.group(3)))
                base_on_al_val_r2.append(float(m.group(4)))
                continue

            m = mcmc_re.search(line)
            if m:
                al_on_mcmc_losses.append(float(m.group(1)))
                al_on_mcmc_r2.append(float(m.group(2)))
                base_on_mcmc_losses.append(float(m.group(3)))
                base_on_mcmc_r2.append(float(m.group(4)))
                continue

            m = static_re.search(line)
            if m:
                al_on_static_losses.append(float(m.group(1)))
                al_on_static_r2.append(float(m.group(2)))
                base_on_static_losses.append(float(m.group(3)))
                base_on_static_r2.append(float(m.group(4)))

    return dict(
        iterations=iterations,
        al_train_losses=al_train_losses,
        al_val_losses=al_val_losses,
        al_r2_scores=al_r2_scores,
        al_train_r2_scores=al_train_r2_scores,
        baseline_train_losses=baseline_train_losses,
        baseline_val_losses=baseline_val_losses,
        baseline_r2_scores=baseline_r2_scores,
        baseline_train_r2_scores=baseline_train_r2_scores,
        al_on_base_val_losses=al_on_base_val_losses,
        al_on_base_val_r2=al_on_base_val_r2,
        base_on_al_val_losses=base_on_al_val_losses,
        base_on_al_val_r2=base_on_al_val_r2,
        al_on_mcmc_losses=al_on_mcmc_losses,
        al_on_mcmc_r2=al_on_mcmc_r2,
        base_on_mcmc_losses=base_on_mcmc_losses,
        base_on_mcmc_r2=base_on_mcmc_r2,
        al_on_static_losses=al_on_static_losses,
        al_on_static_r2=al_on_static_r2,
        base_on_static_losses=base_on_static_losses,
        base_on_static_r2=base_on_static_r2,
        al_n_trains=al_n_trains,
        al_n_vals=al_n_vals,
        baseline_n_trains=baseline_n_trains,
        baseline_n_vals=baseline_n_vals,
        pipeline=pipeline or "transformer",
    )


def plot(data, output_path):
    iters = data["iterations"]
    if not iters:
        print("No completed iterations found in log.")
        return

    pipeline = data["pipeline"]
    _pipeline_labels = {"gp": "GP", "transformer": "Transformer", "tabpfn": "TabPFN", "dnn": "DNN"}
    pipeline_label = _pipeline_labels.get(pipeline, pipeline.title())

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
    has_cross = len(data["al_on_base_val_losses"]) == len(iters)
    if has_cross:
        ax1.plot(iters, data["al_on_base_val_losses"],   "g:", lw=2, marker="^", ms=6, label="AL on Base Val")
        ax1.plot(iters, data["base_on_al_val_losses"],   "g--", lw=2, marker="v", ms=6, label="Base on AL Val")
    has_mcmc = len(data["al_on_mcmc_losses"]) == len(iters)
    if has_mcmc:
        ax1.plot(iters, data["al_on_mcmc_losses"],   "m-",  lw=2, marker="^", ms=5, label="AL on MCMC")
        ax1.plot(iters, data["base_on_mcmc_losses"], "m--", lw=2, marker="v", ms=5, label="Base on MCMC")
    has_static = len(data["al_on_static_losses"]) == len(iters)
    if has_static:
        ax1.plot(iters, data["al_on_static_losses"],   "c-",  lw=2, marker="^", ms=5, label="AL on Static Rnd")
        ax1.plot(iters, data["base_on_static_losses"], "c--", lw=2, marker="v", ms=5, label="Base on Static Rnd")
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Best Loss (MSE)", fontsize=12)
    ax1.set_title("Train/Validation Loss vs Iteration", fontsize=14)
    ax1.set_yscale("log")
    ax1.set_xticks(label_ticks)
    ax1.set_xticks(iters, minor=True)
    ax1.grid(True, which="major", alpha=0.3)
    ax1.grid(True, which="minor", alpha=0.15)
    ax1.legend(fontsize=9)

    # --- R² (training, validation, and cross-validation) ---
    has_al_train_r2 = all(v is not None for v in data["al_train_r2_scores"])
    has_base_train_r2 = all(v is not None for v in data["baseline_train_r2_scores"])
    if has_al_train_r2:
        ax2.plot(iters, data["al_train_r2_scores"],       "b-",  lw=2, marker="o", ms=6, label="AL Train")
    ax2.plot(iters, data["al_r2_scores"],                  "b--", lw=2, marker="s", ms=6, label="AL Validation")
    if has_base_train_r2:
        ax2.plot(iters, data["baseline_train_r2_scores"], "r-",  lw=2, marker="o", ms=6, label="Baseline Train")
    ax2.plot(iters, data["baseline_r2_scores"],            "r--", lw=2, marker="s", ms=6, label="Baseline Validation")
    if has_cross:
        ax2.plot(iters, data["al_on_base_val_r2"],        "g:", lw=2, marker="^", ms=6, label="AL on Base Val")
        ax2.plot(iters, data["base_on_al_val_r2"],        "g--", lw=2, marker="v", ms=6, label="Base on AL Val")
    if has_mcmc:
        ax2.plot(iters, data["al_on_mcmc_r2"],   "m-",  lw=2, marker="^", ms=5, label="AL on MCMC")
        ax2.plot(iters, data["base_on_mcmc_r2"], "m--", lw=2, marker="v", ms=5, label="Base on MCMC")
    if has_static:
        ax2.plot(iters, data["al_on_static_r2"],   "c-",  lw=2, marker="^", ms=5, label="AL on Static Rnd")
        ax2.plot(iters, data["base_on_static_r2"], "c--", lw=2, marker="v", ms=5, label="Base on Static Rnd")
    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("R² Score", fontsize=12)
    ax2.set_title("R² Score vs Iteration", fontsize=14)
    ax2.set_xticks(label_ticks)
    ax2.set_xticks(iters, minor=True)
    ax2.grid(True, which="major", alpha=0.3)
    ax2.grid(True, which="minor", alpha=0.15)
    all_r2 = data["al_r2_scores"] + data["baseline_r2_scores"]
    if has_al_train_r2:
        all_r2 += data["al_train_r2_scores"]
    if has_base_train_r2:
        all_r2 += data["baseline_train_r2_scores"]
    if has_cross:
        all_r2 += data["al_on_base_val_r2"] + data["base_on_al_val_r2"]
    if has_mcmc:
        all_r2 += data["al_on_mcmc_r2"] + data["base_on_mcmc_r2"]
    if has_static:
        all_r2 += data["al_on_static_r2"] + data["base_on_static_r2"]
    finite_r2 = [v for v in all_r2 if v is not None and v == v and abs(v) != float("inf")]
    r2_lower = -1.0
    ax2.set_ylim(r2_lower, 1.05)

    # Draw downward arrows for any R² values that fall below the y-axis limit
    _r2_series = [
        (data["al_train_r2_scores"] if has_al_train_r2 else [], "b"),
        (data["al_r2_scores"], "b"),
        (data["baseline_train_r2_scores"] if has_base_train_r2 else [], "r"),
        (data["baseline_r2_scores"], "r"),
        (data["al_on_base_val_r2"] if has_cross else [], "g"),
        (data["base_on_al_val_r2"] if has_cross else [], "g"),
        (data["al_on_mcmc_r2"] if has_mcmc else [], "m"),
        (data["base_on_mcmc_r2"] if has_mcmc else [], "m"),
        (data["al_on_static_r2"] if has_static else [], "c"),
        (data["base_on_static_r2"] if has_static else [], "c"),
    ]
    for r2_vals, color in _r2_series:
        if not r2_vals:
            continue
        for it, val in zip(iters, r2_vals):
            if val is not None and val == val and val < r2_lower:
                ax2.annotate("", xy=(it, r2_lower), xytext=(it, r2_lower + 0.08),
                             arrowprops=dict(arrowstyle="->", color=color, lw=2))

    ax2.legend(fontsize=9)

    # --- Dataset sizes (AL and Baseline separately) ---
    has_al_sizes = all(v is not None for v in data["al_n_trains"])
    has_base_sizes = all(v is not None for v in data["baseline_n_trains"])
    if has_al_sizes or has_base_sizes:
        if has_al_sizes:
            ax3.plot(iters, data["al_n_trains"], "b-",  lw=2, marker="o", ms=6, label="AL Train")
            ax3.plot(iters, data["al_n_vals"],   "b--", lw=2, marker="s", ms=6, label="AL Validation")
        if has_base_sizes:
            ax3.plot(iters, data["baseline_n_trains"], "r-",  lw=2, marker="o", ms=6, label="Baseline Train")
            ax3.plot(iters, data["baseline_n_vals"],   "r--", lw=2, marker="s", ms=6, label="Baseline Validation")
        ax3.set_ylabel("Number of Samples", fontsize=12)
        ax3.legend(fontsize=9)
    else:
        ax3.text(0.5, 0.5, "Dataset sizes\nnot available\nin worker logs",
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
            Path("active_learning_tabpfn_output/active_learning.log"),
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
