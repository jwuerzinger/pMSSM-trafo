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


def plot(data, output_path, show_raw=False):
    """Adapter that maps the log-parsed `data` dict into the canonical
    al_metrics / baseline_metrics schema and calls
    `pmssm.visualization.plot_iteration_metrics` — keeping per-run end-of-AL
    plots and `plot_progress.py` reloads on a single layout/style.

    Pass `show_raw=True` to overlay the raw per-iteration values behind the
    rolling-mean lines (useful for diagnostics; off by default for cleaner
    talk-ready plots).
    """
    iters = data["iterations"]
    if not iters:
        print("No completed iterations found in log.")
        return

    pipeline = data["pipeline"]
    _pipeline_labels = {"gp": "GP", "transformer": "Transformer",
                        "tabpfn": "TabPFN", "dnn": "DNN"}
    pipeline_label = _pipeline_labels.get(pipeline, pipeline.title())

    # Adapter: build the dicts plot_iteration_metrics expects.
    def _or_none(xs):
        return xs if xs and len(xs) == len(iters) else None

    al_metrics = {
        "train_losses": data["al_train_losses"],
        "val_losses": data["al_val_losses"],
        "train_r2_scores": _or_none(data.get("al_train_r2_scores", [])),
        "r2_scores": data["al_r2_scores"],
        "cross_val_losses": _or_none(data.get("al_on_base_val_losses", [])),
        "cross_val_r2": _or_none(data.get("al_on_base_val_r2", [])),
        "mcmc_eval_losses": _or_none(data.get("al_on_mcmc_losses", [])),
        "mcmc_eval_r2": _or_none(data.get("al_on_mcmc_r2", [])),
        "static_random_eval_losses": _or_none(data.get("al_on_static_losses", [])),
        "static_random_eval_r2": _or_none(data.get("al_on_static_r2", [])),
        "n_train": _or_none(data.get("al_n_trains", [])),
        "n_val": _or_none(data.get("al_n_vals", [])),
    }
    baseline_metrics = {
        "train_losses": data["baseline_train_losses"],
        "val_losses": data["baseline_val_losses"],
        "train_r2_scores": _or_none(data.get("baseline_train_r2_scores", [])),
        "r2_scores": data["baseline_r2_scores"],
        # cross-eval (Base on AL Val) lives under the same shared-eval key on
        # the baseline side, so plot_iteration_metrics can compute Δ = AL − Base.
        "cross_val_losses": _or_none(data.get("base_on_al_val_losses", [])),
        "cross_val_r2": _or_none(data.get("base_on_al_val_r2", [])),
        "mcmc_eval_losses": _or_none(data.get("base_on_mcmc_losses", [])),
        "mcmc_eval_r2": _or_none(data.get("base_on_mcmc_r2", [])),
        "static_random_eval_losses": _or_none(data.get("base_on_static_losses", [])),
        "static_random_eval_r2": _or_none(data.get("base_on_static_r2", [])),
        "n_train": _or_none(data.get("baseline_n_trains", [])),
        "n_val": _or_none(data.get("baseline_n_vals", [])),
    }

    # Derive a self-describing title from the parent run directory name when
    # possible (e.g. "active_learning_exact_gp_entropy_batch_warm_seed1_..."
    # → "exact_gp / entropy_batch / warm / seed1"). Falls back to the parsed
    # pipeline label.
    suffix = f"{pipeline_label} pipeline  —  {len(iters)} iteration(s) completed"
    parent = Path(output_path).parent
    name = parent.name
    if name.startswith("active_learning_"):
        body = name[len("active_learning_"):]
        # Strip a trailing _<YYYYMMDD>_<HHMMSS> if present
        body = re.sub(r"_\d{8}_\d{6}$", "", body)
        suffix = f"{body}  —  {len(iters)} iteration(s) completed"

    from pmssm.visualization import plot_iteration_metrics
    out_path = plot_iteration_metrics(
        iters, al_metrics, baseline_metrics,
        output_dir=parent,
        logger=None,
        title_suffix=suffix,
        filename=Path(output_path).name,
        show_raw=show_raw,
    )
    print(f"Saved plot to {out_path}  ({len(iters)} iteration(s), {pipeline_label} pipeline)")


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
    parser.add_argument(
        "--show-raw", action="store_true",
        help="Overlay the raw (un-smoothed) per-iteration values behind the "
             "rolling-mean lines. Default: off (cleaner talk plots).",
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
    plot(data, output_path, show_raw=args.show_raw)


if __name__ == "__main__":
    main()
