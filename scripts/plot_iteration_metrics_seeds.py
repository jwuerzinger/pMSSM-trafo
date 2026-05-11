"""Multi-seed version of the per-run iteration_metrics plot.

Loads `state.pt` for every run in a (model, strategy, warm_start) cell of the
sweep manifest, aggregates the per-iteration metrics across seeds, and renders
a 2×3 figure showing seed-mean ± SEM bands. Layout mirrors the per-run plot
(pmssm/visualization.py:plot_iteration_metrics) so the two read consistently.

Usage:
    PYTHONPATH=. .pixi/envs/rocm/bin/python scripts/plot_iteration_metrics_seeds.py \\
        --model exact_gp --strategy entropy_batch --warm warm \\
        --output-dir /ptmp/jwuerzin/analysis/all_runs/iter_metrics_seeds

    # or, restrict to a specific sweep_id:
    PYTHONPATH=. ... --model transformer --strategy entropy_batch --warm cold \\
        --sweep-id 20260424_153140

No rolling mean is applied: the seed average already smooths out
per-iteration noise.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# ─── manifest loader ──────────────────────────────────────────────────────────

DEFAULT_MANIFEST = "/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv"
INCLUDE_STATUSES = {"completed", "timeout", "running"}


def load_cell_runs(manifest_path, model, strategy, warm, sweep_id=None):
    """Return list of run dirs matching (model, strategy, warm_start) [+ sweep_id]."""
    out = []
    with open(manifest_path) as f:
        rd = csv.reader(f)
        hdr = next(rd)
        for row in rd:
            # Schema: sweep_id, submit_time, model, strategy, warm_start, seed,
            # job_id, expected_run_dir, status, slurm_log
            if row[2] != model or row[3] != strategy or row[4] != warm:
                continue
            if sweep_id is not None and row[0] != sweep_id:
                continue
            if row[8] not in INCLUDE_STATUSES:
                continue
            if os.path.isdir(row[7]):
                out.append((int(row[5]), row[7]))   # (seed, dir)
    out.sort()
    return out


# ─── state.pt extractor ───────────────────────────────────────────────────────

# Map "panel-key" (canonical name used in the plot) → state.pt field name.
# Both transformer and GP pipelines write these to state.pt.
STATE_KEYS = {
    # own-set
    "train_losses":               "al_train_losses",
    "val_losses":                 "al_val_losses",
    "train_r2_scores":            "al_train_r2_scores",
    "r2_scores":                  "al_r2_scores",
    "n_train":                    "al_n_train",
    "n_val":                      "al_n_val",
    # baseline own-set
    "base_train_losses":          "baseline_train_losses",
    "base_val_losses":            "baseline_val_losses",
    "base_train_r2_scores":       "baseline_train_r2_scores",
    "base_r2_scores":             "baseline_r2_scores",
    "base_n_train":               "baseline_n_train",
    "base_n_val":                 "baseline_n_val",
    # cross-eval
    "al_on_base_val_losses":      "al_on_base_val_losses",
    "al_on_base_val_r2":          "al_on_base_val_r2",
    "base_on_al_val_losses":      "base_on_al_val_losses",
    "base_on_al_val_r2":          "base_on_al_val_r2",
    # MCMC eval
    "al_on_mcmc_losses":          "al_on_mcmc_losses",
    "al_on_mcmc_r2":              "al_on_mcmc_r2",
    "base_on_mcmc_losses":        "baseline_on_mcmc_losses",
    "base_on_mcmc_r2":            "baseline_on_mcmc_r2",
    # static random eval
    "al_on_static_losses":        "al_on_static_random_losses",
    "al_on_static_r2":            "al_on_static_random_r2",
    "base_on_static_losses":      "baseline_on_static_random_losses",
    "base_on_static_r2":          "baseline_on_static_random_r2",
}


def load_seed_series(run_dirs: list[tuple[int, str]]) -> dict[str, np.ndarray]:
    """Load and pad per-iteration series across seeds.

    Returns {panel_key: np.ndarray of shape (n_seeds, max_n_iters)} with NaN
    padding for seeds that completed fewer iterations.
    """
    raw: dict[str, list[list[float]]] = {k: [] for k in STATE_KEYS}
    max_iters = 0
    for _seed, d in run_dirs:
        sp = os.path.join(d, "state.pt")
        if not os.path.exists(sp):
            continue
        try:
            s = torch.load(sp, weights_only=False, map_location="cpu")
        except Exception as e:
            print(f"[warn] failed to load {sp}: {e}")
            continue
        n_iters_this = max(
            (len(s.get(STATE_KEYS["train_losses"]) or []),
             len(s.get(STATE_KEYS["val_losses"]) or []))
        )
        max_iters = max(max_iters, n_iters_this)
        for canonical_key, state_key in STATE_KEYS.items():
            vs = s.get(state_key) or []
            row = [
                float(v) if (v is not None and v == v and abs(float(v)) != float("inf"))
                else float("nan")
                for v in vs
            ]
            raw[canonical_key].append(row)
    # Pad to max_iters with NaN, stack into ndarray
    out = {}
    for k, rows in raw.items():
        if not rows:
            out[k] = np.empty((0, 0))
            continue
        n_seeds = len(rows)
        arr = np.full((n_seeds, max_iters), np.nan)
        for i, row in enumerate(rows):
            n = min(len(row), max_iters)
            arr[i, :n] = row[:n]
        out[k] = arr
    return out


def mean_sem(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-iter NaN-safe mean and SEM across seeds. Returns (mean, sem, n_seeds)."""
    if arr.size == 0:
        return np.empty(0), np.empty(0), np.empty(0, dtype=int)
    with np.errstate(invalid="ignore"):
        n = np.sum(~np.isnan(arr), axis=0)
        mean = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0, ddof=1)
        sem = np.where(n > 1, sd / np.sqrt(n), np.nan)
    return mean, sem, n


# ─── plotting ────────────────────────────────────────────────────────────────

def _draw_series(ax, iters, arr, *, color, linestyle, label, uncertainty="sem"):
    """Draw mean line + uncertainty band (NaN-safe)."""
    if arr.size == 0:
        return
    mean, sem, n = mean_sem(arr)
    if not np.isfinite(mean).any():
        return
    # Mask iterations where no seed has data
    mask = np.isfinite(mean)
    if not mask.any():
        return
    x = np.asarray(iters)[: len(mean)][mask]
    m = mean[mask]
    s = sem[mask]
    ax.plot(x, m, color=color, linestyle=linestyle, linewidth=2.0, label=label)
    # Band only where SEM is defined (≥2 seeds at that iter)
    s_safe = np.where(np.isfinite(s), s, 0.0)
    ax.fill_between(x, m - s_safe, m + s_safe,
                    color=color, alpha=0.18, linewidth=0)


def plot_multiseed_iteration_metrics(seed_arrays, iters, output_path,
                                     title_suffix=None, uncertainty="sem"):
    """Render the 2×3 multi-seed iteration_metrics plot."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=False)
    (ax_loss_own, ax_loss_shr, ax_n) = axes[0]
    (ax_r2_own, ax_r2_shr, ax_delta) = axes[1]

    tick_step = 5 if len(iters) > 10 else 1
    label_ticks = [i for i in iters if i % tick_step == 0]
    if iters and iters[0] not in label_ticks:
        label_ticks = [iters[0]] + label_ticks

    def _style_axis(ax, ylabel, title):
        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, which="major", alpha=0.3)
        ax.grid(True, which="minor", alpha=0.15)
        ax.set_xticks(label_ticks)
        if iters:
            ax.set_xticks(iters, minor=True)

    # ─── (0,0) Own-set Loss ────────────────────────────────────────────────
    _draw_series(ax_loss_own, iters, seed_arrays["train_losses"],
                 color="tab:blue", linestyle="-", label="AL train")
    _draw_series(ax_loss_own, iters, seed_arrays["val_losses"],
                 color="tab:blue", linestyle="--", label="AL val")
    _draw_series(ax_loss_own, iters, seed_arrays["base_train_losses"],
                 color="tab:red", linestyle="-", label="Base train")
    _draw_series(ax_loss_own, iters, seed_arrays["base_val_losses"],
                 color="tab:red", linestyle="--", label="Base val")
    _style_axis(ax_loss_own, "MSE (own-set)", "Own-set loss")
    # Always log-scale loss panels, ylim excluding iter 1.
    _loss_vals = []
    for k in ("train_losses", "val_losses", "base_train_losses", "base_val_losses"):
        arr = seed_arrays.get(k, np.empty((0, 0)))
        if arr.size and arr.shape[1] > 1:
            vals = arr[:, 1:][np.isfinite(arr[:, 1:])]
            _loss_vals.extend(v for v in vals.tolist() if v > 0)
    if _loss_vals:
        ax_loss_own.set_yscale("log")
        ax_loss_own.set_ylim(min(_loss_vals) * 0.5, max(_loss_vals) * 2)
    ax_loss_own.legend(fontsize=8, loc="best", framealpha=0.85, ncol=2)

    # ─── (0,1) Shared-eval Loss ────────────────────────────────────────────
    eval_sets = [
        ("al_on_base_val_losses", "base_on_al_val_losses", "tab:green", "Cross-val"),
        ("al_on_mcmc_losses",     "base_on_mcmc_losses",   "tab:purple", "MCMC"),
        ("al_on_static_losses",   "base_on_static_losses", "tab:cyan",   "Static rnd"),
    ]
    has_shr_loss = False
    _shr_loss_vals = []
    for al_k, bs_k, color, label in eval_sets:
        for k, ls, who in ((al_k, "-", "AL"), (bs_k, "--", "Base")):
            arr = seed_arrays.get(k, np.empty((0, 0)))
            if arr.size == 0:
                continue
            _draw_series(ax_loss_shr, iters, arr,
                         color=color, linestyle=ls, label=f"{who} on {label}")
            has_shr_loss = True
            if arr.shape[1] > 1:
                vals = arr[:, 1:][np.isfinite(arr[:, 1:])]
                _shr_loss_vals.extend(v for v in vals.tolist() if v > 0)
    _style_axis(ax_loss_shr, "MSE (transformed)", "Shared-eval loss")
    if _shr_loss_vals:
        ax_loss_shr.set_yscale("log")
        ax_loss_shr.set_ylim(min(_shr_loss_vals) * 0.5, max(_shr_loss_vals) * 2)
    if has_shr_loss:
        ax_loss_shr.legend(fontsize=8, loc="best", framealpha=0.85, ncol=2)
    else:
        ax_loss_shr.text(0.5, 0.5, "no shared-eval data",
                         transform=ax_loss_shr.transAxes,
                         ha="center", va="center", color="gray")

    # ─── (0,2) Dataset size ───────────────────────────────────────────────
    _draw_series(ax_n, iters, seed_arrays["n_train"],
                 color="tab:blue", linestyle="-", label="AL train")
    _draw_series(ax_n, iters, seed_arrays["n_val"],
                 color="tab:blue", linestyle="--", label="AL val")
    _draw_series(ax_n, iters, seed_arrays["base_n_train"],
                 color="tab:red", linestyle="-", label="Base train")
    _draw_series(ax_n, iters, seed_arrays["base_n_val"],
                 color="tab:red", linestyle="--", label="Base val")
    _style_axis(ax_n, "Number of samples", "Dataset size")
    ax_n.legend(fontsize=8, loc="best", framealpha=0.85, ncol=2)

    # ─── (1,0) Own-set R² ──────────────────────────────────────────────────
    _draw_series(ax_r2_own, iters, seed_arrays["train_r2_scores"],
                 color="tab:blue", linestyle="-", label="AL train")
    _draw_series(ax_r2_own, iters, seed_arrays["r2_scores"],
                 color="tab:blue", linestyle="--", label="AL val")
    _draw_series(ax_r2_own, iters, seed_arrays["base_train_r2_scores"],
                 color="tab:red", linestyle="-", label="Base train")
    _draw_series(ax_r2_own, iters, seed_arrays["base_r2_scores"],
                 color="tab:red", linestyle="--", label="Base val")
    _style_axis(ax_r2_own, "R² (own-set)", "Own-set R²")
    _own_r2 = []
    for k in ("train_r2_scores", "r2_scores", "base_train_r2_scores", "base_r2_scores"):
        arr = seed_arrays.get(k, np.empty((0, 0)))
        if arr.size and arr.shape[1] > 1:
            vals = arr[:, 1:][np.isfinite(arr[:, 1:])]
            _own_r2.extend(vals.tolist())
    if _own_r2:
        ylo, yhi = min(_own_r2), max(_own_r2)
        pad = max(0.05, 0.1 * (yhi - ylo))
        ax_r2_own.set_ylim(max(-1.05, ylo - pad), min(1.05, yhi + pad))
    ax_r2_own.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax_r2_own.legend(fontsize=8, loc="best", framealpha=0.85, ncol=2)

    # ─── (1,1) Shared-eval R²  (symlog) ───────────────────────────────────
    r2_sets = [
        ("al_on_base_val_r2", "base_on_al_val_r2", "tab:green", "Cross-val"),
        ("al_on_mcmc_r2",     "base_on_mcmc_r2",   "tab:purple", "MCMC"),
        ("al_on_static_r2",   "base_on_static_r2", "tab:cyan",   "Static rnd"),
    ]
    has_shr_r2 = False
    _ylim_vals = []
    for al_k, bs_k, color, label in r2_sets:
        for k, ls, who in ((al_k, "-", "AL"), (bs_k, "--", "Base")):
            arr = seed_arrays.get(k, np.empty((0, 0)))
            if arr.size == 0:
                continue
            _draw_series(ax_r2_shr, iters, arr,
                         color=color, linestyle=ls, label=f"{who} on {label}")
            has_shr_r2 = True
            if arr.shape[1] > 1:
                vals = arr[:, 1:][np.isfinite(arr[:, 1:])]
                _ylim_vals.extend(vals.tolist())
    _style_axis(ax_r2_shr, "R² (shared eval)", "Shared-eval R²")
    if has_shr_r2 and _ylim_vals:
        ax_r2_shr.set_yscale("symlog", linthresh=1.0)
        ymin, ymax = min(_ylim_vals), max(_ylim_vals)
        upper = max(ymax * 1.05, ymax + 0.05) if ymax > 0 else 0.05
        lower = ymin - 0.10 * max(abs(ymin), 1)
        ax_r2_shr.set_ylim(lower, upper)
        ax_r2_shr.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
        ax_r2_shr.legend(fontsize=8, loc="best", framealpha=0.85, ncol=2)
    else:
        ax_r2_shr.text(0.5, 0.5, "no shared-eval data",
                       transform=ax_r2_shr.transAxes,
                       ha="center", va="center", color="gray")

    # ─── (1,2) Δ R² (AL − Baseline) ───────────────────────────────────────
    delta_sets = [
        ("al_on_base_val_r2", "base_on_al_val_r2", "tab:green", "Cross-val"),
        ("al_on_mcmc_r2",     "base_on_mcmc_r2",   "tab:purple", "MCMC"),
        ("al_on_static_r2",   "base_on_static_r2", "tab:cyan",   "Static rnd"),
    ]
    has_delta = False
    _delta_vals = []
    for al_k, bs_k, color, label in delta_sets:
        al_arr = seed_arrays.get(al_k, np.empty((0, 0)))
        bs_arr = seed_arrays.get(bs_k, np.empty((0, 0)))
        if al_arr.size == 0 or bs_arr.size == 0:
            continue
        # Per-seed Δ, then aggregate
        # (Both arrays should have shape (n_seeds, n_iters); pair on first axis
        # by row index since seeds are aligned by manifest order.)
        n_iters_min = min(al_arr.shape[1], bs_arr.shape[1])
        n_seeds_min = min(al_arr.shape[0], bs_arr.shape[0])
        delta = al_arr[:n_seeds_min, :n_iters_min] - bs_arr[:n_seeds_min, :n_iters_min]
        _draw_series(ax_delta, iters[:n_iters_min], delta,
                     color=color, linestyle="-", label=label)
        has_delta = True
        if delta.shape[1] > 1:
            vals = delta[:, 1:][np.isfinite(delta[:, 1:])]
            _delta_vals.extend(vals.tolist())
    _style_axis(ax_delta, "Δ R² = AL − Baseline", "Δ R² across shared evals")
    ax_delta.axhline(0, color="black", linewidth=0.8, alpha=0.7)
    if has_delta and _delta_vals:
        absvals = sorted(abs(v) for v in _delta_vals)
        absmax = absvals[int(len(absvals) * 0.95)] if len(absvals) > 5 else max(absvals)
        ax_delta.set_yscale("symlog", linthresh=1.0)
        ax_delta.set_ylim(-absmax * 1.2 - 0.5, absmax * 1.2 + 0.5)
        ax_delta.legend(fontsize=8, loc="best", framealpha=0.85, ncol=2)
    else:
        ax_delta.text(0.5, 0.5, "no shared-eval data",
                      transform=ax_delta.transAxes,
                      ha="center", va="center", color="gray")

    # Suptitle
    suptitle = "Active-learning iteration metrics — multi-seed (mean ± SEM)"
    if title_suffix:
        suptitle += f"\n{title_suffix}"
    fig.suptitle(suptitle, fontsize=13)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST,
                   help=f"Sweep manifest CSV (default: {DEFAULT_MANIFEST})")
    p.add_argument("--model", required=True,
                   help="Model column value, e.g. transformer / deep_gp / "
                        "exact_gp / dnn / tabpfn / transformer_oracle / "
                        "deep_gp_oracle.")
    p.add_argument("--strategy", required=True,
                   help="Strategy column value, e.g. entropy_batch / "
                        "top_k / top_k_tol_only.")
    p.add_argument("--warm", required=True,
                   help="Warm-start column value: 'warm', 'cold', or "
                        "'tabpfn' (sentinel for the no-warm-axis TabPFN runs).")
    p.add_argument("--sweep-id", default=None,
                   help="Optional sweep_id filter. Default: include any.")
    p.add_argument("--output-dir", default="/ptmp/jwuerzin/analysis/all_runs",
                   help="Output directory for the rendered PNG.")
    p.add_argument("--filename", default=None,
                   help="Override output filename. Default: "
                        "iteration_metrics_seeds_<model>_<strategy>_<warm>.png")
    args = p.parse_args()

    runs = load_cell_runs(args.manifest, args.model, args.strategy, args.warm,
                          sweep_id=args.sweep_id)
    if not runs:
        raise SystemExit(
            f"No runs found for ({args.model}, {args.strategy}, {args.warm})"
            + (f" in sweep {args.sweep_id}" if args.sweep_id else "")
        )
    print(f"Loaded {len(runs)} run(s) for "
          f"{args.model}/{args.strategy}/{args.warm}: "
          f"seeds={[s for s, _ in runs]}")

    seed_arrays = load_seed_series(runs)
    # Iterations axis = 1..max_iters
    max_iters = seed_arrays.get("train_losses",
                                 np.empty((0, 0))).shape[1] if seed_arrays else 0
    if max_iters == 0:
        max_iters = seed_arrays.get("al_on_mcmc_r2",
                                    np.empty((0, 0))).shape[1]
    if max_iters == 0:
        raise SystemExit("No usable iteration data in state.pt files.")
    iters = list(range(1, max_iters + 1))

    fn = args.filename or (
        f"iteration_metrics_seeds_{args.model}_{args.strategy}_{args.warm}.png"
    )
    out = Path(args.output_dir) / fn
    title_suffix = (f"{args.model} / {args.strategy} / {args.warm}  —  "
                    f"{len(runs)} seed(s), up to {max_iters} iteration(s)")
    plot_multiseed_iteration_metrics(
        seed_arrays, iters, str(out),
        title_suffix=title_suffix,
    )
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
