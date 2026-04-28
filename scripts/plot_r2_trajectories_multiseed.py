"""Plot mean trajectories of per-iteration metrics with uncertainty bands.

Sister script to `plot_hit_rate_trajectories_multiseed.py` and
`plot_r2_mcmc_trajectories_multiseed.py`. Covers the *other* per-iteration
trajectories tracked by the AL pipeline (which `analyse_runs.py` plots in
`plot_r2_trajectories.png`):

  - val_r2     : AL model's validation R² each iter (`run.al_r2`)
  - static_r2  : AL model's R² on the static-random eval set (`run.al_static_r2`)
  - n_train    : cumulative training-set size (`run.n_train_per_iter`)

For each metric, two plot families are written:

  1. <metric>_strategy_<strategy>.png — one figure per strategy, overlaying
                                        every (model, warm) combo.
                                        Color = model, linestyle = warm.
  2. <metric>_best_per_model.png      — single figure with one curve per model,
                                        picking the (strategy, warm) setting
                                        that maximises mean final-iteration value.

MCMC R² has its own dedicated script (`plot_r2_mcmc_trajectories_multiseed.py`)
so it is not duplicated here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make the repo root importable so we can reuse analyse_runs utilities.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analyse_runs import load_run  # noqa: E402


MODEL_COLORS = {
    "transformer": "tab:blue",
    "exact_gp":    "tab:orange",
    "deep_gp":     "tab:green",
    "tabpfn":      "tab:red",
}
WARM_LS = {
    "warm":   "-",
    "cold":   "--",
    "tabpfn": "-",
}
WARM_MARKER = {
    "warm":   "o",
    "cold":   "s",
    "tabpfn": "^",
}


# ──────────────────────────────────────────────────────────────────────────────
# Per-iteration extractors
# ──────────────────────────────────────────────────────────────────────────────

def _list_trajectory(values) -> tuple[list[int], list[float]]:
    """Helper: drop None / NaN entries and return (iters_1based, values)."""
    iters, out = [], []
    for i, v in enumerate(values or []):
        if v is None:
            continue
        try:
            vf = float(v)
        except (TypeError, ValueError):
            continue
        if np.isnan(vf):
            continue
        iters.append(i + 1)
        out.append(vf)
    return iters, out


def _val_r2_trajectory(run):
    """AL model's validation R² each iter."""
    return _list_trajectory(run.al_r2)


def _static_r2_trajectory(run):
    """AL model's R² on the static-random eval set."""
    return _list_trajectory(run.al_static_r2)


def _n_train_trajectory(run):
    """Cumulative training-set size each iter."""
    return _list_trajectory(run.n_train_per_iter)


# Registry: metric name → (extractor, ylabel, title_word, axhline_at_zero)
METRICS = {
    "val_r2": (
        _val_r2_trajectory,
        "R² (validation)",
        "Validation R²",
        True,
    ),
    "static_r2": (
        _static_r2_trajectory,
        "R² (static random eval)",
        "R² on static-random eval",
        True,
    ),
    "n_train": (
        _n_train_trajectory,
        "Training-set size",
        "Cumulative training-set size",
        False,
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory aggregation (NaN-padded across seeds, partial-run aware)
# ──────────────────────────────────────────────────────────────────────────────

def _band(Y: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (lo, hi) uncertainty bands for an (n_seeds, n_iters) NaN-aware array.

    At each iteration only the seeds that reported a value contribute.
    Iterations with zero or one valid seed get a zero-width band.
    """
    n_per_iter = np.sum(~np.isnan(Y), axis=0)
    mean = np.nanmean(Y, axis=0)
    if mode == "sem":
        with np.errstate(invalid="ignore", divide="ignore"):
            sd = np.nanstd(Y, axis=0, ddof=1)
        sd = np.where(n_per_iter > 1, sd, 0.0)
        half = sd / np.sqrt(np.clip(n_per_iter, 1, None))
        return mean - half, mean + half
    if mode == "sd":
        with np.errstate(invalid="ignore", divide="ignore"):
            sd = np.nanstd(Y, axis=0, ddof=1)
        sd = np.where(n_per_iter > 1, sd, 0.0)
        return mean - sd, mean + sd
    if mode == "iqr":
        return np.nanpercentile(Y, 25, axis=0), np.nanpercentile(Y, 75, axis=0)
    raise ValueError(f"unknown uncertainty mode: {mode}")


def _collect_trajectories(df, min_seeds, extractor):
    """Build {(model, strategy, warm): (iters_axis, Y[n_seeds, n_iters])}.

    NaN-padded so partial runs (status `running` / `timeout`) line up with
    completed seeds at iteration 1; iterations where fewer than `min_seeds`
    seeds reported a value are dropped so the right-hand tail truncates
    cleanly.
    """
    out: dict = {}
    for (model, strat, warm), sub in df.groupby(["model", "strategy", "warm_start"]):
        trajs = []
        for run_dir in sub["expected_run_dir"].dropna():
            try:
                run = load_run(run_dir)
                iters, rates = extractor(run)
                if rates:
                    trajs.append((iters, rates))
            except Exception as exc:
                click.echo(f"[warn] skip {run_dir}: {exc}", err=True)
        if len(trajs) < min_seeds:
            continue
        max_len = max(len(r) for _, r in trajs)
        Y = np.full((len(trajs), max_len), np.nan, dtype=float)
        for i, (_, rates) in enumerate(trajs):
            Y[i, :len(rates)] = rates
        longest_iters = next(its for its, r in trajs if len(r) == max_len)
        iters_ax = np.asarray(longest_iters[:max_len])
        n_per_iter = np.sum(~np.isnan(Y), axis=0)
        keep = n_per_iter >= min_seeds
        if not keep.any():
            continue
        out[(model, strat, warm)] = (iters_ax[keep], Y[:, keep])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def _draw_curve(ax, iters_ax, Y, *, color, linestyle, marker, label, uncertainty):
    lo, hi = _band(Y, uncertainty)
    mean = np.nanmean(Y, axis=0)
    ax.plot(iters_ax, mean, color=color, linestyle=linestyle, marker=marker,
            markersize=3, linewidth=1.5, label=label)
    ax.fill_between(iters_ax, lo, hi, color=color, alpha=0.15)


def _setup_axes(ax, ylabel, axhline_at_zero, y_min, y_max):
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if axhline_at_zero:
        ax.axhline(0.0, color="black", linewidth=0.5, linestyle=":")
    if y_min is not None or y_max is not None:
        cur_lo, cur_hi = ax.get_ylim()
        ax.set_ylim(cur_lo if y_min is None else y_min,
                    cur_hi if y_max is None else y_max)


def _finalize(fig, ax, out_path, ylabel, axhline_at_zero, y_min, y_max):
    _setup_axes(ax, ylabel, axhline_at_zero, y_min, y_max)

    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout()
    if handles:
        fig.subplots_adjust(right=0.78)
        fig.legend(handles, labels,
                   loc="center left", bbox_to_anchor=(0.79, 0.5),
                   fontsize=8, frameon=True, borderaxespad=0.)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_models_per_strategy(traj, uncertainty, out_dir,
                             metric_name, ylabel, title_word,
                             axhline_at_zero, y_min, y_max):
    written = []
    strategies = sorted({s for (_, s, _) in traj})
    for strat in strategies:
        cfgs = [(m, s, w) for (m, s, w) in traj if s == strat]
        if not cfgs:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(f"Strategy: {strat} — {title_word}", fontsize=12)

        for (m, s, w) in sorted(cfgs):
            iters_ax, Y = traj[(m, s, w)]
            _draw_curve(
                ax, iters_ax, Y,
                color=MODEL_COLORS.get(m, "gray"),
                linestyle=WARM_LS.get(w, "-"),
                marker=WARM_MARKER.get(w, "x"),
                label=f"{m}-{w} (n={len(Y)})",
                uncertainty=uncertainty,
            )

        out_path = out_dir / f"{metric_name}_strategy_{strat}.png"
        _finalize(fig, ax, out_path, ylabel, axhline_at_zero, y_min, y_max)
        written.append(out_path)
    return written


def _best_setting_for_model(traj, model):
    """Pick the (strategy, warm) for `model` with highest mean final value."""
    candidates = [(m, s, w) for (m, s, w) in traj if m == model]
    if not candidates:
        return None
    scored = []
    for (m, s, w) in candidates:
        _, Y = traj[(m, s, w)]
        scored.append(((s, w), float(np.nanmean(Y, axis=0)[-1])))
    if not scored:
        return None
    (s, w), score = max(scored, key=lambda kv: kv[1])
    return s, w, score


def plot_best_per_model(traj, uncertainty, out_dir,
                        metric_name, ylabel, title_word,
                        axhline_at_zero, y_min, y_max):
    models = sorted({m for (m, _, _) in traj})
    picks = []
    for model in models:
        chosen = _best_setting_for_model(traj, model)
        if chosen is None:
            continue
        s, w, score = chosen
        picks.append((model, s, w, score))

    if not picks:
        return []

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"Best setting per model — {title_word} (picked by mean final value)",
                 fontsize=12)

    for (m, s, w, _sc) in picks:
        iters_ax, Y = traj[(m, s, w)]
        _draw_curve(
            ax, iters_ax, Y,
            color=MODEL_COLORS.get(m, "gray"),
            linestyle="-",
            marker="o",
            label=f"{m}: {s}-{w} (n={len(Y)})",
            uncertainty=uncertainty,
        )

    out_path = out_dir / f"{metric_name}_best_per_model.png"
    _finalize(fig, ax, out_path, ylabel, axhline_at_zero, y_min, y_max)

    click.echo(f"[best-per-model picks: {title_word}]")
    for (m, s, w, sc) in picks:
        click.echo(f"  {m:12s} -> {s}-{w}  (final mean = {sc:.4f})")

    return [out_path]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
              show_default=True)
@click.option("--sweep-id", default=None,
              help="Filter to one sweep_id (default: all).")
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/all_runs/",
              show_default=True,
              help="Directory for the generated PNGs. For each metric, files "
                   "<metric>_strategy_<strategy>.png and "
                   "<metric>_best_per_model.png are written.")
@click.option("--uncertainty", default="sem",
              type=click.Choice(["sem", "sd", "iqr"]), show_default=True,
              help="Band: SEM (default), SD, or IQR across seeds.")
@click.option("--metrics", default=",".join(METRICS.keys()), show_default=True,
              help="Comma-separated metrics to plot. Available: "
                   + ", ".join(METRICS.keys()))
@click.option("--min-seeds", default=2, type=int, show_default=True,
              help="Drop groups with fewer reporting seeds than this.")
@click.option("--include-status", default="completed,running,timeout",
              show_default=True,
              help="Comma-separated statuses to include from the manifest. "
                   "`running` and `timeout` rows surface partial trajectories "
                   "alongside completed seeds.")
@click.option("--y-min", default=None, type=float,
              help="Clip y-axis lower bound (default: matplotlib autoscale).")
@click.option("--y-max", default=None, type=float,
              help="Clip y-axis upper bound (default: matplotlib autoscale).")
def main(manifest, sweep_id, output_dir, uncertainty, metrics,
         min_seeds, include_status, y_min, y_max):
    df = pd.read_csv(manifest)
    if sweep_id:
        df = df[df["sweep_id"].astype(str) == str(sweep_id)]
    allowed = {s.strip() for s in include_status.split(",") if s.strip()}
    df = df[df["status"].isin(allowed)]
    if df.empty:
        raise click.ClickException("no rows matched manifest filter")

    selected_metrics = [m.strip() for m in metrics.split(",") if m.strip()]
    unknown = [m for m in selected_metrics if m not in METRICS]
    if unknown:
        raise click.UsageError(
            f"unknown metric(s): {unknown}. Available: {list(METRICS)}"
        )

    out_dir = Path(output_dir)
    written = []
    for metric_name in selected_metrics:
        extractor, ylabel, title_word, axhline_at_zero = METRICS[metric_name]
        traj = _collect_trajectories(df, min_seeds, extractor)
        if not traj:
            click.echo(f"[warn] metric '{metric_name}': no groups passed "
                       f"min-seeds filter; skipping", err=True)
            continue
        written += plot_models_per_strategy(
            traj, uncertainty, out_dir,
            metric_name=metric_name, ylabel=ylabel, title_word=title_word,
            axhline_at_zero=axhline_at_zero, y_min=y_min, y_max=y_max,
        )
        written += plot_best_per_model(
            traj, uncertainty, out_dir,
            metric_name=metric_name, ylabel=ylabel, title_word=title_word,
            axhline_at_zero=axhline_at_zero, y_min=y_min, y_max=y_max,
        )

    if not written:
        raise click.ClickException("no plots produced — every metric had too few seeds")

    click.echo(f"[plot] wrote {len(written)} file(s) to {out_dir}")
    for p in written:
        click.echo(f"  {p}")


if __name__ == "__main__":
    main()
