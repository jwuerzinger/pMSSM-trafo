"""Plot mean R² trajectories on the static MCMC eval set with uncertainty bands.

Reads `sweep_manifest.csv`, groups completed runs by (model, strategy, warm_start),
loads each seed's `al_on_mcmc_r2` list from `state.pt`, and renders:

  1. Models per strategy    — one figure per strategy overlaying every
                              (model, warm) combo. Color = model, ls = warm.
  2. Best setting per model — one figure with a single curve per model,
                              picking the setting that maximises mean final
                              R² on MCMC.

Sister script to `plot_hit_rate_trajectories_multiseed.py`; same manifest
filtering, seed-aggregation, and visual encoding so the two views stay
comparable.
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
    "dnn":         "tab:purple",
    "transformer_oracle": "tab:blue",
    "deep_gp_oracle":     "tab:green",
}
STRATEGY_COLORS = {
    "top_k":          "tab:blue",
    "top_k_tol_only": "tab:orange",
    "entropy_batch":  "tab:green",
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


def _band(Y: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (lo, hi) uncertainty bands for an (n_seeds, n_iters) array.

    NaN-aware: at each iteration only the seeds that reported a value contribute.
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


def _r2_mcmc_trajectory(run) -> tuple[list[int], list[float]]:
    """Return (iters, r2) for the AL model's MCMC eval R² per iteration.

    Filters out None / NaN entries so partial logs (e.g. when MCMC eval was
    skipped on a particular iteration) don't poison the seed average.
    """
    iters, rates = [], []
    for i, r2 in enumerate(run.al_mcmc_r2 or []):
        if r2 is None:
            continue
        try:
            r2f = float(r2)
        except (TypeError, ValueError):
            continue
        if np.isnan(r2f):
            continue
        iters.append(i + 1)
        rates.append(r2f)
    return iters, rates


def _collect_trajectories(df, min_seeds):
    """Build {(model, strategy, warm): (iters_axis, Y[n_seeds, n_iters])}.

    Trajectories of different lengths (e.g. partially-completed runs whose
    status is ``running`` or ``timeout``) are NaN-padded to the longest seed's
    length so that each per-iteration mean / band uses whichever seeds have
    data at that iter. Iterations where fewer than `min_seeds` seeds reported
    a value are dropped from the output, so the right-hand tail truncates
    cleanly when only one or two seeds got further than the rest.
    """
    out: dict = {}
    for (model, strat, warm), sub in df.groupby(["model", "strategy", "warm_start"]):
        trajs = []
        for run_dir in sub["expected_run_dir"].dropna():
            try:
                run = load_run(run_dir)
                iters, rates = _r2_mcmc_trajectory(run)
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


def _draw_curve(ax, iters_ax, Y, *, color, linestyle, marker, label, uncertainty):
    lo, hi = _band(Y, uncertainty)
    mean = np.nanmean(Y, axis=0)
    ax.plot(iters_ax, mean, color=color, linestyle=linestyle, marker=marker,
            markersize=3, linewidth=1.5, label=label)
    ax.fill_between(iters_ax, lo, hi, color=color, alpha=0.15)


def _setup_axes(ax, y_min, y_max):
    ax.set_xlabel("Iteration")
    ax.set_ylabel("R² (MCMC eval)")
    ax.grid(alpha=0.3)
    ax.axhline(0.0, color="black", linewidth=0.5, linestyle=":")
    if y_min is not None or y_max is not None:
        cur_lo, cur_hi = ax.get_ylim()
        ax.set_ylim(cur_lo if y_min is None else y_min,
                    cur_hi if y_max is None else y_max)


def _finalize(fig, ax, out_path, y_min, y_max):
    _setup_axes(ax, y_min, y_max)

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


def plot_models_per_strategy(traj, uncertainty, out_dir, y_min, y_max):
    """One figure per strategy; lines are (model, warm) combos for that strategy."""
    written = []
    strategies = sorted({s for (_, s, _) in traj})
    for strat in strategies:
        cfgs = [(m, s, w) for (m, s, w) in traj if s == strat]
        if not cfgs:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(f"Strategy: {strat} — R² on MCMC eval", fontsize=12)

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

        out_path = out_dir / f"r2_mcmc_strategy_{strat}.png"
        _finalize(fig, ax, out_path, y_min, y_max)
        written.append(out_path)
    return written


def _best_setting_for_model(traj, model):
    """Pick the (strategy, warm) for `model` with highest mean final R²."""
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


def plot_best_per_model(traj, uncertainty, out_dir, y_min, y_max):
    """Single figure: one curve per model using its best (strategy, warm) setting."""
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
    fig.suptitle("Best setting per model (picked by mean final R² on MCMC)",
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

    out_path = out_dir / "r2_mcmc_best_per_model.png"
    _finalize(fig, ax, out_path, y_min, y_max)

    click.echo("[best-per-model picks: R² MCMC]")
    for (m, s, w, sc) in picks:
        click.echo(f"  {m:12s} -> {s}-{w}  (final mean R² = {sc:.4f})")

    return [out_path]


@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
              show_default=True)
@click.option("--sweep-id", default=None,
              help="Filter to one sweep_id (default: use all completed rows).")
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/all_runs/",
              show_default=True,
              help="Directory for the generated PNGs. Files written: "
                   "r2_mcmc_strategy_<strategy>.png (one per strategy) and "
                   "r2_mcmc_best_per_model.png.")
@click.option("--uncertainty", default="sem",
              type=click.Choice(["sem", "sd", "iqr"]), show_default=True,
              help="Band: SEM (default), SD, or IQR across seeds.")
@click.option("--min-seeds", default=2, type=int, show_default=True,
              help="Drop groups with fewer completed seeds than this.")
@click.option("--include-status", default="completed,running,timeout",
              show_default=True,
              help="Comma-separated statuses to include from the manifest. "
                   "`running` and `timeout` rows surface partial trajectories "
                   "alongside completed seeds; the per-iteration band uses "
                   "whichever seeds have data at that iter.")
@click.option("--y-min", default=None, type=float,
              help="Clip y-axis lower bound (default: matplotlib autoscale). "
                   "Useful because early-iteration MCMC R² can be very negative.")
@click.option("--y-max", default=None, type=float,
              help="Clip y-axis upper bound (default: matplotlib autoscale).")
def main(manifest, sweep_id, output_dir, uncertainty, min_seeds, include_status,
         y_min, y_max):
    df = pd.read_csv(manifest)
    if sweep_id:
        df = df[df["sweep_id"].astype(str) == str(sweep_id)]
    allowed = {s.strip() for s in include_status.split(",") if s.strip()}
    df = df[df["status"].isin(allowed)]
    if df.empty:
        raise click.ClickException("no rows matched manifest filter")

    traj = _collect_trajectories(df, min_seeds)
    if not traj:
        raise click.ClickException(
            "no (model, strategy, warm) groups had enough seeds with logged MCMC R²"
        )

    out_dir = Path(output_dir)
    written = []
    written += plot_models_per_strategy(traj, uncertainty, out_dir, y_min, y_max)
    written += plot_best_per_model(traj, uncertainty, out_dir, y_min, y_max)

    click.echo(f"[plot] wrote {len(written)} file(s) to {out_dir}")
    for p in written:
        click.echo(f"  {p}")


if __name__ == "__main__":
    main()
