"""Plot mean hit-rate trajectories with uncertainty bands over N seeds per config.

Reads `sweep_manifest.csv`, groups completed runs by (model, strategy, warm_start),
loads each seed's trajectory via `analyse_runs.compute_hit_rate_trajectory`,
truncates to the shortest trajectory in the group, and renders three panels
(one per tolerance).

Visual encoding:
  - colour    : model        (transformer, exact_gp, deep_gp, tabpfn)
  - linestyle : strategy     (top_k '-', top_k_tol_only '--', entropy_batch ':')
  - marker    : warm_start   (warm 'o', cold 's', tabpfn '^')

Groups with fewer than --min-seeds completed runs are skipped with a warning
(so a partially-finished sweep still produces a readable plot).
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

from analyse_runs import compute_hit_rate_trajectory, load_run  # noqa: E402
from pmssm import TARGET_CONFIG  # noqa: E402


MODEL_COLORS = {
    "transformer": "tab:blue",
    "exact_gp":    "tab:orange",
    "deep_gp":     "tab:green",
    "tabpfn":      "tab:red",
}
STRATEGY_LS = {
    "top_k":           "-",
    "top_k_tol_only":  "--",
    "entropy_batch":   ":",
}
WARM_MARKER = {
    "warm":   "o",
    "cold":   "s",
    "tabpfn": "^",
}


def _band(Y: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (lo, hi) uncertainty bands for an (n_seeds, n_iters) array."""
    mean = Y.mean(axis=0)
    if mode == "sem":
        sd = Y.std(axis=0, ddof=1) if len(Y) > 1 else np.zeros_like(mean)
        half = sd / np.sqrt(max(len(Y), 1))
        return mean - half, mean + half
    if mode == "sd":
        sd = Y.std(axis=0, ddof=1) if len(Y) > 1 else np.zeros_like(mean)
        return mean - sd, mean + sd
    if mode == "iqr":
        return np.percentile(Y, 25, axis=0), np.percentile(Y, 75, axis=0)
    raise ValueError(f"unknown uncertainty mode: {mode}")


@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
              show_default=True)
@click.option("--sweep-id", default=None,
              help="Filter to one sweep_id (default: use all completed rows).")
@click.option("--output", default="/ptmp/jwuerzin/analysis/all_runs/hit_rate_trajectories.png",
              show_default=True)
@click.option("--uncertainty", default="sem",
              type=click.Choice(["sem", "sd", "iqr"]), show_default=True,
              help="Band: SEM (default), SD, or IQR across seeds.")
@click.option("--target", default="DMRD", show_default=True,
              help="TARGET_CONFIG key (threshold + true_value source).")
@click.option("--tolerances", default="0.10,0.20,0.50", show_default=True,
              help="Comma-separated relative tolerances for hit-rate panels.")
@click.option("--min-seeds", default=2, type=int, show_default=True,
              help="Drop groups with fewer completed seeds than this.")
@click.option("--include-status", default="completed", show_default=True,
              help="Comma-separated statuses to include from the manifest.")
def main(manifest, sweep_id, output, uncertainty, target, tolerances,
         min_seeds, include_status):
    df = pd.read_csv(manifest)
    if sweep_id:
        df = df[df["sweep_id"].astype(str) == str(sweep_id)]
    allowed = {s.strip() for s in include_status.split(",") if s.strip()}
    df = df[df["status"].isin(allowed)]
    if df.empty:
        raise click.ClickException("no rows matched manifest filter")

    tols = [float(t) for t in tolerances.split(",")]
    true_val = TARGET_CONFIG[target]["true_value"]

    fig, axes = plt.subplots(1, len(tols),
                             figsize=(6 * len(tols), 5),
                             sharey=False)
    if len(tols) == 1:
        axes = [axes]

    groups = df.groupby(["model", "strategy", "warm_start"], sort=True)
    plotted, skipped = 0, 0

    for ax, tol in zip(axes, tols):
        ax.set_title(f"Hit rate (|Ω − {true_val}| / {true_val} < {int(tol*100)}%)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Hit rate")
        ax.grid(alpha=0.3)

        for (model, strat, warm), sub in groups:
            trajs = []
            for run_dir in sub["expected_run_dir"].dropna():
                try:
                    run = load_run(run_dir)
                    iters, rates = compute_hit_rate_trajectory(run, true_val, tol)
                    if rates:
                        trajs.append((iters, rates))
                except Exception as exc:  # corrupted state.pt, partial run, etc.
                    click.echo(f"[warn] skip {run_dir}: {exc}", err=True)

            if len(trajs) < min_seeds:
                skipped += 1
                continue

            min_len = min(len(r) for _, r in trajs)
            iters_ax = trajs[0][0][:min_len]
            Y = np.array([r[:min_len] for _, r in trajs])
            lo, hi = _band(Y, uncertainty)
            mean = Y.mean(axis=0)

            color = MODEL_COLORS.get(model, "gray")
            ls = STRATEGY_LS.get(strat, "-")
            mk = WARM_MARKER.get(warm, "x")
            lbl = f"{model}-{strat}-{warm} (n={len(Y)})"

            ax.plot(iters_ax, mean, color=color, linestyle=ls, marker=mk,
                    markersize=3, linewidth=1.5, label=lbl)
            ax.fill_between(iters_ax, lo, hi, color=color, alpha=0.15)
            plotted += 1

    # Tighten y-axis AFTER plotting so autoscale has run; add 5% headroom.
    for ax in axes:
        _, ymax = ax.get_ylim()
        ax.set_ylim(0, max(ymax, 0.05) * 1.05)

    # Collect deduped handles/labels from all axes (any panel has the full set).
    seen = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)

    fig.tight_layout()
    if seen:
        # Reserve explicit real-estate for a single external legend; no
        # bbox_inches='tight' to avoid fighting with tight_layout.
        fig.subplots_adjust(right=0.84)
        fig.legend(seen.values(), seen.keys(),
                   loc="center left", bbox_to_anchor=(0.85, 0.5),
                   fontsize=8, frameon=True, borderaxespad=0.)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # plotted counts per panel (so divide by n_tols)
    n_groups_drawn = plotted // max(len(tols), 1)
    click.echo(f"[plot] wrote {out_path} "
               f"(groups drawn: {n_groups_drawn}, "
               f"skipped (too few seeds): {skipped // max(len(tols),1)})")


if __name__ == "__main__":
    main()
