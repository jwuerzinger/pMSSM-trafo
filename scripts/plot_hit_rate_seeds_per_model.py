"""Plot every seed's hit-rate trajectory for a given model (no aggregation).

Same manifest + tolerance + metric machinery as
`plot_hit_rate_trajectories_multiseed.py`, but instead of mean ± band across
seeds, each seed gets its own line. Strategy/warm selection mirrors
`plot_best_per_model`: by default, for each model we pick the (strategy, warm)
with the highest mean final hit-rate at the strictest tolerance, and plot all
its seeds. Pass `--strategy` / `--warm-start` to override the pick.

Output: one PNG per model named `<metric>_seeds_<model>_<strategy>_<warm>.png`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analyse_runs import compute_hit_rate_trajectory, load_run  # noqa: E402
from pmssm import TARGET_CONFIG  # noqa: E402

# Reuse helpers and registry from the existing multiseed script so the metric
# definitions stay in sync.
from scripts.plot_hit_rate_trajectories_multiseed import (  # noqa: E402
    METRICS,
    MODEL_COLORS,
    _best_setting_for_model,
    _collect_trajectories,
    _enforce_marker_policy,
    _load_y_full,
    _markers_on,
    _pool_prevalence,
)


def _setup_axes(axes, tols, true_val, title_word, ylabel):
    for ax, tol in zip(axes, tols):
        ax.set_title(f"{title_word} (|Ω − {true_val}| / {true_val} < {int(tol*100)}%)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)


def _finalize(fig, axes, out_path):
    for ax in axes:
        _, ymax = ax.get_ylim()
        ax.set_ylim(0, max(ymax, 0.05) * 1.05)

    seen = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)

    fig.tight_layout()
    if seen:
        fig.subplots_adjust(right=0.82)
        fig.legend(seen.values(), seen.keys(),
                   loc="center left", bbox_to_anchor=(0.83, 0.5),
                   fontsize=8, frameon=True, borderaxespad=0.)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Per-seed panels mix horizons: seed 1 was resumed past 40 and the rest were
    # not, so without this the short seeds alone would carry markers.
    _enforce_marker_policy(fig)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _per_seed_trajectories(df_sub, true_val, tols, traj_fn):
    """Return {tol: list[(seed, iters, rates)]} for one (model, strat, warm) group."""
    runs = []
    for _, row in df_sub.iterrows():
        run_dir = row["expected_run_dir"]
        seed = row.get("seed")
        if pd.isna(run_dir):
            continue
        try:
            runs.append((seed, run_dir, load_run(run_dir)))
        except Exception as exc:
            click.echo(f"[warn] skip {run_dir}: {exc}", err=True)

    per_tol = {}
    for tol in tols:
        rows = []
        for seed, run_dir, run in runs:
            try:
                iters, rates = traj_fn(run, true_val, tol)
                if rates:
                    rows.append((seed, np.asarray(iters), np.asarray(rates)))
            except Exception as exc:
                click.echo(f"[warn] skip {run_dir} tol={tol}: {exc}", err=True)
        if rows:
            per_tol[tol] = rows
    return per_tol


def _per_seed_baseline_trajectories(df_sub, true_val, tols, traj_fn_baseline,
                                    Y_full):
    """Return {tol: list[(seed, iters, rates)]} of random-baseline trajectories."""
    per_tol = {}
    for tol in tols:
        rows = []
        for _, row in df_sub.iterrows():
            run_dir = row.get("expected_run_dir")
            seed = row.get("seed")
            if pd.isna(run_dir) or pd.isna(seed):
                continue
            try:
                iters, rates = traj_fn_baseline(run_dir, int(seed), Y_full,
                                                true_val, tol)
                if rates:
                    rows.append((seed, np.asarray(iters), np.asarray(rates)))
            except Exception as exc:
                click.echo(f"[warn] baseline skip {run_dir} tol={tol}: {exc}",
                           err=True)
        if rows:
            per_tol[tol] = rows
    return per_tol


def _draw_seed_curves(ax, rows, color):
    """Plot each (seed, iters, rates) as its own thin line."""
    n = len(rows)
    cmap = plt.get_cmap("viridis", max(n, 2))
    for i, (seed, iters, rates) in enumerate(sorted(rows, key=lambda r: (r[0] if r[0] is not None else i))):
        lbl = f"seed {int(seed)}" if seed is not None and not pd.isna(seed) else f"run {i}"
        # Line only once a run is long: at the extended budgets the per-point
        # markers merge into a band and hide the curve.
        _mk = "o" if _markers_on(len(list(iters))) else None
        ax.plot(iters, rates,
                color=cmap(i) if color is None else color,
                alpha=0.85, linewidth=1.2, marker=_mk, markersize=2.5,
                label=lbl)


def _draw_baseline_seed_curves(ax, rows, color):
    """Plot each (seed, iters, rates) as a dashed thin line (random baseline)."""
    n = len(rows)
    cmap = plt.get_cmap("viridis", max(n, 2))
    for i, (seed, iters, rates) in enumerate(sorted(rows, key=lambda r: (r[0] if r[0] is not None else i))):
        lbl = (f"seed {int(seed)}: random baseline"
               if seed is not None and not pd.isna(seed)
               else f"run {i}: random baseline")
        ax.plot(iters, rates,
                color=cmap(i) if color is None else color,
                alpha=0.55, linewidth=1.0, linestyle="--",
                label=lbl)


@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
              show_default=True)
@click.option("--sweep-id", default=None,
              help="Filter to one sweep_id (default: use all matching rows).")
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/all_runs/",
              show_default=True)
@click.option("--target", default="DMRD", show_default=True)
@click.option("--tolerances", default="0.10,0.20,0.50", show_default=True)
@click.option("--include-status", default="completed,running,timeout",
              show_default=True)
@click.option("--model", "models", multiple=True,
              help="Restrict to these models (repeatable). Default: all in manifest.")
@click.option("--strategy", default=None,
              help="Override the (strategy) pick. Default: best per model "
                   "(strictest-tolerance final mean).")
@click.option("--warm-start", default=None,
              help="Override the (warm) pick. Used together with --strategy.")
@click.option("--per-model-color", is_flag=True,
              help="Color all seed lines with the model's signature color "
                   "instead of viridis-by-seed.")
@click.option("--baseline-data-dir", default=None,
              help="ROOT data directory used to compute the random-scan "
                   "baseline pool. If set, each seed's dashed random-baseline "
                   "trajectory and a horizontal full-pool prevalence line "
                   "are overlaid.")
def main(manifest, sweep_id, output_dir, target, tolerances, include_status,
         models, strategy, warm_start, per_model_color, baseline_data_dir):
    df = pd.read_csv(manifest)
    if sweep_id:
        df = df[df["sweep_id"].astype(str) == str(sweep_id)]
    allowed = {s.strip() for s in include_status.split(",") if s.strip()}
    df = df[df["status"].isin(allowed)]
    if df.empty:
        raise click.ClickException("no rows matched manifest filter")

    tols = [float(t) for t in tolerances.split(",")]
    true_val = TARGET_CONFIG[target]["true_value"]
    out_dir = Path(output_dir)

    Y_full = None
    prevalence = None
    if baseline_data_dir:
        try:
            Y_full = _load_y_full(baseline_data_dir, target, out_dir)
            prevalence = _pool_prevalence(Y_full, true_val, tols)
            click.echo(f"[baseline] pool prevalence (n={len(Y_full)}): "
                       + ", ".join(f"tol={int(t*100)}%→{r:.4f}"
                                   for t, r in prevalence.items()))
        except Exception as exc:
            click.echo(f"[warn] could not load baseline pool from "
                       f"{baseline_data_dir}: {exc}", err=True)
            Y_full = None
            prevalence = None

    pick_models = list(models) if models else sorted(df["model"].dropna().unique())

    # For "best" selection we need the aggregated trajectories; only build them
    # if the user didn't pin (strategy, warm) explicitly.
    traj_for_pick = {}
    if strategy is None or warm_start is None:
        for metric_name, (traj_fn, _, _, _, _) in METRICS.items():
            traj_for_pick[metric_name] = _collect_trajectories(
                df, true_val, tols, min_seeds=1, traj_fn=traj_fn,
            )

    written = []
    for metric_name, (traj_fn, file_prefix, ylabel, title_word,
                       traj_fn_baseline) in METRICS.items():
        for model in pick_models:
            if strategy is not None and warm_start is not None:
                strat, warm = strategy, warm_start
                tag = "user-pick"
            else:
                chosen = _best_setting_for_model(
                    traj_for_pick[metric_name], model, tols,
                )
                if chosen is None:
                    click.echo(f"[warn] {metric_name}/{model}: no eligible runs", err=True)
                    continue
                strat, warm, tol_used, score = chosen
                tag = f"best@{int(tol_used*100)}%={score:.3f}"

            sub = df[(df["model"] == model)
                     & (df["strategy"] == strat)
                     & (df["warm_start"] == warm)]
            if sub.empty:
                click.echo(f"[warn] {metric_name}/{model}/{strat}-{warm}: no rows", err=True)
                continue

            per_tol = _per_seed_trajectories(sub, true_val, tols, traj_fn)
            if not per_tol:
                click.echo(f"[warn] {metric_name}/{model}/{strat}-{warm}: no trajectories", err=True)
                continue

            per_tol_baseline = {}
            if Y_full is not None:
                per_tol_baseline = _per_seed_baseline_trajectories(
                    sub, true_val, tols, traj_fn_baseline, Y_full,
                )

            fig, axes = plt.subplots(1, len(tols), figsize=(6 * len(tols), 5),
                                     sharey=False, squeeze=False)
            axes = list(axes.flat)
            _setup_axes(axes, tols, true_val, title_word, ylabel)
            n_seeds = max(len(rows) for rows in per_tol.values())
            fig.suptitle(
                f"{title_word} per seed — {model} ({strat}-{warm}, n={n_seeds}, {tag})",
                fontsize=12,
            )
            color = MODEL_COLORS.get(model, "gray") if per_model_color else None
            for ax, tol in zip(axes, tols):
                rows = per_tol.get(tol)
                if not rows:
                    continue
                _draw_seed_curves(ax, rows, color=color)
                b_rows = per_tol_baseline.get(tol)
                if b_rows:
                    _draw_baseline_seed_curves(ax, b_rows, color=color)
                if prevalence is not None and tol in prevalence:
                    ax.axhline(prevalence[tol], color="black", linestyle=":",
                               linewidth=1.4, label="random scan (full pool)")
                    ax.text(0.99, prevalence[tol], f" {prevalence[tol]:.4f}",
                            transform=ax.get_yaxis_transform(), ha="right",
                            va="bottom", fontsize=8, color="black")

            out_path = out_dir / f"{file_prefix}_seeds_{model}_{strat}_{warm}.png"
            _finalize(fig, axes, out_path)
            written.append(out_path)

    if not written:
        raise click.ClickException("no plots produced")

    click.echo(f"[plot] wrote {len(written)} file(s) to {out_dir}")
    for p in written:
        click.echo(f"  {p}")


if __name__ == "__main__":
    main()
