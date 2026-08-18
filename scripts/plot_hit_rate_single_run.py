"""Single-run analogue of `plot_hit_rate_trajectories_multiseed.py`.

Renders `hit_rate_*` and `hits_per_desired_*` panels for one run directory:
the AL curve, the per-run random-baseline curve, and the full-pool prevalence
reference line. Helpers from the multiseed script are reused unchanged so the
metric definitions stay in sync.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analyse_runs import compute_hit_rate_trajectory, load_run  # noqa: E402
from pmssm import TARGET_CONFIG  # noqa: E402
from scripts.plot_hit_rate_trajectories_multiseed import (  # noqa: E402
    MODEL_COLORS,
    _baseline_hit_rate_trajectory,
    _hits_per_desired_trajectory,
    _load_y_full,
    _markers_on,
    _pool_prevalence,
)


METRICS = {
    "hit_rate": (compute_hit_rate_trajectory, "Hit rate", "hit_rate"),
    "hits_per_desired": (_hits_per_desired_trajectory, "Hits / Desired", "hits_per_desired"),
}


def _setup_axes(axes, tols, true_val, ylabel, title_word):
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
        fig.subplots_adjust(right=0.80, wspace=0.28)
        fig.legend(seen.values(), seen.keys(),
                   loc="center left", bbox_to_anchor=(0.81, 0.5),
                   fontsize=9, frameon=True, borderaxespad=0.)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@click.command()
@click.option("--run-dir", required=True,
              help="Active-learning run directory containing state.pt.")
@click.option("--seed", default=42, type=int, show_default=True,
              help="Master seed used for the run (controls baseline replay).")
@click.option("--model-label", default="transformer", show_default=True,
              help="Model name; used only for legend colour & label.")
@click.option("--strategy-label", default="top_k", show_default=True)
@click.option("--warm-label", default="warm", show_default=True)
@click.option("--target", default="DMRD", show_default=True)
@click.option("--tolerances", default="0.10,0.20,0.50", show_default=True)
@click.option("--baseline-data-dir", default="/ptmp/jwuerzin/data/18387358",
              show_default=True,
              help="ROOT data directory for the random-scan baseline. "
                   "Empty string disables baseline + pool-prevalence overlays.")
@click.option("--cache-dir", default="/ptmp/jwuerzin/analysis/all_runs",
              show_default=True,
              help="Where to read/write the Y_full .npy cache.")
@click.option("--output-dir", default=None,
              help="Output directory for PNGs (default: <run-dir>).")
def main(run_dir, seed, model_label, strategy_label, warm_label, target,
         tolerances, baseline_data_dir, cache_dir, output_dir):
    run_dir = Path(run_dir)
    out_dir = Path(output_dir) if output_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tols = [float(t) for t in tolerances.split(",")]
    true_val = TARGET_CONFIG[target]["true_value"]

    Y_full = None
    prevalence = None
    if baseline_data_dir:
        try:
            Y_full = _load_y_full(baseline_data_dir, target, Path(cache_dir))
            prevalence = _pool_prevalence(Y_full, true_val, tols)
            click.echo("[baseline] pool prevalence "
                       + ", ".join(f"tol={int(t*100)}%→{r:.4f}" for t, r in prevalence.items()))
        except Exception as exc:
            click.echo(f"[warn] could not load baseline pool: {exc}", err=True)

    run = load_run(str(run_dir))
    color = MODEL_COLORS.get(model_label, "tab:blue")
    cfg_label = f"{model_label}: {strategy_label}-{warm_label}"

    written = []
    for metric, (traj_fn, title_word, file_prefix) in METRICS.items():
        fig, axes = plt.subplots(1, len(tols), figsize=(6 * len(tols), 5),
                                 sharey=False, squeeze=False)
        axes = list(axes.flat)
        _setup_axes(axes, tols, true_val, title_word, title_word)
        fig.suptitle(f"{title_word} — {run_dir.name}", fontsize=11)

        for ax, tol in zip(axes, tols):
            iters, rates = traj_fn(run, true_val, tol)
            if rates:
                ax.plot(iters, rates, color=color, linestyle="-",
                        marker="o" if _markers_on(len(rates)) else None,
                        markersize=3, linewidth=1.6, label=cfg_label)

            if Y_full is not None:
                try:
                    b_iters, b_rates = _baseline_hit_rate_trajectory(
                        str(run_dir), seed, Y_full, true_val, tol,
                    )
                    if b_rates:
                        ax.plot(b_iters, b_rates, color=color, linestyle="--",
                                linewidth=1.8, alpha=0.85,
                                label=f"{model_label}: random baseline")
                except Exception as exc:
                    click.echo(f"[warn] baseline tol={tol}: {exc}", err=True)

            if prevalence is not None and tol in prevalence:
                ax.axhline(prevalence[tol], color="black", linestyle=":",
                           linewidth=1.4, label="random scan (full pool)")
                ax.text(0.99, prevalence[tol], f" {prevalence[tol]:.4f}",
                        transform=ax.get_yaxis_transform(), ha="right",
                        va="bottom", fontsize=8, color="black")

        out_path = out_dir / f"{file_prefix}_single_run.png"
        _finalize(fig, axes, out_path)
        written.append(out_path)

    click.echo(f"[plot] wrote {len(written)} file(s):")
    for p in written:
        click.echo(f"  {p}")


if __name__ == "__main__":
    main()
