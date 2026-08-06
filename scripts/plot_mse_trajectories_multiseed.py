"""Best-per-model MSE-loss trajectories, one panel per evaluation dataset.

Reads the per-iteration loss trajectories the AL drivers store in
``state.pt`` (no recomputation): for each best-per-model pick
(``mcmc_diagnostics.DEFAULT_AL_PICKS``) and each seed run in the manifest,
the AL model's MSE loss (transformed target space) on

    static_random  (``al_on_static_random_losses``)
    mcmc           (``al_on_mcmc_losses``)
    train          (``al_train_losses``)
    val            (``al_val_losses``, the AL model's own validation split)

is aggregated across seeds (mean +- SEM band, same convention and model
colours as the hit-rate plots) and drawn as a 2x2 figure over global AL
iteration. Note: losses are the run-time values, so no post-hoc sneutrino
veto can be applied here (unlike the hit-rate plots).

Usage (new-generation preview):
    python scripts/plot_mse_trajectories_multiseed.py \\
        --sweep-id 20260803_180047 \\
        --include-status completed,running,timeout,submitted \\
        --output-dir /ptmp/jwuerzin/analysis/all_runs/new_sweep_preview
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import click
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mcmc_diagnostics import DEFAULT_AL_PICKS, MODEL_DISPLAY  # noqa: E402

DATASET_KEYS = {
    "static_random": "al_on_static_random_losses",
    "mcmc": "al_on_mcmc_losses",
    "train": "al_train_losses",
    "val": "al_val_losses",
}
DATASET_TITLES = {
    "static_random": "static random eval set",
    "mcmc": "MCMC eval set",
    "train": "training set",
    "val": "own validation set",
}


def _to_floats(v) -> list[float]:
    out = []
    for x in list(v or []):
        try:
            out.append(float(x.item() if hasattr(x, "item") else x))
        except Exception:
            out.append(float("nan"))
    return out


@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
              show_default=True)
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/all_runs", show_default=True)
@click.option("--sweep-id", default=None,
              help="Only manifest rows whose sweep_id starts with this prefix.")
@click.option("--include-status", default="completed,running,timeout", show_default=True)
@click.option("--models", default=None,
              help="Comma list of picks to include (default: all DEFAULT_AL_PICKS).")
@click.option("--min-seeds", default=2, show_default=True)
@click.option("--logy/--no-logy", default=True, show_default=True)
def main(manifest, output_dir, sweep_id, include_status, models, min_seeds, logy):
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_hit_rate_trajectories_multiseed as phr

    statuses = {s.strip() for s in include_status.split(",")}
    picks = dict(DEFAULT_AL_PICKS)
    if models:
        wanted = {m.strip() for m in models.split(",")}
        picks = {m: sw for m, sw in picks.items() if m in wanted}

    rows = [r for r in csv.DictReader(open(manifest))
            if r["status"] in statuses
            and (sweep_id is None or r.get("sweep_id", "").startswith(sweep_id))]

    # {model: {ds: (iters, Y[n_seeds, n_iters])}}
    traj: dict = {}
    for model, (strat, warm) in picks.items():
        sel = [r for r in rows
               if (r["model"], r["strategy"], r["warm_start"]) == (model, strat, warm)]
        per_ds: dict = {ds: [] for ds in DATASET_KEYS}
        seen_dirs = set()
        for r in sel:
            d = r["expected_run_dir"]
            if d in seen_dirs:
                continue
            seen_dirs.add(d)
            state_path = Path(d) / "state.pt"
            if not state_path.exists():
                continue
            try:
                state = torch.load(state_path, weights_only=False, map_location="cpu")
            except Exception as exc:
                click.echo(f"[warn] skip {d}: {exc}", err=True)
                continue
            for ds, key in DATASET_KEYS.items():
                vals = _to_floats(state.get(key))
                if vals and np.isfinite(vals).any():
                    per_ds[ds].append(vals)
        kept = {ds: t for ds, t in per_ds.items() if len(t) >= min_seeds}
        if kept:
            traj[model] = kept
            click.echo(f"[mse] {model}/{strat}/{warm}: "
                       + ", ".join(f"{ds}:{len(t)} seeds" for ds, t in kept.items()))
        else:
            click.echo(f"[mse] {model}/{strat}/{warm}: no usable runs — skipped")

    if not traj:
        raise click.ClickException("no trajectories found for the given filters")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, ds in zip(axes.ravel(), DATASET_KEYS):
        for model, per_ds in traj.items():
            if ds not in per_ds:
                continue
            trajs = per_ds[ds]
            L = max(len(t) for t in trajs)
            Y = np.full((len(trajs), L), np.nan)
            for i, t in enumerate(trajs):
                Y[i, :len(t)] = t
            n = np.sum(~np.isnan(Y), axis=0)
            keep = n >= min_seeds
            x = np.arange(1, L + 1)[keep]
            mu = np.nanmean(Y, axis=0)[keep]
            sem = (np.nanstd(Y, axis=0, ddof=1) / np.sqrt(n))[keep]
            c = phr.MODEL_COLORS.get(model, "gray")
            ax.plot(x, mu, color=c, lw=1.6,
                    label=MODEL_DISPLAY.get(model, model))
            ax.fill_between(x, mu - sem, mu + sem, color=c, alpha=0.2, lw=0)
        if logy:
            ax.set_yscale("log")
        ax.set_title(DATASET_TITLES[ds], fontsize=11)
        ax.grid(alpha=0.25)
    for ax in axes[1]:
        ax.set_xlabel("AL iteration")
    for ax in axes[:, 0]:
        ax.set_ylabel("MSE loss (transformed space)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Best-per-model AL MSE-loss trajectories (seed mean ± SEM)", fontsize=13)
    fig.tight_layout()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "mse_best_per_model.png"
    fig.savefig(p, dpi=200)
    click.echo(f"[mse] wrote {p}")


if __name__ == "__main__":
    main()
