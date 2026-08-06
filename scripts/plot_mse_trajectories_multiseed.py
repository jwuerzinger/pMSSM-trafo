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
colours as the hit-rate plots) and drawn over global AL iteration: a 2x2
grid of dataset panels (AL model solid, the same-architecture random
baseline dashed), each with its own AL/baseline MSE-ratio subpanel
beneath (per-seed ratios, colour = model). Note:
losses are the run-time values, so no post-hoc sneutrino veto can be
applied here (unlike the hit-rate plots). No in-figure titles beyond the
panel labels (paper figures carry captions instead).

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
BASELINE_KEYS = {
    "static_random": "baseline_on_static_random_losses",
    "mcmc": "baseline_on_mcmc_losses",
    "train": "baseline_train_losses",
    "val": "baseline_val_losses",
}
DATASET_LINESTYLES = {"static_random": "-", "mcmc": "--", "train": ":", "val": "-."}
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

    # {model: {ds: {"al": [traj...], "base": [traj...], "ratio": [traj...]}}}
    traj: dict = {}
    for model, (strat, warm) in picks.items():
        sel = [r for r in rows
               if (r["model"], r["strategy"], r["warm_start"]) == (model, strat, warm)]
        per_ds: dict = {ds: {"al": [], "base": [], "ratio": []} for ds in DATASET_KEYS}
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
            for ds in DATASET_KEYS:
                al = np.asarray(_to_floats(state.get(DATASET_KEYS[ds])), dtype=float)
                ba = np.asarray(_to_floats(state.get(BASELINE_KEYS[ds])), dtype=float)
                if al.size and np.isfinite(al).any():
                    per_ds[ds]["al"].append(al)
                if ba.size and np.isfinite(ba).any():
                    per_ds[ds]["base"].append(ba)
                    L = min(len(al), len(ba))
                    if L and np.isfinite(al[:L]).any():
                        with np.errstate(divide="ignore", invalid="ignore"):
                            per_ds[ds]["ratio"].append(al[:L] / ba[:L])
        kept = {ds: t for ds, t in per_ds.items() if len(t["al"]) >= min_seeds}
        if kept:
            traj[model] = kept
            click.echo(f"[mse] {model}/{strat}/{warm}: "
                       + ", ".join(f"{ds}:{len(t['al'])}+{len(t['base'])} seeds"
                                   for ds, t in kept.items()))
        else:
            click.echo(f"[mse] {model}/{strat}/{warm}: no usable runs — skipped")

    if not traj:
        raise click.ClickException("no trajectories found for the given filters")

    def _band(trajs):
        L = max(len(t) for t in trajs)
        Y = np.full((len(trajs), L), np.nan)
        for i, t in enumerate(trajs):
            Y[i, :len(t)] = t
        n = np.sum(np.isfinite(Y), axis=0)
        keep = n >= min_seeds
        x = np.arange(1, L + 1)[keep]
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(Y, axis=0)[keep]
            sem = (np.nanstd(Y, axis=0, ddof=1) / np.sqrt(n))[keep]
        return x, mu, sem

    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(11.5, 9.5))
    outer = fig.add_gridspec(2, 2, hspace=0.24, wspace=0.20)
    first_loss_ax = None
    for cell, ds in zip([(i, j) for i in (0, 1) for j in (0, 1)], DATASET_KEYS):
        inner = outer[cell].subgridspec(2, 1, height_ratios=(2.6, 1), hspace=0.06)
        ax = fig.add_subplot(inner[0])
        axr = fig.add_subplot(inner[1], sharex=ax)
        if first_loss_ax is None:
            first_loss_ax = ax
        for model, per_ds in traj.items():
            if ds not in per_ds:
                continue
            c = phr.MODEL_COLORS.get(model, "gray")
            x, mu, sem = _band(per_ds[ds]["al"])
            ax.plot(x, mu, color=c, lw=1.6, label=MODEL_DISPLAY.get(model, model))
            ax.fill_between(x, mu - sem, mu + sem, color=c, alpha=0.2, lw=0)
            if per_ds[ds]["base"]:
                xb, mub, _ = _band(per_ds[ds]["base"])
                ax.plot(xb, mub, color=c, lw=1.1, ls="--", alpha=0.7)
            if per_ds[ds]["ratio"]:
                xr, mur, semr = _band(per_ds[ds]["ratio"])
                axr.plot(xr, mur, color=c, lw=1.2)
                axr.fill_between(xr, mur - semr, mur + semr, color=c, alpha=0.15, lw=0)
        if logy:
            ax.set_yscale("log")
        ax.set_title(DATASET_TITLES[ds], fontsize=11)
        ax.grid(alpha=0.25)
        ax.set_ylabel("MSE (transformed)", fontsize=9)
        ax.tick_params(labelbottom=False)
        axr.axhline(1.0, color="black", lw=0.8)
        axr.set_yscale("log")
        axr.set_ylabel("AL / base", fontsize=8)
        axr.grid(alpha=0.25)
        axr.set_xlabel("AL iteration", fontsize=9)
    handles, labels = first_loss_ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="gray", ls="--", lw=1.1))
    labels.append("random baseline (dashed)")
    first_loss_ax.legend(handles, labels, fontsize=7)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "mse_best_per_model.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    click.echo(f"[mse] wrote {p}")


if __name__ == "__main__":
    main()
