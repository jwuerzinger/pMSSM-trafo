"""Best-per-model MSE-loss trajectories, one panel per evaluation dataset.

Reads the per-iteration loss trajectories the AL drivers store in
``state.pt`` (no recomputation): for each best-per-model pick
(``mcmc_diagnostics.DEFAULT_AL_PICKS``) and each seed run in the manifest,
the AL model's MSE loss (transformed target space) on

    static_random  (``al_on_static_random_losses``)
    mcmc           (``al_on_mcmc_losses``)
    own_val        (``al_val_losses`` vs ``baseline_val_losses``: each model on
                   its own split, so different datasets, trends only)
    val_cross      (``al_val_losses`` vs ``base_on_al_val_losses``: both models
                   on the AL run's split, so a like-for-like comparison)

is aggregated across seeds (mean +- SEM band, same convention and model
colours as the hit-rate plots) and drawn over global AL iteration: a 2x2
grid of dataset panels (AL model solid, the same-architecture random
baseline dashed), each with its own AL/baseline MSE-ratio subpanel
beneath (the quotient of the two curves drawn above it, colour = model,
band by error propagation). Note:
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

from mcmc_diagnostics import DEFAULT_AL_PICKS, MODEL_DISPLAY, picks_with_tag  # noqa: E402

DATASET_KEYS = {
    "static_random": "al_on_static_random_losses",
    "mcmc": "al_on_mcmc_losses",
    "own_val": "al_val_losses",
    "val_cross": "al_val_losses",
}
BASELINE_KEYS = {
    "static_random": "baseline_on_static_random_losses",
    "mcmc": "baseline_on_mcmc_losses",
    # own_val: each model on ITS OWN validation split, so the two curves are on
    # different datasets and report each loop's own difficulty, not a like-for-
    # like comparison (the AL split is harder by construction).
    "own_val": "baseline_val_losses",
    # val_cross: BOTH models on the AL run's validation split, so the two curves
    # describe the same dataset and may be read against each other.
    "val_cross": "base_on_al_val_losses",
}
DATASET_LINESTYLES = {"static_random": "-", "mcmc": "--", "own_val": ":", "val_cross": "-."}
DATASET_TITLES = {
    "static_random": "static random eval set",
    "mcmc": "MCMC eval set",
    "own_val": "own validation set (each model on its own split)",
    "val_cross": "AL validation set (both models)",
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
@click.option("--model-tag", default="", show_default=True,
              help="OUTPUT_TAG of a variant sweep (e.g. 'expr'); re-keys the\n                   default per-model picks so tagged manifest rows resolve.")
@click.option("--models", default=None,
              help="Comma list of picks to include (default: all DEFAULT_AL_PICKS).")
@click.option("--min-seeds", default=2, show_default=True)
@click.option("--logy/--no-logy", default=True, show_default=True)
@click.option("--skip-empty-panels/--all-panels", default=False, show_default=True,
              help="Drop evaluation panels no model has data for and re-lay "
                   "out the grid, instead of drawing them empty. A target "
                   "with no posterior reference (ExpR) has no MCMC "
                   "trajectories, so its four-panel figure carries one blank "
                   "cell; this emits the three populated panels in a row.")
@click.option("--out-name", default="mse_best_per_model.png", show_default=True,
              help="Output file name inside --output-dir.")
@click.option("--mark-iteration", default=0, type=int, show_default=True,
              help="Draw a vertical rule at this iteration and label it. Used on "
                   "joint 40-iteration + extension figures to show where the "
                   "multi-seed benchmark ends and the resumed seeds continue "
                   "alone. 0 disables it.")
def main(manifest, output_dir, sweep_id, include_status, models, min_seeds, logy, model_tag,
         skip_empty_panels, out_name, mark_iteration):
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_hit_rate_trajectories_multiseed as phr

    statuses = {s.strip() for s in include_status.split(",")}
    picks = picks_with_tag(model_tag)
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
        per_ds: dict = {ds: {"al": [], "base": []} for ds in DATASET_KEYS}
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
    panels = list(DATASET_KEYS)
    if skip_empty_panels:
        panels = [ds for ds in DATASET_KEYS
                  if any(ds in per_ds for per_ds in traj.values())]
        dropped = [ds for ds in DATASET_KEYS if ds not in panels]
        if dropped:
            click.echo("[mse] dropping empty panel(s): " + ", ".join(dropped))
    if len(panels) == 4:
        nrows, ncols = 2, 2
        figsize = (12.5, 9.5)
    else:
        nrows, ncols = 1, max(1, len(panels))
        figsize = (6.25 * ncols, 4.75)
    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(nrows, ncols, hspace=0.26, wspace=0.30)
    cells = [(i, j) for i in range(nrows) for j in range(ncols)]
    first_loss_ax = None
    for cell, ds in zip(cells, panels):
        inner = outer[cell].subgridspec(2, 1, height_ratios=(2.6, 1), hspace=0.06)
        ax = fig.add_subplot(inner[0])
        axr = fig.add_subplot(inner[1], sharex=ax)
        ratio_tops: list[float] = []
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
                xb, mub, semb = _band(per_ds[ds]["base"])
                ax.plot(xb, mub, color=c, lw=1.1, ls="--", alpha=0.7)
                # The strip is the quotient of the two curves drawn above it, so
                # the two halves of a panel cannot disagree. Averaging per-seed
                # ratios instead reports mean(AL/base), which is not
                # mean(AL)/mean(base): with n=5 and one outlying Exact-GP
                # baseline run (MCMC MSE 4.31 against ~1.7-2.2 elsewhere) the two
                # differed in sign, the curves showing AL ahead at 0.914 while
                # the strip sat at 1.076.
                common, ia, ib = np.intersect1d(x, xb, return_indices=True)
                if common.size:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = mu[ia] / mub[ib]
                        # Propagated, treating the two means as independent; the
                        # AL/baseline pairing within a seed makes that mildly
                        # conservative.
                        semr = np.abs(ratio) * np.sqrt((sem[ia] / mu[ia]) ** 2
                                                       + (semb[ib] / mub[ib]) ** 2)
                    axr.plot(common, ratio, color=c, lw=1.2)
                    axr.fill_between(common, ratio - semr, ratio + semr,
                                     color=c, alpha=0.15, lw=0)
                    ratio_tops.append(np.nanmax(ratio[common >= 5])
                                      if (common >= 5).any() else np.nanmax(ratio))
        if mark_iteration:
            for a_ in (ax, axr):
                a_.axvline(mark_iteration, color="0.35", ls=":", lw=1.1, zorder=0)
            ax.annotate(f"benchmark ends ({mark_iteration} it.)",
                        xy=(mark_iteration, 0.02), xycoords=("data", "axes fraction"),
                        xytext=(4, 0), textcoords="offset points", rotation=90,
                        ha="left", va="bottom", fontsize=6.5, color="0.35")
        if logy:
            ax.set_yscale("log")
        ax.set_title(DATASET_TITLES[ds], fontsize=11)
        ax.grid(alpha=0.25)
        ax.set_ylabel("MSE (transformed)", fontsize=9)
        ax.tick_params(labelbottom=False)
        axr.axhline(1.0, color="black", lw=0.9, ls="--")
        # Linear scale, clipped to the post-burn-in range so the exact-GP
        # small-n interpolation artifact (ratio ~10^2 at iters 2-4) cannot
        # flatten the panel.
        if ratio_tops:
            axr.set_ylim(0, 1.15 * max(max(ratio_tops), 1.2))
        axr.set_ylabel("AL / base", fontsize=8, labelpad=2)
        axr.tick_params(axis="y", labelsize=7)
        axr.grid(alpha=0.25)
        axr.set_xlabel("AL iteration", fontsize=9)
    handles, labels = first_loss_ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="gray", ls="--", lw=1.1))
    labels.append("random baseline (dashed)")
    first_loss_ax.legend(handles, labels, fontsize=7)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / out_name
    fig.savefig(p, dpi=200, bbox_inches="tight")
    click.echo(f"[mse] wrote {p}")


if __name__ == "__main__":
    main()
