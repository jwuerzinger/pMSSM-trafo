"""Preliminary head/strategy arms drawn in the paper's style, one panel per model.

Three figures, all sharing the paper's grammar:

  prelim_hit_rate.png            "Hit rate"      -- in-band fraction of the
                                                 cumulative training set
  prelim_hits_per_desired.png    "Hits / Desired" -- same numerator over the
                                                 REQUESTED sample count
  prelim_accuracy_static_random.png               -- verdict accuracy

Nothing here re-derives a metric. ``compute_hit_rate_trajectory`` and
``_hits_per_desired_trajectory`` are imported from the paper's own plotter, so
the preliminary curves are the same quantity as the published ones and can be
read on the same axis. In particular hits/desired divides by
``n_samples + sum(n_select)`` rather than by a count of simulator calls parsed
out of a log: failed candidates never reach the numerator, so the requested
count already folds in the generation failure rate, and it needs no log at all.

Style
-----
Colour encodes the MODEL, from the paper's ``MODEL_COLORS``, so a model keeps
the hue it has in every published figure. Line style and marker encode the ARM,
which extends the grammar the paper already uses: there, linestyle carries the
warm/cold mode and the Laplace acquisition variant (``WARM_LS``, ``LAPLACE_LS``)
while colour stays with the model. Splitting into one panel per model would free
the colour axis, but reusing it for the arm would break that correspondence with
the rest of the document for no gain, since a panel already names its model.

Reference arms are averaged over every seed in the manifest with a SEM band, as
in the paper. The new arms are single-seed, so they are drawn as bare lines --
the absence of a band is itself the honest signal that they are one run.

Usage
-----
    python scripts/plot_prelim_paper_style.py \
        --headtest-glob '/ptmp/jwuerzin/output/headtest_*_20260821_*' \
        --output-dir /ptmp/jwuerzin/analysis/joint/prelim_20260821
"""
from __future__ import annotations

import csv
import glob as globmod
import json
import re
import sys
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from plot_hit_rate_trajectories_multiseed import (      # noqa: E402
    MODEL_COLORS, MODEL_DISPLAY, _draw_curve, _hits_per_desired_trajectory,
    _load_run,
)
from analyse_runs import compute_hit_rate_trajectory     # noqa: E402

# Panel order: neural surrogates first, since they are the only ones carrying a
# classification head, then the GPs.
MODEL_ORDER = ["transformer", "dnn", "dnn_match_trafo", "exact_gp", "deep_gp",
               "tabpfn"]

# Arm -> (colour, linestyle, marker, legend label).
#
# Colour encodes the ARM, not the model. The paper reserves colour for the model
# because its figures overlay several models on one axis; here every panel holds
# exactly one model and names it, so the colour axis is free and spending it on
# the arm is what makes the comparison legible. Distinguishing five arms by
# marker shape alone did not work on screen. Colours are the same tab: family
# the paper uses, and the accuracy figure of the head-test plotter already uses
# this arm mapping, so the two agree.
ARM_STYLE = {
    "entropy_batch":   ("tab:blue",   "-",  "o", "regression, entropy batch"),
    "top_k":           ("tab:cyan",   "-",  "o", "regression, top-k"),
    "tol_only_random": ("tab:orange", "-",  "D", "regression, tol + uniform"),
    "bald":            ("tab:green",  "-",  "^", "classification, BALD"),
    "cls_entropy":     ("tab:red",    "-",  "v", "classification, mean entropy"),
}
# The random draw: dotted, grey, thin, and behind everything.
BASELINE_STYLE = dict(color="0.45", linestyle=":", linewidth=1.4)
# Band prevalence of the valid random-scan pool, and the pool validity that
# converts it to the hits/desired denominator (which is a REQUESTED count, so a
# uniform draw also pays for the models it generates that are not valid).
POOL_PREVALENCE = 0.0336
POOL_PVALID = 0.5839
ARM_ORDER = ["entropy_batch", "top_k", "tol_only_random", "bald", "cls_entropy"]

_HEADTEST_MODEL = {"transformer": "transformer", "dnn": "dnn",
                   "dnnmatch": "dnn_match_trafo", "exactgp": "exact_gp",
                   "deepgp": "deep_gp", "tabpfn": "tabpfn"}
_HEADTEST_ARM = {"tolrand": "tol_only_random", "bald": "bald",
                 "clsent": "cls_entropy"}
# Manifest model column -> panel. The exclusion-boundary sweep carries the
# OUTPUT_TAG suffix, the relic-density sweep does not, so both spellings map to
# the same panel and one plotter serves both targets. Variant families
# (_oracle, _laplace) are deliberately absent: they are different experiments,
# not different arms of this one.
_PROD_MODEL = {"transformer_expr": "transformer", "dnn_expr": "dnn",
               "dnn_match_trafo_expr": "dnn_match_trafo",
               "exact_gp_expr": "exact_gp", "deep_gp_expr": "deep_gp",
               "tabpfn_expr": "tabpfn",
               "transformer": "transformer", "dnn": "dnn",
               "dnn_match_trafo": "dnn_match_trafo",
               "exact_gp": "exact_gp", "deep_gp": "deep_gp",
               "tabpfn": "tabpfn"}

# The arms this campaign added, as they are spelled in the manifest's strategy
# column. Reading them from the full sweep manifest is what turns them from the
# single-seed probe lines of 2026-08-21 into five-seed means with bands; until
# the campaign runs, the glob path below still supplies the probe runs.
NEW_ARMS = ("tol_only_random", "bald", "cls_entropy")


def iter_arm_rows(arm_manifest, arm_sweep_id=""):
    """Rows of a sweep manifest that belong to the new arms.

    Yields (panel_model, arm, run_dir). Filtered on the strategy so a full sweep
    manifest can be passed without dragging in every benchmark cell, and
    optionally on the sweep id so one campaign can be isolated from an earlier
    one that used the same strategy names.
    """
    import csv as _csv
    if not arm_manifest or not Path(arm_manifest).exists():
        return
    for r in _csv.DictReader(open(arm_manifest)):
        if r.get("strategy") not in NEW_ARMS:
            continue
        if arm_sweep_id and r.get("sweep_id") != arm_sweep_id:
            continue
        mdl = _PROD_MODEL.get(r.get("model", ""))
        d = Path(r.get("expected_run_dir") or "")
        if mdl and d.name and (d / "state.pt").exists():
            yield mdl, r["strategy"], d


def _stack(per_seed: list[tuple[list[int], list[float]]]):
    """(iters_axis, Y[n_seeds, n_iters]) padded with NaN to the longest seed."""
    per_seed = [(np.asarray(i, dtype=float), np.asarray(v, dtype=float))
                for i, v in per_seed if len(i)]
    if not per_seed:
        return np.array([]), np.zeros((0, 0))
    n = max(len(i) for i, _ in per_seed)
    axis = max((i for i, _ in per_seed), key=len)[:n]
    Y = np.full((len(per_seed), n), np.nan)
    for k, (_i, v) in enumerate(per_seed):
        Y[k, :len(v)] = v[:n]
    return axis, Y


def _stack_xy(per_seed, size_maps, use_size: bool):
    """(x_axis, Y[n_seeds, n]) with x either the iteration or the mean |L|.

    Plotting against the labelled-set size is the fairer comparison: the arms
    acquire different numbers of VALID points per iteration (validity ranges
    from 0.13 to 0.83 across these arms), so at a fixed iteration they hold
    different amounts of data and an iteration axis silently rewards whichever
    arm generated more. |L| is read per seed from ``state.pt`` and averaged,
    which is the same averaging the y values get.
    """
    entries = [(np.asarray(i, dtype=float), np.asarray(v, dtype=float), sm)
               for (i, v), sm in zip(per_seed, size_maps) if len(i)]
    if not entries:
        return np.array([]), np.zeros((0, 0))
    n = max(len(i) for i, _, _ in entries)
    Y = np.full((len(entries), n), np.nan)
    X = np.full((len(entries), n), np.nan)
    for k, (i, v, sm) in enumerate(entries):
        Y[k, :len(v)] = v[:n]
        if use_size:
            X[k, :len(i)] = [sm.get(int(t), np.nan) for t in i[:n]]
        else:
            X[k, :len(i)] = i[:n]
    with np.errstate(invalid="ignore"):
        x = np.nanmean(X, axis=0)
    return x, Y


def _size_map(state: dict) -> dict[int, int]:
    """iteration -> labelled set size |L| = n_train + n_val after it."""
    ntr = list(state.get("al_n_train") or [])
    nva = list(state.get("al_n_val") or [])
    return {k + 1: int(ntr[k]) + int(nva[k]) for k in range(min(len(ntr), len(nva)))}


def _accuracy_series(run_dir: Path, dataset: str, role: str = "al"):
    """(iters, values) of verdict accuracy from the run's cache.

    ``role="baseline"`` is the same architecture trained on RANDOM additions at
    the same budget, which is the accuracy figure's random reference.

    ``accuracy_posthoc.json`` wins over the run-time cache for the AL role when
    it exists. A verdict-head GP's run-time accuracy is the positive-class
    fraction rather than an accuracy, and the still-running job appends a fresh
    one every iteration, so any correction written into the run-time cache is
    overtaken within minutes. The post-hoc file only ever holds
    checkpoint-derived values, so preferring it makes the figure immune to that.
    """
    if role == "al":
        ph = run_dir / "accuracy_posthoc.json"
        if ph.exists():
            try:
                series = (json.loads(ph.read_text()) or {}).get(dataset) or {}
            except Exception:
                series = {}
            if series:
                keys = sorted((k for k in series if str(k).isdigit()),
                              key=lambda x: int(x))
                return ([int(k) for k in keys],
                        [float(series[k]) for k in keys])
    p = run_dir / "accuracy_trajectory.json"
    if not p.exists():
        return [], []
    try:
        cache = json.loads(p.read_text())
    except Exception:
        return [], []
    it, val = [], []
    for k in sorted(cache, key=lambda x: int(x)):
        v = (cache[k] or {}).get(role, {}).get(dataset)
        if v is not None and np.isfinite(v):
            it.append(int(k))
            val.append(float(v))
    return it, val


@click.command()
@click.option("--headtest-glob", default="/ptmp/jwuerzin/output/headtest_*",
              show_default=True)
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/joint/manifest_expr.csv",
              show_default=True)
@click.option("--true-value", default=1.0, show_default=True)
@click.option("--tolerance", default=0.10, show_default=True,
              help="Band half-width; 0.10 is the paper's strictest panel.")
@click.option("--dataset", default="static_random", show_default=True,
              help="Accuracy eval set. static_random is the only fair one: "
                   "train/val are each arm's own points.")
@click.option("--uncertainty", default="sem", show_default=True,
              type=click.Choice(["sem", "std"]))
@click.option("--exclude-accuracy", default="", show_default=True,
              help="model:arm cells to drop from the ACCURACY figure only. The "
                   "deep GP's Bernoulli arms were started before the fix that "
                   "made gp_predict return the latent, so their cached accuracy "
                   "is the positive-class fraction rather than an accuracy; one "
                   "such cell rescales every panel. The hit-rate figures are "
                   "unaffected, since they read the acquired points' true "
                   "labels and never the model's output.")
@click.option("--exclude-runs", default="", show_default=True,
              help="Comma-separated substrings of run-directory names to drop "
                   "entirely. Use it when a run has been superseded rather "
                   "than merely extended: the first Laplace GPC arms were "
                   "trained with a zero prior mean and a 400-epoch budget, "
                   "which left the model pinned at the majority class, so "
                   "their curves say nothing about the acquisition and would "
                   "be read as if they did.")
@click.option("--x-axis", default="iteration", show_default=True,
              type=click.Choice(["iteration", "dataset_size", "both"]),
              help="'dataset_size' plots against |L| = n_train + n_val, which "
                   "removes the differing per-iteration validity between arms.")
@click.option("--arm-manifest", default="", show_default=True,
              help="Full sweep manifest to read the new arms from, so they get "
                   "seed means and bands instead of one line per run. The probe "
                   "runs found by --headtest-glob are still included.")
@click.option("--arm-sweep-id", default="", show_default=True,
              help="Restrict --arm-manifest to one sweep id.")
@click.option("--output-dir", required=True)
def main(headtest_glob, manifest, true_value, tolerance, dataset, uncertainty,
         exclude_accuracy, exclude_runs, x_axis, arm_manifest,
         arm_sweep_id, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # (model, arm) -> {"hit_rate": [(it, v)...], "hits_per_desired": [...],
    #                  "accuracy": [...], "ref": bool}
    cells: dict[tuple[str, str], dict] = {}

    def _add(model, arm, run_dir: Path, is_ref: bool):
        try:
            run = _load_run(str(run_dir))
        except Exception as exc:
            click.echo(f"  [skip] {run_dir.name}: {exc}")
            return
        entry = cells.setdefault((model, arm), {"hit_rate": [], "accuracy": [],
                                                "hits_per_desired": [],
                                                "accuracy_baseline": [],
                                                "size": [], "ref": is_ref})
        try:
            entry["size"].append(_size_map(torch.load(
                run_dir / "state.pt", weights_only=False, map_location="cpu")))
        except Exception as exc:
            entry["size"].append({})
            click.echo(f"  [warn] {run_dir.name}: no |L| map "
                       f"({type(exc).__name__}: {exc}); the dataset-size "
                       "figure will omit this run")
        try:
            entry["hit_rate"].append(
                compute_hit_rate_trajectory(run, true_value, tolerance))
        except Exception:
            pass
        try:
            entry["hits_per_desired"].append(
                _hits_per_desired_trajectory(run, true_value, tolerance))
        except Exception:
            pass
        it, v = _accuracy_series(run_dir, dataset)
        if it:
            entry["accuracy"].append((it, v))
        bit, bv = _accuracy_series(run_dir, dataset, role="baseline")
        if bit:
            entry["accuracy_baseline"].append((bit, bv))

    drop_pats = [x.strip() for x in exclude_runs.split(",") if x.strip()]
    for d in sorted(globmod.glob(headtest_glob)):
        d = Path(d)
        m = re.match(r"headtest_([a-z]+)_([a-z]+)_seed(\d+)_", d.name)
        if not m or not (d / "state.pt").exists():
            continue
        if any(pat in d.name for pat in drop_pats):
            click.echo(f"  [drop] {d.name} (matched --exclude-runs)")
            continue
        _add(_HEADTEST_MODEL.get(m.group(1), m.group(1)),
             _HEADTEST_ARM.get(m.group(2), m.group(2)), d, False)

    if manifest and Path(manifest).exists():
        for r in csv.DictReader(open(manifest)):
            model = _PROD_MODEL.get(r["model"])
            d = Path(r["expected_run_dir"])
            if model and (d / "state.pt").exists():
                _add(model, r["strategy"], d, True)

    # is_ref=False on purpose: the "reference: N seeds" annotation is about the
    # benchmark arm each panel is measured against, and these are the arms being
    # measured. They aggregate over seeds exactly like the reference does.
    for model, arm, d in iter_arm_rows(arm_manifest, arm_sweep_id):
        if any(pat in d.name for pat in drop_pats):
            continue
        _add(model, arm, d, False)

    for (mdl, arm), e in sorted(cells.items()):
        click.echo(f"  {mdl:<16} {arm:<16} seeds={len(e['hit_rate'])} "
                   f"{'reference' if e['ref'] else 'new arm'}")

    figures = [
        ("hit_rate", "prelim_hit_rate",
         f"Hit rate (|Y/{true_value:g} - 1| ≤ {tolerance:g})"),
        ("hits_per_desired", "prelim_hits_per_desired",
         f"Hits / Desired (|Y/{true_value:g} - 1| ≤ {tolerance:g})"),
        ("accuracy", f"prelim_accuracy_{dataset}",
         f"Accuracy ($r_{{\\mathrm{{exp}}}}$ ≷ {true_value:g})"),
    ]
    drop_acc = set()
    for spec in (x.strip() for x in exclude_accuracy.split(",") if x.strip()):
        if ":" not in spec:
            raise click.UsageError(f"--exclude-accuracy {spec!r} must be model:arm")
        mdl, arm = spec.split(":", 1)
        if (mdl, arm) in cells:
            cells[(mdl, arm)]["accuracy"] = []
            drop_acc.add((mdl, arm))
    if drop_acc:
        click.echo("[exclude] accuracy dropped for " +
                   ", ".join(f"{m}/{a}" for m, a in sorted(drop_acc)))

    axes_wanted = (["iteration", "dataset_size"] if x_axis == "both"
                   else [x_axis])
    figures = [(m, f, y, ax) for (m, f, y) in figures for ax in axes_wanted]

    for metric, fname, ylabel, xmode in figures:
        models = [m for m in MODEL_ORDER
                  if any(k[0] == m and cells[k][metric] for k in cells)]
        if not models:
            click.echo(f"[skip] {metric}: nothing to draw")
            continue
        ncol = min(3, len(models))
        nrow = int(np.ceil(len(models) / ncol))
        # The paper draws one metric per 7.5x5 axis; a panel grid keeps that
        # per-panel size so a panel here is the same size as a published figure.
        fig, axes = plt.subplots(nrow, ncol, figsize=(7.5 * ncol, 5 * nrow),
                                 squeeze=False, sharex=True, sharey=True)
        flat = [a for row in axes for a in row]
        arms_seen: list[str] = []
        drew_baseline = False
        for ax, model in zip(flat, models):
            keys = [k for k in cells if k[0] == model and cells[k][metric]]
            keys.sort(key=lambda k: (ARM_ORDER.index(k[1])
                                     if k[1] in ARM_ORDER else 99))
            # The random draw first, so every arm is read against it.
            if metric == "hit_rate":
                ax.axhline(POOL_PREVALENCE, zorder=0, **BASELINE_STYLE)
                drew_baseline = True
            elif metric == "hits_per_desired":
                ax.axhline(POOL_PREVALENCE * POOL_PVALID, zorder=0,
                           **BASELINE_STYLE)
                drew_baseline = True
            else:
                # Accuracy has no constant baseline: the reference is the same
                # architecture trained on random additions at the same budget,
                # which the run itself records.
                bkey = next((k for k in keys if cells[k]["ref"]
                             and cells[k]["accuracy_baseline"]), None)
                bkey = bkey or next((k for k in keys
                                     if cells[k]["accuracy_baseline"]), None)
                if bkey:
                    baxis, bY = _stack_xy(cells[bkey]["accuracy_baseline"],
                                          cells[bkey]["size"],
                                          xmode == "dataset_size")
                    if len(baxis):
                        # _draw_curve takes no zorder; it is drawn first
                        # instead, which puts it behind the arms.
                        _draw_curve(ax, baxis, bY, marker=None, label=None,
                                    uncertainty=uncertainty, fill_alpha=0.0,
                                    **BASELINE_STYLE)
                        drew_baseline = True
            for key in keys:
                arm = key[1]
                color, ls, marker, _lbl = ARM_STYLE.get(
                    arm, ("0.25", "-", "x", arm))
                axis, Y = _stack_xy(cells[key][metric], cells[key]["size"],
                                    xmode == "dataset_size")
                if not len(axis) or not np.isfinite(axis).any():
                    continue
                n_seeds = Y.shape[0]
                _draw_curve(ax, axis, Y, color=color,
                            linestyle=ls, marker=marker, label=None,
                            uncertainty=uncertainty,
                            linewidth=1.6,
                            # A band over one seed is not an uncertainty.
                            fill_alpha=0.14 if n_seeds > 1 else 0.0)
                if arm not in arms_seen:
                    arms_seen.append(arm)
            # Only seed 1 was extended past the 40-iteration benchmark horizon,
            # so a reference curve silently stops being a 5-seed mean and
            # becomes one seed. That composition change puts a visible step in
            # the curve (+0.005 on the deep GP's accuracy, +0.008 on its
            # baseline) which is not a physical effect, so the panel has to say
            # how many seeds survive to the end rather than only how many it
            # started with.
            n_ref, n_last, it_last = 0, 0, 0
            for k in keys:
                if not cells[k]["ref"]:
                    continue
                axis_r, Y_r = _stack_xy(cells[k][metric], cells[k]["size"],
                                        xmode == "dataset_size")
                if not len(axis_r) or Y_r.shape[0] <= n_ref:
                    continue
                n_ref = Y_r.shape[0]
                n_last = int((~np.isnan(Y_r[:, -1])).sum())
                it_last = int(axis_r[-1]) if np.isfinite(axis_r[-1]) else 0
            tag = MODEL_DISPLAY.get(model, model)
            ref_tag = ""
            if n_ref > 1:
                ref_tag = f"  (reference: {n_ref} seeds"
                _unit = "|L|" if xmode == "dataset_size" else "iter"
                ref_tag += (f", {n_last} at {_unit} {it_last})"
                            if n_last and n_last != n_ref else ")")
            ax.annotate(f"{tag}{ref_tag}",
                        xy=(0.03, 0.96), xycoords="axes fraction",
                        va="top", ha="left", fontsize=10)
            if metric == "accuracy" and any(m == model for m, _ in drop_acc):
                gone = ", ".join(ARM_STYLE.get(a, (None, None, a))[2]
                                 for m, a in sorted(drop_acc) if m == model)
                ax.annotate(f"omitted: {gone}\n(cached accuracy inverted by a "
                            "now-fixed bug)",
                            xy=(0.03, 0.05), xycoords="axes fraction",
                            va="bottom", ha="left", fontsize=8, color="#B00020")
            ax.grid(alpha=0.3)
        for ax in flat[len(models):]:
            ax.axis("off")
        for r in range(nrow):
            axes[r][0].set_ylabel(ylabel)
        xlab = ("Labelled set size $|L|$" if xmode == "dataset_size"
                else "Iteration")
        for c in range(ncol):
            axes[nrow - 1][c].set_xlabel(xlab)

        handles = [Line2D([0], [0], color=ARM_STYLE[a][0],
                          linestyle=ARM_STYLE[a][1], marker=ARM_STYLE[a][2],
                          markersize=4, lw=1.8, label=ARM_STYLE[a][3])
                   for a in ARM_ORDER if a in arms_seen]
        if drew_baseline:
            handles.append(Line2D([0], [0], label="random draw"
                                  if metric != "accuracy"
                                  else "random additions (same budget)",
                                  **BASELINE_STYLE))
        fig.legend(handles=handles, loc="lower center",
                   ncol=min(len(handles), 4), frameon=False,
                   bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.08 if nrow > 1 else 0.12, 1, 1))
        suffix = "_vs_size" if xmode == "dataset_size" else ""
        png = out / f"{fname}{suffix}.png"
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        click.echo(f"[write] {png}")


if __name__ == "__main__":
    main()
