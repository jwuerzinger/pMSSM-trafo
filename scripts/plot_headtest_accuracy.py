"""Verdict-accuracy trajectories for the head/strategy test arms, from cache only.

The question this answers: does training the surrogate as a *classifier* of
`1[r_exp > 1]` give a better verdict than regressing `log r_exp` and
thresholding it, at equal labelling budget?

Why no GPU and no checkpoint reload is needed
---------------------------------------------
Every AL driver writes `<run_dir>/accuracy_trajectory.json` as it runs
(`pmssm.accuracy.write_iter_accuracies`), holding binary accuracy at the
constraint threshold for the AL and Baseline models on each eval set. This
script reads nothing else: no `state.pt`, no pool, no model.

Regression and classification arms are directly comparable because every head
keeps one scalar output and the same decision point (raw output > 0): a logit is
positive exactly when p > 0.5, and for ExpR the transform is `t = log(r/1)`, so
`exp(t) >= 1` is the verdict for both heads (see the "Diagnostics parity"
section of `pmssm/heads.py`). The accuracy numbers therefore sit on one axis
with no rescaling.

Eval sets
---------
`static_random` is the default and the only fair panel: the run's own held-out
slice of the pool, carved as rows [n_samples, n_samples + static_eval_size) of
the seed-shuffled pool, so two runs on the same pool and seed score the *same*
100k models. `train` and `val` are per-arm sets of the points that arm chose,
so they are harder for the arms that acquire harder points and cannot rank arms
against each other; they are available but off by default.

`mcmc` is deliberately not offered. ExpR has `has_mcmc_reference: False` — the
only chain on disk is the relic-density posterior, so an MCMC accuracy panel
for this target scores the exclusion verdict on the wrong sampler's points.
Some ExpR caches carry `mcmc` entries written by an earlier offline pass; they
are skipped here.

Usage
-----
    P=/ptmp/jwuerzin/pixi-envs/pytorch-conda-forge-2863954108128992291/envs/rocm/bin/python
    $P scripts/plot_headtest_accuracy.py \
        --output-dir /ptmp/jwuerzin/analysis/joint/headtest_acc
"""
from __future__ import annotations

import csv
import glob as globmod
import json
import re
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


# Canonical model name per `headtest_<MODEL>_...` token, so an arm lands in the
# same panel as the benchmark cell it is meant to be compared against.
_HEADTEST_MODEL = {
    "transformer": "transformer",
    "dnn": "dnn",
    "dnnmatch": "dnn_match_trafo",
    "exactgp": "exact_gp",
    "deepgp": "deep_gp",
    "tabpfn": "tabpfn",
}
# Arm token in the run-dir name -> the driver's --selection-strategy value.
_HEADTEST_ARM = {
    "tolrand": "tol_only_random",
    "bald": "bald",
    "clsent": "cls_entropy",
}

MODEL_DISPLAY = {
    "transformer": "Transformer",
    "dnn": "DNN",
    "dnn_match_trafo": "DNN (matched)",
    "exact_gp": "Exact GP",
    "deep_gp": "Deep GP",
    "tabpfn": "TabPFN",
}
# Colour encodes the ARM, since that is what the figure compares; the panel
# already fixes the model. Head is named in the legend because it is the point.
ARM_STYLE = {
    "entropy_batch":   ("tab:blue",   "-",  "o", "regression, entropy batch"),
    "top_k":           ("tab:cyan",   "-",  "o", "regression, top-k"),
    "tol_only_random": ("tab:orange", "-",  "D", "regression, tol + uniform"),
    "bald":            ("tab:green",  "-",  "^", "classification, BALD"),
    "cls_entropy":     ("tab:red",    "-",  "v", "classification, mean entropy"),
}
BASELINE_STYLE = dict(color="0.45", linestyle="--", linewidth=1.2, marker=None)
# Panel order: neural surrogates first (the only ones with a classification
# head), then the GPs, so the arms that have more than one curve read first.
MODEL_ORDER = ["transformer", "dnn", "dnn_match_trafo", "exact_gp", "deep_gp",
               "tabpfn"]


def _load_cache(run_dir: Path) -> dict:
    p = run_dir / "accuracy_trajectory.json"
    if not p.exists():
        return {}
    try:
        with open(p) as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except Exception as exc:
        click.echo(f"[warn] unreadable cache {p}: {exc}", err=True)
        return {}


def _series(run_dir: Path, role: str, dataset: str) -> dict[int, float]:
    """{iteration: accuracy} for one (run, role, dataset), from the cache."""
    out: dict[int, float] = {}
    for key, entry in _load_cache(run_dir).items():
        if not str(key).isdigit() or not isinstance(entry, dict):
            continue
        v = (entry.get(role) or {}).get(dataset)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f:  # not NaN
            out[int(key)] = f
    return out


def _stack(per_seed: list[dict[int, float]]) -> tuple[np.ndarray, np.ndarray]:
    """(iters, [n_seeds, n_iters]) NaN-padded matrix over a cell's seeds."""
    iters = sorted({i for s in per_seed for i in s})
    if not iters:
        return np.array([]), np.zeros((0, 0))
    M = np.full((len(per_seed), len(iters)), np.nan)
    for r, s in enumerate(per_seed):
        for c, i in enumerate(iters):
            if i in s:
                M[r, c] = s[i]
    return np.asarray(iters), M


def _draw(ax, iters, M, *, color, linestyle, marker, label, uncertainty="sem",
          linewidth=1.7, fill_alpha=0.15, mark_every=None):
    if not len(iters):
        return
    mean = np.nanmean(M, axis=0)
    n = (~np.isnan(M)).sum(axis=0)
    if uncertainty == "sem":
        with np.errstate(invalid="ignore"):
            err = np.nanstd(M, axis=0, ddof=0) / np.sqrt(np.maximum(n, 1))
    else:
        err = np.nanstd(M, axis=0, ddof=0)
    if mark_every is None:
        mark_every = max(1, len(iters) // 12)
    ax.plot(iters, mean, color=color, linestyle=linestyle, marker=marker,
            markersize=4.0, markevery=mark_every, linewidth=linewidth,
            label=label)
    multi = (n > 1)
    if multi.any():
        lo = np.where(multi, mean - err, np.nan)
        hi = np.where(multi, mean + err, np.nan)
        ax.fill_between(iters, lo, hi, color=color, alpha=fill_alpha, lw=0)


def _collect_reference(manifest: Path, dataset: str,
                       seeds: set[int] | None) -> dict:
    """{(model, strategy): {'al': (iters, M), 'baseline': (iters, M), 'seeds': n}}
    for the benchmark cells, averaged over the seeds present."""
    if not manifest.exists():
        click.echo(f"[warn] no reference manifest at {manifest}", err=True)
        return {}
    cells: dict[tuple[str, str], dict[str, list]] = {}
    n_rows = 0
    for row in csv.DictReader(open(manifest)):
        seed = int(row["seed"])
        if seeds is not None and seed not in seeds:
            continue
        # Manifest model names carry the sweep's OUTPUT_TAG ("transformer_expr").
        model = re.sub(r"_expr$", "", row["model"])
        key = (model, row["strategy"])
        run_dir = Path(row["expected_run_dir"])
        cells.setdefault(key, {"al": [], "baseline": []})
        cells[key]["al"].append(_series(run_dir, "al", dataset))
        cells[key]["baseline"].append(_series(run_dir, "baseline", dataset))
        n_rows += 1
    out = {}
    for key, roles in cells.items():
        entry = {"seeds": sum(1 for s in roles["al"] if s)}
        for role in ("al", "baseline"):
            entry[role] = _stack(roles[role])
        out[key] = entry
    click.echo(f"[ref] {len(out)} cell(s) from {n_rows} manifest row(s)")
    return out


def _collect_headtest(pattern: str, dataset: str) -> dict:
    """{(model, strategy): {...}} for the headtest arms found on disk.

    Several run dirs can share an arm when a job was resubmitted; the one with
    the most cached iterations wins, which is the one that got furthest.
    """
    best: dict[tuple[str, str], tuple[int, Path, int]] = {}
    for d in sorted(globmod.glob(pattern)):
        p = Path(d)
        m = re.match(r"headtest_([a-z]+)_([a-z]+)_seed(\d+)_", p.name)
        if not m:
            click.echo(f"[warn] unparsed headtest dir name: {p.name}", err=True)
            continue
        model = _HEADTEST_MODEL.get(m.group(1))
        arm = _HEADTEST_ARM.get(m.group(2))
        if model is None or arm is None:
            click.echo(f"[warn] unknown model/arm in {p.name}", err=True)
            continue
        n = len(_series(p, "al", dataset))
        key = (model, arm)
        if n and (key not in best or n > best[key][0]):
            best[key] = (n, p, int(m.group(3)))
    out = {}
    for key, (n, p, seed) in sorted(best.items()):
        out[key] = {
            "al": _stack([_series(p, "al", dataset)]),
            "baseline": _stack([_series(p, "baseline", dataset)]),
            "seeds": 1,
            "run_dir": str(p),
            "seed": seed,
        }
        click.echo(f"[arm] {key[0]:16s} {key[1]:16s} seed={seed} "
                   f"iters={n} {p.name}")
    if not out:
        click.echo(f"[arm] no headtest run with cached accuracy under {pattern}")
    return out


def _dump_csv(path: Path, cells: dict, dataset: str) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "strategy", "role", "iteration",
                    "mean", "sem", "n_seeds"])
        for (model, strat), entry in sorted(cells.items()):
            for role in ("al", "baseline"):
                iters, M = entry[role]
                if not len(iters):
                    continue
                mean = np.nanmean(M, axis=0)
                n = (~np.isnan(M)).sum(axis=0)
                with np.errstate(invalid="ignore"):
                    sem = np.nanstd(M, axis=0, ddof=0) / np.sqrt(np.maximum(n, 1))
                for c, it in enumerate(iters):
                    if not np.isfinite(mean[c]):
                        continue
                    w.writerow([dataset, model, strat, role, int(it),
                                f"{mean[c]:.6f}", f"{sem[c]:.6f}", int(n[c])])
    click.echo(f"[write] {path}")


@click.command()
@click.option("--manifest",
              default="/ptmp/jwuerzin/analysis/joint/manifest_expr.csv",
              show_default=True,
              help="Benchmark manifest supplying the regression reference "
                   "cells. Pass '' to plot the headtest arms alone.")
@click.option("--headtest-glob", default="/ptmp/jwuerzin/output/headtest_*",
              show_default=True,
              help="Glob for the head/strategy test run directories.")
@click.option("--dataset", "datasets", multiple=True, default=("static_random",),
              show_default=True,
              help="Eval set(s) to render. 'static_random' is the only one "
                   "comparable across arms; 'train'/'val' are per-arm sets. "
                   "'mcmc' is refused for ExpR (wrong sampler).")
@click.option("--reference-seeds", default="all", show_default=True,
              help="'all', or a comma-separated seed list, for the benchmark "
                   "cells. The headtest arms are single-seed (seed 1), so "
                   "'1' makes the comparison exactly like-for-like: same pool, "
                   "same shuffle seed, same 100k held-out models.")
@click.option("--exclude", default="", show_default=True,
              help="Comma-separated model:arm cells to drop, e.g. "
                   "'deep_gp:bald,deep_gp:cls_entropy'. Use for cells whose "
                   "cached accuracy is known to be wrong: a single bad cell "
                   "rescales every panel, so leaving it in hides the rest.")
@click.option("--exclude-note", default="", show_default=True,
              help="Text drawn in a panel whose cells were excluded, so the "
                   "omission is visible in the figure rather than only here.")
@click.option("--output-dir", required=True,
              help="Directory for the PNG(s) and the numeric dump.")
@click.option("--uncertainty", default="sem",
              type=click.Choice(["sem", "std"]), show_default=True,
              help="Band shown across seeds where a cell has more than one.")
@click.option("--baselines/--no-baselines", default=True, show_default=True,
              help="Draw each cell's own random-sample baseline. One grey "
                   "dashed curve per panel (they coincide by construction: "
                   "same size, random draws), so only the reference cell's is "
                   "drawn unless a panel has no reference.")
@click.option("--max-iteration", default=0, type=int, show_default=True,
              help="Clip every curve at this iteration (0 = no clip). The "
                   "headtest arms are 40-iteration jobs, so 40 puts them and "
                   "the benchmark cells on one horizon instead of letting the "
                   "resumed reference seeds run on alone.")
@click.option("--target-label", default="r_exp", show_default=True,
              help="Observable name for the y-axis label.")
@click.option("--threshold-label", default="1", show_default=True,
              help="Decision threshold for the y-axis label.")
def main(manifest, headtest_glob, datasets, reference_seeds, exclude,
         exclude_note, output_dir,
         uncertainty, baselines, max_iteration, target_label, threshold_label):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "mcmc" in datasets:
        raise click.BadParameter(
            "the only MCMC chain on disk is the relic-density posterior, so an "
            "MCMC accuracy panel for ExpR scores the exclusion verdict on the "
            "wrong sampler's points; see the module docstring.")

    seeds = None
    if reference_seeds.strip() and reference_seeds.strip() != "all":
        seeds = {int(s) for s in reference_seeds.split(",") if s.strip()}

    for dataset in datasets:
        ref = _collect_reference(Path(manifest), dataset, seeds) if manifest else {}
        arms = _collect_headtest(headtest_glob, dataset)
        dropped: set[tuple[str, str]] = set()
        for spec in (x.strip() for x in exclude.split(",") if x.strip()):
            if ":" not in spec:
                raise click.UsageError(f"--exclude entry {spec!r} must be model:arm")
            mdl, arm = spec.split(":", 1)
            for d in (arms, ref):
                if (mdl, arm) in d:
                    del d[(mdl, arm)]
                    dropped.add((mdl, arm))
        if dropped:
            click.echo("[exclude] dropped " +
                       ", ".join(f"{m}/{a}" for m, a in sorted(dropped)))
        cells = dict(ref)
        cells.update(arms)          # an arm never collides with a benchmark cell
        if max_iteration:
            for entry in cells.values():
                for role in ("al", "baseline"):
                    iters, M = entry[role]
                    if len(iters):
                        keep = iters <= max_iteration
                        entry[role] = (iters[keep], M[:, keep])
        if not cells:
            click.echo(f"[skip] {dataset}: nothing with a cached accuracy")
            continue

        models = [m for m in MODEL_ORDER if any(k[0] == m for k in cells)]
        models += sorted({k[0] for k in cells} - set(models))
        ncol = min(3, len(models))
        nrow = int(np.ceil(len(models) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.3 * nrow),
                                 squeeze=False, sharex=True, sharey=True)
        flat = [a for r in axes for a in r]

        arms_seen: list[str] = []
        drew_baseline = False
        for ax, model in zip(flat, models):
            keys = [k for k in cells if k[0] == model]
            # Reference arms first so the benchmark curve is under the new ones.
            keys.sort(key=lambda k: (k[1] not in ("entropy_batch", "top_k"), k[1]))
            ref_key = next((k for k in keys if k in ref), None)
            if exclude_note and any(m == model for m, _ in dropped):
                ax.annotate(exclude_note, xy=(0.03, 0.06),
                            xycoords="axes fraction", ha="left", va="bottom",
                            fontsize=7.5, color="#B00020")
            for key in keys:
                strat = key[1]
                color, ls, marker, _lbl = ARM_STYLE.get(
                    strat, ("0.2", "-", "x", strat))
                iters, M = cells[key]["al"]
                _draw(ax, iters, M, color=color, linestyle=ls, marker=marker,
                      label=None, uncertainty=uncertainty)
                if strat not in arms_seen and len(iters):
                    arms_seen.append(strat)
                if baselines and key == (ref_key or keys[0]):
                    b_iters, b_M = cells[key]["baseline"]
                    _draw(ax, b_iters, b_M, label=None, fill_alpha=0.10,
                          uncertainty=uncertainty, **BASELINE_STYLE)
                    drew_baseline = drew_baseline or bool(len(b_iters))
            # The resumed seeds run past the benchmark horizon alone, so the
            # panel states both the seed count and how many survive to its last
            # iteration; without that "5 seeds" would over-sell the tail.
            n_ref = cells[ref_key]["seeds"] if ref_key else 0
            n_last = 0
            if ref_key:
                _it, _M = cells[ref_key]["al"]
                if len(_it):
                    n_last = int((~np.isnan(_M[:, -1])).sum())
            ref_tag = ""
            if n_ref:
                ref_tag = f"  (reference: {n_ref} seed{'s' if n_ref != 1 else ''}"
                ref_tag += (f", {n_last} at iter {int(cells[ref_key]['al'][0][-1])})"
                            if n_last and n_last != n_ref else ")")
            tag = MODEL_DISPLAY.get(model, model)
            ax.annotate(f"{tag}{ref_tag}",
                        xy=(0.03, 0.96), xycoords="axes fraction",
                        va="top", ha="left", fontsize=10)
            ax.grid(alpha=0.3)
        for ax in flat[len(models):]:
            ax.axis("off")
        for r in range(nrow):
            axes[r][0].set_ylabel(f"Accuracy ({target_label} "
                                  f"≷ {threshold_label})")
        for c in range(ncol):
            axes[nrow - 1][c].set_xlabel("AL iteration")

        handles = [Line2D([0], [0], color=ARM_STYLE[s][0],
                          linestyle=ARM_STYLE[s][1], marker=ARM_STYLE[s][2],
                          markersize=4.5, lw=1.8, label=ARM_STYLE[s][3])
                   for s in arms_seen if s in ARM_STYLE]
        handles += [Line2D([0], [0], color=ARM_STYLE.get(s, ("0.2",))[0],
                           lw=1.8, label=s)
                    for s in arms_seen if s not in ARM_STYLE]
        if drew_baseline:
            handles.append(Line2D([0], [0], label="random-sample baseline",
                                  **BASELINE_STYLE))
        fig.legend(handles=handles, loc="lower center",
                   ncol=min(len(handles), 3), frameon=False,
                   bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.06 if nrow > 1 else 0.10, 1, 1))
        png = out_dir / f"accuracy_arms_{dataset}.png"
        fig.savefig(png, dpi=180, bbox_inches="tight")
        plt.close(fig)
        click.echo(f"[write] {png}")
        _dump_csv(out_dir / f"accuracy_arms_{dataset}.csv", cells, dataset)

        # Headline numbers: last common iteration per panel, arms vs reference.
        click.echo(f"[final] {dataset}: accuracy at each cell's last iteration")
        for model in models:
            for key in sorted(k for k in cells if k[0] == model):
                iters, M = cells[key]["al"]
                if not len(iters):
                    continue
                mean = np.nanmean(M, axis=0)
                fin = np.where(np.isfinite(mean))[0]
                if not len(fin):
                    continue
                j = fin[-1]
                click.echo(f"  {model:16s} {key[1]:16s} iter {int(iters[j]):>4d}"
                           f"  acc {mean[j]:.4f}  "
                           f"(n_seeds={int((~np.isnan(M[:, j])).sum())})")


if __name__ == "__main__":
    main()
