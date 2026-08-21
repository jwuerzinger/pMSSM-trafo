"""Summarise the Stage A head-swap results: acquisition precision per selector,
and where each selector's picks actually sit.

Reads the JSONs written by ``scripts/head_swap_stage_a.py`` (one per submitted
cell) and aggregates over seeds x initialisations. Two panels, because the
question has two halves that the single precision number conflates:

  left   precision@500 for the +-10% band, per selector, grouped by iteration,
         against the random-pick reference (the pool's own band prevalence).
         This is the acquisition step's yield per valid point.
  right  where the picks are: median |t| (distance to the contour in the trained
         target's own units) against mean distance to the nearest labelled point
         in normalised input space. A selector can win the left panel either by
         sitting on the contour or by exploring, and only this panel says which.

Colours are Okabe-Ito, a published colour-vision-deficiency-safe set, assigned
per selector in a fixed order so that a cell missing one selector does not
repaint the others. No in-figure titles: the caption carries the description.

Usage
-----
    python scripts/plot_head_swap_stage_a.py \
        --inputs '/ptmp/jwuerzin/analysis/head_swap/stage_a_*.json' \
        --output-dir /ptmp/jwuerzin/analysis/head_swap
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito. Fixed assignment per selector: colour follows the entity, never its
# rank, so filtering cells cannot recolour the survivors.
COLOURS = {
    "reg_entropy_batch": "#0072B2",   # blue        production cell
    "reg_topk": "#56B4E9",            # sky blue
    "reg_var_only": "#009E73",        # bluish green
    "reg_var_nocut": "#F0E442",       # yellow
    "cls_bald": "#D55E00",            # vermillion
    "cls_entropy": "#E69F00",         # orange
    "prefilter_random": "#999999",    # grey        reference
    "pool_random": "#666666",         # dark grey   reference
}
ORDER = list(COLOURS)
# Production strategy names are given verbatim so an arm is never mistaken for
# a strategy it is not: prefilter_random is top_k_tol_only with the variance
# ranking replaced by a coin flip, and is not itself a production strategy.
LABEL = {
    "reg_entropy_batch": "regression, entropy_batch [production]",
    "reg_topk": "regression, top_k [production]",
    "reg_var_only": "regression, top_k_tol_only [production]",
    "reg_var_nocut": "regression, raw var, cut disabled [reference]",
    "cls_bald": "classification, BALD",
    "cls_entropy": "classification, entropy",
    "prefilter_random": "tolerance cut then random [reference]",
    "pool_random": "uniform random [reference]",
}


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var / len(vals))


def _collect(paths):
    """(precision, profile, accuracy, reference) keyed by (iteration, selector)."""
    prec, prof, acc, ref = {}, {}, {}, {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        tag = Path(p).stem.replace("stage_a_", "")
        ref[tag] = d.get("random_pick_precision", {}).get("tau10")
        for key, snap in d.get("snapshots", {}).items():
            it = key.split("|")[-1]
            for head, inits in snap.get("arms", {}).items():
                for e in inits:
                    for sel, pr in e.get("M1_precision_at_500", {}).items():
                        prec.setdefault((tag, it, sel), []).append(pr["tau10"])
                    for sel, pp in e.get("M6_pick_profile", {}).items():
                        prof.setdefault((tag, it, sel), []).append(
                            (pp["abs_t_median"], pp["nn_dist_mean"]))
                    a = e.get("M2_verdict_accuracy", {})
                    if a:
                        acc.setdefault((tag, it, head), []).append(
                            (a["all"], a.get("shell_lt_ln1.1")))
    return prec, prof, acc, ref


@click.command()
@click.option("--inputs", default="/ptmp/jwuerzin/analysis/head_swap/stage_a_*.json",
              help="Glob of Stage A result JSONs.")
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/head_swap")
@click.option("--name", default="head_swap_stage_a", help="Output basename.")
def main(inputs, output_dir, name):
    paths = sorted(glob.glob(inputs))
    if not paths:
        raise click.UsageError(f"no files match {inputs}")
    click.echo(f"[read] {len(paths)} result files")
    prec, prof, acc, ref = _collect(paths)
    if not prec:
        raise click.UsageError("no precision entries found")

    reference = next((v for v in ref.values() if v), None)
    cells = sorted({(t, i) for (t, i, _s) in prec})
    sels = [s for s in ORDER if any((t, i, s) in prec for t, i in cells)]

    have_profile = any(prof.get((t, i, s)) for t, i in cells for s in sels)
    height = 0.42 * len(sels) * len(cells) + 2.6
    if have_profile:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, height),
                                       gridspec_kw={"width_ratios": [1.35, 1.0]})
    else:
        fig, axL = plt.subplots(1, 1, figsize=(8.5, height))
        axR = None

    # ---- left: precision per selector, grouped by cell -----------------------
    y, ticks, labels, bars = 0.0, [], [], []
    for tag, it in cells:
        for sel in sels:
            vals = prec.get((tag, it, sel))
            if not vals:
                continue
            m, s = _agg(vals)
            axL.barh(y, m, xerr=s, height=0.68, color=COLOURS[sel],
                     edgecolor="white", linewidth=0.8,
                     error_kw=dict(ecolor="#444444", lw=1.0, capsize=2.5))
            gain = f"  {m / reference:.1f}x" if reference else ""
            bars.append((y, m, s, f"{m:.4f}{gain}"))
            ticks.append(y)
            labels.append(f"{LABEL[sel]}  [{tag} {it}, n={len(vals)}]")
            y += 1.0
        y += 0.6
    if reference:
        axL.axvline(reference, ls="--", lw=1.2, color="#444444", zorder=0)
        axL.text(reference, -1.15, f"random pick = {reference:.4f}", fontsize=8,
                 color="#444444", va="bottom", ha="center")
    axL.set_yticks(ticks)
    axL.set_yticklabels(labels, fontsize=8)
    axL.set_ylim(y - 0.4, -1.6)          # inverted, with room for the annotation
    axL.set_xlabel("precision@500 for the $\\pm10\\%$ band (per valid point)")
    top = max([m + (e or 0) for _yy, m, e, _t in bars]) if bars else 1.0
    axL.set_xlim(0, top * 1.35)
    pad = top * 0.03
    for yy, m, e, text in bars:
        axL.text(m + (e or 0) + pad, yy, text, va="center", ha="left",
                 fontsize=8, color="#333333")
    axL.grid(axis="x", lw=0.5, alpha=0.35)
    axL.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axL.spines[side].set_visible(False)

    # ---- right: where the picks sit ------------------------------------------
    plotted = False
    if axR is None:
        plotted = None
    for tag, it in cells if axR is not None else []:
        for sel in sels:
            vals = prof.get((tag, it, sel))
            if not vals:
                continue
            mx, sx = _agg([v[0] for v in vals])
            my, sy = _agg([v[1] for v in vals])
            if mx is None or my is None:
                continue
            axR.errorbar(mx, my, xerr=sx, yerr=sy, fmt="o", ms=9,
                         color=COLOURS[sel], mec="white", mew=1.2,
                         ecolor="#888888", lw=1.0, zorder=3)
            axR.annotate(f"{sel} ({it})", (mx, my), textcoords="offset points",
                         xytext=(9, 3), fontsize=8, color="#333333")
            plotted = True
    if plotted:
        axR.set_xlabel("median $|t|$ of the 500 picks  (distance to the contour, nats)")
        axR.set_ylabel("mean distance to the nearest labelled point\n"
                       "(normalised input space)")
        axR.grid(lw=0.5, alpha=0.35)
        axR.set_axisbelow(True)
        for side in ("top", "right"):
            axR.spines[side].set_visible(False)
    elif axR is not None:
        axR.set_axis_off()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    png = out / f"{name}.png"
    fig.savefig(png, dpi=180)
    click.echo(f"[write] {png}")

    # ---- the same numbers as text, so the figure is never the only record ----
    click.echo("\n| cell | selector | precision@500 | vs random | n |")
    click.echo("|---|---|---|---|---|")
    for tag, it in cells:
        for sel in sels:
            vals = prec.get((tag, it, sel))
            if not vals:
                continue
            m, s = _agg(vals)
            click.echo(f"| {tag} {it} | {LABEL[sel]} | {m:.4f} +- {s:.4f} | "
                       f"{m / reference:.2f}x | {len(vals)} |")
    click.echo("\n| cell | head | verdict accuracy | near-contour shell | n |")
    click.echo("|---|---|---|---|---|")
    for (tag, it, head), vals in sorted(acc.items()):
        ma, sa = _agg([v[0] for v in vals])
        mb, sb = _agg([v[1] for v in vals])
        click.echo(f"| {tag} {it} | {head} | {ma:.4f} +- {sa:.4f} | "
                   f"{mb:.4f} +- {sb:.4f} | {len(vals)} |")


if __name__ == "__main__":
    main()
