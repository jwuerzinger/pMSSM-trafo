"""In-band support covered by each head/strategy arm, one panel per model.

The question this answers: **does a high-yield arm actually cover the band, or
does it pile points into a corner of it?** Hits/desired counts in-band models
without asking where they are, so an arm that finds the same small region over
and over scores as well as one that maps the whole band. That is exactly the
worry BALD raises, since its acquisition is boundary-anchored and can collapse.

Nothing about the support is re-derived. The cells come from
``plot_support_efficiency._support`` with ``--support-source pool``, i.e. the
paper's definition: equal-occupancy quantile cells in the informative subspace,
built from the random-scan dataset's in-band points, keeping cells with at least
``--min-cell`` of them. An arm is credited for its own in-band points landing in
those cells, in acquisition order, via that module's ``_al_sequence`` and
``_band_mask`` and ``coverage_saturation._cells`` / ``coverage_of``. So a covered
fraction here is the same quantity as in the published support figure and can be
read beside it.

x axis
------
Budget as a **fraction of the reference dataset**: the number of simulated points
spent (train AND validation, since a validation point costs a simulator call
too) divided by the size of the dataset that defines the support. x = 1 is "you
have spent what the whole random scan cost", so an arm reaching a given height
to the left of another gets there more cheaply.

Style follows ``plot_prelim_paper_style``: colour encodes the arm, the panel
names the model, and the random scan's own prefix curve is the dotted reference.

Usage
-----
    python scripts/plot_prelim_support.py \
        --headtest-glob '/ptmp/jwuerzin/output/headtest_*_20260821_*' \
        --exclude-runs '_111422' \
        --output-dir /ptmp/jwuerzin/analysis/joint/prelim_20260821
"""
from __future__ import annotations

import csv
import glob as globmod
import re
import sys
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from coverage_saturation import _cells, coverage_of                 # noqa: E402
from plot_support_efficiency import (                               # noqa: E402
    AXES, _al_curve, _al_sequence, _band_mask, _ref_curve, _support,
)
from pmssm.config import PARAM_ORDER                                # noqa: E402
from plot_prelim_paper_style import (                               # noqa: E402
    ARM_ORDER, ARM_STYLE, BASELINE_STYLE, MODEL_ORDER,
    _HEADTEST_ARM, _HEADTEST_MODEL, _PROD_MODEL, iter_arm_rows,
)
from plot_hit_rate_trajectories_multiseed import MODEL_DISPLAY      # noqa: E402


def _support_from_cache(pool_dir, target, cache_dir, tol, n_bins, min_cell,
                        true_val, band_side):
    """The pool support, built from the parsed-pool .npy cache.

    Mirrors the ``source == "pool"`` branch of
    ``plot_support_efficiency._support`` line for line -- in-band mask, per-axis
    quantile edges with infinite end bins, ``min_cell`` occupancy filter, tmap --
    so the cells are the SAME cells and a covered fraction is comparable with the
    published figure. Only the read differs.

    That read is the whole point: ``_support`` calls ``load_pmssm_data``, which
    parses all 1499 ROOT files of the pool and takes about 19 minutes, while
    ``_load_xy_full`` mmaps a 133 MB .npy cache that already exists for this pool
    and target. 19 minutes does not fit apudev's 15-minute wall clock; seconds
    do. The cell count is asserted against the published value so a silent
    divergence in the support cannot pass.
    """
    from plot_hit_rate_trajectories_multiseed import _load_xy_full
    ax = list(AXES)
    X_full, Y_full = _load_xy_full(pool_dir, target, Path(cache_dir))
    idx = [PARAM_ORDER.index(a) for a in ax]
    X = np.asarray(X_full[:, idx], dtype=np.float32)
    Y = np.asarray(Y_full, dtype=np.float64).ravel()
    inb = _band_mask(Y, true_val, tol, band_side)
    X_def = X[inb]
    nb = n_bins
    edges = [np.quantile(X_def[:, j], np.linspace(0, 1, nb + 1))
             for j in range(len(ax))]
    for e in edges:
        e[0], e[-1] = -np.inf, np.inf
    counts = np.bincount(_cells(X_def, edges), minlength=nb ** len(ax))
    keep = np.where(counts >= min_cell)[0]
    tmap = -np.ones(nb ** len(ax), dtype=np.int64)
    tmap[keep] = np.arange(len(keep))
    click.echo(f"[support] pool (cached): {len(keep)} cells of {nb ** len(ax)} "
               f"from ALL {int(inb.sum()):,} in-band points of {len(Y):,} rows")
    return edges, tmap, len(keep), X, Y, len(Y), 0


@click.command()
@click.option("--headtest-glob", default="/ptmp/jwuerzin/output/headtest_*",
              show_default=True)
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/joint/manifest_expr.csv",
              show_default=True)
@click.option("--exclude-runs", default="", show_default=True,
              help="Substrings of run-dir names to drop (superseded runs).")
@click.option("--pool-dir", default="/ptmp/jwuerzin/data/260804", show_default=True)
@click.option("--target", default="ExpR", show_default=True)
@click.option("--true-value", default=1.0, show_default=True)
@click.option("--tolerance", default=0.1, show_default=True)
@click.option("--n-bins", default=12, show_default=True,
              help="Quantile bins per informative axis. 12 is the paper's "
                   "value and must not be changed casually: with AXES = "
                   "(M_1, M_2, mu) it gives 12^3 = 1728 candidate cells, of "
                   "which 1067 survive --min-cell on the ExpR pool, matching "
                   "the published support figure. A smaller grid silently "
                   "produces a coarser support (n_bins=3 gives 27 cells) on "
                   "which every arm saturates and nothing is discriminated.")
@click.option("--min-cell", default=20, show_default=True,
              help="In-band points a cell needs to join the support.")
@click.option("--n-points", default=40, show_default=True)
@click.option("--min-budget", default=2000, show_default=True,
              help="Budgets below the seed set are not an AL result.")
@click.option("--band-side", default="in", type=click.Choice(["in", "out"]),
              show_default=True)
@click.option("--pool-cache-dir", default="/ptmp/jwuerzin/analysis/pool_cache",
              show_default=True)
@click.option("--use-cache/--parse-root", default=True, show_default=True,
              help="Build the support from the parsed-pool .npy cache instead "
                   "of re-parsing 1499 ROOT files. Same cells either way; "
                   "--parse-root exists to re-verify that.")
@click.option("--expect-cells", default=0, show_default=True,
              help="Fail unless the support has this many cells. 1067 is the "
                   "published ExpR pool support (of 1728 = 12^3); setting it "
                   "turns a silently coarsened grid into an error.")
@click.option("--arm-manifest", default="", show_default=True,
              help="Full sweep manifest to read the new arms from, so each arm "
                   "is a seed mean with a band rather than one line per run.")
@click.option("--arm-sweep-id", default="", show_default=True,
              help="Restrict --arm-manifest to one sweep id.")
@click.option("--output-dir", required=True)
def main(headtest_glob, manifest, exclude_runs, pool_dir, target, true_value,
         tolerance, n_bins, min_cell, n_points, min_budget, band_side,
         pool_cache_dir, use_cache, expect_cells, arm_manifest,
         arm_sweep_id, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    click.echo(f"[support] building pool-sourced cells from {pool_dir}")
    ax_idx = [PARAM_ORDER.index(a) for a in AXES]
    if use_cache:
        edges, tmap, n_target, oX, oY, n_rows, _burn = _support_from_cache(
            pool_dir, target, pool_cache_dir, tolerance, n_bins, min_cell,
            true_value, band_side)
    else:
        edges, tmap, n_target, oX, oY, n_rows, _burn = _support(
            "pool", target, pool_dir, "", tolerance, n_bins, min_cell,
            False, true_value, band_side=band_side)
    if n_target == 0:
        raise click.ClickException("support is empty at these settings")
    if expect_cells and n_target != expect_cells:
        raise click.ClickException(
            f"support has {n_target} cells, expected {expect_cells}. The grid "
            f"or the band changed: n_bins={n_bins}, min_cell={min_cell}, "
            f"tolerance={tolerance}, band_side={band_side}.")
    n_total = n_rows
    S = dict(edges=edges, tmap=tmap, n_target=n_target, n_total=n_total)
    click.echo(f"[support] {n_target} cells over axes {AXES}, "
               f"{n_total:,} reference rows")

    drop = [x.strip() for x in exclude_runs.split(",") if x.strip()]
    cells: dict[tuple[str, str], list[Path]] = {}
    for d in sorted(globmod.glob(headtest_glob)):
        d = Path(d)
        m = re.match(r"headtest_([a-z]+)_([a-z]+)_seed", d.name)
        if not m or not (d / "state.pt").exists():
            continue
        if any(p in d.name for p in drop):
            click.echo(f"  [exclude] {d.name}")
            continue
        key = (_HEADTEST_MODEL.get(m.group(1), m.group(1)),
               _HEADTEST_ARM.get(m.group(2), m.group(2)))
        cells.setdefault(key, []).append(d)
    if manifest and Path(manifest).exists():
        for r in csv.DictReader(open(manifest)):
            mdl = _PROD_MODEL.get(r["model"])
            d = Path(r["expected_run_dir"])
            if mdl and (d / "state.pt").exists():
                cells.setdefault((mdl, r["strategy"]), []).append(d)
    for mdl, arm, d in iter_arm_rows(arm_manifest, arm_sweep_id):
        if any(p in d.name for p in drop):
            continue
        cells.setdefault((mdl, arm), []).append(d)

    curves: dict[tuple[str, str], tuple] = {}
    for key, dirs in sorted(cells.items()):
        seqs = []
        for d in dirs:
            try:
                got = _al_sequence(d, ax_idx, False)
            except Exception as exc:                     # noqa: BLE001
                click.echo(f"  skip {d.name}: {type(exc).__name__}: {exc}")
                continue
            if got is not None:
                seqs.append(got)
        if not seqs:
            continue
        seq_cells = [np.where(_band_mask(Y, true_value, tolerance, band_side),
                              _cells(X, S["edges"]), -1)
                     for X, Y in seqs]
        got = _al_curve(seq_cells, S["tmap"], S["n_target"], S["n_total"],
                        n_points, min_budget, min_seeds=1)
        if got is None:
            continue
        curves[key] = got
        click.echo(f"  {key[0]:<16} {key[1]:<17} {len(seqs)} replica(s), "
                   f"spent {[len(y) for _x, y in seqs]}, "
                   f"final coverage {got[1][-1]:.3f}")

    models = [m for m in MODEL_ORDER if any(k[0] == m for k in curves)]
    ncol = min(3, len(models)) or 1
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.5 * ncol, 5 * nrow),
                             squeeze=False, sharex=True, sharey=True)
    flat = [a for row in axes for a in row]
    arms_seen: list[str] = []
    for ax, model in zip(flat, models):
        # The reference dataset's own prefixes: it reaches 1.0 at x = 1 by
        # construction, which is the support's definition, not a result.
        rcells = np.where(_band_mask(oY, true_value, tolerance, band_side),
                          _cells(oX, S["edges"]), -1)
        rf, rc = _ref_curve(rcells, S["tmap"], S["n_target"], S["n_total"],
                            n_points, min_budget)
        ax.plot(rf, rc, label=None, **BASELINE_STYLE)
        keys = sorted((k for k in curves if k[0] == model),
                      key=lambda k: (ARM_ORDER.index(k[1])
                                     if k[1] in ARM_ORDER else 99))
        for key in keys:
            frac, mean, lo, hi = curves[key]
            color, ls, marker, _lbl = ARM_STYLE.get(
                key[1], ("0.25", "-", "x", key[1]))
            ax.plot(frac, mean, color=color, linestyle=ls, marker=marker,
                    markersize=3, linewidth=1.6,
                    markevery=max(1, len(frac) // 12))
            if len(curves[key][0]) and not np.allclose(lo, hi):
                ax.fill_between(frac, lo, hi, color=color, alpha=0.14, lw=0)
            if key[1] not in arms_seen:
                arms_seen.append(key[1])
        ax.set_xscale("log")
        ax.annotate(MODEL_DISPLAY.get(model, model), xy=(0.03, 0.96),
                    xycoords="axes fraction", va="top", ha="left", fontsize=10)
        ax.grid(alpha=0.3)
    for ax in flat[len(models):]:
        ax.axis("off")
    for r in range(nrow):
        axes[r][0].set_ylabel(f"In-band support covered\n({S['n_target']} cells)")
    for c in range(ncol):
        axes[nrow - 1][c].set_xlabel("Budget / reference dataset size")

    handles = [Line2D([0], [0], color=ARM_STYLE[a][0], linestyle=ARM_STYLE[a][1],
                      marker=ARM_STYLE[a][2], markersize=4.5, lw=1.8,
                      label=ARM_STYLE[a][3])
               for a in ARM_ORDER if a in arms_seen]
    handles.append(Line2D([0], [0], label="random scan (defines the support)",
                          **BASELINE_STYLE))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 3),
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08 if nrow > 1 else 0.12, 1, 1))
    png = out / f"prelim_support_{band_side}band.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"[write] {png}")


if __name__ == "__main__":
    main()
