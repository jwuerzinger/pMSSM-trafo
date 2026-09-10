"""In-band support covered against cumulative surrogate compute, per arm.

The arm-comparison counterpart of the right panel of the compute figure. The
budget axis of ``plot_prelim_support.py`` asks what a dataset costs in simulator
calls; this asks what it costs in GPU time, which is the axis on which a cheap
surrogate can beat an accurate one.

Two series are needed per seed and they have to be aligned on iterations, not
on points:

* coverage, from ``plot_support_efficiency._al_sequence``, which returns every
  simulated point in acquisition order, interleaving each iteration's train and
  validation halves. After iteration ``i`` the prefix is therefore exactly
  ``al_n_train[i] + al_n_val[i]`` points, which is what makes the alignment
  below exact rather than approximate.
* cumulative GPU-hours, from the same helpers ``plot_compute_vs_dataset`` uses:
  per-iteration training time plus selection time, following the resume chain so
  a continued run's earlier iterations are not silently dropped.

Seeds are interpolated onto a shared compute grid rather than having their
compute averaged at fixed iteration. Seeds of one arm can differ several-fold in
GPU time per iteration, and averaging x at fixed iteration makes the mean move
backwards as the shorter seeds run out, which draws a curve that doubles back on
itself.

Usage
-----
    python scripts/plot_prelim_compute_support.py \
        --manifest /ptmp/jwuerzin/analysis/joint/manifest_dmrd.csv \
        --arm-manifest /ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv \
        --arm-sweep-id c200_... --pool-dir /ptmp/jwuerzin/data/18387358 \
        --target DMRD --true-value 0.12 --output-dir <dir>
"""
from __future__ import annotations

import csv
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

from coverage_saturation import _cells                              # noqa: E402
from plot_support_efficiency import AXES, _al_sequence, _band_mask  # noqa: E402
from plot_prelim_support import _support_from_cache                 # noqa: E402
from plot_compute_vs_dataset import (                               # noqa: E402
    _iter_dir_chain, _selection_seconds, _tabpfn_fit_eval_seconds,
    _train_seconds,
)
from plot_prelim_paper_style import (                               # noqa: E402
    ARM_ORDER, ARM_STYLE, MODEL_ORDER, _PROD_MODEL, iter_arm_rows,
)
from plot_hit_rate_trajectories_multiseed import MODEL_DISPLAY      # noqa: E402
from pmssm.config import PARAM_ORDER                                # noqa: E402


def _compute_hours(d: Path) -> list[float]:
    """Cumulative GPU-hours after each iteration, or [] if unreadable."""
    import torch
    s = torch.load(d / "state.pt", weights_only=False, map_location="cpu")
    n_tr = list(s.get("al_n_train") or [])
    if not n_tr:
        return []
    sel = _selection_seconds(d / "active_learning.log")
    fit = _tabpfn_fit_eval_seconds(d / "active_learning.log")
    chain = _iter_dir_chain(d)
    out, cum = [], 0.0
    for i in range(len(n_tr)):
        it = chain.get(i + 1)
        if it is None:
            break
        tr = _train_seconds(it)
        if tr is None:                       # TabPFN takes no gradient step
            tr = fit.get(i + 1)
            if tr is None:
                break
        cum += tr + sel.get(i + 1, 0.0)
        out.append(cum / 3600.0)
    return out


def _seed_curve(d: Path, ax_idx, edges, tmap, n_target, true_value, tolerance,
                band_side):
    """(gpu_hours, coverage) after each iteration for one seed."""
    import torch
    got = _al_sequence(d, ax_idx, False)
    if got is None:
        return None
    X, Y = got
    cid = np.where(_band_mask(Y, true_value, tolerance, band_side),
                   _cells(X, edges), -1)
    s = torch.load(d / "state.pt", weights_only=False, map_location="cpu")
    nt = [int(v) for v in (s.get("al_n_train") or [])]
    nv = [int(v) for v in (s.get("al_n_val") or [])]
    hours = _compute_hours(d)
    n = min(len(nt), len(nv), len(hours))
    if n < 2:
        return None
    # tmap maps a cell id to its index in the support, or -1 when the cell is
    # not part of it. Membership is therefore tmap[cid] >= 0, and cid == -1
    # (out of band) must be excluded before indexing or it wraps to the last
    # entry.
    seen, cov, hrs = set(), [], []
    for i in range(n):
        stop = min(nt[i] + nv[i], len(cid))
        chunk = cid[:stop]
        good = chunk[(chunk >= 0)]
        good = good[tmap[good] >= 0]
        seen |= set(good.tolist())
        cov.append(len(seen) / n_target)
        hrs.append(hours[i])
    return np.asarray(hrs, dtype=float), np.asarray(cov, dtype=float)


def _mean_on_grid(curves, n_points=200):
    """Interpolate each seed onto a shared compute grid, then mean and SEM."""
    curves = [(x, y) for x, y in curves if len(x) >= 2]
    if not curves:
        return None
    lo = max(c[0][0] for c in curves)
    hi = max(c[0][-1] for c in curves)
    if not (hi > lo):
        return None
    grid = np.linspace(lo, hi, n_points)
    Y = np.full((len(curves), grid.size), np.nan)
    for k, (x, y) in enumerate(curves):
        o = np.argsort(x)
        xs, ys = x[o], y[o]
        inside = (grid >= xs[0]) & (grid <= xs[-1])
        Y[k, inside] = np.interp(grid[inside], xs, ys)
    with np.errstate(invalid="ignore"):
        m = np.nanmean(Y, axis=0)
        n = np.sum(np.isfinite(Y), axis=0)
        sd = np.nanstd(Y, axis=0, ddof=1) if Y.shape[0] > 1 else np.zeros_like(m)
        sem = np.where(n > 1, sd / np.sqrt(np.maximum(n, 1)), 0.0)
    return grid, m, m - sem, m + sem, n


@click.command()
@click.option("--manifest", default="", help="Joint manifest for the reference arms.")
@click.option("--arm-manifest", default="", help="Sweep manifest for the new arms.")
@click.option("--arm-sweep-id", default="", show_default=True)
@click.option("--pool-dir", default="/ptmp/jwuerzin/data/18387358", show_default=True)
@click.option("--pool-cache-dir", default="/ptmp/jwuerzin/analysis/pool_cache",
              show_default=True)
@click.option("--target", default="DMRD", show_default=True)
@click.option("--true-value", default=0.12, show_default=True)
@click.option("--tolerance", default=0.10, show_default=True)
@click.option("--n-bins", default=12, show_default=True)
@click.option("--min-cell", default=20, show_default=True)
@click.option("--band-side", default="in", type=click.Choice(["in", "out"]))
@click.option("--expect-cells", default=0, show_default=True,
              help="Assert the support size; 0 disables. A coarsened grid still "
                   "runs and silently saturates every arm.")
@click.option("--output-dir", required=True)
def main(manifest, arm_manifest, arm_sweep_id, pool_dir, pool_cache_dir, target,
         true_value, tolerance, n_bins, min_cell, band_side, expect_cells,
         output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ax_idx = [PARAM_ORDER.index(a) for a in AXES]

    # 7-tuple, not a dict: edges, tmap, n_target, oX, oY, n_rows, burn
    edges, tmap, n_target, _oX, _oY, _n_rows, _burn = _support_from_cache(
        pool_dir, target, pool_cache_dir, tolerance, n_bins, min_cell,
        true_value, band_side)
    click.echo(f"[support] {n_target} cells over {AXES}")
    if expect_cells and n_target != expect_cells:
        raise click.ClickException(
            f"support has {n_target} cells, expected {expect_cells}")

    cells: dict[tuple[str, str], list[Path]] = {}
    if manifest and Path(manifest).exists():
        for r in csv.DictReader(open(manifest)):
            mdl = _PROD_MODEL.get(r["model"])
            d = Path(r["expected_run_dir"])
            if mdl and (d / "state.pt").exists():
                cells.setdefault((mdl, r["strategy"]), []).append(d)
    for mdl, arm, d in iter_arm_rows(arm_manifest, arm_sweep_id):
        cells.setdefault((mdl, arm), []).append(d)

    curves = {}
    for key, dirs in sorted(cells.items()):
        seeds = []
        for d in dirs:
            try:
                c = _seed_curve(d, ax_idx, edges, tmap, n_target, true_value,
                                tolerance, band_side)
            except Exception as exc:                     # noqa: BLE001
                click.echo(f"  skip {d.name}: {type(exc).__name__}: {exc}")
                continue
            if c is not None:
                seeds.append(c)
        got = _mean_on_grid(seeds)
        if got is None:
            continue
        curves[key] = got
        click.echo(f"  {key[0]:<16} {key[1]:<17} {len(seeds)} seed(s), "
                   f"{got[0][-1]:7.1f} GPU h, coverage {got[1][-1]:.3f}")

    models = [m for m in MODEL_ORDER if any(k[0] == m for k in curves)]
    if not models:
        raise click.ClickException("no curves to draw")
    ncol = min(3, len(models)) or 1
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.5 * ncol, 5 * nrow),
                             squeeze=False, sharex=True, sharey=True)
    flat = [a for row in axes for a in row]
    seen_arms: list[str] = []
    for ax, model in zip(flat, models):
        for key in sorted((k for k in curves if k[0] == model),
                          key=lambda k: (ARM_ORDER.index(k[1])
                                         if k[1] in ARM_ORDER else 99)):
            x, m, lo, hi, nseed = curves[key]
            color, ls, marker, _l = ARM_STYLE.get(key[1],
                                                  ("0.25", "-", "x", key[1]))
            ax.plot(x, m, color=color, linestyle=ls, linewidth=1.6)
            if np.nanmax(nseed) > 1:
                ax.fill_between(x, lo, hi, color=color, alpha=0.14, linewidth=0)
            if key[1] not in seen_arms:
                seen_arms.append(key[1])
        ax.set_xscale("log")
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.text(0.02, 0.97, MODEL_DISPLAY.get(model, model), transform=ax.transAxes,
                va="top", ha="left", fontsize=11)
    for ax in flat[len(models):]:
        ax.axis("off")
    for r in range(nrow):
        axes[r][0].set_ylabel(f"In-band support covered\n({n_target} cells)")
    for c in range(ncol):
        axes[nrow - 1][c].set_xlabel("Cumulative surrogate compute (GPU h)")
    handles = [Line2D([], [], color=ARM_STYLE[a][0], linestyle=ARM_STYLE[a][1],
                      label=ARM_STYLE[a][3])
               for a in ARM_ORDER if a in seen_arms]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(4, len(handles)) or 1, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    p = out / f"prelim_compute_support_{band_side}band.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    click.echo(f"[write] {p}")


if __name__ == "__main__":
    main()
