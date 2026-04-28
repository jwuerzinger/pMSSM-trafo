"""Summarise the final-iteration training-set distribution per (model, strategy, warm).

For each (model, strategy, warm) cell with at least `--min-seeds` seeds, this
plots a 2×N_pairs grid of 2-D histograms over the SCATTER_PAIRS parameter
projections used by `analyse_runs.plot_pairwise_scatter_per_run`:

  - Top row    : *mean* normalized density across seeds (= pooled histogram /
                 n_seeds, per bin). Shows where the cell *typically* places
                 its training points.
  - Bottom row : *std* of per-seed normalized densities. Highlights bins
                 where seed-to-seed agreement is weakest, i.e. AL stochasticity.

Optionally overlays an MCMC reference contour on the mean row (default on)
so each cell can be read against the underlying MCMC posterior shape.

One PNG per (model, strategy, warm) cell:
    pairwise_density_<model>_<strategy>_<warm>.png

Free parameters and their normalization match `analyse_runs.normalise_free_params`
([0, 1] using PARAM_RANGES), so x/y axes of every panel are in [0, 1].
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import matplotlib.patheffects as plt_path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analyse_runs import (  # noqa: E402
    FREE_PARAM_NAMES,
    SCATTER_PAIRS,
    load_mcmc_numpy,
    load_run,
    normalise_free_params,
)
from pmssm.config import PARAM_ORDER, TARGET_CONFIG  # noqa: E402


def _load_mcmc_no_file_cut(data_dir: str | Path):
    """Mirror of `pmssm.data.load_mcmc_data` minus the file-level rejection.

    Keeps only the per-sample physical filter ``(Y > 0) & (Y < 1) & (SP_m_h != -1)``.
    Used as a diagnostic to see how strongly the file-level filter
    (chains-must-straddle-Ω=0.12, no-values-below-0.04) shapes the MCMC
    reference distribution.
    """
    import glob
    import uproot

    files = sorted(glob.glob(f"{data_dir}/*.root"))
    target_branch = TARGET_CONFIG["DMRD"]["branch"]
    trees = [uproot.open(f)["susy"] for f in files]
    X_raw = np.column_stack([
        np.concatenate([t[b].array(library="np") for t in trees])
        for b in PARAM_ORDER
    ])
    Y_raw = np.concatenate([t[target_branch].array(library="np") for t in trees])
    sp_mh = np.concatenate([t["SP_m_h"].array(library="np") for t in trees])
    mask = (Y_raw > 0) & (Y_raw < 1.0) & (sp_mh != -1)
    return X_raw[mask].astype(np.float64), int(mask.sum()), int(len(mask)), len(files)


def _pair_indices(name_a: str, name_b: str) -> tuple[int, int]:
    return FREE_PARAM_NAMES.index(name_a), FREE_PARAM_NAMES.index(name_b)


def _per_seed_hist2d(
    X_free_norm: np.ndarray,
    idx_a: int,
    idx_b: int,
    n_bins: int,
) -> np.ndarray:
    """Density-normalized 2D histogram in [0, 1]² for one seed and one pair."""
    H, _, _ = np.histogram2d(
        X_free_norm[:, idx_a],
        X_free_norm[:, idx_b],
        bins=n_bins,
        range=[[0.0, 1.0], [0.0, 1.0]],
        density=True,
    )
    return H


def _stack_hists(
    runs_X: list[np.ndarray],
    pair_idx: tuple[int, int],
    n_bins: int,
) -> np.ndarray:
    """Return (n_seeds, n_bins, n_bins) per-seed densities."""
    a, b = pair_idx
    return np.stack(
        [_per_seed_hist2d(X, a, b, n_bins) for X in runs_X],
        axis=0,
    )


def _mcmc_density(
    mcmc_X_free_norm: np.ndarray,
    pair_idx: tuple[int, int],
    n_bins: int,
) -> np.ndarray:
    a, b = pair_idx
    H, _, _ = np.histogram2d(
        mcmc_X_free_norm[:, a],
        mcmc_X_free_norm[:, b],
        bins=n_bins,
        range=[[0.0, 1.0], [0.0, 1.0]],
        density=True,
    )
    return H


def _plot_one_cell(
    runs_X: list[np.ndarray],
    n_seeds: int,
    title: str,
    out_path: Path,
    n_bins: int,
    mcmc_X_free_norm: np.ndarray | None,
    overlay_mcmc: bool,
):
    n_pairs = len(SCATTER_PAIRS)
    show_mcmc_row = overlay_mcmc and mcmc_X_free_norm is not None
    n_rows = 3 if show_mcmc_row else 2
    panel_size = 2.6  # inches; aspect="equal" makes each panel square
    fig, axes = plt.subplots(
        n_rows, n_pairs,
        figsize=(panel_size * n_pairs + 1.6, panel_size * n_rows + 1.4),
        sharex=False, sharey=False,
        squeeze=False,
    )

    fig.suptitle(f"{title}  (n_seeds={n_seeds})", fontsize=14)

    # Row-index lookup so reordering is local to here.
    if show_mcmc_row:
        ROW_MCMC, ROW_MEAN, ROW_STD = 0, 1, 2
    else:
        ROW_MCMC, ROW_MEAN, ROW_STD = -1, 0, 1

    # Pre-compute the MCMC density per pair (own row).
    mcmc_dens_by_pair = {}
    if show_mcmc_row:
        for (a_name, b_name) in SCATTER_PAIRS:
            try:
                pidx = _pair_indices(a_name, b_name)
            except ValueError:
                continue
            mcmc_dens_by_pair[(a_name, b_name)] = _mcmc_density(
                mcmc_X_free_norm, pidx, n_bins
            )

    # Pre-compute (mean_H, std_H) per pair so we can find a row-shared colour scale.
    panel_data = []
    for (a_name, b_name) in SCATTER_PAIRS:
        try:
            pidx = _pair_indices(a_name, b_name)
        except ValueError:
            panel_data.append(None)
            continue
        H = _stack_hists(runs_X, pidx, n_bins)
        mean_H = H.mean(axis=0)
        std_H = H.std(axis=0, ddof=1) if H.shape[0] > 1 else np.zeros_like(mean_H)
        panel_data.append((mean_H, std_H, a_name, b_name))

    # Row-shared colour ranges. The MCMC and AL-mean rows share a LogNorm so
    # they can be compared by eye; the std row gets its own linear scale.
    valid = [d for d in panel_data if d is not None]
    all_mean = np.concatenate([d[0].ravel() for d in valid]) if valid else np.array([0.0])
    all_std = np.concatenate([d[1].ravel() for d in valid]) if valid else np.array([0.0])
    all_mcmc = (
        np.concatenate([m.ravel() for m in mcmc_dens_by_pair.values()])
        if mcmc_dens_by_pair else np.array([0.0])
    )

    pos_density = np.concatenate([all_mean[all_mean > 0], all_mcmc[all_mcmc > 0]])
    if pos_density.size:
        density_vmax = float(np.percentile(pos_density, 99))
        density_vmin = max(density_vmax / 1e3, float(pos_density.min()))
    else:
        density_vmax, density_vmin = 1.0, 1e-3
    std_vmax = float(np.percentile(all_std[all_std > 0], 99)) if (all_std > 0).any() else 1.0

    density_norm = LogNorm(vmin=density_vmin, vmax=density_vmax)
    std_norm = Normalize(vmin=0.0, vmax=std_vmax)

    extent = [0, 1, 0, 1]
    last_im_density = None
    last_im_std = None

    for col, data in enumerate(panel_data):
        a_name, b_name = SCATTER_PAIRS[col]

        ax_mean = axes[ROW_MEAN, col]
        ax_std = axes[ROW_STD, col]
        ax_mcmc = axes[ROW_MCMC, col] if show_mcmc_row else None

        if data is None:
            for ax in [a for a in (ax_mean, ax_std, ax_mcmc) if a is not None]:
                ax.text(0.5, 0.5, f"{a_name},{b_name}\nnot free",
                        ha="center", va="center",
                        transform=ax.transAxes,
                        fontsize=9, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
            continue

        mean_H, std_H, _, _ = data

        # ── MCMC row (top, when present): independent panel per pair, shared
        #    LogNorm with the mean row.
        if ax_mcmc is not None:
            mcmc_d = mcmc_dens_by_pair.get((a_name, b_name))
            if mcmc_d is not None:
                im = ax_mcmc.imshow(
                    mcmc_d.T, origin="lower", extent=extent, aspect="equal",
                    cmap="viridis", norm=density_norm,
                )
                last_im_density = im
            ax_mcmc.set_title(f"{a_name} vs {b_name}", fontsize=11)
            ax_mcmc.tick_params(axis="both", labelsize=9)
            ax_mcmc.set_xticks([])
            if col != 0:
                ax_mcmc.set_yticks([])

        # ── AL mean row.
        im_mean = ax_mean.imshow(
            mean_H.T, origin="lower", extent=extent, aspect="equal",
            cmap="viridis", norm=density_norm,
        )
        last_im_density = im_mean
        if not show_mcmc_row:
            ax_mean.set_title(f"{a_name} vs {b_name}", fontsize=11)
        ax_mean.tick_params(axis="both", labelsize=9)
        ax_mean.set_xticks([])
        if col != 0:
            ax_mean.set_yticks([])

        # ── AL std row.
        im_std = ax_std.imshow(
            std_H.T, origin="lower", extent=extent, aspect="equal",
            cmap="magma", norm=std_norm,
        )
        last_im_std = im_std
        ax_std.set_xlabel(a_name, fontsize=11)
        ax_std.tick_params(axis="both", labelsize=9)
        if col != 0:
            ax_std.set_yticks([])

    # Row labels (only on the leftmost panels).
    if valid:
        if show_mcmc_row:
            axes[ROW_MCMC, 0].set_ylabel("MCMC density\n(reference, log)",
                                         fontsize=11)
        axes[ROW_MEAN, 0].set_ylabel("AL mean density\nacross seeds (log)",
                                     fontsize=11)
        axes[ROW_STD, 0].set_ylabel("AL std density\nacross seeds (linear)",
                                    fontsize=11)

    # Reserve right margin and place each colorbar against its row.
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])
    if last_im_density is not None:
        ax_top = axes[0, -1]
        ax_mean_right = axes[ROW_MEAN, -1]
        bb_top = ax_top.get_position()
        bb_mean = ax_mean_right.get_position()
        cbar_ax_density = fig.add_axes([
            bb_top.x1 + 0.01,
            bb_mean.y0,
            0.012,
            (bb_top.y1 - bb_mean.y0),
        ])
        fig.colorbar(last_im_density, cax=cbar_ax_density,
                     label="density (log, shared)")
    if last_im_std is not None:
        ax_bot = axes[ROW_STD, -1]
        bb = ax_bot.get_position()
        cbar_ax_std = fig.add_axes([bb.x1 + 0.01, bb.y0, 0.012, bb.height])
        fig.colorbar(last_im_std, cax=cbar_ax_std, label="std (clipped at p99)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
              show_default=True)
@click.option("--mcmc-data-dir", default="/ptmp/jwuerzin/data/19250082",
              show_default=True,
              help="MCMC ROOT dir for the reference contour overlay. Set "
                   "--no-overlay-mcmc to skip and run without loading MCMC.")
@click.option("--sweep-id", default=None,
              help="Filter to one sweep_id (default: all).")
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/all_runs/",
              show_default=True)
@click.option("--min-seeds", default=2, type=int, show_default=True,
              help="Drop cells with fewer reporting seeds than this.")
@click.option("--include-status", default="completed,running,timeout",
              show_default=True,
              help="Comma-separated statuses to include from the manifest.")
@click.option("--n-bins", default=30, type=int, show_default=True,
              help="2D histogram resolution (n_bins × n_bins per panel).")
@click.option("--overlay-mcmc/--no-overlay-mcmc", default=True, show_default=True,
              help="Show the MCMC density as a top reference row.")
@click.option("--mcmc-skip-file-cut", is_flag=True, default=False,
              help="Diagnostic: load MCMC with the per-sample physical filter "
                   "only (skip the file-level reject of chains that don't "
                   "straddle Ω=0.12 or contain values below 0.04). Output "
                   "filenames get a `_rawmcmc` suffix so the standard "
                   "filtered plots are not overwritten.")
def main(manifest, mcmc_data_dir, sweep_id, output_dir, min_seeds,
         include_status, n_bins, overlay_mcmc, mcmc_skip_file_cut):
    df = pd.read_csv(manifest)
    if sweep_id:
        df = df[df["sweep_id"].astype(str) == str(sweep_id)]
    allowed = {s.strip() for s in include_status.split(",") if s.strip()}
    df = df[df["status"].isin(allowed)]
    if df.empty:
        raise click.ClickException("no rows matched manifest filter")

    mcmc_X_free_norm = None
    if overlay_mcmc:
        if mcmc_skip_file_cut:
            click.echo(f"[pairwise] loading MCMC (no file-level cut) from "
                       f"{mcmc_data_dir} ...")
            mcmc_X, n_kept, n_total, n_files = _load_mcmc_no_file_cut(mcmc_data_dir)
            mcmc_X_free_norm = normalise_free_params(mcmc_X)
            click.echo(f"[pairwise] MCMC raw: {n_kept}/{n_total} samples "
                       f"after per-sample filter, from all {n_files} files")
        else:
            click.echo(f"[pairwise] loading MCMC reference from {mcmc_data_dir} ...")
            mcmc_X, _ = load_mcmc_numpy(mcmc_data_dir)
            mcmc_X_free_norm = normalise_free_params(mcmc_X)
            click.echo(f"[pairwise] MCMC: {len(mcmc_X_free_norm)} samples")

    out_dir = Path(output_dir)
    written = []
    for (model, strat, warm), sub in df.groupby(["model", "strategy", "warm_start"]):
        runs_X: list[np.ndarray] = []
        for run_dir in sub["expected_run_dir"].dropna():
            try:
                run = load_run(run_dir)
            except Exception as exc:
                click.echo(f"[warn] skip {run_dir}: {exc}", err=True)
                continue
            X = run.X_free_norm
            if X is None or len(X) == 0:
                continue
            runs_X.append(np.asarray(X))

        if len(runs_X) < min_seeds:
            click.echo(f"[skip] {model}/{strat}/{warm}: only {len(runs_X)} seeds "
                       f"(< min-seeds={min_seeds})", err=True)
            continue

        suffix = "_rawmcmc" if mcmc_skip_file_cut else ""
        out_path = out_dir / f"pairwise_density_{model}_{strat}_{warm}{suffix}.png"
        title = f"{model} / {strat} / {warm}"
        _plot_one_cell(
            runs_X, len(runs_X), title, out_path,
            n_bins=n_bins,
            mcmc_X_free_norm=mcmc_X_free_norm,
            overlay_mcmc=overlay_mcmc,
        )
        written.append(out_path)
        click.echo(f"[plot] {out_path.name}  ({len(runs_X)} seeds)")

    if not written:
        raise click.ClickException("no cells passed min-seeds filter")

    click.echo(f"[plot] wrote {len(written)} file(s) to {out_dir}")


if __name__ == "__main__":
    main()
