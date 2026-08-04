"""Corner plots + input-vs-target plots for AL training sets (and MCMC ref).

Ports the array-based plotting helpers from Run3ModelGen's emcee diagnostics
(`source/Run3ModelGen/scripts/emcee_diagnostics.py`) to active-learning runs:
for each best-per-model cell of the sweep, the seeds' cumulative training sets
are pooled and rendered as

  * ``corner_<model>.png``            — 9 free-parameter corner plot with the
                                        constrained-target (Ω) marginal inset
  * ``<param>_vs_Omega_<model>.png``  — per-parameter scatter + marginal
                                        histograms against Ω

plus ``corner_mcmc.png`` for the MCMC reference set, drawn with the exact
same styling so the panels are directly comparable in the paper.

Usage:
    python scripts/plot_al_input_target_diagnostics.py \\
        --manifest /ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv \\
        --output-dir /ptmp/jwuerzin/analysis/all_runs/al_diag/ \\
        --require-neutralino-lsp
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyse_runs import (  # noqa: E402
    FREE_PARAM_INDICES,
    FREE_PARAM_NAMES,
    filter_run_neutralino_lsp,
    load_run,
)
from mcmc_diagnostics import (  # noqa: E402
    DEFAULT_AL_PICKS,
    _picks_from_manifest,
)

# Cap on points fed to corner.corner / the scatter panels (matches the
# Run3ModelGen diagnostics caller).
MAX_CORNER = 200_000


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers lifted from Run3ModelGen emcee_diagnostics.py (array-only,
# no emcee dependency). Keep in sync with the submodule where practical.
# ──────────────────────────────────────────────────────────────────────────────

def _grid_bins(values, target_nbins=50):
    """Histogram bin edges that MATCH a discrete recording grid, so a coarsely
    quantized observable (micrOMEGAs Ω at %.2e precision) is shown at its
    native resolution instead of aliasing into a comb. Falls back to
    `target_nbins` uniform bins for effectively continuous data."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return target_nbins
    u = np.unique(v)
    if u.size < 2 or u.size > min(20000, 0.5 * v.size):
        return target_nbins                       # effectively continuous
    q1, q3 = np.percentile(v, [25, 75])
    core = u[(u >= q1) & (u <= q3)]
    d = np.diff(core if core.size >= 2 else u)
    d = d[d > 0]
    if d.size == 0:
        return target_nbins
    g = float(np.median(d))                        # bulk grid step
    lo, hi = float(u[0]), float(u[-1])
    if g <= 0 or hi <= lo:
        return target_nbins
    k = max(1, int(round(((hi - lo) / target_nbins) / g)))
    w = k * g
    if (hi - lo) / w > 10000:
        return target_nbins
    start = (np.floor(lo / g) - 0.5) * g
    return np.arange(start, hi + w, w)


def _joint_plot(x, xlabel, y, ylabel, logp, scatter_idx, outpath, bins=50):
    """Scatter of x vs y + marginal histograms. The scatter may be
    sub-sampled via scatter_idx; the histograms use all points."""
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.10, right=0.97, bottom=0.09, top=0.94,
                          wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    ax_x = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_y = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_c = fig.add_subplot(gs[0, 1]); ax_c.axis("off")
    xs, ys = x[scatter_idx], y[scatter_idx]
    if logp is not None:
        sc = ax.scatter(xs, ys, c=logp[scatter_idx], s=7, alpha=0.5, cmap="viridis",
                        edgecolors="none", rasterized=True)
        cax = ax_c.inset_axes([0.30, 0.08, 0.20, 0.86])
        fig.colorbar(sc, cax=cax).set_label(r"$\log p$", fontsize=8)
    else:
        ax.scatter(xs, ys, s=7, alpha=0.4, color="steelblue",
                   edgecolors="none", rasterized=True)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
    ax_x.hist(x, bins=_grid_bins(x, bins), color="steelblue", alpha=0.85)
    ax_x.set_ylabel("count", fontsize=9); ax_x.tick_params(labelbottom=False); ax_x.grid(alpha=0.3)
    ax_y.hist(y, bins=_grid_bins(y, bins), orientation="horizontal", color="indianred", alpha=0.85)
    ax_y.set_xlabel("count", fontsize=9); ax_y.tick_params(labelleft=False); ax_y.grid(alpha=0.3)
    fig.suptitle(f"{xlabel} vs {ylabel}   (N={x.size})", fontsize=11)
    fig.savefig(outpath, dpi=130); plt.close(fig)


def plot_inputs_vs_target_data(chain, y, logp, params, target, outdir, tag,
                               bins=50, max_scatter=200000):
    """One scatter+histogram figure per input dim against the target."""
    valid = np.isfinite(y) & (y > 0)         # drop failed / sentinel (-1) values
    rng = np.random.default_rng(0)
    written = []
    for j, name in enumerate(params):
        x = chain[:, j].astype(float)
        m = valid & np.isfinite(x)
        xj, yj = x[m], y[m]
        lpj = logp[m] if logp is not None else None
        if xj.size < 2:
            continue
        idx = (rng.choice(xj.size, max_scatter, replace=False)
               if xj.size > max_scatter else slice(None))
        out = os.path.join(outdir, f"{name}_vs_{target}_{tag}.png")
        _joint_plot(xj, name, yj, target, lpj, idx, out, bins=bins)
        written.append(out)
    click.echo(f"[al-diag] wrote {len(written)} input-vs-{target} plot(s) ({tag}) -> {outdir}")
    return written


def plot_corner(flat_chain, params, outpath, target_vals=None, target_name=None):
    import corner
    from matplotlib.lines import Line2D
    ndim = flat_chain.shape[1]
    fig = corner.corner(
        flat_chain, labels=params,
        bins=40, smooth=1.0, smooth1d=1.0,          # smooth jagged 2D/1D histograms
        show_titles=True, title_fmt=".2g",
        title_quantiles=[0.16, 0.5, 0.84],          # titles: median +/- 1sigma
        quantiles=None,                             # suppress corner's 1-colour lines
        title_kwargs={"fontsize": 10}, label_kwargs={"fontsize": 13},
        max_n_ticks=4,
        plot_datapoints=False, plot_density=False,  # drop noisy scatter + raw density
        fill_contours=True, levels=(0.393, 0.865, 0.989),  # filled 1,2,3-sigma (2D)
        color="#2c6fbb",
        hist_kwargs={"color": "#2c6fbb", "linewidth": 1.4},
        contour_kwargs={"linewidths": 0.7},
    )
    q = np.percentile(flat_chain, [16, 50, 84], axis=0)          # (3, ndim)
    qstyle = [("16th percentile", "#e8a33d", "--"),
              ("median (50th)",   "#d62728", "-"),
              ("84th percentile", "#2ca02c", "--")]
    axes = np.array(fig.axes).reshape(ndim, ndim)
    for i in range(ndim):
        for k, (_, col, ls) in enumerate(qstyle):
            axes[i, i].axvline(q[k, i], color=col, ls=ls, lw=1.3)
    for ax in fig.get_axes():
        ax.tick_params(labelsize=9)
    handles = [Line2D([0], [0], color=col, ls=ls, lw=1.8, label=lab)
               for lab, col, ls in qstyle]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.985),
               fontsize=13, frameon=True, title="1-D marginal quantiles",
               title_fontsize=13)
    tv = np.asarray(target_vals, float) if target_vals is not None else np.array([])
    tv = tv[np.isfinite(tv)]
    if tv.size > 10:
        axin = fig.add_axes([0.62, 0.55, 0.31, 0.24])           # figure coords
        axin.hist(tv, bins=_grid_bins(tv, 50), color="#6b6b6b", alpha=0.85)
        for k, (_, col, ls) in enumerate(qstyle):
            axin.axvline(np.percentile(tv, [16, 50, 84][k]), color=col, ls=ls, lw=1.3)
        axin.set_title(f"{target_name} (constrained target)", fontsize=13)
        axin.set_xlabel(target_name, fontsize=12)
        axin.set_ylabel("count", fontsize=11)
        axin.tick_params(labelsize=9)
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────

def _pooled_cell_data(run_dirs, require_neutralino_lsp):
    """Pool the cumulative training sets of one cell's seeds.

    Returns (X_free (N, 9), Omega (N,), n_seeds_used)."""
    xs, ys, n_used = [], [], 0
    for d in run_dirs:
        try:
            run = load_run(d)
        except Exception as e:
            click.echo(f"[al-diag]   skipping {d}: {e}", err=True)
            continue
        if require_neutralino_lsp:
            run = filter_run_neutralino_lsp(run)
        xs.append(run.X[:, FREE_PARAM_INDICES])
        ys.append(run.Y)
        n_used += 1
    if not xs:
        return None, None, 0
    return np.concatenate(xs), np.concatenate(ys), n_used


@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
              show_default=True)
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/all_runs/al_diag/",
              show_default=True)
@click.option("--models", default=None,
              help="Comma-separated subset of models to plot (default: all six "
                   "best-per-model picks).")
@click.option("--mcmc-data-dir", default="/ptmp/jwuerzin/data/neutralino_v4",
              show_default=True,
              help="MCMC reference ntuples for corner_mcmc.png. Pass an empty "
                   "string to skip the MCMC corner.")
@click.option("--baseline-data-dir", default="",
              help="Random-scan pool ntuples for corner_random.png (e.g. "
                   "/ptmp/jwuerzin/data/18387358). Empty (default) skips it. "
                   "Uses the x_full/y_full .npy caches in --output-dir's "
                   "cache dir when present.")
@click.option("--cache-dir", default="/ptmp/jwuerzin/analysis/all_runs",
              show_default=True,
              help="Where the baseline pool's x_full/y_full .npy caches live "
                   "(shared with plot_hit_rate_trajectories_multiseed).")
@click.option("--mcmc-max-samples", default=500_000, type=int, show_default=True,
              help="Seeded uniform subsample cap on the MCMC reference.")
@click.option("--require-neutralino-lsp/--no-require-neutralino-lsp",
              default=False, show_default=True,
              help="Drop non-neutralino (sneutrino) rows from AL training sets "
                   "before plotting.")
@click.option("--skip-joint-plots", is_flag=True, default=False,
              help="Only render the corner plots (skip the 9 per-parameter "
                   "input-vs-Omega figures per cell).")
def main(manifest, output_dir, models, mcmc_data_dir, baseline_data_dir,
         cache_dir, mcmc_max_samples, require_neutralino_lsp, skip_joint_plots):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    picks = dict(DEFAULT_AL_PICKS)
    if models == "none":
        picks = {}
    elif models:
        keep = {m.strip() for m in models.split(",") if m.strip()}
        unknown = keep - picks.keys()
        if unknown:
            raise click.ClickException(f"unknown model(s): {sorted(unknown)}")
        picks = {m: sw for m, sw in picks.items() if m in keep}

    # ── AL cells ─────────────────────────────────────────────────────────────
    run_dirs_per_model = _picks_from_manifest(manifest, picks)
    for model, run_dirs in run_dirs_per_model.items():
        strat, warm = picks[model]
        if not run_dirs:
            click.echo(f"[al-diag] {model}: no completed runs for "
                       f"{strat}/{warm}; skipping", err=True)
            continue
        X_free, omega, n_seeds = _pooled_cell_data(run_dirs, require_neutralino_lsp)
        if X_free is None:
            continue
        click.echo(f"[al-diag] {model} ({strat}/{warm}): {n_seeds} seeds, "
                   f"{len(X_free)} pooled training points")
        corner_X, corner_y = X_free, omega
        if len(corner_X) > MAX_CORNER:
            idx = rng.choice(len(corner_X), MAX_CORNER, replace=False)
            corner_X, corner_y = corner_X[idx], corner_y[idx]
        plot_corner(corner_X, FREE_PARAM_NAMES,
                    str(out_dir / f"corner_{model}.png"),
                    target_vals=corner_y, target_name="Omega")
        click.echo(f"[al-diag] wrote corner_{model}.png")
        if not skip_joint_plots:
            plot_inputs_vs_target_data(X_free, omega, None, FREE_PARAM_NAMES,
                                       "Omega", str(out_dir), model)

    # ── Random-scan baseline corner ─────────────────────────────────────────
    if baseline_data_dir:
        import plot_hit_rate_trajectories_multiseed as phr  # noqa: PLC0415
        click.echo(f"[al-diag] loading random pool from {baseline_data_dir} ...")
        Xb, Yb = phr._load_xy_full(baseline_data_dir, "DMRD", Path(cache_dir))
        Xb_free = np.asarray(Xb)[:, FREE_PARAM_INDICES]
        Yb = np.asarray(Yb).ravel()
        if len(Xb_free) > MAX_CORNER:
            idx = rng.choice(len(Xb_free), MAX_CORNER, replace=False)
            Xb_free, Yb = Xb_free[idx], Yb[idx]
        plot_corner(Xb_free, FREE_PARAM_NAMES,
                    str(out_dir / "corner_random.png"),
                    target_vals=Yb, target_name="Omega")
        click.echo("[al-diag] wrote corner_random.png")

    # ── MCMC reference corner ────────────────────────────────────────────────
    if mcmc_data_dir:
        from pmssm.data import load_mcmc_data  # noqa: PLC0415
        click.echo(f"[al-diag] loading MCMC reference from {mcmc_data_dir} ...")
        Xm, Ym = load_mcmc_data(data_dir=mcmc_data_dir,
                                require_neutralino_lsp=require_neutralino_lsp,
                                max_samples=mcmc_max_samples or None)
        Xm = Xm.numpy() if hasattr(Xm, "numpy") else np.asarray(Xm)
        Ym = (Ym.numpy() if hasattr(Ym, "numpy") else np.asarray(Ym)).ravel()
        Xm_free = Xm[:, FREE_PARAM_INDICES]
        if len(Xm_free) > MAX_CORNER:
            idx = rng.choice(len(Xm_free), MAX_CORNER, replace=False)
            Xm_free, Ym = Xm_free[idx], Ym[idx]
        plot_corner(Xm_free, FREE_PARAM_NAMES,
                    str(out_dir / "corner_mcmc.png"),
                    target_vals=Ym, target_name="Omega")
        click.echo("[al-diag] wrote corner_mcmc.png")


if __name__ == "__main__":
    main()
