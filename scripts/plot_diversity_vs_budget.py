"""Space-filling quality of a dataset against the budget spent.

The support-efficiency figure (`plot_support_efficiency.py`) asks how much of a
reference region a run reaches. These figures keep that figure's x axis, its
curves and its ratio panel, and replace the y axis with a geometric measure of
the point set itself. One pass over the data produces three of them:

  nn        mean distance from a point to its NEAREST NEIGHBOUR in the same set.
            The diversity of the set as such: a concentrated acquisition packs
            its points together and sits low, a flat scan spreads them and sits
            high. Note it falls as N^(-1/d) at fixed density, so a set with more
            points is expected lower; the comparison is against the random scan
            AT EQUAL BUDGET, which the lower panel does directly.

  voronoi   mean distance from a PROBE point to the nearest point of the set,
            i.e. the Monte Carlo average over the set's Voronoi cells of the
            distance to their generator. This is a coverage measure and reads
            the opposite way from nn: low means every probe has a sample nearby,
            so the region is covered; high means the set leaves large voids. The
            95th percentile, the near-worst-case fill distance, is in the JSON.

  vorunif   coefficient of variation of the Voronoi CELL VOLUMES, volumes
            estimated by how many probes fall in each cell. Zero would be a
            perfectly even tessellation. Cell volumes need many probes per cell
            to be meaningful, so this one is computed on a random subsample of
            the set capped at ``--vor-cap`` points; it therefore measures the
            SHAPE of the design at fixed size, with the point count divided out,
            which is the opposite convention from the other two.

Distance is Euclidean in the 9 free parameters, each mapped to [0, 1] by its
scan range from ``PARAM_RANGES``, so one unit is the full prior box along that
axis and no dimension dominates through its units (``IN_At`` spans 16 TeV,
``IN_tanb`` spans 59). Every mean is estimated by sampling (``--n-anchors``
points of the set for nn, ``--n-probes`` probes for the other two) and queried
exactly with a KD-tree, which keeps the million-row references feasible.

Two columns, because both "diverse" and "covered" can mean two things here:

  all points   every simulated point of the run, probes drawn UNIFORMLY from the
               prior box. This is the dataset as a training set, and coverage is
               coverage of the whole scan box.
  in-band      only the points inside the tolerance band, probes drawn from the
               posterior's in-band rows (a target with no posterior falls back to
               the scan). This is the useful half, and it is the column to read
               next to the support figure: it asks how far a typical in-band
               model is from the nearest in-band model the run acquired. Nothing
               is held out; a probe that is itself a member of the set being
               scored has its own row skipped for that one query, which is the
               leave-one-out correction described in ``_metrics_at``.

Curves are the same set as the support figure: one per AL cell (mean over seed
replicas, band = min/max), the static random scan in scan order, and the emcee
posterior. Budget is normalised by the size of the random scan, and both axes
count valid models simulated (an AL run's budget is train + validation).

Two accounting notes:

  * ``--pool-files`` reads only the first N ROOT files of the scan, which is a
    genuine prefix of an i.i.d. dataset and covers the plotted range at a
    fraction of the I/O. Pass ``--pool-total-rows`` so the x axis still divides
    by the size of the WHOLE scan (1,336,242 valid models for the relic-density
    pool, as measured by plot_support_efficiency).
  * every curve is charged in simulator calls and divided by the size of the
    random scan, so the emcee curve, whose retained rows are a thinned sample of
    the chains, is charged ``--mcmc-total-rows`` / rows retained calls per row.
    Its whole budget is 36x the random scan's, so most of that curve lies to the
    right of the plotted range, exactly as it does in the support figure's
    random-scan panel.
  * the emcee reference is a seeded uniform subsample of the post-burn-in
    ntuples, and a prefix of it is treated as a partial run. Unlike a coverage
    curve over cells, these metrics at fixed N are insensitive to the row ORDER,
    so the subsample is the cheap and slightly conservative choice: it removes
    the within-chain autocorrelation a genuine prefix would carry, which can
    only make the posterior look MORE spread out than it is.

Usage (relic density, the joint 40-iteration + extension manifest):

    P=/ptmp/jwuerzin/pixi-envs/pytorch-conda-forge-2863954108128992291/envs/rocm/bin/python
    $P scripts/plot_diversity_vs_budget.py \\
        --manifest /ptmp/jwuerzin/analysis/joint/manifest_dmrd.csv \\
        --output-dir /ptmp/jwuerzin/analysis/joint/dmrd_diversity \\
        --pool-files 60 --pool-total-rows 1336242 --min-seeds 1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mcmc_diagnostics import PARAM_ORDER, picks_with_tag  # noqa: E402
from plot_support_efficiency import _al_sequence, _discover  # noqa: E402

RNG_SEED = 20260818
COLUMNS = [("all", "all simulated points, probes uniform in the scan box"),
           ("inband", "in-band points only, probes are in-band models, left out per query")]
SOURCE_STYLE = {"pool": (":", "black"), "mcmc": ("-.", "0.45")}
SOURCE_CURVE = {"pool": "random scan", "mcmc": "MCMC"}
# (key, y label, whether lower means better-spread)
METRICS = {
    "nn": ("mean nearest-neighbour distance within the set\n"
           "(fraction of the scan range per axis, 9-D)", "diversity_nn"),
    "voronoi": ("mean probe distance to the nearest point of the set\n"
                "(Voronoi coverage radius, 9-D box units)", "coverage_voronoi"),
    "vorunif": (f"coefficient of variation of the Voronoi cell volumes\n"
                "(on a fixed-size subsample of the set)", "voronoi_uniformity"),
}
# A set smaller than this gives a nearest-neighbour mean dominated by the
# handful of points that happen to be there. The in-band column hits it first:
# 0.9% of a flat scan is in the relic band, so the shared 2000-point
# initialisation contributes ~18 points.
MIN_SET = 40


def _free_dims():
    """(indices into PARAM_ORDER, lo, hi) of the parameters the scan varies."""
    from pmssm.config import PARAM_RANGES
    idx, lo, hi = [], [], []
    for j, name in enumerate(PARAM_ORDER):
        a, b = PARAM_RANGES[name]
        if b > a:
            idx.append(j)
            lo.append(float(a))
            hi.append(float(b))
    return idx, np.array(lo), np.array(hi)


def _metrics_at(S, probes, probe_pos, n_anchors, vor_cap, rng):
    """All three metrics for one point set, or None if it is too small.

    One KD-tree serves the nearest-neighbour mean (k=2, the first neighbour
    being the anchor itself) and the probe coverage. The cell-volume spread
    needs its own tree on a capped subsample: with tens of thousands of
    generators and thousands of probes almost every cell would hold zero probes
    and the coefficient of variation would measure Poisson noise instead of the
    tessellation.

    ``probe_pos`` gives, per probe, its row position inside ``S``, or -1 when the
    probe is not a member of ``S``; None means no probe can be. Where a probe IS
    a member, its own row is skipped and the SECOND neighbour is taken. That is
    the leave-one-out correction, and it is what lets every dataset keep all of
    its rows: without it a probe drawn from the dataset being scored matches
    itself at distance zero, and the dataset's own curve is dragged towards zero
    as its prefix swallows more of the probes, which is fatal here because that
    curve is the ratio panel's denominator. Only the self-match is skipped, by
    row identity rather than by value, so a genuinely repeated point (emcee
    leaves duplicate rows wherever a proposal was rejected) still contributes its
    real distance of zero.
    """
    from scipy.spatial import cKDTree
    n = len(S)
    if n < MIN_SET:
        return None
    tree = cKDTree(S)
    q = S if n_anchors >= n else S[rng.choice(n, size=n_anchors, replace=False)]
    d_nn, _ = tree.query(q, k=2, workers=-1)
    if probe_pos is None:
        d_pr, _ = tree.query(probes, k=1, workers=-1)
    else:
        dd, ii = tree.query(probes, k=2, workers=-1)
        self_hit = (probe_pos >= 0) & (ii[:, 0] == probe_pos)
        d_pr = np.where(self_hit, dd[:, 1], dd[:, 0])
    # Below the cap the generator count would be n itself, so probes-per-cell
    # would be len(probes)/n and the Poisson contribution to the coefficient of
    # variation, roughly 1/sqrt(probes per cell), would slide along the budget
    # axis: every curve then rises over its first decade for a reason that has
    # nothing to do with the design. Report the spread only where the generator
    # count, and hence that noise floor, is the same for every point.
    if n < vor_cap:
        vorunif, n_gen = float("nan"), 0
    else:
        sub = S if n == vor_cap else S[rng.choice(n, size=vor_cap, replace=False)]
        tsub = cKDTree(sub) if sub is not S else tree
        _d, idx = tsub.query(probes, k=1, workers=-1)
        vol = np.bincount(idx, minlength=len(sub)).astype(np.float64) / len(probes)
        vorunif, n_gen = float(np.std(vol) / np.mean(vol)), len(sub)
    return {"nn": float(np.mean(d_nn[:, 1])),
            "voronoi": float(np.mean(d_pr)),
            "voronoi_p95": float(np.percentile(d_pr, 95)),
            "vorunif": vorunif,
            "n_set": int(n), "n_vor_generators": int(n_gen)}


def _grid(n_max, n_points, lo):
    """Geometric budgets from ``lo`` to ``n_max``, as the support figure does."""
    if n_max <= lo:
        return np.array([n_max], dtype=int)
    g = np.unique(np.round(np.geomspace(lo, n_max, n_points)).astype(int))
    return g[g >= MIN_SET]


def _curve(X, Y, band, budgets, probes, probe_rows, n_anchors, vor_cap, rng):
    """(budgets, {metric: values}) over prefixes, for one column's selection.

    ``probe_rows`` are the probes' row indices in ``X`` when the probes were
    drawn from this very dataset, else None. Each prefix keeps ``X``'s order, so
    the position of a probe inside the scored set is found by searching the
    prefix's own index list, and -1 marks a probe the prefix has not reached yet.
    """
    xs, vals = [], []
    orig = None if probe_rows is None else np.arange(len(X))
    for n in budgets:
        S = X[:n]
        sel = None if orig is None else orig[:n]
        if band is not None:
            S = S[band[:n]]
            if sel is not None:
                sel = sel[band[:n]]
        if sel is None:
            pos = None
        else:
            # sel is ascending, so searchsorted gives the position directly.
            at = np.searchsorted(sel, probe_rows)
            ok = (at < len(sel)) & (sel[np.minimum(at, len(sel) - 1)] == probe_rows)
            pos = np.where(ok, at, -1)
        m = _metrics_at(S, probes, pos, n_anchors, vor_cap, rng)
        if m is not None:
            xs.append(int(n))
            vals.append(m)
    if not xs:
        return np.array([]), {}
    keys = [k for k in vals[0] if k != "n_set" and k != "n_vor_generators"]
    return (np.array(xs, dtype=float),
            {k: np.array([v[k] for v in vals]) for k in keys})


PAPER_METRICS = ("voronoi", "nn")   # the pair the paper carries, top to bottom
# At half height a panel is ~2 inches tall and the full y labels above, set
# sideways, are longer than that: they run into the neighbouring row. The
# combined figure therefore names the measure and leaves the sentence to the
# caption, which is the convention the paper follows anyway.
SHORT_YLABEL = {"nn": "nearest-neighbour distance\n(9-D box units)",
                "voronoi": "Voronoi coverage radius\n(9-D box units)",
                "vorunif": "Voronoi cell-volume CV"}


def _figures(plt, phr, data, ref_calls, n_per_model, n_pool_total, output_dir,
             run_set_label, full_range, panel_titles):
    """One figure per metric, plus the combined figure the paper carries.

    The combined one stacks the two paper metrics, coverage above diversity, on
    half the vertical extent each, so the pair costs the page what one of them
    used to. Its panels are otherwise identical to the standalone figures.
    """
    def _ratio(f, c, own_f, own_c):
        f, c = np.asarray(f, float), np.asarray(c, float)
        ok = (f >= own_f[0]) & (f <= own_f[-1])
        den = np.interp(np.log(f[ok]), np.log(own_f), own_c)
        return f[ok], c[ok] / den

    def _draw(ax, axr, col, met, ylabel, col_title, *, legend=True, xlabel=True,
              fs=1.0, ratio_fs=8.5, tighten=False):
        """The (absolute, ratio) panel pair for one metric and one column."""
        cd = data[col]
        own_f, own_c = cd["ref"]["pool"][0], cd["ref"]["pool"][1][met]
        for rsrc, (f, m) in cd["ref"].items():
            ls, colr = SOURCE_STYLE[rsrc]
            ax.plot(f, m[met], ls=ls, color=colr, lw=1.7,
                    label=f"{SOURCE_CURVE[rsrc]} ({ref_calls[rsrc]:,} calls)")
            if rsrc != "pool":
                rf, rr = _ratio(f, m[met], own_f, own_c)
                axr.plot(rf, rr, ls=ls, color=colr, lw=1.6)
        for key, (g, agg, n_rep) in cd["al"].items():
            model, strat, warm = key
            mu, lo_b, hi_b = agg[met]
            base = phr.MODEL_DISPLAY.get(model, model)
            lbl = base if n_per_model[model] == 1 else f"{base} ({strat}/{warm})"
            colc = phr.MODEL_COLORS.get(model, "gray")
            ax.plot(g, mu, "-", color=colc, lw=1.9, label=lbl)
            if n_rep > 1:
                ax.fill_between(g, lo_b, hi_b, color=colc, alpha=0.15, lw=0)
            rf, rr = _ratio(g, mu, own_f, own_c)
            axr.plot(rf, rr, "-", color=colc, lw=1.8)
            if n_rep > 1:
                _, rlo = _ratio(g, lo_b, own_f, own_c)
                _, rhi = _ratio(g, hi_b, own_f, own_c)
                axr.fill_between(rf, rlo, rhi, color=colc, alpha=0.15, lw=0)
        ax.set_xscale("log")
        # Span the decades where every population is present: from the first
        # budget the emcee curve can express (its MIN_SET rows times its
        # calls-per-row charge) to x = 1, the cost of the whole random scan.
        # The chains run on to x = 36, which is cropped: past x = 1 there is
        # no equal-budget comparison left to make against the scan.
        if not full_range and "mcmc" in cd["ref"]:
            ax.set_xlim(cd["ref"]["mcmc"][0][0], 1.0)
        if met != "vorunif":
            ax.set_yscale("log")
        ax.set_ylabel(ylabel, fontsize=10 * fs if fs != 1.0 else None)
        # The paper's figures carry no in-figure titles: the caption names
        # the columns instead. Kept behind a flag for standalone use.
        if panel_titles:
            ax.set_title(col_title, fontsize=9)
        ax.grid(alpha=0.3, which="both")
        ax.tick_params(axis="x", labelbottom=False)
        # Semi-transparent: "best" has no notion of which curve it covers,
        # and on the uniformity panel it sat exactly on top of the emcee
        # curve's first decade, which read as that curve starting late.
        if legend:
            ax.legend(fontsize=7.5 * fs, loc="best", framealpha=0.55)
        axr.axhline(1.0, color=SOURCE_STYLE["pool"][1], lw=1.4,
                    ls=SOURCE_STYLE["pool"][0], zorder=1)
        if xlabel:
            axr.set_xlabel("points spent / size of the static random scan\n"
                           f"({n_pool_total:,} valid models simulated)")
        else:
            axr.tick_params(axis="x", labelbottom=False)
        axr.set_ylabel("/ random scan", fontsize=ratio_fs * fs)
        axr.grid(alpha=0.3, which="both")
        axr.tick_params(axis="y", labelsize=8 * fs)
        if tighten:
            # Autoscale happens before the x zoom, so the y range is set by the
            # emcee curve's tail out at x = 36, which the panel does not show:
            # a third of the height goes to a decade holding nothing. Rescale to
            # what is inside the zoom. Only the half-height combined figure asks
            # for this, the standalone ones keeping the range they were made
            # with.
            x0, x1 = ax.get_xlim()
            for a in (ax, axr):
                v_lo, v_hi = np.inf, -np.inf
                for ln in a.get_lines():
                    x = np.asarray(ln.get_xdata(), float)
                    y = np.asarray(ln.get_ydata(), float)
                    if x.size != y.size or not x.size:
                        continue
                    sel = (x >= x0) & (x <= x1) & np.isfinite(y)
                    if sel.any():
                        v_lo, v_hi = min(v_lo, y[sel].min()), max(v_hi, y[sel].max())
                if not np.isfinite(v_lo) or v_hi <= v_lo:
                    continue
                if a.get_yscale() == "log":
                    pad = 0.07 * np.log10(v_hi / v_lo)
                    a.set_ylim(v_lo * 10 ** -pad, v_hi * 10 ** pad)
                else:
                    pad = 0.07 * (v_hi - v_lo)
                    a.set_ylim(v_lo - pad, v_hi + pad)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    written = []
    for met, (ylabel, stem_base) in METRICS.items():
        fig, axes = plt.subplots(2, len(COLUMNS), squeeze=False, sharex="col",
                                 figsize=(6.6 * len(COLUMNS), 6.9),
                                 gridspec_kw={"height_ratios": [3.0, 1.2],
                                              "hspace": 0.06})
        for ci, (col, col_title) in enumerate(COLUMNS):
            _draw(axes[0][ci], axes[1][ci], col, met, ylabel, col_title)
        fig.tight_layout(h_pad=0.4)
        p_png = Path(output_dir) / f"{stem_base}_{run_set_label}.png"
        fig.savefig(p_png, dpi=200)
        plt.close(fig)
        written.append(str(p_png))
        click.echo(f"[div] wrote {p_png}")

    fig, axes = plt.subplots(2 * len(PAPER_METRICS), len(COLUMNS),
                             squeeze=False, sharex="col",
                             figsize=(6.6 * len(COLUMNS), 6.9),
                             gridspec_kw={"height_ratios":
                                          [3.0, 1.2] * len(PAPER_METRICS),
                                          "hspace": 0.06, "wspace": 0.26})
    for mi, met in enumerate(PAPER_METRICS):
        for ci, (col, col_title) in enumerate(COLUMNS):
            # One legend for the four rows: the curves are the same set in all
            # of them, and seven entries repeated per panel would take a third
            # of the height each panel now has.
            _draw(axes[2 * mi][ci], axes[2 * mi + 1][ci], col, met,
                  SHORT_YLABEL[met], col_title,
                  legend=(mi == 0 and ci == 0),
                  xlabel=(mi == len(PAPER_METRICS) - 1),
                  ratio_fs=7.5, tighten=True)
    fig.tight_layout(h_pad=0.3)
    p_png = Path(output_dir) / f"coverage_diversity_{run_set_label}.png"
    fig.savefig(p_png, dpi=200)
    plt.close(fig)
    written.append(str(p_png))
    click.echo(f"[div] wrote {p_png}")
    return written


@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
              show_default=True)
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/all_runs", show_default=True)
@click.option("--baseline-data-dir", default="/ptmp/jwuerzin/data/18387358", show_default=True)
@click.option("--mcmc-data-dir", default="/ptmp/jwuerzin/data/neutralino_v4", show_default=True)
@click.option("--target", default="DMRD", show_default=True)
@click.option("--model-tag", default="", show_default=True,
              help="OUTPUT_TAG of a variant sweep (e.g. 'expr').")
@click.option("--run-set-label", default="joint", show_default=True)
@click.option("--include-status", default="completed,running,timeout,submitted",
              show_default=True)
@click.option("--models", default=None, help="Comma list of manifest model names.")
@click.option("--all-cells/--picks-only", default=False, show_default=True)
@click.option("--tolerance", default=0.10, show_default=True)
@click.option("--pool-files", default=60, show_default=True,
              help="ROOT files of the random scan to read (-1 for all). A prefix "
                   "of an i.i.d. dataset, so a partial read is a valid partial "
                   "scan; 60 files is ~267k valid models, past the AL budgets.")
@click.option("--pool-total-rows", default=0, show_default=True,
              help="Valid models in the WHOLE scan, for the x axis. 0 = use "
                   "however many were read, which is only right with "
                   "--pool-files -1.")
@click.option("--mcmc-max-samples", default=300_000, show_default=True)
@click.option("--mcmc-total-rows", default=48_305_152, show_default=True,
              help="Proposals in the raw chains, for the x axis of that curve.")
@click.option("--no-mcmc", is_flag=True, default=False,
              help="Skip the posterior reference.")
@click.option("--n-anchors", default=3000, show_default=True,
              help="Points of the set sampled per budget for the NN mean.")
@click.option("--n-probes", default=8000, show_default=True,
              help="Probe points per budget for the two Voronoi metrics.")
@click.option("--vor-cap", default=2000, show_default=True,
              help="Generators kept for the cell-volume spread, so that the "
                   "probes-per-cell ratio stays high enough to mean anything.")
@click.option("--n-points", default=18, show_default=True,
              help="Budgets per curve.")
@click.option("--min-seeds", default=2, show_default=True)
@click.option("--panel-titles", is_flag=True, default=False,
              help="Print the column description above each panel. Off by "
                   "default, since the paper's convention is that captions, not "
                   "figures, carry descriptions.")
@click.option("--full-range", is_flag=True, default=False,
              help="Skip the x zoom. By default each panel spans from where its "
                   "emcee curve begins to x = 1, the whole random scan, which is "
                   "the range where all three populations are present; the "
                   "chains themselves continue to x = 36 and are cropped.")
@click.option("--pool-cache", default="", show_default=True,
              help="Directory holding the parsed-pool x_full/y_full .npy cache "
                   "(as plot_hit_rate_trajectories_multiseed writes it). Given, "
                   "the scan is read from there instead of from the ROOT files, "
                   "which is the same pool and spares the 11 GB read.")
@click.option("--from-json", "from_json", default=None,
              help="Re-draw the figures from a previous run's "
                   "diversity_metrics_<label>.json instead of recomputing the "
                   "curves. The JSON stores the curves and the pool size but "
                   "not the chains' proposal count, so pass the same "
                   "--mcmc-total-rows that produced it.")
@click.option("--require-neutralino-lsp/--no-require-neutralino-lsp",
              default=False, show_default=True)
def main(manifest, output_dir, baseline_data_dir, mcmc_data_dir, target,
         model_tag, run_set_label, include_status, models, all_cells, tolerance,
         pool_files, pool_total_rows, mcmc_max_samples, mcmc_total_rows, no_mcmc,
         n_anchors, n_probes, vor_cap, n_points, min_seeds, panel_titles,
         full_range, pool_cache, from_json, require_neutralino_lsp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_hit_rate_trajectories_multiseed as phr
    from pmssm.config import TARGET_CONFIG

    if from_json:
        # Everything the figures need is in the JSON the computing run wrote,
        # so a re-draw costs nothing: no ROOT read, no KD-trees.
        rec = json.loads(Path(from_json).read_text())
        cfg = rec["config"]
        n_pool_total = int(cfg["pool_rows_total"])
        data = {}
        for col, c in rec["columns"].items():
            cd: dict[str, dict] = {"ref": {}, "al": {}}
            for rsrc, r in c["reference"].items():
                cd["ref"][rsrc] = (np.asarray(r["fraction"], float),
                                   {k: np.asarray(v, float)
                                    for k, v in r.items() if k != "fraction"})
            for key, r in c["al"].items():
                cd["al"][tuple(key.split("/"))] = (
                    np.asarray(r["fraction"], float),
                    {k: (np.asarray(v["mean"], float),
                         np.asarray(v["min"], float),
                         np.asarray(v["max"], float))
                     for k, v in r.items()
                     if k not in ("fraction", "n_replicas")},
                    int(r["n_replicas"]))
            data[col] = cd
        n_per_model = {}
        for k in next(iter(data.values()))["al"]:
            n_per_model[k[0]] = n_per_model.get(k[0], 0) + 1
        _figures(plt, phr, data,
                 {"pool": n_pool_total, "mcmc": mcmc_total_rows}, n_per_model,
                 n_pool_total, output_dir, cfg.get("run_set", run_set_label),
                 full_range, panel_titles)
        return

    true_val = float(TARGET_CONFIG[target]["true_value"])
    # Before the pool load: load_pmssm_data writes a histogram into plot_dir.
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dims, lo, hi = _free_dims()
    span = hi - lo
    rng = np.random.default_rng(RNG_SEED)
    click.echo(f"[div] target={target} band=|y-{true_val:g}|/{true_val:g} < "
               f"{tolerance:g} | {len(dims)} free dims normalised by their scan "
               f"ranges | {n_anchors} anchors, {n_probes} probes per budget")

    def _norm(X):
        """(N, 9) in [0, 1] per axis, from a (N, 19) physical-unit array."""
        return ((np.asarray(X, dtype=np.float64)[:, dims] - lo) / span)

    # ── AL runs ──────────────────────────────────────────────────────────────
    statuses = {s.strip() for s in include_status.split(",")}
    want = {m.strip() for m in models.split(",")} if models else None
    cells = _discover(manifest, statuses, picks_with_tag(model_tag), all_cells,
                      want)
    if not cells:
        raise click.ClickException(f"no usable runs in {manifest}")
    n_per_model: dict[str, int] = {}
    for m, _s, _w in cells:
        n_per_model[m] = n_per_model.get(m, 0) + 1
    runs: dict[tuple, list] = {}
    for key, dirs in sorted(cells.items()):
        seqs = []
        for d in dirs:
            try:
                got = _al_sequence(d, list(range(len(PARAM_ORDER))),
                                   require_neutralino_lsp)
            except Exception as exc:                        # noqa: BLE001
                click.echo(f"[div]   skip {Path(d).name}: "
                           f"{type(exc).__name__}: {exc}", err=True)
                continue
            if got is not None:
                seqs.append((_norm(got[0]), got[1]))
        if seqs:
            runs[key] = seqs
            click.echo(f"[div] {'/'.join(key):<48} {len(seqs)} replica(s), "
                       f"points spent {[len(y) for _x, y in seqs]}")
    # Guard against a state.pt that stores transformed inputs: the normalisation
    # assumes physical units, and a standardised array would land far outside
    # [0, 1] and make every distance meaningless.
    _chk = next(iter(runs.values()))[0][0]
    click.echo(f"[div] normalised AL inputs span "
               f"[{_chk.min():.3f}, {_chk.max():.3f}] over {_chk.shape[1]} dims "
               f"(expected within [0, 1])")
    if _chk.min() < -0.05 or _chk.max() > 1.05:
        raise click.ClickException(
            "AL inputs are not in physical units after normalisation; the "
            "distance metric would be meaningless")

    # ── static references ────────────────────────────────────────────────────
    refs = {}
    if pool_cache:
        # The .npy cache is what load_pmssm_data returned on the run that wrote
        # it: same validity mask, same row order, the whole scan. It carries no
        # LSP column, so the neutralino veto cannot be applied on this path.
        if require_neutralino_lsp:
            raise click.ClickException(
                "--pool-cache holds no LSP type, so --require-neutralino-lsp "
                "cannot be honoured; drop one of the two.")
        Xp, Yp = phr._load_xy_full(baseline_data_dir, target, Path(pool_cache))
        click.echo(f"[div] pool from cache in {pool_cache}: {len(Yp):,} rows")
    else:
        from pmssm.data import load_pmssm_data
        Xp, Yp = load_pmssm_data(n_datasets=pool_files,
                                 data_dir=baseline_data_dir, target=target,
                                 plot_dir=str(output_dir),
                                 require_neutralino_lsp=require_neutralino_lsp)
    Xp = _norm(Xp.numpy() if hasattr(Xp, "numpy") else Xp)
    Yp = np.asarray(Yp.numpy() if hasattr(Yp, "numpy") else Yp,
                    dtype=np.float64).ravel()
    n_pool_read = len(Yp)
    n_pool_total = pool_total_rows or n_pool_read
    # (X, Y, calls per retained row, label total). Unlike the support figure,
    # which gives each dataset its own panel and its own denominator, every
    # curve here shares the random scan's denominator, so each dataset has to be
    # charged its TRUE number of simulator calls. A pool row is one call; a
    # retained emcee row stands for total_rows / len(sample) proposals, and
    # forgetting that factor shifts the posterior's curve left by exactly that
    # factor and flatters it enormously.
    refs["pool"] = (Xp, Yp, 1.0, n_pool_total)
    click.echo(f"[div] pool: read {n_pool_read:,} valid models from "
               f"{pool_files} file(s); x axis divides by {n_pool_total:,}")
    if not no_mcmc:
        from pmssm.data import load_mcmc_data
        Xm, Ym = load_mcmc_data(data_dir=mcmc_data_dir, target=target,
                                require_neutralino_lsp=require_neutralino_lsp,
                                max_samples=mcmc_max_samples)
        Xm = _norm(Xm.numpy() if hasattr(Xm, "numpy") else Xm)
        Ym = np.asarray(Ym.numpy() if hasattr(Ym, "numpy") else Ym,
                        dtype=np.float64).ravel()
        per_row = mcmc_total_rows / len(Ym)
        # The stored subsample keeps the ntuples' file order, so a PREFIX of it
        # is the first rows of the first chain rather than a draw from the
        # posterior: at small k it is one walker's trajectory, which is far more
        # clumped than the posterior it came from and biases every metric here
        # (NN low, coverage radius high, cell volumes wildly uneven, the last of
        # which produced a spurious rise-and-fall in the uniformity figure).
        # These metrics do not care about order at fixed k, so one seeded shuffle
        # makes a prefix exactly what the docstring claims it is. The support
        # figure faces the same trap and answers it differently, by interleaving
        # the four ensembles round-robin, because a COVERAGE curve does depend on
        # order and has to describe a genuine partial run.
        perm = rng.permutation(len(Ym))
        Xm, Ym = Xm[perm], Ym[perm]
        refs["mcmc"] = (Xm, Ym, per_row, mcmc_total_rows)
        click.echo(f"[div] mcmc: {len(Ym):,} subsampled rows of "
                   f"{mcmc_total_rows:,} proposals, so one retained row is "
                   f"charged {per_row:.1f} calls")

    # ── probe populations, one per column ────────────────────────────────────
    # "all": uniform over the prior box, so coverage means coverage of the scan.
    # "in-band": in-band models of a reference population, so coverage means
    # "how far is a typical in-band model from the nearest in-band model this
    # run acquired". Both are fixed once and shared by every curve, which is
    # what makes the curves comparable.
    #
    # Nothing is held out. The probes for the in-band column are rows of one of
    # the reference datasets, and they stay in it; the self-match is removed per
    # query instead, by row identity, inside ``_metrics_at``. That keeps this
    # figure's convention the same as the support figure's, where every dataset
    # also contributes all of its rows to its own curve, and it avoids a hold-out
    # that emcee's repeated rows would partly defeat anyway: a removed probe can
    # have an exact duplicate of itself still in the sample.
    probe_src = "mcmc" if "mcmc" in refs else "pool"
    Xr, Yr = refs[probe_src][0], refs[probe_src][1]
    inb_ref = np.where(np.abs(Yr - true_val) / true_val < tolerance)[0]
    take = (rng.choice(inb_ref, size=n_probes, replace=False)
            if len(inb_ref) > n_probes else inb_ref)
    take = np.sort(take)
    probes = {"all": (rng.random((n_probes, len(dims))), None, ""),
              "inband": (Xr[take], take, probe_src)}
    click.echo(f"[div] probes: all={len(probes['all'][0]):,} uniform in the box, "
               f"inband={len(take):,} of the {probe_src}'s {len(inb_ref):,} "
               f"in-band rows, kept in it and left out per query")

    out = {"config": {"target": target, "run_set": run_set_label,
                      "tolerance": tolerance, "n_anchors": n_anchors,
                      "n_probes": n_probes, "vor_cap": vor_cap,
                      "free_dims": [PARAM_ORDER[j] for j in dims],
                      "distance": "Euclidean in the 9 free parameters, each "
                                  "normalised to [0, 1] by its scan range",
                      "pool_files": pool_files, "pool_rows_read": n_pool_read,
                      "pool_rows_total": n_pool_total,
                      "budget_unit": "valid models simulated (AL: train+val)",
                      "manifest": manifest},
           "columns": {}}

    # ── one pass over the data, all metrics ──────────────────────────────────
    # {column: {"ref": {src: (f, {metric: v})}, "al": {key: (f, {metric: (mu, lo, hi)})}}}
    data: dict[str, dict] = {}
    for col, _title in COLUMNS:
        P, P_rows, P_src = probes[col]
        cd = {"ref": {}, "al": {}}
        for rsrc, (Xr, Yr, per_row, n_tot) in refs.items():
            band = (np.abs(Yr - true_val) / true_val < tolerance
                    if col == "inband" else None)
            # Floor the grid in CALLS, not rows: a reference charged per_row
            # calls per retained row would otherwise begin at 200 * per_row
            # calls, which for a thinned emcee sample is tens of thousands and
            # cuts the whole low-budget decade off that curve. MIN_SET still
            # applies, since a nearest-neighbour mean over fewer points than
            # that is noise.
            floor = max(MIN_SET, int(200 / per_row))
            # Only the dataset the probes came FROM can self-match.
            f, m = _curve(Xr, Yr, band, _grid(len(Yr), n_points, floor), P,
                          P_rows if rsrc == P_src else None,
                          n_anchors, vor_cap, rng)
            cd["ref"][rsrc] = (f * per_row / n_pool_total, m)
            click.echo(f"[div] {col:<7} ref {rsrc:<5} {len(f)} budgets, "
                       f"nn {m['nn'][-1]:.4f} voronoi {m['voronoi'][-1]:.4f} "
                       f"vorunif {m['vorunif'][-1]:.3f}")
        for key, seqs in runs.items():
            per_seed = []
            for Xs, Ys in seqs:
                band = (np.abs(Ys - true_val) / true_val < tolerance
                        if col == "inband" else None)
                f, m = _curve(Xs, Ys, band, _grid(len(Ys), n_points, 2000), P,
                              None, n_anchors, vor_cap, rng)
                if len(f):
                    per_seed.append((f / n_pool_total, m))
            if not per_seed:
                continue
            # Average on the union grid, keeping budgets at least --min-seeds
            # replicas reached, as the support figure's band does.
            grid = np.unique(np.concatenate([f for f, _m in per_seed]))
            agg = {}
            for met in per_seed[0][1]:
                stack = []
                for f, m in per_seed:
                    v = np.full(len(grid), np.nan)
                    ok = (grid >= f[0]) & (grid <= f[-1])
                    v[ok] = np.interp(grid[ok], f, m[met])
                    stack.append(v)
                agg[met] = np.vstack(stack)
            have = np.sum(np.isfinite(agg["nn"]), axis=0)
            keep = have >= min(min_seeds, len(per_seed))
            g = grid[keep]
            cd["al"][key] = (g, {met: (np.nanmean(M[:, keep], axis=0),
                                       np.nanmin(M[:, keep], axis=0),
                                       np.nanmax(M[:, keep], axis=0))
                                 for met, M in agg.items()},
                             len(per_seed))
            click.echo(f"[div] {col:<7} {'/'.join(key):<44} "
                       f"nn {np.nanmean(agg['nn'][:, keep], axis=0)[-1]:.4f} "
                       f"voronoi {np.nanmean(agg['voronoi'][:, keep], axis=0)[-1]:.4f} "
                       f"vorunif {np.nanmean(agg['vorunif'][:, keep], axis=0)[-1]:.3f}")
        data[col] = cd

    written = _figures(plt, phr, data, {k: v[3] for k, v in refs.items()},
                       n_per_model, n_pool_total, output_dir, run_set_label,
                       full_range, panel_titles)

    for col, cd in data.items():
        rec = {"reference": {}, "al": {}}
        for rsrc, (f, m) in cd["ref"].items():
            rec["reference"][rsrc] = {"fraction": f.tolist(),
                                      **{k: v.tolist() for k, v in m.items()}}
        for key, (g, agg, n_rep) in cd["al"].items():
            rec["al"]["/".join(key)] = {
                "n_replicas": n_rep, "fraction": g.tolist(),
                **{k: {"mean": v[0].tolist(), "min": v[1].tolist(),
                       "max": v[2].tolist()} for k, v in agg.items()}}
        out["columns"][col] = rec
    stem = f"diversity_metrics_{run_set_label}"
    (Path(output_dir) / f"{stem}.json").write_text(json.dumps(out, indent=1))
    click.echo(f"[div] wrote {stem}.json and {len(written)} figures")


if __name__ == "__main__":
    main()
