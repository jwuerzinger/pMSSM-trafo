"""Dropout-vs-Laplace input marginals with a ratio panel per quantity.

Figure 5 (``input_overlay_inband.png``) overlays every surrogate's in-band
marginals against the emcee reference, which answers "which model recovers the
reference?". This script answers a different question: for the three
architectures where both acquisition uncertainties were run, does swapping MC
dropout for the last-layer Laplace posterior move *where the loop looks*?

Same marginals, but each quantity gets a ratio sub-panel (Laplace / dropout),
because a shift of a few percent in a broad marginal is invisible in an overlay
and obvious in a ratio. Omega is included alongside the nine free parameters:
the inputs say where the loop searched, Omega says what it found there.

Only the three matched pairs are drawn. The GPs and TabPFN have native
posteriors and no Laplace arm, so they have nothing to compare against.

Usage:
    python scripts/plot_laplace_input_overlays.py \\
        --manifest /ptmp/jwuerzin/analysis/laplace_vs_dropout/manifest.csv \\
        --output-dir /ptmp/jwuerzin/analysis/laplace_vs_dropout \\
        --require-neutralino-lsp
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D

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
from plot_hit_rate_trajectories_multiseed import (  # noqa: E402
    LAPLACE_LS,
    MODEL_COLORS,
    MODEL_DISPLAY,
)

# The three matched pairs, in the order the appendix discusses them.
PAIRS = [
    ("transformer", "transformer_laplace"),
    ("dnn", "dnn_laplace"),
    ("dnn_match_trafo", "dnn_match_trafo_laplace"),
]

TRUE_VALUE = 0.12
OMEGA_LABEL = r"$\Omega_\chi h^2$"


def _pool_cell(run_dirs, require_neutralino_lsp):
    """Pool one cell's seeds. Returns (X_free (N,9), Omega (N,), n_seeds).

    state.pt's Y is Omega in physical units already, not the transformed
    target, which is why the in-band cut below compares it to TRUE_VALUE
    directly.
    """
    xs, ys, n_used = [], [], 0
    for d in run_dirs:
        try:
            run = load_run(d)
        except Exception as exc:                            # noqa: BLE001
            click.echo(f"[laplace-overlay]   skipping {d}: {exc}", err=True)
            continue
        if require_neutralino_lsp:
            run = filter_run_neutralino_lsp(run)
        xs.append(np.asarray(run.X)[:, FREE_PARAM_INDICES])
        ys.append(np.asarray(run.Y, dtype=float).ravel())
        n_used += 1
    if not xs:
        return None, None, 0
    return np.concatenate(xs), np.concatenate(ys), n_used


def _ratio_with_err(v_lap, v_drop, bins):
    """Density ratio Laplace/dropout per bin, with its Poisson uncertainty.

    The ratio is taken on *densities* so the arms' unequal sample sizes do not
    masquerade as a shape difference, but the error comes from the raw counts,
    which is what says whether a given bin's excursion means anything. Without
    the band these panels are unreadable: at a few thousand points over tens of
    bins, +-20% swings are pure counting noise.
    """
    cl, _ = np.histogram(v_lap, bins=bins)
    cd, _ = np.histogram(v_drop, bins=bins)
    dl, _ = np.histogram(v_lap, bins=bins, density=True)
    dd, _ = np.histogram(v_drop, bins=bins, density=True)
    ratio = np.full(len(bins) - 1, np.nan)
    err = np.full(len(bins) - 1, np.nan)
    ok = (dd > 0) & (cl > 0) & (cd > 0)
    ratio[ok] = dl[ok] / dd[ok]
    err[ok] = ratio[ok] * np.sqrt(1.0 / cl[ok] + 1.0 / cd[ok])
    return ratio, err


def plot_overlay_with_ratio(data, names, outpath, nbins=25, ncols=3,
                            ratio_ylim=(0.5, 1.5), nbins_omega=20,
                            pairs=None, colors=None, labels=None,
                            ratio_label="Lap./drop.", ratio_ref=None):
    """One panel per quantity: density overlay above, B/A ratio below.

    data: {key: values (N, n_quantities)} containing both arms of every pair.
    pairs: list of (A_key, B_key); defaults to the Laplace PAIRS above. The
    ratio is always B over A, so the second element is the variant under test.
    colors/labels: optional overrides keyed like data, for callers whose keys
    are not model names (e.g. the budget probes, where both arms are the same
    architecture and only the budget differs).
    """
    pairs = PAIRS if pairs is None else pairs
    colors = MODEL_COLORS if colors is None else colors
    labels = MODEL_DISPLAY if labels is None else labels
    # With a reference series, the ratio panel compares every arm against it
    # instead of pairing arms with each other. That answers "how far is each
    # from the truth?" rather than "did the budget change anything?", and the
    # reference is drawn in the overlay too so the panels can be read together.
    ref = data.get(ratio_ref) if ratio_ref else None
    n = len(names)
    nrows = int(np.ceil(n / ncols))
    # constrained_layout with a nested 2-row subgridspec per quantity: a flat
    # GridSpec plus tight_layout silently drops the x labels of every row but
    # the last, because the label lands under the next row's panel.
    fig = plt.figure(figsize=(4.9 * ncols, 3.6 * nrows), constrained_layout=True)
    outer = fig.add_gridspec(nrows, ncols)

    present = [(a, b) for a, b in pairs if a in data and b in data]

    for j, name in enumerate(names):
        r, c = divmod(j, ncols)
        inner = outer[r, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
        ax = fig.add_subplot(inner[0])
        axr = fig.add_subplot(inner[1], sharex=ax)

        # Shared binning across every arm, so ratios are defined bin-by-bin.
        finite = [d[:, j][np.isfinite(d[:, j])] for d in data.values()]
        lo = min(float(v.min()) for v in finite if v.size)
        hi = max(float(v.max()) for v in finite if v.size)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            ax.axis("off"); axr.axis("off"); continue
        # Omega is stored quantised (exactly 0.001 here, 25 distinct values
        # across the in-band range), so any binning not aligned to that step
        # aliases into a comb: some bins collect two lattice values and their
        # neighbours collect one. Detect a small value lattice and give each
        # value its own bin instead of choosing a bin count.
        allv = np.concatenate([d[:, j][np.isfinite(d[:, j])] for d in data.values()])
        uniq = np.unique(allv)
        if len(uniq) <= 60:
            step = float(np.median(np.diff(uniq))) if len(uniq) > 1 else 1.0
            bins = np.concatenate([uniq - step / 2.0, [uniq[-1] + step / 2.0]])
        else:
            nb = nbins_omega if name == OMEGA_LABEL else nbins
            bins = np.linspace(lo, hi, nb + 1)
        centres = 0.5 * (bins[:-1] + bins[1:])

        if ref is not None:
            rv = ref[:, j]
            ax.hist(rv[np.isfinite(rv)], bins=bins, density=True,
                    histtype="step", lw=2.2, color="black", zorder=5)
        for base, lap in present:
            for key, ls in ((base, "-"), (lap, LAPLACE_LS)):
                v = data[key][:, j]
                v = v[np.isfinite(v)]
                ax.hist(v, bins=bins, density=True, histtype="step", lw=1.6,
                        color=colors.get(key), linestyle=ls,
                        label=labels.get(key, key))
                if ref is not None:
                    ratio, err = _ratio_with_err(v, ref[:, j], bins)
                    axr.step(centres, ratio, where="mid", lw=1.3,
                             color=colors.get(key), ls=ls)
                    axr.fill_between(centres, ratio - err, ratio + err,
                                     step="mid", color=colors.get(key),
                                     alpha=0.12, lw=0)
            if ref is None:
                ratio, err = _ratio_with_err(data[lap][:, j], data[base][:, j], bins)
                col = colors.get(lap)
                axr.step(centres, ratio, where="mid", lw=1.4, color=col)
                axr.fill_between(centres, ratio - err, ratio + err, step="mid",
                                 color=col, alpha=0.18, lw=0)

        ax.set_yticks([])
        ax.tick_params(labelbottom=False)
        ax.grid(alpha=0.3)
        axr.axhline(1.0, color="0.35", lw=1.0, ls=":")
        axr.set_ylim(*ratio_ylim)
        axr.set_xlabel(name)
        axr.grid(alpha=0.3)
        if c == 0:
            axr.set_ylabel(ratio_label, fontsize=9)

    # Legend handles built to mirror the curves: colour identifies the cell,
    # linestyle identifies the uncertainty. A hardcoded solid marker here would
    # make the two arms of a pair indistinguishable in the legend.
    handles = []
    for base, lap in present:
        handles.append(Line2D([], [], color=colors.get(base), ls="-",
                              lw=1.8, label=labels.get(base, base)))
        handles.append(Line2D([], [], color=colors.get(lap),
                              ls=LAPLACE_LS, lw=1.8,
                              label=labels.get(lap, lap)))
    if ref is not None:
        handles.append(Line2D([], [], color="black", lw=2.2,
                              label=labels.get(ratio_ref, "emcee reference")))
    fig.legend(handles=handles, loc="outside upper center",
               ncol=min(len(handles), 3), fontsize=10, frameon=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


@click.command()
@click.option("--manifest",
              default="/ptmp/jwuerzin/analysis/laplace_vs_dropout/manifest.csv",
              show_default=True)
@click.option("--output-dir",
              default="/ptmp/jwuerzin/analysis/laplace_vs_dropout",
              show_default=True)
@click.option("--tolerance", default=0.10, show_default=True,
              help="In-band half-width as a fraction of the target.")
@click.option("--require-neutralino-lsp/--no-require-neutralino-lsp",
              default=True, show_default=True,
              help="Drop non-neutralino-LSP rows, matching Fig. 5.")
@click.option("--nbins", default=25, show_default=True)
def main(manifest, output_dir, tolerance, require_neutralino_lsp, nbins):
    import csv
    from collections import defaultdict

    rows = list(csv.DictReader(open(manifest)))
    per_model = defaultdict(list)
    for r in rows:
        per_model[r["model"]].append(r["expected_run_dir"])

    pooled = {}
    for key in [m for pair in PAIRS for m in pair]:
        if key not in per_model:
            click.echo(f"[laplace-overlay] {key}: absent from manifest", err=True)
            continue
        X, om, n_seeds = _pool_cell(per_model[key], require_neutralino_lsp)
        if X is None:
            continue
        pooled[key] = (X, om)
        click.echo(f"[laplace-overlay] {key}: {n_seeds} seeds, {len(X)} points")

    missing = [m for pair in PAIRS for m in pair if m not in pooled]
    if missing:
        click.echo(f"[laplace-overlay] incomplete pairs, missing {missing}", err=True)

    names = list(FREE_PARAM_NAMES) + [OMEGA_LABEL]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Two figures. The in-band one is the direct analogue of Fig. 5 and is what
    # the composition discussion needs; the all-points one is the only place the
    # Omega marginal is informative, since the in-band cut truncates it by
    # construction.
    for tag, restrict in (("inband", True), ("all", False)):
        data = {}
        for key, (X, om) in pooled.items():
            if restrict:
                keep = np.abs(om - TRUE_VALUE) / TRUE_VALUE < tolerance
                if keep.sum() < 50:
                    click.echo(f"[laplace-overlay] {key}: only {int(keep.sum())} "
                               f"in-band points; excluded", err=True)
                    continue
                data[key] = np.column_stack([X[keep], om[keep]])
            else:
                data[key] = np.column_stack([X, om])
        if len(data) < 2:
            click.echo(f"[laplace-overlay] {tag}: too few cells; skipped", err=True)
            continue
        path = out / f"laplace_input_overlay_{tag}.png"
        plot_overlay_with_ratio(data, names, str(path), nbins=nbins)
        click.echo(f"[laplace-overlay] wrote {path} ("
                   + ", ".join(f"{k}: n={len(v)}" for k, v in data.items()) + ")")


if __name__ == "__main__":
    main()
