"""In-band support covered versus budget, normalised by each dataset's own cost.

`coverage_saturation.py` answers "does splitting a budget over replicas cover
more?" and puts absolute simulator calls on the x-axis. That leaves the more
basic question unanswered: **is active learning a cheaper way to reach a
dataset's in-band support than generating that dataset was?**

This figure answers it by rescaling the budget axis. One panel per static
dataset. Within a panel the support is defined by that dataset (identical
definition to `coverage_saturation.build_target`: equal-occupancy quantile cells
in the informative subspace, built from a random half of the dataset's in-band
points, cells needing at least ``--min-cell`` of them), and every curve's budget
is divided by the number of models that dataset contains. So

  * x = 1     is "you have spent what the whole dataset cost to generate", and
              is where the dataset's own curve necessarily reaches 1.0, since
              the support is defined by that dataset
  * a curve to the LEFT of the dataset's own curve at the same height is
    cheaper than generating the dataset; to the right, more expensive.

Curves drawn in every panel:

  AL, per model   the accumulated labelled set in acquisition order, averaged
                  over seed replicas (band = min/max over seeds)
  random scan     the whole static random pool, prefixes in scan order
  MCMC            the RAW emcee chains, prefixes in step order with burn-in and
                  the repeated rows left by rejected proposals; the band and the
                  cells still come from the post-burn-in ntuples
  ATLAS scan      the public ATLAS pMSSM EWK scan, both campaigns merged

Nothing is held out: a panel's own dataset defines the support AND supplies its
curve, so that curve reaches 1.0 at x = 1 by construction. That is the definition
of the support rather than a result, and the curve's SHAPE is the claim. The
alternative, splitting in half, halved the defining set and offered no defensible
way to hold out part of four asymmetric chains.

**Off-band variant.** ``--band-side out`` builds the same figure from the
COMPLEMENT of the tolerance band: the support is defined by the dataset's
off-band points and a run is credited only for its own off-band points. It is
the specificity counterpart, showing what coverage of the excluded region a
focused acquisition trades away, and it is written to a ``_offband`` filename.
The two supports are different partitions (the quantile edges follow their own
population), so covered fractions are comparable only within one figure.

**Units.** Both sides are counted in *valid models simulated*: an AL run's
labelled set is train + validation (all of it was simulated and all of it
contributes coverage), and a pool point is one valid model. Counting attempted
models instead would divide the pool's budget by its validity rate and the AL
budget by the AL-selected points' rate, which is the higher of the two, so
valid-model units understate AL's advantage rather than flattering it. One
emcee row is one proposal, i.e. one call; the stored reference is uniformly
subsampled on load, so each retained row stands for ``--mcmc-total-rows`` over
the subsample size.

The support build is the slow part (a full pool ingest), so it is cached under
``--cache-dir`` keyed by every parameter that changes it, and the three run
sets of one target then share one build.

Usage — the three run sets of the relic-density branch:

    P=./.pixi/envs/rocm/bin/python
    $P scripts/plot_support_efficiency.py \\
        --manifest /ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv \\
        --output-dir /ptmp/jwuerzin/analysis/all_runs --run-set-label 40iter
    $P scripts/plot_support_efficiency.py \\
        --manifest /ptmp/jwuerzin/analysis/probe_extended/manifest.csv \\
        --output-dir /ptmp/jwuerzin/analysis/probe_extended \\
        --run-set-label ext160 --all-cells
    $P scripts/plot_support_efficiency.py \\
        --manifest /ptmp/jwuerzin/analysis/probe_20k/manifest.csv \\
        --output-dir /ptmp/jwuerzin/analysis/probe_20k \\
        --run-set-label probe20k --all-cells

and the exclusion-boundary branch, which has no posterior:

    $P scripts/plot_support_efficiency.py --target ExpR --model-tag expr \\
        --manifest /ptmp/jwuerzin/analysis/expr_runs/sweep_manifest.csv \\
        --baseline-data-dir /ptmp/jwuerzin/data/260804 \\
        --output-dir /ptmp/jwuerzin/analysis/expr_runs --run-set-label 40iter
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import click
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coverage_saturation import (  # noqa: E402
    AXES, RNG_SEED, _cells, _support_axis, build_target, coverage_of)
from mcmc_diagnostics import PARAM_ORDER, picks_with_tag  # noqa: E402
from pmssm.config import TARGET_CONFIG as TARGET_CONFIG_  # noqa: E402
from pmssm.data import target_validity_mask as target_validity_mask_  # noqa: E402

# Panel identity: what the dataset is called, and its line style when it appears
# as a reference curve in some other dataset's panel.
SOURCE_LABEL = {"pool": "static random scan", "mcmc": "emcee posterior",
                "atlas": "ATLAS pMSSM EWK scan"}
ATLAS_GLOB = "/viper/ptmp1/jwuerzin/pMSSM_ATLAS_EWK/scans/*/*/*.root"
EMCEE_H5_NPZ = "/ptmp/jwuerzin/analysis/all_runs/emcee_chain_ordered.npz"
SOURCE_STYLE = {"pool": (":", "black"), "mcmc": ("-.", "0.45"),
                "atlas": ((0, (5, 1, 1, 1)), "tab:brown")}
SOURCE_CURVE = {"pool": "random scan", "mcmc": "MCMC",
                "atlas": "ATLAS scan"}
# What each curve is drawn from. Nothing is held out any more: the support
# uses every in-band row and the curve walks the same dataset in order.
SOURCE_NOTE = {"pool": "all rows, scan order",
               "mcmc": "raw chains, burn-in included",
               "atlas": "all rows, scan order"}


# Which side of the tolerance band a point has to be on to count, for the
# support definition and for every curve alike. "out" gives the off-band
# (excluded-region) counterpart of the paper's figure.
BAND_WORD = {"in": "in-band", "out": "off-band"}


def _band_mask(Y, true_val, tol, band_side):
    """Boolean mask of the rows on the requested side of the tolerance band."""
    inside = np.abs(np.asarray(Y) - true_val) / true_val < tol
    return inside if band_side == "in" else ~inside


# ── support, cached ───────────────────────────────────────────────────────────

def _support(source, target, pool_dir, mcmc_dir, tol, n_bins, min_cell,
             veto, true_val, atlas_n_bins=4, band_side="in"):
    """(edges, tmap, n_target, X_axes, Y, n_rows) from the WHOLE dataset, in order.

    No half-split: the support is defined by every in-band point of the dataset,
    and the curve is drawn from the same rows in acquisition/sampling order. The
    dataset's own curve therefore reaches 1.0 at x = 1 by construction, which is
    the definition of its support rather than a result; the curve's SHAPE is the
    claim. Dropping the split also removes the question of how to hold out half
    of four asymmetric chains without biasing them.

    Order matters for the emcee reference: a prefix must be a genuine partial
    run. The four ensembles ran in parallel, so their rows are interleaved
    round-robin (row k of ensemble 0, 1, 2, 3, then row k+1, ...) rather than
    concatenated end to end, which would describe running one ensemble to
    completion before starting the next. The pool is i.i.d., so its stored order
    serves directly.

    Nothing is cached: the arrays are ~0.6 GB for the emcee and the support now
    depends on every row, so a cache would be larger than the read it saves.

    ``band_side`` picks which half of the dataset defines the support: "in" is
    the band itself, the paper's figure, and "out" is its complement, the
    OFF-band support. The off-band version is the specificity counterpart of the
    figure: it asks how much of the region the constraint EXCLUDES a run still
    visits, which is the coverage active learning is supposed to give up in
    exchange for its in-band gain. Everything else (axes, quantile grid,
    ``min_cell``, budget normalisation, the curves and the ratio panel) is
    unchanged, so the two figures are read the same way, but the off-band
    quantile edges come from the off-band population and the cells are therefore
    a different partition -- never compare a covered fraction across the two.
    """
    ax = list(AXES)
    if source == "atlas":
        # Public ATLAS pMSSM EWK scan, both campaigns merged. Same tree and
        # branch names as our own ntuples, but nested one level deeper, so it is
        # read here rather than through load_pmssm_data.
        import glob as _g

        import uproot
        cols = ax + [TARGET_CONFIG_[target]["branch"], "SP_m_h"]
        acc = {c: [] for c in cols}
        for fn in sorted(_g.glob(ATLAS_GLOB)):
            t = uproot.open(fn)["susy"]
            for c in cols:
                acc[c].append(t[c].array(library="np"))
        d = {c: np.concatenate(v) for c, v in acc.items()}
        keep, _ = target_validity_mask_(
            d[cols[-2]], d["SP_m_h"], target=target)
        keep = np.asarray(keep)
        X = np.stack([d[a][keep] for a in ax], axis=1).astype(np.float32)
        Y = np.asarray(d[cols[-2]])[keep].astype(np.float64)
    elif source == "pool":
        from pmssm.data import load_pmssm_data
        X, Y = load_pmssm_data(n_datasets=-1, data_dir=pool_dir, target=target,
                               plot_dir="/tmp", require_neutralino_lsp=veto)
        X = np.asarray(X.numpy() if hasattr(X, "numpy") else X, dtype=np.float32)
        Y = np.asarray(Y.numpy() if hasattr(Y, "numpy") else Y,
                       dtype=np.float64).ravel()
        idx = [PARAM_ORDER.index(a) for a in ax]
        X = X[:, idx]
    else:
        from pmssm.data import load_mcmc_ordered
        ens = load_mcmc_ordered(mcmc_dir, target=target, branches=ax)
        n = len(ens)
        keys = np.concatenate([np.arange(len(y), dtype=np.int64) * n + e
                               for e, (_x, y) in enumerate(ens)])
        X = np.concatenate([x for x, _y in ens])
        Y = np.concatenate([y for _x, y in ens])
        o = np.argsort(keys, kind="stable")
        del keys
        X, Y = X[o], Y[o]
        del o
    burn_rows = 0
    if source == "mcmc":
        # Cells and band stay defined by the ntuples (the converged posterior);
        # only the LINE and the budget come from the raw chains, which include
        # burn-in and the repeated rows left by rejected proposals.
        inb = _band_mask(Y, true_val, tol, band_side)
        X_def = X[inb]
        z = np.load(EMCEE_H5_NPZ)
        X, Y = z["X"], z["Y"].astype(np.float64)
        burn_rows = int(z["burn_rows"])
        click.echo(f"[eff] mcmc: curve from raw chains, {len(Y):,} proposals, "
                   f"burn-in {burn_rows:,} ({burn_rows / len(Y):.1%}); cells "
                   f"still from the ntuples")
    else:
        inb = _band_mask(Y, true_val, tol, band_side)
        X_def = X[inb]
    # The ATLAS scan has 977 in-band points against the pool's 12,343 and the
    # posterior's 24.3M, so a 12-bin grid puts every cell under min_cell and the
    # support comes out EMPTY. Coarsen the grid for that source rather than
    # relaxing the occupancy threshold: min_cell is what makes a cell a real
    # feature of the target region, and lowering it was measured on the relic
    # branch to shrink discrimination. A covered fraction is therefore not
    # comparable between this panel and the others -- state the cell count.
    nb = atlas_n_bins if source == "atlas" else n_bins
    edges = [np.quantile(X_def[:, j], np.linspace(0, 1, nb + 1))
             for j in range(len(ax))]
    for e in edges:
        e[0], e[-1] = -np.inf, np.inf
    counts = np.bincount(_cells(X_def, edges), minlength=nb ** len(ax))
    keep = np.where(counts >= min_cell)[0]
    tmap = -np.ones(nb ** len(ax), dtype=np.int64)
    tmap[keep] = np.arange(len(keep))
    click.echo(f"[eff] {source}: {len(keep)} cells of {nb ** len(ax)} from "
               f"ALL {int(inb.sum()):,} {BAND_WORD[band_side]} points of "
               f"{len(Y):,} rows")
    return edges, tmap, len(keep), X, Y, len(Y), burn_rows


# ── AL runs ──────────────────────────────────────────────────────────────────

def _al_sequence(run_dir, ax_idx, veto):
    """(X[:, axes], Y) for every simulated point, in acquisition order, or None.

    Both halves of each iteration's yield are included: the loop splits its
    newly labelled points into train and validation, and the validation points
    cost a simulator call and cover support just the same. Taking ``state["X"]``
    alone, as the saturation figure does, undercounts the budget by the
    validation fraction (20% here), which would inflate every efficiency ratio.
    """
    import torch
    s = torch.load(Path(run_dir) / "state.pt", weights_only=False,
                   map_location="cpu")
    if "X" not in s or "X_val" not in s:
        return None
    nt = [int(v) for v in (s.get("al_n_train") or [])]
    nv = [int(v) for v in (s.get("al_n_val") or [])]
    if not nt or not nv:
        return None

    def _np(t):
        return None if t is None else np.asarray(
            t.numpy() if hasattr(t, "numpy") else t)

    Xt, Yt, Ft = _np(s["X"]), _np(s["Y"]).ravel(), _np(s.get("F"))
    Xv, Yv, Fv = _np(s["X_val"]), _np(s["Y_val"]).ravel(), _np(s.get("F_val"))
    # Interleave by iteration. Within an iteration the train/val order is not
    # recorded and does not matter: the curve is read at iteration resolution.
    xs, ys, fs = [], [], []
    pt = pv = 0
    for a, b in zip(nt, nv):
        for X, Y, F, lo, hi in ((Xt, Yt, Ft, pt, a), (Xv, Yv, Fv, pv, b)):
            xs.append(X[lo:hi])
            ys.append(Y[lo:hi])
            fs.append(None if F is None else F[lo:hi])
        pt, pv = a, b
    # A run killed mid-iteration has points past the last recorded checkpoint.
    for X, Y, F, lo in ((Xt, Yt, Ft, pt), (Xv, Yv, Fv, pv)):
        if lo < len(X):
            xs.append(X[lo:])
            ys.append(Y[lo:])
            fs.append(None if F is None else F[lo:])
    X_seq = np.concatenate(xs)[:, ax_idx].astype(np.float32)
    Y_seq = np.concatenate(ys).astype(np.float64)
    if veto:
        if any(f is None for f in fs):
            raise click.ClickException(
                f"{run_dir}: --require-neutralino-lsp needs F/F_val, absent here")
        keep = np.isfinite(np.concatenate(fs)).all(axis=1)
        X_seq, Y_seq = X_seq[keep], Y_seq[keep]
    return X_seq, Y_seq


def _discover(manifest, statuses, picks, all_cells, models):
    """{(model, strategy, warm): [run_dir, ...]} from a sweep manifest."""
    rows = [r for r in csv.DictReader(open(manifest))
            if r["status"] in statuses]
    cells: dict[tuple, list[str]] = {}
    for r in rows:
        key = (r["model"], r["strategy"], r["warm_start"])
        if not all_cells and picks.get(r["model"]) != (key[1], key[2]):
            continue
        if models and r["model"] not in models:
            continue
        d = r["expected_run_dir"]
        if d in cells.setdefault(key, []):
            continue
        if not (Path(d) / "state.pt").exists():
            continue
        cells[key].append(d)
    return {k: v for k, v in cells.items() if v}


# ── curves ───────────────────────────────────────────────────────────────────

def _al_curve(seqs, tmap, n_target, n_total, n_points, lo, min_seeds=2):
    """(fraction, mean, lo, hi) over the seed replicas of one cell.

    The grid runs to the LONGEST replica, and each budget averages only the
    replicas that reached it, keeping budgets where at least ``min_seeds`` did.
    Taking the shortest replica instead lets one dead seed erase the cell: the
    ExpR TabPFN seeds spent 2000, 8583, 2727, 2992 and 6248 points, so a min
    truncated the whole curve to 2000 and it rendered as an invisible stub next
    to its own legend entry. This mirrors the trajectory plotters, which have
    always dropped iterations reported by fewer than --min-seeds seeds.
    """
    nmax = max(len(c) for c in seqs)
    if nmax <= lo:
        return None
    need = min(min_seeds, len(seqs))          # single-seed probe manifests
    grid = np.unique(np.geomspace(lo, nmax, n_points).astype(np.int64))
    ys = np.full((len(seqs), len(grid)), np.nan)
    for i, c in enumerate(seqs):
        for j, N in enumerate(grid):
            if N <= len(c):
                ys[i, j] = coverage_of(c[:N], tmap, n_target)
    keep = (~np.isnan(ys)).sum(axis=0) >= need
    if not keep.any():
        return None
    g, y = grid[keep], ys[:, keep]
    return (g / n_total, np.nanmean(y, axis=0),
            np.nanmin(y, axis=0), np.nanmax(y, axis=0))


def _ref_curve(cells, tmap, n_target, n_total, n_points, lo):
    """(fraction, coverage) for PREFIXES of the dataset in its own order.

    No permutation and no repeat-averaging: a prefix is what spending that
    budget actually buys, so there is nothing to average over. Computed from
    each support cell's FIRST occurrence, which makes coverage(N) a searchsorted
    rather than a unique() over an N-row slice at every grid point.
    """
    m = np.where(cells >= 0, tmap[np.where(cells >= 0, cells, 0)], -1)
    pos = np.flatnonzero(m >= 0)
    if len(pos) == 0:
        return np.array([]), np.array([])
    uniq, first_i = np.unique(m[pos], return_index=True)
    firsts = np.sort(pos[first_i])
    grid = np.unique(np.geomspace(max(lo, 1), len(cells), n_points).astype(np.int64))
    cov = np.searchsorted(firsts, grid, side="left") / n_target
    return grid / n_total, cov


def _speedup(f_al, c_al, f_ref, c_ref):
    """(factor, is_lower_bound): budget ratio at equal coverage.

    How much of the reference dataset would have to be generated to reach the
    coverage the AL run reached, divided by what the AL run spent. Coverage is
    monotone in budget, so a plain interpolation inverts it; when the reference
    never gets there within its own half the answer is a lower bound.
    """
    c_ref = np.maximum.accumulate(np.asarray(c_ref, dtype=float))
    f_ref = np.asarray(f_ref, dtype=float)
    ok = np.isfinite(c_ref) & np.isfinite(f_ref) & (f_ref > 0)
    c_ref, f_ref = c_ref[ok], f_ref[ok]
    if len(c_ref) < 2 or not np.isfinite(c_al) or f_al <= 0:
        return float("nan"), False
    if c_al > c_ref[-1]:
        return f_ref[-1] / f_al, True
    return float(np.exp(np.interp(c_al, c_ref, np.log(f_ref)))) / f_al, False


# ── driver ───────────────────────────────────────────────────────────────────

@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
              show_default=True)
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/all_runs", show_default=True)
@click.option("--cache-dir", default="/ptmp/jwuerzin/analysis/all_runs", show_default=True,
              help="Where the (slow) support build is cached. Share it across "
                   "run sets of the same target.")
@click.option("--baseline-data-dir", default="/ptmp/jwuerzin/data/18387358", show_default=True)
@click.option("--mcmc-data-dir", default="/ptmp/jwuerzin/data/neutralino_v4", show_default=True)
@click.option("--target", default="DMRD", show_default=True,
              help="TARGET_CONFIG key: sets the branch and the band centre.")
@click.option("--model-tag", default="", show_default=True,
              help="OUTPUT_TAG of a variant sweep (e.g. 'expr').")
@click.option("--run-set-label", default="40iter", show_default=True,
              help="Goes in the output filename, so run sets do not overwrite "
                   "one another when they share an output dir.")
@click.option("--include-status", default="completed,running,timeout,submitted",
              show_default=True)
@click.option("--models", default=None, help="Comma list of manifest model names.")
@click.option("--all-cells/--picks-only", default=False, show_default=True,
              help="--all-cells plots every (model, strategy, warm) present. "
                   "Needed for the probe manifests, whose cells are not the "
                   "canonical per-model picks.")
@click.option("--tolerance", default=0.10, show_default=True)
@click.option("--band-side", type=click.Choice(["in", "out"]), default="in",
              show_default=True,
              help="Which side of the tolerance band defines the support and "
                   "scores the curves. 'in' is the paper's figure; 'out' is the "
                   "OFF-band counterpart, i.e. coverage of the region the "
                   "constraint excludes, which is what a focused acquisition is "
                   "expected to give up. Writes a '_offband' filename so the two "
                   "coexist.")
@click.option("--n-bins", default=12, show_default=True)
@click.option("--min-cell", default=20, show_default=True)
@click.option("--mcmc-max-samples", default=500_000, show_default=True)
@click.option("--mcmc-total-rows", default=0, show_default=True,
              help="Stored reference rows before subsampling; one row is one "
                   "proposal, hence one simulator call. 0 = MEASURE it from "
                   "the files, which is the default because a hardcoded count "
                   "goes stale the moment the chains are extended (it did: a "
                   "value from 2026-08-07 was 2.0152x too small after the "
                   "2026-08-10 rewrite). Set a value only to override.")
@click.option("--n-repeats", default=8, show_default=True,
              help="Random subsets averaged per reference budget.")
@click.option("--n-points", default=26, show_default=True)
@click.option("--full-range", is_flag=True, default=False,
              help="Skip the x zoom and show every curve to its full extent, "
                   "so a reference that runs far past the AL budget (the emcee "
                   "chains reach 13x the random scan's size) is visible.")
@click.option("--only-atlas", is_flag=True, default=False,
              help="Emit ONLY the ATLAS-support panel, for a standalone "
                   "appendix figure.")
@click.option("--atlas-n-bins", default=4, show_default=True,
              help="Bins per axis for the ATLAS panel only. Its in-band "
                   "population is ~1000 points, so the main --n-bins grid "
                   "leaves every cell below --min-cell and the support is "
                   "empty.")
@click.option("--atlas/--no-atlas", default=False, show_default=True,
              help="Add a panel whose support is defined by the public "
                   "ATLAS pMSSM EWK scan (both campaigns merged).")
@click.option("--min-seeds", default=2, show_default=True,
              help="A budget is plotted when at least this many replicas of the "
                   "cell reached it; capped at the number of replicas present, "
                   "so single-seed probe manifests still draw.")
@click.option("--anchor-min-frac", default=0.4, show_default=True,
              help="A curve whose final budget is below this fraction of the "
                   "longest run's is excluded from setting the common budget "
                   "the headline factors are read at. It is still drawn, and "
                   "gets a factor read at its own endpoint, marked with a "
                   "dagger. Stops one timed-out cell from pulling the anchor "
                   "back to where no model has separated yet.")
@click.option("--require-neutralino-lsp/--no-require-neutralino-lsp",
              default=False, show_default=True)
def main(manifest, output_dir, cache_dir, baseline_data_dir, mcmc_data_dir,
         target, model_tag, run_set_label, include_status, models, all_cells,
         tolerance, band_side, n_bins, min_cell, mcmc_max_samples, mcmc_total_rows,
         n_repeats, n_points, full_range, only_atlas, atlas_n_bins, atlas, min_seeds,
         anchor_min_frac,
         require_neutralino_lsp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_hit_rate_trajectories_multiseed as phr
    from pmssm.config import TARGET_CONFIG

    true_val = float(TARGET_CONFIG[target]["true_value"])
    has_mcmc = bool(TARGET_CONFIG[target].get("has_mcmc_reference", False))
    sources = ["pool"] + (["mcmc"] if has_mcmc else [])
    if atlas or only_atlas:
        sources.append("atlas")
    # The ATLAS scan is heavily PRESELECTED, so its row count is not a generation
    # budget and a curve of its own coverage against its own size means nothing.
    # Its panel therefore borrows the random scan's denominator, uses the random
    # scan as its reference, and draws no ATLAS curve at all.
    panels = ["atlas"] if only_atlas else [s_ for s_ in sources if s_ != "atlas"]
    if atlas and not only_atlas:
        panels.append("atlas")
    ax_idx = [PARAM_ORDER.index(a) for a in AXES]
    rng = np.random.default_rng(RNG_SEED)
    band_word = BAND_WORD[band_side]
    click.echo(f"[eff] target={target} support={band_word} "
               f"(|y-{true_val:g}|/{true_val:g} "
               f"{'<' if band_side == 'in' else '>='} {tolerance:g})  "
               f"datasets={sources}  run set={run_set_label}")

    # ── the AL runs ──────────────────────────────────────────────────────────
    statuses = {s.strip() for s in include_status.split(",")}
    want = {m.strip() for m in models.split(",")} if models else None
    cells = _discover(manifest, statuses, picks_with_tag(model_tag), all_cells,
                      want)
    if not cells:
        raise click.ClickException(f"no usable runs in {manifest}")
    # Only label a cell with its strategy when its model has more than one, so
    # the common case stays a plain model name.
    n_per_model: dict[str, int] = {}
    for m, _s, _w in cells:
        n_per_model[m] = n_per_model.get(m, 0) + 1
    runs: dict[tuple, list] = {}
    for key, dirs in sorted(cells.items()):
        seqs = []
        for d in dirs:
            try:
                got = _al_sequence(d, ax_idx, require_neutralino_lsp)
            except Exception as exc:                        # noqa: BLE001
                click.echo(f"[eff]   skip {Path(d).name}: "
                           f"{type(exc).__name__}: {exc}", err=True)
                continue
            if got is not None:
                seqs.append(got)
        if not seqs:
            continue
        runs[key] = seqs
        click.echo(f"[eff] {'/'.join(key):<48} {len(seqs)} replica(s), "
                   f"points spent {[len(y) for _x, y in seqs]}")

    # 0 means measure, so the emcee denominator tracks the files instead of a
    # constant that goes stale when the chains grow.
    if False:   # --mcmc-total-rows no longer scales anything: 1 row = 1 call
        from pmssm.data import count_mcmc_rows  # noqa: PLC0415
        if require_neutralino_lsp:
            raise click.UsageError(
                "--require-neutralino-lsp changes how many reference rows load, "
                "so a measured total would not match; pass --mcmc-total-rows "
                "explicitly for a vetoed run")
        mcmc_total_rows = count_mcmc_rows(mcmc_data_dir, target=target)
        click.echo(f"[eff] measured emcee stored rows: {mcmc_total_rows:,}")

    # ── panels ───────────────────────────────────────────────────────────────
    supports = {}
    for src in sources:
        edges, tmap, n_tgt, oX, oY, n_rows, burn_rows = _support(
            src, target, baseline_data_dir, mcmc_data_dir, tolerance, n_bins,
            min_cell, require_neutralino_lsp, true_val,
            atlas_n_bins=atlas_n_bins, band_side=band_side)
        if n_tgt == 0:
            click.echo(f"[eff] {src}: support is EMPTY at these settings; "
                       f"panel skipped", err=True)
            continue
        # Every row is one simulator call now: no subsample to undo, so the
        # deflation that used to represent N proposals by N/70.5 sampled rows
        # (and thereby cancelled the posterior's in-band enrichment) is gone.
        supports[src] = dict(edges=edges, tmap=tmap, n_target=n_tgt, held_X=oX,
                             held_Y=oY, n_rows=n_rows, per_entry=1.0,
                             n_total=(supports["pool"]["n_total"]
                                      if src == "atlas" and "pool" in supports
                                      else n_rows),
                             burn_rows=burn_rows)
        click.echo(f"[eff] {src:<5} support {n_tgt} cells | dataset "
                   f"{n_rows:,} rows = {n_rows:,} calls (1 row = 1 call)")

    out = {"config": {"target": target, "run_set": run_set_label, "axes": AXES,
                      "tolerance": tolerance, "band_side": band_side,
                      "support": f"{band_word} support", "n_bins": n_bins,
                      "min_cell": min_cell, "manifest": manifest,
                      "budget_unit": "valid models simulated (AL: train+val)"},
           "panels": {}}

    fig, axes = plt.subplots(2, len(panels), squeeze=False, sharex="col",
                             figsize=(6.6 * len(panels), 6.9),
                             gridspec_kw={"height_ratios": [3.0, 1.2],
                                          "hspace": 0.06})

    def _ratio_to_own(f, c, own_f, own_c, floor):
        """(f, c / own_c(f)) over the budgets where the reference is defined.

        Interpolated in log budget, and restricted to the reference's own
        support in x: extrapolating np.interp would clamp to the reference's
        first value and manufacture a ratio far from 1 where the reference has
        no measurement at all. ``floor`` drops the leftmost budgets, where the
        reference has found only a cell or two and the ratio is dominated by
        the denominator's own discreteness rather than by either method.
        """
        f, c = np.asarray(f, float), np.asarray(c, float)
        ok = (f >= own_f[0]) & (f <= own_f[-1])
        den = np.interp(np.log(f[ok]), np.log(own_f), own_c)
        good = den > floor
        return f[ok][good], c[ok][good] / den[good]

    for col_i, src in enumerate(panels):
        ax, axr = axes[0][col_i], axes[1][col_i]
        S = supports[src]
        rec = {"n_target_cells": int(S["n_target"]),
               "dataset_rows": int(S["n_rows"]),
               "dataset_calls": int(S["n_total"]), "al": {}, "reference": {}}
        # Only guard against dividing by an essentially empty denominator. A
        # floor of a cell or two reads as harmless but on the coarse 27-cell
        # support it is 4 to 7% coverage, which clips the reference ratios over
        # their whole first decade; the curves are means over --n-repeats
        # subsets, so they are smooth well below that.
        r_floor = 0.25 / max(1, S["n_target"])

        # Reference curves. Every static dataset appears in every panel; each is
        # drawn from its own full dataset in order.
        ref_curves = {}
        own_src = "pool" if src == "atlas" else src
        for rsrc in [r for r in sources if r != "atlas"]:
            R = supports[rsrc]
            rcells = np.where(
                _band_mask(R["held_Y"], true_val, tolerance, band_side),
                _cells(R["held_X"], S["edges"]), -1)
            # The grid's floor is set in CALLS, not rows, so a subsampled
            # reference still starts at the same budget as everything else.
            # Otherwise the emcee curve begins 35x further right than the pool's
            # and the ratio panel loses the AL curves' first decade.
            f, c = _ref_curve(rcells, S["tmap"], S["n_target"], S["n_total"],
                              n_points, 200)
            ref_curves[rsrc] = (f, c)
            ls, col = SOURCE_STYLE[rsrc]
            ax.plot(f, c, ls=ls, color=col, lw=1.7,
                    label=f"{SOURCE_CURVE[rsrc]} ({SOURCE_NOTE[rsrc]})")
            rec["reference"][rsrc] = {"fraction": f.tolist(),
                                      "coverage": c.tolist()}
            if rsrc == own_src:
                own_f, own_c = f, c
        # The panel's own dataset is the unity line by construction, so only the
        # other dataset's ratio is worth a curve here.
        for rsrc, (f, c) in ref_curves.items():
            if rsrc == own_src:
                continue
            ls, col = SOURCE_STYLE[rsrc]
            rf, rr = _ratio_to_own(f, c, own_f, own_c, r_floor)
            axr.plot(rf, rr, ls=ls, color=col, lw=1.6)
            rec["reference"][rsrc]["ratio_fraction"] = rf.tolist()
            rec["reference"][rsrc]["ratio_to_own"] = rr.tolist()

        # AL curves. Computed first, plotted second: the cross-model summary is
        # read at the coverage EVERY curve in the panel reaches, which is not
        # known until they are all in hand.
        curves = {}
        for key, seqs in runs.items():
            seq_cells = [np.where(_band_mask(Y, true_val, tolerance, band_side),
                                  _cells(X, S["edges"]), -1)
                         for X, Y in seqs]
            got = _al_curve(seq_cells, S["tmap"], S["n_target"], S["n_total"],
                            n_points, 2000, min_seeds=min_seeds)
            if got is not None:
                curves[key] = got
        # A ratio read at each curve's own endpoint answers "what did this run
        # buy?", but it is not comparable between curves that stopped at
        # different coverages, and it inflates without limit as the reference
        # approaches its ceiling (the last of 27 coarse cells is very expensive
        # to find at random). So the headline number is read at the largest
        # budget EVERY curve in the panel reaches, which is the convention the
        # saturation figure already uses; the endpoint ratio is kept alongside.
        # The anchor is the largest budget every curve reaches, but a run that
        # died early must not drag it down to where nothing has separated yet:
        # TabPFN's ExpR cell timed out at 2,442 points, which pulled the anchor
        # to 0.14% of the dataset and made every factor read 1.0. Curves shorter
        # than --anchor-min-frac of the longest are excluded from setting it
        # (they are still drawn, and still get a factor, read at the anchor if
        # they reach it and at their own endpoint otherwise).
        ends = {k: f[-1] for k, (f, *_ ) in curves.items()}
        longest = max(ends.values()) if ends else 0.0
        anchoring = [v for v in ends.values() if v >= anchor_min_frac * longest]
        f_common = min(anchoring) if anchoring else (min(ends.values()) if ends else 0.0)
        short = {k for k, v in ends.items() if v < anchor_min_frac * longest}
        if short:
            click.echo(f"[eff] excluded from the anchor (shorter than "
                       f"{anchor_min_frac:.0%} of the longest run): "
                       f"{', '.join('/'.join(k) for k in sorted(short))}")
        # Pull the anchor below any curve that has already reached full coverage
        # there: at the ceiling the ratio is decided entirely by how long random
        # sampling takes to stumble on the single hardest cell, which is a
        # property of the support's coarseness and not of the acquisition.
        f_sat = [f[int(np.argmax(mu >= 0.999))]
                 for f, mu, *_ in curves.values() if mu.max() >= 0.999]
        if f_sat:
            f_common = min(f_common, 0.95 * min(f_sat))
        any_at_own = False
        rows_txt = [f"    {'':<34} common budget for the headline column: "
                    f"{f_common * 100:.4f}% of the dataset "
                    f"({int(round(f_common * S['n_total'])):,} calls)"]
        for key, (f, mu, lo_b, hi_b) in curves.items():
            model, strat, warm = key
            fac, bound = _speedup(f[-1], mu[-1], own_f, own_c)
            # A curve excluded from the anchor may not reach it. np.interp would
            # clamp to its final coverage and credit it with a budget it never
            # spent, which understates it; read those at their own endpoint and
            # mark them instead.
            at_own = f[-1] < f_common
            f_read = f[-1] if at_own else f_common
            c_at_common = mu[-1] if at_own else float(np.interp(f_common, f, mu))
            fac_c, bound_c = _speedup(f_read, c_at_common, own_f, own_c)
            any_at_own = any_at_own or at_own
            base = phr.MODEL_DISPLAY.get(model, model)
            lbl = base if n_per_model[model] == 1 else f"{base} ({strat}/{warm})"
            tag = ("" if not np.isfinite(fac_c) else
                   f"  {'≥' if bound_c else ''}×{fac_c:.1f}"
                   if fac_c < 10 else f"  {'≥' if bound_c else ''}×{fac_c:.0f}")
            if tag and at_own:
                tag += "†"
            col = phr.MODEL_COLORS.get(model, "gray")
            ax.plot(f, mu, "-", color=col, lw=1.9, label=lbl + tag)
            if len(runs[key]) > 1:
                ax.fill_between(f, lo_b, hi_b, color=col, alpha=0.15, lw=0)
            rf, rr = _ratio_to_own(f, mu, own_f, own_c, r_floor)
            axr.plot(rf, rr, "-", color=col, lw=1.8)
            if len(runs[key]) > 1:
                _, rlo = _ratio_to_own(f, lo_b, own_f, own_c, r_floor)
                _, rhi = _ratio_to_own(f, hi_b, own_f, own_c, r_floor)
                axr.fill_between(rf, rlo, rhi, color=col, alpha=0.15, lw=0)
            ref_at_al = float(np.interp(f[-1], own_f, own_c))
            rec["al"]["/".join(key)] = {
                "n_replicas": len(runs[key]), "fraction": f.tolist(),
                "coverage": mu.tolist(), "coverage_min": lo_b.tolist(),
                "coverage_max": hi_b.tolist(),
                "ratio_fraction": rf.tolist(), "ratio_to_own": rr.tolist(),
                "ratio_at_common_budget": float(np.interp(f_common, rf, rr))
                if len(rf) else float("nan"),
                "final_fraction": float(f[-1]), "final_coverage": float(mu[-1]),
                "final_calls": int(round(f[-1] * S["n_total"])),
                "reference_coverage_at_final_fraction": ref_at_al,
                "reference_saturated_at_final_fraction": bool(ref_at_al >= 0.999),
                "budget_speedup_at_final_coverage": float(fac),
                "speedup_at_final_is_lower_bound": bool(bound),
                "common_budget_fraction": float(f_common),
                "coverage_at_common_budget": c_at_common,
                "budget_speedup_at_common_budget": float(fac_c),
                "read_at_own_endpoint_not_anchor": bool(at_own),
                "speedup_at_common_is_lower_bound": bool(bound_c)}
            rows_txt.append(f"    {lbl:<34} cov {mu[-1]:.3f} at "
                            f"{f[-1] * 100:8.4f}% of the dataset "
                            f"({int(round(f[-1] * S['n_total'])):>9,} calls) | "
                            f"{SOURCE_CURVE[src]} there {ref_at_al:.3f}"
                            f"{' (SATURATED)' if ref_at_al >= 0.999 else ''} | "
                            f"budget x{fac:.1f} at own endpoint, "
                            f"x{fac_c:.1f} at common budget (cov "
                            f"{c_at_common:.3f})")
        rec["common_budget_fraction"] = float(f_common)

        ax.set_xscale("log")
        # Zoom to the decades that carry the comparison. The references run from
        # 200 points and, in the emcee panel, out to several times the other
        # dataset's cost; plotted in full that is mostly empty canvas and it
        # squeezes the AL curves into one decade. Keep from just below the AL
        # start to just past where the panel's own dataset matches the best AL
        # coverage, which is exactly the span the horizontal gap is read across.
        if curves and not full_range:
            al_lo = min(f[0] for f, *_ in curves.values())
            al_hi = max(f[-1] for f, *_ in curves.values())
            c_max = max(mu[-1] for _f, mu, *_ in curves.values())
            f_need, _b = _speedup(1.0, c_max, own_f, own_c)  # ratio at f_al = 1
            if not np.isfinite(f_need):
                f_need = al_hi
            ax.set_xlim(al_lo / 3.0, max(al_hi, f_need) * 2.5)
        # x = 1 is the whole dataset; x = 0.5 is where its own half completes
        # the support, which is the ceiling this metric is defined against.
        # Only annotate them when the zoom actually reaches that far.
        x_lo, x_hi = ax.get_xlim()
        for xv, txt in ((1.0, "whole dataset"),):
            if not (x_lo < xv < x_hi):
                continue
            for a in (ax, axr):
                a.axvline(xv, color="0.75", ls=(0, (1, 3)), lw=1.0, zorder=0)
            ax.annotate(txt, xy=(xv, 0.015), xytext=(-4, 0),
                        textcoords="offset points", rotation=90, ha="right",
                        va="bottom", fontsize=6.5, color="0.45")
        # Burn-in marker, in EVERY panel where that dataset appears and with
        # that panel's own normalisation: 0.27 of the emcee budget is the same
        # boundary as 9.8x the random scan's size. Inside the x range it is a
        # dashed vertical; outside it (which the zoom makes the usual case, in
        # both panels) it becomes a horizontal arrow off the right edge. The
        # previous code skipped the panel's OWN dataset, so the emcee panel drew
        # its line at 0.27 outside the limits and it never appeared.
        # Only the panel whose OWN dataset burned in gets the marker: in the
        # random-scan panel the same boundary sits at 9.8x its size, which is
        # true but reads as clutter next to a curve nobody is asked to budget
        # against.
        for bsrc in ([src] if src in sources else []):
            B = supports.get(bsrc, {})
            if not B.get("burn_rows"):
                continue
            xb = B["burn_rows"] / S["n_total"]
            col_b = SOURCE_STYLE[bsrc][1]
            if x_lo < xb < x_hi:
                for a_ in (ax, axr):
                    a_.axvline(xb, color=col_b, ls="--", lw=1.2, zorder=0)
                ax.annotate(f"{SOURCE_CURVE[bsrc]} burn-in ends",
                            xy=(xb, 0.015), xytext=(3, 0),
                            textcoords="offset points", rotation=90,
                            ha="left", va="bottom", fontsize=7, color=col_b)
            else:
                # Top-right corner: the legend sits upper-left and the curves
                # rise from the lower left, so this is the one empty region.
                # Text on the same line, to the left of the arrow.
                # Text centred directly ABOVE the arrow's line, both spanning
                # the same x range, so they read as one label.
                y0, x0, x1 = 0.93, 0.66, 0.995
                ax.annotate("", xy=(x1, y0), xytext=(x0, y0),
                            xycoords="axes fraction", textcoords="axes fraction",
                            arrowprops=dict(arrowstyle="-|>", lw=1.7, color=col_b))
                ax.text((x0 + x1) / 2 - 0.035, y0 + 0.012,
                        f"{SOURCE_CURVE[bsrc]} burn-in ends at $x = {xb:.2g}$",
                        transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=8, color=col_b)
        ax.set_ylabel(f"fraction of the {SOURCE_LABEL[src]}'s {band_word} "
                      f"support covered\n({S['n_target']} cells)")
        _support_axis(ax)
        ax.grid(alpha=0.3, which="both")
        ax.tick_params(axis="x", labelbottom=False)
        ax.legend(fontsize=7.5, loc="upper left",
                  title=f"×N: the {SOURCE_LABEL[src]} needs N times the\nbudget "
                        f"for the same coverage, read at the\ncommon budget of "
                        f"{f_common:.2%} of the dataset"
                        + ("\n† run too short to reach it: read at its own end"
                           if any_at_own else ""),
                  title_fontsize=6.5)

        # ── ratio panel: coverage relative to the dataset's own curve ─────────
        # Read at equal budget, so >1 means this method covers more of the
        # support per simulator call than the process that produced the dataset,
        # and <1 means generating more of the dataset would have been the better
        # spend. The dataset's own curve is 1 by construction.
        axr.axhline(1.0, color=SOURCE_STYLE[own_src][1], lw=1.4,
                    ls=SOURCE_STYLE[own_src][0], zorder=1)
        axr.grid(alpha=0.3, which="both")
        axr.set_xlabel(f"points spent / size of the {SOURCE_LABEL[own_src]}\n"
                       f"({S['n_total']:,} valid models simulated)")
        axr.set_ylabel(f"/ {SOURCE_CURVE[own_src]}", fontsize=8.5)
        # Only data inside the zoom sets the range: the other dataset's ratio
        # runs well past the right edge and would otherwise dictate the scale.
        vis = [v for ln in axr.get_lines()
               for x, v in zip(np.atleast_1d(ln.get_xdata()),
                               np.atleast_1d(ln.get_ydata()))
               if np.isfinite(v) and x_lo <= x <= x_hi]
        v_lo, v_hi = (min(vis + [1.0]), max(vis + [1.0])) if vis else (0.5, 1.5)
        pad = 0.06 * max(v_hi - v_lo, 0.2)
        axr.set_ylim(max(0.0, v_lo - pad), v_hi + pad)
        axr.tick_params(axis="y", labelsize=8)
        out["panels"][src] = rec
        click.echo(f"[eff] panel {src}: {S['n_target']} cells")
        for t in rows_txt:
            click.echo(t)

    fig.tight_layout(h_pad=0.4)
    stem = (f"support_efficiency_{run_set_label}"
            + ("_offband" if band_side == "out" else ""))
    p_png = Path(output_dir) / f"{stem}.png"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(p_png, dpi=200)
    (Path(output_dir) / f"{stem}.json").write_text(json.dumps(out, indent=1))
    click.echo(f"[eff] wrote {p_png} and {stem}.json")


if __name__ == "__main__":
    main()
