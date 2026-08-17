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

  * x = 1     is "you have spent what the whole dataset cost to generate"
  * x = 0.5   is where the dataset's own held-out half completes its support,
              by construction of the half-split (that is the ceiling check)
  * a curve to the LEFT of the dataset's own curve at the same height is
    cheaper than generating the dataset; to the right, more expensive.

Curves drawn in every panel:

  AL, per model   the accumulated labelled set in acquisition order, averaged
                  over seed replicas (band = min/max over seeds)
  random scan     the static random pool's held-out half, random subsets
  MCMC            the emcee reference's held-out half, random subsets
                  (only where a posterior exists for the target)

The dataset that defines a panel's support is always scored on its held-out
half, so no population is scored against cells it helped define.

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

# Panel identity: what the dataset is called, and its line style when it appears
# as a reference curve in some other dataset's panel.
SOURCE_LABEL = {"pool": "static random scan", "mcmc": "emcee posterior"}
SOURCE_STYLE = {"pool": (":", "black"), "mcmc": ("-.", "0.45")}
SOURCE_CURVE = {"pool": "random scan", "mcmc": "MCMC"}


# ── support, cached ───────────────────────────────────────────────────────────

def _support(source, target, pool_dir, mcmc_dir, tol, n_bins, min_cell,
             mcmc_max_samples, veto, cache_dir, true_val):
    """(edges, tmap, n_target, held_X, held_Y, n_rows) for one dataset.

    ``build_target`` is used verbatim so this figure's support is the same object
    the saturation and compute figures score against; only the budget axis is
    new. ``held_X`` / ``held_Y`` are the half NOT used to define the cells.
    """
    key = (f"support_efficiency_ref_{target}_{source}_tol{tol:g}"
           f"_b{n_bins}_m{min_cell}_v{int(veto)}_s{mcmc_max_samples}")
    p = Path(cache_dir) / f"{key}.npz"
    if p.exists():
        z = np.load(p)
        edges = [z[f"e{j}"] for j in range(len(AXES))]
        click.echo(f"[eff] {source}: support from cache {p.name}")
        return (edges, z["tmap"], int(z["n_target"]), z["held_X"], z["held_Y"],
                int(z["n_rows"]))
    ax_idx = [PARAM_ORDER.index(a) for a in AXES]
    # A fresh RNG at the module seed reproduces coverage_saturation's split
    # exactly, so the cells and the held-out half match that figure's.
    rng = np.random.default_rng(RNG_SEED)
    edges, tmap, n_target, (X_b, Y_b) = build_target(
        mcmc_dir, ax_idx, n_bins, min_cell, tol, mcmc_max_samples, veto, rng,
        true_val=None, target=target, support_source=source,
        pool_data_dir=pool_dir)
    X_b = np.asarray(X_b, dtype=np.float32)
    Y_b = np.asarray(Y_b, dtype=np.float64).ravel()
    n_rows = 2 * len(X_b)          # half = N//2, so this is N to within one row
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, tmap=tmap, n_target=np.int64(n_target), held_X=X_b,
                        held_Y=Y_b, n_rows=np.int64(n_rows),
                        **{f"e{j}": e for j, e in enumerate(edges)})
    click.echo(f"[eff] {source}: {n_target} cells of {n_bins ** len(AXES)}, "
               f"{n_rows:,} rows, cached to {p.name}")
    return edges, tmap, n_target, X_b, Y_b, n_rows


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


def _ref_curve(cells, tmap, n_target, n_total, calls_per_entry, n_repeats,
               rng, n_points, lo):
    """(fraction, mean coverage) for random subsets of a static dataset.

    Random subsets are taken as prefixes of a fresh permutation, which is the
    same distribution as sampling without replacement but costs one shuffle per
    repeat instead of one per budget.
    """
    grid = np.unique(np.geomspace(max(lo, 1), len(cells), n_points).astype(np.int64))
    acc = np.zeros(len(grid))
    for _ in range(n_repeats):
        c = cells[rng.permutation(len(cells))]
        acc += [coverage_of(c[:N], tmap, n_target) for N in grid]
    return grid * calls_per_entry / n_total, acc / n_repeats


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
@click.option("--n-bins", default=12, show_default=True)
@click.option("--min-cell", default=20, show_default=True)
@click.option("--mcmc-max-samples", default=500_000, show_default=True)
@click.option("--mcmc-total-rows", default=17_498_112, show_default=True,
              help="Stored reference rows before subsampling; one row is one "
                   "proposal, hence one simulator call.")
@click.option("--n-repeats", default=8, show_default=True,
              help="Random subsets averaged per reference budget.")
@click.option("--n-points", default=26, show_default=True)
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
         tolerance, n_bins, min_cell, mcmc_max_samples, mcmc_total_rows,
         n_repeats, n_points, min_seeds, anchor_min_frac,
         require_neutralino_lsp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_hit_rate_trajectories_multiseed as phr
    from pmssm.config import TARGET_CONFIG

    true_val = float(TARGET_CONFIG[target]["true_value"])
    has_mcmc = bool(TARGET_CONFIG[target].get("has_mcmc_reference", False))
    sources = ["pool"] + (["mcmc"] if has_mcmc else [])
    ax_idx = [PARAM_ORDER.index(a) for a in AXES]
    rng = np.random.default_rng(RNG_SEED)
    click.echo(f"[eff] target={target} band=|y-{true_val:g}|/{true_val:g} < "
               f"{tolerance:g}  datasets={sources}  run set={run_set_label}")

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

    # ── panels ───────────────────────────────────────────────────────────────
    supports = {}
    for src in sources:
        edges, tmap, n_tgt, hX, hY, n_rows = _support(
            src, target, baseline_data_dir, mcmc_data_dir, tolerance, n_bins,
            min_cell, mcmc_max_samples, require_neutralino_lsp, cache_dir,
            true_val)
        # A pool row is one call. A reference row stands for the subsampling
        # factor, so its budget in calls is deflated the same way
        # coverage_saturation deflates it.
        per_entry = (1.0 if src == "pool"
                     else max(1.0, mcmc_total_rows / max(1, n_rows)))
        n_calls = n_rows if src == "pool" else int(round(n_rows * per_entry))
        supports[src] = dict(edges=edges, tmap=tmap, n_target=n_tgt, held_X=hX,
                             held_Y=hY, n_rows=n_rows, per_entry=per_entry,
                             n_total=n_calls)
        click.echo(f"[eff] {src:<5} support {n_tgt} cells | dataset "
                   f"{n_rows:,} rows = {n_calls:,} calls "
                   f"(1 row = {per_entry:.1f} calls)")

    out = {"config": {"target": target, "run_set": run_set_label, "axes": AXES,
                      "tolerance": tolerance, "n_bins": n_bins,
                      "min_cell": min_cell, "manifest": manifest,
                      "budget_unit": "valid models simulated (AL: train+val)"},
           "panels": {}}

    fig, axes = plt.subplots(2, len(sources), squeeze=False, sharex="col",
                             figsize=(6.6 * len(sources), 6.9),
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

    for col_i, src in enumerate(sources):
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
        # scored on its held-out half, so the panel's own dataset is never
        # scored against the cells it defined.
        ref_curves = {}
        for rsrc in sources:
            R = supports[rsrc]
            rcells = np.where(
                np.abs(R["held_Y"] - true_val) / true_val < tolerance,
                _cells(R["held_X"], S["edges"]), -1)
            # The grid's floor is set in CALLS, not rows, so a subsampled
            # reference still starts at the same budget as everything else.
            # Otherwise the emcee curve begins 35x further right than the pool's
            # and the ratio panel loses the AL curves' first decade.
            f, c = _ref_curve(rcells, S["tmap"], S["n_target"], S["n_total"],
                              R["per_entry"], n_repeats, rng, n_points,
                              max(4, int(round(200 / R["per_entry"]))))
            ref_curves[rsrc] = (f, c)
            ls, col = SOURCE_STYLE[rsrc]
            ax.plot(f, c, ls, color=col, lw=1.7,
                    label=f"{SOURCE_CURVE[rsrc]} (held-out half)")
            rec["reference"][rsrc] = {"fraction": f.tolist(),
                                      "coverage": c.tolist()}
            if rsrc == src:
                own_f, own_c = f, c
        # The panel's own dataset is the unity line by construction, so only the
        # other dataset's ratio is worth a curve here.
        for rsrc, (f, c) in ref_curves.items():
            if rsrc == src:
                continue
            ls, col = SOURCE_STYLE[rsrc]
            rf, rr = _ratio_to_own(f, c, own_f, own_c, r_floor)
            axr.plot(rf, rr, ls, color=col, lw=1.6)
            rec["reference"][rsrc]["ratio_fraction"] = rf.tolist()
            rec["reference"][rsrc]["ratio_to_own"] = rr.tolist()

        # AL curves. Computed first, plotted second: the cross-model summary is
        # read at the coverage EVERY curve in the panel reaches, which is not
        # known until they are all in hand.
        curves = {}
        for key, seqs in runs.items():
            seq_cells = [np.where(np.abs(Y - true_val) / true_val < tolerance,
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
        if curves:
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
        for xv, txt in ((0.5, "own held-out half"), (1.0, "whole dataset")):
            if not (x_lo < xv < x_hi):
                continue
            for a in (ax, axr):
                a.axvline(xv, color="0.75", ls=(0, (1, 3)), lw=1.0, zorder=0)
            ax.annotate(txt, xy=(xv, 0.015), xytext=(-4, 0),
                        textcoords="offset points", rotation=90, ha="right",
                        va="bottom", fontsize=6.5, color="0.45")
        ax.set_ylabel(f"fraction of the {SOURCE_LABEL[src]}'s in-band support "
                      f"covered\n({S['n_target']} cells)")
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
        axr.axhline(1.0, color=SOURCE_STYLE[src][1], lw=1.4,
                    ls=SOURCE_STYLE[src][0], zorder=1)
        axr.grid(alpha=0.3, which="both")
        axr.set_xlabel(f"points spent / size of the {SOURCE_LABEL[src]}\n"
                       f"({S['n_total']:,} valid models simulated)")
        axr.set_ylabel(f"/ {SOURCE_CURVE[src]}", fontsize=8.5)
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
    stem = f"support_efficiency_{run_set_label}"
    p_png = Path(output_dir) / f"{stem}.png"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(p_png, dpi=200)
    (Path(output_dir) / f"{stem}.json").write_text(json.dumps(out, indent=1))
    click.echo(f"[eff] wrote {p_png} and {stem}.json")


if __name__ == "__main__":
    main()
