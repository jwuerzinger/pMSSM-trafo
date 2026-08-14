"""Union-coverage saturation: does one AL run cover what several do?

The rank-uniformity check asks whether seed replicas realise the same
acquisition distribution. That is the wrong question for an AL loop, which has
no target distribution to be right about (see the paper's Sect. 6.3). The
question it gestures at is answerable directly: at a FIXED simulator budget,
does splitting the budget across independent replicas cover more of the
viable region than spending it all in one run?

Coverage is measured against an external, method-independent target. The
emcee reference is split in half: the first half defines the target support
(equal-occupancy bins per axis over the informative subspace, cells holding
at least ``--min-cell`` in-band reference points), the second half supplies
the MCMC curve, so no method is scored against its own points. For every
method we then count the fraction of target cells hit by N in-band points:

  * AL, k replicas   — first N/k acquired points of each of k replicas
                       (averaged over which replicas, so k is not confounded
                       with a lucky seed)
  * random scan      — N points drawn from the valid pool
  * MCMC             — N points from the held-out reference half

A curve that saturates in N and does not depend on k says one run suffices.
A curve that keeps rising with k at fixed N says single-run coverage is a
strict subset of what the same budget buys when diversified.

Usage:
    python scripts/coverage_saturation.py --require-neutralino-lsp
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

from mcmc_diagnostics import (  # noqa: E402
    DEFAULT_AL_PICKS, MODEL_DISPLAY, PARAM_ORDER, picks_with_tag)

AXES = ["IN_M_1", "IN_M_2", "IN_mu"]   # the axes carrying the constraint's information
RNG_SEED = 20260807


def _cells(X: np.ndarray, edges: list[np.ndarray]) -> np.ndarray:
    """Flat cell index per row for the given per-axis bin edges."""
    nb = len(edges[0]) - 1
    idx = np.zeros(len(X), dtype=np.int64)
    for j, e in enumerate(edges):
        b = np.clip(np.digitize(X[:, j], e[1:-1]), 0, nb - 1)
        idx = idx * nb + b
    return idx



def build_target(mcmc_data_dir: str, ax_idx: list[int], n_bins: int, min_cell: int,
                 tolerance: float, max_samples: int, veto: bool, rng, true_val=0.12,
                 target: str = "DMRD", support_source: str = "mcmc",
                 pool_data_dir: str = ""):
    """Target support from half the reference: (edges, tmap, n_target, held_out).

    Shared with ``plot_compute_vs_dataset.py`` so both figures score coverage
    against the identical support definition.

    ``support_source`` selects the reference population that defines the
    support: "mcmc" (the emcee posterior, the paper's original definition) or
    "pool" (the random-scan dataset, available for every target). Binning,
    tolerance, half-split and ``min_cell`` are shared, so the two are
    like-for-like and differ only in what they call the target region.

    With "mcmc" the support only exists for a target that HAS a posterior. Passing another target used to be silently accepted
    because this call hardcoded "DMRD": the cells then came from the relic
    density's posterior while the run points were selected by the other
    target's band, producing a covered-fraction that mixes two unrelated
    observables. Refuse instead.
    """
    from pmssm.config import TARGET_CONFIG
    from pmssm.data import load_mcmc_data, load_pmssm_data
    if true_val is None:
        true_val = float(TARGET_CONFIG[target]["true_value"])

    if support_source == "pool":
        # Support from the random-scan pool instead of a posterior. Everything
        # downstream (in-band cut, half split, quantile edges, min_cell) is
        # identical, so a pool-sourced panel is directly comparable to the
        # MCMC-sourced one: only the reference population differs. This is the
        # only support definition available for a target with no posterior.
        if not pool_data_dir:
            raise ValueError("support_source='pool' requires pool_data_dir")
        Xm, Ym = load_pmssm_data(n_datasets=-1, data_dir=pool_data_dir,
                                 target=target, plot_dir="/tmp",
                                 require_neutralino_lsp=veto)
    elif support_source == "mcmc":
        if not TARGET_CONFIG[target].get("has_mcmc_reference", False):
            raise ValueError(
                f"in-band support from an emcee reference requires a target "
                f"that has one; {target!r} does not. Use "
                f"support_source='pool'."
            )
        Xm, Ym = load_mcmc_data(data_dir=mcmc_data_dir, target=target,
                                require_neutralino_lsp=veto,
                                max_samples=max_samples)
    else:
        raise ValueError(f"unknown support_source {support_source!r}")
    Xm = np.asarray(Xm.numpy() if hasattr(Xm, "numpy") else Xm)[:, ax_idx]
    Ym = np.asarray(Ym.numpy() if hasattr(Ym, "numpy") else Ym).ravel()
    perm = rng.permutation(len(Xm))
    half = len(perm) // 2
    X_a, Y_a = Xm[perm[:half]], Ym[perm[:half]]
    X_b, Y_b = Xm[perm[half:]], Ym[perm[half:]]
    X_def = X_a[np.abs(Y_a - true_val) / true_val < tolerance]
    edges = [np.quantile(X_def[:, j], np.linspace(0, 1, n_bins + 1))
             for j in range(len(ax_idx))]
    for e in edges:
        e[0], e[-1] = -np.inf, np.inf
    counts = np.bincount(_cells(X_def, edges), minlength=n_bins ** len(ax_idx))
    target = np.where(counts >= min_cell)[0]
    tmap = -np.ones(n_bins ** len(ax_idx), dtype=np.int64)
    tmap[target] = np.arange(len(target))
    return edges, tmap, len(target), (X_b, Y_b)


def _support_axis(ax, full_range: bool = True) -> None:
    """Tick a covered-support axis every 0.05, optionally over the full 0-1.

    The tick spacing is always fixed, so support is read off the same grid
    everywhere. The *range* is a choice per figure. `full_range=True` pins 0-1,
    which is right where the question is "how complete is this?": it stops a
    coverage of 0.4 reading as near-complete and lets two figures be compared.
    `full_range=False` autoscales, which is right where the question is instead
    "how do the curves differ?", as in the compute trade-off figures, where
    pinning the axis compresses all the structure into its lower third.
    """
    from matplotlib.ticker import MultipleLocator
    if full_range:
        ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.tick_params(axis="y", labelsize=8)


def coverage_of(cells: np.ndarray, tmap: np.ndarray, n_target: int) -> float:
    """Fraction of target cells hit; entries < 0 are treated as out-of-band."""
    cells = cells[cells >= 0]
    m = tmap[cells]
    m = m[m >= 0]
    return len(np.unique(m)) / n_target if n_target else float("nan")


@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
              show_default=True)
@click.option("--baseline-data-dir", default="/ptmp/jwuerzin/data/18387358", show_default=True)
@click.option("--mcmc-data-dir", default="/ptmp/jwuerzin/data/neutralino_v4", show_default=True)
@click.option("--cache-dir", default="/ptmp/jwuerzin/analysis/all_runs", show_default=True)
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/all_runs", show_default=True)
@click.option("--sweep-id", default="", show_default=True,
              help="Optional sweep_id prefix filter. Empty by default: the "
                   "manifest marks superseded generations, which the "
                   "include-status filter already drops, and a prefix filter "
                   "would exclude cells resubmitted under a later sweep_id "
                   "(the TabPFN cells carry 20260806).")
@click.option("--include-status", default="completed,running,timeout,submitted",
              show_default=True)
@click.option("--models", default=None, help="Comma list of picks (default: all).")
@click.option("--model-tag", default="", show_default=True,
              help="OUTPUT_TAG of a variant sweep (e.g. 'expr'), so its tagged manifest rows resolve against the default per-model picks.")
@click.option("--tolerance", default=0.10, show_default=True)
@click.option("--n-bins", default=12, show_default=True,
              help="Equal-occupancy bins per axis (n_bins^3 cells).")
@click.option("--min-cell", default=20, show_default=True,
              help="Reference in-band points needed for a cell to count as target.")
@click.option("--mcmc-max-samples", default=500_000, show_default=True)
@click.option("--mcmc-total-rows", default=17_498_112, show_default=True,
              help="Stored reference rows before subsampling; one row is one "
                   "proposal, so a budget of N calls buys N/(total/subsample) "
                   "rows of the subsample.")
@click.option("--n-repeats", default=20, show_default=True,
              help="Random subsets averaged per (k, budget) point.")
@click.option("--require-neutralino-lsp/--no-require-neutralino-lsp",
              default=False, show_default=True)
@click.option("--target", default="DMRD", show_default=True,
              help="TARGET_CONFIG key: sets the branch and the band centre.")
@click.option("--support-source", default="mcmc",
              type=click.Choice(["mcmc", "pool"]), show_default=True,
              help="Population defining the in-band support. 'mcmc' is the "
                   "emcee posterior (the paper's original definition, relic "
                   "density only); 'pool' is the random-scan dataset, which "
                   "exists for every target. Axes, bins, tolerance, half-split "
                   "and min_cell are identical either way, so the two are "
                   "like-for-like and differ only in the reference population. "
                   "The output filename carries the source, so running twice "
                   "gives both figures side by side.")
def main(manifest, baseline_data_dir, mcmc_data_dir, cache_dir, output_dir,
         sweep_id, include_status, models, tolerance, n_bins, min_cell,
         mcmc_max_samples, mcmc_total_rows, n_repeats, require_neutralino_lsp,
         target, support_source, model_tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_hit_rate_trajectories_multiseed as phr
    from analyse_runs import filter_run_neutralino_lsp, load_run
    from pmssm.data import load_mcmc_data

    rng = np.random.default_rng(RNG_SEED)
    ax_idx = [PARAM_ORDER.index(a) for a in AXES]
    from pmssm.config import TARGET_CONFIG as _TC
    # NB `target` is reassigned below to the support cell indices (a numpy
    # array), so the registry key has to be captured before that happens.
    target_key = target
    true_val = float(_TC[target_key]["true_value"])

    # ── target support from the first half of the reference ───────────────────
    # A target with no posterior has nothing to load here, and its support must
    # come from the pool instead. Refuse the incoherent combination rather than
    # scoring one observable against another's reference, and stand in an empty
    # array so the emcee-derived curves below simply produce nothing.
    _has_ref = _TC[target_key].get("has_mcmc_reference", False)
    if not _has_ref:
        if support_source != "pool":
            raise click.UsageError(
                f"target {target_key!r} has no emcee reference; pass "
                f"--support-source pool")
        Xm = np.empty((0, len(ax_idx)))
        Ym = np.empty(0)
    else:
        Xm, Ym = load_mcmc_data(data_dir=mcmc_data_dir, target=target_key,
                                require_neutralino_lsp=require_neutralino_lsp,
                                max_samples=mcmc_max_samples)
        Xm = np.asarray(Xm.numpy() if hasattr(Xm, "numpy") else Xm)[:, ax_idx]
        Ym = np.asarray(Ym.numpy() if hasattr(Ym, "numpy") else Ym).ravel()
    perm = rng.permutation(len(Xm))
    half = len(perm) // 2
    X_a, Y_a = Xm[perm[:half]], Ym[perm[:half]]      # defines the target support
    X_b, Y_b = Xm[perm[half:]], Ym[perm[half:]]      # supplies the MCMC curve
    X_def = X_a[np.abs(Y_a - true_val) / true_val < tolerance]

    if _has_ref:
        edges = [np.quantile(X_def[:, j], np.linspace(0, 1, n_bins + 1)) for j in range(3)]
        for e in edges:                  # guard against duplicate quantiles
            e[0], e[-1] = -np.inf, np.inf
        c_def = _cells(X_def, edges)
        counts = np.bincount(c_def, minlength=n_bins ** 3)
        target = np.where(counts >= min_cell)[0]
        tmap = -np.ones(n_bins ** 3, dtype=np.int64)
        tmap[target] = np.arange(len(target))
        n_target = len(target)
        click.echo(f"[cov] target support: {n_target} cells of {n_bins**3} "
                   f"(>= {min_cell} in-band reference points), from {half} defining points")
    else:
        edges, tmap, n_target = None, None, 0

    # Optionally REPLACE the support with one defined from the random-scan pool.
    # Everything downstream is untouched, so the two figures differ only in what
    # they call the target region. The held-out MCMC half still supplies its
    # curve, which against a pool-defined support answers a genuinely useful
    # question: how much of the flat scan's in-band region does the posterior
    # itself reach?
    if support_source == "pool":
        rng_pool = np.random.default_rng(RNG_SEED)
        edges, tmap, n_target, _held_pool = build_target(
            mcmc_data_dir, ax_idx, n_bins, min_cell, tolerance,
            mcmc_max_samples, require_neutralino_lsp, rng_pool, true_val=None,
            target=target_key, support_source="pool", pool_data_dir=baseline_data_dir)
        click.echo(f"[cov] REPLACED with random-scan support: {n_target} cells "
                   f"of {n_bins**3} (>= {min_cell} in-band pool points)")

    def cover_frac(cells: np.ndarray) -> float:
        cells = cells[cells >= 0]                 # drop out-of-band entries
        m = tmap[cells]
        m = m[m >= 0]                             # drop cells outside the target
        return len(np.unique(m)) / n_target if n_target else float("nan")

    # ── AL cells: per-seed in-band points in acquisition order ───────────────
    statuses = {s.strip() for s in include_status.split(",")}
    picks = picks_with_tag(model_tag)
    if models:
        want = {m.strip() for m in models.split(",")}
        picks = {m: v for m, v in picks.items() if m in want}
    rows = [r for r in csv.DictReader(open(manifest))
            if r["status"] in statuses
            and (not sweep_id or (r.get("sweep_id") or "").startswith(sweep_id))]

    al_seeds: dict[str, list[np.ndarray]] = {}
    for model, (strat, warm) in picks.items():
        seen, seqs = set(), []
        for r in rows:
            if (r["model"], r["strategy"], r["warm_start"]) != (model, strat, warm):
                continue
            d = r["expected_run_dir"]
            if d in seen:
                continue
            seen.add(d)
            try:
                run = load_run(d)
                if require_neutralino_lsp:
                    run = filter_run_neutralino_lsp(run)
            except Exception:
                continue
            Y = np.asarray(run.Y).ravel()
            inb_i = np.abs(Y - true_val) / true_val < tolerance
            cells_i = _cells(np.asarray(run.X)[:, ax_idx], edges)
            # cell index where in-band, -1 otherwise: a budget of N simulator
            # calls is the first N entries, of which only the in-band ones cover
            seqs.append(np.where(inb_i, cells_i, -1))
        if len(seqs) >= 2:
            al_seeds[model] = seqs
            click.echo(f"[cov] {model}: {len(seqs)} replicas, "
                       f"calls {[len(s) for s in seqs]}, "
                       f"in-band {[int((s >= 0).sum()) for s in seqs]}")

    # ── baselines: pool and held-out reference, as cell sequences ────────────
    Xp, Yp = phr._load_xy_full(baseline_data_dir, target_key, Path(cache_dir))
    Yp = np.asarray(Yp).ravel()
    pool_cells = np.where(np.abs(Yp - true_val) / true_val < tolerance,
                          _cells(np.asarray(Xp)[:, ax_idx], edges), -1)
    held_cells = np.where(np.abs(Y_b - true_val) / true_val < tolerance,
                          _cells(X_b, edges), -1)

    # ── curves ───────────────────────────────────────────────────────────────
    # Each model gets its own budget grid running out to its own shortest
    # replica, so that a model with short runs (TabPFN) does not truncate
    # everyone else's curve. `common_calls` is the largest budget every model
    # reaches; the figure marks it, and cross-model comparisons are quoted
    # there rather than at each curve's own endpoint.
    model_max = {m: min(len(s) for s in seqs) for m, seqs in al_seeds.items()}
    common_calls = min(model_max.values())
    max_calls = max(model_max.values())
    budgets = np.unique(np.geomspace(200, common_calls, 16).astype(int))
    out: dict = {"config": {"axes": AXES, "n_bins": n_bins, "min_cell": min_cell,
                            "tolerance": tolerance, "n_target_cells": n_target,
                            "sweep_id": sweep_id, "n_repeats": n_repeats,
                            "common_calls": int(common_calls),
                            "model_max_calls": {m: int(v)
                                                for m, v in model_max.items()}},
                 "al": {}, "baselines": {}}

    for model, seqs in al_seeds.items():
        S = len(seqs)
        out["al"][model] = {}
        m_budgets = np.unique(np.geomspace(200, model_max[model], 16).astype(int))
        for k in (1, S):
            fr = []
            for N in m_budgets:
                per = N // k
                vals = []
                for _ in range(n_repeats if k < S else 1):
                    pick = rng.choice(S, size=k, replace=False)
                    vals.append(cover_frac(np.concatenate([seqs[i][:per] for i in pick])))
                fr.append(float(np.mean(vals)))
            out["al"][model][f"k{k}"] = {"budget": [int(b) for b in m_budgets],
                                         "coverage": fr}
        # value at the largest budget every model reaches, for fair comparison
        for k in (1, S):
            r = out["al"][model][f"k{k}"]
            j = max(i for i, b in enumerate(r["budget"]) if b <= common_calls)
            r["coverage_at_common"] = r["coverage"][j]
        click.echo(f"[cov] {model:<16} k=1 -> {out['al'][model]['k1']['coverage'][-1]:.3f}, "
                   f"k={S} -> {out['al'][model][f'k{S}']['coverage'][-1]:.3f} "
                   f"at N={m_budgets[-1]}  |  at common N={common_calls}: "
                   f"k=1 {out['al'][model]['k1']['coverage_at_common']:.3f}, "
                   f"k={S} {out['al'][model][f'k{S}']['coverage_at_common']:.3f}")

    # One pool point is one simulator call. One reference row is one proposal,
    # but the reference was uniformly subsampled, so each retained row stands
    # for total/subsample proposals; the budget is deflated accordingly.
    mcmc_factor = max(1.0, mcmc_total_rows / max(1, len(Xm)))
    click.echo(f"[cov] one reference row stands for {mcmc_factor:.1f} proposals")
    base_budgets = np.unique(np.geomspace(200, max_calls, 16).astype(int))
    for name, cells, per_entry in (("random_scan", pool_cells, 1.0),
                                   ("mcmc", held_cells, mcmc_factor)):
        fr = []
        for N in base_budgets:
            n_entries = int(N / per_entry)
            if n_entries < 1 or n_entries > len(cells):
                fr.append(float("nan"))
                continue
            vals = [cover_frac(cells[rng.choice(len(cells), size=n_entries,
                                                replace=False)])
                    for _ in range(n_repeats)]
            fr.append(float(np.mean(vals)))
        out["baselines"][name] = {"budget": [int(b) for b in base_budgets],
                                  "coverage": fr, "calls_per_entry": per_entry}
        last = next((v for v in reversed(fr) if v == v), float("nan"))
        click.echo(f"[cov] {name:<16} -> {last:.3f} at the largest usable budget")

    # ── second view: matched IN-BAND points, which removes the productivity
    # difference between k=1 and k=5 and isolates point-set diversity ────────
    out["al_inband"] = {}
    inb_model_max = {m: min(int((s >= 0).sum()) for s in seqs)
                     for m, seqs in al_seeds.items()}
    inb_common = min(inb_model_max.values())
    out["config"]["inband_common"] = int(inb_common)
    out["config"]["inband_model_max"] = {m: int(v) for m, v in inb_model_max.items()}
    for model, seqs in al_seeds.items():
        inb_only = [s[s >= 0] for s in seqs]           # in-band cells, in order
        S = len(inb_only)
        out["al_inband"][model] = {}
        inb_budgets = np.unique(np.geomspace(20, inb_model_max[model], 12).astype(int))
        for k in (1, S):
            fr = []
            for N in inb_budgets:
                per = N // k
                vals = []
                for _ in range(n_repeats if k < S else 1):
                    pick = rng.choice(S, size=k, replace=False)
                    vals.append(cover_frac(np.concatenate(
                        [inb_only[i][:per] for i in pick])))
                fr.append(float(np.mean(vals)))
            out["al_inband"][model][f"k{k}"] = {"budget": [int(b) for b in inb_budgets],
                                               "coverage": fr}
        r = out["al_inband"][model]
        for k in (1, S):
            rr = r[f"k{k}"]
            j = max(i for i, b in enumerate(rr["budget"]) if b <= inb_common)
            rr["coverage_at_common"] = rr["coverage"][j]
        click.echo(f"[cov] {model:<16} in-band-matched at N={inb_budgets[-1]}: "
                   f"k=1 {r['k1']['coverage'][-1]:.3f} -> k={S} "
                   f"{r[f'k{S}']['coverage'][-1]:.3f}  |  at common "
                   f"N={inb_common}: k=1 {r['k1']['coverage_at_common']:.3f} -> "
                   f"k={S} {r[f'k{S}']['coverage_at_common']:.3f}")

    # ── figure ───────────────────────────────────────────────────────────────
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for model, rec in out["al"].items():
        c = phr.MODEL_COLORS.get(model, "gray")
        ks = sorted(rec, key=lambda s: int(s[1:]))
        ax.plot(rec[ks[0]]["budget"], rec[ks[0]]["coverage"], "-", color=c, lw=1.7,
                label=MODEL_DISPLAY.get(model, model))
        ax.plot(rec[ks[-1]]["budget"], rec[ks[-1]]["coverage"], "--", color=c, lw=1.3)
    ax.plot(out["baselines"]["random_scan"]["budget"],
            out["baselines"]["random_scan"]["coverage"], ":", color="black", lw=1.6,
            label="random scan")
    if _has_ref:
        ax.plot(out["baselines"]["mcmc"]["budget"], out["baselines"]["mcmc"]["coverage"],
                "-.", color="0.45", lw=1.6, label="MCMC (held-out half)")
    ax.axvline(common_calls, color="0.6", ls=(0, (1, 3)), lw=1.0, zorder=0)
    ax.annotate("budget all models reach", xy=(common_calls, 0.02),
                xytext=(-4, 0), textcoords="offset points", rotation=90,
                ha="right", va="bottom", fontsize=6.5, color="0.4")
    ax.set_xscale("log")
    ax.set_xlabel("simulator calls (valid models evaluated, total across replicas)")
    _srclbl = "emcee reference" if support_source == "mcmc" else "random scan"
    ax.set_ylabel(f"fraction of {_srclbl} in-band support covered\n({n_target} cells)")
    _support_axis(ax)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left",
              title="solid: one replica   dashed: budget split over 5 replicas",
              title_fontsize=7)
    for model, rec in out["al_inband"].items():
        c = phr.MODEL_COLORS.get(model, "gray")
        ks = sorted(rec, key=lambda s: int(s[1:]))
        ax2.plot(rec[ks[0]]["budget"], rec[ks[0]]["coverage"], "-", color=c, lw=1.7,
                 label=MODEL_DISPLAY.get(model, model))
        ax2.plot(rec[ks[-1]]["budget"], rec[ks[-1]]["coverage"], "--", color=c, lw=1.3)
    ax2.axvline(inb_common, color="0.6", ls=(0, (1, 3)), lw=1.0, zorder=0)
    ax2.set_xscale("log")
    ax2.set_xlabel("in-band points spent (total across replicas)")
    ax2.set_ylabel(f"fraction of {_srclbl} in-band support covered\n({n_target} cells)")
    _support_axis(ax2)
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    _sfx = "" if support_source == "mcmc" else f"_{support_source}"
    p_png = Path(output_dir) / f"coverage_saturation{_sfx}.png"
    fig.savefig(p_png, dpi=200)
    p_json = Path(output_dir) / f"coverage_saturation{_sfx}.json"
    p_json.write_text(json.dumps(out, indent=1))
    click.echo(f"[cov] wrote {p_png} and {p_json}")


if __name__ == "__main__":
    main()
