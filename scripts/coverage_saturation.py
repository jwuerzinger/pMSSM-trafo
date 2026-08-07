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

from mcmc_diagnostics import DEFAULT_AL_PICKS, MODEL_DISPLAY, PARAM_ORDER  # noqa: E402

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


@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
              show_default=True)
@click.option("--baseline-data-dir", default="/ptmp/jwuerzin/data/18387358", show_default=True)
@click.option("--mcmc-data-dir", default="/ptmp/jwuerzin/data/neutralino_v4", show_default=True)
@click.option("--cache-dir", default="/ptmp/jwuerzin/analysis/all_runs", show_default=True)
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/all_runs", show_default=True)
@click.option("--sweep-id", default="20260803_18", show_default=True)
@click.option("--include-status", default="completed,running,timeout,submitted",
              show_default=True)
@click.option("--models", default=None, help="Comma list of picks (default: all).")
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
def main(manifest, baseline_data_dir, mcmc_data_dir, cache_dir, output_dir,
         sweep_id, include_status, models, tolerance, n_bins, min_cell,
         mcmc_max_samples, mcmc_total_rows, n_repeats, require_neutralino_lsp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_hit_rate_trajectories_multiseed as phr
    from analyse_runs import filter_run_neutralino_lsp, load_run
    from pmssm.data import load_mcmc_data

    rng = np.random.default_rng(RNG_SEED)
    ax_idx = [PARAM_ORDER.index(a) for a in AXES]
    true_val = 0.12

    # ── target support from the first half of the reference ───────────────────
    Xm, Ym = load_mcmc_data(data_dir=mcmc_data_dir, target="DMRD",
                            require_neutralino_lsp=require_neutralino_lsp,
                            max_samples=mcmc_max_samples)
    Xm = np.asarray(Xm.numpy() if hasattr(Xm, "numpy") else Xm)[:, ax_idx]
    Ym = np.asarray(Ym.numpy() if hasattr(Ym, "numpy") else Ym).ravel()
    perm = rng.permutation(len(Xm))
    half = len(perm) // 2
    X_a, Y_a = Xm[perm[:half]], Ym[perm[:half]]      # defines the target support
    X_b, Y_b = Xm[perm[half:]], Ym[perm[half:]]      # supplies the MCMC curve
    X_def = X_a[np.abs(Y_a - true_val) / true_val < tolerance]

    edges = [np.quantile(X_def[:, j], np.linspace(0, 1, n_bins + 1)) for j in range(3)]
    for e in edges:                      # guard against duplicate quantiles
        e[0], e[-1] = -np.inf, np.inf
    c_def = _cells(X_def, edges)
    counts = np.bincount(c_def, minlength=n_bins ** 3)
    target = np.where(counts >= min_cell)[0]
    tmap = -np.ones(n_bins ** 3, dtype=np.int64)
    tmap[target] = np.arange(len(target))
    n_target = len(target)
    click.echo(f"[cov] target support: {n_target} cells of {n_bins**3} "
               f"(>= {min_cell} in-band reference points), from {half} defining points")

    def cover_frac(cells: np.ndarray) -> float:
        cells = cells[cells >= 0]                 # drop out-of-band entries
        m = tmap[cells]
        m = m[m >= 0]                             # drop cells outside the target
        return len(np.unique(m)) / n_target if n_target else float("nan")

    # ── AL cells: per-seed in-band points in acquisition order ───────────────
    statuses = {s.strip() for s in include_status.split(",")}
    picks = dict(DEFAULT_AL_PICKS)
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
    Xp, Yp = phr._load_xy_full(baseline_data_dir, "DMRD", Path(cache_dir))
    Yp = np.asarray(Yp).ravel()
    pool_cells = np.where(np.abs(Yp - true_val) / true_val < tolerance,
                          _cells(np.asarray(Xp)[:, ax_idx], edges), -1)
    held_cells = np.where(np.abs(Y_b - true_val) / true_val < tolerance,
                          _cells(X_b, edges), -1)

    # ── curves ───────────────────────────────────────────────────────────────
    max_calls = min(min(len(s) for s in seqs) for seqs in al_seeds.values())
    budgets = np.unique(np.geomspace(200, max_calls, 16).astype(int))
    out: dict = {"config": {"axes": AXES, "n_bins": n_bins, "min_cell": min_cell,
                            "tolerance": tolerance, "n_target_cells": n_target,
                            "sweep_id": sweep_id, "n_repeats": n_repeats},
                 "al": {}, "baselines": {}}

    for model, seqs in al_seeds.items():
        S = len(seqs)
        out["al"][model] = {}
        for k in (1, S):
            fr = []
            for N in budgets:
                per = N // k
                vals = []
                for _ in range(n_repeats if k < S else 1):
                    pick = rng.choice(S, size=k, replace=False)
                    vals.append(cover_frac(np.concatenate([seqs[i][:per] for i in pick])))
                fr.append(float(np.mean(vals)))
            out["al"][model][f"k{k}"] = {"budget": [int(b) for b in budgets],
                                         "coverage": fr}
        click.echo(f"[cov] {model:<16} k=1 -> {out['al'][model]['k1']['coverage'][-1]:.3f}, "
                   f"k={S} -> {out['al'][model][f'k{S}']['coverage'][-1]:.3f} "
                   f"at N={budgets[-1]}")

    # One pool point is one simulator call. One reference row is one proposal,
    # but the reference was uniformly subsampled, so each retained row stands
    # for total/subsample proposals; the budget is deflated accordingly.
    mcmc_factor = max(1.0, mcmc_total_rows / max(1, len(Xm)))
    click.echo(f"[cov] one reference row stands for {mcmc_factor:.1f} proposals")
    for name, cells, per_entry in (("random_scan", pool_cells, 1.0),
                                   ("mcmc", held_cells, mcmc_factor)):
        fr = []
        for N in budgets:
            n_entries = int(N / per_entry)
            if n_entries < 1 or n_entries > len(cells):
                fr.append(float("nan"))
                continue
            vals = [cover_frac(cells[rng.choice(len(cells), size=n_entries,
                                                replace=False)])
                    for _ in range(n_repeats)]
            fr.append(float(np.mean(vals)))
        out["baselines"][name] = {"budget": [int(b) for b in budgets],
                                  "coverage": fr, "calls_per_entry": per_entry}
        last = next((v for v in reversed(fr) if v == v), float("nan"))
        click.echo(f"[cov] {name:<16} -> {last:.3f} at the largest usable budget")

    # ── second view: matched IN-BAND points, which removes the productivity
    # difference between k=1 and k=5 and isolates point-set diversity ────────
    out["al_inband"] = {}
    inb_min = min(int((s >= 0).sum()) for seqs in al_seeds.values() for s in seqs)
    inb_budgets = np.unique(np.geomspace(20, inb_min, 12).astype(int))
    out["config"]["inband_budgets"] = [int(b) for b in inb_budgets]
    for model, seqs in al_seeds.items():
        inb_only = [s[s >= 0] for s in seqs]           # in-band cells, in order
        S = len(inb_only)
        out["al_inband"][model] = {}
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
        click.echo(f"[cov] {model:<16} in-band-matched at N={inb_budgets[-1]}: "
                   f"k=1 {r['k1']['coverage'][-1]:.3f} -> k={S} "
                   f"{r[f'k{S}']['coverage'][-1]:.3f}")

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
    ax.plot(out["baselines"]["mcmc"]["budget"], out["baselines"]["mcmc"]["coverage"],
            "-.", color="0.45", lw=1.6, label="MCMC (held-out half)")
    ax.set_xscale("log")
    ax.set_xlabel("simulator calls (valid models evaluated, total across replicas)")
    ax.set_ylabel("fraction of reference in-band support covered")
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
    ax2.set_xscale("log")
    ax2.set_xlabel("in-band points spent (total across replicas)")
    ax2.set_ylabel("fraction of reference in-band support covered")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    p_png = Path(output_dir) / "coverage_saturation.png"
    fig.savefig(p_png, dpi=200)
    p_json = Path(output_dir) / "coverage_saturation.json"
    p_json.write_text(json.dumps(out, indent=1))
    click.echo(f"[cov] wrote {p_png} and {p_json}")


if __name__ == "__main__":
    main()
