"""Are the neural surrogates steered toward the mode-switching ridges of r_exp?

The exclusion target is a maximum over the SModelS database,
r_exp(theta) = max_a r_exp^(a)(theta), so it is piecewise: along the ridges where
the argmax analysis switches, the function has a kink no monotone transform can
remove (log of a max is the max of the logs). The hypothesis under test is that a
*regression* surrogate's predictive spread is largest at those kinks, which are
scattered over all values of r and mostly nowhere near the r = 1 contour, and
that variance-ranked acquisition therefore spends its budget on them.

The observable: for each acquired point, the per-point SModelS output keeps every
result's ``r_expected``, so the distance to a mode switch is

    gap = log r_(1) - log r_(2)

over the ranked results. A point where two results are nearly tied (gap -> 0)
sits on a switching ridge; a point with one dominant result does not. Ties are
computed both over all results and over distinct AnalysisIDs, since two signal
regions of one analysis also produce a kink but not a physically distinct mode.

Read from the per-point ``.py`` outputs rather than the ntuples: the ntuple
writer keyed per-analysis branches on AnalysisID alone, so signal regions
overwrote one another (see scripts/best_analysis_arms.py, which calibrates that
bias). The random-scan pool kept no such outputs, so the reference here is
*between arms*: the Deep GP is the surrogate whose acquisition works on this
target (0.723 of its labelled set within a factor e of the boundary, coverage
0.620), so if the hypothesis holds, the neural arms should be tie-enriched
relative to it, and most visibly among their out-of-band picks.

The comparison is stratified by |log r|, because points near the contour may have
a different tie structure than points far from it, and the claim under test is
specifically about picks that are *away* from the contour.

Usage
-----
    python scripts/mode_switch_diagnostic.py                    # 1500 points/arm
    python scripts/mode_switch_diagnostic.py --budget 4000 --iters-per-run 16
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pmssm.iteration_housekeeping import iter_smodels_items  # noqa: E402

MANIFEST = "/ptmp/jwuerzin/analysis/joint/manifest_expr.csv"
ARMS = ("deep_gp_expr", "exact_gp_expr", "transformer_expr", "dnn_expr")
# Gap thresholds in nats: 0.1 is a 10% tie, 0.3 a 35% one.
TIE_CUTS = (0.1, 0.3)


def _read_results(item):
    """Ranked (r_expected, AnalysisID) for one point.

    `item` is a Path on a run whose workspaces are intact, or a record dict from
    smodels_best_analysis.json on a packed run. The record's ranking is
    truncated to the top RANK_KEEP, which covers the top result and the gap to
    the next distinct AnalysisID, the two quantities this study measures.
    """
    if isinstance(item, dict):
        rs = [(r, a) for r, a, _tp, _eul in (item.get("ranked") or [])
              if r is not None and r > 0]
        return sorted(rs, key=lambda t: -t[0]) or None
    path = item
    g: dict = {}
    try:
        exec(path.read_text(), g)                       # noqa: S102 - trusted output
    except Exception:
        return None
    er = g.get("smodelsOutput", {}).get("ExptRes")
    if not er:
        return None
    rs = [(r["r_expected"], r.get("AnalysisID", "?")) for r in er
          if r.get("r_expected") is not None and r["r_expected"] > 0]
    if not rs:
        return None
    return sorted(rs, key=lambda t: -t[0])


def _gaps(ranked):
    """(r_max, gap to next result, gap to next distinct analysis) in nats."""
    r1, a1 = ranked[0]
    gap_any = math.log(r1) - math.log(ranked[1][0]) if len(ranked) > 1 else math.inf
    gap_distinct = math.inf
    for r, a in ranked[1:]:
        if a != a1:
            gap_distinct = math.log(r1) - math.log(r)
            break
    return r1, gap_any, gap_distinct


def _sample_paths(run_dirs, iters_per_run, budget):
    """Spread the file budget over iterations, then over workers.

    Iterations are listed one level deep and sampled before any deeper walk: a
    full recursive glob over five seeds and 148 iterations is a GPFS metadata
    traversal costing more than reading the files it finds.
    """
    picked: list[Path] = []
    for d in run_dirs:
        iters = sorted(p for p in d.glob("iteration_*") if p.is_dir())
        if not iters:
            continue
        idx = np.linspace(0, len(iters) - 1, min(iters_per_run, len(iters))).astype(int)
        for i in sorted(set(idx.tolist())):
            # One call covers both layouts: the loose files where they exist,
            # the packed run's JSON records where they do not.
            picked.extend(iter_smodels_items(iters[i]))
    if len(picked) > budget:
        sel = np.linspace(0, len(picked) - 1, budget).astype(int)
        picked = [picked[i] for i in sel]
    return picked


def _summary(rows, label, out_of_band_only):
    """rows: list of (r_max, gap_any, gap_distinct)."""
    if out_of_band_only:
        rows = [r for r in rows if abs(math.log(r[0])) > 1.0]
    if not rows:
        return None
    gd = np.array([r[2] for r in rows if math.isfinite(r[2])])
    ga = np.array([r[1] for r in rows if math.isfinite(r[1])])
    n = len(rows)
    out = {"label": label, "n": n, "n_with_second_analysis": int(len(gd)),
           "median_gap_any": float(np.median(ga)) if len(ga) else None,
           "median_gap_distinct": float(np.median(gd)) if len(gd) else None}
    for c in TIE_CUTS:
        out[f"frac_tie_{c}"] = float((gd < c).mean()) if len(gd) else None
        out[f"frac_single_result"] = float(1 - len(gd) / n)
    return out


@click.command()
@click.option("--manifest", default=MANIFEST)
@click.option("--arms", default=",".join(ARMS))
@click.option("--budget", default=1500, show_default=True,
              help="Per-point outputs read per arm.")
@click.option("--iters-per-run", default=10, show_default=True)
@click.option("--output", default="/ptmp/jwuerzin/analysis/joint/mode_switch.json",
              show_default=True)
def main(manifest, arms, budget, iters_per_run, output):
    rows = list(csv.DictReader(open(manifest)))
    results = {}
    for arm in [a.strip() for a in arms.split(",") if a.strip()]:
        run_dirs = [Path(r["expected_run_dir"]) for r in rows if r["model"] == arm]
        run_dirs = [d for d in run_dirs if d.exists()]
        if not run_dirs:
            click.echo(f"[skip] {arm}: no run dirs")
            continue
        paths = _sample_paths(run_dirs, iters_per_run, budget)
        click.echo(f"[{arm}] reading {len(paths)} per-point outputs "
                   f"from {len(run_dirs)} runs")
        parsed = []
        for p in paths:
            r = _read_results(p)
            if r:
                parsed.append(_gaps(r))
        if not parsed:
            click.echo(f"[skip] {arm}: nothing parsed")
            continue
        results[arm] = {"all": _summary(parsed, arm, False),
                        "out_of_band": _summary(parsed, arm, True)}

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(results, indent=2))

    for scope in ("all", "out_of_band"):
        click.echo(f"\n=== {scope} picks: distance to the nearest mode switch ===")
        click.echo(f"{'arm':<20} {'n':>6} {'median gap':>11} {'tie<0.1':>9} "
                   f"{'tie<0.3':>9} {'single-result':>14}")
        for arm, r in results.items():
            s = r[scope]
            if not s:
                continue
            click.echo(f"{arm:<20} {s['n']:>6} {s['median_gap_distinct']:>11.3f} "
                       f"{s['frac_tie_0.1']:>9.3f} {s['frac_tie_0.3']:>9.3f} "
                       f"{s['frac_single_result']:>14.3f}")
    click.echo(f"\nGaps are log r_(1) - log r_(2) over distinct analyses, in nats; "
               f"smaller means closer to a switching ridge.\n[write] {output}")


if __name__ == "__main__":
    main()
