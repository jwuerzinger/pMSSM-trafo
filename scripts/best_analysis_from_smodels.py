#!/usr/bin/env python3
"""Which SModelS analysis is most sensitive at the exclusion boundary.

Reads SModelS' per-point ``.py`` outputs directly, so it works wherever those
files live and needs nothing from this repository: standard library only, no
numpy, no uproot, no manifest.

WHY NOT THE NTUPLES. ``SModelS_bestExpR_r_expected`` is a maximum over the
per-result ``r_expected``, but the ntuple writer keyed its per-analysis branches
on AnalysisID alone, and SModelS reports an analysis once per signal region, so
later regions overwrote earlier ones and only the last survived. Measured on real
output: the region that set bestExpR survived for 8 of 19 points, and the
identifiable half is selected by whether the winner happened to be written, which
correlates with which analysis wins. Ntuples written after Run3ModelGen c7aebfe
keep the best region per analysis and store ``{AnalysisID}_r_expected``, so for
those the winner is the argmax of those branches and this script is unnecessary.

DEFINITIONS, which must match on both sides of any comparison:

  winner    the ExptRes entry with the largest ``r_expected``. This is exactly
            what bestExpR selects, so the winner's r_expected IS
            bestExpR_r_expected.
  in-band   ``|r_expected(winner) - TRUE_VAL| / TRUE_VAL < TOL``, default
            TRUE_VAL=1.0 and TOL=0.10, i.e. the same +-10% band around the
            exclusion boundary the paper uses.
  effective the exponential of the Shannon entropy of the winning-analysis
            distribution: a flat distribution over k analyses gives k, total
            concentration on one gives 1. Reported alongside the raw count of
            distinct analyses, which is sensitive to rare tails in a way the
            effective number is not.

USAGE

    # the random-scan pool, wherever its SModelS outputs are
    python3 best_analysis_from_smodels.py /path/to/scan --label "random pool" \
        --budget 25000 --jobs 8 --out pool_best_analysis.json

    # compare against the AL arms measured elsewhere
    python3 best_analysis_from_smodels.py --compare pool_best_analysis.json \
        arms_best_analysis.json

``--budget`` samples evenly across the discovered files rather than taking the
first N, so a scan written in generation order is not represented by its
beginning alone. Raise it until the shares stop moving; a few hundred in-band
points already pins a 30% share to about +-3 points, and the differences between
acquisition arms are tens of points.

``--iters-per-dir`` matters only for active-learning trees, where files sit under
``iteration_NNN/``: it samples that many iteration directories per run before
walking, because a full recursive walk of such a tree costs more on a parallel
filesystem than reading the files it finds. Set 0 to walk everything, which is
the right choice for a flat scan directory.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pmssm.iteration_housekeeping import iter_smodels_items  # noqa: E402


# ── reading one point ────────────────────────────────────────────────────────
def read_winner(item, tol, true_val):
    """(analysis, txnames, r_expected, all_entries) for one point, or None.

    SModelS writes these files as executable Python (``smodelsOutput = {...}``),
    which is how Run3ModelGen's ntupler reads them too, so exec is the format's
    intended reader rather than a shortcut.

    `item` is a Path on a run whose workspaces are intact, or a record dict from
    smodels_best_analysis.json on a packed run. For a record the fourth element
    is None: the record keeps only the top RANK_KEEP results, and
    ntuple_style_winner needs all of them, so it must be reported unavailable
    rather than computed from a truncated list.
    """
    if isinstance(item, dict):
        r = item.get("r_expected")
        if r is None or item.get("analysis") is None:
            return None
        if abs(r - true_val) / true_val >= tol:
            return None
        return (item["analysis"], tuple(item.get("txnames") or ("?",)),
                float(r), None)
    path = item
    g = {}
    try:
        with open(path) as fh:
            exec(fh.read(), g)                                  # noqa: S102
    except Exception:
        return None
    er = (g.get("smodelsOutput") or {}).get("ExptRes")
    if not er:
        return None
    cand = [r for r in er if r.get("r_expected") is not None]
    if not cand:
        return None
    best = max(cand, key=lambda r: r["r_expected"])
    if abs(best["r_expected"] - true_val) / true_val >= tol:
        return None
    return (best["AnalysisID"], tuple(best.get("TxNames") or ("?",)),
            float(best["r_expected"]), er)


def ntuple_style_winner(er):
    """The analysis a PRE-c7aebfe ntuple would have implied.

    Keeps the last entry per AnalysisID, then argmax of theory prediction over
    expected upper limit. Only useful for quantifying how wrong that route is;
    never use it as the measurement.
    """
    last = {}
    for r in er:
        last[r["AnalysisID"]] = r
    best_a, best_v = None, -math.inf
    for a, r in last.items():
        tp, eu = r.get("theory prediction (fb)"), r.get("expected upper limit (fb)")
        if tp is None or not eu or eu <= 0:
            continue
        v = tp / eu
        if v > best_v:
            best_a, best_v = a, v
    return best_a


# ── discovery ────────────────────────────────────────────────────────────────
def spread(seq, budget):
    """Evenly spaced subsample, preserving order. Pure stdlib."""
    seq = list(seq)
    if budget <= 0 or len(seq) <= budget:
        return seq
    step = (len(seq) - 1) / (budget - 1) if budget > 1 else 1
    return [seq[min(len(seq) - 1, int(round(i * step)))] for i in range(budget)]


def discover(root, pattern, iters_per_dir):
    """All matching files, optionally sampling iteration_* dirs first."""
    root = Path(root)
    if iters_per_dir:
        its = sorted(p for p in root.rglob("iteration_*") if p.is_dir())
        if its:
            by_run = collections.defaultdict(list)
            for it in its:
                by_run[it.parent].append(it)
            out = []
            for _run, dirs in sorted(by_run.items()):
                for d in spread(sorted(dirs), iters_per_dir):
                    # Loose files where they exist, the packed run's records
                    # where they do not. Only meaningful for the default
                    # *.slha.py pattern; any other pattern stays a pure glob.
                    if pattern == "*.slha.py":
                        out.extend(iter_smodels_items(d))
                    else:
                        out.extend(sorted(d.rglob(pattern)))
            return out
    return sorted(root.rglob(pattern))


# ── measurement ──────────────────────────────────────────────────────────────
def measure(paths, tol, true_val, jobs):
    by_ana = collections.Counter()
    by_tx = collections.Counter()
    agree = disagree = n_inband = nt_unavailable = 0

    def handle(res):
        nonlocal agree, disagree, n_inband, nt_unavailable
        if res is None:
            return
        ana, txs, _r, er = res
        n_inband += 1
        by_ana[ana] += 1
        for tx in txs:
            by_tx[tx] += 1
        if er is None:
            # Packed run: only the top RANK_KEEP results survive in the record,
            # and this comparison needs all of them. Counted apart rather than
            # folded into "disagrees", which would read as a measurement.
            nt_unavailable += 1
        elif ntuple_style_winner(er) == ana:
            agree += 1
        else:
            disagree += 1

    if jobs and jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial
        fn = partial(read_winner, tol=tol, true_val=true_val)
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for res in ex.map(fn, paths, chunksize=64):
                handle(res)
    else:
        for p in paths:
            handle(read_winner(p, tol, true_val))

    n = sum(by_ana.values())
    h = -sum((c / n) * math.log(c / n) for c in by_ana.values() if c) if n else float("nan")
    return {
        "files_read": len(paths),
        "n_inband": n_inband,
        "distinct_analyses": len(by_ana),
        "effective_analyses": math.exp(h) if n else float("nan"),
        "top_share": (by_ana.most_common(1)[0][1] / n) if n else float("nan"),
        "by_analysis": dict(by_ana.most_common()),
        "by_txname": dict(by_tx.most_common()),
        "ntuple_route_agrees": agree,
        "ntuple_route_disagrees": disagree,
        "ntuple_route_unavailable": nt_unavailable,
        "ntuple_route_accuracy": agree / max(agree + disagree, 1),
    }


def report(label, r):
    print(f"\n=== {label} ===")
    print(f"  files read           {r['files_read']:,}")
    print(f"  in-band points       {r['n_inband']:,}")
    print(f"  distinct analyses    {r['distinct_analyses']}")
    print(f"  effective analyses   {r['effective_analyses']:.2f}")
    print(f"  top analysis share   {100 * r['top_share']:.1f}%")
    print(f"  pre-c7aebfe ntuple route would be right for "
          f"{100 * r['ntuple_route_accuracy']:.1f}% of these points")
    n = max(r["n_inband"], 1)
    print("  winning analysis:")
    for a, c in list(r["by_analysis"].items())[:12]:
        # binomial standard error on the share, so a small n is visibly small
        p = c / n
        se = 100 * math.sqrt(max(p * (1 - p) / n, 0.0))
        print(f"    {a:34s} {100 * p:5.1f}% +- {se:.1f}   ({c})")
    if r["by_txname"]:
        print("  winning topology (TxName):")
        for t, c in list(r["by_txname"].items())[:8]:
            print(f"    {t:34s} {100 * c / n:5.1f}%   ({c})")


def compare(paths):
    data = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        # accept either a bare result or {label: result}
        if "by_analysis" in d:
            data[Path(p).stem] = d
        else:
            data.update(d)
    keys = list(data)
    print(f"{'':22s} " + " ".join(f"{k[:14]:>14s}" for k in keys))
    for field, fmt in (("n_inband", "{:14,}"), ("distinct_analyses", "{:14d}"),
                       ("effective_analyses", "{:14.2f}"), ("top_share", "{:14.3f}")):
        print(f"{field:22s} " + " ".join(fmt.format(data[k][field]) for k in keys))
    anas = sorted({a for k in keys for a in data[k]["by_analysis"]},
                  key=lambda a: -sum(data[k]["by_analysis"].get(a, 0) for k in keys))
    print("\nshare of points where each analysis is most sensitive (%):")
    print(f"{'':22s} " + " ".join(f"{k[:14]:>14s}" for k in keys))
    for a in anas[:15]:
        cells = []
        for k in keys:
            n = max(data[k]["n_inband"], 1)
            cells.append(f"{100 * data[k]['by_analysis'].get(a, 0) / n:14.1f}")
        print(f"{a[:22]:22s} " + " ".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", help="directory tree holding SModelS .py outputs")
    ap.add_argument("--label", default=None, help="name for this dataset in the output")
    ap.add_argument("--pattern", default="*.slha.py", help="glob for the per-point files")
    ap.add_argument("--budget", type=int, default=25000,
                    help="max files to read, sampled evenly (0 = all)")
    ap.add_argument("--iters-per-dir", type=int, default=0,
                    help="for AL trees: iteration_* dirs to sample per run (0 = walk all)")
    ap.add_argument("--tol", type=float, default=0.10, help="half-width of the band")
    ap.add_argument("--true-val", type=float, default=1.0, help="band centre")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--out", default=None, help="write JSON here")
    ap.add_argument("--compare", nargs="+", metavar="JSON",
                    help="print a comparison table of previously written JSONs and exit")
    a = ap.parse_args()

    if a.compare:
        compare(a.compare)
        return 0
    if not a.root:
        ap.error("give a directory to scan, or --compare")

    label = a.label or Path(a.root).name
    print(f"[best-ana] discovering {a.pattern} under {a.root} ...", flush=True)
    files = discover(a.root, a.pattern, a.iters_per_dir)
    if not files:
        print(f"[best-ana] no files matched {a.pattern!r} under {a.root}", file=sys.stderr)
        return 1
    print(f"[best-ana] {len(files):,} found; reading {min(len(files), a.budget) if a.budget else len(files):,}",
          flush=True)
    res = measure(spread(files, a.budget), a.tol, a.true_val, a.jobs)
    res["config"] = {"root": str(a.root), "pattern": a.pattern, "tol": a.tol,
                     "true_val": a.true_val, "files_discovered": len(files),
                     "budget": a.budget, "iters_per_dir": a.iters_per_dir}
    report(label, res)
    if a.out:
        Path(a.out).write_text(json.dumps({label: res}, indent=1))
        print(f"\n[best-ana] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
