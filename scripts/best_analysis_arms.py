#!/usr/bin/env python3
"""Which SModelS analysis is most sensitive at the exclusion boundary, per arm.

The ntuples cannot answer this. `bestExpR_r_expected` is a max over the
per-result `r_expected`, but the ntuple writer keyed the per-analysis branches on
AnalysisID alone, so multiple signal regions of one analysis overwrote each other
and only the last survived. The winning result survives for roughly half the
points, and that half is selected by whether the winner happened to be written,
which correlates with the very thing being measured.

The per-point SModelS `.py` outputs keep every entry, so for the AL arms the
winner is exact. They were not retained for the random-scan pool, so the pool can
only be estimated with the biased ntuple route. This script therefore also
CALIBRATES that route: on the arms, where both are available, it measures how far
the ntuple-style estimate lands from the truth, which is what makes the pool
number interpretable rather than merely caveated.

Reported per arm: the distribution over winning AnalysisID, the effective number
of analyses exp(H) (H the Shannon entropy, so a flat distribution over k
analyses gives k and total concentration gives 1), the top analysis' share, and
the same by TxNames topology.
"""
import collections
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

TOL = 0.10
TRUE_VAL = 1.0
MANIFEST = "/ptmp/jwuerzin/analysis/joint/manifest_expr.csv"
OUT = Path("/ptmp/jwuerzin/analysis/joint/best_analysis")
FILE_BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
# Iterations sampled per run, spread evenly so late iterations are represented.
ITERS_PER_RUN = int(sys.argv[2]) if len(sys.argv) > 2 else 12
ARMS = ("deep_gp_expr", "exact_gp_expr", "transformer_expr", "dnn_expr", "tabpfn_expr")


def read_point(path):
    """(winner_entry, all_entries) for one SModelS .py output, or (None, None)."""
    g = {}
    try:
        exec(path.read_text(), g)
    except Exception:
        return None, None
    er = g.get("smodelsOutput", {}).get("ExptRes")
    if not er:
        return None, None
    cand = [r for r in er if r.get("r_expected") is not None]
    if not cand:
        return None, er
    return max(cand, key=lambda r: r["r_expected"]), er


def ntuple_style_winner(er):
    """What the ntuple's last-wins branches would have named as the winner.

    Reproduces the old writer: keep the LAST entry per AnalysisID, then take the
    argmax of theory prediction / expected upper limit over those.
    """
    last = {}
    for r in er:
        last[r["AnalysisID"]] = r
    best_a, best_r = None, -math.inf
    for a, r in last.items():
        tp, eu = r.get("theory prediction (fb)"), r.get("expected upper limit (fb)")
        if tp is None or not eu or eu <= 0:
            continue
        v = tp / eu
        if v > best_r:
            best_a, best_r = a, v
    return best_a


def effective_number(counter):
    n = sum(counter.values())
    if not n:
        return float("nan")
    h = -sum((c / n) * math.log(c / n) for c in counter.values() if c)
    return math.exp(h)


def spread_sample(paths, budget):
    """Even sample across the list, so late iterations are not under-represented."""
    if len(paths) <= budget:
        return paths
    idx = np.linspace(0, len(paths) - 1, budget).astype(int)
    return [paths[i] for i in idx]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(MANIFEST)))
    result = {}
    for arm in ARMS:
        run_dirs = [Path(r["expected_run_dir"]) for r in rows if r["model"] == arm]
        # Sample ITERATIONS before touching the tree. A full rglob over five
        # seeds and up to 148 iterations is a GPFS metadata traversal that costs
        # more than reading the files it finds, so glob one level to list the
        # iterations, spread a fixed number of them across the run, and only
        # walk those.
        pys = []
        for d in run_dirs:
            its = sorted(d.glob("iteration_*"))
            if not its:
                continue
            for it in spread_sample(its, ITERS_PER_RUN):
                pys.extend(sorted(it.glob("worker_*/scan/SModelS/*.slha.py")))
                pys.extend(sorted(it.glob("retry_*/worker_*/scan/SModelS/*.slha.py")))
        pys = spread_sample(pys, FILE_BUDGET)
        by_ana, by_tx = collections.Counter(), collections.Counter()
        agree = disagree = n_inband = n_read = 0
        for p in pys:
            best, er = read_point(p)
            n_read += 1
            if best is None:
                continue
            if abs(best["r_expected"] - TRUE_VAL) / TRUE_VAL >= TOL:
                continue
            n_inband += 1
            by_ana[best["AnalysisID"]] += 1
            for tx in best.get("TxNames") or ["?"]:
                by_tx[tx] += 1
            nt = ntuple_style_winner(er)
            if nt == best["AnalysisID"]:
                agree += 1
            else:
                disagree += 1
        result[arm] = {
            "files_read": n_read, "n_inband": n_inband,
            "distinct_analyses": len(by_ana),
            "effective_analyses": effective_number(by_ana),
            "top_share": (by_ana.most_common(1)[0][1] / n_inband) if n_inband else float("nan"),
            "by_analysis": dict(by_ana.most_common()),
            "by_txname": dict(by_tx.most_common(12)),
            "ntuple_estimator_agrees": agree,
            "ntuple_estimator_disagrees": disagree,
        }
        print(f"[best-ana] {arm:18s} read {n_read:6d} py, in-band {n_inband:5d}, "
              f"distinct {len(by_ana):3d}, effective {effective_number(by_ana):5.2f}, "
              f"top {result[arm]['top_share']*100:4.1f}%, "
              f"ntuple-route accuracy {100*agree/max(agree+disagree,1):4.1f}%", flush=True)
        for a, c in by_ana.most_common(5):
            print(f"              {a:36s} {100*c/max(n_inband,1):5.1f}%", flush=True)
    (OUT / "best_analysis.json").write_text(json.dumps(result, indent=1))
    print(f"[best-ana] wrote {OUT/'best_analysis.json'}")


if __name__ == "__main__":
    main()
