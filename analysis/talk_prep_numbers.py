"""Talk-prep tables: (A) statistical significance of the architecture ranking,
(B) cross-metric ranking stability, (D) best-pick audit.

Reads:
  - /ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv
  - per-run state.pt (for Y values & n_train_per_iter)
  - per-run accuracy_trajectory.json (for in-training accuracy capture)

Reports:
  Table A: per-architecture best-pick final-iter mean ± SEM on four metrics
           (hit_rate@10%, hits_per_desired@10%, acc_static_random, acc_mcmc)
           with pairwise z-scores between architectures.
  Table B: 4×N ranking table — does the same architecture win across metrics?
  Table D: top-2 cells per architecture with margin in SEM units.

Architectures considered: transformer, deep_gp, exact_gp, tabpfn, dnn.
Oracle (*_oracle) cells are excluded from the architecture comparison.

Run from repo root:
    PYTHONPATH=. .pixi/envs/rocm/bin/python analysis/talk_prep_numbers.py
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

MANIFEST = "/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv"
DMRD_TRUE = 0.12
TOL = 0.10
INCLUDE_STATUSES = {"completed", "timeout", "running"}
ITER_COMPLETENESS = 0.9   # filter cells reaching <90% of max iters for the model
ACC_DATASETS = ("static_random", "mcmc", "train", "val")


# ── manifest loader ────────────────────────────────────────────────────────────

def load_manifest_groups():
    """Return groups = {(model, strat, warm): [run_dir, ...]} for runnable rows."""
    groups: dict[tuple, list[str]] = defaultdict(list)
    with open(MANIFEST) as f:
        rd = csv.reader(f)
        hdr = next(rd)
        for row in rd:
            if row[8] not in INCLUDE_STATUSES:
                continue
            if not os.path.isdir(row[7]):
                continue
            groups[(row[2], row[3], row[4])].append(row[7])
    return groups


# ── per-seed metric extractors ────────────────────────────────────────────────

def _final_hit_rate(run_dir: str, tol: float = TOL):
    """Return (final hit_rate@tol, n_iters_completed). Tol relative to DMRD_TRUE."""
    sp = os.path.join(run_dir, "state.pt")
    if not os.path.exists(sp):
        return None, 0
    s = torch.load(sp, weights_only=False, map_location="cpu")
    if "Y" not in s:
        return None, 0
    Y = s["Y"].numpy().reshape(-1)
    n_train = list(s.get("al_n_train") or [])
    if not n_train:
        return None, 0
    n_final = int(n_train[-1])
    if n_final <= 0:
        return None, 0
    Y_slice = Y[:n_final]
    hits = int(np.sum(np.abs(Y_slice - DMRD_TRUE) / DMRD_TRUE < tol))
    return hits / n_final, len(n_train)


def _final_hits_per_desired(run_dir: str, tol: float = TOL,
                            n_samples_init: int = 2000):
    """Hits / desired = cumulative hits / (n_samples_init + cumulative n_select).
    Mirrors the multi-seed plot script's _hits_per_desired metric.
    """
    sp = os.path.join(run_dir, "state.pt")
    if not os.path.exists(sp):
        return None, 0
    s = torch.load(sp, weights_only=False, map_location="cpu")
    Y = s["Y"].numpy().reshape(-1)
    n_train = list(s.get("al_n_train") or [])
    if not n_train:
        return None, 0
    n_final = int(n_train[-1])
    hits = int(np.sum(np.abs(Y[:n_final] - DMRD_TRUE) / DMRD_TRUE < tol))
    n_selected_cumulative = max(n_final - n_samples_init, 0)
    desired = n_samples_init + n_selected_cumulative
    return hits / desired if desired > 0 else None, len(n_train)


def _final_accuracy(run_dir: str, dataset: str, role: str = "al"):
    """Final-iter classification accuracy on `dataset` for `role` from
    accuracy_trajectory.json (in-training capture). Returns None if missing."""
    p = os.path.join(run_dir, "accuracy_trajectory.json")
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p))
    except Exception:
        return None
    if not d:
        return None
    iters = sorted(int(k) for k in d.keys())
    final_iter = iters[-1]
    role_entry = (d.get(str(final_iter)) or {}).get(role) or {}
    return role_entry.get(dataset)


# ── aggregate per cell ─────────────────────────────────────────────────────────

def cell_metrics(run_dirs: list[str]) -> dict:
    """Compute per-seed final-iter values for all 4 metrics + iter counts."""
    out = {
        "hit_rate": [],
        "hits_per_desired": [],
        "acc_static_random": [],
        "acc_mcmc": [],
        "acc_train": [],
        "acc_val": [],
        "iters": [],
    }
    for d in run_dirs:
        hr, ni = _final_hit_rate(d)
        if hr is not None:
            out["hit_rate"].append(hr)
            out["iters"].append(ni)
        hpd, _ = _final_hits_per_desired(d)
        if hpd is not None:
            out["hits_per_desired"].append(hpd)
        for ds in ACC_DATASETS:
            v = _final_accuracy(d, ds, "al")
            if v is not None:
                out[f"acc_{ds}"].append(v)
    return out


def _mean_sem(xs):
    arr = np.asarray(xs, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), 0
    if len(arr) == 1:
        return float(arr[0]), float("nan"), 1
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr))), int(len(arr))


# ── architecture-level pick: best (strategy, warm) per metric ────────────────

REGULAR_MODELS = ("transformer", "deep_gp", "exact_gp", "tabpfn", "dnn")
PRIMARY_METRIC = "hit_rate"  # used to pick the canonical "best" cell


def collect_per_arch(groups):
    """Returns {arch: {(s, w): cell_metrics_dict}} for regular surrogates only."""
    out: dict = {a: {} for a in REGULAR_MODELS}
    for (m, s, w), dirs in groups.items():
        if m not in REGULAR_MODELS:
            continue
        out[m][(s, w)] = cell_metrics(dirs)
    return out


def best_pick(per_arch_cells: dict, arch: str, metric: str = PRIMARY_METRIC):
    """Return (s, w, mean, sem, n_seeds, min_iters) for the best cell of `arch`
    under `metric`, with the iteration-completeness filter applied (only
    consider cells reaching ≥ ITER_COMPLETENESS × max_iters for this arch).
    """
    cells = per_arch_cells.get(arch, {})
    if not cells:
        return None
    # Iteration-completeness filter
    max_iters = 0
    for (_, _), m in cells.items():
        if m["iters"]:
            max_iters = max(max_iters, max(m["iters"]))
    threshold = int(max_iters * ITER_COMPLETENESS) if max_iters > 0 else 0
    eligible = [(sw, m) for sw, m in cells.items()
                if m["iters"] and min(m["iters"]) >= threshold]
    if not eligible:
        eligible = list(cells.items())
    scored = []
    for (sw, m) in eligible:
        if not m[metric]:
            continue
        mean, sem, n = _mean_sem(m[metric])
        scored.append((sw, mean, sem, n, min(m["iters"]) if m["iters"] else 0))
    if not scored:
        return None
    scored.sort(key=lambda r: r[1], reverse=True)
    return scored[0]  # (sw, mean, sem, n, min_iters)


# ── pretty-printing ───────────────────────────────────────────────────────────

def fmt_mean_sem(mean, sem):
    if mean != mean:
        return "    nan"
    if sem != sem:
        return f"{mean:.4f}        "
    return f"{mean:.4f} ± {sem:.4f}"


def z_between(m1, s1, n1, m2, s2, n2):
    """Approximate pairwise z-score (Welch-style) between two means."""
    if n1 < 2 or n2 < 2:
        return float("nan")
    se = np.sqrt(s1 ** 2 + s2 ** 2)
    if se <= 0:
        return float("nan")
    return (m1 - m2) / se


# ── main reports ──────────────────────────────────────────────────────────────

def report_table_A_and_D(per_arch):
    """Table A (per-arch best-pick mean±SEM on 4 metrics) + Table D (top-2 cells per arch)."""
    metrics = ["hit_rate", "hits_per_desired", "acc_static_random", "acc_mcmc"]
    print("\n" + "=" * 90)
    print("TABLE A — Per-architecture best pick (by hit_rate@10%) on 4 metrics")
    print("=" * 90)

    picks = {}
    for arch in REGULAR_MODELS:
        bp = best_pick(per_arch, arch, "hit_rate")
        if bp is None:
            continue
        picks[arch] = bp

    header = f"{'arch':<14}{'pick':<28}{'hit_rate@10%':>20}{'hits/des@10%':>20}{'acc_static_rand':>22}{'acc_mcmc':>22}"
    print(header)
    print("-" * len(header))
    rows_for_z = []
    for arch, ((s, w), mean_hr, sem_hr, n, min_iters) in picks.items():
        # Pull all 4 metrics' mean±SEM for this picked cell
        cell = per_arch[arch][(s, w)]
        mhr, shr, nhr = _mean_sem(cell["hit_rate"])
        mhpd, shpd, nhpd = _mean_sem(cell["hits_per_desired"])
        msr, ssr, nsr = _mean_sem(cell["acc_static_random"])
        mmc, smc, nmc = _mean_sem(cell["acc_mcmc"])
        rows_for_z.append((arch, mhr, shr, nhr))
        print(f"{arch:<14}{s+'/'+w:<28}"
              f"{fmt_mean_sem(mhr, shr):>20}"
              f"{fmt_mean_sem(mhpd, shpd):>20}"
              f"{fmt_mean_sem(msr, ssr):>22}"
              f"{fmt_mean_sem(mmc, smc):>22}")

    # Pairwise z-scores on hit_rate (the primary metric)
    if len(rows_for_z) >= 2:
        print("\nPairwise z-scores (hit_rate@10%) — |z| > 1.96 ≈ significant at 5%:")
        for i in range(len(rows_for_z)):
            for j in range(i + 1, len(rows_for_z)):
                a, m1, s1, n1 = rows_for_z[i]
                b, m2, s2, n2 = rows_for_z[j]
                z = z_between(m1, s1, n1, m2, s2, n2)
                sig = "**" if abs(z) >= 1.96 else "  "
                print(f"  {a:<13} vs {b:<13}: z = {z:+.2f}  {sig}")

    print("\n" + "=" * 90)
    print("TABLE D — Top-2 cells per architecture (hit_rate@10%) + margin in SEM units")
    print("=" * 90)
    for arch in REGULAR_MODELS:
        cells = per_arch.get(arch, {})
        if not cells:
            continue
        # Apply iter-completeness filter
        max_iters = max((max(m["iters"]) for m in cells.values() if m["iters"]),
                        default=0)
        threshold = int(max_iters * ITER_COMPLETENESS)
        scored = []
        for (s, w), m in cells.items():
            if not m["hit_rate"] or not m["iters"] or min(m["iters"]) < threshold:
                continue
            mean, sem, n = _mean_sem(m["hit_rate"])
            scored.append(((s, w), mean, sem, n, min(m["iters"])))
        if not scored:
            continue
        scored.sort(key=lambda r: r[1], reverse=True)
        print(f"\n  {arch}:")
        for i, (sw, mean, sem, n, mi) in enumerate(scored[:3]):
            tag = "← picked" if i == 0 else ""
            print(f"    {i+1}. {sw[0]:<18}/{sw[1]:<8} "
                  f"hit_rate@10% = {fmt_mean_sem(mean, sem)}  "
                  f"(n_seeds={n}, min_iters={mi})  {tag}")
        if len(scored) >= 2:
            top, second = scored[0], scored[1]
            mtop, stop = top[1], top[2]
            m2nd, s2nd = second[1], second[2]
            se = np.sqrt(stop ** 2 + s2nd ** 2)
            margin_sem = (mtop - m2nd) / se if se > 0 else float("nan")
            flag = "⚠ within 1 SEM" if margin_sem < 1.0 else "✓"
            print(f"    margin top-1 over top-2: "
                  f"{(mtop-m2nd)*1000:.2f}e-3  ({margin_sem:.2f} SEM)  {flag}")


def report_table_B(per_arch):
    """Cross-metric ranking stability."""
    metrics = ["hit_rate", "hits_per_desired", "acc_static_random", "acc_mcmc"]
    print("\n" + "=" * 90)
    print("TABLE B — Architecture ranking per metric (1 = best)")
    print("=" * 90)
    # For each metric, score each arch's best pick and rank
    ranks: dict = {m: {} for m in metrics}
    means_by_metric: dict = {m: {} for m in metrics}
    for arch in REGULAR_MODELS:
        # pick canonical cell using hit_rate, then read each metric's value
        bp = best_pick(per_arch, arch, "hit_rate")
        if bp is None:
            continue
        (s, w), *_ = bp
        cell = per_arch[arch][(s, w)]
        for met in metrics:
            mean, _, _ = _mean_sem(cell[met])
            if mean == mean:  # not NaN
                means_by_metric[met][arch] = mean
    for met in metrics:
        sorted_arch = sorted(means_by_metric[met].items(),
                             key=lambda kv: kv[1], reverse=True)
        for r, (arch, _v) in enumerate(sorted_arch, start=1):
            ranks[met][arch] = r
    archs_with_data = sorted({a for met in metrics for a in ranks[met].keys()})
    print(f"\n{'arch':<14}" + "".join(f"{m:>22}" for m in metrics))
    print("-" * (14 + 22 * len(metrics)))
    for arch in archs_with_data:
        row = f"{arch:<14}"
        for met in metrics:
            r = ranks[met].get(arch)
            v = means_by_metric[met].get(arch)
            if r is None:
                row += f"{'--':>22}"
            else:
                row += f"{f'#{r} ({v:.4f})':>22}"
        print(row)

    # Stability summary
    all_winners = {met: min(means_by_metric[met].items(),
                            key=lambda kv: ranks[met][kv[0]])[0]
                   for met in metrics if means_by_metric[met]}
    unique_winners = set(all_winners.values())
    if len(unique_winners) == 1:
        winner = next(iter(unique_winners))
        print(f"\n  → stable: {winner} ranks #1 on every metric")
    else:
        print(f"\n  → UNSTABLE rankings: winners differ across metrics:")
        for met, w in all_winners.items():
            print(f"      {met:<22} -> {w}")


import re
from datetime import datetime
import statistics

ITER_HDR_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[\w+\s*\] "
    r"=== (?:Global|GP Active Learning|DNN Active Learning|pMSSM \(TabPFN\) iteration) "
    r"(?:Iteration )?(\d+)"
)
DONE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[\w+\s*\] Active Learning Complete")


def _wallclock_from_log(run_dir: str, n_select_per_iter: int = 500):
    """Return dict of wall-clock stats for one run.

    Parses the run's active_learning.log for iteration-header timestamps.
    Returns (per_iter_seconds_mean, total_seconds, oracle_calls_per_hour).
    """
    log = os.path.join(run_dir, "active_learning.log")
    if not os.path.exists(log):
        return None
    iter_times = []
    end_t = None
    with open(log) as f:
        for line in f:
            m = ITER_HDR_RE.match(line)
            if m:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                iter_times.append(ts)
                continue
            m = DONE_RE.match(line)
            if m:
                end_t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    if not iter_times:
        return None
    if end_t is None:
        end_t = iter_times[-1]
    deltas = [(iter_times[i+1] - iter_times[i]).total_seconds()
              for i in range(len(iter_times) - 1)]
    if not deltas:
        deltas = [(end_t - iter_times[0]).total_seconds()]
    per_iter = statistics.median(deltas)
    total = (end_t - iter_times[0]).total_seconds()
    oracle_per_h = (n_select_per_iter / per_iter * 3600) if per_iter > 0 else float("nan")
    return {"per_iter_s": per_iter, "total_s": total,
            "oracle_per_h": oracle_per_h, "n_iters": len(iter_times)}


def _final_R2_shared(run_dir: str):
    """Return dict of final-iter AL & Baseline R² on shared eval sets.

    Source of truth: state.pt (always populated by both the transformer and
    GP pipelines). summary.json's `al_metrics`/`baseline_metrics` is only
    populated by the transformer pipeline, so reading state.pt covers all.
    """
    sp = os.path.join(run_dir, "state.pt")
    if not os.path.exists(sp):
        return None
    try:
        s = torch.load(sp, weights_only=False, map_location="cpu")
    except Exception:
        return None
    def _last(xs):
        if not xs:
            return float("nan")
        last = xs[-1]
        return float(last) if last is not None else float("nan")
    return {
        "al_static": _last(s.get("al_on_static_random_r2")),
        "base_static": _last(s.get("baseline_on_static_random_r2")),
        "al_mcmc": _last(s.get("al_on_mcmc_r2")),
        "base_mcmc": _last(s.get("baseline_on_mcmc_r2")),
        "al_cross": _last(s.get("al_on_base_val_r2")),
        "base_cross": _last(s.get("base_on_al_val_r2")),
    }


def report_table_G(per_arch, groups):
    """(G) Shared-eval sanity: AL R² > Baseline R² on shared eval sets."""
    print("\n" + "=" * 90)
    print("TABLE G — Shared-eval AL R² vs Baseline R² (mean across seeds)")
    print("=" * 90)
    print(f"{'arch':<14}{'pick':<28}"
          f"{'static_rand Δ':>22}{'MCMC Δ':>22}{'cross-val Δ':>22}")
    print("-" * 108)
    print(f"{'':<14}{'':<28}{'(AL > Baseline?)':>22}{'(AL > Baseline?)':>22}{'(AL > Baseline?)':>22}")
    print()
    for arch in REGULAR_MODELS:
        bp = best_pick(per_arch, arch, "hit_rate")
        if bp is None:
            continue
        (s, w), *_ = bp
        run_dirs = groups.get((arch, s, w), [])
        deltas = {"static": [], "mcmc": [], "cross": []}
        for d in run_dirs:
            r2s = _final_R2_shared(d)
            if not r2s:
                continue
            for k_short, (alk, bk) in (
                ("static", ("al_static", "base_static")),
                ("mcmc", ("al_mcmc", "base_mcmc")),
                ("cross", ("al_cross", "base_cross")),
            ):
                if r2s[alk] == r2s[alk] and r2s[bk] == r2s[bk]:
                    deltas[k_short].append(r2s[alk] - r2s[bk])
        def _fmt(xs):
            if not xs:
                return f"{'--':>22}"
            mean, sem, _ = _mean_sem(xs)
            sign = "AL>" if mean > 0 else "AL<"
            txt = f"{mean:+.3f} ± {sem:.3f} {sign}"
            return f"{txt:>22}"
        print(f"{arch:<14}{s+'/'+w:<28}"
              f"{_fmt(deltas['static'])}{_fmt(deltas['mcmc'])}{_fmt(deltas['cross'])}")
    print("\n  Interpretation: Δ = R²(AL) − R²(Baseline) on shared eval set.")
    print("  AL > Baseline means the AL pipeline beats random sampling at predicting Ωh².")
    print("  cross-val: AL evaluated on the random-baseline's val set (easy points).")


def report_table_E(per_arch, groups):
    """(E) Wall-clock cost table."""
    print("\n" + "=" * 90)
    print("TABLE E — Wall-clock cost for each architecture's best pick (median across seeds)")
    print("=" * 90)
    print(f"{'arch':<14}{'pick':<28}"
          f"{'iters':>7}{'per-iter (s)':>16}{'total (h)':>14}{'oracle calls/h':>20}")
    print("-" * 99)
    for arch in REGULAR_MODELS:
        bp = best_pick(per_arch, arch, "hit_rate")
        if bp is None:
            continue
        (s, w), *_ = bp
        run_dirs = groups.get((arch, s, w), [])
        stats = []
        for d in run_dirs:
            wc = _wallclock_from_log(d)
            if wc:
                stats.append(wc)
        if not stats:
            continue
        per_iter = statistics.median(w["per_iter_s"] for w in stats)
        total_h = statistics.median(w["total_s"] / 3600 for w in stats)
        oh = statistics.median(w["oracle_per_h"] for w in stats)
        n_iters = statistics.median(w["n_iters"] for w in stats)
        print(f"{arch:<14}{s+'/'+w:<28}"
              f"{int(n_iters):>7}{per_iter:>14.1f} s{total_h:>12.2f} h{oh:>18.0f}/h")
    print("\n  Interpretation: 'oracle calls/h' = n_select / per_iter × 3600.")
    print("  This is the rate at which the AL loop consumes the (expensive) SPheno oracle.")
    print("  Comparison fairness: all architectures use the same n_select=500 per iter,")
    print("  but per-iter wall-clock differs. Per-oracle-call cost = inverse of column 5.")


if __name__ == "__main__":
    groups = load_manifest_groups()
    per_arch = collect_per_arch(groups)
    print(f"Loaded manifest: {sum(len(v) for v in per_arch.values())} cells "
          f"across {sum(1 for v in per_arch.values() if v)} architectures.")
    for a in REGULAR_MODELS:
        if per_arch[a]:
            n_runs = sum(len(per_arch[a][sw]["hit_rate"])
                         for sw in per_arch[a])
            print(f"  {a:<14} {len(per_arch[a])} cells, {n_runs} total seeds")
    report_table_A_and_D(per_arch)
    report_table_B(per_arch)
    report_table_E(per_arch, groups)
    report_table_G(per_arch, groups)
