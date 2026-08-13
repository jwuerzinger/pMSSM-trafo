"""In-band hit rate against labelled-set size, and the gap between surrogates.

The per-iteration hit rate is what a trajectory plot against *iteration* hides:
runs at different budgets advance at wildly different rates, so comparing them
at equal iteration compares them at unequal |L|. Binning by |L| instead puts
every budget on one axis and asks the scaling question directly: given the same
amount of labelled data, which surrogate ranks candidates better?

The lower panel divides each model by the Transformer within the same source, so
the ratio is strategy- and budget-matched even though the three sources differ
in both.

Hit rate here is in-band models per *valid* model, computed from each run's own
state.pt: the fraction of the points acquired in an iteration that landed in the
band. It is not the per-attempt yield, which additionally folds in the fraction
of proposals the simulator rejects.

Usage:
    python scripts/plot_hitrate_vs_labelled.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plot_hit_rate_trajectories_multiseed import (  # noqa: E402
    MODEL_COLORS,
    MODEL_DISPLAY,
)

TRUE_VALUE, TOL = 0.12, 0.10
REF_MODEL = "transformer"

# source key -> (manifest, display tag, linestyle). Benchmark first so it draws
# under the probes.
SOURCES = [
    ("bench", "/ptmp/jwuerzin/analysis/all_runs/manifest_mainbody.csv",
     "40 it", "-"),
    ("ext", "/ptmp/jwuerzin/analysis/probe_extended/manifest.csv",
     "160 it", "--"),
    ("k20", "/ptmp/jwuerzin/analysis/probe_20k/manifest.csv",
     "20k batch", (0, (3, 1, 1, 1))),
]
# Only these cells, so each source contributes one configuration per model.
BENCH_CELLS = {
    ("transformer", "entropy_batch", "cold"), ("deep_gp", "entropy_batch", "warm"),
    ("dnn", "entropy_batch", "cold"), ("dnn_match_trafo", "entropy_batch", "cold"),
}
BENCH_SWEEP_PREFIX = "202608"


def _increments(run_dir):
    """Per-iteration (|L| after the step, n_new, n_inband_new) from state.pt."""
    st = torch.load(run_dir + "/state.pt", weights_only=False, map_location="cpu")
    Y = np.asarray(st["Y"], dtype=float).ravel()
    n = [int(x) for x in (st.get("al_n_train") or [])]
    if not n:
        return []
    inband = (np.abs(Y - TRUE_VALUE) / TRUE_VALUE) < TOL
    out, prev = [], 0
    for cur in n:
        cur = min(cur, len(Y))
        if cur > prev:
            out.append((cur, cur - prev, int(inband[prev:cur].sum())))
        prev = cur
    return out


def _collect_per_seed(manifest, cells, sweep_prefix):
    """{model: {seed: [(L, n_new, n_hit), ...]}} for the benchmark spread band."""
    per = defaultdict(dict)
    with open(manifest) as fh:
        for r in csv.DictReader(fh):
            key = (r["model"], r["strategy"], r["warm_start"])
            if key not in cells:
                continue
            if sweep_prefix and not r.get("sweep_id", "").startswith(sweep_prefix):
                continue
            try:
                per[r["model"]][r["seed"]] = _increments(r["expected_run_dir"])
            except Exception:                                   # noqa: BLE001
                pass
    return per


def _collect(manifest, cells=None, sweep_prefix=None):
    """{model: [(L, n_new, n_hit), ...]} pooled over that cell's seeds."""
    per = defaultdict(list)
    with open(manifest) as fh:
        for r in csv.DictReader(fh):
            key = (r["model"], r["strategy"], r["warm_start"])
            if cells and key not in cells:
                continue
            if sweep_prefix and not r.get("sweep_id", "").startswith(sweep_prefix):
                continue
            try:
                per[r["model"]] += _increments(r["expected_run_dir"])
            except Exception as exc:                            # noqa: BLE001
                click.echo(f"[hr-vs-L]   skip {r['expected_run_dir']}: {exc}", err=True)
    return per


def _binned(points, edges):
    """Pooled hit rate per |L| bin, with its binomial error.

    Pooling counts rather than averaging per-iteration rates is what makes the
    bins comparable: iterations contribute in proportion to how many points they
    actually acquired, which differs by a factor of 40 between the budgets.
    """
    L = np.array([p[0] for p in points], dtype=float)
    nn = np.array([p[1] for p in points], dtype=float)
    hh = np.array([p[2] for p in points], dtype=float)
    mid, rate, err = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (L >= a) & (L < b)
        tot = nn[m].sum()
        if tot < 200:                       # too few points to say anything
            continue
        p = hh[m].sum() / tot
        mid.append(np.sqrt(a * b))          # geometric centre, log axis
        rate.append(p)
        err.append(np.sqrt(max(p * (1 - p), 1e-12) / tot))
    return np.array(mid), np.array(rate), np.array(err)


@click.command()
@click.option("--output-dir", default="/ptmp/jwuerzin/analysis/all_runs",
              show_default=True)
@click.option("--nbins", default=14, show_default=True)
def main(output_dir, nbins):
    data = {}
    for key, man, tag, ls in SOURCES:
        cells = BENCH_CELLS if key == "bench" else None
        prefix = BENCH_SWEEP_PREFIX if key == "bench" else None
        per = _collect(man, cells=cells, sweep_prefix=prefix)
        for model, pts in per.items():
            if pts:
                data[(key, model)] = pts
                click.echo(f"[hr-vs-L] {key:<6}{model:<18}{len(pts):>5} iterations, "
                           f"|L| {min(p[0] for p in pts):>7} - {max(p[0] for p in pts):>7}")

    if not data:
        raise click.ClickException("no runs found")

    allL = np.concatenate([[p[0] for p in v] for v in data.values()])
    edges = np.geomspace(max(allL.min(), 500), allL.max() * 1.01, nbins + 1)

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(9.5, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.35], "hspace": 0.06})

    # The extended probe RESUMES the benchmark's seed-1 run: its first ~1.3e4
    # rows are the very same acquired points, so drawing them again under a
    # second label double-plots one dataset and invites reading the offset
    # between the two curves as a difference in behaviour, when it is only the
    # 5-seed mean against the one seed the probe continues. Draw the extended
    # curves from where the benchmark stops. (The 20k probe shares only the
    # 1600-point initial set, so it is independent and is drawn in full.)
    bench_max = {m: max(p[0] for p in data[(k, m)])
                 for (k, m) in data if k == "bench"}

    curves = {}
    for key, _man, tag, ls in SOURCES:
        for model in ("deep_gp", "transformer", "dnn", "dnn_match_trafo"):
            pts = data.get((key, model))
            if not pts:
                continue
            if key == "ext" and model in bench_max:
                pts = [p for p in pts if p[0] > bench_max[model]]
                if not pts:
                    continue
            mid, rate, err = _binned(pts, edges)
            if len(mid) < 2:
                continue
            curves[(key, model)] = (mid, rate, err)
            ax.errorbar(mid, rate, yerr=err, color=MODEL_COLORS.get(model),
                        ls=ls, lw=1.8, marker="o", ms=3.5, capsize=2)

    # Benchmark seed spread: the probes are single-replica, so the reader needs
    # to see how much of any offset at shared |L| is simply seed scatter.
    per_seed = _collect_per_seed(SOURCES[0][1], BENCH_CELLS, BENCH_SWEEP_PREFIX)
    for model, seeds in per_seed.items():
        band = []
        for pts in seeds.values():
            m, r_, _e = _binned(pts, edges)
            if len(m):
                band.append(dict(zip(m, r_)))
        if len(band) < 2:
            continue
        common = sorted(set.intersection(*[set(b) for b in band]))
        if len(common) < 2:
            continue
        lo = [min(b[x] for b in band) for x in common]
        hi = [max(b[x] for b in band) for x in common]
        ax.fill_between(common, lo, hi, color=MODEL_COLORS.get(model),
                        alpha=0.13, lw=0, zorder=0)

    # Ratio to the Transformer within the same source: strategy- and
    # budget-matched, which a ratio across sources would not be.
    for key, _man, tag, ls in SOURCES:
        ref = curves.get((key, REF_MODEL))
        if ref is None:
            continue
        rmid, rrate, rerr = ref
        for model in ("deep_gp", "dnn", "dnn_match_trafo"):
            cur = curves.get((key, model))
            if cur is None:
                continue
            mid, rate, err = cur
            common = np.intersect1d(mid, rmid)
            if len(common) < 2:
                continue
            i = np.searchsorted(mid, common)
            j = np.searchsorted(rmid, common)
            ratio = rate[i] / rrate[j]
            rel = ratio * np.sqrt((err[i] / rate[i])**2 + (rerr[j] / rrate[j])**2)
            axr.errorbar(common, ratio, yerr=rel, color=MODEL_COLORS.get(model),
                         ls=ls, lw=1.6, marker="o", ms=3.5, capsize=2)

    ax.set_xscale("log")
    ax.set_ylabel("in-band models per valid model")
    ax.grid(alpha=0.3)
    # Two small legends instead of one 12-entry grid: colour and linestyle are
    # independent here, so naming every combination repeats each model four
    # times and each budget three, and the block then covers the low-|L| data.
    from matplotlib.lines import Line2D                       # noqa: PLC0415
    models_present = [m for m in ("deep_gp", "transformer", "dnn", "dnn_match_trafo")
                      if any(k[1] == m for k in curves)]
    lg1 = ax.legend(
        handles=[Line2D([], [], color=MODEL_COLORS.get(m), lw=2.0,
                        label=MODEL_DISPLAY.get(m, m)) for m in models_present],
        fontsize=9, loc="upper left", frameon=True, title="surrogate")
    ax.add_artist(lg1)
    ax.legend(
        handles=[Line2D([], [], color="0.35", lw=1.8, ls=ls, label=tag)
                 for key, _m, tag, ls in SOURCES if any(k[0] == key for k in curves)],
        fontsize=9, loc="lower right", frameon=True, title="budget")
    axr.axhline(1.0, color="0.35", lw=1.0, ls=":")
    axr.set_xscale("log")
    axr.set_xlabel(r"labelled-set size $|L|$")
    axr.set_ylabel("ratio to\nTransformer", fontsize=9)
    axr.grid(alpha=0.3)

    out = Path(output_dir) / "hitrate_vs_labelled.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"[hr-vs-L] wrote {out}")


if __name__ == "__main__":
    main()
